"""Cache frozen-encoder latents for all splits so the bottleneck can be trained on
latents directly (no CT loading, no encoder forward at train time).

For each volume we run the SAME transform (load_ct_merlin) + frozen backbone encoder
(model.encode) and store the posterior parameters = quant_conv output = [mean, logvar]
concatenated on the channel dim. embed_dim=1, so this is 2 channels: [..., 0]=mean,
[..., 1]=logvar. At train time rebuild DiagonalGaussianDistribution(params) and take
.mode() (mean, deterministic target) or .sample() (fresh draw).

Saved per volume as a 4D .nii.gz with layout (d, h, w, c) = (56, 56, 40, 2), and/or a
per-split memmap of shape (N, 2, 56, 56, 40).

Computed and stored in fp32 (the encoder's native precision) -- these latents are the
fixed ground-truth target for bottleneck training, so we do NOT use bf16 autocast here
(that would bake ~8-bit rounding into the target). It's a one-time forward pass.

Per-volume latent for the 4x model: (2, 56, 56, 40) fp32  (~0.96 MB).
Run:
  srun --partition=RAEAI --gres=gpu:l40:1 --cpus-per-task=12 \
    /dataNAS/people/akkumar/Downloads/miniconda3/envs/medvae/bin/python \
    medvae/scripts/cache_latents.py
"""
import os, csv, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

from medvae.utils.factory import create_model
from medvae.utils.loaders import load_ct_merlin

CSV = "medvae/data/ct_data.csv"
DATA = "medvae/data/ct_data"
MODEL = os.environ.get("MODEL", "medvae_4_1_3d")   # or medvae_8_1_3d
OUT = os.environ.get("OUT", "medvae/data/latent_cache")
SPLITS = os.environ.get("SPLITS", "train,val,test").split(",")
BATCH = int(os.environ.get("BATCH", "1"))  # fp32 full-res encode is memory-heavy
WORKERS = int(os.environ.get("WORKERS", "12"))
LIMIT = int(os.environ.get("LIMIT", "0"))  # >0: cap volumes per split (smoke test)
SAVE_NPY = os.environ.get("SAVE_NPY", "1") == "1"  # one memmap array per split
SAVE_NII = os.environ.get("SAVE_NII", "1") == "1"  # one .nii.gz per volume (uuid-named)
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

# Frozen base backbone (identical weights to the bottleneck's frozen backbone).
print(f"model={MODEL}  out={OUT}")
model = create_model(MODEL).to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)


class CTDataset(Dataset):
    """Returns (moments-input volume, ok flag). ok=0 marks a failed/zero load."""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            x = load_ct_merlin(self.paths[i])          # (1,224,224,160), [-1,1]
            ok = 0 if bool(torch.all(x == 0)) else 1   # load_ct_merlin -> zeros on failure
            return x, ok
        except Exception:
            return torch.zeros(1, 224, 224, 160), 0


def rows_for_split(split):
    ids = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                p = os.path.join(DATA, r["image_uuid"] + ".nii.gz")
                if os.path.exists(p):
                    ids.append((r["image_uuid"], p))
    return ids


def cache_split(split):
    done_marker = os.path.join(OUT, f"{split}.done")
    if os.path.exists(done_marker):
        print(f"[{split}] already cached (found {done_marker}), skipping")
        return
    ids = rows_for_split(split)
    if LIMIT > 0:
        ids = ids[:LIMIT]
    n = len(ids)
    print(f"[{split}] {n} volumes" + (f" (LIMIT={LIMIT})" if LIMIT else ""))
    if n == 0:
        return

    # infer latent shape from the first encodable volume
    nii_dir = os.path.join(OUT, split)
    if SAVE_NII:
        os.makedirs(nii_dir, exist_ok=True)

    dl = DataLoader(CTDataset([p for _, p in ids]), batch_size=BATCH,
                    num_workers=WORKERS, shuffle=False)
    mm = None
    c = d = h = w = None
    idx_rows = []
    written = 0
    n_bad = 0
    for bi, (xb, okb) in enumerate(dl):
        xb = xb.to(dev)
        with torch.no_grad():                    # no autocast: native model precision
            post = model.encode(xb)
            moments = post.parameters            # (B, 2*embed_dim, 56,56,40)
        moments = moments.cpu().numpy()          # keep whatever dtype the model output
        if c is None:
            c, d, h, w = moments.shape[1:]
            print(f"[{split}] latent shape per vol = {(c,d,h,w)}  dtype={moments.dtype}")
            if SAVE_NPY:
                mm = np.lib.format.open_memmap(
                    os.path.join(OUT, f"{split}_moments.npy"), mode="w+",
                    dtype=moments.dtype, shape=(n, c, d, h, w))
                print(f"[{split}] memmap {(n,c,d,h,w)} (~{n*c*d*h*w*moments.itemsize/1e9:.2f} GB)")
        for j in range(moments.shape[0]):
            row = written
            uid = ids[row][0]
            ok = int(okb[j])
            if SAVE_NPY:
                mm[row] = moments[j]
            if SAVE_NII:
                # NIfTI wants spatial dims first, channels last: (d,h,w,C)
                vol = np.moveaxis(moments[j], 0, -1)
                nib.save(nib.Nifti1Image(vol, np.eye(4)),
                         os.path.join(nii_dir, f"{uid}.nii.gz"))
            n_bad += (ok == 0)
            idx_rows.append((row, uid, ok))
            written += 1
        if (bi + 1) % 50 == 0:
            print(f"[{split}] {written}/{n}  (bad so far: {n_bad})")
    if mm is not None:
        mm.flush()
    with open(os.path.join(OUT, f"{split}_index.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["row", "image_uuid", "ok"]); wr.writerows(idx_rows)
    open(done_marker, "w").close()
    print(f"[{split}] DONE  wrote {written}  (bad/zero loads: {n_bad})")
    return (c, d, h, w)


shape = None
for s in SPLITS:
    r = cache_split(s)
    shape = r or shape
# record the actual stored dtype (read back from a written array)
dtype = None
for s in SPLITS:
    fp = os.path.join(OUT, f"{s}_moments.npy")
    if os.path.exists(fp):
        dtype = str(np.load(fp, mmap_mode="r").dtype); break
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump({"latent_shape": shape, "dtype": dtype,
               "note": "posterior moments (mean+logvar); rebuild via DiagonalGaussianDistribution"}, f, indent=2)
print("ALL DONE ->", OUT)
