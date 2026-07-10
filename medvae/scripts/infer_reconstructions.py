"""Reconstruct a few TEST-set CT volumes with the trained global-bottleneck model
and save input vs VAE-only vs bottleneck-reconstruction comparison figures.

Single-GPU, forward-only (no_grad + bf16) -> fits one L40.
Run: srun --partition=RAEAI --gres=gpu:l40:1 python medvae/scripts/infer_reconstructions.py
"""
import os, csv, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medvae.models.autoencoder_kl_3d_bottleneck import AutoencoderKLBottleneck
from medvae.utils.loaders import load_ct_merlin

CKPT = os.environ.get("CKPT",
    "medvae/logs/mri_ct_bottleneck_finetuning/runs/2026-07-04_19-03-04/checkpoints/step_19500.pt")
PKL = os.path.join(CKPT, "custom_checkpoint_0.pkl")
CSV = "medvae/data/ct_data.csv"
DATA = "medvae/data/ct_data"
OUT = "medvae/scripts/logs/recon_examples"
N = int(os.environ.get("N", "4"))
os.makedirs(OUT, exist_ok=True)
dev = "cuda"

# --- build model + load trained weights (pkl is the full model state_dict) ---
ddconfig = dict(double_z=True, z_channels=1, resolution=512, in_channels=1, out_ch=1,
                ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
model = AutoencoderKLBottleneck(ddconfig=ddconfig, embed_dim=1)
sd = torch.load(PKL, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("encoder") for k in sd):
    sd = sd["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded {PKL}\n  missing={len(missing)} unexpected={len(unexpected)}")
model = model.to(dev).eval()

# --- pick test-split volumes that exist on disk ---
test_ids = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["split"] == "test":
            p = os.path.join(DATA, r["image_uuid"] + ".nii.gz")
            if os.path.exists(p):
                test_ids.append((r["image_uuid"], p))
            if len(test_ids) >= N:
                break
print(f"using {len(test_ids)} test volumes")

def to01(x):  # [-1,1] -> [0,1]
    return np.clip((x + 1.0) / 2.0, 0, 1)

def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse), mse

rows = []
for i, (uid, path) in enumerate(test_ids):
    x = load_ct_merlin(path)                      # (1,224,224,160), [-1,1]
    xb = x.unsqueeze(0).to(dev)                    # (1,1,224,224,160)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        post = model.encode(xb); z = post.mode()
        vae_rec = model.decode(z)                              # backbone only
        z_hat = model.decode_from_vec(model.encode_to_vec(z), z.shape)
        bn_rec = model.decode(z_hat)                           # through bottleneck
    xin = to01(x[0].float().numpy())
    vae = to01(vae_rec[0, 0].float().cpu().numpy())
    bn  = to01(bn_rec[0, 0].float().cpu().numpy())
    p_vae, m_vae = psnr(xin, vae); p_bn, m_bn = psnr(xin, bn)
    rows.append((uid, p_vae, p_bn))
    print(f"[{i}] {uid[:24]}...  VAE PSNR={p_vae:.2f}dB  bottleneck PSNR={p_bn:.2f}dB")

    # mid slices along each axis
    D, H, W = xin.shape
    planes = [("axial", xin[D//2], vae[D//2], bn[D//2]),
              ("coronal", xin[:, H//2], vae[:, H//2], bn[:, H//2]),
              ("sagittal", xin[:, :, W//2], vae[:, :, W//2], bn[:, :, W//2])]
    fig, ax = plt.subplots(3, 3, figsize=(9, 9))
    cols = ["input", f"VAE only (PSNR {p_vae:.1f})", f"bottleneck (PSNR {p_bn:.1f})"]
    for r, (name, a, b, c) in enumerate(planes):
        for cc, img in enumerate([a, b, c]):
            im = np.rot90(img)
            ax[r, cc].imshow(im, cmap="gray", vmin=0, vmax=1)
            ax[r, cc].set_xticks([]); ax[r, cc].set_yticks([])
            if r == 0: ax[r, cc].set_title(cols[cc], fontsize=12)
            if cc == 0: ax[r, cc].set_ylabel(name, fontsize=12)
    fig.suptitle(f"test vol {uid[:28]}  (step 19500)", fontsize=11)
    fig.tight_layout()
    fp = os.path.join(OUT, f"recon_{i}_{uid[:16]}.png")
    fig.savefig(fp, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("  saved", fp)

print("\n=== summary (PSNR dB) ===")
print(f"{'volume':28s}  VAE-only  bottleneck")
for uid, pv, pb in rows:
    print(f"{uid[:28]:28s}  {pv:7.2f}  {pb:9.2f}")
mv=np.mean([r[1] for r in rows]); mb=np.mean([r[2] for r in rows])
print(f"{'MEAN':28s}  {mv:7.2f}  {mb:9.2f}")
print("DONE")
