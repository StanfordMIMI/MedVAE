"""PCA on the 4096-dim bottleneck vector over test-set CT volumes, to measure the
latent's effective dimensionality (how many components carry the variance).

Encoding only (frozen encoder + bottleneck encoder) -> cheap, single GPU.
Run: srun --partition=RAEAI --gres=gpu:l40:1 python medvae/scripts/pca_latent.py
"""
import os, csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
N = int(os.environ.get("N", "384"))
OUT = "medvae/scripts/logs"
dev = "cuda"

ddconfig = dict(double_z=True, z_channels=1, resolution=512, in_channels=1, out_ch=1,
                ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
model = AutoencoderKLBottleneck(ddconfig=ddconfig, embed_dim=1)
model.load_state_dict(torch.load(PKL, map_location="cpu"), strict=False)
model = model.to(dev).eval()

ids = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["split"] == "test":
            p = os.path.join(DATA, r["image_uuid"] + ".nii.gz")
            if os.path.exists(p):
                ids.append(p)
            if len(ids) >= N:
                break
print(f"encoding {len(ids)} test volumes")

class DS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try: return load_ct_merlin(self.paths[i]), 1
        except Exception: return torch.zeros(1, 224, 224, 160), 0

dl = DataLoader(DS(ids), batch_size=1, num_workers=8, shuffle=False)
vecs = []
for k, (x, ok) in enumerate(dl):
    if int(ok) == 0:
        continue
    x = x.to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.encode(x).mode()
        v = model.encode_to_vec(z)
    vecs.append(v.float().cpu().numpy().reshape(-1))
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{len(ids)}")
X = np.stack(vecs)  # (N, 4096)
print(f"collected latent matrix: {X.shape}")

# --- PCA via SVD on centered data ---
mu = X.mean(0, keepdims=True)
Xc = X - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ev = (S ** 2) / (len(X) - 1)            # eigenvalues (variance per component)
ratio = ev / ev.sum()
cum = np.cumsum(ratio)

def k_for(th): return int(np.searchsorted(cum, th) + 1)
# participation ratio: effective # of dimensions
pr = (ev.sum() ** 2) / (np.sum(ev ** 2))

print("\n=== PCA of 4096-dim bottleneck latent (test set) ===")
print(f"samples={X.shape[0]}  ambient_dim={X.shape[1]}  max_rank={len(ev)}")
print(f"participation ratio (effective dim): {pr:.1f}")
for th in (0.50, 0.90, 0.95, 0.99):
    print(f"  components for {int(th*100)}% variance: {k_for(th)}")
print("top-10 explained-variance ratios:", np.round(ratio[:10], 4).tolist())
# how many components each carry a non-trivial share (>0.1% of variance)
print(f"components with >0.1% variance each: {(ratio > 0.001).sum()}")
print(f"components with >1%   variance each: {(ratio > 0.01).sum()}")

# --- scree / cumulative plot ---
SURF="#fcfcfb";INK="#0b0b0b";MUT="#898781";GRID="#e1e0d9";BLUE="#2a78d6";AQUA="#1baf7a"
plt.rcParams.update({"font.family":"sans-serif","figure.facecolor":SURF,"axes.facecolor":SURF})
fig, ax = plt.subplots(figsize=(9,5), dpi=130)
kk = np.arange(1, len(cum)+1)
ax.plot(kk, cum, color=BLUE, lw=2.2)
for th,lab in [(0.9,"90%"),(0.95,"95%"),(0.99,"99%")]:
    k=k_for(th); ax.axvline(k,color=MUT,ls=(0,(3,3)),lw=1)
    ax.annotate(f"{lab}: {k}",(k,th),color=INK,fontsize=9,xytext=(4,-12),textcoords="offset points")
ax.axhline(1.0,color=GRID,lw=1)
ax.set_title(f"Bottleneck latent PCA — cumulative explained variance  (eff. dim ~{pr:.0f})",
             color=INK,fontsize=12,fontweight="bold",loc="left")
ax.text(0,1.02,f"{X.shape[0]} test volumes · 4096-dim vector · step 19500",transform=ax.transAxes,color=MUT,fontsize=9)
ax.set_xlabel("number of principal components",color=MUT); ax.set_ylabel("cumulative variance",color=MUT)
ax.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors=MUT); ax.set_ylim(0,1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/latent_pca.png",facecolor=SURF,bbox_inches="tight")
print(f"saved {OUT}/latent_pca.png")
print("DONE")
