"""Demonstrate: whitening the bch=32 conv code turns a low-effective-dim (but high-R2)
code into a high-effective-dim embedding, LOSSLESSLY (invertible -> reconstruction/R2
unchanged). Whitening is fit on train codes, evaluated on val codes.
Run: srun --partition=RAEAI --gres=gpu:l40:1 --cpus-per-task=8 python medvae/scripts/whiten_demo.py
"""
import os, csv
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch
from medvae.models.autoencoder_kl_3d_bottleneck import ConvBottleneck

CKPT = os.environ.get("CKPT", "medvae/logs/bn8x_bch32_mse/best.pt")
CACHE = "medvae/data/latent_cache_8x"
dev = "cuda"

bn = ConvBottleneck(in_ch=1, widths=(128, 256), bottleneck_ch=32).to(dev).eval()
bn.load_state_dict(torch.load(CKPT, map_location="cpu"))


def codes(split, n):
    mm = np.load(os.path.join(CACHE, f"{split}_moments.npy"), mmap_mode="r")
    rows = [int(r["row"]) for r in csv.DictReader(open(os.path.join(CACHE, f"{split}_index.csv")))
            if int(r["ok"]) == 1][:n]
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            z = torch.from_numpy(np.asarray(mm[rows[i:i+64], 0:1])).to(dev)
            c, _ = bn.encode(z)
            out.append(c.flatten(1))
    return torch.cat(out)


def eigvals_cov(X):
    Xc = X - X.mean(0)
    s = torch.linalg.svdvals(Xc)
    return s ** 2  # proportional to covariance eigenvalues


def pr(ev):
    return ((ev.sum() ** 2) / (ev ** 2).sum()).item()


def kfor(ev, t):
    r = torch.sort(ev, descending=True).values
    c = torch.cumsum(r, 0) / r.sum()
    return int((c < t).sum().item() + 1)


Ctr = codes("train", 12000)   # > 7840 dims -> full-rank whitening (lossless)
Cva = codes("val", 5000)
D = Ctr.shape[1]
print(f"code dim D={D}   train fit N={Ctr.shape[0]}   val N={Cva.shape[0]}")

# --- raw effective dim (val) ---
ev_raw = eigvals_cov(Cva)
print(f"\nRAW conv code:")
print(f"  effective dim (PR): {pr(ev_raw):.1f}")
print(f"  comps for 90/95/99%: {kfor(ev_raw,.9)}/{kfor(ev_raw,.95)}/{kfor(ev_raw,.99)}")

# --- fit PCA whitening on train, apply to val ---
mu = Ctr.mean(0)
Xc = Ctr - mu
cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
lam, V = torch.linalg.eigh(cov)                 # ascending eigenvalues
lam = torch.clamp(lam, min=0)
eps = 1e-6 * lam.max()
whiten = lambda C: ((C - mu) @ V) / torch.sqrt(lam + eps)
unwhiten = lambda W: (W * torch.sqrt(lam + eps)) @ V.T + mu

Wva = whiten(Cva)
ev_w = eigvals_cov(Wva)
print(f"\nWHITENED embedding (same info, invertible transform):")
print(f"  effective dim (PR): {pr(ev_w):.1f}")
print(f"  comps for 90/95/99%: {kfor(ev_w,.9)}/{kfor(ev_w,.95)}/{kfor(ev_w,.99)}")

# --- losslessness: roundtrip and decoded z_hat identical ---
rt = unwhiten(whiten(Cva))
code_err = (rt - Cva).abs().max().item()
with torch.no_grad():
    sh = (Cva.shape[0], 32, 7, 7, 5)
    zhat_raw = bn.decode(Cva.reshape(sh))
    zhat_rt = bn.decode(rt.reshape(sh))
zhat_err = (zhat_raw - zhat_rt).abs().max().item()
print(f"\nLOSSLESS check:")
print(f"  code roundtrip max|err|:  {code_err:.2e}")
print(f"  decoded z_hat max|err|:   {zhat_err:.2e}   -> reconstruction/R2 UNCHANGED")
print("DONE")
