"""Evaluate the trained ConvBottleneck (latent-space, 8x):
  (1) PCA of the 16x7x7x5 (=3920) bottleneck code over the val set -> effective dim.
  (2) Reconstruction: z -> bottleneck -> z_hat -> frozen 8x decoder, vs original CT
      and vs VAE-only decode(z).
Run: srun --partition=RAEAI --gres=gpu:l40:1 --cpus-per-task=8 python medvae/scripts/eval_conv_bottleneck.py
"""
import os, csv
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from medvae.models.autoencoder_kl_3d_bottleneck import ConvBottleneck
from medvae.utils.factory import create_model
from medvae.utils.loaders import load_ct_merlin

CKPT = os.environ.get("CKPT", "medvae/logs/bottleneck_latent_8x/best.pt")
WIDTHS = tuple(int(x) for x in os.environ.get("WIDTHS", "64,128").split(","))
BCH = int(os.environ.get("BCH", "16"))
MLP = os.environ.get("MLP", "0") == "1"
MLP_DIM = int(os.environ.get("MLP_DIM", "0"))
TAG = os.environ.get("TAG", "conv")
CACHE = "medvae/data/latent_cache_8x"
CSV = "medvae/data/ct_data.csv"; DATA = "medvae/data/ct_data"
OUT = os.path.join("medvae/scripts/logs/conv_bn_eval", TAG); os.makedirs(OUT, exist_ok=True)
N_PCA = int(os.environ.get("N_PCA", "1024"))
N_REC = int(os.environ.get("N_REC", "4"))
dev = "cuda"

bn = ConvBottleneck(in_ch=1, widths=WIDTHS, bottleneck_ch=BCH, mlp=MLP, mlp_dim=MLP_DIM).to(dev).eval()
bn.load_state_dict(torch.load(CKPT, map_location="cpu"))
print(f"[{TAG}] loaded {CKPT}  widths={WIDTHS} bch={BCH} mlp={MLP} mlp_dim={MLP_DIM}")


def embedding(z):
    """The representation of interest for PCA: the mlp global embedding if present,
    else the (post same-dim mixer) conv code, flattened to (B, D)."""
    c, _ = bn.encode(z)
    if bn.mlp is not None:
        return bn.mlp(c.flatten(1))
    if bn.mlp_proj is not None:
        return bn.mlp_act(bn.mlp_proj["down"](c.flatten(1)))   # the mlp_dim embedding
    return c.flatten(1)

# ---------- (1) PCA of the bottleneck code over val latents ----------
mm = np.load(os.path.join(CACHE, "val_moments.npy"), mmap_mode="r")
rows = [int(r["row"]) for r in csv.DictReader(open(os.path.join(CACHE, "val_index.csv"))) if int(r["ok"]) == 1]
rows = rows[:N_PCA]
codes = []
with torch.no_grad():
    for i in range(0, len(rows), 64):
        z = torch.from_numpy(np.asarray(mm[rows[i:i+64], 0:1])).to(dev)  # mean channel
        c = embedding(z)
        codes.append(c.reshape(c.shape[0], -1).cpu().numpy())
X = np.concatenate(codes); print("code matrix:", X.shape)
Xc = X - X.mean(0, keepdims=True)
S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
ev = S**2 / (len(X)-1); ratio = ev/ev.sum(); cum = np.cumsum(ratio)
pr = (ev.sum()**2)/np.sum(ev**2)
def kfor(t): return int(np.searchsorted(cum, t)+1)
print("\n=== PCA of 3920-dim conv bottleneck code (val) ===")
print(f"samples={X.shape[0]} dim={X.shape[1]} max_rank={len(ev)}")
print(f"participation ratio (effective dim): {pr:.1f}")
for t in (0.5,0.9,0.95,0.99): print(f"  {int(t*100)}% variance: {kfor(t)} comps")
print("components >1% var each:", int((ratio>0.01).sum()), " >0.1%:", int((ratio>0.001).sum()))

# ---------- (2) reconstruction through the frozen 8x decoder ----------
backbone = create_model("medvae_8_1_3d").to(dev).eval()
vids = []
for r in csv.DictReader(open(CSV)):
    if r["split"] == "val":
        p = os.path.join(DATA, r["image_uuid"]+".nii.gz")
        if os.path.exists(p): vids.append((r["image_uuid"], p))
        if len(vids) >= N_REC: break

def to01(a): return np.clip((a+1)/2, 0, 1)
def psnr(a,b):
    mse=float(np.mean((a-b)**2)); return (99.0 if mse==0 else 10*np.log10(1/mse))

print("\n=== reconstruction (val) ===")
rows_summary=[]
for i,(uid,path) in enumerate(vids):
    x = load_ct_merlin(path).unsqueeze(0).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = backbone.encode(x).mode()
        z_hat, _ = bn(z)
        vae = backbone.decode(z); rec = backbone.decode(z_hat)
    xi = to01(x[0,0].float().cpu().numpy()); ve = to01(vae[0,0].float().cpu().numpy()); re = to01(rec[0,0].float().cpu().numpy())
    p_v, p_b = psnr(xi,ve), psnr(xi,re); rows_summary.append((uid,p_v,p_b))
    print(f"[{i}] {uid[:22]}  VAE {p_v:.2f}dB   bottleneck {p_b:.2f}dB")
    D,H,W = xi.shape
    planes=[("axial",xi[D//2],ve[D//2],re[D//2]),("coronal",xi[:,H//2],ve[:,H//2],re[:,H//2]),("sagittal",xi[:,:,W//2],ve[:,:,W//2],re[:,:,W//2])]
    fig,ax=plt.subplots(3,3,figsize=(9,9)); cols=["original",f"VAE only ({p_v:.1f})",f"bottleneck ({p_b:.1f})"]
    for rr,(nm,a,b,c) in enumerate(planes):
        for cc,img in enumerate([a,b,c]):
            ax[rr,cc].imshow(np.rot90(img),cmap="gray",vmin=0,vmax=1); ax[rr,cc].set_xticks([]); ax[rr,cc].set_yticks([])
            if rr==0: ax[rr,cc].set_title(cols[cc],fontsize=12)
            if cc==0: ax[rr,cc].set_ylabel(nm,fontsize=12)
    fig.suptitle(f"8x conv-bottleneck (val) {uid[:24]}",fontsize=11); fig.tight_layout()
    fig.savefig(os.path.join(OUT,f"recon_{i}_{uid[:14]}.png"),dpi=110,bbox_inches="tight"); plt.close(fig)
mv=np.mean([r[1] for r in rows_summary]); mb=np.mean([r[2] for r in rows_summary])
print(f"MEAN  VAE {mv:.2f}dB   bottleneck {mb:.2f}dB")
print("figures ->", OUT)
print("DONE")
