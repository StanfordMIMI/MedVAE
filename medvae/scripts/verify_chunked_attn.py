"""Verify the chunked AttnBlock rewrite in diffusionmodels_3d.py.

Run on a GPU node, e.g.:
    srun --partition=RAEAI --gres=gpu:l40:1 python medvae/scripts/verify_chunked_attn.py

Checks:
  (1) the tiled (chunked) attention path is numerically identical to the dense path
  (2) the real full-size 3D encoder+decoder run end-to-end at (224,224,160) within L40 memory
"""
import importlib.util, os, torch, torch.nn as nn

base = "/dataNAS/people/akkumar/Downloads/MedVAE/medvae/utils/vae"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


dm = load("dm3d", os.path.join(base, "diffusionmodels_3d.py"))
dev = "cuda"

# (1) equivalence: dense (chunk huge) vs tiled (chunk small) on same block/weights
torch.manual_seed(0)
blk = dm.AttnBlock(512).to(dev).eval()
x = torch.randn(1, 512, 20, 20, 20, device=dev)  # n=8000 > 4096 so tiling engages
with torch.no_grad():
    dense = blk(x, chunk=10**9)  # force dense path
    tiled = blk(x, chunk=4096)   # default tiled path
    tiled2 = blk(x, chunk=777)   # odd chunk -> exercises ragged final block
print(f"[equiv] dense vs tiled(4096): max|diff|={(dense - tiled).abs().max().item():.2e}")
print(f"[equiv] dense vs tiled(777) : max|diff|={(dense - tiled2).abs().max().item():.2e}")

# (2) full-size real encoder+decoder at (224,224,160)
dd = dict(double_z=True, z_channels=1, resolution=512, in_channels=1, out_ch=1,
          ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
enc = dm.Encoder(**dd).to(dev).eval()
dec = dm.Decoder(**dd).to(dev).eval()
quant = nn.Conv3d(2, 2, 1).to(dev)
post = nn.Conv3d(1, 1, 1).to(dev)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
with torch.no_grad():
    xf = torch.randn(1, 1, 224, 224, 160, device=dev)
    h = enc(xf)
    z = quant(h)[:, :1]
    rec = dec(post(z))
print(f"[full] encode+decode OK: z {tuple(z.shape)}, recon {tuple(rec.shape)}, "
      f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
