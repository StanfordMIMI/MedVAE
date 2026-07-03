"""Measure peak GPU memory for ONE image-space training step of the bottleneck
model with the REAL LPIPSWithDiscriminator criterion, at full (224,224,160).

Run on one L40:
    srun --partition=RAEAI --gres=gpu:l40:1 \
        /dataNAS/people/akkumar/Downloads/miniconda3/envs/medvae/bin/python \
        medvae/scripts/probe_imagespace_mem.py

Env flags:
    CKPT_SLICES=1   checkpoint each per-slice LPIPS/disc forward (recompute in bwd)
    SAVE_ON_CPU=1   offload saved activations to CPU (torch.autograd.graph.save_on_cpu)
    DISC=1          run with the discriminator active (generator adversarial path)
"""
import os
import contextlib
import torch

from medvae.models.autoencoder_kl_3d_bottleneck import AutoencoderKLBottleneck
from medvae.losses.vae_losses import LPIPSWithDiscriminator

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
dev = "cuda"
CKPT_SLICES = os.environ.get("CKPT_SLICES", "0") == "1"
SAVE_ON_CPU = os.environ.get("SAVE_ON_CPU", "0") == "1"
DISC = os.environ.get("DISC", "0") == "1"

ddconfig = dict(double_z=True, z_channels=1, resolution=512, in_channels=1, out_ch=1,
                ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
model = AutoencoderKLBottleneck(ddconfig=ddconfig, embed_dim=1).to(dev)
model.offload_decoder = os.environ.get("OFFLOAD_DECODER", "0") == "1"
model.train()

disc_start = 0 if DISC else 10**9
criterion = LPIPSWithDiscriminator(
    disc_start=disc_start, kl_weight=1e-6, disc_weight=0.5,
    num_channels=1, checkpoint_slices=CKPT_SLICES,
).to(dev)

opt = torch.optim.Adam(model.bottleneck_parameters(), lr=1e-4)
print(f"CKPT_SLICES={CKPT_SLICES}  SAVE_ON_CPU={SAVE_ON_CPU}  "
      f"OFFLOAD_DECODER={model.offload_decoder}  DISC={DISC}")

x = torch.randn(1, 1, 224, 224, 160, device=dev)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if SAVE_ON_CPU else contextlib.nullcontext()

import time
t0 = time.time()
try:
    with autocast, offload:
        rec, posterior, latent = model(x)
        loss, log = criterion(
            inputs=x, reconstructions=rec, latent=latent, posteriors=posterior,
            optimizer_idx=0, global_step=(disc_start if DISC else 0),
            weight_dtype=torch.bfloat16,
            last_layer=model.get_last_layer(), split="train",
        )
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"loss={loss.item():.4f}  step_time={dt:.1f}s")
    print(f"PEAK {torch.cuda.max_memory_allocated()/1e9:.1f} GB  "
          f"(reserved {torch.cuda.max_memory_reserved()/1e9:.1f} GB)")
    print("FITS")
except RuntimeError as e:
    print(f"OOM/ERROR after {time.time()-t0:.1f}s: {str(e)[:200]}")
    print(f"PEAK-at-fail {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
