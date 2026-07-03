"""Measure per-GPU peak memory for ONE full image-space training step of the
bottleneck model split across 2 GPUs (model-parallel), with the real criterion.

Run on two L40s:
    srun --partition=RAEAI --gres=gpu:l40:2 \
        /dataNAS/people/akkumar/Downloads/miniconda3/envs/medvae/bin/python \
        medvae/scripts/probe_model_parallel.py

Env: DISC=1 to include the discriminator; SPLIT=<int> split level (default 0).
"""
import os
import time
import torch

from medvae.models.autoencoder_kl_3d_bottleneck import AutoencoderKLBottleneck
from medvae.losses.vae_losses import LPIPSWithDiscriminator

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DISC = os.environ.get("DISC", "0") == "1"
NGPU = int(os.environ.get("NGPU", "3"))
assert torch.cuda.device_count() >= NGPU, f"need {NGPU} GPUs, have {torch.cuda.device_count()}"

# Predefined balanced schedules for the full-res level of the 4x 3D decoder
# (num_resolutions=3, num_res_blocks=2 -> level 0 has blocks 0,1,2 at 224x224x160).
SCHEDULES = {
    2: ("cuda:0", [(0, 1, "cuda:1")], "cuda:1"),
    # block0 (heavy 256->128) alone on cuda:1; block1,2 + outputs on cuda:2
    3: ("cuda:0", [(0, 0, "cuda:1"), (0, 1, "cuda:2")], "cuda:2"),
    # relieve cuda:0 of the full-res upsample too; spread the 3 full-res blocks
    4: ("cuda:0", [(1, 2, "cuda:1"), (0, 1, "cuda:2"), (0, 2, "cuda:3")], "cuda:3"),
}
base_dev, breakpoints, out_dev = SCHEDULES[NGPU]

ddconfig = dict(double_z=True, z_channels=1, resolution=512, in_channels=1, out_ch=1,
                ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
model = AutoencoderKLBottleneck(ddconfig=ddconfig, embed_dim=1)
model.to_model_parallel(base_dev, breakpoints, output_dev=out_dev)
model.train()

disc_start = 0 if DISC else 10**9
criterion = LPIPSWithDiscriminator(
    disc_start=disc_start, kl_weight=1e-6, disc_weight=0.5,
    num_channels=1, checkpoint_slices=True, adaptive_disc_weight=False,
).to("cuda:0")

opt = torch.optim.Adam(model.bottleneck_parameters(), lr=1e-4)
n_train = sum(p.numel() for p in model.bottleneck_parameters()) / 1e6
print(f"trainable={n_train:.1f}M  DISC={DISC}  NGPU={NGPU}")

x = torch.randn(1, 1, 224, 224, 160, device="cuda:0")
for d in range(NGPU):
    torch.cuda.reset_peak_memory_stats(d)
torch.cuda.empty_cache()

if os.environ.get("FWD_ONLY", "0") == "1":
    with torch.no_grad():
        rec, posterior, z_hat = model(x)
    torch.cuda.synchronize()
    peaks = "  ".join(f"GPU{d} {torch.cuda.max_memory_allocated(d)/1e9:.1f}GB" for d in range(NGPU))
    print("FWD-ONLY PEAKS  " + peaks)
    import sys; sys.exit(0)

t0 = time.time()
try:
    rec, posterior, z_hat = model(x)  # autocast handled inside forward
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # accelerate does this
        loss, log = criterion(
            inputs=x, reconstructions=rec, latent=z_hat, posteriors=posterior,
            optimizer_idx=0, global_step=(disc_start if DISC else 0),
            weight_dtype=torch.bfloat16, last_layer=model.get_last_layer(), split="train",
        )
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    grads = sum(p.grad is not None for p in model.bottleneck_parameters())
    tot = len(model.bottleneck_parameters())
    print(f"loss={loss.item():.4f}  step_time={dt:.1f}s  bottleneck_grads={grads}/{tot}")
    peaks = "  ".join(f"GPU{d} {torch.cuda.max_memory_allocated(d)/1e9:.1f}GB" for d in range(NGPU))
    print("PEAKS  " + peaks)
    print("FITS")
except RuntimeError as e:
    print(f"OOM/ERROR after {time.time()-t0:.1f}s: {str(e)[:150]}")
    peaks = "  ".join(f"GPU{d} {torch.cuda.max_memory_allocated(d)/1e9:.1f}GB" for d in range(NGPU))
    print("PEAKS  " + peaks)
