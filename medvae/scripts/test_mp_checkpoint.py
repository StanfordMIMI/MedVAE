"""Verify accelerator.save_state/load_state round-trips the model-parallel
bottleneck model registered via register_for_checkpointing (mirrors medvae_finetune).
Run: srun --partition=RAEAI --gres=gpu:l40:3 python medvae/scripts/test_mp_checkpoint.py
"""
import os, tempfile, torch
from accelerate import Accelerator
from medvae.models.autoencoder_kl_3d_bottleneck import AutoencoderKLBottleneck

acc = Accelerator(mixed_precision="bf16")
dd = dict(double_z=True, z_channels=1, resolution=512, in_channels=1, out_ch=1,
          ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
model = AutoencoderKLBottleneck(ddconfig=dd, embed_dim=1)
model.to_model_parallel_auto(3)
acc.register_for_checkpointing(model)
opt = torch.optim.Adam(model.bottleneck_parameters(), lr=1e-4)
opt = acc.prepare(opt)

# perturb a bottleneck weight so we can detect a correct restore
w = model.bn_enc_mlp[0].weight
with torch.no_grad():
    w.add_(1.234)
before = w.detach().clone()

d = tempfile.mkdtemp()
ckpt = os.path.join(d, "step_test.pt")
try:
    acc.save_state(ckpt)
    print("save_state OK ->", ckpt)
    print("files:", os.listdir(ckpt))
except Exception as e:
    print("SAVE FAILED:", repr(e)); raise

# corrupt, then load, then compare
with torch.no_grad():
    w.zero_()
try:
    acc.load_state(ckpt)
    print("load_state OK")
except Exception as e:
    print("LOAD FAILED:", repr(e)); raise

restored = model.bn_enc_mlp[0].weight.detach()
maxdiff = (restored - before.to(restored.device)).abs().max().item()
print(f"weight round-trip max|diff| = {maxdiff:.2e}  ({'PASS' if maxdiff < 1e-5 else 'FAIL'})")
print("RESULT:", "CHECKPOINTING OK" if maxdiff < 1e-5 else "CHECKPOINTING BROKEN")
