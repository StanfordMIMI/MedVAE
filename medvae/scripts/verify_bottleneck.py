"""End-to-end check of AutoencoderKLBottleneck (frozen backbone + trainable
bottleneck) mimicking one training step of medvae_finetune.py.

Run on a GPU node:
    srun --partition=RAEAI --gres=gpu:l40:1 python medvae/scripts/verify_bottleneck.py

Loads the module files directly (registering package stubs in sys.modules) so it
does not trigger medvae/__init__ (which currently fails on a wandb/NumPy-2.0 clash).
"""
import sys, types, importlib.util, os, torch

ROOT = "/dataNAS/people/akkumar/Downloads/MedVAE"


def stub_pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []  # mark as package
    sys.modules[name] = m


def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


for p in ["medvae", "medvae.utils", "medvae.utils.vae", "medvae.models"]:
    stub_pkg(p)
load("medvae.utils.vae.diffusionmodels_3d", "medvae/utils/vae/diffusionmodels_3d.py")
load("medvae.utils.vae.distributions", "medvae/utils/vae/distributions.py")
load("medvae.models.autoencoder_kl_3d", "medvae/models/autoencoder_kl_3d.py")
bn = load("medvae.models.autoencoder_kl_3d_bottleneck",
          "medvae/models/autoencoder_kl_3d_bottleneck.py")

dev = "cuda"
ddconfig = dict(double_z=True, z_channels=1, resolution=512, in_channels=1, out_ch=1,
                ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
model = bn.AutoencoderKLBottleneck(ddconfig=ddconfig, embed_dim=1).to(dev)
PURE_BF16 = os.environ.get("PURE_BF16", "0") == "1"
if PURE_BF16:
    model = model.to(torch.bfloat16)
    print("model cast to bfloat16")

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable(bottleneck)={n_train/1e6:.1f}M   frozen(backbone)={n_frozen/1e6:.1f}M")

opt = torch.optim.Adam(model.bottleneck_parameters(), lr=1e-4)
model.train()
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()

x = torch.randn(1, 1, 224, 224, 160, device=dev)
LATENT_LOSS = os.environ.get("LATENT_LOSS", "0") == "1"
import contextlib
ctx = contextlib.nullcontext() if PURE_BF16 else torch.autocast(device_type="cuda", dtype=torch.bfloat16)
if PURE_BF16:
    x = x.to(torch.bfloat16)
with ctx:
    if LATENT_LOSS:
        # latent-space training: no decoder backprop
        with torch.no_grad():
            posterior = model.encode(x)
            z = posterior.mode()
        vec = model.encode_to_vec(z)
        z_hat = model.decode_from_vec(vec, z.shape)
        nll = torch.nn.functional.mse_loss(z_hat, z)
        kl = torch.tensor(0.0, device=dev)
        rec, latent = z_hat, z_hat
    else:
        rec, posterior, latent = model(x)
        nll = (rec - x).abs().mean()          # stand-in for the criterion's nll term
        kl = posterior.kl().sum() / x.shape[0]  # criterion calls posteriors.kl()
print(f"rec {tuple(rec.shape)}  latent {tuple(latent.shape)}  kl(const)={kl.item():.3f}")

# mimic criterion.calculate_adaptive_weight (only active after disc_start): grad
# of nll wrt the last layer. Off by default since it doubles backward memory.
if os.environ.get("ADAPTIVE_W", "0") == "1":
    last = model.get_last_layer()
    g = torch.autograd.grad(nll, last, retain_graph=True)[0]
    print(f"autograd.grad(nll, get_last_layer) OK, |grad|={g.norm().item():.3e}, "
          f"requires_grad={last.requires_grad}")

loss = nll + 1.0 * kl
opt.zero_grad(); loss.backward(); opt.step()

# gradient sanity: bottleneck got grads, backbone did not
bb = [p.grad is not None for n, p in model.named_parameters()
      if not p.requires_grad]
tb = [p.grad is not None for p in model.bottleneck_parameters()]
print(f"backbone params with grad: {sum(bb)} (want 0)   "
      f"bottleneck params with grad: {sum(tb)}/{len(tb)}")
print(f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("OK")
