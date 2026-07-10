"""Latent-space training of the fully-convolutional bottleneck (ConvBottleneck) on
CACHED latents (no CT loading, no backbone in the train loop). Fits one GPU, big batch.

The bottleneck learns z -> code -> z_hat with an L1 (default) or MSE reconstruction
loss on the frozen-encoder latent z (posterior.mode() = mean channel of the cached
moments). Reports R^2 (1 - SS_res/SS_tot) as the scale-free quality metric. Optional
variational mode (--kl_weight > 0) adds a KL penalty on the conv code (for sampling).

Run:
  srun --partition=RAEAI --gres=gpu:l40:1 --cpus-per-task=8 \
    /dataNAS/people/akkumar/Downloads/miniconda3/envs/medvae/bin/python \
    medvae/scripts/train_bottleneck_latent.py
"""
import os, csv, argparse, time, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from medvae.models.autoencoder_kl_3d_bottleneck import ConvBottleneck


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="medvae/data/latent_cache_8x")
    p.add_argument("--loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--sample_input", action="store_true",
                   help="draw z ~ N(mean, std) fresh each step (aug); default uses mode/mean")
    p.add_argument("--widths", default="64,128")
    p.add_argument("--bottleneck_ch", type=int, default=16)
    p.add_argument("--mlp", action="store_true",
                   help="add a single Linear(flat,flat)+SiLU global-mixing layer on the code")
    p.add_argument("--mlp_dim", type=int, default=0,
                   help="projecting MLP: flat->mlp_dim (global embedding)->flat")
    p.add_argument("--kl_weight", type=float, default=0.0, help=">0 -> variational code + KL")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0, help="weight decay (AdamW) - regularizes MLP overfit")
    p.add_argument("--resume", default="", help="load model weights from this .pt and continue")
    p.add_argument("--cosine", action="store_true", help="cosine-anneal LR over --epochs")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ckpt_dir", default="medvae/logs/bottleneck_latent_8x")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="medvae-latent")
    return p.parse_args()


class LatentDataset(Dataset):
    """Serves the frozen-encoder mean latent z=(1,D,H,W) from cached moments; ok rows."""
    def __init__(self, cache, split):
        self.mm = np.load(os.path.join(cache, f"{split}_moments.npy"), mmap_mode="r")
        self.rows = []
        with open(os.path.join(cache, f"{split}_index.csv")) as f:
            for r in csv.DictReader(f):
                if int(r["ok"]) == 1:
                    self.rows.append(int(r["row"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # return full moments (2,D,H,W): [mean, logvar]; z is derived on-GPU so we can
        # optionally sample fresh each step without a per-worker numpy-RNG pitfall.
        return torch.from_numpy(np.asarray(self.mm[self.rows[i]]))


def moments_to_z(m, sample):
    """m: (B,2,D,H,W) -> z (B,1,D,H,W). mode = mean (chan 0); sample = mean + std*eps."""
    mean = m[:, 0:1]
    if not sample:
        return mean
    logvar = torch.clamp(m[:, 1:2], -30.0, 20.0)
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


def batch_r2(z, zhat):
    ss_res = torch.sum((z - zhat) ** 2)
    ss_tot = torch.sum((z - z.mean()) ** 2)
    return (1.0 - ss_res / (ss_tot + 1e-8)).item()


def main():
    a = get_args()
    dev = "cuda"
    os.makedirs(a.ckpt_dir, exist_ok=True)
    widths = tuple(int(x) for x in a.widths.split(","))
    variational = a.kl_weight > 0
    use_wandb = not a.no_wandb

    model = ConvBottleneck(in_ch=1, widths=widths, bottleneck_ch=a.bottleneck_ch,
                           variational=variational, mlp=a.mlp, mlp_dim=a.mlp_dim).to(dev)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location="cpu"))
        print(f"resumed model weights from {a.resume}")
    n_tr = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ConvBottleneck widths={widths} bottleneck_ch={a.bottleneck_ch} "
          f"variational={variational}  params={n_tr:.2f}M  loss={a.loss}  cosine={a.cosine}")

    tr = DataLoader(LatentDataset(a.cache, "train"), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    va = DataLoader(LatentDataset(a.cache, "val"), batch_size=a.batch, shuffle=False,
                    num_workers=a.workers, pin_memory=True)
    print(f"train latents={len(tr.dataset)}  val latents={len(va.dataset)}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.01) \
        if a.cosine else None
    recon = (lambda x, y: F.l1_loss(x, y)) if a.loss == "l1" else (lambda x, y: F.mse_loss(x, y))

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(project=a.wandb_project,
                         name=f"convbn_8x_w{a.widths}_bch{a.bottleneck_ch}_{a.loss}"
                              + ("_mlp" if a.mlp else "") + (f"_mlpdim{a.mlp_dim}" if a.mlp_dim else "")
                              + (f"_wd{a.wd}" if a.wd else ""),
                         config=vars(a))
        print(f"=> wandb run: {run.url}")

    def log(d, step):
        if use_wandb:
            import wandb
            wandb.log(d, step=step)

    step = 0
    best = -math.inf
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        run_l, run_r2 = 0.0, 0.0
        for m in tr:
            m = m.to(dev, non_blocking=True)
            z = moments_to_z(m, a.sample_input)     # mode, or fresh sample each step
            zhat, kl = model(z)
            loss = recon(zhat, z)
            if variational:
                loss = loss + a.kl_weight * kl
            opt.zero_grad(); loss.backward(); opt.step()
            r2 = batch_r2(z, zhat)
            run_l += recon(zhat, z).item(); run_r2 += r2; step += 1
            if step % 20 == 0:
                d = {"train/recon": loss.item(), "train/r2": r2}
                if variational:
                    d["train/kl"] = kl.item()
                log(d, step)
        tr_l, tr_r2 = run_l / len(tr), run_r2 / len(tr)

        # ---- validation: proper R^2 accumulated over the whole val set ----
        model.eval()
        n = 0.0; s_res = 0.0; s_z = 0.0; s_z2 = 0.0; s_l1 = 0.0
        with torch.no_grad():
            for m in va:
                z = moments_to_z(m.to(dev), sample=False)   # val always on the mode
                zhat, _ = model(z)
                s_res += torch.sum((z - zhat) ** 2).item()
                s_z += torch.sum(z).item(); s_z2 += torch.sum(z ** 2).item()
                s_l1 += torch.sum(torch.abs(z - zhat)).item()
                n += z.numel()
        mean = s_z / n
        ss_tot = s_z2 - n * mean * mean
        val_r2 = 1.0 - s_res / (ss_tot + 1e-8)
        val_l1 = s_l1 / n
        dt = time.time() - t0
        print(f"ep {ep:03d}  train_{a.loss} {tr_l:.4f}  train_r2 {tr_r2:.4f}  "
              f"val_l1 {val_l1:.4f}  val_r2 {val_r2:.4f}  ({dt:.1f}s)", flush=True)
        log({"epoch": ep, "train/epoch_recon": tr_l, "train/epoch_r2": tr_r2,
             "val/l1": val_l1, "val/r2": val_r2}, step)

        if sched is not None:
            sched.step()
            log({"lr": opt.param_groups[0]["lr"]}, step)
        torch.save(model.state_dict(), os.path.join(a.ckpt_dir, "last.pt"))
        if val_r2 > best:
            best = val_r2
            torch.save(model.state_dict(), os.path.join(a.ckpt_dir, "best.pt"))
    print(f"DONE  best_val_r2 {best:.4f}")
    if use_wandb:
        run.finish()


if __name__ == "__main__":
    main()
