import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from medvae.models.autoencoder_kl_3d import AutoencoderKL


def _conv_block(cin, cout):
    # halves each spatial axis (stride 2, k=3, pad=1)
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, cout),
        nn.SiLU(),
    )


def _deconv_block(cin, cout, norm=True):
    # doubles each spatial axis (stride 2, k=3, pad=1, output_padding=1)
    layers = [
        nn.ConvTranspose3d(
            cin, cout, kernel_size=3, stride=2, padding=1, output_padding=1
        )
    ]
    if norm:
        layers += [nn.GroupNorm(32, cout), nn.SiLU()]
    return nn.Sequential(*layers)


class ConvBottleneck(nn.Module):
    """Fully-convolutional bottleneck over a VAE latent z (no MLP).

    Strided 3D convs compress z (in_ch, D, H, W) to a small spatial code
    (bottleneck_ch, D/4, H/4, W/4) -- e.g. the 8x latent (1,28,28,20) -> (16,7,7,5)
    ~= 3.9K values, ~4x compression -- and mirrored transposed convs reconstruct it.
    The code keeps coarse spatial structure (unlike a flattened global vector), which
    is far less prone to latent collapse.

    Deterministic by default. If ``variational`` is set, the encoder emits mean+logvar
    for the code, which is reparameterized and returns a KL term (for optional
    generative use); default off.
    """

    def __init__(self, in_ch=1, widths=(64, 128), bottleneck_ch=16, variational=False,
                 groups=8, mlp=False, mlp_dim=0, code_spatial=(7, 7, 5)):
        super().__init__()
        self.variational = variational
        self.mlp_dim = mlp_dim
        w0, w1 = widths

        def gn(c):
            return nn.GroupNorm(groups, c)

        # encoder: two stride-2 downsamples (/4), with a same-res conv per level
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, w0, 3, 2, 1), gn(w0), nn.SiLU(),
            nn.Conv3d(w0, w0, 3, 1, 1), gn(w0), nn.SiLU(),
            nn.Conv3d(w0, w1, 3, 2, 1), gn(w1), nn.SiLU(),
            nn.Conv3d(w1, w1, 3, 1, 1), gn(w1), nn.SiLU(),
        )
        self.to_code = nn.Conv3d(w1, bottleneck_ch * (2 if variational else 1), 3, 1, 1)

        # decoder: mirror back up to the latent resolution/channels
        self.from_code = nn.Conv3d(bottleneck_ch, w1, 3, 1, 1)
        self.dec = nn.Sequential(
            gn(w1), nn.SiLU(),
            nn.Conv3d(w1, w1, 3, 1, 1), gn(w1), nn.SiLU(),
            nn.ConvTranspose3d(w1, w0, 3, 2, 1, output_padding=1), gn(w0), nn.SiLU(),
            nn.Conv3d(w0, w0, 3, 1, 1), gn(w0), nn.SiLU(),
            nn.ConvTranspose3d(w0, in_ch, 3, 2, 1, output_padding=1),
        )
        self.bottleneck_ch = bottleneck_ch

        # Optional single global-mixing layer on the flattened code, SAME dim:
        # Linear(flat, flat) + SiLU, reshaped back to the spatial code. Every output
        # unit mixes all code positions (fully global) without changing the latent size.
        flat = bottleneck_ch * code_spatial[0] * code_spatial[1] * code_spatial[2]
        self.mlp = None       # same-dim single-linear global mixer: Linear(flat,flat)+SiLU
        if mlp:
            self.mlp = nn.Sequential(nn.Linear(flat, flat), nn.SiLU())
        # Projecting MLP: flat -> mlp_dim (the global embedding) -> flat. When
        # mlp_dim < flat this is a real information bottleneck at mlp_dim.
        self.mlp_proj = None
        if mlp_dim > 0:
            self.mlp_proj = nn.ModuleDict({
                "down": nn.Linear(flat, mlp_dim),
                "up": nn.Linear(mlp_dim, flat),
            })
            self.mlp_act = nn.SiLU()

    def encode(self, z):
        c = self.to_code(self.enc(z))
        if self.variational:
            mean, logvar = torch.chunk(c, 2, dim=1)
            return mean, torch.clamp(logvar, -30.0, 20.0)
        return c, None

    def decode(self, code):
        return self.dec(self.from_code(code))

    def forward(self, z):
        mean, logvar = self.encode(z)
        if self.variational:
            code = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
            kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        else:
            code, kl = mean, None
        if self.mlp is not None:
            code = self.mlp(code.flatten(1)).reshape(code.shape)
        if self.mlp_proj is not None:
            sh = code.shape
            v = self.mlp_act(self.mlp_proj["down"](code.flatten(1)))  # (B, mlp_dim) embedding
            code = self.mlp_proj["up"](v).reshape(sh)
        return self.decode(code), kl


class AutoencoderKLBottleneck(AutoencoderKL):
    """MedVAE 3D autoencoder with a frozen pretrained backbone and a trainable
    global bottleneck.

    The frozen pretrained encoder maps an image to a spatial latent ``z`` of
    shape ``(B, embed_dim, D, H, W)``. A trainable *bottleneck encoder* compresses
    that spatial latent to a single ``vec_dim`` vector (default 4096) via strided
    3D convs -> adaptive pool -> MLP, and a mirrored *bottleneck decoder* expands
    the vector back to the spatial-latent shape, which the frozen decoder
    reconstructs into an image. Only the two bottleneck heads are trained.

    The forward signature and return tuple match ``AutoencoderKL`` so the existing
    training loop / criterion in ``medvae_finetune.py`` work unchanged.
    """

    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        apply_channel_ds=True,
        vec_dim=4096,
        conv_channels=(32, 64, 128, 256, 512),
        pool_grid=(2, 2, 2),
        mlp_hidden=4096,
    ):
        # Build the backbone; weights are loaded later via init_from_ckpt so the
        # bottleneck modules (created below) are never touched by the checkpoint.
        super().__init__(
            ddconfig,
            embed_dim,
            ckpt_path=None,
            ignore_keys=ignore_keys,
            apply_channel_ds=apply_channel_ds,
        )

        # Freeze the entire pretrained backbone; only the bottleneck heads train.
        for p in self.parameters():
            p.requires_grad_(False)

        self.vec_dim = vec_dim
        self.pool_grid = tuple(pool_grid)
        self._seed_ch = conv_channels[-1]
        flat = conv_channels[-1] * self.pool_grid[0] * self.pool_grid[1] * self.pool_grid[2]
        self.flat_dim = flat

        # --- bottleneck encoder: (B, embed_dim, D, H, W) -> (B, vec_dim) ---
        chans = (embed_dim,) + tuple(conv_channels)
        self.bn_enc_conv = nn.Sequential(
            *[_conv_block(ci, co) for ci, co in zip(chans[:-1], chans[1:])]
        )
        # AdaptiveAvgPool makes the flattened size fixed regardless of input volume.
        self.bn_enc_pool = nn.AdaptiveAvgPool3d(self.pool_grid)
        self.bn_enc_mlp = nn.Sequential(
            nn.Linear(flat, mlp_hidden), nn.SiLU(), nn.Linear(mlp_hidden, vec_dim)
        )

        # --- bottleneck decoder: (B, vec_dim) -> (B, embed_dim, D, H, W) ---
        self.bn_dec_mlp = nn.Sequential(
            nn.Linear(vec_dim, mlp_hidden), nn.SiLU(), nn.Linear(mlp_hidden, flat)
        )
        rev = tuple(reversed(conv_channels))  # e.g. (256, 128, 64, 32)
        deconv = []
        in_c = rev[0]
        for out_c in rev[1:]:
            deconv.append(_deconv_block(in_c, out_c))
            in_c = out_c
        # final layer -> embed_dim channels, no norm/activation (raw latent)
        deconv.append(
            nn.ConvTranspose3d(
                in_c, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )
        self.bn_dec_deconv = nn.Sequential(*deconv)

        # When True, activations saved for backward through the frozen full-res
        # decoder are offloaded to (pinned) CPU memory. The decoder dominates the
        # backward peak at full volume (~40+ GB); offloading its checkpoint-boundary
        # tensors keeps a full (224,224,160) image-space step within one 48GB GPU,
        # at the cost of extra host<->device copies. Set via ``offload_decoder``.
        self.offload_decoder = False

        # Model-parallel device placement (set by ``to_model_parallel``). When
        # active, the full-res decoder level lives on ``_mp_dev1`` and everything
        # else on ``_mp_dev0``; the forward autocasts and shuttles tensors across.
        self._mp_dev0 = None
        self._mp_dev1 = None
        self._mp_autocast = True

    def to_model_parallel(self, base_dev="cuda:0", breakpoints=None, output_dev=None):
        """Shard the model across N GPUs so a full-volume image-space step fits.

        ``base_dev`` holds the encoder, bottleneck heads, criterion, and the early
        (low-resolution) decoder. ``breakpoints`` is a list of
        ``(i_level, i_block, device)`` tuples in decoder-forward order (levels
        descending, blocks ascending): from each breakpoint onward, decoder blocks
        run on ``device`` until the next breakpoint. This lets the heavy full-res
        level be spread over several cards. ``output_dev`` places the final
        norm/conv/tanh (defaults to the last breakpoint's device). The
        reconstruction is moved back to ``base_dev`` for the criterion.

        Example (3-way): base cuda:0, breakpoints [(0,0,'cuda:1'), (0,2,'cuda:2')]
        puts full-res block0..1 on cuda:1 and block2+outputs on cuda:2.
        """
        breakpoints = breakpoints or []
        self.to(base_dev)
        dec = self.decoder

        order = [
            (l, b)
            for l in reversed(range(dec.num_resolutions))
            for b in range(dec.num_res_blocks + 1)
        ]
        bp_map = {(l, b): d for (l, b, d) in breakpoints}

        dec.block_device = {}
        cur = base_dev
        level_last_dev = {}
        for (l, b) in order:
            if (l, b) in bp_map:
                cur = bp_map[(l, b)]
                dec.block_device[(l, b)] = cur
            dec.up[l].block[b].to(cur)
            for a in dec.up[l].attn:
                a.to(cur)
            # Blocks pushed off the base device are the memory-heavy full-res ones;
            # checkpoint their conv halves internally to shrink the backward peak.
            if cur != base_dev:
                dec.up[l].block[b].inner_checkpoint = True
            level_last_dev[l] = cur
        # each level's upsample runs right after its last block, on that device
        for l in range(dec.num_resolutions):
            if l != 0:
                dec.up[l].upsample.to(level_last_dev[l])

        out_dev = output_dev or cur
        dec.norm_out.to(out_dev)
        dec.conv_out.to(out_dev)
        dec.output_device = out_dev

        self._mp_dev0 = base_dev
        self._mp_out_dev = out_dev
        return self

    def to_model_parallel_auto(self, ngpu):
        """Apply a validated N-GPU schedule for the 4x 3D decoder (num_resolutions=3,
        num_res_blocks=2). The heavy full-res level-0 blocks are spread across cards
        so a full-volume image-space step fits (peak ~41GB/card on 3 GPUs)."""
        schedules = {
            3: ("cuda:0", [(0, 0, "cuda:1"), (0, 1, "cuda:2")], "cuda:2"),
            4: ("cuda:0", [(1, 2, "cuda:1"), (0, 1, "cuda:2"), (0, 2, "cuda:3")], "cuda:3"),
        }
        if ngpu not in schedules:
            raise ValueError(f"model_parallel_gpus must be 3 or 4, got {ngpu}")
        base, breakpoints, out_dev = schedules[ngpu]
        return self.to_model_parallel(base, breakpoints, output_dev=out_dev)

    def encode_to_vec(self, z):
        h = self.bn_enc_conv(z)
        h = self.bn_enc_pool(h)
        h = h.flatten(1)
        return self.bn_enc_mlp(h)

    def decode_from_vec(self, vec, latent_shape):
        b = vec.shape[0]
        h = self.bn_dec_mlp(vec).view(b, self._seed_ch, *self.pool_grid)
        h = self.bn_dec_deconv(h)
        # Upsampling by powers of two overshoots the (possibly non-power-of-two)
        # latent grid; resample to the exact shape the frozen decoder expects.
        if h.shape[2:] != tuple(latent_shape[2:]):
            h = F.interpolate(
                h, size=tuple(latent_shape[2:]), mode="trilinear", align_corners=False
            )
        return h

    def forward(self, input, sample_posterior=True, decode=True):
        mp = self._mp_dev0 is not None
        # In model-parallel mode the model is not wrapped by accelerate, so apply
        # autocast here; it covers CUDA ops on both devices within the context.
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if mp and self._mp_autocast
            else contextlib.nullcontext()
        )
        with autocast:
            if mp:
                input = input.to(self._mp_dev0)
            # Frozen encoder: no graph is stored, and z is detached from the backbone.
            with torch.no_grad():
                posterior = self.encode(input)
                z = posterior.sample() if sample_posterior else posterior.mode()
            z = z.detach()

            vec = self.encode_to_vec(z)              # trainable
            z_hat = self.decode_from_vec(vec, z.shape)  # trainable

            if not decode:
                return vec, posterior, z_hat

            # Frozen decoder: params are not updated, but gradient still flows
            # through it back to z_hat and the bottleneck heads. Optionally offload
            # its saved (checkpoint-boundary) activations to CPU to fit full volumes.
            offload = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if self.offload_decoder and z_hat.requires_grad and not mp
                else contextlib.nullcontext()
            )
            with offload:
                dec = self.decode(z_hat)
            if mp:
                # Return the reconstruction on dev0 where the criterion lives.
                dec = dec.to(self._mp_dev0)
            return dec, posterior, z_hat

    def get_last_layer(self):
        # Last trainable layer feeding the reconstruction, used by the criterion's
        # adaptive-weight computation (autograd.grad needs a param that requires grad).
        return self.bn_dec_deconv[-1].weight

    def bottleneck_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
