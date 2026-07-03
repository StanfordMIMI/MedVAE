import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

__all__ = ["Encoder", "Decoder"]

# ----------------------ENCODER & DECODER DEFINITIONS------------------------


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [checkpoint(self.conv_in, x, use_reentrant=False)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = checkpoint(
                    self.down[i_level].block[i_block], hs[-1], temb, use_reentrant=False
                )
                if len(self.down[i_level].attn) > 0:
                    h = checkpoint(
                        self.down[i_level].attn[i_block], h, use_reeentrant=False
                    )
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(
                    checkpoint(
                        self.down[i_level].downsample, hs[-1], use_reentrant=False
                    )
                )

        # middle
        h = hs[-1]
        h = checkpoint(self.mid.block_1, h, temb, use_reentrant=False)
        h = checkpoint(self.mid.attn_1, h, use_reentrant=False)
        h = checkpoint(self.mid.block_2, h, temb, use_reentrant=False)

        # end
        h = checkpoint(self.norm_out, h, use_reentrant=False)
        h = nonlinearity(h)
        h = checkpoint(self.conv_out, h, use_reentrant=False)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        # in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv3d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = checkpoint(self.conv_in, z, use_reentrant=False)

        # middle
        h = checkpoint(self.mid.block_1, h, temb, use_reentrant=False)
        h = checkpoint(self.mid.attn_1, h, use_reentrant=False)
        h = checkpoint(self.mid.block_2, h, temb, use_reentrant=False)

        # Optional model-parallel split: levels with index < ``split_from_level``
        # (the highest-resolution, most memory-heavy levels) plus the output layers
        # live on ``split_device``. ``h`` is moved across at the boundary; the
        # transfer is autograd-aware so gradients flow back to the first device.
        # Optional model-parallel split. ``block_device`` maps (i_level, i_block)
        # -> device: at that point ``h`` (and subsequent modules) move to that
        # device. ``output_device`` places the final norm/conv/tanh. The moves are
        # autograd-aware so gradients flow back across GPUs. Default (unset) keeps
        # the whole decoder on one device -- behavior is unchanged.
        block_device = getattr(self, "block_device", None)
        output_device = getattr(self, "output_device", None)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if block_device is not None and (i_level, i_block) in block_device:
                    h = h.to(block_device[(i_level, i_block)])
                h = checkpoint(
                    self.up[i_level].block[i_block], h, temb, use_reentrant=False
                )
                if len(self.up[i_level].attn) > 0:
                    h = checkpoint(
                        self.up[i_level].attn[i_block], h, use_reentrant=False
                    )
            if i_level != 0:
                h = checkpoint(self.up[i_level].upsample, h, use_reentrant=False)

        # end
        if self.give_pre_end:
            return h

        if output_device is not None:
            h = h.to(output_device)
        h = checkpoint(self.norm_out, h, use_reentrant=False)
        h = nonlinearity(h)
        h = checkpoint(self.conv_out, h, use_reentrant=False)
        if self.tanh_out:
            h = checkpoint(torch.tanh, h, use_reentrant=False)
        return h


# ----------------------HELPER FUNCTIONS------------------------


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


def nonlinearity(x):
    # Mathematically identical to x * sigmoid(x) (SiLU/Swish), but the fused
    # kernel has a memory-efficient backward that recomputes from the input
    # instead of storing sigmoid(x) and the product as separate full-res tensors.
    # At (224,224,160) full resolution this saves many GB per activation.
    return torch.nn.functional.silu(x)


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class LinAttnBlock(LinearAttention):
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, chunk=4096):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, d, h, w = q.shape
        n = d * h * w
        q = q.reshape(b, c, n)
        q = q.permute(0, 2, 1)  # b,n,c
        k = k.reshape(b, c, n)  # b,c,n
        v = v.reshape(b, c, n)  # b,c,n
        scale = int(c) ** (-0.5)

        if n <= chunk:
            # dense path (unchanged): materializes the full (n, n) matrix
            w_ = torch.bmm(q, k)  # b,n,n    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_ = w_ * scale
            w_ = torch.nn.functional.softmax(w_, dim=2)
            w_ = w_.permute(0, 2, 1)  # b,n,n (first n of k, second of q)
            h_ = torch.bmm(v, w_)  # b,c,n   h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        else:
            # tiled over queries: process `chunk` query positions at a time so the
            # full (n, n) attention matrix is never materialized. For 3D volumes n
            # can be very large (n = d*h*w), and an (n, n) tensor is prohibitive.
            # Numerically identical to the dense path above (each query row is an
            # independent softmax over all keys).
            def _attn_chunk(qs, k, v):
                s = torch.bmm(qs, k) * scale  # b,m,n
                s = torch.nn.functional.softmax(s, dim=2)
                # b,c,m   out[b,c,iq] = sum_j v[b,c,j] s[b,iq,j]
                return torch.bmm(v, s.permute(0, 2, 1))

            outs = []
            for i in range(0, n, chunk):
                qs = q[:, i : i + chunk]  # b,m,c
                # Checkpoint each chunk so its (m, n) score matrix is recomputed in
                # backward instead of all chunks being held at once (which would
                # rebuild the full (n, n) graph and defeat the tiling in backward).
                if self.training and qs.requires_grad:
                    outs.append(
                        checkpoint(_attn_chunk, qs, k, v, use_reentrant=False)
                    )
                else:
                    outs.append(_attn_chunk(qs, k, v))
            h_ = torch.cat(outs, dim=2)  # b,c,n

        h_ = h_.reshape(b, c, d, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def _part1(self, h):
        return self.conv1(nonlinearity(self.norm1(h)))

    def _part2(self, h):
        return self.conv2(self.dropout(nonlinearity(self.norm2(h))))

    def forward(self, x, temb):
        # Optionally checkpoint the two conv halves separately. For the heaviest
        # full-res blocks this roughly halves the recompute peak (only one half's
        # activations are materialized at a time in backward). temb must be None.
        if getattr(self, "inner_checkpoint", False) and self.training and x.requires_grad:
            assert temb is None, "inner_checkpoint path does not support temb"
            h = checkpoint(self._part1, x, use_reentrant=False)
            h = checkpoint(self._part2, h, use_reentrant=False)
        else:
            h = self.norm1(x)
            h = nonlinearity(h)
            h = self.conv1(h)

            if temb is not None:
                h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

            h = self.norm2(h)
            h = nonlinearity(h)
            h = self.dropout(h)
            h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
