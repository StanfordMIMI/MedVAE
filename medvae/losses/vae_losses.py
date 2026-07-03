import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize
import open_clip

from medvae.utils.vae.loss_components import (
    LPIPS,
    NLayerDiscriminator,
    hinge_loss,
    weights_init,
)


__all__ = ["LPIPSWithDiscriminator", "BiomedClipLoss"]

""" 
Take a 2D input array and average out the loss across all the slices to get a 3D loss
@input: net: The network to calculate the loss
@input: inp_arr: The input array
@output: The average loss across all the slices
"""


def discriminator_2d_nets_to_3d(net, inp_arr, checkpoint=False):
    p_loss_arr = []
    dims = ["depth", "height", "width"]

    for dim_idx, dim_name in enumerate(dims, start=2):
        # Iterate over slices along the current dimension
        for j in range(inp_arr.size(dim_idx)):
            # Select the appropriate slice along each dimension
            if dim_name == "depth":
                slice_i = inp_arr[:, :, j, :, :]
            elif dim_name == "height":
                slice_i = inp_arr[:, :, :, j, :]
            else:  # width
                slice_i = inp_arr[:, :, :, :, j]

            # Calculate discriminator logits for the current slice
            if checkpoint and slice_i.requires_grad:
                logits = torch.utils.checkpoint.checkpoint(
                    net, slice_i.contiguous(), use_reentrant=False
                )
            else:
                logits = net(slice_i.contiguous())
            # Reduce each slice's patch map to a scalar so slices taken along axes
            # of different length (non-cubic volumes) can be stacked.
            p_loss_arr.append(logits.mean())

    # Average across all slices
    return torch.mean(torch.stack(p_loss_arr), 0)


class BiomedClipLoss(nn.Module):
    def __init__(self, compute_rec_loss, compute_lat_loss):
        super().__init__()

        self.clip, _, _ = open_clip.create_model_and_transforms(
            # pretrained="/admin/home-mayavarma/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blobs/8792dba76fc3a96544a87bb0f76c82167b4ba509d57c08b98b9c9266f764598b",
            model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )

        for _, param in enumerate(self.clip.parameters()):
            param.requires_grad_(False)
        self.clip.eval()
        self.transform = Compose(
            [
                Resize(size=224, interpolation=3, max_size=None, antialias=True),
                CenterCrop(size=(224, 224)),
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.compute_rec_loss = compute_rec_loss
        self.compute_lat_loss = compute_lat_loss

        # self.transform_img = Compose(
        #     [
        #         Resize(size=48, interpolation=3, max_size=None, antialias=True),
        #         Pad(88),
        #         Normalize(
        #             mean=[0.48145466, 0.4578275, 0.40821073],
        #             std=[0.26862954, 0.26130258, 0.27577711],
        #         ),
        #     ]
        # )
        # self.transform_latent = Compose(
        #     [
        #         Pad(88),
        #         Normalize(
        #             mean=[0.48145466, 0.4578275, 0.40821073],
        #             std=[0.26862954, 0.26130258, 0.27577711],
        #         ),
        #     ]
        # )

    def forward(self, img, rec=None, latent=None):
        img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)

        if img.shape[1] == 1 and len(img.shape) == 4:
            img = img.expand(-1, 3, -1, -1)
        if img.shape[1] == 1 and len(img.shape) == 5:
            img = img.expand(-1, 3, -1, -1, -1)

        if len(img.shape) == 5:
            feature_arr = []

            for dim_idx, dim_name in enumerate(["depth", "height", "width"], start=2):
                # Iterate over slices along the current dimension
                for j in range(img.size(dim_idx)):
                    # Select the appropriate slice along each dimension
                    if dim_name == "depth":
                        slice_i = img[:, :, j, :, :]
                    elif dim_name == "height":
                        slice_i = img[:, :, :, j, :]
                    else:  # width
                        slice_i = img[:, :, :, :, j]

                    img_slice = self.transform(slice_i)

                    feature_arr.append(self.clip.encode_image(img_slice))

            # bc_loss = torch.zeros(img.shape[0]).cuda()
            if self.compute_lat_loss:
                latent = latent / 4.6
                latent = latent.mean(1, keepdim=True)

                latent_arr = []

                for dim_idx, dim_name in enumerate(
                    ["depth", "height", "width"], start=2
                ):
                    # Iterate over slices along the current dimension
                    for j in range(latent.size(dim_idx)):
                        # Select the appropriate slice along each dimension
                        if dim_name == "depth":
                            slice_i = latent[:, :, j, :, :]
                        elif dim_name == "height":
                            slice_i = latent[:, :, :, j, :]
                        else:  # width
                            slice_i = latent[:, :, :, :, j]

                        latent_slice = self.transform(slice_i.expand(-1, 3, -1, -1))
                        latent_arr.append(self.clip.encode_image(latent_slice))

                # Take the average of all the slices
                feature_arr = torch.stack(feature_arr).permute(1, 2, 0)

                latent_arr = torch.stack(latent_arr)

                feature_arr_interpolate = torch.nn.functional.interpolate(
                    feature_arr,
                    size=latent_arr.shape[0],
                    mode="linear",
                    align_corners=False,
                ).permute(2, 0, 1)

                # Do the element wise difference  between feature_arr and latent_arr
                img_rec_loss_arr = []
                for i in range(feature_arr_interpolate.shape[0]):
                    img_rec_loss_arr.append(
                        (
                            (feature_arr_interpolate[i, ...] - latent_arr[i, ...]) ** 2
                        ).sum(1)
                    )

                return torch.stack(img_rec_loss_arr).mean(0)

            if self.compute_rec_loss:
                rec = torch.clamp((rec + 1.0) / 2.0, min=0.0, max=1.0)
                if rec.shape[1] == 1:
                    rec = rec.expand(-1, 3, -1, -1, -1)

                rec_arr = []

                for dim_idx, dim_name in enumerate(
                    ["depth", "height", "width"], start=2
                ):
                    # Iterate over slices along the current dimension
                    for j in range(rec.size(dim_idx)):
                        # Select the appropriate slice along each dimension
                        if dim_name == "depth":
                            slice_i = rec[:, :, j, :, :]
                        elif dim_name == "height":
                            slice_i = rec[:, :, :, j, :]
                        else:  # width
                            slice_i = rec[:, :, :, :, j]

                        rec_slice = self.transform(slice_i)
                        rec_arr.append(self.clip.encode_image(rec_slice))

                # Do the element wise difference  between feature_arr and latent_arr
                img_rec_loss_arr = []
                for i in range(len(feature_arr)):
                    img_rec_loss_arr.append(((feature_arr[i] - rec_arr[i]) ** 2).sum(1))

                return torch.stack(img_rec_loss_arr).mean(axis=0)

        else:
            img = self.transform(img)

            img_features = self.clip.encode_image(img)

            # bc_loss = torch.zeros(img.shape[0]).cuda()
            if self.compute_lat_loss:
                latent = latent / 4.6
                latent = latent.mean(1, keepdim=True)
                latent = self.transform(latent.expand(-1, 3, -1, -1))
                latent_features = self.clip.encode_image(latent)

                img_lat_loss = ((img_features - latent_features) ** 2).sum(1)
                return img_lat_loss

            if self.compute_rec_loss:
                rec = torch.clamp((rec + 1.0) / 2.0, min=0.0, max=1.0)
                if rec.shape[1] == 1:
                    rec = rec.expand(-1, 3, -1, -1)
                rec = self.transform(rec)
                rec_features = self.clip.encode_image(rec)
                img_rec_loss = ((img_features - rec_features) ** 2).sum(1)
                return img_rec_loss


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        kl_weight=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        learn_logvar=False,
        num_channels=3,
        ckpt_path=None,
        ignore_keys=[],
        lora=False,
        use_biomedclip_loss=False,
        checkpoint_slices=False,
        adaptive_disc_weight=True,
    ):
        super().__init__()
        self.learn_logvar = learn_logvar
        # Use a fixed discriminator weight instead of the gradient-norm adaptive
        # one; required for memory-constrained full-volume 3D training (see
        # calculate_adaptive_weight).
        self.adaptive_disc_weight = adaptive_disc_weight
        # When True, each per-slice perceptual/discriminator forward is wrapped in
        # gradient checkpointing so the ~600 slice graphs of a full 3D volume are
        # recomputed in backward instead of all held in memory at once. Numerically
        # identical; only needed for large full-volume 3D inputs (e.g. the bottleneck).
        self.checkpoint_slices = checkpoint_slices
        self.kl_weight = kl_weight  # Weight assigned to KL regularization term
        self.perceptual_loss = LPIPS().eval()  # Perceptual loss function
        # self.monai_perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="vgg", is_fake_3d=True, fake_3d_ratio=0.1)
        self.perceptual_weight = perceptual_weight  # Weight assigned to perceptual loss
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)

        self.discriminator = NLayerDiscriminator(input_nc=num_channels).apply(
            weights_init
        )
        self.discriminator_iter_start = disc_start
        self.discriminator_weight = disc_weight  # Weight assigned to generator loss

        self.lora = lora

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

        self.use_biomedclip_loss = use_biomedclip_loss
        if self.use_biomedclip_loss:
            self.biomed_clip_loss = BiomedClipLoss(
                compute_rec_loss=True, compute_lat_loss=False
            )

    def init_from_ckpt(self, path, ignore_keys):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        state_dict = {}
        for k in list(sd.keys()):
            if k.startswith("loss"):
                state_dict[".".join(k.split(".")[1:])] = sd[k]
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path}")

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        if self.lora:
            return torch.tensor(1.0)
        elif not self.adaptive_disc_weight:
            # Fixed disc weight: skips the two extra autograd.grad passes (each
            # with retain_graph=True) that otherwise keep the whole graph alive and
            # roughly triple peak memory -- infeasible for full-volume 3D backward.
            return torch.tensor(self.discriminator_weight, device=nll_loss.device)
        else:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            d_weight = d_weight * self.discriminator_weight
            return d_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        latent: torch.Tensor,
        posteriors: torch.distributions.Distribution,
        optimizer_idx: int,
        global_step: int,
        weight_dtype: torch.dtype,
        last_layer: nn.Module,
        split: str = "train",
    ):
        bsz = inputs.shape[0]

        if optimizer_idx == 0:
            # Perceptual loss
            # Absolute Error
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if len(inputs.size()) == 5:
                p_loss_arr = []
                dims = ["depth", "height", "width"]

                for dim_idx, dim_name in enumerate(dims, start=2):
                    # Iterate over slices along the current dimension
                    for j in range(inputs.size(dim_idx)):
                        # Select the appropriate slice along each dimension
                        if dim_name == "depth":
                            slice_i = inputs[:, :, j, :, :]
                            slice_r = reconstructions[:, :, j, :, :]
                        elif dim_name == "height":
                            slice_i = inputs[:, :, :, j, :]
                            slice_r = reconstructions[:, :, :, j, :]
                        else:  # width
                            slice_i = inputs[:, :, :, :, j]
                            slice_r = reconstructions[:, :, :, :, j]

                        # Calculate perceptual loss for the current slice
                        if self.checkpoint_slices and slice_r.requires_grad:
                            pl = torch.utils.checkpoint.checkpoint(
                                self.perceptual_loss,
                                slice_i.contiguous(),
                                slice_r.contiguous(),
                                use_reentrant=False,
                            )
                        else:
                            pl = self.perceptual_loss(
                                slice_i.contiguous(), slice_r.contiguous()
                            )
                        # Reduce each slice to a scalar so slices taken along axes of
                        # different length (non-cubic volumes) can be stacked.
                        p_loss_arr.append(pl.mean())

                # Average the perceptual loss across all slices
                p_loss = torch.mean(torch.stack(p_loss_arr))

                # monai_loss = self.monai_perceptual_loss(inputs.contiguous(), reconstructions.contiguous())

                # print("2D Loss is ", p_loss.item(), " while 3D loss: ", monai_loss.item())

            else:
                # Perceptual Error (dim = [bsz x 1 x 1 x1])
                p_loss = self.perceptual_loss(
                    inputs.contiguous(), reconstructions.contiguous()
                )
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = (rec_loss / torch.exp(self.logvar) + self.logvar).sum() / bsz

            # BiomedCLIP loss
            if self.use_biomedclip_loss:
                bc_loss = self.biomed_clip_loss(
                    inputs.contiguous(), rec=reconstructions.contiguous(), latent=None
                )
                bc_loss = bc_loss.sum() / bsz

            # KL regularization loss
            kl_loss = posteriors.kl()
            kl_loss = kl_loss.sum() / bsz

            # Generator loss (−L_adv(D(E(x)))): Forces discriminator logits to be high when reconstructions are provided
            d_valid = 0 if global_step < self.discriminator_iter_start else 1
            d_weight = torch.tensor(0.0)
            g_loss = torch.tensor(0.0)
            if d_valid:
                if len(reconstructions.size()) == 5:
                    logits_fake = discriminator_2d_nets_to_3d(
                        self.discriminator, reconstructions, checkpoint=self.checkpoint_slices
                    )
                else:
                    logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)

                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)

            loss = nll_loss + self.kl_weight * kl_loss + d_weight * d_valid * g_loss
            if self.use_biomedclip_loss:
                loss += 100 * bc_loss

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/g_loss": g_loss.detach().mean(),
            }
            if self.use_biomedclip_loss:
                log[f"{split}/bc_loss"] = bc_loss.detach().mean()

            return loss, log

        elif optimizer_idx == 1:
            # Discriminator loss (log D_phi(x)): Forces discriminator logits to be high (+1) for inputs and low (-1) for reconstructions
            d_valid = 0 if global_step < self.discriminator_iter_start else 1
            if d_valid:
                if len(inputs.size()) == 5:
                    logits_real = discriminator_2d_nets_to_3d(
                        self.discriminator, inputs
                    )
                    logits_fake = discriminator_2d_nets_to_3d(
                        self.discriminator, reconstructions
                    )
                else:
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(
                        reconstructions.contiguous().detach()
                    )

                d_loss = d_valid * hinge_loss(logits_real, logits_fake)

                log = {
                    f"{split}/disc_loss": d_loss.clone().detach().mean(),
                    f"{split}/logits_real": logits_real.detach().mean(),
                    f"{split}/logits_fake": logits_fake.detach().mean(),
                }
            else:
                d_loss = torch.tensor(0.0)
                log = {
                    f"{split}/disc_loss": d_loss.mean(),
                }

            return d_loss, log
