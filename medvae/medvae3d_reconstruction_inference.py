import os
import argparse
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn.functional as F

from medvae.utils.factory import build_model
from medvae.utils.extras import roi_size_calc
from monai.inferers import sliding_window_inference


def linear_transform(x, y_min=0, y_max=1, x_min=None, x_max=None, do_clip=True):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max

    if do_clip:
        x = x.clip(x_min, x_max) if hasattr(x, 'clip') else max(min(x, x_max), x_min)

    x_normalized = (x - x_min) / (x_max - x_min)
    y = y_min + (y_max - y_min) * x_normalized
    return y


def get_pad_size(length, k):
    remainder = length % k
    return 0 if remainder == 0 else (k - remainder)


class MVAERecon(torch.nn.Module):
    def __init__(self, model_name: str, config_fpath=None, ckpt_fpath=None, gpu_dim=160):
        super().__init__()
        
        self.ckpt_fpath = ckpt_fpath
        self.model_name = model_name
        self.model = build_model(
            model_name, 
            config_fpath, 
            ckpt_fpath
        )
        
        self.gpu_dim = gpu_dim
        self.encoded_latent = None
        self.decoded_latent = None

    def apply_transform(self, fpath: str, modality):
        raise NotImplementedError("Rebuild transform for reconstruction.")

    def get_transform(self):
        return self.transform

    def init_from_ckpt(self, state_dict: bool = True):
        self.model.init_from_ckpt(self.ckpt_fpath, state_dict=state_dict)

    def _process_3d(self, img, decode: bool = True):
        """Handle 3D image processing with sliding window, ranging within [-1, 1]."""

        def predict_latent(patch):
            if decode:
                dec, _, z = self.model(patch, decode=True)
                return dec, z
            else:
                z, _, _ = self.model(patch, decode=False)
                return z

        # Ensure the input shape is compatible with MedVAE
        b, c, h, w, d = img.shape
        pad_h = get_pad_size(h, 16)
        pad_w = get_pad_size(w, 16)
        pad_d = get_pad_size(d, 16)
        # (D_left, D_right, W_left, W_right, H_left, H_right)
        pad = (0, pad_d, 0, pad_w, 0, pad_h)
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            img = F.pad(img, pad, mode="constant", value=-1)

        roi_size = roi_size_calc(img.shape[-3:], target_gpu_dim=self.gpu_dim)
        # Each roi_size dim must be divisible by the VAE compression factor so that
        # encoder (stride-2 conv ×3 = 8x) and decoder round-trip exactly. If not,
        # e.g. 84 → enc→10 → dec→80 ≠ 84, and MONAI propagates the 80/84 scale to
        # the full volume: 336*(80/84)=320 instead of 336.
        compression_factor = int(self.model_name.split("_")[1])
        roi_size = [(r // compression_factor) * compression_factor for r in roi_size]
        result = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,
            sw_batch_size=1,
            mode="gaussian",
            predictor=predict_latent,
        )

        if decode:
            dec, latent = result
            # Remove padded region and restore original shape [B, C, H, W, D]
            dec = dec[:, :, :h, :w, :d]

            return dec.squeeze().squeeze(), latent.squeeze().squeeze()
        else:
            # This is the latent representation of the image
            return result.squeeze().squeeze()

    def _process_2d(self, img, decode: bool = False):
        """Handle 2D image processing."""
        if decode:
            dec, _, latent = self.model(img, decode=True)
            # This is the decoded image and the latent representation of the image
            return dec.squeeze().squeeze(), latent.squeeze().squeeze()
        else:
            _, _, latent = self.model(img, decode=False)
            # This is the latent representation of the image
            return latent.squeeze().squeeze()
    
    def encode(self, img: torch.tensor):
        """Encode the image into a latent representation. (S1 for 2D, S2 for 3D)"""
        if "3d" in self.model_name:

            def encode_latent(patch):
                z, _, _ = self.model(patch, decode=False)
                return z

            roi_size = roi_size_calc(img.shape[-3:], target_gpu_dim=self.gpu_dim)
            compression_factor = int(self.model_name.split("_")[1])
            roi_size = [(r // compression_factor) * compression_factor for r in roi_size]
            s2_latent = sliding_window_inference(
                inputs=img,
                roi_size=roi_size,
                sw_batch_size=1,
                mode="gaussian",
                predictor=encode_latent,
            )

            return s2_latent

        if "2d" in self.model_name:
            s1_latent = self.model.encode(img).sample()
            return s1_latent

    def decode(self, latent: torch.tensor):
        """Decode the latent representation into an image. (S1 for 2D, S2 for 3D)"""
        
        if "3d" in self.model_name:
            def decode_latent(patch):
                dec = self.model.decode(patch)
                return dec

            # Extract compression factor from model name (e.g., "medvae_4_1_3d" -> 4)
            compression_factor = int(self.model_name.split("_")[1])

            # Calculate ROI size for the original dimensions
            roi_size = roi_size_calc(
                [x * compression_factor for x in latent.shape[-3:]],
                target_gpu_dim=self.gpu_dim,
            )

            # Scale down the ROI size to match the latent space
            roi_size = [size // compression_factor for size in roi_size]

            dec = sliding_window_inference(
                inputs=latent,
                roi_size=roi_size,
                sw_batch_size=1,
                mode="gaussian",
                predictor=decode_latent,
            )
            return dec

        if "2d" in self.model_name:
            dec = self.model.decode(latent)
            return dec

    """
    Forward pass for the model. It will return the S2 2D and 3D latent representation.
    @param img: The image to run inference on.
    @return: The latent representation of the input image.
    """

    def forward(self, img: torch.tensor, decode: bool = False):
        if "3d" in self.model_name:
            return self._process_3d(img, decode)

        if "2d" in self.model_name:
            return self._process_2d(img, decode)


def read_sitk_image(path, return_type='array'):
    im = sitk.ReadImage(path)
    if return_type == 'image':
        return im
    elif return_type == 'array':
        return sitk.GetArrayFromImage(im)
    else:
        return sitk.GetArrayFromImage(im), im


def write_sitk_image(array, path, reference_image=None):
    im = sitk.GetImageFromArray(array)
    if reference_image is not None:
        im.CopyInformation(reference_image)
    sitk.WriteImage(im, path)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Use this to run inference with MedVAE. This function is used when '
                                                 'you want to manually specify a folder containing an pretrained MedVAE '
                                                 'model. ',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
            '--model_name', type=str, default='medvae_8_1_3d',
            help=(
                "There are six MedVAE models that can be used for inference. Choose between:\n"
                "(1) medvae_4_1_2d: 2D images with a 4x compression in each dim (16x total) with a 1 channel latent.\n"
                "(2) medvae_4_3_2d: 2D images with a 4x compression in each dim (64x total) with a 3 channel latent.\n"
                "(3) medvae_8_1_2d: 2D images with an 8x compression in each dim (64x total) with a 1 channel latent.\n"
                "(4) medvae_8_4_2d: 2D images with an 8x compression in each dim (64x total) with a 4 channel latent.\n"
                "(5) medvae_4_1_3d: 3D images with a 4x compression in each dim (64x total) with a 1 channel latent.\n"
                "(6) medvae_8_1_3d: 3D images with an 8x compression in each dim (64x total) with a 1 channel latent.\n"
            )
        )

    parser.add_argument('--modality', type=str, required=True,
                        help='Modality of the input images. Choose between xray, ct, or mri.')
    
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image.')
    
    parser.add_argument('--ckpt_fpath', type=str, required=False, default='MedVAE/model_weights/vae_8x_1c_3D.ckpt', 
                        help='Path to the checkpoint file. If provided, the model will be loaded from the weight in this file.' + 
                        'Note: This should be a ckpt after stage 2 2D and 3D finetuning. If you want stage 1, then modification need to be made')
    
    parser.add_argument('--config_fpath', type=str, required=False, default='MedVAE/model_weights/medvae_8x1.yaml',
                        help='Path to the config file. If provided, the model will be loaded from the config in this file.')        
    
    parser.add_argument('--roi_size', type=int, default=160, required=False, help='Region of interest size for 3D models. This is the maximum dimension size allowed for processing on the GPU.')
    
    parser.add_argument('--device', type=str, default='cuda', required=False, help="Use this to set the device the inference should run with. Available options are 'cuda' (GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! Use CUDA_VISIBLE_DEVICES=X medvae_inference [...] instead!")
    
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where the output images should be saved.')
    parser.add_argument('--save_input', default=False, action='store_true', help='Whether to save the input volume, too.')
    parser.add_argument('--debug', default=False, action='store_true')
    

    args, unknownargs = parser.parse_known_args()
    if unknownargs:
        print(f"Ignoring arguments: {unknownargs}")
    
    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    
    if args.device == 'cpu':
        # let's allow torch to use lots of threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help medvae if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    
    return args, device


def get_orig_info(x):
    D, H, W = x.shape
    x_min = x.min().item()
    x_max = x.max().item()
    orig_dtype = x.dtype
    orig_range = (x_min, x_max)
    orig_shape = (D, H, W)
    return orig_shape, orig_range, orig_dtype



if __name__ == '__main__':
    args, device = parse_arguments()

    model = MVAERecon(
        model_name=args.model_name,
        config_fpath=args.config_fpath, 
        ckpt_fpath=args.ckpt_fpath, 
        gpu_dim=args.roi_size
    ).to(device)
    
    model.requires_grad_(False)
    model.eval()
    
    input_path = args.input_path
    output_dir = args.output_dir
    basename = os.path.basename(input_path)
    save_input_path = os.path.join(output_dir, 'input', basename)
    save_output_path = os.path.join(output_dir, args.model_name, basename)
    
    input_volume, im = read_sitk_image(input_path, return_type='both')  # [D, H, W]
    orig_shape, orig_range, orig_dtype = get_orig_info(input_volume)
    
    if 'ct' in args.modality.lower():
        orig_range = (-1000, 1000)
    else:
        p005, p995 = np.percentile(input_volume, [0.5, 99.5])
        orig_range = (p005, p995)

    if args.debug:
        print(f'ORIG INPUT INFO | range: {orig_range}; shape: {orig_shape}; dtype: {orig_dtype}')
    
    img = linear_transform(input_volume, x_min=min(orig_range), x_max=max(orig_range), y_min=-1, y_max=1)
    img = torch.from_numpy(img).float().to(device)  # [D, H, W]
    img = img.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]
    
    if args.debug:
        print(f'DIRECT INPUT INFO | range: {img.min(), img.max()}; shape: {img.shape}; dtype: {img.dtype}')
        
    rec, _ = model._process_3d(img, decode=True)  # [H, W, D]
    
    if args.debug:
        print(f'DIRECT OUTPUT INFO | range: {rec.min(), rec.max()}; shape: {rec.shape}; dtype: {rec.dtype}')
    
    output_volume = rec.permute(2, 0, 1)  # [D, H, W]
    output_volume = output_volume.squeeze(1).detach().cpu().numpy()
    output_volume = linear_transform(output_volume, x_min=-1, x_max=1, y_min=min(orig_range), y_max=max(orig_range))
    
    if args.debug:
        print(f'FINAL OUTPUT INFO | range: {output_volume.min(), output_volume.max()}; shape: {output_volume.shape}; dtype: {output_volume.dtype}')
        print(f'Saving volume to: {save_output_path}')
    
    os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
    write_sitk_image(output_volume.astype(orig_dtype), save_output_path, reference_image=im)
    
    if args.save_input:
        os.makedirs(os.path.dirname(save_input_path), exist_ok=True)
        write_sitk_image(input_volume, save_input_path, reference_image=im)
