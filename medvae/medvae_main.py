from medvae.utils.factory import create_model_and_transform
from medvae.utils.extras import roi_size_calc
import torch
from monai.inferers import sliding_window_inference

"""
Large Med-VAE class to abstract all the models.

This allows for interfacing with Med-VAE as a pytorch model.
Can be used for 2D and 3D inference / finetuning.

@param model_name: The name of the model to use. Choose between:
(1) medvae_4_1_2d: 2D images with a 4x compression in each dim (16x total) with a 1 channel latent.
(2) medvae_4_3_2d: 2D images with a 4x compression in each dim (64x total) with a 3 channel latent.
(3) medvae_8_1_2d: 2D images with an 8x compression in each dim (64x total) with a 1 channel latent.
(4) medvae_8_4_2d: 2D images with an 8x compression in each dim (64x total) with a 4 channel latent.
(5) medvae_4_1_3d: 3D images with a 4x compression in each dim (64x total) with a 1 channel latent.
(6) medvae_8_1_3d: 3D images with an 8x compression in each dim (64x total) with a 1 channel latent.

@param modality: Modality of the input images. Choose between xray, ct, or mri.

@param gpu_dim: The maximum dimension size allowed for processing on the GPU (default is 160).

@return (forward): The latent representation of the input image (torch.tensor).
"""


class MVAE(torch.nn.Module):
    def __init__(self, model_name: str, modality: str, gpu_dim=160):
        super(MVAE, self).__init__()

        self.model_name = model_name
        self.modality = modality

        self.model, self.transform = create_model_and_transform(
            self.model_name, self.modality
        )

        self.gpu_dim = gpu_dim

        self.encoded_latent = None
        self.decoded_latent = None

    def apply_transform(self, fpath: str):
        if "3d" in self.model_name:
            return self.transform(fpath).unsqueeze(0)
        elif "2d" in self.model_name:
            return self.transform(
                fpath, merge_channels="1_2d" in self.model_name
            ).unsqueeze(0)
        else:
            raise ValueError(
                f"Model name {self.model_name} not supported. Needs to be a 2D or 3D model."
            )

    def get_transform(self):
        return self.transform

    def init_from_ckpt(self, ckpt_path: str, state_dict: bool = True):
        self.model.init_from_ckpt(ckpt_path, state_dict=state_dict)

    def _process_3d(self, img, decode: bool = False):
        """Handle 3D image processing with sliding window."""

        def predict_latent(patch):
            if decode:
                dec, _, z = self.model(patch, decode=True)
                return dec, z
            else:
                z, _, _ = self.model(patch, decode=False)
                return z

        roi_size = roi_size_calc(img.shape[-3:], target_gpu_dim=self.gpu_dim)
        result = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,
            sw_batch_size=1,
            mode="gaussian",
            predictor=predict_latent,
        )

        if decode:
            dec, latent = result
            # This is the decoded image and the latent representation of the image
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
