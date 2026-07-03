from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    Orientation,
    ScaleIntensity,
    CropForeground,
    ScaleIntensityRange,
    RandSpatialCrop,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
    ToTensord,
)
import torch
import torch.nn.functional as F
from monai.transforms import Transform
import torchvision
from PIL import Image
import polars as pl
import numpy as np


class MonaiNormalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return torchvision.transforms.Normalize(self.mean, self.std)(img)


class MonaiPad(Transform):
    def __init__(self, size, mode="constant", value=0):
        self.size = size
        self.mode = mode
        self.value = value

    def __call__(self, img):
        padding = []
        for i in range(len(img.shape) - 1, 0, -1):  # Start from the last dimension
            total_pad = max(self.size[i - 1] - img.shape[i], 0)
            padding.extend([total_pad // 2, total_pad - total_pad // 2])
        padded_img = F.pad(img, padding, mode=self.mode, value=self.value)
        return padded_img


class MonaiImageOpen(Transform):
    def __init__(self):
        pass

    def __call__(self, path):
        return Image.open(path)


"""
Custom transform to normalize and pad 2D images
@input: path to image (str)
@Output: padding (np.array)
"""


def load_2d(
    path: str,
    merge_channels: bool = False,
    dtype: torch.dtype = torch.float32,
    **kwargs,
):
    img_transforms = Compose(
        transforms=[
            MonaiImageOpen(),
            torchvision.transforms.ToTensor(),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
        ],
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()

        if merge_channels:
            img = img.mean(0, keepdim=True)

        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 384, 384))


def load_2d_finetune(
    path: str, dtype: torch.dtype = torch.float32, merge_channels: bool = True, **kwargs
):
    img_transforms = Compose(
        transforms=[
            MonaiImageOpen(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((384, 384), interpolation=3, antialias=True),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
        ],
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()
        if merge_channels:
            img = img.mean(0, keepdim=True)
        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 384, 384))


"""
Custom transform to normalize, crop, and pad 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_mri_3d(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    mri_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            CropForeground(k_divisible=[16, 16, 16]),
        ],
        lazy=True,
    )

    try:
        mr_augmented = mri_transforms(path).as_tensor()
        return mr_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 128, 128, 128))


"""
Custom transform to normalize, crop, and pad 3D CT volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_ct_3d(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    ct_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            CropForeground(k_divisible=[16, 16, 16]),
        ],
        lazy=True,
    )

    try:
        ct_augmented = ct_transforms(path).as_tensor()
        return ct_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 256, 256, 256))


def load_ct_merlin(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    """Load a full CT volume at Merlin's fixed 224x224x160 grid.

    Unlike ``load_ct_3d_finetune`` (random 64^3 crops), this returns the whole
    resampled volume so the global bottleneck can learn context over the full
    image. Output is normalized to [-1, 1] to match the range the frozen MedVAE
    backbone was trained on.
    """
    transforms_image = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            SpatialPadd(keys=["image"], spatial_size=[224, 224, 160], value=0.0),
            CenterSpatialCropd(
                roi_size=[224, 224, 160],
                keys=["image"],
            ),
            ToTensord(keys=["image"]),
        ]
    )

    try:
        img = transforms_image({"image": path})["image"]
        # [0, 1] -> [-1, 1] to match the MedVAE backbone's expected input range.
        img = MonaiNormalize(mean=[0.5], std=[0.5])(img)
        return img.as_tensor() if hasattr(img, "as_tensor") else img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 224, 224, 160))


"""
Custom transform to normalize, crop, and pad 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_mri_3d_finetune(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    mri_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            MonaiPad(size=[64, 64, 64], value=-1),
            RandSpatialCrop(roi_size=[64, 64, 64]),
        ],
        lazy=True,
    )

    try:
        mr_augmented = mri_transforms(path).as_tensor()
        return mr_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 64, 64, 64))


"""
Custom transform to normalize, crop, and pad ct 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_ct_3d_finetune(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    ct_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            MonaiPad(size=[64, 64, 64], value=-1),
            RandSpatialCrop(roi_size=[64, 64, 64]),
        ],
        lazy=True,
    )

    try:
        ct_augmented = ct_transforms(path).as_tensor()
        return ct_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 64, 64, 64))


def load_labels(
    df: pl.DataFrame,
    dtype: np.dtype = None,
    # fill_null=None,
    fill_nan: float = None,
    squeeze: int = None,
) -> torch.Tensor:
    """Load the labels from a dataframe."""
    # BUG: Polars hangs when trying to convert to numpy in a DataLoader
    x = df.to_pandas().to_numpy()
    if dtype is not None:
        x = x.astype(dtype)

    if isinstance(squeeze, int):
        out = torch.from_numpy(x).squeeze(dim=squeeze)
    else:
        out = torch.from_numpy(x).squeeze()

    if isinstance(fill_nan, float):
        out = torch.where(out.isnan(), fill_nan, out)

    return out
