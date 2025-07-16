# Inference Usage Instruction

MedVAE can be run using either:

- A PyTorch model (programmatic use)
- A command-line interface (CLI) (recommended for beginners)

**Please see the [demo](demo.ipynb) for programmatic examples.**

If you are new to MedVAE and want to downsize your medical images, the CLI approach is recommended.

## **Available MedVAE Models**

MedVAE provides **six pre-trained models** for **2D and 3D medical images**, each with different compression settings:

### **📌 2D Models**

| Model Name | Compression | Latent Channels | Total Compression |
|------------------|------------|-----------------|-------------------|
| `medvae_4_1_2d` | 4× per dim | 1 | 16× total |
| `medvae_4_3_2d` | 4× per dim | 3 | 16× total |
| `medvae_8_1_2d` | 8× per dim | 1 | 64× total |
| `medvae_8_4_2d` | 8× per dim | 4 | 64× total |

### **📌 3D Models**

| Model Name | Compression | Latent Channels | Total Compression |
|------------------|------------|-----------------|-------------------|
| `medvae_4_1_3d` | 4× per dim | 1 | 64× total |
| `medvae_8_1_3d` | 8× per dim | 1 | 512× total |

## 👨‍💻 Programmatic Usage

If you are integrating MedVAE into an existing PyTorch workflow, using it as a PyTorch model is recommended. The [MVAE](../medvae/medvae.py) class provides an easy way to load and use MedVAE models programmatically.

#### **Instantiating a MedVAE Model**

To create an `MVAE` model object, three parameters are needed:

- **`model_name`** – Specifies which of the six available MedVAE models to use.
- **`modality`** – Defines the medical imaging modality (`"xray"`, `"ct"`, or `"mri"`).
- **`gpu_dim`** (optional) – Sets the largest volumetric dimension the GPU can handle.
  - Default: `160`, optimized for a 48GB Nvidia A6000 GPU.

#### **Applying Tranforms**

The `MVAE` class provides an `apply_transforms` method, which automatically applies the appropriate transformation based on the input file type and modality.

- **2D MedVAE models** → Input should be a 2D `.png` file.
- **3D MedVAE models** → Input should be a compressed 3D NIfTI (`*.nii.gz`) file.

For more details, the transforms file is located [here](../medvae/utils/loaders.py).

#### **Example Usage:**

```python
import torch
from medvae import MVAE

fpath = "documentation/data/mmg_data/isJV8hQ2hhJsvEP5rdQNiy.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MVAE(model_name="medvae_4_3_2d", modality="xray").to(device)
img = model.apply_transform(fpath).to(device)

model.requires_grad_(False)
model.eval()

with torch.no_grad():
    latent = model(img)
```

## Obtain Decoded Representations

To obtain the decoded representations, you can set `decode = True` when calling the forward method of the `MVAE` class. Here's an example of how to use it:

```python
with torch.no_grad():
    decoded_img, latent = model(img, decode=True)
```

The `decoded_img` variable will contain the decoded image, and the `latent` variable will contain the latent representation of the input image.

## Using MedVAE part of LDM pipelines

We provide two functions, `encode` and `decode`, to work with latent representations produced by the model: Stage 1 latents for 2D images and Stage 2 latents for 3D volumes. These latents can potentially be used in downstream tasks such as medical latent diffusion models (*though we have not tested this*).

```python
with torch.no_grad():
    latent = model.encode(img)
    dec = model.decode(img)
```

## 🖥️ CLI Usage

The CLI script runs inference using MedVAE, processing 2D or 3D medical images to generate latent representations. It allows users to specify a pretrained MedVAE model and input modalities (X-ray, CT, MRI). Given an input directory, it will process all the medical images into latent representations and save them in the specified folder.

```python
medvae_inference -i INPUT_FOLDER -o OUTPUT_FOLDER -model_name MED_VAE_MODEL -modality MODALITY
```

### Arguments

| Argument | Type | Required | Description |
|--------------|------|----------|-------------------------------------------------------------------------------------------------|
| -i | str | ✅ Yes | Path to the input folder containing images (\*.png for 2D, \*.nii.gz for 3D). The filenames must not contain multiple dots. |
| -o | str | ✅ Yes | Path to the output folder where latent representations will be saved. If the folder does not exist, it will be created. |
| -model_name | str | ✅ Yes | Specifies the Med-VAE model to use. See available options above. |
| -modality | str | ✅ Yes | Specifies the image modality: "xray", "ct", or "mri". |
| -roi_size | int | ❌ No (Default: 160) | Sets the region of interest (ROI) size for 3D models (used to manage GPU memory). |
| -device | str | ❌ No (Default: "cuda") | Specifies the device to run inference on: "cuda" (GPU), "cpu" (CPU), "mps" (Apple M1/M2). Do not specify GPU ID here! Use CUDA_VISIBLE_DEVICES=X instead. |

## 🤗 Model Files on Huggingface

| Total Compression Factor | Channels | Dimensions | Modalities | Anatomies | Config File | Model File |
|----------|----------|----------|----------|----------|----------|----------|
| 16 | 1 | 2D | X-ray | Chest, Breast (FFDM) | [medvae_4x1.yaml ](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/medvae_4x1.yaml) | [vae_4x_1c_2D.ckpt](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/vae_4x_1c_2D.ckpt)
| 16 | 3 | 2D | X-ray | Chest, Breast (FFDM) | [medvae_4x3.yaml](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/medvae_4x3.yaml) | [vae_4x_3c_2D.ckpt](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/vae_4x_3c_2D.ckpt)
| 64 | 1 | 2D | X-ray | Chest, Breast (FFDM) | [medvae_8x1.yaml](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/medvae_8x1.yaml) | [vae_8x_1c_2D.ckpt](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/vae_8x_1c_2D.ckpt)
| 64 | 3 | 2D | X-ray | Chest, Breast (FFDM) | [medvae_8x4.yaml](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/medvae_8x4.yaml) | [vae_8x_4c_2D.ckpt](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/vae_8x_4c_2D.ckpt)
| 64 | 1 | 3D | MRI, CT | Whole-Body | [medvae_4x1.yaml ](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/medvae_4x1.yaml) | [vae_4x_1c_3D.ckpt](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/vae_4x_1c_3D.ckpt)
| 512 | 1 | 3D | MRI, CT | Whole-Body | [medvae_8x1.yaml](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/medvae_8x1.yaml) | [vae_8x_1c_3D.ckpt](https://huggingface.co/stanfordmimi/MedVAE/blob/main/model_weights/vae_8x_1c_3D.ckpt)

## Creating a MedVAE conda environment

Run the following in your terminal or command prompt:

```python
conda create --name medvae python=3.9
```

To activate the environment, enter:

```python
conda activate medvae
```

To delete the environment, enter:

```python
conda remove --name medvae --all
```

## Running pre-commit

To install the project as a development package, run the following command in your terminal or command prompt:

```python
pip install -e .[dev]
```

Install pre-commit

```python
pre-commit install
```

Run pre-commit

```python
pre - commit
```
