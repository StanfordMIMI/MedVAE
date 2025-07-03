# MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders

[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/stanfordmimi/MedVAE)    [![pypi](https://img.shields.io/pypi/v/medvae?style=for-the-badge)](https://pypi.org/project/medvae/)    [![arXiv](https://img.shields.io/badge/arXiv-2502.14753-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2502.14753)    [![Watch the Talk on YouTube](https://img.shields.io/badge/YouTube-Talk-red?style=for-the-badge&logo=youtube)]([https://youtu.be/XWmCkbpXOUw?si=6GggZgj9U4kbLAKx](https://www.youtube.com/watch?v=5zoxHz71ZgY))    [![License](https://img.shields.io/github/license/stanfordmimi/medvae?style=for-the-badge)](LICENSE)

This repository contains the official PyTorch implementation for [MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders](https://arxiv.org/abs/2502.14753) (MIDL 2025).

![Overview](documentation/assets/overview.png)

## 🫁 What is MedVAE?

MedVAE is a family of six large-scale, generalizable 2D and 3D variational autoencoders (VAEs) designed for medical imaging. It is trained on over one million medical images across multiple anatomical regions and modalities. MedVAE autoencoders encode medical images as downsized latent representations and decode latent representations back to high-resolution images. Across diverse tasks obtained from 20 medical image datasets, we demonstrate that utilizing MedVAE latent representations in place of high-resolution images when training downstream models can lead to efficiency benefits (up to 70x improvement in throughput) while simultaneously preserving clinically-relevant features.

## ⚡️ Installation

To install MedVAE, you can simply run:

```python
pip install medvae
```

For an editable installation, use the following commands to clone and install this repository.

```python
git clone https://github.com/StanfordMIMI/MedVAE.git
cd MedVAE
pip install -e .[dev]
pre-commit install
pre-commit
```

## 🚀 Inference Instructions

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

We also developed an easy-to-use CLI inference tool for compressing your high-dimensional medical images into usable latents:

```python
medvae_inference -i INPUT_FOLDER -o OUTPUT_FOLDER -model_name MED_VAE_MODEL -modality MODALITY
```

For more information, please check our [inference documentation](/documentation/inference.md) and [demo](documentation/demo.ipynb).

## 🔧 Finetuning Instructions

Easily finetune MedVAE on **your own dataset**! Follow the instructions below (requires Python 3.9 and cloning the repository).

Run the following commands depending on your finetuning scenario:

**Stage 1 (2D) Finetuning**

```bash
medvae_finetune experiment=medvae_4x_1c_2d_finetuning
```

**Stage 2 (2D) Finetuning:**

```bash
medvae_finetune_s2 experiment=medvae_4x_1c_2d_s2_finetuning
```

**Stage 2 (3D) Finetuning:**

```bash
medvae_finetune experiment=medvae_4x_1c_3d_finetuning
```

This setup supports multi-GPU training and includes integration with Weights & Biases for experiment tracking.

For detailed finetuning guidelines, see the [Finetuning Documentation](documentation/finetune.md).

To create classification models using downsized latent representations, refer to the [Classification Documentation](documentation/classification.md).

## 📎 Citation

If you find this repository useful for your work, please cite the following paper:

```bibtex
@misc{varma2025medvaeefficientautomatedinterpretation,
      title={MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders}, 
      author={Maya Varma and Ashwin Kumar and Rogier van der Sluijs and Sophie Ostmeier and Louis Blankemeier and Pierre Chambon and Christian Bluethgen and Jip Prince and Curtis Langlotz and Akshay Chaudhari},
      year={2025},
      eprint={2502.14753},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2502.14753}, 
}
```

This repository is powered by [Hydra](https://github.com/facebookresearch/hydra) and [HuggingFace Accelerate](https://github.com/huggingface/accelerate). Our implementation of MedVAE is inspired by prior work on diffusion models from [CompVis](https://github.com/CompVis/latent-diffusion) and [Stability AI](https://github.com/Stability-AI/stablediffusion).
