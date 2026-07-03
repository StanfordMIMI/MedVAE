from huggingface_hub import hf_hub_download
from medvae.models import AutoencoderKL_2D, AutoencoderKL_3D, AutoencoderKLBottleneck
from omegaconf import OmegaConf
from medvae.utils.lora import inject_trainable_lora_extended
from medvae.utils.loaders import load_mri_3d, load_ct_3d, load_2d

HF_REPO_PATH = "stanfordmimi/MedVAE"

FILE_DICT_ASSOCIATIONS = {
    "medvae_4_1_2d": {
        "config": "model_weights/medvae_4x1.yaml",
        "ckpt": "model_weights/vae_4x_1c_2D.ckpt",
    },
    "medvae_4_3_2d": {
        "config": "model_weights/medvae_4x3.yaml",
        "ckpt": "model_weights/vae_4x_3c_2D.ckpt",
    },
    "medvae_4_4_2d": {
        "config": "model_weights/medvae_4x4.yaml",
        "ckpt": "model_weights/vae_4x_4c_2D.ckpt",
    },
    "medvae_8_1_2d": {
        "config": "model_weights/medvae_8x1.yaml",
        "ckpt": "model_weights/vae_8x_1c_2D.ckpt",
    },
    "medvae_8_4_2d": {
        "config": "model_weights/medvae_8x4.yaml",
        "ckpt": "model_weights/vae_8x_4c_2D.ckpt",
    },
    "medvae_4_1_3d": {
        "config": "model_weights/medvae_4x1.yaml",
        "ckpt": "model_weights/vae_4x_1c_3D.ckpt",
    },
    "medvae_8_1_3d": {
        "config": "model_weights/medvae_8x1.yaml",
        "ckpt": "model_weights/vae_8x_1c_3D.ckpt",
    },
    # Frozen backbone + trainable global bottleneck (reuse the base 3D weights).
    "medvae_4_1_3d_bottleneck": {
        "config": "model_weights/medvae_4x1.yaml",
        "ckpt": "model_weights/vae_4x_1c_3D.ckpt",
    },
    "medvae_8_1_3d_bottleneck": {
        "config": "model_weights/medvae_8x1.yaml",
        "ckpt": "model_weights/vae_8x_1c_3D.ckpt",
    },
}

""" 
Download model weights from Hugging Face Hub
"""


def download_model_weights(hfpath):
    fpath = hf_hub_download(repo_id=HF_REPO_PATH, filename=hfpath)
    return fpath


""" 
Build the Med-VAE models for inference using the model weights
"""


def build_model(
    model_name: str,
    config_fpath: str,
    ckpt_fpath: str,
    training: bool = False,
    existing_weight: str = None,
    state_dict: bool = True,
):
    if (
        model_name == "medvae_4_1_2d"
        or model_name == "medvae_8_1_2d"
        or model_name == "medvae_4_4_2d"
    ):
        conf = OmegaConf.load(config_fpath)
        model = AutoencoderKL_2D(
            ddconfig=conf.ddconfig,
            embed_dim=conf.embed_dim,
            ckpt_path=ckpt_fpath if existing_weight is None else existing_weight,
            state_dict=state_dict,
        )
    elif model_name == "medvae_4_3_2d" or model_name == "medvae_8_4_2d":
        conf = OmegaConf.load(config_fpath)
        model = AutoencoderKL_2D(
            ddconfig=conf.model.params.ddconfig,
            embed_dim=conf.model.params.embed_dim,
        )
        # If training, freeze the encoder and decoder and inject the lora
        if training:
            print(
                "Trainable Params before LORA:",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
            model.encoder.requires_grad_(False)
            model.decoder.requires_grad_(False)
            _, _ = inject_trainable_lora_extended(
                model, {"ResnetBlock", "AttnBlock"}, r=4
            )
            print(
                "Trainable Params after LORA:",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
        else:
            _, _ = inject_trainable_lora_extended(
                model, {"ResnetBlock", "AttnBlock"}, r=4
            )

        model.init_from_ckpt(
            ckpt_fpath if existing_weight is None else existing_weight,
            state_dict=state_dict,
        )
    elif model_name == "medvae_4_1_3d" or model_name == "medvae_8_1_3d":
        conf = OmegaConf.load(config_fpath)
        model = AutoencoderKL_3D(
            ddconfig=conf.ddconfig,
            embed_dim=conf.embed_dim,
        )
        model.init_from_ckpt(
            ckpt_fpath if existing_weight is None else existing_weight,
            state_dict=state_dict,
        )
    elif model_name in ("medvae_4_1_3d_bottleneck", "medvae_8_1_3d_bottleneck"):
        conf = OmegaConf.load(config_fpath)
        model = AutoencoderKLBottleneck(
            ddconfig=conf.ddconfig,
            embed_dim=conf.embed_dim,
        )
        # Loads pretrained weights into the (frozen) backbone; the bottleneck
        # heads are absent from the checkpoint and stay at their initialization.
        model.init_from_ckpt(
            ckpt_fpath if existing_weight is None else existing_weight,
            state_dict=state_dict,
        )

    return model


"""
Build the transform for the model
"""


def build_transform(model_name: str, modality: str):
    if "3d" in model_name:
        if "ct" in modality.lower():
            transform = load_ct_3d
        elif "mri" in modality.lower():
            transform = load_mri_3d
        else:
            raise ValueError(f"Modality {modality} not supported for 3D models")
    elif "2d" in model_name:
        transform = load_2d
    else:
        raise ValueError(
            f"Model name {model_name} not supported. Needs to be a 2D or 3D model."
        )

    return transform


"""
Create a model and transform from a model name
"""


def create_model_and_transform(
    model_name: str,
    modality: str,
):
    # Check if model_name is in FILE_DICT_ASSOCIATIONS
    if model_name not in FILE_DICT_ASSOCIATIONS:
        raise ValueError(f"Model name {model_name} not found in FILE_DICT_ASSOCIATIONS")

    # Download the model_weights
    config_fpath = download_model_weights(FILE_DICT_ASSOCIATIONS[model_name]["config"])
    ckpt_fpath = download_model_weights(FILE_DICT_ASSOCIATIONS[model_name]["ckpt"])

    # Build the model
    model = build_model(model_name, config_fpath, ckpt_fpath)

    # Get the transform
    transform = build_transform(model_name, modality)

    return model, transform


"""
Create a model from a model name
"""


def create_model(
    model_name: str,
    existing_weight: str = None,
    training: bool = True,
    state_dict: bool = True,
):
    # Check if model_name is in FILE_DICT_ASSOCIATIONS
    if model_name not in FILE_DICT_ASSOCIATIONS:
        raise ValueError(f"Model name {model_name} not found in FILE_DICT_ASSOCIATIONS")

    # Download the model_weights
    config_fpath = download_model_weights(FILE_DICT_ASSOCIATIONS[model_name]["config"])
    ckpt_fpath = download_model_weights(FILE_DICT_ASSOCIATIONS[model_name]["ckpt"])

    # Build the model
    model = build_model(
        model_name,
        config_fpath,
        ckpt_fpath,
        training=training,
        existing_weight=existing_weight,
        state_dict=state_dict,
    )

    return model
