import torch
import pyrootutils
from medvae.utils.extras import cite_function
from medvae.utils.factory import create_model
from medvae.utils.extras import sanitize_dataloader_kwargs, set_seed
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import GradientAccumulationPlugin
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from medvae.utils.vae.train_components import training_epoch, validation_epoch

"""
Process configuration from Hydra instead of command line arguments.

Args:
    cfg: Hydra configuration object
    
Returns:
    The processed configuration: hydra config object
"""


def parse_arguments(cfg: DictConfig):
    # Print a message to cite the med-vae paper
    cite_function()

    # Validate model_name
    valid_model_names = [
        "medvae_4_1_2d",
        "medvae_4_3_2d",
        "medvae_4_4_2d",
        "medvae_8_1_2d",
        "medvae_8_4_2d",
        "medvae_4_1_3d",
        "medvae_8_1_3d",
        "medvae_4_1_3d_bottleneck",
        "medvae_8_1_3d_bottleneck",
    ]
    assert cfg.model_name in valid_model_names, (
        f"model_name must be one of {valid_model_names}. Got: {cfg.model_name}."
    )

    assert cfg.stage2 is False, (
        f"stage2 must be False. This script is not used for 2D stage 2 finetuning. Got: {cfg.stage2}."
    )

    return cfg


# Set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")

# Register configuration resolvers
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(
    version_base="1.2", config_path=config_dir, config_name="finetuned_vae.yaml"
)
def main(cfg: DictConfig):
    cfg = parse_arguments(cfg)

    # Instantiating config
    print(f"=> Starting [experiment={cfg.get('task_name', 'default')}]")
    cfg = instantiate(cfg)

    # Seeding
    if cfg.get("seed", None) is not None:
        print(f"=> Setting seed [seed={cfg.seed}]")
        set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup accelerator
    logger_kwargs = cfg.get("logger", None)
    is_logging = bool(logger_kwargs)
    print(f"=> Instantiate accelerator [logging={is_logging}]")

    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            adjust_scheduler=False,
        ),
        mixed_precision=cfg.get("mixed_precision", None),
        log_with="wandb" if is_logging else None,
        split_batches=True,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True,
            )
        ],
    )
    accelerator.init_trackers(
        "medvae", config=cfg, init_kwargs={"wandb": logger_kwargs}
    )

    # Determine the mode
    print(f"=> Mixed precision: {accelerator.mixed_precision}")

    inference_mode = cfg.get("inference", False)
    print(f"=> Running in inference mode: {inference_mode}")

    print(f"=> Instantiating train dataloader [device={accelerator.device}]")
    train_dataloader = DataLoader(
        **sanitize_dataloader_kwargs(cfg["dataloader"]["train"])
    )

    print(f"=> Instantiating valid dataloader [device={accelerator.device}]")
    valid_dataloader = DataLoader(
        **sanitize_dataloader_kwargs(cfg["dataloader"]["valid"])
    )

    # Create loss function
    criterion = cfg.criterion

    # Only run the discriminator when its needed (stage 1 -- 2D; Stage 2 -- 3D)
    # We delay the discriminator by setting a start to avoid potential mode collapse
    discriminator_iter_start = criterion.discriminator_iter_start

    # Create model
    model = create_model(cfg.model_name)

    # Model-parallel: shard the frozen decoder across GPUs so a full-volume 3D
    # image-space step fits. The model is placed manually and NOT handed to
    # accelerate.prepare (which would move it to one device / DDP-wrap it).
    model_parallel = cfg.get("model_parallel", False)
    if model_parallel:
        ngpu = cfg.get("model_parallel_gpus", 3)
        print(f"=> Model-parallel across {ngpu} GPUs")
        model.to_model_parallel_auto(ngpu)
        # register so accelerator.save_state / load_state checkpoint the bottleneck
        accelerator.register_for_checkpointing(model)
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Create two optimizers: one for the autoencoder and one for the discriminator
    print(f"=> Instantiating the optimizer [device={accelerator.device}]")

    batch_size, lr = cfg.batch_size, cfg.base_learning_rate
    lr = gradient_accumulation_steps * batch_size * lr

    # Create autoencoder parameters. For the bottleneck model the backbone is
    # frozen, so only the trainable bottleneck heads go to the optimizer.
    if hasattr(model, "bottleneck_parameters"):
        ae_params = list(model.bottleneck_parameters())
    else:
        ae_params = (
            list(model.encoder.parameters())
            + list(model.decoder.parameters())
            + list(model.quant_conv.parameters())
            + list(model.post_quant_conv.parameters())
        )

    if criterion.learn_logvar:
        ae_params.append(criterion.logvar)
    opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))

    opt_disc = torch.optim.Adam(
        criterion.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
    )

    num_metrics = 5
    # This mean uses lora and needs biomedclip loss
    if cfg.model_name in ["medvae_4_3_2d", "medvae_8_4_2d"]:
        num_metrics += 1

    # Prepare components for multi-gpu/mixed precision training. In model-parallel
    # mode the model is placed manually across GPUs and is intentionally excluded
    # from prepare; its forward applies autocast internally.
    if model_parallel:
        (train_dataloader, valid_dataloader, opt_ae, opt_disc, criterion) = (
            accelerator.prepare(
                train_dataloader, valid_dataloader, opt_ae, opt_disc, criterion
            )
        )
    else:
        (train_dataloader, valid_dataloader, model, opt_ae, opt_disc, criterion) = (
            accelerator.prepare(
                train_dataloader,
                valid_dataloader,
                model,
                opt_ae,
                opt_disc,
                criterion,
            )
        )

    # Create metrics: aeloss, discloss, recloss, data(time), batch(time)
    default_metrics = accelerator.prepare(*[MeanMetric() for _ in range(num_metrics)])
    if len(cfg["metrics"]) > 0:
        names, metrics = list(zip(*cfg["metrics"]))
        metrics = list(zip(names, accelerator.prepare(*metrics)))
    else:
        metrics = []

    # Resume from checkpoint
    start_epoch = cfg.start_epoch
    if cfg.resume_from_ckpt is not None:
        print("Loading Model from Checkpoint: ", cfg.resume_from_ckpt)
        accelerator.load_state(cfg.resume_from_ckpt)

    options = {
        "max_epoch": cfg["max_epoch"],
        "is_logging": is_logging,
        "log_every_n_steps": cfg["log_every_n_steps"],
        "ckpt_every_n_steps": cfg["ckpt_every_n_steps"],
        "ckpt_dir": cfg["ckpt_dir"],
        "fast_dev_run": cfg["fast_dev_run"],
    }

    print(f"=> Starting model training [epochs={cfg['max_epoch']}]")
    min_loss = None
    global_step = cfg.get("global_step", 0)
    for epoch in range(start_epoch, cfg["max_epoch"]):
        global_step = training_epoch(
            options=options,
            epoch=epoch,
            global_step=global_step,
            accelerator=accelerator,
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            discriminator_iter_start=discriminator_iter_start,
            default_metrics=default_metrics,
            rec_metrics=metrics,
            optimizer_ae=opt_ae,
            optimizer_disc=opt_disc,
        )

        accelerator.wait_for_everyone()

        loss = validation_epoch(
            options=options,
            epoch=epoch,
            accelerator=accelerator,
            dataloader=valid_dataloader,
            model=model,
            criterion=criterion,
            default_metrics=default_metrics,
            rec_metrics=metrics,
            global_step=global_step,
        )

        # save the best model
        if min_loss is None or loss < min_loss:
            try:
                accelerator.save_state(
                    os.path.join(cfg.ckpt_dir, "best.pt"), safe_serialization=False
                )
            except Exception as e:
                print(e)
            min_loss = loss

        # save checkpoint
        if (epoch + 1) % cfg.get("ckpt_every_n_epochs", 1) == 0:
            print(f"=> Saving checkpoint [epoch={epoch}]")
            try:
                accelerator.save_state(
                    os.path.join(cfg.ckpt_dir, f"epoch-{epoch:04d}.pt"),
                    safe_serialization=False,
                )
            except Exception as e:
                print(e)

    # save last model
    accelerator.save_state(
        os.path.join(cfg.ckpt_dir, "last.pt"), safe_serialization=False
    )

    # save model to output directory
    accelerator.save(
        model, os.path.join(cfg.output, "model.pt"), safe_serialization=False
    )

    print(f"=> Finished model training [epochs={cfg['max_epoch']}, metric={min_loss}]")
    accelerator.end_training()


if __name__ == "__main__":
    main()
