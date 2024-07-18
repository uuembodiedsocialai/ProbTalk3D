# Adapted from the code of TEMOS

import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pathlib import Path
import framework.launch.prepare  # noqa

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def _train(cfg: DictConfig):
    cfg.trainer.enable_progress_bar = True
    return train(cfg)


def train(cfg: DictConfig) -> None:
    working_dir = cfg.path.working_dir
    logger.info("Training script. The outputs will be stored in:")
    logger.info(f"{working_dir}")

    # Delayed imports to get faster parsing
    logger.info("Loading libraries")
    import pytorch_lightning as pl
    logger.info("Libraries loaded")

    logger.info(f"Set the seed to {cfg.seed}")
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.data_name}' loaded")

    logger.info("Loading model")
    # Check if the model is resumed training or training from scratch
    if cfg.state == "resume":
        resumed_training = True
    elif cfg.state == "new":
        resumed_training = False
    else:
        raise ValueError(f"State unknown, please set training state to 'resume' or 'new'")

    # Load model and metric monitor
    metric_monitor, model = pre_setting(cfg, data_module, resumed_training)

    logger.info("Loading callbacks")
    callbacks = [
        pl.callbacks.RichProgressBar(),
        instantiate(cfg.callback.progress, metric_monitor=metric_monitor),
        instantiate(cfg.callback.best_epoch_ckpt),  # Best epoch checkpoint (on validation loss)
        instantiate(cfg.callback.last_ckpt),        # Latest checkpoint and last.ckpt
        instantiate(cfg.callback.early_stopping),
    ]
    logger.info("Callbacks initialized")

    logger.info("Loading trainer")
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=None,
        callbacks=callbacks,
    )
    logger.info("Trainer initialized")

    logger.info("Fitting the model..")
    if not resumed_training:
        print("Training from scratch")
        trainer.fit(model, datamodule=data_module)
    else:
        if os.path.exists(cfg.ckpt_path):
            print("Resume training from a checkpoint")
            trainer.fit(model, datamodule=data_module, ckpt_path=Path(cfg.ckpt_path))
        else:
            raise ValueError(f"Checkpoint path not found: {cfg.ckpt_path}")

    logger.info("Fitting done")

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(f"Training done.")


def prior_model_init(cfg, data_module, resumed_training):
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        resumed_training=resumed_training,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    return model


def pred_model_init(cfg, data_module, resumed_training):
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        split_path=data_module.split_path,
                        one_hot_dim=data_module.one_hot_dim,
                        resumed_training=resumed_training,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded.")
    return model


def pre_setting(cfg, data_module, resumed_training):
    if hasattr(cfg.model, 'vae_prior') and cfg.model.vae_prior:
        metric_monitor = {
            "L_kl_val": "kl/val",
            "L_rec_exp_val": "recons/exp/val",
            "L_rec_jaw_val": "recons/jaw/val",
            "L_total_val": "total/val",
            "M_mean_l2": "Metrics/mean_l2",
            "M_mean_var": "Metrics/mean_var",
            "L_kl_train": "kl/train",
            "L_rec_exp_train": "recons/exp/train",
            "L_rec_jaw_train": "recons/jaw/train",
            "L_total_train": "total/train"
        }
        model = prior_model_init(cfg, data_module, resumed_training)
    elif hasattr(cfg.model, 'vqvae_prior') and cfg.model.vqvae_prior:
        metric_monitor = {
            "L_quant_val": "quant/val",
            "L_rec_exp_val": "recons/exp/val",
            "L_rec_jaw_val": "recons/jaw/val",
            "L_total_val": "total/val",
            "M_mean_l2": "Metrics/mean_l2",
            "M_mean_var": "Metrics/mean_var",
            "L_quant_train": "quant/train",
            "L_rec_exp_train": "recons/exp/train",
            "L_rec_jaw_train": "recons/jaw/train",
            "L_total_train": "total/train"
        }
        model = prior_model_init(cfg, data_module, resumed_training)
    elif hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        metric_monitor = {
            "L_kl_val": "kl/val",
            "L_rec_vert_val": "recons/vert/val",
            "L_latent_manifold_val": "latent/manifold/val",
            "L_rec_exp_val": "recons/exp/val",
            "L_rec_jaw_val": "recons/jaw/val",
            "L_total_val": "total/val",
            "M_mean_l2": "Metrics/mean_l2",
            "M_mean_var": "Metrics/mean_var",
            "L_kl_train": "kl/train",
            "L_rec_vert_train": "recons/vert/train",
            "L_latent_manifold_train": "latent/manifold/train",
            "L_rec_exp_train": "recons/exp/train",
            "L_rec_jaw_train": "recons/jaw/train",
            "L_total_train": "total/train"
        }
        model = pred_model_init(cfg, data_module, resumed_training)
    elif hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        metric_monitor = {
            "L_cross_val": "crossEntropy/val",
            "L_latent_manifold_val": "latent/manifold/val",
            "L_rec_exp_val": "recons/exp/val",
            "L_rec_jaw_val": "recons/jaw/val",
            "L_total_val": "total/val",
            "M_mean_l2": "Metrics/mean_l2",
            "M_mean_var": "Metrics/mean_var",
            "L_cross_train": "crossEntropy/train",
            "L_latent_manifold_train": "latent/manifold/train",
            "L_rec_exp_train": "recons/exp/train",
            "L_rec_jaw_train": "recons/jaw/train",
            "L_total_train": "total/train"
        }
        model = pred_model_init(cfg, data_module, resumed_training)
    else:
        raise ValueError(f"Model setting unavailable...")

    return metric_monitor, model


if __name__ == '__main__':
    _train()
