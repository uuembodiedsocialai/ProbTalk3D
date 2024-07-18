from typing import List, Optional, Dict
import logging
import torch
from torch import nn, Tensor
from torch.distributions.distribution import Distribution
from hydra.utils import instantiate

from framework.model.metrics.compute import ComputeMetrics
from framework.model.base import BaseModel

logger = logging.getLogger(__name__)


class VAE(BaseModel):
    def __init__(self, 
                 nfeats: int,  # input dimension
                 resumed_training: bool,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.resumed_training = resumed_training
        self.working_dir = self.hparams.working_dir

        self.motion_encoder = instantiate(self.hparams.motion_encoder, nfeats=nfeats)
        logger.info(f"(1). Motion Encoder '{self.motion_encoder.hparams.name}' loaded. ")
        self.motion_decoder = instantiate(self.hparams.motion_decoder, nfeats=nfeats)
        logger.info(f"(2). Motion Decoder '{self.motion_decoder.hparams.name}' loaded.")

        self.mean = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.logvar = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)

        self.optimizer = instantiate(self.hparams.optim, params=self.parameters())
        self._losses = torch.nn.ModuleDict({split: instantiate(self.hparams.losses, _recursive_=False)
                                            for split in ["losses_train", "losses_test", "losses_val"]})

        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.metrics = ComputeMetrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None    # training setting

        self.__post_init__()

    # Forward: audio => motion, called during sampling
    def forward(self, batch: Dict) -> List[Tensor]:
        self.motion_encoder.to(self.device)
        self.mean.to(self.device)
        self.logvar.to(self.device)
        self.motion_decoder.to(self.device)

        batch['motion'] = torch.cat(batch['motion'], dim=0).to(self.device)

        # Encode
        encoded_motion = self.motion_encoder(batch['motion'])
        B, T = encoded_motion.shape[:2]
        encoded_motion = encoded_motion.reshape(B * T, -1)
        mean = self.mean(encoded_motion)
        logvar = self.logvar(encoded_motion)
        std = torch.exp(0.5 * logvar)

        # reparameterization trick
        distribution = torch.distributions.Normal(mean, std)
        latent_vector = self.sample_from_distribution(distribution)     # [B*T, latent_dim]
        latent_vector = latent_vector.reshape(B, T, -1)

        # Decode
        motion_pred = self.motion_decoder(latent_vector)
        return motion_pred

    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc         # the mean parameter of the distribution

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()   # default

        # Resclale the eps, at inference
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector
    
    def get_latent_vector(self, batch_motion):
        encoded_motion = self.motion_encoder(batch_motion)
        B, T = encoded_motion.shape[:2]
        encoded_motion = encoded_motion.reshape(B * T, -1)
        mean = self.mean(encoded_motion)
        logvar = self.logvar(encoded_motion)
        std = torch.exp(0.5 * logvar)
        # reparameterization trick
        distribution = torch.distributions.Normal(mean, std)
        latent_vector = self.sample_from_distribution(distribution)     # [B*T, latent_dim]
        latent_vector = latent_vector.reshape(B, T, -1)  # [B, T, 256]
        return latent_vector, distribution

    # Called during training
    def allsplit_step(self, split: str, batch, batch_idx):
        batch['motion'] = torch.cat(batch['motion'], dim=0).float()     # only works for batch_size=1

        encoded_motion = self.motion_encoder(batch['motion'])
        B, T = encoded_motion.shape[:2]
        encoded_motion = encoded_motion.reshape(B * T, -1)
        mean = self.mean(encoded_motion)
        logvar = self.logvar(encoded_motion)
        std = torch.exp(0.5 * logvar)

        # reparameterization trick
        distribution = torch.distributions.Normal(mean, std)
        latent_vector = self.sample_from_distribution(distribution)     # [B*T, latent_dim]
        latent_vector = latent_vector.reshape(B, T, -1)                 # [B, T, 256]

        # Decode
        motion_pred = self.motion_decoder(latent_vector)                # [B, T, 53]

        # GT data
        motion_ref = batch["motion"]
        # Standard normal distribution
        mean_ref = torch.zeros_like(distribution.loc)
        scale_ref = torch.ones_like(distribution.scale)
        distribution_ref = torch.distributions.Normal(mean_ref, scale_ref)

        # Compute the losses
        loss = self.losses[split].update(motion_pred=motion_pred,
                                         motion_ref=motion_ref,
                                         distribution_pred=distribution,
                                         distribution_ref=distribution_ref)
        if loss is None:
            raise ValueError("Loss is None, this happened with torchmetrics > 0.7")

        # Compute the metrics
        if split == "val":
            self.metrics.update(motion_pred.detach(),
                                motion_ref.detach(),
                                [motion_ref.shape[1]]*motion_ref.shape[0])

        # Log the losses
        self.allsplit_batch_end(split, batch_idx)

        if "total/train" in self.trainer.callback_metrics:
            loss_train = self.trainer.callback_metrics["total/train"].item()
            self.log("loss_train", loss_train, prog_bar=True, on_step=True, on_epoch=False)

        if "total/val" in self.trainer.callback_metrics:
            loss_val = self.trainer.callback_metrics["total/val"].item()
            self.log("loss_val", loss_val, prog_bar=True, on_step=True, on_epoch=False)

        return loss