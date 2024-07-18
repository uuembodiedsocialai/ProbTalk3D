import torch
from hydra.utils import instantiate
import logging

from framework.model.utils.vqvae_quantizer import VectorQuantizer
from framework.model.metrics.compute import ComputeMetrics
from framework.model.base import BaseModel

logger = logging.getLogger(__name__)


class VQVAE(BaseModel):
    def __init__(self,
                 # passed trough datamodule
                 nfeats: int,
                 resumed_training: bool,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.resumed_training = resumed_training
        self.working_dir = self.hparams.working_dir

        self.motion_encoder = instantiate(self.hparams.motion_encoder, nfeats=nfeats)
        logger.info(f"(1). Motion Encoder '{self.motion_encoder.hparams.name}' loaded")
        self.motion_decoder = instantiate(self.hparams.motion_decoder, nfeats=nfeats)
        logger.info(f"(2). Motion Decoder '{self.motion_decoder.hparams.name}' loaded")

        self.quantize = VectorQuantizer(self.hparams.n_embed, self.hparams.zquant_dim, beta=0.25)

        self.optimizer = instantiate(self.hparams.optim, params=self.parameters())
        self._losses = torch.nn.ModuleDict({split: instantiate(self.hparams.losses, _recursive_=False)
                                            for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.metrics = ComputeMetrics()

        self.__post_init__()

    def forward(self, batch):
        self.motion_encoder.to(self.device)
        self.quantize.to(self.device)
        self.motion_decoder.to(self.device)

        batch['motion'] = torch.cat(batch['motion'], dim=0).to(self.device)  # only works for batch_size=1
        encoder_motion = self.motion_encoder(batch['motion'])
        # get the closest embedding vector
        quant, _, _ = self.quantize(encoder_motion)
        motion_pred = self.motion_decoder(quant)
        return motion_pred

    def get_quant(self, x):
        encoder_features = self.motion_encoder(x)
        quant_z, _, info = self.quantize(encoder_features)
        indices = info[2]
        return quant_z, indices  # [B, T, 256]

    # Called during training
    def allsplit_step(self, split: str, batch, batch_idx):
        batch['motion'] = torch.cat(batch['motion'], dim=0).float()     # only works for batch_size=1

        encoder_motion = self.motion_encoder(batch['motion'])           # [B, T, 256]
        quant, emb_loss, _ = self.quantize(encoder_motion)
        motion_pred = self.motion_decoder(quant)                        # [B, T, 53]
        motion_ref = batch['motion']

        # Compute the losses
        loss = self.losses[split].update(quant_loss=emb_loss,
                                         motion_pred=motion_pred,
                                         motion_ref=motion_ref)
        if loss is None:
            raise ValueError("Loss is None, this happened with torchmetrics > 0.7")

        # Compute the metrics
        if split == "val":
            self.metrics.update(motion_pred.detach(),
                                motion_ref.detach(),
                                [motion_ref.shape[1]]*motion_ref.shape[0])  # list of lengths

        # Log the losses
        self.allsplit_batch_end(split, batch_idx)

        if "total/train" in self.trainer.callback_metrics:
            loss_train = self.trainer.callback_metrics["total/train"].item()
            self.log("loss_train", loss_train, prog_bar=True, on_step=True, on_epoch=False)

        if "total/val" in self.trainer.callback_metrics:
            loss_val = self.trainer.callback_metrics["total/val"].item()
            self.log("loss_val", loss_val, prog_bar=True, on_step=True, on_epoch=False)

        return loss

