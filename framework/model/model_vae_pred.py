import os
import torch
from torch import Tensor
from typing import List
from hydra.utils import instantiate
import logging

from framework.model.utils.tools import load_checkpoint, create_one_hot, resample_input
from framework.model.metrics.compute import ComputeMetrics
from framework.model.base import BaseModel
from framework.data.utils import get_split_keyids

logger = logging.getLogger(__name__)


class VaePredict(BaseModel):
    def __init__(self,
                 # passed trough datamodule
                 nfeats: int,
                 split_path: str,
                 one_hot_dim: List,
                 resumed_training: bool,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.nfeats = nfeats
        self.resumed_training = resumed_training
        self.working_dir = self.hparams.working_dir

        self.feature_extractor = instantiate(self.hparams.feature_extractor)
        logger.info(f"1. Audio feature extractor '{self.feature_extractor.hparams.name}' loaded")
        audio_encoded_dim = self.feature_extractor.audio_encoded_dim    # 768

        # Style one-hot embedding
        self.all_identity_list = get_split_keyids(path=split_path, split="train")
        self.all_identity_onehot = torch.eye(len(self.all_identity_list))

        # Load motion prior
        self.motion_prior = instantiate(self.hparams.motion_prior,
                                        nfeats=self.hparams.nfeats,
                                        logger_name="none",
                                        resumed_training=False,
                                        _recursive_=False)
        logger.info(f"2. '{self.motion_prior.hparams.modelname}' loaded")
        # load the motion prior in eval mode
        if os.path.exists(self.hparams.ckpt_path_prior):
            load_checkpoint(model=self.motion_prior,
                            ckpt_path=self.hparams.ckpt_path_prior,
                            eval_mode=True,
                            device=self.device)
        else:
            raise ValueError(f"Motion Autoencoder path not found: {self.hparams.ckpt_path_prior}")
        for param in self.motion_prior.parameters():
            param.requires_grad = False

        self.feature_predictor = instantiate(self.hparams.feature_predictor,
                                             audio_dim=audio_encoded_dim * 2,   # 768*2
                                             one_hot_dim=sum(one_hot_dim[:]))   # 32+8+3
        logger.info(f"3. 'Audio Encoder' loaded")

        self.optimizer = instantiate(self.hparams.optim, params=self.parameters())
        self._losses = torch.nn.ModuleDict({split: instantiate(self.hparams.losses, _recursive_=False)
                                            for split in ["losses_train", "losses_test", "losses_val"]})

        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.metrics = ComputeMetrics()

        self.__post_init__()

    # Forward: audio => motion, called during sampling
    def forward(self, batch, sample=None, generation=True) -> Tensor:
        self.feature_predictor.to(self.device)
        self.feature_extractor.to(self.device)
        self.motion_prior.to(self.device)

        # style embedding
        style_ont_hot = create_one_hot(keyids=batch["keyid"],
                                       IDs_list=self.all_identity_list,
                                       IDs_labels=self.all_identity_onehot,
                                       one_hot_dim=self.hparams.one_hot_dim)

        # audio feature extraction
        audio_feature = self.feature_extractor(batch['audio'], False)   # list of [B, Ts, 768]
        resample_audio_feature = []
        for idx in range(len(audio_feature)):
            if audio_feature[idx].shape[1] % 2 != 0:
                audio_feature_one = audio_feature[idx][:, :audio_feature[idx].shape[1] - 1, :]
            else:
                audio_feature_one = audio_feature[idx]

            # for evaluation, make sure the pred motion length equals gt (this may not be necessary)
            if not generation:
                # print("NOT GENERATION", audio_feature_one.shape, batch['motion'][0].shape)
                if audio_feature_one.shape[1] > batch['motion'][0].shape[1] * 2:
                    print("Shape checking:", audio_feature_one.shape, batch['motion'][0].shape)
                    audio_feature_one = audio_feature_one[:, :batch['motion'].shape[1] * 2, :]

            audio_feature_one = torch.reshape(audio_feature_one,
                                              (1, audio_feature_one.shape[1] // 2, audio_feature_one.shape[2] * 2))
            resample_audio_feature.append(audio_feature_one)
        assert len(resample_audio_feature) == 1, "Batch size > 1 not supported"

        # only works for batch_size=1
        batch['audio'] = torch.cat(resample_audio_feature, dim=0)
        prediction = self.feature_predictor(batch['audio'], style_ont_hot.to(self.device))
        motion_latent_pred = self.get_latent_no_encoder(prediction)
        motion_out = self.motion_prior.motion_decoder(motion_latent_pred)

        return motion_out

    # no motion encoder
    def get_latent_no_encoder(self, prediction):
        B, T = prediction.shape[:2]
        prediction = prediction.reshape(B * T, -1)
        mean = self.motion_prior.mean(prediction)
        logvar = self.motion_prior.logvar(prediction)
        std = torch.exp(0.5 * logvar)
        distribution = torch.distributions.Normal(mean, std)
        latent_vector = self.motion_prior.sample_from_distribution(distribution)
        latent_pred = latent_vector.reshape(B, T, -1)  # [B, T, 256]
        return latent_pred

    # Called during training
    def allsplit_step(self, split: str, batch, batch_idx):
        # extract audio features
        audio_feature = self.feature_extractor(batch['audio'], False)       # list of [B, Ts, 768]
        # Style embedding
        style_ont_hot = create_one_hot(keyids=batch["keyid"],
                                       IDs_list=self.all_identity_list,
                                       IDs_labels=self.all_identity_onehot,
                                       one_hot_dim=self.hparams.one_hot_dim)

        resample_audio_feature = []
        resample_motion_feature = []
        for idx in range(len(audio_feature)):
            resample_audio, resample_motion = resample_input(audio_feature[idx], batch['motion'][idx],
                                                             self.feature_extractor.hparams.output_framerate,
                                                             self.hparams.video_framerate)
            resample_audio_feature.append(resample_audio)
            resample_motion_feature.append(resample_motion)
        assert len(resample_audio_feature) == 1, "Batch size > 1 not supported"

        # only works for batch_size=1
        batch['audio'] = torch.cat(resample_audio_feature, dim=0)
        batch['motion'] = torch.cat(resample_motion_feature, dim=0)

        prediction = self.feature_predictor(batch['audio'], style_ont_hot.to(self.device))      # [B, T, 256]
        latent_pred = self.get_latent_no_encoder(prediction)
        motion_pred = self.motion_prior.motion_decoder(latent_pred)
        
        latent_ref, _ = self.motion_prior.get_latent_vector(batch['motion'])                    # [B, T, 53]
        motion_ref = batch['motion']
        assert motion_pred.shape == motion_ref.shape, "Dimension mismatch between prediction and reference motion."

        loss = self.losses[split].update(latent_pred=latent_pred, latent_ref=latent_ref,
                                         motion_pred=motion_pred, motion_ref=motion_ref,)

        # Compute the metrics
        if split == "val":
            self.metrics.update(motion_pred.detach(),
                                motion_ref.detach(),
                                [motion_ref.shape[1]] * motion_pred.shape[0])

        # Log the losses
        self.allsplit_batch_end(split, batch_idx)
        
        # Show loss on progress bar
        if "total/train" in self.trainer.callback_metrics:
            loss_train = self.trainer.callback_metrics["total/train"].item()
            self.log("loss_train", loss_train, prog_bar=True, on_step=True, on_epoch=False)

        if "total/val" in self.trainer.callback_metrics:
            loss_val = self.trainer.callback_metrics["total/val"].item()
            self.log("loss_val", loss_val, prog_bar=True, on_step=True, on_epoch=False)

        return loss


