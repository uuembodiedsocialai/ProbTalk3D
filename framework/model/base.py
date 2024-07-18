import numpy as np
import torch
from pytorch_lightning import LightningModule


class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)

        # Need to define:
        # forward
        # allsplit_step()
        # metrics()
        # losses()
        # optimizer
        # resumed_training

    # Calculates the number of trainable and non-trainable parameters
    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self.allsplit_step("val", batch, batch_idx)

    # this is called by trainer.test, not used in this project
    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self.allsplit_step("test", batch, batch_idx)

    def allsplit_batch_end(self, split: str, batch_idx: int):
        losses = self.losses[split]
        loss_dict = losses.compute()
        dico = {losses.loss_to_logname(loss, split): value.item()
                for loss, value in loss_dict.items()}

        if split == "val":
            metrics_dict = self.metrics.compute()
            dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})

        dico.update({"epoch": float(self.trainer.current_epoch),
                     "step": float(self.trainer.current_epoch)})
        self.log_dict(dico)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}

    def on_train_start(self):
        if self.resumed_training:
            # Reset the patience counter to 0 when resuming from a checkpoint
            # self.trainer.callbacks[2].wait_count = 0
            self.resumed_training = False

        for param_group in self.trainer.optimizers[0].param_groups:
            if param_group['lr'] != self.hparams.optim.lr:
                param_group['lr'] = self.hparams.optim.lr
                print("New learning rate:", self.hparams.optim.lr)

        print("Current learning rate:", self.optimizer.param_groups[0]['lr'])



