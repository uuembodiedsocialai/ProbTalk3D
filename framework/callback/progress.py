import logging
import psutil
import os
from pathlib import Path
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


class ProgressLogger(Callback):
    def __init__(self,
                 metric_monitor: dict,
                 precision: int = 4):
        # Metric to monitor
        self.metric_monitor = metric_monitor
        self.precision = precision

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, padding=False, **kwargs) -> None:
        metric_format = f"{{:.{self.precision}f}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding
        metrics_str = []
        metrics_dico = {}         # Tensorboard epoch loss checking
        losses_dict = trainer.callback_metrics

        for metric_name, dico_name in self.metric_monitor.items():
            if dico_name in losses_dict:
                metric = losses_dict[dico_name].item()
                metric = metric_format.format(metric)
                metrics_dico[metric_name] = metric    # Tensorboard epoch loss checking
                metric = f"\n{metric_name} {metric}"
                metrics_str.append(metric)

        if len(metrics_str) == 0:
            return

        # Save the losses by epoch
        os.makedirs(Path(pl_module.working_dir)/"loss_by_epoch", exist_ok=True)
        for key, value in metrics_dico.items():
            if "_train" in key:
                key = key.replace("_train", "")
                log_dir = Path(pl_module.working_dir) / f"loss_by_epoch/{key}/train"
            elif "_val" in key:
                key = key.replace("_val", "")
                log_dir = Path(pl_module.working_dir) / f"loss_by_epoch/{key}/val"
            else:
                log_dir = Path(pl_module.working_dir) / f"loss_by_epoch/{key}"

            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar(key, float(value), global_step=trainer.current_epoch)
            writer.close()

        memory = f"Memory {psutil.virtual_memory().percent}%"
        line = line + ": " + "   ".join(metrics_str) + "   " + memory
        logger.info(line)
 