import torch
import os
from hydra.utils import instantiate
from pathlib import Path
from torch.nn import Module
from tensorboardX import SummaryWriter


class ComputeLosses(Module):
    def __init__(self,  **kwargs):
        super().__init__()

        # losses = ["recons", "vel", "total"]
        losses = ["recons", "total"]

        self.losses_values = {}
        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))

        self.register_buffer("count", torch.tensor(0.0))
        self.losses = losses

        # Instantiate loss functions
        self._losses_func = {loss: instantiate(kwargs[loss + "_func"])
                             for loss in losses if loss not in ["total"]}
        # Save the lambda parameters
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}

    def update(self, pred=None, ref=None):
        total: float = 0.0
        total += self._update_loss("recons", outputs=pred, inputs=ref)

        # pred_diff = latent_pred[:, 1:, :] - latent_pred[:, :-1, :]
        # ref_diff = latent_ref[:, 1:, :] - latent_ref[:, :-1, :]
        # total += self._update_loss("vel", outputs=pred_diff, inputs=ref_diff)

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total

    def compute(self):
        count = self.count
        loss_dict = {loss: getattr(self, loss) / count for loss in self.losses}
        return loss_dict

    def _update_loss(self, loss: str, *, outputs=None, inputs=None):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss_to_logname(self, loss: str, split: str):
        if loss in ["recons", "vel", "total"]:
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

