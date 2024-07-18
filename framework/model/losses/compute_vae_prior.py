import torch
from hydra.utils import instantiate
from torch.nn import Module


class ComputeLosses(Module):
    def __init__(self, **kwargs):
        super().__init__()

        losses = ["kl", "recons_exp", "recons_jaw", "total"]
        self.losses_values = {}
        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))

        self.register_buffer("count", torch.tensor(0.0))
        self.losses = losses

        # Instantiate loss functions
        self._losses_func = {loss: instantiate(kwargs[loss + "_func"])
                             for loss in losses if loss != "total"}
        # Save the lambda parameters
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}

    def update(self, motion_pred=None, motion_ref=None, distribution_pred=None, distribution_ref=None):
        total: float = 0.0

        total += self._update_loss("kl", distribution_pred, distribution_ref)
        total += self._update_loss("recons_exp", motion_pred[:, :, :50], motion_ref[:, :, :50])
        total += self._update_loss("recons_jaw", motion_pred[:, :, 50:], motion_ref[:, :, 50:])

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total

    def compute(self):
        count = self.count
        loss_dict = {loss: getattr(self, loss) / count for loss in self.losses}
        return loss_dict

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss_to_logname(self, loss: str, split: str):
        if loss in ["total", "kl"]:
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
