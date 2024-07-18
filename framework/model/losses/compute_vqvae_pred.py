import torch
from hydra.utils import instantiate
from torch.nn import Module


class ComputeLosses(Module):
    def __init__(self,  **kwargs):
        super().__init__()

        losses = ["latent_manifold", "recons_exp", "recons_jaw", "total"]

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

    def update(self, motion_quant_pred=None, motion_quant_ref=None, motion_pred=None, motion_ref=None):
        total: float = 0.0
        total += self._update_loss("latent_manifold", outputs=motion_quant_pred, inputs=motion_quant_ref)
        total += self._update_loss("recons_exp", outputs=motion_pred[:, :, :50], inputs=motion_ref[:, :, :50])
        total += self._update_loss("recons_jaw", outputs=motion_pred[:, :, 50:], inputs=motion_ref[:, :, 50:])
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
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss_to_logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

