from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric


def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)


class ComputeMetrics(Metric):
    def __init__(self, force_in_meter: bool = True,
                 dist_sync_on_step=False, **kwargs):
        super().__init__()

        self.force_in_meter = force_in_meter
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # L2 error
        self.add_state("l2", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.ml2_metrics = ["l2"]

        # Variance error
        self.add_state("var", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.mvar_metrics = ["var"]

        # All metric
        self.metrics = self.ml2_metrics + self.mvar_metrics


    def compute(self):
        # Compute average of mean_l2 (frame-wise)
        count = self.count
        ml2_metrics = {metric: getattr(self, metric) / count for metric in self.ml2_metrics}
        ml2_metrics["mean_l2"] = self.l2.mean() / count
        ml2_metrics.pop("l2")       # Remove arrays

        # Compute average of AVEs (sequence-wise)
        count_seq = self.count_seq
        mvar_metrics = {metric: getattr(self, metric) / count_seq for metric in self.mvar_metrics}
        mvar_metrics["mean_var"] = self.var.mean() / count_seq
        mvar_metrics.pop("var")    # Remove arrays
        
        return {**ml2_metrics, **mvar_metrics}

    def update(self, pred: Tensor, ref: Tensor, lengths: List[int]):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # transfer tensor to list
        pred = [pred[i, :, :] for i in range(pred.shape[0])]
        ref = [ref[i, :, :] for i in range(ref.shape[0])]

        for i in range(len(lengths)):
            self.l2 += l2_norm(pred[i], ref[i], dim=1).sum()

            sigma_motion_pred = variance(pred[i], lengths[i], dim=0)
            sigma_motion_ref = variance(ref[i], lengths[i], dim=0)
            self.var += l2_norm(sigma_motion_pred, sigma_motion_ref, dim=0)

