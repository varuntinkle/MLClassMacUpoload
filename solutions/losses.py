import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# SOLUTIONS (for instructor reference - remove before distributing):
class MSELossSolution(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets, reduction=self.reduction)


class CrossEntropyLossSolution(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            predictions,
            targets,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )


class BCELossSolution(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(predictions, targets, reduction=self.reduction)


class BCEWithLogitsLossSolution(nn.Module):
    def __init__(
        self, reduction: str = "mean", pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            predictions, targets, reduction=self.reduction, pos_weight=self.pos_weight
        )


class NLLLossSolution(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(
            predictions,
            targets,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )
