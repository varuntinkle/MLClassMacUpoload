import torch
import torch.nn as nn
import torch.nn.functional as F


# SOLUTIONS (for instructor reference - remove before distributing):
class ReLUSolution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
        # Alternative implementations:
        # return torch.clamp(x, min=0)
        # return torch.maximum(x, torch.zeros_like(x))
        # return torch.where(x > 0, x, torch.zeros_like(x))


class SigmoidSolution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
        # Alternative implementation:
        # return 1.0 / (1.0 + torch.exp(-x))


class TanhSolution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
        # Alternative implementations:
        # exp_x = torch.exp(x)
        # exp_neg_x = torch.exp(-x)
        # return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        #
        # Or: return 2 * torch.sigmoid(2 * x) - 1


class LeakyReLUSolution(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)
        # Alternative implementation:
        # return torch.where(x >= 0, x, self.negative_slope * x)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"
