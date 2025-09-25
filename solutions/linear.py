import torch
import torch.nn as nn
import torch.nn.functional as F


# SOLUTION (for instructor reference - remove before distributing):
class LinearLayerSolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Create weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier/Glorot uniform initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
