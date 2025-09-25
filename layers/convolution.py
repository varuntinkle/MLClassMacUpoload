import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Helper function to convert int to tuple of two ints.
    Used for kernel_size, stride, padding parameters.
    """
    # TODO: You can try implementing this as well
    raise NotImplementedError


class Conv2D(nn.Module):
    """
    2D Convolutional Layer implementation.

    Applies a 2D convolution over an input tensor with shape (N, C_in, H, W) where:
    - N is batch size
    - C_in is number of input channels
    - H is input height
    - W is input width

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (number of filters)
        kernel_size (int or tuple): Size of convolving kernel
        stride (int or tuple): Stride of convolution. Default: 1
        padding (int or tuple): Zero-padding added to both sides of input. Default: 0
        bias (bool): If True, adds learnable bias. Default: False

    Shape:
        Input: (N, C_in, H_in, W_in)
        Output: (N, C_out, H_out, W_out) where
            H_out = floor((H_in + 2*padding[0] - kernel_size[0]) / stride[0]) + 1
            W_out = floor((W_in + 2*padding[1] - kernel_size[1]) / stride[1]) + 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = False,
    ):
        super().__init__()

        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Store all arguments as instance variables (in_channels, out_channels, etc.)
        # 2. Convert kernel_size, stride, padding to tuples if they're integers
        #    - Use _pair() helper function or handle manually
        #    - Example: kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int)
        # 3. Create weight parameter with shape (out_channels, in_channels, kernel_height, kernel_width)
        # 4. Initialize weights using Kaiming/He initialization (good for ReLU)
        # 5. Create bias parameter with shape (out_channels,) if bias=True
        # 6. Initialize bias to zeros if it exists
        ##############################
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Conv2D layer.

        Args:
            x: Input tensor of shape (N, C_in, H_in, W_in)

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out)
        """
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Implement it without the torch.nn.conv2d
        # 1. Pass: input, weight, bias, stride, padding
        # 3. Make sure to handle the case when bias=None
        # 4. Alternative: implement convolution manually using unfold() - much more complex!
        ##############################
        raise NotImplementedError

    def extra_repr(self) -> str:
        """
        Set the extra representation of the module.
        """
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None}"
        )


class MaxPool2D(nn.Module):
    """
    2D Max Pooling layer implementation.

    Applies a 2D max pooling over an input tensor. Reduces spatial dimensions
    by taking the maximum value in each pooling window.

    Args:
        kernel_size (int or tuple): Size of pooling window
        stride (int or tuple): Stride of pooling. If None, defaults to kernel_size
        padding (int or tuple): Zero-padding added to both sides. Default: 0

    Shape:
        Input: (N, C, H_in, W_in)
        Output: (N, C, H_out, W_out) where
            H_out = floor((H_in + 2*padding[0] - kernel_size[0]) / stride[0]) + 1
            W_out = floor((W_in + 2*padding[1] - kernel_size[1]) / stride[1]) + 1
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()

        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Store kernel_size, stride, padding as instance variables
        # 2. Convert to tuples if they're integers
        # 3. If stride is None, set it equal to kernel_size (common default)
        # 4. No learnable parameters needed for max pooling!
        ##############################
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MaxPool2D layer.

        Args:
            x: Input tensor of shape (N, C, H_in, W_in)

        Returns:
            Output tensor of shape (N, C, H_out, W_out)
        """
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Use torch.nn.functional.max_pool2d()
        # 2. Pass: input, kernel_size, stride, padding
        # 3. Much simpler than convolution!
        ##############################
        raise NotImplementedError

    def extra_repr(self) -> str:
        """
        Set the extra representation of the module.
        """
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )
