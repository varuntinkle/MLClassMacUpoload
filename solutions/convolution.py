import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_raw(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Raw implementation of 2D convolution

    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Kernel tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride for convolution (int or tuple)
        padding: Padding for input (int or tuple)

    Returns:
        Output tensor after convolution
    """
    # Handle stride and padding as tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    batch_size, in_channels, input_h, input_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Apply padding
    if padding[0] > 0 or padding[1] > 0:
        input_tensor = F.pad(
            input_tensor, (padding[1], padding[1], padding[0], padding[0])
        )
        input_h += 2 * padding[0]
        input_w += 2 * padding[1]

    # Calculate output dimensions
    output_h = (input_h - kernel_h) // stride[0] + 1
    output_w = (input_w - kernel_w) // stride[1] + 1

    # Initialize output tensor
    output = torch.zeros(
        batch_size,
        out_channels,
        output_h,
        output_w,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # Perform convolution
    for b in range(batch_size):
        for oc in range(out_channels):
            for oh in range(output_h):
                for ow in range(output_w):
                    # Calculate input region boundaries
                    ih_start = oh * stride[0]
                    ih_end = ih_start + kernel_h
                    iw_start = ow * stride[1]
                    iw_end = iw_start + kernel_w

                    # Extract input patch
                    input_patch = input_tensor[b, :, ih_start:ih_end, iw_start:iw_end]

                    # Compute convolution for this output position
                    conv_result = torch.sum(input_patch * weight[oc])

                    # Add bias if provided
                    if bias is not None:
                        conv_result += bias[oc]

                    output[b, oc, oh, ow] = conv_result

    return output


class Conv2DRaw(nn.Module):
    """
    Custom Conv2D layer using raw implementation
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(Conv2DRaw, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters properly
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return conv2d_raw(x, self.weight, self.bias, self.stride, self.padding)
