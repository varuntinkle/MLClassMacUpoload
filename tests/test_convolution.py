import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.convolution import Conv2D, MaxPool2D, _pair


class TestPairHelper:
    """Test suite for _pair helper function."""
    
    def test_int_input(self):
        """Test _pair with integer input."""
        assert _pair(3) == (3, 3)
        assert _pair(5) == (5, 5)
        assert _pair(1) == (1, 1)
    
    def test_tuple_input(self):
        """Test _pair with tuple input."""
        assert _pair((3, 5)) == (3, 5)
        assert _pair((1, 2)) == (1, 2)
        assert _pair((7, 7)) == (7, 7)


class TestConv2D:
    """Test suite for Conv2D layer."""
    
    def test_initialization_basic(self):
        """Test basic Conv2D initialization."""
        conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
        
        # Check attributes
        assert conv.in_channels == 3
        assert conv.out_channels == 16
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (1, 1)
        assert conv.padding == (0, 0)
        assert conv.bias is None
        
        # Check weight shape
        assert conv.weight.shape == (16, 3, 3, 3)
        assert isinstance(conv.weight, nn.Parameter)
    
    def test_initialization_with_bias(self):
        """Test Conv2D initialization with bias."""
        conv = Conv2D(in_channels=8, out_channels=32, kernel_size=5, bias=True)
        
        assert conv.bias is not None
        assert conv.bias.shape == (32,)
        assert isinstance(conv.bias, nn.Parameter)
    
    def test_initialization_with_tuples(self):
        """Test Conv2D initialization with tuple parameters."""
        conv = Conv2D(
            in_channels=16, 
            out_channels=64, 
            kernel_size=(3, 5),
            stride=(2, 1),
            padding=(1, 2)
        )
        
        assert conv.kernel_size == (3, 5)
        assert conv.stride == (2, 1)
        assert conv.padding == (1, 2)
        assert conv.weight.shape == (64, 16, 3, 5)
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32)  # batch=2, channels=3, 32x32 image
        
        output = conv(x)
        
        # With padding=1 and kernel=3, spatial dimensions should be preserved
        assert output.shape == (2, 16, 32, 32)
    
    def test_forward_output_size_calculation(self):
        """Test that output sizes are calculated correctly."""
        conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        x = torch.randn(1, 1, 8, 8)
        
        output = conv(x)
        
        # Expected: floor((8 + 2*1 - 3) / 2) + 1 = floor(7/2) + 1 = 3 + 1 = 4
        assert output.shape == (1, 1, 4, 4)
    
    def test_different_input_sizes(self):
        """Test conv with different input sizes."""
        conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        
        # Different spatial sizes
        for h, w in [(28, 28), (64, 64), (224, 224), (13, 17)]:
            x = torch.randn(1, 3, h, w)
            output = conv(x)
            assert output.shape == (1, 8, h, w)  # padding=1 preserves size
    
    def test_gradient_flow(self):
        """Test gradient flow through Conv2D."""
        conv = Conv2D(in_channels=2, out_channels=4, kernel_size=3, bias=True)
        x = torch.randn(1, 2, 10, 10, requires_grad=True)
        
        output = conv(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None
        assert x.grad is not None
        
        # Check gradient shapes
        assert conv.weight.grad.shape == conv.weight.shape
        assert conv.bias.grad.shape == conv.bias.shape
    
    def test_comparison_with_pytorch_conv2d(self):
        """Test that our Conv2D matches PyTorch's nn.Conv2d."""
        in_channels, out_channels = 6, 12
        kernel_size, stride, padding = 3, 2, 1
        
        # Create both layers
        our_conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        torch_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        
        # Copy weights to make them identical
        with torch.no_grad():
            torch_conv.weight.copy_(our_conv.weight)
            torch_conv.bias.copy_(our_conv.bias)
        
        # Test with same input
        x = torch.randn(2, in_channels, 16, 16)
        
        our_output = our_conv(x)
        torch_output = torch_conv(x)
        
        assert torch.allclose(our_output, torch_output, atol=1e-6)
    
    def test_parameter_count(self):
        """Test parameter count calculation."""
        # Without bias
        conv_no_bias = Conv2D(in_channels=8, out_channels=16, kernel_size=3, bias=False)
        params_no_bias = sum(p.numel() for p in conv_no_bias.parameters())
        expected_params = 16 * 8 * 3 * 3  # out_ch * in_ch * k_h * k_w
        assert params_no_bias == expected_params
        
        # With bias
        conv_with_bias = Conv2D(in_channels=8, out_channels=16, kernel_size=3, bias=True)
        params_with_bias = sum(p.numel() for p in conv_with_bias.parameters())
        expected_params_bias = 16 * 8 * 3 * 3 + 16  # weights + bias
        assert params_with_bias == expected_params_bias
    
    def test_1x1_convolution(self):
        """Test 1x1 convolution (pointwise convolution)."""
        conv = Conv2D(in_channels=64, out_channels=32, kernel_size=1)
        x = torch.randn(4, 64, 28, 28)
        
        output = conv(x)
        
        # 1x1 conv preserves spatial dimensions
        assert output.shape == (4, 32, 28, 28)
    
    def test_large_kernel(self):
        """Test with large kernel size."""
        conv = Conv2D(in_channels=3, out_channels=16, kernel_size=7, padding=3)
        x = torch.randn(1, 3, 32, 32)
        
        output = conv(x)
        
        # With padding=3 and kernel=7, spatial dims preserved
        assert output.shape == (1, 16, 32, 32)
    
    def test_extra_repr(self):
        """Test string representation."""
        conv = Conv2D(
            in_channels=16, 
            out_channels=32, 
            kernel_size=(3, 5), 
            stride=2, 
            padding=1, 
            bias=True
        )
        
        repr_str = conv.extra_repr()
        
        assert "in_channels=16" in repr_str
        assert "out_channels=32" in repr_str
        assert "kernel_size=(3, 5)" in repr_str
        assert "stride=(2, 2)" in repr_str
        assert "padding=(1, 1)" in repr_str
        assert "bias=True" in repr_str


class TestMaxPool2D:
    """Test suite for MaxPool2D layer."""
    
    def test_initialization_basic(self):
        """Test basic MaxPool2D initialization."""
        pool = MaxPool2D(kernel_size=2)
        
        assert pool.kernel_size == (2, 2)
        assert pool.stride == (2, 2)  # Should default to kernel_size
        assert pool.padding == (0, 0)
    
    def test_initialization_with_stride(self):
        """Test MaxPool2D initialization with custom stride."""
        pool = MaxPool2D(kernel_size=3, stride=1, padding=1)
        
        assert pool.kernel_size == (3, 3)
        assert pool.stride == (1, 1)
        assert pool.padding == (1, 1)
    
    def test_initialization_with_tuples(self):
        """Test MaxPool2D initialization with tuple parameters."""
        pool = MaxPool2D(kernel_size=(2, 4), stride=(1, 2), padding=(0, 1))
        
        assert pool.kernel_size == (2, 4)
        assert pool.stride == (1, 2)
        assert pool.padding == (0, 1)
    
    def test_forward_basic(self):
        """Test basic MaxPool2D forward pass."""
        pool = MaxPool2D(kernel_size=2, stride=2)
        x = torch.randn(1, 3, 8, 8)
        
        output = pool(x)
        
        # Should halve spatial dimensions
        assert output.shape == (1, 3, 4, 4)
    
    def test_forward_with_padding(self):
        """Test MaxPool2D with padding."""
        pool = MaxPool2D(kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 16, 10, 10)
        
        output = pool(x)
        
        # With padding=1, stride=1, kernel=3: output size = input size
        assert output.shape == (2, 16, 10, 10)
    
    def test_max_pooling_correctness(self):
        """Test that max pooling actually takes maximum values."""
        pool = MaxPool2D(kernel_size=2, stride=2)
        
        # Create simple input where we know the max values
        x = torch.tensor([[[[1.0, 2.0, 5.0, 6.0],
                            [3.0, 4.0, 7.0, 8.0],
                            [9.0, 10.0, 13.0, 14.0],
                            [11.0, 12.0, 15.0, 16.0]]]])
        
        output = pool(x)
        
        # Expected: max of each 2x2 block
        expected = torch.tensor([[[[4.0, 8.0],
                                   [12.0, 16.0]]]])
        
        assert torch.allclose(output, expected)
    
    def test_channels_preserved(self):
        """Test that number of channels is preserved."""
        pool = MaxPool2D(kernel_size=2)
        
        for channels in [1, 3, 64, 256]:
            x = torch.randn(1, channels, 16, 16)
            output = pool(x)
            assert output.shape[1] == channels  # Channels preserved
    
    def test_gradient_flow(self):
        """Test gradient flow through MaxPool2D."""
        pool = MaxPool2D(kernel_size=2)
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        
        output = pool(x)
        loss = output.sum()
        loss.backward()
        
        # Gradients should exist and have correct shape
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_comparison_with_pytorch_maxpool2d(self):
        """Test that our MaxPool2D matches PyTorch's nn.MaxPool2d."""
        kernel_size, stride, padding = 3, 2, 1
        
        our_pool = MaxPool2D(kernel_size, stride, padding)
        torch_pool = nn.MaxPool2d(kernel_size, stride, padding)
        
        x = torch.randn(2, 16, 32, 32)
        
        our_output = our_pool(x)
        torch_output = torch_pool(x)
        
        assert torch.allclose(our_output, torch_output)
    
    def test_no_learnable_parameters(self):
        """Test that MaxPool2D has no learnable parameters."""
        pool = MaxPool2D(kernel_size=3, stride=2, padding=1)
        
        param_count = sum(p.numel() for p in pool.parameters())
        assert param_count == 0
    
    def test_different_kernel_stride_combinations(self):
        """Test various kernel size and stride combinations."""
        test_cases = [
            (2, 2),  # Standard 2x2 pooling
            (3, 3),  # 3x3 pooling
            (2, 1),  # Overlapping pooling
            (4, 2),  # Large kernel, stride 2
        ]
        
        for kernel_size, stride in test_cases:
            pool = MaxPool2D(kernel_size=kernel_size, stride=stride)
            x = torch.randn(1, 8, 16, 16)
            
            output = pool(x)
            
            # Check that output shape is reasonable
            assert len(output.shape) == 4
            assert output.shape[0] == 1  # Batch preserved
            assert output.shape[1] == 8  # Channels preserved
            assert output.shape[2] > 0   # Valid height
            assert output.shape[3] > 0   # Valid width
    
    def test_extra_repr(self):
        """Test string representation."""
        pool = MaxPool2D(kernel_size=(2, 3), stride=1, padding=(0, 1))
        
        repr_str = pool.extra_repr()
        
        assert "kernel_size=(2, 3)" in repr_str
        assert "stride=(1, 1)" in repr_str
        assert "padding=(0, 1)" in repr_str


class TestConvMaxPoolCombination:
    """Test suite for combining Conv2D and MaxPool2D layers."""
    
    def test_conv_followed_by_maxpool(self):
        """Test typical CNN pattern: Conv -> MaxPool."""
        conv = Conv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        pool = MaxPool2D(kernel_size=2, stride=2)
        
        x = torch.randn(4, 3, 32, 32)  # Batch of 4 RGB images
        
        # Forward pass
        conv_out = conv(x)
        pool_out = pool(conv_out)
        
        assert conv_out.shape == (4, 32, 32, 32)  # Conv preserves spatial size
        assert pool_out.shape == (4, 32, 16, 16)  # MaxPool halves spatial size
    
    def test_multiple_conv_pool_blocks(self):
        """Test multiple Conv-Pool blocks (like in typical CNNs)."""
        # Block 1
        conv1 = Conv2D(3, 32, kernel_size=3, padding=1)
        pool1 = MaxPool2D(2)
        
        # Block 2  
        conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        pool2 = MaxPool2D(2)
        
        x = torch.randn(2, 3, 64, 64)
        
        # Forward through both blocks
        x = conv1(x)    # (2, 32, 64, 64)
        x = pool1(x)    # (2, 32, 32, 32)
        x = conv2(x)    # (2, 64, 32, 32)
        x = pool2(x)    # (2, 64, 16, 16)
        
        assert x.shape == (2, 64, 16, 16)
    
    def test_gradient_flow_through_combination(self):
        """Test gradient flow through Conv -> MaxPool combination."""
        conv = Conv2D(in_channels=8, out_channels=16, kernel_size=3, bias=True)
        pool = MaxPool2D(kernel_size=2)
        
        x = torch.randn(1, 8, 16, 16, requires_grad=True)
        
        # Forward pass
        conv_out = conv(x)
        pool_out = pool(conv_out)
        
        # Backward pass
        loss = pool_out.sum()
        loss.backward()
        
        # Check all gradients exist
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None
        assert x.grad is not None


@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,padding", [
    (1, 1, 3, 1, 0),
    (3, 16, 3, 1, 1),
    (16, 32, 5, 2, 2),
    (64, 128, 1, 1, 0),  # 1x1 conv
    (32, 64, (3, 5), (1, 2), (1, 2)),  # Asymmetric kernel
])
def test_conv2d_configurations(in_channels, out_channels, kernel_size, stride, padding):
    """Test Conv2D with various configurations."""
    conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    x = torch.randn(2, in_channels, 32, 32)
    
    output = conv(x)
    
    # Check that forward pass works and output has correct batch and channel dims
    assert output.shape[0] == 2  # Batch size preserved
    assert output.shape[1] == out_channels  # Correct output channels
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    assert conv.weight.grad is not None
    assert conv.bias.grad is not None


@pytest.mark.parametrize("kernel_size,stride,padding", [
    (2, 2, 0),
    (3, 3, 0),
    (2, 1, 0),  # Overlapping
    (3, 1, 1),  # With padding
    ((2, 3), (1, 2), (0, 1)),  # Asymmetric
])
def test_maxpool2d_configurations(kernel_size, stride, padding):
    """Test MaxPool2D with various configurations."""
    pool = MaxPool2D(kernel_size, stride, padding)
    x = torch.randn(2, 16, 32, 32)
    
    output = pool(x)
    
    # Check that forward pass works and preserves batch and channel dims
    assert output.shape[0] == 2   # Batch size preserved
    assert output.shape[1] == 16  # Channels preserved
    
    # Test gradient flow
    x_grad = torch.randn(2, 16, 32, 32, requires_grad=True)
    output_grad = pool(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None