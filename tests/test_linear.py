import pytest
import torch
import torch.nn as nn

# Implementations
from layers.linear import LinearLayer


class TestLinearLayer:
    """Test suite for LinearLayer implementation."""

    def test_initialization_without_bias(self):
        """Test layer initialization without bias."""
        layer = LinearLayer(10, 5, bias=False)

        # Check attributes
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.bias is None

        # Check weight shape
        assert layer.weight.shape == (5, 10)
        assert isinstance(layer.weight, nn.Parameter)

        # Check that weights are initialized (not all zeros)
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_initialization_with_bias(self):
        """Test layer initialization with bias."""
        layer = LinearLayer(8, 3, bias=True)

        # Check attributes
        assert layer.in_features == 8
        assert layer.out_features == 3
        assert layer.bias is not None

        # Check parameter shapes
        assert layer.weight.shape == (3, 8)
        assert layer.bias.shape == (3,)
        assert isinstance(layer.weight, nn.Parameter)
        assert isinstance(layer.bias, nn.Parameter)

    def test_forward_without_bias(self):
        """Test forward pass without bias."""
        layer = LinearLayer(4, 2, bias=False)
        x = torch.randn(3, 4)  # batch_size=3, in_features=4

        output = layer(x)

        # Check output shape
        assert output.shape == (3, 2)

        # Manual computation check
        expected = torch.matmul(x, layer.weight.t())
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_with_bias(self):
        """Test forward pass with bias."""
        layer = LinearLayer(4, 2, bias=True)
        x = torch.randn(3, 4)

        output = layer(x)

        # Check output shape
        assert output.shape == (3, 2)

        # Manual computation check
        expected = torch.matmul(x, layer.weight.t()) + layer.bias
        assert torch.allclose(output, expected, atol=1e-6)

    def test_different_input_shapes(self):
        """Test layer with different input tensor shapes."""
        layer = LinearLayer(5, 3, bias=True)

        # 1D input
        x1d = torch.randn(5)
        out1d = layer(x1d)
        assert out1d.shape == (3,)

        # 2D input (batch)
        x2d = torch.randn(10, 5)
        out2d = layer(x2d)
        assert out2d.shape == (10, 3)

        # 3D input
        x3d = torch.randn(2, 8, 5)
        out3d = layer(x3d)
        assert out3d.shape == (2, 8, 3)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer properly."""
        layer = LinearLayer(3, 2, bias=True)
        x = torch.randn(5, 3, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad is not None

        # Check gradient shapes
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape

    def test_comparison_with_pytorch_linear(self):
        """Test that our implementation matches PyTorch's nn.Linear."""
        in_features, out_features = 6, 4

        # Create both layers
        our_layer = LinearLayer(in_features, out_features, bias=True)
        torch_layer = nn.Linear(in_features, out_features, bias=True)

        # Copy weights to make them identical
        with torch.no_grad():
            torch_layer.weight.copy_(our_layer.weight)
            torch_layer.bias.copy_(our_layer.bias)

        # Test with same input
        x = torch.randn(10, in_features)

        our_output = our_layer(x)
        torch_output = torch_layer(x)

        assert torch.allclose(our_output, torch_output, atol=1e-6)

    def test_parameter_count(self):
        """Test that the layer has the correct number of parameters."""
        # Without bias
        layer_no_bias = LinearLayer(10, 5, bias=False)
        params_no_bias = sum(p.numel() for p in layer_no_bias.parameters())
        assert params_no_bias == 10 * 5  # only weights

        # With bias
        layer_with_bias = LinearLayer(10, 5, bias=True)
        params_with_bias = sum(p.numel() for p in layer_with_bias.parameters())
        assert params_with_bias == 10 * 5 + 5  # weights + bias

    def test_extra_repr(self):
        """Test the string representation of the layer."""
        layer_no_bias = LinearLayer(8, 4, bias=False)
        layer_with_bias = LinearLayer(8, 4, bias=True)

        repr_no_bias = layer_no_bias.extra_repr()
        repr_with_bias = layer_with_bias.extra_repr()

        assert "in_features=8" in repr_no_bias
        assert "out_features=4" in repr_no_bias
        assert "bias=False" in repr_no_bias

        assert "in_features=8" in repr_with_bias
        assert "out_features=4" in repr_with_bias
        assert "bias=True" in repr_with_bias

    def test_wrong_input_dimensions(self):
        """Test error handling for incorrect input dimensions."""
        layer = LinearLayer(5, 3)

        # Wrong number of features
        x_wrong = torch.randn(10, 7)  # should be 5 features, not 7

        with pytest.raises(RuntimeError):
            layer(x_wrong)

    @pytest.mark.parametrize(
        "in_features,out_features,bias",
        [
            (1, 1, False),
            (1, 1, True),
            (100, 1, False),
            (1, 100, True),
            (784, 128, False),
            (128, 10, True),
        ],
    )
    def test_various_configurations(self, in_features, out_features, bias):
        """Test layer with various feature configurations."""
        layer = LinearLayer(in_features, out_features, bias=bias)
        x = torch.randn(32, in_features)  # batch_size=32

        output = layer(x)
        assert output.shape == (32, out_features)

        # Test gradient flow
        loss = output.sum()
        loss.backward()

        assert layer.weight.grad is not None
        if bias:
            assert layer.bias.grad is not None
