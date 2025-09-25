import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.activations import ReLU, Sigmoid, Tanh, LeakyReLU


class TestReLU:
    """Test suite for ReLU activation function."""

    def test_positive_values(self):
        """Test ReLU with positive values (should remain unchanged)."""
        relu = ReLU()
        x = torch.tensor([1.0, 2.5, 10.0, 0.1])
        output = relu(x)

        expected = torch.tensor([1.0, 2.5, 10.0, 0.1])
        assert torch.allclose(output, expected)

    def test_negative_values(self):
        """Test ReLU with negative values (should become zero)."""
        relu = ReLU()
        x = torch.tensor([-1.0, -2.5, -10.0, -0.1])
        output = relu(x)

        expected = torch.zeros_like(x)
        assert torch.allclose(output, expected)

    def test_zero_values(self):
        """Test ReLU with zero values."""
        relu = ReLU()
        x = torch.tensor([0.0, 0.0, 0.0])
        output = relu(x)

        expected = torch.zeros_like(x)
        assert torch.allclose(output, expected)

    def test_mixed_values(self):
        """Test ReLU with mixed positive and negative values."""
        relu = ReLU()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = relu(x)

        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(output, expected)

    def test_different_shapes(self):
        """Test ReLU with different tensor shapes."""
        relu = ReLU()

        # 2D tensor
        x2d = torch.randn(3, 4)
        out2d = relu(x2d)
        assert out2d.shape == (3, 4)
        assert torch.all(out2d >= 0)  # All values should be non-negative

        # 3D tensor
        x3d = torch.randn(2, 3, 4)
        out3d = relu(x3d)
        assert out3d.shape == (2, 3, 4)
        assert torch.all(out3d >= 0)

    def test_gradient_flow(self):
        """Test gradient computation through ReLU."""
        relu = ReLU()
        x = torch.tensor([1.0, -1.0, 2.0, -2.0], requires_grad=True)

        output = relu(x)
        loss = output.sum()
        loss.backward()

        # Gradients should be 1 for positive inputs, 0 for negative inputs
        expected_grad = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert torch.allclose(x.grad, expected_grad)

    def test_comparison_with_pytorch(self):
        """Test that our ReLU matches PyTorch's F.relu."""
        our_relu = ReLU()
        x = torch.randn(10, 5)

        our_output = our_relu(x)
        torch_output = F.relu(x)

        assert torch.allclose(our_output, torch_output)


class TestSigmoid:
    """Test suite for Sigmoid activation function."""

    def test_zero_input(self):
        """Test Sigmoid with zero input (should be 0.5)."""
        sigmoid = Sigmoid()
        x = torch.tensor([0.0])
        output = sigmoid(x)

        expected = torch.tensor([0.5])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_large_positive_values(self):
        """Test Sigmoid with large positive values (should approach 1)."""
        sigmoid = Sigmoid()
        x = torch.tensor([10.0, 20.0, 100.0])
        output = sigmoid(x)

        # Should be very close to 1
        assert torch.all(output > 0.999)
        assert torch.all(output < 1.0)

    def test_large_negative_values(self):
        """Test Sigmoid with large negative values (should approach 0)."""
        sigmoid = Sigmoid()
        x = torch.tensor([-10.0, -20.0, -100.0])
        output = sigmoid(x)

        # Should be very close to 0
        assert torch.all(output < 0.001)
        assert torch.all(output > 0.0)

    def test_output_range(self):
        """Test that Sigmoid output is always in range (0, 1)."""
        sigmoid = Sigmoid()
        x = torch.randn(100) * 10  # Random values in large range
        output = sigmoid(x)

        assert torch.all(output > 0.0)
        assert torch.all(output < 1.0)

    def test_symmetry(self):
        """Test that Sigmoid is symmetric around 0.5."""
        sigmoid = Sigmoid()
        x = torch.tensor([1.0, -1.0, 2.0, -2.0])
        output = sigmoid(x)

        # sigmoid(-x) + sigmoid(x) should equal 1
        assert torch.allclose(output[1] + output[0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(output[3] + output[2], torch.tensor(1.0), atol=1e-6)

    def test_comparison_with_pytorch(self):
        """Test that our Sigmoid matches PyTorch's torch.sigmoid."""
        our_sigmoid = Sigmoid()
        x = torch.randn(10, 5)

        our_output = our_sigmoid(x)
        torch_output = torch.sigmoid(x)

        assert torch.allclose(our_output, torch_output, atol=1e-6)


class TestTanh:
    """Test suite for Tanh activation function."""

    def test_zero_input(self):
        """Test Tanh with zero input (should be 0)."""
        tanh = Tanh()
        x = torch.tensor([0.0])
        output = tanh(x)

        expected = torch.tensor([0.0])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_large_positive_values(self):
        """Test Tanh with large positive values (should approach 1)."""
        tanh = Tanh()
        x = torch.tensor([10.0, 20.0, 100.0])
        output = tanh(x)

        # Should be very close to 1
        assert torch.all(output > 0.999)
        assert torch.all(output < 1.0)

    def test_large_negative_values(self):
        """Test Tanh with large negative values (should approach -1)."""
        tanh = Tanh()
        x = torch.tensor([-10.0, -20.0, -100.0])
        output = tanh(x)

        # Should be very close to -1
        assert torch.all(output < -0.999)
        assert torch.all(output > -1.0)

    def test_output_range(self):
        """Test that Tanh output is always in range (-1, 1)."""
        tanh = Tanh()
        x = torch.randn(100) * 10  # Random values in large range
        output = tanh(x)

        assert torch.all(output > -1.0)
        assert torch.all(output < 1.0)

    def test_antisymmetry(self):
        """Test that Tanh is antisymmetric: tanh(-x) = -tanh(x)."""
        tanh = Tanh()
        x = torch.tensor([1.0, 2.0, 0.5, 3.0])

        pos_output = tanh(x)
        neg_output = tanh(-x)

        assert torch.allclose(neg_output, -pos_output, atol=1e-6)

    def test_comparison_with_pytorch(self):
        """Test that our Tanh matches PyTorch's torch.tanh."""
        our_tanh = Tanh()
        x = torch.randn(10, 5)

        our_output = our_tanh(x)
        torch_output = torch.tanh(x)

        assert torch.allclose(our_output, torch_output, atol=1e-6)


class TestLeakyReLU:
    """Test suite for Leaky ReLU activation function."""

    def test_default_negative_slope(self):
        """Test Leaky ReLU with default negative slope (0.01)."""
        leaky_relu = LeakyReLU()
        assert leaky_relu.negative_slope == 0.01

    def test_custom_negative_slope(self):
        """Test Leaky ReLU with custom negative slope."""
        leaky_relu = LeakyReLU(negative_slope=0.1)
        assert leaky_relu.negative_slope == 0.1

    def test_positive_values(self):
        """Test Leaky ReLU with positive values (should remain unchanged)."""
        leaky_relu = LeakyReLU()
        x = torch.tensor([1.0, 2.5, 10.0, 0.1])
        output = leaky_relu(x)

        expected = torch.tensor([1.0, 2.5, 10.0, 0.1])
        assert torch.allclose(output, expected)

    def test_negative_values(self):
        """Test Leaky ReLU with negative values."""
        negative_slope = 0.1
        leaky_relu = LeakyReLU(negative_slope=negative_slope)
        x = torch.tensor([-1.0, -2.0, -0.5])
        output = leaky_relu(x)

        expected = torch.tensor([-0.1, -0.2, -0.05])  # x * negative_slope
        assert torch.allclose(output, expected)

    def test_zero_values(self):
        """Test Leaky ReLU with zero values."""
        leaky_relu = LeakyReLU()
        x = torch.tensor([0.0, 0.0, 0.0])
        output = leaky_relu(x)

        expected = torch.zeros_like(x)
        assert torch.allclose(output, expected)

    def test_mixed_values(self):
        """Test Leaky ReLU with mixed positive and negative values."""
        negative_slope = 0.01
        leaky_relu = LeakyReLU(negative_slope=negative_slope)
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = leaky_relu(x)

        expected = torch.tensor([-0.02, -0.01, 0.0, 1.0, 2.0])
        assert torch.allclose(output, expected)

    def test_gradient_flow(self):
        """Test gradient computation through Leaky ReLU."""
        negative_slope = 0.1
        leaky_relu = LeakyReLU(negative_slope=negative_slope)
        x = torch.tensor([1.0, -1.0, 2.0, -2.0], requires_grad=True)

        output = leaky_relu(x)
        loss = output.sum()
        loss.backward()

        # Gradients should be 1 for positive, negative_slope for negative
        expected_grad = torch.tensor([1.0, 0.1, 1.0, 0.1])
        assert torch.allclose(x.grad, expected_grad)

    def test_comparison_with_pytorch(self):
        """Test that our Leaky ReLU matches PyTorch's F.leaky_relu."""
        negative_slope = 0.2
        our_leaky_relu = LeakyReLU(negative_slope=negative_slope)
        x = torch.randn(10, 5)

        our_output = our_leaky_relu(x)
        torch_output = F.leaky_relu(x, negative_slope=negative_slope)

        assert torch.allclose(our_output, torch_output)

    def test_extra_repr(self):
        """Test the string representation of Leaky ReLU."""
        leaky_relu = LeakyReLU(negative_slope=0.05)
        repr_str = leaky_relu.extra_repr()

        assert "negative_slope=0.05" in repr_str


class TestActivationComparisons:
    """Test comparisons between different activation functions."""

    @pytest.mark.parametrize(
        "activation_class,pytorch_func",
        [
            (ReLU, F.relu),
            (Sigmoid, torch.sigmoid),
            (Tanh, torch.tanh),
        ],
    )
    def test_activation_vs_pytorch(self, activation_class, pytorch_func):
        """Test that our activations match PyTorch equivalents."""
        our_activation = activation_class()
        x = torch.randn(20, 10)

        our_output = our_activation(x)
        torch_output = pytorch_func(x)

        assert torch.allclose(our_output, torch_output, atol=1e-6)

    def test_activation_shapes_preserved(self):
        """Test that all activations preserve input shapes."""
        activations = [ReLU(), Sigmoid(), Tanh(), LeakyReLU()]

        for activation in activations:
            # Test different shapes
            for shape in [(5,), (3, 4), (2, 3, 4), (1, 2, 3, 4)]:
                x = torch.randn(shape)
                output = activation(x)
                assert output.shape == shape

    def test_no_learnable_parameters(self):
        """Test that activation functions have no learnable parameters."""
        activations = [ReLU(), Sigmoid(), Tanh(), LeakyReLU()]

        for activation in activations:
            param_count = sum(p.numel() for p in activation.parameters())
            assert param_count == 0, (
                f"{type(activation).__name__} should have no parameters"
            )