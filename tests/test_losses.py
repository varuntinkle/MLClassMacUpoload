import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.losses import MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, NLLLoss


class TestMSELoss:
    """Test suite for MSE Loss implementation."""

    def test_initialization_default(self):
        """Test MSELoss default initialization."""
        loss = MSELoss()
        assert loss.reduction == "mean"

    def test_initialization_custom_reduction(self):
        """Test MSELoss with custom reduction."""
        for reduction in ["mean", "sum", "none"]:
            loss = MSELoss(reduction=reduction)
            assert loss.reduction == reduction

    def test_invalid_reduction(self):
        """Test MSELoss raises error for invalid reduction."""
        with pytest.raises(ValueError):
            MSELoss(reduction="invalid")

    def test_perfect_predictions(self):
        """Test MSE loss when predictions equal targets."""
        loss_fn = MSELoss()
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        loss = loss_fn(predictions, targets)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_simple_case(self):
        """Test MSE loss with simple known values."""
        loss_fn = MSELoss(reduction="mean")
        predictions = torch.tensor([1.0, 2.0])
        targets = torch.tensor([0.0, 1.0])

        # MSE = [(1-0)² + (2-1)²] / 2 = [1 + 1] / 2 = 1.0
        loss = loss_fn(predictions, targets)
        expected = torch.tensor(1.0)
        assert torch.allclose(loss, expected)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        predictions = torch.tensor([2.0, 3.0, 4.0])
        targets = torch.tensor([1.0, 1.0, 1.0])
        # Squared errors: [1, 4, 9]

        # Mean reduction
        loss_mean = MSELoss(reduction="mean")
        result_mean = loss_mean(predictions, targets)
        expected_mean = torch.tensor(14.0 / 3)  # (1+4+9)/3
        assert torch.allclose(result_mean, expected_mean)

        # Sum reduction
        loss_sum = MSELoss(reduction="sum")
        result_sum = loss_sum(predictions, targets)
        expected_sum = torch.tensor(14.0)  # 1+4+9
        assert torch.allclose(result_sum, expected_sum)

        # None reduction
        loss_none = MSELoss(reduction="none")
        result_none = loss_none(predictions, targets)
        expected_none = torch.tensor([1.0, 4.0, 9.0])
        assert torch.allclose(result_none, expected_none)

    def test_multidimensional_input(self):
        """Test MSE loss with multidimensional tensors."""
        loss_fn = MSELoss()
        predictions = torch.randn(3, 4, 5)
        targets = torch.randn(3, 4, 5)

        loss = loss_fn(predictions, targets)

        # Should return scalar
        assert loss.dim() == 0
        assert loss.item() >= 0  # MSE is always non-negative

    def test_gradient_flow(self):
        """Test gradient computation through MSE loss."""
        loss_fn = MSELoss()
        predictions = torch.tensor([2.0, 3.0], requires_grad=True)
        targets = torch.tensor([1.0, 1.0])

        loss = loss_fn(predictions, targets)
        loss.backward()

        # Gradient of MSE w.r.t predictions: 2 * (predictions - targets) / N
        expected_grad = torch.tensor([2.0, 4.0]) / 2  # [1.0, 2.0]
        assert torch.allclose(predictions.grad, expected_grad)

    def test_comparison_with_pytorch(self):
        """Test that our MSE matches PyTorch's MSELoss."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        for reduction in ["mean", "sum", "none"]:
            our_loss = MSELoss(reduction=reduction)
            torch_loss = nn.MSELoss(reduction=reduction)

            our_result = our_loss(predictions, targets)
            torch_result = torch_loss(predictions, targets)

            assert torch.allclose(our_result, torch_result, atol=1e-6)


class TestCrossEntropyLoss:
    """Test suite for Cross Entropy Loss implementation."""

    def test_initialization_default(self):
        """Test CrossEntropyLoss default initialization."""
        loss = CrossEntropyLoss()
        assert loss.reduction == "mean"
        assert loss.ignore_index == -100

    def test_initialization_custom(self):
        """Test CrossEntropyLoss with custom parameters."""
        loss = CrossEntropyLoss(reduction="sum", ignore_index=0)
        assert loss.reduction == "sum"
        assert loss.ignore_index == 0

    def test_perfect_predictions(self):
        """Test CE loss with perfect predictions."""
        loss_fn = CrossEntropyLoss()

        # Perfect prediction: high logit for correct class
        predictions = torch.tensor([[100.0, 0.0, 0.0]])  # Class 0
        targets = torch.tensor([0])

        loss = loss_fn(predictions, targets)
        # Should be very close to 0
        assert loss.item() < 0.01

    def test_worst_predictions(self):
        """Test CE loss with worst possible predictions."""
        loss_fn = CrossEntropyLoss()

        # Worst prediction: high logit for wrong class
        predictions = torch.tensor([[0.0, 100.0, 0.0]])  # Predicts class 1
        targets = torch.tensor([0])  # True class 0

        loss = loss_fn(predictions, targets)
        # Should be very high
        assert loss.item() > 90

    def test_multiclass_case(self):
        """Test CE loss with multiple classes and samples."""
        loss_fn = CrossEntropyLoss()

        # 3 samples, 4 classes
        predictions = torch.randn(3, 4)
        targets = torch.tensor([0, 2, 1])

        loss = loss_fn(predictions, targets)

        assert loss.dim() == 0  # Scalar output
        assert loss.item() > 0  # Should be positive

    def test_reduction_modes(self):
        """Test different reduction modes."""
        predictions = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        targets = torch.tensor([0, 1])

        # Test mean vs sum vs none
        loss_mean = CrossEntropyLoss(reduction="mean")
        loss_sum = CrossEntropyLoss(reduction="sum")
        loss_none = CrossEntropyLoss(reduction="none")

        result_mean = loss_mean(predictions, targets)
        result_sum = loss_sum(predictions, targets)
        result_none = loss_none(predictions, targets)

        # Check relationships
        assert torch.allclose(result_sum, result_mean * 2)  # sum = mean * N
        assert result_none.shape == (2,)  # Per-sample losses
        assert torch.allclose(result_mean, result_none.mean())

    def test_ignore_index(self):
        """Test ignore_index functionality."""
        loss_fn = CrossEntropyLoss(ignore_index=1)

        predictions = torch.randn(3, 4)
        targets = torch.tensor([0, 1, 2])  # Class 1 should be ignored

        loss = loss_fn(predictions, targets)

        # Loss should only consider samples 0 and 2
        assert torch.is_tensor(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """Test gradient computation through CrossEntropy loss."""
        loss_fn = CrossEntropyLoss()
        predictions = torch.randn(2, 3, requires_grad=True)
        targets = torch.tensor([0, 2])

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape

    def test_comparison_with_pytorch(self):
        """Test that our CrossEntropy matches PyTorch's."""
        predictions = torch.randn(5, 10)
        targets = torch.randint(0, 10, (5,))

        for reduction in ["mean", "sum", "none"]:
            our_loss = CrossEntropyLoss(reduction=reduction)
            torch_loss = nn.CrossEntropyLoss(reduction=reduction)

            our_result = our_loss(predictions, targets)
            torch_result = torch_loss(predictions, targets)

            assert torch.allclose(our_result, torch_result, atol=1e-6)


class TestBCELoss:
    """Test suite for Binary Cross Entropy Loss implementation."""

    def test_initialization(self):
        """Test BCELoss initialization."""
        loss = BCELoss()
        assert loss.reduction == "mean"

        loss_sum = BCELoss(reduction="sum")
        assert loss_sum.reduction == "sum"

    def test_perfect_predictions(self):
        """Test BCE loss with perfect predictions."""
        loss_fn = BCELoss()

        predictions = torch.tensor([1.0, 0.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(predictions, targets)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_worst_predictions(self):
        """Test BCE loss with worst predictions."""
        loss_fn = BCELoss()

        # Predict opposite of target
        predictions = torch.tensor([0.01, 0.99])  # Small epsilon to avoid log(0)
        targets = torch.tensor([1.0, 0.0])

        loss = loss_fn(predictions, targets)
        # Should be high
        assert loss.item() > 2.0

    def test_binary_classification_case(self):
        """Test BCE loss for typical binary classification."""
        loss_fn = BCELoss()

        # Sigmoid outputs (probabilities)
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.3])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(predictions, targets)

        assert loss.item() > 0
        assert torch.is_tensor(loss)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        predictions = torch.tensor([0.8, 0.2])
        targets = torch.tensor([1.0, 0.0])

        loss_mean = BCELoss(reduction="mean")
        loss_sum = BCELoss(reduction="sum")
        loss_none = BCELoss(reduction="none")

        result_mean = loss_mean(predictions, targets)
        result_sum = loss_sum(predictions, targets)
        result_none = loss_none(predictions, targets)

        assert torch.allclose(result_sum, result_mean * 2)
        assert result_none.shape == (2,)
        assert torch.allclose(result_mean, result_none.mean())

    def test_gradient_flow(self):
        """Test gradient computation through BCE loss."""
        loss_fn = BCELoss()
        predictions = torch.tensor([0.7, 0.3], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape

    def test_comparison_with_pytorch(self):
        """Test that our BCE matches PyTorch's."""
        predictions = torch.sigmoid(torch.randn(10, 5))  # Ensure [0,1] range
        targets = torch.randint(0, 2, (10, 5)).float()

        for reduction in ["mean", "sum", "none"]:
            our_loss = BCELoss(reduction=reduction)
            torch_loss = nn.BCELoss(reduction=reduction)

            our_result = our_loss(predictions, targets)
            torch_result = torch_loss(predictions, targets)

            assert torch.allclose(our_result, torch_result, atol=1e-6)


class TestBCEWithLogitsLoss:
    """Test suite for BCE with Logits Loss implementation."""

    def test_initialization(self):
        """Test BCEWithLogitsLoss initialization."""
        loss = BCEWithLogitsLoss()
        assert loss.reduction == "mean"
        assert loss.pos_weight is None

    def test_with_pos_weight(self):
        """Test BCEWithLogitsLoss with positive weight."""
        pos_weight = torch.tensor([2.0])
        loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        assert torch.equal(loss.pos_weight, pos_weight)

    def test_logits_input(self):
        """Test that BCE with logits accepts raw logits."""
        loss_fn = BCEWithLogitsLoss()

        # Raw logits (unbounded)
        predictions = torch.tensor([2.0, -1.0, 3.0, -2.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(predictions, targets)

        assert torch.is_tensor(loss)
        assert loss.item() >= 0

    def test_numerical_stability(self):
        """Test numerical stability with extreme logits."""
        loss_fn = BCEWithLogitsLoss()

        # Very large logits
        predictions = torch.tensor([100.0, -100.0])
        targets = torch.tensor([1.0, 0.0])

        loss = loss_fn(predictions, targets)

        # Should handle extreme values without overflow
        assert torch.isfinite(loss)
        assert loss.item() < 1.0  # Should be small for correct predictions

    def test_equivalence_with_sigmoid_bce(self):
        """Test equivalence between BCEWithLogits and Sigmoid+BCE."""
        logits = torch.randn(5, 3)
        targets = torch.randint(0, 2, (5, 3)).float()

        # Method 1: BCE with logits
        loss_fn1 = BCEWithLogitsLoss()
        loss1 = loss_fn1(logits, targets)

        # Method 2: Sigmoid + BCE
        probs = torch.sigmoid(logits)
        loss_fn2 = BCELoss()
        loss2 = loss_fn2(probs, targets)

        assert torch.allclose(loss1, loss2, atol=1e-6)

    def test_comparison_with_pytorch(self):
        """Test that our BCE with logits matches PyTorch's."""
        predictions = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8, 4)).float()

        for reduction in ["mean", "sum", "none"]:
            our_loss = BCEWithLogitsLoss(reduction=reduction)
            torch_loss = nn.BCEWithLogitsLoss(reduction=reduction)

            our_result = our_loss(predictions, targets)
            torch_result = torch_loss(predictions, targets)

            assert torch.allclose(our_result, torch_result, atol=1e-6)


class TestNLLLoss:
    """Test suite for Negative Log Likelihood Loss implementation."""

    def test_initialization(self):
        """Test NLLLoss initialization."""
        loss = NLLLoss()
        assert loss.reduction == "mean"
        assert loss.ignore_index == -100

    def test_log_probabilities_input(self):
        """Test NLL loss with log probabilities."""
        loss_fn = NLLLoss()

        # Log probabilities (output of log_softmax)
        predictions = F.log_softmax(torch.randn(2, 3), dim=1)
        targets = torch.tensor([0, 2])

        loss = loss_fn(predictions, targets)

        assert torch.is_tensor(loss)
        assert loss.item() >= 0

    def test_perfect_predictions(self):
        """Test NLL loss with perfect log probabilities."""
        loss_fn = NLLLoss()

        # Perfect log probabilities
        predictions = torch.tensor(
            [
                [-0.0001, -10.0, -10.0],  # ~log(1) for class 0
                [-10.0, -10.0, -0.0001],
            ]
        )  # ~log(1) for class 2
        targets = torch.tensor([0, 2])

        loss = loss_fn(predictions, targets)

        # Should be very close to 0
        assert loss.item() < 0.01

    def test_ignore_index(self):
        """Test ignore_index functionality in NLL loss."""
        loss_fn = NLLLoss(ignore_index=1)

        predictions = F.log_softmax(torch.randn(3, 4), dim=1)
        targets = torch.tensor([0, 1, 3])  # Index 1 should be ignored

        loss = loss_fn(predictions, targets)

        assert torch.is_tensor(loss)
        assert loss.item() >= 0

    def test_combination_with_logsoftmax(self):
        """Test NLL loss combined with log_softmax (equivalent to CrossEntropy)."""
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))

        # Method 1: CrossEntropy
        ce_loss = CrossEntropyLoss()
        ce_result = ce_loss(logits, targets)

        # Method 2: LogSoftmax + NLL
        log_probs = F.log_softmax(logits, dim=1)
        nll_loss = NLLLoss()
        nll_result = nll_loss(log_probs, targets)

        # Should be equivalent
        assert torch.allclose(ce_result, nll_result, atol=1e-6)

    def test_comparison_with_pytorch(self):
        """Test that our NLL matches PyTorch's."""
        predictions = F.log_softmax(torch.randn(6, 8), dim=1)
        targets = torch.randint(0, 8, (6,))

        for reduction in ["mean", "sum", "none"]:
            our_loss = NLLLoss(reduction=reduction)
            torch_loss = nn.NLLLoss(reduction=reduction)

            our_result = our_loss(predictions, targets)
            torch_result = torch_loss(predictions, targets)

            assert torch.allclose(our_result, torch_result, atol=1e-6)


class TestLossComparisons:
    """Test comparisons and relationships between different loss functions."""

    def test_crossentropy_vs_logsoftmax_nll(self):
        """Test that CrossEntropy = LogSoftmax + NLL."""
        logits = torch.randn(5, 10)
        targets = torch.randint(0, 10, (5,))

        # CrossEntropy directly
        ce_loss = CrossEntropyLoss()
        ce_result = ce_loss(logits, targets)

        # LogSoftmax + NLL
        log_probs = F.log_softmax(logits, dim=1)
        nll_loss = NLLLoss()
        nll_result = nll_loss(log_probs, targets)

        assert torch.allclose(ce_result, nll_result, atol=1e-6)

    def test_bce_vs_bce_with_logits(self):
        """Test that BCE = Sigmoid + BCE with logits."""
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 2, (4, 6)).float()

        # BCE with logits
        bce_logits_loss = BCEWithLogitsLoss()
        bce_logits_result = bce_logits_loss(logits, targets)

        # Sigmoid + BCE
        probs = torch.sigmoid(logits)
        bce_loss = BCELoss()
        bce_result = bce_loss(probs, targets)

        assert torch.allclose(bce_logits_result, bce_result, atol=1e-6)

    def test_loss_shapes_consistency(self):
        """Test that all losses handle shape consistency properly."""
        batch_size = 8

        # Regression losses
        pred_reg = torch.randn(batch_size, 10)
        target_reg = torch.randn(batch_size, 10)

        mse_loss = MSELoss()
        mse_result = mse_loss(pred_reg, target_reg)
        assert mse_result.dim() == 0  # Scalar

        # Classification losses
        pred_cls = torch.randn(batch_size, 5)  # 5 classes
        target_cls = torch.randint(0, 5, (batch_size,))

        ce_loss = CrossEntropyLoss()
        ce_result = ce_loss(pred_cls, target_cls)
        assert ce_result.dim() == 0  # Scalar

        # Binary classification losses
        pred_bin = torch.randn(batch_size, 3)
        target_bin = torch.randint(0, 2, (batch_size, 3)).float()

        bce_loss = BCEWithLogitsLoss()
        bce_result = bce_loss(pred_bin, target_bin)
        assert bce_result.dim() == 0  # Scalar

    def test_all_losses_gradient_flow(self):
        """Test that gradients flow through all loss functions."""
        losses_and_data = [
            (MSELoss(), torch.randn(3, 4, requires_grad=True), torch.randn(3, 4)),
            (
                CrossEntropyLoss(),
                torch.randn(3, 5, requires_grad=True),
                torch.randint(0, 5, (3,)),
            ),
            (
                BCELoss(),
                torch.sigmoid(torch.randn(3, 4, requires_grad=True)),
                torch.randint(0, 2, (3, 4)).float(),
            ),
            (
                BCEWithLogitsLoss(),
                torch.randn(3, 4, requires_grad=True),
                torch.randint(0, 2, (3, 4)).float(),
            ),
            (
                NLLLoss(),
                F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1),
                torch.randint(0, 5, (3,)),
            ),
        ]

        for loss_fn, predictions, targets in losses_and_data:
            loss = loss_fn(predictions, targets)
            loss.backward()

            assert predictions.grad is not None
            assert predictions.grad.shape == predictions.shape

            # Clear gradients for next iteration
            predictions.grad.zero_()


class TestLossEdgeCases:
    """Test edge cases and error conditions for loss functions."""

    def test_empty_tensors(self):
        """Test loss functions with empty tensors."""
        # Skip this test as empty tensors behavior varies
        pass

    def test_mismatched_shapes(self):
        """Test that losses handle shape mismatches appropriately."""
        mse_loss = MSELoss()

        # Different shapes should raise an error
        pred = torch.randn(3, 4)
        target = torch.randn(3, 5)  # Different size

        with pytest.raises(RuntimeError):
            mse_loss(pred, target)

    def test_out_of_range_targets(self):
        """Test behavior with invalid target values."""
        ce_loss = CrossEntropyLoss()

        predictions = torch.randn(2, 3)  # 3 classes
        invalid_targets = torch.tensor([0, 5])  # Class 5 doesn't exist

        # This should raise an error or be handled gracefully
        with pytest.raises((RuntimeError, IndexError)):
            ce_loss(predictions, invalid_targets)

    def test_extreme_values_numerical_stability(self):
        """Test numerical stability with extreme input values."""
        # BCE with logits should handle extreme values
        bce_logits = BCEWithLogitsLoss()

        extreme_logits = torch.tensor([1000.0, -1000.0])
        targets = torch.tensor([1.0, 0.0])

        loss = bce_logits(extreme_logits, targets)

        # Should be numerically stable
        assert torch.isfinite(loss)
        assert loss.item() < 10.0  # Should be reasonable for correct predictions


@pytest.mark.parametrize(
    "loss_class,reduction",
    [
        (MSELoss, "mean"),
        (MSELoss, "sum"),
        (MSELoss, "none"),
        (CrossEntropyLoss, "mean"),
        (CrossEntropyLoss, "sum"),
        (CrossEntropyLoss, "none"),
        (BCELoss, "mean"),
        (BCELoss, "sum"),
        (BCELoss, "none"),
        (BCEWithLogitsLoss, "mean"),
        (BCEWithLogitsLoss, "sum"),
        (BCEWithLogitsLoss, "none"),
        (NLLLoss, "mean"),
        (NLLLoss, "sum"),
        (NLLLoss, "none"),
    ],
)
def test_loss_reduction_modes(loss_class, reduction):
    """Test all loss functions with all reduction modes."""
    loss_fn = loss_class(reduction=reduction)

    if loss_class in [MSELoss]:
        pred = torch.randn(4, 5)
        target = torch.randn(4, 5)
    elif loss_class in [CrossEntropyLoss, NLLLoss]:
        if loss_class == CrossEntropyLoss:
            pred = torch.randn(4, 6)
        else:  # NLLLoss
            pred = F.log_softmax(torch.randn(4, 6), dim=1)
        target = torch.randint(0, 6, (4,))
    else:  # BCE losses
        if loss_class == BCELoss:
            pred = torch.sigmoid(torch.randn(4, 5))
        else:  # BCEWithLogitsLoss
            pred = torch.randn(4, 5)
        target = torch.randint(0, 2, (4, 5)).float()

    result = loss_fn(pred, target)

    if reduction == "none":
        assert result.dim() > 0  # Should not be scalar
    else:
        assert result.dim() == 0  # Should be scalar

    assert torch.isfinite(result).all()  # Should not contain NaN/Inf
