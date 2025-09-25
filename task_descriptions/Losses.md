# Loss Functions

## Overview

In this assignment, you will implement the essential loss functions used to train neural networks. Loss functions define the objective that models optimize during training - they measure how far predictions are from target values and provide gradients for backpropagation.

## Why Loss Functions Matter

Loss functions are the **optimization objectives** that guide learning:
- **Define what "good" means** for your model's predictions
- **Provide gradients** that flow backward through your network
- **Shape model behavior** (different losses encourage different prediction patterns)
- **Enable different task types** (regression, classification, ranking, etc.)

Without loss functions, there's no way to measure prediction quality or compute gradients for learning!

## Tasks

You need to implement five essential loss functions in `layers/losses.py`:

### 1. MSE Loss (Mean Squared Error)

**Use Case:** Regression tasks (predicting continuous values)

**Mathematical Definition:**
```
MSE = (1/N) × Σ(predictions - targets)²
```

**Properties:**
- Penalizes large errors heavily (quadratic penalty)
- Smooth and differentiable everywhere
- Scale-sensitive (units matter)
- Common for regression problems

**When to Use:** House price prediction, temperature forecasting, stock prices

### 2. Cross Entropy Loss

**Use Case:** Multi-class classification (predicting one of many classes)

**Mathematical Definition:**
```
CE = -log(softmax(predictions)[target_class])
```

**Properties:**
- Combines softmax activation + negative log likelihood
- Encourages confident, correct predictions
- Heavily penalizes confident wrong predictions
- Standard for multi-class classification

**When to Use:** Image classification, text classification, medical diagnosis

### 3. BCE Loss (Binary Cross Entropy)

**Use Case:** Binary classification (predicting 0 or 1)

**Mathematical Definition:**
```
BCE = -[targets×log(predictions) + (1-targets)×log(1-predictions)]
```

**Properties:**
- Expects probabilities as input (0-1 range)
- Symmetric penalty for both classes
- Can extend to multi-label classification

**When to Use:** Spam detection, medical tests, fraud detection

### 4. BCE with Logits Loss

**Use Case:** Binary classification with improved numerical stability

**Mathematical Definition:**
```
BCE_logits = BCE(sigmoid(logits), targets)
```

**Properties:**
- Accepts raw logits (unbounded values)
- More numerically stable than sigmoid + BCE
- Preferred over separate sigmoid + BCE
- Supports positive class weighting

**When to Use:** Same as BCE, but preferred for numerical reasons

### 5. NLL Loss (Negative Log Likelihood)

**Use Case:** Classification when you already have log probabilities

**Mathematical Definition:**
```
NLL = -log_probabilities[target_class]
```

**Properties:**
- Expects log probabilities as input
- Combined with log_softmax gives Cross Entropy
- Foundation for many other losses

**When to Use:** Building blocks for other losses, custom probability models

## Implementation Details

### Reduction Modes

All losses support three reduction modes:

```python
# 'mean': Average loss across all elements (default)
loss_fn = MSELoss(reduction='mean')
result = loss_fn(predictions, targets)  # Scalar

# 'sum': Sum loss across all elements  
loss_fn = MSELoss(reduction='sum')
result = loss_fn(predictions, targets)  # Scalar

# 'none': No reduction, return per-element losses
loss_fn = MSELoss(reduction='none')
result = loss_fn(predictions, targets)  # Same shape as input
```

### Common Parameters

**reduction:** Controls output aggregation
- `'mean'`: Average (default for most cases)
- `'sum'`: Total sum (useful for batch-size normalization)
- `'none'`: Per-element losses (useful for weighting)

**ignore_index:** (Classification losses) Skip certain target values
- Useful for padding tokens, unknown classes
- Default: -100

**pos_weight:** (BCE with logits) Weight positive examples more heavily
- Useful for imbalanced datasets
- Default: None (equal weighting)

## Getting Started

### 1. Understanding Loss Function Shapes

```python
# Regression: predictions and targets same shape
predictions = torch.randn(32, 10)  # Batch of 32, 10 outputs each
targets = torch.randn(32, 10)      # Same shape
mse_loss = MSELoss()
loss = mse_loss(predictions, targets)  # Scalar

# Multi-class classification: different shapes
predictions = torch.randn(32, 10)  # Batch of 32, 10 classes
targets = torch.randint(0, 10, (32,))  # Batch of 32 class indices
ce_loss = CrossEntropyLoss()
loss = ce_loss(predictions, targets)  # Scalar

# Binary classification: same shapes (usually)
predictions = torch.randn(32, 5)   # Raw logits
targets = torch.randint(0, 2, (32, 5)).float()  # Binary targets
bce_loss = BCEWithLogitsLoss()
loss = bce_loss(predictions, targets)  # Scalar
```

### 2. Implementation Pattern

Each loss function follows this pattern:

```python
class YourLoss(nn.Module):
    def __init__(self, reduction='mean', ...):
        super().__init__()
        # Validate and store parameters
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        # Your implementation
```


### 3. Testing Your Implementation

```bash
# Test individual losses
uv run pytest tests/test_losses.py::TestMSELoss -v
uv run pytest tests/test_losses.py::TestCrossEntropyLoss -v

# Test all losses
uv run pytest tests/test_losses.py -v
```

## Loss Function Relationships

Understanding how losses connect:

### Cross Entropy = Log Softmax + NLL
```python
# These are equivalent:
ce_loss = CrossEntropyLoss()
result1 = ce_loss(logits, targets)

log_probs = F.log_softmax(logits, dim=1)
nll_loss = NLLLoss()
result2 = nll_loss(log_probs, targets)

assert torch.allclose(result1, result2)
```

### BCE with Logits = Sigmoid + BCE
```python
# These are equivalent (but first is more stable):
bce_logits = BCEWithLogitsLoss()
result1 = bce_logits(logits, targets)

probs = torch.sigmoid(logits)
bce_loss = BCELoss()
result2 = bce_loss(probs, targets)

assert torch.allclose(result1, result2)
```

## Practical Usage Examples

### Building a Complete Training Example

Once you implement the losses, you can create full training pipelines:

```python
# Regression example
model = nn.Sequential(
    LinearLayer(784, 128, bias=True),
    ReLU(),
    LinearLayer(128, 1, bias=True)
)
loss_fn = MSELoss()

# Forward pass
predictions = model(x)
loss = loss_fn(predictions, targets)

# Backward pass
loss.backward()  # Gradients flow through your entire network!

# Classification example  
model = nn.Sequential(
    LinearLayer(784, 128, bias=True),
    ReLU(),
    LinearLayer(128, 10, bias=True)  # 10 classes
)
loss_fn = CrossEntropyLoss()

predictions = model(x)  # Raw logits
loss = loss_fn(predictions, targets)  # targets are class indices
loss.backward()
```

### Advanced Usage

```python
# Weighted loss for imbalanced data
pos_weight = torch.tensor([3.0])  # Weight positive class 3x
loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)

# Per-sample losses for custom weighting
loss_fn = MSELoss(reduction='none')
per_sample_losses = loss_fn(predictions, targets)
sample_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])
weighted_loss = (per_sample_losses * sample_weights).mean()

# Ignoring certain classes
loss_fn = CrossEntropyLoss(ignore_index=0)  # Ignore class 0 (padding)
```

## Common Pitfalls and Solutions

### 1. Input Range Issues
```python
# BCE expects probabilities [0,1]
predictions = torch.sigmoid(logits)  # Convert logits to probabilities
loss = BCELoss()(predictions, targets)

# OR use BCE with logits (preferred)
loss = BCEWithLogitsLoss()(logits, targets)  # More stable
```

### 2. Shape Mismatches
```python
# Cross Entropy expects:
predictions = torch.randn(batch_size, num_classes)  # (N, C)
targets = torch.randint(0, num_classes, (batch_size,))  # (N,)

# NOT:
targets = torch.randn(batch_size, num_classes)  # Wrong! This is for MSE
```

### 3. Numerical Stability
```python
# Bad: Manual sigmoid + BCE
probs = torch.sigmoid(very_large_logits)  # Can overflow
loss = BCELoss()(probs, targets)

# Good: BCE with logits
loss = BCEWithLogitsLoss()(very_large_logits, targets)  # Stable
```

## Testing Your Understanding

The test suite validates:

- ✅ **Correct mathematical computation** for all loss functions
- ✅ **Proper reduction handling** ('mean', 'sum', 'none')
- ✅ **Shape compatibility** with expected input/output formats
- ✅ **Gradient flow** through all loss functions
- ✅ **Equivalence with PyTorch** built-in implementations
- ✅ **Edge cases** (perfect predictions, extreme values)
- ✅ **Parameter validation** (invalid reduction modes, etc.)

## Mathematical Background

### Why These Specific Functions?

**MSE:** Assumes Gaussian noise, leads to maximum likelihood estimation
**Cross Entropy:** Derived from information theory, optimal for probability distributions
**BCE:** Special case of cross entropy for binary outcomes
**NLL:** Foundation of maximum likelihood estimation

### Gradient Properties

Each loss has different gradient characteristics:
- **MSE:** Linear gradients (constant rate)
- **Cross Entropy:** Exponential gradients (faster convergence)
- **BCE:** Sigmoid-shaped gradients (smooth)

## Debugging Tips

### Sanity Checks
```python
# Perfect predictions should give near-zero loss
perfect_pred = targets.clone()
loss = loss_fn(perfect_pred, targets)
assert loss.item() < 1e-6

# Loss should be positive (for most losses)
loss = loss_fn(random_pred, targets)
assert loss.item() >= 0

# Gradient check
pred = torch.randn(5, 3, requires_grad=True)
loss = loss_fn(pred, targets)
loss.backward()
assert pred.grad is not None
```

### Common Debug Steps
1. **Print shapes:** Ensure input/output shapes are correct
2. **Check ranges:** Verify inputs are in expected ranges
3. **Test gradients:** Ensure gradients flow properly
4. **Compare with PyTorch:** Validate against known implementations

## Resources

- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Cross Entropy Explained](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Loss Function Guide](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [Information Theory Background](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)