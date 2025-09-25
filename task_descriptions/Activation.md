# Activation Functions Implementation

## Overview

In this assignment, you will implement common activation functions used in neural networks. Unlike the Linear layer from Assignment 1, activation functions don't have learnable parameters - they simply apply mathematical transformations to their inputs element-wise.

## Why Activation Functions Matter

Without activation functions, neural networks would just be chains of linear transformations, which can always be reduced to a single linear transformation. Activation functions introduce **non-linearity**, enabling networks to learn complex patterns and approximate any continuous function.

## Tasks

You need to implement four activation functions in `layers/activations.py`:

### 1. ReLU (Rectified Linear Unit)

**Mathematical Definition:** `ReLU(x) = max(0, x)`

**Properties:**
- Most popular activation function in deep learning
- Computationally efficient  
- Helps with vanishing gradient problem
- Output range: [0, ∞)

**Your Task:** Implement the forward pass that zeros out negative values and keeps positive values unchanged.

### 2. Sigmoid

**Mathematical Definition:** `Sigmoid(x) = 1 / (1 + exp(-x))`

**Properties:**
- Classic activation function, historically important
- Smooth, differentiable everywhere
- Output range: (0, 1) - useful for probabilities
- Can suffer from vanishing gradients for large |x|

**Your Task:** Implement the sigmoid transformation, handling numerical stability for extreme values.

### 3. Tanh (Hyperbolic Tangent)

**Mathematical Definition:** `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

**Properties:**
- Zero-centered (unlike Sigmoid)
- Output range: (-1, 1)
- Antisymmetric: `tanh(-x) = -tanh(x)`
- Often works better than Sigmoid in practice

**Your Task:** Implement the hyperbolic tangent function.

### 4. Leaky ReLU

**Mathematical Definition:** `LeakyReLU(x) = max(negative_slope × x, x)`

**Properties:**
- Addresses "dying ReLU" problem
- Small gradient for negative values instead of zero
- Configurable negative slope (typically 0.01)
- Output range: (-∞, ∞)

**Your Task:** Implement Leaky ReLU with configurable negative slope parameter.

## Implementation Details

### Class Structure

Each activation function should:
- Inherit from `nn.Module`
- Have a simple `__init__()` method (no learnable parameters except for Leaky ReLU's slope)
- Implement `forward()` method that applies the transformation element-wise
- Preserve input tensor shape exactly

### Example Usage

```python
# Create activation functions
relu = ReLU()
sigmoid = Sigmoid()
tanh = Tanh()
leaky_relu = LeakyReLU(negative_slope=0.01)

# Apply to tensors
x = torch.randn(32, 128)  # Batch of 32, features of 128

y1 = relu(x)      # Same shape as x, negative values become 0
y2 = sigmoid(x)   # Same shape as x, values in range (0, 1)  
y3 = tanh(x)      # Same shape as x, values in range (-1, 1)
y4 = leaky_relu(x)  # Same shape as x, negative values scaled by 0.01
```


## Getting Started

1. **Open `layers/activations.py`**
2. **Implement each class one by one:**
   - Start with ReLU (simplest)
   - Then Sigmoid and Tanh
   - Finish with Leaky ReLU (has a parameter)
3. **Test incrementally:**
   ```bash
   # Test just ReLU
   pytest tests/test_activations.py::TestReLU -v
   
   # Test all activations
   pytest tests/test_activations.py -v
   ```

## Key Concepts to Remember

### Shape Preservation
```python
# Input and output shapes must match exactly
x = torch.randn(2, 3, 4, 5)
activation = ReLU()
y = activation(x)
assert y.shape == x.shape  # Must be True
```

### Element-wise Operations
```python
# Operations apply to each element independently
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
relu = ReLU()
y = relu(x)  # tensor([0.0, 0.0, 1.0, 2.0])
```

### No Learnable Parameters (except Leaky ReLU)
```python
relu = ReLU()
list(relu.parameters())  # Should be empty []

leaky_relu = LeakyReLU(negative_slope=0.1)
list(leaky_relu.parameters())  # Also empty [] (slope is not learnable)
```

### Gradients Flow Through
```python
x = torch.tensor([1.0, -1.0], requires_grad=True)
relu = ReLU()
y = relu(x)
loss = y.sum()
loss.backward()
print(x.grad)  # tensor([1.0, 0.0]) - gradients based on activation derivative
```

## Testing Your Implementation

The test suite covers:

- ✅ **Correctness:** Output values match mathematical definitions
- ✅ **Shape preservation:** Input and output shapes are identical  
- ✅ **Edge cases:** Zero inputs, very large/small inputs
- ✅ **Gradient flow:** Backpropagation works correctly
- ✅ **Equivalence:** Results match PyTorch's built-in functions
- ✅ **Properties:** Mathematical properties (antisymmetry, ranges, etc.)

Run tests with:
```bash
uv run pytest tests/test_activations.py -v
```

## Common Pitfalls

- **Shape mismatch:** Make sure output has exact same shape as input
- **In-place operations:** Avoid modifying input tensor directly (can break gradients)
- **Numerical instability:** Be careful with very large exponentials
- **Wrong parameter handling:** Remember most activations have no learnable parameters

## Combining with Linear Layers

Once complete, you can chain your layers together:

```python
from layers.linear import LinearLayer
from layers.activations import ReLU, Sigmoid

# Create a simple network
layer1 = LinearLayer(784, 128, bias=True)
relu = ReLU() 
layer2 = LinearLayer(128, 10, bias=True)
sigmoid = Sigmoid()

# Forward pass
x = torch.randn(32, 784)  # Batch of images
h1 = layer1(x)           # Linear transformation
h1_activated = relu(h1)   # Apply non-linearity
h2 = layer2(h1_activated) # Another linear transformation  
output = sigmoid(h2)      # Final activation
```

## Mathematical Background

### Derivatives (for understanding gradients):

- **ReLU:** `d/dx ReLU(x) = 1 if x > 0, else 0`
- **Sigmoid:** `d/dx Sigmoid(x) = Sigmoid(x) × (1 - Sigmoid(x))`  
- **Tanh:** `d/dx Tanh(x) = 1 - Tanh(x)²`
- **Leaky ReLU:** `d/dx LeakyReLU(x) = 1 if x > 0, else negative_slope`

### Activation Function Comparison

| Function | Range | Zero-Centered | Computationally Efficient | Vanishing Gradient Issue |
|----------|-------|---------------|---------------------------|-------------------------|
| ReLU | [0, ∞) | No | High | Dying ReLU problem |
| Sigmoid | (0, 1) | No | Medium | Yes, for large \|x\| |
| Tanh | (-1, 1) | Yes | Medium | Yes, for large \|x\| |
| Leaky ReLU | (-∞, ∞) | No | High | Reduced |

## Next Steps

After completing this assignment, you'll be ready to:
1. **Combine layers** to build multi-layer networks
2. **Implement loss functions** for training
3. **Create optimizers** for parameter updates
4. **Build complete training loops**

## Resources

- [Understanding Activation Functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [Mathematical Functions in PyTorch](https://pytorch.org/docs/stable/torch.html#math-operations)
- [Activation Functions Deep Dive](https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions)
- [Gradient Flow and Activation Functions](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
