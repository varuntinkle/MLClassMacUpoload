# Linear Layer
A Linear layer (also called Fully Connected or Dense layer) performs a linear transformation on input data:

$$ y = xW^T + b $$

Where:
- `x` is the input tensor of shape `(..., in_features)`
- `W` is the weight matrix of shape `(out_features, in_features)`
- `b` is the bias vector of shape `(out_features,)` (optional)
- `y` is the output tensor of shape `(..., out_features)`

### Your Implementation

You need to complete the `LinearLayer` class in [`layers/linear.py`](../layers/linear.py). The class skeleton is provided with two main methods to implement:

#### 1. `__init__(self, in_features: int, out_features: int, bias: bool = False)`

**Your tasks:**
- Store `in_features` and `out_features` as instance variables
- Create a weight parameter using `nn.Parameter` with shape `(out_features, in_features)`
- Initialize weights using Xavier/Glorot uniform initialization
- If `bias=True`, create a bias parameter with shape `(out_features,)` initialized to zeros
- If `bias=False`, ensure the bias parameter is `None`

**Hints:**
- Use `torch.empty()` to create uninitialized tensors
- Use `nn.init.xavier_uniform_()` for weight initialization
- Use `nn.init.zeros_()` for bias initialization
- Use `self.register_parameter('bias', None)` when bias is disabled

#### 2. `forward(self, x: torch.Tensor) -> torch.Tensor`

**Your tasks:**
- Implement the linear transformation: `y = xW^T + b`
- Handle both cases: with and without bias
- Ensure the output has the correct shape

**Hints:**
- Use `torch.matmul(x, self.weight.t())` or `x @ self.weight.t()` for matrix multiplication
- Alternatively, use `torch.nn.functional.linear(x, self.weight, self.bias)`
- Remember to add bias only if it exists (`self.bias is not None`)

### Expected Behavior

Your implementation should:

1. **Handle different input shapes:**
   ```python
   layer = LinearLayer(5, 3)
   
   # 1D input
   x1 = torch.randn(5)
   out1 = layer(x1)  # Shape: (3,)
   
   # 2D input (batched)
   x2 = torch.randn(10, 5)
   out2 = layer(x2)  # Shape: (10, 3)
   
   # 3D input
   x3 = torch.randn(2, 8, 5)
   out3 = layer(x3)  # Shape: (2, 8, 3)
   ```

2. **Work with and without bias:**
   ```python
   layer_no_bias = LinearLayer(4, 2, bias=False)
   layer_with_bias = LinearLayer(4, 2, bias=True)
   ```

3. **Support gradient computation:**
   ```python
   x = torch.randn(5, 4, requires_grad=True)
   output = layer(x)
   loss = output.sum()
   loss.backward()  # Should work without errors
   ```

4. **Match PyTorch's nn.Linear behavior** (when weights are identical)


## Common Pitfalls to Avoid

- **Weight shape confusion:** Remember weights should be `(out_features, in_features)`, not the other way around
- **Bias handling:** Make sure to check if bias exists before using it
- **Parameter registration:** Use `nn.Parameter()` to make tensors learnable
- **Initialization:** Don't forget to initialize your parameters properly

## Testing

Your implementation will be tested on:

- ✅ Correct parameter initialization and shapes
- ✅ Forward pass computation accuracy  
- ✅ Handling different input tensor shapes
- ✅ Gradient flow through the layer
- ✅ Equivalence with PyTorch's `nn.Linear`
- ✅ Edge cases and error handling

## Resources

- [PyTorch nn.Module Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch nn.Parameter Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)
- [PyTorch Initialization Functions](https://pytorch.org/docs/stable/nn.init.html)
- [Understanding nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)