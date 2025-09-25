import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU(nn.Module):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Applies the function: ReLU(x) = max(0, x) element-wise.
    
    Shape:
        Input: (*) where * means any number of dimensions
        Output: (*) same shape as input
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU activation function.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor with same shape as input, with ReLU applied element-wise
        """
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. ReLU(x) = max(0, x)
        # 2. Do not use Torch's implementation
        ##############################
        raise NotImplementedError


class Sigmoid(nn.Module):
    """
    Sigmoid activation function.
    
    Applies the function: Sigmoid(x) = 1 / (1 + exp(-x)) element-wise.
    
    Shape:
        Input: (*) where * means any number of dimensions
        Output: (*) same shape as input, values in range (0, 1)
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Sigmoid activation function.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor with same shape as input, with Sigmoid applied element-wise
        """
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Sigmoid(x) = 1 / (1 + exp(-x))
        # 2. Do not use Torch's implementation
        ##############################
        raise NotImplementedError

class Tanh(nn.Module):
    """
    Hyperbolic Tangent (Tanh) activation function.
    
    Applies the function: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) element-wise.
    
    Shape:
        Input: (*) where * means any number of dimensions  
        Output: (*) same shape as input, values in range (-1, 1)
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Tanh activation function.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor with same shape as input, with Tanh applied element-wise
        """
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        # 2. Alternative formula: Tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        # 3. Or: Tanh(x) = 2 * Sigmoid(2x) - 1
        ##############################
        raise NotImplementedError

class LeakyReLU(nn.Module):
    """
    Leaky ReLU activation function.
    
    Applies the function: LeakyReLU(x) = max(negative_slope * x, x) element-wise.
    
    Args:
        negative_slope (float): Controls the angle of the negative slope. Default: 0.01
        
    Shape:
        Input: (*) where * means any number of dimensions
        Output: (*) same shape as input
    """
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Store negative_slope as an instance variable
        # 2. Make sure negative_slope is a reasonable value (typically small positive number)
        ##############################
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Leaky ReLU activation function.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor with same shape as input, with Leaky ReLU applied element-wise
        """
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. LeakyReLU(x) = max(negative_slope * x, x)
        # 2. For x >= 0: output = x
        # 3. For x < 0: output = negative_slope * x
        ##############################
        raise NotImplementedError
        
    def extra_repr(self) -> str:
        """
        Set the extra representation of the module.
        """
        return f'negative_slope={self.negative_slope}'