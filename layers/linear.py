import math

import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    """
    A custom implementation of a linear (fully connected) layer.

    This layer performs the operation: y = xW^T + b (if bias=True) or y = xW^T (if bias=False)
    where W is the weight matrix and b is the bias vector.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): If True, adds a learnable bias. Default: False

    Shape:
        Input: (*, in_features) where * means any number of dimensions
        Output: (*, out_features) where * is the same as input
    """


    '''

    Verify if setting self.weight 
    nn.Parameter(Tensort =(2,3)) just adding the deimsnsion is enogh.
    '''
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #Add it
        self.weight = nn.Parameter(Tensor=(out_features,
                                           in_features), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)


        if bias:
            self.bias = torch.tensor(1, out_features)
            nn.init.xavier_uniform_(self.bias)



        ########################
        # YOUR CODE HERE
        # Hints:
        # 1. Store in_features and out_features as instance variables
        # 2. Create weight parameter using nn.Parameter with shape (out_features, in_features)
        # 3. Initialize weights using Xavier/Glorot uniform initialization
        # 4. If bias=True, create bias parameter with shape (out_features,) and initialize to zeros
        # 5. Use torch.empty() to create uninitialized tensors, then fill them
        #
        # Remember: nn.Parameter automatically registers tensors as learnable parameters`t= - .......................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................... .....0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000...............................................................................p`

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        
        Varun
        use batched matrix multiplementation
        Verify ift here is a quick add function iun pytorc.  
       output =  x @ weight 
        if bias:
        output = output + bias
        """
        output =  x @ self.weight.t()
        if self.bias:
            output = output + self.bias

    def extra_repr(self) -> str:
        """
        Set the extra representation of the module.
        This is useful for debugging and model inspection.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
