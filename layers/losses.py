import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss implementation.

    Computes the mean squared error between predictions and targets:
    MSE = (1/N) * Σ(predictions - targets)²

    Commonly used for regression tasks.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'mean': the mean of the output
            'sum': the sum of the output
            'none': no reduction will be applied
            Default: 'mean'

    Shape:
        Input: (*) where * means any number of dimensions
        Target: (*) same shape as input
        Output: scalar if reduction is 'mean' or 'sum', otherwise same shape as input
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()

        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Store reduction as instance variable
        # 2. Validate that reduction is one of: 'mean', 'sum', 'none'
        # 3. Raise ValueError for invalid reduction types
        ##############################
        if reduction !='sum' or reduction !='mean' or reduction !='none':
            return ValueError
        self.reduction = reduction
        '''
        if reduction 
            return ValueError
        self.reduction


        '''

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predictions and targets.

        Args:
            predictions: Predicted values
            targets: True target values

        Returns:
            MSE loss according to specified reduction
        """
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Compute squared differences: (predictions - targets) ** 2
        # 2. Apply reduction: 'mean' -> .mean(), 'sum' -> .sum(), 'none' -> no reduction
        ##############################
        difference = (predictions - targets) ** 2
        if self.reduction == 'mean':
            return difference.mean()
        elif self.reduction == 'sum':
            return difference.sum()
        else:
            return difference

            
class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss implementation for classification.

    Combines LogSoftmax and NLLLoss in one single class:
    CE = -log(softmax(predictions)[target_class])

    For multi-class classification tasks.

    Args:
        reduction (str): Reduction to apply ('mean', 'sum', 'none'). Default: 'mean'
        ignore_index (int): Class index to ignore in loss computation. Default: -100

    Shape:
        Input: (N, C) where N is batch size, C is number of classes
        Target: (N,) containing class indices in range [0, C)
        Output: scalar if reduction is 'mean' or 'sum', otherwise (N,)
    """

    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()

        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Store reduction and ignore_index as instance variables
        # 2. Validate reduction parameter
        # 3. ignore_index is used to ignore certain target values (like padding tokens)
        ##############################
        if reduction != 'mean' or reduction !='sum' or reduction != 'none':
            return ValueError

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            predictions: Raw logits of shape (N, C)
            targets: Target class indices of shape (N,)

        Returns:
            Cross-entropy loss according to specified reduction
        """
        ##############################
        # YOUR CODE HERE
        #
        ##############################
        '''
        modified_target = torch.tensor(N, C).zeroes
()
        for row_index in range(N):
            row = modified_target[row_index]
            class_index = targets[row_index]
            if class_index != self.ignore_index:
                row[class_index] = -1

        loss = predictions @ modified_target.t()
        loss = predictions * modified_target
        if self.reduction == 'mean':
            return loss.mean()
        else if self.reduction == 'sum':
            reutrn loss.sum()
        return loss
        '''

        raise NotImplementedError


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss implementation.

    Used for binary classification tasks. Assumes predictions are probabilities (0-1).
    BCE = -[targets * log(predictions) + (1-targets) * log(1-predictions)]

    Args:
        reduction (str): Reduction to apply ('mean', 'sum', 'none'). Default: 'mean'

    Shape:
        Input: (*) where * means any number of dimensions
        Target: (*) same shape as input, values should be 0 or 1
        Output: scalar if reduction is 'mean' or 'sum', otherwise same shape as input
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()

        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Store reduction parameter
        # 2. Validate reduction parameter
        ##############################
        raise NotImplementedError

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.

        Args:
            predictions: Predicted probabilities (should be in range [0,1])
            targets: Binary target values (0 or 1)

        Returns:
            BCE loss according to specified reduction
        """
        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        ##############################
        raise NotImplementedError


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss implementation.

    Combines Sigmoid and BCELoss in one class for numerical stability.
    Accepts raw logits (unbounded) rather than probabilities.

    BCE_logits = -[targets * log(sigmoid(logits)) + (1-targets) * log(1-sigmoid(logits))]

    Args:
        reduction (str): Reduction to apply ('mean', 'sum', 'none'). Default: 'mean'
        pos_weight (Tensor, optional): Weight for positive examples. Default: None

    Shape:
        Input: (*) where * means any number of dimensions
        Target: (*) same shape as input, values should be 0 or 1
        Output: scalar if reduction is 'mean' or 'sum', otherwise same shape as input
    
    
        Apply sigmoid   1/1+e^{-x}  then take a log of that 
        then multiply it with class probabiltiy.

        
    
    """

    def __init__(
        self, reduction: str = "mean", pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()

        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Store reduction and pos_weight parameters
        # 2. pos_weight allows weighting positive examples more heavily
        # 3. Validate reduction parameter
        ##############################
        raise NotImplementedError

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy with logits loss.

        Args:
            predictions: Raw logits (unbounded values)
            targets: Binary target values (0 or 1)

        Returns:
            BCE with logits loss according to specified reduction
        """
        ##############################
        # YOUR CODE HERE
        #
        ##############################
        raise NotImplementedError


class NLLLoss(nn.Module):
    """
    Negative Log Likelihood Loss implementation.

    Used for classification when predictions are already log probabilities.
    Expects log-softmax outputs, not raw logits.

    NLL = -log_probabilities[target_class]

    Args:
        reduction (str): Reduction to apply ('mean', 'sum', 'none'). Default: 'mean'
        ignore_index (int): Class index to ignore. Default: -100

    Shape:
        Input: (N, C) where N is batch size, C is number of classes (log probabilities)
        Target: (N,) containing class indices in range [0, C)
        Output: scalar if reduction is 'mean' or 'sum', otherwise (N,)
    """

    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()

        ##############################
        # YOUR CODE HERE
        #
        # Hints:
        # 1. Store reduction and ignore_index parameters
        # 2. Validate reduction parameter
        # 3. This loss expects log probabilities as input, not raw logits
        ##############################
        raise NotImplementedError

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log likelihood loss.

        Args:
            predictions: Log probabilities of shape (N, C)
            targets: Target class indices of shape (N,)

        Returns:
            NLL loss according to specified reduction
        """
        ##############################
        # YOUR CODE HERE
        #
        ##############################
        raise NotImplementedError
