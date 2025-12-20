#!/usr/bin/env python3

"""
Normalization layers for transformer models.

Filepath: ./model/normalization.py
Project: CPEN455-Project-2025W1
Description: Normalization techniques used to stabilize training and improve model performance.
"""

import torch
from torch import nn


class LlamaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is a simpler and more efficient alternative to LayerNorm that
    normalizes using only the root mean square (RMS) without centering (no mean subtraction).
    
    Formula: output = x / sqrt(mean(x^2) + eps) * weight
    
    This is equivalent to T5's LayerNorm and is used in LLaMA models for its
    computational efficiency while maintaining similar performance to standard LayerNorm.
    
    Args:
        hidden_size: Dimension of the input features
        eps: Small constant for numerical stability (default: 1e-6)
    
    Example:
        >>> norm = LlamaRMSNorm(hidden_size=768)
        >>> x = torch.randn(2, 10, 768)
        >>> output = norm(x)  # Normalized output with same shape
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        """
        Initialize RMSNorm layer.
        
        Args:
            hidden_size: Size of the input features to normalize
            eps: Small epsilon value to prevent division by zero
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Apply RMS normalization to input hidden states.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Normalized tensor with same shape as input
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        """String representation showing the layer's parameters."""
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"