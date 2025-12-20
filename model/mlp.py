#!/usr/bin/env python3

"""
Multi-Layer Perceptron (MLP) components for transformer models.

Filepath: ./model/mlp.py
Project: CPEN455-Project-2025W1
Description: Feed-forward network with gating mechanism used in each transformer layer.
"""

import torch
from torch import nn

ACT2FN = {
    "silu": nn.functional.silu,      # Sigmoid Linear Unit (Swish)
    "gelu": nn.functional.gelu,      # Gaussian Error Linear Unit
    "relu": nn.functional.relu,      # Rectified Linear Unit
    "swish": nn.functional.silu,     # Alias for SiLU
}


class LlamaMLP(nn.Module):
    """
    LLaMA's Multi-Layer Perceptron (Feed-Forward Network).
    
    This is the feed-forward component of each transformer layer, implementing the
    SwiGLU architecture (Gated Linear Unit with Swish activation).
    
    Architecture:
    ------------
    The MLP uses a gating mechanism inspired by GLU (Gated Linear Units):
    
        output = down_proj(act_fn(gate_proj(x)) * up_proj(x))
    
    Where:
    - gate_proj: Projects input to intermediate dimension, then applies activation
    - up_proj: Projects input to intermediate dimension (no activation)
    - Element-wise multiplication (*) acts as the gating mechanism
    - down_proj: Projects back to original hidden dimension
    
    This gating allows the network to control information flow, where gate_proj learns
    what information to "let through" and up_proj provides the actual content.
    
    The SwiGLU variant (using SiLU/Swish activation) has been shown to outperform
    standard FFN designs in large language models.
    
    Args:
        config: Model configuration with:
            - hidden_size: Input/output dimension
            - intermediate_size: Hidden dimension (typically 4x hidden_size or more)
            - hidden_act: Activation function name (typically "silu")
            - mlp_bias: Whether to use bias in linear layers (typically False)
    
    Shape:
        - Input: [batch, seq_len, hidden_size]
        - Output: [batch, seq_len, hidden_size]
        - Internal: [batch, seq_len, intermediate_size]
    
    Example:
        >>> config = LlamaConfig(hidden_size=768, intermediate_size=3072)
        >>> mlp = LlamaMLP(config)
        >>> x = torch.randn(2, 10, 768)
        >>> output = mlp(x)  # shape: [2, 10, 768]
    """
    
    def __init__(self, config):
        """
        Initialize the MLP module.
        
        Args:
            config: Configuration object with MLP parameters
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate projection: learns what to "gate" through
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # Up projection: provides the actual content/values
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # Down projection: projects back to hidden size
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
        # Activation function (typically SiLU/Swish for SwiGLU)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Implements: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Apply gating: activated gate controls what information flows through
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj