#!/usr/bin/env python3

"""
Transformer decoder layers.

Filepath: ./model/layers.py
Project: CPEN455-Project-2025W1
Description: Decoder layer combining self-attention and feed-forward networks.

Usage:
    uv run python -m model.layers
"""

import torch
from torch import nn
from typing import Optional
from .attention import LlamaAttention
from .mlp import LlamaMLP
from .normalization import LlamaRMSNorm
from .cache import Cache


class LlamaDecoderLayer(nn.Module):
    """
    A single transformer decoder layer.
    
    This is the fundamental building block of the LLaMA transformer model. Each layer
    consists of two main components connected via residual connections:
    
    1. **Self-Attention Block**: Allows tokens to attend to each other, capturing
       contextual relationships. Uses RMSNorm before attention.
       
    2. **Feed-Forward Block (MLP)**: Processes each token independently with a
       non-linear transformation. Uses RMSNorm before MLP.
    
    Architecture (Pre-Norm variant):
    --------------------------------
    
        Input
          ↓
        RMSNorm → Self-Attention ──┐
          ↓                         │ (residual)
          └─────────────────────────┤
                                    ↓
                                  Add
                                    ↓
        RMSNorm → MLP ──────────────┐
          ↓                         │ (residual)
          └─────────────────────────┤
                                    ↓
                                  Add
                                    ↓
                                 Output
    
    The Pre-Norm design (normalization before sub-layers) has been shown to provide
    more stable training compared to Post-Norm (normalization after sub-layers).
    
    Residual connections allow gradients to flow directly through the network,
    enabling training of very deep models (e.g., 80+ layers in large LLaMA models).
    
    Args:
        config: Model configuration with layer parameters:
            - hidden_size: Model dimension
            - num_attention_heads: Number of attention heads
            - intermediate_size: MLP hidden dimension
            - rms_norm_eps: Epsilon for RMSNorm stability
        layer_idx: Index of this layer in the model (used for caching)
    
    Shape:
        - Input: [batch, seq_len, hidden_size]
        - Output: [batch, seq_len, hidden_size]
    
    Example:
        >>> config = LlamaConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072)
        >>> layer = LlamaDecoderLayer(config, layer_idx=0)
        >>> hidden_states = torch.randn(2, 10, 768)
        >>> position_embeddings = rope_module(hidden_states, position_ids)
        >>> output = layer(hidden_states, position_embeddings=position_embeddings)
    """
    
    def __init__(self, config, layer_idx: int):
        """
        Initialize the decoder layer.
        
        Args:
            config: Model configuration
            layer_idx: Layer index in the model
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention mechanism
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # Feed-forward network (MLP)
        self.mlp = LlamaMLP(config)
        
        # Layer normalization (RMSNorm) for attention and MLP
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder layer.
        
        The layer applies:
        1. Residual connection with self-attention
        2. Residual connection with feed-forward network
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Causal attention mask [batch, 1, seq_len, total_len]
            position_ids: Position indices [batch, seq_len]
            past_key_values: Cache for key-value pairs from previous forward passes
            use_cache: Whether to return updated cache
            cache_position: Position indices for cache
            position_embeddings: Pre-computed RoPE embeddings (cos, sin)
            **kwargs: Additional arguments
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Save input for residual connection
        residual = hidden_states
        
        # Pre-norm: Normalize before attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-Attention with residual connection
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Add residual connection
        hidden_states = residual + hidden_states

        # Feed-Forward Network with residual connection
        residual = hidden_states
        # Pre-norm: Normalize before MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # Add residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states