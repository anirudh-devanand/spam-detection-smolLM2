#!/usr/bin/env python3

"""
Attention mechanisms for transformer models.

Filepath: ./model/attention.py
Project: CPEN455-Project-2025W1
Description: Multi-headed attention with key-value caching and causal masking.
"""

import torch
from torch import nn
from typing import Optional, Callable
from .positional_encoding import apply_rotary_pos_emb
from .cache import Cache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match the number of query heads.
    
    This is used in Grouped Query Attention (GQA) where we have fewer key-value heads
    than query heads to reduce memory usage. Each key-value head is repeated to match
    the corresponding query heads.
    
    For example, if we have 32 query heads and 8 key-value heads, each KV head will
    be repeated 4 times (n_rep=4) to align with the query heads.
    
    Args:
        hidden_states: Key or value tensor [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each head
        
    Returns:
        Expanded tensor [batch, num_kv_heads * n_rep, seq_len, head_dim]
        
    Note:
        This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep)
        but implemented more efficiently.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    # Expand and reshape to repeat each head n_rep times
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def create_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Create a causal attention mask for autoregressive generation.
    
    A causal mask ensures that each token can only attend to itself and previous tokens,
    preventing information leakage from future tokens. This is essential for language
    modeling where we predict the next token based only on previous context.
    
    The mask uses -inf for positions that should not be attended to (future tokens),
    and 0 for allowed positions (past and current tokens).
    
    Args:
        config: Model configuration
        input_embeds: Input embeddings [batch, seq_len, hidden_size]
        attention_mask: Optional mask for padding tokens [batch, total_len]
        cache_position: Position indices for current tokens [seq_len] or [batch, seq_len]
        past_key_values: Cache with previously computed key-values
        position_ids: Position IDs [batch, seq_len]
        
    Returns:
        Causal mask [batch, 1, seq_len, total_len] where total_len includes cached tokens
        
    Example:
        For seq_len=3, the causal mask pattern is:
        [[0, -inf, -inf],
         [0,    0, -inf],
         [0,    0,    0]]
        
        Token 0 can only attend to itself
        Token 1 can attend to tokens 0 and 1
        Token 2 can attend to tokens 0, 1, and 2
    """
    batch_size, seq_length = input_embeds.shape[:2]
    dtype = input_embeds.dtype
    device = input_embeds.device
    
    # Calculate the total sequence length including past cached tokens
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    target_length = past_seen_tokens + seq_length
    
    # Create causal mask initialized with -inf (no attention allowed)
    causal_mask = torch.full(
        (batch_size, 1, seq_length, target_length),
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    
    # Make it causal: only attend to past and current positions
    # Create position indices for comparison
    position_ids_target = torch.arange(target_length, device=device).unsqueeze(0)  # [1, target_length]
    cache_pos = cache_position.unsqueeze(-1)  # [seq_length, 1] or [batch, seq_length, 1]
    
    # Expand cache_position to match batch dimension if needed
    if cache_pos.dim() == 2:
        cache_pos = cache_pos.unsqueeze(0).expand(batch_size, seq_length, target_length)
    elif cache_pos.dim() == 3:
        cache_pos = cache_pos.expand(batch_size, seq_length, target_length)
    
    # Create boolean mask: position_ids <= cache_pos means we can attend
    mask = position_ids_target <= cache_pos  # [batch, seq_length, target_length]
    mask = mask.unsqueeze(1)  # [batch, 1, seq_length, target_length]
    
    # Set allowed positions to 0, keep -inf for disallowed positions
    causal_mask = torch.where(mask, torch.tensor(0, dtype=dtype, device=device), causal_mask)
    
    # Apply padding attention mask if provided
    if attention_mask is not None:
        # attention_mask: [batch, target_length] where 0 means padding
        expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, target_length)
        # Set padding positions to -inf
        causal_mask = causal_mask.masked_fill(expanded_mask == 0, torch.finfo(dtype).min)
    
    return causal_mask


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Standard scaled dot-product attention implementation.
    
    This implements the core attention mechanism from "Attention Is All You Need":
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Steps:
    1. Compute attention scores: Q @ K^T
    2. Scale by sqrt(head_dim) to prevent softmax saturation
    3. Apply causal mask (set future positions to -inf)
    4. Apply softmax to get attention weights (probabilities)
    5. Apply dropout for regularization
    6. Multiply by values: weights @ V
    
    Args:
        module: The attention module (used to access num_key_value_groups)
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_len_k, head_dim]
        value: Value tensor [batch, num_kv_heads, seq_len_k, head_dim]
        attention_mask: Causal mask [batch, 1, seq_len, seq_len_k] or None
        scaling: Scaling factor (1/sqrt(head_dim))
        dropout: Dropout probability for attention weights
        **kwargs: Additional arguments (unused)
        
    Returns:
        Tuple of (attention_output, attention_weights):
            - attention_output: [batch, num_heads, seq_len, head_dim]
            - attention_weights: [batch, num_heads, seq_len, seq_len_k]
    """
    # Repeat key-value heads if using Grouped Query Attention
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Compute attention scores: Q @ K^T, scaled
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    # Apply causal mask if provided
    if attention_mask is not None:
        # Slice mask to match key length (for cached keys)
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask  # Adding -inf to masked positions

    # Apply softmax to get attention probabilities
    # Use float32 for numerical stability
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # Apply dropout
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    
    # Compute weighted sum of values
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """
    Multi-headed attention with Grouped Query Attention (GQA) support.
    
    This implements the attention mechanism from "Attention Is All You Need" with several
    enhancements for efficiency:
    
    1. **Grouped Query Attention (GQA)**: Reduces memory usage by sharing key-value heads
       across multiple query heads. For example, 32 query heads might share 8 key-value heads,
       reducing the KV cache size by 4x.
    
    2. **Rotary Position Embedding (RoPE)**: Encodes positional information by rotating
       query and key vectors before computing attention.
    
    3. **Key-Value Caching**: Stores previous key-value pairs to avoid recomputing them
       during autoregressive generation.
    
    Architecture:
    ------------
    Input -> [Q_proj, K_proj, V_proj] -> Apply RoPE -> Attention -> O_proj -> Output
    
    Where:
    - Q, K, V projections split input into multiple heads
    - RoPE encodes position information
    - Attention computes weighted combination of values
    - O projection combines heads back to hidden dimension
    
    Args:
        config: Model configuration with:
            - hidden_size: Model dimension
            - num_attention_heads: Number of query heads
            - num_key_value_heads: Number of key-value heads (for GQA)
            - head_dim: Dimension per attention head
            - attention_bias: Whether to use bias in projections
            - attention_dropout: Dropout rate for attention weights
        layer_idx: Index of this layer in the model
    
    Example:
        >>> config = LlamaConfig(hidden_size=768, num_attention_heads=12, num_key_value_heads=4)
        >>> attention = LlamaAttention(config, layer_idx=0)
        >>> hidden_states = torch.randn(2, 10, 768)
        >>> position_embeddings = rope_module(hidden_states, position_ids)
        >>> output, weights = attention(hidden_states, position_embeddings, None)
    """
    
    def __init__(self, config, layer_idx: int):
        """
        Initialize the attention module.
        
        Args:
            config: Model configuration
            layer_idx: Layer index (used for caching)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        
        # Grouped Query Attention: fewer KV heads than Q heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        
        # Attention scaling factor: 1/sqrt(head_dim)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) from RoPE [batch, seq_len, head_dim]
            attention_mask: Causal attention mask [batch, 1, seq_len, total_len]
            past_key_values: Cache for storing previous key-values
            cache_position: Position indices for current tokens
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (attention_output, attention_weights):
                - attention_output: [batch, seq_len, hidden_size]
                - attention_weights: [batch, num_heads, seq_len, total_len]
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project to queries, keys, values and reshape to [batch, num_heads, seq_len, head_dim]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply Rotary Position Embedding to queries and keys
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update cache with new key-value pairs
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Compute attention using the selected implementation
        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # Reshape back to [batch, seq_len, hidden_size] and apply output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights