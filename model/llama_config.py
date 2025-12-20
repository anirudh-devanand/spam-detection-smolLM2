#!/usr/bin/env python3

"""
Configuration class for LLaMA models.

Filepath: ./model/llama_config.py
Project: CPEN455-Project-2025W1
Description: Configuration parameters for LLaMA transformer models.

Usage:
    uv run python -m model.llama_config
"""


class LlamaConfig:
    """
    Configuration class for LLaMA models.
    
    This class stores all hyperparameters and architectural choices for a LLaMA model.
    It provides a centralized way to configure model architecture, training parameters,
    and optimization settings.
    
    Architecture Parameters:
    -----------------------
    - vocab_size: Size of the vocabulary (number of unique tokens)
    - hidden_size: Dimension of hidden states (model dimension)
    - intermediate_size: Dimension of the MLP hidden layer (typically 4x hidden_size)
    - num_hidden_layers: Number of transformer decoder layers
    - num_attention_heads: Number of attention heads for queries
    - num_key_value_heads: Number of key-value heads (for Grouped Query Attention)
    
    Position Encoding:
    -----------------
    - max_position_embeddings: Maximum sequence length supported
    - rope_theta: Base for RoPE frequency computation (typically 10000.0)
    - rope_scaling: Optional scaling configuration for extending context length
    
    Normalization & Regularization:
    ------------------------------
    - rms_norm_eps: Epsilon for RMSNorm numerical stability
    - attention_dropout: Dropout rate for attention weights
    - hidden_act: Activation function ("silu", "gelu", "relu")
    
    Generation:
    ----------
    - use_cache: Whether to use key-value caching during generation
    - pad_token_id: ID for padding tokens
    - bos_token_id: ID for beginning-of-sequence token
    - eos_token_id: ID for end-of-sequence token
    
    Example:
        >>> # Small model for testing
        >>> config = LlamaConfig(
        ...     vocab_size=32000,
        ...     hidden_size=512,
        ...     num_hidden_layers=8,
        ...     num_attention_heads=8,
        ...     num_key_value_heads=4  # GQA with 2x compression
        ... )
        >>> 
        >>> # Large model configuration
        >>> config = LlamaConfig(
        ...     vocab_size=32000,
        ...     hidden_size=4096,
        ...     num_hidden_layers=32,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8  # GQA with 4x compression
        ... )
    """
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        _attn_implementation="eager",
        **kwargs,
    ):
        """
        Initialize LlamaConfig.
        
        Args:
            vocab_size: Number of tokens in vocabulary
            hidden_size: Dimension of hidden states
            intermediate_size: Dimension of MLP hidden layer
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of KV heads (None = same as attention heads)
            hidden_act: Activation function name
            max_position_embeddings: Maximum sequence length
            initializer_range: Standard deviation for weight initialization
            rms_norm_eps: Epsilon for RMSNorm
            use_cache: Enable KV caching for generation
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            tie_word_embeddings: Share input/output embeddings
            rope_theta: RoPE frequency base
            rope_scaling: RoPE scaling configuration
            attention_bias: Use bias in attention projections
            attention_dropout: Dropout rate for attention
            mlp_bias: Use bias in MLP projections
            head_dim: Dimension per attention head (None = hidden_size // num_heads)
            _attn_implementation: Attention implementation type
            **kwargs: Additional arguments (for extensibility)
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self._attn_implementation = _attn_implementation
        
        # Backward compatibility: if there is a 'type' field in rope_scaling, copy it to 'rope_type'
        if self.rope_scaling is not None and isinstance(self.rope_scaling, dict):
            if "type" in self.rope_scaling and "rope_type" not in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]