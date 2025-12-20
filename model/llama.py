#!/usr/bin/env python3

"""
LLaMA model implementations.

Filepath: ./model/llama.py
Project: CPEN455-Project-2025W1
Description: Main LLaMA model classes for language modeling tasks.
"""

import torch
from torch import nn
from typing import Optional, Tuple
from .llama_config import LlamaConfig
from .layers import LlamaDecoderLayer
from .normalization import LlamaRMSNorm
from .positional_encoding import LlamaRotaryEmbedding
from .attention import create_causal_mask
from .cache import Cache, DynamicCache


class LlamaModel(nn.Module):
    """
    The LLaMA transformer model (encoder-decoder stack).
    
    This is the core transformer model that processes input tokens and produces
    contextualized hidden states. It consists of:
    
    1. **Token Embeddings**: Convert token IDs to dense vectors
    2. **Rotary Position Embeddings (RoPE)**: Encode positional information
    3. **Transformer Layers**: Stack of decoder layers for processing
    4. **Final Layer Norm**: Normalize outputs
    5. **Language Modeling Head**: Project normalized states to vocabulary logits
    
    Architecture Flow:
    -----------------
    
        Token IDs → Embeddings → [Layer 1 → Layer 2 → ... → Layer N] → Norm → LM Head → Logits
                                   ↑                                    ↑
                                   └── RoPE applied at each layer ──────┘
    
    The public ``forward`` method returns vocabulary logits along with an optional
    KV cache pointer for fast autoregressive decoding. Hidden states remain
    internal but can be exposed by extending the class if feature extraction is
    required.
    
    Key Features:
    ------------
    - **Causal Masking**: Ensures autoregressive property (tokens can't see future)
    - **KV Caching**: Stores attention key-values for efficient generation
    - **Gradient Checkpointing**: Trade computation for memory (not implemented yet)
    - **Flexible Input**: Supports both input_ids and input_embeds
    
    Args:
        config: Model configuration (LlamaConfig)
    
    Example:
        >>> config = LlamaConfig(vocab_size=32000, hidden_size=768, num_hidden_layers=12)
        >>> model = LlamaModel(config)
        >>> input_ids = torch.randint(0, 32000, (2, 10))
        >>> logits, cache = model(input_ids)
    """
    
    def __init__(self, config: LlamaConfig):
        """
        Initialize the LlamaModel.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings: map token IDs to dense vectors
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Stack of transformer decoder layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final layer normalization
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language modeling head projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Rotary position embedding module
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
        # Gradient checkpointing (for memory efficiency during training)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask for padding [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_values: Cached key-value pairs from previous forward passes
            inputs_embeds: Pre-computed embeddings [batch, seq_len, hidden_size]
                          (alternative to input_ids)
            cache_position: Position indices for caching
            use_cache: Whether to return updated cache
            **kwargs: Additional arguments

        Returns:
            Tuple of ``(logits, past_key_values)`` where ``logits`` has shape
            ``[batch, seq_len, vocab_size]`` and ``past_key_values`` contains
            the updated KV cache when ``use_cache`` is enabled.
        """
        # Ensure exactly one of input_ids or inputs_embeds is provided
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Get embeddings from input_ids if not provided
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Compute cache positions (indices of tokens in the full sequence)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Compute position IDs if not provided
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal attention mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Initialize hidden states with embeddings
        hidden_states = inputs_embeds
        
        # Compute rotary position embeddings (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Pass through all transformer layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs, past_key_values