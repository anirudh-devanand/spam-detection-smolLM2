#!/usr/bin/env python3

"""
Cache implementations for efficient token generation.

Filepath: ./model/cache.py
Project: CPEN455-Project-2025W1
Description: Cache classes that store key-value pairs from attention layers during generation.

Usage:
    uv run python -m model.cache
"""

import torch
from typing import Optional


class Cache:
    """
    Base class for cache implementations.
    
    Cache stores key and value tensors from attention layers to enable efficient
    incremental decoding. During generation, instead of recomputing attention for
    all previous tokens, the cache allows us to only process new tokens.
    """
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key and value states.
        
        Args:
            key_states: New key tensor to add to cache
            value_states: New value tensor to add to cache
            layer_idx: Index of the transformer layer
            cache_kwargs: Optional additional arguments
            
        Returns:
            Tuple of (full_key_states, full_value_states) including cached and new states
        """
        raise NotImplementedError("Subclasses must implement the `update` method")
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Get the sequence length of cached states.
        
        Args:
            layer_idx: Index of the transformer layer
            
        Returns:
            Length of the cached sequence
        """
        raise NotImplementedError("Subclasses must implement the `get_seq_length` method")


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated.
    
    This cache stores key-value pairs for each layer and concatenates new states
    with existing cached states during generation. It automatically handles cache
    initialization and growth.
    
    Example:
        >>> cache = DynamicCache()
        >>> # During first forward pass
        >>> cached_keys, cached_values = cache.update(keys, values, layer_idx=0)
        >>> # During subsequent passes, cache grows automatically
        >>> cached_keys, cached_values = cache.update(new_keys, new_values, layer_idx=0)
        >>> print(cache.get_seq_length(0))  # Returns total sequence length
    """
    
    def __init__(self, config=None):
        """
        Initialize an empty dynamic cache.
        
        Args:
            config: Optional model configuration (currently unused but kept for compatibility)
        """
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.config = config
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key and value states, concatenating with existing cache.
        
        Args:
            key_states: New key tensor [batch, num_heads, seq_len, head_dim]
            value_states: New value tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: Index of the transformer layer
            cache_kwargs: Optional additional arguments (unused)
            
        Returns:
            Tuple of (full_keys, full_values) with shape [batch, num_heads, total_seq_len, head_dim]
        """
        # Ensure we have enough layers in the cache
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        
        if self.key_cache[layer_idx] is None:
            # First time caching for this layer
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate new states with cached states along sequence dimension
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Get the sequence length of cached states for a specific layer.
        
        Args:
            layer_idx: Index of the transformer layer
            
        Returns:
            Length of the cached sequence, or 0 if no cache exists for this layer
        """
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-2]
        return 0