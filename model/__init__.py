#!/usr/bin/env python3

"""
Model package for LLaMA transformer implementation.

Filepath: ./model/__init__.py
Project: CPEN455-Project-2025W1
Description: Package initialization and API exports for the LLaMA model implementation.

This package provides a complete implementation of the LLaMA transformer model
organized into logical components.

Usage:
    from model import (
    LlamaConfig,
        LlamaModel,
        sample,
    )
"""

# Configuration
from .llama_config import LlamaConfig

# Cache
from .cache import Cache, DynamicCache

from torch.nn.functional import silu
from .normalization import LlamaRMSNorm
from .positional_encoding import (
    LlamaRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    ROPE_INIT_FUNCTIONS,
)

# Model building blocks
from .mlp import LlamaMLP
from .attention import LlamaAttention, repeat_kv, create_causal_mask, eager_attention_forward
from .layers import LlamaDecoderLayer

# Main models
from .llama import LlamaModel
