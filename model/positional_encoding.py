#!/usr/bin/env python3

"""
Rotary Position Embedding (RoPE) for transformer models.

Filepath: ./model/positional_encoding.py
Project: CPEN455-Project-2025W1
Description: RoPE position encoding technique for transformer models.

RoPE is a position encoding technique that encodes absolute positions with rotation matrices
and naturally incorporates relative position information through rotation composition.
Unlike traditional sinusoidal embeddings that are added to token embeddings, RoPE is applied
directly to query and key projections in the attention mechanism.

Key advantages of RoPE:
1. Naturally encodes relative positions through rotation properties
2. No need for extra parameters (position embeddings)
3. Better extrapolation to longer sequences than seen during training
4. Maintains dot-product structure of attention (relative position only depends on rotation difference)

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)

Usage:
    uv run python -m model.positional_encoding
"""

import torch
from torch import nn
from typing import Optional
import math


def _compute_default_rope_parameters(
    config,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, float]:
    """
    Compute the inverse frequencies for RoPE embeddings.
    
    RoPE uses different rotation frequencies for different dimensions of the embedding.
    Lower dimensions rotate slower (capture long-range dependencies), while higher
    dimensions rotate faster (capture short-range patterns).
    
    The inverse frequency for dimension d is computed as:
        inv_freq[d] = 1 / (base^(2d/dim))
    
    where base is typically 10000 (rope_theta parameter).
    
    Args:
        config: Model configuration containing:
            - rope_theta: Base for computing frequencies (default: 10000)
            - head_dim: Dimension of attention heads
            - partial_rotary_factor: Fraction of dimensions to apply RoPE to (default: 1.0)
        device: Device to place tensors on
        
    Returns:
        Tuple of (inv_freq, attention_factor):
            - inv_freq: Inverse frequencies for each dimension [dim/2]
            - attention_factor: Scaling factor for attention (1.0 for default RoPE)
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = config.head_dim
    dim = int(head_dim * partial_rotary_factor)
    
    attention_factor = 1.0  # Unused in default RoPE
    
    # Compute the inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


# Registry of RoPE initialization functions (extensible for variants like NTK-aware RoPE)
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
}


class LlamaRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.
    
    This module generates cosine and sine embeddings based on position IDs, which are then
    used to rotate query and key vectors in the attention mechanism. The rotation encodes
    positional information while preserving the inner product structure needed for attention.
    
    The core idea is to represent position as a rotation angle, where positions that are
    far apart will have very different rotation angles. When computing attention, the dot
    product between rotated queries and keys naturally captures relative position information.
    
    Mathematical Details:
    ---------------------
    For position m and dimension d, RoPE computes rotation angle: θ_d * m
    where θ_d = 1 / (base^(2d/dim))
    
    The rotation is applied as:
        q_rotated = q * cos(θ*m) + rotate_half(q) * sin(θ*m)
        k_rotated = k * cos(θ*n) + rotate_half(k) * sin(θ*n)
    
    The attention score becomes:
        q_rotated @ k_rotated.T = q @ R(θ*(m-n)) @ k.T
    
    This shows attention only depends on relative position (m-n), not absolute positions.
    
    Attributes:
        rope_type: Type of RoPE initialization ("default" or custom variants)
        max_seq_len_cached: Maximum sequence length supported
        inv_freq: Inverse frequencies for each dimension
        attention_scaling: Scaling factor for attention scores
    """
    
    inv_freq: torch.Tensor  # Type annotation for linting
    
    def __init__(self, config, device=None):
        """
        Initialize the Rotary Position Embedding module.
        
        Args:
            config: Model configuration with RoPE parameters:
                - rope_scaling: Optional dict with 'rope_type' or 'type' field
                - max_position_embeddings: Maximum sequence length
                - rope_theta: Base for frequency computation (default: 10000)
                - head_dim: Dimension of attention heads
            device: Device to place tensors on
        """
        super().__init__()
        
        # Backward compatibility: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
            
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # Compute inverse frequencies and scaling factor
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Generate cosine and sine position embeddings.
        
        This method computes the rotation angles for each position and dimension,
        then returns the cosine and sine values that will be used to rotate
        query and key vectors in the attention mechanism.
        
        Args:
            x: Input tensor (used only for dtype and device) [batch, seq_len, ...]
            position_ids: Position indices for each token [batch, seq_len]
            
        Returns:
            Tuple of (cos, sin) embeddings, each with shape [batch, seq_len, head_dim]
                - cos: Cosine values for rotation
                - sin: Sine values for rotation
        """
        # Expand dimensions for broadcasting: inv_freq [dim/2] -> [batch, dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        # position_ids [batch, seq_len] -> [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Compute frequencies: [batch, dim/2, seq_len]
        # This performs the matrix multiplication: inv_freq @ position_ids
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32 for precision
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # Duplicate frequencies to match full head dimension: [batch, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # Compute cos and sin, apply attention scaling
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    Rotate half the hidden dimensions of the input.
    
    This is a key operation in RoPE. It swaps and negates the first and second halves
    of the feature dimension, effectively performing a 90-degree rotation in the complex plane.
    
    For a vector [a, b, c, d], this returns [-c, -d, a, b].
    
    When combined with the original vector via cos and sin multiplication, this creates
    the rotation matrix application:
        [cos(θ) -sin(θ)] [x1]   [x1*cos(θ) - x2*sin(θ)]
        [sin(θ)  cos(θ)] [x2] = [x1*sin(θ) + x2*cos(θ)]
    
    Args:
        x: Input tensor [..., dim]
        
    Returns:
        Rotated tensor with same shape, where first and second halves are swapped and negated
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply Rotary Position Embedding to query and key tensors.
    
    This function performs the core RoPE operation by rotating query and key vectors
    using the precomputed cosine and sine values. The rotation encodes positional
    information directly into the attention mechanism.
    
    The rotation formula is:
        rotated = original * cos + rotate_half(original) * sin
    
    This is equivalent to applying a 2D rotation matrix in complex space:
        [cos(θ) -sin(θ)]
        [sin(θ)  cos(θ)]
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine values [batch, seq_len, head_dim]
        sin: Sine values [batch, seq_len, head_dim]
        position_ids: Deprecated, kept for compatibility
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
            - 1 for shape [batch, num_heads, seq_len, head_dim]
            - 2 for shape [batch, seq_len, num_heads, head_dim]
    
    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs, now encoding position information
        
    Example:
        >>> cos, sin = rope_module(x, position_ids)
        >>> q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
        >>> # Now q_rotated and k_rotated can be used in attention with positional info
    """
    # Unsqueeze to match tensor dimensions for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply rotation: original * cos + rotate_half(original) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed