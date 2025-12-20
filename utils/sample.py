#!/usr/bin/env python3

"""
Autoregressive text generation utility for the procedural LLaMA causal LM.

This module now exposes :func:`sample`, a lightweight replacement for the
``generate`` method previously defined on ``LlamaForCausalLM``.
"""

from typing import Optional

import torch
from model.cache import Cache
from model.llama import LlamaModel


@torch.no_grad()
def sample(
    model: LlamaModel,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    **kwargs,
) -> torch.LongTensor:
    """Generate tokens autoregressively using the procedural causal LM."""
    generated = input_ids
    past_key_values: Optional[Cache] = None

    for _ in range(max_new_tokens):
        step_inputs = generated if past_key_values is None else generated[:, -1:]
        logits, past_key_values = model(
            input_ids=step_inputs,
            past_key_values=past_key_values,
            use_cache=True,
            **kwargs,
        )

        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature
        
        if top_k > 0:
            top_k_values, _ = torch.topk(next_token_logits, top_k)
            cutoff = top_k_values[..., -1, None]
            mask = next_token_logits < cutoff
            next_token_logits = next_token_logits.masked_fill(mask, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        config = model.config
        if getattr(config, "eos_token_id", None) is not None and (next_token == config.eos_token_id).all():
            break

    return generated
