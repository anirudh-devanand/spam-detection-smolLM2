import torch
from torch import nn
from .llama import LlamaModel
from .cache import DynamicCache
from typing import Iterable, List

class PrefixLlamaModel(nn.Module):
    def __init__(self, base_model: LlamaModel, prefix_length: int) -> None:
        super().__init__()
        self.model = base_model
        self.prefix_length = prefix_length
        self.config = base_model.config
        
        self.num_layers = self.config.num_hidden_layers
        self.num_kv_heads = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        prefix_shape = (self.num_layers, self.num_kv_heads, prefix_length, self.head_dim)
        self.prefix_keys = nn.Parameter(torch.empty(prefix_shape))
        self.prefix_values = nn.Parameter(torch.empty(prefix_shape))

        nn.init.normal_(self.prefix_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.prefix_values, mean=0.0, std=0.02)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield self.prefix_keys
        yield self.prefix_values

    def convert_prefix_to_cache(self, batch_size: int, device: torch.device) -> DynamicCache:
        key_cache: List[torch.Tensor] = []
        value_cache: List[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            keys = self.prefix_keys[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
            values = self.prefix_values[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
            key_cache.append(keys.to(device))
            value_cache.append(values.to(device))

        cache = DynamicCache(config=self.model.config)
        cache.key_cache = key_cache
        cache.value_cache = value_cache
        return cache

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        past_key_values = kwargs.get("past_key_values")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("PrefixLlamaModel requires either input_ids or inputs_embeds.")

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        
        
        if past_key_values is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            
            past_key_values = self.convert_prefix_to_cache(
                batch_size=batch_size,
                device=device,
            )

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and self.prefix_length > 0:
            prefix_mask = torch.ones(
                (batch_size, self.prefix_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            kwargs["attention_mask"] = attention_mask
        
        self.model.eval()
        return self.model(*args, **kwargs, past_key_values=past_key_values)
        
