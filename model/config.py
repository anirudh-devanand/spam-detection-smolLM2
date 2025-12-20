#!/usr/bin/env python3

"""
Base configuration loader for model configuration files.

Filepath: ./model/config.py
Project: CPEN455-Project-2025W1
Description: Generic configuration class for loading and managing model configuration.

Usage:
    uv run python -m model.config
"""

import json
from pathlib import Path

from utils.download import ensure_asset_exists

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Ensure derived attributes exist for downstream components
        if not hasattr(self, "head_dim") or self.head_dim is None:
            if getattr(self, "hidden_size", None) is None or getattr(self, "num_attention_heads", None) is None:
                raise ValueError("Config must define hidden_size and num_attention_heads to infer head_dim.")
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) not divisible by num_attention_heads ({self.num_attention_heads}); "
                    "cannot derive head_dim."
                )
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def _find_config_files(cls, base_path: Path) -> "Config":
        config_path = ensure_asset_exists(base_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            return cls(config_dict=json.load(handle))