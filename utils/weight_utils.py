#!/usr/bin/env python3

"""
Utility functions for downloading and loading model weights.

Filepath: ./utils/weight_utils.py
Project: CPEN455-Project-2025W1
Description: Functions for managing model weight downloads, caching, and loading.

Usage:
    uv run python -m utils.weight_utils
"""

import os
import torch
from typing import Optional, Dict

from .download import download_file_from_hub, get_model_cache_path


def load_cached_weights(checkpoint: str, cache_dir: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Try to load model weights from local cache.
    
    Args:
        checkpoint: Model checkpoint name
        cache_dir: Base cache directory (e.g., HF_HOME)
    
    Returns:
        State dict if weights found in cache, None otherwise
    """
    model_folder = get_model_cache_path(checkpoint, cache_dir)
    
    try:
        if os.path.exists(model_folder):
            snapshots = os.listdir(model_folder)
            if snapshots:
                print(f"Found {len(snapshots)} snapshots in cache")
                snapshot_dir = os.path.join(model_folder, snapshots[0])
                
                weights_file = os.path.join(snapshot_dir, "model.safetensors")
                if os.path.exists(weights_file):
                    print(f"Loading model weights from: {weights_file}")
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_file)
                    return state_dict
                else:
                    print(f"Model weights file not found: {weights_file}")
            else:
                print(f"No snapshots found in model directory: {model_folder}")
        else:
            print(f"Model directory not found: {model_folder}")
    except Exception as e:
        print(f"Error checking model directory: {e}")
    
    return None


def download_model_weights(
    checkpoint: str,
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Download model weights from HuggingFace Hub.
    
    Args:
        checkpoint: Model checkpoint name
        cache_dir: Cache directory for models (optional)
        device: Device to use for dtype selection ("cuda", "mps", or "cpu")
    
    Returns:
        State dict of the downloaded model
    
    Raises:
        RuntimeError: If download fails
    """
    print("Downloading model weights using huggingface_hub...")
    
    try:
        # Download the model.safetensors file directly
        weights_path = download_file_from_hub(
            checkpoint, 
            "model.safetensors", 
            cache_dir
        )
        
        print(f"Model weights downloaded to: {weights_path}")
        
        # Load the weights using safetensors
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        
        print("Model successfully downloaded!")
        return state_dict
    except Exception as e:
        print(f"Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to download model from {checkpoint}") from e


def process_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Process state dict to ensure compatibility with custom model.
    
    This function:
    1. Strips the leading ``model.`` prefix used by HuggingFace checkpoints
    2. Ensures ``lm_head.weight`` exists, creating it from the token embeddings if needed
    
    Args:
        state_dict: Raw state dict from downloaded model
    
    Returns:
        Processed state dict ready for loading
    """
    processed_state: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[len("model."):]
        else:
            new_key = key
        processed_state[new_key] = value

    if "lm_head.weight" not in processed_state:
        if "embed_tokens.weight" in processed_state:
            print("Creating lm_head.weight from embed_tokens.weight")
            processed_state["lm_head.weight"] = processed_state["embed_tokens.weight"]
        elif "model.embed_tokens.weight" in state_dict:
            # Fallback for checkpoints already processed partially
            print("Creating lm_head.weight from model.embed_tokens.weight")
            processed_state["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    return processed_state


def load_model_weights(
    model,
    checkpoint: str,
    cache_dir: Optional[str] = None,
    device: str = "cpu"
) -> None:
    """
    Load model weights automatically from cache or download if needed.
    
    This is the main entry point for loading model weights. It will:
    1. Try to load from local cache
    2. Download from HuggingFace Hub if not cached
    3. Process the state dict
    4. Load weights into the model
    
    Args:
        model: The model instance to load weights into
        checkpoint: Model checkpoint name
        cache_dir: Base cache directory (HF_HOME)
        device: Device for dtype selection
    """
    state_dict = load_cached_weights(checkpoint, cache_dir)
    
    if state_dict is None:
        model_cache = os.environ.get('MODEL_CACHE_DIR', cache_dir)
        state_dict = download_model_weights(checkpoint, model_cache)
    
    state_dict = process_state_dict(state_dict)
    
    model.load_state_dict(state_dict, strict=False)
    
    model_dict = model.state_dict()
    missing_keys = [key for key in model_dict.keys() if key not in state_dict]
    unexpected_keys = [key for key in state_dict.keys() if key not in model_dict]
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")