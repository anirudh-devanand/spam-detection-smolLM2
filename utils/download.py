#!/usr/bin/env python3

"""
Download utility functions for fetching files from HuggingFace Hub.

Filepath: ./utils/download.py
Project: CPEN455-Project-2025W1
Description: Functions for downloading and managing model files from HuggingFace Hub.

Usage:
    uv run python -m utils.download
"""

import os
from typing import Optional
from pathlib import Path


def ensure_asset_exists(base_path: Path, filename: str) -> Path:
    """
    Return the resolved path for a required asset, ensuring it exists.
    """
    asset_path = base_path / filename
    if not asset_path.exists():
        raise FileNotFoundError(f"Required file '{filename}' not found at '{asset_path}'.")
    return asset_path


def download_file_from_hub(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None,
    revision: str = "main"
) -> str:
    """
    Download a file from HuggingFace Hub using huggingface_hub library.
    
    Args:
        repo_id: Repository ID (e.g., "HuggingFaceTB/SmolLM2-135M-Instruct")
        filename: Name of the file to download (e.g., "config.json", "model.safetensors")
        cache_dir: Directory to cache the file
        revision: Git revision (branch, tag, or commit hash)
    
    Returns:
        Path to the downloaded file
    """
    from huggingface_hub import hf_hub_download
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=False
        )
        return file_path
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {e}")
        raise


def download_config(checkpoint: str, cache_dir: Optional[str] = None) -> str:
    """
    Download model configuration file from HuggingFace Hub.
    
    Args:
        checkpoint: Model checkpoint name
        cache_dir: Cache directory (optional)
    
    Returns:
        Path to the downloaded config.json file
    """
    print(f"Downloading config.json from {checkpoint}...")
    config_path = download_file_from_hub(checkpoint, "config.json", cache_dir)
    print(f"Config downloaded to: {config_path}")
    return config_path


def get_model_cache_path(checkpoint: str, cache_dir: str) -> str:
    """
    Get the cache path for a HuggingFace model.
    
    Args:
        checkpoint: Model checkpoint name (e.g., "HuggingFaceTB/SmolLM2-135M-Instruct")
        cache_dir: Base cache directory (e.g., HF_HOME)
    
    Returns:
        Path to the model's snapshot directory
    """
    model_folder = f"{cache_dir}/models--{checkpoint.replace('/', '--')}/snapshots/"
    return model_folder


def _resolve_snapshot_path(checkpoint: str, cache_dir: Optional[str]) -> Path:
    """Resolve the filesystem path containing tokenizer assets.
    
    If files are not found in cache, attempts to download them.
    """
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path

    if cache_dir is None:
        raise FileNotFoundError(
            f"Tokenizer files for '{checkpoint}' not found locally and no cache_dir provided."
        )

    snapshots_root = Path(get_model_cache_path(checkpoint, cache_dir))
    if not snapshots_root.exists():
        # Try to download the required files
        print(f"No cached snapshots found for checkpoint '{checkpoint}'. Attempting to download...")
        try:
            download_file_from_hub(checkpoint, "config.json", cache_dir)
            download_file_from_hub(checkpoint, "tokenizer.json", cache_dir)
            download_file_from_hub(checkpoint, "tokenizer_config.json", cache_dir)
            # After download, the snapshots_root should exist
            if not snapshots_root.exists():
                raise FileNotFoundError(
                    f"Failed to download files for checkpoint '{checkpoint}'."
                )
        except Exception as e:
            raise FileNotFoundError(
                f"No cached snapshots found and download failed for checkpoint '{checkpoint}': {e}"
            )

    snapshots = sorted(
        (p for p in snapshots_root.iterdir() if p.is_dir()),
        key=os.path.getmtime,
        reverse=True,
    )

    if not snapshots:
        raise FileNotFoundError(
            f"No snapshot directories found for checkpoint '{checkpoint}' in '{snapshots_root}'."
        )

    return snapshots[0]