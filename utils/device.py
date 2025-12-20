#!/usr/bin/env python3

"""
Device detection and configuration utilities.

Filepath: ./utils/device.py
Project: CPEN455-Project-2025W1
Description: Utility functions for detecting and setting the appropriate compute device (CUDA, MPS, or CPU).
"""

import torch

def set_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device