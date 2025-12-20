#!/usr/bin/env python3

"""
Minimal inference example for SmolLM2-135M-Instruct.

Filepath: ./examples/chatbot_example.py
Project: CPEN455-Project-2025W1
Description: Loads the model and runs a single inference.

Usage:
    uv run -m examples.chatbot_example
"""

import os
import pdb
from dotenv import load_dotenv

from model import LlamaModel
from utils.sample import sample
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device

if __name__ == "__main__":
    load_dotenv()
    
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")

    # Set device to GPU if available, to MPS if on Mac with M-series chip, else CPU
    device = set_device()

    # Tokenizer and config loading now automatically download if not cached
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    model = LlamaModel(config)

    load_model_weights(model, checkpoint, cache_dir=model_cache_dir, device=device)

    model = model.to(device)
    model.eval()

    messages = [
        {"role": "user", "content": "What is gravity?"}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(input_text)
    
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = sample(
        model,
        inputs,
        max_new_tokens=500,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
    )
    print(tokenizer.decode(outputs[0]))
