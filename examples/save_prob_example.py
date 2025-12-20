#!/usr/bin/env python3

"""
Minimal save probabilities example for SmolLM2-135M-Instruct.

Filepath: ./examples/save_prob_example.py
Project: CPEN455-Project-2025W1
Description: This script demonstrates how to save the predicted probabilities of a model on a test dataset.

Usage:
    uv run -m examples.save_prob_example
"""

import os
from dotenv import load_dotenv
import argparse

import torch
from torch.utils.data import DataLoader

from autograder.dataset import CPEN455_2025_W1_Dataset
from model import LlamaModel
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from examples.bayes_inverse import save_probs

if __name__ == "__main__":
    # random seed for reproducibility
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")

    # Path to fine-tuned model weights (state dict)
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        default="examples/ckpts/bayesian_trained_data_synth.pt",
        help=(
            "Path to the fine-tuned LlamaModel state_dict (.pt file). "
            "Defaults to bayesian_trained_data_synth.pt"
        ),
    )

    parser.add_argument("--name_suffix", type=str, default="test")

    # Training hyperparameters
    parser.add_argument("--num_iterations", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    args = parser.parse_args()

    load_dotenv()

    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")

    # Set device to GPU if available, to MPS if on Mac with M-series chip, else CPU
    device = set_device()

    # Load tokenizer and config
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)

    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    # Load model
    model = LlamaModel(config)

    if not os.path.exists(args.finetuned_model_path):
        raise FileNotFoundError(
            f"Could not find fine-tuned model at {args.finetuned_model_path}. "
            "Either update the default path in save_prob_example.py or pass "
            "--finetuned_model_path explicitly."
        )

    state_dict = torch.load(args.finetuned_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
        )

    save_probs(args, model, tokenizer, test_dataloader, device=device, name=args.name_suffix)
