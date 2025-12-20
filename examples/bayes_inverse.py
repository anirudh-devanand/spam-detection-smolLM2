#!/usr/bin/env python3

"""
Minimal bayes inverse example for SmolLM2-135M-Instruct.

Filepath: ./examples/bayes_inverse_example.py
Project: CPEN455-Project-2025W1
Description: integrates three different ways to perform bayes inverse classification with LLMs for spam detection.

Usage:
    uv run -m examples.bayes_inverse
"""

import os
import pdb
import wandb
from dotenv import load_dotenv
from einops import rearrange
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


from autograder.dataset import CPEN455_2025_W1_Dataset, ENRON_LABEL_INDEX_MAP, prepare_subset
from model import LlamaModel
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from utils.prompt_template import get_prompt
from utils.logger import avg_logger, avg_acc_logger

def get_seq_log_prob(prompts, tokenizer, model, device):
    encoded_batch = tokenizer.encode(
        prompts, return_tensors="pt", return_attention_mask=True
    )

    input_ids = encoded_batch["input_ids"].to(device)
    attention_mask = encoded_batch["attention_mask"].to(device)

    log_prob, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    shifted_log_prob = log_prob[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]
    shifted_attention_mask = attention_mask[:, 1:]

    gathered_log_prob = shifted_log_prob.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    gathered_log_prob = gathered_log_prob * shifted_attention_mask

    return gathered_log_prob.sum(dim=-1)


METHOD_SET = ["zero_shot", "naive_prompting", "full_finetune"]

def is_required_training(method: str) -> bool:
    assert method in METHOD_SET, f"Method {method} not recognized. Choose from {METHOD_SET}."
    return method in METHOD_SET[2:]

def bayes_inverse_llm_classifier(args, model, batch, tokenizer, device):

    _, subjects, messages, labels = batch

    prompts_ham = [get_prompt(subject=subj, message=msg, label=ENRON_LABEL_INDEX_MAP.inv[0], max_seq_length=args.max_seq_len, user_prompt=args.user_prompt) for subj, msg in zip(subjects, messages)]
    prompts_spam = [get_prompt(subject=subj, message=msg, label=ENRON_LABEL_INDEX_MAP.inv[1], max_seq_length=args.max_seq_len, user_prompt=args.user_prompt) for subj, msg in zip(subjects, messages)]

    # The first half are ham, the second half are spam
    prompts = prompts_ham + prompts_spam
    with torch.no_grad():
        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device)

        '''
        Rearrange to (batch_size, 2), in this way, the second dimension 0 is ham, 1 is spam.
        '''
        seq_log_prob = rearrange(seq_log_prob, '(c b) -> b c', c=2)

        '''
        Apply softmax over ham/spam dimension to get probabilities.
        The shape of probs will be (2, batch_size), where probs[0, :] is ham probability and probs[1, :] is spam probability.
        probs[:, i] gives the category distribution used to classify spam and ham for the i-th email in the batch.
        '''
        probs = F.softmax(seq_log_prob, dim=-1)

        labels_pred = torch.argmax(probs, dim=-1)

        if  -1 in labels:
            is_correct = None
        else:
            is_correct = labels_pred.cpu() == labels

        return is_correct, (probs.detach().cpu(), labels_pred.detach().cpu())

def train_or_test(args, model, tokenizer, batch, optimizer=None, is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()

    _, subjects, messages, label_indexs = batch

    if -1 in label_indexs:
        bpd = None
    else:
        labels_text = [ENRON_LABEL_INDEX_MAP.inv[int(label_index)] for label_index in label_indexs]

        prompts = [get_prompt(subject=subj, message=msg, label=label, max_seq_length=args.max_seq_len) for subj, msg, label in zip(subjects, messages, labels_text)]

        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device=device)

        num_characters = torch.tensor([len(prompt) for prompt in prompts], device=device).sum()
        bpd = -seq_log_prob.sum()/num_characters

        if is_training:
            assert optimizer is not None, "Optimizer must be provided during training."
            optimizer.zero_grad()
            bpd.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    is_correct, (probs, labels_pred) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device=device)

    return bpd, is_correct, (probs, labels_pred)

def save_probs(args, model, tokenizer, dataloader, device, name = "test"):
    save_path = os.path.join(os.getcwd(), f"{args.prob_output_folder}/{name}_dataset_probs.csv")

    if os.path.exists(save_path):
        os.remove(save_path)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="saving probabilities"):

            _, (probs, _) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device = device)

            data_index, _, _, _ = batch
            indices = torch.as_tensor(data_index).view(-1).tolist()

            rows = zip(indices, probs[:, 0].tolist(), probs[:, 1].tolist())

            file_exists = os.path.exists(save_path)
            with open(save_path, "a", newline="") as handle:
                if not file_exists:
                    handle.write("data_index,prob_ham,prob_spam\n")
                handle.writelines(f"{idx},{ham},{spam}\n" for idx, ham, spam in rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="zero-shot", choices=METHOD_SET)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--dataset_path", type=str, default="autograder/cpen455_released_datasets/train_val_augmented.csv")
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")

    # Training hyperparameters
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        default="examples/ckpts/bayesian_trained_data_synth.pt",
        help=(
            "Where to save the fine-tuned LlamaModel state_dict (.pt file). "
            "Defaults to examples/ckpts/bayesian_trained_data_synth.pt"
        ),
    )

    parser.add_argument(
        "--save_finetuned_model",
        default=True,
        action="store_true",
        help="If set, save the fine-tuned model to --finetuned_model_path.",
    )

    parser.add_argument("--seed", default=0)

    args = parser.parse_args()

    # random seed for reproducibility
    torch.manual_seed(args.seed)

    load_dotenv()

    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")

    run = None
    if not is_required_training(args.method):
        run = wandb.init(
            project=os.getenv("PROJECT_NAME"),
            name=f"bayes-inverse-{args.method}_msl{args.max_seq_len}",
        )
    else:
        run = wandb.init(
            project=os.getenv("PROJECT_NAME"),
            name=f"bayes-inverse-{args.method}_msl{args.max_seq_len}_ni{args.num_iterations}_bs{args.batch_size}",
        )

    wandb.config.update(args)

    # Set device to GPU if available, to MPS if on Mac with M-series chip, else CPU
    device = set_device()

    # Load tokenizer and config
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)

    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    # Load model
    model = LlamaModel(config)
    load_model_weights(model, checkpoint, cache_dir=model_cache_dir, device=device)
    model = model.to(device)

    # Set up datasets and dataloaders
    train_n_val_dataset = CPEN455_2025_W1_Dataset(csv_path=args.dataset_path)
    training_dataset, val_dataset = prepare_subset(train_n_val_dataset, int(0.8 * len(train_n_val_dataset)), ratio_spam=0.5, return_remaining=True)
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    if os.path.exists(args.prob_output_folder) == False:
        os.makedirs(args.prob_output_folder)

    for iteration in tqdm(range(args.num_iterations), desc="Training"):

        if (iteration + 1) % 10 == 0:
            val_acc_logger = avg_acc_logger()
            val_bpd_logger = avg_logger()

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Evaluating on validation set during training"):

                    bpd, is_correct, (probs, labels_pred) = train_or_test(
                        args = args,
                        model = model,
                        tokenizer = tokenizer,
                        batch = batch,
                        is_training=False)

                    val_acc_logger.update(is_correct)
                    val_bpd_logger.update(bpd.item())

                    wandb.log({
                        "val_avg_bpd": val_bpd_logger.compute_average(),
                        "val_avg_accuracy": val_acc_logger.compute_accuracy(),
                        "training_iteration": iteration,
                        })

        if not is_required_training(args.method):
            break

        batch = next(iter(training_dataloader))

        bpd, is_correct, _ = train_or_test(
            args = args,
            model = model,
            tokenizer = tokenizer,
            batch = batch,
            optimizer = optimizer,
            is_training = True)

        wandb.log({
            "training_batch_bpd": bpd.item(),
            "training_batch_acc": is_correct.float().mean().item(),
            "training_iteration": iteration,
            })

    if args.save_finetuned_model:
        ckpt_dir = os.path.dirname(args.finetuned_model_path)
        if ckpt_dir != "":
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), args.finetuned_model_path)
        print(f"[bayes_inverse] Saved fine-tuned model checkpoint to {args.finetuned_model_path}")

    # After training, save probabilities on test set
    train_n_val_dataloader = DataLoader(
        train_n_val_dataset,
        batch_size=args.batch_size,
        shuffle=False
        )
    save_probs(args, model, tokenizer, train_n_val_dataloader, device=device, name = "train_n_val")
    save_probs(args, model, tokenizer, test_dataloader, device=device, name = "test")
