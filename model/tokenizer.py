#!/usr/bin/env python3

"""
Tokenizer implementation for the SmolLM2-135M-Instruct model.

Filepath: ./model/tokenizer.py
Project: CPEN455-Project-2025W1
Description: Define the Tokenizer for SmolLM2-135M-Instruct.

Usage:
    uv run main.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from utils.download import _resolve_snapshot_path, ensure_asset_exists


@dataclass
class TokenizerFiles:
    """Collection of tokenizer files required for initialization."""

    tokenizer: Path
    config: Path


def _find_tokenizer_files(base_path: Path) -> TokenizerFiles:
    tokenizer_file = ensure_asset_exists(base_path, "tokenizer.json")
    config_file = ensure_asset_exists(base_path, "tokenizer_config.json")
    return TokenizerFiles(tokenizer=tokenizer_file, config=config_file)


class Tokenizer:
    """Lightweight tokenizer compatible with the project's requirements."""

    _GPT2_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\d| ?[^\s\w]+| ?\w+|\s+(?!\S)|\s+""",
        re.IGNORECASE,
    )

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: Sequence[str],
        added_tokens: Sequence[dict],
        config: dict,
    ):
        self.encoder = vocab
        max_id = max(vocab.values())
        self.id_to_token: List[str] = [""] * (max_id + 1)
        for token, idx in vocab.items():
            if idx < 0:
                raise ValueError(f"Negative token id {idx} encountered for token '{token}'.")
            if idx >= len(self.id_to_token):
                self.id_to_token.extend([""] * (idx - len(self.id_to_token) + 1))
            self.id_to_token[idx] = token

        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            tuple(merge.split()): idx for idx, merge in enumerate(merges)
        }
        self._cache: Dict[str, Tuple[str, ...]] = {}

        self.config = config
        self.bos_token = config.get("bos_token")
        self.eos_token = config.get("eos_token")
        self.pad_token = config.get("pad_token")
        self.unk_token = config.get("unk_token")
        self.add_generation_prompt = True
        self.chat_template = config.get("chat_template")

        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        self.special_tokens: Dict[str, int] = {
            token_info["content"]: token_info["id"] for token_info in added_tokens
        }
        self.pad_token_id = None
        if self.pad_token is not None:
            if self.pad_token in self.special_tokens:
                self.pad_token_id = self.special_tokens[self.pad_token]
            elif self.pad_token in self.encoder:
                self.pad_token_id = self.encoder[self.pad_token]
        self.unk_token_id = None
        if self.unk_token is not None:
            if self.unk_token in self.special_tokens:
                self.unk_token_id = self.special_tokens[self.unk_token]
            elif self.unk_token in self.encoder:
                self.unk_token_id = self.encoder[self.unk_token]

        special_pattern = (
            "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            if self.special_tokens
            else ""
        )
        self._special_splitter = re.compile(f"({special_pattern})") if special_pattern else None

    @classmethod
    def from_pretrained(
        cls, checkpoint: str, cache_dir: Optional[str] = None
    ) -> "Tokenizer":
        base_path = _resolve_snapshot_path(checkpoint, cache_dir)
        files = _find_tokenizer_files(base_path)

        with open(files.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(files.tokenizer, "r", encoding="utf-8") as f:
            tokenizer_payload = json.load(f)

        vocab = tokenizer_payload["model"]["vocab"]
        merges = tokenizer_payload["model"]["merges"]
        added_tokens = tokenizer_payload.get("added_tokens", [])

        return cls(vocab, merges, added_tokens, config)

    def encode(
        self,
        text: str | Sequence[str],
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = False,
    ) -> (
        Sequence[int]
        | Sequence[Sequence[int]]
        | torch.Tensor
        | Dict[str, Sequence[Sequence[int]]]
        | Dict[str, torch.Tensor]
    ):
        if isinstance(text, str):
            input_ids = self._encode_text(text)

            if return_tensors is None:
                if return_attention_mask:
                    attention_mask = [1] * len(input_ids)
                    return {"input_ids": input_ids, "attention_mask": attention_mask}
                return input_ids

            if return_tensors != "pt":
                raise ValueError("Only return_tensors='pt' is supported in Tokenizer.encode")

            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            if return_attention_mask:
                attention_tensor = torch.ones_like(input_tensor, dtype=torch.long)
                return {"input_ids": input_tensor, "attention_mask": attention_tensor}
            return input_tensor

        if not hasattr(text, "__iter__"):
            raise TypeError("text must be a string or an iterable of strings.")

        batch_texts = list(text)
        if any(not isinstance(item, str) for item in batch_texts):
            raise TypeError("All items in the batch must be strings.")

        if not batch_texts:
            if return_tensors is None:
                if return_attention_mask:
                    return {"input_ids": [], "attention_mask": []}
                return []
            if return_tensors != "pt":
                raise ValueError("Only return_tensors='pt' is supported in Tokenizer.encode")
            empty_tensor = torch.empty((0, 0), dtype=torch.long)
            if return_attention_mask:
                return {"input_ids": empty_tensor, "attention_mask": empty_tensor.clone()}
            return empty_tensor

        batch_input_ids: List[List[int]] = [self._encode_text(item) for item in batch_texts]

        if return_tensors is None:
            if return_attention_mask:
                attention_masks = [[1] * len(ids) for ids in batch_input_ids]
                return {"input_ids": batch_input_ids, "attention_mask": attention_masks}
            return batch_input_ids

        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported in Tokenizer.encode")

        lengths = [len(ids) for ids in batch_input_ids]
        max_length = max(lengths, default=0)

        if any(length != max_length for length in lengths):
            if self.pad_token_id is None:
                raise ValueError(
                    "Batch tokenization with return_tensors='pt' requires consistent sequence lengths "
                    "or a pad_token to be specified in the tokenizer configuration."
                )
            padded = [
                ids + [self.pad_token_id] * (max_length - len(ids)) for ids in batch_input_ids
            ]
        else:
            padded = batch_input_ids

        input_tensor = torch.tensor(padded, dtype=torch.long)

        if return_attention_mask:
            attention_mask = [
                [1] * lengths[idx] + [0] * (max_length - lengths[idx]) for idx in range(len(batch_input_ids))
            ]
            attention_tensor = torch.tensor(attention_mask, dtype=torch.long)
            return {"input_ids": input_tensor, "attention_mask": attention_tensor}

        return input_tensor

    def decode(
        self,
        token_ids: Iterable[int] | torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim == 2:
                token_ids = token_ids[0]
            token_ids = token_ids.tolist()

        pieces: List[str] = []
        for index in token_ids:
            if index < 0 or index >= len(self.id_to_token):
                raise ValueError(f"Token id {index} is out of bounds for decoder vocabulary.")
            token = self.id_to_token[index]

            if skip_special_tokens and token in self.special_tokens:
                continue

            if token in self.special_tokens:
                pieces.append(token)
                continue

            decoded_bytes = bytearray()
            for char in token:
                if char not in self._byte_decoder:
                    raise ValueError(f"Unknown byte-level character '{char}' encountered during decoding.")
                decoded_bytes.append(self._byte_decoder[char])
            pieces.append(decoded_bytes.decode("utf-8", errors="replace"))

        return "".join(pieces)

    def apply_chat_template(
        self,
        messages: Sequence[dict],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ):
        if not isinstance(messages, Sequence):
            raise TypeError("messages must be a sequence of dictionaries")

        pieces: List[str] = []
        default_system = (
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, "
            "trained by Hugging Face<|im_end|>\n"
        )

        if messages and messages[0].get("role") != "system":
            pieces.append(default_system)

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n")

        rendered = "".join(pieces)

        if tokenize:
            return self.encode(rendered)

        return rendered

    def _encode_text(self, text: str) -> List[int]:
        if not text:
            return []

        segments = self._split_special_tokens(text)
        token_ids: List[int] = []
        for segment in segments:
            if not segment:
                continue
            if segment in self.special_tokens:
                token_ids.append(self.special_tokens[segment])
            else:
                token_ids.extend(self._encode_regular_text(segment))
        return token_ids

    def _split_special_tokens(self, text: str) -> List[str]:
        if not self._special_splitter:
            return [text]

        parts = self._special_splitter.split(text)
        return [part for part in parts if part]

    def _encode_regular_text(self, text: str) -> List[int]:
        ids: List[int] = []
        for token in self._GPT2_PATTERN.findall(text):
            if not token:
                continue

            transformed = "".join(self._byte_encoder[b] for b in token.encode("utf-8"))
            for bpe_token in self._bpe(transformed):
                token_id = self.encoder.get(bpe_token)
                if token_id is None:
                    if self.unk_token_id is not None:
                        ids.append(self.unk_token_id)
                        continue
                    raise KeyError(f"Token '{bpe_token}' not found in vocabulary and no unk_token is set.")
                ids.append(token_id)

        return ids

    def _bpe(self, token: str) -> Tuple[str, ...]:
        if token in self._cache:
            return self._cache[token]

        if not token:
            return ()

        word = list(token)
        pairs = self._get_pairs(word)

        while pairs:
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            merged: List[str] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == first
                    and word[i + 1] == second
                ):
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1

            word = merged
            if len(word) == 1:
                break

            pairs = self._get_pairs(word)

        result = tuple(word)
        self._cache[token] = result
        return result

    @staticmethod
    def _get_pairs(word: Sequence[str]) -> set[Tuple[str, str]]:
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs


def _bytes_to_unicode() -> Dict[int, str]:
    """Return byte-to-unicode mapping compatible with GPT-2 style BPE tokenizers."""

    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table
    from rich import box
    
    load_dotenv()
    
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    
    # Simple test to verify tokenizer functionality
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    test_text = "Hello, world! This is a test."
    encoded = tokenizer.encode(test_text)
    print("Encoded:", encoded)
    
    
    print("-" * 50)
    console = Console()
    table = Table(title="Token Analysis", box=box.SIMPLE_HEAVY)
    table.add_column("Token ID", justify="right", no_wrap=True)
    table.add_column("Token", justify="left")

    for token in encoded:
        decoded_token = tokenizer.decode([token])
        table.add_row(str(token), f"'{decoded_token}'")

    console.print(table)
    print("-" * 50)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
