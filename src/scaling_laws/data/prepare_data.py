import os
import json
import argparse
import random

import numpy as np
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer


def set_seed(seed: int):
    """Set Python's random seed for reproducibility."""
    random.seed(seed)


def ensure_dir(path: str):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    """Save a Python object as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_text(text):
    """
    Basic text cleaning:
    - keep only string values
    - strip surrounding whitespace
    - drop empty strings
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return text


def load_yaml_config(path: str) -> dict:
    """Load YAML config from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def apply_config_defaults(parser: argparse.ArgumentParser, config: dict):
    """
    Override argparse defaults with values loaded from YAML config.
    CLI flags still take precedence over config values.
    """
    parser.set_defaults(**config)


def load_and_split_dataset(
    dataset_name: str,
    text_column: str,
    val_fraction: float,
    seed: int,
    max_train_texts: int | None = None,
    max_val_texts: int | None = None,
):
    """
    Load the dataset, clean texts, shuffle them, and split into train/validation.

    Splitting is done at the story/row level.
    """
    ds = load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=seed)

    filtered_texts = []
    for x in ds[text_column]:
        t = clean_text(x)
        if t is not None:
            filtered_texts.append(t)

    n_total = len(filtered_texts)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    train_texts = filtered_texts[:n_train]
    val_texts = filtered_texts[n_train:]

    if max_train_texts is not None:
        train_texts = train_texts[:max_train_texts]

    if max_val_texts is not None:
        val_texts = val_texts[:max_val_texts]

    return train_texts, val_texts


def encode_and_chunk(texts, tokenizer: Tokenizer, context_length: int):
    """
    Encode texts and pack them into fixed-length chunks.

    The tokenizer already adds BOS/EOS through its saved post-processor,
    so this function simply uses tokenizer.encode(text).ids.

    Any remainder shorter than context_length is dropped.
    """
    chunks = []
    buffer = []

    total_token_count = 0
    total_texts = 0

    for text in texts:
        ids = tokenizer.encode(text).ids
        total_token_count += len(ids)
        total_texts += 1

        buffer.extend(ids)

        while len(buffer) >= context_length:
            chunk = buffer[:context_length]
            chunks.append(chunk)
            buffer = buffer[context_length:]

    dropped_tokens = len(buffer)

    if len(chunks) == 0:
        chunk_array = np.zeros((0, context_length), dtype=np.int32)
    else:
        chunk_array = np.asarray(chunks, dtype=np.int32)

    return {
        "chunks": chunk_array,
        "n_texts": total_texts,
        "total_token_count": total_token_count,
        "n_chunks": int(chunk_array.shape[0]),
        "dropped_tokens": dropped_tokens,
    }


def save_preview(tokenizer: Tokenizer, train_chunks: np.ndarray, output_dir: str):
    """
    Save a small preview of the first training chunk for sanity checking.
    """
    preview_path = os.path.join(output_dir, "sample_batch_preview.json")

    if len(train_chunks) == 0:
        preview = {
            "message": "No training chunks were generated."
        }
        save_json(preview, preview_path)
        return

    first_chunk = train_chunks[0].tolist()
    first_tokens = [tokenizer.id_to_token(token_id) for token_id in first_chunk[:32]]

    preview = {
        "first_chunk_shape": list(train_chunks[0].shape),
        "first_chunk_ids_first_32": first_chunk[:32],
        "first_chunk_tokens_first_32": first_tokens,
    }
    save_json(preview, preview_path)


def main():
    """
    Main entry point.

    This script:
    1. Loads the Greek dataset
    2. Cleans and splits it
    3. Loads the frozen tokenizer
    4. Encodes train/validation texts
    5. Packs token streams into fixed-length chunks
    6. Saves .npy outputs and metadata
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_argv = config_parser.parse_known_args()

    config = {}
    if config_args.config is not None:
        config = load_yaml_config(config_args.config)

    parser = argparse.ArgumentParser(parents=[config_parser])

    parser.add_argument("--dataset_name", type=str, default="alexliap/tinystories-gr")
    parser.add_argument("--text_column", type=str, default="greek_translation")

    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tokenizer_path", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=str, default="artifacts/data")

    # Optional limits for dry runs / debugging
    parser.add_argument("--max_train_texts", type=int, default=None)
    parser.add_argument("--max_val_texts", type=int, default=None)

    apply_config_defaults(parser, config)
    args = parser.parse_args(remaining_argv)

    if args.tokenizer_path is None:
        raise ValueError("tokenizer_path must be provided via config or CLI.")

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    print("\nLoading and splitting dataset...\n")
    train_texts, val_texts = load_and_split_dataset(
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_train_texts=args.max_train_texts,
        max_val_texts=args.max_val_texts,
    )

    print(f"Train texts: {len(train_texts)}")
    print(f"Val texts:   {len(val_texts)}")

    print("\nLoading tokenizer...\n")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print("\nEncoding and chunking train split...\n")
    train_result = encode_and_chunk(
        texts=train_texts,
        tokenizer=tokenizer,
        context_length=args.context_length,
    )

    print("\nEncoding and chunking validation split...\n")
    val_result = encode_and_chunk(
        texts=val_texts,
        tokenizer=tokenizer,
        context_length=args.context_length,
    )

    train_chunks_path = os.path.join(args.output_dir, "train_chunks.npy")
    val_chunks_path = os.path.join(args.output_dir, "val_chunks.npy")

    np.save(train_chunks_path, train_result["chunks"])
    np.save(val_chunks_path, val_result["chunks"])

    metadata = {
        "dataset_name": args.dataset_name,
        "text_column": args.text_column,
        "context_length": args.context_length,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "tokenizer_path": args.tokenizer_path,
        "output_dir": args.output_dir,
        "n_train_texts": train_result["n_texts"],
        "n_val_texts": val_result["n_texts"],
        "train_total_token_count": train_result["total_token_count"],
        "val_total_token_count": val_result["total_token_count"],
        "n_train_chunks": train_result["n_chunks"],
        "n_val_chunks": val_result["n_chunks"],
        "train_dropped_tokens": train_result["dropped_tokens"],
        "val_dropped_tokens": val_result["dropped_tokens"],
        "train_chunks_shape": list(train_result["chunks"].shape),
        "val_chunks_shape": list(val_result["chunks"].shape),
        "max_train_texts": args.max_train_texts,
        "max_val_texts": args.max_val_texts,
    }

    save_json(metadata, os.path.join(args.output_dir, "data_prep_metadata.json"))
    save_preview(tokenizer, train_result["chunks"], args.output_dir)

    print("\nSaved outputs:")
    print(f"- {train_chunks_path}")
    print(f"- {val_chunks_path}")
    print(f"- {os.path.join(args.output_dir, 'data_prep_metadata.json')}")
    print(f"- {os.path.join(args.output_dir, 'sample_batch_preview.json')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
