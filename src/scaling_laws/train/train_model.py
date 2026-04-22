import os
import json
import time
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import yaml
#from src.scaling_laws.models.gpt
from scaling_laws.models.gpt import build_model 


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    """Save a Python object as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv_row(path: str, row: dict):
    """Append one row to a CSV file, creating the header if needed."""
    import csv

    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_yaml_config(path: str) -> dict:
    """Load YAML config from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def apply_config_defaults(parser: argparse.ArgumentParser, config: dict):
    """Override argparse defaults with YAML config values."""
    parser.set_defaults(**config)


def load_numpy_chunks(path: str):
    """Load precomputed token chunks from .npy file."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array at {path}, got shape {arr.shape}")
    return arr


def make_batch(data: np.ndarray, batch_size: int, device: torch.device):
    """
    Sample a random batch from chunked token data.

    Input chunks are shape (N, T).
    We create:
      x = chunk[:, :-1]
      y = chunk[:, 1:]
    """
    idx = np.random.randint(0, len(data), size=batch_size)
    batch = data[idx]  # (B, T)

    x = torch.tensor(batch[:, :-1], dtype=torch.long, device=device)
    y = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data: np.ndarray, batch_size: int, eval_steps: int, device: torch.device):
    """Estimate average loss over a number of random mini-batches."""
    model.eval()

    losses = []
    for _ in range(eval_steps):
        x, y = make_batch(data, batch_size=batch_size, device=device)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_argv = config_parser.parse_known_args()

    config = {}
    if config_args.config is not None:
        config = load_yaml_config(config_args.config)

    parser = argparse.ArgumentParser(parents=[config_parser])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_data_path", type=str, required=False, default=None)
    parser.add_argument("--val_data_path", type=str, required=False, default=None)
    parser.add_argument("--model_config_path", type=str, required=False, default=None)
    parser.add_argument("--tokenizer_config_path", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=str, default="artifacts/runs/run_debug")
    parser.add_argument("--results_csv_path", type=str, default="results/results.csv")

    apply_config_defaults(parser, config)
    args = parser.parse_args(remaining_argv)

    if args.train_data_path is None:
        raise ValueError("train_data_path must be provided via config or CLI.")
    if args.val_data_path is None:
        raise ValueError("val_data_path must be provided via config or CLI.")
    if args.model_config_path is None:
        raise ValueError("model_config_path must be provided via config or CLI.")
    if args.tokenizer_config_path is None:
        raise ValueError("tokenizer_config_path must be provided via config or CLI.")

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    print("\nLoading training/validation chunks...\n")
    train_data = load_numpy_chunks(args.train_data_path)
    val_data = load_numpy_chunks(args.val_data_path)

    print(f"Train chunks shape: {train_data.shape}")
    print(f"Val chunks shape:   {val_data.shape}")

    print("\nLoading model config and tokenizer config...\n")
    model_cfg = load_yaml_config(args.model_config_path)

    with open(args.tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_cfg = json.load(f)

    vocab_size = tokenizer_cfg["vocab_size"]

    print("\nBuilding model...\n")
    model = build_model(model_cfg=model_cfg, vocab_size=vocab_size).to(device)
    n_params = model.count_parameters(trainable_only=True)

    print(f"Vocab size: {vocab_size}")
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_curve_path = os.path.join(args.output_dir, "train_loss_curve.csv")
    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    summary_path = os.path.join(args.output_dir, "run_summary.json")
    ckpt_path = os.path.join(args.output_dir, "final_model.pt")
    results_csv_path = args.results_csv_path
    ensure_dir(os.path.dirname(results_csv_path))
    
    ensure_dir("results")

    start_time = time.time()
    last_val_loss = None

    print("\nStarting training...\n")
    model.train()

    for step in range(1, args.max_steps + 1):
        x, y = make_batch(train_data, batch_size=args.batch_size, device=device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        append_csv_row(
            train_curve_path,
            {
                "step": step,
                "train_loss": float(loss.item()),
            },
        )

        if step == 1 or step % args.eval_every == 0 or step == args.max_steps:
            train_loss = float(loss.item())
            val_loss = estimate_loss(
                model=model,
                data=val_data,
                batch_size=args.batch_size,
                eval_steps=args.eval_steps,
                device=device,
            )
            last_val_loss = val_loss

            print(f"step={step:4d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            save_json(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                metrics_path,
            )

    wall_clock_sec = time.time() - start_time

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg,
            "vocab_size": vocab_size,
        },
        ckpt_path,
    )

    context_length = int(model_cfg["context_length"])
    n_tokens = int(args.max_steps * args.batch_size * (context_length - 1))
    flops = int(6 * n_params * n_tokens)

    run_summary = {
        "device": str(device),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "eval_steps": args.eval_steps,
        "seed": args.seed,
        "train_data_path": args.train_data_path,
        "val_data_path": args.val_data_path,
        "model_config_path": args.model_config_path,
        "tokenizer_config_path": args.tokenizer_config_path,
        "output_dir": args.output_dir,
        "n_params": n_params,
        "n_tokens": n_tokens,
        "flops": flops,
        "final_val_loss": last_val_loss,
        "wall_clock_sec": wall_clock_sec,
    }
    save_json(run_summary, summary_path)

    append_csv_row(
        results_csv_path,
        {
            "run_name": os.path.basename(args.output_dir),
            "n_params": n_params,
            "n_tokens": n_tokens,
            "flops": flops,
            "val_loss": last_val_loss,
            "device": str(device),
            "context_length": context_length,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
    )

    print("\nSaved outputs:")
    print(f"- {train_curve_path}")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")
    print(f"- {ckpt_path}")
    print(f"- {results_csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
