import os
import sys
import json
import math
import argparse
import subprocess

import yaml


def ensure_dir(path: str):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    """Save a Python object as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_yaml(path: str) -> dict:
    """Load YAML config from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def compute_max_steps(token_budget: int, batch_size: int, context_length: int) -> tuple[int, int]:
    """
    Compute how many optimization steps are needed to consume approximately
    the requested token budget.

    Tokens consumed per step are approximated as:
        batch_size * (context_length - 1)

    because training uses x = tokens[:-1], y = tokens[1:].
    """
    tokens_per_step = batch_size * (context_length - 1)
    max_steps = max(1, token_budget // tokens_per_step)
    actual_tokens = max_steps * tokens_per_step
    return max_steps, actual_tokens


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--token_budget", type=int, required=True)
    parser.add_argument("--train_config_path", type=str, required=True)
    parser.add_argument("--tokenizer_config_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="artifacts/runs")
    parser.add_argument("--results_csv", type=str, default="results/results.csv")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config_path)
    train_cfg = load_yaml(args.train_config_path)

    model_name = model_cfg["model_name"]
    context_length = model_cfg["context_length"]
    batch_size = train_cfg["batch_size"]

    max_steps, actual_tokens = compute_max_steps(
        token_budget=args.token_budget,
        batch_size=batch_size,
        context_length=context_length,
    )

    run_name = f"{model_name}_tok_{args.token_budget}"
    output_dir = os.path.join(args.output_root, run_name)
    ensure_dir(output_dir)

    run_spec = {
        "run_name": run_name,
        "model_config_path": args.model_config_path,
        "token_budget": args.token_budget,
        "actual_tokens_from_steps": actual_tokens,
        "train_config_path": args.train_config_path,
        "tokenizer_config_path": args.tokenizer_config_path,
        "train_data_path": args.train_data_path,
        "val_data_path": args.val_data_path,
        "output_root": args.output_root,
        "output_dir": output_dir,
        "results_csv": args.results_csv,
        "seed": args.seed,
        "batch_size": batch_size,
        "context_length": context_length,
        "max_steps": max_steps,
    }
    save_json(run_spec, os.path.join(output_dir, "run_spec.json"))

    cmd = [
        sys.executable,
        "./src/scaling_laws/train/train_model.py",
        "--config", args.train_config_path,
        "--model_config_path", args.model_config_path,
        "--tokenizer_config_path", args.tokenizer_config_path,
        "--train_data_path", args.train_data_path,
        "--val_data_path", args.val_data_path,
        "--output_dir", output_dir,
        "--results_csv_path", args.results_csv,
        "--max_steps", str(max_steps),
        "--seed", str(args.seed),
    ]

    print("\nRunning experiment with:")
    print(f"  model_name:      {model_name}")
    print(f"  token_budget:    {args.token_budget}")
    print(f"  tokens/step:     {batch_size * (context_length - 1)}")
    print(f"  max_steps:       {max_steps}")
    print(f"  actual_tokens:   {actual_tokens}")
    print(f"  output_dir:      {output_dir}")
    print()

    subprocess.run(cmd, check=True)

    print("\nExperiment completed.")
    print(f"Run spec saved to: {os.path.join(output_dir, 'run_spec.json')}")


if __name__ == "__main__":
    main()
