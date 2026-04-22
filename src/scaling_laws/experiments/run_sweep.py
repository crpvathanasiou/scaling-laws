import os
import sys
import json
import argparse
import itertools
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    sweep_cfg = load_yaml(args.config)

    model_configs = sweep_cfg["model_configs"]
    token_budgets = sweep_cfg["token_budgets"]
    train_config_path = sweep_cfg["train_config_path"]
    tokenizer_config_path = sweep_cfg["tokenizer_config_path"]
    train_data_path = sweep_cfg["train_data_path"]
    val_data_path = sweep_cfg["val_data_path"]
    output_root = sweep_cfg["output_root"]
    results_csv = sweep_cfg["results_csv"]
    seed = sweep_cfg.get("seed", 42)

    ensure_dir(output_root)
    ensure_dir(os.path.dirname(results_csv))

    experiments = []
    for model_config_path, token_budget in itertools.product(model_configs, token_budgets):
        experiments.append(
            {
                "model_config_path": model_config_path,
                "token_budget": int(token_budget),
                "train_config_path": train_config_path,
                "tokenizer_config_path": tokenizer_config_path,
                "train_data_path": train_data_path,
                "val_data_path": val_data_path,
                "output_root": output_root,
                "results_csv": results_csv,
                "seed": seed,
            }
        )

    sweep_plan_path = os.path.join(output_root, "sweep_plan.json")
    save_json(
        {
            "n_experiments": len(experiments),
            "experiments": experiments,
        },
        sweep_plan_path,
    )

    print("\nSweep plan:")
    print(f"  config:         {args.config}")
    print(f"  n_experiments:  {len(experiments)}")
    print(f"  output_root:    {output_root}")
    print(f"  results_csv:    {results_csv}")
    print(f"  sweep_plan:     {sweep_plan_path}")
    print()

    for i, exp in enumerate(experiments, start=1):
        print(f"[{i}/{len(experiments)}] model={exp['model_config_path']} | token_budget={exp['token_budget']}")

        cmd = [
            sys.executable,
            "./src/scaling_laws/experiments/run_experiment.py",
            "--model_config_path", exp["model_config_path"],
            "--token_budget", str(exp["token_budget"]),
            "--train_config_path", exp["train_config_path"],
            "--tokenizer_config_path", exp["tokenizer_config_path"],
            "--train_data_path", exp["train_data_path"],
            "--val_data_path", exp["val_data_path"],
            "--output_root", exp["output_root"],
            "--results_csv", exp["results_csv"],
            "--seed", str(exp["seed"]),
        ]

        if args.dry_run:
            print("  DRY RUN:", " ".join(cmd))
            continue

        subprocess.run(cmd, check=True)

    print("\nSweep completed.")


if __name__ == "__main__":
    main()
