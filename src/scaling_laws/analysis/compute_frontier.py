import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
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


def load_results_dataframe(cfg: dict) -> pd.DataFrame:
    """
    Prefer cleaned_results.csv if it already exists from the previous analysis step.
    Otherwise fall back to results.csv and filter by run_name_prefix if provided.
    """
    output_dir = cfg["output_dir"]
    outputs_cfg = cfg.get("outputs", {})
    cleaned_name = outputs_cfg.get("cleaned_results_csv", "cleaned_results.csv")
    cleaned_path = os.path.join(output_dir, cleaned_name)

    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    else:
        df = pd.read_csv(cfg["results_csv"])
        run_name_prefix = cfg.get("run_name_prefix", None)
        if run_name_prefix:
            df = df[df["run_name"].astype(str).str.startswith(run_name_prefix)].copy()

    required_cols = ["run_name", "n_params", "n_tokens", "flops", "val_loss"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results dataframe: {missing}")

    if len(df) == 0:
        raise ValueError("No rows available for frontier computation.")

    return df.sort_values(["flops", "val_loss"]).reset_index(drop=True)


def compute_empirical_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the empirical compute-optimal frontier.

    A run is on the frontier if, after sorting by increasing FLOPs,
    it achieves a strictly lower validation loss than all cheaper runs.
    """
    df = df.sort_values(["flops", "val_loss"]).reset_index(drop=True)

    frontier_rows = []
    best_loss_so_far = float("inf")

    for _, row in df.iterrows():
        loss = float(row["val_loss"])
        if loss < best_loss_so_far:
            frontier_rows.append(row.to_dict())
            best_loss_so_far = loss

    frontier_df = pd.DataFrame(frontier_rows)
    return frontier_df.reset_index(drop=True)


def plot_frontier(all_df: pd.DataFrame, frontier_df: pd.DataFrame, output_path: str):
    """
    Plot all runs in FLOPs-loss space and overlay the empirical frontier.
    """
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    ax.scatter(all_df["flops"], all_df["val_loss"], label="all runs")

    ax.plot(
        frontier_df["flops"],
        frontier_df["val_loss"],
        marker="o",
        linestyle="--",
        label="empirical frontier",
    )

    for _, row in frontier_df.iterrows():
        ax.annotate(
            row["run_name"],
            (row["flops"], row["val_loss"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Estimated FLOPs")
    ax.set_ylabel("Validation loss")
    ax.set_title("Compute-Optimal Frontier (Empirical)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    plots_cfg = cfg.get("plots", {})
    outputs_cfg = cfg.get("outputs", {})

    frontier_plot_name = plots_cfg.get("compute_frontier", "compute_frontier.png")
    frontier_csv_name = outputs_cfg.get("compute_frontier_csv", "compute_frontier.csv")
    frontier_json_name = outputs_cfg.get("compute_frontier_json", "compute_frontier.json")

    print("\nLoading analysis results...\n")
    df = load_results_dataframe(cfg)

    print("Computing empirical frontier...\n")
    frontier_df = compute_empirical_frontier(df)

    frontier_csv_path = os.path.join(output_dir, frontier_csv_name)
    frontier_json_path = os.path.join(output_dir, frontier_json_name)
    frontier_plot_path = os.path.join(output_dir, frontier_plot_name)

    frontier_df.to_csv(frontier_csv_path, index=False)

    frontier_summary = {
        "n_all_runs": int(len(df)),
        "n_frontier_runs": int(len(frontier_df)),
        "frontier_runs": frontier_df.to_dict(orient="records"),
    }
    save_json(frontier_summary, frontier_json_path)

    plot_frontier(df, frontier_df, frontier_plot_path)

    print("Frontier runs:")
    print(frontier_df[["run_name", "n_params", "n_tokens", "flops", "val_loss"]].to_string(index=False))

    print("\nSaved outputs:")
    print(f"- {frontier_csv_path}")
    print(f"- {frontier_json_path}")
    print(f"- {frontier_plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
