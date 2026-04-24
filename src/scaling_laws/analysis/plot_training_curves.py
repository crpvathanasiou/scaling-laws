import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import yaml


def ensure_dir(path: str):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> dict:
    """Load YAML config from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_results_dataframe(results_csv: str, run_name_prefix: str | None = None) -> pd.DataFrame:
    """
    Load results.csv and optionally keep only runs whose name starts
    with a given prefix.
    """
    df = pd.read_csv(results_csv)

    required_cols = ["run_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")

    if run_name_prefix:
        df = df[df["run_name"].astype(str).str.startswith(run_name_prefix)].copy()

    if len(df) == 0:
        raise ValueError("No runs found after filtering.")

    return df.reset_index(drop=True)


def load_curve_csv(curve_path: str) -> pd.DataFrame:
    """Load one train_loss_curve.csv file."""
    if not os.path.exists(curve_path):
        raise FileNotFoundError(f"Missing training curve file: {curve_path}")

    df = pd.read_csv(curve_path)

    required_cols = ["step", "train_loss"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in curve CSV {curve_path}: {missing}")

    return df


def plot_individual_curve(curve_df: pd.DataFrame, run_name: str, output_path: str):
    """Plot one individual training-loss curve."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(curve_df["step"], curve_df["train_loss"])
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title(f"Training Curve: {run_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined_curves(curves: list[tuple[str, pd.DataFrame]], output_path: str):
    """Plot all training curves on one figure."""
    plt.figure(figsize=(9, 6))

    for run_name, curve_df in curves:
        plt.plot(curve_df["step"], curve_df["train_loss"], label=run_name)

    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title("Training Curves Across All Runs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_summary_rows(curves: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Create a small summary table for the training curves."""
    rows = []

    for run_name, curve_df in curves:
        first_loss = float(curve_df["train_loss"].iloc[0])
        last_loss = float(curve_df["train_loss"].iloc[-1])
        n_steps = int(curve_df["step"].max())

        rows.append(
            {
                "run_name": run_name,
                "first_train_loss": first_loss,
                "last_train_loss": last_loss,
                "n_steps": n_steps,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    results_csv = cfg["results_csv"]
    runs_root = cfg["runs_root"]
    run_name_prefix = cfg.get("run_name_prefix", None)

    individual_output_dir = cfg["individual_output_dir"]
    combined_output_path = cfg["combined_output_path"]
    summary_output_path = cfg["summary_output_path"]

    ensure_dir(individual_output_dir)
    ensure_dir(os.path.dirname(combined_output_path))
    ensure_dir(os.path.dirname(summary_output_path))

    print("\nLoading run list...\n")
    results_df = load_results_dataframe(results_csv, run_name_prefix=run_name_prefix)

    curves = []

    for run_name in results_df["run_name"].tolist():
        curve_path = os.path.join(runs_root, run_name, "train_loss_curve.csv")
        curve_df = load_curve_csv(curve_path)
        curves.append((run_name, curve_df))

        individual_output_path = os.path.join(
            individual_output_dir,
            f"{run_name}_train_curve.png",
        )
        plot_individual_curve(curve_df, run_name, individual_output_path)

    print(f"Loaded {len(curves)} training curves.")

    print("\nCreating combined plot...\n")
    plot_combined_curves(curves, combined_output_path)

    print("Creating summary table...\n")
    summary_df = build_summary_rows(curves)
    summary_df.to_csv(summary_output_path, index=False)

    print("Saved outputs:")
    print(f"- {individual_output_dir}")
    print(f"- {combined_output_path}")
    print(f"- {summary_output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
