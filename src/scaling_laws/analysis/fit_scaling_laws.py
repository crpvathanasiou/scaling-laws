import os
import json
import argparse

import numpy as np
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


def load_and_filter_results(results_csv: str, run_name_prefix: str | None = None) -> pd.DataFrame:
    """
    Load results.csv and optionally keep only runs whose name starts
    with a given prefix.
    """
    df = pd.read_csv(results_csv)

    required_cols = ["run_name", "n_params", "n_tokens", "flops", "val_loss"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")

    if run_name_prefix:
        df = df[df["run_name"].astype(str).str.startswith(run_name_prefix)].copy()

    if len(df) == 0:
        raise ValueError("No rows left after filtering results.")

    df = df.sort_values(["n_params", "n_tokens"]).reset_index(drop=True)
    return df


def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit a power law of the form:
        y = a * x^b

    by linear regression in log-space:
        log(y) = log(a) + b * log(x)
    """
    if np.any(x <= 0):
        raise ValueError("All x values must be > 0 for log-log fitting.")
    if np.any(y <= 0):
        raise ValueError("All y values must be > 0 for log-log fitting.")

    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    y_pred_log = intercept + slope * log_x
    y_pred = np.exp(y_pred_log)

    ss_res = np.sum((log_y - y_pred_log) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    a = float(np.exp(intercept))
    b = float(slope)

    return {
        "form": "y = a * x^b",
        "a": a,
        "b": b,
        "r2_log_space": float(r2),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
    }


def add_power_law_line(ax, x_values: np.ndarray, fit: dict, label: str):
    """
    Draw a fitted power-law curve on existing axes.
    """
    x_line = np.logspace(np.log10(np.min(x_values)), np.log10(np.max(x_values)), 200)
    y_line = fit["a"] * (x_line ** fit["b"])
    ax.plot(x_line, y_line, linestyle="--", label=label)


def plot_loss_vs_params(df: pd.DataFrame, output_path: str, fit: dict):
    """
    Plot validation loss vs number of parameters on log-log axes.
    Color points by token budget.
    """
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    for token_budget in sorted(df["n_tokens"].unique()):
        subset = df[df["n_tokens"] == token_budget]
        ax.scatter(subset["n_params"], subset["val_loss"], label=f"tokens={token_budget}")

    add_power_law_line(ax, df["n_params"].to_numpy(), fit, label=f"fit: b={fit['b']:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of parameters (N)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation Loss vs Parameters")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loss_vs_tokens(df: pd.DataFrame, output_path: str, fit: dict):
    """
    Plot validation loss vs training tokens on log-log axes.
    Color points by model size.
    """
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    for n_params in sorted(df["n_params"].unique()):
        subset = df[df["n_params"] == n_params]
        ax.scatter(subset["n_tokens"], subset["val_loss"], label=f"params={n_params}")

    add_power_law_line(ax, df["n_tokens"].to_numpy(), fit, label=f"fit: b={fit['b']:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training tokens (D)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation Loss vs Training Tokens")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loss_vs_flops(df: pd.DataFrame, output_path: str, fit: dict):
    """
    Plot validation loss vs FLOPs on log-log axes.
    """
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    ax.scatter(df["flops"], df["val_loss"], label="runs")
    add_power_law_line(ax, df["flops"].to_numpy(), fit, label=f"fit: b={fit['b']:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Estimated FLOPs")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation Loss vs FLOPs")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    results_csv = cfg["results_csv"]
    output_dir = cfg["output_dir"]
    run_name_prefix = cfg.get("run_name_prefix", None)

    plots_cfg = cfg["plots"]
    outputs_cfg = cfg["outputs"]

    ensure_dir(output_dir)

    print("\nLoading results...\n")
    df = load_and_filter_results(results_csv, run_name_prefix=run_name_prefix)

    cleaned_results_path = os.path.join(output_dir, outputs_cfg["cleaned_results_csv"])
    df.to_csv(cleaned_results_path, index=False)

    print(f"Loaded {len(df)} runs after filtering.")
    print(df[["run_name", "n_params", "n_tokens", "flops", "val_loss"]].to_string(index=False))

    print("\nFitting power laws...\n")
    fit_params = fit_power_law(df["n_params"].to_numpy(dtype=float), df["val_loss"].to_numpy(dtype=float))
    fit_tokens = fit_power_law(df["n_tokens"].to_numpy(dtype=float), df["val_loss"].to_numpy(dtype=float))
    fit_flops = fit_power_law(df["flops"].to_numpy(dtype=float), df["val_loss"].to_numpy(dtype=float))

    scaling_fits = {
        "loss_vs_params": fit_params,
        "loss_vs_tokens": fit_tokens,
        "loss_vs_flops": fit_flops,
    }

    scaling_fits_path = os.path.join(output_dir, outputs_cfg["scaling_fits_json"])
    save_json(scaling_fits, scaling_fits_path)

    print("Creating plots...\n")
    plot_loss_vs_params(
        df=df,
        output_path=os.path.join(output_dir, plots_cfg["loss_vs_params"]),
        fit=fit_params,
    )
    plot_loss_vs_tokens(
        df=df,
        output_path=os.path.join(output_dir, plots_cfg["loss_vs_tokens"]),
        fit=fit_tokens,
    )
    plot_loss_vs_flops(
        df=df,
        output_path=os.path.join(output_dir, plots_cfg["loss_vs_flops"]),
        fit=fit_flops,
    )

    analysis_summary = {
        "n_runs": int(len(df)),
        "results_csv": results_csv,
        "cleaned_results_csv": cleaned_results_path,
        "plots": {
            "loss_vs_params": os.path.join(output_dir, plots_cfg["loss_vs_params"]),
            "loss_vs_tokens": os.path.join(output_dir, plots_cfg["loss_vs_tokens"]),
            "loss_vs_flops": os.path.join(output_dir, plots_cfg["loss_vs_flops"]),
        },
        "fits": scaling_fits,
    }

    analysis_summary_path = os.path.join(output_dir, outputs_cfg["analysis_summary_json"])
    save_json(analysis_summary, analysis_summary_path)

    print("Saved outputs:")
    print(f"- {cleaned_results_path}")
    print(f"- {scaling_fits_path}")
    print(f"- {os.path.join(output_dir, plots_cfg['loss_vs_params'])}")
    print(f"- {os.path.join(output_dir, plots_cfg['loss_vs_tokens'])}")
    print(f"- {os.path.join(output_dir, plots_cfg['loss_vs_flops'])}")
    print(f"- {analysis_summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
