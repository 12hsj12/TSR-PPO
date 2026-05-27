from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import ensure_output, setup_style


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def generate_evaluation_figures(summary_csv: str | Path, output_dir: str | Path) -> list[Path]:
    setup_style()
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary metrics CSV not found: {summary_csv}")
    out = ensure_output(output_dir)
    df = pd.read_csv(summary_csv, encoding="utf-8")
    saved: list[Path] = []
    if {"algorithm", "cmax_mean"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df["algorithm"], df["cmax_mean"], yerr=df["cmax_std"] if "cmax_std" in df else None, capsize=3)
        ax.set_title("Algorithm Cmax Comparison")
        ax.set_ylabel("Cmax")
        ax.tick_params(axis="x", rotation=20)
        path = out / "algorithm_cmax_comparison.png"
        _save(fig, path)
        saved.append(path)
    else:
        print("Warning: skip algorithm_cmax_comparison.png; missing algorithm/cmax_mean")

    if {"algorithm", "utilization_mean", "load_balance_std"}.issubset(df.columns):
        fig, ax1 = plt.subplots(figsize=(8, 4))
        x = range(len(df))
        ax1.bar([i - 0.18 for i in x], df["utilization_mean"], width=0.36, label="utilization")
        ax2 = ax1.twinx()
        ax2.bar([i + 0.18 for i in x], df["load_balance_std"], width=0.36, color="#d62728", alpha=0.75, label="load balance")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(df["algorithm"], rotation=20)
        ax1.set_ylabel("Utilization")
        ax2.set_ylabel("Load balance std")
        ax1.set_title("Utilization and Load Balance")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False)
        path = out / "utilization_balance.png"
        _save(fig, path)
        saved.append(path)
    else:
        print("Warning: skip utilization_balance.png; missing utilization/load balance columns")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation summary figures.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_evaluation_figures(args.summary, args.output)


if __name__ == "__main__":
    main()
