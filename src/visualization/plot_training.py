from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import ensure_output, setup_style


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    out = ensure_output(args.output)
    df = pd.read_csv(args.input)
    window = min(20, max(3, len(df) // 20)) if len(df) >= 3 else 1
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df["episode"], df["train_reward"], lw=1.6)
    if window > 1:
        ax[0].plot(df["episode"], df["train_reward"].rolling(window, min_periods=1).mean(), lw=2.0, label=f"MA{window}")
        ax[0].legend(frameon=False)
    ax[0].set_title("训练奖励")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[1].plot(df["episode"], df["train_cmax"], label="train", lw=1.0, alpha=0.45)
    if window > 1:
        ax[1].plot(df["episode"], df["train_cmax"].rolling(window, min_periods=1).mean(), label=f"train MA{window}", lw=1.8)
    if "eval_cmax_mean" in df:
        eval_df = df.dropna(subset=["eval_cmax_mean"])
        ax[1].plot(eval_df["episode"], eval_df["eval_cmax_mean"], label="eval", lw=1.8)
    elif "last_eval_cmax_mean" in df:
        eval_df = df.dropna(subset=["last_eval_cmax_mean"])
        ax[1].plot(eval_df["episode"], eval_df["last_eval_cmax_mean"], label="last eval", lw=1.8)
    ax[1].set_title("Cmax 变化")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Cmax")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", bbox_inches="tight")
