from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import ensure_output, setup_style


def _read_training_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"training curve CSV not found: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def _ma_window(length: int, requested: int) -> int:
    if length <= 1:
        return 1
    return max(2, min(requested, length))


def _warn(message: str) -> None:
    print(f"Warning: {message}")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def _require(df: pd.DataFrame, fields: list[str], figure_name: str) -> bool:
    missing = [f for f in fields if f not in df.columns]
    if missing:
        _warn(f"skip {figure_name}; missing columns: {missing}")
        return False
    return True


def generate_training_figures(input_csv: str | Path, output_dir: str | Path, ma_window: int = 20) -> list[Path]:
    """Generate training diagnostic figures from training_curve.csv."""
    setup_style()
    input_csv = Path(input_csv)
    out = ensure_output(output_dir)
    df = _read_training_csv(input_csv)
    if df.empty:
        raise ValueError(f"training curve CSV is empty: {input_csv}")
    window = _ma_window(len(df), ma_window)
    saved: list[Path] = []

    if _require(df, ["episode", "train_reward"], "training_reward_curve.png"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["episode"], df["train_reward"], lw=0.9, alpha=0.45, label="raw")
        ax.plot(df["episode"], df["train_reward"].rolling(window, min_periods=1).mean(), lw=2.0, label=f"MA{window}")
        ax.set_title("Training Reward Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Train reward")
        ax.legend(frameon=False)
        path = out / "training_reward_curve.png"
        _save(fig, path)
        saved.append(path)

    if _require(df, ["episode", "train_cmax"], "training_cmax_curve.png"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["episode"], df["train_cmax"], lw=0.9, alpha=0.45, label="raw")
        ax.plot(df["episode"], df["train_cmax"].rolling(window, min_periods=1).mean(), lw=2.0, label=f"MA{window}")
        ax.set_title("Training Cmax Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Train Cmax")
        ax.legend(frameon=False)
        path = out / "training_cmax_curve.png"
        _save(fig, path)
        saved.append(path)

    if "episode" not in df.columns:
        _warn("skip eval_cmax_curve.png; missing columns: ['episode']")
    elif "eval_cmax_mean" not in df.columns and "last_eval_cmax_mean" not in df.columns:
        _warn("skip eval_cmax_curve.png; missing eval_cmax_mean and last_eval_cmax_mean")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = False
        if "eval_cmax_mean" in df.columns:
            eval_df = df.dropna(subset=["eval_cmax_mean"])
            eval_df = eval_df[eval_df["eval_cmax_mean"].astype(str).str.len() > 0]
            if not eval_df.empty:
                ax.plot(eval_df["episode"], pd.to_numeric(eval_df["eval_cmax_mean"]), marker="o", lw=1.8, label="eval")
                plotted = True
        if "last_eval_cmax_mean" in df.columns:
            last_df = df.dropna(subset=["last_eval_cmax_mean"])
            last_df = last_df[last_df["last_eval_cmax_mean"].astype(str).str.len() > 0]
            if not last_df.empty:
                ax.step(last_df["episode"], pd.to_numeric(last_df["last_eval_cmax_mean"]), where="post", lw=1.2, alpha=0.65, label="last eval")
                plotted = True
        if plotted:
            ax.set_title("Fixed Validation Cmax Curve")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Eval Cmax")
            ax.legend(frameon=False)
            path = out / "eval_cmax_curve.png"
            _save(fig, path)
            saved.append(path)
        else:
            plt.close(fig)
            _warn("skip eval_cmax_curve.png; no non-empty eval values")

    loss_cols = [c for c in ["policy_loss", "value_loss", "total_loss"] if c in df.columns]
    if "episode" not in df.columns or not loss_cols:
        _warn("skip loss_curve.png; missing episode or loss columns")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in loss_cols:
            ax.plot(df["episode"], pd.to_numeric(df[col], errors="coerce"), lw=1.4, label=col)
        ax.set_title("PPO Loss Curves")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        path = out / "loss_curve.png"
        _save(fig, path)
        saved.append(path)

    if _require(df, ["episode", "entropy"], "entropy_curve.png"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["episode"], pd.to_numeric(df["entropy"], errors="coerce"), lw=1.6)
        ax.set_title("Policy Entropy Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Entropy")
        path = out / "entropy_curve.png"
        _save(fig, path)
        saved.append(path)

    if "learning_rate" not in df.columns:
        _warn("skip learning_rate_curve.png; missing learning_rate")
    elif "episode" not in df.columns:
        _warn("skip learning_rate_curve.png; missing episode")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["episode"], pd.to_numeric(df["learning_rate"], errors="coerce"), lw=1.6)
        ax.set_title("Learning Rate Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Learning rate")
        path = out / "learning_rate_curve.png"
        _save(fig, path)
        saved.append(path)

    if not saved:
        _warn("no figures were generated; please check CSV columns")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training diagnostic figures.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ma_window", type=int, default=20)
    args = parser.parse_args()
    generate_training_figures(args.input, args.output, args.ma_window)


if __name__ == "__main__":
    main()
