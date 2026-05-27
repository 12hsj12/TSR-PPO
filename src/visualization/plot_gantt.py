from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import ensure_output, setup_style


COLORS = {"横切": "#1f77b4", "纵剪": "#2ca02c", "复合剪": "#d62728"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--algorithm", default="TSR-PPO")
    parser.add_argument("--instance_id", default=None)
    parser.add_argument("--rolling_delta", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    out = ensure_output(args.output)
    df = pd.read_csv(args.input)
    df = df[df["algorithm"] == args.algorithm]
    if args.instance_id:
        df = df[df["instance_id"] == args.instance_id]
    elif not df.empty:
        df = df[df["instance_id"] == df["instance_id"].iloc[0]]
    machines = list(df["selected_machine"].drop_duplicates())
    ypos = {m: i for i, m in enumerate(machines)}
    fig, ax = plt.subplots(figsize=(12, max(4, len(machines) * 0.35)))
    for _, row in df.iterrows():
        hatch = "//" if bool(row["is_carryover"]) else ""
        alpha = 0.75 if bool(row["is_new_arrival"]) else 0.55
        ax.barh(
            ypos[row["selected_machine"]],
            row["completion_time"] - row["start_time"],
            left=row["start_time"],
            color=COLORS.get(row["process_type"], "#7f7f7f"),
            edgecolor="black",
            linewidth=0.25,
            hatch=hatch,
            alpha=alpha,
        )
    horizon = float(df["completion_time"].max()) if not df.empty else 0.0
    boundary = 0.0
    while boundary <= horizon + args.rolling_delta:
        ax.axvline(boundary, color="#444444", linestyle="--", linewidth=0.7)
        boundary += args.rolling_delta
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines, fontsize=8)
    ax.set_xlabel("时间")
    ax.set_title("带滚动周期边界的甘特图")
    fig.tight_layout()
    fig.savefig(out / "rolling_gantt.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
