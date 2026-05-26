from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import schedule_frame


def setup_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#D9D9D9"
    plt.rcParams["grid.linewidth"] = 0.6


PALETTE = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#FF7F0E", "#17BECF"]


def save_convergence(history: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(history["episode"], history["reward"], color=PALETTE[0], lw=1.8)
    axes[0].set_title("训练奖励收敛曲线")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.7)
    axes[1].plot(history["episode"], history["Cmax"], color=PALETTE[1], lw=1.8)
    axes[1].set_title("Cmax 收敛曲线")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cmax/min")
    axes[1].grid(True, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_reward.png", bbox_inches="tight")
    plt.close(fig)


def save_algorithm_bars(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(df["algorithm"], df["Cmax"], color=PALETTE[: len(df)])
    ax.set_ylabel("Cmax/min")
    ax.set_title("不同算法 Cmax 对比")
    ax.grid(axis="y", alpha=0.7)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "algorithm_cmax_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(df["algorithm"], df["Cmax"], color=["#7F7F7F", "#1F77B4", "#2CA02C", "#D62728"])
    ax.set_ylabel("Cmax/min")
    ax.set_title("TSR-PPO 消融实验")
    ax.grid(axis="y", alpha=0.7)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_cmax.png", bbox_inches="tight")
    plt.close(fig)


def save_sensitivity(df: pd.DataFrame, x_col: str, title: str, filename: str, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, (alg, g) in enumerate(df.groupby("algorithm")):
        ax.plot(g[x_col].astype(str), g["Cmax"], marker="o", lw=1.8, label=alg, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(x_col)
    ax.set_ylabel("Cmax/min")
    ax.set_title(title)
    ax.grid(True, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_gantt(schedule: list[dict], out_dir: Path) -> None:
    setup_style()
    df = schedule_frame(schedule)
    if df.empty:
        return
    lines = list(df["line_id"].drop_duplicates())
    ymap = {line: i for i, line in enumerate(lines)}
    colors = {p: PALETTE[i] for i, p in enumerate(sorted(df["process"].unique()))}
    fig, ax = plt.subplots(figsize=(10, max(4, len(lines) * 0.28)))
    for _, row in df.iterrows():
        ax.barh(
            ymap[row["line_id"]],
            row["finish"] - row["start"],
            left=row["start"],
            height=0.72,
            color=colors[row["process"]],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_yticks(range(len(lines)))
    ax.set_yticklabels(lines, fontsize=8)
    ax.set_xlabel("时间/min")
    ax.set_title("TSR-PPO 调度甘特图")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors.keys()), frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.65)
    fig.tight_layout()
    fig.savefig(out_dir / "gantt_tsr_ppo.png", bbox_inches="tight")
    plt.close(fig)


def save_util_balance(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(8.0, 4.2))
    ax1.bar(x - 0.18, df["utilization"], width=0.36, color="#1F77B4", label="产线利用率")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, df["load_balance"], width=0.36, color="#D62728", label="负载均衡度")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["algorithm"], rotation=20)
    ax1.set_ylabel("利用率")
    ax2.set_ylabel("负载标准差")
    ax1.set_title("产线利用率与负载均衡对比")
    ax1.grid(axis="y", alpha=0.6)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "utilization_balance.png", bbox_inches="tight")
    plt.close(fig)
