from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import read_json
from .common import ensure_output, setup_style


def _load_instances(path: Path) -> list[dict]:
    obj = read_json(path)
    if isinstance(obj, dict) and "test_instances" in obj:
        return obj["test_instances"]
    return obj


def plot_heatmap(instances: list[dict], out: Path) -> None:
    inst = instances[0]
    jobs = inst["jobs"][: min(30, len(inst["jobs"]))]
    machines = [m["machine_id"] for m in inst["machines"]]
    mat = np.full((len(jobs), len(machines)), np.nan)
    for i, job in enumerate(jobs):
        for j, machine in enumerate(machines):
            if machine in job["processing_times"]:
                mat[i, j] = job["processing_times"][machine]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_title("任务-机器加工时间热力图")
    ax.set_xlabel("机器")
    ax.set_ylabel("任务")
    ax.set_xticks(range(len(machines)))
    ax.set_xticklabels(machines, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(jobs)))
    ax.set_yticklabels([j["job_id"].split("_")[-1] for j in jobs], fontsize=7)
    fig.colorbar(im, ax=ax, label="p[j,k]")
    fig.tight_layout()
    fig.savefig(out / "processing_time_heatmap.png", bbox_inches="tight")


def plot_rank_distribution(raw: pd.DataFrame, out: Path) -> None:
    tmp = raw.assign(rank_group=np.where(raw["machine_rank"] == 1, "fastest", np.where(raw["machine_rank"] == 2, "second-fastest", "others")))
    tab = pd.crosstab(tmp["algorithm"], tmp["rank_group"], normalize="index")
    tab.reindex(columns=["fastest", "second-fastest", "others"], fill_value=0).plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.title("机器选择排名分布")
    plt.ylabel("比例")
    plt.tight_layout()
    plt.savefig(out / "machine_rank_distribution.png", bbox_inches="tight")


def plot_mismatch(raw: pd.DataFrame, out: Path) -> None:
    stat = raw.groupby("algorithm")["machine_mismatch"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(stat["algorithm"], stat["mean"], yerr=stat["std"], capsize=3)
    ax.set_title("机器适配损失对比")
    ax.set_ylabel("machine_mismatch")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "machine_mismatch_comparison.png", bbox_inches="tight")


def plot_efficiency_load(raw: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alg, g in raw.groupby("algorithm"):
        ax.scatter(g["p_selected"], g["available_time_before"], s=12, alpha=0.45, label=alg)
    ax.set_xlabel("p_selected")
    ax.set_ylabel("available_time_before")
    ax.set_title("效率-负载权衡散点图")
    ax.legend(frameon=False, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out / "efficiency_load_tradeoff.png", bbox_inches="tight")


def plot_heterogeneity_improvement(summary: pd.DataFrame, out: Path) -> None:
    if "heterogeneity" not in summary.columns:
        return
    rows = []
    for level, g in summary.groupby("heterogeneity"):
        tsr = g.loc[g["algorithm"].eq("TSR-PPO"), "cmax_mean"]
        if tsr.empty:
            continue
        tsr_val = float(tsr.iloc[0])
        for base in ["FIFO", "SPT", "GA"]:
            b = g.loc[g["algorithm"].eq(base), "cmax_mean"]
            if not b.empty:
                rows.append({"heterogeneity": level, "baseline": base, "improvement": (float(b.iloc[0]) - tsr_val) / float(b.iloc[0]) * 100})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for base, g in df.groupby("baseline"):
        ax.plot(g["heterogeneity"], g["improvement"], marker="o", label=base)
    ax.set_title("异构程度下 TSR-PPO 相对改进率")
    ax.set_ylabel("Improvement/%")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "heterogeneity_improvement.png", bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="raw_schedule_results.csv")
    parser.add_argument("--instances", type=Path, required=True, help="test_instances.json 或 dataset json")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    out = ensure_output(args.output)
    raw = pd.read_csv(args.input)
    instances = _load_instances(args.instances)
    plot_heatmap(instances, out)
    plot_rank_distribution(raw, out)
    plot_mismatch(raw, out)
    plot_efficiency_load(raw, out)
    if args.summary and args.summary.exists():
        plot_heterogeneity_improvement(pd.read_csv(args.summary), out)


if __name__ == "__main__":
    main()
