from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ensure_output, setup_style


PROCESS_COLORS = {"横切": "#1f77b4", "纵剪": "#2ca02c", "复合剪": "#d62728"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def generate_eval_diagnostic_figures(raw_csv: str | Path, output_dir: str | Path, rolling_delta: float = 120.0) -> list[Path]:
    setup_style()
    raw_csv = Path(raw_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(f"raw eval schedule CSV not found: {raw_csv}")
    out = ensure_output(output_dir)
    raw = pd.read_csv(raw_csv, encoding="utf-8")
    saved: list[Path] = []
    if raw.empty:
        print(f"Warning: skip eval diagnostic figures; empty raw schedule: {raw_csv}")
        return saved

    rank_group = np.where(raw["machine_rank"] == 1, "fastest", np.where(raw["machine_rank"] == 2, "second-fastest", "others"))
    tab = pd.Series(rank_group).value_counts(normalize=True).reindex(["fastest", "second-fastest", "others"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tab.index, tab.values, color=["#2ca02c", "#1f77b4", "#7f7f7f"])
    ax.set_title("Machine Rank Distribution")
    ax.set_ylabel("Ratio")
    path = out / "machine_rank_distribution.png"
    _save(fig, path)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw["machine_mismatch"], bins=24, color="#9467bd", alpha=0.78)
    ax.set_title("Machine Mismatch Distribution")
    ax.set_xlabel("machine_mismatch")
    ax.set_ylabel("Count")
    path = out / "machine_mismatch_distribution.png"
    _save(fig, path)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(raw["p_selected"], raw["available_time_before"], s=14, alpha=0.45)
    ax.set_title("Efficiency-Load Tradeoff")
    ax.set_xlabel("p_selected")
    ax.set_ylabel("available_time_before")
    path = out / "efficiency_load_tradeoff.png"
    _save(fig, path)
    saved.append(path)

    first_instance = raw["instance_id"].iloc[0]
    df = raw[raw["instance_id"] == first_instance].copy()
    machines = list(df["selected_machine"].drop_duplicates())
    ypos = {machine: idx for idx, machine in enumerate(machines)}
    fig, ax = plt.subplots(figsize=(12, max(4, len(machines) * 0.35)))
    for _, row in df.iterrows():
        ax.barh(
            ypos[row["selected_machine"]],
            row["completion_time"] - row["start_time"],
            left=row["start_time"],
            color=PROCESS_COLORS.get(row["process_type"], "#7f7f7f"),
            edgecolor="black",
            linewidth=0.25,
            hatch="//" if bool(row["is_carryover"]) else "",
            alpha=0.78 if bool(row["is_new_arrival"]) else 0.55,
        )
    horizon = float(df["completion_time"].max()) if not df.empty else 0.0
    boundary = 0.0
    while boundary <= horizon + rolling_delta:
        ax.axvline(boundary, color="#444444", linestyle="--", linewidth=0.7)
        boundary += rolling_delta
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines, fontsize=8)
    ax.set_xlabel("Time")
    ax.set_title(f"Gantt Chart with Rolling Boundaries ({first_instance})")
    path = out / "gantt_eval_instance.png"
    _save(fig, path)
    saved.append(path)

    return saved
