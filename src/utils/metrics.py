from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


def compute_cmax(schedule: list[dict]) -> float:
    """Cmax 为所有子任务完成时刻的最大值。"""
    return float(max((row["completion_time"] for row in schedule), default=0.0))


def machine_loads(schedule: list[dict], machines: list[dict]) -> dict[str, float]:
    loads = {m["machine_id"]: 0.0 for m in machines}
    for row in schedule:
        loads[row["selected_machine"]] += float(row["completion_time"] - row["start_time"])
    return loads


def utilization(schedule: list[dict], machines: list[dict]) -> float:
    cmax = compute_cmax(schedule)
    if cmax <= 0:
        return 0.0
    loads = machine_loads(schedule, machines)
    return float(sum(loads.values()) / (len(machines) * cmax))


def load_balance_std(schedule: list[dict], machines: list[dict]) -> float:
    vals = np.array(list(machine_loads(schedule, machines).values()), dtype=float)
    return float(vals.std(ddof=0))


def summarize_raw_results(raw: pd.DataFrame, instances: list[dict], scale: str, seed: int) -> pd.DataFrame:
    rows = []
    machines_by_instance = {inst["instance_id"]: inst["machines"] for inst in instances}
    for (instance_id, alg), g in raw.groupby(["instance_id", "algorithm"]):
        sched = g.to_dict("records")
        machines = machines_by_instance[instance_id]
        rows.append(
            {
                "scale": scale,
                "algorithm": alg,
                "seed": seed,
                "instance_id": instance_id,
                "cmax": compute_cmax(sched),
                "utilization": utilization(sched, machines),
                "load_balance": load_balance_std(sched, machines),
                "avg_machine_mismatch": float(g["machine_mismatch"].mean()),
                "fastest_machine_ratio": float((g["machine_rank"] == 1).mean()),
                "second_fastest_ratio": float((g["machine_rank"] == 2).mean()),
                "others_ratio": float((g["machine_rank"] > 2).mean()),
                "runtime": float(g["runtime"].max()) if "runtime" in g else math.nan,
                "reschedule_count": int(g["is_carryover"].sum()),
                "avg_waiting_time": float((g["start_time"] - g.get("release_time", g["start_time"])).mean()),
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["scale", "algorithm", "seed"], as_index=False)
        .agg(
            cmax_mean=("cmax", "mean"),
            cmax_std=("cmax", "std"),
            utilization_mean=("utilization", "mean"),
            load_balance_std=("load_balance", "mean"),
            avg_machine_mismatch=("avg_machine_mismatch", "mean"),
            fastest_machine_ratio=("fastest_machine_ratio", "mean"),
            second_fastest_ratio=("second_fastest_ratio", "mean"),
            others_ratio=("others_ratio", "mean"),
            runtime=("runtime", "sum"),
            reschedule_count=("reschedule_count", "mean"),
            avg_waiting_time=("avg_waiting_time", "mean"),
        )
        .fillna(0.0)
    )
    return summary
