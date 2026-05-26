from __future__ import annotations

import numpy as np
import pandas as pd


def cmax(schedule: list[dict]) -> float:
    return float(max((row["finish"] for row in schedule), default=0.0))


def line_loads(schedule: list[dict], lines: list[dict]) -> dict[str, float]:
    loads = {line["line_id"]: 0.0 for line in lines}
    for row in schedule:
        loads[row["line_id"]] += float(row["duration"])
    return loads


def utilization(schedule: list[dict], lines: list[dict]) -> float:
    horizon = cmax(schedule)
    if horizon <= 0:
        return 0.0
    loads = line_loads(schedule, lines)
    return float(sum(loads.values()) / (len(lines) * horizon))


def load_balance(schedule: list[dict], lines: list[dict]) -> float:
    vals = np.array(list(line_loads(schedule, lines).values()), dtype=float)
    return float(vals.std(ddof=0))


def rpi(value: float, best: float, worst: float) -> float:
    if abs(worst - best) < 1e-9:
        return 0.0
    return float((value - best) / (worst - best) * 100.0)


def summarize_schedule(name: str, schedule: list[dict], lines: list[dict], runtime: float) -> dict:
    return {
        "algorithm": name,
        "Cmax": cmax(schedule),
        "utilization": utilization(schedule, lines),
        "load_balance": load_balance(schedule, lines),
        "runtime_sec": runtime,
    }


def schedule_frame(schedule: list[dict]) -> pd.DataFrame:
    cols = ["job_id", "process", "line_id", "start", "finish", "duration", "ratio", "release_time"]
    return pd.DataFrame(schedule, columns=cols)
