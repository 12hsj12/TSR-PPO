from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, PROCESS_TYPES, GeneratorConfig


def process_from_line(name: str) -> str:
    """从机组名称识别工艺类型；真实样本缺失的工艺在仿真时补足。"""
    text = str(name)
    if "复合" in text:
        return "复合剪"
    if "纵" in text:
        return "纵剪"
    return "横切"


def load_real_profile(raw_xlsx: Path) -> dict:
    if not raw_xlsx.exists():
        return _fallback_profile()
    df = pd.read_excel(raw_xlsx)
    df["process"] = df["机组"].map(process_from_line)
    start = pd.to_datetime(df["实际开始时间"])
    finish = pd.to_datetime(df["实际完成时间"])
    df["duration"] = ((finish - start).dt.total_seconds() / 60).clip(lower=5)
    profile: dict[str, dict] = {"processes": {}}
    for proc, g in df.groupby("process"):
        lines = []
        median = float(g["duration"].median())
        for line, lg in g.groupby("机组"):
            line_med = float(lg["duration"].median())
            speed = median / max(line_med, 1.0)
            lines.append({"line_id": str(line), "speed": speed, "samples": int(len(lg))})
        profile["processes"][proc] = {
            "duration_mean": float(g["duration"].mean()),
            "duration_std": float(g["duration"].std(ddof=0) or 10),
            "lines": lines,
        }
    # 真实数据主要是横切与少量复合剪，纵剪按相近钢卷工序补足。
    if "纵剪" not in profile["processes"]:
        base = profile["processes"].get("横切", {"duration_mean": 45, "duration_std": 25})
        profile["processes"]["纵剪"] = {
            "duration_mean": base["duration_mean"] * 0.85,
            "duration_std": base["duration_std"] * 0.8,
            "lines": [
                {"line_id": "纵剪1号线", "speed": 1.12, "samples": 0},
                {"line_id": "纵剪2号线", "speed": 0.95, "samples": 0},
                {"line_id": "纵剪3号线", "speed": 0.82, "samples": 0},
            ],
        }
    return profile


def _fallback_profile() -> dict:
    return {
        "processes": {
            "纵剪": {"duration_mean": 38, "duration_std": 18, "lines": [{"line_id": f"纵剪{i}号线", "speed": s, "samples": 0} for i, s in enumerate([1.2, 1.0, 0.85], 1)]},
            "横切": {"duration_mean": 47, "duration_std": 32, "lines": [{"line_id": f"横切{i}号线", "speed": s, "samples": 0} for i, s in enumerate([1.25, 1.1, 1.0, 0.9, 0.75], 1)]},
            "复合剪": {"duration_mean": 55, "duration_std": 35, "lines": [{"line_id": f"复合剪{i}号线", "speed": s, "samples": 0} for i, s in enumerate([1.05, 0.9], 1)]},
        }
    }


def _heterogeneity_scale(level: str) -> float:
    return {"low": 0.35, "medium": 0.75, "high": 1.25}.get(level, 0.75)


def _arrival_gap(level: str) -> float:
    return {"low": 18.0, "medium": 10.0, "high": 5.0}.get(level, 10.0)


def generate_instance(cfg: GeneratorConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    profile = load_real_profile(cfg.raw_xlsx)
    hscale = _heterogeneity_scale(cfg.heterogeneity)

    lines = []
    for proc in PROCESS_TYPES:
        pdata = profile["processes"][proc]
        for line in pdata["lines"]:
            speed = 1.0 + (float(line["speed"]) - 1.0) * hscale
            lines.append({"line_id": line["line_id"], "process": proc, "speed": max(0.35, speed)})

    proc_prob = np.array([0.22, 0.66, 0.12])
    proc_prob = proc_prob / proc_prob.sum()
    gaps = rng.exponential(_arrival_gap(cfg.arrival_intensity), size=cfg.n_jobs)
    releases = np.cumsum(gaps)
    releases -= releases.min()

    jobs = []
    for j in range(cfg.n_jobs):
        proc = str(rng.choice(PROCESS_TYPES, p=proc_prob))
        pdata = profile["processes"][proc]
        mean = max(pdata["duration_mean"], 8)
        std = max(pdata["duration_std"], 3)
        sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
        mu = np.log(mean) - 0.5 * sigma * sigma
        base = float(np.clip(rng.lognormal(mu, sigma), 8, 260))
        p_times = {}
        for line in lines:
            if line["process"] == proc:
                noise = rng.normal(1.0, 0.06)
                p_times[line["line_id"]] = float(max(3.0, base / line["speed"] * noise))
        jobs.append(
            {
                "job_id": f"J{j + 1:04d}",
                "process": proc,
                "release_time": float(releases[j]),
                "base_time": base,
                "max_split": int(cfg.split_limit),
                "p_times": p_times,
            }
        )
    return {
        "meta": {
            "seed": cfg.seed,
            "n_jobs": cfg.n_jobs,
            "split_limit": cfg.split_limit,
            "heterogeneity": cfg.heterogeneity,
            "arrival_intensity": cfg.arrival_intensity,
            "rolling_delta": cfg.rolling_delta,
        },
        "lines": lines,
        "jobs": jobs,
    }


def save_instance(instance: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "generated" / "instance_100.json")
    args = parser.parse_args()
    inst = generate_instance(GeneratorConfig(n_jobs=args.n_jobs, seed=args.seed))
    save_instance(inst, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
