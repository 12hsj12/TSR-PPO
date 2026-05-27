from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_global_seed

PROCESS_TYPES = ["横切", "纵剪", "复合剪"]
PROCESS_PROBS = {"横切": 0.70, "纵剪": 0.20, "复合剪": 0.10}


@dataclass(frozen=True)
class ScaleSpec:
    num_jobs: int
    num_machines: int
    rolling_periods: int
    rolling_delta: float


SCALE_SPECS = {
    "small": ScaleSpec(num_jobs=50, num_machines=9, rolling_periods=4, rolling_delta=120.0),
    "medium": ScaleSpec(num_jobs=150, num_machines=15, rolling_periods=6, rolling_delta=120.0),
    "large": ScaleSpec(num_jobs=300, num_machines=21, rolling_periods=9, rolling_delta=120.0),
    "quick_debug": ScaleSpec(num_jobs=12, num_machines=6, rolling_periods=3, rolling_delta=80.0),
}


def _profile_from_real_xlsx(raw_xlsx: Path | None) -> dict[str, tuple[float, float]]:
    default = {"横切": (46.0, 28.0), "纵剪": (38.0, 20.0), "复合剪": (58.0, 30.0)}
    if raw_xlsx is None or not raw_xlsx.exists():
        return default
    try:
        df = pd.read_excel(raw_xlsx)
        start = pd.to_datetime(df["实际开始时间"])
        finish = pd.to_datetime(df["实际完成时间"])
        df["duration"] = ((finish - start).dt.total_seconds() / 60.0).clip(lower=5, upper=300)
        df["process"] = df["机组"].astype(str).map(lambda x: "复合剪" if "复合" in x else ("纵剪" if "纵" in x else "横切"))
        for proc, g in df.groupby("process"):
            default[proc] = (float(g["duration"].mean()), float(g["duration"].std(ddof=0) or default[proc][1]))
    except Exception:
        pass
    return default


def _machine_counts(total: int) -> dict[str, int]:
    counts = {
        "横切": max(3, int(round(total * 0.58))),
        "纵剪": max(2, int(round(total * 0.27))),
    }
    counts["复合剪"] = max(1, total - counts["横切"] - counts["纵剪"])
    while sum(counts.values()) > total:
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < total:
        counts["横切"] += 1
    return counts


def _speed_factors(rng: np.random.Generator, count: int, heterogeneity: str) -> np.ndarray:
    sigma = {"low": 0.08, "medium": 0.22, "high": 0.42}[heterogeneity]
    speeds = rng.lognormal(mean=0.0, sigma=sigma, size=count)
    speeds = speeds / np.mean(speeds)
    return np.clip(speeds, 0.35, 2.8)


def _arrival_times(rng: np.random.Generator, spec: ScaleSpec, arrival_intensity: str) -> np.ndarray:
    # high 表示单位时间到达更多任务，因此平均间隔更短，语义不反转。
    multipliers = {"low": 1.35, "medium": 1.0, "high": 0.62}
    horizon = spec.rolling_periods * spec.rolling_delta
    mean_gap = horizon / spec.num_jobs * multipliers[arrival_intensity]
    release = np.cumsum(rng.exponential(mean_gap, size=spec.num_jobs))
    release = release / max(release.max(), 1.0) * (horizon * 0.92)
    return np.sort(release)


def generate_instance(
    instance_id: str,
    scale: str,
    seed: int,
    heterogeneity: str = "medium",
    arrival_intensity: str = "medium",
    split_limit: int = 2,
    rolling_delta: float | None = None,
    raw_xlsx: Path | None = Path("data/raw/加工机组执行信息.xlsx"),
) -> dict:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    base_spec = SCALE_SPECS[scale]
    spec = ScaleSpec(
        num_jobs=base_spec.num_jobs,
        num_machines=base_spec.num_machines,
        rolling_periods=base_spec.rolling_periods,
        rolling_delta=float(rolling_delta or base_spec.rolling_delta),
    )
    profile = _profile_from_real_xlsx(raw_xlsx)

    machines = []
    for proc, count in _machine_counts(spec.num_machines).items():
        for idx, speed in enumerate(_speed_factors(rng, count, heterogeneity), start=1):
            machines.append(
                {
                    "machine_id": f"{proc}_M{idx:02d}",
                    "process_type": proc,
                    "speed_factor": float(speed),
                    "setup_bias": float(rng.uniform(0.92, 1.12)),
                }
            )

    releases = _arrival_times(rng, spec, arrival_intensity)
    proc_names = list(PROCESS_PROBS)
    proc_probs = np.array([PROCESS_PROBS[p] for p in proc_names], dtype=float)
    jobs = []
    for j in range(spec.num_jobs):
        proc = str(rng.choice(proc_names, p=proc_probs))
        mean, std = profile[proc]
        workload = float(np.clip(rng.lognormal(np.log(mean) - 0.5 * 0.35**2, 0.35), 8, 240))
        eligible = [m for m in machines if m["process_type"] == proc]
        pjk = {}
        for m in eligible:
            spec_fit = rng.uniform(0.82, 1.22)
            noise = rng.normal(1.0, {"low": 0.03, "medium": 0.07, "high": 0.12}[heterogeneity])
            pjk[m["machine_id"]] = float(max(2.0, workload * m["setup_bias"] * spec_fit / m["speed_factor"] * noise))
        release = float(releases[j])
        rolling_period = int(release // spec.rolling_delta)
        jobs.append(
            {
                "instance_id": instance_id,
                "job_id": f"{instance_id}_J{j + 1:04d}",
                "process_type": proc,
                "release_time": release,
                "due_time": release + float(np.mean(list(pjk.values()))) * rng.uniform(2.5, 4.5),
                "quantity": workload,
                "workload": workload,
                "eligible_machines": list(pjk.keys()),
                "processing_times": pjk,
                "rolling_period": rolling_period,
                "max_split": split_limit,
            }
        )

    return {
        "instance_id": instance_id,
        "scale": scale,
        "seed": seed,
        "heterogeneity": heterogeneity,
        "arrival_intensity": arrival_intensity,
        "split_limit": split_limit,
        "rolling_delta": spec.rolling_delta,
        "rolling_periods": spec.rolling_periods,
        "machines": machines,
        "jobs": jobs,
        "spec": asdict(spec),
    }


def generate_dataset(
    output_dir: Path,
    scale: str,
    seed: int,
    train_instances: int = 32,
    val_instances: int = 8,
    test_instances: int = 8,
    heterogeneity: str = "medium",
    arrival_intensity: str = "medium",
    split_limit: int = 2,
    rolling_delta: float | None = None,
) -> dict[str, list[dict]]:
    output_dir = ensure_dir(output_dir)
    dataset = {}
    for split, count, offset in [("train_instances", train_instances, 0), ("val_instances", val_instances, 10000), ("test_instances", test_instances, 20000)]:
        instances = [
            generate_instance(
                f"{scale}_{split}_{i:03d}",
                scale,
                seed + offset + i,
                heterogeneity=heterogeneity,
                arrival_intensity=arrival_intensity,
                split_limit=split_limit,
                rolling_delta=rolling_delta,
            )
            for i in range(count)
        ]
        dataset[split] = instances
        write_json(instances, output_dir / f"{split}.json")
    write_json(
        {
            "scale": scale,
            "seed": seed,
            "train_instances": train_instances,
            "val_instances": val_instances,
            "test_instances": test_instances,
            "heterogeneity": heterogeneity,
            "arrival_intensity": arrival_intensity,
            "split_limit": split_limit,
            "rolling_delta": rolling_delta,
        },
        output_dir / "dataset_config.json",
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("data/generated/formal_medium"))
    parser.add_argument("--scale", choices=list(SCALE_SPECS), default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_debug", action="store_true")
    args = parser.parse_args()
    scale = "quick_debug" if args.quick_debug else args.scale
    counts = (2, 1, 1) if args.quick_debug else (32, 8, 8)
    generate_dataset(args.output_dir, scale, args.seed, *counts)
    print(f"dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
