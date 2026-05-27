from __future__ import annotations

import argparse
from pathlib import Path

from src.data.generator import generate_dataset
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="准备敏感性实验数据与命令，不执行正式训练")
    parser.add_argument("--experiment", choices=["scale", "heterogeneity", "arrival_intensity", "rolling_delta", "split_limit"], required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("results/sensitivity_plan"))
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    levels = {
        "scale": ["small", "medium", "large"],
        "heterogeneity": ["low", "medium", "high"],
        "arrival_intensity": ["low", "medium", "high"],
        "rolling_delta": [60, 120, 240],
        "split_limit": [1, 2, 3],
    }[args.experiment]
    commands = []
    for level in levels:
        scale = level if args.experiment == "scale" else args.scale
        dataset_dir = output_dir / f"data_{args.experiment}_{level}"
        kwargs = {}
        if args.experiment == "heterogeneity":
            kwargs["heterogeneity"] = level
        if args.experiment == "arrival_intensity":
            kwargs["arrival_intensity"] = level
        if args.experiment == "split_limit":
            kwargs["split_limit"] = int(level)
        if args.experiment == "rolling_delta":
            kwargs["rolling_delta"] = float(level)
        generate_dataset(dataset_dir, scale=scale, seed=args.seed, **kwargs)
        commands.append(f"python -m src.experiments.train --episodes {args.episodes} --scale {scale} --seed {args.seed} --device cuda --dataset_dir {dataset_dir} --output_dir {output_dir / ('train_' + str(level))}")
    write_json({"experiment": args.experiment, "levels": levels, "commands": commands}, output_dir / "sensitivity_plan.json")
    print(f"sensitivity plan saved to {output_dir / 'sensitivity_plan.json'}")


if __name__ == "__main__":
    main()
