from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="生成消融实验运行计划，不伪造结果")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("results/ablation_plan"))
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    variants = ["PPO", "PPO+Mask", "PPO+Mask+Split", "PPO+Mask+Split+Rolling", "TSR-PPO full"]
    commands = [
        f"python -m src.experiments.train --episodes {args.episodes} --scale {args.scale} --seed {args.seed} --variant \"{v.replace(' full', '')}\" --output_dir {output_dir / v.replace(' ', '_').replace('+', '_')}"
        for v in variants
    ]
    write_json({"variants": variants, "commands": commands}, output_dir / "ablation_plan.json")
    print(f"ablation plan saved to {output_dir / 'ablation_plan.json'}")


if __name__ == "__main__":
    main()
