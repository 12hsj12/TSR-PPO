from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="生成正式实验命令清单，不直接启动长训练")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--output_dir", type=Path, default=Path("results/formal_plan"))
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    commands = []
    for scale in ["small", "medium", "large"]:
        for s in range(args.num_seeds):
            seed = args.seed + s
            commands.append(f"python -m src.experiments.train --episodes {args.episodes} --scale {scale} --seed {seed} --device cuda --output_dir results/formal_{scale}_seed{seed}")
    write_json({"episodes": args.episodes, "commands": commands}, output_dir / "formal_experiment_plan.json")
    print(f"formal plan saved to {output_dir / 'formal_experiment_plan.json'}")


if __name__ == "__main__":
    main()
