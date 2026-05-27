from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.agents.tsr_ppo import TSRPPOAgent, TSRPPOConfig
from src.baselines import run_eat, run_fifo, run_ga, run_spt
from src.data.dataset import load_dataset
from src.env.scheduling_env import EnvParams, SchedulingEnv
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import summarize_raw_results
from src.utils.seed import set_global_seed


def _agent_from_checkpoint(checkpoint: Path, instance: dict, device: str, seed: int) -> TSRPPOAgent:
    env = SchedulingEnv(instance, EnvParams(seed=seed))
    obs = env.get_observation()
    agent = TSRPPOAgent(len(obs["state"]), obs["action_features"].shape[1], TSRPPOConfig(), device=device, seed=seed)
    agent.load_checkpoint(checkpoint)
    return agent


def run_agent(instance: dict, agent: TSRPPOAgent, name: str, seed: int) -> list[dict]:
    env = SchedulingEnv(instance, EnvParams(seed=seed, max_split=instance["split_limit"]))
    obs = env.get_observation()
    done = False
    start = time.perf_counter()
    while not done:
        action_idx, _ = agent.select_action(obs, deterministic=True)
        obs, _, done, _ = env.step(action_index=action_idx)
    runtime = time.perf_counter() - start
    return env.raw_schedule(name, runtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同一 test_instances 上评估所有算法")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ppo_checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("results/eval_medium"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--include_ga", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    dataset_dir = args.dataset_dir or Path("data/generated") / f"formal_{args.scale}_seed{args.seed}"
    dataset = load_dataset(dataset_dir, auto_generate=True, scale=args.scale, seed=args.seed)
    test_instances = dataset["test_instances"]
    raw_rows: list[dict] = []
    for inst_idx, instance in enumerate(test_instances):
        for runner in [run_fifo, run_spt, run_eat]:
            rows, _ = runner(instance)
            raw_rows.extend(rows)
        if args.include_ga:
            rows, _ = run_ga(instance, seed=args.seed + inst_idx)
            raw_rows.extend(rows)
    if args.checkpoint:
        agent = _agent_from_checkpoint(args.checkpoint, test_instances[0], args.device, args.seed)
        for inst_idx, instance in enumerate(test_instances):
            raw_rows.extend(run_agent(instance, agent, "TSR-PPO", args.seed + inst_idx))
    if args.ppo_checkpoint:
        agent = _agent_from_checkpoint(args.ppo_checkpoint, test_instances[0], args.device, args.seed)
        for inst_idx, instance in enumerate(test_instances):
            raw_rows.extend(run_agent(instance, agent, "PPO", args.seed + 1000 + inst_idx))
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output_dir / "raw_schedule_results.csv", index=False, encoding="utf-8-sig")
    summary = summarize_raw_results(raw, test_instances, args.scale, args.seed)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "scale": args.scale,
            "seed": args.seed,
            "dataset_dir": str(dataset_dir),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "ppo_checkpoint": str(args.ppo_checkpoint) if args.ppo_checkpoint else None,
        },
        output_dir / "experiment_config.json",
    )


if __name__ == "__main__":
    main()
