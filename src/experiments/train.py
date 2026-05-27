from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 正式环境应通过 requirements.txt 安装 tqdm
    class tqdm:  # type: ignore[override]
        def __init__(self, iterable, total=None, desc=None, unit=None):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, values):
            return None

from src.agents.tsr_ppo import TSRPPOAgent, TSRPPOConfig
from src.data.dataset import load_dataset
from src.env.scheduling_env import EnvParams, SchedulingEnv
from src.utils.io import ensure_dir, write_json
from src.utils.logger import CsvLogger
from src.utils.seed import set_global_seed


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    ckpts = sorted(checkpoint_dir.glob("checkpoint_ep*.pt"))
    return ckpts[-1] if ckpts else None


def rollout(agent: TSRPPOAgent, instance: dict, seed: int, deterministic: bool = False, split_mode: str = "weighted") -> tuple[list[dict], dict]:
    env = SchedulingEnv(instance, EnvParams(seed=seed, max_split=instance["split_limit"]))
    obs = env.get_observation()
    trajectory = []
    done = False
    total_reward = 0.0
    while not done:
        action_idx, info = agent.select_action(obs, deterministic=deterministic)
        next_obs, reward, done, _ = env.step(action_index=action_idx, split_mode=split_mode)
        if not deterministic:
            trajectory.append(
                {
                    "state": obs["state"],
                    "action_features": obs["action_features"],
                    "action": action_idx,
                    "log_prob": info["log_prob"],
                    "value": info["value"],
                    "reward": reward,
                    "done": done,
                }
            )
        total_reward += reward
        obs = next_obs
    return trajectory, {"reward": total_reward, "cmax": env.prev_cmax, "schedule": env.raw_schedule("TSR-PPO", 0.0)}


def evaluate(agent: TSRPPOAgent, instances: list[dict], seed: int) -> dict:
    cmax = []
    for i, inst in enumerate(instances):
        _, info = rollout(agent, inst, seed + i, deterministic=True)
        cmax.append(info["cmax"])
    return {"eval_cmax_mean": float(np.mean(cmax)), "eval_cmax_std": float(np.std(cmax, ddof=0))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="正式 TSR-PPO 训练入口")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick_debug", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--variant",
        choices=["PPO", "PPO+Mask", "PPO+Mask+Split", "PPO+Mask+Split+Rolling", "TSR-PPO"],
        default="TSR-PPO",
    )
    return parser.parse_args()


def torch_runtime_info(device: str) -> dict:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for training. Install CUDA PyTorch first, then requirements.txt.") from exc
    cuda_available = bool(torch.cuda.is_available())
    if device == "cuda" and not cuda_available:
        raise RuntimeError("Using --device cuda was requested, but torch.cuda.is_available() is False.")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "device": device,
    }


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    if args.quick_debug:
        return Path("results") / f"quick_debug_{args.device}"
    return Path("results") / f"formal_{args.scale}_seed{args.seed}"


def announce_start(args: argparse.Namespace, output_dir: Path, dataset_dir: Path, runtime: dict) -> None:
    mode = "quick_debug" if args.quick_debug else "formal"
    print(f"mode: {mode}")
    print(f"episodes: {args.episodes}")
    print(f"scale: {args.scale}")
    print(f"seed: {args.seed}")
    print(f"Using device: {runtime['device']}")
    if runtime["device"] == "cuda":
        print(f"GPU: {runtime['gpu_name']}")
        print(f"torch CUDA version: {runtime['torch_cuda_version']}")
    print(f"output_dir: {output_dir}")
    print(f"dataset_dir: {dataset_dir}")


def main() -> None:
    args = parse_args()
    explicit_output_dir = "--output_dir" in sys.argv
    if args.config:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)
    if args.quick_debug:
        args.episodes = min(args.episodes, 5)
        args.scale = "small"
        args.eval_interval = 1
        args.save_interval = 1
        if not explicit_output_dir:
            args.output_dir = None

    set_global_seed(args.seed)
    runtime_info = torch_runtime_info(args.device)
    output_dir = ensure_dir(resolve_output_dir(args))
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    dataset_dir = args.dataset_dir or Path("data/generated") / f"formal_{args.scale}_seed{args.seed}"
    announce_start(args, output_dir, dataset_dir, runtime_info)
    dataset = load_dataset(dataset_dir, auto_generate=True, scale=args.scale, seed=args.seed)
    write_json(vars(args) | {"output_dir": output_dir, "dataset_dir": dataset_dir, **runtime_info}, output_dir / "experiment_config.json")

    probe_env = SchedulingEnv(dataset["train_instances"][0], EnvParams(seed=args.seed))
    probe_obs = probe_env.get_observation()
    state_dim = len(probe_obs["state"])
    action_dim = probe_obs["action_features"].shape[1]
    variant_flags = {
        "PPO": dict(use_mask=False, use_split=False, use_rolling=False, use_machine_mismatch_feature=False),
        "PPO+Mask": dict(use_mask=True, use_split=False, use_rolling=False, use_machine_mismatch_feature=False),
        "PPO+Mask+Split": dict(use_mask=True, use_split=True, use_rolling=False, use_machine_mismatch_feature=False),
        "PPO+Mask+Split+Rolling": dict(use_mask=True, use_split=True, use_rolling=True, use_machine_mismatch_feature=False),
        "TSR-PPO": dict(use_mask=True, use_split=True, use_rolling=True, use_machine_mismatch_feature=True),
    }[args.variant]
    ppo_params = TSRPPOConfig(episodes=args.episodes, eval_interval=args.eval_interval, save_interval=args.save_interval, **variant_flags)
    agent = TSRPPOAgent(state_dim, action_dim, ppo_params, device=args.device, seed=args.seed)

    start_episode = 1
    if args.resume:
        ckpt = latest_checkpoint(checkpoint_dir)
        if ckpt:
            payload = agent.load_checkpoint(ckpt)
            start_episode = int(payload["episode"]) + 1

    logger = CsvLogger(
        output_dir / "training_curve.csv",
        [
            "episode",
            "train_reward",
            "train_cmax",
            "eval_cmax_mean",
            "eval_cmax_std",
            "last_eval_cmax_mean",
            "last_eval_cmax_std",
            "policy_loss",
            "value_loss",
            "entropy",
            "learning_rate",
        ],
    )
    train_instances = dataset["train_instances"]
    last_eval = {"last_eval_cmax_mean": None, "last_eval_cmax_std": None}
    progress = tqdm(
        range(start_episode, args.episodes + 1),
        total=max(args.episodes - start_episode + 1, 0),
        desc="Training",
        unit="episode",
    )
    for episode in progress:
        inst = train_instances[(episode - 1) % len(train_instances)]
        split_mode = "weighted" if ppo_params.use_split else "single"
        trajectory, train_info = rollout(agent, inst, args.seed + episode, deterministic=False, split_mode=split_mode)
        losses = agent.update(trajectory)
        eval_info = {"eval_cmax_mean": None, "eval_cmax_std": None}
        if episode % args.eval_interval == 0 or episode == 1:
            eval_info = evaluate(agent, dataset["val_instances"], args.seed + 5000)
            last_eval = {
                "last_eval_cmax_mean": eval_info["eval_cmax_mean"],
                "last_eval_cmax_std": eval_info["eval_cmax_std"],
            }
        logger.write(
            {
                "episode": episode,
                "train_reward": train_info["reward"],
                "train_cmax": train_info["cmax"],
                **eval_info,
                **last_eval,
                **losses,
            }
        )
        progress.set_postfix(
            {
                "episode": episode,
                "train_reward": f"{train_info['reward']:.3f}",
                "train_cmax": f"{train_info['cmax']:.2f}",
                "eval_cmax_mean": f"{last_eval['last_eval_cmax_mean']:.2f}" if last_eval["last_eval_cmax_mean"] is not None else "-",
                "policy_loss": f"{losses.get('policy_loss', 0.0):.4f}",
                "value_loss": f"{losses.get('value_loss', 0.0):.4f}",
                "entropy": f"{losses.get('entropy', 0.0):.4f}",
                "device": args.device,
            }
        )
        if episode % args.save_interval == 0 or episode == args.episodes:
            agent.save_checkpoint(checkpoint_dir / f"checkpoint_ep{episode:04d}.pt", episode=episode, seed=args.seed, extra={"scale": args.scale})

    print("Training finished successfully.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
