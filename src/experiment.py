from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from baselines import dispatch, run_ga
from config import DATA_DIR, RESULTS_DIR, ExperimentConfig, GeneratorConfig
from data_generator import generate_instance, save_instance
from metrics import rpi, schedule_frame
from tsr_ppo import make_agent
from utils.io import make_json_serializable, write_json
from visualize import (
    save_ablation,
    save_algorithm_bars,
    save_convergence,
    save_gantt,
    save_sensitivity,
    save_util_balance,
)


def _mode_params(mode: str) -> tuple[int, int]:
    cfg = ExperimentConfig(mode=mode)
    if mode == "full":
        return cfg.full_jobs, cfg.full_episodes
    return cfg.quick_jobs, cfg.quick_episodes


def _instance(seed: int, n_jobs: int, split_limit: int = 2, heterogeneity: str = "medium", arrival: str = "high", delta: float = 120.0):
    return generate_instance(
        GeneratorConfig(
            seed=seed,
            n_jobs=n_jobs,
            split_limit=split_limit,
            heterogeneity=heterogeneity,
            arrival_intensity=arrival,
            rolling_delta=delta,
        )
    )


def train_agents(seed: int, n_jobs: int, episodes: int):
    def factory(ep: int):
        return _instance(seed + ep, n_jobs, split_limit=2)

    agents = {}
    for name in ["PPO", "PPO+Mask", "PPO+Mask+Split", "TSR-PPO"]:
        agent = make_agent(name, seed, episodes)
        agent.train(factory)
        agents[name] = agent
    return agents


def performance_table(instance: dict, agents: dict, seed: int):
    rows = []
    schedules = {}
    for rule in ["FIFO", "SPT", "EAT"]:
        row, sched = dispatch(instance, rule)
        rows.append(row)
        schedules[rule] = sched
    ga_row, ga_sched = run_ga(instance, seed=seed, generations=12 if len(instance["jobs"]) <= 80 else 22)
    rows.append(ga_row)
    schedules["GA"] = ga_sched
    for name in ["PPO", "TSR-PPO"]:
        row, sched = agents[name].evaluate(instance)
        rows.append(row)
        schedules[name] = sched
    df = pd.DataFrame(rows)
    best, worst = df["Cmax"].min(), df["Cmax"].max()
    df["RPI"] = [rpi(v, best, worst) for v in df["Cmax"]]
    return df, schedules


def ablation_table(instance: dict, agents: dict):
    rows = []
    schedules = {}
    for name in ["PPO", "PPO+Mask", "PPO+Mask+Split", "TSR-PPO"]:
        row, sched = agents[name].evaluate(instance)
        rows.append(row)
        schedules[name] = sched
    df = pd.DataFrame(rows)
    best, worst = df["Cmax"].min(), df["Cmax"].max()
    df["RPI"] = [rpi(v, best, worst) for v in df["Cmax"]]
    return df, schedules


def sensitivity(seed: int, agents: dict, quick: bool):
    rows = []
    scales = [50, 100, 200] if quick else [50, 100, 200, 500]
    for n in scales:
        inst = _instance(seed + n, n)
        for alg in ["FIFO", "SPT", "TSR-PPO"]:
            row, _ = dispatch(inst, alg) if alg != "TSR-PPO" else agents["TSR-PPO"].evaluate(inst)
            row["scenario"] = "规模"
            row["level"] = n
            rows.append(row)
    for m in [1, 2, 3]:
        inst = _instance(seed + 10 + m, 80 if quick else 160, split_limit=m)
        for alg in ["SPT", "TSR-PPO"]:
            row, _ = dispatch(inst, alg) if alg != "TSR-PPO" else agents["TSR-PPO"].evaluate(inst)
            row["scenario"] = "拆分上限"
            row["level"] = m
            rows.append(row)
    for h in ["low", "medium", "high"]:
        inst = _instance(seed + len(h), 80 if quick else 160, heterogeneity=h)
        for alg in ["SPT", "TSR-PPO"]:
            row, _ = dispatch(inst, alg) if alg != "TSR-PPO" else agents["TSR-PPO"].evaluate(inst)
            row["scenario"] = "异构程度"
            row["level"] = h
            rows.append(row)
    for delta in ([60, 120, 240] if quick else [30, 60, 120, 240]):
        inst = _instance(seed + int(delta), 80 if quick else 160, delta=float(delta))
        for alg in ["SPT", "TSR-PPO"]:
            row, _ = dispatch(inst, alg) if alg != "TSR-PPO" else agents["TSR-PPO"].evaluate(inst)
            row["scenario"] = "滚动步长"
            row["level"] = delta
            rows.append(row)
    for arr in ["low", "medium", "high"]:
        inst = _instance(seed + 100 + len(arr), 80 if quick else 160, arrival=arr)
        for alg in ["SPT", "TSR-PPO"]:
            row, _ = dispatch(inst, alg) if alg != "TSR-PPO" else agents["TSR-PPO"].evaluate(inst)
            row["scenario"] = "到达强度"
            row["level"] = arr
            rows.append(row)
    return pd.DataFrame(rows)


def run(mode: str) -> None:
    seed = 42
    n_jobs, episodes = _mode_params(mode)
    quick = mode != "full"
    fig_dir = RESULTS_DIR / "figures"
    tab_dir = RESULTS_DIR / "tables"
    model_dir = RESULTS_DIR / "models"
    for d in [DATA_DIR / "generated", fig_dir, tab_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base_instance = _instance(seed, n_jobs)
    save_instance(base_instance, DATA_DIR / "generated" / f"{mode}_base_instance.json")
    started = time.perf_counter()
    agents = train_agents(seed, max(30, n_jobs // 2) if quick else n_jobs, episodes)
    elapsed = time.perf_counter() - started

    history = pd.DataFrame(agents["TSR-PPO"].history)
    history.to_csv(tab_dir / "convergence_history.csv", index=False, encoding="utf-8")
    save_convergence(history, fig_dir)
    agents["TSR-PPO"].model.save(model_dir / "tsr_ppo_linear_actor_critic.npz")

    perf, schedules = performance_table(base_instance, agents, seed)
    perf.to_csv(tab_dir / "performance_comparison.csv", index=False, encoding="utf-8")
    save_algorithm_bars(perf, fig_dir)
    save_util_balance(perf, fig_dir)
    schedule_frame(schedules["TSR-PPO"]).to_csv(tab_dir / "tsr_ppo_schedule.csv", index=False, encoding="utf-8")
    save_gantt(schedules["TSR-PPO"], fig_dir)

    abl, _ = ablation_table(base_instance, agents)
    abl.to_csv(tab_dir / "ablation_results.csv", index=False, encoding="utf-8")
    save_ablation(abl, fig_dir)

    sens = sensitivity(seed, agents, quick)
    sens.to_csv(tab_dir / "sensitivity_results.csv", index=False, encoding="utf-8")
    save_sensitivity(sens[sens["scenario"] == "规模"], "level", "任务规模敏感性", "sensitivity_task_scale.png", fig_dir)
    save_sensitivity(sens[sens["scenario"] == "拆分上限"], "level", "拆分上限敏感性", "sensitivity_split_limit.png", fig_dir)
    save_sensitivity(sens[sens["scenario"] == "异构程度"], "level", "产线异构程度敏感性", "sensitivity_heterogeneity.png", fig_dir)
    save_sensitivity(sens[sens["scenario"] == "滚动步长"], "level", "滚动步长敏感性", "sensitivity_rolling_delta.png", fig_dir)
    save_sensitivity(sens[sens["scenario"] == "到达强度"], "level", "动态到达强度敏感性", "sensitivity_arrival_intensity.png", fig_dir)

    summary = {"mode": mode, "seed": seed, "jobs": n_jobs, "episodes": episodes, "training_runtime_sec": elapsed}
    write_json(summary, tab_dir / "run_summary.json")
    print(json.dumps(make_json_serializable(summary), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="TSR-PPO 滚动调度实验入口")
    parser.add_argument("--mode", choices=["quick", "full", "all"], default="quick")
    args = parser.parse_args()
    run("quick" if args.mode == "all" else args.mode)


if __name__ == "__main__":
    main()
