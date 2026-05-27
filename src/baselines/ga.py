from __future__ import annotations

import time

import numpy as np

from src.baselines.common import earliest_completion_score
from src.env.scheduling_env import EnvParams, SchedulingEnv


def _simulate(instance: dict, priority: list[str], split_bias: float, params: EnvParams) -> tuple[float, list[dict]]:
    env = SchedulingEnv(instance, params)
    order = {job_id: i for i, job_id in enumerate(priority)}
    done = False
    while not done:
        actions = env.action_space()
        def score(action):
            job_order = order[action["job_id"]]
            split_term = -split_bias * len(action["machines"])
            return (job_order, split_term, earliest_completion_score(env, action))
        idx = min(enumerate(actions), key=lambda x: score(x[1]))[0]
        _, _, done, _ = env.step(action_index=idx)
    return env.prev_cmax, env.schedule


def run_ga(instance: dict, seed: int = 42, population: int = 32, generations: int = 60):
    """简化但真实可用的 GA：染色体为任务优先序 + 拆分偏好。"""
    rng = np.random.default_rng(seed)
    params = EnvParams(max_split=instance["split_limit"])
    job_ids = [j["job_id"] for j in sorted(instance["jobs"], key=lambda j: j["release_time"])]
    pop = []
    for _ in range(population):
        perm = job_ids[:]
        rng.shuffle(perm)
        pop.append((perm, float(rng.uniform(0.0, 1.0))))
    start = time.perf_counter()
    best_score = float("inf")
    best_schedule = []
    for _ in range(generations):
        scored = []
        for chrom in pop:
            score, sched = _simulate(instance, chrom[0], chrom[1], params)
            scored.append((score, chrom, sched))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_score:
            best_score, _, best_schedule = scored[0]
        elites = [chrom for _, chrom, _ in scored[: max(2, population // 5)]]
        new_pop = elites[:]
        while len(new_pop) < population:
            p1, p2 = rng.choice(len(elites), size=2, replace=True)
            a, b = elites[int(p1)], elites[int(p2)]
            cut1, cut2 = sorted(rng.choice(len(job_ids), size=2, replace=False))
            child = [None] * len(job_ids)
            child[cut1:cut2] = a[0][cut1:cut2]
            fill = [x for x in b[0] if x not in child]
            pos = 0
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = fill[pos]
                    pos += 1
            if rng.random() < 0.25:
                i, j = rng.choice(len(child), size=2, replace=False)
                child[int(i)], child[int(j)] = child[int(j)], child[int(i)]
            split_bias = float(np.clip((a[1] + b[1]) / 2 + rng.normal(0, 0.08), 0, 1))
            new_pop.append((child, split_bias))
        pop = new_pop
    runtime = time.perf_counter() - start
    for row in best_schedule:
        row["algorithm"] = "GA"
        row["runtime"] = runtime
    return best_schedule, {"cmax": best_score, "runtime": runtime}
