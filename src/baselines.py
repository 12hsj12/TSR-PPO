from __future__ import annotations

import time

import numpy as np

from config import EnvConfig
from env import RollingSchedulingEnv
from metrics import summarize_schedule


def _best_combo(env: RollingSchedulingEnv, job: dict, mode: str):
    combos = env._combos_for_job(job)
    if mode == "EAT":
        return min(combos, key=lambda c: np.mean([env.line_available[k] for k in c]))
    return min(combos, key=lambda c: np.mean([job["p_times"][k] for k in c]))


def dispatch(instance: dict, rule: str):
    env = RollingSchedulingEnv(instance, EnvConfig(max_split=instance["meta"]["split_limit"], rolling_delta=instance["meta"]["rolling_delta"]))
    start = time.perf_counter()
    while len(env.done_jobs) < len(env.jobs):
        jobs = env.available_jobs()
        if rule == "FIFO":
            job = min(jobs, key=lambda j: (j["release_time"], j["job_id"]))
        elif rule == "SPT":
            job = min(jobs, key=lambda j: min(j["p_times"].values()))
        elif rule == "EAT":
            job = min(jobs, key=lambda j: min(env.line_available[k] for k in env.line_by_process[j["process"]]))
        else:
            raise ValueError(rule)
        combo = _best_combo(env, job, "EAT" if rule == "EAT" else "SPT")
        env.step((job["job_id"], combo), use_split_decoder=True)
    runtime = time.perf_counter() - start
    return summarize_schedule(rule, env.schedule, instance["lines"], runtime), env.schedule


def run_ga(instance: dict, seed: int = 42, generations: int = 25, population: int = 18):
    rng = np.random.default_rng(seed)
    base_jobs = [j["job_id"] for j in sorted(instance["jobs"], key=lambda x: x["release_time"])]

    def evaluate(order):
        env = RollingSchedulingEnv(instance, EnvConfig(max_split=instance["meta"]["split_limit"], rolling_delta=instance["meta"]["rolling_delta"]))
        pending = list(order)
        while pending:
            available_ids = {j["job_id"] for j in env.available_jobs()}
            pick_idx = next((i for i, jid in enumerate(pending) if jid in available_ids), None)
            if pick_idx is None:
                env._advance_to_next_release()
                continue
            jid = pending.pop(pick_idx)
            job = next(j for j in env.jobs if j["job_id"] == jid)
            combo = _best_combo(env, job, "SPT")
            env.step((jid, combo), use_split_decoder=True)
        return env.current_cmax, env.schedule

    pop = []
    pop.append(base_jobs)
    for _ in range(population - 1):
        perm = base_jobs[:]
        rng.shuffle(perm)
        pop.append(perm)
    start = time.perf_counter()
    best_order = pop[0]
    best_score, best_schedule = evaluate(best_order)
    for _ in range(generations):
        scored = sorted((evaluate(ind)[0], ind) for ind in pop)
        if scored[0][0] < best_score:
            best_score, best_order = scored[0]
            _, best_schedule = evaluate(best_order)
        elites = [ind for _, ind in scored[: max(2, population // 4)]]
        new_pop = elites[:]
        while len(new_pop) < population:
            a, b = rng.choice(len(elites), size=2, replace=True)
            p1, p2 = elites[int(a)], elites[int(b)]
            cut1, cut2 = sorted(rng.choice(len(base_jobs), size=2, replace=False))
            child = [None] * len(base_jobs)
            child[cut1:cut2] = p1[cut1:cut2]
            fill = [x for x in p2 if x not in child]
            pos = 0
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = fill[pos]
                    pos += 1
            if rng.random() < 0.25:
                i, j = rng.choice(len(child), size=2, replace=False)
                child[int(i)], child[int(j)] = child[int(j)], child[int(i)]
            new_pop.append(child)
        pop = new_pop
    runtime = time.perf_counter() - start
    return summarize_schedule("GA", best_schedule, instance["lines"], runtime), best_schedule
