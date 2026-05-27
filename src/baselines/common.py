from __future__ import annotations

import time
from typing import Callable

from src.env.scheduling_env import EnvParams, SchedulingEnv


def run_dispatch_rule(instance: dict, algorithm: str, selector: Callable[[SchedulingEnv], int], params: EnvParams | None = None) -> tuple[list[dict], dict]:
    env = SchedulingEnv(instance, params or EnvParams(max_split=instance["split_limit"]))
    start = time.perf_counter()
    done = False
    while not done:
        idx = selector(env)
        _, _, done, _ = env.step(action_index=idx)
    runtime = time.perf_counter() - start
    return env.raw_schedule(algorithm, runtime), {"cmax": env.prev_cmax, "runtime": runtime}


def action_job(env: SchedulingEnv, action: dict) -> dict:
    return env.jobs_by_id[action["job_id"]]


def earliest_completion_score(env: SchedulingEnv, action: dict) -> float:
    job = action_job(env, action)
    best_finish = 0.0
    for m, ratio in env.decode_split(job, tuple(action["machines"])).items():
        start = max(env.machine_available[m], job["release_time"], env.current_time)
        best_finish = max(best_finish, start + job["processing_times"][m] * ratio)
    return best_finish
