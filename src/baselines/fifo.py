from __future__ import annotations

from src.baselines.common import action_job, earliest_completion_score, run_dispatch_rule


def run_fifo(instance: dict):
    def selector(env):
        actions = env.action_space()
        ranked = sorted(
            enumerate(actions),
            key=lambda x: (action_job(env, x[1])["release_time"], action_job(env, x[1])["job_id"], earliest_completion_score(env, x[1])),
        )
        return ranked[0][0]

    return run_dispatch_rule(instance, "FIFO", selector)
