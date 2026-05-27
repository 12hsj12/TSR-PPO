from __future__ import annotations

from src.baselines.common import action_job, run_dispatch_rule


def run_spt(instance: dict):
    def selector(env):
        actions = env.action_space()
        ranked = sorted(
            enumerate(actions),
            key=lambda x: (min(action_job(env, x[1])["processing_times"][m] for m in x[1]["machines"]), len(x[1]["machines"])),
        )
        return ranked[0][0]

    return run_dispatch_rule(instance, "SPT", selector)
