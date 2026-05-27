from __future__ import annotations

from src.baselines.common import earliest_completion_score, run_dispatch_rule


def run_eat(instance: dict):
    def selector(env):
        actions = env.action_space()
        ranked = sorted(
            enumerate(actions),
            key=lambda x: (
                min(env.machine_available[m] for m in x[1]["machines"]),
                earliest_completion_score(env, x[1]),
            ),
        )
        return ranked[0][0]

    return run_dispatch_rule(instance, "EAT", selector)


def run_ect(instance: dict):
    def selector(env):
        actions = env.action_space()
        return min(enumerate(actions), key=lambda x: earliest_completion_score(env, x[1]))[0]

    return run_dispatch_rule(instance, "ECT", selector)
