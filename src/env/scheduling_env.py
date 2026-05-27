from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass
class EnvParams:
    seed: int = 42
    max_split: int = 2
    min_split_ratio: float = 0.05
    min_split_workload: float = 1.0
    coordination_cost: float = 2.0
    split_penalty: float = 0.015
    idle_penalty: float = 0.02
    balance_penalty: float = 0.03
    mismatch_penalty: float = 0.10
    illegal_action_penalty: float = 5.0
    max_candidate_combos: int = 32


class SchedulingEnv:
    """正式滚动调度环境。

    问题设定：每个任务只属于一种工艺；拆分只在该工艺的可加工非等效并行机内发生；
    滚动周期内可见任务由新到达任务和上周期未调度结转任务组成，已调度子任务固定执行。
    """

    def __init__(self, instance: dict[str, Any], params: EnvParams | None = None):
        self.instance = instance
        self.params = params or EnvParams()
        self.rng = np.random.default_rng(self.params.seed)
        self.reset(instance=instance, seed=self.params.seed)

    def reset(self, instance: dict[str, Any] | None = None, seed: int | None = None) -> dict:
        if instance is not None:
            self.instance = instance
        if seed is not None:
            self.params.seed = seed
            self.rng = np.random.default_rng(seed)
        self.jobs = sorted(self.instance["jobs"], key=lambda j: (j["release_time"], j["job_id"]))
        self.machines = self.instance["machines"]
        self.machine_by_id = {m["machine_id"]: m for m in self.machines}
        self.jobs_by_id = {j["job_id"]: j for j in self.jobs}
        self.rolling_delta = float(self.instance["rolling_delta"])
        self.current_period = 0
        self.current_time = 0.0
        self.machine_available = {m["machine_id"]: 0.0 for m in self.machines}
        self.machine_load = {m["machine_id"]: 0.0 for m in self.machines}
        self.completed_jobs: set[str] = set()
        self.frozen_jobs: set[str] = set()
        self.schedule: list[dict] = []
        self.illegal_actions = 0
        self.prev_cmax = 0.0
        self.lower_bound = self._lower_bound()
        self._advance_to_visible_period()
        return self.get_observation()

    def clone(self) -> "SchedulingEnv":
        other = SchedulingEnv(self.instance, self.params)
        other.current_period = self.current_period
        other.current_time = self.current_time
        other.machine_available = dict(self.machine_available)
        other.machine_load = dict(self.machine_load)
        other.completed_jobs = set(self.completed_jobs)
        other.frozen_jobs = set(self.frozen_jobs)
        other.schedule = [dict(x) for x in self.schedule]
        other.illegal_actions = self.illegal_actions
        other.prev_cmax = self.prev_cmax
        return other

    def _lower_bound(self) -> float:
        by_proc: dict[str, float] = {}
        for job in self.jobs:
            by_proc.setdefault(job["process_type"], 0.0)
            by_proc[job["process_type"]] += min(job["processing_times"].values())
        machine_counts: dict[str, int] = {}
        for m in self.machines:
            machine_counts[m["process_type"]] = machine_counts.get(m["process_type"], 0) + 1
        proc_lb = [load / max(machine_counts.get(proc, 1), 1) for proc, load in by_proc.items()]
        release_lb = max(j["release_time"] + min(j["processing_times"].values()) for j in self.jobs)
        return max(max(proc_lb, default=0.0), release_lb, 1.0)

    def _period_end(self) -> float:
        return (self.current_period + 1) * self.rolling_delta

    def _advance_to_visible_period(self) -> None:
        while not self.visible_jobs() and len(self.completed_jobs) < len(self.jobs):
            next_release = min(j["release_time"] for j in self.jobs if j["job_id"] not in self.completed_jobs)
            self.current_period = max(self.current_period + 1, int(next_release // self.rolling_delta))
            self.current_time = self.current_period * self.rolling_delta

    def visible_jobs(self) -> list[dict]:
        boundary = self._period_end()
        return [
            job
            for job in self.jobs
            if job["job_id"] not in self.completed_jobs and job["release_time"] <= boundary + 1e-9
        ]

    def _candidate_combos(self, job: dict) -> list[tuple[str, ...]]:
        max_split = min(int(job.get("max_split", self.params.max_split)), self.params.max_split, len(job["eligible_machines"]))
        combos: list[tuple[str, ...]] = []
        for r in range(1, max_split + 1):
            combos.extend(tuple(c) for c in combinations(job["eligible_machines"], r))

        def score(combo: tuple[str, ...]) -> float:
            p_mean = np.mean([job["processing_times"][m] for m in combo])
            wait = np.mean([max(self.machine_available[m] - self.current_time, 0.0) for m in combo])
            return p_mean + 0.15 * wait + self.params.coordination_cost * max(len(combo) - 1, 0)

        combos = [c for c in combos if self._split_feasible(job, c)]
        combos.sort(key=score)
        return combos[: self.params.max_candidate_combos]

    def _split_feasible(self, job: dict, combo: tuple[str, ...]) -> bool:
        if not combo or len(combo) > min(job.get("max_split", self.params.max_split), self.params.max_split):
            return False
        if any(m not in job["eligible_machines"] for m in combo):
            return False
        ratios = self.decode_split(job, combo)
        if not ratios:
            return False
        return all(r >= self.params.min_split_ratio and r * job["workload"] >= self.params.min_split_workload for r in ratios.values())

    def decode_split(self, job: dict, combo: tuple[str, ...], mode: str = "weighted") -> dict[str, float]:
        if len(combo) == 1 or mode == "single":
            return {combo[0]: 1.0}
        if mode == "equal":
            return {m: 1.0 / len(combo) for m in combo}
        weights = []
        for m in combo:
            p = job["processing_times"][m]
            wait = max(self.machine_available[m] - self.current_time, 0.0)
            weights.append((1.0 / p) * (1.0 / (1.0 + wait / max(self.rolling_delta, 1.0))))
        arr = np.asarray(weights, dtype=float)
        arr = arr / max(arr.sum(), 1e-12)
        kept = [(m, float(r)) for m, r in zip(combo, arr) if r >= self.params.min_split_ratio]
        if not kept:
            best = combo[int(np.argmax(arr))]
            return {best: 1.0}
        total = sum(r for _, r in kept)
        return {m: r / total for m, r in kept}

    def action_space(self) -> list[dict]:
        actions = []
        for job in self.visible_jobs():
            for combo in self._candidate_combos(job):
                actions.append({"job_id": job["job_id"], "machines": combo, "split_count": len(combo)})
        return actions

    def action_mask_debug(self) -> list[dict]:
        rows = []
        for job in self.jobs:
            for m in job["eligible_machines"]:
                reasons = []
                if job["job_id"] in self.completed_jobs:
                    reasons.append("completed")
                if job["release_time"] > self._period_end():
                    reasons.append("not_released_in_period")
                if self.machine_by_id[m]["process_type"] != job["process_type"]:
                    reasons.append("wrong_process")
                rows.append({"job_id": job["job_id"], "machine": m, "valid_singleton": not reasons, "reasons": reasons})
        return rows

    def get_observation(self) -> dict:
        actions = self.action_space()
        return {
            "state": self.state_vector(),
            "actions": actions,
            "action_features": self.action_features(actions),
            "mask_debug": self.action_mask_debug(),
            "period": self.current_period,
            "time": self.current_time,
        }

    def state_vector(self) -> np.ndarray:
        visible = self.visible_jobs()
        remaining = len(self.jobs) - len(self.completed_jobs)
        loads = np.array(list(self.machine_load.values()), dtype=float)
        avails = np.array(list(self.machine_available.values()), dtype=float)
        proc_counts = [sum(1 for j in visible if j["process_type"] == p) / max(len(self.jobs), 1) for p in ["横切", "纵剪", "复合剪"]]
        carry = sum(1 for j in visible if j["rolling_period"] < self.current_period) / max(len(self.jobs), 1)
        return np.array(
            proc_counts
            + [
                len(visible) / max(len(self.jobs), 1),
                remaining / max(len(self.jobs), 1),
                carry,
                self.current_period / max(self.instance["rolling_periods"], 1),
                self.current_time / self.lower_bound,
                avails.mean() / self.lower_bound,
                avails.std() / self.lower_bound,
                loads.mean() / self.lower_bound,
                loads.std() / self.lower_bound,
                self.prev_cmax / self.lower_bound,
            ],
            dtype=np.float32,
        )

    def action_features(self, actions: list[dict]) -> np.ndarray:
        feats = []
        for action in actions:
            job = self.jobs_by_id[action["job_id"]]
            combo = action["machines"]
            pvals = np.array([job["processing_times"][m] for m in combo], dtype=float)
            best = min(job["processing_times"].values())
            avails = np.array([self.machine_available[m] for m in combo], dtype=float)
            proc_onehot = [1.0 if job["process_type"] == p else 0.0 for p in ["横切", "纵剪", "复合剪"]]
            feats.append(
                proc_onehot
                + [
                    job["release_time"] / self.lower_bound,
                    max(self.current_time - job["release_time"], 0.0) / self.lower_bound,
                    job["workload"] / 200.0,
                    len(combo) / max(self.params.max_split, 1),
                    pvals.min() / self.lower_bound,
                    pvals.mean() / self.lower_bound,
                    (pvals.mean() - best) / max(best, 1e-9),
                    avails.min() / self.lower_bound,
                    avails.mean() / self.lower_bound,
                    np.std(list(self.machine_load.values())) / self.lower_bound,
                ]
            )
        return np.asarray(feats, dtype=np.float32)

    def step(self, action_index: int | None = None, action: dict | None = None, split_mode: str = "weighted") -> tuple[dict, float, bool, dict]:
        actions = self.action_space()
        if action is None:
            if action_index is None or action_index < 0 or action_index >= len(actions):
                self.illegal_actions += 1
                return self.get_observation(), -self.params.illegal_action_penalty, False, {"illegal": True}
            action = actions[action_index]
        if action not in actions:
            self.illegal_actions += 1
            return self.get_observation(), -self.params.illegal_action_penalty, False, {"illegal": True}

        job = self.jobs_by_id[action["job_id"]]
        combo = tuple(action["machines"])
        ratios = self.decode_split(job, combo, mode=split_mode)
        if not self._split_feasible(job, combo):
            self.illegal_actions += 1
            return self.get_observation(), -self.params.illegal_action_penalty, False, {"illegal": True}

        p_best = min(job["processing_times"].values())
        sorted_m = sorted(job["processing_times"], key=job["processing_times"].get)
        cmax_before = self.prev_cmax
        idle = 0.0
        mismatch_sum = 0.0
        completion_times = []
        for split_id, (machine_id, ratio) in enumerate(ratios.items(), start=1):
            available_before = self.machine_available[machine_id]
            start = max(available_before, job["release_time"], self.current_time)
            p_selected_full = float(job["processing_times"][machine_id])
            process_time = p_selected_full * ratio + self.params.coordination_cost * max(len(ratios) - 1, 0)
            finish = start + process_time
            idle += max(0.0, start - available_before)
            self.machine_available[machine_id] = finish
            self.machine_load[machine_id] += process_time
            mismatch = (p_selected_full - p_best) / max(p_best, 1e-9)
            mismatch_sum += mismatch
            completion_times.append(finish)
            self.schedule.append(
                {
                    "instance_id": self.instance["instance_id"],
                    "scale": self.instance["scale"],
                    "algorithm": "",
                    "job_id": job["job_id"],
                    "subjob_id": f"{job['job_id']}_S{split_id}",
                    "process_type": job["process_type"],
                    "selected_machine": machine_id,
                    "machine_rank": sorted_m.index(machine_id) + 1,
                    "p_selected": p_selected_full,
                    "p_best": p_best,
                    "machine_mismatch": mismatch,
                    "available_time_before": available_before,
                    "release_time": job["release_time"],
                    "start_time": start,
                    "completion_time": finish,
                    "split_id": split_id,
                    "split_ratio": ratio,
                    "rolling_period": self.current_period,
                    "is_carryover": job["rolling_period"] < self.current_period,
                    "is_new_arrival": job["rolling_period"] == self.current_period,
                    "runtime": 0.0,
                }
            )
        self.completed_jobs.add(job["job_id"])
        if min(completion_times) <= self._period_end():
            self.frozen_jobs.add(job["job_id"])
        self.prev_cmax = max(self.prev_cmax, max(completion_times))
        balance = np.std(list(self.machine_load.values()))
        delta_cmax = self.prev_cmax - cmax_before
        reward = -(
            delta_cmax
            + self.params.idle_penalty * idle
            + self.params.balance_penalty * balance
            + self.params.mismatch_penalty * mismatch_sum
            + self.params.split_penalty * max(len(ratios) - 1, 0) * job["workload"]
        ) / self.lower_bound
        done = len(self.completed_jobs) == len(self.jobs)
        if done:
            reward -= self.prev_cmax / self.lower_bound
        self._advance_to_visible_period()
        return self.get_observation(), float(reward), done, {"cmax": self.prev_cmax, "illegal": False}

    def raw_schedule(self, algorithm: str, runtime: float) -> list[dict]:
        rows = []
        for row in self.schedule:
            out = dict(row)
            out["algorithm"] = algorithm
            out["runtime"] = runtime
            rows.append(out)
        return rows
