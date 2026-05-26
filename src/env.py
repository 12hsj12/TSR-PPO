from __future__ import annotations

from itertools import combinations

import numpy as np

from config import EnvConfig, PROCESS_TYPES
from metrics import load_balance
from splitter import decode_split


class RollingSchedulingEnv:
    """任务拆分、非等效并行产线与滚动到达调度环境。"""

    def __init__(self, instance: dict, cfg: EnvConfig | None = None):
        self.instance = instance
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.lines = list(instance["lines"])
        self.jobs = list(instance["jobs"])
        self.line_by_process: dict[str, list[str]] = {}
        for line in self.lines:
            self.line_by_process.setdefault(line["process"], []).append(line["line_id"])
        self.reset()

    def reset(self) -> np.ndarray:
        self.done_jobs: set[str] = set()
        self.line_available = {line["line_id"]: 0.0 for line in self.lines}
        self.line_load = {line["line_id"]: 0.0 for line in self.lines}
        self.line_count = {line["line_id"]: 0 for line in self.lines}
        self.schedule: list[dict] = []
        self.tau = 0.0
        self.period = 1
        self.current_cmax = 0.0
        self._advance_to_next_release()
        return self.state_vector()

    def clone(self) -> "RollingSchedulingEnv":
        other = RollingSchedulingEnv(self.instance, self.cfg)
        other.done_jobs = set(self.done_jobs)
        other.line_available = dict(self.line_available)
        other.line_load = dict(self.line_load)
        other.line_count = dict(self.line_count)
        other.schedule = [dict(x) for x in self.schedule]
        other.tau = self.tau
        other.period = self.period
        other.current_cmax = self.current_cmax
        return other

    def _advance_to_next_release(self) -> None:
        while not self.available_jobs() and len(self.done_jobs) < len(self.jobs):
            future = [j["release_time"] for j in self.jobs if j["job_id"] not in self.done_jobs]
            self.tau = max(self.tau, min(future))
            self.period = int(self.tau // self.cfg.rolling_delta) + 1

    def available_jobs(self) -> list[dict]:
        return [
            j for j in self.jobs
            if j["job_id"] not in self.done_jobs and float(j["release_time"]) <= self.tau + 1e-9
        ]

    def _combos_for_job(self, job: dict) -> list[tuple[str, ...]]:
        lines = self.line_by_process.get(job["process"], [])
        limit = min(int(job.get("max_split", self.cfg.max_split)), self.cfg.max_split, len(lines))
        all_combos: list[tuple[str, ...]] = []
        for r in range(1, limit + 1):
            all_combos.extend(tuple(c) for c in combinations(lines, r))
        # 保留代表性组合，兼顾最早可用与最快产线，控制动作空间。
        def score(combo: tuple[str, ...]) -> float:
            return min(job["p_times"][k] for k in combo) + 0.1 * np.mean([self.line_available[k] for k in combo])
        all_combos.sort(key=score)
        return all_combos[: self.cfg.max_combo_per_job]

    def valid_actions(self, use_mask: bool = True) -> list[tuple[str, tuple[str, ...]]]:
        source = self.available_jobs() if use_mask else [j for j in self.jobs if j["job_id"] not in self.done_jobs]
        actions = []
        for job in source:
            for combo in self._combos_for_job(job):
                if len(combo) <= min(job.get("max_split", self.cfg.max_split), self.cfg.max_split):
                    actions.append((job["job_id"], combo))
        return actions

    def action_features(self, actions: list[tuple[str, tuple[str, ...]]]) -> np.ndarray:
        feats = []
        job_map = {j["job_id"]: j for j in self.jobs}
        proc_map = {p: i for i, p in enumerate(PROCESS_TYPES)}
        loads = np.array(list(self.line_load.values()), dtype=float)
        for job_id, combo in actions:
            job = job_map[job_id]
            ps = np.array([job["p_times"][k] for k in combo], dtype=float)
            av = np.array([self.line_available[k] for k in combo], dtype=float)
            onehot = [1.0 if proc_map[job["process"]] == i else 0.0 for i in range(len(PROCESS_TYPES))]
            feats.append(
                onehot
                + [
                    job["release_time"] / 1000.0,
                    max(self.tau - job["release_time"], 0.0) / 1000.0,
                    job["base_time"] / 200.0,
                    len(combo) / max(1, self.cfg.max_split),
                    ps.min() / 200.0,
                    ps.mean() / 200.0,
                    av.min() / 1000.0,
                    av.mean() / 1000.0,
                    loads.std() / 500.0,
                    self.current_cmax / 1000.0,
                ]
            )
        return np.asarray(feats, dtype=float)

    def state_vector(self) -> np.ndarray:
        avail = self.available_jobs()
        loads = np.array(list(self.line_load.values()), dtype=float)
        line_avail = np.array(list(self.line_available.values()), dtype=float)
        process_counts = [sum(1 for j in avail if j["process"] == p) for p in PROCESS_TYPES]
        remaining = len(self.jobs) - len(self.done_jobs)
        scaled_counts = [x / max(len(self.jobs), 1) for x in process_counts]
        return np.array(
            scaled_counts
            + [
                len(avail) / max(len(self.jobs), 1),
                remaining / max(len(self.jobs), 1),
                self.period / 100.0,
                self.tau / 1000.0,
                line_avail.mean() / 1000.0,
                line_avail.std() / 1000.0,
                loads.mean() / 500.0,
                loads.std() / 500.0,
                self.current_cmax / 1000.0,
            ],
            dtype=float,
        )

    def step(self, action: tuple[str, tuple[str, ...]], use_split_decoder: bool = True):
        job_id, combo = action
        job = next(j for j in self.jobs if j["job_id"] == job_id)
        prev_cmax = self.current_cmax
        ratios = decode_split(job, combo, self.line_available, self.tau, self.cfg.min_split_ratio, use_split_decoder)
        starts = []
        finishes = []
        idle = 0.0
        for line_id, ratio in ratios.items():
            start = max(self.line_available[line_id], float(job["release_time"]), self.tau)
            duration = float(job["p_times"][line_id]) * float(ratio)
            finish = start + duration
            idle += max(0.0, start - self.line_available[line_id])
            self.line_available[line_id] = finish
            self.line_load[line_id] += duration
            self.line_count[line_id] += 1
            self.schedule.append(
                {
                    "job_id": job_id,
                    "process": job["process"],
                    "line_id": line_id,
                    "start": start,
                    "finish": finish,
                    "duration": duration,
                    "ratio": float(ratio),
                    "release_time": float(job["release_time"]),
                }
            )
            starts.append(start)
            finishes.append(finish)
        self.done_jobs.add(job_id)
        self.current_cmax = max(self.current_cmax, max(finishes, default=prev_cmax))
        if not self.available_jobs() and len(self.done_jobs) < len(self.jobs):
            next_release = min(j["release_time"] for j in self.jobs if j["job_id"] not in self.done_jobs)
            boundary = (int(self.tau // self.cfg.rolling_delta) + 1) * self.cfg.rolling_delta
            self.tau = max(self.tau, min(next_release, boundary))
            if self.tau >= boundary - 1e-9:
                self.period += 1
            self._advance_to_next_release()
        reward = -(
            self.current_cmax - prev_cmax
            + self.cfg.idle_penalty * idle
            + self.cfg.balance_penalty * load_balance(self.schedule, self.lines)
        )
        done = len(self.done_jobs) == len(self.jobs)
        if done:
            reward -= 0.05 * self.current_cmax
        return self.state_vector(), float(reward), done, {"cmax": self.current_cmax}
