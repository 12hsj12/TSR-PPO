from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from config import EnvConfig, PPOConfig
from env import RollingSchedulingEnv
from metrics import summarize_schedule
from policy import LinearActorCritic


class TSRPPOAgent:
    """TSR-PPO/消融 PPO 的 NumPy 实现。

    该实现保留 PPO 的核心：旧策略 log-prob、概率比、clip 目标、价值函数更新。
    动作掩码和拆分解码通过配置开关控制，便于消融实验。
    """

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.model: LinearActorCritic | None = None
        self.history: list[dict] = []

    def _ensure_model(self, env: RollingSchedulingEnv) -> None:
        actions = env.valid_actions(use_mask=True)
        feat_dim = env.action_features(actions).shape[1]
        state_dim = env.state_vector().shape[0]
        if self.model is None:
            self.model = LinearActorCritic(feat_dim, state_dim, self.cfg.seed)

    def _select(self, env: RollingSchedulingEnv):
        assert self.model is not None
        actions = env.valid_actions(use_mask=self.cfg.use_action_mask)
        if not actions:
            actions = env.valid_actions(use_mask=True)
        feats = env.action_features(actions)
        probs = self.model.action_probs(feats)
        if not self.cfg.use_action_mask:
            # 无掩码 PPO：保留探索噪声，但环境仍只执行可行工艺组合。
            probs = 0.85 * probs + 0.15 / len(probs)
        idx = int(self.rng.choice(len(actions), p=probs))
        return actions[idx], idx, feats, float(np.log(probs[idx] + 1e-12))

    def rollout(self, instance: dict, train: bool = True):
        env = RollingSchedulingEnv(instance, EnvConfig(seed=self.cfg.seed, max_split=instance["meta"]["split_limit"], rolling_delta=instance["meta"]["rolling_delta"]))
        self._ensure_model(env)
        trajectory = []
        state = env.state_vector()
        total_reward = 0.0
        done = False
        while not done:
            action, idx, feats, old_logp = self._select(env)
            value = self.model.value(state) if self.model else 0.0
            next_state, reward, done, info = env.step(action, use_split_decoder=self.cfg.use_split_decoder)
            trajectory.append(
                {
                    "state": state,
                    "features": feats,
                    "action_idx": idx,
                    "old_logp": old_logp,
                    "reward": reward,
                    "value": value,
                }
            )
            total_reward += reward
            state = next_state
        if train:
            self._update(trajectory)
        return env.schedule, {"reward": total_reward, "cmax": env.current_cmax}

    def _returns(self, rewards: list[float]) -> np.ndarray:
        vals = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.cfg.gamma * g
            vals.append(g)
        return np.asarray(list(reversed(vals)), dtype=float) / 100.0

    def _update(self, trajectory: list[dict]) -> None:
        assert self.model is not None
        rewards = [x["reward"] for x in trajectory]
        returns = self._returns(rewards)
        values = np.asarray([x["value"] for x in trajectory])
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(self.cfg.update_epochs):
            for item, ret, a in zip(trajectory, returns, adv):
                feats = item["features"]
                probs = self.model.action_probs(feats)
                idx = item["action_idx"]
                logp = float(np.log(probs[idx] + 1e-12))
                ratio = np.exp(logp - item["old_logp"])
                clipped = np.clip(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
                active = ratio == clipped or (ratio * a) < (clipped * a)
                grad_logp = feats[idx] - probs @ feats
                entropy_grad = -(np.log(probs[idx] + 1e-12) + 1.0) * grad_logp
                if active:
                    self.model.theta += self.cfg.actor_lr * (ratio * a * grad_logp + self.cfg.entropy_coef * entropy_grad)
                    self.model.theta = np.clip(self.model.theta, -8.0, 8.0)
                v = self.model.value(item["state"])
                self.model.value_w += self.cfg.critic_lr * (ret - v) * item["state"]
                self.model.value_w = np.clip(np.nan_to_num(self.model.value_w), -20.0, 20.0)

    def train(self, instance_factory) -> list[dict]:
        self.history = []
        for ep in range(1, self.cfg.episodes + 1):
            instance = instance_factory(ep)
            _, info = self.rollout(instance, train=True)
            self.history.append({"episode": ep, "reward": info["reward"], "Cmax": info["cmax"]})
        return self.history

    def evaluate(self, instance: dict):
        start = time.perf_counter()
        schedule, _ = self.rollout(instance, train=False)
        runtime = time.perf_counter() - start
        return summarize_schedule(self.cfg.name, schedule, instance["lines"], runtime), schedule


def make_agent(name: str, seed: int, episodes: int) -> TSRPPOAgent:
    variants = {
        "PPO": PPOConfig(seed=seed, episodes=episodes, use_action_mask=False, use_split_decoder=False, use_rolling_features=False, name="PPO"),
        "PPO+Mask": PPOConfig(seed=seed, episodes=episodes, use_action_mask=True, use_split_decoder=False, use_rolling_features=False, name="PPO+Mask"),
        "PPO+Mask+Split": PPOConfig(seed=seed, episodes=episodes, use_action_mask=True, use_split_decoder=True, use_rolling_features=False, name="PPO+Mask+Split"),
        "TSR-PPO": PPOConfig(seed=seed, episodes=episodes, use_action_mask=True, use_split_decoder=True, use_rolling_features=True, name="TSR-PPO"),
    }
    return TSRPPOAgent(replace(variants[name]))
