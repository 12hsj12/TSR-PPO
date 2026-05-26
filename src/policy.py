from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / max(exp.sum(), 1e-12)


class LinearActorCritic:
    """轻量 Actor-Critic：按动作特征打分，适合无深度学习依赖的可复现实验。"""

    def __init__(self, action_dim: int, state_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0, 0.02, size=action_dim)
        self.value_w = rng.normal(0, 0.02, size=state_dim)

    def action_probs(self, action_features: np.ndarray) -> np.ndarray:
        return softmax(action_features @ self.theta)

    def value(self, state: np.ndarray) -> float:
        return float(state @ self.value_w)

    def save(self, path) -> None:
        np.savez(path, theta=self.theta, value_w=self.value_w)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls(len(data["theta"]), len(data["value_w"]))
        obj.theta = data["theta"]
        obj.value_w = data["value_w"]
        return obj
