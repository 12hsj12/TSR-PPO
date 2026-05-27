from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - 允许无 torch 环境做静态检查
    torch = None
    nn = None
    optim = None


@dataclass
class PPOHyperParams:
    episodes: int = 1000
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    eval_interval: int = 20
    save_interval: int = 100


class ActorCritic(nn.Module if nn else object):
    """动作特征打分 Actor + 全局状态 Critic，支持可变动作集合。"""

    def __init__(self, state_dim: int, action_feature_dim: int, hidden_dim: int = 128):
        if nn is None:
            raise RuntimeError("PyTorch is required for formal PPO training. Please install requirements.txt.")
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(action_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def logits(self, action_features):
        return self.actor(action_features).squeeze(-1)

    def value(self, state):
        return self.critic(state).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim: int, action_feature_dim: int, params: PPOHyperParams, device: str = "cpu", seed: int = 42):
        if torch is None:
            raise RuntimeError("PyTorch is required for formal PPO training. Please install requirements.txt.")
        self.params = params
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.model = ActorCritic(state_dim, action_feature_dim).to(self.device)
        self.optimizer = optim.Adam(
            [
                {"params": self.model.actor.parameters(), "lr": params.actor_lr},
                {"params": self.model.critic.parameters(), "lr": params.critic_lr},
            ]
        )

    def select_action(self, obs: dict, deterministic: bool = False) -> tuple[int, dict]:
        features = torch.as_tensor(obs["action_features"], dtype=torch.float32, device=self.device)
        if features.shape[0] == 0:
            return -1, {"log_prob": 0.0, "entropy": 0.0, "value": 0.0}
        state = torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device)
        logits = self.model.logits(features)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits) if deterministic else dist.sample()
        return int(action.item()), {
            "log_prob": float(dist.log_prob(action).detach().cpu()),
            "entropy": float(dist.entropy().detach().cpu()),
            "value": float(self.model.value(state).detach().cpu()),
        }

    def evaluate_action_batch(self, states, action_features_list, actions):
        log_probs, entropies, values = [], [], []
        for state_np, feat_np, action_idx in zip(states, action_features_list, actions):
            state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
            feats = torch.as_tensor(feat_np, dtype=torch.float32, device=self.device)
            logits = self.model.logits(feats)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.tensor(action_idx, dtype=torch.long, device=self.device)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(self.model.value(state))
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)

    def update(self, trajectory: list[dict]) -> dict:
        rewards = np.asarray([t["reward"] for t in trajectory], dtype=np.float32)
        dones = np.asarray([t["done"] for t in trajectory], dtype=np.float32)
        values = np.asarray([t["value"] for t in trajectory] + [0.0], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.params.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.params.gamma * self.params.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = [t["state"] for t in trajectory]
        feats = [t["action_features"] for t in trajectory]
        actions = [t["action"] for t in trajectory]
        old_log_probs = torch.as_tensor([t["log_prob"] for t in trajectory], dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        last = {}
        for _ in range(self.params.update_epochs):
            log_probs, entropy, value_pred = self.evaluate_action_batch(states, feats, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - self.params.clip_ratio, 1 + self.params.clip_ratio) * adv_t
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = ((value_pred - returns_t) ** 2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + self.params.value_coef * value_loss + self.params.entropy_coef * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            last = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.mean().detach().cpu()),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        return last

    def save_checkpoint(self, path: str | Path, episode: int, seed: int, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "episode": episode,
                "seed": seed,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "params": asdict(self.params),
                "extra": extra or {},
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt
