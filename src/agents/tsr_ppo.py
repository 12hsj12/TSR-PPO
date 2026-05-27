from __future__ import annotations

from dataclasses import dataclass

from .ppo import PPOAgent, PPOHyperParams


@dataclass
class TSRPPOConfig(PPOHyperParams):
    use_mask: bool = True
    use_split: bool = True
    use_rolling: bool = True
    use_machine_mismatch_feature: bool = True


class TSRPPOAgent(PPOAgent):
    """完整 TSR-PPO。模块开关用于消融：Mask / Split / Rolling / Full。"""

    def __init__(self, state_dim: int, action_feature_dim: int, params: TSRPPOConfig, device: str = "cpu", seed: int = 42):
        super().__init__(state_dim, action_feature_dim, params, device=device, seed=seed)
        self.tsr_params = params
