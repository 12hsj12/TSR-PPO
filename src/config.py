from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

PROCESS_TYPES = ["纵剪", "横切", "复合剪"]


@dataclass
class GeneratorConfig:
    seed: int = 42
    n_jobs: int = 100
    split_limit: int = 2
    heterogeneity: str = "medium"
    arrival_intensity: str = "medium"
    rolling_delta: float = 120.0
    raw_xlsx: Path = DATA_DIR / "raw" / "加工机组执行信息.xlsx"


@dataclass
class EnvConfig:
    seed: int = 42
    max_split: int = 2
    min_split_ratio: float = 0.05
    rolling_delta: float = 120.0
    idle_penalty: float = 0.01
    balance_penalty: float = 0.02
    max_combo_per_job: int = 8


@dataclass
class PPOConfig:
    seed: int = 42
    episodes: int = 40
    gamma: float = 0.98
    clip_ratio: float = 0.2
    actor_lr: float = 0.015
    critic_lr: float = 0.03
    entropy_coef: float = 0.002
    update_epochs: int = 3
    use_action_mask: bool = True
    use_split_decoder: bool = True
    use_rolling_features: bool = True
    name: str = "TSR-PPO"


@dataclass
class ExperimentConfig:
    seed: int = 42
    mode: str = "quick"
    quick_jobs: int = 60
    full_jobs: int = 200
    quick_episodes: int = 35
    full_episodes: int = 180
    algorithms: list[str] = field(
        default_factory=lambda: ["FIFO", "SPT", "EAT", "GA", "PPO", "TSR-PPO"]
    )
