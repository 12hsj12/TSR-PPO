from __future__ import annotations

from pathlib import Path

from src.data.generator import generate_dataset
from src.utils.io import read_json


def load_dataset(dataset_dir: str | Path, auto_generate: bool = True, scale: str = "medium", seed: int = 42) -> dict[str, list[dict]]:
    dataset_dir = Path(dataset_dir)
    files = {name: dataset_dir / f"{name}.json" for name in ["train_instances", "val_instances", "test_instances"]}
    if not all(path.exists() for path in files.values()):
        if not auto_generate:
            missing = [str(path) for path in files.values() if not path.exists()]
            raise FileNotFoundError(f"missing dataset files: {missing}")
        return generate_dataset(dataset_dir, scale=scale, seed=seed)
    return {name: read_json(path) for name, path in files.items()}
