from __future__ import annotations

import csv
from pathlib import Path

from .io import ensure_dir


class CsvLogger:
    """追加式 CSV 日志，不在训练中画图。"""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        ensure_dir(self.path.parent)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(row)
