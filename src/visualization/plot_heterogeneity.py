from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .plot_non_equivalent_machine import plot_heterogeneity_improvement
from .common import ensure_output, setup_style


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    plot_heterogeneity_improvement(pd.read_csv(args.summary), ensure_output(args.output))


if __name__ == "__main__":
    main()
