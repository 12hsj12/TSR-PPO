from __future__ import annotations

import numpy as np


def decode_split(
    job: dict,
    combo: tuple[str, ...],
    line_available: dict[str, float],
    tau: float,
    min_ratio: float = 0.05,
    use_decoder: bool = True,
) -> dict[str, float]:
    """按加工效率与最早可用时间解码拆分比例。

    use_decoder=False 时使用均分比例，用于“普通 PPO/消融”对比。
    """
    if not combo:
        return {}
    if not use_decoder or len(combo) == 1:
        return {k: 1.0 / len(combo) for k in combo}
    weights = []
    for line_id in combo:
        p = max(float(job["p_times"][line_id]), 1e-6)
        wait = max(line_available.get(line_id, 0.0) - tau, 0.0)
        weights.append((1.0 / p) * (1.0 / (1.0 + wait)))
    arr = np.asarray(weights, dtype=float)
    if arr.sum() <= 0:
        arr = np.ones(len(combo), dtype=float)
    ratios = arr / arr.sum()
    kept = [(k, r) for k, r in zip(combo, ratios) if r >= min_ratio]
    if not kept:
        best = int(np.argmax(ratios))
        return {combo[best]: 1.0}
    total = sum(r for _, r in kept)
    return {k: float(r / total) for k, r in kept}
