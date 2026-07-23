from __future__ import annotations

import math
from typing import Any


def _ranked_detectors(values: list[tuple[str, float]]) -> list[dict[str, Any]]:
    ranked = sorted(values, key=lambda item: (-item[1], item[0]))
    return [{"rank": index + 1, "detector": name, "value": value} for index, (name, value) in enumerate(ranked)]


def _spearman_rho(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    x_ranks = _average_ranks([x for x, _y in pairs])
    y_ranks = _average_ranks([y for _x, y in pairs])
    return _pearson(x_ranks, y_ranks)


def _kendall_tau_b(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    concordant = 0
    discordant = 0
    x_ties = 0
    y_ties = 0
    for i, (x_i, y_i) in enumerate(pairs):
        for x_j, y_j in pairs[i + 1 :]:
            x_cmp = _sign(x_i - x_j)
            y_cmp = _sign(y_i - y_j)
            if x_cmp == 0 and y_cmp == 0:
                continue
            if x_cmp == 0:
                x_ties += 1
            elif y_cmp == 0:
                y_ties += 1
            elif x_cmp == y_cmp:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + x_ties) * (concordant + discordant + y_ties))
    if denominator == 0.0:
        return None
    return float((concordant - discordant) / denominator)


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _value in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_den = sum((x - left_mean) ** 2 for x in left)
    right_den = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_den * right_den)
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
