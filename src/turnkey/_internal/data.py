from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_hex(payload)


def safe_div(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def safe_div_or_none(num: int | float, den: int | float) -> float | None:
    if not den:
        return None
    return float(num) / float(den)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def non_negative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def safe_div_if_positive(num: Any, den: Any) -> float | None:
    if not finite_number(num) or not finite_number(den) or float(den) <= 0.0:
        return None
    return float(num) / float(den)


def sub_or_none(left: Any, right: Any) -> float | None:
    if finite_number(left) and finite_number(right):
        return float(left) - float(right)
    return None


def binary_f1_or_none(*, tp: int, fp: int, fn: int) -> float | None:
    precision = safe_div_or_none(tp, tp + fp)
    recall = safe_div_or_none(tp, tp + fn)
    if precision is None or recall is None:
        return None
    return safe_div_or_none(2.0 * precision * recall, precision + recall)


def binary_f1(*, tp: int, fp: int, fn: int) -> float:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2.0 * precision * recall, precision + recall)


def binary_auc(score_labels: list[tuple[float, int]]) -> float | None:
    positives = sum(1 for _score, label in score_labels if label == 1)
    negatives = sum(1 for _score, label in score_labels if label == 0)
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(score_labels, key=lambda item: item[0])
    rank_sum_pos = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum_pos += average_rank * sum(1 for _score, label in ordered[index:end] if label == 1)
        index = end
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def string_set(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {item for item in value if isinstance(item, str)}


def nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
