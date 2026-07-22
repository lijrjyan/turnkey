from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from turnkey._internal.data import nested_value as _nested_value
from turnkey._internal.io import load_json_object, load_jsonl_objects


def _load_json_object(path: Path, *, errors: list[str]) -> dict[str, Any] | None:
    try:
        return load_json_object(path, root_error="json root must be object")
    except TypeError as exc:
        errors.append(str(exc))
        return None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: invalid json: {exc}")
        return None


def _load_jsonl_rows(path: Path, *, errors: list[str]) -> list[dict[str, Any]]:
    try:
        return load_jsonl_objects(path, skip_non_objects=False)
    except TypeError as exc:
        errors.append(str(exc))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: invalid json: {exc}")
    return []


def _benign_harmful_key(row: dict[str, Any]) -> str | None:
    if row.get("is_benign") is True:
        return "benign"
    if row.get("is_benign") is False:
        return "harmful"
    return None


def _detector_decision_key(row: dict[str, Any]) -> str | None:
    block = _nested_value(row, "intervention", "detector", "block")
    if block is True:
        return "blocked"
    if block is False:
        return "allowed"
    return None


def _ngram_ppl_bucket_key(row: dict[str, Any]) -> str:
    params = row.get("attack_params")
    if isinstance(params, dict):
        for key in ("ngram_ppl_bucket", "naturalness_bucket", "lm_ppl_bucket"):
            value = params.get(key)
            if isinstance(value, str) and value:
                return value
    return "unknown"


def _detector_blocked(row: dict[str, Any], pass_name: str) -> bool:
    return _nested_value(row, pass_name, "detector", "block") is True


def _judge_bool(row: dict[str, Any], pass_name: str, key: str) -> bool:
    return _nested_value(row, pass_name, "judge", key) is True


def _nested_str(value: Any, *keys: str) -> str | None:
    result = _nested_value(value, *keys)
    return result if isinstance(result, str) else None


def _expect_equal(*, errors: list[str], loc: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{loc}: expected {expected!r}, got {actual!r}")


def _expect_close(*, errors: list[str], loc: str, actual: Any, expected: float) -> None:
    if not isinstance(actual, (int, float)):
        errors.append(f"{loc}: expected numeric {expected!r}, got {actual!r}")
        return
    if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
        errors.append(f"{loc}: expected {expected!r}, got {actual!r}")
