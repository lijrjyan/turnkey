from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_object(path: str | Path, *, root_error: str = "expected JSON object") -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {root_error}")
    return value


def load_jsonl_objects(
    path: str | Path,
    *,
    missing_ok: bool = False,
    skip_invalid: bool = False,
    skip_non_objects: bool = True,
) -> list[dict[str, Any]]:
    path = Path(path)
    if missing_ok and not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception:  # noqa: BLE001
            if skip_invalid:
                continue
            raise
        if isinstance(value, dict):
            rows.append(value)
        elif not skip_non_objects:
            raise TypeError(f"{path}:{line_no}: expected JSON object")
    return rows


def json_object_text(value: dict[str, Any], *, sort_keys: bool = False) -> str:
    return json.dumps(value, indent=2, sort_keys=sort_keys, ensure_ascii=False)


def write_json_object(path: str | Path, value: dict[str, Any], *, sort_keys: bool = False) -> None:
    path = Path(path)
    path.write_text(json_object_text(value, sort_keys=sort_keys) + "\n", encoding="utf-8")
