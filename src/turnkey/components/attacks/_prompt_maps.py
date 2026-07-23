from __future__ import annotations

import json
from pathlib import Path


def load_replay_prompt_map(
    path: Path,
    *,
    attack_name: str,
    key_field: str = "sample_id",
    prompt_fields: tuple[str, ...] = ("prompt",),
    missing_prompt_label: str = "prompt field",
) -> dict[str, str]:
    if path.suffix.lower() == ".json":
        return _load_json_prompt_map(path, attack_name=attack_name)

    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{attack_name}: invalid jsonl at {path}:{lineno}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{attack_name}: expected object at {path}:{lineno}")

            key = row.get(key_field)
            if not isinstance(key, str) or not key:
                raise ValueError(f"{attack_name}: missing {key_field} at {path}:{lineno}")

            prompt = _prompt_from_row(row, prompt_fields=prompt_fields)
            if prompt is None:
                raise ValueError(f"{attack_name}: missing {missing_prompt_label} at {path}:{lineno}")
            out[key] = prompt
    return out


def _load_json_prompt_map(path: Path, *, attack_name: str) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{attack_name}: .json prompt_map must be an object {{id: prompt}}")
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{attack_name}: .json prompt_map must be {{str: str}}")
        out[key] = value
    return out


def _prompt_from_row(row: dict, *, prompt_fields: tuple[str, ...]) -> str | None:
    for field in prompt_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None
