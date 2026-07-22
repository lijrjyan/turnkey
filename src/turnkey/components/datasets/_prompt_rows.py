from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_SORRYBENCH_202406_TEXT_FIELDS = ("prompt", "instruction", "instructions", "question", "request", "user_prompt", "behavior", "goal", "harmful_request", "answer_prompt")
_TURN_TEXT_KEYS = ("content", "text", "value", "prompt")


def instruction_content_answer_prompt(row: Mapping[str, Any], *, empty_message: str) -> str:
    parts = _strings(row.get("instructions"))
    parts += _strings(row.get("content"), allow_list=True, require_all_list_strings=True)
    parts += _strings(row.get("answer_prompt"))
    return _joined_or_raise(parts, row=row, empty_message=empty_message)


def sorrybench_202406_prompt(row: Mapping[str, Any], *, empty_message: str) -> str:
    parts = [
        text
        for field in _SORRYBENCH_202406_TEXT_FIELDS
        for text in _strings(row.get(field), allow_list=True, require_all_list_strings=True)
    ]
    parts.extend(_strings(row.get("content"), allow_list=True))
    parts.extend(_turn_strings(row.get("turns")))
    return _joined_or_raise(parts, row=row, empty_message=empty_message, include_available_fields=True)


def _strings(value: object, *, allow_list: bool = False, require_all_list_strings: bool = False) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list) or not allow_list:
        return []
    if require_all_list_strings and not all(isinstance(item, str) for item in value):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _turn_strings(value: object) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    parts: list[str] = []
    for turn in value:
        if isinstance(turn, str) and turn.strip():
            parts.append(turn.strip())
        elif isinstance(turn, dict):
            for key in _TURN_TEXT_KEYS:
                values = _strings(turn.get(key))
                if values:
                    parts.extend(values)
                    break
    return parts


def _joined_or_raise(
    parts: list[str],
    *,
    row: Mapping[str, Any],
    empty_message: str,
    include_available_fields: bool = False,
) -> str:
    text = "\n".join(parts)
    if text:
        return text
    if include_available_fields:
        available = ", ".join(sorted(str(key) for key in row))
        raise ValueError(f"{empty_message}; fields={available}")
    raise ValueError(empty_message)
