from __future__ import annotations

import os
from typing import Any

from turnkey.schema import Sample

QWEN_HIDDEN_STATE_TEXT_MAX_LENGTH = 8192


def _qwen_messages_from_sample(sample: Sample) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": _qwen_image_reference(image.path)} for image in sample.images
    ]
    content.append({"type": "text", "text": sample.prompt})
    return [{"role": "user", "content": content}]


def _qwen_image_reference(path: str) -> str:
    if path.startswith(("http://", "https://", "file://")):
        return path
    return f"file://{os.path.abspath(path)}"
