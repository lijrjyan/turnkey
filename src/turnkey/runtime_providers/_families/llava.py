from __future__ import annotations

import importlib
import re
from typing import Any

from turnkey.schema import Sample

LLAVA_HIDDEN_STATE_DEFAULT_MAX_LENGTH = 4096


def _llava_prompt_from_sample(
    sample: Sample,
    *,
    model_name: str,
    model: Any,
    conv_templates: Any,
    constants: dict[str, Any],
) -> str:
    conv = conv_templates[_llava_conv_mode(model_name)].copy()
    query = sample.prompt
    if sample.images:
        query = _llava_adjust_query_for_images(query, model=model, constants=constants)
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _llava_conv_mode(model_name: str) -> str:
    lowered = str(model_name).lower()
    if "llama-2" in lowered:
        return "llava_llama_2"
    if "mistral" in lowered:
        return "mistral_instruct"
    if "v1.6-34b" in lowered:
        return "chatml_direct"
    if "v1" in lowered:
        return "llava_v1"
    if "mpt" in lowered:
        return "mpt"
    return "llava_v0"


def _llava_adjust_query_for_images(query: str, *, model: Any, constants: dict[str, Any]) -> str:
    image_token_se = (
        constants["DEFAULT_IM_START_TOKEN"]
        + constants["DEFAULT_IMAGE_TOKEN"]
        + constants["DEFAULT_IM_END_TOKEN"]
    )
    placeholder = constants["IMAGE_PLACEHOLDER"]
    use_start_end = bool(getattr(getattr(model, "config", None), "mm_use_im_start_end", False))
    if placeholder in query:
        replacement = image_token_se if use_start_end else constants["DEFAULT_IMAGE_TOKEN"]
        return re.sub(placeholder, replacement, query)
    prefix = image_token_se if use_start_end else constants["DEFAULT_IMAGE_TOKEN"]
    return f"{prefix}\n{query}"


def _llava_effective_max_length(*, model: Any, configured_max_length: int | None) -> int:
    if configured_max_length is not None:
        return int(configured_max_length)
    return int(getattr(getattr(model, "config", None), "max_position_embeddings", LLAVA_HIDDEN_STATE_DEFAULT_MAX_LENGTH))


def _llava_image_dtype(model: Any, input_ids: Any) -> Any:
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        return model_dtype
    torch = importlib.import_module("torch")
    return torch.float16 if getattr(input_ids, "is_cuda", False) else torch.float32
