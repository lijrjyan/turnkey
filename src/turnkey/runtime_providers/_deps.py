from __future__ import annotations

import importlib
from typing import Any

from turnkey._internal.hf_deps import (
    import_transformers,
    make_hf_causal_lm_importer,
)
from turnkey.schema import Sample


_import_hf_dependencies = make_hf_causal_lm_importer(
    error_message="gradient provider requires optional HF deps. Install with: pip install -e '.[hf]'"
)


def _import_hidden_state_dependencies():
    deps = import_transformers(
        vision=True,
        required=("AutoModelForCausalLM", "AutoProcessor", "AutoTokenizer"),
        error_message="hidden-state provider requires optional HF deps. Install with: pip install -e '.[hf]'",
    )
    return (
        deps["torch"],
        deps["AutoModelForCausalLM"],
        deps["AutoModelForImageTextToText"],
        deps["AutoModelForVision2Seq"],
        deps["AutoProcessor"],
        deps["AutoTokenizer"],
    )


def _import_internvl_dependencies():
    deps = import_transformers(
        internvl=True,
        required=("AutoModel", "AutoTokenizer"),
        error_message=(
            "hidden-state provider model_family=internvl requires optional HF deps. "
            "Install with: pip install -e '.[hf]'"
        ),
    )
    return deps["torch"], deps["AutoModel"], deps["AutoTokenizer"]


def _import_qwen_process_vision_info():
    try:
        qwen_vl_utils = importlib.import_module("qwen_vl_utils")
        return qwen_vl_utils.process_vision_info
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "hidden-state provider model_family=qwen multimodal samples require qwen_vl_utils. "
            "Install the Qwen VL optional dependencies before using image inputs."
        ) from exc


def _import_llava_dependencies() -> dict[str, Any]:
    try:
        constants = importlib.import_module("llava.constants")
        conversation = importlib.import_module("llava.conversation")
        builder = importlib.import_module("llava.model.builder")
        mm_utils = importlib.import_module("llava.mm_utils")
        return {
            "IMAGE_TOKEN_INDEX": constants.IMAGE_TOKEN_INDEX,
            "DEFAULT_IMAGE_TOKEN": constants.DEFAULT_IMAGE_TOKEN,
            "DEFAULT_IM_START_TOKEN": constants.DEFAULT_IM_START_TOKEN,
            "DEFAULT_IM_END_TOKEN": constants.DEFAULT_IM_END_TOKEN,
            "IMAGE_PLACEHOLDER": constants.IMAGE_PLACEHOLDER,
            "conv_templates": conversation.conv_templates,
            "load_pretrained_model": builder.load_pretrained_model,
            "process_images": mm_utils.process_images,
            "tokenizer_image_token": mm_utils.tokenizer_image_token,
            "get_model_name_from_path": mm_utils.get_model_name_from_path,
        }
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "hidden-state provider model_family=llava requires LLaVA optional dependencies. "
            "Install the LLaVA package before using this extraction family."
        ) from exc


def _format_prompt_for_images(prompt: str, *, n_images: int, image_token: str, auto_insert: bool) -> str:
    if n_images <= 0 or not auto_insert:
        return prompt
    token = str(image_token)
    if not token or token in prompt:
        return prompt
    prefix = "\n".join(token for _ in range(n_images))
    return f"{prefix}\n{prompt}"


def _load_pil_images(sample: Sample) -> list[Any]:
    if not sample.images:
        return []
    try:
        image_mod = importlib.import_module("PIL.Image")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("hidden-state provider image inputs require Pillow. Install with: pip install Pillow") from exc
    return [image_mod.open(image.path).convert("RGB") for image in sample.images]
