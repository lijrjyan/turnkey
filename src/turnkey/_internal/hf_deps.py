from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any


ImportModule = Callable[[str], Any]

HF_INSTALL_HINT = "Install with: pip install -e '.[hf]'"


def _import_module(import_module: ImportModule | None, name: str) -> Any:
    loader = import_module or importlib.import_module
    return loader(name)


def import_torch(
    *,
    error_message: str | None = None,
    import_module: ImportModule | None = None,
) -> tuple[Any, Any, Any]:
    """Import torch plus the nn/functionals commonly used by HF adapters."""
    try:
        torch = _import_module(import_module, "torch")
        nn = _import_module(import_module, "torch.nn")
        functional = _import_module(import_module, "torch.nn.functional")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(error_message or f"HF optional deps require torch. {HF_INSTALL_HINT}") from exc
    return torch, nn, functional


def import_transformers(
    *,
    vision: bool = False,
    internvl: bool = False,
    required: tuple[str, ...] = (),
    error_message: str | None = None,
    import_module: ImportModule | None = None,
) -> dict[str, Any]:
    """Import torch and selected transformers Auto* classes.

    Optional vision classes are returned as ``None`` when absent unless listed
    in ``required``. This preserves the existing fallback behavior in providers.
    """
    try:
        torch = _import_module(import_module, "torch")
        transformers = _import_module(import_module, "transformers")
        deps: dict[str, Any] = {
            "torch": torch,
            "AutoModelForCausalLM": getattr(transformers, "AutoModelForCausalLM", None),
            "AutoTokenizer": getattr(transformers, "AutoTokenizer", None),
        }
        if vision:
            deps.update(
                {
                    "AutoModelForImageTextToText": getattr(
                        transformers, "AutoModelForImageTextToText", None
                    ),
                    "AutoModelForVision2Seq": getattr(transformers, "AutoModelForVision2Seq", None),
                    "AutoProcessor": getattr(transformers, "AutoProcessor", None),
                }
            )
        if internvl:
            deps["AutoModel"] = getattr(transformers, "AutoModel", None)
        missing = [name for name in required if deps.get(name) is None]
        if missing:
            raise AttributeError(f"missing transformers dependencies: {', '.join(missing)}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(error_message or f"HF adapters require optional deps. {HF_INSTALL_HINT}") from exc
    return deps


def import_hf_causal_lm(
    *,
    error_message: str | None = None,
    import_module: ImportModule | None = None,
) -> tuple[Any, Any, Any]:
    deps = import_transformers(
        required=("AutoModelForCausalLM", "AutoTokenizer"),
        error_message=error_message,
        import_module=import_module,
    )
    return deps["torch"], deps["AutoModelForCausalLM"], deps["AutoTokenizer"]


def make_hf_causal_lm_importer(
    *, error_message: str, import_module: ImportModule | None = None
) -> Callable[[], tuple[Any, Any, Any]]:
    return lambda: import_hf_causal_lm(error_message=error_message, import_module=import_module)


def import_datasets(
    *,
    error_message: str | None = None,
    import_module: ImportModule | None = None,
) -> Any:
    try:
        datasets = _import_module(import_module, "datasets")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(error_message or f"HF dataset loaders require `datasets`. {HF_INSTALL_HINT}") from exc
    return datasets.load_dataset


def import_pil(
    *,
    error_message: str | None = None,
    import_module: ImportModule | None = None,
) -> Any:
    try:
        image_mod = _import_module(import_module, "PIL.Image")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(error_message or "image inputs require Pillow. Install with: pip install Pillow") from exc
    return image_mod


def env_hf_token(token_env: str) -> str | None:
    token = os.environ.get(token_env)
    return token if token else None


def cached_hf_login_token(*, import_module: ImportModule | None = None) -> str | None:
    try:
        huggingface_hub = _import_module(import_module, "huggingface_hub")
    except Exception:  # noqa: BLE001
        return None
    token = huggingface_hub.get_token()
    return token if isinstance(token, str) and token.strip() else None


def cached_hf_token(token_env: str, *, import_module: ImportModule | None = None) -> str | None:
    """Return an explicit env token, then a cached Hugging Face login token."""
    return env_hf_token(token_env) or cached_hf_login_token(import_module=import_module)


def optional_hf_token(
    token_env: str,
    *,
    cached_token_fn: Callable[[], str | None] | None = None,
) -> str | bool | None:
    token = env_hf_token(token_env)
    if token:
        return token
    cached_token = cached_token_fn() if cached_token_fn is not None else cached_hf_login_token()
    if cached_token:
        # `True` asks Hugging Face Hub to use an existing login token, if present.
        return True
    return None
