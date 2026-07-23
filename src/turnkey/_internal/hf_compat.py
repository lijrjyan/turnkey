from __future__ import annotations

QWEN35_MODEL_TYPE = "qwen3_5"


def qwen35_compat_hint(*, model_id: str, error: BaseException | str | None = None) -> str | None:
    message = str(error or "")
    if not ("qwen3.5" in model_id.lower() or QWEN35_MODEL_TYPE in message):
        return None
    return (
        "Qwen3.5 uses the `qwen3_5` hybrid architecture. The current stable "
        "Transformers/vLLM/SGLang stack may not support it; use an isolated environment "
        "with Transformers main, vLLM nightly, or SGLang main before treating this as a "
        "Turnkey model/config error."
    )


def append_compat_hint(message: str, *, model_id: str, error: BaseException | str | None = None) -> str:
    hint = qwen35_compat_hint(model_id=model_id, error=error)
    if hint is None:
        return message
    return f"{message} {hint}"
