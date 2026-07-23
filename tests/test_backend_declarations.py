from __future__ import annotations

from turnkey.components.backends import available_backends, load_backend
from turnkey.components.backends.declarations import (
    BACKEND_DECLARATIONS,
    backend_declaration_errors,
    backend_runtime_engine_from_cache_key,
)
from turnkey.config import ModelConfig
from turnkey.runner.io import runtime_cache_key


def test_backend_declarations_are_registered() -> None:
    assert backend_declaration_errors() == ()
    assert set(BACKEND_DECLARATIONS).issubset(set(available_backends()))


def test_load_backend_uses_declared_backend_factory() -> None:
    backend = load_backend(ModelConfig(backend="dummy", model_id="declared-dummy"))

    assert backend.generate(
        prompt="hello",
        max_new_tokens=1,
        temperature=0.0,
    ).model_id == "declared-dummy"


def test_backend_runtime_engine_from_cache_key_uses_backend_name() -> None:
    assert (
        backend_runtime_engine_from_cache_key(
            runtime_cache_key(ModelConfig(backend="dummy", model_id="declared-dummy"))
        )
        == "dummy"
    )
    assert (
        backend_runtime_engine_from_cache_key(
            runtime_cache_key(ModelConfig(backend="vllm", model_id="external-engine"))
        )
        == "vllm"
    )
