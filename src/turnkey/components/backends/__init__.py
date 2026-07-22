from __future__ import annotations

from turnkey.components.backends.base import LLMBackend
from turnkey.components.backends.declarations import BACKEND_DECLARATIONS
from turnkey.config import ModelConfig
from turnkey.registry import BACKENDS


def _register_declared_backends() -> None:
    for declaration in BACKEND_DECLARATIONS.values():
        BACKENDS.register(declaration.name, declaration.build)


_register_declared_backends()


def available_backends() -> list[str]:
    return BACKENDS.list()


def load_backend(cfg: ModelConfig) -> LLMBackend:
    return BACKENDS.get(cfg.backend)(cfg)
