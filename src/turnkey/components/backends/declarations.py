from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from turnkey.components.backends.base import LLMBackend
from turnkey.config import ModelConfig


BackendFactory = Callable[[ModelConfig], LLMBackend]


@dataclass(frozen=True)
class BackendDeclaration:
    name: str
    runtime_engine: str
    factory: BackendFactory

    def build(self, cfg: ModelConfig) -> LLMBackend:
        return self.factory(cfg)


def _build_dummy(cfg: ModelConfig) -> LLMBackend:
    from turnkey.components.backends.dummy import DummyBackend

    return DummyBackend(model_id=cfg.model_id)


def _build_hf(cfg: ModelConfig) -> LLMBackend:
    from turnkey.components.backends.hf import HFBackend

    return HFBackend(
        model_id=cfg.model_id,
        revision=cfg.revision,
        device=cfg.device,
        trust_remote_code=cfg.trust_remote_code,
    )


def _build_openai_compat(cfg: ModelConfig) -> LLMBackend:
    from turnkey.components.backends.openai_compat import OpenAICompatBackend

    if not cfg.base_url:
        raise ValueError("openai_compat backend requires model.base_url")
    return OpenAICompatBackend(
        model_id=cfg.model_id,
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
        timeout_s=cfg.timeout_s,
    )


BACKEND_DECLARATIONS: dict[str, BackendDeclaration] = {
    "dummy": BackendDeclaration(
        name="dummy",
        runtime_engine="dummy",
        factory=_build_dummy,
    ),
    "hf": BackendDeclaration(
        name="hf",
        runtime_engine="hf",
        factory=_build_hf,
    ),
    "openai_compat": BackendDeclaration(
        name="openai_compat",
        runtime_engine="openai_compat",
        factory=_build_openai_compat,
    ),
}


def backend_declaration_for(name: str) -> BackendDeclaration | None:
    return BACKEND_DECLARATIONS.get(name)


def backend_runtime_engine(name: str) -> str:
    declaration = backend_declaration_for(name)
    if declaration is not None:
        return declaration.runtime_engine
    return name


def backend_declaration_errors(
    declarations: Mapping[str, BackendDeclaration] | None = None,
) -> tuple[str, ...]:
    active = dict(declarations or BACKEND_DECLARATIONS)
    errors: list[str] = []
    runtime_engines: dict[str, str] = {}
    for name, declaration in sorted(active.items()):
        if not name.strip():
            errors.append("backend declaration names must be non-empty")
        if name != declaration.name:
            errors.append(f"{name}: declaration name mismatch {declaration.name!r}")
        if not declaration.runtime_engine.strip():
            errors.append(f"{name}: runtime_engine must be non-empty")
        existing = runtime_engines.setdefault(declaration.runtime_engine, name)
        if existing != name:
            errors.append(
                f"{name}: runtime_engine {declaration.runtime_engine!r} already used by {existing}"
            )
    return tuple(errors)


def backend_runtime_engine_from_cache_key(key: Any) -> str | None:
    if not isinstance(key, str):
        return None
    try:
        payload = json.loads(key)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend:
        return None
    return backend_runtime_engine(backend)
