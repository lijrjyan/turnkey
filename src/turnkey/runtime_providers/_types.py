from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from turnkey.capabilities import BackendCapabilities


ProviderKind = Literal[
    "content",
    "attack_state",
    "model_generate",
    "multi_generate",
    "model_signals",
    "hidden_states",
    "gradients",
    "run_metadata",
]

ProviderStatus = Literal["ok", "unsupported", "not_implemented", "partial", "error"]


@dataclass(frozen=True)
class ProviderSummary:
    name: str
    kind: ProviderKind
    requested: tuple[str, ...] = field(default_factory=tuple)
    materialized: tuple[str, ...] = field(default_factory=tuple)
    capabilities: dict[str, Any] = field(default_factory=dict)
    status: ProviderStatus = "ok"
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "requested": list(self.requested),
            "materialized": list(self.materialized),
            "capabilities": dict(self.capabilities),
            "status": self.status,
            "message": self.message,
        }


def backend_capabilities_dict(caps: BackendCapabilities) -> dict[str, Any]:
    return asdict(caps)
