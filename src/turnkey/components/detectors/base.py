from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
import json
from typing import TYPE_CHECKING, Any

from turnkey.schema import DetectorDecision, Sample

if TYPE_CHECKING:
    from turnkey.methods import Provider
    from turnkey.policy import Component, Policy


@dataclass(frozen=True)
class DetectorManifest:
    name: str
    version: str = "v1"
    schema_version: str = "detector_manifest/v4"
    required_inputs: tuple[str, ...] = ("sample",)
    reproducibility: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("detector manifest name must be non-empty")
        if not self.version.strip():
            raise ValueError("detector manifest version must be non-empty")
        if not self.required_inputs:
            raise ValueError("detector manifest required_inputs must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "required_inputs": list(self.required_inputs),
            "reproducibility": dict(self.reproducibility),
        }
        return data


class Detector(ABC):
    def method_providers(self) -> tuple[Provider[Any], ...]:
        return ()

    def policy(self, *, calibration_artifact: Any | None = None) -> Policy:  # noqa: ARG002
        from turnkey.policy import DetectorPolicy

        return DetectorPolicy(self.decide)

    def component(
        self,
        *,
        name: str,
        calibration_artifact: Any | None = None,
    ) -> Component:
        from turnkey.policy import Component

        parameters = self.effective_parameters()
        return Component(
            name=name,
            policy=self.policy(calibration_artifact=calibration_artifact),
            providers=self.method_providers(),
            parameters=parameters,
        )

    def effective_parameters(self) -> dict[str, Any]:
        if not is_dataclass(self):
            return {}
        parameters: dict[str, Any] = {}
        for item in fields(self):
            if (
                not item.init
                or item.name.startswith("_")
                or item.metadata.get("component_parameter", True) is False
            ):
                continue
            _require_string_mapping_keys(
                getattr(self, item.name),
                loc=f"{self.__class__.__name__}.{item.name}",
            )
            try:
                value = json.loads(
                    json.dumps(
                        getattr(self, item.name),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{self.__class__.__name__}.{item.name} must be JSON-serializable"
                ) from exc
            parameters[item.name] = value
        return parameters

    def manifest(self, *, name: str | None = None) -> DetectorManifest:
        return DetectorManifest(name=name or self.__class__.__name__)

    @abstractmethod
    def decide(self, sample: Sample) -> DetectorDecision:
        raise NotImplementedError


def _require_string_mapping_keys(value: Any, *, loc: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{loc} must use string keys in nested mappings")
            _require_string_mapping_keys(item, loc=f"{loc}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_string_mapping_keys(item, loc=f"{loc}[{index}]")
