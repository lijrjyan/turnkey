from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from turnkey.components.detectors import available_detectors, load_detector
from turnkey.config import DetectorConfig
from turnkey.methods import (
    _attach_secondary_failure,
    is_entrypoint,
    load_entrypoint_with_identity,
)
from turnkey.policy import Component


@dataclass(frozen=True)
class ResolvedComponent:
    component: Component
    source: Mapping[str, Any]


def resolve_component(
    reference: str,
    *,
    params: Mapping[str, Any] | None = None,
    calibration_artifact: Any | None = None,
) -> ResolvedComponent:
    configured_params = dict(params or {})
    validate_component_calibration(reference, calibration_artifact)
    if reference in available_detectors():
        detector = load_detector(DetectorConfig(name=reference, params=configured_params))
        component = detector.component(
            name=reference,
            calibration_artifact=calibration_artifact,
        )
        close = getattr(detector, "close", None)
        if component.cleanup is None and callable(close):
            component = replace(component, cleanup=close)
        return ResolvedComponent(
            component=component,
            source={"kind": "builtin", "name": reference},
        )

    if not is_entrypoint(reference):
        raise ValueError(
            f"unknown component alias {reference!r}; available: {available_detectors()}"
        )
    loaded = load_entrypoint_with_identity(reference)
    value = loaded.value
    if isinstance(value, Component):
        if loaded.source["kind"] == "module":
            _reject_component(
                value,
                ValueError(
                    f"module entrypoint {reference!r} exports a process-cached Component; "
                    "export a builder so each run receives fresh lifecycle state"
                ),
            )
        if configured_params:
            _reject_component(
                value,
                ValueError(
                    f"concrete Component entrypoint {reference!r} cannot accept configured "
                    "params; export a builder instead"
                ),
            )
        component = value
    elif callable(value):
        component = value(**configured_params)
    else:
        component = value
    if not isinstance(component, Component):
        raise TypeError(
            f"component entrypoint {reference!r} must resolve to a Component "
            "or a builder returning one"
        )
    missing_parameters = sorted(set(configured_params) - set(component.parameters))
    if missing_parameters:
        _reject_component(
            component,
            ValueError(
                f"component builder {reference!r} did not persist configured parameter keys "
                f"{missing_parameters}; include every replay-relevant builder argument in "
                "Component.parameters"
            ),
        )
    return ResolvedComponent(
        component=component,
        source=loaded.source,
    )


def validate_component_calibration(
    reference: str,
    calibration_artifact: Any | None,
) -> None:
    if calibration_artifact is not None and is_entrypoint(reference):
        raise ValueError(
            "external components do not support calibration artifacts; "
            "load calibration explicitly in the external builder"
        )


def _reject_component(component: Component, error: Exception) -> None:
    if component.cleanup is not None:
        try:
            component.cleanup()
        except BaseException as cleanup_error:
            _attach_secondary_failure(
                error,
                label="rejected component cleanup",
                secondary=cleanup_error,
            )
    raise error
