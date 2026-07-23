from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from turnkey.components.detectors.allow_all import AllowAllDetector
from turnkey.components.detectors.base import Detector
from turnkey.methods import MethodContext, Request, load_entrypoint
import turnkey.runtime_providers as runtime_providers
from turnkey.signals import SignalBundle


@dataclass(frozen=True)
class _Square(Request[int]):
    value: int


@dataclass(frozen=True)
class _Label(Request[str]):
    value: str


@dataclass(frozen=True)
class _Unknown(Request[str]):
    value: str


@dataclass
class _Unhashable(Request[int]):
    values: list[int]


class _SquareProvider:
    request_type = _Square

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[_Square] = []

    def provide(self, request: _Square) -> int:
        self.calls.append(request)
        self.events.append(f"square:{request.value}")
        return request.value**2

    def close(self) -> None:
        self.events.append("close:square")


class _LabelProvider:
    request_type = _Label

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def provide(self, request: _Label) -> str:
        self.events.append(f"label:{request.value}")
        return request.value.upper()

    def close(self) -> None:
        self.events.append("close:label")


class _UnhashableProvider:
    request_type = _Unhashable

    def provide(self, request: _Unhashable) -> int:
        return len(request.values)


class _FailingLabelProvider(_LabelProvider):
    def close(self) -> None:
        self.events.append("close:label")
        raise RuntimeError("label close failed")


def test_detector_authoring_surface_has_no_context_executor() -> None:
    assert not hasattr(Detector, "decide_with_context")


def test_signal_bundle_has_no_string_capability_bridge() -> None:
    assert "capabilities" not in {field.name for field in fields(SignalBundle)}


def test_runtime_providers_do_not_export_legacy_string_capabilities() -> None:
    legacy_exports = {
        "CapabilityBroker",
        "CapabilityBundle",
        "CapabilityInstance",
        "CapabilityProvider",
        "CapabilityValue",
        "UnsupportedCapabilitySummary",
    }

    assert all(not hasattr(runtime_providers, name) for name in legacy_exports)


def test_load_entrypoint_supports_module_objects() -> None:
    loaded = load_entrypoint(
        "turnkey.components.detectors.allow_all:AllowAllDetector"
    )

    assert loaded is AllowAllDetector


def test_load_entrypoint_supports_single_file_methods(tmp_path: Path) -> None:
    method_file = tmp_path / "external_method.py"
    method_file.write_text(
        "def build():\n"
        "    return {'kind': 'external', 'source': __file__}\n",
        encoding="utf-8",
    )

    build = load_entrypoint(f"{method_file}:build")

    assert build() == {"kind": "external", "source": str(method_file)}


def test_load_entrypoint_rejects_invalid_specs() -> None:
    with pytest.raises(ValueError, match="module:object or path.py:object"):
        load_entrypoint("missing_separator")


def test_method_context_resolves_lazily_reuses_requests_and_closes_in_reverse_order() -> None:
    events: list[str] = []
    square = _SquareProvider(events)
    label = _LabelProvider(events)

    with MethodContext([square, label]) as context:
        with context.scope("reference"):
            assert context.get(_Square(3)) == 9
            assert context.get(_Square(3)) == 9
        with context.scope("intervention"):
            assert context.get(_Label("safe")) == "SAFE"
            assert context.get(_Square(4)) == 16
        assert [use.cache_hit for use in context.uses] == [False, True, False, False]
        assert [use.scope for use in context.uses] == [
            "reference",
            "reference",
            "intervention",
            "intervention",
        ]
        assert context.uses[1].duration_s == 0.0
        assert all(use.duration_s >= 0.0 for use in context.uses)
        assert [use.request_type for use in context.uses] == [
            f"{_Square.__module__}.{_Square.__qualname__}",
            f"{_Square.__module__}.{_Square.__qualname__}",
            f"{_Label.__module__}.{_Label.__qualname__}",
            f"{_Square.__module__}.{_Square.__qualname__}",
        ]

    assert square.calls == [_Square(3), _Square(4)]
    assert events == [
        "square:3",
        "label:safe",
        "square:4",
        "close:label",
        "close:square",
    ]
    context.close()
    assert events[-2:] == ["close:label", "close:square"]


def test_method_context_restores_nested_request_scopes() -> None:
    context = MethodContext([_SquareProvider([])])

    with context.scope("outer"):
        context.get(_Square(1))
        with context.scope("inner"):
            context.get(_Square(2))
        context.get(_Square(3))

    context.get(_Square(4))
    assert [use.scope for use in context.uses] == ["outer", "inner", "outer", None]


def test_method_context_rejects_unsupported_and_unhashable_requests() -> None:
    context = MethodContext([_SquareProvider([]), _UnhashableProvider()])

    with pytest.raises(LookupError, match=r"no provider for .*_Unknown"):
        context.get(_Unknown("missing"))
    with pytest.raises(TypeError, match="requests must be hashable"):
        context.get(_Unhashable([1, 2]))


def test_method_context_rejects_duplicate_request_providers() -> None:
    with pytest.raises(ValueError, match=r"duplicate provider for .*_Square"):
        MethodContext([_SquareProvider([]), _SquareProvider([])])


def test_method_context_closes_all_materialized_providers_after_close_failure() -> None:
    events: list[str] = []
    context = MethodContext([_SquareProvider(events), _FailingLabelProvider(events)])
    context.get(_Square(2))
    context.get(_Label("safe"))

    with pytest.raises(RuntimeError, match="failed to close 1 method provider"):
        context.close()

    assert events[-2:] == ["close:label", "close:square"]
    context.close()
