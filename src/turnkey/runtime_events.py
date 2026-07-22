from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import sys
from time import perf_counter
from typing import Any, Iterator, Literal

from turnkey._internal.redact import redact_text


EVENT_SCHEMA = "turnkey_event/v1"
EventStatus = Literal["running", "ok", "error"]


@dataclass
class RuntimeEvent:
    event_id: str
    sequence: int
    case_id: str
    pass_name: str
    kind: str
    name: str
    parent_event_id: str | None = None
    provider: str | None = None
    cache_hit: bool | None = None
    model_forwards: int = 0
    status: EventStatus = "running"
    end_sequence: int | None = None
    duration_s: float | None = None
    result: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    device_memory_bytes_before: int | None = None
    device_memory_bytes_after: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": EVENT_SCHEMA,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "end_sequence": self.end_sequence,
            "case_id": self.case_id,
            "pass": self.pass_name,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "duration_s": self.duration_s,
            "model_forwards": self.model_forwards,
        }
        for key, item in (
            ("parent_event_id", self.parent_event_id),
            ("provider", self.provider),
            ("cache_hit", self.cache_hit),
            ("result", self.result),
            ("request", self.request),
            ("error", self.error),
            ("device_memory_bytes_before", self.device_memory_bytes_before),
            ("device_memory_bytes_after", self.device_memory_bytes_after),
        ):
            if item is not None:
                value[key] = item
        return value


class EventRecorder:
    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []
        self._sequence = 0

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)

    @contextmanager
    def span(
        self,
        *,
        case_id: str,
        pass_name: str,
        kind: str,
        name: str,
        parent_event_id: str | None = None,
        provider: str | None = None,
        cache_hit: bool | None = None,
        model_forwards: int = 0,
    ) -> Iterator[RuntimeEvent]:
        self._sequence += 1
        event = RuntimeEvent(
            event_id=f"event-{len(self._events) + 1:06d}",
            sequence=self._sequence,
            case_id=case_id,
            pass_name=pass_name,
            kind=kind,
            name=name,
            parent_event_id=parent_event_id,
            provider=provider,
            cache_hit=cache_hit,
            model_forwards=model_forwards,
            device_memory_bytes_before=_available_device_memory_bytes(),
        )
        self._events.append(event)
        started = perf_counter()
        try:
            yield event
        except BaseException as exc:
            event.status = "error"
            event.error = error_summary(exc)
            raise
        else:
            event.status = "ok"
        finally:
            event.duration_s = max(0.0, perf_counter() - started)
            event.device_memory_bytes_after = _available_device_memory_bytes()
            self._sequence += 1
            event.end_sequence = self._sequence


def result_summary(value: Any) -> dict[str, Any] | None:
    fields = (
        "executed",
        "backend",
        "model_id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_s",
    )
    result = {
        field: getattr(value, field)
        for field in fields
        if getattr(value, field, None) is not None
    }
    return result or None


def request_summary(value: Any) -> dict[str, Any] | None:
    prompt = getattr(value, "prompt", None)
    result: dict[str, Any] = {}
    if isinstance(prompt, str):
        result["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        result["prompt_chars"] = len(prompt)
    images = getattr(value, "images", None)
    if isinstance(images, tuple):
        result["image_count"] = len(images)
    for field in ("max_new_tokens", "temperature"):
        item = getattr(value, field, None)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result[field] = item
    return result or None


def error_summary(error: BaseException) -> dict[str, str]:
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": redact_text(str(error)),
    }


def _available_device_memory_bytes() -> int | None:
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return None
    try:
        if not cuda.is_available():
            return None
        free_bytes, _total_bytes = cuda.mem_get_info()
    except Exception:  # device telemetry must never change execution behavior
        return None
    if isinstance(free_bytes, int) and not isinstance(free_bytes, bool) and free_bytes >= 0:
        return free_bytes
    return None
