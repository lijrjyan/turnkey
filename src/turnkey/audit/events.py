from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ._helpers import _expect_close


EVENT_SCHEMA = "turnkey_event/v1"
EVENT_KINDS = {"policy", "request", "judge"}


def check_events(
    *,
    path: Path,
    events: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    metrics: dict[str, Any] | None,
    errors: list[str],
) -> None:
    case_ids = {case_id for case in cases if isinstance((case_id := case.get("case_id")), str)}
    by_id: dict[str, dict[str, Any]] = {}
    boundaries: set[int] = set()

    for index, event in enumerate(events, start=1):
        loc = f"{path}:{index}"
        if event.get("schema_version") != EVENT_SCHEMA:
            errors.append(f"{loc}: schema_version must be {EVENT_SCHEMA!r}")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            errors.append(f"{loc}: event_id must be non-empty string")
        elif event_id in by_id:
            errors.append(f"{loc}: duplicate event_id: {event_id}")
        else:
            by_id[event_id] = event

        sequence = event.get("sequence")
        end_sequence = event.get("end_sequence")
        if not _non_negative_int(sequence) or sequence == 0:
            errors.append(f"{loc}: sequence must be positive integer")
        elif sequence in boundaries:
            errors.append(f"{loc}: duplicate span boundary: {sequence}")
        else:
            boundaries.add(sequence)
        if not _non_negative_int(end_sequence) or (
            _non_negative_int(sequence) and end_sequence <= sequence
        ):
            errors.append(f"{loc}: end_sequence must be integer after sequence")
        elif end_sequence in boundaries:
            errors.append(f"{loc}: duplicate span boundary: {end_sequence}")
        else:
            boundaries.add(end_sequence)

        case_id = event.get("case_id")
        if case_id not in case_ids:
            errors.append(f"{loc}: case_id does not exist: {case_id!r}")
        if event.get("pass") not in {"reference", "intervention"}:
            errors.append(f"{loc}: pass must be 'reference' or 'intervention'")
        if event.get("status") not in {"ok", "error"}:
            errors.append(f"{loc}: status must be 'ok' or 'error'")
        kind = event.get("kind")
        if kind not in EVENT_KINDS:
            errors.append(f"{loc}: kind must be one of {sorted(EVENT_KINDS)!r}")
        if not isinstance(event.get("name"), str) or not event.get("name"):
            errors.append(f"{loc}: name must be non-empty string")
        if kind == "request":
            if not isinstance(event.get("provider"), str) or not event.get("provider"):
                errors.append(f"{loc}: request provider must be non-empty string")
            if type(event.get("cache_hit")) is not bool:
                errors.append(f"{loc}: request cache_hit must be bool")
        _check_non_negative_number(loc=loc, key="duration_s", event=event, errors=errors)
        if not _non_negative_int(event.get("model_forwards")):
            errors.append(f"{loc}: model_forwards must be non-negative integer")
        _check_optional_cost_fields(loc=loc, event=event, errors=errors)

    for index, event in enumerate(events, start=1):
        parent_id = event.get("parent_event_id")
        if parent_id is None:
            continue
        loc = f"{path}:{index}"
        parent = by_id.get(parent_id) if isinstance(parent_id, str) else None
        if parent is None:
            errors.append(f"{loc}: parent_event_id does not exist: {parent_id!r}")
            continue
        if parent.get("case_id") != event.get("case_id") or parent.get("pass") != event.get("pass"):
            errors.append(f"{loc}: parent event must share case_id and pass")
        if not _strictly_contains(parent, event):
            errors.append(f"{loc}: parent event span does not contain child span")

    expected_boundaries = set(range(1, len(events) * 2 + 1))
    if boundaries != expected_boundaries:
        errors.append(f"{path}: span boundaries must be contiguous from 1 to {len(events) * 2}")
    _check_event_tree(path=path, events=events, cases=cases, errors=errors)
    _check_blocked_original_calls(path=path, events=events, cases=cases, errors=errors)
    _check_extra_forwards(path=path, events=events, cases=cases, metrics=metrics, errors=errors)


def _check_event_tree(
    *,
    path: Path,
    events: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    errors: list[str],
) -> None:
    for case in cases:
        case_id = case.get("case_id")
        for pass_name in ("reference", "intervention"):
            scoped = [
                event
                for event in events
                if event.get("case_id") == case_id and event.get("pass") == pass_name
            ]
            policies = [event for event in scoped if event.get("kind") == "policy"]
            judges = [event for event in scoped if event.get("kind") == "judge"]
            loc = f"{path}: case {case_id!r} pass {pass_name!r}"
            if len(policies) != 1:
                errors.append(f"{loc}: expected exactly one policy event, got {len(policies)}")
                continue
            policy = policies[0]
            policy_id = policy.get("event_id")
            if policy.get("parent_event_id") is not None:
                errors.append(f"{loc}: policy event must be a root span")
            if len(judges) != 1:
                errors.append(f"{loc}: expected exactly one judge event, got {len(judges)}")
            for event in scoped:
                kind = event.get("kind")
                if kind in {"request", "judge"} and event.get("parent_event_id") != policy_id:
                    errors.append(f"{loc}: {kind} parent must be the pass policy")


def _check_blocked_original_calls(
    *,
    path: Path,
    events: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    errors: list[str],
) -> None:
    for case in cases:
        intervention = case.get("intervention")
        detector = intervention.get("detector") if isinstance(intervention, dict) else None
        if not isinstance(detector, dict) or detector.get("block") is not True:
            continue
        prompt = case.get("prompt")
        prompt_hash = prompt.get("sha256") if isinstance(prompt, dict) else None
        case_id = case.get("case_id")
        for event in events:
            request = event.get("request")
            if (
                event.get("case_id") == case_id
                and event.get("pass") == "intervention"
                and event.get("kind") == "request"
                and str(event.get("name", "")).endswith(".Generate")
                and event.get("cache_hit") is False
                and isinstance(request, dict)
                and request.get("prompt_sha256") == prompt_hash
            ):
                errors.append(
                    f"{path}: case {case_id!r}: blocked intervention called original target"
                )
                break


def _check_extra_forwards(
    *,
    path: Path,
    events: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    metrics: dict[str, Any] | None,
    errors: list[str],
) -> None:
    forwards = sum(
        event.get("model_forwards", 0)
        for event in events
        if event.get("pass") == "intervention"
        and event.get("kind") == "request"
        and _non_negative_int(event.get("model_forwards"))
    )
    expected = forwards / len(cases) if cases else 0.0
    cost = metrics.get("cost") if isinstance(metrics, dict) else None
    if not isinstance(cost, dict):
        return
    _expect_close(
        errors=errors,
        loc=f"{path.parent / 'metrics.json'}: metrics.cost.extra_forwards_avg",
        actual=cost.get("extra_forwards_avg"),
        expected=expected,
    )


def _check_optional_cost_fields(
    *,
    loc: str,
    event: dict[str, Any],
    errors: list[str],
) -> None:
    result = event.get("result")
    if result is not None and not isinstance(result, dict):
        errors.append(f"{loc}: result must be object")
        return
    if isinstance(result, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in result and not _non_negative_int(result[key]):
                errors.append(f"{loc}: result.{key} must be non-negative integer")
        if "latency_s" in result:
            _check_non_negative_number(
                loc=f"{loc}: result",
                key="latency_s",
                event=result,
                errors=errors,
            )
    for key in ("device_memory_bytes_before", "device_memory_bytes_after"):
        if key in event and not _non_negative_int(event[key]):
            errors.append(f"{loc}: {key} must be non-negative integer")


def _strictly_contains(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    parent_start = parent.get("sequence")
    parent_end = parent.get("end_sequence")
    child_start = child.get("sequence")
    child_end = child.get("end_sequence")
    return all(
        _non_negative_int(value) for value in (parent_start, parent_end, child_start, child_end)
    ) and (parent_start < child_start < child_end < parent_end)


def _check_non_negative_number(
    *,
    loc: str,
    key: str,
    event: dict[str, Any],
    errors: list[str],
) -> None:
    value = event.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        errors.append(f"{loc}: {key} must be finite non-negative number")


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
