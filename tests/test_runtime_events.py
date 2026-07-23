from __future__ import annotations

from dataclasses import dataclass

import pytest

from turnkey.components.backends.dummy import DummyBackend
from turnkey.components.judges.dummy import DummyRefusalJudge
from turnkey.config import Config, ModelConfig, NaturalnessConfig
from turnkey.methods import MethodContext, Request
from turnkey.policy import Component, PolicyChain
from turnkey.runner.policy_executor import run_policy_pair
from turnkey.runtime_events import EventRecorder
from turnkey.schema import ModelOutput, Sample


@dataclass(frozen=True)
class _Value(Request[int]):
    value: int


class _ValueProvider:
    request_type = _Value
    model_forwards_per_call = 2

    def provide(self, request: _Value) -> int:
        return request.value * 2


@dataclass(frozen=True)
class _Fail(Request[int]):
    value: int


class _FailProvider:
    request_type = _Fail
    model_forwards_per_call = 1

    def provide(self, request: _Fail) -> int:
        raise ValueError(f"cannot provide {request.value}")


@dataclass(frozen=True)
class _GenerateLike(Request[ModelOutput]):
    prompt: str


class _GenerateLikeProvider:
    request_type = _GenerateLike
    model_forwards_per_call = 1

    def provide(self, request: _GenerateLike) -> ModelOutput:
        return ModelOutput(
            executed=True,
            backend="test",
            model_id="test-model",
            response_text="private response",
            prompt_tokens=3,
            completion_tokens=5,
            total_tokens=8,
            latency_s=0.25,
        )


def test_method_context_records_nested_request_events_and_cache_hits() -> None:
    recorder = EventRecorder()
    with recorder.span(
        case_id="case-1",
        pass_name="intervention",
        kind="policy",
        name="test-policy",
    ) as policy_span:
        with MethodContext((_ValueProvider(),), event_recorder=recorder) as context:
            with context.scope(
                case_id="case-1",
                pass_name="intervention",
                parent_event_id=policy_span.event_id,
            ):
                assert context.get(_Value(4)) == 8
                assert context.get(_Value(4)) == 8

    events = [event.to_dict() for event in recorder.events]

    assert [event["event_id"] for event in events] == [
        "event-000001",
        "event-000002",
        "event-000003",
    ]
    policy, miss, hit = events
    assert policy["kind"] == "policy"
    assert policy["status"] == "ok"
    assert miss["parent_event_id"] == hit["parent_event_id"] == policy["event_id"]
    assert miss["case_id"] == hit["case_id"] == "case-1"
    assert miss["pass"] == hit["pass"] == "intervention"
    assert miss["kind"] == hit["kind"] == "request"
    assert miss["cache_hit"] is False
    assert miss["model_forwards"] == 2
    assert hit["cache_hit"] is True
    assert hit["model_forwards"] == 0
    assert policy["sequence"] < miss["sequence"] < miss["end_sequence"]
    assert policy["sequence"] < hit["sequence"] < hit["end_sequence"]
    assert policy["end_sequence"] > hit["end_sequence"]


def test_method_context_records_provider_errors_without_caching_them() -> None:
    recorder = EventRecorder()
    with MethodContext((_FailProvider(),), event_recorder=recorder) as context:
        with context.scope(case_id="case-2", pass_name="reference"):
            with pytest.raises(ValueError, match="cannot provide 7"):
                context.get(_Fail(7))

    event = recorder.events[0].to_dict()
    assert event["kind"] == "request"
    assert event["status"] == "error"
    assert event["cache_hit"] is False
    assert event["model_forwards"] == 1
    assert set(event["error"]) == {"type", "message"}
    assert event["error"]["type"] == "builtins.ValueError"
    assert event["error"]["message"].startswith("<redacted sha256=")
    assert "cannot provide 7" not in str(event)


def test_request_event_records_model_cost_without_plaintext() -> None:
    recorder = EventRecorder()
    with MethodContext((_GenerateLikeProvider(),), event_recorder=recorder) as context:
        with context.scope(case_id="case-3", pass_name="reference"):
            context.get(_GenerateLike("private prompt"))

    event = recorder.events[0].to_dict()
    assert event["result"] == {
        "executed": True,
        "backend": "test",
        "model_id": "test-model",
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
        "latency_s": 0.25,
    }
    assert "private prompt" not in str(event)
    assert "private response" not in str(event)


def test_policy_pair_emits_case_policy_request_and_judge_tree() -> None:
    result = run_policy_pair(
        samples=[
            Sample(
                sample_id="case-4",
                behavior_id="behavior-4",
                is_benign=True,
                prompt="hello",
            )
        ],
        reference=Component(name="reference", policy=PolicyChain()),
        intervention=Component(name="intervention", policy=PolicyChain()),
        cfg=Config(
            model=ModelConfig(backend="dummy", model_id="target"),
            naturalness=NaturalnessConfig(enabled=False),
        ),
        backend=DummyBackend(model_id="target"),
        judge=DummyRefusalJudge(),
    )

    events = [event.to_dict() for event in result.events]
    policies = [event for event in events if event["kind"] == "policy"]
    requests = [event for event in events if event["kind"] == "request"]
    judges = [event for event in events if event["kind"] == "judge"]
    policy_by_pass = {event["pass"]: event for event in policies}

    assert len(policies) == len(requests) == len(judges) == 2
    assert {event["case_id"] for event in events} == {"case-4"}
    assert [event["cache_hit"] for event in requests] == [False, True]
    assert [event["model_forwards"] for event in requests] == [1, 0]
    for event in [*requests, *judges]:
        assert event["parent_event_id"] == policy_by_pass[event["pass"]]["event_id"]
