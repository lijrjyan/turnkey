from dataclasses import replace

import pytest

from turnkey.methods import MethodContext
from turnkey.policy import (
    Component,
    DetectorPolicy,
    Generate,
    Outcome,
    PolicyChain,
    PolicyRequest,
    TargetSession,
)
from turnkey.components.detectors.keyword import KeywordDetector
from turnkey.schema import ModelOutput, Sample


class _GenerateProvider:
    request_type = Generate

    def __init__(self) -> None:
        self.calls: list[Generate] = []

    def provide(self, request: Generate) -> ModelOutput:
        self.calls.append(request)
        return ModelOutput(
            executed=True,
            backend="policy-test",
            model_id="target",
            response_text=f"target:{request.prompt}",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.01,
        )


class _BlockPolicy:
    def apply(self, request, call_next, context):  # noqa: ANN001, ANN201, ARG002
        return Outcome.blocked(
            request.target,
            score=1.0,
            reason="blocked-by-test-policy",
            diagnostics={"policy": "block"},
        )


class _RewritePolicy:
    def apply(self, request, call_next, context):  # noqa: ANN001, ANN201, ARG002
        rewritten = replace(
            request,
            target=replace(request.target, prompt=f"safe-prefix {request.target.prompt}"),
        )
        return call_next(rewritten)


class _UppercaseResponsePolicy:
    def apply(self, request, call_next, context):  # noqa: ANN001, ANN201, ARG002
        outcome = call_next(request)
        assert outcome.model is not None
        return replace(
            outcome,
            model=replace(outcome.model, response_text=(outcome.model.response_text or "").upper()),
        )


class _ChooseSecondPolicy:
    def apply(self, request, call_next, context):  # noqa: ANN001, ANN201, ARG002
        first = call_next(
            replace(request, target=replace(request.target, prompt=f"{request.target.prompt}:a"))
        )
        second = call_next(
            replace(request, target=replace(request.target, prompt=f"{request.target.prompt}:b"))
        )
        return replace(
            second,
            diagnostics={
                "candidates": [
                    first.model.response_text if first.model is not None else None,
                    second.model.response_text if second.model is not None else None,
                ]
            },
        )


def test_target_session_reuses_identical_generate_requests() -> None:
    provider = _GenerateProvider()
    request = _policy_request()

    with MethodContext([provider]) as context:
        session = TargetSession(context)
        first = session.run(request)
        second = session.run(request)

        assert first.action == second.action == "allow"
        assert first.model is second.model
        assert [use.cache_hit for use in context.uses] == [False, True]

    assert provider.calls == [request.target]


def test_policy_chain_can_block_without_calling_target() -> None:
    provider = _GenerateProvider()
    with MethodContext([provider]) as context:
        outcome = PolicyChain((_BlockPolicy(),)).run(
            _policy_request(),
            TargetSession(context),
        )

    assert outcome.action == "block"
    assert outcome.model is None
    assert outcome.score == 1.0
    assert outcome.reason == "blocked-by-test-policy"
    assert outcome.diagnostics == {"policy": "block"}
    assert provider.calls == []


def test_detector_policy_preserves_detector_decision_without_a_separate_kernel() -> None:
    provider = _GenerateProvider()
    detector = KeywordDetector(("original",))
    with MethodContext([provider]) as context:
        outcome = PolicyChain((DetectorPolicy(detector.decide),)).run(
            _policy_request(),
            TargetSession(context),
        )

    assert outcome.action == "block"
    assert outcome.model is None
    assert outcome.score == 1.0
    assert outcome.reason == "keyword:original"
    assert outcome.diagnostics == {}
    assert provider.calls == []


def test_policy_chain_can_rewrite_requests_and_transform_responses() -> None:
    provider = _GenerateProvider()
    with MethodContext([provider]) as context:
        outcome = PolicyChain((_RewritePolicy(), _UppercaseResponsePolicy())).run(
            _policy_request(),
            TargetSession(context),
        )

    assert [call.prompt for call in provider.calls] == ["safe-prefix original"]
    assert outcome.target.prompt == "safe-prefix original"
    assert outcome.model is not None
    assert outcome.model.response_text == "TARGET:SAFE-PREFIX ORIGINAL"


def test_policy_chain_can_call_target_multiple_times_and_select_an_outcome() -> None:
    provider = _GenerateProvider()
    with MethodContext([provider]) as context:
        outcome = PolicyChain((_ChooseSecondPolicy(),)).run(
            _policy_request(),
            TargetSession(context),
        )

    assert [call.prompt for call in provider.calls] == ["original:a", "original:b"]
    assert outcome.target.prompt == "original:b"
    assert outcome.model is not None
    assert outcome.model.response_text == "target:original:b"
    assert outcome.diagnostics == {
        "candidates": ["target:original:a", "target:original:b"]
    }


def test_component_contains_policy_providers_and_effective_parameters() -> None:
    provider = _GenerateProvider()
    policy = PolicyChain(())

    component = Component(
        name="external-method",
        policy=policy,
        providers=(provider,),
        parameters={"threshold": 0.5},
    )

    assert component.name == "external-method"
    assert component.policy is policy
    assert component.providers == (provider,)
    assert component.parameters == {"threshold": 0.5}


def test_outcome_rejects_inconsistent_action_and_model_state() -> None:
    target = _policy_request().target
    model = _GenerateProvider().provide(target)

    with pytest.raises(ValueError, match="allow outcome requires a model output"):
        Outcome(action="allow", target=target)
    with pytest.raises(ValueError, match="block outcome cannot contain a model output"):
        Outcome(action="block", target=target, model=model)


def _policy_request() -> PolicyRequest:
    return PolicyRequest(
        sample=Sample(
            sample_id="policy-sample",
            behavior_id="policy:behavior",
            is_benign=True,
            prompt="original",
        ),
        target=Generate(prompt="original", max_new_tokens=8, temperature=0.0),
    )
