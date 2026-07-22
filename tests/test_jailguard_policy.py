from dataclasses import dataclass
from types import SimpleNamespace

from turnkey.components.detectors.jailguard import JailGuardDetector, JailGuardPolicy
from turnkey.methods import MethodContext
from turnkey.policy import Generate, PolicyChain, PolicyRequest, TargetSession
from turnkey.schema import ModelOutput, Sample


@dataclass
class _GenerateProvider:
    request_type = Generate

    def __post_init__(self) -> None:
        self.calls: list[Generate] = []

    def provide(self, request: Generate) -> ModelOutput:
        self.calls.append(request)
        return ModelOutput(
            executed=True,
            backend="jailguard-policy-test",
            model_id="target",
            response_text=f"response:{request.prompt}",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.01,
        )


class BadRequestError(RuntimeError):
    pass


@dataclass
class _BadRequestProvider(_GenerateProvider):
    def provide(self, request: Generate) -> ModelOutput:
        self.calls.append(request)
        raise BadRequestError("rejected variant")


def test_jailguard_backend_defaults_to_policy_runtime_without_own_model() -> None:
    detector = JailGuardDetector(response_mode="backend", n_variants=2)

    assert isinstance(detector.policy(), JailGuardPolicy)


def test_jailguard_echo_policy_matches_detector_decision() -> None:
    detector = _detector(response_mode="echo_prompt")
    sample = _sample()
    provider = _GenerateProvider()

    outcome = _run_policy(detector, sample, provider)
    decision = detector.decide(sample)

    assert outcome.action == "allow"
    assert outcome.score == decision.score
    assert outcome.reason == decision.reason
    assert outcome.diagnostics == decision.diagnostics
    assert provider.calls == [_target(sample)]


def test_jailguard_backend_policy_generates_variants_then_original_request() -> None:
    detector = _detector(response_mode="backend")
    sample = _sample()
    provider = _GenerateProvider()
    variants = detector._variants(sample)  # noqa: SLF001

    outcome = _run_policy(detector, sample, provider)
    expected = detector._decision_from_responses(  # noqa: SLF001
        variants=variants,
        responses=[f"response:{variant}" for variant in variants],
    )

    assert outcome.action == "allow"
    assert outcome.score == expected.score
    assert outcome.reason == expected.reason
    assert outcome.diagnostics == expected.diagnostics
    assert [call.prompt for call in provider.calls] == [*variants, sample.prompt]
    assert all(call.max_new_tokens == detector.max_new_tokens for call in provider.calls[:-1])
    assert provider.calls[-1] == _target(sample)


def test_jailguard_backend_policy_preserves_bad_request_block_fallback() -> None:
    detector = _detector(response_mode="backend", threshold=999.0)
    sample = _sample()
    provider = _BadRequestProvider()

    outcome = _run_policy(detector, sample, provider)

    assert outcome.action == "block"
    assert outcome.model is None
    assert outcome.diagnostics["all_block"] is True
    assert outcome.diagnostics["min_refusal_phrases"] >= 1
    assert len(provider.calls) == detector.n_variants


def test_jailguard_policy_uses_calibration_artifact_threshold() -> None:
    detector = _detector(response_mode="echo_prompt", threshold=999.0)
    sample = _sample()
    provider = _GenerateProvider()
    artifact = SimpleNamespace(operating_point={"threshold": -1.0})

    outcome = _run_policy(
        detector,
        sample,
        provider,
        calibration_artifact=artifact,
    )

    assert outcome.action == "block"
    assert "threshold=-1.0000" in (outcome.reason or "")
    assert provider.calls == []


def _detector(*, response_mode: str, threshold: float = 999.0) -> JailGuardDetector:
    return JailGuardDetector(
        n_variants=3,
        mutator="PI",
        char_rate=0.1,
        threshold=threshold,
        similarity="bow",
        response_mode=response_mode,
        max_new_tokens=4,
        seed=7,
    )


def _sample() -> Sample:
    return Sample(
        sample_id="jailguard-policy",
        behavior_id="fixtures:benign:jailguard-policy",
        is_benign=True,
        prompt="Explain the water cycle briefly.",
    )


def _target(sample: Sample) -> Generate:
    return Generate(
        prompt=sample.prompt,
        images=sample.images,
        max_new_tokens=16,
        temperature=0.0,
    )


def _run_policy(
    detector: JailGuardDetector,
    sample: Sample,
    provider: _GenerateProvider,
    *,
    calibration_artifact=None,  # noqa: ANN001
):
    request = PolicyRequest(sample=sample, target=_target(sample))
    with MethodContext([provider]) as context:
        return PolicyChain(
            (detector.policy(calibration_artifact=calibration_artifact),)
        ).run(request, TargetSession(context))
