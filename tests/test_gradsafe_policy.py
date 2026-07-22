from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from turnkey.components.detectors.gradsafe import GradSafeDetector, GradSafePolicy
from turnkey.methods import MethodContext
from turnkey.policy import Generate, PolicyChain, PolicyRequest, TargetSession
from turnkey.runtime_providers import (
    GradientScoreConfig,
    GradientScoreRequest,
    GradientScoreRequestProvider,
    GradientScoreResult,
    HFGradientScoreProvider,
    ProviderSummary,
)
from turnkey.schema import ModelOutput, Sample


class _FakeHFProvider:
    def __init__(self, cfg: GradientScoreConfig, events: list[str]) -> None:
        self.cfg = cfg
        self.events = events
        self.calls: list[tuple[str, str]] = []
        events.append(f"open:{cfg.model_id}")

    def score(self, *, prompt: str, anchor_response: str) -> GradientScoreResult:
        self.calls.append((prompt, anchor_response))
        return _gradient_result(0.8)

    def close(self) -> None:
        self.events.append(f"close:{self.cfg.model_id}")


class _FailingCloseHFProvider(_FakeHFProvider):
    def close(self) -> None:
        super().close()
        raise RuntimeError("close failed")


@dataclass
class _GradientProvider:
    request_type = GradientScoreRequest
    result: GradientScoreResult

    def __post_init__(self) -> None:
        self.requests: list[GradientScoreRequest] = []

    def provide(self, request: GradientScoreRequest) -> GradientScoreResult:
        self.requests.append(request)
        return self.result


@dataclass
class _GenerateProvider:
    request_type = Generate

    def __post_init__(self) -> None:
        self.calls: list[Generate] = []

    def provide(self, request: Generate) -> ModelOutput:
        self.calls.append(request)
        return ModelOutput(
            executed=True,
            backend="gradsafe-policy-test",
            model_id="target",
            response_text="OK",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.01,
        )


def test_gradsafe_defaults_to_typed_policy_runtime() -> None:
    detector = GradSafeDetector()

    assert isinstance(detector.policy(), GradSafePolicy)
    assert len(detector.method_providers()) == 1


def test_hf_gradient_score_delegates_to_shared_gradient_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    provider = HFGradientScoreProvider(
        GradientScoreConfig(
            model_id="gradient-model",
            device="cpu",
            normalize_by_tokens=True,
        )
    )
    calls: list[tuple[str, str]] = []

    def fake_gradient_tensors(*, prompt: str, anchor_response: str):
        calls.append((prompt, anchor_response))
        return {"layer.weight": torch.tensor([3.0, 4.0])}, 5

    monkeypatch.setattr(provider, "gradient_tensors", fake_gradient_tensors, raising=False)

    result = provider.score(prompt="probe", anchor_response="Sure")

    assert result.score == pytest.approx(1.0)
    assert result.target_tokens == 5
    assert result.n_tensors == 1
    assert calls == [("probe", "Sure")]


def test_gradient_request_provider_reuses_configured_hf_provider_and_closes() -> None:
    events: list[str] = []
    built: list[_FakeHFProvider] = []

    def build(cfg: GradientScoreConfig) -> _FakeHFProvider:
        provider = _FakeHFProvider(cfg, events)
        built.append(provider)
        return provider

    cfg = GradientScoreConfig(model_id="gradient-model", device="cpu")
    provider = GradientScoreRequestProvider(factory=build)
    first = GradientScoreRequest(cfg, prompt="first", anchor_response="Sure")
    second = GradientScoreRequest(cfg, prompt="second", anchor_response="Sure")

    with MethodContext([provider]) as context:
        assert context.get(first).score == 0.8
        assert context.get(first).score == 0.8
        assert context.get(second).score == 0.8

    assert len(built) == 1
    assert built[0].calls == [("first", "Sure"), ("second", "Sure")]
    assert events == ["open:gradient-model", "close:gradient-model"]


def test_gradient_request_provider_closes_all_configs_after_one_failure() -> None:
    events: list[str] = []

    def build(cfg: GradientScoreConfig):  # noqa: ANN202
        provider_type = _FailingCloseHFProvider if cfg.model_id == "first" else _FakeHFProvider
        return provider_type(cfg, events)

    provider = GradientScoreRequestProvider(factory=build)
    provider.provide(
        GradientScoreRequest(
            GradientScoreConfig(model_id="first"),
            prompt="one",
            anchor_response="Sure",
        )
    )
    provider.provide(
        GradientScoreRequest(
            GradientScoreConfig(model_id="second"),
            prompt="two",
            anchor_response="Sure",
        )
    )

    with pytest.raises(RuntimeError, match="failed to close 1 gradient score provider"):
        provider.close()

    assert events[-2:] == ["close:second", "close:first"]
    provider.close()


def test_gradsafe_policy_uses_typed_score_and_preserves_diagnostics() -> None:
    detector = GradSafeDetector(
        model_id="gradient-model",
        threshold=0.5,
        device="cpu",
    )
    gradient = _GradientProvider(_gradient_result(0.8))
    target = _GenerateProvider()

    outcome = _run_policy(detector, gradient, target)

    assert outcome.action == "block"
    assert outcome.model is None
    assert outcome.score == 0.8
    assert outcome.diagnostics == {
        "gradient_score": 0.8,
        "gradient_norm": 0.8,
        "score_mode": "gradient_norm",
        "target_tokens": 3,
        "selected_parameters": 2,
    }
    assert gradient.requests[0].config.model_id == "gradient-model"
    assert gradient.requests[0].prompt == _sample().prompt
    assert target.calls == []


def test_gradsafe_policy_applies_calibration_threshold_and_reference_file() -> None:
    detector = GradSafeDetector(
        model_id="gradient-model",
        threshold=0.95,
        device="cpu",
        score_mode="reference_cosine",
    )
    gradient = _GradientProvider(_gradient_result(0.8, n_features=7))
    target = _GenerateProvider()
    artifact = SimpleNamespace(
        threshold=0.5,
        operating_point={"score_mode": "reference_cosine"},
        files={"reference": {"path": "/tmp/reference.pt"}},
    )

    outcome = _run_policy(
        detector,
        gradient,
        target,
        calibration_artifact=artifact,
    )

    assert outcome.action == "block"
    assert outcome.diagnostics["gradient_cosine_score"] == 0.8
    assert outcome.diagnostics["reference_cosine_features"] == 7
    assert gradient.requests[0].config.reference_artifact == "/tmp/reference.pt"


def _gradient_result(score: float, *, n_features: int | None = None) -> GradientScoreResult:
    return GradientScoreResult(
        score=score,
        target_tokens=3,
        n_tensors=2,
        provider=ProviderSummary(
            name="gradient_score",
            kind="gradients",
            requested=("anchor_loss_gradient",),
            materialized=("gradient_norm",),
            capabilities={"n_features": n_features},
            status="ok",
        ),
    )


def _sample() -> Sample:
    return Sample(
        sample_id="gradsafe-policy",
        behavior_id="fixtures:unsafe:gradsafe-policy",
        is_benign=False,
        prompt="UNSAFE_PLACEHOLDER: request redacted.",
    )


def _run_policy(
    detector: GradSafeDetector,
    gradient: _GradientProvider,
    target: _GenerateProvider,
    *,
    calibration_artifact=None,  # noqa: ANN001
):
    sample = _sample()
    request = PolicyRequest(
        sample=sample,
        target=Generate(prompt=sample.prompt, max_new_tokens=16),
    )
    with MethodContext([gradient, target]) as context:
        return PolicyChain(
            (detector.policy(calibration_artifact=calibration_artifact),)
        ).run(request, TargetSession(context))
