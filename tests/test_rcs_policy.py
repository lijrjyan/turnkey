from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from turnkey.components.detectors import rcs
from turnkey.components.detectors.rcs import RCSDetector
from turnkey.methods import MethodContext
from turnkey.policy import Generate, Outcome, PolicyRequest
from turnkey.runtime_providers import (
    LastTokenHiddenStateRequest,
    LastTokenHiddenStateResult,
    ProviderSummary,
)
from turnkey.schema import DetectorDecision, ImageInput, ModelOutput, Sample


@dataclass
class _HiddenStateProvider:
    request_type = LastTokenHiddenStateRequest

    def __post_init__(self) -> None:
        self.requests: list[LastTokenHiddenStateRequest] = []

    def provide(self, request: LastTokenHiddenStateRequest) -> LastTokenHiddenStateResult:
        self.requests.append(request)
        return LastTokenHiddenStateResult(
            last_token_by_layer=(request.prompt, request.images),
            n_layers=2,
            hidden_size=3,
            device="cpu",
            provider=ProviderSummary(
                name="last_token_hidden_states",
                kind="hidden_states",
                requested=("last_token_by_layer",),
                materialized=("last_token_by_layer",),
                status="ok",
            ),
        )


def _generated(request: PolicyRequest) -> Outcome:
    return Outcome.generated(
        request.target,
        ModelOutput(
            executed=True,
            backend="test",
            model_id="target",
            response_text="OK",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.01,
        ),
    )


def test_rcs_toy_policy_preserves_detector_decision() -> None:
    detector = RCSDetector(
        mode="toy",
        method="mcd",
        threshold=0.0,
        prototype_image_count=1,
    )
    sample = Sample(
        "unsafe",
        "behavior",
        False,
        "UNSAFE_PLACEHOLDER: request redacted.",
        images=(ImageInput(path="image.png"),),
    )
    expected = detector.decide(sample)
    target_calls: list[PolicyRequest] = []

    def call_next(request: PolicyRequest) -> Outcome:
        target_calls.append(request)
        return _generated(request)

    with MethodContext(()) as context:
        outcome = detector.policy().apply(
            PolicyRequest(sample=sample, target=Generate(prompt=sample.prompt, images=sample.images)),
            call_next,
            context,
        )

    assert outcome.action == ("block" if expected.block else "allow")
    assert outcome.score == expected.score
    assert outcome.reason == expected.reason
    assert len(target_calls) == (0 if expected.block else 1)


def test_rcs_paper_policy_fits_and_scores_through_typed_hidden_state_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = RCSDetector(
        mode="paper",
        method="kcd",
        model={"model_id": "hidden-model", "device": "cpu", "local_files_only": True},
        layer=1,
        calibrate_threshold=False,
        val_ratio=0.0,
        projection_dim=2,
        projection_epochs=1,
        benign_prompts=["benign 0", "benign 1"],
        malicious_prompts=["malicious 0", "malicious 1"],
    )
    scoring = SimpleNamespace(threshold=0.0, k=1)
    fit_values: list[object] = []
    decision_values: list[object] = []

    def fake_fit(**kwargs):  # noqa: ANN003, ANN202
        for index, example in enumerate(kwargs["examples"]):
            sample = Sample(
                sample_id=f"train-{index}",
                behavior_id="rcs:train",
                is_benign=example.is_benign,
                prompt=example.prompt,
                images=example.images,
            )
            fit_values.append(kwargs["hidden_state_provider"](sample))
        return scoring

    def fake_decide(state, hidden_states, sample, *, device):  # noqa: ANN001, ANN202, ARG001
        assert state is scoring
        decision_values.append(hidden_states)
        return DetectorDecision(
            block=True,
            score=0.75,
            reason="paper typed decision",
            diagnostics={"rcs_layer_selection": {"strategy": "fixed", "selected_layer": 1}},
        )

    monkeypatch.setattr(rcs, "_fit_paper_scoring_state", fake_fit)
    monkeypatch.setattr(rcs, "_decide_paper_from_hidden_states", fake_decide)
    provider = _HiddenStateProvider()
    sample = Sample(
        "eval",
        "behavior",
        False,
        "evaluate",
        images=(ImageInput(path="eval.png"),),
    )

    with MethodContext((provider,)) as context:
        outcome = detector.policy().apply(
            PolicyRequest(sample=sample, target=Generate(prompt=sample.prompt, images=sample.images)),
            _generated,
            context,
        )

    assert outcome.action == "block"
    assert outcome.score == 0.75
    assert outcome.reason == "paper typed decision"
    assert outcome.diagnostics == {
        "rcs_layer_selection": {"strategy": "fixed", "selected_layer": 1}
    }
    assert len(fit_values) == 4
    assert len(decision_values) == 1
    assert decision_values[0].last_token_by_layer == (  # type: ignore[union-attr]
        "evaluate",
        (ImageInput(path="eval.png"),),
    )
    assert [request.prompt for request in provider.requests] == [
        "benign 0",
        "benign 1",
        "malicious 0",
        "malicious 1",
        "evaluate",
    ]
    assert all(request.config.model_id == "hidden-model" for request in provider.requests)


def test_rcs_paper_method_provider_is_exclusive_and_config_keyed() -> None:
    detector = RCSDetector(
        mode="paper",
        method="kcd",
        model={"model_id": "hidden-model", "device": "cpu", "local_files_only": True},
        layer=1,
        benign_prompts=["benign 0", "benign 1"],
        malicious_prompts=["malicious 0", "malicious 1"],
    )

    providers = detector.method_providers()

    assert len(providers) == 1
    assert providers[0].request_type is LastTokenHiddenStateRequest
    assert providers[0].model_forwards_per_call == 1
    assert providers[0].requires_exclusive_target is True


def test_rcs_paper_policy_loads_calibration_state_without_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = RCSDetector(
        mode="paper",
        method="kcd",
        model={"model_id": "hidden-model", "device": "cpu", "local_files_only": True},
        layer=1,
        benign_prompts=["benign 0", "benign 1"],
        malicious_prompts=["malicious 0", "malicious 1"],
    )
    artifact = object()
    scoring = SimpleNamespace(threshold=0.25, k=2)
    loaded: list[object] = []

    monkeypatch.setattr(rcs, "_scoring_state_path_from_artifact", lambda value: loaded.append(value) or "state.pt")
    monkeypatch.setattr(rcs, "_load_rcs_paper_scoring_state", lambda path: scoring if path == "state.pt" else None)

    def fail_fit(**_kwargs):  # noqa: ANN003, ANN202
        raise AssertionError("calibration state must bypass fitting")

    monkeypatch.setattr(rcs, "_fit_paper_scoring_state", fail_fit)
    monkeypatch.setattr(
        rcs,
        "_decide_paper_from_hidden_states",
        lambda state, hidden_states, sample, device: DetectorDecision(  # noqa: ARG005
            block=False,
            score=0.1,
            reason="calibrated paper decision",
        ),
    )
    provider = _HiddenStateProvider()
    sample = Sample("eval", "behavior", True, "evaluate")

    with MethodContext((provider,)) as context:
        outcome = detector.policy(calibration_artifact=artifact).apply(
            PolicyRequest(sample=sample, target=Generate(prompt=sample.prompt)),
            _generated,
            context,
        )

    assert outcome.action == "allow"
    assert outcome.reason == "calibrated paper decision"
    assert loaded == [artifact]
    assert [request.prompt for request in provider.requests] == ["evaluate"]
    assert detector.threshold == 0.25
    assert detector.k == 2
