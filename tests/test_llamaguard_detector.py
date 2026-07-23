from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import DetectorConfig, load_config
from turnkey.methods import MethodContext
from turnkey.policy import Generate, Outcome, PolicyRequest
from turnkey.runner import run_eval
from turnkey.schema import ModelOutput, Sample


def _detector_api():
    from turnkey.components.detectors import available_detectors, load_detector
    from turnkey.components.detectors.llamaguard import LlamaGuardDetector
    from turnkey.components.llamaguard_runtime import (
        LlamaGuardInputProvider,
        LlamaGuardInputRequest,
        LlamaGuardResult,
    )

    return (
        available_detectors,
        load_detector,
        LlamaGuardDetector,
        LlamaGuardInputProvider,
        LlamaGuardInputRequest,
        LlamaGuardResult,
    )


def _request() -> PolicyRequest:
    sample = Sample(
        sample_id="sample-1",
        behavior_id="behavior-1",
        is_benign=False,
        prompt="classify this prompt",
    )
    return PolicyRequest(sample=sample, target=Generate(prompt=sample.prompt))


def _generated(request: PolicyRequest) -> Outcome:
    return Outcome.generated(
        request.target,
        ModelOutput(
            executed=True,
            backend="test",
            model_id="target",
            response_text="OK",
            prompt_tokens=2,
            completion_tokens=1,
            total_tokens=3,
            latency_s=0.0,
        ),
    )


def test_llamaguard_detector_is_registered_without_loading_model() -> None:
    (
        available_detectors,
        load_detector,
        LlamaGuardDetector,
        *_,
    ) = _detector_api()

    assert "llamaguard" in available_detectors()
    detector = load_detector(
        DetectorConfig(
            name="llamaguard",
            params={
                "revision": "acf7aafa60f0410f8f42b1fa35e077d705892029",
                "require_token": False,
                "local_files_only": True,
            },
        )
    )
    assert isinstance(detector, LlamaGuardDetector)
    [provider] = detector.method_providers()
    assert provider.model_forwards_per_call == 1
    assert provider.requires_exclusive_target is True


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"fail_closed": 1}, "fail_closed"),
    ],
)
def test_llamaguard_detector_rejects_invalid_parameters(
    params: dict[str, object],
    message: str,
) -> None:
    *_, LlamaGuardDetector, _provider, _request_type, _result = _detector_api()

    with pytest.raises((TypeError, ValueError), match=message):
        LlamaGuardDetector(**params)


@pytest.mark.parametrize(
    ("label", "fail_closed", "expected_action", "expected_score"),
    [
        ("unsafe", True, "block", 1.0),
        ("safe", True, "allow", 0.0),
        (None, True, "block", None),
        (None, False, "allow", None),
    ],
)
def test_llamaguard_detector_classifies_user_prompt_only(
    label: str | None,
    fail_closed: bool,
    expected_action: str,
    expected_score: float | None,
) -> None:
    (
        _available,
        _load,
        LlamaGuardDetector,
        _provider,
        LlamaGuardInputRequest,
        LlamaGuardResult,
    ) = _detector_api()
    requests: list[object] = []

    @dataclass
    class StaticProvider:
        request_type = LlamaGuardInputRequest
        model_forwards_per_call = 1
        requires_exclusive_target = True

        def provide(self, request):  # noqa: ANN001
            requests.append(request)
            return LlamaGuardResult(label=label, categories=("S2",) if label == "unsafe" else ())

    detector = LlamaGuardDetector(fail_closed=fail_closed, require_token=False)
    with MethodContext((StaticProvider(),)) as context:
        outcome = detector.policy().apply(_request(), _generated, context)

    assert outcome.action == expected_action
    assert outcome.score == expected_score
    assert len(requests) == 1
    assert requests[0].prompt == "classify this prompt"
    assert outcome.diagnostics["safety_label"] == label
    assert outcome.diagnostics["categories"] == (["S2"] if label == "unsafe" else [])
    assert "classify this prompt" not in str(outcome.diagnostics)


def test_llamaguard_input_provider_uses_user_only_messages_and_closes_runtime() -> None:
    (
        _available,
        _load,
        _detector,
        LlamaGuardInputProvider,
        LlamaGuardInputRequest,
        LlamaGuardResult,
    ) = _detector_api()

    class FakeRuntime:
        def __init__(self) -> None:
            self.messages = None
            self.closed = False

        def classify(self, messages):  # noqa: ANN001
            self.messages = messages
            return LlamaGuardResult(label="safe", categories=())

        def close(self) -> None:
            self.closed = True

    runtime = FakeRuntime()
    provider = LlamaGuardInputProvider(runtime)

    result = provider.provide(LlamaGuardInputRequest(prompt="user prompt"))
    provider.close()

    assert result.label == "safe"
    assert runtime.messages == ({"role": "user", "content": "user prompt"},)
    assert runtime.closed is True


def test_llamaguard_runner_stages_target_release_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from turnkey.components.detectors import llamaguard as detector_module
    from turnkey.components.llamaguard_runtime import LlamaGuardResult

    events: list[str] = []

    class FakeRuntime:
        def __init__(self, **kwargs):  # noqa: ANN003
            events.append(f"runtime:init:{kwargs['revision']}")

        def classify(self, messages):  # noqa: ANN001
            events.append(f"runtime:classify:{messages[0]['content']}")
            return LlamaGuardResult(label="unsafe", categories=("S2",))

        def close(self) -> None:
            events.append("runtime:close")

    monkeypatch.setattr(detector_module, "LlamaGuardRuntime", FakeRuntime)
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["name"] = "llamaguard-detector-smoke"
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["run"]["max_samples"] = 2
    raw["detector"] = {
        "name": "llamaguard",
        "params": {
            "revision": "acf7aafa60f0410f8f42b1fa35e077d705892029",
            "require_token": False,
        },
    }
    cfg_path = tmp_path / "llamaguard_detector.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(cfg_path))

    assert audit_run_dir(run_dir) == []
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cost"]["extra_forwards_avg"] == pytest.approx(1.0)
    assert events[0].startswith("runtime:init:acf7aafa")
    assert sum(event.startswith("runtime:classify:") for event in events) == 2
    assert events[-1] == "runtime:close"
