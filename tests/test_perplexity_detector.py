from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import DetectorConfig, load_config
from turnkey.methods import MethodContext
from turnkey.policy import Outcome, PolicyRequest
from turnkey.runtime_providers import PromptLogprobsRequest
from turnkey.runner import run_eval
from turnkey.schema import ModelOutput, PromptLogprobs, Sample, TokenLogprob


@dataclass
class _StaticPromptLogprobsProvider:
    value: PromptLogprobs
    request_type = PromptLogprobsRequest
    model_forwards_per_call = 1

    def provide(self, request: PromptLogprobsRequest) -> PromptLogprobs:  # noqa: ARG002
        return self.value


def _detector_api():
    from turnkey.components.detectors import available_detectors, load_detector
    from turnkey.components.detectors.perplexity import PerplexityDetector

    return available_detectors, load_detector, PerplexityDetector


def _sample() -> Sample:
    return Sample(
        sample_id="sample-1",
        behavior_id="behavior-1",
        is_benign=False,
        prompt="probe prompt",
    )


def _request() -> PolicyRequest:
    from turnkey.policy import Generate

    return PolicyRequest(sample=_sample(), target=Generate(prompt="probe prompt"))


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


def _logprobs(average: float | None) -> PromptLogprobs:
    return PromptLogprobs(
        tokens=(
            TokenLogprob(token="probe", token_id=1, logprob=None),
            TokenLogprob(token=" prompt", token_id=2, logprob=average),
        ),
        logprob_sum=average,
        logprob_avg=average,
    )


def test_perplexity_detector_is_registered_with_stable_name() -> None:
    available_detectors, load_detector, PerplexityDetector = _detector_api()

    assert "perplexity" in available_detectors()
    detector = load_detector(DetectorConfig(name="perplexity", params={"threshold": 5.0}))
    assert isinstance(detector, PerplexityDetector)
    assert detector.effective_parameters() == {"threshold": 5.0}


@pytest.mark.parametrize("threshold", [0.0, -1.0, math.inf, math.nan])
def test_perplexity_detector_rejects_non_positive_or_non_finite_threshold(
    threshold: float,
) -> None:
    _, _, PerplexityDetector = _detector_api()

    with pytest.raises(ValueError, match="finite and greater than zero"):
        PerplexityDetector(threshold=threshold)


def test_perplexity_policy_blocks_strictly_above_threshold_without_generation() -> None:
    _, _, PerplexityDetector = _detector_api()
    detector = PerplexityDetector(threshold=1.5)
    called = False

    def call_next(request: PolicyRequest) -> Outcome:
        nonlocal called
        called = True
        return _generated(request)

    with MethodContext((_StaticPromptLogprobsProvider(_logprobs(-0.5)),)) as context:
        outcome = detector.policy().apply(_request(), call_next, context)

    assert called is False
    assert outcome.action == "block"
    assert outcome.score == pytest.approx(math.exp(0.5))
    assert outcome.diagnostics == {
        "threshold": 1.5,
        "mean_logprob": -0.5,
        "scored_tokens": 1,
    }


def test_perplexity_policy_allows_score_equal_to_threshold() -> None:
    _, _, PerplexityDetector = _detector_api()
    threshold = math.exp(0.5)
    request = _request()

    with MethodContext((_StaticPromptLogprobsProvider(_logprobs(-0.5)),)) as context:
        outcome = (
            PerplexityDetector(threshold=threshold)
            .policy()
            .apply(
                request,
                _generated,
                context,
            )
        )

    assert outcome.action == "allow"
    assert outcome.model is not None
    assert outcome.score == pytest.approx(threshold)


@pytest.mark.parametrize("average", [None, math.inf, math.nan, -1000.0])
def test_perplexity_policy_rejects_unusable_mean_logprob(average: float | None) -> None:
    _, _, PerplexityDetector = _detector_api()

    with MethodContext((_StaticPromptLogprobsProvider(_logprobs(average)),)) as context:
        with pytest.raises(ValueError, match="finite mean logprob|finite perplexity"):
            PerplexityDetector(threshold=5.0).policy().apply(
                _request(),
                _generated,
                context,
            )


def test_perplexity_runner_records_one_extra_forward_and_audits(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["name"] = "perplexity-smoke"
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["run"]["max_samples"] = 2
    raw["detector"] = {"name": "perplexity", "params": {"threshold": 1.0}}
    cfg_path = tmp_path / "perplexity.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(cfg_path))

    assert audit_run_dir(run_dir) == []
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics["cost"]["extra_forwards_avg"] == pytest.approx(1.0)
    assert all(row["intervention"]["detector"]["block"] is True for row in rows)
    assert all(row["intervention"]["model"]["executed"] is False for row in rows)
