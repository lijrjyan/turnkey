from __future__ import annotations

import pytest

from turnkey.metrics import compute_nsg_metrics
from turnkey.schema import AttackBudget, DetectorDecision, JudgeOutput, ModelOutput, Record, ThreatContext


def _record(
    sample_id: str,
    *,
    is_benign: bool,
    blocked: bool,
    score: float,
    baseline_harmful: bool | None = False,
    baseline_refusal: bool | None = False,
    with_detector_harmful: bool | None = False,
    attack_params: dict | None = None,
) -> tuple[Record, Record]:
    common = {
        "sample_id": sample_id,
        "behavior_id": sample_id,
        "is_benign": is_benign,
        "attack_family": "T2",
        "attack_method": "persona",
        "attack_params": dict(attack_params or {}),
        "threat": ThreatContext(tier="T2", attack_family="T2", attack_method="persona"),
        "budget": AttackBudget(bucket="none"),
        "prompt_hash": f"sha-{sample_id}",
        "prompt_chars": 12,
    }
    baseline = Record(
        **common,
        model=ModelOutput(executed=True, backend="dummy", model_id="dummy"),
        detector=DetectorDecision(block=False, score=0.0, reason="allow_all"),
        judge=JudgeOutput(is_refusal=baseline_refusal, is_harmful_effective=baseline_harmful),
    )
    with_detector = Record(
        **common,
        model=ModelOutput(executed=not blocked, backend="dummy", model_id="dummy"),
        detector=DetectorDecision(block=blocked, score=score, reason="test"),
        judge=JudgeOutput(
            is_refusal=None if blocked else baseline_refusal,
            is_harmful_effective=None if blocked else with_detector_harmful,
        ),
    )
    return baseline, with_detector


def test_metrics_include_traditional_detector_baselines() -> None:
    pairs = [
        _record(
            "harmful-success-blocked",
            is_benign=False,
            blocked=True,
            score=0.9,
            baseline_harmful=True,
            with_detector_harmful=False,
        ),
        _record(
            "harmful-refusal-allowed",
            is_benign=False,
            blocked=False,
            score=0.8,
            baseline_refusal=True,
            with_detector_harmful=False,
        ),
        _record("benign-blocked", is_benign=True, blocked=True, score=0.2),
        _record("benign-allowed", is_benign=True, blocked=False, score=0.1),
    ]

    metrics = compute_nsg_metrics(
        baseline=[baseline for baseline, _with_detector in pairs],
        with_detector=[with_detector for _baseline, with_detector in pairs],
    )

    assert metrics["ASR_strict"] == 0.5
    assert metrics["ASR_with_detector"] == 0.0
    assert metrics["ASR_reduction"] == 0.5
    assert metrics["AUC"] == 1.0
    assert metrics["F1"] == 0.5
    assert metrics["counts"]["n_with_detector_harmful_effective"] == 0
    assert metrics["judge_coverage"] == {
        "required_verdicts": 7,
        "available_verdicts": 7,
        "missing_verdicts": 0,
        "coverage_ratio": 1.0,
        "by_pass": {
            "baseline": {
                "required_verdicts": 6,
                "available_verdicts": 6,
                "missing_verdicts": 0,
                "coverage_ratio": 1.0,
            },
            "with_detector": {
                "required_verdicts": 1,
                "available_verdicts": 1,
                "missing_verdicts": 0,
                "coverage_ratio": 1.0,
            },
        },
    }
    assert metrics["groups"]["attack_family"]["T2"]["AUC"] == 1.0


def test_metrics_group_by_ngram_ppl_bucket() -> None:
    pairs = [
        _record(
            "harmful-low",
            is_benign=False,
            blocked=True,
            score=0.9,
            baseline_harmful=True,
            attack_params={"ngram_ppl_bucket": "low", "ngram_ppl": 3.2},
        ),
        _record(
            "benign-high",
            is_benign=True,
            blocked=False,
            score=0.1,
            attack_params={"ngram_ppl_bucket": "high", "ngram_ppl": 9.7},
        ),
    ]

    metrics = compute_nsg_metrics(
        baseline=[baseline for baseline, _with_detector in pairs],
        with_detector=[with_detector for _baseline, with_detector in pairs],
    )

    assert metrics["ngram_ppl_bucket_counts"] == {"high": 1, "low": 1}
    assert metrics["ngram_ppl_bucket"] == {"high": 1, "low": 1}
    assert metrics["groups"]["ngram_ppl_bucket"]["low"]["counts"]["n_samples"] == 1
    assert metrics["groups"]["ngram_ppl_bucket"]["high"]["counts"]["n_samples"] == 1


@pytest.mark.parametrize(
    ("field", "record_index"),
    [
        pytest.param("is_harmful_effective", 0, id="baseline-harmful-verdict"),
        pytest.param("is_refusal", 0, id="baseline-refusal-verdict"),
    ],
)
def test_metrics_fail_fast_on_missing_baseline_judge_verdict(
    field: str,
    record_index: int,
) -> None:
    baseline, with_detector = _record(
        "harmful",
        is_benign=False,
        blocked=False,
        score=0.1,
        baseline_harmful=True,
    )
    baseline_records = [baseline]
    object.__setattr__(baseline_records[record_index].judge, field, None)

    with pytest.raises(ValueError, match=rf"baseline.*harmful.*{field}.*missing"):
        compute_nsg_metrics(baseline=baseline_records, with_detector=[with_detector])


def test_metrics_fail_fast_on_missing_allowed_detector_judge_verdict() -> None:
    baseline, with_detector = _record(
        "harmful-allowed",
        is_benign=False,
        blocked=False,
        score=0.1,
        baseline_harmful=True,
        with_detector_harmful=None,
    )

    with pytest.raises(
        ValueError,
        match=r"with_detector.*harmful-allowed.*is_harmful_effective.*missing",
    ):
        compute_nsg_metrics(baseline=[baseline], with_detector=[with_detector])


def test_metrics_allow_missing_judge_verdicts_for_blocked_detector_pass() -> None:
    baseline, with_detector = _record(
        "harmful-blocked",
        is_benign=False,
        blocked=True,
        score=0.9,
        baseline_harmful=True,
    )

    metrics = compute_nsg_metrics(baseline=[baseline], with_detector=[with_detector])

    assert with_detector.judge.is_harmful_effective is None
    assert metrics["judge_coverage"]["by_pass"]["with_detector"]["required_verdicts"] == 0
