from __future__ import annotations

from pathlib import Path

from turnkey.cli import main
from turnkey.analysis.matrix_analysis import build_matrix_analysis, write_matrix_analysis
from json_fixtures import load_json, write_json, write_jsonl


def _sample(
    sample_id: str,
    *,
    is_benign: bool,
    blocked: bool,
    baseline_harmful: bool = False,
    baseline_refusal: bool = False,
    detector_score: float | None = None,
    with_detector_harmful: bool | None = None,
    prompt_chars: int = 128,
    budget_bucket: str = "medium",
    naturalness_bucket: str = "high",
) -> dict:
    if detector_score is None:
        detector_score = 0.7 if blocked else 0.1
    if with_detector_harmful is None:
        with_detector_harmful = baseline_harmful
    return {
        "schema_version": "turnkey_case/v1",
        "case_id": sample_id,
        "behavior_id": sample_id,
        "is_benign": is_benign,
        "dataset": {"name": "fixtures_smoke", "version": "v1"},
        "attack_family": "T2",
        "attack_method": "persona",
        "attack_params": {"naturalness_bucket": naturalness_bucket},
        "threat": {"tier": "T2", "attack_family": "T2", "attack_method": "persona"},
        "budget": {
            "queries": None,
            "steps": None,
            "turns": None,
            "shots": 8,
            "suffix_len": None,
            "extra_forwards": None,
            "bucket": budget_bucket,
        },
        "prompt": {"sha256": f"sha-{sample_id}", "chars": prompt_chars},
        "reference": {
            "detector": {"block": False, "score": 0.0, "reason": "allow_all", "diagnostics": {}},
            "model": {"executed": True, "backend": "dummy", "model_id": "dummy"},
            "judge": {"is_refusal": baseline_refusal, "is_harmful_effective": baseline_harmful},
        },
        "intervention": {
            "detector": {"block": blocked, "score": detector_score, "reason": "test"},
            "model": {"executed": not blocked, "backend": "dummy", "model_id": "dummy"},
            "judge": {"is_refusal": baseline_refusal, "is_harmful_effective": with_detector_harmful},
        },
    }


def _benchmark_report(*, n_samples: int, threshold: float, invocations: int) -> dict:
    return {
        "overview": {"counts": {"n_samples": n_samples}},
        "cost": {
            "avg_prompt_tokens": 10.0,
            "avg_completion_tokens": 3.0,
            "avg_latency_s": 0.25,
            "extra_forwards_avg": 2.0,
            "detectors": {
                "with_detector": {
                    "name": "keyword",
                }
            },
            "provider_invocations": {
                "with_detector": {
                    "sample_count": n_samples,
                    "total_invocations": invocations,
                    "providers": [
                        {
                            "name": "model_responses",
                            "kind": "multi_generate",
                            "status": "ok",
                            "invocation_count": invocations,
                            "sample_count": n_samples,
                            "requested": ["model_responses"],
                            "materialized": ["model_responses"],
                        }
                    ],
                }
            },
        },
        "calibration_artifacts": {
            "with_detector": {
                "mode": "loaded",
                "detector_name": "keyword",
                "artifact_kind": "threshold",
                "target_model": {"model_id": "dummy", "backend": "dummy"},
                "identity": {"path": "artifact.json", "sha256": "abc", "bytes": 123},
                "threshold": threshold,
                "method": {
                    "procedure_id": "fixture",
                    "reproduction_scope": "bounded_fixture",
                    "threshold_rule": "fixture_rule",
                },
                "operating_point": {
                    "threshold": threshold,
                    "threshold_source": "fixture",
                    "calibration_mode": "bounded",
                },
            }
        },
    }


def _matrix_result(
    path: Path,
    *,
    detector: str,
    run_dir: Path,
    with_skip: bool = False,
    lofo: dict | None = None,
) -> None:
    success_row = {
        "id": f"{detector}-fixtures_smoke-persona",
        "status": "success",
        "stage": "audit",
        "run_dir": str(run_dir),
        "dataset": {"name": "fixtures_smoke"},
        "attack": {"name": "persona"},
        "detector": {"name": detector},
        "model": {"model_id": "dummy"},
        "resource_tier": "medium",
        "metrics": {},
        "audit_errors": [],
    }
    if lofo is not None:
        success_row["lofo"] = lofo
    result_rows = [success_row]
    if with_skip:
        result_rows.append(
            {
                "id": f"{detector}-fixtures_smoke-none",
                "status": "skipped",
                "stage": "plan",
                "reason": "covered elsewhere",
                "dataset": {"name": "fixtures_smoke"},
                "attack": {"name": "none"},
                "detector": {"name": detector},
                "model": {"model_id": "dummy"},
                "resource_tier": "medium",
                "audit_errors": [],
            }
        )
    write_json(
        path,
        {
            "schema_version": "turnkey_matrix_results/v1",
            "generated_at_utc": "2026-04-30T00:00:00+00:00",
            "plan_name": f"{detector}-plan",
            "counts": {"entries": len(result_rows), "attempted": 1, "success": 1},
            "results": result_rows,
        },
    )


def _write_run(run_dir: Path, rows: list[dict] | None = None) -> None:
    if rows is None:
        rows = [
            _sample("harmful-success-blocked", is_benign=False, blocked=True, baseline_harmful=True),
            _sample("harmful-refusal-blocked", is_benign=False, blocked=True, baseline_refusal=True),
            _sample("benign-blocked", is_benign=True, blocked=True),
            _sample("benign-refusal-allowed", is_benign=True, blocked=False, baseline_refusal=True),
        ]
    report = _benchmark_report(n_samples=4, threshold=0.5, invocations=8)
    write_jsonl(run_dir / "cases.jsonl", rows)
    write_jsonl(
        run_dir / "events.jsonl",
        [
            {
                "case_id": rows[index % len(rows)]["case_id"],
                "pass": "intervention",
                "kind": "request",
                "name": "fixture.Request",
                "provider": "fixture.Provider",
                "cache_hit": False,
                "status": "ok",
                "model_forwards": 1,
            }
            for index in range(8)
        ],
    )
    write_json(
        run_dir / "metrics.json",
        {
            "counts": {"n_samples": len(rows)},
            "cost": report["cost"],
        },
    )
    write_json(
        run_dir / "run.json",
        {
            "calibration_artifacts": {
                "intervention": report["calibration_artifacts"]["with_detector"]
            },
            "reproduction": {},
        },
    )


def test_matrix_analysis_exports_rich_detector_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "keyword"
    _write_run(run_dir)
    results_path = tmp_path / "results.json"
    _matrix_result(results_path, detector="keyword", run_dir=run_dir, with_skip=True)

    analysis = build_matrix_analysis(results_paths=[results_path], name="fixture-analysis")

    detector = analysis["detectors"]["keyword"]
    overall = detector["overall"]
    metrics = overall["metrics"]
    assert analysis["schema_version"] == "turnkey_matrix_analysis/v1"
    assert analysis["cohort"]["common_entry_count"] == 1
    assert detector["entry_status"]["counts"] == {"skipped": 1, "success": 1}
    assert overall["counts"]["n_samples"] == 4
    assert metrics["NSG_abs"] == 0.5
    assert metrics["NSG_rel"] == 1.0
    assert metrics["WBR"] == 0.5
    assert metrics["AUC"] == 0.75
    assert metrics["F1"] == 0.8
    assert metrics["ASR_with_detector"] == 0.0
    assert metrics["ASR_reduction"] == 0.5
    comparison = analysis["comparison_rows"][0]
    assert comparison["NSG_per_extra_forward"] == 0.25
    assert comparison["NSG_per_detector_invocation"] == 0.25
    assert metrics["harm_block_rate"] == 1.0
    assert metrics["benign_block_rate"] == 0.5
    assert metrics["net_harm_block_minus_benign_block"] == 0.5
    assert metrics["productive_block_rate"] == 0.5
    assert detector["by_budget_bucket"]["medium"]["counts"]["n_samples"] == 4
    assert detector["by_naturalness_bucket"]["high"]["counts"]["n_samples"] == 4
    assert detector["by_model"]["dummy"]["counts"]["n_samples"] == 4
    assert detector["by_prompt_length_bucket"]["short"]["counts"]["n_samples"] == 4
    assert detector["judge_outcomes"]["baseline_outcome_counts"] == {
        "benign_nonrefusal": 1,
        "benign_refusal": 1,
        "harmful_effective": 1,
        "harmful_refusal": 1,
    }
    assert detector["cost"]["provider_invocations_per_sample"] == 2.0
    assert detector["calibration"]["threshold_counts"] == {"0.5": 1}


def test_matrix_analysis_writes_json_markdown_and_cli(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "runs" / "keyword"
    _write_run(run_dir)
    results_path = tmp_path / "results.json"
    _matrix_result(
        results_path,
        detector="keyword",
        run_dir=run_dir,
        lofo={
            "mode": "leave_one_family_out",
            "holdout_attack": "persona",
            "train_attacks": ["none", "manyshot"],
        },
    )
    out_path = tmp_path / "analysis.json"
    md_path = tmp_path / "analysis.md"

    write_matrix_analysis(
        results_paths=[results_path],
        out_path=out_path,
        markdown_path=md_path,
        name="fixture-analysis",
    )

    assert load_json(out_path)["comparison_rows"][0]["detector"] == "keyword"
    assert "Detector Comparison" in md_path.read_text(encoding="utf-8")

    cli_out = tmp_path / "cli-analysis.json"
    rc = main(["matrix", "analyze", "--results", str(results_path), "--out", str(cli_out)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == str(cli_out)
    assert load_json(cli_out)["schema_version"] == "turnkey_matrix_analysis/v1"


def test_matrix_analysis_reports_traditional_vs_nsg_rankings(tmp_path: Path) -> None:
    nsg_run = tmp_path / "runs" / "nsg"
    _write_run(
        nsg_run,
        rows=[
            _sample(
                "harmful-success",
                is_benign=False,
                blocked=True,
                baseline_harmful=True,
                detector_score=0.3,
                with_detector_harmful=False,
            ),
            _sample(
                "harmful-refusal",
                is_benign=False,
                blocked=False,
                baseline_refusal=True,
                detector_score=0.2,
                with_detector_harmful=False,
            ),
            _sample("benign-low", is_benign=True, blocked=False, detector_score=0.1),
            _sample("benign-high", is_benign=True, blocked=False, detector_score=0.4),
        ],
    )
    auc_run = tmp_path / "runs" / "auc"
    _write_run(
        auc_run,
        rows=[
            _sample(
                "harmful-success",
                is_benign=False,
                blocked=False,
                baseline_harmful=True,
                detector_score=0.9,
                with_detector_harmful=True,
            ),
            _sample(
                "harmful-refusal",
                is_benign=False,
                blocked=False,
                baseline_refusal=True,
                detector_score=0.8,
                with_detector_harmful=False,
            ),
            _sample("benign-low", is_benign=True, blocked=False, detector_score=0.1),
            _sample("benign-high", is_benign=True, blocked=False, detector_score=0.2),
        ],
    )
    nsg_results = tmp_path / "nsg-results.json"
    auc_results = tmp_path / "auc-results.json"
    _matrix_result(nsg_results, detector="nsg_guard", run_dir=nsg_run)
    _matrix_result(auc_results, detector="auc_only_guard", run_dir=auc_run)

    analysis = build_matrix_analysis(results_paths=[nsg_results, auc_results], name="ranking-fixture")

    rankings = {
        row["traditional_metric"]: row
        for row in analysis["ranking_comparisons"]
        if isinstance(row, dict)
    }
    auc = rankings["AUC"]
    assert auc["target_metric"] == "NSG_abs"
    assert auc["detector_count"] == 2
    assert auc["kendall_tau"] == -1.0
    assert auc["spearman_rho"] == -1.0
    assert auc["traditional_ranking"][0]["detector"] == "auc_only_guard"
    assert auc["nsg_ranking"][0]["detector"] == "nsg_guard"
