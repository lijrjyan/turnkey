from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnkey.cli import main
from turnkey.analysis.detector_overlap import build_detector_overlap, write_detector_overlap
from json_fixtures import write_json, write_jsonl


def _sample(sample_id: str, *, is_benign: bool, blocked: bool) -> dict:
    return {
        "case_id": sample_id,
        "behavior_id": sample_id,
        "is_benign": is_benign,
        "attack_family": "T2",
        "attack_method": "persona",
        "attack_params": {},
        "threat": {"tier": "T2", "attack_family": "T2", "attack_method": "persona"},
        "budget": {"bucket": "medium"},
        "prompt": {"sha256": f"sha-{sample_id}", "chars": 64},
        "_matrix": {"dataset": "fixtures_smoke", "attack": "persona"},
        "reference": {
            "detector": {"block": False, "score": 0.0, "reason": "allow_all"},
            "model": {"executed": True, "backend": "dummy", "model_id": "dummy"},
            "judge": {"is_refusal": False, "is_harmful_effective": not is_benign},
        },
        "intervention": {
            "detector": {"block": blocked, "score": 0.7 if blocked else 0.1, "reason": "test"},
            "model": {"executed": not blocked, "backend": "dummy", "model_id": "dummy"},
            "judge": {"is_refusal": False, "is_harmful_effective": not is_benign},
        },
    }


def _matrix_results_for_detector(
    path: Path, *, detector: str, run_dir: Path
) -> None:
    write_json(
        path,
        {
            "schema_version": "turnkey_matrix_results/v1",
            "generated_at_utc": "2026-04-30T00:00:00+00:00",
            "plan_name": f"{detector}-plan",
            "counts": {"entries": 1, "attempted": 1, "success": 1},
            "results": [
                {
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
            ],
        },
    )


def _materialize(tmp_path: Path, detector: str, sample_blocks: dict[str, tuple[bool, bool]]):
    """sample_blocks: sample_id -> (is_benign, blocked)."""
    run_dir = tmp_path / "runs" / detector
    rows = [_sample(sid, is_benign=is_benign, blocked=blocked) for sid, (is_benign, blocked) in sample_blocks.items()]
    write_jsonl(run_dir / "cases.jsonl", rows)
    write_jsonl(run_dir / "events.jsonl", [])
    write_json(run_dir / "run.json", {})
    write_json(run_dir / "metrics.json", {"counts": {"n_samples": len(rows)}, "cost": {}})
    results_path = tmp_path / f"results-{detector}.json"
    _materialize_results(results_path, detector=detector, run_dir=run_dir)
    return results_path


def _materialize_results(path: Path, *, detector: str, run_dir: Path) -> None:
    _matrix_results_for_detector(path, detector=detector, run_dir=run_dir)


def test_build_detector_overlap_basic_kappa(tmp_path: Path) -> None:
    blocks_a = {
        "h1": (False, True), "h2": (False, True), "h3": (False, False), "h4": (False, False),
        "b1": (True, False), "b2": (True, False),
    }
    blocks_b = {
        "h1": (False, True), "h2": (False, False), "h3": (False, True), "h4": (False, False),
        "b1": (True, False), "b2": (True, True),
    }
    rp_a = _materialize(tmp_path, "det_a", blocks_a)
    rp_b = _materialize(tmp_path, "det_b", blocks_b)

    analysis = build_detector_overlap(results_paths=[rp_a, rp_b], name="fixture")

    assert analysis["schema_version"] == "turnkey_detector_overlap/v1"
    assert analysis["cohort"]["detectors"] == ["det_a", "det_b"]
    assert analysis["cohort"]["joined_sample_count"] == 6
    assert analysis["cohort"]["joined_sample_harmful_count"] == 4

    [pair] = analysis["kappa_matrix"]
    assert pair["detector_a"] == "det_a" and pair["detector_b"] == "det_b"
    # both block: h1 (1); both allow: h4, b1 (2); a-only: h2 (1); b-only: h3, b2 (2). N=6.
    assert pair["n_both_block"] == 1
    assert pair["n_both_allow"] == 2
    assert pair["n_a_only_block"] == 1
    assert pair["n_b_only_block"] == 2
    # p_o = 3/6 = 0.5 ; p_a_block = 2/6 = 1/3 ; p_b_block = 3/6 = 0.5 ;
    # p_e = 1/3 * 1/2 + 2/3 * 1/2 = 1/2 ; kappa = (0.5 - 0.5) / (1 - 0.5) = 0
    assert pair["p_observed"] == 0.5
    assert pair["p_expected"] == 0.5
    assert pair["cohen_kappa"] == 0.0


def test_independent_contribution_and_ensemble(tmp_path: Path) -> None:
    blocks_a = {
        "h1": (False, True), "h2": (False, True), "h3": (False, False),
        "b1": (True, False),
    }
    blocks_b = {
        "h1": (False, True), "h2": (False, False), "h3": (False, True),
        "b1": (True, True),
    }
    rp_a = _materialize(tmp_path, "det_a", blocks_a)
    rp_b = _materialize(tmp_path, "det_b", blocks_b)
    analysis = build_detector_overlap(results_paths=[rp_a, rp_b])

    # det_a unique blocks: h2; det_b unique blocks: h3, b1. Harmful keys: h1, h2, h3.
    ic = analysis["independent_contribution"]
    assert ic["det_a"]["unique_blocks"] == 1
    assert ic["det_a"]["unique_blocks_on_harmful"] == 1
    assert ic["det_b"]["unique_blocks"] == 2
    assert ic["det_b"]["unique_blocks_on_harmful"] == 1

    ens = analysis["ensemble"]
    # Union of harmful blocks: h1, h2, h3 → 3/3
    assert ens["union_block_rate_on_harmful"] == 1.0
    # Intersection of harmful blocks: h1 only → 1/3
    assert ens["intersection_block_rate_on_harmful"] == round(1 / 3, 6)


def test_write_detector_overlap_json_markdown_and_cli(tmp_path: Path, capsys) -> None:
    blocks_a = {"h1": (False, True), "h2": (False, False), "b1": (True, False)}
    blocks_b = {"h1": (False, True), "h2": (False, True), "b1": (True, True)}
    rp_a = _materialize(tmp_path, "det_a", blocks_a)
    rp_b = _materialize(tmp_path, "det_b", blocks_b)

    out_path = tmp_path / "overlap.json"
    md_path = tmp_path / "overlap.md"
    write_detector_overlap(
        results_paths=[rp_a, rp_b], out_path=out_path, markdown_path=md_path, name="fixture"
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "turnkey_detector_overlap/v1"
    md = md_path.read_text(encoding="utf-8")
    assert "Detector Overlap" in md
    assert "Cohen κ matrix" in md

    cli_out = tmp_path / "cli-overlap.json"
    rc = main(
        [
            "analyze",
            "detector-overlap",
            "--results", str(rp_a),
            "--results", str(rp_b),
            "--out", str(cli_out),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == str(cli_out)
    assert json.loads(cli_out.read_text(encoding="utf-8"))["schema_version"] == "turnkey_detector_overlap/v1"


def test_overlap_requires_two_detectors(tmp_path: Path) -> None:
    blocks = {"h1": (False, True), "b1": (True, False)}
    rp = _materialize(tmp_path, "det_only", blocks)
    with pytest.raises(ValueError, match="≥ 2 detectors"):
        build_detector_overlap(results_paths=[rp])


def test_overlap_requires_shared_keys(tmp_path: Path) -> None:
    """Two detectors over disjoint sample IDs share no keys."""
    blocks_a = {"h1": (False, True)}
    blocks_b = {"h2": (False, True)}
    rp_a = _materialize(tmp_path, "det_a", blocks_a)
    rp_b = _materialize(tmp_path, "det_b", blocks_b)
    with pytest.raises(ValueError, match="no shared"):
        build_detector_overlap(results_paths=[rp_a, rp_b])
