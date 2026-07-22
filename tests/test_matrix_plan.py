from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from turnkey.config import load_config
from turnkey.matrix import run_matrix_plan, summarize_matrix_results, write_matrix_plan

from json_fixtures import load_json


def _write_matrix_spec(path: Path, spec: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def _runtime_spec(tmp_path: Path) -> Path:
    return _write_matrix_spec(
        tmp_path / "matrix.yaml",
        {
            "schema_version": "turnkey_matrix_spec/v1",
            "name": "runtime-check",
            "defaults": {
                "run": {
                    "out_dir": str(tmp_path / "runs"),
                    "redact": True,
                    "max_samples": 2,
                },
                "content": {"shuffle": False, "limit": 2},
                "model": {"backend": "dummy", "model_id": "dummy-smoke"},
                "judge": {"name": "dummy_refusal", "params": {}},
                "nsg": {
                    "baseline_detector": {"name": "allow_all", "params": {}}
                },
            },
            "datasets": [{"name": "fixtures_smoke", "params": {"n_samples": 2}}],
            "attacks": [{"name": "none", "params": {}}],
            "detectors": [
                {
                    "name": "keyword",
                    "resource_tier": "cheap",
                    "params": {"keywords": ["UNSAFE_PLACEHOLDER"]},
                },
                {"name": "allow_all", "resource_tier": "cheap", "params": {}},
                {"name": "rcs", "resource_tier": "heavy", "params": {}},
            ],
            "skip_rules": [
                {"detector": "rcs", "reason": "heavy example lane"}
            ],
        },
    )


def test_public_example_generates_loadable_configs(tmp_path: Path) -> None:
    plan_path = write_matrix_plan(
        spec_path=Path("configs/matrix/example.yaml"),
        out_dir=tmp_path / "plan",
    )
    plan = load_json(plan_path)

    assert plan["schema_version"] == "turnkey_matrix_plan/v1"
    assert plan["name"] == "example"
    assert plan["counts"] == {"entries": 2, "planned": 2, "skipped": 0}
    assert {entry["detector"]["name"] for entry in plan["entries"]} == {
        "allow_all",
        "keyword",
    }
    for entry in plan["entries"]:
        config = load_config(plan_path.parent / entry["config_path"])
        assert config.model.backend == "dummy"
        assert config.run.max_samples == 2


def test_matrix_plan_applies_skip_rules(tmp_path: Path) -> None:
    plan_path = write_matrix_plan(
        spec_path=_runtime_spec(tmp_path),
        out_dir=tmp_path / "plan",
    )
    plan = load_json(plan_path)

    assert plan["counts"] == {"entries": 3, "planned": 2, "skipped": 1}
    skipped = next(entry for entry in plan["entries"] if entry["status"] == "skipped")
    assert skipped["reason"] == "heavy example lane"
    assert "config_path" not in skipped


def test_matrix_run_audits_and_resumes_generic_entries(tmp_path: Path) -> None:
    plan_path = write_matrix_plan(
        spec_path=_runtime_spec(tmp_path),
        out_dir=tmp_path / "plan",
    )

    results_path = run_matrix_plan(plan_path=plan_path, max_runs=1)
    first = load_json(results_path)
    first_success = next(row for row in first["results"] if row["status"] == "success")
    assert first["counts"] == {
        "entries": 3,
        "attempted": 1,
        "success": 1,
        "audit_failed": 0,
        "failed": 0,
        "skipped": 1,
        "not_run": 1,
    }
    assert Path(first_success["artifacts"]["events_jsonl"]).is_file()

    resumed_path = run_matrix_plan(plan_path=plan_path, max_runs=2, resume=True)
    resumed = load_json(resumed_path)
    resumed_first = next(
        row for row in resumed["results"] if row["id"] == first_success["id"]
    )
    assert resumed["counts"]["success"] == 2
    assert resumed_first["run_dir"] == first_success["run_dir"]

    summary = summarize_matrix_results(results_path=resumed_path)
    assert summary["schema_version"] == "turnkey_matrix_summary/v1"
    assert summary["counts"]["success"] == 2


def test_matrix_run_preserves_non_t0_threat_tier(tmp_path: Path) -> None:
    spec = yaml.safe_load(_runtime_spec(tmp_path).read_text(encoding="utf-8"))
    spec["attacks"] = [
        {
            "name": "persona",
            "params": {
                "persona": "Compliance Auditor",
                "use_markers": True,
                "attack_family": "T2",
            },
        }
    ]
    spec["detectors"] = spec["detectors"][:1]
    spec["skip_rules"] = []
    spec_path = _write_matrix_spec(tmp_path / "tiered.yaml", spec)
    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "tiered-plan")

    results = load_json(run_matrix_plan(plan_path=plan_path))
    success = next(row for row in results["results"] if row["status"] == "success")
    rows = [
        json.loads(line)
        for line in Path(success["artifacts"]["cases_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert {row["attack_family"] for row in rows} == {"T2"}
    assert {row["threat"]["tier"] for row in rows} == {"T2"}
