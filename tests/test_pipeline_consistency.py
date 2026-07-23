from __future__ import annotations

import json
from pathlib import Path

import yaml

from turnkey.config import load_config
from turnkey.runner import run_eval
from turnkey.validation import validate_cases_jsonl


def _run_smoke(tmp_path: Path, config_name: str = "smoke.yaml") -> Path:
    raw = yaml.safe_load(Path(f"configs/runs/{config_name}").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    cfg_path = tmp_path / config_name
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return run_eval(load_config(cfg_path))


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_pipeline_consistency_passes_for_smoke(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    assert validate_cases_jsonl(run_dir / "cases.jsonl") == []


def test_pipeline_consistency_passes_for_content_and_attack_smoke(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke_persona.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["content"] = {"sample_ids": ["fx-0003", "fx-0001"], "limit": 2}
    cfg_path = tmp_path / "content_persona.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    run_dir = run_eval(load_config(cfg_path))
    assert validate_cases_jsonl(run_dir / "cases.jsonl") == []


def test_pipeline_consistency_fails_on_mismatched_schema_version(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    path = run_dir / "cases.jsonl"
    rows = _load_rows(path)
    rows[0]["schema_version"] = "turnkey_case/v2"
    _write_rows(path, rows)

    errors = validate_cases_jsonl(path)
    assert any("schema_version" in error for error in errors)


def test_pipeline_consistency_fails_on_blocked_path_model_execution(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    path = run_dir / "cases.jsonl"
    rows = _load_rows(path)
    blocked = next(row for row in rows if row["intervention"]["detector"]["block"] is True)
    blocked["intervention"]["model"]["executed"] = True
    _write_rows(path, rows)

    errors = validate_cases_jsonl(path)
    assert any("blocked path must have model.executed=false" in error for error in errors)


def test_pipeline_consistency_fails_on_unredacted_response(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    path = run_dir / "cases.jsonl"
    rows = _load_rows(path)
    rows[0]["reference"]["model"]["response_text"] = "OK"
    _write_rows(path, rows)

    errors = validate_cases_jsonl(path)
    assert any("response_text is not redacted" in error for error in errors)


def test_pipeline_consistency_fails_on_threat_identity_drift(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    path = run_dir / "cases.jsonl"
    rows = _load_rows(path)
    rows[0]["threat"]["attack_method"] = "drifted"
    _write_rows(path, rows)

    errors = validate_cases_jsonl(path)
    assert any("threat.attack_method" in error for error in errors)


def test_validation_rejects_missing_metric_judge_verdict_without_consistency_flag(
    tmp_path: Path,
) -> None:
    run_dir = _run_smoke(tmp_path)
    path = run_dir / "cases.jsonl"
    rows = _load_rows(path)
    harmful = next(row for row in rows if row["is_benign"] is False)
    harmful["reference"]["judge"]["is_harmful_effective"] = None
    _write_rows(path, rows)

    errors = validate_cases_jsonl(path)

    assert any("reference.judge.is_harmful_effective must be bool for metrics" in error for error in errors)


def test_validation_allows_missing_judge_verdicts_for_blocked_detector_pass(
    tmp_path: Path,
) -> None:
    run_dir = _run_smoke(tmp_path)
    path = run_dir / "cases.jsonl"
    rows = _load_rows(path)
    blocked = next(row for row in rows if row["intervention"]["detector"]["block"] is True)
    blocked["intervention"]["judge"]["is_refusal"] = None
    blocked["intervention"]["judge"]["is_harmful_effective"] = None
    _write_rows(path, rows)

    assert validate_cases_jsonl(path) == []
