from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from turnkey.analysis.report import write_run_report
from turnkey.cli import main
from turnkey.export import export_cases
from turnkey.inspect import INSPECT_CATEGORIES, classify_case, inspect_run
from turnkey.config import load_config
from turnkey.runner import run_eval


def _outcome(
    *,
    block: bool,
    harmful: bool | None,
    refusal: bool | None,
) -> dict[str, Any]:
    return {
        "detector": {"block": block, "score": None, "reason": "fixture", "diagnostics": {}},
        "model": {
            "executed": not block,
            "backend": "dummy",
            "model_id": "dummy",
            "response_text": None if block else "<redacted sha256=abc chars=1>",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "latency_s": 0.0,
            "prompt_logprobs": None,
            "prefix_logprobs": None,
        },
        "judge": {
            "is_refusal": refusal,
            "is_harmful": harmful,
            "is_harmful_effective": harmful,
            "confidence": 1.0,
            "details": {"blocked": True} if block else {},
        },
        "signals": {},
    }


def _case(
    case_id: str,
    *,
    is_benign: bool,
    reference: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "turnkey_case/v1",
        "case_id": case_id,
        "behavior_id": case_id,
        "is_benign": is_benign,
        "dataset": {"name": "fixture", "version": "v1"},
        "attack_family": "none",
        "attack_method": "none",
        "attack_params": {},
        "threat": {"attack_family": "none", "attack_method": "none", "tier": "direct"},
        "budget": {"bucket": "single"},
        "prompt": {"sha256": "a" * 64, "chars": 1},
        "reference": reference,
        "intervention": intervention,
    }


def _run_smoke(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return run_eval(load_config(config_path), source_config_path=str(config_path))


def test_inspect_classifies_every_track_category() -> None:
    allow_harm = _outcome(block=False, harmful=True, refusal=False)
    allow_safe = _outcome(block=False, harmful=False, refusal=False)
    refusal = _outcome(block=False, harmful=False, refusal=True)
    blocked = _outcome(block=True, harmful=None, refusal=None)
    fixtures = {
        "saved": _case("saved", is_benign=False, reference=allow_harm, intervention=blocked),
        "missed_harm": _case(
            "missed", is_benign=False, reference=allow_harm, intervention=allow_harm
        ),
        "redundant_block": _case(
            "redundant", is_benign=False, reference=refusal, intervention=blocked
        ),
        "benign_regression": _case(
            "benign", is_benign=True, reference=allow_safe, intervention=blocked
        ),
        "safety_regression": _case(
            "regression", is_benign=False, reference=allow_safe, intervention=allow_harm
        ),
        "unsupported/error": _case(
            "unsupported", is_benign=False, reference=allow_safe, intervention=allow_safe
        ),
    }
    error_event = {
        "case_id": "unsupported",
        "status": "error",
        "error": {"type": "RuntimeError", "message": "unsupported"},
    }

    for category, case in fixtures.items():
        events = [error_event] if category == "unsupported/error" else []
        assert category in classify_case(case, events=events)
    assert set(INSPECT_CATEGORIES) == set(fixtures)


def test_inspect_filters_case_and_includes_related_events(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    cases = [json.loads(line) for line in (run_dir / "cases.jsonl").read_text().splitlines()]
    saved_id = next(
        case["case_id"]
        for case in cases
        if case["is_benign"] is False and case["intervention"]["detector"]["block"] is True
    )

    result = inspect_run(run_dir, category="saved", case_id=saved_id)

    assert result["filters"] == {"category": "saved", "case_id": saved_id}
    assert [case["case_id"] for case in result["cases"]] == [saved_id]
    assert result["cases"][0]["categories"] == ["saved"]
    assert result["cases"][0]["events"]
    assert result["artifacts"] == {
        "run": "run.json",
        "cases": "cases.jsonl",
        "events": "events.jsonl",
        "metrics": "metrics.json",
    }
    with pytest.raises(ValueError, match="unknown inspect category"):
        inspect_run(run_dir, category="unknown")


def test_report_and_export_are_optional_regenerable_views(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    report_json = tmp_path / "derived" / "report.json"
    report_md = tmp_path / "derived" / "report.md"
    jsonl_path = tmp_path / "derived" / "cases.jsonl"
    csv_path = tmp_path / "derived" / "cases.csv"

    report = write_run_report(run_dir, json_path=report_json, markdown_path=report_md)
    export_cases(run_dir, out_path=jsonl_path, output_format="jsonl")
    export_cases(run_dir, out_path=csv_path, output_format="csv")

    assert report["schema_version"] == "turnkey_report/v1"
    assert report["metrics"]["NSG_abs"] == 1.0
    assert report_json.is_file() and "# Turnkey Run Report" in report_md.read_text()
    exported = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert exported and {"case_id", "categories", "reference", "intervention"} <= set(exported[0])
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows and "categories" in rows[0]

    report_json.unlink()
    report_md.unlink()
    jsonl_path.unlink()
    csv_path.unlink()
    write_run_report(run_dir, json_path=report_json, markdown_path=report_md)
    export_cases(run_dir, out_path=csv_path, output_format="csv")
    assert report_json.is_file() and report_md.is_file() and csv_path.is_file()


def test_inspect_report_export_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _run_smoke(tmp_path)
    report_path = tmp_path / "report.json"
    export_path = tmp_path / "export.jsonl"

    assert main(["inspect", str(run_dir), "--category", "saved", "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["count"] == 2
    assert main(["report", str(run_dir), "--json", str(report_path)]) == 0
    assert capsys.readouterr().out.strip() == str(report_path)
    assert main(
        ["export", str(run_dir), "--format", "jsonl", "--out", str(export_path)]
    ) == 0
    assert capsys.readouterr().out.strip() == str(export_path)
