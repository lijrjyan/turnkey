from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import load_config
from turnkey.runner import run_eval


def _run_smoke(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return run_eval(load_config(config_path), source_config_path=str(config_path))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _refresh_artifact_identity(run_dir: Path, name: str, *, count: int) -> None:
    path = run_dir / f"{name}.jsonl" if name != "metrics" else run_dir / "metrics.json"
    data = path.read_bytes()
    run_path = run_dir / "run.json"
    run = _load_json(run_path)
    run["artifacts"][name] = {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "count": count,
    }
    _write_json(run_path, run)


def _has_error(errors: list[str], text: str) -> bool:
    return any(text in error for error in errors)


def test_four_file_smoke_audits_cleanly(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)

    assert audit_run_dir(run_dir) == []


def test_audit_rejects_core_artifact_hash_drift(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    cases_path = run_dir / "cases.jsonl"
    cases = _load_jsonl(cases_path)
    cases[0]["prompt"]["sha256"] = "0" * 64
    _write_jsonl(cases_path, cases)

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "artifacts.cases.sha256")


def test_audit_rejects_missing_event_parent_after_identity_refresh(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    child = next(event for event in events if "parent_event_id" in event)
    child["parent_event_id"] = "event-missing"
    _write_jsonl(events_path, events)
    _refresh_artifact_identity(run_dir, "events", count=len(events))

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "parent_event_id does not exist")


def test_audit_rejects_invalid_event_type_and_duplicate_span_boundary(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    events[0]["kind"] = "unknown"
    events[0]["name"] = ""
    events[1]["end_sequence"] = events[0]["end_sequence"]
    _write_jsonl(events_path, events)
    _refresh_artifact_identity(run_dir, "events", count=len(events))

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "kind must be one of")
    assert _has_error(errors, "name must be non-empty string")
    assert _has_error(errors, "duplicate span boundary")


def test_audit_rejects_request_without_provider_or_cache_state(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    request = next(event for event in events if event["kind"] == "request")
    request.pop("provider")
    request.pop("cache_hit")
    request.pop("parent_event_id")
    _write_jsonl(events_path, events)
    _refresh_artifact_identity(run_dir, "events", count=len(events))

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "request provider must be non-empty string")
    assert _has_error(errors, "request cache_hit must be bool")
    assert _has_error(errors, "request parent must be the pass policy")


def test_audit_rejects_invalid_event_json_even_when_identity_matches(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    with events_path.open("a", encoding="utf-8") as file:
        file.write("{invalid-json}\n")
    _refresh_artifact_identity(run_dir, "events", count=len(events))

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "events.jsonl: invalid json")


def test_audit_recomputes_extra_forwards_from_events(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    metrics_path = run_dir / "metrics.json"
    metrics = _load_json(metrics_path)
    metrics["cost"]["extra_forwards_avg"] += 1.0
    _write_json(metrics_path, metrics)
    _refresh_artifact_identity(run_dir, "metrics", count=1)

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "metrics.cost.extra_forwards_avg")


def test_audit_recomputes_all_headline_metrics_and_reference_cost(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    metrics_path = run_dir / "metrics.json"
    metrics = _load_json(metrics_path)
    for key in ("ASR_with_detector", "ASR_reduction", "AUC", "F1", "benign_block_rate"):
        metrics[key] = -1.0
    metrics["counts"]["n_with_detector_harmful_effective"] = -1
    metrics["cost"]["avg_prompt_tokens"] = -1.0
    _write_json(metrics_path, metrics)
    _refresh_artifact_identity(run_dir, "metrics", count=1)

    errors = audit_run_dir(run_dir)

    for key in (
        "ASR_with_detector",
        "ASR_reduction",
        "AUC",
        "F1",
        "benign_block_rate",
        "counts.n_with_detector_harmful_effective",
        "cost.avg_prompt_tokens",
    ):
        assert _has_error(errors, f"metrics.{key}"), key


def test_audit_rejects_plaintext_in_public_cases(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    cases_path = run_dir / "cases.jsonl"
    cases = _load_jsonl(cases_path)
    cases[0]["prompt"]["text"] = "public plaintext"
    _write_jsonl(cases_path, cases)
    _refresh_artifact_identity(run_dir, "cases", count=len(cases))

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "public case must not contain prompt.text")


def test_audit_rejects_missing_required_judge_verdict(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    cases_path = run_dir / "cases.jsonl"
    cases = _load_jsonl(cases_path)
    harmful = next(case for case in cases if case["is_benign"] is False)
    harmful["reference"]["judge"]["is_harmful_effective"] = None
    _write_jsonl(cases_path, cases)
    _refresh_artifact_identity(run_dir, "cases", count=len(cases))

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "reference.judge.is_harmful_effective must be bool")


def test_audit_rejects_blocked_intervention_original_target_call(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    cases = _load_jsonl(run_dir / "cases.jsonl")
    blocked = next(case for case in cases if case["intervention"]["detector"]["block"] is True)
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    judge_event = next(
        event
        for event in events
        if event["case_id"] == blocked["case_id"]
        and event["pass"] == "intervention"
        and event["kind"] == "judge"
    )
    judge_event.update(
        {
            "kind": "request",
            "name": "turnkey.policy.Generate",
            "provider": "turnkey.runner.policy_executor.BackendGenerateProvider",
            "cache_hit": False,
            "model_forwards": 1,
            "request": {"prompt_sha256": blocked["prompt"]["sha256"]},
        }
    )
    _write_jsonl(events_path, events)
    _refresh_artifact_identity(run_dir, "events", count=len(events))

    errors = audit_run_dir(run_dir)

    assert _has_error(errors, "blocked intervention called original target")
