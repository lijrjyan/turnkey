from __future__ import annotations

from pathlib import Path
from typing import Any

from turnkey._internal.data import string_list as _string_list
from turnkey._internal.io import load_json_object, load_jsonl_objects
from turnkey.matrix import load_matrix_results


def _load_results_document(path: Path) -> dict[str, Any]:
    document = load_matrix_results(path)
    document["_results_path"] = str(path)
    return document


def _load_success_run_records(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for document in documents:
        for result in document.get("results", []):
            if not isinstance(result, dict) or result.get("status") != "success":
                continue
            run_dir_raw = result.get("run_dir")
            if not isinstance(run_dir_raw, str):
                continue
            run_dir = Path(run_dir_raw)
            cases = load_jsonl_objects(run_dir / "cases.jsonl", missing_ok=True)
            events = load_jsonl_objects(run_dir / "events.jsonl", missing_ok=True)
            run_path = run_dir / "run.json"
            metrics_path = run_dir / "metrics.json"
            run = load_json_object(run_path) if run_path.exists() else {}
            metrics = load_json_object(metrics_path) if metrics_path.exists() else {}
            rows = [_legacy_analysis_row(case) for case in cases]
            report = _analysis_report(run=run, metrics=metrics, events=events)
            records.append(
                {
                    "results_path": document["_results_path"],
                    "plan_name": document.get("plan_name"),
                    "run_id": result.get("id"),
                    "run_dir": str(run_dir),
                    "dataset_name": _ref_name(result.get("dataset")),
                    "attack_name": _ref_name(result.get("attack")),
                    "detector_name": _ref_name(result.get("detector")),
                    "resource_tier": result.get("resource_tier"),
                    "model_id": _ref_model_id(result.get("model")),
                    "lofo": _lofo_ref(result.get("lofo")),
                    "rows": rows,
                    "report": report,
                }
            )
    return records


def _legacy_analysis_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **case,
        "sample_id": case.get("case_id"),
        "baseline": case.get("reference"),
        "with_detector": case.get("intervention"),
    }


def _analysis_report(
    *,
    run: dict[str, Any],
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    calibration = run.get("calibration_artifacts")
    calibration = calibration if isinstance(calibration, dict) else {}
    return {
        "overview": {"counts": metrics.get("counts", {})},
        "cost": {
            **(metrics.get("cost") if isinstance(metrics.get("cost"), dict) else {}),
            "provider_invocations": _event_invocations(events),
        },
        "calibration_artifacts": {
            "baseline": calibration.get("reference"),
            "with_detector": calibration.get("intervention"),
        },
        "reproduction": run.get("reproduction", {}),
    }


def _event_invocations(events: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pass_name in ("reference", "intervention"):
        providers: dict[tuple[str, str], dict[str, Any]] = {}
        for event in events:
            if event.get("pass") != pass_name or event.get("kind") != "request":
                continue
            provider = str(event.get("provider") or "unknown")
            name = str(event.get("name") or "unknown")
            row = providers.setdefault(
                (provider, name),
                {
                    "kind": "runtime_request",
                    "name": provider,
                    "status": "ok",
                    "requested": [name],
                    "materialized": [name],
                    "invocation_count": 0,
                    "sample_count": 0,
                },
            )
            row["invocation_count"] += event.get("cache_hit") is False
            row["sample_count"] += 1
            if event.get("status") == "error":
                row["status"] = "error"
        result[pass_name] = {"providers": list(providers.values())}
    return result


def _ref_name(value: Any) -> str:
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name:
            return name
    if isinstance(value, str) and value:
        return value
    return "unknown"


def _ref_model_id(value: Any) -> str | None:
    if isinstance(value, dict):
        model_id = value.get("model_id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return None


def _lofo_ref(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    holdout = value.get("holdout_attack")
    if not isinstance(holdout, str) or not holdout:
        return None
    train_attacks = _string_list(value.get("train_attacks"))
    result = {"holdout_attack": holdout, "train_attacks": train_attacks}
    mode = value.get("mode")
    if isinstance(mode, str) and mode:
        result["mode"] = mode
    return result
