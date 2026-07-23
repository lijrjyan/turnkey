from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from turnkey._internal.io import load_json_object

from turnkey.matrix.schema import load_matrix_results


def _existing_results_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing = load_matrix_results(path)
    result: dict[str, dict[str, Any]] = {}
    for item in existing["results"]:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            result[item["id"]] = item
    return result


def _skipped_result(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        **_result_base(entry),
        "status": "skipped",
        "stage": "plan",
        "reason": entry.get("reason"),
        "audit_errors": [],
    }


def _not_run_result(entry: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        **_result_base(entry),
        "status": "not_run",
        "stage": "limit",
        "reason": reason,
        "audit_errors": [],
    }


def _failed_result(
    *,
    entry: dict[str, Any],
    stage: str,
    reason: str,
    started_at: str,
    finished_at: str,
    command: list[str] | None = None,
    config_path: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    result = {
        **_result_base(entry),
        "status": "failed",
        "stage": stage,
        "reason": reason,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "audit_errors": [],
    }
    if command is not None:
        result["command"] = command
    if config_path is not None:
        result["config_path"] = str(config_path)
    if run_dir is not None:
        result["run_dir"] = str(run_dir)
        result["artifacts"] = _artifact_paths(run_dir)
    return result


def _result_base(entry: dict[str, Any]) -> dict[str, Any]:
    result = {
        "id": entry.get("id"),
        "dataset": entry.get("dataset"),
        "attack": entry.get("attack"),
        "detector": entry.get("detector"),
        "judge": entry.get("judge"),
        "model": entry.get("model"),
        "resource_tier": entry.get("resource_tier"),
    }
    if isinstance(entry.get("lofo"), dict):
        result["lofo"] = entry["lofo"]
    return result


def _result_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(result.get("status")) for result in results)
    attempted = counts.get("success", 0) + counts.get("audit_failed", 0) + counts.get("failed", 0)
    return {
        "entries": len(results),
        "attempted": attempted,
        "success": counts.get("success", 0),
        "audit_failed": counts.get("audit_failed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "not_run": counts.get("not_run", 0),
    }


def _select_merged_result(*, default: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best = deepcopy(default)
    best_priority = _merge_result_priority(best)
    for candidate in candidates:
        priority = _merge_result_priority(candidate)
        if priority > best_priority:
            best = deepcopy(candidate)
            best_priority = priority
        elif priority == best_priority and best.get("reason") == "not present in merged result sources":
            best = deepcopy(candidate)
    return best


def _merge_result_priority(result: dict[str, Any]) -> int:
    status = result.get("status")
    if status == "success":
        return 40
    if status in {"audit_failed", "failed"}:
        return 30
    if status == "skipped":
        return 20
    if status == "not_run":
        return 10
    return 0


def _artifact_paths(run_dir: Path) -> dict[str, str]:
    return {
        "run_json": str(run_dir / "run.json"),
        "cases_jsonl": str(run_dir / "cases.jsonl"),
        "events_jsonl": str(run_dir / "events.jsonl"),
        "metrics_json": str(run_dir / "metrics.json"),
    }


def _metrics_summary(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    result: dict[str, Any] = {}
    if metrics_path.exists():
        metrics = load_json_object(metrics_path)
        for key in ("ASR_strict", "RR_harm", "ORR_benign", "NSG_abs", "NSG_rel", "WBR"):
            if key in metrics:
                result[key] = metrics[key]
        if isinstance(metrics.get("counts"), dict):
            result["counts"] = metrics["counts"]
    return result


def _summary_success(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "run_dir": result.get("run_dir"),
        "metrics": result.get("metrics", {}),
    }


def _summary_failure(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "status": result.get("status"),
        "stage": result.get("stage"),
        "reason": result.get("reason"),
        "audit_errors": result.get("audit_errors", []),
    }
