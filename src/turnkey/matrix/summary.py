from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turnkey._internal.data import string_list as _string_list
from turnkey._internal.io import write_json_object
from turnkey.matrix.results import (
    _not_run_result,
    _result_counts,
    _select_merged_result,
    _skipped_result,
    _summary_failure,
    _summary_success,
)
from turnkey.matrix.schema import MATRIX_RESULTS_SCHEMA, MATRIX_SUMMARY_SCHEMA, load_matrix_plan, load_matrix_results

def summarize_matrix_results(*, results_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    results_path = Path(results_path)
    results_doc = load_matrix_results(results_path)
    results = [result for result in results_doc["results"] if isinstance(result, dict)]
    summary = {
        "schema_version": MATRIX_SUMMARY_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_path": str(results_path),
        "plan_name": results_doc.get("plan_name"),
        "counts": _result_counts(results),
        "by_status": dict(sorted(Counter(str(result.get("status")) for result in results).items())),
        "by_stage": dict(sorted(Counter(str(result.get("stage")) for result in results).items())),
        "successes": [_summary_success(result) for result in results if result.get("status") == "success"],
        "failures": [
            _summary_failure(result)
            for result in results
            if result.get("status") in {"audit_failed", "failed"}
        ],
        "skipped": [
            _summary_failure(result)
            for result in results
            if result.get("status") in {"skipped", "not_run"}
        ],
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_object(out_path, summary)
    return summary


def merge_matrix_results(
    *,
    plan_path: str | Path,
    results_paths: list[str | Path],
    out_path: str | Path,
) -> Path:
    plan_path = Path(plan_path)
    plan = load_matrix_plan(plan_path)
    out_path = Path(out_path)
    if not results_paths:
        raise ValueError("at least one --results path is required")

    candidates_by_id: dict[str, list[dict[str, Any]]] = {}
    merge_sources: list[dict[str, Any]] = []
    include_datasets: set[str] = set()
    exclude_datasets: set[str] = set()
    exclude_attacks: set[str] = set()

    for raw_path in results_paths:
        results_path = Path(raw_path)
        doc = load_matrix_results(results_path)
        source_results = [result for result in doc["results"] if isinstance(result, dict)]
        filters = doc.get("filters")
        if isinstance(filters, dict):
            include_datasets.update(_string_list(filters.get("include_datasets")))
            exclude_datasets.update(_string_list(filters.get("exclude_datasets")))
            exclude_attacks.update(_string_list(filters.get("exclude_attacks")))
        merge_sources.append(
            {
                "path": str(results_path),
                "plan_name": doc.get("plan_name"),
                "counts": _result_counts(source_results),
                "filters": filters if isinstance(filters, dict) else {},
            }
        )
        for result in source_results:
            entry_id = result.get("id")
            if isinstance(entry_id, str):
                candidates_by_id.setdefault(entry_id, []).append(result)

    results: list[dict[str, Any]] = []
    for raw_entry in plan["entries"]:
        if not isinstance(raw_entry, dict):
            continue
        entry_id = raw_entry.get("id")
        if not isinstance(entry_id, str):
            continue
        default = (
            _skipped_result(raw_entry)
            if raw_entry.get("status") == "skipped"
            else _not_run_result(raw_entry, reason="not present in merged result sources")
        )
        results.append(_select_merged_result(default=default, candidates=candidates_by_id.get(entry_id, [])))

    output = {
        "schema_version": MATRIX_RESULTS_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "plan_name": plan.get("name"),
        "counts": _result_counts(results),
        "filters": {
            "include_datasets": sorted(include_datasets),
            "exclude_datasets": sorted(exclude_datasets),
            "exclude_attacks": sorted(exclude_attacks),
        },
        "merge_sources": merge_sources,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_object(out_path, output)
    return out_path
