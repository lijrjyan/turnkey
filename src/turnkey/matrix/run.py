from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turnkey._internal.io import write_json_object
from turnkey.analysis.threshold_sweep import write_threshold_sweep
from turnkey.audit import audit_run_dir
from turnkey.config import load_config
from turnkey.matrix.results import (
    _artifact_paths,
    _existing_results_by_id,
    _failed_result,
    _metrics_summary,
    _not_run_result,
    _result_base,
    _result_counts,
    _skipped_result,
)
from turnkey.matrix.schema import MATRIX_RESULTS_SCHEMA, load_matrix_plan
from turnkey.runner import RunResourceCache, run_eval

MATRIX_THRESHOLD_GRID = (
    -10.0,
    -5.0,
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    2.0,
    5.0,
    10.0,
)


def run_matrix_plan(
    *,
    plan_path: str | Path,
    out_path: str | Path | None = None,
    max_runs: int | None = None,
    resume: bool = False,
    include_datasets: set[str] | None = None,
    exclude_datasets: set[str] | None = None,
    exclude_attacks: set[str] | None = None,
) -> Path:
    plan_path = Path(plan_path)
    plan = load_matrix_plan(plan_path)
    if max_runs is not None and max_runs < 0:
        raise ValueError("max_runs must be non-negative")
    included_datasets = set(include_datasets or set())
    excluded_datasets = set(exclude_datasets or set())
    excluded_attacks = set(exclude_attacks or set())

    if out_path is None:
        out_path = plan_path.parent / "results.json"
    out_path = Path(out_path)
    existing_by_id = _existing_results_by_id(out_path) if resume else {}
    runtime_cache = RunResourceCache()

    results: list[dict[str, Any]] = []
    attempted = 0
    for raw_entry in plan["entries"]:
        if not isinstance(raw_entry, dict):
            continue
        entry_id = raw_entry.get("id")
        if not isinstance(entry_id, str):
            continue
        existing = existing_by_id.get(entry_id)
        if existing is not None and existing.get("status") != "not_run":
            results.append(existing)
            if existing.get("status") in {"success", "audit_failed", "failed"}:
                attempted += 1
            continue

        if raw_entry.get("status") == "skipped":
            results.append(_skipped_result(raw_entry))
            continue
        dataset_name = _entry_dataset_name(raw_entry)
        if included_datasets and dataset_name not in included_datasets:
            results.append(_not_run_result(raw_entry, reason=f"excluded dataset={dataset_name}"))
            continue
        if dataset_name in excluded_datasets:
            results.append(_not_run_result(raw_entry, reason=f"excluded dataset={dataset_name}"))
            continue
        attack_name = _entry_attack_name(raw_entry)
        if attack_name in excluded_attacks:
            results.append(_not_run_result(raw_entry, reason=f"excluded attack={attack_name}"))
            continue
        if max_runs is not None and attempted >= max_runs:
            results.append(_not_run_result(raw_entry, reason=f"max_runs={max_runs} reached"))
            continue

        attempted += 1
        result = _run_matrix_entry(plan_path=plan_path, entry=raw_entry, runtime_cache=runtime_cache)
        results.append(result)
        _write_results_checkpoint(
            out_path=out_path,
            plan_path=plan_path,
            plan=plan,
            results=results,
            runtime_cache=runtime_cache,
            included_datasets=included_datasets,
            excluded_datasets=excluded_datasets,
            excluded_attacks=excluded_attacks,
        )

    _write_results_checkpoint(
        out_path=out_path,
        plan_path=plan_path,
        plan=plan,
        results=results,
        runtime_cache=runtime_cache,
        included_datasets=included_datasets,
        excluded_datasets=excluded_datasets,
        excluded_attacks=excluded_attacks,
    )
    return out_path


def _write_results_checkpoint(
    *,
    out_path: Path,
    plan_path: Path,
    plan: dict[str, Any],
    results: list[dict[str, Any]],
    runtime_cache: RunResourceCache,
    included_datasets: set[str],
    excluded_datasets: set[str],
    excluded_attacks: set[str],
) -> None:
    output = {
        "schema_version": MATRIX_RESULTS_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "plan_name": plan.get("name"),
        "counts": _result_counts(results),
        "filters": {
            "include_datasets": sorted(included_datasets),
            "exclude_datasets": sorted(excluded_datasets),
            "exclude_attacks": sorted(excluded_attacks),
        },
        "runtime_cache": runtime_cache.summary(),
        "runtime_cache_manifest": runtime_cache.manifest(),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
    write_json_object(tmp_path, output)
    tmp_path.replace(out_path)


def _run_matrix_entry(*, plan_path: Path, entry: dict[str, Any], runtime_cache: RunResourceCache) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    config_ref = entry.get("config_path")
    if not isinstance(config_ref, str):
        return _failed_result(
            entry=entry,
            stage="plan",
            reason="planned entry missing config_path",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    config_path = plan_path.parent / config_ref
    command = ["turnkey", "run", "--config", str(config_path)]
    try:
        cfg = load_config(config_path)
        run_dir = run_eval(
            cfg,
            source_config_path=str(config_path),
            command=command,
            runtime_cache=runtime_cache,
        )
    except Exception as exc:  # noqa: BLE001
        return _failed_result(
            entry=entry,
            stage="run",
            reason=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            command=command,
            config_path=config_path,
        )

    try:
        _write_matrix_threshold_sweep(run_dir)
    except Exception as exc:  # noqa: BLE001
        return _failed_result(
            entry=entry,
            stage="threshold_sweep",
            reason=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            command=command,
            config_path=config_path,
            run_dir=run_dir,
        )

    audit_errors = audit_run_dir(run_dir)
    finished_at = datetime.now(timezone.utc).isoformat()
    if audit_errors:
        return {
            **_result_base(entry),
            "status": "audit_failed",
            "stage": "audit",
            "reason": f"turnkey audit failed with {len(audit_errors)} error(s)",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "command": command,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "audit_errors": audit_errors,
            "artifacts": _artifact_paths(run_dir),
            "metrics": _metrics_summary(run_dir),
        }

    return {
        **_result_base(entry),
        "status": "success",
        "stage": "audit",
        "reason": None,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "command": command,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "audit_errors": [],
        "artifacts": _artifact_paths(run_dir),
        "metrics": _metrics_summary(run_dir),
    }


def _entry_attack_name(entry: dict[str, Any]) -> str | None:
    attack = entry.get("attack")
    if not isinstance(attack, dict):
        return None
    name = attack.get("name")
    return name if isinstance(name, str) else None


def _entry_dataset_name(entry: dict[str, Any]) -> str | None:
    dataset = entry.get("dataset")
    if not isinstance(dataset, dict):
        return None
    name = dataset.get("name")
    return name if isinstance(name, str) else None


def _write_matrix_threshold_sweep(run_dir: Path) -> Path:
    return write_threshold_sweep(
        run_dir=run_dir,
        out_path=run_dir / "reports" / "threshold_sweep.json",
        thresholds=list(MATRIX_THRESHOLD_GRID),
    )
