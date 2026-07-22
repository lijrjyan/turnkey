from __future__ import annotations

from pathlib import Path
from typing import Any

from turnkey.validation import validate_cases_jsonl

from ._helpers import _load_json_object, _load_jsonl_rows
from .events import check_events
from .metrics import _check_metrics
from .run_bundle import check_run


REQUIRED_FILES = {
    "run": Path("run.json"),
    "cases": Path("cases.jsonl"),
    "events": Path("events.jsonl"),
    "metrics": Path("metrics.json"),
}


def audit_run_dir(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    paths = {name: run_dir / rel for name, rel in REQUIRED_FILES.items()}
    errors: list[str] = []

    if not run_dir.exists():
        errors.append(f"{run_dir}: run directory does not exist")
        return errors
    if not run_dir.is_dir():
        errors.append(f"{run_dir}: run path must be a directory")
        return errors

    for name, rel_path in REQUIRED_FILES.items():
        if not paths[name].exists():
            errors.append(f"{run_dir}: missing {rel_path.as_posix()}")

    cases: list[dict[str, Any]] = []
    if paths["cases"].exists():
        errors.extend(validate_cases_jsonl(paths["cases"]))
        cases = _load_jsonl_rows(paths["cases"], errors=errors)

    events = _load_jsonl_rows(paths["events"], errors=errors) if paths["events"].exists() else []

    metrics = (
        _load_json_object(paths["metrics"], errors=errors) if paths["metrics"].exists() else None
    )
    run = _load_json_object(paths["run"], errors=errors) if paths["run"].exists() else None

    if metrics is not None:
        _check_metrics(
            path=paths["metrics"],
            metrics=metrics,
            rows=cases,
            errors=errors,
        )
    check_events(
        path=paths["events"],
        events=events,
        cases=cases,
        metrics=metrics,
        errors=errors,
    )
    if run is not None:
        check_run(
            path=paths["run"],
            run=run,
            run_dir=run_dir,
            cases=cases,
            events=events,
            errors=errors,
        )

    return errors


__all__ = ["audit_run_dir"]
