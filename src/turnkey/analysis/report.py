from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from turnkey._internal.io import load_json_object, load_jsonl_objects
from turnkey.inspect import INSPECT_CATEGORIES, classify_case


def write_run_report(
    run_dir: str | Path,
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run = load_json_object(run_dir / "run.json", root_error="run.json root must be object")
    metrics = load_json_object(
        run_dir / "metrics.json",
        root_error="metrics.json root must be object",
    )
    cases = load_jsonl_objects(run_dir / "cases.jsonl")
    events = load_jsonl_objects(run_dir / "events.jsonl")
    events_by_case: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        case_id = event.get("case_id")
        if isinstance(case_id, str):
            events_by_case.setdefault(case_id, []).append(event)

    category_counts: Counter[str] = Counter()
    for case in cases:
        for category in classify_case(
            case,
            events=events_by_case.get(str(case.get("case_id")), []),
        ):
            category_counts[category] += 1
    report = {
        "schema_version": "turnkey_report/v1",
        "run": {
            "run_id": run.get("run_id"),
            "generated_at_utc": run.get("generated_at_utc"),
            "model": run.get("model"),
            "components": run.get("components"),
        },
        "metrics": metrics,
        "failure_categories": {
            category: category_counts.get(category, 0)
            for category in INSPECT_CATEGORIES
        },
        "events": {
            "count": len(events),
            "errors": sum(event.get("status") == "error" for event in events),
            "model_forwards": sum(
                event.get("model_forwards", 0)
                for event in events
                if isinstance(event.get("model_forwards"), int)
            ),
        },
    }
    if json_path is not None:
        _write_derived_json(Path(json_path), report)
    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_run_report_markdown(report), encoding="utf-8")
    return report


def _write_derived_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_report_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics")
    categories = report.get("failure_categories")
    events = report.get("events")
    return "\n".join(
        [
            "# Turnkey Run Report",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Failure Categories",
            "",
            "```json",
            json.dumps(categories, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Runtime Events",
            "",
            "```json",
            json.dumps(events, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
