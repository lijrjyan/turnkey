from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from turnkey._internal.io import load_jsonl_objects
from turnkey.inspect import classify_case


ExportFormat = Literal["jsonl", "csv"]


def export_cases(
    run_dir: str | Path,
    *,
    out_path: str | Path,
    output_format: ExportFormat,
) -> Path:
    if output_format not in {"jsonl", "csv"}:
        raise ValueError("output_format must be 'jsonl' or 'csv'")
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    cases = load_jsonl_objects(run_dir / "cases.jsonl")
    events = load_jsonl_objects(run_dir / "events.jsonl")
    events_by_case: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        case_id = event.get("case_id")
        if isinstance(case_id, str):
            events_by_case.setdefault(case_id, []).append(event)

    rows = [
        _export_row(case, events=events_by_case.get(str(case.get("case_id")), []))
        for case in cases
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        out_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    else:
        fieldnames = [
            "case_id",
            "behavior_id",
            "is_benign",
            "attack_family",
            "categories",
            "reference_block",
            "reference_harmful",
            "reference_refusal",
            "intervention_block",
            "intervention_harmful",
            "intervention_refusal",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **{key: row.get(key) for key in fieldnames},
                        "categories": ",".join(row["categories"]),
                        "reference_block": row["reference"]["block"],
                        "reference_harmful": row["reference"]["harmful"],
                        "reference_refusal": row["reference"]["refusal"],
                        "intervention_block": row["intervention"]["block"],
                        "intervention_harmful": row["intervention"]["harmful"],
                        "intervention_refusal": row["intervention"]["refusal"],
                    }
                )
    return out_path


def _export_row(case: dict[str, Any], *, events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "behavior_id": case.get("behavior_id"),
        "is_benign": case.get("is_benign"),
        "attack_family": case.get("attack_family"),
        "categories": classify_case(case, events=events),
        "reference": _outcome_summary(case.get("reference")),
        "intervention": _outcome_summary(case.get("intervention")),
    }


def _outcome_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"block": None, "harmful": None, "refusal": None}
    detector = value.get("detector")
    judge = value.get("judge")
    return {
        "block": detector.get("block") if isinstance(detector, dict) else None,
        "harmful": judge.get("is_harmful_effective") if isinstance(judge, dict) else None,
        "refusal": judge.get("is_refusal") if isinstance(judge, dict) else None,
    }
