from __future__ import annotations

from collections import Counter
from html import escape
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
    html_path: str | Path | None = None,
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
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            _run_report_html(report, cases=cases, events_by_case=events_by_case),
            encoding="utf-8",
        )
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


def _run_report_html(
    report: dict[str, Any],
    *,
    cases: list[dict[str, Any]],
    events_by_case: dict[str, list[dict[str, Any]]],
) -> str:
    run = report.get("run") if isinstance(report.get("run"), dict) else {}
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    categories = (
        report.get("failure_categories")
        if isinstance(report.get("failure_categories"), dict)
        else {}
    )
    event_summary = report.get("events") if isinstance(report.get("events"), dict) else {}
    metric_cards = "".join(
        _metric_card(label, metrics.get(key))
        for key, label in (
            ("NSG_abs", "NSG"),
            ("ASR_reduction", "ASR reduction"),
            ("F1", "F1"),
            ("AUC", "AUC"),
            ("WBR", "WBR"),
        )
    )
    category_rows = "".join(
        f"<tr><th>{escape(str(category))}</th><td>{escape(str(count))}</td></tr>"
        for category, count in categories.items()
    )
    case_details = "".join(
        _case_html(case, events=events_by_case.get(str(case.get("case_id")), []))
        for case in cases
    )
    run_id = escape(str(run.get("run_id") or "unknown"))
    generated_at = escape(str(run.get("generated_at_utc") or "unknown"))
    event_count = escape(str(event_summary.get("count", 0)))
    event_errors = escape(str(event_summary.get("errors", 0)))
    model_forwards = escape(str(event_summary.get("model_forwards", 0)))
    metrics_json = escape(json.dumps(metrics, indent=2, ensure_ascii=False))
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Turnkey Run Report — {run_id}</title>
  <style>
    :root {{ color-scheme: light dark; --accent: #f2a93b; --line: #8885; }}
    body {{ font: 15px/1.55 system-ui, sans-serif; max-width: 1100px; margin: auto; padding: 2rem; }}
    h1, h2 {{ line-height: 1.2; }}
    .muted {{ opacity: .72; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .75rem; }}
    .card, details {{ border: 1px solid var(--line); border-radius: .7rem; padding: .8rem 1rem; }}
    .value {{ color: var(--accent); font-size: 1.45rem; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: .45rem; text-align: left; vertical-align: top; }}
    details {{ margin: .65rem 0; }}
    summary {{ cursor: pointer; font-weight: 650; }}
    pre {{ overflow: auto; border: 1px solid var(--line); border-radius: .5rem; padding: .8rem; }}
    .ok {{ color: #39a96b; }} .error {{ color: #dc5a5a; }}
    @media (max-width: 640px) {{ body {{ padding: 1rem; }} .events {{ display: block; overflow-x: auto; }} }}
  </style>
</head>
<body>
  <header>
    <p class="muted">Redacted, regenerable view of public run artifacts</p>
    <h1>Turnkey Run Report</h1>
    <p><strong>{run_id}</strong> · {generated_at}</p>
  </header>
  <main>
    <h2>Metrics</h2>
    <div class="grid">{metric_cards}</div>
    <h2>Failure categories</h2>
    <table><tbody>{category_rows}</tbody></table>
    <h2>Runtime events</h2>
    <div class="grid">
      {_metric_card("Events", event_count)}
      {_metric_card("Errors", event_errors)}
      {_metric_card("Model forwards", model_forwards)}
    </div>
    <h2>Cases and Runtime event timeline</h2>
    {case_details or '<p class="muted">No cases were recorded.</p>'}
    <details><summary>Complete metrics JSON</summary><pre>{metrics_json}</pre></details>
  </main>
</body>
</html>
'''


def _metric_card(label: str, value: Any) -> str:
    rendered = "—" if value is None else str(value)
    return (
        '<div class="card">'
        f'<div class="muted">{escape(label)}</div>'
        f'<div class="value">{escape(rendered)}</div>'
        "</div>"
    )


def _case_html(case: dict[str, Any], *, events: list[dict[str, Any]]) -> str:
    case_id = str(case.get("case_id") or "unknown")
    categories = classify_case(case, events=events)
    category_label = ", ".join(categories) or "uncategorized"
    sample_kind = "benign" if case.get("is_benign") is True else "harmful"
    reference = _decision_label(case.get("reference"))
    intervention = _decision_label(case.get("intervention"))
    rows = "".join(_event_html(event) for event in sorted(events, key=_event_sequence))
    return f'''<details>
  <summary>{escape(case_id)} · {escape(sample_kind)} · {escape(category_label)}</summary>
  <p><strong>Reference:</strong> {escape(reference)}<br>
     <strong>Intervention:</strong> {escape(intervention)}</p>
  <div class="events"><table>
    <thead><tr><th>Pass</th><th>Kind</th><th>Name</th><th>Status</th><th>Duration</th><th>Forwards</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</details>'''


def _decision_label(outcome: Any) -> str:
    if not isinstance(outcome, dict):
        return "unavailable"
    detector = outcome.get("detector")
    if not isinstance(detector, dict):
        return "unavailable"
    action = "block" if detector.get("block") is True else "allow"
    reason = detector.get("reason")
    score = detector.get("score")
    parts = [action]
    if reason is not None:
        parts.append(f"reason={reason}")
    if score is not None:
        parts.append(f"score={score}")
    return " · ".join(parts)


def _event_sequence(event: dict[str, Any]) -> int:
    sequence = event.get("sequence")
    return sequence if isinstance(sequence, int) else 0


def _event_html(event: dict[str, Any]) -> str:
    status = str(event.get("status") or "unknown")
    duration = event.get("duration_s")
    duration_ms = f"{float(duration) * 1000:.2f} ms" if isinstance(duration, (int, float)) else "—"
    values = (
        str(event.get("pass") or "—"),
        str(event.get("kind") or "—"),
        str(event.get("name") or "—"),
        status,
        duration_ms,
        str(event.get("model_forwards", 0)),
    )
    cells = "".join(f"<td>{escape(value)}</td>" for value in values)
    return f'<tr class="{escape(status)}">{cells}</tr>'
