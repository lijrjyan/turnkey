"""Cross-detector decision overlap (Cohen κ + set overlap + independent contribution).

Inputs are matrix `results.json` files — one per detector lane. The module joins
sample-level decisions across detectors by `(dataset, attack, sample_id)` and
emits a JSON (and optional Markdown) report. No new model runs required.

Used to support paper-track ensemble-routing claims: see issue #163.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from turnkey.analysis.matrix_analysis import (
    _detector_blocked,
    _load_results_document,
    _load_success_run_records,
)


SCHEMA_VERSION = "turnkey_detector_overlap/v1"


def build_detector_overlap(
    *,
    results_paths: list[str | Path],
    name: str | None = None,
) -> dict[str, Any]:
    documents = [_load_results_document(Path(p)) for p in results_paths]
    records = _load_success_run_records(documents)

    decisions = _decisions_by_detector(records)
    detectors = sorted(decisions)
    if len(detectors) < 2:
        raise ValueError(
            f"detector-overlap analysis needs ≥ 2 detectors with success runs; got {detectors}"
        )

    joined_keys = _intersect_keys(decisions)
    if not joined_keys:
        raise ValueError("no shared (dataset, attack, sample_id) keys across detectors")

    is_harmful = {key: decisions[detectors[0]][key]["is_harmful"] for key in joined_keys}

    per_detector_block = {
        det: {key: decisions[det][key]["blocked"] for key in joined_keys} for det in detectors
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "cohort": {
            "detectors": detectors,
            "joined_sample_count": len(joined_keys),
            "per_detector_total_sample_count": {
                det: len(decisions[det]) for det in detectors
            },
            "joined_sample_harmful_count": sum(1 for k in joined_keys if is_harmful[k]),
        },
        "kappa_matrix": _kappa_matrix(detectors, per_detector_block),
        "set_overlap": _set_overlap(detectors, per_detector_block, is_harmful),
        "independent_contribution": _independent_contribution(
            detectors, per_detector_block, is_harmful
        ),
        "ensemble": _ensemble_summary(detectors, per_detector_block, is_harmful),
    }


def write_detector_overlap(
    *,
    results_paths: list[str | Path],
    out_path: str | Path,
    markdown_path: str | Path | None = None,
    name: str | None = None,
) -> Path:
    analysis = build_detector_overlap(results_paths=results_paths, name=name)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if markdown_path is not None:
        Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_path).write_text(
            detector_overlap_markdown(analysis), encoding="utf-8"
        )
    return out


def detector_overlap_markdown(analysis: dict[str, Any]) -> str:
    cohort = analysis.get("cohort", {})
    detectors: list[str] = cohort.get("detectors", [])
    joined = cohort.get("joined_sample_count")
    harmful = cohort.get("joined_sample_harmful_count")

    lines: list[str] = ["# Detector Overlap"]
    if analysis.get("name"):
        lines.append(f"_{analysis['name']}_")
    lines.extend(
        [
            "",
            f"- Detectors: {', '.join(detectors)}",
            f"- Joined samples: {joined} ({harmful} harmful)",
            "",
            "## Cohen κ matrix",
            "",
            "| Detector | " + " | ".join(detectors) + " |",
            "|---|" + "---|" * len(detectors),
        ]
    )
    kappa_lookup = {
        (row["detector_a"], row["detector_b"]): row["cohen_kappa"]
        for row in analysis.get("kappa_matrix", [])
    }
    for det_a in detectors:
        cells: list[str] = []
        for det_b in detectors:
            if det_a == det_b:
                cells.append("1.00")
                continue
            value = kappa_lookup.get((det_a, det_b)) or kappa_lookup.get((det_b, det_a))
            cells.append("—" if value is None else f"{value:.3f}")
        lines.append(f"| {det_a} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Independent contribution (joined harmful samples only)", "",
                  "| Detector | unique blocks | unique-block rate (vs all blocks) |",
                  "|---|---:|---:|"])
    for det in detectors:
        ic = analysis.get("independent_contribution", {}).get(det, {})
        rate = ic.get("unique_block_rate_on_harmful")
        rate_s = "—" if rate is None else f"{rate:.3f}"
        lines.append(f"| {det} | {ic.get('unique_blocks_on_harmful', 0)} | {rate_s} |")

    ensemble = analysis.get("ensemble", {})
    union = ensemble.get("union_block_rate_on_harmful")
    inter = ensemble.get("intersection_block_rate_on_harmful")
    lines.extend(
        [
            "",
            "## Ensemble bound (joined harmful samples)",
            "",
            f"- union(any-detector blocks) / harmful: {_fmt(union)}",
            f"- intersection(all-detector blocks) / harmful: {_fmt(inter)}",
        ]
    )
    return "\n".join(lines) + "\n"


# --- internals ---


def _decisions_by_detector(
    records: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, str, str], dict[str, bool]]]:
    decisions: dict[str, dict[tuple[str, str, str], dict[str, bool]]] = defaultdict(dict)
    for record in records:
        det = record["detector_name"]
        dataset = record["dataset_name"]
        attack = record["attack_name"]
        for row in record.get("rows", []):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str):
                continue
            decisions[det][(dataset, attack, sample_id)] = {
                "blocked": _detector_blocked(row),
                "is_harmful": row.get("is_benign") is False,
            }
    return decisions


def _intersect_keys(
    decisions: dict[str, dict[tuple[str, str, str], dict[str, bool]]],
) -> list[tuple[str, str, str]]:
    iterator = iter(decisions.values())
    common = set(next(iterator).keys())
    for table in iterator:
        common &= table.keys()
    return sorted(common)


def _kappa_matrix(
    detectors: list[str],
    per_detector_block: dict[str, dict[tuple[str, str, str], bool]],
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    keys = next(iter(per_detector_block.values())).keys()
    for i, det_a in enumerate(detectors):
        for det_b in detectors[i + 1 :]:
            both_block = both_allow = a_only = b_only = 0
            for key in keys:
                a = per_detector_block[det_a][key]
                b = per_detector_block[det_b][key]
                if a and b:
                    both_block += 1
                elif a and not b:
                    a_only += 1
                elif b and not a:
                    b_only += 1
                else:
                    both_allow += 1
            n = both_block + both_allow + a_only + b_only
            p_o = (both_block + both_allow) / n if n else None
            p_a_block = (both_block + a_only) / n if n else 0.0
            p_b_block = (both_block + b_only) / n if n else 0.0
            p_e = p_a_block * p_b_block + (1 - p_a_block) * (1 - p_b_block)
            kappa = None if (n == 0 or p_e == 1.0) else (p_o - p_e) / (1 - p_e)
            matrix.append(
                {
                    "detector_a": det_a,
                    "detector_b": det_b,
                    "n": n,
                    "cohen_kappa": None if kappa is None else round(kappa, 6),
                    "p_observed": None if p_o is None else round(p_o, 6),
                    "p_expected": round(p_e, 6),
                    "n_both_block": both_block,
                    "n_both_allow": both_allow,
                    "n_a_only_block": a_only,
                    "n_b_only_block": b_only,
                }
            )
    return matrix


def _set_overlap(
    detectors: list[str],
    per_detector_block: dict[str, dict[tuple[str, str, str], bool]],
    is_harmful: dict[tuple[str, str, str], bool],
) -> dict[str, Any]:
    block_sets = {
        det: {key for key, blocked in table.items() if blocked}
        for det, table in per_detector_block.items()
    }
    harmful_block_sets = {
        det: {key for key in block_set if is_harmful.get(key, False)}
        for det, block_set in block_sets.items()
    }
    pairs: list[dict[str, Any]] = []
    for i, det_a in enumerate(detectors):
        for det_b in detectors[i + 1 :]:
            sa = block_sets[det_a]
            sb = block_sets[det_b]
            inter = len(sa & sb)
            union = len(sa | sb)
            pairs.append(
                {
                    "detector_a": det_a,
                    "detector_b": det_b,
                    "intersection": inter,
                    "union": union,
                    "jaccard": None if union == 0 else round(inter / union, 6),
                }
            )
    all_inter = set.intersection(*block_sets.values()) if block_sets else set()
    any_union = set.union(*block_sets.values()) if block_sets else set()
    return {
        "per_detector_block_count": {det: len(s) for det, s in block_sets.items()},
        "per_detector_block_count_on_harmful": {
            det: len(s) for det, s in harmful_block_sets.items()
        },
        "pairs": pairs,
        "all_blocks_intersection": len(all_inter),
        "any_block_union": len(any_union),
    }


def _independent_contribution(
    detectors: list[str],
    per_detector_block: dict[str, dict[tuple[str, str, str], bool]],
    is_harmful: dict[tuple[str, str, str], bool],
) -> dict[str, dict[str, Any]]:
    block_sets = {
        det: {key for key, blocked in table.items() if blocked}
        for det, table in per_detector_block.items()
    }
    harmful_keys = {key for key, h in is_harmful.items() if h}
    out: dict[str, dict[str, Any]] = {}
    for det in detectors:
        others = [block_sets[other] for other in detectors if other != det]
        union_others = set().union(*others) if others else set()
        unique = block_sets[det] - union_others
        unique_harm = unique & harmful_keys
        all_blocks = len(block_sets[det])
        out[det] = {
            "unique_blocks": len(unique),
            "unique_blocks_on_harmful": len(unique_harm),
            "unique_block_rate": None if all_blocks == 0 else round(len(unique) / all_blocks, 6),
            "unique_block_rate_on_harmful": (
                None if all_blocks == 0 else round(len(unique_harm) / all_blocks, 6)
            ),
        }
    return out


def _ensemble_summary(
    detectors: list[str],
    per_detector_block: dict[str, dict[tuple[str, str, str], bool]],
    is_harmful: dict[tuple[str, str, str], bool],
) -> dict[str, Any]:
    block_sets = {
        det: {key for key, blocked in table.items() if blocked}
        for det, table in per_detector_block.items()
    }
    harmful_keys = {key for key, h in is_harmful.items() if h}
    n_harm = len(harmful_keys)
    if n_harm == 0:
        return {"union_block_rate_on_harmful": None, "intersection_block_rate_on_harmful": None}
    union = set().union(*block_sets.values()) & harmful_keys
    inter = set.intersection(*block_sets.values()) & harmful_keys
    return {
        "union_block_rate_on_harmful": round(len(union) / n_harm, 6),
        "intersection_block_rate_on_harmful": round(len(inter) / n_harm, 6),
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"
