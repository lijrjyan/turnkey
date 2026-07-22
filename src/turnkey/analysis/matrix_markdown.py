from __future__ import annotations

from typing import Any

from turnkey._internal.data import string_list as _string_list


def matrix_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        f"# Matrix Analysis: {analysis.get('name') or 'unnamed'}",
        "",
        f"Schema: `{analysis.get('schema_version')}`",
        "",
    ]
    cohort = analysis.get("cohort") if isinstance(analysis.get("cohort"), dict) else {}
    lines.extend(
        [
            "## Cohort",
            "",
            f"- Detectors: {', '.join(_string_list(cohort.get('detectors')))}",
            f"- Common entries: {cohort.get('common_entry_count', 0)}",
            f"- Union entries: {cohort.get('union_entry_count', 0)}",
            "",
        ]
    )
    missing = cohort.get("missing_by_detector")
    if isinstance(missing, dict) and any(missing.values()):
        lines.append("Missing entries are explicit in the JSON `cohort.missing_by_detector` field.")
        lines.append("")

    lines.extend(
        [
            "## Detector Comparison",
            "",
            "| Detector | Entries | Samples | Harm block | Benign block | NSG_abs | NSG_rel | WBR | Net harm-benign | Extra forwards | Detector inv/sample |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis.get("comparison_rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {detector} | {entries} | {samples} | {harm_block} | {benign_block} | "
            "{nsg_abs} | {nsg_rel} | {wbr} | {net} | {extra_forwards} | {cost} |".format(
                detector=row.get("detector"),
                entries=row.get("entry_count"),
                samples=row.get("n_samples"),
                harm_block=_pct(row.get("harm_block_rate")),
                benign_block=_pct(row.get("benign_block_rate")),
                nsg_abs=_num(row.get("NSG_abs")),
                nsg_rel=_num(row.get("NSG_rel")),
                wbr=_num(row.get("WBR")),
                net=_pp(row.get("net_harm_block_minus_benign_block")),
                extra_forwards=_num(row.get("extra_forwards_avg")),
                cost=_num(row.get("detector_provider_invocations_per_sample")),
            )
        )

    lines.extend(["", "## Naturalness Coverage", ""])
    lines.append("| Detector | Known bucket samples | Unknown bucket samples | Prompt length buckets |")
    lines.append("| --- | ---: | ---: | --- |")
    detectors = analysis.get("detectors") if isinstance(analysis.get("detectors"), dict) else {}
    for detector, value in sorted(detectors.items()):
        if not isinstance(value, dict):
            continue
        naturalness = value.get("by_naturalness_bucket") if isinstance(value.get("by_naturalness_bucket"), dict) else {}
        unknown = _group_sample_count(naturalness.get("unknown"))
        known = sum(_group_sample_count(group) for name, group in naturalness.items() if name != "unknown")
        length_buckets = value.get("by_prompt_length_bucket")
        length_counts = {}
        if isinstance(length_buckets, dict):
            length_counts = {key: _group_sample_count(group) for key, group in sorted(length_buckets.items())}
        lines.append(
            f"| {detector} | {known} | {unknown} | "
            f"{', '.join(f'{key}:{count}' for key, count in length_counts.items())} |"
        )

    lines.extend(["", "## Ranking Comparisons", ""])
    lines.append("| Traditional metric | Target metric | Detectors | Kendall tau | Spearman rho |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in analysis.get("ranking_comparisons", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {metric} | {target} | {count} | {tau} | {rho} |".format(
                metric=row.get("traditional_metric"),
                target=row.get("target_metric"),
                count=row.get("detector_count"),
                tau=_num(row.get("kendall_tau")),
                rho=_num(row.get("spearman_rho")),
            )
        )

    lofo_rows = analysis.get("lofo_generalization")
    if isinstance(lofo_rows, list) and lofo_rows:
        lines.extend(["", "## LOFO Generalization", ""])
        lines.append("| Detector | Holdout attack | Entries | Samples | NSG_abs | Benign block | Train attacks |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for row in lofo_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {detector} | {holdout} | {entries} | {samples} | {nsg} | {benign} | {train} |".format(
                    detector=row.get("detector"),
                    holdout=row.get("holdout_attack"),
                    entries=row.get("entry_count"),
                    samples=row.get("n_samples"),
                    nsg=_num(row.get("NSG_abs")),
                    benign=_pct(row.get("benign_block_rate")),
                    train=", ".join(_string_list(row.get("train_attacks"))),
                )
            )

    lines.extend(["", "## Notes", ""])
    lines.append("- Metrics are aggregated from raw `cases.jsonl` rows for completed runs.")
    lines.append("- `NSG_abs` and `NSG_rel` only credit detector blocks that stop baseline-successful harmful outputs.")
    lines.append("- `AUC`, `F1`, and `ASR_reduction` are traditional detector metrics computed on the same rows.")
    lines.append("- `Net harm-benign` is `harm_block_rate - benign_block_rate`, in percentage points.")
    lines.append("- Naturalness buckets are reported when sample metadata provides them; otherwise coverage is explicit as `unknown`.")
    return "\n".join(lines) + "\n"


def _group_sample_count(group: Any) -> int:
    if not isinstance(group, dict):
        return 0
    counts = group.get("counts")
    if not isinstance(counts, dict):
        return 0
    value = counts.get("n_samples")
    return value if isinstance(value, int) else 0


def _num(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"{float(value):.3f}"


def _pct(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"{float(value) * 100:.1f}%"


def _pp(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"{float(value) * 100:.1f} pp"
