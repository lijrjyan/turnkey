from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from turnkey.audit import audit_run_dir
from turnkey.components.attacks import available_attacks
from turnkey.components.backends import available_backends
from turnkey.calibration import calibrate_detector_from_config
from turnkey.config import load_config
from turnkey.components.datasets import available_datasets
from turnkey.analysis.detector_overlap import write_detector_overlap
from turnkey.analysis.report import write_run_report
from turnkey.components.detectors import available_detectors
from turnkey.components.judges import available_judges
from turnkey.dev import run_method_dev
from turnkey.export import export_cases
from turnkey.inspect import INSPECT_CATEGORIES, inspect_run
from turnkey.matrix import merge_matrix_results, run_matrix_plan, summarize_matrix_results, write_matrix_plan
from turnkey.analysis.matrix_analysis import write_matrix_analysis
from turnkey.components.detectors.rcs.text_pool import build_rcs_text_pool, parse_source_spec
from turnkey.runner import run_eval
from turnkey.analysis.threshold_sweep import write_threshold_sweep
from turnkey.validation import validate_cases_jsonl


def _cmd_backends(_: argparse.Namespace) -> int:
    for name in available_backends():
        print(name)
    return 0


def _cmd_data_list(_: argparse.Namespace) -> int:
    for name in available_datasets():
        print(name)
    return 0


def _cmd_data_build_rcs_text_pool(args: argparse.Namespace) -> int:
    manifest = build_rcs_text_pool(
        benign_sources=[parse_source_spec(value, is_benign=True) for value in args.benign_source],
        malicious_sources=[parse_source_spec(value, is_benign=False) for value in args.malicious_source],
        out_jsonl=Path(args.out),
        manifest_path=Path(args.manifest),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        max_per_source=args.max_per_source,
        balance_classes=not bool(args.no_balance),
    )
    print(json.dumps({"out": str(args.out), "manifest": str(args.manifest), "counts": manifest["counts"]}))
    return 0


def _cmd_detector_list(_: argparse.Namespace) -> int:
    for name in available_detectors():
        print(name)
    return 0


def _cmd_detector_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    command = ["turnkey", "detector", "calibrate", "--config", str(args.config), "--out", str(args.out)]
    if args.report is not None:
        command.extend(["--report", str(args.report)])
    if args.holdout_family is not None:
        command.extend(["--holdout-family", str(args.holdout_family)])
    artifact_path = calibrate_detector_from_config(
        cfg,
        artifact_path=args.out,
        report_path=args.report,
        source_config_path=str(args.config),
        command=command,
        holdout_family=args.holdout_family,
    )
    print(str(artifact_path))
    return 0


def _cmd_judge_list(_: argparse.Namespace) -> int:
    for name in available_judges():
        print(name)
    return 0


def _cmd_attack_list(_: argparse.Namespace) -> int:
    for name in available_attacks():
        print(name)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if getattr(args, "private", False):
        cfg = replace(cfg, run=replace(cfg.run, unsafe_log_plaintext=True))
    if getattr(args, "input_manifest", None) is not None:
        cfg = replace(cfg, run=replace(cfg.run, input_manifest_path=str(args.input_manifest)))
    command = ["turnkey", "run", "--config", str(args.config)]
    if getattr(args, "private", False):
        command.append("--private")
    if getattr(args, "input_manifest", None) is not None:
        command.extend(["--input-manifest", str(args.input_manifest)])
    run_dir = run_eval(cfg, source_config_path=str(args.config), command=command)
    print(str(run_dir))
    return 0


def _cmd_dev(args: argparse.Namespace) -> int:
    run_dir = run_method_dev(
        args.entrypoint,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
    )
    print(str(run_dir))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    errors = validate_cases_jsonl(path)
    if errors:
        for e in errors[:50]:
            print(e, file=sys.stderr)
        print(f"FAILED: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("OK")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    result = inspect_run(
        args.run_dir,
        category=args.category,
        case_id=args.case_id,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for case in result["cases"]:
            categories = ",".join(case["categories"]) or "-"
            print(f"{case['case_id']}\t{categories}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = write_run_report(
        args.run_dir,
        json_path=args.json_path,
        markdown_path=args.markdown_path,
    )
    paths = [path for path in (args.json_path, args.markdown_path) if path is not None]
    if paths:
        for path in paths:
            print(path)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    path = export_cases(
        args.run_dir,
        out_path=args.out,
        output_format=args.format,
    )
    print(path)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    errors = audit_run_dir(run_dir)
    if errors:
        for e in errors[:50]:
            print(e, file=sys.stderr)
        print(f"FAILED: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("OK")
    return 0


def _cmd_matrix_plan(args: argparse.Namespace) -> int:
    plan_path = write_matrix_plan(spec_path=args.spec, out_dir=args.out)
    print(str(plan_path))
    return 0


def _cmd_matrix_run(args: argparse.Namespace) -> int:
    results_path = run_matrix_plan(
        plan_path=args.plan,
        out_path=args.out,
        max_runs=args.max_runs,
        resume=bool(args.resume),
        include_datasets=set(args.include_dataset or []),
        exclude_datasets=set(args.exclude_dataset or []),
        exclude_attacks=set(args.exclude_attack or []),
    )
    print(str(results_path))
    return 0


def _cmd_matrix_summarize(args: argparse.Namespace) -> int:
    summary = summarize_matrix_results(results_path=args.results, out_path=args.out)
    if args.out is not None:
        print(str(args.out))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_matrix_merge(args: argparse.Namespace) -> int:
    out_path = merge_matrix_results(plan_path=args.plan, results_paths=args.results, out_path=args.out)
    print(str(out_path))
    return 0


def _cmd_matrix_analyze(args: argparse.Namespace) -> int:
    write_matrix_analysis(
        results_paths=args.results,
        out_path=args.out,
        markdown_path=args.markdown,
        name=args.name,
    )
    print(str(args.out))
    return 0


def _cmd_analyze_threshold_sweep(args: argparse.Namespace) -> int:
    out_path = write_threshold_sweep(
        run_dir=args.run_dir,
        out_path=args.out,
        thresholds=_parse_thresholds(args.thresholds),
    )
    print(str(out_path))
    return 0


def _cmd_analyze_detector_overlap(args: argparse.Namespace) -> int:
    out_path = write_detector_overlap(
        results_paths=args.results,
        out_path=args.out,
        markdown_path=args.markdown,
        name=args.name,
    )
    print(str(out_path))
    return 0


def _parse_thresholds(value: str | None) -> list[float] | None:
    if value is None:
        return None
    thresholds: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    if not thresholds:
        raise ValueError("--thresholds must contain at least one numeric value")
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="turnkey")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_backends = sub.add_parser("backends", help="List available inference backends")
    p_backends.set_defaults(func=_cmd_backends)

    p_data = sub.add_parser("data", help="Dataset utilities")
    data_sub = p_data.add_subparsers(dest="data_cmd", required=True)
    p_data_list = data_sub.add_parser("list", help="List available datasets")
    p_data_list.set_defaults(func=_cmd_data_list)
    p_data_rcs_pool = data_sub.add_parser(
        "build-rcs-text-pool",
        help="Build an RCS text-only paper-style calibration pool JSONL and manifest",
    )
    p_data_rcs_pool.add_argument(
        "--benign-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Benign text source (.json/.jsonl/.csv). Repeat for multiple sources.",
    )
    p_data_rcs_pool.add_argument(
        "--malicious-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Malicious text source (.json/.jsonl/.csv). Repeat for multiple sources.",
    )
    p_data_rcs_pool.add_argument("--out", required=True, help="Path to write rcs_train_jsonl/v1 records")
    p_data_rcs_pool.add_argument("--manifest", required=True, help="Path to write a redacted pool manifest")
    p_data_rcs_pool.add_argument("--seed", type=int, default=0, help="Deterministic sampling/split seed")
    p_data_rcs_pool.add_argument("--val-ratio", type=float, default=0.2, help="Per-source validation ratio")
    p_data_rcs_pool.add_argument("--max-per-source", type=int, help="Optional cap per source before class balancing")
    p_data_rcs_pool.add_argument("--no-balance", action="store_true", help="Do not cap classes to equal counts")
    p_data_rcs_pool.set_defaults(func=_cmd_data_build_rcs_text_pool)

    p_detector = sub.add_parser("detector", help="Detector utilities")
    det_sub = p_detector.add_subparsers(dest="det_cmd", required=True)
    p_det_list = det_sub.add_parser("list", help="List available detectors")
    p_det_list.set_defaults(func=_cmd_detector_list)
    p_det_calibrate = det_sub.add_parser("calibrate", help="Calibrate a detector and write an artifact")
    p_det_calibrate.add_argument("--config", required=True, help="Path to YAML run config")
    p_det_calibrate.add_argument("--out", required=True, help="Path to write calibration artifact JSON")
    p_det_calibrate.add_argument("--report", help="Optional path to write calibration report JSON")
    p_det_calibrate.add_argument(
        "--holdout-family",
        help="Optional LOFO holdout family to record in the calibration artifact metadata",
    )
    p_det_calibrate.set_defaults(func=_cmd_detector_calibrate)

    p_judge = sub.add_parser("judge", help="Judge utilities")
    judge_sub = p_judge.add_subparsers(dest="judge_cmd", required=True)
    p_judge_list = judge_sub.add_parser("list", help="List available judges")
    p_judge_list.set_defaults(func=_cmd_judge_list)

    p_attack = sub.add_parser("attack", help="Attack utilities")
    attack_sub = p_attack.add_subparsers(dest="attack_cmd", required=True)
    p_attack_list = attack_sub.add_parser("list", help="List available attacks")
    p_attack_list.set_defaults(func=_cmd_attack_list)

    p_run = sub.add_parser("run", help="Run a Turnkey evaluation")
    p_run.add_argument("--config", required=True, help="Path to YAML config")
    p_run.add_argument(
        "--input-manifest",
        help="Verify selected and attacked input identity against a previous run.json",
    )
    p_run.add_argument(
        "--private",
        action="store_true",
        help="Write plaintext only to the private/ sidecar",
    )
    p_run.set_defaults(func=_cmd_run)

    p_dev = sub.add_parser("dev", help="Run a bounded smoke for an external method")
    p_dev.add_argument(
        "entrypoint",
        help="Built-in alias, module/file builder, or file Component entrypoint",
    )
    p_dev.add_argument("--out-dir", default="outputs", help="Directory for development run artifacts")
    p_dev.add_argument("--max-samples", type=int, default=4, help="Maximum fixture samples to run")
    p_dev.set_defaults(func=_cmd_dev)

    p_val = sub.add_parser("validate", help="Validate cases.jsonl")
    p_val.add_argument("path", help="Path to cases.jsonl")
    p_val.set_defaults(func=_cmd_validate)

    p_audit = sub.add_parser("audit", help="Audit a complete run directory")
    p_audit.add_argument("run_dir", help="Path to outputs/<run_id>")
    p_audit.set_defaults(func=_cmd_audit)

    p_inspect = sub.add_parser("inspect", help="Inspect paired outcomes and runtime failures")
    p_inspect.add_argument("run_dir", help="Path to outputs/<run_id>")
    p_inspect.add_argument("--category", choices=INSPECT_CATEGORIES)
    p_inspect.add_argument("--case-id")
    p_inspect.add_argument("--json", action="store_true", help="Write structured JSON to stdout")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_report = sub.add_parser("report", help="Generate optional run report views")
    p_report.add_argument("run_dir", help="Path to outputs/<run_id>")
    p_report.add_argument("--json", dest="json_path", help="Path to write report JSON")
    p_report.add_argument("--markdown", dest="markdown_path", help="Path to write report Markdown")
    p_report.set_defaults(func=_cmd_report)

    p_export = sub.add_parser("export", help="Export interoperable paired case rows")
    p_export.add_argument("run_dir", help="Path to outputs/<run_id>")
    p_export.add_argument("--format", choices=("jsonl", "csv"), required=True)
    p_export.add_argument("--out", required=True, help="Path to write exported rows")
    p_export.set_defaults(func=_cmd_export)

    p_matrix = sub.add_parser("matrix", help="Matrix validation utilities")
    matrix_sub = p_matrix.add_subparsers(dest="matrix_cmd", required=True)
    p_matrix_plan = matrix_sub.add_parser("plan", help="Generate matrix plan and run configs")
    p_matrix_plan.add_argument("--spec", required=True, help="Path to matrix YAML spec")
    p_matrix_plan.add_argument("--out", required=True, help="Directory for plan.json and generated configs")
    p_matrix_plan.set_defaults(func=_cmd_matrix_plan)
    p_matrix_run = matrix_sub.add_parser("run", help="Run matrix plan entries and audit artifacts")
    p_matrix_run.add_argument("--plan", required=True, help="Path to matrix plan.json")
    p_matrix_run.add_argument("--out", help="Path to results.json (default: next to plan.json)")
    p_matrix_run.add_argument("--max-runs", type=int, help="Maximum number of planned entries to execute")
    p_matrix_run.add_argument("--resume", action="store_true", help="Reuse existing non-not_run results in --out")
    p_matrix_run.add_argument(
        "--include-dataset",
        action="append",
        help="Only run planned entries for this dataset name. Repeat to include multiple datasets.",
    )
    p_matrix_run.add_argument(
        "--exclude-dataset",
        action="append",
        help="Skip planned entries for this dataset name. Repeat to exclude multiple datasets.",
    )
    p_matrix_run.add_argument(
        "--exclude-attack",
        action="append",
        help="Skip planned entries for an attack name. Repeat to exclude multiple attacks.",
    )
    p_matrix_run.set_defaults(func=_cmd_matrix_run)
    p_matrix_summary = matrix_sub.add_parser("summarize", help="Summarize matrix results")
    p_matrix_summary.add_argument("--results", required=True, help="Path to matrix results.json")
    p_matrix_summary.add_argument("--out", help="Optional path to write summary JSON")
    p_matrix_summary.set_defaults(func=_cmd_matrix_summarize)
    p_matrix_merge = matrix_sub.add_parser("merge", help="Merge sharded matrix results into plan order")
    p_matrix_merge.add_argument("--plan", required=True, help="Path to matrix plan.json")
    p_matrix_merge.add_argument(
        "--results",
        action="append",
        required=True,
        help="Path to matrix results.json shard. Repeat for each source to merge.",
    )
    p_matrix_merge.add_argument("--out", required=True, help="Path to write merged results.json")
    p_matrix_merge.set_defaults(func=_cmd_matrix_merge)
    p_matrix_analyze = matrix_sub.add_parser(
        "analyze",
        help="Export rich analysis tables from one or more matrix results.json files",
    )
    p_matrix_analyze.add_argument(
        "--results",
        action="append",
        required=True,
        help="Path to matrix results.json. Repeat for detector lanes that should be compared.",
    )
    p_matrix_analyze.add_argument("--out", required=True, help="Path to write analysis JSON")
    p_matrix_analyze.add_argument("--markdown", help="Optional path to write Markdown summary")
    p_matrix_analyze.add_argument("--name", help="Optional analysis name")
    p_matrix_analyze.set_defaults(func=_cmd_matrix_analyze)

    p_analyze = sub.add_parser("analyze", help="Analysis artifact utilities")
    analyze_sub = p_analyze.add_subparsers(dest="analyze_cmd", required=True)
    p_threshold_sweep = analyze_sub.add_parser(
        "threshold-sweep",
        help="Write an artifact-aware detector threshold sweep for a completed run",
    )
    p_threshold_sweep.add_argument("--run-dir", required=True, help="Path to outputs/<run_id>")
    p_threshold_sweep.add_argument("--out", required=True, help="Path to write threshold sweep JSON")
    p_threshold_sweep.add_argument(
        "--thresholds",
        help="Optional comma-separated thresholds. Defaults to score-derived thresholds plus the artifact threshold.",
    )
    p_threshold_sweep.set_defaults(func=_cmd_analyze_threshold_sweep)

    p_overlap = analyze_sub.add_parser(
        "detector-overlap",
        help="Cross-detector decision overlap (Cohen κ + set overlap + independent contribution)",
    )
    p_overlap.add_argument(
        "--results",
        action="append",
        required=True,
        help="Path to a matrix results.json. Repeat once per detector lane to compare.",
    )
    p_overlap.add_argument("--out", required=True, help="Path to write overlap analysis JSON")
    p_overlap.add_argument("--markdown", help="Optional path to write Markdown summary")
    p_overlap.add_argument("--name", help="Optional analysis name")
    p_overlap.set_defaults(func=_cmd_analyze_detector_overlap)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
