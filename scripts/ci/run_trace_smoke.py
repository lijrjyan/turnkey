from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


def validation_commands(run_dir: Path) -> list[list[str]]:
    cases = run_dir / "cases.jsonl"
    return [
        ["turnkey", "validate", str(cases)],
        ["turnkey", "audit", str(run_dir)],
    ]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _materialize_config(
    config_path: Path,
    *,
    out_root: Path,
    model_base_url: str | None = None,
) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw.setdefault("run", {})["out_dir"] = str(out_root / config_path.stem / "runs")
    if model_base_url is not None:
        raw.setdefault("model", {})["base_url"] = model_base_url.rstrip("/")

    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ci_config_path = cfg_dir / config_path.name
    ci_config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return ci_config_path


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="")
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _run_turnkey(config_path: Path) -> Path:
    result = _run_command(["turnkey", "run", "--config", str(config_path)])
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("turnkey run did not print a run directory")
    run_dir = Path(lines[-1])
    if not run_dir.exists():
        raise RuntimeError(f"turnkey run directory does not exist: {run_dir}")
    return run_dir


def validate_run(run_dir: Path) -> None:
    for command in validation_commands(run_dir):
        _run_command(command)


def assert_expected_contents(config_path: Path, run_dir: Path) -> None:
    rows = _load_rows(run_dir / "cases.jsonl")
    if not rows:
        raise AssertionError(f"{run_dir}: cases.jsonl is empty")

    config_name = config_path.name
    events = _load_rows(run_dir / "events.jsonl")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("schema_version") != "turnkey_run/v1":
        raise AssertionError(f"{config_name}: unexpected run schema")
    if not events or any(event.get("schema_version") != "turnkey_event/v1" for event in events):
        raise AssertionError(f"{config_name}: invalid or empty runtime events")
    case_ids = {row.get("case_id") for row in rows}
    if any(event.get("case_id") not in case_ids for event in events):
        raise AssertionError(f"{config_name}: event references an unknown case")

    if config_name in {"smoke.yaml", "smoke_pair.yaml"}:
        datasets = {
            str(row.get("dataset", {}).get("name"))
            for row in rows
            if isinstance(row.get("dataset"), dict)
        }
        if datasets != {"fixtures_smoke"}:
            raise AssertionError(f"{config_name}: expected fixtures_smoke rows, got {sorted(datasets)}")

    if config_name == "smoke_pair.yaml":
        attack_methods = {row.get("threat", {}).get("attack_method") for row in rows}
        if attack_methods != {"pair"}:
            raise AssertionError(f"{config_name}: expected attack_method=pair, got {sorted(attack_methods)}")
        tiers = {row.get("threat", {}).get("tier") for row in rows}
        if tiers != {"T2"}:
            raise AssertionError(f"{config_name}: expected threat.tier=T2, got {sorted(tiers)}")

    baseline_not_executed = [
        row.get("case_id", "<unknown>")
        for row in rows
        if row.get("reference", {}).get("model", {}).get("executed") is not True
    ]
    if baseline_not_executed:
        raise AssertionError(f"{config_name}: baseline model did not execute: {baseline_not_executed}")

    blocked_executed = [
        row.get("case_id", "<unknown>")
        for row in rows
        if row.get("intervention", {}).get("detector", {}).get("block") is True
        and row.get("intervention", {}).get("model", {}).get("executed") is not False
    ]
    if blocked_executed:
        raise AssertionError(
            f"{config_name}: blocked with-detector rows executed model: {blocked_executed}"
        )

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("ASR_strict", "ORR_benign", "NSG_abs", "NSG_rel"):
            if key not in metrics:
                raise AssertionError(f"{config_name}: metrics missing {key}")


def run_config(config_path: Path, *, out_root: Path, model_base_url: str | None = None) -> Path:
    ci_config_path = _materialize_config(
        config_path,
        out_root=out_root,
        model_base_url=model_base_url,
    )
    run_dir = _run_turnkey(ci_config_path)
    validate_run(run_dir)
    assert_expected_contents(config_path, run_dir)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CI trace smoke checks")
    parser.add_argument("--config", action="append", required=True, help="Run config to execute")
    parser.add_argument("--out-root", default="outputs/ci-trace-smoke")
    parser.add_argument("--model-base-url", help="Override model.base_url in generated CI config")
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for config in args.config:
        run_dir = run_config(
            Path(config),
            out_root=out_root,
            model_base_url=args.model_base_url,
        )
        print(f"OK {config}: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
