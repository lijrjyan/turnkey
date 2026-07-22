from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from turnkey._internal.io import load_json_object
from turnkey.matrix import write_matrix_plan


MATRIX_SPECS: dict[str, dict[str, Any]] = {
    "example": {
        "path": Path("configs/matrix/example.yaml"),
        "counts": {"entries": 2, "planned": 2, "skipped": 0},
        "detectors": {"allow_all", "keyword"},
        "model_ids": {"dummy-smoke"},
        "judges": {"dummy_refusal"},
    }
}


def _planned_entries(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in plan["entries"] if entry["status"] == "planned"]


def _assert_set(actual: set[str], expected: set[str], *, label: str, spec_name: str) -> None:
    if actual != expected:
        raise AssertionError(
            f"{spec_name}: {label} drifted: expected {sorted(expected)}, got {sorted(actual)}"
        )


def check_one(spec_name: str, *, out_root: Path) -> Path:
    expected = MATRIX_SPECS[spec_name]
    plan_path = write_matrix_plan(spec_path=expected["path"], out_dir=out_root / spec_name)
    plan = load_json_object(plan_path)

    if plan["counts"] != expected["counts"]:
        raise AssertionError(
            f"{spec_name}: counts drifted: expected {expected['counts']}, got {plan['counts']}"
        )

    planned = _planned_entries(plan)
    if len(planned) != expected["counts"]["planned"]:
        raise AssertionError(f"{spec_name}: planned entry list length does not match counts")

    _assert_set(
        {entry["detector"]["name"] for entry in planned},
        set(expected["detectors"]),
        label="detectors",
        spec_name=spec_name,
    )
    _assert_set(
        {entry["model"]["model_id"] for entry in planned},
        set(expected["model_ids"]),
        label="model ids",
        spec_name=spec_name,
    )
    _assert_set(
        {entry["judge"]["name"] for entry in planned},
        set(expected["judges"]),
        label="judges",
        spec_name=spec_name,
    )

    print(f"OK {spec_name}: {plan_path}")
    return plan_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check CI-critical Turnkey matrix plans")
    parser.add_argument("--out-root", default="outputs/ci-matrix-plan", help="Output root")
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for spec_name in MATRIX_SPECS:
        check_one(spec_name, out_root=out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
