from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from turnkey._internal.io import load_json_object
from turnkey.matrix import write_matrix_plan


MATRIX_SPECS: dict[str, dict[str, Any]] = {
    "qwen3_0_6b_bounded_reproduction_v9": {
        "path": Path("configs/matrix/qwen3_0_6b_bounded_reproduction_v9.yaml"),
        "counts": {"entries": 84, "planned": 84, "skipped": 0},
        "detectors": {"rcs_paper_v3", "gradsafe_v3", "jailguard_v3"},
        "model_ids": {"Qwen/Qwen3-0.6B"},
        "judges": {"strongreject"},
    },
    "qwen35_2b_main_v11": {
        "path": Path("configs/matrix/qwen35_2b_main_v11.yaml"),
        "counts": {"entries": 160, "planned": 160, "skipped": 0},
        "detectors": {"allow_all_v3", "keyword_v3", "rcs_paper_v3", "gradsafe_v3", "jailguard_v3"},
        "model_ids": {"Qwen/Qwen3.5-2B"},
        "judges": {"strongreject"},
    },
    "qwen35_2b_lofo_v11": {
        "path": Path("configs/matrix/qwen35_2b_lofo_v11.yaml"),
        "counts": {"entries": 35, "planned": 35, "skipped": 0},
        "detectors": {"rcs_paper_v3", "gradsafe_v3", "jailguard_v3", "keyword_v3", "allow_all_v3"},
        "model_ids": {"Qwen/Qwen3.5-2B"},
        "judges": {"strongreject"},
        "lofo_holdouts": {"persona", "manyshot", "deepinception", "artprompt", "tap", "crescendo", "pair"},
    },
    "qwen25_7b_robustness_v11": {
        "path": Path("configs/matrix/qwen25_7b_robustness_v11.yaml"),
        "counts": {"entries": 10, "planned": 10, "skipped": 0},
        "detectors": {"allow_all_v3", "keyword_v3", "rcs_paper_v3", "gradsafe_v3", "jailguard_v3"},
        "model_ids": {"Qwen/Qwen2.5-7B-Instruct"},
        "judges": {"strongreject"},
    },
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

    if "lofo_holdouts" in expected:
        _assert_set(
            {entry["lofo"]["holdout_attack"] for entry in planned},
            set(expected["lofo_holdouts"]),
            label="LOFO holdouts",
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
