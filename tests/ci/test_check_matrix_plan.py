from __future__ import annotations

from pathlib import Path
from types import ModuleType

from script_loader import load_script_module

def test_matrix_plan_ci_helper_generates_expected_plan_set(
    tmp_path: Path,
) -> None:
    helper: ModuleType = load_script_module(Path("scripts/ci/check_matrix_plan.py"))

    exit_code = helper.main(["--out-root", str(tmp_path / "plans")])

    assert exit_code == 0
    expected = {
        "qwen3_0_6b_bounded_reproduction_v9",
        "qwen35_2b_main_v11",
        "qwen35_2b_lofo_v11",
        "qwen25_7b_robustness_v11",
    }
    assert {path.parent.name for path in (tmp_path / "plans").glob("*/plan.json")} == expected
