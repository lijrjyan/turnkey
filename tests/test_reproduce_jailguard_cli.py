from __future__ import annotations

import json
import subprocess
import sys


def _run_reproduce_jailguard(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "scripts/reproduce_jailguard.py", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_reproduce_jailguard_accepts_reference_mask_mutators() -> None:
    result = _run_reproduce_jailguard(
        "--mode",
        "extra-echo",
        "--mutator",
        "TI",
        "--n-variants",
        "2",
        "--threshold",
        "10",
        "--similarity",
        "bow",
        "--seed",
        "1",
    )

    assert set(result) == {"block", "score", "reason"}
    assert "jailguard(" in result["reason"]


def test_reproduce_jailguard_accepts_reference_policy_override() -> None:
    result = _run_reproduce_jailguard(
        "--mode",
        "extra-echo",
        "--mutator",
        "PL",
        "--policy-pool",
        "PI,TI,TL",
        "--policy-probs",
        "0,1,0",
        "--n-variants",
        "2",
        "--threshold",
        "10",
        "--similarity",
        "bow",
        "--seed",
        "1",
    )

    assert "jailguard(" in result["reason"]
