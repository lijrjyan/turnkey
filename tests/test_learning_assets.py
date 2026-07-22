from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


NOTEBOOKS = (
    "00_quickstart.ipynb",
    "01_build_your_own_detector.ipynb",
    "02_debug_a_missed_case.ipynb",
    "03_typed_signals.ipynb",
)
NOTEBOOK_ROOT = Path("docs/notebooks")


def test_learning_notebooks_are_cpu_safe_and_have_turnkey_metadata() -> None:
    for name in NOTEBOOKS:
        path = NOTEBOOK_ROOT / name
        document = json.loads(path.read_text(encoding="utf-8"))

        assert document["nbformat"] == 4
        assert document["metadata"]["kernelspec"]["name"] == "python3"
        assert document["metadata"]["turnkey"]["requires_gpu"] is False
        sources = "\n".join(
            "".join(cell.get("source", [])) for cell in document["cells"]
        )
        assert "Turnkey" in sources
        assert "pip install torch" not in sources


def test_mkdocs_navigation_links_every_learning_notebook() -> None:
    config = Path("mkdocs.yml").read_text(encoding="utf-8")
    for name in NOTEBOOKS:
        assert f"notebooks/{name}" in config


def test_notebook_runner_lists_the_cpu_safe_suite() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/ci/run_notebooks.py", "--list"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [str(NOTEBOOK_ROOT / name) for name in NOTEBOOKS]
