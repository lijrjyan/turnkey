#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = REPO_ROOT / "docs" / "notebooks"


def notebook_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted(NOTEBOOK_ROOT.glob("*.ipynb")):
        document = json.loads(path.read_text(encoding="utf-8"))
        turnkey_metadata = document.get("metadata", {}).get("turnkey", {})
        if turnkey_metadata.get("requires_gpu") is False:
            paths.append(path)
    return paths


def execute_notebook(path: Path) -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError("notebook execution requires `uv sync --extra docs`") from exc

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute Turnkey's CPU-safe learning notebooks")
    parser.add_argument("--list", action="store_true", help="List notebooks without executing them")
    args = parser.parse_args(argv)

    paths = notebook_paths()
    if args.list:
        for path in paths:
            print(path.relative_to(REPO_ROOT))
        return 0

    for path in paths:
        print(f"executing {path.relative_to(REPO_ROOT)}", flush=True)
        execute_notebook(path)
    print(f"executed {len(paths)} CPU-safe notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
