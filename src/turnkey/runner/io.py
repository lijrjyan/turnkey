from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turnkey._internal.redact import sha256_hex


def runtime_cache_key(value: Any) -> str:
    return json.dumps(asdict(value), sort_keys=True, ensure_ascii=False, default=str)


def utc_run_id(name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in name.strip()) or "run"
    return f"{ts}-{safe}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        commit = None

    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(status.strip())
    except Exception:  # noqa: BLE001
        dirty = None

    return {"commit": commit, "dirty": dirty}


def file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": sha256_hex(data.decode("utf-8")),
        "bytes": len(data),
    }
