from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


CACHE_ENTRY_MANIFEST_SCHEMA = "turnkey_cache_entry_manifest/v1"
CACHE_MANIFEST_FILENAME = "cache_manifest.json"


@dataclass(frozen=True)
class CacheEntryWriter:
    root: Path
    entry_name: str
    capability: str
    params_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def staging_dir(self) -> Path:
        return self.root / f"{self.entry_name}.tmp"

    @property
    def final_dir(self) -> Path:
        return self.root / self.entry_name

    def commit(self) -> Path:
        if not self.staging_dir.exists() or not self.staging_dir.is_dir():
            raise FileNotFoundError(f"cache staging directory does not exist: {self.staging_dir}")
        if self.final_dir.exists():
            raise FileExistsError(f"cache entry already exists: {self.final_dir}")

        manifest = build_cache_manifest(
            entry_dir=self.staging_dir,
            entry_name=self.entry_name,
            capability=self.capability,
            params_hash=self.params_hash,
            metadata=self.metadata,
        )
        manifest_path = self.staging_dir / CACHE_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.staging_dir.replace(self.final_dir)
        return self.final_dir


def begin_cache_entry(
    root: str | Path,
    entry_name: str,
    *,
    capability: str,
    params_hash: str,
    metadata: dict[str, Any] | None = None,
) -> CacheEntryWriter:
    _validate_entry_name(entry_name)
    if not capability.strip():
        raise ValueError("cache capability must be non-empty")
    if not params_hash.strip():
        raise ValueError("cache params_hash must be non-empty")

    writer = CacheEntryWriter(
        root=Path(root),
        entry_name=entry_name,
        capability=capability,
        params_hash=params_hash,
        metadata=dict(metadata or {}),
    )
    writer.root.mkdir(parents=True, exist_ok=True)
    if writer.staging_dir.exists():
        raise FileExistsError(f"cache staging directory already exists: {writer.staging_dir}")
    if writer.final_dir.exists():
        raise FileExistsError(f"cache entry already exists: {writer.final_dir}")
    writer.staging_dir.mkdir()
    return writer


def _validate_entry_name(entry_name: str) -> None:
    if not entry_name.strip():
        raise ValueError("cache entry name must be non-empty")
    if entry_name != entry_name.strip():
        raise ValueError("cache entry name must not contain leading or trailing whitespace")
    if entry_name in {".", ".."} or "/" in entry_name or "\\" in entry_name or Path(entry_name).is_absolute():
        raise ValueError("cache entry name must be a single relative path segment")


def build_cache_manifest(
    *,
    entry_dir: Path,
    entry_name: str,
    capability: str,
    params_hash: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_ENTRY_MANIFEST_SCHEMA,
        "ready": True,
        "entry_name": entry_name,
        "capability": capability,
        "params_hash": params_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
        "files": _entry_files(entry_dir),
    }


def load_cache_manifest(entry_dir: str | Path) -> dict[str, Any]:
    manifest_path = _manifest_path(Path(entry_dir))
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{manifest_path}: cache manifest must be a JSON object")
    return value


def cache_entry_ready(entry_dir: str | Path) -> bool:
    entry_path = Path(entry_dir)
    try:
        manifest = load_cache_manifest(entry_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if manifest.get("schema_version") != CACHE_ENTRY_MANIFEST_SCHEMA:
        return False
    if manifest.get("ready") is not True:
        return False
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return False
    for rel_path, identity in files.items():
        if not isinstance(rel_path, str) or not isinstance(identity, dict):
            return False
        if not _valid_manifest_relative_path(rel_path):
            return False
        if identity.get("path") != rel_path:
            return False
        path = entry_path / rel_path
        if not path.exists() or not path.is_file():
            return False
        actual = _file_identity(path, relative_to=entry_path)
        if actual != identity:
            return False
    return True


def _manifest_path(entry_dir: Path) -> Path:
    if entry_dir.name == CACHE_MANIFEST_FILENAME:
        return entry_dir
    return entry_dir / CACHE_MANIFEST_FILENAME


def _valid_manifest_relative_path(rel_path: str) -> bool:
    if not rel_path or "\\" in rel_path:
        return False
    path = PurePosixPath(rel_path)
    if path.is_absolute():
        return False
    return all(part not in {"", ".", ".."} for part in path.parts)


def _entry_files(entry_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(entry_dir.rglob("*")):
        if not path.is_file() or path.name == CACHE_MANIFEST_FILENAME:
            continue
        rel_path = path.relative_to(entry_dir).as_posix()
        files[rel_path] = _file_identity(path, relative_to=entry_dir)
    return files


def _file_identity(path: Path, *, relative_to: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
