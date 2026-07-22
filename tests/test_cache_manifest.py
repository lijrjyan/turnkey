from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnkey.cache_manifest import (
    CACHE_ENTRY_MANIFEST_SCHEMA,
    begin_cache_entry,
    cache_entry_ready,
    load_cache_manifest,
)


def test_cache_entry_commit_writes_ready_manifest_after_promoting_staging_dir(tmp_path: Path) -> None:
    writer = begin_cache_entry(
        tmp_path / "cache",
        "hidden-state-qwen",
        capability="last_token_by_layer",
        params_hash="params-sha",
        metadata={"model_id": "qwen"},
    )
    assert writer.staging_dir.exists()
    assert not writer.final_dir.exists()
    assert cache_entry_ready(writer.staging_dir) is False

    payload = writer.staging_dir / "tensor.bin"
    payload.write_bytes(b"hidden-state-bytes")

    final_dir = writer.commit()

    assert final_dir == tmp_path / "cache" / "hidden-state-qwen"
    assert final_dir.exists()
    assert not writer.staging_dir.exists()
    assert cache_entry_ready(final_dir) is True

    manifest = load_cache_manifest(final_dir)
    assert manifest["schema_version"] == CACHE_ENTRY_MANIFEST_SCHEMA
    assert manifest["ready"] is True
    assert manifest["capability"] == "last_token_by_layer"
    assert manifest["params_hash"] == "params-sha"
    assert manifest["metadata"] == {"model_id": "qwen"}
    assert manifest["files"] == {
        "tensor.bin": {
            "path": "tensor.bin",
            "sha256": "b3adc1030652f3206881d7b5ae6c34687dede472c161caac898143cc75887bd9",
            "bytes": len(b"hidden-state-bytes"),
        }
    }


def test_cache_entry_ready_rejects_missing_or_corrupt_manifest(tmp_path: Path) -> None:
    entry_dir = tmp_path / "cache" / "incomplete"
    entry_dir.mkdir(parents=True)
    (entry_dir / "payload.bin").write_bytes(b"payload")

    assert cache_entry_ready(entry_dir) is False

    (entry_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CACHE_ENTRY_MANIFEST_SCHEMA,
                "ready": True,
                "entry_name": "incomplete",
                "capability": "model_responses",
                "params_hash": "abc",
                "metadata": {},
                "files": {
                    "payload.bin": {
                        "path": "payload.bin",
                        "sha256": "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5",
                        "bytes": 7,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert cache_entry_ready(entry_dir) is True

    (entry_dir / "payload.bin").write_bytes(b"tampered")

    assert cache_entry_ready(entry_dir) is False


@pytest.mark.parametrize("entry_name", ["../escape", "/absolute", "nested/name", "nested\\name", ".", ".."])
def test_begin_cache_entry_rejects_path_like_entry_names(tmp_path: Path, entry_name: str) -> None:
    with pytest.raises(ValueError, match="cache entry name"):
        begin_cache_entry(
            tmp_path / "cache",
            entry_name,
            capability="model_responses",
            params_hash="abc",
        )


def test_cache_entry_ready_rejects_manifest_paths_outside_entry(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    entry_dir = cache_root / "entry"
    entry_dir.mkdir(parents=True)
    outside_payload = cache_root / "outside.bin"
    outside_payload.write_bytes(b"payload")

    (entry_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CACHE_ENTRY_MANIFEST_SCHEMA,
                "ready": True,
                "entry_name": "entry",
                "capability": "model_responses",
                "params_hash": "abc",
                "metadata": {},
                "files": {
                    "../outside.bin": {
                        "path": "../outside.bin",
                        "sha256": "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5",
                        "bytes": 7,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert cache_entry_ready(entry_dir) is False
