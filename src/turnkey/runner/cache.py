from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

from turnkey.config import JudgeConfig, ModelConfig

from .cache_keys import runtime_cache_entry_key_sha256
from .io import runtime_cache_key


RUNTIME_CACHE_MANIFEST_SCHEMA_V1 = "turnkey_runtime_cache_manifest/v1"
RUNTIME_CACHE_MANIFEST_SCHEMA = "turnkey_runtime_cache_manifest/v2"


def _runner_attr(name: str) -> Any:
    import turnkey.runner as runner

    return getattr(runner, name)


@dataclass
class RunResourceCache:
    """Reusable runtime resources for a sequence of evaluations in one process."""

    backends: dict[str, Any] = field(default_factory=dict)
    judges: dict[str, Any] = field(default_factory=dict)
    _released_backend_entries: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def _cached(self, bucket: dict[Any, Any], key: Any, factory) -> Any:
        value = bucket.get(key)
        if value is None:
            value = factory()
            bucket[key] = value
        return value

    def get_backend(self, cfg: ModelConfig) -> Any:
        key = runtime_cache_key(cfg)
        return self._cached(self.backends, key, lambda: _runner_attr("load_backend")(cfg))

    def get_judge(self, cfg: JudgeConfig) -> Any:
        key = runtime_cache_key(cfg)
        return self._cached(self.judges, key, lambda: _runner_attr("load_judge")(cfg))

    def release_backend(self, cfg: ModelConfig) -> None:
        key = runtime_cache_key(cfg)
        value = self.backends.get(key)
        if value is None:
            return
        entry = _cache_entry(bucket_name="backends", key=key, value=value, ready=False)
        self._released_backend_entries[entry["key_sha256"]] = entry
        self.backends.pop(key, None)
        _close_cached_value(value)
        self.release_unused_accelerator_memory()

    def release_unused_accelerator_memory(self) -> None:
        gc.collect()
        try:
            import torch
        except Exception:  # noqa: BLE001
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def summary(self) -> dict[str, int]:
        return {
            "backends": len(self.backends),
            "judges": len(self.judges),
        }

    def manifest(self) -> dict[str, Any]:
        current_entries = _cache_entries({"backends": self.backends, "judges": self.judges})
        entries_by_identity = {
            ("backends", key_sha256): dict(entry)
            for key_sha256, entry in self._released_backend_entries.items()
        }
        for entry in current_entries:
            entries_by_identity[(entry["bucket"], entry["key_sha256"])] = entry

        return {
            "schema_version": RUNTIME_CACHE_MANIFEST_SCHEMA,
            "counts": self.summary(),
            "entries": sorted(
                entries_by_identity.values(),
                key=lambda entry: (entry["bucket"], entry["key_sha256"]),
            ),
        }


def _cache_entries(buckets: dict[str, dict[Any, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for bucket_name, bucket in buckets.items():
        for key, value in bucket.items():
            entries.append(_cache_entry(bucket_name=bucket_name, key=key, value=value))
    return sorted(entries, key=lambda entry: (entry["bucket"], entry["key_sha256"]))


def _cache_entry(
    *,
    bucket_name: str,
    key: Any,
    value: Any,
    ready: bool = True,
) -> dict[str, Any]:
    value_type = type(value)
    return {
        "bucket": bucket_name,
        "key_sha256": runtime_cache_entry_key_sha256(key),
        "object_type": f"{value_type.__module__}.{value_type.__qualname__}",
        "ready": ready,
    }


def _close_cached_value(value: Any) -> None:
    close = getattr(value, "close", None)
    if not callable(close):
        return
    close()
