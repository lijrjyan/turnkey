from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from turnkey.config import ContentConfig
from turnkey._internal.redact import sha256_hex
from turnkey.schema import Sample


@dataclass(frozen=True)
class ContentSelectionReport:
    config: ContentConfig
    input_count: int
    selected_count: int
    sample_ids: list[str]
    behavior_ids: list[str]
    selection_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "input_count": self.input_count,
            "selected_count": self.selected_count,
            "sample_ids": list(self.sample_ids),
            "behavior_ids": list(self.behavior_ids),
            "selection_hash": self.selection_hash,
        }


def select_content(samples: list[Sample], cfg: ContentConfig) -> tuple[list[Sample], ContentSelectionReport]:
    _validate_content_config(cfg)
    selected = list(samples)

    if cfg.sample_ids:
        selected = _select_by_sample_ids(selected, cfg.sample_ids)

    if cfg.behavior_ids:
        selected = _select_by_behavior_ids(selected, cfg.behavior_ids)

    if cfg.shuffle:
        rng = random.Random(int(cfg.seed))  # noqa: S311
        rng.shuffle(selected)

    if cfg.limit is not None:
        selected = selected[: int(cfg.limit)]

    sample_ids = [sample.sample_id for sample in selected]
    behavior_ids = [sample.behavior_id for sample in selected]
    selection_hash = sha256_hex(
        json.dumps(
            {
                "sample_ids": sample_ids,
                "behavior_ids": behavior_ids,
                "config": asdict(cfg),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )

    return selected, ContentSelectionReport(
        config=cfg,
        input_count=len(samples),
        selected_count=len(selected),
        sample_ids=sample_ids,
        behavior_ids=behavior_ids,
        selection_hash=selection_hash,
    )


def _validate_content_config(cfg: ContentConfig) -> None:
    if not isinstance(cfg.sample_ids, list) or not all(isinstance(item, str) for item in cfg.sample_ids):
        raise ValueError("content.sample_ids must be a list[str]")
    if not isinstance(cfg.behavior_ids, list) or not all(isinstance(item, str) for item in cfg.behavior_ids):
        raise ValueError("content.behavior_ids must be a list[str]")
    if len(set(cfg.sample_ids)) != len(cfg.sample_ids):
        raise ValueError("content.sample_ids contains duplicates")
    if len(set(cfg.behavior_ids)) != len(cfg.behavior_ids):
        raise ValueError("content.behavior_ids contains duplicates")
    if cfg.limit is not None and int(cfg.limit) < 0:
        raise ValueError("content.limit must be >= 0")


def _select_by_sample_ids(samples: list[Sample], sample_ids: list[str]) -> list[Sample]:
    by_id = {sample.sample_id: sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        available = sorted(by_id)[:10]
        raise ValueError(
            f"content.sample_ids not found: {missing}. "
            f"First available sample_ids: {available}"
        )
    return [by_id[sample_id] for sample_id in sample_ids]


def _select_by_behavior_ids(samples: list[Sample], behavior_ids: list[str]) -> list[Sample]:
    requested = set(behavior_ids)
    selected = [sample for sample in samples if sample.behavior_id in requested]
    found = {sample.behavior_id for sample in selected}
    missing = [behavior_id for behavior_id in behavior_ids if behavior_id not in found]
    if missing:
        available = sorted({sample.behavior_id for sample in samples})[:10]
        raise ValueError(
            f"content.behavior_ids not found: {missing}. "
            f"First available behavior_ids: {available}"
        )
    return selected
