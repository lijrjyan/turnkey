from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from turnkey.schema import Sample


@dataclass(frozen=True)
class JsonlPromptsParams:
    path: str
    shuffle: bool = False
    seed: int = 0
    limit: int | None = None
    prompt_field: str = "prompt"
    is_benign_field: str = "is_benign"
    sample_id_field: str = "sample_id"
    behavior_id_field: str = "behavior_id"


def load_jsonl_prompts(*, params: JsonlPromptsParams) -> list[Sample]:
    path = Path(params.path)
    if not path.exists():
        raise FileNotFoundError(f"jsonl_prompts dataset path does not exist: {path}")

    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl_prompts invalid JSON at {path}:{i}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_prompts expected object at {path}:{i}")
            samples.append(_row_to_sample(row, index=len(samples), params=params, source=str(path)))

    if params.shuffle:
        rng = random.Random(int(params.seed))
        rng.shuffle(samples)
    if params.limit is not None:
        samples = samples[: max(0, int(params.limit))]
    return samples


def _row_to_sample(row: dict[str, Any], *, index: int, params: JsonlPromptsParams, source: str) -> Sample:
    prompt = row.get(params.prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"jsonl_prompts missing non-empty {params.prompt_field!r} at row {index}")
    is_benign = row.get(params.is_benign_field)
    if not isinstance(is_benign, bool):
        raise ValueError(f"jsonl_prompts missing boolean {params.is_benign_field!r} at row {index}")

    sample_id = row.get(params.sample_id_field)
    if not isinstance(sample_id, str) or not sample_id.strip():
        sample_id = f"jsonl-{index:04d}"
    behavior_id = row.get(params.behavior_id_field)
    if not isinstance(behavior_id, str) or not behavior_id.strip():
        behavior_id = f"jsonl_prompts:{'benign' if is_benign else 'harmful'}:{index:04d}"

    attack_params = row.get("attack_params")
    if not isinstance(attack_params, dict):
        attack_params = {}
    attack_params = {"dataset": "jsonl_prompts", "source": source, **attack_params}

    attack_family = row.get("attack_family")
    attack_method = row.get("attack_method")
    return Sample(
        sample_id=sample_id,
        behavior_id=behavior_id,
        is_benign=is_benign,
        prompt=prompt.strip(),
        attack_family=attack_family if isinstance(attack_family, str) else "T0",
        attack_method=attack_method if isinstance(attack_method, str) else "none",
        attack_params=attack_params,
    )
