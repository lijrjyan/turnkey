from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from turnkey._internal.hf_deps import cached_hf_login_token, env_hf_token
from turnkey._internal.hf_deps import import_datasets as _import_datasets
from turnkey.components.datasets._prompt_rows import sorrybench_202406_prompt
from turnkey.components.datasets._revision import DatasetLoadResult, resolve_loaded_dataset_revision
from turnkey.schema import Sample


DATASET_ID = "sorry-bench/sorry-bench-202406"


@dataclass(frozen=True)
class SorryBench202406Params:
    revision: str | None = None
    shuffle: bool = True
    seed: int = 0
    limit: int | None = None
    split: str = "train"
    token_env: str = "HF_TOKEN"


def _auth_token(token_env: str) -> str | bool:
    token = env_hf_token(token_env)
    if token:
        return token
    if _has_cached_hf_token():
        return True
    raise RuntimeError(
        f"{DATASET_ID} is gated. Set {token_env}, or run `hf auth login` / "
        "`huggingface-cli login`, then retry."
    )


def _has_cached_hf_token() -> bool:
    return bool(cached_hf_login_token())


def _row_value(row: dict[str, Any], *keys: str) -> object:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def load_sorrybench_202406(*, params: SorryBench202406Params) -> list[Sample]:
    return load_sorrybench_202406_with_metadata(params=params).samples


def load_sorrybench_202406_with_metadata(*, params: SorryBench202406Params) -> DatasetLoadResult:
    load_dataset = _import_datasets()
    try:
        token = _auth_token(params.token_env)
        ds = load_dataset(
            DATASET_ID,
            split=params.split,
            revision=params.revision,
            token=token,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"{DATASET_ID} is gated or unavailable. Request access on Hugging Face and set "
            f"{params.token_env}, or run `hf auth login` / `huggingface-cli login`, then retry."
        ) from e

    resolved_revision = resolve_loaded_dataset_revision([ds], requested_revision=params.revision)
    if params.shuffle:
        ds = ds.shuffle(seed=params.seed)
    limit = None if params.limit is None else max(0, int(params.limit))
    if limit == 0:
        return DatasetLoadResult(samples=[], resolved_revision=resolved_revision)

    entries: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if limit is not None and len(entries) >= limit:
            break
        row_dict = dict(row)
        try:
            prompt = sorrybench_202406_prompt(
                row_dict,
                empty_message="SORRY-Bench 202406 row produced empty prompt text",
            )
        except ValueError as e:
            if "produced empty prompt text" not in str(e):
                raise
            continue
        row_id = _row_value(row_dict, "id", "sample_id", "behavior_id", "question_id") or f"{i:04d}"
        category = _row_value(row_dict, "category", "topic", "policy", "clf_label")
        prompt_style = _row_value(row_dict, "prompt_style", "style", "mutation")
        entries.append(
            {
                "source_index": i,
                "row_id": str(row_id),
                "category": category,
                "prompt_style": prompt_style,
                "prompt": prompt,
            }
        )

    row_id_counts = Counter(str(entry["row_id"]) for entry in entries)
    out: list[Sample] = []
    for entry in entries:
        row_id = str(entry["row_id"])
        sample_id = f"sorrybench_202406-{row_id}"
        if row_id_counts[row_id] > 1:
            sample_id = f"{sample_id}-{int(entry['source_index']):04d}"
        out.append(
            Sample(
                sample_id=sample_id,
                behavior_id=f"sorrybench_202406:{row_id}",
                is_benign=False,
                prompt=str(entry["prompt"]),
                attack_family="T0",
                attack_method="none",
                attack_params={
                    "dataset": "sorry-bench/sorry-bench-202406",
                    "category": entry["category"],
                    "prompt_style": entry["prompt_style"],
                },
            )
        )
    return DatasetLoadResult(samples=out, resolved_revision=resolved_revision)
