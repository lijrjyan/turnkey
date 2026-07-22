from __future__ import annotations

from dataclasses import dataclass

from turnkey._internal.hf_deps import import_datasets as _import_datasets
from turnkey.components.datasets._prompt_rows import instruction_content_answer_prompt
from turnkey.components.datasets._revision import DatasetLoadResult, resolve_loaded_dataset_revision
from turnkey.schema import Sample


@dataclass(frozen=True)
class SorryBenchPublicParams:
    revision: str | None = None
    shuffle: bool = True
    seed: int = 0
    limit: int | None = None
    split: str = "train"


def load_sorrybench_public(*, params: SorryBenchPublicParams) -> list[Sample]:
    return load_sorrybench_public_with_metadata(params=params).samples


def load_sorrybench_public_with_metadata(*, params: SorryBenchPublicParams) -> DatasetLoadResult:
    """
    Public, HF-hosted subset compatible with the SorryBench schema.

    We use `AlignmentResearch/SorryBench` as a lightweight, accessible stand-in
    for early integration tests. It is **not** the full `sorry-bench/sorry-bench-202406`
    dataset (which is gated).
    """
    load_dataset = _import_datasets()
    ds = load_dataset("AlignmentResearch/SorryBench", split=params.split, revision=params.revision)
    resolved_revision = resolve_loaded_dataset_revision([ds], requested_revision=params.revision)
    if params.shuffle:
        ds = ds.shuffle(seed=params.seed)
    if params.limit is not None:
        ds = ds.select(range(max(0, int(params.limit))))

    out: list[Sample] = []
    for i, row in enumerate(ds):
        prompt = instruction_content_answer_prompt(
            row,
            empty_message="SorryBench row produced empty prompt text",
        )
        out.append(
            Sample(
                sample_id=f"sorrybench_public-{i:04d}",
                behavior_id=f"sorrybench_public:{i:04d}",
                is_benign=False,
                prompt=prompt,
                attack_family="T0",
                attack_method="none",
                attack_params={"dataset": "SorryBench(public)", "clf_label": row.get("clf_label")},
            )
        )
    return DatasetLoadResult(samples=out, resolved_revision=resolved_revision)
