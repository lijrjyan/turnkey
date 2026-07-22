from __future__ import annotations

from dataclasses import dataclass

from turnkey._internal.hf_deps import import_datasets as _import_datasets
from turnkey.components.datasets._revision import DatasetLoadResult, resolve_loaded_dataset_revision
from turnkey.schema import Sample


@dataclass(frozen=True)
class JBBBehaviorsParams:
    revision: str | None = None
    shuffle: bool = True
    seed: int = 0
    harmful_limit: int | None = None
    benign_limit: int | None = None


def load_jbb_behaviors(*, params: JBBBehaviorsParams) -> list[Sample]:
    return load_jbb_behaviors_with_metadata(params=params).samples


def load_jbb_behaviors_with_metadata(*, params: JBBBehaviorsParams) -> DatasetLoadResult:
    """
    JailbreakBench/JBB-Behaviors loader.

    Notes:
    - The HF dataset provides two splits: `harmful` and `benign`.
    - We default to using the `Goal` field as the prompt text.
    - Prompts are not logged in plaintext by default (handled by the runner).
    """
    load_dataset = _import_datasets()
    harmful = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        "behaviors",
        split="harmful",
        revision=params.revision,
    )
    benign = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        "behaviors",
        split="benign",
        revision=params.revision,
    )
    resolved_revision = resolve_loaded_dataset_revision(
        [harmful, benign],
        requested_revision=params.revision,
    )

    if params.shuffle:
        harmful = harmful.shuffle(seed=params.seed)
        benign = benign.shuffle(seed=params.seed)

    if params.harmful_limit is not None:
        harmful = harmful.select(range(max(0, int(params.harmful_limit))))
    if params.benign_limit is not None:
        benign = benign.select(range(max(0, int(params.benign_limit))))

    out: list[Sample] = []
    for split_name, ds, is_benign in (("harmful", harmful, False), ("benign", benign, True)):
        for row in ds:
            idx = row.get("Index")
            if not isinstance(idx, int):
                idx = None
            sample_id = f"jbb-{split_name}-{idx if idx is not None else len(out)}"
            behavior_id = f"jbb:{idx}" if idx is not None else f"jbb:row:{len(out)}"
            goal = row.get("Goal")
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("JBB-Behaviors row missing non-empty `Goal` field")

            out.append(
                Sample(
                    sample_id=sample_id,
                    behavior_id=behavior_id,
                    is_benign=is_benign,
                    prompt=goal,
                    attack_family="T0",
                    attack_method="none",
                    attack_params={
                        "dataset": "JBB-Behaviors",
                        "category": row.get("Category"),
                        "source": row.get("Source"),
                    },
                )
            )
    return DatasetLoadResult(samples=out, resolved_revision=resolved_revision)
