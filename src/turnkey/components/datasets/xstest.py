from __future__ import annotations

from dataclasses import dataclass

from turnkey._internal.hf_deps import import_datasets as _import_datasets
from turnkey.components.datasets._prompt_rows import instruction_content_answer_prompt
from turnkey.components.datasets._revision import DatasetLoadResult, resolve_loaded_dataset_revision
from turnkey.schema import Sample


@dataclass(frozen=True)
class XSTestParams:
    revision: str | None = None
    shuffle: bool = True
    seed: int = 0
    benign_limit: int | None = None
    harmful_limit: int | None = None
    split: str = "validation"


def load_xstest(*, params: XSTestParams) -> list[Sample]:
    return load_xstest_with_metadata(params=params).samples


def load_xstest_with_metadata(*, params: XSTestParams) -> DatasetLoadResult:
    """
    XSTest loader (AlignmentResearch/XSTest).

    The dataset includes a `clf_label` column; we treat:
    - clf_label == 0 as benign (should comply)
    - clf_label != 0 as harmful/should-refuse bucket

    This matches the dataset's intent of measuring exaggerated refusals while
    still including unsafe cases.
    """
    load_dataset = _import_datasets()
    ds = load_dataset("AlignmentResearch/XSTest", split=params.split, revision=params.revision)
    resolved_revision = resolve_loaded_dataset_revision([ds], requested_revision=params.revision)
    if params.shuffle:
        ds = ds.shuffle(seed=params.seed)

    benign_rows = []
    harmful_rows = []
    for row in ds:
        label = row.get("clf_label")
        is_benign = isinstance(label, int) and label == 0
        (benign_rows if is_benign else harmful_rows).append(row)

    if params.benign_limit is not None:
        benign_rows = benign_rows[: max(0, int(params.benign_limit))]
    if params.harmful_limit is not None:
        harmful_rows = harmful_rows[: max(0, int(params.harmful_limit))]

    out: list[Sample] = []
    for is_benign, rows in ((True, benign_rows), (False, harmful_rows)):
        for i, row in enumerate(rows):
            prompt = instruction_content_answer_prompt(
                row,
                empty_message="XSTest row produced empty prompt text",
            )
            sample_id = f"xstest-{'benign' if is_benign else 'harmful'}-{i:04d}"
            behavior_id = f"xstest:{row.get('gen_target') or row.get('proxy_gen_target') or i}"
            out.append(
                Sample(
                    sample_id=sample_id,
                    behavior_id=behavior_id,
                    is_benign=is_benign,
                    prompt=prompt,
                    attack_family="T0",
                    attack_method="none",
                    attack_params={"dataset": "XSTest", "clf_label": row.get("clf_label")},
                )
            )
    return DatasetLoadResult(samples=out, resolved_revision=resolved_revision)
