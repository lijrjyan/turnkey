from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from turnkey.schema import ImageInput, Sample


_RCS_TRAIN_JSONL_SCHEMA = {
    "version": "rcs_train_jsonl/v1",
    "required_fields": ["prompt:str", "is_benign:bool"],
    "optional_fields": [
        "dataset:str",
        "split:train|validation",
        "images:str|list[str|{path:str,mime_type?:str}]",
    ],
}


@dataclass(frozen=True)
class _RCSTrainExample:
    prompt: str
    images: tuple[ImageInput, ...]
    is_benign: bool
    dataset: str
    split: str | None = None


def _normalize_train_split(value: Any, *, path: str, line_no: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"rcs: invalid split at {path}:{line_no} (expected str)")
    split = value.strip().lower()
    if split in {"", "none"}:
        return None
    if split in {"val", "valid", "validation", "dev"}:
        return "validation"
    if split == "train":
        return "train"
    raise ValueError(f"rcs: invalid split at {path}:{line_no} (expected train or validation)")


_HiddenStateProvider = Callable[[Sample], Any]


def _coerce_images(value: Any) -> tuple[ImageInput, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (ImageInput(path=value),)
    if isinstance(value, (list, tuple)):
        out: list[ImageInput] = []
        for item in value:
            if isinstance(item, str):
                out.append(ImageInput(path=item))
            elif isinstance(item, dict):
                path = item.get("path")
                if not isinstance(path, str) or not path:
                    raise ValueError("rcs: invalid images[].path (expected non-empty str)")
                mime = item.get("mime_type")
                out.append(ImageInput(path=path, mime_type=mime if isinstance(mime, str) else None))
            else:
                raise ValueError("rcs: invalid images[] entry (expected str or dict)")
        return tuple(out)
    raise ValueError("rcs: invalid images (expected str or list)")


def _load_train_examples_jsonl(path: str) -> list[_RCSTrainExample]:
    p = Path(path)
    rows = p.read_text(encoding="utf-8").splitlines()
    out: list[_RCSTrainExample] = []
    for i, line in enumerate(rows, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"rcs: invalid jsonl at {path}:{i}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"rcs: invalid jsonl record at {path}:{i} (expected object)")
        prompt = obj.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"rcs: missing/invalid prompt at {path}:{i}")
        is_benign = obj.get("is_benign")
        if not isinstance(is_benign, bool):
            raise ValueError(f"rcs: missing/invalid is_benign at {path}:{i}")
        dataset = obj.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            dataset = "default"
        images = _coerce_images(obj.get("images"))
        split = _normalize_train_split(obj.get("split"), path=path, line_no=i)
        out.append(_RCSTrainExample(prompt=prompt, images=images, is_benign=is_benign, dataset=dataset, split=split))
    if not out:
        raise ValueError(f"rcs: no training examples loaded from: {path}")
    return out


def _validate_paper_training_examples(
    examples: list[_RCSTrainExample],
    *,
    source: str,
    require_balanced_train: bool,
) -> None:
    if not examples:
        raise ValueError(f"rcs: no paper training examples loaded from: {source}")
    benign_count = sum(1 for ex in examples if ex.is_benign)
    malicious_count = len(examples) - benign_count
    if benign_count <= 0 or malicious_count <= 0:
        raise ValueError(
            "rcs: paper training examples require at least one benign and one malicious example "
            f"(source={source}, benign={benign_count}, malicious={malicious_count})"
        )
    if require_balanced_train and benign_count != malicious_count:
        raise ValueError(
            "rcs: require_balanced_train=true expects equal benign and malicious training examples "
            f"(source={source}, benign={benign_count}, malicious={malicious_count})"
        )


def _paper_training_example_summary(
    *,
    examples: list[_RCSTrainExample],
    source: str,
    source_type: str,
    require_balanced_train: bool,
    prototype_image_count: int,
    prototype_image_path: str | None,
) -> dict[str, Any]:
    benign_count = sum(1 for ex in examples if ex.is_benign)
    malicious_count = len(examples) - benign_count
    by_dataset: dict[str, dict[str, int | str]] = {}
    by_split: dict[str, dict[str, int | str]] = {}
    for ex in examples:
        row = by_dataset.setdefault(
            ex.dataset,
            {"dataset": ex.dataset, "count": 0, "benign_count": 0, "malicious_count": 0},
        )
        row["count"] = int(row["count"]) + 1
        key = "benign_count" if ex.is_benign else "malicious_count"
        row[key] = int(row[key]) + 1
        split_name = ex.split or "unspecified"
        split_row = by_split.setdefault(
            split_name,
            {"split": split_name, "count": 0, "benign_count": 0, "malicious_count": 0},
        )
        split_row["count"] = int(split_row["count"]) + 1
        split_row[key] = int(split_row[key]) + 1

    return {
        "source": source,
        "source_type": source_type,
        "schema": dict(_RCS_TRAIN_JSONL_SCHEMA),
        "count": len(examples),
        "benign_count": benign_count,
        "malicious_count": malicious_count,
        "balanced": benign_count == malicious_count,
        "require_balanced_train": bool(require_balanced_train),
        "dataset_counts": [by_dataset[name] for name in sorted(by_dataset)],
        "split_counts": [by_split[name] for name in sorted(by_split)],
        "prototype_image_count": prototype_image_count,
        "prototype_image_path": prototype_image_path,
    }


def _paper_proto_images(*, prototype_image_count: int, prototype_image_path: str | None) -> tuple[ImageInput, ...]:
    if int(prototype_image_count) <= 0:
        return ()
    if prototype_image_path is None:
        return tuple(ImageInput(path=f"__proto_image_{i}__") for i in range(int(prototype_image_count)))
    return tuple(ImageInput(path=str(prototype_image_path)) for _ in range(int(prototype_image_count)))


def _sample_from_train_example(ex: _RCSTrainExample, *, index: int) -> Sample:
    return Sample(
        sample_id=f"rcs-train-{index}",
        behavior_id=ex.dataset,
        is_benign=ex.is_benign,
        prompt=ex.prompt,
        images=ex.images,
    )


def _split_paper_examples(
    examples: list[_RCSTrainExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[_RCSTrainExample], list[_RCSTrainExample]]:
    explicit = [ex for ex in examples if ex.split is not None]
    if explicit:
        if len(explicit) != len(examples):
            raise ValueError("rcs: explicit train_jsonl split requires every example to include split")
        train = [ex for ex in examples if ex.split == "train"]
        val = [ex for ex in examples if ex.split == "validation"]
        if not train:
            raise ValueError("rcs: explicit train_jsonl split produced empty train split")
        if val:
            val_labels = {bool(ex.is_benign) for ex in val}
            if len(val_labels) < 2:
                raise ValueError("rcs: explicit train_jsonl validation split requires both benign and malicious examples")
        return train, val

    rng = random.Random(int(seed))
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_val = int(round(float(val_ratio) * float(len(examples))))
    n_val = max(0, min(n_val, len(examples) - 1))
    val_idx = set(order[:n_val])
    train = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val = [ex for i, ex in enumerate(examples) if i in val_idx]
    return train, val


def _paper_examples_from_params(
    *,
    train_jsonl: str | None,
    require_balanced_train: bool,
    prototype_image_count: int,
    prototype_image_path: str | None,
    benign_prompts: list[str],
    malicious_prompts: list[str],
) -> list[_RCSTrainExample]:
    if train_jsonl is not None:
        examples = _load_train_examples_jsonl(train_jsonl)
        _validate_paper_training_examples(
            examples,
            source=train_jsonl,
            require_balanced_train=bool(require_balanced_train),
        )
        return examples

    proto_images = _paper_proto_images(
        prototype_image_count=prototype_image_count,
        prototype_image_path=prototype_image_path,
    )
    examples = [
        _RCSTrainExample(prompt=p, images=proto_images, is_benign=True, dataset="benign_proto")
        for p in benign_prompts
    ] + [
        _RCSTrainExample(prompt=p, images=proto_images, is_benign=False, dataset="malicious_proto")
        for p in malicious_prompts
    ]
    _validate_paper_training_examples(
        examples,
        source="inline_prototypes",
        require_balanced_train=bool(require_balanced_train),
    )
    return examples
