from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from turnkey._internal.data import sha256_hex


RCS_TEXT_POOL_MANIFEST_SCHEMA = "turnkey_rcs_text_pool_manifest/v1"

_PROMPT_FIELDS = (
    "prompt",
    "txt",
    "text",
    "Goal",
    "goal",
    "instruction",
    "question",
    "content",
    "query",
    "jailbreak_query",
    "harmful_instruction",
)


@dataclass(frozen=True)
class RCSTextSource:
    name: str
    path: Path
    is_benign: bool


@dataclass(frozen=True)
class RCSTextPoolRecord:
    prompt: str
    is_benign: bool
    dataset: str
    split: str
    source_index: int

    def to_jsonl_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "is_benign": self.is_benign,
            "dataset": self.dataset,
            "split": self.split,
            "source_index": self.source_index,
            "prompt_sha256": sha256_hex(self.prompt),
            "prompt_chars": len(self.prompt),
        }


def parse_source_spec(value: str, *, is_benign: bool) -> RCSTextSource:
    name, sep, path = value.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise ValueError("source spec must be NAME=PATH")
    return RCSTextSource(name=name.strip(), path=Path(path.strip()), is_benign=bool(is_benign))


def build_rcs_text_pool(
    *,
    benign_sources: list[RCSTextSource],
    malicious_sources: list[RCSTextSource],
    out_jsonl: Path,
    manifest_path: Path,
    seed: int = 0,
    val_ratio: float = 0.2,
    max_per_source: int | None = None,
    balance_classes: bool = True,
) -> dict[str, Any]:
    if not benign_sources:
        raise ValueError("rcs text pool requires at least one benign source")
    if not malicious_sources:
        raise ValueError("rcs text pool requires at least one malicious source")
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be >= 0 and < 1")
    if max_per_source is not None and int(max_per_source) <= 0:
        raise ValueError("max_per_source must be positive when provided")

    source_records: list[tuple[RCSTextSource, list[RCSTextPoolRecord]]] = []
    selected: list[RCSTextPoolRecord] = []
    for source in [*benign_sources, *malicious_sources]:
        records = _records_from_source(source, seed=seed, max_per_source=max_per_source)
        source_records.append((source, records))
        selected.extend(records)

    if balance_classes:
        selected = _balance_records(selected, seed=seed)
    selected = _assign_splits(selected, seed=seed, val_ratio=val_ratio)
    selected = sorted(selected, key=lambda r: (r.dataset, r.split, r.source_index, r.prompt))

    benign_count = sum(1 for r in selected if r.is_benign)
    malicious_count = len(selected) - benign_count
    if benign_count <= 0 or malicious_count <= 0:
        raise ValueError("rcs text pool must include both benign and malicious records")
    if balance_classes and benign_count != malicious_count:
        raise ValueError("rcs text pool balancing failed to produce equal benign/malicious counts")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "\n".join(json.dumps(r.to_jsonl_record(), sort_keys=True, ensure_ascii=False) for r in selected) + "\n",
        encoding="utf-8",
    )

    manifest = _build_manifest(
        selected=selected,
        sources=[source for source, _records in source_records],
        out_jsonl=out_jsonl,
        seed=seed,
        val_ratio=val_ratio,
        max_per_source=max_per_source,
        balance_classes=balance_classes,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _records_from_source(
    source: RCSTextSource,
    *,
    seed: int,
    max_per_source: int | None,
) -> list[RCSTextPoolRecord]:
    rows = list(_load_rows(source.path))
    records: list[RCSTextPoolRecord] = []
    for index, row in enumerate(rows):
        prompt = _extract_prompt(row)
        if prompt is None:
            continue
        records.append(
            RCSTextPoolRecord(
                prompt=prompt,
                is_benign=source.is_benign,
                dataset=source.name,
                split="train",
                source_index=index,
            )
        )
    if not records:
        raise ValueError(f"{source.name}: no usable prompt records loaded from {source.path}")
    rng = random.Random(_source_seed(seed=seed, name=source.name, path=source.path))
    rng.shuffle(records)
    if max_per_source is not None:
        records = records[: int(max_per_source)]
    return records


def _load_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield obj
        return
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(obj, dict):
            values = obj.values()
            for item in values:
                if isinstance(item, dict):
                    yield item
            return
        raise ValueError(f"{path}: expected JSON array or object")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
        return
    raise ValueError(f"{path}: unsupported source format; expected .jsonl, .json, or .csv")


def _extract_prompt(row: dict[str, Any]) -> str | None:
    instruction = row.get("instruction")
    input_text = row.get("input")
    if isinstance(instruction, str) and instruction.strip():
        parts = [instruction.strip()]
        if isinstance(input_text, str) and input_text.strip():
            parts.append(input_text.strip())
        return "\n".join(parts)

    for field_name in _PROMPT_FIELDS:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            prompt = "\n".join(item.strip() for item in value if item.strip())
            if prompt:
                return prompt

    messages = row.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
        if parts:
            return "\n".join(parts)
    return None


def _balance_records(records: list[RCSTextPoolRecord], *, seed: int) -> list[RCSTextPoolRecord]:
    benign = [r for r in records if r.is_benign]
    malicious = [r for r in records if not r.is_benign]
    if not benign or not malicious:
        return records
    n = min(len(benign), len(malicious))
    rng = random.Random(_source_seed(seed=seed, name="class-balance", path=Path("")))
    rng.shuffle(benign)
    rng.shuffle(malicious)
    return benign[:n] + malicious[:n]


def _assign_splits(
    records: list[RCSTextPoolRecord],
    *,
    seed: int,
    val_ratio: float,
) -> list[RCSTextPoolRecord]:
    grouped: dict[tuple[str, bool], list[RCSTextPoolRecord]] = {}
    for record in records:
        grouped.setdefault((record.dataset, record.is_benign), []).append(record)

    out: list[RCSTextPoolRecord] = []
    for (dataset, is_benign), group in grouped.items():
        rng = random.Random(_source_seed(seed=seed, name=f"{dataset}:{is_benign}:split", path=Path("")))
        group = list(group)
        rng.shuffle(group)
        n_val = int(round(float(val_ratio) * float(len(group))))
        if val_ratio > 0 and len(group) > 1:
            n_val = max(1, n_val)
        n_val = min(n_val, max(0, len(group) - 1))
        val_ids = {id(record) for record in group[:n_val]}
        for record in group:
            out.append(
                RCSTextPoolRecord(
                    prompt=record.prompt,
                    is_benign=record.is_benign,
                    dataset=record.dataset,
                    split="validation" if id(record) in val_ids else "train",
                    source_index=record.source_index,
                )
            )
    return out


def _build_manifest(
    *,
    selected: list[RCSTextPoolRecord],
    sources: list[RCSTextSource],
    out_jsonl: Path,
    seed: int,
    val_ratio: float,
    max_per_source: int | None,
    balance_classes: bool,
) -> dict[str, Any]:
    return {
        "schema_version": RCS_TEXT_POOL_MANIFEST_SCHEMA,
        "rcs_train_schema_version": "rcs_train_jsonl/v1",
        "reproduction_scope": "rcs_text_only_paper_style_adaptation",
        "paper_strict": False,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "max_per_source": max_per_source,
        "balance_classes": bool(balance_classes),
        "output": _file_identity(out_jsonl),
        "counts": _counts(selected),
        "sources": [
            {
                "name": source.name,
                "path": str(source.path),
                "is_benign": source.is_benign,
                "identity": _file_identity(source.path),
            }
            for source in sources
        ],
        "source_counts": _source_counts(selected),
        "split_counts": _split_counts(selected),
        "prompt_hashes": [
            {
                "dataset": record.dataset,
                "split": record.split,
                "is_benign": record.is_benign,
                "source_index": record.source_index,
                "sha256": sha256_hex(record.prompt),
                "chars": len(record.prompt),
            }
            for record in selected
        ],
        "method_notes": {
            "reference_chain": "RCS paper-style hidden-state/projection/KCD-MCD/threshold calibration chain",
            "threshold_split": "explicit train/validation split encoded in the generated rcs_train_jsonl/v1 records",
            "deviation": "Text-only pool excludes LVLM image sources and must not be reported as paper-strict LVLM reproduction.",
        },
    }


def _counts(records: list[RCSTextPoolRecord]) -> dict[str, int]:
    benign = sum(1 for record in records if record.is_benign)
    malicious = len(records) - benign
    return {"count": len(records), "benign_count": benign, "malicious_count": malicious}


def _source_counts(records: list[RCSTextPoolRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RCSTextPoolRecord]] = {}
    for record in records:
        grouped.setdefault(record.dataset, []).append(record)
    out = []
    for name in sorted(grouped):
        rows = grouped[name]
        counts = _counts(rows)
        out.append({"dataset": name, **counts})
    return out


def _split_counts(records: list[RCSTextPoolRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RCSTextPoolRecord]] = {}
    for record in records:
        grouped.setdefault(record.split, []).append(record)
    out = []
    for name in sorted(grouped):
        rows = grouped[name]
        counts = _counts(rows)
        out.append({"split": name, **counts})
    return out


def _file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _source_seed(*, seed: int, name: str, path: Path) -> int:
    data = f"{int(seed)}\n{name}\n{path}".encode("utf-8")
    return int(hashlib.sha256(data).hexdigest()[:16], 16)
