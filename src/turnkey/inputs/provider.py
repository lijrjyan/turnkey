from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from turnkey.components.attacks import load_attack
from turnkey.components.datasets import load_dataset_with_metadata
from turnkey.config import Config
from turnkey.content import ContentSelectionReport, select_content
from turnkey.boreiko_ngram_resource import BoreikoNgramScorer
from turnkey.inputs.manifest import (
    build_input_manifest,
    compare_input_manifests,
    input_manifest_file_identity,
    load_input_manifest,
)
from turnkey.naturalness import annotate_samples_with_ngram_ppl, load_prompt_corpus
from turnkey.schema import Sample


InputProviderMode = Literal["generated", "verified"]


@dataclass(frozen=True)
class InputProviderBundle:
    selected_samples: list[Sample]
    attacked_samples: list[Sample]
    content_report: ContentSelectionReport
    manifest: dict[str, Any]
    mode: InputProviderMode
    resolved_dataset_revision: str | None = None
    expected_manifest: dict[str, Any] | None = None
    expected_identity: dict[str, Any] | None = None


def materialize_input_provider(cfg: Config) -> InputProviderBundle:
    loaded_dataset = load_dataset_with_metadata(cfg.dataset, _warning_stacklevel=3)
    samples = loaded_dataset.samples
    selected_samples, content_report = select_content(samples, cfg.content)
    run_samples = (
        selected_samples[: cfg.run.max_samples] if cfg.run.max_samples else selected_samples
    )
    attack = load_attack(cfg.attack)
    attacked_samples = [attack.apply(sample) for sample in run_samples]
    if cfg.naturalness.enabled:
        scorer = None
        if cfg.naturalness.backend == "boreiko_ngram":
            if cfg.naturalness.boreiko_manifest_path is None:
                raise ValueError("naturalness.backend=boreiko_ngram requires boreiko_manifest_path")
            scorer = BoreikoNgramScorer.from_manifest(
                cfg.naturalness.boreiko_manifest_path,
                base_path=cfg.naturalness.boreiko_base_path,
                tokenizer_id=cfg.naturalness.boreiko_tokenizer_id,
                window_size=cfg.naturalness.boreiko_window_size,
            )
        elif cfg.naturalness.backend != "local_ngram":
            raise ValueError(f"unknown naturalness.backend: {cfg.naturalness.backend}")
        reference_prompts = (
            load_prompt_corpus(cfg.naturalness.reference_path)
            if cfg.naturalness.reference_path is not None and scorer is None
            else None
        )
        bucket_reference_prompts = (
            load_prompt_corpus(cfg.naturalness.bucket_reference_path)
            if cfg.naturalness.bucket_reference_path is not None
            else None
        )
        reference_id = (
            cfg.naturalness.reference_id or cfg.naturalness.reference_path or "run_benign"
        )
        attacked_samples = annotate_samples_with_ngram_ppl(
            attacked_samples,
            reference_prompts=reference_prompts,
            bucket_reference_prompts=bucket_reference_prompts,
            reference_id=reference_id,
            scorer=scorer,
        )

    manifest = build_input_manifest(
        cfg=cfg,
        content_report=content_report,
        selected_samples=run_samples,
        attacked_samples=attacked_samples,
        include_plaintext=False,
    )

    if cfg.run.input_manifest_path is None:
        return InputProviderBundle(
            selected_samples=run_samples,
            attacked_samples=attacked_samples,
            content_report=content_report,
            manifest=manifest,
            mode="generated",
            resolved_dataset_revision=loaded_dataset.resolved_revision,
        )

    expected_manifest = load_input_manifest(cfg.run.input_manifest_path)
    mismatches = compare_input_manifests(expected=expected_manifest, actual=manifest)
    if mismatches:
        raise ValueError("input manifest verification failed: " + "; ".join(mismatches[:5]))

    return InputProviderBundle(
        selected_samples=run_samples,
        attacked_samples=attacked_samples,
        content_report=content_report,
        manifest=manifest,
        mode="verified",
        resolved_dataset_revision=loaded_dataset.resolved_revision,
        expected_manifest=expected_manifest,
        expected_identity=input_manifest_file_identity(Path(cfg.run.input_manifest_path)),
    )
