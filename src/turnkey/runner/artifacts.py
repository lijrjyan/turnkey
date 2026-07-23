from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from turnkey._internal.redact import redact_text, sha256_hex
from turnkey.config import Config, dataset_identity
from turnkey.inputs.provider import InputProviderBundle
from turnkey.policy import Component
from turnkey.runtime_events import RuntimeEvent
from turnkey.runtime_providers import ProviderSummary
from turnkey.schema import DetectorDecision, JudgeOutput, ModelOutput, Record, Sample
from turnkey.signals import SignalBundle
from turnkey.threat import budget_for_sample, threat_context_for_sample

from .io import file_identity, read_git_revision


RUN_SCHEMA = "turnkey_run/v1"
CASE_SCHEMA = "turnkey_case/v1"
PRIVATE_CASE_SCHEMA = "turnkey_private_case/v1"


def records_to_case_rows(
    *,
    cfg: Config,
    reference_records: list[Record],
    intervention_records: list[Record],
    sample_order: list[str],
) -> list[dict[str, Any]]:
    reference_by_id = {record.sample_id: record for record in reference_records}
    intervention_by_id = {record.sample_id: record for record in intervention_records}
    dataset = dataset_identity(cfg.dataset.name).to_metadata()
    rows: list[dict[str, Any]] = []
    for case_id in sample_order:
        reference = reference_by_id.get(case_id)
        intervention = intervention_by_id.get(case_id)
        if reference is None or intervention is None:
            continue
        rows.append(
            {
                "schema_version": CASE_SCHEMA,
                "case_id": case_id,
                "behavior_id": reference.behavior_id,
                "is_benign": reference.is_benign,
                "dataset": dataset,
                "attack_family": reference.attack_family,
                "attack_method": reference.attack_method,
                "attack_params": dict(reference.attack_params),
                "threat": asdict(reference.threat),
                "budget": asdict(reference.budget),
                "prompt": {
                    "sha256": reference.prompt_hash,
                    "chars": reference.prompt_chars,
                },
                "reference": _public_outcome(reference),
                "intervention": _public_outcome(intervention),
            }
        )
    return rows


def records_to_private_rows(
    *,
    samples: list[Sample],
    reference_records: list[Record],
    intervention_records: list[Record],
) -> list[dict[str, Any]]:
    reference_by_id = {record.sample_id: record for record in reference_records}
    intervention_by_id = {record.sample_id: record for record in intervention_records}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        reference = reference_by_id.get(sample.sample_id)
        intervention = intervention_by_id.get(sample.sample_id)
        if reference is None or intervention is None:
            continue
        rows.append(
            {
                "schema_version": PRIVATE_CASE_SCHEMA,
                "case_id": sample.sample_id,
                "prompt": sample.prompt,
                "reference": {
                    "model": asdict(reference.model),
                    "detector_diagnostics": dict(reference.detector.diagnostics),
                    "judge_details": dict(reference.judge.details),
                },
                "intervention": {
                    "model": asdict(intervention.model),
                    "detector_diagnostics": dict(intervention.detector.diagnostics),
                    "judge_details": dict(intervention.judge.details),
                },
            }
        )
    return rows


def event_rows(events: tuple[RuntimeEvent, ...]) -> list[dict[str, Any]]:
    return [event.to_dict() for event in events]


def build_run_metadata(
    *,
    cfg: Config,
    effective_config: dict[str, Any],
    run_dir: Path,
    source_config_path: str | None,
    command: list[str] | None,
    input_bundle: InputProviderBundle,
    reference_component: Component,
    intervention_component: Component,
    reference_source: Mapping[str, Any],
    intervention_source: Mapping[str, Any],
    runtime_cache: dict[str, Any],
    resolved_model_revision: str | None,
    calibration_artifacts: dict[str, dict[str, Any]],
    artifact_identities: dict[str, dict[str, Any]],
    private_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    source_config_identity: dict[str, Any] | None = None
    if source_config_path is not None:
        source_path = Path(source_config_path)
        source_config_identity = {"path": source_config_path}
        if source_path.is_file():
            source_config_identity.update(file_identity(source_path))

    dataset_revision = cfg.dataset.params.get("revision")
    dataset = {
        "params": dict(cfg.dataset.params),
        "requested_revision": dataset_revision,
        "resolved_revision": input_bundle.resolved_dataset_revision,
        **dataset_identity(cfg.dataset.name).to_metadata(),
    }
    input_value: dict[str, Any] = {
        "mode": input_bundle.mode,
        "case_ids": list(input_bundle.manifest.get("sample_ids", [])),
        "manifest": input_bundle.manifest,
    }
    if input_bundle.expected_identity is not None:
        input_value["expected_manifest"] = dict(input_bundle.expected_identity)

    redaction: dict[str, Any] = {
        "public_redacted": True,
        "private_enabled": private_identity is not None,
    }
    if private_identity is not None:
        redaction["private_artifact"] = private_identity

    return {
        "schema_version": RUN_SCHEMA,
        "status": "complete",
        "run_id": run_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": read_git_revision(),
        "command": {
            "argv": list(command) if command is not None else None,
            "source_config": source_config_identity,
        },
        "config": effective_config,
        "model": {
            "backend": cfg.model.backend,
            "model_id": cfg.model.model_id,
            "requested_revision": cfg.model.revision,
            "resolved_revision": resolved_model_revision,
            "device": cfg.model.device,
            "trust_remote_code": cfg.model.trust_remote_code,
        },
        "components": {
            "dataset": dataset,
            "content": input_bundle.content_report.to_dict(),
            "attack": asdict(cfg.attack),
            "reference": _component_summary(reference_component, source=reference_source),
            "intervention": _component_summary(
                intervention_component,
                source=intervention_source,
            ),
            "judge": asdict(cfg.judge),
        },
        "reproduction": asdict(cfg.reproduction),
        "input": input_value,
        "calibration_artifacts": calibration_artifacts,
        "runtime_cache": runtime_cache,
        "redaction": redaction,
        "artifacts": artifact_identities,
    }


def artifact_identity(path: Path, *, run_dir: Path, count: int) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "count": count,
    }


def record_for_sample(
    *,
    sample: Sample,
    detector_decision: DetectorDecision,
    model_out: ModelOutput,
    judge_out: JudgeOutput,
    signal_summary: dict[str, Any] | None = None,
) -> Record:
    extra: dict[str, Any] = {}
    if signal_summary is not None:
        extra["signals"] = signal_summary
    return Record(
        sample_id=sample.sample_id,
        behavior_id=sample.behavior_id,
        is_benign=sample.is_benign,
        attack_family=sample.attack_family,
        attack_method=sample.attack_method,
        attack_params=dict(sample.attack_params),
        threat=threat_context_for_sample(sample),
        budget=budget_for_sample(sample),
        prompt_hash=sha256_hex(sample.prompt),
        prompt_chars=len(sample.prompt),
        detector=detector_decision,
        model=model_out,
        judge=judge_out,
        extra=extra,
    )


def model_output_signals(*, cfg: Config, signals: SignalBundle):
    prompt_logprobs = signals.prompt_logprobs if cfg.model.return_prompt_logprobs else None
    prefix_logprobs = signals.prefix_logprobs if cfg.model.prefix_logprob_text is not None else None
    return prompt_logprobs, prefix_logprobs


def signal_summary_with_providers(
    signals: SignalBundle,
    providers: list[ProviderSummary],
) -> dict[str, Any]:
    summary = signals.summary()
    summary["providers"] = [provider.to_dict() for provider in providers]
    return summary


def _component_summary(
    component: Component,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": component.name,
        "parameters": dict(component.parameters),
        "source": dict(source),
    }


def _public_outcome(record: Record) -> dict[str, Any]:
    model = asdict(record.model)
    model["response_text"] = redact_text(record.model.response_text)
    _redact_logprob_text(model.get("prompt_logprobs"))
    prefix_logprobs = model.get("prefix_logprobs")
    _redact_logprob_text(prefix_logprobs)
    if isinstance(prefix_logprobs, dict):
        prefix_logprobs["prefix_text"] = redact_text(prefix_logprobs.get("prefix_text"))
    return {
        "detector": {
            **asdict(record.detector),
            "diagnostics": _redact_sensitive(record.detector.diagnostics),
        },
        "model": model,
        "judge": {
            **asdict(record.judge),
            "details": _redact_sensitive(record.judge.details),
        },
        "signals": _redact_sensitive(record.extra.get("signals", {})),
    }


def _redact_logprob_text(value: Any) -> None:
    if not isinstance(value, dict):
        return
    tokens = value.get("tokens")
    if not isinstance(tokens, list):
        return
    for token in tokens:
        if isinstance(token, dict) and isinstance(token.get("token"), str):
            token["token"] = redact_text(token["token"])


def _redact_sensitive(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            item_key: _redact_sensitive(item, key=item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive(item, key=key) for item in value]
    lowered = key.lower()
    if isinstance(value, str) and (
        lowered in {"prompt", "response", "text", "content"}
        or lowered.endswith(("_prompt", "_response", "_text", "_content"))
    ):
        return redact_text(value)
    return value
