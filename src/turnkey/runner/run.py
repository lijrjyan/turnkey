from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
import re
from typing import Any

from turnkey.calibration import (
    CalibrationArtifact,
    calibration_artifact_receipt,
    load_calibration_artifact,
    no_calibration_receipt,
    validate_calibration_artifact,
)
from turnkey._internal.data import json_sha256
from turnkey.component_loader import resolve_component, validate_component_calibration
from turnkey.config import Config, DetectorConfig, ModelConfig
from turnkey.inputs.provider import materialize_input_provider
from turnkey.metrics import compute_nsg_metrics
from turnkey.policy import Component

from .artifacts import (
    artifact_identity,
    build_run_metadata,
    event_rows,
    records_to_case_rows,
    records_to_private_rows,
)
from .cache import RunResourceCache
from .io import utc_run_id, write_jsonl
from .policy_executor import (
    PolicyPairResult,
    cleanup_components_after_error,
    run_policy_pair,
)


IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def run_eval(
    cfg: Config,
    *,
    source_config_path: str | None = None,
    command: list[str] | None = None,
    runtime_cache: RunResourceCache | None = None,
) -> Path:
    if runtime_cache is None:
        runtime_cache = RunResourceCache()

    validate_component_calibration(
        cfg.nsg.baseline_detector.name,
        cfg.nsg.baseline_detector.calibration_artifact,
    )
    validate_component_calibration(
        cfg.detector.name,
        cfg.detector.calibration_artifact,
    )

    out_dir = Path(cfg.run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / utc_run_id(cfg.run.name)
    run_dir.mkdir(parents=True, exist_ok=False)

    input_bundle = materialize_input_provider(cfg)
    attacked_samples = input_bundle.attacked_samples

    judge = runtime_cache.get_judge(cfg.judge)
    backend = runtime_cache.get_backend(cfg.model)
    resolved_model_revision = _resolved_backend_revision(
        backend=backend,
        requested=cfg.model.revision,
        require_resolution=_requires_remote_hf_resolution(cfg.model),
    )
    baseline_calibration, baseline_calibration_receipt = _load_detector_calibration(
        detector_cfg=cfg.nsg.baseline_detector,
        cfg=cfg,
    )
    detector_calibration, detector_calibration_receipt = _load_detector_calibration(
        detector_cfg=cfg.detector,
        cfg=cfg,
    )

    baseline_resolution = resolve_component(
        cfg.nsg.baseline_detector.name,
        params=cfg.nsg.baseline_detector.params,
        calibration_artifact=baseline_calibration,
    )
    try:
        detector_resolution = resolve_component(
            cfg.detector.name,
            params=cfg.detector.params,
            calibration_artifact=detector_calibration,
        )
    except BaseException as exc:
        cleanup_components_after_error(exc, baseline_resolution.component)
        raise
    baseline_component = baseline_resolution.component
    detector_component = detector_resolution.component
    try:
        effective_config = _effective_config_for_components(
            cfg=cfg,
            baseline_component=baseline_component,
            detector_component=detector_component,
            baseline_source=baseline_resolution.source,
            detector_source=detector_resolution.source,
        )
        if any(
            bool(getattr(provider, "requires_exclusive_target", False))
            for provider in baseline_component.providers
        ):
            raise RuntimeError(
                "exclusive typed providers are not supported on the reference component"
            )
        requires_target_release = any(
            bool(getattr(provider, "requires_exclusive_target", False))
            for provider in detector_component.providers
        )
        release_target = (
            (lambda: runtime_cache.release_backend(cfg.model))
            if requires_target_release
            else None
        )
    except BaseException as exc:
        cleanup_components_after_error(exc, baseline_component, detector_component)
        raise
    pair_result = run_policy_pair(
        samples=attacked_samples,
        reference=baseline_component,
        intervention=detector_component,
        cfg=cfg,
        backend=backend,
        judge=judge,
        release_target_before_intervention=release_target,
    )
    baseline_records = list(pair_result.reference_records)
    det_records = list(pair_result.intervention_records)
    measured_extra_forwards_avg = _policy_extra_forwards_avg(pair_result)

    sample_order = [sample.sample_id for sample in attacked_samples]
    cases = records_to_case_rows(
        cfg=cfg,
        reference_records=baseline_records,
        intervention_records=det_records,
        sample_order=sample_order,
    )
    events = event_rows(pair_result.events)

    cases_path = run_dir / "cases.jsonl"
    events_path = run_dir / "events.jsonl"
    write_jsonl(cases_path, cases)
    write_jsonl(events_path, events)

    metrics = compute_nsg_metrics(
        baseline=baseline_records,
        with_detector=det_records,
        extra_forwards_avg=measured_extra_forwards_avg,
    )
    metrics = {"schema_version": "turnkey_metrics/v1", **metrics}
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    private_identity = None
    if cfg.run.unsafe_log_plaintext:
        private_dir = run_dir / "private"
        private_dir.mkdir(parents=True, exist_ok=False)
        private_path = private_dir / "cases.jsonl"
        private_rows = records_to_private_rows(
            samples=attacked_samples,
            reference_records=baseline_records,
            intervention_records=det_records,
        )
        write_jsonl(private_path, private_rows)
        private_identity = artifact_identity(
            private_path,
            run_dir=run_dir,
            count=len(private_rows),
        )

    artifact_identities = {
        "cases": artifact_identity(cases_path, run_dir=run_dir, count=len(cases)),
        "events": artifact_identity(events_path, run_dir=run_dir, count=len(events)),
        "metrics": artifact_identity(metrics_path, run_dir=run_dir, count=1),
    }
    run_metadata = build_run_metadata(
        cfg=cfg,
        effective_config=effective_config,
        run_dir=run_dir,
        source_config_path=source_config_path,
        command=command,
        input_bundle=input_bundle,
        reference_component=baseline_component,
        intervention_component=detector_component,
        reference_source=baseline_resolution.source,
        intervention_source=detector_resolution.source,
        runtime_cache=runtime_cache.manifest(),
        resolved_model_revision=resolved_model_revision,
        calibration_artifacts={
            "reference": baseline_calibration_receipt,
            "intervention": detector_calibration_receipt,
        },
        artifact_identities=artifact_identities,
        private_identity=private_identity,
    )
    (run_dir / "run.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return run_dir


def _requires_remote_hf_resolution(model_cfg: ModelConfig) -> bool:
    model_id = model_cfg.model_id
    explicit_local_path = (
        Path(model_id).is_absolute() or model_id.startswith(("./", "../", "~"))
    )
    return bool(
        model_cfg.backend == "hf"
        and not explicit_local_path
        and isinstance(model_cfg.revision, str)
        and IMMUTABLE_REVISION_RE.fullmatch(model_cfg.revision)
    )


def _effective_config_for_components(
    *,
    cfg: Config,
    baseline_component: Component,
    detector_component: Component,
    baseline_source: Mapping[str, Any] | None = None,
    detector_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective = asdict(cfg)
    for configured, component, source in (
        (effective["nsg"]["baseline_detector"], baseline_component, baseline_source),
        (effective["detector"], detector_component, detector_source),
    ):
        parameters = dict(component.parameters)
        raw_params = configured.get("params")
        if isinstance(raw_params, dict):
            configured["params"] = {
                key: value for key, value in raw_params.items() if key in parameters
            }
        configured["component_parameters_sha256"] = json_sha256(parameters)
        if source is not None:
            configured["resolved_name"] = component.name
            configured["component_source"] = dict(source)
    return effective


def _resolved_backend_revision(
    *,
    backend: Any,
    requested: str | None,
    require_resolution: bool,
) -> str | None:
    resolver = getattr(backend, "resolved_revision", None)
    resolved = resolver() if callable(resolver) else None
    if resolved is None:
        if require_resolution:
            raise RuntimeError(
                "remote HF backend did not report a resolved revision for the requested commit"
            )
        return None
    if not isinstance(resolved, str) or IMMUTABLE_REVISION_RE.fullmatch(resolved) is None:
        raise RuntimeError(f"backend resolved_revision must be a full commit SHA, got {resolved!r}")
    if (
        isinstance(requested, str)
        and IMMUTABLE_REVISION_RE.fullmatch(requested)
        and resolved != requested
    ):
        raise RuntimeError(
            f"backend resolved revision {resolved!r} does not match requested revision {requested!r}"
        )
    return resolved


def _policy_extra_forwards_avg(pair_result: PolicyPairResult) -> float:
    n_samples = len(pair_result.reference_records)
    if n_samples == 0:
        return 0.0
    extra_forwards = sum(
        event.model_forwards
        for event in pair_result.events
        if event.pass_name == "intervention" and event.kind == "request"
    )
    return float(extra_forwards / n_samples)


def _load_detector_calibration(
    *,
    detector_cfg: DetectorConfig,
    cfg: Config,
) -> tuple[CalibrationArtifact | None, dict[str, Any]]:
    path = detector_cfg.calibration_artifact
    if path is None:
        return None, no_calibration_receipt()

    artifact = load_calibration_artifact(path)
    validate_calibration_artifact(
        artifact,
        detector_name=detector_cfg.name,
        model_id=cfg.model.model_id,
        revision=cfg.model.revision,
        detector_params=detector_cfg.params,
    )
    return artifact, calibration_artifact_receipt(path, artifact)
