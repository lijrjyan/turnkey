import json
from pathlib import Path
import textwrap

import pytest
import yaml

from turnkey.audit import audit_run_dir
from turnkey.component_loader import ResolvedComponent
from turnkey.config import (
    Config,
    DatasetConfig,
    DetectorConfig,
    ModelConfig,
    NaturalnessConfig,
    RunConfig,
    load_config,
)
from turnkey.policy import Component, PolicyChain
from turnkey.runner import run_eval


def test_run_eval_uses_external_component_resolver_and_writes_replayable_identity(
    tmp_path: Path,
) -> None:
    method_file = tmp_path / "external_threshold.py"
    method_file.write_text(
        textwrap.dedent(
            """
            from dataclasses import replace

            from turnkey.policy import Component, Outcome


            class ThresholdPolicy:
                def __init__(self, threshold):
                    self.threshold = threshold

                def apply(self, request, call_next, context):
                    score = 1.0 if "UNSAFE_PLACEHOLDER" in request.sample.prompt else 0.0
                    if score >= self.threshold:
                        return Outcome.blocked(request.target, score=score, reason="external")
                    return replace(call_next(request), score=score, reason="external")


            def build(*, threshold):
                return Component(
                    name="external-threshold",
                    policy=ThresholdPolicy(threshold),
                    parameters={"threshold": threshold},
                )
            """
        ).lstrip(),
        encoding="utf-8",
    )
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["detector"] = {
        "name": f"{method_file}:build",
        "params": {"threshold": 0.5},
    }
    config_path = tmp_path / "external.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(config_path), source_config_path=str(config_path))

    assert audit_run_dir(run_dir) == []
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    effective = run["config"]
    detector = effective["detector"]
    assert detector["name"] == f"{method_file}:build"
    assert detector["resolved_name"] == "external-threshold"
    assert detector["component_source"]["path"] == str(method_file)
    assert detector["component_source"]["object"] == "build"
    component = run["components"]["intervention"]
    assert component["name"] == "external-threshold"
    assert component["parameters"] == {"threshold": 0.5}
    assert component["source"] == detector["component_source"]


def test_detector_audit_rejects_component_source_drift(tmp_path: Path) -> None:
    method_file = tmp_path / "external_allow.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "component = Component(name='external-allow', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["detector"] = {"name": f"{method_file}:component", "params": {}}
    config_path = tmp_path / "external.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    run_dir = run_eval(load_config(config_path), source_config_path=str(config_path))
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["components"]["intervention"]["source"]["sha256"] = "0" * 64
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    errors = audit_run_dir(run_dir)

    assert any("components.intervention.source" in error for error in errors)


def test_audit_does_not_depend_on_current_external_source_contents(tmp_path: Path) -> None:
    method_file = tmp_path / "external_allow.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "component = Component(name='external-allow', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["detector"] = {"name": f"{method_file}:component", "params": {}}
    config_path = tmp_path / "external.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    run_dir = run_eval(load_config(config_path), source_config_path=str(config_path))

    method_file.write_text("# source changed after the run\n", encoding="utf-8")

    assert audit_run_dir(run_dir) == []


def test_run_eval_cleans_reference_when_intervention_resolution_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import turnkey.runner.run as run_module

    events: list[str] = []
    calls = 0

    def resolve(reference, **kwargs):  # noqa: ANN001, ARG001
        nonlocal calls
        calls += 1
        if calls == 1:
            return ResolvedComponent(
                component=Component(
                    name="reference",
                    policy=PolicyChain(),
                    cleanup=lambda: events.append("reference:cleanup"),
                ),
                source={"kind": "test"},
            )
        raise RuntimeError("intervention resolution failed")

    monkeypatch.setattr(run_module, "resolve_component", resolve)
    cfg = Config(
        run=RunConfig(out_dir=str(tmp_path / "outputs"), max_samples=1),
        dataset=DatasetConfig(name="fixtures_smoke@v1", params={"n_samples": 1}),
        model=ModelConfig(backend="dummy", model_id="dummy"),
        naturalness=NaturalnessConfig(enabled=False),
    )

    with pytest.raises(RuntimeError, match="intervention resolution failed"):
        run_eval(cfg)

    assert events == ["reference:cleanup"]


def test_run_eval_preserves_resolution_error_when_reference_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import turnkey.runner.run as run_module

    calls = 0

    def fail_cleanup() -> None:
        raise OSError("reference cleanup failed")

    def resolve(reference, **kwargs):  # noqa: ANN001, ARG001
        nonlocal calls
        calls += 1
        if calls == 1:
            return ResolvedComponent(
                component=Component(
                    name="reference",
                    policy=PolicyChain(),
                    cleanup=fail_cleanup,
                ),
                source={"kind": "test"},
            )
        raise ValueError("intervention resolution failed")

    monkeypatch.setattr(run_module, "resolve_component", resolve)
    cfg = Config(
        run=RunConfig(out_dir=str(tmp_path / "outputs"), max_samples=1),
        dataset=DatasetConfig(name="fixtures_smoke@v1", params={"n_samples": 1}),
        model=ModelConfig(backend="dummy", model_id="dummy"),
        naturalness=NaturalnessConfig(enabled=False),
    )

    with pytest.raises(ValueError, match="intervention resolution failed") as error:
        run_eval(cfg)

    assert error.value.__turnkey_secondary_failures__[0][0] == "component cleanup"


def test_run_eval_rejects_external_calibration_before_loading_artifact(tmp_path: Path) -> None:
    cfg = Config(
        run=RunConfig(out_dir=str(tmp_path / "outputs"), max_samples=1),
        detector=DetectorConfig(
            name=f"{tmp_path / 'missing_method.py'}:build",
            calibration_artifact=str(tmp_path / "missing_calibration.json"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="external components do not support calibration artifacts",
    ):
        run_eval(cfg)

    assert not (tmp_path / "outputs").exists()
