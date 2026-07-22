from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import yaml


RUN_FILE_REFERENCE = re.compile(
    r"\b(?:(?:tests|scripts)/[\w./-]+\.py|configs/[\w./-]+\.yaml)\b"
)


def _iter_run_commands(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "run" and isinstance(child, str):
                yield child
            else:
                yield from _iter_run_commands(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_run_commands(child)


def test_ci_workflow_declares_sglang_lite_stages() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))

    assert set(workflow["jobs"]) >= {
        "lint",
        "unit-fast",
        "architecture-contracts",
        "trace-smoke-audit",
        "matrix-plan",
        "openai-compat-fake",
        "targeted-smokes",
        "nightly-hf",
    }
    assert "scripts/ci/run_trace_smoke.py" in str(workflow)
    assert "scripts/ci/check_matrix_plan.py" in str(workflow)
    assert "scripts/ci/fake_openai_compat_server.py" in str(workflow)


def test_architecture_contracts_run_ci_architecture_tests() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    arch_steps = workflow["jobs"]["architecture-contracts"]["steps"]
    contract_step = next(step for step in arch_steps if step.get("name") == "Detector and provider contracts")

    assert "tests/ci/test_conformance_architecture.py" in contract_step["run"]


def test_architecture_contracts_use_minimal_runtime_contract_tests() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    arch_steps = workflow["jobs"]["architecture-contracts"]["steps"]
    contract_step = next(step for step in arch_steps if step.get("name") == "Detector and provider contracts")

    assert "tests/test_minimal_runtime_contracts.py" in contract_step["run"]
    assert "tests/test_core_run_audit.py" in contract_step["run"]


def test_workflow_run_file_references_exist() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    referenced_paths = {
        Path(match)
        for command in _iter_run_commands(workflow)
        for match in RUN_FILE_REFERENCE.findall(command)
    }

    assert referenced_paths
    assert sorted(str(path) for path in referenced_paths if not path.is_file()) == []


def test_architecture_contracts_gate_trace_runtime_contracts() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    arch_steps = workflow["jobs"]["architecture-contracts"]["steps"]
    contract_step = next(step for step in arch_steps if step.get("name") == "Detector and provider contracts")

    assert "tests/test_minimal_runtime_contracts.py" in contract_step["run"]
    assert "tests/ci/test_run_trace_smoke.py" in contract_step["run"]


def test_architecture_contracts_gate_cache_manifest_and_calibration_sidecars() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    arch_steps = workflow["jobs"]["architecture-contracts"]["steps"]
    contract_step = next(step for step in arch_steps if step.get("name") == "Detector and provider contracts")

    assert "tests/test_cache_manifest.py" in contract_step["run"]
    assert "tests/test_detector_calibrate_cli.py" in contract_step["run"]


def test_trace_smoke_audit_always_exercises_model_responses_provider() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    trace_matrix = workflow["jobs"]["trace-smoke-audit"]["strategy"]["matrix"]["config"]

    assert "configs/runs/smoke_jailguard.yaml" in trace_matrix


def test_jailguard_path_filter_tracks_component_location_and_paper_config() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jailguard_paths = workflow["jobs"]["changes"]["outputs"]["detector_jailguard"]
    filter_step = next(step for step in workflow["jobs"]["changes"]["steps"] if step.get("id") == "filter")
    filter_paths = filter_step["with"]["filters"]

    assert "steps.filter.outputs.detector_jailguard" in jailguard_paths
    assert "src/turnkey/components/detectors/jailguard.py" in filter_paths
    assert "configs/runs/jailguard_paper_openai.yaml" in filter_paths
    assert "src/turnkey/detectors/jailguard.py" not in filter_paths


def test_change_filters_track_canonical_source_locations() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    filter_step = next(step for step in workflow["jobs"]["changes"]["steps"] if step.get("id") == "filter")
    filters = yaml.safe_load(filter_step["with"]["filters"])

    shared_detector_paths = {
        "src/turnkey/components/detectors/__init__.py",
        "src/turnkey/components/detectors/base.py",
        "src/turnkey/runner/run.py",
    }
    expected_paths = {
        "detector_rcs": {"src/turnkey/components/detectors/rcs/**", *shared_detector_paths},
        "detector_gradsafe": {
            "src/turnkey/components/detectors/gradsafe.py",
            *shared_detector_paths,
        },
        "detector_jailguard": {
            "src/turnkey/components/detectors/jailguard.py",
            *shared_detector_paths,
        },
        "attacks": {"src/turnkey/components/attacks/**"},
        "datasets": {"src/turnkey/components/datasets/**"},
        "judges": {"src/turnkey/components/judges/**"},
        "backends": {
            "src/turnkey/components/backends/**",
            "src/turnkey/runtime_providers/**",
            "src/turnkey/methods.py",
            "src/turnkey/policy.py",
            "src/turnkey/runtime_events.py",
            "src/turnkey/runner/policy_executor.py",
            "src/turnkey/runner/run.py",
        },
        "matrix": {
            "src/turnkey/matrix/**",
            "src/turnkey/analysis/matrix*.py",
        },
    }
    legacy_paths = {
        "src/turnkey/detectors/rcs.py",
        "src/turnkey/detectors/rcs/**",
        "src/turnkey/detectors/gradsafe.py",
        "src/turnkey/detectors/gradsafe/**",
        "src/turnkey/detectors/jailguard.py",
        "src/turnkey/attacks/**",
        "src/turnkey/datasets/**",
        "src/turnkey/judges/**",
        "src/turnkey/backends/**",
        "src/turnkey/providers.py",
        "src/turnkey/providers/**",
        "src/turnkey/matrix.py",
        "src/turnkey/matrix_analysis.py",
        "src/turnkey/matrix_analysis/**",
    }

    for filter_name, paths in expected_paths.items():
        assert paths <= set(filters[filter_name])
    assert legacy_paths.isdisjoint(path for paths in filters.values() for path in paths)


def test_targeted_gradsafe_gate_uses_fake_provider_smoke() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    targeted_steps = workflow["jobs"]["targeted-smokes"]["steps"]
    gradsafe_step = next(step for step in targeted_steps if step.get("name") == "GradSafe smoke")

    assert "tests/test_gradsafe_detector.py::test_gradsafe_policy_run_records_gradient_provider_and_audits" in (
        gradsafe_step["run"]
    )
    assert "configs/runs/smoke_gradsafe.yaml" not in gradsafe_step["run"]


def test_jailguard_smoke_exercises_model_responses_provider() -> None:
    smoke = yaml.safe_load(Path("configs/runs/smoke_jailguard.yaml").read_text(encoding="utf-8"))

    assert smoke["detector"]["name"] == "jailguard_v3"
    assert smoke["detector"]["params"]["response_mode"] == "backend"
    assert smoke["model"]["backend"] == "dummy"
