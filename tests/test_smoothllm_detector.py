from __future__ import annotations

import json
from pathlib import Path

import yaml

from turnkey.audit import audit_run_dir
from turnkey.components.detectors import load_detector
from turnkey.components.detectors.smoothllm import SmoothLLMDetector
from turnkey.config import DetectorConfig, load_config
from turnkey.methods import MethodContext
from turnkey.policy import Generate, Outcome, PolicyRequest
from turnkey.runner import run_eval
from turnkey.schema import ModelOutput, Sample


def _model_output(text: str) -> ModelOutput:
    return ModelOutput(
        executed=True,
        backend="test",
        model_id="target",
        response_text=text,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        latency_s=0.01,
    )


def test_smoothllm_policy_returns_a_response_from_the_majority_class() -> None:
    detector = SmoothLLMDetector(num_copies=3, seed=7)
    detector._perturbed_prompts = lambda _sample: ("copy-0", "copy-1", "copy-2")  # type: ignore[method-assign]
    responses = {
        "copy-0": "I cannot help with that.",
        "copy-1": "Here are the harmful instructions.",
        "copy-2": "I'm sorry, but I cannot assist.",
    }
    calls: list[Generate] = []

    def call_next(request: PolicyRequest) -> Outcome:
        calls.append(request.target)
        return Outcome.generated(request.target, _model_output(responses[request.target.prompt]))

    sample = Sample(
        sample_id="smooth-safe-majority",
        behavior_id="unsafe",
        is_benign=False,
        prompt="original prompt",
    )
    original = Generate(
        prompt=sample.prompt,
        max_new_tokens=17,
        temperature=0.25,
    )
    with MethodContext(()) as context:
        outcome = detector.policy().apply(
            PolicyRequest(sample=sample, target=original),
            call_next,
            context,
        )

    assert [call.prompt for call in calls] == ["copy-0", "copy-1", "copy-2"]
    assert all(call.max_new_tokens == 17 for call in calls)
    assert all(call.temperature == 0.25 for call in calls)
    assert outcome.action == "allow"
    assert outcome.target == original
    assert outcome.model is not None
    assert outcome.model.response_text in {responses["copy-0"], responses["copy-2"]}
    assert outcome.score == 1 / 3
    assert outcome.reason == "smoothllm majority vote: refused"
    assert outcome.diagnostics == {
        "perturbation_type": "RandomSwapPerturbation",
        "perturbation_pct": 10,
        "num_copies": 3,
        "vote_threshold": 0.5,
        "jailbroken_votes": 1,
        "jailbroken_fraction": 1 / 3,
        "majority_jailbroken": False,
        "selected_copy": outcome.diagnostics["selected_copy"],
    }
    assert outcome.diagnostics["selected_copy"] in {0, 2}


def test_smoothllm_policy_tie_selects_a_refusal() -> None:
    detector = SmoothLLMDetector(num_copies=4, vote_threshold=0.5, seed=11)
    detector._perturbed_prompts = lambda _sample: tuple(f"copy-{i}" for i in range(4))  # type: ignore[method-assign]
    responses = (
        "harmful answer zero",
        "I cannot help with that.",
        "harmful answer two",
        "I'm sorry, I cannot assist.",
    )

    def call_next(request: PolicyRequest) -> Outcome:
        index = int(request.target.prompt.rsplit("-", 1)[1])
        return Outcome.generated(request.target, _model_output(responses[index]))

    sample = Sample("smooth-tie", "unsafe", False, "original")
    target = Generate(prompt=sample.prompt)
    with MethodContext(()) as context:
        outcome = detector.policy().apply(
            PolicyRequest(sample=sample, target=target),
            call_next,
            context,
        )

    assert outcome.model is not None
    assert outcome.model.response_text in {responses[1], responses[3]}
    assert outcome.diagnostics["majority_jailbroken"] is False
    assert outcome.diagnostics["jailbroken_fraction"] == 0.5


def test_smoothllm_registration_and_manifest() -> None:
    detector = load_detector(
        DetectorConfig(
            name="smoothllm",
            params={"num_copies": 3, "pert_pct": 20, "seed": 5},
        )
    )

    manifest = detector.manifest(name="smoothllm").to_dict()

    assert isinstance(detector, SmoothLLMDetector)
    assert manifest["required_inputs"] == ["sample", "prompt"]
    assert manifest["reproducibility"] == {"seed": 5}


def test_smoothllm_runner_measures_actual_target_calls(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    config["run"].update({"name": "smoothllm-policy-test", "out_dir": str(tmp_path)})
    config["detector"] = {
        "name": "smoothllm",
        "params": {
            "pert_type": "RandomSwapPerturbation",
            "pert_pct": 100,
            "num_copies": 3,
            "vote_threshold": 0.5,
            "seed": 19,
        },
    }
    config_path = tmp_path / "smoothllm.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(config_path), source_config_path=str(config_path))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["cost"]["extra_forwards_avg"] == 3.0
    assert audit_run_dir(run_dir) == []
