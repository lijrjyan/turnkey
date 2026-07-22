from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import load_config
from turnkey.components.detectors.jailguard import JailGuardDetector
from turnkey.runner import run_eval
from turnkey.schema import Sample


def test_jailguard_detector_echo_mode_runs_and_is_deterministic() -> None:
    sample = Sample(
        sample_id="fx-0001",
        behavior_id="fixtures:benign:greeting",
        is_benign=True,
        prompt="Say hello in one short sentence.",
    )

    det1 = JailGuardDetector(
        n_variants=4,
        mutator="PI",
        char_rate=0.1,
        threshold=2.0,
        similarity="bow",
        response_mode="echo_prompt",
    )
    det2 = JailGuardDetector(
        n_variants=4,
        mutator="PI",
        char_rate=0.1,
        threshold=2.0,
        similarity="bow",
        response_mode="echo_prompt",
    )

    d1 = det1.decide(sample)
    d2 = det2.decide(sample)

    assert d1.block == d2.block
    assert d1.score == d2.score
    assert "response_mode=echo_prompt" in (d1.reason or "")
    assert d1.diagnostics["max_divergence"] == d1.score
    assert d1.diagnostics["response_mode"] == "echo_prompt"


def test_jailguard_reference_refusal_keywords_drive_all_block_fallback() -> None:
    det = JailGuardDetector(
        n_variants=2,
        mutator="PI",
        threshold=10.0,
        similarity="bow",
        response_mode="echo_prompt",
    )

    decision = det._decision_from_responses(  # noqa: SLF001
        variants=["a", "b"],
        responses=["OpenAI cannot help with that.", "This is illegal and unethical."],
    )

    assert decision.block is True
    assert decision.score < 10.0
    assert decision.diagnostics["all_block"] is True
    assert decision.diagnostics["min_refusal_phrases"] >= 1


def test_jailguard_reference_mask_mutators_are_available() -> None:
    sample = Sample(
        sample_id="fx-mask",
        behavior_id="fixtures:benign:question",
        is_benign=True,
        prompt="Explain photosynthesis briefly. Question: what does sunlight do?",
    )

    for mutator in ("RR", "RI", "TR", "TI"):
        det = JailGuardDetector(
            n_variants=2,
            mutator=mutator,
            char_rate=0.05,
            threshold=10.0,
            similarity="bow",
            response_mode="echo_prompt",
            seed=3,
        )
        variants = det._variants(sample)  # noqa: SLF001
        assert len(variants) == 2
        assert all("[Mask]" in variant for variant in variants)


def test_jailguard_accepts_dependency_gated_reference_mutators() -> None:
    for mutator in ("SR", "TL"):
        det = JailGuardDetector(mutator=mutator, response_mode="echo_prompt")
        assert det.mutator == mutator


def test_jailguard_sr_mutator_uses_wordnet_synonyms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("turnkey.components.detectors.jailguard._nltk_stopwords", lambda: {"the"})
    monkeypatch.setattr(
        "turnkey.components.detectors.jailguard._wordnet_synonyms",
        lambda word: ["fast"] if word == "quick" else [],
    )
    sample = Sample(
        sample_id="fx-sr",
        behavior_id="fixtures:benign:sr",
        is_benign=True,
        prompt="the quick guide",
    )
    det = JailGuardDetector(
        n_variants=2,
        mutator="SR",
        synonym_level=3,
        threshold=10.0,
        response_mode="echo_prompt",
        seed=11,
    )

    variants = det._variants(sample)  # noqa: SLF001

    assert variants == ["the fast guide", "the fast guide"]


def test_jailguard_textaugment_punctuation_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAEDA:
        def punct_insertion(self, text: str) -> str:
            return f"{text} !"

    fake_module = types.SimpleNamespace(AEDA=lambda: FakeAEDA())
    monkeypatch.setitem(sys.modules, "textaugment", fake_module)
    sample = Sample(
        sample_id="fx-pi-textaugment",
        behavior_id="fixtures:benign:pi",
        is_benign=True,
        prompt="hello world",
    )
    det = JailGuardDetector(
        n_variants=2,
        mutator="PI",
        punctuation_backend="textaugment",
        threshold=10.0,
        response_mode="echo_prompt",
        seed=5,
    )

    variants = det._variants(sample)  # noqa: SLF001

    assert variants == ["hello world !", "hello world !"]


def test_jailguard_tl_mutator_uses_textaugment_translate(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTranslate:
        def __init__(self, *, src: str, to: str) -> None:
            self.src = src
            self.to = to

        def augment(self, text: str) -> str:
            return f"{self.src}->{self.to}:{text}"

    fake_module = types.SimpleNamespace(Translate=FakeTranslate)
    monkeypatch.setitem(sys.modules, "textaugment", fake_module)
    sample = Sample(
        sample_id="fx-tl",
        behavior_id="fixtures:benign:tl",
        is_benign=True,
        prompt="hello café",
    )
    det = JailGuardDetector(
        n_variants=2,
        mutator="TL",
        translation_target_langs=("fr",),
        threshold=10.0,
        response_mode="echo_prompt",
        seed=5,
    )

    variants = det._variants(sample)  # noqa: SLF001

    assert variants == ["en->fr:hello caf", "en->fr:hello caf"]


def test_jailguard_policy_can_select_dependency_gated_tl(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTranslate:
        def __init__(self, *, src: str, to: str) -> None:
            self.src = src
            self.to = to

        def augment(self, text: str) -> str:
            return f"{self.src}->{self.to}:{text}"

    fake_module = types.SimpleNamespace(Translate=FakeTranslate)
    monkeypatch.setitem(sys.modules, "textaugment", fake_module)
    sample = Sample(
        sample_id="fx-pl-tl",
        behavior_id="fixtures:benign:pl",
        is_benign=True,
        prompt="hello",
    )
    det = JailGuardDetector(
        n_variants=2,
        mutator="PL",
        policy_pool=("PI", "TI", "TL"),
        policy_probs=(0.0, 0.0, 1.0),
        translation_target_langs=("fr",),
        threshold=10.0,
        response_mode="echo_prompt",
        seed=5,
    )

    variants = det._variants(sample)  # noqa: SLF001

    assert variants == ["en->fr:hello", "en->fr:hello"]


def test_jailguard_manifest_records_inputs_and_reproducibility() -> None:
    manifest = JailGuardDetector(
        n_variants=3,
        response_mode="backend",
        mutator="PI",
    ).manifest(name="jailguard")
    data = manifest.to_dict()

    assert data["name"] == "jailguard"
    assert data["required_inputs"] == ["sample", "prompt"]
    assert data["reproducibility"]["response_mode"] == "backend"


def test_jailguard_echo_run_records_policy_component_and_audits(tmp_path: Path) -> None:
    cfg_path = _jailguard_config(
        tmp_path,
        run_name="jailguard-echo",
        detector_params={
            "n_variants": 3,
            "mutator": "PI",
            "char_rate": 0.1,
            "threshold": 2.0,
            "similarity": "bow",
            "response_mode": "echo_prompt",
            "seed": 7,
        },
    )

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))

    assert audit_run_dir(run_dir) == []
    run = _load_run(run_dir)
    component = run["components"]["intervention"]
    assert component["name"] == "jailguard"
    assert component["parameters"]["response_mode"] == "echo_prompt"
    assert component["parameters"]["n_variants"] == 3
    events = _load_events(run_dir)
    assert all("jailguard_echo_prompt" not in event["name"] for event in events)


def test_jailguard_backend_run_uses_policy_target_calls(tmp_path: Path) -> None:
    cfg_path = _jailguard_config(
        tmp_path,
        run_name="jailguard-backend",
        detector_params={
            "n_variants": 2,
            "mutator": "PI",
            "char_rate": 0.1,
            "threshold": 2.0,
            "similarity": "bow",
            "response_mode": "backend",
            "max_new_tokens": 4,
            "seed": 7,
        },
    )

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))

    assert audit_run_dir(run_dir) == []
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cost"]["extra_forwards_avg"] == 1.5

    events = _load_events(run_dir)
    assert all("multi_generate" not in event["name"] for event in events)


def _jailguard_config(tmp_path: Path, *, run_name: str, detector_params: dict) -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["name"] = run_name
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["detector"] = {
        "name": "jailguard",
        "params": detector_params,
    }
    cfg_path = tmp_path / f"{run_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def _load_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _load_events(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
