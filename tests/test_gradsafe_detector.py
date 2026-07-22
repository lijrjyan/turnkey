import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from turnkey._internal.hf_deps import optional_hf_token
from turnkey.audit import audit_run_dir
from turnkey.config import DetectorConfig, load_config
from turnkey.components.detectors import available_detectors, load_detector
from turnkey.components.detectors import gradsafe
from turnkey.components.detectors.gradsafe import (
    DEFAULT_MODEL_ID,
    GradSafeDetector,
    _decision_from_score,
)
from turnkey.runtime_providers import (
    GradientScoreConfig,
    GradientScoreResult,
    HFGradientScoreProvider,
    ProviderSummary,
    build_gradient_prompt_texts,
    build_llama2_inst_tokenized_inputs,
    build_qwen_chat_prompt_texts,
    gradsafe_reference_cosine_score,
    load_gradsafe_reference_artifact,
)
from turnkey.runtime_providers import gradient as gradient_provider_module
from turnkey.runtime_providers.gradient import _separator_token_id as provider_separator_token_id
from turnkey.runtime_providers._deps import _import_hf_dependencies
from turnkey.runner import run_eval
from turnkey.schema import Sample


def _sample() -> Sample:
    return Sample(
        sample_id="sample-1",
        behavior_id="behavior-1",
        is_benign=False,
        prompt="UNSAFE_PLACEHOLDER: request redacted.",
        attack_family="T1",
        attack_method="placeholder",
    )


def test_gradsafe_registered_and_loadable_without_model_load() -> None:
    assert "gradsafe_v3" in available_detectors()
    detector = load_detector(DetectorConfig(name="gradsafe_v3", params={"threshold": 0.5}))
    assert isinstance(detector, GradSafeDetector)
    assert detector.model_id == DEFAULT_MODEL_ID
    assert len(detector.method_providers()) == 1


def test_prompt_and_anchor_format() -> None:
    prefix, target, full_text = build_gradient_prompt_texts("Say hello.", "Sure.")
    assert prefix == "User: Say hello.\nAssistant:"
    assert target == " Sure."
    assert full_text == "User: Say hello.\nAssistant: Sure."


def test_llama2_prompt_template_matches_reference_shape() -> None:
    prefix, target, full_text = build_gradient_prompt_texts(
        "Say hello.",
        "Sure",
        prompt_template="llama2_inst",
        system_prompt="system",
        sep_token="<unk>",
        eos_token="</s>",
    )

    assert prefix == "<s>[INST] <<SYS>> system <</SYS>> Say hello. [/INST]"
    assert target == "<unk> Sure </s>"
    assert full_text == "<s>[INST] <<SYS>> system <</SYS>> Say hello. [/INST]<unk> Sure </s>"


def test_gradient_provider_separator_token_id_uses_explicit_separator_without_unk_token() -> None:
    class FakeTokenizer:
        unk_token = None
        unk_token_id = None

        def __call__(self, text: str, *, add_special_tokens: bool):
            assert text == "<|endoftext|>"
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[151643])

    assert (
        provider_separator_token_id(
            FakeTokenizer(),
            sep_token="<|endoftext|>",
            fallback_id=None,
        )
        == 151643
    )


def test_qwen_chat_prompt_template_uses_tokenizer_chat_template() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
            assert tokenize is False
            rendered = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            if add_generation_prompt:
                rendered += "<|im_start|>assistant\n"
            return rendered

    prefix, full_text = build_qwen_chat_prompt_texts(
        FakeTokenizer(),
        prompt="Say hello.",
        anchor_response="Sure",
    )

    assert prefix.endswith("<|im_start|>assistant\n")
    assert "Say hello." in prefix
    assert full_text.endswith("<|im_start|>assistant\nSure<|im_end|>\n")


def test_llama2_inst_tokenization_preserves_separator_when_truncating_left() -> None:
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            if text == "<sep>":
                return SimpleNamespace(input_ids=[999])
            assert kwargs["return_tensors"] == "pt"
            assert kwargs["truncation"] is False
            prefix_len = text.index("<sep>")
            ids = list(range(prefix_len)) + [999, 1000, 1001]
            values = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": values, "attention_mask": torch.ones_like(values)}

    inputs, sep = build_llama2_inst_tokenized_inputs(
        FakeTokenizer(),
        prompt="x" * 20,
        anchor_response="Sure",
        system_prompt="system",
        sep_token="<sep>",
        sep_token_id=999,
        eos_token="</s>",
        max_length=6,
        error_prefix="fixture",
    )

    assert inputs["input_ids"].shape[-1] == 6
    assert sep == 3
    assert inputs["input_ids"][0].tolist()[sep] == 999
    assert inputs["input_ids"][0].tolist()[-2:] == [1000, 1001]


def test_decision_threshold() -> None:
    blocked = _decision_from_score(score=0.75, threshold=0.5)
    allowed = _decision_from_score(score=0.25, threshold=0.5)

    assert blocked.block is True
    assert blocked.score == 0.75
    assert blocked.reason.startswith("gradsafe(")
    assert allowed.block is False
    assert allowed.score == 0.25


def test_decide_uses_score_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeScorer:
        def __init__(self, config):  # noqa: ANN001
            events.append(f"open:{config.model_id}")

        def score(self, *, prompt: str, anchor_response: str) -> GradientScoreResult:
            assert prompt == _sample().prompt
            assert anchor_response == "Sure"
            return GradientScoreResult(
                score=0.9,
                target_tokens=1,
                n_tensors=1,
                provider=ProviderSummary(
                    name="gradient_score",
                    kind="gradients",
                    requested=("anchor_loss_gradient",),
                    materialized=("gradient_norm",),
                ),
            )

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(gradsafe, "HFGradientScoreProvider", FakeScorer, raising=False)
    detector = GradSafeDetector(threshold=0.5)

    out = detector.decide(_sample())

    assert out.block is True
    assert out.score == 0.9
    assert out.reason == "gradsafe(score=0.9)"
    assert events == [f"open:{DEFAULT_MODEL_ID}", "close"]


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="max_length"):
        GradSafeDetector(max_length=0)
    with pytest.raises(ValueError, match="threshold"):
        GradSafeDetector(threshold=float("inf"))
    with pytest.raises(ValueError, match="anchor_response"):
        GradSafeDetector(anchor_response=" ")
    with pytest.raises(ValueError, match="prompt_template"):
        GradSafeDetector(prompt_template="missing")
    with pytest.raises(re.error):
        GradSafeDetector(parameter_regex="[")


def test_optional_token_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert optional_hf_token("HF_TOKEN", cached_token_fn=lambda: "cached-token") == "env-token"


def test_optional_token_uses_existing_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert optional_hf_token("HF_TOKEN", cached_token_fn=lambda: "cached-token") is True


def test_optional_token_absent_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert optional_hf_token("HF_TOKEN", cached_token_fn=lambda: None) is None


def test_missing_hf_dependencies_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "torch":
            raise ImportError("missing torch")
        return real_import_module(name)

    monkeypatch.setattr("turnkey._internal.hf_deps.importlib.import_module", fake_import_module)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[hf\]'"):
        _import_hf_dependencies()


def test_gradsafe_manifest_records_inputs_and_reproducibility() -> None:
    manifest = GradSafeDetector(threshold=0.5, model_id="test-model").manifest(name="gradsafe_v3")
    data = manifest.to_dict()

    assert data["name"] == "gradsafe_v3"
    assert data["required_inputs"] == ["sample", "prompt"]
    assert data["reproducibility"]["anchor_response"] == "Sure"
    assert data["reproducibility"]["prompt_template"] == "simple_chat"
    assert data["reproducibility"]["token_env"] == "HF_TOKEN"
    assert data["reproducibility"]["parameter_regex"] == "(mlp|self)"
    assert data["reproducibility"]["score_mode"] == "gradient_norm"


def test_gradsafe_manifest_accepts_qwen_chat_template() -> None:
    manifest = GradSafeDetector(prompt_template="qwen_chat").manifest(name="gradsafe_v3")
    assert manifest.to_dict()["reproducibility"]["prompt_template"] == "qwen_chat"


def test_gradsafe_reference_cosine_score_matches_reference_feature_filter() -> None:
    torch = pytest.importorskip("torch")
    gradients = {"layers.0.mlp.weight": torch.tensor([[-1.0, 0.0], [0.0, 1.0]])}
    reference = {"layers.0.mlp.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    minus_row = {"layers.0.mlp.weight": torch.tensor([2.0, 0.0])}
    minus_col = {"layers.0.mlp.weight": torch.tensor([0.0, 2.0])}

    score, n_features = gradsafe_reference_cosine_score(
        gradients=gradients,
        reference_gradients=reference,
        minus_row=minus_row,
        minus_col=minus_col,
        gap_threshold=1.0,
    )

    assert score == pytest.approx(0.0, abs=1e-6)
    assert n_features == 2


def test_gradsafe_reference_cosine_score_rejects_empty_feature_selection() -> None:
    torch = pytest.importorskip("torch")
    gradients = {"layers.0.mlp.weight": torch.eye(2)}
    reference = {"layers.0.mlp.weight": torch.eye(2)}
    gaps = {"layers.0.mlp.weight": torch.zeros(2)}

    with pytest.raises(RuntimeError, match="selected zero cosine features"):
        gradsafe_reference_cosine_score(
            gradients=gradients,
            reference_gradients=reference,
            minus_row=gaps,
            minus_col=gaps,
            gap_threshold=1.0,
        )


def test_gradsafe_reference_artifact_contract_summary(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    artifact_path = tmp_path / "gradsafe-reference.pt"
    torch.save(
        {
            "gradient_norms_compare": {
                "layers.0.mlp.weight": torch.eye(2),
                "layers.0.mlp.bias": torch.ones(2),
            },
            "minus_row_cos": {
                "layers.0.mlp.weight": torch.tensor([2.0, 0.0]),
                "layers.0.mlp.bias": torch.tensor([0.0]),
            },
            "minus_col_cos": {
                "layers.0.mlp.weight": torch.tensor([0.0, 2.0]),
                "layers.0.mlp.bias": torch.tensor([0.0]),
            },
        },
        artifact_path,
    )

    loaded = load_gradsafe_reference_artifact(str(artifact_path), gap_threshold=1.0)
    summary = loaded["summary"]

    assert summary["schema"] == "gradsafe_reference_artifact/v1"
    assert len(summary["sha256"]) == 64
    assert summary["n_tensors"] == 2
    assert summary["n_usable_tensors"] == 1
    assert summary["n_ignored_tensors"] == 1
    assert summary["selected_row_features"] == 1
    assert summary["selected_col_features"] == 1


def test_gradsafe_reference_artifact_rejects_shape_mismatch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    artifact_path = tmp_path / "bad-gradsafe-reference.pt"
    torch.save(
        {
            "reference_gradients": {"layers.0.mlp.weight": torch.eye(2)},
            "minus_row": {"layers.0.mlp.weight": torch.tensor([1.0])},
            "minus_col": {"layers.0.mlp.weight": torch.tensor([1.0, 1.0])},
        },
        artifact_path,
    )

    with pytest.raises(RuntimeError, match="gap shape mismatch"):
        load_gradsafe_reference_artifact(str(artifact_path), gap_threshold=0.0)


def test_gradsafe_reference_artifact_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    calls: list[tuple[str | None, float]] = []

    def fake_load(path: str | None, *, gap_threshold: float):
        calls.append((path, gap_threshold))
        return {
            "reference_gradients": {"layers.0.mlp.weight": torch.eye(2)},
            "minus_row": {"layers.0.mlp.weight": torch.ones(2)},
            "minus_col": {"layers.0.mlp.weight": torch.ones(2)},
        }

    monkeypatch.setattr(gradient_provider_module, "load_gradsafe_reference_artifact", fake_load)
    provider = HFGradientScoreProvider(
        GradientScoreConfig(
            model_id="gradient-model",
            device="cpu",
            score_mode="reference_cosine",
            reference_artifact="reference.pt",
            cosine_gap_threshold=0.0,
        )
    )
    monkeypatch.setattr(
        provider,
        "gradient_tensors",
        lambda **_kwargs: ({"layers.0.mlp.weight": torch.eye(2)}, 1),
    )

    assert provider.score(prompt="first", anchor_response="Sure").score == pytest.approx(1.0)
    assert provider.score(prompt="second", anchor_response="Sure").score == pytest.approx(1.0)
    provider.close()

    assert calls == [("reference.pt", 0.0)]


def test_gradsafe_reference_prompt_smoke_config_declares_provider_controls() -> None:
    cfg = load_config(Path("configs/runs/v9_qwen3_0_6b_gradsafe_reference_prompt_smoke.yaml"))
    detector = load_detector(cfg.detector)
    manifest = detector.manifest(name=cfg.detector.name).to_dict()

    assert cfg.run.name == "v9-qwen3-0-6b-gradsafe-reference-prompt-smoke"
    assert cfg.detector.name == "gradsafe_v3"
    assert cfg.detector.params["prompt_template"] == "llama2_inst"
    assert cfg.detector.params["separator_token"] == "<|endoftext|>"
    assert cfg.detector.params["anchor_response"] == "Sure"
    assert cfg.detector.params["parameter_regex"] == "(mlp|self)"
    assert manifest["reproducibility"]["prompt_template"] == "llama2_inst"
    assert manifest["reproducibility"]["separator_token"] == "<|endoftext|>"
    assert manifest["reproducibility"]["anchor_response"] == "Sure"


def test_gradsafe_policy_run_records_gradient_provider_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from turnkey.components.detectors import gradsafe

    class FakeGradientRequestProvider:
        request_type = gradsafe.GradientScoreRequest
        model_forwards_per_call = 1
        requires_exclusive_target = True

        def provide(self, request):  # noqa: ANN001
            return GradientScoreResult(
                score=0.9,
                target_tokens=3,
                n_tensors=2,
                provider=ProviderSummary(
                    name="gradient_score",
                    kind="gradients",
                    requested=("anchor_loss_gradient",),
                    materialized=("gradient_norm",),
                    capabilities={
                        "model_id": request.config.model_id,
                        "target_tokens": 3,
                        "n_tensors": 2,
                        "prompt_template": request.config.prompt_template,
                        "separator_token": request.config.separator_token,
                        "score_mode": request.config.score_mode,
                        "n_features": None,
                    },
                    status="ok",
                    message="fake",
                ),
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(gradsafe, "GradientScoreRequestProvider", FakeGradientRequestProvider)
    cfg_path = _gradsafe_v3_config(tmp_path)

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))

    assert audit_run_dir(run_dir) == []
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    component = run["components"]["intervention"]
    assert component["name"] == "gradsafe_v3"
    assert component["parameters"]["model_id"] == "fake-gradient-model"
    assert component["parameters"]["score_mode"] == "gradient_norm"
    sample = json.loads((run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    diagnostics = sample["intervention"]["detector"]["diagnostics"]
    assert diagnostics["score_mode"] == "gradient_norm"
    assert diagnostics["target_tokens"] == 3
    assert diagnostics["selected_parameters"] == 2

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cost"]["extra_forwards_avg"] == 1.0

    runtime_cache = run["runtime_cache"]
    assert runtime_cache["schema_version"] == "turnkey_runtime_cache_manifest/v2"
    gradient_cache_entries = [
        entry
        for entry in runtime_cache["entries"]
        if entry["bucket"] == "gradient_providers"
    ]
    assert gradient_cache_entries == []
    assert "fake-gradient-model" not in json.dumps(runtime_cache)



def _gradsafe_v3_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["name"] = "gradsafe-v3"
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["run"]["max_samples"] = 2
    raw["detector"] = {
        "name": "gradsafe_v3",
        "params": {
            "model_id": "fake-gradient-model",
            "threshold": 0.5,
            "max_length": 64,
            "parameter_regex": "(lm_head|embed_tokens|wte)",
        },
    }
    cfg_path = tmp_path / "gradsafe_v3.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path
