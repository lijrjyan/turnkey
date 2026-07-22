from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import pytest

from turnkey.components.judges import strongreject
from turnkey.components.judges.strongreject import StrongRejectJudge, _strongreject_finetuned_tokenizer_model_id
from turnkey.schema import Sample


def test_strongreject_judge_records_reference_metadata(monkeypatch) -> None:
    def fake_import(*, evaluator: str = "strongreject_finetuned", **_kwargs):
        assert evaluator == "strongreject_finetuned"

        def evaluate(_prompt, _response, _evaluators):
            return [{"score": 0.9}]

        return evaluate

    monkeypatch.setattr("turnkey.components.judges.strongreject._import_strong_reject", fake_import)
    monkeypatch.setattr("turnkey.components.judges.strongreject._maybe_add_strong_reject_to_syspath", lambda _path: None)

    judge = StrongRejectJudge(
        testing_mode=True,
        reference_repo="https://github.com/dsbowen/strong_reject",
        reference_commit="7a551d5b440ec7b75d4f6f5bb7c1719965b76b47",
    )
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="unsafe placeholder")

    out = judge.judge(sample=sample, model_text="response")

    assert out.is_harmful_effective is True
    assert out.details["reference_repo"] == "https://github.com/dsbowen/strong_reject"
    assert out.details["reference_commit"] == "7a551d5b440ec7b75d4f6f5bb7c1719965b76b47"


def test_strongreject_finetuned_import_skips_optional_generation_dependency(
    monkeypatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "strong_reject"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "evaluate.py").write_text(
        "import os\n"
        "cached_models = {}\n"
        "registered_evaluators = {}\n"
        "if os.getenv('READTHEDOCS') != '1':\n"
        "    raise ModuleNotFoundError('litellm')\n"
        "def evaluate(_prompt, _response, _evaluators):\n"
        "    return [{'score': 0.0}]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("READTHEDOCS", raising=False)
    for name in list(sys.modules):
        if name == "strong_reject" or name.startswith("strong_reject."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    from turnkey.components.judges.strongreject import _import_strong_reject

    evaluate = _import_strong_reject(evaluator="strongreject_finetuned")

    assert evaluate("prompt", "response", ["strongreject_finetuned"]) == [{"score": 0.0}]
    assert "READTHEDOCS" not in os.environ


def test_strongreject_finetuned_tokenizer_uses_adapter_base_model(monkeypatch, tmp_path: Path) -> None:
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text('{"base_model_name_or_path": "google/gemma-2b"}', encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=(
                lambda _repo, _filename, revision, local_files_only=False: (
                    str(adapter_config)
                    if revision == "4bd893d32390d2cace4f067dc2e3ef5294fd78a2"
                    else pytest.fail(f"unexpected revision: {revision}")
                )
            )
        ),
    )

    assert (
        _strongreject_finetuned_tokenizer_model_id(
            "qylu4156/strongreject-15k-v1",
            revision="4bd893d32390d2cace4f067dc2e3ef5294fd78a2",
        )
        == "google/gemma-2b"
    )


def test_strongreject_pins_adapter_and_base_model_loads(monkeypatch) -> None:
    adapter_revision = "4bd893d32390d2cace4f067dc2e3ef5294fd78a2"
    base_revision = "9cf48e52b224239de00d483ec8eb84fb8d0f3a3a"
    calls: list[tuple[str, object, dict[str, object]]] = []

    class FakeModel:
        def eval(self) -> None:
            return None

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append(("base_model", model_id, kwargs))
            return FakeModel()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, model_id: str, **kwargs):
            calls.append(("adapter", model_id, kwargs))
            return model

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return cls()

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel))
    monkeypatch.setattr(
        strongreject,
        "_strongreject_finetuned_tokenizer_model_id",
        lambda _model_id, *, revision: "google/gemma-2b",
    )
    fake_reference = types.SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeTokenizer,
        torch=types.SimpleNamespace(bfloat16="bfloat16"),
    )

    model, tokenizer = strongreject._load_strongreject_finetuned_assets(
        strong_reject_evaluate=fake_reference,
        adapter_model_id="qylu4156/strongreject-15k-v1",
        adapter_revision=adapter_revision,
        base_model_id="google/gemma-2b",
        base_model_revision=base_revision,
    )

    assert isinstance(model, FakeModel)
    assert isinstance(tokenizer, FakeTokenizer)
    assert calls == [
        (
            "base_model",
            "google/gemma-2b",
            {"revision": base_revision, "device_map": "auto", "torch_dtype": "bfloat16"},
        ),
        ("adapter", "qylu4156/strongreject-15k-v1", {"revision": adapter_revision}),
        (
            "tokenizer",
            "google/gemma-2b",
            {"revision": base_revision, "padding_side": "left", "truncation_side": "left"},
        ),
    ]


def test_strongreject_real_finetuned_requires_peft_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        "turnkey.components.judges.strongreject._missing_modules",
        lambda _module_names: ["peft"],
    )
    monkeypatch.setattr(
        "turnkey.components.judges.strongreject._maybe_add_strong_reject_to_syspath",
        lambda _path: None,
    )

    judge = StrongRejectJudge()
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="unsafe")

    try:
        judge.judge(sample=sample, model_text="response")
    except RuntimeError as exc:
        assert "qylu4156/strongreject-15k-v1" in str(exc)
        assert "uv sync --extra hf" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing peft dependency should fail before importing StrongREJECT")


def test_strongreject_testing_mode_restores_testing_env(monkeypatch) -> None:
    def fake_import(*, evaluator: str = "strongreject_finetuned", **_kwargs):
        def evaluate(_prompt, _response, _evaluators):
            assert os.environ.get("TESTING") == "1"
            return [{"score": 0.0}]

        return evaluate

    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr("turnkey.components.judges.strongreject._import_strong_reject", fake_import)
    monkeypatch.setattr("turnkey.components.judges.strongreject._maybe_add_strong_reject_to_syspath", lambda _path: None)

    judge = StrongRejectJudge(testing_mode=True)
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="unsafe")

    judge.judge(sample=sample, model_text="response")

    assert "TESTING" not in os.environ


def test_strongreject_propagates_evaluator_failure(monkeypatch) -> None:
    def fake_import(*, evaluator: str = "strongreject_finetuned", **_kwargs):
        def evaluate(_prompt, _response, _evaluators):
            raise RuntimeError("evaluator failed")

        return evaluate

    monkeypatch.setattr("turnkey.components.judges.strongreject._import_strong_reject", fake_import)
    monkeypatch.setattr("turnkey.components.judges.strongreject._maybe_add_strong_reject_to_syspath", lambda _path: None)
    judge = StrongRejectJudge(testing_mode=True)
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="unsafe")

    with pytest.raises(RuntimeError, match="StrongREJECT evaluator failed.*evaluator failed"):
        judge.judge(sample=sample, model_text="response")

    assert "TESTING" not in os.environ


def test_strongreject_rejects_response_without_numeric_score(monkeypatch) -> None:
    def fake_import(*, evaluator: str = "strongreject_finetuned", **_kwargs):
        def evaluate(_prompt, _response, _evaluators):
            return [{"unexpected": 0.9}]

        return evaluate

    monkeypatch.setattr("turnkey.components.judges.strongreject._import_strong_reject", fake_import)
    monkeypatch.setattr("turnkey.components.judges.strongreject._maybe_add_strong_reject_to_syspath", lambda _path: None)
    judge = StrongRejectJudge(testing_mode=True)
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="unsafe")

    with pytest.raises(RuntimeError, match="StrongREJECT evaluator returned no numeric score"):
        judge.judge(sample=sample, model_text="response")
