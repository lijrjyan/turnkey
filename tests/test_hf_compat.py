from __future__ import annotations

from types import SimpleNamespace

import pytest

from turnkey._internal.hf_compat import append_compat_hint, qwen35_compat_hint
from turnkey.components.backends.hf import HFBackend


def test_qwen35_hint_matches_model_id() -> None:
    hint = qwen35_compat_hint(model_id="Qwen/Qwen3.5-0.8B")
    assert hint is not None
    assert "qwen3_5" in hint
    assert "Transformers main" in hint


def test_qwen35_hint_matches_architecture_error() -> None:
    hint = qwen35_compat_hint(
        model_id="other/model",
        error="checkpoint has model type `qwen3_5`",
    )
    assert hint is not None
    assert "hybrid architecture" in hint


def test_append_compat_hint_leaves_other_models_unchanged() -> None:
    assert append_compat_hint("failed", model_id="Qwen/Qwen3-8B") == "failed"


def test_append_compat_hint_includes_architecture_error_hint() -> None:
    message = append_compat_hint("failed", model_id="other/model", error="unknown model type qwen3_5")

    assert message.startswith("failed ")
    assert "qwen3_5" in message


def test_hf_backend_reports_loaded_model_commit() -> None:
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    backend = object.__new__(HFBackend)
    backend._model = SimpleNamespace(config=SimpleNamespace(_commit_hash=revision))
    backend._tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": revision})

    assert backend.resolved_revision() == revision


def test_hf_backend_rejects_inconsistent_loaded_commits() -> None:
    backend = object.__new__(HFBackend)
    backend._model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="c1899de289a04d12100db370d81485cdf75e47ca")
    )
    backend._tokenizer = SimpleNamespace(
        init_kwargs={"_commit_hash": "0" * 40}
    )

    with pytest.raises(RuntimeError, match="inconsistent resolved revisions"):
        backend.resolved_revision()
