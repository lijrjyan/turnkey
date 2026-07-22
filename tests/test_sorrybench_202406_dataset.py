from __future__ import annotations

import pytest

from turnkey.components.datasets import available_datasets, load_dataset
from turnkey.config import DatasetConfig
from turnkey.components.datasets import sorrybench_202406
from turnkey.components.datasets.sorrybench_202406 import SorryBench202406Params, load_sorrybench_202406


class _FakeDataset:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.shuffle_seed: int | None = None

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def shuffle(self, *, seed: int):
        self.shuffle_seed = seed
        return self

    def select(self, indexes):
        return _FakeDataset([self.rows[index] for index in indexes])


def test_sorrybench_202406_registered() -> None:
    assert "sorrybench_202406" in available_datasets()


def test_sorrybench_202406_loads_with_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return _FakeDataset(
            [
                {
                    "id": "row-1",
                    "prompt": "Do a placeholder unsafe task.",
                    "category": "placeholder",
                }
            ]
        )

    monkeypatch.setenv("HF_TOKEN", "token-value")
    monkeypatch.setattr(sorrybench_202406, "_has_cached_hf_token", lambda: False)
    monkeypatch.setattr(sorrybench_202406, "_import_datasets", lambda: fake_load_dataset)

    samples = load_sorrybench_202406(
        params=SorryBench202406Params(shuffle=False, limit=1, split="train")
    )

    assert calls["args"] == ("sorry-bench/sorry-bench-202406",)
    assert calls["kwargs"]["token"] == "token-value"
    assert samples[0].sample_id == "sorrybench_202406-row-1"
    assert samples[0].prompt == "Do a placeholder unsafe task."
    assert samples[0].attack_params["dataset"] == "sorry-bench/sorry-bench-202406"
    assert samples[0].attack_params["category"] == "placeholder"


def test_sorrybench_202406_uses_existing_login_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_load_dataset(*args, **kwargs):
        calls["kwargs"] = kwargs
        return _FakeDataset([{"sample_id": "s1", "instructions": "Instruction text."}])

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sorrybench_202406, "_has_cached_hf_token", lambda: True)
    monkeypatch.setattr(sorrybench_202406, "_import_datasets", lambda: fake_load_dataset)

    samples = load_dataset(
        DatasetConfig(
            name="sorrybench_202406",
            params={"shuffle": False, "limit": 1, "split": "train"},
        )
    )

    assert calls["kwargs"]["token"] is True
    assert samples[0].behavior_id == "sorrybench_202406:s1"


def test_sorrybench_202406_loads_real_turns_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001
        return _FakeDataset(
            [
                {
                    "question_id": 7,
                    "category": "1",
                    "turns": ["Turn one.", "Turn two."],
                    "prompt_style": "base",
                }
            ]
        )

    monkeypatch.setenv("HF_TOKEN", "token-value")
    monkeypatch.setattr(sorrybench_202406, "_import_datasets", lambda: fake_load_dataset)

    samples = load_sorrybench_202406(
        params=SorryBench202406Params(shuffle=False, limit=1, split="train")
    )

    assert samples[0].sample_id == "sorrybench_202406-7"
    assert samples[0].behavior_id == "sorrybench_202406:7"
    assert samples[0].prompt == "Turn one.\nTurn two."
    assert samples[0].attack_params["category"] == "1"
    assert samples[0].attack_params["prompt_style"] == "base"


def test_sorrybench_202406_keeps_duplicate_question_variants_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001
        return _FakeDataset(
            [
                {
                    "question_id": 7,
                    "turns": ["Variant A."],
                    "prompt_style": "base",
                },
                {
                    "question_id": 7,
                    "turns": ["Variant B."],
                    "prompt_style": "mutated",
                },
            ]
        )

    monkeypatch.setenv("HF_TOKEN", "token-value")
    monkeypatch.setattr(sorrybench_202406, "_import_datasets", lambda: fake_load_dataset)

    samples = load_sorrybench_202406(
        params=SorryBench202406Params(shuffle=False, limit=2, split="train")
    )

    assert [sample.sample_id for sample in samples] == [
        "sorrybench_202406-7-0000",
        "sorrybench_202406-7-0001",
    ]
    assert {sample.behavior_id for sample in samples} == {"sorrybench_202406:7"}
    assert [sample.prompt for sample in samples] == ["Variant A.", "Variant B."]


def test_sorrybench_202406_skips_empty_turns_rows_for_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001
        return _FakeDataset(
            [
                {
                    "question_id": 415,
                    "category": "42",
                    "turns": [None],
                    "prompt_style": "question",
                },
                {
                    "question_id": 416,
                    "category": "42",
                    "turns": ["Valid prompt."],
                    "prompt_style": "question",
                },
            ]
        )

    monkeypatch.setenv("HF_TOKEN", "token-value")
    monkeypatch.setattr(sorrybench_202406, "_import_datasets", lambda: fake_load_dataset)

    samples = load_sorrybench_202406(
        params=SorryBench202406Params(shuffle=False, limit=1, split="train")
    )

    assert [sample.sample_id for sample in samples] == ["sorrybench_202406-416"]
    assert samples[0].prompt == "Valid prompt."


def test_sorrybench_202406_gated_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("gated dataset")

    monkeypatch.setenv("HF_TOKEN", "token-value")
    monkeypatch.setattr(sorrybench_202406, "_import_datasets", lambda: fake_load_dataset)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        load_sorrybench_202406(params=SorryBench202406Params(shuffle=False, limit=1))


def test_sorrybench_202406_missing_auth_fails_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("load_dataset should not run without auth")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sorrybench_202406, "_has_cached_hf_token", lambda: False)
    monkeypatch.setattr(sorrybench_202406, "_import_datasets", lambda: fake_load_dataset)

    with pytest.raises(RuntimeError, match="hf auth login"):
        load_sorrybench_202406(params=SorryBench202406Params(shuffle=False, limit=1))
