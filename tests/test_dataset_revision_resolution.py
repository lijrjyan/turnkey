from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from turnkey.components.datasets import load_dataset
from turnkey.components.datasets import jbb_behaviors, sorrybench_202406, sorrybench_public, xstest
from turnkey.audit import audit_run_dir
from turnkey.config import Config, DatasetConfig, ModelConfig, NaturalnessConfig, RunConfig
from turnkey.inputs.provider import materialize_input_provider
from turnkey.runner import run_eval


_RESOLVED_REVISION = "1234567890abcdef1234567890abcdef12345678"


class _FakeDataset:
    def __init__(self, rows: list[dict], *, revision: str) -> None:
        self._rows = rows
        self._revision = revision
        self.info = SimpleNamespace(
            download_checksums={
                f"hf://datasets/AlignmentResearch/XSTest@{revision}/data.jsonl": {
                    "num_bytes": 1,
                }
            }
        )

    def __iter__(self):
        return iter(self._rows)

    def shuffle(self, *, seed: int):  # noqa: ARG002
        return self

    def select(self, indexes):
        return _FakeDataset([self._rows[index] for index in indexes], revision=self._revision)


def _xstest_config(*, revision: str | None) -> Config:
    return Config(
        dataset=DatasetConfig(
            name="xstest@v1",
            params={
                "revision": revision,
                "shuffle": False,
                "benign_limit": 1,
                "harmful_limit": 0,
            },
        ),
        naturalness=NaturalnessConfig(enabled=False),
    )


@pytest.mark.parametrize("requested_revision", [_RESOLVED_REVISION, None, "main"])
def test_input_provider_records_dataset_revision_from_loaded_metadata(
    monkeypatch: pytest.MonkeyPatch,
    requested_revision: str | None,
) -> None:
    dataset = _FakeDataset(
        [{"instructions": "Benign request.", "clf_label": 0}],
        revision=_RESOLVED_REVISION,
    )
    monkeypatch.setattr(xstest, "_import_datasets", lambda: lambda *args, **kwargs: dataset)
    cfg = _xstest_config(revision=requested_revision)

    bundle = materialize_input_provider(cfg)

    assert bundle.resolved_dataset_revision == _RESOLVED_REVISION
    assert load_dataset(cfg.dataset) == bundle.selected_samples


def test_input_provider_rejects_dataset_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeDataset(
        [{"instructions": "Benign request.", "clf_label": 0}],
        revision="abcdef1234567890abcdef1234567890abcdef12",
    )
    monkeypatch.setattr(xstest, "_import_datasets", lambda: lambda *args, **kwargs: dataset)

    with pytest.raises(ValueError, match="expected requested revision"):
        materialize_input_provider(_xstest_config(revision=_RESOLVED_REVISION))


def test_run_metadata_records_and_audits_loaded_dataset_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeDataset(
        [{"instructions": "Benign request.", "clf_label": 0}],
        revision=_RESOLVED_REVISION,
    )
    monkeypatch.setattr(xstest, "_import_datasets", lambda: lambda *args, **kwargs: dataset)
    cfg = replace(
        _xstest_config(revision=_RESOLVED_REVISION),
        run=RunConfig(name="dataset-revision", out_dir=str(tmp_path)),
        model=ModelConfig(backend="dummy", model_id="dummy"),
    )

    run_dir = run_eval(cfg)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    dataset_receipt = run["components"]["dataset"]
    assert dataset_receipt["requested_revision"] == _RESOLVED_REVISION
    assert dataset_receipt["resolved_revision"] == _RESOLVED_REVISION
    assert audit_run_dir(run_dir) == []

    dataset_receipt["resolved_revision"] = "0" * 40
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    assert any(
        "components.dataset.resolved_revision" in error
        for error in audit_run_dir(run_dir)
    )

    dataset_receipt["resolved_revision"] = None
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    assert any(
        "components.dataset.resolved_revision" in error
        for error in audit_run_dir(run_dir)
    )


def test_input_provider_rejects_unresolved_immutable_dataset_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeDataset(
        [{"instructions": "Benign request.", "clf_label": 0}],
        revision="main",
    )
    monkeypatch.setattr(xstest, "_import_datasets", lambda: lambda *args, **kwargs: dataset)

    with pytest.raises(ValueError, match="could not resolve"):
        materialize_input_provider(_xstest_config(revision=_RESOLVED_REVISION))


def test_input_provider_trusts_pinned_revision_without_checksum_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeDataset(
        [{"instructions": "Benign request.", "clf_label": 0}],
        revision=_RESOLVED_REVISION,
    )
    dataset.info.download_checksums = None  # datasets >= 4 records no checksums
    monkeypatch.setattr(xstest, "_import_datasets", lambda: lambda *args, **kwargs: dataset)

    bundle = materialize_input_provider(_xstest_config(revision=_RESOLVED_REVISION))

    assert bundle.resolved_dataset_revision == _RESOLVED_REVISION


def test_local_dataset_has_no_resolved_external_revision() -> None:
    bundle = materialize_input_provider(
        Config(
            dataset=DatasetConfig(name="fixtures_smoke@v1", params={"n_samples": 1}),
            naturalness=NaturalnessConfig(enabled=False),
        )
    )

    assert bundle.resolved_dataset_revision is None


def test_input_provider_rejects_multiple_resolved_dataset_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeDataset(
        [{"instructions": "Benign request.", "clf_label": 0}],
        revision=_RESOLVED_REVISION,
    )
    other_revision = "abcdef1234567890abcdef1234567890abcdef12"
    dataset.info.download_checksums[
        f"hf://datasets/AlignmentResearch/XSTest@{other_revision}/other.jsonl"
    ] = {"num_bytes": 1}
    monkeypatch.setattr(xstest, "_import_datasets", lambda: lambda *args, **kwargs: dataset)

    with pytest.raises(ValueError, match="multiple revisions"):
        materialize_input_provider(_xstest_config(revision=None))


def test_jbb_loader_propagates_resolved_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _FakeDataset([{"Goal": "Test goal."}], revision=_RESOLVED_REVISION)
    monkeypatch.setattr(
        jbb_behaviors,
        "_import_datasets",
        lambda: lambda *args, **kwargs: dataset,
    )
    cfg = Config(
        dataset=DatasetConfig(
            name="jbb_behaviors@v1",
            params={
                "revision": _RESOLVED_REVISION,
                "shuffle": False,
                "harmful_limit": 1,
                "benign_limit": 0,
            },
        ),
        naturalness=NaturalnessConfig(enabled=False),
    )

    bundle = materialize_input_provider(cfg)

    assert bundle.resolved_dataset_revision == _RESOLVED_REVISION


def test_sorrybench_public_loader_propagates_resolved_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeDataset(
        [{"instructions": "Test request.", "clf_label": 1}],
        revision=_RESOLVED_REVISION,
    )
    monkeypatch.setattr(
        sorrybench_public,
        "_import_datasets",
        lambda: lambda *args, **kwargs: dataset,
    )
    cfg = Config(
        dataset=DatasetConfig(
            name="sorrybench_public@v1",
            params={"revision": _RESOLVED_REVISION, "shuffle": False, "limit": 1},
        ),
        naturalness=NaturalnessConfig(enabled=False),
    )

    bundle = materialize_input_provider(cfg)

    assert bundle.resolved_dataset_revision == _RESOLVED_REVISION


def test_sorrybench_202406_loader_propagates_resolved_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeDataset(
        [{"question_id": 1, "turns": ["Test request."]}],
        revision=_RESOLVED_REVISION,
    )
    monkeypatch.setattr(sorrybench_202406, "_auth_token", lambda token_env: True)
    monkeypatch.setattr(
        sorrybench_202406,
        "_import_datasets",
        lambda: lambda *args, **kwargs: dataset,
    )
    cfg = Config(
        dataset=DatasetConfig(
            name="sorrybench_202406@v2",
            params={"revision": _RESOLVED_REVISION, "shuffle": False, "limit": 1},
        ),
        naturalness=NaturalnessConfig(enabled=False),
    )

    bundle = materialize_input_provider(cfg)

    assert bundle.resolved_dataset_revision == _RESOLVED_REVISION
