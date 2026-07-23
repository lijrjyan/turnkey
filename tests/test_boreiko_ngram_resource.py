from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnkey.boreiko_ngram_resource import BoreikoNgramScorer, build_boreiko_ngram_manifest


def test_boreiko_ngram_manifest_records_download_ids_and_file_hashes(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    repo = tmp_path / "llm-threat-model"
    repo.mkdir()
    (repo / "download_unpack_ngrams.sh").write_text(
        "gdown 1HOhjxkjwi-gLG2-gAZ2u5ZxNrpCt3RU7\n"
        "gdown 17Y2QJzDrsb5yUxLrY313ArAxTzM_s4dS\n",
        encoding="utf-8",
    )
    base = tmp_path / "ngrams_results_final" / "joined_final"
    unigram = base / "arity=1" / "df_gutenberg_unigrams_dict_normalized_hashed.parquet"
    bigram = base / "arity=2" / "df_gutenberg_bigrams_dict_normalized_hashed.parquet"
    unigram.parent.mkdir(parents=True)
    bigram.parent.mkdir(parents=True)
    pq.write_table(
        pyarrow.table({"unigram": [[1]], "normalized_count": [0.5]}),
        unigram,
    )
    pq.write_table(
        pyarrow.table({"bigram": [[1, 2]], "normalized_count": [0.25]}),
        bigram,
    )
    lock = tmp_path / "LOCK.json"
    lock.write_text(
        json.dumps({"repos": {"llm_threat_model": {"head": "ab775a74627469346d08d6cba88a6f48c1017824"}}}),
        encoding="utf-8",
    )

    out = tmp_path / "manifest.json"
    manifest = build_boreiko_ngram_manifest(repo_path=repo, base_path=base, out_path=out, lock_path=lock)

    assert manifest["schema_version"] == "turnkey_boreiko_ngram_resource_manifest/v1"
    assert manifest["reference_commit"] == "ab775a74627469346d08d6cba88a6f48c1017824"
    assert manifest["download_script"]["gdrive_file_ids"] == [
        "1HOhjxkjwi-gLG2-gAZ2u5ZxNrpCt3RU7",
        "17Y2QJzDrsb5yUxLrY313ArAxTzM_s4dS",
    ]
    assert manifest["ready"] is True
    assert manifest["missing_files"] == []
    assert manifest["invalid_files"] == []
    assert all(row["exists"] for row in manifest["expected_files"])
    assert all(row["schema_valid"] for row in manifest["expected_files"])
    assert json.loads(out.read_text(encoding="utf-8")) == manifest


def test_boreiko_ngram_manifest_rejects_invalid_parquet_schema(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    repo = tmp_path / "llm-threat-model"
    repo.mkdir()
    (repo / "download_unpack_ngrams.sh").write_text(
        "gdown 1HOhjxkjwi-gLG2-gAZ2u5ZxNrpCt3RU7\n",
        encoding="utf-8",
    )
    base = tmp_path / "ngrams_results_final" / "joined_final"
    unigram = base / "arity=1" / "df_gutenberg_unigrams_dict_normalized_hashed.parquet"
    bigram = base / "arity=2" / "df_gutenberg_bigrams_dict_normalized_hashed.parquet"
    unigram.parent.mkdir(parents=True)
    bigram.parent.mkdir(parents=True)
    pq.write_table(pyarrow.table({"unigram": [[1]], "wrong_count": [0.5]}), unigram)
    bigram.write_bytes(b"not a parquet")

    manifest = build_boreiko_ngram_manifest(
        repo_path=repo,
        base_path=base,
        out_path=tmp_path / "manifest.json",
        lock_path=None,
    )

    assert manifest["ready"] is False
    assert manifest["missing_files"] == []
    assert manifest["invalid_files"] == [
        "arity=1/df_gutenberg_unigrams_dict_normalized_hashed.parquet",
        "arity=2/df_gutenberg_bigrams_dict_normalized_hashed.parquet",
    ]
    by_path = {row["relative_path"]: row for row in manifest["expected_files"]}
    assert by_path["arity=1/df_gutenberg_unigrams_dict_normalized_hashed.parquet"]["missing_columns"] == ["normalized_count"]
    assert "schema_error" in by_path["arity=2/df_gutenberg_bigrams_dict_normalized_hashed.parquet"]


def test_boreiko_ngram_scorer_matches_bigram_window_formula() -> None:
    scorer = BoreikoNgramScorer.from_counts(
        unigrams={(1,): 0.2, (2,): 0.1},
        bigrams={(1, 2): 0.05},
        tokenizer=_TokenTokenizer({"alpha": 1, "beta": 2}),
        window_size=2,
        total_unigrams=100,
        total_bigrams=100,
    )

    unigram_prob = ((0.2 * 100) + 1.0) / (100 + 2)
    bigram_prob = ((0.05 * 100) + 1.0) / (100 + 1)
    expected = 1.0 / bigram_prob

    assert scorer.tokenize("alpha beta") == [1, 2]
    assert scorer.window_perplexity([1, 2]) == pytest.approx(expected)
    assert scorer.window_perplexity([1]) == pytest.approx(1.0 / unigram_prob)
    assert scorer.perplexity("alpha beta") == pytest.approx(expected)
    assert scorer.filter_passes("alpha beta", threshold=-1_000_000.0) is True


class _TokenTokenizer:
    def __init__(self, token_ids: dict[str, int]):
        self._token_ids = dict(token_ids)

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [self._token_ids[token] for token in text.split()]}
