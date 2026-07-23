from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from turnkey.config import load_config
from turnkey._internal.redact import sha256_hex
from turnkey.runner import run_eval


def test_smoke_pipeline_outputs(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_config(cfg_path)
    run_dir = run_eval(cfg)

    cases = run_dir / "cases.jsonl"
    metrics = run_dir / "metrics.json"

    assert cases.exists()
    assert metrics.exists()
    assert (run_dir / "run.json").exists()
    assert (run_dir / "events.jsonl").exists()

    data = json.loads(metrics.read_text(encoding="utf-8"))
    for k in ("ASR_strict", "RR_harm", "ORR_benign", "NSG_abs", "NSG_rel", "WBR", "cost"):
        assert k in data
    assert "ngram_ppl_bucket" in data["groups"]
    assert sum(data["ngram_ppl_bucket_counts"].values()) == data["counts"]["n_samples"]

    rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    blocked_rows = [
        row
        for row in rows
        if row["intervention"]["detector"]["block"] is True
    ]
    blocked_executed_rows = [
        row
        for row in blocked_rows
        if row["intervention"]["model"]["executed"] is True
    ]
    assert len(rows) == data["counts"]["n_samples"]
    assert len(blocked_rows) == data["counts"]["n_blocked_total"]
    assert blocked_executed_rows == []
    for row in rows:
        params = row["attack_params"]
        assert math.isfinite(params["ngram_ppl"])
        assert params["ngram_ppl_bucket"] in {"low", "medium", "high"}


def test_smoke_pipeline_uses_configured_naturalness_reference(tmp_path: Path) -> None:
    reference = tmp_path / "naturalness_reference.jsonl"
    reference.write_text(
        '{"prompt": "Please explain a safe everyday topic."}\n'
        '{"prompt": "A calm benign instruction with common words."}\n',
        encoding="utf-8",
    )
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["naturalness"] = {
        "enabled": True,
        "reference_path": str(reference),
        "bucket_reference_path": str(reference),
        "reference_id": "unit-reference",
    }
    cfg_path = tmp_path / "smoke-naturalness.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(cfg_path))
    rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert rows
    assert {row["attack_params"]["ngram_ppl_reference"] for row in rows} == {"unit-reference"}
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["input"]["manifest"]
    assert manifest["naturalness"]["reference_path"] == str(reference)


def test_smoke_mm_pipeline_uses_images(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke_mm.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    cfg_path = tmp_path / "smoke_mm.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_config(cfg_path)
    run_dir = run_eval(cfg)

    # DummyBackend returns "OK_MM" iff images are passed through the pipeline.
    ok_mm_hash = sha256_hex("OK_MM")

    rows = []
    with (run_dir / "cases.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    assert rows
    benign = [r for r in rows if r.get("is_benign") is True]
    assert benign
    r0 = benign[0]
    for k in ("reference", "intervention"):
        resp = r0[k]["model"]["response_text"]
        assert isinstance(resp, str)
        assert f"sha256={ok_mm_hash}" in resp
