from __future__ import annotations

import json
from pathlib import Path

import yaml

from turnkey.config import load_config
from turnkey.runner import run_eval


def test_smoke_pipeline_logs_prompt_and_prefix_logprobs(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke_logprobs.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    cfg_path = tmp_path / "smoke_logprobs.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_config(cfg_path)
    run_dir = run_eval(cfg)

    with (run_dir / "cases.jsonl").open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert rows
    row = rows[0]

    for key in ("reference", "intervention"):
        model = row[key]["model"]
        prompt_logprobs = model["prompt_logprobs"]
        prefix_logprobs = model["prefix_logprobs"]

        assert prompt_logprobs is not None
        assert prefix_logprobs is not None
        assert prompt_logprobs["tokens"]
        assert prefix_logprobs["prefix_text"].startswith("<redacted sha256=")
        assert isinstance(prefix_logprobs["tokens"], list)


def test_blocked_sample_still_records_prompt_logprobs(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke_logprobs.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["detector"]["params"]["keywords"] = ["Say hello"]
    cfg_path = tmp_path / "smoke_logprobs_blocked.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_config(cfg_path)
    run_dir = run_eval(cfg)

    with (run_dir / "cases.jsonl").open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    blocked = next(row for row in rows if row["intervention"]["detector"]["block"] is True)
    model = blocked["intervention"]["model"]

    assert model["executed"] is False
    assert model["prompt_logprobs"] is not None
    assert model["prefix_logprobs"] is not None
