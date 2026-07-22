from __future__ import annotations

from pathlib import Path
import re

import yaml


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JUDGE_REVISION_KEYS = {
    "llamaguard": ("revision",),
    "qwen3guard": ("revision",),
    "guardreasoner": ("revision",),
    "strongreject": ("adapter_revision", "base_model_revision"),
}


def _judge_configs(value: object):
    if isinstance(value, dict):
        name = value.get("name")
        params = value.get("params")
        if name in JUDGE_REVISION_KEYS and isinstance(params, dict):
            yield name, params
        for child in value.values():
            yield from _judge_configs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _judge_configs(child)


def test_committed_judge_configs_pin_hugging_face_models() -> None:
    found: set[str] = set()
    paths = sorted(Path("configs/runs").glob("*.yaml")) + sorted(
        Path("configs/matrix").glob("*.yaml")
    )
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for judge_name, params in _judge_configs(raw):
            found.add(judge_name)
            for key in JUDGE_REVISION_KEYS[judge_name]:
                value = params.get(key)
                assert isinstance(value, str) and COMMIT_RE.fullmatch(value), (
                    f"{path}: judge {judge_name!r} requires full immutable {key}, got {value!r}"
                )
            if judge_name == "strongreject" and params.get("testing_mode") is True:
                value = params.get("testing_model_revision")
                assert isinstance(value, str) and COMMIT_RE.fullmatch(value), (
                    f"{path}: testing StrongREJECT requires full immutable testing_model_revision, got {value!r}"
                )

    assert found == set(JUDGE_REVISION_KEYS)
