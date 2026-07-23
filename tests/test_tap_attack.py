from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def test_tap_attack_tag_mode_updates_metadata_only() -> None:
    attack = load_attack(AttackConfig(name="tap", params={"mode": "tag"}))
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="UNSAFE_PLACEHOLDER")
    out = attack.apply(sample)

    assert out.prompt == "UNSAFE_PLACEHOLDER"
    assert out.attack_method == "tap"
    assert out.attack_params["tap_mode"] == "tag"


def test_tap_attack_replay_mode_replaces_prompt(tmp_path: Path) -> None:
    prompt_map = {"s1": "attacked prompt"}
    path = tmp_path / "tap_map.json"
    path.write_text(json.dumps(prompt_map), encoding="utf-8")

    attack = load_attack(AttackConfig(name="tap", params={"mode": "replay", "prompt_map_path": str(path)}))
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="orig")
    out = attack.apply(sample)

    assert out.prompt == "attacked prompt"
    assert out.attack_method == "tap"


def test_tap_attack_replay_mode_strict_missing_raises(tmp_path: Path) -> None:
    path = tmp_path / "tap_map.json"
    path.write_text(json.dumps({"other": "x"}), encoding="utf-8")

    attack = load_attack(
        AttackConfig(
            name="tap",
            params={"mode": "replay", "prompt_map_path": str(path), "strict": True},
        )
    )
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="orig")
    with pytest.raises(KeyError):
        attack.apply(sample)


def test_tap_attack_replay_mode_nonstrict_missing_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "tap_map.json"
    path.write_text(json.dumps({"other": "x"}), encoding="utf-8")

    attack = load_attack(
        AttackConfig(
            name="tap",
            params={"mode": "replay", "prompt_map_path": str(path), "strict": False},
        )
    )
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=False, prompt="orig")
    out = attack.apply(sample)

    assert out.prompt == "orig"
    assert out.attack_params["tap_replay_missing"] is True
