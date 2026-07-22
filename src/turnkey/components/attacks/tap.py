from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from turnkey.components.attacks._prompt_maps import load_replay_prompt_map
from turnkey.components.attacks.core import Attack, register_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


@register_attack("tap")
def _build_tap(cfg: AttackConfig) -> Attack:
    params: dict[str, Any] = dict(cfg.params)
    mode = params.get("mode", "tag")
    if not isinstance(mode, str):
        raise ValueError("tap: attack.params.mode must be a string")
    mode = mode.strip().lower()
    if mode not in {"tag", "replay"}:
        raise ValueError("tap: attack.params.mode must be 'tag' or 'replay'")

    attack_family = params.get("attack_family")
    if attack_family is not None and not isinstance(attack_family, str):
        raise ValueError("tap: attack.params.attack_family must be a string or null")

    key_field = params.get("key_field", "sample_id")
    if not isinstance(key_field, str) or not key_field:
        raise ValueError("tap: attack.params.key_field must be a non-empty string")

    prompt_map: dict[str, str] | None = None
    prompt_map_path = params.get("prompt_map_path")
    if prompt_map_path is not None and not isinstance(prompt_map_path, str):
        raise ValueError("tap: attack.params.prompt_map_path must be a string or null")

    strict = bool(params.get("strict", True))

    if mode == "replay":
        if not prompt_map_path:
            raise ValueError("tap: mode=replay requires attack.params.prompt_map_path")
        prompt_map = load_replay_prompt_map(
            Path(prompt_map_path),
            attack_name="tap",
            key_field=key_field,
            prompt_fields=("prompt", "adv_prompt"),
        )

    class _TapAttack(Attack):
        def apply(self, sample: Sample) -> Sample:
            out_family = attack_family if attack_family is not None else sample.attack_family

            out_params: dict[str, Any] = dict(sample.attack_params)
            out_params.update(
                {
                    "base_attack_family": sample.attack_family,
                    "base_attack_method": sample.attack_method,
                    "tap_mode": mode,
                    "tap_key_field": key_field,
                    "tap_prompt_map_path": prompt_map_path,
                    "tap_strict": strict,
                }
            )

            if mode != "replay":
                return replace(
                    sample,
                    attack_family=out_family,
                    attack_method="tap",
                    attack_params=out_params,
                )

            assert prompt_map is not None
            key = getattr(sample, key_field, None)
            if not isinstance(key, str) or not key:
                raise ValueError(f"tap: sample has no string field {key_field!r}")

            attacked_prompt = prompt_map.get(key)
            if attacked_prompt is None:
                if strict:
                    raise KeyError(f"tap: missing prompt for {key_field}={key!r} (strict=true)")
                attacked_prompt = sample.prompt
                out_params["tap_replay_missing"] = True

            return replace(
                sample,
                prompt=attacked_prompt,
                attack_family=out_family,
                attack_method="tap",
                attack_params=out_params,
            )

    return _TapAttack()
