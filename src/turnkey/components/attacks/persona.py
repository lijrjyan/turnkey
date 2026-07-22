from __future__ import annotations

from dataclasses import replace
from typing import Any

from turnkey.components.attacks.core import Attack, register_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def _build_prompt(*, persona_prompt: str, instruction: str, use_markers: bool) -> str:
    if use_markers:
        # Paper shape uses system + user messages; we represent this as plaintext markers because
        # the runner currently passes a single prompt string to all backends.
        return f"[System] {persona_prompt}\n[User] {instruction}\n"
    return f"{persona_prompt}\n\n{instruction}\n"


@register_attack("persona")
def _build_persona(cfg: AttackConfig) -> Attack:
    params: dict[str, Any] = dict(cfg.params)

    persona = params.get("persona")
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError("attack=persona requires attack.params.persona (string)")
    persona = persona.strip()
    if len(persona) > 256:
        raise ValueError("persona: attack.params.persona is too long (max 256 chars)")

    persona_prompt = params.get("persona_prompt", f"You must act as {persona}.")
    if not isinstance(persona_prompt, str) or not persona_prompt.strip():
        raise ValueError("persona: attack.params.persona_prompt must be a non-empty string")
    persona_prompt = persona_prompt.strip()

    use_markers = bool(params.get("use_markers", True))

    attack_family = params.get("attack_family")
    if attack_family is not None and not isinstance(attack_family, str):
        raise ValueError("persona: attack.params.attack_family must be a string or null")

    class _PersonaAttack(Attack):
        def apply(self, sample: Sample) -> Sample:
            attacked_prompt = _build_prompt(
                persona_prompt=persona_prompt,
                instruction=sample.prompt,
                use_markers=use_markers,
            )

            out_family = attack_family if attack_family is not None else sample.attack_family
            out_params: dict[str, Any] = dict(sample.attack_params)
            out_params.update(
                {
                    "base_attack_family": sample.attack_family,
                    "base_attack_method": sample.attack_method,
                    "persona_persona": persona,
                    "persona_prompt": persona_prompt,
                    "persona_use_markers": use_markers,
                }
            )
            return replace(
                sample,
                prompt=attacked_prompt,
                attack_family=out_family,
                attack_method="persona",
                attack_params=out_params,
            )

    return _PersonaAttack()
