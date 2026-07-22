"""Names for inputs and target-model signals emitted by the current runner."""

from __future__ import annotations


STATE_SAMPLE = "sample"
STATE_PROMPT = "prompt"
STATE_IMAGES = "images"
STATE_ATTACK_METADATA = "attack_metadata"
STATE_INPUT_MANIFEST = "input_manifest"
STATE_PROMPT_LOGPROBS = "prompt_logprobs"
STATE_PREFIX_LOGPROBS = "prefix_logprobs"


_INPUT_STATES = {
    STATE_SAMPLE,
    STATE_PROMPT,
    STATE_IMAGES,
    STATE_ATTACK_METADATA,
    STATE_INPUT_MANIFEST,
}
_SIGNAL_STATES = {
    STATE_PROMPT_LOGPROBS,
    STATE_PREFIX_LOGPROBS,
}


def input_state_names() -> set[str]:
    return set(_INPUT_STATES)


def registered_state_names() -> set[str]:
    return _INPUT_STATES | _SIGNAL_STATES
