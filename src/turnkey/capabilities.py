from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackendCapabilities:
    """
    Capability declaration used to avoid forking inference engines:
    detectors/judges can request signals (logprobs/hidden states/streaming/gradients),
    and the runner can pick an appropriate backend or a fallback.
    """

    streaming: bool = False
    prompt_logprobs: bool = False
    prefix_logprobs: bool = False
    token_logprobs: bool = False
    prompt_hidden_states: Literal["none", "last_layer", "select_layers", "all_layers"] = "none"
    decode_hidden_states: Literal["none", "last_token_last_layer", "select_layers"] = "none"
    gradients: bool = False
