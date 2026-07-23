from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from turnkey.components.detectors.base import Detector, DetectorManifest
from turnkey.components.llamaguard_runtime import (
    DEFAULT_MODEL_ID,
    LlamaGuardInputProvider,
    LlamaGuardInputRequest,
    LlamaGuardRuntime,
)
from turnkey.methods import MethodContext, Provider
from turnkey.policy import NextPolicy, Outcome, Policy, PolicyRequest
from turnkey.schema import DetectorDecision, Sample


@dataclass
class LlamaGuardDetector(Detector):
    """Use Llama Guard as an input-side prompt safety detector."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = None
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 32
    trust_remote_code: bool = False
    token_env: str = "HF_TOKEN"
    require_token: bool = True
    local_files_only: bool = False
    fail_closed: bool = True
    _provider: LlamaGuardInputProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.fail_closed) is not bool:
            raise TypeError("Llama Guard fail_closed must be a boolean")
        runtime = LlamaGuardRuntime(
            model_id=self.model_id,
            revision=self.revision,
            device=self.device,
            torch_dtype=self.torch_dtype,
            max_new_tokens=self.max_new_tokens,
            trust_remote_code=self.trust_remote_code,
            token_env=self.token_env,
            require_token=self.require_token,
            local_files_only=self.local_files_only,
        )
        self._provider = LlamaGuardInputProvider(runtime)

    def method_providers(self) -> tuple[Provider[Any], ...]:
        return (self._provider,)

    def policy(self, *, calibration_artifact: Any | None = None) -> Policy:  # noqa: ARG002
        return LlamaGuardPolicy(fail_closed=self.fail_closed)

    def decide(self, sample: Sample) -> DetectorDecision:  # noqa: ARG002
        raise RuntimeError("Llama Guard detector requires its typed model provider")

    def manifest(self, *, name: str | None = None) -> DetectorManifest:
        return DetectorManifest(
            name=name or "llamaguard",
            version="standardized",
            required_inputs=("sample", "prompt"),
            reproducibility={
                "model_id": self.model_id,
                "revision": self.revision,
                "lane": "input_guard",
            },
        )


@dataclass(frozen=True)
class LlamaGuardPolicy:
    fail_closed: bool

    def apply(
        self,
        request: PolicyRequest,
        call_next: NextPolicy,
        context: MethodContext,
    ) -> Outcome:
        result = context.get(LlamaGuardInputRequest(prompt=request.sample.prompt))
        ambiguous = result.label not in {"safe", "unsafe"}
        block = result.label == "unsafe" or (ambiguous and self.fail_closed)
        score = 1.0 if result.label == "unsafe" else 0.0 if result.label == "safe" else None
        diagnostics = {
            "safety_label": result.label,
            "categories": list(result.categories),
            "ambiguous": ambiguous,
            "fail_closed": self.fail_closed,
        }
        if block:
            return Outcome.blocked(
                request.target,
                score=score,
                reason=(
                    "Llama Guard classified the input as unsafe"
                    if result.label == "unsafe"
                    else "Llama Guard returned an ambiguous label"
                ),
                diagnostics=diagnostics,
            )
        outcome = call_next(request)
        return replace(
            outcome,
            score=score,
            reason=(
                "Llama Guard classified the input as safe"
                if result.label == "safe"
                else "Llama Guard returned an ambiguous label"
            ),
            diagnostics={**diagnostics, **outcome.diagnostics},
        )
