from __future__ import annotations

from abc import ABC, abstractmethod

from turnkey.capabilities import BackendCapabilities
from turnkey.schema import ImageInput, ModelOutput, PrefixLogprobs, PromptLogprobs


class LLMBackend(ABC):
    def resolved_revision(self) -> str | None:
        return None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def get_prompt_logprobs(
        self,
        *,
        prompt: str,
        images: tuple[ImageInput, ...] | None = None,
    ) -> PromptLogprobs:
        raise NotImplementedError(f"{self.__class__.__name__} does not support prompt logprobs")

    def get_prefix_logprobs(
        self,
        *,
        prompt: str,
        prefix_text: str,
        images: tuple[ImageInput, ...] | None = None,
    ) -> PrefixLogprobs:
        raise NotImplementedError(f"{self.__class__.__name__} does not support prefix logprobs")

    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
        images: tuple[ImageInput, ...] | None = None,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelOutput:
        raise NotImplementedError
