from __future__ import annotations

# ruff: noqa: E402

import os
import time
import warnings
from dataclasses import dataclass
from math import fsum
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
    category=Warning,
    module=r"requests",
)

import requests

from turnkey.components.backends.base import LLMBackend
from turnkey.capabilities import BackendCapabilities
from turnkey.schema import ImageInput, ModelOutput, PrefixLogprobs, PromptLogprobs, TokenLogprob


@dataclass
class OpenAICompatBackend(LLMBackend):
    model_id: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 300.0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(prompt_logprobs=True, prefix_logprobs=True, token_logprobs=True)

    def _ensure_text_only(self, images: tuple[ImageInput, ...] | None = None) -> None:
        if images:
            raise ValueError(
                "openai_compat backend does not support images yet (use a multimodal backend)"
            )

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _completion_echo(self, text: str) -> dict[str, Any]:
        return self._post(
            "/v1/completions",
            {
                "model": self.model_id,
                "prompt": text,
                "max_tokens": 0,
                "temperature": 0.0,
                "echo": True,
                "logprobs": 1,
            },
        )

    @staticmethod
    def _parse_chat_text(data: Any) -> str:
        if not isinstance(data, dict):
            raise RuntimeError("openai_compat protocol error: response must be an object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("openai_compat protocol error: response choices must be a non-empty list")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("openai_compat protocol error: choices[0] must be an object")

        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            raise RuntimeError(
                "openai_compat protocol error: choices[0].message.content must be a non-empty string"
            )

        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text
        raise RuntimeError(
            "openai_compat protocol error: choices[0] must contain string message.content or text"
        )

    @staticmethod
    def _parse_completion_logprobs(data: dict[str, Any]) -> tuple[TokenLogprob, ...]:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("openai_compat logprobs response missing choices")
        choice0 = choices[0]
        logprobs = choice0.get("logprobs") or {}
        tokens = logprobs.get("tokens") or []
        token_logprobs = logprobs.get("token_logprobs") or []
        token_ids = (
            logprobs.get("token_ids")
            or logprobs.get("tokens_ids")
            or choice0.get("token_ids")
            or []
        )

        parsed: list[TokenLogprob] = []
        for index, token in enumerate(tokens):
            token_id = None
            if index < len(token_ids) and isinstance(token_ids[index], int):
                token_id = int(token_ids[index])
            logprob = token_logprobs[index] if index < len(token_logprobs) else None
            parsed.append(
                TokenLogprob(
                    token=str(token),
                    token_id=token_id,
                    logprob=float(logprob) if isinstance(logprob, (int, float)) else None,
                )
            )
        return tuple(parsed)

    @staticmethod
    def _summarize(tokens: tuple[TokenLogprob, ...]) -> tuple[float | None, float | None]:
        values = [token.logprob for token in tokens if token.logprob is not None]
        if not values:
            return None, None
        total = float(fsum(values))
        return total, total / len(values)

    def get_prompt_logprobs(
        self,
        *,
        prompt: str,
        images: tuple[ImageInput, ...] | None = None,
    ) -> PromptLogprobs:
        self._ensure_text_only(images)
        tokens = self._parse_completion_logprobs(self._completion_echo(prompt))
        logprob_sum, logprob_avg = self._summarize(tokens)
        return PromptLogprobs(tokens=tokens, logprob_sum=logprob_sum, logprob_avg=logprob_avg)

    def get_prefix_logprobs(
        self,
        *,
        prompt: str,
        prefix_text: str,
        images: tuple[ImageInput, ...] | None = None,
    ) -> PrefixLogprobs:
        self._ensure_text_only(images)
        prompt_tokens = self._parse_completion_logprobs(self._completion_echo(prompt))
        full_tokens = self._parse_completion_logprobs(self._completion_echo(prompt + prefix_text))
        prefix_tokens = full_tokens[len(prompt_tokens) :]
        logprob_sum, logprob_avg = self._summarize(prefix_tokens)
        return PrefixLogprobs(
            prefix_text=prefix_text,
            tokens=tuple(prefix_tokens),
            logprob_sum=logprob_sum,
            logprob_avg=logprob_avg,
        )

    def generate(
        self,
        *,
        prompt: str,
        images: tuple[ImageInput, ...] | None = None,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelOutput:
        self._ensure_text_only(images)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }

        t0 = time.time()
        data = self._post("/v1/chat/completions", payload)
        latency_s = time.time() - t0

        text = self._parse_chat_text(data)

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        return ModelOutput(
            executed=True,
            backend="openai_compat",
            model_id=self.model_id,
            response_text=text,
            prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            completion_tokens=int(completion_tokens) if isinstance(completion_tokens, int) else None,
            total_tokens=int(total_tokens) if isinstance(total_tokens, int) else None,
            latency_s=latency_s,
        )
