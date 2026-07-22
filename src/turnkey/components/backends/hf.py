from __future__ import annotations

import time
from dataclasses import dataclass
from math import fsum

from turnkey.components.backends.base import LLMBackend
from turnkey.capabilities import BackendCapabilities
from turnkey._internal.hf_deps import import_hf_causal_lm
from turnkey._internal.hf_compat import append_compat_hint
from turnkey.schema import ImageInput, ModelOutput, PrefixLogprobs, PromptLogprobs, TokenLogprob


def _import_hf():
    _, AutoModelForCausalLM, AutoTokenizer = import_hf_causal_lm(
        error_message="HF backend requires optional deps. Install with: pip install -e '.[hf]'"
    )
    return AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFBackend(LLMBackend):
    model_id: str
    revision: str | None = None
    device: str = "auto"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        AutoModelForCausalLM, AutoTokenizer = _import_hf()
        import torch

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=self.trust_remote_code
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                torch_dtype="auto",
            )
        except Exception as exc:  # noqa: BLE001
            message = append_compat_hint(
                f"hf backend: failed to load `{self.model_id}`.",
                model_id=self.model_id,
                error=exc,
            )
            raise RuntimeError(f"{message} Original error: {exc}") from exc

        if self.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device)

        self._model.to(self._device)
        self._model.eval()

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch
        except Exception:  # noqa: BLE001
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def resolved_revision(self) -> str | None:
        revisions = {
            value
            for value in (
                getattr(getattr(self._model, "config", None), "_commit_hash", None),
                getattr(self._tokenizer, "init_kwargs", {}).get("_commit_hash"),
            )
            if isinstance(value, str) and value
        }
        if len(revisions) > 1:
            raise RuntimeError(f"hf backend loaded inconsistent resolved revisions: {sorted(revisions)}")
        return next(iter(revisions), None)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(prompt_logprobs=True, prefix_logprobs=True, token_logprobs=True)

    def _ensure_text_only(self, images: tuple[ImageInput, ...] | None = None) -> None:
        if images:
            raise ValueError("hf backend does not support images (use a multimodal backend)")

    def _pad_token_id(self) -> int | None:
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
        return pad_token_id

    def _token_text(self, token_id: int) -> str:
        return self._tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _forward_token_logprobs(
        self,
        *,
        input_ids,
        attention_mask,
    ) -> tuple[TokenLogprob, ...]:
        import torch

        with torch.inference_mode():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        target_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]

        token_logprobs: list[TokenLogprob] = []
        for index, token_id in enumerate(input_ids[0].tolist()):
            token_logprobs.append(
                TokenLogprob(
                    token=self._token_text(token_id),
                    token_id=int(token_id),
                    logprob=None if index == 0 else float(target_logprobs[index - 1].item()),
                )
            )
        return tuple(token_logprobs)

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

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        tokens = self._forward_token_logprobs(input_ids=input_ids, attention_mask=attention_mask)
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

        prompt_inputs = self._tokenizer(prompt, return_tensors="pt")
        full_inputs = self._tokenizer(prompt + prefix_text, return_tensors="pt")
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        input_ids = full_inputs["input_ids"].to(self._device)
        attention_mask = full_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        all_tokens = self._forward_token_logprobs(input_ids=input_ids, attention_mask=attention_mask)
        prefix_tokens = all_tokens[prompt_len:]
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
        import torch

        self._ensure_text_only(images)

        t0 = time.time()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        prompt_tokens = int(input_ids.shape[-1])

        do_sample = temperature > 1e-6
        pad_token_id = self._pad_token_id()

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        with torch.inference_mode():
            out = self._model.generate(**gen_kwargs)
        latency_s = time.time() - t0

        out_ids = out[0]
        total_tokens = int(out_ids.shape[-1])
        completion_tokens = max(0, total_tokens - prompt_tokens)
        text = self._tokenizer.decode(out_ids[prompt_tokens:], skip_special_tokens=True)

        return ModelOutput(
            executed=True,
            backend="hf",
            model_id=self.model_id,
            response_text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=latency_s,
        )

    def generate_batch(
        self,
        *,
        prompts: tuple[str, ...],
        images: tuple[ImageInput, ...] | None = None,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[ModelOutput, ...]:
        import torch

        self._ensure_text_only(images)
        if not prompts:
            return ()

        t0 = time.time()
        old_padding_side = getattr(self._tokenizer, "padding_side", "right")
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(list(prompts), return_tensors="pt", padding=True)
        finally:
            self._tokenizer.padding_side = old_padding_side

        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        prompt_token_counts = (
            attention_mask.sum(dim=1).tolist()
            if attention_mask is not None
            else [int(input_ids.shape[-1]) for _ in prompts]
        )

        do_sample = temperature > 1e-6
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._pad_token_id(),
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.inference_mode():
            out = self._model.generate(**gen_kwargs)
        latency_s = time.time() - t0

        prompt_width = int(input_ids.shape[-1])
        results: list[ModelOutput] = []
        for index, out_ids in enumerate(out):
            completion_ids = out_ids[prompt_width:]
            completion_tokens = int(completion_ids.shape[-1])
            prompt_tokens = int(prompt_token_counts[index])
            text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append(
                ModelOutput(
                    executed=True,
                    backend="hf",
                    model_id=self.model_id,
                    response_text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_s=latency_s / max(1, len(prompts)),
                )
            )
        return tuple(results)
