from __future__ import annotations

import gc
import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from turnkey._internal.hf_deps import make_hf_causal_lm_importer, optional_hf_token
from turnkey.methods import Request


DEFAULT_MODEL_ID = "meta-llama/Llama-Guard-3-1B"
SAFETY_LABEL_RE = re.compile(r"\b(safe|unsafe)\b", re.IGNORECASE)
CATEGORY_RE = re.compile(r"\bS(?:1[0-4]?|[2-9])\b", re.IGNORECASE)


_import_hf_dependencies = make_hf_causal_lm_importer(
    error_message="Llama Guard requires optional HF deps. Install with: pip install -e '.[hf]'",
    import_module=lambda name: importlib.import_module(name),
)


def _auth_token(*, token_env: str, require_token: bool) -> str | bool | None:
    token = optional_hf_token(token_env)
    if token is not None:
        return token
    if not require_token:
        return None
    raise RuntimeError(
        "Llama Guard models under `meta-llama` are gated on Hugging Face. "
        f"Set `{token_env}` to an approved token or run `hf auth login` "
        "/ `huggingface-cli login`. To use a local or ungated mirror, set "
        "`require_token: false` in the component params."
    )


def parse_llamaguard_output(text: str) -> dict[str, Any]:
    label_match = SAFETY_LABEL_RE.search(text or "")
    label = label_match.group(1).lower() if label_match else None
    categories: list[str] = []
    for category in CATEGORY_RE.findall(text or ""):
        normalized = category.upper()
        if normalized not in categories:
            categories.append(normalized)
    return {"label": label, "categories": categories}


@dataclass(frozen=True)
class LlamaGuardResult:
    label: str | None
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class LlamaGuardInputRequest(Request[LlamaGuardResult]):
    prompt: str


@dataclass
class LlamaGuardRuntime:
    """Lazy text-only runtime shared by Llama Guard judges and detectors."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = None
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 32
    trust_remote_code: bool = False
    token_env: str = "HF_TOKEN"
    require_token: bool = True
    local_files_only: bool = False
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("Llama Guard max_new_tokens must be a positive integer")

    def _resolved_torch_dtype(self, torch):  # noqa: ANN001, ANN201
        if self.torch_dtype == "auto":
            return "auto"
        try:
            return getattr(torch, self.torch_dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown torch_dtype: {self.torch_dtype}") from exc

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        torch, model_cls, tokenizer_cls = _import_hf_dependencies()
        token = _auth_token(token_env=self.token_env, require_token=self.require_token)
        load_kwargs: dict[str, Any] = {
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "token": token,
            "local_files_only": self.local_files_only,
        }
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        try:
            tokenizer = tokenizer_cls.from_pretrained(self.model_id, **load_kwargs)
            model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype=self._resolved_torch_dtype(torch),
                **load_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load Llama Guard model `{self.model_id}`. "
                f"Confirm Hugging Face access, `{self.token_env}`, and optional HF deps. "
                f"Original error: {exc}"
            ) from exc

        device_name = self.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model.to(device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def classify(self, messages: Sequence[Mapping[str, str]]) -> LlamaGuardResult:
        self._load_model()
        message_list = [dict(message) for message in messages]
        encoded = self._tokenizer.apply_chat_template(
            message_list,
            return_tensors="pt",
        )
        encoded = encoded.to(self._device)
        if isinstance(encoded, Mapping):
            model_inputs = dict(encoded)
            input_ids = model_inputs["input_ids"]
        else:
            input_ids = encoded
            model_inputs = {"input_ids": input_ids}
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
        with self._torch.inference_mode():
            outputs = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        output_ids = outputs[0][input_ids.shape[-1] :]
        raw_output = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        parsed = parse_llamaguard_output(raw_output)
        return LlamaGuardResult(
            label=parsed["label"],
            categories=tuple(parsed["categories"]),
        )

    def close(self) -> None:
        model = self._model
        self._model = None
        self._tokenizer = None
        self._device = None
        torch = self._torch
        self._torch = None
        del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            empty_cache = getattr(torch.cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()


@dataclass
class LlamaGuardInputProvider:
    runtime: LlamaGuardRuntime
    request_type = LlamaGuardInputRequest
    model_forwards_per_call = 1
    requires_exclusive_target = True

    def provide(self, request: LlamaGuardInputRequest) -> LlamaGuardResult:
        return self.runtime.classify(({"role": "user", "content": request.prompt},))

    def close(self) -> None:
        self.runtime.close()
