from __future__ import annotations

from collections.abc import Callable
import gc
import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from turnkey._internal.hf_deps import optional_hf_token
from turnkey.methods import Request

from ._deps import _import_hf_dependencies
from ._types import ProviderSummary


GradientScoreMode = Literal["gradient_norm", "reference_cosine"]
GRADIENT_PROMPT_TEMPLATES = ("simple_chat", "llama2_inst", "qwen_chat")
GRADIENT_SCORE_MODES: tuple[GradientScoreMode, ...] = ("gradient_norm", "reference_cosine")
GRADSAFE_REFERENCE_ARTIFACT_SCHEMA = "gradsafe_reference_artifact/v1"


@dataclass(frozen=True)
class GradientScoreConfig:
    model_id: str
    revision: str | None = None
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    token_env: str = "HF_TOKEN"
    max_length: int = 256
    parameter_regex: str | None = None
    normalize_by_tokens: bool = True
    prompt_template: str = "simple_chat"
    system_prompt: str = "You are a helpful assistant. Help me with the following query: "
    separator_token: str | None = None
    score_mode: GradientScoreMode = "gradient_norm"
    reference_artifact: str | None = None
    cosine_gap_threshold: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_mode", normalize_gradient_score_mode(self.score_mode))

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.model_id,
            self.revision,
            self.device,
            self.torch_dtype,
            self.trust_remote_code,
            self.local_files_only,
            self.token_env,
            self.max_length,
            self.parameter_regex,
            self.normalize_by_tokens,
            self.prompt_template,
            self.system_prompt,
            self.separator_token,
            self.score_mode,
            self.reference_artifact,
            self.cosine_gap_threshold,
        )


@dataclass(frozen=True)
class GradientScoreResult:
    score: float
    target_tokens: int
    n_tensors: int
    provider: ProviderSummary


@dataclass(frozen=True)
class GradientScoreRequest(Request[GradientScoreResult]):
    config: GradientScoreConfig
    prompt: str
    anchor_response: str



class HFGradientScoreProvider:
    def __init__(self, cfg: GradientScoreConfig):
        if cfg.max_length <= 0:
            raise ValueError("gradient provider: max_length must be > 0")
        self.cfg = cfg
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._parameter_pattern = re.compile(cfg.parameter_regex) if cfg.parameter_regex is not None else None
        self._reference_artifact: dict[str, Any] | None = None
        self._reference_artifact_by_device: dict[str, dict[str, Any]] = {}
        if cfg.score_mode == "reference_cosine":
            self._reference_artifact = load_gradsafe_reference_artifact(
                cfg.reference_artifact,
                gap_threshold=float(cfg.cosine_gap_threshold),
            )

    def score(self, *, prompt: str, anchor_response: str) -> GradientScoreResult:
        gradients, target_tokens = self.gradient_tensors(
            prompt=prompt,
            anchor_response=anchor_response,
        )

        materialized = ("gradient_norm",)
        message = "anchor_loss_gradient"
        n_features = None
        if self.cfg.score_mode == "reference_cosine":
            assert self._reference_artifact is not None
            reference = self._reference_artifact_for_device()
            score, n_features = gradsafe_reference_cosine_score(
                gradients=gradients,
                reference_gradients=reference["reference_gradients"],
                minus_row=reference["minus_row"],
                minus_col=reference["minus_col"],
                gap_threshold=float(self.cfg.cosine_gap_threshold),
            )
            materialized = ("gradient_cosine_score",)
            message = "reference_gradient_cosine"
        else:
            grad_norm = gradient_norm_from_tensors(gradients.values())
            score = grad_norm / float(max(1, target_tokens)) if self.cfg.normalize_by_tokens else grad_norm
        return GradientScoreResult(
            score=float(score),
            target_tokens=int(target_tokens),
            n_tensors=len(gradients),
            provider=ProviderSummary(
                name="gradient_score",
                kind="gradients",
                requested=("anchor_loss_gradient",),
                materialized=materialized,
                capabilities={
                    "model_id": self.cfg.model_id,
                    "revision": self.cfg.revision,
                    "device": str(self._device),
                    "max_length": self.cfg.max_length,
                    "parameter_regex": self.cfg.parameter_regex,
                    "normalize_by_tokens": self.cfg.normalize_by_tokens,
                    "prompt_template": self.cfg.prompt_template,
                    "separator_token": self.cfg.separator_token,
                    "score_mode": self.cfg.score_mode,
                    "reference_artifact": self.cfg.reference_artifact,
                    "cosine_gap_threshold": self.cfg.cosine_gap_threshold,
                    "target_tokens": int(target_tokens),
                    "n_tensors": len(gradients),
                    "n_features": n_features,
                    "reference_artifact_summary": reference.get("summary") if self.cfg.score_mode == "reference_cosine" else None,
                },
                status="ok",
                message=message,
            ),
        )

    def gradient_tensors(
        self,
        *,
        prompt: str,
        anchor_response: str,
    ) -> tuple[dict[str, Any], int]:
        self._load_model()
        inputs, labels, target_tokens = self._build_inputs(prompt=prompt, anchor_response=anchor_response)

        self._model.zero_grad(set_to_none=True)
        outputs = self._model(**inputs, labels=labels)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError("gradient provider: model did not return a loss for labels")
        loss.backward()

        gradients = self._selected_gradients()
        self._model.zero_grad(set_to_none=True)
        if not gradients:
            raise RuntimeError("gradient provider: no gradients found for selected parameters")
        return gradients, int(target_tokens)

    def _reference_artifact_for_device(self) -> dict[str, Any]:
        if self._reference_artifact is None:
            raise RuntimeError("gradient provider: reference artifact is not loaded")
        torch = importlib.import_module("torch")
        device_key = str(self._device)
        cached = self._reference_artifact_by_device.get(device_key)
        if cached is not None:
            return cached

        def tensor_map(values: dict[str, Any]) -> dict[str, Any]:
            return {
                name: torch.as_tensor(value).detach().float().to(device=self._device)
                for name, value in values.items()
            }

        cached = {
            **self._reference_artifact,
            "reference_gradients": tensor_map(self._reference_artifact["reference_gradients"]),
            "minus_row": tensor_map(self._reference_artifact["minus_row"]),
            "minus_col": tensor_map(self._reference_artifact["minus_col"]),
        }
        self._reference_artifact_by_device[device_key] = cached
        return cached

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        torch, AutoModelForCausalLM, AutoTokenizer = _import_hf_dependencies()
        token = optional_hf_token(self.cfg.token_env)
        load_kwargs: dict[str, Any] = {
            "revision": self.cfg.revision,
            "trust_remote_code": self.cfg.trust_remote_code,
            "token": token,
            "local_files_only": self.cfg.local_files_only,
        }
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, **load_kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_id,
                torch_dtype=self._torch_dtype(torch),
                **load_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"gradient provider: failed to load `{self.cfg.model_id}`. Confirm model access/cache, "
                f"optional HF deps, and local_files_only/token settings. Original error: {exc}"
            ) from exc

        if self.cfg.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.cfg.device)
        self._model.to(self._device)
        self._model.eval()

    def _torch_dtype(self, torch):
        if self.cfg.torch_dtype == "auto":
            return "auto"
        try:
            return getattr(torch, self.cfg.torch_dtype)
        except AttributeError as exc:
            raise ValueError(f"gradient provider: unknown torch_dtype: {self.cfg.torch_dtype}") from exc

    def close(self) -> None:
        model = self._model
        if model is not None:
            zero_grad = getattr(model, "zero_grad", None)
            if callable(zero_grad):
                try:
                    zero_grad(set_to_none=True)
                except Exception:  # cleanup must still release the model reference
                    pass
        self._model = None
        self._tokenizer = None
        self._device = None
        self._reference_artifact = None
        self._reference_artifact_by_device.clear()
        del model
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except Exception:  # noqa: BLE001
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_inputs(self, *, prompt: str, anchor_response: str):
        prompt_template = normalize_gradient_prompt_template(self.cfg.prompt_template)
        if prompt_template == "llama2_inst":
            return self._build_llama2_inst_inputs(prompt=prompt, anchor_response=anchor_response)
        if prompt_template == "qwen_chat":
            return self._build_qwen_chat_inputs(prompt=prompt, anchor_response=anchor_response)

        prefix, _, full_text = build_gradient_prompt_texts(
            prompt,
            anchor_response,
            prompt_template=prompt_template,
            system_prompt=self.cfg.system_prompt,
        )
        prefix_ids = self._tokenizer(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"]
        inputs = self._tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=True,
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        prefix_len = min(int(prefix_ids.shape[-1]), int(labels.shape[-1]))
        labels[:, :prefix_len] = -100
        target_tokens = int((labels != -100).sum().item())
        if target_tokens <= 0:
            raise RuntimeError("gradient provider: anchor response was truncated; increase max_length")

        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        labels = labels.to(self._device)
        return inputs, labels, target_tokens

    def _build_llama2_inst_inputs(self, *, prompt: str, anchor_response: str):
        torch = importlib.import_module("torch")
        sep_token = self.cfg.separator_token or getattr(self._tokenizer, "unk_token", None)
        sep_token_id = _separator_token_id(
            self._tokenizer,
            sep_token=sep_token,
            fallback_id=getattr(self._tokenizer, "unk_token_id", None),
        )
        eos_token = getattr(self._tokenizer, "eos_token", None)
        if sep_token is None or sep_token_id is None:
            raise RuntimeError(
                "gradient provider: prompt_template=llama2_inst requires tokenizer unk_token/unk_token_id"
            )
        if eos_token is None:
            eos_token = ""

        inputs, sep = build_llama2_inst_tokenized_inputs(
            self._tokenizer,
            prompt=prompt,
            anchor_response=anchor_response,
            system_prompt=self.cfg.system_prompt,
            sep_token=str(sep_token),
            sep_token_id=int(sep_token_id),
            eos_token=str(eos_token),
            max_length=self.cfg.max_length,
            error_prefix="gradient provider",
        )
        input_ids = inputs["input_ids"]

        keep = torch.ones((int(input_ids.shape[-1]),), dtype=torch.bool)
        keep[sep] = False
        input_ids = input_ids[:, keep]
        labels = input_ids.clone()
        labels[:, :sep] = -100
        target_tokens = int((labels != -100).sum().item())
        if target_tokens <= 0:
            raise RuntimeError("gradient provider: anchor response was truncated; increase max_length")

        out = {"input_ids": input_ids}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            out["attention_mask"] = attention_mask[:, keep]
        out = {key: value.to(self._device) for key, value in out.items()}
        labels = labels.to(self._device)
        return out, labels, target_tokens

    def _build_qwen_chat_inputs(self, *, prompt: str, anchor_response: str):
        prefix, full_text = build_qwen_chat_prompt_texts(
            self._tokenizer,
            prompt=prompt,
            anchor_response=anchor_response,
        )
        prefix_ids = self._tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        inputs = self._tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=False,
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        prefix_len = min(int(prefix_ids.shape[-1]), int(labels.shape[-1]))
        labels[:, :prefix_len] = -100
        target_tokens = int((labels != -100).sum().item())
        if target_tokens <= 0:
            raise RuntimeError("gradient provider: anchor response was truncated; increase max_length")

        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        labels = labels.to(self._device)
        return inputs, labels, target_tokens

    def _selected_gradients(self) -> dict[str, Any]:
        def include(name: str) -> bool:
            return self._parameter_pattern is None or self._parameter_pattern.search(name) is not None

        selected: dict[str, Any] = {}
        for name, param in self._model.named_parameters():
            grad = getattr(param, "grad", None)
            if grad is None or not include(name):
                continue
            tensor = grad.detach().float()
            selected[name] = tensor if self.cfg.score_mode == "reference_cosine" else tensor.cpu()

        if not selected and self._parameter_pattern is not None:
            for name, param in self._model.named_parameters():
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue
                tensor = grad.detach().float()
                selected[name] = tensor if self.cfg.score_mode == "reference_cosine" else tensor.cpu()

        return selected


class GradientScoreRequestProvider:
    request_type = GradientScoreRequest
    model_forwards_per_call = 1
    requires_exclusive_target = True

    def __init__(
        self,
        *,
        factory: Callable[[GradientScoreConfig], Any] = HFGradientScoreProvider,
    ) -> None:
        self._factory = factory
        self._providers: dict[GradientScoreConfig, Any] = {}

    def provide(self, request: GradientScoreRequest) -> GradientScoreResult:
        provider = self._providers.get(request.config)
        if provider is None:
            provider = self._factory(request.config)
            self._providers[request.config] = provider
        return cast(
            GradientScoreResult,
            provider.score(
                prompt=request.prompt,
                anchor_response=request.anchor_response,
            ),
        )

    def close(self) -> None:
        errors: list[Exception] = []
        for provider in reversed(tuple(self._providers.values())):
            close = getattr(provider, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # close every materialized provider before failing
                errors.append(exc)
        self._providers.clear()
        if errors:
            raise RuntimeError(f"failed to close {len(errors)} gradient score provider(s)") from errors[0]


def normalize_gradient_prompt_template(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in GRADIENT_PROMPT_TEMPLATES:
        allowed = ", ".join(GRADIENT_PROMPT_TEMPLATES)
        raise ValueError(f"gradient provider: prompt_template must be one of: {allowed}")
    return normalized


def normalize_gradient_score_mode(value: str) -> GradientScoreMode:
    normalized = str(value).strip().lower()
    if normalized not in GRADIENT_SCORE_MODES:
        allowed = ", ".join(GRADIENT_SCORE_MODES)
        raise ValueError(f"gradient provider: score_mode must be one of: {allowed}")
    return cast(GradientScoreMode, normalized)


def gradient_norm_from_tensors(tensors) -> float:
    norm_sq = 0.0
    for tensor in tensors:
        norm_sq += float(tensor.detach().float().pow(2).sum().item())
    return math.sqrt(norm_sq)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _shape_tuple(value: Any) -> tuple[int, ...]:
    torch = importlib.import_module("torch")
    return tuple(int(x) for x in torch.as_tensor(value).shape)


def load_gradsafe_reference_artifact(
    path: str | None,
    *,
    gap_threshold: float = 1.0,
) -> dict[str, Any]:
    if not path:
        raise RuntimeError("gradient provider: score_mode=reference_cosine requires reference_artifact")
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise RuntimeError(f"gradient provider: reference_artifact does not exist: {path}")
    torch = importlib.import_module("torch")
    raw = torch.load(artifact_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise RuntimeError("gradient provider: reference_artifact must be a dict")
    reference = raw.get("reference_gradients", raw.get("gradient_norms_compare"))
    minus_row = raw.get("minus_row", raw.get("minus_row_cos"))
    minus_col = raw.get("minus_col", raw.get("minus_col_cos"))
    if not isinstance(reference, dict) or not isinstance(minus_row, dict) or not isinstance(minus_col, dict):
        raise RuntimeError(
            "gradient provider: reference_artifact requires reference_gradients/gradient_norms_compare, "
            "minus_row/minus_row_cos, and minus_col/minus_col_cos dicts"
        )

    reference_keys = set(reference)
    row_keys = set(minus_row)
    col_keys = set(minus_col)
    missing_row = sorted(reference_keys - row_keys)
    missing_col = sorted(reference_keys - col_keys)
    if missing_row or missing_col:
        raise RuntimeError(
            "gradient provider: reference_artifact row/column gap keys must cover all reference gradients"
        )

    n_usable = 0
    n_ignored = 0
    row_features = 0
    col_features = 0
    selected_row_features = 0
    selected_col_features = 0
    for name in sorted(reference_keys):
        grad_shape = _shape_tuple(reference[name])
        if len(grad_shape) < 2:
            n_ignored += 1
            continue
        row_shape = _shape_tuple(minus_row[name])
        col_shape = _shape_tuple(minus_col[name])
        expected_row = (grad_shape[0],)
        expected_col = (grad_shape[1],)
        if row_shape != expected_row or col_shape != expected_col:
            raise RuntimeError(
                "gradient provider: reference_artifact gap shape mismatch for "
                f"{name}: reference={grad_shape}, row={row_shape}, col={col_shape}"
            )
        row_gap = torch.as_tensor(minus_row[name], dtype=torch.float32)
        col_gap = torch.as_tensor(minus_col[name], dtype=torch.float32)
        n_usable += 1
        row_features += int(row_gap.numel())
        col_features += int(col_gap.numel())
        selected_row_features += int((row_gap > float(gap_threshold)).sum().item())
        selected_col_features += int((col_gap > float(gap_threshold)).sum().item())

    if n_usable <= 0:
        raise RuntimeError("gradient provider: reference_artifact selected zero usable matrix tensors")
    if selected_row_features + selected_col_features <= 0:
        raise RuntimeError("gradient provider: reference_artifact selected zero reference-cosine features")

    summary = {
        "schema": GRADSAFE_REFERENCE_ARTIFACT_SCHEMA,
        "path": str(artifact_path),
        "sha256": _file_sha256(artifact_path),
        "n_tensors": len(reference),
        "n_usable_tensors": n_usable,
        "n_ignored_tensors": n_ignored,
        "candidate_row_features": row_features,
        "candidate_col_features": col_features,
        "selected_row_features": selected_row_features,
        "selected_col_features": selected_col_features,
        "gap_threshold": float(gap_threshold),
    }
    return {
        "reference_gradients": reference,
        "minus_row": minus_row,
        "minus_col": minus_col,
        "summary": summary,
    }


def gradsafe_reference_cosine_score(
    *,
    gradients: dict[str, Any],
    reference_gradients: dict[str, Any],
    minus_row: dict[str, Any],
    minus_col: dict[str, Any],
    gap_threshold: float = 1.0,
) -> tuple[float, int]:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    feature_sum = None
    feature_count = 0
    for name, grad in gradients.items():
        reference = reference_gradients.get(name)
        row_gap = minus_row.get(name)
        col_gap = minus_col.get(name)
        if reference is None or row_gap is None or col_gap is None:
            continue
        grad_t = torch.as_tensor(grad).detach().float()
        ref_t = torch.as_tensor(reference).detach().float().to(device=grad_t.device)
        if grad_t.shape != ref_t.shape or grad_t.ndim < 2:
            continue
        row_cos = torch.nan_to_num(functional.cosine_similarity(grad_t, ref_t, dim=1))
        col_cos = torch.nan_to_num(functional.cosine_similarity(grad_t, ref_t, dim=0))
        row_mask = torch.as_tensor(row_gap, dtype=torch.float32, device=grad_t.device) > float(gap_threshold)
        col_mask = torch.as_tensor(col_gap, dtype=torch.float32, device=grad_t.device) > float(gap_threshold)
        if row_mask.shape == row_cos.shape:
            values = row_cos[row_mask]
            if values.numel():
                value_sum = values.sum()
                feature_sum = value_sum if feature_sum is None else feature_sum + value_sum
                feature_count += int(values.numel())
        if col_mask.shape == col_cos.shape:
            values = col_cos[col_mask]
            if values.numel():
                value_sum = values.sum()
                feature_sum = value_sum if feature_sum is None else feature_sum + value_sum
                feature_count += int(values.numel())

    if feature_sum is None or feature_count <= 0:
        raise RuntimeError("gradient provider: reference_cosine selected zero cosine features")
    return float((feature_sum / feature_count).detach().cpu().item()), feature_count


def build_gradient_prompt_texts(
    prompt: str,
    anchor_response: str,
    *,
    prompt_template: str = "simple_chat",
    system_prompt: str = "You are a helpful assistant. Help me with the following query: ",
    sep_token: str = "<unk>",
    eos_token: str = "</s>",
) -> tuple[str, str, str]:
    template = normalize_gradient_prompt_template(prompt_template)
    target = f" {anchor_response.strip()}"
    if template == "llama2_inst":
        prefix = f"<s>[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]"
        target = f"{sep_token}{target} {eos_token}"
        return prefix, target, prefix + target

    prefix = f"User: {prompt}\nAssistant:"
    return prefix, target, prefix + target


def build_qwen_chat_prompt_texts(tokenizer: Any, *, prompt: str, anchor_response: str) -> tuple[str, str]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError("gradient provider: prompt_template=qwen_chat requires tokenizer.apply_chat_template")

    user_messages = [{"role": "user", "content": prompt}]
    full_messages = [*user_messages, {"role": "assistant", "content": anchor_response.strip()}]
    prefix = apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    full_text = apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    if not isinstance(prefix, str) or not isinstance(full_text, str):
        raise RuntimeError("gradient provider: tokenizer.apply_chat_template must return text")
    if not prefix.strip() or not full_text.strip():
        raise RuntimeError("gradient provider: qwen_chat template produced empty text")
    return prefix, full_text


def build_llama2_inst_tokenized_inputs(
    tokenizer: Any,
    *,
    prompt: str,
    anchor_response: str,
    system_prompt: str,
    sep_token: str,
    sep_token_id: int,
    eos_token: str,
    max_length: int,
    error_prefix: str,
) -> tuple[dict[str, Any], int]:
    _, _, full_text = build_gradient_prompt_texts(
        prompt,
        anchor_response,
        prompt_template="llama2_inst",
        system_prompt=system_prompt,
        sep_token=sep_token,
        eos_token=eos_token,
    )
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"]
    ids = input_ids[0].tolist()
    try:
        sep = ids.index(int(sep_token_id))
    except ValueError as exc:
        raise RuntimeError(
            f"{error_prefix}: prompt_template=llama2_inst separator token was not found; "
            "check tokenizer unk_token/separator_token"
        ) from exc

    if int(input_ids.shape[-1]) <= int(max_length):
        return dict(inputs), sep

    suffix_len = int(input_ids.shape[-1]) - sep
    prefix_budget = int(max_length) - suffix_len
    if prefix_budget <= 0:
        raise RuntimeError(f"{error_prefix}: anchor response was truncated; increase max_length")

    torch = importlib.import_module("torch")
    cropped: dict[str, Any] = {
        "input_ids": torch.cat(
            [input_ids[:, sep - prefix_budget : sep], input_ids[:, sep:]],
            dim=-1,
        )
    }
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        cropped["attention_mask"] = torch.cat(
            [attention_mask[:, sep - prefix_budget : sep], attention_mask[:, sep:]],
            dim=-1,
        )
    return cropped, prefix_budget


def _separator_token_id(tokenizer: Any, *, sep_token: str | None, fallback_id: int | None) -> int | None:
    if sep_token is None:
        return fallback_id
    encoded = tokenizer(sep_token, add_special_tokens=False).input_ids
    if len(encoded) != 1:
        raise RuntimeError(
            "gradient provider: prompt_template=llama2_inst separator_token must map to exactly one token"
        )
    return int(encoded[0])
