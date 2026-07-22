from __future__ import annotations

from collections.abc import Callable
import gc
import importlib
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from turnkey._internal.hf_deps import optional_hf_token
from turnkey._internal.hf_compat import append_compat_hint
from turnkey.methods import Request
from turnkey.schema import ImageInput, Sample

from ._deps import (
    _format_prompt_for_images,
    _import_hidden_state_dependencies,
    _import_internvl_dependencies,
    _import_llava_dependencies,
    _import_qwen_process_vision_info,
    _load_pil_images,
)
from ._families.internvl import (
    INTERNVL_IMAGE_SIZE as INTERNVL_IMAGE_SIZE,
    INTERNVL_MAX_IMAGE_TILES as INTERNVL_MAX_IMAGE_TILES,
    _internvl_pixel_values_from_image_path,
    _internvl_query_with_image_tokens,
)
from ._families.llava import (
    LLAVA_HIDDEN_STATE_DEFAULT_MAX_LENGTH,
    _llava_effective_max_length,
    _llava_image_dtype,
    _llava_prompt_from_sample,
)
from ._families.qwen import QWEN_HIDDEN_STATE_TEXT_MAX_LENGTH, _qwen_messages_from_sample
from ._types import ProviderSummary


HiddenStateTokenStrategy = Literal["last_token", "mean_pool", "last_5_tokens"]
HiddenStateModelFamily = Literal["generic", "qwen", "llava", "internvl"]

HIDDEN_STATE_TOKEN_STRATEGIES: tuple[HiddenStateTokenStrategy, ...] = (
    "last_token",
    "mean_pool",
    "last_5_tokens",
)
HIDDEN_STATE_MODEL_FAMILIES: tuple[HiddenStateModelFamily, ...] = ("generic", "qwen", "llava", "internvl")


class ModelFamilyAdapter(Protocol):
    def load(self, provider: HFLastTokenHiddenStateProvider, *, has_images: bool) -> None: ...

    def build_inputs(self, provider: HFLastTokenHiddenStateProvider, sample: Sample) -> dict[str, Any]: ...

    def forward(self, provider: HFLastTokenHiddenStateProvider, inputs: dict[str, Any]) -> Any: ...


class GenericModelFamilyAdapter:
    def load(self, provider: HFLastTokenHiddenStateProvider, *, has_images: bool) -> None:
        provider._load_generic_model(has_images=has_images)

    def build_inputs(self, provider: HFLastTokenHiddenStateProvider, sample: Sample) -> dict[str, Any]:
        return provider._build_generic_inputs(sample)

    def forward(self, provider: HFLastTokenHiddenStateProvider, inputs: dict[str, Any]) -> Any:
        return provider._model(**inputs, output_hidden_states=True, return_dict=True)


class QwenModelFamilyAdapter(GenericModelFamilyAdapter):
    def build_inputs(self, provider: HFLastTokenHiddenStateProvider, sample: Sample) -> dict[str, Any]:
        return provider._move_inputs_to_device(provider._build_qwen_inputs(sample))


class LlavaModelFamilyAdapter:
    def load(self, provider: HFLastTokenHiddenStateProvider, *, has_images: bool) -> None:  # noqa: ARG002
        provider._load_llava_model()

    def build_inputs(self, provider: HFLastTokenHiddenStateProvider, sample: Sample) -> dict[str, Any]:
        return provider._build_llava_inputs(sample)

    def forward(self, provider: HFLastTokenHiddenStateProvider, inputs: dict[str, Any]) -> Any:
        return provider._model(
            inputs["input_ids"],
            images=inputs.get("images"),
            image_sizes=inputs.get("image_sizes"),
            output_hidden_states=True,
        )


class InternVLModelFamilyAdapter:
    def load(self, provider: HFLastTokenHiddenStateProvider, *, has_images: bool) -> None:  # noqa: ARG002
        provider._load_internvl_model()

    def build_inputs(self, provider: HFLastTokenHiddenStateProvider, sample: Sample) -> dict[str, Any]:
        return provider._build_internvl_inputs(sample)

    def forward(self, provider: HFLastTokenHiddenStateProvider, inputs: dict[str, Any]) -> Any:
        if inputs.get("pixel_values") is not None:
            return provider._model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
                image_flags=inputs["image_flags"],
                output_hidden_states=True,
            )
        language_model = getattr(getattr(provider._model, "language_model", None), "model", None)
        if language_model is not None:
            return language_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
            )
        return provider._model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
        )


_FAMILY_HANDLERS: dict[HiddenStateModelFamily, ModelFamilyAdapter] = {
    "generic": GenericModelFamilyAdapter(),
    "qwen": QwenModelFamilyAdapter(),
    "llava": LlavaModelFamilyAdapter(),
    "internvl": InternVLModelFamilyAdapter(),
}


@dataclass(frozen=True)
class LastTokenHiddenStateConfig:
    model_id: str
    revision: str | None = None
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    token_env: str = "HF_TOKEN"
    image_token: str = "<image>"
    auto_insert_image_tokens: bool = True
    token_strategy: HiddenStateTokenStrategy = "last_token"
    include_embedding_layer: bool = False
    model_family: HiddenStateModelFamily = "generic"
    max_length: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_strategy", normalize_hidden_state_token_strategy(self.token_strategy))
        object.__setattr__(self, "model_family", normalize_hidden_state_model_family(self.model_family))
        object.__setattr__(self, "max_length", normalize_hidden_state_max_length(self.max_length))

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.model_id,
            self.revision,
            self.device,
            self.torch_dtype,
            self.trust_remote_code,
            self.local_files_only,
            self.token_env,
            self.image_token,
            self.auto_insert_image_tokens,
            self.token_strategy,
            self.include_embedding_layer,
            self.model_family,
            self.max_length,
        )


@dataclass(frozen=True)
class LastTokenHiddenStateResult:
    last_token_by_layer: Any
    n_layers: int
    hidden_size: int
    device: str
    provider: ProviderSummary


@dataclass(frozen=True)
class LastTokenHiddenStateRequest(Request[LastTokenHiddenStateResult]):
    config: LastTokenHiddenStateConfig
    sample_id: str = field(compare=False)
    behavior_id: str = field(compare=False)
    is_benign: bool = field(compare=False)
    prompt: str
    images: tuple[ImageInput, ...] = ()

    @classmethod
    def from_sample(
        cls,
        config: LastTokenHiddenStateConfig,
        sample: Sample,
    ) -> "LastTokenHiddenStateRequest":
        return cls(
            config=config,
            sample_id=sample.sample_id,
            behavior_id=sample.behavior_id,
            is_benign=sample.is_benign,
            prompt=sample.prompt,
            images=sample.images,
        )

    def to_sample(self) -> Sample:
        return Sample(
            sample_id=self.sample_id,
            behavior_id=self.behavior_id,
            is_benign=self.is_benign,
            prompt=self.prompt,
            images=self.images,
        )


class HFLastTokenHiddenStateProvider:
    def __init__(self, cfg: LastTokenHiddenStateConfig):
        self.cfg = cfg
        self._model: Any = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._image_processor: Any | None = None
        self._model_name: str | None = None
        self._device: Any = None
        self._family_adapter = _FAMILY_HANDLERS[cfg.model_family]

    def last_token_by_layer(self, sample: Sample) -> LastTokenHiddenStateResult:
        self._load_model(has_images=bool(sample.images))
        torch = importlib.import_module("torch")
        inputs = self._build_inputs(sample)

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("hidden-state provider: processor/tokenizer did not return input_ids")
        if int(input_ids.shape[-1]) <= 0:
            raise RuntimeError("hidden-state provider: empty input_ids")
        attention_mask = inputs.get("attention_mask")

        with torch.inference_mode():
            out = self._forward_hidden_state_model(inputs)
        hidden_states = self._hidden_states_from_output(out)
        if hidden_states is None:
            raise RuntimeError("hidden-state provider: model did not return hidden_states")

        hs_list = hidden_state_layer_list(hidden_states, include_embedding_layer=self.cfg.include_embedding_layer)
        if not hs_list:
            raise RuntimeError("hidden-state provider: empty hidden_states")

        vecs = [
            aggregate_hidden_state_tensor(
                tensor,
                attention_mask=attention_mask,
                token_strategy=self.cfg.token_strategy,
            ).to(dtype=torch.float32)
            for tensor in hs_list
        ]
        stacked = torch.stack(vecs, dim=0)
        n_layers = int(stacked.shape[0])
        hidden_size = int(stacked.shape[-1])
        return LastTokenHiddenStateResult(
            last_token_by_layer=stacked,
            n_layers=n_layers,
            hidden_size=hidden_size,
            device=str(self._device),
            provider=ProviderSummary(
                name="last_token_hidden_states",
                kind="hidden_states",
                requested=("last_token_by_layer",),
                materialized=("last_token_by_layer",),
                capabilities={
                    "model_id": self.cfg.model_id,
                    "revision": self.cfg.revision,
                    "device": str(self._device),
                    "torch_dtype": self.cfg.torch_dtype,
                    "n_layers": n_layers,
                    "hidden_size": hidden_size,
                    "n_images": len(sample.images),
                    "image_token": self.cfg.image_token,
                    "auto_insert_image_tokens": self.cfg.auto_insert_image_tokens,
                    "token_strategy": self.cfg.token_strategy,
                    "include_embedding_layer": self.cfg.include_embedding_layer,
                    "model_family": self.cfg.model_family,
                    "max_length": hidden_state_effective_max_length(
                        model_family=self.cfg.model_family,
                        max_length=self.cfg.max_length,
                    ),
                },
                status="ok",
                message="last_token_by_layer",
            ),
        )

    def _load_model(self, *, has_images: bool = False) -> None:
        if self._model is not None:
            return
        self._family_adapter.load(self, has_images=has_images)

    def _load_generic_model(self, *, has_images: bool = False) -> None:
        (
            torch,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
            AutoProcessor,
            AutoTokenizer,
        ) = _import_hidden_state_dependencies()
        token = optional_hf_token(self.cfg.token_env)
        load_kwargs: dict[str, Any] = {
            "revision": self.cfg.revision,
            "trust_remote_code": self.cfg.trust_remote_code,
            "token": token,
            "local_files_only": self.cfg.local_files_only,
        }
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}

        try:
            self._processor = AutoProcessor.from_pretrained(self.cfg.model_id, **load_kwargs)
        except Exception:  # noqa: BLE001
            self._processor = None

        if self._processor is not None:
            self._tokenizer = getattr(self._processor, "tokenizer", None)
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, **load_kwargs)
            except Exception:  # noqa: BLE001
                self._tokenizer = None

        if has_images:
            model_classes = (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM)
        else:
            model_classes = (AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq)

        model = None
        last_error: Exception | None = None
        for model_cls in model_classes:
            if model_cls is None:
                continue
            try:
                model = model_cls.from_pretrained(
                    self.cfg.model_id,
                    torch_dtype=self._torch_dtype(torch),
                    **load_kwargs,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if model is None:
            message = append_compat_hint(
                f"hidden-state provider: failed to load `{self.cfg.model_id}`. Confirm model access/cache, "
                "optional HF deps, and local_files_only/token settings.",
                model_id=self.cfg.model_id,
                error=last_error,
            )
            raise RuntimeError(f"{message} Original error: {last_error}") from last_error

        if self._tokenizer is not None and hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            n_emb = getattr(emb, "num_embeddings", None)
            vocab = getattr(self._tokenizer, "vocab_size", None)
            if isinstance(n_emb, int) and isinstance(vocab, int) and n_emb < vocab:
                raise ValueError(
                    "hidden-state provider: tokenizer/model mismatch "
                    "(tokenizer vocab_size exceeds model embeddings)"
                )

        if self.cfg.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.cfg.device)
        self._model = model.to(self._device)
        self._model.eval()

    def _load_llava_model(self) -> None:
        torch = importlib.import_module("torch")
        deps = _import_llava_dependencies()
        self._model_name = deps["get_model_name_from_path"](self.cfg.model_id)
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": self._llava_torch_dtype(torch),
        }
        self._tokenizer, self._model, self._image_processor, _context_len = deps["load_pretrained_model"](
            model_path=self.cfg.model_id,
            model_base=None,
            model_name=self._model_name,
            **model_kwargs,
        )

        if self.cfg.device == "auto":
            self._device = _model_device(self._model, torch) or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(self.cfg.device)
            if hasattr(self._model, "to"):
                self._model = self._model.to(self._device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _load_internvl_model(self) -> None:
        torch, AutoModel, AutoTokenizer = _import_internvl_dependencies()
        token = optional_hf_token(self.cfg.token_env)
        load_kwargs: dict[str, Any] = {
            "revision": self.cfg.revision,
            "trust_remote_code": True,
            "token": token,
            "local_files_only": self.cfg.local_files_only,
        }
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModel.from_pretrained(
            self.cfg.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            **load_kwargs,
        )
        if hasattr(self._model, "eval"):
            self._model = self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            use_fast=False,
            **load_kwargs,
        )
        if self.cfg.device == "auto":
            self._device = _model_device(self._model, torch) or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(self.cfg.device)
            if hasattr(self._model, "to"):
                self._model = self._model.to(self._device)

    def _torch_dtype(self, torch):
        if self.cfg.torch_dtype == "auto":
            return "auto"
        try:
            return getattr(torch, self.cfg.torch_dtype)
        except AttributeError as exc:
            raise ValueError(f"hidden-state provider: unknown torch_dtype: {self.cfg.torch_dtype}") from exc

    def _llava_torch_dtype(self, torch):
        if self.cfg.torch_dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        return self._torch_dtype(torch)

    def _forward_hidden_state_model(self, inputs: dict[str, Any]) -> Any:
        return self._family_adapter.forward(self, inputs)

    def _build_inputs(self, sample: Sample) -> dict[str, Any]:
        return self._family_adapter.build_inputs(self, sample)

    def _build_generic_inputs(self, sample: Sample) -> dict[str, Any]:
        text = _format_prompt_for_images(
            sample.prompt,
            n_images=len(sample.images),
            image_token=self.cfg.image_token,
            auto_insert=self.cfg.auto_insert_image_tokens,
        )
        pil_images = _load_pil_images(sample)
        inputs: dict[str, Any]
        if self._processor is not None:
            if pil_images:
                inputs = self._processor(text=text, images=pil_images, return_tensors="pt")
            else:
                inputs = self._processor(text=text, return_tensors="pt")
        elif self._tokenizer is not None:
            if pil_images:
                raise ValueError("hidden-state provider: model does not expose a processor; cannot encode images")
            inputs = self._tokenizer(text, return_tensors="pt")
        else:
            raise RuntimeError("hidden-state provider: failed to load tokenizer/processor")

        return self._move_inputs_to_device(inputs)

    def _build_qwen_inputs(self, sample: Sample) -> dict[str, Any]:
        if sample.images:
            return self._build_qwen_multimodal_inputs(sample)
        return self._build_qwen_text_inputs(sample)

    def _build_qwen_text_inputs(self, sample: Sample) -> dict[str, Any]:
        if self._tokenizer is None:
            raise RuntimeError("hidden-state provider: model_family=qwen requires a tokenizer")
        max_length = hidden_state_effective_max_length(
            model_family=self.cfg.model_family,
            max_length=self.cfg.max_length,
        )
        return self._tokenizer(
            sample.prompt,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

    def _build_qwen_multimodal_inputs(self, sample: Sample) -> dict[str, Any]:
        if self._processor is None:
            raise RuntimeError("hidden-state provider: model_family=qwen multimodal samples require a processor")
        process_vision_info = _import_qwen_process_vision_info()
        messages = _qwen_messages_from_sample(sample)
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        vision_info = process_vision_info(messages)
        image_inputs = vision_info[0] if len(vision_info) > 0 else None
        video_inputs = vision_info[1] if len(vision_info) > 1 else None
        return self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def _build_llava_inputs(self, sample: Sample) -> dict[str, Any]:
        deps = _import_llava_dependencies()
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("hidden-state provider: model_family=llava requires a loaded LLaVA model")

        model_name = self._model_name or deps["get_model_name_from_path"](self.cfg.model_id)
        prompt = _llava_prompt_from_sample(
            sample,
            model_name=model_name,
            model=self._model,
            conv_templates=deps["conv_templates"],
            constants=deps,
        )
        input_ids = deps["tokenizer_image_token"](
            prompt,
            self._tokenizer,
            deps["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0)
        max_length = _llava_effective_max_length(
            model=self._model,
            configured_max_length=self.cfg.max_length,
        )
        if int(input_ids.shape[-1]) > max_length:
            input_ids = input_ids[:, :max_length]
        input_ids = input_ids.to(self._device)

        images_tensor = None
        image_sizes = None
        if sample.images:
            pil_images = _load_pil_images(sample)
            image_sizes = [image.size for image in pil_images]
            images_tensor = deps["process_images"](pil_images, self._image_processor, self._model.config)
            if isinstance(images_tensor, list):
                raise RuntimeError("hidden-state provider: LLaVA process_images returned a list, expected tensor")
            if hasattr(images_tensor, "to"):
                images_tensor = images_tensor.to(
                    self._device,
                    dtype=_llava_image_dtype(self._model, input_ids),
                )

        return {
            "input_ids": input_ids,
            "images": images_tensor,
            "image_sizes": image_sizes,
        }

    def _build_internvl_inputs(self, sample: Sample) -> dict[str, Any]:
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("hidden-state provider: model_family=internvl requires a loaded InternVL model")
        if len(sample.images) > 1:
            raise NotImplementedError(
                "hidden-state provider: model_family=internvl currently supports one image per sample"
            )
        if sample.images:
            pixel_values = _internvl_pixel_values_from_image_path(
                sample.images[0].path,
                device=self._device,
                dtype=getattr(self._model, "dtype", None),
            )
            query = _internvl_query_with_image_tokens(
                sample.prompt,
                model=self._model,
                tokenizer=self._tokenizer,
                num_patches=int(pixel_values.size(0)),
            )
            enc = self._tokenizer(query, return_tensors="pt")
            inputs = self._move_inputs_to_device(dict(enc))
            torch = importlib.import_module("torch")
            inputs["pixel_values"] = pixel_values
            inputs["image_flags"] = torch.ones(
                (int(pixel_values.size(0)), 1),
                device=self._device,
                dtype=torch.long,
            )
            return inputs

        enc = self._tokenizer(sample.prompt, return_tensors="pt")
        return self._move_inputs_to_device(dict(enc))

    def _move_inputs_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self._device)
        return inputs

    def close(self) -> None:
        model = self._model
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._image_processor = None
        self._model_name = None
        self._device = None
        del model
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except Exception:  # noqa: BLE001
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _hidden_states_from_output(out: Any) -> Any:
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states
        lm = getattr(out, "language_model_outputs", None)
        hidden_states = getattr(lm, "hidden_states", None) if lm is not None else None
        if hidden_states is not None:
            return hidden_states
        if isinstance(out, dict):
            return out.get("hidden_states")
        return None


class LastTokenHiddenStateRequestProvider:
    request_type = LastTokenHiddenStateRequest
    model_forwards_per_call = 1
    requires_exclusive_target = True

    def __init__(
        self,
        *,
        factory: Callable[[LastTokenHiddenStateConfig], Any] = HFLastTokenHiddenStateProvider,
    ) -> None:
        self._factory = factory
        self._providers: dict[LastTokenHiddenStateConfig, Any] = {}

    def provide(self, request: LastTokenHiddenStateRequest) -> LastTokenHiddenStateResult:
        provider = self._providers.get(request.config)
        if provider is None:
            provider = self._factory(request.config)
            self._providers[request.config] = provider
        return cast(LastTokenHiddenStateResult, provider.last_token_by_layer(request.to_sample()))

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
            raise RuntimeError(f"failed to close {len(errors)} hidden-state provider(s)") from errors[0]


def normalize_hidden_state_token_strategy(value: str) -> HiddenStateTokenStrategy:
    strategy = str(value).strip().lower()
    if strategy not in HIDDEN_STATE_TOKEN_STRATEGIES:
        allowed = ", ".join(HIDDEN_STATE_TOKEN_STRATEGIES)
        raise ValueError(f"hidden-state provider: token_strategy must be one of: {allowed}")
    return cast(HiddenStateTokenStrategy, strategy)


def normalize_hidden_state_model_family(value: str) -> HiddenStateModelFamily:
    family = str(value).strip().lower()
    if family not in HIDDEN_STATE_MODEL_FAMILIES:
        allowed = ", ".join(HIDDEN_STATE_MODEL_FAMILIES)
        raise ValueError(f"hidden-state provider: model_family must be one of: {allowed}")
    return cast(HiddenStateModelFamily, family)


def normalize_hidden_state_max_length(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("hidden-state provider: max_length must be a positive integer")
    max_length = int(value)
    if max_length <= 0:
        raise ValueError("hidden-state provider: max_length must be a positive integer")
    return max_length


def hidden_state_effective_max_length(
    *,
    model_family: HiddenStateModelFamily,
    max_length: int | None,
) -> int | None:
    if max_length is not None:
        return int(max_length)
    family = normalize_hidden_state_model_family(model_family)
    if family == "qwen":
        return QWEN_HIDDEN_STATE_TEXT_MAX_LENGTH
    if family == "llava":
        return LLAVA_HIDDEN_STATE_DEFAULT_MAX_LENGTH
    return None


def _model_device(model: Any, torch: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except Exception:  # noqa: BLE001
            return None
    return None


def hidden_state_layer_list(hidden_states: Any, *, include_embedding_layer: bool) -> list[Any]:
    hs_list = list(hidden_states)
    if include_embedding_layer:
        return hs_list
    if len(hs_list) >= 2:
        return hs_list[1:]
    return hs_list


def aggregate_hidden_state_tensor(
    tensor: Any,
    *,
    attention_mask: Any | None,
    token_strategy: HiddenStateTokenStrategy,
) -> Any:
    strategy = normalize_hidden_state_token_strategy(token_strategy)
    if getattr(tensor, "ndim", None) != 3:
        raise ValueError("hidden-state provider: hidden state tensor must have shape [batch, seq_len, hidden_size]")
    if int(tensor.shape[0]) < 1:
        raise ValueError("hidden-state provider: hidden state tensor batch is empty")
    seq_len = int(tensor.shape[1])
    if seq_len <= 0:
        raise ValueError("hidden-state provider: hidden state sequence is empty")

    tokens = tensor[0]
    valid_mask = _valid_attention_mask(attention_mask, seq_len=seq_len, device=tokens.device)
    if strategy == "last_token":
        if valid_mask is not None and bool(valid_mask.any().item()):
            return tokens[valid_mask][-1]
        return tokens[-1]
    if strategy == "mean_pool":
        if valid_mask is not None and bool(valid_mask.any().item()):
            return tokens[valid_mask].mean(dim=0)
        return tokens.mean(dim=0)
    if strategy == "last_5_tokens":
        if valid_mask is not None and bool(valid_mask.any().item()):
            valid = tokens[valid_mask]
            return valid[-min(5, int(valid.shape[0])) :].mean(dim=0)
        return tokens[-min(5, seq_len) :].mean(dim=0)
    raise AssertionError(f"unreachable token strategy: {strategy}")


def _valid_attention_mask(attention_mask: Any | None, *, seq_len: int, device: Any) -> Any | None:
    if attention_mask is None or not hasattr(attention_mask, "shape"):
        return None
    mask = attention_mask[0] if getattr(attention_mask, "ndim", 0) >= 2 else attention_mask
    if int(mask.shape[-1]) != int(seq_len):
        return None
    return mask.to(device=device).reshape(-1) > 0
