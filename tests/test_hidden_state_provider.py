from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from turnkey.runtime_providers import (
    HFLastTokenHiddenStateProvider,
    LastTokenHiddenStateConfig,
    LastTokenHiddenStateRequest,
    LastTokenHiddenStateRequestProvider,
    LastTokenHiddenStateResult,
    ProviderSummary,
    aggregate_hidden_state_tensor,
    hidden_state_layer_list,
)
from turnkey.methods import MethodContext
from turnkey.schema import ImageInput, Sample


def test_hidden_state_request_provider_reuses_configured_hf_provider_and_closes() -> None:
    events: list[str] = []
    built: list[object] = []

    class FakeProvider:
        def __init__(self, cfg: LastTokenHiddenStateConfig) -> None:
            self.cfg = cfg
            self.samples: list[Sample] = []
            events.append(f"open:{cfg.model_id}")

        def last_token_by_layer(self, sample: Sample) -> LastTokenHiddenStateResult:
            self.samples.append(sample)
            return LastTokenHiddenStateResult(
                last_token_by_layer=((1.0,),),
                n_layers=1,
                hidden_size=1,
                device="cpu",
                provider=ProviderSummary(
                    name="last_token_hidden_states",
                    kind="hidden_states",
                    requested=("last_token_by_layer",),
                    materialized=("last_token_by_layer",),
                    status="ok",
                ),
            )

        def close(self) -> None:
            events.append(f"close:{self.cfg.model_id}")

    def build(cfg: LastTokenHiddenStateConfig) -> FakeProvider:
        provider = FakeProvider(cfg)
        built.append(provider)
        return provider

    cfg = LastTokenHiddenStateConfig(model_id="hidden-model", device="cpu")
    provider = LastTokenHiddenStateRequestProvider(factory=build)
    image = ImageInput(path="image.png", mime_type="image/png")
    first = LastTokenHiddenStateRequest.from_sample(
        cfg,
        Sample("first", "behavior", True, "first prompt", images=(image,)),
    )
    second = LastTokenHiddenStateRequest.from_sample(
        cfg,
        Sample("second", "behavior", False, "second prompt"),
    )
    same_input_with_different_metadata = LastTokenHiddenStateRequest.from_sample(
        cfg,
        Sample("alias", "other-behavior", False, "first prompt", images=(image,)),
    )

    with MethodContext([provider]) as context:
        assert context.get(first).n_layers == 1
        assert context.get(first).n_layers == 1
        assert context.get(same_input_with_different_metadata).n_layers == 1
        assert context.get(second).n_layers == 1

    assert len(built) == 1
    fake = built[0]
    assert isinstance(fake, FakeProvider)
    assert [(sample.sample_id, sample.is_benign, sample.images) for sample in fake.samples] == [
        ("first", True, (image,)),
        ("second", False, ()),
    ]
    assert events == ["open:hidden-model", "close:hidden-model"]


def test_hidden_state_request_provider_closes_all_configs_after_one_failure() -> None:
    events: list[str] = []

    class FakeProvider:
        def __init__(self, cfg: LastTokenHiddenStateConfig) -> None:
            self.cfg = cfg

        def last_token_by_layer(self, sample: Sample) -> LastTokenHiddenStateResult:  # noqa: ARG002
            return LastTokenHiddenStateResult(
                last_token_by_layer=((1.0,),),
                n_layers=1,
                hidden_size=1,
                device="cpu",
                provider=ProviderSummary(
                    name="last_token_hidden_states",
                    kind="hidden_states",
                    status="ok",
                ),
            )

        def close(self) -> None:
            events.append(f"close:{self.cfg.model_id}")
            if self.cfg.model_id == "first":
                raise RuntimeError("close failed")

    provider = LastTokenHiddenStateRequestProvider(factory=FakeProvider)
    for model_id in ("first", "second"):
        provider.provide(
            LastTokenHiddenStateRequest.from_sample(
                LastTokenHiddenStateConfig(model_id=model_id),
                Sample(model_id, "behavior", True, model_id),
            )
        )

    with pytest.raises(RuntimeError, match="failed to close 1 hidden-state provider"):
        provider.close()

    assert events == ["close:second", "close:first"]
    provider.close()


def test_hf_hidden_state_provider_close_drops_loaded_resources() -> None:
    provider = HFLastTokenHiddenStateProvider(LastTokenHiddenStateConfig(model_id="unused"))
    provider._model = object()  # noqa: SLF001
    provider._processor = object()  # noqa: SLF001
    provider._tokenizer = object()  # noqa: SLF001
    provider._image_processor = object()  # noqa: SLF001
    provider._model_name = "loaded"  # noqa: SLF001
    provider._device = object()  # noqa: SLF001

    provider.close()

    assert provider._model is None  # noqa: SLF001
    assert provider._processor is None  # noqa: SLF001
    assert provider._tokenizer is None  # noqa: SLF001
    assert provider._image_processor is None  # noqa: SLF001
    assert provider._model_name is None  # noqa: SLF001
    assert provider._device is None  # noqa: SLF001


def test_last_token_hidden_state_config_cache_key_tracks_identity() -> None:
    cfg = LastTokenHiddenStateConfig(
        model_id="model",
        revision="rev",
        device="cpu",
        local_files_only=True,
        image_token="<image>",
    )

    assert cfg.cache_key() == (
        "model",
        "rev",
        "cpu",
        "auto",
        False,
        True,
        "HF_TOKEN",
        "<image>",
        True,
        "last_token",
        False,
        "generic",
        None,
    )


def test_hidden_state_config_cache_key_tracks_qwen_extractor_controls() -> None:
    cfg = LastTokenHiddenStateConfig(
        model_id="model",
        model_family="qwen",
        max_length=8192,
        token_strategy="mean_pool",
        include_embedding_layer=True,
    )

    assert cfg.cache_key()[-4:] == ("mean_pool", True, "qwen", 8192)


def test_hidden_state_config_cache_key_tracks_llava_family() -> None:
    cfg = LastTokenHiddenStateConfig(
        model_id="model",
        model_family="llava",
        max_length=4096,
    )

    assert cfg.cache_key()[-2:] == ("llava", 4096)


def test_hidden_state_config_cache_key_tracks_internvl_family() -> None:
    cfg = LastTokenHiddenStateConfig(
        model_id="model",
        model_family="internvl",
    )

    assert cfg.cache_key()[-2:] == ("internvl", None)


def test_qwen_text_only_inputs_use_reference_tokenizer_path() -> None:
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def __call__(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    tokenizer = FakeTokenizer()
    provider = HFLastTokenHiddenStateProvider(
        LastTokenHiddenStateConfig(model_id="model", model_family="qwen")
    )
    provider._tokenizer = tokenizer
    provider._processor = object()
    provider._device = torch.device("cpu")

    inputs = provider._build_inputs(Sample(sample_id="s", behavior_id="b", is_benign=True, prompt="hello"))

    assert inputs["input_ids"].device.type == "cpu"
    assert tokenizer.calls == [
        {
            "text": "hello",
            "padding": True,
            "return_tensors": "pt",
            "truncation": True,
            "max_length": 8192,
        }
    ]


def test_qwen_multimodal_inputs_use_reference_processor_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from turnkey.runtime_providers import hidden_state as providers_module

    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-used-by-mock")

    class FakeProcessor:
        def __init__(self) -> None:
            self.template_calls: list[dict] = []
            self.processor_calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.template_calls.append({"messages": messages, **kwargs})
            return "templated-qwen-prompt"

        def __call__(self, **kwargs):
            self.processor_calls.append(kwargs)
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    process_calls = []

    def fake_process_vision_info(messages):
        process_calls.append(messages)
        return (["image-payload"], None)

    provider = HFLastTokenHiddenStateProvider(LastTokenHiddenStateConfig(model_id="model", model_family="qwen"))
    provider._processor = FakeProcessor()
    provider._device = torch.device("cpu")
    monkeypatch.setattr(providers_module, "_import_qwen_process_vision_info", lambda: fake_process_vision_info)

    inputs = provider._build_inputs(
        Sample(
            sample_id="s",
            behavior_id="b",
            is_benign=True,
            prompt="describe",
            images=(ImageInput(path=str(image_path)),),
        )
    )

    assert inputs["input_ids"].device.type == "cpu"
    assert len(provider._processor.template_calls) == 1
    template_call = provider._processor.template_calls[0]
    assert template_call["tokenize"] is False
    assert template_call["add_generation_prompt"] is True
    expected_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    assert template_call["messages"] == expected_messages
    assert process_calls == [expected_messages]
    assert provider._processor.processor_calls == [
        {
            "text": ["templated-qwen-prompt"],
            "images": ["image-payload"],
            "videos": None,
            "padding": True,
            "return_tensors": "pt",
        }
    ]


def test_qwen_multimodal_inputs_require_qwen_vl_utils(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from turnkey.runtime_providers import hidden_state as providers_module

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002
            return "templated"

    def missing_process_vision_info():
        raise RuntimeError("missing qwen_vl_utils")

    provider = HFLastTokenHiddenStateProvider(LastTokenHiddenStateConfig(model_id="model", model_family="qwen"))
    provider._processor = FakeProcessor()
    provider._device = torch.device("cpu")
    monkeypatch.setattr(providers_module, "_import_qwen_process_vision_info", missing_process_vision_info)

    with pytest.raises(RuntimeError, match="missing qwen_vl_utils"):
        provider._build_inputs(
            Sample(
                sample_id="s",
                behavior_id="b",
                is_benign=True,
                prompt="describe",
                images=(ImageInput(path="image.png"),),
            )
        )


def test_llava_text_inputs_use_conversation_template_and_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from turnkey.runtime_providers import hidden_state as providers_module

    token_calls: list[dict] = []
    deps = _fake_llava_deps(torch, token_calls=token_calls)
    monkeypatch.setattr(providers_module, "_import_llava_dependencies", lambda: deps)

    provider = HFLastTokenHiddenStateProvider(
        LastTokenHiddenStateConfig(model_id="/models/llava-v1", model_family="llava")
    )
    provider._tokenizer = "tokenizer"
    provider._image_processor = "image-processor"
    provider._model = _FakeLlavaModel(torch)
    provider._model_name = "llava-v1"
    provider._device = torch.device("cpu")

    inputs = provider._build_inputs(Sample(sample_id="s", behavior_id="b", is_benign=True, prompt="hello"))

    assert inputs["input_ids"].device.type == "cpu"
    assert inputs["images"] is None
    assert inputs["image_sizes"] is None
    assert token_calls == [
        {
            "prompt": "USER:hello\nASSISTANT:None",
            "tokenizer": "tokenizer",
            "image_token_index": -200,
            "return_tensors": "pt",
        }
    ]


def test_llava_image_inputs_use_image_token_and_process_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from turnkey.runtime_providers import hidden_state as providers_module

    token_calls: list[dict] = []
    process_calls: list[dict] = []
    deps = _fake_llava_deps(torch, token_calls=token_calls, process_calls=process_calls)
    monkeypatch.setattr(providers_module, "_import_llava_dependencies", lambda: deps)
    monkeypatch.setattr(providers_module, "_load_pil_images", lambda sample: [_FakeImage(size=(11, 22))])

    provider = HFLastTokenHiddenStateProvider(
        LastTokenHiddenStateConfig(model_id="/models/llava-v1", model_family="llava")
    )
    provider._tokenizer = "tokenizer"
    provider._image_processor = "image-processor"
    provider._model = _FakeLlavaModel(torch)
    provider._model_name = "llava-v1"
    provider._device = torch.device("cpu")

    inputs = provider._build_inputs(
        Sample(
            sample_id="s",
            behavior_id="b",
            is_benign=True,
            prompt="describe",
            images=(ImageInput(path="image.png"),),
        )
    )

    assert token_calls[0]["prompt"] == "USER:<image>\ndescribe\nASSISTANT:None"
    assert process_calls == [
        {
            "images": [_FakeImage(size=(11, 22))],
            "image_processor": "image-processor",
            "config": provider._model.config,
        }
    ]
    assert inputs["image_sizes"] == [(11, 22)]
    assert inputs["images"].device.type == "cpu"


def test_llava_forward_uses_reference_argument_shape() -> None:
    torch = pytest.importorskip("torch")
    model = _FakeLlavaModel(torch)
    provider = HFLastTokenHiddenStateProvider(
        LastTokenHiddenStateConfig(model_id="/models/llava-v1", model_family="llava")
    )
    provider._model = model

    out = provider._forward_hidden_state_model(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "images": torch.ones((1, 1)),
            "image_sizes": [(11, 22)],
        }
    )

    assert out == "llava-output"
    assert len(model.calls) == 1
    assert model.calls[0]["input_ids"].tolist() == [[1, 2]]
    assert torch.allclose(model.calls[0]["images"], torch.ones((1, 1)))
    assert model.calls[0]["image_sizes"] == [(11, 22)]
    assert model.calls[0]["output_hidden_states"] is True


def test_internvl_text_inputs_use_tokenizer_path() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _FakeInternVLTokenizer(torch)
    provider = HFLastTokenHiddenStateProvider(
        LastTokenHiddenStateConfig(model_id="/models/internvl", model_family="internvl")
    )
    provider._tokenizer = tokenizer
    provider._model = _FakeInternVLModel(torch)
    provider._device = torch.device("cpu")

    inputs = provider._build_inputs(Sample(sample_id="s", behavior_id="b", is_benign=True, prompt="hello"))

    assert inputs["input_ids"].device.type == "cpu"
    assert tokenizer.calls == [{"text": "hello", "return_tensors": "pt"}]


def test_internvl_image_inputs_insert_img_context_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from turnkey.runtime_providers import hidden_state as providers_module

    tokenizer = _FakeInternVLTokenizer(torch)
    model = _FakeInternVLModel(torch)
    provider = HFLastTokenHiddenStateProvider(
        LastTokenHiddenStateConfig(model_id="/models/internvl", model_family="internvl")
    )
    provider._tokenizer = tokenizer
    provider._model = model
    provider._device = torch.device("cpu")
    monkeypatch.setattr(
        providers_module,
        "_internvl_pixel_values_from_image_path",
        lambda path, *, device, dtype: torch.ones((2, 3, 2, 2), device=device, dtype=torch.float32),
    )

    inputs = provider._build_inputs(
        Sample(
            sample_id="s",
            behavior_id="b",
            is_benign=True,
            prompt="describe <image>",
            images=(ImageInput(path="image.png"),),
        )
    )

    expected_image_tokens = "<img>" + ("<IMG_CONTEXT>" * 6) + "</img>"
    assert tokenizer.calls == [{"text": f"describe {expected_image_tokens}", "return_tensors": "pt"}]
    assert model.img_context_token_id == 123
    assert inputs["pixel_values"].shape == (2, 3, 2, 2)
    assert inputs["image_flags"].shape == (2, 1)


def test_internvl_forward_uses_language_model_for_text() -> None:
    torch = pytest.importorskip("torch")
    model = _FakeInternVLModel(torch)
    provider = HFLastTokenHiddenStateProvider(
        LastTokenHiddenStateConfig(model_id="/models/internvl", model_family="internvl")
    )
    provider._model = model

    out = provider._forward_hidden_state_model(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
    )

    assert out == "internvl-language-output"
    assert model.language_model.model.calls[0]["input_ids"].tolist() == [[1, 2]]
    assert model.language_model.model.calls[0]["output_hidden_states"] is True


def test_internvl_forward_uses_pixel_values_for_images() -> None:
    torch = pytest.importorskip("torch")
    model = _FakeInternVLModel(torch)
    provider = HFLastTokenHiddenStateProvider(
        LastTokenHiddenStateConfig(model_id="/models/internvl", model_family="internvl")
    )
    provider._model = model

    out = provider._forward_hidden_state_model(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "pixel_values": torch.ones((2, 3, 2, 2)),
            "image_flags": torch.ones((2, 1), dtype=torch.long),
        }
    )

    assert out == "internvl-output"
    assert len(model.calls) == 1
    assert model.calls[0]["pixel_values"].shape == (2, 3, 2, 2)
    assert model.calls[0]["image_flags"].shape == (2, 1)


@dataclass(frozen=True)
class _FakeImage:
    size: tuple[int, int]


class _FakeLlavaConfig:
    mm_use_im_start_end = False
    max_position_embeddings = 4096


class _FakeLlavaModel:
    def __init__(self, torch) -> None:
        self.config = _FakeLlavaConfig()
        self.dtype = torch.float32
        self.calls: list[dict] = []

    def __call__(self, input_ids, *, images, image_sizes, output_hidden_states):  # noqa: ANN001
        self.calls.append(
            {
                "input_ids": input_ids,
                "images": images,
                "image_sizes": image_sizes,
                "output_hidden_states": output_hidden_states,
            }
        )
        return "llava-output"


class _FakeInternVLLanguageBackbone:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, input_ids, attention_mask, output_hidden_states):  # noqa: ANN001
        self.calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": output_hidden_states,
            }
        )
        return "internvl-language-output"


class _FakeInternVLLanguageModel:
    def __init__(self) -> None:
        self.model = _FakeInternVLLanguageBackbone()


class _FakeInternVLModel:
    def __init__(self, torch) -> None:
        self.dtype = torch.float32
        self.num_image_token = 3
        self.img_context_token_id = None
        self.language_model = _FakeInternVLLanguageModel()
        self.calls: list[dict] = []

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_flags=None,
        output_hidden_states,
    ):  # noqa: ANN001
        self.calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_flags": image_flags,
                "output_hidden_states": output_hidden_states,
            }
        )
        return "internvl-output"


class _FakeInternVLTokenizer:
    def __init__(self, torch) -> None:
        self._torch = torch
        self.calls: list[dict] = []

    def __call__(self, text: str, *, return_tensors: str):
        self.calls.append({"text": text, "return_tensors": return_tensors})
        return {
            "input_ids": self._torch.tensor([[1, 2, 3]], dtype=self._torch.long),
            "attention_mask": self._torch.tensor([[1, 1, 1]], dtype=self._torch.long),
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<IMG_CONTEXT>"
        return 123


class _FakeLlavaConv:
    roles = ("USER", "ASSISTANT")

    def __init__(self) -> None:
        self.messages: list[tuple[str, str | None]] = []

    def copy(self):
        return _FakeLlavaConv()

    def append_message(self, role: str, message: str | None) -> None:
        self.messages.append((role, message))

    def get_prompt(self) -> str:
        return "\n".join(f"{role}:{message}" for role, message in self.messages)


def _fake_llava_deps(torch, *, token_calls: list[dict], process_calls: list[dict] | None = None) -> dict:
    def tokenizer_image_token(prompt, tokenizer, image_token_index, *, return_tensors):  # noqa: ANN001
        token_calls.append(
            {
                "prompt": prompt,
                "tokenizer": tokenizer,
                "image_token_index": image_token_index,
                "return_tensors": return_tensors,
            }
        )
        return torch.tensor([1, 2, 3], dtype=torch.long)

    def process_images(images, image_processor, config):  # noqa: ANN001
        if process_calls is not None:
            process_calls.append(
                {
                    "images": images,
                    "image_processor": image_processor,
                    "config": config,
                }
            )
        return torch.ones((1, 3, 2, 2), dtype=torch.float32)

    return {
        "IMAGE_TOKEN_INDEX": -200,
        "DEFAULT_IMAGE_TOKEN": "<image>",
        "DEFAULT_IM_START_TOKEN": "<im_start>",
        "DEFAULT_IM_END_TOKEN": "<im_end>",
        "IMAGE_PLACEHOLDER": "<image-placeholder>",
        "conv_templates": {
            "llava_v0": _FakeLlavaConv(),
            "llava_v1": _FakeLlavaConv(),
        },
        "load_pretrained_model": None,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "get_model_name_from_path": lambda path: "llava-v1",
    }


def test_hidden_state_layer_list_can_include_embedding_layer() -> None:
    torch = pytest.importorskip("torch")
    hidden_states = (
        torch.full((1, 2, 3), 0.0),
        torch.full((1, 2, 3), 1.0),
        torch.full((1, 2, 3), 2.0),
    )

    dropped = hidden_state_layer_list(hidden_states, include_embedding_layer=False)
    included = hidden_state_layer_list(hidden_states, include_embedding_layer=True)

    assert len(dropped) == 2
    assert float(dropped[0][0, 0, 0]) == 1.0
    assert len(included) == 3
    assert float(included[0][0, 0, 0]) == 0.0


def test_aggregate_hidden_state_tensor_supports_reference_token_strategies() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.tensor(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ]
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

    last = aggregate_hidden_state_tensor(tensor, attention_mask=attention_mask, token_strategy="last_token")
    pooled = aggregate_hidden_state_tensor(tensor, attention_mask=attention_mask, token_strategy="mean_pool")
    last_5 = aggregate_hidden_state_tensor(tensor, attention_mask=attention_mask, token_strategy="last_5_tokens")

    assert torch.allclose(last, torch.tensor([4.0, 40.0]))
    assert torch.allclose(pooled, torch.tensor([2.5, 25.0]))
    assert torch.allclose(last_5, torch.tensor([2.5, 25.0]))
