from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
import re
import types
from typing import Any, get_args, get_origin, get_type_hints

import yaml


DEFAULT_DATASET_VERSION = "v1"
DATASET_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")


@dataclass(frozen=True)
class RunConfig:
    name: str = "run"
    out_dir: str = "outputs"
    redact: bool = True
    unsafe_log_plaintext: bool = False
    max_samples: int | None = None
    input_manifest_path: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "toy"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetIdentity:
    requested_name: str
    base_name: str
    version: str
    legacy_unversioned: bool

    @property
    def versioned_name(self) -> str:
        return f"{self.base_name}@{self.version}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.base_name,
            "version": self.version,
            "versioned_name": self.versioned_name,
        }


def dataset_identity(name: str) -> DatasetIdentity:
    base_name, version = split_dataset_name_version(name)
    return DatasetIdentity(
        requested_name=name,
        base_name=base_name,
        version=version or DEFAULT_DATASET_VERSION,
        legacy_unversioned=version is None,
    )


def split_dataset_name_version(name: str) -> tuple[str, str | None]:
    if "@" not in name:
        if not name:
            raise ValueError("dataset.name must be non-empty")
        return name, None

    base_name, version = name.rsplit("@", 1)
    if not base_name:
        raise ValueError(f"dataset.name has empty base name: {name!r}")
    if not DATASET_VERSION_RE.fullmatch(version):
        raise ValueError(f"dataset.name version suffix must match @vN: {name!r}")
    return base_name, version


@dataclass(frozen=True)
class ContentConfig:
    sample_ids: list[str] = field(default_factory=list)
    behavior_ids: list[str] = field(default_factory=list)
    shuffle: bool = False
    seed: int = 0
    limit: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "hf"  # hf | openai_compat | ...
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    device: str = "auto"
    trust_remote_code: bool = False

    # openai_compat
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 300.0

    # generation
    max_new_tokens: int = 96
    temperature: float = 0.0

    # optional model-side signals
    return_prompt_logprobs: bool = False
    prefix_logprob_text: str | None = None


@dataclass(frozen=True)
class DetectorConfig:
    name: str = "allow_all"
    params: dict[str, Any] = field(default_factory=dict)
    calibration_artifact: str | None = None


@dataclass(frozen=True)
class AttackConfig:
    name: str = "none"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeConfig:
    name: str = "dummy_refusal"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NSGConfig:
    enabled: bool = True
    baseline_detector: DetectorConfig = field(default_factory=DetectorConfig)


@dataclass(frozen=True)
class NaturalnessConfig:
    enabled: bool = True
    backend: str = "local_ngram"
    reference_path: str | None = None
    bucket_reference_path: str | None = None
    reference_id: str | None = None
    boreiko_manifest_path: str | None = None
    boreiko_base_path: str | None = None
    boreiko_tokenizer_id: str = "meta-llama/Llama-2-7b-chat-hf"
    boreiko_window_size: int = 8


@dataclass(frozen=True)
class ReproductionConfig:
    schema_version: str = "turnkey_reproduction/v1"
    claim: str = "unspecified"
    setting: str = "unspecified"
    main_model: str | None = None
    reference_target_model: str | None = None
    paper_sources: list[str] = field(default_factory=list)
    reference_repos: list[str] = field(default_factory=list)
    setting_differences: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    nsg: NSGConfig = field(default_factory=NSGConfig)
    naturalness: NaturalnessConfig = field(default_factory=NaturalnessConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)


def _type_name(expected: Any) -> str:
    origin = get_origin(expected)
    if origin in (types.UnionType,):
        return " or ".join(_type_name(option) for option in get_args(expected))
    if origin is list:
        return "list"
    if origin is dict:
        return "dict"
    if expected is type(None):
        return "None"
    return getattr(expected, "__name__", str(expected))


def _config_path(prefix: str, key: object) -> str:
    return f"{prefix}.{key}" if prefix else str(key)


def _validate_config_value(path: str, value: Any, expected: Any) -> None:
    if expected is Any:
        return

    origin = get_origin(expected)
    if origin is types.UnionType:
        if any(_config_value_matches(value, option) for option in get_args(expected)):
            return
        raise TypeError(f"{path}: expected {_type_name(expected)}, got {type(value).__name__}")

    if is_dataclass(expected):
        _validate_config_mapping(value, expected, path)
        return

    if origin is list:
        if type(value) is not list:
            raise TypeError(f"{path}: expected list, got {type(value).__name__}")
        (item_type,) = get_args(expected)
        for index, item in enumerate(value):
            _validate_config_value(_config_path(path, index), item, item_type)
        return

    if origin is dict:
        if type(value) is not dict:
            raise TypeError(f"{path}: expected dict, got {type(value).__name__}")
        key_type, value_type = get_args(expected)
        for key, item in value.items():
            _validate_config_value(_config_path(path, "<key>"), key, key_type)
            _validate_config_value(_config_path(path, key), item, value_type)
        return

    if not _config_value_matches(value, expected):
        raise TypeError(f"{path}: expected {_type_name(expected)}, got {type(value).__name__}")


def _config_value_matches(value: Any, expected: Any) -> bool:
    if expected is Any:
        return True
    origin = get_origin(expected)
    if origin is types.UnionType:
        return any(_config_value_matches(value, option) for option in get_args(expected))
    if expected is type(None):
        return value is None
    if expected is bool:
        return type(value) is bool
    if expected is int:
        return type(value) is int
    if expected is float:
        return type(value) in (int, float)
    if expected is str:
        return type(value) is str
    if origin is list:
        return type(value) is list
    if origin is dict:
        return type(value) is dict
    return isinstance(value, expected)


def _validate_config_mapping(value: Any, cls: type, path: str = "") -> dict[str, Any]:
    if type(value) is not dict:
        label = path or "config root"
        raise TypeError(f"{label}: expected dict, got {type(value).__name__}")

    field_names = {item.name for item in fields(cls)}
    unknown = sorted(set(value) - field_names, key=str)
    if unknown:
        unknown_path = _config_path(path, unknown[0])
        raise ValueError(f"{unknown_path}: unknown config key")

    hints = get_type_hints(cls)
    for key, item in value.items():
        _validate_config_value(_config_path(path, key), item, hints[key])
    return value


def _validate_component_names(config: Config) -> None:
    # Component packages import these config dataclasses, so registry discovery
    # must stay lazy until this module and the parsed Config are fully initialized.
    from turnkey.components.attacks import available_attacks
    from turnkey.components.backends import available_backends
    from turnkey.components.datasets import available_datasets
    from turnkey.components.detectors import available_detectors
    from turnkey.components.judges import available_judges
    from turnkey.methods import is_entrypoint

    checks = (
        ("dataset.name", "dataset", config.dataset.name, available_datasets()),
        ("attack.name", "attack", config.attack.name, available_attacks()),
        ("model.backend", "backend", config.model.backend, available_backends()),
        ("detector.name", "detector", config.detector.name, available_detectors()),
        ("judge.name", "judge", config.judge.name, available_judges()),
        (
            "nsg.baseline_detector.name",
            "detector",
            config.nsg.baseline_detector.name,
            available_detectors(),
        ),
    )
    for path, kind, name, available in checks:
        if kind == "detector" and is_entrypoint(name):
            continue
        if name not in available:
            raise ValueError(f"{path}: unknown {kind} {name!r}; available: {available}")


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if type(raw) is not dict:
        raise TypeError("Config root must be a mapping")

    detector_raw = raw.get("detector")
    if isinstance(detector_raw, dict):
        for generated_key in (
            "component_parameters_sha256",
            "component_source",
            "resolved_name",
        ):
            detector_raw.pop(generated_key, None)
    nsg_raw = raw.get("nsg")
    baseline_detector_raw = nsg_raw.get("baseline_detector") if isinstance(nsg_raw, dict) else None
    if isinstance(baseline_detector_raw, dict):
        for generated_key in (
            "component_parameters_sha256",
            "component_source",
            "resolved_name",
        ):
            baseline_detector_raw.pop(generated_key, None)

    _validate_config_mapping(raw, Config)
    run_raw = raw.get("run", {})
    dataset_raw = raw.get("dataset", {})
    content_raw = raw.get("content", {})
    model_raw = raw.get("model", {})
    attack_raw = raw.get("attack", {})
    detector_raw = raw.get("detector", {})
    judge_raw = raw.get("judge", {})
    nsg_raw = raw.get("nsg", {})
    naturalness_raw = raw.get("naturalness", {})
    reproduction_raw = raw.get("reproduction", {})
    baseline_detector_raw = nsg_raw.get("baseline_detector", {})

    config = Config(
        run=RunConfig(**run_raw),
        dataset=DatasetConfig(**dataset_raw),
        content=ContentConfig(**content_raw),
        model=ModelConfig(**model_raw),
        attack=AttackConfig(**attack_raw),
        detector=DetectorConfig(**detector_raw),
        judge=JudgeConfig(**judge_raw),
        nsg=NSGConfig(
            enabled=nsg_raw.get("enabled", True),
            baseline_detector=DetectorConfig(**baseline_detector_raw),
        ),
        naturalness=NaturalnessConfig(**naturalness_raw),
        reproduction=ReproductionConfig(**reproduction_raw),
    )
    _validate_component_names(config)
    return config
