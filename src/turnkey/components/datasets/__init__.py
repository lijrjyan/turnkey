from __future__ import annotations

import warnings

from turnkey.config import DatasetConfig, dataset_identity
from turnkey.schema import Sample

from turnkey.registry import DATASETS, register_dataset

from .fixtures_smoke import load_fixtures_smoke_dataset
from .fixtures_smoke_mm import load_fixtures_smoke_mm_dataset
from .jbb_behaviors import JBBBehaviorsParams, load_jbb_behaviors_with_metadata
from .jsonl_prompts import JsonlPromptsParams, load_jsonl_prompts
from .sorrybench_202406 import SorryBench202406Params, load_sorrybench_202406_with_metadata
from .sorrybench_public import SorryBenchPublicParams, load_sorrybench_public_with_metadata
from .toy import load_toy_dataset
from .xstest import XSTestParams, load_xstest_with_metadata
from ._revision import DatasetLoadResult


@register_dataset("toy")
@register_dataset("toy@v1")
def _load_toy(cfg: DatasetConfig) -> list[Sample]:
    return load_toy_dataset(**cfg.params)


@register_dataset("fixtures_smoke")
@register_dataset("fixtures_smoke@v1")
def _load_fixtures(cfg: DatasetConfig) -> list[Sample]:
    return load_fixtures_smoke_dataset(**cfg.params)


@register_dataset("fixtures_smoke_mm")
@register_dataset("fixtures_smoke_mm@v1")
def _load_fixtures_mm(cfg: DatasetConfig) -> list[Sample]:
    return load_fixtures_smoke_mm_dataset(**cfg.params)


@register_dataset("jsonl_prompts")
@register_dataset("jsonl_prompts@v1")
def _load_jsonl_prompts(cfg: DatasetConfig) -> list[Sample]:
    params = JsonlPromptsParams(**cfg.params)
    return load_jsonl_prompts(params=params)


@register_dataset("jbb_behaviors")
@register_dataset("jbb_behaviors@v1")
def _load_jbb_behaviors(cfg: DatasetConfig) -> DatasetLoadResult:
    params = JBBBehaviorsParams(**cfg.params)
    return load_jbb_behaviors_with_metadata(params=params)


@register_dataset("xstest")
@register_dataset("xstest@v1")
def _load_xstest(cfg: DatasetConfig) -> DatasetLoadResult:
    params = XSTestParams(**cfg.params)
    return load_xstest_with_metadata(params=params)


@register_dataset("sorrybench_public")
@register_dataset("sorrybench_public@v1")
def _load_sorrybench_public(cfg: DatasetConfig) -> DatasetLoadResult:
    params = SorryBenchPublicParams(**cfg.params)
    return load_sorrybench_public_with_metadata(params=params)


@register_dataset("sorrybench_202406")
@register_dataset("sorrybench_202406@v1")
@register_dataset("sorrybench_202406@v2")
def _load_sorrybench_202406(cfg: DatasetConfig) -> DatasetLoadResult:
    params = SorryBench202406Params(**cfg.params)
    return load_sorrybench_202406_with_metadata(params=params)


def available_datasets() -> list[str]:
    return DATASETS.list()


def load_dataset(cfg: DatasetConfig) -> list[Sample]:
    return load_dataset_with_metadata(cfg, _warning_stacklevel=3).samples


def load_dataset_with_metadata(
    cfg: DatasetConfig,
    *,
    _warning_stacklevel: int = 2,
) -> DatasetLoadResult:
    identity = dataset_identity(cfg.name)
    if identity.legacy_unversioned:
        warnings.warn(
            f"dataset name {cfg.name!r} is unversioned; treating it as {identity.versioned_name!r}",
            DeprecationWarning,
            stacklevel=_warning_stacklevel,
        )
    loaded = DATASETS.get(cfg.name)(cfg)
    if isinstance(loaded, DatasetLoadResult):
        return loaded
    return DatasetLoadResult(samples=loaded)
