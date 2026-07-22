from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

OBSOLETE_ROOT_FILES = {
    Path("src/turnkey/detector_contract.py"),
    Path("src/turnkey/input_manifest.py"),
    Path("src/turnkey/input_provider.py"),
    Path("src/turnkey/state_registry.py"),
    Path("src/turnkey/detector_conformance.py"),
    Path("src/turnkey/runtime_providers/requests.py"),
    Path("src/turnkey/runner/capabilities.py"),
    Path("src/turnkey/runner/provider_declaration_types.py"),
    Path("src/turnkey/runner/provider_declarations.py"),
    Path("src/turnkey/runner/providers.py"),
    Path("src/turnkey/runner/sample_provider_declarations.py"),
    Path("scripts/probe_qwen35_compat.py"),
    Path("scripts/setup_qwen35_uv_env.sh"),
}
TOP_LEVEL_PRIVATE_HELPER_MODULES = {
    Path("src/turnkey/_data.py"),
    Path("src/turnkey/_hf_deps.py"),
    Path("src/turnkey/_io.py"),
    Path("src/turnkey/detector_manifest_contract.py"),
    Path("src/turnkey/hf_compat.py"),
    Path("src/turnkey/redact.py"),
}
INTERNAL_HELPER_MODULES = {
    Path("src/turnkey/_internal/data.py"),
    Path("src/turnkey/_internal/hf_compat.py"),
    Path("src/turnkey/_internal/hf_deps.py"),
    Path("src/turnkey/_internal/io.py"),
    Path("src/turnkey/_internal/redact.py"),
}
OBSOLETE_HF_COMPAT_HELPERS = {"compact_error", "dependency_versions"}
PRIVATE_STRING_LIST_ALLOWLIST = {Path("src/turnkey/lofo_calibration.py")}
REPEATED_NESTED_HELPERS = {"_nested", "_nested_value"}
REPEATED_STRING_SET_HELPERS = {"_string_set"}
PRIVATE_SAFE_DIV_ALLOWLIST = {Path("src/turnkey/analysis/matrix_analysis.py")}
BINARY_AUC_ALLOWLIST = {Path("src/turnkey/components/detectors/rcs/paper.py")}
REPEATED_BINARY_F1_HELPERS = {"_binary_f1"}
DATASET_IMPORT_HELPER_PATHS = (
    Path("src/turnkey/components/datasets/jbb_behaviors.py"),
    Path("src/turnkey/components/datasets/sorrybench_202406.py"),
    Path("src/turnkey/components/datasets/sorrybench_public.py"),
    Path("src/turnkey/components/datasets/xstest.py"),
)
DATASET_PROMPT_ROW_HELPER_PATHS = (
    Path("src/turnkey/components/datasets/sorrybench_202406.py"),
    Path("src/turnkey/components/datasets/sorrybench_public.py"),
    Path("src/turnkey/components/datasets/xstest.py"),
)
HF_CAUSAL_LM_IMPORT_HELPER_PATHS = (
    Path("src/turnkey/runtime_providers/_deps.py"),
    Path("src/turnkey/components/judges/llamaguard.py"),
)
ATTACK_PROMPT_MAP_HELPER_PATHS = (
    Path("src/turnkey/components/attacks/crescendo.py"),
    Path("src/turnkey/components/attacks/tap.py"),
)
GRADSAFE_PROVIDER_HELPERS = {"_grad_norm_from_tensors", "_prompt_and_anchor"}
PROVIDER_DEPS_TOKEN_WRAPPERS = {"_cached_hf_token", "_optional_token"}
HF_ADAPTER_TOKEN_WRAPPER_PATHS = (
    Path("src/turnkey/runtime_providers/gradient.py"),
    Path("src/turnkey/components/judges/llamaguard.py"),
)
POST_RUN_ANALYSIS_MODULE_PATHS = {
    Path("src/turnkey/detector_overlap.py"): Path("src/turnkey/analysis/detector_overlap.py"),
    Path("src/turnkey/matrix_analysis.py"): Path("src/turnkey/analysis/matrix_analysis.py"),
    Path("src/turnkey/report.py"): Path("src/turnkey/analysis/report.py"),
    Path("src/turnkey/threshold_sweep.py"): Path("src/turnkey/analysis/threshold_sweep.py"),
}
DETECTOR_SPECIFIC_TOOL_PATHS = {
    Path("src/turnkey/rcs_text_pool.py"): Path("src/turnkey/components/detectors/rcs/text_pool.py"),
}
MATRIX_PACKAGE_MODULE_PATHS = {
    Path("src/turnkey/matrix/__init__.py"),
    Path("src/turnkey/matrix/plan.py"),
    Path("src/turnkey/matrix/run.py"),
    Path("src/turnkey/matrix/schema.py"),
    Path("src/turnkey/matrix/summary.py"),
    Path("src/turnkey/matrix/results.py"),
}
MATRIX_PLAN_IMPLEMENTATION_FUNCTIONS = {
    "write_matrix_plan",
    "build_matrix_plan",
    "_run_config_for_entry",
    "_validate_component_names",
    "_component_entries",
    "_lofo_spec",
    "_entry_id",
    "_format_template_value",
}
MATRIX_RUN_IMPLEMENTATION_FUNCTIONS = {
    "run_matrix_plan",
    "_run_matrix_entry",
    "_releases_detector_side_resources",
    "_entry_attack_name",
    "_entry_dataset_name",
    "_write_matrix_threshold_sweep",
}
MATRIX_SUMMARY_IMPLEMENTATION_FUNCTIONS = {
    "summarize_matrix_results",
    "merge_matrix_results",
}
INPUTS_PACKAGE_MODULE_PATHS = {
    Path("src/turnkey/inputs/__init__.py"),
    Path("src/turnkey/inputs/manifest.py"),
    Path("src/turnkey/inputs/provider.py"),
}
PIPELINE_STATE_CONTRACT_MODULE_PATH = Path("src/turnkey/pipeline_states.py")
OBSOLETE_STATE_REGISTRY_TEXT = {
    "STATE_PAPER_TRAINING_EXAMPLES",
    "STATE_PROJECTION_STATE",
    "STATE_CALIBRATION_STATE",
    "paper_training_examples",
    "projection_state",
    "calibration_state",
}
MATRIX_ANALYSIS_LOCAL_NUMERIC_HELPERS = {
    "_safe_div",
    "_safe_div_if_positive",
    "_sub_or_none",
    "_binary_f1",
    "_finite_number",
}
MATRIX_ANALYSIS_MARKDOWN_FUNCTIONS = {
    "matrix_analysis_markdown",
    "_group_sample_count",
    "_num",
    "_pct",
    "_pp",
}
MATRIX_ANALYSIS_STATS_FUNCTIONS = {
    "_average_ranks",
    "_kendall_tau_b",
    "_pearson",
    "_ranked_detectors",
    "_sign",
    "_spearman_rho",
}
MATRIX_ANALYSIS_RECORD_FUNCTIONS = {
    "_load_results_document",
    "_load_success_run_records",
    "_lofo_ref",
    "_ref_model_id",
    "_ref_name",
}
NON_NEGATIVE_INT_HELPER_PATHS = (
    Path("src/turnkey/analysis/report.py"),
    Path("src/turnkey/analysis/matrix_analysis.py"),
)
TEXT_HASH_HELPER_PATHS = (
    Path("src/turnkey/refs.py"),
    Path("src/turnkey/components/detectors/rcs/text_pool.py"),
    Path("src/turnkey/_internal/redact.py"),
)
MATRIX_RUNTIME_SPEC_TESTS = {
    "test_matrix_run_can_exclude_attack_entries",
    "test_matrix_run_can_include_dataset_entries",
    "test_matrix_results_can_merge_dataset_shards",
    "test_matrix_run_preserves_non_t0_threat_tier_metrics",
    "test_matrix_run_releases_detector_side_resources_after_heavy_entry",
}
STANDARDIZED_DETECTOR_TESTS = (
    Path("tests/test_allow_all_standardized.py"),
    Path("tests/test_keyword_standardized.py"),
)
JSON_FIXTURE_HELPER_TESTS = (
    Path("tests/test_detector_overlap.py"),
    Path("tests/test_input_manifest.py"),
    Path("tests/test_matrix_analysis.py"),
    Path("tests/test_matrix_plan.py"),
    Path("tests/test_rcs_detector.py"),
)
LOCAL_JSON_FIXTURE_HELPERS = {
    "_load_json",
    "_write_json",
    "_load_jsonl",
    "_write_jsonl",
    "_load_rows",
    "_write_rows",
}
SHARED_JSON_IO_IMPORTS = {
    Path("src/turnkey/inputs/manifest.py"): {"load_json_object"},
    Path("src/turnkey/analysis/matrix_analysis.py"): {
        "write_json_object",
    },
    Path("src/turnkey/analysis/matrix_records.py"): {
        "load_json_object",
        "load_jsonl_objects",
    },
    Path("src/turnkey/analysis/threshold_sweep.py"): {
        "load_json_object",
        "load_jsonl_objects",
        "write_json_object",
    },
    Path("src/turnkey/matrix/plan.py"): {"write_json_object"},
    Path("src/turnkey/matrix/run.py"): {"write_json_object"},
    Path("src/turnkey/matrix/schema.py"): {"load_json_object"},
    Path("src/turnkey/matrix/summary.py"): {"write_json_object"},
    Path("src/turnkey/matrix/results.py"): {"load_json_object"},
}
REPEATED_JSON_IO_FUNCTIONS = {
    "_load_json",
    "_load_jsonl",
    "_load_report",
    "_load_samples",
    "_canonical_json",
}
COMPONENT_PACKAGE_NAMES = {"attacks", "backends", "datasets", "detectors", "judges"}
COMPONENT_IMPLEMENTATION_DIRS = {
    Path(f"src/turnkey/components/{name}") for name in COMPONENT_PACKAGE_NAMES
}
OBSOLETE_COMPONENT_FACADE_DIRS = {
    Path(f"src/turnkey/{name}") for name in COMPONENT_PACKAGE_NAMES
}
RUNTIME_PROVIDER_MODULE_PATHS = {
    Path("src/turnkey/runtime_providers/__init__.py"),
    Path("src/turnkey/runtime_providers/_deps.py"),
    Path("src/turnkey/runtime_providers/_types.py"),
    Path("src/turnkey/runtime_providers/gradient.py"),
    Path("src/turnkey/runtime_providers/hidden_state.py"),
    Path("src/turnkey/runtime_providers/_families/__init__.py"),
    Path("src/turnkey/runtime_providers/_families/internvl.py"),
    Path("src/turnkey/runtime_providers/_families/llava.py"),
    Path("src/turnkey/runtime_providers/_families/qwen.py"),
}
OBSOLETE_PROVIDER_FACADE_PATHS = {
    Path("src/turnkey/providers/__init__.py"),
    Path("src/turnkey/providers/_deps.py"),
    Path("src/turnkey/providers/_types.py"),
    Path("src/turnkey/providers/gradient.py"),
    Path("src/turnkey/providers/hidden_state.py"),
    Path("src/turnkey/providers/requests.py"),
    Path("src/turnkey/providers/_families/__init__.py"),
    Path("src/turnkey/providers/_families/internvl.py"),
    Path("src/turnkey/providers/_families/llava.py"),
    Path("src/turnkey/providers/_families/qwen.py"),
}


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _source_paths(root: str = "src/turnkey") -> list[Path]:
    return sorted(Path(root).rglob("*.py"))


def _top_level_function_names(path: Path) -> set[str]:
    return {
        node.name
        for node in _parse_module(path).body
        if isinstance(node, ast.FunctionDef)
    }


def _top_level_function_offenders(
    paths: Iterable[Path],
    names: set[str],
    *,
    allowlist: set[Path] | None = None,
) -> list[str]:
    allowed = allowlist or set()
    offenders: list[str] = []
    for path in paths:
        if path in allowed:
            continue
        for node in _parse_module(path).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                offenders.append(f"{path}:{node.lineno}: {node.name}")
    return offenders


def _imported_names(module: ast.Module, *, from_module: str | None = None) -> set[str]:
    return {
        alias.name
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        and (from_module is None or node.module == from_module)
        for alias in node.names
    }


def _imported_module_names(module: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _all_exports(module: ast.Module) -> set[str]:
    exports: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        if isinstance(node.value, ast.List):
            exports.update(
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return exports


def test_obsolete_compatibility_files_stay_removed() -> None:
    assert [str(path) for path in sorted(OBSOLETE_ROOT_FILES) if path.exists()] == []


def test_private_helper_modules_live_under_internal_package() -> None:
    assert [str(path) for path in sorted(TOP_LEVEL_PRIVATE_HELPER_MODULES) if path.exists()] == []
    assert [str(path) for path in sorted(INTERNAL_HELPER_MODULES) if not path.exists()] == []


def test_probe_only_hf_compat_helpers_stay_removed() -> None:
    assert OBSOLETE_HF_COMPAT_HELPERS.isdisjoint(
        _top_level_function_names(Path("src/turnkey/_internal/hf_compat.py"))
    )


def test_repeated_string_list_helpers_use_shared_helper() -> None:
    assert _top_level_function_offenders(
        _source_paths(),
        {"_string_list"},
        allowlist=PRIVATE_STRING_LIST_ALLOWLIST,
    ) == []


def test_repeated_nested_value_helpers_use_shared_helper() -> None:
    assert _top_level_function_offenders(_source_paths(), REPEATED_NESTED_HELPERS) == []


def test_repeated_string_set_helpers_use_shared_helper() -> None:
    assert _top_level_function_offenders(_source_paths(), REPEATED_STRING_SET_HELPERS) == []


def test_zero_default_safe_div_helpers_use_shared_helper() -> None:
    assert _top_level_function_offenders(
        _source_paths(),
        {"_safe_div"},
        allowlist=PRIVATE_SAFE_DIV_ALLOWLIST,
    ) == []


def test_rank_sum_binary_auc_uses_shared_helper() -> None:
    assert _top_level_function_offenders(
        _source_paths(),
        {"_binary_auc"},
        allowlist=BINARY_AUC_ALLOWLIST,
    ) == []


def test_binary_f1_helpers_use_shared_helper() -> None:
    assert _top_level_function_offenders(_source_paths(), REPEATED_BINARY_F1_HELPERS) == []


def test_hf_dataset_loaders_use_shared_import_helper() -> None:
    assert _top_level_function_offenders(
        DATASET_IMPORT_HELPER_PATHS,
        {"_import_datasets"},
    ) == []


def test_hf_dataset_prompt_rows_use_shared_helper() -> None:
    assert _top_level_function_offenders(
        DATASET_PROMPT_ROW_HELPER_PATHS,
        {"_row_to_prompt"},
    ) == []


def test_hf_causal_lm_adapters_use_shared_import_helper() -> None:
    assert _top_level_function_offenders(
        HF_CAUSAL_LM_IMPORT_HELPER_PATHS,
        {"_import_hf_dependencies"},
    ) == []


def test_runtime_provider_implementations_live_under_runtime_providers_package() -> None:
    assert [
        str(path)
        for path in sorted(RUNTIME_PROVIDER_MODULE_PATHS)
        if not path.exists()
    ] == []
    assert [
        str(path)
        for path in sorted(OBSOLETE_PROVIDER_FACADE_PATHS)
        if path.exists()
    ] == []


def test_core_package_imports_runtime_providers_not_compat_provider_facade() -> None:
    offenders: list[str] = []
    for path in _source_paths():
        module_names = _imported_module_names(_parse_module(path))
        for module_name in module_names:
            if module_name == "turnkey.providers" or module_name.startswith("turnkey.providers."):
                offenders.append(f"{path}: imports {module_name}")

    assert offenders == []


def test_replay_attack_prompt_maps_use_shared_helper() -> None:
    assert _top_level_function_offenders(
        ATTACK_PROMPT_MAP_HELPER_PATHS,
        {"_load_prompt_map"},
    ) == []


def test_benchmark_component_implementations_live_under_components_package() -> None:
    assert Path("src/turnkey/components/__init__.py").exists()
    assert [
        str(path)
        for path in sorted(COMPONENT_IMPLEMENTATION_DIRS)
        if not path.is_dir()
    ] == []
    assert [
        str(path / "__init__.py")
        for path in sorted(COMPONENT_IMPLEMENTATION_DIRS)
        if not (path / "__init__.py").exists()
    ] == []
    assert [
        str(path)
        for path in sorted(OBSOLETE_COMPONENT_FACADE_DIRS)
        if path.exists()
    ] == []


def test_core_package_imports_components_not_top_level_component_facades() -> None:
    old_prefixes = tuple(f"turnkey.{name}" for name in sorted(COMPONENT_PACKAGE_NAMES))
    offenders: list[str] = []
    for path in _source_paths():
        module_names = _imported_module_names(_parse_module(path))
        for module_name in module_names:
            if module_name in old_prefixes or module_name.startswith((*old_prefixes,)):
                offenders.append(f"{path}: imports {module_name}")

    assert offenders == []


def test_gradsafe_reuses_gradient_provider_helpers() -> None:
    assert GRADSAFE_PROVIDER_HELPERS.isdisjoint(
        _top_level_function_names(Path("src/turnkey/components/detectors/gradsafe.py"))
    )


def test_gradsafe_sample_average_helpers_delegate_to_gradient_iterators() -> None:
    module = _parse_module(Path("src/turnkey/components/detectors/gradsafe.py"))
    helper_names = {"_average_gradients_for_samples", "_average_reference_cosines_for_samples"}
    offenders: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in helper_names:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in {"as_tensor", "cosine_similarity"}:
                offenders.append(f"{node.name}:{child.lineno}: {child.attr}")

    assert offenders == []


def test_provider_deps_do_not_wrap_hf_token_resolution() -> None:
    assert PROVIDER_DEPS_TOKEN_WRAPPERS.isdisjoint(
        _top_level_function_names(Path("src/turnkey/runtime_providers/_deps.py"))
    )


def test_hf_adapters_use_shared_token_resolution() -> None:
    assert _top_level_function_offenders(
        HF_ADAPTER_TOKEN_WRAPPER_PATHS,
        PROVIDER_DEPS_TOKEN_WRAPPERS,
    ) == []


def test_post_run_analysis_modules_live_under_analysis_package() -> None:
    assert [str(path) for path in POST_RUN_ANALYSIS_MODULE_PATHS if path.exists()] == []
    assert [
        str(path)
        for path in POST_RUN_ANALYSIS_MODULE_PATHS.values()
        if not path.exists()
    ] == []


def test_detector_specific_tools_live_with_detector_package() -> None:
    assert [str(path) for path in DETECTOR_SPECIFIC_TOOL_PATHS if path.exists()] == []
    assert [
        str(path)
        for path in DETECTOR_SPECIFIC_TOOL_PATHS.values()
        if not path.exists()
    ] == []


def test_matrix_runtime_lives_in_matrix_package() -> None:
    assert not Path("src/turnkey/matrix.py").exists()
    assert [str(path) for path in sorted(MATRIX_PACKAGE_MODULE_PATHS) if not path.exists()] == []


def test_matrix_plan_implementation_lives_outside_facade() -> None:
    facade = Path("src/turnkey/matrix/__init__.py")
    facade_module = _parse_module(facade)
    assert MATRIX_PLAN_IMPLEMENTATION_FUNCTIONS.isdisjoint(_top_level_function_names(facade))
    assert {"build_matrix_plan", "write_matrix_plan"}.issubset(
        _imported_names(facade_module, from_module="turnkey.matrix.plan")
    )


def test_matrix_run_and_summary_implementation_lives_outside_facade() -> None:
    facade = Path("src/turnkey/matrix/__init__.py")
    facade_module = _parse_module(facade)
    facade_functions = _top_level_function_names(facade)
    assert MATRIX_RUN_IMPLEMENTATION_FUNCTIONS.isdisjoint(facade_functions)
    assert MATRIX_SUMMARY_IMPLEMENTATION_FUNCTIONS.isdisjoint(facade_functions)
    assert {"MATRIX_THRESHOLD_GRID", "run_matrix_plan"}.issubset(
        _imported_names(facade_module, from_module="turnkey.matrix.run")
    )
    assert {"merge_matrix_results", "summarize_matrix_results"}.issubset(
        _imported_names(facade_module, from_module="turnkey.matrix.summary")
    )


def test_input_runtime_lives_in_inputs_package() -> None:
    assert not Path("src/turnkey/input_manifest.py").exists()
    assert not Path("src/turnkey/input_provider.py").exists()
    assert [str(path) for path in sorted(INPUTS_PACKAGE_MODULE_PATHS) if not path.exists()] == []

    offenders: list[str] = []
    for path in [*_source_paths(), *sorted(Path("tests").rglob("*.py"))]:
        if path == Path("tests/ci/test_conformance_architecture.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if "turnkey.input_manifest" in text or "turnkey.input_provider" in text:
            offenders.append(str(path))

    assert offenders == []


def test_pipeline_state_contracts_live_in_pipeline_states_module() -> None:
    assert PIPELINE_STATE_CONTRACT_MODULE_PATH.exists()
    assert not Path("src/turnkey/state_registry.py").exists()

    offenders: list[str] = []
    for path in [*_source_paths(), *sorted(Path("tests").rglob("*.py"))]:
        if path == Path("tests/ci/test_conformance_architecture.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if "turnkey.state_registry" in text:
            offenders.append(str(path))

    assert offenders == []


def test_detector_specific_diagnostic_states_stay_out_of_global_pipeline_states() -> None:
    text = PIPELINE_STATE_CONTRACT_MODULE_PATH.read_text(encoding="utf-8")
    assert [item for item in sorted(OBSOLETE_STATE_REGISTRY_TEXT) if item in text] == []


def test_pipeline_state_contract_stays_minimal() -> None:
    text = PIPELINE_STATE_CONTRACT_MODULE_PATH.read_text(encoding="utf-8")

    assert len(text.splitlines()) < 50
    assert "ProviderSpec" not in text
    assert "ManifestProviderContract" not in text


def test_matrix_analysis_uses_shared_optional_numeric_helpers() -> None:
    assert _top_level_function_offenders(
        [Path("src/turnkey/analysis/matrix_analysis.py")],
        MATRIX_ANALYSIS_LOCAL_NUMERIC_HELPERS,
    ) == []


def test_matrix_analysis_markdown_rendering_lives_in_render_module() -> None:
    facade = Path("src/turnkey/analysis/matrix_analysis.py")
    render_module = Path("src/turnkey/analysis/matrix_markdown.py")
    assert render_module.exists()
    assert _top_level_function_offenders([facade], MATRIX_ANALYSIS_MARKDOWN_FUNCTIONS) == []
    assert {"matrix_analysis_markdown"}.issubset(
        _imported_names(_parse_module(facade), from_module="turnkey.analysis.matrix_markdown")
    )


def test_matrix_analysis_ranking_stats_live_in_stats_module() -> None:
    facade = Path("src/turnkey/analysis/matrix_analysis.py")
    stats_module = Path("src/turnkey/analysis/matrix_stats.py")
    assert stats_module.exists()
    assert _top_level_function_offenders([facade], MATRIX_ANALYSIS_STATS_FUNCTIONS) == []
    assert {"_kendall_tau_b", "_ranked_detectors", "_spearman_rho"}.issubset(
        _imported_names(_parse_module(facade), from_module="turnkey.analysis.matrix_stats")
    )


def test_matrix_analysis_record_loading_lives_in_records_module() -> None:
    facade = Path("src/turnkey/analysis/matrix_analysis.py")
    records_module = Path("src/turnkey/analysis/matrix_records.py")
    assert records_module.exists()
    assert _top_level_function_offenders([facade], MATRIX_ANALYSIS_RECORD_FUNCTIONS) == []
    assert {
        "_load_results_document",
        "_load_success_run_records",
        "_lofo_ref",
        "_ref_name",
    }.issubset(_imported_names(_parse_module(facade), from_module="turnkey.analysis.matrix_records"))


def test_analysis_modules_use_shared_non_negative_int_helper() -> None:
    assert _top_level_function_offenders(
        NON_NEGATIVE_INT_HELPER_PATHS,
        {"_non_negative_int"},
    ) == []


def test_utf8_text_hashing_uses_shared_helper() -> None:
    assert _top_level_function_offenders(
        TEXT_HASH_HELPER_PATHS,
        {"_sha256_text", "sha256_hex"},
    ) == []


def test_script_module_tests_use_shared_loader() -> None:
    offenders: list[str] = []
    for path in sorted(Path("tests").rglob("test_*.py")):
        module = _parse_module(path)
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_load_paper_module":
                offenders.append(f"{path}:{node.lineno}: local _load_paper_module")
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "insert":
                continue
            if not isinstance(node.func.value, ast.Attribute) or node.func.value.attr != "path":
                continue
            if not isinstance(node.func.value.value, ast.Name) or node.func.value.value.id != "sys":
                continue
            offenders.append(f"{path}:{node.lineno}: sys.path.insert")

    assert offenders == []


def test_matrix_runtime_tests_share_spec_builder() -> None:
    module = _parse_module(Path("tests/test_matrix_plan.py"))
    offenders: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in MATRIX_RUNTIME_SPEC_TESTS:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and child.value == "turnkey_matrix_spec/v1":
                offenders.append(f"{node.name}:{child.lineno}")

    assert offenders == []


def test_matrix_plan_tests_share_yaml_spec_writer() -> None:
    module = _parse_module(Path("tests/test_matrix_plan.py"))
    safe_dump_calls = [
        node.lineno
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "safe_dump"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "yaml"
    ]

    assert len(safe_dump_calls) == 1, safe_dump_calls


def test_standardized_detector_tests_do_not_redeclare_signal_requirements() -> None:
    offenders: list[str] = []
    for path in STANDARDIZED_DETECTOR_TESTS:
        module = _parse_module(path)
        for node in ast.walk(module):
            if isinstance(node, ast.Constant) and node.value == "prompt_logprobs":
                offenders.append(f"{path}:{node.lineno}")

    assert offenders == []


def test_json_fixture_tests_share_file_helpers() -> None:
    assert _top_level_function_offenders(
        JSON_FIXTURE_HELPER_TESTS,
        LOCAL_JSON_FIXTURE_HELPERS,
    ) == []


def test_core_json_path_io_uses_shared_helpers() -> None:
    offenders: list[str] = []
    for path, required_imports in SHARED_JSON_IO_IMPORTS.items():
        module = _parse_module(path)
        imported_names = _imported_names(module, from_module="turnkey._internal.io")
        missing_imports = sorted(required_imports - imported_names)
        if missing_imports:
            offenders.append(f"{path}: missing shared JSON IO imports {missing_imports}")
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name in REPEATED_JSON_IO_FUNCTIONS:
                offenders.append(f"{path}:{node.lineno}: local {node.name}")

    assert offenders == []


def test_runner_facade_does_not_export_concrete_runtime_providers() -> None:
    module = _parse_module(Path("src/turnkey/runner/__init__.py"))
    concrete_exports = {
        "HFGradientScoreProvider",
        "HFLastTokenHiddenStateProvider",
        "backend_capabilities_dict",
    }
    assert concrete_exports.isdisjoint(_imported_names(module))
    assert concrete_exports.isdisjoint(_all_exports(module))


def test_gradsafe_detector_does_not_own_hf_scorer_state() -> None:
    module = _parse_module(Path("src/turnkey/components/detectors/gradsafe.py"))
    detector = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "GradSafeDetector"
    )
    forbidden_methods = {
        "_build_inputs",
        "_build_llama2_inst_inputs",
        "_build_qwen_chat_inputs",
        "_gradient_tensors",
        "_load_model",
        "_load_reference_artifact",
        "_score_prompt",
        "_selected_gradients",
        "_torch_dtype",
    }
    methods = {
        node.name
        for node in detector.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    annotated_fields = {
        node.target.id
        for node in detector.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert forbidden_methods.isdisjoint(methods)
    assert {"_model", "_tokenizer", "_device", "_parameter_pattern"}.isdisjoint(
        annotated_fields
    )


def test_backend_facade_does_not_import_concrete_backend_classes() -> None:
    module = _parse_module(Path("src/turnkey/components/backends/__init__.py"))
    concrete_imports = {
        "DummyBackend",
        "HFBackend",
        "OpenAICompatBackend",
    }
    assert concrete_imports.isdisjoint(_imported_names(module))


def test_legacy_manifest_contract_helper_stays_removed() -> None:
    assert not Path("src/turnkey/_internal/detector_manifest_contract.py").exists()
