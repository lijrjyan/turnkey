from pathlib import Path
import hashlib
import importlib
import json
import os
import sys
import textwrap

import pytest

from turnkey.component_loader import resolve_component
from turnkey.policy import Component, PolicyChain


def test_resolve_component_builds_builtin_alias_with_effective_parameters() -> None:
    resolved = resolve_component("keyword", params={"keywords": ["deny"]})

    assert resolved.component.name == "keyword"
    assert resolved.component.parameters == {"keywords": ["deny"]}
    assert resolved.source == {"kind": "builtin", "name": "keyword"}


def test_resolve_component_calls_external_builder_with_configured_parameters(
    tmp_path: Path,
) -> None:
    method_file = tmp_path / "external_method.py"
    method_file.write_text(
        textwrap.dedent(
            """
            from turnkey.policy import Component, PolicyChain


            def build(*, threshold):
                return Component(
                    name="external-threshold",
                    policy=PolicyChain(),
                    parameters={"threshold": threshold},
                )
            """
        ).lstrip(),
        encoding="utf-8",
    )

    resolved = resolve_component(
        f"{method_file}:build",
        params={"threshold": 0.75},
    )

    assert resolved.component.name == "external-threshold"
    assert resolved.component.parameters == {"threshold": 0.75}
    assert resolved.source["kind"] == "file"
    assert resolved.source["path"] == str(method_file)
    assert resolved.source["object"] == "build"
    assert len(resolved.source["sha256"]) == 64


def test_resolve_component_accepts_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_file = tmp_path / "installed_method.py"
    module_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='installed', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    resolved = resolve_component("installed_method:build")

    assert resolved.component.name == "installed"
    assert resolved.source["kind"] == "module"
    assert resolved.source["module"] == "installed_method"
    assert resolved.source["object"] == "build"


def test_module_entrypoint_uses_actual_shadowing_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "shadowed_component_method"
    distribution_root = tmp_path / "distribution"
    shadow_root = tmp_path / "shadow"
    distribution_root.mkdir()
    shadow_root.mkdir()
    (distribution_root / f"{module_name}.py").write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='distribution', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    dist_info = distribution_root / "shadowed_component_method-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: shadowed-component-method\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text(f"{module_name}\n", encoding="utf-8")
    (dist_info / "RECORD").write_text(
        f"{module_name}.py,,\n"
        "shadowed_component_method-1.0.dist-info/METADATA,,\n"
        "shadowed_component_method-1.0.dist-info/top_level.txt,,\n"
        "shadowed_component_method-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    shadow_file = shadow_root / f"{module_name}.py"
    shadow_source = (
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='shadow-local', policy=PolicyChain())\n"
    )
    shadow_file.write_text(shadow_source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(distribution_root))
    monkeypatch.syspath_prepend(str(shadow_root))
    importlib.invalidate_caches()

    resolved = resolve_component(f"{module_name}:build")

    assert resolved.component.name == "shadow-local"
    assert resolved.source["path"] == str(shadow_file)
    assert resolved.source["sha256"] == hashlib.sha256(
        shadow_source.encode("utf-8")
    ).hexdigest()
    assert "distribution" not in resolved.source


def test_editable_module_identity_rejects_paths_outside_declared_import_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "editable_shadow_method"
    editable_root = tmp_path / "editable"
    source_root = editable_root / "src"
    shadow_root = source_root / "shadow"
    site_root = editable_root / "site"
    source_root.mkdir(parents=True)
    shadow_root.mkdir()
    site_root.mkdir()
    (source_root / f"{module_name}.py").write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='distribution', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    shadow_file = shadow_root / f"{module_name}.py"
    shadow_source = (
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='shadow-local', policy=PolicyChain())\n"
    )
    shadow_file.write_text(shadow_source, encoding="utf-8")
    pth_name = "__editable__.editable_shadow_method-1.0.pth"
    (site_root / pth_name).write_text(f"{source_root}\n", encoding="utf-8")
    dist_info = site_root / "editable_shadow_method-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: editable-shadow-method\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text(f"{module_name}\n", encoding="utf-8")
    (dist_info / "direct_url.json").write_text(
        json.dumps(
            {"url": editable_root.as_uri(), "dir_info": {"editable": True}}
        ),
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text(
        f"{pth_name},,\n"
        "editable_shadow_method-1.0.dist-info/METADATA,,\n"
        "editable_shadow_method-1.0.dist-info/top_level.txt,,\n"
        "editable_shadow_method-1.0.dist-info/direct_url.json,,\n"
        "editable_shadow_method-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(site_root))
    monkeypatch.syspath_prepend(str(source_root))
    monkeypatch.syspath_prepend(str(shadow_root))
    importlib.invalidate_caches()

    resolved = resolve_component(f"{module_name}:build")

    assert resolved.component.name == "shadow-local"
    assert resolved.source["path"] == str(shadow_file)
    assert resolved.source["sha256"] == hashlib.sha256(
        shadow_source.encode("utf-8")
    ).hexdigest()
    assert "distribution" not in resolved.source


def test_resolve_component_requires_builder_for_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "module-component-cleaned"
    module_file = tmp_path / "installed_concrete.py"
    module_file.write_text(
        "from pathlib import Path\n"
        "from turnkey.policy import Component, PolicyChain\n"
        f"component = Component(name='installed', policy=PolicyChain(), "
        f"cleanup=lambda: Path({str(marker)!r}).write_text('cleaned', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="process-cached Component"):
        resolve_component("installed_concrete:component")

    assert marker.read_text(encoding="utf-8") == "cleaned"


def test_module_entrypoint_identity_changes_with_local_source_across_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "changing_installed_method"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='first-aa', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    first = resolve_component(f"{module_name}:build")
    original_stat = module_file.stat()
    sys.modules.pop(module_name, None)
    module_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='later-bb', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    os.utime(
        module_file,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    importlib.invalidate_caches()

    second = resolve_component(f"{module_name}:build")

    assert second.component.name == "later-bb"
    assert second.source["sha256"] != first.source["sha256"]


def test_resolve_component_rejects_unverifiable_preloaded_local_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "preloaded_local_method"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='preloaded', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.import_module(module_name)

    with pytest.raises(RuntimeError, match="imported before Turnkey"):
        resolve_component(f"{module_name}:build")


def test_resolve_component_reloads_changed_file_before_recording_source_identity(
    tmp_path: Path,
) -> None:
    method_file = tmp_path / "changing_method.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "component = Component(name='first', policy=PolicyChain())\n",
        encoding="utf-8",
    )
    first = resolve_component(f"{method_file}:component")
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "component = Component(name='second', policy=PolicyChain())\n",
        encoding="utf-8",
    )

    second = resolve_component(f"{method_file}:component")

    assert second.component.name == "second"
    assert second.source["sha256"] != first.source["sha256"]


def test_resolve_component_records_the_exact_file_bytes_that_were_executed(
    tmp_path: Path,
) -> None:
    method_file = tmp_path / "self_modifying_method.py"
    source = (
        "from pathlib import Path\n"
        "from turnkey.policy import Component, PolicyChain\n"
        "Path(__file__).write_text('# changed after load\\n', encoding='utf-8')\n"
        "component = Component(name='loaded', policy=PolicyChain())\n"
    )
    method_file.write_text(source, encoding="utf-8")

    resolved = resolve_component(f"{method_file}:component")

    assert resolved.component.name == "loaded"
    assert resolved.source["sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert resolved.source["bytes"] == len(source.encode("utf-8"))


def test_resolve_component_rebuilds_concrete_file_component_for_each_resolution(
    tmp_path: Path,
) -> None:
    method_file = tmp_path / "concrete_method.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "component = Component(name='concrete', policy=PolicyChain())\n",
        encoding="utf-8",
    )

    first = resolve_component(f"{method_file}:component")
    second = resolve_component(f"{method_file}:component")

    assert second.component is not first.component


def test_resolve_component_rejects_parameters_for_concrete_component(tmp_path: Path) -> None:
    method_file = tmp_path / "concrete_method.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "component = Component(name='concrete', policy=PolicyChain())\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot accept configured params"):
        resolve_component(f"{method_file}:component", params={"threshold": 0.5})


def test_resolve_component_rejects_non_component_result(tmp_path: Path) -> None:
    method_file = tmp_path / "invalid_method.py"
    method_file.write_text("def build():\n    return object()\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must resolve to a Component"):
        resolve_component(f"{method_file}:build")


def test_resolve_component_rejects_unpersisted_builder_parameters(tmp_path: Path) -> None:
    method_file = tmp_path / "incomplete_method.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build(*, factory_seed):\n"
        "    return Component(name='incomplete', policy=PolicyChain(), parameters={})\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not persist configured parameter keys"):
        resolve_component(f"{method_file}:build", params={"factory_seed": 7})


def test_resolve_component_preserves_validation_error_when_rejected_cleanup_fails(
    tmp_path: Path,
) -> None:
    method_file = tmp_path / "cleanup_failure.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def fail_cleanup():\n    raise OSError('cleanup failed')\n"
        "def build(*, factory_seed):\n"
        "    return Component(name='incomplete', policy=PolicyChain(), "
        "parameters={}, cleanup=fail_cleanup)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not persist") as error:
        resolve_component(f"{method_file}:build", params={"factory_seed": 7})

    assert error.value.__turnkey_secondary_failures__[0][0] == "rejected component cleanup"


def test_resolve_component_rejects_unknown_short_name() -> None:
    with pytest.raises(ValueError, match="unknown component alias"):
        resolve_component("missing_detector")


def test_resolve_component_rejects_external_calibration_artifact(tmp_path: Path) -> None:
    method_file = tmp_path / "external_method.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build():\n    return Component(name='external', policy=PolicyChain())\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="external components do not support calibration artifacts"):
        resolve_component(f"{method_file}:build", calibration_artifact=object())


def test_component_canonicalizes_external_parameters_at_the_runtime_boundary() -> None:
    component = Component(
        name="external",
        policy=PolicyChain(),
        parameters={"labels": ("safe", "unsafe"), "nested": {"threshold": 0.5}},
    )

    assert component.parameters == {
        "labels": ["safe", "unsafe"],
        "nested": {"threshold": 0.5},
    }


def test_component_rejects_noncanonical_external_parameters() -> None:
    with pytest.raises(TypeError, match="string keys"):
        Component(name="external", policy=PolicyChain(), parameters={"nested": {1: True}})

    with pytest.raises(TypeError, match="JSON-serializable"):
        Component(name="external", policy=PolicyChain(), parameters={"threshold": float("nan")})

    with pytest.raises(TypeError, match="cleanup must be callable"):
        Component(name="external", policy=PolicyChain(), cleanup="close")  # type: ignore[arg-type]
