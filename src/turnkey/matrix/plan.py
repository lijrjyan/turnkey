from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import yaml

from turnkey._internal.io import write_json_object
from turnkey.components.attacks import available_attacks
from turnkey.components.backends import available_backends
from turnkey.components.datasets import available_datasets
from turnkey.components.detectors import available_detectors
from turnkey.components.judges import available_judges
from turnkey.matrix.schema import MATRIX_PLAN_SCHEMA, load_matrix_spec
from turnkey.methods import is_entrypoint

def write_matrix_plan(*, spec_path: str | Path, out_dir: str | Path) -> Path:
    spec_path = Path(spec_path)
    out_dir = Path(out_dir)
    spec = load_matrix_spec(spec_path)
    plan, configs = build_matrix_plan(spec=spec, spec_path=spec_path, out_dir=out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, config in configs.items():
        path = out_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    plan_path = out_dir / "plan.json"
    write_json_object(plan_path, plan)
    return plan_path


def build_matrix_plan(
    *,
    spec: dict[str, Any],
    spec_path: str | Path,
    out_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    spec_path = Path(spec_path)
    out_dir = Path(out_dir)
    name = _required_string(spec, "name")
    defaults = _mapping(spec.get("defaults"), "defaults")
    datasets = _component_entries(spec.get("datasets"), "datasets")
    attacks = _component_entries(spec.get("attacks"), "attacks")
    detectors = _component_entries(spec.get("detectors"), "detectors")
    skip_rules = _skip_rules(spec.get("skip_rules", []))
    lofo = _lofo_spec(spec.get("lofo"), attacks)
    attack_entries = _lofo_attack_entries(attacks=attacks, lofo=lofo)
    _validate_component_names(defaults=defaults, datasets=datasets, attacks=attacks, detectors=detectors)

    entries: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for dataset in datasets:
        for attack in attack_entries:
            lofo_fold = _lofo_fold(lofo, holdout_attack=attack["name"])
            for detector in detectors:
                entry_id = _entry_id(
                    matrix_name=name,
                    dataset=dataset["name"],
                    attack=attack["name"],
                    detector=detector["name"],
                    fold=lofo_fold["holdout_attack"] if lofo_fold else None,
                )
                if entry_id in seen_ids:
                    raise ValueError(f"duplicate matrix entry id: {entry_id}")
                seen_ids.add(entry_id)

                skip_reason = _skip_reason(
                    skip_rules=skip_rules,
                    dataset=dataset,
                    attack=attack,
                    detector=detector,
                )
                resource_tier = str(detector.get("resource_tier", "unspecified"))
                entry: dict[str, Any] = {
                    "id": entry_id,
                    "status": "skipped" if skip_reason else "planned",
                    "reason": skip_reason,
                    "dataset": _entry_ref(dataset),
                    "attack": _entry_ref(attack),
                    "detector": _entry_detector_ref(detector, matrix_name=name, lofo_fold=lofo_fold),
                    "judge": _judge_ref(defaults),
                    "model": _model_ref(defaults),
                    "resource_tier": resource_tier,
                }
                if lofo_fold is not None:
                    entry["lofo"] = lofo_fold
                if skip_reason is None:
                    rel_config_path = f"configs/{entry_id}.yaml"
                    config = _run_config_for_entry(
                        defaults=defaults,
                        dataset=dataset,
                        attack=attack,
                        detector=detector,
                        matrix_name=name,
                        run_name=entry_id,
                        lofo_fold=lofo_fold,
                    )
                    configs[rel_config_path] = config
                    entry.update(
                        {
                            "config_path": rel_config_path,
                            "command": ["turnkey", "run", "--config", rel_config_path],
                            "audit_command": ["turnkey", "audit", "<run_dir>"],
                        }
                    )
                entries.append(entry)

    planned = sum(1 for entry in entries if entry["status"] == "planned")
    skipped = sum(1 for entry in entries if entry["status"] == "skipped")
    plan = {
        "schema_version": MATRIX_PLAN_SCHEMA,
        "name": name,
        "spec_path": str(spec_path),
        "plan_dir": str(out_dir),
        "counts": {
            "entries": len(entries),
            "planned": planned,
            "skipped": skipped,
        },
        "entries": entries,
    }
    return plan, configs


def _run_config_for_entry(
    *,
    defaults: dict[str, Any],
    dataset: dict[str, Any],
    attack: dict[str, Any],
    detector: dict[str, Any],
    matrix_name: str,
    run_name: str,
    lofo_fold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_defaults = _mapping(defaults.get("run"), "defaults.run")
    content_defaults = _mapping(defaults.get("content"), "defaults.content")
    config = {
        "run": _deep_merge(run_defaults, {"name": run_name}),
        "dataset": {"name": dataset["name"], "params": deepcopy(dataset.get("params", {}))},
        "content": _deep_merge(content_defaults, _mapping(dataset.get("content"), f"datasets.{dataset['name']}.content")),
        "attack": {"name": attack["name"], "params": deepcopy(attack.get("params", {}))},
        "model": deepcopy(_mapping(defaults.get("model"), "defaults.model")),
        "nsg": deepcopy(_mapping(defaults.get("nsg"), "defaults.nsg")),
        "detector": _detector_config_ref(detector, matrix_name=matrix_name, lofo_fold=lofo_fold),
        "judge": deepcopy(_mapping(defaults.get("judge"), "defaults.judge")),
    }
    naturalness = deepcopy(_mapping(defaults.get("naturalness"), "defaults.naturalness"))
    if naturalness:
        config["naturalness"] = naturalness
    reproduction = _deep_merge(
        _mapping(defaults.get("reproduction"), "defaults.reproduction"),
        _mapping(detector.get("reproduction"), f"detectors.{detector['name']}.reproduction"),
    )
    if lofo_fold is not None:
        notes = [item for item in reproduction.get("notes", []) if isinstance(item, str)]
        notes.append(
            "LOFO holdout_attack={holdout}; train_attacks={train}".format(
                holdout=lofo_fold["holdout_attack"],
                train=",".join(lofo_fold["train_attacks"]),
            )
        )
        reproduction["notes"] = notes
    if reproduction:
        config["reproduction"] = reproduction
    return config


def _validate_component_names(
    *,
    defaults: dict[str, Any],
    datasets: list[dict[str, Any]],
    attacks: list[dict[str, Any]],
    detectors: list[dict[str, Any]],
) -> None:
    _validate_names("datasets", [entry["name"] for entry in datasets], available_datasets())
    _validate_names("attacks", [entry["name"] for entry in attacks], available_attacks())
    _validate_names(
        "detectors",
        [entry["name"] for entry in detectors],
        available_detectors(),
        allow_entrypoints=True,
    )

    judge = _mapping(defaults.get("judge"), "defaults.judge")
    judge_name = judge.get("name")
    if isinstance(judge_name, str):
        _validate_names("defaults.judge.name", [judge_name], available_judges())

    nsg = _mapping(defaults.get("nsg"), "defaults.nsg")
    baseline_detector = _mapping(nsg.get("baseline_detector"), "defaults.nsg.baseline_detector")
    baseline_name = baseline_detector.get("name")
    if isinstance(baseline_name, str):
        _validate_names(
            "defaults.nsg.baseline_detector.name",
            [baseline_name],
            available_detectors(),
            allow_entrypoints=True,
        )

    model = _mapping(defaults.get("model"), "defaults.model")
    backend = model.get("backend")
    if isinstance(backend, str):
        _validate_names("defaults.model.backend", [backend], available_backends())


def _validate_names(
    section: str,
    names: list[str],
    available: list[str],
    *,
    allow_entrypoints: bool = False,
) -> None:
    allowed = set(available)
    unknown = sorted(
        {
            name
            for name in names
            if name not in allowed and not (allow_entrypoints and is_entrypoint(name))
        }
    )
    if unknown:
        raise ValueError(f"{section}: unknown names: {unknown}; available={sorted(allowed)}")


def _component_entries(value: Any, section: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{section}: must be a non-empty list")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            entry = {"name": item, "params": {}}
        elif isinstance(item, dict):
            entry = deepcopy(item)
        else:
            raise TypeError(f"{section}[{index}]: must be string or mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{section}[{index}].name: must be non-empty string")
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"{section}[{index}].params: must be mapping")
        entry["params"] = params
        entries.append(entry)
    return entries


def _skip_rules(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("skip_rules: must be a list")
    result: list[dict[str, Any]] = []
    for index, rule in enumerate(value):
        if not isinstance(rule, dict):
            raise TypeError(f"skip_rules[{index}]: must be mapping")
        reason = rule.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"skip_rules[{index}].reason: must be non-empty string")
        result.append(deepcopy(rule))
    return result


def _lofo_spec(value: Any, attacks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("lofo: must be a mapping")
    if not bool(value.get("enabled", False)):
        return None

    attack_names = [attack["name"] for attack in attacks]
    holdout_raw = value.get("holdout_attacks", value.get("holdout_families", attack_names))
    if not isinstance(holdout_raw, list) or not holdout_raw:
        raise ValueError("lofo.holdout_attacks: must be a non-empty list")
    holdout_attacks = []
    for index, item in enumerate(holdout_raw):
        if not isinstance(item, str) or not item:
            raise ValueError(f"lofo.holdout_attacks[{index}]: must be a non-empty string")
        holdout_attacks.append(item)

    unknown = sorted(set(holdout_attacks) - set(attack_names))
    if unknown:
        raise ValueError(f"lofo.holdout_attacks: unknown attacks: {unknown}; available={attack_names}")

    train_raw = value.get("train_attacks", attack_names)
    if not isinstance(train_raw, list) or not train_raw:
        raise ValueError("lofo.train_attacks: must be a non-empty list when provided")
    train_attacks = []
    for index, item in enumerate(train_raw):
        if not isinstance(item, str) or not item:
            raise ValueError(f"lofo.train_attacks[{index}]: must be a non-empty string")
        train_attacks.append(item)
    unknown_train = sorted(set(train_attacks) - set(attack_names))
    if unknown_train:
        raise ValueError(f"lofo.train_attacks: unknown attacks: {unknown_train}; available={attack_names}")

    return {
        "mode": "leave_one_family_out",
        "holdout_attacks": holdout_attacks,
        "train_attacks": train_attacks,
    }


def _lofo_attack_entries(*, attacks: list[dict[str, Any]], lofo: dict[str, Any] | None) -> list[dict[str, Any]]:
    if lofo is None:
        return attacks
    holdouts = set(lofo["holdout_attacks"])
    return [attack for attack in attacks if attack["name"] in holdouts]


def _lofo_fold(lofo: dict[str, Any] | None, *, holdout_attack: str) -> dict[str, Any] | None:
    if lofo is None:
        return None
    train_attacks = [attack for attack in lofo["train_attacks"] if attack != holdout_attack]
    if not train_attacks:
        raise ValueError(f"lofo fold {holdout_attack!r}: train_attacks cannot be empty after holdout")
    return {
        "mode": "leave_one_family_out",
        "holdout_attack": holdout_attack,
        "train_attacks": train_attacks,
    }


def _skip_reason(
    *,
    skip_rules: list[dict[str, Any]],
    dataset: dict[str, Any],
    attack: dict[str, Any],
    detector: dict[str, Any],
) -> str | None:
    if isinstance(detector.get("skip_reason"), str) and detector["skip_reason"]:
        return detector["skip_reason"]
    for rule in skip_rules:
        if (
            _rule_matches(rule.get("dataset"), dataset["name"])
            and _rule_matches(rule.get("attack"), attack["name"])
            and _rule_matches(rule.get("detector"), detector["name"])
            and _rule_matches(rule.get("resource_tier"), detector.get("resource_tier"))
        ):
            return str(rule["reason"])
    return None


def _rule_matches(pattern: Any, value: Any) -> bool:
    if pattern is None or pattern == "*":
        return True
    if isinstance(pattern, list):
        return any(_rule_matches(item, value) for item in pattern)
    return str(pattern) == str(value)


def _entry_id(*, matrix_name: str, dataset: str, attack: str, detector: str, fold: str | None = None) -> str:
    suffix = f"-lofo-{fold}" if fold else ""
    return _slug(f"{matrix_name}-{dataset}-{attack}-{detector}{suffix}")


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    if not result:
        raise ValueError("matrix entry id cannot be empty")
    return result


def _entry_ref(entry: dict[str, Any]) -> dict[str, Any]:
    result = {"name": entry["name"]}
    params = entry.get("params")
    if isinstance(params, dict) and params:
        result["params"] = deepcopy(params)
    calibration_artifact = entry.get("calibration_artifact")
    if isinstance(calibration_artifact, str) and calibration_artifact:
        result["calibration_artifact"] = calibration_artifact
    return result


def _entry_detector_ref(
    detector: dict[str, Any],
    *,
    matrix_name: str,
    lofo_fold: dict[str, Any] | None,
) -> dict[str, Any]:
    result = _entry_ref(detector)
    params = result.get("params")
    if isinstance(params, dict):
        result["params"] = _format_template_value(
            params,
            values=_detector_template_values(
                detector,
                matrix_name=matrix_name,
                lofo_fold=lofo_fold,
            ),
            loc=f"detectors.{detector['name']}.params",
        )
    calibration_artifact = _detector_calibration_artifact(
        detector,
        matrix_name=matrix_name,
        lofo_fold=lofo_fold,
    )
    if calibration_artifact is not None:
        result["calibration_artifact"] = calibration_artifact
    return result


def _detector_config_ref(
    detector: dict[str, Any],
    *,
    matrix_name: str,
    lofo_fold: dict[str, Any] | None,
) -> dict[str, Any]:
    values = _detector_template_values(detector, matrix_name=matrix_name, lofo_fold=lofo_fold)
    result = {
        "name": detector["name"],
        "params": _format_template_value(
            deepcopy(detector.get("params", {})),
            values=values,
            loc=f"detectors.{detector['name']}.params",
        ),
    }
    calibration_artifact = _detector_calibration_artifact(
        detector,
        matrix_name=matrix_name,
        lofo_fold=lofo_fold,
    )
    if calibration_artifact is not None:
        result["calibration_artifact"] = calibration_artifact
    return result


def _detector_calibration_artifact(
    detector: dict[str, Any],
    *,
    matrix_name: str,
    lofo_fold: dict[str, Any] | None,
) -> str | None:
    calibration_artifact = detector.get("calibration_artifact")
    if isinstance(calibration_artifact, str) and calibration_artifact:
        return calibration_artifact
    template = detector.get("calibration_artifact_template")
    if not isinstance(template, str) or not template:
        return None
    values = _detector_template_values(detector, matrix_name=matrix_name, lofo_fold=lofo_fold)
    try:
        return template.format(**values)
    except KeyError as exc:
        keys = ", ".join(sorted(values))
        raise ValueError(f"detectors.{detector['name']}.calibration_artifact_template unknown key {exc}; allowed={keys}") from exc


def _detector_template_values(
    detector: dict[str, Any],
    *,
    matrix_name: str,
    lofo_fold: dict[str, Any] | None,
) -> dict[str, str]:
    holdout = ""
    if lofo_fold is not None:
        raw_holdout = lofo_fold.get("holdout_attack")
        holdout = raw_holdout if isinstance(raw_holdout, str) else ""
    return {
        "matrix_name": matrix_name,
        "detector": detector["name"],
        "holdout_attack": holdout,
        "slug_matrix_name": _slug(matrix_name),
        "slug_detector": _slug(detector["name"]),
        "slug_holdout_attack": _slug(holdout) if holdout else "",
    }


def _format_template_value(value: Any, *, values: dict[str, str], loc: str) -> Any:
    if isinstance(value, str):
        if "{" not in value and "}" not in value:
            return value
        try:
            return value.format(**values)
        except KeyError as exc:
            keys = ", ".join(sorted(values))
            raise ValueError(f"{loc}: unknown template key {exc}; allowed={keys}") from exc
    if isinstance(value, dict):
        return {
            key: _format_template_value(item, values=values, loc=f"{loc}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _format_template_value(item, values=values, loc=f"{loc}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _judge_ref(defaults: dict[str, Any]) -> dict[str, Any]:
    judge = _mapping(defaults.get("judge"), "defaults.judge")
    return _entry_ref({"name": judge.get("name", "dummy_refusal"), "params": judge.get("params", {})})


def _model_ref(defaults: dict[str, Any]) -> dict[str, Any]:
    model = _mapping(defaults.get("model"), "defaults.model")
    result = {
        "backend": model.get("backend"),
        "model_id": model.get("model_id"),
        "device": model.get("device"),
    }
    return {key: value for key, value in result.items() if value is not None}


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key}: must be non-empty string")
    return result


def _mapping(value: Any, loc: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{loc}: must be mapping")
    return value


def _deep_merge(*values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(result.get(key), dict):
                result[key] = _deep_merge(result[key], item)
            else:
                result[key] = deepcopy(item)
    return result
