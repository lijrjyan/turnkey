from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from turnkey.config import DATASET_VERSION_RE


REDACTED_PREFIX = "<redacted sha256="
CASE_SCHEMA = "turnkey_case/v1"


def validate_cases_jsonl(path: str | Path) -> list[str]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}:{line_no}: invalid json: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_no}: row must be object")
                continue
            _validate_case_row(path=path, line_no=line_no, row=row, errors=errors)
            rows.append(row)

    seen: set[str] = set()
    datasets: set[tuple[str, str]] = set()
    for line_no, row in enumerate(rows, start=1):
        case_id = row.get("case_id")
        loc = f"{path}:{line_no}"
        if isinstance(case_id, str) and case_id:
            if case_id in seen:
                errors.append(f"{loc}: duplicate case_id: {case_id}")
            seen.add(case_id)
        dataset = row.get("dataset")
        if isinstance(dataset, dict):
            name = dataset.get("name")
            version = dataset.get("version")
            if isinstance(name, str) and isinstance(version, str):
                datasets.add((name, version))
    if len(datasets) > 1:
        errors.append(f"{path}: mixed dataset identities in one cases.jsonl: {sorted(datasets)}")
    return errors


def _validate_case_row(
    *,
    path: Path,
    line_no: int,
    row: dict[str, Any],
    errors: list[str],
) -> None:
    loc = f"{path}:{line_no}"
    required = {
        "schema_version",
        "case_id",
        "behavior_id",
        "is_benign",
        "dataset",
        "attack_family",
        "attack_method",
        "attack_params",
        "threat",
        "budget",
        "prompt",
        "reference",
        "intervention",
    }
    missing = sorted(required - set(row))
    if missing:
        errors.append(f"{loc}: missing top keys: {missing}")
    if row.get("schema_version") != CASE_SCHEMA:
        errors.append(f"{loc}: schema_version must be {CASE_SCHEMA!r}")
    if not isinstance(row.get("case_id"), str) or not row.get("case_id"):
        errors.append(f"{loc}: case_id must be non-empty string")
    if type(row.get("is_benign")) is not bool:
        errors.append(f"{loc}: is_benign must be bool")

    dataset = row.get("dataset")
    if not isinstance(dataset, dict):
        errors.append(f"{loc}: dataset must be object")
    else:
        name = dataset.get("name")
        version = dataset.get("version")
        if not isinstance(name, str) or not name:
            errors.append(f"{loc}: dataset.name must be non-empty string")
        if not isinstance(version, str) or not DATASET_VERSION_RE.fullmatch(version):
            errors.append(f"{loc}: dataset.version must match vN")

    prompt = row.get("prompt")
    if not isinstance(prompt, dict):
        errors.append(f"{loc}: prompt must be object")
    else:
        _check_public_prompt(loc=loc, prompt=prompt, errors=errors)

    _check_threat_context(loc=loc, row=row, errors=errors)
    for pass_name in ("reference", "intervention"):
        _check_case_pass(loc=loc, pass_name=pass_name, row=row, errors=errors)
    _check_case_metric_verdicts(loc=loc, row=row, errors=errors)


def _check_public_prompt(
    *,
    loc: str,
    prompt: dict[str, Any],
    errors: list[str],
) -> None:
    prompt_hash = prompt.get("sha256")
    if (
        not isinstance(prompt_hash, str)
        or len(prompt_hash) != 64
        or any(char not in "0123456789abcdef" for char in prompt_hash)
    ):
        errors.append(f"{loc}: prompt.sha256 must be a hex sha256 string")
    prompt_chars = prompt.get("chars")
    if not isinstance(prompt_chars, int) or isinstance(prompt_chars, bool) or prompt_chars < 0:
        errors.append(f"{loc}: prompt.chars must be a non-negative int")
    if "text" in prompt:
        errors.append(f"{loc}: public case must not contain prompt.text")


def _check_case_pass(
    *,
    loc: str,
    pass_name: str,
    row: dict[str, Any],
    errors: list[str],
) -> None:
    outcome = row.get(pass_name)
    if not isinstance(outcome, dict):
        errors.append(f"{loc}: {pass_name} must be object")
        return
    missing = sorted({"detector", "model", "judge"} - set(outcome))
    if missing:
        errors.append(f"{loc}: missing {pass_name} keys: {missing}")
        return
    _check_pass(loc=loc, pass_name=pass_name, row=row, errors=errors)


def _check_case_metric_verdicts(
    *,
    loc: str,
    row: dict[str, Any],
    errors: list[str],
) -> None:
    is_benign = row.get("is_benign")
    if type(is_benign) is not bool:
        return

    def require(pass_name: str, field: str) -> None:
        outcome = row.get(pass_name)
        judge = outcome.get("judge") if isinstance(outcome, dict) else None
        if not isinstance(judge, dict) or type(judge.get(field)) is not bool:
            errors.append(f"{loc}: {pass_name}.judge.{field} must be bool for metrics")

    require("reference", "is_refusal")
    if is_benign:
        return
    require("reference", "is_harmful_effective")
    intervention = row.get("intervention")
    detector = intervention.get("detector") if isinstance(intervention, dict) else None
    if isinstance(detector, dict) and detector.get("block") is False:
        require("intervention", "is_harmful_effective")


def _check_threat_context(*, loc: str, row: dict[str, Any], errors: list[str]) -> None:
    threat = row.get("threat")
    if not isinstance(threat, dict):
        errors.append(f"{loc}: threat must be object")
        return
    if threat.get("attack_family") != row.get("attack_family"):
        errors.append(f"{loc}: threat.attack_family does not match row.attack_family")
    if threat.get("attack_method") != row.get("attack_method"):
        errors.append(f"{loc}: threat.attack_method does not match row.attack_method")

    budget = row.get("budget")
    if not isinstance(budget, dict):
        errors.append(f"{loc}: budget must be object")
    elif not isinstance(budget.get("bucket"), str):
        errors.append(f"{loc}: budget.bucket must be string")


def _check_pass(*, loc: str, pass_name: str, row: dict[str, Any], errors: list[str]) -> None:
    block = row.get(pass_name)
    if not isinstance(block, dict):
        return

    detector = block.get("detector")
    model = block.get("model")
    judge = block.get("judge")
    if not isinstance(detector, dict):
        errors.append(f"{loc}: {pass_name}.detector must be object")
        return
    if not isinstance(model, dict):
        errors.append(f"{loc}: {pass_name}.model must be object")
        return
    if not isinstance(judge, dict):
        errors.append(f"{loc}: {pass_name}.judge must be object")
        return

    is_blocked = detector.get("block")
    executed = model.get("executed")
    if not isinstance(is_blocked, bool):
        errors.append(f"{loc}: {pass_name}.detector.block must be bool")
        return
    if not isinstance(executed, bool):
        errors.append(f"{loc}: {pass_name}.model.executed must be bool")
        return

    response_text = model.get("response_text")
    judge_details = judge.get("details")

    if is_blocked:
        if executed is not False:
            errors.append(f"{loc}: {pass_name} blocked path must have model.executed=false")
        if response_text is not None:
            errors.append(f"{loc}: {pass_name} blocked path must not include response_text")
        if not (isinstance(judge_details, dict) and judge_details.get("blocked") is True):
            errors.append(f"{loc}: {pass_name} blocked path must mark judge.details.blocked=true")
        return

    if executed is not True:
        errors.append(f"{loc}: {pass_name} non-blocked path must have model.executed=true")

    if not isinstance(response_text, str) or not response_text:
        errors.append(f"{loc}: {pass_name} non-blocked path must include response_text")
    elif not response_text.startswith(REDACTED_PREFIX):
        errors.append(f"{loc}: {pass_name} response_text is not redacted")
