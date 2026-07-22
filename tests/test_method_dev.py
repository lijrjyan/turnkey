import hashlib
import json
from pathlib import Path
import textwrap

import pytest

from turnkey.cli import main
from turnkey.dev import run_method_dev


def test_method_dev_runs_external_component_and_writes_minimal_bundle(
    tmp_path: Path,
    capsys,
) -> None:  # noqa: ANN001
    method_file = tmp_path / "external_risk_method.py"
    method_file.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass, replace

            from turnkey.methods import Request
            from turnkey.policy import Component, Outcome


            @dataclass(frozen=True)
            class Risk(Request[float]):
                prompt: str


            class RiskProvider:
                request_type = Risk

                def provide(self, request):
                    return 1.0 if "UNSAFE_PLACEHOLDER" in request.prompt else 0.0


            class RiskGate:
                def apply(self, request, call_next, context):
                    score = context.get(Risk(request.sample.prompt))
                    if score >= 0.5:
                        return Outcome.blocked(
                            request.target,
                            score=score,
                            reason="external-risk",
                            diagnostics={"risk": score},
                        )
                    return replace(
                        call_next(request),
                        score=score,
                        reason="external-risk",
                        diagnostics={"risk": score},
                    )


            def build():
                return Component(
                    name="external-risk",
                    policy=RiskGate(),
                    providers=(RiskProvider(),),
                    parameters={"threshold": 0.5},
                )
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert main(
        [
            "dev",
            f"{method_file}:build",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--max-samples",
            "4",
        ]
    ) == 0

    run_dir = Path(capsys.readouterr().out.strip())
    assert {path.name for path in run_dir.iterdir()} == {
        "run.json",
        "cases.jsonl",
        "events.jsonl",
        "metrics.json",
    }

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["schema_version"] == "turnkey_run/v1"
    assert run["status"] == "complete"
    assert run["config"]["detector"]["name"] == f"{method_file}:build"
    assert run["components"]["intervention"] | {"source": None} == {
        "name": "external-risk",
        "parameters": {"threshold": 0.5},
        "source": None,
    }
    source_identity = run["components"]["intervention"]["source"]
    assert source_identity["path"] == str(method_file)
    assert len(source_identity["sha256"]) == 64

    cases = _read_jsonl(run_dir / "cases.jsonl")
    assert len(cases) == 4
    assert [case["intervention"]["detector"]["block"] for case in cases] == [
        False,
        False,
        True,
        True,
    ]
    assert all(case["reference"]["model"]["executed"] is True for case in cases)
    assert all(case["intervention"]["model"]["executed"] is False for case in cases[2:])
    assert [case["intervention"]["detector"]["diagnostics"]["risk"] for case in cases] == [
        0.0,
        0.0,
        1.0,
        1.0,
    ]

    events = _read_jsonl(run_dir / "events.jsonl")
    request_events = [event for event in events if event["kind"] == "request"]
    target_events = [event for event in request_events if event["name"].endswith(".Generate")]
    assert len(target_events) == 6
    assert sum(event["cache_hit"] is False for event in target_events) == 3
    assert all(
        event["cache_hit"] is True
        for event in target_events
        if event["pass"] == "intervention"
    )
    assert {event["pass"] for event in request_events} == {
        "reference",
        "intervention",
    }
    assert all(event["duration_s"] >= 0.0 for event in request_events)
    assert sum(event["cache_hit"] is True for event in request_events) == 4

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["counts"]["n_samples"] == 4
    assert metrics["NSG_abs"] == 1.0
    assert metrics["WBR"] == 0.0
    assert metrics["cost"]["extra_forwards_avg"] == 0.0

    serialized = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())
    assert "UNSAFE_PLACEHOLDER" not in serialized


def test_method_dev_hashes_source_bytes_without_assuming_utf8(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    method_file = tmp_path / "latin1_method.py"
    source = (
        b"# -*- coding: latin-1 -*-\n"
        b"# caf\xe9\n"
        b"from turnkey.policy import Component, PolicyChain\n"
        b"component = Component(name='latin1', policy=PolicyChain())\n"
    )
    method_file.write_bytes(source)

    assert main(
        [
            "dev",
            f"{method_file}:component",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--max-samples",
            "1",
        ]
    ) == 0

    run_dir = Path(capsys.readouterr().out.strip())
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["components"]["intervention"]["source"]["sha256"] == hashlib.sha256(source).hexdigest()


def test_method_dev_resolves_builtin_alias_through_shared_loader(tmp_path: Path) -> None:
    run_dir = run_method_dev("allow_all", out_dir=tmp_path / "outputs", max_samples=1)

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["components"]["intervention"] == {
        "name": "allow_all",
        "parameters": {},
        "source": {"kind": "builtin", "name": "allow_all"},
    }


def test_method_dev_runs_installed_module_provider_with_reuse_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "provider-closed"
    module_file = tmp_path / "installed_custom_method.py"
    module_file.write_text(
        textwrap.dedent(
            f"""
            from dataclasses import dataclass, replace
            from pathlib import Path

            from turnkey.methods import Request
            from turnkey.policy import Component


            @dataclass(frozen=True)
            class PromptLength(Request[int]):
                prompt: str


            class PromptLengthProvider:
                request_type = PromptLength

                def provide(self, request):
                    return len(request.prompt)

                def close(self):
                    Path({str(marker)!r}).write_text("closed", encoding="utf-8")


            class ObserveTwice:
                def apply(self, request, call_next, context):
                    value = context.get(PromptLength(request.sample.prompt))
                    repeated = context.get(PromptLength(request.sample.prompt))
                    return replace(call_next(request), diagnostics={{"length": value + repeated}})


            def build():
                return Component(
                    name="installed-custom",
                    policy=ObserveTwice(),
                    providers=(PromptLengthProvider(),),
                )
            """
        ).lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    run_dir = run_method_dev(
        "installed_custom_method:build",
        out_dir=tmp_path / "outputs",
        max_samples=2,
    )

    assert marker.read_text(encoding="utf-8") == "closed"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    source_identity = run["components"]["intervention"]["source"]
    assert source_identity["kind"] == "module"
    assert source_identity["module"] == "installed_custom_method"
    request_events = [
        event
        for event in _read_jsonl(run_dir / "events.jsonl")
        if event["kind"] == "request"
        and event["name"].endswith(".PromptLength")
    ]
    assert len(request_events) == 4
    assert sum(event["cache_hit"] is True for event in request_events) == 2


def test_method_dev_measures_actual_extra_calls_for_rewrite_defense(
    tmp_path: Path,
    capsys,
) -> None:  # noqa: ANN001
    method_file = tmp_path / "external_rewrite_defense.py"
    method_file.write_text(
        textwrap.dedent(
            """
            from dataclasses import replace

            from turnkey.policy import Component


            class RewriteDefense:
                def apply(self, request, call_next, context):
                    defended_target = replace(
                        request.target,
                        prompt="[defended] " + request.target.prompt,
                    )
                    return call_next(replace(request, target=defended_target))


            component = Component(name="rewrite-defense", policy=RewriteDefense())
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert main(
        [
            "dev",
            f"{method_file}:component",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--max-samples",
            "4",
        ]
    ) == 0

    run_dir = Path(capsys.readouterr().out.strip())
    events = _read_jsonl(run_dir / "events.jsonl")
    target_events = [
        event
        for event in events
        if event["kind"] == "request" and event["name"].endswith(".Generate")
    ]
    assert [event["pass"] for event in target_events].count("reference") == 4
    assert sum(
        event["pass"] == "reference" and event["cache_hit"] is False
        for event in target_events
    ) == 3
    assert sum(
        event["pass"] == "intervention" and event["cache_hit"] is False
        for event in target_events
    ) == 3

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cost"]["extra_forwards_avg"] == 0.75


def test_method_dev_runs_external_read_only_probe(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    method_file = tmp_path / "external_read_only_probe.py"
    method_file.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass, replace

            from turnkey.methods import Request
            from turnkey.policy import Component


            @dataclass(frozen=True)
            class PromptChars(Request[int]):
                prompt: str


            class PromptCharsProvider:
                request_type = PromptChars

                def provide(self, request):
                    return len(request.prompt)


            class ObservePromptChars:
                def apply(self, request, call_next, context):
                    prompt_chars = context.get(PromptChars(request.sample.prompt))
                    return replace(
                        call_next(request),
                        diagnostics={"prompt_chars": prompt_chars},
                    )


            component = Component(
                name="prompt-chars-probe",
                policy=ObservePromptChars(),
                providers=(PromptCharsProvider(),),
            )
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert main(
        [
            "dev",
            f"{method_file}:component",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--max-samples",
            "4",
        ]
    ) == 0

    run_dir = Path(capsys.readouterr().out.strip())
    cases = _read_jsonl(run_dir / "cases.jsonl")
    assert {case["intervention"]["detector"]["block"] for case in cases} == {False}
    assert [
        case["intervention"]["detector"]["diagnostics"]["prompt_chars"]
        for case in cases
    ] == [
        case["prompt"]["chars"] for case in cases
    ]

    events = _read_jsonl(run_dir / "events.jsonl")
    target_events = [
        event
        for event in events
        if event["kind"] == "request" and event["name"].endswith(".Generate")
    ]
    probe_events = [
        event
        for event in events
        if event["kind"] == "request"
        and event["name"].endswith(".PromptChars")
    ]
    assert {
        event["pass"]
        for event in target_events
        if event["cache_hit"] is False
    } == {"reference"}
    assert len(probe_events) == 4
    assert sum(event["cache_hit"] is True for event in probe_events) == 1


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
