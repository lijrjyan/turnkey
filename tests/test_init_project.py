from __future__ import annotations

from pathlib import Path

import pytest

from turnkey.cli import main
from turnkey.config import load_config


def test_init_creates_a_runnable_external_detector_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_dir = tmp_path / "my-detector"

    assert main(["init", str(project_dir)]) == 0

    assert capsys.readouterr().out.strip() == str(project_dir)
    assert {path.name for path in project_dir.iterdir()} == {
        "README.md",
        "detector.py",
        "run.yaml",
    }
    config = load_config(project_dir / "run.yaml")
    assert config.detector.name == "./detector.py:build"
    assert config.model.backend == "dummy"
    dev_outputs = tmp_path / "dev-outputs"
    assert (
        main(
            [
                "dev",
                f"{project_dir / 'detector.py'}:build",
                "--max-samples",
                "2",
                "--out-dir",
                str(dev_outputs),
            ]
        )
        == 0
    )
    assert str(dev_outputs) in capsys.readouterr().out


def test_init_refuses_to_overwrite_an_existing_path(tmp_path: Path) -> None:
    project_dir = tmp_path / "existing"
    project_dir.mkdir()
    marker = project_dir / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        main(["init", str(project_dir)])

    assert marker.read_text(encoding="utf-8") == "keep me"
