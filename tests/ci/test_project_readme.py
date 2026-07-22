import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

WORKSPACE_ONLY_FILES = (
    "AGENTS.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "RELEASING.md",
    "SECURITY.md",
    "SUPPORT.md",
    ".githooks",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/ISSUE_TEMPLATE",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "scripts/install-git-hooks.sh",
)


def test_readme_is_a_project_front_door() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_fragments = (
        '<div align="center">',
        "Bring Your Own Detector",
        "https://pypi.org/project/turnkey/",
        "https://deepwiki.com/lijrjyan/turnkey",
        "img.shields.io/github/stars/lijrjyan/turnkey?style=for-the-badge&logo=github&label=stars",
        "img.shields.io/github/license/lijrjyan/turnkey?style=for-the-badge",
        "img.shields.io/github/issues-closed-raw/lijrjyan/turnkey?style=for-the-badge&label=closed%20issues",
        "img.shields.io/github/issues-raw/lijrjyan/turnkey?style=for-the-badge&label=open%20issues",
        "img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge",
        "free public index cannot read a private GitHub repository",
        "## About",
        "## Why Turnkey",
        "## Quick Start",
        "## Documentation",
        "## Community & Support",
        "## License",
    )
    for fragment in required_fragments:
        assert fragment in readme

    assert "Star History" not in readme
    assert "star-history.com" not in readme
    assert "actions/workflows/ci.yml/badge.svg" not in readme
    assert "img.shields.io/pypi/v/turnkey" not in readme


def test_deepwiki_configuration_preserves_product_boundaries() -> None:
    config = json.loads((ROOT / ".devin/wiki.json").read_text(encoding="utf-8"))

    notes = "\n".join(note["content"] for note in config["repo_notes"])
    assert "detector-development harness" in notes
    assert "research result or leaderboard" in notes
    assert "private" in notes.casefold()


def test_repository_level_docs_are_intentionally_minimal() -> None:
    assert (ROOT / "README.md").is_file()
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text

    for relative_path in WORKSPACE_ONLY_FILES:
        assert not (ROOT / relative_path).exists(), (
            f"workspace-only material leaked into product repository: {relative_path}"
        )


def test_repository_automation_is_reviewable_and_immutable() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action_refs = re.findall(r"\buses:\s+[^\s@]+@([^\s#]+)", workflow)
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)
    assert "branches: [dev, main]" in workflow
    assert "permissions:\n  contents: read" in workflow
