import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PUBLIC_GOVERNANCE_FILES = (
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "RELEASING.md",
    "SECURITY.md",
    "SUPPORT.md",
)


def test_readme_is_a_project_front_door() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_fragments = (
        '<div align="center">',
        "Bring Your Own Detector",
        "https://pypi.org/project/turnkey/",
        "https://deepwiki.com/lijrjyan/turnkey",
        "free public index cannot read a private GitHub repository",
        "## About",
        "## Why Turnkey",
        "## Quick Start",
        "## Documentation",
        "## Community & Support",
        "## License",
        "[CONTRIBUTING.md](CONTRIBUTING.md)",
    )
    for fragment in required_fragments:
        assert fragment in readme


def test_deepwiki_configuration_preserves_product_boundaries() -> None:
    config = json.loads((ROOT / ".devin/wiki.json").read_text(encoding="utf-8"))

    notes = "\n".join(note["content"] for note in config["repo_notes"])
    assert "detector-development harness" in notes
    assert "research result or leaderboard" in notes
    assert "private" in notes.casefold()


def test_public_governance_boundary_is_complete() -> None:
    for relative_path in PUBLIC_GOVERNANCE_FILES:
        path = ROOT / relative_path
        assert path.is_file(), f"missing public governance file: {relative_path}"
        assert len(path.read_text(encoding="utf-8").splitlines()) >= 5

    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    support = (ROOT / "SUPPORT.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    releasing = (ROOT / "RELEASING.md").read_text(encoding="utf-8")

    assert "Do not open a public issue" in security
    assert "best effort" in support
    assert "uv run ruff check ." in contributing
    assert "dev" in releasing and "main" in releasing


def test_repository_automation_is_reviewable_and_immutable() -> None:
    required_files = (
        ".github/CODEOWNERS",
        ".github/dependabot.yml",
        ".github/ISSUE_TEMPLATE/bug.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/documentation.yml",
        ".github/ISSUE_TEMPLATE/feature.yml",
    )
    for relative_path in required_files:
        assert (ROOT / relative_path).is_file(), f"missing repository file: {relative_path}"

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action_refs = re.findall(r"\buses:\s+[^\s@]+@([^\s#]+)", workflow)
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)
    assert "branches: [dev, main]" in workflow
    assert "permissions:\n  contents: read" in workflow
