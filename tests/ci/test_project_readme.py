import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
    )
    for fragment in required_fragments:
        assert fragment in readme


def test_deepwiki_configuration_preserves_product_boundaries() -> None:
    config = json.loads((ROOT / ".devin/wiki.json").read_text(encoding="utf-8"))

    notes = "\n".join(note["content"] for note in config["repo_notes"])
    assert "detector-development harness" in notes
    assert "research result or leaderboard" in notes
    assert "private" in notes.casefold()
