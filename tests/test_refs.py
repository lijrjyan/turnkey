from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from turnkey._internal.data import sha256_hex
from turnkey.refs import _load_manifest, fetch_all


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _commit_file(repo: Path, *, text: str, message: str) -> str:
    (repo / "value.txt").write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "value.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Turnkey Test",
            "-c",
            "user.email=turnkey@example.invalid",
            "commit",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return _git("rev-parse", "HEAD", cwd=repo)


def _write_repo_manifest(
    path: Path,
    *,
    url: str,
    commit: str,
    repo_path: str = "repos/example",
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "papers": [],
                "repos": [
                    {
                        "id": "example",
                        "url": url,
                        "path": repo_path,
                        "depth": 1,
                        "commit": commit,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _fetch_repos(*, root: Path, manifest: Path, force: bool = False) -> Path:
    return fetch_all(
        root=root,
        manifest_path=manifest,
        fetch_papers=False,
        fetch_repos=True,
        dry_run=False,
        force=force,
    )


def test_manifest_requires_full_repo_commit(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "papers": [],
                "repos": [
                    {
                        "id": "example",
                        "url": "https://example.invalid/repo.git",
                        "path": "repos/example",
                        "depth": 1,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"repos\[0\]\.commit"):
        _load_manifest(manifest)


def test_fetch_repos_checks_out_manifest_commit_and_records_resolution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    pinned_commit = _commit_file(source, text="pinned\n", message="pinned")
    _commit_file(source, text="newer\n", message="newer")

    manifest = tmp_path / "manifest.yaml"
    _write_repo_manifest(manifest, url=str(source), commit=pinned_commit)
    root = tmp_path / "references"

    lock_path = _fetch_repos(root=root, manifest=manifest)

    checkout = root / "repos/example"
    assert _git("rev-parse", "HEAD", cwd=checkout) == pinned_commit
    assert _git("branch", "--show-current", cwd=checkout) == ""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["repos"]["example"]["requested_commit"] == pinned_commit
    assert lock["repos"]["example"]["resolved_commit"] == pinned_commit


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_fetch_repos_refuses_dirty_checkout_before_writing_lock(
    tmp_path: Path,
    dirty_kind: str,
) -> None:
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    pinned_commit = _commit_file(source, text="pinned\n", message="pinned")
    manifest = tmp_path / "manifest.yaml"
    _write_repo_manifest(manifest, url=str(source), commit=pinned_commit)

    root = tmp_path / "references"
    checkout = root / "repos/example"
    checkout.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(source), str(checkout)], check=True)
    subprocess.run(["git", "checkout", "--detach", pinned_commit], cwd=checkout, check=True)
    dirty_path = checkout / ("value.txt" if dirty_kind == "tracked" else "untracked.txt")
    dirty_path.write_text(f"{dirty_kind} change\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"local changes.*--force"):
        _fetch_repos(root=root, manifest=manifest)

    assert dirty_path.read_text(encoding="utf-8") == f"{dirty_kind} change\n"
    assert not (root / "LOCK.json").exists()


def test_fetch_repos_reuses_clean_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    pinned_commit = _commit_file(source, text="pinned\n", message="pinned")
    manifest = tmp_path / "manifest.yaml"
    _write_repo_manifest(manifest, url=str(source), commit=pinned_commit)
    root = tmp_path / "references"

    _fetch_repos(root=root, manifest=manifest)
    checkout = root / "repos/example"
    reuse_marker = checkout / ".git/turnkey-reuse-marker"
    reuse_marker.write_text("keep\n", encoding="utf-8")

    lock_path = _fetch_repos(root=root, manifest=manifest)

    assert reuse_marker.read_text(encoding="utf-8") == "keep\n"
    assert _git("rev-parse", "HEAD", cwd=checkout) == pinned_commit
    assert _git("branch", "--show-current", cwd=checkout) == ""
    assert _git("status", "--porcelain", cwd=checkout) == ""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["repos"]["example"]["resolved_commit"] == pinned_commit


def test_fetch_repos_force_replaces_dirty_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    pinned_commit = _commit_file(source, text="pinned\n", message="pinned")
    manifest = tmp_path / "manifest.yaml"
    _write_repo_manifest(manifest, url=str(source), commit=pinned_commit)
    root = tmp_path / "references"

    _fetch_repos(root=root, manifest=manifest)
    checkout = root / "repos/example"
    (checkout / "value.txt").write_text("tracked change\n", encoding="utf-8")
    (checkout / "untracked.txt").write_text("untracked change\n", encoding="utf-8")
    reuse_marker = checkout / ".git/turnkey-reuse-marker"
    reuse_marker.write_text("remove\n", encoding="utf-8")

    lock_path = _fetch_repos(root=root, manifest=manifest, force=True)

    assert not reuse_marker.exists()
    assert not (checkout / "untracked.txt").exists()
    assert (checkout / "value.txt").read_text(encoding="utf-8") == "pinned\n"
    assert _git("rev-parse", "HEAD", cwd=checkout) == pinned_commit
    assert _git("branch", "--show-current", cwd=checkout) == ""
    assert _git("status", "--porcelain", cwd=checkout) == ""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["repos"]["example"]["resolved_commit"] == pinned_commit


def test_fetch_repos_refuses_checkout_with_wrong_origin(tmp_path: Path) -> None:
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    pinned_commit = _commit_file(source, text="pinned\n", message="pinned")
    alternate = tmp_path / "alternate.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(source), str(alternate)], check=True)
    manifest = tmp_path / "manifest.yaml"
    _write_repo_manifest(manifest, url=str(alternate), commit=pinned_commit)

    root = tmp_path / "references"
    checkout = root / "repos/example"
    checkout.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(source), str(checkout)], check=True)
    subprocess.run(["git", "checkout", "--detach", pinned_commit], cwd=checkout, check=True)

    with pytest.raises(RuntimeError, match=r"origin.*--force"):
        _fetch_repos(root=root, manifest=manifest)

    assert not (root / "LOCK.json").exists()


@pytest.mark.parametrize("changed_field", ["commit", "url", "path"])
def test_partial_fetch_refuses_stale_repo_lock(
    tmp_path: Path,
    changed_field: str,
) -> None:
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    pinned_commit = _commit_file(source, text="pinned\n", message="pinned")
    newer_commit = _commit_file(source, text="newer\n", message="newer")
    manifest = tmp_path / "manifest.yaml"
    _write_repo_manifest(manifest, url=str(source), commit=pinned_commit)
    root = tmp_path / "references"
    lock_path = _fetch_repos(root=root, manifest=manifest)
    original_lock = lock_path.read_bytes()

    if changed_field == "commit":
        _write_repo_manifest(manifest, url=str(source), commit=newer_commit)
    elif changed_field == "url":
        alternate = tmp_path / "alternate.git"
        subprocess.run(["git", "clone", "-q", "--bare", str(source), str(alternate)], check=True)
        _write_repo_manifest(manifest, url=str(alternate), commit=pinned_commit)
    else:
        _write_repo_manifest(
            manifest,
            url=str(source),
            commit=pinned_commit,
            repo_path="repos/moved",
        )

    with pytest.raises(RuntimeError, match=r"fetch.*repos"):
        fetch_all(
            root=root,
            manifest_path=manifest,
            fetch_papers=True,
            fetch_repos=False,
            dry_run=False,
            force=False,
        )

    assert lock_path.read_bytes() == original_lock


def test_partial_dry_run_does_not_require_existing_lock(tmp_path: Path) -> None:
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    pinned_commit = _commit_file(source, text="pinned\n", message="pinned")
    manifest = tmp_path / "manifest.yaml"
    _write_repo_manifest(manifest, url=str(source), commit=pinned_commit)
    root = tmp_path / "references"

    lock_path = fetch_all(
        root=root,
        manifest_path=manifest,
        fetch_papers=True,
        fetch_repos=False,
        dry_run=True,
        force=False,
    )

    assert lock_path == root / "LOCK.json"
    assert not lock_path.exists()


def test_tracked_reference_lock_matches_manifest_commits() -> None:
    manifest_path = Path("references/manifest.yaml")
    _papers, repos, manifest_sha = _load_manifest(manifest_path)
    lock = json.loads(Path("references/LOCK.json").read_text(encoding="utf-8"))

    assert manifest_sha == sha256_hex(manifest_path.read_text(encoding="utf-8"))
    assert lock["manifest_sha256"] == manifest_sha
    assert set(lock["repos"]) == {repo.id for repo in repos}
    for repo in repos:
        locked = lock["repos"][repo.id]
        assert locked["requested_commit"] == repo.commit
        assert locked["resolved_commit"] == repo.commit
