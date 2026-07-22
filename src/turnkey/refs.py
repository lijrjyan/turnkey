from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
    category=Warning,
    module=r"requests",
)

import requests
import yaml

from turnkey._internal.data import sha256_hex


@dataclass(frozen=True)
class PaperRef:
    id: str
    url: str
    path: str
    title: str | None = None


@dataclass(frozen=True)
class RepoRef:
    id: str
    url: str
    path: str
    commit: str
    depth: int | None = 1


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> tuple[list[PaperRef], list[RepoRef], str]:
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise TypeError("Manifest root must be a mapping")

    papers: list[PaperRef] = []
    for p in raw.get("papers") or []:
        papers.append(PaperRef(id=p["id"], url=p["url"], path=p["path"], title=p.get("title")))

    repos: list[RepoRef] = []
    for index, r in enumerate(raw.get("repos") or []):
        commit = r.get("commit")
        if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
            raise ValueError(
                f"{path}: repos[{index}].commit must be a full immutable commit SHA, got {commit!r}"
            )
        repos.append(
            RepoRef(
                id=r["id"],
                url=r["url"],
                path=r["path"],
                commit=commit,
                depth=int(r["depth"]) if r.get("depth") is not None else None,
            )
        )

    return papers, repos, sha256_hex(text)


def _download(url: str, dest: Path, *, tmp_dir: Path, timeout_s: float = 300.0) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / (dest.name + ".tmp")
    headers = {"User-Agent": "Mozilla/5.0 (turnkey-refs)"}
    with requests.get(url, stream=True, timeout=timeout_s, headers=headers) as r:
        r.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(dest)


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _git_output(path: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as e:
        detail = e.output.strip() if e.output else str(e)
        raise RuntimeError(
            f"Invalid reference checkout at {path}: {detail}. Re-run with --force."
        ) from e


def _validate_reusable_checkout(path: Path, *, expected_url: str) -> None:
    origin = _git_output(path, "remote", "get-url", "origin")
    if origin != expected_url:
        raise RuntimeError(
            f"Reference checkout at {path} has origin {origin!r}, expected {expected_url!r}. "
            "Re-run with --force to replace it."
        )
    status = _git_output(path, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            f"Reference checkout at {path} has local changes. Re-run with --force to replace it."
        )


def _validate_resolved_checkout(path: Path, *, expected_url: str, expected_commit: str) -> str:
    _validate_reusable_checkout(path, expected_url=expected_url)
    head = _git_head(path)
    if head != expected_commit:
        raise RuntimeError(
            f"Reference checkout resolved to {head!r}, expected pinned commit {expected_commit!r}"
        )
    attached = subprocess.run(
        ["git", "-C", str(path), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if attached.returncode == 0:
        raise RuntimeError(f"Reference checkout at {path} must be detached at {expected_commit}")
    if attached.returncode != 1:
        detail = attached.stderr.strip() or attached.stdout.strip()
        raise RuntimeError(f"Could not verify detached HEAD at {path}: {detail}")
    return expected_commit


def _git_fetch_commit(dest: Path, *, commit: str, depth: int | None) -> None:
    cmd = ["git", "-C", str(dest), "fetch", "--no-tags"]
    if depth and depth > 0:
        cmd += ["--depth", str(depth)]
    cmd += ["origin", commit]
    subprocess.run(cmd, check=True)


def _git_checkout_commit(dest: Path, *, url: str, commit: str, depth: int | None) -> str:
    present = subprocess.run(
        ["git", "-C", str(dest), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        capture_output=True,
    )
    if present.returncode != 0:
        _git_fetch_commit(dest, commit=commit, depth=depth)
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", commit], check=True)
    return _validate_resolved_checkout(dest, expected_url=url, expected_commit=commit)


def _git_clone(url: str, dest: Path, *, commit: str, depth: int | None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
    _git_fetch_commit(dest, commit=commit, depth=depth)
    return _git_checkout_commit(dest, url=url, commit=commit, depth=depth)


def fetch_all(
    *,
    root: Path,
    manifest_path: Path,
    fetch_papers: bool,
    fetch_repos: bool,
    dry_run: bool,
    force: bool,
) -> Path:
    papers, repos, manifest_sha = _load_manifest(manifest_path)
    prev_lock: dict[str, Any] | None = None
    prev_lock_path = root / "LOCK.json"
    if prev_lock_path.exists():
        try:
            prev_lock = json.loads(prev_lock_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev_lock = None
    lock: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "papers": {},
        "repos": {},
        "errors": [],
    }
    if prev_lock and isinstance(prev_lock, dict):
        if not fetch_papers and isinstance(prev_lock.get("papers"), dict):
            lock["papers"] = prev_lock["papers"]

    if not fetch_repos and repos and not dry_run:
        previous_repos = prev_lock.get("repos") if isinstance(prev_lock, dict) else None
        if not isinstance(previous_repos, dict):
            raise RuntimeError("reference lock has no repository state; fetch repos first")
        for repo in repos:
            previous = previous_repos.get(repo.id)
            if not isinstance(previous, dict):
                raise RuntimeError(
                    f"reference lock has no state for {repo.id!r}; fetch repos first"
                )
            requested = previous.get("requested_commit") or previous.get("head")
            resolved = previous.get("resolved_commit") or previous.get("head")
            head = previous.get("head") or resolved
            expected_path = str(root / repo.path)
            if (
                previous.get("url") != repo.url
                or previous.get("path") != expected_path
                or requested != repo.commit
                or resolved != repo.commit
                or head != repo.commit
            ):
                raise RuntimeError(
                    f"reference lock for {repo.id!r} does not match the manifest; "
                    "fetch repos before a partial fetch"
                )
            lock["repos"][repo.id] = {
                "url": repo.url,
                "path": expected_path,
                "requested_commit": repo.commit,
                "resolved_commit": resolved,
                "head": head,
                "depth": repo.depth,
            }

    tmp_dir = root / "tmp"

    if fetch_repos and not force and not dry_run:
        for repo in repos:
            dest = root / repo.path
            if dest.exists():
                _validate_reusable_checkout(dest, expected_url=repo.url)

    if fetch_papers:
        for p in papers:
            dest = root / p.path
            if dry_run:
                print(f"[dry-run] paper {p.id} -> {dest} ({p.url})")
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and not force:
                    sha = _sha256_file(dest)
                else:
                    if dest.exists() and force:
                        dest.unlink()
                    _download(p.url, dest, tmp_dir=tmp_dir)
                    sha = _sha256_file(dest)
                lock["papers"][p.id] = {
                    "url": p.url,
                    "path": str(dest),
                    "sha256": sha,
                    "title": p.title,
                }
            except Exception as e:  # noqa: BLE001
                lock["errors"].append({"type": "paper", "id": p.id, "url": p.url, "error": str(e)})

    if fetch_repos:
        for r in repos:
            dest = root / r.path
            if dry_run:
                print(f"[dry-run] repo  {r.id} -> {dest} ({r.url} @ {r.commit})")
                continue
            try:
                if dest.exists() and force:
                    shutil.rmtree(dest)
                if not dest.exists():
                    head = _git_clone(r.url, dest, commit=r.commit, depth=r.depth)
                else:
                    head = _git_checkout_commit(dest, url=r.url, commit=r.commit, depth=r.depth)
                lock["repos"][r.id] = {
                    "url": r.url,
                    "path": str(dest),
                    "requested_commit": r.commit,
                    "resolved_commit": head,
                    "head": head,
                    "depth": r.depth,
                }
            except Exception as e:  # noqa: BLE001
                lock["errors"].append({"type": "repo", "id": r.id, "url": r.url, "error": str(e)})

    if not dry_run:
        lock_path = root / "LOCK.json"
        lock_path.write_text(
            json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if lock["errors"]:
            raise SystemExit(
                f"Reference fetch finished with {len(lock['errors'])} error(s). See {lock_path}."
            )
        return lock_path
    return root / "LOCK.json"


def _cmd_fetch(args: argparse.Namespace) -> int:
    root = Path(args.root)
    manifest_path = Path(args.manifest)
    fetch_papers = args.all or args.papers
    fetch_repos = args.all or args.repos
    if not (fetch_papers or fetch_repos):
        raise SystemExit("Specify one of: --all, --papers, --repos")

    lock_path = fetch_all(
        root=root,
        manifest_path=manifest_path,
        fetch_papers=fetch_papers,
        fetch_repos=fetch_repos,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
    )
    if not args.dry_run:
        print(str(lock_path))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    papers, repos, _ = _load_manifest(Path(args.manifest))
    for p in papers:
        print(f"paper\t{p.id}\t{p.path}\t{p.url}")
    for r in repos:
        print(f"repo\t{r.id}\t{r.path}\t{r.url}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="turnkey-refs")
    p.add_argument("--manifest", default="references/manifest.yaml")
    p.add_argument("--root", default="references")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List references from manifest")
    p_list.set_defaults(func=_cmd_list)

    p_fetch = sub.add_parser("fetch", help="Fetch papers and/or repos into references/")
    p_fetch.add_argument("--all", action="store_true", help="Fetch all papers and repos")
    p_fetch.add_argument("--papers", action="store_true", help="Fetch only papers")
    p_fetch.add_argument("--repos", action="store_true", help="Fetch only repos")
    p_fetch.add_argument("--dry-run", action="store_true")
    p_fetch.add_argument("--force", action="store_true", help="Re-download / re-clone if exists")
    p_fetch.set_defaults(func=_cmd_fetch)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
