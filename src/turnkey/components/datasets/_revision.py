from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from urllib.parse import urlparse

from turnkey.schema import Sample


_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class DatasetLoadResult:
    samples: list[Sample]
    resolved_revision: str | None = None


def resolve_loaded_dataset_revision(
    loaded_datasets: Sequence[object],
    *,
    requested_revision: str | None,
) -> str | None:
    revisions_by_dataset = [_metadata_revisions(dataset) for dataset in loaded_datasets]
    requested_commit = _immutable_revision(requested_revision)

    if requested_commit is not None:
        for revisions in revisions_by_dataset:
            if revisions is not None and not revisions:
                raise ValueError(
                    "could not resolve the requested immutable dataset revision "
                    "from loaded dataset metadata"
                )

    known = [revisions for revisions in revisions_by_dataset if revisions is not None]
    revisions = set().union(*known) if known else set()
    if len(revisions) > 1:
        raise ValueError(
            "loaded dataset metadata resolved to multiple revisions: " + ", ".join(sorted(revisions))
        )
    if not revisions:
        # `datasets` >= 4 no longer records download_checksums, so there is no
        # in-band metadata to cross-check. The loader passed the immutable
        # revision to load_dataset, which fetches exactly that commit.
        return requested_commit

    resolved = next(iter(revisions))
    if requested_commit is not None and resolved != requested_commit:
        raise ValueError(
            f"loaded dataset resolved to {resolved}, expected requested revision {requested_commit}"
        )
    return resolved


def _metadata_revisions(dataset: object) -> set[str] | None:
    """Revisions recorded in dataset metadata, or None when metadata is absent.

    `datasets` >= 4 returns download_checksums=None, which is "no metadata"
    rather than "metadata that failed to resolve".
    """
    info = getattr(dataset, "info", None)
    checksums = getattr(info, "download_checksums", None)
    if not isinstance(checksums, Mapping):
        return None
    return {
        revision
        for value in checksums
        if isinstance(value, str)
        if (revision := _hf_url_revision(value)) is not None
    }


def _hf_url_revision(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.scheme != "hf":
        return None
    path = f"{parsed.netloc}{parsed.path}"
    if "@" not in path:
        return None
    candidate = path.rsplit("@", 1)[1].split("/", 1)[0]
    return _immutable_revision(candidate)


def _immutable_revision(value: str | None) -> str | None:
    if not isinstance(value, str) or _IMMUTABLE_REVISION_RE.fullmatch(value) is None:
        return None
    return value.lower()
