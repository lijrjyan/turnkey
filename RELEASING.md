# Releasing Turnkey

`dev` is the development integration branch. `main` is the stable public branch
and receives reviewed changes only after maintainer approval. Package, source,
and documentation publication are separate release gates.

## Prepare

1. Choose the release version and update `pyproject.toml`, public release notes,
   and `CHANGELOG.md` together.
2. Verify third-party licenses, source attributions, and package contents.
3. Run the complete CPU-safe gate:

   ```bash
   uv sync --extra dev --extra notebooks
   uv run ruff check .
   uv run pytest -q
   uv run make smoke
   uv run python scripts/ci/run_notebooks.py
   uv build
   ```

4. Inspect the wheel and source distribution from a clean environment.
5. Run a tracked-tree and staged-diff secret scan.
6. Confirm that PyPI instructions describe the artifact being published, not
   the existing `0.0.1` namespace reservation.

GPU/model integrations have separate evidence requirements. Their absence from
the CPU-safe gate must remain explicit in release notes.

## Publish

1. Promote the reviewed commit to stable `main` using the repository's approved
   review process.
2. Create an immutable version tag from that exact commit.
3. Build distributions again from the tagged clean checkout.
4. Publish only after the maintainer verifies the target PyPI project, version,
   credentials, and package metadata.
5. Update the official site to the released product commit and verify notebook
   provenance before deploying it.

## Record

Record the product commit and tag, distribution digests, Python test count,
notebook and documentation results, secret-scan result, PyPI URL, and official
site commit. If any release step is skipped or differs from this procedure,
record the exception rather than implying the full gate passed.
