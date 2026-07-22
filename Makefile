.PHONY: docs fmt lint notebooks smoke test

lint:
	ruff check src tests scripts

fmt:
	ruff format src tests scripts

test:
	pytest -q

smoke:
	@RUN_DIR=$$(turnkey run --config configs/runs/smoke.yaml); \
		echo $$RUN_DIR; \
		turnkey validate $$RUN_DIR/cases.jsonl; \
		turnkey audit $$RUN_DIR

notebooks:
	python scripts/ci/run_notebooks.py

docs:
	mkdocs build --strict
