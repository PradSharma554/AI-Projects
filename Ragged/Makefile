.PHONY: dev test lint format ingest serve eval clean

dev:
	source .venv/bin/activate && uvicorn ragged.api.app:create_app --reload --factory

serve:
	source .venv/bin/activate && python -m ragged.cli serve

test:
	source .venv/bin/activate && pytest -v

test-cov:
	source .venv/bin/activate && pytest --cov=ragged --cov-report=term-missing

lint:
	source .venv/bin/activate && ruff check ragged/ tests/

format:
	source .venv/bin/activate && ruff format ragged/ tests/

ingest:
	source .venv/bin/activate && python -m ragged.cli ingest $(DIR)

eval:
	source .venv/bin/activate && python eval/runner.py --config config.yaml

clean:
	rm -rf data/ .pytest_cache __pycache__ ragged/__pycache__ eval/results.json
