.PHONY: install test lint bench bench-real clean

PY ?= python3
export PYTHONPATH := src:.

install:
	pip install -e ".[dev]"

test:
	$(PY) -m pytest tests -q

lint:
	ruff check src tests bench

bench:
	$(PY) -m bench.run

bench-real:
	$(PY) -m bench.run --real --embedder sentence-transformers

clean:
	rm -rf .pytest_cache **/__pycache__ .coverage
