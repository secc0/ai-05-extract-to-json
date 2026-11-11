# Convenience wrappers. Every target is a plain command you can also run directly.
# Activate your virtual environment first so `python` points at it.
PYTHON ?= python
IMAGE ?= ai-05-extract-to-json

.PHONY: setup run batch test lint format eval smoke docker-build

setup:
	$(PYTHON) -m pip install -e ".[dev]"

run:
	$(PYTHON) -m order_extractor $(or $(FILE),examples/01.txt)

batch:
	$(PYTHON) -m order_extractor examples --batch --today 2026-11-11

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

eval:
	$(PYTHON) -m evals.run --out evals/results/latest.json

smoke:
	$(PYTHON) -m evals.run --smoke

docker-build:
	docker build -t $(IMAGE) .
