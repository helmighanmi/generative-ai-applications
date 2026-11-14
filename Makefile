PYTHON ?= python

.PHONY: help install lint format format-check compile check test eval-doc-parsing security clean

help:
	@echo "Repository commands:"
	@echo "  make install       Install root development tooling"
	@echo "  make lint          Run Ruff lint checks"
	@echo "  make format        Apply Ruff formatting"
	@echo "  make format-check  Verify Ruff formatting"
	@echo "  make compile       Compile all Python source for syntax errors"
	@echo "  make check         Run every static quality gate used by CI"
	@echo "  make test          Run lightweight project tests"
	@echo "  make eval-doc-parsing  Run the page-classification evaluation"
	@echo "  make security      Audit root environment dependencies"
	@echo "  make clean         Remove Python cache/build artifacts"

install:
	$(PYTHON) -m pip install -e '.[dev]'

lint:
	ruff check projects labs scripts

format:
	ruff format projects labs scripts

format-check:
	ruff format --check projects labs scripts

compile:
	$(PYTHON) -m compileall -q projects labs scripts

check: lint format-check compile
	$(PYTHON) scripts/check_docs_links.py

test:
	$(PYTHON) scripts/run_tests.py

eval-doc-parsing:
	$(PYTHON) projects/document-parsing/evals/evaluate.py

security:
	pip-audit --strict

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
