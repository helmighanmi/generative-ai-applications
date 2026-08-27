# Path: Makefile
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

PYTHON ?= python

.PHONY: help install lint format format-check compile test security clean

help:
	@echo "Repository commands:"
	@echo "  make install       Install root development tooling"
	@echo "  make lint          Run Ruff lint checks"
	@echo "  make format        Apply Ruff formatting"
	@echo "  make format-check  Verify Ruff formatting"
	@echo "  make compile       Compile all Python source for syntax errors"
	@echo "  make test          Run lightweight project tests"
	@echo "  make security      Audit root environment dependencies"
	@echo "  make clean         Remove Python cache/build artifacts"

install:
	$(PYTHON) -m pip install -e '.[dev]'

lint:
	ruff check projects scripts

format:
	ruff format projects scripts

format-check:
	ruff format --check projects scripts

compile:
	$(PYTHON) -m compileall -q projects scripts

test:
	$(PYTHON) scripts/run_tests.py

security:
	pip-audit

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
