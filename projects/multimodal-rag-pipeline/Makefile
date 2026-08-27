# Path: projects/multimodal-rag-pipeline/Makefile
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

.PHONY: install test lint format build-demo docker-build

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest -q

lint:
	ruff check src tests main.py

format:
	ruff format src tests main.py

build-demo:
	python main.py build

docker-build:
	docker build -t multimodal-rag .
