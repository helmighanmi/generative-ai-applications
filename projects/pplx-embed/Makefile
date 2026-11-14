# ═══════════════════════════════════════════════════════════════════
#  pplx-embed · Makefile
#  Developer shortcuts — wraps launch.sh and common tasks.
#
#  Usage: make <target>
# ═══════════════════════════════════════════════════════════════════

.PHONY: help install docker docker-down api ui all stop \
        test lint format typecheck download projector clean

# Default target
help:
	@echo ""
	@echo "  \033[1mpplx-embed · Makefile\033[0m"
	@echo ""
	@echo "  \033[36mSetup\033[0m"
	@echo "    make install       Create .venv and pip install -e .[dev]"
	@echo "    make download      Download model weights (~1.2 GB)"
	@echo ""
	@echo "  \033[36mRun (Docker)\033[0m"
	@echo "    make docker        Start full stack with Docker Compose"
	@echo "    make docker-down   Stop Docker containers"
	@echo ""
	@echo "  \033[36mRun (Local Python)\033[0m"
	@echo "    make api           Run FastAPI with hot-reload"
	@echo "    make ui            Run Streamlit"
	@echo "    make all           Run API + Streamlit locally"
	@echo "    make stop          Kill local processes"
	@echo ""
	@echo "  \033[36mDev tools\033[0m"
	@echo "    make test          Run pytest"
	@echo "    make lint          Run ruff linter"
	@echo "    make format        Auto-format with ruff"
	@echo "    make typecheck     Run mypy"
	@echo "    make projector     Export demo projector HTML"
	@echo "    make clean         Remove build artefacts and caches"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────
install:
	./launch.sh install

download:
	./launch.sh download

# ── Docker ────────────────────────────────────────────────────────
docker:
	./launch.sh docker

docker-down:
	./launch.sh docker:down

# ── Local Python ──────────────────────────────────────────────────
api:
	./launch.sh api

ui:
	./launch.sh ui

all:
	./launch.sh all

stop:
	./launch.sh stop

# ── Dev tools ─────────────────────────────────────────────────────
test:
	.venv/bin/pytest tests/ -v --tb=short

lint:
	.venv/bin/ruff check api/ streamlit_app/ scripts/ tests/

format:
	.venv/bin/ruff format api/ streamlit_app/ scripts/ tests/
	.venv/bin/ruff check --fix api/ streamlit_app/ scripts/ tests/

typecheck:
	.venv/bin/mypy api/ scripts/ --ignore-missing-imports

projector:
	./launch.sh projector --demo --output projector_demo.html
	@echo "  Open: file://$(shell pwd)/projector_demo.html"

# ── Clean ─────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"  -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
