#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  pplx-embed · launch.sh
#  Unified launcher — Docker or local Python (venv)
#
#  Usage:
#    ./launch.sh [command] [options]
#
#  Commands:
#    docker          Start full stack (API + Streamlit) with Docker Compose
#    docker:api      Start only the API container
#    docker:ui       Start only the Streamlit container
#    docker:down     Stop all containers
#    docker:logs     Follow logs of all containers
#
#    api             Run FastAPI locally (needs venv activated or pip install -e .[dev])
#    ui              Run Streamlit locally
#    all             Run API + Streamlit locally (two background processes)
#    stop            Kill local processes started by this script
#
#    install         Create venv and install the project (pip install -e .[dev])
#    download        Download model weights to ./model/
#    projector       Export embedding projector to standalone HTML
#
#    help            Show this message
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RESET="\033[0m"
BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
MAGENTA="\033[35m"

log()    { echo -e "${CYAN}${BOLD}[pplx-embed]${RESET} $*"; }
ok()     { echo -e "${GREEN}${BOLD}  ✓${RESET} $*"; }
warn()   { echo -e "${YELLOW}${BOLD}  ⚠${RESET}  $*"; }
error()  { echo -e "${RED}${BOLD}  ✗${RESET}  $*" >&2; }
header() { echo -e "\n${MAGENTA}${BOLD}━━━  $*  ━━━${RESET}\n"; }

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/myenv"
PID_FILE="$SCRIPT_DIR/.pids"

# ── Helpers ───────────────────────────────────────────────────────────────────

check_docker() {
    if ! command -v docker &>/dev/null; then
        error "Docker not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    if ! docker compose version &>/dev/null; then
        error "Docker Compose v2 not found. Update Docker Desktop."
        exit 1
    fi
}

check_python() {
    if ! command -v python3 &>/dev/null; then
        error "python3 not found."
        exit 1
    fi
}

activate_venv() {
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        ok "Virtualenv activated: $VENV_DIR"
    else
        warn "No virtualenv found at $VENV_DIR"
        warn "Run: ./launch.sh install"
        exit 1
    fi
}

print_banner() {
    echo -e "${MAGENTA}${BOLD}"
    echo "  ██████╗ ██████╗ ██╗     ██╗  ██╗      ███████╗███╗   ███╗██████╗ ███████╗██████╗ "
    echo "  ██╔══██╗██╔══██╗██║     ╚██╗██╔╝      ██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗"
    echo "  ██████╔╝██████╔╝██║      ╚███╔╝ █████╗█████╗  ██╔████╔██║██████╔╝█████╗  ██║  ██║"
    echo "  ██╔═══╝ ██╔═══╝ ██║      ██╔██╗ ╚════╝██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██║  ██║"
    echo "  ██║     ██║     ███████╗██╔╝ ██╗      ███████╗██║ ╚═╝ ██║██████╔╝███████╗██████╔╝"
    echo "  ╚═╝     ╚═╝     ╚══════╝╚═╝  ╚═╝      ╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚═════╝ "
    echo -e "${RESET}"
    echo -e "  ${CYAN}Local embedding stack · pplx-embed-v1-0.6B · CPU · 100% offline${RESET}\n"
}

# ═══ Commands ═════════════════════════════════════════════════════════════════

cmd_docker() {
    check_docker
    header "Starting full Docker stack"
    cd "$SCRIPT_DIR"
    docker compose up --build -d
    echo ""
    ok "Stack running:"
    echo -e "   ${CYAN}Interface Streamlit${RESET}  →  http://localhost:8501"
    echo -e "   ${CYAN}API FastAPI${RESET}          →  http://localhost:8000"
    echo -e "   ${CYAN}API Docs${RESET}             →  http://localhost:8000/docs"
    echo ""
    log "Follow logs with: ./launch.sh docker:logs"
}

cmd_docker_api() {
    check_docker
    header "Starting API container only"
    cd "$SCRIPT_DIR"
    docker compose up --build -d api
    ok "API running → http://localhost:8000"
    ok "Docs        → http://localhost:8000/docs"
}

cmd_docker_ui() {
    check_docker
    header "Starting Streamlit container only"
    cd "$SCRIPT_DIR"
    docker compose up --build -d streamlit
    ok "Streamlit running → http://localhost:8501"
}

cmd_docker_down() {
    check_docker
    header "Stopping Docker containers"
    cd "$SCRIPT_DIR"
    docker compose down
    ok "All containers stopped."
}

cmd_docker_logs() {
    check_docker
    cd "$SCRIPT_DIR"
    docker compose logs -f
}

cmd_api() {
    check_python
    activate_venv
    header "Starting FastAPI locally"
    log "API will be available at http://localhost:8000"
    log "Press Ctrl+C to stop."
    cd "$SCRIPT_DIR"
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
}

cmd_ui() {
    check_python
    activate_venv
    header "Starting Streamlit locally"
    log "UI will be available at http://localhost:8501"
    log "Press Ctrl+C to stop."
    cd "$SCRIPT_DIR"
    # When running locally, point to localhost API
    export API_URL="${API_URL:-http://localhost:8000}"
    streamlit run streamlit_app/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --browser.gatherUsageStats false
}

cmd_all() {
    check_python
    activate_venv
    header "Starting API + Streamlit locally (background)"
    cd "$SCRIPT_DIR"

    export API_URL="http://localhost:8000"

    log "Starting API on port 8000..."
    uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    echo "$API_PID" > "$PID_FILE"

    log "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -sf http://localhost:8000/health &>/dev/null; then
            ok "API ready!"
            break
        fi
        sleep 2
        echo -n "."
    done

    log "Starting Streamlit on port 8501..."
    streamlit run streamlit_app/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --browser.gatherUsageStats false &
    UI_PID=$!
    echo "$UI_PID" >> "$PID_FILE"

    echo ""
    ok "Both services running:"
    echo -e "   ${CYAN}Streamlit${RESET}  →  http://localhost:8501"
    echo -e "   ${CYAN}API${RESET}        →  http://localhost:8000"
    echo ""
    log "Stop with: ./launch.sh stop"

    # Wait so Ctrl+C kills both
    wait
}

cmd_stop() {
    if [[ -f "$PID_FILE" ]]; then
        log "Stopping local processes..."
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" && ok "Killed PID $pid"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
        ok "Done."
    else
        warn "No PID file found. Nothing to stop."
    fi
}

cmd_install() {
    check_python
    header "Setting up virtualenv and installing project"
    cd "$SCRIPT_DIR"

    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtualenv at $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
        ok "Virtualenv created."
    fi

    source "$VENV_DIR/bin/activate"
    log "Installing project with dev extras..."
    pip install --upgrade pip -q
    pip install -e ".[dev]"
    ok "Installation complete!"
    echo ""
    echo -e "  Activate with: ${CYAN}source .venv/bin/activate${RESET}"
    echo -e "  Then run:      ${CYAN}./launch.sh api${RESET}  or  ${CYAN}./launch.sh ui${RESET}"
}

cmd_download() {
    check_python
    activate_venv
    header "Downloading model weights"
    cd "$SCRIPT_DIR"
    python scripts/download_model.py
}

cmd_projector() {
    check_python
    activate_venv
    header "Exporting embedding projector to HTML"
    cd "$SCRIPT_DIR"
    python scripts/export_projector.py "$@"
}

cmd_help() {
    print_banner
    echo -e "${BOLD}Usage:${RESET}  ./launch.sh <command> [options]\n"
    echo -e "${BOLD}Docker commands:${RESET}"
    echo -e "  ${CYAN}docker${RESET}         Start full stack (API + Streamlit) via Docker Compose"
    echo -e "  ${CYAN}docker:api${RESET}     Start only the API container"
    echo -e "  ${CYAN}docker:ui${RESET}      Start only the Streamlit container"
    echo -e "  ${CYAN}docker:down${RESET}    Stop all containers"
    echo -e "  ${CYAN}docker:logs${RESET}    Follow container logs"
    echo ""
    echo -e "${BOLD}Local Python commands (requires ./launch.sh install first):${RESET}"
    echo -e "  ${CYAN}install${RESET}        Create .venv and run pip install -e .[dev]"
    echo -e "  ${CYAN}api${RESET}            Run FastAPI locally with hot-reload"
    echo -e "  ${CYAN}ui${RESET}             Run Streamlit locally"
    echo -e "  ${CYAN}all${RESET}            Run API + Streamlit locally (background)"
    echo -e "  ${CYAN}stop${RESET}           Kill local processes started by 'all'"
    echo ""
    echo -e "${BOLD}Tools:${RESET}"
    echo -e "  ${CYAN}download${RESET}       Download model weights to ./model/"
    echo -e "  ${CYAN}projector${RESET}      Export embedding projector to standalone HTML"
    echo ""
    echo -e "${BOLD}Examples:${RESET}"
    echo -e "  ./launch.sh install          # First-time setup"
    echo -e "  ./launch.sh docker           # Production: full Docker stack"
    echo -e "  ./launch.sh api              # Dev: API only, hot-reload"
    echo -e "  ./launch.sh all              # Dev: API + UI locally"
    echo -e "  ./launch.sh projector        # Export projector HTML"
}

# ═══ Dispatch ═════════════════════════════════════════════════════════════════

CMD="${1:-help}"
shift || true

case "$CMD" in
    docker)        print_banner; cmd_docker ;;
    docker:api)    print_banner; cmd_docker_api ;;
    docker:ui)     print_banner; cmd_docker_ui ;;
    docker:down)   cmd_docker_down ;;
    docker:logs)   cmd_docker_logs ;;
    api)           print_banner; cmd_api ;;
    ui)            print_banner; cmd_ui ;;
    all)           print_banner; cmd_all ;;
    stop)          cmd_stop ;;
    install)       print_banner; cmd_install ;;
    download)      print_banner; cmd_download ;;
    projector)     print_banner; cmd_projector "$@" ;;
    help|--help|-h) cmd_help ;;
    *)
        error "Unknown command: '$CMD'"
        echo ""
        cmd_help
        exit 1
        ;;
esac
