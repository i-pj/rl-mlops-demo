#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# RL MLOps Demo — One-Command Setup
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# What it does:
#   1. Verifies uv is installed
#   2. Installs all Python dependencies via uv
#   3. Runs the doctor health check
#   4. Prints success or detailed error info
#
# Supports:
#   - macOS (Apple Silicon + Intel)
#   - Linux (with NVIDIA GPU or CPU)
#   - Offline mode (if model .zip is pre-placed in models/)
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

info()    { echo -e "${CYAN}ℹ ${NC} $*"; }
success() { echo -e "${GREEN}✓ ${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠ ${NC} $*"; }
fail()    { echo -e "${RED}✗ ${NC} $*"; exit 1; }

# ── Banner ───────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  🏎️  RL MLOps Demo — Setup                          ║${NC}"
echo -e "${CYAN}║  From RL Agent to Production                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check uv ────────────────────────────────────────────
info "Checking for uv …"
if ! command -v uv &> /dev/null; then
    fail "uv not found. Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
UV_VERSION=$(uv --version 2>&1)
success "uv found: $UV_VERSION"

# ── Step 2: Navigate to project root ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
info "Working directory: $SCRIPT_DIR"

# ── Step 3: Install dependencies ─────────────────────────────────
info "Installing dependencies via uv sync …"
uv sync 2>&1 | tail -5
success "Dependencies installed"

# ── Step 4: Verify critical imports ──────────────────────────────
info "Verifying Python imports …"
uv run python -c "
import torch
import gymnasium
import stable_baselines3
import sb3_contrib
print(f'  torch={torch.__version__}')
print(f'  gymnasium={gymnasium.__version__}')
print(f'  sb3={stable_baselines3.__version__}')

# Check device
if torch.backends.mps.is_available():
    print('  device=MPS (Apple Silicon GPU) ✓')
elif torch.cuda.is_available():
    print(f'  device=CUDA ({torch.cuda.get_device_name()}) ✓')
else:
    print('  device=CPU')
" || fail "Import verification failed. Check dependencies."
success "All imports verified"

# ── Step 3: Doctor check ─────────────────────────────────────────
info "Running health checks (this may take a moment to load torch) …"
uv run rl-demo doctor

# ── Done ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅  Setup complete!                                 ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                      ║${NC}"
echo -e "${GREEN}║  Quick start:                                        ║${NC}"
echo -e "${GREEN}║    uv run rl-demo train --timesteps 50000            ║${NC}"
echo -e "${GREEN}║    uv run rl-demo eval -n 5 --clean                  ║${NC}"
echo -e "${GREEN}║    uv run rl-demo demo --clean                       ║${NC}"
echo -e "${GREEN}║    uv run rl-demo tune --n-trials 5                  ║${NC}"
echo -e "${GREEN}║    uv run rl-demo mlflow-ui                          ║${NC}"
echo -e "${GREEN}║                                                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
