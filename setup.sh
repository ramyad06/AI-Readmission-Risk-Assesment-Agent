#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-time environment setup for Hospital Readmission Prediction Agent
# Usage: bash setup.sh
# =============================================================================

set -e

echo "=============================================="
echo " Hospital Readmission Prediction Agent Setup"
echo "=============================================="

# ── Step 1: Create virtual environment ────────────────────────────────────────
echo ""
echo "[1/5] Creating Python virtual environment..."
python -m venv venv
echo "      ✔ Virtual environment created at ./venv"

# ── Step 2: Activate venv ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Activating virtual environment..."
source venv/Scripts/activate
echo "      ✔ Virtual environment activated"

# ── Step 3: Upgrade pip ───────────────────────────────────────────────────────
echo ""
echo "[3/5] Upgrading pip..."
pip install --upgrade pip --quiet
echo "      ✔ pip upgraded"

# ── Step 4: Install dependencies ──────────────────────────────────────────────
echo ""
echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt
echo "      ✔ All packages installed"

# ── Step 5: Verify Ollama & pull llama3:8b ────────────────────────────────────
echo ""
echo "[5/5] Checking Ollama availability..."

if ! command -v ollama &> /dev/null; then
    echo "      ✘ Ollama CLI not found."
    echo "        Please install Ollama from https://ollama.com/download"
    echo "        Then re-run this script."
    exit 1
fi

echo "      ✔ Ollama CLI detected"
echo "      Pulling llama3:8b model (this may take a few minutes on first run)..."
ollama pull llama3:8b

echo ""
echo "      ✔ Model llama3:8b is ready"
echo ""
echo "      Verifying model with a quick test..."
RESULT=$(ollama run llama3:8b "Reply with exactly: READY" 2>/dev/null || echo "ERROR")

if echo "$RESULT" | grep -qi "READY"; then
    echo "      ✔ Model responds correctly"
else
    echo "      ⚠ Model responded (output may vary): $RESULT"
fi

echo ""
echo "=============================================="
echo " Setup Complete!"
echo "=============================================="
echo ""
echo "  To start the application:"
echo "    1. source venv/Scripts/activate"
echo "    2. ollama serve  (if not already running)"
echo "    3. streamlit run app.py"
echo ""
