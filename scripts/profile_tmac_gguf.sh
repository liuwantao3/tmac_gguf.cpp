#!/bin/bash
#
# profile_tmac_gguf.sh — Profile tmac_gguf with Xcode Instruments
#
# Usage:
#   ./scripts/profile_tmac_gguf.sh [n_tokens]
#
# Steps:
#   1. Build tmac_gguf with --profile-wait
#   2. Launch it — it pauses, prints PID, waits for Enter from /dev/tty
#   3. Attach Xcode Instruments to the printed PID
#   4. Press Enter to resume inference
#

set -e

MODEL_PATH="$HOME/fpga/models/model.tmac"
N_TOKENS="${1:-30}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FPGA_SIM_DIR="$HOME/fpga/sim"

SOURCES="$PROJECT_DIR/tmac_gguf.cpp $FPGA_SIM_DIR/matmul_q8.cpp"
INCLUDES="-I$FPGA_SIM_DIR"
FRAMEWORKS="-framework Metal -framework MetalPerformanceShaders -framework Foundation -lobjc"

BINARY="$PROJECT_DIR/tmac_gguf_profiled"

TRACE_DIR="$PROJECT_DIR/tmac_gguf_trace"
rm -rf "$TRACE_DIR"
mkdir -p "$TRACE_DIR"

echo "============================================"
echo "  tmac_gguf Profiler Setup"
echo "============================================"
echo "  Model:    $MODEL_PATH"
echo "  Tokens:   $N_TOKENS"
echo "  Binary:   $BINARY"
echo "============================================"
echo ""

#
# Build
#
echo "[1/3] Building tmac_gguf (--profile-wait)..."
cd "$PROJECT_DIR"
clang++ -x objective-c++ -o "$BINARY" $SOURCES $INCLUDES $FRAMEWORKS -std=c++17 -O2

if [ ! -f "$BINARY" ]; then
    echo "Build failed!"
    exit 1
fi
echo "      Built: $BINARY"
echo ""

#
# Create prompt tokens file
#
TOKEN_FILE="$TRACE_DIR/prompt_tokens.txt"
echo -e "9906\n1050\n374\n1953\n1291" > "$TOKEN_FILE"

#
# Launch with --profile-wait
# --profile-wait reads from /dev/tty (not stdin) so we can control when it continues
#
echo "[2/3] Starting tmac_gguf with --profile-wait..."
echo "      Command: $BINARY $MODEL_PATH --metal-fused --generate $N_TOKENS --profile-wait"
echo ""

# Pass tokens via stdin (not tty), but --profile-wait reads from tty
"$BINARY" "$MODEL_PATH" --metal-fused --generate "$N_TOKENS" --profile-wait < "$TOKEN_FILE" 2>&1 &
PID=$!

echo "      PID: $PID"
echo ""

#
# Instructions
#
echo "============================================"
echo "  READY TO ATTACH — DO THIS NOW"
echo "============================================"
echo ""
echo "  1. Open Xcode Instruments.app"
echo "  2. Choose 'Time Profiler' or 'Metal System Trace'"
echo "  3. Press Cmd+Shift+A (or click Attach to Process)"
echo "  4. Find and select PID $PID"
echo "  5. Click Attach"
echo "  6. Then press Enter here to START inference..."
echo ""
read -r dummy

echo "[3/3] Inference started (profiling)..."
echo ""

# Wait for process to finish
wait $PID 2>/dev/null || true

echo ""
echo "============================================"
echo "  Done"
echo "============================================"
echo ""
echo "Stop the recording in Instruments and save."
echo "Output dir: $TRACE_DIR/"
echo ""
