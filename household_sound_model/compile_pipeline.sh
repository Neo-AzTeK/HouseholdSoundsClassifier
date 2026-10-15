#!/usr/bin/env bash
# End-to-end pipeline to generate Hailo .hef from PyTorch checkpoint
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

HW_ARCH="${1:-hailo10h}"
echo "=========================================================="
echo " Household Sound Classifier - Hailo HEF Pipeline"
echo " Target Architecture: ${HW_ARCH}"
echo " Python Interpreter:   ${PYTHON}"
echo "=========================================================="

echo ""
echo "[Step 1/3] Exporting PyTorch checkpoint to ONNX..."
$PYTHON export_onnx.py

echo ""
echo "[Step 2/3] Generating calibration dataset for PTQ quantization..."
$PYTHON prepare_calibration.py --num-samples 128

echo ""
echo "[Step 3/3] Compiling ONNX to Hailo HEF (${HW_ARCH})..."
$PYTHON compile_hef.py --hw-arch "$HW_ARCH"

echo ""
echo "Pipeline execution finished successfully."
