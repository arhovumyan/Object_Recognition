#!/bin/bash
# Script to upgrade PyTorch for RTX 5070 (Blackwell architecture) support
# This requires PyTorch 2.6+ which supports sm_120 compute capability

echo "=========================================="
echo "Upgrading PyTorch for RTX 5070 GPU Support"
echo "=========================================="
echo ""
echo "Your RTX 5070 requires CUDA compute capability sm_120"
echo "Current PyTorch version only supports up to sm_90"
echo ""
echo "Installing PyTorch nightly build with CUDA 12.4 support..."
echo ""

# Uninstall old PyTorch
pip3 uninstall -y torch torchvision torchaudio

# Install PyTorch nightly with CUDA 12.4 support
# This version should support sm_120 (Blackwell)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Upgrade complete!"
echo ""
echo "Note: If GPU still doesn't work, PyTorch may not yet support Blackwell."
echo "In that case, the script will continue to use CPU mode which works well."
