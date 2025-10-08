#!/bin/bash
# Install CUDA 13.0 Toolkit for RTX 5070 (sm_120) support

set -e

echo "=============================================="
echo "  Installing CUDA 13.0 Toolkit"
echo "  Required for RTX 5070 (sm_120) Support"
echo "=============================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires sudo privileges."
    echo "Please run with: sudo ./install_cuda_13.sh"
    exit 1
fi

echo "📥 Adding NVIDIA CUDA repository..."

# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

echo ""
echo "📦 Updating package lists..."
apt-get update

echo ""
echo "⚙️  Installing CUDA 13.0 Toolkit..."
echo "   This may take 10-15 minutes..."
apt-get install -y cuda-toolkit-13-0

echo ""
echo "=============================================="
echo "  CUDA 13.0 Installation Complete!"
echo "=============================================="
echo ""

# Check installation
if [ -d "/usr/local/cuda-13.0" ]; then
    echo "✅ CUDA 13.0 installed at: /usr/local/cuda-13.0"
    
    # Update cuda symlink to point to 13.0
    rm -f /usr/local/cuda
    ln -s /usr/local/cuda-13.0 /usr/local/cuda
    
    echo "✅ Updated /usr/local/cuda symlink to CUDA 13.0"
    echo ""
    echo "Verify installation:"
    /usr/local/cuda-13.0/bin/nvcc --version
else
    echo "❌ CUDA 13.0 installation may have failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "  Next Steps"
echo "=============================================="
echo ""
echo "1. Rebuild PyTorch with CUDA 13.0:"
echo "   cd /home/aro/Documents/ObjectRec"
echo "   ./build_pytorch_sm120.sh"
echo ""
echo "2. Or use the pre-built approach:"
echo "   Check if newer PyTorch nightly supports sm_120"
echo ""
