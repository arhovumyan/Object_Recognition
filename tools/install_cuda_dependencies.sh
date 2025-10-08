#!/bin/bash
# Step 1: Install CUDA Toolkit and Dependencies for Building PyTorch

set -e

echo "=============================================="
echo "  Installing CUDA Build Dependencies"
echo "=============================================="
echo ""

echo "This will install:"
echo "  - CUDA Toolkit (nvcc compiler)"
echo "  - CUDNN development libraries"
echo "  - Build tools (cmake, ninja)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "📦 Updating package lists..."
sudo apt update

echo ""
echo "📦 Installing CUDA Toolkit..."
sudo apt install -y nvidia-cuda-toolkit

echo ""
echo "📦 Installing CUDNN..."
sudo apt install -y libcudnn8 libcudnn8-dev

echo ""
echo "📦 Installing build tools..."
sudo apt install -y build-essential cmake ninja-build git

echo ""
echo "📦 Installing Python development headers..."
sudo apt install -y python3-dev

echo ""
echo "=============================================="
echo "  Verifying Installation"
echo "=============================================="
echo ""

echo "✓ CUDA Compiler:"
nvcc --version | grep release

echo ""
echo "✓ CUDNN Headers:"
ls /usr/include/cudnn*.h | head -1

echo ""
echo "✓ CUDNN Libraries:"
ls /usr/lib/x86_64-linux-gnu/libcudnn*.so* | head -1

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""

echo "Now you can build PyTorch with:"
echo "  ./build_pytorch_with_cuda.sh"
echo ""
