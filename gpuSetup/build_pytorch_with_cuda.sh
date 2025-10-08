#!/bin/bash
# Build PyTorch from source with CUDA support for RTX 5070 (sm_120)

set -e  # Exit on error

echo "=============================================="
echo "  Building PyTorch with CUDA Support"
echo "  Target: RTX 5070 (sm_120 / CUDA 13.0)"
echo "=============================================="
echo ""

# Configuration
PYTORCH_SOURCE_DIR="$HOME/pytorch"
VENV_DIR="/home/aro/Documents/ObjectRec/.venv"
CUDA_VERSION="12.4"  # Compatible with your CUDA 13.0 driver

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: CUDA not found!"
    echo "Please install CUDA toolkit first:"
    echo "  sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

echo "✓ CUDA found: $(nvcc --version | grep release)"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

echo "✓ Virtual environment found"
echo ""

# Clone PyTorch if not exists
if [ ! -d "$PYTORCH_SOURCE_DIR" ]; then
    echo "📥 Cloning PyTorch repository..."
    cd "$HOME"
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    # Checkout a stable tag (adjust as needed)
    git checkout v2.6.0
    git submodule sync
    git submodule update --init --recursive
else
    echo "✓ PyTorch source found at $PYTORCH_SOURCE_DIR"
    cd "$PYTORCH_SOURCE_DIR"
fi

echo ""
echo "=============================================="
echo "  Installing Build Dependencies"
echo "=============================================="
echo ""

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install build dependencies
pip install -U pip setuptools wheel
pip install cmake ninja
pip install numpy pyyaml mkl mkl-include setuptools cffi typing_extensions future six requests dataclasses

echo ""
echo "=============================================="
echo "  Configuring Build with CUDA"
echo "=============================================="
echo ""

# Set build environment variables
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which python))/../"}
export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="12.0"  # sm_120 for RTX 5070
export CUDA_HOME=/usr/local/cuda
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu
export CUDNN_INCLUDE_DIR=/usr/include
export MAX_JOBS=4  # Adjust based on your CPU cores
export BUILD_TEST=0  # Skip tests for faster build
export USE_MKLDNN=1
export USE_DISTRIBUTED=1

echo "Build Configuration:"
echo "  USE_CUDA: $USE_CUDA"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  MAX_JOBS: $MAX_JOBS"
echo ""

echo "=============================================="
echo "  Building PyTorch (This will take 30-60 min)"
echo "=============================================="
echo ""

# Clean previous build
python setup.py clean

# Build and install
echo "Starting build at $(date)"
python setup.py develop

echo ""
echo "=============================================="
echo "  Build Complete!"
echo "=============================================="
echo ""

echo "📦 Installing torchvision..."
pip install torchvision --index-url https://download.pytorch.org/whl/cu124 --no-deps

echo ""
echo "=============================================="
echo "  Verifying Installation"
echo "=============================================="
echo ""

python -c "
import torch
print('✓ PyTorch Version:', torch.__version__)
print('✓ CUDA Available:', torch.cuda.is_available())
print('✓ CUDA Version:', torch.version.cuda)
if torch.cuda.is_available():
    print('✓ GPU Device:', torch.cuda.get_device_name(0))
    print('')
    print('Testing GPU computation...')
    t = torch.tensor([1.0, 2.0, 3.0]).cuda()
    result = t * 2
    print('✓ GPU Test Result:', result.cpu().numpy())
    print('')
    print('🎉 GPU IS FULLY WORKING!')
else:
    print('❌ CUDA not available - build may have failed')
"

echo ""
echo "=============================================="
echo "  Installation Summary"
echo "=============================================="
echo ""
echo "Virtual Environment: $VENV_DIR"
echo "PyTorch Source: $PYTORCH_SOURCE_DIR"
echo ""
echo "To use GPU detection:"
echo "  cd /home/aro/Documents/ObjectRec"
echo "  source .venv/bin/activate"
echo "  python live_object_detection.py"
echo ""
echo "The script will automatically detect and use GPU!"
echo ""
