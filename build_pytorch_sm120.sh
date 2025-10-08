#!/bin/bash
# Build PyTorch from source with sm_120 support for RTX 5070

set -e

echo "=============================================="
echo "  Building PyTorch with sm_120 Support"
echo "  Target: RTX 5070 (Blackwell Architecture)"
echo "=============================================="
echo ""

cd /home/aro/Documents/ObjectRec

# Activate virtual environment
source .venv/bin/activate

echo "📦 Uninstalling existing PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo ""
echo "📥 Setting up PyTorch source..."
cd pytorch

# Get the latest code
echo "Fetching latest updates..."
git fetch origin
git checkout main
git pull origin main
git submodule sync
git submodule update --init --recursive

echo ""
echo "🔧 Installing build dependencies..."
pip install -U pip setuptools wheel
pip install cmake ninja numpy pyyaml mkl mkl-include cffi typing_extensions

echo ""
echo "=============================================="
echo "  Configuring Build"
echo "=============================================="
echo ""

# Clean previous build
echo "Cleaning previous build..."
python setup.py clean
rm -rf build/

# Set build environment variables for sm_120
export CMAKE_PREFIX_PATH="$(dirname $(which python))/../"
export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="12.0"  # sm_120 for RTX 5070 (Blackwell)
export CUDA_HOME=/usr/local/cuda-13.0
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu
export CUDNN_INCLUDE_DIR=/usr/include
export MAX_JOBS=6  # Use 6 parallel jobs
export BUILD_TEST=0  # Skip tests
export USE_MKLDNN=1
export USE_DISTRIBUTED=0  # Disable to speed up build
export BUILD_CAFFE2=0  # Disable Caffe2
export USE_NCCL=0  # Disable NCCL

# Use GCC 12 which is compatible with CUDA 12.4
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12

# CUDA compiler flags
export TORCH_NVCC_FLAGS="-std=c++17"
export CXXFLAGS="-std=c++17"

echo "Build Configuration:"
echo "  USE_CUDA: $USE_CUDA"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST (sm_120)"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  MAX_JOBS: $MAX_JOBS"
echo ""

echo "=============================================="
echo "  Building PyTorch"
echo "  ⏱️  Estimated time: 30-60 minutes"
echo "=============================================="
echo ""

# Build with develop mode (faster than install)
echo "⚙️  Starting build at $(date)..."
python setup.py develop

echo ""
echo "=============================================="
echo "  Build Complete!"
echo "=============================================="
echo ""

echo "✅ Testing PyTorch installation..."
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('GPU Device:', torch.cuda.get_device_name(0))
    print()
    print('🧪 Testing GPU computation...')
    try:
        t = torch.tensor([1.0, 2.0, 3.0]).cuda()
        result = t * 2
        print('✅ GPU Test PASSED!')
        print('   Result:', result.cpu().numpy())
        print()
        print('🎉 GPU IS FULLY WORKING!')
    except Exception as e:
        print('❌ GPU Test FAILED:', str(e))
        print('   The build completed but GPU operations are not working.')
else:
    print('❌ CUDA not available')
"

echo ""
echo "=============================================="
echo "  Next Steps"
echo "=============================================="
echo ""
echo "To run object detection with GPU:"
echo "  cd /home/aro/Documents/ObjectRec"
echo "  ./launch_detection.sh"
echo ""
