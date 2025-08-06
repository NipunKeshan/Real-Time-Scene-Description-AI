#!/bin/bash

# Real-Time AI Scene Description System - Quick Setup Script
# This script automates the installation and setup process

set -e  # Exit on any error

echo "🤖 Real-Time AI Scene Description System - Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        
        # Check if version is 3.8+
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible (3.8+)"
        else
            print_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
}

# Check CUDA availability
check_cuda() {
    print_status "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_success "NVIDIA GPU detected with CUDA $CUDA_VERSION"
        GPU_AVAILABLE=true
    else
        print_warning "NVIDIA GPU not detected. CPU-only mode will be used."
        GPU_AVAILABLE=false
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install based on GPU availability
    if [ "$GPU_AVAILABLE" = true ]; then
        print_status "Installing GPU-enabled PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_status "Installing other requirements..."
    pip install -r requirements.txt
    
    print_success "Dependencies installed successfully"
}

# Download models (optional)
download_models() {
    print_status "Downloading AI models..."
    
    python3 -c "
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

print('Downloading BLIP model...')
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

print('Downloading YOLO model...')
yolo = YOLO('yolov8n.pt')

print('Models downloaded successfully!')
"
    
    print_success "AI models downloaded and cached"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    python3 -c "
import sys
import torch
import cv2
import transformers
import ultralytics
import streamlit
import fastapi

print(f'✅ Python: {sys.version}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
print(f'✅ OpenCV: {cv2.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print('✅ All dependencies working!')
"
    
    print_success "Installation test passed"
}

# Create desktop shortcut (Linux)
create_shortcuts() {
    print_status "Creating shortcuts..."
    
    # Create run scripts
    cat > run_streamlit.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python -m src.main --mode streamlit
EOF

    cat > run_cli.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python -m src.main --mode cli
EOF

    cat > run_api.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python -m src.main --mode api
EOF

    chmod +x run_streamlit.sh run_cli.sh run_api.sh
    
    print_success "Run scripts created"
}

# Main installation process
main() {
    echo
    print_status "Starting installation process..."
    
    # Check system requirements
    check_python
    check_cuda
    
    # Setup environment
    create_venv
    
    # Install dependencies
    install_dependencies
    
    # Download models
    read -p "Download AI models now? This may take several minutes. (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_models
    else
        print_warning "Models will be downloaded on first run"
    fi
    
    # Test installation
    test_installation
    
    # Create shortcuts
    create_shortcuts
    
    echo
    echo "🎉 Installation completed successfully!"
    echo "=================================================="
    echo
    echo "Quick Start Commands:"
    echo "  ./run_streamlit.sh    # Launch web interface"
    echo "  ./run_cli.sh          # Launch CLI version"
    echo "  ./run_api.sh          # Launch API server"
    echo
    echo "Manual Commands:"
    echo "  source venv/bin/activate"
    echo "  python -m src.main --mode streamlit"
    echo
    echo "Documentation:"
    echo "  README.md - Full documentation"
    echo "  CONTRIBUTING.md - Development guide"
    echo
    echo "Have fun with your AI scene description system! 🚀"
}

# Run main function
main "$@"
