#!/bin/bash

# Quick Start Script for Flyer Processing System
# This script helps set up and test the enhanced flyer API

set -e

echo "========================================================================"
echo "  Flyer Processing System - Quick Start"
echo "========================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}Current directory:${NC} $SCRIPT_DIR"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check Python version
echo "1. Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python found: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    print_status "Python found: $PYTHON_VERSION"
    PYTHON_CMD="python"
else
    print_error "Python not found! Please install Python 3.8 or higher"
    exit 1
fi
echo ""

# Check pip
echo "2. Checking pip installation..."
if $PYTHON_CMD -m pip --version &> /dev/null; then
    PIP_VERSION=$($PYTHON_CMD -m pip --version | cut -d' ' -f2)
    print_status "pip found: $PIP_VERSION"
else
    print_error "pip not found! Please install pip"
    exit 1
fi
echo ""

# Check if requirements.txt exists
echo "3. Checking requirements file..."
if [ -f "requirements.txt" ]; then
    print_status "requirements.txt found"
else
    print_warning "requirements.txt not found, creating it..."
    cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
opencv-python==4.8.1.78
Pillow==10.1.0
numpy==1.24.3
transformers==4.46.3
torch==2.1.1
sqlalchemy==2.0.23
requests==2.31.0
pydantic==2.5.0
EOF
    print_status "requirements.txt created"
fi
echo ""

# Install dependencies
echo "4. Installing dependencies..."
read -p "   Install/upgrade Python packages? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Installing packages (this may take a few minutes)..."
    $PYTHON_CMD -m pip install -r requirements.txt --upgrade
    print_status "Packages installed"
else
    print_warning "Skipped package installation"
fi
echo ""

# Create output directory
echo "5. Creating output directories..."
mkdir -p outputs/flyer_crops
print_status "Output directory created: outputs/flyer_crops/"
echo ""

# Create test_images directory
echo "6. Checking test images directory..."
if [ -d "test_images" ]; then
    IMAGE_COUNT=$(ls test_images/*.{jpg,png} 2>/dev/null | wc -l)
    if [ $IMAGE_COUNT -gt 0 ]; then
        print_status "Found $IMAGE_COUNT test images"
    else
        print_warning "test_images/ exists but no images found"
        print_info "Please add JPG or PNG flyer images to test_images/"
    fi
else
    mkdir -p test_images
    print_warning "Created test_images/ directory"
    print_info "Please add JPG or PNG flyer images to test_images/"
fi
echo ""

# Check if model is needed
echo "7. DeepSeek-OCR Model..."
print_info "The model (~10GB) will be downloaded on first run"
print_info "Make sure you have sufficient disk space and internet connection"
echo ""

# Summary
echo "========================================================================"
echo "  Setup Complete!"
echo "========================================================================"
echo ""
echo "Next Steps:"
echo ""
echo "  1. Add test flyer images to: test_images/"
echo ""
echo "  2. Start the API server:"
echo "     cd $SCRIPT_DIR"
echo "     $PYTHON_CMD app_enhanced.py"
echo ""
echo "  3. In another terminal, run tests:"
echo "     cd $SCRIPT_DIR"
echo "     $PYTHON_CMD test_enhanced.py"
echo ""
echo "  4. Or test with curl:"
echo "     curl -X POST http://localhost:8000/detect \\"
echo "          -F \"file=@test_images/your_flyer.jpg\" \\"
echo "          -F \"flyer_id=test_001\""
echo ""
echo "Documentation:"
echo "  - Implementation Guide: IMPLEMENTATION_GUIDE.md"
echo "  - Analysis Summary: ANALYSIS_SUMMARY.md"
echo ""
echo "========================================================================"
echo ""

# Ask if user wants to start the server
read -p "Start the API server now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "========================================================================"
    echo "  Starting Flyer Processing API Server"
    echo "========================================================================"
    echo ""
    print_info "Server will start on http://localhost:8000"
    print_info "Press Ctrl+C to stop the server"
    echo ""
    sleep 2
    $PYTHON_CMD app_enhanced.py
else
    echo ""
    print_info "Setup complete! Run '$PYTHON_CMD app_enhanced.py' when ready"
    echo ""
fi
