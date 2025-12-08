#!/bin/bash
# Setup script for TeachTime TutorBench evaluation environment using uv

set -e  # Exit on error

echo "Setting up TeachTime evaluation environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment with uv
echo "Creating virtual environment..."
uv venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install all dependencies with uv
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run evaluation:"
echo "  python run_eval.py --provider together --samples 10"
echo ""
echo "Required: Set API keys in .env or environment:"
echo "  ANTHROPIC_API_KEY=your-key"
echo "  TOGETHER_API_KEY=your-key"
