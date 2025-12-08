#!/bin/bash
# Setup script for TeachTime TutorBench evaluation environment

set -e  # Exit on error

echo "🔧 Setting up TeachTime evaluation environment..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✓ Virtual environment created"
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "1. Set your API keys in .env file:"
echo "   export ANTHROPIC_API_KEY='your-key-here'"
echo "   export TOGETHER_API_KEY='your-key-here'"
echo ""
echo "2. Try the example:"
echo "   python -m evals.example"
echo ""
