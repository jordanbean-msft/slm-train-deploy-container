#!/bin/bash
# Quick setup script for training data

set -e

echo "🚀 Setting up training data..."

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data/processed
fi

# Check if example file exists
if [ ! -f "data/training_data.jsonl.example" ]; then
    echo "❌ Example file not found!"
    exit 1
fi

# Check if training data already exists
if [ -f "data/training_data.jsonl" ]; then
    echo "⚠️  data/training_data.jsonl already exists!"
    read -p "Do you want to overwrite it with the example? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing training_data.jsonl"
        exit 0
    fi
fi

# Copy example to training data
echo "📋 Copying example file to data/training_data.jsonl..."
cp data/training_data.jsonl.example data/training_data.jsonl

echo "✅ Training data setup complete!"
echo ""
echo "📁 File location: data/training_data.jsonl"
echo "📝 This is example data with 10 samples."
echo ""
echo "Next steps:"
echo "  1. Edit data/training_data.jsonl with your own training examples"
echo "  2. Run notebooks/02-prepare-data.ipynb to validate and upload"
echo ""
echo "Example format:"
echo '  {"prompt": "Your input", "completion": "Expected output"}'
