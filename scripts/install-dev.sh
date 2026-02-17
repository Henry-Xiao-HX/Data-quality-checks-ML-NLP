#!/bin/bash

# Install development dependencies
echo "Installing base dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "✗ Failed to install base dependencies"
    exit 1
fi

echo "Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy isort

if [ $? -eq 0 ]; then
    echo "✓ Development dependencies installed successfully"
else
    echo "✗ Failed to install development dependencies"
    exit 1
fi
