#!/bin/bash

# Install basic dependencies
echo "Installing base dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Base dependencies installed successfully"
else
    echo "✗ Failed to install base dependencies"
    exit 1
fi
