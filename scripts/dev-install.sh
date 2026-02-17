#!/bin/bash

# Install package in development mode
echo "Installing package in development mode..."
pip install -e ".[dev]"

if [ $? -eq 0 ]; then
    echo "✓ Package installed in development mode"
else
    echo "✗ Failed to install package in development mode"
    exit 1
fi
