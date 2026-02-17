#!/bin/bash

# Format code with black and isort
echo "Running black code formatter..."
black src/ tests/ examples/

if [ $? -ne 0 ]; then
    echo "✗ Black formatting failed"
    exit 1
fi

echo "Running isort import sorter..."
isort src/ tests/ examples/

if [ $? -eq 0 ]; then
    echo "✓ Code formatting completed successfully"
else
    echo "✗ Import sorting failed"
    exit 1
fi
