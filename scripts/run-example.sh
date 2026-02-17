#!/bin/bash

# Run a single example
# Usage: bash scripts/run-example.sh example_1_basic_usage.py

if [ -z "$1" ]; then
    echo "Usage: bash run-example.sh <example_file_name>"
    echo "Example: bash run-example.sh example_1_basic_usage.py"
    exit 1
fi

EXAMPLE=$1

echo "Running example: $EXAMPLE..."
python examples/"$EXAMPLE"

if [ $? -eq 0 ]; then
    echo "✓ Example ran successfully"
else
    echo "✗ Example failed"
    exit 1
fi
