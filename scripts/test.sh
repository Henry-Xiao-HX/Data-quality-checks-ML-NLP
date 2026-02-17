#!/bin/bash

# Run all tests with coverage
echo "Running tests with coverage..."
python -m pytest tests/ -v --cov=src

if [ $? -eq 0 ]; then
    echo "✓ Tests completed successfully"
else
    echo "✗ Some tests failed"
    exit 1
fi
