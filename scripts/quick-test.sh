#!/bin/bash

# Quick test without coverage
echo "Running quick tests (no coverage)..."
python -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "✓ Quick tests completed successfully"
else
    echo "✗ Some tests failed"
    exit 1
fi
