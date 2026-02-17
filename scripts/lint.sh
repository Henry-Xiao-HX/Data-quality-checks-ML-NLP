#!/bin/bash

# Lint code
echo "Running flake8 linter..."
flake8 src/ tests/ examples/

if [ $? -ne 0 ]; then
    echo "✗ Linting failed"
    exit 1
fi

echo "Running mypy type checker..."
mypy src/ --ignore-missing-imports

if [ $? -eq 0 ]; then
    echo "✓ Linting completed successfully"
else
    echo "✗ Type checking failed"
    exit 1
fi
