#!/bin/bash

# Full development setup
set -e  # Exit on any error

echo "Starting full development setup..."
echo ""

echo "Step 1: Cleaning cache..."
bash scripts/clean.sh

echo ""
echo "Step 2: Installing development dependencies..."
bash scripts/install-dev.sh

echo ""
echo "Step 3: Running linter..."
bash scripts/lint.sh

echo ""
echo "Step 4: Formatting code..."
bash scripts/format.sh

echo ""
echo "Step 5: Running tests..."
bash scripts/test.sh

echo ""
echo "✓ Development environment ready!"
