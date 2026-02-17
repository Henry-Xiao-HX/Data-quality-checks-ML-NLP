#!/bin/bash

# Clean cache and build files
echo "Cleaning Python cache files..."

echo "  Removing .pyc files..."
find . -type f -name '*.pyc' -delete

echo "  Removing __pycache__ directories..."
find . -type d -name '__pycache__' -delete

echo "  Removing .pytest_cache..."
find . -type d -name '.pytest_cache' -delete

echo "  Removing .mypy_cache..."
find . -type d -name '.mypy_cache' -delete

echo "  Removing egg-info directories..."
find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null

echo "  Removing build directory..."
rm -rf build/

echo "  Removing dist directory..."
rm -rf dist/

echo "✓ Cleanup completed successfully"
