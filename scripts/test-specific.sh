#!/bin/bash

# Run specific test file
# Usage: bash scripts/test-specific.sh test_file_name.py

if [ -z "$1" ]; then
    echo "Usage: bash test-specific.sh <test_file_name>"
    echo "Example: bash test-specific.sh test_data_quality_checker.py"
    exit 1
fi

TEST_FILE=$1

echo "Running specific test file: $TEST_FILE..."
python -m pytest tests/"$TEST_FILE" -v

if [ $? -eq 0 ]; then
    echo "✓ Test completed successfully"
else
    echo "✗ Some tests failed"
    exit 1
fi
