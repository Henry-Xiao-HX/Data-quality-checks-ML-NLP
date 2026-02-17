#!/bin/bash

# Run all examples
echo "Running Example 1: Basic Usage"
python examples/example_1_basic_usage.py

if [ $? -ne 0 ]; then
    echo "✗ Example 1 failed"
    exit 1
fi

echo ""
echo "---"
echo ""

echo "Running Example 2: Batch Aggregation"
python examples/example_2_batch_aggregation.py

if [ $? -ne 0 ]; then
    echo "✗ Example 2 failed"
    exit 1
fi

echo ""
echo "---"
echo ""

echo "Running Example 3: Quality Detection"
python examples/example_3_quality_detection.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All examples ran successfully"
else
    echo "✗ Example 3 failed"
    exit 1
fi
