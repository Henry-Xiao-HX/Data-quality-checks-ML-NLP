#!/bin/bash

# Legacy orchestration script - Deprecated
# Use manage.sh, run_tests.py, or run_examples.py instead
#
# This script is maintained for backward compatibility.
# For comprehensive project management, use:
#   - bash manage.sh <command>  (bash alternative with detailed options)
#   - python run_tests.py        (python test runner)
#   - python run_examples.py     (python example runner)
#
# Examples:
#   bash manage.sh help
#   python run_tests.py
#   python run_examples.py -l

exec bash manage.sh "$@"
