#!/usr/bin/env python3
"""
Consolidated Test Runner

Run tests with various options:
  python run_tests.py              - Run all tests with coverage
  python run_tests.py --quick      - Run tests without coverage
  python run_tests.py test_file    - Run specific test file
"""

import sys
import subprocess
import argparse


def run_tests_with_coverage(test_path="tests/"):
    """Run tests with coverage report."""
    cmd = ["python", "-m", "pytest", test_path, "-v", "--cov=src"]
    return subprocess.call(cmd)


def run_tests_quick(test_path="tests/"):
    """Run tests without coverage."""
    cmd = ["python", "-m", "pytest", test_path, "-v"]
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated test runner for quality checks"
    )
    parser.add_argument(
        "test_path", nargs="?", default="tests/", help="Path to test file or directory"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run tests without coverage"
    )

    args = parser.parse_args()

    if args.quick:
        exit_code = run_tests_quick(args.test_path)
    else:
        exit_code = run_tests_with_coverage(args.test_path)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
