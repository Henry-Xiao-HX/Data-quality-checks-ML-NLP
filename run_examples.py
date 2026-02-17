#!/usr/bin/env python3
"""
Consolidated Example Runner

Run examples with various options:
  python run_examples.py           - Run all examples
  python run_examples.py example_1 - Run specific example
"""

import sys
import subprocess
import argparse
from pathlib import Path


def find_examples():
    """Find all example files."""
    examples_dir = Path("examples")
    return sorted(examples_dir.glob("example_*.py"))


def run_all_examples():
    """Run all examples."""
    examples = find_examples()

    if not examples:
        print("No examples found in examples/ directory")
        return 1

    failed = []
    for example in examples:
        print(f"\n{'=' * 70}")
        print(f"Running {example.name}...")
        print("=" * 70)
        
        result = subprocess.call([sys.executable, str(example)])
        
        if result != 0:
            failed.append(example.name)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"✗ {len(failed)} example(s) failed:")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print(f"✓ All {len(examples)} examples completed successfully")
        return 0


def run_specific_example(example_name):
    """Run a specific example."""
    # Try exact name first
    example_file = Path(f"examples/{example_name}.py")
    
    # Try with example_ prefix if not found
    if not example_file.exists():
        example_file = Path(f"examples/example_{example_name}.py")
    
    # Try without .py extension
    if not example_file.exists():
        example_name_no_ext = example_name.replace(".py", "")
        example_file = Path(f"examples/example_{example_name_no_ext}.py")

    if not example_file.exists():
        print(f"✗ Example not found: {example_name}")
        print("\nAvailable examples:")
        for ex in find_examples():
            print(f"  - {ex.stem.replace('example_', '')}")
        return 1

    print(f"Running example: {example_file.name}...")
    return subprocess.call([sys.executable, str(example_file)])


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated example runner"
    )
    parser.add_argument(
        "example", nargs="?", help="Specific example to run (without 'example_' prefix or .py)"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="List all available examples"
    )

    args = parser.parse_args()

    if args.list or (not args.example and len(sys.argv) > 1 and sys.argv[1] in ["-l", "--list"]):
        print("Available examples:")
        for ex in find_examples():
            print(f"  - {ex.stem.replace('example_', '')}")
        return 0

    if args.example:
        return run_specific_example(args.example)
    else:
        return run_all_examples()


if __name__ == "__main__":
    sys.exit(main())
