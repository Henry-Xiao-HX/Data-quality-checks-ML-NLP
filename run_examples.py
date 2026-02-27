#!/usr/bin/env python3
"""
Consolidated Example Runner

Discovers and runs examples organized in subdirectories:
  examples/generative_ai/        - BLEU and ROUGE evaluation examples
  examples/binary_classification/ - Binary classification metric examples
  examples/regression/            - Regression metric examples

Usage:
  python run_examples.py                        - Run all examples
  python run_examples.py generative_ai          - Run all generative AI examples
  python run_examples.py binary_classification  - Run all binary classification examples
  python run_examples.py regression             - Run all regression examples
  python run_examples.py generative_ai/example_1_basic_bleu_rouge  - Run specific example
  python run_examples.py -l                     - List all available examples
"""

import sys
import subprocess
import argparse
from pathlib import Path

EXAMPLES_DIR = Path("examples")

CATEGORIES = {
    "generative_ai": "Generative AI (BLEU & ROUGE)",
    "binary_classification": "Binary Classification",
    "regression": "Regression",
}


def find_examples(category: str = None):  # type: ignore[assignment]
    """Find all example files, optionally filtered by category."""
    if category:
        search_dir = EXAMPLES_DIR / category
        if not search_dir.exists():
            return []
        return sorted(search_dir.glob("example_*.py"))

    all_examples = []
    for cat in CATEGORIES:
        cat_dir = EXAMPLES_DIR / cat
        if cat_dir.exists():
            all_examples.extend(sorted(cat_dir.glob("example_*.py")))
    return all_examples


def list_examples():
    """Print all available examples grouped by category."""
    print("Available examples:\n")
    for cat, label in CATEGORIES.items():
        cat_dir = EXAMPLES_DIR / cat
        examples = sorted(cat_dir.glob("example_*.py")) if cat_dir.exists() else []
        print(f"  [{label}]")
        if examples:
            for ex in examples:
                print(f"    - {cat}/{ex.stem}")
        else:
            print("    (no examples found)")
        print()


def run_example(example_path: Path) -> int:
    """Run a single example file and return its exit code."""
    result = subprocess.call([sys.executable, str(example_path)])
    return result


def run_examples_list(examples):
    """Run a list of example files, report results."""
    if not examples:
        print("No examples found.")
        return 1

    failed = []
    for example in examples:
        category = example.parent.name
        label = CATEGORIES.get(category, category)
        print(f"\n{'=' * 70}")
        print(f"[{label}] Running {example.name}...")
        print("=" * 70)

        if run_example(example) != 0:
            failed.append(f"{category}/{example.name}")

    print(f"\n{'=' * 70}")
    if failed:
        print(f"✗ {len(failed)} example(s) failed:")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print(f"✓ All {len(examples)} examples completed successfully")
        return 0


def resolve_example(name: str):
    """
    Resolve an example name to a Path. Accepts:
      - category name (e.g. 'generative_ai')
      - category/stem (e.g. 'generative_ai/example_1_basic_bleu_rouge')
      - bare stem (e.g. 'example_1_basic_bleu_rouge') — searches all categories
    """
    # Check if it's a known category
    if name in CATEGORIES:
        return find_examples(name)

    # Check for category/stem format
    if "/" in name:
        parts = name.split("/", 1)
        cat, stem = parts[0], parts[1]
        candidate = EXAMPLES_DIR / cat / f"{stem}.py"
        if not candidate.exists():
            candidate = EXAMPLES_DIR / cat / stem  # already has .py
        if candidate.exists():
            return [candidate]
        print(f"✗ Example not found: {name}")
        return None

    # Search all categories for a matching stem
    for cat in CATEGORIES:
        candidate = EXAMPLES_DIR / cat / f"{name}.py"
        if candidate.exists():
            return [candidate]

    print(f"✗ Example not found: '{name}'")
    print("\nRun with -l to list all available examples.")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run examples for Evaluating-ML-GenAI-Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Category or example to run (e.g. 'generative_ai' or 'generative_ai/example_1_basic_bleu_rouge')"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available examples"
    )

    args = parser.parse_args()

    if args.list:
        list_examples()
        return 0

    if args.target:
        examples = resolve_example(args.target)
        if examples is None:
            return 1
        return run_examples_list(examples)

    # Default: run all examples
    return run_examples_list(find_examples())


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
