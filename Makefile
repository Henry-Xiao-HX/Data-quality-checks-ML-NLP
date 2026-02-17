.PHONY: install install-dev test lint format clean run-examples

# Install package dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Run all tests
test:
	python -m pytest tests/ -v --cov=src

# Run specific test file
test-specific:
	python -m pytest tests/$(TEST_FILE) -v

# Lint code
lint:
	flake8 src/ tests/ examples/
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

# Clean cache and build files
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.mypy_cache' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build/ dist/

# Run examples
run-examples:
	@echo "Running Example 1: Basic Usage"
	cd examples && python example_1_basic_usage.py
	@echo "\n---\n"
	@echo "Running Example 2: Batch Aggregation"
	cd examples && python example_2_batch_aggregation.py
	@echo "\n---\n"
	@echo "Running Example 3: Quality Detection"
	cd examples && python example_3_quality_detection.py

# Run single example
run-example:
	cd examples && python $(EXAMPLE)

# Install package in development mode
dev-install:
	pip install -e ".[dev]"

# Quick test (no coverage)
quick-test:
	python -m pytest tests/ -v

# Full development setup
setup-dev: clean install-dev lint format test
	@echo "Development environment ready!"

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make test            - Run all tests with coverage"
	@echo "  make test-specific   - Run specific test (TEST_FILE=test_name.py)"
	@echo "  make lint            - Lint code"
	@echo "  make format          - Format code with black"
	@echo "  make clean           - Clean cache files"
	@echo "  make run-examples    - Run all examples"
	@echo "  make run-example     - Run specific example (EXAMPLE=example_name.py)"
	@echo "  make setup-dev       - Full development setup"
	@echo "  make help            - Show this help message"
