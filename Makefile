.PHONY: help install install-dev dev-install setup-dev test test-quick lint format clean examples

help:
	@echo "Data Quality Checks - Project Management"
	@echo ""
	@echo "USAGE: make <target>"
	@echo ""
	@echo "INSTALLATION:"
	@echo "  make install          Install base dependencies"
	@echo "  make install-dev      Install base + dev dependencies"
	@echo "  make dev-install      Install package in development mode"
	@echo "  make setup-dev        Full dev setup (clean, install, lint, format, test)"
	@echo ""
	@echo "TESTING:"
	@echo "  make test             Run all tests with coverage"
	@echo "  make test-quick       Run tests without coverage"
	@echo ""
	@echo "CODE QUALITY:"
	@echo "  make lint             Run flake8 linter and mypy type checker"
	@echo "  make format           Format code with black and isort"
	@echo "  make clean            Clean cache, build, and dist files"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make examples         Run all examples"
	@echo "  make help             Show this help message"

install:
	@echo "→ Installing base dependencies..."
	pip install -r requirements.txt
	@echo "✓ Base dependencies installed"

install-dev: install
	@echo "→ Installing development dependencies..."
	pip install pytest pytest-cov black flake8 mypy isort
	@echo "✓ Development dependencies installed"

dev-install:
	@echo "→ Installing package in development mode..."
	pip install -e ".[dev]"
	@echo "✓ Package installed in development mode"

setup-dev: clean install-dev lint format test
	@echo "✓ Full development setup completed"

test:
	@echo "→ Running tests with coverage..."
	python -m pytest tests/ -v --cov=src
	@echo "✓ Tests completed"

test-quick:
	@echo "→ Running quick tests (no coverage)..."
	python -m pytest tests/ -v
	@echo "✓ Quick tests completed"

lint:
	@echo "→ Running flake8 linter..."
	flake8 src/ tests/ examples/
	@echo "→ Running mypy type checker..."
	mypy src/ --ignore-missing-imports
	@echo "✓ Linting completed"

format:
	@echo "→ Running black code formatter..."
	black src/ tests/ examples/
	@echo "→ Running isort import sorter..."
	isort src/ tests/ examples/
	@echo "✓ Code formatting completed"

clean:
	@echo "→ Cleaning Python cache files..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.mypy_cache' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "✓ Cleanup completed"

examples:
	@echo "→ Running all examples..."
	@for example in examples/example_*.py; do \
		echo ""; \
		echo "→ Running $$(basename $$example)..."; \
		python "$$example"; \
	done
	@echo ""
	@echo "✓ All examples completed"
