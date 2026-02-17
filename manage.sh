#!/bin/bash

# Consolidated Project Management Script
# Usage: bash manage.sh <command> [args]

set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Helper functions
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_info() { echo -e "${YELLOW}→${NC} $1"; }

# Show help
show_help() {
    cat << EOF
Data Quality Checks - Project Management

USAGE:
  bash manage.sh <command> [args]

INSTALLATION:
  install                Install base dependencies
  install-dev            Install base + dev dependencies  
  dev-install            Install package in development mode
  setup-dev              Full dev setup (clean, install, lint, format, test)

TESTING:
  test                   Run all tests with coverage
  test-quick             Run tests without coverage
  test <file>            Run specific test file

CODE QUALITY:
  lint                   Run flake8 linter and mypy type checker
  format                 Format code with black and isort
  clean                  Clean cache, build, and dist files

EXAMPLES:
  examples               Run all examples
  example <name>         Run specific example (e.g., example_1_basic_usage)

UTILITIES:
  help                   Show this help message

EXAMPLES:
  bash manage.sh install
  bash manage.sh install-dev
  bash manage.sh test
  bash manage.sh test tests/test_utils.py
  bash manage.sh lint
  bash manage.sh format
  bash manage.sh examples
  bash manage.sh example example_1_basic_usage

EOF
}

# Install dependencies
install_base() {
    log_info "Installing base dependencies..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        log_success "Base dependencies installed"
    else
        log_error "Failed to install base dependencies"
        exit 1
    fi
}

install_dev() {
    install_base
    log_info "Installing development dependencies..."
    pip install pytest pytest-cov black flake8 mypy isort
    if [ $? -eq 0 ]; then
        log_success "Development dependencies installed"
    else
        log_error "Failed to install development dependencies"
        exit 1
    fi
}

dev_install() {
    log_info "Installing package in development mode..."
    pip install -e ".[dev]"
    if [ $? -eq 0 ]; then
        log_success "Package installed in development mode"
    else
        log_error "Failed to install package in development mode"
        exit 1
    fi
}

setup_dev() {
    log_info "Starting full development setup..."
    log_info "Step 1: Cleaning cache..."
    do_clean
    log_info "Step 2: Installing development dependencies..."
    install_dev
    log_info "Step 3: Running linter..."
    do_lint
    log_info "Step 4: Formatting code..."
    do_format
    log_info "Step 5: Running tests..."
    do_test
    log_success "Full development setup completed"
}

# Testing
do_test() {
    local test_file="${1:-tests/}"
    log_info "Running tests with coverage from: $test_file..."
    python -m pytest "$test_file" -v --cov=src
    if [ $? -eq 0 ]; then
        log_success "Tests completed"
    else
        log_error "Some tests failed"
        exit 1
    fi
}

do_test_quick() {
    log_info "Running quick tests (no coverage)..."
    python -m pytest tests/ -v
    if [ $? -eq 0 ]; then
        log_success "Quick tests completed"
    else
        log_error "Some tests failed"
        exit 1
    fi
}

# Code quality
do_lint() {
    log_info "Running flake8 linter..."
    flake8 src/ tests/ examples/
    if [ $? -ne 0 ]; then
        log_error "Linting failed"
        exit 1
    fi
    
    log_info "Running mypy type checker..."
    mypy src/ --ignore-missing-imports
    if [ $? -eq 0 ]; then
        log_success "Linting completed"
    else
        log_error "Type checking failed"
        exit 1
    fi
}

do_format() {
    log_info "Running black code formatter..."
    black src/ tests/ examples/
    if [ $? -ne 0 ]; then
        log_error "Black formatting failed"
        exit 1
    fi
    
    log_info "Running isort import sorter..."
    isort src/ tests/ examples/
    if [ $? -eq 0 ]; then
        log_success "Code formatting completed"
    else
        log_error "Import sorting failed"
        exit 1
    fi
}

do_clean() {
    log_info "Cleaning Python cache files..."
    find . -type f -name '*.pyc' -delete
    find . -type d -name '__pycache__' -delete
    find . -type d -name '.pytest_cache' -delete
    find . -type d -name '.mypy_cache' -delete
    find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null
    rm -rf build/ dist/ .coverage htmlcov/
    log_success "Cleanup completed"
}

# Examples
run_all_examples() {
    log_info "Running all examples..."
    for example in examples/example_*.py; do
        if [ -f "$example" ]; then
            example_name=$(basename "$example")
            log_info "Running $example_name..."
            python "$example"
            if [ $? -ne 0 ]; then
                log_error "Example $example_name failed"
                exit 1
            fi
            echo ""
        fi
    done
    log_success "All examples completed"
}

run_single_example() {
    local example_name="$1"
    local example_file="examples/${example_name}.py"
    
    if [ ! -f "$example_file" ]; then
        example_file="examples/example_${example_name}.py"
    fi
    
    if [ ! -f "$example_file" ]; then
        log_error "Example file not found: $example_file"
        echo "Available examples:"
        ls -1 examples/example_*.py | sed 's/examples\/example_/  - /' | sed 's/\.py//'
        exit 1
    fi
    
    log_info "Running example: $(basename $example_file)..."
    python "$example_file"
    if [ $? -eq 0 ]; then
        log_success "Example completed"
    else
        log_error "Example failed"
        exit 1
    fi
}

# Main command dispatcher
main() {
    if [ -z "$1" ]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    shift
    
    case $COMMAND in
        install)
            install_base
            ;;
        install-dev)
            install_dev
            ;;
        dev-install)
            dev_install
            ;;
        setup-dev)
            setup_dev
            ;;
        test)
            if [ -z "$1" ]; then
                do_test
            else
                do_test "$1"
            fi
            ;;
        test-quick)
            do_test_quick
            ;;
        lint)
            do_lint
            ;;
        format)
            do_format
            ;;
        clean)
            do_clean
            ;;
        examples)
            run_all_examples
            ;;
        example)
            if [ -z "$1" ]; then
                log_error "Example name required"
                run_all_examples
                exit 1
            fi
            run_single_example "$1"
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
