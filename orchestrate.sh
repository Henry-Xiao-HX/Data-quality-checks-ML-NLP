#!/bin/bash

# Main orchestration script - replaces Makefile
# Usage: bash orchestrate.sh <command> [args]

if [ -z "$1" ]; then
    bash scripts/help.sh
    exit 0
fi

COMMAND=$1
shift

case $COMMAND in
    install)
        bash scripts/install.sh "$@"
        ;;
    install-dev)
        bash scripts/install-dev.sh "$@"
        ;;
    dev-install)
        bash scripts/dev-install.sh "$@"
        ;;
    test)
        bash scripts/test.sh "$@"
        ;;
    test-specific)
        bash scripts/test-specific.sh "$@"
        ;;
    quick-test)
        bash scripts/quick-test.sh "$@"
        ;;
    lint)
        bash scripts/lint.sh "$@"
        ;;
    format)
        bash scripts/format.sh "$@"
        ;;
    clean)
        bash scripts/clean.sh "$@"
        ;;
    run-examples)
        bash scripts/run-examples.sh "$@"
        ;;
    run-example)
        bash scripts/run-example.sh "$@"
        ;;
    setup-dev)
        bash scripts/setup-dev.sh "$@"
        ;;
    help)
        bash scripts/help.sh "$@"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        bash scripts/help.sh
        exit 1
        ;;
esac
