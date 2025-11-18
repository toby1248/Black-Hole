#!/bin/bash
#
# TDE-SPH Test Runner Script
#
# Provides convenient shortcuts for running different test suites.
#
# Usage:
#   ./run_tests.sh                  # Run all tests
#   ./run_tests.sh fast             # Run only fast tests
#   ./run_tests.sh regression       # Run only regression tests
#   ./run_tests.sh module <name>    # Run tests for specific module
#   ./run_tests.sh coverage         # Run with coverage report
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}TDE-SPH Test Suite${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

# Print success
print_success() {
    echo ""
    echo -e "${GREEN}✓ Tests passed!${NC}"
    echo ""
}

# Print failure
print_failure() {
    echo ""
    echo -e "${RED}✗ Tests failed!${NC}"
    echo ""
}

# Ensure we're in the right directory
if [ ! -d "tests" ]; then
    echo -e "${RED}Error: tests/ directory not found. Run from project root.${NC}"
    exit 1
fi

# Parse command
COMMAND=${1:-all}

print_header

case "$COMMAND" in
    all)
        echo "Running all tests..."
        pytest tests/ -v || { print_failure; exit 1; }
        ;;

    fast)
        echo "Running fast tests only..."
        pytest tests/ -v -m "fast" || { print_failure; exit 1; }
        ;;

    regression)
        echo "Running regression tests..."
        pytest tests/test_regression.py -v -m regression || { print_failure; exit 1; }
        ;;

    regression-fast)
        echo "Running fast regression tests..."
        pytest tests/test_regression.py -v -m "regression and fast" || { print_failure; exit 1; }
        ;;

    module)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please specify module name${NC}"
            echo "Usage: ./run_tests.sh module <name>"
            echo "Example: ./run_tests.sh module energy_conservation"
            exit 1
        fi
        MODULE="$2"
        echo "Running tests for module: $MODULE"
        pytest tests/test_${MODULE}.py -v || { print_failure; exit 1; }
        ;;

    coverage)
        echo "Running tests with coverage report..."
        pytest tests/ -v --cov=src/tde_sph --cov-report=term-missing --cov-report=html || { print_failure; exit 1; }
        echo ""
        echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
        ;;

    unit)
        echo "Running unit tests (excluding regression)..."
        pytest tests/ -v -m "not regression" || { print_failure; exit 1; }
        ;;

    export)
        echo "Running export tool tests..."
        pytest tests/test_export_to_blender.py -v || { print_failure; exit 1; }
        ;;

    help)
        echo "TDE-SPH Test Runner"
        echo ""
        echo "Usage: ./run_tests.sh [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  all                 Run all tests (default)"
        echo "  fast                Run only fast tests (< 5 seconds)"
        echo "  regression          Run all regression tests"
        echo "  regression-fast     Run fast regression tests only"
        echo "  module <name>       Run tests for specific module"
        echo "  coverage            Run with HTML coverage report"
        echo "  unit                Run unit tests (exclude regression)"
        echo "  export              Run export tool tests"
        echo "  help                Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh"
        echo "  ./run_tests.sh fast"
        echo "  ./run_tests.sh module energy_conservation"
        echo "  ./run_tests.sh coverage"
        echo ""
        exit 0
        ;;

    *)
        echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

print_success
