#!/bin/bash
# Test runner script for VisLang-UltraLow-Resource

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${1}${2}${NC}"
}

# Function to print section headers
print_header() {
    echo
    print_color $BLUE "=================================================="
    print_color $BLUE "$1"
    print_color $BLUE "=================================================="
    echo
}

# Default values
TEST_TYPE="all"
COVERAGE=true
PARALLEL=false
VERBOSE=false
CLEAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE      Type of tests to run (unit|integration|e2e|all)"
            echo "  --no-coverage        Skip coverage reporting"
            echo "  -p, --parallel       Run tests in parallel"
            echo "  -v, --verbose        Verbose output"
            echo "  -c, --clean          Clean test artifacts before running"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                   # Run all tests with coverage"
            echo "  $0 -t unit           # Run only unit tests"
            echo "  $0 -p --no-coverage  # Run tests in parallel without coverage"
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Clean previous test artifacts if requested
if [ "$CLEAN" = true ]; then
    print_header "Cleaning test artifacts"
    rm -rf htmlcov/
    rm -rf .coverage
    rm -rf .pytest_cache/
    rm -rf test-results/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    print_color $GREEN "✓ Cleaned test artifacts"
fi

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]] && [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    print_color $YELLOW "Warning: No virtual environment detected"
    print_color $YELLOW "Consider activating a virtual environment before running tests"
fi

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add coverage options
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml"
fi

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Set test markers based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit/"
        print_header "Running Unit Tests"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration/ -m integration"
        print_header "Running Integration Tests"
        ;;
    e2e)
        PYTEST_CMD="$PYTEST_CMD tests/e2e/ -m e2e"
        print_header "Running End-to-End Tests"
        ;;
    all)
        print_header "Running All Tests"
        ;;
    *)
        print_color $RED "Invalid test type: $TEST_TYPE"
        print_color $RED "Valid types: unit, integration, e2e, all"
        exit 1
        ;;
esac

# Create test results directory
mkdir -p test-results

# Run pre-test checks
print_header "Pre-test Environment Check"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
print_color $BLUE "Python version: $PYTHON_VERSION"

# Check if required packages are installed
print_color $BLUE "Checking required packages..."
python -c "import pytest; print(f'pytest: {pytest.__version__}')" || {
    print_color $RED "pytest not found. Install with: pip install pytest"
    exit 1
}

if [ "$COVERAGE" = true ]; then
    python -c "import pytest_cov; print('pytest-cov: available')" || {
        print_color $RED "pytest-cov not found. Install with: pip install pytest-cov"
        exit 1
    }
fi

if [ "$PARALLEL" = true ]; then
    python -c "import pytest_xdist; print('pytest-xdist: available')" || {
        print_color $RED "pytest-xdist not found. Install with: pip install pytest-xdist"
        exit 1
    }
fi

print_color $GREEN "✓ Environment check passed"

# Run linting before tests (optional)
if command -v flake8 &> /dev/null; then
    print_header "Code Quality Check"
    print_color $BLUE "Running flake8..."
    flake8 src/ tests/ || {
        print_color $YELLOW "Warning: Linting issues found"
    }
    print_color $GREEN "✓ Code quality check completed"
fi

# Create sample test data if needed
print_header "Preparing Test Data"
python -c "
import sys
sys.path.append('tests/fixtures')
from sample_data import save_sample_data_to_files, create_sample_images
from pathlib import Path

fixtures_dir = Path('tests/fixtures')
data_dir = fixtures_dir / 'data'
images_dir = fixtures_dir / 'images'

save_sample_data_to_files(data_dir)
create_sample_images(images_dir)
print('✓ Test data prepared')
" || {
    print_color $YELLOW "Warning: Could not create sample test data"
}

# Run the tests
print_header "Executing Tests"
print_color $BLUE "Command: $PYTEST_CMD"
echo

# Execute pytest with timing
start_time=$(date +%s)
$PYTEST_CMD
test_exit_code=$?
end_time=$(date +%s)

# Calculate duration
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

# Print results
echo
print_header "Test Results Summary"

if [ $test_exit_code -eq 0 ]; then
    print_color $GREEN "✓ All tests passed!"
else
    print_color $RED "✗ Some tests failed"
fi

print_color $BLUE "Execution time: ${minutes}m ${seconds}s"

# Coverage report
if [ "$COVERAGE" = true ] && [ $test_exit_code -eq 0 ]; then
    echo
    print_color $BLUE "Coverage report generated:"
    print_color $BLUE "  HTML: htmlcov/index.html"
    print_color $BLUE "  XML: coverage.xml"
    
    # Show coverage summary
    if command -v coverage &> /dev/null; then
        echo
        coverage report --show-missing
    fi
fi

# Test artifacts information
echo
print_color $BLUE "Test artifacts:"
print_color $BLUE "  Test results: test-results/"
print_color $BLUE "  Pytest cache: .pytest_cache/"

# Exit with the same code as pytest
exit $test_exit_code