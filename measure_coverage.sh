#!/bin/bash
# Script to measure test coverage for image_cli.py and sora_cli.py

echo "======================================================================"
echo "Running test coverage analysis"
echo "======================================================================"
echo

# Run coverage
echo "Running tests with coverage..."
uv run python -m coverage run -m pytest tests/ -v

echo
echo "======================================================================"
echo "Coverage Report"
echo "======================================================================"
echo

# Generate coverage report for specific files
uv run python -m coverage report --include=image_cli.py,sora_cli.py -m

echo
echo "======================================================================"
echo "Generating HTML Report"
echo "======================================================================"
echo

# Generate HTML report
uv run python -m coverage html --include=image_cli.py,sora_cli.py

echo
echo "HTML report generated in: htmlcov/index.html"
echo
echo "To view detailed coverage:"
echo "  open htmlcov/index.html"
