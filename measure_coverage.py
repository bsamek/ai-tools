#!/usr/bin/env python3
"""
Script to measure test coverage for image_cli.py and sora_cli.py
"""

import subprocess
import sys
import os

def main():
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 70)
    print("Running test coverage analysis")
    print("=" * 70)
    print()

    # Check if coverage is installed
    try:
        import coverage
    except ImportError:
        print("Error: coverage package not found")
        print("Install it with: pip install coverage")
        sys.exit(1)

    # Run coverage
    print("Running tests with coverage...")
    result = subprocess.run([
        "uv", "run", "python", "-m", "coverage", "run",
        "-m", "pytest",
        "tests/",
        "-v"
    ])

    if result.returncode != 0:
        print("\nWarning: Some tests failed, but continuing with coverage report...")

    print("\n" + "=" * 70)
    print("Coverage Report")
    print("=" * 70)
    print()

    # Generate coverage report for specific files
    subprocess.run([
        "uv", "run", "python", "-m", "coverage", "report",
        "--include=image_cli.py,sora_cli.py",
        "-m"
    ])

    print("\n" + "=" * 70)
    print("Detailed Coverage by File")
    print("=" * 70)
    print()

    # Generate HTML report
    print("Generating HTML coverage report...")
    subprocess.run([
        "uv", "run", "python", "-m", "coverage", "html",
        "--include=image_cli.py,sora_cli.py"
    ])

    print("\nHTML report generated in: htmlcov/index.html")
    print("\nTo view detailed coverage:")
    print("  open htmlcov/index.html")

if __name__ == "__main__":
    main()
