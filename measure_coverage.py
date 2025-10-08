#!/usr/bin/env python3
"""Run the tests with coverage focused on the CLI modules."""

from __future__ import annotations

import argparse
import importlib
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TESTS_DIR = REPO_ROOT / "tests"
TARGET_MODULES = ("image_cli", "sora_cli")
TARGET_FILES = tuple(f"{module}.py" for module in TARGET_MODULES)


def ensure_dependency(module_name: str) -> None:
    """Exit early with a helpful message if a dependency is missing."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Required dependency '{module_name}' is not installed. "
            f"Install it with 'pip install {module_name}'."
        ) from exc


def run_command(command: list[str]) -> None:
    """Run a subprocess command within the repository root."""
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure coverage for image_cli.py and sora_cli.py."
    )
    parser.add_argument(
        "--pytest-args",
        metavar="ARGS",
        help="Additional arguments to pass to pytest (e.g. '-k test_name').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dependency("coverage")
    ensure_dependency("pytest")

    run_command([sys.executable, "-m", "coverage", "erase"])

    coverage_run_cmd = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        f"--source={','.join(TARGET_MODULES)}",
        "-m",
        "pytest",
        str(TESTS_DIR),
    ]

    if args.pytest_args:
        coverage_run_cmd.extend(shlex.split(args.pytest_args))

    run_command(coverage_run_cmd)

    include_patterns = ",".join(TARGET_FILES)
    report_cmd = [
        sys.executable,
        "-m",
        "coverage",
        "report",
        f"--include={include_patterns}",
        "--show-missing",
    ]

    run_command(report_cmd)


if __name__ == "__main__":
    main()
