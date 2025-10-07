#!/usr/bin/env python3
"""Trace-based coverage helper for the CLI entry points.

This script provides a zero-dependency alternative to ``coverage.py`` by leveraging the
standard library :mod:`trace` module. It executes ``pytest`` while recording which lines of
``image_cli.py`` and ``sora_cli.py`` run, then reports per-file and overall coverage. Invoke it
directly or through ``uv`` and pass a ``--fail-under`` value to enforce a minimum percentage. Any
additional arguments after ``--`` are forwarded to ``pytest`` untouched.
"""

from __future__ import annotations

import argparse
import dis
import sys
import trace
from pathlib import Path
from typing import Dict, Iterable, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    REPO_ROOT / "image_cli.py",
    REPO_ROOT / "sora_cli.py",
]


def statement_lines(path: Path) -> Set[int]:
    """Return the set of executable line numbers for *path*."""
    source = path.read_text(encoding="utf-8")
    code = compile(source, str(path), "exec")
    lines: Set[int] = set()

    def walk(code_obj):
        for _, lineno in dis.findlinestarts(code_obj):
            lines.add(int(lineno))
        for const in code_obj.co_consts:
            if isinstance(const, type(code_obj)):
                walk(const)

    walk(code)
    lines.discard(0)
    return lines


def collect_executed_lines(result: trace.CoverageResults, targets: Iterable[Path]) -> Dict[Path, Set[int]]:
    resolved = {str(path.resolve()): path for path in targets if path.exists()}
    executed: Dict[Path, Set[int]] = {path: set() for path in resolved.values()}

    for (filename, lineno), count in result.counts.items():
        if count <= 0:
            continue
        try:
            key = str(Path(filename).resolve())
        except FileNotFoundError:
            continue
        target = resolved.get(key)
        if target is not None:
            executed[target].add(int(lineno))

    return executed


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pytest with lightweight coverage tracking.")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Fail if overall coverage percentage is below this value.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to pytest (prefix with --).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    pytest_args = args.pytest_args or []
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    existing_sources = [path for path in SOURCE_FILES if path.exists()]
    if not existing_sources:
        print("No source files found to measure coverage.", file=sys.stderr)
        return 1

    tracer = trace.Trace(count=True, trace=False, ignoredirs=[sys.prefix, sys.exec_prefix])

    import pytest  # Local import to avoid dependency if not running tests.

    exit_code = tracer.runfunc(pytest.main, pytest_args)
    if exit_code:
        # Propagate pytest's exit code without attempting coverage reporting.
        return int(exit_code)

    results = tracer.results()
    executed = collect_executed_lines(results, existing_sources)

    per_file = []
    total_statements = 0
    total_covered = 0

    for path in existing_sources:
        statements = statement_lines(path)
        executed_lines = executed.get(path, set())
        covered = len(statements & executed_lines)
        total = len(statements)
        percent = 100.0 if total == 0 else (covered / total) * 100.0
        per_file.append((path, covered, total, percent))
        total_statements += total
        total_covered += covered

    per_file.sort(key=lambda item: str(item[0]))

    print("Coverage summary (trace-based):")
    for path, covered, total, percent in per_file:
        rel = path.relative_to(REPO_ROOT)
        print(f"  {rel}: {percent:.2f}% ({covered}/{total})")

    overall = 100.0 if total_statements == 0 else (total_covered / total_statements) * 100.0
    print(f"Overall coverage: {overall:.2f}% ({total_covered}/{total_statements})")

    if overall + 1e-9 < args.fail_under:
        print(
            f"Coverage {overall:.2f}% is below the required minimum of {args.fail_under:.2f}%.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation entry point
    raise SystemExit(main())
