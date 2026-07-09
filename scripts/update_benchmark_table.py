#!/usr/bin/env python3
"""Regenerate a benchmark table in README.md from a run of a benchmark binary.

Runs the given benchmark binary, extracts the plain-text results table from
its stdout, and splices it verbatim into README.md between a pair of marker
comments, replacing whatever fenced code block is there now. Everything else
around the table (prose, analysis) is left untouched — only the table itself
is regenerated.

Used for both benchmark binaries in this repo:
    # CPU benchmark (default markers/binary):
    python3 scripts/update_benchmark_table.py

    # GPU benchmark:
    python3 scripts/update_benchmark_table.py \\
        --binary build/examples/benchmark_gpu \\
        --start-marker "<!-- BENCHMARK_TABLE_GPU:START -->" \\
        --end-marker "<!-- BENCHMARK_TABLE_GPU:END -->"

If the binary runs but reports no table (e.g. benchmark_gpu on a machine with
no CUDA device — see its own "No CUDA device available" message), this is
treated as a soft no-op: it prints a note and exits 0 without touching
README.md, rather than failing. That keeps this script (and the CMake targets
that call it) safe to run on machines that can't produce results yet.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_START_MARKER = "<!-- BENCHMARK_TABLE:START -->"
DEFAULT_END_MARKER = "<!-- BENCHMARK_TABLE:END -->"

SEPARATOR_RE = re.compile(r"^-{10,}$")


class NoTableProduced(Exception):
    """Raised when the binary ran successfully but printed no results table
    (e.g. no GPU present). Distinct from a real failure: callers should treat
    this as a no-op, not an error."""


def extract_table(benchmark_stdout: str) -> str:
    """Pulls the fixed-width results table out of a benchmark's stdout.

    Both bench.cpp's and bench_gpu.cu's print_results() bracket the table
    with a leading and trailing line of dashes; everything between (and
    including) the first and last such line is the table.
    """
    lines = benchmark_stdout.splitlines()
    sep_indices = [i for i, line in enumerate(lines) if SEPARATOR_RE.match(line)]
    if len(sep_indices) < 2:
        raise NoTableProduced(
            "No results table (no dashed separator lines) in the benchmark's output — "
            "it likely reported no device/hardware available rather than actually failing. "
            "Full output was:\n" + benchmark_stdout
        )
    first, last = sep_indices[0], sep_indices[-1]
    return "\n".join(lines[first : last + 1])


def splice_readme(readme_text: str, table: str, start_marker: str, end_marker: str) -> str:
    start = readme_text.find(start_marker)
    end = readme_text.find(end_marker)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find both {start_marker!r} and {end_marker!r} markers in README.md")
    start_of_block = start + len(start_marker)
    replacement = f"\n```\n{table}\n```\n"
    return readme_text[:start_of_block] + replacement + readme_text[end:]


def run_benchmark(binary: Path) -> str:
    if not binary.exists():
        raise FileNotFoundError(f"Benchmark binary not found at {binary}. Build it first.")
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO_ROOT / "build" / "examples" / "benchmark",
        help="Path to the built benchmark executable.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=REPO_ROOT / "README.md",
        help="Path to the README to update.",
    )
    parser.add_argument(
        "--start-marker",
        default=DEFAULT_START_MARKER,
        help="Marker comment immediately before the fenced table to replace.",
    )
    parser.add_argument(
        "--end-marker",
        default=DEFAULT_END_MARKER,
        help="Marker comment immediately after the fenced table to replace.",
    )
    args = parser.parse_args()

    try:
        stdout = run_benchmark(args.binary)
        table = extract_table(stdout)
    except NoTableProduced as exc:
        print(f"note: {exc}\nLeaving README.md untouched.", file=sys.stderr)
        return 0
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        readme_text = args.readme.read_text()
        updated = splice_readme(readme_text, table, args.start_marker, args.end_marker)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if updated == readme_text:
        print("Benchmark table unchanged.")
    else:
        args.readme.write_text(updated)
        print(f"Updated benchmark table in {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
