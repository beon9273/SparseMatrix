#!/usr/bin/env python3
"""Regenerate the benchmark table in README.md from a run of `benchmark`.

Runs the benchmark binary (building it first if needed), extracts the
plain-text results table from its stdout, and splices it verbatim into
README.md between the `<!-- BENCHMARK_TABLE:START -->` /
`<!-- BENCHMARK_TABLE:END -->` markers, replacing whatever fenced code block
is there now. Everything else in the Benchmarks section (prose, analysis) is
left untouched — only the table itself is regenerated.

Usage:
    python3 scripts/update_benchmark_table.py [--binary PATH] [--readme PATH]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

START_MARKER = "<!-- BENCHMARK_TABLE:START -->"
END_MARKER = "<!-- BENCHMARK_TABLE:END -->"

SEPARATOR_RE = re.compile(r"^-{10,}$")


def extract_table(benchmark_stdout: str) -> str:
    """Pulls the fixed-width results table out of the benchmark's stdout.

    print_results() (examples/bench.cpp) always brackets the table with a
    leading and trailing line of dashes; everything between (and including)
    the first and last such line is the table.
    """
    lines = benchmark_stdout.splitlines()
    sep_indices = [i for i, line in enumerate(lines) if SEPARATOR_RE.match(line)]
    if len(sep_indices) < 2:
        raise ValueError(
            "Could not find the benchmark table (no dashed separator lines) "
            "in the benchmark's output."
        )
    first, last = sep_indices[0], sep_indices[-1]
    return "\n".join(lines[first : last + 1])


def splice_readme(readme_text: str, table: str) -> str:
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"Could not find both {START_MARKER!r} and {END_MARKER!r} markers in README.md"
        )
    start_of_block = start + len(START_MARKER)
    replacement = f"\n```\n{table}\n```\n"
    return readme_text[:start_of_block] + replacement + readme_text[end:]


def run_benchmark(binary: Path) -> str:
    if not binary.exists():
        raise FileNotFoundError(
            f"Benchmark binary not found at {binary}. Build it first with:\n"
            "  cmake --build build --target benchmark"
        )
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    try:
        stdout = run_benchmark(args.binary)
        table = extract_table(stdout)
        readme_text = args.readme.read_text()
        updated = splice_readme(readme_text, table)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
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
