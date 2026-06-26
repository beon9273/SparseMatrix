#!/usr/bin/env python3
"""Read a CSV matrix and print the corresponding SparseMat<> type declaration."""

import csv
import sys


def main():
    
    if len(sys.argv) < 2:
        print("Usage: generate.py <matrix.csv>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    matrix = []

    with open(path, newline="") as f:
        for row in csv.reader(f):
            matrix.append([float(v) for v in row])

    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0

    nonzero_indices = [
        r * cols + c
        for r, row in enumerate(matrix)
        for c, val in enumerate(row)
        if val != 0.0
    ]

    values = [
        matrix[r][c]
        for r in range(rows)
        for c in range(cols)
        if matrix[r][c] != 0.0
    ]

    indices_str = ", ".join(str(i) for i in nonzero_indices)
    values_str  = ", ".join(str(v) for v in values)

    print(f"SparseMat<double, {rows}, {cols}, {indices_str}> m({values_str});")


if __name__ == "__main__":
    main()
