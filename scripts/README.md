# scripts

## generate.py

Reads a CSV matrix and prints the corresponding `SparseMat<>` declaration.

```bash
python3 generate.py <matrix.csv>
```

Each row in the CSV is a matrix row. Zero values are treated as structural zeros and excluded from the index pack. Non-zero values appear as constructor arguments in row-major order.

**Example** — `sample.csv` (6×8, ~30% density):

```bash
python3 generate.py sample.csv
# SparseMat<double,int, 6, 8, 1, 2, 5, 6, 8, 11, 13, 16, 18, 32, 33, 34, 35, 38, 42, 43> m(3.2, ...);
```

## update_benchmark_table.py

Runs the `benchmark` executable and splices its results table into `README.md`, replacing whatever is between the `<!-- BENCHMARK_TABLE:START -->` / `<!-- BENCHMARK_TABLE:END -->` markers in the Benchmarks section. Normally invoked via the CMake target rather than directly:

```bash
cmake --build build --target update-benchmarks
```

which builds `benchmark` first if needed, then runs:

```bash
python3 update_benchmark_table.py --binary <path-to-benchmark> --readme <path-to-README.md>
```

Both `--binary` and `--readme` default to `build/examples/benchmark` and `README.md` at the repo root, so `python3 scripts/update_benchmark_table.py` from the repo root works after a normal build. Only the table itself is regenerated — the surrounding prose/analysis in the Benchmarks section is written by hand and is not touched.
