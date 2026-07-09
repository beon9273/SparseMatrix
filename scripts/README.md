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

Runs a benchmark executable and splices its results table into `README.md`, replacing whatever is between a pair of marker comments. Used for both benchmark binaries in this repo; normally invoked via a CMake target rather than directly:

```bash
cmake --build build --target update-benchmarks       # CPU table
cmake --build build --target update-gpu-benchmarks   # GPU table (requires -DSPARSEMAT_ENABLE_CUDA=ON)
```

which build the relevant binary first if needed, then run e.g.:

```bash
python3 update_benchmark_table.py \
  --binary <path-to-benchmark> \
  --readme <path-to-README.md> \
  --start-marker "<!-- BENCHMARK_TABLE:START -->" \
  --end-marker "<!-- BENCHMARK_TABLE:END -->"
```

`--binary` and `--readme` default to `build/examples/benchmark` and `README.md` at the repo root; `--start-marker`/`--end-marker` default to the CPU table's markers. So `python3 scripts/update_benchmark_table.py` from the repo root works after a normal build, and the GPU table needs its binary/markers passed explicitly (which is what the `update-gpu-benchmarks` CMake target does — see `examples/CMakeLists.txt`).

Only the table itself is regenerated — the surrounding prose/analysis is written by hand and is not touched.

If the binary runs but prints no results table (`benchmark_gpu` on a machine with no CUDA device reports this and exits), the script treats it as a no-op: it prints a note and exits 0 without touching `README.md`, rather than failing.
