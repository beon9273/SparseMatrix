---
name: update-benchmarks
description: Rebuild and rerun the sparsemat benchmark suite (CPU and, if a GPU is available, GPU) and refresh the corresponding table(s) in README.md. Use when asked to update/refresh/regenerate the benchmarks, or before a release when benchmark numbers may be stale.
---

# Update benchmarks

Regenerates the benchmark results table(s) in `README.md` from a fresh run
of the benchmark executable(s):
- CPU table (`<!-- BENCHMARK_TABLE:START/END -->`, under `## Benchmarks`) —
  always available.
- GPU table (`<!-- BENCHMARK_TABLE_GPU:START/END -->`, under `### GPU
  benchmarks`) — only produces real numbers on a machine with an actual
  CUDA-capable GPU; see the GPU section below.

## Steps

1. Make sure the build is configured:
   ```bash
   cmake -B build
   ```
2. Run the `update-benchmarks` target. This builds `benchmark` (and its
   `dist` dependency) if needed, runs it, and splices its stdout table into
   `README.md` via `scripts/update_benchmark_table.py`:
   ```bash
   cmake --build build --target update-benchmarks
   ```
3. If this machine has a CUDA-capable GPU, also refresh the GPU table (see
   the GPU section below for what happens if it doesn't).
4. Show a diff of `README.md` so the change is reviewable:
   ```bash
   git diff README.md
   ```
5. Read the diff. Only the table(s) between the markers should have changed.
   If any numbers moved enough to invalidate a claim in the hand-written
   commentary directly below the CPU table (the **vs Eigen sparse**, **vs
   Eigen dense**, **Sparsity vs fill level**, and **Solving linear systems**
   paragraphs — e.g. a stated ratio like "~30–60×" or "~35–40×" no longer
   matches), update that prose to match. Don't touch it otherwise — it is
   not auto-generated and small run-to-run noise is expected and fine to
   ignore. (The GPU table has no such commentary yet — see below.)
6. Do not commit automatically; leave the change staged/unstaged for the
   user to review, unless they've explicitly asked for a commit.

## GPU benchmarks

`examples/bench_gpu.cu` needs `SPARSEMAT_ENABLE_CUDA=ON` at configure time
*and* an actual CUDA-capable GPU at run time — building only proves the
kernels compile for the device, not that they were exercised.

```bash
cmake -B build -DSPARSEMAT_ENABLE_CUDA=ON
cmake --build build --target update-gpu-benchmarks
```

This is safe to run on any machine that can build it, GPU or not:
`scripts/update_benchmark_table.py` detects when `benchmark_gpu` printed no
results table (its own "No CUDA device available" message) and leaves
`README.md` untouched in that case, exiting 0 rather than failing. So:

- **No GPU on this machine**: run it if you like — it's a documented no-op,
  not an error. Don't treat a "leaving README.md untouched" note as a
  problem to fix.
- **GPU present and the table actually updates**: the GPU table currently
  has no hand-written analysis paragraphs below it (unlike the CPU table) —
  don't invent ratio commentary to match the CPU table's style unless the
  user asks for it; just report what changed.

## Notes

- Benchmark numbers are hardware- and load-dependent and will vary run to
  run — don't be alarmed by noise of a few percent, or even 2x on the
  smallest (sub-5ns, or sub-100ns for GPU) timings. Only the raw table(s)
  are regenerated; nothing else in the Benchmarks sections is touched
  automatically.
- If `cmake --build build --target update-benchmarks` fails because the
  target doesn't exist, Eigen3 wasn't found at configure time (the
  `benchmark` target — and `update-benchmarks` alongside it — is only
  defined when `find_package(Eigen3)` succeeds; see
  `examples/CMakeLists.txt`). Install Eigen3 and re-run `cmake -B build`.
- Similarly, if `update-gpu-benchmarks` doesn't exist, either
  `SPARSEMAT_ENABLE_CUDA` wasn't set, no CUDA compiler (`nvcc`) was found, or
  Eigen3 wasn't found — `benchmark_gpu` (and `update-gpu-benchmarks`
  alongside it) needs all three (see `examples/CMakeLists.txt`).
- The underlying script (`scripts/update_benchmark_table.py`) can also be
  run directly with `--binary`/`--readme`/`--start-marker`/`--end-marker`
  overrides; see `scripts/README.md`.
