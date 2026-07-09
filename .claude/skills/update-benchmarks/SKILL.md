---
name: update-benchmarks
description: Rebuild and rerun the sparsemat benchmark suite and refresh the benchmark table in README.md. Use when asked to update/refresh/regenerate the benchmarks, or before a release when benchmark numbers may be stale.
---

# Update benchmarks

Regenerates the benchmark results table in `README.md` (between the
`<!-- BENCHMARK_TABLE:START -->` / `<!-- BENCHMARK_TABLE:END -->` markers,
under the `## Benchmarks` heading) from a fresh run of the `benchmark`
executable.

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
3. Show a diff of `README.md` so the change is reviewable:
   ```bash
   git diff README.md
   ```
4. Read the diff. Only the table between the markers should have changed.
   If any numbers moved enough to invalidate a claim in the hand-written
   commentary directly below the table (the **vs Eigen sparse**, **vs Eigen
   dense**, **Sparsity vs fill level**, and **Solving linear systems**
   paragraphs — e.g. a stated ratio like "~30–60×" or "~35–40×" no longer
   matches), update that prose to match. Don't touch it otherwise — it is
   not auto-generated and small run-to-run noise is expected and fine to
   ignore.
5. Do not commit automatically; leave the change staged/unstaged for the
   user to review, unless they've explicitly asked for a commit.

## Notes

- Benchmark numbers are hardware- and load-dependent and will vary run to
  run — don't be alarmed by noise of a few percent, or even 2x on the
  smallest (sub-5ns) timings. Only the raw table is regenerated; nothing
  else in the Benchmarks section is touched automatically.
- If `cmake --build build --target update-benchmarks` fails because the
  target doesn't exist, Eigen3 wasn't found at configure time (the
  `benchmark` target — and `update-benchmarks` alongside it — is only
  defined when `find_package(Eigen3)` succeeds; see
  `examples/CMakeLists.txt`). Install Eigen3 and re-run `cmake -B build`.
- The underlying script (`scripts/update_benchmark_table.py`) can also be
  run directly with `--binary`/`--readme` overrides; see
  `scripts/README.md`.
