# Changelog

All notable changes to sparsemat are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because every sparsity pattern is a distinct type, the *result type* of an
operation is part of this library's public API: a change to how a result's
sparsity is derived can change a caller's `decltype` even when the call itself
still compiles. Such changes are called out explicitly below.

## [0.1.0] — 2026-08-13

First release.

### Added

- **Fused products**, each avoiding a materialized intermediate — the transpose
  or the triple product is never formed:
  `ata` (AᵀA), `aat` (AAᵀ), `atb` (AᵀB), `abt` (ABᵀ), `atba` (AᵀBA),
  `abat` (ABAᵀ), and the addition-folding `abat_add` (ABAᵀ + C) and
  `atba_add` (AᵀBA + C). The Kalman-shaped chain `F P Fᵀ + Q` is now one call.
- **Blocks and concatenation**: `submatrix<Row0, Col0, NRows, NCols>`, `row<I>`,
  `col<J>`, `hcat`, `vcat`.
- **Permutations and fill-reducing ordering**: `permute<RowPerm, ColPerm>`,
  `symmetric_permute<Perm>` (P A Pᵀ), `permute_rows<Perm>`, `permute_cols<Perm>`,
  `inverse_permutation<Perm>`, `identity_permutation`, `is_permutation`, plus
  `bandwidth<T>()` and `rcm_ordering<T>()` — reverse Cuthill–McKee evaluated at
  compile time over a matrix type's pattern, so the ordering is baked into the
  permuted type rather than computed at runtime.
- **Rank-1 updates**: `outer`, `rank1_update` (A + αxyᵀ, accumulated in one pass
  without forming the outer product), `symmetric_rank1_update`.
- **Norms and diagnostics**: `max_abs`, `norm_1`, `norm_inf`, and
  `condition_number` (an exact 1-norm condition number; it forms the inverse
  explicitly, so it is a validation-time diagnostic, not a hot-path call).
- `SPARSEMAT_MAX_NONZEROS` — turns runaway result density into a named error at
  the instantiation responsible, instead of a multi-minute build or an
  inscrutable non-type-template-argument diagnostic. Defaults to 4096; define to
  `0` to disable.
- Version macros in `sparsemat/version.h` (`SPARSEMAT_VERSION`,
  `SPARSEMAT_VERSION_CHECK`), usable from the preprocessor for feature guards.
- CUDA support: every operation is marked `SPARSEMAT_HD` and runs in device code.
  `sparsemat::sparsemat` propagates `--expt-relaxed-constexpr` to consumers
  compiling as CUDA.
- Installable CMake package (`find_package(sparsemat)` →
  `sparsemat::sparsemat`), a single-header amalgamation at `dist/sparsemat.h`,
  and Doxygen output via the `docs` target.

### Changed

- `least_squares_solve` now builds the normal equations with `ata`/`aat` rather
  than `transpose` + `multiply`, so the transpose is no longer materialized.
  Results are numerically identical; the intermediate types are not.
- Matrix operations unroll via `std::index_sequence` and fold expressions
  instead of linear template recursion, which previously hit the compiler's
  ~900-deep instantiation budget and capped matrices near 10×10. Instantiation
  depth is now O(1) in matrix size, and the practical ceiling is the compiler's
  *constexpr evaluation* budget (tighter under nvcc than g++/clang).
- `dense`, `trace`, `set_diagonal`, `is_full_symmetric` and `print` use memoized
  lookup tables and runtime loops: they only copy values, so eliminating a
  zero term buys nothing and the fold cost was pure overhead. `dense()`
  previously exhausted the instantiation budget at 32×32.
- `validate_indices` is linear rather than O(nonZeroCount²); the quadratic form
  ran for every instantiation, including dense results, and exhausted nvcc's
  constexpr budget at 32×32.
- Mixed-scalar operations are now rejected by `static_assert` instead of
  silently promoting. Convert explicitly — `a.template convert<double>()`.
- `inverse_permutation` takes its permutation as a template parameter
  (`inverse_permutation<perm>()`) rather than a function argument, so a
  malformed ordering is a compile error. Inverting scatters through `perm[i]` as
  an index, so an out-of-range entry previously wrote past the end of the
  result.

### Fixed

- `printDense()` never compiled: it indexed the `SparseMat` returned by `dense()`
  with `operator[]`, which does not exist. As a member of a class template
  nothing instantiated it, but it was documented.
- `dense()` was documented as returning `std::array` but returns a densified
  `SparseMat`. Documentation corrected, and `to_array()` added for the
  plain-buffer case.
- `operations/utils.h` used `assert()` without including `<cassert>`, and four
  headers used `MatrixUtilities` without including it. Every header is now
  self-contained.

[0.1.0]: https://github.com/beon9273/SparseMatrix/releases/tag/v0.1.0
