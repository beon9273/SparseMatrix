# sparsemat

A header-only C++20 sparse matrix library that encodes sparsity patterns as template parameters, letting the compiler unroll matrix operations and skip noops at compile time.

> **Preface** — This library was built for fun, to test the limits of C++ compile-time constructs like template parameter packs, index sequences and `constexpr`. The underlying question was whether compile-time coding could meaningfully speed up basic linear algebra operations versus leaving it to blas.
>
> AI Disclosure: AI has been used heavily in the development of this library. A protype for this library was written, by hand, many years ago. AI assited development was used to turn that prototype into a release worthy library. 

## Leveraging known sparsity information at compile time

The project was born while developing a GPU-enabled Kalman filter for particle track reconstruction in high energy physics. In that application, one must perform billions of
independent Kalman filters, all of which require many linear algebra operations using a small number of small (O(10x10)) sparse matrices.

Idea: if the non-zero structure of your matrices is fixed and known at compile time, the compiler can eliminate a whole bunch of operations.

Solution: user provides sparsity pattern as template parameters. Sparsity patterns of operations (e.g. `C = A × B`) are computed at compile time. `std::index_sequence`/fold expressions unroll operation loops at compile time. Zero operations (i.e, multiplications/additions where one of the terms is zero) are skipped
using compile-time constructs( e.g. constexpr if).  


Limitations: 
  - entering sparsity patterns is awkward and bug-prone.
  - every distinct sparsity pattern is a distinct type, which makes compile times long for large configurations and binary size could explode pretty easily. 
  - matrix operations are unrolled at compile time via `std::index_sequence`/fold expressions, which is O(1) in instantiation depth regardless of matrix size. What remains is the compile-time/binary-size cost above — that scales with how many non-zeros an operation actually touches (density), so a genuinely sparse 40x40+ matrix is practical, while a dense-ish one of the same size can still be slow to compile.
  - the practical size limit is the compiler's *constexpr evaluation* budget, and nvcc's is markedly tighter than g++'s or clang's. Sparsity computation is memoized into dense boolean grids to stay within it (see `MatrixUtilities::to_dense_bool`).
  - a certain level of sparsity is required.  Whether the performance gain is worth that cost depends heavily on how sparse your matrices are and how many operations you're doing.


```cpp
// SparseMat<DataType, IntType, Rows, Cols, NonZeroIndices...>
// Non-zero indices are flat row-major indices.

SparseMat<double, int, 3, 3, 0, 4, 8> A(1, 2, 3);  // diagonal matrix
SparseMat<double, int, 3, 3, 0, 1, 2> B(4, 5, 6);  // first row only

// Result type and sparsity computed at compile time
// C === SparseMat<double, int, 3, 3, 0, 3, 6>
auto C = A.mult(B);
```

### Compile-time budget

Compile time and binary size scale with how many non-zeros a result actually has, and an operation chain can quietly produce a much denser result than intended (`multiply` and `kronecker` both can). When that happens the symptom is a build that takes minutes, or a compiler that exhausts some internal budget and reports something inscrutable about a non-type template argument — neither of which points at the cause.

`SPARSEMAT_MAX_NONZEROS` turns that into a named error, at the instantiation responsible:

```cpp
#define SPARSEMAT_MAX_NONZEROS 512   // fail the build past 512 stored values
#include "sparsemat.h"
```

The default is 4096 — generous enough not to interfere with ordinary use (a fully dense 64×64 fits), low enough to catch runaway density. Define it to `0` to disable the check.

### Fill-reducing ordering, at compile time

The other way to spend less of that budget is to make the result genuinely sparser. `rcm_ordering<T>()` runs reverse Cuthill–McKee over a matrix type's pattern and returns a permutation that pulls its entries towards the diagonal; `symmetric_permute<perm>()` applies it as `P A Pᵀ`. Since the factors of a banded matrix stay inside the band, a smaller bandwidth bounds the fill-in `lu_factorize` and `cholesky_factorize` produce — and here fill-in *is* compile time and binary size.

Ordinary sparse libraries do this at runtime, on the matrix in hand. Here the pattern is a compile-time constant, so the whole ordering is derived at compile time and baked into the permuted type:

```cpp
constexpr auto perm = SparseLinearAlgebra::rcm_ordering<decltype(A)>();
auto reordered = A.symmetric_permute<perm>();
auto y = reordered.solve(SparseLinearAlgebra::permute_rows<perm>(b));
// y is in permuted coordinates: original index perm[i] holds y[i].
// SparseLinearAlgebra::inverse_permutation<perm>() maps back.
```

`bandwidth<T>()` reports what the reordering achieved. The reordering is only worth applying when it actually shrinks the bandwidth — an already-banded matrix (tridiagonal, say) is optimally ordered to begin with.

### Fused products

Several operations exist purely to skip an intermediate that would otherwise be materialized, sparsity computation and all. `ata`/`aat` form AᵀA and AAᵀ without the transpose (and compute only the upper triangle, mirroring the rest); `atb`/`abt` do the same for two distinct operands; `atba`/`abat` evaluate the triple products AᵀBA and ABAᵀ in one traversal; `abat_add`/`atba_add` fold an addition into that same pass; `rank1_update` adds `alpha·x yᵀ` without building the outer product. The Kalman-shaped chain `F P Fᵀ + Q` is one call, not three matrices.

## Requirements

- C++20
- CMake 3.16+

## Platform Support

This project is currently tested only on Linux. Other platforms and toolchains may work, but they are not currently part of the tested matrix.

CI covers GCC 13 and Clang 18 on x86-64, plus ASan/UBSan and a `nvcc` build. Note that the CI runners have no GPU: the CUDA job proves every `SPARSEMAT_HD` operation *compiles* for the device, and `sparsemat_tests_gpu` reports itself skipped rather than run. Device *execution* is verified manually on a real GPU (see [GPU (CUDA)](#gpu-cuda)), not on every commit.

Template parameter constraints for `SparseMat<DataType, IntType, Rows, Cols, NonZeroIndices...>`:
- `Rows > 0` and `Cols > 0`.
- Every `NonZeroIndex` must be non-negative.
- Every `NonZeroIndex` must be `< Rows * Cols`.
- `NonZeroIndices` must be unique (no duplicates).


## Building

```bash
cmake -B build
cmake --build build
```

## Linking

This is a header-only library. There are two ways to use it:

**Multi-header:** Add `include/` to your compiler search path and include the top-level header:
```cpp
#include "sparsemat/api/sparsemat.h"
```

**Single-header amalgamation:** Build the `dist` target to generate a self-contained header, then copy it into your project:
```bash
cmake --build build --target dist
# produces dist/sparsemat.h
```
```cpp
#include "sparsemat.h"
```


## Running tests

```bash
cd build && ctest --output-on-failure
```

## Generating docs

```bash
cmake --build build --target docs
# output: build/docs/html/index.html
```

## API

### Template parameters

```cpp
SparseMat<DataType, IntType, Rows, Cols, NonZeroIndices...>
```

### Constructors

```cpp
SparseMat<double, int, 2, 2, 0, 3> m;             // zero-initialized
SparseMat<double, int, 2, 2, 0, 3> m(1.0, 2.0);  // value per non-zero
```

Writing the `NonZeros` pack by hand means computing flat (`row*cols+col`) indices yourself and keeping the constructor's value list in that exact order — easy to get wrong as a matrix grows. `make_sparse_matrix()` builds the same type from a plain list of `(row, col, value)` entries instead, in any order:

```cpp
constexpr auto entries = std::array{
    SparseLinearAlgebra::SparseEntry{0, 0, 4.0},
    SparseLinearAlgebra::SparseEntry{1, 1, 2.0},
    SparseLinearAlgebra::SparseEntry{0, 1, 5.0},
};
auto A = SparseLinearAlgebra::make_sparse_matrix<double, int, 3, 3, entries>();
```

`entries` has to be a named `constexpr` variable (a C++20 non-type template argument, since it determines the resulting type) — see `include/sparsemat/api/sparsemat.h` for the full doc comment. Duplicate or out-of-bounds entries are caught by the same `static_assert` a hand-written declaration would hit.

### Pattern builders

Writing flat index packs by hand is the limitation this library leads with, and `make_sparse_matrix()` only helps when the values are known up front. These build the *shape* — every stored value starts at zero, to be filled in later or repeatedly:

```cpp
using namespace SparseLinearAlgebra;

auto t = tridiagonal<double, int, 5>();              // 5x5, 13 stored values
auto v = tridiagonal<double, int, 5>(-1.0, 4.0, -1.0);  // ... with constant bands
auto b = banded<double, int, 8, 8, 2, 1>();          // 2 sub-, 1 super-diagonal
auto d = banded<double, int, 4, 4, 0, 0>();          // diagonal

// Give only the lower triangle; mirror positions are added for you.
constexpr auto lower = std::array{
    SparsePosition{0, 0}, SparsePosition{1, 0}, SparsePosition{1, 1},
};
auto s = symmetric_from_lower<double, int, 2, lower>();

// Or an arbitrary pattern, in any order.
constexpr auto shape = std::array{SparsePosition{1, 1}, SparsePosition{0, 0}};
auto p = make_pattern<double, int, 2, 2, shape>();
```

Positions are sorted internally, so declaration order does not affect the resulting type, and exact duplicates collapse rather than tripping the duplicate-index `static_assert`.

`block_diagonal(a, b)` composes two matrices into `diag(a, b)`. Unlike `kronecker`, whose result pattern is the *product* of both operands', this one is the *sum* — the result has exactly `nnz(a) + nnz(b)` stored values — which makes it the cheap way to fold several independent sub-problems into one solve.

### Element access

```cpp
m.get<I, J>()       // compile-time indices; returns 0 for zero elements
m.get(i, j)         // runtime indices
m.set<I, J>(value)  // compile-time; static_assert if (I,J) is a zero index
m.set(i, j, value)  // runtime; returns false if (I,J) is a zero index
m.fill(value)       // set all non-zero storage to value
m(i, j)             // runtime read; alias for get(i, j)
```

`m(i, j)` is read-only on purpose. A structurally zero position has no storage to return a reference to, so a reference-returning overload would have to hand back a shared dummy and let `m(0, 1) = 5.0` silently do nothing. Use `set(i, j, value)`, which returns `false` when the write does not land.

Stored values are iterable directly, and `entries()` pairs each with its position:

```cpp
for (double value : m) { ... }                     // stored (non-zero) values, in index order
for (auto [row, col, value] : m.entries()) { ... }  // ... with their positions
m.size();                                           // number of stored values
```

### Fusing a chain of operations

Every operator materializes its result, so `2*A + 3*B - C` builds three intermediate matrices to produce one answer. `fuse()` applies a function across several matrices element-wise in a single pass, materializing only the result:

```cpp
// Three temporaries: A*2, B*3, and the sum.
auto eager = a.scale(2.0).add(b.scale(3.0)).subtract(c);

// None.
auto fused = SparseLinearAlgebra::fuse(
    [](double x, double y, double z) { return (2 * x) + (3 * y) - z; }, a, b, c);
```

Both produce the same type and the same values. Any arity works, operands may repeat, and a position that is structurally zero in some operand simply contributes `0` to the function — so `fn` always gets one value per operand.

This generalizes the fusions the library already hand-rolls: `add(a, b, alpha, beta)`, `hadamard(a, b, multiplier)`, and `axpy` all exist to avoid exactly these intermediates.

**Element-wise only.** Each operand is read at the same `(i,j)` as the result, so `fuse` cannot express a matrix product — use `multiply` for that. The restriction is the point: because each operand element is touched exactly once per result element, there is nothing to recompute and fusing always pays. Fusing across a product would not, since each result element there reads a whole row and column.

**Result pattern.** By default the result stores the *union* of the operands' patterns. That is correct for any `fn` with `fn(0,...,0) == 0`, but not always minimal — a position stored by only one operand stays stored even if `fn` maps it to zero. For product-like functions, where a structurally zero operand forces a zero result, ask for the tighter pattern:

```cpp
auto weighted = SparseLinearAlgebra::fuse<SparseLinearAlgebra::FusePattern::Intersection>(
    [](double x, double y) { return x * y * 0.5; }, a, b);
```

**Nothing lazy escapes.** Evaluation finishes before `fuse` returns; there is no expression object holding references to the operands. The usual expression-template hazard — an expression captured with `auto` outliving the matrices it refers to — cannot arise here.

**On the GPU**, `fn` is called from wherever `fuse` is called, so it must be device-callable. A struct with a `SPARSEMAT_HD operator()` works on every toolchain, as does a lambda written inside a kernel. A lambda defined in host code and passed into device code needs nvcc's `--extended-lambda`, which this library does not require of its consumers.

### Comparison

`==` and `!=` compare *values*, not sparsity patterns — two matrices are equal when every element matches, whether or not both store that position explicitly. This matters because derived result patterns are often wider than the values warrant (`add` unions both operands' patterns, so `a + b` can hold an explicit `0.0` where a hand-written literal holds a structural zero):

```cpp
a == a.dense()                          // true — same values, different patterns
SparseLinearAlgebra::approx_equal(a, b, 1e-9)  // tolerant form, for post-solve results
```

`==` is exact; use `approx_equal` for anything that has been through a factorization.

### Scalar types

Binary operations require both operands to have the same `DataType`. Mixed types are a compile error rather than a silent promotion or truncation — the result would have to pick one type, and quietly narrowing a `double` operand into a `float` result is hard to notice afterwards. Convert explicitly to opt in:

```cpp
SparseMat<float, int, 3, 3, 0, 4, 8>  f(1, 2, 3);
SparseMat<double, int, 3, 3, 0, 4, 8> d(1, 2, 3);
// auto bad = f.add(d);                  // static_assert: mixed DataType
auto good = f.convert<double>().add(d);  // explicit, no surprise
```

### Operations

All operations return a new matrix with the result sparsity inferred at compile time.

| Method | Description |
|---|---|
| `a.mult(b)` | Matrix multiplication |
| `a.add(b)` | Addition (result sparsity = union) |
| `a.subtract(b)` | Subtraction (result sparsity = union) |
| `a.hadamard(b)` | Element-wise multiply (result sparsity = intersection) |
| `fuse(fn, a, b, ...)` | Apply `fn` element-wise across several matrices in one pass, no intermediates |
| `a.kronecker(b)` | Kronecker (tensor) product |
| `a.transpose()` | Matrix transpose |
| `a.ata()` | Gram matrix AᵀA, without materializing the transpose |
| `a.aat()` | Gram matrix AAᵀ, without materializing the transpose |
| `a.atba(b)` | Congruence transform AᵀBA, in one pass (no Aᵀ, no BA) |
| `a.abat(b)` | Congruence transform ABAᵀ — the Kalman covariance propagation `F P Fᵀ` |
| `a.abat_add(b, c)` | Fused `ABAᵀ + C` (e.g. `F P Fᵀ + Q`); `a.atba_add(b, c)` for the other side |
| `a.atb(b)` / `a.abt(b)` | AᵀB and ABᵀ without materializing a transpose |
| `a.submatrix<Row0, Col0, NRows, NCols>()` | Extract a rectangular block |
| `a.row<I>()` / `a.col<J>()` | Extract a single row or column as a matrix |
| `a.hcat(b)` / `a.vcat(b)` | Concatenate side by side or stacked |
| `a.permute<RowPerm, ColPerm>()` | Reorder rows and columns |
| `a.symmetric_permute<Perm>()` | Reorder symmetrically: `P A Pᵀ` |
| `rcm_ordering<T>()` | Reverse Cuthill–McKee ordering for a pattern, computed at compile time |
| `bandwidth<T>()` | Largest `\|i-j\|` over a pattern's stored entries |
| `outer(x, y, alpha)` | Outer product `alpha·x yᵀ` of two column vectors |
| `a.rank1_update(x, y, alpha)` | Fused `A + alpha·x yᵀ`; `a.symmetric_rank1_update(x, alpha)` for `x xᵀ` |
| `a.norm_1()` / `a.norm_inf()` / `a.max_abs()` | Largest absolute column sum, row sum, and element |
| `condition_number(a)` | 1-norm condition number `‖A‖₁·‖A⁻¹‖₁` (forms the inverse — a diagnostic, not a hot path) |
| `a.scale(factor)` | Scalar multiply, returns new matrix |
| `a.scale_inplace(factor)` | Scalar multiply in place |
| `a.shift(factor)` | Scalar add factor, returns new matrix |
| `a.shift_inplace(factor)` | Scalar add factor in place |
| `a.normalize()` | Divide by Frobenius norm, returns new matrix (no-op on the zero matrix) |
| `a.normalize_inplace()` | Divide by Frobenius norm in place (no-op on the zero matrix) |
| `a.frobenius()` | Frobenius norm (scalar) |
| `a.trace()` | Sum of diagonal elements |
| `a.dot(b)` | Dot product; `a` must be a 1×N row vector and `b` an N×1 column vector |
| `a.solve(b)` | Solve `a * x = b`; dispatches to forward/backward substitution or LU depending on sparsity (LU path does **no pivoting** — see below) |
| `a.cholesky()` | Cholesky factorization; returns a `Result<CholeskyFactor<L>>` — check `.ok()` before calling `.solve(b)` |
| `a.set_diagonal(value)` / `a.set_diagonal(values)` | Write to every stored diagonal entry, from a scalar or a `std::array` |
| `a.is_structurally_symmetric()` | Sparsity pattern is symmetric (values ignored) |
| `a.is_sparse_symmetric(tol)` | Stored non-zero values are symmetric within `tol` |
| `a.is_full_symmetric(tol)` | Full dense matrix equals its transpose within `tol` |
| `a.is_structurally_lower_triangular()` / `a.is_structurally_upper_triangular()` | No above/below-diagonal non-zeros |
| `a.is_numerically_lower_triangular(tol)` / `a.is_numerically_upper_triangular(tol)` | Above/below-diagonal stored values are within `tol` of zero |
| `a.determinant()` | Determinant, via the diagonal for triangular matrices and LU otherwise |
| `a.inverse()` / `cholesky_inverse(a)` | Matrix inverse by solving `A*X = I` (result is generally dense) |
| `a.least_squares_solve(b)` | Least-squares / minimum-norm solve for non-square `a` |
| `a.block_diagonal(b)` | Compose block-diagonally: `diag(a, b)` |
| `a.dense()` | Densify: same values, but with every position stored (returns a `SparseMat`) |
| `a.to_array()` | Convert to a plain `std::array<DataType, Rows*Cols>` in row-major order |
| `a.convert<T>()` | Copy with the scalar type changed to `T` |
| `a.entries()` | Every stored element as a `(row, col, value)` triple |
| `a == b` / `a != b` / `approx_equal(a, b, tol)` | Element-wise comparison by value (patterns need not match) |
| `a.print()` | Print non-zero values with their flat indices |
| `a.printDense()` | Print the full matrix, including zeros |
+ more 

Free functions are also available under `SparseLinearAlgebra::` (`multiply`, `add`, `subtract`, `hadamard`, `kronecker`, `transpose`, `ata`, `aat`, `atba`, `abat`, `abat_add`, `atba_add`, `atb`, `abt`, `submatrix`, `row`, `col`, `hcat`, `vcat`, `permute`, `symmetric_permute`, `permute_rows`, `permute_cols`, `rcm_ordering`, `bandwidth`, `inverse_permutation`, `outer`, `rank1_update`, `symmetric_rank1_update`, `norm_1`, `norm_inf`, `max_abs`, `condition_number`, `scale`, `scale_inplace`, `shift`, `shift_inplace`, `normalize`, `normalize_inplace`, `frobenius`, `trace`, `dense`, `fuse`, `power<N>`, `forward_solve`, `backward_solve`, `lu_solve`, `cholesky_solve`).

Solvers and factorizations (`solve`, `cholesky`, `lu_solve`, `forward_solve`, `backward_solve`, `cholesky_solve`, `lu_factorize`, `cholesky_factorize`) return a `Result<T>` rather than throwing, since these routines are also usable from CUDA device code where exceptions aren't available. Check `.ok()` (or `bool(result)`) before calling `.value()` — a failed result still holds a value, but it is not meaningful.

**Non-square systems.** `solve()` requires a square matrix. `least_squares_solve(b)` handles the rectangular cases: minimising `||Ax-b||₂` when overdetermined, and returning the minimum-norm solution when underdetermined. It works via the normal equations (`AᵀA x = Aᵀb`, solved with Cholesky), which **squares the condition number** — a matrix that is merely ill-conditioned for a direct solve can be numerically hopeless here, and a rank-deficient one fails outright via `ok()`. A QR-based solve avoids this and is the right answer for anything demanding; it isn't implemented because a compile-time-sparsity Householder QR is a much larger piece of work than reusing the existing Cholesky. `residual(a, x, b)` gives `A*x - b` for measuring the fit, since a least-squares solution doesn't generally satisfy `A*x == b`.

**Inverses are dense.** `inverse()` (and `cholesky_inverse()` for SPD matrices) solve `A*X = I` using the existing block-RHS triangular solves. The inverse of a sparse matrix is generally dense, so the result type carries many more stored values than the input — this is the one operation where compile-time sparsity works against you. Prefer `solve()` when you only need `A⁻¹b` for particular right-hand sides; it's faster and more accurate.

**No pivoting.** Neither the LU nor the Cholesky path does row swaps, so both are only valid for matrices that factorize stably without them (diagonally dominant, or otherwise pivot-free). A pivot that is merely *tiny* is as fatal as an exactly-zero one, so both report it as singular via `ok()` rather than dividing through and returning garbage. A pivot-stable but badly conditioned matrix will still return a poor answer with `ok() == true`; prefer `cholesky()` when the matrix is symmetric positive definite.

### Operators

```cpp
a * b     a + b     a - b     -a          // matrix arithmetic
a * 2.0   2.0 * a   a / 2.0               // scalar arithmetic, either ordering
a *= 2.0  a /= 2.0  a += b    a -= b      // in place
```

`+=` and `-=` cannot widen the sparsity pattern — it is part of the type, so the left operand has to keep its own. The right operand's non-zeros must be a subset, which is checked at compile time; use `a = a + b` when the union pattern is what you want.

### Static factories

```cpp
SparseMat<double, int, 3, 3>::identity()  // 3x3 identity matrix
```

## Benchmarks

Note: Benchmarks were performed on a laptop running all sorts of other things at the same time. Take these results with a grain of salt. More serious benchmarking would be needed before even considering adopting this.  

Comparison against Eigen's `SparseMatrix` and fixed-size `Matrix` (dense). Measured at 500,000 iterations with 50,000 warmup on `-O3 -march=native`. Times are nanoseconds per operation. Eigen is not using BLAS. 

Numbers are hardware-dependent and will drift over time; table will be updated with each release.

<!-- 
Developer Notes:

Run the following to update the table. 
cmake --build build --target update-benchmarks

This builds and runs `benchmark`, then rewrites the table below in place via `scripts/update_benchmark_table.py`. You will have to update the conclusions. 

Alternatively, there is a claude skill defined in .claude/skills/update-benchmarks. This 
will run the benchmarks, update the table, and update the conclusions. Make sure you READ them before commiting. 
-->

<!-- BENCHMARK_TABLE:START -->
```
------------------------------------------------------------------------------------------------------
Configuration                 Operation                sparsemat  Eigen sparse   Eigen dense
------------------------------------------------------------------------------------------------------
3x3 diagonal (33%)            multiply                      1.76 ns        266.03 ns          1.78 ns
3x3 diagonal (33%)            add                           1.78 ns         54.43 ns          1.73 ns
3x3 diagonal (33%)            transpose                     1.77 ns         57.63 ns          1.72 ns
3x3 diagonal (33%)            scale                         1.79 ns         48.45 ns          1.72 ns
3x3 diagonal (33%)            frobenius                     1.74 ns          5.62 ns          1.71 ns

3x3 first-row (33%)           multiply                      1.78 ns        263.76 ns          1.70 ns
3x3 first-row (33%)           add                           1.78 ns         55.31 ns          1.71 ns
3x3 first-row (33%)           scale                         1.78 ns         49.76 ns          1.72 ns
3x3 first-row (33%)           frobenius                     1.74 ns          5.37 ns          1.71 ns

3x3 full (100%)               multiply                      1.71 ns        333.84 ns          1.70 ns
3x3 full (100%)               add                           1.79 ns         92.18 ns          1.71 ns
3x3 full (100%)               scale                         1.97 ns         99.44 ns          1.70 ns
3x3 full (100%)               frobenius                     1.78 ns          7.00 ns          2.16 ns

5x5 diagonal (20%)            multiply                      1.78 ns        293.61 ns          1.92 ns
5x5 diagonal (20%)            add                           1.71 ns         65.59 ns          1.71 ns
5x5 diagonal (20%)            transpose                     1.76 ns         64.04 ns          1.72 ns
5x5 diagonal (20%)            scale                         1.77 ns         63.84 ns          1.70 ns
5x5 diagonal (20%)            frobenius                     1.72 ns          8.98 ns          1.70 ns

5x5 tridiagonal (52%)         multiply                      1.73 ns        446.65 ns          1.94 ns
5x5 tridiagonal (52%)         add                           1.78 ns        104.85 ns          1.70 ns
5x5 tridiagonal (52%)         scale                         1.78 ns         96.34 ns          1.74 ns
5x5 tridiagonal (52%)         frobenius                     1.77 ns          8.54 ns          1.73 ns

5x5 random sparse (24%)       multiply                      1.85 ns        306.76 ns          2.03 ns
5x5 random sparse (24%)       add                           1.80 ns         69.79 ns          2.19 ns
5x5 random sparse (24%)       scale                         1.84 ns         59.62 ns          1.77 ns
5x5 random sparse (24%)       frobenius                     1.81 ns          9.27 ns          1.90 ns

5x5 dense-ish (76%)           multiply                      1.73 ns        521.67 ns          2.03 ns
5x5 dense-ish (76%)           add                           1.77 ns        118.32 ns          1.71 ns
5x5 dense-ish (76%)           scale                         1.80 ns        103.89 ns          1.71 ns
5x5 dense-ish (76%)           frobenius                     1.77 ns         10.82 ns          1.75 ns

8x8 diagonal (12%)            multiply                      1.78 ns        328.02 ns         53.78 ns
8x8 diagonal (12%)            add                           1.78 ns         74.95 ns          1.69 ns
8x8 diagonal (12%)            transpose                     2.45 ns         70.80 ns          1.71 ns
8x8 diagonal (12%)            scale                         1.77 ns         72.58 ns          1.71 ns
8x8 diagonal (12%)            frobenius                     1.74 ns         11.81 ns          1.73 ns

8x8 tridiagonal (34%)         multiply                      1.78 ns        568.40 ns         49.61 ns
8x8 tridiagonal (34%)         add                           1.78 ns        130.45 ns          1.74 ns
8x8 tridiagonal (34%)         scale                         1.90 ns        117.19 ns          1.69 ns
8x8 tridiagonal (34%)         frobenius                     1.75 ns         12.73 ns          1.76 ns

8x8 random sparse (25%)       multiply                      1.78 ns        482.89 ns         49.74 ns
8x8 random sparse (25%)       add                           1.78 ns        101.04 ns          1.70 ns
8x8 random sparse (25%)       scale                         1.79 ns         77.66 ns          1.73 ns
8x8 random sparse (25%)       frobenius                     1.77 ns         14.73 ns          1.71 ns

5x5 SPD tridiagonal           cholesky (1 rhs)             27.84 ns        836.63 ns        121.00 ns
5x5 SPD tridiagonal           lu (1 rhs)                   52.79 ns       1436.15 ns        103.72 ns
5x5 SPD tridiagonal           cholesky (3 rhs)             43.51 ns        873.45 ns        150.99 ns
5x5 SPD tridiagonal           lu (3 rhs)                   62.75 ns       1624.00 ns        165.44 ns
------------------------------------------------------------------------------------------------------
```
<!-- BENCHMARK_TABLE:END -->

**vs Eigen sparse** — sparsemat wins across the board, roughly 2–345× depending on the operation. The gap is largest for multiply (`Eigen::SparseMatrix` pays CSC bookkeeping overhead per non-zero) and smallest for frobenius, which is memory-bandwidth bound rather than compute bound so there's less for compile-time zero elimination to remove.

**vs Eigen dense** — competitive on most operations. The largest win is multiply on 8×8: sparsemat is ~28–31× faster than Eigen dense because it skips all zero multiplications at compile time. Add, scale, and frobenius are roughly equal since those are memory-bandwidth bound rather than compute bound.

**Sparsity vs fill level** — multiply is where compile-time zero elimination pays off most. Add and scale show little sensitivity to fill level since they are bounded by the number of non-zeros regardless.

**Solving linear systems** (`5x5 SPD tridiagonal`) — `cholesky_solve`/`lu_solve` factorize and solve in one call, timed the same way as constructing+solving an Eigen solver each iteration. sparsemat's Cholesky solve is ~20–30× faster than `Eigen::SimplicialLLT` and ~3.5–4.5× faster than dense `.llt()`, since the fill-in pattern and every zero pivot skip are known at compile time — there's no runtime symbolic analysis step to pay for. Cholesky beats LU by roughly 2× here (as expected: LU does twice the factorization work on a matrix that's actually SPD), and the gap between 1 RHS and 3 RHS is much smaller than 3× because the RHS columns share the same factorization and just add more of the recursive-template solve step.

**Conclusion (CPU)** On the CPU it is probably a wash. sparsemat beats `Eigen::SparseMatrix` comfortably, but that is the easy comparison — at these sizes Eigen's own *dense* fixed-size matrices are already about as fast, except for multiply on 8x8. Set against the cost of declaring sparsity patterns at compile time (awkward, error prone, and slow to compile), the CPU case does not obviously pay for itself. See the GPU conclusion below for where it does.

To run the benchmarks yourself, run:
```bash
cmake --build build --target benchmark
./build/examples/benchmark
```
Or to run them and refresh the table above in the same step, use `cmake --build build --target update-benchmarks` (see above).

## GPU (CUDA)

This library is designed for small matrices, where parallelizing individual matrix operations across GPU threads is generally not beneficial. Nevertheless, every SparseMat operation is annotated with SPARSEMAT_HD (__host__ __device__ when compiled with nvcc, and a no-op otherwise), allowing the same header to be included and invoked from both host and CUDA device code without requiring a separate GPU build of the library.

This enables sparse matrix operations to be performed directly within GPU kernels, with each GPU thread executing its own independent matrix computations. The Kalman filter example demonstrates a practical application of this programming model.

Solvers and factorizations (`solve`, `cholesky`, `lu_solve`, `forward_solve`, `backward_solve`, `cholesky_solve`) never throw; they return a `Result<T>` instead, since exceptions aren't usable in device code. Always check `.ok()` before trusting `.value()`.

`std::cout`-based methods (`print()`, `printDense()`) are intentionally host-only — there's no portable way to make them callable from a kernel.

CUDA support is off by default and opt-in via CMake:

```bash
cmake -B build-cuda -DSPARSEMAT_ENABLE_CUDA=ON
cmake --build build-cuda
```

 When enabled, it builds:
- `test_sparsemat_gpu` — compiles `tests/test_sparsemat.cpp` through `nvcc` and runs the exact same tests from a device-side kernel, proving the `SPARSEMAT_HD` operations both compile for and run correctly on the device. On a machine with no physical GPU (e.g. CI), it detects this at runtime and reports itself as skipped rather than failed.
- `examples/kalman_gpu.cu` — a GPU build of the Kalman filter example.
- `benchmark_gpu` (also requires Eigen3) — see GPU benchmarks below.

### GPU benchmarks

`examples/bench_gpu.cu` compares three things, at a couple of batch sizes, all running the same 3×3/5×5/8×8 configurations (plus a 5×5 SPD Cholesky solve) as the CPU benchmarks above:
- **sparsemat GPU** — one CUDA thread per instance, each doing a full sparsemat operation on its own small matrix.
- **Eigen dense GPU** — the same one-thread-per-instance shape, but using `Eigen::Matrix<N,N>` directly in device code (Eigen's dense ops are device-callable; `Eigen::SparseMatrix` is not, so there's no "Eigen sparse GPU" column). This is the "what if you just used a dense matrix" GPU baseline.
- **sparsemat CPU** — the identical workload run single-threaded on the host, i.e. does moving this to the GPU even help.

Unlike the CPU benchmarks (which time a single call's latency), this times amortized nanoseconds-per-instance over a batch, since a single kernel launch alone costs microseconds — throughput over many independent small problems is the only thing that makes sense to measure on a GPU. The Cholesky-solve configuration has no Eigen-GPU column (`n/a`) since Eigen's `LLT` isn't device-callable.

Requires an actual CUDA-capable GPU to produce results (not just `nvcc` — build-only machines will report "no device" and skip, same as `test_sparsemat_gpu`):
```bash
cmake -B build -DSPARSEMAT_ENABLE_CUDA=ON
cmake --build build --target update-gpu-benchmarks
```
This builds `benchmark_gpu`, runs it, and splices its output into the table below via the same `scripts/update_benchmark_table.py` used for the CPU table (just pointed at different markers/binary). If no GPU is present, it leaves the table below untouched rather than failing.

<!-- BENCHMARK_TABLE_GPU:START -->
```
----------------------------------------------------------------------------------------------------
Configuration             Batch N            sparsemat GPU     Eigen dense GPUsparsemat CPU (1 core)
----------------------------------------------------------------------------------------------------
3x3 diagonal multiply     100000                   0.16 ns             0.72 ns              42.57 ns

5x5 tridiagonal multiply  100000                   1.34 ns             4.28 ns             326.48 ns

8x8 random sparse multiply100000                   1.44 ns            15.66 ns             511.32 ns

5x5 chol factorize+solve  100000                   4.12 ns                 n/a             397.89 ns

5x5 chol presolved        100000                   1.97 ns                 n/a             276.61 ns

3x3 diagonal multiply     1000000                  0.27 ns             0.73 ns              43.55 ns

5x5 tridiagonal multiply  1000000                  1.39 ns             3.37 ns             300.52 ns

8x8 random sparse multiply1000000                  1.60 ns            11.41 ns             529.58 ns

5x5 chol factorize+solve  1000000                  2.29 ns                 n/a             407.97 ns

5x5 chol presolved        1000000                  1.18 ns                 n/a             273.29 ns
----------------------------------------------------------------------------------------------------

Achieved global-memory bandwidth (operand reads + result write per instance)
----------------------------------------------------------------------------------------------------
Configuration             Batch N            sparsemat GPU     Eigen dense GPU
----------------------------------------------------------------------------------------------------
3x3 diagonal multiply     100000         341.8 GB/s (56 B)  212.1 GB/s (152 B)

5x5 tridiagonal multiply  100000        161.0 GB/s (216 B)   95.3 GB/s (408 B)

8x8 random sparse multiply100000        182.8 GB/s (264 B)  65.9 GB/s (1032 B)

5x5 chol factorize+solve  100000          11.7 GB/s (48 B)                 n/a

5x5 chol presolved        100000          24.4 GB/s (48 B)                 n/a

3x3 diagonal multiply     1000000        205.6 GB/s (56 B)  207.3 GB/s (152 B)

5x5 tridiagonal multiply  1000000       155.1 GB/s (216 B)  121.1 GB/s (408 B)

8x8 random sparse multiply1000000       164.8 GB/s (264 B)  90.5 GB/s (1032 B)

5x5 chol factorize+solve  1000000         20.9 GB/s (48 B)                 n/a

5x5 chol presolved        1000000         40.6 GB/s (48 B)                 n/a
----------------------------------------------------------------------------------------------------
```
<!-- BENCHMARK_TABLE_GPU:END -->

**Conclusion (GPU)** This is where the design pays for itself, and it is also the workload it was built for. At 5x5 and 8x8, sparsemat is roughly 2–4x faster per instance than the same one-thread-per-instance kernel using `Eigen::Matrix`, and the gap widens with size as compile-time zero elimination removes more work — the opposite of the CPU picture, where Eigen dense had already caught up. The Cholesky solve has no dense-GPU column at all because Eigen's `LLT` is not device-callable, so for a batch of small SPD solves inside a kernel there is no real alternative to compare against.

Note the CPU column in that table is the same workload run single-threaded on the host: 400–1000 ns/instance against roughly 0.2–0.5 ns on the GPU. Batched small-matrix work is exactly the case where moving to the GPU helps, and it is the original motivation — billions of independent Kalman filters, each doing many small sparse operations.

So the honest summary is: on the CPU, probably not worth the ergonomic cost; on the GPU, for batched small sparse problems, it is.

## Example

```cpp
#include "sparsemat.h"

int main() {
  SparseMat<double, int, 3, 3, 0, 4, 8> A(1, 2, 3);  // diagonal
  SparseMat<double, int, 3, 3, 0, 1, 2> B(4, 5, 6);  // first row

  auto C = A.mult(B);
  auto D = A.add(B);
  auto At = A.transpose();

  C.print();
  std::cout << "Frobenius norm of A: " << A.frobenius() << "\n";
}
```
