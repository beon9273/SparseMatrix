# sparsemat

> **Experimental** — built to explore what happens when you push sparsity into the type system. Not intended for production use. 

A header-only C++20 sparse matrix library where sparsity patterns are encoded as template parameters. 

Idea: if the non-zero structure of your matrices is fixed and known at compile time, the compiler can eliminate a whole bunch of operations.

Solution: user provides sparsity pattern as template parameters. Sparsity patterns of operations (e.g. `C = A × B`) are computed at compile time. Recursive templates are used to unroll operation loops at runtime. Zero operations (i.e, multiplications/additions where one of the terms is zero) are skipped 
using compile-time constructs( e.g. constexpr if).  


The tradeoff: 

Limitations: 
  - entering sparsity patterns at runtime is awkward and bugprone. Consider using scripts/generate.py
    to generate class names for a given sparse matrix. 
  - every distinct sparsity pattern is a distinct type, which makes compile times long for large configurations and binary size could expload pretty easily. 
  - designed for SMALL matrices: recursive templates are used to unroll matrix operations. Template recursion limits will be encountered for larger matrices. Probably only useful for up to 10x10
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

## Requirements

- C++20
- CMake 3.16+

## Platform Support

This project is currently tested only on Linux. Other platforms and toolchains may work, but they are not currently part of the tested matrix.

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

## Linking.

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

### Element access

```cpp
m.get<I, J>()       // compile-time indices; returns 0 for zero elements
m.get(i, j)         // runtime indices
m.set<I, J>(value)  // compile-time; static_assert if (I,J) is a zero index
m.set(i, j, value)  // runtime; returns false if (I,J) is a zero index
m.fill(value)       // set all non-zero storage to value
```

### Operations

All operations return a new matrix with the result sparsity inferred at compile time.

| Method | Description |
|---|---|
| `a.mult(b)` | Matrix multiplication |
| `a.add(b)` | Addition (result sparsity = union) |
| `a.subtract(b)` | Subtraction (result sparsity = union) |
| `a.hadamard(b)` | Element-wise multiply (result sparsity = intersection) |
| `a.kronecker(b)` | Kronecker (tensor) product |
| `a.transpose()` | Matrix transpose |
| `a.scale(factor)` | Scalar multiply, returns new matrix |
| `a.scale_inplace(factor)` | Scalar multiply in place |
| `a.shift(factor)` | Scalar add factor, returns new matrix |
| `a.shift_inplace(factor)` | Scalar add factor in place |
| `a.normalize()` | Divide by Frobenius norm, returns new matrix (no-op on the zero matrix) |
| `a.normalize_inplace()` | Divide by Frobenius norm in place (no-op on the zero matrix) |
| `a.frobenius()` | Frobenius norm (scalar) |
| `a.trace()` | Sum of diagonal elements |
| `a.dot(b)` | Dot product; `a` must be a 1×N row vector and `b` an N×1 column vector |
| `a.solve(b)` | Solve `a * x = b`; dispatches to forward/backward substitution or LU depending on sparsity |
| `a.cholesky()` | Cholesky factorization; returns a `Result<CholeskyFactor<L>>` — check `.ok()` before calling `.solve(b)` |
| `a.set_diagonal(value)` / `a.set_diagonal(values)` | Write to every stored diagonal entry, from a scalar or a `std::array` |
| `a.is_structurally_symmetric()` | Sparsity pattern is symmetric (values ignored) |
| `a.is_sparse_symmetric(tol)` | Stored non-zero values are symmetric within `tol` |
| `a.is_full_symmetric(tol)` | Full dense matrix equals its transpose within `tol` |
| `a.is_structurally_lower_triangular()` / `a.is_structurally_upper_triangular()` | No above/below-diagonal non-zeros |
| `a.is_numerically_lower_triangular(tol)` / `a.is_numerically_upper_triangular(tol)` | Above/below-diagonal stored values are within `tol` of zero |
| `a.dense()` | Convert to a dense array |
| `a.print()` | Print non-zero values with their flat indices |
| `a.printDense()` | Print the full matrix, including zeros |
+ more 

Free functions are also available under `SparseLinearAlgebra::` (`multiply`, `add`, `subtract`, `hadamard`, `kronecker`, `transpose`, `scale`, `scale_inplace`, `shift`, `shift_inplace`, `normalize`, `normalize_inplace`, `frobenius`, `trace`, `dense`, `power<N>`, `forward_solve`, `backward_solve`, `lu_solve`, `cholesky_solve`).

Solvers and factorizations (`solve`, `cholesky`, `lu_solve`, `forward_solve`, `backward_solve`, `cholesky_solve`, `lu_factorize`, `cholesky_factorize`) return a `Result<T>` rather than throwing, since these routines are also usable from CUDA device code where exceptions aren't available. Check `.ok()` (or `bool(result)`) before calling `.value()` — a failed result still holds a value, but it is not meaningful.

### Static factories

```cpp
SparseMat<double, int, 3, 3>::identity()  // 3x3 identity matrix
```

## Benchmarks

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
3x3 diagonal (33%)            multiply                      2.83 ns        485.93 ns          3.51 ns
3x3 diagonal (33%)            add                           2.84 ns        125.34 ns          2.24 ns
3x3 diagonal (33%)            transpose                     2.85 ns        106.63 ns          2.82 ns
3x3 diagonal (33%)            scale                         1.96 ns         85.37 ns          4.72 ns
3x3 diagonal (33%)            frobenius                     1.88 ns          4.84 ns          2.00 ns

3x3 first-row (33%)           multiply                      5.66 ns        324.35 ns          1.87 ns
3x3 first-row (33%)           add                           1.84 ns         60.13 ns          5.22 ns
3x3 first-row (33%)           scale                         3.53 ns         60.75 ns          1.93 ns
3x3 first-row (33%)           frobenius                     4.61 ns          4.78 ns          2.62 ns

3x3 full (100%)               multiply                      1.83 ns        464.65 ns          4.11 ns
3x3 full (100%)               add                           1.91 ns        129.42 ns          1.86 ns
3x3 full (100%)               scale                         1.83 ns        100.30 ns          1.90 ns
3x3 full (100%)               frobenius                     1.98 ns          4.93 ns          1.78 ns

5x5 diagonal (20%)            multiply                      1.83 ns        428.58 ns          1.80 ns
5x5 diagonal (20%)            add                           1.83 ns         66.74 ns          1.85 ns
5x5 diagonal (20%)            transpose                     1.89 ns         89.84 ns          3.78 ns
5x5 diagonal (20%)            scale                         1.83 ns         71.22 ns          1.68 ns
5x5 diagonal (20%)            frobenius                     1.88 ns          6.36 ns          1.82 ns

5x5 tridiagonal (52%)         multiply                      1.91 ns        500.13 ns          1.84 ns
5x5 tridiagonal (52%)         add                           2.83 ns        241.88 ns          1.80 ns
5x5 tridiagonal (52%)         scale                         1.83 ns        161.09 ns          2.76 ns
5x5 tridiagonal (52%)         frobenius                     2.06 ns          9.66 ns          1.83 ns

5x5 random sparse (24%)       multiply                      1.73 ns        405.89 ns          1.89 ns
5x5 random sparse (24%)       add                           3.17 ns         89.40 ns          1.81 ns
5x5 random sparse (24%)       scale                         2.00 ns         70.09 ns          3.15 ns
5x5 random sparse (24%)       frobenius                     2.48 ns          9.58 ns          1.87 ns

5x5 dense-ish (76%)           multiply                      2.68 ns        755.52 ns          2.31 ns
5x5 dense-ish (76%)           add                           1.81 ns        199.50 ns          2.70 ns
5x5 dense-ish (76%)           scale                         1.92 ns        153.27 ns          1.87 ns
5x5 dense-ish (76%)           frobenius                     1.94 ns          9.46 ns          1.87 ns

8x8 diagonal (12%)            multiply                      1.85 ns        496.86 ns         76.63 ns
8x8 diagonal (12%)            add                           5.26 ns        119.56 ns          1.84 ns
8x8 diagonal (12%)            transpose                     2.99 ns        106.81 ns          3.94 ns
8x8 diagonal (12%)            scale                         1.89 ns        107.55 ns          1.93 ns
8x8 diagonal (12%)            frobenius                     1.91 ns         11.65 ns          1.83 ns

8x8 tridiagonal (34%)         multiply                      1.82 ns        625.61 ns         56.42 ns
8x8 tridiagonal (34%)         add                           1.84 ns        199.22 ns          1.87 ns
8x8 tridiagonal (34%)         scale                         1.83 ns        132.87 ns          1.89 ns
8x8 tridiagonal (34%)         frobenius                     1.89 ns         14.42 ns          1.84 ns

8x8 random sparse (25%)       multiply                      2.07 ns        793.51 ns         60.53 ns
8x8 random sparse (25%)       add                           1.94 ns        105.93 ns          1.93 ns
8x8 random sparse (25%)       scale                         1.68 ns        103.09 ns          1.86 ns
8x8 random sparse (25%)       frobenius                     1.81 ns         12.96 ns          1.76 ns

5x5 SPD tridiagonal           cholesky (1 rhs)             30.07 ns       1052.77 ns        160.36 ns
5x5 SPD tridiagonal           lu (1 rhs)                  124.00 ns       1750.02 ns        130.51 ns
5x5 SPD tridiagonal           cholesky (3 rhs)             51.63 ns       1228.43 ns        231.50 ns
5x5 SPD tridiagonal           lu (3 rhs)                   88.19 ns       2308.15 ns        186.32 ns
------------------------------------------------------------------------------------------------------
```
<!-- BENCHMARK_TABLE:END -->

**vs Eigen sparse** — sparsemat wins across the board, roughly 2–450× depending on the operation. The gap is largest for multiply (`Eigen::SparseMatrix` pays CSC bookkeeping overhead per non-zero) and smallest for frobenius, which is memory-bandwidth bound rather than compute bound so there's less for compile-time zero elimination to remove.

**vs Eigen dense** — competitive on most operations. The largest win is multiply on 8×8: sparsemat is ~30–60× faster than Eigen dense because it skips all zero multiplications at compile time. Add, scale, and frobenius are roughly equal since those are memory-bandwidth bound rather than compute bound.

**Sparsity vs fill level** — multiply is where compile-time zero elimination pays off most. Add and scale show little sensitivity to fill level since they are bounded by the number of non-zeros regardless.

**Solving linear systems** (`5x5 SPD tridiagonal`) — `cholesky_solve`/`lu_solve` factorize and solve in one call, timed the same way as constructing+solving an Eigen solver each iteration. sparsemat's Cholesky solve is ~35–40× faster than `Eigen::SimplicialLLT` and ~4–6× faster than dense `.llt()`, since the fill-in pattern and every zero pivot skip are known at compile time — there's no runtime symbolic analysis step to pay for. Cholesky beats LU by roughly 2× here (as expected: LU does twice the factorization work on a matrix that's actually SPD), and the gap between 1 RHS and 3 RHS is much smaller than 3× because the RHS columns share the same factorization and just add more of the recursive-template solve step.

**Conclusion** It is probably a wash -- Defining sparsity at Runtime is a pain (and really error prone). It is probably not worth it.

To run the benchmarks yourself, run:
```bash
cmake --build build --target benchmark
./build/examples/benchmark
```
Or to run them and refresh the table above in the same step, use `cmake --build build --target update-benchmarks` (see above).

## GPU (CUDA)

Every `SparseMat` operation is marked `SPARSEMAT_HD` (`__host__ __device__` under `nvcc`, a no-op elsewhere), so the same header can be included and called from CUDA device code — no separate GPU build of the library is needed.

Solvers and factorizations (`solve`, `cholesky`, `lu_solve`, `forward_solve`, `backward_solve`, `cholesky_solve`) never throw; they return a `Result<T>` instead, since exceptions aren't usable in device code. Always check `.ok()` before trusting `.value()`.

`std::cout`-based methods (`print()`, `printDense()`) are intentionally host-only — there's no portable way to make them callable from a kernel.

CUDA support is off by default and opt-in via CMake:

```bash
cmake -B build-cuda -DSPARSEMAT_ENABLE_CUDA=ON
cmake --build build-cuda
```

This requires `nvcc` on your `PATH`; if no CUDA compiler is found, CMake emits a warning and skips the CUDA targets rather than failing the configure step. When enabled, it builds:
- `test_sparsemat_gpu` — compiles `tests/test_sparsemat.cpp` through `nvcc` and runs the exact same tests from a device-side kernel, proving the `SPARSEMAT_HD` operations both compile for and run correctly on the device. On a machine with no physical GPU (e.g. CI), it detects this at runtime and reports itself as skipped rather than failed.
- `examples/kalman_gpu.cu` — a GPU build of the Kalman filter example.

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
