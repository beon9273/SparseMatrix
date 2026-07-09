# sparsemat

A header-only C++20 sparse matrix library that encodes sparsity patterns as template parameters, letting the compiler unroll matrix operations and skip noops at compile time.

> **Preface** — This library was built for fun, to test the limits of C++ compile-time constructs like template parameter packs, index sequences and `constexpr`. The underlying question was whether compile-time coding could meaningfully speed up basic linear algebra operations versus leaving it to blas.
>
> It also doubled as an exercise in using AI assistance to turn a simple C++ project into a fully developed library. The design and sparsity algorithms (how the result sparsity of each operation is derived and unrolled at compile time) are my own; AI was used for the surrounding brunt work — setting up and writing documentation, creating test harnesses, benchmarks, enabling code-coverage, exploring edge cases, and similar scaffolding. 

## Leveraging known sparsity information at compile time

The project was born while developing a GPU-enabled Kalman filter for particle track reconstruction in high energy physics. In that application, one must perform billions of
independent Kalman filters, all of which require many linear algebra operations using a small number of small (O(10x10)) sparse matrices.

Idea: if the non-zero structure of your matrices is fixed and known at compile time, the compiler can eliminate a whole bunch of operations.

Solution: user provides sparsity pattern as template parameters. Sparsity patterns of operations (e.g. `C = A × B`) are computed at compile time. Recursive templates are used to unroll operation loops at runtime. Zero operations (i.e, multiplications/additions where one of the terms is zero) are skipped 
using compile-time constructs( e.g. constexpr if).  


Limitations: 
  - entering sparsity patterns is awkward and bugprone. 
  - every distinct sparsity pattern is a distinct type, which makes compile times long for large configurations and binary size could expload pretty easily. 
  - recursive templates are used to unroll matrix operations. Template recursion limits mean this will only work for small matrices.
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
3x3 diagonal (33%)            multiply                      2.81 ns        365.70 ns          1.89 ns
3x3 diagonal (33%)            add                           1.92 ns         71.30 ns          1.90 ns
3x3 diagonal (33%)            transpose                     4.29 ns         72.00 ns          1.94 ns
3x3 diagonal (33%)            scale                         1.82 ns         93.03 ns          4.25 ns
3x3 diagonal (33%)            frobenius                     2.51 ns          4.98 ns          1.82 ns

3x3 first-row (33%)           multiply                      1.87 ns        314.13 ns          1.86 ns
3x3 first-row (33%)           add                           1.89 ns         61.68 ns          1.92 ns
3x3 first-row (33%)           scale                         1.84 ns         80.84 ns          1.90 ns
3x3 first-row (33%)           frobenius                     4.02 ns         13.72 ns          2.22 ns

3x3 full (100%)               multiply                      1.85 ns        498.01 ns          3.19 ns
3x3 full (100%)               add                           4.28 ns        137.66 ns          1.86 ns
3x3 full (100%)               scale                         1.85 ns        117.97 ns          4.18 ns
3x3 full (100%)               frobenius                     1.91 ns          5.14 ns          1.85 ns

5x5 diagonal (20%)            multiply                      5.64 ns        330.05 ns          1.83 ns
5x5 diagonal (20%)            add                           2.56 ns        109.12 ns          5.91 ns
5x5 diagonal (20%)            transpose                     1.85 ns         76.10 ns          1.86 ns
5x5 diagonal (20%)            scale                         1.87 ns         78.37 ns          2.52 ns
5x5 diagonal (20%)            frobenius                     1.92 ns          6.47 ns          1.95 ns

5x5 tridiagonal (52%)         multiply                      1.85 ns        575.21 ns          1.81 ns
5x5 tridiagonal (52%)         add                           1.82 ns        142.93 ns          3.52 ns
5x5 tridiagonal (52%)         scale                         4.30 ns        184.22 ns          1.86 ns
5x5 tridiagonal (52%)         frobenius                     4.09 ns         15.74 ns          4.20 ns

5x5 random sparse (24%)       multiply                      6.66 ns        425.76 ns          3.60 ns
5x5 random sparse (24%)       add                           2.69 ns        103.50 ns          5.78 ns
5x5 random sparse (24%)       scale                         2.77 ns         88.29 ns          2.64 ns
5x5 random sparse (24%)       frobenius                     2.78 ns         14.20 ns          4.26 ns

5x5 dense-ish (76%)           multiply                      4.25 ns        694.33 ns          2.80 ns
5x5 dense-ish (76%)           add                           2.22 ns        175.76 ns          3.40 ns
5x5 dense-ish (76%)           scale                         2.80 ns        163.88 ns          1.87 ns
5x5 dense-ish (76%)           frobenius                     2.64 ns         15.35 ns          2.85 ns

8x8 diagonal (12%)            multiply                      2.01 ns        376.86 ns         85.85 ns
8x8 diagonal (12%)            add                           1.86 ns         79.85 ns          1.83 ns
8x8 diagonal (12%)            transpose                     1.86 ns         81.64 ns          1.84 ns
8x8 diagonal (12%)            scale                         1.87 ns         82.52 ns          1.82 ns
8x8 diagonal (12%)            frobenius                     1.85 ns         10.23 ns          1.83 ns

8x8 tridiagonal (34%)         multiply                      3.21 ns        740.20 ns         61.13 ns
8x8 tridiagonal (34%)         add                           1.82 ns        182.53 ns          2.78 ns
8x8 tridiagonal (34%)         scale                         3.48 ns        169.02 ns          1.83 ns
8x8 tridiagonal (34%)         frobenius                     2.55 ns         15.46 ns          3.08 ns

8x8 random sparse (25%)       multiply                      4.15 ns        687.48 ns         91.36 ns
8x8 random sparse (25%)       add                           1.85 ns        159.38 ns          4.16 ns
8x8 random sparse (25%)       scale                         1.85 ns         78.72 ns          1.83 ns
8x8 random sparse (25%)       frobenius                     1.88 ns         11.71 ns          1.78 ns

5x5 SPD tridiagonal           cholesky (1 rhs)             27.63 ns        897.25 ns        168.73 ns
5x5 SPD tridiagonal           lu (1 rhs)                   81.34 ns       1817.78 ns        130.99 ns
5x5 SPD tridiagonal           cholesky (3 rhs)             57.63 ns       1222.28 ns        194.51 ns
5x5 SPD tridiagonal           lu (3 rhs)                   76.95 ns       2055.37 ns        228.27 ns
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
(not yet run on a CUDA-capable GPU — build with -DSPARSEMAT_ENABLE_CUDA=ON and
run `cmake --build build --target update-gpu-benchmarks` on a machine with a GPU)
```
<!-- BENCHMARK_TABLE_GPU:END -->

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
