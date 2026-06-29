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
// SparseMat<DataType, Rows, Cols, NonZeroIndices...>
// Non-zero indices are flat row-major indices.

SparseMat<double, 3, 3, 0, 4, 8> A(1, 2, 3);  // diagonal matrix
SparseMat<double, 3, 3, 0, 1, 2> B(4, 5, 6);  // first row only

// Result type and sparsity computed at compile time
// C === SparseMat<double, 3, 3, 0, 3, 6>
auto C = A.mult(B);
```

## Requirements

- C++20
- CMake 3.16+


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
# produces build/dist/sparsemat_dist.h
```
```cpp
#include "sparsemat_dist.h"
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
SparseMat<DataType, Rows, Cols, NonZeroIndices...>
```

### Constructors

```cpp
SparseMat<double, 2, 2, 0, 3> m;             // zero-initialized
SparseMat<double, 2, 2, 0, 3> m(1.0, 2.0);  // value per non-zero
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
| `a.transpose()` | Matrix transpose |
| `a.scale(factor)` | Scalar multiply, returns new matrix |
| `a.scale_inplace(factor)` | Scalar multiply in place |
| `a.shift(factor)` | Scalar add factor, returns new matrix |
| `a.shift_inplace(factor)` | Scalar add factor in place |
| `a.normalize()` | Divide by Frobenius norm, returns new matrix |
| `a.normalize_inplace()` | Divide by Frobenius norm in place |
| `a.frobenius()` | Frobenius norm (scalar) |
| `a.dot(b)` | Dot product for row or column vectors |
| `a.dense()` | Convert to a dense array |
| `a.print()` | Print non-zero values with their flat indices |
+ more 

Free functions are also available under `SparseLinearAlgebra::` (`multiply`, `add`, `subtract`, `hadamard`, `transpose`, `scale`, `scale_inplace`, `normalize`, `normalize_inplace`, `frobenius`, `trace`, `dense`, `power<N>`).

### Static factories

```cpp
SparseMat<double, 1, 1, 0>::identity<3>()  // 3x3 identity matrix
```

## Benchmarks

Comparison against Eigen's `SparseMatrix` and fixed-size `Matrix` (dense). Measured at 500,000 iterations with 50,000 warmup on `-O3 -march=native`. Times are nanoseconds per operation. Eigen is not using BLAS. 

```
Configuration                  Operation     sparsemat   Eigen sparse   Eigen dense
-------------------------------------------------------------------------------------
3x3 diagonal (33%)             multiply          2.22 ns      429.91 ns        3.18 ns
3x3 diagonal (33%)             add               4.44 ns       77.64 ns        2.10 ns
3x3 diagonal (33%)             transpose         3.91 ns      108.74 ns        5.02 ns
3x3 diagonal (33%)             scale             2.81 ns       86.77 ns        3.94 ns
3x3 diagonal (33%)             frobenius         2.11 ns        7.40 ns        1.87 ns

3x3 first-row (33%)            multiply          1.96 ns      451.13 ns        3.05 ns
3x3 first-row (33%)            add               2.69 ns      107.16 ns        2.83 ns
3x3 first-row (33%)            scale             2.77 ns       97.50 ns        2.67 ns
3x3 first-row (33%)            frobenius         3.04 ns        7.35 ns        3.37 ns

3x3 full (100%)                multiply          2.65 ns      522.83 ns        3.74 ns
3x3 full (100%)                add               2.77 ns      224.13 ns        2.68 ns
3x3 full (100%)                scale             4.15 ns      168.67 ns        2.93 ns
3x3 full (100%)                frobenius         3.19 ns        7.85 ns        3.43 ns

5x5 diagonal (20%)             multiply          6.16 ns      385.82 ns        3.82 ns
5x5 diagonal (20%)             add               2.78 ns      181.73 ns        2.70 ns
5x5 diagonal (20%)             transpose         1.78 ns       84.32 ns        2.72 ns
5x5 diagonal (20%)             scale             2.77 ns      112.58 ns        1.83 ns
5x5 diagonal (20%)             frobenius         1.71 ns        6.60 ns        2.56 ns

5x5 tridiagonal (52%)          multiply          2.06 ns      606.53 ns        3.17 ns
5x5 tridiagonal (52%)          add               1.77 ns      131.59 ns        2.00 ns
5x5 tridiagonal (52%)          scale             1.74 ns      122.84 ns        1.83 ns
5x5 tridiagonal (52%)          frobenius         2.12 ns        7.42 ns        1.88 ns

5x5 random sparse (24%)        multiply          2.28 ns      378.41 ns        2.90 ns
5x5 random sparse (24%)        add               1.85 ns       82.27 ns        2.25 ns
5x5 random sparse (24%)        scale             2.13 ns       71.24 ns        1.81 ns
5x5 random sparse (24%)        frobenius         1.84 ns       14.73 ns        2.16 ns

5x5 dense-ish (76%)            multiply          3.27 ns      599.29 ns        1.89 ns
5x5 dense-ish (76%)            add               1.90 ns      148.74 ns        2.24 ns
5x5 dense-ish (76%)            scale             1.82 ns      132.14 ns        1.84 ns
5x5 dense-ish (76%)            frobenius         1.83 ns        9.19 ns        1.83 ns

8x8 diagonal (12%)             multiply          1.82 ns      392.69 ns       70.13 ns
8x8 diagonal (12%)             add               2.06 ns       99.33 ns        1.83 ns
8x8 diagonal (12%)             transpose         3.22 ns       80.45 ns        3.59 ns
8x8 diagonal (12%)             scale             1.77 ns       96.23 ns        3.21 ns
8x8 diagonal (12%)             frobenius         1.82 ns       11.58 ns        1.90 ns

8x8 tridiagonal (34%)          multiply          1.81 ns      699.45 ns       55.78 ns
8x8 tridiagonal (34%)          add               1.85 ns      163.07 ns        1.85 ns
8x8 tridiagonal (34%)          scale             1.85 ns      134.85 ns        1.84 ns
8x8 tridiagonal (34%)          frobenius         3.62 ns       12.25 ns        1.82 ns

8x8 random sparse (25%)        multiply          1.82 ns      710.19 ns       66.26 ns
8x8 random sparse (25%)        add               1.67 ns      140.12 ns        1.81 ns
8x8 random sparse (25%)        scale             1.84 ns       94.57 ns        1.95 ns
8x8 random sparse (25%)        frobenius         1.68 ns       10.99 ns        1.74 ns
```

**vs Eigen sparse** — sparsemat wins by 50–400× across the board. `Eigen::SparseMatrix` carries dynamic allocation and CSC bookkeeping overhead that dominates at small sizes.

**vs Eigen dense** — competitive on all operations. The largest win is multiply on 8×8: sparsemat is ~35–38× faster than Eigen dense because it skips all zero multiplications at compile time. Add, scale, and frobenius are roughly equal since those are memory-bandwidth bound rather than compute bound.

**Sparsity vs fill level** — multiply is where compile-time zero elimination pays off most. Add and scale show little sensitivity to fill level since they are bounded by the number of non-zeros regardless.

**Conclusion** It is probably a wash -- Defining sparsity at Runtime is a pain (and really error prone). It is probably not worth it.

To run benchmarks yourself:
```bash
cmake --build build --target benchmark
./build/benchmark
```

## Example

```cpp
#include "sparsemat.h"

int main() {
  SparseMat<double, 3, 3, 0, 4, 8> A(1, 2, 3);  // diagonal
  SparseMat<double, 3, 3, 0, 1, 2> B(4, 5, 6);  // first row

  auto C = A.mult(B);
  auto D = A.add(B);
  auto At = A.transpose();

  C.print();
  std::cout << "Frobenius norm of A: " << A.frobenius() << "\n";
}
```
