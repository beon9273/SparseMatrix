# Examples

All examples include `sparsemat.h` from the amalgamated single-header in `build/dist/`. Build them from the project root:

```sh
cmake -B build
cmake --build build
```

To skip building examples:

```sh
cmake -B build -DSPARSEMAT_BUILD_EXAMPLES=OFF
```

---

## demo

**Binary:** `build/examples/sparsemat`

Basic usage of the core matrix operations: multiply, add, subtract, and transpose on two 3×3 sparse matrices with compile-time sparsity patterns.

```sh
./build/examples/sparsemat
```

---

## kalman

**Binary:** `build/examples/kalman`

A Kalman filter tracking a projectile on a parabolic trajectory. The number of GPS sensors (`NumGPS`) is a compile-time template parameter — all measurement matrix sizes and indices are resolved at compile time. Demonstrates using `SparseMat` for the state transition and covariance matrices in a realistic estimation problem.

```sh
./build/examples/kalman
```

Key parameters (edit at the top of `kalman.cpp`):

| Constant | Default | Description |
|----------|---------|-------------|
| `NumGPS` | `10` | Number of GPS sensors |
| `dt` | `0.5` s | Time step |
| `N` | `10` | Simulation steps |
| `GPS_STD` | `5.0` m | GPS noise standard deviation |

---

## benchmark

**Binary:** `build/examples/benchmark`  
**Requires:** Eigen3 (`apt install libeigen3-dev` or equivalent)

Benchmarks `SparseMat` multiply and add against equivalent Eigen dense and sparse operations. Reports throughput in iterations/second and relative speedup. Runs a warmup pass before timing to avoid cold-cache noise.

```sh
./build/examples/benchmark
```
