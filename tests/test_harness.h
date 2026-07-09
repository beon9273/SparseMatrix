#pragma once

// Minimal test harness shared by the CPU and GPU test runners.
//
// check()/check_eq()/check_near() are SPARSEMAT_HD so the exact same
// test_*() functions in test_sparsemat.cpp can run unmodified on the host
// (tests/test_sparsemat.cpp as its own executable) or inside a CUDA kernel
// (tests/test_sparsemat_gpu.cu) — one set of tests, two runners.
//
// Device code can't use std::cout/std::string, so results are tracked as
// two plain counters (passed/failed) rather than a printed PASS/FAIL log.
// On a CUDA build there are two independent counter pairs: the host ones
// (used when running as a normal host program) and a __device__ pair (used
// when check() executes inside a kernel); the GPU runner reads the device
// pair back via cudaMemcpyFromSymbol after the kernel completes. A plain
// (non-CUDA) build only ever sees the host pair.

#include <cmath>
#include <cstdio>

// Relies on SPARSEMAT_HD already being defined, so callers must
// #include "sparsemat.h" (or any sparsemat header) before this one. Not
// included directly here because the CPU/GPU test targets only add the
// flattened dist/ header to their include path, not the include/ tree that
// sparsemat/concepts/concepts.h lives under.

namespace test_harness {

static int host_passed = 0;
static int host_failed = 0;

#if defined(__CUDACC__)
// Declared unconditionally under nvcc (not just __CUDA_ARCH__) so host code
// in the GPU runner can reference this symbol via cudaMemcpyFromSymbol.
__device__ static int device_passed = 0;
__device__ static int device_failed = 0;
#endif

SPARSEMAT_HD inline void check(bool condition, const char* name) {
#if defined(__CUDA_ARCH__)
  if (condition) {
    atomicAdd(&device_passed, 1);
  } else {
    printf("[FAIL] %s\n", name);
    atomicAdd(&device_failed, 1);
  }
#else
  if (condition) {
    std::printf("[PASS] %s\n", name);
    ++host_passed;
  } else {
    std::printf("[FAIL] %s\n", name);
    ++host_failed;
  }
#endif
}

}  // namespace test_harness

using test_harness::check;

template<typename T>
SPARSEMAT_HD inline void check_eq(T a, T b, const char* name) {
  check(a == b, name);
}

SPARSEMAT_HD inline void check_near(double a, double b, const char* name, double eps = 1e-9) {
  check(std::abs(a - b) < eps, name);
}
