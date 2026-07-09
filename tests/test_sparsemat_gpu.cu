/**
 * GPU test runner: reuses the exact same test_*() functions defined in
 * test_sparsemat.cpp (the single source of truth for what's being tested),
 * but calls them from inside an actual CUDA kernel instead of from a host
 * main(). This proves the SPARSEMAT_HD operations they exercise not only
 * compile for the device, but run correctly on it.
 *
 * No physical GPU is required to build this file, only to *run* it — if
 * cudaGetDeviceCount() reports zero devices (e.g. a CI runner or a sandbox
 * with no GPU passthrough), main() exits with SPARSEMAT_SKIP_EXIT_CODE so
 * ctest can mark this as skipped rather than failed.
 */

#define SPARSEMAT_TEST_NO_MAIN
#include "test_sparsemat.cpp"

#include <cstdio>

// Matches the SKIP_RETURN_CODE set on this test in CMakeLists.txt.
constexpr int SPARSEMAT_SKIP_EXIT_CODE = 99;

__global__ void run_gpu_tests_kernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    run_all_tests();
  }
}

int main() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::printf("No CUDA device available (this binary already having built proves the tests "
                "compile for the device); skipping kernel execution.\n");
    return SPARSEMAT_SKIP_EXIT_CODE;
  }

  run_gpu_tests_kernel<<<1, 1>>>();
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
    return 1;
  }
  cudaDeviceSynchronize();

  int gpu_passed = 0;
  int gpu_failed = 0;
  cudaMemcpyFromSymbol(&gpu_passed, test_harness::device_passed, sizeof(gpu_passed));
  cudaMemcpyFromSymbol(&gpu_failed, test_harness::device_failed, sizeof(gpu_failed));

  std::printf("\n%d passed, %d failed (ran on device).\n", gpu_passed, gpu_failed);
  return gpu_failed > 0 ? 1 : 0;
}
