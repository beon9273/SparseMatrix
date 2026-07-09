/**
 * GPU throughput benchmark: sparsemat vs Eigen dense, both run as one
 * instance per CUDA thread over a batch of N independent small matrices,
 * compared against the same sparsemat workload run single-threaded on the
 * CPU.
 *
 * This is a different shape from examples/bench.cpp: that benchmark times a
 * single call's latency (nanoseconds for one 3x3 multiply), which is
 * meaningless on GPU — a single kernel launch alone costs several
 * microseconds, dwarfing a handful of FLOPs. What actually matters for GPU
 * work is throughput over a batch of independent problems, so here the
 * batch size N is the swept variable and every timing is reported as
 * amortized nanoseconds-per-instance (total kernel time / N).
 *
 * Every instance's matrix is built once on the host and transferred to the
 * device as an already-typed object (cudaMemcpy of a std::vector<Mat3>, the
 * same pattern examples/kalman_gpu.cu uses for its filter array), not as a
 * flat scalar buffer reconstructed field-by-field inside the timed kernel.
 * The timed kernel does nothing but index in and compute — it still has to
 * read its operands from global memory (that's an unavoidable, and
 * legitimate, part of "how fast can the GPU push through N independent
 * problems"), but it does no extra gather/reconstruction work beyond that
 * single indexed read, so the measurement isn't inflated by busywork that
 * has nothing to do with the operation being benchmarked.
 *
 * Eigen dense matrices are usable inside CUDA device code (Eigen's core
 * dense ops are marked EIGEN_DEVICE_FUNC and compile under nvcc), so the
 * "Eigen dense GPU" column is a real per-thread Eigen::Matrix computation,
 * not a host-side stand-in. Eigen::SparseMatrix has no such device support
 * (it's a dynamically-allocated, pointer-based CSR/CSC structure with no
 * GPU backend in Eigen itself), so there is no "Eigen sparse GPU" column —
 * that gap is exactly the comparison this benchmark exists to make.
 *
 * Eigen's factorization classes (LLT, PartialPivLU, ...) are not
 * device-callable, so the Cholesky-solve configuration below only compares
 * sparsemat GPU vs sparsemat CPU; the Eigen column is reported as "n/a".
 *
 * No physical GPU is required to build this file, only to run its timed
 * kernels — see main() for the device-count check.
 */

// Fixed-size Eigen matrices default to requiring SIMD alignment for some
// sizes, which std::vector's allocator doesn't guarantee. These benchmarks
// don't need Eigen's own vectorization (the comparison is sparsemat's
// compile-time zero-elimination vs a plain dense loop, not vs hand-tuned
// SIMD), so vectorization/alignment is disabled to keep host and device
// storage layout simple and safe to cudaMemcpy as plain arrays.
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "sparsemat.h"
using SparseLinearAlgebra::SparseMat;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                                       \
      std::exit(1);                                                           \
    }                                                                          \
  } while (0)

namespace {

constexpr long long BATCH_SIZES[] = {100'000, 1'000'000};
constexpr int BLOCK = 256;
constexpr int WARMUP_LAUNCHES = 2;

// Deterministic per-instance "random-ish" fill: distinct instances get
// distinct (but reproducible) values without needing a device-side RNG.
double fill_value(long long instance, int slot) {
  return 1.0 + static_cast<double>((instance * 7 + slot * 3) % 11) * 0.37;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

struct Row {
  std::string config;
  long long n;
  double sparsemat_gpu_ns;
  double eigen_gpu_ns;  // NaN => "n/a" (Eigen has no device-callable path for this op)
  double sparsemat_cpu_ns;
};

std::vector<Row> results;

void print_results() {
  constexpr int W0 = 24;
  constexpr int W1 = 12;
  constexpr int W2 = 20;
  constexpr int W3 = 20;
  constexpr int W4 = 20;
  std::string sep(W0 + W1 + W2 + W3 + W4 + 4, '-');

  std::printf("\n%s\n", sep.c_str());
  std::printf("%-*s%-*s%*s%*s%*s\n", W0, "Configuration", W1, "Batch N", W2, "sparsemat GPU", W3,
              "Eigen dense GPU", W4, "sparsemat CPU");
  std::printf("%s\n", sep.c_str());

  std::string last_config;
  for (const auto& r : results) {
    if (r.config != last_config) {
      if (!last_config.empty()) {
        std::printf("\n");
      }
      last_config = r.config;
    }
    char eigen_buf[32];
    if (std::isnan(r.eigen_gpu_ns)) {
      std::snprintf(eigen_buf, sizeof(eigen_buf), "n/a");
    } else {
      std::snprintf(eigen_buf, sizeof(eigen_buf), "%.2f ns", r.eigen_gpu_ns);
    }
    std::printf("%-*s%-*lld%*.2f ns%*s%*.2f ns\n", W0, r.config.c_str(), W1, r.n, W2 - 3,
                r.sparsemat_gpu_ns, W3, eigen_buf, W4 - 3, r.sparsemat_cpu_ns);
  }
  std::printf("%s\n", sep.c_str());
}

// ---------------------------------------------------------------------------
// GPU timing helper: warms up, times the next launch with CUDA events,
// returns amortized ns/instance. Host-side chrono is deliberately not used
// here — kernel launches are asynchronous, so only cudaEvent timestamps
// recorded on the GPU's own timeline give an accurate kernel duration.
// ---------------------------------------------------------------------------

template<typename Kernel, typename... Args>
double time_gpu_kernel(Kernel kernel, long long n, Args... args) {
  int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);

  for (int i = 0; i < WARMUP_LAUNCHES; ++i) {
    kernel<<<grid, BLOCK>>>(args..., n);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  kernel<<<grid, BLOCK>>>(args..., n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return (static_cast<double>(elapsed_ms) * 1e6) / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// Generic multiply kernels/CPU-loop, templated on the matrix type. Each
// thread/iteration reads its two already-built operands straight out of the
// (device/host) array — no per-element gather from a separate raw buffer.
// ---------------------------------------------------------------------------

template<typename Mat>
__global__ void sparsemat_multiply_kernel(const Mat* a, const Mat* b, double* out, long long n) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  auto r = a[i].mult(b[i]);
  out[i] = r.values[0];
}

template<typename EigMat>
__global__ void eigen_multiply_kernel(const EigMat* a, const EigMat* b, double* out, long long n) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  EigMat r = a[i] * b[i];
  out[i] = r(0, 0);
}

template<typename Mat>
double sparsemat_cpu_multiply(const std::vector<Mat>& a, const std::vector<Mat>& b, long long n) {
  double sink = 0.0;
  auto start = std::chrono::steady_clock::now();
  for (long long i = 0; i < n; ++i) {
    auto r = a[i].mult(b[i]);
    sink += r.values[0];
  }
  auto end = std::chrono::steady_clock::now();
  std::printf("  (cpu sink=%f)\n", sink);
  return std::chrono::duration<double, std::nano>(end - start).count() / static_cast<double>(n);
}

// Builds n instances of Mat (sparse, nonZeroCount values each) and EigMat
// (dense, rows*cols values each) on the host, transfers both as arrays of
// already-typed objects, times sparsemat-GPU / Eigen-GPU / sparsemat-CPU,
// and frees device storage. Dense values cover the same fill_value()
// range/spread as the sparse ones — this is deliberately the "what if you
// just used a dense matrix" baseline, not a sparse-in-disguise one.
template<typename Mat, typename EigMat>
void bench_multiply(const std::string& cfg, long long n) {
  constexpr int nnz = Mat::nonZeroCount;
  constexpr int rows = EigMat::RowsAtCompileTime;
  constexpr int cols = EigMat::ColsAtCompileTime;

  std::vector<Mat> h_sparse_a(n);
  std::vector<Mat> h_sparse_b(n);
  for (long long i = 0; i < n; ++i) {
    for (int k = 0; k < nnz; ++k) {
      h_sparse_a[i].values[k] = fill_value(i, k);
      h_sparse_b[i].values[k] = fill_value(i, k + nnz);
    }
  }

  std::vector<EigMat> h_dense_a(n);
  std::vector<EigMat> h_dense_b(n);
  for (long long i = 0; i < n; ++i) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        h_dense_a[i](r, c) = fill_value(i, (r * cols) + c);
        h_dense_b[i](r, c) = fill_value(i, (r * cols) + c + (rows * cols));
      }
    }
  }

  Mat* d_sparse_a = nullptr;
  Mat* d_sparse_b = nullptr;
  double* d_sparse_out = nullptr;
  EigMat* d_dense_a = nullptr;
  EigMat* d_dense_b = nullptr;
  double* d_dense_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sparse_a, n * sizeof(Mat)));
  CUDA_CHECK(cudaMalloc(&d_sparse_b, n * sizeof(Mat)));
  CUDA_CHECK(cudaMalloc(&d_sparse_out, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dense_a, n * sizeof(EigMat)));
  CUDA_CHECK(cudaMalloc(&d_dense_b, n * sizeof(EigMat)));
  CUDA_CHECK(cudaMalloc(&d_dense_out, n * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_sparse_a, h_sparse_a.data(), n * sizeof(Mat), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sparse_b, h_sparse_b.data(), n * sizeof(Mat), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dense_a, h_dense_a.data(), n * sizeof(EigMat), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dense_b, h_dense_b.data(), n * sizeof(EigMat), cudaMemcpyHostToDevice));

  double sparsemat_gpu_ns =
      time_gpu_kernel(sparsemat_multiply_kernel<Mat>, n, d_sparse_a, d_sparse_b, d_sparse_out);
  double eigen_gpu_ns =
      time_gpu_kernel(eigen_multiply_kernel<EigMat>, n, d_dense_a, d_dense_b, d_dense_out);
  double sparsemat_cpu_ns = sparsemat_cpu_multiply(h_sparse_a, h_sparse_b, n);

  results.push_back({cfg, n, sparsemat_gpu_ns, eigen_gpu_ns, sparsemat_cpu_ns});

  CUDA_CHECK(cudaFree(d_sparse_a));
  CUDA_CHECK(cudaFree(d_sparse_b));
  CUDA_CHECK(cudaFree(d_sparse_out));
  CUDA_CHECK(cudaFree(d_dense_a));
  CUDA_CHECK(cudaFree(d_dense_b));
  CUDA_CHECK(cudaFree(d_dense_out));
}

// ---------------------------------------------------------------------------
// 5x5 SPD tridiagonal Cholesky solve (Ax=b, 1 RHS): sparsemat only.
// Diagonally dominant symmetric tridiagonal (diag=4, off-diag=-1), same
// pattern as bench.cpp's bench_5x5_solve(). Eigen's LLT is host-only, so
// there is no Eigen GPU column for this configuration. The system matrix A
// is identical for every instance (only the RHS varies), mirroring "many
// independent solves against a fixed system" — e.g. the batched-Kalman-
// filter use case this library was built for — so, same as the multiply
// benchmarks above, it's built once on the host and transferred as a single
// already-built object rather than reconstructed per thread/iteration
// inside the timed region.
// ---------------------------------------------------------------------------

using MatSPD = SparseMat<double, int, 5, 5, 0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24>;
using VecSPD = SparseMat<double, int, 5, 1, 0, 1, 2, 3, 4>;

__global__ void sparsemat_cholesky_solve_kernel(const MatSPD* a, const VecSPD* rhs, double* out,
                                                 long long n) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  auto r = SparseLinearAlgebra::cholesky_solve(*a, rhs[i]);
  out[i] = r.ok() ? r.value().values[0] : 0.0;
}

double sparsemat_cpu_cholesky_solve(const MatSPD& a, const std::vector<VecSPD>& rhs, long long n) {
  double sink = 0.0;
  auto start = std::chrono::steady_clock::now();
  for (long long i = 0; i < n; ++i) {
    auto r = SparseLinearAlgebra::cholesky_solve(a, rhs[i]);
    sink += r.ok() ? r.value().values[0] : 0.0;
  }
  auto end = std::chrono::steady_clock::now();
  std::printf("  (cpu sink=%f)\n", sink);
  return std::chrono::duration<double, std::nano>(end - start).count() / static_cast<double>(n);
}

void bench_cholesky_solve(long long n) {
  MatSPD h_a(4, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1, -1, 4);

  std::vector<VecSPD> h_rhs(n);
  for (long long i = 0; i < n; ++i) {
    for (int k = 0; k < 5; ++k) {
      h_rhs[i].values[k] = fill_value(i, k);
    }
  }

  MatSPD* d_a = nullptr;
  VecSPD* d_rhs = nullptr;
  double* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(MatSPD)));
  CUDA_CHECK(cudaMalloc(&d_rhs, n * sizeof(VecSPD)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(MatSPD), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rhs, h_rhs.data(), n * sizeof(VecSPD), cudaMemcpyHostToDevice));

  double sparsemat_gpu_ns = time_gpu_kernel(sparsemat_cholesky_solve_kernel, n, d_a, d_rhs, d_out);
  double sparsemat_cpu_ns = sparsemat_cpu_cholesky_solve(h_a, h_rhs, n);

  results.push_back(
      {"5x5 SPD cholesky solve", n, sparsemat_gpu_ns, std::numeric_limits<double>::quiet_NaN(),
       sparsemat_cpu_ns});

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_rhs));
  CUDA_CHECK(cudaFree(d_out));
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::printf("No CUDA device available (this binary already having built proves the "
                "kernels compile for the device); skipping GPU benchmark run.\n");
    return 0;
  }

  std::printf("sparsemat GPU throughput benchmark\n");
  std::printf("Batch sizes: ");
  for (long long n : BATCH_SIZES) {
    std::printf("%lld ", n);
  }
  std::printf("\nTimes are amortized nanoseconds per instance (kernel/loop time divided by batch "
              "size).\n");

  using Mat3 = SparseMat<double, int, 3, 3, 0, 4, 8>;
  using Mat5 = SparseMat<double, int, 5, 5, 0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24>;
  using Mat8 = SparseMat<double,
                         int,
                         8,
                         8,
                         0,
                         3,
                         5,
                         9,
                         14,
                         17,
                         20,
                         24,
                         29,
                         33,
                         38,
                         42,
                         47,
                         51,
                         58,
                         63>;
  using Eig3 = Eigen::Matrix<double, 3, 3>;
  using Eig5 = Eigen::Matrix<double, 5, 5>;
  using Eig8 = Eigen::Matrix<double, 8, 8>;

  for (long long n : BATCH_SIZES) {
    bench_multiply<Mat3, Eig3>("3x3 diagonal multiply", n);
    bench_multiply<Mat5, Eig5>("5x5 tridiagonal multiply", n);
    bench_multiply<Mat8, Eig8>("8x8 random sparse multiply", n);
    bench_cholesky_solve(n);
  }

  print_results();
  return 0;
}
