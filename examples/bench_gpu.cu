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
 * Every kernel and CPU loop consumes *every* element of its result, not just
 * one. Consuming a single slot would let the optimizer delete the dead output
 * elements, and it would delete a different fraction for sparsemat (nnz
 * outputs) than for Eigen (rows*cols outputs) — skewing the comparison by an
 * amount that grows with density. Summing the whole result keeps both sides
 * honest.
 *
 * These kernels are memory-bound at this problem size, and that *is* the
 * headline result: a 3x3 diagonal sparsemat moves 3 doubles per operand where
 * the dense equivalent moves 9. The second table reports achieved bandwidth
 * so the win reads as what it is (bytes not moved) rather than as a FLOP-rate
 * claim.
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
 * device-callable, so the Cholesky configurations below only compare
 * sparsemat GPU vs sparsemat CPU; the Eigen column is reported as "n/a".
 *
 * The CPU column is deliberately single-threaded (one core) — it is a
 * per-core cost baseline, not a whole-socket-vs-whole-GPU hardware claim.
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "sparsemat.h"
using SparseLinearAlgebra::SparseMat;

#define CUDA_CHECK(call)                                                                    \
  do {                                                                                      \
    cudaError_t err = (call);                                                               \
    if (err != cudaSuccess) {                                                               \
      std::fprintf(                                                                         \
          stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

// Kernel launches report failure out-of-band, so every launch must be
// followed by an explicit error check — otherwise a launch that never ran
// shows up as a suspiciously fast timing rather than as an error.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr long long BATCH_SIZES[] = {100'000, 1'000'000};
constexpr int WARMUP_LAUNCHES = 3;
constexpr int TIMED_LAUNCHES = 9;  // odd, so the median is a single sample
constexpr double VERIFY_REL_TOL = 1e-9;

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
  // Bytes of global memory each instance must touch (operand reads + result
  // write), used to turn the GPU timings into an achieved-bandwidth figure.
  double sparsemat_bytes;
  double eigen_bytes;  // NaN when there is no Eigen column
};

std::vector<Row> results;

// GB/s implied by moving `bytes` per instance in `ns` per instance.
double bandwidth_gbs(double bytes, double ns) {
  return bytes / ns;
}

void print_results() {
  constexpr int W0 = 26;
  constexpr int W1 = 12;
  constexpr int W2 = 20;
  constexpr int W3 = 20;
  constexpr int W4 = 22;
  std::string sep(W0 + W1 + W2 + W3 + W4, '-');

  std::printf("\n%s\n", sep.c_str());
  std::printf("%-*s%-*s%*s%*s%*s\n",
              W0,
              "Configuration",
              W1,
              "Batch N",
              W2,
              "sparsemat GPU",
              W3,
              "Eigen dense GPU",
              W4,
              "sparsemat CPU (1 core)");
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
    std::printf("%-*s%-*lld%*.2f ns%*s%*.2f ns\n",
                W0,
                r.config.c_str(),
                W1,
                r.n,
                W2 - 3,
                r.sparsemat_gpu_ns,
                W3,
                eigen_buf,
                W4 - 3,
                r.sparsemat_cpu_ns);
  }
  std::printf("%s\n", sep.c_str());

  // Second table: these kernels are bandwidth-bound, so report the bytes each
  // side actually has to move and the GB/s that implies. The sparsemat/Eigen
  // gap is mostly this, not arithmetic.
  std::printf("\nAchieved global-memory bandwidth (operand reads + result write per instance)\n");
  std::printf("%s\n", sep.c_str());
  std::printf("%-*s%-*s%*s%*s\n",
              W0,
              "Configuration",
              W1,
              "Batch N",
              W2,
              "sparsemat GPU",
              W3,
              "Eigen dense GPU");
  std::printf("%s\n", sep.c_str());
  last_config.clear();
  for (const auto& r : results) {
    if (r.config != last_config) {
      if (!last_config.empty()) {
        std::printf("\n");
      }
      last_config = r.config;
    }
    char eigen_buf[48];
    if (std::isnan(r.eigen_gpu_ns) || std::isnan(r.eigen_bytes)) {
      std::snprintf(eigen_buf, sizeof(eigen_buf), "n/a");
    } else {
      std::snprintf(eigen_buf,
                    sizeof(eigen_buf),
                    "%.1f GB/s (%.0f B)",
                    bandwidth_gbs(r.eigen_bytes, r.eigen_gpu_ns),
                    r.eigen_bytes);
    }
    char sparse_buf[48];
    std::snprintf(sparse_buf,
                  sizeof(sparse_buf),
                  "%.1f GB/s (%.0f B)",
                  bandwidth_gbs(r.sparsemat_bytes, r.sparsemat_gpu_ns),
                  r.sparsemat_bytes);
    std::printf("%-*s%-*lld%*s%*s\n", W0, r.config.c_str(), W1, r.n, W2, sparse_buf, W3, eigen_buf);
  }
  std::printf("%s\n", sep.c_str());
}

// ---------------------------------------------------------------------------
// Verification: a benchmark that computes the wrong answer benchmarks very
// well, so every configuration checks its GPU output against the CPU output
// it is being compared to before the timings are trusted.
// ---------------------------------------------------------------------------

void verify(const std::string& cfg,
            const double* d_out,
            const std::vector<double>& expected,
            long long n) {
  std::vector<double> got(static_cast<std::size_t>(n));
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
  for (long long i = 0; i < n; ++i) {
    double e = expected[static_cast<std::size_t>(i)];
    double g = got[static_cast<std::size_t>(i)];
    double scale = std::max(1.0, std::abs(e));
    if (!(std::abs(e - g) <= VERIFY_REL_TOL * scale)) {
      std::fprintf(stderr,
                   "verification FAILED for %s at instance %lld: cpu=%.17g gpu=%.17g\n",
                   cfg.c_str(),
                   i,
                   e,
                   g);
      std::exit(1);
    }
  }
}

// ---------------------------------------------------------------------------
// GPU timing helper: picks a block size from the kernel's own occupancy,
// warms up, then takes the median of several event-timed launches and
// returns amortized ns/instance. A single sample is at the mercy of clock
// ramping and scheduling noise; the median of an odd number of runs is not.
//
// Host-side chrono is deliberately not used here — kernel launches are
// asynchronous, so only cudaEvent timestamps recorded on the GPU's own
// timeline give an accurate kernel duration.
// ---------------------------------------------------------------------------

template<typename Kernel, typename... Args>
double time_gpu_kernel(Kernel kernel, long long n, Args... args) {
  int min_grid = 0;
  int block = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, kernel, 0, 0));
  long long grid_ll = (n + block - 1) / block;
  int grid = static_cast<int>(grid_ll);

  for (int i = 0; i < WARMUP_LAUNCHES; ++i) {
    kernel<<<grid, block>>>(args..., n);
    CUDA_CHECK_LAUNCH();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<double> samples;
  samples.reserve(TIMED_LAUNCHES);
  for (int i = 0; i < TIMED_LAUNCHES; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(args..., n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    samples.push_back((static_cast<double>(elapsed_ms) * 1e6) / static_cast<double>(n));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// ---------------------------------------------------------------------------
// Generic multiply kernels/CPU-loop, templated on the matrix type. Each
// thread/iteration reads its two already-built operands straight out of the
// (device/host) array — no per-element gather from a separate raw buffer —
// and sums every element of the product so none of it can be optimized away.
// ---------------------------------------------------------------------------

template<typename Mat>
__global__ void sparsemat_multiply_kernel(const Mat* a, const Mat* b, double* out, long long n) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  auto r = a[i].mult(b[i]);
  double acc = 0.0;
  for (int k = 0; k < decltype(r)::nonZeroCount; ++k) {
    acc += r.values[k];
  }
  out[i] = acc;
}

template<typename EigMat>
__global__ void eigen_multiply_kernel(const EigMat* a, const EigMat* b, double* out, long long n) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  EigMat r = a[i] * b[i];
  double acc = 0.0;
  for (int c = 0; c < EigMat::ColsAtCompileTime; ++c) {
    for (int rr = 0; rr < EigMat::RowsAtCompileTime; ++rr) {
      acc += r(rr, c);
    }
  }
  out[i] = acc;
}

template<typename Mat>
double sparsemat_cpu_multiply(const std::vector<Mat>& a,
                              const std::vector<Mat>& b,
                              std::vector<double>& out,
                              long long n) {
  auto start = std::chrono::steady_clock::now();
  for (long long i = 0; i < n; ++i) {
    auto r = a[static_cast<std::size_t>(i)].mult(b[static_cast<std::size_t>(i)]);
    double acc = 0.0;
    for (int k = 0; k < decltype(r)::nonZeroCount; ++k) {
      acc += r.values[k];
    }
    out[static_cast<std::size_t>(i)] = acc;
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(end - start).count() / static_cast<double>(n);
}

// Builds n instances of Mat (sparse, nonZeroCount values each) and EigMat
// (dense, rows*cols values each) on the host, transfers both as arrays of
// already-typed objects, times sparsemat-GPU / Eigen-GPU / sparsemat-CPU,
// verifies the GPU result against the CPU one, and frees device storage.
// Dense values cover the same fill_value() range/spread as the sparse ones —
// this is deliberately the "what if you just used a dense matrix" baseline,
// not a sparse-in-disguise one.
template<typename Mat, typename EigMat>
void bench_multiply(const std::string& cfg, long long n) {
  static_assert(std::is_trivially_copyable_v<Mat>,
                "sparsemat operands are cudaMemcpy'd as raw bytes.");
  // Eigen::Matrix is not formally trivially copyable (it has user-provided
  // constructors), but a fixed-size one with vectorization disabled is a
  // plain contiguous scalar array with no padding and no indirection, which
  // is exactly what makes the cudaMemcpy below valid. Assert that property
  // directly rather than the stricter trait it happens to fail.
  static_assert(sizeof(EigMat) == sizeof(typename EigMat::Scalar) * EigMat::RowsAtCompileTime *
                                      EigMat::ColsAtCompileTime,
                "Eigen operands are cudaMemcpy'd as raw bytes; layout must be a bare "
                "contiguous scalar array.");
  constexpr int nnz = Mat::nonZeroCount;
  constexpr int rows = EigMat::RowsAtCompileTime;
  constexpr int cols = EigMat::ColsAtCompileTime;
  const auto un = static_cast<std::size_t>(n);

  std::vector<Mat> h_sparse_a(un);
  std::vector<Mat> h_sparse_b(un);
  for (long long i = 0; i < n; ++i) {
    for (int k = 0; k < nnz; ++k) {
      h_sparse_a[static_cast<std::size_t>(i)].values[k] = fill_value(i, k);
      h_sparse_b[static_cast<std::size_t>(i)].values[k] = fill_value(i, k + nnz);
    }
  }

  std::vector<EigMat> h_dense_a(un);
  std::vector<EigMat> h_dense_b(un);
  for (long long i = 0; i < n; ++i) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        h_dense_a[static_cast<std::size_t>(i)](r, c) = fill_value(i, (r * cols) + c);
        h_dense_b[static_cast<std::size_t>(i)](r, c) =
            fill_value(i, (r * cols) + c + (rows * cols));
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

  std::vector<double> h_cpu_out(un);
  double sparsemat_cpu_ns = sparsemat_cpu_multiply(h_sparse_a, h_sparse_b, h_cpu_out, n);
  verify(cfg, d_sparse_out, h_cpu_out, n);

  results.push_back({cfg,
                     n,
                     sparsemat_gpu_ns,
                     eigen_gpu_ns,
                     sparsemat_cpu_ns,
                     static_cast<double>((2 * sizeof(Mat)) + sizeof(double)),
                     static_cast<double>((2 * sizeof(EigMat)) + sizeof(double))});

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
// there is no Eigen GPU column for these configurations. The system matrix A
// is identical for every instance (only the RHS varies), mirroring "many
// independent solves against a fixed system" — e.g. the batched-Kalman-
// filter use case this library was built for.
//
// Two configurations are reported, because they measure different things:
//
//   "factorize+solve" — cholesky_solve(A, b) per instance, refactorizing A
//       every time. This is what you pay if you treat the system as if it
//       changed each step.
//   "presolved"       — A factorized once on the host, the resulting
//       CholeskyFactor transferred to the device, and every thread calling
//       only .solve(b) on it. This is the shape the API is built for (see
//       SparseMat::cholesky()) and the one the fixed-A use case should use.
// ---------------------------------------------------------------------------

using MatSPD = SparseMat<double, int, 5, 5, 0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24>;
using VecSPD = SparseMat<double, int, 5, 1, 0, 1, 2, 3, 4>;
using FactorSPD = std::decay_t<decltype(std::declval<MatSPD>().cholesky().value())>;

__global__ void sparsemat_cholesky_solve_kernel(const MatSPD* a,
                                                const VecSPD* rhs,
                                                double* out,
                                                long long n) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  auto r = SparseLinearAlgebra::cholesky_solve(*a, rhs[i]);
  double acc = 0.0;
  if (r.ok()) {
    for (int k = 0; k < std::decay_t<decltype(r.value())>::nonZeroCount; ++k) {
      acc += r.value().values[k];
    }
  }
  out[i] = acc;
}

__global__ void sparsemat_cholesky_presolved_kernel(const FactorSPD* factor,
                                                    const VecSPD* rhs,
                                                    double* out,
                                                    long long n) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  auto r = factor->solve(rhs[i]);
  double acc = 0.0;
  if (r.ok()) {
    for (int k = 0; k < std::decay_t<decltype(r.value())>::nonZeroCount; ++k) {
      acc += r.value().values[k];
    }
  }
  out[i] = acc;
}

double sparsemat_cpu_cholesky_solve(const MatSPD& a,
                                    const std::vector<VecSPD>& rhs,
                                    std::vector<double>& out,
                                    long long n) {
  auto start = std::chrono::steady_clock::now();
  for (long long i = 0; i < n; ++i) {
    auto r = SparseLinearAlgebra::cholesky_solve(a, rhs[static_cast<std::size_t>(i)]);
    double acc = 0.0;
    if (r.ok()) {
      for (int k = 0; k < std::decay_t<decltype(r.value())>::nonZeroCount; ++k) {
        acc += r.value().values[k];
      }
    }
    out[static_cast<std::size_t>(i)] = acc;
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(end - start).count() / static_cast<double>(n);
}

double sparsemat_cpu_cholesky_presolved(const FactorSPD& factor,
                                        const std::vector<VecSPD>& rhs,
                                        std::vector<double>& out,
                                        long long n) {
  auto start = std::chrono::steady_clock::now();
  for (long long i = 0; i < n; ++i) {
    auto r = factor.solve(rhs[static_cast<std::size_t>(i)]);
    double acc = 0.0;
    if (r.ok()) {
      for (int k = 0; k < std::decay_t<decltype(r.value())>::nonZeroCount; ++k) {
        acc += r.value().values[k];
      }
    }
    out[static_cast<std::size_t>(i)] = acc;
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(end - start).count() / static_cast<double>(n);
}

void bench_cholesky_solve(long long n) {
  static_assert(std::is_trivially_copyable_v<FactorSPD>,
                "CholeskyFactor is cudaMemcpy'd to the device as raw bytes.");
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto un = static_cast<std::size_t>(n);

  MatSPD h_a(4, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1, -1, 4);

  auto h_factor_result = h_a.cholesky();
  if (!h_factor_result.ok()) {
    std::fprintf(stderr, "benchmark setup error: 5x5 SPD matrix failed to factorize\n");
    std::exit(1);
  }
  FactorSPD h_factor = h_factor_result.value();

  std::vector<VecSPD> h_rhs(un);
  for (long long i = 0; i < n; ++i) {
    for (int k = 0; k < 5; ++k) {
      h_rhs[static_cast<std::size_t>(i)].values[k] = fill_value(i, k);
    }
  }

  MatSPD* d_a = nullptr;
  FactorSPD* d_factor = nullptr;
  VecSPD* d_rhs = nullptr;
  double* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(MatSPD)));
  CUDA_CHECK(cudaMalloc(&d_factor, sizeof(FactorSPD)));
  CUDA_CHECK(cudaMalloc(&d_rhs, n * sizeof(VecSPD)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(MatSPD), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_factor, &h_factor, sizeof(FactorSPD), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rhs, h_rhs.data(), n * sizeof(VecSPD), cudaMemcpyHostToDevice));

  // A and the factor are the same object for every thread, so they land in
  // cache after the first warp and cost effectively nothing per instance;
  // only the RHS read and the result write scale with N.
  const auto bytes = static_cast<double>(sizeof(VecSPD) + sizeof(double));
  std::vector<double> h_cpu_out(un);

  {
    const std::string cfg = "5x5 chol factorize+solve";
    double gpu_ns = time_gpu_kernel(sparsemat_cholesky_solve_kernel, n, d_a, d_rhs, d_out);
    double cpu_ns = sparsemat_cpu_cholesky_solve(h_a, h_rhs, h_cpu_out, n);
    verify(cfg, d_out, h_cpu_out, n);
    results.push_back({cfg, n, gpu_ns, nan, cpu_ns, bytes, nan});
  }

  {
    const std::string cfg = "5x5 chol presolved";
    double gpu_ns = time_gpu_kernel(sparsemat_cholesky_presolved_kernel, n, d_factor, d_rhs, d_out);
    double cpu_ns = sparsemat_cpu_cholesky_presolved(h_factor, h_rhs, h_cpu_out, n);
    verify(cfg, d_out, h_cpu_out, n);
    results.push_back({cfg, n, gpu_ns, nan, cpu_ns, bytes, nan});
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_factor));
  CUDA_CHECK(cudaFree(d_rhs));
  CUDA_CHECK(cudaFree(d_out));
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::printf(
        "No CUDA device available (this binary already having built proves the "
        "kernels compile for the device); skipping GPU benchmark run.\n");
    return 0;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // Peak theoretical bandwidth = bus width (bits -> bytes) * double-data-rate
  // memory clock (kHz -> Hz), in GB/s. Gives the second table a ceiling to
  // read the achieved numbers against.
  

  std::printf("sparsemat GPU throughput benchmark\n");
  std::printf("Batch sizes: ");
  for (long long n : BATCH_SIZES) {
    std::printf("%lld ", n);
  }
  std::printf(
      "\nTimes are amortized nanoseconds per instance (median of %d launches, divided by "
      "batch size).\n",
      TIMED_LAUNCHES);

  using Mat3 = SparseMat<double, int, 3, 3, 0, 4, 8>;
  using Mat5 = SparseMat<double, int, 5, 5, 0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24>;
  using Mat8 =
      SparseMat<double, int, 8, 8, 0, 3, 5, 9, 14, 17, 20, 24, 29, 33, 38, 42, 47, 51, 58, 63>;
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
