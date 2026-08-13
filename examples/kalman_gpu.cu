/**
 * Batched GPU Kalman filter demo.
 *
 * Runs N independent instances of Kalman::KalmanFilter<1> (one GPS sensor
 * each) forward in time on the GPU, one CUDA thread per filter. Every filter
 * tracks the same projectile trajectory but receives independently-noised
 * measurements, so this is a stand-in for "N independent tracked objects."
 *
 * The sparsity-aware SparseMat operations (mult/add/transpose/cholesky/
 * triangular solves) that KalmanFilter::step() relies on are compiled as
 * __host__ __device__ via the SPARSEMAT_HD macro, so the exact same header
 * code that runs on the CPU (examples/kalman.cpp) also runs unmodified on
 * the device here.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>

#include "kalman.h"
using KF = Kalman::KalmanFilter<1>;

constexpr double G = 9.81;
constexpr double DT = 0.5;
constexpr double GPS_STD = 5.0;
constexpr double GPS_VAR = GPS_STD * GPS_STD;
constexpr int STEPS = 50;

#define CUDA_CHECK(call)                                                                    \
  do {                                                                                      \
    cudaError_t err = (call);                                                               \
    if (err != cudaSuccess) {                                                               \
      std::fprintf(                                                                         \
          stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

// One thread per filter: advances it one predict+update step using this
// step's noisy measurement.
__global__ void kalman_step_kernel(KF* filters,
                                   const double* meas_px,
                                   const double* meas_py,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  KF::Z_type z(meas_px[i], meas_py[i]);
  static_cast<void>(filters[i].step(z, GPS_VAR));
}

double true_px(double t) {
  return 50.0 * t;
}
double true_py(double t) {
  return (100.0 * t) - (0.5 * G * t * t);
}

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;
  std::printf("Running %d batched Kalman filters for %d steps on the GPU...\n", n, STEPS);

  // Every filter starts from the same launch conditions and covariance.
  KF::State x0(0.0, 0.0, 50.0, 100.0);
  auto P0 = SparseMat<double, int, 4, 4, 0, 5, 10, 15>(100.0, 100.0, 10.0, 10.0).dense();
  KF host_template({.x0 = x0, .P0 = P0, .dt = DT, .g = G});

  std::vector<KF> host_filters(n, host_template);
  KF* device_filters = nullptr;
  double* device_px = nullptr;
  double* device_py = nullptr;
  CUDA_CHECK(cudaMalloc(&device_filters, n * sizeof(KF)));
  CUDA_CHECK(cudaMalloc(&device_px, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&device_py, n * sizeof(double)));
  CUDA_CHECK(
      cudaMemcpy(device_filters, host_filters.data(), n * sizeof(KF), cudaMemcpyHostToDevice));

  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0.0, GPS_STD);
  std::vector<double> px(n);
  std::vector<double> py(n);

  const int block = 256;
  const int grid = (n + block - 1) / block;

  auto start = std::chrono::steady_clock::now();
  for (int step = 1; step <= STEPS; ++step) {
    double t = step * DT;
    double tpx = true_px(t);
    double tpy = true_py(t);
    for (int i = 0; i < n; ++i) {
      px[i] = tpx + noise(rng);
      py[i] = tpy + noise(rng);
    }
    CUDA_CHECK(cudaMemcpy(device_px, px.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_py, py.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Launch the kernel to perform one step of the Kalman filter for all threads.
    // Thread divergence occurs if some threads have filters that fail while others succeed,
    // causing different execution paths within the kernel. A finite state machine manager
    // with state buffers for each thread could be used to track the state of each thread
    // and ensure consistent execution paths.
    kalman_step_kernel<<<grid, block>>>(device_filters, device_px, device_py, n);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();

  CUDA_CHECK(
      cudaMemcpy(host_filters.data(), device_filters, n * sizeof(KF), cudaMemcpyDeviceToHost));

  bool all_ok = true;
  for (auto& filter : host_filters) {
    if (!filter.ok) {
      all_ok = false;
      break;
    }
  }
  std::printf("All filters ok: %d\n", all_ok);

  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::printf("Done in %.2f ms (%.1f filter-steps/sec)\n",
              elapsed_ms,
              (double(n) * STEPS) / (elapsed_ms / 1000.0));

  double final_t = STEPS * DT;
  std::printf("True final position:      (%.2f, %.2f)\n", true_px(final_t), true_py(final_t));
  std::printf("Filter 0 estimated state:  px=%.2f py=%.2f vx=%.2f vy=%.2f\n",
              host_filters[0].x.values[0],
              host_filters[0].x.values[1],
              host_filters[0].x.values[2],
              host_filters[0].x.values[3]);
  std::printf("Filter %d estimated state:  px=%.2f py=%.2f vx=%.2f vy=%.2f\n",
              n - 1,
              host_filters[n - 1].x.values[0],
              host_filters[n - 1].x.values[1],
              host_filters[n - 1].x.values[2],
              host_filters[n - 1].x.values[3]);

  CUDA_CHECK(cudaFree(device_filters));
  CUDA_CHECK(cudaFree(device_px));
  CUDA_CHECK(cudaFree(device_py));
  return 0;
}
