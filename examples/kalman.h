#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "sparsemat.h"

using namespace SparseLinearAlgebra;

namespace Kalman {

// ---------------------------------------------------------------------------
// Compile-time index generators for variable-size measurement matrices
// ---------------------------------------------------------------------------

// H is 2*NumGPS × 4.  GPS sensor i observes px (col 0) on row 2i and
// py (col 1) on row 2i+1.  Flat indices in a 4-column matrix:
//   (2i,   0) → 8i
//   (2i+1, 1) → 8i + 5
template<int NumGPS>
consteval auto h_indices() {
  std::array<int, 2 * NumGPS> arr{};
  for (int i = 0; i < NumGPS; ++i) {
    arr[2 * i] = 8 * i;
    arr[(2 * i) + 1] = (8 * i) + 5;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// KalmanFilter<NumGPS>
//
// State:       x = [px, py, vx, vy]^T  (4×1)
// Measurement: one [px, py]^T per GPS sensor, stacked into a 2*NumGPS × 1 z
// ---------------------------------------------------------------------------
template<int NumGPS = 1>
struct KalmanFilter {
  static constexpr int M = 2 * NumGPS;  // measurement dimension

  // ---- Compile-time sparsity arrays --------------------------------------
  static constexpr auto H_arr = h_indices<NumGPS>();

  // ---- Matrix types -------------------------------------------------------

  // Measurement matrix H (M × 4) and its transpose HT (4 × M).
  using H_type = decltype(SparseLinearAlgebra::MatrixUtilities<SparseMat<double, int, M, 4>>::
                              template make<M, 4, H_arr>(std::make_index_sequence<M>{}));

  // Measurement noise covariance R (M × M, diagonal).
  using R_type = decltype(identity<double, int, M, M>());

  // Stacked measurement vector z (M × 1, dense).
  using Z_type = decltype(dense<double, int, M, 1>());

  // State vector x (4 × 1, dense) and covariance P (4 × 4, dense).
  // P is always fully populated after the first update so we fix it to the
  // dense type up front.
  using State = decltype(dense<double, int, 4, 1>());
  using Cov = decltype(dense<double, int, 4, 4>());

  // ---- State members ------------------------------------------------------
  State x;
  Cov P;
  double dt;  // time step
  double g;   // gravity
  bool ok;    // Indicates whether the filter is in a valid state

  // State-transition matrix F (4 × 4).
  SparseMat<double, int, 4, 4, 0, 2, 5, 7, 10, 15> F;

  // Process-noise covariance Q (4 × 4, diagonal).
  SparseMat<double, int, 4, 4, 0, 5, 10, 15> Q;

  // Gravity correction applied to vy slot (index 3).
  SparseMat<double, int, 4, 1, 3> gravity;

  struct KalmanInput {
    State x0;
    Cov P0;
    double dt{};
    double g{};
  };

  SPARSEMAT_HD KalmanFilter(KalmanInput input)
      : x(input.x0),
        P(input.P0),
        dt(input.dt),
        g(input.g),
        ok(true),
        F(1.0, dt, 1.0, dt, 1.0, 1.0),
        Q(0.1, 0.1, 1.0, 1.0),
        gravity(-g * dt) {}

  // ---- Predict step -------------------------------------------------------
  // Advances x and P by one time step dt.  Gravity is subtracted from vy.
  SPARSEMAT_HD [[nodiscard]] bool step(const Z_type& measurements, double gps_var) {
    ok = predict();
    ok = update(measurements, gps_var);
    return ok;
  }

  SPARSEMAT_HD [[nodiscard]] bool predict() {
    if (ok) {
      x = F.mult(x).add(gravity).dense();
      P = F.mult(P).mult(F.transpose()).add(Q).dense();
    }
    return ok;
  }

  // ---- Update step --------------------------------------------------------
  // Accepts stacked measurements from all NumGPS sensors.
  // measurements[i] = {px_i, py_i} for sensor i.
  // gps_var = GPS noise variance (sigma^2), assumed identical across sensors.
  SPARSEMAT_HD [[nodiscard]] bool update(const Z_type& measurements, double gps_var) {
    if (!ok)
      return ok;

    // Build H and R.
    H_type H{};
    H.values.fill(1.0);
    R_type R{};
    R.values.fill(gps_var);

    auto HT = H.transpose();
    auto PHT = P.mult(HT);  // 4 × M
    auto HP = H.mult(P);    // M × 4

    // Innovation.
    auto innovation = measurements.subtract(H.mult(x));

    // innovation covariance S = H*P*H^T + R  (M × M). cholesky() reports
    // failure via Result::ok() rather than throwing (this runs in CUDA
    // device code); S is SPD by construction here as long as R is, so the
    // factorization should succeed.
    auto cholesky_result = HP.mult(HT).add(R).cholesky();
    ok = cholesky_result.ok();
    if (!ok)
      return ok;

    auto cholesky = cholesky_result.value();

    auto innovation_solution = cholesky.solve(innovation);
    ok = innovation_solution.ok();
    if (!ok) {
      return ok;
    }

    auto hp_solution = cholesky.solve(HP);
    ok = hp_solution.ok();
    if (!ok)
      return ok;

    // State and covariance update.
    x = x.add(PHT.mult(innovation_solution.value())).dense();
    P = P.subtract(PHT.mult(hp_solution.value())).dense();
    return ok;
  }
};

}  // namespace Kalman
