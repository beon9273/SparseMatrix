/**
 * Kalman filter for a projectile on a parabolic trajectory.
 *
 * State:       x = [px, py, vx, vy]^T
 * Transition:  px' = px + vx*dt,  py' = py + vy*dt,  vx' = vx,  vy' = vy - g*dt
 * Measurement: z = stacked [px, py]^T from each GPS sensor
 *
 * NumGPS is a compile-time parameter: all measurement matrices (H, R, z)
 * are sized and indexed at compile time via template meta-programming.
 */

#include "kalman.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>

constexpr std::size_t NumGPS = 10;             // <-- change to 2, 5, 10, etc.
constexpr double g = 9.81;                     // gravity in m/s²
constexpr double dt = 0.5;                     // time step in seconds
constexpr int N = 10;                          // number of time steps to simulate
constexpr double GPS_STD = 5.0;                // GPS noise standard deviation in meters
constexpr double GPS_VAR = GPS_STD * GPS_STD;  // GPS noise variance in meters²
// ---------------------------------------------------------------------------
// Normal random sample via Box-Muller.
// ---------------------------------------------------------------------------
double normal(std::mt19937& rng, double sigma) {
  std::uniform_real_distribution<double> u(1e-10, 1.0);
  double a = u(rng);
  double b = u(rng);
  return sigma * std::sqrt(-2.0 * std::log(a)) * std::cos(2.0 * M_PI * b);
}

template<std::size_t I, std::size_t NUM_GPS>
void set_measurements(typename Kalman::KalmanFilter<NUM_GPS>::Z_type& z,
                      std::mt19937& rng,
                      double true_px,
                      double true_py,
                      double gps_std) {
  if constexpr (I < NUM_GPS) {
    z.template set<2 * I, 0>(true_px + normal(rng, gps_std));
    z.template set<(2 * I) + 1, 0>(true_py + normal(rng, gps_std));
    set_measurements<I + 1, NUM_GPS>(z, rng, true_px, true_py, gps_std);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
  std::mt19937 rng(42);

  // Initial state: launch from origin, 50 m/s horizontal, 100 m/s vertical.
  Kalman::KalmanFilter<NumGPS>::State x0(0.0, 0.0, 50.0, 100.0);
  // Initial covariance: high position uncertainty, moderate velocity.
  auto P0 = SparseMat<double, 4, 4, 0, 5, 10, 15>(100.0, 100.0, 10.0, 10.0).dense();

  Kalman::KalmanFilter<NumGPS> kf({.x0 = x0, .P0 = P0, .dt = dt, .g = g});

  // Ground-truth integrator.
  double true_px = 0.0;
  double true_py = 0.0;
  double true_vx = 50.0;
  double true_vy = 100.0;

  std::cout << "step |  true_px   true_py  |  gps_px    gps_py   |  est_px    est_py\n";
  std::cout << "-----+---------------------+---------------------+-------------------\n";

  Kalman::KalmanFilter<NumGPS>::Z_type measurements{};

  for (int step = 1; step <= N; ++step) {
    // Ground-truth trajectory.
    true_px += true_vx * dt;
    true_py += true_vy * dt;
    true_vy -= g * dt;

    // Simulated GPS measurements — one noisy [px, py] per sensor.
    set_measurements<0, NumGPS>(measurements, rng, true_px, true_py, GPS_STD);
    kf.step(measurements, GPS_VAR);

    // Print the first sensor's GPS reading alongside the estimate.
    std::cout << "  " << step << "  | " << true_px << "  " << true_py << " | "
              << measurements.get(0, 0) << "  " << measurements.get(1, 0) << " | "
              << kf.x.get<0, 0>() << "  " << kf.x.get<1, 0>() << "\n";
  }

  return 0;
}
