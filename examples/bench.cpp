#include <array>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "sparsemat.h"
using SparseLinearAlgebra::SparseMat;
// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

static volatile double sink = 0.0;  // prevents dead-code elimination

static constexpr int WARMUP = 50'000;
static constexpr int ITERS = 500'000;

template<typename Fn>
double time_ns(const Fn& fn) {
  for (int i = 0; i < WARMUP; ++i) {
    fn();
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < ITERS; ++i) {
    fn();
  }
  auto end = std::chrono::high_resolution_clock::now();
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) /
         ITERS;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

struct Row {
  std::string config;
  std::string op;
  double sparse_ns;
  double eigen_sparse_ns;
  double eigen_dense_ns;
};

static std::vector<Row> results;

void print_results() {
  constexpr int W0 = 30;
  constexpr int W1 = 20;
  constexpr int W2 = 14;
  constexpr int W3 = 14;
  constexpr int W4 = 14;
  std::string sep(W0 + W1 + W2 + W3 + W4 + 10, '-');

  std::cout << "\n" << sep << "\n";
  std::cout << std::left << std::setw(W0) << "Configuration" << std::setw(W1) << "Operation"
            << std::right << std::setw(W2) << "sparsemat" << std::setw(W3) << "Eigen sparse"
            << std::setw(W4) << "Eigen dense" << "\n";
  std::cout << sep << "\n";

  std::string last_config{};
  for (const auto& r : results) {
    if (r.config != last_config) {
      if (!last_config.empty()) {
        std::cout << "\n";
      }
      last_config = r.config;
    }
    std::cout << std::left << std::setw(W0) << r.config << std::setw(W1) << r.op << std::right
              << std::fixed << std::setprecision(2) << std::setw(W2) << r.sparse_ns << " ns"
              << std::setw(W3) << r.eigen_sparse_ns << " ns" << std::setw(W4) << r.eigen_dense_ns
              << " ns" << "\n";
  }
  std::cout << sep << "\n";
}

// ---------------------------------------------------------------------------
// Eigen helpers
// ---------------------------------------------------------------------------

// Build an Eigen::SparseMatrix from flat non-zero indices
template<int Rows, int Cols>
Eigen::SparseMatrix<double> make_eigen_sparse(const std::vector<int>& flat_indices,
                                              double value = 1.0) {
  Eigen::SparseMatrix<double> m(Rows, Cols);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(flat_indices.size());
  for (int idx : flat_indices) {
    triplets.emplace_back(idx / Cols, idx % Cols, value);
  }
  m.setFromTriplets(triplets.begin(), triplets.end());
  m.makeCompressed();
  return m;
}

template<int N>
Eigen::Matrix<double, N, N> make_eigen_dense(const std::vector<int>& flat_indices,
                                             double value = 1.0) {
  Eigen::Matrix<double, N, N> m = Eigen::Matrix<double, N, N>::Zero();
  for (int idx : flat_indices) {
    m(idx / N, idx % N) = value;
  }
  return m;
}

// ---------------------------------------------------------------------------
// Benchmark runner — one configuration, one operation
// ---------------------------------------------------------------------------

void bench(const std::string& config,
           const std::string& op,
           double sparse_ns,
           double eigen_sparse_ns,
           double eigen_dense_ns) {
  results.push_back({config, op, sparse_ns, eigen_sparse_ns, eigen_dense_ns});
}

// ---------------------------------------------------------------------------
// Configurations
// ---------------------------------------------------------------------------

// --- 3x3 diagonal: NZ at {0,4,8} ---
void bench_3x3_diagonal() {
  const std::string cfg = "3x3 diagonal (33%)";
  const std::vector<int> nz = {0, 4, 8};

  SparseMat<double, int, 3, 3, 0, 4, 8> A(1, 2, 3);
  SparseMat<double, int, 3, 3, 0, 4, 8> B(4, 5, 6);
  auto ES_A = make_eigen_sparse<3, 3>(nz, 1.0);
  auto ES_B = make_eigen_sparse<3, 3>(nz, 2.0);
  auto ED_A = make_eigen_dense<3>(nz, 1.0);
  auto ED_B = make_eigen_dense<3>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "transpose",
        time_ns([&] {
          auto r = A.transpose();
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A.transpose();
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A.transpose();
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 3x3 first-row only: NZ at {0,1,2} ---
void bench_3x3_first_row() {
  const std::string cfg = "3x3 first-row (33%)";
  const std::vector<int> nz = {0, 1, 2};

  SparseMat<double, int, 3, 3, 0, 1, 2> A(1, 2, 3);
  SparseMat<double, int, 3, 3, 0, 1, 2> B(4, 5, 6);
  auto ES_A = make_eigen_sparse<3, 3>(nz, 1.0);
  auto ES_B = make_eigen_sparse<3, 3>(nz, 2.0);
  auto ED_A = make_eigen_dense<3>(nz, 1.0);
  auto ED_B = make_eigen_dense<3>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 3x3 full: all 9 NZ ---
void bench_3x3_full() {
  const std::string cfg = "3x3 full (100%)";
  const std::vector<int> nz = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  SparseMat<double, int, 3, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8> A(1, 2, 3, 4, 5, 6, 7, 8, 9);
  SparseMat<double, int, 3, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8> B(9, 8, 7, 6, 5, 4, 3, 2, 1);
  auto ES_A = make_eigen_sparse<3, 3>(nz, 1.0);
  auto ES_B = make_eigen_sparse<3, 3>(nz, 2.0);
  auto ED_A = make_eigen_dense<3>(nz, 1.0);
  auto ED_B = make_eigen_dense<3>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 5x5 diagonal: NZ at {0,6,12,18,24} ---
void bench_5x5_diagonal() {
  const std::string cfg = "5x5 diagonal (20%)";
  const std::vector<int> nz = {0, 6, 12, 18, 24};

  SparseMat<double, int, 5, 5, 0, 6, 12, 18, 24> A(1, 2, 3, 4, 5);
  SparseMat<double, int, 5, 5, 0, 6, 12, 18, 24> B(5, 4, 3, 2, 1);
  auto ES_A = make_eigen_sparse<5, 5>(nz, 1.0);
  auto ES_B = make_eigen_sparse<5, 5>(nz, 2.0);
  auto ED_A = make_eigen_dense<5>(nz, 1.0);
  auto ED_B = make_eigen_dense<5>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "transpose",
        time_ns([&] {
          auto r = A.transpose();
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A.transpose();
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A.transpose();
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 5x5 tridiagonal banded: NZ at main + upper + lower diagonals ---
// Indices: {0,1,5,6,7,11,12,13,17,18,19,23,24} = 13 NZ / 25 = 52%
void bench_5x5_tridiagonal() {
  const std::string cfg = "5x5 tridiagonal (52%)";
  const std::vector<int> nz = {0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24};

  SparseMat<double, int, 5, 5, 0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24> A(
      1, 1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1, 5);
  SparseMat<double, int, 5, 5, 0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24> B(
      2, 2, 2, 3, 2, 2, 4, 2, 2, 5, 2, 2, 6);
  auto ES_A = make_eigen_sparse<5, 5>(nz, 1.0);
  auto ES_B = make_eigen_sparse<5, 5>(nz, 2.0);
  auto ED_A = make_eigen_dense<5>(nz, 1.0);
  auto ED_B = make_eigen_dense<5>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 5x5 random sparse ~24%: 6 NZ scattered ---
// Indices: {1,3,7,12,16,22}
void bench_5x5_random_sparse() {
  const std::string cfg = "5x5 random sparse (24%)";
  const std::vector<int> nz = {1, 3, 7, 12, 16, 22};

  SparseMat<double, int, 5, 5, 1, 3, 7, 12, 16, 22> A(1, 2, 3, 4, 5, 6);
  SparseMat<double, int, 5, 5, 1, 3, 7, 12, 16, 22> B(6, 5, 4, 3, 2, 1);
  auto ES_A = make_eigen_sparse<5, 5>(nz, 1.0);
  auto ES_B = make_eigen_sparse<5, 5>(nz, 2.0);
  auto ED_A = make_eigen_dense<5>(nz, 1.0);
  auto ED_B = make_eigen_dense<5>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 5x5 random dense ~76%: 19 NZ ---
// Indices: all except {2,8,15,20,23} (5 zeros)
void bench_5x5_dense() {
  const std::string cfg = "5x5 dense-ish (76%)";
  const std::vector<int> nz = {0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22};

  SparseMat<double, int, 5, 5, 0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22>
      A(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);
  SparseMat<double, int, 5, 5, 0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22>
      B(19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
  auto ES_A = make_eigen_sparse<5, 5>(nz, 1.0);
  auto ES_B = make_eigen_sparse<5, 5>(nz, 2.0);
  auto ED_A = make_eigen_dense<5>(nz, 1.0);
  auto ED_B = make_eigen_dense<5>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 8x8 diagonal: NZ at {0,9,18,27,36,45,54,63} ---
void bench_8x8_diagonal() {
  const std::string cfg = "8x8 diagonal (12%)";
  const std::vector<int> nz = {0, 9, 18, 27, 36, 45, 54, 63};

  SparseMat<double, int, 8, 8, 0, 9, 18, 27, 36, 45, 54, 63> A(1, 2, 3, 4, 5, 6, 7, 8);
  SparseMat<double, int, 8, 8, 0, 9, 18, 27, 36, 45, 54, 63> B(8, 7, 6, 5, 4, 3, 2, 1);
  auto ES_A = make_eigen_sparse<8, 8>(nz, 1.0);
  auto ES_B = make_eigen_sparse<8, 8>(nz, 2.0);
  auto ED_A = make_eigen_dense<8>(nz, 1.0);
  auto ED_B = make_eigen_dense<8>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "transpose",
        time_ns([&] {
          auto r = A.transpose();
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A.transpose();
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A.transpose();
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 8x8 tridiagonal banded ---
// Main diag + 1 above + 1 below: {0,1,8,9,10,17,18,19,26,27,28,35,36,37,44,45,46,53,54,55,62,63}
void bench_8x8_tridiagonal() {
  const std::string cfg = "8x8 tridiagonal (34%)";
  const std::vector<int> nz = {0,  1,  8,  9,  10, 17, 18, 19, 26, 27, 28,
                               35, 36, 37, 44, 45, 46, 53, 54, 55, 62, 63};

  SparseMat<double,
            int,
            8,
            8,
            0,
            1,
            8,
            9,
            10,
            17,
            18,
            19,
            26,
            27,
            28,
            35,
            36,
            37,
            44,
            45,
            46,
            53,
            54,
            55,
            62,
            63>
      A(1, 1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1, 5, 1, 1, 6, 1, 1, 7, 1, 1, 8);
  SparseMat<double,
            int,
            8,
            8,
            0,
            1,
            8,
            9,
            10,
            17,
            18,
            19,
            26,
            27,
            28,
            35,
            36,
            37,
            44,
            45,
            46,
            53,
            54,
            55,
            62,
            63>
      B(2, 2, 2, 3, 2, 2, 4, 2, 2, 5, 2, 2, 6, 2, 2, 7, 2, 2, 8, 2, 2, 9);
  auto ES_A = make_eigen_sparse<8, 8>(nz, 1.0);
  auto ES_B = make_eigen_sparse<8, 8>(nz, 2.0);
  auto ED_A = make_eigen_dense<8>(nz, 1.0);
  auto ED_B = make_eigen_dense<8>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 8x8 random sparse ~25%: 16 NZ ---
// Indices: {0,3,5,9,14,17,20,24,29,33,38,42,47,51,58,63}
void bench_8x8_random_sparse() {
  const std::string cfg = "8x8 random sparse (25%)";
  const std::vector<int> nz = {0, 3, 5, 9, 14, 17, 20, 24, 29, 33, 38, 42, 47, 51, 58, 63};

  SparseMat<double, int, 8, 8, 0, 3, 5, 9, 14, 17, 20, 24, 29, 33, 38, 42, 47, 51, 58, 63> A(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  SparseMat<double, int, 8, 8, 0, 3, 5, 9, 14, 17, 20, 24, 29, 33, 38, 42, 47, 51, 58, 63> B(
      16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
  auto ES_A = make_eigen_sparse<8, 8>(nz, 1.0);
  auto ES_B = make_eigen_sparse<8, 8>(nz, 2.0);
  auto ED_A = make_eigen_dense<8>(nz, 1.0);
  auto ED_B = make_eigen_dense<8>(nz, 2.0);

  bench(cfg,
        "multiply",
        time_ns([&] {
          auto r = A.mult(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A * ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A * ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "add",
        time_ns([&] {
          auto r = A.add(B);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = ES_A + ES_B;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = ED_A + ED_B;
          sink += r(0, 0);
        }));

  bench(cfg,
        "scale",
        time_ns([&] {
          auto r = A.scale(2.0);
          sink += r.values[0];
        }),
        time_ns([&] {
          Eigen::SparseMatrix<double> r = 2.0 * ES_A;
          sink += r.coeff(0, 0);
        }),
        time_ns([&] {
          auto r = 2.0 * ED_A;
          sink += r(0, 0);
        }));

  bench(cfg,
        "frobenius",
        time_ns([&] { sink += A.frobenius(); }),
        time_ns([&] { sink += ES_A.norm(); }),
        time_ns([&] { sink += ED_A.norm(); }));
}

// --- 5x5 SPD tridiagonal: solve Ax=b and AX=B via Cholesky and LU ---
// Diagonally dominant symmetric tridiagonal matrix (diag=4, off-diag=-1),
// same {0,1,5,6,7,11,12,13,17,18,19,23,24} sparsity pattern as
// bench_5x5_tridiagonal, but with values chosen to make it symmetric
// positive definite so that both Cholesky and LU are valid factorizations
// of the same system. Each timed lambda re-factorizes from scratch (matching
// what SparseLinearAlgebra::cholesky_solve/lu_solve and the Eigen solver
// constructors do), so this measures factorize+solve together, not solve
// alone.
void bench_5x5_solve() {
  const std::string cfg = "5x5 SPD tridiagonal";

  SparseMat<double, int, 5, 5, 0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24> A(
      4, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1, -1, 4);

  // Single right-hand side (Ax = b).
  SparseMat<double, int, 5, 1, 0, 1, 2, 3, 4> b(1, 2, 3, 4, 5);

  // Block right-hand side with 3 columns (AX = B).
  SparseMat<double, int, 5, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14> B(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  Eigen::SparseMatrix<double> ES_A(5, 5);
  {
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < 5; ++i) {
      triplets.emplace_back(i, i, 4.0);
      if (i + 1 < 5) {
        triplets.emplace_back(i, i + 1, -1.0);
        triplets.emplace_back(i + 1, i, -1.0);
      }
    }
    ES_A.setFromTriplets(triplets.begin(), triplets.end());
    ES_A.makeCompressed();
  }
  Eigen::Matrix<double, 5, 5> ED_A = Eigen::Matrix<double, 5, 5>(ES_A);

  Eigen::VectorXd ev_b(5);
  ev_b << 1, 2, 3, 4, 5;

  Eigen::MatrixXd ev_B(5, 3);
  ev_B << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;

  bench(cfg,
        "cholesky (1 rhs)",
        time_ns([&] {
          auto r = SparseLinearAlgebra::cholesky_solve(A, b);
          sink += r.value().values[0];
        }),
        time_ns([&] {
          Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(ES_A);
          Eigen::VectorXd x = solver.solve(ev_b);
          sink += x(0);
        }),
        time_ns([&] {
          Eigen::VectorXd x = ED_A.llt().solve(ev_b);
          sink += x(0);
        }));

  bench(cfg,
        "lu (1 rhs)",
        time_ns([&] {
          auto r = SparseLinearAlgebra::lu_solve(A, b);
          sink += r.value().values[0];
        }),
        time_ns([&] {
          Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(ES_A);
          Eigen::VectorXd x = solver.solve(ev_b);
          sink += x(0);
        }),
        time_ns([&] {
          Eigen::VectorXd x = ED_A.partialPivLu().solve(ev_b);
          sink += x(0);
        }));

  bench(cfg,
        "cholesky (3 rhs)",
        time_ns([&] {
          auto r = SparseLinearAlgebra::cholesky_solve(A, B);
          sink += r.value().values[0];
        }),
        time_ns([&] {
          Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(ES_A);
          Eigen::MatrixXd X = solver.solve(ev_B);
          sink += X(0, 0);
        }),
        time_ns([&] {
          Eigen::MatrixXd X = ED_A.llt().solve(ev_B);
          sink += X(0, 0);
        }));

  bench(cfg,
        "lu (3 rhs)",
        time_ns([&] {
          auto r = SparseLinearAlgebra::lu_solve(A, B);
          sink += r.value().values[0];
        }),
        time_ns([&] {
          Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(ES_A);
          Eigen::MatrixXd X = solver.solve(ev_B);
          sink += X(0, 0);
        }),
        time_ns([&] {
          Eigen::MatrixXd X = ED_A.partialPivLu().solve(ev_B);
          sink += X(0, 0);
        }));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
  std::cout << "sparsemat benchmark\n";
  std::cout << "Iterations: " << ITERS << " (warmup: " << WARMUP << ")\n";
  std::cout << "Timing in nanoseconds per operation\n";

  std::cout << "\nRunning 3x3 configurations...";
  std::cout.flush();
  bench_3x3_diagonal();
  bench_3x3_first_row();
  bench_3x3_full();

  std::cout << " done\nRunning 5x5 configurations...";
  std::cout.flush();
  bench_5x5_diagonal();
  bench_5x5_tridiagonal();
  bench_5x5_random_sparse();
  bench_5x5_dense();

  std::cout << " done\nRunning 8x8 configurations...";
  std::cout.flush();
  bench_8x8_diagonal();
  bench_8x8_tridiagonal();
  bench_8x8_random_sparse();

  std::cout << " done\nRunning solve configurations...";
  std::cout.flush();
  bench_5x5_solve();

  std::cout << " done\n";

  print_results();

  std::cout << "\n(sink=" << sink << " — prevents dead-code elimination)\n";
  return 0;
}
