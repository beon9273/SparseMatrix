#pragma once

#include <cstddef>

#include "concepts/concepts.h"
#include "operations/diagonal.h"
#include "operations/triangular.h"
namespace {

/**
 * @brief Compile-time symbolic LU fill computation for a square sparse matrix.
 *
 * Computes which positions in L and U can be non-zero after Doolittle
 * factorization, without pivoting.  The fill rule is: position (i, j) fills
 * in if A[i][j] is non-zero, OR any elimination step k < min(i,j) connects
 * row i to column j through existing fill.
 *
 * Positions in L below the diagonal, and in U on and above the diagonal, are
 * read from the resulting fill array.  L has a unit diagonal (always stored).
 *
 * @tparam SparseMat Square input matrix type.
 */
template<typename SparseMat>
class LUSparsity {
  static_assert(SparseMat::rows == SparseMat::cols, "LU factorization requires a square matrix.");
  static constexpr std::size_t N = SparseMat::rows;

  // Initialise fill from A's structural non-zeros, then propagate: for each
  // pivot column k, any row i with fill[i][k] inherits all of row k's fill.
  static constexpr auto compute_fill() {
    std::array<std::array<bool, N>, N> fill{};
    for (auto idx : SparseMat::indices) {
      fill[idx / SparseMat::cols][idx % SparseMat::cols] = true;
    }
    for (std::size_t k = 0; k < N; ++k) {
      for (std::size_t i = k + 1; i < N; ++i) {
        if (!fill[i][k])
          continue;
        for (std::size_t j = k; j < N; ++j) {
          if (fill[k][j])
            fill[i][j] = true;
        }
      }
    }
    return fill;
  }

 public:
  static constexpr auto fill = compute_fill();

  /// Returns @c true if L[row][col] is structurally non-zero.
  /// L has a unit diagonal, so diagonal positions always return @c true.
  static constexpr bool l_nonzero(std::size_t row, std::size_t col) {
    if (row == col)
      return true;
    if (col > row)
      return false;
    return fill[row][col];
  }

  /// Returns @c true if U[row][col] is structurally non-zero.
  static constexpr bool u_nonzero(std::size_t row, std::size_t col) {
    if (col < row)
      return false;
    return fill[row][col];
  }
};

/**
 * @brief Result-type helper for the L factor of an LU factorization.
 *
 * Satisfies the @c OperationUtilities interface so that @c calculate_sparsity()
 * and @c num_nonzeros() produce the correct packed flat-index array for
 * constructing the L @c SparseMat type via @c make_result().
 *
 * @tparam SparseMat Input matrix type whose sparsity drives the symbolic fill.
 */
template<typename SparseMat>
class LMatrix {
 public:
  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat::cols;

  constexpr static bool is_result_index_nonzero(std::size_t row, std::size_t col) {
    return LUSparsity<SparseMat>::l_nonzero(row, col);
  }

  static constexpr std::size_t numNonzeros =
      SparseLinearAlgebra::OperationUtilities<LMatrix>::num_nonzeros();
  static constexpr auto sparsity =
      SparseLinearAlgebra::OperationUtilities<LMatrix>::calculate_sparsity();

  static auto make_result() {
    return SparseMat::template make<rows, cols, sparsity>(std::make_index_sequence<numNonzeros>{});
  }
};

/**
 * @brief Result-type helper for the U factor of an LU factorization.
 *
 * @tparam SparseMat Input matrix type whose sparsity drives the symbolic fill.
 */
template<typename SparseMat>
class UMatrix {
 public:
  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat::cols;

  constexpr static bool is_result_index_nonzero(std::size_t row, std::size_t col) {
    return LUSparsity<SparseMat>::u_nonzero(row, col);
  }

  static constexpr std::size_t numNonzeros =
      SparseLinearAlgebra::OperationUtilities<UMatrix>::num_nonzeros();
  static constexpr auto sparsity =
      SparseLinearAlgebra::OperationUtilities<UMatrix>::calculate_sparsity();

  static auto make_result() {
    return SparseMat::template make<rows, cols, sparsity>(std::make_index_sequence<numNonzeros>{});
  }
};

/**
 * @brief Numeric Doolittle LU factorization for a square sparse matrix.
 *
 * Given a pre-allocated L and U (from @c LMatrix and @c UMatrix), fills in
 * their values in-place.  L has a unit diagonal (stored as 1).  No pivoting
 * is performed; the factorization is only numerically stable for diagonally
 * dominant or otherwise pivot-free matrices.
 *
 * At each step k:
 *   U[k][j] = A[k][j] - sum(L[k][m] * U[m][j], m < k)   for j >= k
 *   L[i][k] = (A[i][k] - sum(L[i][m] * U[m][k], m < k)) / U[k][k]   for i > k
 *
 * @tparam SparseMat Input matrix type.
 */
template<typename SparseMat>
class LUFactorization {
  static_assert(SparseMat::rows == SparseMat::cols, "LU factorization requires a square matrix.");
  static constexpr std::size_t N = SparseMat::rows;
  using DataType = typename SparseMat::DataType;
  using MU = SparseLinearAlgebra::MatrixUtilities<SparseMat>;

  template<typename L, typename U, std::size_t I, std::size_t K, std::size_t M = 0>
  static auto inner_loop_inner(const SparseMat& a, L& l, U& u) {
    if constexpr (M < K) {
      constexpr int lim = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, M);
      constexpr int umk = SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(M, K);
      DataType sum = DataType(0);
      if constexpr (lim >= 0 && umk >= 0) {
        sum += l.values[lim] * u.values[umk];
      }
      return sum + inner_loop_inner<L, U, I, K, M + 1>(a, l, u);
    } else {
      return DataType(0);
    }
  }

  template<typename L, typename U, std::size_t K, std::size_t J = K>
  static void inner_loop_1(const SparseMat& a, L& l, U& u) {
    if constexpr (J < N) {
      constexpr int u_idx = SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(K, J);
      if constexpr (u_idx >= 0) {
        DataType sum = inner_loop_inner<L, U, K, K, 0>(a, l, u);
        constexpr int a_idx = MU::getSparseIndex(K, J);
        if constexpr (a_idx >= 0) {
          u.values[u_idx] = a.values[a_idx] - sum;
        } else {
          u.values[u_idx] = -sum;
        }
      }
      inner_loop_1<L, U, K, J + 1>(a, l, u);
    }
  }

  template<typename L, typename U, std::size_t K, std::size_t I = K + 1>
  static void inner_loop_2(const SparseMat& a, L& l, U& u, DataType pivot) {
    if constexpr (I < N) {
      constexpr int l_idx = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, K);
      if constexpr (l_idx >= 0) {
        DataType sum = inner_loop_inner<L, U, I, K, 0>(a, l, u);
        constexpr int a_idx = MU::getSparseIndex(I, K);
        if constexpr (a_idx < 0) {
          l.values[l_idx] = -sum / pivot;
        } else {
          l.values[l_idx] = (a.values[a_idx] - sum) / pivot;
        }
      }
      inner_loop_2<L, U, K, I + 1>(a, l, u, pivot);
    }
  }

  template<typename L, typename U, std::size_t K>
  static void outer_loop_over_rows(const SparseMat& a, L& l, U& u) {
    if constexpr (K < N) {
      inner_loop_1<L, U, K>(a, l, u);

      // Get the pivot. Might blow up if the pivot is structurally zero, but that's a user error.
      constexpr int u_kk = SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(K, K);
      DataType pivot = DataType(0);
      if constexpr (u_kk >= 0) {
        pivot = u.values[u_kk];
      }

      inner_loop_2<L, U, K>(a, l, u, pivot);
      outer_loop_over_rows<L, U, K + 1>(a, l, u);
    }
  }

 public:
  template<typename L, typename U>
  static void factorize(const SparseMat& a, L& l, U& u) {
    SparseLinearAlgebra::set_diagonal<L>(l, DataType(1));
    outer_loop_over_rows<L, U, 0>(a, l, u);
  }
};

}  // namespace

namespace SparseLinearAlgebra {

/**
 * @brief Performs the LU factorization.
 *
 * Given a square sparse matrix A, computes L and U such that A = LU.
 * L is lower triangular with a unit diagonal, and U is upper triangular.
 * The sparsity of L and U is determined at compile time, and the numeric
 * factorization is performed at runtime.  No pivoting is performed; the
 * factorization is only numerically stable for diagonally dominant or
 * otherwise pivot-free matrices.
 */
template<SparseMatrixType SparseMat>
auto lu_factorize(const SparseMat& A) {
  auto l = LMatrix<SparseMat>::make_result();
  auto u = UMatrix<SparseMat>::make_result();
  LUFactorization<SparseMat>::factorize(A, l, u);
  return std::make_pair(l, u);
}

/**
 * @brief Solves a triangular system given pre-factored L and U matrices.
 *
 * Performs two triangular solves against the right-hand side:
 *   1. Forward substitution:  Ly = RHS
 *   2. Back substitution:     Ux = y
 *
 * Use this when L and U have already been computed via @c lu_factorize and
 * need to be reused for multiple right-hand sides.
 *
 * @param l Lower-triangular factor with unit diagonal.
 * @param u Upper-triangular factor.
 * @param RHS Right-hand side matrix or column vector.
 * @return Solution x such that LUx = RHS.
 */
template<SparseMatrixType SparseMat, SparseMatrixType L, SparseMatrixType U>
auto lu_solve(L& l, U& u, const SparseMat& RHS) {
  auto y = forward_solve(l, RHS);
  return backward_solve(u, y);
}

/**
 * @brief Solves the linear system Ax = b via LU factorization.
 *
 * Computes the sparsity of L and U at compile time, performs numeric
 * Doolittle factorization at runtime, then solves in two triangular steps:
 *   1. Forward substitution:  Ly = b
 *   2. Back substitution:     Ux = y
 *
 * No pivoting is performed.  The matrix must be non-singular and the
 * factorization must be stable without row swaps (e.g. diagonally dominant).
 *
 * @param A Square input matrix.
 * @param b Right-hand side column vector.
 * @return  Solution vector x such that Ax = b.
 */
template<SparseMatrixType SparseMat, SparseMatrixType RHS>
auto lu_solve(const SparseMat& A, const RHS& b) {
  auto lu = lu_factorize(A);
  return lu_solve(lu.first, lu.second, b);
}

}  // namespace SparseLinearAlgebra
