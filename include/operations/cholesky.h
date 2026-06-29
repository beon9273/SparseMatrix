#pragma once

#include <cmath>
#include <cstddef>

#include "concepts/concepts.h"
#include "operations/diagonal.h"
#include "operations/triangular.h"

namespace {

/**
 * @brief Compile-time symbolic Cholesky fill computation for a square sparse matrix.
 *
 * Determines which positions in the lower-triangular factor L are structurally
 * non-zero after Cholesky factorization, without pivoting.
 *
 * Fill propagation rule: a Cholesky outer-product update at step k creates
 * fill at (i, j) whenever both L[i][k] and L[j][k] are non-zero (i ≥ j > k).
 * Iterating k from 0 to N-1 propagates all reachable fill in one pass.
 *
 * @tparam SparseMat Square symmetric positive-definite input matrix type.
 */
template<typename SparseMat>
class CholeskySparsity {
  static_assert(SparseMat::rows == SparseMat::cols,
                "Cholesky factorization requires a square matrix.");
  static constexpr std::size_t N = SparseMat::rows;

  static constexpr auto compute_fill() {
    std::array<std::array<bool, N>, N> fill{};
    // Seed with the lower-triangular non-zeros of A.
    for (auto idx : SparseMat::indices) {
      std::size_t row = idx / SparseMat::cols;
      std::size_t col = idx % SparseMat::cols;
      if (row >= col) {
        fill[row][col] = true;
}
    }
    // Propagate: outer-product update at step k creates fill at (i,j)
    // when both column-k entries are non-zero (i >= j > k).
    for (std::size_t k = 0; k < N; ++k) {
      for (std::size_t i = k + 1; i < N; ++i) {
        if (!fill[i][k]) {
          continue;
}
        for (std::size_t j = k + 1; j <= i; ++j) {
          if (fill[j][k]) {
            fill[i][j] = true;
}
        }
      }
    }
    return fill;
  }

 public:
  static constexpr auto fill = compute_fill();

  /**
   * @brief Returns @c true if L[row][col] is structurally non-zero.
   *
   * Diagonal entries are always stored (they hold the Cholesky pivot values).
   * Upper-triangle entries are always zero.  Sub-diagonal entries are governed
   * by the fill array computed above.
   */
  static constexpr bool l_nonzero(std::size_t row, std::size_t col) {
    if (col > row) {
      return false;  // upper triangle
}
    if (row == col) {
      return true;  // diagonal always stored
}
    return fill[row][col];
  }
};

// ---------------------------------------------------------------------------

/**
 * @brief Result-type helper for the L factor of a Cholesky factorization.
 *
 * Satisfies the @c OperationUtilities interface so that @c calculate_sparsity()
 * and @c num_nonzeros() produce the correct packed flat-index array for
 * constructing the L @c SparseMat type via @c make_result().
 *
 * @tparam SparseMat Input matrix type whose sparsity drives the symbolic fill.
 */
template<typename SparseMat>
class LCholeskyMatrix {
 public:
  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat::cols;

  constexpr static bool is_result_index_nonzero(std::size_t row, std::size_t col) {
    return CholeskySparsity<SparseMat>::l_nonzero(row, col);
  }

  static constexpr std::size_t numNonzeros =
      SparseLinearAlgebra::OperationUtilities<LCholeskyMatrix>::num_nonzeros();
  static constexpr auto sparsity =
      SparseLinearAlgebra::OperationUtilities<LCholeskyMatrix>::calculate_sparsity();

  static auto make_result() {
    return SparseMat::template make<rows, cols, sparsity>(std::make_index_sequence<numNonzeros>{});
  }
};

// ---------------------------------------------------------------------------

/**
 * @brief Numeric Cholesky factorization for a square sparse matrix.
 *
 * Computes the lower-triangular factor L such that A = L * L^T.
 * The algorithm (column j, left-looking):
 * @code
 *   L[j][j] = sqrt(A[j][j] - sum_{m<j} L[j][m]^2)
 *   L[i][j] = (A[i][j] - sum_{m<j} L[i][m]*L[j][m]) / L[j][j]   for i > j
 * @endcode
 *
 * All loops are unrolled at compile time via template recursion.  No pivoting
 * is performed; the matrix must be symmetric positive definite.
 *
 * @tparam SparseMat Square SPD input matrix type.
 */
template<typename SparseMat>
class CholeskyFactorization {
  static_assert(SparseMat::rows == SparseMat::cols,
                "Cholesky factorization requires a square matrix.");
  static constexpr std::size_t N = SparseMat::rows;
  using DataType = typename SparseMat::DataType;
  using MU = SparseLinearAlgebra::MatrixUtilities<SparseMat>;

  /**
   * Computes sum_{m=0}^{J-1} L[I][m] * L[J][m].
   * Used for both diagonal (I==J) and sub-diagonal (I>J) entries.
   */
  template<typename L, std::size_t I, std::size_t J, std::size_t M = 0>
  static DataType inner_sum(const L& l) {
    if constexpr (M < J) {
      constexpr int lim = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, M);
      constexpr int ljm = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(J, M);
      auto term = DataType(0);
      if constexpr (lim >= 0 && ljm >= 0) {
        term = l.values[lim] * l.values[ljm];
      }
      return term + inner_sum<L, I, J, M + 1>(l);
    } else {
      return DataType(0);
    }
  }

  /**
   * Fills column J of L: diagonal first (I==J), then sub-diagonal rows (I>J).
   */
  template<typename L, std::size_t J, std::size_t I = J>
  static void compute_column(const SparseMat& a, L& l) {
    if constexpr (I >= N) {
      return;
    } else if constexpr (I == J) {
      // Diagonal: L[J][J] = sqrt(A[J][J] - sum_{m<J} L[J][m]^2)
      constexpr int l_jj = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(J, J);
      static_assert(l_jj >= 0, "Cholesky: diagonal of L must be structurally non-zero.");
      DataType sum = inner_sum<L, J, J>(l);
      constexpr int a_jj = MU::getSparseIndex(J, J);
      DataType a_val = (a_jj >= 0) ? a.values[a_jj] : DataType(0);
      l.values[l_jj] = std::sqrt(a_val - sum);
      compute_column<L, J, J + 1>(a, l);
    } else {
      // Sub-diagonal: L[I][J] = (A[I][J] - sum_{m<J} L[I][m]*L[J][m]) / L[J][J]
      constexpr int l_ij = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, J);
      if constexpr (l_ij >= 0) {
        DataType sum = inner_sum<L, I, J>(l);
        constexpr int a_ij = MU::getSparseIndex(I, J);
        DataType a_val = (a_ij >= 0) ? a.values[a_ij] : DataType(0);
        constexpr int l_jj = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(J, J);
        l.values[l_ij] = (a_val - sum) / l.values[l_jj];
      }
      compute_column<L, J, I + 1>(a, l);
    }
  }

  template<typename L, std::size_t J = 0>
  static void outer_loop(const SparseMat& a, L& l) {
    if constexpr (J < N) {
      compute_column<L, J>(a, l);
      outer_loop<L, J + 1>(a, l);
    }
  }

 public:
  template<typename L>
  static void factorize(const SparseMat& a, L& l) {
    outer_loop(a, l);
  }
};

}  // namespace

namespace SparseLinearAlgebra {

/**
 * @brief Handle returned by @c SparseMat::cholesky(), holding the L factor.
 *
 * Exposes a @c solve() method that reuses the pre-computed L to solve one or
 * more right-hand sides without re-factorizing.  The RHS may be a column
 * vector (single solve) or a matrix with multiple columns (block solve).
 *
 * @tparam L_Type Type of the lower-triangular Cholesky factor.
 */
template<SparseMatrixType L_Type>
class CholeskyFactor {
  L_Type l_;

 public:
  explicit CholeskyFactor(L_Type l) : l_(std::move(l)) {}

  /**
   * @brief Solves (L * L^T) * X = RHS for X.
   *
   * Runs forward substitution (L * Y = RHS) followed by back substitution
   * (L^T * X = Y).  Works for both column-vector and block (multi-column) RHS.
   *
   * @param rhs Right-hand side — one or more column vectors.
   * @return    Solution X with the same column count as @p rhs.
   */
  template<SparseMatrixType RHS>
  [[nodiscard]] auto solve(const RHS& rhs) const {
    auto y = forward_solve(l_, rhs);
    return backward_solve(l_.transpose(), y);
  }
};

/**
 * @brief Performs the Cholesky factorization A = L * L^T.
 *
 * Computes the lower-triangular factor L at compile-time sparsity and fills
 * its values at runtime.  The input matrix must be symmetric positive definite.
 * No pivoting is performed.
 *
 * @tparam SparseMat Square SPD input matrix type satisfying @c SparseMatrixType.
 * @param  A         Input matrix.
 * @return           Lower-triangular L such that A = L * L^T.
 */
template<SparseMatrixType SparseMat>
auto cholesky_factorize(const SparseMat& A) {
  auto l = LCholeskyMatrix<SparseMat>::make_result();
  CholeskyFactorization<SparseMat>::factorize(A, l);
  return l;
}

/**
 * @brief Solves the linear system Ax = b via Cholesky factorization.
 *
 * Factorizes A = L * L^T at compile-time sparsity, then solves in two steps:
 *   1. Forward substitution:   L * y   = b
 *   2. Backward substitution:  L^T * x = y
 *
 * No pivoting is performed.  The matrix must be symmetric positive definite.
 * To reuse a pre-factored L across multiple right-hand sides, call
 * @c cholesky_factorize then use @c forward_solve / @c backward_solve directly.
 *
 * @param A Square SPD input matrix.
 * @param b Right-hand side column vector.
 * @return  Solution vector x such that Ax = b.
 */
template<SparseMatrixType SparseMat, SparseMatrixType RHS>
auto cholesky_solve(const SparseMat& A, const RHS& b) {
  auto l = cholesky_factorize(A);
  auto y = forward_solve(l, b);
  return backward_solve(l.transpose(), y);
}

template<SparseMatrixType SparseMat>
struct Cholesky {
  decltype(cholesky_factorize(std::declval<SparseMat>())) lower;

  Cholesky(const SparseMat& a) : lower(cholesky_factorize(a)) { }

  template<SparseMatrixType RHS>
  auto solve(const RHS& b) {
    auto y = forward_solve(lower, b);
    return backward_solve(lower.transpose(), y);
  };
};

}  // namespace SparseLinearAlgebra
