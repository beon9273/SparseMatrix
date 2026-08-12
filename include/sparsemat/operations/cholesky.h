#pragma once

#include <cmath>
#include <cstddef>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/diagonal.h"
#include "sparsemat/operations/result.h"
#include "sparsemat/operations/triangular.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

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
  using Int = typename SparseMat::Int;
  static_assert(SparseMat::rows == SparseMat::cols,
                "Cholesky factorization requires a square matrix.");
  static constexpr auto N = SparseMat::rows;

  static constexpr auto compute_fill() {
    std::array<std::array<bool, N>, N> fill{};  // N is of type SparseMat::Int
    // Seed with the lower-triangular non-zeros of A.
    for (auto idx : SparseMat::indices()) {
      Int row = idx / SparseMat::cols;
      Int col = idx % SparseMat::cols;
      if (row >= col) {
        fill[row][col] = true;
      }
    }
    // Propagate: outer-product update at step k creates fill at (i,j)
    // when both column-k entries are non-zero (i >= j > k).
    for (Int k = 0; k < N; ++k) {
      for (Int i = k + 1; i < N; ++i) {
        if (!fill[i][k]) {
          continue;
        }
        for (Int j = k + 1; j <= i; ++j) {
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
  SPARSEMAT_HD static constexpr bool l_nonzero(Int row, Int col) {
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
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;

  SPARSEMAT_HD constexpr static bool is_result_index_nonzero(Int row, Int col) {
    return CholeskySparsity<SparseMat>::l_nonzero(row, col);
  }

  static constexpr auto numNonzeros =
      SparseLinearAlgebra::OperationUtilities<LCholeskyMatrix>::num_nonzeros();
  static constexpr auto sparsity =
      SparseLinearAlgebra::OperationUtilities<LCholeskyMatrix>::calculate_sparsity();

  SPARSEMAT_HD static auto make_result() {
    return SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
        std::make_index_sequence<static_cast<std::size_t>(numNonzeros)>{});
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
  static constexpr auto N = SparseMat::rows;
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  using MU = SparseLinearAlgebra::MatrixUtilities<SparseMat>;

  /// Single term of sum_{m=0}^{J-1} L[I][m] * L[J][m].
  template<typename L, Int I, Int J, std::size_t M>
  SPARSEMAT_HD static DataType inner_sum_term(const L& l) {
    constexpr auto lim =
        SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, static_cast<Int>(M));
    constexpr auto ljm =
        SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(J, static_cast<Int>(M));
    if constexpr (lim >= 0 && ljm >= 0) {
      return l.values[lim] * l.values[ljm];
    } else {
      return DataType(0);
    }
  }
  template<typename L, Int I, Int J, std::size_t... Ms>
  SPARSEMAT_HD static DataType inner_sum_fold(const L& l, std::index_sequence<Ms...> /*seq*/) {
    return (inner_sum_term<L, I, J, Ms>(l) + ...);
  }
  /**
   * Computes sum_{m=0}^{J-1} L[I][m] * L[J][m], used for both diagonal
   * (I==J) and sub-diagonal (I>J) entries. (pack + ...) is ill-formed for an
   * empty pack, so J==0 (the first column, nothing yet to sum) needs an
   * explicit early-out.
   */
  template<typename L, Int I, Int J>
  SPARSEMAT_HD static DataType inner_sum(const L& l) {
    if constexpr (J == 0) {
      return DataType(0);
    } else {
      return inner_sum_fold<L, I, J>(l, std::make_index_sequence<static_cast<std::size_t>(J)>{});
    }
  }

  /// Diagonal entry of column J: L[J][J] = sqrt(A[J][J] - sum_{m<J} L[J][m]^2).
  /// Must run before any sub-diagonal entry in the same column, since those
  /// divide by L[J][J].
  template<typename L, Int J>
  SPARSEMAT_HD static void diag_step(const SparseMat& a, L& l, bool& ok) {
    constexpr auto l_jj = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(J, J);
    static_assert(l_jj >= 0, "Cholesky: diagonal of L must be structurally non-zero.");
    DataType sum = inner_sum<L, J, J>(l);
    constexpr auto a_jj = MU::getSparseIndex(J, J);
    DataType a_val = (a_jj >= 0) ? a.values[a_jj] : DataType(0);
    DataType diag = a_val - sum;
    // Non-positive diag means A is not (numerically) positive definite. The
    // threshold also rejects a diag that is positive but negligible relative
    // to the A[J][J] it came from: sqrt() of it produces a pivot the
    // sub-diagonal entries then divide by, so accepting it yields garbage
    // with ok() == true. See singular_pivot_threshold().
    const DataType a_mag = a_val < DataType(0) ? -a_val : a_val;
    if (diag <= singular_pivot_threshold<DataType>() * a_mag) {
      ok = false;
      l.values[l_jj] = DataType(0);
      return;
    }
    l.values[l_jj] = std::sqrt(diag);
  }

  /// Sub-diagonal entry (I,J): L[I][J] = (A[I][J] - sum_{m<J} L[I][m]*L[J][m]) / L[J][J].
  template<typename L, Int J, std::size_t IOff>
  SPARSEMAT_HD static void subdiag_step(const SparseMat& a, L& l, bool& ok) {
    constexpr Int I = J + 1 + static_cast<Int>(IOff);
    constexpr auto l_ij = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, J);
    if constexpr (l_ij >= 0) {
      DataType sum = inner_sum<L, I, J>(l);
      constexpr auto a_ij = MU::getSparseIndex(I, J);
      DataType a_val = (a_ij >= 0) ? a.values[a_ij] : DataType(0);
      constexpr auto l_jj = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(J, J);
      if (l.values[l_jj] == DataType(0)) {
        ok = false;
        l.values[l_ij] = DataType(0);
        return;
      }
      l.values[l_ij] = (a_val - sum) / l.values[l_jj];
    }
  }
  // Fills all sub-diagonal rows I in [J+1, N) via a comma fold. Row order
  // doesn't matter here (each I writes an independent L cell) — only the
  // diagonal-before-subdiagonal ordering above is load-bearing.
  template<typename L, Int J, std::size_t... IOffs>
  SPARSEMAT_HD static void compute_subdiag(const SparseMat& a,
                                           L& l,
                                           bool& ok,
                                           std::index_sequence<IOffs...> /*seq*/) {
    (subdiag_step<L, J, IOffs>(a, l, ok), ...);
  }

  /// Fills column J of L: diagonal first, then sub-diagonal rows.
  template<typename L, Int J>
  SPARSEMAT_HD static void compute_column(const SparseMat& a, L& l, bool& ok) {
    diag_step<L, J>(a, l, ok);
    compute_subdiag<L, J>(a,
                          l,
                          ok,
                          std::make_index_sequence<static_cast<std::size_t>(N - J - 1)>{});
  }

  // Comma fold over columns J=0..N-1, in order: the comma operator inside a
  // fold expression is the built-in sequencing comma (left fully sequenced
  // before right), which is what makes it safe to replace the original
  // sequential recursion here — column J+1's inner_sum reads L values
  // written by column J (and earlier), so column order is load-bearing.
  template<typename L, std::size_t... Js>
  SPARSEMAT_HD static void outer_loop_fold(const SparseMat& a,
                                           L& l,
                                           bool& ok,
                                           std::index_sequence<Js...> /*seq*/) {
    (compute_column<L, static_cast<Int>(Js)>(a, l, ok), ...);
  }
  template<typename L>
  SPARSEMAT_HD static void outer_loop(const SparseMat& a, L& l, bool& ok) {
    outer_loop_fold<L>(a, l, ok, std::make_index_sequence<static_cast<std::size_t>(N)>{});
  }

 public:
  template<typename L>
  SPARSEMAT_HD static void factorize(const SparseMat& a, L& l, bool& ok) {
    outer_loop(a, l, ok);
  }
};

}  // namespace SparseLinearAlgebra::detail

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
  SPARSEMAT_HD explicit CholeskyFactor(L_Type l) : l_(std::move(l)) {}

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
  [[nodiscard]] SPARSEMAT_HD auto solve(const RHS& rhs) const {
    auto y = forward_solve(l_, rhs);
    if (!y.ok()) {
      return Result(std::move(y.value()), SolveStatus::SingularMatrix);
    }
    auto x = backward_solve(l_.transpose(), y.value());
    return Result(std::move(x.value()),
                  x.ok() ? SolveStatus::Success : SolveStatus::SingularMatrix);
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
 * @return           @c Result wrapping the lower-triangular L such that
 *                    A = L * L^T; @c ok() is @c false if a diagonal pivot
 *                    was zero or negative (A is not positive definite).
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto cholesky_factorize(const SparseMat& A) {
  auto l = detail::LCholeskyMatrix<SparseMat>::make_result();
  bool ok = true;
  detail::CholeskyFactorization<SparseMat>::factorize(A, l, ok);
  return Result(std::move(l), ok ? SolveStatus::Success : SolveStatus::SingularMatrix);
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
 * @return  @c Result wrapping the solution vector x such that Ax = b;
 *          @c ok() is @c false if the factorization or either triangular
 *          solve hit a zero/non-positive pivot.
 */
template<SparseMatrixType SparseMat, SparseMatrixType RHS>
SPARSEMAT_HD auto cholesky_solve(const SparseMat& A, const RHS& b) {
  auto l = cholesky_factorize(A);
  if (!l.ok()) {
    using LType = decltype(detail::LCholeskyMatrix<SparseMat>::make_result());
    using YType = decltype(detail::LowerTriangular<LType, RHS>::make_result());
    return Result(YType{}, SolveStatus::SingularMatrix);
  }
  auto y = forward_solve(l.value(), b);
  if (!y.ok()) {
    return Result(std::move(y.value()), SolveStatus::SingularMatrix);
  }
  auto x = backward_solve(l.value().transpose(), y.value());
  return Result(std::move(x.value()), x.ok() ? SolveStatus::Success : SolveStatus::SingularMatrix);
}

}  // namespace SparseLinearAlgebra
