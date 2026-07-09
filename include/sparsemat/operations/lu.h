#pragma once

#include <cstddef>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/diagonal.h"
#include "sparsemat/operations/result.h"
#include "sparsemat/operations/triangular.h"
namespace SparseLinearAlgebra::detail {

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
  using Int = typename SparseMat::Int;
  static_assert(SparseMat::rows == SparseMat::cols, "LU factorization requires a square matrix.");
  static constexpr auto N = SparseMat::rows;

  // Initialise fill from A's structural non-zeros, then propagate: for each
  // pivot column k, any row i with fill[i][k] inherits all of row k's fill.
  static constexpr auto compute_fill() {
    std::array<std::array<bool, N>, N> fill{};
    for (auto idx : SparseMat::indices()) {
      fill[idx / SparseMat::cols][idx % SparseMat::cols] = true;
    }
    for (Int k = 0; k < N; ++k) {
      for (Int i = k + 1; i < N; ++i) {
        if (!fill[i][k]) {
          continue;
        }
        for (Int j = k; j < N; ++j) {
          if (fill[k][j]) {
            fill[i][j] = true;
          }
        }
      }
    }
    return fill;
  }

 public:
  static constexpr auto fill = compute_fill();

  /// Returns @c true if L[row][col] is structurally non-zero.
  /// L has a unit diagonal, so diagonal positions always return @c true.
  SPARSEMAT_HD static constexpr bool l_nonzero(Int row, Int col) {
    if (row == col) {
      return true;
    }
    if (col > row) {
      return false;
    }
    return fill[row][col];
  }

  /// Returns @c true if U[row][col] is structurally non-zero.
  SPARSEMAT_HD static constexpr bool u_nonzero(Int row, Int col) {
    if (col < row) {
      return false;
    }
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
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;

  SPARSEMAT_HD constexpr static bool is_result_index_nonzero(Int row, Int col) {
    return LUSparsity<SparseMat>::l_nonzero(row, col);
  }

  static constexpr auto numNonzeros =
      SparseLinearAlgebra::OperationUtilities<LMatrix>::num_nonzeros();
  static constexpr auto sparsity =
      SparseLinearAlgebra::OperationUtilities<LMatrix>::calculate_sparsity();

  SPARSEMAT_HD static auto make_result() {
    return SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
        std::make_index_sequence<numNonzeros>{});
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
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;

  SPARSEMAT_HD constexpr static bool is_result_index_nonzero(Int row, Int col) {
    return LUSparsity<SparseMat>::u_nonzero(row, col);
  }

  static constexpr auto numNonzeros =
      SparseLinearAlgebra::OperationUtilities<UMatrix>::num_nonzeros();
  static constexpr auto sparsity =
      SparseLinearAlgebra::OperationUtilities<UMatrix>::calculate_sparsity();

  SPARSEMAT_HD static auto make_result() {
    return SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
        std::make_index_sequence<numNonzeros>{});
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
  static constexpr auto N = SparseMat::rows;
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  using MU = SparseLinearAlgebra::MatrixUtilities<SparseMat>;

  template<typename L, typename U, Int I, Int Bound, Int Col, Int M = 0>
  SPARSEMAT_HD static auto inner_loop_inner(const SparseMat& a, L& l, U& u) {
    if constexpr (M < Bound) {
      constexpr auto lim = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, M);
      constexpr auto umk = SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(M, Col);
      DataType sum = DataType(0);
      if constexpr (lim >= 0 && umk >= 0) {
        sum += l.values[lim] * u.values[umk];
      }
      return sum + inner_loop_inner<L, U, I, Bound, Col, M + 1>(a, l, u);
    } else {
      return DataType(0);
    }
  }

  template<typename L, typename U, Int K, Int J = K>
  SPARSEMAT_HD static void inner_loop_1(const SparseMat& a, L& l, U& u) {
    if constexpr (J < N) {
      constexpr auto u_idx = SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(K, J);
      if constexpr (u_idx >= 0) {
        DataType sum = inner_loop_inner<L, U, K, K, J, 0>(a, l, u);
        constexpr auto a_idx = MU::getSparseIndex(K, J);
        if constexpr (a_idx >= 0) {
          u.values[u_idx] = a.values[a_idx] - sum;
        } else {
          u.values[u_idx] = -sum;
        }
      }
      inner_loop_1<L, U, K, J + 1>(a, l, u);
    }
  }

  template<typename L, typename U, Int K, Int I = K + 1>
  SPARSEMAT_HD static void inner_loop_2(const SparseMat& a, L& l, U& u, DataType pivot) {
    if constexpr (I < N) {
      constexpr auto l_idx = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, K);
      if constexpr (l_idx >= 0) {
        if (pivot == DataType(0)) {
          l.values[l_idx] = DataType(0);
          inner_loop_2<L, U, K, I + 1>(a, l, u, pivot);
          return;
        }
        DataType sum = inner_loop_inner<L, U, I, K, K, 0>(a, l, u);
        constexpr auto a_idx = MU::getSparseIndex(I, K);
        if constexpr (a_idx < 0) {
          l.values[l_idx] = -sum / pivot;
        } else {
          l.values[l_idx] = (a.values[a_idx] - sum) / pivot;
        }
      }
      inner_loop_2<L, U, K, I + 1>(a, l, u, pivot);
    }
  }

  template<typename L, typename U, Int K>
  SPARSEMAT_HD static void outer_loop_over_rows(const SparseMat& a, L& l, U& u, bool& ok) {
    if constexpr (K < N) {
      inner_loop_1<L, U, K>(a, l, u);

      // Structurally zero pivot is a user error (caught elsewhere); a
      // structurally-present but numerically zero pivot is a runtime
      // singular-matrix condition reported back via `ok`.
      constexpr auto u_kk = SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(K, K);
      DataType pivot = DataType(0);
      if constexpr (u_kk >= 0) {
        pivot = u.values[u_kk];
      }
      if (pivot == DataType(0)) {
        ok = false;
      }

      inner_loop_2<L, U, K>(a, l, u, pivot);
      outer_loop_over_rows<L, U, K + 1>(a, l, u, ok);
    }
  }

 public:
  template<typename L, typename U, Int K = 0>
  SPARSEMAT_HD static void factorize(const SparseMat& a, L& l, U& u, bool& ok) {
    SparseLinearAlgebra::set_diagonal<L>(l, DataType(1));
    outer_loop_over_rows<L, U, 0>(a, l, u, ok);
  }
};

}  // namespace SparseLinearAlgebra::detail

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
 *
 * @return @c Result wrapping the (L, U) pair; @c ok() is @c false if a zero
 *         pivot was hit during elimination.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto lu_factorize(const SparseMat& A) {
  auto l = detail::LMatrix<SparseMat>::make_result();
  auto u = detail::UMatrix<SparseMat>::make_result();
  bool ok = true;
  detail::LUFactorization<SparseMat>::factorize(A, l, u, ok);
  return Result(std::make_pair(l, u), ok ? SolveStatus::Success : SolveStatus::SingularMatrix);
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
 * @return @c Result wrapping the solution x such that LUx = RHS; @c ok() is
 *         @c false if either triangular solve hit a zero pivot.
 */
template<SparseMatrixType SparseMat, SparseMatrixType L, SparseMatrixType U>
SPARSEMAT_HD auto lu_solve(const L& l, const U& u, const SparseMat& RHS) {
  auto y = forward_solve(l, RHS);
  if (!y.ok()) {
    return Result(std::move(y.value()), SolveStatus::SingularMatrix);
  }
  auto x = backward_solve(u, y.value());
  return Result(std::move(x.value()), x.ok() ? SolveStatus::Success : SolveStatus::SingularMatrix);
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
 * @return  @c Result wrapping the solution vector x such that Ax = b;
 *          @c ok() is @c false if a zero pivot was hit anywhere in the
 *          factorization or the two triangular solves.
 */
template<SparseMatrixType SparseMat, SparseMatrixType RHS>
SPARSEMAT_HD auto lu_solve(const SparseMat& A, const RHS& b) {
  auto lu = lu_factorize(A);
  if (!lu.ok()) {
    using LType = decltype(detail::LMatrix<SparseMat>::make_result());
    using YType = decltype(detail::LowerTriangular<LType, RHS>::make_result());
    return Result(YType{}, SolveStatus::SingularMatrix);
  }
  auto solved = lu_solve(lu.value().first, lu.value().second, b);
  return Result(std::move(solved.value()),
                solved.ok() ? SolveStatus::Success : SolveStatus::SingularMatrix);
}

}  // namespace SparseLinearAlgebra
