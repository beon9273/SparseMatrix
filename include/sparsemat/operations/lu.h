#pragma once

#include <cstddef>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/diagonal.h"
#include "sparsemat/operations/result.h"
#include "sparsemat/operations/triangular.h"
#include "sparsemat/operations/utils.h"
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
        std::make_index_sequence<static_cast<std::size_t>(numNonzeros)>{});
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
        std::make_index_sequence<static_cast<std::size_t>(numNonzeros)>{});
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

  // Single term of sum(L[I][m]*U[m][Col], m < Bound).
  template<typename L, typename U, Int I, Int Col, std::size_t M>
  SPARSEMAT_HD static DataType inner_sum_term(const L& l, const U& u) {
    constexpr auto lim =
        SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, static_cast<Int>(M));
    constexpr auto umk =
        SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(static_cast<Int>(M), Col);
    if constexpr (lim >= 0 && umk >= 0) {
      return l.values[lim] * u.values[umk];
    } else {
      return DataType(0);
    }
  }

  // Fold over m in [0, Bound) computing sum(L[I][m]*U[m][Col]). Bound is
  // passed as the pack size directly (an exact bound, unlike the
  // over-generate-and-filter approach used elsewhere in this codebase) since
  // it's always available as a compile-time constant at every call site
  // below.
  template<typename L, typename U, Int I, Int Col, std::size_t... Ms>
  SPARSEMAT_HD static DataType inner_loop_inner_fold(const L& l,
                                                     const U& u,
                                                     std::index_sequence<Ms...> /*seq*/) {
    return (inner_sum_term<L, U, I, Col, Ms>(l, u) + ...);
  }
  template<typename L, typename U, Int I, Int Bound, Int Col>
  SPARSEMAT_HD static auto inner_loop_inner(const SparseMat& /*a*/, L& l, U& u) {
    // (pack + ...) is ill-formed for an empty pack (unlike &&/||, + has no
    // built-in empty-fold identity in C++17), so Bound==0 — the first
    // column/row, with nothing yet to sum — needs an explicit early-out.
    if constexpr (Bound == 0) {
      return DataType(0);
    } else {
      return inner_loop_inner_fold<L, U, I, Col>(
          l, u, std::make_index_sequence<static_cast<std::size_t>(Bound)>{});
    }
  }

  // Single column step for U's row K: U[K][J] = A[K][J] - sum(L[K][m]*U[m][J], m<K).
  template<typename L, typename U, Int K, std::size_t JOff>
  SPARSEMAT_HD static void u_col_step(const SparseMat& a, L& l, U& u) {
    constexpr Int J = K + static_cast<Int>(JOff);
    constexpr auto u_idx = SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(K, J);
    if constexpr (u_idx >= 0) {
      DataType sum = inner_loop_inner<L, U, K, K, J>(a, l, u);
      constexpr auto a_idx = MU::getSparseIndex(K, J);
      if constexpr (a_idx >= 0) {
        u.values[u_idx] = a.values[a_idx] - sum;
      } else {
        u.values[u_idx] = -sum;
      }
    }
  }
  // Fills U's entire row K via a fold over J in [K, N). Column order doesn't
  // matter here (each J writes an independent U cell), but a comma fold is
  // used for consistency with the K-sequential fold in outer_loop_over_rows.
  template<typename L, typename U, Int K, std::size_t... JOffs>
  SPARSEMAT_HD static void inner_loop_1(const SparseMat& a,
                                        L& l,
                                        U& u,
                                        std::index_sequence<JOffs...> /*seq*/) {
    (u_col_step<L, U, K, JOffs>(a, l, u), ...);
  }

  // Single row step for L's column K: L[I][K] = (A[I][K] - sum(L[I][m]*U[m][K], m<K)) / pivot.
  template<typename L, typename U, Int K, std::size_t IOff>
  SPARSEMAT_HD static void l_col_step(const SparseMat& a, L& l, U& u, DataType pivot) {
    constexpr Int I = K + 1 + static_cast<Int>(IOff);
    constexpr auto l_idx = SparseLinearAlgebra::MatrixUtilities<L>::getSparseIndex(I, K);
    if constexpr (l_idx >= 0) {
      if (pivot == DataType(0)) {
        l.values[l_idx] = DataType(0);
        return;
      }
      DataType sum = inner_loop_inner<L, U, I, K, K>(a, l, u);
      constexpr auto a_idx = MU::getSparseIndex(I, K);
      if constexpr (a_idx < 0) {
        l.values[l_idx] = -sum / pivot;
      } else {
        l.values[l_idx] = (a.values[a_idx] - sum) / pivot;
      }
    }
  }
  // Fills L's entire column K via a fold over I in [K+1, N). Row order
  // doesn't matter here (each I writes an independent L cell).
  template<typename L, typename U, Int K, std::size_t... IOffs>
  SPARSEMAT_HD static void inner_loop_2(const SparseMat& a,
                                        L& l,
                                        U& u,
                                        [[maybe_unused]] DataType pivot,
                                        std::index_sequence<IOffs...> /*seq*/) {
    // pivot is unused when IOffs is empty (K == N-1, the last row has no
    // sub-diagonal entries to fill).
    (l_col_step<L, U, K, IOffs>(a, l, u, pivot), ...);
  }

  // Single pivot step K: fills U's row K, reads the pivot, then fills L's
  // column K using it.
  template<typename L, typename U, std::size_t K>
  SPARSEMAT_HD static void outer_step(const SparseMat& a, L& l, U& u, bool& ok) {
    inner_loop_1<L, U, static_cast<Int>(K)>(
        a, l, u, std::make_index_sequence<static_cast<std::size_t>(N) - K>{});

    // Structurally zero pivot is a user error (caught elsewhere); a
    // structurally-present but numerically negligible pivot is a runtime
    // singular-matrix condition reported back via `ok`.
    //
    // The test is a magnitude threshold rather than `== 0` because this
    // factorization does no pivoting: without row swaps a merely *tiny*
    // pivot is just as fatal as an exactly-zero one — it divides through and
    // produces enormous, meaningless multipliers — but an exact-equality test
    // waves it through with ok() == true. Scaling the threshold by the
    // largest magnitude seen in U so far makes it a relative test, so it
    // behaves the same for a matrix expressed in metres and one in
    // micrometres.
    constexpr auto u_kk =
        SparseLinearAlgebra::MatrixUtilities<U>::getSparseIndex(static_cast<Int>(K),
                                                                static_cast<Int>(K));
    DataType pivot = DataType(0);
    if constexpr (u_kk >= 0) {
      pivot = u.values[u_kk];
    }
    DataType scale = DataType(0);
    for (const auto& v : u.values) {
      const DataType mag = v < DataType(0) ? -v : v;
      if (mag > scale) {
        scale = mag;
      }
    }
    const DataType pivot_mag = pivot < DataType(0) ? -pivot : pivot;
    if (pivot_mag <= singular_pivot_threshold<DataType>() * scale) {
      ok = false;
    }

    inner_loop_2<L, U, static_cast<Int>(K)>(
        a, l, u, pivot, std::make_index_sequence<static_cast<std::size_t>(N) - K - 1>{});
  }

  // Comma fold over pivot steps K=0..N-1, in order: the comma operator
  // inside a fold expression is the built-in sequencing comma (left fully
  // sequenced before right), which is what makes it safe to replace the
  // original sequential recursion here — step K+1's inner_loop_inner reads
  // L/U values written by step K (and earlier), so step order is
  // load-bearing.
  template<typename L, typename U, std::size_t... Ks>
  SPARSEMAT_HD static void outer_loop_over_rows(
      const SparseMat& a, L& l, U& u, bool& ok, std::index_sequence<Ks...> /*seq*/) {
    (outer_step<L, U, Ks>(a, l, u, ok), ...);
  }

 public:
  template<typename L, typename U>
  SPARSEMAT_HD static void factorize(const SparseMat& a, L& l, U& u, bool& ok) {
    SparseLinearAlgebra::set_diagonal<L>(l, DataType(1));
    outer_loop_over_rows<L, U>(
        a, l, u, ok, std::make_index_sequence<static_cast<std::size_t>(N)>{});
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
