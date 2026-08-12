#pragma once

#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/cholesky.h"
#include "sparsemat/operations/lu.h"
#include "sparsemat/operations/result.h"
#include "sparsemat/operations/triangular.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Builds the N x N identity as a right-hand side for an inverse solve.
 *
 * Kept separate from @c SparseMat::identity() because the result type has to be
 * rebound from whichever matrix is being inverted, and because the inverse
 * solves want it as an explicit block RHS.
 */
template<typename SparseMat, std::size_t... Is>
SPARSEMAT_HD static auto make_identity_rhs(std::index_sequence<Is...> /*seq*/) {
  using Int = typename SparseMat::Int;
  constexpr Int n = SparseMat::rows;
  typename SparseMat::template Rebind<n, n, (static_cast<Int>(Is) * (n + 1))...> result{};
  result.values.fill(typename SparseMat::DataType(1));
  return result;
}

template<typename SparseMat>
SPARSEMAT_HD static auto identity_rhs() {
  return make_identity_rhs<SparseMat>(
      std::make_index_sequence<static_cast<std::size_t>(SparseMat::rows)>{});
}

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Computes the determinant via the existing LU factorization.
 *
 * With Doolittle LU (unit diagonal on L) the determinant is just the product
 * of U's diagonal. There is no sign to track because this factorization does
 * no row swaps — the same reason it is only valid for pivot-free matrices in
 * the first place.
 *
 * A structurally triangular matrix skips the factorization entirely and
 * multiplies its own diagonal.
 *
 * @warning @c ok() is @c false when the factorization hit a negligible pivot.
 * The accompanying value is then the product of whatever diagonal entries were
 * computed, which is near zero but not otherwise meaningful — treat a failed
 * result as "singular, determinant is numerically zero" rather than reading the
 * number. A determinant of exactly 0 with @c ok() == @c true is possible too,
 * for a matrix that is singular in a way elimination reached cleanly.
 *
 * @tparam SparseMat Square input matrix type.
 * @param  a         Matrix whose determinant is wanted.
 * @return           @c Result wrapping the determinant.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto determinant(const SparseMat& a) {
  static_assert(SparseMat::rows == SparseMat::cols, "determinant requires a square matrix.");
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  constexpr Int n = SparseMat::rows;

  // A triangular matrix is already factorized: its determinant is the product
  // of its diagonal, so there is no reason to run elimination over it.
  if constexpr (detail::Triangular<SparseMat>::structurally_lower ||
                detail::Triangular<SparseMat>::structurally_upper) {
    constexpr auto offsets = MatrixUtilities<SparseMat>::storage_index_grid();
    DataType product = 1;
    for (Int i = 0; i < n; ++i) {
      const auto offset = offsets[static_cast<std::size_t>((i * n) + i)];
      // A structurally zero diagonal entry means a zero pivot, hence a zero
      // determinant — and it is an exact, trustworthy zero, so this is a
      // success rather than a failure.
      if (offset < 0) {
        return Result<DataType>(DataType(0), SolveStatus::Success);
      }
      product *= a.values[static_cast<std::size_t>(offset)];
    }
    return Result<DataType>(product, SolveStatus::Success);
  } else {
    auto lu = lu_factorize(a);
    const auto& u = lu.value().second;
    using UType = typename std::remove_cvref_t<decltype(u)>;
    constexpr auto u_offsets = MatrixUtilities<UType>::storage_index_grid();
    DataType product = 1;
    for (Int i = 0; i < n; ++i) {
      const auto offset = u_offsets[static_cast<std::size_t>((i * n) + i)];
      if (offset < 0) {
        return Result<DataType>(DataType(0), SolveStatus::Success);
      }
      product *= u.values[static_cast<std::size_t>(offset)];
    }
    return Result<DataType>(product, lu.ok() ? SolveStatus::Success : SolveStatus::SingularMatrix);
  }
}

/**
 * @brief Computes the matrix inverse by solving A * X = I.
 *
 * Reuses the existing block-RHS triangular solves: one LU factorization, then a
 * forward and a back substitution against the identity as a multi-column
 * right-hand side. That is the same work as N separate single-column solves but
 * with the factorization paid for once.
 *
 * @note The inverse of a sparse matrix is generally *dense*, so the result type
 * usually has far more stored values than the input — this is the one operation
 * where compile-time sparsity works against you. Prefer @c solve() when you
 * only need @c A⁻¹b for particular right-hand sides, which is most of the time;
 * forming the inverse explicitly is both slower and less accurate.
 *
 * @warning No pivoting, exactly as for @c lu_solve(). Check @c ok().
 *
 * @tparam SparseMat Square input matrix type.
 * @param  a         Matrix to invert.
 * @return           @c Result wrapping @c A⁻¹.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto inverse(const SparseMat& a) {
  static_assert(SparseMat::rows == SparseMat::cols, "inverse requires a square matrix.");
  const auto rhs = detail::identity_rhs<SparseMat>();
  return lu_solve(a, rhs);
}

/**
 * @brief Computes the inverse of a symmetric positive definite matrix via
 *        Cholesky.
 *
 * Half the factorization work of @c inverse(), and it will report a matrix that
 * is not (numerically) positive definite rather than quietly producing a wrong
 * answer — so prefer this whenever the matrix is known to be SPD.
 *
 * @note As with @c inverse(), the result is generally dense.
 *
 * @tparam SparseMat Square SPD input matrix type.
 * @param  a         Matrix to invert.
 * @return           @c Result wrapping @c A⁻¹; @c ok() is @c false if @p a is
 *                   not numerically SPD.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto cholesky_inverse(const SparseMat& a) {
  static_assert(SparseMat::rows == SparseMat::cols, "cholesky_inverse requires a square matrix.");
  const auto rhs = detail::identity_rhs<SparseMat>();
  return cholesky_solve(a, rhs);
}

}  // namespace SparseLinearAlgebra
