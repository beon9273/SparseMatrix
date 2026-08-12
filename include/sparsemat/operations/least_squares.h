#pragma once

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/cholesky.h"
#include "sparsemat/operations/multiply.h"
#include "sparsemat/operations/result.h"
#include "sparsemat/operations/transpose.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra {

/**
 * @brief Solves a non-square system @c A*x = b in the least-squares sense.
 *
 * @c solve() requires a square coefficient matrix. This handles both
 * rectangular cases, dispatching at compile time on the shape:
 *
 * - **Overdetermined** (@c rows > @c cols — more equations than unknowns, the
 *   usual fitting problem): returns the @c x minimising @c ||A*x - b||₂, via
 *   the normal equations @c AᵀA*x = Aᵀb. @c AᵀA is symmetric positive
 *   semi-definite, so it is solved with Cholesky.
 * - **Underdetermined** (@c rows < @c cols — infinitely many exact solutions):
 *   returns the minimum-norm one, the @c x with smallest @c ||x||₂ among those
 *   satisfying @c A*x = b, via @c x = Aᵀ*(A*Aᵀ)⁻¹*b.
 * - **Square**: the normal-equations path still applies, but you almost
 *   certainly want @c solve() instead — see the warning below.
 *
 * @warning **This squares the condition number.** Forming @c AᵀA doubles the
 * number of digits lost to conditioning, so a matrix that is merely
 * ill-conditioned for a direct solve can be numerically hopeless here, and one
 * that is rank-deficient will fail outright (@c AᵀA is then singular, reported
 * via @c ok()). A QR-based least-squares solve avoids this and is the right
 * answer for anything demanding; it is not implemented here because a
 * compile-time-sparsity Householder QR is a substantially larger piece of work
 * than reusing the existing Cholesky. For the small, well-conditioned,
 * full-rank problems this library targets, the normal equations are usually
 * fine — but check @c ok(), and do not reach for this when accuracy is the
 * point.
 *
 * @note @c AᵀA (or @c AAᵀ) is generally much denser than @c A, so this costs
 * more compile time than a same-sized square solve.
 *
 * @tparam A   Coefficient matrix type (any shape).
 * @tparam RHS Right-hand side type; @c RHS::rows must equal @c A::rows.
 * @param  a   Coefficient matrix.
 * @param  b   Right-hand side — one or more columns.
 * @return     @c Result wrapping the solution; @c ok() is @c false if the
 *             normal-equations matrix was singular, which for a full-rank
 *             @p a means it is too ill-conditioned for this method.
 */
template<SparseMatrixType A, SparseMatrixType RHS>
SPARSEMAT_HD auto least_squares_solve(const A& a, const RHS& b) {
  static_assert(A::rows == RHS::rows, "least_squares_solve requires RHS::rows == A::rows.");
  static_assert(SparseLinearAlgebra::SameDataType<A, RHS>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types — convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  const auto at = transpose(a);

  if constexpr (A::rows >= A::cols) {
    // Overdetermined (or square): AᵀA x = Aᵀb.
    return cholesky_solve(multiply(at, a), multiply(at, b));
  } else {
    // Underdetermined: solve (A Aᵀ) y = b, then x = Aᵀ y, which is the
    // minimum-norm solution.
    auto y = cholesky_solve(multiply(a, at), b);
    auto x = multiply(at, y.value());
    return Result<decltype(x)>(std::move(x), y.status());
  }
}

/**
 * @brief Residual @c A*x - b, for checking a least-squares fit.
 *
 * A least-squares solution does not generally satisfy @c A*x == b — that is the
 * point — so @c ok() alone says nothing about fit quality. This gives the
 * residual to measure it with; @c frobenius() of it is the @c ||A*x - b||₂ that
 * @c least_squares_solve() minimises.
 *
 * @param a Coefficient matrix.
 * @param x Solution returned by @c least_squares_solve().
 * @param b Original right-hand side.
 * @return  @c A*x - b.
 */
template<SparseMatrixType A, SparseMatrixType X, SparseMatrixType RHS>
SPARSEMAT_HD auto residual(const A& a, const X& x, const RHS& b) {
  return subtract(multiply(a, x), b);
}

}  // namespace SparseLinearAlgebra
