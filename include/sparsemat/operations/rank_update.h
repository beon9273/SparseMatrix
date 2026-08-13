#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for the outer product @c x yᵀ.
 *
 * @p x is m × 1 and @p y is n × 1, giving an m × n result whose pattern is the
 * product of the two vectors' patterns: (i, j) is stored exactly when both
 * @c x[i] and @c y[j] are.
 *
 * Every result element is a single multiplication, so there is no
 * zero-skipping left to exploit at fill time and the fill is a runtime loop
 * over a precomputed table rather than a fold — see
 * @c MatrixUtilities::storage_index_grid().
 *
 * @tparam VecX Column vector type (m × 1).
 * @tparam VecY Column vector type (n × 1).
 */
template<SparseMatrixType VecX, SparseMatrixType VecY>
class Outer {
 public:
  using DataType = typename VecX::DataType;
  using Int = typename VecX::Int;
  static constexpr Int rows = VecX::rows;
  static constexpr Int cols = VecY::rows;

  static_assert(VecX::cols == 1 && VecY::cols == 1,
                "An outer product takes two column vectors; transpose or reshape first.");
  static_assert(SparseLinearAlgebra::SameDataType<VecX, VecY>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types — convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  static constexpr auto x_slots = SparseLinearAlgebra::MatrixUtilities<VecX>::storage_index_grid();
  static constexpr auto y_slots = SparseLinearAlgebra::MatrixUtilities<VecY>::storage_index_grid();

  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return x_slots[static_cast<std::size_t>(row)] >= 0 &&
           y_slots[static_cast<std::size_t>(col)] >= 0;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Outer>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Outer>::calculate_sparsity();
  }

  /// The (x, y) storage offsets feeding each result slot.
  SPARSEMAT_HD constexpr static auto source_slots() {
    constexpr auto sparsity = calculate_sparsity();
    std::array<std::array<Int, 2>, sparsity.size()> slots{};
    for (std::size_t k = 0; k < sparsity.size(); ++k) {
      slots[k][0] = x_slots[static_cast<std::size_t>(sparsity[k] / cols)];
      slots[k][1] = y_slots[static_cast<std::size_t>(sparsity[k] % cols)];
    }
    return slots;
  }

  SPARSEMAT_HD static auto outer(const VecX& x, const VecY& y, DataType alpha) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseLinearAlgebra::MatrixUtilities<VecX>::template make<rows, cols, sparsity>(
        std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    constexpr auto slots = source_slots();
    for (std::size_t k = 0; k < slots.size(); ++k) {
      result.values[k] = alpha * x.values[static_cast<std::size_t>(slots[k][0])] *
                         y.values[static_cast<std::size_t>(slots[k][1])];
    }
    return result;
  }
};

/**
 * @brief Implementation policy for the rank-1 update @c A + alpha·x yᵀ.
 *
 * The update every Kalman gain application, BFGS step, and Sherman–Morrison
 * correction is built from. Fusing it matters for the same reason @c axpy
 * exists: written out, @c a.add(outer(x, y).scale(alpha)) materializes the
 * full outer product — which is dense wherever both vectors are — purely to
 * add it and throw it away. Here it is accumulated straight into the result.
 *
 * Result pattern is the union of A's and the outer product's, which is the
 * correct superset: the update can create non-zeros where A had none, and
 * cannot be assumed to cancel any that A has.
 *
 * @tparam SparseMat Base matrix type (m × n).
 * @tparam VecX      Column vector type (m × 1).
 * @tparam VecY      Column vector type (n × 1).
 */
template<SparseMatrixType SparseMat, SparseMatrixType VecX, SparseMatrixType VecY>
class Rank1Update {
 public:
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr Int rows = SparseMat::rows;
  static constexpr Int cols = SparseMat::cols;

  static_assert(VecX::cols == 1 && VecY::cols == 1,
                "A rank-1 update takes two column vectors; transpose or reshape first.");
  static_assert(VecX::rows == SparseMat::rows && VecY::rows == SparseMat::cols,
                "Incompatible dimensions for a rank-1 update: x must be A::rows × 1 and y must "
                "be A::cols × 1.");
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, VecX> &&
                    SparseLinearAlgebra::SameDataType<SparseMat, VecY>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types — convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  static constexpr auto a_slots =
      SparseLinearAlgebra::MatrixUtilities<SparseMat>::storage_index_grid();
  static constexpr auto x_slots = SparseLinearAlgebra::MatrixUtilities<VecX>::storage_index_grid();
  static constexpr auto y_slots = SparseLinearAlgebra::MatrixUtilities<VecY>::storage_index_grid();

  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    if (a_slots[static_cast<std::size_t>((row * cols) + col)] >= 0) {
      return true;
    }
    return x_slots[static_cast<std::size_t>(row)] >= 0 &&
           y_slots[static_cast<std::size_t>(col)] >= 0;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Rank1Update>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Rank1Update>::calculate_sparsity();
  }

  /// Per result slot: the offsets into A, x and y, each -1 when that operand
  /// contributes nothing there.
  SPARSEMAT_HD constexpr static auto source_slots() {
    constexpr auto sparsity = calculate_sparsity();
    std::array<std::array<Int, 3>, sparsity.size()> slots{};
    for (std::size_t k = 0; k < sparsity.size(); ++k) {
      const Int i = sparsity[k] / cols;
      const Int j = sparsity[k] % cols;
      slots[k][0] = a_slots[static_cast<std::size_t>(sparsity[k])];
      slots[k][1] = x_slots[static_cast<std::size_t>(i)];
      slots[k][2] = y_slots[static_cast<std::size_t>(j)];
    }
    return slots;
  }

  SPARSEMAT_HD static auto update(const SparseMat& a,
                                  const VecX& x,
                                  const VecY& y,
                                  DataType alpha) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    constexpr auto slots = source_slots();
    for (std::size_t k = 0; k < slots.size(); ++k) {
      DataType value =
          slots[k][0] >= 0 ? a.values[static_cast<std::size_t>(slots[k][0])] : DataType(0);
      if (slots[k][1] >= 0 && slots[k][2] >= 0) {
        value += alpha * x.values[static_cast<std::size_t>(slots[k][1])] *
                 y.values[static_cast<std::size_t>(slots[k][2])];
      }
      result.values[k] = value;
    }
    return result;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Outer product @c alpha·x yᵀ of two column vectors.
 *
 * @p x is m × 1, @p y is n × 1, and the result is m × n, stored wherever both
 * vectors are.
 *
 * @tparam X     Left vector type (m × 1).
 * @tparam Y     Right vector type (n × 1).
 * @param  x     Left vector.
 * @param  y     Right vector.
 * @param  alpha Scalar multiplier (default 1).
 * @return       @c alpha·x yᵀ.
 */
template<SparseMatrixType X, SparseMatrixType Y>
SPARSEMAT_HD auto outer(const X& x,
                        const Y& y,
                        typename X::DataType alpha = typename X::DataType(1)) {
  return detail::Outer<X, Y>::outer(x, y, alpha);
}

/**
 * @brief Rank-1 update @c A + alpha·x yᵀ, computed in one pass.
 *
 * The outer product is never materialized: each result element is written once
 * as @c A[i,j] + alpha·x[i]·y[j]. The result pattern is the union of @p a's and
 * the outer product's, so the update may widen the pattern — which is inherent
 * to the operation, not an artifact of fusing it.
 *
 * @tparam A     Base matrix type (m × n).
 * @tparam X     Left vector type (m × 1).
 * @tparam Y     Right vector type (n × 1).
 * @param  a     Base matrix.
 * @param  x     Left vector.
 * @param  y     Right vector.
 * @param  alpha Scalar multiplier on the update (default 1; pass -1 to
 *               downdate).
 * @return       @c A + alpha·x yᵀ.
 */
template<SparseMatrixType A, SparseMatrixType X, SparseMatrixType Y>
SPARSEMAT_HD auto rank1_update(const A& a,
                               const X& x,
                               const Y& y,
                               typename A::DataType alpha = typename A::DataType(1)) {
  return detail::Rank1Update<A, X, Y>::update(a, x, y, alpha);
}

/**
 * @brief Symmetric rank-1 update @c A + alpha·x xᵀ.
 *
 * The common special case, and the one that keeps a symmetric matrix
 * symmetric — a covariance correction, or a quasi-Newton curvature update.
 *
 * @tparam A     Base matrix type (n × n).
 * @tparam X     Vector type (n × 1).
 * @param  a     Base matrix.
 * @param  x     Update vector.
 * @param  alpha Scalar multiplier (default 1).
 * @return       @c A + alpha·x xᵀ.
 */
template<SparseMatrixType A, SparseMatrixType X>
SPARSEMAT_HD auto symmetric_rank1_update(const A& a,
                                         const X& x,
                                         typename A::DataType alpha = typename A::DataType(1)) {
  static_assert(A::rows == A::cols, "symmetric_rank1_update requires a square matrix.");
  return detail::Rank1Update<A, X, X>::update(a, x, x, alpha);
}

}  // namespace SparseLinearAlgebra
