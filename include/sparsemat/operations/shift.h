#pragma once
#include <cmath>

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for adding a scalar to a sparse matrix's stored
 *        values.
 *
 * The sparsity pattern is unchanged: only *stored* values are shifted, so a
 * structurally zero position stays zero rather than becoming @c factor. (This
 * is deliberately not the same as adding the scalar to every element of the
 * full matrix, which would make a sparse matrix dense.)
 *
 * @tparam SparseMat The matrix type to shift.
 */
template<SparseMatrixType SparseMat>
class Shift {
 public:
  using DataType = typename SparseMat::DataType;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;
  static constexpr auto num_non_zeros = SparseMat::nonZeroCount;
  static constexpr auto num_zeros = (rows * cols) - num_non_zeros;
  static constexpr auto total_elements = rows * cols;

  /// Returns the unchanged input sparsity as the result sparsity.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() { return SparseMat::indices(); }

  /// Returns a copy of @p a with @p factor added to every stored value.
  SPARSEMAT_HD static auto shift(const SparseMat& a, DataType factor) {
    SparseMat result;
    result.values = a.values;

    for (auto& it : result.values) {
      it += factor;
    }
    return result;
  }

  /// Adds @p factor to every stored value of @p a in place.
  SPARSEMAT_HD static void shift_inplace(SparseMat& a, DataType factor) {
    for (auto& it : a.values) {
      it += factor;
    }
  }
};
}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Returns a shifted copy of @p a: every non-zero element increased by @p factor.
 *
 * The sparsity pattern is preserved unchanged; structural zeros are not stored
 * and are unaffected.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Input matrix.
 * @param  factor    Scalar to add.
 * @return           Shifted matrix with the same sparsity pattern as @p a.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto shift(const SparseMat& a, typename SparseMat::DataType factor) {
  return detail::Shift<SparseMat>::shift(a, factor);
}

/**
 * @brief Adds @p factor to every non-zero element of @p a in place.
 *
 * Modifies @p a directly without allocating a new matrix.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Matrix to modify.
 * @param  factor    Scalar to add.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD void shift_inplace(SparseMat& a, typename SparseMat::DataType factor) {
  detail::Shift<SparseMat>::shift_inplace(a, factor);
}

}  // namespace SparseLinearAlgebra
