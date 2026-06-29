#pragma once
#include <cmath>

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for scalar multiplication of a sparse matrix.
 *
 * The sparsity pattern is unchanged: multiplying by a scalar cannot introduce
 * new non-zeros or eliminate existing ones (structural zeros remain zero).
 *
 * @tparam SparseMat The matrix type to scale.
 */
template<SparseMatrixType SparseMat>
class Shift {
 public:
  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat::cols;
  static constexpr std::size_t num_non_zeros = SparseMat::nonZeroCount;
  static constexpr std::size_t num_zeros = SparseMat::zeroCount;
  static constexpr std::size_t total_elements = rows * cols;

  /// Returns the unchanged input sparsity as the result sparsity.
  constexpr static auto calculate_sparsity() { return SparseMat::indices; }

  /// Returns a copy of @p a with every stored value multiplied by @p factor.
  static auto shift(const SparseMat& a, DataType factor) {
    SparseMat result;
    result.values = a.values;

    for (auto& it : result.values) {
      it += factor;
    }
    return result;
  }

  /// Adds @p factor to every stored value of @p a in place.
  static void shift_inplace(SparseMat& a, DataType factor) {
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
auto shift(const SparseMat& a, typename SparseMat::DataType factor) {
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
template<typename SparseMat>
void shift_inplace(SparseMat& a, typename SparseMat::DataType factor) {
  detail::Shift<SparseMat>::shift_inplace(a, factor);
}

}  // namespace SparseLinearAlgebra
