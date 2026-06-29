#pragma once
#include <cmath>

#include "sparsemat/concepts/concepts.h"
namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for scalar multiplication of a sparse matrix.
 *
 * The sparsity pattern is unchanged: multiplying by a scalar cannot introduce
 * new non-zeros or eliminate existing ones (structural zeros remain zero).
 *
 * @tparam SparseMat The matrix type to scale.
 */
template<typename SparseMat>
class Scale {
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
  static auto scale(const SparseMat& a, DataType factor) {
    SparseMat result;
    result.values = a.values;

    for (auto& it : result.values) {
      it *= factor;
    }
    return result;
  }

  /// Multiplies every stored value of @p a by @p factor in place.
  static void scale_inplace(SparseMat& a, DataType factor) {
    for (auto& it : a.values) {
      it *= factor;
    }
  }
};
}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Returns a scaled copy of @p a: every non-zero element multiplied by @p factor.
 *
 * The sparsity pattern is preserved unchanged; structural zeros are not stored
 * and are unaffected.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Input matrix.
 * @param  factor    Scalar multiplier.
 * @return           Scaled matrix with the same sparsity pattern as @p a.
 */
template<SparseMatrixType SparseMat>
auto scale(const SparseMat& a, typename SparseMat::DataType factor) {
  return detail::Scale<SparseMat>::scale(a, factor);
}

/**
 * @brief Multiplies every non-zero element of @p a by @p factor in place.
 *
 * Modifies @p a directly without allocating a new matrix.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Matrix to modify.
 * @param  factor    Scalar multiplier.
 */
template<SparseMatrixType SparseMat>
void scale_inplace(SparseMat& a, typename SparseMat::DataType factor) {
  detail::Scale<SparseMat>::scale_inplace(a, factor);
}

/**
 * @brief Computes the Frobenius norm of @p a: √(Σ aᵢⱼ²).
 *
 * Only iterates over the stored non-zero values; structural zeros contribute
 * nothing to the sum.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Input matrix.
 * @return           Frobenius norm as the matrix's @c DataType.
 */
template<SparseMatrixType SparseMat>
auto frobenius(const SparseMat& a) {
  using DataType = typename SparseMat::DataType;
  DataType sq_sum = 0;
  for (auto& it : a.values) {
    sq_sum += it * it;
  }
  return std::sqrt(sq_sum);
}

/**
 * @brief Returns a unit-norm copy of @p a, scaled by @c 1/frobenius(a).
 *
 * The resulting matrix has a Frobenius norm of 1.  Behaviour is undefined if
 * @p a is the zero matrix (division by zero).
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Input matrix.
 * @return           Normalized matrix with the same sparsity pattern as @p a.
 */
template<SparseMatrixType SparseMat>
auto normalize(const SparseMat& a) {
  return detail::Scale<SparseMat>::scale(a, 1.0 / frobenius(a));
}

/**
 * @brief Normalizes @p a to unit Frobenius norm in place.
 *
 * Divides every stored element by @c frobenius(a).  Behavior is undefined if
 * @p a is the zero matrix.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Matrix to normalize in place.
 */
template<SparseMatrixType SparseMat>
auto normalize_inplace(SparseMat& a) {
  detail::Scale<SparseMat>::scale_inplace(a, 1.0 / frobenius(a));
}

}  // namespace SparseLinearAlgebra
