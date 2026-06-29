#pragma once
#include <cmath>

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for expanding a sparse matrix into a dense array.
 *
 * Iterates at compile time over the stored non-zero values and writes each
 * one to its flat row-major position in a zero-initialised output array.
 * Positions not in the sparsity pattern remain zero.
 *
 * @tparam SparseMat The sparse matrix type to densify.
 */
template<SparseMatrixType SparseMat>
class Dense {
 public:
  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat::cols;
  static constexpr std::size_t num_non_zeros = rows * cols;
  static constexpr std::size_t num_zeros = 0;
  static constexpr std::size_t total_elements = rows * cols;

  /// Returns true if at least one shared k makes both A[row,k] and B[k,col] non-zero.
  constexpr static auto is_result_index_nonzero(std::size_t /*unused*/, std::size_t /*unused*/) { return true; }

  /// Delegates to OperationUtilities to count result non-zeros.
  constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Dense>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Dense>::calculate_sparsity();
  }

  template<typename Result, std::size_t index = 0>
  static auto dense(const SparseMat& a, Result& r) {
    if constexpr (index < rows * cols) {
      constexpr auto flat =
          SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(index / cols,
                                                                          index % cols);
      if constexpr (flat >= 0) {
        r.values[index] = a.values[flat];
      } else {
        r.values[index] = static_cast<DataType>(0);
      }
      dense<Result, index + 1>(a, r);
    }
  }

  static auto dense(const SparseMat& a) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseMat::template make<rows, cols, sparsity>(std::make_index_sequence<num_nonzeros()>{});
    dense(a, result);
    return result;
  }
};
}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

template<SparseMatrixType SparseMat>
auto dense(const SparseMat& a) {
  return detail::Dense<SparseMat>::dense(a);
}

}  // namespace SparseLinearAlgebra
