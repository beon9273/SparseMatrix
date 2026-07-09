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
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;
  static constexpr auto num_non_zeros = rows * cols;
  static constexpr auto num_zeros = 0;
  static constexpr auto total_elements = rows * cols;

  /// Returns true if at least one shared k makes both A[row,k] and B[k,col] non-zero.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int /*unused*/, Int /*unused*/) {
    return true;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Dense>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Dense>::calculate_sparsity();
  }

  template<typename Result, Int index = 0>
  SPARSEMAT_HD static auto dense(const SparseMat& a, Result& r) {
    if constexpr (index < total_elements) {
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

  SPARSEMAT_HD static auto dense(const SparseMat& a) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
            std::make_index_sequence<num_nonzeros()>{});
    dense(a, result);
    return result;
  }
};
}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto dense(const SparseMat& a) {
  return detail::Dense<SparseMat>::dense(a);
}

}  // namespace SparseLinearAlgebra
