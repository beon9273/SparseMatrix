#pragma once

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for computing the trace of a sparse matrix.
 *
 * Iterates at compile time over the diagonal positions and accumulates only
 * those that are stored as non-zeros, skipping structurally zero entries
 * without any runtime branch.
 *
 * @tparam SparseMat The matrix type; need not be square, but only the
 *                   @c min(rows, cols) diagonal entries are summed.
 */
template<SparseMatrixType SparseMat>
class Trace {
 public:
  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat::cols;
  static constexpr std::size_t num_non_zeros = SparseMat::nonZeroCount;
  static constexpr std::size_t num_zeros = SparseMat::zeroCount;
  static constexpr std::size_t total_elements = rows * cols;

  /// Compile-time recursive accumulation of diagonal elements starting at (N, N).
  template<int N>
  static DataType trace(const SparseMat& a) {
    if constexpr (N >= rows || N >= cols) {
      return 0;
    } else if constexpr (SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(N, N)) {
      constexpr int index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(N, N);
      return a.values[index] + trace<N + 1>(a);
    } else {
      return trace<N + 1>(a);
    }
  };
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Computes the trace of a sparse matrix (sum of diagonal elements).
 *
 * Only diagonal positions that are structurally non-zero contribute to the
 * sum; structurally zero diagonal entries are skipped at compile time.
 * For non-square matrices the sum runs over the @c min(rows, cols) diagonal.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Input matrix.
 * @return           Sum of stored diagonal elements as @c SparseMat::DataType.
 */
template<SparseMatrixType SparseMat>
auto trace(const SparseMat& a) {
  return detail::Trace<SparseMat>::template trace<0>(a);
}

}  // namespace SparseLinearAlgebra
