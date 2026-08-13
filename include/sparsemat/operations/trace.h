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
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;
  static constexpr auto num_non_zeros = SparseMat::nonZeroCount;
  static constexpr auto num_zeros = (rows * cols) - num_non_zeros;
  static constexpr auto total_elements = rows * cols;

  using Int = typename SparseMat::Int;

  /// Length of the diagonal that is actually summed.
  static constexpr auto diag_length = (rows < cols) ? rows : cols;

  /**
   * @brief Storage offsets of every *stored* diagonal entry, in row order.
   *
   * Computed once at compile time, so the runtime sum below touches only the
   * diagonal positions that actually exist — structurally zero ones are
   * skipped here rather than contributing a zero term at runtime, which is
   * the compile-time elimination that matters for trace.
   */
  SPARSEMAT_HD static constexpr auto diagonal_offsets() {
    std::array<Int,
               static_cast<std::size_t>(
                   SparseLinearAlgebra::MatrixUtilities<SparseMat>::diagonal_nonzeros())>
        offsets{};
    constexpr auto grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::storage_index_grid();
    std::size_t k = 0;
    for (Int i = 0; i < diag_length; ++i) {
      const auto offset = grid[static_cast<std::size_t>((i * cols) + i)];
      if (offset >= 0) {
        offsets[k++] = offset;
      }
    }
    return offsets;
  }

  /// Sums the stored diagonal entries with a plain runtime loop. The
  /// zero-skipping is already done (see diagonal_offsets), so there is nothing
  /// left for a fold to eliminate — and a fold over the diagonal of a large
  /// matrix would run into clang's 256-argument nesting limit for no benefit.
  SPARSEMAT_HD static DataType trace(const SparseMat& a) {
    constexpr auto offsets = diagonal_offsets();
    DataType sum = 0;
    for (auto offset : offsets) {
      sum += a.values[static_cast<std::size_t>(offset)];
    }
    return sum;
  }
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
SPARSEMAT_HD auto trace(const SparseMat& a) {
  return detail::Trace<SparseMat>::trace(a);
}

}  // namespace SparseLinearAlgebra
