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

  /// Every position of a dense result is stored, so this is unconditionally true.
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

  /**
   * @brief Scatters @p a's stored values into a fully-dense result.
   *
   * A plain runtime loop rather than a compile-time fold or recursion, because
   * densifying is pure data movement: every stored value is copied to the slot
   * its flat index names, and there is no structurally-zero term to eliminate
   * at compile time. Unrolling therefore buys nothing here while costing one
   * instantiation per element — and imposing a hard size ceiling on top
   * (linear recursion blew the 900-deep instantiation budget at 32x32; a fold
   * over the same 1024 values exceeds clang's 256-argument nesting limit).
   *
   * The result's storage index equals its flat row-major index, since its
   * sparsity array is 0, 1, ..., rows*cols-1 in order. It is value-initialized,
   * so positions this loop never touches are already zero.
   */
  SPARSEMAT_HD static auto dense(const SparseMat& a) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    constexpr auto inds = SparseMat::indices();
    // Guarded because the bound is a compile-time constant: when it is zero the
    // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
    if constexpr (SparseMat::nonZeroCount != 0) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(SparseMat::nonZeroCount); ++i) {
        result.values[static_cast<std::size_t>(inds[i])] = a.values[i];
      }
    }
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
