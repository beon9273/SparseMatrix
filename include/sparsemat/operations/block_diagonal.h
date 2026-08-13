#pragma once

#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for the block-diagonal composition
 *        @c diag(A, B).
 *
 * Produces an (A.rows + B.rows) x (A.cols + B.cols) matrix holding @c A in the
 * top-left block and @c B in the bottom-right, with the two off-diagonal
 * blocks structurally zero:
 * @code
 *   [ A  0 ]
 *   [ 0  B ]
 * @endcode
 *
 * Unlike @c Kronecker — whose result pattern is the *product* of both
 * operands' patterns and so grows fast — this is the *sum*: the result has
 * exactly @c nonZeroCount(A) + nonZeroCount(B) stored values, no matter how
 * the two patterns relate. That makes it the cheap way to assemble one solve
 * out of several independent sub-problems, which is the usual reason to want
 * it.
 *
 * @tparam SparseMat  Top-left block type.
 * @tparam SparseMat1 Bottom-right block type.
 */
template<SparseMatrixType SparseMat, SparseMatrixType SparseMat1>
class BlockDiagonal {
 public:
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, SparseMat1>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types — convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows + SparseMat1::rows;
  static constexpr auto cols = SparseMat::cols + SparseMat1::cols;

  /// Position (row, col) is stored iff it falls inside one of the two blocks
  /// and is stored in that block.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    if (row < SparseMat::rows && col < SparseMat::cols) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(row, col);
    }
    if (row >= SparseMat::rows && col >= SparseMat::cols) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMat1>::isNonZero(row - SparseMat::rows,
                                                                         col - SparseMat::cols);
    }
    return false;  // off-diagonal block
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<BlockDiagonal>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<BlockDiagonal>::calculate_sparsity();
  }

  /**
   * @brief Copies both blocks into the result.
   *
   * A runtime scatter over each operand's stored values rather than a
   * compile-time fold over result positions: this is pure data movement with
   * no structurally-zero term to eliminate, so unrolling buys nothing (see the
   * note on @c MatrixUtilities::storage_index_grid()). Each block's values are
   * copied straight across, offset into the result's coordinate space.
   */
  template<typename Result>
  SPARSEMAT_HD static void fill(Result& r, const SparseMat& a, const SparseMat1& b) {
    constexpr auto result_offsets =
        SparseLinearAlgebra::MatrixUtilities<Result>::storage_index_grid();
    constexpr auto a_indices = SparseMat::indices();
    constexpr auto b_indices = SparseMat1::indices();

    // Guarded because the bound is a compile-time constant: when it is zero the
    // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
    if constexpr (SparseMat::nonZeroCount != 0) {
      for (std::size_t k = 0; k < static_cast<std::size_t>(SparseMat::nonZeroCount); ++k) {
        const Int row = a_indices[k] / SparseMat::cols;
        const Int col = a_indices[k] % SparseMat::cols;
        r.values[static_cast<std::size_t>(
            result_offsets[static_cast<std::size_t>((row * cols) + col)])] = a.values[k];
      }
    }
    // Guarded because the bound is a compile-time constant: when it is zero the
    // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
    if constexpr (SparseMat1::nonZeroCount != 0) {
      for (std::size_t k = 0; k < static_cast<std::size_t>(SparseMat1::nonZeroCount); ++k) {
        const Int row = (b_indices[k] / SparseMat1::cols) + SparseMat::rows;
        const Int col = (b_indices[k] % SparseMat1::cols) + SparseMat::cols;
        r.values[static_cast<std::size_t>(
            result_offsets[static_cast<std::size_t>((row * cols) + col)])] = b.values[k];
      }
    }
  }

  SPARSEMAT_HD static auto block_diagonal(const SparseMat& a, const SparseMat1& b) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill<decltype(result)>(result, a, b);
    return result;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Composes two matrices block-diagonally: @c diag(a, b).
 *
 * The result is (a.rows + b.rows) x (a.cols + b.cols), with @p a in the
 * top-left block, @p b in the bottom-right, and the off-diagonal blocks
 * structurally zero. Its stored-value count is exactly the sum of the two
 * inputs'.
 *
 * @code
 * auto stacked = SparseLinearAlgebra::block_diagonal(a, b);
 * auto x = stacked.solve(rhs);   // both sub-problems in one solve
 * @endcode
 *
 * @tparam A Top-left block type.
 * @tparam B Bottom-right block type.
 * @param  a Top-left block.
 * @param  b Bottom-right block.
 * @return   Block-diagonal composition of @p a and @p b.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto block_diagonal(const A& a, const B& b) {
  return detail::BlockDiagonal<A, B>::block_diagonal(a, b);
}

}  // namespace SparseLinearAlgebra
