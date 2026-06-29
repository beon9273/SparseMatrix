#pragma once

#include "operations/utils.h"

namespace {

/**
 * @brief Implementation policy for sparse matrix transposition.
 *
 * Swaps the row and column dimensions and remaps every non-zero flat index
 * from @c row*cols+col to @c col*rows+row.  The non-zero count is unchanged.
 *
 * @tparam SparseMat The matrix type to transpose.
 */
template<SparseMatrixType SparseMat>
class Transpose {
 public:
  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::cols;
  static constexpr std::size_t cols = SparseMat::rows;
  static constexpr std::size_t num_non_zeros = SparseMat::nonZeroCount;
  static constexpr std::size_t num_zeros = (rows * cols) - num_non_zeros;

  /// Returns true if (row, col) in the result corresponds to a non-zero at (col, row) in the input.
  constexpr static auto is_result_index_nonzero(int row, int col) {
    return (SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(col, row));
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Transpose>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Transpose>::calculate_sparsity();
  }

  /// Recursively copies a.values[J,I] into r.values[I,J] for every non-zero result position.
  template<SparseMatrixType Result, std::size_t I, std::size_t J>
  static void transpose(Result& r, const SparseMat& a) {
    if constexpr (I >= Result::rows) {
      return;
    } else if constexpr (J >= Result::cols) {
      return transpose<Result, I + 1, 0>(r, a);
    } else {
      constexpr int sparse_index =
          SparseLinearAlgebra::MatrixUtilities<Result>::getSparseIndex(I, J);
      if constexpr (sparse_index >= 0) {
        constexpr int a_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(J, I);
        static_assert(
            a_index >= 0,
            "Transpose index mismatch: expected non-zero element in the original matrix.");
        r.values[sparse_index] = a.values[a_index];
      }
      transpose<Result, I, J + 1>(r, a);
    }
  }

  /// Constructs the transposed SparseMat and fills it via the recursive helper.
  static auto transpose(const SparseMat& a) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseMat::template make<SparseMat::cols, SparseMat::rows, sparsity>(
        std::make_index_sequence<num_nonzeros()>{});
    transpose<decltype(result), 0, 0>(result, a);
    return result;
  }
};
}  // namespace

namespace SparseLinearAlgebra {

/**
 * @brief Returns the transpose of a sparse matrix.
 *
 * Produces a new @c SparseMat with swapped dimensions and remapped non-zero
 * indices.  The value at position (i, j) in the result equals the value at
 * (j, i) in @p a.
 *
 * @tparam SparseMat Input matrix type.
 * @param  a         Matrix to transpose.
 * @return           Transposed matrix of type @c SparseMat<DType, Cols, Rows, ...>.
 */
template<SparseMatrixType SparseMat>
auto transpose(const SparseMat& a) {
  return Transpose<SparseMat>::transpose(a);
}

}  // namespace SparseLinearAlgebra
