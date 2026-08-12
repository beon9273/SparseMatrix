#pragma once

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

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
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::cols;
  static constexpr auto cols = SparseMat::rows;
  static constexpr auto num_non_zeros = SparseMat::nonZeroCount;
  static constexpr auto num_zeros = (rows * cols) - num_non_zeros;

  // Precomputed once — see Add::a_grid/b_grid for why (identical reasoning).
  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();

  /// Returns true if (row, col) in the result corresponds to a non-zero at (col, row) in the input.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return a_grid[col][row];
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Transpose>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Transpose>::calculate_sparsity();
  }

  /// Copies a.values[J,I] into result storage slot @p Idx (flat row-major
  /// index @c Result::indices()[Idx] == I*Result::cols+J). Iterating the
  /// result's own sparsity array (rather than the full rows*cols grid) keeps
  /// instantiation count and compile-time work proportional to the result's
  /// non-zero count instead of its dimensions.
  template<SparseMatrixType Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r, const SparseMat& a) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr Int I = flat / Result::cols;
    constexpr Int J = flat % Result::cols;
    constexpr auto a_index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(J, I);
    static_assert(a_index >= 0,
                  "Transpose index mismatch: expected non-zero element in the original matrix.");
    r.values[Idx] = a.values[a_index];
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const SparseMat& a,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, a), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h for why this is
  /// neither one flat fold (clang caps those at 256 terms) nor a per-element
  /// linear recursion (that exhausts the instantiation-depth budget).
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(Result& r, const SparseMat& a) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, a, std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, a);
      fill_range<Result, Begin + half, Count - half>(r, a);
    }
  }

  /// Constructs the transposed SparseMat and fills it via fill_all.
  SPARSEMAT_HD static auto transpose(const SparseMat& a) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseLinearAlgebra::MatrixUtilities<SparseMat>::
        template make<SparseMat::cols, SparseMat::rows, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(result, a);
    return result;
  }
};
}  // namespace SparseLinearAlgebra::detail

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
 * @return           Transposed matrix of type @c SparseMat<DType, IType, Cols, Rows, ...>.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto transpose(const SparseMat& a) {
  return detail::Transpose<SparseMat>::transpose(a);
}

}  // namespace SparseLinearAlgebra
