#pragma once

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for the Hadamard (element-wise) product.
 *
 * Result sparsity is the intersection of both input patterns: only positions
 * that are non-zero in @em both matrices appear in the result.  An optional
 * scalar multiplier is applied in the same pass to avoid an extra traversal.
 *
 * @tparam SparseMat  Left-hand matrix type.
 * @tparam SparseMat1 Right-hand matrix type; must have the same shape as
 *                    @c SparseMat.
 */
template<SparseMatrixType SparseMat, SparseMatrixType SparseMat1>
class Hadamard {
 public:
  static_assert(SparseMat::rows == SparseMat1::rows && SparseMat::cols == SparseMat1::cols,
                "Incompatible matrix dimensions for Hadamard operation.");
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, SparseMat1>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types \u2014 convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;

  // Precomputed once — see Add::a_grid/b_grid for why (identical reasoning).
  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();
  static constexpr auto b_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat1>::to_dense_bool();

  /// Returns true only when (row, col) is non-zero in both input matrices.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return a_grid[row][col] && b_grid[row][col];
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Hadamard>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Hadamard>::calculate_sparsity();
  }

  /// Fills result storage slot @p Idx (flat row-major index
  /// @c Result::indices()[Idx]) with a[I,J] * b[I,J] * multiplier. Iterating
  /// the result's own sparsity array (rather than the full rows*cols grid)
  /// keeps instantiation count and compile-time work proportional to the
  /// result's non-zero count instead of its dimensions.
  template<SparseMatrixType Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r,
                                     const SparseMat& a,
                                     const SparseMat1& b,
                                     const DataType multiplier) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr Int I = flat / Result::cols;
    constexpr Int J = flat % Result::cols;
    constexpr auto a_index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
    constexpr auto b_index = SparseLinearAlgebra::MatrixUtilities<SparseMat1>::getSparseIndex(I, J);
    static_assert(a_index >= 0 && b_index >= 0, "Invalid sparse indices for Hadamard operation.");
    r.values[Idx] = a.values[a_index] * b.values[b_index] * multiplier;
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const SparseMat& a,
                                      const SparseMat1& b,
                                      const DataType multiplier,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, a, b, multiplier), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h for why this is
  /// neither one flat fold (clang caps those at 256 terms) nor a per-element
  /// linear recursion (that exhausts the instantiation-depth budget).
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(Result& r,
                                      const SparseMat& a,
                                      const SparseMat1& b,
                                      const DataType multiplier) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, a, b, multiplier, std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, a, b, multiplier);
      fill_range<Result, Begin + half, Count - half>(r, a, b, multiplier);
    }
  }

  /// Constructs the result SparseMat and fills it via fill_all.
  SPARSEMAT_HD static auto hadamard(const SparseMat& a,
                                    const SparseMat1& b,
                                    const DataType multiplier) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(result,
                                                                              a,
                                                                              b,
                                                                              multiplier);
    return result;
  }
};
}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Element-wise (Hadamard) product of two sparse matrices: @p a ⊙ @p b.
 *
 * Result sparsity is the intersection of both input patterns — only positions
 * that are non-zero in both matrices are stored in the result.  Both matrices
 * must have the same dimensions.
 *
 * @tparam A Left-hand matrix type.
 * @tparam B Right-hand matrix type.
 * @param  a Left-hand operand.
 * @param  b Right-hand operand.
 * @return   Element-wise product matrix.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto hadamard(const A& a, const B& b) {
  return detail::Hadamard<A, B>::hadamard(a, b, 1);
}

/**
 * @brief Fused element-wise product and scalar scale: (@p a ⊙ @p b) × @p multiplier.
 *
 * Equivalent to @c hadamard(a,b) followed by @c scale(..., multiplier), but
 * computed in a single pass.  Result sparsity is the intersection of both
 * input patterns.
 *
 * @tparam A        Left-hand matrix type.
 * @tparam B        Right-hand matrix type.
 * @tparam DataType Scalar type of the multiplier.
 * @param  a          Left-hand operand.
 * @param  b          Right-hand operand.
 * @param  multiplier Scalar applied to every result element after the product.
 * @return            Scaled element-wise product matrix.
 */
template<SparseMatrixType A, SparseMatrixType B, MatrixDataType DataType>
SPARSEMAT_HD auto hadamard(const A& a, const B& b, const DataType multiplier) {
  return detail::Hadamard<A, B>::hadamard(a, b, multiplier);
}

}  // namespace SparseLinearAlgebra
