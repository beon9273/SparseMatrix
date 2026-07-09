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

  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat1::cols;

  /// Returns true only when (row, col) is non-zero in both input matrices.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return (SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(row, col) &&
            SparseLinearAlgebra::MatrixUtilities<SparseMat1>().isNonZero(row, col));
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Hadamard>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Hadamard>::calculate_sparsity();
  }

  /// Recursively fills result positions with a[I,J] * b[I,J] * multiplier.
  template<SparseMatrixType Result, Int I, Int J>
  SPARSEMAT_HD static void recursive_hadamard(Result& r,
                                              const SparseMat& a,
                                              const SparseMat1& b,
                                              const DataType multiplier) {
    if constexpr (I >= SparseMat::rows) {
      return;
    } else if constexpr (J >= SparseMat1::cols) {
      return recursive_hadamard<Result, I + 1, 0>(r, a, b, multiplier);
    } else {
      constexpr auto sparse_index =
          SparseLinearAlgebra::MatrixUtilities<Result>::getSparseIndex(I, J);
      if constexpr (sparse_index >= 0) {
        constexpr auto a_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
        constexpr auto b_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat1>::getSparseIndex(I, J);
        static_assert(a_index >= 0 && b_index >= 0,
                      "Invalid sparse indices for Hadamard operation.");
        r.values[sparse_index] = a.values[a_index] * b.values[b_index] * multiplier;
      }
      recursive_hadamard<Result, I, J + 1>(r, a, b, multiplier);
    }
  }

  /// Constructs the result SparseMat and fills it via recursive_hadamard.
  SPARSEMAT_HD static auto hadamard(const SparseMat& a,
                                    const SparseMat1& b,
                                    const DataType multiplier) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
            std::make_index_sequence<num_nonzeros()>{});
    recursive_hadamard<decltype(result), 0, 0>(result, a, b, multiplier);
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
