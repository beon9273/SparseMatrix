#pragma once

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for the Kronecker (tensor) product.
 *
 * The result is an (A.rows*B.rows) × (A.cols*B.cols) matrix where each
 * element of A is replaced by a scaled copy of B.  Result position (i,j) is
 * non-zero iff A[i/B.rows, j/B.cols] and B[i%B.rows, j%B.cols] are both
 * non-zero.
 *
 * @tparam SparseMat  Left-hand matrix type.
 * @tparam SparseMat1 Right-hand matrix type.
 */
template<SparseMatrixType SparseMat, SparseMatrixType SparseMat1>
class Kronecker {
 public:
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows * SparseMat1::rows;
  static constexpr auto cols = SparseMat::cols * SparseMat1::cols;

  /// Returns true when (row, col) in the result is non-zero.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(row / SparseMat1::rows,
                                                                      col / SparseMat1::cols) &&
           SparseLinearAlgebra::MatrixUtilities<SparseMat1>::isNonZero(row % SparseMat1::rows,
                                                                       col % SparseMat1::cols);
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Kronecker>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Kronecker>::calculate_sparsity();
  }

  /// Recursively fills result positions with a[I/B.rows, J/B.cols] * b[I%B.rows, J%B.cols].
  template<SparseMatrixType Result, Int I, Int J>
  SPARSEMAT_HD static void recursive_kronecker(Result& r, const SparseMat& a, const SparseMat1& b) {
    if constexpr (I >= SparseMat::rows * SparseMat1::rows) {
      return;
    } else if constexpr (J >= SparseMat::cols * SparseMat1::cols) {
      return recursive_kronecker<Result, I + 1, 0>(r, a, b);
    } else {
      constexpr auto sparse_index =
          SparseLinearAlgebra::MatrixUtilities<Result>::getSparseIndex(I, J);
      if constexpr (sparse_index >= 0) {
        constexpr auto a_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I / SparseMat1::rows,
                                                                            J / SparseMat1::cols);
        constexpr auto b_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat1>::getSparseIndex(I % SparseMat1::rows,
                                                                             J % SparseMat1::cols);
        static_assert(a_index >= 0 && b_index >= 0,
                      "Invalid sparse indices for Kronecker operation.");
        r.values[sparse_index] = a.values[a_index] * b.values[b_index];
      }
      recursive_kronecker<Result, I, J + 1>(r, a, b);
    }
  }

  /// Constructs the result SparseMat and fills it via recursive_kronecker.
  SPARSEMAT_HD static auto kronecker(const SparseMat& a, const SparseMat1& b) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<
        SparseMat::rows * SparseMat1::rows,
        SparseMat::cols * SparseMat1::cols,
        sparsity>(std::make_index_sequence<num_nonzeros()>{});
    recursive_kronecker<decltype(result), 0, 0>(result, a, b);
    return result;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Kronecker (tensor) product of two sparse matrices: @p a ⊗ @p b.
 *
 * Produces an (a.rows*b.rows) × (a.cols*b.cols) matrix where each non-zero
 * element of @p a is replaced by a scaled copy of @p b.  Result sparsity is
 * computed at compile time as the outer product of both sparsity patterns.
 *
 * @tparam A Left-hand matrix type.
 * @tparam B Right-hand matrix type.
 * @param  a Left-hand operand.
 * @param  b Right-hand operand.
 * @return   Kronecker product matrix.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto kronecker(const A& a, const B& b) {
  return detail::Kronecker<A, B>::kronecker(a, b);
}

}  // namespace SparseLinearAlgebra
