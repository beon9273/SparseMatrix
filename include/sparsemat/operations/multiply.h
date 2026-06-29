#pragma once

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for sparse matrix multiplication A × B.
 *
 * Result sparsity: position (row, col) is non-zero when there exists at least
 * one shared index @c k such that A[row,k] and B[k,col] are both non-zero.
 * All sparsity decisions are made at compile time.
 *
 * @tparam SparseMat  Left-hand matrix type.
 * @tparam SparseMat1 Right-hand matrix type; @c SparseMat::cols must equal
 *                    @c SparseMat1::rows.
 */
template<SparseMatrixType SparseMat, SparseMatrixType SparseMat1>
class Multiply {
 public:
  static_assert(SparseMat::cols == SparseMat1::rows,
                "Incompatible matrix dimensions for multiplication.");

  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat1::cols;

  /// Returns true if at least one shared k makes both A[row,k] and B[k,col] non-zero.
  constexpr static auto is_result_index_nonzero(std::size_t row, std::size_t col) {
    for (std::size_t i = 0; i < SparseMat::cols; i++) {
      if (SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(row, i) &&
          SparseLinearAlgebra::MatrixUtilities<SparseMat1>().isNonZero(i, col)) {
        return true;
      }
    }
    return false;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Multiply>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Multiply>::calculate_sparsity();
  }

  /// Compile-time accumulation of A[I,*] · B[*,J] over non-zero pairs at column k=i.
  template<std::size_t I, std::size_t J, std::size_t i = 0>
  static DataType do_inner_product(const SparseMat& a, const SparseMat1& b) {
    if constexpr (i >= SparseMat::cols) {
      return 0;
    } else {
      if constexpr (SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(I, i) &&
                    SparseLinearAlgebra::MatrixUtilities<SparseMat1>().isNonZero(i, J)) {
        constexpr int a_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, i);
        constexpr int b_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat1>::getSparseIndex(i, J);
        return (a.values[a_index] * b.values[b_index]) + do_inner_product<I, J, i + 1>(a, b);
      } else {
        return do_inner_product<I, J, i + 1>(a, b);
      }
    }
  }

  /// Recursively fills result positions (I,J) in row-major order.
  template<typename Result, std::size_t I, std::size_t J>
  static void recursive_multiply(Result& r, const SparseMat& a, const SparseMat1& b) {
    if constexpr (I >= SparseMat::rows) {
      return;
    } else if constexpr (J >= SparseMat1::cols) {
      return recursive_multiply<Result, I + 1, 0>(r, a, b);
    } else {
      constexpr int sparse_index =
          SparseLinearAlgebra::MatrixUtilities<Result>::getSparseIndex(I, J);
      if constexpr (sparse_index >= 0) {
        r.values[sparse_index] = do_inner_product<I, J>(a, b);
      }
      recursive_multiply<Result, I, J + 1>(r, a, b);
    }
  }

  /// Constructs the result SparseMat and fills it via recursive_multiply.
  static auto multiply(const SparseMat& a, const SparseMat1& b) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseMat::template make<SparseMat::rows, SparseMat1::cols, sparsity>(
        std::make_index_sequence<num_nonzeros()>{});
    recursive_multiply<decltype(result), 0, 0>(result, a, b);
    return result;
  }
};
}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Multiplies two sparse matrices: @p a × @p b.
 *
 * Result sparsity is computed at compile time from the intersection of column
 * non-zeros in @p a and row non-zeros in @p b.  The inner dimension must match
 * (@c A::cols == @c B::rows).
 *
 * @tparam A Left-hand matrix type.
 * @tparam B Right-hand matrix type.
 * @param  a Left-hand matrix.
 * @param  b Right-hand matrix.
 * @return   Product matrix whose type encodes the result sparsity pattern.
 */
template<SparseMatrixType A, SparseMatrixType B>
auto multiply(const A& a, const B& b) {
  return detail::Multiply<A, B>::multiply(a, b);
}

/**
 * @brief Raises a square sparse matrix to the power @p N via repeated multiplication.
 *
 * Uses compile-time recursion: @c power<1>(a) returns @p a unchanged;
 * @c power<N>(a) returns @c multiply(a, power<N-1>(a)).  Note that each
 * intermediate product may have a different (wider) sparsity pattern, so the
 * return type changes with @p N.
 *
 * @tparam A Type of the input matrix (must be square).
 * @tparam N Exponent; must be greater than 0.
 * @param  a Square sparse matrix to exponentiate.
 * @return   @p a raised to the @p N-th power.
 */
template<SparseMatrixType A, int N>
auto power(const A& a) {
  static_assert(N > 0, "Matrix exponent must be greater than 0");
  if constexpr (N == 1) {
    return a;
  } else {
    return multiply(a, power<A, N - 1>(a));
  }
}

}  // namespace SparseLinearAlgebra
