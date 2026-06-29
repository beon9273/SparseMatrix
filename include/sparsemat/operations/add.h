#pragma once

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for sparse matrix addition and subtraction.
 *
 * Computes @c a + multiplier*b.  Using @c multiplier = 1 gives addition;
 * @c multiplier = -1 gives subtraction.  Result sparsity is the union of both
 * input sparsity patterns.
 *
 * @tparam SparseMat  Left-hand matrix type.
 * @tparam SparseMat1 Right-hand matrix type; must have the same shape as
 *                    @c SparseMat.
 */
template<SparseMatrixType SparseMat, SparseMatrixType SparseMat1>
class Add {
 public:
  static_assert(SparseMat::rows == SparseMat1::rows && SparseMat::cols == SparseMat1::cols,
                "Incompatible matrix dimensions for addition.");

  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat1::cols;

  /// Returns true if (row, col) is non-zero in either input matrix.
  constexpr static auto is_result_index_nonzero(std::size_t row, std::size_t col) {
    return (SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(row, col) ||
            SparseLinearAlgebra::MatrixUtilities<SparseMat1>().isNonZero(row, col));
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Add>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Add>::calculate_sparsity();
  }

  /// Recursively fills result positions (I,J) with a[I,J] + multiplier*b[I,J].
  template<SparseMatrixType Result, std::size_t I, std::size_t J>
  static void recursive_add(Result& r,
                            const SparseMat& a,
                            const SparseMat1& b,
                            const DataType multiplier) {
    if constexpr (I >= SparseMat::rows) {
      return;
    } else if constexpr (J >= SparseMat1::cols) {
      return recursive_add<Result, I + 1, 0>(r, a, b, multiplier);
    } else {
      constexpr int sparse_index =
          SparseLinearAlgebra::MatrixUtilities<Result>::getSparseIndex(I, J);
      if constexpr (sparse_index >= 0) {
        constexpr int a_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
        constexpr int b_index =
            SparseLinearAlgebra::MatrixUtilities<SparseMat1>::getSparseIndex(I, J);

        r.values[sparse_index] = 0;
        if constexpr (a_index >= 0) {
          r.values[sparse_index] = a.values[a_index];
        }
        if constexpr (b_index >= 0) {
          r.values[sparse_index] += multiplier * b.values[b_index];
        }
      }
      recursive_add<Result, I, J + 1>(r, a, b, multiplier);
    }
  }

  /// Constructs the result SparseMat and fills it via recursive_add.
  static auto add(const SparseMat& a, const SparseMat1& b, const DataType multiplier) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseMat::template make<SparseMat::rows, SparseMat1::cols, sparsity>(
        std::make_index_sequence<num_nonzeros()>{});
    recursive_add<decltype(result), 0, 0>(result, a, b, multiplier);
    return result;
  }
};
}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Element-wise addition of two sparse matrices: @p a + @p b.
 *
 * Result sparsity is the union of both input patterns.  Both matrices must
 * have identical dimensions.
 *
 * @tparam A Left-hand matrix type.
 * @tparam B Right-hand matrix type.
 * @param  a Left-hand operand.
 * @param  b Right-hand operand.
 * @return   Sum matrix whose sparsity covers every non-zero in either input.
 */
template<SparseMatrixType A, SparseMatrixType B>
auto add(const A& a, const B& b) {
  return detail::Add<A, B>::add(a, b, 1);
}

/**
 * @brief Element-wise subtraction of two sparse matrices: @p a - @p b.
 *
 * Equivalent to @c scaled_add(a, b, -1).  Result sparsity is the union of
 * both input patterns.
 *
 * @tparam A Left-hand matrix type.
 * @tparam B Right-hand matrix type.
 * @param  a Minuend.
 * @param  b Subtrahend.
 * @return   Difference matrix.
 */
template<SparseMatrixType A, SparseMatrixType B>
auto subtract(const A& a, const B& b) {
  return detail::Add<A, B>::add(a, b, -1);
}

/**
 * @brief Computes @p a + @p multiplier × @p b in a single fused operation.
 *
 * Avoids a separate scale pass; @p multiplier is applied element-wise to
 * @p b during the addition traversal.  Result sparsity is the union of both
 * input patterns.
 *
 * @tparam A        Left-hand matrix type.
 * @tparam B        Right-hand matrix type.
 * @tparam DataType Scalar type of the multiplier.
 * @param  a          Base matrix.
 * @param  b          Matrix to scale and add.
 * @param  multiplier Scalar factor applied to @p b before adding.
 * @return            Result matrix @c a + multiplier*b.
 */
template<SparseMatrixType A, SparseMatrixType B, MatrixDataType DataType>
auto scaled_add(const A& a, const B& b, const DataType multiplier) {
  return detail::Add<A, B>::add(a, b, multiplier);
}

}  // namespace SparseLinearAlgebra
