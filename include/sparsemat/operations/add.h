#pragma once

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for sparse matrix addition, subtraction, and
 *        scaled addition.
 *
 * Computes @c alpha*a + beta*b.  Using @c alpha = beta = 1 gives addition;
 * @c alpha = 1, @c beta = -1 gives subtraction; any other combination gives
 * a general scaled addition without a separate scale pass over either
 * operand.  Result sparsity is the union of both input sparsity patterns
 * (independent of the runtime values of @c alpha/@c beta).
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
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, SparseMat1>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types \u2014 convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;

  // Precomputed once per (SparseMat, SparseMat1) instantiation — not per
  // is_result_index_nonzero call — so each of the O(rows*cols) calls
  // OperationUtilities makes below is an O(1) grid lookup instead of two
  // O(nonZeroCount) linear scans. See MatrixUtilities::to_dense_bool().
  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();
  static constexpr auto b_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat1>::to_dense_bool();

  /// Returns true if (row, col) is non-zero in either input matrix.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return a_grid[row][col] || b_grid[row][col];
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Add>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Add>::calculate_sparsity();
  }

  /// Fills result storage slot @p Idx (whose flat row-major index is
  /// @c Result::indices()[Idx]) with alpha*a[I,J] + beta*b[I,J]. Iterating
  /// only over the result's own sparsity array — rather than recursing over
  /// every (I,J) in the full rows*cols grid — keeps both the number of
  /// template instantiations and the compile-time work proportional to the
  /// result's non-zero count instead of its dimensions.
  template<SparseMatrixType Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r,
                                     const SparseMat& a,
                                     const SparseMat1& b,
                                     const DataType alpha,
                                     const DataType beta) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr Int I = flat / Result::cols;
    constexpr Int J = flat % Result::cols;
    constexpr auto a_index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
    constexpr auto b_index = SparseLinearAlgebra::MatrixUtilities<SparseMat1>::getSparseIndex(I, J);

    DataType value = 0;
    if constexpr (a_index >= 0) {
      value = alpha * a.values[a_index];
    }
    if constexpr (b_index >= 0) {
      value += beta * b.values[b_index];
    }
    r.values[Idx] = value;
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const SparseMat& a,
                                      const SparseMat1& b,
                                      const DataType alpha,
                                      const DataType beta,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, a, b, alpha, beta), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h for why this is
  /// neither one flat fold (clang caps those at 256 terms) nor a per-element
  /// linear recursion (that exhausts the instantiation-depth budget).
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(Result& r,
                                      const SparseMat& a,
                                      const SparseMat1& b,
                                      const DataType alpha,
                                      const DataType beta) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, a, b, alpha, beta, std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, a, b, alpha, beta);
      fill_range<Result, Begin + half, Count - half>(r, a, b, alpha, beta);
    }
  }

  /// Constructs the result SparseMat and fills it via fill_all.
  SPARSEMAT_HD static auto add(const SparseMat& a,
                               const SparseMat1& b,
                               const DataType alpha,
                               const DataType beta) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseLinearAlgebra::MatrixUtilities<SparseMat>::
        template make<SparseMat::rows, SparseMat::cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(
        result, a, b, alpha, beta);
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
SPARSEMAT_HD auto add(const A& a, const B& b) {
  return detail::Add<A, B>::add(a, b, 1, 1);
}

/**
 * @brief Element-wise subtraction of two sparse matrices: @p a - @p b.
 *
 * Equivalent to @c add(a, b, 1, -1).  Result sparsity is the union of both
 * input patterns.
 *
 * @tparam A Left-hand matrix type.
 * @tparam B Right-hand matrix type.
 * @param  a Minuend.
 * @param  b Subtrahend.
 * @return   Difference matrix.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto subtract(const A& a, const B& b) {
  return detail::Add<A, B>::add(a, b, 1.0, -1.0);
}

/**
 * @brief Computes @p alpha*a + @p beta*b in a single fused operation.
 *
 * Avoids separate scale passes; @p alpha and @p beta are applied
 * element-wise to @p a and @p b respectively during the same traversal that
 * computes the sum.  Result sparsity is the union of both input patterns.
 *
 * @tparam A        Left-hand matrix type.
 * @tparam B        Right-hand matrix type.
 * @tparam DataType Scalar type of @p alpha and @p beta.
 * @param  a     Left-hand operand.
 * @param  b     Right-hand operand.
 * @param  alpha Scalar factor applied to @p a.
 * @param  beta  Scalar factor applied to @p b.
 * @return       Result matrix @c alpha*a + beta*b.
 */
template<SparseMatrixType A, SparseMatrixType B, MatrixDataType DataType>
SPARSEMAT_HD auto add(const A& a, const B& b, const DataType alpha, const DataType beta) {
  return detail::Add<A, B>::add(a, b, alpha, beta);
}

/**
 * @brief Computes @p a + @p beta*b in a single fused operation.
 *
 * Convenience overload equivalent to @c add(a, b, 1, beta): @p alpha is
 * implicitly @c 1, so only @p b is scaled.  Avoids a separate scale pass;
 * @p beta is applied element-wise to @p b during the addition traversal.
 * Result sparsity is the union of both input patterns.
 *
 * @tparam A        Left-hand matrix type.
 * @tparam B        Right-hand matrix type.
 * @tparam DataType Scalar type of @p beta.
 * @param  a    Base matrix, added unscaled.
 * @param  b    Matrix to scale and add.
 * @param  beta Scalar factor applied to @p b before adding.
 * @return      Result matrix @c a + beta*b.
 */
template<SparseMatrixType A, SparseMatrixType B, MatrixDataType DataType>
SPARSEMAT_HD auto add(const A& a, const B& b, const DataType beta) {
  return detail::Add<A, B>::add(a, b, 1.0, beta);
}

}  // namespace SparseLinearAlgebra
