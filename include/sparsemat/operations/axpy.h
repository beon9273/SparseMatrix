#pragma once

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for the fused operation @c z = alpha*A*x + beta*y.
 *
 * Computes @c alpha*A*x + beta*y in a single traversal, without materializing
 * the intermediate product @c A*x as its own @c SparseMat. @c A is a sparse
 * matrix, @c x is a sparse column vector compatible with @c A's columns,
 * and @c y is a sparse column vector compatible with @c A's rows. Result
 * sparsity is the union of @c A*x's structural sparsity and @c y's
 * sparsity (independent of the runtime values of @c alpha/@c beta).
 *
 * @tparam SparseMat Sparse matrix type (the @c A operand).
 * @tparam VecX      Sparse column vector type; @c VecX::rows must equal
 *                   @c SparseMat::cols.
 * @tparam VecY      Sparse column vector type; @c VecY::rows must equal
 *                   @c SparseMat::rows.
 */
template<SparseMatrixType SparseMat, SparseMatrixType VecX, SparseMatrixType VecY>
class Axpy {
 public:
  static_assert(VecX::cols == 1 && VecY::cols == 1, "Axpy requires x and y to be column vectors.");
  static_assert(SparseMat::cols == VecX::rows,
                "Axpy requires x's length to match A's column count.");
  static_assert(SparseMat::rows == VecY::rows, "Axpy requires y's length to match A's row count.");
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, VecX> &&
                    SparseLinearAlgebra::SameDataType<SparseMat, VecY>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types \u2014 convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = VecY::cols;

  // Precomputed once — see Add::a_grid/b_grid (add.h) for why.
  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();
  static constexpr auto x_grid = SparseLinearAlgebra::MatrixUtilities<VecX>::to_dense_bool();
  static constexpr auto y_grid = SparseLinearAlgebra::MatrixUtilities<VecY>::to_dense_bool();

  /// Returns true if row @p row of the result (A*x + y) is non-zero: either
  /// A*x has a non-zero contribution there, or y does.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int /*col*/) {
    if (y_grid[row][0]) {
      return true;
    }
    for (Int k = 0; k < SparseMat::cols; ++k) {
      if (a_grid[row][k] && x_grid[k][0]) {
        return true;
      }
    }
    return false;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Axpy>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Axpy>::calculate_sparsity();
  }

  /// Single term of the A[I,*] . x[*] inner product at index k: A[I,k]*x[k]
  /// if both are structurally non-zero, else 0.
  template<Int I, Int k>
  SPARSEMAT_HD static DataType dot_row_term(const SparseMat& a, const VecX& x) {
    if constexpr (SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(I, k) &&
                  SparseLinearAlgebra::MatrixUtilities<VecX>().isNonZero(k, 0)) {
      constexpr auto a_index =
          SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, k);
      constexpr auto x_index = SparseLinearAlgebra::MatrixUtilities<VecX>::getSparseIndex(k, 0);
      return a.values[a_index] * x.values[x_index];
    } else {
      return DataType(0);
    }
  }

  /// Compile-time accumulation of A[I,*] . x[*] via a fold over all columns
  /// k. The right-fold form (pack + ...) matches the original recursion's
  /// right-associated summation order exactly.
  template<Int I, std::size_t... Ks>
  SPARSEMAT_HD static DataType dot_row_fold(const SparseMat& a,
                                            const VecX& x,
                                            std::index_sequence<Ks...> /*seq*/) {
    return (dot_row_term<I, Ks>(a, x) + ...);
  }

  template<Int I>
  SPARSEMAT_HD static DataType dot_row(const SparseMat& a, const VecX& x) {
    return dot_row_fold<I>(a,
                           x,
                           std::make_index_sequence<static_cast<std::size_t>(SparseMat::cols)>{});
  }

  /// Fills result storage slot @p Idx (whose row is @c Result::indices()[Idx],
  /// since the result is a column vector) with alpha*dot_row(A,x) + beta*y[I].
  template<SparseMatrixType Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(
      Result& r, const SparseMat& a, const VecX& x, const VecY& y, DataType alpha, DataType beta) {
    constexpr Int I = Result::indices()[Idx];
    DataType value = alpha * dot_row<I>(a, x);
    constexpr auto y_index = SparseLinearAlgebra::MatrixUtilities<VecY>::getSparseIndex(I, 0);
    if constexpr (y_index >= 0) {
      value += beta * y.values[y_index];
    }
    r.values[Idx] = value;
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const SparseMat& a,
                                      const VecX& x,
                                      const VecY& y,
                                      DataType alpha,
                                      DataType beta,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, a, x, y, alpha, beta), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h for why this is
  /// neither one flat fold (clang caps those at 256 terms) nor a per-element
  /// linear recursion (that exhausts the instantiation-depth budget).
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(
      Result& r, const SparseMat& a, const VecX& x, const VecY& y, DataType alpha, DataType beta) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, a, x, y, alpha, beta, std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, a, x, y, alpha, beta);
      fill_range<Result, Begin + half, Count - half>(r, a, x, y, alpha, beta);
    }
  }

  /// Constructs the result vector and fills it via fill_all, in a single
  /// pass (no separate scaling pass over r.values — that would have to
  /// re-derive each row's sparse storage index anyway, since not every row
  /// is necessarily stored).
  SPARSEMAT_HD static auto axpy(
      const SparseMat& a, const VecX& x, const VecY& y, DataType alpha, DataType beta) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseLinearAlgebra::MatrixUtilities<SparseMat>::
        template make<SparseMat::rows, 1, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(
        result, a, x, y, alpha, beta);
    return result;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Computes @c alpha*A*x + beta*y in a single fused pass, without
 *        materializing the intermediate product @c A*x.
 *
 * @c alpha and @c beta default to @c 1, so @c axpy(a, x, y) alone computes
 * plain @c A*x + y.
 *
 * @tparam A    Sparse matrix type.
 * @tparam VecX Sparse column-vector type; length must equal @c A::cols.
 * @tparam VecY Sparse column-vector type; length must equal @c A::rows.
 * @param  a     Matrix operand.
 * @param  x     Vector multiplied by @p a.
 * @param  y     Vector added to the product.
 * @param  alpha Scalar multiplier for @c A*x.
 * @param  beta  Scalar multiplier for @p y.
 * @return       Result column vector @c alpha*a*x + beta*y.
 */
template<SparseMatrixType A, SparseMatrixType VecX, SparseMatrixType VecY>
SPARSEMAT_HD auto axpy(const A& a,
                       const VecX& x,
                       const VecY& y,
                       typename A::DataType alpha = typename A::DataType(1),
                       typename A::DataType beta = typename A::DataType(1)) {
  return detail::Axpy<A, VecX, VecY>::axpy(a, x, y, alpha, beta);
}

}  // namespace SparseLinearAlgebra
