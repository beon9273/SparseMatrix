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
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, SparseMat1>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types \u2014 convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat1::cols;

  // Precomputed once per (SparseMat, SparseMat1) instantiation — not per
  // is_result_index_nonzero call. This one matters most of all the
  // operations using this pattern: without it, every one of the O(rows*cols)
  // calls OperationUtilities makes below does its own O(SparseMat::cols)
  // loop, each iteration paying two O(nonZeroCount) linear scans — an
  // O(rows*cols*cols*nonZeroCount) total that's what actually exhausted a
  // compiler's constexpr-evaluation budget at a mere 40x40. With the grids,
  // each iteration of the k-loop below is an O(1) lookup, cutting the total
  // to O(rows*cols*cols) (plus O(rows*cols) once, to build the grids).
  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();
  static constexpr auto b_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat1>::to_dense_bool();

  /**
   * @brief The result's structural non-zero pattern, computed once.
   *
   * Derived by walking A's *stored* entries rather than the result grid: for
   * each stored A[i,k], every non-zero B[k,j] contributes a term to result
   * position (i,j). That makes this O(nonZeroCount(A) * cols) once — versus
   * the O(rows*cols*cols) it costs to answer the same question by testing
   * every (row, col) against every shared index k, which is what the
   * per-position loop this replaced did (and OperationUtilities calls that
   * predicate rows*cols times, twice over).
   *
   * The difference is not academic. On the 60x60 tridiagonal case in the test
   * suite the old form ran ~216,000 constexpr iterations per walk, which sits
   * right at clang's default -fconstexpr-steps ceiling of 1048576 — close
   * enough that unrelated changes elsewhere would tip it over into "non-type
   * template argument is not a constant expression", a diagnostic that gives
   * no hint of the real cause. This form runs ~10,700, restoring a wide
   * margin.
   */
  SPARSEMAT_HD constexpr static auto compute_result_grid() {
    std::array<std::array<bool, static_cast<std::size_t>(cols)>, static_cast<std::size_t>(rows)>
        grid{};
    for (auto flat : SparseMat::indices()) {
      const Int i = flat / SparseMat::cols;
      const Int k = flat % SparseMat::cols;
      for (Int j = 0; j < cols; ++j) {
        if (b_grid[k][j]) {
          grid[i][j] = true;
        }
      }
    }
    return grid;
  }
  static constexpr auto result_grid = compute_result_grid();

  /// Returns true if at least one shared k makes both A[row,k] and B[k,col] non-zero.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return result_grid[row][col];
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Multiply>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Multiply>::calculate_sparsity();
  }

  /// Single term of the A[I,*] · B[*,J] inner product at shared index k:
  /// A[I,k]*B[k,J] if both are structurally non-zero, else 0.
  template<Int I, Int J, Int k>
  SPARSEMAT_HD static DataType inner_product_term(const SparseMat& a, const SparseMat1& b) {
    if constexpr (SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(I, k) &&
                  SparseLinearAlgebra::MatrixUtilities<SparseMat1>().isNonZero(k, J)) {
      constexpr auto a_index =
          SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, k);
      constexpr auto b_index =
          SparseLinearAlgebra::MatrixUtilities<SparseMat1>::getSparseIndex(k, J);
      return a.values[a_index] * b.values[b_index];
    } else {
      return DataType(0);
    }
  }

  /// Compile-time accumulation of A[I,*] · B[*,J] via a fold over all shared
  /// indices k. The right-fold form (pack + ...) matches the original
  /// recursion's right-associated summation order (term(0) + (term(1) +
  /// (term(2) + ...))) exactly, so this is not just numerically equivalent
  /// but bit-for-bit identical.
  template<Int I, Int J, std::size_t... Ks>
  SPARSEMAT_HD static DataType inner_product_fold(const SparseMat& a,
                                                  const SparseMat1& b,
                                                  std::index_sequence<Ks...> /*seq*/) {
    return (inner_product_term<I, J, Ks>(a, b) + ...);
  }

  template<Int I, Int J>
  SPARSEMAT_HD static DataType do_inner_product(const SparseMat& a, const SparseMat1& b) {
    return inner_product_fold<I, J>(
        a, b, std::make_index_sequence<static_cast<std::size_t>(SparseMat::cols)>{});
  }

  /// Fills result storage slot @p Idx (whose flat row-major index is
  /// @c Result::indices()[Idx]) with the corresponding inner product. See
  /// add.h's fill_cell for why iterating the result's own sparsity array
  /// (rather than the full rows*cols grid) is both correct and preferable
  /// here: multiply only ever writes to result cells this array enumerates.
  template<typename Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r, const SparseMat& a, const SparseMat1& b) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr Int I = flat / Result::cols;
    constexpr Int J = flat % Result::cols;
    r.values[Idx] = do_inner_product<I, J>(a, b);
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const SparseMat& a,
                                      const SparseMat1& b,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, a, b), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h for why this is
  /// neither one flat fold (clang caps those at 256 terms) nor a per-element
  /// linear recursion (that exhausts the instantiation-depth budget).
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(Result& r, const SparseMat& a, const SparseMat1& b) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, a, b, std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, a, b);
      fill_range<Result, Begin + half, Count - half>(r, a, b);
    }
  }

  /// Constructs the result SparseMat and fills it via fill_all.
  SPARSEMAT_HD static auto multiply(const SparseMat& a, const SparseMat1& b) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseLinearAlgebra::MatrixUtilities<SparseMat>::
        template make<SparseMat::rows, SparseMat1::cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(result, a, b);
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
SPARSEMAT_HD auto multiply(const A& a, const B& b) {
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
SPARSEMAT_HD auto power(const A& a) {
  static_assert(N > 0, "Matrix exponent must be greater than 0");
  if constexpr (N == 1) {
    return a;
  } else {
    return multiply(a, power<A, N - 1>(a));
  }
}

}  // namespace SparseLinearAlgebra
