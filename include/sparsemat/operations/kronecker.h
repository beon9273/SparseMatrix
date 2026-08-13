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
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, SparseMat1>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types \u2014 convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows * SparseMat1::rows;
  static constexpr auto cols = SparseMat::cols * SparseMat1::cols;

  // Precomputed once — see Add::a_grid/b_grid for why (identical reasoning).
  // Kronecker's result grid is the PRODUCT of both operands' dimensions,
  // making its O(rows*cols) sparsity-computation calls the most numerous of
  // any operation in this library — so this matters here more than anywhere
  // else.
  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();
  static constexpr auto b_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat1>::to_dense_bool();

  /// Returns true when (row, col) in the result is non-zero.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return a_grid[row / SparseMat1::rows][col / SparseMat1::cols] &&
           b_grid[row % SparseMat1::rows][col % SparseMat1::cols];
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Kronecker>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Kronecker>::calculate_sparsity();
  }

  /// Fills result storage slot @p Idx (flat row-major index
  /// @c Result::indices()[Idx]) with a[I/B.rows, J/B.cols] * b[I%B.rows,
  /// J%B.cols]. Kronecker's result grid grows as the PRODUCT of both
  /// operands' dimensions, making it the fastest-growing operation in the
  /// library — iterating the result's own sparsity array (rather than the
  /// full result_rows*result_cols grid) is what keeps both instantiation
  /// count and compile-time work proportional to the result's non-zero
  /// count instead of that product.
  template<SparseMatrixType Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r, const SparseMat& a, const SparseMat1& b) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr Int I = flat / Result::cols;
    constexpr Int J = flat % Result::cols;
    constexpr auto a_index =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I / SparseMat1::rows,
                                                                        J / SparseMat1::cols);
    constexpr auto b_index =
        SparseLinearAlgebra::MatrixUtilities<SparseMat1>::getSparseIndex(I % SparseMat1::rows,
                                                                         J % SparseMat1::cols);
    static_assert(a_index >= 0 && b_index >= 0, "Invalid sparse indices for Kronecker operation.");
    r.values[Idx] = a.values[a_index] * b.values[b_index];
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
  SPARSEMAT_HD static auto kronecker(const SparseMat& a, const SparseMat1& b) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<
        SparseMat::rows * SparseMat1::rows,
        SparseMat::cols * SparseMat1::cols,
        sparsity>(std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(result, a, b);
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
