#pragma once
#include <cmath>
#include <stdexcept>

#include "operations/utils.h"

namespace {

/**
 * @brief Implementation policy for analytically inverting a 2×2 sparse matrix.
 *
 * Computes the inverse using the closed-form formula:
 * @code
 *   [a b]^{-1}  =  (1/det) * [ e  -b]
 *   [c e]                     [-c   a]
 * @endcode
 * where @c det = a*e - b*c.
 *
 * The result sparsity is derived at compile time: a result position is
 * non-zero only when the corresponding source element that contributes to it
 * is non-zero in the input sparsity pattern.  For example, result(0,0) comes
 * from input(1,1), so it is only non-zero when input(1,1) is non-zero.
 *
 * @tparam SparseMat A 2×2 sparse matrix type satisfying @c SparseMatrixType.
 */
template<SparseMatrixType SparseMat>
class Invert2x2 {
 public:
  static_assert(SparseMat::rows == SparseMat::cols && SparseMat::rows == 2,
                "Invert2x2 only supports 2x2 matrices.");
  using DataType = typename SparseMat::DataType;
  static constexpr std::size_t rows = SparseMat::rows;
  static constexpr std::size_t cols = SparseMat::cols;

  /**
   * @brief Returns true when the result position (@p row, @p col) is non-zero.
   *
   * Each result element is non-zero iff the source element it comes from is
   * non-zero in the input sparsity pattern:
   *  - (0,0) ← input(1,1)
   *  - (0,1) ← input(0,1)
   *  - (1,0) ← input(1,0)
   *  - (1,1) ← input(0,0)
   */
  constexpr static auto is_result_index_nonzero(std::size_t row, std::size_t col) {
    if (row == 0 && col == 0) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(1, 1);
    } 
    if (row == 0 && col == 1) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(0, 1);
    } 
    if (row == 1 && col == 0) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(1, 0);
    }
    if (row == 1 && col == 1) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMat>().isNonZero(0, 0);
    }
    return false;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Invert2x2>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Invert2x2>::calculate_sparsity();
  }

  /**
   * @brief Computes and returns the inverse of @p a.
   *
   * Throws @c std::runtime_error if the matrix is singular (determinant zero).
   * The result type has a sparsity pattern derived from the input: positions
   * whose contributing source element is structurally zero are omitted.
   *
   * @param a  The 2×2 matrix to invert.
   * @return   Inverse of @p a with compile-time sparsity.
   */
  static auto invert(const SparseMat& a) {
    auto det = (a.template get<0, 0>() * a.template get<1, 1>()) -
               (a.template get<0, 1>() * a.template get<1, 0>());
    if (det == 0) {
      throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }
    auto inv_det = 1 / det;
    constexpr auto sparsity =
        SparseLinearAlgebra::OperationUtilities<Invert2x2>::calculate_sparsity();
    auto result =
        SparseMat::template make<2, 2, sparsity>(std::make_index_sequence<num_nonzeros()>{});

    if constexpr (is_result_index_nonzero(0, 0)) {
      result.template set<0, 0>(a.template get<1, 1>() * inv_det);
    }
    if constexpr (is_result_index_nonzero(0, 1)) {
      result.template set<0, 1>(-a.template get<0, 1>() * inv_det);
    }
    if constexpr (is_result_index_nonzero(1, 0)) {
      result.template set<1, 0>(-a.template get<1, 0>() * inv_det);
    }
    if constexpr (is_result_index_nonzero(1, 1)) {
      result.template set<1, 1>(a.template get<0, 0>() * inv_det);
    }
    return result;
  }
};
}  // namespace

namespace SparseLinearAlgebra {

/**
 * @brief Returns the analytical inverse of a 2×2 sparse matrix.
 *
 * Uses the closed-form formula for 2×2 matrices.  The result sparsity is
 * computed at compile time from the input pattern.  Throws
 * @c std::runtime_error if the matrix is singular.
 *
 * @tparam SparseMat A 2×2 type satisfying @c SparseMatrixType.
 * @param  a         Matrix to invert.
 * @return           Inverse of @p a.
 */
template<SparseMatrixType SparseMat>
auto invert2x2(const SparseMat& a) {
  return Invert2x2<SparseMat>::invert(a);
}

}  // namespace SparseLinearAlgebra
