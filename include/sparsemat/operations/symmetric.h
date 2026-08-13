#pragma once
#include <cmath>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"
namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for determining symmetry of a sparse matrix.
 *
 *
 * @tparam SparseMat The matrix type to analyze.
 */
template<typename SparseMat>
class Symmetric {
 public:
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;
  static constexpr auto num_non_zeros = SparseMat::nonZeroCount;
  static constexpr auto num_zeros = (rows * cols) - num_non_zeros;
  static constexpr auto total_elements = rows * cols;

  // A function rather than a `static constexpr` data member for the same
  // device-code reason as offsets() below: is_structurally_symmetric() is
  // reachable at runtime through the free function of the same name, and a
  // subscripted static member would need device storage it does not get.
  //
  // Memoizing the grid at all matters because the double loop below would
  // otherwise call isNonZero twice per cell, each an O(nonZeroCount) linear
  // scan, for O(rows*cols*nonZeroCount) constexpr work. Every other operation
  // in this library already memoizes this way — see
  // MatrixUtilities::to_dense_bool().
  SPARSEMAT_HD constexpr static auto pattern() {
    return SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();
  }

  SPARSEMAT_HD constexpr static bool is_structurally_symmetric() {
    if constexpr (rows != cols) {
      return false;
    } else {
      constexpr auto grid = pattern();
      for (Int i = 0; i < rows; ++i) {
        for (Int j = i + 1; j < cols; ++j) {
          if (grid[i][j] != grid[j][i]) {
            return false;
          }
        }
      }
      return true;
    }
  }

  // Held as a function rather than a `static constexpr` data member, and
  // copied into a local at each use below. The lookups here happen at
  // *runtime*, so a static member would be ODR-used and would need device
  // storage — which it does not get, giving "identifier ... is undefined in
  // device code" under nvcc. Same reasoning as SparseMat::indices(); contrast
  // Multiply::a_grid, which is only ever read during constant evaluation and
  // so is safe as a static member.
  SPARSEMAT_HD constexpr static auto offsets() {
    return SparseLinearAlgebra::MatrixUtilities<SparseMat>::storage_index_grid();
  }

  SPARSEMAT_HD static bool values_within(DataType lhs, DataType rhs, DataType TOLERANCE) {
    const DataType diff = lhs - rhs;
    return (diff < DataType(0) ? -diff : diff) <= TOLERANCE;
  }

  /**
   * @brief Compares each stored value against its mirror position.
   *
   * Only positions stored in *both* (i, j) and (j, i) are compared, which is
   * why the structural check above is a precondition: it guarantees every
   * stored position has a stored mirror.
   */
  SPARSEMAT_HD static bool is_sparse_symmetric(const SparseMat& a, DataType TOLERANCE = 1e-6) {
    if constexpr (!is_structurally_symmetric()) {
      return false;
    } else {
      constexpr auto inds = SparseMat::indices();
      constexpr auto grid = offsets();
      // Guarded because the bound is a compile-time constant: when it is zero the
      // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
      if constexpr (SparseMat::nonZeroCount != 0) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(SparseMat::nonZeroCount); ++k) {
          const Int row = inds[k] / cols;
          const Int col = inds[k] % cols;
          const auto mirror = grid[static_cast<std::size_t>((col * cols) + row)];
          if (!values_within(a.values[k], a.values[static_cast<std::size_t>(mirror)], TOLERANCE)) {
            return false;
          }
        }
      }
      return true;
    }
  }

  /**
   * @brief Tests whether the matrix equals its own transpose.
   *
   * Unlike is_sparse_symmetric, a stored value whose mirror is *structurally*
   * zero is not skipped: it must itself be numerically zero for the full
   * matrix to equal its transpose.
   */
  SPARSEMAT_HD static bool is_full_symmetric(const SparseMat& a, DataType TOLERANCE = 1e-6) {
    if constexpr (rows != cols) {
      return false;
    } else {
      constexpr auto inds = SparseMat::indices();
      constexpr auto grid = offsets();
      // Guarded because the bound is a compile-time constant: when it is zero the
      // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
      if constexpr (SparseMat::nonZeroCount != 0) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(SparseMat::nonZeroCount); ++k) {
          const Int row = inds[k] / cols;
          const Int col = inds[k] % cols;
          const auto mirror = grid[static_cast<std::size_t>((col * cols) + row)];
          const DataType mirror_value =
              (mirror >= 0) ? a.values[static_cast<std::size_t>(mirror)] : DataType(0);
          if (!values_within(a.values[k], mirror_value, TOLERANCE)) {
            return false;
          }
        }
      }
      return true;
    }
  }
};
}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Returns true if the sparsity pattern of @p a is symmetric (i.e. non-zero positions are
 * mirrored across the diagonal).
 *
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Input matrix.
 * @return           True if the sparsity pattern is symmetric, false otherwise.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto is_structurally_symmetric([[maybe_unused]] const SparseMat& a) {
  return detail::Symmetric<SparseMat>::is_structurally_symmetric();
}

/**
 * @brief Returns true if the matrix @p a is symmetric within a given tolerance.
 *
 * Checks both the sparsity pattern and the values of stored elements, but only
 * compares positions that are stored in *both* (i, j) and (j, i). It therefore
 * returns @c false for any structurally asymmetric pattern, even one whose
 * unmatched values are all numerically zero — use @c is_full_symmetric() when
 * you need to treat such a matrix as symmetric.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Input matrix.
 * @param  TOLERANCE Tolerance for comparing non-zero values (default 1e-6).
 * @return           True if the matrix is symmetric, false otherwise.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto is_sparse_symmetric(const SparseMat& a,
                                      typename SparseMat::DataType TOLERANCE = 1e-6) {
  return detail::Symmetric<SparseMat>::is_sparse_symmetric(a, TOLERANCE);
}

/**
 * @brief Returns true if @p a equals its own transpose within @p TOLERANCE.
 *
 * Unlike @c is_sparse_symmetric(), a stored value whose mirror position is
 * structurally zero does not automatically fail — it only fails if that value
 * is itself outside @p TOLERANCE of zero. A non-square matrix is never
 * symmetric and returns @c false.
 *
 * @tparam SparseMat Matrix type.
 * @param  a         Input matrix.
 * @param  TOLERANCE Tolerance for comparing values (default 1e-6).
 * @return           True if the matrix equals its transpose, false otherwise.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto is_full_symmetric(const SparseMat& a,
                                    typename SparseMat::DataType TOLERANCE = 1e-6) {
  return detail::Symmetric<SparseMat>::is_full_symmetric(a, TOLERANCE);
}

}  // namespace SparseLinearAlgebra