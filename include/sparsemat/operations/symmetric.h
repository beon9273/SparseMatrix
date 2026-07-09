#pragma once
#include <cmath>

#include "sparsemat/concepts/concepts.h"
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

  SPARSEMAT_HD constexpr static bool is_structurally_symmetric() {
    if constexpr (rows != cols) {
      return false;
    } else {
      for (Int i = 0; i < rows; ++i) {
        for (Int j = 0; j < cols; ++j) {
          if (SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(i, j) !=
              SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(j, i)) {
            return false;
          }
        }
      }
      return true;
    }
  }

  template<Int I = 0, Int J = 0>
  SPARSEMAT_HD constexpr static bool is_sparse_symmetric(const SparseMat& a,
                                                         DataType TOLERANCE = 1e-6) {
    if constexpr (!is_structurally_symmetric()) {
      return false;
    } else {
      if constexpr (I >= rows) {
        return true;
      } else if constexpr (J >= cols) {
        return is_sparse_symmetric<I + 1, 0>(a, TOLERANCE);
      } else {
        if (SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(I, J)) {
          constexpr auto index_ij =
              SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
          constexpr auto index_ji =
              SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(J, I);
          static_assert(index_ij >= 0 && index_ji >= 0,
                        "Non-zero positions must have valid indices.");
          if (std::abs(a.values[index_ij] - a.values[index_ji]) > TOLERANCE) {
            return false;
          }
        }
        return is_sparse_symmetric<I, J + 1>(a, TOLERANCE);
      }
      return true;
    }
  }

  template<Int I = 0>
  SPARSEMAT_HD constexpr static bool is_full_symmetric(const SparseMat& a,
                                                       DataType TOLERANCE = 1e-6) {
    if constexpr (I >= SparseMat::nonZeroCount) {
      return true;
    } else {
      constexpr auto index = SparseMat::indices()[I];
      constexpr auto row = index / SparseMat::cols;
      constexpr auto col = index % SparseMat::cols;

      const auto& value = a.values[I];
      if constexpr (SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(col, row)) {
        constexpr auto index_ji =
            SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(col, row);
        if (std::abs(value - a.values[index_ji]) > TOLERANCE) {
          return false;  // Non-zero values are not symmetric within tolerance
        }
      } else {
        if (std::abs(value) > TOLERANCE) {
          return false;  // Non-zero position does not have a corresponding symmetric position
        }
      }
      return is_full_symmetric<I + 1>(a, TOLERANCE);
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
 * Checks both the sparsity pattern and the values of non-zero elements. Does not check
   TRUE symmetry for non-zero elements that might be numerically zero. For true symmetry,.
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

template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto is_full_symmetric(const SparseMat& a,
                                    typename SparseMat::DataType TOLERANCE = 1e-6) {
  return detail::Symmetric<SparseMat>::is_full_symmetric(a, TOLERANCE);
}  // namespace SparseLinearAlgebra

}  // namespace SparseLinearAlgebra