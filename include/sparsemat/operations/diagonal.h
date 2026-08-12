#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/// Length of the diagonal of @c A — @c min(rows, cols).
template<SparseMatrixType A>
inline constexpr auto diagonal_length = (A::rows < A::cols) ? A::rows : A::cols;

/**
 * @brief Storage offsets of every *stored* diagonal entry of @c A, in row order.
 *
 * The compile-time part of set_diagonal is deciding which diagonal positions
 * exist at all; once that list is known, writing to them is pure data
 * movement and a runtime loop does it without a template instantiation per
 * position (and without running into clang's 256-argument fold nesting limit
 * on a large matrix).
 */
template<SparseMatrixType A>
SPARSEMAT_HD constexpr auto stored_diagonal_offsets() {
  using Int = typename A::Int;
  std::array<Int,
             static_cast<std::size_t>(SparseLinearAlgebra::MatrixUtilities<A>::diagonal_nonzeros())>
      offsets{};
  constexpr auto grid = SparseLinearAlgebra::MatrixUtilities<A>::storage_index_grid();
  std::size_t k = 0;
  for (Int i = 0; i < diagonal_length<A>; ++i) {
    const auto offset = grid[static_cast<std::size_t>((i * A::cols) + i)];
    if (offset >= 0) {
      offsets[k++] = offset;
    }
  }
  return offsets;
}

/**
 * @brief Set every stored diagonal element of a sparse matrix to a single value.
 *
 * Structurally zero diagonal positions are silently skipped — the sparsity
 * pattern is immutable, so there is nowhere to put the value.
 *
 * @param a     Matrix whose diagonal is to be set.
 * @param value Scalar value written to every stored diagonal entry.
 */
template<SparseMatrixType A>
SPARSEMAT_HD void set_diagonal_impl(A& a, typename A::DataType value) {
  constexpr auto offsets = stored_diagonal_offsets<A>();
  for (auto offset : offsets) {
    a.values[static_cast<std::size_t>(offset)] = value;
  }
}

/**
 * @brief Set the stored diagonal elements of a sparse matrix from a value array.
 *
 * Only positions that are structurally non-zero consume an entry from
 * @p values; structurally-zero diagonal positions are skipped without
 * consuming one. @c stored_diagonal_offsets() is already in row order, so the
 * k-th entry of @p values lands in the k-th stored diagonal position.
 *
 * @param a      Matrix whose diagonal is to be set.
 * @param values Values to write into stored diagonal entries, in row order.
 */
template<SparseMatrixType A>
SPARSEMAT_HD void set_diagonal_impl(
    A& a,
    const std::array<typename A::DataType,
                     SparseLinearAlgebra::MatrixUtilities<A>::diagonal_nonzeros()>& values) {
  constexpr auto offsets = stored_diagonal_offsets<A>();
  for (std::size_t k = 0; k < offsets.size(); ++k) {
    a.values[static_cast<std::size_t>(offsets[k])] = values[k];
  }
}

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Set every stored diagonal element of a sparse matrix to a single value.
 * @param a     Matrix to modify.
 * @param value Scalar written to every stored diagonal entry.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD void set_diagonal(SparseMat& a, typename SparseMat::DataType value) {
  detail::set_diagonal_impl(a, value);
}

/**
 * @brief Set the stored diagonal elements of a sparse matrix from a value array.
 *
 * Only structurally non-zero diagonal positions are written; each consumes one
 * entry from @p values in order.
 *
 * @param a      Matrix to modify.
 * @param values Values to write into stored diagonal entries, in row order.
 */
template<SparseMatrixType SparseMat>
SPARSEMAT_HD void set_diagonal(
    SparseMat& a,
    const std::array<typename SparseMat::DataType,
                     SparseLinearAlgebra::MatrixUtilities<SparseMat>::diagonal_nonzeros()>&
        values) {
  detail::set_diagonal_impl(a, values);
}

}  // namespace SparseLinearAlgebra
