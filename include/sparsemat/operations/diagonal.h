#pragma once

#include <cstddef>
#include <span>

#include "sparsemat/concepts/concepts.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Set every stored diagonal element of a sparse matrix to a single value.
 *
 * Iterates diagonal positions (N, N) at compile time.  Positions that are
 * structurally zero (not stored in the sparse format) are silently skipped.
 *
 * @tparam A     Sparse matrix type.
 * @tparam N     Current diagonal index (template recursion counter; leave at default).
 * @param  a     Matrix whose diagonal is to be set.
 * @param  value Scalar value written to every stored diagonal entry.
 */
template<SparseMatrixType A, typename A::Int N = 0>
SPARSEMAT_HD void set_diagonal_impl(A& a, typename A::DataType value) {
  constexpr auto min_dim = (A::rows < A::cols) ? A::rows : A::cols;
  if constexpr (N < min_dim) {
    constexpr auto idx = SparseLinearAlgebra::MatrixUtilities<A>::getSparseIndex(N, N);
    if constexpr (idx >= 0) {
      a.values[idx] = value;
    }
    set_diagonal_impl<A, N + 1>(a, value);
  }
}

/**
 * @brief Set the stored diagonal elements of a sparse matrix from a value array.
 *
 * Iterates diagonal positions (N, N) at compile time.  Only positions that are
 * structurally non-zero consume an entry from @p values; structurally-zero
 * diagonal positions are skipped without advancing @p index.
 *
 * @tparam A      Sparse matrix type.
 * @tparam N      Current diagonal index (template recursion counter; leave at default).
 * @tparam index  Current position within @p values (template recursion counter; leave at default).
 * @param  a      Matrix whose diagonal is to be set.
 * @param  values Span of values to write into stored diagonal entries, in order.
 */
template<SparseMatrixType A, typename A::Int N = 0, typename A::Int index = 0>
SPARSEMAT_HD void set_diagonal_impl(
    A& a,
    std::array<typename A::DataType, SparseLinearAlgebra::MatrixUtilities<A>::diagonal_nonzeros()>
        values) {
  constexpr auto min_dim = (A::rows < A::cols) ? A::rows : A::cols;
  if constexpr (N < min_dim &&
                index < SparseLinearAlgebra::MatrixUtilities<A>::diagonal_nonzeros()) {
    constexpr auto idx = SparseLinearAlgebra::MatrixUtilities<A>::getSparseIndex(N, N);
    if constexpr (idx >= 0) {
      a.values[idx] = values[index];
      set_diagonal_impl<A, N + 1, index + 1>(a, values);
    } else {
      set_diagonal_impl<A, N + 1, index>(a, values);
    }
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
    std::array<typename SparseMat::DataType,
               SparseLinearAlgebra::MatrixUtilities<SparseMat>::diagonal_nonzeros()> values) {
  detail::set_diagonal_impl(a, values);
}

}  // namespace SparseLinearAlgebra
