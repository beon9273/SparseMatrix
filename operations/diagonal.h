#pragma once

#include <cstddef>
#include <span>

#include "concepts/concepts.h"

namespace {

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
template<SparseMatrixType A, std::size_t N = 0>
void set_diagonal_impl(A& a, typename A::DataType value) {
  if constexpr (N < A::rows) {
    constexpr int idx = SparseLinearAlgebra::MatrixUtilities<A>::getSparseIndex(N, N);
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
template<SparseMatrixType A, std::size_t N = 0, std::size_t index = 0>
void set_diagonal_impl(A& a, std::span<typename A::DataType> values) {
  if constexpr (N < A::rows) {
    constexpr int idx = SparseLinearAlgebra::MatrixUtilities<A>::getSparseIndex(N, N);
    if constexpr (idx >= 0) {
      a.values[idx] = values[index];
      set_diagonal_impl<A, N + 1, index + 1>(a, values);
    } else {
      set_diagonal_impl<A, N + 1, index>(a, values);
    }
  }
}

}  // namespace

namespace SparseLinearAlgebra {

/**
 * @brief Set every stored diagonal element of a sparse matrix to a single value.
 * @param a     Matrix to modify.
 * @param value Scalar written to every stored diagonal entry.
 */
template<SparseMatrixType SparseMat>
void set_diagonal(SparseMat& a, typename SparseMat::DataType value) {
  set_diagonal_impl(a, value);
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
void set_diagonal(SparseMat& a, std::span<typename SparseMat::DataType> values) {
  set_diagonal_impl(a, values);
}

}  // namespace SparseLinearAlgebra
