#pragma once

#include <cmath>
#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for comparing two sparse matrices.
 *
 * Comparison is *mathematical*, not structural: two matrices are equal when
 * every element of one equals the corresponding element of the other,
 * regardless of whether the two sparsity patterns agree. A diagonal matrix
 * and a full matrix holding the same numbers therefore compare equal, which
 * is what makes @c == usable on operation results (whose patterns are derived,
 * and often wider than the values actually warrant — @c add() unions both
 * operands' patterns, so @c a.add(b) can carry an explicitly-stored 0.0 where
 * a hand-written literal would carry a structural zero).
 *
 * Only positions stored in one matrix or the other are visited: positions
 * structurally zero in both are 0 == 0 and need no check. That keeps the work
 * at O(nnzA + nnzB) rather than O(rows*cols).
 *
 * @tparam A Left-hand matrix type.
 * @tparam B Right-hand matrix type; must have the same shape as @c A.
 */
template<SparseMatrixType A, SparseMatrixType B>
class Compare {
 public:
  static_assert(A::rows == B::rows && A::cols == B::cols,
                "Incompatible matrix dimensions for comparison.");
  static_assert(SparseLinearAlgebra::SameDataType<A, B>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types \u2014 convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  using DataType = typename A::DataType;
  using Int = typename A::Int;

  // Held as functions rather than `static constexpr` data members, and copied
  // into locals at each use. These lookups happen at *runtime*, so static
  // members would be ODR-used and would need device storage they do not get,
  // giving "identifier ... is undefined in device code" under nvcc. Same
  // reasoning as SparseMat::indices().
  //
  // Comparison is pure value reading — there is no structurally-zero term to
  // eliminate at compile time — so the loops below are ordinary runtime loops
  // with O(1) lookups rather than folds. A fold would cost one instantiation
  // per stored value and cap out at clang's 256-term nesting limit, which a
  // densified 32x32 operand (1024 values) exceeds immediately.
  SPARSEMAT_HD constexpr static auto a_offsets() {
    return SparseLinearAlgebra::MatrixUtilities<A>::storage_index_grid();
  }
  SPARSEMAT_HD constexpr static auto b_offsets() {
    return SparseLinearAlgebra::MatrixUtilities<B>::storage_index_grid();
  }

  SPARSEMAT_HD static bool equal(const A& a, const B& b, DataType tolerance) {
    constexpr auto a_inds = A::indices();
    constexpr auto b_inds = B::indices();
    constexpr auto a_grid = a_offsets();
    constexpr auto b_grid = b_offsets();

    // Every position stored by A must match B there (B contributing 0 when it
    // does not store that position).
    // Guarded because the bound is a compile-time constant: when it is zero the
    // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
    if constexpr (A::nonZeroCount != 0) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(A::nonZeroCount); ++i) {
        const auto b_offset = b_grid[static_cast<std::size_t>(a_inds[i])];
        const DataType b_value =
            (b_offset >= 0) ? b.values[static_cast<std::size_t>(b_offset)] : DataType(0);
        const DataType diff = a.values[i] - b_value;
        if ((diff < DataType(0) ? -diff : diff) > tolerance) {
          return false;
        }
      }
    }

    // And the mirror direction, to catch positions B stores that A does not.
    // Positions stored by both get checked twice, which is harmless and
    // cheaper than computing the union of the two patterns to deduplicate.
    // Guarded because the bound is a compile-time constant: when it is zero the
    // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
    if constexpr (B::nonZeroCount != 0) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(B::nonZeroCount); ++i) {
        const auto a_offset = a_grid[static_cast<std::size_t>(b_inds[i])];
        const DataType a_value =
            (a_offset >= 0) ? a.values[static_cast<std::size_t>(a_offset)] : DataType(0);
        const DataType diff = a_value - b.values[i];
        if ((diff < DataType(0) ? -diff : diff) > tolerance) {
          return false;
        }
      }
    }
    return true;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Exact element-wise equality of two sparse matrices.
 *
 * Compares mathematical values, not sparsity patterns: a matrix that stores a
 * numeric zero equals one where that position is structurally zero. Both
 * operands must have the same dimensions and the same @c DataType.
 *
 * This is an exact floating-point comparison; use @c approx_equal() with an
 * explicit tolerance for anything that has been through a factorization or a
 * long chain of arithmetic.
 *
 * @tparam A Left-hand matrix type.
 * @tparam B Right-hand matrix type.
 * @return   @c true if every element of @p a equals the corresponding element of @p b.
 */
template<SparseMatrixType A, SparseMatrixType B>
[[nodiscard]] SPARSEMAT_HD bool operator==(const A& a, const B& b) {
  return detail::Compare<A, B>::equal(a, b, typename A::DataType(0));
}

/// Negation of @c operator==. (C++20 synthesizes this, but spelling it out
/// keeps the intent obvious and costs nothing.)
template<SparseMatrixType A, SparseMatrixType B>
[[nodiscard]] SPARSEMAT_HD bool operator!=(const A& a, const B& b) {
  return !(a == b);
}

/**
 * @brief Element-wise equality within @p tolerance.
 *
 * Same value-not-pattern semantics as @c operator==, but every element
 * comparison allows an absolute difference of up to @p tolerance. This is the
 * form to use on anything that has been through a solve or factorization.
 *
 * @param a         Left-hand operand.
 * @param b         Right-hand operand.
 * @param tolerance Maximum absolute difference allowed per element.
 */
template<SparseMatrixType A, SparseMatrixType B>
[[nodiscard]] SPARSEMAT_HD bool approx_equal(const A& a,
                                             const B& b,
                                             typename A::DataType tolerance = 1e-6) {
  return detail::Compare<A, B>::equal(a, b, tolerance);
}

}  // namespace SparseLinearAlgebra
