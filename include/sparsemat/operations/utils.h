#pragma once

#include "sparsemat/concepts/concepts.h"

namespace SparseLinearAlgebra {

/**
 * @brief Compile-time utilities for inspecting the sparsity pattern of a
 *        sparse matrix type.
 *
 * All methods are @c constexpr so they can be evaluated at compile time when
 * the indices are known.
 *
 * @tparam SparseMatrix A @c SparseMat instantiation whose @c indices()
 *                      accessor and dimension constants are accessible.
 */
template<typename SparseMatrix>
class MatrixUtilities {
 public:
  using Int = typename SparseMatrix::Int;

  /**
   * @brief Forwards to @c SparseMatrix::make to rebuild a matrix type from a
   *        compile-time sparsity array.
   *
   * @tparam NRows Target row count.
   * @tparam NCols Target column count.
   * @tparam Arr   Compile-time array whose elements become the @c NonZeros pack.
   * @tparam Is    Index sequence over @p Arr.
   * @return       Default-constructed rebuilt sparse matrix.
   */
  template<Int NRows, Int NCols, auto Arr, std::size_t... Is>
  SPARSEMAT_HD static auto make(std::index_sequence<Is...> /*seq*/) {
    return typename SparseMatrix::template Rebind<NRows, NCols, Arr[static_cast<Int>(Is)]...>{};
  }

  /**
   * @brief Checks whether position (row, col) is a non-zero element.
   *
   * Scans @c SparseMatrix::indices() for the flat index @c row*cols + col.
   * @c indices() returns its own array by value rather than exposing a
   * `static constexpr` data member, so this loop always indexes a local
   * copy — safe to call from @c SPARSEMAT_HD code on both host and device,
   * since there's no host-only static storage being referenced.
   *
   * @param row Row index.
   * @param col Column index.
   * @return    @c true if the position has a stored (non-zero) entry.
   */
  SPARSEMAT_HD constexpr static bool isNonZero(Int row, Int col) {
    auto index = (row * SparseMatrix::cols) + col;
    for (Int i = 0; i < SparseMatrix::nonZeroCount; ++i) {
      if (SparseMatrix::indices()[i] == index) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Returns the offset into @c values[] for position (row, col).
   *
   * Scans @c SparseMatrix::indices() for the flat index @c row*cols + col
   * and returns its position in the packed storage array.
   *
   * @param row Row index.
   * @param col Column index.
   * @return    Zero-based index into @c values, or @c -1 if the position is
   *            structurally zero.
   */
  SPARSEMAT_HD constexpr static auto getSparseIndex(Int row, Int col) {
    // Causes compile error at compile time, assert as normal at runtime.
    assert(row >= 0 && row < SparseMatrix::rows && col >= 0 && col < SparseMatrix::cols);

    if constexpr (SparseMatrix::nonZeroCount != 0) {
      auto index = (row * SparseMatrix::cols) + col;
      for (Int i = 0; i < SparseMatrix::nonZeroCount; ++i) {
        if (SparseMatrix::indices()[i] == index) {
          return i;
        }
      }
    }
    return Int(-1);
  }

  SPARSEMAT_HD constexpr static auto diagonal_nonzeros() {
    Int count = 0;
    for (Int i = 0; i < SparseMatrix::rows && i < SparseMatrix::cols; ++i) {
      if (isNonZero(i, i)) {
        ++count;
      }
    }
    return count;
  }
  SPARSEMAT_HD constexpr static auto num_nonzeros() { return SparseMatrix::nonZeroCount; }
};

/**
 * @brief Compile-time utilities for computing the result sparsity of a matrix
 *        operation.
 *
 * The @p Operation type must expose:
 *   - @c Operation::rows and @c Operation::cols (result dimensions), and
 *   - @c Operation::is_result_index_nonzero(row, col) — a @c constexpr
 *     predicate that returns @c true when the result at (row, col) may be
 *     non-zero.
 *
 * @tparam Operation Internal operation policy class (e.g. @c Multiply, @c Add).
 */
template<OperationType Operation>
class OperationUtilities {
 public:
  using Int = typename Operation::Int;

  /**
   * @brief Counts the number of non-zero positions in the operation result.
   *
   * Iterates over all (i, j) pairs and calls
   * @c Operation::is_result_index_nonzero to determine the result sparsity.
   *
   * @return Number of non-zero elements in the result matrix.
   */
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    Int count = 0;
    for (Int i = 0; i < Operation::rows; i++) {
      for (Int j = 0; j < Operation::cols; j++) {
        if (Operation::is_result_index_nonzero(i, j)) {
          ++count;
        }
      }
    }
    return count;
  }

  /**
   * @brief Builds the flat row-major index array describing the result sparsity.
   *
   * Iterates in row-major order and records every (i, j) for which
   * @c Operation::is_result_index_nonzero returns @c true.
   *
   * @return @c std::array<typename Operation::Int, num_nonzeros()> of flat indices.
   */
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    std::array<Int, num_nonzeros()> sparsity{};
    Int c = 0;
    for (Int i = 0; i < Operation::rows; i++) {
      for (Int j = 0; j < Operation::cols; j++) {
        if (Operation::is_result_index_nonzero(i, j)) {
          sparsity[c++] = (i * Operation::cols) + j;
        }
      }
    }
    return sparsity;
  }
};

}  // namespace SparseLinearAlgebra
