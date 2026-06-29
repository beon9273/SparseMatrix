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
 * @tparam SparseMatrix A @c SparseMat instantiation whose static @c indices
 *                      array and dimension constants are accessible.
 */
template<SparseMatrixType SparseMatrix>
class MatrixUtilities {
 public:
  /**
   * @brief Checks whether position (row, col) is a non-zero element.
   *
   * Linearly scans the @c SparseMatrix::indices array for the flat index
   * @c row*cols + col.
   *
   * @param row Row index.
   * @param col Column index.
   * @return    @c true if the position has a stored (non-zero) entry.
   */
  constexpr static bool isNonZero(std::size_t row, std::size_t col) {
    std::size_t index = (row * SparseMatrix::cols) + col;
    for (std::size_t i = 0; i < SparseMatrix::nonZeroCount; ++i) {
      if (SparseMatrix::indices[i] == index) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Returns the offset into @c values[] for position (row, col).
   *
   * Scans @c SparseMatrix::indices for the flat index @c row*cols + col
   * and returns its position in the packed storage array.
   *
   * @param row Row index.
   * @param col Column index.
   * @return    Zero-based index into @c values, or @c -1 if the position is
   *            structurally zero.
   */
  constexpr static int getSparseIndex(std::size_t row, std::size_t col) {
    std::size_t index = (row * SparseMatrix::cols) + col;
    for (std::size_t i = 0; i < SparseMatrix::nonZeroCount; ++i) {
      if (SparseMatrix::indices[i] == index) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
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
  /**
   * @brief Counts the number of non-zero positions in the operation result.
   *
   * Iterates over all (i, j) pairs and calls
   * @c Operation::is_result_index_nonzero to determine the result sparsity.
   *
   * @return Number of non-zero elements in the result matrix.
   */
  constexpr static auto num_nonzeros() {
    int count = 0;
    for (std::size_t i = 0; i < Operation::rows; i++) {
      for (std::size_t j = 0; j < Operation::cols; j++) {
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
   * @return @c std::array<std::size_t, num_nonzeros()> of flat indices.
   */
  constexpr static auto calculate_sparsity() {
    std::array<std::size_t, num_nonzeros()> sparsity{};
    std::size_t c = 0;
    for (std::size_t i = 0; i < Operation::rows; i++) {
      for (std::size_t j = 0; j < Operation::cols; j++) {
        if (Operation::is_result_index_nonzero(i, j)) {
          sparsity[c++] = (i * Operation::cols) + j;
        }
      }
    }
    return sparsity;
  }
};

}  // namespace SparseLinearAlgebra
