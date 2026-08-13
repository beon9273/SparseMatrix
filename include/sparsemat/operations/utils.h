#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"

namespace SparseLinearAlgebra {

/**
 * @brief Largest number of terms to put in a single fold expression.
 *
 * clang caps expression nesting at 256 (`-fbracket-depth`), and a fold
 * expression counts against it: `(f<Is>(), ...)` over more than 256 indices is
 * rejected outright with "instantiating fold expression with N arguments
 * exceeded expression nesting limit of 256". Since an operation's fold runs
 * over its result's non-zero count, that ceiling is reached by ordinary
 * matrices — a 60x60 tridiagonal squared has 294 non-zeros — so unrolling has
 * to be split into chunks rather than emitted as one flat fold.
 *
 * 64 leaves generous headroom under the limit while keeping the number of
 * chunk-splitting instantiations small.
 */
inline constexpr std::size_t kUnrollChunkSize = 64;

/**
 * @brief Chunked unrolling.
 *
 * Each operation drives its per-element step function through a
 * @c fill_range<Begin, Count> helper that halves @p Count until it fits in one
 * fold of at most @c kUnrollChunkSize terms. Recursion depth is
 * log2(Count / kUnrollChunkSize) — 3 for a thousand elements, versus the
 * Count-deep linear recursion that blows the compiler's 900-deep instantiation
 * budget, and without the flat fold's 256-term nesting ceiling.
 *
 * Splitting preserves order: the low half is fully expanded before the high
 * half, and within a chunk the comma operator in a fold expression is the
 * built-in sequencing comma. That matters for the operations whose steps are
 * order-dependent (the triangular solves, and the column loops in the LU and
 * Cholesky factorizations).
 *
 * The helpers are written out per operation, over each one's existing
 * @c fill_cell static member, rather than factored into one generic
 * higher-order function here: a generic version would have to take a callable,
 * and the natural way to pass one is a lambda — which nvcc only accepts in
 * @c __device__ code under @c --extended-lambda, a flag this library should
 * not force on its consumers.
 */

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

  /**
   * @brief Builds a dense row-major boolean grid marking every structurally
   *        non-zero position, from the compact @c indices() array.
   *
   * @c isNonZero()/@c getSparseIndex() are each an @c O(nonZeroCount) linear
   * scan. That's fine for a one-off lookup, but operations whose
   * @c is_result_index_nonzero() needs many lookups per call (e.g. Multiply,
   * checking every shared index @c k) or gets called @c O(rows*cols) times
   * by @c OperationUtilities — which is every operation's sparsity
   * computation — pay that scan cost repeatedly, un-memoized, for the same
   * matrix. Precomputing this grid once (here, @c O(rows*cols) to
   * zero-initialize plus @c O(nonZeroCount) to populate) turns each
   * subsequent lookup into an @c O(1) array access instead. This matters a
   * lot for constexpr-evaluation cost at larger matrix sizes: an unmemoized
   * scan-per-lookup pattern compounds into the compiler's constexpr
   * evaluation step budget (distinct from, and unrelated to, the
   * template-instantiation-depth budget) well before a matrix gets large
   * enough to need it for any other reason.
   *
   * @return @c std::array<std::array<bool, cols>, rows> with @c true at
   *         every structurally non-zero (row, col).
   */
  SPARSEMAT_HD constexpr static auto to_dense_bool() {
    std::array<std::array<bool, static_cast<std::size_t>(SparseMatrix::cols)>,
               static_cast<std::size_t>(SparseMatrix::rows)>
        grid{};
    for (auto idx : SparseMatrix::indices()) {
      grid[idx / SparseMatrix::cols][idx % SparseMatrix::cols] = true;
    }
    return grid;
  }

  /**
   * @brief Builds a flat row-major table mapping each position to its offset
   *        in @c values[], or @c -1 if the position is structurally zero.
   *
   * The @c to_dense_bool() grid answers "is this position stored?"; this
   * answers "*where* is it stored?", which is what any operation that moves
   * values around (rather than deciding sparsity) actually needs.
   *
   * Its purpose is to let those operations use an ordinary runtime loop
   * instead of a compile-time fold. Folds are the right tool where there is
   * zero-skipping to exploit — multiply, add, the solves — because a
   * structurally-zero term can be eliminated entirely at compile time. But
   * @c dense(), @c trace(), @c set_diagonal() and the comparison operators
   * just copy or read values; there is nothing to eliminate, so a fold buys
   * nothing and costs a template instantiation per element. It also imposes a
   * hard ceiling: clang rejects a fold expression with more than 256
   * arguments ("exceeded expression nesting limit"), which a 32x32 dense
   * matrix (1024 stored values) trips immediately. A runtime loop over this
   * table is O(1) per lookup, has no size ceiling on any compiler, and
   * optimizes to the same code for the small matrices this library targets.
   *
   * @return @c std::array<Int, rows*cols> of storage offsets, @c -1 where
   *         the position is structurally zero.
   */
  SPARSEMAT_HD constexpr static auto storage_index_grid() {
    std::array<Int,
               static_cast<std::size_t>(SparseMatrix::rows) *
                   static_cast<std::size_t>(SparseMatrix::cols)>
        grid{};
    for (auto& slot : grid) {
      slot = Int(-1);
    }
    constexpr auto inds = SparseMatrix::indices();
    for (Int i = 0; i < SparseMatrix::nonZeroCount; ++i) {
      grid[static_cast<std::size_t>(inds[i])] = i;
    }
    return grid;
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
    // Explicit cast: num_nonzeros() returns the matrix's signed Int type,
    // but std::array's size is std::size_t. nvcc's frontend rejects the
    // implicit signed-to-unsigned conversion as narrowing once the
    // surrounding non-type template argument list gets large enough (seen
    // in practice once a matrix has on the order of a hundred-plus stored
    // elements), even though g++/clang accept it unconditionally.
    std::array<Int, static_cast<std::size_t>(num_nonzeros())> sparsity{};
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
