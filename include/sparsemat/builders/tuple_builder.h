#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"

// Forward declaration only: this header names the concrete SparseMat class
// template directly (unlike operations/*.h, which stay generic over any
// SparseMatrixType-constrained template parameter and never need to spell
// SparseMat itself). The forward declaration is enough to parse this file
// standalone; make_sparse_matrix()'s body only needs the complete type at
// the point it's actually instantiated, by which time a caller will have
// included the full sparsemat/api/sparsemat.h.
namespace SparseLinearAlgebra {
template<typename DType, typename IntType, int Rows, int Cols, IntType... NonZeros>
class SparseMat;
}

namespace SparseLinearAlgebra {

/**
 * @brief One (row, col, value) entry used to build a matrix via
 *        @c make_sparse_matrix(), instead of hand-computing flat row-major
 *        indices and keeping a separate values list in sync with them.
 *
 * A plain aggregate (not a class template), so a list of these can be
 * passed directly as a non-type template argument without repeating a type
 * argument at every entry. @c row/@c col use a wide integer type and
 * @c value uses @c double regardless of the target matrix's actual
 * @c IntType/@c DataType — @c make_sparse_matrix() converts down to those at
 * the point of construction, the same way @c SparseMat's own variadic
 * constructor already does for its arguments.
 */
struct SparseEntry {
  long long row;
  long long col;
  double value;
};

namespace detail {

/**
 * @brief Sorts a copy of @p arr into ascending flat row-major index order.
 *
 * A hand-rolled insertion sort rather than @c std::sort: this needs to run
 * as part of a @c constexpr non-type-template-argument initializer, and
 * this library hand-rolls its own constexpr loops throughout (rather than
 * leaning on @c <algorithm>) partly to avoid ever needing to find out how
 * well a given compiler's constexpr evaluator supports library-provided
 * constexpr algorithms. Entry counts here are sparsity-pattern sizes, not
 * matrix dimensions, so quadratic behavior is not a practical concern.
 */
template<long long Cols, std::size_t N>
SPARSEMAT_HD constexpr auto sort_entries_by_flat_index(std::array<SparseEntry, N> arr) {
  for (std::size_t i = 1; i < N; ++i) {
    SparseEntry key = arr[i];
    auto key_flat = (key.row * Cols) + key.col;
    std::size_t j = i;
    while (j > 0 && (((arr[j - 1].row * Cols) + arr[j - 1].col) > key_flat)) {
      arr[j] = arr[j - 1];
      --j;
    }
    arr[j] = key;
  }
  return arr;
}

template<typename DType, typename IntType, int Rows, int Cols, auto Sorted, std::size_t... Is>
SPARSEMAT_HD constexpr auto build_sparse_matrix(std::index_sequence<Is...> /*seq*/) {
  return SparseMat<DType,
                   IntType,
                   Rows,
                   Cols,
                   static_cast<IntType>((Sorted[Is].row * Cols) + Sorted[Is].col)...>(
      std::array<DType, sizeof...(Is)>{static_cast<DType>(Sorted[Is].value)...});
}

}  // namespace detail

/**
 * @brief Builds a @c SparseMat from a list of (row, col, value) entries,
 *        instead of hand-computing flat row-major indices and keeping a
 *        separate values list in the same order as them.
 *
 * @c Entries must be a compile-time @c std::array<SparseEntry, N> — pass it
 * as a named @c constexpr variable, since that's the readable way to supply
 * a non-type template argument in C++20. Entries are sorted by flat
 * row-major index internally, so listing order doesn't matter and the
 * resulting type is canonical regardless of what order they're given in.
 * Duplicate or out-of-bounds (row, col) pairs are caught by @c SparseMat's
 * own @c static_assert — the same one that fires for the hand-written
 * constructor form — since this ultimately constructs a perfectly ordinary
 * @c SparseMat.
 *
 * @code
 * constexpr auto entries = std::array{
 *     SparseLinearAlgebra::SparseEntry{0, 0, 4.0},
 *     SparseLinearAlgebra::SparseEntry{1, 1, 2.0},
 *     SparseLinearAlgebra::SparseEntry{0, 1, 5.0},
 * };
 * auto A = SparseLinearAlgebra::make_sparse_matrix<double, int, 3, 3, entries>();
 * @endcode
 *
 * @tparam DType   Scalar element type of the resulting matrix.
 * @tparam IntType Signed integer type used for indices (see @c SparseMat).
 * @tparam Rows    Number of rows.
 * @tparam Cols    Number of columns.
 * @tparam Entries Compile-time list of (row, col, value) entries.
 * @return         A @c SparseMat<DType, IntType, Rows, Cols, ...> storing
 *                 exactly the given entries, in row-major order.
 */
template<typename DType, typename IntType, int Rows, int Cols, auto Entries>
SPARSEMAT_HD constexpr auto make_sparse_matrix() {
  constexpr auto sorted = detail::sort_entries_by_flat_index<Cols>(Entries);
  return detail::build_sparse_matrix<DType, IntType, Rows, Cols, sorted>(
      std::make_index_sequence<Entries.size()>{});
}

}  // namespace SparseLinearAlgebra
