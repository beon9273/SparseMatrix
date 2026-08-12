#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"

// Forward declaration only, for the same reason as builders/tuple_builder.h:
// this header names the concrete SparseMat class template directly, and the
// declaration is enough to parse standalone. The complete type is only needed
// where these functions are actually instantiated, by which point a caller
// will have included sparsemat/api/sparsemat.h.
namespace SparseLinearAlgebra {
template<typename DType, typename IntType, int Rows, int Cols, IntType... NonZeros>
class SparseMat;
}

namespace SparseLinearAlgebra {

/**
 * @brief One (row, col) position, for declaring a sparsity pattern without
 *        also supplying a value.
 *
 * @c SparseEntry (builders/tuple_builder.h) pairs a position with a value,
 * which is what you want when the numbers are known up front. This is the
 * other common case — and the one the Kalman example needs — where the shape
 * is fixed at compile time but the values are filled in later, or repeatedly,
 * at runtime.
 */
struct SparsePosition {
  long long row;
  long long col;
};

namespace detail {

/**
 * @brief A sparsity pattern as flat row-major indices, plus its length.
 *
 * The array is sized to the worst case (a fully dense Rows x Cols) so that a
 * generator can fill it without knowing the final count up front — the count
 * travels alongside instead. Callers expand only the valid prefix, so the
 * padding never reaches the resulting type.
 *
 * All members are public and of structural type, so this can be used directly
 * as a non-type template argument — which it has to be, since the pattern
 * determines the resulting @c SparseMat type.
 */
template<int Rows, int Cols>
struct FlatPattern {
  std::array<long long, static_cast<std::size_t>(Rows) * static_cast<std::size_t>(Cols)> flat{};
  std::size_t count{};

  /// Appends a position, ignoring exact duplicates so that generators which
  /// may visit a position twice (@c symmetric_from_lower on a diagonal entry,
  /// say) do not produce an invalid pattern.
  SPARSEMAT_HD constexpr void add(long long row, long long col) {
    const long long index = (row * Cols) + col;
    for (std::size_t i = 0; i < count; ++i) {
      if (flat[i] == index) {
        return;
      }
    }
    flat[count++] = index;
  }
};

/**
 * @brief Sorts a pattern into ascending flat-index order.
 *
 * Hand-rolled insertion sort, for the same reason as
 * @c sort_entries_by_flat_index in builders/tuple_builder.h: this has to run
 * inside a @c constexpr non-type-template-argument initializer, and this
 * library avoids depending on how well a given compiler's constexpr evaluator
 * handles library algorithms. Sorting makes the resulting type canonical
 * regardless of the order a generator happened to emit positions in.
 */
template<int Rows, int Cols>
SPARSEMAT_HD constexpr auto sorted(FlatPattern<Rows, Cols> pattern) {
  for (std::size_t i = 1; i < pattern.count; ++i) {
    const long long key = pattern.flat[i];
    std::size_t j = i;
    while (j > 0 && pattern.flat[j - 1] > key) {
      pattern.flat[j] = pattern.flat[j - 1];
      --j;
    }
    pattern.flat[j] = key;
  }
  return pattern;
}

template<typename DType, typename IntType, int Rows, int Cols, auto Pattern, std::size_t... Is>
SPARSEMAT_HD constexpr auto build_from_pattern(std::index_sequence<Is...> /*seq*/) {
  return SparseMat<DType, IntType, Rows, Cols, static_cast<IntType>(Pattern.flat[Is])...>{};
}

// --- Pattern generators ---

template<int Rows, int Cols, auto Positions>
SPARSEMAT_HD constexpr auto positions_pattern() {
  FlatPattern<Rows, Cols> pattern{};
  for (const auto& position : Positions) {
    pattern.add(position.row, position.col);
  }
  return sorted(pattern);
}

template<int N, auto Positions>
SPARSEMAT_HD constexpr auto mirrored_pattern() {
  FlatPattern<N, N> pattern{};
  for (const auto& position : Positions) {
    pattern.add(position.row, position.col);
    pattern.add(position.col, position.row);
  }
  return sorted(pattern);
}

template<int Rows, int Cols, int Lower, int Upper>
SPARSEMAT_HD constexpr auto banded_pattern() {
  FlatPattern<Rows, Cols> pattern{};
  for (int row = 0; row < Rows; ++row) {
    for (int col = 0; col < Cols; ++col) {
      const int offset = col - row;
      if (offset >= -Lower && offset <= Upper) {
        pattern.add(row, col);
      }
    }
  }
  return sorted(pattern);  // already ascending, but keep the invariant explicit
}

}  // namespace detail

/**
 * @brief Builds a zero-valued matrix with exactly the given sparsity pattern.
 *
 * The counterpart to @c make_sparse_matrix(): that one takes
 * @c (row, col, value) triples, this one takes @c (row, col) positions and
 * leaves every stored value at zero, for when the shape is fixed at compile
 * time but the numbers arrive later.
 *
 * @c Positions must be a compile-time @c std::array<SparsePosition, N> passed
 * as a named @c constexpr variable, since it determines the resulting type.
 * Order does not matter — positions are sorted internally, so the type is
 * canonical — and exact duplicates are collapsed rather than producing an
 * invalid pattern. Out-of-bounds positions are still caught by @c SparseMat's
 * own @c static_assert.
 *
 * @code
 * constexpr auto shape = std::array{
 *     SparseLinearAlgebra::SparsePosition{0, 0},
 *     SparseLinearAlgebra::SparsePosition{1, 1},
 *     SparseLinearAlgebra::SparsePosition{0, 1},
 * };
 * auto A = SparseLinearAlgebra::make_pattern<double, int, 2, 2, shape>();
 * A.set(0, 0, 4.0);  // fill in later
 * @endcode
 *
 * @tparam DType     Scalar element type.
 * @tparam IntType   Signed integer type used for indices.
 * @tparam Rows      Number of rows.
 * @tparam Cols      Number of columns.
 * @tparam Positions Compile-time list of (row, col) positions.
 */
template<typename DType, typename IntType, int Rows, int Cols, auto Positions>
SPARSEMAT_HD constexpr auto make_pattern() {
  constexpr auto pattern = detail::positions_pattern<Rows, Cols, Positions>();
  return detail::build_from_pattern<DType, IntType, Rows, Cols, pattern>(
      std::make_index_sequence<pattern.count>{});
}

/**
 * @brief Builds a zero-valued square matrix whose pattern is @p Positions
 *        mirrored across the diagonal.
 *
 * Give only the lower triangle (or only the upper — it is symmetric either
 * way) and the mirror positions are added for you. A symmetric pattern is
 * exactly what the factorizations want, and writing both halves by hand is
 * both tedious and easy to get subtly wrong.
 *
 * Diagonal positions mirror onto themselves and are not duplicated.
 *
 * @code
 * constexpr auto lower = std::array{
 *     SparseLinearAlgebra::SparsePosition{0, 0},
 *     SparseLinearAlgebra::SparsePosition{1, 0},
 *     SparseLinearAlgebra::SparsePosition{1, 1},
 * };
 * // Pattern is {(0,0), (0,1), (1,0), (1,1)}.
 * auto A = SparseLinearAlgebra::symmetric_from_lower<double, int, 2, lower>();
 * @endcode
 *
 * @tparam N         Matrix dimension (square).
 * @tparam Positions Compile-time list of (row, col) positions to mirror.
 */
template<typename DType, typename IntType, int N, auto Positions>
SPARSEMAT_HD constexpr auto symmetric_from_lower() {
  constexpr auto pattern = detail::mirrored_pattern<N, Positions>();
  return detail::build_from_pattern<DType, IntType, N, N, pattern>(
      std::make_index_sequence<pattern.count>{});
}

/**
 * @brief Builds a zero-valued banded matrix.
 *
 * Position (row, col) is stored when @c -Lower <= col-row <= Upper, i.e.
 * @p Lower sub-diagonals and @p Upper super-diagonals around the main
 * diagonal. @c banded<..., 0, 0>() is diagonal; @c banded<..., 1, 1>() is
 * tridiagonal.
 *
 * @tparam Rows  Number of rows.
 * @tparam Cols  Number of columns.
 * @tparam Lower Number of sub-diagonals to include.
 * @tparam Upper Number of super-diagonals to include.
 */
template<typename DType, typename IntType, int Rows, int Cols, int Lower, int Upper>
SPARSEMAT_HD constexpr auto banded() {
  static_assert(Lower >= 0 && Upper >= 0, "Band widths must be non-negative.");
  constexpr auto pattern = detail::banded_pattern<Rows, Cols, Lower, Upper>();
  return detail::build_from_pattern<DType, IntType, Rows, Cols, pattern>(
      std::make_index_sequence<pattern.count>{});
}

/**
 * @brief Builds a zero-valued N x N tridiagonal matrix.
 *
 * Shorthand for @c banded<DType, IntType, N, N, 1, 1>() — the single most
 * common pattern in practice, and the one the benchmarks use.
 */
template<typename DType, typename IntType, int N>
SPARSEMAT_HD constexpr auto tridiagonal() {
  return banded<DType, IntType, N, N, 1, 1>();
}

/**
 * @brief Builds an N x N tridiagonal matrix with constant bands.
 *
 * @param sub   Value written to every sub-diagonal entry.
 * @param diag  Value written to every diagonal entry.
 * @param super Value written to every super-diagonal entry.
 */
template<typename DType, typename IntType, int N>
SPARSEMAT_HD constexpr auto tridiagonal(DType sub, DType diag, DType super) {
  auto result = tridiagonal<DType, IntType, N>();
  constexpr auto indices = decltype(result)::indices();
  for (std::size_t k = 0; k < indices.size(); ++k) {
    const auto row = indices[k] / N;
    const auto col = indices[k] % N;
    if (col < row) {
      result.values[k] = sub;
    } else if (col > row) {
      result.values[k] = super;
    } else {
      result.values[k] = diag;
    }
  }
  return result;
}

}  // namespace SparseLinearAlgebra
