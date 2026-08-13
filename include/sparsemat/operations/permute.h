#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra {

/**
 * @brief @c true if @p Perm contains each index in @c [0, N) exactly once.
 *
 * Every entry point that consumes a permutation checks this first: a repeated
 * or out-of-range index would silently duplicate or drop entries, and the
 * resulting matrix would look entirely plausible. Because permutations here are
 * compile-time arrays, the check is a @c static_assert rather than a runtime
 * guard — malformed orderings are rejected at the instantiation that used them.
 *
 * @tparam Perm Permutation array to check.
 * @tparam N    Index range the permutation must cover.
 * @return      @c true if @p Perm is a genuine permutation of @c [0, N).
 */
template<auto Perm, auto N>
SPARSEMAT_HD constexpr bool is_permutation() {
  if (Perm.size() != static_cast<std::size_t>(N)) {
    return false;
  }
  std::array<bool, static_cast<std::size_t>(N)> seen{};
  for (auto p : Perm) {
    if (p < 0 || static_cast<std::size_t>(p) >= static_cast<std::size_t>(N) ||
        seen[static_cast<std::size_t>(p)]) {
      return false;
    }
    seen[static_cast<std::size_t>(p)] = true;
  }
  return true;
}

}  // namespace SparseLinearAlgebra

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for a row/column permutation.
 *
 * @c result[i,j] == @c a[RowPerm[i], ColPerm[j]]. Values move, nothing is
 * computed, and the non-zero count is unchanged — only the pattern's shape
 * changes, which is the entire point (see @c rcm_ordering).
 *
 * @tparam SparseMat Source matrix type.
 * @tparam RowPerm   Array of @c SparseMat::rows source row indices.
 * @tparam ColPerm   Array of @c SparseMat::cols source column indices.
 */
template<SparseMatrixType SparseMat, auto RowPerm, auto ColPerm>
class Permute {
 public:
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr Int rows = SparseMat::rows;
  static constexpr Int cols = SparseMat::cols;

  static_assert(RowPerm.size() == static_cast<std::size_t>(SparseMat::rows),
                "Row permutation length must equal the matrix's row count.");
  static_assert(ColPerm.size() == static_cast<std::size_t>(SparseMat::cols),
                "Column permutation length must equal the matrix's column count.");

  /// Rejects anything that is not a genuine permutation — a repeated or
  /// out-of-range index would silently duplicate or drop entries, and the
  /// resulting matrix would look plausible.
  static_assert(SparseLinearAlgebra::is_permutation<RowPerm, SparseMat::rows>(),
                "Row permutation must contain each index in [0, rows) exactly once.");
  static_assert(SparseLinearAlgebra::is_permutation<ColPerm, SparseMat::cols>(),
                "Column permutation must contain each index in [0, cols) exactly once.");

  static constexpr auto source_grid =
      SparseLinearAlgebra::MatrixUtilities<SparseMat>::storage_index_grid();

  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    const auto flat = (RowPerm[static_cast<std::size_t>(row)] * SparseMat::cols) +
                      ColPerm[static_cast<std::size_t>(col)];
    return source_grid[static_cast<std::size_t>(flat)] >= 0;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Permute>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Permute>::calculate_sparsity();
  }

  /// Source storage offset per result slot, resolved once so the fill is a
  /// flat copy loop — see block.h for why permutation-like operations use a
  /// loop rather than a fold.
  SPARSEMAT_HD constexpr static auto source_slots() {
    constexpr auto sparsity = calculate_sparsity();
    std::array<Int, sparsity.size()> slots{};
    for (std::size_t k = 0; k < sparsity.size(); ++k) {
      const Int i = sparsity[k] / cols;
      const Int j = sparsity[k] % cols;
      slots[k] = source_grid[static_cast<std::size_t>(
          (RowPerm[static_cast<std::size_t>(i)] * SparseMat::cols) +
          ColPerm[static_cast<std::size_t>(j)])];
    }
    return slots;
  }

  SPARSEMAT_HD static auto permute(const SparseMat& a) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    constexpr auto slots = source_slots();
    for (std::size_t k = 0; k < slots.size(); ++k) {
      result.values[k] = a.values[static_cast<std::size_t>(slots[k])];
    }
    return result;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/// The identity permutation of length @p N, for the axis you do not want to
/// touch.
template<typename Int, Int N>
SPARSEMAT_HD constexpr auto identity_permutation() {
  std::array<Int, static_cast<std::size_t>(N)> perm{};
  for (Int i = 0; i < N; ++i) {
    perm[static_cast<std::size_t>(i)] = i;
  }
  return perm;
}

/**
 * @brief Inverts a permutation.
 *
 * @c permute reads @c a[Perm[i]] into result row @c i, so a solution computed
 * in permuted coordinates has to be mapped back through the inverse:
 * @c x[Perm[i]] == @c y[i], i.e. @c y[inverse[k]] is the value belonging to
 * original index @c k.
 *
 * @p Perm is a template parameter rather than a function argument so that it can
 * be validated: inverting scatters through @c perm[i] as an index, so an entry
 * outside @c [0, N) would write past the end of the result. Every other
 * permutation entry point in this header rejects a malformed ordering at compile
 * time, and so does this one.
 *
 * @tparam Perm Permutation to invert.
 * @return      Its inverse.
 */
template<auto Perm>
SPARSEMAT_HD constexpr auto inverse_permutation() {
  static_assert(is_permutation<Perm, Perm.size()>(),
                "inverse_permutation requires a genuine permutation: each index in [0, N) "
                "exactly once.");
  using Int = std::remove_cvref_t<decltype(Perm[0])>;
  std::array<Int, Perm.size()> inverse{};
  for (std::size_t i = 0; i < Perm.size(); ++i) {
    inverse[static_cast<std::size_t>(Perm[i])] = static_cast<Int>(i);
  }
  return inverse;
}

namespace detail {

/**
 * @brief Symmetrized adjacency of a matrix type's pattern, excluding the
 *        diagonal.
 *
 * An edge is taken when either (i, j) or (j, i) is stored, which is what makes
 * @c rcm_ordering meaningful for structurally non-symmetric matrices. Self-edges
 * are dropped: they tell the traversal nothing.
 *
 * @tparam A Square matrix type.
 * @return   @c adj[i][j] — whether nodes @c i and @c j are adjacent.
 */
template<SparseMatrixType A>
SPARSEMAT_HD constexpr auto symmetrized_adjacency() {
  constexpr auto n = static_cast<std::size_t>(A::rows);
  std::array<std::array<bool, n>, n> adj{};
  for (auto flat : A::indices()) {
    const auto i = static_cast<std::size_t>(flat / A::cols);
    const auto j = static_cast<std::size_t>(flat % A::cols);
    if (i != j) {
      adj[i][j] = true;
      adj[j][i] = true;
    }
  }
  return adj;
}

/// Neighbour count per node.
template<typename Int, std::size_t N>
SPARSEMAT_HD constexpr auto adjacency_degrees(const std::array<std::array<bool, N>, N>& adj) {
  std::array<Int, N> degree{};
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (adj[i][j]) {
        ++degree[i];
      }
    }
  }
  return degree;
}

/// Lowest-degree unvisited node, or @c N if every node has been visited. The
/// standard cheap substitute for a pseudo-peripheral start node.
template<typename Int, std::size_t N>
SPARSEMAT_HD constexpr std::size_t lowest_degree_unvisited(const std::array<Int, N>& degree,
                                                           const std::array<bool, N>& visited) {
  std::size_t best = N;
  for (std::size_t i = 0; i < N; ++i) {
    if (!visited[i] && (best == N || degree[i] < degree[best])) {
      best = i;
    }
  }
  return best;
}

/// Lowest-degree unvisited neighbour of @p current, or @c N if it has none left.
/// Visiting neighbours in ascending degree order is what makes the traversal
/// Cuthill–McKee rather than a plain breadth-first walk.
template<typename Int, std::size_t N>
SPARSEMAT_HD constexpr std::size_t lowest_degree_neighbour(
    const std::array<std::array<bool, N>, N>& adj,
    const std::array<Int, N>& degree,
    const std::array<bool, N>& visited,
    std::size_t current) {
  std::size_t best = N;
  for (std::size_t j = 0; j < N; ++j) {
    if (adj[current][j] && !visited[j] && (best == N || degree[j] < degree[best])) {
      best = j;
    }
  }
  return best;
}

}  // namespace detail

/**
 * @brief Bandwidth of a matrix type's pattern: the largest @c |i-j| over its
 *        stored entries.
 *
 * A direct measure of what a fill-reducing ordering is trying to shrink — the
 * factors of a banded matrix stay inside the band, so a smaller bandwidth
 * bounds the fill-in that @c lu_factorize and @c cholesky_factorize produce.
 *
 * @tparam A Matrix type to measure.
 * @return   Bandwidth, or 0 for an empty pattern.
 */
template<SparseMatrixType A>
SPARSEMAT_HD constexpr auto bandwidth() {
  using Int = typename A::Int;
  Int worst = 0;
  for (auto flat : A::indices()) {
    const Int i = flat / A::cols;
    const Int j = flat % A::cols;
    const Int spread = i > j ? i - j : j - i;
    if (spread > worst) {
      worst = spread;
    }
  }
  return worst;
}

/**
 * @brief Computes a reverse Cuthill–McKee ordering for a matrix type's pattern.
 *
 * RCM renumbers a symmetric-ish pattern to pull its entries towards the
 * diagonal, which shrinks the bandwidth and with it the fill-in a subsequent
 * factorization produces. Ordinary sparse libraries have to do this at runtime
 * on the matrix in hand; here the pattern is a compile-time constant, so the
 * whole ordering is derived at compile time and baked into the permuted type.
 * That makes it a direct attack on this library's real cost centre — the
 * factor's stored-value count is what drives compile time and binary size, and
 * a reordering that halves it halves both.
 *
 * Usage is two steps, because the reordering has to be undone on the solution:
 *
 * @code
 * constexpr auto p = SparseLinearAlgebra::rcm_ordering<decltype(a)>();
 * auto reordered = SparseLinearAlgebra::symmetric_permute<p>(a);   // P A Pᵀ
 * auto y = SparseLinearAlgebra::cholesky_solve(reordered, permute_rows<p>(b));
 * // y is in permuted coordinates: original index p[i] holds y[i].
 * @endcode
 *
 * The pattern is symmetrized first (an edge is taken when either (i,j) or
 * (j,i) is stored), so this is meaningful for structurally non-symmetric
 * matrices too. Each component starts from its lowest-degree unvisited node —
 * the standard cheap substitute for a pseudo-peripheral start, which costs
 * extra passes for a usually-marginal improvement — and disconnected
 * components are handled by restarting.
 *
 * @tparam A Matrix type whose pattern to reorder; must be square.
 * @return   Permutation array: @c perm[i] is the original index that moves to
 *           position @c i.
 */
template<SparseMatrixType A>
SPARSEMAT_HD constexpr auto rcm_ordering() {
  using Int = typename A::Int;
  static_assert(A::rows == A::cols, "rcm_ordering requires a square matrix.");
  constexpr auto n = static_cast<std::size_t>(A::rows);

  constexpr auto adj = detail::symmetrized_adjacency<A>();
  constexpr auto degree = detail::adjacency_degrees<Int>(adj);

  std::array<Int, n> order{};
  std::array<bool, n> visited{};
  std::size_t head = 0;
  std::size_t tail = 0;

  while (tail < n) {
    if (head == tail) {
      // Start (or restart, for a disconnected component) at the unvisited node
      // of lowest degree.
      const auto start = detail::lowest_degree_unvisited(degree, visited);
      visited[start] = true;
      order[tail++] = static_cast<Int>(start);
    }

    const auto current = static_cast<std::size_t>(order[head++]);
    // Enqueue unvisited neighbours in ascending degree order — the choice that
    // makes this Cuthill–McKee rather than a plain breadth-first walk.
    for (auto next = detail::lowest_degree_neighbour(adj, degree, visited, current); next != n;
         next = detail::lowest_degree_neighbour(adj, degree, visited, current)) {
      visited[next] = true;
      order[tail++] = static_cast<Int>(next);
    }
  }

  // Reversing is what makes it *reverse* Cuthill–McKee: same bandwidth, but
  // reliably less fill-in during factorization.
  std::array<Int, n> perm{};
  for (std::size_t i = 0; i < n; ++i) {
    perm[i] = order[n - 1 - i];
  }
  return perm;
}

/**
 * @brief Permutes rows and columns: @c result[i,j] == @c a[RowPerm[i], ColPerm[j]].
 *
 * Both permutations are compile-time arrays, so the reordered pattern — and
 * therefore the result type — is known at compile time. The non-zero count is
 * unchanged; only the shape of the pattern moves.
 *
 * @tparam RowPerm Array of source row indices, one per result row.
 * @tparam ColPerm Array of source column indices, one per result column.
 * @tparam A       Source matrix type.
 * @param  a       Source matrix.
 * @return         Permuted matrix.
 */
template<auto RowPerm, auto ColPerm, SparseMatrixType A>
SPARSEMAT_HD auto permute(const A& a) {
  return detail::Permute<A, RowPerm, ColPerm>::permute(a);
}

/**
 * @brief Applies the same permutation to rows and columns: @c P A Pᵀ.
 *
 * The form a fill-reducing ordering takes for a square system, and the one
 * that preserves symmetry and definiteness — so a symmetric positive definite
 * matrix stays Cholesky-solvable after reordering.
 *
 * @tparam Perm Permutation array.
 * @tparam A    Source matrix type (square).
 * @param  a    Source matrix.
 * @return      @c P A Pᵀ.
 */
template<auto Perm, SparseMatrixType A>
SPARSEMAT_HD auto symmetric_permute(const A& a) {
  static_assert(A::rows == A::cols, "symmetric_permute requires a square matrix.");
  return detail::Permute<A, Perm, Perm>::permute(a);
}

/**
 * @brief Permutes rows only — for moving a right-hand side into the same
 *        coordinates as a reordered system.
 *
 * @tparam Perm Permutation array.
 * @tparam A    Source matrix type.
 * @param  a    Source matrix.
 * @return      @c P A.
 */
template<auto Perm, SparseMatrixType A>
SPARSEMAT_HD auto permute_rows(const A& a) {
  return detail::Permute<A, Perm, identity_permutation<typename A::Int, A::cols>()>::permute(a);
}

/**
 * @brief Permutes columns only.
 *
 * @tparam Perm Permutation array.
 * @tparam A    Source matrix type.
 * @param  a    Source matrix.
 * @return      @c A Pᵀ.
 */
template<auto Perm, SparseMatrixType A>
SPARSEMAT_HD auto permute_cols(const A& a) {
  return detail::Permute<A, identity_permutation<typename A::Int, A::rows>(), Perm>::permute(a);
}

}  // namespace SparseLinearAlgebra
