#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for extracting a rectangular sub-block.
 *
 * The complement of @c block_diagonal: that one assembles a large matrix from
 * independent pieces, this one takes the pieces back out — the partitioned
 * state vectors and measurement models that motivate block assembly need both
 * directions.
 *
 * Purely structural. Every stored entry of the window keeps its value and is
 * re-indexed into the smaller grid, so the result's non-zero count is however
 * many of A's entries fall inside it. Nothing is computed, which is why the
 * fill is an ordinary runtime loop over a precomputed table rather than a fold
 * — see @c MatrixUtilities::storage_index_grid() for why that is the right
 * shape for value-moving operations.
 *
 * @tparam SparseMat Source matrix type.
 * @tparam Row0      First row of the window.
 * @tparam Col0      First column of the window.
 * @tparam NRows     Window height.
 * @tparam NCols     Window width.
 */
template<SparseMatrixType SparseMat,
         typename SparseMat::Int Row0,
         typename SparseMat::Int Col0,
         typename SparseMat::Int NRows,
         typename SparseMat::Int NCols>
class Submatrix {
 public:
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;
  static constexpr Int rows = NRows;
  static constexpr Int cols = NCols;

  static_assert(NRows > 0 && NCols > 0, "A submatrix must have a positive extent.");
  static_assert(Row0 >= 0 && Col0 >= 0, "A submatrix origin must be non-negative.");
  static_assert(Row0 + NRows <= SparseMat::rows && Col0 + NCols <= SparseMat::cols,
                "Submatrix window extends past the edge of the source matrix.");

  static constexpr auto source_grid =
      SparseLinearAlgebra::MatrixUtilities<SparseMat>::storage_index_grid();

  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    const auto flat = ((row + Row0) * SparseMat::cols) + (col + Col0);
    return source_grid[static_cast<std::size_t>(flat)] >= 0;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Submatrix>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Submatrix>::calculate_sparsity();
  }

  /// Source storage offset for each result slot, resolved once at compile time
  /// so the fill itself is a flat copy loop.
  SPARSEMAT_HD constexpr static auto source_slots() {
    constexpr auto sparsity = calculate_sparsity();
    std::array<Int, sparsity.size()> slots{};
    for (std::size_t k = 0; k < sparsity.size(); ++k) {
      const Int i = sparsity[k] / cols;
      const Int j = sparsity[k] % cols;
      slots[k] = source_grid[static_cast<std::size_t>(((i + Row0) * SparseMat::cols) + (j + Col0))];
    }
    return slots;
  }

  SPARSEMAT_HD static auto submatrix(const SparseMat& a) {
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

/// Which way a concatenation joins its operands.
enum class ConcatAxis : std::uint8_t {
  /// Side by side: same row count, column counts add.
  Horizontal,
  /// Stacked: same column count, row counts add.
  Vertical,
};

/**
 * @brief Implementation policy for concatenating two matrices.
 *
 * Like @c Submatrix, this only moves values: each operand's entries keep their
 * value and are re-indexed into the combined grid, so the result stores
 * exactly @c nnz(a) + @c nnz(b) values. @c block_diagonal is the special case
 * where the two operands are offset along both axes at once.
 *
 * @tparam SparseMatA Left/top operand.
 * @tparam SparseMatB Right/bottom operand.
 * @tparam Axis       Which way to join them.
 */
template<SparseMatrixType SparseMatA, SparseMatrixType SparseMatB, ConcatAxis Axis>
class Concat {
 public:
  using DataType = typename SparseMatA::DataType;
  using Int = typename SparseMatA::Int;
  static constexpr bool horizontal = (Axis == ConcatAxis::Horizontal);
  static constexpr Int rows = horizontal ? SparseMatA::rows : SparseMatA::rows + SparseMatB::rows;
  static constexpr Int cols = horizontal ? SparseMatA::cols + SparseMatB::cols : SparseMatA::cols;

  static_assert(horizontal ? SparseMatB::rows == SparseMatA::rows
                           : SparseMatB::cols == SparseMatA::cols,
                "Incompatible matrix dimensions for concatenation: hcat requires equal row "
                "counts, vcat requires equal column counts.");
  static_assert(SparseLinearAlgebra::SameDataType<SparseMatA, SparseMatB>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types — convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  /// Where B's block starts in the combined grid.
  static constexpr Int row_offset = horizontal ? Int(0) : SparseMatA::rows;
  static constexpr Int col_offset = horizontal ? SparseMatA::cols : Int(0);

  static constexpr auto a_slots =
      SparseLinearAlgebra::MatrixUtilities<SparseMatA>::storage_index_grid();
  static constexpr auto b_slots =
      SparseLinearAlgebra::MatrixUtilities<SparseMatB>::storage_index_grid();

  /// True inside A's block when A stores the position, inside B's when B does.
  /// The blocks are disjoint by construction, so no position is claimed twice.
  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    if (row < SparseMatA::rows && col < SparseMatA::cols) {
      return a_slots[static_cast<std::size_t>((row * SparseMatA::cols) + col)] >= 0;
    }
    if (row >= row_offset && col >= col_offset) {
      const Int i = row - row_offset;
      const Int j = col - col_offset;
      if (i < SparseMatB::rows && j < SparseMatB::cols) {
        return b_slots[static_cast<std::size_t>((i * SparseMatB::cols) + j)] >= 0;
      }
    }
    return false;
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Concat>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Concat>::calculate_sparsity();
  }

  /// For each result slot, which operand supplies it and at which offset.
  /// Encoded as a signed offset per operand, -1 meaning "not this one", so the
  /// fill is two flat copy loops with no branching on structure.
  SPARSEMAT_HD constexpr static auto source_slots() {
    constexpr auto sparsity = calculate_sparsity();
    std::array<std::array<Int, 2>, sparsity.size()> slots{};
    for (std::size_t k = 0; k < sparsity.size(); ++k) {
      const Int i = sparsity[k] / cols;
      const Int j = sparsity[k] % cols;
      slots[k][0] = Int(-1);
      slots[k][1] = Int(-1);
      if (i < SparseMatA::rows && j < SparseMatA::cols) {
        slots[k][0] = a_slots[static_cast<std::size_t>((i * SparseMatA::cols) + j)];
      }
      if (slots[k][0] < 0) {
        const Int bi = i - row_offset;
        const Int bj = j - col_offset;
        slots[k][1] = b_slots[static_cast<std::size_t>((bi * SparseMatB::cols) + bj)];
      }
    }
    return slots;
  }

  SPARSEMAT_HD static auto concat(const SparseMatA& a, const SparseMatB& b) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMatA>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    constexpr auto slots = source_slots();
    for (std::size_t k = 0; k < slots.size(); ++k) {
      result.values[k] = slots[k][0] >= 0 ? a.values[static_cast<std::size_t>(slots[k][0])]
                                          : b.values[static_cast<std::size_t>(slots[k][1])];
    }
    return result;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Extracts the @p NRows × @p NCols block of @p a whose top-left corner
 *        is (@p Row0, @p Col0).
 *
 * The window is a compile-time constant, so the result's sparsity — the subset
 * of @p a's entries that fall inside it — is too. Values are copied unchanged.
 *
 * @code
 * auto top_left = SparseLinearAlgebra::submatrix<0, 0, 2, 2>(a);
 * @endcode
 *
 * @tparam Row0  First row of the window.
 * @tparam Col0  First column of the window.
 * @tparam NRows Window height.
 * @tparam NCols Window width.
 * @tparam A     Source matrix type.
 * @param  a     Source matrix.
 * @return       The extracted block.
 */
template<auto Row0, auto Col0, auto NRows, auto NCols, SparseMatrixType A>
SPARSEMAT_HD auto submatrix(const A& a) {
  using Int = typename A::Int;
  return detail::Submatrix<A,
                           static_cast<Int>(Row0),
                           static_cast<Int>(Col0),
                           static_cast<Int>(NRows),
                           static_cast<Int>(NCols)>::submatrix(a);
}

/**
 * @brief Extracts row @p I of @p a as a 1 × @c A::cols matrix.
 *
 * @tparam I Row index.
 * @tparam A Source matrix type.
 * @param  a Source matrix.
 * @return   The row, as a matrix.
 */
template<auto I, SparseMatrixType A>
SPARSEMAT_HD auto row(const A& a) {
  return submatrix<I, 0, 1, A::cols>(a);
}

/**
 * @brief Extracts column @p J of @p a as an @c A::rows × 1 matrix.
 *
 * @tparam J Column index.
 * @tparam A Source matrix type.
 * @param  a Source matrix.
 * @return   The column, as a matrix.
 */
template<auto J, SparseMatrixType A>
SPARSEMAT_HD auto col(const A& a) {
  return submatrix<0, J, A::rows, 1>(a);
}

/**
 * @brief Joins @p a and @p b side by side: @c [A B].
 *
 * Both must have the same row count. The result stores exactly
 * @c nnz(a) + @c nnz(b) values — concatenation never creates or destroys a
 * non-zero.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param  a Left operand.
 * @param  b Right operand.
 * @return   @c A::rows × (@c A::cols + @c B::cols) matrix.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto hcat(const A& a, const B& b) {
  return detail::Concat<A, B, detail::ConcatAxis::Horizontal>::concat(a, b);
}

/**
 * @brief Stacks @p a above @p b: @c [A; B].
 *
 * Both must have the same column count. As with @c hcat, the result stores
 * exactly @c nnz(a) + @c nnz(b) values.
 *
 * @tparam A Top operand type.
 * @tparam B Bottom operand type.
 * @param  a Top operand.
 * @param  b Bottom operand.
 * @return   (@c A::rows + @c B::rows) × @c A::cols matrix.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto vcat(const A& a, const B& b) {
  return detail::Concat<A, B, detail::ConcatAxis::Vertical>::concat(a, b);
}

}  // namespace SparseLinearAlgebra
