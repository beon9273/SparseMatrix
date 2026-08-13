#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra {

/**
 * @brief Which result positions a fused element-wise operation stores.
 *
 * The result pattern cannot be inferred from an arbitrary callable, so it is
 * chosen explicitly. @c Union is the safe default: it stores every position
 * that is non-zero in *any* operand, which is a correct superset for any
 * function satisfying @c f(0, 0, ..., 0) == 0 — true of every arithmetic
 * combination worth fusing.
 */
enum class FusePattern : std::uint8_t {
  /// Store a position if any operand stores it. Correct for any f with
  /// f(0,...,0) == 0; the pattern of a sum, difference, or scaled combination.
  Union,
  /// Store a position only if *every* operand stores it. The pattern of a
  /// product: if any factor is structurally zero the result is too. Tighter
  /// than @c Union, but wrong for anything sum-like, so it is opt-in.
  Intersection,
};

}  // namespace SparseLinearAlgebra

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for a fused element-wise operation over N
 *        operands.
 *
 * Applies a caller-supplied function to the corresponding elements of every
 * operand in a single traversal, writing straight into the result — so a
 * combination like @c 2*A + 3*B - C materializes one matrix instead of three.
 *
 * This generalizes the fusions the library already hand-rolls one case at a
 * time: @c add(a, b, alpha, beta) folds scaling into the addition traversal,
 * @c hadamard(a, b, multiplier) does the same for the product, and @c axpy
 * avoids materializing @c A*x. Each exists to skip an intermediate; this is
 * the same idea with the combining function left to the caller.
 *
 * **Element-wise only.** Every operand is read at the *same* (row, col) as the
 * result. That restriction is what makes fusion unconditionally profitable
 * here: each operand element is touched exactly once per result element, so
 * there is nothing to recompute. Fusing across a matrix product would not be —
 * there each result element reads a whole row and column, so a lazily-evaluated
 * operand would be recomputed O(n) times per result element. (Note that
 * @c axpy fuses the scaling and addition *around* its product while leaving
 * @c A and @c x as stored matrices, which is exactly this boundary.)
 *
 * @tparam Pattern Which positions the result stores.
 * @tparam Fn      Callable taking one @c DataType per operand.
 * @tparam Mats    Operand matrix types; all must share shape and @c DataType.
 */
template<FusePattern Pattern, typename Fn, SparseMatrixType... Mats>
class Fuse {
  using First = std::tuple_element_t<0, std::tuple<Mats...>>;

 public:
  static_assert(sizeof...(Mats) > 0, "fuse requires at least one operand.");
  static_assert(((Mats::rows == First::rows) && ...) && ((Mats::cols == First::cols) && ...),
                "Incompatible matrix dimensions for a fused element-wise operation: every "
                "operand must have the same shape.");
  static_assert((SparseLinearAlgebra::SameDataType<Mats, First> && ...),
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types — convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  using DataType = typename First::DataType;
  using Int = typename First::Int;
  static constexpr auto rows = First::rows;
  static constexpr auto cols = First::cols;

  /// Storage offsets for every operand, memoized once per instantiation.
  /// See MatrixUtilities::storage_index_grid(): this answers "where is
  /// (row, col) stored?" in O(1), rather than re-scanning indices() per lookup.
  template<typename M>
  static constexpr auto grid_for = SparseLinearAlgebra::MatrixUtilities<M>::storage_index_grid();

  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    const auto flat = static_cast<std::size_t>((row * cols) + col);
    if constexpr (Pattern == FusePattern::Intersection) {
      return ((grid_for<Mats>[flat] >= 0) && ...);
    } else {
      return ((grid_for<Mats>[flat] >= 0) || ...);
    }
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Fuse>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Fuse>::calculate_sparsity();
  }

  /// Reads one operand at the position held by result slot @p Idx, yielding a
  /// structural zero without touching storage when it is not stored there.
  template<typename Result, std::size_t Idx, typename M>
  SPARSEMAT_HD static DataType operand_at(const M& m) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr auto offset = grid_for<M>[static_cast<std::size_t>(flat)];
    if constexpr (offset >= 0) {
      return m.values[static_cast<std::size_t>(offset)];
    } else {
      return DataType(0);
    }
  }

  /// Fills result slot @p Idx with @c fn applied to every operand's element
  /// at that position.
  template<typename Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r, const Fn& fn, const Mats&... mats) {
    r.values[Idx] = fn(operand_at<Result, Idx, Mats>(mats)...);
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const Fn& fn,
                                      const Mats&... mats,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, fn, mats...), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h for why this is
  /// neither one flat fold (clang caps those at 256 terms) nor a per-element
  /// linear recursion (that exhausts the instantiation-depth budget).
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(Result& r, const Fn& fn, const Mats&... mats) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, fn, mats..., std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, fn, mats...);
      fill_range<Result, Begin + half, Count - half>(r, fn, mats...);
    }
  }

  /// Constructs the result SparseMat and fills it in a single pass.
  SPARSEMAT_HD static auto fuse(const Fn& fn, const Mats&... mats) {
    constexpr auto sparsity = calculate_sparsity();
    auto result = SparseLinearAlgebra::MatrixUtilities<First>::template make<rows, cols, sparsity>(
        std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(result, fn, mats...);
    return result;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Applies @p fn element-wise across several matrices in a single pass.
 *
 * Each stored position of the result is computed as
 * @c fn(a(i,j), b(i,j), ...), reading every operand once, so a combination
 * that would otherwise materialize an intermediate per operator materializes
 * only the final result:
 *
 * @code
 * // Three temporaries: A*2, B*3, and the sum.
 * auto eager = a.scale(2.0).add(b.scale(3.0)).subtract(c);
 *
 * // None.
 * auto fused = SparseLinearAlgebra::fuse(
 *     [](double x, double y, double z) { return (2 * x) + (3 * y) - z; }, a, b, c);
 * @endcode
 *
 * All operands must have the same shape and the same @c DataType. Operands
 * that are structurally zero at a position contribute @c DataType(0) there,
 * so @p fn always receives one value per operand.
 *
 * **Evaluation is complete when this returns.** Nothing lazy escapes: there is
 * no expression object holding references to the operands, so the usual
 * expression-template hazard — an expression captured with @c auto outliving
 * the matrices it refers to — cannot arise.
 *
 * **Element-wise only.** Every operand is read at the same (row, col) as the
 * result, so this cannot express a matrix product; use @c multiply for that.
 * The restriction is what makes fusing unconditionally worthwhile here — see
 * the note on @c detail::Fuse.
 *
 * @note **Result pattern.** By default the result stores the union of the
 * operands' patterns, which is correct for any @p fn with
 * @c fn(0, ..., 0) == 0 but not always minimal: a position stored by only one
 * operand is stored in the result even if @p fn maps it to zero. Pass
 * @c FusePattern::Intersection for product-like functions, where a
 * structurally zero operand forces a zero result:
 * @code
 * auto weighted = SparseLinearAlgebra::fuse<SparseLinearAlgebra::FusePattern::Intersection>(
 *     [](double x, double y) { return x * y * 0.5; }, a, b);
 * @endcode
 *
 * @note **Device use.** @p fn is called from wherever @c fuse is called, so on
 * the GPU it must be callable from device code. A function object with a
 * @c SPARSEMAT_HD @c operator() works everywhere; so does a lambda written
 * inside a kernel. A lambda defined in host code and passed to a kernel needs
 * nvcc's @c --extended-lambda, which this library does not require of its
 * consumers.
 *
 * @tparam Pattern Which positions the result stores (default @c Union).
 * @tparam Fn      Callable taking one @c DataType per operand.
 * @tparam Mats    Operand matrix types; all must share shape and @c DataType.
 * @param  fn      Combining function.
 * @param  mats    Operands, in the order @p fn expects them.
 * @return         Result matrix, fully evaluated.
 */
template<FusePattern Pattern = FusePattern::Union, typename Fn, SparseMatrixType... Mats>
SPARSEMAT_HD auto fuse(const Fn& fn, const Mats&... mats) {
  return detail::Fuse<Pattern, Fn, Mats...>::fuse(fn, mats...);
}

}  // namespace SparseLinearAlgebra
