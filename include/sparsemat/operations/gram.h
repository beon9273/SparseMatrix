#pragma once

#include <cstdint>
#include <type_traits>

#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Which one-sided product a @c Gram instantiation computes.
 */
enum class GramSide : std::uint8_t {
  /// AᵀA — an @c A::cols × @c A::cols matrix of column inner products.
  Transposed,
  /// AAᵀ — an @c A::rows × @c A::rows matrix of row inner products.
  Straight,
};

/**
 * @brief Implementation policy for the Gram products AᵀA and AAᵀ.
 *
 * Neither form materializes the transpose. AᵀA[i,j] is the inner product of
 * columns @c i and @c j of A, and AAᵀ[i,j] the inner product of rows @c i and
 * @c j — both read straight out of A's storage, so the intermediate matrix
 * @c transpose(a).multiply(a) would build (and the sparsity computation it
 * would need) never exists.
 *
 * Result sparsity: for AᵀA, position (i, j) is non-zero when some row @c k has
 * both A[k,i] and A[k,j] stored; for AAᵀ, when some column @c k has both
 * A[i,k] and A[j,k] stored. Either pattern is symmetric by construction.
 *
 * @tparam SparseMat Input matrix type.
 * @tparam Side      Which product to form.
 */
template<SparseMatrixType SparseMat, GramSide Side>
class Gram {
 public:
  using DataType = typename SparseMat::DataType;
  using Int = typename SparseMat::Int;

  /// Length of the summed dimension: A's rows for AᵀA, A's cols for AAᵀ.
  static constexpr Int inner = (Side == GramSide::Transposed) ? SparseMat::rows : SparseMat::cols;
  static constexpr Int rows = (Side == GramSide::Transposed) ? SparseMat::cols : SparseMat::rows;
  static constexpr Int cols = rows;

  // Precomputed once per instantiation — see Multiply::a_grid for why the
  // O(nonZeroCount) scans this replaces are not affordable per lookup.
  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();

  /// Reads A at the (row, col) orientation this side sums over: element @c k
  /// of "vector" @c v is A[k,v] for AᵀA and A[v,k] for AAᵀ.
  SPARSEMAT_HD constexpr static bool a_at(Int k, Int v) {
    if constexpr (Side == GramSide::Transposed) {
      return a_grid[k][v];
    } else {
      return a_grid[v][k];
    }
  }

  /**
   * @brief The result's structural non-zero pattern, computed once.
   *
   * Walks the summed dimension once, and for each slice pairs up the vectors
   * that store an entry in it — O(inner * rows²) worst case, but proportional
   * to the *stored* entries in practice, and paid a single time rather than
   * once per @c is_result_index_nonzero call.
   */
  SPARSEMAT_HD constexpr static auto compute_result_grid() {
    std::array<std::array<bool, static_cast<std::size_t>(cols)>, static_cast<std::size_t>(rows)>
        grid{};
    for (Int k = 0; k < inner; ++k) {
      for (Int i = 0; i < rows; ++i) {
        if (!a_at(k, i)) {
          continue;
        }
        for (Int j = 0; j < cols; ++j) {
          if (a_at(k, j)) {
            grid[i][j] = true;
          }
        }
      }
    }
    return grid;
  }
  static constexpr auto result_grid = compute_result_grid();

  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return result_grid[row][col];
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Gram>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Gram>::calculate_sparsity();
  }

  /// Storage offset of the element of "vector" @p V at summed index @p K, or
  /// -1 when that element is structurally zero.
  template<Int V, Int K>
  SPARSEMAT_HD constexpr static Int offset() {
    if constexpr (Side == GramSide::Transposed) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(K, V);
    } else {
      return SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(V, K);
    }
  }

  /// Single term of the inner product at summed index @p K, elided entirely at
  /// compile time when either factor is structurally zero.
  template<Int I, Int J, std::size_t K>
  SPARSEMAT_HD static DataType inner_product_term(const SparseMat& a) {
    constexpr Int lhs = offset<I, static_cast<Int>(K)>();
    constexpr Int rhs = offset<J, static_cast<Int>(K)>();
    if constexpr (lhs >= 0 && rhs >= 0) {
      return a.values[lhs] * a.values[rhs];
    } else {
      return DataType(0);
    }
  }

  template<Int I, Int J, std::size_t... Ks>
  SPARSEMAT_HD static DataType inner_product_fold(const SparseMat& a,
                                                  std::index_sequence<Ks...> /*seq*/) {
    return (inner_product_term<I, J, Ks>(a) + ...);
  }

  /**
   * @brief Fills result slot @p Idx, mirroring the strict lower triangle from
   *        the upper rather than recomputing it.
   *
   * Both Gram products are symmetric — result[i,j] and result[j,i] are the same
   * inner product — so computing both halves would emit the entire fold twice
   * per off-diagonal pair, doubling the instantiation count of the operations
   * most likely to strain the density budget in the first place.
   *
   * Reading the mirror slot is safe because it is always already written:
   * the pattern is symmetric so (J,I) is stored whenever (I,J) is, its flat
   * row-major index is smaller when @c I > @c J, and fill order follows the
   * result's ascending index array (fill_range expands the low half before the
   * high half, and the comma in a fold expression is the sequencing comma —
   * see the note on chunked unrolling in utils.h).
   */
  template<typename Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r, const SparseMat& a) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr Int I = flat / Result::cols;
    constexpr Int J = flat % Result::cols;
    if constexpr (I > J) {
      constexpr Int mirror = SparseLinearAlgebra::MatrixUtilities<Result>::getSparseIndex(J, I);
      static_assert(mirror >= 0,
                    "Gram result pattern is not symmetric; this should be impossible.");
      r.values[Idx] = r.values[static_cast<std::size_t>(mirror)];
    } else {
      r.values[Idx] =
          inner_product_fold<I, J>(a, std::make_index_sequence<static_cast<std::size_t>(inner)>{});
    }
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const SparseMat& a,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, a), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h for why this is
  /// neither one flat fold (clang caps those at 256 terms) nor a per-element
  /// linear recursion (that exhausts the instantiation-depth budget).
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(Result& r, const SparseMat& a) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, a, std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, a);
      fill_range<Result, Begin + half, Count - half>(r, a);
    }
  }

  SPARSEMAT_HD static auto gram(const SparseMat& a) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(result, a);
    return result;
  }
};

/**
 * @brief Placeholder addend for the congruence forms that have none.
 *
 * @c Congruence covers both the plain triple product and the Joseph form
 * (which adds a matrix to it). Rather than duplicate every fill helper for the
 * two arities, the no-addend case passes this empty type and the addend
 * contributions compile away.
 */
struct NoAddend {};

/// Shape check for the addend, written as a function so the @c NoAddend case
/// never names @c rows/@c cols — a plain @c !has_addend || ... static_assert
/// would still instantiate the right-hand side and fail on the sentinel.
template<typename Addend, auto Rows, auto Cols>
SPARSEMAT_HD constexpr bool addend_shape_matches() {
  if constexpr (std::is_same_v<Addend, NoAddend>) {
    return true;
  } else {
    return Addend::rows == Rows && Addend::cols == Cols;
  }
}

/**
 * @brief Implementation policy for the congruence transforms AᵀBA and ABAᵀ,
 *        optionally with an added matrix (the Joseph form).
 *
 * The product is evaluated in a single traversal — for AᵀBA, result[i,j] is
 * Σ_p Σ_q A[p,i] * B[p,q] * A[q,j] — so neither the transpose nor the
 * intermediate product is ever materialized. Written eagerly as
 * @c a.transpose() * b * a, both intermediates cost a full sparsity
 * computation and a result of their own, and the transpose in particular is
 * pure data movement this form skips by reading A in the other orientation.
 *
 * The two sides are the two halves of a Kalman update: @c ABAᵀ propagates a
 * covariance (@c F P Fᵀ), @c AᵀBA forms a weighted normal matrix
 * (@c Hᵀ R⁻¹ H). With an addend both become the fused forms actually used —
 * @c F P Fᵀ + Q, and the Joseph-form covariance update.
 *
 * Result sparsity is derived through the same intermediate pattern the eager
 * form would produce, then unioned with the addend's.
 *
 * @tparam SparseMatA Outer matrix A.
 * @tparam SparseMatB Inner matrix B; square, matching A's summed dimension.
 * @tparam Side       @c Transposed for AᵀBA, @c Straight for ABAᵀ.
 * @tparam Addend     Matrix added to the product, or @c NoAddend.
 */
template<SparseMatrixType SparseMatA,
         SparseMatrixType SparseMatB,
         GramSide Side,
         typename Addend = NoAddend>
class Congruence {
  static constexpr bool has_addend = !std::is_same_v<Addend, NoAddend>;

 public:
  using DataType = typename SparseMatA::DataType;
  using Int = typename SparseMatA::Int;

  /// Length of the summed dimension, and therefore B's order: A's rows for
  /// AᵀBA, A's cols for ABAᵀ.
  static constexpr Int inner = (Side == GramSide::Transposed) ? SparseMatA::rows : SparseMatA::cols;
  static constexpr Int rows = (Side == GramSide::Transposed) ? SparseMatA::cols : SparseMatA::rows;
  static constexpr Int cols = rows;

  static_assert(SparseMatB::rows == inner && SparseMatB::cols == inner,
                "Incompatible matrix dimensions for a congruence transform: B must be square, "
                "with order A::rows for AᵀBA and A::cols for ABAᵀ.");
  static_assert(SparseLinearAlgebra::SameDataType<SparseMatA, SparseMatB>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types — convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");
  static_assert(addend_shape_matches<Addend, rows, cols>(),
                "The added matrix must have the same shape as the triple product.");

  // Precomputed once per instantiation — see Multiply::a_grid.
  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMatA>::to_dense_bool();
  static constexpr auto b_grid = SparseLinearAlgebra::MatrixUtilities<SparseMatB>::to_dense_bool();

  /// Reads A at the orientation this side sums over: element (k, v) is A[k,v]
  /// for AᵀBA (A is read column-wise) and A[v,k] for ABAᵀ (row-wise).
  SPARSEMAT_HD constexpr static bool a_at(Int k, Int v) {
    if constexpr (Side == GramSide::Transposed) {
      return a_grid[k][v];
    } else {
      return a_grid[v][k];
    }
  }

  /// Storage offset of that same element, or -1 when structurally zero.
  template<Int V, Int K>
  SPARSEMAT_HD constexpr static Int a_offset() {
    if constexpr (Side == GramSide::Transposed) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMatA>::getSparseIndex(K, V);
    } else {
      return SparseLinearAlgebra::MatrixUtilities<SparseMatA>::getSparseIndex(V, K);
    }
  }

  /// Pattern of the inner product B·A (or B·Aᵀ), built by walking B's stored
  /// entries — O(nonZeroCount(B) * cols), the traversal shape Multiply uses
  /// and for the same constexpr-budget reasons.
  SPARSEMAT_HD constexpr static auto compute_ba_grid() {
    std::array<std::array<bool, static_cast<std::size_t>(cols)>, static_cast<std::size_t>(inner)>
        grid{};
    for (auto flat : SparseMatB::indices()) {
      const Int p = flat / SparseMatB::cols;
      const Int q = flat % SparseMatB::cols;
      for (Int j = 0; j < cols; ++j) {
        if (a_at(q, j)) {
          grid[p][j] = true;
        }
      }
    }
    return grid;
  }
  static constexpr auto ba_grid = compute_ba_grid();

  /// Pattern of the full triple product, unioned with the addend's: a stored
  /// A element at summed index p pairs with every non-zero (BA)[p,j].
  SPARSEMAT_HD constexpr static auto compute_result_grid() {
    std::array<std::array<bool, static_cast<std::size_t>(cols)>, static_cast<std::size_t>(rows)>
        grid{};
    for (Int p = 0; p < inner; ++p) {
      for (Int i = 0; i < rows; ++i) {
        if (!a_at(p, i)) {
          continue;
        }
        for (Int j = 0; j < cols; ++j) {
          if (ba_grid[p][j]) {
            grid[i][j] = true;
          }
        }
      }
    }
    if constexpr (has_addend) {
      for (auto flat : Addend::indices()) {
        grid[flat / Addend::cols][flat % Addend::cols] = true;
      }
    }
    return grid;
  }
  static constexpr auto result_grid = compute_result_grid();

  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return result_grid[row][col];
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<Congruence>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<Congruence>::calculate_sparsity();
  }

  /// One B[P,Q]·A term of the inner sum, elided when either factor is
  /// structurally zero.
  template<Int P, Int J, std::size_t Q>
  SPARSEMAT_HD static DataType ba_term(const SparseMatA& a, const SparseMatB& b) {
    constexpr Int b_index =
        SparseLinearAlgebra::MatrixUtilities<SparseMatB>::getSparseIndex(P, static_cast<Int>(Q));
    constexpr Int a_index = a_offset<J, static_cast<Int>(Q)>();
    if constexpr (b_index >= 0 && a_index >= 0) {
      return b.values[b_index] * a.values[a_index];
    } else {
      return DataType(0);
    }
  }

  template<Int P, Int J, std::size_t... Qs>
  SPARSEMAT_HD static DataType ba_fold(const SparseMatA& a,
                                       const SparseMatB& b,
                                       std::index_sequence<Qs...> /*seq*/) {
    return (ba_term<P, J, Qs>(a, b) + ...);
  }

  /// One A·(BA)[P,J] term of the outer sum. The whole inner fold is discarded
  /// at compile time when A's factor is structurally zero — which is what
  /// keeps this cheap for the sparse A these transforms are applied to.
  template<Int I, Int J, std::size_t P>
  SPARSEMAT_HD static DataType outer_term(const SparseMatA& a, const SparseMatB& b) {
    constexpr Int a_index = a_offset<I, static_cast<Int>(P)>();
    if constexpr (a_index >= 0) {
      return a.values[a_index] *
             ba_fold<static_cast<Int>(P), J>(
                 a, b, std::make_index_sequence<static_cast<std::size_t>(inner)>{});
    } else {
      return DataType(0);
    }
  }

  template<Int I, Int J, std::size_t... Ps>
  SPARSEMAT_HD static DataType outer_fold(const SparseMatA& a,
                                          const SparseMatB& b,
                                          std::index_sequence<Ps...> /*seq*/) {
    return (outer_term<I, J, Ps>(a, b) + ...);
  }

  /// The addend's contribution at result slot @p Idx: zero when there is no
  /// addend, or when it is structurally zero there.
  template<typename Result, std::size_t Idx>
  SPARSEMAT_HD static DataType addend_at(const Addend& c) {
    if constexpr (has_addend) {
      constexpr auto flat = Result::indices()[Idx];
      constexpr Int offset =
          SparseLinearAlgebra::MatrixUtilities<Addend>::getSparseIndex(flat / Result::cols,
                                                                       flat % Result::cols);
      if constexpr (offset >= 0) {
        return c.values[static_cast<std::size_t>(offset)];
      } else {
        return DataType(0);
      }
    } else {
      (void)c;
      return DataType(0);
    }
  }

  template<typename Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r,
                                     const SparseMatA& a,
                                     const SparseMatB& b,
                                     const Addend& c) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr Int I = flat / Result::cols;
    constexpr Int J = flat % Result::cols;
    r.values[Idx] =
        outer_fold<I, J>(a, b, std::make_index_sequence<static_cast<std::size_t>(inner)>{}) +
        addend_at<Result, Idx>(c);
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const SparseMatA& a,
                                      const SparseMatB& b,
                                      const Addend& c,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, a, b, c), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h.
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(Result& r,
                                      const SparseMatA& a,
                                      const SparseMatB& b,
                                      const Addend& c) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, a, b, c, std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, a, b, c);
      fill_range<Result, Begin + half, Count - half>(r, a, b, c);
    }
  }

  SPARSEMAT_HD static auto congruence(const SparseMatA& a, const SparseMatB& b, const Addend& c) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMatA>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(result, a, b, c);
    return result;
  }
};

/**
 * @brief Implementation policy for the mixed products AᵀB and ABᵀ.
 *
 * The general two-operand form of the Gram products: same one-pass evaluation,
 * same skipped transpose, but with independent left and right matrices, so the
 * result is not symmetric and nothing can be mirrored.
 *
 * @tparam SparseMatA Left matrix.
 * @tparam SparseMatB Right matrix.
 * @tparam Side       @c Transposed for AᵀB, @c Straight for ABᵀ.
 */
template<SparseMatrixType SparseMatA, SparseMatrixType SparseMatB, GramSide Side>
class TransposedMultiply {
 public:
  using DataType = typename SparseMatA::DataType;
  using Int = typename SparseMatA::Int;

  /// The dimension summed over: the rows both operands share for AᵀB, the
  /// columns they share for ABᵀ.
  static constexpr Int inner = (Side == GramSide::Transposed) ? SparseMatA::rows : SparseMatA::cols;
  static constexpr Int rows = (Side == GramSide::Transposed) ? SparseMatA::cols : SparseMatA::rows;
  static constexpr Int cols = (Side == GramSide::Transposed) ? SparseMatB::cols : SparseMatB::rows;

  static_assert(Side == GramSide::Transposed ? SparseMatB::rows == SparseMatA::rows
                                             : SparseMatB::cols == SparseMatA::cols,
                "Incompatible matrix dimensions: AᵀB requires equal row counts, ABᵀ requires "
                "equal column counts.");
  static_assert(SparseLinearAlgebra::SameDataType<SparseMatA, SparseMatB>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types — convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");

  static constexpr auto a_grid = SparseLinearAlgebra::MatrixUtilities<SparseMatA>::to_dense_bool();
  static constexpr auto b_grid = SparseLinearAlgebra::MatrixUtilities<SparseMatB>::to_dense_bool();

  /// Element @p k of the left operand's "vector" @p v — A[k,v] for AᵀB,
  /// A[v,k] for ABᵀ. The right operand mirrors it.
  SPARSEMAT_HD constexpr static bool a_at(Int k, Int v) {
    if constexpr (Side == GramSide::Transposed) {
      return a_grid[k][v];
    } else {
      return a_grid[v][k];
    }
  }
  SPARSEMAT_HD constexpr static bool b_at(Int k, Int v) {
    if constexpr (Side == GramSide::Transposed) {
      return b_grid[k][v];
    } else {
      return b_grid[v][k];
    }
  }

  template<Int V, Int K>
  SPARSEMAT_HD constexpr static Int a_offset() {
    if constexpr (Side == GramSide::Transposed) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMatA>::getSparseIndex(K, V);
    } else {
      return SparseLinearAlgebra::MatrixUtilities<SparseMatA>::getSparseIndex(V, K);
    }
  }
  template<Int V, Int K>
  SPARSEMAT_HD constexpr static Int b_offset() {
    if constexpr (Side == GramSide::Transposed) {
      return SparseLinearAlgebra::MatrixUtilities<SparseMatB>::getSparseIndex(K, V);
    } else {
      return SparseLinearAlgebra::MatrixUtilities<SparseMatB>::getSparseIndex(V, K);
    }
  }

  SPARSEMAT_HD constexpr static auto compute_result_grid() {
    std::array<std::array<bool, static_cast<std::size_t>(cols)>, static_cast<std::size_t>(rows)>
        grid{};
    for (Int k = 0; k < inner; ++k) {
      for (Int i = 0; i < rows; ++i) {
        if (!a_at(k, i)) {
          continue;
        }
        for (Int j = 0; j < cols; ++j) {
          if (b_at(k, j)) {
            grid[i][j] = true;
          }
        }
      }
    }
    return grid;
  }
  static constexpr auto result_grid = compute_result_grid();

  SPARSEMAT_HD constexpr static auto is_result_index_nonzero(Int row, Int col) {
    return result_grid[row][col];
  }

  /// Delegates to OperationUtilities to count result non-zeros.
  SPARSEMAT_HD constexpr static auto num_nonzeros() {
    return SparseLinearAlgebra::OperationUtilities<TransposedMultiply>::num_nonzeros();
  }

  /// Delegates to OperationUtilities to compute result sparsity indices.
  SPARSEMAT_HD constexpr static auto calculate_sparsity() {
    return SparseLinearAlgebra::OperationUtilities<TransposedMultiply>::calculate_sparsity();
  }

  template<Int I, Int J, std::size_t K>
  SPARSEMAT_HD static DataType inner_product_term(const SparseMatA& a, const SparseMatB& b) {
    constexpr Int lhs = a_offset<I, static_cast<Int>(K)>();
    constexpr Int rhs = b_offset<J, static_cast<Int>(K)>();
    if constexpr (lhs >= 0 && rhs >= 0) {
      return a.values[lhs] * b.values[rhs];
    } else {
      return DataType(0);
    }
  }

  template<Int I, Int J, std::size_t... Ks>
  SPARSEMAT_HD static DataType inner_product_fold(const SparseMatA& a,
                                                  const SparseMatB& b,
                                                  std::index_sequence<Ks...> /*seq*/) {
    return (inner_product_term<I, J, Ks>(a, b) + ...);
  }

  template<typename Result, std::size_t Idx>
  SPARSEMAT_HD static void fill_cell(Result& r, const SparseMatA& a, const SparseMatB& b) {
    constexpr auto flat = Result::indices()[Idx];
    constexpr Int I = flat / Result::cols;
    constexpr Int J = flat % Result::cols;
    r.values[Idx] =
        inner_product_fold<I, J>(a, b, std::make_index_sequence<static_cast<std::size_t>(inner)>{});
  }

  /// Expands one chunk of at most kUnrollChunkSize result slots.
  template<typename Result, std::size_t Begin, std::size_t... Is>
  SPARSEMAT_HD static void fill_chunk(Result& r,
                                      const SparseMatA& a,
                                      const SparseMatB& b,
                                      std::index_sequence<Is...> /*seq*/) {
    (fill_cell<Result, Begin + Is>(r, a, b), ...);
  }

  /// Fills result slots [Begin, Begin+Count) by halving Count until it fits a
  /// single fold. See the note on chunked unrolling in utils.h.
  template<typename Result, std::size_t Begin, std::size_t Count>
  SPARSEMAT_HD static void fill_range(Result& r, const SparseMatA& a, const SparseMatB& b) {
    if constexpr (Count == 0) {
      return;
    } else if constexpr (Count <= SparseLinearAlgebra::kUnrollChunkSize) {
      fill_chunk<Result, Begin>(r, a, b, std::make_index_sequence<Count>{});
    } else {
      constexpr std::size_t half = Count / 2;
      fill_range<Result, Begin, half>(r, a, b);
      fill_range<Result, Begin + half, Count - half>(r, a, b);
    }
  }

  SPARSEMAT_HD static auto product(const SparseMatA& a, const SparseMatB& b) {
    constexpr auto sparsity = calculate_sparsity();
    auto result =
        SparseLinearAlgebra::MatrixUtilities<SparseMatA>::template make<rows, cols, sparsity>(
            std::make_index_sequence<static_cast<std::size_t>(num_nonzeros())>{});
    fill_range<decltype(result), 0, static_cast<std::size_t>(num_nonzeros())>(result, a, b);
    return result;
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Computes the Gram matrix AᵀA.
 *
 * Element (i, j) is the inner product of columns @c i and @c j of @p a, so the
 * result is an @c A::cols × @c A::cols matrix, symmetric by construction.
 *
 * Equivalent to @c multiply(transpose(a), a) but computed in one pass: the
 * transpose is never materialized, the result's sparsity is derived directly
 * from @p a's pattern, and only the upper triangle's inner products are
 * emitted — the lower triangle is mirrored from it.
 *
 * @tparam A Input matrix type.
 * @param  a Input matrix.
 * @return   AᵀA, with sparsity encoded in the type.
 */
template<SparseMatrixType A>
SPARSEMAT_HD auto ata(const A& a) {
  return detail::Gram<A, detail::GramSide::Transposed>::gram(a);
}

/**
 * @brief Computes the Gram matrix AAᵀ.
 *
 * Element (i, j) is the inner product of rows @c i and @c j of @p a, so the
 * result is an @c A::rows × @c A::rows matrix, symmetric by construction.
 * As with @c ata, the transpose is never materialized and only the upper
 * triangle is computed.
 *
 * @tparam A Input matrix type.
 * @param  a Input matrix.
 * @return   AAᵀ, with sparsity encoded in the type.
 */
template<SparseMatrixType A>
SPARSEMAT_HD auto aat(const A& a) {
  return detail::Gram<A, detail::GramSide::Straight>::gram(a);
}

/**
 * @brief Computes AᵀB without materializing Aᵀ.
 *
 * The two-operand generalization of @c ata: element (i, j) is the inner
 * product of column @c i of @p a with column @c j of @p b. Both operands must
 * have the same number of rows; the result is @c A::cols × @c B::cols.
 *
 * @tparam A Left matrix type.
 * @tparam B Right matrix type.
 * @param  a Left matrix.
 * @param  b Right matrix.
 * @return   AᵀB.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto atb(const A& a, const B& b) {
  return detail::TransposedMultiply<A, B, detail::GramSide::Transposed>::product(a, b);
}

/**
 * @brief Computes ABᵀ without materializing Bᵀ.
 *
 * The two-operand generalization of @c aat: element (i, j) is the inner
 * product of row @c i of @p a with row @c j of @p b. Both operands must have
 * the same number of columns; the result is @c A::rows × @c B::rows.
 *
 * @tparam A Left matrix type.
 * @tparam B Right matrix type.
 * @param  a Left matrix.
 * @param  b Right matrix.
 * @return   ABᵀ.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto abt(const A& a, const B& b) {
  return detail::TransposedMultiply<A, B, detail::GramSide::Straight>::product(a, b);
}

/**
 * @brief Computes the congruence transform AᵀBA.
 *
 * @p a is m × n and @p b is m × m, giving an n × n result — the standard form
 * for changing the basis of a quadratic form or assembling a weighted normal
 * equation (with @p b the weight matrix, @c ata is the @c B = I case).
 *
 * Both Aᵀ and the intermediate BA are skipped: the triple product is
 * accumulated in a single traversal, with structurally zero factors dropped at
 * compile time. Note that the result is symmetric only when @p b is.
 *
 * @tparam A Outer matrix type (m × n).
 * @tparam B Inner matrix type (m × m).
 * @param  a Outer matrix.
 * @param  b Inner matrix.
 * @return   AᵀBA, with sparsity encoded in the type.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto atba(const A& a, const B& b) {
  return detail::Congruence<A, B, detail::GramSide::Transposed>::congruence(a,
                                                                            b,
                                                                            detail::NoAddend{});
}

/**
 * @brief Computes the congruence transform ABAᵀ.
 *
 * The other orientation of @c atba: @p a is m × n, @p b is n × n, and the
 * result is m × m. This is the covariance propagation @c F P Fᵀ of a Kalman
 * filter, and likewise skips both Aᵀ and the intermediate BAᵀ.
 *
 * @tparam A Outer matrix type (m × n).
 * @tparam B Inner matrix type (n × n).
 * @param  a Outer matrix.
 * @param  b Inner matrix.
 * @return   ABAᵀ, with sparsity encoded in the type.
 */
template<SparseMatrixType A, SparseMatrixType B>
SPARSEMAT_HD auto abat(const A& a, const B& b) {
  return detail::Congruence<A, B, detail::GramSide::Straight>::congruence(a, b, detail::NoAddend{});
}

/**
 * @brief Computes @c ABAᵀ + C in a single pass.
 *
 * The fused form of a covariance propagation (@c F P Fᵀ + Q) and of the
 * Joseph-form covariance update (@c (I-KH) P (I-KH)ᵀ + K R Kᵀ, one term at a
 * time). The addition costs nothing beyond the wider result pattern: @p c is
 * read at the position already being written, so the intermediate @c ABAᵀ that
 * @c abat(a, b).add(c) would materialize never exists.
 *
 * @tparam A Outer matrix type (m × n).
 * @tparam B Inner matrix type (n × n).
 * @tparam C Addend type (m × m).
 * @param  a Outer matrix.
 * @param  b Inner matrix.
 * @param  c Matrix added to the product.
 * @return   ABAᵀ + C, whose pattern is the union of the product's and @p c's.
 */
template<SparseMatrixType A, SparseMatrixType B, SparseMatrixType C>
SPARSEMAT_HD auto abat_add(const A& a, const B& b, const C& c) {
  return detail::Congruence<A, B, detail::GramSide::Straight, C>::congruence(a, b, c);
}

/**
 * @brief Computes @c AᵀBA + C in a single pass.
 *
 * The @c atba counterpart of @c abat_add — an information-filter update
 * (@c Hᵀ R⁻¹ H + P⁻¹) in one traversal.
 *
 * @tparam A Outer matrix type (m × n).
 * @tparam B Inner matrix type (m × m).
 * @tparam C Addend type (n × n).
 * @param  a Outer matrix.
 * @param  b Inner matrix.
 * @param  c Matrix added to the product.
 * @return   AᵀBA + C, whose pattern is the union of the product's and @p c's.
 */
template<SparseMatrixType A, SparseMatrixType B, SparseMatrixType C>
SPARSEMAT_HD auto atba_add(const A& a, const B& b, const C& c) {
  return detail::Congruence<A, B, detail::GramSide::Transposed, C>::congruence(a, b, c);
}

}  // namespace SparseLinearAlgebra
