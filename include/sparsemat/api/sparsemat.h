#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <type_traits>
#include <tuple>


#include "sparsemat/operations/add.h"
#include "sparsemat/operations/axpy.h"
#include "sparsemat/operations/block.h"
#include "sparsemat/operations/block_diagonal.h"
#include "sparsemat/operations/cholesky.h"
#include "sparsemat/operations/compare.h"
#include "sparsemat/operations/convert.h"
#include "sparsemat/operations/dense.h"
#include "sparsemat/operations/diagonal.h"
#include "sparsemat/operations/fuse.h"
#include "sparsemat/operations/gram.h"
#include "sparsemat/operations/hadamard.h"
#include "sparsemat/operations/invert.h"
#include "sparsemat/operations/kronecker.h"
#include "sparsemat/operations/least_squares.h"
#include "sparsemat/operations/lu.h"
#include "sparsemat/operations/multiply.h"
#include "sparsemat/operations/permute.h"
#include "sparsemat/operations/rank_update.h"
#include "sparsemat/operations/scale.h"
#include "sparsemat/operations/shift.h"
#include "sparsemat/operations/symmetric.h"
#include "sparsemat/operations/trace.h"
#include "sparsemat/operations/transpose.h"
#include "sparsemat/operations/triangular.h"
#include "sparsemat/operations/utils.h"
#include "sparsemat/version.h"

namespace SparseLinearAlgebra {

/**
 * @brief Compile-time sparse matrix whose non-zero positions are encoded as
 *        template parameters.
 *
 * Non-zero positions are supplied as flat (row-major) indices via the
 * @p NonZeros pack.  Only the values at those positions are stored, so the
 * @c values array has exactly @c nonZeroCount elements.  All arithmetic
 * operations produce a new @c SparseMat whose sparsity pattern is computed
 * at compile time.
 *
 * @tparam DType    Scalar element type (e.g. @c float, @c double).
 * @tparam IntType  Signed integer type used for indices/offsets that may be
 *                  negative (e.g. "not found" sentinels or out-of-bounds
 *                  runtime indices). Defaults to @c int.
 * @tparam Rows     Number of rows.
 * @tparam Cols     Number of columns.
 * @tparam NonZeros Flat (row-major) indices of every non-zero element.
 */
template<typename DType, typename IntType, int Rows, int Cols, IntType... NonZeros>
class SparseMat {
  static_assert(std::is_signed_v<IntType>, "SparseMat::Int must be a signed integer type.");
  static_assert(Rows > 0 && Cols > 0,
                "SparseMat dimensions must be positive (Rows > 0 and Cols > 0).");
  // Compile time and binary size scale with the stored-value count, so a
  // runaway-density result is worth naming explicitly rather than letting it
  // surface as a mysteriously slow build or an inscrutable compiler limit.
  // See SPARSEMAT_MAX_NONZEROS in concepts/concepts.h to raise, lower, or
  // disable this.
  static_assert(SPARSEMAT_MAX_NONZEROS == 0 ||
                    sizeof...(NonZeros) <= static_cast<std::size_t>(SPARSEMAT_MAX_NONZEROS),
                "This SparseMat exceeds SPARSEMAT_MAX_NONZEROS stored values. Compile time "
                "and binary size scale with that count, so this usually means an operation "
                "chain produced a far denser result than intended (multiply and kronecker "
                "both can). Check the density of the operands, or raise/disable the ceiling "
                "by defining SPARSEMAT_MAX_NONZEROS before including sparsemat.");

  /// Helper that builds an N×M identity matrix from an index sequence.
  template<std::size_t... Is>
  SPARSEMAT_HD static auto make_identity_impl(std::index_sequence<Is...> /*seq*/) {
    SparseMat<DType, IntType, rows, cols, (static_cast<IntType>(Is) * (cols + 1))...> result{};
    result.values.fill(1);
    return result;
  }

 public:
  /**
   * @brief Rebinds this template to a different shape and non-zero set.
   * @tparam R  New row count.
   * @tparam C  New column count.
   * @tparam NZ New non-zero flat indices.
   */
  template<IntType R, IntType C, IntType... NZ>
  using Rebind = SparseMat<DType, IntType, R, C, NZ...>;

  /**
   * @brief Rebinds this template to a different scalar type, keeping the shape
   *        and sparsity pattern.
   *
   * Used by @c convert() — the counterpart to @c Rebind, which varies the
   * shape while holding the scalar type fixed.
   *
   * @tparam D New scalar element type.
   */
  template<typename D>
  using RebindData = SparseMat<D, IntType, Rows, Cols, NonZeros...>;

  /// Scalar element type.
  using DataType = DType;
  /// Signed integer type used for indices/offsets that may be negative.
  using Int = IntType;
  static constexpr IntType rows = Rows;  ///< Number of rows.
  static constexpr IntType cols = Cols;  ///< Number of columns.
  static constexpr IntType nonZeroCount =
      sizeof...(NonZeros);  ///< Number of stored (non-zero) elements.
  std::array<DataType, nonZeroCount>
      values{};  ///< Values of the stored (non-zero) elements, in index order.

  /**
   * @brief Return the nonzero indices. This is a function rather than
   * a static array, so that the returned array is a local copy and can be
   * safely used in @c SPARSEMAT_HD code on both host and device.
   */
  SPARSEMAT_HD static constexpr auto indices() {
    return std::array<IntType, sizeof...(NonZeros)>{NonZeros...};
  }

  /**
   * @brief Validates the sparsity pattern given in the @c NonZeros pack.
   *
   * Checks that every index is non-negative, within bounds (< Rows*Cols), and
   * that no index is repeated. Evaluated by the @c static_assert immediately
   * below, so an invalid pattern is a compile error at the point of
   * declaration.
   *
   * Duplicate detection marks each index in a @c Rows*Cols bitmap rather than
   * comparing every pair, making this O(Rows*Cols + nonZeroCount) instead of
   * O(nonZeroCount²). That matters because this assert fires for *every*
   * @c SparseMat instantiation, including the fully-dense results of
   * @c dense(): at 32x32 the quadratic form was ~1M constexpr operations,
   * which is enough to exhaust nvcc's constexpr-evaluation budget ("excessive
   * constexpr function call complexity") even though g++ and clang accept it.
   * This was the real remaining ceiling on matrix size.
   *
   * @return @c true if the sparsity pattern is valid, @c false otherwise.
   */
  [[nodiscard]] SPARSEMAT_HD static constexpr bool validate_indices() {
    constexpr auto inds = indices();
    std::array<bool, static_cast<std::size_t>(Rows) * static_cast<std::size_t>(Cols)> seen{};
    for (IntType i = 0; i < nonZeroCount; ++i) {
      if (inds[i] < 0) {
        return false;
      }
      if (inds[i] >= rows * cols) {
        return false;
      }
      const auto slot = static_cast<std::size_t>(inds[i]);
      if (seen[slot]) {
        return false;  // duplicate index
      }
      seen[slot] = true;
    }
    return true;
  }
  static_assert(validate_indices(), "Sparsity pattern indices are invalid.");

  // --- Constructors ---

  /// Default-constructs all non-zero values to zero.
  SparseMat() = default; // No annotation as it causes warning. 

  /**
   * @brief Constructs from a pre-built values array.
   * @param vals Array of @c nonZeroCount values in index order.
   */
  SPARSEMAT_HD SparseMat(std::array<DataType, nonZeroCount> vals) : values(std::move(vals)) {}

  /**
   * @brief Variadic constructor; each argument initialises one non-zero slot.
   *
   * The number of arguments must equal @c nonZeroCount, and every argument
   * must be convertible to @c DataType.
   *
   * Both requirements are constraints rather than a body @c static_assert so
   * that this constructor drops out of overload resolution instead of hard-
   * erroring when it doesn't apply. Unconstrained, it accepts any argument
   * list at all as far as overload resolution can see: @c SparseMat would
   * satisfy @c std::is_constructible_v with completely unrelated types, would
   * act as a greedy converting constructor in any overload set it appears in,
   * and a genuinely bad call would surface as a @c static_cast error deep
   * inside the constructor body rather than as "no matching constructor".
   *
   * @tparam Vals Deduced value types (must be convertible to @c DataType).
   * @param  vals Values in the same order as the @c NonZeros index pack.
   */
  template<typename... Vals>
    requires(sizeof...(Vals) == static_cast<std::size_t>(nonZeroCount) &&
             (std::is_convertible_v<Vals, DType> && ...))
  SPARSEMAT_HD SparseMat(Vals... vals) : values{static_cast<DataType>(vals)...} {}

  // --- Static factories ---

  /**
   * @brief Creates an identity matrix with the current dimensions.
   *
   * The returned type has non-zeros only on the main diagonal.
   *
   * @return   Identity matrix with @c DataType values of 1.
   */
  SPARSEMAT_HD static auto identity() {
    constexpr auto N = (rows < cols) ? rows : cols;
    return make_identity_impl(std::make_index_sequence<N>{});
  }

  // --- Element access ---

  /**
   * @brief Compile-time element read at position (I, J).
   *
   * Returns the stored value if (I, J) is a non-zero position, otherwise
   * returns @c DataType(0) without any storage access.
   *
   * @tparam I Row index.
   * @tparam J Column index.
   * @return   Element value, or zero if the position is structurally zero.
   */
  template<IntType I, IntType J>
  [[nodiscard]] SPARSEMAT_HD DataType get() const {
    if constexpr (MatrixUtilities<SparseMat>::isNonZero(I, J)) {
      constexpr auto index = MatrixUtilities<SparseMat>::getSparseIndex(I, J);
      return values[index];
    }
    return static_cast<DataType>(0);
  }

  /**
   * @brief Runtime element read at position (i, j).
   *
   * Returns @c DataType(0) for structurally zero positions without touching
   * @c values.
   *
   * @param i Row index.
   * @param j Column index.
   * @return  Element value, or zero if the position is structurally zero.
   */
  [[nodiscard]] SPARSEMAT_HD DataType get(Int i, Int j) const {
    if (i < 0 || j < 0 || i >= static_cast<Int>(rows) || j >= static_cast<Int>(cols)) {
      return static_cast<DataType>(0);
    }
    if (MatrixUtilities<SparseMat>::isNonZero(i, j)) {
      auto index = MatrixUtilities<SparseMat>::getSparseIndex(i, j);
      return values[index];
    }
    return static_cast<DataType>(0);
  }

  /**
   * @brief Runtime element read at position (i, j); alias for @c get(i, j).
   *
   * Read-only by design: there is deliberately no reference-returning
   * overload, because a structurally zero position has no storage to hand
   * back a reference to, and returning a reference to a shared dummy zero
   * would let `m(0, 1) = 5.0` silently do nothing. Use @c set(i, j, value),
   * which reports whether the write landed.
   *
   * @param i Row index.
   * @param j Column index.
   * @return  Element value, or zero if the position is structurally zero.
   */
  [[nodiscard]] SPARSEMAT_HD DataType operator()(Int i, Int j) const { return get(i, j); }

  // --- Iteration over stored values ---
  //
  // Iterates the packed storage, i.e. exactly the non-zero positions, in flat
  // row-major index order. Pair with indices() (a parallel array) or use
  // entries() below when the positions matter too.

  /// Number of stored (non-zero) elements — the length of the iteration range.
  [[nodiscard]] SPARSEMAT_HD constexpr std::size_t size() const {
    return static_cast<std::size_t>(nonZeroCount);
  }

  /// @c true if this matrix stores no values at all (the structural zero matrix).
  [[nodiscard]] SPARSEMAT_HD constexpr bool empty() const { return nonZeroCount == 0; }

  [[nodiscard]] SPARSEMAT_HD auto begin() { return values.begin(); }
  [[nodiscard]] SPARSEMAT_HD auto end() { return values.end(); }
  [[nodiscard]] SPARSEMAT_HD auto begin() const { return values.begin(); }
  [[nodiscard]] SPARSEMAT_HD auto end() const { return values.end(); }
  [[nodiscard]] SPARSEMAT_HD auto cbegin() const { return values.cbegin(); }
  [[nodiscard]] SPARSEMAT_HD auto cend() const { return values.cend(); }

  /// One stored element, as returned by @c entries().
  struct Entry {
    Int row;         ///< Row index of the stored element.
    Int col;         ///< Column index of the stored element.
    DataType value;  ///< The stored value.
  };

  /**
   * @brief Returns every stored element as a (row, col, value) triple.
   *
   * Saves callers from re-deriving positions out of @c indices() by hand,
   * which previously was the only way to iterate structure and values
   * together. Returned by value (like @c indices()) so the result is a local
   * copy usable from both host and device code.
   *
   * @code
   * for (auto [row, col, value] : m.entries()) { ... }
   * @endcode
   */
  [[nodiscard]] SPARSEMAT_HD auto entries() const {
    std::array<Entry, static_cast<std::size_t>(nonZeroCount)> result{};
    constexpr auto inds = indices();
    // Guarded because the bound is a compile-time constant: when it is zero the
    // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
    if constexpr (nonZeroCount != 0) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(nonZeroCount); ++i) {
        result[i] =
            Entry{static_cast<Int>(inds[i] / cols), static_cast<Int>(inds[i] % cols), values[i]};
      }
    }
    return result;
  }

  /// Returns the sum of the diagonal elements (tr(A)).
  [[nodiscard]] SPARSEMAT_HD DataType trace() const { return SparseLinearAlgebra::trace(*this); }

  /**
   * @brief Tests whether the sparsity pattern is symmetric about the diagonal.
   *
   * Checks only the index structure — values are ignored.  A matrix is
   * structurally symmetric if for every non-zero at (i, j) there is also a
   * non-zero at (j, i).
   *
   * @return @c true if the sparsity pattern is symmetric.
   */
  [[nodiscard]] SPARSEMAT_HD bool is_structurally_symmetric() const {
    return SparseLinearAlgebra::is_structurally_symmetric(*this);
  }

  /**
   * @brief Tests whether the stored non-zero values are symmetric.
   *
   * Only compares positions that are non-zero in both (i, j) and (j, i).
   * Positions that are structurally zero on one side are not checked, so this
   * can return @c true even when @c is_structurally_symmetric() returns @c false.
   *
   * @param TOLERANCE Maximum absolute difference allowed between a(i,j) and a(j,i).
   * @return @c true if all paired non-zero values are within @p TOLERANCE.
   */
  [[nodiscard]] SPARSEMAT_HD bool is_sparse_symmetric(
      typename SparseMat::DataType TOLERANCE = 1e-6) const {
    return SparseLinearAlgebra::is_sparse_symmetric(*this, TOLERANCE);
  }

  /**
   * @brief Tests whether the matrix is fully symmetric (A == Aᵀ).
   *
   * Expands both sides to their full dense representation and compares every
   * element, including structural zeros.  Use this when you need to verify
   * that structurally zero positions on one side are truly zero on the other.
   *
   * @param TOLERANCE Maximum absolute difference allowed between a(i,j) and a(j,i).
   * @return @c true if the matrix equals its transpose within @p TOLERANCE.
   */
  [[nodiscard]] SPARSEMAT_HD bool is_full_symmetric(
      typename SparseMat::DataType TOLERANCE = 1e-6) const {
    return SparseLinearAlgebra::is_full_symmetric(*this, TOLERANCE);
  }

  /// Returns @c true if the sparsity pattern has no above-diagonal non-zeros.
  [[nodiscard]] SPARSEMAT_HD static constexpr bool is_structurally_lower_triangular() {
    return SparseLinearAlgebra::is_structurally_lower_triangular(SparseMat{});
  }

  /// Returns @c true if the sparsity pattern has no below-diagonal non-zeros.
  [[nodiscard]] SPARSEMAT_HD static constexpr bool is_structurally_upper_triangular() {
    return SparseLinearAlgebra::is_structurally_upper_triangular(SparseMat{});
  }

  /**
   * @brief Returns @c true if every above-diagonal stored value is within
   *        @p tolerance of zero.
   * @param tolerance Maximum absolute value permitted above the diagonal.
   */
  [[nodiscard]] SPARSEMAT_HD bool is_numerically_lower_triangular(DataType tolerance = 1e-6) const {
    return SparseLinearAlgebra::is_numerically_lower_triangular(*this, tolerance);
  }

  /**
   * @brief Returns @c true if every below-diagonal stored value is within
   *        @p tolerance of zero.
   * @param tolerance Maximum absolute value permitted below the diagonal.
   */
  [[nodiscard]] SPARSEMAT_HD bool is_numerically_upper_triangular(DataType tolerance = 1e-6) const {
    return SparseLinearAlgebra::is_numerically_upper_triangular(*this, tolerance);
  }

  /**
   * @brief Compile-time element write at position (I, J).
   *
   * Triggers a static assertion if (I, J) is a structurally zero position
   * because the sparsity pattern is immutable.
   *
   * @tparam I     Row index.
   * @tparam J     Column index.
   * @param  value New value to store.
   */
  template<IntType I, IntType J>
  SPARSEMAT_HD void set(DataType value) {
    if constexpr (MatrixUtilities<SparseMat>::isNonZero(I, J)) {
      constexpr auto index = MatrixUtilities<SparseMat>::getSparseIndex(I, J);
      values[index] = value;
    } else {
      static_assert(MatrixUtilities<SparseMat>::isNonZero(I, J),
                    "Attempting to set a value at a zero index.");
    }
  }

  /**
   * @brief Runtime element write at position (i, j).
   *
   * Does nothing and returns @c false if (i, j) is a structurally zero
   * position; the sparsity pattern cannot be changed at runtime.
   *
   * @param i     Row index.
   * @param j     Column index.
   * @param value New value to store.
   * @return      @c true if the position is non-zero and the value was stored;
   *              @c false otherwise.
   */
  SPARSEMAT_HD bool set(Int i, Int j, DataType value) {
    if (i < 0 || j < 0 || i >= static_cast<Int>(rows) || j >= static_cast<Int>(cols)) {
      return false;
    }
    if (MatrixUtilities<SparseMat>::isNonZero(i, j)) {
      auto index = MatrixUtilities<SparseMat>::getSparseIndex(i, j);
      values[index] = value;
      return true;
    }
    return false;
  }

  /**
   * @brief Sets every stored (non-zero) element to @p value.
   *
   * Equivalent to calling @c std::array::fill on the underlying storage;
   * structurally zero positions remain zero.
   *
   * @param value Value to broadcast across all non-zero slots.
   */
  SPARSEMAT_HD void fill(DataType value) { values.fill(value); }

  /**
   * @brief Sets every stored diagonal element to @p value.
   *
   * Structurally zero diagonal positions are unaffected.
   *
   * @param value Scalar written to every stored diagonal entry.
   */
  SPARSEMAT_HD void set_diagonal(DataType value) {
    SparseLinearAlgebra::set_diagonal(*this, value);
  }

  /**
   * @brief Sets the stored diagonal elements from a value array.
   *
   * Only structurally non-zero diagonal positions are written; each consumes
   * one entry from @p values in row order.
   *
   * @param values Values to write into stored diagonal entries, in row order.
   */
  template<std::size_t N>
  SPARSEMAT_HD void set_diagonal(std::array<DataType, N> values) {
    static_assert(N == static_cast<std::size_t>(
                           SparseLinearAlgebra::MatrixUtilities<SparseMat>::diagonal_nonzeros()),
                  "Size of values must match the number of stored diagonal entries.");
    SparseLinearAlgebra::set_diagonal(*this, values);
  }

  // --- Operations ---

  /**
   * @brief Matrix multiplication: @c *this × @p m.
   *
   * Result sparsity is determined at compile time.
   *
   * @tparam Matrix Right-hand sparse matrix type.
   * @param  m      Right-hand matrix; its row count must equal @c cols.
   * @return        Product matrix.
   */
  template<typename Matrix>
  [[nodiscard]] SPARSEMAT_HD auto mult(const Matrix& m) const {
    return SparseLinearAlgebra::multiply(*this, m);
  }

  /**
   * @brief Solves the linear system @c *this * x = b.
   *
   * Dispatches at compile time based on the sparsity pattern:
   * - Lower triangular → forward substitution via @c forward_solve.
   * - Upper triangular → back substitution via @c backward_solve.
   * - Neither → LU factorization via @c lu_solve.
   *
   * A diagonal matrix satisfies both conditions; forward substitution is used.
   *
   * @warning The LU path does **no pivoting**. It is only valid for matrices
   * that factorize stably without row swaps (diagonally dominant, or
   * otherwise pivot-free). A matrix that merely *needs* a row swap is
   * reported as singular via @c ok() rather than silently solved, but a
   * matrix that is pivot-stable yet badly conditioned will still return a
   * poor answer with @c ok() == @c true. Check @c ok() on the result, and
   * prefer @c cholesky() when the matrix is symmetric positive definite.
   *
   * @tparam Matrix Right-hand side column vector type.
   * @param  b      Right-hand side vector.
   * @return        Solution vector x.
   */
  template<typename Matrix>
  SPARSEMAT_HD auto solve(const Matrix& b) const {
    if constexpr (is_structurally_lower_triangular()) {
      return SparseLinearAlgebra::forward_solve(*this, b);
    } else if constexpr (is_structurally_upper_triangular()) {
      return SparseLinearAlgebra::backward_solve(*this, b);
    } else {
      return SparseLinearAlgebra::lu_solve(*this, b);
    }
  }

  /**
   * @brief Element-wise addition: @c *this + @p other.
   *
   * Result sparsity is the union of both sparsity patterns.
   *
   * @tparam Matrix Addend matrix type; must have the same shape as @c This.
   * @param  other  Right-hand operand.
   * @return        Sum matrix.
   */
  template<typename Matrix>
  [[nodiscard]] SPARSEMAT_HD auto add(const Matrix& other) const {
    return SparseLinearAlgebra::add(*this, other);
  }

  /**
   * @brief Element-wise subtraction: @c *this - @p other.
   *
   * Result sparsity is the union of both sparsity patterns.
   *
   * @tparam Matrix Subtrahend matrix type; must have the same shape as @c This.
   * @param  other  Right-hand operand.
   * @return        Difference matrix.
   */
  template<typename Matrix>
  [[nodiscard]] SPARSEMAT_HD auto subtract(const Matrix& other) const {
    return SparseLinearAlgebra::subtract(*this, other);
  }

  /**
   * @brief Vector dot product.
   *
   * Only works with a row vector multiplied by a column vector. Implementation calls
   * @c multiply and retrieves the (0,0) element of the resulting 1x1 matrix.
   *
   * @tparam A     Type of the other vector.
   * @param  b     The other vector.
   * @return       Scalar dot product value.
   */
  template<typename Matrix>
  SPARSEMAT_HD auto dot(const Matrix& b) const {
    static_assert(rows == 1 && Matrix::cols == 1 && cols == Matrix::rows,
                  "Dot product requires 1xN times Nx1 vectors with matching length.");
    return mult(b).template get<0, 0>();
  }

  /**
   * @brief Fused multiply-add: @c alpha * (*this) * @p x + @p beta * @p y,
   *        in a single pass.
   *
   * Computes @c alpha*A*x + beta*y without materializing the intermediate
   * product @c A*x. @p x and @p y must be column vectors, with @p x's
   * length matching @c cols and @p y's length matching @c rows. @p alpha
   * and @p beta default to @c 1, so @c axpy(x, y) alone computes plain
   * @c A*x + y.
   *
   * @tparam VecX Column-vector type multiplied by @c *this.
   * @tparam VecY Column-vector type added to the product.
   * @param  x     Vector multiplied by @c *this.
   * @param  y     Vector added to the product.
   * @param  alpha Scalar multiplier for @c (*this)*x.
   * @param  beta  Scalar multiplier for @p y.
   * @return       Result column vector @c alpha*(*this)*x + beta*y.
   */
  template<typename VecX, typename VecY>
  [[nodiscard]] SPARSEMAT_HD auto axpy(const VecX& x,
                                       const VecY& y,
                                       DataType alpha = DataType(1),
                                       DataType beta = DataType(1)) const {
    return SparseLinearAlgebra::axpy(*this, x, y, alpha, beta);
  }

  /**
   * @brief Element-wise (Hadamard) product: @c *this ⊙ @p a.
   *
   * Result sparsity is the intersection of both sparsity patterns.
   *
   * @tparam A   Right-hand matrix type; must have the same shape as @c This.
   * @param  a   Right-hand operand.
   * @return     Element-wise product matrix.
   */
  template<typename A>
  SPARSEMAT_HD auto hadamard(const A& a) const {
    return SparseLinearAlgebra::hadamard(*this, a);
  }

  /**
   * @brief Kronecker (tensor) product: @c *this ⊗ @p b.
   *
   * Produces an (rows*b.rows) × (cols*b.cols) matrix where each non-zero
   * element of @c *this is replaced by a scaled copy of @p b.  Result
   * sparsity is the outer product of both sparsity patterns, computed at
   * compile time.
   *
   * @tparam B Right-hand matrix type.
   * @param  b Right-hand operand.
   * @return   Kronecker product matrix.
   */
  template<typename B>
  SPARSEMAT_HD auto kronecker(const B& b) const {
    return SparseLinearAlgebra::kronecker(*this, b);
  }

  /**
   * @brief Returns the transpose of this matrix.
   *
   * Produces a @c SparseMat<DType, IntType, Cols, Rows, ...> with remapped indices.
   *
   * @return Transposed matrix.
   */
  [[nodiscard]] SPARSEMAT_HD auto transpose() const {
    return SparseLinearAlgebra::transpose(*this);
  }

  /**
   * @brief Returns the Gram matrix AᵀA (@c Cols × @c Cols).
   *
   * Equivalent to @c transpose().multiply(*this), but the transpose is never
   * materialized: element (i, j) is read directly as the inner product of
   * columns @c i and @c j.
   *
   * @return AᵀA, symmetric by construction.
   */
  [[nodiscard]] SPARSEMAT_HD auto ata() const { return SparseLinearAlgebra::ata(*this); }

  /**
   * @brief Returns the Gram matrix AAᵀ (@c Rows × @c Rows).
   *
   * The row-wise counterpart of @c ata(); likewise skips the transpose.
   *
   * @return AAᵀ, symmetric by construction.
   */
  [[nodiscard]] SPARSEMAT_HD auto aat() const { return SparseLinearAlgebra::aat(*this); }

  /**
   * @brief Returns the congruence transform AᵀBA (@c Cols × @c Cols).
   *
   * @p b must be square with @c B::rows == @c B::cols == @c Rows. Neither Aᵀ
   * nor the intermediate BA is materialized. The result is symmetric only when
   * @p b is.
   *
   * @tparam B Inner matrix type.
   * @param  b Inner (weight) matrix.
   * @return   AᵀBA.
   */
  template<typename B>
  [[nodiscard]] SPARSEMAT_HD auto atba(const B& b) const {
    return SparseLinearAlgebra::atba(*this, b);
  }

  /**
   * @brief Returns the congruence transform ABAᵀ (@c Rows × @c Rows).
   *
   * The other orientation of @c atba(), and the covariance propagation
   * @c F P Fᵀ of a Kalman filter. @p b must be @c Cols × @c Cols.
   *
   * @tparam B Inner matrix type.
   * @param  b Inner matrix.
   * @return   ABAᵀ.
   */
  template<typename B>
  [[nodiscard]] SPARSEMAT_HD auto abat(const B& b) const {
    return SparseLinearAlgebra::abat(*this, b);
  }

  /**
   * @brief Returns @c ABAᵀ + C in a single pass — the fused covariance
   *        propagation @c F P Fᵀ + Q.
   *
   * @tparam B Inner matrix type (@c Cols × @c Cols).
   * @tparam C Addend type (@c Rows × @c Rows).
   * @param  b Inner matrix.
   * @param  c Matrix added to the product.
   * @return   ABAᵀ + C.
   */
  template<typename B, typename C>
  [[nodiscard]] SPARSEMAT_HD auto abat_add(const B& b, const C& c) const {
    return SparseLinearAlgebra::abat_add(*this, b, c);
  }

  /**
   * @brief Returns @c AᵀBA + C in a single pass.
   *
   * @tparam B Inner matrix type (@c Rows × @c Rows).
   * @tparam C Addend type (@c Cols × @c Cols).
   * @param  b Inner matrix.
   * @param  c Matrix added to the product.
   * @return   AᵀBA + C.
   */
  template<typename B, typename C>
  [[nodiscard]] SPARSEMAT_HD auto atba_add(const B& b, const C& c) const {
    return SparseLinearAlgebra::atba_add(*this, b, c);
  }

  /**
   * @brief Returns AᵀB without materializing Aᵀ. @p b must have @c Rows rows.
   *
   * @tparam B Right matrix type.
   * @param  b Right matrix.
   * @return   AᵀB.
   */
  template<typename B>
  [[nodiscard]] SPARSEMAT_HD auto atb(const B& b) const {
    return SparseLinearAlgebra::atb(*this, b);
  }

  /**
   * @brief Returns ABᵀ without materializing Bᵀ. @p b must have @c Cols columns.
   *
   * @tparam B Right matrix type.
   * @param  b Right matrix.
   * @return   ABᵀ.
   */
  template<typename B>
  [[nodiscard]] SPARSEMAT_HD auto abt(const B& b) const {
    return SparseLinearAlgebra::abt(*this, b);
  }

  /**
   * @brief Extracts the @p NRows × @p NCols block whose top-left corner is
   *        (@p Row0, @p Col0).
   *
   * @tparam Row0  First row of the window.
   * @tparam Col0  First column of the window.
   * @tparam NRows Window height.
   * @tparam NCols Window width.
   * @return       The extracted block.
   */
  template<auto Row0, auto Col0, auto NRows, auto NCols>
  [[nodiscard]] SPARSEMAT_HD auto submatrix() const {
    return SparseLinearAlgebra::submatrix<Row0, Col0, NRows, NCols>(*this);
  }

  /**
   * @brief Extracts row @p I as a 1 × @c Cols matrix.
   * @tparam I Row index.
   */
  template<auto I>
  [[nodiscard]] SPARSEMAT_HD auto row() const {
    return SparseLinearAlgebra::row<I>(*this);
  }

  /**
   * @brief Extracts column @p J as a @c Rows × 1 matrix.
   * @tparam J Column index.
   */
  template<auto J>
  [[nodiscard]] SPARSEMAT_HD auto col() const {
    return SparseLinearAlgebra::col<J>(*this);
  }

  /**
   * @brief Joins @p b to the right: @c [A B]. Row counts must match.
   *
   * @tparam B Right operand type.
   * @param  b Right operand.
   * @return   @c Rows × (@c Cols + @c B::cols) matrix.
   */
  template<typename B>
  [[nodiscard]] SPARSEMAT_HD auto hcat(const B& b) const {
    return SparseLinearAlgebra::hcat(*this, b);
  }

  /**
   * @brief Stacks @p b underneath: @c [A; B]. Column counts must match.
   *
   * @tparam B Bottom operand type.
   * @param  b Bottom operand.
   * @return   (@c Rows + @c B::rows) × @c Cols matrix.
   */
  template<typename B>
  [[nodiscard]] SPARSEMAT_HD auto vcat(const B& b) const {
    return SparseLinearAlgebra::vcat(*this, b);
  }

  /**
   * @brief Permutes rows and columns: @c result[i,j] == @c a[RowPerm[i], ColPerm[j]].
   *
   * @tparam RowPerm Array of source row indices.
   * @tparam ColPerm Array of source column indices.
   * @return         Permuted matrix.
   */
  template<auto RowPerm, auto ColPerm>
  [[nodiscard]] SPARSEMAT_HD auto permute() const {
    return SparseLinearAlgebra::permute<RowPerm, ColPerm>(*this);
  }

  /**
   * @brief Applies the same permutation to rows and columns: @c P A Pᵀ.
   *
   * The form a fill-reducing ordering takes for a square system; pair it with
   * @c rcm_ordering<T>() to shrink the bandwidth before factorizing.
   *
   * @tparam Perm Permutation array.
   * @return      @c P A Pᵀ.
   */
  template<auto Perm>
  [[nodiscard]] SPARSEMAT_HD auto symmetric_permute() const {
    return SparseLinearAlgebra::symmetric_permute<Perm>(*this);
  }

  /**
   * @brief Returns @c A + alpha·x yᵀ without materializing the outer product.
   *
   * @tparam X     Left vector type (@c Rows × 1).
   * @tparam Y     Right vector type (@c Cols × 1).
   * @param  x     Left vector.
   * @param  y     Right vector.
   * @param  alpha Scalar multiplier (default 1; pass -1 to downdate).
   * @return       @c A + alpha·x yᵀ, whose pattern may be wider than @c A's.
   */
  template<typename X, typename Y>
  [[nodiscard]] SPARSEMAT_HD auto rank1_update(const X& x,
                                               const Y& y,
                                               DataType alpha = DataType(1)) const {
    return SparseLinearAlgebra::rank1_update(*this, x, y, alpha);
  }

  /**
   * @brief Returns @c A + alpha·x xᵀ — the symmetry-preserving special case.
   *
   * @tparam X     Vector type (@c Rows × 1).
   * @param  x     Update vector.
   * @param  alpha Scalar multiplier (default 1).
   * @return       @c A + alpha·x xᵀ.
   */
  template<typename X>
  [[nodiscard]] SPARSEMAT_HD auto symmetric_rank1_update(const X& x,
                                                         DataType alpha = DataType(1)) const {
    return SparseLinearAlgebra::symmetric_rank1_update(*this, x, alpha);
  }

  /**
   * @brief Returns a scaled copy: every non-zero element multiplied by @p factor.
   *
   * Sparsity pattern is unchanged.
   *
   * @param factor Scalar multiplier.
   * @return       Scaled matrix.
   */
  [[nodiscard]] SPARSEMAT_HD auto scale(DataType factor) const {
    return SparseLinearAlgebra::scale(*this, factor);
  }

  /**
   * @brief Multiplies every non-zero element by @p factor in place.
   * @param factor Scalar multiplier.
   */
  SPARSEMAT_HD auto& scale_inplace(DataType factor) {
    SparseLinearAlgebra::scale_inplace(*this, factor);
    return *this;
  }

  /**
   * @brief Returns a copy with @p factor added to every non-zero element.
   *
   * Sparsity pattern is unchanged; structurally zero elements are not affected.
   *
   * @param factor Scalar to add to each stored value.
   * @return       Shifted matrix.
   */
  [[nodiscard]] SPARSEMAT_HD auto shift(DataType factor) const {
    return SparseLinearAlgebra::shift(*this, factor);
  }

  /**
   * @brief Adds @p factor to every non-zero element in place.
   * @param factor Scalar to add.
   */
  SPARSEMAT_HD auto& shift_inplace(DataType factor) {
    SparseLinearAlgebra::shift_inplace(*this, factor);
    return *this;
  }

  /**
   * @brief Returns a unit-norm copy of this matrix (divided by its Frobenius norm).
   * @return Normalized matrix.
   */
  [[nodiscard]] SPARSEMAT_HD auto normalize() const {
    return SparseLinearAlgebra::normalize(*this);
  }

  /**
   * @brief Divides every non-zero element by the Frobenius norm in place.
   */
  SPARSEMAT_HD auto& normalize_inplace() {
    SparseLinearAlgebra::normalize_inplace(*this);
    return *this;
  }

  /**
   * @brief Computes the Cholesky factorization and returns a solve handle.
   *
   * Factorizes this matrix as L * L^T (the matrix must be symmetric positive
   * definite) and returns a @c CholeskyFactor handle.  Call @c .solve(b) on
   * the handle to solve one or more right-hand sides without re-factorizing.
   *
   * @return A @c Result wrapping a @c CholeskyFactor<L> handle; @c ok() is
   *         @c false if a diagonal pivot was zero or negative, meaning this
   *         matrix is not (numerically) symmetric positive definite.
   */
  [[nodiscard]] SPARSEMAT_HD auto cholesky() const
    requires(rows == cols)
  {
    using LType = decltype(SparseLinearAlgebra::detail::LCholeskyMatrix<SparseMat>::make_result());
    auto l = SparseLinearAlgebra::cholesky_factorize(*this);
    if (!l.ok()) {
      return SparseLinearAlgebra::Result<CholeskyFactor<LType>>(CholeskyFactor<LType>(LType{}),
                                                                l.status());
    }
    return SparseLinearAlgebra::Result<CholeskyFactor<LType>>(CholeskyFactor<LType>(
                                                                  std::move(l.value())),
                                                              l.status());
  }

  /**
   * @brief Computes the determinant.
   *
   * Uses the diagonal directly for a structurally triangular matrix, and the
   * LU factorization otherwise (no pivoting, as everywhere else). Check
   * @c ok() before trusting the value — see @c SparseLinearAlgebra::determinant.
   */
  [[nodiscard]] SPARSEMAT_HD auto determinant() const
    requires(rows == cols)
  {
    return SparseLinearAlgebra::determinant(*this);
  }

  /**
   * @brief Returns @c A⁻¹, by solving @c A*X = I.
   *
   * @note The inverse of a sparse matrix is generally dense, so the result
   * type carries many more stored values than this one. Prefer @c solve() when
   * you only need @c A⁻¹b for specific right-hand sides — it is faster and
   * more accurate. Use @c cholesky_inverse() when the matrix is SPD.
   */
  [[nodiscard]] SPARSEMAT_HD auto inverse() const
    requires(rows == cols)
  {
    return SparseLinearAlgebra::inverse(*this);
  }

  /**
   * @brief Solves @c *this * x = @p b in the least-squares sense, for a
   *        non-square matrix.
   *
   * @c solve() requires a square matrix; this handles the rectangular cases —
   * minimising @c ||Ax-b||₂ when overdetermined, and returning the
   * minimum-norm solution when underdetermined. It works via the normal
   * equations, which squares the condition number; see
   * @c SparseLinearAlgebra::least_squares_solve for when that matters.
   */
  template<typename Matrix>
  [[nodiscard]] SPARSEMAT_HD auto least_squares_solve(const Matrix& b) const {
    return SparseLinearAlgebra::least_squares_solve(*this, b);
  }

  /**
   * @brief Composes block-diagonally with @p b: @c diag(*this, b).
   *
   * Result is (rows + b.rows) x (cols + b.cols), with this matrix top-left and
   * @p b bottom-right. Stored-value count is exactly the sum of the two.
   */
  template<typename Matrix>
  [[nodiscard]] SPARSEMAT_HD auto block_diagonal(const Matrix& b) const {
    return SparseLinearAlgebra::block_diagonal(*this, b);
  }

  /**
   * @brief Computes the Frobenius norm: √(Σ aᵢⱼ²) over all non-zero elements.
   * @return Frobenius norm as the matrix's @c DataType.
   */
  [[nodiscard]] SPARSEMAT_HD auto frobenius() const {
    return SparseLinearAlgebra::frobenius(*this);
  }

  /**
   * @brief Largest absolute stored value (the max-norm).
   * @return @c max|aᵢⱼ|.
   */
  [[nodiscard]] SPARSEMAT_HD auto max_abs() const { return SparseLinearAlgebra::max_abs(*this); }

  /**
   * @brief 1-norm: the largest absolute column sum.
   * @return @c max_j Σᵢ|aᵢⱼ|.
   */
  [[nodiscard]] SPARSEMAT_HD auto norm_1() const { return SparseLinearAlgebra::norm_1(*this); }

  /**
   * @brief ∞-norm: the largest absolute row sum.
   * @return @c max_i Σⱼ|aᵢⱼ|.
   */
  [[nodiscard]] SPARSEMAT_HD auto norm_inf() const { return SparseLinearAlgebra::norm_inf(*this); }

  /**
   * @brief Expands the sparse matrix into a fully dense @c SparseMat.
   *
   * The result is another @c SparseMat whose sparsity pattern covers every
   * position, so structural zeros become explicitly stored @c DataType(0)
   * values. It is *not* a bare array: keeping it a @c SparseMat is what lets
   * a densified result feed straight back into any other operation (see the
   * Kalman example, which chains @c .add(Q).dense() across filter steps).
   *
   * Use @c to_array() when a plain row-major buffer is what's wanted.
   *
   * @return A @c SparseMat<DataType, Int, Rows, Cols, 0, 1, ..., Rows*Cols-1>.
   */
  [[nodiscard]] SPARSEMAT_HD auto dense() const { return SparseLinearAlgebra::dense(*this); }

  /**
   * @brief Expands the sparse matrix into a plain dense row-major array.
   *
   * Zero positions are explicitly written as @c DataType(0).
   *
   * @return @c std::array<DataType, Rows*Cols> in row-major order.
   */
  [[nodiscard]] SPARSEMAT_HD auto to_array() const {
    std::array<DataType, static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)> result{};
    constexpr auto inds = indices();
    // Guarded because the bound is a compile-time constant: when it is zero the
    // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
    if constexpr (nonZeroCount != 0) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(nonZeroCount); ++i) {
        result[static_cast<std::size_t>(inds[i])] = values[i];
      }
    }
    return result;
  }

  /**
   * @brief Returns a copy of this matrix with its scalar type changed to @p D.
   *
   * Binary operations reject mixed scalar types rather than silently
   * promoting or truncating one side; this is the explicit opt-in. The
   * sparsity pattern, dimensions, and index type are all preserved.
   *
   * @code
   * auto as_double = float_matrix.template convert<double>();
   * @endcode
   *
   * @tparam D Target scalar element type.
   */
  template<typename D>
  [[nodiscard]] SPARSEMAT_HD auto convert() const {
    return SparseLinearAlgebra::convert<D>(*this);
  }

  // --- Operator overloads ---

  /** @brief Matrix multiplication: @c *this × @p rhs. */
  template<typename Matrix>
  SPARSEMAT_HD auto operator*(const Matrix& rhs) const {
    return mult(rhs);
  }

  /** @brief Element-wise addition: @c *this + @p rhs. */
  template<typename Matrix>
  SPARSEMAT_HD auto operator+(const Matrix& rhs) const {
    return add(rhs);
  }

  /** @brief Element-wise subtraction: @c *this - @p rhs. */
  template<typename Matrix>
  SPARSEMAT_HD auto operator-(const Matrix& rhs) const {
    return subtract(rhs);
  }

  /** @brief Unary negation: returns a copy with every stored value negated. */
  [[nodiscard]] SPARSEMAT_HD auto operator-() const { return scale(static_cast<DataType>(-1)); }

  /** @brief Scalar multiply (returns new matrix): @c *this * @p factor. */
  SPARSEMAT_HD auto operator*(DataType factor) const { return scale(factor); }

  /** @brief Scalar divide (returns new matrix): @c *this / @p divisor. */
  [[nodiscard]] SPARSEMAT_HD auto operator/(DataType divisor) const {
    return scale(static_cast<DataType>(1) / divisor);
  }

  /** @brief Scalar multiply in place: @c *this *= @p factor. */
  SPARSEMAT_HD SparseMat& operator*=(DataType factor) {
    scale_inplace(factor);
    return *this;
  }

  /** @brief Scalar divide in place: @c *this /= @p divisor. */
  SPARSEMAT_HD SparseMat& operator/=(DataType divisor) {
    scale_inplace(static_cast<DataType>(1) / divisor);
    return *this;
  }

  /**
   * @brief Element-wise addition in place: @c *this += @p rhs.
   *
   * Unlike @c operator+, this cannot widen the sparsity pattern — the pattern
   * is part of the type and @c *this has to keep its own. @p rhs's non-zeros
   * must therefore be a subset of this matrix's, which is checked at compile
   * time. Use @c a = a + b when the union pattern is what's wanted.
   */
  template<typename Matrix>
  SPARSEMAT_HD SparseMat& operator+=(const Matrix& rhs) {
    auto result = add(rhs);
    static_assert(std::is_same_v<decltype(result), SparseMat>,
                  "operator+= cannot change the sparsity pattern: the right-hand operand has "
                  "non-zeros this matrix does not. Use 'a = a + b' to get the union pattern.");
    values = result.values;
    return *this;
  }

  /**
   * @brief Element-wise subtraction in place: @c *this -= @p rhs.
   *
   * Same subset requirement as @c operator+=.
   */
  template<typename Matrix>
  SPARSEMAT_HD SparseMat& operator-=(const Matrix& rhs) {
    auto result = subtract(rhs);
    static_assert(std::is_same_v<decltype(result), SparseMat>,
                  "operator-= cannot change the sparsity pattern: the right-hand operand has "
                  "non-zeros this matrix does not. Use 'a = a - b' to get the union pattern.");
    values = result.values;
    return *this;
  }

  // --- Utility ---

  /**
   * @brief Prints each non-zero index and its value to @c std::cout.
   *
   * Iterates at compile time over the @c nonZeroCount stored elements.
   *
   * Intentionally host-only (not @c SPARSEMAT_HD): @c std::cout isn't usable
   * from device code, and CUDA's device-side @c printf has a different
   * signature, so there's no portable way to make this callable from a
   * kernel.
   *
   * Iterates the packed storage directly rather than recursing over it, so
   * this no longer costs one template instantiation per stored value.
   */
  void print() const {
    constexpr auto inds = indices();
    // Guarded because the bound is a compile-time constant: when it is zero the
    // comparison is `unsigned < 0`, which nvcc reports as a pointless comparison.
    if constexpr (nonZeroCount != 0) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(nonZeroCount); ++i) {
        std::cout << "Value at index " << inds[i] << ": " << values[i] << '\n';
      }
    }
  }

  /**
   * @brief Prints the full Rows×Cols matrix to @c std::cout, including zeros.
   *
   * Each row is printed on its own line with space-separated values.
   */
  void printDense() const {
    const auto d = to_array();
    for (IntType i = 0; i < rows; ++i) {
      for (IntType j = 0; j < cols; ++j) {
        std::cout << d[static_cast<std::size_t>((i * cols) + j)] << " ";
      }
      std::cout << '\n';
    }
  }
};

/**
 * @brief Scalar multiply with the scalar on the left: @p factor * @p m.
 *
 * The member @c operator* only covers `matrix * scalar`; this free function
 * completes the pair so both orderings work.
 *
 * @param factor Scalar multiplier.
 * @param m      Matrix to scale.
 */
template<SparseMatrixType Matrix>
[[nodiscard]] SPARSEMAT_HD auto operator*(typename Matrix::DataType factor, const Matrix& m) {
  return m.scale(factor);
}

/**
 * @brief Aliases for commonly used sparse matrix types with specific data and index types.
 *
 * These aliases simplify the creation of sparse matrices with double or float
 * data types and 32-bit or 64-bit integer index types.
 */
/// @c double-valued sparse matrix indexed with @c int32_t.
template<int Rows, int Cols, int32_t... NonZeros>
using SparseMatrix_64_32 = SparseMat<double, int32_t, Rows, Cols, NonZeros...>;

/// @c double-valued sparse matrix indexed with @c int64_t.
template<int Rows, int Cols, int64_t... NonZeros>
using SparseMatrix_64_64 = SparseMat<double, int64_t, Rows, Cols, NonZeros...>;

/// @c float-valued sparse matrix indexed with @c int32_t.
template<int Rows, int Cols, int32_t... NonZeros>
using SparseMatrix_32_32 = SparseMat<float, int32_t, Rows, Cols, NonZeros...>;

/// @c float-valued sparse matrix indexed with @c int64_t.
template<int Rows, int Cols, int64_t... NonZeros>
using SparseMatrix_32_64 = SparseMat<float, int64_t, Rows, Cols, NonZeros...>;

/**
 * @brief Creates an identity sparse matrix of the specified type and dimensions.
 *
 * @tparam DType   Data type of the matrix elements.
 * @tparam IntType Integer type for the matrix indices.
 * @tparam Rows    Number of rows in the matrix.
 * @tparam Cols    Number of columns in the matrix.
 * @return        Identity sparse matrix of type @c SparseMat<DType, IntType, Rows, Cols>.
 */
template<typename DType, typename IntType, int Rows, int Cols>
SPARSEMAT_HD auto identity() {
  return SparseMat<DType, IntType, Rows, Cols>::identity();
}

/**
 * @brief Creates a zero sparse matrix of the specified type and dimensions.
 *
 * @tparam DType   Data type of the matrix elements.
 * @tparam IntType Integer type for the matrix indices.
 * @tparam Rows    Number of rows in the matrix.
 * @tparam Cols    Number of columns in the matrix.
 * @return        Zero sparse matrix of type @c SparseMat<DType, IntType, Rows, Cols>.
 */
template<typename DType, typename IntType, int Rows, int Cols>
SPARSEMAT_HD auto zero() {
  return SparseMat<DType, IntType, Rows, Cols>();
}

/**
 * @brief Creates a dense representation of the sparse matrix of the specified type and dimensions.
 *
 * @tparam DType   Data type of the matrix elements.
 * @tparam IntType Integer type for the matrix indices.
 * @tparam Rows    Number of rows in the matrix.
 * @tparam Cols    Number of columns in the matrix.
 * @return        Dense representation of the sparse matrix of type @c SparseMat<DType, IntType,
 * Rows, Cols>.
 */
template<typename DType, typename IntType, int Rows, int Cols>
SPARSEMAT_HD auto dense() {
  return SparseMat<DType, IntType, Rows, Cols>().dense();
}



/**
 * @brief Builds a @c SparseMat from a compile-time sparsity pattern plus a
 *        dense initializer array.
 *
 * Given a type satisfying @c SparsityPatternType, derives the resulting
 * @c SparseMat's dimensions and flat non-zero index list at compile time, so
 * callers can populate a sparse matrix from an ordinary dense
 * array-of-arrays without hand-computing @c NonZeros indices themselves.
 *
 * @tparam SparsityPattern Type exposing a @c static @c constexpr @c sparsity()
 *                         function; see @c SparsityPatternType.
 * @tparam DType           Scalar element type of the built matrix.
 * @tparam IntType         Signed integer type used for indices/offsets.
 */
template<SparsityPatternType SparsityPattern, typename DType, typename IntType>
struct SparseMatBuilder {

    /// Number of rows in @c SparsityPattern::sparsity().
    SPARSEMAT_HD static constexpr IntType get_row_count() { return SparsityPattern::sparsity().size(); }

    /// Number of columns in @c SparsityPattern::sparsity() (rows are assumed uniform in length).
    SPARSEMAT_HD static constexpr IntType get_column_count() {
      return SparsityPattern::sparsity()[0].size();
    }

    /// Counts the non-zero (1) entries in @c SparsityPattern::sparsity().
    SPARSEMAT_HD static constexpr IntType get_nonzero_count() {
      auto s = SparsityPattern::sparsity();
      IntType count = 0;
      for (IntType i = 0; i < get_row_count(); ++i) {
          for (IntType j = 0; j < get_column_count(); ++j) {
              if (s[i][j]) {
                  ++count;
              }
          }
      }
      return count;
    }

    /**
     * @brief Flattens the non-zero positions of @c SparsityPattern::sparsity()
     *        into row-major flat indices.
     * @return Array of @c get_nonzero_count() flat indices, in row-major order.
     */
    SPARSEMAT_HD static constexpr auto flatten() {
      std::array<IntType, get_nonzero_count()> inds{};
      auto s = SparsityPattern::sparsity();
      IntType idx = 0;
      for (IntType i = 0; i < get_row_count(); ++i) {
          for (IntType j = 0; j < get_column_count(); ++j) {
              if (s[i][j]) {
                  inds[idx++] = i * get_column_count() + j;
              }
          }
      }
      return inds;
    }

    /**
     * @brief Rebuilds @c flatten()'s flat index array as the @c SparseMat
     *        @c NonZeros template pack, so its type can be captured in @c Matrix.
     */
    template<std::size_t... Is>
    static constexpr auto make_matrix_type(std::index_sequence<Is...> /*seq*/)
        -> SparseMat<DType,
                     IntType,
                     static_cast<int>(get_row_count()),
                     static_cast<int>(get_column_count()),
                     static_cast<IntType>(flatten()[Is])...>;

    /// The concrete @c SparseMat type produced by this builder.
    using Matrix = decltype(make_matrix_type(std::make_index_sequence<get_nonzero_count()>{}));

    /**
     * @brief Builds a @c Matrix from a dense array, keeping only the values at
     *        the pattern's non-zero positions.
     *
     * @param data Dense @c get_row_count() x @c get_column_count() array;
     *             values at structurally zero positions are discarded.
     * @return     @c Matrix populated with @p data's values at every non-zero position.
     */
    SPARSEMAT_HD auto build(std::array<std::array<DType, get_column_count()>, get_row_count()> data) {
        constexpr auto sparsity = flatten();

        std::array<DType, get_nonzero_count()> flat_data{};
        IntType idx = 0;
        for (IntType i = 0; i < get_row_count(); ++i) {
            for (IntType j = 0; j < get_column_count(); ++j) {
                if (idx < get_nonzero_count() && sparsity[idx] == i * get_column_count() + j) {
                    flat_data[idx++] = data[i][j];
                }
            }
        }
        return Matrix(flat_data);
    }

};

/**
 * @brief Builds a @c SparseMat from a compile-time sparsity pattern given as
 *        a coordinate (COO-style) list, plus a matching list of (row, col,
 *        value) triples.
 *
 * Unlike @c SparseMatBuilder — whose @c SparsityPattern spells out a full
 * dense 0/1 grid — @c SparseMatBuilderCSR's @c SparsityPattern lists only the
 * non-zero coordinates directly, which scales better as matrices grow
 * sparser. A conforming @c SparsityPattern type must expose:
 *  - @c static @c constexpr @c int @c rows and @c cols giving the matrix
 *    dimensions (unlike @c SparseMatBuilder, these are not inferred from
 *    @c sparsity()).
 *  - @c static @c constexpr @c sparsity() returning an array of
 *    @c rows*cols row/col pairs (any type exposing @c .first / @c .second,
 *    e.g. @c std::pair<int,int>) — one per non-zero position, in any order.
 *
 * @c build() likewise takes its values as a list of @c (row, col, value)
 * triples rather than a dense array-of-arrays; the triples may be supplied
 * in any order and are matched against the pattern by coordinate, not by
 * position — at the cost of an O(n^2) match against @c flatten() (see
 * @c build()).
 *
 * @tparam SparsityPattern Type exposing @c rows, @c cols, and @c sparsity()
 *                         as described above.
 * @tparam DType           Scalar element type of the built matrix. Defaults
 *                         to @c double.
 * @tparam IntType         Signed integer type used for indices/offsets.
 *                         Defaults to @c int32_t.
 */
template<typename SparsityPattern, typename DType = double, typename IntType = int32_t>
struct SparseMatBuilderCSR {

    /// Number of rows, taken from @c SparsityPattern::rows.
    SPARSEMAT_HD static constexpr IntType get_row_count() { return SparsityPattern::rows; }

    /// Number of columns, taken from @c SparsityPattern::cols.
    SPARSEMAT_HD static constexpr IntType get_column_count() {
      return SparsityPattern::cols;
    }

    /// Number of non-zero entries, taken from @c SparsityPattern::sparsity()'s length.
    SPARSEMAT_HD static constexpr IntType get_nonzero_count() {
      return SparsityPattern::sparsity().size();
    }

    /**
     * @brief Flattens @c SparsityPattern::sparsity()'s (row, col) coordinates
     *        into row-major flat indices.
     * @return Array of @c get_nonzero_count() flat indices, in the same order
     *         as @c SparsityPattern::sparsity().
     */
    SPARSEMAT_HD static constexpr auto flatten() {
      std::array<IntType, get_nonzero_count()> inds{};
      auto s = SparsityPattern::sparsity(); // Array of pairs (row,col) for nonzero positions
      IntType idx = 0;
      for (IntType i = 0; i < get_nonzero_count(); ++i) {
          inds[idx++] = s[i].first * get_column_count() + s[i].second;
      }
      return inds;
    }

    /**
     * @brief Rebuilds @c flatten()'s flat index array as the @c SparseMat
     *        @c NonZeros template pack, so its type can be captured in @c Matrix.
     */
    template<std::size_t... Is>
    static constexpr auto make_matrix_type(std::index_sequence<Is...> /*seq*/)
        -> SparseMat<DType,
                     IntType,
                     static_cast<int>(get_row_count()),
                     static_cast<int>(get_column_count()),
                     static_cast<IntType>(flatten()[Is])...>;

    /// The concrete @c SparseMat type produced by this builder.
    using Matrix = decltype(make_matrix_type(std::make_index_sequence<get_nonzero_count()>{}));

    /**
     * @brief Builds a @c Matrix from (row, col, value) triples.
     *
     * @param data @c get_nonzero_count() triples of @c (row, col, value),
     *             one per non-zero position, in any order — each is matched
     *             against @c flatten() by coordinate rather than by
     *             position. Every triple's @c (row, col) must appear in
     *             @c SparsityPattern::sparsity(); triples for coordinates
     *             outside the pattern are silently dropped.
     * @return     @c Matrix populated with @p data's values at every non-zero position.
     */
    SPARSEMAT_HD auto build(std::array<std::tuple<IntType, IntType, DType>, get_nonzero_count()> data) {
        constexpr auto sparsity = flatten();
        //This is not great. We are doing a nested loop to match the flat index, which is O(n^2).
        std::array<DType, get_nonzero_count()> flat_data{};
        for (IntType i = 0; i < get_nonzero_count(); ++i) {
           auto row = std::get<0>(data[i]);
           auto col = std::get<1>(data[i]);
           auto index = row * get_column_count() + col;
           for (IntType j = 0; j < get_nonzero_count(); ++j) {
               if (sparsity[j] == index) {
                   flat_data[j] = std::get<2>(data[i]);
                   break;
               }
           }
        }
        return Matrix(flat_data);
    }

};


}  // namespace SparseLinearAlgebra

#include "sparsemat/builders/patterns.h"
#include "sparsemat/builders/tuple_builder.h"
