#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <type_traits>

#include "sparsemat/operations/add.h"
#include "sparsemat/operations/cholesky.h"
#include "sparsemat/operations/dense.h"
#include "sparsemat/operations/diagonal.h"
#include "sparsemat/operations/hadamard.h"
#include "sparsemat/operations/kronecker.h"
#include "sparsemat/operations/lu.h"
#include "sparsemat/operations/multiply.h"
#include "sparsemat/operations/scale.h"
#include "sparsemat/operations/shift.h"
#include "sparsemat/operations/symmetric.h"
#include "sparsemat/operations/trace.h"
#include "sparsemat/operations/transpose.h"
#include "sparsemat/operations/triangular.h"
#include "sparsemat/operations/utils.h"

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
   * @brief Validates that the provided values array matches the sparsity pattern.
   *
   * Checks that indices provided for the sparsity pattern are within bounds and unique.
   *
   * @param vals Array of values to validate.
   * @return     @c true if the sparsity pattern is valid, @c false otherwise.
   */
  [[nodiscard]] SPARSEMAT_HD static constexpr bool validate_indices() {
    constexpr auto inds = indices();
    for (IntType i = 0; i < nonZeroCount; ++i) {
      if (inds[i] < 0) {
        return false;
      }
      if (inds[i] >= rows * cols) {
        return false;
      }
      for (IntType j = i + 1; j < nonZeroCount; ++j) {
        if (inds[i] == inds[j]) {
          return false;
        }
      }
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
   * The number of arguments must equal @c nonZeroCount; a static assertion
   * enforces this at compile time.
   *
   * @tparam Vals Deduced value types (must be convertible to @c DataType).
   * @param  vals Values in the same order as the @c NonZeros index pack.
   */
  template<typename... Vals>
  SPARSEMAT_HD SparseMat(Vals... vals) : values{static_cast<DataType>(vals)...} {
    static_assert(sizeof...(Vals) == nonZeroCount, "Number of values must match non-zero count.");
  }

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
   * - Neither → LU factorization via @c lu_solve (no pivoting; requires a non-singular,
   * pivot-stable matrix).
   *
   * A diagonal matrix satisfies both conditions; forward substitution is used.
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
  SPARSEMAT_HD auto add(const Matrix& other) const {
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
   * Produces a @c SparseMat<DType, Cols, Rows, ...> with remapped indices.
   *
   * @return Transposed matrix.
   */
  [[nodiscard]] SPARSEMAT_HD auto transpose() const {
    return SparseLinearAlgebra::transpose(*this);
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
   * @brief Computes the Frobenius norm: √(Σ aᵢⱼ²) over all non-zero elements.
   * @return Frobenius norm as the matrix's @c DataType.
   */
  [[nodiscard]] SPARSEMAT_HD auto frobenius() const {
    return SparseLinearAlgebra::frobenius(*this);
  }

  /**
   * @brief Expands the sparse matrix into a fully dense row-major array.
   *
   * Zero positions are explicitly written as @c DataType(0).
   *
   * @return @c std::array<DataType, Rows*Cols> in row-major order.
   */
  [[nodiscard]] SPARSEMAT_HD auto dense() const { return SparseLinearAlgebra::dense(*this); }

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

  /** @brief Scalar multiply (returns new matrix): @c *this * @p factor. */
  SPARSEMAT_HD auto operator*(DataType factor) const { return scale(factor); }

  /** @brief Scalar multiply in place: @c *this *= @p factor. */
  SPARSEMAT_HD SparseMat& operator*=(DataType factor) {
    scale_inplace(factor);
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
   * @tparam i Current compile-time index into @c values (default 0).
   */
  template<IntType i = 0>
  void print() const {
    if constexpr (i < nonZeroCount) {
      std::cout << "Value at index " << indices()[i] << ": " << values[i] << '\n';
      print<i + 1>();
    }
  }

  /**
   * @brief Prints the full Rows×Cols matrix to @c std::cout, including zeros.
   *
   * Each row is printed on its own line with space-separated values.
   */
  void printDense() const {
    auto d = dense();
    for (IntType i = 0; i < rows; ++i) {
      for (IntType j = 0; j < cols; ++j) {
        std::cout << d[(i * cols) + j] << " ";
      }
      std::cout << '\n';
    }
  }
};

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
