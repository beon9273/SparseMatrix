#pragma once

#include <array>
#include <iostream>

#include "operations/add.h"
#include "operations/cholesky.h"
#include "operations/dense.h"
#include "operations/diagonal.h"
#include "operations/hadamard.h"
#include "operations/invert.h"
#include "operations/kronecker.h"
#include "operations/lu.h"
#include "operations/multiply.h"
#include "operations/scale.h"
#include "operations/shift.h"
#include "operations/symmetric.h"
#include "operations/trace.h"
#include "operations/transpose.h"
#include "operations/triangular.h"
#include "operations/utils.h"

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
 * @tparam Rows     Number of rows.
 * @tparam Cols     Number of columns.
 * @tparam NonZeros Flat (row-major) indices of every non-zero element.
 */
template<typename DType, std::size_t Rows, std::size_t Cols, std::size_t... NonZeros>
class SparseMat {
  /// Helper that builds an N×N identity matrix from an index sequence.
  template<std::size_t N, std::size_t... Is>
  static auto make_identity_impl(std::index_sequence<Is...> /*seq*/) {
    SparseMat<DType, N, N, (Is * (N + 1))...> result;
    result.values.fill(1);
    return result;
  }

  /**
   * @brief Rebinds this template to a different shape and non-zero set.
   * @tparam R  New row count.
   * @tparam C  New column count.
   * @tparam NZ New non-zero flat indices.
   */
  template<std::size_t R, std::size_t C, std::size_t... NZ>
  using Rebind = SparseMat<DType, R, C, NZ...>;

 public:
  /// Scalar element type.
  using DataType = DType;
  static constexpr std::size_t rows = Rows;  ///< Number of rows.
  static constexpr std::size_t cols = Cols;  ///< Number of columns.
  static constexpr std::size_t nonZeroCount =
      sizeof...(NonZeros);  ///< Number of stored (non-zero) elements.
  static constexpr std::array<std::size_t, sizeof...(NonZeros)> indices{
      NonZeros...};                             ///< indices of nonzeros (flat)
  std::array<DataType, nonZeroCount> values{};  // Values of nonzeros (runtime)

  /**
   * @brief Unpacks a compile-time index array into template parameters to
   *        produce a @c Rebind matrix with the correct sparsity.
   *
   * Intended for internal use by operation helpers that compute a result
   * sparsity as a @c std::array and need to turn it back into a @c SparseMat
   * type.
   *
   * @tparam NRows Target row count.
   * @tparam NCols Target column count.
   * @tparam Arr   Compile-time array whose elements become the @c NonZeros pack.
   * @tparam Is    Index sequence over @p Arr.
   * @return       Default-constructed @c Rebind<NRows,NCols,Arr[Is]...>.
   */
  template<std::size_t NRows, std::size_t NCols, auto Arr, std::size_t... Is>
  static auto make(std::index_sequence<Is...> /*seq*/) {
    return Rebind<NRows, NCols, Arr[Is]...>{};
  }

  // --- Constructors ---

  /// Default-constructs all non-zero values to zero.
  SparseMat() = default;

  /**
   * @brief Constructs from a pre-built values array.
   * @param vals Array of @c nonZeroCount values in index order.
   */
  SparseMat(std::array<DataType, nonZeroCount> vals) : values(std::move(vals)) {}

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
  SparseMat(Vals... vals) : values{static_cast<DataType>(vals)...} {
    static_assert(sizeof...(Vals) == nonZeroCount, "Number of values must match non-zero count.");
  }

  // --- Static factories ---

  /**
   * @brief Creates an N×N identity matrix at compile time.
   *
   * The returned type has non-zeros only on the main diagonal.
   *
   * @tparam N Side length of the square identity matrix.
   * @return   Identity matrix with @c DataType values of 1.
   */
  template<std::size_t N>
  static auto identity() {
    return make_identity_impl<N>(std::make_index_sequence<N>{});
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
  template<std::size_t I, std::size_t J>
  [[nodiscard]] DataType get() const {
    if constexpr (SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(I, J)) {
      constexpr int index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
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
  [[nodiscard]] DataType get(int i, int j) const {
    if (SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(i, j)) {
      int index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(i, j);
      return values[index];
    }
    return static_cast<DataType>(0);
  }

  /// Returns the sum of the diagonal elements (tr(A)).
  [[nodiscard]] DataType trace() const { return SparseLinearAlgebra::trace(*this); }

  /**
   * @brief Tests whether the sparsity pattern is symmetric about the diagonal.
   *
   * Checks only the index structure — values are ignored.  A matrix is
   * structurally symmetric if for every non-zero at (i, j) there is also a
   * non-zero at (j, i).
   *
   * @return @c true if the sparsity pattern is symmetric.
   */
  [[nodiscard]] bool is_structurally_symmetric() const {
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
  [[nodiscard]] bool is_sparse_symmetric(typename SparseMat::DataType TOLERANCE = 1e-6) const {
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
  [[nodiscard]] bool is_full_symmetric(typename SparseMat::DataType TOLERANCE = 1e-6) const {
    return SparseLinearAlgebra::is_full_symmetric(*this, TOLERANCE);
  }

  /// Returns @c true if the sparsity pattern has no above-diagonal non-zeros.
  [[nodiscard]] static constexpr bool is_structurally_lower_triangular() {
    return SparseLinearAlgebra::is_structurally_lower_triangular(SparseMat{});
  }

  /// Returns @c true if the sparsity pattern has no below-diagonal non-zeros.
  [[nodiscard]] static constexpr bool is_structurally_upper_triangular() {
    return SparseLinearAlgebra::is_structurally_upper_triangular(SparseMat{});
  }

  /**
   * @brief Returns @c true if every above-diagonal stored value is within
   *        @p tolerance of zero.
   * @param tolerance Maximum absolute value permitted above the diagonal.
   */
  [[nodiscard]] bool is_numerically_lower_triangular(DataType tolerance = 1e-6) const {
    return SparseLinearAlgebra::is_numerically_lower_triangular(*this, tolerance);
  }

  /**
   * @brief Returns @c true if every below-diagonal stored value is within
   *        @p tolerance of zero.
   * @param tolerance Maximum absolute value permitted below the diagonal.
   */
  [[nodiscard]] bool is_numerically_upper_triangular(DataType tolerance = 1e-6) const {
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
  template<std::size_t I, std::size_t J>
  void set(DataType value) {
    if constexpr (SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(I, J)) {
      constexpr int index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
      values[index] = value;
    } else {
      static_assert(SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(I, J),
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
  bool set(int i, int j, DataType value) {
    if (SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(i, j)) {
      int index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(i, j);
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
  void fill(DataType value) { values.fill(value); }

  /**
   * @brief Sets every stored diagonal element to @p value.
   *
   * Structurally zero diagonal positions are unaffected.
   *
   * @param value Scalar written to every stored diagonal entry.
   */
  void set_diagonal(DataType value) { SparseLinearAlgebra::set_diagonal(*this, value); }

  /**
   * @brief Sets the stored diagonal elements from a value array.
   *
   * Only structurally non-zero diagonal positions are written; each consumes
   * one entry from @p values in row order.
   *
   * @param values Values to write into stored diagonal entries, in row order.
   */
  void set_diagonal(std::span<DataType> values) {
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
  auto mult(const Matrix& m) const {
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
  auto solve(const Matrix& b) const {
    if constexpr (is_structurally_lower_triangular()) {
      return SparseLinearAlgebra::forward_solve(*this, b);
    } else if constexpr (is_structurally_upper_triangular()) {
      return SparseLinearAlgebra::backward_solve(*this, b);
    } else {
      return SparseLinearAlgebra::lu_solve(*this, b);
    }
  }

  auto cholesky() const { return SparseLinearAlgebra::Cholesky(*this); }

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
  auto add(const Matrix& other) {
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
  auto subtract(const Matrix& other) const {
    return SparseLinearAlgebra::subtract(*this, other);
  }

  /**
   * @brief Vector dot product.
   *
   * Requires either @c Rows == 1 (row vector) or @c Cols == 1 (column
   * vector).  For a row vector, computes @c *this × b and returns the scalar
   * at (0,0).  For a column vector, delegates to @c b.dot(*this).
   *
   * @tparam A     Type of the other vector.
   * @param  b     The other vector.
   * @return       Scalar dot product value.
   */
  template<typename Matrix>
  auto dot(const Matrix& b) const {
    if constexpr (Rows == 1) {
      return mult(b).template get<0, 0>();
    } else if constexpr (Cols == 1) {
      return b.dot(*this);
    } else {
      static_assert(Rows == 1 || Cols == 1, "Dot product is only defined for vectors.");
    }
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
  auto hadamard(const A& a) const {
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
  auto kronecker(const B& b) const {
    return SparseLinearAlgebra::kronecker(*this, b);
  }

  /**
   * @brief Returns the transpose of this matrix.
   *
   * Produces a @c SparseMat<DType, Cols, Rows, ...> with remapped indices.
   *
   * @return Transposed matrix.
   */
  auto transpose() const { return SparseLinearAlgebra::transpose(*this); }

  /**
   * @brief Returns a scaled copy: every non-zero element multiplied by @p factor.
   *
   * Sparsity pattern is unchanged.
   *
   * @param factor Scalar multiplier.
   * @return       Scaled matrix.
   */
  auto scale(DataType factor) const { return SparseLinearAlgebra::scale(*this, factor); }

  /**
   * @brief Multiplies every non-zero element by @p factor in place.
   * @param factor Scalar multiplier.
   */
  auto& scale_inplace(DataType factor) {
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
  auto shift(DataType factor) const { return SparseLinearAlgebra::shift(*this, factor); }

  /**
   * @brief Adds @p factor to every non-zero element in place.
   * @param factor Scalar to add.
   */
  auto& shift_inplace(DataType factor) {
    SparseLinearAlgebra::shift_inplace(*this, factor);
    return *this;
  }

  /**
   * @brief Returns a unit-norm copy of this matrix (divided by its Frobenius norm).
   * @return Normalized matrix.
   */
  auto normalize() const { return SparseLinearAlgebra::normalize(*this); }

  /**
   * @brief Divides every non-zero element by the Frobenius norm in place.
   */
  auto& normalize_inplace() {
    SparseLinearAlgebra::normalize_inplace(*this);
    return *this;
  }

  auto invert2x2() const
    requires(rows == 2 && cols == 2)
  {
    return SparseLinearAlgebra::invert2x2(*this);
  }

  /**
   * @brief Computes the Cholesky factorization and returns a solve handle.
   *
   * Factorizes this matrix as L * L^T (the matrix must be symmetric positive
   * definite) and returns a @c CholeskyFactor handle.  Call @c .solve(b) on
   * the handle to solve one or more right-hand sides without re-factorizing.
   *
   * @return A @c CholeskyFactor<L> handle for subsequent solves.
   */
  auto cholesky() const
    requires(rows == cols)
  {
    auto l = SparseLinearAlgebra::cholesky_factorize(*this);
    return SparseLinearAlgebra::CholeskyFactor<decltype(l)>(std::move(l));
  }

  /**
   * @brief Computes the Frobenius norm: √(Σ aᵢⱼ²) over all non-zero elements.
   * @return Frobenius norm as the matrix's @c DataType.
   */
  auto frobenius() const { return SparseLinearAlgebra::frobenius(*this); }

  /**
   * @brief Expands the sparse matrix into a fully dense row-major array.
   *
   * Zero positions are explicitly written as @c DataType(0).
   *
   * @return @c std::array<DataType, Rows*Cols> in row-major order.
   */
  auto dense() const { return SparseLinearAlgebra::dense(*this); }

  // --- Operator overloads ---

  /** @brief Matrix multiplication: @c *this × @p rhs. */
  template<typename Matrix>
  auto operator*(const Matrix& rhs) const {
    return mult(rhs);
  }

  /** @brief Element-wise addition: @c *this + @p rhs. */
  template<typename Matrix>
  auto operator+(const Matrix& rhs) const {
    return add(rhs);
  }

  /** @brief Element-wise subtraction: @c *this - @p rhs. */
  template<typename Matrix>
  auto operator-(const Matrix& rhs) const {
    return subtract(rhs);
  }

  /** @brief Scalar multiply (returns new matrix): @c *this * @p factor. */
  auto operator*(DataType factor) const { return scale(factor); }

  /** @brief Scalar multiply in place: @c *this *= @p factor. */
  SparseMat& operator*=(DataType factor) {
    scale_inplace(factor);
    return *this;
  }

  // --- Utility ---

  /**
   * @brief Prints each non-zero index and its value to @c std::cout.
   *
   * Iterates at compile time over the @c nonZeroCount stored elements.
   *
   * @tparam i Current compile-time index into @c values (default 0).
   */
  template<std::size_t i = 0>
  void print() const {
    if constexpr (i < nonZeroCount) {
      std::cout << "Value at index " << indices[i] << ": " << values[i] << '\n';
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
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        std::cout << d[(i * cols) + j] << " ";
      }
      std::cout << '\n';
    }
  }
};
