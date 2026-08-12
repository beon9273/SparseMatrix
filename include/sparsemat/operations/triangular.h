#pragma once
#include <cmath>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/result.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Compile-time and runtime triangular structure checks for a sparse matrix.
 *
 * Provides four independent predicates:
 *  - **Structurally lower/upper** — determined entirely from the sparsity
 *    pattern (the @c NonZeros index pack).  These are @c constexpr booleans
 *    computed once at instantiation time.
 *  - **Numerically lower/upper** — walks the stored values at runtime and
 *    verifies that every element above (lower) or below (upper) the main
 *    diagonal is within a caller-supplied tolerance of zero.
 *
 * A matrix is lower triangular if every element where `j > i` is zero;
 * upper triangular if every element where `i > j` is zero.
 *
 * @tparam SparseMat The sparse matrix type to analyze.
 */
template<typename SparseMat>
class Triangular {
  using Int = typename SparseMat::Int;

  // Fold over the matrix's own stored non-zeros (rather than walking the full
  // rows*cols grid) checking that none of them sits above the diagonal.
  // A structurally-zero position trivially satisfies "is zero", so only
  // stored positions need checking.
  template<std::size_t Idx>
  constexpr static bool structurally_lower_ok() {
    constexpr auto flat = SparseMat::indices()[Idx];
    constexpr Int row = flat / cols;
    constexpr Int col = flat % cols;
    return !(col > row);
  }
  template<std::size_t... Is>
  constexpr static bool is_structurally_lower_fold(std::index_sequence<Is...> /*seq*/) {
    return (structurally_lower_ok<Is>() && ...);
  }
  constexpr static bool is_structurally_lower() {
    return is_structurally_lower_fold(
        std::make_index_sequence<static_cast<std::size_t>(SparseMat::nonZeroCount)>{});
  }

  // Same idea for the below-diagonal check.
  template<std::size_t Idx>
  constexpr static bool structurally_upper_ok() {
    constexpr auto flat = SparseMat::indices()[Idx];
    constexpr Int row = flat / cols;
    constexpr Int col = flat % cols;
    return !(row > col);
  }
  template<std::size_t... Is>
  constexpr static bool is_structurally_upper_fold(std::index_sequence<Is...> /*seq*/) {
    return (structurally_upper_ok<Is>() && ...);
  }
  constexpr static bool is_structurally_upper() {
    return is_structurally_upper_fold(
        std::make_index_sequence<static_cast<std::size_t>(SparseMat::nonZeroCount)>{});
  }

 public:
  using DataType = typename SparseMat::DataType;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = SparseMat::cols;
  static constexpr auto num_non_zeros = SparseMat::nonZeroCount;
  static constexpr auto total_elements = rows * cols;

  /// @c true if no above-diagonal index is structurally non-zero.
  static constexpr bool structurally_lower = is_structurally_lower();
  /// @c true if no below-diagonal index is structurally non-zero.
  static constexpr bool structurally_upper = is_structurally_upper();

  /**
   * @brief Returns @c true if every above-diagonal stored value is within
   *        @p TOLERANCE of zero.
   *
   * Positions that are structurally zero are not visited (they are always
   * zero by definition).  Only non-zero positions with `J > I` are checked.
   *
   * @param a         Matrix to test.
   * @param TOLERANCE Maximum absolute value allowed above the diagonal.
   */
  // Fold over the matrix's own stored non-zeros (rather than walking the
  // full rows*cols grid): a structurally-zero position is trivially within
  // any tolerance, so only stored positions above the diagonal need
  // checking. Idx doubles as both the pack index and the storage index into
  // a.values, since SparseMat::indices() and values are parallel arrays.
  template<std::size_t Idx>
  SPARSEMAT_HD static bool numerically_lower_ok(const SparseMat& a, DataType TOLERANCE) {
    constexpr auto flat = SparseMat::indices()[Idx];
    constexpr Int row = flat / cols;
    constexpr Int col = flat % cols;
    if constexpr (col > row) {
      return std::abs(a.values[Idx]) <= TOLERANCE;
    } else {
      return true;
    }
  }
  template<std::size_t... Is>
  SPARSEMAT_HD static bool is_numerically_lower_fold(const SparseMat& a,
                                                     DataType TOLERANCE,
                                                     std::index_sequence<Is...> /*seq*/) {
    return (numerically_lower_ok<Is>(a, TOLERANCE) && ...);
  }
  SPARSEMAT_HD static bool is_numerically_lower(const SparseMat& a, DataType TOLERANCE = 1e-6) {
    return is_numerically_lower_fold(
        a,
        TOLERANCE,
        std::make_index_sequence<static_cast<std::size_t>(SparseMat::nonZeroCount)>{});
  }

  /**
   * @brief Returns @c true if every below-diagonal stored value is within
   *        @p TOLERANCE of zero.
   *
   * Positions that are structurally zero are not visited.  Only non-zero
   * positions with `I > J` are checked.
   *
   * @param a         Matrix to test.
   * @param TOLERANCE Maximum absolute value allowed below the diagonal.
   */
  template<std::size_t Idx>
  SPARSEMAT_HD static bool numerically_upper_ok(const SparseMat& a, DataType TOLERANCE) {
    constexpr auto flat = SparseMat::indices()[Idx];
    constexpr Int row = flat / cols;
    constexpr Int col = flat % cols;
    if constexpr (row > col) {
      return std::abs(a.values[Idx]) <= TOLERANCE;
    } else {
      return true;
    }
  }
  template<std::size_t... Is>
  SPARSEMAT_HD static bool is_numerically_upper_fold(const SparseMat& a,
                                                     DataType TOLERANCE,
                                                     std::index_sequence<Is...> /*seq*/) {
    return (numerically_upper_ok<Is>(a, TOLERANCE) && ...);
  }
  SPARSEMAT_HD static bool is_numerically_upper(const SparseMat& a, DataType TOLERANCE = 1e-6) {
    return is_numerically_upper_fold(
        a,
        TOLERANCE,
        std::make_index_sequence<static_cast<std::size_t>(SparseMat::nonZeroCount)>{});
  }

  // Single term of sum(A[I][J] * result[J][C]) for J < I — the already-solved
  // lower entries for column C of the block RHS, used by forward_solve. J
  // ranges over a fold-generated pack sized to A::rows (an over-generous
  // upper bound on I, since I < A::rows); terms with J >= I are filtered out
  // here rather than by sizing the pack exactly, since the exact bound (I)
  // isn't known until this template is instantiated anyway and the filtered
  // terms cost nothing at runtime.
  template<SparseMatrixType A,
           SparseMatrixType B,
           typename A::Int I,
           typename B::Int C,
           std::size_t J>
  SPARSEMAT_HD static DataType lower_product_term(const A& a, const B& b) {
    if constexpr (static_cast<typename A::Int>(J) >= I) {
      return DataType(0);
    } else {
      constexpr auto sparse_a_index =
          SparseLinearAlgebra::MatrixUtilities<A>::getSparseIndex(I,
                                                                  static_cast<typename A::Int>(J));
      constexpr auto sparse_b_index =
          SparseLinearAlgebra::MatrixUtilities<B>::getSparseIndex(static_cast<typename A::Int>(J),
                                                                  C);
      if constexpr (sparse_a_index < 0 || sparse_b_index < 0) {
        return DataType(0);
      } else {
        return a.values[sparse_a_index] * b.values[sparse_b_index];
      }
    }
  }
  template<SparseMatrixType A,
           SparseMatrixType B,
           typename A::Int I,
           typename B::Int C,
           std::size_t... Js>
  SPARSEMAT_HD static DataType do_lower_product_fold(const A& a,
                                                     const B& b,
                                                     std::index_sequence<Js...> /*seq*/) {
    return (lower_product_term<A, B, I, C, Js>(a, b) + ...);
  }
  template<SparseMatrixType A, SparseMatrixType B, typename A::Int I, typename B::Int C>
  SPARSEMAT_HD static DataType do_lower_product(const A& a, const B& b) {
    return do_lower_product_fold<A, B, I, C>(
        a, b, std::make_index_sequence<static_cast<std::size_t>(A::rows)>{});
  }

  // Single term of sum(A[I][J] * result[J][C]) for J > I — the already-solved
  // upper entries for column C of the block RHS, used by backward_solve.
  // J ranges over the full [0, cols) pack, same "filter inside the term"
  // approach as do_lower_product above.
  template<SparseMatrixType A,
           SparseMatrixType B,
           typename A::Int I,
           typename B::Int C,
           std::size_t J>
  SPARSEMAT_HD static DataType upper_product_term(const A& a, const B& b) {
    if constexpr (static_cast<typename A::Int>(J) <= I) {
      return DataType(0);
    } else {
      constexpr auto sparse_a_index =
          SparseLinearAlgebra::MatrixUtilities<A>::getSparseIndex(I,
                                                                  static_cast<typename A::Int>(J));
      constexpr auto sparse_b_index =
          SparseLinearAlgebra::MatrixUtilities<B>::getSparseIndex(static_cast<typename A::Int>(J),
                                                                  C);
      if constexpr (sparse_a_index < 0 || sparse_b_index < 0) {
        return DataType(0);
      } else {
        return a.values[sparse_a_index] * b.values[sparse_b_index];
      }
    }
  }
  template<SparseMatrixType A,
           SparseMatrixType B,
           typename A::Int I,
           typename B::Int C,
           std::size_t... Js>
  SPARSEMAT_HD static DataType do_upper_product_fold(const A& a,
                                                     const B& b,
                                                     std::index_sequence<Js...> /*seq*/) {
    return (upper_product_term<A, B, I, C, Js>(a, b) + ...);
  }
  template<SparseMatrixType A, SparseMatrixType B, typename A::Int I, typename B::Int C>
  SPARSEMAT_HD static DataType do_upper_product(const A& a, const B& b) {
    return do_upper_product_fold<A, B, I, C>(
        a, b, std::make_index_sequence<static_cast<std::size_t>(cols)>{});
  }

  /**
   * @brief Solves a lower triangular system Ax = b via forward substitution.
   *
   * Iterates from row 0 up to N-1.  At each row I:
   *   x[I] = (b[I] - sum(A[I][J] * x[J], J < I)) / A[I][I]
   *
   * @tparam Result Type of the output vector (column vector SparseMat).
   * @tparam RHS    Type of the right-hand side vector.
   * @param result  Output vector, mutated in place.
   * @param A       Lower triangular matrix.
   * @param b       Right-hand side vector.
   */
  // Single RHS-column step for a fixed row I in forward substitution.
  template<SparseMatrixType Result, SparseMatrixType RHS, Int I, std::size_t COff>
  SPARSEMAT_HD static void forward_solve_col_step(Result& result,
                                                  const SparseMat& A,
                                                  const RHS& b,
                                                  bool& ok) {
    constexpr Int C = static_cast<Int>(COff);
    constexpr auto diag_index =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, I);
    constexpr auto result_index =
        SparseLinearAlgebra::MatrixUtilities<Result>::getSparseIndex(I, C);
    if constexpr (result_index >= 0) {
      if constexpr (diag_index < 0) {
        result.values[result_index] = DataType(0);
      } else {
        if (A.values[diag_index] == DataType(0)) {
          ok = false;
          result.values[result_index] = DataType(0);
          return;
        }
        constexpr auto rhs_index = SparseLinearAlgebra::MatrixUtilities<RHS>::getSparseIndex(I, C);
        DataType sum = do_lower_product<SparseMat, Result, I, C>(A, result);
        if constexpr (rhs_index >= 0) {
          result.values[result_index] = (b.values[rhs_index] - sum) / A.values[diag_index];
        } else {
          result.values[result_index] = -sum / A.values[diag_index];
        }
      }
    }
  }
  // Fills all RHS columns for a fixed row I in forward substitution. Column
  // order doesn't matter (each writes an independent result cell), but a
  // comma fold is used for consistency with the row-sequential fold below.
  template<SparseMatrixType Result, SparseMatrixType RHS, Int I, std::size_t... COffs>
  SPARSEMAT_HD static void forward_solve_col(Result& result,
                                             const SparseMat& A,
                                             const RHS& b,
                                             bool& ok,
                                             std::index_sequence<COffs...> /*seq*/) {
    (forward_solve_col_step<Result, RHS, I, COffs>(result, A, b, ok), ...);
  }

  // Single row step: row I's forward_solve_col reads result cells from rows
  // < I written by earlier steps, so row order is load-bearing.
  template<SparseMatrixType Result, SparseMatrixType RHS, std::size_t IOff>
  SPARSEMAT_HD static void forward_solve_row_step(Result& result,
                                                  const SparseMat& A,
                                                  const RHS& b,
                                                  bool& ok) {
    forward_solve_col<Result, RHS, static_cast<Int>(IOff)>(
        result, A, b, ok, std::make_index_sequence<static_cast<std::size_t>(RHS::cols)>{});
  }
  // Comma fold over rows 0..rows-1, in order: the comma operator inside a
  // fold expression is the built-in sequencing comma (left fully sequenced
  // before right), which is what makes it safe to replace the original
  // sequential recursion here — row I+1 is only ever evaluated after row I
  // has finished writing its result cells.
  template<SparseMatrixType Result, SparseMatrixType RHS, std::size_t... IOffs>
  SPARSEMAT_HD static Result& forward_solve_rows(Result& result,
                                                 const SparseMat& A,
                                                 const RHS& b,
                                                 bool& ok,
                                                 std::index_sequence<IOffs...> /*seq*/) {
    (forward_solve_row_step<Result, RHS, IOffs>(result, A, b, ok), ...);
    return result;
  }

  template<SparseMatrixType Result, SparseMatrixType RHS>
  SPARSEMAT_HD auto forward_solve(Result& result, const SparseMat& A, const RHS& b, bool& ok) {
    static_assert(structurally_lower, "Matrix is not structurally lower triangular.");
    return forward_solve_rows<Result, RHS>(result, A, b, ok, std::make_index_sequence<rows>{});
  }

  // Single RHS-column step for a fixed row I in back substitution.
  template<SparseMatrixType Result, SparseMatrixType RHS, Int I, std::size_t COff>
  SPARSEMAT_HD static void backward_solve_col_step(Result& result,
                                                   const SparseMat& A,
                                                   const RHS& b,
                                                   bool& ok) {
    constexpr Int C = static_cast<Int>(COff);
    constexpr auto diag_index =
        SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, I);
    constexpr auto result_index =
        SparseLinearAlgebra::MatrixUtilities<Result>::getSparseIndex(I, C);
    if constexpr (result_index >= 0) {
      if constexpr (diag_index < 0) {
        result.values[result_index] = DataType(0);
      } else {
        if (A.values[diag_index] == DataType(0)) {
          ok = false;
          result.values[result_index] = DataType(0);
          return;
        }
        constexpr auto rhs_index = SparseLinearAlgebra::MatrixUtilities<RHS>::getSparseIndex(I, C);
        DataType sum = do_upper_product<SparseMat, Result, I, C>(A, result);
        if constexpr (rhs_index >= 0) {
          result.values[result_index] = (b.values[rhs_index] - sum) / A.values[diag_index];
        } else {
          result.values[result_index] = -sum / A.values[diag_index];
        }
      }
    }
  }
  template<SparseMatrixType Result, SparseMatrixType RHS, Int I, std::size_t... COffs>
  SPARSEMAT_HD static void backward_solve_col(Result& result,
                                              const SparseMat& A,
                                              const RHS& b,
                                              bool& ok,
                                              std::index_sequence<COffs...> /*seq*/) {
    (backward_solve_col_step<Result, RHS, I, COffs>(result, A, b, ok), ...);
  }

  /**
   * @brief Solves an upper triangular system Ax = b via back substitution.
   *
   * Iterates from row N-1 down to 0.  At each row I:
   *   x[I][C] = (b[I][C] - sum(A[I][J] * x[J][C], J > I)) / A[I][I]
   * for each RHS column C.
   *
   * @tparam Result Type of the output matrix.
   * @tparam RHS    Type of the right-hand side (one or more columns).
   * @param result  Output matrix, mutated in place.
   * @param A       Upper triangular matrix.
   * @param b       Right-hand side.
   */
  // Single row step, descending: IOff=0,1,2,... maps to I=rows-1,rows-2,...,0
  // so the ascending fold below still visits rows in the original
  // descending order without needing a genuinely-reversed index_sequence.
  template<SparseMatrixType Result, SparseMatrixType RHS, std::size_t IOff>
  SPARSEMAT_HD static void backward_solve_row_step(Result& result,
                                                   const SparseMat& A,
                                                   const RHS& b,
                                                   bool& ok) {
    constexpr Int I = static_cast<Int>(rows) - 1 - static_cast<Int>(IOff);
    backward_solve_col<Result, RHS, I>(
        result, A, b, ok, std::make_index_sequence<static_cast<std::size_t>(RHS::cols)>{});
  }
  template<SparseMatrixType Result, SparseMatrixType RHS, std::size_t... IOffs>
  SPARSEMAT_HD static Result& backward_solve_rows(Result& result,
                                                  const SparseMat& A,
                                                  const RHS& b,
                                                  bool& ok,
                                                  std::index_sequence<IOffs...> /*seq*/) {
    (backward_solve_row_step<Result, RHS, IOffs>(result, A, b, ok), ...);
    return result;
  }

  template<SparseMatrixType Result, SparseMatrixType RHS>
  SPARSEMAT_HD auto backward_solve(Result& result, const SparseMat& A, const RHS& b, bool& ok) {
    static_assert(structurally_upper, "Matrix is not structurally upper triangular.");
    return backward_solve_rows<Result, RHS>(result, A, b, ok, std::make_index_sequence<rows>{});
  }
};

/**
 * @brief Result-type helper for forward_solve.
 *
 * Computes the sparsity of the solution vector x when solving Lx = b via
 * forward sparsity propagation: x[i] is non-zero if b[i] is non-zero, or if
 * any x[j] (j < i) is non-zero and L[i][j] is non-zero.  Provides `make_result()`
 * which returns a default-constructed output vector of the correct type.
 *
 * @tparam SparseMat Lower triangular matrix type.
 * @tparam RHS       Right-hand side vector type.
 */
template<typename SparseMat, typename RHS>
class LowerTriangular {
 public:
  using Int = typename SparseMat::Int;

 private:
  // Forward sparsity propagation: x[i] is non-zero if b[i] is non-zero, or if
  // any already-determined non-zero x[j] (j < i) connects through L[i][j].
  // Row i is potentially non-zero if any column of RHS has a non-zero at row i,
  // or if any earlier non-zero row j connects to i through L[i][j].
  //
  // Both operands are memoized as dense bool grids first (see
  // MatrixUtilities::to_dense_bool()): the inner connectivity test below runs
  // inside a triple-nested loop, and a raw indices() scan there would make
  // this O(rows^2 * nonZeroCount) constexpr work rather than O(rows^2).
  static constexpr std::array<bool, SparseMat::rows> compute_nonzero_rows() {
    constexpr auto l_grid = SparseLinearAlgebra::MatrixUtilities<SparseMat>::to_dense_bool();
    constexpr auto rhs_grid = SparseLinearAlgebra::MatrixUtilities<RHS>::to_dense_bool();
    std::array<bool, SparseMat::rows> nz{};
    for (Int i = 0; i < SparseMat::rows; ++i) {
      for (Int c = 0; c < RHS::cols; ++c) {
        if (rhs_grid[i][c]) {
          nz[i] = true;
          break;
        }
      }
      if (nz[i]) {
        continue;
      }
      for (Int j = 0; j < i; ++j) {
        if (nz[j] && l_grid[i][j]) {
          nz[i] = true;
          break;
        }
      }
    }
    return nz;
  }

  static constexpr Int count_nonzero_rows_lower() {
    Int count = 0;
    for (bool b : compute_nonzero_rows()) {
      if (b) {
        ++count;
      }
    }
    return count;
  }

  static constexpr auto get_nonzero_rows_lower() {
    auto nz = compute_nonzero_rows();
    std::array<Int, static_cast<std::size_t>(count_nonzero_rows_lower())> result{};
    Int k = 0;
    for (Int i = 0; i < SparseMat::rows; ++i) {
      if (nz[i]) {
        result[k++] = i;
      }
    }
    return result;
  }

 public:
  using DataType = typename SparseMat::DataType;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = RHS::cols;  // block: result has same column count as RHS
  static constexpr std::array<Int, static_cast<std::size_t>(count_nonzero_rows_lower())> nonZeros =
      get_nonzero_rows_lower();

  // (row, col) is non-zero for every column of a potentially non-zero row.
  SPARSEMAT_HD constexpr static bool is_result_index_nonzero(Int row, Int /*col*/) {
    for (Int idx : nonZeros) {
      if (idx == row) {
        return true;
      }
    }
    return false;
  }

  static constexpr auto numNonzeros =
      SparseLinearAlgebra::OperationUtilities<LowerTriangular>::num_nonzeros();
  static constexpr auto sparsity =
      SparseLinearAlgebra::OperationUtilities<LowerTriangular>::calculate_sparsity();

  SPARSEMAT_HD static auto make_result() {
    return SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
        std::make_index_sequence<static_cast<std::size_t>(numNonzeros)>{});
  }
};

/**
 * @brief Result-type helper for backward_solve.
 *
 * Computes the sparsity of the solution vector x when solving Ux = b via
 * backward sparsity propagation: x[i] is non-zero if b[i] is non-zero, or if
 * any x[j] (j > i) is non-zero and U[i][j] is non-zero.  Provides `make_result()`
 * which returns a default-constructed output vector of the correct type.
 *
 * @tparam SparseMat Upper triangular matrix type.
 * @tparam RHS       Right-hand side vector type.
 */
template<typename SparseMat, typename RHS>
class UpperTriangular {
 public:
  using Int = typename SparseMat::Int;

 private:
  // Row i is potentially non-zero if any column of RHS has a non-zero at row i,
  // or if any later non-zero row j connects to i through U[i][j].
  static constexpr std::array<bool, SparseMat::rows> compute_nonzero_rows() {
    std::array<bool, SparseMat::rows> nz{};
    for (Int ii = 0; ii < SparseMat::rows; ++ii) {
      Int i = SparseMat::rows - 1 - ii;
      for (auto idx : RHS::indices()) {
        if (idx / RHS::cols == i) {
          nz[i] = true;
          break;
        }
      }
      if (nz[i]) {
        continue;
      }
      for (Int j = i + 1; j < SparseMat::cols; ++j) {
        if (!nz[j]) {
          continue;
        }
        Int flat = i * SparseMat::cols + j;
        for (auto idx : SparseMat::indices()) {
          if (idx == flat) {
            nz[i] = true;
            break;
          }
        }
        if (nz[i]) {
          break;
        }
      }
    }
    return nz;
  }

  static constexpr Int count_nonzero_rows_upper() {
    Int count = 0;
    for (bool b : compute_nonzero_rows()) {
      if (b) {
        ++count;
      }
    }
    return count;
  }

  static constexpr auto get_nonzero_rows_upper() {
    auto nz = compute_nonzero_rows();
    std::array<Int, static_cast<std::size_t>(count_nonzero_rows_upper())> result{};
    Int k = 0;
    for (Int i = 0; i < SparseMat::rows; ++i) {
      if (nz[i]) {
        result[k++] = i;
      }
    }
    return result;
  }

 public:
  using DataType = typename SparseMat::DataType;
  static constexpr auto rows = SparseMat::rows;
  static constexpr auto cols = RHS::cols;  // block: result has same column count as RHS
  static constexpr std::array<Int, static_cast<std::size_t>(count_nonzero_rows_upper())> nonZeros =
      get_nonzero_rows_upper();

  // (row, col) is non-zero for every column of a potentially non-zero row.
  SPARSEMAT_HD constexpr static bool is_result_index_nonzero(Int row, Int /*col*/) {
    for (Int idx : nonZeros) {
      if (idx == row) {
        return true;
      }
    }
    return false;
  }

  static constexpr auto numNonzeros =
      SparseLinearAlgebra::OperationUtilities<UpperTriangular>::num_nonzeros();
  static constexpr auto sparsity =
      SparseLinearAlgebra::OperationUtilities<UpperTriangular>::calculate_sparsity();
  SPARSEMAT_HD static auto make_result() {
    return SparseLinearAlgebra::MatrixUtilities<SparseMat>::template make<rows, cols, sparsity>(
        std::make_index_sequence<static_cast<std::size_t>(numNonzeros)>{});
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/// Returns @c true if the sparsity pattern has no above-diagonal non-zeros.
template<SparseMatrixType SparseMat>
SPARSEMAT_HD constexpr auto is_structurally_lower_triangular(const SparseMat& /*a*/) {
  return detail::Triangular<SparseMat>::structurally_lower;
}

/// Returns @c true if every above-diagonal stored value is within tolerance of zero.
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto is_numerically_lower_triangular(const SparseMat& a,
                                                  typename SparseMat::DataType tolerance = 1e-6) {
  return detail::Triangular<SparseMat>::is_numerically_lower(a, tolerance);
}

/// Returns @c true if the sparsity pattern has no below-diagonal non-zeros.
template<SparseMatrixType SparseMat>
SPARSEMAT_HD constexpr auto is_structurally_upper_triangular(const SparseMat& /*a*/) {
  return detail::Triangular<SparseMat>::structurally_upper;
}

/// Returns @c true if every below-diagonal stored value is within tolerance of zero.
template<SparseMatrixType SparseMat>
SPARSEMAT_HD auto is_numerically_upper_triangular(const SparseMat& a,
                                                  typename SparseMat::DataType tolerance = 1e-6) {
  return detail::Triangular<SparseMat>::is_numerically_upper(a, tolerance);
}

/**
 * @brief Solves the lower triangular system Ax = b.
 *
 * Allocates the result vector via @c LowerTriangular, then runs forward
 * substitution.  A must be structurally lower triangular (enforced by
 * static_assert inside the solver).
 *
 * @param A Lower triangular matrix.
 * @param b Right-hand side column vector.
 * @return  @c Result wrapping the solution vector x such that Ax = b;
 *          @c ok() is @c false if a zero pivot was hit on the diagonal.
 */
template<SparseMatrixType SparseMat, SparseMatrixType RHS>
SPARSEMAT_HD auto forward_solve(const SparseMat& A, const RHS& b) {
  static_assert(SparseMat::rows == SparseMat::cols,
                "forward_solve requires a square coefficient matrix.");
  static_assert(SparseMat::rows == RHS::rows, "forward_solve requires RHS::rows == A.rows.");
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, RHS>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types \u2014 convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");
  auto result = detail::LowerTriangular<SparseMat, RHS>::make_result();
  bool ok = true;
  detail::Triangular<SparseMat>().forward_solve(result, A, b, ok);
  return Result<decltype(result)>(std::move(result),
                                  ok ? SolveStatus::Success : SolveStatus::SingularMatrix);
}

/**
 * @brief Solves the upper triangular system Ax = b.
 *
 * Allocates the result vector via @c UpperTriangular, then runs back
 * substitution.  A must be structurally upper triangular (enforced by
 * static_assert inside the solver).
 *
 * @param A Upper triangular matrix.
 * @param b Right-hand side column vector.
 * @return  @c Result wrapping the solution vector x such that Ax = b;
 *          @c ok() is @c false if a zero pivot was hit on the diagonal.
 */
template<SparseMatrixType SparseMat, SparseMatrixType RHS>
SPARSEMAT_HD auto backward_solve(const SparseMat& A, const RHS& b) {
  static_assert(SparseMat::rows == SparseMat::cols,
                "backward_solve requires a square coefficient matrix.");
  static_assert(SparseMat::rows == RHS::rows, "backward_solve requires RHS::rows == A.rows.");
  static_assert(SparseLinearAlgebra::SameDataType<SparseMat, RHS>,
                "Operands must have the same DataType. sparsemat does not promote mixed "
                "scalar types \u2014 convert one operand explicitly first, e.g. "
                "a.template convert<double>() or SparseLinearAlgebra::convert<double>(a).");
  auto result = detail::UpperTriangular<SparseMat, RHS>::make_result();
  bool ok = true;
  detail::Triangular<SparseMat>().backward_solve(result, A, b, ok);
  return Result<decltype(result)>(std::move(result),
                                  ok ? SolveStatus::Success : SolveStatus::SingularMatrix);
}

}  // namespace SparseLinearAlgebra
