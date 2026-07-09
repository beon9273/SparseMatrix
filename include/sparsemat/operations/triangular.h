#pragma once
#include <cmath>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/result.h"

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

  // Walks the index pack at compile time; returns false as soon as any
  // above-diagonal position (J > I) is structurally non-zero.
  template<Int I = 0, Int J = 1>
  constexpr static bool is_structurally_lower() {
    if constexpr (I >= rows) {
      return true;
    } else if constexpr (J >= cols) {
      return is_structurally_lower<I + 1, I + 2>();
    } else if constexpr (J > I &&
                         SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(I, J)) {
      return false;
    } else {
      return is_structurally_lower<I, J + 1>();
    }
  }

  // Walks the index pack at compile time; returns false as soon as any
  // below-diagonal position (I > J) is structurally non-zero.
  template<Int I = 1, Int J = 0>
  constexpr static bool is_structurally_upper() {
    if constexpr (I >= rows) {
      return true;
    } else if constexpr (J >= cols) {
      return is_structurally_upper<I + 1, 0>();
    } else if constexpr (I > J &&
                         SparseLinearAlgebra::MatrixUtilities<SparseMat>::isNonZero(I, J)) {
      return false;
    } else {
      return is_structurally_upper<I, J + 1>();
    }
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
  template<Int I = 0, Int J = 1>
  SPARSEMAT_HD static bool is_numerically_lower(const SparseMat& a, DataType TOLERANCE = 1e-6) {
    if constexpr (I >= rows) {
      return true;
    } else if constexpr (J >= cols) {
      return is_numerically_lower<I + 1, I + 2>(a, TOLERANCE);
    } else if constexpr (J > I) {
      constexpr auto index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
      if constexpr (index >= 0) {
        if (std::abs(a.values[index]) > TOLERANCE) {
          return false;
        }
      }
      return is_numerically_lower<I, J + 1>(a, TOLERANCE);
    } else {
      return is_numerically_lower<I, J + 1>(a, TOLERANCE);
    }
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
  template<Int I = 1, Int J = 0>
  SPARSEMAT_HD static bool is_numerically_upper(const SparseMat& a, DataType TOLERANCE = 1e-6) {
    if constexpr (I >= rows) {
      return true;
    } else if constexpr (J >= cols) {
      return is_numerically_upper<I + 1, 0>(a, TOLERANCE);
    } else if constexpr (I > J) {
      constexpr auto index = SparseLinearAlgebra::MatrixUtilities<SparseMat>::getSparseIndex(I, J);
      if constexpr (index >= 0) {
        if (std::abs(a.values[index]) > TOLERANCE) {
          return false;
        }
      }
      return is_numerically_upper<I, J + 1>(a, TOLERANCE);
    } else {
      return is_numerically_upper<I, J + 1>(a, TOLERANCE);
    }
  }

  // Computes sum(A[I][J] * result[J][C]) for J < I — the already-solved lower
  // entries for column C of the block RHS, used by forward_solve.
  template<SparseMatrixType A,
           SparseMatrixType B,
           typename A::Int I,
           typename B::Int C,
           typename A::Int J = 0>
  SPARSEMAT_HD static DataType do_lower_product(const A& a, const B& b) {
    if constexpr (J >= I) {
      return DataType(0);
    } else {
      constexpr auto sparse_a_index = SparseLinearAlgebra::MatrixUtilities<A>::getSparseIndex(I, J);
      constexpr auto sparse_b_index = SparseLinearAlgebra::MatrixUtilities<B>::getSparseIndex(J, C);
      if constexpr (sparse_a_index < 0 || sparse_b_index < 0) {
        return do_lower_product<A, B, I, C, J + 1>(a, b);
      } else {
        return a.values[sparse_a_index] * b.values[sparse_b_index] +
               do_lower_product<A, B, I, C, J + 1>(a, b);
      }
    }
  }

  // Computes sum(A[I][J] * result[J][C]) for J > I — the already-solved upper
  // entries for column C of the block RHS, used by backward_solve.
  template<SparseMatrixType A,
           SparseMatrixType B,
           typename A::Int I,
           typename B::Int C,
           typename A::Int J = 0>
  SPARSEMAT_HD static DataType do_upper_product(const A& a, const B& b) {
    if constexpr (J >= cols) {
      return DataType(0);
    } else if constexpr (J <= I) {
      return do_upper_product<A, B, I, C, J + 1>(a, b);
    } else {
      constexpr auto sparse_a_index = SparseLinearAlgebra::MatrixUtilities<A>::getSparseIndex(I, J);
      constexpr auto sparse_b_index = SparseLinearAlgebra::MatrixUtilities<B>::getSparseIndex(J, C);
      if constexpr (sparse_a_index < 0 || sparse_b_index < 0) {
        return do_upper_product<A, B, I, C, J + 1>(a, b);
      } else {
        return a.values[sparse_a_index] * b.values[sparse_b_index] +
               do_upper_product<A, B, I, C, J + 1>(a, b);
      }
    }
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
  // Inner loop: fills all RHS columns for a fixed row I in forward substitution.
  template<SparseMatrixType Result, SparseMatrixType RHS, Int I, Int C = 0>
  SPARSEMAT_HD void forward_solve_col(Result& result, const SparseMat& A, const RHS& b, bool& ok) {
    if constexpr (C < RHS::cols) {
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
            forward_solve_col<Result, RHS, I, C + 1>(result, A, b, ok);
            return;
          }
          constexpr auto rhs_index =
              SparseLinearAlgebra::MatrixUtilities<RHS>::getSparseIndex(I, C);
          DataType sum = do_lower_product<SparseMat, Result, I, C>(A, result);
          if constexpr (rhs_index >= 0) {
            result.values[result_index] = (b.values[rhs_index] - sum) / A.values[diag_index];
          } else {
            result.values[result_index] = -sum / A.values[diag_index];
          }
        }
      }
      forward_solve_col<Result, RHS, I, C + 1>(result, A, b, ok);
    }
  }

  template<SparseMatrixType Result, SparseMatrixType RHS, Int I = 0>
  SPARSEMAT_HD auto forward_solve(Result& result, const SparseMat& A, const RHS& b, bool& ok) {
    static_assert(structurally_lower, "Matrix is not structurally lower triangular.");
    if constexpr (I < rows) {
      forward_solve_col<Result, RHS, I>(result, A, b, ok);
      return forward_solve<Result, RHS, I + 1>(result, A, b, ok);
    } else {
      return result;
    }
  }

  // Inner loop: fills all RHS columns for a fixed row I in back substitution.
  template<SparseMatrixType Result, SparseMatrixType RHS, Int I, Int C = 0>
  SPARSEMAT_HD void backward_solve_col(Result& result, const SparseMat& A, const RHS& b, bool& ok) {
    if constexpr (C < RHS::cols) {
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
            backward_solve_col<Result, RHS, I, C + 1>(result, A, b, ok);
            return;
          }
          constexpr auto rhs_index =
              SparseLinearAlgebra::MatrixUtilities<RHS>::getSparseIndex(I, C);
          DataType sum = do_upper_product<SparseMat, Result, I, C>(A, result);
          if constexpr (rhs_index >= 0) {
            result.values[result_index] = (b.values[rhs_index] - sum) / A.values[diag_index];
          } else {
            result.values[result_index] = -sum / A.values[diag_index];
          }
        }
      }
      backward_solve_col<Result, RHS, I, C + 1>(result, A, b, ok);
    }
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
  template<SparseMatrixType Result, SparseMatrixType RHS, int I = rows - 1>
  SPARSEMAT_HD auto backward_solve(Result& result, const SparseMat& A, const RHS& b, bool& ok) {
    static_assert(structurally_upper, "Matrix is not structurally upper triangular.");
    if constexpr (I >= 0) {
      backward_solve_col<Result, RHS, I>(result, A, b, ok);
      return backward_solve<Result, RHS, I - 1>(result, A, b, ok);
    } else {
      return result;
    }
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
  static constexpr std::array<bool, SparseMat::rows> compute_nonzero_rows() {
    std::array<bool, SparseMat::rows> nz{};
    for (Int i = 0; i < SparseMat::rows; ++i) {
      for (auto idx : RHS::indices()) {
        if (idx / RHS::cols == i) {
          nz[i] = true;
          break;
        }
      }
      if (nz[i]) {
        continue;
      }
      for (Int j = 0; j < i; ++j) {
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
    std::array<Int, count_nonzero_rows_lower()> result{};
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
  static constexpr std::array<Int, count_nonzero_rows_lower()> nonZeros = get_nonzero_rows_lower();

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
        std::make_index_sequence<numNonzeros>{});
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
    std::array<Int, count_nonzero_rows_upper()> result{};
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
  static constexpr std::array<Int, count_nonzero_rows_upper()> nonZeros = get_nonzero_rows_upper();

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
        std::make_index_sequence<numNonzeros>{});
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
  auto result = detail::UpperTriangular<SparseMat, RHS>::make_result();
  bool ok = true;
  detail::Triangular<SparseMat>().backward_solve(result, A, b, ok);
  return Result<decltype(result)>(std::move(result),
                                  ok ? SolveStatus::Success : SolveStatus::SingularMatrix);
}

}  // namespace SparseLinearAlgebra
