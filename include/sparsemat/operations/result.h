#pragma once

#include <utility>

#include "sparsemat/concepts/concepts.h"

namespace SparseLinearAlgebra {

/// Outcome of a numeric solve/factorization.
enum class SolveStatus {
  Success,
  SingularMatrix,  ///< Hit a zero (or non-positive, for Cholesky) pivot.
};

/**
 * @brief Non-throwing result wrapper for solvers that may hit a zero pivot.
 *
 * Exceptions aren't an option here because solve/factorize routines are
 * marked @c SPARSEMAT_HD and may run in CUDA device code. @c Result always
 * carries a value — computed using whatever pivot was found, even if it was
 * zero — plus a status flag the caller must check before trusting it.
 *
 * @tparam T Wrapped value type (e.g. a solution vector or factor matrix).
 */
template<typename T>
class Result {
 public:
  SPARSEMAT_HD explicit Result(T value, SolveStatus status = SolveStatus::Success)
      : value_(std::move(value)), status_(status) {}

  /// @c true if no zero pivot was encountered.
  [[nodiscard]] SPARSEMAT_HD bool ok() const { return status_ == SolveStatus::Success; }
  [[nodiscard]] SPARSEMAT_HD explicit operator bool() const { return ok(); }
  [[nodiscard]] SPARSEMAT_HD SolveStatus status() const { return status_; }

  /// Accesses the wrapped value. Only meaningful when @c ok() is @c true;
  /// a failed result still holds a (numerically garbage) value rather than
  /// asserting, since assertions/exceptions aren't safe in device code.
  [[nodiscard]] SPARSEMAT_HD const T& value() const { return value_; }
  [[nodiscard]] SPARSEMAT_HD T& value() { return value_; }

 private:
  T value_;
  SolveStatus status_;
};

}  // namespace SparseLinearAlgebra
