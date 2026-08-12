#pragma once

#include <cstdint>
#include <limits>
#include <utility>

#include "sparsemat/concepts/concepts.h"

namespace SparseLinearAlgebra {

/// Outcome of a numeric solve/factorization.
enum class SolveStatus : std::uint8_t {
  Success,
  SingularMatrix,  ///< Hit a negligible (or non-positive, for Cholesky) pivot.
};

/**
 * @brief Relative magnitude below which a pivot is treated as singular.
 *
 * None of the factorizations here pivot, so a merely *tiny* pivot is as fatal
 * as an exactly-zero one: it is divided through and yields enormous,
 * meaningless multipliers. Callers get told via @c SolveStatus::SingularMatrix
 * rather than silently receiving garbage with @c ok() == @c true.
 *
 * The value is a few epsilons of the scalar type — loose enough not to reject
 * merely ill-conditioned-but-usable systems, tight enough to catch a pivot
 * that has been annihilated by cancellation. It is multiplied by a magnitude
 * scale drawn from the matrix at the call site, making the test relative
 * rather than absolute.
 */
template<typename DataType>
SPARSEMAT_HD constexpr DataType singular_pivot_threshold() {
  return DataType(8) * std::numeric_limits<DataType>::epsilon();
}

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
