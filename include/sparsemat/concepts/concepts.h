#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

#if defined(__CUDACC__)
#define SPARSEMAT_HD __host__ __device__
#else
#define SPARSEMAT_HD
#endif

namespace SparseLinearAlgebra {

template<typename SparseMatrix>
class MatrixUtilities;

/**
 * @brief Concept satisfied by scalar element types suitable for matrix storage.
 *
 * Requires the type to be a floating point type and be default-constructible.
 * Covers @c float, @c double, and @c long @c double.
 */
template<typename T>
concept MatrixDataType = std::is_floating_point_v<T> && std::is_default_constructible_v<T>;

/**
 * @brief Base concept for sparse matrix types, excluding the @c Rebind check.
 *
 * Captures everything @c SparseMatrixType requires except the ability to
 * rebind to new dimensions. Kept separate so @c RebindType can verify that
 * @c Rebind actually produces a well-formed sparse matrix type without
 * recursing back into @c RebindType itself (which would recurse through
 * @c Rebind forever).
 */
template<typename T>
concept SparseMatrixBase =
    requires { typename T::DataType; } && MatrixDataType<typename T::DataType> &&
    requires { typename T::Int; } && std::is_signed_v<typename T::Int> && requires {
      { T::rows } -> std::convertible_to<typename T::Int>;
      { T::cols } -> std::convertible_to<typename T::Int>;
      { T::nonZeroCount } -> std::convertible_to<typename T::Int>;
      {
        T::indices()
      } -> std::convertible_to<
            std::array<typename T::Int, static_cast<typename T::Int>(T::nonZeroCount)>>;
    } && requires(T mat, typename T::DataType value) {
      { mat.fill(value) };
      { mat.values };
    };

/**
 * @brief Concept satisfied by types exposing a working @c Rebind alias.
 *
 * Checks not just that @c Rebind with some arbitrary dimensions and non-zero
 * counts compiles, but that the type it produces is itself a well-formed
 * sparse matrix type (@c SparseMatrixBase), not just any type.
 */
template<typename T>
concept RebindType =
    requires {
      typename T::DataType;
      typename T::Int;
    } &&
    requires {
      typename T::template Rebind<typename T::Int{3},
                                  typename T::Int{3},
                                  typename T::Int{2},
                                  typename T::Int{5},
                                  typename T::Int{7}>;
    } &&
    SparseMatrixBase<typename T::template Rebind<typename T::Int{3},
                                                 typename T::Int{3},
                                                 typename T::Int{2},
                                                 typename T::Int{5},
                                                 typename T::Int{7}>>;

/**
 * @brief Concept satisfied by full sparse matrix types.
 *
 * Combines @c SparseMatrixBase with @c RebindType so that both the type's own
 * shape and its ability to rebind to a new, well-formed sparse matrix type
 * are checked.
 */
template<typename T>
concept SparseMatrixType = SparseMatrixBase<T> && RebindType<T>;

/**
 * @brief Concept satisfied by the internal operation policy classes
 *        (e.g. @c Multiply, @c Add, @c Kronecker).
 *
 * Every operation must expose the result dimensions and a compile-time
 * predicate that decides whether a given output position is non-zero.
 * @c OperationUtilities drives sparsity computation through this interface.
 *
 * Note: @c num_nonzeros() and @c calculate_sparsity() are intentionally
 * excluded — they delegate to @c OperationUtilities, which itself requires
 * @c OperationType, and including them would create a circular constraint.
 */
template<typename T>
concept OperationType = requires {
  { T::rows } -> std::convertible_to<typename T::Int>;
  { T::cols } -> std::convertible_to<typename T::Int>;
} && requires(typename T::Int row, typename T::Int col) {
  { T::is_result_index_nonzero(row, col) } -> std::convertible_to<bool>;
};

namespace detail {

/// Checks that every row of @p T::sparsity() has the same length.
template<typename T>
constexpr bool is_rectangular_sparsity() {
  auto pattern = T::sparsity();
  auto cols = pattern[0].size();
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    if (pattern[i].size() != cols) {
      return false;
    }
  }
  return true;
}

/// Checks that every entry of @p T::sparsity() is either 0 or 1.
template<typename T>
constexpr bool is_zero_or_one_sparsity() {
  auto pattern = T::sparsity();
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    for (std::size_t j = 0; j < pattern[i].size(); ++j) {
      if (pattern[i][j] != 0 && pattern[i][j] != 1) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace detail

/**
 * @brief Concept satisfied by types describing a compile-time sparsity pattern.
 *
 * A sparsity pattern type exposes a @c static @c constexpr @c sparsity()
 * function returning a rectangular array-of-arrays of 0/1 entries: a 1 marks
 * a stored (non-zero) position, a 0 marks an empty one. @c SparseMatBuilder
 * uses this shape to derive a @c SparseMat's dimensions and flat non-zero
 * index list at compile time.
 */
template<typename T>
concept SparsityPatternType = requires {
  { T::sparsity() };
} && requires {
  requires(T::sparsity().size() > 0);
  requires(T::sparsity()[0].size() > 0);
  { T::sparsity()[0][0] } -> std::convertible_to<int>;
  requires detail::is_rectangular_sparsity<T>();
  requires detail::is_zero_or_one_sparsity<T>();
};

}  // namespace SparseLinearAlgebra
