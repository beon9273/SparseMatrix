#pragma once

#include <array>
#include <concepts>
#include <cstddef>

namespace SparseLinearAlgebra {

/**
 * @brief Concept satisfied by scalar element types suitable for matrix storage.
 *
 * Requires the type to support arithmetic operations and be default-constructible.
 * Covers @c float, @c double, @c long @c double, and integral types.
 */
template<typename T>
concept MatrixDataType = std::is_arithmetic_v<T> && std::is_default_constructible_v<T>;

/**
 * @brief Base concept for sparse matrix types, excluding the @c make factory.
 *
 * Used as the return-type constraint on @c make itself to avoid a self-referential
 * definition in @c SparseMatrixType.
 */
template<typename T>
concept SparseMatrixBase =
    requires { typename T::DataType; } && MatrixDataType<typename T::DataType> && requires {
      { T::rows } -> std::convertible_to<std::size_t>;
      { T::cols } -> std::convertible_to<std::size_t>;
      { T::nonZeroCount } -> std::convertible_to<std::size_t>;
      { T::indices };
    } && requires(T mat, typename T::DataType value) {
      { mat.fill(value) };
      { mat.values };
      { mat.fill(value) };
    } && requires {
      { T::template make<1, 1, std::array<std::size_t, 0>{}>(std::index_sequence<>{}) };
    };

/**
 * @brief Concept satisfied by any @c SparseMat instantiation.
 *
 * Extends @c SparseMatrixBase with the requirement that the static @c make
 * factory exists and returns a type that is itself a @c SparseMatrixBase.
 * Note: Technically SparseMatrixType is required for return, but this is a circular
 * dependency. This covers 99% cases. Errors will be gross in cases where the .
 * SparseMatrixTypeBase returned from T::make does not have a make function
 * that returns a SparseMatrixType.
 */
template<typename T>
concept SparseMatrixType =
    SparseMatrixBase<T> &&
    // static factory used by all operations to construct result types
    requires {
      {
        T::template make<1, 1, std::array<std::size_t, 0>{}>(std::index_sequence<>{})
      } -> SparseMatrixBase;
    };

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
  { T::rows } -> std::convertible_to<std::size_t>;
  { T::cols } -> std::convertible_to<std::size_t>;
} && requires(std::size_t row, std::size_t col) {
  { T::is_result_index_nonzero(row, col) } -> std::convertible_to<bool>;
};

}  // namespace SparseLinearAlgebra
