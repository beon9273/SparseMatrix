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

/**
 * @def SPARSEMAT_MAX_NONZEROS
 * @brief Ceiling on the stored-value count of any single @c SparseMat, enforced
 *        by @c static_assert.
 *
 * Compile time and binary size are the real costs of encoding sparsity in the
 * type system, and both scale with how many non-zeros an operation's result
 * actually has. When a chain of operations quietly produces a much denser
 * result than intended — and multiply and Kronecker both can — the symptom is a
 * build that takes minutes, or a compiler that runs out of some internal
 * budget and reports something inscrutable about a non-type template argument
 * not being a constant expression. Neither points at the cause.
 *
 * This turns that into a named error naming the actual count, at the
 * instantiation that caused it. The default is deliberately generous: it should
 * catch runaway density, not ordinary use (a fully dense 64x64 fits). Lower it
 * to put a tighter budget on a particular translation unit, raise it if you
 * genuinely need larger results, or define it to 0 to disable the check:
 *
 * @code
 * // Fail the build if any single matrix exceeds 512 stored values.
 * #define SPARSEMAT_MAX_NONZEROS 512
 * #include "sparsemat/api/sparsemat.h"
 * @endcode
 */
// This has to be a macro rather than a constexpr constant: the whole point is
// that a consumer can override it by defining it before including this header.
#if !defined(SPARSEMAT_MAX_NONZEROS)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SPARSEMAT_MAX_NONZEROS 4096
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
 * @brief Concept satisfied when two sparse matrix types share a scalar type.
 *
 * Binary operations in this library do not promote mixed scalar types: the
 * result would have to pick one, and silently truncating a @c double operand
 * to a @c float result is exactly the kind of precision loss that is very hard
 * to notice after the fact. Mismatches are therefore rejected at compile time
 * (the only place they *can* be rejected — exceptions are not usable in the
 * @c SPARSEMAT_HD device code these operations must support). Convert
 * explicitly with @c convert<T>() to opt into the change.
 */
template<typename A, typename B>
concept SameDataType = std::is_same_v<typename A::DataType, typename B::DataType>;

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

}  // namespace SparseLinearAlgebra
