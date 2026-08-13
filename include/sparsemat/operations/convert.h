#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "sparsemat/concepts/concepts.h"
#include "sparsemat/operations/utils.h"

namespace SparseLinearAlgebra::detail {

/**
 * @brief Implementation policy for changing a matrix's scalar type.
 *
 * The sparsity pattern is carried over unchanged — converting the scalar type
 * cannot create or destroy structural non-zeros — so this is a straight
 * element-wise @c static_cast over the stored values, with no sparsity
 * computation to do at all.
 *
 * @tparam NewDType  Target scalar type.
 * @tparam SparseMat Source matrix type.
 */
template<typename NewDType, SparseMatrixType SparseMat>
class Convert {
 public:
  using ResultType = typename SparseMat::template RebindData<NewDType>;

  template<std::size_t... Is>
  SPARSEMAT_HD static auto convert(const SparseMat& a, std::index_sequence<Is...> /*seq*/) {
    return ResultType(std::array<NewDType, sizeof...(Is)>{static_cast<NewDType>(a.values[Is])...});
  }

  SPARSEMAT_HD static auto convert(const SparseMat& a) {
    return convert(a,
                   std::make_index_sequence<static_cast<std::size_t>(SparseMat::nonZeroCount)>{});
  }
};

}  // namespace SparseLinearAlgebra::detail

namespace SparseLinearAlgebra {

/**
 * @brief Returns a copy of @p a with its scalar type changed to @p NewDType.
 *
 * Binary operations in this library reject mixed scalar types outright (see
 * the @c SameDataType concept) rather than silently promoting or truncating
 * one operand. This is the explicit opt-in: convert first, then operate.
 *
 * @code
 * SparseMat<float, int, 3, 3, 0, 4, 8>  f(1, 2, 3);
 * SparseMat<double, int, 3, 3, 0, 4, 8> d(1, 2, 3);
 * // auto bad = f.add(d);                  // static_assert: mixed DataType
 * auto good = SparseLinearAlgebra::convert<double>(f).add(d);
 * @endcode
 *
 * Each stored value is @c static_cast individually, so narrowing conversions
 * (@c double to @c float) round exactly as a plain cast would; the sparsity
 * pattern, dimensions, and index type are all preserved.
 *
 * @tparam NewDType Target scalar element type.
 * @tparam A        Source matrix type.
 * @param  a        Matrix to convert.
 * @return          The same matrix with @c DataType == @p NewDType.
 */
template<typename NewDType, SparseMatrixType A>
SPARSEMAT_HD auto convert(const A& a) {
  static_assert(MatrixDataType<NewDType>,
                "convert<T>() target must be a floating point scalar type.");
  return detail::Convert<NewDType, A>::convert(a);
}

}  // namespace SparseLinearAlgebra
