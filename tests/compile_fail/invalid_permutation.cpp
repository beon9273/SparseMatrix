#include <array>

#include "sparsemat/api/sparsemat.h"

// A permutation with a repeated index would silently duplicate one row and
// drop another, producing a matrix that looks plausible — so it is rejected at
// compile time rather than accepted.
int main() {
  SparseLinearAlgebra::SparseMat<double, int, 3, 3, 0, 4, 8> a(1.0, 2.0, 3.0);
  constexpr std::array<int, 3> bad{0, 0, 2};
  auto permuted = a.symmetric_permute<bad>();
  return static_cast<int>(permuted.get(0, 0));
}
