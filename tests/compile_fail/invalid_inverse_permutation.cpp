#include <array>

#include "sparsemat/api/sparsemat.h"

// Inverting scatters through perm[i] as an index, so an out-of-range entry
// would write past the end of the result array. The ordering is a template
// parameter precisely so that this is caught at compile time instead.
int main() {
  constexpr std::array<int, 3> bad{0, 3, 1};
  constexpr auto inverse = SparseLinearAlgebra::inverse_permutation<bad>();
  return inverse[0];
}
