#include "sparsemat/api/sparsemat.h"

// hcat joins matrices side by side, so their row counts must agree; a 3-row
// matrix and a 2-row one have no meaningful horizontal join.
int main() {
  SparseLinearAlgebra::SparseMat<double, int, 3, 2, 0, 3> a(1.0, 2.0);
  SparseLinearAlgebra::SparseMat<double, int, 2, 2, 0, 3> b(3.0, 4.0);
  auto joined = a.hcat(b);
  return static_cast<int>(joined.get(0, 0));
}
