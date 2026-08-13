#include "sparsemat/api/sparsemat.h"

// Binary operations reject mixed scalar types rather than silently promoting
// or truncating; convert<T>() is the explicit opt-in.
int main() {
  SparseLinearAlgebra::SparseMat<float, int, 2, 2, 0, 3> f(1.0F, 2.0F);
  SparseLinearAlgebra::SparseMat<double, int, 2, 2, 0, 3> d(1.0, 2.0);
  auto sum = d.add(f);
  return static_cast<int>(sum.get(0, 0));
}
