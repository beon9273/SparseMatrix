#include "sparsemat/api/sparsemat.h"

// operator+= cannot widen the sparsity pattern: the pattern is part of the
// type, so the left operand has to keep its own.
int main() {
  SparseLinearAlgebra::SparseMat<double, int, 2, 2, 0, 3> a(1.0, 2.0);
  SparseLinearAlgebra::SparseMat<double, int, 2, 2, 0, 1, 3> b(1.0, 2.0, 3.0);
  a += b;
  return static_cast<int>(a.get(0, 0));
}
