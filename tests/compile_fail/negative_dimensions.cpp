// Must fail to compile: matrix dimensions must be positive.
#include "sparsemat/api/sparsemat.h"
using namespace SparseLinearAlgebra;
int main() {
  SparseMat<double, int, -2, 3, 0> mat;
  (void)mat;
  return 0;
}
