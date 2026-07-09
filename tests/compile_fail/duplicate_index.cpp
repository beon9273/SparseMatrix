// Must fail to compile: index 0 appears twice.
#include "sparsemat/api/sparsemat.h"
using namespace SparseLinearAlgebra;
int main() {
  SparseMat<double, int, 2, 2, 0, 0> mat;
  (void)mat;
  return 0;
}
