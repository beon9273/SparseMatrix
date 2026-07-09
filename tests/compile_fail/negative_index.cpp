// Must fail to compile: negative sparsity indices are invalid.
#include "sparsemat/api/sparsemat.h"
using namespace SparseLinearAlgebra;
int main() {
  SparseMat<double, int, 2, 2, -1> mat;
  (void)mat;
  return 0;
}
