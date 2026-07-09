// Must fail to compile: index 4 is out of bounds for a 2x2 (rows*cols == 4) matrix.
#include "sparsemat/api/sparsemat.h"
using namespace SparseLinearAlgebra;
int main() {
  SparseMat<double, int, 2, 2, 0, 4> mat;
  (void)mat;
  return 0;
}
