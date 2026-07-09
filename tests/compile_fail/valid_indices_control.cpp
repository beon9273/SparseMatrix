// Control case: unique, in-bounds indices must compile fine. Guards against
// the compile-fail tests passing for the wrong reason (e.g. broken include
// paths making every compile-fail source fail).
#include "sparsemat/api/sparsemat.h"
using namespace SparseLinearAlgebra;
int main() {
  SparseMat<double, int, 2, 2, 0, 1, 2, 3> mat;
  (void)mat;
  return 0;
}
