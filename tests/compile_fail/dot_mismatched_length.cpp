// Must fail to compile: row and column vector lengths do not match for dot().
#include "sparsemat/api/sparsemat.h"
using namespace SparseLinearAlgebra;
int main() {
  SparseMat<double, int, 1, 3, 0, 1, 2> row(1.0, 2.0, 3.0);
  SparseMat<double, int, 4, 1, 0, 1, 2, 3> col(4.0, 5.0, 6.0, 7.0);
  auto bad = row.dot(col);
  (void)bad;
  return 0;
}
