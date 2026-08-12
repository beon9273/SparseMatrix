// SPARSEMAT_MAX_NONZEROS puts a named ceiling on a matrix's stored-value count,
// so a runaway-density result fails with an actionable message rather than a
// slow build or an inscrutable compiler limit.
#define SPARSEMAT_MAX_NONZEROS 8
#include "sparsemat/api/sparsemat.h"

int main() {
  // Tridiagonal 6x6 has 16 stored values, over the ceiling of 8 set above.
  auto over_budget = SparseLinearAlgebra::tridiagonal<double, int, 6>();
  return static_cast<int>(over_budget.get(0, 0));
}
