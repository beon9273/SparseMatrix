#include <iostream>

#include "sparsemat/api/sparsemat.h"

using namespace SparseLinearAlgebra;

int main() {
  // 3x3 diagonal matrix: non-zeros at flat indices 0, 4, 8
  SparseMat<double, 3, 3, 0, 4, 8> a(1.0, 2.0, 3.0);

  // 3x3 first-row matrix: non-zeros at flat indices 0, 1, 2
  SparseMat<double, 3, 3, 0, 1, 2> b(4.0, 5.0, 6.0);

  auto c = a.mult(b);

  std::cout << "A:\n";
  a.print();
  std::cout << "B:\n";
  b.print();
  std::cout << "A * B:\n";
  c.print();
}
