
#include "api/sparsemat.h"

int main() {
  SparseMat<double, 3, 3, 0, 4, 8> mat(1, 2, 3);
  SparseMat<double, 3, 3, 0, 1, 2> mat2(4, 5, 6);
  auto m3 = mat.mult(mat2);
  auto m4 = mat.add(mat2);
  auto m5 = mat.subtract(mat2);
  auto m6 = mat2.transpose();

  std::cout << "Matrix F (Transpose of B):" << '\n';
  m6.print();
  std::cout << "Matrix A:" << '\n';
  mat.print();
  std::cout << "Matrix B:" << '\n';
  mat2.print();
  std::cout << "Matrix C (A * B):" << '\n';
  m3.print();
  std::cout << "Matrix D (A + B):" << '\n';
  m4.print();
  std::cout << "Matrix E (A - B):" << '\n';
  m5.print();
  return 0;
}
