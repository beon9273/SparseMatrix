#include <cmath>
#include <iostream>
#include <string>

#include "sparsemat_dist.h"

// --- Minimal test harness ---

static int passed = 0;
static int failed = 0;

void check(bool condition, const std::string& name) {
  if (condition) {
    std::cout << "[PASS] " << name << "\n";
    ++passed;
  } else {
    std::cout << "[FAIL] " << name << "\n";
    ++failed;
  }
}

template<typename T>
void check_eq(T a, T b, const std::string& name) {
  check(a == b, name);
}

void check_near(double a, double b, const std::string& name, double eps = 1e-9) {
  check(std::abs(a - b) < eps, name);
}

// --- Tests ---

void test_construction() {
  SparseMat<double, 3, 3, 0, 4, 8> mat(1, 2, 3);
  check_eq(mat.values[0], 1.0, "construction: values[0]");
  check_eq(mat.values[1], 2.0, "construction: values[1]");
  check_eq(mat.values[2], 3.0, "construction: values[2]");
  check_eq(decltype(mat)::nonZeroCount, std::size_t(3), "construction: nonZeroCount");
}

void test_get() {
  // A = diag(1,2,3)
  SparseMat<double, 3, 3, 0, 4, 8> mat(1, 2, 3);
  check_eq(mat.template get<0, 0>(), 1.0, "get: (0,0)");
  check_eq(mat.template get<1, 1>(), 2.0, "get: (1,1)");
  check_eq(mat.template get<2, 2>(), 3.0, "get: (2,2)");
  check_eq(mat.template get<0, 1>(), 0.0, "get: zero element (0,1)");
  check_eq(mat.get(1, 1), 2.0, "get runtime: (1,1)");
  check_eq(mat.get(0, 1), 0.0, "get runtime: zero element (0,1)");
}

void test_set() {
  SparseMat<double, 3, 3, 0, 4, 8> mat(1, 2, 3);
  mat.template set<0, 0>(10.0);
  check_eq(mat.template get<0, 0>(), 10.0, "set: (0,0)");

  bool ok = mat.set(1, 1, 20.0);
  check(ok, "set runtime: returns true for nonzero");
  check_eq(mat.get(1, 1), 20.0, "set runtime: (1,1)");

  bool fail = mat.set(0, 1, 99.0);
  check(!fail, "set runtime: returns false for zero index");
  check_eq(mat.get(0, 1), 0.0, "set runtime: zero index unchanged");
}

void test_fill() {
  SparseMat<double, 3, 3, 0, 4, 8> mat;
  mat.fill(5.0);
  check_eq(mat.values[0], 5.0, "fill: values[0]");
  check_eq(mat.values[1], 5.0, "fill: values[1]");
  check_eq(mat.values[2], 5.0, "fill: values[2]");
}

void test_multiply() {
  // A = diag(1,2,3), B = first row [4,5,6]
  // A*B = [[4,5,6],[0,0,0],[0,0,0]]
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto c = a.mult(b);
  check_eq(c.template get<0, 0>(), 4.0, "multiply: (0,0)");
  check_eq(c.template get<0, 1>(), 5.0, "multiply: (0,1)");
  check_eq(c.template get<0, 2>(), 6.0, "multiply: (0,2)");
  check_eq(c.template get<1, 0>(), 0.0, "multiply: (1,0) zero");
  check_eq(decltype(c)::nonZeroCount, std::size_t(3), "multiply: result nonZeroCount");
}

void test_add() {
  // A = diag(1,2,3), B = first row [4,5,6]
  // A+B: (0,0)=5, (0,1)=5, (0,2)=6, (1,1)=2, (2,2)=3
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto c = a.add(b);
  check_eq(c.template get<0, 0>(), 5.0, "add: (0,0)");
  check_eq(c.template get<0, 1>(), 5.0, "add: (0,1)");
  check_eq(c.template get<0, 2>(), 6.0, "add: (0,2)");
  check_eq(c.template get<1, 1>(), 2.0, "add: (1,1)");
  check_eq(c.template get<2, 2>(), 3.0, "add: (2,2)");
  check_eq(decltype(c)::nonZeroCount, std::size_t(5), "add: result nonZeroCount");
}

void test_subtract() {
  // A-B: (0,0)=-3, (0,1)=-5, (0,2)=-6, (1,1)=2, (2,2)=3
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto c = a.subtract(b);
  check_eq(c.template get<0, 0>(), -3.0, "subtract: (0,0)");
  check_eq(c.template get<0, 1>(), -5.0, "subtract: (0,1)");
  check_eq(c.template get<0, 2>(), -6.0, "subtract: (0,2)");
  check_eq(c.template get<1, 1>(), 2.0, "subtract: (1,1)");
  check_eq(c.template get<2, 2>(), 3.0, "subtract: (2,2)");
}

void test_transpose() {
  // B = first row [4,5,6] → B^T = first col
  SparseMat<double, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto bt = b.transpose();
  check_eq(bt.template get<0, 0>(), 4.0, "transpose: (0,0)");
  check_eq(bt.template get<1, 0>(), 5.0, "transpose: (1,0)");
  check_eq(bt.template get<2, 0>(), 6.0, "transpose: (2,0)");
  check_eq(bt.template get<0, 1>(), 0.0, "transpose: zero (0,1)");
  check_eq(decltype(bt)::nonZeroCount, std::size_t(3), "transpose: nonZeroCount preserved");
}

void test_transpose_nonsquare() {
  // 4x2 matrix (rows > cols) transposed to 2x4.
  // Nonzeros at (0,0)=flat 0 and (1,1)=flat 3 in the 4x2 grid.
  // Transpose swaps to (0,0)=flat 0 and (1,1)=flat 5 in the 2x4 grid.
  SparseMat<double, 4, 2, 0, 3> a(3.0, 7.0);
  auto at = a.transpose();
  check_eq(decltype(at)::rows, std::size_t(2), "transpose nonsquare: result rows");
  check_eq(decltype(at)::cols, std::size_t(4), "transpose nonsquare: result cols");
  check_eq(decltype(at)::nonZeroCount,
           std::size_t(2),
           "transpose nonsquare: nonZeroCount preserved");
  check_eq(at.template get<0, 0>(), 3.0, "transpose nonsquare: (0,0)");
  check_eq(at.template get<1, 1>(), 7.0, "transpose nonsquare: (1,1)");
  check_eq(at.template get<0, 1>(), 0.0, "transpose nonsquare: zero (0,1)");
  check_eq(at.template get<1, 0>(), 0.0, "transpose nonsquare: zero (1,0)");
  check_eq(at.template get<0, 2>(), 0.0, "transpose nonsquare: zero (0,2)");
  check_eq(at.template get<0, 3>(), 0.0, "transpose nonsquare: zero (0,3)");

  // 4x3 matrix (rows > cols) transposed to 3x4.
  // Nonzeros at (0,0)=flat 0, (1,1)=flat 4, (2,2)=flat 8 in a 4x3 grid.
  // All source nonzeros are in rows 0..2 (< Source.cols=3), so all transposed
  // positions land in result columns 0..2 (< Result.rows=3) and are reachable.
  SparseMat<double, 4, 3, 0, 4, 8> b(4.0, 5.0, 6.0);
  auto bt = b.transpose();
  check_eq(decltype(bt)::rows, std::size_t(3), "transpose 4x3: result rows");
  check_eq(decltype(bt)::cols, std::size_t(4), "transpose 4x3: result cols");
  check_eq(decltype(bt)::nonZeroCount, std::size_t(3), "transpose 4x3: nonZeroCount preserved");
  check_eq(bt.template get<0, 0>(), 4.0, "transpose 4x3: (0,0)");
  check_eq(bt.template get<1, 1>(), 5.0, "transpose 4x3: (1,1)");
  check_eq(bt.template get<2, 2>(), 6.0, "transpose 4x3: (2,2)");
  check_eq(bt.template get<0, 1>(), 0.0, "transpose 4x3: zero (0,1)");
  check_eq(bt.template get<0, 3>(), 0.0, "transpose 4x3: zero (0,3)");

  // Regression: 2x4 → 4x2 (rows < cols, previously hit a static_assert).
  SparseMat<double, 2, 4, 0, 5> h(3.0, 4.0);
  auto ht = h.transpose();
  check_eq(decltype(ht)::rows, std::size_t(4), "transpose 2x4: result rows");
  check_eq(decltype(ht)::cols, std::size_t(2), "transpose 2x4: result cols");
  check_eq(ht.template get<0, 0>(), 3.0, "transpose 2x4: (0,0)");
  check_eq(ht.template get<1, 1>(), 4.0, "transpose 2x4: (1,1)");
  check_eq(ht.template get<0, 1>(), 0.0, "transpose 2x4: zero (0,1)");
  check_eq(ht.template get<1, 0>(), 0.0, "transpose 2x4: zero (1,0)");

  // Regression: 3x1 → 1x3 (column vector to row vector, previously silently dropped values).
  SparseMat<double, 3, 1, 0, 1, 2> col(1.0, 2.0, 3.0);
  auto row = col.transpose();
  check_eq(decltype(row)::rows, std::size_t(1), "transpose 3x1: result rows");
  check_eq(decltype(row)::cols, std::size_t(3), "transpose 3x1: result cols");
  check_eq(row.template get<0, 0>(), 1.0, "transpose 3x1: (0,0)");
  check_eq(row.template get<0, 1>(), 2.0, "transpose 3x1: (0,1)");
  check_eq(row.template get<0, 2>(), 3.0, "transpose 3x1: (0,2)");

  // Regression: 6x3 with a nonzero at source row 5 (previously silently dropped).
  SparseMat<double, 6, 3, 0, 7, 17> c(4.0, 5.0, 6.0);
  auto ct = c.transpose();
  check_eq(decltype(ct)::rows, std::size_t(3), "transpose 6x3: result rows");
  check_eq(decltype(ct)::cols, std::size_t(6), "transpose 6x3: result cols");
  check_eq(ct.template get<0, 0>(), 4.0, "transpose 6x3: (0,0)");
  check_eq(ct.template get<1, 2>(), 5.0, "transpose 6x3: (1,2)");
  check_eq(ct.template get<2, 5>(), 6.0, "transpose 6x3: (2,5)");
}

void test_scale() {
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  auto b = a.scale(2.0);
  check_eq(b.template get<0, 0>(), 2.0, "scale: (0,0)");
  check_eq(b.template get<1, 1>(), 4.0, "scale: (1,1)");
  check_eq(b.template get<2, 2>(), 6.0, "scale: (2,2)");
  // original unchanged
  check_eq(a.template get<0, 0>(), 1.0, "scale: original unchanged");
}

void test_scale_inplace() {
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  a.scale_inplace(3.0);
  check_eq(a.template get<0, 0>(), 3.0, "scale_inplace: (0,0)");
  check_eq(a.template get<1, 1>(), 6.0, "scale_inplace: (1,1)");
  check_eq(a.template get<2, 2>(), 9.0, "scale_inplace: (2,2)");
}

void test_hadamard() {
  // A = diag(1,2,3), B = first row [4,5,6]
  // intersection of nonzeros: only (0,0) is nonzero in both
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto c = a.hadamard(b);
  check_eq(c.template get<0, 0>(), 4.0, "hadamard: (0,0)");
  check_eq(decltype(c)::nonZeroCount,
           std::size_t(1),
           "hadamard: result nonZeroCount (intersection)");
}

void test_frobenius() {
  // A = diag(1,2,3), ||A||_F = sqrt(1+4+9) = sqrt(14)
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  check_near(a.frobenius(), std::sqrt(14.0), "frobenius norm");
}

void test_normalize() {
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  auto n = a.normalize();
  check_near(n.frobenius(), 1.0, "normalize: frobenius norm = 1");
}

void test_normalize_inplace() {
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  a.normalize_inplace();
  check_near(a.frobenius(), 1.0, "normalize_inplace: frobenius norm = 1");
}

void test_dense() {
  // A = diag(1,2,3)
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  auto d = a.dense();
  check_eq(d.get<0, 0>(), 1.0, "dense: index 0");
  check_eq(d.get<1, 1>(), 2.0, "dense: index 4");
  check_eq(d.get<2, 2>(), 3.0, "dense: index 8");
  check_eq(d.get<0, 1>(), 0.0, "dense: zero element");
}

void test_trace() {
  // A = diag(1,2,3), trace = 6
  SparseMat<double, 3, 3, 0, 4, 8> a(1, 2, 3);
  check_eq(SparseLinearAlgebra::trace(a), 6.0, "trace: diag(1,2,3)");
}

void test_dot() {
  // row vector [1,2,3] · col vector [4,5,6] = 32
  SparseMat<double, 1, 3, 0, 1, 2> row(1, 2, 3);
  SparseMat<double, 3, 1, 0, 1, 2> col(4, 5, 6);
  check_eq(row.dot(col), 32.0, "dot: [1,2,3]·[4,5,6]");
}

void test_identity() {
  auto I = SparseMat<double, 3, 3, 0>::template identity<3>();
  check_eq(I.template get<0, 0>(), 1.0, "identity: (0,0)");
  check_eq(I.template get<1, 1>(), 1.0, "identity: (1,1)");
  check_eq(I.template get<2, 2>(), 1.0, "identity: (2,2)");
  check_eq(I.template get<0, 1>(), 0.0, "identity: off-diagonal zero");
  check_eq(decltype(I)::nonZeroCount, std::size_t(3), "identity: nonZeroCount");
}

void test_kronecker() {
  // A = diag(1, 2)  (2x2), B = [[3,4],[5,6]]  (2x2 full)
  // A ⊗ B = [[3,4,0,0],[5,6,0,0],[0,0,6,8],[0,0,10,12]]
  SparseMat<double, 2, 2, 0, 3> A(1, 2);
  SparseMat<double, 2, 2, 0, 1, 2, 3> B(3, 4, 5, 6);
  auto C = A.kronecker(B);

  check_eq(decltype(C)::rows, std::size_t(4), "kronecker: result rows");
  check_eq(decltype(C)::cols, std::size_t(4), "kronecker: result cols");
  check_eq(decltype(C)::nonZeroCount, std::size_t(8), "kronecker: nonZeroCount");

  // top-left 2x2 block = 1*B
  check_eq(C.get(0, 0), 3.0, "kronecker: (0,0)");
  check_eq(C.get(0, 1), 4.0, "kronecker: (0,1)");
  check_eq(C.get(1, 0), 5.0, "kronecker: (1,0)");
  check_eq(C.get(1, 1), 6.0, "kronecker: (1,1)");

  // bottom-right 2x2 block = 2*B
  check_eq(C.get(2, 2), 6.0, "kronecker: (2,2)");
  check_eq(C.get(2, 3), 8.0, "kronecker: (2,3)");
  check_eq(C.get(3, 2), 10.0, "kronecker: (3,2)");
  check_eq(C.get(3, 3), 12.0, "kronecker: (3,3)");

  // off-diagonal blocks are zero
  check_eq(C.get(0, 2), 0.0, "kronecker: off-block zero (0,2)");
  check_eq(C.get(2, 0), 0.0, "kronecker: off-block zero (2,0)");

  // sparsity — result should only store the 8 non-zeros
  check_eq(decltype(C)::nonZeroCount, std::size_t(8), "kronecker: nonZeroCount");
}

void test_triangular() {
  // Lower triangular: non-zeros only on and below diagonal
  // 3x3: indices 0(0,0), 3(1,0), 4(1,1), 6(2,0), 7(2,1), 8(2,2)
  using Lower = SparseMat<double, 3, 3, 0, 3, 4, 6, 7, 8>;
  Lower lower(1, 2, 3, 4, 5, 6);
  check(Lower::is_structurally_lower_triangular(), "triangular: lower is structurally lower");
  check(!Lower::is_structurally_upper_triangular(), "triangular: lower is not structurally upper");
  check(lower.is_numerically_lower_triangular(), "triangular: lower is numerically lower");
  check(!lower.is_numerically_upper_triangular(), "triangular: lower is not numerically upper");

  // Upper triangular: non-zeros only on and above diagonal
  // 3x3: indices 0(0,0), 1(0,1), 2(0,2), 4(1,1), 5(1,2), 8(2,2)
  using Upper = SparseMat<double, 3, 3, 0, 1, 2, 4, 5, 8>;
  Upper upper(1, 2, 3, 4, 5, 6);
  check(!Upper::is_structurally_lower_triangular(), "triangular: upper is not structurally lower");
  check(Upper::is_structurally_upper_triangular(), "triangular: upper is structurally upper");
  check(!upper.is_numerically_lower_triangular(), "triangular: upper is not numerically lower");
  check(upper.is_numerically_upper_triangular(), "triangular: upper is numerically upper");

  // Diagonal: both lower and upper triangular
  using Diag = SparseMat<double, 3, 3, 0, 4, 8>;
  Diag diag(1, 2, 3);
  check(Diag::is_structurally_lower_triangular(), "triangular: diagonal is structurally lower");
  check(Diag::is_structurally_upper_triangular(), "triangular: diagonal is structurally upper");

  // Structurally upper but numerically not: above-diagonal value is non-zero
  // 3x3: non-zeros at 0(0,0), 1(0,1), 4(1,1), 8(2,2) — has above-diagonal entry
  using Mixed = SparseMat<double, 3, 3, 0, 1, 4, 8>;
  Mixed mixed(1, 2, 3, 4);
  // (0,1) is above diagonal → structurally not lower
  check(!Mixed::is_structurally_lower_triangular(), "triangular: mixed is not structurally lower");
  // numerically lower: set above-diagonal value to near-zero
  mixed.template set<0, 1>(0.0);
  check(mixed.is_numerically_lower_triangular(),
        "triangular: mixed is numerically lower when above-diagonal is zero");
}

void test_forward_solve() {
  // L = [[2,0,0],[1,3,0],[4,5,6]], b = [4,11,47]
  // x[0] = 4/2 = 2
  // x[1] = (11 - 1*2)/3 = 3
  // x[2] = (47 - 4*2 - 5*3)/6 = 4
  // indices: (0,0)=0, (1,0)=3, (1,1)=4, (2,0)=6, (2,1)=7, (2,2)=8
  SparseMat<double, 3, 3, 0, 3, 4, 6, 7, 8> L(2, 1, 3, 4, 5, 6);
  SparseMat<double, 3, 1, 0, 1, 2> b(4, 11, 47);

  auto x = SparseLinearAlgebra::forward_solve(L, b);

  check_near(x.get(0, 0), 2.0, "forward_solve: x[0]");
  check_near(x.get(1, 0), 3.0, "forward_solve: x[1]");
  check_near(x.get(2, 0), 4.0, "forward_solve: x[2]");

  // Verify Lx = b
  auto residual = L.mult(x);
  check_near(residual.get(0, 0), 4.0, "forward_solve: residual b[0]");
  check_near(residual.get(1, 0), 11.0, "forward_solve: residual b[1]");
  check_near(residual.get(2, 0), 47.0, "forward_solve: residual b[2]");
}

void test_backward_solve() {
  // U = [[2,1,4],[0,3,5],[0,0,6]], b = [23,29,24]
  // x[2] = 24/6 = 4
  // x[1] = (29 - 5*4)/3 = 3
  // x[0] = (23 - 1*3 - 4*4)/2 = 2
  // indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,1)=4, (1,2)=5, (2,2)=8
  SparseMat<double, 3, 3, 0, 1, 2, 4, 5, 8> U(2, 1, 4, 3, 5, 6);
  SparseMat<double, 3, 1, 0, 1, 2> b(23, 29, 24);

  auto x = SparseLinearAlgebra::backward_solve(U, b);

  check_near(x.get(0, 0), 2.0, "backward_solve: x[0]");
  check_near(x.get(1, 0), 3.0, "backward_solve: x[1]");
  check_near(x.get(2, 0), 4.0, "backward_solve: x[2]");

  // Verify Ux = b
  auto residual = U.mult(x);
  check_near(residual.get(0, 0), 23.0, "backward_solve: residual b[0]");
  check_near(residual.get(1, 0), 29.0, "backward_solve: residual b[1]");
  check_near(residual.get(2, 0), 24.0, "backward_solve: residual b[2]");
}

void test_lu_solve() {
  // A = [[2,1,1],[4,3,3],[8,7,9]], b = [1,1,1], x = [1,-1,1] via LU
  // Flat indices (3x3): (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5,(2,0)=6,(2,1)=7,(2,2)=8
  SparseMat<double, 3, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8> A(2, 1, 1, 4, 3, 3, 8, 7, 9);
  SparseMat<double, 3, 1, 0, 1, 2> b(1, 1, 1);

  auto x = A.solve(b);

  check_near(x.get(0, 0), 1.0, "lu_solve: x[0]");
  check_near(x.get(1, 0), -1.0, "lu_solve: x[1]");
  check_near(x.get(2, 0), 0.0, "lu_solve: x[2]");

  // Verify Ax = b
  auto residual = A.mult(x);
  check_near(residual.get(0, 0), 1.0, "lu_solve: residual b[0]");
  check_near(residual.get(1, 0), 1.0, "lu_solve: residual b[1]");
  check_near(residual.get(2, 0), 1.0, "lu_solve: residual b[2]");
}

void test_set_diagonal_scalar() {
  // Full diagonal matrix: all three diagonal entries are stored.
  SparseMat<double, 3, 3, 0, 4, 8> diag(1, 2, 3);
  diag.set_diagonal(7.0);
  check_eq(diag.template get<0, 0>(), 7.0, "set_diagonal scalar: (0,0)");
  check_eq(diag.template get<1, 1>(), 7.0, "set_diagonal scalar: (1,1)");
  check_eq(diag.template get<2, 2>(), 7.0, "set_diagonal scalar: (2,2)");

  // General matrix: off-diagonal stored values must be left untouched.
  // indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1), 8=(2,2)
  SparseMat<double, 3, 3, 0, 1, 3, 4, 8> mat(1, 2, 3, 4, 5);
  mat.set_diagonal(9.0);
  check_eq(mat.template get<0, 0>(), 9.0, "set_diagonal scalar general: (0,0)");
  check_eq(mat.template get<1, 1>(), 9.0, "set_diagonal scalar general: (1,1)");
  check_eq(mat.template get<2, 2>(), 9.0, "set_diagonal scalar general: (2,2)");
  check_eq(mat.template get<0, 1>(), 2.0, "set_diagonal scalar general: off-diag (0,1) unchanged");
  check_eq(mat.template get<1, 0>(), 3.0, "set_diagonal scalar general: off-diag (1,0) unchanged");

  // Matrix where (1,1) is structurally zero: set_diagonal must not touch it.
  // indices: 0=(0,0), 1=(0,1), 3=(1,0), 8=(2,2)
  SparseMat<double, 3, 3, 0, 1, 3, 8> missing(1, 2, 3, 4);
  missing.set_diagonal(5.0);
  check_eq(missing.template get<0, 0>(), 5.0, "set_diagonal scalar missing: (0,0)");
  check_eq(missing.template get<2, 2>(), 5.0, "set_diagonal scalar missing: (2,2)");
  check_eq(missing.template get<1, 1>(), 0.0, "set_diagonal scalar missing: (1,1) stays zero");
  check_eq(missing.template get<0, 1>(),
           2.0,
           "set_diagonal scalar missing: off-diag (0,1) unchanged");
}

void test_set_diagonal_span() {
  // Full diagonal: span values map to (0,0), (1,1), (2,2) in order.
  SparseMat<double, 3, 3, 0, 4, 8> diag(1, 2, 3);
  std::array<double, 3> v1{10.0, 20.0, 30.0};
  diag.set_diagonal(std::span<double>(v1));
  check_eq(diag.template get<0, 0>(), 10.0, "set_diagonal span full: (0,0)");
  check_eq(diag.template get<1, 1>(), 20.0, "set_diagonal span full: (1,1)");
  check_eq(diag.template get<2, 2>(), 30.0, "set_diagonal span full: (2,2)");

  // General matrix: off-diagonal stored values must not be overwritten.
  // indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1), 8=(2,2)
  SparseMat<double, 3, 3, 0, 1, 3, 4, 8> mat(1, 2, 3, 4, 5);
  std::array<double, 3> v2{7.0, 8.0, 9.0};
  mat.set_diagonal(std::span<double>(v2));
  check_eq(mat.template get<0, 0>(), 7.0, "set_diagonal span general: (0,0)");
  check_eq(mat.template get<1, 1>(), 8.0, "set_diagonal span general: (1,1)");
  check_eq(mat.template get<2, 2>(), 9.0, "set_diagonal span general: (2,2)");
  check_eq(mat.template get<0, 1>(), 2.0, "set_diagonal span general: off-diag (0,1) unchanged");
  check_eq(mat.template get<1, 0>(), 3.0, "set_diagonal span general: off-diag (1,0) unchanged");

  // Structurally missing (1,1): span of size 2 covers only the two stored diagonals.
  // (1,1) is skipped without consuming a span entry.
  // indices: 0=(0,0), 1=(0,1), 3=(1,0), 8=(2,2)
  SparseMat<double, 3, 3, 0, 1, 3, 8> missing(1, 2, 3, 4);
  std::array<double, 2> v3{11.0, 33.0};
  missing.set_diagonal(std::span<double>(v3));
  check_eq(missing.template get<0, 0>(), 11.0, "set_diagonal span missing: (0,0)");
  check_eq(missing.template get<2, 2>(), 33.0, "set_diagonal span missing: (2,2)");
  check_eq(missing.template get<1, 1>(), 0.0, "set_diagonal span missing: (1,1) stays zero");
  check_eq(missing.template get<0, 1>(),
           2.0,
           "set_diagonal span missing: off-diag (0,1) unchanged");
}

void test_invert2x2() {
  // Full 2x2: [2 1; 1 3] → det=5, inv=[3/5 -1/5; -1/5 2/5]
  // Use values whose inverse is exact in IEEE 754 doubles via /5.
  SparseMat<double, 2, 2, 0, 1, 2, 3> a(2.0, 1.0, 1.0, 3.0);
  auto ai = a.invert2x2();
  check_near(ai.template get<0, 0>(), 3.0 / 5.0, "invert2x2 full: (0,0)");
  check_near(ai.template get<0, 1>(), -1.0 / 5.0, "invert2x2 full: (0,1)");
  check_near(ai.template get<1, 0>(), -1.0 / 5.0, "invert2x2 full: (1,0)");
  check_near(ai.template get<1, 1>(), 2.0 / 5.0, "invert2x2 full: (1,1)");

  // Verify A * A^{-1} ≈ I.
  auto id = a.mult(ai);
  check_near(id.template get<0, 0>(), 1.0, "invert2x2 full: A*inv (0,0)");
  check_near(id.template get<1, 1>(), 1.0, "invert2x2 full: A*inv (1,1)");
  check_near(id.template get<0, 1>(), 0.0, "invert2x2 full: A*inv (0,1)");
  check_near(id.template get<1, 0>(), 0.0, "invert2x2 full: A*inv (1,0)");

  // Diagonal 2x2: [4 0; 0 2] → inv=[1/4 0; 0 1/2]
  SparseMat<double, 2, 2, 0, 3> d(4.0, 2.0);
  auto di = d.invert2x2();
  check_eq(di.template get<0, 0>(), 0.25, "invert2x2 diag: (0,0)");
  check_eq(di.template get<1, 1>(), 0.50, "invert2x2 diag: (1,1)");
  // off-diagonal slots are structurally zero — verify via dense
  auto di_dense = di.dense();
  check_eq(di_dense.template get<0, 1>(), 0.0, "invert2x2 diag: (0,1) zero");
  check_eq(di_dense.template get<1, 0>(), 0.0, "invert2x2 diag: (1,0) zero");
}

void test_block_triangular_solve() {
  // forward_solve with 2-column RHS
  // L = [2 0 0; 1 3 0; 2 1 4],  L * [1 2; 1 1; 3 2] = [2 4; 4 5; 15 13]
  SparseMat<double, 3, 3, 0, 3, 4, 6, 7, 8> L(2.0, 1.0, 3.0, 2.0, 1.0, 4.0);
  SparseMat<double, 3, 2, 0, 1, 2, 3, 4, 5> B(2.0, 4.0, 4.0, 5.0, 15.0, 13.0);
  auto X = SparseLinearAlgebra::forward_solve(L, B);
  check_near(X.template get<0, 0>(), 1.0, "block fwd: x[0][0]");
  check_near(X.template get<1, 0>(), 1.0, "block fwd: x[1][0]");
  check_near(X.template get<2, 0>(), 3.0, "block fwd: x[2][0]");
  check_near(X.template get<0, 1>(), 2.0, "block fwd: x[0][1]");
  check_near(X.template get<1, 1>(), 1.0, "block fwd: x[1][1]");
  check_near(X.template get<2, 1>(), 2.0, "block fwd: x[2][1]");

  // backward_solve with 2-column RHS
  // U = [4 1 2; 0 3 1; 0 0 2],  U * [1 2; 1 1; 3 3] = [11 15; 6 6; 6 6]
  SparseMat<double, 3, 3, 0, 1, 2, 4, 5, 8> U(4.0, 1.0, 2.0, 3.0, 1.0, 2.0);
  SparseMat<double, 3, 2, 0, 1, 2, 3, 4, 5> Brhs(11.0, 15.0, 6.0, 6.0, 6.0, 6.0);
  auto Y = SparseLinearAlgebra::backward_solve(U, Brhs);
  check_near(Y.template get<0, 0>(), 1.0, "block bwd: y[0][0]");
  check_near(Y.template get<1, 0>(), 1.0, "block bwd: y[1][0]");
  check_near(Y.template get<2, 0>(), 3.0, "block bwd: y[2][0]");
  check_near(Y.template get<0, 1>(), 2.0, "block bwd: y[0][1]");
  check_near(Y.template get<1, 1>(), 1.0, "block bwd: y[1][1]");
  check_near(Y.template get<2, 1>(), 3.0, "block bwd: y[2][1]");
}

void test_cholesky() {
  // Diagonal 2x2 SPD: A = diag(4, 9) → L = diag(2, 3)
  SparseMat<double, 2, 2, 0, 3> A2(4.0, 9.0);
  auto L2 = SparseLinearAlgebra::cholesky_factorize(A2);
  check_near(L2.template get<0, 0>(), 2.0, "cholesky diag 2x2: L[0][0]");
  check_near(L2.template get<1, 1>(), 3.0, "cholesky diag 2x2: L[1][1]");

  // Full 3x3 SPD: A = [4 2 0; 2 5 2; 0 2 5] → L = [2 0 0; 1 2 0; 0 1 2]
  SparseMat<double, 3, 3, 0, 1, 3, 4, 5, 7, 8> A3(4.0, 2.0, 2.0, 5.0, 2.0, 2.0, 5.0);
  auto L3 = SparseLinearAlgebra::cholesky_factorize(A3);
  check_near(L3.template get<0, 0>(), 2.0, "cholesky 3x3: L[0][0]");
  check_near(L3.template get<1, 0>(), 1.0, "cholesky 3x3: L[1][0]");
  check_near(L3.template get<1, 1>(), 2.0, "cholesky 3x3: L[1][1]");
  check_near(L3.template get<2, 0>(), 0.0, "cholesky 3x3: L[2][0] structurally zero");
  check_near(L3.template get<2, 1>(), 1.0, "cholesky 3x3: L[2][1]");
  check_near(L3.template get<2, 2>(), 2.0, "cholesky 3x3: L[2][2]");

  // Verify L * L^T ≈ A
  auto LLT = L3.mult(L3.transpose()).dense();
  check_near(LLT.template get<0, 0>(), 4.0, "cholesky 3x3 LLT: (0,0)");
  check_near(LLT.template get<0, 1>(), 2.0, "cholesky 3x3 LLT: (0,1)");
  check_near(LLT.template get<1, 1>(), 5.0, "cholesky 3x3 LLT: (1,1)");
  check_near(LLT.template get<1, 2>(), 2.0, "cholesky 3x3 LLT: (1,2)");
  check_near(LLT.template get<2, 2>(), 5.0, "cholesky 3x3 LLT: (2,2)");
  check_near(LLT.template get<0, 2>(), 0.0, "cholesky 3x3 LLT: (0,2) zero");

  // Solve A*x = b: A*[1;0;1] = [4;4;5]
  SparseMat<double, 3, 1, 0, 1, 2> b(4.0, 4.0, 5.0);
  auto x = SparseLinearAlgebra::cholesky_solve(A3, b);
  check_near(x.template get<0, 0>(), 1.0, "cholesky solve: x[0]");
  check_near(x.template get<1, 0>(), 0.0, "cholesky solve: x[1]");
  check_near(x.template get<2, 0>(), 1.0, "cholesky solve: x[2]");
}

// --- Main ---

int main() {
  test_construction();
  test_get();
  test_set();
  test_fill();
  test_multiply();
  test_add();
  test_subtract();
  test_transpose();
  test_transpose_nonsquare();
  test_scale();
  test_scale_inplace();
  test_hadamard();
  test_frobenius();
  test_normalize();
  test_normalize_inplace();
  test_dense();
  test_trace();
  test_dot();
  test_identity();
  test_kronecker();
  test_triangular();
  test_forward_solve();
  test_backward_solve();
  test_lu_solve();
  test_set_diagonal_scalar();
  test_set_diagonal_span();
  test_invert2x2();
  test_block_triangular_solve();
  test_cholesky();

  std::cout << "\n" << passed << " passed, " << failed << " failed.\n";
  return failed > 0 ? 1 : 0;
}
