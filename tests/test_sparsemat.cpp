#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

#include "sparsemat.h"
#include "test_harness.h"
using namespace SparseLinearAlgebra;

// --- Tests ---
//
// Every test_*() function below is SPARSEMAT_HD so this exact same file can
// run as a normal host program (test_sparsemat, this file's main() below) or
// inside a CUDA kernel (test_sparsemat_gpu.cu, which #includes this file
// with SPARSEMAT_TEST_NO_MAIN defined and calls run_all_tests() from a
// __global__ kernel instead). One set of tests, two runners — see
// test_harness.h for how check()/check_eq()/check_near() adapt to each.

using IntType = int;

SPARSEMAT_HD void test_construction() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> mat(1, 2, 3);
  check_eq(mat.values[0], 1.0, "construction: values[0]");
  check_eq(mat.values[1], 2.0, "construction: values[1]");
  check_eq(mat.values[2], 3.0, "construction: values[2]");
  check_eq(decltype(mat)::nonZeroCount, IntType(3), "construction: nonZeroCount");
}

// Regression: the variadic constructor used to be unconstrained, which made
// SparseMat report itself constructible from literally any argument list.
// That poisons is_constructible_v, makes it a greedy converting constructor
// in any overload set, and turns a bad call into a static_cast error inside
// the constructor body instead of "no matching constructor". These are
// namespace-scope static_asserts because there is nothing to run — the whole
// point is what the type does and doesn't claim at compile time.
namespace ctor_constraints {
struct NotAScalar {};
using Mat1 = SparseMat<double, IntType, 1, 1, 0>;
using Mat3 = SparseMat<double, IntType, 3, 3, 0, 4, 8>;

static_assert(!std::is_constructible_v<Mat1, NotAScalar>,
              "SparseMat must not claim constructibility from unrelated types");
static_assert(!std::is_constructible_v<Mat3, NotAScalar, NotAScalar, NotAScalar>,
              "SparseMat must not claim constructibility from unrelated types");
static_assert(!std::is_constructible_v<Mat3, double, double>,
              "SparseMat must not claim constructibility from the wrong argument count");
static_assert(std::is_constructible_v<Mat3, double, double, double>,
              "SparseMat must still be constructible from nonZeroCount scalars");
static_assert(std::is_constructible_v<Mat3, int, int, int>,
              "SparseMat must still accept scalars convertible to DataType");
static_assert(std::is_copy_constructible_v<Mat1> && std::is_copy_assignable_v<Mat1>,
              "SparseMat must remain copyable");
}  // namespace ctor_constraints

SPARSEMAT_HD void test_copy_construction() {
  SparseMat<double, IntType, 1, 1, 0> one(7.0);
  SparseMat<double, IntType, 1, 1, 0> one_copy(one);  // non-const lvalue source
  check_eq(one_copy.values[0], 7.0, "copy construction: 1x1 from non-const lvalue");

  SparseMat<double, IntType, 3, 3, 0, 4, 8> mat(1, 2, 3);
  SparseMat<double, IntType, 3, 3, 0, 4, 8> mat_copy(mat);  // non-const lvalue source
  check_eq(mat_copy.values[0], 1.0, "copy construction: values[0]");
  check_eq(mat_copy.values[1], 2.0, "copy construction: values[1]");
  check_eq(mat_copy.values[2], 3.0, "copy construction: values[2]");

  const auto& mat_const = mat;
  SparseMat<double, IntType, 3, 3, 0, 4, 8> mat_copy2(mat_const);
  check_eq(mat_copy2.values[2], 3.0, "copy construction: from const lvalue");

  SparseMat<double, IntType, 3, 3, 0, 4, 8> mat_assigned = mat_copy;
  mat_assigned = mat_const;
  check_eq(mat_assigned.values[1], 2.0, "copy assignment: values[1]");
}

SPARSEMAT_HD void test_get() {
  // A = diag(1,2,3)
  SparseMat<double, IntType, 3, 3, 0, 4, 8> mat(1, 2, 3);
  check_eq(mat.template get<0, 0>(), 1.0, "get: (0,0)");
  check_eq(mat.template get<1, 1>(), 2.0, "get: (1,1)");
  check_eq(mat.template get<2, 2>(), 3.0, "get: (2,2)");
  check_eq(mat.template get<0, 1>(), 0.0, "get: zero element (0,1)");
  check_eq(mat.get(1, 1), 2.0, "get runtime: (1,1)");
  check_eq(mat.get(0, 1), 0.0, "get runtime: zero element (0,1)");
}

SPARSEMAT_HD void test_set() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> mat(1, 2, 3);
  mat.template set<0, 0>(10.0);
  check_eq(mat.template get<0, 0>(), 10.0, "set: (0,0)");

  bool ok = mat.set(1, 1, 20.0);
  check(ok, "set runtime: returns true for nonzero");
  check_eq(mat.get(1, 1), 20.0, "set runtime: (1,1)");

  bool fail = mat.set(0, 1, 99.0);
  check(!fail, "set runtime: returns false for zero index");
  check_eq(mat.get(0, 1), 0.0, "set runtime: zero index unchanged");
}

SPARSEMAT_HD void test_runtime_bounds() {
  // Regression: runtime get/set should reject out-of-bounds coordinates.
  // Current bug: (0,3) in a 3x3 matrix aliases flat index 3, i.e. (1,0).
  SparseMat<double, IntType, 3, 3, 3> mat(42.0);

  check_eq(mat.get(0, 3), 0.0, "runtime bounds: get rejects out-of-bounds column");

  bool wrote = mat.set(0, 3, 7.0);
  check(!wrote, "runtime bounds: set rejects out-of-bounds column");
  check_eq(mat.template get<1, 0>(),
           42.0,
           "runtime bounds: out-of-bounds set does not alias storage");
}

SPARSEMAT_HD void test_fill() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> mat;
  mat.fill(5.0);
  check_eq(mat.values[0], 5.0, "fill: values[0]");
  check_eq(mat.values[1], 5.0, "fill: values[1]");
  check_eq(mat.values[2], 5.0, "fill: values[2]");
}

SPARSEMAT_HD void test_multiply() {
  // A = diag(1,2,3), B = first row [4,5,6]
  // A*B = [[4,5,6],[0,0,0],[0,0,0]]
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, IntType, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto c = a.mult(b);
  check_eq(c.template get<0, 0>(), 4.0, "multiply: (0,0)");
  check_eq(c.template get<0, 1>(), 5.0, "multiply: (0,1)");
  check_eq(c.template get<0, 2>(), 6.0, "multiply: (0,2)");
  check_eq(c.template get<1, 0>(), 0.0, "multiply: (1,0) zero");
  check_eq(decltype(c)::nonZeroCount, IntType(3), "multiply: result nonZeroCount");
}

SPARSEMAT_HD void test_add() {
  // A = diag(1,2,3), B = first row [4,5,6]
  // A+B: (0,0)=5, (0,1)=5, (0,2)=6, (1,1)=2, (2,2)=3
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, IntType, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto c = a.add(b);
  check_eq(c.template get<0, 0>(), 5.0, "add: (0,0)");
  check_eq(c.template get<0, 1>(), 5.0, "add: (0,1)");
  check_eq(c.template get<0, 2>(), 6.0, "add: (0,2)");
  check_eq(c.template get<1, 1>(), 2.0, "add: (1,1)");
  check_eq(c.template get<2, 2>(), 3.0, "add: (2,2)");
  check_eq(decltype(c)::nonZeroCount, IntType(5), "add: result nonZeroCount");
}

SPARSEMAT_HD void test_subtract() {
  // A-B: (0,0)=-3, (0,1)=-5, (0,2)=-6, (1,1)=2, (2,2)=3
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, IntType, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto c = a.subtract(b);
  check_eq(c.template get<0, 0>(), -3.0, "subtract: (0,0)");
  check_eq(c.template get<0, 1>(), -5.0, "subtract: (0,1)");
  check_eq(c.template get<0, 2>(), -6.0, "subtract: (0,2)");
  check_eq(c.template get<1, 1>(), 2.0, "subtract: (1,1)");
  check_eq(c.template get<2, 2>(), 3.0, "subtract: (2,2)");
}

SPARSEMAT_HD void test_transpose() {
  // B = first row [4,5,6] → B^T = first col
  SparseMat<double, IntType, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto bt = b.transpose();
  check_eq(bt.template get<0, 0>(), 4.0, "transpose: (0,0)");
  check_eq(bt.template get<1, 0>(), 5.0, "transpose: (1,0)");
  check_eq(bt.template get<2, 0>(), 6.0, "transpose: (2,0)");
  check_eq(bt.template get<0, 1>(), 0.0, "transpose: zero (0,1)");
  check_eq(decltype(bt)::nonZeroCount, IntType(3), "transpose: nonZeroCount preserved");
}

SPARSEMAT_HD void test_transpose_nonsquare() {
  // 4x2 matrix (rows > cols) transposed to 2x4.
  // Nonzeros at (0,0)=flat 0 and (1,1)=flat 3 in the 4x2 grid.
  // Transpose swaps to (0,0)=flat 0 and (1,1)=flat 5 in the 2x4 grid.
  SparseMat<double, IntType, 4, 2, 0, 3> a(3.0, 7.0);
  auto at = a.transpose();
  check_eq(decltype(at)::rows, IntType(2), "transpose nonsquare: result rows");
  check_eq(decltype(at)::cols, IntType(4), "transpose nonsquare: result cols");
  check_eq(decltype(at)::nonZeroCount, IntType(2), "transpose nonsquare: nonZeroCount preserved");
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
  SparseMat<double, IntType, 4, 3, 0, 4, 8> b(4.0, 5.0, 6.0);
  auto bt = b.transpose();
  check_eq(decltype(bt)::rows, IntType(3), "transpose 4x3: result rows");
  check_eq(decltype(bt)::cols, IntType(4), "transpose 4x3: result cols");
  check_eq(decltype(bt)::nonZeroCount, IntType(3), "transpose 4x3: nonZeroCount preserved");
  check_eq(bt.template get<0, 0>(), 4.0, "transpose 4x3: (0,0)");
  check_eq(bt.template get<1, 1>(), 5.0, "transpose 4x3: (1,1)");
  check_eq(bt.template get<2, 2>(), 6.0, "transpose 4x3: (2,2)");
  check_eq(bt.template get<0, 1>(), 0.0, "transpose 4x3: zero (0,1)");
  check_eq(bt.template get<0, 3>(), 0.0, "transpose 4x3: zero (0,3)");

  // Regression: 2x4 → 4x2 (rows < cols, previously hit a static_assert).
  SparseMat<double, IntType, 2, 4, 0, 5> h(3.0, 4.0);
  auto ht = h.transpose();
  check_eq(decltype(ht)::rows, IntType(4), "transpose 2x4: result rows");
  check_eq(decltype(ht)::cols, IntType(2), "transpose 2x4: result cols");
  check_eq(ht.template get<0, 0>(), 3.0, "transpose 2x4: (0,0)");
  check_eq(ht.template get<1, 1>(), 4.0, "transpose 2x4: (1,1)");
  check_eq(ht.template get<0, 1>(), 0.0, "transpose 2x4: zero (0,1)");
  check_eq(ht.template get<1, 0>(), 0.0, "transpose 2x4: zero (1,0)");

  // Regression: 3x1 → 1x3 (column vector to row vector, previously silently dropped values).
  SparseMat<double, IntType, 3, 1, 0, 1, 2> col(1.0, 2.0, 3.0);
  auto row = col.transpose();
  check_eq(decltype(row)::rows, IntType(1), "transpose 3x1: result rows");
  check_eq(decltype(row)::cols, IntType(3), "transpose 3x1: result cols");
  check_eq(row.template get<0, 0>(), 1.0, "transpose 3x1: (0,0)");
  check_eq(row.template get<0, 1>(), 2.0, "transpose 3x1: (0,1)");
  check_eq(row.template get<0, 2>(), 3.0, "transpose 3x1: (0,2)");

  // Regression: 6x3 with a nonzero at source row 5 (previously silently dropped).
  SparseMat<double, IntType, 6, 3, 0, 7, 17> c(4.0, 5.0, 6.0);
  auto ct = c.transpose();
  check_eq(decltype(ct)::rows, IntType(3), "transpose 6x3: result rows");
  check_eq(decltype(ct)::cols, IntType(6), "transpose 6x3: result cols");
  check_eq(ct.template get<0, 0>(), 4.0, "transpose 6x3: (0,0)");
  check_eq(ct.template get<1, 2>(), 5.0, "transpose 6x3: (1,2)");
  check_eq(ct.template get<2, 5>(), 6.0, "transpose 6x3: (2,5)");
}

SPARSEMAT_HD void test_scale() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  auto b = a.scale(2.0);
  check_eq(b.template get<0, 0>(), 2.0, "scale: (0,0)");
  check_eq(b.template get<1, 1>(), 4.0, "scale: (1,1)");
  check_eq(b.template get<2, 2>(), 6.0, "scale: (2,2)");
  // original unchanged
  check_eq(a.template get<0, 0>(), 1.0, "scale: original unchanged");
}

SPARSEMAT_HD void test_scale_inplace() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  a.scale_inplace(3.0);
  check_eq(a.template get<0, 0>(), 3.0, "scale_inplace: (0,0)");
  check_eq(a.template get<1, 1>(), 6.0, "scale_inplace: (1,1)");
  check_eq(a.template get<2, 2>(), 9.0, "scale_inplace: (2,2)");
}

SPARSEMAT_HD void test_hadamard() {
  // A = diag(1,2,3), B = first row [4,5,6]
  // intersection of nonzeros: only (0,0) is nonzero in both
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, IntType, 3, 3, 0, 1, 2> b(4, 5, 6);
  auto c = a.hadamard(b);
  check_eq(c.template get<0, 0>(), 4.0, "hadamard: (0,0)");
  check_eq(decltype(c)::nonZeroCount, IntType(1), "hadamard: result nonZeroCount (intersection)");
}

SPARSEMAT_HD void test_frobenius() {
  // A = diag(1,2,3), ||A||_F = sqrt(1+4+9) = sqrt(14)
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  check_near(a.frobenius(), std::sqrt(14.0), "frobenius norm");
}

SPARSEMAT_HD void test_normalize() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  auto n = a.normalize();
  check_near(n.frobenius(), 1.0, "normalize: frobenius norm = 1");
}

SPARSEMAT_HD void test_normalize_inplace() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  a.normalize_inplace();
  check_near(a.frobenius(), 1.0, "normalize_inplace: frobenius norm = 1");
}

SPARSEMAT_HD void test_normalize_zero_returns_zero() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a;
  a.fill(0.0);
  auto n = a.normalize();
  check_near(n.frobenius(), 0.0, "normalize: zero matrix stays zero");
}

SPARSEMAT_HD void test_normalize_inplace_zero_noop() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a;
  a.fill(0.0);
  a.normalize_inplace();
  check_near(a.frobenius(), 0.0, "normalize_inplace: zero matrix stays zero");
}

SPARSEMAT_HD void test_dense() {
  // A = diag(1,2,3)
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  auto d = a.dense();
  check_eq(d.get<0, 0>(), 1.0, "dense: index 0");
  check_eq(d.get<1, 1>(), 2.0, "dense: index 4");
  check_eq(d.get<2, 2>(), 3.0, "dense: index 8");
  check_eq(d.get<0, 1>(), 0.0, "dense: zero element");
}

SPARSEMAT_HD void test_trace() {
  // A = diag(1,2,3), trace = 6
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  check_eq(SparseLinearAlgebra::trace(a), 6.0, "trace: diag(1,2,3)");
}

SPARSEMAT_HD void test_dot() {
  // row vector [1,2,3] · col vector [4,5,6] = 32
  SparseMat<double, IntType, 1, 3, 0, 1, 2> row(1, 2, 3);
  SparseMat<double, IntType, 3, 1, 0, 1, 2> col(4, 5, 6);
  check_eq(row.dot(col), 32.0, "dot: [1,2,3]·[4,5,6]");
}

SPARSEMAT_HD void test_axpy() {
  // A = diag(1,2,3), x = [4,5,6]^T  =>  A*x = [4,10,18]^T
  // y = [100, 0, 200]^T (structurally zero at row 1, so that row exercises
  // "A*x contributes, y doesn't"; row 2's A*x=18 combined with y=200
  // exercises both contributing; row 1 overall is a case where only A*x
  // contributes to a structurally non-zero result row).
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1, 2, 3);
  SparseMat<double, IntType, 3, 1, 0, 1, 2> x(4, 5, 6);
  SparseMat<double, IntType, 3, 1, 0, 2> y(100, 200);

  auto z = a.axpy(x, y);
  check_eq(z.template get<0, 0>(), 104.0, "axpy: row 0 (A*x and y both contribute)");
  check_eq(z.template get<1, 0>(), 10.0, "axpy: row 1 (only A*x contributes)");
  check_eq(z.template get<2, 0>(), 218.0, "axpy: row 2 (A*x and y both contribute)");

  // Free-function form should agree with the member form.
  auto z2 = axpy(a, x, y);
  check_eq(z2.template get<0, 0>(), 104.0, "axpy: free function matches member");

  // Row that's structurally zero in both A*x and y should be structurally
  // elided from the result's sparsity pattern, not just numerically zero.
  SparseMat<double, IntType, 3, 3, 0, 8> b(1, 3);  // row 1 has no non-zeros
  SparseMat<double, IntType, 3, 1, 0, 2> y2;       // row 1 structurally zero
  auto z3 = b.axpy(x, y2);
  check_eq(decltype(z3)::nonZeroCount, IntType(2), "axpy: result sparsity excludes zero row");
  check_eq(z3.template get<0, 0>(), 4.0, "axpy: row 0 of structurally-sparse case");
  check_eq(z3.template get<1, 0>(), 0.0, "axpy: structurally zero row stays zero");
  check_eq(z3.template get<2, 0>(), 18.0, "axpy: row 2 of structurally-sparse case");

  // Scaled form: alpha*A*x + beta*y. With alpha=2, beta=3:
  // row 0 = 2*4   + 3*100 = 308
  // row 1 = 2*10  + 3*0   = 20   (y structurally zero at row 1)
  // row 2 = 2*18  + 3*200 = 636
  auto z4 = a.axpy(x, y, 2.0, 3.0);
  check_eq(z4.template get<0, 0>(), 308.0, "axpy: scaled row 0");
  check_eq(z4.template get<1, 0>(), 20.0, "axpy: scaled row 1");
  check_eq(z4.template get<2, 0>(), 636.0, "axpy: scaled row 2");
}

SPARSEMAT_HD void test_identity() {
  auto I = identity<double, IntType, 3, 3>();
  check_eq(I.template get<0, 0>(), 1.0, "identity: (0,0)");
  check_eq(I.template get<1, 1>(), 1.0, "identity: (1,1)");
  check_eq(I.template get<2, 2>(), 1.0, "identity: (2,2)");
  check_eq(I.template get<0, 1>(), 0.0, "identity: off-diagonal zero");
  check_eq(decltype(I)::nonZeroCount, IntType(3), "identity: nonZeroCount");

  auto I2 = identity<double, IntType, 4, 2>();
  check_eq(I2.template get<0, 0>(), 1.0, "identity: (0,0) in 4x2");
  check_eq(I2.template get<1, 1>(), 1.0, "identity: (1,1) in 4x2");
  check_eq(I2.template get<2, 0>(), 0.0, "identity: (2,0) in 4x2");
  check_eq(I2.template get<3, 1>(), 0.0, "identity: (3,1) in 4x2");
  check_eq(decltype(I2)::nonZeroCount, IntType(2), "identity: nonZeroCount in 4x2");
}

SPARSEMAT_HD void test_make_sparse_matrix() {
  // Entries deliberately out of row-major order, to exercise the internal
  // sort: (1,1) and (0,1) are listed before (0,0).
  constexpr auto entries = std::array{SparseLinearAlgebra::SparseEntry{1, 1, 2.0},
                                      SparseLinearAlgebra::SparseEntry{0, 1, 5.0},
                                      SparseLinearAlgebra::SparseEntry{0, 0, 4.0}};
  auto A = SparseLinearAlgebra::make_sparse_matrix<double, IntType, 3, 3, entries>();

  check_eq(decltype(A)::rows, IntType(3), "make_sparse_matrix: rows");
  check_eq(decltype(A)::cols, IntType(3), "make_sparse_matrix: cols");
  check_eq(decltype(A)::nonZeroCount, IntType(3), "make_sparse_matrix: nonZeroCount");
  check_eq(A.template get<0, 0>(), 4.0, "make_sparse_matrix: (0,0)");
  check_eq(A.template get<0, 1>(), 5.0, "make_sparse_matrix: (0,1)");
  check_eq(A.template get<1, 1>(), 2.0, "make_sparse_matrix: (1,1)");
  check_eq(A.template get<0, 2>(), 0.0, "make_sparse_matrix: structurally zero (0,2)");
  check_eq(A.template get<1, 0>(), 0.0, "make_sparse_matrix: structurally zero (1,0)");

  // Listing order shouldn't matter: the same entries in ascending order
  // should produce an equivalent (here, an identical-typed) matrix.
  constexpr auto entries_sorted = std::array{SparseLinearAlgebra::SparseEntry{0, 0, 4.0},
                                             SparseLinearAlgebra::SparseEntry{0, 1, 5.0},
                                             SparseLinearAlgebra::SparseEntry{1, 1, 2.0}};
  auto A2 = SparseLinearAlgebra::make_sparse_matrix<double, IntType, 3, 3, entries_sorted>();
  static_assert(std::is_same_v<decltype(A), decltype(A2)>,
                "make_sparse_matrix: listing order should not affect the resulting type");
  check(A2 == A, "make_sparse_matrix: listing order does not affect the values either");

  // The result is a fully-ordinary SparseMat — usable in any operation.
  SparseMat<double, IntType, 3, 1, 0, 1, 2> x(1, 1, 1);
  auto Ax = A.mult(x);
  check_eq(Ax.template get<0, 0>(), 9.0, "make_sparse_matrix: usable in mult() — row 0");
  check_eq(Ax.template get<1, 0>(), 2.0, "make_sparse_matrix: usable in mult() — row 1");
}

SPARSEMAT_HD void test_kronecker() {
  // A = diag(1, 2)  (2x2), B = [[3,4],[5,6]]  (2x2 full)
  // A ⊗ B = [[3,4,0,0],[5,6,0,0],[0,0,6,8],[0,0,10,12]]
  SparseMat<double, IntType, 2, 2, 0, 3> A(1, 2);
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> B(3, 4, 5, 6);
  auto C = A.kronecker(B);

  check_eq(decltype(C)::rows, IntType(4), "kronecker: result rows");
  check_eq(decltype(C)::cols, IntType(4), "kronecker: result cols");
  check_eq(decltype(C)::nonZeroCount, IntType(8), "kronecker: nonZeroCount");

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
  check_eq(decltype(C)::nonZeroCount, IntType(8), "kronecker: nonZeroCount");
}

// Regression for the fold-expression flattening of the recursive compile-time
// loops (add/multiply/hadamard/kronecker/transpose/triangular/LU/Cholesky):
// Kronecker's result grid is the PRODUCT of both operands' dimensions, making
// its old linear template recursion depth (rows_total*cols_total) the
// fastest-growing of any operation in this library. Two 8x8 operands produce
// a 64x64 result — recursion depth 4096 under the old scheme, ~4.5x over the
// compiler's default ~900-deep template-instantiation budget — so this would
// not have compiled before that flattening. It's now O(1) depth regardless
// of matrix size.
SPARSEMAT_HD void test_kronecker_large() {
  // Two 8x8 diagonal matrices; a Kronecker product of diagonal matrices is
  // itself diagonal (nonzero only where row/8==col/8 AND row%8==col%8, i.e.
  // row==col), giving a 64x64 diagonal result with 64 stored entries.
  SparseMat<double, IntType, 8, 8, 0, 9, 18, 27, 36, 45, 54, 63> a(1, 2, 3, 4, 5, 6, 7, 8);
  SparseMat<double, IntType, 8, 8, 0, 9, 18, 27, 36, 45, 54, 63> b(1, 2, 3, 4, 5, 6, 7, 8);
  auto c = a.kronecker(b);

  check_eq(decltype(c)::rows, IntType(64), "kronecker large: result rows");
  check_eq(decltype(c)::cols, IntType(64), "kronecker large: result cols");
  check_eq(decltype(c)::nonZeroCount, IntType(64), "kronecker large: nonZeroCount");

  check_eq(c.template get<0, 0>(), 1.0, "kronecker large: (0,0) = A(0,0)*B(0,0)");
  // row = 8*3+4 = 28: A(3,3)*B(4,4) = 4*5 = 20
  check_eq(c.template get<28, 28>(), 20.0, "kronecker large: (28,28) = A(3,3)*B(4,4)");
  // row = 8*7+7 = 63: A(7,7)*B(7,7) = 8*8 = 64
  check_eq(c.template get<63, 63>(), 64.0, "kronecker large: (63,63) = A(7,7)*B(7,7)");
  check_eq(c.template get<0, 1>(), 0.0, "kronecker large: off-diagonal zero");
}

SPARSEMAT_HD void test_triangular() {
  // Lower triangular: non-zeros only on and below diagonal
  // 3x3: indices 0(0,0), 3(1,0), 4(1,1), 6(2,0), 7(2,1), 8(2,2)
  using Lower = SparseMat<double, IntType, 3, 3, 0, 3, 4, 6, 7, 8>;
  Lower lower(1, 2, 3, 4, 5, 6);
  check(Lower::is_structurally_lower_triangular(), "triangular: lower is structurally lower");
  check(!Lower::is_structurally_upper_triangular(), "triangular: lower is not structurally upper");
  check(lower.is_numerically_lower_triangular(), "triangular: lower is numerically lower");
  check(!lower.is_numerically_upper_triangular(), "triangular: lower is not numerically upper");

  // Upper triangular: non-zeros only on and above diagonal
  // 3x3: indices 0(0,0), 1(0,1), 2(0,2), 4(1,1), 5(1,2), 8(2,2)
  using Upper = SparseMat<double, IntType, 3, 3, 0, 1, 2, 4, 5, 8>;
  Upper upper(1, 2, 3, 4, 5, 6);
  check(!Upper::is_structurally_lower_triangular(), "triangular: upper is not structurally lower");
  check(Upper::is_structurally_upper_triangular(), "triangular: upper is structurally upper");
  check(!upper.is_numerically_lower_triangular(), "triangular: upper is not numerically lower");
  check(upper.is_numerically_upper_triangular(), "triangular: upper is numerically upper");

  // Diagonal: both lower and upper triangular
  using Diag = SparseMat<double, IntType, 3, 3, 0, 4, 8>;
  Diag diag(1, 2, 3);
  check(Diag::is_structurally_lower_triangular(), "triangular: diagonal is structurally lower");
  check(Diag::is_structurally_upper_triangular(), "triangular: diagonal is structurally upper");

  // Structurally upper but numerically not: above-diagonal value is non-zero
  // 3x3: non-zeros at 0(0,0), 1(0,1), 4(1,1), 8(2,2) — has above-diagonal entry
  using Mixed = SparseMat<double, IntType, 3, 3, 0, 1, 4, 8>;
  Mixed mixed(1, 2, 3, 4);
  // (0,1) is above diagonal → structurally not lower
  check(!Mixed::is_structurally_lower_triangular(), "triangular: mixed is not structurally lower");
  // numerically lower: set above-diagonal value to near-zero
  mixed.template set<0, 1>(0.0);
  check(mixed.is_numerically_lower_triangular(),
        "triangular: mixed is numerically lower when above-diagonal is zero");

  // is_numerically_lower/upper walk every above/below-diagonal stored entry
  // and short-circuit on the first out-of-tolerance value. The checks above
  // all hit that short-circuit on the very first entry, so the tail of the
  // walk (later I/J template instantiations) never runs. Exercise the full
  // walk here by zeroing every off-triangle entry instead of just one.
  Upper upper_zeroed(1, 0, 0, 4, 0, 6);
  check(upper_zeroed.is_numerically_lower_triangular(),
        "triangular: upper with all above-diagonal values zeroed is numerically lower");

  Lower lower_zeroed(1, 0, 4, 0, 0, 6);
  check(lower_zeroed.is_numerically_upper_triangular(),
        "triangular: lower with all below-diagonal values zeroed is numerically upper");
}

SPARSEMAT_HD void test_forward_solve() {
  // L = [[2,0,0],[1,3,0],[4,5,6]], b = [4,11,47]
  // x[0] = 4/2 = 2
  // x[1] = (11 - 1*2)/3 = 3
  // x[2] = (47 - 4*2 - 5*3)/6 = 4
  // indices: (0,0)=0, (1,0)=3, (1,1)=4, (2,0)=6, (2,1)=7, (2,2)=8
  SparseMat<double, IntType, 3, 3, 0, 3, 4, 6, 7, 8> L(2, 1, 3, 4, 5, 6);
  SparseMat<double, IntType, 3, 1, 0, 1, 2> b(4, 11, 47);

  auto result = SparseLinearAlgebra::forward_solve(L, b);
  check(result.ok(), "forward_solve: reports success on a non-singular matrix");
  auto x = result.value();

  check_near(x.get(0, 0), 2.0, "forward_solve: x[0]");
  check_near(x.get(1, 0), 3.0, "forward_solve: x[1]");
  check_near(x.get(2, 0), 4.0, "forward_solve: x[2]");

  // Verify Lx = b
  auto residual = L.mult(x);
  check_near(residual.get(0, 0), 4.0, "forward_solve: residual b[0]");
  check_near(residual.get(1, 0), 11.0, "forward_solve: residual b[1]");
  check_near(residual.get(2, 0), 47.0, "forward_solve: residual b[2]");
}

SPARSEMAT_HD void test_backward_solve() {
  // U = [[2,1,4],[0,3,5],[0,0,6]], b = [23,29,24]
  // x[2] = 24/6 = 4
  // x[1] = (29 - 5*4)/3 = 3
  // x[0] = (23 - 1*3 - 4*4)/2 = 2
  // indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,1)=4, (1,2)=5, (2,2)=8
  SparseMat<double, IntType, 3, 3, 0, 1, 2, 4, 5, 8> U(2, 1, 4, 3, 5, 6);
  SparseMat<double, IntType, 3, 1, 0, 1, 2> b(23, 29, 24);

  auto result = SparseLinearAlgebra::backward_solve(U, b);
  check(result.ok(), "backward_solve: reports success on a non-singular matrix");
  auto x = result.value();

  check_near(x.get(0, 0), 2.0, "backward_solve: x[0]");
  check_near(x.get(1, 0), 3.0, "backward_solve: x[1]");
  check_near(x.get(2, 0), 4.0, "backward_solve: x[2]");

  // Verify Ux = b
  auto residual = U.mult(x);
  check_near(residual.get(0, 0), 23.0, "backward_solve: residual b[0]");
  check_near(residual.get(1, 0), 29.0, "backward_solve: residual b[1]");
  check_near(residual.get(2, 0), 24.0, "backward_solve: residual b[2]");
}

SPARSEMAT_HD void test_lu_solve() {
  // A = [[2,1,1],[4,3,3],[8,7,9]], b = [1,1,1], x = [1,-1,1] via LU
  // Flat indices (3x3): (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5,(2,0)=6,(2,1)=7,(2,2)=8
  SparseMat<double, IntType, 3, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8> A(2, 1, 1, 4, 3, 3, 8, 7, 9);
  SparseMat<double, IntType, 3, 1, 0, 1, 2> b(1, 1, 1);

  auto result = A.solve(b);
  check(result.ok(), "lu_solve: reports success on a non-singular matrix");
  auto x = result.value();

  check_near(x.get(0, 0), 1.0, "lu_solve: x[0]");
  check_near(x.get(1, 0), -1.0, "lu_solve: x[1]");
  check_near(x.get(2, 0), 0.0, "lu_solve: x[2]");

  // Verify Ax = b
  auto residual = A.mult(x);
  check_near(residual.get(0, 0), 1.0, "lu_solve: residual b[0]");
  check_near(residual.get(1, 0), 1.0, "lu_solve: residual b[1]");
  check_near(residual.get(2, 0), 1.0, "lu_solve: residual b[2]");
}

SPARSEMAT_HD void test_lu_factorize() {
  // A = [[2,1,3],[4,6,10],[6,10,20]]
  // U[0] = [2,1,3]
  // L[1][0] = 4/2 = 2
  // U[1][1] = 6 - 2*1 = 4
  // U[1][2] = 10 - 2*3 = 4   (depends on column 2 of U, not column 1!)
  // L[2][0] = 6/2 = 3
  // L[2][1] = (10 - 3*1)/4 = 1.75
  // U[2][2] = 20 - (3*3 + 1.75*4) = 4
  SparseMat<double, IntType, 3, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8> A(2, 1, 3, 4, 6, 10, 6, 10, 20);

  auto result = SparseLinearAlgebra::lu_factorize(A);
  check(result.ok(), "lu_factorize: reports success on a non-singular matrix");
  auto& l = result.value().first;
  auto& u = result.value().second;

  check_near(u.get(0, 0), 2.0, "lu_factorize: U[0][0]");
  check_near(u.get(0, 1), 1.0, "lu_factorize: U[0][1]");
  check_near(u.get(0, 2), 3.0, "lu_factorize: U[0][2]");
  check_near(l.get(1, 0), 2.0, "lu_factorize: L[1][0]");
  check_near(u.get(1, 1), 4.0, "lu_factorize: U[1][1]");
  check_near(u.get(1, 2), 4.0, "lu_factorize: U[1][2]");
  check_near(l.get(2, 0), 3.0, "lu_factorize: L[2][0]");
  check_near(l.get(2, 1), 1.75, "lu_factorize: L[2][1]");
  check_near(u.get(2, 2), 4.0, "lu_factorize: U[2][2]");

  // Verify LU = A as well.
  auto product = l.mult(u);
  check_near(product.get(0, 0), 2.0, "lu_factorize: LU[0][0]");
  check_near(product.get(0, 1), 1.0, "lu_factorize: LU[0][1]");
  check_near(product.get(0, 2), 3.0, "lu_factorize: LU[0][2]");
  check_near(product.get(1, 0), 4.0, "lu_factorize: LU[1][0]");
  check_near(product.get(1, 1), 6.0, "lu_factorize: LU[1][1]");
  check_near(product.get(1, 2), 10.0, "lu_factorize: LU[1][2]");
  check_near(product.get(2, 0), 6.0, "lu_factorize: LU[2][0]");
  check_near(product.get(2, 1), 10.0, "lu_factorize: LU[2][1]");
  check_near(product.get(2, 2), 20.0, "lu_factorize: LU[2][2]");
}

SPARSEMAT_HD void test_set_diagonal_scalar() {
  // Full diagonal matrix: all three diagonal entries are stored.
  SparseMat<double, IntType, 3, 3, 0, 4, 8> diag(1, 2, 3);
  diag.set_diagonal(7.0);
  check_eq(diag.template get<0, 0>(), 7.0, "set_diagonal scalar: (0,0)");
  check_eq(diag.template get<1, 1>(), 7.0, "set_diagonal scalar: (1,1)");
  check_eq(diag.template get<2, 2>(), 7.0, "set_diagonal scalar: (2,2)");

  // General matrix: off-diagonal stored values must be left untouched.
  // indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1), 8=(2,2)
  SparseMat<double, IntType, 3, 3, 0, 1, 3, 4, 8> mat(1, 2, 3, 4, 5);
  mat.set_diagonal(9.0);
  check_eq(mat.template get<0, 0>(), 9.0, "set_diagonal scalar general: (0,0)");
  check_eq(mat.template get<1, 1>(), 9.0, "set_diagonal scalar general: (1,1)");
  check_eq(mat.template get<2, 2>(), 9.0, "set_diagonal scalar general: (2,2)");
  check_eq(mat.template get<0, 1>(), 2.0, "set_diagonal scalar general: off-diag (0,1) unchanged");
  check_eq(mat.template get<1, 0>(), 3.0, "set_diagonal scalar general: off-diag (1,0) unchanged");

  // Matrix where (1,1) is structurally zero: set_diagonal must not touch it.
  // indices: 0=(0,0), 1=(0,1), 3=(1,0), 8=(2,2)
  SparseMat<double, IntType, 3, 3, 0, 1, 3, 8> missing(1, 2, 3, 4);
  missing.set_diagonal(5.0);
  check_eq(missing.template get<0, 0>(), 5.0, "set_diagonal scalar missing: (0,0)");
  check_eq(missing.template get<2, 2>(), 5.0, "set_diagonal scalar missing: (2,2)");
  check_eq(missing.template get<1, 1>(), 0.0, "set_diagonal scalar missing: (1,1) stays zero");
  check_eq(missing.template get<0, 1>(),
           2.0,
           "set_diagonal scalar missing: off-diag (0,1) unchanged");
}

SPARSEMAT_HD void test_set_diagonal_array() {
  // Full diagonal: span values map to (0,0), (1,1), (2,2) in order.
  SparseMat<double, IntType, 3, 3, 0, 4, 8> diag(1, 2, 3);
  std::array<double, 3> v1{10.0, 20.0, 30.0};
  diag.set_diagonal(v1);
  check_eq(diag.template get<0, 0>(), 10.0, "set_diagonal span full: (0,0)");
  check_eq(diag.template get<1, 1>(), 20.0, "set_diagonal span full: (1,1)");
  check_eq(diag.template get<2, 2>(), 30.0, "set_diagonal span full: (2,2)");

  // General matrix: off-diagonal stored values must not be overwritten.
  // indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1), 8=(2,2)
  SparseMat<double, IntType, 3, 3, 0, 1, 3, 4, 8> mat(1, 2, 3, 4, 5);
  std::array<double, 3> v2{7.0, 8.0, 9.0};
  mat.set_diagonal(v2);
  check_eq(mat.template get<0, 0>(), 7.0, "set_diagonal span general: (0,0)");
  check_eq(mat.template get<1, 1>(), 8.0, "set_diagonal span general: (1,1)");
  check_eq(mat.template get<2, 2>(), 9.0, "set_diagonal span general: (2,2)");
  check_eq(mat.template get<0, 1>(), 2.0, "set_diagonal span general: off-diag (0,1) unchanged");
  check_eq(mat.template get<1, 0>(), 3.0, "set_diagonal span general: off-diag (1,0) unchanged");

  // Structurally missing (1,1): span of size 2 covers only the two stored diagonals.
  // (1,1) is skipped without consuming a span entry.
  // indices: 0=(0,0), 1=(0,1), 3=(1,0), 8=(2,2)
  SparseMat<double, IntType, 3, 3, 0, 1, 3, 8> missing(1, 2, 3, 4);
  std::array<double, 2> v3{11.0, 33.0};
  missing.set_diagonal(v3);
  check_eq(missing.template get<0, 0>(), 11.0, "set_diagonal span missing: (0,0)");
  check_eq(missing.template get<2, 2>(), 33.0, "set_diagonal span missing: (2,2)");
  check_eq(missing.template get<1, 1>(), 0.0, "set_diagonal span missing: (1,1) stays zero");
  check_eq(missing.template get<0, 1>(),
           2.0,
           "set_diagonal span missing: off-diag (0,1) unchanged");

  // Test case for bug: 4x2 matrix with only (3,0) stored should not be affected by set_diagonal
  // indices: 6=(3,0)
  SparseMat<double, IntType, 4, 2, 6> single_elem(1.0);
  single_elem.set_diagonal(9.0);
  check_eq(single_elem.template get<3, 0>(), 1.0, "set_diagonal: non-diagonal (3,0) unchanged");
}

SPARSEMAT_HD void test_block_triangular_solve() {
  // forward_solve with 2-column RHS
  // L = [2 0 0; 1 3 0; 2 1 4],  L * [1 2; 1 1; 3 2] = [2 4; 4 5; 15 13]
  SparseMat<double, IntType, 3, 3, 0, 3, 4, 6, 7, 8> L(2.0, 1.0, 3.0, 2.0, 1.0, 4.0);
  SparseMat<double, IntType, 3, 2, 0, 1, 2, 3, 4, 5> B(2.0, 4.0, 4.0, 5.0, 15.0, 13.0);
  auto X = SparseLinearAlgebra::forward_solve(L, B).value();
  check_near(X.template get<0, 0>(), 1.0, "block fwd: x[0][0]");
  check_near(X.template get<1, 0>(), 1.0, "block fwd: x[1][0]");
  check_near(X.template get<2, 0>(), 3.0, "block fwd: x[2][0]");
  check_near(X.template get<0, 1>(), 2.0, "block fwd: x[0][1]");
  check_near(X.template get<1, 1>(), 1.0, "block fwd: x[1][1]");
  check_near(X.template get<2, 1>(), 2.0, "block fwd: x[2][1]");

  // backward_solve with 2-column RHS
  // U = [4 1 2; 0 3 1; 0 0 2],  U * [1 2; 1 1; 3 3] = [11 15; 6 6; 6 6]
  SparseMat<double, IntType, 3, 3, 0, 1, 2, 4, 5, 8> U(4.0, 1.0, 2.0, 3.0, 1.0, 2.0);
  SparseMat<double, IntType, 3, 2, 0, 1, 2, 3, 4, 5> Brhs(11.0, 15.0, 6.0, 6.0, 6.0, 6.0);
  auto Y = SparseLinearAlgebra::backward_solve(U, Brhs).value();
  check_near(Y.template get<0, 0>(), 1.0, "block bwd: y[0][0]");
  check_near(Y.template get<1, 0>(), 1.0, "block bwd: y[1][0]");
  check_near(Y.template get<2, 0>(), 3.0, "block bwd: y[2][0]");
  check_near(Y.template get<0, 1>(), 2.0, "block bwd: y[0][1]");
  check_near(Y.template get<1, 1>(), 1.0, "block bwd: y[1][1]");
  check_near(Y.template get<2, 1>(), 3.0, "block bwd: y[2][1]");
}

SPARSEMAT_HD void test_symmetric() {
  // Structurally symmetric, values symmetric.
  // indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> sym(1.0, 2.0, 2.0, 3.0);
  check_eq(SparseLinearAlgebra::is_structurally_symmetric(sym),
           true,
           "is_structurally_symmetric: symmetric pattern");
  check_eq(SparseLinearAlgebra::is_sparse_symmetric(sym),
           true,
           "is_sparse_symmetric: symmetric values");

  // Structurally asymmetric: (0,1) stored, (1,0) not.
  // indices: 0=(0,0), 1=(0,1), 3=(1,1)
  SparseMat<double, IntType, 2, 2, 0, 1, 3> asym(1.0, 2.0, 3.0);
  check_eq(SparseLinearAlgebra::is_structurally_symmetric(asym),
           false,
           "is_structurally_symmetric: asymmetric pattern");

  // Structurally symmetric pattern but asymmetric values.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> unsym_vals(1.0, 2.0, 5.0, 3.0);
  check_eq(SparseLinearAlgebra::is_sparse_symmetric(unsym_vals),
           false,
           "is_sparse_symmetric: asymmetric values");

  // is_full_symmetric: fully symmetric matrix.
  check_eq(SparseLinearAlgebra::is_full_symmetric(sym),
           true,
           "is_full_symmetric: symmetric values");

  // is_full_symmetric: symmetric pattern, asymmetric values.
  check_eq(SparseLinearAlgebra::is_full_symmetric(unsym_vals),
           false,
           "is_full_symmetric: asymmetric values");

  // is_full_symmetric: stored (0,1) has no mirrored (1,0) entry and is non-zero.
  SparseMat<double, IntType, 2, 2, 0, 1, 3> full_asym(1.0, 2.0, 3.0);
  check_eq(SparseLinearAlgebra::is_full_symmetric(full_asym),
           false,
           "is_full_symmetric: missing mirrored position");

  // is_full_symmetric walks every stored entry and short-circuits on the
  // first asymmetric one. `full_asym` above short-circuits on its very
  // first stored entry, so the tail of the walk never runs for this
  // sparsity pattern. Exercise it to completion: same pattern (0,1 has no
  // mirrored 1,0 entry), but the unmirrored value is zero, which is
  // consistent with the implied zero at (1,0).
  SparseMat<double, IntType, 2, 2, 0, 1, 3> full_asym_zeroed(5.0, 0.0, 7.0);
  check_eq(SparseLinearAlgebra::is_full_symmetric(full_asym_zeroed),
           true,
           "is_full_symmetric: missing mirrored position with zero value is symmetric");
}

SPARSEMAT_HD void test_cholesky() {
  // Diagonal 2x2 SPD: A = diag(4, 9) → L = diag(2, 3)
  SparseMat<double, IntType, 2, 2, 0, 3> A2(4.0, 9.0);
  auto L2_result = SparseLinearAlgebra::cholesky_factorize(A2);
  check(L2_result.ok(), "cholesky diag 2x2: reports success on an SPD matrix");
  auto L2 = L2_result.value();
  check_near(L2.template get<0, 0>(), 2.0, "cholesky diag 2x2: L[0][0]");
  check_near(L2.template get<1, 1>(), 3.0, "cholesky diag 2x2: L[1][1]");

  // Full 3x3 SPD: A = [4 2 0; 2 5 2; 0 2 5] → L = [2 0 0; 1 2 0; 0 1 2]
  SparseMat<double, IntType, 3, 3, 0, 1, 3, 4, 5, 7, 8> A3(4.0, 2.0, 2.0, 5.0, 2.0, 2.0, 5.0);
  auto L3_result = SparseLinearAlgebra::cholesky_factorize(A3);
  check(L3_result.ok(), "cholesky 3x3: reports success on an SPD matrix");
  auto L3 = L3_result.value();
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
  SparseMat<double, IntType, 3, 1, 0, 1, 2> b(4.0, 4.0, 5.0);
  auto x_result = SparseLinearAlgebra::cholesky_solve(A3, b);
  check(x_result.ok(), "cholesky solve: reports success on an SPD matrix");
  auto x = x_result.value();
  check_near(x.template get<0, 0>(), 1.0, "cholesky solve: x[0]");
  check_near(x.template get<1, 0>(), 0.0, "cholesky solve: x[1]");
  check_near(x.template get<2, 0>(), 1.0, "cholesky solve: x[2]");
}

SPARSEMAT_HD void test_singular_solves() {
  // Lower triangular with a zero pivot at (0,0): forward substitution would
  // divide by zero. `ok()` must report the failure instead of silently
  // returning inf/nan.
  SparseMat<double, IntType, 2, 2, 0, 2, 3> L(0.0, 1.0, 2.0);
  SparseMat<double, IntType, 2, 1, 0, 1> lb(1.0, 2.0);
  auto fwd = SparseLinearAlgebra::forward_solve(L, lb);
  check(!fwd.ok(), "forward_solve: reports failure on a zero diagonal pivot");

  // Upper triangular with a zero pivot at (1,1): back substitution would
  // divide by zero.
  SparseMat<double, IntType, 2, 2, 0, 1, 3> U(2.0, 1.0, 0.0);
  SparseMat<double, IntType, 2, 1, 0, 1> ub(1.0, 2.0);
  auto bwd = SparseLinearAlgebra::backward_solve(U, ub);
  check(!bwd.ok(), "backward_solve: reports failure on a zero diagonal pivot");

  // Singular matrix (rows are linearly dependent): elimination hits a zero
  // pivot at the second step.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> singular(1.0, 2.0, 2.0, 4.0);
  SparseMat<double, IntType, 2, 1, 0, 1> sb(1.0, 2.0);
  auto lu_fact = SparseLinearAlgebra::lu_factorize(singular);
  check(!lu_fact.ok(), "lu_factorize: reports failure on a singular matrix");
  auto lu = SparseLinearAlgebra::lu_solve(singular, sb);
  check(!lu.ok(), "lu_solve: reports failure on a singular matrix");
  auto solve_result = singular.solve(sb);
  check(!solve_result.ok(), "SparseMat::solve: reports failure on a singular matrix");

  // Symmetric but not positive definite (determinant < 0): the diagonal
  // value under the square root goes negative.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> not_pd(1.0, 2.0, 2.0, 1.0);
  auto chol = SparseLinearAlgebra::cholesky_factorize(not_pd);
  check(!chol.ok(), "cholesky_factorize: reports failure on a non-positive-definite matrix");
  auto chol_handle = not_pd.cholesky();
  check(!chol_handle.ok(),
        "SparseMat::cholesky: reports failure on a non-positive-definite matrix");
}

// --- Larger-matrix stress test ---
//
// Proves the fold-expression flattening lifts the recursion-depth ceiling at
// a size well beyond anything exercised above (previously nothing larger
// than ~6x7 was tested — the old recursive scheme's depth would already be
// over the compiler's default ~900-deep template-instantiation budget for
// several of the fill-shaped ops here). Host-only in every sense — not
// SPARSEMAT_HD, not called from run_all_tests(), and (via this #ifndef)
// never even compiled into test_sparsemat_gpu.cu — for two independent
// reasons:
//   1. LU/Cholesky's reduction folds aren't yet narrowed to each
//      column/row's actual fill pattern (they're sized to the full
//      column/row bound and rely on if-constexpr to skip structurally-zero
//      terms), so their total instantiation count grows faster than the
//      matrix dimension — a naive 100x100 here exhausted g++'s memory. That's
//      the separate, pre-existing "width"/compile-time wall this refactor
//      doesn't address (see the plan's honest scope-setting note), not a
//      regression in the depth fix itself — Kronecker's dedicated test above
//      already proves depth at a much larger effective size (64x64 =
//      recursion depth 4096 under the old scheme) because fill ops don't
//      carry that same unnarrowed-fold cost.
//   2. nvcc's own constexpr evaluator (used even for host-side constexpr
//      when compiling a .cu file) has a much lower complexity/step budget
//      than g++'s for evaluating calculate_sparsity()'s O(rows*cols) scan,
//      independent of the template-depth fix — so even a size g++ handles
//      easily can exceed nvcc's limit.
// kN below is picked to comfortably clear the old ~6x7 ceiling and exercise
// every op at a non-trivial size without hitting wall #1 above; wall #2 is
// avoided entirely by keeping this whole section out of the nvcc build.
//
// Builds a diagonally-dominant symmetric tridiagonal matrix (SPD, so both LU
// and Cholesky are numerically stable without pivoting) and runs add,
// multiply, lu_solve, and cholesky_solve end-to-end against it.
#ifndef SPARSEMAT_TEST_NO_MAIN

namespace stress_test {
constexpr int kN = 60;
constexpr int kNnz = kN + (2 * (kN - 1));  // diagonal + both off-diagonals

// Flat row-major indices of a tridiagonal NxN pattern, in ascending order.
constexpr auto make_indices() {
  std::array<int, kNnz> idx{};
  int k = 0;
  for (int i = 0; i < kN; ++i) {
    if (i > 0) {
      idx[k++] = (i * kN) + (i - 1);
    }
    idx[k++] = (i * kN) + i;
    if (i < kN - 1) {
      idx[k++] = (i * kN) + (i + 1);
    }
  }
  return idx;
}
constexpr auto kIndices = make_indices();

template<std::size_t... Is>
constexpr auto make_matrix_type(std::index_sequence<Is...> /*seq*/) {
  return SparseMat<double, IntType, kN, kN, kIndices[Is]...>{};
}
using Matrix100 = decltype(make_matrix_type(std::make_index_sequence<kNnz>{}));

template<std::size_t... Is>
constexpr auto make_vector_type(std::index_sequence<Is...> /*seq*/) {
  return SparseMat<double, IntType, kN, 1, static_cast<IntType>(Is)...>{};
}
using Vector100 = decltype(make_vector_type(std::make_index_sequence<kN>{}));

// diag=4, off-diag=-1: diagonally dominant, so SPD (Cholesky-stable) and
// pivot-free (LU-stable without pivoting).
std::array<double, kNnz> make_values() {
  std::array<double, kNnz> vals{};
  int k = 0;
  for (int i = 0; i < kN; ++i) {
    if (i > 0) {
      vals[k++] = -1.0;
    }
    vals[k++] = 4.0;
    if (i < kN - 1) {
      vals[k++] = -1.0;
    }
  }
  return vals;
}
}  // namespace stress_test

// Lambdas are the spelling most callers will use; they are exercised here
// rather than in run_all_tests() because that also runs inside a CUDA kernel,
// where a host-defined lambda would need nvcc's --extended-lambda.
void test_fuse_lambda() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1.0, 2.0, 3.0);
  SparseMat<double, IntType, 3, 3, 0, 1, 2> b(4.0, 5.0, 6.0);
  SparseMat<double, IntType, 3, 3, 0, 4, 8> c(10.0, 20.0, 30.0);

  auto fused = SparseLinearAlgebra::fuse(
      [](double x, double y, double z) { return (2.0 * x) + (3.0 * y) - z; }, a, b, c);
  check(fused == a.scale(2.0).add(b.scale(3.0)).subtract(c),
        "fuse: lambda matches the eager equivalent");

  // A capturing lambda works too — the function object is taken by reference
  // and called per element, so captured state is live throughout.
  const double weight = 0.5;
  auto weighted = SparseLinearAlgebra::fuse([weight](double x) { return x * weight; }, a);
  check_near(weighted.get(2, 2), 1.5, "fuse: capturing lambda");

  // Large pattern: 40x40 tridiagonal has 118 stored values, past
  // kUnrollChunkSize (64), so this takes the chunk-splitting path rather than
  // a single fold.
  auto big = SparseLinearAlgebra::tridiagonal<double, IntType, 40>(-1.0, 4.0, -1.0);
  auto doubled = SparseLinearAlgebra::fuse([](double x) { return 2.0 * x; }, big);
  check_eq(decltype(doubled)::nonZeroCount,
           decltype(big)::nonZeroCount,
           "fuse: large pattern is preserved");
  check_near(doubled.trace(), 2.0 * big.trace(), "fuse: large chunked fill is correct");
  check_near(doubled.get(39, 38), -2.0, "fuse: last chunk is filled correctly");
}

void test_large_sparse_stress() {
  using namespace stress_test;

  Matrix100 a(make_values());
  std::array<double, kN> ones{};
  ones.fill(1.0);
  Vector100 b(ones);

  // add: A+A should double every stored entry.
  auto sum = a.add(a);
  check_eq(decltype(sum)::nonZeroCount, IntType(kNnz), "stress: add nonZeroCount");
  check_near(sum.template get<0, 0>(), 8.0, "stress: add (0,0) = 4+4");
  check_near(sum.template get<0, 1>(), -2.0, "stress: add (0,1) = -1+-1");

  // multiply: (A*A) is pentadiagonal; spot-check a couple of entries against
  // the hand-derived tridiagonal recurrence.
  auto product = a.mult(a);
  // (A*A)(0,0) = A(0,0)*A(0,0) + A(0,1)*A(1,0) = 4*4 + (-1)*(-1) = 17
  check_near(product.template get<0, 0>(), 17.0, "stress: multiply (0,0)");
  // (A*A)(0,2) = A(0,1)*A(1,2) = (-1)*(-1) = 1
  check_near(product.template get<0, 2>(), 1.0, "stress: multiply (0,2)");

  // lu_solve and cholesky_solve: verify Ax=b by checking the residual, not
  // exact values (no independent closed-form solution at this size).
  auto lu_result = SparseLinearAlgebra::lu_solve(a, b);
  check(lu_result.ok(), "stress: lu_solve reports success");
  auto lu_residual = a.mult(lu_result.value()).subtract(b);
  for (int i = 0; i < kN; ++i) {
    check_near(lu_residual.get(i, 0), 0.0, "stress: lu_solve residual", 1e-6);
  }

  auto chol_result = SparseLinearAlgebra::cholesky_solve(a, b);
  check(chol_result.ok(), "stress: cholesky_solve reports success");
  auto chol_residual = a.mult(chol_result.value()).subtract(b);
  for (int i = 0; i < kN; ++i) {
    check_near(chol_residual.get(i, 0), 0.0, "stress: cholesky_solve residual", 1e-6);
  }
}

#endif  // SPARSEMAT_TEST_NO_MAIN

// --- Comparison operators ---

SPARSEMAT_HD void test_comparison() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1.0, 2.0, 3.0);
  SparseMat<double, IntType, 3, 3, 0, 4, 8> same(1.0, 2.0, 3.0);
  SparseMat<double, IntType, 3, 3, 0, 4, 8> different(1.0, 2.0, 4.0);

  check(a == same, "operator==: identical matrices compare equal");
  check(!(a == different), "operator==: differing values compare unequal");
  check(a != different, "operator!=: differing values");
  check(!(a != same), "operator!=: identical matrices");

  // Comparison is on values, not patterns: a densified copy stores explicit
  // zeros where the original has structural ones, and must still compare equal.
  check(a == a.dense(), "operator==: matches a densified copy of itself");
  check(a.dense() == a, "operator==: is symmetric across differing patterns");

  // A wider pattern whose extra entries are non-zero must NOT compare equal.
  SparseMat<double, IntType, 3, 3, 0, 1, 4, 8> wider(1.0, 5.0, 2.0, 3.0);
  check(a != wider, "operator==: extra non-zero in a wider pattern compares unequal");

  // approx_equal tolerates small differences that == rejects.
  SparseMat<double, IntType, 3, 3, 0, 4, 8> nudged(1.0, 2.0, 3.0 + 1e-9);
  check(a != nudged, "operator==: is exact");
  check(SparseLinearAlgebra::approx_equal(a, nudged), "approx_equal: tolerates 1e-9");
  check(!SparseLinearAlgebra::approx_equal(a, different, 1e-6),
        "approx_equal: still rejects a real difference");
}

// --- Arithmetic operators ---

SPARSEMAT_HD void test_operators() {
  SparseMat<double, IntType, 2, 2, 0, 3> a(2.0, 4.0);
  SparseMat<double, IntType, 2, 2, 0, 3> b(1.0, 1.0);

  check_near((-a).get(0, 0), -2.0, "operator-: unary negation (0,0)");
  check_near((-a).get(1, 1), -4.0, "operator-: unary negation (1,1)");

  check_near((a * 3.0).get(0, 0), 6.0, "operator*: scalar on the right");
  check_near((3.0 * a).get(0, 0), 6.0, "operator*: scalar on the left");
  check_near((a / 2.0).get(1, 1), 2.0, "operator/: scalar divide");

  auto compound = a;
  compound += b;
  check_near(compound.get(0, 0), 3.0, "operator+=: (0,0)");
  check_near(compound.get(1, 1), 5.0, "operator+=: (1,1)");

  compound -= b;
  check(compound == a, "operator-=: undoes operator+=");

  compound *= 2.0;
  check_near(compound.get(0, 0), 4.0, "operator*=: scalar multiply in place");
  compound /= 2.0;
  check(compound == a, "operator/=: undoes operator*=");

  // A narrower right-hand pattern is fine — it is a subset of the left's.
  SparseMat<double, IntType, 2, 2, 0> narrow(10.0);
  auto widened = a;
  widened += narrow;
  check_near(widened.get(0, 0), 12.0, "operator+=: accepts a subset pattern");
  check_near(widened.get(1, 1), 4.0, "operator+=: leaves untouched entries alone");
}

// --- Iteration ---

SPARSEMAT_HD void test_iteration() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1.0, 2.0, 3.0);

  check_eq(a.size(), std::size_t(3), "size(): counts stored values");
  check(!a.empty(), "empty(): false for a matrix with stored values");

  double sum = 0.0;
  for (const auto& value : a) {
    sum += value;
  }
  check_near(sum, 6.0, "begin()/end(): iterates every stored value");

  // entries() pairs each value with its position.
  double weighted = 0.0;
  int count = 0;
  for (auto entry : a.entries()) {
    check_eq(entry.row, entry.col, "entries(): this matrix is diagonal");
    weighted += entry.value * static_cast<double>(entry.row);
    ++count;
  }
  check_eq(count, 3, "entries(): yields one entry per stored value");
  check_near(weighted, (2.0 * 1) + (3.0 * 2), "entries(): values pair with the right positions");

  // operator() mirrors get(i, j), including for structural zeros.
  check_near(a(0, 0), 1.0, "operator(): reads a stored value");
  check_near(a(0, 1), 0.0, "operator(): returns zero for a structural zero");
  check_near(a(9, 9), 0.0, "operator(): returns zero when out of bounds");
}

// --- to_array / convert ---

SPARSEMAT_HD void test_to_array() {
  SparseMat<double, IntType, 2, 3, 0, 4> a(7.0, 9.0);
  auto flat = a.to_array();
  check_eq(flat.size(), std::size_t(6), "to_array(): length is rows*cols");
  check_near(flat[0], 7.0, "to_array(): stored value at flat index 0");
  check_near(flat[4], 9.0, "to_array(): stored value at flat index 4");
  check_near(flat[1], 0.0, "to_array(): structural zero written as 0");
  check_near(flat[5], 0.0, "to_array(): trailing structural zero written as 0");
}

SPARSEMAT_HD void test_convert() {
  SparseMat<float, IntType, 3, 3, 0, 4, 8> f(1.5F, 2.5F, 3.5F);
  auto d = f.template convert<double>();

  check(std::is_same_v<typename decltype(d)::DataType, double>, "convert(): changes DataType");
  check_eq(decltype(d)::nonZeroCount, IntType(3), "convert(): preserves the sparsity pattern");
  check_eq(decltype(d)::rows, IntType(3), "convert(): preserves dimensions");
  check_near(d.get(0, 0), 1.5, "convert(): preserves values exactly when widening");
  check_near(d.get(2, 2), 3.5, "convert(): preserves the last value");

  // Round-tripping back through float is lossless for these exact binary
  // fractions, so the values must come back identical.
  auto round_tripped = d.template convert<float>();
  check(round_tripped == f, "convert(): float -> double -> float round-trips");

  // The whole point: a converted matrix can be combined with the target type.
  SparseMat<double, IntType, 3, 3, 0, 4, 8> target(0.5, 0.5, 0.5);
  auto sum = d.add(target);
  check_near(sum.get(0, 0), 2.0, "convert(): result combines with the target DataType");

  // The free-function spelling behaves identically.
  auto free_converted = SparseLinearAlgebra::convert<double>(f);
  check(free_converted == d, "convert(): free function matches the member");
}

// --- power<N> ---

SPARSEMAT_HD void test_power() {
  SparseMat<double, IntType, 2, 2, 0, 3> diag(2.0, 3.0);

  auto p1 = SparseLinearAlgebra::power<decltype(diag), 1>(diag);
  check(p1 == diag, "power<1>: returns the matrix unchanged");

  auto p2 = SparseLinearAlgebra::power<decltype(diag), 2>(diag);
  check_near(p2.get(0, 0), 4.0, "power<2>: (0,0) = 2^2");
  check_near(p2.get(1, 1), 9.0, "power<2>: (1,1) = 3^2");

  auto p3 = SparseLinearAlgebra::power<decltype(diag), 3>(diag);
  check_near(p3.get(0, 0), 8.0, "power<3>: (0,0) = 2^3");
  check_near(p3.get(1, 1), 27.0, "power<3>: (1,1) = 3^3");

  // A non-diagonal case, where the sparsity pattern actually widens.
  SparseMat<double, IntType, 3, 3, 1, 5> shift_up(1.0, 1.0);  // (0,1) and (1,2)
  auto shifted = SparseLinearAlgebra::power<decltype(shift_up), 2>(shift_up);
  check_near(shifted.get(0, 2), 1.0, "power<2>: nilpotent shift lands at (0,2)");
  check_near(shifted.get(0, 1), 0.0, "power<2>: original entry is gone");
}

// --- Near-singular pivots ---
//
// Neither factorization pivots, so a merely *tiny* pivot is as fatal as an
// exactly-zero one: it divides through and yields meaningless multipliers.
// These must be reported via ok() rather than silently returning garbage.

SPARSEMAT_HD void test_near_singular_pivots() {
  // Very nearly linearly dependent rows: elimination leaves a second pivot
  // around 1e-18 relative to the entries it came from.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> nearly(1.0, 2.0, 1.0, 2.0 + 1e-18);
  SparseMat<double, IntType, 2, 1, 0, 1> b(1.0, 2.0);

  auto lu = SparseLinearAlgebra::lu_solve(nearly, b);
  check(!lu.ok(), "lu_solve: reports failure on a negligible (not exactly zero) pivot");

  // Cholesky: SPD on paper, but the second pivot cancels to nothing.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> nearly_pd(1.0, 1.0, 1.0, 1.0 + 1e-18);
  auto chol = SparseLinearAlgebra::cholesky_factorize(nearly_pd);
  check(!chol.ok(), "cholesky_factorize: reports failure on a negligible pivot");

  // A genuinely well-conditioned matrix of similar magnitude must still pass,
  // i.e. the threshold is not so loose that it rejects usable systems.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> fine(2.0, 1.0, 1.0, 2.0);
  auto ok_lu = SparseLinearAlgebra::lu_solve(fine, b);
  check(ok_lu.ok(), "lu_solve: still succeeds on a well-conditioned matrix");
  auto ok_chol = SparseLinearAlgebra::cholesky_factorize(fine);
  check(ok_chol.ok(), "cholesky_factorize: still succeeds on a well-conditioned matrix");
}

// --- Large dense() ---
//
// dense() used to recurse once per element, blowing the compiler's default
// 900-deep instantiation budget at 32x32 (1024 elements). This is a
// regression guard for that ceiling as much as a correctness check.

SPARSEMAT_HD void test_dense_large() {
  SparseMat<double, IntType, 32, 32, 0, 33, 1023> sparse32(1.0, 2.0, 3.0);
  auto densified = sparse32.dense();

  check_eq(decltype(densified)::nonZeroCount,
           IntType(1024),
           "dense() 32x32: every position stored");
  check_near(densified.get(0, 0), 1.0, "dense() 32x32: preserves (0,0)");
  check_near(densified.get(1, 1), 2.0, "dense() 32x32: preserves (1,1)");
  check_near(densified.get(31, 31), 3.0, "dense() 32x32: preserves (31,31)");
  check_near(densified.get(5, 7), 0.0, "dense() 32x32: structural zero is an explicit 0");
  check(densified == sparse32, "dense() 32x32: equals the matrix it came from");

  // trace() and set_diagonal() had the same linear-recursion shape.
  check_near(sparse32.trace(), 6.0, "trace() 32x32: sums the stored diagonal");
  sparse32.set_diagonal(5.0);
  check_near(sparse32.trace(), 15.0, "set_diagonal() 32x32: writes every stored diagonal entry");
}

// --- Pattern builders ---

SPARSEMAT_HD void test_make_pattern() {
  // Positions given out of order; the resulting type must be canonical.
  constexpr auto shape = std::array{SparseLinearAlgebra::SparsePosition{1, 1},
                                    SparseLinearAlgebra::SparsePosition{0, 0},
                                    SparseLinearAlgebra::SparsePosition{0, 1}};
  auto a = SparseLinearAlgebra::make_pattern<double, IntType, 2, 2, shape>();

  check_eq(decltype(a)::nonZeroCount, IntType(3), "make_pattern: stores one value per position");
  check_near(a.frobenius(), 0.0, "make_pattern: every value starts at zero");

  // Declaring order must not affect the type — the whole point of sorting.
  constexpr auto reordered = std::array{SparseLinearAlgebra::SparsePosition{0, 1},
                                        SparseLinearAlgebra::SparsePosition{1, 1},
                                        SparseLinearAlgebra::SparsePosition{0, 0}};
  auto b = SparseLinearAlgebra::make_pattern<double, IntType, 2, 2, reordered>();
  check(std::is_same_v<decltype(a), decltype(b)>,
        "make_pattern: type is canonical regardless of declaration order");
  check(a == b, "make_pattern: both orderings start out equal (all zero)");

  // The pattern is what was asked for, and filling it works.
  check(a.set(0, 0, 4.0), "make_pattern: (0,0) is storable");
  check(a.set(0, 1, 5.0), "make_pattern: (0,1) is storable");
  check(a.set(1, 1, 2.0), "make_pattern: (1,1) is storable");
  check(!a.set(1, 0, 9.0), "make_pattern: (1,0) was not requested, so is not storable");
  check_near(a.get(0, 0), 4.0, "make_pattern: filled value reads back");
  check_near(a.get(1, 0), 0.0, "make_pattern: unrequested position reads as zero");

  // Exact duplicates collapse rather than producing an invalid pattern.
  constexpr auto duplicated = std::array{SparseLinearAlgebra::SparsePosition{0, 0},
                                         SparseLinearAlgebra::SparsePosition{0, 0},
                                         SparseLinearAlgebra::SparsePosition{1, 1}};
  auto deduped = SparseLinearAlgebra::make_pattern<double, IntType, 2, 2, duplicated>();
  check_eq(decltype(deduped)::nonZeroCount, IntType(2), "make_pattern: collapses duplicates");
  check(deduped.set(0, 0, 1.0) && deduped.set(1, 1, 2.0),
        "make_pattern: both surviving positions are storable");
  check_near(deduped.trace(), 3.0, "make_pattern: deduplicated pattern holds its values");
}

SPARSEMAT_HD void test_symmetric_from_lower() {
  constexpr auto lower = std::array{SparseLinearAlgebra::SparsePosition{0, 0},
                                    SparseLinearAlgebra::SparsePosition{1, 0},
                                    SparseLinearAlgebra::SparsePosition{1, 1},
                                    SparseLinearAlgebra::SparsePosition{2, 2}};
  auto a = SparseLinearAlgebra::symmetric_from_lower<double, IntType, 3, lower>();

  // (1,0) gains its mirror (0,1); the three diagonal entries do not double up.
  check_eq(decltype(a)::nonZeroCount, IntType(5), "symmetric_from_lower: mirrors off-diagonals");
  check(a.is_structurally_symmetric(), "symmetric_from_lower: pattern is symmetric");
  check(a.set(0, 1, 7.0), "symmetric_from_lower: mirrored position is storable");
  check(a.set(2, 2, 1.0), "symmetric_from_lower: diagonal position is storable");
  check(!a.set(0, 2, 1.0), "symmetric_from_lower: unrelated position is not storable");
}

SPARSEMAT_HD void test_banded() {
  // Tridiagonal 5x5: 5 diagonal + 4 sub + 4 super.
  auto tri = SparseLinearAlgebra::banded<double, IntType, 5, 5, 1, 1>();
  check_eq(decltype(tri)::nonZeroCount, IntType(13), "banded<1,1>: 5x5 has 13 stored values");
  check(tri.set(0, 0, 1.0), "banded<1,1>: diagonal is stored");
  check(tri.set(1, 0, 1.0), "banded<1,1>: sub-diagonal is stored");
  check(tri.set(0, 1, 1.0), "banded<1,1>: super-diagonal is stored");
  check(!tri.set(0, 2, 1.0), "banded<1,1>: second super-diagonal is not stored");

  // tridiagonal<N>() is the same type as banded<N,N,1,1>().
  auto named = SparseLinearAlgebra::tridiagonal<double, IntType, 5>();
  check(std::is_same_v<decltype(tri), decltype(named)>,
        "tridiagonal<N>: same type as banded<N,N,1,1>");
  check_near(named.frobenius(), 0.0, "tridiagonal<N>: starts out zero-valued");

  // Band widths of zero give a plain diagonal.
  auto diag = SparseLinearAlgebra::banded<double, IntType, 4, 4, 0, 0>();
  check_eq(decltype(diag)::nonZeroCount, IntType(4), "banded<0,0>: is diagonal");
  check(decltype(diag)::is_structurally_lower_triangular() &&
            decltype(diag)::is_structurally_upper_triangular(),
        "banded<0,0>: is both upper and lower triangular");
  diag.set_diagonal(3.0);
  check_near(diag.trace(), 12.0, "banded<0,0>: every diagonal position is storable");

  // Asymmetric bands, and the non-square case where the band gets clipped.
  auto lower_only = SparseLinearAlgebra::banded<double, IntType, 4, 4, 2, 0>();
  check_eq(decltype(lower_only)::nonZeroCount, IntType(9), "banded<2,0>: 4 + 3 + 2");
  check(decltype(lower_only)::is_structurally_lower_triangular(),
        "banded<2,0>: is lower triangular");
  check(lower_only.set(3, 1, 1.0), "banded<2,0>: second sub-diagonal is stored");
  check(!lower_only.set(3, 0, 1.0), "banded<2,0>: third sub-diagonal is not");
  auto wide = SparseLinearAlgebra::banded<double, IntType, 2, 4, 0, 1>();
  check_eq(decltype(wide)::nonZeroCount, IntType(4), "banded: non-square band is clipped to shape");
  check(wide.set(1, 2, 5.0), "banded: in-band position of a non-square matrix is stored");
  check(!wide.set(1, 3, 5.0), "banded: out-of-band position of a non-square matrix is not");

  // The value-filling overload writes each band correctly.
  auto filled = SparseLinearAlgebra::tridiagonal<double, IntType, 4>(-1.0, 4.0, -2.0);
  check_near(filled.get(0, 0), 4.0, "tridiagonal(sub,diag,super): diagonal");
  check_near(filled.get(1, 0), -1.0, "tridiagonal(sub,diag,super): sub-diagonal");
  check_near(filled.get(0, 1), -2.0, "tridiagonal(sub,diag,super): super-diagonal");
  check_near(filled.trace(), 16.0, "tridiagonal(sub,diag,super): trace is 4*diag");
}

// --- Block diagonal ---

SPARSEMAT_HD void test_block_diagonal() {
  SparseMat<double, IntType, 2, 2, 0, 3> a(1.0, 2.0);
  SparseMat<double, IntType, 2, 2, 0, 1, 3> b(3.0, 4.0, 5.0);
  auto composed = SparseLinearAlgebra::block_diagonal(a, b);

  check_eq(decltype(composed)::rows, IntType(4), "block_diagonal: rows add");
  check_eq(decltype(composed)::cols, IntType(4), "block_diagonal: cols add");
  // The defining property: stored count is the *sum*, never more.
  check_eq(decltype(composed)::nonZeroCount,
           IntType(decltype(a)::nonZeroCount + decltype(b)::nonZeroCount),
           "block_diagonal: stored count is the sum of both blocks'");

  check_near(composed.get(0, 0), 1.0, "block_diagonal: A lands top-left");
  check_near(composed.get(1, 1), 2.0, "block_diagonal: A's second entry");
  check_near(composed.get(2, 2), 3.0, "block_diagonal: B lands bottom-right");
  check_near(composed.get(2, 3), 4.0, "block_diagonal: B's off-diagonal entry is offset correctly");
  check_near(composed.get(3, 3), 5.0, "block_diagonal: B's last entry");
  check_near(composed.get(0, 2), 0.0, "block_diagonal: top-right block is zero");
  check_near(composed.get(3, 0), 0.0, "block_diagonal: bottom-left block is zero");
  check(!composed.set(0, 2, 1.0), "block_diagonal: off-diagonal block is structurally zero");

  // The point of composing: one solve over both blocks gives the same answers
  // as solving each block separately. The stacked right-hand side is written
  // out as a single 4x1 column on purpose — block_diagonal() on two *column
  // vectors* would produce a 4x2 matrix (each vector in its own column), which
  // is a valid block solve but not the single-RHS system being checked here.
  SparseMat<double, IntType, 2, 2, 0, 2, 3> la(2.0, 1.0, 4.0);
  SparseMat<double, IntType, 2, 2, 0, 2, 3> lb(5.0, 2.0, 1.0);
  SparseMat<double, IntType, 2, 1, 0, 1> ba(2.0, 8.0);
  SparseMat<double, IntType, 2, 1, 0, 1> bb(10.0, 3.0);
  SparseMat<double, IntType, 4, 1, 0, 1, 2, 3> stacked(2.0, 8.0, 10.0, 3.0);

  auto joint = SparseLinearAlgebra::block_diagonal(la, lb).solve(stacked);
  auto solo_a = la.solve(ba);
  auto solo_b = lb.solve(bb);
  check(joint.ok() && solo_a.ok() && solo_b.ok(), "block_diagonal: joint and separate solves ok");
  check_near(joint.value().get(0, 0),
             solo_a.value().get(0, 0),
             "block_diagonal: joint solve matches block A, row 0");
  check_near(joint.value().get(1, 0),
             solo_a.value().get(1, 0),
             "block_diagonal: joint solve matches block A, row 1");
  check_near(joint.value().get(2, 0),
             solo_b.value().get(0, 0),
             "block_diagonal: joint solve matches block B, row 0");
  check_near(joint.value().get(3, 0),
             solo_b.value().get(1, 0),
             "block_diagonal: joint solve matches block B, row 1");

  // Two column vectors composed block-diagonally give a 4x2 block RHS, with
  // each vector confined to its own column — worth pinning down, since it is
  // the shape the check above deliberately avoids.
  auto side_by_side = SparseLinearAlgebra::block_diagonal(ba, bb);
  check_eq(decltype(side_by_side)::cols, IntType(2), "block_diagonal: two vectors give 2 columns");
  check_near(side_by_side.get(0, 0), 2.0, "block_diagonal: first vector stays in column 0");
  check_near(side_by_side.get(2, 1), 10.0, "block_diagonal: second vector moves to column 1");
  check_near(side_by_side.get(2, 0), 0.0, "block_diagonal: second vector's column-0 entry is zero");
}

// --- Determinant ---

SPARSEMAT_HD void test_determinant() {
  // 2x2: det = 1*4 - 2*3 = -2.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> general(1.0, 2.0, 3.0, 4.0);
  auto det = general.determinant();
  check(det.ok(), "determinant: reports success on a non-singular matrix");
  check_near(det.value(), -2.0, "determinant: 2x2 general");

  // Diagonal / triangular matrices take the short path (product of diagonal).
  SparseMat<double, IntType, 3, 3, 0, 4, 8> diag(2.0, 3.0, 4.0);
  check_near(diag.determinant().value(), 24.0, "determinant: diagonal is the product");
  SparseMat<double, IntType, 3, 3, 0, 3, 4, 6, 7, 8> lower(2.0, 1.0, 3.0, 1.0, 1.0, 4.0);
  check_near(lower.determinant().value(), 24.0, "determinant: lower triangular is the product");

  // A structurally zero diagonal entry means an exact zero determinant, which
  // is a trustworthy answer rather than a failure.
  SparseMat<double, IntType, 3, 3, 0, 8> missing_diag(2.0, 4.0);
  auto missing = missing_diag.determinant();
  check_near(missing.value(), 0.0, "determinant: structurally zero pivot gives zero");
  check(missing.ok(), "determinant: an exact zero determinant is a success, not a failure");

  // Singular via linear dependence: reported through ok().
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> singular(1.0, 2.0, 2.0, 4.0);
  auto bad = singular.determinant();
  check(!bad.ok(), "determinant: reports failure on a singular matrix");

  // Identity, and a scaling identity: det(cI) = c^n.
  auto identity = SparseMat<double, IntType, 4, 4>::identity();
  check_near(identity.determinant().value(), 1.0, "determinant: identity is 1");
  check_near(identity.scale(2.0).determinant().value(), 16.0, "determinant: det(2I) = 2^4");

  // det(A*B) == det(A)*det(B), on matrices where LU is stable.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> lhs(3.0, 1.0, 1.0, 2.0);
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> rhs(2.0, 1.0, 1.0, 3.0);
  check_near(lhs.mult(rhs).determinant().value(),
             lhs.determinant().value() * rhs.determinant().value(),
             "determinant: det(A*B) == det(A)*det(B)");
}

// --- Inverse ---

SPARSEMAT_HD void test_inverse() {
  // inv([[4,7],[2,6]]) = [[0.6,-0.7],[-0.2,0.4]]
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> a(4.0, 7.0, 2.0, 6.0);
  auto inv = a.inverse();
  check(inv.ok(), "inverse: reports success");
  check_near(inv.value().get(0, 0), 0.6, "inverse: (0,0)");
  check_near(inv.value().get(0, 1), -0.7, "inverse: (0,1)");
  check_near(inv.value().get(1, 0), -0.2, "inverse: (1,0)");
  check_near(inv.value().get(1, 1), 0.4, "inverse: (1,1)");

  // The defining property, in both orders.
  const auto identity2 = SparseMat<double, IntType, 2, 2>::identity();
  check(SparseLinearAlgebra::approx_equal(a.mult(inv.value()), identity2, 1e-9),
        "inverse: A * A_inv == I");
  check(SparseLinearAlgebra::approx_equal(inv.value().mult(a), identity2, 1e-9),
        "inverse: A_inv * A == I");

  // A larger, sparser case: inverting a tridiagonal matrix (whose inverse is
  // dense) still reproduces the identity.
  auto tri = SparseLinearAlgebra::tridiagonal<double, IntType, 4>(-1.0, 4.0, -1.0);
  auto tri_inv = tri.inverse();
  check(tri_inv.ok(), "inverse: succeeds on a 4x4 tridiagonal matrix");
  check(SparseLinearAlgebra::approx_equal(tri.mult(tri_inv.value()),
                                          SparseMat<double, IntType, 4, 4>::identity(),
                                          1e-9),
        "inverse: tridiagonal A * A_inv == I");

  // Singular matrices are reported, not silently inverted.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> singular(1.0, 2.0, 2.0, 4.0);
  check(!singular.inverse().ok(), "inverse: reports failure on a singular matrix");

  // cholesky_inverse on an SPD matrix agrees with the general inverse.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> spd(4.0, 1.0, 1.0, 3.0);
  auto chol_inv = SparseLinearAlgebra::cholesky_inverse(spd);
  check(chol_inv.ok(), "cholesky_inverse: reports success on an SPD matrix");
  check(SparseLinearAlgebra::approx_equal(spd.mult(chol_inv.value()), identity2, 1e-9),
        "cholesky_inverse: A * A_inv == I");
  check(SparseLinearAlgebra::approx_equal(chol_inv.value(), spd.inverse().value(), 1e-9),
        "cholesky_inverse: agrees with the general inverse");

  // A non-SPD matrix is rejected rather than producing a wrong answer.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> not_pd(1.0, 2.0, 2.0, 1.0);
  check(!SparseLinearAlgebra::cholesky_inverse(not_pd).ok(),
        "cholesky_inverse: reports failure on a non-positive-definite matrix");
}

// --- Least squares ---

SPARSEMAT_HD void test_least_squares() {
  // Overdetermined and *consistent*: fitting y = 1 + 2x through three points
  // that lie exactly on that line must recover it with zero residual.
  SparseMat<double, IntType, 3, 2, 0, 1, 2, 3, 4, 5> design(1.0, 0.0, 1.0, 1.0, 1.0, 2.0);
  SparseMat<double, IntType, 3, 1, 0, 1, 2> exact(1.0, 3.0, 5.0);
  auto fit = design.least_squares_solve(exact);
  check(fit.ok(), "least_squares_solve: reports success on a full-rank system");
  check_near(fit.value().get(0, 0), 1.0, "least_squares_solve: recovers the intercept");
  check_near(fit.value().get(1, 0), 2.0, "least_squares_solve: recovers the slope");
  check_near(SparseLinearAlgebra::residual(design, fit.value(), exact).frobenius(),
             0.0,
             "least_squares_solve: consistent system has zero residual",
             1e-9);

  // Overdetermined and *inconsistent*: no exact solution exists, so check the
  // known minimiser for y = (1, 3, 6) — slope 5/2, intercept 5/6.
  SparseMat<double, IntType, 3, 1, 0, 1, 2> noisy(1.0, 3.0, 6.0);
  auto approx = design.least_squares_solve(noisy);
  check(approx.ok(), "least_squares_solve: succeeds on an inconsistent system");
  check_near(approx.value().get(0, 0), 5.0 / 6.0, "least_squares_solve: least-squares intercept");
  check_near(approx.value().get(1, 0), 5.0 / 2.0, "least_squares_solve: least-squares slope");

  // It really is the minimiser: perturbing either coefficient must not reduce
  // the residual norm. This is the property that distinguishes a least-squares
  // solve from any old solve, so it is worth testing directly rather than
  // trusting the closed form above.
  const auto best = SparseLinearAlgebra::residual(design, approx.value(), noisy).frobenius();
  auto perturbed_intercept = approx.value();
  perturbed_intercept.set(0, 0, approx.value().get(0, 0) + 0.1);
  check(SparseLinearAlgebra::residual(design, perturbed_intercept, noisy).frobenius() > best,
        "least_squares_solve: perturbing the intercept increases the residual");
  auto perturbed_slope = approx.value();
  perturbed_slope.set(1, 0, approx.value().get(1, 0) + 0.1);
  check(SparseLinearAlgebra::residual(design, perturbed_slope, noisy).frobenius() > best,
        "least_squares_solve: perturbing the slope increases the residual");

  // Underdetermined: x0 + x1 = 2 has infinitely many solutions; the
  // minimum-norm one is (1, 1).
  SparseMat<double, IntType, 1, 2, 0, 1> single_equation(1.0, 1.0);
  SparseMat<double, IntType, 1, 1, 0> target(2.0);
  auto min_norm = single_equation.least_squares_solve(target);
  check(min_norm.ok(), "least_squares_solve: succeeds on an underdetermined system");
  check_near(min_norm.value().get(0, 0), 1.0, "least_squares_solve: minimum-norm x0");
  check_near(min_norm.value().get(1, 0), 1.0, "least_squares_solve: minimum-norm x1");
  // It must still satisfy the equation exactly, and beat any other solution on
  // norm — (2, 0) also satisfies it, but has a larger norm.
  check_near(SparseLinearAlgebra::residual(single_equation, min_norm.value(), target).frobenius(),
             0.0,
             "least_squares_solve: underdetermined solution is exact",
             1e-9);
  check(min_norm.value().frobenius() < 2.0,
        "least_squares_solve: underdetermined solution has smaller norm than (2,0)");

  // Rank-deficient: the normal-equations matrix is singular, and that is
  // reported rather than returning nonsense with ok() == true.
  SparseMat<double, IntType, 2, 2, 0, 1, 2, 3> rank_deficient(1.0, 1.0, 1.0, 1.0);
  SparseMat<double, IntType, 2, 1, 0, 1> rhs(1.0, 2.0);
  check(!rank_deficient.least_squares_solve(rhs).ok(),
        "least_squares_solve: reports failure on a rank-deficient system");
}

// --- Fused element-wise operations ---
//
// These use function objects rather than lambdas because run_all_tests() also
// runs inside a CUDA kernel: a struct with a SPARSEMAT_HD operator() is
// callable from device code on every toolchain, whereas a host-defined lambda
// would need nvcc's --extended-lambda. The host-only section below covers the
// lambda spelling, which is what most callers will actually write.

struct FuseScaledSum {
  SPARSEMAT_HD double operator()(double x, double y, double z) const {
    return (2.0 * x) + (3.0 * y) - z;
  }
};
struct FuseProduct {
  SPARSEMAT_HD double operator()(double x, double y) const { return x * y; }
};
struct FuseSquare {
  SPARSEMAT_HD double operator()(double x) const { return x * x; }
};
struct FuseSum4 {
  SPARSEMAT_HD double operator()(double a, double b, double c, double d) const {
    return a + b + c + d;
  }
};

SPARSEMAT_HD void test_fuse() {
  SparseMat<double, IntType, 3, 3, 0, 4, 8> a(1.0, 2.0, 3.0);  // diagonal
  SparseMat<double, IntType, 3, 3, 0, 1, 2> b(4.0, 5.0, 6.0);  // first row
  SparseMat<double, IntType, 3, 3, 0, 4, 8> c(10.0, 20.0, 30.0);

  // The defining property: a fused chain equals the eager one element for
  // element, and has the same result pattern — it just skips the temporaries.
  auto eager = a.scale(2.0).add(b.scale(3.0)).subtract(c);
  auto fused = SparseLinearAlgebra::fuse(FuseScaledSum{}, a, b, c);
  check(std::is_same_v<decltype(fused), decltype(eager)>,
        "fuse: result type matches the eager equivalent");
  check(fused == eager, "fuse: result matches the eager equivalent");
  check_near(fused.get(0, 0), (2 * 1.0) + (3 * 4.0) - 10.0, "fuse: (0,0) where all three overlap");
  check_near(fused.get(0, 1), 3 * 5.0, "fuse: (0,1) where only b is stored");
  check_near(fused.get(1, 1), (2 * 2.0) - 20.0, "fuse: (1,1) where only a and c are stored");

  // Arity is not fixed: one operand, and four (including a repeat).
  auto squared = SparseLinearAlgebra::fuse(FuseSquare{}, a);
  check_near(squared.get(1, 1), 4.0, "fuse: unary");
  check_eq(decltype(squared)::nonZeroCount, IntType(3), "fuse: unary preserves the pattern");
  auto sum4 = SparseLinearAlgebra::fuse(FuseSum4{}, a, b, c, a);
  check_near(sum4.get(0, 0), 1.0 + 4.0 + 10.0 + 1.0, "fuse: four operands, one repeated");

  // Union (the default) is a superset: it keeps a position stored by any
  // operand, so the result pattern is the union of a, b and c's.
  check_eq(decltype(fused)::nonZeroCount, IntType(5), "fuse: Union pattern is the union");

  // Intersection stores only positions every operand has — the product
  // pattern, which is exactly what hadamard computes.
  auto intersected =
      SparseLinearAlgebra::fuse<SparseLinearAlgebra::FusePattern::Intersection>(FuseProduct{},
                                                                                a,
                                                                                b);
  auto hadamard = a.hadamard(b);
  check(std::is_same_v<decltype(intersected), decltype(hadamard)>,
        "fuse<Intersection>: result type matches hadamard");
  check(intersected == hadamard, "fuse<Intersection>: matches hadamard");
  check_eq(decltype(intersected)::nonZeroCount,
           IntType(1),
           "fuse<Intersection>: only (0,0) is stored in both");

  // Union on the same product function gives the same *values* but a wider
  // pattern — the extra positions hold explicit zeros. Worth pinning down,
  // since it is the documented cost of the safe default.
  auto union_product = SparseLinearAlgebra::fuse(FuseProduct{}, a, b);
  check(union_product == hadamard, "fuse: Union and Intersection agree on values");
  check(decltype(union_product)::nonZeroCount > decltype(intersected)::nonZeroCount,
        "fuse: Union stores explicit zeros the Intersection pattern omits");
  check_near(union_product.get(1, 1), 0.0, "fuse: Union's extra position holds an explicit zero");

  // Operands whose patterns are disjoint still get one value each.
  SparseMat<double, IntType, 2, 2, 0> only_first(7.0);
  SparseMat<double, IntType, 2, 2, 3> only_last(9.0);
  auto disjoint = SparseLinearAlgebra::fuse(FuseProduct{}, only_first, only_last);
  check_eq(decltype(disjoint)::nonZeroCount, IntType(2), "fuse: disjoint operands union to 2");
  check_near(disjoint.get(0, 0), 0.0, "fuse: structurally zero operand contributes 0");
  check_near(disjoint.get(1, 1), 0.0, "fuse: and again at the other position");

  // Non-square, and a pattern big enough to exercise the chunked unrolling
  // (kUnrollChunkSize is 64, so 100 stored values takes the splitting path).
  SparseMat<double, IntType, 2, 3, 0, 1, 2, 3, 4, 5> wide(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  auto scaled_wide = SparseLinearAlgebra::fuse(FuseSquare{}, wide);
  check_near(scaled_wide.get(1, 2), 36.0, "fuse: non-square operand");
}

// --- Run everything ---
//
// Shared by both runners: test_sparsemat's main() below calls this directly
// on the host, and test_sparsemat_gpu.cu's __global__ kernel calls it from
// the device instead.

SPARSEMAT_HD inline void run_all_tests() {
  test_construction();
  test_copy_construction();
  test_get();
  test_set();
  test_runtime_bounds();
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
  test_normalize_zero_returns_zero();
  test_normalize_inplace_zero_noop();
  test_dense();
  test_trace();
  test_dot();
  test_axpy();
  test_identity();
  test_make_sparse_matrix();
  test_kronecker();
  test_kronecker_large();
  test_triangular();
  test_forward_solve();
  test_backward_solve();
  test_lu_solve();
  test_lu_factorize();
  test_set_diagonal_scalar();
  test_set_diagonal_array();
  test_block_triangular_solve();
  test_cholesky();
  test_singular_solves();
  test_symmetric();
  test_comparison();
  test_operators();
  test_iteration();
  test_to_array();
  test_convert();
  test_power();
  test_near_singular_pivots();
  test_dense_large();
  test_make_pattern();
  test_symmetric_from_lower();
  test_banded();
  test_block_diagonal();
  test_determinant();
  test_inverse();
  test_least_squares();
  test_fuse();
}

// --- Main ---
//
// Guarded so test_sparsemat_gpu.cu can #include this file (to reuse every
// test_*() function and run_all_tests() above) without pulling in a second,
// conflicting main() — it defines its own that launches a kernel instead.
#ifndef SPARSEMAT_TEST_NO_MAIN

int main() {
  run_all_tests();
  test_fuse_lambda();
  test_large_sparse_stress();

  std::printf("\n%d passed, %d failed.\n", test_harness::host_passed, test_harness::host_failed);
  return test_harness::host_failed > 0 ? 1 : 0;
}

#endif  // SPARSEMAT_TEST_NO_MAIN
