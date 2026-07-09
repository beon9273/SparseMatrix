#include <cmath>

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

// --- SparseMatBuilder ---
//
// One test per sparsity-pattern shape: a diagonal pattern (mostly zero), a
// fully dense pattern (no zeros), and a pattern whose last (bottom-right)
// position is structurally zero — a regression case for a bug where build()
// read one past the end of its flattened non-zero index array once the
// final scanned position wasn't itself a non-zero.

struct DiagonalPattern3x3 {
  SPARSEMAT_HD static constexpr auto sparsity() {
    return std::array<std::array<int, 3>, 3>{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  }
};

SPARSEMAT_HD void test_builder_diagonal() {
  SparseMatBuilder<DiagonalPattern3x3, double, IntType> builder;
  auto mat = builder.build({{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}});
  check_eq(decltype(mat)::rows, IntType(3), "builder diagonal: rows");
  check_eq(decltype(mat)::cols, IntType(3), "builder diagonal: cols");
  check_eq(decltype(mat)::nonZeroCount, IntType(3), "builder diagonal: nonZeroCount");
  check_eq(mat.template get<0, 0>(), 1.0, "builder diagonal: (0,0)");
  check_eq(mat.template get<1, 1>(), 2.0, "builder diagonal: (1,1)");
  check_eq(mat.template get<2, 2>(), 3.0, "builder diagonal: (2,2)");
  check_eq(mat.template get<0, 1>(), 0.0, "builder diagonal: zero element (0,1)");
}

struct DensePattern3x2 {
  SPARSEMAT_HD static constexpr auto sparsity() {
    return std::array<std::array<int, 2>, 3>{{{1, 1}, {1, 1}, {1, 1}}};
  }
};

SPARSEMAT_HD void test_builder_dense() {
  SparseMatBuilder<DensePattern3x2, double, IntType> builder;
  auto mat = builder.build({{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}});
  check_eq(decltype(mat)::rows, IntType(3), "builder dense: rows");
  check_eq(decltype(mat)::cols, IntType(2), "builder dense: cols");
  check_eq(decltype(mat)::nonZeroCount, IntType(6), "builder dense: nonZeroCount");
  check_eq(mat.template get<0, 0>(), 1.0, "builder dense: (0,0)");
  check_eq(mat.template get<1, 1>(), 4.0, "builder dense: (1,1)");
  check_eq(mat.template get<2, 1>(), 6.0, "builder dense: (2,1)");
}

struct FirstColumnPattern2x2 {
  SPARSEMAT_HD static constexpr auto sparsity() {
    return std::array<std::array<int, 2>, 2>{{{1, 0}, {1, 0}}};
  }
};

SPARSEMAT_HD void test_builder_trailing_zero() {
  SparseMatBuilder<FirstColumnPattern2x2, double, IntType> builder;
  auto mat = builder.build({{{5.0, 99.0}, {7.0, 99.0}}});
  check_eq(decltype(mat)::nonZeroCount, IntType(2), "builder trailing zero: nonZeroCount");
  check_eq(mat.template get<0, 0>(), 5.0, "builder trailing zero: (0,0)");
  check_eq(mat.template get<1, 0>(), 7.0, "builder trailing zero: (1,0)");
  check_eq(mat.template get<0, 1>(), 0.0, "builder trailing zero: discarded value at (0,1)");
  check_eq(mat.template get<1, 1>(), 0.0, "builder trailing zero: discarded value at (1,1)");
}

// --- SparseMatBuilderCSR ---
//
// Unlike SparseMatBuilder's dense 0/1 grid, SparseMatBuilderCSR's pattern is
// a coordinate (row, col) list, and build() takes matching (row, col, value)
// triples. One test for the basic case (triples in sparsity() order), one
// for triples supplied out of order — a regression case for a bug where
// build() wrote each matched value back to the input triple's own position
// instead of its matched position in the flattened sparsity order, silently
// scrambling values whenever the input order didn't match sparsity() — and
// one using the builder's default DType/IntType template arguments.

struct DiagonalPatternCSR3x3 {
  static constexpr int rows = 3;
  static constexpr int cols = 3;
  SPARSEMAT_HD static constexpr auto sparsity() {
    return std::array<std::pair<int, int>, 3>{{{0, 0}, {1, 1}, {2, 2}}};
  }
};

SPARSEMAT_HD void test_builder_csr_diagonal() {
  SparseMatBuilderCSR<DiagonalPatternCSR3x3, double, IntType> builder;
  auto mat = builder.build({{{0, 0, 1.0}, {1, 1, 2.0}, {2, 2, 3.0}}});
  check_eq(decltype(mat)::rows, IntType(3), "builder csr diagonal: rows");
  check_eq(decltype(mat)::cols, IntType(3), "builder csr diagonal: cols");
  check_eq(decltype(mat)::nonZeroCount, IntType(3), "builder csr diagonal: nonZeroCount");
  check_eq(mat.template get<0, 0>(), 1.0, "builder csr diagonal: (0,0)");
  check_eq(mat.template get<1, 1>(), 2.0, "builder csr diagonal: (1,1)");
  check_eq(mat.template get<2, 2>(), 3.0, "builder csr diagonal: (2,2)");
  check_eq(mat.template get<0, 1>(), 0.0, "builder csr diagonal: zero element (0,1)");
}

struct SparsePatternCSR2x3 {
  static constexpr int rows = 2;
  static constexpr int cols = 3;
  SPARSEMAT_HD static constexpr auto sparsity() {
    return std::array<std::pair<int, int>, 3>{{{0, 0}, {0, 2}, {1, 1}}};
  }
};

SPARSEMAT_HD void test_builder_csr_out_of_order_triples() {
  SparseMatBuilderCSR<SparsePatternCSR2x3, double, IntType> builder;
  // Triples supplied in a different order than sparsity(); build() must
  // match each by (row, col) coordinate rather than by input position.
  auto mat = builder.build({{{1, 1, 5.0}, {0, 2, 7.0}, {0, 0, 9.0}}});
  check_eq(decltype(mat)::rows, IntType(2), "builder csr out-of-order: rows");
  check_eq(decltype(mat)::cols, IntType(3), "builder csr out-of-order: cols");
  check_eq(decltype(mat)::nonZeroCount, IntType(3), "builder csr out-of-order: nonZeroCount");
  check_eq(mat.template get<0, 0>(), 9.0, "builder csr out-of-order: (0,0)");
  check_eq(mat.template get<0, 2>(), 7.0, "builder csr out-of-order: (0,2)");
  check_eq(mat.template get<1, 1>(), 5.0, "builder csr out-of-order: (1,1)");
  check_eq(mat.template get<0, 1>(), 0.0, "builder csr out-of-order: zero element (0,1)");
}

struct DefaultTypePatternCSR2x2 {
  static constexpr int rows = 2;
  static constexpr int cols = 2;
  SPARSEMAT_HD static constexpr auto sparsity() {
    return std::array<std::pair<int, int>, 2>{{{0, 1}, {1, 0}}};
  }
};

SPARSEMAT_HD void test_builder_csr_default_types() {
  // Omits the DType/IntType template arguments, exercising the builder's
  // documented double/int32_t defaults.
  SparseMatBuilderCSR<DefaultTypePatternCSR2x2> builder;
  auto mat = builder.build({{{0, 1, 4.0}, {1, 0, 6.0}}});
  static_assert(std::is_same_v<decltype(mat)::DataType, double>,
                "builder csr default types: DataType should default to double");
  static_assert(std::is_same_v<decltype(mat)::Int, int32_t>,
                "builder csr default types: Int should default to int32_t");
  check_eq(mat.template get<0, 1>(), 4.0, "builder csr default types: (0,1)");
  check_eq(mat.template get<1, 0>(), 6.0, "builder csr default types: (1,0)");
  check_eq(mat.template get<0, 0>(), 0.0, "builder csr default types: zero element (0,0)");
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

// --- Run everything ---
//
// Shared by both runners: test_sparsemat's main() below calls this directly
// on the host, and test_sparsemat_gpu.cu's __global__ kernel calls it from
// the device instead.

SPARSEMAT_HD inline void run_all_tests() {
  test_construction();
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
  test_builder_diagonal();
  test_builder_dense();
  test_builder_trailing_zero();
  test_builder_csr_diagonal();
  test_builder_csr_out_of_order_triples();
  test_builder_csr_default_types();
  test_trace();
  test_dot();
  test_identity();
  test_kronecker();
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
}

// --- Main ---
//
// Guarded so test_sparsemat_gpu.cu can #include this file (to reuse every
// test_*() function and run_all_tests() above) without pulling in a second,
// conflicting main() — it defines its own that launches a kernel instead.
#ifndef SPARSEMAT_TEST_NO_MAIN

int main() {
  run_all_tests();

  std::printf("\n%d passed, %d failed.\n", test_harness::host_passed, test_harness::host_failed);
  return test_harness::host_failed > 0 ? 1 : 0;
}

#endif  // SPARSEMAT_TEST_NO_MAIN
