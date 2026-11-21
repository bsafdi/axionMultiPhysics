// fluids general sparse LU (cpp/include/axionmultiphysics/fluids/sparse_lu.hpp):
//   1. CsrMatrix::from_eigen gives the CSR of the SAME matrix from either Eigen storage order.
//   2. Every available backend solves a deliberately NONSYMMETRIC system (an Oseen-type convection-diffusion
//      block with strong convection, coupled to a divergence-like constraint with a zero block) to round-off,
//      checked against the known solution.
//   3. The CSC-as-CSR mistake SymSolver makes for symmetric matrices is detected: handing the column-major
//      arrays of the same nonsymmetric matrix as CSR gives an O(1) error, so test 2 would fail on that bug.
//   4. The serial backends gather a DistCsr: one-rank DistCsr and the CsrMatrix give the same answer.
//   5. The shared-memory STRUMPACK backend goes through the STRUMPACK guard of sym_solver.cpp: its calls are counted
//      there, its construction runs alone, and it leaves no entries in STRUMPACK's process-wide timer lists.
//   6. On MPI_COMM_WORLD: the nonsymmetric matrix distributed with uneven rows and an empty
//      rank, every backend; duplicate entries refused (and summed by sum_duplicates); a numerically singular matrix
//      fails on EVERY rank (no deadlock: STRUMPACK-MPI reports the zero pivot on one rank only); refinement stats.
//   7. The index width of Eigen's factorization on numbers: 32-bit iff n <= 46 340 and nnz(A) <= 2^28, else 64-bit;
//      Eigen's initial estimate of the factor, min(20 (nnz + 1) / n, n) n, for measured 3D Stokes sizes (by hand).
//   8. The 64-bit instantiation, reached through the test bound n = 100: bitwise the 32-bit solution, a dense
//      partial-pivoting solve to round-off, and stats().index_bits from the solver object that ran.
//   9. Eigen's failures on rank 0 (an expansion failure, memInit's failure that leaves info() at Success, and a
//      std::bad_alloc before factorize), through the failure hook: one error on every rank, with the facts and the
//      remedies. Registered on 1 and 2 ranks; a rank left waiting hangs, and the ctest TIMEOUT ends it.
//  10. A SparseLU reused after Eigen's failure factorizes the next matrix (a fresh Eigen solver per factorization).
//      A failed factorization releases its solver, and solve() afterwards is a clean error, not a crash.
//  11. A failure BEFORE rank 0's Eigen work, on a SparseLU that had factorized and solved a smaller matrix: the input
//      contract (duplicate entries), and rank 0's allocation of the gathered matrix (a replaced operator new that
//      fails once). Every rank throws the same error (the gather's names n and nnz(A)); solve() afterwards is the
//      clean "no factorization" error on every rank, not the previous factor applied to the new, larger matrix (a
//      crash); a healthy factorization of the new matrix on the same object then solves to round-off.
#include <mpi.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fluids/sparse_lu.hpp"
#include "axionmultiphysics/sym_solver.hpp"

using namespace axmp::fluids;
static int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

using SpMat = Eigen::SparseMatrix<double>;

// Section 11's allocation failure: the next allocation of exactly g_fail_bytes bytes throws std::bad_alloc, once
// (0: disarmed). This replaces the global operator new of the whole program; disarmed, it is malloc with the
// new-handler loop of the standard one.
static std::atomic<std::size_t> g_fail_bytes{0};
void* operator new(std::size_t sz) {
  std::size_t f = g_fail_bytes.load();
  if (f != 0 && sz == f && g_fail_bytes.compare_exchange_strong(f, 0)) throw std::bad_alloc();
  for (;;) {
    if (void* p = std::malloc(sz ? sz : 1)) return p;
    std::new_handler h = std::get_new_handler();
    if (h == nullptr) throw std::bad_alloc();
    h();
  }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

// [ L  G ; D  0 ]: L = -Lap + Pe (c . grad) on a g x g grid (upwind-free central convection: nonsymmetric),
// G a forward-difference "gradient" (g*g x m), D = G^T scaled differently (so the coupling is nonsymmetric too)
static SpMat oseen_like(int g, double pe, int m) {
  const int n = g * g;
  std::vector<Eigen::Triplet<double>> t;
  auto id = [&](int i, int j) { return i * g + j; };
  const double cx = 1.0, cy = 0.5;
  for (int i = 0; i < g; ++i)
    for (int j = 0; j < g; ++j) {
      const int k = id(i, j);
      t.emplace_back(k, k, 4.0);
      if (i > 0) t.emplace_back(k, id(i - 1, j), -1.0 - 0.5 * pe * cx);
      if (i + 1 < g) t.emplace_back(k, id(i + 1, j), -1.0 + 0.5 * pe * cx);
      if (j > 0) t.emplace_back(k, id(i, j - 1), -1.0 - 0.5 * pe * cy);
      if (j + 1 < g) t.emplace_back(k, id(i, j + 1), -1.0 + 0.5 * pe * cy);
    }
  for (int c = 0; c < m; ++c) {
    const int a = (c * 7) % n, b = (c * 13 + 1) % n;
    t.emplace_back(a, n + c, 1.0);
    t.emplace_back(b, n + c, -1.0);
    t.emplace_back(n + c, a, 0.7);
    t.emplace_back(n + c, b, -1.3);
  }
  SpMat A(n + m, n + m);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

static double rel(const std::vector<double>& a, const Eigen::VectorXd& b) {
  double d = 0, s = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    d += (a[i] - b(i)) * (a[i] - b(i));
    s += b(i) * b(i);
  }
  return std::sqrt(d / s);
}

// ------------------------------------------------------------------------------------------------ sections 7-11
static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// true on every rank iff every rank of comm passed the same bytes. Collective.
static bool same_on_all_ranks(const std::string& s, MPI_Comm comm) {
  int len = static_cast<int>(s.size()), lo = 0, hi = 0;
  MPI_Allreduce(&len, &lo, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&len, &hi, 1, MPI_INT, MPI_MAX, comm);
  if (lo != hi) return false;
  std::string r0 = s;
  MPI_Bcast(r0.data(), len, MPI_CHAR, 0, comm);
  int mine = r0 == s ? 1 : 0, all = 0;
  MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, comm);
  return all == 1;
}

// The test hooks are process-wide: put them back when a section ends, however it ends.
struct HookReset {
  ~HookReset() {
    try {
      set_eigen_lu_max_n32_for_testing(0);
      set_eigen_lu_failure_for_testing(EigenLUTestFailure::None);
    } catch (...) {
    }
  }
};

// S distributed over comm with uneven rows (rank p holds a share proportional to p + 1); dist receives the offsets.
static DistCsr distributed(const SpMat& S, MPI_Comm comm, std::vector<int>& dist) {
  int P = 1;
  MPI_Comm_size(comm, &P);
  const long n = S.rows();
  dist.assign(P + 1, 0);
  const long tot = static_cast<long>(P) * (P + 1) / 2;
  long acc = 0;
  for (int p = 0; p < P; ++p) {
    acc += p + 1;
    dist[p + 1] = static_cast<int>(n * acc / tot);
  }
  return DistCsr::distribute(CsrMatrix::from_eigen(S), dist, comm);
}

// 7. The index width of Eigen's factorization, on numbers. 32-bit iff n <= 46 340 and nnz(A) <= 2^28:
//    46 340^2 = 2 147 395 600 and 46 340^2 + 46 340 = 2 147 441 940 are below 2^31 - 1 = 2 147 483 647, while
//    46 341^2 = 2 147 488 281 is above it; 2^28 = 268 435 456. The sizes "3D N = k" are those measured for the
//    bordered P2/P1 Stokes system on MakeCartesian3D(k, k, k, TETRAHEDRON) (enclosed, mean-value multiplier).
static void index_rule() {
  const long k28 = 268435456L;
  CHECK(kEigenLUIndex32MaxN == 46340L && kEigenLUIndex32MaxNnz == k28, "the rule's constants are %ld and %ld",
        static_cast<long>(kEigenLUIndex32MaxN), static_cast<long>(kEigenLUIndex32MaxNnz));
  struct Case {
    long n, nnz;
    int bits;
    const char* what;
  };
  const Case cases[] = {
      {46340, k28, 32, "n = 46 340 and nnz = 2^28 (both bounds, with equality)"},
      {46340, k28 + 1, 64, "n = 46 340, nnz = 2^28 + 1"},
      {46341, 46341, 64, "n = 46 341, nnz = n"},
      {46341, 4000000, 64, "n = 46 341, 86 entries per row"},
      {46340, 4000000, 32, "n = 46 340, 86 entries per row"},
      {1, 1, 32, "n = 1"},
      {15469, 1350568, 32, "3D N = 8"},
      {49073, 4462073, 64, "3D N = 12"},
      {368573, 34734647, 64, "3D N = 24"},
      {859813, 81348184, 64, "3D N = 32"},
  };
  for (const auto& c : cases) {
    const int bits = eigen_lu_index_bits(c.n, c.nnz);
    CHECK(bits == c.bits, "index rule, %s: %d-bit, expected %d-bit", c.what, bits, c.bits);
  }
  // the rule with an explicit n bound (the form factorize applies under the test bound)
  CHECK(eigen_lu_index_bits(100, 1000, 100) == 32, "index rule with the n bound 100: n = 100 is not 32-bit");
  CHECK(eigen_lu_index_bits(101, 1000, 100) == 64, "index rule with the n bound 100: n = 101 is not 64-bit");
  CHECK(eigen_lu_index_bits(100, k28 + 1, 100) == 64, "index rule with the n bound 100: nnz = 2^28 + 1 is not 64-bit");
  // the test bound reaches factorize only: the two-argument rule keeps the constants
  set_eigen_lu_max_n32_for_testing(100);
  const int under_hook = eigen_lu_index_bits(46340, k28);
  set_eigen_lu_max_n32_for_testing(0);
  CHECK(under_hook == 32, "eigen_lu_index_bits(n, nnz) followed the test bound (%d-bit)", under_hook);

  // Eigen's initial estimate of the factor (memInit with fillfactor 20), min(20 (nnz + 1) / n, n) n in integer
  // arithmetic, done by hand for the measured 3D sizes:
  //   N = 24 (n = 368 573, nnz = 34 734 647): 20 x 34 734 648 = 694 692 960; 368 573 x 1 884 = 694 391 532 leaves
  //     301 428 < n, so the quotient is 1 884 (below n) and the estimate 1 884 x 368 573 = 694 391 532.
  //   N = 32 (n = 859 813, nnz = 81 348 184): 20 x 81 348 185 = 1 626 963 700; 859 813 x 1 892 = 1 626 766 196
  //     leaves 197 504 < n, so the quotient is 1 892 (859 813 x 1 891 would leave 1 057 317, more than n) and the
  //     estimate 1 626 766 196.
  //   N = 8 (n = 15 469, nnz = 1 350 568): 20 x 1 350 569 = 27 011 380; 15 469 x 1 746 = 27 008 874 leaves 2 506,
  //     so the estimate is 27 008 874.
  //   n = 10, nnz = 100 (dense): 20 x 101 / 10 = 202 is above n, so the min takes n: 10 x 10 = 100.
  //   n = 20, nnz = 19: 20 x 20 / 20 = 20, so the estimate is 20 x 20 = 400 (without the +1: 19, and 380).
  //   n = 0: 0 (no division).
  struct Est {
    long n, nnz, est;
    const char* what;
  };
  const Est ests[] = {{368573, 34734647, 694391532L, "3D N = 24"},
                      {859813, 81348184, 1626766196L, "3D N = 32"},
                      {15469, 1350568, 27008874L, "3D N = 8"},
                      {10, 100, 100L, "a dense 10 x 10 (the min)"},
                      {20, 19, 400L, "n = 20, nnz = 19 (20 (nnz + 1) / n = 20, not 19)"},
                      {0, 0, 0L, "n = 0"}};
  for (const auto& e : ests) {
    const long got = eigen_lu_factor_nnz_estimate(e.n, e.nnz);
    CHECK(got == e.est, "Eigen's initial estimate, %s: %ld, expected %ld", e.what, got, e.est);
  }

  // the failure test: the message (memInit's failure sets only the message), then info()
  CHECK(!eigen_lu_failed(Eigen::Success, ""), "eigen_lu_failed: a clean factorization read as a failure");
  CHECK(eigen_lu_failed(Eigen::NumericalIssue, "UNABLE TO EXPAND MEMORY IN COLUMN_BMOD() "),
        "eigen_lu_failed: an expansion failure read as a success");
  CHECK(eigen_lu_failed(Eigen::Success, "UNABLE TO ALLOCATE WORKING MEMORY\n\n"),
        "eigen_lu_failed: memInit's failure (message set, info() at Success) read as a success");
  CHECK(eigen_lu_failed(Eigen::NumericalIssue, ""), "eigen_lu_failed: info() != Success read as a success");
}

// 8. The 64-bit instantiation, reached through the test bound n = 100, on the ranks of comm.
static void index64_path(MPI_Comm comm) {
  int P = 1, me = 0;
  MPI_Comm_size(comm, &P);
  MPI_Comm_rank(comm, &me);
  HookReset reset;
  const SpMat S = oseen_like(12, 3.0, 10);  // n = 154 > 100: a nonsymmetric saddle point, 10 rows with a zero block
  const int n = static_cast<int>(S.rows());
  std::vector<int> dist;
  const DistCsr D = distributed(S, comm, dist);
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> U(-1, 1);
  Eigen::VectorXd xs(n);
  for (int i = 0; i < n; ++i) xs(i) = U(rng);
  const Eigen::VectorXd b = S * xs;
  const std::vector<double> bl(b.data() + dist[me], b.data() + dist[me + 1]);
  const long nnz = static_cast<long>(S.nonZeros());
  const long est = std::min(20 * (nnz + 1) / n, static_cast<long>(n)) * n;  // memInit's initial estimate

  struct Run {
    std::vector<double> x, x_plain;
    LUStats st;
    int live = -1;
  };
  auto run = [&](long bound) {
    Run r;
    if (bound > 0) set_eigen_lu_max_n32_for_testing(bound);
    {
      SparseLU lu(LUBackend::EigenLU);
      lu.factorize(D);
      r.live = lu.live_eigen_index_bits_for_testing();
      r.x = lu.solve(bl);
      r.st = lu.stats();
    }
    {
      SparseLU lu(LUBackend::EigenLU);
      lu.set_refinement(0, 1e-12);  // the plain LU solve: what the factorization alone gives
      lu.factorize(D);
      r.x_plain = lu.solve(bl);
    }
    if (bound > 0) set_eigen_lu_max_n32_for_testing(0);
    return r;
  };
  const Run r32 = run(0), r64 = run(100);
  const int live32 = me == 0 ? 32 : 0, live64 = me == 0 ? 64 : 0;  // the solver object lives on rank 0 only
  CHECK(r32.st.index_bits == 32 && r32.live == live32,
        "rank %d, n = %d, no test bound: stats().index_bits %d, solver object %d-bit (expected 32 and %d)", me, n,
        r32.st.index_bits, r32.live, live32);
  CHECK(r64.st.index_bits == 64 && r64.live == live64,
        "rank %d, n = %d, test bound 100: stats().index_bits %d, solver object %d-bit (expected 64 and %d)", me, n,
        r64.st.index_bits, r64.live, live64);
  CHECK(r32.st.factor_nnz_estimate == est && r64.st.factor_nnz_estimate == est,
        "rank %d: stats().factor_nnz_estimate %ld (32-bit), %ld (64-bit), expected %ld", me,
        r32.st.factor_nnz_estimate, r64.st.factor_nnz_estimate, est);
  CHECK(r64.st.factor_nnz > 0 && r64.st.factor_nnz == r32.st.factor_nnz, "rank %d: fill %ld (64-bit) vs %ld (32-bit)",
        me, r64.st.factor_nnz, r32.st.factor_nnz);
  auto bitwise = [](const std::vector<double>& a, const std::vector<double>& c) {
    return a.size() == c.size() && std::memcmp(a.data(), c.data(), a.size() * sizeof(double)) == 0;
  };
  CHECK(bitwise(r64.x, r32.x), "rank %d: the refined 64-bit solution differs from the 32-bit one", me);
  CHECK(bitwise(r64.x_plain, r32.x_plain), "rank %d: the plain 64-bit LU solve differs from the 32-bit one", me);

  // Against a dense partial-pivoting LU of the same system. Tolerance: for any x, x - x* = A^-1 (A x - b), so
  //   ||x - x_dense|| <= (||A x - b|| + ||A x_dense - b||) / sigma_min(A),
  // with the residuals evaluated in long double (x, A and b are doubles, so the evaluation error is about 1e-19 of
  // |A||x|, far below the double-precision residuals) and sigma_min from a dense SVD (relative error about
  // kappa u, 1e-13 here); a factor 2 covers both. The bound is meaningful only if it is at round-off, so each
  // solve must also be backward stable: ||A x - b|| <= n u (||A|| ||x|| + ||b||) (normwise backward error at most
  // n u, Rigal and Gaches), which makes the bound at most 4 n u kappa ||x||.
  std::vector<int> counts(P);
  for (int p = 0; p < P; ++p) counts[p] = dist[p + 1] - dist[p];
  auto gather_all = [&](const std::vector<double>& xl) {
    Eigen::VectorXd x(n);
    MPI_Allgatherv(xl.data(), static_cast<int>(xl.size()), MPI_DOUBLE, x.data(), counts.data(), dist.data(),
                   MPI_DOUBLE, comm);
    return x;
  };
  auto residual = [&](const Eigen::VectorXd& x) {
    std::vector<long double> r(n);
    for (int i = 0; i < n; ++i) r[i] = b(i);
    for (int k = 0; k < S.outerSize(); ++k)
      for (SpMat::InnerIterator it(S, k); it; ++it)
        r[it.row()] -= static_cast<long double>(it.value()) * static_cast<long double>(x(k));
    long double s = 0;
    for (long double v : r) s += v * v;
    return static_cast<double>(std::sqrt(s));
  };
  const Eigen::MatrixXd Sd(S);
  const Eigen::VectorXd xd = Sd.partialPivLu().solve(b);
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(Sd);
  const double smax = svd.singularValues()(0), smin = svd.singularValues()(n - 1);
  const double u = std::ldexp(1.0, -53);
  const double rd = residual(xd);
  CHECK(rd <= n * u * (smax * xd.norm() + b.norm()), "the dense reference is not backward stable (%g)", rd);
  for (int k = 0; k < 2; ++k) {
    const char* what = k == 0 ? "refined" : "plain";
    const Eigen::VectorXd x = gather_all(k == 0 ? r64.x : r64.x_plain);
    const double rs = residual(x), diff = (x - xd).norm(), bound = 2.0 * (rs + rd) / smin;
    if (me == 0)
      std::printf("64-bit %s solve vs dense LU: |x - x_dense| / |x| = %.2e, bound %.2e; backward errors %.2e and "
                  "%.2e (n u = %.2e); kappa_2 = %.3g\n",
                  what, diff / xd.norm(), bound / xd.norm(), rs / (smax * x.norm() + b.norm()),
                  rd / (smax * xd.norm() + b.norm()), n * u, smax / smin);
    CHECK(rs <= n * u * (smax * x.norm() + b.norm()), "rank %d: the 64-bit %s solve is not backward stable (%g)", me,
          what, rs / (smax * x.norm() + b.norm()));
    CHECK(diff <= bound, "rank %d: the 64-bit %s solve is %g from the dense one, above the bound %g", me, what,
          diff / xd.norm(), bound / xd.norm());
  }

  // a matrix with n <= 100 stays 32-bit under the same test bound: the hook lowers the bound, it does not force 64
  {
    const SpMat S2 = oseen_like(9, 3.0, 8);  // n = 89
    std::vector<int> d2;
    const DistCsr D2 = distributed(S2, comm, d2);
    set_eigen_lu_max_n32_for_testing(100);
    SparseLU lu(LUBackend::EigenLU);
    lu.factorize(D2);
    const int live = lu.live_eigen_index_bits_for_testing(), bits = lu.stats().index_bits;
    set_eigen_lu_max_n32_for_testing(0);
    CHECK(bits == 32 && live == live32, "rank %d, n = 89, test bound 100: index_bits %d, solver object %d-bit", me,
          bits, live);
  }
}

// 9. Eigen's failures on rank 0, through the failure hook, are thrown on every rank of comm with one message that
// carries the facts and the remedies. The LU runs on a duplicate of comm and this test's own collectives on comm,
// so a rank left waiting inside the LU's collectives cannot be released by them: it hangs, and the TIMEOUT ends it.
static void failures_collective(MPI_Comm comm) {
  int P = 1, me = 0;
  MPI_Comm_size(comm, &P);
  MPI_Comm_rank(comm, &me);
  HookReset reset;
  MPI_Comm lc;
  MPI_Comm_dup(comm, &lc);
  const SpMat S = oseen_like(12, 3.0, 10);
  const long n = S.rows(), nnz = S.nonZeros();
  const long est = std::min(20 * (nnz + 1) / n, n) * n;  // memInit's initial estimate
  std::vector<int> dist;
  const DistCsr D = distributed(S, lc, dist);
  struct Mode {
    EigenLUTestFailure f;
    const char* eigen_text;
    const char* name;
  };
  const Mode modes[] = {
      {EigenLUTestFailure::ExpandMemory, "UNABLE TO EXPAND MEMORY IN COLUMN_BMOD()", "expansion failure"},
      {EigenLUTestFailure::MemInit, "UNABLE TO ALLOCATE WORKING MEMORY", "memInit failure (info() at Success)"},
      {EigenLUTestFailure::BadAlloc, "std::bad_alloc", "std::bad_alloc before factorize"},
  };
  bool printed = false;
  for (const auto& m : modes)
    for (long bound : {0L, 100L}) {
      const int bits = bound > 0 ? 64 : 32;
      std::string what;
      int kind = 0;  // 0: no exception, 1: axmp::Error, 2: another exception
      try {
        if (bound > 0) set_eigen_lu_max_n32_for_testing(bound);
        set_eigen_lu_failure_for_testing(m.f);
        SparseLU lu(LUBackend::EigenLU);
        lu.factorize(D);
      } catch (const axmp::Error& e) {
        what = e.what();
        kind = 1;
      } catch (const std::exception& e) {
        what = e.what();
        kind = 2;
      }
      set_eigen_lu_failure_for_testing(EigenLUTestFailure::None);
      set_eigen_lu_max_n32_for_testing(0);
      CHECK(kind == 1, "rank %d, %s, %d-bit: %s%s", me, m.name, bits,
            kind == 0 ? "factorize did not throw" : "not an axmp::Error: ", what.c_str());
      if (kind == 1) {
        const std::string parts[] = {m.eigen_text,
                                     "n = " + std::to_string(n) + " unknowns",
                                     "nnz(A) = " + std::to_string(nnz),
                                     std::to_string(bits) + "-bit indices",
                                     "initial estimate of the factor was " + std::to_string(est) + " entries",
                                     "scripts/build_strumpack.sh",
                                     "lu=\"strumpack\"",
                                     "lu=\"strumpack-mpi\"",
                                     "solver=\"minres\"",
                                     "(reported by rank 0)"};
        for (const auto& p : parts)
          CHECK(has(what, p), "rank %d, %s, %d-bit: the error lacks '%s': %s", me, m.name, bits, p.c_str(),
                what.c_str());
        if (me == 0 && !printed) {
          std::printf("Eigen failure (%s, %d-bit): %s\n", m.name, bits, what.c_str());
          printed = true;
        }
      }
      // every rank threw an axmp::Error, with the same text (checked on comm, not on the LU's communicator)
      int mine = kind == 1 ? 1 : 0, all = 0;
      MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, comm);
      CHECK(all == 1, "rank %d, %s, %d-bit: the error was not raised on every rank", me, m.name, bits);
      CHECK(same_on_all_ranks(what, comm), "rank %d, %s, %d-bit: the ranks report different errors", me, m.name, bits);
    }
  // the LU's communicator is still in step: a healthy factorization and solve on it
  {
    std::vector<double> bl(D.local_rows, 1.0);
    bool ok = true;
    try {
      SparseLU lu(LUBackend::EigenLU);
      lu.factorize(D);
      lu.solve(bl);
    } catch (const std::exception& e) {
      ok = false;
      CHECK(false, "rank %d: after the failures, a healthy solve failed: %s", me, e.what());
    }
    int mine = ok ? 1 : 0, all = 0;
    MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, comm);
    CHECK(all == 1, "rank %d: after the failures, a healthy solve failed on some rank", me);
  }
  MPI_Comm_free(&lc);
}

// 10. One SparseLU, two factorizations: a failed one, then a healthy one. Eigen's SparseLU sets lastErrorMessage()
// on a failure and never clears it (a later success sets only info()), so a solver object kept from the failed
// factorization would report the healthy one as failed under the message-first failure test.
static void fresh_solver() {
  HookReset reset;
  const SpMat S = oseen_like(12, 3.0, 10);  // n = 154
  const int n = static_cast<int>(S.rows());
  SpMat Z = S;
  Z.prune([](Eigen::Index, Eigen::Index c, double) { return c != 5; });  // column 5 empty: structurally singular
  const DistCsr Dz = DistCsr::from_serial(CsrMatrix::from_eigen(Z), MPI_COMM_SELF);
  const DistCsr Ds = DistCsr::from_serial(CsrMatrix::from_eigen(S), MPI_COMM_SELF);
  std::mt19937 rng(13);
  std::uniform_real_distribution<double> U(-1, 1);
  Eigen::VectorXd xs(n);
  for (int i = 0; i < n; ++i) xs(i) = U(rng);
  const Eigen::VectorXd b = S * xs;
  const std::vector<double> bv(b.data(), b.data() + n);
  for (long bound : {0L, 100L}) {  // 32-bit, then 64-bit
    const int bits = bound > 0 ? 64 : 32;
    if (bound > 0) set_eigen_lu_max_n32_for_testing(bound);
    SparseLU lu(LUBackend::EigenLU);
    std::string what;
    bool threw = false;
    try {
      lu.factorize(Dz);
    } catch (const axmp::Error& e) {
      threw = true;
      what = e.what();
    }
    CHECK(threw, "%d-bit: a matrix with an empty column was factorized", bits);
    CHECK(lu.live_eigen_index_bits_for_testing() == 0, "%d-bit: the solver of the failed factorization is still held",
          bits);
    // solve() after a failed factorization: a clean error, not a null dereference (on a SparseLU that never succeeded)
    {
      bool solve_threw = false;
      std::string solve_what;
      try {
        lu.solve(bv);
      } catch (const axmp::Error& e) {
        solve_threw = true;
        solve_what = e.what();
      }
      CHECK(solve_threw && has(solve_what, "no factorization"), "%d-bit: solve() after a failed factorize: %s", bits,
            solve_threw ? solve_what.c_str() : "no error");
    }
    // the same after a SUCCESSFUL factorization followed by a failed one
    {
      SparseLU lu2(LUBackend::EigenLU);
      lu2.factorize(Ds);
      bool failed = false;
      try {
        lu2.factorize(Dz);
      } catch (const axmp::Error&) {
        failed = true;
      }
      CHECK(failed, "%d-bit: the second factorization (empty column) did not fail", bits);
      CHECK(lu2.live_eigen_index_bits_for_testing() == 0, "%d-bit: the failed re-factorization left a solver alive", bits);
      bool solve_threw = false;
      std::string solve_what;
      try {
        lu2.solve(bv);
      } catch (const axmp::Error& e) {
        solve_threw = true;
        solve_what = e.what();
      }
      CHECK(solve_threw && has(solve_what, "no factorization"), "%d-bit: solve() after a failed re-factorization: %s",
            bits, solve_threw ? solve_what.c_str() : "no error");
    }
    CHECK(has(what, "zero pivot or a singular matrix"), "%d-bit: the singular matrix's error lost its hint: %s", bits,
          what.c_str());
    try {
      lu.factorize(Ds);
      const auto x = lu.solve(bv);
      CHECK(rel(x, xs) < 1e-10, "%d-bit: reused SparseLU, solution error %g", bits, rel(x, xs));
      CHECK(lu.stats().index_bits == bits, "%d-bit: reused SparseLU, index_bits %d", bits, lu.stats().index_bits);
    } catch (const axmp::Error& e) {
      CHECK(false, "%d-bit: a SparseLU reused after a failed factorization refused a healthy matrix: %s", bits,
            e.what());
    }
    if (bound > 0) set_eigen_lu_max_n32_for_testing(0);
  }
}

// 11. A failure before rank 0's Eigen work, on the ranks of comm, after a successful factorization and solve of a
// smaller matrix on the same SparseLU. `bound` is the 32-bit limit of the test hook (0: the real rule, so the first
// factorization is 32-bit; 100: it is 64-bit), so that the solver held by the failed factorization is released in both
// index widths.
static void failure_before_eigen(MPI_Comm comm, long bound) {
  int P = 1, me = 0;
  MPI_Comm_size(comm, &P);
  MPI_Comm_rank(comm, &me);
  HookReset reset;
  if (bound > 0) set_eigen_lu_max_n32_for_testing(bound);
  const int first_bits = bound > 0 ? 64 : 32;
  const SpMat S1 = oseen_like(12, 3.0, 10);  // n = 154
  const SpMat S2 = oseen_like(16, 3.0, 12);  // n = 268, larger than the factor of S1
  std::vector<int> d1, d2;
  const DistCsr D1 = distributed(S1, comm, d1), D2 = distributed(S2, comm, d2);
  const long n2 = S2.rows(), nnz2 = S2.nonZeros();
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> U(-1, 1);
  Eigen::VectorXd xs2(n2);
  for (long i = 0; i < n2; ++i) xs2(i) = U(rng);
  const Eigen::VectorXd b2 = S2 * xs2, b1 = S1 * Eigen::VectorXd::Ones(S1.rows());
  const std::vector<double> b1l(b1.data() + d1[me], b1.data() + d1[me + 1]);
  const std::vector<double> b2l(b2.data() + d2[me], b2.data() + d2[me + 1]);
  // D2 with every entry split in two halves: refused by the input contract
  DistCsr Ddup = D2;
  {
    std::vector<int> rp(1, 0), cc;
    std::vector<double> vv;
    for (int i = 0; i < D2.local_rows; ++i) {
      for (int k = D2.rowptr[i]; k < D2.rowptr[i + 1]; ++k) {
        cc.insert(cc.end(), 2, D2.col[k]);
        vv.insert(vv.end(), 2, 0.5 * D2.val[k]);
      }
      rp.push_back(static_cast<int>(cc.size()));
    }
    Ddup.rowptr.swap(rp);
    Ddup.col.swap(cc);
    Ddup.val.swap(vv);
  }
  // true on every rank iff it holds on every rank
  auto on_all = [&](bool mine) {
    int m = mine ? 1 : 0, all = 0;
    MPI_Allreduce(&m, &all, 1, MPI_INT, MPI_MIN, comm);
    return all == 1;
  };
  for (const char* kind : {"input contract", "gather allocation"}) {
    const bool gather = std::strcmp(kind, "gather allocation") == 0;
    SparseLU lu(LUBackend::EigenLU);
    bool first_ok = true;
    try {
      lu.factorize(D1);
      lu.solve(b1l);
    } catch (const std::exception& e) {
      first_ok = false;
      CHECK(false, "rank %d, %s: the first factorization and solve failed: %s", me, kind, e.what());
    }
    if (first_ok && me == 0)
      CHECK(lu.live_eigen_index_bits_for_testing() == first_bits, "%s: the first factorization is not %d-bit", kind,
            first_bits);
    if (!on_all(first_ok)) continue;
    // the failing factorization of the larger matrix
    std::string what;
    int how = 0;  // 0: no exception, 1: axmp::Error, 2: another exception
    if (gather && me == 0) g_fail_bytes = static_cast<std::size_t>(nnz2) * sizeof(double);  // the gathered values
    try {
      lu.factorize(gather ? D2 : Ddup);
    } catch (const axmp::Error& e) {
      what = e.what();
      how = 1;
    } catch (const std::exception& e) {
      what = e.what();
      how = 2;
    }
    const bool armed_left = g_fail_bytes.exchange(0) != 0;
    CHECK(!armed_left, "rank %d, %s: the planted allocation failure never happened", me, kind);
    CHECK(how == 1, "rank %d, %s: %s%s", me, kind, how == 0 ? "factorize did not throw" : "not an axmp::Error: ",
          what.c_str());
    CHECK(on_all(how == 1), "rank %d, %s: the error was not raised on every rank", me, kind);
    CHECK(same_on_all_ranks(what, comm), "rank %d, %s: the ranks report different errors: %s", me, kind, what.c_str());
    if (gather && how == 1) {
      const std::string parts[] = {"could not allocate the gathered matrix",
                                   "n = " + std::to_string(n2) + " unknowns",
                                   "nnz(A) = " + std::to_string(nnz2),
                                   "(reported by rank 0)",
                                   "lu=\"strumpack-mpi\"",
                                   "solver=\"minres\""};
      for (const auto& p : parts)
        CHECK(has(what, p), "rank %d, %s: the error lacks '%s': %s", me, kind, p.c_str(), what.c_str());
    }
    CHECK(lu.live_eigen_index_bits_for_testing() == 0, "rank %d, %s: the previous factorization's solver is still held",
          me, kind);
    // solve() afterwards: the clean error, the same on every rank
    std::string sw;
    int shown = 0;
    try {
      lu.solve(b2l);
    } catch (const axmp::Error& e) {
      sw = e.what();
      shown = 1;
    } catch (const std::exception& e) {
      sw = e.what();
      shown = 2;
    }
    CHECK(shown == 1 && has(sw, "no factorization"), "rank %d, %s: solve() after the failed factorize: %s", me, kind,
          shown == 0 ? "returned" : sw.c_str());
    CHECK(on_all(shown == 1), "rank %d, %s: solve() after the failed factorize did not fail on every rank", me, kind);
    CHECK(same_on_all_ranks(sw, comm), "rank %d, %s: solve() errors differ between the ranks", me, kind);
    // the same object factorizes the larger matrix and solves it
    bool ok = true;
    double err = 1;
    try {
      lu.factorize(D2);
      const auto x = lu.solve(b2l);
      double loc[2] = {0, 0}, glob[2] = {0, 0};
      for (int i = 0; i < D2.local_rows; ++i) {
        loc[0] += std::pow(x[i] - xs2(d2[me] + i), 2);
        loc[1] += std::pow(xs2(d2[me] + i), 2);
      }
      MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
      err = std::sqrt(glob[0] / glob[1]);
    } catch (const std::exception& e) {
      ok = false;
      CHECK(false, "rank %d, %s: the healthy factorization afterwards failed: %s", me, kind, e.what());
    }
    CHECK(on_all(ok) && err < 1e-10, "rank %d, %s: after the failure, the healthy solve's error is %g", me, kind, err);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  const SpMat A = oseen_like(30, 3.0, 25);
  const int n = static_cast<int>(A.rows());
  const double asym = (A - SpMat(A.transpose())).norm() / A.norm();
  CHECK(asym > 0.1, "test matrix is not strongly nonsymmetric (%g)", asym);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1, 1);
  Eigen::VectorXd xs(n);
  for (int i = 0; i < n; ++i) xs(i) = U(rng);
  const Eigen::VectorXd b = A * xs;
  std::vector<double> bv(b.data(), b.data() + n);

  // 1. storage orders
  {
    const CsrMatrix c1 = CsrMatrix::from_eigen(A);
    const CsrMatrix c2 = CsrMatrix::from_eigen(Eigen::SparseMatrix<double, Eigen::RowMajor>(A));
    CHECK(c1.rowptr == c2.rowptr && c1.col == c2.col && c1.val == c2.val, "from_eigen differs by storage order");
    std::vector<double> y(n);
    c1.mult(xs.data(), y.data());
    CHECK(rel(y, b) < 1e-14, "CsrMatrix::mult is not A*x (%g)", rel(y, b));
  }
  const DistCsr D = DistCsr::from_serial(CsrMatrix::from_eigen(A), MPI_COMM_SELF);

  // 2. every available backend
  for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
    if (!lu_backend_available(be)) {
      std::printf("skip %s (not in this build)\n", to_string(be));
      continue;
    }
    axmp::strumpack_guard_stats(true);  // reset the guard's maxima
    SparseLU lu(be);
    lu.factorize(D);
    const auto x = lu.solve(bv);
    const double ex = rel(x, xs);
    std::printf("%-14s err %.2e residual %.2e fill %ld\n", to_string(be), ex, lu.stats().rel_residual,
                lu.stats().factor_nnz);
    CHECK(ex < 1e-10, "%s: solution error %g", to_string(be), ex);
    CHECK(lu.stats().rel_residual < 1e-13, "%s: residual %g", to_string(be), lu.stats().rel_residual);
    CHECK(lu.stats().factor_nnz > 0, "%s: no fill reported", to_string(be));
    // 5. the STRUMPACK guard
    if (be == LUBackend::StrumpackLU) {
      const auto st = axmp::strumpack_guard_stats(false);
      CHECK(st.max_concurrent >= 1 && st.max_concurrent_during_life == 1,
            "strumpack: the calls did not go through the STRUMPACK guard (max_concurrent %d, during a "
            "construction %d)", st.max_concurrent, st.max_concurrent_during_life);
      if (st.internals_known)   // the concurrent guard (STRUMPACK 8.0.x) empties them; the serialized one cannot see them
        CHECK(st.timer_entries == 0, "strumpack: %zu entries left in STRUMPACK's timer lists", st.timer_entries);
    }
  }

  // 3. the CSC-as-CSR hand-over solves with A^T: it must be caught by the check above
  {
    CsrMatrix wrong;
    wrong.n = n;
    wrong.rowptr.assign(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);  // column pointers, posing as rows
    wrong.col.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    wrong.val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());
    const DistCsr Dw = DistCsr::from_serial(wrong, MPI_COMM_SELF);
    for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU}) {
      if (!lu_backend_available(be)) continue;
      SparseLU lu(be);
      lu.factorize(Dw);
      const auto x = lu.solve(bv);
      const double ex = rel(x, xs);
      CHECK(ex > 1e-3, "%s: CSC passed as CSR went undetected (error %g)", to_string(be), ex);
    }
  }

  // 4. structurally singular saddle point (a zero column) is reported, not solved
  {
    SpMat S(3, 3);
    S.insert(0, 0) = 1;
    S.insert(1, 0) = 1;
    S.insert(0, 1) = 2;
    S.insert(1, 1) = 2;  // rank 1 in the top-left, third row/col empty -> singular
    S.makeCompressed();
    const DistCsr Ds = DistCsr::from_serial(CsrMatrix::from_eigen(SpMat(S)), MPI_COMM_SELF);
    bool threw = false;
    try {
      SparseLU lu(LUBackend::EigenLU);
      lu.factorize(Ds);
    } catch (const axmp::Error&) {
      threw = true;
    }
    CHECK(threw, "Eigen SparseLU accepted a singular matrix silently");
  }

  // 5. backend names
  CHECK(lu_backend_from_string("strumpack-mpi") == LUBackend::StrumpackMPI, "backend name parse");
  bool threw = false;
  try {
    lu_backend_from_string("umfpack");
  } catch (const axmp::Error&) {
    threw = true;
  }
  CHECK(threw, "unknown backend name accepted");

  // 6. distributed (MPI_COMM_WORLD)
  {
    int P = 1, me = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    const CsrMatrix C = CsrMatrix::from_eigen(A);
    // uneven rows 1:2:3:...; rank 1 empty when P > 1
    std::vector<int> dist(P + 1, 0);
    {
      std::vector<double> w(P);
      double tot = 0;
      for (int p = 0; p < P; ++p) tot += (w[p] = (P > 1 && p == 1) ? 0.0 : p + 1.0);
      double acc = 0;
      for (int p = 0; p < P; ++p) {
        acc += w[p];
        dist[p + 1] = static_cast<int>(std::lround(n * acc / tot));
      }
      dist[P] = n;
    }
    const DistCsr Dd = DistCsr::distribute(C, dist, MPI_COMM_WORLD);
    std::vector<double> bl(bv.begin() + dist[me], bv.begin() + dist[me + 1]);
    for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
      if (!lu_backend_available(be)) continue;
      SparseLU lu(be);
      lu.factorize(Dd);
      const auto x = lu.solve(bl);
      double loc = 0, gl = 0;
      for (int i = 0; i < Dd.local_rows; ++i) loc += std::pow(x[i] - xs(dist[me] + i), 2);
      MPI_Allreduce(&loc, &gl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double ex = std::sqrt(gl) / xs.norm();
      if (me == 0) std::printf("distributed %d ranks %-14s err %.2e residual %.2e (plain %.2e, %d steps)\n", P,
                               to_string(be), ex, lu.stats().rel_residual, lu.stats().rel_residual_before,
                               lu.stats().refine_steps);
      CHECK(ex < 1e-10 && lu.stats().rel_residual <= 1e-12, "distributed %s: error %g", to_string(be), ex);
    }
    // duplicates: refused; summed by sum_duplicates()
    DistCsr dup = Dd;
    {
      std::vector<int> rp(1, 0), cc;
      std::vector<double> vv;
      for (int i = 0; i < dup.local_rows; ++i) {
        for (int k = dup.rowptr[i]; k < dup.rowptr[i + 1]; ++k) {  // split every entry in two halves
          cc.push_back(dup.col[k]);
          vv.push_back(0.5 * dup.val[k]);
          cc.push_back(dup.col[k]);
          vv.push_back(0.5 * dup.val[k]);
        }
        rp.push_back(static_cast<int>(cc.size()));
      }
      dup.rowptr.swap(rp);
      dup.col.swap(cc);
      dup.val.swap(vv);
    }
    bool threw = false;
    try {
      SparseLU lu(LUBackend::EigenLU);
      lu.factorize(dup);
    } catch (const axmp::Error&) {
      threw = true;
    }
    CHECK(threw, "duplicate CSR entries were not refused");
    dup.sum_duplicates();
    {
      SparseLU lu(LUBackend::EigenLU);
      lu.factorize(dup);
      lu.solve(bl);
      CHECK(lu.stats().rel_residual <= 1e-12, "after sum_duplicates: residual %g", lu.stats().rel_residual);
    }
    // numerically singular (row 3 = row 1 + row 2 of a small dense-ish matrix): every rank must throw
    {
      SpMat S0 = oseen_like(6, 2.0, 0);
      Eigen::SparseMatrix<double, Eigen::RowMajor> R(S0);
      R.row(2) = R.row(0) + R.row(1);
      R.makeCompressed();
      R.prune(0.0);
      const CsrMatrix Cs = CsrMatrix::from_eigen(R);
      std::vector<int> ds(P + 1, 0);
      for (int p = 0; p <= P; ++p) ds[p] = static_cast<int>(static_cast<long>(Cs.n) * p / P);
      const DistCsr Ds = DistCsr::distribute(Cs, ds, MPI_COMM_WORLD);
      std::vector<double> bs(Ds.local_rows, 1.0);
      for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
        if (!lu_backend_available(be)) continue;
        int mine = 0;
        try {
          SparseLU lu(be);
          lu.factorize(Ds);
          lu.solve(bs);  // a backend that factors a singular matrix must still miss the residual target
        } catch (const axmp::Error&) {
          mine = 1;
        }
        int all = 0;
        MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        CHECK(all == 1, "%s: a singular matrix did not fail on every rank", to_string(be));
      }
    }
  }

  // 7-11: the index width of Eigen's factorization and its failures
  {
    int P = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    // a launcher that starts N one-rank jobs instead of one N-rank job must not pass for the 2-rank registration
    if (const char* e = std::getenv("AXMP_EXPECT_RANKS"))
      CHECK(P == std::atoi(e), "started on %d rank(s), expected %s", P, e);
    auto section = [](const char* name, auto&& body) {
      try {
        body();
      } catch (const std::exception& e) {
        CHECK(false, "section %s: %s", name, e.what());
      }
    };
    section("7 (index rule)", [] { index_rule(); });
    section("8 (64-bit path)", [] { index64_path(MPI_COMM_WORLD); });
    section("9 (collective failures)", [] { failures_collective(MPI_COMM_WORLD); });
    section("10 (fresh solver)", [] { fresh_solver(); });
    section("11 (failure before Eigen, 32-bit first)", [] { failure_before_eigen(MPI_COMM_WORLD, 0); });
    section("11 (failure before Eigen, 64-bit first)", [] { failure_before_eigen(MPI_COMM_WORLD, 100); });
  }

  std::printf(failures ? "test_fluids_sparse_lu: %d FAILURES\n" : "test_fluids_sparse_lu: OK\n", failures);
  MPI_Finalize();
  return failures ? 1 : 0;
}
