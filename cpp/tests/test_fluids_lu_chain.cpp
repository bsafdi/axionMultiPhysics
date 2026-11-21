// fluids sparse LU, the recorded retry chain inside SparseLU (cpp/include/axionmultiphysics/fluids/sparse_lu.hpp:
// LUStage, default_chain, LUAcceptance, LUAttempt, lu_backward_error, DistCsr::abs_mult, the blockwise kernel probe and
// the test hooks).
//
// Registered on 1 and 2 ranks with a TIMEOUT. Every SparseLU here runs on a duplicate of MPI_COMM_WORLD while the
// checks run on MPI_COMM_WORLD itself: a rank that leaves a chain on another stage than rank 0 waits in a collective
// call of the LU's communicator that the other rank never enters, so it hangs, and the TIMEOUT ends the test. Every
// chain's attempt records are compared between the ranks bit for bit (rank 0 decides and its numbers are recorded).
//
//   1. Stage names and the stage lists of default_chain (the caller's STRUMPACK backend with its matching, then
//      with MC64 job 2, then Eigen up to kLUEigenRetryMaxN unknowns, and never without matching; the 3D list, first
//      without matching: without, then with it, then Eigen up to kLUEigenRetryMaxN3D; Eigen alone, uncapped), the
//      test-only override, and the gates and caps that measurement confirmed (kept as they are).
//   2. DistCsr::abs_mult and lu_backward_error on a 6 x 6 example worked by hand: two blocks, a Dirichlet row (-1)
//      and two constraint rows (-2) on two ranks.
//   3. The backward error is invariant under scaling one row block by 2^27, on ONE fixed solution, bit for bit, while
//      the relative residual of the same solution grows by orders of magnitude; the scaled SOLVE (another
//      factorization) has a backward error at round-off and a relative residual orders of magnitude above it.
//   4. One-stage chains are today's single attempt: SparseLU(LUBackend) and SparseLU({stage}) give the same solution
//      bit for bit, meet the manufactured solution, and record one accepted attempt with its backward error.
//   5. The forced outcomes of the test hook (threw, missed_target, refused) on stage 0 of a two-stage chain: outcome,
//      reason and the stage of the accepted solve; the hook's count; a stage whose backend is not in this build.
//   6. RelResidual acceptance rejects a stage whose relative residual misses the bound WHATEVER its backward error:
//      a block scaled by 2^27 with a zero right-hand side on it makes the two numbers straddle the bound.
//   7. Verdict acceptance: called once per iterate on every rank, collective, rank 0's value decides and is recorded.
//   8. The accepted stage is sticky over ten solves of one factorization; a later miss walks on from it; factorize()
//      starts again at stage 0; at most one backend object is alive at any time.
//   9. The perturbation hook's numbers against the test's own long double evaluation; with no stage accepted the
//      best iterate by the acceptance measure is returned (ties: the smaller backward error) and stats() describes it.
//  10. Rank 0 decides: the rank-1 replacement of the acceptance inputs leaves the records identical on every rank.
//  11. The blockwise kernel probe on a small staggered-grid Stokes matrix whose kernel is known by construction: the
//      constant pressure left in is refused at every length unit from 1e-6 to 1e6; pinned or bordered it is accepted
//      at every unit (where the unscaled probe refuses the bordered matrix at 1e-6); without row blocks, today's probe.
//  12. STRUMPACK builds: the MC64 job in the options of the solver object is the stage's.
//  13. A chain whose every stage fails to factorize throws on every rank with every stage's reason.
//  14. The constructor and set_acceptance refuse what is not available yet.
//  15. With a RelResidual acceptance and no stage accepted, the iterate with the smaller relative residual is
//      returned although the other stage's backward error is smaller: two stages perturbed differently (the iterate
//      hook), so that the two measures order them oppositely.
//  16. After a solve that no stage accepted, the next solve's record and stats() are those of the stage whose
//      factorization is held, not of the stage whose iterate was returned.
//  17. A non-finite iterate is rejected whatever its measure and is not the best iterate while a finite one exists; a
//      NaN measure counts as +infinity in the choice of the best iterate.
//  18. Wrong row blocks on one rank are an error on every rank, and an x or b of the wrong size is an error whose
//      hint is not the row blocks' one; the blocks are read by factorize() (a solve after set_row_blocks with other
//      ids records the backward error of the blocks factorize() read).
//  19. A stage's size cap (LUStage::max_n): above it the stage is "refused" with a reason "size...", at it the stage
//      factorizes; the first stage refused by its cap passes the solve on to the next.
//  20. An iterate whose residual overflows has no backward error: lu_backward_error and the attempt record NaN, not
//      the 0 that a maximum skipping the NaN block would give; in a tie of the acceptance measure the NaN counts as
//      +infinity, so the iterate with a finite backward error is returned although it came second.
//  21. A walk whose only later stage is refused by its size cap keeps the factorization it
//      holds: a solve that every stage misses leaves the held stage usable, so the next solve solves (a sticky mass
//      LU solves many times on one factorization) instead of failing with "no factorization".
//  22. STRUMPACK builds: the default chains walked to their last stage (the iterate hook makes the first two miss):
//      every stage is tried in the order of the list and recorded under its name, the STRUMPACK stages with their own
//      MC64 job; an explicit chain is walked as given, so a stage without matching stays available on request.
#include <mpi.h>

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fluids/sparse_lu.hpp"

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
using T = Eigen::Triplet<double>;

// ------------------------------------------------------------------------------------------------ helpers
// [ L  G ; D  0 ]: L = -Lap + Pe (c . grad) on a g x g grid (central convection: nonsymmetric), G a difference
// "gradient" (g*g x m), D = G^T scaled differently: a nonsymmetric saddle point with an m x m zero block.
static SpMat oseen_like(int g, double pe, int m) {
  const int n = g * g;
  std::vector<T> t;
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

static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
static bool starts(const std::string& s, const std::string& head) { return s.compare(0, head.size(), head) == 0; }

static int rank_of(MPI_Comm c) {
  int r = 0;
  MPI_Comm_rank(c, &r);
  return r;
}
static int size_of(MPI_Comm c) {
  int s = 1;
  MPI_Comm_size(c, &s);
  return s;
}

// true on every rank iff it holds on every rank of comm
static bool on_all(bool mine, MPI_Comm comm) {
  int m = mine ? 1 : 0, all = 0;
  MPI_Allreduce(&m, &all, 1, MPI_INT, MPI_MIN, comm);
  return all == 1;
}

// true on every rank iff every rank of comm passed the same bytes. Collective.
static bool same_on_all_ranks(const std::string& s, MPI_Comm comm) {
  int len = static_cast<int>(s.size()), lo = 0, hi = 0;
  MPI_Allreduce(&len, &lo, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&len, &hi, 1, MPI_INT, MPI_MAX, comm);
  if (lo != hi) return false;
  std::string r0 = s;
  MPI_Bcast(r0.data(), len, MPI_CHAR, 0, comm);
  return on_all(r0 == s, comm);
}

static std::string hex(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%a", v);
  return b;
}
static bool bitwise(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
static bool bitwise(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// Every field of every attempt (doubles in hexadecimal), and what stats() and accepted_stage() say about the solve.
static std::string records(const SparseLU& lu) {
  std::string s;
  for (const auto& a : lu.attempts()) {
    s += a.stage + "|" + a.outcome + "|" + a.reason + "|" + hex(a.measure) + "|" + hex(a.bound) + "|" +
         hex(a.rel_residual_before) + "|" + hex(a.rel_residual) + "|" + hex(a.gate_residual) + "|" +
         hex(a.backward_error) + "|" + std::to_string(a.constraint_ratios.size()) + ":";
    for (double c : a.constraint_ratios) s += hex(c) + ",";
    s += "|" + std::to_string(a.refine_steps) + "|" + std::to_string(a.krylov_iterations) + "|" +
         std::to_string(a.factor_nnz) + "|" + std::to_string(a.index_bits) + "|" + hex(a.analyze_seconds) + "|" +
         hex(a.factor_seconds) + "|" + hex(a.solve_seconds) + "\n";
  }
  const LUStats& st = lu.stats();
  s += "accepted " + std::to_string(lu.accepted_stage()) + " stats " + st.stage + " " + hex(st.measure) + " " +
       hex(st.backward_error) + " " + std::to_string(st.attempts) + " " + hex(st.rel_residual);
  return s;
}
static void check_records_same(const SparseLU& lu, MPI_Comm comm, const char* what) {
  const std::string r = records(lu);
  const bool same = same_on_all_ranks(r, comm);
  CHECK(same, "%s: the attempt records differ between the ranks (rank %d's:\n%s)", what, rank_of(comm), r.c_str());
}

static std::string outcomes(const SparseLU& lu) {
  std::string s;
  for (const auto& a : lu.attempts()) s += (s.empty() ? "" : ",") + a.outcome;
  return s;
}

// S distributed over comm with uneven rows (rank p holds a share proportional to p + 1); dist receives the offsets.
static DistCsr distributed(const SpMat& S, MPI_Comm comm, std::vector<int>& dist) {
  const int P = size_of(comm);
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

template <class V>
static std::vector<double> local_of(const V& g, const std::vector<int>& dist, int me) {
  std::vector<double> l(dist[me + 1] - dist[me]);
  for (int i = dist[me]; i < dist[me + 1]; ++i) l[i - dist[me]] = g[i];
  return l;
}
static std::vector<int> local_ints(const std::vector<int>& g, const std::vector<int>& dist, int me) {
  return std::vector<int>(g.begin() + dist[me], g.begin() + dist[me + 1]);
}

// ||x - xs|| / ||xs|| over all ranks (x local, xs global)
static double global_err(const std::vector<double>& x, const Eigen::VectorXd& xs, const std::vector<int>& dist,
                         MPI_Comm comm) {
  const int me = rank_of(comm);
  double loc[2] = {0, 0}, glob[2] = {0, 0};
  for (int i = 0; i < dist[me + 1] - dist[me]; ++i) {
    const double d = x[i] - xs(dist[me] + i);
    loc[0] += d * d;
    loc[1] += xs(dist[me] + i) * xs(dist[me] + i);
  }
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
  return std::sqrt(glob[0] / glob[1]);
}

// The gathered vector (every rank)
static std::vector<double> gather_all(const std::vector<double>& xl, const std::vector<int>& dist, MPI_Comm comm) {
  const int P = size_of(comm);
  std::vector<int> counts(P);
  for (int p = 0; p < P; ++p) counts[p] = dist[p + 1] - dist[p];
  std::vector<double> x(dist[P]);
  MPI_Allgatherv(xl.data(), static_cast<int>(xl.size()), MPI_DOUBLE, x.data(), counts.data(), dist.data(),
                 MPI_DOUBLE, comm);
  return x;
}

// The test's own evaluation, in long double, of what the chain records for an iterate x of D x = b: the relative
// residual over all rows, the blockwise backward error (blocks as lu_backward_error), and the constraint ratios in
// global row order. Independent of the library's code: x, D and b are doubles, so the long double evaluation errs by
// about 1e-19 of |D| |x|, far below the residuals it is compared with (those of a perturbed iterate, about 1e-7).
struct Ref {
  double rel_residual = 0, gate_residual = 0, backward_error = 0;
  std::vector<double> constraint_ratios;
};
static Ref reference(const DistCsr& D, const std::vector<double>& xl, const std::vector<double>& bl,
                     const std::vector<int>& blocks, const std::vector<char>& mask, const std::vector<int>& dist,
                     MPI_Comm comm) {
  const std::vector<double> x = gather_all(xl, dist, comm);
  int kmax = blocks.empty() ? 0 : -1;  // no blocks: one block of all rows
  for (int k : blocks) kmax = std::max(kmax, k);
  int kg = -1;
  MPI_Allreduce(&kmax, &kg, 1, MPI_INT, MPI_MAX, comm);
  const int nb = kg + 1;
  // sums: [0] r^2, [1] b^2, [2] masked r^2, [3] masked b^2, then per block (r^2, d^2)
  std::vector<long double> loc(4 + 2 * nb, 0.0L), glob(4 + 2 * nb, 0.0L);
  std::vector<double> ratios;
  for (int i = 0; i < D.local_rows; ++i) {
    long double ax = 0, ad = 0;
    for (int k = D.rowptr[i]; k < D.rowptr[i + 1]; ++k) {
      ax += static_cast<long double>(D.val[k]) * x[D.col[k]];
      ad += std::fabs(static_cast<long double>(D.val[k]) * x[D.col[k]]);
    }
    const long double r = bl[i] - ax, d = ad + std::fabs(static_cast<long double>(bl[i]));
    loc[0] += r * r;
    loc[1] += static_cast<long double>(bl[i]) * bl[i];
    if (!mask.empty() && mask[i]) {
      loc[2] += r * r;
      loc[3] += static_cast<long double>(bl[i]) * bl[i];
    }
    const int b = blocks.empty() ? 0 : blocks[i];
    if (b >= 0) {
      loc[4 + 2 * b] += r * r;
      loc[5 + 2 * b] += d * d;
    } else if (b == -2) {
      ratios.push_back(d > 0 ? static_cast<double>(std::fabs(r) / d) : 0.0);
    }
  }
  MPI_Allreduce(loc.data(), glob.data(), static_cast<int>(loc.size()), MPI_LONG_DOUBLE, MPI_SUM, comm);
  Ref out;
  out.rel_residual = static_cast<double>(std::sqrt(glob[0] / glob[1]));
  out.gate_residual = glob[3] > 0 ? static_cast<double>(std::sqrt(glob[2] / glob[3])) : -1;
  long double m = 0;
  for (int b = 0; b < nb; ++b)
    if (glob[5 + 2 * b] > 0) m = std::max(m, std::sqrt(glob[4 + 2 * b] / glob[5 + 2 * b]));
  out.backward_error = static_cast<double>(m);
  // the constraint rows in global row order: rows are distributed in rank order
  const int P = size_of(comm);
  int nl = static_cast<int>(ratios.size());
  std::vector<int> cnt(P), off(P + 1, 0);
  MPI_Allgather(&nl, 1, MPI_INT, cnt.data(), 1, MPI_INT, comm);
  for (int p = 0; p < P; ++p) off[p + 1] = off[p] + cnt[p];
  out.constraint_ratios.resize(off[P]);
  MPI_Allgatherv(ratios.data(), nl, MPI_DOUBLE, out.constraint_ratios.data(), cnt.data(), off.data(), MPI_DOUBLE,
                 comm);
  return out;
}

// relative difference, guarded against zero
static double reldiff(double a, double b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-300); }

// ||b - D x|| / ||b|| in long double (x local)
static double rel_res_ld(const DistCsr& D, const std::vector<double>& xl, const std::vector<double>& bl,
                         const std::vector<int>& dist, MPI_Comm comm) {
  return reference(D, xl, bl, {}, {}, dist, comm).rel_residual;
}

// The test hooks are process-wide: put them back when a section ends, however it ends.
struct HookReset {
  ~HookReset() {
    try {
      set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    } catch (...) {
    }
    try {
      set_lu_solution_perturbation_for_testing(0.0);
    } catch (...) {
    }
    try {
      set_lu_rank1_acceptance_flip_for_testing(false);
    } catch (...) {
    }
    try {
      set_default_lu_chain_for_testing({});
    } catch (...) {
    }
    try {
      set_lu_iterate_hook_for_testing({});
    } catch (...) {
    }
  }
};

static LUStage stage(LUBackend b, MatchingJob m = MatchingJob::MaxDiagonalProductScaling, bool eq = false) {
  LUStage s;
  s.backend = b;
  s.matching = m;
  s.equilibrate = eq;
  return s;
}
static std::vector<LUStage> eigens(int k) { return std::vector<LUStage>(k, stage(LUBackend::EigenLU)); }

static LUAcceptance rel_residual_acceptance(double bound) {
  LUAcceptance a;
  a.kind = LUAcceptance::Kind::RelResidual;
  a.bound = bound;
  return a;
}

// A Verdict whose value is scripted per call on this rank: values[k] on the k-th call (the last value repeats). It
// counts its calls, keeps a copy of every iterate it is given, and is collective (an Allreduce on the LU's
// communicator), as the caller's verdicts are.
struct ScriptedVerdict {
  std::vector<double> values;
  MPI_Comm comm = MPI_COMM_NULL;
  int calls = 0;
  std::vector<std::vector<double>> seen;
  LUAcceptance acceptance(double bound, const std::string& name = "") {
    LUAcceptance a;
    a.kind = LUAcceptance::Kind::Verdict;
    a.bound = bound;
    a.name = name;
    a.verdict = [this](const std::vector<double>& x) {
      double s = 0, g = 0;
      for (double v : x) s += v;
      MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);  // collective, like the projection's residual
      seen.push_back(x);
      const double v = values[std::min<std::size_t>(calls, values.size() - 1)];
      ++calls;
      return v;
    };
    return a;
  }
};

// The system of sections 4-10: oseen_like(30, 3, m) (n = 900 + m; m = 25: a saddle point, which STRUMPACK factorizes
// only with matching; m = 0: no zero block, for the matching-free stages) distributed on the LU's communicator, a
// manufactured solution xs and b = A xs.
struct Sys {
  SpMat S;
  DistCsr D;
  std::vector<int> dist;
  Eigen::VectorXd xs, b;
  std::vector<double> bl;
  int me = 0;
};
static Sys make_sys(MPI_Comm lc, unsigned seed = 7, int m = 25) {
  Sys s;
  s.S = oseen_like(30, 3.0, m);
  s.D = distributed(s.S, lc, s.dist);
  s.me = rank_of(lc);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1, 1);
  s.xs.resize(s.S.rows());
  for (int i = 0; i < s.S.rows(); ++i) s.xs(i) = U(rng);
  s.b = s.S * s.xs;
  s.bl = local_of(s.b, s.dist, s.me);
  return s;
}

// ------------------------------------------------------------------------------------------------ 1
// Names and default_chain. Reference: the stage spelling of sparse_lu.hpp, and the measured lists. A STRUMPACK first
// stage with MC64 job 5 is retried with MC64 job 2 and then with Eigen: on the systems that job 5 misses (the planar
// fine-mesh Womersley starters and 28 bordered projections on an AMD EPYC 7662, 30 on an Intel Xeon Gold 6330) job 2
// rescued every one, at STRUMPACK's cost (0.31 s against Eigen's 4.0 to 4.3 s on the starters), while the stage
// without matching rescued none that job 2 did not and threw ZERO_PIVOT on every starter, so it is not in that list.
// The 3D list (first stage without matching) is retried with matching (the Stokes solver's former retry) and then
// with Eigen: every 3D miss of the first stage was a ZERO_PIVOT that job 5 rescued. The Eigen retry stage is capped at
// 3e5 unknowns (2D) and 5e4 (3D: one factorization at n = 49 073 took 642 s); Eigen alone has no retry and no cap. A
// first stage with MC64 job 2 is no caller's; its list is held to the rules every list keeps: the caller's stage
// first, no configuration twice (a deterministic retry of the same configuration rescues nothing), no stage without
// matching after a matched one, Eigen last and capped.
static std::string names(const std::vector<LUStage>& c) {
  std::string s;
  for (const auto& st : c) s += (s.empty() ? "" : ",") + st.name();
  return s;
}
static const char* job_name(MatchingJob m) {
  return m == MatchingJob::None ? "none" : m == MatchingJob::MaxSmallestDiagonal ? "mc64-2" : "mc64-5";
}
static void names_and_default_chains() {
  HookReset reset;
  using M = MatchingJob;
  struct N {
    LUStage s;
    const char* name;
  };
  const N cases[] = {
      {stage(LUBackend::EigenLU), "eigen"},
      {stage(LUBackend::EigenLU, M::None), "eigen"},
      {stage(LUBackend::EigenLU, M::MaxSmallestDiagonal), "eigen"},
      {stage(LUBackend::EigenLU, M::MaxDiagonalProductScaling, true), "eigen:eq"},
      {stage(LUBackend::StrumpackLU), "strumpack:mc64-5"},
      {stage(LUBackend::StrumpackLU, M::None), "strumpack:none"},
      {stage(LUBackend::StrumpackLU, M::MaxSmallestDiagonal), "strumpack:mc64-2"},
      {stage(LUBackend::StrumpackLU, M::MaxDiagonalProductScaling, true), "strumpack:mc64-5:eq"},
      {stage(LUBackend::StrumpackMPI), "strumpack-mpi:mc64-5"},
      {stage(LUBackend::StrumpackMPI, M::None), "strumpack-mpi:none"},
      {stage(LUBackend::StrumpackMPI, M::MaxSmallestDiagonal), "strumpack-mpi:mc64-2"},
  };
  for (const auto& c : cases)
    CHECK(c.s.name() == c.name, "stage name '%s', expected '%s'", c.s.name().c_str(), c.name);
  {
    const LUStage d;  // the defaults: Eigen, MC64 job 5, no equilibration
    CHECK(d.backend == LUBackend::EigenLU && d.matching == M::MaxDiagonalProductScaling && !d.equilibrate,
          "LUStage's defaults are not {EigenLU, MaxDiagonalProductScaling, false}");
  }
  struct DC {
    LUBackend first;
    M job;
    const char* chain;
    std::vector<long> caps;  // LUStage::max_n of each stage
  };
  CHECK(kLUEigenRetryMaxN == 300000 && kLUEigenRetryMaxN3D == 50000, "the Eigen retry caps are %ld and %ld, expected "
        "300000 (2D) and 50000 (3D)", kLUEigenRetryMaxN, kLUEigenRetryMaxN3D);
  // The callers' gates, confirmed by measurement and kept: the steps of Boussinesq, GLS and the transport Jacobian
  // (max(newton_rtol, kFinalSolveResidualMax), and max(1e-2 rtol, kFinalSolveResidualMax) for transport) never
  // escalated on a system with a solution, the largest measure being 0.068, 0.0037 and 0.0014 of its bound.
  CHECK(kFinalSolveResidualMax == 1e-11 && kDualMassResidualMax == 1e-10 && kProjectionResidualMax == 1e-13,
        "the gates are %.3g (final solve), %.3g (dual-norm mass solves), %.3g (projection), expected 1e-11, 1e-10 "
        "and 1e-13", kFinalSolveResidualMax, kDualMassResidualMax, kProjectionResidualMax);
  const long c2 = kLUEigenRetryMaxN, c3 = kLUEigenRetryMaxN3D;
  const DC lists[] = {
      {LUBackend::EigenLU, M::MaxDiagonalProductScaling, "eigen", {0}},
      {LUBackend::EigenLU, M::None, "eigen", {0}},
      {LUBackend::EigenLU, M::MaxSmallestDiagonal, "eigen", {0}},
      {LUBackend::StrumpackLU, M::MaxDiagonalProductScaling, "strumpack:mc64-5,strumpack:mc64-2,eigen", {0, 0, c2}},
      {LUBackend::StrumpackMPI, M::MaxDiagonalProductScaling, "strumpack-mpi:mc64-5,strumpack-mpi:mc64-2,eigen",
       {0, 0, c2}},
      {LUBackend::StrumpackLU, M::None, "strumpack:none,strumpack:mc64-5,eigen", {0, 0, c3}},
      {LUBackend::StrumpackMPI, M::None, "strumpack-mpi:none,strumpack-mpi:mc64-5,eigen", {0, 0, c3}},
  };
  for (const auto& c : lists) {
    const std::vector<LUStage> chain = default_chain(c.first, c.job);
    const std::string got = names(chain);
    CHECK(got == c.chain, "default_chain(%s, %s) = [%s], expected [%s]", to_string(c.first),
          job_name(c.job), got.c_str(), c.chain);
    std::vector<long> caps;
    for (const auto& st : chain) caps.push_back(st.max_n);
    CHECK(caps == c.caps, "default_chain(%s, %s): the size caps of its stages are not the expected ones (stage 0: "
          "%ld; last: %ld)", to_string(c.first), job_name(c.job), caps.empty() ? -1L : caps.front(),
          caps.empty() ? -1L : caps.back());
    for (const auto& st : chain)
      CHECK(!st.equilibrate, "default_chain(%s, %s) holds the equilibrated stage %s, which measurement did not "
            "adopt", to_string(c.first), job_name(c.job), st.name().c_str());
  }
  // the lists after a matched first stage hold no stage without matching
  for (LUBackend be : {LUBackend::StrumpackLU, LUBackend::StrumpackMPI})
    for (M job : {M::MaxDiagonalProductScaling, M::MaxSmallestDiagonal})
      for (const auto& st : default_chain(be, job))
        CHECK(!(st.backend != LUBackend::EigenLU && st.matching == M::None), "default_chain(%s, %s) = [%s] holds "
              "a stage without matching", to_string(be), job_name(job), names(default_chain(be, job)).c_str());
  // a first stage with MC64 job 2 (no caller's): the caller's stage first, no configuration twice, Eigen last, capped
  for (LUBackend be : {LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
    const std::vector<LUStage> chain = default_chain(be, M::MaxSmallestDiagonal);
    const std::string got = names(chain);
    std::vector<std::string> seen;
    bool twice = false;
    for (const auto& st : chain) {
      twice = twice || std::find(seen.begin(), seen.end(), st.name()) != seen.end();
      seen.push_back(st.name());
    }
    const std::string first = std::string(to_string(be)) + ":mc64-2";
    CHECK(chain.size() >= 2 && chain.front().name() == first && chain.front().max_n == 0 && !twice &&
              chain.back().backend == LUBackend::EigenLU && chain.back().max_n == c2,
          "default_chain(%s, mc64-2) = [%s]: expected %s first (uncapped), no configuration twice and Eigen last, capped "
          "at %ld", to_string(be), got.c_str(), first.c_str(), c2);
  }
  CHECK(LUStage().max_n == 0, "LUStage's default size cap is %ld, expected 0 (none)", LUStage().max_n);
  // the override: every caller's chain, whatever its first backend; cleared by an empty chain
  set_default_lu_chain_for_testing(eigens(2));
  for (const auto& c : lists) {
    const std::string got = names(default_chain(c.first, c.job));
    CHECK(got == "eigen,eigen", "under the override, default_chain(%s, %s) = [%s], expected [eigen,eigen]",
          to_string(c.first), job_name(c.job), got.c_str());
  }
  set_default_lu_chain_for_testing({});
  const std::string back = names(default_chain(LUBackend::StrumpackLU, M::None));
  CHECK(back == "strumpack:none,strumpack:mc64-5,eigen", "after clearing the override: [%s]", back.c_str());
  // an explicit chain is the chain: chain_or_default keeps it as given (names and caps), even under the override
  std::vector<LUStage> given = {stage(LUBackend::StrumpackLU, M::None), stage(LUBackend::EigenLU)};
  given[1].max_n = 7;
  set_default_lu_chain_for_testing(eigens(3));
  for (M job : {M::MaxDiagonalProductScaling, M::None}) {
    const std::vector<LUStage> c = chain_or_default(given, LUBackend::StrumpackLU, job);
    CHECK(names(c) == "strumpack:none,eigen" && c.size() == 2 && c[0].max_n == 0 && c[1].max_n == 7,
          "chain_or_default of the explicit chain [strumpack:none, eigen/7] (first matching %s) = [%s], expected it as "
          "given", job_name(job), names(c).c_str());
  }
  set_default_lu_chain_for_testing({});
}

// ------------------------------------------------------------------------------------------------ 2
// abs_mult and lu_backward_error on a 6 x 6 example worked by hand, x = (1, -2, 1, 2, -1, 1):
//   row  entries               A x   |A||x|  b    r = b - A x   d = |A||x| + |b|   block
//   0    (0,0)=2  (0,1)=1       0     4      3      3             7                 0
//   1    (1,1)=-3 (1,2)=4      10    10     14      4            24                 0
//   2    (2,2)=4  (2,3)=3      10    10     11      1            21                 1
//   3    (3,0)=1  (3,4)=1       0     2      1      1             3                -2 (constraint)
//   4    (4,4)=1               -1     1      5      6             6                -1 (Dirichlet)
//   5    (5,0)=-1 (5,5)=2       1     3      2      1             5                -2 (constraint)
// block 0: ||(3, 4)|| / ||(7, 24)|| = 5 / 25 = 0.2; block 1: 1 / 21; the maximum is 0.2. The constraint ratios are
// 1/3 (row 3) and 1/5 (row 5), in global row order. Rows 3 and 4 have ratios 1/3 and 1 above 0.2, so counting either
// in the maximum shows. With one block of all rows: ||r|| = sqrt(64) = 8, ||d|| = sqrt(1136).
// On two ranks rows 0-3 are on rank 0 and rows 4-5 on rank 1, so the constraint rows are on both.
static void backward_error_by_hand(MPI_Comm comm) {
  const int P = size_of(comm), me = rank_of(comm);
  SpMat A(6, 6);
  const T e[] = {{0, 0, 2}, {0, 1, 1}, {1, 1, -3}, {1, 2, 4}, {2, 2, 4}, {2, 3, 3},
                 {3, 0, 1}, {3, 4, 1}, {4, 4, 1},  {5, 0, -1}, {5, 5, 2}};
  A.setFromTriplets(std::begin(e), std::end(e));
  A.makeCompressed();
  std::vector<int> dist(P + 1, 6);
  dist[0] = 0;
  if (P > 1) dist[1] = 4;
  const DistCsr D = DistCsr::distribute(CsrMatrix::from_eigen(A), dist, comm);
  const std::vector<double> xg = {1, -2, 1, 2, -1, 1}, bg = {3, 14, 11, 1, 5, 2}, absax = {4, 10, 10, 2, 1, 3};
  const std::vector<int> blocks_g = {0, 0, 1, -2, -1, -2};
  const auto x = local_of(xg, dist, me), b = local_of(bg, dist, me);
  const auto blocks = local_ints(blocks_g, dist, me);

  std::vector<double> y(D.local_rows, -7.0);
  D.abs_mult(x.data(), y.data());
  bool ok = true;
  for (int i = 0; i < D.local_rows; ++i) ok = ok && y[i] == absax[dist[me] + i];
  CHECK(on_all(ok, comm), "rank %d: abs_mult is not |A| |x| = (4, 10, 10, 2, 1, 3)", me);

  std::vector<double> ratios;
  const double be = lu_backward_error(D, x, b, blocks, &ratios);
  CHECK(reldiff(be, 0.2) <= 1e-15, "rank %d: blockwise backward error %.17g, expected 0.2 (block 0)", me, be);
  CHECK(ratios.size() == 2 && reldiff(ratios[0], 1.0 / 3) <= 1e-15 && reldiff(ratios[1], 0.2) <= 1e-15,
        "rank %d: constraint ratios (%zu) %.17g %.17g, expected 1/3 and 1/5 in global row order", me, ratios.size(),
        ratios.empty() ? -1.0 : ratios[0], ratios.size() < 2 ? -1.0 : ratios[1]);
  CHECK(same_on_all_ranks(hex(be), comm), "rank %d: the backward error differs between the ranks", me);
  // the ids are labels: renumbered blocks (no rows with id 1) give the same maximum
  {
    std::vector<int> relabel_g = {2, 2, 0, -2, -1, -2};
    const double be2 = lu_backward_error(D, x, b, local_ints(relabel_g, dist, me));
    CHECK(reldiff(be2, 0.2) <= 1e-15, "rank %d: blocks relabelled {2, 2, 0}: %.17g, expected 0.2", me, be2);
  }
  // one block of all rows (the default): 8 / sqrt(1136); no constraint rows
  {
    std::vector<double> r2(3, 1.0);
    const double be1 = lu_backward_error(D, x, b, {}, &r2);
    const double ref = 8.0 / std::sqrt(1136.0);
    CHECK(reldiff(be1, ref) <= 1e-15 && r2.empty(), "rank %d: one block: %.17g (expected %.17g), %zu ratios", me, be1,
          ref, r2.size());
  }
  // an exact solution of this system: b = A x gives 0
  {
    std::vector<double> bx(D.local_rows);
    D.mult(x.data(), bx.data());
    const double be0 = lu_backward_error(D, x, bx, blocks);
    CHECK(be0 == 0.0, "rank %d: the backward error of an exact solution is %g", me, be0);
  }
}

// ------------------------------------------------------------------------------------------------ 3
// The invariance, on ONE fixed solution: b is zero on block 1 (the 25 constraint-like rows of oseen_like) and x is
// the unscaled solve's solution. Multiplying block 1's rows by 2^27 multiplies every product, sum, square and square
// root of that block by an exact power of two (no rounding changes, barring overflow and underflow, far from these
// numbers), so the backward error is the same bit for bit. The relative residual of the same x is not invariant:
// block 1's residual, at the round-off of its terms, is multiplied by 2^27 while ||b|| is not.
static void invariance(MPI_Comm comm, MPI_Comm lc) {
  const int me = rank_of(comm);
  const SpMat S = oseen_like(30, 3.0, 25);
  const int n = static_cast<int>(S.rows()), n0 = 900;
  std::vector<int> dist;
  const DistCsr D = distributed(S, lc, dist);
  std::mt19937 rng(23);
  std::uniform_real_distribution<double> U(-1, 1);
  Eigen::VectorXd bg = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n0; ++i) bg(i) = U(rng);
  std::vector<int> blocks_g(n);
  for (int i = 0; i < n; ++i) blocks_g[i] = i < n0 ? 0 : 1;
  const auto bl = local_of(bg, dist, me);
  const auto blocks = local_ints(blocks_g, dist, me);
  std::vector<double> x;
  {
    SparseLU lu(LUBackend::EigenLU);
    lu.factorize(D);
    x = lu.solve(bl);
  }
  DistCsr DA = D;  // block 1's rows times 2^27
  for (int i = 0; i < DA.local_rows; ++i)
    if (DA.row_begin + i >= n0)
      for (int k = DA.rowptr[i]; k < DA.rowptr[i + 1]; ++k) DA.val[k] = std::ldexp(DA.val[k], 27);
  const double be = lu_backward_error(D, x, bl, blocks), be_s = lu_backward_error(DA, x, bl, blocks);
  const double rr = rel_res_ld(D, x, bl, dist, comm), rr_s = rel_res_ld(DA, x, bl, dist, comm);
  if (me == 0)
    std::printf("invariance (one x): backward error %.3e unscaled, %.3e scaled by 2^27; relative residual %.3e -> "
                "%.3e (x %.2g)\n",
                be, be_s, rr, rr_s, rr_s / rr);
  CHECK(bitwise(be, be_s), "rank %d: the backward error of one x changed under a 2^27 block scaling: %a -> %a", me, be,
        be_s);
  // an accurate solve: within the normwise bound n u of a backward-stable solve (1.0e-13 here)
  CHECK(be <= n * std::ldexp(1.0, -53), "rank %d: the unscaled solve's backward error %g is not at round-off", me, be);
  // "orders of magnitude": at least 1e4 (2^27 = 1.3e8 if block 1's residual dominated before the scaling too)
  CHECK(rr_s >= 1e4 * rr, "rank %d: the relative residual of the same x grew only from %g to %g", me, rr, rr_s);

  // the scaled SOLVE: another factorization (partial pivoting sees other rows), so only round-off is asserted
  {
    SparseLU lu(LUBackend::EigenLU);
    lu.set_throw_on_miss(false);  // the relative residual cannot reach the refinement target 1e-12
    lu.set_row_blocks(blocks);
    lu.factorize(DA);
    const auto xs = lu.solve(bl);
    const double be2 = lu.attempts().empty() ? -1 : lu.attempts()[0].backward_error;
    const double rr2 = lu.stats().rel_residual;
    if (me == 0)
      std::printf("scaled solve: backward error %.3e, relative residual %.3e\n", be2, rr2);
    CHECK(lu.attempts().size() == 1 && be2 >= 0 && be2 <= 1e-15,
          "rank %d: the scaled solve's backward error %g is not at round-off (1e-15)", me, be2);
    CHECK(rr2 >= 1e3 * be2 && rr2 >= 1e-13,
          "rank %d: the scaled solve's relative residual %g is not orders of magnitude above its backward error %g", me,
          rr2, be2);
    CHECK(bitwise(be2, lu_backward_error(DA, xs, bl, blocks)) && bitwise(lu.stats().backward_error, be2),
          "rank %d: the recorded backward error is not lu_backward_error of the returned solution", me);
    check_records_same(lu, comm, "scaled solve");
  }
}

// ------------------------------------------------------------------------------------------------ 4
// One-stage chains: today's single attempt. Reference: the manufactured solution (round-off, platform independent);
// the bitwise comparison with today's code path is the before/after script of the change, not this test.
static void one_stage(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const int me = s.me;
  for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
    if (!lu_backend_available(be)) continue;
    const std::string name = stage(be).name();
    // one thread: STRUMPACK's multithreaded factorization is not bitwise reproducible from one object to the next
    // (the fluids callers use lu_threads = 1 for that reason), and x1 and x2 are compared bit for bit
    std::vector<double> x1, x2;
    {
      SparseLU lu(be);
      lu.set_threads(1);
      lu.factorize(s.D);
      CHECK(lu.attempts().empty() && lu.accepted_stage() == -1,
            "%s: after factorize, %zu attempts, accepted stage %d (expected none)", name.c_str(), lu.attempts().size(),
            lu.accepted_stage());
      x1 = lu.solve(s.bl);
      const double err = global_err(x1, s.xs, s.dist, comm);
      CHECK(err < 1e-10, "SparseLU(%s): solution error %g", to_string(be), err);
      CHECK(lu.chain().size() == 1 && lu.chain()[0].name() == name, "SparseLU(%s): chain [%s], expected [%s]",
            to_string(be), names(lu.chain()).c_str(), name.c_str());
      const auto& at = lu.attempts();
      CHECK(at.size() == 1 && at[0].stage == name && at[0].outcome == "accepted" &&
                at[0].reason == "no acceptance test" && lu.accepted_stage() == 0,
            "SparseLU(%s): attempts [%s] (%s), accepted stage %d", to_string(be), outcomes(lu).c_str(),
            at.empty() ? "" : at[0].reason.c_str(), lu.accepted_stage());
      if (at.size() == 1) {
        const double ber = lu_backward_error(s.D, x1, s.bl, {});
        CHECK(bitwise(at[0].backward_error, ber) && ber <= 1e-15,
              "SparseLU(%s): recorded backward error %a, lu_backward_error of the solution %a", to_string(be),
              at[0].backward_error, ber);
        CHECK(bitwise(at[0].measure, at[0].rel_residual) && at[0].bound == -1 &&
                  bitwise(at[0].rel_residual, lu.stats().rel_residual),
              "SparseLU(%s): Kind::None records measure %g (relative residual %g, stats %g), bound %g", to_string(be),
              at[0].measure, at[0].rel_residual, lu.stats().rel_residual, at[0].bound);
        CHECK(at[0].constraint_ratios.empty() && at[0].gate_residual == -1,
              "SparseLU(%s): %zu constraint ratios and gate residual %g without blocks or mask", to_string(be),
              at[0].constraint_ratios.size(), at[0].gate_residual);
        CHECK(at[0].factor_nnz == lu.stats().factor_nnz && at[0].index_bits == lu.stats().index_bits,
              "SparseLU(%s): the attempt's fill %ld / index bits %d differ from stats() (%ld / %d)", to_string(be),
              at[0].factor_nnz, at[0].index_bits, lu.stats().factor_nnz, lu.stats().index_bits);
      }
      const LUStats& st = lu.stats();
      CHECK(st.stage == name && st.attempts == 1 && st.backend == to_string(be) &&
                (at.empty() ||
                 (bitwise(st.backward_error, at[0].backward_error) && bitwise(st.measure, at[0].measure))),
            "SparseLU(%s): stats() stage '%s', attempts %d, backend %s", to_string(be), st.stage.c_str(), st.attempts,
            st.backend.c_str());
      check_records_same(lu, comm, name.c_str());
    }
    {
      SparseLU lu(std::vector<LUStage>{stage(be)});
      lu.set_threads(1);
      lu.factorize(s.D);
      x2 = lu.solve(s.bl);
      CHECK(lu.attempts().size() == 1 && lu.accepted_stage() == 0, "SparseLU({%s}): %zu attempts", name.c_str(),
            lu.attempts().size());
    }
    CHECK(on_all(bitwise(x1, x2), comm), "SparseLU(%s) and SparseLU({%s}) give different solutions", to_string(be),
          name.c_str());
  }
  // set_matching(false): stage 0 without matching (on the matrix without a zero block)
  if (lu_backend_available(LUBackend::StrumpackLU)) {
    const Sys s0 = make_sys(lc, 7, 0);
    SparseLU lu(LUBackend::StrumpackLU);
    lu.set_matching(false);
    lu.factorize(s0.D);
    lu.solve(s0.bl);
    CHECK(lu.chain().size() == 1 && lu.chain()[0].name() == "strumpack:none" && lu.attempts().size() == 1 &&
              lu.attempts()[0].stage == "strumpack:none",
          "set_matching(false): chain [%s], attempt '%s'", names(lu.chain()).c_str(),
          lu.attempts().empty() ? "" : lu.attempts()[0].stage.c_str());
  }
}

// ------------------------------------------------------------------------------------------------ 5
static void forced_outcomes(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const int me = s.me;
  struct F {
    LUForcedOutcome f;
    const char* outcome;
    bool iterate;  // the stage produced an iterate (its measure and backward error are recorded)
  };
  const F fs[] = {{LUForcedOutcome::Threw, "threw", false},
                  {LUForcedOutcome::MissedTarget, "missed_target", true},
                  {LUForcedOutcome::Refused, "refused", false}};
  for (const auto& f : fs) {
    set_lu_forced_outcome_for_testing(f.f, 1);
    SparseLU lu(eigens(2));
    lu.factorize(s.D);
    // a stage passed over by factorize() is listed before any solve
    const std::size_t before = lu.attempts().size(), expect_before = f.iterate ? 0 : 1;
    const auto x = lu.solve(s.bl);
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    const auto& at = lu.attempts();
    CHECK(before == expect_before, "forced %s: %zu attempts after factorize, expected %zu", f.outcome, before,
          expect_before);
    CHECK(at.size() == 2 && at[0].outcome == f.outcome && has(at[0].reason, "test hook") &&
              at[1].outcome == "accepted" && lu.accepted_stage() == 1,
          "forced %s on stage 0: attempts [%s], reason '%s', accepted stage %d (expected [%s,accepted], 1)",
          f.outcome, outcomes(lu).c_str(), at.empty() ? "" : at[0].reason.c_str(), lu.accepted_stage(), f.outcome);
    if (at.size() == 2) {
      CHECK(at[0].stage == "eigen" && at[1].stage == "eigen", "forced %s: stages '%s', '%s'", f.outcome,
            at[0].stage.c_str(), at[1].stage.c_str());
      if (f.iterate)
        CHECK(at[0].measure >= 0 && at[0].backward_error >= 0 && at[0].rel_residual >= 0,
              "forced %s: the rejected iterate's numbers are not recorded (measure %g, backward error %g)", f.outcome,
              at[0].measure, at[0].backward_error);
      else
        CHECK(at[0].measure == -1 && at[0].backward_error == -1 && at[0].rel_residual == -1,
              "forced %s: a stage without an iterate records measure %g, backward error %g", f.outcome, at[0].measure,
              at[0].backward_error);
      CHECK(at[1].backward_error >= 0 && at[1].backward_error <= 1e-15,
            "forced %s: the accepted stage's backward error %g", f.outcome, at[1].backward_error);
    }
    CHECK(lu.stats().attempts == 2 && lu.stats().stage == "eigen", "forced %s: stats() attempts %d, stage '%s'",
          f.outcome, lu.stats().attempts, lu.stats().stage.c_str());
    const double err = global_err(x, s.xs, s.dist, comm);
    CHECK(err < 1e-10, "forced %s: solution error %g", f.outcome, err);
    check_records_same(lu, comm, f.outcome);
  }
  // the hook's count: 3 forced rejections, then none (one solve per fresh SparseLU)
  {
    set_lu_forced_outcome_for_testing(LUForcedOutcome::MissedTarget, 3);
    std::string got;
    for (int k = 0; k < 5; ++k) {
      SparseLU lu(eigens(2));
      lu.factorize(s.D);
      lu.solve(s.bl);
      got += std::to_string(lu.attempts().size());
    }
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    CHECK(got == "22211", "count 3: attempts per solve %s, expected 22211", got.c_str());
  }
  // only stage 0 is forced: a solve that starts on another stage does not consume the count
  {
    set_lu_forced_outcome_for_testing(LUForcedOutcome::MissedTarget, 2);
    SparseLU lu(eigens(2));
    lu.factorize(s.D);
    lu.solve(s.bl);  // stage 0 forced, stage 1 accepted (count 1 left)
    lu.solve(s.bl);  // starts on stage 1 (sticky): nothing forced
    const std::size_t second = lu.attempts().size();
    SparseLU lu2(eigens(2));
    lu2.factorize(s.D);
    lu2.solve(s.bl);  // stage 0: the count's last one
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    CHECK(second == 1 && lu.accepted_stage() == 1 && lu2.attempts().size() == 2,
          "count 2: the sticky solve had %zu attempts (accepted stage %d), the next fresh solve %zu (expected 1, 1, 2)",
          second, lu.accepted_stage(), lu2.attempts().size());
  }
  // STRUMPACK builds: the stages' names, the backend that produced the solution, and the released STRUMPACK solver
  if (lu_backend_available(LUBackend::StrumpackLU)) {
    reset_lu_backend_object_peak_for_testing();
    set_lu_forced_outcome_for_testing(LUForcedOutcome::MissedTarget, 1);
    SparseLU lu(std::vector<LUStage>{stage(LUBackend::StrumpackLU), stage(LUBackend::EigenLU)});
    lu.factorize(s.D);
    const auto x = lu.solve(s.bl);
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    const auto& at = lu.attempts();
    CHECK(at.size() == 2 && at[0].stage == "strumpack:mc64-5" && at[0].outcome == "missed_target" &&
              at[1].stage == "eigen" && at[1].outcome == "accepted",
          "[strumpack, eigen], stage 0 forced: attempts [%s]", outcomes(lu).c_str());
    CHECK(lu.stats().backend == "eigen" && lu.stats().stage == "eigen",
          "[strumpack, eigen]: stats() backend '%s', stage '%s' (expected eigen: the stage that produced the solution)",
          lu.stats().backend.c_str(), lu.stats().stage.c_str());
    const auto oc = lu_backend_objects_for_testing();
    const int one = me == 0 ? 1 : 0;  // the serial backends' objects live on rank 0
    CHECK(oc.live == one && oc.peak == one && lu.live_eigen_index_bits_for_testing() == (me == 0 ? 32 : 0) &&
              lu.strumpack_matching_job_for_testing() == -1,
          "rank %d, [strumpack, eigen]: %d backend objects alive (peak %d), Eigen %d-bit, STRUMPACK job %d (expected "
          "%d, %d, the Eigen solver only)",
          me, oc.live, oc.peak, lu.live_eigen_index_bits_for_testing(), lu.strumpack_matching_job_for_testing(), one,
          one);
    CHECK(global_err(x, s.xs, s.dist, comm) < 1e-10, "[strumpack, eigen]: solution error %g",
          global_err(x, s.xs, s.dist, comm));
    check_records_same(lu, comm, "[strumpack, eigen]");
  }
  // a stage whose backend is not in this build is recorded and passed over
  LUBackend absent = LUBackend::EigenLU;
  for (LUBackend b : {LUBackend::StrumpackMPI, LUBackend::StrumpackLU})
    if (!lu_backend_available(b)) absent = b;
  if (absent != LUBackend::EigenLU) {
    set_lu_forced_outcome_for_testing(LUForcedOutcome::MissedTarget, 1);
    SparseLU lu(std::vector<LUStage>{stage(LUBackend::EigenLU), stage(absent), stage(LUBackend::EigenLU)});
    lu.factorize(s.D);
    lu.solve(s.bl);
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    const auto& at = lu.attempts();
    const std::string nm = stage(absent).name();
    CHECK(at.size() == 3 && at[1].stage == nm && at[1].outcome == "not_available" &&
              has(at[1].reason, "not in this build") && at[2].outcome == "accepted" && lu.accepted_stage() == 2,
          "[eigen, %s, eigen], stage 0 forced: attempts [%s] (expected missed_target,not_available,accepted)",
          nm.c_str(), outcomes(lu).c_str());
    check_records_same(lu, comm, "not available");
  } else if (me == 0) {
    std::printf("skip the not_available case: every backend is in this build\n");
  }
}

// ------------------------------------------------------------------------------------------------ 6
// A stage is rejected on the caller's measure, not on its backward error. Block 1 (the 25 rows n0..n-1 of oseen_like)
// is multiplied by 2^27 and b is zero on it: an accurate solve leaves block 1's residual at the round-off of its
// terms, about 2^27 u times them, which dominates the relative residual over all rows (||b|| is block 0's), while the
// blockwise backward error stays at round-off. The test measures both numbers on the system and asserts that they
// straddle the bound (their geometric mean) before asserting the rejection.
static void straddle(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const int me = rank_of(comm);
  const SpMat S = oseen_like(30, 3.0, 25);
  const int n = static_cast<int>(S.rows()), n0 = 900;
  std::vector<int> dist;
  DistCsr DA = distributed(S, lc, dist);
  for (int i = 0; i < DA.local_rows; ++i)
    if (DA.row_begin + i >= n0)
      for (int k = DA.rowptr[i]; k < DA.rowptr[i + 1]; ++k) DA.val[k] = std::ldexp(DA.val[k], 27);
  std::mt19937 rng(29);
  std::uniform_real_distribution<double> U(-1, 1);
  Eigen::VectorXd bg = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n0; ++i) bg(i) = U(rng);
  std::vector<int> blocks_g(n);
  for (int i = 0; i < n; ++i) blocks_g[i] = i < n0 ? 0 : 1;
  const auto bl = local_of(bg, dist, me);
  const auto blocks = local_ints(blocks_g, dist, me);
  // the two numbers of an accurate solve of this system
  double rr = 0, be = 0;
  std::vector<double> x_ref;
  {
    SparseLU lu(LUBackend::EigenLU);
    lu.set_throw_on_miss(false);
    lu.factorize(DA);
    x_ref = lu.solve(bl);
    rr = lu.stats().rel_residual;
    be = lu_backward_error(DA, x_ref, bl, blocks);
  }
  const double bound = std::sqrt(rr * be);
  if (me == 0) std::printf("straddle: backward error %.3e < bound %.3e < relative residual %.3e\n", be, bound, rr);
  CHECK(be > 0 && rr >= 1e4 * be, "the straddle construction failed: backward error %g, relative residual %g", be, rr);
  {
    SparseLU lu(eigens(2));
    lu.set_throw_on_miss(false);
    lu.set_row_blocks(blocks);
    lu.set_acceptance(rel_residual_acceptance(bound));
    lu.factorize(DA);
    const auto x = lu.solve(bl);
    const auto& at = lu.attempts();
    CHECK(at.size() == 2 && at[0].outcome == "missed_target" && at[1].outcome == "missed_target" &&
              lu.accepted_stage() == -1,
          "RelResidual %g: attempts [%s], accepted stage %d (expected both rejected: the relative residual misses the "
          "bound although the backward error meets it)",
          bound, outcomes(lu).c_str(), lu.accepted_stage());
    for (std::size_t k = 0; k < at.size(); ++k) {
      CHECK(at[k].rel_residual > bound && at[k].backward_error < bound,
            "attempt %zu: relative residual %g and backward error %g do not straddle %g", k, at[k].rel_residual,
            at[k].backward_error, bound);
      CHECK(bitwise(at[k].measure, at[k].rel_residual) && bitwise(at[k].bound, bound),
            "attempt %zu: measure %g (relative residual %g), bound %g (expected %g)", k, at[k].measure,
            at[k].rel_residual, at[k].bound, bound);
      CHECK(starts(at[k].reason, "relative residual ") && has(at[k].reason, " > "),
            "attempt %zu: reason '%s' (expected 'relative residual <value> > <bound>')", k, at[k].reason.c_str());
    }
    // no stage accepted: the iterate with the smallest relative residual; both stages are the same arithmetic
    CHECK(on_all(bitwise(x, x_ref), comm), "rank %d: the returned iterate is not the stages' (identical) one", me);
    check_records_same(lu, comm, "straddle");
  }
  // the control: a bound above the relative residual accepts stage 0
  {
    SparseLU lu(eigens(2));
    lu.set_throw_on_miss(false);
    lu.set_row_blocks(blocks);
    lu.set_acceptance(rel_residual_acceptance(10 * rr));
    lu.factorize(DA);
    lu.solve(bl);
    const auto& at = lu.attempts();
    CHECK(at.size() == 1 && at[0].outcome == "accepted" && lu.accepted_stage() == 0 &&
              starts(at[0].reason, "relative residual ") && has(at[0].reason, " <= "),
          "RelResidual %g (above the relative residual %g): attempts [%s], reason '%s'", 10 * rr, rr,
          outcomes(lu).c_str(), at.empty() ? "" : at[0].reason.c_str());
  }
}

// ------------------------------------------------------------------------------------------------ 7
static void verdict(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const int me = s.me;
  // rank 0's values reject stage 0 and accept stage 1; the other ranks' values say the opposite, and must not count
  {
    ScriptedVerdict v;
    v.comm = lc;
    v.values = me == 0 ? std::vector<double>{5.0, 0.25} : std::vector<double>{0.25, 5.0};
    SparseLU lu(eigens(3));
    lu.set_acceptance(v.acceptance(1.0, "projection residual"));
    lu.factorize(s.D);
    const auto x = lu.solve(s.bl);
    const auto& at = lu.attempts();
    CHECK(v.calls == 2, "rank %d: the verdict was called %d times (expected once per iterate: 2)", me, v.calls);
    CHECK(at.size() == 2 && at[0].outcome == "missed_target" && at[1].outcome == "accepted" &&
              lu.accepted_stage() == 1,
          "rank %d, Verdict: attempts [%s], accepted stage %d (expected [missed_target,accepted], 1)", me,
          outcomes(lu).c_str(), lu.accepted_stage());
    if (at.size() == 2) {
      CHECK(at[0].measure == 5.0 && at[1].measure == 0.25 && at[0].bound == 1.0 && at[1].bound == 1.0,
            "rank %d, Verdict: recorded measures %g, %g and bounds %g, %g (expected rank 0's 5 and 0.25, bound 1)", me,
            at[0].measure, at[1].measure, at[0].bound, at[1].bound);
      CHECK(starts(at[0].reason, "projection residual ") && has(at[0].reason, " > ") &&
                starts(at[1].reason, "projection residual ") && has(at[1].reason, " <= "),
            "Verdict reasons '%s', '%s'", at[0].reason.c_str(), at[1].reason.c_str());
      CHECK(at[0].backward_error >= 0 && at[1].backward_error >= 0,
            "Verdict: the backward error is not recorded on every attempt (%g, %g)", at[0].backward_error,
            at[1].backward_error);
    }
    CHECK(v.seen.size() == 2 && v.seen[0].size() == static_cast<std::size_t>(s.D.local_rows) &&
              bitwise(x, v.seen.back()),
          "rank %d: the returned solution is not the iterate the verdict accepted", me);
    CHECK(lu.stats().measure == 0.25, "rank %d: stats().measure %g, expected 0.25", me, lu.stats().measure);
    check_records_same(lu, comm, "Verdict");
  }
  // the bound is met with equality: accepted
  {
    ScriptedVerdict v;
    v.comm = lc;
    v.values = {1.0};
    SparseLU lu(eigens(2));
    lu.set_acceptance(v.acceptance(1.0));
    lu.factorize(s.D);
    lu.solve(s.bl);
    CHECK(lu.attempts().size() == 1 && lu.accepted_stage() == 0 && v.calls == 1 &&
              starts(lu.attempts()[0].reason, "verdict "),
          "a verdict equal to the bound: attempts [%s], %d calls, reason '%s' (expected accepted, named 'verdict')",
          outcomes(lu).c_str(), v.calls, lu.attempts().empty() ? "" : lu.attempts()[0].reason.c_str());
  }
  // a non-finite verdict is a miss
  {
    ScriptedVerdict v;
    v.comm = lc;
    v.values = {std::numeric_limits<double>::quiet_NaN(), 0.5};
    SparseLU lu(eigens(2));
    lu.set_acceptance(v.acceptance(1.0));
    lu.factorize(s.D);
    lu.solve(s.bl);
    CHECK(outcomes(lu) == "missed_target,accepted" && lu.accepted_stage() == 1,
          "a NaN verdict: attempts [%s] (expected missed_target,accepted)", outcomes(lu).c_str());
  }
}

// ------------------------------------------------------------------------------------------------ 8
static void sticky(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const int me = s.me;
  const int one = me == 0 ? 1 : 0;  // Eigen's solver object lives on rank 0
  std::mt19937 rng(31);
  std::uniform_real_distribution<double> U(-1, 1);
  auto rhs = [&](Eigen::VectorXd& xs) {
    xs.resize(s.S.rows());
    for (int i = 0; i < s.S.rows(); ++i) xs(i) = U(rng);
    const Eigen::VectorXd b = s.S * xs;
    return local_of(b, s.dist, me);
  };
  // (a) the hook rejects stage 0 once; the next nine solves of the same factorization stay on stage 1
  {
    set_lu_forced_outcome_for_testing(LUForcedOutcome::MissedTarget, 1);
    SparseLU lu(eigens(2));
    lu.factorize(s.D);
    std::string per;
    double worst = 0;
    for (int k = 0; k < 10; ++k) {
      Eigen::VectorXd xs;
      const auto x = lu.solve(rhs(xs));
      per += std::to_string(lu.attempts().size()) + ":" + std::to_string(lu.accepted_stage()) + " ";
      worst = std::max(worst, global_err(x, xs, s.dist, comm));
    }
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    CHECK(per == "2:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1 ",
          "sticky over ten solves: attempts:accepted stage per solve %s (expected 2:1, then 1:1)", per.c_str());
    CHECK(worst < 1e-10, "sticky: worst solution error %g", worst);
  }
  // (b) a later miss walks on from the sticky stage; factorize() starts again at stage 0; one backend object
  {
    ScriptedVerdict v;
    v.comm = lc;
    // calls: solve 1 (stage 0 rejected, stage 1 accepted), solves 2-10 (stage 1), solve 11 (stage 1 rejected, stage 2
    // accepted), solve 12 (stage 2), then after a new factorize, solve 13 (stage 0)
    v.values = {2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5};
    reset_lu_backend_object_peak_for_testing();
    SparseLU lu(eigens(3));
    lu.set_acceptance(v.acceptance(1.0));
    lu.factorize(s.D);
    std::string per;
    double worst = 0;
    for (int k = 0; k < 12; ++k) {
      Eigen::VectorXd xs;
      const auto x = lu.solve(rhs(xs));
      per += std::to_string(lu.attempts().size()) + ":" + std::to_string(lu.accepted_stage()) + " ";
      worst = std::max(worst, global_err(x, xs, s.dist, comm));
      if (k == 10) {
        CHECK(outcomes(lu) == "missed_target,accepted", "the sticky stage's miss: attempts [%s]", outcomes(lu).c_str());
        check_records_same(lu, comm, "walk on");
      }
    }
    const auto oc = lu_backend_objects_for_testing();
    CHECK(oc.live == one && oc.peak == one,
          "rank %d: %d backend objects alive, at most %d at once over two escalations (expected %d and %d: the "
          "previous stage's object is released before the next stage factorizes)",
          me, oc.live, oc.peak, one, one);
    lu.factorize(s.D);
    Eigen::VectorXd xs;
    const auto x = lu.solve(rhs(xs));
    per += std::to_string(lu.attempts().size()) + ":" + std::to_string(lu.accepted_stage());
    worst = std::max(worst, global_err(x, xs, s.dist, comm));
    CHECK(per == "2:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1 2:2 1:2 1:0",
          "walk on: attempts:accepted stage per solve %s (expected 2:1, nine 1:1, 2:2, 1:2, then 1:0 after factorize)",
          per.c_str());
    CHECK(v.calls == 15, "rank %d: %d verdict calls, expected 15 (one per iterate)", me, v.calls);
    CHECK(worst < 1e-10, "walk on: worst solution error %g", worst);
    const auto oc2 = lu_backend_objects_for_testing();
    CHECK(oc2.live == one && oc2.peak == one, "rank %d: after the new factorize, %d alive (peak %d), expected %d", me,
          oc2.live, oc2.peak, one);
  }
  const auto oc = lu_backend_objects_for_testing();
  CHECK(oc.live == 0, "rank %d: %d backend objects alive after every SparseLU was destroyed", me, oc.live);
}

// ------------------------------------------------------------------------------------------------ 9
// The perturbation hook multiplies stage 0's iterate by (1 + eps): with b = A xs, its residual is
// (1 + eps) r - eps b, i.e. -eps b up to the solve's round-off, so its relative residual is eps to a relative 1e-8,
// and the test evaluates every recorded number again in long double from the iterate the verdict was given.
static void best_iterate(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const int me = s.me;
  const int n = static_cast<int>(s.S.rows()), n0 = 900;
  const double eps = 1e-7;
  std::vector<int> blocks_g(n);  // rows n0.. are constraint rows: 25 ratios recorded per attempt
  std::vector<char> mask_g(n);
  for (int i = 0; i < n; ++i) {
    blocks_g[i] = i < n0 ? 0 : -2;
    mask_g[i] = i < n0;
  }
  const auto blocks = local_ints(blocks_g, s.dist, me);
  const std::vector<char> mask(mask_g.begin() + s.dist[me], mask_g.begin() + s.dist[me + 1]);
  auto chain2 = [&](ScriptedVerdict& v, double bound) {
    auto lu = std::make_unique<SparseLU>(eigens(2));
    lu->set_throw_on_miss(false);  // the returned iterate may be the perturbed one
    lu->set_row_blocks(blocks);
    lu->set_residual_rows(mask);
    lu->set_acceptance(v.acceptance(bound));
    return lu;
  };
  // (a) the hook's numbers; stage 0 rejected, stage 1 accepted
  {
    ScriptedVerdict v;
    v.comm = lc;
    v.values = {2.0, 0.5};
    set_lu_solution_perturbation_for_testing(eps);
    auto lu = chain2(v, 1.0);
    lu->factorize(s.D);
    const auto x = lu->solve(s.bl);
    set_lu_solution_perturbation_for_testing(0.0);
    const auto& at = lu->attempts();
    CHECK(at.size() == 2 && v.seen.size() == 2, "perturbation: %zu attempts, %zu iterates", at.size(), v.seen.size());
    if (at.size() == 2 && v.seen.size() == 2) {
      const Ref r0 = reference(s.D, v.seen[0], s.bl, blocks, mask, s.dist, comm);
      const Ref r1 = reference(s.D, v.seen[1], s.bl, blocks, mask, s.dist, comm);
      bool scaled = true;  // stage 0's iterate is stage 1's times (1 + eps): the two stages are the same arithmetic
      for (std::size_t i = 0; i < v.seen[0].size(); ++i)
        scaled = scaled && std::fabs(v.seen[0][i] - (1 + eps) * v.seen[1][i]) <= 4e-16 * std::fabs(v.seen[1][i]);
      CHECK(on_all(scaled, comm), "rank %d: stage 0's iterate is not stage 1's times (1 + %g)", me, eps);
      if (me == 0)
        std::printf("perturbed stage: relative residual %.6e (eps %.1e), gate %.6e, backward error %.6e (ref %.6e)\n",
                    at[0].rel_residual, eps, at[0].gate_residual, at[0].backward_error, r0.backward_error);
      CHECK(reldiff(at[0].rel_residual, eps) <= 1e-6 && reldiff(at[0].rel_residual, r0.rel_residual) <= 1e-6,
            "perturbed: relative residual %.9g, expected eps = %g (long double: %.9g)", at[0].rel_residual, eps,
            r0.rel_residual);
      CHECK(reldiff(at[0].gate_residual, r0.gate_residual) <= 1e-6,
            "perturbed: gate residual %.9g, long double %.9g", at[0].gate_residual, r0.gate_residual);
      CHECK(reldiff(at[0].backward_error, r0.backward_error) <= 1e-6,
            "perturbed: backward error %.9g, long double %.9g", at[0].backward_error, r0.backward_error);
      bool ratios_ok =
          at[0].constraint_ratios.size() == r0.constraint_ratios.size() && r0.constraint_ratios.size() == 25;
      for (std::size_t j = 0; ratios_ok && j < r0.constraint_ratios.size(); ++j)
        ratios_ok = std::fabs(at[0].constraint_ratios[j] - r0.constraint_ratios[j]) <=
                    1e-6 * r0.constraint_ratios[j] + 4e-16;
      CHECK(ratios_ok, "perturbed: the %zu constraint ratios differ from the long double ones (%zu)",
            at[0].constraint_ratios.size(), r0.constraint_ratios.size());
      CHECK(at[0].rel_residual_before <= 1e-12, "perturbed: rel_residual_before %g is not the plain solve's",
            at[0].rel_residual_before);
      // the unperturbed stage: its numbers are those of its own iterate
      CHECK(at[1].rel_residual <= 1e-12 && at[1].backward_error <= 1e-15 && at[1].constraint_ratios.size() == 25 &&
                at[1].gate_residual >= 0 && at[1].gate_residual <= 1e-12,
            "unperturbed stage: relative residual %g, backward error %g, %zu ratios, gate %g", at[1].rel_residual,
            at[1].backward_error, at[1].constraint_ratios.size(), at[1].gate_residual);
      CHECK(bitwise(at[1].backward_error, lu_backward_error(s.D, v.seen[1], s.bl, blocks)),
            "the accepted stage's recorded backward error is not lu_backward_error of its iterate");
      (void)r1;
    }
    CHECK(global_err(x, s.xs, s.dist, comm) < 1e-10, "perturbation: solution error %g",
          global_err(x, s.xs, s.dist, comm));
    check_records_same(*lu, comm, "perturbation");
  }
  // (b) no stage accepted: the best by the acceptance measure, although stage 0 (perturbed) has the larger relative
  // residual and backward error
  {
    ScriptedVerdict v;
    v.comm = lc;
    v.values = {1.0, 3.0};
    set_lu_solution_perturbation_for_testing(eps);
    auto lu = chain2(v, 0.5);
    lu->factorize(s.D);
    const auto x = lu->solve(s.bl);
    set_lu_solution_perturbation_for_testing(0.0);
    const auto& at = lu->attempts();
    CHECK(outcomes(*lu) == "missed_target,missed_target" && lu->accepted_stage() == -1,
          "none accepted: attempts [%s], accepted stage %d", outcomes(*lu).c_str(), lu->accepted_stage());
    CHECK(v.seen.size() == 2 && on_all(bitwise(x, v.seen[0]), comm),
          "rank %d: the returned iterate is not stage 0's, the best by the verdict (1 against 3)", me);
    if (at.size() == 2)
      CHECK(lu->stats().measure == 1.0 && bitwise(lu->stats().rel_residual, at[0].rel_residual) &&
                bitwise(lu->stats().backward_error, at[0].backward_error) && lu->stats().attempts == 2,
            "none accepted: stats() describes measure %g, relative residual %g (stage 0's: 1, %g)", lu->stats().measure,
            lu->stats().rel_residual, at[0].rel_residual);
    check_records_same(*lu, comm, "best iterate");
  }
  // (c) a tie in the measure: the smaller backward error (stage 1, unperturbed)
  {
    ScriptedVerdict v;
    v.comm = lc;
    v.values = {2.0, 2.0};
    set_lu_solution_perturbation_for_testing(eps);
    auto lu = chain2(v, 0.5);
    lu->factorize(s.D);
    const auto x = lu->solve(s.bl);
    set_lu_solution_perturbation_for_testing(0.0);
    const auto& at = lu->attempts();
    CHECK(v.seen.size() == 2 && on_all(bitwise(x, v.seen[1]), comm),
          "rank %d: a tie in the measure did not return the iterate with the smaller backward error (stage 1)", me);
    if (at.size() == 2)
      CHECK(at[1].backward_error < at[0].backward_error && bitwise(lu->stats().backward_error, at[1].backward_error),
            "tie: backward errors %g (stage 0), %g (stage 1), stats() %g", at[0].backward_error, at[1].backward_error,
            lu->stats().backward_error);
  }
  // (d) RelResidual, no stage accepted: the smaller relative residual (stage 1; stage 0 perturbed)
  {
    set_lu_solution_perturbation_for_testing(eps);
    SparseLU lu(eigens(2));
    lu.set_throw_on_miss(false);
    lu.set_acceptance(rel_residual_acceptance(1e-30));
    lu.factorize(s.D);
    const auto x = lu.solve(s.bl);
    set_lu_solution_perturbation_for_testing(0.0);
    const auto& at = lu.attempts();
    CHECK(outcomes(lu) == "missed_target,missed_target" && at.size() == 2 &&
              bitwise(lu.stats().measure, at[1].measure) && bitwise(lu.stats().rel_residual, at[1].rel_residual),
          "RelResidual, none accepted: attempts [%s], stats() measure %g (expected stage 1's %g)",
          outcomes(lu).c_str(), lu.stats().measure, at.size() == 2 ? at[1].measure : -1.0);
    CHECK(global_err(x, s.xs, s.dist, comm) < 1e-10, "RelResidual, none accepted: solution error %g (stage 0's?)",
          global_err(x, s.xs, s.dist, comm));
  }
}

// ------------------------------------------------------------------------------------------------ 10
// The rank-1 replacement of the acceptance inputs: a correct chain takes rank 0's decision on every rank. A chain
// that decided per rank would send rank 1 to another stage (a collective on the LU's communicator that rank 0 never
// enters): the test hangs, and the TIMEOUT ends it.
static void rank0_decides(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  set_lu_rank1_acceptance_flip_for_testing(true);
  {  // rank 0 accepts stage 0
    SparseLU lu(eigens(2));
    lu.set_acceptance(rel_residual_acceptance(1e-9));
    lu.factorize(s.D);
    lu.solve(s.bl);
    CHECK(outcomes(lu) == "accepted", "flip, rank 0 accepts: attempts [%s]", outcomes(lu).c_str());
    check_records_same(lu, comm, "flip, accept");
  }
  {  // rank 0 rejects stage 0 (perturbed by 1e-7 against a bound of 1e-9)
    set_lu_solution_perturbation_for_testing(1e-7);
    SparseLU lu(eigens(2));
    lu.set_acceptance(rel_residual_acceptance(1e-9));
    lu.factorize(s.D);
    lu.solve(s.bl);
    set_lu_solution_perturbation_for_testing(0.0);
    CHECK(outcomes(lu) == "missed_target,accepted", "flip, rank 0 rejects: attempts [%s]", outcomes(lu).c_str());
    check_records_same(lu, comm, "flip, reject");
  }
  {  // a Verdict
    ScriptedVerdict v;
    v.comm = lc;
    v.values = {2.0, 0.5};
    SparseLU lu(eigens(2));
    lu.set_acceptance(v.acceptance(1.0));
    lu.factorize(s.D);
    lu.solve(s.bl);
    CHECK(outcomes(lu) == "missed_target,accepted" && v.calls == 2, "flip, Verdict: attempts [%s], %d calls",
          outcomes(lu).c_str(), v.calls);
    check_records_same(lu, comm, "flip, Verdict");
  }
  set_lu_rank1_acceptance_flip_for_testing(false);
}

// ------------------------------------------------------------------------------------------------ 11
// A staggered-grid (MAC) Stokes matrix on g x g cells of a square of side `s` (h = s / g), with the scalings of the
// finite-element projection: momentum h^2 (mass-like, a 5-point pattern), divergence and gradient h, mean-value
// weights h^2 (cell areas), Dirichlet rows the identity. Unknowns: u on the (g + 1) g vertical faces, v on the
// g (g + 1) horizontal faces (boundary faces: Dirichlet identity rows, block -1), the g^2 cell pressures, and for
// Bordered the multiplier. The gradient of a face between cells L and R is h (0.1 + 0.2) p_R - h 0.3 p_L, so the
// constant pressure is in the kernel of the momentum rows to round-off (0.1 + 0.2 - 0.3 = 5.6e-17), as in a finite-
// element matrix; the divergence rows are its transpose. Constant: the constant pressure is left in; Pinned: the
// pressure of cell (0, 0) has an identity row (block -2); Bordered: a multiplier row w^T p and column w (block -2).
enum class Nullspace { Constant, Pinned, Bordered };
struct Stokes {
  SpMat A;
  std::vector<int> blocks;
  Eigen::VectorXd z;  // the constant pressure
};
static Stokes stokes_mac(int g, double side, Nullspace ns) {
  const double h = side / g, kap = h * h, cR = h * (0.1 + 0.2), cL = h * 0.3, w = h * h;
  const int nuf = (g + 1) * g, nvf = g * (g + 1), nvel = nuf + nvf, np = g * g;
  const int n = nvel + np + (ns == Nullspace::Bordered ? 1 : 0);
  auto U = [&](int i, int j) { return i * g + j; };              // i in [0, g], j in [0, g)
  auto V = [&](int i, int j) { return nuf + i * (g + 1) + j; };  // i in [0, g), j in [0, g]
  auto Pc = [&](int i, int j) { return nvel + i * g + j; };      // cell (i, j)
  auto u_in = [&](int i, int j) { return i > 0 && i < g && j >= 0 && j < g; };
  auto v_in = [&](int i, int j) { return j > 0 && j < g && i >= 0 && i < g; };
  std::vector<T> t;
  Stokes out;
  out.blocks.assign(n, 1);  // continuity rows
  const int p0 = Pc(0, 0);
  auto add = [&](int r, int c, double v) {
    if (ns == Nullspace::Pinned && r == p0) return;  // the pinned row is replaced below
    t.emplace_back(r, c, v);
  };
  const int d[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  for (int i = 0; i <= g; ++i)
    for (int j = 0; j < g; ++j) {
      const int k = U(i, j);
      if (!u_in(i, j)) {
        add(k, k, 1.0);
        out.blocks[k] = -1;
        continue;
      }
      out.blocks[k] = 0;
      add(k, k, 4 * kap);
      for (const auto& e : d)
        if (u_in(i + e[0], j + e[1])) add(k, U(i + e[0], j + e[1]), -kap);
      add(k, Pc(i, j), cR);
      add(k, Pc(i - 1, j), -cL);
      add(Pc(i, j), k, cR);
      add(Pc(i - 1, j), k, -cL);
    }
  for (int i = 0; i < g; ++i)
    for (int j = 0; j <= g; ++j) {
      const int k = V(i, j);
      if (!v_in(i, j)) {
        add(k, k, 1.0);
        out.blocks[k] = -1;
        continue;
      }
      out.blocks[k] = 0;
      add(k, k, 4 * kap);
      for (const auto& e : d)
        if (v_in(i + e[0], j + e[1])) add(k, V(i + e[0], j + e[1]), -kap);
      add(k, Pc(i, j), cR);
      add(k, Pc(i, j - 1), -cL);
      add(Pc(i, j), k, cR);
      add(Pc(i, j - 1), k, -cL);
    }
  if (ns == Nullspace::Bordered) {
    const int lam = n - 1;
    for (int q = 0; q < np; ++q) {
      t.emplace_back(nvel + q, lam, w);
      t.emplace_back(lam, nvel + q, w);
    }
    out.blocks[lam] = -2;
  }
  if (ns == Nullspace::Pinned) {
    t.emplace_back(p0, p0, 1.0);
    out.blocks[p0] = -2;
  }
  out.A.resize(n, n);
  out.A.setFromTriplets(t.begin(), t.end());
  out.A.makeCompressed();
  out.z = Eigen::VectorXd::Zero(n);
  for (int q = 0; q < np; ++q) out.z(nvel + q) = 1.0;
  return out;
}

// ||A z|| / (||A||_F ||z||): today's probe, evaluated by the test itself (long double)
static double unscaled_probe(const SpMat& A, const Eigen::VectorXd& z) {
  long double az = 0, fa = 0, zz = 0;
  Eigen::VectorXd y = A * z;
  for (int i = 0; i < y.size(); ++i) az += static_cast<long double>(y(i)) * y(i);
  for (int k = 0; k < A.outerSize(); ++k)
    for (SpMat::InnerIterator it(A, k); it; ++it) fa += static_cast<long double>(it.value()) * it.value();
  for (int i = 0; i < z.size(); ++i) zz += static_cast<long double>(z(i)) * z(i);
  return static_cast<double>(std::sqrt(az) / (std::sqrt(fa) * std::sqrt(zz)));
}

static void kernel_probe(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const int me = rank_of(comm);
  const int g = 16;
  const double tol = 1e-14;  // the Navier-Stokes solver's tolerance
  // the guard of this section's teeth: today's probe refuses the bordered matrix at side 1e-6 (the row w^T 1 = s^2
  // against ||A||_F ~ sqrt(64) of the Dirichlet rows and ||z|| = 16: about 8e-15)
  {
    const Stokes B = stokes_mac(g, 1e-6, Nullspace::Bordered);
    const double old = unscaled_probe(B.A, B.z);
    if (me == 0) std::printf("kernel probe: today's ratio on the bordered matrix at side 1e-6: %.3e\n", old);
    CHECK(old <= tol, "the bordered matrix at side 1e-6 is not refused by today's probe (%g): no teeth", old);
  }
  const char* nsname[] = {"constant left in", "pinned", "bordered"};
  for (double side : {1e-6, 1e-3, 1.0, 1e3, 1e6})
    for (Nullspace ns : {Nullspace::Constant, Nullspace::Pinned, Nullspace::Bordered}) {
      const Stokes M = stokes_mac(g, side, ns);
      std::vector<int> dist;
      const DistCsr D = distributed(M.A, lc, dist);
      const char* nm = nsname[static_cast<int>(ns)];
      std::string what;
      bool refused = false;
      double ratio = -1;
      try {
        SparseLU lu(LUBackend::EigenLU);
        lu.set_row_blocks(local_ints(M.blocks, dist, me));
        lu.set_kernel_probe(local_of(M.z, dist, me), tol);
        try {
          lu.factorize(D);
        } catch (const axmp::Error& e) {
          refused = true;
          what = e.what();
        }
        ratio = lu.stats().kernel_probe;
      } catch (const std::exception& e) {
        what = e.what();
        CHECK(false, "rank %d, %s at side %g: %s", me, nm, side, e.what());
      }
      const bool expect_refused = ns == Nullspace::Constant;
      CHECK(refused == expect_refused && (!refused || has(what, "kernel probe")),
            "rank %d, %s at side %g: %s (blockwise ratio %g, tol %g) %s", me, nm, side,
            refused ? "refused" : "accepted", ratio, tol, what.c_str());
      if (expect_refused)
        CHECK(ratio >= 0 && ratio <= tol, "rank %d, %s at side %g: stats().kernel_probe %g", me, nm, side, ratio);
      else
        CHECK(ratio >= 0.5 && ratio <= 1.0 + 1e-15,
              "rank %d, %s at side %g: stats().kernel_probe %g (expected of order one: the constraint row's own ratio "
              "|w^T 1| / (|w|^T 1) or 1)",
              me, nm, side, ratio);
      CHECK(on_all(refused == expect_refused, comm), "%s at side %g: not %s on every rank", nm, side,
            expect_refused ? "refused" : "accepted");
    }
  // without row blocks: today's probe (a caller that declares none is unchanged)
  {
    const Stokes M = stokes_mac(g, 1e-6, Nullspace::Bordered);
    std::vector<int> dist;
    const DistCsr D = distributed(M.A, lc, dist);
    bool refused = false;
    SparseLU lu(LUBackend::EigenLU);
    lu.set_kernel_probe(local_of(M.z, dist, me), tol);
    try {
      lu.factorize(D);
    } catch (const axmp::Error& e) {
      refused = has(e.what(), "kernel probe");
    }
    CHECK(on_all(refused, comm), "rank %d: without row blocks the bordered matrix at side 1e-6 was accepted (ratio %g)",
          me, lu.stats().kernel_probe);
  }
  // the small Stokes matrix as a manufactured solution (bordered, unit 1), through a two-stage chain with blocks
  {
    HookReset r2;
    const Stokes M = stokes_mac(g, 1.0, Nullspace::Bordered);
    std::vector<int> dist;
    const DistCsr D = distributed(M.A, lc, dist);
    std::mt19937 rng(37);
    std::uniform_real_distribution<double> U(-1, 1);
    Eigen::VectorXd xs(M.A.rows());
    for (int i = 0; i < xs.size(); ++i) xs(i) = U(rng);
    const Eigen::VectorXd b = M.A * xs;
    set_lu_forced_outcome_for_testing(LUForcedOutcome::MissedTarget, 1);
    SparseLU lu(eigens(2));
    lu.set_row_blocks(local_ints(M.blocks, dist, me));
    lu.set_kernel_probe(local_of(M.z, dist, me), tol);
    lu.factorize(D);
    const auto x = lu.solve(local_of(b, dist, me));
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    const double err = global_err(x, xs, dist, comm);
    CHECK(err < 1e-10 && outcomes(lu) == "missed_target,accepted", "small Stokes: error %g, attempts [%s]", err,
          outcomes(lu).c_str());
    // the multiplier row is the one constraint row: one ratio per attempt
    for (const auto& a : lu.attempts())
      CHECK(a.constraint_ratios.size() == 1, "small Stokes: %zu constraint ratios recorded, expected 1",
            a.constraint_ratios.size());
    check_records_same(lu, comm, "small Stokes");
  }
}

// ------------------------------------------------------------------------------------------------ 12
static void strumpack_matching(MPI_Comm comm, MPI_Comm lc) {
  const int me = rank_of(comm);
  const Sys s = make_sys(lc, 7, 0);  // no zero block: the matching-free stage factorizes it
  using M = MatchingJob;
  struct J {
    M m;
    int job;
  };
  const J jobs[] = {{M::None, 0}, {M::MaxSmallestDiagonal, 2}, {M::MaxDiagonalProductScaling, 5}};
  for (LUBackend be : {LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
    if (!lu_backend_available(be)) {
      if (me == 0) std::printf("skip %s (not in this build)\n", to_string(be));
      continue;
    }
    const bool everywhere = be == LUBackend::StrumpackMPI;  // the distributed solver has an object on every rank
    auto expect = [&](int job) { return everywhere || me == 0 ? job : -1; };
    for (const auto& j : jobs) {
      SparseLU lu(std::vector<LUStage>{stage(be, j.m)});
      lu.factorize(s.D);
      const int got = lu.strumpack_matching_job_for_testing();
      CHECK(got == expect(j.job), "rank %d, %s: the solver object's MC64 job is %d, expected %d", me,
            stage(be, j.m).name().c_str(), got, expect(j.job));
      const auto x = lu.solve(s.bl);
      CHECK(global_err(x, s.xs, s.dist, comm) < 1e-10, "%s: solution error %g", stage(be, j.m).name().c_str(),
            global_err(x, s.xs, s.dist, comm));
    }
    for (bool on : {true, false}) {  // the shim
      SparseLU lu(be);
      lu.set_matching(on);
      lu.factorize(s.D);
      const int got = lu.strumpack_matching_job_for_testing();
      CHECK(got == expect(on ? 5 : 0), "rank %d, SparseLU(%s) with set_matching(%d): job %d", me, to_string(be), on,
            got);
    }
  }
}

// ------------------------------------------------------------------------------------------------ 13
static void every_stage_threw(MPI_Comm comm, MPI_Comm lc) {
  const int me = rank_of(comm);
  SpMat Z = oseen_like(12, 3.0, 10);
  Z.prune([](Eigen::Index, Eigen::Index c, double) { return c != 5; });  // column 5 empty: structurally singular
  std::vector<int> dist;
  const DistCsr D = distributed(Z, lc, dist);
  std::vector<double> bl(D.local_rows, 1.0);
  auto factorize_error = [&](SparseLU& lu) {
    try {
      lu.factorize(D);
    } catch (const axmp::Error& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  // one stage: today's error
  std::string today, one;
  {
    SparseLU lu(LUBackend::EigenLU);
    today = factorize_error(lu);
  }
  {
    SparseLU lu(eigens(1));
    one = factorize_error(lu);
  }
  CHECK(!today.empty() && one == today, "rank %d: a one-stage chain's error differs from today's:\n%s\n%s", me,
        one.c_str(), today.c_str());
  // two stages: both recorded, one error that names both
  SparseLU lu(eigens(2));
  const std::string what = factorize_error(lu);
  const auto& at = lu.attempts();
  CHECK(!what.empty() && has(what, "every stage of the sparse LU chain failed"),
        "rank %d: a chain whose stages all fail: %s", me, what.empty() ? "factorize did not throw" : what.c_str());
  CHECK(at.size() == 2 && at[0].outcome == "threw" && at[1].outcome == "threw", "rank %d: attempts [%s]", me,
        outcomes(lu).c_str());
  for (const auto& a : at)
    CHECK(!a.reason.empty() && has(what, a.stage + ": " + a.reason.substr(0, 40)),
          "rank %d: the error does not carry stage %s's reason '%s'", me, a.stage.c_str(), a.reason.c_str());
  CHECK(on_all(!what.empty(), comm) && same_on_all_ranks(what, comm), "rank %d: the ranks report different errors",
        me);
  check_records_same(lu, comm, "every stage threw");
  std::string sw;
  try {
    lu.solve(bl);
  } catch (const axmp::Error& e) {
    sw = e.what();
  }
  CHECK(has(sw, "no factorization"), "rank %d: solve() after a chain that never factorized: %s", me,
        sw.empty() ? "returned" : sw.c_str());
}

// ------------------------------------------------------------------------------------------------ 14
// Each refusal is an axmp::Error that names what it refuses.
static void refusals() {
  auto error_of = [](const std::function<void()>& f) {
    try {
      f();
    } catch (const axmp::Error& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  std::string w = error_of([] { SparseLU lu(std::vector<LUStage>{}); });
  CHECK(has(w, "empty"), "an empty chain: %s", w.empty() ? "accepted" : w.c_str());
  w = error_of([] { SparseLU lu(std::vector<LUStage>{stage(LUBackend::EigenLU, MatchingJob::None, true)}); });
  CHECK(has(w, "equilibrat"), "a chain with an equilibrated stage (not available yet): %s",
        w.empty() ? "accepted" : w.c_str());
  for (LUBackend b : {LUBackend::StrumpackLU, LUBackend::StrumpackMPI})
    if (!lu_backend_available(b)) {
      w = error_of([b] { SparseLU lu(std::vector<LUStage>{stage(b), stage(LUBackend::EigenLU)}); });
      CHECK(has(w, "not in this build"), "a chain whose first stage (%s) is not in this build: %s", to_string(b),
            w.empty() ? "accepted" : w.c_str());
    }
  w = error_of([] {
    SparseLU lu(eigens(2));
    LUAcceptance a = rel_residual_acceptance(1e-10);
    a.stop_on_backward_error = 1e-14;
    lu.set_acceptance(a);
  });
  CHECK(has(w, "stop_on_backward_error"), "stop_on_backward_error != 0 (not available until measured): %s",
        w.empty() ? "accepted" : w.c_str());
  w = error_of([] {
    SparseLU lu(eigens(2));
    LUAcceptance a;
    a.kind = LUAcceptance::Kind::Verdict;
    a.bound = 1e-13;
    lu.set_acceptance(a);
  });
  CHECK(has(w, "verdict function"), "a Verdict acceptance without a function: %s", w.empty() ? "accepted" : w.c_str());
  {
    const LUAcceptance a;
    CHECK(a.kind == LUAcceptance::Kind::None && a.stop_on_backward_error == 0,
          "LUAcceptance's defaults are not Kind::None with the shortcut off");
  }
}

// ------------------------------------------------------------------------------------------------ 15
// The best iterate of a RelResidual chain is the one with the smallest relative residual, also when another stage
// has the smaller backward error. The system is section 6's (block 1, the 25 rows n0.., times 2^27, b zero on it).
// No uniform perturbation separates the two measures (the residual of (1 + eps) x is -eps b up to round-off, which
// raises both), so the two stages are perturbed differently:
//   stage 0: the unknowns n0.. (the columns of the zero block) times (1 + 1e-4). Block 1's rows have no entry in
//            those columns, so only block 0's residual grows, to about 1e-4 of its terms: the relative residual and
//            the backward error are both of that order (2.4e-5 and 6.5e-6);
//   stage 1: the unknowns below n0 times (1 +- 1e-9), the sign alternating with the index. Both blocks' residuals
//            grow to about 1e-9 of their terms, which is the backward error (1.0e-9); block 1's rows carry 2^27, so
//            the relative residual over all rows is about 1e-9 x 2^27 (8.3e-3), far above stage 0's.
// So stage 0 has the smaller relative residual and stage 1 the smaller backward error. The test asserts that order
// on the recorded numbers (and the numbers against its own long double evaluation) before it asserts which iterate
// is returned.
// The sizes, and why they hold for any right-hand side. Nothing here is compared with the round-off of a particular
// solve (the unperturbed solve's relative residual ranges from 6e-16 to 6e-10 over right-hand sides of this kind):
//   - the comparison with the long double evaluation allows what a double evaluation of the residual can be off by,
//     the a priori bound E = || gamma (|A| |x| + |b|) || / ||b|| with gamma = (entries of the row + 2) u per row
//     (the classical bound of a computed inner product), which the test evaluates for each iterate (3.7e-9 here; the
//     2^27 rows make it that large). For the backward error, a quotient by || |A| |x| + |b| ||, the same bound is
//     gamma itself: 2 (largest row + 2) u is allowed;
//   - the perturbations, 1e-4 and 1e-9, stand far above those: asserted, each relative residual at least 100 E and
//     each backward error at least 100 times its allowance, so the numbers of another iterate cannot pass;
//   - the two measures order the stages oppositely by a factor of ten at least (asserted).
// Over 200 right-hand sides of this kind (seeds 1 to 200) the smallest margins were: E against the actual difference
// 11.7; relative residual against E 3200 (asserted 100); stage 1's relative residual against stage 0's 211 (asserted
// 10); stage 0's backward error against stage 1's 3830 (asserted 10); the backward errors' allowance against the
// actual difference 47 with 2^-49, which this matrix's 2 (largest row + 2) u is not below.
struct RoundoffBound {
  double rel_residual = 0;    // E
  double backward_error = 0;  // 2 (largest row + 2) u
};
static RoundoffBound roundoff_bound(const DistCsr& D, const std::vector<double>& xl, const std::vector<double>& bl,
                                    const std::vector<int>& dist, MPI_Comm comm) {
  const std::vector<double> x = gather_all(xl, dist, comm);
  const long double u = std::ldexp(1.0L, -53);
  long double loc[3] = {0, 0, 0}, glob[3] = {0, 0, 0};  // sum (gamma d)^2, sum b^2, largest row
  for (int i = 0; i < D.local_rows; ++i) {
    long double d = std::fabs(static_cast<long double>(bl[i]));
    for (int k = D.rowptr[i]; k < D.rowptr[i + 1]; ++k) d += std::fabs(static_cast<long double>(D.val[k]) * x[D.col[k]]);
    const long double gamma = (D.rowptr[i + 1] - D.rowptr[i] + 2) * u;
    loc[0] += gamma * d * gamma * d;
    loc[1] += static_cast<long double>(bl[i]) * bl[i];
    loc[2] = std::max(loc[2], static_cast<long double>(D.rowptr[i + 1] - D.rowptr[i]));
  }
  MPI_Allreduce(loc, glob, 2, MPI_LONG_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(loc + 2, glob + 2, 1, MPI_LONG_DOUBLE, MPI_MAX, comm);
  RoundoffBound out;
  out.rel_residual = static_cast<double>(std::sqrt(glob[0] / glob[1]));
  out.backward_error = static_cast<double>(2 * (glob[2] + 2) * u);
  return out;
}
static void best_iterate_opposite_orders(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const int me = rank_of(comm);
  const SpMat S = oseen_like(30, 3.0, 25);
  const int n = static_cast<int>(S.rows()), n0 = 900;
  std::vector<int> dist;
  DistCsr DA = distributed(S, lc, dist);
  for (int i = 0; i < DA.local_rows; ++i)
    if (DA.row_begin + i >= n0)
      for (int k = DA.rowptr[i]; k < DA.rowptr[i + 1]; ++k) DA.val[k] = std::ldexp(DA.val[k], 27);
  std::mt19937 rng(41);
  std::uniform_real_distribution<double> U(-1, 1);
  Eigen::VectorXd bg = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n0; ++i) bg(i) = U(rng);
  std::vector<int> blocks_g(n);
  for (int i = 0; i < n; ++i) blocks_g[i] = i < n0 ? 0 : 1;
  const auto bl = local_of(bg, dist, me);
  const auto blocks = local_ints(blocks_g, dist, me);
  const int row0 = dist[me];
  std::vector<std::vector<double>> seen(2);  // the perturbed iterate of each stage
  std::vector<int> calls(2, 0);
  set_lu_iterate_hook_for_testing([&](int k, std::vector<double>& x) {
    if (k < 0 || k > 1) return;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const int gi = row0 + static_cast<int>(i);
      if (k == 0 && gi >= n0) x[i] *= 1.0 + 1e-4;
      if (k == 1 && gi < n0) x[i] *= 1.0 + (gi % 2 == 0 ? 1e-9 : -1e-9);
    }
    seen[k] = x;
    ++calls[k];
  });
  SparseLU lu(eigens(2));
  lu.set_throw_on_miss(false);
  lu.set_row_blocks(blocks);
  lu.set_acceptance(rel_residual_acceptance(1e-30));
  lu.factorize(DA);
  const auto x = lu.solve(bl);
  set_lu_iterate_hook_for_testing({});
  const auto& at = lu.attempts();
  CHECK(at.size() == 2 && outcomes(lu) == "missed_target,missed_target" && lu.accepted_stage() == -1 &&
            calls[0] == 1 && calls[1] == 1,
        "rank %d: attempts [%s], accepted stage %d, hook calls %d and %d (expected both stages rejected, one call each)",
        me, outcomes(lu).c_str(), lu.accepted_stage(), calls[0], calls[1]);
  if (at.size() != 2 || calls[0] != 1 || calls[1] != 1) return;
  if (me == 0)
    std::printf("opposite orders: stage 0 relative residual %.3e, backward error %.3e; stage 1 %.3e, %.3e\n",
                at[0].rel_residual, at[0].backward_error, at[1].rel_residual, at[1].backward_error);
  // the recorded numbers are those of the perturbed iterates (the test's own long double evaluation), to within what
  // a double evaluation of the residual can be off by; and the perturbations stand far above that
  for (int k = 0; k < 2; ++k) {
    const Ref ref = reference(DA, seen[k], bl, blocks, {}, dist, comm);
    const RoundoffBound tol = roundoff_bound(DA, seen[k], bl, dist, comm);
    if (me == 0)
      std::printf("  stage %d against long double: relative residual off by %.2g of the bound %.3g, backward error by "
                  "%.2g of %.3g\n",
                  k, std::fabs(at[k].rel_residual - ref.rel_residual) / tol.rel_residual, tol.rel_residual,
                  std::fabs(at[k].backward_error - ref.backward_error) / tol.backward_error, tol.backward_error);
    CHECK(at[k].rel_residual >= 100 * tol.rel_residual && at[k].backward_error >= 100 * tol.backward_error,
          "stage %d: the perturbation does not stand above the round-off of the evaluation: relative residual %g "
          "(bound %g), backward error %g (bound %g)",
          k, at[k].rel_residual, tol.rel_residual, at[k].backward_error, tol.backward_error);
    CHECK(std::fabs(at[k].rel_residual - ref.rel_residual) <= tol.rel_residual &&
              std::fabs(at[k].backward_error - ref.backward_error) <= tol.backward_error &&
              bitwise(at[k].measure, at[k].rel_residual),
          "stage %d: recorded relative residual %.9g (long double %.9g), backward error %.9g (%.9g), measure %.9g", k,
          at[k].rel_residual, ref.rel_residual, at[k].backward_error, ref.backward_error, at[k].measure);
  }
  // the construction: the two measures order the stages oppositely, each by a factor of ten at least
  const bool opposite =
      at[0].rel_residual * 10 <= at[1].rel_residual && at[1].backward_error * 10 <= at[0].backward_error;
  CHECK(opposite,
        "the construction failed: relative residuals %g, %g and backward errors %g, %g do not order the stages "
        "oppositely",
        at[0].rel_residual, at[1].rel_residual, at[0].backward_error, at[1].backward_error);
  // the acceptance measure chooses: stage 0, whose backward error is the larger one
  CHECK(on_all(bitwise(x, seen[0]), comm),
        "rank %d: the returned iterate is not stage 0's, the one with the smaller relative residual (stage 1's, the "
        "one with the smaller backward error: %s)",
        me, bitwise(x, seen[1]) ? "yes" : "no");
  CHECK(bitwise(lu.stats().measure, at[0].measure) && bitwise(lu.stats().rel_residual, at[0].rel_residual) &&
            bitwise(lu.stats().backward_error, at[0].backward_error) && lu.stats().attempts == 2,
        "stats() does not describe stage 0: measure %g (stage 0's %g), backward error %g (stage 0's %g)",
        lu.stats().measure, at[0].measure, lu.stats().backward_error, at[0].backward_error);
  check_records_same(lu, comm, "opposite orders");
}

// ------------------------------------------------------------------------------------------------ 16
// A walk that ends without an accepted stage returns the best iterate and describes its stage in stats(), but the
// factorization held is the last stage's. The next solve of the same factorization is that stage's: its record
// (index width, fill, factorization times) and stats() (stage, backend) must say so. The stages are made to differ:
//   (a) two Eigen stages, the second one factorized with the 64-bit index type (the test bound of the 32-bit one is
//       lowered after stage 0 has factorized), in every build;
//   (b) STRUMPACK then Eigen, and Eigen then STRUMPACK, in builds with STRUMPACK.
// The verdicts are scripted: 1 and 3 against a bound of 0.5 (nobody accepted, stage 0 the best), then 0.1.
static void held_stage_record(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  struct Max32Reset {
    ~Max32Reset() { set_eigen_lu_max_n32_for_testing(0); }
  } reset32;
  const int me = rank_of(comm);
  const Sys s = make_sys(lc);        // a saddle point: STRUMPACK needs the matching
  const Sys s0 = make_sys(lc, 7, 0);  // no zero block: STRUMPACK without matching factorizes it
  struct Case {
    const char* what;
    std::vector<LUStage> chain;
    const Sys* sys;
    bool second_64bit;
    const char* held_stage;
    const char* held_backend;
    int held_bits;  // index_bits of the held stage (rank 0's, recorded on every rank)
  };
  std::vector<Case> cases = {{"[eigen 32-bit, eigen 64-bit]", eigens(2), &s, true, "eigen", "eigen", 64}};
  if (lu_backend_available(LUBackend::StrumpackLU)) {
    cases.push_back({"[strumpack, eigen]", {stage(LUBackend::StrumpackLU), stage(LUBackend::EigenLU)}, &s, false,
                     "eigen", "eigen", 32});
    cases.push_back({"[eigen, strumpack:none]",
                     {stage(LUBackend::EigenLU), stage(LUBackend::StrumpackLU, MatchingJob::None)}, &s0, false,
                     "strumpack:none", "strumpack", 0});
  } else if (me == 0) {
    std::printf("skip the STRUMPACK cases of the held stage (not in this build)\n");
  }
  for (const auto& c : cases) {
    ScriptedVerdict v;
    v.comm = lc;
    v.values = {1.0, 3.0, 0.1};
    SparseLU lu(c.chain);
    lu.set_throw_on_miss(false);
    lu.set_acceptance(v.acceptance(0.5));
    lu.factorize(c.sys->D);
    if (c.second_64bit) set_eigen_lu_max_n32_for_testing(100);  // stage 1 factorizes during the first solve
    lu.solve(c.sys->bl);
    set_eigen_lu_max_n32_for_testing(0);
    const std::vector<LUAttempt> first = lu.attempts();
    const LUStats st1 = lu.stats();
    CHECK(first.size() == 2 && outcomes(lu) == "missed_target,missed_target" && lu.accepted_stage() == -1 &&
              st1.stage == c.chain[0].name() && st1.measure == 1.0,
          "%s, solve 1: attempts [%s], accepted stage %d, stats() stage '%s' measure %g (expected nobody accepted and "
          "stage 0, the best, described)",
          c.what, outcomes(lu).c_str(), lu.accepted_stage(), st1.stage.c_str(), st1.measure);
    if (first.size() != 2) continue;
    // the construction: the two stages' records differ, so a record of the wrong stage shows
    CHECK(first[0].index_bits != first[1].index_bits && first[1].index_bits == c.held_bits,
          "%s: the stages' index widths are %d and %d (expected different, the second %d)", c.what,
          first[0].index_bits, first[1].index_bits, c.held_bits);
    const auto x = lu.solve(c.sys->bl);  // starts on the held stage, which the verdict now accepts
    const auto& at = lu.attempts();
    const LUStats& st = lu.stats();
    CHECK(at.size() == 1 && at[0].outcome == "accepted" && lu.accepted_stage() == 1, "%s, solve 2: attempts [%s], "
          "accepted stage %d (expected the held stage 1 accepted)", c.what, outcomes(lu).c_str(), lu.accepted_stage());
    if (at.size() != 1) continue;
    CHECK(at[0].stage == c.held_stage && at[0].index_bits == c.held_bits && at[0].factor_nnz == first[1].factor_nnz &&
              bitwise(at[0].analyze_seconds, first[1].analyze_seconds) &&
              bitwise(at[0].factor_seconds, first[1].factor_seconds),
          "%s, solve 2: the attempt is stage '%s' with index width %d, fill %ld, factor time %a (the held stage's: "
          "'%s', %d, %ld, %a)",
          c.what, at[0].stage.c_str(), at[0].index_bits, at[0].factor_nnz, at[0].factor_seconds, c.held_stage,
          c.held_bits, first[1].factor_nnz, first[1].factor_seconds);
    CHECK(st.stage == c.held_stage && st.backend == c.held_backend && st.index_bits == c.held_bits &&
              st.factor_nnz == first[1].factor_nnz && st.measure == 0.1 && st.attempts == 1,
          "%s, solve 2: stats() stage '%s', backend '%s', index width %d, fill %ld, measure %g (expected '%s', '%s', "
          "%d, %ld, 0.1)",
          c.what, st.stage.c_str(), st.backend.c_str(), st.index_bits, st.factor_nnz, st.measure, c.held_stage,
          c.held_backend, c.held_bits, first[1].factor_nnz);
    // what is held is what the record says
    const int live = lu.live_eigen_index_bits_for_testing();
    CHECK(live == (me == 0 ? c.held_bits : 0), "%s: rank %d holds an Eigen object of width %d (expected %d)", c.what,
          me, live, me == 0 ? c.held_bits : 0);
    CHECK(global_err(x, c.sys->xs, c.sys->dist, comm) < 1e-10, "%s, solve 2: solution error %g", c.what,
          global_err(x, c.sys->xs, c.sys->dist, comm));
    check_records_same(lu, comm, c.what);
  }
}

// ------------------------------------------------------------------------------------------------ 17
// Non-finite iterates and measures. The iterate hook plants a NaN in the first local entry of the LAST rank (rank 1 on
// two ranks: rank 0's part stays finite, so the finiteness has to be reduced over the ranks). The verdicts are
// scripted and do not look at the iterate, as a caller's verdict need not see every entry.
static bool finite_everywhere(const std::vector<double>& x, MPI_Comm comm) {
  bool ok = true;
  for (double v : x) ok = ok && std::isfinite(v);
  return on_all(ok, comm);
}
static void non_finite(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const int me = s.me, last = size_of(lc) - 1;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::vector<double>> seen(2);
  int nan_stages = 1;  // the stages 0 .. nan_stages - 1 get the NaN
  auto hook = [&](int k, std::vector<double>& x) {
    if (k < nan_stages && me == last && !x.empty()) x[0] = nan;
    if (k >= 0 && k < 2) seen[k] = x;
  };
  auto chain = [&](ScriptedVerdict& v) {
    v.comm = lc;
    auto lu = std::make_unique<SparseLU>(eigens(2));
    lu->set_throw_on_miss(false);
    lu->set_acceptance(v.acceptance(0.5));
    lu->factorize(s.D);
    return lu;
  };
  // (a) a verdict within the bound does not make a non-finite iterate acceptable
  {
    ScriptedVerdict v;
    v.values = {0.1};
    auto lu = chain(v);
    set_lu_iterate_hook_for_testing(hook);
    const auto x = lu->solve(s.bl);
    set_lu_iterate_hook_for_testing({});
    const auto& at = lu->attempts();
    CHECK(outcomes(*lu) == "missed_target,accepted" && lu->accepted_stage() == 1 &&
              at[0].reason == "non-finite solution" && at[0].measure == 0.1,
          "rank %d, a NaN in stage 0's iterate on rank %d with verdict 0.1 <= 0.5: attempts [%s], reason '%s' (expected "
          "stage 0 rejected as a non-finite solution on every rank)",
          me, last, outcomes(*lu).c_str(), at.empty() ? "" : at[0].reason.c_str());
    CHECK(finite_everywhere(x, comm) && global_err(x, s.xs, s.dist, comm) < 1e-10,
          "rank %d: the returned iterate is not the finite stage's", me);
    check_records_same(*lu, comm, "non-finite, accepted next");
  }
  // (b) nobody accepted: the non-finite iterate has the smaller verdict (1 against 3), and is not the best
  {
    ScriptedVerdict v;
    v.values = {1.0, 3.0};
    auto lu = chain(v);
    set_lu_iterate_hook_for_testing(hook);
    const auto x = lu->solve(s.bl);
    set_lu_iterate_hook_for_testing({});
    const auto& at = lu->attempts();
    CHECK(outcomes(*lu) == "missed_target,missed_target" && lu->accepted_stage() == -1 &&
              at[0].reason == "non-finite solution" && starts(at[1].reason, "verdict 3 > "),
          "non-finite with the smaller verdict: attempts [%s], reasons '%s', '%s'", outcomes(*lu).c_str(),
          at.empty() ? "" : at[0].reason.c_str(), at.size() < 2 ? "" : at[1].reason.c_str());
    CHECK(finite_everywhere(x, comm) && on_all(bitwise(x, seen[1]), comm) && lu->stats().measure == 3.0,
          "rank %d: the best iterate is not the finite one (stage 1, verdict 3): returned %s, stats() measure %g", me,
          finite_everywhere(x, comm) ? "finite" : "NON-FINITE", lu->stats().measure);
    check_records_same(*lu, comm, "non-finite, best");
  }
  // (c) every stage's iterate non-finite: one of them is returned (there is no other), and nothing is accepted
  {
    ScriptedVerdict v;
    v.values = {0.1, 0.2};
    auto lu = chain(v);
    nan_stages = 2;
    set_lu_iterate_hook_for_testing(hook);
    const auto x = lu->solve(s.bl);
    set_lu_iterate_hook_for_testing({});
    nan_stages = 1;
    const auto& at = lu->attempts();
    CHECK(outcomes(*lu) == "missed_target,missed_target" && at[0].reason == "non-finite solution" &&
              at[1].reason == "non-finite solution" && !finite_everywhere(x, comm),
          "every iterate non-finite: attempts [%s]", outcomes(*lu).c_str());
  }
  // (d) a NaN verdict is not the smallest measure: the best is the stage with the finite one (3), although NaN
  // compares false with everything
  {
    ScriptedVerdict v;
    v.values = {nan, 3.0};
    auto lu = chain(v);
    set_lu_iterate_hook_for_testing([&](int k, std::vector<double>& x) {
      if (k >= 0 && k < 2) seen[k] = x;
    });
    const auto x = lu->solve(s.bl);
    set_lu_iterate_hook_for_testing({});
    CHECK(outcomes(*lu) == "missed_target,missed_target" && lu->stats().measure == 3.0 &&
              bitwise(lu->stats().backward_error, lu->attempts()[1].backward_error),
          "verdicts NaN and 3: attempts [%s], stats() measure %g (expected stage 1's 3)", outcomes(*lu).c_str(),
          lu->stats().measure);
    check_records_same(*lu, comm, "NaN verdict");
  }
  // (e) the same with the NaN second: the first stage (3) stays the best
  {
    ScriptedVerdict v;
    v.values = {3.0, nan};
    auto lu = chain(v);
    lu->solve(s.bl);
    CHECK(lu->stats().measure == 3.0 && bitwise(lu->stats().backward_error, lu->attempts()[0].backward_error),
          "verdicts 3 and NaN: stats() measure %g (expected stage 0's 3)", lu->stats().measure);
  }
}

// ------------------------------------------------------------------------------------------------ 18
// Row blocks that are wrong on ONE rank (the last one) are an axmp::Error on EVERY rank: a rank that threw alone would
// leave the others in the next collective call (the TIMEOUT would end the test). And the blocks are read by
// factorize(): a later set_row_blocks changes nothing until the next factorize().
static void row_block_errors(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const int me = s.me, last = size_of(lc) - 1, n0 = 900;
  std::vector<int> good(s.D.local_rows);
  for (int i = 0; i < s.D.local_rows; ++i) good[i] = s.D.row_begin + i < n0 ? 0 : 1;
  std::vector<int> too_long = good, bad_id = good;
  if (me == last) {
    too_long.push_back(0);
    bad_id[0] = -3;
  }
  auto error_of = [](const std::function<void()>& f) {
    try {
      f();
    } catch (const axmp::Error& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  struct B {
    const char* what;
    const std::vector<int>* blocks;
    const char* names;
  };
  const B bad[] = {{"one entry too many", &too_long, "row blocks have"}, {"an id of -3", &bad_id, "row block id -3"}};
  for (const auto& b : bad) {
    SparseLU lu(LUBackend::EigenLU);
    lu.set_row_blocks(*b.blocks);
    const std::string w = error_of([&] { lu.factorize(s.D); });
    CHECK(on_all(has(w, b.names), comm) && same_on_all_ranks(w, comm),
          "rank %d, factorize with %s on rank %d: %s (expected one error on every rank)", me, b.what, last,
          w.empty() ? "no error" : w.c_str());
    const std::string wb = error_of([&] { lu_backward_error(s.D, s.bl, s.bl, *b.blocks); });
    CHECK(on_all(has(wb, b.names), comm) && same_on_all_ranks(wb, comm),
          "rank %d, lu_backward_error with %s on rank %d: %s (expected one error on every rank)", me, b.what, last,
          wb.empty() ? "no error" : wb.c_str());
    const std::string ws = error_of([&] { lu.solve(s.bl); });
    CHECK(on_all(has(ws, "no factorization"), comm), "rank %d: solve after the refused factorize: %s", me,
          ws.empty() ? "returned" : ws.c_str());
  }
  auto hint_of = [](const std::function<void()>& f) {
    try {
      f();
    } catch (const axmp::Error& e) {
      return e.hint();
    }
    return std::string("(no error)");
  };
  {  // x of the wrong size on one rank: its own error, and a hint that is not the row blocks' one
    std::vector<double> x = s.bl;
    if (me == last) x.push_back(0.0);
    const std::string w = error_of([&] { lu_backward_error(s.D, x, s.bl, good); });
    CHECK(on_all(has(w, "local rows"), comm), "rank %d, lu_backward_error with x one entry too long on rank %d: %s", me,
          last, w.empty() ? "no error" : w.c_str());
    const std::string h = hint_of([&] { lu_backward_error(s.D, x, s.bl, good); });
    CHECK(on_all(!h.empty() && !has(h, "block id") && !has(h, "set_row_blocks"), comm),
          "rank %d, lu_backward_error with x one entry too long on rank %d: the hint is about the row blocks, not "
          "about x and b ('%s')", me, last, h.c_str());
    std::vector<double> b = s.bl;  // the same with b
    if (me == last) b.pop_back();
    const std::string hb = hint_of([&] { lu_backward_error(s.D, s.bl, b, good); });
    CHECK(on_all(!hb.empty() && !has(hb, "block id") && !has(hb, "set_row_blocks"), comm),
          "rank %d, lu_backward_error with b one entry short on rank %d: the hint is about the row blocks ('%s')", me,
          last, hb.c_str());
    // the row blocks' own errors keep the row blocks' hint
    const std::string hi = hint_of([&] { lu_backward_error(s.D, s.bl, s.bl, bad_id); });
    CHECK(on_all(has(hi, "block id"), comm), "rank %d, lu_backward_error with an id of -3: the hint does not say what "
          "to give ('%s')", me, hi.c_str());
  }
  // the blocks are factorize()'s: changed afterwards to other valid ids (every row one block), the solve records the
  // backward error of the blocks it was factorized with, which differs from that of the new ids; changed to a vector
  // that would be refused, the next factorize() refuses it
  {
    std::vector<int> one_block(s.D.local_rows, 0);
    SparseLU lu(LUBackend::EigenLU);
    lu.set_row_blocks(good);
    lu.factorize(s.D);
    lu.set_row_blocks(one_block);
    std::vector<double> x;
    const std::string w = error_of([&] { x = lu.solve(s.bl); });
    CHECK(on_all(w.empty(), comm), "rank %d: solve after set_row_blocks following factorize: %s", me, w.c_str());
    if (on_all(w.empty() && lu.attempts().size() == 1, comm)) {
      const double be = lu_backward_error(s.D, x, s.bl, good), be1 = lu_backward_error(s.D, x, s.bl, one_block);
      // the teeth of this case: the two block layouts give different numbers for this x
      CHECK(!bitwise(be, be1), "the backward errors of the two block layouts are equal (%a): this case cannot tell "
            "which blocks the solve used", be);
      CHECK(bitwise(lu.attempts()[0].backward_error, be),
            "the recorded backward error %a is not that of the blocks given to factorize (%a; the later blocks "
            "give %a)", lu.attempts()[0].backward_error, be, be1);
    }
    lu.set_row_blocks(too_long);
    const std::string w1 = error_of([&] { x = lu.solve(s.bl); });
    CHECK(on_all(w1.empty(), comm), "rank %d: solve after set_row_blocks (too long) following factorize: %s", me,
          w1.c_str());
    const std::string w2 = error_of([&] { lu.factorize(s.D); });
    CHECK(on_all(has(w2, "row blocks have"), comm), "rank %d: the next factorize did not read the new blocks: %s", me,
          w2.empty() ? "no error" : w2.c_str());
  }
}

// ------------------------------------------------------------------------------------------------ 19
// The size cap of a stage (LUStage::max_n). Reference: the size of the system (n = 925) against the cap.
static void size_cap(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const long n = s.S.rows();
  auto capped = [](long cap) {
    LUStage st = stage(LUBackend::EigenLU);
    st.max_n = cap;
    return st;
  };
  {  // above the cap: refused, reason "size...", and the next stage solves
    SparseLU lu(std::vector<LUStage>{capped(n - 1), stage(LUBackend::EigenLU)});
    lu.set_acceptance(rel_residual_acceptance(1e-10));
    lu.factorize(s.D);
    const std::vector<double> x = lu.solve(s.bl);
    const auto& at = lu.attempts();
    CHECK(at.size() == 2 && at[0].outcome == "refused" && starts(at[0].reason, "size") && at[1].outcome == "accepted",
          "a stage capped at n - 1 = %ld on n = %ld: attempts [%s], expected [refused (size...), accepted]", n - 1, n,
          outcomes(lu).c_str());
    if (!at.empty())
      CHECK(at[0].factor_seconds == 0 && at[0].measure == -1, "the refused stage records a factorization (%.3g s) or "
            "a measure (%.3g)", at[0].factor_seconds, at[0].measure);
    CHECK(lu.accepted_stage() == 1, "accepted stage %d, expected 1", lu.accepted_stage());
    CHECK(global_err(x, s.xs, s.dist, comm) < 1e-10, "the capped chain's solution misses the manufactured one");
    check_records_same(lu, comm, "size cap");
  }
  {  // at the cap: factorized
    SparseLU lu(std::vector<LUStage>{capped(n), stage(LUBackend::EigenLU)});
    lu.set_acceptance(rel_residual_acceptance(1e-10));
    lu.factorize(s.D);
    lu.solve(s.bl);
    CHECK(lu.attempts().size() == 1 && lu.attempts()[0].outcome == "accepted",
          "a stage capped at n = %ld on n = %ld: attempts [%s], expected [accepted]", n, n, outcomes(lu).c_str());
  }
  {  // a one-stage chain refused by its cap: the error names the size
    SparseLU lu(std::vector<LUStage>{capped(10)});
    std::string w;
    try {
      lu.factorize(s.D);
    } catch (const axmp::Error& e) {
      w = e.what();
    }
    CHECK(on_all(has(w, "size"), comm), "a one-stage chain capped at 10 on n = %ld: '%s' (expected an error naming the "
          "size)", n, w.empty() ? "no error" : w.c_str());
  }
}

// ------------------------------------------------------------------------------------------------ 20
// A tie of the acceptance measure where one iterate's backward error is NaN: the NaN counts as +infinity. Stage 0's
// iterate is multiplied by 1e200 (the iterate hook): it stays finite, but the sums of squares of its residual and of
// |A||x| + |b| overflow, so its blockwise ratio is inf / inf = NaN: it has no backward error, and a maximum over the
// blocks that skipped the NaN would record 0, the best possible value. A scripted Verdict gives both stages the same
// value, above the bound, so no stage is accepted and the tie is broken by the backward error: stage 1's iterate
// (finite backward error) is returned, although stage 0's came first.
static void nan_backward_error_tie(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  set_lu_iterate_hook_for_testing([](int k, std::vector<double>& x) {
    if (k == 0)
      for (double& v : x) v *= 1e200;
  });
  ScriptedVerdict v;
  v.values = {5.0};
  v.comm = lc;
  SparseLU lu(eigens(2));
  lu.set_acceptance(v.acceptance(1.0));
  lu.set_throw_on_miss(false);  // as every fluids caller: the returned iterate is what is checked here
  lu.factorize(s.D);
  const std::vector<double> x = lu.solve(s.bl);
  const auto& at = lu.attempts();
  CHECK(at.size() == 2 && at[0].outcome == "missed_target" && at[1].outcome == "missed_target",
        "attempts [%s], expected [missed_target, missed_target]", outcomes(lu).c_str());
  if (rank_of(comm) == 0 && at.size() == 2)
    std::printf("NaN tie: measures %g %g, backward errors %g %g, rel. residuals %g %g\n", at[0].measure, at[1].measure,
                at[0].backward_error, at[1].backward_error, at[0].rel_residual, at[1].rel_residual);
  if (at.size() != 2) return;
  // the construction: a tie of the measure, a NaN backward error on stage 0, a finite one on stage 1
  CHECK(at[0].measure == 5.0 && at[1].measure == 5.0, "the measures %g and %g are not tied at 5", at[0].measure,
        at[1].measure);
  CHECK(std::isnan(at[0].backward_error) && std::isfinite(at[1].backward_error),
        "backward errors %g and %g: an iterate whose residual overflows (stage 0) has none (expected NaN), stage "
        "1's is finite", at[0].backward_error, at[1].backward_error);
  {  // the same number from the function the chain records with
    std::vector<double> big(s.bl.size());
    const std::vector<double> xs = local_of(s.xs, s.dist, s.me);
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = 1e200 * xs[i];
    const double be = lu_backward_error(s.D, big, s.bl, {});
    CHECK(std::isnan(be), "lu_backward_error of 1e200 times the solution is %g, expected NaN (its residual overflows)",
          be);
  }
  CHECK(lu.accepted_stage() == -1, "accepted stage %d, expected -1 (none)", lu.accepted_stage());
  double big = 0;
  for (double e : x) big = std::max(big, std::fabs(e));
  double gbig = 0;
  MPI_Allreduce(&big, &gbig, 1, MPI_DOUBLE, MPI_MAX, comm);
  CHECK(gbig < 1e10 && global_err(x, s.xs, s.dist, comm) < 1e-10,
        "the returned iterate is stage 0's (max |x| = %.3g), whose backward error is NaN: a NaN must lose the tie",
        gbig);
  CHECK(std::isfinite(lu.stats().backward_error) && bitwise(lu.stats().backward_error, at[1].backward_error),
        "stats() describes the iterate with backward error %g, expected stage 1's %g", lu.stats().backward_error,
        at[1].backward_error);
  check_records_same(lu, comm, "NaN backward-error tie");
}

// ------------------------------------------------------------------------------------------------ 21
// [eigen, eigen capped below n], acceptance at a bound no solve meets (relative residual 0):
// every solve walks to the capped stage, which is refused without being factorized. Reference: the held stage is
// stage 0 throughout, so three solves give the same records and the same iterate, bit for bit.
static void capped_tail_keeps_the_held_stage(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const Sys s = make_sys(lc);
  const long n = s.S.rows();
  LUStage capped = stage(LUBackend::EigenLU);
  capped.max_n = n - 1;
  SparseLU lu(std::vector<LUStage>{stage(LUBackend::EigenLU), capped});
  lu.set_acceptance(rel_residual_acceptance(0.0));
  lu.set_throw_on_miss(false);
  lu.factorize(s.D);
  std::vector<double> first;
  for (int k = 0; k < 3; ++k) {
    std::vector<double> x;
    std::string w;
    try {
      x = lu.solve(s.bl);
    } catch (const axmp::Error& e) {
      w = e.what();
    }
    CHECK(on_all(w.empty(), comm), "solve %d after a walk that ended on a size-capped stage: %s", k + 1,
          w.empty() ? "solved on another rank" : w.c_str());
    if (!on_all(w.empty(), comm)) return;
    const auto& at = lu.attempts();
    CHECK(at.size() == 2 && at[0].stage == "eigen" && at[0].outcome == "missed_target" && at[1].outcome == "refused" &&
              starts(at[1].reason, "size"),
          "solve %d: attempts [%s], expected [eigen:missed_target, eigen:refused (size)]", k + 1, outcomes(lu).c_str());
    CHECK(lu.accepted_stage() == -1 && lu.solution_stage() == 0, "solve %d: accepted stage %d, solution stage %d, "
          "expected -1 and 0", k + 1, lu.accepted_stage(), lu.solution_stage());
    CHECK(global_err(x, s.xs, s.dist, comm) < 1e-10, "solve %d: the returned iterate misses the manufactured solution",
          k + 1);
    if (k == 0) first = x;
    else CHECK(on_all(bitwise(x, first), comm), "solve %d: the iterate differs from the first solve's", k + 1);
    check_records_same(lu, comm, "capped tail");
  }
}

// ------------------------------------------------------------------------------------------------ 22
// The default chains walked, in STRUMPACK builds. The iterate hook multiplies the iterates of chain indices 0 and 1 by
// 1 + 1e-6, so that both miss a relative-residual bound of 1e-10 (their residual is about 1e-6 of ||b||) and the walk
// reaches the last stage, which is accepted on the manufactured solution. Reference: the lists of section 1 and the
// order of the walk; the system has no zero block, so every stage factorizes it (a stage without matching included).
// A walk in which only stage 0 misses holds stage 1's solver object, whose MC64 job is the stage's: 2 for the second
// stage of the 2D list, 5 for the second stage of the 3D list. An explicit chain is walked as given.
static void default_chains_walked(MPI_Comm comm, MPI_Comm lc) {
  HookReset reset;
  const int me = rank_of(comm);
  if (!lu_backend_available(LUBackend::StrumpackLU)) {
    if (me == 0) std::printf("skip (STRUMPACK is not in this build)\n");
    return;
  }
  using M = MatchingJob;
  const Sys s = make_sys(lc, 7, 0);
  auto miss = [](int upto) {
    set_lu_iterate_hook_for_testing([upto](int k, std::vector<double>& x) {
      if (k > upto) return;
      for (double& v : x) v *= 1.0 + 1e-6;
    });
  };
  struct W {
    const char* what;
    std::vector<LUStage> chain;
    const char* stages;  // the expected walk, stage names
    int job1;            // the MC64 job of stage 1's solver object (rank 0)
  };
  std::vector<LUStage> explicit_chain = {stage(LUBackend::StrumpackLU, M::None),
                                         stage(LUBackend::StrumpackLU, M::MaxSmallestDiagonal),
                                         stage(LUBackend::EigenLU)};
  const W walks[] = {
      {"the 2D list", default_chain(LUBackend::StrumpackLU, M::MaxDiagonalProductScaling),
       "strumpack:mc64-5,strumpack:mc64-2,eigen", 2},
      {"the 3D list", default_chain(LUBackend::StrumpackLU, M::None), "strumpack:none,strumpack:mc64-5,eigen", 5},
      {"an explicit chain", explicit_chain, "strumpack:none,strumpack:mc64-2,eigen", 2},
  };
  for (const auto& w : walks) {
    {  // stages 0 and 1 miss: every stage is tried, in order, and the last is accepted
      SparseLU lu(w.chain);
      lu.set_acceptance(rel_residual_acceptance(1e-10));
      lu.factorize(s.D);
      miss(1);
      const std::vector<double> x = lu.solve(s.bl);
      set_lu_iterate_hook_for_testing({});
      std::string got;
      for (const auto& a : lu.attempts()) got += (got.empty() ? "" : ",") + a.stage;
      CHECK(got == w.stages && outcomes(lu) == "missed_target,missed_target,accepted" && lu.accepted_stage() == 2,
            "%s walked to its end: stages [%s], outcomes [%s], accepted stage %d; expected [%s], "
            "[missed_target,missed_target,accepted], 2", w.what, got.c_str(), outcomes(lu).c_str(), lu.accepted_stage(),
            w.stages);
      CHECK(lu.stats().stage == "eigen", "%s: stats().stage '%s', expected 'eigen'", w.what, lu.stats().stage.c_str());
      CHECK(global_err(x, s.xs, s.dist, comm) < 1e-10, "%s: the accepted solution misses the manufactured one (%.3g)",
            w.what, global_err(x, s.xs, s.dist, comm));
      check_records_same(lu, comm, w.what);
    }
    {  // stage 0 alone misses: stage 1 is accepted, and its solver object has the stage's MC64 job
      SparseLU lu(w.chain);
      lu.set_acceptance(rel_residual_acceptance(1e-10));
      lu.factorize(s.D);
      miss(0);
      const std::vector<double> x = lu.solve(s.bl);
      set_lu_iterate_hook_for_testing({});
      const int job = lu.strumpack_matching_job_for_testing();
      CHECK(outcomes(lu) == "missed_target,accepted" && lu.accepted_stage() == 1 &&
                lu.attempts().size() == 2 && lu.attempts()[1].stage == lu.chain()[1].name(),
            "%s, stage 0 missing: outcomes [%s], accepted stage %d (expected [missed_target,accepted], 1)", w.what,
            outcomes(lu).c_str(), lu.accepted_stage());
      CHECK(job == (me == 0 ? w.job1 : -1), "rank %d, %s: stage 1's solver object has MC64 job %d, expected %d", me,
            w.what, job, me == 0 ? w.job1 : -1);
      CHECK(global_err(x, s.xs, s.dist, comm) < 1e-10, "%s, stage 1: the solution misses the manufactured one", w.what);
      check_records_same(lu, comm, w.what);
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // a run ended by the TIMEOUT still shows the sections that finished
  const MPI_Comm comm = MPI_COMM_WORLD;
  const int P = size_of(comm);
  // a launcher that starts N one-rank jobs instead of one N-rank job must not pass for the 2-rank registration
  if (const char* e = std::getenv("AXMP_EXPECT_RANKS"))
    CHECK(P == std::atoi(e), "started on %d rank(s), expected %s", P, e);
  MPI_Comm lc;  // the LU's communicator: a rank left waiting in it cannot be released by this test's collectives
  MPI_Comm_dup(comm, &lc);
  auto section = [&](const char* name, auto&& body) {
    const int before = failures;
    try {
      body();
    } catch (const std::exception& e) {
      CHECK(false, "section %s: %s", name, e.what());
    }
    if (rank_of(comm) == 0) std::printf("section %s: %s\n", name, failures == before ? "ok" : "FAILED");
  };
  section("1 (names, default_chain)", [] { names_and_default_chains(); });
  section("2 (backward error by hand)", [&] { backward_error_by_hand(lc); });
  section("3 (invariance)", [&] { invariance(comm, lc); });
  section("4 (one-stage chains)", [&] { one_stage(comm, lc); });
  section("5 (forced outcomes)", [&] { forced_outcomes(comm, lc); });
  section("6 (RelResidual straddle)", [&] { straddle(comm, lc); });
  section("7 (Verdict)", [&] { verdict(comm, lc); });
  section("8 (sticky stage)", [&] { sticky(comm, lc); });
  section("9 (best iterate)", [&] { best_iterate(comm, lc); });
  section("10 (rank 0 decides)", [&] { rank0_decides(comm, lc); });
  section("11 (kernel probe)", [&] { kernel_probe(comm, lc); });
  section("12 (STRUMPACK matching)", [&] { strumpack_matching(comm, lc); });
  section("13 (every stage threw)", [&] { every_stage_threw(comm, lc); });
  section("14 (refusals)", [] { refusals(); });
  section("15 (best iterate, opposite orders)", [&] { best_iterate_opposite_orders(comm, lc); });
  section("16 (held stage's record)", [&] { held_stage_record(comm, lc); });
  section("17 (non-finite)", [&] { non_finite(comm, lc); });
  section("18 (row-block errors)", [&] { row_block_errors(comm, lc); });
  section("19 (size cap)", [&] { size_cap(comm, lc); });
  section("20 (NaN backward error in a tie)", [&] { nan_backward_error_tie(comm, lc); });
  section("21 (a size-capped tail keeps the held stage)", [&] { capped_tail_keeps_the_held_stage(comm, lc); });
  section("22 (the default chains walked)", [&] { default_chains_walked(comm, lc); });
  MPI_Comm_free(&lc);
  std::printf(failures ? "test_fluids_lu_chain: %d FAILURES\n" : "test_fluids_lu_chain: OK\n", failures);
  MPI_Finalize();
  return failures ? 1 : 0;
}
