// General (nonsymmetric, indefinite) sparse direct solvers for the fluids module.
//
// axmp::SymSolver is symmetric-only by design: it hands Eigen's CSC arrays to STRUMPACK as if they were CSR and
// uses LDL^T without pivoting. Neither is valid for Oseen/Newton Jacobians, nor safe for the indefinite Stokes
// saddle point. This file is the general path:
//
//   * CsrMatrix / DistCsr: TRUE compressed-row storage (row pointers, column indices and values of A, not of A^T).
//     DistCsr is a block-row distribution: each rank holds rows [row_begin, row_begin + local_rows) with GLOBAL
//     column indices, the layout of a merged mfem::HypreParMatrix and of STRUMPACK's MPIDist interface.
//   * SparseLU: one interface over three backends,
//       - EigenLU         Eigen::SparseLU with COLAMD ordering (serial; the robust baseline),
//       - StrumpackLU     STRUMPACK multifrontal LU, shared memory (OpenMP), METIS ordering, MC64 matching,
//       - StrumpackMPI    STRUMPACK distributed LU (StrumpackSparseSolverMPIDist), only in a build against an MPI
//                         STRUMPACK (AXMP_STRUMPACK_MPI).
//     The serial backends gather the matrix to rank 0, factor there, and scatter the solution; so every backend
//     accepts the same DistCsr on any number of ranks (rank invariance of the direct path).
//
// Accuracy contract. MC64 matching keeps the zero (2,2) block of a saddle point from
// producing a zero pivot, but it does NOT make the factorization accurate: STRUMPACK pivots only inside the fronts,
// and on Stokes matrices the plain LU solve can stop at a relative residual of 1e-9..1e-8. Every solve() therefore
// runs residual-checked iterative refinement with the caller's (true) matrix, up to `max_refine` steps, until
// ||A x - b|| <= refine_rtol ||b||, and THROWS (on every rank) when the target is missed.
//
// Input contract: within each row, column indices may come in any order but must be unique (duplicates are refused;
// call sum_duplicates() first), in range [0, n_global), with nondecreasing row pointers. A matrix with a known
// nullspace is not detected by the factorization (a singular saddle point is often "solved" with an arbitrary
// constant): pass the null vector with set_kernel_probe() and factorize() refuses the matrix when A z ~ 0.
//
// Errors are collective: a failure on any rank (e.g. a ZERO_PIVOT that STRUMPACK-MPI reports on one rank only) is
// reduced over the communicator and thrown on all ranks, so no rank is left waiting in a collective call.
//
// An internal API that may change; it is kept free of anything EM- or spec-specific.
//
// Index widths. The CSR arrays and the gather are int (nnz and n_global < 2^31). Eigen's SparseLU keeps its factor in
// arrays addressed by its StorageIndex, so the Eigen backend picks the width per factorization (eigen_lu_index_bits):
// the 32-bit instantiation when n <= 46 340 and nnz(A) <= 2^28, the 64-bit one above. The bound is proved, not
// estimated, from Eigen 3.4's SparseLU. A supernode has at most maxsuper = 128 columns (SparseLU.h:464, enforced in
// SparseLU_column_dfs.h:140), and one that starts at column fsupc has at most n - fsupc rows, because only the rows
// not yet used as pivots enter L. SparseLU_column_bmod.h:116-119 and :133-141 pad each lusup column to a multiple of
// the SIMD packet size, by at most 7 entries (a packet of 8 doubles). Column j therefore holds at most
// (n - j) + 127 + 7 entries, and every offset Eigen keeps (xlusup, xlsub, xusub; the used lengths of lusup, lsub,
// usub) is at most n (n + 1) / 2 + 134 n. At n = 46 340 that is 1 073 720 970 + 6 209 560 = 1 079 930 530 < 2^31 - 1 =
// 2 147 483 647. (Counting n entries per column instead would give n^2 + n = 2 147 441 940, but the padding makes
// that argument fail for wide packets.) COLAMD's workspace has Alen = 2 nnz + nnz/5 + O(n) entries, which
// nnz <= 2^28 keeps below 6e8. A fill law fitted to measured sizes cannot replace it. The 64-bit solver is fed a
// 64-bit copy of the gathered CSR.
// Above the bound the Eigen backend is limited by the memory (and time) of rank 0, where it runs alone: nothing is
// checked before factorizing, because Eigen allocates its arrays lazily and halves its first request when an
// allocation fails, so no number computed beforehand bounds what the fill will need. A factorization that runs out
// of memory in Eigen's own handling, or whose allocation throws on rank 0, is reported on every rank with the sizes
// and the remedies (STRUMPACK, or an iterative solver). Where the job's memory limit or the kernel's OOM killer ends
// the process first, nothing can be reported.
#pragma once

#include <mpi.h>

#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Sparse>

namespace mfem {
class Mesh;
}

namespace axmp::fluids {

// Sequential CSR. rowptr has n+1 entries; col/val have rowptr[n] entries. Column indices need not be sorted.
struct CsrMatrix {
  int n = 0;
  std::vector<int> rowptr, col;
  std::vector<double> val;
  long nnz() const { return rowptr.empty() ? 0 : rowptr.back(); }
  // y = A x
  void mult(const double* x, double* y) const;
  // From an Eigen matrix of either storage order: the result is always CSR of the SAME matrix.
  static CsrMatrix from_eigen(const Eigen::SparseMatrix<double, Eigen::ColMajor>& A);
  static CsrMatrix from_eigen(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A);
  Eigen::SparseMatrix<double, Eigen::ColMajor> to_eigen() const;
};

// Refuse a mesh that is partitioned over more than one MPI rank (a ParMesh on a communicator of size > 1): the serial
// one-rank operators (ScalarTransport, BoussinesqSolver) would otherwise solve one independent problem per partition.
// A plain mfem::Mesh is serial by construction and is accepted.
void require_single_rank(const mfem::Mesh& mesh, const char* what);

// Block-row distributed CSR with global column indices. On one rank it is a CsrMatrix with row_begin = 0.
struct DistCsr {
  MPI_Comm comm = MPI_COMM_WORLD;
  long n_global = 0;
  long row_begin = 0;
  int local_rows = 0;
  std::vector<int> rowptr, col;  // col: GLOBAL indices (int: hypre here is built with 32-bit HYPRE_BigInt)
  std::vector<double> val;
  // row offsets of all ranks (size nranks+1), collective
  std::vector<int> row_dist() const;
  // y_local = (A x)_local with x distributed like the rows; collective (gathers x)
  void mult(const double* x_local, double* y_local) const;
  // gather the whole matrix onto `root` (empty on the other ranks); collective
  CsrMatrix gather(int root = 0) const;
  static DistCsr from_serial(const CsrMatrix& A, MPI_Comm comm = MPI_COMM_SELF);
  // Block-row distribution of a serial matrix held identically on every rank of comm: rank p keeps rows
  // [dist[p], dist[p+1]) (dist of size nranks+1; empty ranks allowed). Collective.
  static DistCsr distribute(const CsrMatrix& A, const std::vector<int>& dist, MPI_Comm comm);
  // Sort the column indices of each row and sum duplicate entries (as Eigen's setFromTriplets does).
  void sum_duplicates();
  // Check the input contract; returns "" or a description of the first violation (local rows). Not collective.
  std::string contract_violation() const;
  // Frobenius norm over all ranks. Collective.
  double frobenius_norm() const;
  // y_local = (|A| |x|)_local (entrywise absolute values of A and of x), x distributed like the rows; collective
  void abs_mult(const double* x_local, double* y_local) const;
};

enum class LUBackend { EigenLU, StrumpackLU, StrumpackMPI };
const char* to_string(LUBackend b);
// "eigen" | "strumpack" | "strumpack-mpi"; throws axmp::Error with the accepted names otherwise
LUBackend lu_backend_from_string(const std::string& s);
bool lu_backend_available(LUBackend b);

// The MC64 job of a STRUMPACK stage (ignored by the Eigen backend).
enum class MatchingJob { None, MaxSmallestDiagonal /* MC64 job 2 */, MaxDiagonalProductScaling /* MC64 job 5 */ };

// One stage of the retry chain: a full configuration of one factorization.
struct LUStage {
  LUBackend backend = LUBackend::EigenLU;
  MatchingJob matching = MatchingJob::MaxDiagonalProductScaling;  // STRUMPACK backends only
  bool equilibrate = false;  // two-sided scaling before the backend: not implemented (a chain with it is refused)
  // A size cap: when > 0 and the matrix has more than max_n unknowns, the stage is not factorized and is recorded
  // with outcome "refused" and a reason that starts with "size" ("size: n = 312345 > 300000"). 0: no cap.
  // default_chain caps its Eigen retry stage (kLUEigenRetryMaxN, kLUEigenRetryMaxN3D); name() does not show it.
  long max_n = 0;
  // The one spelling used in every record: the backend name, then ":none" | ":mc64-2" | ":mc64-5" for the STRUMPACK
  // backends (nothing for Eigen), then ":eq" when equilibrate is set: "eigen", "eigen:eq", "strumpack:mc64-5",
  // "strumpack:none", "strumpack:mc64-2", "strumpack-mpi:none", "strumpack:mc64-5:eq".
  std::string name() const;
};

// The chain every caller's first backend gets, written once:
//   first EigenLU (any matching)                          -> [eigen]
//   first StrumpackLU/StrumpackMPI, first_matching None   -> [first:none, first:mc64-5, eigen]  (3D lists)
//   first StrumpackLU/StrumpackMPI, first_matching mc64-5 -> [first:mc64-5, first:mc64-2, eigen]  (2D lists; 3D Stokes with
//                                                           the matching first is [mc64-5, none, eigen], see below)
//   first StrumpackLU/StrumpackMPI, first_matching mc64-2 -> [first:mc64-2, first:mc64-5, eigen]
// No configuration is listed twice (a second factorization of the same configuration returns the same iterate), and the
// 2D lists have no stage without matching. Measured with STRUMPACK (shared memory) on 2D systems that the first stage
// (MC64 job 5) missed on its caller's gate: job 2 accepted 34 of 34 on an AMD EPYC 7662 (28 bordered projections with
// the mean-value multiplier and the 6 planar k = 5 Womersley starters, n = 74 691, at 0.31 s to 0.33 s per factorization where
// Eigen's LU takes 4.0 s to 4.3 s) and 30 of 30 projections on an Intel Xeon Gold 6330; the stage without matching
// rescued nothing that job 2 did not, and threw ZERO_PIVOT on all six starters and on 8 of 10 Newton steps sampled on
// the lid-driven cavity. In 3D, job 2 costs what job 5 costs and rescued the same 35 retries of the quick cases, so the
// 3D list keeps job 5, and StokesSolver::lu_chain gives a 3D system that starts with the matching the retry measured on
// 3D saddle points (the stage without matching), because job 2 after a job-5 miss was measured on 2D systems only.
// The Eigen stage after a STRUMPACK stage is capped (LUStage::max_n): kLUEigenRetryMaxN,
// kLUEigenRetryMaxN3D for the 3D lists. A first Eigen stage (the caller chose Eigen) is not capped. A caller that
// knows the dimension of its system sets the cap by it (StokesSolver::lu_chain). Two things are in no default list
// because no system measured needed them: an equilibrated stage (each of the 54 projection cases has a stage that
// passes on both processors), and a shortcut that accepts an iterate on its backward error (no iterate that missed
// its caller's measure had a backward error of 1e-13 or less).
// Availability is not checked here (SparseLU checks it). While the test-only override is set, it returns the
// override for every argument.
std::vector<LUStage> default_chain(LUBackend first, MatchingJob first_matching);
// The chain of a caller's options: `chain` when it is not empty, else default_chain(first, first_matching).
inline std::vector<LUStage> chain_or_default(const std::vector<LUStage>& chain, LUBackend first,
                                             MatchingJob first_matching) {
  return chain.empty() ? default_chain(first, first_matching) : chain;
}
// The size caps of default_chain's Eigen retry stage (LUStage::max_n), which keep a failing solve from turning into
// hours of factorization: 2D lists (first matching not None) and 3D lists (first matching None). The 3D cap: one
// Eigen factorization of a 3D Taylor-Hood Stokes system took 642 s at n = 49 073 and did not finish in 50 minutes at
// n = 112 725 (the fill per row grows like n^0.6 there). The 2D cap is the fluids policy's "Eigen retries for 2D
// sizes only" read as a number.
inline constexpr long kLUEigenRetryMaxN = 300000;
inline constexpr long kLUEigenRetryMaxN3D = 50000;

struct LUStats {
  std::string backend;
  long n = 0, nnz = 0;
  long factor_nnz = -1;            // fill (nnz(L)+nnz(U)); -1 if the backend does not report it
  double factor_mem_bytes = -1;    // STRUMPACK's own estimate (factors only); -1 for Eigen
  double analyze_seconds = 0, factor_seconds = 0, solve_seconds = 0;
  double rel_residual_before = -1; // ||A x0 - b|| / ||b|| of the plain LU solve (before refinement)
  double rel_residual = -1;        // after refinement: what solve() guarantees (<= refine_rtol)
  int refine_steps = 0;
  int krylov_iterations = 0;       // GMRES iterations after a stagnated refinement (0 when not needed)
  bool krylov_fallback = false;    // the GMRES fallback ran (the refinement stagnated above the gated target)
  bool matching = true;            // MC64 matching/scaling was used by the factorization behind these stats
  double kernel_probe = -1;        // ||A z|| / (||A||_F ||z||) of the probe vector (set_kernel_probe), -1 if none
  int threads = 1, ranks = 1;
  // Eigen backend: the StorageIndex width (32 or 64) of the Eigen solver object that factorized; 0 otherwise.
  int index_bits = 0;
  // Eigen backend: Eigen's initial estimate of the factor, eigen_lu_factor_nnz_estimate(n, nnz); -1 otherwise.
  long factor_nnz_estimate = -1;
  // The stage these stats describe (LUStage::name(): the accepted stage of the last solve, else the one whose iterate
  // was returned, else the factorized one), its acceptance measure, its blockwise backward error
  // (lu_backward_error), and the number of attempts of the last solve (attempts().size()).
  std::string stage;
  double measure = -1;
  double backward_error = -1;
  int attempts = 0;
};

// The record of one stage tried by one solve. Every field holds rank 0's value on every rank.
struct LUAttempt {
  std::string stage;    // LUStage::name()
  std::string outcome;  // "accepted" | "threw" | "missed_target" | "refused" | "not_available"
  // accepted: "<measure name> <measure> <= <bound>" (Kind::None: "no acceptance test"); missed_target:
  // "<measure name> <measure> > <bound>", or "non-finite solution"; threw: the exception's what(); refused or
  // missed_target forced by the test hook: "test hook"; not_available: "<backend> is not in this build". Numbers %.3g.
  std::string reason;
  double measure = -1;  // the acceptance measure and its bound; -1 when the stage produced no iterate (bound: also
  double bound = -1;    // -1 for Kind::None)
  double rel_residual_before = -1, rel_residual = -1;
  double gate_residual = -1;   // the residual over set_residual_rows' rows relative to b over them; -1 without a mask
  double backward_error = -1;  // lu_backward_error of the stage's iterate with the declared row blocks
  std::vector<double> constraint_ratios;  // one per constraint row (block id -2), in global row order
  int refine_steps = 0, krylov_iterations = 0;
  long factor_nnz = -1;
  int index_bits = 0;  // Eigen: 32 or 64; 0 otherwise
  double analyze_seconds = 0, factor_seconds = 0, solve_seconds = 0;
};

// What the direct solves of one call (or, summed, of a run) needed, from their attempts: solves; the solves with
// more than one attempt; whether the last solve of the call was one of them (0 or 1; summed over a run); every
// attempt; the attempts with outcome "threw", "missed_target" or "refused"; the solves that no stage accepted; the
// count of solves per accepting stage name; the analyze, factor and solve seconds of the rejected attempts.
struct LUCounts {
  long solves = 0, escalated = 0, escalated_final = 0, attempts = 0, rejected = 0, failed = 0;
  std::map<std::string, long> accepted_by;
  double rejected_seconds = 0;
};
// Books one solve into c: its attempts (SparseLU::attempts() after solve(), or after a factorize() that threw, when
// no solve followed). escalated_final is set to whether this solve escalated, so after a call it describes the call's
// last solve.
void lu_count_solve(LUCounts& c, const std::vector<LUAttempt>& attempts);
// Adds b to a field by field (escalated_final too: summed over a run).
LUCounts& operator+=(LUCounts& a, const LUCounts& b);

// When a stage's iterate is accepted.
struct LUAcceptance {
  // None: accept every stage that produced an iterate (a single attempt). RelResidual: the refined relative
  // residual over all rows (stats().rel_residual) at most `bound`. Verdict: `verdict(x_local)` at most `bound`.
  // Accepted also requires a finite iterate and a finite measure.
  enum class Kind { None, RelResidual, Verdict } kind = Kind::None;
  double bound = 0;
  // Kind::Verdict: a collective function of the stage's iterate (local rows), called once per iterate on every rank;
  // the value rank 0 returns decides and is recorded.
  std::function<double(const std::vector<double>&)> verdict;
  // The measure's name in the reasons; empty: "relative residual" (RelResidual), "verdict" (Verdict).
  std::string name;
  // Stop escalating when a stage misses but its backward error is at most this: not implemented (must stay 0).
  double stop_on_backward_error = 0;
};

// The blockwise backward error of x as a solution of A x = b. x, b and row_blocks are the local rows; row_blocks holds
// a block id per row (>= 0: a block; -1: left out, e.g. Dirichlet identity rows; -2: a constraint row, left out of
// the maximum); empty: one block of all rows. For every block k >= 0 whose denominator is nonzero,
//   ||(b - A x)_k||_2 / || (|A| |x| + |b|)_k ||_2,
// and the maximum over them (0 when no block counts). A block whose ratio is not a number (its sums of squares
// overflow, inf / inf, or hold a NaN) makes the backward error NaN: such an iterate has none, and it must not read as
// the best possible value. For every constraint row j, in global row order, constraint_ratios (when not null)
// receives |(b - A x)_j| / (|A| |x| + |b|)_j (0 when that denominator is 0). It is invariant under scaling the rows
// of a block by one factor. Collective; the same value on every rank.
double lu_backward_error(const DistCsr& A, const std::vector<double>& x_local, const std::vector<double>& b_local,
                         const std::vector<int>& row_blocks_local, std::vector<double>* constraint_ratios = nullptr);

// The index width of the Eigen backend's factorization (the proof is in the header comment): bounds on n and nnz(A)
// under which the 32-bit instantiation is proved to be enough. The n bound is conservative: the proof holds up to
// n = 65 401.
// Measured on 50 matrices (34 first factorizations of the published 2D cases at each level above 46 340 unknowns, 14
// full-system matrices of the cases whose first factorization is a smaller block, and the 3D Stokes systems at
// N = 8 and 12): the 64-bit instantiation gives bitwise identical solutions (same COLAMD order, same nnz(L) and
// nnz(U)) and a peak resident memory 17.6 % to 26.9 % higher (the 25 matrices whose 32-bit peak is above 0.5 GiB;
// the repeat on an AMD EPYC 7401 node). The factor time differed by -5.5 %
// to +4.1 % (24 matrices with a 32-bit factor time above 5 s; one process at a time on an AMD EPYC 7401 node): no cost.
inline constexpr long kEigenLUIndex32MaxN = 46340;        // within the proof's bound (header comment)
inline constexpr long kEigenLUIndex32MaxNnz = 268435456;  // 2^28
// 32 or 64: the StorageIndex width for n unknowns and nnz(A) entries, 32 iff n <= max_n32 and
// nnz <= kEigenLUIndex32MaxNnz. The two-argument form applies kEigenLUIndex32MaxN (never the test bound).
int eigen_lu_index_bits(long n, long nnz);
int eigen_lu_index_bits(long n, long nnz, long max_n32);
// Eigen's initial estimate of each of L and U (SparseLU's memInit, fillfactor 20): min(20 (nnz + 1) / n, n) n in
// integer arithmetic; 0 for n <= 0.
long eigen_lu_factor_nnz_estimate(long n, long nnz);
// The failure test of an Eigen factorization: a non-empty lastErrorMessage() (memInit's failure sets only the
// message), or info() != Success.
bool eigen_lu_failed(Eigen::ComputationInfo info, const std::string& message);

// Test-only hooks, process-wide; a test resets what it sets.
// The n bound of the 32-bit instantiation that SparseLU::factorize applies, in place of kEigenLUIndex32MaxN;
// max_n <= 0 restores the constant.
void set_eigen_lu_max_n32_for_testing(long max_n);
// Makes rank 0's Eigen work in every following SparseLU::factorize fail, until reset with None:
//   ExpandMemory: the failure test is fed (Eigen::NumericalIssue, "UNABLE TO EXPAND MEMORY IN COLUMN_BMOD() ")
//                 in place of the solver's own pair;
//   MemInit:      the failure test is fed (Eigen::Success, "UNABLE TO ALLOCATE WORKING MEMORY\n\n");
//   BadAlloc:     std::bad_alloc is thrown in rank 0's Eigen work before the solver factorizes.
enum class EigenLUTestFailure { None, ExpandMemory, MemInit, BadAlloc };
void set_eigen_lu_failure_for_testing(EigenLUTestFailure mode);
// The next `count` attempts of chain index 0 (in any SparseLU of this process) take `outcome`, reason "test hook":
//   Threw:        that stage's factorization throws axmp::Error("test hook ...") on every rank (outcome "threw");
//   Refused:      that stage is not factorized (outcome "refused");
//   MissedTarget: that stage factorizes and solves, its numbers are recorded, and it is rejected ("missed_target").
// None or count <= 0 clears it.
enum class LUForcedOutcome { None, Threw, MissedTarget, Refused };
void set_lu_forced_outcome_for_testing(LUForcedOutcome outcome, int count);
// eps != 0: the iterate of every attempt of chain index 0 is multiplied by (1 + eps) entrywise after the stage's
// refinement and Krylov finisher; every number recorded for that attempt, its acceptance measure and the returned
// iterate are those of the perturbed iterate (rel_residual_before stays the plain solve's). 0 clears it.
void set_lu_solution_perturbation_for_testing(double eps);
// A function applied to the iterate (local rows) of every attempt, with the attempt's chain index, after the stage's
// refinement and Krylov finisher and after the perturbation above: a perturbation that need not be uniform, of any
// stage. As above, every number recorded for the attempt, its acceptance measure and the returned iterate are those
// of the changed iterate. The function is called on every rank and must not be collective. Empty clears it.
void set_lu_iterate_hook_for_testing(std::function<void(int chain_index, std::vector<double>& x_local)> hook);
// On rank 1 only: the (finite, measure) pair that decides acceptance is replaced, after the stage's solve, by one on
// the other side of the bound: (false, NaN) when it would be accepted, (true, 0) when it would be rejected (kinds
// RelResidual and Verdict). Rank 0 decides, so a correct chain is unaffected.
void set_lu_rank1_acceptance_flip_for_testing(bool on);
// While non-empty, default_chain returns this chain for every argument; empty clears it.
void set_default_lu_chain_for_testing(std::vector<LUStage> chain);
// Backend solver objects (Eigen 32/64-bit, STRUMPACK, STRUMPACK-MPI) alive on this rank, in every SparseLU of the
// process, and the largest number alive at once since the last reset (counted when an object is constructed).
struct LUBackendObjectCount {
  int live = 0;
  int peak = 0;
};
LUBackendObjectCount lu_backend_objects_for_testing();
void reset_lu_backend_object_peak_for_testing();  // peak := live

// The accuracy gate of every dual-norm mass solve (the dual norm ||R||_* = sqrt(R^T M^-1 R) is what each fluids
// convergence verdict is expressed in). A mass solve of relative residual eps returns y = M^-1 (b + e) with
// ||e|| <= eps ||b||, so |b . M^-1 e| <= eps sqrt(kappa(M)) ||b||_{M^-1}^2: the relative error of ||R||_*^2 is at most
// eps sqrt(kappa(M)), and of ||R||_* at most half that. 1e-10 keeps the dual norm's relative error below 1e-3 for any
// mass matrix with kappa(M) <= 1e14, which no usable consistent (exactly integrated, SPD) mass matrix exceeds. The
// solves are refined to 1e-15 with refinement to stagnation, which drives the RESIDUAL (not the error) to round-off
// whatever the conditioning (measured 3e-15 on a mass-like matrix whose smallest diagonal was scanned down to 1e-24),
// so the gate cannot fire on a healthy problem. (The P1/P1 GLS solver holds its unrefined sparse Cholesky mass solves
// to the same bound.) It catches inaccuracy, not singularity: a numerically singular mass
// matrix is solved to round-off residual with a different answer, which is the job of SparseLU::set_kernel_probe.
inline constexpr double kDualMassResidualMax = 1e-10;

// The accuracy the FINAL direct solve of a nonlinear solve must reach, at least: max(newton_rtol, 1e-11) unless
// NonlinearOptions::final_solve_residual_max sets it. It is not tied to newton_rtol alone because the two measure
// different things: newton_rtol bounds the NONLINEAR residual (recomputed from the assembled operators after every
// step, so an inexact linear solve cannot make it pass falsely), whereas the linear residual of a step is limited by
// the conditioning of that step's matrix. On the published 132 098-velocity-dof solves the final step reaches 4.3e-13
// (planar manufactured solution, newton_rtol 1e-12) and 1.4e-13 (symmetric-gradient form at newton_rtol = 1e-13, whose
// nonlinear residual is 7e-15), and an intermediate step 1.0e-12: a bound at newton_rtol refuses the second. 1e-11 is
// a factor 23 above the worst final step measured and still refuses a solve that is no solution of its system (the
// time-step floor is 1e-6, the refinement targets are at most 1e-12).
// Measured with STRUMPACK (AMD EPYC 7662) on the published cases in quick mode (81 cases) and on one full-level case per
// caller, no solve of a problem with a solution is refused by the three bounds that follow from it, and the largest
// measure over bound is 0.068 (Boussinesq), 0.0037 (GLS) and 0.0014 (scalar transport Jacobian), all on the full-level
// cases.
inline constexpr double kFinalSolveResidualMax = 1e-11;

// The gate on the initial Leray projection (NavierStokesSolver::project_divergence_free), which solves
// [M B^T; B 0] (du, dp) = -(R_u, R_p) for the correction du of the interpolant u0. What is gated is a backward error:
// for each block, the Euclidean norm of the residual over the Euclidean norm of the sizes of the terms of that block,
//   momentum     || M du + B^T dp + R_u ||   /   || |M| |du| + |B^T| |dp| + |R_u| ||
//   continuity   || B (u0 + du) - rhs_p ||   /   || |B| (|u0| + |du|) + |rhs_p| ||
// (|.| entrywise), and the larger of the two. It is the residual in units of the round-off of evaluating it, so an
// accurate solve gives a number near the unit round-off whatever the mesh, the shape of the domain and the length
// unit are. A residual relative to the data does not have that property, for two reasons that were both measured:
//   - the momentum rows carry L^2 and the continuity rows L (L the length unit), so one norm over both blocks weighs
//     them by the unit: the same 1000:1 channel gave 1.4e-15 as 1e-3 x 1e-6 and 1.0e-12 as 1e5 x 100;
//   - in a channel of aspect ratio a the multiplier is of size a (in units of the transverse scale) while its
//     transverse gradient is of size one, so B^T dp cancels to one part in a n across the channel (n cells), and the
//     momentum residual relative to ||M du|| is a n times the round-off: 9.9e-14, 9.7e-13, 1.1e-11 and 1.0e-10 at
//     a = 1e2, 1e3, 1e4, 1e5 (n = 32), where the measure above stays at 5.2e-17 to 5.7e-17.
// The sum is over each block, not a largest ratio over the rows (the componentwise backward error), because a row in
// a region where the field vanishes has terms at round-off and a ratio of order one: a field confined to one corner
// of a 1000:1 channel gave a largest row ratio of 1.1e-7 for a projection whose block measure is 6.0e-17.
//
// The bound does not depend on newton_rtol, because the measure does not: the solve is refined until it stagnates
// whatever the target is. Measured with Eigen LU (AMD EPYC 7452; an initial field that is not divergence free on
// N x N right-triangle meshes; newton_rtol 1e-6 to 1e-14; docs/physics/fluids.md has the tables): 4.2e-17 to 6.0e-17
// over the unit square at N = 8 to 128 (578 to 132 098 velocity dofs), channels of aspect ratio 10 to 1e5, squares of
// side 1e-6 to 1e6, graded and 100:1-cell meshes, P2 and P3 velocities, planar and axisymmetric, on 1 and 2 ranks,
// and 6.9e-17 for a field that is already divergence free on the 8 x 8 mesh; 3.8e-17 to 6.9e-17 on an Intel Xeon
// Gold 6330 (the same 6.9e-17 at N = 8). 1e-13 is a factor 1400 above the largest of these. The margin
// belongs to Eigen LU and to lengths between 1e-6 and 1e6:
//   - with lengths of 1e7 and more Eigen LU loses accuracy too (2.3e-14 on 1e7 x 100, accepted; 1e9 x 1e4 refused);
//   - STRUMPACK's first configuration (MC64 job 5, with the mean-value multiplier) returns measurably inaccurate
//     projections on some scaled or stretched domains, and which ones depends on the processor: accepted at 1.4e-14
//     (square of side 100, N = 64, EPYC 7452) and 7.1e-14 (1e4 x 100, N = 16, Xeon Gold 6330), and missed at 0.37
//     (1e5 x 1, N = 32, Xeon), 2.1e-9 (10 x 1e-3, N = 16, Xeon) and 0.060 (1e7 x 1e4, N = 32, Xeon), where
//     STRUMPACK with job 2 gives 5.3e-17, 5.5e-17 and 5.8e-17. The gate refuses those, and the chain walks on to job 2
//     (accepted on every case measured, both processors);
//   - Eigen's LU with the mean-value multiplier is the one that is refused on 1e6 x 1e4 (4.4e-12, N = 16, Xeon):
//     STRUMPACK's first configuration with the multiplier passes there (5.3e-17 EPYC, 5.4e-17 Xeon), and with the
//     pinned pressure both backends give about 5e-17. If no configuration passes, the error of the next step names the
//     remedy: the pinned pressure (nullspace="pin" in Python), which is not accurate everywhere either.
// What the bound refuses. A solve whose solution is wrong by a relative eps leaves the continuity residual
// eps (B u0 - rhs_p) over terms that are N times larger than it because B differentiates: 0.68 eps / N on the squares
// and 0.024 eps to 0.028 eps on the channels at N = 32, the same in every length unit. So eps = 1e-8 leaves 2.1e-11
// to 1.1e-9 on these cases, at least a factor 200 above the bound (the factor falls like 1/N: 34 at N = 192 with
// nonzero boundary data), and the bound refuses relative errors above about 1.5e-13 N. The measure is a backward
// error, not an error of the state, so an error confined to the momentum block of a channel of aspect ratio a is
// seen as about eps / (a n), by the cancellation described above: on 1e4 x 1 and 1e5 x 1 (N = 32) a divergence-free
// error that leaves the state wrong by 1.9e-8 of its largest velocity measures 3.5e-14 and 3.5e-15, and a multiplier
// scaled by 1 + 1e-8 measures 5.3e-14 and 5.3e-15, all accepted.
inline constexpr double kProjectionResidualMax = 1e-13;

// A SparseLU holds an ordered chain of stages (LUStage). factorize() factorizes the first stage that factorizes
// (stages that throw, are refused or are not in this build are recorded and passed over); solve() solves with the
// current stage and, when the iterate is not accepted (LUAcceptance), releases that stage's backend object, factorizes
// the next stage and solves again, until a stage is accepted or the chain ends. Then the accepted iterate is returned,
// else the one with the smallest acceptance measure (ties: the smaller backward error). In that choice an iterate
// that is not finite on every rank, or whose measure is NaN, counts as measure +infinity, and a NaN backward error as
// +infinity: a non-finite iterate is returned only when no stage produced a finite one. The accepted stage stays the
// current stage for the next solve() on the same factorization (a later miss walks on from it); factorize() starts
// again at stage 0. When no stage was accepted, stats() describes the stage whose iterate was returned, while the
// factorization held is the last stage's: the next solve() starts there, and its record and stats() are that
// stage's. At most one backend object is alive per SparseLU: the previous stage's is destroyed before the next stage
// factorizes. Every accept/escalate decision is rank 0's, broadcast; every rank enters every stage. A stage whose
// size cap (LUStage::max_n) the matrix exceeds is recorded as "refused" and passed over without being factorized.
// SparseLU(LUBackend) is the one-stage chain {backend, MC64 job 5} with Kind::None: a single attempt, accepted
// whatever its residual (set_throw_on_miss decides what a miss of the refinement target does).
//
// When no stage is accepted and the last stage tried could not be factorized, no factorization is held any more (the
// stage before it was released to make room): the best iterate is still returned, and the next solve() needs a new
// factorize(). A chain whose last stage solved keeps that stage for the next solve().
//
// What the record costs. Every solve, of a one-stage chain too, evaluates A x and |A| |x| in one more pass over the
// matrix and makes six small collective calls (the row distribution, the gather of x, two reductions, the two
// broadcasts of the decision and of the record). The solution is not touched. The pass costs about what one
// refinement step's residual costs; it is noticeable only next to the triangular solves of a small matrix.
//
// What rank 0 decides, and what it does not. The accept/escalate decision of each stage and the doubles of its record
// are rank 0's, broadcast, so the ranks cannot leave a stage on different sides. INSIDE a stage, the refinement loop
// and the GMRES finisher branch on MPI_Allreduce'd doubles (and GMRES builds its iterate from them): they rely on
// MPI_Allreduce returning the same bits on every rank, which the chain does not change.
//
// Why the previous stage's backend object is destroyed before the next stage factorizes: a rejected factor's memory
// is freed before the next factor is computed, and a STRUMPACK solver's constructor installs a std::new_handler that
// aborts the process on an allocation failure, put back only by its destructor. The shared-memory backend's guard
// restores the handler right after construction; a live STRUMPACK-MPI solver keeps it installed, so a later Eigen
// stage's allocation failure would abort instead of being reported. The same holds for a long-lived SparseLU of the
// STRUMPACK-MPI backend that another SparseLU's Eigen stage runs beside (a caller's mass factorization, kept for the
// whole run): that is the caller's to know, the chain cannot release an object it does not own.
class SparseLU {
 public:
  explicit SparseLU(LUBackend backend);
  // Throws axmp::Error for an empty chain (the message says "empty"), a stage with equilibrate set ("equilibration"),
  // or a first stage whose backend is not in this build ("not in this build"); a later stage whose backend is not in
  // this build is recorded as "not_available" when the walk reaches it.
  explicit SparseLU(std::vector<LUStage> chain);
  ~SparseLU();
  SparseLU(const SparseLU&) = delete;
  SparseLU& operator=(const SparseLU&) = delete;

  // Analyze + factor. Collective over A.comm. Throws axmp::Error on a singular/failed factorization: for a one-stage
  // chain the backend's own error; for a longer chain when no stage factorizes, "every stage of the sparse LU chain failed: "
  // followed by "<stage>: <reason>" for every attempt, separated by "; ".
  // A is NOT copied: it must stay alive and unchanged until the last solve() (used for the residual check and for
  // the factorizations of later stages).
  void factorize(const DistCsr& A);
  // Solve A x = b for the local block rows of b; returns the local block of x. Collective.
  std::vector<double> solve(const std::vector<double>& b_local);
  const LUStats& stats() const { return stats_; }
  LUBackend backend() const { return backend_; }  // the first stage's backend
  const std::vector<LUStage>& chain() const { return chain_; }
  // The acceptance test of every stage (before solve; default Kind::None). Throws axmp::Error for a Verdict without
  // a function (the message says "verdict function"), a negative or NaN bound, or a nonzero stop_on_backward_error
  // (the message names "stop_on_backward_error").
  void set_acceptance(LUAcceptance acceptance);
  // Block ids of the local rows (see lu_backward_error): used for the recorded backward error and constraint ratios
  // and, when non-empty, for the blockwise kernel probe. factorize() reads them: their size must be A.local_rows then,
  // and a change takes effect at the next factorize(). A wrong size or an id below -2 on any rank is an axmp::Error
  // on every rank (of factorize(); lu_backward_error checks its own argument the same way).
  void set_row_blocks(std::vector<int> blocks_local);
  // The attempts of the last solve(), in order (the first solve after factorize() also lists the stages factorize()
  // passed over); after a factorize() without a solve, the stages it passed over.
  const std::vector<LUAttempt>& attempts() const { return attempts_; }
  // The chain index of the stage the last solve() accepted; -1 when none was accepted or before a solve.
  int accepted_stage() const { return accepted_stage_; }
  // The chain index of the stage whose iterate the last solve() returned: the accepted one, else the best one; -1
  // before a solve.
  int solution_stage() const { return solution_stage_; }
  // Test-only: the MC64 job (0 none, 2, 5: the value of strumpack::MatchingJob) in the options of the STRUMPACK
  // solver object this SparseLU holds; -1 when it holds none (other ranks for the serial backend, other backends).
  int strumpack_matching_job_for_testing() const;
  // Test-only: the StorageIndex width (32 or 64) of the Eigen solver object this SparseLU holds (rank 0, after a
  // successful factorize with the Eigen backend); 0 when it holds none (other ranks, other backends).
  int live_eigen_index_bits_for_testing() const;

  // STRUMPACK options (before factorize). MC64 matching is on by default: without it the zero (2,2) block of a
  // saddle point can give ZERO_PIVOT. Accuracy comes from the refinement in solve(), not from the matching. (On 3D
  // Taylor-Hood saddle points the matching DAMAGES the factorization -- a plain residual of 0.14 against 5e-16 without
  // it was observed -- so the 3D Stokes path runs without it and retries with it, see StokesOptions::lu_matching.)
  // It sets the first stage's MC64 job: job 5 (on) or none (off).
  void set_matching(bool on) {
    chain_[0].matching = on ? MatchingJob::MaxDiagonalProductScaling : MatchingJob::None;
  }
  // Iterative refinement: at most max_steps corrections with the caller's matrix, target ||r|| <= rtol ||b||.
  // max_steps = 0 disables it (the plain LU solve; for measurements only). solve() throws when rtol is missed.
  void set_refinement(int max_steps, double rtol) {
    max_refine_ = max_steps;
    refine_rtol_ = rtol;
  }
  // Throw when the refinement target is missed (default true). false: return the best iterate and report
  // stats().rel_residual; the caller's own acceptance test decides (the Stokes solver does).
  void set_throw_on_miss(bool on) { throw_on_miss_ = on; }
  // Refine until a step gains less than 10 % (or max_steps), even below the target: the best attainable accuracy
  // (the Stokes path uses it; small velocities next to O(1) pressures need it, e.g. the exact no-flow state)
  void set_refine_to_stagnation(bool on) { to_stagnation_ = on; }
  // When the refinement stagnates above its target, restarted GMRES preconditioned by the LU (opt-in)
  void set_krylov_fallback(bool on) { krylov_fallback_ = on; }
  // Rows (local; true = counted) of the residual the Krylov fallback stops on: the caller's acceptance rows. The
  // Stokes solver passes its non-Dirichlet rows. Empty: all rows.
  void set_residual_rows(std::vector<char> mask) { row_mask_ = std::move(mask); }
  // OpenMP threads for factorize() and solve() (0: leave the global setting). 1 is deterministic for STRUMPACK.
  void set_threads(int n) { threads_ = n; }
  // At least this many refinement steps even when the target is met (a "tightened" solve; capped by max_steps).
  void set_min_refinement(int steps) { min_refine_ = steps; }
  // A vector z (local block rows) with A z = 0 when a nullspace was NOT removed (e.g. the constant pressure).
  // factorize() refuses A (before any stage) if ||A z|| <= tol ||A||_F ||z||; when row blocks are declared
  // (set_row_blocks, non-empty), blockwise instead: refused when the maximum over the blocks k >= 0 and over every
  // constraint row (id -2, a block of its own) of ||(A z)_k|| / || (|A| |z|)_k || is at most tol, a block whose
  // denominator is 0 being left out (rows with id -1 are left out). stats().kernel_probe is the ratio tested.
  void set_kernel_probe(std::vector<double> z_local, double tol = 1e-12) {
    probe_ = std::move(z_local);
    probe_tol_ = tol;
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  LUBackend backend_;  // the first stage's
  int max_refine_ = 3;
  int min_refine_ = 0;
  bool throw_on_miss_ = true;
  bool to_stagnation_ = false;
  bool krylov_fallback_ = false;
  std::vector<char> row_mask_;
  int threads_ = 0;
  struct ThreadGuard {
    explicit ThreadGuard(int n);
    ~ThreadGuard();
    int saved_;
    bool active_;
  };
  double refine_rtol_ = 1e-12;
  std::vector<double> probe_;
  double probe_tol_ = 1e-12;
  LUStats stats_;
  LUStats matrix_stats_;  // what stats_ says of the matrix (sizes, threads, ranks, kernel probe): every stage starts from it
  LUStats stage_stats_;   // stats_ as the factorization of the held stage left it: every solve with that stage starts from it
  std::vector<LUStage> chain_;
  LUAcceptance acceptance_;
  std::vector<int> row_blocks_;  // as set; held_blocks_ is the copy factorize() checked, which the solves use
  std::vector<int> held_blocks_;
  std::vector<LUAttempt> attempts_;
  bool attempts_from_factorize_ = false;  // attempts_ holds the stages factorize() passed over: the next solve keeps them
  int accepted_stage_ = -1;
  int solution_stage_ = -1;
  int current_ = -1;  // the chain index of the stage whose factorization is held; -1: none
  // Destroy the backend object held (any backend). Collective for the STRUMPACK-MPI backend.
  void release_backend();
  // Factorize stage k of the chain (the previous backend object is released first); stats_ then describes that stage.
  // Throws axmp::Error on every rank when the backend fails.
  void factorize_stage(const DistCsr& A, int k);
  // Try stage k: true when it factorized; otherwise the attempt is recorded (not_available, refused, threw) and
  // *error, when given, receives the exception of a stage that threw. Collective.
  bool try_stage(const DistCsr& A, int k, std::exception_ptr* error);
  // One backend solve with the current stage, refined (and finished by GMRES when that is on): the whole solve of one
  // factorization. Sets the solve's part of stats_. Collective.
  std::vector<double> solve_stage(const std::vector<double>& b_local);
  std::vector<double> raw_solve(const std::vector<double>& b_local);  // one backend solve, collective
  const DistCsr* A_ = nullptr;  // the caller's matrix, for the residual check and the factorizations of later stages
};

}  // namespace axmp::fluids
