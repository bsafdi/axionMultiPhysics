// General sparse LU for the fluids module: see cpp/include/axionmultiphysics/fluids/sparse_lu.hpp.
#include "axionmultiphysics/fluids/sparse_lu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <numeric>
#include <type_traits>

#include <Eigen/SparseLU>

#include <mfem.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/omp_shim.hpp"
#include "axionmultiphysics/sym_solver.hpp"

#ifdef AXMP_STRUMPACK
#include <StrumpackSparseSolver.hpp>
#ifdef AXMP_STRUMPACK_MPI
#include <StrumpackSparseSolverMPIDist.hpp>
#endif
#endif

namespace axmp::fluids {

namespace {
int rank_of(MPI_Comm c) {
  int r = 0;
  MPI_Comm_rank(c, &r);
  return r;
}
int size_of(MPI_Comm c) {
  int s = 1;
  MPI_Comm_size(c, &s);
  return s;
}
double wtime() { return MPI_Wtime(); }
}  // namespace

// ------------------------------------------------------------------------------------------------ CsrMatrix
void CsrMatrix::mult(const double* x, double* y) const {
  for (int i = 0; i < n; ++i) {
    double s = 0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) s += val[k] * x[col[k]];
    y[i] = s;
  }
}

CsrMatrix CsrMatrix::from_eigen(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A0) {
  AXMP_REQUIRE(A0.rows() == A0.cols(), "CsrMatrix: the matrix must be square");
  Eigen::SparseMatrix<double, Eigen::RowMajor> A = A0;
  A.makeCompressed();
  CsrMatrix c;
  c.n = static_cast<int>(A.rows());
  c.rowptr.assign(A.outerIndexPtr(), A.outerIndexPtr() + c.n + 1);
  c.col.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
  c.val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());
  return c;
}

CsrMatrix CsrMatrix::from_eigen(const Eigen::SparseMatrix<double, Eigen::ColMajor>& A) {
  // the storage-order conversion transposes the arrays, not the matrix: this is the step SymSolver skips
  return from_eigen(Eigen::SparseMatrix<double, Eigen::RowMajor>(A));
}

Eigen::SparseMatrix<double, Eigen::ColMajor> CsrMatrix::to_eigen() const {
  Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> M(n, n, nnz(), rowptr.data(), col.data(),
                                                                    val.data());
  Eigen::SparseMatrix<double, Eigen::ColMajor> A(M);
  A.makeCompressed();
  return A;
}

// ------------------------------------------------------------------------------------------------ DistCsr
std::vector<int> DistCsr::row_dist() const {
  const int P = size_of(comm);
  std::vector<int> counts(P), dist(P + 1, 0);
  int lr = local_rows;
  MPI_Allgather(&lr, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  for (int p = 0; p < P; ++p) dist[p + 1] = dist[p] + counts[p];
  AXMP_REQUIRE(dist[P] == n_global, "DistCsr: local row counts do not add up to n_global");
  return dist;
}

void DistCsr::mult(const double* x_local, double* y_local) const {
  const auto dist = row_dist();
  const int P = size_of(comm);
  std::vector<int> counts(P);
  for (int p = 0; p < P; ++p) counts[p] = dist[p + 1] - dist[p];
  std::vector<double> x(n_global);
  MPI_Allgatherv(x_local, local_rows, MPI_DOUBLE, x.data(), counts.data(), dist.data(), MPI_DOUBLE, comm);
  for (int i = 0; i < local_rows; ++i) {
    double s = 0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) s += val[k] * x[col[k]];
    y_local[i] = s;
  }
}

CsrMatrix DistCsr::gather(int root) const {
  const int P = size_of(comm), me = rank_of(comm);
  const auto dist = row_dist();
  int nnz_l = rowptr.empty() ? 0 : rowptr[local_rows];
  std::vector<int> nnzc(P), nnzd(P + 1, 0), rc(P);
  MPI_Gather(&nnz_l, 1, MPI_INT, nnzc.data(), 1, MPI_INT, root, comm);
  for (int p = 0; p < P; ++p) {
    nnzd[p + 1] = nnzd[p] + nnzc[p];
    rc[p] = dist[p + 1] - dist[p];
  }
  CsrMatrix A;
  // The root's allocation is the one that can fail here (the gathered matrix is held on one rank); the outcome is
  // shared before any gather, so that a failure raises on every rank instead of leaving the others in MPI_Gatherv.
  // nnz(A) is shared with it: only the root has received the counts, and every rank must throw the same text.
  long alloc_info[2] = {0, nnzd[P]};  // failed, nnz(A)
  if (me == root) {
    try {
      A.n = static_cast<int>(n_global);
      A.rowptr.assign(A.n + 1, 0);
      A.col.resize(nnzd[P]);
      A.val.resize(nnzd[P]);
    } catch (const std::bad_alloc&) {
      alloc_info[0] = 1;
    }
  }
  MPI_Bcast(alloc_info, 2, MPI_LONG, root, comm);
  if (alloc_info[0] != 0)
    throw Error(strcat("SparseLU: rank ", root, " could not allocate the gathered matrix (n = ", n_global,
                       " unknowns, nnz(A) = ", alloc_info[1], "); the serial direct solvers run on that rank alone and "
                       "are limited by its memory (reported by rank ", root, ")"),
                "", "Use STRUMPACK-MPI, which keeps the matrix distributed (scripts/build_strumpack_mpi.sh, then "
                    "lu=\"strumpack-mpi\"), or an iterative solver (solver=\"minres\" for Stokes).");
  // row lengths, then shift into a global row pointer
  std::vector<int> len(local_rows);
  for (int i = 0; i < local_rows; ++i) len[i] = rowptr[i + 1] - rowptr[i];
  MPI_Gatherv(len.data(), local_rows, MPI_INT, me == root ? A.rowptr.data() + 1 : nullptr, rc.data(),
              dist.data(), MPI_INT, root, comm);
  MPI_Gatherv(col.data(), nnz_l, MPI_INT, A.col.data(), nnzc.data(), nnzd.data(), MPI_INT, root, comm);
  MPI_Gatherv(val.data(), nnz_l, MPI_DOUBLE, A.val.data(), nnzc.data(), nnzd.data(), MPI_DOUBLE, root, comm);
  if (me == root)
    for (int i = 0; i < A.n; ++i) A.rowptr[i + 1] += A.rowptr[i];
  return A;
}

void require_single_rank(const mfem::Mesh& mesh, const char* what) {
  const auto* pm = dynamic_cast<const mfem::ParMesh*>(&mesh);
  if (pm == nullptr) return;
  AXMP_REQUIRE(size_of(pm->GetComm()) == 1,
              axmp::strcat(what, " needs one MPI rank: it is a one-rank operator and would solve one independent "
                                "problem per mesh partition on ", size_of(pm->GetComm()), " ranks"),
              "mpi", "Run on one rank (mpirun -np 1).");
}

DistCsr DistCsr::from_serial(const CsrMatrix& A, MPI_Comm comm) {
  AXMP_REQUIRE(size_of(comm) == 1, "DistCsr::from_serial needs a one-rank communicator");
  DistCsr d;
  d.comm = comm;
  d.n_global = A.n;
  d.local_rows = A.n;
  d.rowptr = A.rowptr;
  d.col = A.col;
  d.val = A.val;
  return d;
}

DistCsr DistCsr::distribute(const CsrMatrix& A, const std::vector<int>& dist, MPI_Comm comm) {
  const int P = size_of(comm), me = rank_of(comm);
  AXMP_REQUIRE(static_cast<int>(dist.size()) == P + 1 && dist[0] == 0 && dist[P] == A.n,
              "DistCsr::distribute: dist must have nranks+1 entries from 0 to n");
  DistCsr d;
  d.comm = comm;
  d.n_global = A.n;
  d.row_begin = dist[me];
  d.local_rows = dist[me + 1] - dist[me];
  const int k0 = A.rowptr[dist[me]], k1 = A.rowptr[dist[me + 1]];
  d.rowptr.resize(d.local_rows + 1);
  for (int i = 0; i <= d.local_rows; ++i) d.rowptr[i] = A.rowptr[dist[me] + i] - k0;
  d.col.assign(A.col.begin() + k0, A.col.begin() + k1);
  d.val.assign(A.val.begin() + k0, A.val.begin() + k1);
  return d;
}

void DistCsr::sum_duplicates() {
  std::vector<std::pair<int, double>> row;
  std::vector<int> rp(1, 0), cc;
  std::vector<double> vv;
  cc.reserve(col.size());
  vv.reserve(val.size());
  for (int i = 0; i < local_rows; ++i) {
    row.clear();
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) row.emplace_back(col[k], val[k]);
    std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const int start = static_cast<int>(cc.size());
    for (const auto& e : row) {
      if (static_cast<int>(cc.size()) > start && cc.back() == e.first) vv.back() += e.second;
      else {
        cc.push_back(e.first);
        vv.push_back(e.second);
      }
    }
    rp.push_back(static_cast<int>(cc.size()));
  }
  rowptr.swap(rp);
  col.swap(cc);
  val.swap(vv);
}

std::string DistCsr::contract_violation() const {
  if (static_cast<int>(rowptr.size()) != local_rows + 1 || (local_rows >= 0 && !rowptr.empty() && rowptr[0] != 0))
    return "row pointer array has the wrong size or does not start at 0";
  if (col.size() != val.size() || static_cast<long>(col.size()) != rowptr[local_rows])
    return "column/value arrays do not match the row pointers";
  std::vector<int> c;
  for (int i = 0; i < local_rows; ++i) {
    if (rowptr[i + 1] < rowptr[i]) return strcat("row pointers decrease at local row ", i);
    c.assign(col.begin() + rowptr[i], col.begin() + rowptr[i + 1]);
    for (int j : c)
      if (j < 0 || j >= n_global) return strcat("column index ", j, " out of range in global row ", row_begin + i);
    std::sort(c.begin(), c.end());
    if (std::adjacent_find(c.begin(), c.end()) != c.end())
      return strcat("duplicate column index in global row ", row_begin + i,
                    " (call DistCsr::sum_duplicates() to sum them)");
  }
  return "";
}

double DistCsr::frobenius_norm() const {
  double s = 0, g = 0;
  for (double v : val) s += v * v;
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
  return std::sqrt(g);
}

void DistCsr::abs_mult(const double* x_local, double* y_local) const {
  const auto dist = row_dist();
  const int P = size_of(comm);
  std::vector<int> counts(P);
  for (int p = 0; p < P; ++p) counts[p] = dist[p + 1] - dist[p];
  std::vector<double> x(n_global);
  MPI_Allgatherv(x_local, local_rows, MPI_DOUBLE, x.data(), counts.data(), dist.data(), MPI_DOUBLE, comm);
  for (int i = 0; i < local_rows; ++i) {
    double s = 0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) s += std::fabs(val[k]) * std::fabs(x[col[k]]);
    y_local[i] = s;
  }
}

// ------------------------------------------------------------------------------------------------ blockwise measures
namespace {
// The largest block id over all ranks (-1 when no row is in a block; blocks empty: every row is in block 0), and
// whether any rank has a constraint row (id -2). Collective.
void block_layout(MPI_Comm comm, int local_rows, const std::vector<int>& blocks, int& kmax, bool& constraints) {
  int loc[2] = {blocks.empty() && local_rows > 0 ? 0 : -1, 0}, glob[2] = {-1, 0};
  for (int id : blocks) {
    loc[0] = std::max(loc[0], id);
    if (id == -2) loc[1] = 1;
  }
  MPI_Allreduce(loc, glob, 2, MPI_INT, MPI_MAX, comm);
  kmax = glob[0];
  constraints = glob[1] != 0;
}

// The local values `mine`, one per constraint row of this rank, gathered over the ranks: global row order, because
// the rows are distributed in rank order. Collective.
std::vector<double> gather_in_rank_order(MPI_Comm comm, const std::vector<double>& mine) {
  const int P = size_of(comm);
  int n = static_cast<int>(mine.size());
  std::vector<int> counts(P), offsets(P + 1, 0);
  MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  for (int p = 0; p < P; ++p) offsets[p + 1] = offsets[p] + counts[p];
  std::vector<double> all(offsets[P]);
  MPI_Allgatherv(mine.data(), n, MPI_DOUBLE, all.data(), counts.data(), offsets.data(), MPI_DOUBLE, comm);
  return all;
}

void collective_check(MPI_Comm comm, const std::string& local_msg, const std::string& hint);

// What is wrong with this rank's row blocks ("" when nothing is): the caller passes it to collective_check, so that a
// wrong vector on one rank is an error on every rank and no rank goes on alone into a collective call.
std::string block_ids_error(const DistCsr& A, const std::vector<int>& blocks, const char* who) {
  if (!blocks.empty() && static_cast<int>(blocks.size()) != A.local_rows)
    return strcat(who, ": the row blocks have ", blocks.size(), " entries for ", A.local_rows, " local rows");
  for (int id : blocks)
    if (id < -2)
      return strcat(who, ": row block id ", id, " (ids are >= 0, -1 for a row left out, -2 for a constraint row)");
  return "";
}
const char* kRowBlocksHint = "Give one block id per local row (SparseLU::set_row_blocks before factorize), or none.";
const char* kXbSizeHint = "Pass x and b with one entry per local row of A (A.local_rows entries each).";

// What the chain records of an iterate x of A x = b besides its relative residual: the blockwise backward error, the
// constraint rows' own ratios, and the residual over the rows of `mask` relative to b over them (-1 when no rank has
// a mask). One pass over the rows evaluates A x and |A| |x| together. Collective; every rank gets the same values.
struct IterateMeasures {
  double backward_error = 0;
  double gate_residual = -1;
  std::vector<double> constraint_ratios;
};
IterateMeasures iterate_measures(const DistCsr& A, const double* x_local, const double* b_local,
                                 const std::vector<int>& blocks, const std::vector<char>& mask) {
  const auto dist = A.row_dist();
  const int P = size_of(A.comm);
  std::vector<int> counts(P);
  for (int p = 0; p < P; ++p) counts[p] = dist[p + 1] - dist[p];
  std::vector<double> x(A.n_global);
  MPI_Allgatherv(x_local, A.local_rows, MPI_DOUBLE, x.data(), counts.data(), dist.data(), MPI_DOUBLE, A.comm);
  int kmax = -1;
  bool constraints = false;
  block_layout(A.comm, A.local_rows, blocks, kmax, constraints);
  // sums over the rows: [0] masked r^2, [1] masked b^2, [2] ranks with a mask, then (r^2, d^2) per block id
  std::vector<double> loc(3 + 2 * (kmax + 1), 0.0), glob(loc.size(), 0.0);
  loc[2] = mask.empty() ? 0.0 : 1.0;
  std::vector<double> ratios;
  for (int i = 0; i < A.local_rows; ++i) {
    double s = 0, a = 0;
    for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
      s += A.val[k] * x[A.col[k]];
      a += std::fabs(A.val[k]) * std::fabs(x[A.col[k]]);
    }
    const double r = b_local[i] - s, d = a + std::fabs(b_local[i]);
    if (!mask.empty() && mask[i]) {
      loc[0] += r * r;
      loc[1] += b_local[i] * b_local[i];
    }
    const int id = blocks.empty() ? 0 : blocks[i];
    if (id >= 0) {
      loc[3 + 2 * id] += r * r;
      loc[4 + 2 * id] += d * d;
    } else if (id == -2) {
      ratios.push_back(d > 0 ? std::fabs(r) / d : 0.0);
    }
  }
  MPI_Allreduce(loc.data(), glob.data(), static_cast<int>(loc.size()), MPI_DOUBLE, MPI_SUM, A.comm);
  IterateMeasures m;
  if (glob[2] > 0) m.gate_residual = glob[1] > 0 ? std::sqrt(glob[0] / glob[1]) : std::sqrt(glob[0]);
  // The two square roots are taken apart: a block whose rows are scaled by a power of two then gives the same bits.
  // A block whose ratio is not a number (sums of squares that overflowed, inf / inf, or that hold a NaN) leaves the
  // iterate without a backward error: NaN, which no later block's maximum may hide (std::max drops a NaN second
  // argument, and a maximum that skipped the block would record the best possible value).
  for (int k = 0; k <= kmax && !std::isnan(m.backward_error); ++k) {
    const double r2 = glob[3 + 2 * k], d2 = glob[4 + 2 * k];
    if (std::isnan(r2) || std::isnan(d2)) {
      m.backward_error = std::numeric_limits<double>::quiet_NaN();
    } else if (d2 > 0) {
      const double ratio = std::sqrt(r2) / std::sqrt(d2);
      m.backward_error = std::isnan(ratio) ? ratio : std::max(m.backward_error, ratio);
    }
  }
  if (constraints) m.constraint_ratios = gather_in_rank_order(A.comm, ratios);
  return m;
}
}  // namespace

double lu_backward_error(const DistCsr& A, const std::vector<double>& x_local, const std::vector<double>& b_local,
                         const std::vector<int>& row_blocks_local, std::vector<double>* constraint_ratios) {
  const bool sizes = static_cast<int>(x_local.size()) == A.local_rows &&
                     static_cast<int>(b_local.size()) == A.local_rows;
  // the hint is the failing rank's (collective_check), so each rank passes the one of its own error
  collective_check(A.comm,
                   sizes ? block_ids_error(A, row_blocks_local, "lu_backward_error")
                         : std::string("lu_backward_error: x and b must hold the local rows"),
                   sizes ? kRowBlocksHint : kXbSizeHint);
  IterateMeasures m = iterate_measures(A, x_local.data(), b_local.data(), row_blocks_local, {});
  if (constraint_ratios != nullptr) *constraint_ratios = std::move(m.constraint_ratios);
  return m.backward_error;
}

// ------------------------------------------------------------------------------------------------ backends
const char* to_string(LUBackend b) {
  switch (b) {
    case LUBackend::EigenLU: return "eigen";
    case LUBackend::StrumpackLU: return "strumpack";
    case LUBackend::StrumpackMPI: return "strumpack-mpi";
  }
  return "?";
}

LUBackend lu_backend_from_string(const std::string& s) {
  if (s == "eigen") return LUBackend::EigenLU;
  if (s == "strumpack") return LUBackend::StrumpackLU;
  if (s == "strumpack-mpi") return LUBackend::StrumpackMPI;
  throw Error("unknown sparse LU backend '" + s + "'", "", "Use eigen, strumpack or strumpack-mpi.");
}

bool lu_backend_available(LUBackend b) {
  switch (b) {
    case LUBackend::EigenLU: return true;
    case LUBackend::StrumpackLU:
#ifdef AXMP_STRUMPACK
      return true;
#else
      return false;
#endif
    case LUBackend::StrumpackMPI:
#ifdef AXMP_STRUMPACK_MPI
      return true;
#else
      return false;
#endif
  }
  return false;
}

std::string LUStage::name() const {
  std::string s = to_string(backend);
  if (backend != LUBackend::EigenLU)
    s += matching == MatchingJob::None ? ":none" : matching == MatchingJob::MaxSmallestDiagonal ? ":mc64-2" : ":mc64-5";
  if (equilibrate) s += ":eq";
  return s;
}

// ------------------------------------------------------------------------------------------------ test hooks
namespace {
struct ChainHooks {
  std::mutex mutex;
  LUForcedOutcome forced = LUForcedOutcome::None;
  int forced_count = 0;
  double perturbation = 0;
  bool rank1_flip = false;
  std::function<void(int, std::vector<double>&)> iterate_hook;
  std::vector<LUStage> default_chain;
};
ChainHooks& chain_hooks() {
  static ChainHooks h;
  return h;
}
// The forced outcome of the attempt of chain index 0 that is starting, and one less to go. Threw and Refused are
// taken when stage 0 is about to factorize, MissedTarget when stage 0 has produced an iterate. Every rank runs the
// same sequence of attempts, so every rank's count stays the same.
LUForcedOutcome take_forced_outcome(bool at_factorization) {
  ChainHooks& h = chain_hooks();
  std::lock_guard<std::mutex> lock(h.mutex);
  if (h.forced == LUForcedOutcome::None || h.forced_count <= 0) return LUForcedOutcome::None;
  if ((h.forced == LUForcedOutcome::MissedTarget) == at_factorization) return LUForcedOutcome::None;
  --h.forced_count;
  return h.forced;
}

std::atomic<int> g_backend_objects{0}, g_backend_objects_peak{0};
void backend_object_made() {
  const int live = ++g_backend_objects;
  int peak = g_backend_objects_peak.load();
  while (live > peak && !g_backend_objects_peak.compare_exchange_weak(peak, live)) {
  }
}
void backend_object_gone() { --g_backend_objects; }
}  // namespace

void set_lu_forced_outcome_for_testing(LUForcedOutcome outcome, int count) {
  ChainHooks& h = chain_hooks();
  std::lock_guard<std::mutex> lock(h.mutex);
  const bool on = outcome != LUForcedOutcome::None && count > 0;
  h.forced = on ? outcome : LUForcedOutcome::None;
  h.forced_count = on ? count : 0;
}
void set_lu_solution_perturbation_for_testing(double eps) {
  ChainHooks& h = chain_hooks();
  std::lock_guard<std::mutex> lock(h.mutex);
  h.perturbation = eps;
}
void set_lu_iterate_hook_for_testing(std::function<void(int, std::vector<double>&)> hook) {
  ChainHooks& h = chain_hooks();
  std::lock_guard<std::mutex> lock(h.mutex);
  h.iterate_hook = std::move(hook);
}
void set_lu_rank1_acceptance_flip_for_testing(bool on) {
  ChainHooks& h = chain_hooks();
  std::lock_guard<std::mutex> lock(h.mutex);
  h.rank1_flip = on;
}
void set_default_lu_chain_for_testing(std::vector<LUStage> chain) {
  ChainHooks& h = chain_hooks();
  std::lock_guard<std::mutex> lock(h.mutex);
  h.default_chain = std::move(chain);
}
LUBackendObjectCount lu_backend_objects_for_testing() {
  LUBackendObjectCount c;
  c.live = g_backend_objects.load();
  c.peak = g_backend_objects_peak.load();
  return c;
}
void reset_lu_backend_object_peak_for_testing() { g_backend_objects_peak = g_backend_objects.load(); }

// A solve that begins with a matched STRUMPACK factorization (MC64 job 5) goes on with MC64 job 2 and then to Eigen;
// one that begins with job 2 goes on with job 5; one that begins without the matching (3D saddle points, where the
// matching damages the factorization) goes on with it and then to Eigen; Eigen has no retry. No configuration is
// listed twice, and the stage without matching follows no matched one (the header has the measurements). The Eigen
// stage after a STRUMPACK stage is capped (the header says why).
std::vector<LUStage> default_chain(LUBackend first, MatchingJob first_matching) {
  {
    ChainHooks& h = chain_hooks();
    std::lock_guard<std::mutex> lock(h.mutex);
    if (!h.default_chain.empty()) return h.default_chain;
  }
  auto stage = [](LUBackend b, MatchingJob m, long max_n) {
    LUStage s;
    s.backend = b;
    s.matching = m;
    s.max_n = max_n;
    return s;
  };
  const MatchingJob job5 = MatchingJob::MaxDiagonalProductScaling;
  const MatchingJob job2 = MatchingJob::MaxSmallestDiagonal;
  if (first == LUBackend::EigenLU) return {stage(first, job5, 0)};
  if (first_matching == MatchingJob::None)
    return {stage(first, MatchingJob::None, 0), stage(first, job5, 0),
            stage(LUBackend::EigenLU, job5, kLUEigenRetryMaxN3D)};
  const MatchingJob other = first_matching == job2 ? job5 : job2;
  return {stage(first, first_matching, 0), stage(first, other, 0), stage(LUBackend::EigenLU, job5, kLUEigenRetryMaxN)};
}

using EigenLU32 = Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, Eigen::COLAMDOrdering<int>>;
using EigenLU64 =
    Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>, Eigen::COLAMDOrdering<std::int64_t>>;

struct SparseLU::Impl {
  MPI_Comm comm = MPI_COMM_WORLD;
  std::vector<int> dist;  // row distribution of the DistCsr
  // serial backends (rank 0 only)
  // Eigen's solver, one object per factorization (see factorize): at most one of the two is alive, the 32-bit
  // instantiation or the 64-bit one (eigen_lu_index_bits).
  std::unique_ptr<EigenLU32> eigen32;
  std::unique_ptr<EigenLU64> eigen64;
  void drop_eigen() {
    if (eigen32) backend_object_gone();
    if (eigen64) backend_object_gone();
    eigen32.reset();
    eigen64.reset();
  }
#ifdef AXMP_STRUMPACK
  // Every call into this solver goes through the STRUMPACK guard of sym_solver.cpp (axmp::strumpack_guarded_life
  // around construction and destruction, axmp::strumpack_guarded_call around the rest, the ordering also under
  // axmp::metis_mutex()): the EM solvers call the same STRUMPACK in the same process, and STRUMPACK keeps process-wide
  // state (per-thread timer lists, thread parameters, the std::new_handler, METIS's signal handlers) that every
  // caller has to coordinate.  In a build against the static MPI STRUMPACK (AXMP_STRUMPACK_PREFIX) libaxionmultiphysics keeps
  // its STRUMPACK symbols private and this library links a second copy: the guard's locks still order these calls
  // with the EM solvers', but the guard empties the timer lists of libaxionmultiphysics's copy, not of this one (whose list
  // for thread number 0 then keeps a few entries per factorization and solve).
  std::unique_ptr<strumpack::StrumpackSparseSolver<double, int>> sp;
#endif
#ifdef AXMP_STRUMPACK_MPI
  // Through the collective guard, like the distributed full-wave solver (model3d_fullwave.cpp;
  // axmp::strumpack_guarded_collective_life around construction and destruction, axmp::strumpack_guarded_collective_call
  // around the rest): one collective caller per rank, outside any OpenMP parallel region (thread number 0, which the
  // timer vector always covers). With the serialized guard every call holds the process-wide STRUMPACK lock across
  // its collectives. With the concurrent guard (8.0.x) the calls just run, so a distributed fluids solve that runs
  // while another thread of the same rank is inside a guarded STRUMPACK call is not coordinated with it. In neither
  // mode does its ordering take axmp::metis_mutex() (METIS's signal handlers), and the std::new_handler is left to
  // STRUMPACK.
  struct MpiDelete {
    void operator()(strumpack::StrumpackSparseSolverMPIDist<double, int>* s) const {
      axmp::strumpack_guarded_collective_life([&] { delete s; });
    }
  };
  std::unique_ptr<strumpack::StrumpackSparseSolverMPIDist<double, int>, MpiDelete> spmpi;
#endif
};

#ifdef AXMP_STRUMPACK
namespace {
template <class S>
void configure_strumpack(S& s, MatchingJob job) {
  auto& o = s.options();
  o.set_verbose(false);
  o.set_reordering_method(strumpack::ReorderingStrategy::METIS);
  o.set_matching(job == MatchingJob::None                  ? strumpack::MatchingJob::NONE
                 : job == MatchingJob::MaxSmallestDiagonal ? strumpack::MatchingJob::MAX_SMALLEST_DIAGONAL
                                                           : strumpack::MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING);
  o.set_compression(strumpack::CompressionType::NONE);
  o.set_Krylov_solver(strumpack::KrylovSolver::DIRECT);  // plain LU solve: the residual we report is the LU's
}
std::string rc_name(strumpack::ReturnCode rc) {
  switch (rc) {
    case strumpack::ReturnCode::SUCCESS: return "SUCCESS";
    case strumpack::ReturnCode::MATRIX_NOT_SET: return "MATRIX_NOT_SET";
    case strumpack::ReturnCode::REORDERING_ERROR: return "REORDERING_ERROR";
    case strumpack::ReturnCode::ZERO_PIVOT: return "ZERO_PIVOT";
    case strumpack::ReturnCode::NO_CONVERGENCE: return "NO_CONVERGENCE";
    default: return "error " + std::to_string(static_cast<int>(rc));
  }
}
}  // namespace
#endif

// ------------------------------------------------------------------------------------------------ index width
namespace {
std::atomic<long> g_max_n32{0};  // the test bound; <= 0: kEigenLUIndex32MaxN
std::atomic<int> g_failure{0};   // an EigenLUTestFailure
}  // namespace

int eigen_lu_index_bits(long n, long nnz, long max_n32) {
  return n <= max_n32 && nnz <= kEigenLUIndex32MaxNnz ? 32 : 64;
}
int eigen_lu_index_bits(long n, long nnz) { return eigen_lu_index_bits(n, nnz, kEigenLUIndex32MaxN); }
long eigen_lu_factor_nnz_estimate(long n, long nnz) {
  return n <= 0 ? 0 : std::min(20 * (nnz + 1) / n, n) * n;
}
// Eigen's memInit failure sets lastErrorMessage() and leaves info() alone, so the message decides first.
bool eigen_lu_failed(Eigen::ComputationInfo info, const std::string& message) {
  return !message.empty() || info != Eigen::Success;
}
void set_eigen_lu_max_n32_for_testing(long max_n) { g_max_n32 = max_n; }
void set_eigen_lu_failure_for_testing(EigenLUTestFailure mode) { g_failure = static_cast<int>(mode); }
int SparseLU::live_eigen_index_bits_for_testing() const {
  return impl_->eigen32 ? 32 : impl_->eigen64 ? 64 : 0;
}

namespace {
struct EigenRun {
  double analyze_seconds = 0, factor_seconds = 0;
  long factor_nnz = -1;
  Eigen::ComputationInfo info = Eigen::Success;
  std::string message;  // Eigen's lastErrorMessage(), or the failure the test hook feeds
  bool failed = false;
};

// Factorize the gathered matrix G with a fresh solver of type Solver (StorageIndex Index). Throws what Eigen or the
// copies throw (std::bad_alloc).
template <class Solver, class Index>
void run_eigen(std::unique_ptr<Solver>& slot, const CsrMatrix& G, EigenRun& r) {
  // A fresh object: Eigen never clears lastErrorMessage() on success, and info() is indeterminate until a
  // factorization sets it.
  slot.reset(new Solver);
  backend_object_made();
  const auto hook = static_cast<EigenLUTestFailure>(g_failure.load());
  if (hook == EigenLUTestFailure::BadAlloc) throw std::bad_alloc();
  double t0 = wtime();
  Eigen::SparseMatrix<double, Eigen::ColMajor, Index> M;
  if constexpr (std::is_same_v<Index, int>) {
    M = G.to_eigen();  // the 32-bit path as it always was
  } else {
    // CsrMatrix indexes with int: a copy in the wide type, converted to Eigen's column-major storage
    std::vector<Index> rp(G.rowptr.begin(), G.rowptr.end()), cl(G.col.begin(), G.col.end());
    Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, Index>> R(G.n, G.n, G.nnz(), rp.data(), cl.data(),
                                                                         G.val.data());
    M = R;
  }
  slot->analyzePattern(M);
  double t1 = wtime();
  slot->factorize(M);
  double t2 = wtime();
  r.analyze_seconds = t1 - t0;
  r.factor_seconds = t2 - t1;
  r.message = slot->lastErrorMessage();
  r.info = r.message.empty() ? slot->info() : Eigen::NumericalIssue;  // info() only when no message: see eigen_lu_failed
  if (hook == EigenLUTestFailure::ExpandMemory) {
    r.info = Eigen::NumericalIssue;
    r.message = "UNABLE TO EXPAND MEMORY IN COLUMN_BMOD() ";
  } else if (hook == EigenLUTestFailure::MemInit) {
    r.info = Eigen::Success;
    r.message = "UNABLE TO ALLOCATE WORKING MEMORY\n\n";
  }
  r.failed = eigen_lu_failed(r.info, r.message);
  if (!r.failed) r.factor_nnz = static_cast<long>(slot->nnzL() + slot->nnzU());
}

std::string rtrim(std::string s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
  return s;
}

const char* kEigenMemoryHint =
    "build with STRUMPACK (scripts/build_strumpack.sh, or scripts/build_strumpack_mpi.sh for the distributed "
    "solver) and use lu=\"strumpack\" or lu=\"strumpack-mpi\", or use an iterative solver (solver=\"minres\" for "
    "Stokes).";
}  // namespace

namespace {
std::vector<LUStage> one_stage(LUBackend backend) {
  LUStage s;
  s.backend = backend;
  return {s};
}
}  // namespace

SparseLU::SparseLU(LUBackend backend) : SparseLU(one_stage(backend)) {}

SparseLU::SparseLU(std::vector<LUStage> chain) : impl_(new Impl), backend_(LUBackend::EigenLU), chain_(std::move(chain)) {
  AXMP_REQUIRE(!chain_.empty(), "SparseLU: the chain of stages is empty", "",
               "Give at least one stage, or SparseLU(LUBackend) for a single attempt.");
  for (const auto& st : chain_)
    AXMP_REQUIRE(!st.equilibrate, "SparseLU: stage '" + st.name() + "' asks for equilibration, which is not available",
                 "", "Leave LUStage::equilibrate false.");
  backend_ = chain_[0].backend;
  AXMP_REQUIRE(lu_backend_available(backend_),
              std::string("sparse LU backend '") + to_string(backend_) + "' is not in this build",
              "", backend_ == LUBackend::StrumpackMPI
                      ? "Configure axionMultiPhysics with -DAXMP_STRUMPACK_PREFIX=<MPI STRUMPACK prefix> "
                        "(scripts/build_strumpack_mpi.sh), or use eigen/strumpack."
                      : "Build with STRUMPACK (scripts/build_strumpack.sh), or use eigen.");
  stats_.backend = to_string(backend_);
}

SparseLU::~SparseLU() {
  if (impl_) release_backend();
}

void SparseLU::release_backend() {
  impl_->drop_eigen();
#ifdef AXMP_STRUMPACK
  if (impl_->sp) {
    axmp::strumpack_guarded_life([&] { impl_->sp.reset(); });
    backend_object_gone();
  }
#endif
#ifdef AXMP_STRUMPACK_MPI
  if (impl_->spmpi) {
    impl_->spmpi.reset();  // its deleter runs under the collective guard
    backend_object_gone();
  }
#endif
}

void SparseLU::set_acceptance(LUAcceptance acceptance) {
  AXMP_REQUIRE(acceptance.stop_on_backward_error == 0,
               "SparseLU::set_acceptance: stop_on_backward_error is not available (it must stay 0)");
  AXMP_REQUIRE(acceptance.kind != LUAcceptance::Kind::Verdict || acceptance.verdict,
               "SparseLU::set_acceptance: a Verdict acceptance needs a verdict function");
  AXMP_REQUIRE(acceptance.bound >= 0,  // false for a NaN too
               strcat("SparseLU::set_acceptance: the bound ", acceptance.bound, " is negative or not a number"));
  acceptance_ = std::move(acceptance);
}

void SparseLU::set_row_blocks(std::vector<int> blocks_local) { row_blocks_ = std::move(blocks_local); }

int SparseLU::strumpack_matching_job_for_testing() const {
#ifdef AXMP_STRUMPACK
  if (impl_->sp) return static_cast<int>(impl_->sp->options().matching());
#endif
#ifdef AXMP_STRUMPACK_MPI
  if (impl_->spmpi) return static_cast<int>(impl_->spmpi->options().matching());
#endif
  return -1;
}

namespace {
// Collective failure: every rank passes its local message ("" = fine); if any rank failed, all ranks throw the
// message and hint of the lowest failing rank. Prevents the deadlock of a rank-local throw before a collective call.
void collective_check(MPI_Comm comm, const std::string& local_msg, const std::string& hint) {
  const int me = rank_of(comm);
  int mine = local_msg.empty() ? size_of(comm) : me, first = 0;
  MPI_Allreduce(&mine, &first, 1, MPI_INT, MPI_MIN, comm);
  if (first == size_of(comm)) return;
  std::string msg = local_msg;
  int len = static_cast<int>(msg.size());
  MPI_Bcast(&len, 1, MPI_INT, first, comm);
  msg.resize(len);
  MPI_Bcast(msg.data(), len, MPI_CHAR, first, comm);
  // The hint comes from the failing rank too: only that rank knows the kind of failure.
  std::string h = hint;
  len = static_cast<int>(h.size());
  MPI_Bcast(&len, 1, MPI_INT, first, comm);
  h.resize(len);
  MPI_Bcast(h.data(), len, MPI_CHAR, first, comm);
  throw Error(msg + strcat(" (reported by rank ", first, ")"), "", h);
}
const char* kSingularHint =
    "The factorization met a zero pivot or a singular matrix. If this is a saddle point, check that every "
    "nullspace was removed (pressure constant: pin or mean-value multiplier) and that the boundary conditions "
    "determine the solution. Note that a singular saddle point is NOT reliably reported by a zero pivot: use "
    "SparseLU::set_kernel_probe().";
const char* kNoFactorizationHint = "Call factorize() and let it return without an error before solve().";
}  // namespace

SparseLU::ThreadGuard::ThreadGuard(int n) : saved_(omp_get_max_threads()), active_(n > 0) {
  if (active_) omp_set_num_threads(n);
}
SparseLU::ThreadGuard::~ThreadGuard() {
  if (active_) omp_set_num_threads(saved_);
}

namespace {
// "<stage>: <reason>" of every attempt: the text of the error when no stage of a chain could be used.
std::string chain_failure(const std::vector<LUAttempt>& attempts) {
  std::string s = "every stage of the sparse LU chain failed: ";
  for (std::size_t k = 0; k < attempts.size(); ++k)
    s += (k > 0 ? "; " : "") + attempts[k].stage + ": " + attempts[k].reason;
  return s;
}
}  // namespace

void SparseLU::factorize(const DistCsr& A) {
  ThreadGuard tg(threads_);
  // No factorization until this one succeeds: A_ is set at the end, after the last collective check, so solve()
  // after a failure anywhere in here (the input contract, the kernel probe, the gather, the backend) refuses on every
  // rank instead of applying the previous factor to the new matrix. The previous factor is released before the
  // gather allocates.
  A_ = nullptr;
  current_ = -1;
  accepted_stage_ = -1;
  solution_stage_ = -1;
  attempts_.clear();
  attempts_from_factorize_ = false;
  release_backend();
  impl_->comm = A.comm;
  impl_->dist = A.row_dist();
  stats_ = LUStats{};
  stats_.backend = to_string(backend_);
  stats_.n = A.n_global;
  long nnz_l = A.rowptr.empty() ? 0 : A.rowptr[A.local_rows];
  MPI_Allreduce(&nnz_l, &stats_.nnz, 1, MPI_LONG, MPI_SUM, A.comm);
  stats_.threads = omp_get_max_threads();
  stats_.ranks = size_of(A.comm);

  collective_check(A.comm, A.contract_violation(), "Fix the CSR input (see sparse_lu.hpp, input contract).");
  collective_check(A.comm, block_ids_error(A, row_blocks_, "SparseLU::factorize"), kRowBlocksHint);
  held_blocks_ = row_blocks_;
  if (!probe_.empty()) {
    AXMP_REQUIRE(static_cast<int>(probe_.size()) == A.local_rows, "SparseLU: kernel probe has the wrong local size");
    std::vector<double> y(A.local_rows);
    A.mult(probe_.data(), y.data());
    // row blocks declared on any rank: the blockwise probe (a rank without rows cannot declare any)
    int declared_l = row_blocks_.empty() ? 0 : 1, declared = 0;
    MPI_Allreduce(&declared_l, &declared, 1, MPI_INT, MPI_MAX, A.comm);
    const char* kHint = "Remove the nullspace first (pressure: pin or mean-value multiplier; or a Krylov solver with "
                        "OrthoSolver).";
    if (declared == 0) {
      double loc[2] = {0, 0}, glob[2];
      for (int i = 0; i < A.local_rows; ++i) {
        loc[0] += y[i] * y[i];
        loc[1] += probe_[i] * probe_[i];
      }
      MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, A.comm);
      const double nA = A.frobenius_norm();
      stats_.kernel_probe = (nA > 0 && glob[1] > 0) ? std::sqrt(glob[0]) / (nA * std::sqrt(glob[1])) : 0.0;
      AXMP_REQUIRE(stats_.kernel_probe > probe_tol_,
                  strcat("the matrix maps the kernel probe to ~0 (||A z|| / (||A|| ||z||) = ", stats_.kernel_probe,
                         "): a nullspace was not removed, and an LU would return an arbitrary component along it"),
                  "", kHint);
    } else {
      // Blockwise: each block's ||A z|| against the size of its own terms, || |A| |z| ||, so that neither the length
      // unit nor the other blocks' scale enters. A constraint row is a block of its own: the row that removes the
      // nullspace gives a ratio of order one (|w^T z| against |w|^T |z|), while a nullspace left in gives round-off
      // in every block that sees z at all.
      std::vector<double> a(A.local_rows);
      A.abs_mult(probe_.data(), a.data());
      int kmax = -1;
      bool constraints = false;
      block_layout(A.comm, A.local_rows, row_blocks_, kmax, constraints);
      std::vector<double> loc(1 + 2 * (kmax + 1), 0.0), glob(loc.size(), 0.0);  // [0]: the constraint rows' largest
      for (int i = 0; i < A.local_rows; ++i) {
        const int id = row_blocks_.empty() ? 0 : row_blocks_[i];
        if (id >= 0) {
          loc[1 + 2 * id] += y[i] * y[i];
          loc[2 + 2 * id] += a[i] * a[i];
        } else if (id == -2 && a[i] > 0) {
          loc[0] = std::max(loc[0], std::fabs(y[i]) / a[i]);
        }
      }
      MPI_Allreduce(loc.data() + 1, glob.data() + 1, static_cast<int>(loc.size()) - 1, MPI_DOUBLE, MPI_SUM, A.comm);
      MPI_Allreduce(loc.data(), glob.data(), 1, MPI_DOUBLE, MPI_MAX, A.comm);
      double ratio = glob[0];
      for (int k = 0; k <= kmax; ++k)
        if (glob[2 + 2 * k] > 0) ratio = std::max(ratio, std::sqrt(glob[1 + 2 * k]) / std::sqrt(glob[2 + 2 * k]));
      stats_.kernel_probe = ratio;
      AXMP_REQUIRE(stats_.kernel_probe > probe_tol_,
                  strcat("the matrix maps the kernel probe to ~0 in every row block (the largest ||(A z)_k|| / "
                         "||(|A| |z|)_k|| over the blocks and the constraint rows is ", stats_.kernel_probe,
                         "): a nullspace was not removed, and an LU would return an arbitrary component along it"),
                  "", kHint);
    }
  }
  matrix_stats_ = stats_;

  // the first stage that factorizes; the stages passed over are recorded
  std::exception_ptr error;
  for (int k = 0; k < static_cast<int>(chain_.size()) && current_ < 0; ++k)
    if (try_stage(A, k, &error)) current_ = k;
  stats_.attempts = static_cast<int>(attempts_.size());
  if (current_ < 0) {
    if (chain_.size() == 1 && error) std::rethrow_exception(error);  // a single attempt: its own error
    throw Error(chain_failure(attempts_));
  }
  A_ = &A;
  attempts_from_factorize_ = true;
}

bool SparseLU::try_stage(const DistCsr& A, int k, std::exception_ptr* error) {
  const LUStage& st = chain_[k];
  LUAttempt at;
  at.stage = st.name();
  if (!lu_backend_available(st.backend)) {
    at.outcome = "not_available";
    at.reason = std::string(to_string(st.backend)) + " is not in this build";
    attempts_.push_back(std::move(at));
    return false;
  }
  if (st.max_n > 0 && A.n_global > st.max_n) {  // n_global is the same on every rank
    at.outcome = "refused";
    at.reason = strcat("size: n = ", A.n_global, " > ", st.max_n);
    attempts_.push_back(std::move(at));
    return false;
  }
  const LUForcedOutcome forced = k == 0 ? take_forced_outcome(true) : LUForcedOutcome::None;
  if (forced == LUForcedOutcome::Refused) {
    at.outcome = "refused";
    at.reason = "test hook";
    attempts_.push_back(std::move(at));
    return false;
  }
  try {
    if (forced == LUForcedOutcome::Threw) throw Error("test hook: the factorization of this stage was made to fail");
    factorize_stage(A, k);
    return true;
  } catch (const Error& e) {  // thrown on every rank (collective_check), so every rank is here
    release_backend();
    at.outcome = "threw";
    at.reason = e.what();
    double t[2] = {stats_.analyze_seconds, stats_.factor_seconds};
    MPI_Bcast(t, 2, MPI_DOUBLE, 0, A.comm);
    at.analyze_seconds = t[0];
    at.factor_seconds = t[1];
    attempts_.push_back(std::move(at));
    if (error != nullptr) *error = std::current_exception();
    return false;
  }
}

void SparseLU::factorize_stage(const DistCsr& A, int k) {
  const LUStage& stage = chain_[k];
  const int me = rank_of(A.comm);
  release_backend();  // at most one backend object: the previous stage's goes before this one's is made
  stats_ = matrix_stats_;
  stats_.backend = to_string(stage.backend);
  stats_.stage = stage.name();
  if (stage.backend == LUBackend::StrumpackMPI) {
#ifdef AXMP_STRUMPACK_MPI
    axmp::strumpack_guarded_collective_life([&] {
      impl_->spmpi.reset(new strumpack::StrumpackSparseSolverMPIDist<double, int>(A.comm, false));
      backend_object_made();
    });
    auto& s = *impl_->spmpi;
    configure_strumpack(s, stage.matching);
    double t0 = wtime();
    auto rc = strumpack::ReturnCode::SUCCESS;
    axmp::strumpack_guarded_collective_call([&] {
      s.set_distributed_csr_matrix(A.local_rows, A.rowptr.data(), A.col.data(), A.val.data(), impl_->dist.data(),
                                   false);
      rc = s.reorder();
    });
    collective_check(A.comm, rc == strumpack::ReturnCode::SUCCESS ? "" : "STRUMPACK-MPI reordering failed (" +
                                                                            rc_name(rc) + ")", kSingularHint);
    double t1 = wtime();
    axmp::strumpack_guarded_collective_call([&] { rc = s.factor(); });
    // STRUMPACK-MPI reports ZERO_PIVOT only on the rank that met it: reduce before anything collective
    collective_check(A.comm, rc == strumpack::ReturnCode::SUCCESS ? "" : "STRUMPACK-MPI factorization failed (" +
                                                                            rc_name(rc) + ")", kSingularHint);
    double t2 = wtime();
    stats_.analyze_seconds = t1 - t0;
    stats_.factor_seconds = t2 - t1;
    axmp::strumpack_guarded_collective_call([&] {
      stats_.factor_nnz = static_cast<long>(s.factor_nonzeros());
      stats_.factor_mem_bytes = static_cast<double>(s.factor_memory());
    });
#endif
    stage_stats_ = stats_;
    return;
  }

  // serial backends: gather to rank 0
  CsrMatrix G = A.gather(0);
  std::string why;
  std::string hint = kSingularHint;  // the failing rank's hint is the one every rank throws
  int bits = 0;
  if (stage.backend == LUBackend::EigenLU) {
    // n and the Allreduced nnz(A) are the same on every rank: the width needs no broadcast
    const long bound = g_max_n32.load();
    bits = eigen_lu_index_bits(stats_.n, stats_.nnz, bound > 0 ? bound : kEigenLUIndex32MaxN);
    stats_.factor_nnz_estimate = eigen_lu_factor_nnz_estimate(stats_.n, stats_.nnz);
  }
  if (me == 0) {
    if (stage.backend == LUBackend::EigenLU) {
      EigenRun run;
      std::string caught;
      try {
        if (bits == 32) run_eigen<EigenLU32, int>(impl_->eigen32, G, run);
        else run_eigen<EigenLU64, std::int64_t>(impl_->eigen64, G, run);
      } catch (const std::exception& e) {
        caught = e.what();
      }
      stats_.analyze_seconds = run.analyze_seconds;
      stats_.factor_seconds = run.factor_seconds;
      if (!caught.empty() || run.failed) {
        impl_->drop_eigen();
        const std::string text = rtrim(caught.empty() ? run.message : caught);
        if (!caught.empty() || run.message.find("MEMORY") != std::string::npos) {
          why = strcat("Eigen's sparse LU could not factorize this system on rank 0 (Eigen: \"", text, "\"): n = ",
                       stats_.n, " unknowns, nnz(A) = ", stats_.nnz, ", ", bits,
                       "-bit indices; Eigen's initial estimate of the factor was ", stats_.factor_nnz_estimate,
                       " entries for each of L and U, and it grows its arrays as the fill requires. The fluids "
                       "library's Eigen LU runs on rank 0 alone and is limited by that rank's memory.");
          hint = kEigenMemoryHint;
        } else {
          why = "Eigen SparseLU failed: " + text;
        }
      } else {
        stats_.factor_nnz = run.factor_nnz;
        stats_.index_bits = bits;
      }
    } else {
#ifdef AXMP_STRUMPACK
      axmp::strumpack_guarded_life([&] {
        impl_->sp.reset(new strumpack::StrumpackSparseSolver<double, int>(false, true));
        backend_object_made();
      });
      auto& s = *impl_->sp;
      configure_strumpack(s, stage.matching);
      double t0 = wtime();
      auto rc = strumpack::ReturnCode::SUCCESS;
      axmp::strumpack_guarded_call([&] {
        // TRUE CSR: row pointers / column indices / values of A itself (not the CSC arrays of an Eigen matrix)
        s.set_csr_matrix(G.n, G.rowptr.data(), G.col.data(), G.val.data(), false);
        std::lock_guard<std::mutex> metis(axmp::metis_mutex());
        rc = s.reorder();
      });
      double t1 = wtime();
      if (rc == strumpack::ReturnCode::SUCCESS) axmp::strumpack_guarded_call([&] { rc = s.factor(); });
      double t2 = wtime();
      stats_.analyze_seconds = t1 - t0;
      stats_.factor_seconds = t2 - t1;
      if (rc != strumpack::ReturnCode::SUCCESS) why = "STRUMPACK LU failed (" + rc_name(rc) + ")";
      else {
        axmp::strumpack_guarded_call([&] {
          stats_.factor_nnz = static_cast<long>(s.factor_nonzeros());
          stats_.factor_mem_bytes = static_cast<double>(s.factor_memory());
        });
      }
#endif
    }
  }
  collective_check(A.comm, why, hint);
  double tt[3] = {stats_.analyze_seconds, stats_.factor_seconds, stats_.factor_mem_bytes};
  MPI_Bcast(tt, 3, MPI_DOUBLE, 0, A.comm);
  stats_.analyze_seconds = tt[0];
  stats_.factor_seconds = tt[1];
  stats_.factor_mem_bytes = tt[2];
  long fn = stats_.factor_nnz;
  MPI_Bcast(&fn, 1, MPI_LONG, 0, A.comm);
  stats_.factor_nnz = fn;
  int ib = stats_.index_bits;
  MPI_Bcast(&ib, 1, MPI_INT, 0, A.comm);
  stats_.index_bits = ib;
  stage_stats_ = stats_;
}

std::vector<double> SparseLU::raw_solve(const std::vector<double>& b_local) {
  const DistCsr& A = *A_;
  const int me = rank_of(A.comm), P = size_of(A.comm);
  const LUBackend backend = chain_[current_].backend;
  std::vector<double> x(A.local_rows, 0.0);
  if (backend == LUBackend::StrumpackMPI) {
#ifdef AXMP_STRUMPACK_MPI
    auto rc = strumpack::ReturnCode::SUCCESS;
    axmp::strumpack_guarded_collective_call([&] { rc = impl_->spmpi->solve(b_local.data(), x.data(), false); });
    collective_check(A.comm, rc == strumpack::ReturnCode::SUCCESS ? "" : "STRUMPACK-MPI solve failed (" +
                                                                            rc_name(rc) + ")", kSingularHint);
#endif
    return x;
  }
  const auto& dist = impl_->dist;
  std::vector<int> counts(P);
  for (int p = 0; p < P; ++p) counts[p] = dist[p + 1] - dist[p];
  std::vector<double> b, xg;
  if (me == 0) {
    b.resize(A.n_global);
    xg.resize(A.n_global);
  }
  MPI_Gatherv(b_local.data(), A.local_rows, MPI_DOUBLE, b.data(), counts.data(), dist.data(), MPI_DOUBLE, 0, A.comm);
  std::string why, hint = kSingularHint;
  if (me == 0) {
    if (backend == LUBackend::EigenLU) {
      Eigen::Map<Eigen::VectorXd> bv(b.data(), A.n_global);
      try {
        auto solve_with = [&](auto& solver) {
          Eigen::VectorXd xv = solver->solve(bv);
          if (solver->info() != Eigen::Success) why = "Eigen SparseLU solve failed";
          else std::copy(xv.data(), xv.data() + A.n_global, xg.data());
        };
        if (impl_->eigen32) solve_with(impl_->eigen32);
        else if (impl_->eigen64) solve_with(impl_->eigen64);
        else {
          why = "SparseLU::solve: no factorization (the last factorize failed)";
          hint = kNoFactorizationHint;
        }
      } catch (const std::exception& e) {
        why = std::string("Eigen SparseLU solve failed: ") + e.what();
      }
    } else {
#ifdef AXMP_STRUMPACK
      auto rc = strumpack::ReturnCode::SUCCESS;
      axmp::strumpack_guarded_call([&] { rc = impl_->sp->solve(b.data(), xg.data(), false); });
      if (rc != strumpack::ReturnCode::SUCCESS) why = "STRUMPACK solve failed (" + rc_name(rc) + ")";
#endif
    }
  }
  collective_check(A.comm, why, hint);  // before the scatter: no rank may be left waiting in it
  MPI_Scatterv(xg.data(), counts.data(), dist.data(), MPI_DOUBLE, x.data(), A.local_rows, MPI_DOUBLE, 0, A.comm);
  return x;
}

namespace {
// r = b - A x (local rows) and ||r|| / ||b|| over all rows (||r|| when b is zero). Collective.
double relative_residual(const DistCsr& A, const std::vector<double>& b_local, const std::vector<double>& x,
                         std::vector<double>& r) {
  A.mult(x.data(), r.data());
  double loc[2] = {0, 0}, glob[2];
  for (int i = 0; i < A.local_rows; ++i) {
    r[i] = b_local[i] - r[i];
    loc[0] += r[i] * r[i];
    loc[1] += b_local[i] * b_local[i];
  }
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, A.comm);
  return glob[1] > 0 ? std::sqrt(glob[0] / glob[1]) : std::sqrt(glob[0]);
}
}  // namespace

std::vector<double> SparseLU::solve_stage(const std::vector<double>& b_local) {
  const DistCsr& A = *A_;
  double t0 = wtime();
  auto rel_res = [&](const std::vector<double>& x, std::vector<double>& r) {
    return relative_residual(A, b_local, x, r);
  };
  // The residual the caller GATES on: over the rows of set_residual_rows() only (the Stokes solver
  // excludes its Dirichlet rows, which are identity rows carrying the boundary data), relative to the norm of the
  // rhs over the same rows. Without a mask it is rel_res. Used by the Krylov fallback only: the refinement and the
  // reported stats keep their definitions.
  auto gate_res = [&](const std::vector<double>& x, std::vector<double>& r) {
    if (row_mask_.empty()) return rel_res(x, r);
    A.mult(x.data(), r.data());
    double loc[2] = {0, 0}, glob[2];
    for (int i = 0; i < A.local_rows; ++i) {
      if (!row_mask_[i]) continue;
      const double ri = b_local[i] - r[i];
      loc[0] += ri * ri;
      loc[1] += b_local[i] * b_local[i];
    }
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, A.comm);
    return glob[1] > 0 ? std::sqrt(glob[0] / glob[1]) : std::sqrt(glob[0]);
  };
  std::vector<double> x = raw_solve(b_local), r(A.local_rows);
  double res = rel_res(x, r);
  stats_.rel_residual_before = res;
  stats_.refine_steps = 0;
  // iterative refinement with the true matrix; stops on stagnation (a step that gains less than 10 %) once
  // min_refine_ is done, and keeps the best iterate
  std::vector<double> best = x;
  double best_res = res;
  while (std::isfinite(res) && (res > refine_rtol_ || stats_.refine_steps < min_refine_ || to_stagnation_) &&
         stats_.refine_steps < max_refine_ && res > 0) {
    std::vector<double> dx = raw_solve(r);
    for (int i = 0; i < A.local_rows; ++i) x[i] += dx[i];
    const double prev = res;
    res = rel_res(x, r);
    ++stats_.refine_steps;
    if (std::isfinite(res) && res < best_res) {
      best = x;
      best_res = res;
    }
    if (stats_.refine_steps >= min_refine_ && !(res < 0.9 * prev)) break;  // stagnation: < 10 % gained
  }
  // When the plain refinement (a Richardson iteration with the LU) leaves the GATED residual (gate_res: over the rows
  // of set_residual_rows) above the target -- a poor static-pivoting factorization of a 3D saddle point (STRUMPACK-MPI,
  // MC64; observed on a 3D traction problem at 0.86 M unknowns: plain solve 0.14, refinement diverging) -- restarted
  // GMRES right-preconditioned by the same LU, from the best iterate. Opt-in (set_krylov_fallback; the 3D API turns it
  // on). The gated residual decides, not the one over all rows that the refinement stops on: it runs although the
  // refinement reached its target when the gated rows did not, and it does not run when they did.
  stats_.krylov_iterations = 0;
  stats_.krylov_fallback = false;
  stats_.matching = chain_[current_].matching != MatchingJob::None;
  double best_gate = std::numeric_limits<double>::infinity();
  if (krylov_fallback_ && std::isfinite(best_res) && max_refine_ > 0) best_gate = gate_res(best, r);
  if (krylov_fallback_ && std::isfinite(best_res) && best_gate > refine_rtol_ && max_refine_ > 0) {
    stats_.krylov_fallback = true;
    const int n = A.local_rows, m = 40, maxit = 400;
    auto dot = [&](const std::vector<double>& a, const std::vector<double>& c) {
      double l = 0, g = 0;
      for (int i = 0; i < n; ++i) l += a[i] * c[i];
      MPI_Allreduce(&l, &g, 1, MPI_DOUBLE, MPI_SUM, A.comm);
      return g;
    };
    // stopping norm: the gated residual's denominator (rows of the mask; all rows without one)
    double bn2 = 0;
    for (int i = 0; i < n; ++i)
      if (row_mask_.empty() || row_mask_[i]) bn2 += b_local[i] * b_local[i];
    double bn2g = 0;
    MPI_Allreduce(&bn2, &bn2g, 1, MPI_DOUBLE, MPI_SUM, A.comm);
    const double bnorm = std::sqrt(bn2g);
    std::vector<double> xg = best, w(n);
    int it = 0;
    while (it < maxit && bnorm > 0) {
      A.mult(xg.data(), w.data());
      std::vector<double> r0(n);
      for (int i = 0; i < n; ++i) r0[i] = b_local[i] - w[i];
      const double beta = std::sqrt(dot(r0, r0));
      if (!(gate_res(xg, w) > 0.5 * refine_rtol_)) break;  // the gated residual (not the full one) decides
      std::vector<std::vector<double>> V(1, r0), Z;
      for (int i = 0; i < n; ++i) V[0][i] /= beta;
      std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
      std::vector<double> cs(m), sn(m), g(m + 1, 0.0);
      g[0] = beta;
      int k = 0;
      for (; k < m && it < maxit; ++k, ++it) {
        Z.push_back(raw_solve(V[k]));
        A.mult(Z[k].data(), w.data());
        for (int j = 0; j <= k; ++j) {  // modified Gram-Schmidt
          H[j][k] = dot(w, V[j]);
          for (int i = 0; i < n; ++i) w[i] -= H[j][k] * V[j][i];
        }
        H[k + 1][k] = std::sqrt(dot(w, w));
        for (int j = 0; j < k; ++j) {  // previous Givens rotations
          const double t = cs[j] * H[j][k] + sn[j] * H[j + 1][k];
          H[j + 1][k] = -sn[j] * H[j][k] + cs[j] * H[j + 1][k];
          H[j][k] = t;
        }
        const double den = std::hypot(H[k][k], H[k + 1][k]);
        cs[k] = den > 0 ? H[k][k] / den : 1.0;
        sn[k] = den > 0 ? H[k + 1][k] / den : 0.0;
        const double hk1 = H[k + 1][k];
        H[k][k] = den;
        g[k + 1] = -sn[k] * g[k];
        g[k] = cs[k] * g[k];
        if (!(hk1 > 0)) {
          ++k;
          ++it;
          break;
        }
        V.emplace_back(n);
        for (int i = 0; i < n; ++i) V[k + 1][i] = w[i] / hk1;
        if (std::fabs(g[k + 1]) / bnorm <= 0.5 * refine_rtol_) {
          ++k;
          ++it;
          break;
        }
      }
      std::vector<double> y(k, 0.0);  // back substitution H y = g
      for (int j = k - 1; j >= 0; --j) {
        double t = g[j];
        for (int l = j + 1; l < k; ++l) t -= H[j][l] * y[l];
        y[j] = H[j][j] != 0 ? t / H[j][j] : 0.0;
      }
      for (int j = 0; j < k; ++j)
        for (int i = 0; i < n; ++i) xg[i] += y[j] * Z[j][i];
    }
    stats_.krylov_iterations = it;
    const double rg_gate = gate_res(xg, r);
    const double rg = rel_res(xg, r);
    if (std::isfinite(rg_gate) && rg_gate < best_gate) {
      best = xg;
      best_res = rg;
    }
  }
  x = std::move(best);
  res = best_res;
  stats_.rel_residual = res;
  stats_.solve_seconds = wtime() - t0;
  return x;
}

namespace {
bool all_finite(const std::vector<double>& x) {
  for (double v : x)
    if (!std::isfinite(v)) return false;
  return true;
}
std::string g3(double v) {
  char b[40];
  std::snprintf(b, sizeof b, "%.3g", v);
  return b;
}
}  // namespace

std::vector<double> SparseLU::solve(const std::vector<double>& b_local) {
  AXMP_REQUIRE(A_ != nullptr && current_ >= 0,
               "SparseLU::solve: no factorization (factorize was not called, or the last one failed)", "",
               kNoFactorizationHint);
  const DistCsr& A = *A_;
  AXMP_REQUIRE(static_cast<int>(b_local.size()) == A.local_rows, "SparseLU::solve: wrong local rhs size");
  ThreadGuard tg(threads_);
  const int me = rank_of(A.comm), nstages = static_cast<int>(chain_.size());
  const LUAcceptance::Kind kind = acceptance_.kind;
  const std::string measure_name = !acceptance_.name.empty()               ? acceptance_.name
                                   : kind == LUAcceptance::Kind::Verdict ? "verdict"
                                                                         : "relative residual";
  double perturbation = 0;
  bool rank1_flip = false;
  std::function<void(int, std::vector<double>&)> iterate_hook;
  {
    ChainHooks& h = chain_hooks();
    std::lock_guard<std::mutex> lock(h.mutex);
    perturbation = h.perturbation;
    rank1_flip = h.rank1_flip;
    iterate_hook = h.iterate_hook;
  }
  if (!attempts_from_factorize_) attempts_.clear();  // the first solve also lists what factorize() passed over
  attempts_from_factorize_ = false;
  accepted_stage_ = -1;
  solution_stage_ = -1;

  // The walk. Every rank is in the same stage at every step: a stage is left on rank 0's decision (broadcast), or
  // on an error that every rank throws.
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> x, best;
  LUStats best_stats;
  int best_stage = -1;
  double best_measure = inf, best_backward_error = inf;
  bool have_iterate = false, accepted = false;
  std::exception_ptr solve_error;
  while (true) {
    const int k = current_;
    // The held stage's own factorization stats: after a solve that no stage accepted, stats_ describes the stage
    // whose iterate was returned, which need not be the one held.
    stats_ = stage_stats_;
    LUAttempt at;
    at.stage = chain_[k].name();
    at.factor_nnz = stats_.factor_nnz;
    at.index_bits = stats_.index_bits;
    bool iterate = true;
    try {
      x = solve_stage(b_local);
    } catch (const Error& e) {  // a backend's solve failed: thrown on every rank (collective_check)
      iterate = false;
      at.outcome = "threw";
      at.reason = e.what();
      solve_error = std::current_exception();
    }
    if (iterate) {
      const bool perturb = k == 0 && perturbation != 0;
      if (perturb || iterate_hook) {  // test hooks: the numbers below are the changed iterate's
        if (perturb)
          for (double& v : x) v *= 1.0 + perturbation;
        if (iterate_hook) iterate_hook(k, x);
        std::vector<double> r(A.local_rows);
        stats_.rel_residual = relative_residual(A, b_local, x, r);
      }
      // what is recorded of this iterate whatever the acceptance: one more pass over the matrix and four collectives
      IterateMeasures im = iterate_measures(A, x.data(), b_local.data(), held_blocks_, row_mask_);
      // the acceptance measure: the caller's own verdict, else the refined relative residual over all rows
      double measure = kind == LUAcceptance::Kind::Verdict ? acceptance_.verdict(x) : stats_.rel_residual;
      int finite = 1;
      if (kind != LUAcceptance::Kind::None) {
        const int mine = all_finite(x) ? 1 : 0;
        MPI_Allreduce(&mine, &finite, 1, MPI_INT, MPI_MIN, A.comm);
      }
      auto meets = [&] { return finite != 0 && std::isfinite(measure) && measure <= acceptance_.bound; };
      if (rank1_flip && me == 1 && kind != LUAcceptance::Kind::None) {
        // test hook: this rank's inputs of the decision land on the other side of the bound; rank 0's are used
        const bool would = meets();
        finite = would ? 0 : 1;
        measure = would ? std::numeric_limits<double>::quiet_NaN() : 0.0;
      }
      const bool forced_miss = k == 0 && take_forced_outcome(false) == LUForcedOutcome::MissedTarget;
      // Rank 0 decides and every rank follows: bit 0 accepted, bit 1 the iterate is finite, bit 2 the test hook.
      int decision = 0;
      if (me == 0)
        decision = ((kind == LUAcceptance::Kind::None || meets()) && !forced_miss ? 1 : 0) | (finite != 0 ? 2 : 0) |
                   (forced_miss ? 4 : 0);
      MPI_Bcast(&decision, 1, MPI_INT, 0, A.comm);
      // the record is rank 0's on every rank (the verdict and the times are per rank)
      double rec[8] = {measure,          stats_.rel_residual_before, stats_.rel_residual,   im.gate_residual,
                       im.backward_error, stats_.analyze_seconds,     stats_.factor_seconds, stats_.solve_seconds};
      MPI_Bcast(rec, 8, MPI_DOUBLE, 0, A.comm);
      at.measure = rec[0];
      at.bound = kind == LUAcceptance::Kind::None ? -1 : acceptance_.bound;
      at.rel_residual_before = rec[1];
      at.rel_residual = rec[2];
      at.gate_residual = rec[3];
      at.backward_error = rec[4];
      at.analyze_seconds = rec[5];
      at.factor_seconds = rec[6];
      at.solve_seconds = rec[7];
      at.constraint_ratios = std::move(im.constraint_ratios);
      at.refine_steps = stats_.refine_steps;
      at.krylov_iterations = stats_.krylov_iterations;
      stats_.measure = at.measure;
      stats_.backward_error = at.backward_error;
      const bool accept = (decision & 1) != 0;
      at.outcome = accept ? "accepted" : "missed_target";
      if ((decision & 4) != 0) at.reason = "test hook";
      else if (kind == LUAcceptance::Kind::None) at.reason = "no acceptance test";
      else if ((decision & 2) == 0) at.reason = "non-finite solution";
      else at.reason = measure_name + " " + g3(at.measure) + (accept ? " <= " : " > ") + g3(at.bound);
      if (accept) {
        accepted = true;
        accepted_stage_ = k;
      } else {
        // The best iterate so far by the acceptance measure; ties: the backward error. A non-finite iterate (the
        // measure need not see it: a verdict may not read the bad entries) and a NaN measure count as +infinity.
        const double m = (decision & 2) == 0 || std::isnan(at.measure) ? inf : at.measure;
        const double be = std::isnan(at.backward_error) ? inf : at.backward_error;
        if (!have_iterate || m < best_measure || (m == best_measure && be < best_backward_error)) {
          best = std::move(x);
          best_stats = stats_;
          best_stage = k;
          best_measure = m;
          best_backward_error = be;
        }
        have_iterate = true;
      }
    }
    attempts_.push_back(std::move(at));
    if (accepted) break;
    // The next stage that factorizes. The current stage's backend object is released first, and only when a later
    // stage can be tried at all (in this build, and within its size cap): a chain that ends here keeps its
    // factorization for the next solve.
    int next = -1;
    for (int j = k + 1; j < nstages && next < 0; ++j) {
      const bool capped = chain_[j].max_n > 0 && A.n_global > chain_[j].max_n;
      if (lu_backend_available(chain_[j].backend) && !capped && current_ >= 0) {
        release_backend();
        current_ = -1;
      }
      if (try_stage(A, j, nullptr)) next = j;
    }
    if (next < 0) break;
    current_ = next;
  }
  if (!accepted) {
    stats_.attempts = static_cast<int>(attempts_.size());
    if (!have_iterate) {
      if (nstages == 1 && solve_error) std::rethrow_exception(solve_error);  // a single attempt: its own error
      throw Error(chain_failure(attempts_));
    }
    x = std::move(best);
    stats_ = best_stats;  // the stage whose iterate is returned
    solution_stage_ = best_stage;
  } else {
    solution_stage_ = accepted_stage_;
  }
  stats_.attempts = static_cast<int>(attempts_.size());
  if (max_refine_ > 0 && throw_on_miss_)  // collective: the residual is the same on every rank
    AXMP_REQUIRE(std::isfinite(stats_.rel_residual) && stats_.rel_residual <= refine_rtol_,
                strcat("sparse LU (", stats_.backend, ") missed its accuracy target: relative residual ",
                       stats_.rel_residual, " > ", refine_rtol_, " after ", stats_.refine_steps,
                       " refinement steps (plain solve ", stats_.rel_residual_before, ")"),
                "", "Likely causes: a singular operator (an undetected nullspace, incompatible data), or an ill-scaled "
                    "one (large coefficient contrast, high aspect ratio). Check the boundary conditions; for a scaled "
                    "problem loosen the target (SparseLU::set_refinement; in the Stokes API: direct_tol).");
  return x;
}

void lu_count_solve(LUCounts& c, const std::vector<LUAttempt>& attempts) {
  const bool escalated = attempts.size() > 1;
  bool accepted = false;
  ++c.solves;
  c.attempts += static_cast<long>(attempts.size());
  if (escalated) ++c.escalated;
  c.escalated_final = escalated ? 1 : 0;
  for (const LUAttempt& a : attempts) {
    if (a.outcome == "accepted") {
      accepted = true;
      ++c.accepted_by[a.stage];
    } else if (a.outcome == "threw" || a.outcome == "missed_target" || a.outcome == "refused") {
      ++c.rejected;
      c.rejected_seconds += a.analyze_seconds + a.factor_seconds + a.solve_seconds;
    }
  }
  if (!accepted) ++c.failed;
}

LUCounts& operator+=(LUCounts& a, const LUCounts& b) {
  a.solves += b.solves;
  a.escalated += b.escalated;
  a.escalated_final += b.escalated_final;
  a.attempts += b.attempts;
  a.rejected += b.rejected;
  a.failed += b.failed;
  for (const auto& kv : b.accepted_by) a.accepted_by[kv.first] += kv.second;
  a.rejected_seconds += b.rejected_seconds;
  return a;
}

}  // namespace axmp::fluids
