// fluids sparse LU chain at its callers: the acceptance each caller gives its chain and the records each caller
// keeps, through the C++ solvers, on 1 and 2 ranks.
//
// Registered on 1 and 2 ranks with a TIMEOUT. A chain's decisions are rank 0's, and the projection's verdict is a
// collective function of the stage's solution, called on every rank: every record compared here must be the same on
// every rank, and a rank sent to another stage than rank 0 waits in a collective call (the TIMEOUT ends the test).
// The escalations are forced with the test hooks of sparse_lu.hpp and the test-only default chain [eigen, eigen], so
// this file runs in every build (with or without STRUMPACK). The Python side (tests/test_fluids_lu_chain.py) holds
// the per-caller tests of every entry point, the run counts, the aliases and the rescue cases.
//
//   1. The initial projection's acceptance is its own gate (a Verdict): the accepted attempt's measure is
//      projection_residual bit for bit and its bound kProjectionResidualMax; a stage-0 solution perturbed by 1e-8 is
//      rejected with the measure the perturbation is known to leave (8.06e-10 on the 8 x 8 square), and the next
//      stage is accepted; the projection's attempts are its own, not the step counts'.
//   2. Steady Navier-Stokes: the step solves are accepted on their relative residual at the final solve's bound
//      (max(newton_rtol, kFinalSolveResidualMax)); a refused stage 0 of the dual-norm mass LUs and of the first step
//      is escalated and recorded per step (NonlinearStep::lu_attempts, lu_stage) and in the counts of the call.
//   3. Stokes (solve_direct): accepted on its relative residual over all rows at direct_tol (today's retry trigger);
//      an escalation names the stage that produced the solution while `solver` stays the configured backend.
//   4. The Stokes direct solve's chain: the default list of its first matching (with matching: then MC64 job 2, then
//      Eigen; without: then with matching, then Eigen), whose Eigen retry stage is capped by the mesh dimension
//      (kLUEigenRetryMaxN3D on a 3D mesh, kLUEigenRetryMaxN on a 2D one); an explicit chain
//      (StokesOptions::lu_chain) is used as given, a stage without matching in 2D included.
//   5. STRUMPACK builds: the Stokes direct solve walks its default chain in order, recorded in its diagnostics: on the
//      2D lid-driven square [strumpack:mc64-5, strumpack:mc64-2, eigen], on a 3D cube without matching first
//      [strumpack:none, strumpack:mc64-5, eigen] (the iterate hook makes stages 0 and 1 miss direct_tol).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fluids/navier_stokes.hpp"
#include "axionmultiphysics/fluids/sparse_lu.hpp"
#include "axionmultiphysics/fluids/stokes.hpp"

using namespace axmp::fluids;
static int failures = 0;
static bool rank0 = true;
#define CHECK(cond, ...)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      std::printf("FAIL (rank %d): ", rank0 ? 0 : 1);        \
      std::printf(__VA_ARGS__);                              \
      std::printf("\n");                                     \
      ++failures;                                            \
    }                                                        \
  } while (0)

static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
static bool bitwise(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
static std::string hex(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%a", v);
  return b;
}

// true on every rank iff it holds on every rank
static bool on_all(bool mine) {
  int m = mine ? 1 : 0, all = 0;
  MPI_Allreduce(&m, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return all == 1;
}
// true on every rank iff every rank passed the same bytes
static bool same_on_all_ranks(const std::string& s) {
  int len = static_cast<int>(s.size()), lo = 0, hi = 0;
  MPI_Allreduce(&len, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&len, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (lo != hi) return false;
  std::string r0 = s;
  MPI_Bcast(r0.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
  return on_all(r0 == s);
}

// the fields of the attempts that every rank must hold alike (the times are rank 0's too, but are not compared:
// they are not part of what the caller decides on)
static std::string records(const std::vector<LUAttempt>& at) {
  std::string s;
  for (const auto& a : at)
    s += a.stage + "|" + a.outcome + "|" + a.reason + "|" + hex(a.measure) + "|" + hex(a.bound) + "|" +
         hex(a.rel_residual) + "|" + hex(a.backward_error) + "\n";
  return s;
}
static std::string outcomes(const std::vector<LUAttempt>& at) {
  std::string s;
  for (const auto& a : at) s += (s.empty() ? "" : ",") + a.stage + ":" + a.outcome;
  return s;
}
static std::string counts(const LUCounts& c) {
  std::string s = "solves " + std::to_string(c.solves) + " escalated " + std::to_string(c.escalated) + " final " +
                  std::to_string(c.escalated_final) + " attempts " + std::to_string(c.attempts) + " rejected " +
                  std::to_string(c.rejected) + " failed " + std::to_string(c.failed) + " accepted_by {";
  for (const auto& kv : c.accepted_by) s += kv.first + ":" + std::to_string(kv.second) + " ";
  return s + "}";
}

// The test hooks are process-wide: put them back when a section ends, however it ends.
struct HookReset {
  ~HookReset() {
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    set_lu_solution_perturbation_for_testing(0.0);
    set_default_lu_chain_for_testing({});
    set_lu_iterate_hook_for_testing({});
  }
};
static std::vector<LUStage> two_eigen() {
  LUStage e;
  e.backend = LUBackend::EigenLU;
  return {e, e};
}

// ------------------------------------------------------------------------------------------------ 1
// The projection of the field of test_fluids_trust.cpp's q5_projection (not divergence free, zero on the boundary) on
// the 8 x 8 unit square (pinned pressure: Eigen's nullspace treatment), as a transient solver projects its initial
// state. References: the gate's two evaluations of one function on the same (du, dp) agree bit for bit; the measure
// of a solution scaled by 1 + 1e-8 is K_8 1e-8 = 8.060e-10 on this mesh on any number of ranks (test_fluids_trust.cpp,
// projection_measure; tests/test_fluids_trust.py, K_8), held here to 5 %.
static void projection() {
  HookReset reset;
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(8, 8, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ParMesh pm(MPI_COMM_WORLD, sm);
  mfem::ConstantCoefficient mu(0.05);
  StokesBoundary b;
  b.dirichlet(1).dirichlet(2).dirichlet(3).dirichlet(4);
  NavierStokesOptions o;
  o.stokes.nu = &mu;
  o.stokes.lu = LUBackend::EigenLU;
  o.rho = 1.25;
  o.nonlinear.strategy = NonlinearStrategy::TransientNewton;
  NavierStokesSolver S(pm, o, b);
  mfem::VectorFunctionCoefficient u0(2, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(2);
    const double s = x(0) * (1 - x(0)) * x(1) * (1 - x(1));
    v(0) = s * (1 + x(0));
    v(1) = s * x(1);
  });
  mfem::ParGridFunction g(&S.velocity_space());
  g.ProjectCoefficient(u0);
  mfem::Vector uI(S.velocity_size());
  g.GetTrueDofs(uI);
  const double k8 = 0.0806;

  // (a) the default chain of lu = eigen: one stage, accepted on the projection's own measure
  {
    const bool ok = S.project_divergence_free(uI);
    const NavierStokesDiagnostics& d = S.diagnostics();
    const auto& at = d.projection_lu_attempts;
    CHECK(ok && d.projection_converged, "projection (a): refused (%.3e)", d.projection_residual);
    CHECK(at.size() == 1, "projection (a): %zu attempts recorded ([%s]), expected 1", at.size(), outcomes(at).c_str());
    if (at.size() == 1) {
      CHECK(at[0].stage == "eigen" && at[0].outcome == "accepted",
            "projection (a): attempt [%s], expected eigen:accepted", outcomes(at).c_str());
      CHECK(bitwise(at[0].measure, d.projection_residual), "projection (a): the accepted attempt's measure %a is not "
            "projection_residual %a: the chain accepted on another measure than the gate", at[0].measure,
            d.projection_residual);
      CHECK(at[0].bound == kProjectionResidualMax, "projection (a): bound %.3g, expected kProjectionResidualMax %.3g",
            at[0].bound, kProjectionResidualMax);
      CHECK(has(at[0].reason, "projection residual"), "projection (a): the reason '%s' does not name the projection "
            "residual", at[0].reason.c_str());
    }
    CHECK(d.lu_counts.solves == 0 && d.lu_counts.attempts == 0, "projection (a): the projection is counted with the "
          "step solves (%s)", counts(d.lu_counts).c_str());
    CHECK(same_on_all_ranks(records(at)), "projection (a): the attempt records differ between the ranks");
  }
  // (b) [eigen, eigen], stage 0 forced to missed_target: both stages' measures are the gate of their (identical)
  // solutions, and the second is accepted
  {
    set_default_lu_chain_for_testing(two_eigen());
    set_lu_forced_outcome_for_testing(LUForcedOutcome::MissedTarget, 1);
    const bool ok = S.project_divergence_free(uI);
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    const NavierStokesDiagnostics& d = S.diagnostics();
    const auto& at = d.projection_lu_attempts;
    CHECK(ok, "projection (b): refused (%.3e)", d.projection_residual);
    CHECK(at.size() == 2 && at[0].outcome == "missed_target" && at[0].reason == "test hook" &&
              at[1].outcome == "accepted",
          "projection (b): attempts [%s], expected [eigen:missed_target (test hook), eigen:accepted]",
          outcomes(at).c_str());
    if (at.size() == 2) {
      CHECK(bitwise(at[1].measure, d.projection_residual) && bitwise(at[0].measure, at[1].measure),
            "projection (b): measures %a (stage 0) and %a (stage 1) against projection_residual %a: each stage's "
            "measure must be the gate of its solution (the two solutions are the same arithmetic)", at[0].measure,
            at[1].measure, d.projection_residual);
      CHECK(at[0].bound == kProjectionResidualMax && at[1].bound == kProjectionResidualMax,
            "projection (b): bounds %.3g and %.3g, expected %.3g", at[0].bound, at[1].bound, kProjectionResidualMax);
    }
    CHECK(d.lu_counts.solves == 0 && d.lu_counts.escalated == 0, "projection (b): the projection's escalation is "
          "booked to the step counts (%s)", counts(d.lu_counts).c_str());
    CHECK(same_on_all_ranks(records(at)), "projection (b): the attempt records differ between the ranks");
  }
  // (c) [eigen, eigen], stage 0's solution scaled by 1 + 1e-8: rejected on the gate with the known measure; the next
  // stage is accepted at round-off
  {
    set_lu_solution_perturbation_for_testing(1e-8);
    const bool ok = S.project_divergence_free(uI);
    set_lu_solution_perturbation_for_testing(0.0);
    const NavierStokesDiagnostics& d = S.diagnostics();
    const auto& at = d.projection_lu_attempts;
    CHECK(ok && d.projection_residual < 1e-15, "projection (c): refused, or not at round-off (%.3e)",
          d.projection_residual);
    CHECK(at.size() == 2 && at[0].outcome == "missed_target" && at[1].outcome == "accepted",
          "projection (c): attempts [%s], expected [eigen:missed_target, eigen:accepted]", outcomes(at).c_str());
    if (at.size() == 2) {
      CHECK(std::fabs(at[0].measure - k8 * 1e-8) < 0.05 * k8 * 1e-8, "projection (c): stage 0's measure %.4e, expected "
            "the gate of a solution perturbed by 1e-8, %.4e (a relative residual would be about 1e-8)", at[0].measure,
            k8 * 1e-8);
      CHECK(at[0].bound == kProjectionResidualMax && has(at[0].reason, "projection residual") &&
                has(at[0].reason, ">"),
            "projection (c): stage 0's bound %.3g or reason '%s'", at[0].bound, at[0].reason.c_str());
      CHECK(bitwise(at[1].measure, d.projection_residual),
            "projection (c): accepted measure %a, projection_residual %a", at[1].measure, d.projection_residual);
    }
    CHECK(same_on_all_ranks(records(at)), "projection (c): the attempt records differ between the ranks");
    if (rank0 && at.size() == 2)
      std::printf("projection: stage 0 perturbed by 1e-8: measure %.4e (%s); stage 1 %.3e\n", at[0].measure,
                  at[0].reason.c_str(), at[1].measure);
  }
}

// ------------------------------------------------------------------------------------------------ 2
// The lid-driven cavity of test_fluids_trust.cpp's final_solve_gate (8 x 8, Re small): steady Newton with a Stokes
// start. Hook: [eigen, eigen], and the next three factorizations of a stage 0 refused. The mass LUs (velocity and
// pressure) are factorized when the solver is built, before any step can be solved (the lift's dual norm needs them),
// so the three refusals are those of the two mass LUs and of the first step solve (the Stokes start). Reference: the
// hook's count.
static void steady_navier_stokes() {
  HookReset reset;
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(8, 8, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ParMesh pm(MPI_COMM_WORLD, sm);
  mfem::ConstantCoefficient mu(0.05);
  mfem::VectorFunctionCoefficient lid(2, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(2);
    v(0) = 16.0 * x(0) * x(0) * (1 - x(0)) * (1 - x(0));
    v(1) = 0.0;
  });
  StokesBoundary b;
  b.dirichlet(1).dirichlet(2).dirichlet(3, &lid).dirichlet(4);
  NavierStokesOptions o;
  o.stokes.nu = &mu;
  o.stokes.lu = LUBackend::EigenLU;
  // (a) no hook: one accepted attempt per step, on the relative residual at the final solve's bound
  {
    NavierStokesSolver S(pm, o, b);
    const NavierStokesDiagnostics d = S.solve();
    const double gate = std::max(o.nonlinear.newton_rtol, kFinalSolveResidualMax);
    CHECK(d.converged, "steady NS (a): not converged (%s)", d.failure.c_str());
    const auto& at = d.linear.lu_attempts;
    CHECK(at.size() == 1 && at[0].outcome == "accepted", "steady NS (a): the last step's attempts [%s], expected one "
          "accepted", outcomes(at).c_str());
    if (at.size() == 1) {
      CHECK(at[0].bound == gate && bitwise(at[0].measure, d.linear.lu.rel_residual) &&
                has(at[0].reason, "relative residual"),
            "steady NS (a): the step's acceptance is measure %a bound %.3g ('%s'), expected the relative residual %a "
            "at max(newton_rtol, kFinalSolveResidualMax) = %.3g", at[0].measure, at[0].bound, at[0].reason.c_str(),
            d.linear.lu.rel_residual, gate);
    }
    bool every = !d.history.empty();
    for (const auto& h : d.history) every = every && h.lu_attempts == 1 && h.lu_stage == "eigen";
    CHECK(every, "steady NS (a): a history entry records other than one attempt of stage eigen");
    const long n = static_cast<long>(d.history.size());
    CHECK(d.lu_counts.solves == n && d.lu_counts.escalated == 0 && d.lu_counts.attempts == n &&
              d.lu_counts.rejected == 0 && d.lu_counts.failed == 0 && d.mass_lu_escalated == 0,
          "steady NS (a): counts %s, mass escalated %ld, for %ld steps without a rejection",
          counts(d.lu_counts).c_str(), d.mass_lu_escalated, n);
    CHECK(d.linear.lu_stage == "eigen" && d.linear.lu.backend == "eigen" && d.linear.solver == "direct:eigen",
          "steady NS (a): lu_stage '%s', backend '%s', solver '%s'", d.linear.lu_stage.c_str(),
          d.linear.lu.backend.c_str(), d.linear.solver.c_str());
  }
  // (b) [eigen, eigen], three refusals: the two mass LUs and the Stokes start escalate
  {
    set_default_lu_chain_for_testing(two_eigen());
    set_lu_forced_outcome_for_testing(LUForcedOutcome::Refused, 3);
    NavierStokesSolver S(pm, o, b);
    const NavierStokesDiagnostics d = S.solve();
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    CHECK(d.converged, "steady NS (b): not converged (%s)", d.failure.c_str());
    CHECK(d.failure.empty(), "steady NS (b): a converged solve keeps a failure text '%s'", d.failure.c_str());
    const long n = static_cast<long>(d.history.size());
    std::string per_step;
    for (const auto& h : d.history) per_step += std::to_string(h.lu_attempts) + h.lu_stage + " ";
    bool want = n >= 2 && d.history[0].kind == "stokes" && d.history[0].lu_attempts == 2;
    for (long k = 1; k < n; ++k) want = want && d.history[k].lu_attempts == 1;
    for (const auto& h : d.history) want = want && h.lu_stage == "eigen";
    CHECK(want, "steady NS (b): per-step attempts [%s], expected 2 for the Stokes start and 1 for every later step",
          per_step.c_str());
    const LUCounts& c = d.lu_counts;
    CHECK(c.solves == n && c.escalated == 1 && c.escalated_final == 0 && c.attempts == n + 1 && c.rejected == 1 &&
              c.failed == 0 && c.accepted_by.size() == 1 && c.accepted_by.count("eigen") == 1 &&
              c.accepted_by.at("eigen") == n,
          "steady NS (b): counts %s for %ld steps, one of them escalated once (not the last)", counts(c).c_str(), n);
    CHECK(d.mass_lu_escalated == 2, "steady NS (b): %ld mass solves escalated, expected 2 (velocity and pressure mass)",
          d.mass_lu_escalated);
    CHECK(same_on_all_ranks(per_step + counts(c) + std::to_string(d.mass_lu_escalated)),
          "steady NS (b): the records differ between the ranks");
    if (rank0) std::printf("steady NS: per-step attempts [%s]; %s; mass escalated %ld\n", per_step.c_str(),
                           counts(c).c_str(), d.mass_lu_escalated);
  }
}

// ------------------------------------------------------------------------------------------------ 3
// Stokes in the lid-driven 8 x 8 unit square (an enclosed flow: Eigen pins a pressure).
static void stokes_direct() {
  HookReset reset;
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(8, 8, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ParMesh pm(MPI_COMM_WORLD, sm);
  mfem::ConstantCoefficient nu(1.0);
  mfem::VectorFunctionCoefficient lid(2, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(2);
    v(0) = 16.0 * x(0) * x(0) * (1 - x(0)) * (1 - x(0));
    v(1) = 0.0;
  });
  StokesBoundary b;
  b.dirichlet(1).dirichlet(2).dirichlet(3, &lid).dirichlet(4);
  StokesOptions o;
  o.nu = &nu;
  o.lu = LUBackend::EigenLU;
  {  // (a) no hook: accepted on the relative residual over all rows at direct_tol
    StokesSolver S(pm, o, b);
    const StokesDiagnostics d = S.solve();
    CHECK(d.converged, "Stokes (a): not converged");
    const auto& at = d.lu_attempts;
    CHECK(at.size() == 1 && at[0].outcome == "accepted" && d.lu_stage == "eigen",
          "Stokes (a): attempts [%s], lu_stage '%s'", outcomes(at).c_str(), d.lu_stage.c_str());
    if (at.size() == 1)
      CHECK(at[0].bound == o.direct_tol && bitwise(at[0].measure, d.lu.rel_residual),
            "Stokes (a): measure %a bound %.3g, expected the relative residual over all rows %a at direct_tol %.3g",
            at[0].measure, at[0].bound, d.lu.rel_residual, o.direct_tol);
    CHECK(same_on_all_ranks(records(at)), "Stokes (a): the attempt records differ between the ranks");
  }
  {  // (b) [eigen, eigen], stage 0 refused: the solution is stage 1's, and the configured backend stays the label
    set_default_lu_chain_for_testing(two_eigen());
    set_lu_forced_outcome_for_testing(LUForcedOutcome::Refused, 1);
    StokesSolver S(pm, o, b);
    const StokesDiagnostics d = S.solve();
    set_lu_forced_outcome_for_testing(LUForcedOutcome::None, 0);
    const auto& at = d.lu_attempts;
    CHECK(d.converged, "Stokes (b): not converged");
    CHECK(at.size() == 2 && at[0].outcome == "refused" && at[0].reason == "test hook" && at[1].outcome == "accepted",
          "Stokes (b): attempts [%s], expected [eigen:refused (test hook), eigen:accepted]", outcomes(at).c_str());
    CHECK(d.lu_stage == "eigen" && d.lu.backend == "eigen" && d.solver == "direct:eigen",
          "Stokes (b): lu_stage '%s', backend '%s', solver '%s'", d.lu_stage.c_str(), d.lu.backend.c_str(),
          d.solver.c_str());
    CHECK(same_on_all_ranks(records(at)), "Stokes (b): the attempt records differ between the ranks");
  }
}

// ------------------------------------------------------------------------------------------------ 4
// StokesSolver::lu_chain(), the chain solve_direct factorizes with. Reference: the two caps of
// sparse_lu.hpp, by dimension; the stage lists are default_chain's (test_fluids_lu_chain section 1): with matching
// first (the 2D solvers' default) the retry is MC64 job 2, never the stage without matching; without matching first
// (the 3D default) the retry is with matching.
static void stokes_chain_caps() {
  HookReset reset;
  mfem::ConstantCoefficient nu(1.0);
  auto caps = [](const std::vector<LUStage>& c) {
    std::string s;
    for (const auto& st : c) s += (s.empty() ? "" : ",") + st.name() + "/" + std::to_string(st.max_n);
    return s;
  };
  const std::string c2 = std::to_string(kLUEigenRetryMaxN), c3 = std::to_string(kLUEigenRetryMaxN3D);
  for (int dim : {2, 3}) {
    mfem::Mesh sm = dim == 2 ? mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::TRIANGLE, true, 1.0, 1.0)
                             : mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::TETRAHEDRON, 1.0, 1.0, 1.0);
    mfem::ParMesh pm(MPI_COMM_WORLD, sm);
    StokesBoundary b;
    for (int a = 1; a <= (dim == 2 ? 4 : 6); ++a) b.dirichlet(a);
    const std::string cap = dim == 3 ? c3 : c2;
    for (bool matching : {true, false}) {
      StokesOptions o;
      o.nu = &nu;
      o.lu = LUBackend::StrumpackLU;  // the list only: availability is checked when a stage is tried
      o.lu_matching = matching;
      StokesSolver S(pm, o, b);
      const std::string got = caps(S.lu_chain());
      // 2D with matching: job 5, job 2, Eigen. 3D with matching keeps the retry measured on 3D saddle points (the
      // stage without matching), not the 2D list's job 2. Without matching: the 3D list, in 2D as well.
      const std::string want = matching ? std::string("strumpack:mc64-5/0,strumpack:") + (dim == 3 ? "none" : "mc64-2") +
                                              "/0,eigen/" + cap
                                        : "strumpack:none/0,strumpack:mc64-5/0,eigen/" + cap;
      CHECK(got == want, "%dD Stokes, lu_matching %d: chain [%s], expected [%s]", dim, matching ? 1 : 0, got.c_str(),
            want.c_str());
    }
    StokesOptions o;
    o.nu = &nu;
    LUStage e;
    e.backend = LUBackend::EigenLU;
    e.max_n = 7;
    o.lu_chain = {e, e};
    StokesSolver S(pm, o, b);
    CHECK(caps(S.lu_chain()) == "eigen/7,eigen/7", "%dD Stokes, explicit chain: [%s], expected it as given", dim,
          caps(S.lu_chain()).c_str());
    // a stage without matching stays available on request, with its cap (none) as given
    StokesOptions on;
    on.nu = &nu;
    on.lu = LUBackend::StrumpackLU;
    LUStage none, two;
    none.backend = two.backend = LUBackend::StrumpackLU;
    none.matching = MatchingJob::None;
    two.matching = MatchingJob::MaxSmallestDiagonal;
    on.lu_chain = {none, two, e};
    StokesSolver SN(pm, on, b);
    CHECK(caps(SN.lu_chain()) == "strumpack:none/0,strumpack:mc64-2/0,eigen/7",
          "%dD Stokes, explicit chain with a stage without matching: [%s], expected it as given", dim,
          caps(SN.lu_chain()).c_str());
  }
}

// ------------------------------------------------------------------------------------------------ 5
// The Stokes direct solve walks its default chain. The iterate hook multiplies the iterates of chain indices 0 and 1
// by 1 + 1e-6: their relative residual (about 1e-6) misses direct_tol (1e-10), so the walk reaches Eigen, which is
// accepted. A stage that throws instead (a factorization without matching may meet a zero pivot on a saddle point) is
// recorded as "threw" and passes the solve on as well. Reference: the lists of test_fluids_lu_chain section 1.
static void stokes_default_chain_walked() {
  HookReset reset;
  if (!lu_backend_available(LUBackend::StrumpackLU)) {
    if (rank0) std::printf("skip (STRUMPACK is not in this build)\n");
    return;
  }
  mfem::ConstantCoefficient nu(1.0);
  mfem::VectorFunctionCoefficient lid2(2, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(2);
    v(0) = 16.0 * x(0) * x(0) * (1 - x(0)) * (1 - x(0));
    v(1) = 0.0;
  });
  mfem::VectorFunctionCoefficient lid3(3, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(3);
    v(0) = 16.0 * x(0) * (1 - x(0)) * x(1) * (1 - x(1));
    v(1) = v(2) = 0.0;
  });
  for (int dim : {2, 3}) {
    mfem::Mesh sm = dim == 2 ? mfem::Mesh::MakeCartesian2D(8, 8, mfem::Element::TRIANGLE, true, 1.0, 1.0)
                             : mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::TETRAHEDRON, 1.0, 1.0, 1.0);
    mfem::ParMesh pm(MPI_COMM_WORLD, sm);
    StokesBoundary b;
    // MakeCartesian2D: 1 bottom, 2 right, 3 top, 4 left; MakeCartesian3D: 6 is the top face (z = 1)
    const int top = dim == 2 ? 3 : 6;
    for (int a = 1; a <= (dim == 2 ? 4 : 6); ++a) {
      if (a == top) b.dirichlet(a, dim == 2 ? &lid2 : &lid3);
      else b.dirichlet(a);
    }
    StokesOptions o;
    o.nu = &nu;
    o.lu = LUBackend::StrumpackLU;
    o.lu_matching = dim == 2;  // the 2D solvers' default; the 3D API's default is without matching
    const std::string want =
        dim == 2 ? "strumpack:mc64-5,strumpack:mc64-2,eigen" : "strumpack:none,strumpack:mc64-5,eigen";
    StokesSolver S(pm, o, b);
    set_lu_iterate_hook_for_testing([](int k, std::vector<double>& x) {
      if (k > 1) return;
      for (double& v : x) v *= 1.0 + 1e-6;
    });
    const StokesDiagnostics d = S.solve();
    set_lu_iterate_hook_for_testing({});
    const auto& at = d.lu_attempts;
    std::string got;
    for (const auto& a : at) got += (got.empty() ? "" : ",") + a.stage;
    const bool early_ok = at.size() == 3 && (at[0].outcome == "missed_target" || at[0].outcome == "threw") &&
                          (at[1].outcome == "missed_target" || at[1].outcome == "threw");
    CHECK(got == want && early_ok && at.size() == 3 && at[2].outcome == "accepted",
          "%dD Stokes, default chain: attempts [%s], expected the stages [%s], the first two missed or threw and the last "
          "accepted", dim, outcomes(at).c_str(), want.c_str());
    CHECK(d.converged && d.lu_stage == "eigen" && d.lu.backend == "eigen" && d.solver == "direct:strumpack",
          "%dD Stokes, default chain: converged %d, lu_stage '%s', backend '%s', solver '%s'", dim, d.converged ? 1 : 0,
          d.lu_stage.c_str(), d.lu.backend.c_str(), d.solver.c_str());
    CHECK(same_on_all_ranks(records(at)), "%dD Stokes, default chain: the attempt records differ between the ranks",
          dim);
  }
}

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // a run ended by the TIMEOUT still shows the sections that finished
  rank0 = mfem::Mpi::WorldRank() == 0;
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // a launcher that starts N one-rank jobs instead of one N-rank job must not pass for the 2-rank registration
  if (const char* e = std::getenv("AXMP_EXPECT_RANKS"))
    CHECK(size == std::atoi(e), "started on %d rank(s), expected %s", size, e);
  auto section = [&](const char* name, auto&& body) {
    const int before = failures;
    try {
      body();
    } catch (const std::exception& e) {
      CHECK(false, "section %s: %s", name, e.what());
    }
    if (rank0) std::printf("section %s: %s\n", name, failures == before ? "ok" : "FAILED");
  };
  section("1 (projection: its own gate)", [] { projection(); });
  section("2 (steady Navier-Stokes: steps and mass solves)", [] { steady_navier_stokes(); });
  section("3 (Stokes direct solve)", [] { stokes_direct(); });
  section("4 (Stokes chain caps by dimension)", [] { stokes_chain_caps(); });
  section("5 (Stokes default chain walked)", [] { stokes_default_chain_walked(); });
  if (rank0) std::printf(failures ? "test_fluids_lu_callers: %d FAILURES\n" : "test_fluids_lu_callers: OK\n", failures);
  return failures ? 1 : 0;
}
