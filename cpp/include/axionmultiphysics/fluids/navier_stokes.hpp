// Steady incompressible Navier–Stokes, planar and axisymmetric with swirl, Taylor–Hood (beta; formulation:
// docs/guides/fluids.md).
//
//   rho (u.grad)u - div sigma_form(u, p) = f,   div u = 0,   rho > 0 constant, mu(x) > 0
//
// DYNAMIC units: the viscosity coefficient of StokesOptions is the dynamic viscosity mu (nu := mu in the Stokes
// assembly), the force a force density, p the physical pressure.  The linear part (viscous form, pressure coupling,
// Dirichlet/axis elimination, traction loads, nullspace detection and treatment) is the Stokes assembly,
// reused unchanged; the convective term is cpp/include/axionmultiphysics/fluids/convection.hpp (one kernel).
//
// Nonlinear solver (our own driver, nonlinear_driver.hpp):
// a TRUE Stokes start (no convective term in matrix or residual), Picard (Oseen) steps, Newton with the exact
// Jacobian and an optional Armijo line search on the dual-norm merit (lambda = 1, 1/2, ..., 1/64; 1/64 accepted
// without decrease and counted), Re continuation as viscosity scale factors, pseudo-transient continuation
// (J + rho/dt M_u, switched evolution relaxation, release to Newton).  Every residual is the discrete DUAL norm
// ||R||_*^2 = R_u^T M_u^-1 R_u + R_p^T M_p^-1 R_p over the rows of the system actually solved; the gate and the stop
// are the steady relative residual ||R(x)||_* / ||R(x_lift)||_*.  That residual cannot be driven below the round-off
// of its own evaluation, a floor that grows under refinement (docs/physics/fluids.md); the Newton strategy measures
// it and stops there when a step no longer reduces the residual and the residual is still within a decade of
// newton_rtol, reporting it (residual_floor, stopped_at_floor).  Direct sparse LU only.
//
// Only tested features are offered: Dirichlet, axis, traction and do-nothing boundaries; slip, symmetry, periodic,
// a pressure point, grad-div, MINRES and swirl-free axisymmetry are REFUSED (with a message saying what to use).
#pragma once

#include <mfem.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "axionmultiphysics/fluids/convection.hpp"
#include "axionmultiphysics/fluids/nonlinear_driver.hpp"
#include "axionmultiphysics/fluids/mhd.hpp"
#include "axionmultiphysics/fluids/sparse_lu.hpp"
#include "axionmultiphysics/fluids/stokes.hpp"

namespace axmp::fluids {

// TransientNewton / TransientSemiImplicit solve ONE time step of the system the steady driver solves plus the
// time term set by set_time_term (used by TransientNavierStokes, transient_ns.hpp), starting from the state set by
// set_state (Dirichlet DOFs lifted to the current data); no Stokes start.
enum class NonlinearStrategy { Newton, PicardOnly, PseudoTransient, TransientNewton, TransientSemiImplicit };
const char* to_string(NonlinearStrategy s);

struct NonlinearOptions {
  NonlinearStrategy strategy = NonlinearStrategy::Newton;
  int n_picard = 3;                 // Picard steps after the Stokes start (Newton strategy, first stage)
  bool line_search = true;          // Armijo on the dual-norm merit
  double armijo_c1 = 1e-4;
  int line_search_halvings = 6;     // minimum step 2^-6 = 1/64, accepted (and counted) without decrease
  double newton_rtol = 1e-12;       // on the STEADY relative residual (dual norm)
  // A steady Newton solve whose step no longer reduces the residual, within this factor of the MEASURED round-off
  // floor of the residual evaluation (NonlinearDriver::check_floor), stops there and is converged, with the floor
  // and the stop in the diagnostics; 0 leaves newton_rtol as the only gate.  The factor covers the spread of the
  // measurement, which is the difference of two round-off samples and therefore sits a factor ~sqrt(2) above the
  // achieved residual (measured ratios 0.52 to 0.75 on the Hartmann strip and 0.88 on a Re = 300 cavity,
  // docs/physics/fluids.md).  Such a stop never accepts more than NonlinearDriver::floor_stop_rtol_ceiling times
  // newton_rtol.  Strategy Newton only: the Picard-only, pseudo-transient and time-stepping branches and the
  // Boussinesq and GLS solvers stop on newton_rtol alone and ignore this field.
  double residual_floor_factor = 4;
  // accuracy the final direct solve must reach (refined relative residual); <= 0: max(newton_rtol, kFinalSolveResidualMax)
  double final_solve_residual_max = 0;
  int newton_maxit = 30;            // per continuation stage / after the PTC release
  int picard_maxit = 200;           // PicardOnly
  double ptc_dt0 = 0.05, ptc_release_dt = 1e12;
  int ptc_maxit = 200;
  std::vector<double> viscosity_scales;  // continuation schedule s_1..s_m (mu_j = s_j mu); empty = one stage (1)
};

struct NavierStokesOptions {
  StokesOptions stokes;             // nu := mu (dynamic); solver must be Direct, grad_div 0, axisymmetric with swirl
  double rho = 1.0;                 // constant density (> 0)
  ConvectiveForm convection = ConvectiveForm::Standard;
  // 0 = the exact degree of the convective form: 3p + q - 2 planar, 3p + 2q - 2 axisymmetric, p the velocity order and q
  // the geometric order of the mesh (3p - 1 and 3p on straight triangles). A value below it is REFUSED at construction
  // (it would under-integrate the convective term while the Newton gate still reports convergence); above it is
  // accepted (costlier only).
  int quad_order_convection = 0;
  int quad_order_mass = 0;          // 0 = 2p + 2 (the mass matrices of the dual norm and of PTC)
  NonlinearOptions nonlinear;
  // inductionless MHD (mhd.hpp). Requires geometry Planar25D; adds the potential block and the Lorentz force to
  // the residual and Jacobian; rho = 0 is then allowed (a Stokes-like operator with the Lorentz coupling).
  const MhdOptions* mhd = nullptr;
};

// R over ALL velocity true DOFs (no elimination) and the pressure rows, with the SOLVED operators (viscous form and
// rules, convective form, the final stage's viscosity scale): for volume-integral (Babuška–Miller) functionals.
struct FullResidual {
  std::vector<double> r_u, r_p;      // r_u ordered byVDIM like the velocity space
  std::vector<double> node_xy;       // coordinates of the scalar velocity nodes (2 per node)
  std::vector<std::pair<int, std::vector<long>>> attribute_nodes;  // nodes on each boundary attribute (corners twice)
};

struct NavierStokesDiagnostics {
  // from the Stokes assembly; lu, lu_attempts and lu_stage describe the last direct solve (after
  // project_divergence_free(): the projection's)
  StokesDiagnostics linear;
  std::vector<NonlinearStep> history;
  bool converged = false;
  int iterations = 0;               // steps after the Stokes start
  double relative_residual = -1;    // final steady relative residual
  double residual_floor = -1;       // measured round-off floor of that residual (-1: the iteration never stagnated)
  // stopped at that floor instead of reaching newton_rtol: the nonlinear verdict is "converged" either way, and the
  // gates on the linear algebra below still apply (a floor stop whose final direct solve misses its bound keeps
  // stopped_at_floor = true and reports converged = false)
  bool stopped_at_floor = false;
  double residual_lift = -1;        // ||R(x_lift)||_* at the final stage's viscosity
  int line_search_min_step_accepts = 0;
  std::string convection, strategy, failure;  // failure: why converged = false ("" if converged)
  std::vector<int> quad_orders;     // viscous, convection, load, boundary, mass
  double seconds = 0;
  std::vector<double> viscosity_scales;
  // Aliases kept for one release, derived from lu_counts: the step solves of this call that needed more than one
  // attempt (lu_counts.escalated), and the stage that produced the solution of the last of them ("" if none).
  long lu_fallbacks = 0;
  std::string lu_fallback_backend;
  // The step solves of this solve() call (the projection's are projection_lu_attempts; the dual-norm mass solves
  // count in mass_lu_escalated only), the mass solves of this call that needed more than one attempt, and the
  // attempts of the last project_divergence_free() (empty: none ran).
  LUCounts lu_counts;
  long mass_lu_escalated = 0;
  std::vector<LUAttempt> projection_lu_attempts;
  // Accuracy of the linear algebra the verdict rests on. The FINAL step's direct solve (the one whose solution is
  // reported) with a refined relative residual above the caller's gate, and a dual-norm mass solve above
  // kDualMassResidualMax (sparse_lu.hpp), make solve() report converged = false with the reason in `failure`.
  // Intermediate steps are recorded and counted but not enforced: the verdict is the nonlinear residual, recomputed
  // after every step.
  double lu_residual_max = -1;      // worst recorded per-step refined LU residual of this solve; -1 = none
  long lu_target_missed = 0;        // steps whose recorded residual exceeded the refinement target min(1e-12, 1e-2 rtol)
  double mass_lu_residual_max = -1; // worst relative residual of a dual-norm mass solve of this solve; -1 = none taken
  // the last project_divergence_free(): the backward error of its saddle solve (-1: no projection run; sparse_lu.hpp
  // defines it at kProjectionResidualMax) and whether it met that bound
  double projection_residual = -1;
  bool projection_converged = false;
};

class NavierStokesSolver {
 public:
  NavierStokesSolver(mfem::ParMesh& mesh, const NavierStokesOptions& opt, const StokesBoundary& bc);
  ~NavierStokesSolver();
  NavierStokesSolver(const NavierStokesSolver&) = delete;
  NavierStokesSolver& operator=(const NavierStokesSolver&) = delete;

  const NavierStokesDiagnostics& solve();  // never throws on non-convergence
  const NavierStokesDiagnostics& diagnostics() const { return diag_; }
  const NavierStokesOptions& options() const { return opt_; }
  // fields of a converged solve (throw otherwise); pressure mean-free (r-weighted) for an enclosed flow
  const mfem::ParGridFunction& velocity() const;
  const mfem::ParGridFunction& pressure() const;
  mfem::ParFiniteElementSpace& velocity_space();
  // MHD (options().mhd != nullptr): the potential of a converged solve (mean-free when it is defined up to a constant)
  // and the coupling diagnostics; the potential is refused (throws) for a non-MHD solver.
  const mfem::GridFunction& potential() const;
  const MhdDiagnostics& mhd_diagnostics() const;

  // ---- Time-stepper interface: internal, used by TransientNavierStokes (transient_ns.hpp) and its Python binding;
  // not a stable API. All on true-DOF vectors, MPI-safe.
  // data at the current time: re-evaluate force/traction/Dirichlet coefficients (StokesSolver::refresh_data) and take
  // the new lift, load and continuity right-hand side
  void refresh_data();
  // momentum residual += zero_ess(c M u - hist) with M the (consistent, r-weighted) velocity mass; c = rho alpha_0/dt
  // (NOT multiplied here), hist a full velocity vector; the Jacobian gets + c M. c = 0 clears the term.
  void set_time_term(double c, const mfem::Vector& hist);
  // advecting velocity of the semi-implicit strategy (full true-DOF vector)
  void set_advecting(const mfem::Vector& w);
  // current state (true DOFs; pressure internal, not normalized); set_state replaces Dirichlet DOFs by the lift
  void set_state(const mfem::Vector& u, const mfem::Vector& p, bool lift_boundary = true);
  const mfem::Vector& state_u() const;
  const mfem::Vector& state_p() const;
  const mfem::Vector& lift() const;
  mfem::ParFiniteElementSpace& pressure_space();
  // subtract the r-weighted mean of p when the flow is enclosed (the reported, mean-free pressure)
  void normalize_pressure(mfem::Vector& p) const;
  // fields of the current state regardless of the steady convergence flag (a time step's own gate is diagnostics())
  const mfem::ParGridFunction& velocity_field() const;
  const mfem::ParGridFunction& pressure_field() const;
  int velocity_size() const;
  int pressure_size() const;
  void mass_mult(const mfem::Vector& x, mfem::Vector& y) const;       // y = M_u x (full, r-weighted in axisymmetry)
  void mass_mult_time(const mfem::Vector& x, mfem::Vector& y) const;  // y = the time mass times x (M_u)
  void zero_essential(mfem::Vector& v) const;
  // sqrt(x^T M x) with the (r-weighted in axisymmetry) velocity mass M
  double mass_norm(const mfem::Vector& x) const;
  // discrete L2 (Leray) projection of u_target onto the discretely divergence-free fields with the nodal Dirichlet
  // data; sets the state.
  // Gated on the backward error of its saddle solve against kProjectionResidualMax (sparse_lu.hpp: 1e-13 at every
  // newton_rtol, on every domain shape and in every length unit), recorded in diagnostics().projection_residual /
  // projection_converged. Returns false on a failed or inaccurate projection; the state is then untouched and
  // velocity_field() / pressure_field() refuse.
  bool project_divergence_free(const mfem::Vector& u_target);
  // Test hook: scale the solution of the projection's saddle solve by (1 + eps) before its residual is formed, which
  // makes the projection inaccurate by a known amount (a backward error of about 0.7 eps / N on an N x N mesh). 0
  // (the default) changes nothing. It exists so that the tests can show the gate refusing an inaccurate projection.
  void set_projection_solve_perturbation(double eps) { projection_perturbation_ = projection_perturbation_p_ = eps; }
  // The same hook with the velocity correction and the multiplier scaled separately, by (1 + eps_u) and (1 + eps_p):
  // eps_u = 0 leaves the continuity block exact and puts the whole error in the momentum block.
  void set_projection_solve_perturbation(double eps_u, double eps_p) {
    projection_perturbation_ = eps_u;
    projection_perturbation_p_ = eps_p;
  }

  // Taylor-test / force API on the free DOFs, one rank only
  std::vector<double> state() const;
  std::vector<double> residual(const std::vector<double>& x_free, double viscosity_scale = 1.0) const;
  std::vector<double> jacobian_action(const std::vector<double>& x_free, const std::vector<double>& d,
                                      double viscosity_scale = 1.0) const;
  DistCsr jacobian(const std::vector<double>& x_free, double viscosity_scale = 1.0) const;
  double dual_norm(const std::vector<double>& R) const;
  std::pair<long, long> layout() const;
  FullResidual full_residual() const;
  // Babuška–Miller reaction with the convective term of the solved form; dynamic units
  Reaction reaction(const std::vector<int>& body, const std::vector<double>& center = {},
                    ViscousForm residual_form = ViscousForm::SymmetricStress) const;

 struct Impl;  // opaque (navier_stokes_internal.hpp); public only so that BoussinesqSolver's solver system can name it

 private:
  friend class BoussinesqSolver;
  // One direct solve of the monolithic system with velocity block Juu and right-hand side -(Ru, Rp [, R_aux]), on
  // the solver's chain (StokesOptions::lu_chain or the default chain of its backend), accepted on `acceptance`. Its
  // stats, attempts and stage are left in diag_.linear (lu, lu_attempts, lu_stage). false when no stage produced a
  // finite solution.
  bool lin_solve_(const mfem::HypreParMatrix& Juu, const mfem::Vector& Ru, const mfem::Vector& Rp, mfem::Vector& du,
                  mfem::Vector& dp, int& refine, double& lu_residual, const LUAcceptance& acceptance);
  std::unique_ptr<Impl> impl_;
  NavierStokesOptions opt_;
  NavierStokesDiagnostics diag_;
  double projection_perturbation_ = 0, projection_perturbation_p_ = 0;
};

}  // namespace axmp::fluids
