// Steady Navier–Stokes: see cpp/include/axionmultiphysics/fluids/navier_stokes.hpp.
#include "axionmultiphysics/fluids/navier_stokes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fluids/axi_integrators.hpp"
#include "mhd_internal.hpp"
#include "navier_stokes_internal.hpp"
#include "stokes_internal.hpp"

namespace axmp::fluids {

const char* to_string(NonlinearStrategy s) {
  switch (s) {
    case NonlinearStrategy::Newton: return "newton";
    case NonlinearStrategy::PicardOnly: return "picard";
    case NonlinearStrategy::PseudoTransient: return "ptc";
    case NonlinearStrategy::TransientNewton: return "transient_newton";
    case NonlinearStrategy::TransientSemiImplicit: return "transient_semi";
  }
  return "?";
}

namespace {

// block-diagonal vector mass on a vector H1 space (element vdofs in component blocks): int c phi_a phi_b per component
class VecMass : public mfem::BilinearFormIntegrator {
 public:
  VecMass(mfem::Coefficient& c, int vdim, int order) : c_(c), vdim_(vdim), order_(order) {}
  void AssembleElementMatrix(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                             mfem::DenseMatrix& elmat) override {
    const int nd = el.GetDof();
    elmat.SetSize(vdim_ * nd);
    elmat = 0.0;
    mfem::Vector N(nd);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), order_);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      el.CalcShape(ip, N);
      const double w = ip.weight * T.Weight() * c_.Eval(T, ip);
      for (int c = 0; c < vdim_; ++c)
        for (int a = 0; a < nd; ++a)
          for (int b = 0; b < nd; ++b) elmat(c * nd + a, c * nd + b) += w * N(a) * N(b);
    }
  }

 private:
  mfem::Coefficient& c_;
  int vdim_, order_;
};

class ScalarMass : public mfem::BilinearFormIntegrator {
 public:
  ScalarMass(mfem::Coefficient& c, int order) : c_(c), order_(order) {}
  void AssembleElementMatrix(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                             mfem::DenseMatrix& elmat) override {
    const int nd = el.GetDof();
    elmat.SetSize(nd);
    elmat = 0.0;
    mfem::Vector N(nd);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), order_);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      el.CalcShape(ip, N);
      const double w = ip.weight * T.Weight() * c_.Eval(T, ip);
      for (int a = 0; a < nd; ++a)
        for (int b = 0; b < nd; ++b) elmat(a, b) += w * N(a) * N(b);
    }
  }

 private:
  mfem::Coefficient& c_;
  int order_;
};

// the inner integrator on the rule of the given order for each element's geometry (as the Stokes solver's RuleBFI)
class RuleBFI : public mfem::BilinearFormIntegrator {
 public:
  RuleBFI(mfem::BilinearFormIntegrator* in, int order) : in_(in), order_(order) {}
  void AssembleElementMatrix(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                             mfem::DenseMatrix& m) override {
    in_->SetIntRule(&mfem::IntRules.Get(el.GetGeomType(), order_));
    in_->AssembleElementMatrix(el, T, m);
  }

 private:
  std::unique_ptr<mfem::BilinearFormIntegrator> in_;
  int order_;
};

}  // namespace

NavierStokesSolver::NavierStokesSolver(mfem::ParMesh& mesh, const NavierStokesOptions& opt, const StokesBoundary& bc)
    : impl_(new Impl), opt_(opt) {
  Impl& I = *impl_;
  const StokesOptions& so = opt.stokes;
  // ---- only tested features are offered: refuse what no verification case or unit test exercises
  AXMP_REQUIRE(std::isfinite(opt.rho) && (opt.rho > 0 || (opt.rho == 0 && opt.mhd != nullptr)),
              strcat("the density must be positive and finite (got ", opt.rho, "; rho = 0 is offered with MHD only)"),
              "NavierStokesOptions.rho");
  AXMP_REQUIRE((opt.mhd != nullptr) == (so.geometry == Geometry::Planar25D),
              "geometry Planar25D and the MHD options go together: the 2.5D formulation is offered with MHD only, and "
              "MHD only in 2.5D (axisymmetric MHD is not offered)", "NavierStokesOptions.mhd",
              "Pass geometry = 'planar25d' together with the mhd options.");
  AXMP_REQUIRE(so.solver == LinearSolverKind::Direct,
              "Navier-Stokes: only the direct sparse LU is offered; MINRES with the block-diagonal Stokes "
              "preconditioner is not an Oseen preconditioner", "StokesOptions.solver",
              "Use the direct solver (LU 'strumpack' or 'eigen'); iterative Oseen solvers are not implemented.");
  AXMP_REQUIRE(so.grad_div == 0.0, "Navier-Stokes: grad-div stabilization is not offered with convection",
              "StokesOptions.grad_div", "Use grad_div = 0.");
  AXMP_REQUIRE(!so.pressure_point.active, "Navier-Stokes: a pressure point constraint is not offered with convection",
              "StokesOptions.pressure_point",
              "Remove it (an enclosed flow's pressure is returned mean-free).");
  AXMP_REQUIRE(so.nullspace != Nullspace::Ortho, "Navier-Stokes: Nullspace::Ortho is the Krylov treatment",
              "StokesOptions.nullspace", "Use Auto, PinPressure or MeanMultiplier.");
  for (const auto& c : bc.conditions)
    AXMP_REQUIRE(c.kind == BCKind::Dirichlet || c.kind == BCKind::Axis || c.kind == BCKind::Traction ||
                    c.kind == BCKind::DoNothing,
                strcat("Navier-Stokes: boundary condition '", to_string(c.kind), "' on attribute ", c.attribute,
                       " is not offered with convection"),
                "StokesBoundary",
                "Slip, symmetry and periodic conditions with convection are not offered; use the Stokes "
                "solver for such problems.");
  if (so.geometry == Geometry::Axisymmetric)
    AXMP_REQUIRE(so.swirl, "Navier-Stokes: axisymmetric flow is offered WITH swirl only (3 velocity components)",
                "StokesOptions.swirl",
                "Pose a swirl-free problem with u_theta = 0 data and no theta force: the swirl solver keeps u_theta = 0 "
                "exactly.");
  if (opt.mhd != nullptr) {
    // only tested features are offered: no MHD case or unit test exercises these with the Lorentz coupling
    AXMP_REQUIRE(so.nu != nullptr && dynamic_cast<mfem::ConstantCoefficient*>(so.nu) != nullptr,
                "MHD: a variable viscosity is not offered with the Lorentz coupling (untested)", "StokesOptions.nu",
                "Pass a constant mu.");
    AXMP_REQUIRE(opt.convection == ConvectiveForm::Standard,
                "MHD: the skew-symmetric convective form is not offered with the Lorentz coupling (untested)",
                "NavierStokesOptions.convection", "Use convection = 'standard'.");
    AXMP_REQUIRE(opt.nonlinear.strategy != NonlinearStrategy::PseudoTransient,
                "MHD: pseudo-transient continuation is not offered with the Lorentz coupling (untested)",
                "NonlinearOptions.strategy", "Use strategy = 'newton'.");
    AXMP_REQUIRE(opt.nonlinear.viscosity_scales.size() <= 1,
                "MHD: viscosity (Reynolds) continuation is not offered with the Lorentz coupling (it also rescales the "
                "Lorentz/viscous ratio; untested)", "NonlinearOptions.viscosity_scales",
                "Use a single stage (no continuation_re / viscosity_scales).");
    AXMP_REQUIRE(opt.nonlinear.strategy != NonlinearStrategy::TransientNewton &&
                    opt.nonlinear.strategy != NonlinearStrategy::TransientSemiImplicit,
                "MHD: transient time stepping is not offered with the Lorentz coupling (untested; the time steps carry "
                "no potential unknowns)", "NonlinearOptions.strategy", "Use the steady strategy 'newton'.");
  }
  const auto& nl = opt.nonlinear;
  AXMP_REQUIRE(nl.newton_rtol > 0 && std::isfinite(nl.newton_rtol), "newton_rtol must be positive", "NonlinearOptions");
  AXMP_REQUIRE(nl.residual_floor_factor >= 0 && std::isfinite(nl.residual_floor_factor),
              "residual_floor_factor must be >= 0 (0 stops only at newton_rtol)", "NonlinearOptions");
  AXMP_REQUIRE(nl.n_picard >= 0 && nl.newton_maxit >= 0 && nl.picard_maxit >= 0 && nl.ptc_maxit >= 0 &&
                  nl.line_search_halvings >= 0,
              "iteration budgets must be >= 0", "NonlinearOptions");
  AXMP_REQUIRE(nl.ptc_dt0 > 0 && nl.ptc_release_dt > 0, "PTC time steps must be positive", "NonlinearOptions");
  for (double s : nl.viscosity_scales)
    AXMP_REQUIRE(std::isfinite(s) && s > 0, strcat("viscosity scale ", s, " is not positive"),
                "NonlinearOptions.viscosity_scales");
  AXMP_REQUIRE(nl.strategy == NonlinearStrategy::Newton || nl.viscosity_scales.size() <= 1,
              "a continuation schedule runs Newton stages: use strategy Newton", "NonlinearOptions");

  I.mesh = &mesh;
  I.comm = mesh.GetComm();
  I.axi = so.geometry == Geometry::Axisymmetric;
  I.order = so.velocity_order;

  // The convective rule: the exact degree of the trilinear form on the mesh actually used (it depends on the geometric
  // order of the elements, not only on the velocity order). A user rule below it under-integrates the convective term:
  // the discrete operator changes while the Newton gate, which measures the residual of whatever system was assembled,
  // still reports convergence. Refused before any assembly, collectively (every rank takes the same maximum).
  int conv_exact = 0;
  {
    mfem::H1_FECollection vfec(I.order, mesh.Dimension());
    int under = 0;  // an element on which the caller's rule is not exact
    for (int e = 0; e < mesh.GetNE(); ++e) {
      const mfem::FiniteElement* fe = vfec.FiniteElementForGeometry(mesh.GetElementBaseGeometry(e));
      mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
      conv_exact = std::max(conv_exact, ConvectionIntegrator::exact_degree(*fe, T, I.axi));
      if (opt.quad_order_convection > 0 && !ConvectionIntegrator::rule_is_exact(*fe, T, I.axi, opt.quad_order_convection))
        under = 1;
    }
    int loc[2] = {conv_exact, under}, g[2] = {0, 0};
    MPI_Allreduce(loc, g, 2, MPI_INT, MPI_MAX, I.comm);
    conv_exact = g[0];
    AXMP_REQUIRE(!g[1],
                strcat("NavierStokesOptions.quad_order_convection = ", opt.quad_order_convection,
                       " is below the exact degree ", conv_exact, " of the convective form (",
                       I.axi ? "axisymmetric" : "planar", ", velocity order ", I.order, ", mesh geometry order ",
                       mesh.GetNodalFESpace() ? mesh.GetNodalFESpace()->GetMaxElementOrder() : 1, ")"),
                "NavierStokesOptions.quad_order_convection",
                strcat("Use quad_order_convection = 0 (the exact minimum, ", conv_exact, ") or a value >= ", conv_exact,
                       "; a lower rule under-integrates the convective term, which changes the discrete operator while "
                       "the Newton gate still reports convergence."));
  }
  I.S.reset(new StokesSolver(mesh, so, bc));
  StokesSolver& S = *I.S;
  S.assemble();
  StokesSolver::Impl& SI = *S.impl_;
  I.vdim = SI.vdim;
  I.nu_loc = SI.offsets[1];
  I.np_loc = SI.offsets[2] - SI.offsets[1];
  I.A_el = SI.A.get();
  I.A0 = SI.A0.get();
  I.B = SI.B.get();
  I.Bt = SI.Bt.get();
  I.B0 = SI.B0.get();
  I.F = SI.Fvec;
  I.rhs_p = SI.rhs->GetBlock(1);
  I.lift = SI.x->GetBlock(0);
  I.mean_weights = SI.mean_weights;
  I.ess = SI.ess_tdofs;
  I.ess_mask.assign(I.nu_loc, 0);
  for (int i : I.ess) I.ess_mask[i] = 1;
  I.nullspace = S.diag_.nullspace_detected;
  StokesSolver* Sp = &S;
  // the direct path's monolithic matrix (nullspace treatment of the Stokes solver) with our velocity block: the
  // Stokes solver's own monolithic() run on a temporarily swapped A (restored on every path)
  I.mono = [Sp](const mfem::HypreParMatrix& A00) {
    StokesSolver::Impl& J = *Sp->impl_;
    struct Swap {
      std::unique_ptr<mfem::HypreParMatrix>& slot;
      mfem::HypreParMatrix* saved;
      ~Swap() {
        slot.release();
        slot.reset(saved);
      }
    } guard{J.A, J.A.release()};
    J.A.reset(const_cast<mfem::HypreParMatrix*>(&A00));
    return Sp->monolithic(true, nullptr);
  };
  // pinned pressure DOF (local index on its owner)
  if (S.options().nullspace == Nullspace::PinPressure && I.nullspace) {
    (void)I.mono(*I.A_el);  // fills diag_.pinned_dof (the Stokes solver's pin choice)
    const long gpin = S.diag_.pinned_dof;
    const long rb = I.A_el->RowPart()[0] + SI.B->RowPart()[0];  // monolithic row begin = u rows + p rows before me
    const long loc = gpin - rb - I.nu_loc;
    if (loc >= 0 && loc < I.np_loc) I.pin_local = loc;
  }

  // convection and its quadrature
  mfem::ParFiniteElementSpace& ufes = S.velocity_space();
  I.N.reset(new mfem::ParNonlinearForm(&ufes));
  I.conv = new ConvectionIntegrator(opt.rho, I.vdim, I.axi, opt.convection, opt.quad_order_convection);
  I.N->AddDomainIntegrator(I.conv);

  // mass matrices of the dual norm and of PTC: consistent, exactly integrated, r-weighted in axisymmetry
  I.qm = opt.quad_order_mass > 0 ? opt.quad_order_mass : 2 * I.order + 2;
  I.one.reset(new mfem::ConstantCoefficient(1.0));
  I.rweight.reset(new RadiusWeight());
  mfem::Coefficient& wc = I.axi ? *I.rweight : *I.one;
  {
    mfem::ParBilinearForm mu(&ufes);
    mu.AddDomainIntegrator(new VecMass(wc, I.vdim, I.qm));
    mu.Assemble();
    mu.Finalize();
    I.Mu.reset(mu.ParallelAssemble());
    mfem::ParBilinearForm mp(&S.pressure_space());
    mp.AddDomainIntegrator(new ScalarMass(wc, I.qm));
    mp.Assemble();
    mp.Finalize();
    I.Mp.reset(mp.ParallelAssemble());
  }
  I.Mu_one.reset(new mfem::HypreParMatrix(*I.Mu));
  I.Mt = I.Mu.get();
  I.Mu_one->EliminateBC(I.ess, mfem::Operator::DIAG_ONE);
  I.Mu_zero.reset(new mfem::HypreParMatrix(*I.Mu));
  I.Mu_zero->EliminateBC(I.ess, mfem::Operator::DIAG_ZERO);
  I.Mtz = I.Mu_zero.get();
  I.Mp_one.reset(new mfem::HypreParMatrix(*I.Mp));
  {
    mfem::Array<int> pin;
    if (I.pin_local >= 0) pin.Append(static_cast<int>(I.pin_local));
    I.Mp_one->EliminateBC(pin, mfem::Operator::DIAG_ONE);
  }
  I.Mu_csr = to_dist(*I.Mu_one, I.comm);
  I.Mp_csr = to_dist(*I.Mp_one, I.comm);
  // the mass solves are accepted on their own gate, the relative residual at kDualMassResidualMax (sparse_lu.hpp)
  const std::vector<LUStage> lu_chain = chain_or_default(so.lu_chain, so.lu, MatchingJob::MaxDiagonalProductScaling);
  LUAcceptance mass_acceptance;
  mass_acceptance.kind = LUAcceptance::Kind::RelResidual;
  mass_acceptance.bound = kDualMassResidualMax;
  for (auto* pr : {&I.Mu_lu, &I.Mp_lu}) {
    pr->reset(new SparseLU(lu_chain));
    (*pr)->set_acceptance(mass_acceptance);
    (*pr)->set_refinement(10, 1e-15);
    (*pr)->set_throw_on_miss(false);
    (*pr)->set_refine_to_stagnation(true);
    (*pr)->set_threads(so.lu_threads);
  }
  I.Mu_lu->factorize(I.Mu_csr);
  I.Mp_lu->factorize(I.Mp_csr);

  if (opt.mhd) {
    I.M.reset(new MhdCoupling(mesh, ufes, *opt.mhd, I.ess));
    I.na = I.M->na();
    I.Kuu_sp = const_cast<mfem::SparseMatrix*>(&I.M->Kuu_e());
    I.kuu_rows[0] = 0;
    I.kuu_rows[1] = I.nu_loc;
    I.Kuu_h.reset(new mfem::HypreParMatrix(I.comm, I.nu_loc, I.kuu_rows, I.Kuu_sp));
    I.Ma_csr = to_dist_sparse(I.M->mass_e(), I.comm);
    I.Ma_lu.reset(new SparseLU(lu_chain));
    I.Ma_lu->set_acceptance(mass_acceptance);
    I.Ma_lu->set_refinement(10, 1e-15);
    I.Ma_lu->set_throw_on_miss(false);
    I.Ma_lu->set_refine_to_stagnation(true);
    I.Ma_lu->set_threads(so.lu_threads);
    I.Ma_lu->factorize(I.Ma_csr);
  }
  diag_.convection = to_string(opt.convection);
  diag_.strategy = to_string(nl.strategy);
  diag_.quad_orders = {so.viscous_quad_order(), opt.quad_order_convection > 0 ? opt.quad_order_convection : conv_exact,
                       so.load_quad_order(), so.boundary_quad_order(), I.qm};
  diag_.linear = S.diag_;
  I.u = I.lift;
  I.p.SetSize(I.np_loc + I.na);
  I.p = 0.0;
  for (int i = 0; i < I.na; ++i) I.p(I.np_loc + i) = I.M->lift()(i);
}

NavierStokesSolver::~NavierStokesSolver() = default;

mfem::ParFiniteElementSpace& NavierStokesSolver::velocity_space() { return impl_->S->velocity_space(); }

// one linear solve with velocity block Juu and right-hand side -(Ru, Rp [, R_aux]): the monolithic direct matrix with
// the Stokes solver's nullspace treatment (and, with MHD, the potential rows and columns appended)
bool NavierStokesSolver::lin_solve_(const mfem::HypreParMatrix& Juu, const mfem::Vector& Ru, const mfem::Vector& Rp,
                                    mfem::Vector& du, mfem::Vector& dp, int& refine, double& lu_residual,
                                    const LUAcceptance& acceptance) {
  Impl& I = *impl_;
  const auto& nl = opt_.nonlinear;
  DistCsr D = I.mono(Juu);
  const int nD = D.local_rows;
  if (I.M) D = I.extend(D);
  std::vector<double> b(D.local_rows, 0.0);
  for (int i = 0; i < I.nu_loc; ++i) b[i] = -Ru(i);
  for (int i = 0; i < I.np_loc; ++i) b[I.nu_loc + i] = -Rp(i);
  for (int i = 0; i < I.na; ++i) b[nD + i] = -Rp(I.np_loc + i);
  std::vector<double> z(D.local_rows, 0.0);
  for (int i = I.nu_loc; i < I.nu_loc + I.np_loc; ++i) z[i] = 1.0;
  // The row blocks of the recorded backward error and of the kernel probe: velocity, pressure and (MHD) potential
  // rows; the Dirichlet rows of the velocity and of the potential are left out (-1); the pinned pressure row and the
  // bordered rows (the flow-rate constraints, the mean-value multiplier; on the last rank) are constraint rows (-2).
  std::vector<int> blocks(D.local_rows);
  for (int i = 0; i < I.nu_loc; ++i) blocks[i] = I.ess_mask[i] ? -1 : 0;
  for (int i = 0; i < I.np_loc; ++i) blocks[I.nu_loc + i] = i == I.pin_local ? -2 : 1;
  for (int i = I.nu_loc + I.np_loc; i < nD; ++i) blocks[i] = -2;
  for (int i = 0; i < I.na; ++i) blocks[nD + i] = I.M->ess_mask()[i] ? -1 : 2;
  // Every solve starts at the chain's first stage. A stage whose solution misses the caller's acceptance passes the
  // solve on to the next stage; the rejected stages' reasons are in the attempts, never in diag_.failure. When no
  // stage is accepted the best solution is returned, and the caller's gates decide on it.
  SparseLU lu(chain_or_default(opt_.stokes.lu_chain, opt_.stokes.lu, MatchingJob::MaxDiagonalProductScaling));
  lu.set_acceptance(acceptance);
  lu.set_row_blocks(std::move(blocks));
  lu.set_refinement(10, std::min(1e-12, 1e-2 * nl.newton_rtol));
  lu.set_throw_on_miss(false);
  lu.set_threads(opt_.stokes.lu_threads);
  lu.set_refine_to_stagnation(true);
  lu.set_kernel_probe(z, 1e-14);
  std::vector<double> x;
  bool solved = true;
  try {
    lu.factorize(D);
    x = lu.solve(b);
  } catch (const std::exception& e) {  // no stage produced a solution: the text lists every attempt
    diag_.failure = std::string("linear solve failed: ") + e.what();
    solved = false;
  }
  diag_.linear.lu = lu.stats();
  diag_.linear.lu_attempts = lu.attempts();
  diag_.linear.lu_stage = solved ? lu.stats().stage : std::string();
  if (!solved) return false;
  refine = lu.stats().refine_steps;
  lu_residual = lu.stats().rel_residual;  // the returned solution's refined relative residual (identical on every rank)
  du.SetSize(I.nu_loc);
  dp.SetSize(I.np_loc + I.na);
  for (int i = 0; i < I.nu_loc; ++i) du(i) = x[i];
  for (int i = 0; i < I.np_loc; ++i) dp(i) = x[I.nu_loc + i];
  for (int i = 0; i < I.na; ++i) dp(I.np_loc + i) = I.M->ess_mask()[i] ? 0.0 : x[nD + i];
  for (int i : I.ess) du(i) = 0.0;
  for (int i = 0; i < du.Size(); ++i)
    if (!std::isfinite(du(i))) return false;
  return true;
}

const NavierStokesDiagnostics& NavierStokesSolver::solve() {
  Impl& I = *impl_;
  StokesSolver& S = *I.S;
  const auto& nl = opt_.nonlinear;
  const double t0 = MPI_Wtime();
  diag_.history.clear();
  diag_.converged = false;
  diag_.residual_floor = -1;
  diag_.stopped_at_floor = false;
  diag_.iterations = 0;
  diag_.line_search_min_step_accepts = 0;
  diag_.failure.clear();
  I.mass_res_max = -1;
  I.mass_escalated = 0;
  diag_.linear = S.diag_;
  diag_.lu_counts = LUCounts();
  diag_.mass_lu_escalated = 0;
  diag_.lu_fallbacks = 0;
  diag_.lu_fallback_backend.clear();
  const bool transient = nl.strategy == NonlinearStrategy::TransientNewton ||
                         nl.strategy == NonlinearStrategy::TransientSemiImplicit;
  const bool semi = nl.strategy == NonlinearStrategy::TransientSemiImplicit;
  std::vector<double> scales = nl.viscosity_scales;
  if (scales.empty()) scales.push_back(1.0);
  diag_.viscosity_scales = scales;
  int me = 0;
  MPI_Comm_rank(I.comm, &me);
  if (me == 0 && opt_.stokes.lu == LUBackend::EigenLU && diag_.linear.n_total > 20000)
    std::fprintf(stderr, "axionmultiphysics.fluids: Navier-Stokes with Eigen's SparseLU at %ld unknowns: slow (~1e2 s per "
                         "factorization at 1e5); use the STRUMPACK backend\n", diag_.linear.n_total);

  const double rtol = nl.newton_rtol;
  const double gate = semi ? std::max(rtol, 1e-11) : rtol;  // semi-implicit: one direct linear solve
  // The bound of the final step's direct solve (sparse_lu.hpp, kFinalSolveResidualMax), and the acceptance of every
  // step's: any step may be the final one, so each is accepted on the final step's verdict.
  const double lu_gate = nl.final_solve_residual_max > 0 ? nl.final_solve_residual_max
                                                        : std::max(gate, kFinalSolveResidualMax);
  LUAcceptance step_acceptance;
  step_acceptance.kind = LUAcceptance::Kind::RelResidual;
  step_acceptance.bound = lu_gate;
  // every step solve is booked in this call's counts; its attempts and stage go into its history entry
  int step_attempts = 0;
  std::string step_stage;
  auto lin_solve = [&](const mfem::HypreParMatrix& Juu, const mfem::Vector& Ru, const mfem::Vector& Rp,
                       mfem::Vector& du, mfem::Vector& dp, int& refine, double& lures) -> bool {
    const bool ok = lin_solve_(Juu, Ru, Rp, du, dp, refine, lures, step_acceptance);
    const std::vector<LUAttempt>& at = diag_.linear.lu_attempts;
    lu_count_solve(diag_.lu_counts, at);
    step_attempts = static_cast<int>(at.size());
    step_stage = diag_.linear.lu_stage;
    if (at.size() > 1) diag_.lu_fallback_backend = diag_.linear.lu_stage;
    return ok;
  };
  mfem::Vector Ru, Rp, du, dp, tu, tp, Tu, Tp;
  auto rel = [&](double den) { return I.dual(Ru, Rp) / den; };
  auto push = [&](const char* kind, double r, double lam, double dt, int stage, double s, int refine, double lures) {
    NonlinearStep h;
    h.kind = kind;
    h.residual = r;
    h.step = lam;
    h.dt = dt;
    h.stage = stage;
    h.viscosity_scale = s;
    h.lu_refine = refine;
    h.lu_residual = lures;
    h.lu_attempts = step_attempts;
    h.lu_stage = step_stage;
    diag_.history.push_back(h);
  };
  auto fail = [&](const std::string& why) {
    if (diag_.failure.empty()) diag_.failure = why;
    return false;
  };
  const mfem::Vector u_lift = I.lift;
  if (transient) {  // a time step: the predictor is the state in hand with the Dirichlet DOFs lifted to the current data
    for (int i : I.ess) I.u(i) = u_lift(i);
    if (semi) {
      AXMP_REQUIRE(I.advect.Size() == I.nu_loc, "the semi-implicit step needs set_advecting()", "TransientSemiImplicit");
      I.conv->set_grad_mode(ConvectionIntegrator::GradMode::Oseen);
      auto& G = dynamic_cast<mfem::HypreParMatrix&>(I.N->GetGradient(I.advect));
      I.oseen_full.reset(new mfem::HypreParMatrix(G));
      I.conv->set_grad_mode(ConvectionIntegrator::GradMode::Newton);
    } else {
      I.oseen_full.reset();
    }
  }
  mfem::Vector p_zero(I.np_loc + I.na);
  p_zero = 0.0;
  for (int i = 0; i < I.na; ++i) p_zero(I.np_loc + i) = I.M->lift()(i);

  // the denominator ||R(x_lift)||_* at viscosity scale s
  auto lift_norm = [&](double s) {
    mfem::Vector a, b;
    I.residual(u_lift, p_zero, s, true, a, b);
    return I.dual(a, b);
  };

  // the Newton / Picard step with the line search lives in the generic NonlinearDriver; the state is the
  // concatenation (u, p [, aux]) and every update is elementwise
  struct NSSystem : NonlinearSystem {
    Impl& I;
    std::function<bool(const mfem::HypreParMatrix&, const mfem::Vector&, const mfem::Vector&, mfem::Vector&,
                       mfem::Vector&, int&, double&)> lin;
    const int* attempts = nullptr;
    const std::string* stage = nullptr;
    explicit NSSystem(Impl& i) : I(i) {}
    void lu_record(int& n, std::string& st) const override {
      n = *attempts;
      st = *stage;
    }
    int npt() const { return I.np_loc + I.na; }
    void split(const mfem::Vector& x, mfem::Vector& u, mfem::Vector& p) const {
      u.SetSize(I.nu_loc);
      p.SetSize(npt());
      for (int i = 0; i < I.nu_loc; ++i) u(i) = x(i);
      for (int i = 0; i < npt(); ++i) p(i) = x(I.nu_loc + i);
    }
    void residual(const mfem::Vector& x, double s, mfem::Vector& R) override {
      mfem::Vector u, p, Ru, Rp;
      split(x, u, p);
      I.residual(u, p, s, true, Ru, Rp);
      R.SetSize(I.nu_loc + npt());
      for (int i = 0; i < I.nu_loc; ++i) R(i) = Ru(i);
      for (int i = 0; i < npt(); ++i) R(I.nu_loc + i) = Rp(i);
    }
    double dual_norm(const mfem::Vector& R) override {
      mfem::Vector Ru, Rp;
      split(R, Ru, Rp);
      return I.dual(Ru, Rp);
    }
    bool solve_linear(const mfem::Vector& x, const mfem::Vector& R, double s, StepMode mode, mfem::Vector& d,
                      int& refine, double& lu_residual, std::string& failure) override {
      mfem::Vector u, p, Ru, Rp, du, dp;
      split(x, u, p);
      split(R, Ru, Rp);
      auto C = I.conv_grad(u, mode == StepMode::Picard ? ConvectionIntegrator::GradMode::Oseen
                                                       : ConvectionIntegrator::GradMode::Newton);
      auto J = I.velocity_block(s, C.get(), I.tr_c);  // + the time mass in a time step (tr_c = 0 otherwise)
      if (!lin(*J, Ru, Rp, du, dp, refine, lu_residual)) return false;
      d.SetSize(I.nu_loc + npt());
      for (int i = 0; i < I.nu_loc; ++i) d(i) = du(i);
      for (int i = 0; i < npt(); ++i) d(I.nu_loc + i) = dp(i);
      return true;
    }
  } nssys(I);
  nssys.lin = lin_solve;
  nssys.attempts = &step_attempts;
  nssys.stage = &step_stage;
  NewtonControl ctl;
  ctl.line_search = nl.line_search;
  ctl.armijo_c1 = nl.armijo_c1;
  ctl.line_search_halvings = nl.line_search_halvings;
  ctl.rtol = nl.newton_rtol;
  ctl.maxit = nl.newton_maxit;
  ctl.n_picard = nl.n_picard;
  ctl.floor_factor = nl.residual_floor_factor;  // consulted by run_stages (strategy Newton) only
  NonlinearDriver driver(nssys, ctl, diag_.history, diag_.iterations, diag_.line_search_min_step_accepts,
                         diag_.failure);
  driver.scale_name = "viscosity scale";
  mfem::Vector Xs(I.nu_loc + nssys.npt()), Rs(I.nu_loc + nssys.npt());
  auto to_state = [&] {
    for (int i = 0; i < I.nu_loc; ++i) Xs(i) = I.u(i);
    for (int i = 0; i < nssys.npt(); ++i) Xs(I.nu_loc + i) = I.p(i);
  };
  auto from_state = [&] {
    nssys.split(Xs, I.u, I.p);
    nssys.split(Rs, Ru, Rp);
  };
  auto newton_step = [&](double s, double den, bool picard, int stage, double& r) -> bool {
    to_state();
    const bool ok = driver.step(Xs, Rs, s, den, picard, stage, r);
    from_state();
    return ok;
  };

  double r = 0, den = 0;
  bool ok = true;
  const double s0 = scales.front();
  den = lift_norm(s0);
  diag_.residual_lift = den;
  if (!(std::isfinite(den))) {
    ok = fail("the residual of the lift is not finite");
  } else if (den == 0.0) {
    // no data and no force: the solution is the lift (zero); converged without iterations
    I.u = u_lift;
    I.p = p_zero;
    diag_.relative_residual = 0.0;
    diag_.converged = true;
    I.scale_final = scales.back();
  } else {
    // ---- the TRUE Stokes start: K(s0) d = -R_S(x_lift), no convective vector, no convective matrix
    // (not in a time step, which starts from the predictor)
    if (!transient) {
      I.u = u_lift;
      I.p = p_zero;
      I.residual(I.u, I.p, s0, false, Ru, Rp);
      auto K = I.velocity_block(s0, nullptr, 0.0);
      int refine = 0;
      double lures = -1;
      if (!lin_solve(*K, Ru, Rp, du, dp, refine, lures)) {
        ok = fail("linear solve failed in the Stokes step");
      } else {
        I.u.Add(1.0, du);
        I.p.Add(1.0, dp);
        I.residual(I.u, I.p, s0, true, Ru, Rp);
        r = rel(den);
        push("stokes", r, 1.0, 0.0, 0, s0, refine, lures);
        if (!std::isfinite(r)) ok = fail("the residual is not finite after the Stokes step");
      }
    }
    if (ok && transient) {
      // ---- one time step from the predictor
      I.residual(I.u, I.p, s0, true, Ru, Rp);
      r = rel(den);
      if (!std::isfinite(r)) {
        ok = fail("the residual of the predictor is not finite");
      } else if (semi) {
        if (r > gate) {
          auto C = std::unique_ptr<mfem::HypreParMatrix>(new mfem::HypreParMatrix(*I.oseen_full));
          C->EliminateBC(I.ess, mfem::Operator::DIAG_ZERO);
          auto J = I.velocity_block(s0, C.get(), I.tr_c);
          int refine = 0;
          double lures = -1;
          if (!lin_solve(*J, Ru, Rp, du, dp, refine, lures)) {
            ok = fail("linear solve failed in a semi-implicit step");
          } else {
            I.u.Add(1.0, du);
            I.p.Add(1.0, dp);
            I.residual(I.u, I.p, s0, true, Ru, Rp);
            r = rel(den);
            push("semi", r, 1.0, 0.0, 0, s0, refine, lures);
            ++diag_.iterations;
            if (!std::isfinite(r)) ok = fail("the residual is not finite after the semi-implicit solve");
          }
        }
        if (ok && r > gate) ok = fail("the semi-implicit linear solve missed the gate");
      } else {
        // Newton per step: the generic driver's step (line search per the options); the stopping rule is the step's
        const int maxit = nl.newton_maxit;
        int k = 0;
        while (ok && r > rtol && k < maxit) {
          ok = newton_step(s0, den, false, 0, r);
          ++k;
        }
        if (ok && r > rtol) ok = fail("Newton budget exhausted in a time step");
      }
    } else if (ok && nl.strategy == NonlinearStrategy::PicardOnly) {
      int it = 0;
      while (ok && r > rtol && it < nl.picard_maxit) {
        ok = newton_step(s0, den, true, 0, r);
        ++it;
      }
      if (ok && r > rtol) ok = fail("Picard iteration budget exhausted");
    } else if (ok && nl.strategy == NonlinearStrategy::PseudoTransient) {
      // ---- pseudo-transient continuation from the Stokes solution
      double dt = nl.ptc_dt0, r_prev = r;
      int it = 0;
      bool released = false;
      while (ok && r > rtol && it < nl.ptc_maxit) {
        I.residual(I.u, I.p, s0, true, Ru, Rp);
        auto C = I.conv_grad(I.u, ConvectionIntegrator::GradMode::Newton);
        auto J = I.velocity_block(s0, C.get(), opt_.rho / dt);
        int refine = 0;
        double lures = -1;
        if (!lin_solve(*J, Ru, Rp, du, dp, refine, lures)) {
          ok = fail("linear solve failed in a PTC step");
          break;
        }
        I.u.Add(1.0, du);
        I.p.Add(1.0, dp);
        I.residual(I.u, I.p, s0, true, Ru, Rp);
        r_prev = r;
        r = rel(den);
        push("ptc", r, 1.0, dt, 0, s0, refine, lures);
        ++diag_.iterations;
        ++it;
        if (!std::isfinite(r)) {
          ok = fail("the residual is not finite (PTC divergence)");
          break;
        }
        dt = dt * r_prev / r;  // switched evolution relaxation
        if (dt < 1e-10 * nl.ptc_dt0) {
          ok = fail("PTC pseudo-time step collapsed (dt < 1e-10 dt0)");
          break;
        }
        if (dt > nl.ptc_release_dt) {
          released = true;
          break;
        }
      }
      if (ok && released) {
        int k = 0;
        while (ok && r > rtol && k < nl.newton_maxit) {
          ok = newton_step(s0, den, false, 0, r);
          ++k;
        }
      }
      if (ok && r > rtol) ok = fail(released ? "Newton budget exhausted after the PTC release"
                                             : "PTC iteration budget exhausted");
    } else if (ok) {
      // ---- Newton with Picard start and Re continuation (viscosity scales): the generic driver
      to_state();
      ok = driver.run_stages(Xs, Rs, scales, lift_norm, den, diag_.residual_lift, r);
      from_state();
    }
    diag_.relative_residual = r;
    diag_.residual_floor = driver.residual_floor();
    diag_.converged = ok && std::isfinite(r) && r <= gate;
    // The iteration stopped because a step no longer reduced the residual and it stands at the round-off floor of
    // the residual evaluation: no state double arithmetic can represent has a meaningfully smaller residual, so the
    // solve is converged.  It is reported as such (stopped_at_floor next to the measured floor), never silently, and
    // the residual it accepts is still bounded (NonlinearDriver::floor_stop_rtol_ceiling times newton_rtol).
    if (!diag_.converged && ok && std::isfinite(r) && driver.stopped_at_floor()) {
      diag_.converged = true;
      diag_.stopped_at_floor = true;
    }
    if (!diag_.converged && diag_.failure.empty()) diag_.failure = "not converged";
    I.scale_final = scales.back();
  }
  // ---- accuracy gates on the linear algebra the verdict rests on
  // (1) the per-step direct solves: every one is recorded in the history (lu_residual) and the worst reported. The
  // enforced one is the FINAL step's, the one whose solution is reported, at max(newton_rtol, kFinalSolveResidualMax)
  // (sparse_lu.hpp says why not at newton_rtol): an intermediate Newton or Picard step with a less accurate linear
  // solve is an inexact step that changes the path, not the verdict, which is the nonlinear residual recomputed from
  // the assembled operators after the step.
  {
    const double target = std::min(1e-12, 1e-2 * rtol);  // the refinement target lin_solve_ sets
    diag_.lu_residual_max = -1;
    diag_.lu_target_missed = 0;
    for (const NonlinearStep& h : diag_.history) {
      if (h.lu_residual < 0) continue;
      diag_.lu_residual_max = std::max(diag_.lu_residual_max, h.lu_residual);
      if (!(h.lu_residual <= target)) ++diag_.lu_target_missed;
    }
    double last = -1;
    for (const NonlinearStep& h : diag_.history)
      if (h.lu_residual >= 0) last = h.lu_residual;
    if (!(last <= lu_gate)) {
      diag_.converged = false;
      if (diag_.failure.empty())
        diag_.failure = strcat("the final direct solve missed its accuracy target (refined LU residual ", last,
                               " > ", lu_gate, ")");
    }
  }
  // (2) the dual-norm mass solves: every verdict above is expressed in that norm, so it is not reported on an
  // unchecked one
  diag_.mass_lu_residual_max = I.mass_res_max;
  if (!(I.mass_res_max <= kDualMassResidualMax)) {
    diag_.converged = false;
    if (diag_.failure.empty() || diag_.failure == "not converged")
      diag_.failure = strcat("a dual-norm mass solve missed its accuracy target (worst relative residual ",
                           I.mass_res_max, " > ", kDualMassResidualMax,
                           "): every convergence verdict of this solver is expressed in that norm");
  }
  diag_.seconds = MPI_Wtime() - t0;
  diag_.mass_lu_escalated = I.mass_escalated;
  diag_.lu_fallbacks = diag_.lu_counts.escalated;
  // fields: velocity as solved; pressure mean-free (r-weighted) for an enclosed flow, raw otherwise
  I.ug.reset(new mfem::ParGridFunction(&S.velocity_space()));
  I.pg.reset(new mfem::ParGridFunction(&S.pressure_space()));
  I.ug->SetFromTrueDofs(I.u);
  mfem::Vector pn(I.np_loc);
  for (int i = 0; i < I.np_loc; ++i) pn(i) = I.p(i);
  if (I.M) {
    mfem::Vector a(I.p.GetData() + I.np_loc, I.na);
    I.M->set_fields(a);
  }
  if (I.nullspace) {
    double loc[2] = {I.mean_weights * pn, I.mean_weights.Sum()}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, I.comm);
    pn -= glob[0] / glob[1];
  }
  I.pg->SetFromTrueDofs(pn);
  I.solved = true;
  if (diag_.converged) diag_.failure.clear();
  return diag_;
}

const mfem::ParGridFunction& NavierStokesSolver::velocity() const {
  AXMP_REQUIRE(impl_->solved && diag_.converged,
              "the Navier-Stokes solve did not converge; refusing to return its velocity", "",
              strcat("final steady relative residual ", diag_.relative_residual, " > newton_rtol ",
                     opt_.nonlinear.newton_rtol, " (", diag_.failure, "). Inspect diagnostics() and the history; "
                     "use Re continuation or pseudo-transient continuation for a hard problem."));
  return *impl_->ug;
}

const mfem::GridFunction& NavierStokesSolver::potential() const {
  AXMP_REQUIRE(impl_->M, "potential(): this solver has no MHD coupling");
  AXMP_REQUIRE(impl_->solved && diag_.converged,
              "the MHD solve did not converge; refusing to return its potential", "",
              strcat("final steady relative residual ", diag_.relative_residual, " (", diag_.failure, ")"));
  return impl_->M->potential();
}

const MhdDiagnostics& NavierStokesSolver::mhd_diagnostics() const {
  AXMP_REQUIRE(impl_->M, "mhd_diagnostics(): this solver has no MHD coupling");
  return impl_->M->diagnostics();
}

const mfem::ParGridFunction& NavierStokesSolver::pressure() const {
  AXMP_REQUIRE(impl_->solved && diag_.converged,
              "the Navier-Stokes solve did not converge; refusing to return its pressure", "",
              strcat("final steady relative residual ", diag_.relative_residual, " (", diag_.failure, ")"));
  return *impl_->pg;
}

// ------------------------------------------------------------------------------------------- free-DOF API
std::pair<long, long> NavierStokesSolver::layout() const {
  const Impl& I = *impl_;
  long nf = 0;
  for (int i = 0; i < I.nu_loc; ++i) nf += I.ess_mask[i] ? 0 : 1;
  long nq = I.np_loc - (I.pin_local >= 0 ? 1 : 0);
  for (int i = 0; i < I.na; ++i) nq += I.M->ess_mask()[i] ? 0 : 1;
  return {nf, nq};
}

namespace {
void require_one_rank(MPI_Comm c, const char* what) {
  AXMP_REQUIRE(comm_size(c) == 1, strcat(what, "() needs one MPI rank (the free-DOF state is rank-local)"));
}
}  // namespace

std::vector<double> NavierStokesSolver::state() const {
  const Impl& I = *impl_;
  require_one_rank(I.comm, "state");
  std::vector<double> x;
  for (int i = 0; i < I.nu_loc; ++i)
    if (!I.ess_mask[i]) x.push_back(I.u(i));
  for (int i = 0; i < I.np_loc; ++i)
    if (i != I.pin_local) x.push_back(I.p(i));
  for (int i = 0; i < I.na; ++i)
    if (!I.M->ess_mask()[i]) x.push_back(I.p(I.np_loc + i));
  return x;
}

namespace {
// x_free -> (u with the lift on the essential DOFs, p with the pinned value 0 [, aux a with its lift on the essential
// aux DOFs appended to p])
void scatter(const std::vector<double>& x, const mfem::Vector& lift, const std::vector<char>& ess_mask, long pin,
             int nu, int np, mfem::Vector& u, mfem::Vector& p, bool zero_lift, const MhdCoupling* M = nullptr) {
  const int na = M ? M->na() : 0;
  u.SetSize(nu);
  p.SetSize(np + na);
  size_t k = 0;
  for (int i = 0; i < nu; ++i) {
    if (ess_mask[i]) {
      u(i) = zero_lift ? 0.0 : lift(i);
    } else {
      AXMP_REQUIRE(k < x.size(), "state vector too short for the layout", "x_free");
      u(i) = x[k++];
    }
  }
  for (int i = 0; i < np; ++i) {
    if (i == pin) {
      p(i) = 0.0;
    } else {
      AXMP_REQUIRE(k < x.size(), "state vector too short for the layout", "x_free");
      p(i) = x[k++];
    }
  }
  for (int i = 0; i < na; ++i) {
    if (M->ess_mask()[i]) {
      p(np + i) = zero_lift ? 0.0 : M->lift()(i);
    } else {
      AXMP_REQUIRE(k < x.size(), "state vector too short for the layout", "x_free");
      p(np + i) = x[k++];
    }
  }
  AXMP_REQUIRE(k == x.size(), strcat("state vector has ", x.size(), " entries; the layout has ", k), "x_free");
}
std::vector<double> gather(const mfem::Vector& Ru, const mfem::Vector& Rp, const std::vector<char>& ess_mask,
                           long pin, int np = -1, const MhdCoupling* M = nullptr) {
  std::vector<double> r;
  if (np < 0) np = Rp.Size();
  for (int i = 0; i < Ru.Size(); ++i)
    if (!ess_mask[i]) r.push_back(Ru(i));
  for (int i = 0; i < np; ++i)
    if (i != pin) r.push_back(Rp(i));
  if (M)
    for (int i = 0; i < M->na(); ++i)
      if (!M->ess_mask()[i]) r.push_back(Rp(np + i));
  return r;
}
}  // namespace

std::vector<double> NavierStokesSolver::residual(const std::vector<double>& x, double s) const {
  const Impl& I = *impl_;
  require_one_rank(I.comm, "residual");
  mfem::Vector u, p, Ru, Rp;
  scatter(x, I.lift, I.ess_mask, I.pin_local, I.nu_loc, I.np_loc, u, p, false, I.M.get());
  I.residual(u, p, s, true, Ru, Rp);
  return gather(Ru, Rp, I.ess_mask, I.pin_local, I.np_loc, I.M.get());
}

std::vector<double> NavierStokesSolver::jacobian_action(const std::vector<double>& x, const std::vector<double>& d,
                                                        double s) const {
  const Impl& I = *impl_;
  require_one_rank(I.comm, "jacobian_action");
  mfem::Vector u, p, du, dp;
  scatter(x, I.lift, I.ess_mask, I.pin_local, I.nu_loc, I.np_loc, u, p, false, I.M.get());
  scatter(d, I.lift, I.ess_mask, I.pin_local, I.nu_loc, I.np_loc, du, dp, true, I.M.get());
  auto C = I.conv_grad(u, ConvectionIntegrator::GradMode::Newton);
  mfem::Vector Ju(I.nu_loc), Jp(I.np_loc + I.na), t(I.nu_loc);
  mfem::Vector dph(dp.GetData(), I.np_loc), Jph(Jp.GetData(), I.np_loc);
  I.A0->Mult(du, Ju);
  Ju *= s;
  C->Mult(du, t);
  Ju += t;
  I.Bt->Mult(dph, t);
  Ju += t;
  if (I.M) {
    mfem::Vector da(dp.GetData() + I.np_loc, I.na), Ja(I.na);
    I.M->Kuu_e().Mult(du, t);
    Ju += t;
    I.M->Kua_e().Mult(da, t);
    Ju += t;
    I.M->Kau_e().Mult(du, Ja);
    I.M->Kaa_e().AddMult(da, Ja, 1.0);
    for (int i = 0; i < I.na; ++i) Jp(I.np_loc + i) = Ja(i);
  }
  I.zero_ess(Ju);
  I.B->Mult(du, Jph);
  if (I.pin_local >= 0) Jp(static_cast<int>(I.pin_local)) = 0.0;
  return gather(Ju, Jp, I.ess_mask, I.pin_local, I.np_loc, I.M.get());
}

DistCsr NavierStokesSolver::jacobian(const std::vector<double>& x, double s) const {
  const Impl& I = *impl_;
  require_one_rank(I.comm, "jacobian");
  AXMP_REQUIRE(!I.M, "jacobian(): the assembled Jacobian is not offered with MHD (use jacobian_action)");
  mfem::Vector u, p;
  scatter(x, I.lift, I.ess_mask, I.pin_local, I.nu_loc, I.np_loc, u, p, false);
  auto C = I.conv_grad(u, ConvectionIntegrator::GradMode::Newton);
  auto J = I.velocity_block(s, C.get(), 0.0);
  mfem::Array2D<const mfem::HypreParMatrix*> blocks(2, 2);
  blocks(0, 0) = J.get();
  blocks(0, 1) = I.Bt;
  blocks(1, 0) = I.B;
  blocks(1, 1) = nullptr;
  std::unique_ptr<mfem::HypreParMatrix> K(mfem::HypreParMatrixFromBlocks(blocks));
  DistCsr full = to_dist(*K, I.comm);
  // keep the free rows and columns (x_free layout)
  const int n = full.local_rows;
  std::vector<int> map(n, -1);
  int k = 0;
  for (int i = 0; i < n; ++i) {
    const bool drop = i < I.nu_loc ? I.ess_mask[i] : (i - I.nu_loc == I.pin_local);
    if (!drop) map[i] = k++;
  }
  DistCsr D;
  D.comm = I.comm;
  D.n_global = k;
  D.row_begin = 0;
  D.local_rows = k;
  D.rowptr.push_back(0);
  for (int i = 0; i < n; ++i) {
    if (map[i] < 0) continue;
    for (int q = full.rowptr[i]; q < full.rowptr[i + 1]; ++q)
      if (map[full.col[q]] >= 0) {
        D.col.push_back(map[full.col[q]]);
        D.val.push_back(full.val[q]);
      }
    D.rowptr.push_back(static_cast<int>(D.col.size()));
  }
  return D;
}

double NavierStokesSolver::dual_norm(const std::vector<double>& R) const {
  const Impl& I = *impl_;
  require_one_rank(I.comm, "dual_norm");
  mfem::Vector Ru, Rp;
  scatter(R, I.lift, I.ess_mask, I.pin_local, I.nu_loc, I.np_loc, Ru, Rp, true, I.M.get());
  return I.dual(Ru, Rp);
}

FullResidual NavierStokesSolver::full_residual() const {
  const Impl& I = *impl_;
  require_one_rank(I.comm, "full_residual");
  AXMP_REQUIRE(!I.M, "full_residual(): not offered with MHD");
  AXMP_REQUIRE(I.solved && diag_.converged, "full_residual(): needs a converged solve");
  const double s = I.scale_final;
  mfem::Vector pt;
  I.pg->GetTrueDofs(pt);
  mfem::Vector R(I.nu_loc), t(I.nu_loc);
  I.A0->Mult(I.u, R);
  R *= s;
  I.N->Mult(I.u, t);
  R += t;
  I.B0->MultTranspose(pt, t);
  R += t;
  R -= I.F;
  FullResidual out;
  out.r_u.assign(R.GetData(), R.GetData() + R.Size());
  mfem::Vector Ru, Rp;
  I.residual(I.u, I.p, s, true, Ru, Rp);
  out.r_p.assign(Rp.GetData(), Rp.GetData() + Rp.Size());
  mfem::ParFiniteElementSpace& ufes = I.S->velocity_space();
  const int nn = ufes.GetNDofs();
  out.node_xy.assign(2 * nn, 0.0);
  mfem::Array<int> dofs;
  for (int e = 0; e < I.mesh->GetNE(); ++e) {
    ufes.GetElementDofs(e, dofs);
    const mfem::IntegrationRule& nodes = ufes.GetFE(e)->GetNodes();
    mfem::ElementTransformation* T = I.mesh->GetElementTransformation(e);
    for (int k = 0; k < dofs.Size(); ++k) {
      double x[3];
      mfem::Vector X(x, 2);
      T->Transform(nodes.IntPoint(k), X);
      const int d = dofs[k] >= 0 ? dofs[k] : -1 - dofs[k];
      out.node_xy[2 * d] = x[0];
      out.node_xy[2 * d + 1] = x[1];
    }
  }
  std::vector<int> attrs(I.mesh->bdr_attributes.begin(), I.mesh->bdr_attributes.end());
  for (int a : attrs) {
    std::vector<char> seen(nn, 0);
    std::vector<long> list;
    for (int be = 0; be < I.mesh->GetNBE(); ++be) {
      if (I.mesh->GetBdrAttribute(be) != a) continue;
      ufes.GetBdrElementDofs(be, dofs);
      for (int d0 : dofs) {
        const int d = d0 >= 0 ? d0 : -1 - d0;
        if (!seen[d]) {
          seen[d] = 1;
          list.push_back(d);
        }
      }
    }
    std::sort(list.begin(), list.end());
    out.attribute_nodes.push_back({a, list});
  }
  return out;
}

Reaction NavierStokesSolver::reaction(const std::vector<int>& body, const std::vector<double>& center,
                                      ViscousForm residual_form) const {
  const Impl& I = *impl_;
  AXMP_REQUIRE(I.solved && diag_.converged, "reaction(): needs a converged solve");
  AXMP_REQUIRE(!I.M, "reaction(): not offered with MHD");
  StokesSolver& S = *I.S;
  const StokesOptions& so = opt_.stokes;
  const double s = I.scale_final;
  MPI_Comm comm = I.comm;
  const int nattr = I.mesh->bdr_attributes.Max();
  mfem::Array<int> marker(nattr);
  marker = 0;
  for (int a : body) {
    AXMP_REQUIRE(a >= 1 && a <= nattr, strcat("reaction(): attribute ", a, " is not a boundary attribute of the mesh"));
    marker[a - 1] = 1;
  }
  mfem::ParFiniteElementSpace& ufes = S.velocity_space();
  // the viscous residual operator: the solved one (A0) when residual_form is the solved form, else the requested
  // form on the solve's rule (constant mu: the S-form residual is the force whatever form
  // was solved; variable mu: only the solved form is consistent)
  std::unique_ptr<mfem::HypreParMatrix> Aown;
  const mfem::HypreParMatrix* A = I.A0;
  if (residual_form != so.form) {
    mfem::ParBilinearForm a(&ufes);
    mfem::ConstantCoefficient zero(0.0);
    const int qv = so.viscous_quad_order();
    if (I.axi) {
      a.AddDomainIntegrator(new AxiViscousIntegrator(*so.nu, I.vdim, residual_form == ViscousForm::SymmetricStress,
                                                     0.0, qv));
    } else if (residual_form == ViscousForm::Laplacian) {
      a.AddDomainIntegrator(new RuleBFI(new mfem::VectorDiffusionIntegrator(*so.nu), qv));
    } else {
      a.AddDomainIntegrator(new RuleBFI(new mfem::ElasticityIntegrator(zero, *so.nu), qv));
    }
    a.Assemble();
    a.Finalize();
    Aown.reset(a.ParallelAssemble());
    A = Aown.get();
  }
  mfem::Vector pt;
  I.pg->GetTrueDofs(pt);
  mfem::Vector R(I.nu_loc), tmp(I.nu_loc);
  A->Mult(I.u, R);
  R *= s;
  I.N->Mult(I.u, tmp);  // the convective term of the solved form (dynamic units: rho is in it)
  R += tmp;
  I.B0->MultTranspose(pt, tmp);
  R += tmp;
  R -= I.F;
  auto restricted = [&](mfem::VectorCoefficient& field, int comp_only) {
    mfem::ParGridFunction w(&ufes);
    w.ProjectCoefficient(field);
    mfem::Vector wt;
    w.GetTrueDofs(wt);
    mfem::Vector keep(wt.Size());
    keep = 0.0;
    for (int c = 0; c < I.vdim; ++c) {
      if (comp_only >= 0 && c != comp_only) continue;
      mfem::Array<int> l;
      ufes.GetEssentialTrueDofs(marker, l, c);
      for (int t : l) keep(t) = wt(t);
    }
    return -mfem::InnerProduct(comm, keep, R);
  };
  Reaction out;
  out.residual_form = to_string(residual_form);
  const int vd = I.vdim;
  if (!I.axi) {
    for (int i = 0; i < 2; ++i) {
      mfem::Vector e(vd);
      e = 0.0;
      e(i) = 1.0;
      mfem::VectorConstantCoefficient ec(e);
      out.force[i] = restricted(ec, i);
    }
    const double cx = center.size() > 0 ? center[0] : 0.0, cy = center.size() > 1 ? center[1] : 0.0;
    mfem::VectorFunctionCoefficient rot(vd, [=](const mfem::Vector& x, mfem::Vector& v) {
      v.SetSize(vd);
      v = 0.0;
      v(0) = -(x(1) - cy);
      v(1) = x(0) - cx;
    });
    out.torque = restricted(rot, -1);
  } else {
    const double twopi = 2.0 * M_PI;
    mfem::Vector e(vd);
    e = 0.0;
    e(1) = 1.0;
    mfem::VectorConstantCoefficient ez(e);
    out.force[2] = twopi * restricted(ez, 1);
    mfem::VectorFunctionCoefficient rot(vd, [=](const mfem::Vector& x, mfem::Vector& v) {
      v.SetSize(vd);
      v = 0.0;
      v(2) = x(0);
    });
    out.torque = twopi * restricted(rot, 2);
  }
  return out;
}

// ============================================================================================== time-step hooks
// (transient_ns.hpp): everything the time stepper needs from the
// steady core. The hooks that change the state or the system refuse an MHD solver (no potential unknowns here).
namespace {
void require_no_mhd(const NavierStokesSolver::Impl& I, const char* what) {
  AXMP_REQUIRE(!I.M, strcat(what, "(): not offered with the MHD coupling (time stepping carries no potential unknowns)"));
}
}  // namespace

void NavierStokesSolver::refresh_data() {
  Impl& I = *impl_;
  require_no_mhd(I, "refresh_data");
  StokesSolver& S = *I.S;
  S.refresh_data();
  StokesSolver::Impl& SI = *S.impl_;
  I.F = SI.Fvec;
  I.rhs_p = SI.rhs->GetBlock(1);
  I.lift = SI.x->GetBlock(0);
  diag_.linear.flux_defect = S.diag_.flux_defect;
  diag_.linear.flux_defect_abs = S.diag_.flux_defect_abs;
}

void NavierStokesSolver::set_time_term(double c, const mfem::Vector& hist) {
  Impl& I = *impl_;
  require_no_mhd(I, "set_time_term");
  if (c == 0.0) {  // clear the term
    I.tr_c = 0.0;
    I.tr_hist.SetSize(0);
    return;
  }
  I.tr_c = c;
  I.tr_hist = hist;
}

void NavierStokesSolver::set_advecting(const mfem::Vector& w) {
  require_no_mhd(*impl_, "set_advecting");
  impl_->advect = w;
}

void NavierStokesSolver::set_state(const mfem::Vector& u, const mfem::Vector& p, bool lift_boundary) {
  Impl& I = *impl_;
  require_no_mhd(I, "set_state");
  AXMP_REQUIRE(u.Size() == I.nu_loc && p.Size() == I.np_loc, "state vectors have the wrong size", "set_state");
  I.u = u;
  I.p = p;
  if (lift_boundary)
    for (int i : I.ess) I.u(i) = I.lift(i);
}

const mfem::Vector& NavierStokesSolver::state_u() const { return impl_->u; }
const mfem::Vector& NavierStokesSolver::state_p() const { return impl_->p; }
const mfem::Vector& NavierStokesSolver::lift() const { return impl_->lift; }
mfem::ParFiniteElementSpace& NavierStokesSolver::pressure_space() { return impl_->S->pressure_space(); }
void NavierStokesSolver::normalize_pressure(mfem::Vector& pn) const {
  const Impl& I = *impl_;
  if (!I.nullspace) return;
  double loc[2] = {I.mean_weights * pn, I.mean_weights.Sum()}, glob[2];
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, I.comm);
  pn -= glob[0] / glob[1];
}
int NavierStokesSolver::velocity_size() const { return impl_->nu_loc; }
int NavierStokesSolver::pressure_size() const { return impl_->np_loc; }

const mfem::ParGridFunction& NavierStokesSolver::velocity_field() const {
  AXMP_REQUIRE(impl_->solved, "no time step has been solved yet", "velocity_field");
  return *impl_->ug;
}
const mfem::ParGridFunction& NavierStokesSolver::pressure_field() const {
  AXMP_REQUIRE(impl_->solved, "no time step has been solved yet", "pressure_field");
  return *impl_->pg;
}

void NavierStokesSolver::mass_mult(const mfem::Vector& x, mfem::Vector& y) const {
  y.SetSize(impl_->nu_loc);
  impl_->Mu->Mult(x, y);
}
void NavierStokesSolver::mass_mult_time(const mfem::Vector& x, mfem::Vector& y) const {
  y.SetSize(impl_->nu_loc);
  impl_->Mt->Mult(x, y);
}
void NavierStokesSolver::zero_essential(mfem::Vector& v) const { impl_->zero_ess(v); }

double NavierStokesSolver::mass_norm(const mfem::Vector& x) const {
  Impl& I = *impl_;
  const mfem::HypreParMatrix* M = I.Mu.get();
  mfem::Vector y(I.nu_loc);
  M->Mult(x, y);
  return std::sqrt(std::max(gdot(I.comm, x, y), 0.0));
}

bool NavierStokesSolver::project_divergence_free(const mfem::Vector& u_target) {
  Impl& I = *impl_;
  require_no_mhd(I, "project_divergence_free");
  // The projection is accepted on the backward error of its own saddle-point solve (formed below), against
  // kProjectionResidualMax (sparse_lu.hpp says why the bound is a constant and holds the measurements behind it).
  const double target = kProjectionResidualMax;
  diag_.projection_residual = -1;
  diag_.projection_converged = false;
  diag_.projection_lu_attempts.clear();
  mfem::Vector u0(u_target), p0(I.np_loc), Ru, Rp, t(I.nu_loc), du, dp;
  p0 = 0.0;
  for (int i : I.ess) u0(i) = I.lift(i);
  // R_u = M (u0 - u_I) on the free rows (zero unless the data differ), R_p = B u0 - rhs_p (the lifted continuity)
  {
    mfem::Vector d(u0);
    d -= u_target;
    I.Mu->Mult(d, t);
    I.zero_ess(t);
    Ru = t;
    Rp.SetSize(I.np_loc);
    I.B->Mult(u0, Rp);
    Rp -= I.rhs_p;
    if (I.pin_local >= 0) Rp(static_cast<int>(I.pin_local)) = 0.0;
  }
  // The measure. The residual of the saddle system [M B^T; B 0] (du, dp) = -(R_u, R_p) is formed with the ASSEMBLED
  // operators (not the CSR the LU used), and each of its two blocks is compared, in the Euclidean norm, with the sizes
  // of the terms that make that block up:
  //   momentum     || M du + B^T dp + R_u ||   /   || |M| |du| + |B^T| |dp| + |R_u| ||
  //   continuity   || B (u0 + du) - rhs_p ||   /   || |B| (|u0| + |du|) + |rhs_p| ||
  // (|.| entrywise). projection_residual is the larger of the two ratios: a backward error, in units of the round-off
  // of evaluating each block (sparse_lu.hpp says why it is this and not a residual relative to the data, and holds
  // the measurements). The two blocks are never added, because they carry different units (the momentum rows scale
  // like L^2 and the continuity rows like L, L the length unit), and each denominator scales exactly like its
  // numerator. It is not LUStats::rel_residual, and the two numbers are not interchangeable. Collective.
  auto measure = [&](const mfem::Vector& eu, const mfem::Vector& ep) {
    mfem::Vector qu(I.nu_loc), qp(I.np_loc), su(I.nu_loc), sp(I.np_loc), qb(I.nu_loc), a(I.nu_loc), ap(I.np_loc);
    I.Mu_one->Mult(eu, qu);
    mfem::Vector dph(const_cast<double*>(ep.GetData()), I.np_loc);
    I.Bt->Mult(dph, qb);
    qu += qb;
    qu += Ru;
    I.zero_ess(qu);
    I.B->Mult(eu, qp);
    qp += Rp;
    for (int i = 0; i < I.nu_loc; ++i) a(i) = std::fabs(eu(i));
    I.Mu_one->AbsMult(a, su);
    for (int i = 0; i < I.np_loc; ++i) ap(i) = std::fabs(ep(i));
    I.Bt->AbsMult(1.0, ap, 1.0, su);
    for (int i = 0; i < I.nu_loc; ++i) su(i) += std::fabs(Ru(i));
    I.zero_ess(su);
    for (int i = 0; i < I.nu_loc; ++i) a(i) = std::fabs(u0(i)) + std::fabs(eu(i));
    I.B->AbsMult(a, sp);
    for (int i = 0; i < I.np_loc; ++i) sp(i) += std::fabs(I.rhs_p(i));
    // The pinned pressure row is not an equation of the solve, so it is left out. The continuity rows sum to the
    // boundary flux, so what that row holds is minus the sum of the other rows' round-off (benign, and larger than
    // any one of them: with Eigen's LU 3.7e-15 against 5e-17 on a 100 x 1 channel).
    if (I.pin_local >= 0) qp(static_cast<int>(I.pin_local)) = sp(static_cast<int>(I.pin_local)) = 0.0;
    double loc[4] = {qu * qu, su * su, qp * qp, sp * sp}, glob[4];
    MPI_Allreduce(loc, glob, 4, MPI_DOUBLE, MPI_SUM, I.comm);
    // a block whose terms all vanish (a zero field) has a zero residual and a ratio of zero
    auto ratio = [](double r2, double s2) {
      if (!std::isfinite(r2) || !std::isfinite(s2)) return std::numeric_limits<double>::infinity();
      if (r2 == 0) return 0.0;
      return s2 > 0 ? std::sqrt(r2 / s2) : std::numeric_limits<double>::infinity();
    };
    return std::max(ratio(glob[0], glob[1]), ratio(glob[2], glob[3]));
  };
  // The chain accepts a stage on this very gate: the measure of the stage's (du, dp), read from its solution exactly as
  // lin_solve_ hands it over (velocity rows, Dirichlet entries zero; pressure rows), so the accepted stage's measure is
  // projection_residual bit for bit (both are this function of the same numbers).
  LUAcceptance acceptance;
  acceptance.kind = LUAcceptance::Kind::Verdict;
  acceptance.bound = target;
  acceptance.name = "projection residual";
  acceptance.verdict = [&](const std::vector<double>& x) {
    mfem::Vector eu(I.nu_loc), ep(I.np_loc);
    for (int i = 0; i < I.nu_loc; ++i) eu(i) = x[i];
    for (int i : I.ess) eu(i) = 0.0;
    for (int i = 0; i < I.np_loc; ++i) ep(i) = x[I.nu_loc + i];
    return measure(eu, ep);
  };
  {
    int refine = 0;
    double lures = -1;
    const bool solved = lin_solve_(*I.Mu_one, Ru, Rp, du, dp, refine, lures, acceptance);
    diag_.projection_lu_attempts = diag_.linear.lu_attempts;
    if (!solved) {
      I.solved = false;
      return false;
    }
    // the test hook: an inaccurate solve of known size
    if (projection_perturbation_ != 0) du *= 1.0 + projection_perturbation_;
    if (projection_perturbation_p_ != 0) dp *= 1.0 + projection_perturbation_p_;
  }
  diag_.projection_residual = measure(du, dp);
  diag_.projection_converged = std::isfinite(diag_.projection_residual) && diag_.projection_residual <= target;
  if (!diag_.projection_converged) {
    I.solved = false;  // the state is untouched; velocity_field() / pressure_field() refuse
    return false;
  }
  u0 += du;
  I.u = u0;
  I.p = dp;
  // fields for evaluation
  I.ug.reset(new mfem::ParGridFunction(&I.S->velocity_space()));
  I.pg.reset(new mfem::ParGridFunction(&I.S->pressure_space()));
  I.ug->SetFromTrueDofs(I.u);
  mfem::Vector pn(I.p);
  if (I.nullspace) {
    double loc[2] = {I.mean_weights * pn, I.mean_weights.Sum()}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, I.comm);
    pn -= glob[0] / glob[1];
  }
  I.pg->SetFromTrueDofs(pn);
  I.solved = true;
  return true;
}

}  // namespace axmp::fluids
