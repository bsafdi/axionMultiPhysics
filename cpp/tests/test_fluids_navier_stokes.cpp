// Steady Navier–Stokes (cpp/include/axionmultiphysics/fluids/{convection,navier_stokes}.hpp): the C++ unit tests (the
// Python ones are tests/test_fluids_navier_stokes_api.py):
//   U2  planar standard residual and Newton Jacobian = MFEM's VectorConvectionNLFIntegrator (Q = rho), element by
//       element (an independent implementation);
//   U3  planar skew Oseen matrix O(w) (rho = 1) = MFEM's SkewSymmetricVectorConvectionNLFIntegrator::AssembleElementGrad
//       at w (the antisymmetrized c_A as an operator) + the element boundary matrix 1/2 int_dK (w.n)(du.v), element by
//       element; and the skew residual = O(u) u;
//   U4  axisymmetric kernel = a 3D Cartesian evaluation of rho ((u.grad)u).v for theta-independent fields (e_r, e_theta
//       frame at phi = 0, d/dy = (1/r) d/dphi), random element data, both forms;
//   U8  the refusals of unsupported options (grad-div, MINRES, a pressure point, slip/periodic, swirl-free axisymmetry) and
//       of the transient strategies with the MHD coupling;
//   U9  (when run on more than one rank, e.g. mpirun -n 4) the solution and the history equal those of a one-rank
//       solve of the same problem to round-off;
//   U10 AxiViscousIntegrator honours the given rule (degree 6 differs from the degree-5 fallback on a hoop-term
//       integral; order 0 is still that fallback), and the Stokes reaction's re-assembly site (stokes.cpp, the other
//       form's residual) runs with the solve's rule (a smoke check: finite torques);
//   U11 the stop at the measured round-off floor of the residual: a gate just below the floor is met by a stop
//       there, with the same state as the solve that reaches 1e-12, while a residual far above the floor, a floor
//       far above the gate, and residual_floor_factor = 0 are not accepted.
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fluids/axi_integrators.hpp"
#include "axionmultiphysics/fluids/convection.hpp"
#include "axionmultiphysics/fluids/navier_stokes.hpp"
#include "axionmultiphysics/fluids/stokes.hpp"

using namespace axmp::fluids;
static int failures = 0;
static bool rank0 = true;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

static double maxabs(const mfem::DenseMatrix& A) {
  double m = 0;
  for (int i = 0; i < A.Height(); ++i)
    for (int j = 0; j < A.Width(); ++j) m = std::max(m, std::fabs(A(i, j)));
  return m;
}

// a slightly distorted mesh on [0.2, 1.2] x [0, 1] (r > 0 for the axisymmetric checks)
static mfem::Mesh test_mesh() {
  mfem::Mesh m = mfem::Mesh::MakeCartesian2D(3, 3, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  for (int v = 0; v < m.GetNV(); ++v) {
    double* x = m.GetVertex(v);
    const double dx = 0.03 * std::sin(7 * x[0] + 3 * x[1]) * x[0] * (1 - x[0]) * x[1] * (1 - x[1]);
    x[0] = 0.2 + x[0] + dx;
    x[1] = x[1] - dx;
  }
  return m;
}

// the element boundary matrix 1/2 int_dK (w.n)(du . v), du, v in the element's vector P2 basis (component blocks)
static void boundary_matrix(mfem::Mesh& mesh, int e, const mfem::FiniteElement& fe, const mfem::Vector& welem,
                            mfem::DenseMatrix& Mb) {
  const int nd = fe.GetDof();
  Mb.SetSize(2 * nd);
  Mb = 0.0;
  mfem::Array<int> edges, ori;
  mesh.GetElementEdges(e, edges, ori);
  mfem::Vector N(nd), nor(2);
  for (int f : edges) {
    mfem::FaceElementTransformations* Tr = mesh.GetFaceElementTransformations(f);
    const bool first = Tr->Elem1No == e;
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 8);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetAllIntPoints(&ip);
      fe.CalcShape(first ? Tr->GetElement1IntPoint() : Tr->GetElement2IntPoint(), N);
      mfem::CalcOrtho(Tr->Jacobian(), nor);  // outward from Elem1, length |J|
      if (!first) nor.Neg();
      double w[2] = {0, 0};
      for (int c = 0; c < 2; ++c)
        for (int a = 0; a < nd; ++a) w[c] += welem(c * nd + a) * N(a);
      const double wn = w[0] * nor(0) + w[1] * nor(1);
      for (int c = 0; c < 2; ++c)
        for (int a = 0; a < nd; ++a)
          for (int b = 0; b < nd; ++b) Mb(c * nd + a, c * nd + b) += 0.5 * ip.weight * wn * N(a) * N(b);
    }
  }
}

static void u2_u3() {
  mfem::Mesh mesh = test_mesh();
  mfem::H1_FECollection fec(2, 2);
  mfem::FiniteElementSpace fes(&mesh, &fec, 2);
  std::mt19937 gen(11);
  std::normal_distribution<double> nd01;
  const double rho = 1.37;
  mfem::ConstantCoefficient Q(rho), one(1.0);
  ConvectionIntegrator std_k(rho, 2, false, ConvectiveForm::Standard);
  ConvectionIntegrator skew_k(1.0, 2, false, ConvectiveForm::Skew);
  mfem::VectorConvectionNLFIntegrator mf(Q);
  mfem::SkewSymmetricVectorConvectionNLFIntegrator mfs(one);
  mf.SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 5));
  mfs.SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 5));
  double d_res = 0, d_jac = 0, d_skew = 0, d_skres = 0, scale = 0;
  for (int e = 0; e < mesh.GetNE(); ++e) {
    const mfem::FiniteElement& fe = *fes.GetFE(e);
    mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
    mfem::Vector u(2 * fe.GetDof());
    for (int i = 0; i < u.Size(); ++i) u(i) = nd01(gen);
    mfem::Vector r1, r2;
    mfem::DenseMatrix J1, J2;
    std_k.AssembleElementVector(fe, T, u, r1);
    mf.AssembleElementVector(fe, T, u, r2);
    std_k.set_grad_mode(ConvectionIntegrator::GradMode::Newton);
    std_k.AssembleElementGrad(fe, T, u, J1);
    mf.AssembleElementGrad(fe, T, u, J2);
    r1 -= r2;
    d_res = std::max(d_res, r1.Normlinf() / r2.Normlinf());
    scale = maxabs(J2);
    J1 -= J2;
    d_jac = std::max(d_jac, maxabs(J1) / scale);
    // U3: O_skew(w) = MFEM skew grad at w + 1/2 int_dK (w.n)(du.v)
    mfem::DenseMatrix O, A, Mb;
    skew_k.set_grad_mode(ConvectionIntegrator::GradMode::Oseen);
    skew_k.AssembleElementGrad(fe, T, u, O);
    mfs.AssembleElementGrad(fe, T, u, A);
    boundary_matrix(mesh, e, fe, u, Mb);
    A += Mb;
    const double s = maxabs(O);
    O -= A;
    d_skew = std::max(d_skew, maxabs(O) / s);
    // the skew residual is O(u) u
    mfem::Vector rs, Ou(u.Size());
    skew_k.AssembleElementVector(fe, T, u, rs);
    skew_k.AssembleElementGrad(fe, T, u, O);
    O.Mult(u, Ou);
    Ou -= rs;
    d_skres = std::max(d_skres, Ou.Normlinf() / rs.Normlinf());
  }
  CHECK(d_res < 1e-13, "U2 residual vs MFEM VectorConvectionNLFIntegrator: rel %.3e", d_res);
  CHECK(d_jac < 1e-13, "U2 Newton Jacobian vs MFEM: rel %.3e", d_jac);
  CHECK(d_skew < 1e-12, "U3 skew Oseen vs MFEM skew grad + boundary matrix: rel %.3e", d_skew);
  CHECK(d_skres < 1e-13, "U3 skew residual vs O(u) u: rel %.3e", d_skres);
  if (rank0) std::printf("U2/U3: res %.2e jac %.2e skew %.2e skew-res %.2e\n", d_res, d_jac, d_skew, d_skres);
}

static void u4() {
  mfem::Mesh mesh = test_mesh();
  mfem::H1_FECollection fec(2, 2);
  mfem::FiniteElementSpace fes(&mesh, &fec, 3);
  std::mt19937 gen(5);
  std::normal_distribution<double> nd01;
  const double rho = 0.8;
  double dmax = 0;
  for (int skew = 0; skew < 2; ++skew) {
    ConvectionIntegrator k(rho, 3, true, skew ? ConvectiveForm::Skew : ConvectiveForm::Standard);
    for (int e = 0; e < mesh.GetNE(); ++e) {
      const mfem::FiniteElement& fe = *fes.GetFE(e);
      const int nd = fe.GetDof();
      mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
      mfem::Vector u(3 * nd), r1, r2(3 * nd);
      for (int i = 0; i < u.Size(); ++i) u(i) = nd01(gen);
      k.AssembleElementVector(fe, T, u, r1);
      r2 = 0.0;
      mfem::Vector N(nd);
      mfem::DenseMatrix dN(nd, 2);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe.GetGeomType(), 6);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        T.SetIntPoint(&ip);
        fe.CalcShape(ip, N);
        fe.CalcPhysDShape(T, dN);
        double x[3];
        mfem::Vector X(x, 2);
        T.Transform(ip, X);
        const double r = x[0];
        // cylindrical components (r, theta, z) from the internal (r, z, theta)
        const int map[3] = {0, 2, 1};
        double c[3] = {0, 0, 0}, dr[3] = {0, 0, 0}, dz[3] = {0, 0, 0};
        for (int i = 0; i < 3; ++i)
          for (int a = 0; a < nd; ++a) {
            const double v = u(map[i] * nd + a);
            c[i] += v * N(a);
            dr[i] += v * dN(a, 0);
            dz[i] += v * dN(a, 1);
          }
        // Cartesian at phi = 0: U = (u_r, u_theta, u_z); d_x U = d_r, d_y U = (1/r) d_phi U = (1/r)(-u_t, u_r, 0)
        const double U[3] = {c[0], c[1], c[2]};
        const double dX[3] = {dr[0], dr[1], dr[2]}, dY[3] = {-c[1] / r, c[0] / r, 0.0}, dZ[3] = {dz[0], dz[1], dz[2]};
        double adv[3];
        for (int i = 0; i < 3; ++i) adv[i] = U[0] * dX[i] + U[1] * dY[i] + U[2] * dZ[i];
        const double divU = dX[0] + dY[1] + dZ[2];
        const double W = ip.weight * T.Weight() * r;
        for (int i = 0; i < 3; ++i)
          for (int a = 0; a < nd; ++a)
            r2(map[i] * nd + a) += W * rho * (adv[i] + (skew ? 0.5 * divU * U[i] : 0.0)) * N(a);
      }
      r1 -= r2;
      dmax = std::max(dmax, r1.Normlinf() / r2.Normlinf());
    }
  }
  CHECK(dmax < 1e-13, "U4 axisymmetric kernel vs the 3D Cartesian evaluation: rel %.3e", dmax);
  if (rank0) std::printf("U4: %.2e\n", dmax);
}

// lid-driven cavity on the unit square (attributes: 1 bottom, 2 right, 3 top, 4 left)
struct Cavity {
  mfem::ConstantCoefficient mu{0.01};
  mfem::VectorFunctionCoefficient lid{2, [](const mfem::Vector& x, mfem::Vector& v) {
                                        v.SetSize(2);
                                        v(0) = 16 * x(0) * x(0) * (1 - x(0)) * (1 - x(0));
                                        v(1) = 0;
                                      }};
  StokesBoundary bc() {
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3, &lid).dirichlet(4);
    return b;
  }
};

static bool throws(const std::function<void()>& f) {
  try {
    f();
  } catch (const axmp::Error&) {
    return true;
  }
  return false;
}

static void u8_refusals() {
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ParMesh pm(MPI_COMM_WORLD, sm);
  Cavity C;
  auto opts = [&]() {
    NavierStokesOptions o;
    o.stokes.nu = &C.mu;
    o.stokes.lu = LUBackend::EigenLU;
    return o;
  };
  {
    auto o = opts();
    o.stokes.grad_div = 1.0;
    CHECK(throws([&] { NavierStokesSolver s(pm, o, C.bc()); }), "grad-div must be refused");
  }
  {
    auto o = opts();
    o.stokes.solver = LinearSolverKind::MinresBlockDiag;
    CHECK(throws([&] { NavierStokesSolver s(pm, o, C.bc()); }), "MINRES must be refused");
  }
  {
    auto o = opts();
    o.stokes.pressure_point.active = true;
    o.stokes.pressure_point.x0 = {0.5, 0.5};
    CHECK(throws([&] { NavierStokesSolver s(pm, o, C.bc()); }), "a pressure point must be refused");
  }
  {
    auto o = opts();
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3, &C.lid).slip(4);
    CHECK(throws([&] { NavierStokesSolver s(pm, o, b); }), "slip must be refused");
  }
  {
    auto o = opts();
    o.stokes.geometry = Geometry::Axisymmetric;
    o.stokes.swirl = false;
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3).axis(4);
    CHECK(throws([&] { NavierStokesSolver s(pm, o, b); }), "swirl-free axisymmetry must be refused");
  }
  {
    auto o = opts();
    o.rho = 0.0;
    CHECK(throws([&] { NavierStokesSolver s(pm, o, C.bc()); }), "rho = 0 must be refused");
  }
  // the transient strategies (time steps, transient_ns.hpp) carry no potential unknowns: refused with the MHD coupling
  // at construction. The message is asserted, not only the throw: the two-component boundary data of this problem
  // would be refused later anyway, by the Planar25D Stokes assembly.
  for (NonlinearStrategy st : {NonlinearStrategy::TransientNewton, NonlinearStrategy::TransientSemiImplicit}) {
    auto o = opts();
    MhdOptions m;
    o.stokes.geometry = Geometry::Planar25D;
    o.stokes.form = ViscousForm::Laplacian;
    o.mhd = &m;
    o.nonlinear.strategy = st;
    std::string what;
    try {
      NavierStokesSolver s(pm, o, C.bc());
    } catch (const axmp::Error& e) {
      what = e.message();
    }
    CHECK(what.find("transient time stepping is not offered") != std::string::npos,
          "a transient strategy with MHD must be refused (got '%s')", what.c_str());
  }
}

// U9: the same problem on all ranks of MPI_COMM_WORLD and on one rank (rank 0, MPI_COMM_SELF)
static void u9() {
  int size = 1, me = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  // A launcher that was asked for several ranks exports the count (cmake/run_mpi_ctest.sh): ranks that started as
  // separate one-rank jobs must fail here, not pass through the one-rank return below.
  if (const char* want = std::getenv("AXMP_EXPECT_RANKS"))
    CHECK(size == std::atoi(want), "U9: MPI_Comm_size is %d but the launcher was asked for %s ranks (AXMP_EXPECT_RANKS)",
          size, want);
  Cavity C;
  C.mu.constant = 1.0 / 300;
  auto run = [&](MPI_Comm comm, std::vector<double>& hist, double& ekin) {
    mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(12, 12, mfem::Element::TRIANGLE, true, 1.0, 1.0);
    mfem::ParMesh pm(comm, sm);
    NavierStokesOptions o;
    o.stokes.nu = &C.mu;
    o.stokes.lu = lu_backend_available(LUBackend::StrumpackMPI) && comm != MPI_COMM_SELF ? LUBackend::StrumpackMPI
                                                                                         : LUBackend::EigenLU;
    o.stokes.nullspace = Nullspace::MeanMultiplier;
    if (o.stokes.lu == LUBackend::EigenLU) o.stokes.nullspace = Nullspace::Auto;
    NavierStokesSolver s(pm, o, C.bc());
    const auto& d = s.solve();
    CHECK(d.converged, "U9 solve did not converge (%s)", d.failure.c_str());
    hist.clear();
    for (const auto& h : d.history) hist.push_back(h.residual);
    mfem::Vector z(2);
    z = 0.0;
    mfem::VectorConstantCoefficient zc(z);
    const double l2 = s.velocity().ComputeL2Error(zc);
    ekin = 0.5 * l2 * l2;
  };
  std::vector<double> hw, hs;
  double ew = 0, es = 0;
  run(MPI_COMM_WORLD, hw, ew);
  if (size == 1) {
    if (rank0) std::printf("U9: one rank (run under mpirun -n 4 for the rank-invariance check); Ekin %.15e\n", ew);
    return;
  }
  if (me == 0) {
    run(MPI_COMM_SELF, hs, es);
    CHECK(std::fabs(ew - es) <= 1e-12 * es, "U9 Ekin %d ranks %.16e vs 1 rank %.16e", size, ew, es);
    CHECK(hw.size() == hs.size(), "U9 history length %zu vs %zu", hw.size(), hs.size());
    for (size_t k = 0; k + 1 < std::min(hw.size(), hs.size()); ++k)  // the last one is at round-off
      CHECK(std::fabs(hw[k] - hs[k]) <= 1e-8 * hs[k] + 1e-14, "U9 history[%zu] %.6e vs %.6e", k, hw[k], hs[k]);
    std::printf("U9: %d ranks vs 1: Ekin rel %.2e, history lengths %zu/%zu\n", size, std::fabs(ew - es) / es,
                hw.size(), hs.size());
  }
}

// U10: AxiViscousIntegrator honours the rule; the Stokes reaction passes the solve's rule (degree 6)
static void u10() {
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  for (int v = 0; v < sm.GetNV(); ++v) sm.GetVertex(v)[0] += 0.05;  // r in [0.05, 1.05]: a strong 1/r variation
  mfem::H1_FECollection fec(2, 2);
  mfem::FiniteElementSpace fes(&sm, &fec, 3);
  mfem::ConstantCoefficient nu(1.0);
  AxiViscousIntegrator i5(nu, 3, false, 0.0, 5), i6(nu, 3, false, 0.0, 6), i0(nu, 3, false, 0.0, 0);
  mfem::DenseMatrix m5, m6, m0;
  i5.AssembleElementMatrix(*fes.GetFE(0), *sm.GetElementTransformation(0), m5);
  i6.AssembleElementMatrix(*fes.GetFE(0), *sm.GetElementTransformation(0), m6);
  i0.AssembleElementMatrix(*fes.GetFE(0), *sm.GetElementTransformation(0), m0);
  mfem::DenseMatrix d(m6);
  d -= m5;
  CHECK(maxabs(d) > 1e-10 * maxabs(m6),
        "U10: degree 5 and 6 rules agree (%.3e): the test integral cannot tell them apart", maxabs(d));
  d = m0;
  d -= m5;
  CHECK(maxabs(d) == 0.0, "U10: the order-0 fallback is no longer the degree-5 rule");
  // the reaction site: the Stokes reaction on a Couette-like axisymmetric problem equals the solved-form residual
  // re-assembled on degree 6 (it differs from degree 5)
  mfem::Mesh am = mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true, 0.5, 0.5);
  for (int v = 0; v < am.GetNV(); ++v) am.GetVertex(v)[0] += 0.5;  // r in [0.5, 1], z in [0, 0.5]
  mfem::ParMesh pm(MPI_COMM_WORLD, am);
  mfem::ConstantCoefficient mu(0.1);
  mfem::VectorFunctionCoefficient wall(3, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(3);
    v = 0.0;
    v(2) = x(0) < 0.5 + 1e-12 ? 2.0 * x(0) : 0.0;
  });
  StokesOptions o;
  o.nu = &mu;
  o.geometry = Geometry::Axisymmetric;
  o.lu = LUBackend::EigenLU;
  o.form = ViscousForm::Laplacian;
  StokesBoundary b;
  b.dirichlet(1).dirichlet(2).dirichlet(3).dirichlet(4, &wall);
  StokesSolver S(pm, o, b);
  S.solve();
  const Reaction rs = S.reaction({4}, {}, ViscousForm::SymmetricStress);  // re-assembled: the fixed site
  o.form = ViscousForm::SymmetricStress;
  StokesSolver S2(pm, o, b);
  S2.solve();
  const Reaction r2 = S2.reaction({4}, {}, ViscousForm::SymmetricStress);  // the solved operator (degree 6)
  CHECK(std::isfinite(rs.torque) && std::isfinite(r2.torque), "U10: reactions not finite");
  if (rank0) std::printf("U10: torque (lap solve, S residual) %.10e, (S solve) %.10e\n", rs.torque, r2.torque);
}

// U11: the stop at the round-off floor of the residual (NonlinearDriver::check_floor,
// NonlinearOptions::residual_floor_factor), on the Re = 300 cavity and on a channel of extremely long cells:
//   (a) newton_rtol 1e-12 is reached, so nothing is stopped at the floor;
//   (c) with residual_floor_factor 0 and a gate below every attainable residual the solve stays unconverged, which
//       is also how the test measures what this problem can attain (the smallest residual of the whole budget);
//   (b) a gate just below that attainable residual: a step stops reducing the residual, the solve is converged with
//       stopped_at_floor, a measured residual_floor, a residual within the factor of it AND within a decade of
//       newton_rtol, and its velocity equals (a)'s, so the accepted state is the same solution;
//   (g) a gate a hundred times below the attainable residual: the iteration still stagnates at the measured floor,
//       so the ceiling on its own refuses the stop;
//   (d) a solve that stops far above the floor (the Picard steps only, no Newton step) is not accepted;
//   (e) nor one that STAGNATES far above the floor (ten Picard steps at Re = 2000, whose residual rises in between,
//       so the stagnation branch runs and measures the floor) when the gate sits just below that stagnation level,
//       so that the magnitude test is the only clause that can refuse it;
//   (f) nor one whose FLOOR stands far above newton_rtol (a 4 x 8 channel on [0, 1e6] x [0, 1], cells 2.5e5 long
//       and 0.125 high): the iteration stagnates at ~1e-10, a hundred times newton_rtol, within the factor of its
//       measured floor - so the ceiling is the only clause that refuses the stop, and the budget runs out.
static void u11_residual_floor() {
  Cavity C;
  C.mu.constant = 1.0 / 300;
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(12, 12, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  auto solve = [&](double rtol, double factor, int maxit, NavierStokesDiagnostics& d, mfem::Vector& u,
                   double mu = 1.0 / 300, int n_picard = 3) {
    C.mu.constant = mu;
    mfem::ParMesh pm(MPI_COMM_SELF, sm);
    NavierStokesOptions o;
    o.stokes.nu = &C.mu;
    o.stokes.lu = LUBackend::EigenLU;
    o.nonlinear.newton_rtol = rtol;
    o.nonlinear.residual_floor_factor = factor;
    o.nonlinear.newton_maxit = maxit;
    o.nonlinear.n_picard = n_picard;
    NavierStokesSolver s(pm, o, C.bc());
    d = s.solve();
    if (d.converged) s.velocity().GetTrueDofs(u);
  };
  NavierStokesDiagnostics da, db, dc, dd, de, dg;
  mfem::Vector ua, ub, uc, ud, ue, ug;
  solve(1e-12, 4.0, 30, da, ua);
  CHECK(da.converged && !da.stopped_at_floor, "U11a: newton_rtol 1e-12 must be reached (converged %d, floor stop %d, "
        "r %.3e)", static_cast<int>(da.converged), static_cast<int>(da.stopped_at_floor), da.relative_residual);
  solve(1e-16, 0.0, 40, dc, uc);
  CHECK(!dc.converged && !dc.stopped_at_floor && !dc.failure.empty(),
        "U11c: residual_floor_factor 0 must leave the solve unconverged (converged %d, '%s')",
        static_cast<int>(dc.converged), dc.failure.c_str());
  double att = 1e300;  // the smallest residual this problem attains, measured, not assumed
  for (const auto& h : dc.history) att = std::min(att, h.residual);
  CHECK(att > 0 && att < 1e-13, "U11c: the attainable residual %.3e must be a round-off quantity", att);
  const double rtol_b = att / 2;  // a gate just below what round-off allows: a near miss, not an impossibility
  solve(rtol_b, 4.0, 30, db, ub);
  CHECK(db.converged && db.stopped_at_floor, "U11b: the solve must stop at the round-off floor (r %.3e, floor %.3e, "
        "'%s')", db.relative_residual, db.residual_floor, db.failure.c_str());
  CHECK(db.residual_floor > 0 && db.relative_residual <= 4.0 * db.residual_floor,
        "U11b: the residual %.3e must be within 4 x the measured floor %.3e", db.relative_residual,
        db.residual_floor);
  // an accepted stop keeps the residual within a decade of the tolerance that was asked for
  CHECK(db.relative_residual <= NonlinearDriver::floor_stop_rtol_ceiling * rtol_b,
        "U11b: the accepted residual %.3e must be within ten times newton_rtol %.3e", db.relative_residual, rtol_b);
  CHECK(db.relative_residual < 1e-11, "U11b: the floor of this problem is far below 1e-11, measured %.3e",
        db.relative_residual);
  // the stop is taken on a step that no longer reduced the residual, never while the iteration still makes progress
  CHECK(db.history.size() >= 2 &&
            db.history.back().residual >= db.history[db.history.size() - 2].residual,
        "U11b: the solve must stop on a step that gained nothing (%.3e after %.3e)", db.history.back().residual,
        db.history.size() >= 2 ? db.history[db.history.size() - 2].residual : -1.0);
  double dmax = 0;
  for (int i = 0; i < ua.Size() && i < ub.Size(); ++i) dmax = std::max(dmax, std::fabs(ua(i) - ub(i)));
  // both states solve the same discrete problem to the round-off floor of its residual, so they differ by the
  // backward error of the direct solve only (measured 1.4e-13 on this problem; the bound keeps a factor seven)
  CHECK(ua.Size() == ub.Size() && dmax <= 1e-12, "U11b: the accepted state differs from the converged one by %.3e",
        dmax);
  // (g) the same stagnation at the same measured floor, but a gate a hundred times below what round-off allows:
  // the magnitude test still passes and the ceiling alone refuses the stop
  solve(att / 100, 4.0, 30, dg, ug);
  CHECK(!dg.converged && !dg.stopped_at_floor && !dg.failure.empty(),
        "U11g: a gate far below the attainable residual must not be met (converged %d, r %.3e, '%s')",
        static_cast<int>(dg.converged), dg.relative_residual, dg.failure.c_str());
  CHECK(dg.residual_floor > 0 && dg.relative_residual <= 4.0 * dg.residual_floor &&
            dg.relative_residual > NonlinearDriver::floor_stop_rtol_ceiling * att / 100,
        "U11g: the ceiling alone must refuse this stop (r %.3e, floor %.3e, newton_rtol %.3e)",
        dg.relative_residual, dg.residual_floor, att / 100);
  solve(1e-12, 4.0, 0, dd, ud);  // the Stokes start and 3 Picard steps, no Newton step
  CHECK(!dd.converged && !dd.stopped_at_floor && dd.relative_residual > 1e-8,
        "U11d: a stop far above the floor must not be accepted (r %.3e, floor %.3e, floor stop %d)",
        dd.relative_residual, dd.residual_floor, static_cast<int>(dd.stopped_at_floor));
  // (e) ten Picard steps at Re = 2000 stagnate far above the floor.  The gate is set just below that stagnation
  // level (measured first with the rule off), so that the ceiling is satisfied and the magnitude test is the only
  // clause that can refuse the stop.
  solve(1e-12, 0.0, 0, de, ue, 1.0 / 2000, 10);
  CHECK(!de.converged && de.relative_residual > 1e-4, "U11e: Re = 2000 must stagnate far above the floor (r %.3e)",
        de.relative_residual);
  NavierStokesDiagnostics dh;
  mfem::Vector uh;
  solve(de.relative_residual / 4, 4.0, 0, dh, uh, 1.0 / 2000, 10);
  CHECK(!dh.converged && !dh.stopped_at_floor && dh.residual_floor > 0 &&
            dh.relative_residual <= NonlinearDriver::floor_stop_rtol_ceiling * de.relative_residual / 4 &&
            dh.relative_residual > 1e6 * dh.residual_floor,
        "U11e: the magnitude test alone must refuse this stagnation (r %.3e, floor %.3e, newton_rtol %.3e, floor "
        "stop %d)", dh.relative_residual, dh.residual_floor, de.relative_residual / 4,
        static_cast<int>(dh.stopped_at_floor));
  // (f) a channel 1e6 long and 1 high in 4 x 8 cells: the floor of this problem stands a hundred times above
  // newton_rtol, so no stop there would keep the residual near the tolerance asked for.  The traction and the body
  // force drive a Poiseuille flow; rho = 1 makes it weakly nonlinear.
  mfem::ConstantCoefficient one(1.0);
  mfem::VectorFunctionCoefficient pull(2, [](const mfem::Vector&, mfem::Vector& v) {
    v.SetSize(2);
    v(0) = 1e6;  // the traction G L at the outlet, with G = 1 the pressure gradient
    v(1) = 0;
  });
  mfem::VectorFunctionCoefficient push(2, [](const mfem::Vector&, mfem::Vector& v) {
    v.SetSize(2);
    v(0) = 1.0;
    v(1) = 0;
  });
  mfem::Mesh chs = mfem::Mesh::MakeCartesian2D(4, 8, mfem::Element::TRIANGLE, true, 1e6, 1.0);
  NavierStokesDiagnostics df;
  {
    mfem::ParMesh pm(MPI_COMM_SELF, chs);
    NavierStokesOptions o;
    o.stokes.nu = &one;
    o.stokes.force = &push;
    o.stokes.lu = LUBackend::EigenLU;
    o.rho = 1.0;
    StokesBoundary b;
    b.dirichlet(1).dirichlet(3).traction(2, &pull).do_nothing(4);
    NavierStokesSolver s(pm, o, b);
    df = s.solve();
  }
  CHECK(!df.converged && !df.stopped_at_floor && !df.failure.empty(),
        "U11f: a floor far above newton_rtol must not be accepted (converged %d, r %.3e, floor %.3e, '%s')",
        static_cast<int>(df.converged), df.relative_residual, df.residual_floor, df.failure.c_str());
  CHECK(df.residual_floor > 1e-11 && df.relative_residual > NonlinearDriver::floor_stop_rtol_ceiling * 1e-12,
        "U11f: this problem must stagnate far above newton_rtol (r %.3e, floor %.3e)", df.relative_residual,
        df.residual_floor);
  // the stagnation and the magnitude clauses are both met here (a floor was measured, and the residual is within
  // the factor of it), so the refusal above is the ceiling's
  CHECK(df.relative_residual <= 4.0 * df.residual_floor,
        "U11f: the residual %.3e must stand within 4 x the measured floor %.3e", df.relative_residual,
        df.residual_floor);
  if (rank0)
    std::printf("U11: attainable %.3e, floor %.3e, accepted residual %.3e (ratio %.2f), state vs converged %.2e, "
                "refused: gate %.3e below the floor (r %.3e), picard-only %.2e, Re = 2000 stagnation %.2e with "
                "floor %.2e, stretched channel %.2e with floor %.2e\n",
                att, db.residual_floor, db.relative_residual, db.relative_residual / db.residual_floor, dmax,
                att / 100, dg.relative_residual, dd.relative_residual, dh.relative_residual, dh.residual_floor,
                df.relative_residual, df.residual_floor);
}

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  rank0 = mfem::Mpi::WorldRank() == 0;
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) {
    u2_u3();
    u4();
    u10();
  }
  if (rank0) u11_residual_floor();  // one rank: MPI_COMM_SELF, the floor is measured per rank
  u8_refusals();
  u9();
  if (rank0) std::printf(failures ? "test_fluids_navier_stokes: %d FAILURES\n" : "test_fluids_navier_stokes: OK\n",
                         failures);
  return failures ? 1 : 0;
}
