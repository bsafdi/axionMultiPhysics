// Accuracy gates of the fluids module that a caller cannot miss (cpp/src/fluids/{convection,navier_stokes,gls,
// stokes}.cpp). The Python side is tests/test_fluids_trust.py.
//
//   - the convective quadrature rule: the default is the exact degree of the trilinear form for the velocity order
//       and the mesh geometry order actually used (triangles: 3p + q - 2 planar and 3p + 2q - 2 axisymmetric on a mesh
//       of geometric order q, so 3p - 1 and 3p on straight ones; quadrilaterals: 3p + q - 1 and 3p + 2q - 1 per
//       direction), the residual assembled on it equals the residual on a much higher rule, the next lower rule does
//       not, and a user rule that is not exact is refused;
//   - the axisymmetric Stokes reaction: the stress reaction re-assembled from the Laplacian solve equals the
//       stress solve's reaction, and the same functional moves when the viscous rule is lowered (so the comparison
//       can tell two rules apart);
//   - the axisymmetric load (linear-form) rule: StokesOptions::load_quad_order() reproduces the closed-form
//       integral int r f_r dr dz of a polynomial force, and the degree-5 and degree-6 rules do not and differ;
//   - the dual-norm mass factorizations of the GLS solver: a pressure mass that is not positive definite is
//       refused, instead of being used through an unchecked Cholesky factorization;
//   - the initial Leray projection: the projected state's discrete continuity residual in the pressure dual norm
//       is at the accuracy the projection solve targets, not merely smaller than the interpolant's, and a projection
//       solve made inaccurate on purpose is refused; the measure that is gated, and so the verdict, depends neither
//       on the aspect ratio of the domain, nor on the length unit, nor on newton_rtol, and the bound is 1e-13 (these
//       on one rank and on several);
//   - the stop at the round-off floor of the residual and the gates on the linear algebra: a floor stop passes the
//       gates, and does not shield a final direct solve above its bound.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fluids/axi_integrators.hpp"
#include "axionmultiphysics/fluids/convection.hpp"
#include "axionmultiphysics/fluids/gls.hpp"
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

// the message of an axmp::Error, or "" when the call did not throw one
static std::string error_message(const std::function<void()>& f) {
  try {
    f();
  } catch (const axmp::Error& e) {
    return e.message().empty() ? std::string("<empty>") : e.message();
  }
  return "";
}

// A 2x2 mesh of triangles or quadrilaterals on [x0, x0 + 1] x [0, 1]: straight (geom_order 1; the quadrilaterals are
// then bilinear and not parallelograms, because the centre vertex is moved) or isoparametric of order geom_order with
// a smooth, non-affine distortion of its nodes. x0 > 0 keeps every quadrature point at r > 0 for the axisymmetric
// kernel's hoop terms.
static mfem::Mesh test_mesh(double x0, int geom_order, bool quads = false) {
  mfem::Mesh m = mfem::Mesh::MakeCartesian2D(2, 2, quads ? mfem::Element::QUADRILATERAL : mfem::Element::TRIANGLE, true,
                                             1.0, 1.0);
  for (int v = 0; v < m.GetNV(); ++v) {
    double* x = m.GetVertex(v);
    if (quads && std::fabs(x[0] - 0.5) < 1e-12 && std::fabs(x[1] - 0.5) < 1e-12) {
      x[0] += 0.11;
      x[1] -= 0.07;
    }
    x[0] += x0;
  }
  if (geom_order > 1) {
    m.SetCurvature(geom_order);
    mfem::GridFunction& n = *m.GetNodes();
    mfem::FiniteElementSpace& nf = *n.FESpace();
    for (int i = 0; i < nf.GetNDofs(); ++i) {
      const double x = n(nf.DofToVDof(i, 0)), y = n(nf.DofToVDof(i, 1));
      const double b = 0.07 * std::sin(3 * x) * std::cos(2 * y);  // smooth: no element is inverted
      n(nf.DofToVDof(i, 0)) = x + b;
      n(nf.DofToVDof(i, 1)) = y - 0.5 * b;
    }
  }
  return m;
}

// ------------------------------ convective rule
// The exact quadrature degree of c(w; u, v) = rho ((w.grad)u, v) (and of the skew form's (div w)(u, v) term), for a
// velocity of order p on an isoparametric element whose geometry has order q. One factor J^-1 from the physical
// gradient cancels against det J from the measure, so the integrand is a polynomial on the reference element.
// Triangles (the TOTAL degree counts):
//   planar         w (p) . adj(J)^T grad_ref u ((q-1) + (p-1)) . v (p)                      = 3p + q - 2
//   axisymmetric   the same times the weight r (q), the hoop terms w_t u_t (2p) times det J (2(q-1)) and v (p)
//                                                                                           = 3p + 2q - 2
// Quadrilaterals (the degree PER DIRECTION counts, and a tensor Gauss rule is exact to it in each direction): d_xi u
// has degree p - 1 in xi but p in eta, and the entry of adj(J) that multiplies it, d_eta x, has degree q in xi, so
//   planar         w (p) . adj(J)^T grad_ref u (q + (p-1)) . v (p)                          = 3p + q - 1
//   axisymmetric   the same times the weight r (q); the hoop terms w_t u_t v (3p) times det J (2q - 1) are equal
//                                                                                           = 3p + 2q - 1
// A straight (bilinear) quadrilateral has q = 1: 3p and 3p + 1, one more than a straight triangle, and the same holds
// on a rectangle (the count above does not use the distortion).
static int exact_convection_degree(bool axi, int p, int q, bool quads = false) {
  return (axi ? 3 * p + 2 * q - 2 : 3 * p + q - 2) + (quads ? 1 : 0);
}

// the largest degree whose rule is less exact than `degree`: degree - 1 on a triangle; on a quadrilateral the n-point
// Gauss rule is exact to 2n - 1, so the rule of an even degree is that of the next odd one and the step can be two
static int next_lower_rule(bool quads, int degree) {
  const mfem::Geometry::Type g = quads ? mfem::Geometry::SQUARE : mfem::Geometry::TRIANGLE;
  int lower = degree - 1;
  while (lower > 0 && mfem::IntRules.Get(g, lower).GetOrder() >= degree) --lower;
  return lower;
}

// the convective residual of a fixed pseudo-random state over the whole mesh, assembled on an explicit rule
static std::vector<double> convective_residual(mfem::Mesh& mesh, bool axi, int p, int rule, bool allow_under = true) {
  const int vd = axi ? 3 : 2;
  mfem::H1_FECollection fec(p, 2);
  mfem::FiniteElementSpace fes(&mesh, &fec, vd);
  ConvectionIntegrator k(1.3, vd, axi, ConvectiveForm::Standard, rule, allow_under);
  std::mt19937 gen(12345);
  std::normal_distribution<double> nd01;
  std::vector<double> out;
  for (int e = 0; e < mesh.GetNE(); ++e) {
    const mfem::FiniteElement& fe = *fes.GetFE(e);
    mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
    mfem::Vector u(vd * fe.GetDof()), r;
    for (int i = 0; i < u.Size(); ++i) u(i) = nd01(gen);
    k.AssembleElementVector(fe, T, u, r);
    for (int i = 0; i < r.Size(); ++i) out.push_back(r(i));
  }
  return out;
}

static double rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0, s = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    d = std::max(d, std::fabs(a[i] - b[i]));
    s = std::max(s, std::fabs(b[i]));
  }
  return s > 0 ? d / s : d;
}

static void q1_convection_rule() {
  // (a) the integrand IS a polynomial of the stated degree: the rule of that degree agrees with a rule six degrees
  // higher to round-off, and the next lower rule does not. Measured over these cases, the relative difference at the
  // exact degree stays below 1.2e-14 (triangles) and 1.5e-15 (quadrilaterals), and the next lower rule never falls
  // below 5.7e-8 and 7.1e-8, so 1e-13 and 1e-9 separate the two by nearly a decade above the round-off and more than
  // one below the smallest quadrature error.
  for (int quads = 0; quads < 2; ++quads) {
    double worst_at = 0, best_below = 1.0;
    const char* shape = quads ? "quadrilaterals" : "triangles";
    for (int p : {2, 3})
      for (int q : {1, 2, 3})
        for (int axi = 0; axi < 2; ++axi) {
          mfem::Mesh mesh = test_mesh(axi ? 0.6 : 0.0, q, quads != 0);
          const int ex = exact_convection_degree(axi != 0, p, q, quads != 0);
          const int lower = next_lower_rule(quads != 0, ex);
          const std::vector<double> ref = convective_residual(mesh, axi != 0, p, ex + 6);
          const double at = rel_diff(convective_residual(mesh, axi != 0, p, ex), ref);
          const double below = rel_diff(convective_residual(mesh, axi != 0, p, lower), ref);
          worst_at = std::max(worst_at, at);
          best_below = std::min(best_below, below);
          CHECK(at < 1e-13, "convective rule %s %s p=%d q=%d: the exact degree %d does not integrate the form exactly (rel %.3e)",
                shape, axi ? "axisymmetric" : "planar", p, q, ex, at);
          CHECK(below > 1e-9,
                "convective rule %s %s p=%d q=%d: the rule of degree %d is already exact, so degree %d is not the minimum (rel %.3e)",
                shape, axi ? "axisymmetric" : "planar", p, q, lower, ex, below);
        }
    if (rank0)
      std::printf("convective rule, %s: exact degree worst %.3e, next lower rule best %.3e\n", shape, worst_at, best_below);
  }

  // (b) the solver's default rule (quad_order_convection = 0) IS that exact degree, for the velocity order and the
  // mesh geometry order actually used; diagnostics().quad_orders[1] reports it
  mfem::ConstantCoefficient mu(0.1);
  auto rule_used = [&](bool quads, bool axi, int p, int q, int requested) {
    mfem::Mesh sm = test_mesh(axi ? 0.6 : 0.0, q, quads);
    mfem::ParMesh pm(MPI_COMM_WORLD, sm);
    NavierStokesOptions o;
    o.stokes.nu = &mu;
    o.stokes.lu = LUBackend::EigenLU;
    o.stokes.velocity_order = p;
    o.stokes.geometry = axi ? Geometry::Axisymmetric : Geometry::Planar;
    o.stokes.swirl = axi;
    o.quad_order_convection = requested;
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3).dirichlet(4);
    NavierStokesSolver s(pm, o, b);
    return s.diagnostics().quad_orders.at(1);
  };
  for (int quads = 0; quads < 2; ++quads)
    for (int p : {2, 3})
      for (int q : {1, 2})
        for (int axi = 0; axi < 2; ++axi) {
          const int ex = exact_convection_degree(axi != 0, p, q, quads != 0);
          const int got = rule_used(quads != 0, axi != 0, p, q, 0);
          CHECK(got == ex, "convective rule %s %s p=%d q=%d: the default convective rule is degree %d, the exact degree is %d",
                quads ? "quadrilaterals" : "triangles", axi ? "axisymmetric" : "planar", p, q, got, ex);
        }

  // (c) a user rule that is NOT exact is refused, naming the option; an exact one is accepted. An under-integrated
  // convective term changes the discrete operator while the Newton gate still measures the under-integrated residual
  // and reports convergence, which is a wrong answer reported as converged. What counts is the rule, not the number
  // asked for: on straight axisymmetric quadrilaterals the exact degree is 7 and degree 6 selects the same 4x4 Gauss
  // rule, so 6 is accepted and 5 (3x3) is refused.
  for (int quads = 0; quads < 2; ++quads)
    for (int q : {1, 2})
      for (int axi = 0; axi < 2; ++axi) {
        const char* shape = quads ? "quadrilaterals" : "triangles";
        const int ex = exact_convection_degree(axi != 0, 2, q, quads != 0);
        const int lower = next_lower_rule(quads != 0, ex);
        const std::string why = error_message([&] { rule_used(quads != 0, axi != 0, 2, q, lower); });
        CHECK(!why.empty(), "convective rule %s %s q=%d: convective rule degree %d (exact %d) was accepted silently", shape,
              axi ? "axisymmetric" : "planar", q, lower, ex);
        CHECK(why.find("quad_order_convection") != std::string::npos,
              "convective rule %s %s q=%d: the refusal does not name quad_order_convection (got '%s')", shape,
              axi ? "axisymmetric" : "planar", q, why.c_str());
        for (int ok = lower + 1; ok <= ex; ++ok)  // every degree whose rule is the exact one
          CHECK(error_message([&] { rule_used(quads != 0, axi != 0, 2, q, ok); }).empty(),
                "convective rule %s %s q=%d: degree %d selects an exact rule (exact degree %d) and must be accepted", shape,
                axi ? "axisymmetric" : "planar", q, ok, ex);
        CHECK(error_message([&] { rule_used(quads != 0, axi != 0, 2, q, ex + 2); }).empty(),
              "convective rule %s %s q=%d: a rule above the exact degree must be accepted (only costlier)", shape,
              axi ? "axisymmetric" : "planar", q);
        CHECK(rule_used(quads != 0, axi != 0, 2, q, ex + 2) == ex + 2,
              "convective rule %s %s q=%d: the accepted rule is not the reported one", shape, axi ? "axisymmetric" : "planar", q);
      }
  CHECK(next_lower_rule(true, 7) == 5 && next_lower_rule(true, 6) == 5 && next_lower_rule(false, 5) == 4,
        "convective rule: the rule one step below the exact one is not the one this test assumes");
  // (d) the integrator itself refuses such a rule when it assembles an element (the second check, for an integrator
  // built without the solver), unless the caller opted in, which is what the measurement in (a) does
  for (int quads = 0; quads < 2; ++quads)
    for (int axi = 0; axi < 2; ++axi) {
      mfem::Mesh mesh = test_mesh(axi ? 0.6 : 0.0, 2, quads != 0);
      const int ex = exact_convection_degree(axi != 0, 2, 2, quads != 0);
      const int lower = next_lower_rule(quads != 0, ex);
      const std::string why = error_message([&] { convective_residual(mesh, axi != 0, 2, lower, false); });
      CHECK(why.find("exact degree") != std::string::npos,
            "convective rule %s %s: the integrator assembled on degree %d (exact %d) without the opt-in (got '%s')",
            quads ? "quadrilaterals" : "triangles", axi ? "axisymmetric" : "planar", lower, ex, why.c_str());
      CHECK(error_message([&] { convective_residual(mesh, axi != 0, 2, ex, false); }).empty(),
            "convective rule %s %s: the integrator refused its exact degree %d", quads ? "quadrilaterals" : "triangles",
            axi ? "axisymmetric" : "planar", ex);
    }
  if (rank0) std::printf("convective rule: convective rule (exact degree, default, refusal) checked\n");
}

// ------------------------------ axisymmetric reaction
// The axisymmetric swirl problem of the reaction check: u_theta = 2 r on the outer wall, the other walls at rest.
struct AxiSwirl {
  mfem::ConstantCoefficient mu{0.1};
  mfem::VectorFunctionCoefficient wall{3, [](const mfem::Vector& x, mfem::Vector& v) {
                                         v.SetSize(3);
                                         v = 0.0;
                                         v(2) = x(0) < 0.5 + 1e-12 ? 2.0 * x(0) : 0.0;
                                       }};
  StokesBoundary bc() {
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3).dirichlet(4, &wall);
    return b;
  }
};

static void q2_reaction() {
  mfem::Mesh am = mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true, 0.5, 0.5);
  for (int v = 0; v < am.GetNV(); ++v) am.GetVertex(v)[0] += 0.5;  // r in [0.5, 1], z in [0, 0.5]
  mfem::ParMesh pm(MPI_COMM_WORLD, am);
  AxiSwirl P;
  auto torque = [&](ViscousForm solved, int quad_viscous) {
    StokesOptions o;
    o.nu = &P.mu;
    o.geometry = Geometry::Axisymmetric;
    o.lu = LUBackend::EigenLU;
    o.form = solved;
    o.quad_order_viscous = quad_viscous;
    StokesSolver S(pm, o, P.bc());
    S.solve();
    return S.reaction({4}, {}, ViscousForm::SymmetricStress).torque;
  };
  // The stress reaction re-assembled from the Laplacian solve and the stress solve's own reaction are the same
  // discrete functional: for a constant viscosity and all-Dirichlet data the two viscous operators have the same
  // solution, and the reaction is re-assembled in the same form on the same rule. What is left is two independent
  // LU solves, each refined to the Stokes acceptance target. Measured agreement 1.3e-16; the smallest change the
  // same functional shows when its rule is wrong is 7.6e-10 (one degree above the solve's rule) and 3.1e-7 (one
  // degree below), so 1e-11 is four decades above the round-off and two below the smallest rule error.
  const double lap = torque(ViscousForm::Laplacian, 0), str = torque(ViscousForm::SymmetricStress, 0);
  const double d = std::fabs(lap - str) / std::fabs(str);
  CHECK(d < 1e-11, "axisymmetric reaction reaction: re-assembled %.17e vs solved %.17e (rel %.3e)", lap, str, d);
  // the same comparison on a lower viscous rule must move: a reaction assembled on the wrong rule is detectable
  const double low = torque(ViscousForm::Laplacian, 5);
  const double dl = std::fabs(low - lap) / std::fabs(lap);
  CHECK(dl > 1e-8, "axisymmetric reaction reaction: the degree-5 viscous rule gives the same torque (rel %.3e): the comparison above "
                   "cannot tell two rules apart", dl);
  if (rank0) std::printf("axisymmetric reaction: reaction rel %.3e, rule sensitivity %.3e\n", d, dl);
}

// ------------------------------ axisymmetric load rule
// The axisymmetric load (linear form) int r f . v dr dz on the solver's rule. Summing the r-component rows uses the
// P2 partition of unity sum_a N_a = 1, so the sum is int r f_r dr dz, which is elementary for a monomial force.
static void q3_load_rule() {
  mfem::Mesh m(2, 3, 1);  // one affine triangle (1,0), (2,0), (1,1): the map (xi, eta) -> (1 + xi, eta), |det J| = 1
  m.AddVertex(1.0, 0.0);
  m.AddVertex(2.0, 0.0);
  m.AddVertex(1.0, 1.0);
  int vi[3] = {0, 1, 2};
  m.AddTriangle(vi, 1);
  m.FinalizeTriMesh(1, 0, true);
  mfem::H1_FECollection fec(2, 2);
  mfem::FiniteElementSpace fes(&m, &fec, 3, mfem::Ordering::byNODES);
  // f = (r^4 z^4, 0, 0): the integrand of the sum is r^5 z^4, total degree 9
  mfem::VectorFunctionCoefficient f(3, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(3);
    v = 0.0;
    v(0) = std::pow(x(0), 4) * std::pow(x(1), 4);
  });
  RadiusWeight r;
  mfem::ScalarVectorProductCoefficient rf(r, f);
  // int_0^1 eta^4 int_0^{1-eta} (1 + xi)^5 dxi deta = (1/6) int_0^1 eta^4 ((2 - eta)^6 - 1) deta
  //   = (1/6) [ sum_k C(6,k) 2^(6-k) (-1)^k / (5 + k) - 1/5 ]
  long double exact = 0;
  const long double binom[7] = {1, 6, 15, 20, 15, 6, 1};
  for (int k = 0; k <= 6; ++k)
    exact += binom[k] * std::pow(2.0L, 6 - k) * ((k % 2) ? -1.0L : 1.0L) / (5.0L + k);
  exact = (exact - 1.0L / 5.0L) / 6.0L;
  auto load_sum = [&](int rule) {
    mfem::LinearForm L(&fes);
    auto* in = new mfem::VectorDomainLFIntegrator(rf);
    in->SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TRIANGLE, rule));
    L.AddDomainIntegrator(in);
    L.Assemble();
    double s = 0;
    for (int a = 0; a < fes.GetFE(0)->GetDof(); ++a) s += L.Elem(a);  // byNODES: the r component comes first
    return s;
  };
  StokesOptions o;
  o.velocity_order = 2;
  const double ref = static_cast<double>(exact);
  const double d0 = std::fabs(load_sum(o.load_quad_order()) - ref) / ref;
  const double d5 = std::fabs(load_sum(5) - ref) / ref;
  const double d6 = std::fabs(load_sum(6) - ref) / ref;
  const double d56 = std::fabs(load_sum(5) - load_sum(6)) / ref;
  // the default rule (2p + 8 = degree 12 for P2) is exact for a degree-9 integrand: only round-off is left
  CHECK(d0 < 1e-13, "axisymmetric load rule load rule: the default rule (degree %d) misses int r^5 z^4 = %.17g by rel %.3e",
        o.load_quad_order(), ref, d0);
  CHECK(d5 > 1e-4 && d6 > 1e-5, "axisymmetric load rule load rule: degree 5/6 already integrate the test load (rel %.3e, %.3e): the "
                                "integral cannot tell the rules apart", d5, d6);
  CHECK(d56 > 1e-6, "axisymmetric load rule load rule: degrees 5 and 6 agree (rel %.3e)", d56);
  if (rank0) std::printf("axisymmetric load rule: load rule degree %d exact to %.3e; degree 5 %.3e, degree 6 %.3e\n",
                         o.load_quad_order(), d0, d5, d6);

  // The solver's own load path uses that rule: an axisymmetric Stokes solve driven only by the body force
  // f = (r^4 z^4, 0, 0) between walls at rest. With P2 test functions the load integrand r f . v has degree 11, so the
  // default rule (degree 12) and a rule two degrees higher assemble the same load and the two solves agree to the
  // accuracy of two refined direct solves, while a degree-5 load rule assembles a different load and the velocity
  // moves. Measured: 3.7e-16 between the default and degree 14, 4.3e-5 for degree 5; 1e-11 and 1e-6 sit four decades
  // above the first and more than one below the second.
  mfem::Mesh am = mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  for (int v = 0; v < am.GetNV(); ++v) am.GetVertex(v)[0] += 1.0;  // r in [1, 2], z in [0, 1]
  mfem::ParMesh pm(MPI_COMM_WORLD, am);
  mfem::ConstantCoefficient mu(0.1);
  auto speed = [&](int quad_load) {
    StokesOptions so;
    so.nu = &mu;
    so.geometry = Geometry::Axisymmetric;
    so.lu = LUBackend::EigenLU;
    so.force = &f;
    so.quad_order_load = quad_load;
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3).dirichlet(4);
    StokesSolver S(pm, so, b);
    S.solve();
    mfem::Vector z(3);
    z = 0.0;
    mfem::VectorConstantCoefficient zc(z);
    return S.velocity().ComputeL2Error(zc);
  };
  const double u_def = speed(0), u_hi = speed(o.load_quad_order() + 2), u_lo = speed(5);
  const double e_hi = std::fabs(u_def - u_hi) / u_hi, e_lo = std::fabs(u_lo - u_hi) / u_hi;
  CHECK(u_hi > 0 && e_hi < 1e-11,
        "axisymmetric load rule: the solve on the default load rule differs from the one two degrees higher (rel %.3e, |u| %.3e)",
        e_hi, u_hi);
  CHECK(e_lo > 1e-6, "axisymmetric load rule: a degree-5 load rule does not move the solution (rel %.3e): the solver's "
                     "load path ignores StokesOptions::quad_order_load", e_lo);
  if (rank0) std::printf("axisymmetric load rule: solver path default vs degree %d rel %.3e; degree 5 rel %.3e\n",
                         o.load_quad_order() + 2, e_hi, e_lo);
}

// ------------------------------ GLS mass factorization
// GLS: the dual norm is sqrt(R^T M^-1 R) with two sparse Cholesky factorizations of consistent P1 mass matrices. A
// vertex that no triangle uses gives an all-zero row and column in the pressure mass, so the factorization fails and
// every dual norm taken with it is meaningless; the solver must refuse the data instead of using it.
static void q4_gls_mass() {
  GlsData D;  // unit square as two triangles, plus a fifth vertex no triangle refers to
  D.V = {0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5};
  D.T = {0, 1, 2, 0, 2, 3};
  D.mu_nodal.assign(5, 1.0);
  D.dirichlet_nodes = {0, 1, 2, 3};
  D.dirichlet_values = {0, 0, 1, 0, 1, 0, 0, 0};
  GlsOptions o;
  o.lu = LUBackend::EigenLU;
  std::string why = error_message([&] {
    GlsProblem P(D, o);
    P.solve();
  });
  CHECK(!why.empty(), "GLS mass factorization: a GLS pressure mass that is not positive definite was accepted");
  CHECK(why.find("mass") != std::string::npos,
        "GLS mass factorization: the refusal does not name the mass factorization (got '%s')", why.c_str());
  // control: the same data with every vertex used must still be accepted and converge
  GlsData C = D;
  C.V.resize(8);
  C.mu_nodal.assign(4, 1.0);
  GlsProblem Pc(C, o);
  const GlsDiagnostics& dc = Pc.solve();
  CHECK(dc.converged, "GLS mass factorization control: the well-posed GLS problem no longer converges (%s)", dc.failure.c_str());
  // every dual-norm solve of that verdict had its residual formed, and it is far inside the gate (measured 1e-16)
  CHECK(dc.mass_lu_residual_max >= 0 && dc.mass_lu_residual_max <= 1e-13,
        "GLS mass solves: the worst dual-norm solve residual %.3e is not recorded or not at round-off", dc.mass_lu_residual_max);
  CHECK(dc.residual_lift > 1e-3, "GLS mass factorization control: the lift residual %.3e is not a sane dual norm", dc.residual_lift);
  if (rank0) std::printf("GLS mass factorization: GLS mass refusal '%s'\n", why.empty() ? "<none>" : why.substr(0, 60).c_str());
}

// ------------------------------ final direct solve
// The final step's direct solve must reach the accuracy the solver is configured to require, or a converged verdict is
// refused and the reason names the direct solve. The accuracy required is max(newton_rtol, 1e-11) by default
// (sparse_lu.hpp says why), so a healthy steady solve converges, and NonlinearOptions::final_solve_residual_max below
// what any solve delivers makes the same solve fail; without the check the two runs would be identical.
static void final_solve_gate() {
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(8, 8, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ParMesh pm(MPI_COMM_WORLD, sm);
  mfem::ConstantCoefficient mu(0.05);
  mfem::VectorFunctionCoefficient lid(2, [](const mfem::Vector& x, mfem::Vector& v) {  // vanishes at the corners
    v.SetSize(2);
    v(0) = 16.0 * x(0) * x(0) * (1 - x(0)) * (1 - x(0));
    v(1) = 0.0;
  });
  auto run = [&](double required) {
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3, &lid).dirichlet(4);
    NavierStokesOptions o;
    o.stokes.nu = &mu;
    o.stokes.lu = LUBackend::EigenLU;
    o.nonlinear.final_solve_residual_max = required;
    NavierStokesSolver S(pm, o, b);
    return S.solve();
  };
  const NavierStokesDiagnostics d0 = run(0.0);
  CHECK(d0.converged, "final direct solve: the cavity does not converge (%s)", d0.failure.c_str());
  CHECK(d0.history.back().lu_residual >= 0 && d0.history.back().lu_residual <= 1e-11,
        "final direct solve: the final step's refined residual %.3e is not recorded or above 1e-11",
        d0.history.back().lu_residual);
  const NavierStokesDiagnostics d1 = run(1e-20);
  CHECK(!d1.converged, "final direct solve: a final solve above the required %.1e (it reached %.3e) was accepted", 1e-20,
        d1.history.back().lu_residual);
  CHECK(d1.failure.find("final direct solve") != std::string::npos,
        "final direct solve: the refusal does not name the direct solve (got '%s')", d1.failure.c_str());
  if (rank0) std::printf("final direct solve: final residual %.3e, refused at 1e-20: '%s'\n", d0.history.back().lu_residual,
                         d1.failure.substr(0, 60).c_str());
}

// ------------------------------ the stop at the round-off floor and the gates on the linear algebra
// A steady solve that stops at the measured round-off floor of its residual (NonlinearOptions::residual_floor_factor)
// is accepted with a nonlinear residual above newton_rtol. The gates on the linear algebra are applied after that
// verdict and are independent of it: they bound the final direct solve by max(newton_rtol, 1e-11) and the dual-norm
// mass solves by 1e-10, neither of which a stagnated iteration approaches (its last linear solve is as accurate as
// any other). So a floor stop is not refused by them on a healthy problem, and a floor stop does not shield a solve
// whose final direct solve misses its bound.
static void floor_stop_and_gates() {
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(12, 12, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ConstantCoefficient mu(1.0 / 300);
  mfem::VectorFunctionCoefficient lid(2, [](const mfem::Vector& x, mfem::Vector& v) {  // vanishes at the corners
    v.SetSize(2);
    v(0) = 16.0 * x(0) * x(0) * (1 - x(0)) * (1 - x(0));
    v(1) = 0.0;
  });
  auto run = [&](double rtol, double factor, double final_max) {
    mfem::ParMesh pm(MPI_COMM_SELF, sm);
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3, &lid).dirichlet(4);
    NavierStokesOptions o;
    o.stokes.nu = &mu;
    o.stokes.lu = LUBackend::EigenLU;
    o.nonlinear.newton_rtol = rtol;
    o.nonlinear.residual_floor_factor = factor;
    o.nonlinear.newton_maxit = 40;
    o.nonlinear.final_solve_residual_max = final_max;
    NavierStokesSolver S(pm, o, b);
    return S.solve();
  };
  // the residual this problem attains, measured: a gate below every attainable residual, without the floor stop
  const NavierStokesDiagnostics d0 = run(1e-16, 0.0, 0.0);
  double att = 1e300;
  for (const auto& h : d0.history) att = std::min(att, h.residual);
  CHECK(!d0.converged && att > 0 && att < 1e-13, "floor stop and gates: the attainable residual %.3e is not a round-off quantity", att);
  const double rtol = att / 2;  // a near miss: the floor stop accepts it
  const NavierStokesDiagnostics d1 = run(rtol, 4.0, 0.0);
  CHECK(d1.converged && d1.stopped_at_floor && d1.failure.empty(),
        "floor stop and gates: a solve stopped at the measured floor is refused (converged %d, floor stop %d, r %.3e, floor %.3e, '%s')",
        static_cast<int>(d1.converged), static_cast<int>(d1.stopped_at_floor), d1.relative_residual, d1.residual_floor,
        d1.failure.c_str());
  CHECK(d1.relative_residual > rtol, "floor stop and gates: the stop is not above newton_rtol (r %.3e, newton_rtol %.3e), so this "
                                     "test does not exercise an accepted floor stop", d1.relative_residual, rtol);
  // the gates saw that solve: its final direct solve and its mass solves are recorded and inside their bounds, with
  // room (measured 9.8e-16 and 2.0e-16 at a stop on a relative residual of 1.4e-15; the gates' bounds are 1e-11 and
  // 1e-10, and 1e-13 here is two decades above what is measured)
  const double last = d1.history.empty() ? -1.0 : d1.history.back().lu_residual;
  CHECK(last >= 0 && last <= 1e-13, "floor stop and gates: the final direct solve of the floor stop has residual %.3e", last);
  CHECK(d1.mass_lu_residual_max >= 0 && d1.mass_lu_residual_max <= 1e-13,
        "floor stop and gates: the dual-norm mass solves of the floor stop have residual %.3e", d1.mass_lu_residual_max);
  // the same solve with a bound on the final direct solve that no solve meets: the iteration still stops at the
  // floor, and the solve is refused for the direct solve
  const NavierStokesDiagnostics d2 = run(rtol, 4.0, 1e-20);
  CHECK(!d2.converged && d2.stopped_at_floor, "floor stop and gates: a floor stop shields a final direct solve above its bound "
                                              "(converged %d, floor stop %d)", static_cast<int>(d2.converged),
        static_cast<int>(d2.stopped_at_floor));
  CHECK(d2.failure.find("final direct solve") != std::string::npos,
        "floor stop and gates: the refusal does not name the direct solve (got '%s')", d2.failure.c_str());
  if (rank0)
    std::printf("floor stop and gates: attainable %.3e, stopped at %.3e (floor %.3e), final LU %.3e, mass %.3e; refused at 1e-20: '%s'\n",
                att, d1.relative_residual, d1.residual_floor, last, d1.mass_lu_residual_max, d2.failure.substr(0, 60).c_str());
}

// ------------------------------ initial projection
// The initial Leray projection solves [M B^T; B 0] (u, lambda) = (M u_I, rhs_p). The property it exists for is that
// the projected state is DISCRETELY divergence free: the continuity rows B u - rhs_p vanish. Their dual norm
// (M_p^-1, the norm every fluids gate is expressed in), relative to the interpolant's, is the projection's own
// accuracy and must be at the accuracy the solve targets, not merely smaller than the interpolant's.
static void q5_projection() {
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
  mfem::VectorFunctionCoefficient u0(2, [](const mfem::Vector& x, mfem::Vector& v) {  // not divergence free, zero on dK
    v.SetSize(2);
    const double s = x(0) * (1 - x(0)) * x(1) * (1 - x(1));
    v(0) = s * (1 + x(0));
    v(1) = s * x(1);
  });
  mfem::ParGridFunction g(&S.velocity_space());
  g.ProjectCoefficient(u0);
  mfem::Vector uI(S.velocity_size()), p0(S.pressure_size());
  g.GetTrueDofs(uI);
  p0 = 0.0;
  const long nu = S.layout().first;
  auto continuity = [&] {  // the pressure rows of the residual at the current state, in the pressure dual norm
    std::vector<double> R = S.residual(S.state(), 1.0);
    for (long i = 0; i < nu; ++i) R[i] = 0.0;
    return S.dual_norm(R);
  };
  S.set_state(uI, p0, true);
  const double d_interp = continuity();
  const bool ok = S.project_divergence_free(uI);
  const double d_proj = continuity();
  CHECK(ok, "initial projection: project_divergence_free failed");
  CHECK(d_interp > 1e-3, "initial projection: the interpolant's continuity residual %.3e is too small to measure against", d_interp);
  // The projection solve is refined to min(1e-12, 1e-2 newton_rtol) = 1e-14 at the default newton_rtol, and this
  // ratio measures 5.0e-16 here. 1e-13 is more than two decades above that and eleven below the interpolant's; a projection
  // accepted at a fixed 1e-6 would land near 1e-6 of the interpolant's residual and is refused.
  CHECK(d_proj < 1e-13 * d_interp, "initial projection: the projected state is not discretely divergence free: continuity dual norm "
                                   "%.3e, interpolant %.3e (rel %.3e)", d_proj, d_interp, d_proj / d_interp);
  if (rank0) std::printf("initial projection: continuity dual norm interpolant %.3e, projected %.3e (rel %.3e)\n", d_interp, d_proj,
                         d_proj / d_interp);

  // The gate itself. A projection solve made inaccurate by a known amount (the solver's test hook scales the solve's
  // solution by 1 + eps, which leaves a backward error of 0.68 eps / N on an N x N mesh) is refused, the state is left
  // untouched, and the same solver accepts the projection again once the perturbation is removed. eps = 1e-8 leaves
  // 8.1e-10 here: almost four decades above the bound and still below 1e-6, so a gate that is absent, or a fixed
  // acceptance at 1e-6, accepts it.
  S.set_state(uI, p0, true);
  const mfem::Vector u_before(S.state_u());
  S.set_projection_solve_perturbation(1e-8);
  const bool bad = S.project_divergence_free(uI);
  const double res_bad = S.diagnostics().projection_residual;
  mfem::Vector moved(S.state_u());
  moved -= u_before;
  CHECK(!bad && !S.diagnostics().projection_converged, "initial projection: a solve perturbed by 1e-8 (residual %.3e) was accepted",
        res_bad);
  CHECK(res_bad > 1e-11 && res_bad < 1e-6, "initial projection: the perturbed solve's residual %.3e is not of the order of 1e-8", res_bad);
  CHECK(moved.Normlinf() == 0.0, "initial projection: a refused projection changed the state (by %.3e)", moved.Normlinf());
  S.set_projection_solve_perturbation(0.0);
  CHECK(S.project_divergence_free(uI) && S.diagnostics().projection_converged && S.diagnostics().projection_residual < 1e-13,
        "initial projection: the unperturbed projection is refused after a perturbed one (residual %.3e)",
        S.diagnostics().projection_residual);
  if (rank0) std::printf("initial projection: perturbed by 1e-8: residual %.3e, refused\n", res_bad);
}

// ------------------------------ initial projection: the shape of the domain, the length unit, the bound
// The measure that is gated (sparse_lu.hpp, kProjectionResidualMax) compares each block of the residual of the
// projection's saddle system with the sizes of the terms of that block. Three properties are checked here, each of
// which a residual relative to the data, taken over both blocks at once, does not have:
//   - it does not depend on the aspect ratio of the domain: in a channel of aspect ratio a the multiplier is of size a
//     and its transverse gradient of size one, so the momentum residual relative to the data is a n times the
//     round-off (n cells across), 1e-11 at a = 1e4, and a gate on it at 1e-13 refuses an accurate projection;
//   - it does not depend on the length unit: the momentum rows carry L^2 and the continuity rows L;
//   - it does not depend on newton_rtol, and neither does the bound.
// The field is the one of q5_projection mapped to [0, lx] x [0, ly]. These run on any number of ranks.
struct ProjectionRun {
  bool ok = false;
  double measure = -1;
};
static ProjectionRun project_on(double lx, double ly, int n, double newton_rtol, double perturbation, double perturbation_p) {
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::TRIANGLE, true, lx, ly);
  mfem::ParMesh pm(MPI_COMM_WORLD, sm);
  mfem::ConstantCoefficient mu(0.05);
  StokesBoundary b;
  b.dirichlet(1).dirichlet(2).dirichlet(3).dirichlet(4);
  NavierStokesOptions o;
  o.stokes.nu = &mu;
  o.stokes.lu = LUBackend::EigenLU;
  o.rho = 1.25;
  o.nonlinear.strategy = NonlinearStrategy::TransientNewton;
  o.nonlinear.newton_rtol = newton_rtol;
  NavierStokesSolver S(pm, o, b);
  mfem::VectorFunctionCoefficient u0(2, [lx, ly](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(2);
    const double X = x(0) / lx, Y = x(1) / ly, s = X * (1 - X) * Y * (1 - Y);
    v(0) = s * (1 + X);
    v(1) = s * Y;
  });
  mfem::ParGridFunction g(&S.velocity_space());
  g.ProjectCoefficient(u0);
  mfem::Vector uI(S.velocity_size());
  g.GetTrueDofs(uI);
  S.set_projection_solve_perturbation(perturbation, perturbation_p);
  ProjectionRun r;
  r.ok = S.project_divergence_free(uI);
  r.measure = S.diagnostics().projection_residual;
  CHECK(r.ok == S.diagnostics().projection_converged, "projection measure: the return value and projection_converged differ");
  // The measure is a sum over the ranks and the verdict is collective: every rank must hold the same number and the
  // same verdict, bit for bit. A measure formed from each rank's own rows differs between the ranks.
  double mine[4] = {r.measure, -r.measure, r.ok ? 1.0 : 0.0, r.ok ? -1.0 : 0.0}, most[4];
  MPI_Allreduce(mine, most, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  CHECK(most[0] == -most[1] && most[2] == -most[3],
        "projection measure: the ranks disagree on the measure (%.17g to %.17g) or on the verdict (%g to %g)", -most[1], most[0],
        -most[3], most[2]);
  return r;
}

static ProjectionRun project_on(double lx, double ly, int n, double newton_rtol, double perturbation) {
  return project_on(lx, ly, n, newton_rtol, perturbation, perturbation);
}

static void projection_measure() {
  // What an accurate projection is held to: measured 6.0e-17 at most over everything below (AMD EPYC 7452, one and
  // two ranks), so a factor 17 above the largest and a factor 100 below the bound.
  const double roundoff = 1e-15;
  // (1) newton_rtol and the value of the bound, on the 8x8 unit square. A solve perturbed by eps measures k eps with
  // k = 0.0806 (the continuity residual eps (B u0 - rhs_p) over the sizes of the terms of B (u0 + du)), so
  //   eps = 4e-12 is refused at every newton_rtol: the bound is below 3.2e-13 (not 1e-12, and it does not follow a
  //     loose newton_rtol up);
  //   eps = 4e-13 is accepted at every newton_rtol: the bound is above 3.2e-14 (not 1e-14, and it does not follow a
  //     tight newton_rtol down);
  //   an unperturbed solve is accepted at newton_rtol = 1e-16 and 1e-20, where a bound proportional to newton_rtol is
  //     below the unit round-off and refuses every projection.
  // The measure of the perturbed solve is a property of the discrete problem, so it is the same number on any number
  // of ranks: 8.060e-10 on one rank (AMD EPYC 7452 and Intel Xeon Gold 6330) is the reference on every rank count.
  // The tolerance 1e-3 covers the four digits of the reference (6e-5) and the round-off of forming the measure
  // (5e-17 against 8e-10); the sums of one rank of two are a different number of rows and miss it.
  const double k8 = project_on(1.0, 1.0, 8, 1e-12, 1e-8).measure / 1e-8;
  CHECK(std::fabs(k8 * 1e-8 - 8.060e-10) < 1e-3 * 8.060e-10,
        "projection measure: a solve perturbed by 1e-8 measures %.6e, expected 8.060e-10 on any number of ranks", k8 * 1e-8);
  for (double rtol : {1e-6, 1e-9, 1e-10, 1e-12, 1e-14, 1e-16, 1e-20}) {
    const ProjectionRun r0 = project_on(1.0, 1.0, 8, rtol, 0.0), r8 = project_on(1.0, 1.0, 8, rtol, 1e-8),
                        hi = project_on(1.0, 1.0, 8, rtol, 4e-12), lo = project_on(1.0, 1.0, 8, rtol, 4e-13);
    CHECK(r0.ok && r0.measure > 0 && r0.measure < roundoff,
          "projection measure: newton_rtol %.0e: an accurate projection is refused or not at round-off (%.3e)", rtol, r0.measure);
    CHECK(!r8.ok && std::fabs(r8.measure - k8 * 1e-8) < 1e-3 * k8 * 1e-8,
          "projection measure: newton_rtol %.0e: a solve perturbed by 1e-8 is accepted, or its measure %.6e is not %.6e", rtol,
          r8.measure, k8 * 1e-8);
    CHECK(!hi.ok && hi.measure > 2.5e-13 && hi.measure < 5e-13,
          "projection measure: newton_rtol %.0e: a solve of measure %.3e (above the bound 1e-13) is accepted", rtol, hi.measure);
    CHECK(lo.ok && lo.measure > 2.5e-14 && lo.measure < 5e-14,
          "projection measure: newton_rtol %.0e: a solve of measure %.3e (below the bound 1e-13) is refused", rtol, lo.measure);
  }
  // (2) the aspect ratio: channels a x 1 on 32x32 meshes (16x16 for the first, so that the mesh size is not the only
  // one tested), at the default newton_rtol and at a loose and a tight one
  double worst = 0, pmin = 1e300, pmax = 0;
  for (double a : {10.0, 1e3, 1e4, 1e5}) {
    const int n = a == 10.0 ? 16 : 32;
    for (double rtol : {1e-6, 1e-12, 1e-14}) {
      const ProjectionRun r0 = project_on(a, 1.0, n, rtol, 0.0), r8 = project_on(a, 1.0, n, rtol, 1e-8);
      CHECK(r0.ok && r0.measure > 0 && r0.measure < roundoff,
            "projection measure: channel %.0e x 1, newton_rtol %.0e: an accurate projection is refused or not at round-off (%.3e)", a,
            rtol, r0.measure);
      CHECK(!r8.ok && r8.measure > 1e-10 && r8.measure < 1e-9,
            "projection measure: channel %.0e x 1, newton_rtol %.0e: a solve perturbed by 1e-8 is accepted or measures %.3e", a, rtol,
            r8.measure);
      worst = std::max(worst, r0.measure);
      pmin = std::min(pmin, r8.measure);
      pmax = std::max(pmax, r8.measure);
    }
  }
  // (3) the length unit: the 1000 x 1 channel and the unit square, each scaled by 1e-6, 1e-3, 1 and 100. The measure
  // of an accurate solve is at round-off in every unit; the measure of a solve perturbed by 1e-8 is a property of the
  // discrete problem and the same in every unit, to the round-off of forming it (5e-17 against 2e-10: 1e-5 is a
  // factor 40 above that).
  double spread = 0;
  for (int shape = 0; shape < 2; ++shape) {
    const double lx = shape == 0 ? 1000.0 : 1.0;
    const int n = shape == 0 ? 32 : 16;
    const double ref = project_on(lx, 1.0, n, 1e-12, 1e-8).measure;
    for (double unit : {1e-6, 1e-3, 100.0}) {
      const ProjectionRun r0 = project_on(lx * unit, unit, n, 1e-12, 0.0), r8 = project_on(lx * unit, unit, n, 1e-12, 1e-8);
      CHECK(r0.ok && r0.measure > 0 && r0.measure < roundoff,
            "projection measure: %.0e x 1 in the unit %.0e: an accurate projection is refused or not at round-off (%.3e)", lx, unit,
            r0.measure);
      CHECK(!r8.ok && std::fabs(r8.measure - ref) <= 1e-5 * ref,
            "projection measure: %.0e x 1 in the unit %.0e: a solve perturbed by 1e-8 is accepted, or its measure %.9e is not %.9e", lx,
            unit, r8.measure, ref);
      worst = std::max(worst, r0.measure);
      spread = std::max(spread, std::fabs(r8.measure - ref) / ref);
    }
  }
  // (4) the momentum block. Scaling the whole solution leaves the momentum residual at round-off (its right-hand side
  // vanishes for this field), so everything above is decided by the continuity block. With the multiplier alone
  // scaled by 1 + 1e-8 the continuity block is exact and the error is the momentum residual 1e-8 B^T dp, over terms
  // that are larger than it by the cancellation in B^T dp: refused, on the square (measured 1.3e-9) and on a channel
  // (1.1e-10 at 10 x 1; the cancellation grows with the aspect ratio).
  const ProjectionRun m1 = project_on(1.0, 1.0, 8, 1e-12, 0.0, 1e-8), m2 = project_on(10.0, 1.0, 16, 1e-12, 0.0, 1e-8);
  CHECK(!m1.ok && m1.measure > 1e-11 && m1.measure < 1e-8,
        "projection measure: a multiplier perturbed by 1e-8 is accepted on the square, or measures %.3e", m1.measure);
  CHECK(!m2.ok && m2.measure > 1e-12 && m2.measure < 1e-8,
        "projection measure: a multiplier perturbed by 1e-8 is accepted on the 10 x 1 channel, or measures %.3e", m2.measure);
  if (rank0) std::printf("projection measure: multiplier perturbed by 1e-8: %.3e (square), %.3e (10 x 1 channel)\n", m1.measure, m2.measure);
  if (rank0)
    std::printf("projection measure: perturbed by 1e-8 on the 8x8 square %.9e; accurate, channels and units: at most %.3e; perturbed by "
                "1e-8 on the channels %.3e to %.3e; relative spread over the length units %.1e\n",
                k8 * 1e-8, worst, pmin, pmax, spread);
}

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  rank0 = mfem::Mpi::WorldRank() == 0;
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) {  // serial assemblies, the serial GLS solver and the one-rank field evaluation
    q1_convection_rule();
    q2_reaction();
    q3_load_rule();
    q4_gls_mass();
    q5_projection();
    final_solve_gate();
    floor_stop_and_gates();
  }
  // A launcher that starts N one-rank jobs instead of one N-rank job must not pass for a multi-rank run
  // (cmake/run_mpi_ctest.sh exports the rank count it asked for).
  if (const char* e = std::getenv("AXMP_EXPECT_RANKS"))
    CHECK(size == std::atoi(e), "started on %d rank(s), expected %s", size, e);
  projection_measure();
  if (rank0) std::printf(failures ? "test_fluids_trust: %d FAILURES\n" : "test_fluids_trust: OK\n", failures);
  return failures ? 1 : 0;
}
