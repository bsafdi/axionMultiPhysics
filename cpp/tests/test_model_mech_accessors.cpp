// AxiModel::elasticity() and AxiModel::elastic_solution(): the elastic problem and solution of the last solve.
//   1. before any solve: elasticity() is nullptr and elastic_solution() throws std::logic_error;
//   2. after a solve with mechanics enabled: the displacement read through the accessors equals sample_mech() at
//      several points (the same discrete solution, so to round-off) and the Lame solution of a thick-walled cylinder
//      under internal pressure (plane strain, ends fixed in z) to the discretization error (1e-3, as in
//      tests/test_mechanics.py: order-2 elements, 0.02 m size);
//   3. after set_mesh() without a new solve: no solution is reported (the old one belongs to the replaced space),
//      and sample_mech() refuses instead of reading it;
//   4. a new solve on the new mesh gives a solution again, consistent with sample_mech(); a solve that skips the
//      mechanics (minimal_post) and a design without mechanics report none.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/axi.hpp"
#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model.hpp"
#include "axionmultiphysics/spec.hpp"

using json = nlohmann::json;
using namespace axmp;

static int failures = 0;
static void check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
  }
}

static const double E = 200e9, NU = 0.3, A = 0.1, B = 0.3, H = 4.0, P = 1.0e6;

static json lame_doc(bool with_mechanics) {
  json doc = {
      {"schema", "axionmultiphysics.design/1"},
      {"name", "lame"},
      {"materials", {{"steel", {{"kind", "linear"}, {"mechanical", {{"E_Pa", E}, {"nu", NU}, {"alpha", 0.0}}}}}}},
      {"regions",
       json::array({{{"name", "bore"}, {"shape", {{"rect", {{"r", {0.0, A}}, {"z", {-H / 2, H / 2}}}}}}, {"material", "vacuum"}},
                    {{"name", "cyl"}, {"shape", {{"rect", {{"r", {A, B}}, {"z", {-H / 2, H / 2}}}}}}, {"material", "steel"},
                     {"mesh_size", 0.02}},
                    {{"name", "coil"}, {"shape", {{"rect", {{"r", {3.0, 3.05}}, {"z", {-0.05, 0.05}}}}}}, {"material", "winding"}}})},
      {"windings", json::array({{{"region", "coil"}, {"type", "azimuthal"}, {"ampere_turns", 1.0}}})},
      {"mesh", {{"order", 2}, {"size", 0.3}, {"domain_radius", 12.0}}}};
  if (with_mechanics)
    doc["mechanics"] = {{"pressures", json::array({{{"on", "cyl"}, {"from", "bore"}, {"p_Pa", P}}})},
                        {"supports", json::array({{{"region", "cyl"}, {"fix", "z"}}})}};
  return doc;
}

// displacement through the accessors at one physical point; false when the point is not in the mesh
static bool accessor_displacement(const AxiModel& m, double r, double z, double& ur, double& uz) {
  const axi::Space& sp = m.space();
  mfem::DenseMatrix pts(2, 1);
  pts(0, 0) = r;
  pts(1, 0) = z;
  mfem::Array<int> eid;
  mfem::Array<mfem::IntegrationPoint> ips;
  sp.mesh.FindPoints(pts, eid, ips, false);
  if (eid[0] < 0) return false;
  axi::PointEval pe;
  sp.eval_at(eid[0], ips[0], pe);
  m.elasticity()->displacement(m.elastic_solution(), eid[0], pe, ur, uz);
  return true;
}

static bool throws_logic_error(const AxiModel& m) {
  try {
    (void)m.elastic_solution();
  } catch (const std::logic_error& e) {
    return std::string(e.what()).find("no elastic solution") != std::string::npos;
  } catch (...) {
  }
  return false;
}

int main() {
  SolveOptions o;
  o.workdir = "test_model_mech_accessors_run";
  o.verbose = 0;

  AxiModel m(spec::parse_design(lame_doc(true)));
  // 1. before any solve
  check(m.elasticity() == nullptr, "elasticity() is nullptr before any solve");
  check(throws_logic_error(m), "elastic_solution() throws std::logic_error before any solve");

  // 2. after a solve
  m.run(o);
  check(m.elasticity() != nullptr, "elasticity() is set after a solve with mechanics");
  bool no_throw = true;
  try {
    (void)m.elastic_solution();
  } catch (...) {
    no_throw = false;
  }
  check(no_throw, "elastic_solution() does not throw after a solve");
  if (m.elasticity() == nullptr || !no_throw) return 1;
  const double w = B - A;
  std::vector<double> rs, zs;
  for (int i = 0; i < 9; ++i) {
    rs.push_back(A + 0.1 * w + 0.8 * w * i / 8.0);
    zs.push_back(0.0);
  }
  rs.push_back(0.2);
  zs.push_back(1.3);
  rs.push_back(0.12);
  zs.push_back(-1.7);
  const auto samples = m.sample_mech(rs, zs);
  double worst_acc = 0, worst_lame = 0;
  for (size_t i = 0; i < rs.size(); ++i) {
    double ur = 0, uz = 0;
    check(accessor_displacement(m, rs[i], zs[i], ur, uz), "point " + std::to_string(i) + " lies in the mesh");
    worst_acc = std::max({worst_acc, std::fabs(ur - samples[i].ur) / std::fabs(samples[i].ur),
                          std::fabs(uz - samples[i].uz) / (std::fabs(samples[i].ur) + 1e-300)});
    const double r = rs[i], k = P * A * A / (B * B - A * A);
    const double ur_exact = (1 + NU) / E * k * ((1 - 2 * NU) * r + B * B / r);  // plane strain, ends fixed in z
    worst_lame = std::max(worst_lame, std::fabs(ur / ur_exact - 1));
    check(std::fabs(uz) <= 1e-6 * std::fabs(ur), "axial displacement vanishes (ends fixed) at point " + std::to_string(i));
  }
  std::printf("accessor vs sample_mech: %.3e;  accessor vs Lame: %.3e\n", worst_acc, worst_lame);
  check(worst_acc < 1e-12, "displacement through the accessors equals sample_mech");
  check(worst_lame < 1e-3, "displacement agrees with the Lame solution");
  check(m.elastic_solution().closure_Fz_round_off < 1e-10, "the accessor's solution passes the equilibrium check");

  // 3. set_mesh() without a solve: the old solution belongs to the replaced space
  {
    axi::MeshOptions mo;
    mo.workdir = "test_model_mech_accessors_run";
    mo.mesher = o.mesher;
    axi::MeshBundle mb = axi::build_mesh(m.design(), mo);
    m.set_mesh(std::move(mb));
    check(m.elasticity() == nullptr, "elasticity() is nullptr after set_mesh() without a solve");
    check(throws_logic_error(m), "elastic_solution() throws after set_mesh() without a solve");
    bool refused = false;
    try {
      (void)m.sample_mech({0.2}, {0.0});
    } catch (const Error&) {
      refused = true;
    }
    check(refused, "sample_mech() refuses after set_mesh() without a solve");
  }

  // 4. a new solve on the new mesh
  m.solve(o);
  check(m.elasticity() != nullptr, "elasticity() is set again after solve() on the new mesh");
  {
    double ur = 0, uz = 0;
    const auto s = m.sample_mech({0.2}, {0.0});
    check(accessor_displacement(m, 0.2, 0.0, ur, uz) && std::fabs(ur - s[0].ur) <= 1e-12 * std::fabs(s[0].ur),
          "accessors agree with sample_mech after the re-solve");
  }

  // a solve that skips the mechanics does not leave the earlier solution behind
  {
    SolveOptions om = o;
    om.minimal_post = true;
    m.solve(om);
    check(m.elasticity() == nullptr, "elasticity() is nullptr after a solve that skips the mechanics");
    check(throws_logic_error(m), "elastic_solution() throws after a solve that skips the mechanics");
  }

  // a design without mechanics reports none
  {
    AxiModel plain(spec::parse_design(lame_doc(false)));
    plain.run(o);
    check(plain.elasticity() == nullptr, "elasticity() is nullptr for a design without mechanics");
    check(throws_logic_error(plain), "elastic_solution() throws for a design without mechanics");
  }

  if (failures == 0) std::printf("all passed\n");
  return failures == 0 ? 0 : 1;
}
