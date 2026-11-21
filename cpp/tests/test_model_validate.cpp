// Validation of a model document (cpp/include/axionmultiphysics/model_validate.hpp): route resolution, the route-level
// checks of the interface design note (sections 3.3, 3.6 to 3.8, 6.2, 6.3, 8.3 to 8.5, 10.3), the four finding kinds
// of section 14.2 and the collection of route-level findings.  Built in every build; nothing here needs the
// mesher or an engine.
//
// INDEPENDENT REFERENCES.
//   * The four documents of section 4 of the note, verbatim in cpp/tests/data/model/, and variants written here.
//   * The texts that the note gives (copied, with the section at each), the engine cells that the engine halves of the
//     chemistry and of the flow and heat matrix tests confirmed, and section 8.5's "needs" column for the requirements of a row (transcribed by
//     hand below; the transient rows take any number of ranks, as the engine half established).
//   * Arithmetic for the ramp end: 0.05000000005 differs from 0.05 by a relative 1e-9 (refused, the bound is 1e-12),
//     0.05000000000001 by 2e-13 (accepted).
//
// TEXTS.  A text that the note gives is copied from it, with the section.  Where the note gives no text, the text here is
// the test author's (AUTHOR TEXT) and is the specification.  Deviations from a note text are marked DEVIATION with the
// reason: route names are the registry's one-geometry names (navier_stokes_steady_planar, not navier_stokes_steady_2d;
// section 8.1 makes a row one geometry class); markdown emphasis and section references are not part of a runtime text;
// a remedy that is false in this version (laws on a route, region-wise data) is left out.
//
// NUMBERS in texts are printed as the shortest decimal text that reads back as the same double ("998.2", "870",
// "0.05000000005"), a number with an integral value without a decimal point.
//
// THE API THE TESTS USE (namespace axmp::model; model_validate.hpp, model_graph.hpp):
//   validate_model(json) -> ValidationReport { errors, unavailable, deferred, warnings (vectors of axmp::Error),
//                                              model, regions, steps, geometry_order, ok() }
//   StepReport { id, graph, routes, strategy }; RouteUse { physics, route }.
//   Each finding is an axmp::Error: message(), where(), hint(), code(), details() (JSON text).
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_validate.hpp"
#include "axionmultiphysics/physics_registry.hpp"

using nlohmann::json;
using axmp::Error;
namespace M = axmp::model;

static int failures = 0;
static void fail(const std::string& what) {
  std::printf("FAIL: %s\n", what.c_str());
  ++failures;
}
#define CHECK(cond, what)                 \
  do {                                    \
    if (!(cond)) fail(std::string(what)); \
  } while (0)

namespace {

using Strs = std::vector<std::string>;

json J(const char* s) { return json::parse(s); }

json load(const std::string& name) {
#ifdef AXMP_SOURCE_DIR
  const std::string path = std::string(AXMP_SOURCE_DIR) + "/cpp/tests/data/model/" + name;
#else
  const std::string path = "cpp/tests/data/model/" + name;
#endif
  std::ifstream in(path);
  if (!in) {
    fail("cannot read " + path);
    return json::object();
  }
  json d;
  in >> d;
  return d;
}
json channel() { return load("channel.json"); }
json cavity() { return load("heated_cavity.json"); }
json plate() { return load("plate_cell.json"); }
json busbar() { return load("busbar.json"); }

json details_of(const Error& e) {
  if (e.details().empty()) return nullptr;
  try {
    return json::parse(e.details());
  } catch (const std::exception&) {
    fail("details are not JSON: " + e.details());
    return nullptr;
  }
}

std::vector<Error> with_code(const std::vector<Error>& v, const std::string& code) {
  std::vector<Error> out;
  for (const auto& e : v)
    if (e.code() == code) out.push_back(e);
  return out;
}
std::string codes(const std::vector<Error>& v) {
  std::string s;
  for (const auto& e : v) s += (s.empty() ? "" : ", ") + e.code() + " at " + e.where();
  return "[" + s + "]";
}

// What one finding must be.  Empty strings and a null `details` are not compared.
struct Want {
  std::string code, where, message, hint;
  json details = nullptr;
};

// The findings of `v` with the code of `w`: exactly one, and it is `w`.
void expect_one(const std::string& label, const std::vector<Error>& v, const Want& w) {
  const auto hits = with_code(v, w.code);
  if (hits.size() != 1) {
    fail(label + ": expected one " + w.code + ", found " + std::to_string(hits.size()) + " among " + codes(v));
    return;
  }
  const Error& e = hits[0];
  if (!w.where.empty()) CHECK(e.where() == w.where, label + ": where '" + e.where() + "', expected '" + w.where + "'");
  if (!w.message.empty()) CHECK(e.message() == w.message, label + ": message\n    got:  " + e.message() + "\n    want: " + w.message);
  if (!w.hint.empty()) CHECK(e.hint() == w.hint, label + ": hint\n    got:  " + e.hint() + "\n    want: " + w.hint);
  if (!w.details.is_null())
    CHECK(details_of(e) == w.details, label + ": details\n    got:  " + e.details() + "\n    want: " + w.details.dump());
}

void run(const char* name, const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception& e) {
    fail(std::string(name) + " threw: " + e.what());
  }
}

// ------------------------------------------------------------------------------------------------ documents
// A planar slab of water, 1 m x 1 m, with a heat instance advected by a prescribed velocity: the row
// heat_prescribed_advection_planar.  `form` is written when not empty.
json slab(const std::string& form = "") {
  json d = J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "slab", "symmetry": "planar",
    "materials": {"water": {"density": 998.2, "thermal": {"k_W_mK": 0.6, "cp_J_kgK": 4182.0}}},
    "regions": [{"name": "slab", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}, "material": "water"}],
    "boundaries": {"left": {"of": "slab", "where": {"x": 0}}, "right": {"of": "slab", "where": {"x": 1}},
                   "bottom": {"of": "slab", "where": {"y": 0}}, "top": {"of": "slab", "where": {"y": 1}}},
    "physics": {"heat": {"type": "heat.transfer",
      "boundary_conditions": {"left": {"kind": "temperature", "T_K": 300}, "right": {"kind": "outflow"},
                              "bottom": {"kind": "insulation"}, "top": {"kind": "insulation"}},
      "sources": [{"kind": "prescribed_advection", "velocity_m_s": [0.001, 0]}]}},
    "mesh": {"size": 0.1}
  })j");
  if (!form.empty()) d["physics"]["heat"]["options"]["advective_form"] = form;
  return d;
}
// The slab without the source: the conduction row heat_planar.
json conduction_slab() {
  json d = slab();
  d["physics"]["heat"].erase("sources");
  return d;
}
// The slab cut into two halves 'plate' (x in [0, 0.5]) and 'gap' (x in [0.5, 1]) of materials `ma` and `mb`.
json split_slab(const json& ma, const json& mb, const std::string& name_a = "plate", const std::string& name_b = "gap",
                const std::string& mat_a = "copper", const std::string& mat_b = "titanium") {
  json d = slab();
  d["materials"] = json::object();
  d["materials"][mat_a] = ma;
  d["materials"][mat_b] = mb;
  d["regions"] = json::array();
  d["regions"].push_back({{"name", name_a}, {"shape", {{"rect", {{"x", {0, 0.5}}, {"y", {0, 1}}}}}}, {"material", mat_a}});
  d["regions"].push_back({{"name", name_b}, {"shape", {{"rect", {{"x", {0.5, 1}}, {"y", {0, 1}}}}}}, {"material", mat_b}});
  d["boundaries"] = J(R"j({"left": {"of": "LEFT", "where": {"x": 0}}, "right": {"of": "RIGHT", "where": {"x": 1}}})j");
  d["boundaries"]["left"]["of"] = name_a;
  d["boundaries"]["right"]["of"] = name_b;
  d["physics"]["heat"]["boundary_conditions"] =
      J(R"j({"left": {"kind": "temperature", "T_K": 300}, "right": {"kind": "outflow"}, "remaining": {"kind": "insulation"}})j");
  return d;
}

// Water over oil in a closed planar box: 'tank' (y in [0, 1]) of water and 'cap' (y in [1, 2]) of oil, a flow driven
// by the moving lid y = 2.  Section 8.5's water-and-oil example.
json two_fluids(const std::string& equation) {
  json d = J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "two_fluids", "symmetry": "planar",
    "materials": {"water": {"density": 998.2, "fluid": {"mu_Pa_s": 1.002e-3}},
                  "oil": {"density": 870.0, "fluid": {"mu_Pa_s": 0.03}}},
    "regions": [{"name": "tank", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}, "material": "water"},
                {"name": "cap", "shape": {"rect": {"x": [0, 1], "y": [1, 2]}}, "material": "oil"}],
    "boundaries": {"lid": {"of": "cap", "where": {"y": 2}}, "floor": {"of": "tank", "where": {"y": 0}}},
    "physics": {"flow": {"type": "fluid.incompressible", "domains": ["tank", "cap"], "options": {"equation": "EQ"},
      "boundary_conditions": {"lid": {"kind": "wall", "velocity_m_s": [0.01, 0]}, "remaining": {"kind": "wall"}}}},
    "mesh": {"size": 0.1}
  })j");
  d["physics"]["flow"]["options"]["equation"] = equation;
  return d;
}

// The heated cavity of section 4.2 cut into two regions, 'plate' (x < Lc/2) and 'pad', of materials copper and
// titanium that agree in everything the Boussinesq row needs uniform; their conductivities are `k_plate`, `k_pad`.
json split_cavity(double k_plate, double k_pad) {
  json d = cavity();
  json m = J(R"j({"density": "rho", "fluid": {"mu_Pa_s": "mu", "beta_1_K": "beta"}, "thermal": {"k_W_mK": 1.0, "cp_J_kgK": "cp"}})j");
  d["materials"] = json::object();
  d["materials"]["copper"] = m;
  d["materials"]["copper"]["thermal"]["k_W_mK"] = k_plate;
  d["materials"]["titanium"] = m;
  d["materials"]["titanium"]["thermal"]["k_W_mK"] = k_pad;
  d["regions"] = J(R"j([{"name": "plate", "shape": {"rect": {"x": [0, "Lc/2"], "y": [0, "Lc"]}}, "material": "copper"},
                        {"name": "pad", "shape": {"rect": {"x": ["Lc/2", "Lc"], "y": [0, "Lc"]}}, "material": "titanium"}])j");
  d["boundaries"] = J(R"j({"hot": {"of": "plate", "where": {"x": 0}}, "cold": {"of": "pad", "where": {"x": "Lc"}}})j");
  d["outputs"] = J(R"j({"Q_hot": {"kind": "flux", "of": "heat", "on": "hot"}})j");
  return d;
}

// The axisymmetric Stokes pipe of the parser's tests: inlet at z = 0, outlet at z = Lp, wall at r = R.
json pipe() {
  return J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "pipe", "symmetry": "axisymmetric",
    "parameters": {"R": 0.01, "Lp": 0.05},
    "materials": {"water": {"density": 998.2, "fluid": {"mu_Pa_s": 1.002e-3}}},
    "regions": [{"name": "pipe", "shape": {"rect": {"r": [0, "R"], "z": [0, "Lp"]}}, "material": "water"}],
    "boundaries": {"inlet": {"of": "pipe", "where": {"z": 0}}, "outlet": {"of": "pipe", "where": {"z": "Lp"}},
                   "wall": {"of": "pipe", "except": ["inlet", "outlet"]}},
    "physics": {"flow": {"type": "fluid.incompressible", "options": {"equation": "stokes"},
      "boundary_conditions": {"inlet": {"kind": "inlet", "velocity_m_s": [0, "2*(1-(r/R)^2)", 0]},
                              "outlet": {"kind": "outlet", "pressure_Pa": 0}, "wall": {"kind": "wall"}}}}
  })j");
}

// A 0D reactor with no study (section 3.8).
json reactor() {
  return J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "tank_model",
    "reactions": {"species": ["A", "B"], "mechanism": [{"reactants": {"A": 1}, "products": {"B": 1}, "A": 0.7, "Ea_J_mol": 0.0}]},
    "physics": {"tank": {"type": "chem.reactor"}}
  })j");
}

// The flow and heat of the cavity with the advection term alone (the one-way row).
json one_way_cavity() {
  json d = cavity();
  d["couplings"] = J(R"j([{"term": "advection", "from": "flow", "to": "heat"}])j");
  d["study"] = J(R"j([{"id": "s", "type": "stationary"}])j");
  return d;
}

const char* kNotImplementedHint =
    "The document is valid; this version validates it and cannot solve it yet. `axmp physics --routes` lists every route "
    "and whether it is implemented.";

// ================================================================================================ routes
// Section 4.4: the busbar matches no row and is the whole inventory in one finding.
void check_busbar_inventory() {
  const auto r = M::validate_model(busbar());
  CHECK(r.errors.empty(), "busbar: no error, got " + codes(r.errors));
  CHECK(r.unavailable.size() == 1, "busbar: exactly one unavailable finding, got " + codes(r.unavailable));
  expect_one("busbar", r.unavailable,
             Want{"route.not_implemented", "/physics",
                  // section 4.4, verbatim (the note wraps the message after "3d")
                  "no route for {current.conduction, heat.transfer, solid.linear} with {joule_heating, thermal_expansion} on a 3d "
                  "geometry",
                  "not implemented in this version: physics type 'current.conduction'; coupling terms 'joule_heating' and "
                  "'thermal_expansion'; 'heat.transfer' on a 3d geometry; 'solid.linear' as a physics instance. The document does "
                  "not need to change when they are. `axmp physics --routes` lists every combination that can be solved.",
                  nullptr});
  const auto hits = with_code(r.unavailable, "route.not_implemented");
  if (hits.size() == 1) {
    const json d = details_of(hits[0]);
    // no row, so no row data (the plan: "no requires or implemented in details")
    CHECK(d.is_object() && !d.contains("requires") && !d.contains("implemented") && !d.contains("route"),
          "busbar: details carry no row data: " + hits[0].details());
    // AUTHOR TEXT (the keys): what is missing, and the combination that was looked for
    CHECK(d.is_object() && d.value("missing", json()) ==
                               J(R"(["physics type 'current.conduction'", "coupling terms 'joule_heating' and 'thermal_expansion'",
                                     "'heat.transfer' on a 3d geometry", "'solid.linear' as a physics instance"])"),
          "busbar: details.missing: " + hits[0].details());
    CHECK(d.is_object() && d.value("physics", json()) == J(R"(["current.conduction", "heat.transfer", "solid.linear"])") &&
              d.value("couplings", json()) == J(R"(["joule_heating", "thermal_expansion"])") &&
              d.value("geometry", json()) == "3d" && d.value("study", json()) == "stationary",
          "busbar: details.physics, couplings, geometry, study: " + hits[0].details());
  }
  CHECK(r.steps.size() == 1 && r.steps[0].routes.size() == 1 && r.steps[0].routes[0].route.empty(),
        "busbar: one group, with no row");
}

// The channel, the cavity and the cell match a row; no row is implemented in this version.
void check_routes_found_are_not_implemented_in_u1d() {
  struct Case {
    const char* name;
    json doc;
    const char* message;
    const char* route;
    Strs physics;
    const char* engine;
    json requires;  // section 8.5, "needs" (any ranks: 0; one rank, one process: 1); every row with a geometry needs the mesher
    bool fluids_row;
  };
  const std::vector<Case> cases = {
      {"channel", channel(),
       "physics 'flow' (fluid.incompressible) takes the route 'navier_stokes_steady_planar', which is not implemented yet",
       "navier_stokes_steady_planar", {"flow"}, "NavierStokesSolver",
       J(R"({"mesher": true, "mpi": true, "fluids": true, "engine_3d": false, "max_ranks": 0})"), true},
      {"cavity", cavity(),
       "physics 'flow' (fluid.incompressible) and 'heat' (heat.transfer) take the route 'boussinesq_planar', which is not "
       "implemented yet",
       "boussinesq_planar", {"flow", "heat"}, "BoussinesqSolver",
       J(R"({"mesher": true, "mpi": true, "fluids": true, "engine_3d": false, "max_ranks": 1})"), true},
      {"cell", plate(),
       "physics 'cell' (electrochem.current_distribution) takes the route 'current_distribution_planar', which is not "
       "implemented yet",
       "current_distribution_planar", {"cell"}, "chem::CurrentDistribution",
       J(R"({"mesher": true, "mpi": false, "fluids": false, "engine_3d": false, "max_ranks": 1})"), false},
  };
  for (const auto& c : cases) {
    const auto r = M::validate_model(c.doc);
    const std::string n = c.name;
    CHECK(r.errors.empty(), n + ": no error, got " + codes(r.errors));
    CHECK(r.unavailable.size() == 1, n + ": exactly one unavailable finding, got " + codes(r.unavailable));
    // what this build lacks: the MPI build has the MPI engines and the fluids library (CMakeLists.txt builds the fluids
    // library exactly when AXMP_MPI is on); the sequential build has neither
    json missing = json::array();
#ifdef AXMP_NO_MPI
    if (c.fluids_row) missing = J(R"(["mpi", "fluids"])");
#endif
    json details = {{"route", c.route},
                    {"physics", c.physics},
                    {"engine", c.engine},
                    {"requires", c.requires},
                    {"implemented", false},
                    {"missing_in_this_build", missing}};
    expect_one(n, r.unavailable, Want{"route.not_implemented", "/physics", c.message, kNotImplementedHint, details});
    CHECK(r.steps.size() == 1 && r.steps[0].routes.size() == 1 && r.steps[0].routes[0].route == c.route &&
              r.steps[0].routes[0].physics == c.physics,
          n + ": the step's route use");
  }
  // the transient row: any number of MPI ranks (the engine half established that a time step needs no one-rank
  // accessor; section 8.5's "one rank" is corrected), the rest as the steady Navier-Stokes row
  {
    json d = channel();
    d["study"] = J(R"j([{"id": "run", "type": "transient", "time": {"t_end_s": 1.0}}])j");
    const auto rt = M::validate_model(d);
    CHECK(rt.errors.empty(), "transient channel: no error, got " + codes(rt.errors));
    json missing = json::array();
#ifdef AXMP_NO_MPI
    missing = J(R"(["mpi", "fluids"])");
#endif
    expect_one("transient channel", rt.unavailable,
               Want{"route.not_implemented", "/physics",
                    "physics 'flow' (fluid.incompressible) takes the route 'navier_stokes_transient_planar', which is not "
                    "implemented yet",
                    kNotImplementedHint,
                    json{{"route", "navier_stokes_transient_planar"},
                         {"physics", {"flow"}},
                         {"engine", "TransientNavierStokes"},
                         {"requires", J(R"({"mesher": true, "mpi": true, "fluids": true, "engine_3d": false, "max_ranks": 0})")},
                         {"implemented", false},
                         {"missing_in_this_build", missing}}});
  }
  // the reactor with its transient step: the geometry-free row, no mesher
  json tank = reactor();
  tank["study"] = J(R"j([{"id": "run", "type": "transient", "time": {"t_end_s": 10.0}}])j");
  const auto r = M::validate_model(tank);
  CHECK(r.errors.empty(), "reactor with a transient step: no error, got " + codes(r.errors));
  expect_one("reactor", r.unavailable,
             Want{"route.not_implemented", "/physics",
                  "physics 'tank' (chem.reactor) takes the route 'chem_reactor', which is not implemented yet", kNotImplementedHint,
                  nullptr});
  // the details as for the other rows; the rank limit of the reactor row is the registry's (section 8.5 states none for
  // it), so it is not transcribed here: no mesher, no MPI, no fluids library, nothing missing in any build
  const auto hits = with_code(r.unavailable, "route.not_implemented");
  if (hits.size() == 1) {
    json d = details_of(hits[0]);
    const json q = d.is_object() ? d.value("requires", json()) : json();
    CHECK(q.is_object() && q.value("mesher", true) == false && q.value("mpi", true) == false && q.value("fluids", true) == false &&
              q.value("engine_3d", true) == false,
          "reactor: details.requires: no mesher, no MPI, no fluids library, no 3D engine: " + hits[0].details());
    if (d.is_object()) d.erase("requires");
    CHECK(d == (json{{"route", "chem_reactor"},
                     {"physics", {"tank"}},
                     {"engine", M::registry().route("chem_reactor") ? M::registry().route("chem_reactor")->engine : ""},
                     {"implemented", false},
                     {"missing_in_this_build", json::array()}}),
          "reactor: details: " + hits[0].details());
  }
}

// The collection of findings, the route half: independent route-level errors of two instances are both reported; an instance with a grammar
// error gets no route-level finding.
void check_route_findings_are_collected() {
  const json d = J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "three_parts", "symmetry": "planar",
    "materials": {"water": {"density": 998.2, "fluid": {"mu_Pa_s": 1.002e-3}}, "bath": {"electrolyte": {"kappa_S_m": 1.0}}},
    "regions": [{"name": "duct", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}, "material": "water"},
                {"name": "pool", "shape": {"rect": {"x": [2, 3], "y": [0, 1]}}, "material": "bath"},
                {"name": "duct2", "shape": {"rect": {"x": [4, 5], "y": [0, 1]}}, "material": "water"}],
    "boundaries": {"in1": {"of": "duct", "where": {"x": 0}}, "out1": {"of": "duct", "where": {"x": 1}},
                   "anode": {"of": "pool", "where": {"x": 2}}, "cathode": {"of": "pool", "where": {"x": 3}},
                   "in2": {"of": "duct2", "where": {"x": 4}}, "out2": {"of": "duct2", "where": {"x": 5}}},
    "reactions": {"electrode": {"plate": {"kind": "linear", "i0_A_m2": 10.0, "E_eq_V": 0.0, "alpha_a": 0.5, "alpha_c": 0.5}}},
    "physics": {
      "flow": {"type": "fluid.incompressible", "domains": ["duct"], "options": {"equation": "navier_stokes"},
               "boundary_conditions": {"in1": {"kind": "inlet", "velocity_m_s": ["4*y*(1-y)", 0]}, "out1": {"kind": "outlet"},
                                       "remaining": {"kind": "slip"}}},
      "flow2": {"type": "fluid.incompressible", "domains": ["duct2"], "options": {"equation": "navier_stokes", "convection": "skewed"},
                "boundary_conditions": {"in2": {"kind": "inlet", "velocity_m_s": ["4*y*(1-y)", 0]}, "out2": {"kind": "outlet"},
                                        "remaining": {"kind": "slip"}}},
      "cell": {"type": "electrochem.current_distribution", "domains": ["pool"],
               "boundary_conditions": {"anode": {"kind": "electrode", "reactions": ["plate"], "current_A_m": 0.1},
                                       "cathode": {"kind": "electrode", "reactions": ["plate"], "current_A_m": -0.1},
                                       "remaining": {"kind": "insulation"}}}
    }
  })j");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 3, "collected: three errors, got " + codes(r.errors));
  CHECK(!r.errors.empty() && r.errors[0].code() == "physics.unknown_option" &&
            r.errors[0].where() == "/physics/flow2/options/convection",
        "collected: the grammar error comes first: " + codes(r.errors));
  expect_one("collected flow", r.errors, Want{"fluid.condition_not_offered", "/physics/flow/boundary_conditions/remaining/kind", "", "", nullptr});
  expect_one("collected cell", r.errors, Want{"electrochem.potential_level_free", "/physics/cell/boundary_conditions", "", "", nullptr});
  for (const auto& e : r.errors)
    CHECK(e.where().rfind("/physics/flow2", 0) != 0 || e.code() == "physics.unknown_option",
          "collected: flow2 has a grammar error and no route-level finding, got " + e.code() + " at " + e.where());
  for (const auto& e : r.unavailable)
    CHECK(details_of(e).is_object() && details_of(e).value("physics", json()) != json::array({"flow2"}),
          "collected: no route finding for flow2");
}

// Section 3.8: a reactor has no stationary solve.
void check_reactor_without_study() {
  const auto r = M::validate_model(reactor());
  CHECK(r.errors.size() == 1, "reactor: one error, got " + codes(r.errors));
  // section 3.8, verbatim
  expect_one("reactor", r.errors,
             Want{"study.type_not_offered", "/study", "physics 'tank' (chem.reactor) has no stationary solve",
                  "add \"study\": [{\"id\": \"run\", \"type\": \"transient\", \"time\": {\"t_end_s\": ...}}]",
                  J(R"({"physics": "tank", "type": "chem.reactor", "offered": ["transient"]})")});
  // a written stationary step (AUTHOR TEXT for the hint)
  json d = reactor();
  d["study"] = J(R"j([{"id": "steady", "type": "stationary"}])j");
  const auto r2 = M::validate_model(d);
  CHECK(r2.errors.size() == 1, "reactor, written stationary step: one error, got " + codes(r2.errors));
  expect_one("reactor written", r2.errors,
             Want{"study.type_not_offered", "/study/0/type", "physics 'tank' (chem.reactor) has no stationary solve",
                  "use a transient step: {\"id\": \"run\", \"type\": \"transient\", \"time\": {\"t_end_s\": ...}}",
                  J(R"({"physics": "tank", "type": "chem.reactor", "offered": ["transient"]})")});
}

// ================================================================================================ condition kinds
// Section 8.3 (cells confirmed by the engine half): axisymmetric Stokes and planar Navier-Stokes refuse slip; periodic
// is not exposed.  AUTHOR TEXT.
void check_flow_condition_not_offered() {
  const json convective = J(R"(["wall", "inlet", "outlet", "open", "traction"])");
  {
    json d = pipe();
    d["physics"]["flow"]["boundary_conditions"]["wall"] = J(R"({"kind": "slip"})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "axisymmetric Stokes slip: one error, got " + codes(r.errors));
    expect_one("axisymmetric Stokes slip", r.errors,
               Want{"fluid.condition_not_offered", "/physics/flow/boundary_conditions/wall/kind",
                    "physics 'flow': the condition kind 'slip' is not offered on this route",
                    "The engine of the route 'stokes_axisymmetric' (StokesSolver) refuses it. The kinds offered there are: wall, "
                    "inlet, outlet, open, traction.",
                    json{{"kind", "slip"}, {"route", "stokes_axisymmetric"}, {"geometry", "axisymmetric"}, {"equation", "stokes"},
                         {"offered", convective}, {"why", "engine_refuses"}}});
  }
  {
    json d = channel();
    d["physics"]["flow"]["boundary_conditions"]["walls"] = J(R"({"kind": "slip"})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "Navier-Stokes slip: one error, got " + codes(r.errors));
    expect_one("Navier-Stokes slip", r.errors,
               Want{"fluid.condition_not_offered", "/physics/flow/boundary_conditions/walls/kind",
                    "physics 'flow': the condition kind 'slip' is not offered on this route",
                    "The engine of the route 'navier_stokes_steady_planar' (NavierStokesSolver) refuses it. The kinds offered "
                    "there are: wall, inlet, outlet, open, traction.",
                    json{{"kind", "slip"}, {"route", "navier_stokes_steady_planar"}, {"geometry", "planar"},
                         {"equation", "navier_stokes"}, {"offered", convective}, {"why", "engine_refuses"}}});
    // the same kind on the planar Stokes row is offered
    d["physics"]["flow"]["options"]["equation"] = "stokes";
    const auto r2 = M::validate_model(d);
    CHECK(with_code(r2.errors, "fluid.condition_not_offered").empty(), "planar Stokes slip is offered");
  }
  {
    // periodic on the planar Stokes row: not exposed, with the reason and the roadmap item of the row
    json d = channel();
    d["physics"]["flow"]["options"]["equation"] = "stokes";
    d["boundaries"] = J(R"j({"inlet": {"of": "channel", "where": {"x": 0}}, "outlet": {"of": "channel", "where": {"x": "L"}},
                             "bottom": {"of": "channel", "where": {"y": 0}}, "top": {"of": "channel", "where": {"y": "H"}}})j");
    d["physics"]["flow"]["boundary_conditions"] =
        J(R"j({"inlet": {"kind": "inlet", "velocity_m_s": ["6*U_mean*y*(H-y)/H^2", 0]}, "outlet": {"kind": "outlet", "pressure_Pa": 0},
               "top": {"kind": "periodic", "partner": "bottom", "translation_m": [0, "-H"]}})j");
    d["outputs"] = J(R"j({"Q_out": {"kind": "flux", "of": "volume", "on": "outlet"}})j");
    std::string reason;
    if (const M::Route* row = M::registry().route("stokes_planar")) {
      const std::string why = row->not_offered.at("fluid.incompressible").at("periodic");
      reason = why.substr(std::string("not_exposed: ").size());
    }
    CHECK(!reason.empty(), "periodic: the row carries a reason");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "periodic: one error, got " + codes(r.errors));
    expect_one("periodic", r.errors,
               Want{"fluid.condition_not_offered", "/physics/flow/boundary_conditions/top/kind",
                    "physics 'flow': the condition kind 'periodic' is not offered on this route",
                    "The engine could take it, but the document path does not expose it yet: " + reason +
                        ". Roadmap item: 'periodic flow'. The kinds offered on the route 'stokes_planar' are: wall, inlet, "
                        "outlet, open, traction, slip, symmetry.",
                    json{{"kind", "periodic"}, {"route", "stokes_planar"}, {"geometry", "planar"}, {"equation", "stokes"},
                         {"offered", J(R"(["wall", "inlet", "outlet", "open", "traction", "slip", "symmetry"])")},
                         {"why", "not_exposed"}, {"reason", reason}, {"roadmap", "periodic flow"}}});
  }
}

// Section 8.3: the heat kinds by advective form.  outflow with conservative_flux is a layer rule; total_flux with the
// convective form and heat_flux with conservative_flux are engine refusals (confirmed by the engine half); on the
// Boussinesq row total_flux is refused for every form.  AUTHOR TEXT.
void check_heat_condition_not_offered() {
  const char* route = "heat_prescribed_advection_planar";
  {
    json d = slab("conservative_flux");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "outflow with conservative_flux: one error, got " + codes(r.errors));
    expect_one("outflow with conservative_flux", r.errors,
               Want{"heat.condition_not_offered", "/physics/heat/boundary_conditions/right/kind",
                    "physics 'heat': the condition kind 'outflow' is not offered with \"advective_form\": \"conservative_flux\"",
                    "With the conservative_flux form a zero flux is a zero total flux, which would block the advected heat. Use "
                    "total_flux (q_W_m2 is the total flux into the domain), or the convective or conservative_divergence form.",
                    json{{"kind", "outflow"}, {"route", route}, {"geometry", "planar"}, {"advective_form", "conservative_flux"},
                         {"offered", J(R"(["temperature", "total_flux", "convection", "insulation"])")}, {"why", "layer_rule"},
                         {"rule", "outflow_with_conservative_flux"}}});
  }
  {
    json d = slab();
    d["physics"]["heat"]["boundary_conditions"]["right"] = J(R"({"kind": "total_flux", "q_W_m2": -10.0})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "total_flux with convective: one error, got " + codes(r.errors));
    expect_one("total_flux with convective", r.errors,
               Want{"heat.condition_not_offered", "/physics/heat/boundary_conditions/right/kind",
                    "physics 'heat': the condition kind 'total_flux' is not offered with \"advective_form\": \"convective\"",
                    "The convective form's natural datum is the conductive flux. Use heat_flux (q_W_m2 is the conductive flux into "
                    "the domain), or \"advective_form\": \"conservative_flux\".",
                    json{{"kind", "total_flux"}, {"route", route}, {"geometry", "planar"}, {"advective_form", "convective"},
                         {"offered", J(R"(["temperature", "heat_flux", "convection", "insulation", "outflow"])")},
                         {"why", "engine_refuses"}}});
    // with the form that takes it, accepted
    d["physics"]["heat"]["options"]["advective_form"] = "conservative_flux";
    CHECK(M::validate_model(d).errors.empty(), "total_flux with conservative_flux is accepted");
  }
  {
    json d = slab("conservative_flux");
    d["physics"]["heat"]["boundary_conditions"]["right"] = J(R"({"kind": "temperature", "T_K": 300})");
    d["physics"]["heat"]["boundary_conditions"]["top"] = J(R"({"kind": "heat_flux", "q_W_m2": 5.0})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "heat_flux with conservative_flux: one error, got " + codes(r.errors));
    expect_one("heat_flux with conservative_flux", r.errors,
               Want{"heat.condition_not_offered", "/physics/heat/boundary_conditions/top/kind",
                    "physics 'heat': the condition kind 'heat_flux' is not offered with \"advective_form\": \"conservative_flux\"",
                    "With the conservative_flux form the natural datum is the total flux. Use total_flux (q_W_m2 is the total flux "
                    "into the domain), or the convective or conservative_divergence form.",
                    json{{"kind", "heat_flux"}, {"route", route}, {"geometry", "planar"}, {"advective_form", "conservative_flux"},
                         {"offered", J(R"(["temperature", "total_flux", "convection", "insulation"])")},
                         {"why", "engine_refuses"}}});
  }
  {
    // the conduction row: outflow is refused with conservative_flux there too (the layer rule applies on every heat row
    // whose form is selectable)
    json d = conduction_slab();
    d["physics"]["heat"]["options"]["advective_form"] = "conservative_flux";
    const auto r = M::validate_model(d);
    expect_one("conduction outflow with conservative_flux", r.errors,
               Want{"heat.condition_not_offered", "/physics/heat/boundary_conditions/right/kind", "", "", nullptr});
  }
  {
    json d = cavity();
    d["physics"]["heat"]["boundary_conditions"]["cold"] = J(R"({"kind": "total_flux", "q_W_m2": -10.0})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "Boussinesq total_flux: one error, got " + codes(r.errors));
    expect_one("Boussinesq total_flux", r.errors,
               Want{"heat.condition_not_offered", "/physics/heat/boundary_conditions/cold/kind",
                    "physics 'heat': the condition kind 'total_flux' is not offered on this route",
                    "The engine of the route 'boussinesq_planar' (BoussinesqSolver) refuses it. The kinds offered there are: "
                    "temperature, heat_flux, convection, insulation, outflow.",
                    json{{"kind", "total_flux"}, {"route", "boussinesq_planar"}, {"geometry", "planar"},
                         {"offered", J(R"(["temperature", "heat_flux", "convection", "insulation", "outflow"])")},
                         {"why", "engine_refuses"}}});
  }
  CHECK(M::validate_model(slab()).errors.empty(), "the base slab (outflow, convective form) is accepted");
}

// ================================================================================================ uniform data
// Section 8.5: two densities on a Navier-Stokes row (the note's text and details), none on a Stokes row.
void check_property_not_uniform() {
  {
    const auto r = M::validate_model(two_fluids("navier_stokes"));
    CHECK(r.errors.size() == 1, "two densities, Navier-Stokes: one error, got " + codes(r.errors));
    // section 8.5, verbatim; DEVIATION: the route is the registry's navier_stokes_steady_planar (the note:
    // navier_stokes_steady_2d)
    expect_one("two densities, Navier-Stokes", r.errors,
               Want{"physics.property_not_uniform_over_domain", "/physics/flow/domains",
                    "physics 'flow': the density must be the same in every region of its domain on this route",
                    "'tank' (water) has density 998.2 kg/m^3 and 'cap' (oil) has 870 kg/m^3. The steady Navier-Stokes engine "
                    "takes one density per solve. Use one physics instance per fluid, or equation \"stokes\" (which does not use "
                    "the density).",
                    J(R"({"property": "density", "route": "navier_stokes_steady_planar", "regions": {"tank": 998.2, "cap": 870.0}})")});
  }
  {
    const auto r = M::validate_model(two_fluids("stokes"));
    CHECK(r.errors.empty(), "two densities and two viscosities, Stokes: accepted, got " + codes(r.errors));
  }
  {
    // the viscosity is uniform on the Stokes row only for a force output with the Laplacian form (the engine's physical
    // reaction).  AUTHOR TEXT.
    json d = two_fluids("stokes");
    d["physics"]["flow"]["options"]["viscous_form"] = "laplacian";
    d["outputs"] = J(R"j({"drag": {"kind": "force", "on": "lid"}})j");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "two viscosities with a Laplacian force: one error, got " + codes(r.errors));
    expect_one("two viscosities, Laplacian force", r.errors,
               Want{"physics.property_not_uniform_over_domain", "/physics/flow/domains",
                    "physics 'flow': the viscosity must be the same in every region of its domain on this route",
                    "'tank' (water) has viscosity 0.001002 Pa s and 'cap' (oil) has 0.03 Pa s. With the Laplacian viscous form the "
                    "engine's physical reaction is defined for one viscosity, and the force output 'drag' needs it. Use "
                    "\"viscous_form\": \"stress\", or remove the force output.",
                    J(R"({"property": "fluid.mu_Pa_s", "route": "stokes_planar", "regions": {"tank": 0.001002, "cap": 0.03}, "output": "drag"})")});
    d["physics"]["flow"]["options"]["viscous_form"] = "stress";
    CHECK(M::validate_model(d).errors.empty(), "two viscosities with a force and the stress form: accepted");
    d["physics"]["flow"]["options"]["viscous_form"] = "laplacian";
    d.erase("outputs");
    CHECK(M::validate_model(d).errors.empty(), "two viscosities, Laplacian form, no force output: accepted");
  }
  // section 8.5 "The heat rows, read the same way": the water-and-oil heat instance on the prescribed-advection row and
  // on the one-way row, for two densities and for two heat capacities.  The note gives the hint "the transport operator
  // advects with one capacity rho cp per solve"; the rest is AUTHOR TEXT.
  const json water = J(R"({"density": 998.2, "fluid": {"mu_Pa_s": 1.002e-3}, "thermal": {"k_W_mK": 0.6, "cp_J_kgK": 4182.0}})");
  struct HeatCase {
    const char* name;
    bool one_way;
    const char* property;
  };
  for (const HeatCase& hc : {HeatCase{"prescribed, density", false, "density"}, HeatCase{"prescribed, cp", false, "thermal.cp_J_kgK"},
                             HeatCase{"one-way, density", true, "density"}, HeatCase{"one-way, cp", true, "thermal.cp_J_kgK"}}) {
    json oil = water;
    if (std::string(hc.property) == "density") oil["density"] = 870.0;
    else oil["thermal"]["cp_J_kgK"] = 1900.0;
    json d = two_fluids("stokes");
    d["materials"]["water"] = water;
    d["materials"]["oil"] = oil;
    d["physics"]["heat"] = J(R"j({"type": "heat.transfer",
      "boundary_conditions": {"lid": {"kind": "temperature", "T_K": 310}, "floor": {"kind": "temperature", "T_K": 300},
                              "remaining": {"kind": "insulation"}}})j");
    if (hc.one_way) {
      d["couplings"] = J(R"j([{"term": "advection", "from": "flow", "to": "heat"}])j");
    } else {
      d["physics"].erase("flow");
      d["physics"]["heat"]["sources"] = J(R"j([{"kind": "prescribed_advection", "velocity_m_s": [0, 0]}])j");
    }
    const std::string route = hc.one_way ? "flow_then_heat_planar" : "heat_prescribed_advection_planar";
    const bool density = std::string(hc.property) == "density";
    const std::string values = density ? "'tank' (water) has density 998.2 kg/m^3 and 'cap' (oil) has 870 kg/m^3."
                                       : "'tank' (water) has heat capacity 4182 J/(kg K) and 'cap' (oil) has 1900 J/(kg K).";
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, std::string(hc.name) + ": one error, got " + codes(r.errors));
    expect_one(hc.name, r.errors,
               Want{"physics.property_not_uniform_over_domain", "/physics/heat/domains",
                    std::string("physics 'heat': the ") + (density ? "density" : "heat capacity") +
                        " must be the same in every region of its domain on this route",
                    values + " The transport operator advects with one capacity rho cp per solve. Use materials that agree in "
                             "density and thermal.cp_J_kgK over the instance.",
                    json{{"property", hc.property},
                         {"route", route},
                         {"regions", density ? J(R"({"tank": 998.2, "cap": 870.0})") : J(R"({"tank": 4182.0, "cap": 1900.0})")}}});
  }
}

// Section 8.5: one-expression data.  The Boussinesq row (the note's text) and the three other heat rows as registered
// (one expression until the transport operator takes data per region).
void check_heat_data_not_regionwise() {
  {
    const auto r = M::validate_model(split_cavity(400.0, 7.5));
    CHECK(r.errors.size() == 1, "Boussinesq, two conductivities: one error, got " + codes(r.errors));
    // section 8.5.  DEVIATION: the route is boussinesq_planar (the note: boussinesq_2d), and the note's remedy "Use one
    // material law over the domain, or solve the heat instance without the buoyancy coupling (the conduction and
    // advection routes take data per region)" is replaced by the one that exists in this version: a law is refused on
    // every route of this version (route.law_not_offered) and the other heat rows take one expression until the
    // transport operator takes data per region.
    expect_one("Boussinesq, two conductivities", r.errors,
               Want{"heat.data_not_regionwise", "/physics/heat/domains",
                    "physics 'heat': the conductivity must be the same in every region of its domain on this route",
                    "'plate' (copper) has k = 400 W/(m K) and 'pad' (titanium) has 7.5. The Boussinesq engine takes the "
                    "conductivity as one function of position and is not told the region. Give the regions of the instance one "
                    "conductivity.",
                    J(R"({"data": "thermal.k_W_mK", "route": "boussinesq_planar", "regions": {"plate": 400.0, "pad": 7.5}})")});
  }
  // the conduction row (AUTHOR TEXT: the engine sentence of the transport operator)
  {
    const json copper = J(R"({"density": 8960.0, "thermal": {"k_W_mK": 400.0, "cp_J_kgK": 385.0}})");
    const json titanium = J(R"({"density": 8960.0, "thermal": {"k_W_mK": 7.5, "cp_J_kgK": 385.0}})");
    json d = split_slab(copper, titanium, "plate", "pad");
    d["physics"]["heat"].erase("sources");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "conduction row, two conductivities: one error, got " + codes(r.errors));
    expect_one("conduction row, two conductivities", r.errors,
               Want{"heat.data_not_regionwise", "/physics/heat/domains",
                    "physics 'heat': the conductivity must be the same in every region of its domain on this route",
                    "'plate' (copper) has k = 400 W/(m K) and 'pad' (titanium) has 7.5. The transport operator takes the "
                    "conductivity as one function of position in this version and is not told the region. Give the regions of "
                    "the instance one conductivity.",
                    J(R"({"data": "thermal.k_W_mK", "route": "heat_planar", "regions": {"plate": 400.0, "pad": 7.5}})")});
    // the prescribed-advection row: the same refusal (the capacities agree, so it is the only one)
    json p = split_slab(copper, titanium, "plate", "pad");
    const auto rp = M::validate_model(p);
    CHECK(rp.errors.size() == 1, "prescribed row, two conductivities: one error, got " + codes(rp.errors));
    expect_one("prescribed row, two conductivities", rp.errors,
               Want{"heat.data_not_regionwise", "/physics/heat/domains", "", "",
                    J(R"({"data": "thermal.k_W_mK", "route": "heat_prescribed_advection_planar", "regions": {"plate": 400.0, "pad": 7.5}})")});
  }
  // the one-way row
  {
    json d = split_cavity(400.0, 7.5);
    d["couplings"] = J(R"j([{"term": "advection", "from": "flow", "to": "heat"}])j");
    d["study"] = J(R"j([{"id": "s", "type": "stationary"}])j");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "one-way row, two conductivities: one error, got " + codes(r.errors));
    expect_one("one-way row, two conductivities", r.errors,
               Want{"heat.data_not_regionwise", "/physics/heat/domains", "", "",
                    J(R"({"data": "thermal.k_W_mK", "route": "flow_then_heat_planar", "regions": {"plate": 400.0, "pad": 7.5}})")});
  }
}

// "One expression" may vary in space and may not jump by region: the same coordinate expression of the heat source
// written in every region of the Boussinesq instance is accepted (treating one-expression data as uniform would refuse
// it); two different expressions are refused.  AUTHOR TEXT.
void check_same_coordinate_expression_is_accepted_on_the_boussinesq_row() {
  json d = split_cavity(0.026, 0.026);
  d["physics"]["heat"]["sources"] = J(R"j([{"kind": "heat_source", "in": "plate", "q_W_m3": "1e3*x"},
                                          {"kind": "heat_source", "in": "pad", "q_W_m3": "1e3*x"}])j");
  {
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "the same coordinate expression in every region: accepted, got " + codes(r.errors));
  }
  {
    json one = d;
    one["physics"]["heat"]["sources"] = J(R"j([{"kind": "heat_source", "q_W_m3": "1e3*x"}])j");
    CHECK(M::validate_model(one).errors.empty(), "one coordinate expression over the instance: accepted");
  }
  d["physics"]["heat"]["sources"][1]["q_W_m3"] = "2e3*x";
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "two expressions: one error, got " + codes(r.errors));
  expect_one("two heat-source expressions", r.errors,
             Want{"heat.data_not_regionwise", "/physics/heat/sources",
                  "physics 'heat': the heat source must be the same in every region of its domain on this route",
                  "'plate' has q_W_m3 = 1e3*x and 'pad' has 2e3*x. The Boussinesq engine takes the heat source as one function of "
                  "position and is not told the region. Write one source over the whole instance.",
                  J(R"({"data": "heat_source", "route": "boussinesq_planar", "regions": {"plate": "1e3*x", "pad": "2e3*x"}})")});
}

// ================================================================================================ options
// Section 8.5: heat order 1 on the Boussinesq row (the note's text); the Boussinesq advective form (an option of which
// the engine has one value; AUTHOR TEXT).
void check_heat_option_not_offered() {
  {
    json d = cavity();
    d["physics"]["heat"]["options"] = J(R"({"order": 1})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "Boussinesq heat order 1: one error, got " + codes(r.errors));
    // section 8.5, verbatim; DEVIATION: the route is boussinesq_planar (the note: boussinesq_2d)
    expect_one("Boussinesq heat order 1", r.errors,
               Want{"heat.option_not_offered", "/physics/heat/options/order",
                    "physics 'heat': the temperature order 1 is not offered on this route",
                    "the monolithic Boussinesq engine refuses a P1 temperature (\"the temperature is P2 (the coupled solver refuses "
                    "P1 T)\"); it solves Taylor-Hood P2/P1 velocity and pressure with a P2 temperature. Use \"order\": 2, or drop "
                    "the buoyancy coupling (the conduction and advection routes offer P1).",
                    J(R"({"option": "order", "value": 1, "offered": [2], "route": "boussinesq_planar", "engine": "BoussinesqSolver"})")});
  }
  {
    json d = cavity();
    d["physics"]["heat"]["options"] = J(R"({"advective_form": "conservative_flux"})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "Boussinesq advective form: one error, got " + codes(r.errors));
    expect_one("Boussinesq advective form", r.errors,
               Want{"heat.option_not_offered", "/physics/heat/options/advective_form",
                    "physics 'heat': the advective form 'conservative_flux' is not offered on this route",
                    "The monolithic Boussinesq engine has the convective form only. Use \"advective_form\": \"convective\" (the "
                    "default), or drop the buoyancy coupling (the conduction and advection routes offer the three forms).",
                    J(R"({"option": "advective_form", "value": "conservative_flux", "offered": ["convective"], "route": "boussinesq_planar", "engine": "BoussinesqSolver"})")});
  }
  {
    json d = conduction_slab();
    d["physics"]["heat"]["options"] = J(R"({"order": 1})");
    CHECK(M::validate_model(d).errors.empty(), "heat order 1 on the conduction row: accepted");
  }
}

// The transient rows take the standard convective form only (the engine half's cell; code fluid.option_not_offered).
// AUTHOR TEXT.
void check_flow_option_not_offered() {
  json d = channel();
  d["study"] = J(R"j([{"id": "run", "type": "transient", "time": {"t_end_s": 1.0}}])j");
  d["physics"]["flow"]["options"]["convection"] = "skew";
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "transient skew: one error, got " + codes(r.errors));
  expect_one("transient skew", r.errors,
             Want{"fluid.option_not_offered", "/physics/flow/options/convection",
                  "physics 'flow': the convective form 'skew' is not offered on this route",
                  "The transient Navier-Stokes engine (TransientNavierStokes) takes the standard convective form only. Use "
                  "\"convection\": \"standard\" (the default).",
                  J(R"({"option": "convection", "value": "skew", "offered": ["standard"], "route": "navier_stokes_transient_planar", "engine": "TransientNavierStokes"})")});
  d.erase("study");
  CHECK(M::validate_model(d).errors.empty(), "skew on the steady Navier-Stokes row: accepted");
}

// Section 8.3: SUPG needs an advecting velocity (the engine refuses it without one).  AUTHOR TEXT.
void check_supg_needs_advection() {
  json d = conduction_slab();
  d["physics"]["heat"]["options"] = J(R"({"stabilization": "supg"})");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "supg without advection: one error, got " + codes(r.errors));
  expect_one("supg without advection", r.errors,
             Want{"heat.supg_needs_advection", "/physics/heat/options/stabilization",
                  "physics 'heat': \"stabilization\": \"supg\" needs an advecting velocity, and nothing advects this instance",
                  "SUPG stabilizes advection, and the transport operator refuses it without a velocity. Use \"stabilization\": "
                  "\"none\", or advect the instance (an advection term from a flow, or a prescribed_advection source).",
                  J(R"({"option": "stabilization", "value": "supg", "route": "heat_planar"})")});
  json p = slab();
  p["physics"]["heat"]["options"] = J(R"({"stabilization": "supg"})");
  CHECK(M::validate_model(p).errors.empty(), "supg with a prescribed velocity: accepted");
}

// Section 8.3: one prescribed velocity covering the whole instance.
void check_prescribed_advection_partial() {
  const json water = J(R"({"density": 998.2, "thermal": {"k_W_mK": 0.6, "cp_J_kgK": 4182.0}})");
  {
    json d = split_slab(water, water, "plate", "gap", "water_a", "water_b");
    d["physics"]["heat"]["sources"][0]["in"] = "gap";
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "partial velocity: one error, got " + codes(r.errors));
    // section 8.3.  DEVIATION: the note's markdown emphasis on "continuously" is dropped, "the grammar of section 7" is
    // "the expression grammar", and the text is split at its first sentence into message and hint, with a full stop at
    // the end of the hint.
    expect_one("partial velocity", r.errors,
               Want{"heat.prescribed_advection_partial", "/physics/heat/sources/0/in",
                    "a prescribed velocity covers the whole heat instance on this route (the transport operator advects with one "
                    "velocity field and one capacity rho cp per solve)",
                    "Write one field expression over 'plate' and 'gap' that goes to zero continuously where nothing moves (the "
                    "expression grammar has no step and no conditional: `max(0, ...)` and the smooth functions are what there is), "
                    "which this route then needs 'plate' and 'gap' to agree on in `density` and `thermal.cp_J_kgK`.",
                    J(R"({"regions": ["plate", "gap"], "in": "gap", "capacities_agree": true})")});
  }
  {
    json d = split_slab(water, J(R"({"density": 8960.0, "thermal": {"k_W_mK": 0.6, "cp_J_kgK": 385.0}})"), "plate", "gap",
                        "water_a", "metal_b");
    d["physics"]["heat"]["sources"][0]["in"] = "gap";
    const auto r = M::validate_model(d);
    const auto hits = with_code(r.errors, "heat.prescribed_advection_partial");
    CHECK(hits.size() == 1 && details_of(hits[0]).value("capacities_agree", json()) == false,
          "partial velocity, capacities that differ: capacities_agree false");
  }
  {
    // a second prescribed velocity.  AUTHOR TEXT.
    json d = slab();
    d["physics"]["heat"]["sources"].push_back(J(R"({"kind": "prescribed_advection", "velocity_m_s": [0, 0.001]})"));
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "two prescribed velocities: one error, got " + codes(r.errors));
    expect_one("two prescribed velocities", r.errors,
               Want{"heat.prescribed_advection_partial", "/physics/heat/sources/1",
                    "physics 'heat' has two prescribed_advection sources; this route takes one",
                    "The transport operator advects with one velocity field per solve. Write one field expression over the whole "
                    "instance.",
                    J(R"({"sources": [0, 1]})")});
  }
}

// A prescribed velocity and an advection term on one instance.  AUTHOR TEXT.
void check_two_advections() {
  json d = one_way_cavity();
  d["physics"]["heat"]["sources"] = J(R"j([{"kind": "prescribed_advection", "velocity_m_s": [0, 0]}])j");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "two advections: one error, got " + codes(r.errors));
  expect_one("two advections", r.errors,
             Want{"heat.two_advections", "/physics/heat/sources/0",
                  "physics 'heat' is advected twice: by the prescribed_advection source and by the advection term from 'flow'",
                  "The transport operator advects with one velocity per solve. Keep the advection term to use the solved flow, or "
                  "the prescribed_advection source to use a given velocity.",
                  J(R"({"physics": "heat", "source": 0, "term_from": "flow", "coupling": 0})")});
}

// ================================================================================================ domains
// Section 8.5: the one-way row solves flow and heat on one mesh.  AUTHOR TEXT.
void check_domains_must_coincide() {
  json d = J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "duct", "symmetry": "planar",
    "materials": {"water": {"density": 998.2, "fluid": {"mu_Pa_s": 1.002e-3}, "thermal": {"k_W_mK": 0.6, "cp_J_kgK": 4182.0}}},
    "regions": [{"name": "channel", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}, "material": "water"},
                {"name": "pool", "shape": {"rect": {"x": [0, 1], "y": [1, 2]}}, "material": "water"}],
    "boundaries": {"inlet": {"of": "channel", "where": {"x": 0}}, "outlet": {"of": "channel", "where": {"x": 1}}},
    "physics": {
      "flow": {"type": "fluid.incompressible", "domains": ["channel"], "options": {"equation": "stokes"},
               "boundary_conditions": {"inlet": {"kind": "inlet", "velocity_m_s": ["4*y*(1-y)", 0]}, "outlet": {"kind": "outlet"},
                                       "remaining": {"kind": "wall"}}},
      "heat": {"type": "heat.transfer", "domains": ["channel", "pool"],
               "boundary_conditions": {"inlet": {"kind": "temperature", "T_K": 300}, "outlet": {"kind": "outflow"},
                                       "remaining": {"kind": "insulation"}}}
    },
    "couplings": [{"term": "advection", "from": "flow", "to": "heat"}]
  })j");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "domains must coincide: one error, got " + codes(r.errors));
  expect_one("domains must coincide", r.errors,
             Want{"route.domains_must_coincide", "/physics/heat/domains",
                  "physics 'flow' and 'heat' must have the same domain on this route",
                  "The route 'flow_then_heat_planar' solves both on one mesh with no interpolation: 'flow' is on 'channel' and "
                  "'heat' on 'channel', 'pool'. Give them the same domains.",
                  J(R"({"route": "flow_then_heat_planar", "physics": {"flow": ["channel"], "heat": ["channel", "pool"]}})")});
}

// Section 6.2: a region whose material lacks the route's blocks (the note's text; AUTHOR TEXT for the hint).
void check_material_block_missing() {
  json d = channel();
  d["materials"]["steel"] = J(R"({"thermal": {"k_W_mK": 45.0}})");
  d["regions"].push_back(J(R"({"name": "post", "shape": {"disk": {"center": ["L/2", "H/2"], "radius": "H/4"}}, "material": "steel"})"));
  {
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "the default domain leaves out a region without the blocks, got " + codes(r.errors));
    CHECK(r.regions.count("flow") && r.regions.at("flow") == Strs{"channel"}, "the default domain of flow is [channel]");
  }
  d["physics"]["flow"]["domains"] = J(R"(["channel", "post"])");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "block missing: one error, got " + codes(r.errors));
  // section 6.2, verbatim
  expect_one("block missing", r.errors,
             Want{"physics.material_block_missing", "/physics/flow/domains",
                  "region 'post' is in physics 'flow' (fluid.incompressible), but its material 'steel' has no `fluid` block and no "
                  "`density`",
                  "Add the missing data to material 'steel', or remove 'post' from the domains of 'flow'.",
                  J(R"({"region": "post", "material": "steel", "physics": "flow", "missing": ["fluid", "density"]})")});
}

// Section 3.6: two instances of one family on a shared region.  AUTHOR TEXT.
void check_domains_overlap() {
  json d = channel();
  json f = d["physics"]["flow"];
  f["domains"] = J(R"(["channel"])");
  d["physics"] = {{"flow_a", f}, {"flow_b", f}};
  d["outputs"] = J(R"j({"Q_out": {"kind": "flux", "of": "volume", "on": "outlet", "physics": "flow_a"}})j");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "overlap: one error, got " + codes(r.errors));
  expect_one("overlap", r.errors,
             Want{"physics.domains_overlap", "/physics/flow_b/domains", "physics 'flow_a' and 'flow_b' (family fluid) share the region 'channel'",
                  "Two instances of one family may not overlap: give them disjoint domains, or use one instance.",
                  J(R"({"physics": ["flow_a", "flow_b"], "family": "fluid", "regions": ["channel"]})")});
}

// ================================================================================================ Boussinesq row
// Gravity along the vertical axis only, either way up (the engine half's up_sign cell).  AUTHOR TEXT.
void check_gravity_direction() {
  json d = cavity();
  d["couplings"][1]["gravity_m_s2"] = J("[1, -9.81]");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "tilted gravity: one error, got " + codes(r.errors));
  expect_one("tilted gravity", r.errors,
             Want{"fluid.gravity_direction", "/couplings/1/gravity_m_s2", "the buoyancy term needs gravity along the y axis on this route",
                  "The Boussinesq engine takes gravity along the vertical axis only (y in a planar model, z in an axisymmetric one), "
                  "pointing up or down: write [0, g].",
                  J(R"({"gravity_m_s2": [1, -9.81], "route": "boussinesq_planar", "axis": "y"})")});
  d["couplings"][1]["gravity_m_s2"] = J("[0, 9.81]");
  CHECK(M::validate_model(d).errors.empty(), "gravity along +y: accepted");
  // axisymmetric: the vertical axis is z, the meridional vector is (r, z)
  json a = cavity();
  a["symmetry"] = "axisymmetric";
  a["regions"][0]["shape"] = J(R"j({"rect": {"r": [0, "Lc"], "z": [0, "Lc"]}})j");
  a["boundaries"] = J(R"j({"hot": {"of": "cavity", "where": {"z": 0}}, "cold": {"of": "cavity", "where": {"z": "Lc"}}})j");
  CHECK(M::validate_model(a).errors.empty(), "axisymmetric cavity, gravity along -z: accepted");
  a["couplings"][1]["gravity_m_s2"] = J("[-9.81, 0]");
  const auto ra = M::validate_model(a);
  CHECK(ra.errors.size() == 1, "axisymmetric radial gravity: one error, got " + codes(ra.errors));
  expect_one("axisymmetric radial gravity", ra.errors,
             Want{"fluid.gravity_direction", "/couplings/1/gravity_m_s2", "the buoyancy term needs gravity along the z axis on this route",
                  "The Boussinesq engine takes gravity along the vertical axis only (y in a planar model, z in an axisymmetric one), "
                  "pointing up or down: write [0, g].",
                  J(R"({"gravity_m_s2": [-9.81, 0], "route": "boussinesq_axisymmetric", "axis": "z"})")});
}

// The buoyancy term needs the expansion coefficient.  AUTHOR TEXT.
void check_buoyancy_needs_beta() {
  json d = cavity();
  d["materials"]["air"]["fluid"].erase("beta_1_K");
  d["physics"]["flow"]["domains"] = J(R"(["cavity"])");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "no beta: one error (not also physics.material_block_missing), got " + codes(r.errors));
  expect_one("no beta", r.errors,
             Want{"fluid.buoyancy_needs_beta", "/materials/air/fluid",
                  "region 'cavity' is in physics 'flow' with the buoyancy term, but its material 'air' has no `fluid.beta_1_K`",
                  "The Boussinesq buoyancy term needs the thermal expansion coefficient of the fluid: add \"beta_1_K\" to the fluid "
                  "block of 'air'.",
                  J(R"({"region": "cavity", "material": "air", "physics": "flow"})")});
}

// ================================================================================================ chemistry
// Every electrode galvanostatic (the chemistry cell confirmed against CurrentDistribution).  AUTHOR TEXT.
void check_potential_level_free() {
  json d = plate();
  d["physics"]["cell"]["boundary_conditions"]["anode"] = J(R"({"kind": "electrode", "reactions": ["plate"], "current_A_m": 0.1})");
  d["physics"]["cell"]["boundary_conditions"]["cathode"] = J(R"({"kind": "electrode", "reactions": ["plate"], "current_A_m": -0.1})");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "every electrode galvanostatic: one error, got " + codes(r.errors));
  expect_one("every electrode galvanostatic", r.errors,
             Want{"electrochem.potential_level_free", "/physics/cell/boundary_conditions",
                  "physics 'cell': every electrode is galvanostatic, so the potential level is free",
                  "Give at least one electrode a potential (phi_s_V); the engine refuses a problem whose electrodes are all "
                  "galvanostatic.",
                  J(R"({"electrodes": ["anode", "cathode"]})")});
  d["physics"]["cell"]["boundary_conditions"]["cathode"] = J(R"({"kind": "electrode", "reactions": ["plate"], "phi_s_V": 0.0})");
  CHECK(M::validate_model(d).errors.empty(), "one galvanostatic electrode and one at a potential: accepted");
}

// ================================================================================================ geometry order
// Section 3.3: the two orders.
void check_geometry_order() {
  json p1 = plate();
  p1["physics"]["cell"]["options"]["order"] = 1;
  {
    json d = p1;
    d["mesh"]["geometry_order"] = 2;
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "P1 with geometry order 2: one error, got " + codes(r.errors));
    // section 3.3, verbatim, split at its first sentence into message and hint (a full stop added at the end)
    expect_one("P1 with geometry order 2", r.errors,
               Want{"model.geometry_order_exceeds_field_order", "/mesh/geometry_order",
                    "mesh.geometry_order 2 exceeds the field order 1 of physics 'cell'",
                    "Element geometry of higher order than the fields breaks the exactness of rigid motions, and the P1 "
                    "current-distribution engine takes straight elements only. Use order 2 for 'cell', or geometry order 1.",
                    J(R"({"geometry_order": 2, "field_order": 1, "physics": "cell"})")});
  }
  {
    // the default is lowered to 1 by the P1 instance; on a rectangle there is nothing curved to warn about
    const auto r = M::validate_model(p1);
    CHECK(r.errors.empty(), "P1, no key: accepted");
    CHECK(r.geometry_order == 1, "P1, no key: geometry order 1, got " + std::to_string(r.geometry_order));
    CHECK(with_code(r.warnings, "mesh.geometry_order_lowered").empty(), "P1 on a rectangle: no warning");
  }
  {
    // with a curved boundary (a disk, by the syntactic test): the warning names the instance.  AUTHOR TEXT.
    json d = p1;
    d["regions"].push_back(J(R"({"name": "probe", "shape": {"disk": {"center": ["gap/2", "Hc/2"], "radius": "gap/4"}}, "material": "bath"})"));
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "P1 with a disk: accepted, got " + codes(r.errors));
    CHECK(r.geometry_order == 1, "P1 with a disk: geometry order 1");
    expect_one("P1 with a disk", r.warnings,
               Want{"mesh.geometry_order_lowered", "/mesh",
                    "the geometry order is lowered to 1 because physics 'cell' has field order 1, and this model has curved "
                    "boundaries",
                    "On straight elements a curved boundary is a polygon, and every instance of the model is solved on this "
                    "geometry. Use order 2 for 'cell' to keep curved elements, or write \"geometry_order\": 1 in \"mesh\" to state it.",
                    J(R"({"geometry_order": 1, "physics": "cell", "cause": "order=1"})")});
    d["mesh"]["geometry_order"] = 1;
    const auto r2 = M::validate_model(d);
    CHECK(with_code(r2.warnings, "mesh.geometry_order_lowered").empty(), "an explicit geometry order 1: no warning");
  }
  {
    const auto r = M::validate_model(plate());
    CHECK(r.geometry_order == 2, "P2, no key: geometry order 2 (curved boundaries are curved)");
  }
  {
    // supg with an explicit 2: section 3.3, verbatim, split at its last sentence (a full stop added at the end)
    json d = slab();
    d["physics"]["heat"]["options"] = J(R"({"stabilization": "supg"})");
    d["mesh"]["geometry_order"] = 2;
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "supg with geometry order 2: one error, got " + codes(r.errors));
    expect_one("supg with geometry order 2", r.errors,
               Want{"model.geometry_order_not_offered", "/mesh/geometry_order",
                    "mesh.geometry_order 2 is not offered with \"stabilization\": \"supg\" of physics 'heat': the transport operator "
                    "does not offer SUPG on isoparametric (curved) elements (the mapped Hessian is not implemented)",
                    "Use geometry order 1, or \"stabilization\": \"none\".",
                    J(R"({"physics": "heat", "option": "stabilization", "value": "supg", "engine_text": "SUPG on isoparametric (curved) elements is not offered"})")});
  }
  {
    // supg lowers the default; with a curved boundary it warns.  AUTHOR TEXT.
    json d = slab();
    d["physics"]["heat"]["options"] = J(R"({"stabilization": "supg"})");
    d["regions"].push_back(J(R"({"name": "pebble", "shape": {"disk": {"center": [0.5, 0.5], "radius": 0.1}}, "material": "water"})"));
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "supg with a disk: accepted, got " + codes(r.errors));
    CHECK(r.geometry_order == 1, "supg: geometry order 1");
    expect_one("supg with a disk", r.warnings,
               Want{"mesh.geometry_order_lowered", "/mesh",
                    "the geometry order is lowered to 1 because physics 'heat' has \"stabilization\": \"supg\", and this model has "
                    "curved boundaries",
                    "SUPG needs straight elements. On straight elements a curved boundary is a polygon, and every instance of the "
                    "model, a flow included, is solved on this geometry. Use \"stabilization\": \"none\" to keep curved elements, "
                    "or write \"geometry_order\": 1 in \"mesh\" to state it.",
                    J(R"({"geometry_order": 1, "physics": "heat", "cause": "stabilization=supg"})")});
  }
  {
    // an imported mesh: the order checks are deferred (the order is the file's).  AUTHOR TEXT.
    json d = p1;
    d["mesh"] = J(R"({"file": "cell.msh", "info_file": "cell.json"})");
    d["boundaries"] = J(R"({"anode": {"group": "anode"}, "cathode": {"group": "cathode"}})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "P1 on an imported mesh: no error, got " + codes(r.errors));
    CHECK(r.geometry_order == 0, "an imported mesh: the geometry order is the file's (0 here)");
    expect_one("P1 on an imported mesh", r.deferred,
               Want{"model.geometry_order_exceeds_field_order", "/mesh/file",
                    "the geometry order of the imported mesh 'cell.msh' is checked against the field order 1 of physics 'cell' when "
                    "the file is read",
                    "An order-2 mesh is refused then: re-export it at order 1, or use order 2 for 'cell'.",
                    J(R"({"physics": "cell", "field_order": 1, "file": "cell.msh"})")});
  }
  {
    json d = slab();
    d["physics"]["heat"]["options"] = J(R"({"stabilization": "supg"})");
    d["mesh"] = J(R"({"file": "duct.msh", "info_file": "duct.json"})");
    d["boundaries"] = J(R"({"left": {"group": "left"}, "right": {"group": "right"}, "bottom": {"group": "bottom"}, "top": {"group": "top"}})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "supg on an imported mesh: no error, got " + codes(r.errors));
    expect_one("supg on an imported mesh", r.deferred,
               Want{"model.geometry_order_not_offered", "/mesh/file",
                    "the geometry order of the imported mesh 'duct.msh' is checked against \"stabilization\": \"supg\" of physics "
                    "'heat' when the file is read",
                    "An order-2 mesh is refused then: re-export it at order 1, or use \"stabilization\": \"none\".",
                    J(R"({"physics": "heat", "option": "stabilization", "value": "supg", "file": "duct.msh"})")});
    // the element types of the file are deferred too (the heat rows take triangles only)
    expect_one("heat on an imported mesh", r.deferred,
               Want{"model.element_type_not_offered", "/mesh/file",
                    "the element types of the imported mesh 'duct.msh' are checked against physics 'heat' when the file is read",
                    "The route 'heat_prescribed_advection_planar' takes triangles only: a quadrilateral or mixed mesh is refused "
                    "then.",
                    J(R"({"physics": "heat", "route": "heat_prescribed_advection_planar", "accepts": ["triangle"], "file": "duct.msh"})")});
  }
}

// ================================================================================================ study
// Section 3.8 and the example of section 4.3.
json ramped_cell(const json& v_cell, const json& values, const std::string& ramp = "/physics/cell/boundary_conditions/anode/phi_s_V") {
  json d = plate();
  d["parameters"]["V_cell"] = v_cell;
  d["study"] = json::array({{{"id", "s"}, {"type", "stationary"}, {"continuation", {{"ramp", ramp}, {"values", values}}}}});
  return d;
}

void check_ramps() {
  {
    const auto r = M::validate_model(ramped_cell(0.6, J(R"([0.2, 0.4, "V_cell"])")));
    CHECK(r.errors.empty(), "a ramp ending in \"V_cell\" = the document's value: accepted, got " + codes(r.errors));
  }
  {
    const auto r = M::validate_model(ramped_cell(0.05, J(R"([0.2, 0.4, 0.6])")));
    CHECK(r.errors.size() == 1, "ramp end mismatch: one error, got " + codes(r.errors));
    // section 4.3 gives the message; AUTHOR TEXT for the hint
    expect_one("ramp end mismatch", r.errors,
               Want{"study.ramp_end_mismatch", "/study/0/continuation/values/2", "the ramp ends at 0.6 and the document says 0.05",
                    "The last value of a ramp is the value the document gives the field "
                    "(/physics/cell/boundary_conditions/anode/phi_s_V), so that the document states the model that is finally "
                    "solved. End the ramp at 0.05, or give the field the value 0.6.",
                    J(R"({"ramp": "/physics/cell/boundary_conditions/anode/phi_s_V", "last": 0.6, "document": 0.05})")});
  }
  {
    // a relative 1e-9: refused (the bound is a relative 1e-12)
    const auto r = M::validate_model(ramped_cell(0.05, J(R"([0.02, 0.05000000005])")));
    expect_one("ramp end 1e-9 off", r.errors,
               Want{"study.ramp_end_mismatch", "/study/0/continuation/values/1",
                    "the ramp ends at 0.05000000005 and the document says 0.05", "", nullptr});
  }
  {
    // a relative 2e-13: accepted
    const auto r = M::validate_model(ramped_cell(0.05, J(R"([0.02, 0.05000000000001])")));
    CHECK(r.errors.empty(), "ramp end 2e-13 off: accepted, got " + codes(r.errors));
  }
  {
    // not rampable.  AUTHOR TEXT.
    const auto r = M::validate_model(ramped_cell(0.05, J(R"([0.02, 0.05])"), "/parameters/V_cell"));
    CHECK(r.errors.size() == 1, "not rampable: one error, got " + codes(r.errors));
    expect_one("not rampable", r.errors,
               Want{"study.field_not_rampable", "/study/0/continuation/ramp", "'/parameters/V_cell' is not a rampable field",
                    "A ramp steps a field that the engine can warm-start: in this version one electrode's phi_s_V or current. The "
                    "rampable fields of this document are: /physics/cell/boundary_conditions/anode/phi_s_V, "
                    "/physics/cell/boundary_conditions/cathode/phi_s_V.",
                    J(R"({"ramp": "/parameters/V_cell", "rampable": ["/physics/cell/boundary_conditions/anode/phi_s_V", "/physics/cell/boundary_conditions/cathode/phi_s_V"]})")});
    const auto r2 = M::validate_model(ramped_cell(0.05, J(R"([290, 298.15])"), "/physics/cell/options/T_K"));
    expect_one("an option is not rampable", r2.errors, Want{"study.field_not_rampable", "/study/0/continuation/ramp", "", "", nullptr});
  }
}

// Section 3.8: the default strategy, a strategy the route does not offer, and a two-way group without one.  The note
// gives "this route offers: monolithic" (section 4.2); the rest is AUTHOR TEXT.
void check_strategies() {
  {
    const auto r = M::validate_model(one_way_cavity());
    CHECK(r.errors.empty(), "one-way: accepted, got " + codes(r.errors));
    CHECK(r.steps.size() == 1 && r.steps[0].strategy == "one_way", "one-way: the default strategy one_way on an acyclic graph");
    json d = one_way_cavity();
    d["study"][0]["coupling"] = J(R"({"strategy": "monolithic"})");
    const auto r2 = M::validate_model(d);
    CHECK(r2.errors.size() == 1, "one-way with monolithic: one error, got " + codes(r2.errors));
    expect_one("one-way with monolithic", r2.errors,
               Want{"study.strategy_not_offered", "/study/0/coupling/strategy",
                    "the coupling strategy 'monolithic' is not offered on the route 'flow_then_heat_planar'; this route offers: "
                    "one_way",
                    "Remove \"strategy\" to use the route's default (one_way), or write one of the offered strategies.",
                    J(R"({"strategy": "monolithic", "route": "flow_then_heat_planar", "offered": ["one_way"], "step": "s"})")});
  }
  {
    json d = cavity();
    d["study"][0]["coupling"]["strategy"] = "gauss_seidel";
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "Boussinesq with gauss_seidel: one error, got " + codes(r.errors));
    expect_one("Boussinesq with gauss_seidel", r.errors,
               Want{"study.strategy_not_offered", "/study/0/coupling/strategy",
                    "the coupling strategy 'gauss_seidel' is not offered on the route 'boussinesq_planar'; this route offers: "
                    "monolithic",
                    "Remove \"strategy\" to use the route's default (monolithic), or write one of the offered strategies.",
                    J(R"({"strategy": "gauss_seidel", "route": "boussinesq_planar", "offered": ["monolithic"], "step": "steady"})")});
  }
  {
    // the busbar's two-way group {current, heat} (closed by the property law) has no route, so no default strategy
    json d = busbar();
    d["study"][0].erase("coupling");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "busbar without a strategy: one error, got " + codes(r.errors));
    expect_one("busbar without a strategy", r.errors,
               Want{"study.two_way_needs_strategy", "/study/0",
                    "physics 'current' and 'heat' form a two-way group in step 'steady', which needs a coupling strategy",
                    "Write \"coupling\": {\"strategy\": ...} in the step: gauss_seidel, aitken or monolithic, as the route of the "
                    "group offers. A two-way group is never run as one unlabelled sweep.",
                    J(R"({"step": "steady", "group": ["current", "heat"]})")});
    CHECK(r.steps.size() == 1 && r.steps[0].strategy.empty(), "busbar without a strategy: the step's strategy is empty");
  }
}

// Section 6.3: a law in a property a route of this version reads; a law whose field nothing provides.
void check_laws_on_routes() {
  {
    json d = channel();
    d["materials"]["water"]["fluid"]["mu_Pa_s"] =
        J(R"j({"law": "expression", "of": ["T"], "expr": "1.002e-3*(1 + 0*(T - 293.15))", "range": {"T": [273.15, 373.15]}, "source": "test"})j");
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "given": {"T": 293.15}}])j");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "a law on a route: one error, got " + codes(r.errors));
    // AUTHOR TEXT
    expect_one("a law on a route", r.errors,
               Want{"route.law_not_offered", "/materials/water/fluid/mu_Pa_s",
                    "physics 'flow': the viscosity of 'water' is a law of T, and this route takes a constant",
                    "The engine of the route 'navier_stokes_steady_planar' takes fluid.mu_Pa_s as one number per region; a law of a "
                    "field is not offered on any route of this version. Write a number or an expression over the parameters.",
                    J(R"({"property": "fluid.mu_Pa_s", "material": "water", "route": "navier_stokes_steady_planar", "physics": "flow"})")});
  }
  {
    // section 3.7: the field must be provided in the step, or given; both remedies in the hint.  AUTHOR TEXT.
    json d = J(R"j({
      "schema": "axionmultiphysics.design/1", "name": "bar", "symmetry": "3d",
      "materials": {"copper": {"density": 8960.0,
        "electric": {"sigma_S_m": {"law": "expression", "of": ["T"], "expr": "5.998e7/(1 + 3.9e-3*(T - 293.15))",
                                   "range": {"T": [250.0, 500.0]}, "source": "test"}}}},
      "regions": [{"name": "bar", "solid": {"box": {"min": [0, 0, 0], "max": [0.1, 0.01, 0.01]}}, "material": "copper"}],
      "boundaries": {"end_a": {"of": "bar", "where": {"x": 0}}, "end_b": {"of": "bar", "where": {"x": 0.1}}},
      "physics": {"current": {"type": "current.conduction",
        "boundary_conditions": {"end_a": {"kind": "potential", "V_V": 0.001}, "end_b": {"kind": "ground"},
                                "remaining": {"kind": "insulation"}}}}
    })j");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "a field nothing provides: one error, got " + codes(r.errors));
    expect_one("a field nothing provides", r.errors,
               Want{"law.field_not_provided", "/materials/copper/electric/sigma_S_m",
                    "the law of electric.sigma_S_m in material 'copper' reads T, which no physics of step 'default' provides on "
                    "'bar'",
                    "Add a physics that provides T on 'bar' to the step (heat.transfer provides T), or give T a fixed value in the "
                    "step's \"given\": {\"T\": ...}.",
                    J(R"({"property": "electric.sigma_S_m", "material": "copper", "field": "T", "step": "default", "regions": ["bar"], "providers": ["heat.transfer"]})")});
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "given": {"T": 293.15}}])j");
    CHECK(M::validate_model(d).errors.empty(), "the field given in the step: accepted");
  }
}

// Section 10.3: a force with the Laplacian form on a Navier-Stokes row, on a moving wall (the note's text) and on a wall
// at rest (accepted).
void check_force_rule() {
  json d = channel();
  d["boundaries"] = J(R"j({"inlet": {"of": "channel", "where": {"x": 0}}, "outlet": {"of": "channel", "where": {"x": "L"}},
                           "rotor": {"of": "channel", "where": {"y": "H"}}, "floor": {"of": "channel", "where": {"y": 0}}})j");
  d["physics"]["flow"]["boundary_conditions"] =
      J(R"j({"inlet": {"kind": "inlet", "velocity_m_s": ["6*U_mean*y*(H-y)/H^2", 0]}, "outlet": {"kind": "outlet", "pressure_Pa": 0},
             "rotor": {"kind": "wall", "velocity_m_s": [0.01, 0]}, "floor": {"kind": "wall"}})j");
  d["outputs"] = J(R"j({"drag": {"kind": "force", "on": "rotor"}, "F_floor": {"kind": "force", "on": "floor"}})j");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "force on a moving wall: one error (the wall at rest is accepted), got " + codes(r.errors));
  // section 10.3, verbatim; DEVIATION: the route is navier_stokes_steady_planar (the note: navier_stokes_steady_2d)
  expect_one("force on a moving wall", r.errors,
             Want{"output.force_needs_stress_form", "/outputs/drag",
                  "output 'drag': a force on 'rotor' is not available with the Laplacian viscous form on this route",
                  "'rotor' is a moving wall. The Navier-Stokes engine returns, for the Laplacian form, the pseudo-traction mu du/dn "
                  "- p n, which is the force only on walls at rest. Use \"viscous_form\": \"stress\" (the physical form), or "
                  "\"equation\": \"stokes\" if the flow is creeping.",
                  J(R"({"boundary": "rotor", "condition": "wall with velocity_m_s", "route": "navier_stokes_steady_planar", "accepted_on": "wall boundaries at rest"})")});
  d["physics"]["flow"]["options"]["viscous_form"] = "stress";
  CHECK(M::validate_model(d).errors.empty(), "the stress form: the force is accepted on any wall");
  CHECK(M::validate_model(channel()).errors.empty(), "the channel's force on its walls at rest is accepted (section 4.1)");
}

// ================================================================================================ warnings
void check_warnings() {
  {
    // section 17: one beta warning per type, term and route.  AUTHOR TEXT (code model.beta_feature).
    const auto r = M::validate_model(cavity());
    CHECK(r.warnings.size() == 5, "cavity: five warnings (two types, two terms, one route), got " + codes(r.warnings));
    const char* verified =
        "Beta: verified against manufactured, closed-form and cross-code solutions; no published benchmark value is compared.";
    const char* beta = "Beta: no published benchmark value is compared.";
    struct W5 {
      const char* message;
      const char* where;
      std::string hint;
      json details;
    };
    const std::vector<W5> want = {
        {"physics type 'fluid.incompressible' is beta", "/physics/flow/type", std::string(verified) + " See docs/validation/fluids.md.",
         J(R"({"feature": "physics_type", "name": "fluid.incompressible"})")},
        {"physics type 'heat.transfer' is beta", "/physics/heat/type", std::string(verified) + " See docs/validation/fluids.md.",
         J(R"({"feature": "physics_type", "name": "heat.transfer"})")},
        {"coupling term 'advection' is beta", "/couplings/0/term", beta, J(R"({"feature": "coupling_term", "name": "advection"})")},
        {"coupling term 'buoyancy' is beta", "/couplings/1/term", beta, J(R"({"feature": "coupling_term", "name": "buoyancy"})")},
        {"route 'boussinesq_planar' is beta", "/physics", beta, J(R"({"feature": "route", "name": "boussinesq_planar"})")},
    };
    for (size_t i = 0; i < want.size() && i < r.warnings.size(); ++i) {
      const Error& e = r.warnings[i];
      const std::string n = "beta warning " + std::to_string(i);
      CHECK(e.code() == "model.beta_feature", n + ": code " + e.code());
      CHECK(e.message() == want[i].message, n + ": message " + e.message());
      CHECK(e.where() == want[i].where, n + ": where " + e.where());
      CHECK(e.hint() == want[i].hint, n + ": hint " + e.hint());
      CHECK(details_of(e) == want[i].details, n + ": details " + e.details());
    }
    // the busbar: heat.transfer is the only beta type; the absent types and terms are unavailable, not beta
    const auto rb = M::validate_model(busbar());
    const auto bw = with_code(rb.warnings, "model.beta_feature");
    CHECK(bw.size() == 1 && details_of(bw[0]) == J(R"({"feature": "physics_type", "name": "heat.transfer"})"),
          "busbar: one beta warning, for heat.transfer, got " + codes(rb.warnings));
  }
  {
    // the stress form with an outlet.  AUTHOR TEXT.
    json d = channel();
    d["physics"]["flow"]["options"].erase("viscous_form");
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "stress form with an outlet: valid");
    expect_one("outlet with the stress form", r.warnings,
               Want{"fluid.outlet_stress_form", "/physics/flow/boundary_conditions/outlet",
                    "physics 'flow': the outlet 'outlet' imposes zero total traction with the stress viscous form, which a fully "
                    "developed profile does not satisfy",
                    "The tangential traction of a developed profile is mu du_t/dn, so the flow bends near the outlet and the mean "
                    "outlet pressure is not exactly the value written. Use \"viscous_form\": \"laplacian\" when the outlet is "
                    "straight and normal to the flow, or a longer outlet section. The developed-outflow condition is roadmap item "
                    "F2b.",
                    J(R"({"boundary": "outlet", "kind": "outlet", "viscous_form": "stress"})")});
    CHECK(with_code(M::validate_model(channel()).warnings, "fluid.outlet_stress_form").empty(), "the Laplacian form: no warning");
  }
  {
    // the Laplacian form with a traction on a curved boundary.  AUTHOR TEXT.
    json d = channel();
    d["materials"]["steel"] = J(R"({"thermal": {"k_W_mK": 45.0}})");
    d["regions"].push_back(J(R"({"name": "post", "shape": {"disk": {"center": ["L/2", "H/2"], "radius": "H/4"}}, "material": "steel"})"));
    d["boundaries"]["cylinder"] = J(R"({"between": ["channel", "post"]})");
    d["boundaries"]["walls"] = J(R"({"of": "channel", "except": ["inlet", "outlet", "cylinder"]})");
    d["physics"]["flow"]["boundary_conditions"]["cylinder"] = J(R"({"kind": "traction", "traction_Pa": [0, 0]})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "a traction on a curved boundary: valid, got " + codes(r.errors));
    expect_one("traction on a curved boundary", r.warnings,
               Want{"fluid.pseudo_traction_on_curved_boundary", "/physics/flow/boundary_conditions/cylinder",
                    "physics 'flow': the traction condition on 'cylinder', a curved boundary, imposes a pseudo-traction with the "
                    "Laplacian viscous form, not the physical traction",
                    "With \"viscous_form\": \"laplacian\" the natural condition is mu du/dn - p n, which equals the Cauchy traction "
                    "only on straight boundaries. Use \"viscous_form\": \"stress\" for a physical traction on a curved boundary.",
                    J(R"({"boundary": "cylinder", "kind": "traction", "viscous_form": "laplacian"})")});
    d["physics"]["flow"]["options"]["viscous_form"] = "stress";
    CHECK(with_code(M::validate_model(d).warnings, "fluid.pseudo_traction_on_curved_boundary").empty(), "the stress form: no pseudo-traction warning");
  }
}

// ================================================================================================ time and failed materials
// The time of a transient step.  The parser makes t a variable of an instance's field expressions when a transient step
// solves the instance; a stationary step that solves the same instance has no time, so a datum that uses t is refused
// there rather than evaluated at an assumed time.  AUTHOR TEXT (the code is the parser's expr.unknown_variable).
void check_time_in_a_stationary_step() {
  json d = channel();
  d["physics"]["flow"]["boundary_conditions"]["inlet"]["velocity_m_s"][0] = "6*U_mean*y*(H-y)/H^2*(1-exp(-t))";
  d["study"] = J(R"j([{"id": "steady", "type": "stationary", "physics": ["flow"]},
                     {"id": "ramp", "type": "transient", "physics": ["flow"], "initial_from": "steady", "time": {"t_end_s": 1.0}}])j");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "t in a stationary step: one error, got " + codes(r.errors));
  expect_one("t in a stationary step", r.errors,
             Want{"expr.unknown_variable", "/physics/flow/boundary_conditions/inlet/velocity_m_s/0",
                  "'t' is not a variable in the stationary step 'steady', which solves physics 'flow'",
                  "The time t is a variable only in a transient step. Solve 'flow' in transient steps only, or write the datum "
                  "without t.",
                  J(R"({"name": "t", "step": "steady", "physics": "flow"})")});
  // without t, the two steps are accepted
  d["physics"]["flow"]["boundary_conditions"]["inlet"]["velocity_m_s"][0] = "6*U_mean*y*(H-y)/H^2";
  CHECK(M::validate_model(d).errors.empty(), "the same steps without t: accepted");
  // with the transient step alone, t is a variable
  json t = channel();
  t["physics"]["flow"]["boundary_conditions"]["inlet"]["velocity_m_s"][0] = "6*U_mean*y*(H-y)/H^2*(1-exp(-t))";
  t["study"] = J(R"j([{"id": "ramp", "type": "transient", "time": {"t_end_s": 1.0}}])j");
  CHECK(M::validate_model(t).errors.empty(), "t in a transient step only: accepted");
}

// A material whose override of a built-in one has a grammar error: the parser keeps the built-in in the partial model,
// and validation must not report route-level findings about the built-in's data in its place.  The regions of a material
// with a grammar finding (a pointer under /materials/<name>) take part in no material check of a route.
void check_failed_material_gives_no_route_finding() {
  json d = split_slab(J(R"({"density": 8960.0, "thermal": {"k_W_mK": 400.0, "cp_J_kgK": 385.0, "bogus": 1}})"),
                      J(R"({"density": 8960.0, "thermal": {"k_W_mK": 400.0, "cp_J_kgK": 385.0}})"), "plate", "pad");
  const auto r = M::validate_model(d);
  CHECK(!r.errors.empty(), "a failed material: its grammar error is reported");
  for (const auto& e : r.errors)
    CHECK(e.where().rfind("/materials/copper", 0) == 0,
          "a failed material: only its own grammar error, got " + e.code() + " at " + e.where());
}

// ================================================================================================ deferred checks
// Section 14.2: what needs the geometry kernel (the resolution of the named boundaries, the partition rule of section
// 5.5) is reported as deferred, one finding for the boundaries and one per physics that has a boundary.  IMPLEMENTER
// TEXT throughout this block and the ones below it: the note names the codes and gives no text for a deferred finding.
const char* kPartitionChecks = R"(["physics.boundary_uncovered", "physics.boundary_double", "physics.boundary_interior"])";
std::string partition_hint(const std::string& id) {
  return "Every part of the boundary of the domain of '" + id + "' takes exactly one condition, and a boundary interior to the domain "
         "takes none. This validation reads the structure of the document only; an uncovered part, a part covered twice and a "
         "condition on an interior boundary are refused then.";
}
void check_deferred_boundaries() {
  {
    const auto r = M::validate_model(channel());
    CHECK(r.errors.empty(), "channel: no error, got " + codes(r.errors));
    CHECK(r.deferred.size() == 2, "channel: two deferred findings (the boundaries, the partition rule of flow), got " + codes(r.deferred));
    expect_one("channel boundaries", r.deferred,
               Want{"boundary.empty_selection", "/boundaries",
                    "the named boundaries 'inlet', 'outlet' and 'walls' are resolved when the geometry is built",
                    "This validation reads the structure of the document only. The mesher finds the boundary entities that each "
                    "selector names; a boundary that selects nothing is refused then.",
                    J(R"({"boundaries": ["inlet", "outlet", "walls"], "file": null})")});
    json details = J(R"({"physics": "flow", "conditions": ["inlet", "outlet", "walls"]})");
    details["checks"] = J(kPartitionChecks);
    expect_one("channel partition", r.deferred,
               Want{"physics.boundary_uncovered", "/physics/flow/boundary_conditions",
                    "physics 'flow': whether its boundary conditions cover the boundary of its domain exactly once is checked when "
                    "the geometry is built",
                    partition_hint("flow"), details});
  }
  {
    // one partition finding per physics; "remaining" is a key like another
    const auto r = M::validate_model(cavity());
    const auto part = with_code(r.deferred, "physics.boundary_uncovered");
    CHECK(part.size() == 2, "cavity: the partition rule of flow and of heat, got " + codes(r.deferred));
    if (part.size() == 2) {
      CHECK(part[0].where() == "/physics/flow/boundary_conditions" && part[1].where() == "/physics/heat/boundary_conditions",
            "cavity: the partition findings are at the two instances, got " + codes(part));
      CHECK(details_of(part[0])["conditions"] == J(R"(["remaining"])"), "cavity: flow's only key is \"remaining\": " + part[0].details());
      CHECK(details_of(part[1])["conditions"] == J(R"(["cold", "hot", "remaining"])"), "cavity: heat's keys: " + part[1].details());
    }
    CHECK(with_code(r.deferred, "boundary.empty_selection").size() == 1, "cavity: one finding for the named boundaries");
    CHECK(r.deferred.size() == 3, "cavity: three deferred findings, got " + codes(r.deferred));
  }
  {
    const auto r = M::validate_model(plate());
    CHECK(r.deferred.size() == 2, "plate cell: two deferred findings, got " + codes(r.deferred));
    expect_one("plate cell partition", r.deferred, Want{"physics.boundary_uncovered", "/physics/cell/boundary_conditions", "", "", nullptr});
  }
  {
    // a physics without a geometry has no boundary, and a document without named boundaries nothing to resolve
    const auto r = M::validate_model(reactor());
    CHECK(r.deferred.empty(), "reactor: nothing is deferred, got " + codes(r.deferred));
  }
  {
    // an imported mesh: the groups of its info file resolve the boundaries
    json d = slab();
    d["mesh"] = J(R"({"file": "duct.msh", "info_file": "duct.json"})");
    d["boundaries"] = J(R"({"left": {"group": "left"}, "right": {"group": "right"}, "bottom": {"group": "bottom"}, "top": {"group": "top"}})");
    const auto r = M::validate_model(d);
    expect_one("imported mesh boundaries", r.deferred,
               Want{"boundary.empty_selection", "/boundaries",
                    "the named boundaries 'bottom', 'left', 'right' and 'top' are resolved when the imported mesh 'duct.msh' is read",
                    "This validation reads the structure of the document only. The groups of the mesh's info file give the "
                    "entities that each selector names; a boundary that selects nothing is refused then.",
                    J(R"({"boundaries": ["bottom", "left", "right", "top"], "file": "duct.msh"})")});
    expect_one("imported mesh partition", r.deferred,
               Want{"physics.boundary_uncovered", "/physics/heat/boundary_conditions",
                    "physics 'heat': whether its boundary conditions cover the boundary of its domain exactly once is checked when "
                    "the imported mesh 'duct.msh' is read",
                    partition_hint("heat"), nullptr});
  }
  {
    // an instance with a grammar finding has no deferred finding of its own (one defect, one finding)
    json d = cavity();
    d["physics"]["heat"]["boundary_conditions"]["hot"]["kind"] = "nonsense";
    const auto r = M::validate_model(d);
    const auto part = with_code(r.deferred, "physics.boundary_uncovered");
    CHECK(part.size() == 1 && part[0].where() == "/physics/flow/boundary_conditions",
          "a physics with a grammar error: only the other physics has a partition finding, got " + codes(r.deferred));
  }
}

// Section 10.3 with section 14.2: the force rule of the Navier-Stokes rows needs the condition on the boundary.  On a
// boundary that is not a key of the physics' conditions (it is part of "remaining", or of another boundary) the rule
// is deferred, not skipped.
void check_force_rule_on_an_unkeyed_boundary() {
  json d = channel();
  d["boundaries"]["upper"] = J(R"({"of": "channel", "where": {"y": "H"}})");
  d["outputs"]["F_upper"] = J(R"({"kind": "torque", "on": "upper"})");
  const auto r = M::validate_model(d);
  CHECK(r.errors.empty(), "force on an unkeyed boundary: no error, got " + codes(r.errors));
  expect_one("force on an unkeyed boundary", r.deferred,
             Want{"output.force_needs_stress_form", "/outputs/F_upper",
                  "output 'F_upper': whether a torque on 'upper' is available with the Laplacian viscous form is checked when the "
                  "boundaries are resolved",
                  "'upper' is not a key of the boundary conditions of 'flow', so the conditions on it are not known before that. "
                  "With the Laplacian form the Navier-Stokes engine returns the pseudo-traction mu du/dn - p n, which is the force "
                  "only on walls at rest: the output is refused then if 'upper' has any other part. With \"viscous_form\": "
                  "\"stress\" it is available on every boundary.",
                  J(R"({"boundary": "upper", "physics": "flow", "route": "navier_stokes_steady_planar", "accepted_on": "wall boundaries at rest"})")});
  // the channel's own force, on a key that is a wall at rest, defers nothing
  CHECK(with_code(M::validate_model(channel()).deferred, "output.force_needs_stress_form").empty(),
        "the channel's force on 'walls' is decided now");
  // the stress form and the Stokes engine have a physical reaction on every boundary: nothing to defer
  json stress = d;
  stress["physics"]["flow"]["options"]["viscous_form"] = "stress";
  CHECK(with_code(M::validate_model(stress).deferred, "output.force_needs_stress_form").empty(), "the stress form defers no force rule");
  json stokes = d;
  stokes["physics"]["flow"]["options"]["equation"] = "stokes";
  CHECK(with_code(M::validate_model(stokes).deferred, "output.force_needs_stress_form").empty(), "the Stokes row defers no force rule");
}

// Section 10.3 (the Boussinesq row: "no force output on the row, stated in its limits") and the registry's limits of
// the one-way row: a force or torque of the flow is refused there.
void check_force_on_rows_without_a_reaction() {
  {
    json d = cavity();
    d["outputs"]["drag"] = J(R"({"kind": "force", "on": "hot"})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "force on the Boussinesq row: one error, got " + codes(r.errors));
    expect_one("force on the Boussinesq row", r.errors,
               Want{"output.force_not_offered", "/outputs/drag",
                    "output 'drag': a force on 'hot' is not offered on the route 'boussinesq_planar'",
                    "The engine of this route (BoussinesqSolver) exposes no reaction, so the route has no force and no torque "
                    "output. Remove the output, or compute it in a document that solves 'flow' alone: the Stokes and Navier-Stokes "
                    "routes have force outputs.",
                    J(R"({"output": "drag", "kind": "force", "boundary": "hot", "physics": "flow", "route": "boussinesq_planar"})")});
    // one defect, one finding: the Navier-Stokes force rule of the flow's own row is not reported beside it
    CHECK(with_code(r.deferred, "output.force_needs_stress_form").empty(), "force on the Boussinesq row: no deferred force rule");
    const M::Route* row = M::registry().route("boussinesq_planar");
    CHECK(row && row->force_reaction.empty(), "the registry gives the Boussinesq row no force reaction");
  }
  {
    json d = one_way_cavity();
    d["outputs"]["twist"] = J(R"({"kind": "torque", "on": "cold", "physics": "flow"})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "torque on the one-way row: one error, got " + codes(r.errors));
    expect_one("torque on the one-way row", r.errors,
               Want{"output.force_not_offered", "/outputs/twist",
                    "output 'twist': a torque on 'cold' is not offered on the route 'flow_then_heat_planar'", "",
                    J(R"({"output": "twist", "kind": "torque", "boundary": "cold", "physics": "flow", "route": "flow_then_heat_planar"})")});
  }
}

// ================================================================================================ collection
// A grammar finding in an entry of "couplings" takes the two ends of that entry out of the route-level checks,
// and no other instance.  The document has three independent instances on disjoint regions; each of 'flow' and 'cell'
// has one route-level error (an axisymmetric slip, section 8.3; every electrode galvanostatic).
json three_independent(const json& couplings) {
  json d = J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "three", "symmetry": "axisymmetric", "parameters": {"R": 0.01, "Lp": 0.05},
    "materials": {"water": {"density": 998.2, "fluid": {"mu_Pa_s": 0.001002}}, "bath": {"electrolyte": {"kappa_S_m": 1.0}},
                  "air": {"density": 1.2, "thermal": {"k_W_mK": 0.026, "cp_J_kgK": 1005.0}}},
    "regions": [{"name": "pipe", "shape": {"rect": {"r": [0, "R"], "z": [0, "Lp"]}}, "material": "water"},
                {"name": "cellr", "shape": {"rect": {"r": [0, "R"], "z": ["Lp", "2*Lp"]}}, "material": "bath"},
                {"name": "box", "shape": {"rect": {"r": [0, "R"], "z": ["2*Lp", "3*Lp"]}}, "material": "air"}],
    "boundaries": {"inlet": {"of": "pipe", "where": {"z": 0}}, "top": {"of": "cellr", "where": {"z": "2*Lp"}},
                   "side": {"of": "pipe", "where": {"r": "R"}}, "cside": {"of": "cellr", "where": {"r": "R"}}},
    "reactions": {"electrode": {"plate": {"kind": "linear", "i0_A_m2": 10.0, "E_eq_V": 0.0, "alpha_a": 0.5, "alpha_c": 0.5}}},
    "physics": {
      "flow": {"type": "fluid.incompressible", "domains": ["pipe"], "options": {"equation": "stokes"},
               "boundary_conditions": {"inlet": {"kind": "inlet", "velocity_m_s": [0, 1, 0]}, "side": {"kind": "slip"},
                                       "remaining": {"kind": "wall"}}},
      "cell": {"type": "electrochem.current_distribution", "domains": ["cellr"],
               "boundary_conditions": {"top": {"kind": "electrode", "reactions": ["plate"], "current_A": 0.1},
                                       "cside": {"kind": "electrode", "reactions": ["plate"], "current_A": -0.1},
                                       "remaining": {"kind": "insulation"}}},
      "heat": {"type": "heat.transfer", "domains": ["box"], "boundary_conditions": {"remaining": {"kind": "insulation"}}}
    },
    "mesh": {"size": 0.002}
  })j");
  if (!couplings.is_null()) d["couplings"] = couplings;
  return d;
}
void check_a_bad_coupling_entry_excludes_its_ends_only() {
  const auto routes_of = [](const M::ValidationReport& r) {
    std::string s;
    for (const auto& e : r.unavailable) s += details_of(e).value("route", std::string("?")) + " ";
    return s;
  };
  {
    // the reference: no coupling, every instance is checked
    const auto r = M::validate_model(three_independent(nullptr));
    CHECK(r.errors.size() == 2, "three instances: two route-level errors, got " + codes(r.errors));
    CHECK(routes_of(r) == "current_distribution_axisymmetric stokes_axisymmetric heat_axisymmetric ", "three instances: three routes, got " + routes_of(r));
  }
  {
    // an entry from 'heat' to a name that is no instance: 'heat' is left out, 'flow' and 'cell' keep their findings
    const auto r = M::validate_model(three_independent(J(R"([{"term": "advection", "from": "heat", "to": "ghost"}])")));
    CHECK(r.errors.size() == 3, "a bad coupling of heat: its grammar error and the two route-level errors, got " + codes(r.errors));
    expect_one("a bad coupling of heat", r.errors, Want{"coupling.unknown_physics", "/couplings/0/to", "unknown physics 'ghost'", "", nullptr});
    expect_one("a bad coupling of heat", r.errors, Want{"fluid.condition_not_offered", "/physics/flow/boundary_conditions/side/kind", "", "", nullptr});
    expect_one("a bad coupling of heat", r.errors, Want{"electrochem.potential_level_free", "/physics/cell/boundary_conditions", "", "", nullptr});
    CHECK(routes_of(r) == "current_distribution_axisymmetric stokes_axisymmetric ", "a bad coupling of heat: the routes of cell and flow, got " + routes_of(r));
    const auto part = with_code(r.deferred, "physics.boundary_uncovered");
    CHECK(part.size() == 2, "a bad coupling of heat: no partition finding for heat, got " + codes(part));
  }
  {
    // a name within two edits of an id is taken for that instance: 'flow' is left out with 'heat'
    const auto r = M::validate_model(three_independent(J(R"([{"term": "advection", "from": "flo", "to": "heat"}])")));
    CHECK(r.errors.size() == 2, "a misspelt end: the grammar error and the error of cell, got " + codes(r.errors));
    expect_one("a misspelt end", r.errors, Want{"electrochem.potential_level_free", "", "", "", nullptr});
    CHECK(routes_of(r) == "current_distribution_axisymmetric ", "a misspelt end: the route of cell only, got " + routes_of(r));
  }
  {
    // a term between types it does not connect: both ends are read and left out
    const auto r = M::validate_model(three_independent(J(R"([{"term": "advection", "from": "cell", "to": "heat"}])")));
    expect_one("a mismatched term", r.errors, Want{"coupling.type_mismatch", "", "", "", nullptr});
    expect_one("a mismatched term", r.errors, Want{"fluid.condition_not_offered", "", "", "", nullptr});
    CHECK(r.errors.size() == 2 && routes_of(r) == "stokes_axisymmetric ", "a mismatched term: flow alone is checked, got " + codes(r.errors) + routes_of(r));
  }
  {
    // an end that cannot be read (not a string): nothing says which instances the entry meant, and none is checked
    const auto r = M::validate_model(three_independent(J(R"([{"term": "advection", "from": 3, "to": "heat"}])")));
    CHECK(r.errors.size() == 1 && r.errors[0].where().rfind("/couplings/0", 0) == 0, "an unreadable end: the grammar error alone, got " + codes(r.errors));
    CHECK(r.unavailable.empty(), "an unreadable end: no route is reported, got " + routes_of(r));
  }
}

// The pointers of the route-level findings are positions in the document as written: an entry of "couplings" or a step
// of "study" that the grammar dropped does not shift the entries after it.
void check_pointers_after_a_dropped_entry() {
  {
    // entry 0 names two instances that do not exist (so no instance is left out); gravity is not vertical in entry 2
    json d = cavity();
    d["couplings"] = J(R"j([{"term": "advection", "from": "nobody", "to": "nowhere"},
                             {"term": "advection", "from": "flow", "to": "heat"},
                             {"term": "buoyancy", "from": "heat", "to": "flow", "T_ref_K": 298.15, "gravity_m_s2": [1.0, -9.81]}])j");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 2, "dropped coupling entry: the grammar error and the gravity, got " + codes(r.errors));
    expect_one("dropped coupling entry", r.errors, Want{"coupling.unknown_physics", "/couplings/0/from", "", "", nullptr});
    expect_one("dropped coupling entry", r.errors, Want{"fluid.gravity_direction", "/couplings/2/gravity_m_s2", "", "", nullptr});
    Strs beta;
    for (const auto& w : with_code(r.warnings, "model.beta_feature"))
      if (w.where().rfind("/couplings/", 0) == 0) beta.push_back(w.where());
    CHECK(beta == (Strs{"/couplings/1/term", "/couplings/2/term"}), "dropped coupling entry: the beta warnings of the terms are at entries 1 and 2");
    if (r.steps.size() == 1) {
      std::vector<int> at;
      for (const auto& e : r.steps[0].graph.edges) at.push_back(e.coupling);
      CHECK(at == (std::vector<int>{1, 2}), "dropped coupling entry: the edges are entries 1 and 2");
      CHECK(r.steps[0].routes.size() == 1 && r.steps[0].routes[0].route == "boussinesq_planar", "dropped coupling entry: the group keeps its row");
    } else {
      fail("dropped coupling entry: one step");
    }
  }
  {
    // the same for a term that the row cannot have beside a prescribed velocity: details.coupling is the entry
    json d = one_way_cavity();
    d["couplings"] = J(R"j([{"term": "advection", "from": "nobody", "to": "nowhere"}, {"term": "advection", "from": "flow", "to": "heat"}])j");
    d["physics"]["heat"]["sources"] = J(R"j([{"kind": "prescribed_advection", "velocity_m_s": [0.001, 0]}])j");
    const auto r = M::validate_model(d);
    const auto e = with_code(r.errors, "heat.two_advections");
    CHECK(e.size() == 1 && details_of(e[0]).value("coupling", -1) == 1, "dropped coupling entry: two_advections names entry 1, got " + codes(r.errors));
  }
  {
    // step 0 has a reserved type and is dropped; the strategy of step 1 is not offered on its row
    json d = cavity();
    d["study"] = J(R"j([{"id": "broken", "type": "eigen", "physics": ["flow"]},
                         {"id": "steady", "type": "stationary", "physics": ["flow", "heat"], "coupling": {"strategy": "gauss_seidel"}}])j");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 2, "dropped step: the reserved type and the strategy, got " + codes(r.errors));
    expect_one("dropped step", r.errors, Want{"study.type_reserved", "/study/0/type", "", "", nullptr});
    expect_one("dropped step", r.errors, Want{"study.strategy_not_offered", "/study/1/coupling/strategy", "", "", nullptr});
  }
  {
    // and a ramp in step 1 that does not end at the document's value
    json d = plate();
    d["study"] = J(R"j([{"id": "broken", "type": "frequency"},
                         {"id": "s", "type": "stationary",
                          "continuation": {"ramp": "/physics/cell/boundary_conditions/anode/phi_s_V", "values": [0.01, 0.02]}}])j");
    const auto r = M::validate_model(d);
    expect_one("dropped step, ramp", r.errors, Want{"study.ramp_end_mismatch", "/study/1/continuation/values/1", "", "", nullptr});
  }
}

// ================================================================================================ study, continued
// Section 3.8: a field is rampable "for the route", so the ramp names a field of an instance that its own step solves.
void check_ramp_of_a_physics_the_step_does_not_solve() {
  json d = plate();
  d["materials"]["water"] = J(R"({"density": 998.2, "fluid": {"mu_Pa_s": 0.001}})");
  d["regions"].push_back(J(R"({"name": "duct", "shape": {"rect": {"x": ["gap", "2*gap"], "y": [0, "Hc"]}}, "material": "water"})"));
  d["physics"]["flow"] = J(R"({"type": "fluid.incompressible", "domains": ["duct"], "options": {"equation": "stokes"},
                               "boundary_conditions": {"remaining": {"kind": "wall"}}})");
  const char* pointer = "/physics/cell/boundary_conditions/anode/phi_s_V";
  const json ramp = {{"ramp", pointer}, {"values", J(R"([0.01, "V_cell"])")}};
  {
    // the ramp in the step that solves the cell: accepted
    d["study"] = json::array({{{"id", "cellstep"}, {"type", "stationary"}, {"physics", {"cell"}}, {"continuation", ramp}},
                              {{"id", "flowstep"}, {"type", "stationary"}, {"physics", {"flow"}}}});
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "a ramp in the step of its physics: accepted, got " + codes(r.errors));
  }
  {
    // the same ramp in the step that solves the flow only
    d["study"] = json::array({{{"id", "cellstep"}, {"type", "stationary"}, {"physics", {"cell"}}},
                              {{"id", "flowstep"}, {"type", "stationary"}, {"physics", {"flow"}}, {"continuation", ramp}}});
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "a ramp of a physics of another step: one error, got " + codes(r.errors));
    json details = {{"ramp", pointer}, {"step", "flowstep"}, {"physics", "cell"}};
    details["rampable"] = J(R"(["/physics/cell/boundary_conditions/anode/phi_s_V", "/physics/cell/boundary_conditions/cathode/phi_s_V"])");
    expect_one("a ramp of a physics of another step", r.errors,
               Want{"study.field_not_rampable", "/study/1/continuation/ramp",
                    "'/physics/cell/boundary_conditions/anode/phi_s_V' is not a rampable field of step 'flowstep', which does not "
                    "solve physics 'cell'",
                    "A ramp steps a field of a physics that its own step solves, each stage starting from the previous solution. "
                    "Move the continuation to a step that solves 'cell', or add 'cell' to the physics of step 'flowstep'.",
                    details});
  }
  {
    // the cell in no step: the pointer is not a rampable field of the document
    d["study"] = json::array({{{"id", "flowstep"}, {"type", "stationary"}, {"physics", {"flow"}}, {"continuation", ramp}}});
    const auto r = M::validate_model(d);
    expect_one("a ramp of a physics of no step", r.errors,
               Want{"study.field_not_rampable", "/study/0/continuation/ramp",
                    "'/physics/cell/boundary_conditions/anode/phi_s_V' is not a rampable field", "", nullptr});
  }
}

// Section 3.8: "A two-way group never runs as one unlabelled sweep", also when the document writes one_way.
void check_one_way_on_a_two_way_group() {
  json d = busbar();
  d["study"][0]["coupling"] = J(R"({"strategy": "one_way"})");
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "busbar with one_way: one error, got " + codes(r.errors));
  expect_one("busbar with one_way", r.errors,
             Want{"study.strategy_not_offered", "/study/0/coupling/strategy",
                  "the coupling strategy 'one_way' is not offered for the two-way group 'current' and 'heat' of step 'steady'",
                  "A two-way group is never run as one unlabelled sweep. Write gauss_seidel, aitken or monolithic (no route lists "
                  "the strategies of this group yet).",
                  J(R"({"strategy": "one_way", "step": "steady", "group": ["current", "heat"], "offered": []})")});
  // on a row, the refusal is the row's (one finding)
  json c = cavity();
  c["study"][0]["coupling"]["strategy"] = "one_way";
  const auto r2 = M::validate_model(c);
  CHECK(r2.errors.size() == 1, "Boussinesq with one_way: one error, got " + codes(r2.errors));
  expect_one("Boussinesq with one_way", r2.errors,
             Want{"study.strategy_not_offered", "/study/0/coupling/strategy",
                  "the coupling strategy 'one_way' is not offered on the route 'boussinesq_planar'; this route offers: monolithic", "",
                  nullptr});
  // one_way on an acyclic graph stays valid
  json o = one_way_cavity();
  o["study"][0]["coupling"] = J(R"({"strategy": "one_way"})");
  CHECK(M::validate_model(o).errors.empty(), "one_way written on an acyclic graph: accepted");
}

// Section 3.8: a scale multiplies a coefficient of the flow engines (viscosity_scales, buoyancy_scales), so the step
// has that coefficient.
void check_continuation_scale_has_its_coefficient() {
  {
    json d = channel();
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "continuation": {"scale": "buoyancy", "factors": [0.1, 1.0]}}])j");
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "a buoyancy scale without the term: one error, got " + codes(r.errors));
    expect_one("a buoyancy scale without the term", r.errors,
               Want{"study.scale_not_applicable", "/study/0/continuation/scale",
                    "the continuation scale 'buoyancy' has nothing to scale in step 's': no buoyancy term couples two physics of the "
                    "step",
                    "A scale multiplies one coefficient of the flow in stages: \"viscosity\" the viscosity of a "
                    "fluid.incompressible physics, \"buoyancy\" the buoyancy term. Add the buoyancy term to \"couplings\" and both "
                    "of its physics to the step, or remove the continuation.",
                    J(R"({"scale": "buoyancy", "step": "s", "needs": "coupling term buoyancy"})")});
    d["study"][0]["continuation"]["scale"] = "viscosity";
    CHECK(M::validate_model(d).errors.empty(), "a viscosity scale on a flow: accepted");
  }
  {
    json d = plate();
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "continuation": {"scale": "viscosity", "factors": [10, 1]}}])j");
    const auto r = M::validate_model(d);
    expect_one("a viscosity scale without a flow", r.errors,
               Want{"study.scale_not_applicable", "/study/0/continuation/scale",
                    "the continuation scale 'viscosity' has nothing to scale in step 's': the step solves no fluid.incompressible "
                    "physics",
                    "A scale multiplies one coefficient of the flow in stages: \"viscosity\" the viscosity of a "
                    "fluid.incompressible physics, \"buoyancy\" the buoyancy term. Add the flow to the physics of the step, or "
                    "remove the continuation.",
                    J(R"({"scale": "viscosity", "step": "s", "needs": "physics type fluid.incompressible"})")});
  }
  // the cavity of section 4.2 scales the buoyancy of its own term
  CHECK(with_code(M::validate_model(cavity()).errors, "study.scale_not_applicable").empty(), "the cavity's buoyancy scale is accepted");
  {
    // the buoyancy term in the document, but the step solves the flow alone
    json d = cavity();
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "physics": ["flow"], "continuation": {"scale": "buoyancy", "factors": [0.1, 1.0]}}])j");
    CHECK(with_code(M::validate_model(d).errors, "study.scale_not_applicable").size() == 1, "a buoyancy scale in a step without the heat: refused");
  }
}

// ================================================================================================ "in" of a source
// Section 3.7 ("checked to be a subset"), applied to the "in" of a source: it names regions of its instance.
void check_source_in_stays_inside_the_instance() {
  json d = slab();
  d["materials"]["rock"] = J(R"({"density": 2000, "thermal": {"k_W_mK": 2.0, "cp_J_kgK": 800.0}})");
  d["regions"].push_back(J(R"({"name": "ground", "shape": {"rect": {"x": [0, 1], "y": [-1, 0]}}, "material": "rock"})"));
  d["domains"] = J(R"({"all": ["slab", "ground"]})");
  d["boundaries"].erase("bottom");
  d["physics"]["heat"]["boundary_conditions"].erase("bottom");
  d["physics"]["heat"]["boundary_conditions"]["remaining"] = J(R"({"kind": "insulation"})");
  d["physics"]["heat"]["domains"] = J(R"(["slab"])");
  d["physics"]["heat"]["sources"][0]["in"] = "all";
  {
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, "a source beyond its instance: one error, got " + codes(r.errors));
    expect_one("a source beyond its instance", r.errors,
               Want{"physics.source_outside_domain", "/physics/heat/sources/0/in",
                    "physics 'heat': \"in\": \"all\" of its prescribed_advection source is not inside the domain of the physics",
                    "'heat' is on 'slab', and 'all' also names 'ground'. Remove \"in\" for a source over the whole instance, or name "
                    "regions of its domain.",
                    J(R"({"physics": "heat", "source": 0, "kind": "prescribed_advection", "in": ["slab", "ground"], "regions": ["slab"], "outside": ["ground"]})")});
    // the source covers the whole instance: it is not a partial prescribed velocity
    CHECK(with_code(r.errors, "heat.prescribed_advection_partial").empty(), "a source beyond its instance is not a partial velocity");
  }
  {
    // "in" equal to the instance: accepted
    json ok = d;
    ok["physics"]["heat"]["sources"][0]["in"] = "slab";
    CHECK(M::validate_model(ok).errors.empty(), "a source on the whole instance, written with \"in\": accepted");
  }
  {
    // the instance on both regions and the source on one: the partial velocity of section 8.3, as before
    json part = d;
    part["physics"]["heat"]["domains"] = J(R"(["slab", "ground"])");
    part["physics"]["heat"]["sources"][0]["in"] = "slab";
    const auto r = M::validate_model(part);
    CHECK(with_code(r.errors, "heat.prescribed_advection_partial").size() == 1 && with_code(r.errors, "physics.source_outside_domain").empty(),
          "a source on part of its instance: heat.prescribed_advection_partial, got " + codes(r.errors));
  }
  {
    // a heat source follows the same rule
    json q = d;
    q["physics"]["heat"]["sources"] = J(R"([{"kind": "heat_source", "in": "ground", "q_W_m3": 100.0}])");
    const auto r = M::validate_model(q);
    expect_one("a heat source beyond its instance", r.errors,
               Want{"physics.source_outside_domain", "/physics/heat/sources/0/in",
                    "physics 'heat': \"in\": \"ground\" of its heat_source source is not inside the domain of the physics", "", nullptr});
  }
}


// ================================================================================================ gaps in the checks above
// A misspelt end of a coupling may be any instance within two edits that the term can have at that end; an instance of
// another type cannot be the one that was meant, so its checks are kept.
void check_a_misspelt_end_leaves_out_only_instances_that_could_be_meant() {
  const auto routes_of = [](const M::ValidationReport& r) {
    std::string s;
    for (const auto& e : r.unavailable) s += details_of(e).value("route", std::string("?")) + " ";
    return s;
  };
  // 'cel' is within one edit of 'cell', which is an electrochemistry instance and never the consumer of advection
  const auto r = M::validate_model(three_independent(J(R"([{"term": "advection", "from": "flow", "to": "cel"}])")));
  CHECK(r.errors.size() == 2, "a misspelt end that no instance of that type matches: the grammar error and the error of cell, got " + codes(r.errors));
  expect_one("a misspelt end of the wrong type", r.errors, Want{"coupling.unknown_physics", "/couplings/0/to", "", "", nullptr});
  expect_one("a misspelt end of the wrong type", r.errors, Want{"electrochem.potential_level_free", "/physics/cell/boundary_conditions", "", "", nullptr});
  CHECK(routes_of(r) == "current_distribution_axisymmetric heat_axisymmetric ",
        "a misspelt end of the wrong type: the routes of cell and heat (flow is the valid end of the entry), got " + routes_of(r));
  // 'hea' is within one edit of 'heat', a consumer of advection: heat is left out, and flow and cell are checked
  const auto h = M::validate_model(three_independent(J(R"([{"term": "advection", "from": "flow", "to": "hea"}])")));
  CHECK(h.errors.size() == 2, "a misspelt end of heat: the grammar error and the error of cell, got " + codes(h.errors));
  // the producer end is read the same way: 'hea' cannot be the producer of advection, so heat is checked
  const auto p = M::validate_model(three_independent(J(R"([{"term": "advection", "from": "hea", "to": "heat"}])")));
  CHECK(with_code(p.errors, "coupling.unknown_physics").size() == 1, "a misspelt producer: the grammar error, got " + codes(p.errors));
}

// An instance without "boundary_conditions" is accepted by the grammar, and the partition rule is deferred for it as for
// the others.
void check_partition_is_deferred_for_an_instance_without_conditions() {
  json d = cavity();
  d["physics"]["heat"].erase("boundary_conditions");
  const auto r = M::validate_model(d);
  const auto part = with_code(r.deferred, "physics.boundary_uncovered");
  bool heat = false;
  for (const auto& e : part) heat = heat || e.where() == "/physics/heat/boundary_conditions";
  CHECK(heat, "no conditions: the partition rule of heat is deferred, got " + codes(r.deferred));
  for (const auto& e : part)
    if (e.where() == "/physics/heat/boundary_conditions") CHECK(details_of(e)["conditions"] == J("[]"), "no conditions: the list is empty: " + e.details());
}

// A force on a boundary that is a key of another flow is that flow's, and the rule of the first flow does not see it.
void check_a_force_on_the_boundary_of_another_flow_is_that_flows() {
  json d = channel();
  d["regions"].push_back(J(R"({"name": "channel2", "shape": {"rect": {"x": [0, "L"], "y": ["2*H", "3*H"]}}, "material": "water"})"));
  d["boundaries"]["inlet2"] = J(R"({"of": "channel2", "where": {"x": 0}})");
  d["physics"]["flow"]["domains"] = J(R"(["channel"])");
  d["physics"]["flow2"] = J(R"({"type": "fluid.incompressible", "domains": ["channel2"], "options": {"equation": "stokes"},
                                 "boundary_conditions": {"inlet2": {"kind": "inlet", "velocity_m_s": [0.01, 0]}, "remaining": {"kind": "wall"}}})");
  d["outputs"]["F2"] = J(R"({"kind": "force", "on": "inlet2"})");
  const auto r = M::validate_model(d);
  CHECK(r.errors.empty(), "a force on the boundary of the second flow: no error, got " + codes(r.errors));
  CHECK(with_code(r.deferred, "output.force_needs_stress_form").empty(),
        "a force on the boundary of the second flow (Stokes) is not deferred to the first flow, got " + codes(r.deferred));
}

// "in" of a source is checked against the domain of its instance only when the instance has no finding: one that has
// one is reported by that finding alone.
void check_in_is_not_checked_in_a_group_with_a_finding() {
  {
    json d = slab();
    d["materials"]["rock"] = J(R"({"density": 2000, "thermal": {"k_W_mK": 2.0, "cp_J_kgK": 800.0}})");
    d["regions"].push_back(J(R"({"name": "ground", "shape": {"rect": {"x": [0, 1], "y": [-1, 0]}}, "material": "rock"})"));
    d["boundaries"].erase("bottom");
    d["physics"]["heat"]["boundary_conditions"].erase("bottom");
    d["physics"]["heat"]["boundary_conditions"]["remaining"] = J(R"({"kind": "insulation"})");
    d["physics"]["heat"]["domains"] = J(R"(["slab"])");
    d["physics"]["heat"]["sources"] = J(R"([{"kind": "heat_source", "in": "ground", "q_W_m3": 100.0}])");
    CHECK(with_code(M::validate_model(d).errors, "physics.source_outside_domain").size() == 1, "a source beyond its instance: refused");
    d["physics"]["heat"]["boundary_conditions"]["remaining"]["kind"] = "nonsense";
    const auto r = M::validate_model(d);
    CHECK(!r.errors.empty() && with_code(r.errors, "physics.source_outside_domain").empty(),
          "a source beyond an instance with a finding: not refused again, got " + codes(r.errors));
  }
}

// one_way on a two-way group that has a finding is refused by that finding alone.
void check_one_way_is_not_refused_for_a_group_with_a_finding() {
  json d = busbar();
  d["study"][0]["coupling"] = J(R"({"strategy": "one_way"})");
  CHECK(with_code(M::validate_model(d).errors, "study.strategy_not_offered").size() == 1, "one_way on the busbar: refused");
  json& conditions = d["physics"]["heat"]["boundary_conditions"];
  conditions.begin().value()["kind"] = "nonsense";
  const auto r = M::validate_model(d);
  CHECK(!r.errors.empty() && with_code(r.errors, "study.strategy_not_offered").empty(),
        "one_way on a two-way group with a finding: not refused again, got " + codes(r.errors));
}

// A force output on the one-way row is refused for the row, and the viscosity that differs between the regions (which only
// a force output with the Laplacian form makes a defect) is not refused as well.
void check_a_force_on_a_row_without_reaction_does_not_ask_for_one_viscosity() {
  json d = cavity();
  d["materials"]["air2"] = d["materials"]["air"];
  d["materials"]["air2"]["fluid"]["mu_Pa_s"] = "2*mu";
  d["regions"][0]["shape"] = J(R"({"rect": {"x": [0, "Lc/2"], "y": [0, "Lc"]}})");
  d["regions"].push_back(J(R"({"name": "cavity2", "shape": {"rect": {"x": ["Lc/2", "Lc"], "y": [0, "Lc"]}}, "material": "air2"})"));
  d["boundaries"] = J(R"({"hot": {"of": "cavity", "where": {"x": 0}}, "cold": {"of": "cavity2", "where": {"x": "Lc"}}})");
  d["physics"]["flow"]["options"]["viscous_form"] = "laplacian";
  d["couplings"] = J(R"([{"term": "advection", "from": "flow", "to": "heat"}])");
  d["outputs"] = J(R"({"drag": {"kind": "force", "on": "hot"}})");
  d.erase("study");
  const auto r = M::validate_model(d);
  CHECK(with_code(r.errors, "output.force_not_offered").size() == 1, "the force on the one-way row is refused, got " + codes(r.errors));
  CHECK(r.errors.size() == 1, "the force on the one-way row is the only error (no second finding about the viscosity), got " + codes(r.errors));
  // without the force output, the viscosity that differs is no defect of the one-way row
  d["outputs"] = json::object();
  CHECK(M::validate_model(d).errors.empty(), "the one-way row without the force output: accepted");
}

// The note ties the scales to the schedules the flow engines have; of the routes, only the steady Navier-Stokes engine has
// a viscosity schedule.
void check_viscosity_scale_needs_the_steady_navier_stokes_engine() {
  const char* step = R"j([{"id": "s", "type": "stationary", "continuation": {"scale": "viscosity", "factors": [10, 1]}}])j";
  const auto refused = [&](const std::string& what, const json& d) {
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1, what + ": one error, got " + codes(r.errors));
    expect_one(what, r.errors,
               Want{"study.scale_not_applicable", "/study/0/continuation/scale",
                    "the continuation scale 'viscosity' has nothing to scale in step 's': no fluid.incompressible physics of the step has a "
                    "viscosity schedule",
                    "", J(R"({"scale": "viscosity", "step": "s", "needs": "a stationary fluid.incompressible physics with equation navier_stokes and no buoyancy term"})")});
  };
  {
    json d = channel();
    d["study"] = J(step);
    CHECK(M::validate_model(d).errors.empty(), "a viscosity scale on the steady Navier-Stokes row: accepted");
    d["physics"]["flow"]["options"]["equation"] = "stokes";
    refused("a viscosity scale on the Stokes row", d);
  }
  {
    json d = cavity();
    d["study"] = J(step);
    refused("a viscosity scale on the Boussinesq row", d);
    d["study"][0]["continuation"]["scale"] = "buoyancy";
    CHECK(with_code(M::validate_model(d).errors, "study.scale_not_applicable").empty(), "a buoyancy scale on the Boussinesq row: accepted");
  }
  {
    json d = channel();
    d["study"] = J(R"j([{"id": "s", "type": "transient", "time": {"t_end_s": 1.0}, "continuation": {"scale": "viscosity", "factors": [10, 1]}}])j");
    const auto r = M::validate_model(d);
    CHECK(!with_code(r.errors, "study.scale_not_applicable").empty(), "a viscosity scale in a transient step: refused, got " + codes(r.errors));
  }
}

// A force or a torque is the reaction of a flow.
void check_a_force_or_torque_needs_a_flow() {
  for (const char* kind : {"force", "torque"}) {
    {
      json d = cavity();
      d["outputs"] = json::object();
      d["outputs"]["F"] = json{{"kind", kind}, {"on", "hot"}, {"physics", "heat"}};
      const auto r = M::validate_model(d);
      expect_one(std::string("a ") + kind + " of the heat", r.errors,
                 Want{"output.force_not_offered", "/outputs/F",
                      std::string("output 'F': a ") + kind + " on 'hot' needs a flow, and the physics 'heat' is a heat.transfer, not a fluid.incompressible",
                      "A force or a torque is the reaction of a flow solve on a boundary. Write the id of a fluid.incompressible physics in \"physics\", or remove the output.",
                      json{{"output", "F"}, {"kind", kind}, {"boundary", "hot"}, {"physics", "heat"}, {"type", "heat.transfer"}}});
      CHECK(r.errors.size() == 1, std::string("a ") + kind + " of the heat: one error, got " + codes(r.errors));
    }
    {
      json d = plate();
      d["outputs"] = json::object();
      d["outputs"]["F"] = json{{"kind", kind}, {"on", d["boundaries"].begin().key()}};
      const auto r = M::validate_model(d);
      expect_one(std::string("a ") + kind + " in a document without a flow", r.errors,
                 Want{"output.force_not_offered", "/outputs/F", "", "",
                      json{{"output", "F"}, {"kind", kind}, {"boundary", d["boundaries"].begin().key()}, {"physics", ""}, {"type", ""}}});
    }
  }
  // the flows keep their own rules: a force on the channel's walls is accepted
  CHECK(M::validate_model(channel()).errors.empty(), "the channel's force on its walls: accepted");
}

// A physics that has a finding of its own is reported by that finding alone: a force or torque does not add a finding about
// a flow that is not there.
void check_force_owners_do_not_cascade_from_a_physics_with_a_finding() {
  const auto only = [&](const std::string& what, const json& d, const char* code) {
    const auto r = M::validate_model(d);
    CHECK(r.errors.size() == 1 && r.errors[0].code() == code, what + ": exactly the first error " + code + ", got " + codes(r.errors));
  };
  json named = channel();
  named["physics"]["flow"]["type"] = "fluid-incompressible";
  named["outputs"]["F_walls"]["physics"] = "flow";
  only("a misspelt flow type with a force that names the flow", named, "physics.unknown_type");
  json implied = channel();
  implied["physics"]["flow"]["type"] = "fluid-incompressible";
  only("a misspelt flow type with a force that names no physics", implied, "physics.unknown_type");
  json bad = channel();
  bad["physics"]["flow"] = 5;
  const auto r = M::validate_model(bad);
  CHECK(!r.errors.empty() && with_code(r.errors, "output.force_not_offered").empty(),
        "a physics that is not an object: no finding about the force, got " + codes(r.errors));
  json other = cavity();
  other["outputs"] = json::object();
  other["outputs"]["F"] = json{{"kind", "force"}, {"on", "hot"}, {"physics", "heat"}};
  other["physics"]["heat"]["boundary_conditions"]["hot"]["kind"] = "nonsense";
  const auto o = M::validate_model(other);
  CHECK(with_code(o.errors, "output.force_not_offered").empty(), "a force that names a physics with a finding: only that finding, got " + codes(o.errors));
}

// A boundary that is refused is not named in the deferred resolution, and neither is a member of a cycle of names.
void check_deferred_resolution_leaves_out_refused_boundaries() {
  json d = cavity();
  d["boundaries"]["a"] = "b";
  d["boundaries"]["b"] = "a";
  const auto r = M::validate_model(d);
  CHECK(with_code(r.errors, "boundary.cycle").size() == 1, "a cycle of boundaries: one finding, got " + codes(r.errors));
  const auto res = with_code(r.deferred, "boundary.empty_selection");
  CHECK(res.size() == 1, "a cycle of boundaries: one resolution finding, got " + codes(r.deferred));
  if (res.size() == 1) CHECK(details_of(res[0])["boundaries"] == J(R"(["cold", "hot"])"), "the cycle is not in the list: " + res[0].details());
  // a boundary whose selector is not a selector is not resolved later either
  json e = cavity();
  e["boundaries"]["g"] = J(R"({"of": "cavity", "bogus": 1})");
  const auto s = M::validate_model(e);
  const auto res2 = with_code(s.deferred, "boundary.empty_selection");
  CHECK(res2.size() == 1, "a refused selector: one resolution finding, got " + codes(s.deferred));
  if (res2.size() == 1) CHECK(details_of(res2[0])["boundaries"] == J(R"(["cold", "hot"])"), "the refused selector is not in the list: " + res2[0].details());
}

// An "in" that names nothing of the instance has the finding of the source and no partial-velocity finding, and its hint
// does not say that a region also names itself.
void check_a_source_in_that_is_disjoint_from_its_instance() {
  json d = slab();
  d["materials"]["rock"] = J(R"({"density": 2000, "thermal": {"k_W_mK": 2.0, "cp_J_kgK": 800.0}})");
  d["regions"].push_back(J(R"({"name": "ground", "shape": {"rect": {"x": [0, 1], "y": [-1, 0]}}, "material": "rock"})"));
  d["boundaries"].erase("bottom");
  d["physics"]["heat"]["boundary_conditions"].erase("bottom");
  d["physics"]["heat"]["boundary_conditions"]["remaining"] = J(R"({"kind": "insulation"})");
  d["physics"]["heat"]["domains"] = J(R"(["slab"])");
  d["physics"]["heat"]["sources"][0]["in"] = "ground";
  const auto r = M::validate_model(d);
  CHECK(r.errors.size() == 1, "a source disjoint from its instance: one error, got " + codes(r.errors));
  expect_one("a source disjoint from its instance", r.errors,
             Want{"physics.source_outside_domain", "/physics/heat/sources/0/in", "",
                  "'heat' is on 'slab', and 'ground' is not in it. Remove \"in\" for a source over the whole instance, or name regions of its domain.", nullptr});
  CHECK(with_code(r.errors, "heat.prescribed_advection_partial").empty(), "a source disjoint from its instance is not a partial velocity");
  // a domain whose regions are all outside is named with its regions
  json e = d;
  e["domains"] = J(R"({"below": ["ground"]})");
  e["physics"]["heat"]["sources"][0]["in"] = "below";
  expect_one("a source in a domain that is disjoint from its instance", M::validate_model(e).errors,
             Want{"physics.source_outside_domain", "/physics/heat/sources/0/in", "",
                  "'heat' is on 'slab', and none of the regions of 'below' ('ground') is in it. Remove \"in\" for a source over the whole instance, or name regions of its domain.",
                  nullptr});
}

// ================================================================================================ inventory texts
// Section 4.4's inventory for other documents: the article of the geometry, "it is" for one missing piece, and absent
// types named together as the absent terms are.
void check_inventory_grammar() {
  {
    json d = three_independent(nullptr);
    d["materials"]["water"]["mechanical"] = J(R"({"E_Pa": 1.0e9, "nu": 0.3})");
    d["physics"] = J(R"({"solid": {"type": "solid.linear", "domains": ["pipe"], "boundary_conditions": {"remaining": {"kind": "free"}}}})");
    const auto r = M::validate_model(d);
    CHECK(r.errors.empty(), "an absent type alone: no error, got " + codes(r.errors));
    expect_one("an absent type alone", r.unavailable,
               Want{"route.not_implemented", "/physics", "no route for {solid.linear} on an axisymmetric geometry",
                    "not implemented in this version: 'solid.linear' as a physics instance. The document does not need to change "
                    "when it is. `axmp physics --routes` lists every combination that can be solved.",
                    nullptr});
  }
  {
    // several pieces, as in the busbar of section 4.4, are "they"
    const auto r = M::validate_model(busbar());
    const auto e = with_code(r.unavailable, "route.not_implemented");
    CHECK(e.size() == 1 && e[0].hint().find("when they are.") != std::string::npos, "the busbar's inventory has several pieces: \"they are\"");
  }
}

}  // namespace

int main() {
  run("check_busbar_inventory", check_busbar_inventory);
  run("check_routes_found_are_not_implemented_in_u1d", check_routes_found_are_not_implemented_in_u1d);
  run("check_route_findings_are_collected", check_route_findings_are_collected);
  run("check_reactor_without_study", check_reactor_without_study);
  run("check_flow_condition_not_offered", check_flow_condition_not_offered);
  run("check_heat_condition_not_offered", check_heat_condition_not_offered);
  run("check_property_not_uniform", check_property_not_uniform);
  run("check_heat_data_not_regionwise", check_heat_data_not_regionwise);
  run("check_same_coordinate_expression_is_accepted_on_the_boussinesq_row",
      check_same_coordinate_expression_is_accepted_on_the_boussinesq_row);
  run("check_heat_option_not_offered", check_heat_option_not_offered);
  run("check_flow_option_not_offered", check_flow_option_not_offered);
  run("check_supg_needs_advection", check_supg_needs_advection);
  run("check_prescribed_advection_partial", check_prescribed_advection_partial);
  run("check_two_advections", check_two_advections);
  run("check_domains_must_coincide", check_domains_must_coincide);
  run("check_material_block_missing", check_material_block_missing);
  run("check_domains_overlap", check_domains_overlap);
  run("check_gravity_direction", check_gravity_direction);
  run("check_buoyancy_needs_beta", check_buoyancy_needs_beta);
  run("check_potential_level_free", check_potential_level_free);
  run("check_geometry_order", check_geometry_order);
  run("check_ramps", check_ramps);
  run("check_strategies", check_strategies);
  run("check_laws_on_routes", check_laws_on_routes);
  run("check_force_rule", check_force_rule);
  run("check_warnings", check_warnings);
  run("check_time_in_a_stationary_step", check_time_in_a_stationary_step);
  run("check_failed_material_gives_no_route_finding", check_failed_material_gives_no_route_finding);
  run("check_deferred_boundaries", check_deferred_boundaries);
  run("check_force_rule_on_an_unkeyed_boundary", check_force_rule_on_an_unkeyed_boundary);
  run("check_force_on_rows_without_a_reaction", check_force_on_rows_without_a_reaction);
  run("check_a_bad_coupling_entry_excludes_its_ends_only", check_a_bad_coupling_entry_excludes_its_ends_only);
  run("check_pointers_after_a_dropped_entry", check_pointers_after_a_dropped_entry);
  run("check_ramp_of_a_physics_the_step_does_not_solve", check_ramp_of_a_physics_the_step_does_not_solve);
  run("check_one_way_on_a_two_way_group", check_one_way_on_a_two_way_group);
  run("check_continuation_scale_has_its_coefficient", check_continuation_scale_has_its_coefficient);
  run("check_source_in_stays_inside_the_instance", check_source_in_stays_inside_the_instance);
  run("check_a_misspelt_end_leaves_out_only_instances_that_could_be_meant", check_a_misspelt_end_leaves_out_only_instances_that_could_be_meant);
  run("check_partition_is_deferred_for_an_instance_without_conditions", check_partition_is_deferred_for_an_instance_without_conditions);
  run("check_a_force_on_the_boundary_of_another_flow_is_that_flows", check_a_force_on_the_boundary_of_another_flow_is_that_flows);
  run("check_in_is_not_checked_in_a_group_with_a_finding", check_in_is_not_checked_in_a_group_with_a_finding);
  run("check_one_way_is_not_refused_for_a_group_with_a_finding", check_one_way_is_not_refused_for_a_group_with_a_finding);
  run("check_a_force_on_a_row_without_reaction_does_not_ask_for_one_viscosity", check_a_force_on_a_row_without_reaction_does_not_ask_for_one_viscosity);
  run("check_viscosity_scale_needs_the_steady_navier_stokes_engine", check_viscosity_scale_needs_the_steady_navier_stokes_engine);
  run("check_a_force_or_torque_needs_a_flow", check_a_force_or_torque_needs_a_flow);
  run("check_force_owners_do_not_cascade_from_a_physics_with_a_finding", check_force_owners_do_not_cascade_from_a_physics_with_a_finding);
  run("check_deferred_resolution_leaves_out_refused_boundaries", check_deferred_resolution_leaves_out_refused_boundaries);
  run("check_a_source_in_that_is_disjoint_from_its_instance", check_a_source_in_that_is_disjoint_from_its_instance);
  run("check_inventory_grammar", check_inventory_grammar);
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("test_model_validate: OK\n");
  return 0;
}
