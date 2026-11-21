// The case list of the matrix test and its minimal documents: see cpp/src/model_matrix.hpp.
#include "model_matrix.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "model_errors.hpp"

namespace axmp::model {

using json = nlohmann::json;
using Strs = std::vector<std::string>;

namespace {

constexpr const char* kFluid = "fluid.incompressible";
constexpr const char* kHeat = "heat.transfer";
constexpr const char* kCurrent = "electrochem.current_distribution";

bool has(const Strs& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

[[noreturn]] void table_error(const std::string& what) {
  throw std::logic_error("model matrix: " + what + " (the route table and the engine column of the matrix cases "
                         "disagree; update both)");
}

std::string geometry_of(const Route& r) { return r.geometries.empty() ? std::string() : r.geometries[0]; }

// ------------------------------------------------------------------------------------------------ the engines
// The engine a cell of row `r` about type `t` is handed to.
std::string cell_engine(const Route& r, const std::string& t) {
  const std::string& n = r.name;
  if (n.rfind("stokes_", 0) == 0) return "StokesSolver";
  if (n.rfind("navier_stokes_steady_", 0) == 0) return "NavierStokesSolver";
  if (n.rfind("navier_stokes_transient_", 0) == 0) return "TransientNavierStokes";
  if (n.rfind("boussinesq_", 0) == 0) return "BoussinesqSolver";
  if (n.rfind("heat_", 0) == 0 || n.rfind("flow_then_heat_", 0) == 0) return "transport::ScalarTransport";
  if (n == "current_distribution_3d") return "chem::CurrentDistribution3D";
  if (n.rfind("current_distribution_", 0) == 0) return "chem::CurrentDistribution";
  if (n == "chem_reactor") return "chem reactor problems";
  (void)t;
  table_error("no engine for row '" + n + "'");
}

bool flow_engine(const std::string& e) {
  return e == "StokesSolver" || e == "NavierStokesSolver" || e == "TransientNavierStokes";
}
bool heat_engine(const std::string& e) { return e == "transport::ScalarTransport" || e == "BoussinesqSolver"; }

// Whether the cells of row `r` have an engine column in the matrix test: the planar and axisymmetric rows whose
// engine the test can build (the fluids library's engines in an MPI build, the chemistry engine in every build).  The
// 3D rows are confirmed when they are implemented; the reactor has no mesh.
bool engine_column(const Route& r) {
  const std::string g = geometry_of(r);
  if (g != "planar" && g != "axisymmetric") return false;
  const std::string e = cell_engine(r, "");
  return flow_engine(e) || heat_engine(e) || e == "chem::CurrentDistribution";
}

// The call through which an offered cell is checked: the transport operator checks its conditions in finalize; the
// flow engines and the Boussinesq solver (which finalizes its transport operator inside) in their constructors; the
// chemistry engine accepts a problem when solve converges.
std::string offered_call(const std::string& engine) {
  if (engine == "transport::ScalarTransport") return "finalize";
  if (engine == "chem::CurrentDistribution") return "solve";
  return "constructor";
}

constexpr const char* kTextTotalFlux = "a total-flux datum needs the conservative_flux form";
constexpr const char* kTextDiffusiveFlux = "a diffusive-flux datum on a natural part is refused with conservative_flux";

struct Refusal {
  std::string call, text;
};

// A refused condition kind: where the engine refuses it, and its text.
Refusal kind_refusal(const std::string& engine, const std::string& geometry, const std::string& kind) {
  if (engine == "StokesSolver" && geometry == "axisymmetric") {
    if (kind == "slip" || kind == "symmetry") return {"constructor", "not implemented in axisymmetric geometry"};
    if (kind == "periodic") return {"constructor", "periodic conditions are planar only"};
  }
  if ((engine == "NavierStokesSolver" || engine == "TransientNavierStokes") &&
      (kind == "slip" || kind == "symmetry" || kind == "periodic"))
    return {"constructor", "is not offered with convection"};
  if (heat_engine(engine)) {
    // the transport operator checks a flux kind against the advective form in add_bc; the Boussinesq solver adds the
    // conditions to its transport operator inside its own constructor, so its refusal comes from there
    const char* call = engine == "BoussinesqSolver" ? "constructor" : "add_bc";
    if (kind == "total_flux") return {call, kTextTotalFlux};
    if (kind == "heat_flux") return {call, kTextDiffusiveFlux};
  }
  table_error("no engine refusal known for kind '" + kind + "' on " + engine + " (" + geometry + ")");
}

std::string kind_code(const std::string& type) {
  if (type == kFluid) return codes::fluid_condition_not_offered;
  if (type == kHeat) return codes::heat_condition_not_offered;
  table_error("no refusal code for a condition kind of " + type);
}

// A refused option value.
struct OptionRefusal {
  std::string call, text, code;
};
OptionRefusal option_refusal(const std::string& engine, const std::string& name, const std::string& value) {
  if (engine == "TransientNavierStokes" && name == "convection" && value == "skew")
    return {"constructor", "transient Navier-Stokes: the skew convective form is not offered", codes::fluid_option_not_offered};
  if (engine == "transport::ScalarTransport" && name == "stabilization" && value == "supg")
    return {"finalize", "SUPG needs a velocity", codes::heat_supg_needs_advection};
  if (engine == "BoussinesqSolver" && name == "order" && value == "1")
    return {"constructor", "Boussinesq: the temperature is P2 (the coupled solver refuses P1 T)",
            codes::heat_option_not_offered};
  table_error("no engine refusal known for option " + name + "=" + value + " on " + engine);
}

// The member of the engine's interface that holds one value where the document gives one per region.
std::string slot_member(const std::string& engine, const std::string& property) {
  static const std::map<std::pair<std::string, std::string>, std::string> m = {
      {{"NavierStokesSolver", "density"}, "NavierStokesOptions::rho"},
      {{"TransientNavierStokes", "density"}, "NavierStokesOptions::rho"},
      {{"transport::ScalarTransport", "density"}, "Velocity::scale"},
      {{"transport::ScalarTransport", "thermal.cp_J_kgK"}, "Velocity::scale"},
      {{"transport::ScalarTransport", "thermal.k_W_mK"}, "ScalarTransport::set_diffusivity"},
      {{"transport::ScalarTransport", "heat_source"}, "ScalarTransport::set_source"},
      {{"transport::ScalarTransport", "prescribed_advection.velocity"}, "Velocity::fn"},
      {{"BoussinesqSolver", "density"}, "NavierStokesOptions::rho"},
      {{"BoussinesqSolver", "thermal.cp_J_kgK"}, "BoussinesqOptions::cp"},
      {{"BoussinesqSolver", "fluid.beta_1_K"}, "BoussinesqOptions::beta_g"},
      {{"BoussinesqSolver", "thermal.k_W_mK"}, "BoussinesqOptions::k"},
      {{"BoussinesqSolver", "heat_source"}, "BoussinesqOptions::heat_source"},
  };
  auto it = m.find({engine, property});
  if (it == m.end()) table_error("no engine member known for '" + property + "' on " + engine);
  return it->second;
}

// The slot an engine would need for a value of an option of which it has one value only: the member whose absence
// the matrix test asserts.
std::string absent_member(const std::string& engine, const std::string& option) {
  if (engine == "BoussinesqSolver" && option == "advective_form") return "BoussinesqOptions: no advective form";
  table_error("no engine member known for the forced option '" + option + "' on " + engine);
}

std::string option_code(const std::string& type) {
  if (type == kHeat) return codes::heat_option_not_offered;
  table_error("no refusal code for an option value of " + type);
}

// The two-rank refusal of a one-rank engine.
Refusal rank_refusal(const std::string& engine) {
  if (engine == "transport::ScalarTransport") return {"constructor", "ScalarTransport needs one MPI rank"};
  if (engine == "BoussinesqSolver")
    return {"constructor", "BoussinesqSolver() needs one MPI rank (the Boussinesq solver runs on one rank)"};
  table_error("no two-rank refusal known for " + engine);
}

constexpr const char* kTextGalvanostatic = "no grounded electrode: every electrode is galvanostatic";
constexpr const char* kTextCurvedP1 = "curved edge nodes need order 2 (isoparametric P2)";
constexpr const char* kTextQuad = "ScalarTransport: only triangular meshes are offered";
constexpr const char* kTextSupgNodes = "SUPG on isoparametric (curved) elements is not offered";
constexpr const char* kTextSparse = "boundary attribute 1 has no condition";

const LayerRule* layer_rule(const std::string& name) {
  for (const auto& l : registry().layer_rules)
    if (l.name == name) return &l;
  table_error("unknown layer rule '" + name + "'");
}

const OptionSpec* option_spec(const PhysicsType& t, const std::string& name) {
  for (const auto& o : t.options)
    if (o.name == name) return &o;
  return nullptr;
}

const OptionRule* option_rule(const Route& r, const std::string& type, const std::string& name) {
  auto it = r.options.find(type);
  if (it == r.options.end()) return nullptr;
  for (const auto& o : it->second)
    if (o.name == name) return &o;
  return nullptr;
}

// ------------------------------------------------------------------------------------------------ the cases of a row
struct RowCases {
  const Route& r;
  std::vector<MatrixCase>& out;
  std::string geometry;
  bool column;  // the row has an engine column

  MatrixCase base(const std::string& physics, const std::string& axis, const std::string& value, CellValue cls) const {
    MatrixCase c;
    c.row = r.name;
    c.engine = cell_engine(r, physics);
    c.physics = physics;
    c.axis = axis;
    c.value = value;
    c.cls = cls;
    return c;
  }
  // The meshes of a cell checked against the engine; empty when the row has no engine column or the class asserts
  // nothing of the engine.
  void column_of(MatrixCase& c, Strs meshes) const {
    if (!column) return;
    if (c.cls != CellValue::offered && c.cls != CellValue::engine_refuses) return;
    c.meshes = std::move(meshes);
  }
  // The straight meshes of the chemistry column: the base mesh and the mesh whose P2 edge nodes curve one edge.
  Strs straight_meshes(const std::string& engine) const {
    return engine == "chem::CurrentDistribution" ? Strs{"base", "nodes"} : Strs{"base"};
  }

  void condition_kinds(const std::string& t, const PhysicsType& ty) {
    Strs forms;
    if (const OptionRule* f = option_rule(r, t, "advective_form"); f && f->forced.empty()) forms = f->offered;
    const auto offered_it = r.condition_kinds.find(t);
    auto offered = [&](const std::string& k) {
      return offered_it != r.condition_kinds.end() && has(offered_it->second, k);
    };
    for (const auto& k : ty.condition_kinds) {
      const Strs values = forms.empty() ? Strs{""} : forms;
      for (const auto& f : values) {
        std::string why;  // "" offered, else the reason string of the registry
        if (!f.empty()) {
          auto w = r.not_offered_with.find(t);
          if (w != r.not_offered_with.end()) {
            auto wf = w->second.find("advective_form=" + f);
            if (wf != w->second.end() && wf->second.count(k.name)) why = wf->second.at(k.name);
          }
        }
        if (why.empty() && !offered(k.name)) {
          auto n = r.not_offered.find(t);
          if (n == r.not_offered.end() || !n->second.count(k.name))
            table_error("kind '" + k.name + "' of " + t + " is neither offered nor refused on " + r.name);
          why = n->second.at(k.name);
        }
        const std::string value = f.empty() ? k.name : k.name + "@" + f;
        CellValue cls = CellValue::offered;
        if (why == "engine_refuses") cls = CellValue::engine_refuses;
        else if (why.rfind("not_exposed: ", 0) == 0) cls = CellValue::not_exposed;
        else if (why.rfind("layer_rule: ", 0) == 0) cls = CellValue::layer_rule;
        else if (!why.empty()) table_error("unknown reason '" + why + "' on " + r.name);
        MatrixCase c = base(t, "condition_kind", value, cls);
        switch (cls) {
          case CellValue::offered:
            c.call = offered_call(c.engine);
            break;
          case CellValue::engine_refuses: {
            const Refusal x = kind_refusal(c.engine, geometry, k.name);
            c.call = x.call;
            c.engine_text = x.text;
            c.code = kind_code(t);
            break;
          }
          case CellValue::not_exposed: {
            c.code = kind_code(t);
            c.reason = why.substr(std::string("not_exposed: ").size());
            auto rm = r.roadmap.find(t);
            if (rm != r.roadmap.end() && rm->second.count(k.name)) c.roadmap = rm->second.at(k.name);
            break;
          }
          case CellValue::layer_rule:
            c.code = kind_code(t);
            c.reason = layer_rule(why.substr(std::string("layer_rule: ").size()))->description;
            break;
          case CellValue::engine_cannot_express: break;
        }
        column_of(c, straight_meshes(c.engine));
        out.push_back(c);
      }
    }
    // the chemistry engine's refusal that is not a kind of its own: every electrode galvanostatic
    if (t == kCurrent) {
      MatrixCase c = base(t, "condition_kind", "electrode:every_galvanostatic", CellValue::engine_refuses);
      c.call = "layout";
      c.engine_text = kTextGalvanostatic;
      c.code = codes::electrochem_potential_level_free;
      column_of(c, straight_meshes(c.engine));
      out.push_back(c);
    }
  }

  void option_values(const std::string& t, const PhysicsType& ty) {
    auto it = r.options.find(t);
    if (it == r.options.end()) return;
    for (const auto& rule : it->second) {
      if (r.when.count(rule.name)) continue;
      const OptionSpec* spec = option_spec(ty, rule.name);
      if (!spec) table_error("option rule '" + rule.name + "' of " + r.name + " is not an option of " + t);
      if (!rule.forced.empty()) {
        // an option of which the engine has one value and no member to set: the forced value is offered, and every
        // other value is a cell the engine cannot express
        for (const auto& v : spec->values) {
          const bool ok = v == rule.forced;
          MatrixCase c = base(t, "option_value", rule.name + "=" + v,
                              ok ? CellValue::offered : CellValue::engine_cannot_express);
          if (ok) {
            c.call = offered_call(c.engine);
            column_of(c, straight_meshes(c.engine));
          } else {
            c.member = absent_member(c.engine, rule.name);
            c.code = option_code(t);
          }
          out.push_back(c);
        }
        continue;
      }
      for (const auto& v : spec->values) {
        const bool ok = has(rule.offered, v);
        MatrixCase c =
            base(t, "option_value", rule.name + "=" + v, ok ? CellValue::offered : CellValue::engine_refuses);
        Strs meshes = straight_meshes(c.engine);
        if (ok) {
          c.call = offered_call(c.engine);
          // P1 on the curved mesh is the geometry-order cell; the stabilization without SUPG runs on both meshes
          // wherever SUPG is offered, so that the mesh with nodes is shown to be refused for SUPG only
          if (c.engine == "chem::CurrentDistribution" && rule.name == "order" && v == "1") meshes = {"base"};
          if (rule.name == "stabilization" && v == "none" && has(rule.offered, "supg")) meshes = {"base", "nodes"};
        } else {
          const OptionRefusal x = option_refusal(c.engine, rule.name, v);
          c.call = x.call;
          c.engine_text = x.text;
          c.code = x.code;
        }
        column_of(c, meshes);
        out.push_back(c);
      }
    }
  }

  void data_slots() {
    for (const auto* list : {&r.uniform, &r.one_expression}) {
      const bool uniform = list == &r.uniform;
      for (const auto& [t, props] : *list)
        for (const auto& p : props) {
          MatrixCase c = base(t, uniform ? "uniform" : "one_expression", p, CellValue::engine_cannot_express);
          c.engine = cell_engine(r, t);
          c.member = slot_member(c.engine, p);
          c.code = uniform ? codes::physics_property_not_uniform
                           : (p == "prescribed_advection.velocity" ? codes::heat_prescribed_advection_partial
                                                                   : codes::heat_data_not_regionwise);
          out.push_back(c);
        }
    }
  }

  void element_type() {
    // the quadrilateral mesh is built for the engines of the fluids library; the chemistry engine takes triangles by
    // construction (its mesh input has three vertices per element)
    if (!r.needs.fluids || (geometry != "planar" && geometry != "axisymmetric")) return;
    const std::string t = r.cell_types.at(0);
    const bool ok = has(r.element_types, "quadrilateral");
    MatrixCase c = base(t, "element_type", "quadrilateral", ok ? CellValue::offered : CellValue::engine_refuses);
    c.call = "constructor";
    if (!ok) {
      c.engine_text = kTextQuad;
      c.code = codes::element_type_not_offered;
    }
    column_of(c, {"quad"});
    out.push_back(c);
  }

  void geometry_order(const std::string& t, const PhysicsType& ty) {
    if (geometry != "planar" && geometry != "axisymmetric") return;
    // an option value that needs straight elements, refused by the engine on a mesh with nodes
    if (const OptionRule* s = option_rule(r, t, "stabilization"); s && has(s->offered, "supg")) {
      const OptionSpec* spec = option_spec(ty, "stabilization");
      if (spec && spec->max_geometry_order.count("supg") && spec->max_geometry_order.at("supg") == 1) {
        MatrixCase c = base(t, "geometry_order", "supg", CellValue::engine_refuses);
        c.call = "constructor";
        c.engine_text = kTextSupgNodes;
        c.code = codes::geometry_order_not_offered;
        column_of(c, {"nodes"});
        out.push_back(c);
      }
    }
    // a geometry order above the field order: a layer rule where the row lists it, an engine refusal where the
    // engine itself refuses curved elements with P1
    if (!has(r.layer_rules, "geometry_order_exceeds_field_order")) {
      const OptionRule* o = option_rule(r, t, "order");
      if (o && has(o->offered, "1")) {
        MatrixCase c = base(t, "geometry_order", "order=1", CellValue::engine_refuses);
        c.call = "constructor";
        c.engine_text = kTextCurvedP1;
        c.code = codes::geometry_order_exceeds_field_order;
        column_of(c, {"nodes"});
        out.push_back(c);
      }
    }
  }

  void ranks() {
    // the engines of the fluids library take a partitioned mesh; the chemistry engines are one process by construction
    if (!r.needs.fluids || (geometry != "planar" && geometry != "axisymmetric")) return;
    const std::string t = r.cell_types.at(0);
    const bool any = r.needs.max_ranks == 0;
    MatrixCase c = base(t, "ranks", "2", any ? CellValue::offered : CellValue::engine_refuses);
    c.ranks = 2;
    if (any) {
      // a stationary engine is checked through its solve, the time stepper through its first time steps
      c.call = c.engine == "TransientNavierStokes" ? "first_step" : "solve";
    } else {
      const Refusal x = rank_refusal(c.engine);
      c.call = x.call;
      c.engine_text = x.text;
      c.code = codes::route_single_rank;
    }
    column_of(c, {"base"});
    out.push_back(c);
  }

  void study_types() {
    // a geometry-free row whose type has no route of the other study type (a reactor has no stationary
    // solve, and the default stationary step is refused with the step to write)
    if (!r.geometries.empty()) return;
    for (const char* st : {"stationary", "transient"}) {
      const bool ok = has(r.study_types, st);
      MatrixCase c = base(r.cell_types.at(0), "study_type", st, ok ? CellValue::offered : CellValue::layer_rule);
      if (!ok) {
        c.code = codes::study_type_not_offered;
        c.reason = "a reactor is solved in time: the default stationary step, or a written one, is refused with the "
                   "transient step to write, because an end time has no sensible default";
      }
      out.push_back(c);
    }
  }

  void all() {
    for (const auto& t : r.cell_types) {
      const PhysicsType* ty = registry().type(t);
      if (!ty) table_error("cell type " + t + " of " + r.name + " is not a type of the registry");
      condition_kinds(t, *ty);
      option_values(t, *ty);
      geometry_order(t, *ty);
    }
    data_slots();
    element_type();
    ranks();
    study_types();
  }
};

// The sparse boundary numbering of a master mesh (2001 to 2004 and 900, not 1..n), handed to each engine.  Not a row:
// these cells pin the engine fact for which the instance mesh is numbered densely.
void sparse_cases(std::vector<MatrixCase>& out) {
  struct S {
    const char* engine;
    const char* physics;
    bool accepts;
  };
  for (const S& s : {S{"StokesSolver", kFluid, true}, S{"NavierStokesSolver", kFluid, true},
                     S{"TransientNavierStokes", kFluid, true}, S{"BoussinesqSolver", kHeat, false},
                     S{"transport::ScalarTransport", kHeat, false}}) {
    MatrixCase c;
    c.engine = s.engine;
    c.physics = s.physics;
    c.axis = "mesh_numbering";
    c.value = "sparse_attributes";
    c.cls = s.accepts ? CellValue::offered : CellValue::engine_refuses;
    c.meshes = {"sparse"};
    if (s.accepts) {
      c.call = "constructor";
    } else {
      // the transport operator counts its conditions in finalize, which the Boussinesq solver calls in its constructor
      c.call = c.engine == "BoussinesqSolver" ? "constructor" : "finalize";
      c.engine_text = kTextSparse;
      // a document cannot produce this numbering (the instance mesh is renumbered densely); an engine that met it
      // would report a route-table error
      c.code = codes::route_table_out_of_date;
    }
    out.push_back(c);
  }
}

std::string make_id(const MatrixCase& c) {
  std::string meshes;
  for (const auto& m : c.meshes) meshes += (meshes.empty() ? "" : "+") + m;
  return (c.row.empty() ? c.engine : c.row) + "/" + c.physics + "/" + c.axis + "=" + c.value + "@" +
         (meshes.empty() ? "none" : meshes) + "/" + std::to_string(c.ranks);
}

}  // namespace

std::vector<MatrixCase> matrix_cases() {
  std::vector<MatrixCase> out;
  for (const Route& r : registry().routes) {
    RowCases rc{r, out, geometry_of(r), engine_column(r)};
    rc.all();
  }
  sparse_cases(out);
  std::set<std::string> ids;
  for (auto& c : out) {
    c.id = make_id(c);
    if (!ids.insert(c.id).second) table_error("duplicate case id " + c.id);
  }
  return out;
}

// ================================================================================================ documents
namespace {

json zeros(const std::string& geometry, bool meridional) {
  if (geometry == "planar" || (geometry == "axisymmetric" && meridional)) return json::array({0, 0});
  return json::array({0, 0, 0});
}

// The boundaries of the base document: the condition keys of every instance.  The axis of an axisymmetric model
// (its left side, r = 0) is never written.
Strs boundary_names(const std::string& g) {
  if (g == "planar") return {"left", "right", "bottom", "top"};
  if (g == "axisymmetric") return {"right", "bottom", "top"};
  return {"left", "right", "front", "back", "bottom", "top"};
}
// The boundary that carries a condition-kind cell.
std::string variable_boundary(const std::string& g) { return g == "axisymmetric" ? "right" : "top"; }

json boundary_selectors(const std::string& g, bool imported) {
  json b = json::object();
  std::map<std::string, std::pair<std::string, int>> where;
  if (g == "planar") where = {{"left", {"x", 0}}, {"right", {"x", 1}}, {"bottom", {"y", 0}}, {"top", {"y", 1}}};
  else if (g == "axisymmetric") where = {{"right", {"r", 1}}, {"bottom", {"z", 0}}, {"top", {"z", 1}}};
  else
    where = {{"left", {"x", 0}},  {"right", {"x", 1}}, {"front", {"y", 0}},
             {"back", {"y", 1}},  {"bottom", {"z", 0}}, {"top", {"z", 1}}};
  for (const auto& [name, w] : where) {
    if (imported) b[name] = {{"group", name}};
    else b[name] = {{"of", "box"}, {"where", {{w.first, w.second}}}};
  }
  return b;
}

// One region, or two halves (the uniform and one-expression cells), named box or box_a and box_b.
json regions(const std::string& g, bool two) {
  json out = json::array();
  auto rect = [&](double a0, double a1) -> json {
    if (g == "planar") return {{"shape", {{"rect", {{"x", {a0, a1}}, {"y", {0.0, 1.0}}}}}}};
    if (g == "axisymmetric") return {{"shape", {{"rect", {{"r", {0.0, 1.0}}, {"z", {a0, a1}}}}}}};
    return {{"solid", {{"box", {{"min", {a0, 0.0, 0.0}}, {"max", {a1, 1.0, 1.0}}}}}}};
  };
  if (!two) {
    json r = rect(0, 1);
    r["name"] = "box";
    r["material"] = "medium";
    out.push_back(r);
  } else {
    json a = rect(0, 0.5), b = rect(0.5, 1);
    a["name"] = "box_a";
    a["material"] = "medium_a";
    b["name"] = "box_b";
    b["material"] = "medium_b";
    out.push_back(a);
    out.push_back(b);
  }
  return out;
}

// A material that carries `blocks` (the registry's block names: "fluid", "density", "fluid.beta_1_K", "thermal",
// "thermal.cp_J_kgK", "electrolyte"); `differ` names a property given another value.
json material(const std::set<std::string>& blocks, const std::string& differ) {
  const double f = 2.0;
  auto v = [&](const std::string& prop, double x) { return prop == differ ? f * x : x; };
  json m = json::object();
  if (blocks.count("density")) m["density"] = v("density", 1.0);
  if (blocks.count("fluid") || blocks.count("fluid.beta_1_K")) {
    m["fluid"] = {{"mu_Pa_s", 1.0}};
    if (blocks.count("fluid.beta_1_K")) m["fluid"]["beta_1_K"] = v("fluid.beta_1_K", 0.003);
  }
  if (blocks.count("thermal") || blocks.count("thermal.cp_J_kgK")) {
    m["thermal"] = {{"k_W_mK", v("thermal.k_W_mK", 1.0)}};
    if (blocks.count("thermal.cp_J_kgK")) m["thermal"]["cp_J_kgK"] = v("thermal.cp_J_kgK", 1.0);
  }
  if (blocks.count("electrolyte")) m["electrolyte"] = {{"kappa_S_m", 1.0}};
  return m;
}

json flow_condition(const std::string& kind, const std::string& g) {
  if (kind == "wall" || kind == "open" || kind == "slip" || kind == "symmetry") return {{"kind", kind}};
  if (kind == "inlet") return {{"kind", "inlet"}, {"velocity_m_s", zeros(g, false)}};
  if (kind == "outlet") return {{"kind", "outlet"}, {"pressure_Pa", 0}};
  if (kind == "traction") return {{"kind", "traction"}, {"traction_Pa", zeros(g, false)}};
  table_error("no document condition for the flow kind '" + kind + "'");
}

json heat_condition(const std::string& kind) {
  if (kind == "temperature") return {{"kind", "temperature"}, {"T_K", 305.0}};
  if (kind == "heat_flux" || kind == "total_flux") return {{"kind", kind}, {"q_W_m2", 10.0}};
  if (kind == "convection") return {{"kind", "convection"}, {"h_W_m2K", 10.0}, {"T_ext_K", 300.0}};
  if (kind == "insulation" || kind == "outflow") return {{"kind", kind}};
  table_error("no document condition for the heat kind '" + kind + "'");
}

json electrode(double phi) { return {{"kind", "electrode"}, {"reactions", json::array({"plate"})}, {"phi_s_V", phi}}; }

json current_condition(const std::string& kind) {
  if (kind == "electrode") return electrode(0.0);
  if (kind == "insulation") return {{"kind", "insulation"}};
  if (kind == "normal_current") return {{"kind", "normal_current"}, {"i_A_m2", 1.0}};
  table_error("no document condition for the current-distribution kind '" + kind + "'");
}

// The base flow instance: a channel (planar: inflow on the left; axisymmetric: a pipe with inflow at z = 0; 3D: a duct
// along x) with walls elsewhere, and `cond` on the variable boundary.
json flow_instance(const std::string& g, const std::string& equation, const std::string& kind) {
  json bc = json::object();
  const std::string var = variable_boundary(g);
  if (g == "planar") {
    bc["left"] = {{"kind", "inlet"}, {"velocity_m_s", {"4*y*(1-y)", 0}}};
    bc["right"] = {{"kind", "outlet"}, {"pressure_Pa", 0}};
    bc["bottom"] = {{"kind", "wall"}};
  } else if (g == "axisymmetric") {
    bc["bottom"] = {{"kind", "inlet"}, {"velocity_m_s", {0, "2*(1-r^2)", 0}}};
    bc["top"] = {{"kind", "outlet"}, {"pressure_Pa", 0}};
  } else {
    bc["left"] = {{"kind", "inlet"}, {"velocity_m_s", {"16*y*(1-y)*z*(1-z)", 0, 0}}};
    bc["right"] = {{"kind", "outlet"}, {"pressure_Pa", 0}};
    for (const char* w : {"front", "back", "bottom"}) bc[w] = {{"kind", "wall"}};
  }
  bc[var] = {{"kind", "wall"}};
  if (kind == "periodic") {
    // a periodic pair top -> bottom, driven by a pressure drop; the partner carries no condition of its own
    json t = g == "3d" ? json::array({0, 0, -1}) : json::array({0, -1});
    bc["top"] = {{"kind", "periodic"}, {"partner", "bottom"}, {"translation_m", t}, {"pressure_drop_Pa", 1.0}};
    bc.erase("bottom");
  } else if (!kind.empty()) {
    bc[var] = flow_condition(kind, g);
  }
  json inst = {{"type", kFluid}, {"options", {{"equation", equation}}}, {"boundary_conditions", bc}};
  return inst;
}

// The base heat instance.  Without advection: two temperatures and insulation.  With advection (a prescribed
// velocity or a solved flow): the inflow temperature, the outflow, a heated wall.  The Boussinesq cavity: a hot and a
// cold side.  `kind` goes on the variable boundary (insulation in the base document); `form` is the advective form
// the instance is written with (empty: the row's default), and with conservative_flux the outflow is replaced by a
// temperature, since `outflow` is refused with that form.
enum class HeatBase { conduction, advected, cavity };
json heat_instance(const std::string& g, HeatBase hb, const std::string& kind, const std::string& form) {
  json bc = json::object();
  const std::string var = variable_boundary(g);
  const json outflow = form == "conservative_flux" ? json{{"kind", "temperature"}, {"T_K", 300.0}}
                                                   : json{{"kind", "outflow"}};
  auto T = [](double t) { return json{{"kind", "temperature"}, {"T_K", t}}; };
  if (g == "planar") {
    switch (hb) {
      case HeatBase::conduction:
        bc["left"] = T(300), bc["right"] = T(310), bc["bottom"] = {{"kind", "insulation"}};
        break;
      case HeatBase::advected: bc["left"] = T(300), bc["right"] = outflow, bc["bottom"] = T(310); break;
      case HeatBase::cavity:
        bc["left"] = T(310), bc["right"] = T(300), bc["bottom"] = {{"kind", "insulation"}};
        break;
    }
  } else {
    switch (hb) {
      case HeatBase::conduction: bc["bottom"] = T(300); bc["top"] = T(310); break;
      case HeatBase::advected: bc["bottom"] = T(300); bc["top"] = outflow; break;
      case HeatBase::cavity: bc["bottom"] = T(310); bc["top"] = T(300); break;
    }
  }
  bc[var] = kind.empty() ? json{{"kind", "insulation"}} : heat_condition(kind);
  return {{"type", kHeat}, {"boundary_conditions", bc}};
}

// The base current-distribution instance: an anode and a cathode with a linear kinetics, insulation elsewhere.
json current_instance(const std::string& g, const std::string& kind, const std::string& current_key) {
  json bc = json::object();
  const std::string var = variable_boundary(g);
  std::string anode = "left", cathode = "right";
  if (g == "axisymmetric") anode = "bottom", cathode = "top";
  bc[anode] = electrode(0.05);
  bc[cathode] = electrode(0.0);
  for (const auto& b : boundary_names(g))
    if (b != anode && b != cathode) bc[b] = {{"kind", "insulation"}};
  if (kind == "electrode:every_galvanostatic") {
    bc[anode] = {{"kind", "electrode"}, {"reactions", json::array({"plate"})}, {current_key, 0.1}};
    bc[cathode] = {{"kind", "electrode"}, {"reactions", json::array({"plate"})}, {current_key, -0.1}};
  } else if (!kind.empty()) {
    bc[var] = current_condition(kind);
  }
  return {{"type", kCurrent}, {"options", {{"kinetics", "secondary"}, {"order", 2}}}, {"boundary_conditions", bc}};
}

// The option value of `spec` as a document writes it: a number for an integer or numeric option, else the text.
json option_value(const OptionSpec& spec, const std::string& v) {
  if (spec.type == "integer") return std::stoi(v);
  if (spec.type == "number") return std::stod(v);
  return v;
}

}  // namespace

json minimal_document(const MatrixCase& c) {
  if (c.row.empty() || c.axis == "ranks") return nullptr;
  const Registry& R = registry();
  const Route* r = R.route(c.row);
  if (!r) table_error("case " + c.id + " names the unknown row '" + c.row + "'");
  const std::string g = geometry_of(*r);
  const std::string& n = r->name;

  json doc = {{"schema", "axionmultiphysics.design/1"}, {"name", "matrix_case"},
              {"description", "matrix case " + c.id}};
  json physics = json::object();

  // ---- the geometry-free reactor
  if (g.empty()) {
    const json step0 = {{"reactants", {{"A", 1}}}, {"products", {{"B", 1}}}, {"A", 0.7}, {"Ea_J_mol", 0.0}};
    doc["reactions"] = {{"species", json::array({"A", "B"})}, {"mechanism", json::array({step0})}};
    for (const auto& t : r->physics) physics[t == "chem.reactor" ? "tank" : t] = {{"type", t}};
    doc["physics"] = physics;
    const std::string st = c.axis == "study_type" ? c.value : r->study_types.at(0);
    json step = {{"id", st == "transient" ? "run" : "steady"}, {"type", st}};
    if (st == "transient") step["time"] = {{"t_end_s", 1.0}};
    doc["study"] = json::array({step});
    return doc;
  }

  doc["symmetry"] = g;
  const bool two = c.axis == "uniform" || c.axis == "one_expression";
  const bool imported = c.axis == "element_type";

  // ---- materials: the blocks of every type of the row
  std::set<std::string> blocks;
  for (const auto& [t, b] : r->material_blocks) blocks.insert(b.begin(), b.end());
  json materials = json::object();
  if (two) {
    materials["medium_a"] = material(blocks, "");
    materials["medium_b"] = material(blocks, c.value);
  } else {
    materials["medium"] = material(blocks, "");
  }
  doc["materials"] = materials;
  doc["regions"] = regions(g, two);
  if (two) doc["domains"] = {{"box", json::array({"box_a", "box_b"})}};
  doc["boundaries"] = boundary_selectors(g, imported);

  // ---- the cell's variation, as read by the instance builders.  The advective form is the cell's own: the form of
  // a kind cell "<kind>@<form>", or the value of an advective_form cell, so that the base conditions are those the
  // form offers; empty, the row's default, which the document does not write.
  const std::string kind = c.axis == "condition_kind" ? c.value.substr(0, c.value.find('@')) : "";
  const auto at = c.value.find('@');
  std::string form = (c.axis == "condition_kind" && at != std::string::npos) ? c.value.substr(at + 1) : "";
  const std::string form_option = "advective_form=";
  if (c.axis == "option_value" && c.value.rfind(form_option, 0) == 0) form = c.value.substr(form_option.size());

  // ---- instances
  const bool has_flow = has(r->physics, kFluid), has_heat = has(r->physics, kHeat);
  if (has_flow) {
    std::string equation = "stokes";
    if (auto w = r->when.find("equation"); w != r->when.end()) equation = w->second;
    const bool flow_cell = c.physics == kFluid && c.axis == "condition_kind";
    json flow = flow_instance(g, equation, flow_cell ? kind : "");
    if (n.rfind("boussinesq_", 0) == 0) {
      // the cavity: walls all round
      json bc = json::object();
      for (const auto& b : boundary_names(g)) bc[b] = {{"kind", "wall"}};
      flow["boundary_conditions"] = bc;
    }
    physics["flow"] = flow;
  }
  if (has_heat) {
    HeatBase hb = HeatBase::conduction;
    if (n.rfind("boussinesq_", 0) == 0) hb = HeatBase::cavity;
    else if (n.rfind("heat_prescribed_advection_", 0) == 0 || n.rfind("flow_then_heat_", 0) == 0)
      hb = HeatBase::advected;
    json heat = heat_instance(g, hb, c.physics == kHeat ? kind : "", form);
    if (!form.empty()) heat["options"]["advective_form"] = form;
    json sources = json::array();
    if (n.rfind("heat_prescribed_advection_", 0) == 0) {
      json pa = {{"kind", "prescribed_advection"},
                 {"velocity_m_s", g == "planar" ? json::array({1, 0}) : json::array({0, 1})}};
      if (c.axis == "one_expression" && c.value == "prescribed_advection.velocity") pa["in"] = "box_a";
      sources.push_back(pa);
    }
    if (c.axis == "one_expression" && c.value == "heat_source") {
      sources.push_back({{"kind", "heat_source"}, {"in", "box_a"}, {"q_W_m3", 1.0}});
      sources.push_back({{"kind", "heat_source"}, {"in", "box_b"}, {"q_W_m3", 2.0}});
    }
    if (!sources.empty()) heat["sources"] = sources;
    physics["heat"] = heat;
  }
  if (has(r->physics, kCurrent)) {
    std::string current_key = "current_A";
    if (g == "planar") current_key = "current_A_m";
    physics["cell"] = current_instance(g, c.physics == kCurrent ? kind : "", current_key);
    const json plate = {{"kind", "linear"}, {"i0_A_m2", 10.0}, {"E_eq_V", 0.0}, {"alpha_a", 0.5}, {"alpha_c", 0.5}};
    doc["reactions"] = {{"electrode", {{"plate", plate}}}};
  }

  // ---- the option of an option-value or geometry-order cell, on the instance of its type
  auto instance_of = [&](const std::string& type) -> json* {
    for (auto& [id, inst] : physics.items())
      if (inst.value("type", "") == type) return &inst;
    return nullptr;
  };
  auto set_option = [&](const std::string& type, const std::string& name, const std::string& value) {
    json* inst = instance_of(type);
    const PhysicsType* ty = R.type(type);
    const OptionSpec* spec = ty ? option_spec(*ty, name) : nullptr;
    if (!inst || !spec) table_error("case " + c.id + ": no option " + name + " on an instance of " + type);
    (*inst)["options"][name] = option_value(*spec, value);
  };
  if (c.axis == "option_value") {
    const auto eq = c.value.find('=');
    set_option(c.physics, c.value.substr(0, eq), c.value.substr(eq + 1));
  }
  json mesh = {{"size", 0.125}};
  if (c.axis == "geometry_order") {
    mesh["geometry_order"] = 2;
    if (c.value == "supg") set_option(c.physics, "stabilization", "supg");
    else if (c.value.rfind("order=", 0) == 0) set_option(c.physics, "order", c.value.substr(6));
  }
  if (imported) mesh = {{"file", "quadrilaterals.msh"}, {"info_file", "quadrilaterals.json"}};
  doc["physics"] = physics;

  // ---- couplings and study
  json couplings = json::array();
  for (const auto& term : r->couplings) {
    if (term == "advection") couplings.push_back({{"term", "advection"}, {"from", "flow"}, {"to", "heat"}});
    else if (term == "buoyancy")
      couplings.push_back({{"term", "buoyancy"},
                           {"from", "heat"},
                           {"to", "flow"},
                           {"T_ref_K", 305.0},
                           {"gravity_m_s2", json::array({0, -9.81})}});
    else table_error("no document entry for the coupling term '" + term + "'");
  }
  if (!couplings.empty()) doc["couplings"] = couplings;
  if (has(r->study_types, "transient") && !has(r->study_types, "stationary")) {
    const json step = {{"id", "run"}, {"type", "transient"}, {"time", {{"t_end_s", 0.1}}}};
    doc["study"] = json::array({step});
  }
  doc["mesh"] = mesh;
  return doc;
}

}  // namespace axmp::model
