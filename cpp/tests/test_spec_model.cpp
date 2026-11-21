// The parser of the model document (cpp/include/axionmultiphysics/spec_model.hpp, cpp/src/spec_model.cpp): is_model_document,
// parse_model_collect (every finding, section by section, in the order of the layers of the design note, section 3.1) and
// parse_model (the first error), for the sections other than the materials, which are test_spec_model_materials.cpp.
// Built in every build; nothing here needs the mesher or an engine.
//
// INDEPENDENT REFERENCES.
//   * The four documents of section 4 of the interface design note, verbatim, in cpp/tests/data/model/.
//   * The unchanged electromagnetic parser, parse_design, run on the "twin" of a model document (the same regions,
//     parameters and materials, with the keys of the electromagnetic grammar only): where the two parsers share a
//     grammar, the model path's message, pointer and hint must equal the twin's, and no text is typed here for them.
//   * The catalogue registry (physics_registry.hpp) for the names it lists: the kinds, options, terms, quantities and the
//     reserved ids.  Its cell values (classes, engines) are not read.
//   * Hand arithmetic for the numbers: the parabolic inlet profile 6 U y (H - y) / H^2 and T_ref = (T_hot + T_cold) / 2.
//
// TEXTS.  A text that the note gives is copied from it, with the section.  Where the note gives no text, the text below is
// the test author's and is the specification (marked AUTHOR TEXT).  An exact message, hint, code, pointer or details key
// asserted here is a requirement on the parser; what is compared by substring is marked so.
//
// THE RULES OF COLLECTION (the plan's architecture, made precise here).
//   1. One parser per section, in the order: identity and schema, parameters, symmetry, regions, curves, materials, mesh,
//      domains, boundaries, reactions, physics, couplings, study, outputs.  Findings are added in that order; within a
//      section, entries are checked in the order of the document (a JSON object iterates in key order, an array in index
//      order).  Findings::first_error() is the first error added; parse_model throws it.
//   2. An entry is a material, a region, a curve, a domain, a boundary, an instance, one boundary condition, one source,
//      one coupling, one study step, one output.  Within an entry the first error only is recorded, and the entry is then
//      dropped from the partial model; the other entries are checked.  A dropped entry KEEPS ITS NAME (a material, region,
//      curve, domain, boundary, instance id, step id, output name, reaction name) and, for an instance, its type, so that a
//      reference to it is not reported as unknown and its quantities stay in the vocabulary of the outputs.
//   3. A section that is not of the JSON type its grammar needs (a "materials" that is not an object, a "regions" that is
//      not an array) is reported once and is unusable as a whole.  A section is skipped, with no finding of its own, when
//      its prerequisite is unusable: every section that evaluates an expression when "parameters" has any error; every
//      section after "symmetry" when "symmetry" has an error; domains and boundaries when regions or curves are unusable;
//      physics when regions, curves, materials, domains or boundaries are unusable; couplings, study and outputs when
//      physics is unusable or skipped; the check of the electrode reactions that a condition names when "reactions" is
//      unusable.  An entry-level error skips nothing, with these two consequences, which keep one defect one finding:
//        - the names of the materials (the keys of "materials" and the built-in ones) are known before the regions are
//          parsed, because a region names its material;
//        - an instance whose "type" is missing or unknown has no catalogue entry: the checks that need its type (its
//          options, conditions and sources, the type of a coupling's ends, and the vocabulary of the outputs) are skipped
//          without a finding, and an output quantity is checked only when every instance has a known type.
//
// THE API THE TESTS USE (declared by cpp/include/axionmultiphysics/spec_model.hpp, namespaces axmp::spec and axmp::model;
// the findings sink by cpp/src/model_errors.hpp, namespace axmp::model):
//   axmp::spec::is_model_document(json) -> bool
//   axmp::spec::parse_model_collect(json, axmp::model::Findings&) -> axmp::model::ResolvedModel
//   axmp::spec::parse_model(json) -> axmp::model::ResolvedModel          (throws the first error, an axmp::Error)
//   axmp::model::Finding { enum class Kind { error, unavailable, deferred, warning } kind; axmp::Error error; }
//   axmp::model::Findings { add(Finding::Kind, Error); has_errors(); first_error(); all() }
//   the members of ResolvedModel, Instance, Condition, SourceTerm, Datum, Selector, Domain, Coupling, StudyStep, Output,
//   Reactions, MeshKeys, MaterialProps, Property, Law, as used below (the full declarations are in the header).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/physics_registry.hpp"
#include "axionmultiphysics/spec.hpp"
#include "axionmultiphysics/spec_model.hpp"
#include "axionmultiphysics/templates.hpp"
#include "../src/model_errors.hpp"
#include "../src/model_matrix.hpp"

using nlohmann::json;
using axmp::Error;
namespace M = axmp::model;
namespace E = axmp::model::msg;

static int failures = 0;
static void fail(const std::string& what) {
  std::printf("FAIL: %s\n", what.c_str());
  ++failures;
}
#define CHECK(cond, what)                      \
  do {                                         \
    if (!(cond)) fail(std::string(what));      \
  } while (0)

namespace {

using Strs = std::vector<std::string>;

// ======================================================================================================= helpers
std::string join(const Strs& v, const std::string& sep = ", ") {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
  return s;
}

std::string only_list(const std::set<std::string>& keys) {  // the hint of Node::only: sorted, "Allowed fields here: a, b."
  return "Allowed fields here: " + join(Strs(keys.begin(), keys.end())) + ".";
}

bool close(double a, double b, double rel = 1e-14) { return std::fabs(a - b) <= rel * std::max(std::fabs(a), std::fabs(b)); }

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
  try {
    json d;
    in >> d;
    return d;
  } catch (const std::exception& e) {
    fail(path + " is not JSON: " + e.what());
    return json::object();
  }
}
json channel() { return load("channel.json"); }
json heated() { return load("heated_cavity.json"); }
json plate() { return load("plate_cell.json"); }
json busbar() { return load("busbar.json"); }

json& at(json& d, const std::string& pointer) { return d[json::json_pointer(pointer)]; }

// An axisymmetric pipe: regions in (r, z), a flow with three velocity components.
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

// The plate cell of section 4.3 made axisymmetric: its electrodes are the bottom and the top.
json plate_axisymmetric() {
  json d = plate();
  d["symmetry"] = "axisymmetric";
  d["regions"][0]["shape"] = J(R"j({"rect": {"r": [0, "gap"], "z": [0, "Hc"]}})j");
  d["boundaries"]["anode"] = J(R"j({"of": "bulk", "where": {"z": 0}})j");
  d["boundaries"]["cathode"] = J(R"j({"of": "bulk", "where": {"z": "Hc"}})j");
  return d;
}

// A condition or a datum of a parsed instance that a defective parser may not have produced: a missing one is a failure of
// the test that asks for it, not an exception that ends the run.
const M::Condition& cond(const M::Instance& inst, const std::string& key) {
  static const M::Condition none;
  const auto it = inst.conditions.find(key);
  if (it == inst.conditions.end()) {
    fail("instance '" + inst.id + "' has no condition '" + key + "'");
    return none;
  }
  return it->second;
}
template <class Holder>  // a condition, a source or a coupling
const M::Datum& datum_of(const Holder& c, const std::string& key) {
  static const M::Datum none;
  const auto it = c.data.find(key);
  if (it == c.data.end()) {
    fail("a condition, source or coupling has no datum '" + key + "'");
    return none;
  }
  return it->second;
}

struct Parsed {
  M::Findings f;
  M::ResolvedModel m;
};
Parsed parse(const json& doc) {
  Parsed p;
  p.m = axmp::spec::parse_model_collect(doc, p.f);
  return p;
}
std::vector<Error> of_kind(const M::Findings& f, M::Finding::Kind k) {
  std::vector<Error> out;
  for (const auto& x : f.all())
    if (x.kind == k) out.push_back(x.error);
  return out;
}
std::vector<Error> errors_of(const M::Findings& f) { return of_kind(f, M::Finding::Kind::error); }

json details_of(const Error& e) { return e.details().empty() ? json(nullptr) : json::parse(e.details()); }

// every key of `want` is in `have` with a subset value; arrays and scalars compare equal
bool subset(const json& have, const json& want) {
  if (want.is_object()) {
    if (!have.is_object()) return false;
    for (auto it = want.begin(); it != want.end(); ++it)
      if (!have.contains(it.key()) || !subset(have.at(it.key()), it.value())) return false;
    return true;
  }
  return have == want;
}

// What one error must be.  Empty strings are not compared, except `code`, which is always compared ("" means no code).
struct Want {
  std::string code, where, message, message_has, hint, hint_has;
  json details = nullptr;
  size_t count = 1;  // the number of errors in the whole document
  Want& with_hint(std::string h) { hint = std::move(h); return *this; }
  Want& with_hint_has(std::string h) { hint_has = std::move(h); return *this; }
  Want& with_message_has(std::string m) { message_has = std::move(m); return *this; }
  Want& with_details(json d) { details = std::move(d); return *this; }
  Want& with_count(size_t n) { count = n; return *this; }
};
Want W(std::string code, std::string where, std::string message = "") {
  Want w;
  w.code = std::move(code);
  w.where = std::move(where);
  w.message = std::move(message);
  return w;
}

void compare_error(const std::string& what, const Error& e, const Want& w) {
  if (e.code() != w.code) fail(what + ": code '" + e.code() + "', want '" + w.code + "' (" + e.what() + ")j");
  if (e.where() != w.where) fail(what + ": pointer '" + e.where() + "', want '" + w.where + "'");
  if (!w.message.empty() && e.message() != w.message)
    fail(what + ": message\n    '" + e.message() + "'\n  want\n    '" + w.message + "'");
  if (!w.message_has.empty() && e.message().find(w.message_has) == std::string::npos)
    fail(what + ": message '" + e.message() + "' lacks '" + w.message_has + "'");
  if (!w.hint.empty() && e.hint() != w.hint)
    fail(what + ": hint\n    '" + e.hint() + "'\n  want\n    '" + w.hint + "'");
  if (!w.hint_has.empty() && e.hint().find(w.hint_has) == std::string::npos)
    fail(what + ": hint '" + e.hint() + "' lacks '" + w.hint_has + "'");
  if (!w.details.is_null()) {
    const json d = details_of(e);
    if (!subset(d, w.details)) fail(what + ": details " + d.dump() + " do not contain " + w.details.dump());
  }
}

// The document has exactly w.count errors, the first is `w`, and parse_model throws that first one and nothing else.
void check_one(const std::string& what, const json& doc, const Want& w) {
  const Parsed p = parse(doc);
  const auto errs = errors_of(p.f);
  if (errs.size() != w.count) {
    std::string all;
    for (const auto& e : errs) all += "\n    [" + e.code() + "] " + e.what();
    fail(what + ": " + std::to_string(errs.size()) + " error(s), want " + std::to_string(w.count) + all);
    if (errs.empty()) return;
  }
  compare_error(what, errs.front(), w);
  try {
    axmp::spec::parse_model(doc);
    fail(what + ": parse_model did not throw");
  } catch (const Error& t) {
    if (std::string(t.what()) != errs.front().what() || t.code() != errs.front().code() ||
        t.where() != errs.front().where())
      fail(what + ": parse_model threw '" + t.what() + "', not the first error '" + errs.front().what() + "'");
  } catch (const std::exception& t) {
    fail(what + ": parse_model threw a bare exception: " + t.what());
  }
}

bool clean(const std::string& what, const json& doc) {
  const Parsed p = parse(doc);
  const auto errs = errors_of(p.f);
  for (const auto& e : errs) fail(what + ": unexpected error [" + e.code() + "] " + e.what() + " [at " + e.where() + "]");
  try {
    axmp::spec::parse_model(doc);
  } catch (const std::exception& e) {
    if (errs.empty()) fail(what + ": parse_model threw: " + e.what());
  }
  return errs.empty();
}

// What parse_design says about an electromagnetic document: its message, pointer and hint.
struct Triple {
  bool threw = false;
  std::string message, where, hint;
};
Triple em_error(const json& em) {
  Triple t;
  try {
    axmp::spec::parse_design(em);
  } catch (const Error& e) {
    t.threw = true;
    t.message = e.message();
    t.where = e.where();
    t.hint = e.hint();
  } catch (const std::exception& e) {
    t.threw = true;
    t.message = std::string("(bare) ") + e.what();
  }
  return t;
}

// A model document and its electromagnetic twin give the same refusal.  `model` must have exactly one error.
void same_as_twin(const std::string& what, const json& model, const json& em) {
  const Triple t = em_error(em);
  if (!t.threw) {
    fail(what + ": the electromagnetic twin was not refused, so there is no reference");
    return;
  }
  const Parsed p = parse(model);
  const auto errs = errors_of(p.f);
  if (errs.size() != 1) {
    fail(what + ": " + std::to_string(errs.size()) + " errors in the model document, want 1");
    for (const auto& e : errs) std::printf("    [%s] %s [at %s]\n", e.code().c_str(), e.message().c_str(), e.where().c_str());
    return;
  }
  const Error& e = errs.front();
  if (e.message() != t.message) fail(what + ": message '" + e.message() + "', the electromagnetic parser says '" + t.message + "'");
  if (e.where() != t.where) fail(what + ": pointer '" + e.where() + "', the electromagnetic parser says '" + t.where + "'");
  if (e.hint() != t.hint) fail(what + ": hint '" + e.hint() + "', the electromagnetic parser says '" + t.hint + "'");
}

// the registry's lists, in the registry's order
Strs type_names() {
  Strs v;
  for (const auto& t : M::registry().types) v.push_back(t.name);
  return v;
}
Strs kind_names(const std::string& type) {
  Strs v;
  const M::PhysicsType* t = M::registry().type(type);
  if (t) for (const auto& k : t->condition_kinds) v.push_back(k.name);
  return v;
}
Strs source_names(const std::string& type) {
  Strs v;
  const M::PhysicsType* t = M::registry().type(type);
  if (t) for (const auto& k : t->source_kinds) v.push_back(k.name);
  return v;
}
Strs option_names(const std::string& type) {
  Strs v;
  const M::PhysicsType* t = M::registry().type(type);
  if (t) for (const auto& o : t->options) v.push_back(o.name);
  return v;
}
Strs term_names() {
  Strs v;
  for (const auto& t : M::registry().terms) v.push_back(t.name);
  return v;
}
Strs quantity_names(const std::string& type, const std::string& geometry) {
  Strs v;
  const M::PhysicsType* t = M::registry().type(type);
  if (t && t->quantities.count(geometry))
    for (const auto& q : t->quantities.at(geometry)) v.push_back(q.name);
  return v;
}
json jarr(const Strs& v) {
  json a = json::array();
  for (const auto& s : v) a.push_back(s);
  return a;
}
std::string list_sentence(const std::string& plural, const std::string& owner, const Strs& v) {
  return "The " + plural + " of '" + owner + "' are: " + join(v) + ".";
}
std::string geometry_phrase(const std::string& g) {
  return g == "planar" ? "a planar model" : g == "axisymmetric" ? "an axisymmetric model" : "a 3d model";
}

// ================================================================================================ section 3.2 item 4
// AUTHOR TEXT: none; the text is the note's (section 3.2 item 4, the block under "In a model document the electromagnetic
// analysis sections ... are refused"), the hint with its line breaks joined by one space.
const char* const kEmSectionHint =
    "electromagnetic analyses and the physics section are solved by different engines in this version. Keep the "
    "electromagnetic design in its own document. Coupling them (deformation feedback into the electromagnetic figures "
    "of merit) is planned; `axmp physics --routes` lists what can be combined.";
const Strs kEmSections = {"windings",  "circuits", "superconductors", "pickup",   "background",
                          "fullwave",  "cavity",   "hts",             "electrostatics", "mechanics",
                          "thermal",   "benchmarks", "scan",          "cost",     "conductors"};

// ============================================================================================================ checks

void check_four_documents_parse() {
  for (const char* name : {"channel.json", "heated_cavity.json", "plate_cell.json", "busbar.json"})
    clean(name, load(name));

  // ---- channel (section 4.1)
  {
    const Parsed p = parse(channel());
    const M::ResolvedModel& m = p.m;
    CHECK(m.name == "channel" && m.symmetry == "planar", "channel: identity");
    CHECK(m.description.rfind("2D water channel", 0) == 0, "channel: description");
    CHECK(close(m.parameters.at("L"), 0.04) && close(m.parameters.at("H"), 0.01) && close(m.parameters.at("mu"), 1.002e-3) &&
              close(m.parameters.at("rho"), 998.2),
          "channel: parameters");
    CHECK(m.regions.size() == 1 && m.regions[0].name == "channel" && m.regions[0].material == "water", "channel: region");
    if (m.regions.size() == 1) {
      const auto& s = m.regions[0].shape;  // a planar rect keeps x in r and y in z
      CHECK(s.kind == axmp::spec::Shape::Kind::Rect && s.r0 == 0.0 && close(s.r1, 0.04) && s.z0 == 0.0 && close(s.z1, 0.01),
            "channel: the rectangle is [0, L] x [0, H]");
    }
    CHECK(m.materials.count("water") == 1 && m.materials.count("copper") == 1, "channel: the document's material and a built-in one");
    if (m.materials.count("water")) {
      const M::MaterialProps& w = m.materials.at("water");
      CHECK(w.em.kind == axmp::spec::MaterialKind::Vacuum && close(w.em.density, 998.2), "channel: water has no kind, so vacuum, and a density");
      CHECK(w.has_fluid && w.mu_Pa_s.given && !w.mu_Pa_s.is_law() && close(w.mu_Pa_s.value, 1.002e-3), "channel: water's viscosity");
    }
    CHECK(m.boundaries.size() == 3, "channel: three boundaries");
    if (m.boundaries.count("outlet") && m.boundaries.count("walls") && m.boundaries.count("inlet")) {
      const M::Selector& in = m.boundaries.at("inlet");
      CHECK(in.kind == "of" && in.name == "channel" && in.where.count("x") == 1 && in.where.at("x").size() == 1 &&
                in.where.at("x")[0] == 0.0,
            "channel: inlet is {of channel, where x = 0}");
      const M::Selector& out = m.boundaries.at("outlet");
      CHECK(out.where.count("x") == 1 && out.where.at("x").size() == 1 && close(out.where.at("x")[0], 0.04),
            "channel: the outlet's x = L is evaluated");
      const M::Selector& wl = m.boundaries.at("walls");
      CHECK(wl.kind == "of" && wl.where.empty() && wl.except == Strs({"inlet", "outlet"}), "channel: walls is {of channel, except inlet, outlet}");
    }
    CHECK(m.instances.size() == 1 && m.instances.count("flow") == 1, "channel: one instance, 'flow'");
    if (m.instances.count("flow")) {
      const M::Instance& f = m.instances.at("flow");
      CHECK(f.id == "flow" && f.type == "fluid.incompressible" && !f.domains_written && f.domains.empty(), "channel: instance identity and no domains");
      auto opt = [&](const char* k) -> const M::OptionValue* { return f.options.count(k) ? &f.options.at(k) : nullptr; };
      CHECK(opt("equation") && opt("equation")->written && opt("equation")->value == "navier_stokes", "channel: equation is written");
      CHECK(opt("viscous_form") && opt("viscous_form")->written && opt("viscous_form")->value == "laplacian", "channel: viscous_form is written");
      CHECK(opt("convection") && !opt("convection")->written && opt("convection")->value == "standard", "channel: convection is the default");
      CHECK(opt("order") && !opt("order")->written && opt("order")->value == 2, "channel: order is the default 2");
      CHECK(f.conditions.size() == 3, "channel: three conditions");
      if (f.conditions.count("inlet") && f.conditions.count("outlet") && f.conditions.count("walls")) {
        const M::Condition& in = cond(f, "inlet");
        CHECK(in.boundary == "inlet" && in.kind == "inlet" && in.data.count("velocity_m_s") == 1, "channel: the inlet condition");
        if (in.data.count("velocity_m_s")) {
          const M::Datum& v = datum_of(in, "velocity_m_s");
          CHECK(v.field_expression && v.exprs.size() == 2, "channel: the inlet velocity is two compiled expressions");
          // the parabola 6 U y (H - y) / H^2 at x = 0.02, y = H/4, from hand arithmetic (no use of the evaluator)
          const double U = 0.01, H = 0.01, y = 0.0025;
          const double want = 6.0 * U * y * (H - y) / (H * H);
          const double pt[2] = {0.02, y};
          const std::vector<double> got = v.eval(pt);
          CHECK(got.size() == 2 && close(got[0], want, 1e-12) && got[1] == 0.0, "channel: the inlet profile at y = H/4");
          const double mid[2] = {0.0, 0.005};
          const std::vector<double> peak = v.eval(mid);
          CHECK(peak.size() == 2 && close(peak[0], 1.5 * U, 1e-12), "channel: the profile peaks at 1.5 U_mean");
        }
        const M::Condition& out = cond(f, "outlet");
        CHECK(out.kind == "outlet" && out.data.count("pressure_Pa") == 1 && datum_of(out, "pressure_Pa").values == std::vector<double>({0.0}),
              "channel: the outlet pressure is 0");
        CHECK(cond(f, "walls").kind == "wall" && cond(f, "walls").data.empty(), "channel: walls is a wall without data");
      }
    }
    CHECK(m.study.size() == 1 && m.study[0].id == "default" && m.study[0].type == "stationary" && m.study[0].implicit_default &&
              m.study[0].physics == Strs({"flow"}),
          "channel: the implicit stationary step over [flow]");
    CHECK(close(m.mesh.size, 0.000625) && m.mesh.exterior == "none" && !m.mesh.geometry_order_written, "channel: mesh size H/16, exterior none, no geometry order written");
    CHECK(m.outputs.size() == 7, "channel: seven outputs");
    if (m.outputs.size() == 7) {
      CHECK(m.outputs.at("Q_in").kind == "flux" && m.outputs.at("Q_in").of == "volume" && m.outputs.at("Q_in").on == "inlet", "channel: Q_in");
      CHECK(m.outputs.at("tau_max").kind == "maximum" && m.outputs.at("tau_max").of == "wall_shear_mag" && m.outputs.at("tau_max").on == "walls", "channel: tau_max");
      CHECK(m.outputs.at("F_walls").kind == "force" && m.outputs.at("F_walls").on == "walls", "channel: F_walls");
      CHECK(m.outputs.at("dp").kind == "expression" && m.outputs.at("dp").depends_on == Strs({"p_in", "p_out"}), "channel: dp reads p_in and p_out");
      const auto pos = [&](const char* n) { return std::find(m.output_order.begin(), m.output_order.end(), n) - m.output_order.begin(); };
      CHECK(m.output_order.size() == 7 && pos("p_in") < pos("dp") && pos("p_out") < pos("dp"), "channel: dp is evaluated after p_in and p_out");
    }
  }

  // ---- heated cavity (section 4.2)
  {
    const Parsed p = parse(heated());
    const M::ResolvedModel& m = p.m;
    CHECK(m.instances.size() == 2 && m.instances.count("flow") && m.instances.count("heat"), "cavity: flow and heat");
    if (m.instances.count("flow") && m.instances.count("heat")) {
      const M::Instance& f = m.instances.at("flow");
      CHECK(f.conditions.size() == 1 && f.conditions.count("remaining") && cond(f, "remaining").kind == "wall", "cavity: flow has one condition, 'remaining': wall");
      const M::Instance& h = m.instances.at("heat");
      CHECK(h.type == "heat.transfer" && h.conditions.size() == 3, "cavity: heat has hot, cold and remaining");
      if (h.conditions.count("hot")) {
        const double pt[2] = {0.0, 0.0};
        const auto v = datum_of(cond(h, "hot"), "T_K").eval(pt);
        CHECK(cond(h, "hot").kind == "temperature" && v.size() == 1 && close(v[0], 303.15), "cavity: the hot wall is at T_hot");
      }
      CHECK(h.options.count("order") && !h.options.at("order").written && h.options.at("order").value == 2, "cavity: heat order is the default 2");
      CHECK(h.options.count("stabilization") && h.options.at("stabilization").value == "none", "cavity: stabilization defaults to none");
      CHECK(h.options.count("advective_form") && h.options.at("advective_form").value == "convective", "cavity: advective_form defaults to convective");
    }
    CHECK(m.couplings.size() == 2, "cavity: two couplings");
    if (m.couplings.size() == 2) {
      CHECK(m.couplings[0].term == "advection" && m.couplings[0].from == "flow" && m.couplings[0].to == "heat" && m.couplings[0].in.empty() && m.couplings[0].on.empty(), "cavity: advection flow -> heat");
      const M::Coupling& b = m.couplings[1];
      CHECK(b.term == "buoyancy" && b.from == "heat" && b.to == "flow", "cavity: buoyancy heat -> flow");
      CHECK(b.data.count("T_ref_K") == 1 && datum_of(b, "T_ref_K").values.size() == 1 && close(datum_of(b, "T_ref_K").values[0], 0.5 * (303.15 + 293.15), 1e-13),
            "cavity: T_ref_K = (T_hot + T_cold) / 2");
      CHECK(b.data.count("gravity_m_s2") == 1 && datum_of(b, "gravity_m_s2").values == std::vector<double>({0.0, -9.81}), "cavity: gravity (0, -9.81)j");
    }
    CHECK(m.study.size() == 1, "cavity: one study step");
    if (m.study.size() == 1) {
      const M::StudyStep& s = m.study[0];
      CHECK(s.id == "steady" && s.type == "stationary" && !s.implicit_default && s.physics == Strs({"flow", "heat"}), "cavity: the written step");
      CHECK(s.coupling.given && s.coupling.strategy == "monolithic", "cavity: monolithic strategy");
      CHECK(s.continuation.given && s.continuation.scale == "buoyancy" && s.continuation.ramp.empty() &&
                s.continuation.values == std::vector<double>({0.01, 0.1, 1.0}),
            "cavity: buoyancy continuation 0.01, 0.1, 1");
    }
    CHECK(m.outputs.size() == 4 && m.outputs.at("Nu").depends_on == Strs({"Q_hot"}), "cavity: Nu reads Q_hot");
    const auto pos = [&](const char* n) { return std::find(m.output_order.begin(), m.output_order.end(), n) - m.output_order.begin(); };
    CHECK(m.output_order.size() == 4 && pos("Q_hot") < pos("Nu"), "cavity: Nu after Q_hot");
  }

  // ---- plate cell (section 4.3)
  {
    const Parsed p = parse(plate());
    const M::ResolvedModel& m = p.m;
    CHECK(m.reactions.given && m.reactions.electrode.count("plate") == 1, "plate: reactions.electrode.plate");
    if (m.reactions.electrode.count("plate")) {
      const M::ElectrodeReaction& r = m.reactions.electrode.at("plate");
      CHECK(r.kind == "linear" && close(r.i0_A_m2, 10.0) && r.E_eq_V == 0.0 && close(r.alpha_a, 0.5) && close(r.alpha_c, 0.5), "plate: the linear kinetics");
    }
    CHECK(m.materials.count("bath") && m.materials.at("bath").has_electrolyte && close(m.materials.at("bath").kappa_S_m.value, 1.0), "plate: kappa_S_m = 1");
    CHECK(m.instances.count("cell") == 1, "plate: instance 'cell'");
    if (m.instances.count("cell")) {
      const M::Instance& c = m.instances.at("cell");
      CHECK(c.type == "electrochem.current_distribution", "plate: type");
      CHECK(c.options.at("kinetics").written && c.options.at("kinetics").value == "secondary" && c.options.at("order").written &&
                c.options.at("order").value == 2 && c.options.at("T_K").written && close(c.options.at("T_K").value.get<double>(), 298.15),
            "plate: options as written");
      CHECK(c.conditions.size() == 3 && cond(c, "anode").kind == "electrode" && cond(c, "anode").reactions == Strs({"plate"}), "plate: the anode names the reaction 'plate'");
      CHECK(close(datum_of(cond(c, "anode"), "phi_s_V").values[0], 0.05) && datum_of(cond(c, "cathode"), "phi_s_V").values[0] == 0.0, "plate: potentials 0.05 and 0");
      CHECK(cond(c, "remaining").kind == "insulation", "plate: remaining is insulation");
    }
    CHECK(m.outputs.size() == 4 && m.outputs.at("i_avg").depends_on == Strs({"I_anode"}) && m.outputs.at("eta_worst").kind == "extreme", "plate: outputs");
  }

  // ---- busbar (section 4.4)
  {
    const Parsed p = parse(busbar());
    const M::ResolvedModel& m = p.m;
    CHECK(m.symmetry == "3d" && m.regions.size() == 2, "busbar: 3d with two regions");
    CHECK(m.domains.count("metal") && m.domains.at("metal").regions == Strs({"bar", "bolt"}) && m.domains.at("metal").disabled.empty(), "busbar: domain metal = bar, bolt");
    if (m.materials.count("copper") && m.materials.count("titanium")) {
      const M::MaterialProps& c = m.materials.at("copper");
      CHECK(c.has_electric && c.electric_sigma_S_m.is_law(), "busbar: copper's conductivity is a law");
      if (c.electric_sigma_S_m.is_law()) {
        const M::Law& l = *c.electric_sigma_S_m.law;
        CHECK(l.kind == "expression" && l.of == Strs({"T"}) && l.range.count("T") && l.range.at("T") == std::make_pair(250.0, 500.0) &&
                  l.out_of_range == "error" && l.source == "linear resistivity model, temperature coefficient 3.9e-3 1/K",
              "busbar: the expression law, its range, the default out_of_range and its source");
      }
      CHECK(close(c.k_W_mK.value, 400.0) && close(c.cp_J_kgK.value, 385.0) && c.em.mech.given && close(c.em.mech.E_Pa, 110e9) &&
                close(c.em.mech.nu, 0.35) && close(c.alpha_1_K.value, 17e-6) && close(c.em.density, 8960.0),
            "busbar: copper's thermal and mechanical blocks");
      const M::MaterialProps& t = m.materials.at("titanium");
      CHECK(t.electric_sigma_S_m.given && !t.electric_sigma_S_m.is_law() && close(t.electric_sigma_S_m.value, 7.407e5) && close(t.alpha_1_K.value, 9.4e-6), "busbar: titanium");
    } else {
      fail("busbar: materials copper and titanium");
    }
    CHECK(m.boundaries.count("bolt_ends") && m.boundaries.at("bolt_ends").kind == "between" &&
              m.boundaries.at("bolt_ends").between == Strs({"bolt", "exterior"}),
          "busbar: bolt_ends is between bolt and exterior");
    CHECK(m.boundaries.count("skin") && m.boundaries.at("skin").kind == "of" && m.boundaries.at("skin").name == "metal", "busbar: skin is {of metal}");
    CHECK(m.boundaries.count("terminal") && close(m.boundaries.at("terminal").where.at("x")[0], 0.09), "busbar: terminal at x = L");
    CHECK(m.instances.size() == 3, "busbar: three instances (types that are absent in this version are still known)j");
    if (m.instances.size() == 3) {
      const M::Instance& cur = m.instances.at("current");
      CHECK(cur.type == "current.conduction" && cur.domains_written && cur.domains == Strs({"metal"}), "busbar: current on metal");
      const double pt[3] = {0, 0, 0};
      CHECK(cond(cur, "terminal").kind == "potential" && close(datum_of(cond(cur, "terminal"), "V_V").eval(pt)[0], 0.002), "busbar: V_app");
      CHECK(cond(cur, "bolt_ends").kind == "ground" && cond(cur, "remaining").kind == "insulation", "busbar: ground and insulation");
      const M::Condition& sk = cond(m.instances.at("heat"), "skin");
      CHECK(sk.kind == "convection" && close(datum_of(sk, "h_W_m2K").values[0], 5.0) && close(datum_of(sk, "T_ext_K").values[0], 293.15), "busbar: convection on the skin");
      CHECK(m.instances.at("solid").type == "solid.linear" && cond(m.instances.at("solid"), "bolt_ends").kind == "fixed" &&
                cond(m.instances.at("solid"), "remaining").kind == "free",
            "busbar: the solid");
    }
    CHECK(m.couplings.size() == 2 && m.couplings[0].term == "joule_heating" && m.couplings[1].term == "thermal_expansion" &&
              close(datum_of(m.couplings[1], "T_ref_K").values[0], 293.15),
          "busbar: the two terms");
    CHECK(m.study.size() == 1 && m.study[0].physics == Strs({"current", "heat", "solid"}) && !m.study[0].implicit_default &&
              m.study[0].coupling.strategy == "gauss_seidel" && close(m.study[0].coupling.tolerance, 1e-8) && m.study[0].coupling.max_iterations == 25,
          "busbar: the step has no 'physics' key, so it covers every instance, and a gauss_seidel coupling");
    CHECK(m.outputs.size() == 6 && close(m.mesh.size, 0.004) && close(m.mesh.surface_size, 0.0015), "busbar: outputs and mesh");
  }
}

void check_is_model_document_looks_only_at_the_key() {
  CHECK(axmp::spec::is_model_document(json::parse(R"j({"physics": {}})j")), "an empty physics object");
  CHECK(axmp::spec::is_model_document(json::parse(R"j({"physics": 5})j")), "a physics that is not an object is still the key");
  CHECK(axmp::spec::is_model_document(json::parse(R"j({"physics": null})j")), "a null physics is still the key");
  CHECK(axmp::spec::is_model_document(json::parse(R"j({"windings": [], "physics": {}, "name": 1})j")), "nothing else is looked at");
  CHECK(!axmp::spec::is_model_document(json::object()), "an empty object");
  CHECK(!axmp::spec::is_model_document(J(R"j({"Physics": {}})j")), "the key is case-sensitive");
  CHECK(!axmp::spec::is_model_document(J(R"j({"regions": [], "boundaries": {}, "study": []})j")), "the other new sections do not make a model document");
  CHECK(!axmp::spec::is_model_document(json::array({"physics"})), "an array");
  CHECK(!axmp::spec::is_model_document(json("physics")), "a string");
  CHECK(!axmp::spec::is_model_document(json(5)), "a number");
  CHECK(!axmp::spec::is_model_document(json(nullptr)), "null");
  // electromagnetic designs of the repository are not model documents
  CHECK(!axmp::spec::is_model_document(axmp::templates::make("toroid")), "the toroid template");
}

void check_findings_sink() {
  M::Findings f;
  CHECK(!f.has_errors() && f.all().empty(), "an empty sink");
  f.add(M::Finding::Kind::warning, Error("w", "/w"));
  f.add(M::Finding::Kind::deferred, Error("d", "/d", "", "x.code"));
  CHECK(!f.has_errors(), "warnings and deferred findings are not errors");
  f.add(M::Finding::Kind::error, Error("e1", "/e1", "hint", "a.b", "{\"k\":1}"));
  f.add(M::Finding::Kind::unavailable, Error("u"));
  f.add(M::Finding::Kind::error, Error("e2"));
  CHECK(f.has_errors() && f.first_error().message() == "e1" && f.first_error().code() == "a.b" && f.first_error().where() == "/e1",
        "first_error is the first error added, with its code");
  CHECK(f.all().size() == 5, "all() keeps every finding");
  if (f.all().size() == 5) {
    const M::Finding::Kind want[5] = {M::Finding::Kind::warning, M::Finding::Kind::deferred, M::Finding::Kind::error,
                                      M::Finding::Kind::unavailable, M::Finding::Kind::error};
    for (int i = 0; i < 5; ++i) CHECK(f.all()[i].kind == want[i], "all() is in the order of addition");
    CHECK(f.all()[2].error.details() == "{\"k\":1}", "the error is kept whole");
  }
}

// ----------------------------------------------------------------------------------------------- collection

void check_grammar_errors_are_collected_per_section() {
  // three defects in three sections: an unknown material key, a reserved instance id, an unknown output quantity
  {
    json d = channel();
    at(d, "/materials/water/bogus") = 1;
    at(d, "/physics/em") = J(R"j({"type": "heat.transfer"})j");
    at(d, "/outputs/bad") = J(R"j({"kind": "maximum", "of": "nonsense_quantity", "in": "channel"})j");
    const Parsed p = parse(d);
    const auto errs = errors_of(p.f);
    CHECK(errs.size() == 3, "three errors, one per section: got " + std::to_string(errs.size()));
    if (errs.size() == 3) {
      compare_error("collected 1 (materials)", errs[0], W("", "/materials/water", "unknown field 'bogus'"));
      compare_error("collected 2 (physics)", errs[1], W("physics.reserved_id", "/physics/em"));
      compare_error("collected 3 (outputs)", errs[2], W("output.unknown_quantity", "/outputs/bad/of"));
    }
    CHECK(p.f.first_error().where() == "/materials/water", "first_error is the first of the layers");
    try {
      axmp::spec::parse_model(d);
      fail("parse_model did not throw");
    } catch (const Error& e) {
      CHECK(e.where() == "/materials/water" && e.message() == "unknown field 'bogus'", "parse_model throws the first error");
    }
  }
  // two bad materials give two errors, in key order
  {
    json d = channel();
    at(d, "/materials/a") = J(R"j({"kind": "nope"})j");
    at(d, "/materials/b") = J(R"j({"mu_r": 0.5})j");
    const auto errs = errors_of(parse(d).f);
    CHECK(errs.size() == 2, "two bad materials give two errors");
    if (errs.size() == 2) {
      CHECK(errs[0].where() == "/materials/a/kind" && errs[1].where() == "/materials/b/mu_r", "in key order, each with its own pointer");
      CHECK(errs[0].message() == "unknown material kind 'nope'" && errs[1].message() == "mu_r must be >= 1", "with the electromagnetic parser's texts");
    }
  }
  // two bad conditions of one instance give two errors; the instance is not dropped as a whole
  {
    json d = channel();
    at(d, "/physics/flow/boundary_conditions/walls/kind") = "no_slip";
    at(d, "/physics/flow/boundary_conditions/outlet/kind") = "outlt";
    const auto errs = errors_of(parse(d).f);
    CHECK(errs.size() == 2, "two bad conditions give two errors");
    if (errs.size() == 2)
      CHECK(errs[0].where() == "/physics/flow/boundary_conditions/outlet/kind" && errs[1].where() == "/physics/flow/boundary_conditions/walls/kind" &&
                errs[0].code() == "physics.unknown_condition_kind" && errs[1].code() == "physics.unknown_condition_kind",
            "in key order, each with its own pointer");
  }
  // an entry that fails before its name is read keeps its name: the references to it are not reported as unknown
  {
    json d = channel();
    at(d, "/regions/0/bogus") = 1;
    check_one("a region with an unknown key that a domain and the boundaries name", d, W("", "/regions/0", "unknown field 'bogus'"));
    d = load("curves_planar.json");
    d["curves"][0]["bogus"] = 1;
    check_one("a curve with an unknown key that a boundary names", d, W("", "/curves/0", "unknown field 'bogus'"));
    d = heated();
    d["study"][0]["bogus"] = 1;
    d["study"].push_back(J(R"j({"id": "ramp", "type": "transient", "physics": ["flow"], "initial_from": "steady", "time": {"t_end_s": 1.0}})j"));
    check_one("a study step with an unknown key that a later step starts from", d, W("", "/study/0", "unknown field 'bogus'"));
    d = heated();
    d["study"][0].erase("id");
    d["study"].push_back(J(R"j({"id": "ramp", "type": "transient", "physics": ["flow"], "initial_from": "steady", "time": {"t_end_s": 1.0}})j"));
    check_one("a study step without an id that a later step starts from", d, W("", "/study/0", "missing required field 'id'"));
  }
  // a bad parameter: that one error, and nothing from the sections that evaluate expressions
  {
    json d = channel();
    at(d, "/parameters/bad") = "nope+1";
    at(d, "/materials/water/bogus") = 1;
    at(d, "/physics/em") = J(R"j({"type": "heat.transfer"})j");
    at(d, "/outputs/bad") = J(R"j({"kind": "maximum", "of": "nonsense_quantity", "in": "channel"})j");
    at(d, "/mesh/order") = 2;
    const json em = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "parameters": {"bad": "nope+1"}})j");
    same_as_twin("a bad parameter", d, em);
    const Triple t = em_error(em);
    CHECK(t.where == "/parameters/bad", "the twin's pointer is the parameter's");
  }
  // a section that is not of the JSON type the grammar needs: one error, nothing cascades
  {
    json d = channel();
    d["materials"] = 5;
    at(d, "/physics/em") = J(R"j({"type": "heat.transfer"})j");
    at(d, "/outputs/bad") = J(R"j({"kind": "maximum", "of": "nonsense_quantity", "in": "channel"})j");
    check_one("materials that is not an object", d, W("", "/materials", "materials must be an object of name: definition"));
  }
  {
    json d = channel();
    d["regions"] = 5;
    check_one("regions that is not a list", d, W("", "/regions", "regions must be a list"));
  }
  {
    json d = channel();
    d["curves"] = 5;
    check_one("curves that is not a list", d, W("", "/curves", "curves must be a list"));
  }
  {
    json d = channel();
    d["physics"] = json::array();
    check_one("physics that is not an object", d, W("", "/physics", "physics must be an object of id: definition"));
  }
  {
    json d = channel();
    d["domains"] = 5;
    check_one("domains that is not an object", d, W("", "/domains", "domains must be an object of name: list of regions"));
  }
  {
    json d = channel();
    d["boundaries"] = 5;
    check_one("boundaries that is not an object", d, W("", "/boundaries", "boundaries must be an object of name: selector"));
  }
  // a dropped entry keeps its name: a reserved instance id that the study, a coupling and an output name is one error
  {
    json d = channel();
    at(d, "/physics/em") = J(R"j({"type": "heat.transfer"})j");
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "physics": ["flow", "em"]}])j");
    d["couplings"] = J(R"j([{"term": "advection", "from": "flow", "to": "em"}])j");
    at(d, "/outputs/T") = J(R"j({"kind": "maximum", "of": "T", "in": "channel", "physics": "em"})j");
    check_one("a reserved id that is referenced", d, W("physics.reserved_id", "/physics/em"));
  }
  // a material with an error keeps its name: the region that uses it is not reported as using an unknown material
  {
    json d = channel();
    at(d, "/materials/water/mu_r") = 0.5;
    check_one("a failed material that a region uses", d, W("", "/materials/water/mu_r", "mu_r must be >= 1"));
  }
  // a failed region keeps its name: a domain and a boundary that name it are not reported
  {
    json d = channel();
    at(d, "/regions/0/mesh_size") = -1;
    d["domains"] = J(R"j({"wet": ["channel"]})j");
    check_one("a failed region that a domain and the boundaries name", d, W("", "/regions/0/mesh_size", "mesh_size must be > 0 (or absent for automatic)"));
  }
  // a symmetry that is refused: the sections that need it are skipped
  {
    json d = channel();
    d["symmetry"] = "xsec";
    check_one("an unusable symmetry", d, W("model.symmetry_not_offered", "/symmetry"));
  }
}

void check_symmetry_rules() {
  json d = channel();
  d["symmetry"] = "xsec";
  check_one("xsec is refused", d,
            W("model.symmetry_not_offered", "/symmetry", "symmetry 'xsec' is not offered in a model document")  // AUTHOR TEXT
                .with_hint("Use \"planar\" (x, y), \"axisymmetric\" (r, z) or \"3d\". The cross-section analysis of a cavity belongs to an electromagnetic design."));
  d["symmetry"] = "spherical";
  check_one("an unknown symmetry", d, W("", "/symmetry", "symmetry must be 'planar', 'axisymmetric' or '3d'"));  // AUTHOR TEXT

  // planar without mesh.file is accepted in a model document (the electromagnetic parser refuses it)
  d = channel();
  CHECK(!d.contains("mesh") || !d["mesh"].contains("file"), "the channel has no mesh file");
  clean("planar without mesh.file", d);
  const json twin = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "symmetry": "planar",
                          "regions": [{"name": "a", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}}]})j");
  CHECK(em_error(twin).threw, "the electromagnetic parser refuses planar without mesh.file (the reference of this test)j");
  at(d, "/mesh/file") = "m.msh";
  at(d, "/mesh/info_file") = "m.json";
  const Parsed p = parse(d);
  CHECK(!p.f.has_errors(), "planar with mesh.file");

  // a planar region needs a shape in the (x, y) plane (AUTHOR TEXT; the axisymmetric and 3d texts are the electromagnetic parser's)
  d = channel();
  d["regions"][0].erase("shape");
  check_one("a planar region without a shape", d, W("", "/regions/0", "planar regions need a 'shape' in the (x, y) plane"));

  // the default is axisymmetric, and the coordinates are r, z
  json a = pipe();
  a.erase("symmetry");
  const Parsed pa = parse(a);
  CHECK(!pa.f.has_errors() && pa.m.symmetry == "axisymmetric", "the symmetry defaults to axisymmetric");
  // planar coordinates are x and y; a planar region written with r and z is the electromagnetic cross-section grammar's
  // business (accepted there), so only x, y are asserted: the channel's x/y rectangle parsed above
  // 3d: a region needs a solid (the electromagnetic text)
  const json em3 = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "symmetry": "3d",
                         "regions": [{"name": "b", "shape": {"rect": {"r": [0, 1], "z": [0, 1]}}}]})j");
  json m3 = em3;
  m3["physics"] = J(R"j({"h": {"type": "heat.transfer"}})j");
  same_as_twin("a 3d region without a solid", m3, em3);
}

void check_mesh_keys() {
  json d = channel();
  at(d, "/mesh/order") = 2;
  check_one("mesh.order", d,
            W("model.mesh_order", "/mesh/order",
              // the note, section 3.3: the whole sentence is the message
              "`mesh.order` is the field order of an electromagnetic solve. In a model document write the field order per physics "
              "(`/physics/flow/options/order`) and the geometry order in `mesh.geometry_order`"));

  d = channel();
  at(d, "/mesh/exterior") = "none";
  at(d, "/mesh/refinements") = 2;
  at(d, "/mesh/geometry_order") = 1;
  at(d, "/mesh/surface_size") = "H/32";
  at(d, "/mesh/threads") = 2;
  if (clean("the model's mesh keys", d)) {
    const M::MeshKeys& k = parse(d).m.mesh;
    CHECK(k.exterior == "none" && k.refinements == 2 && k.geometry_order == 1 && k.geometry_order_written && close(k.surface_size, 0.01 / 32) && k.threads == 2,
          "the mesh keys are read");
  }

  d = channel();
  at(d, "/mesh/exterior") = "open";
  check_one("mesh.exterior open", d,
            W("model.exterior_not_offered", "/mesh/exterior", "mesh.exterior \"open\" is not offered in a model document")  // AUTHOR TEXT
                .with_hint("Only \"none\" is offered: the union of the enabled regions is the whole domain, with no air and no infinity shell."));

  d = channel();
  at(d, "/mesh/adapt") = J(R"j({"cycles": 1})j");
  check_one("mesh.adapt", d,
            W("model.mesh_key_not_offered", "/mesh/adapt", "mesh.adapt is not offered in a model document")  // AUTHOR TEXT
                .with_hint("Use \"refinements\" for uniform refinement, or a smaller \"size\"."));

  d = channel();
  at(d, "/mesh/geometry_order") = 2;
  at(d, "/mesh/file") = "m.msh";
  at(d, "/mesh/info_file") = "m.json";
  check_one("geometry_order beside file", d,
            W("model.geometry_order_from_file", "/mesh/geometry_order",
              // the note, section 3.3
              "the geometry order of an imported mesh is read from the file; remove `mesh.geometry_order`"));

  d = channel();
  at(d, "/mesh/geometry_order") = 3;
  check_one("geometry_order 3", d, W("", "/mesh/geometry_order", "mesh.geometry_order must be 1 or 2"));  // AUTHOR TEXT

  // an unknown key lists the model document's mesh keys, section 3.3: size, surface_size, geometry_order, refinements,
  // exterior, file, info_file, threads (not the electromagnetic list)
  const std::set<std::string> mesh_keys = {"size", "surface_size", "geometry_order", "refinements", "exterior", "file", "info_file", "threads"};
  d = channel();
  at(d, "/mesh/bogus") = 1;
  check_one("an unknown mesh key", d, W("", "/mesh", "unknown field 'bogus'").with_hint(only_list(mesh_keys)));
  d = channel();
  at(d, "/mesh/domain_radius") = 1.0;  // a key of the electromagnetic mesh section
  check_one("an electromagnetic mesh key", d, W("", "/mesh", "unknown field 'domain_radius'").with_hint(only_list(mesh_keys)));
}

void check_em_sections_are_refused() {
  for (const std::string& s : kEmSections) {
    json d = channel();
    d[s] = json::object();
    check_one("section " + s, d,
              W("model.em_sections_not_combinable", "/" + s, "'" + s + "' cannot be combined with 'physics' yet").with_hint(kEmSectionHint));
  }
  json all = channel();
  for (const std::string& s : kEmSections) all[s] = json::object();
  const auto errs = errors_of(parse(all).f);
  std::set<std::string> wheres;
  for (const auto& e : errs) {
    wheres.insert(e.where());
    CHECK(e.code() == "model.em_sections_not_combinable", "every refusal carries the code");
  }
  CHECK(errs.size() == 15 && wheres.size() == 15, "all fifteen are reported, each with its own pointer");
  // the other top-level keys are listed in the hint of an unknown one
  json u = channel();
  u["bogus"] = 1;
  const std::set<std::string> top = {"schema", "name", "description", "notes", "symmetry", "parameters", "regions", "curves", "materials",
                                     "domains", "boundaries", "reactions", "mesh", "physics", "couplings", "study", "outputs"};
  check_one("an unknown top-level key", u, W("", "", "unknown field 'bogus'").with_hint(only_list(top)));
}

void check_identity() {
  json d = channel();
  d["schema"] = "axionmultiphysics.design/2";
  const json em = J(R"j({"schema": "axionmultiphysics.design/2", "name": "t"})j");
  same_as_twin("an unsupported schema", d, em);
  d = channel();
  d.erase("name");
  same_as_twin("a missing name", d, J(R"j({"schema": "axionmultiphysics.design/1"})j"));
}

void check_names() {
  // ---- instance ids: [a-z][a-z0-9_]*  (section 3.6)
  for (const char* id : {"Flow", "1x", "my-flow", "flow x"}) {
    json d = channel();
    json inst = d["physics"]["flow"];
    d["physics"] = json::object();
    d["physics"][id] = inst;
    check_one(std::string("the id '") + id + "'", d,
              W("physics.invalid_id", std::string("/physics/") + id, std::string("'") + id + "' is not a valid physics id")  // AUTHOR TEXT
                  .with_hint("An id starts with a lowercase letter and continues with lowercase letters, digits and underscores ([a-z][a-z0-9_]*).")
                  .with_details(json{{"id", id}}));
  }
  {
    json d = channel();
    json inst = d["physics"]["flow"];
    d["physics"] = json::object();
    d["physics"]["f_2"] = inst;
    clean("the id 'f_2'", d);
  }
  // ---- the eight reserved ids (section 3.6)
  const Strs reserved = {"em", "mechanics", "thermal", "electrostatics", "hts", "fullwave", "cavity", "mesh_motion"};
  CHECK(M::registry().reserved_ids == reserved, "the registry lists the eight reserved ids of section 3.6");
  for (const std::string& id : reserved) {
    json d = channel();
    json inst = d["physics"]["flow"];
    d["physics"] = json::object();
    d["physics"][id] = inst;
    check_one("the reserved id '" + id + "'", d,
              W("physics.reserved_id", "/physics/" + id, "'" + id + "' is a reserved physics id")  // AUTHOR TEXT
                  .with_hint("The ids em, mechanics, thermal, electrostatics, hts, fullwave, cavity, mesh_motion are kept for the electromagnetic "
                             "sections and for later physics types; choose another id.")
                  .with_details(json{{"id", id}, {"reserved", jarr(reserved)}}));
  }

  // ---- one namespace for regions, curves, domains and boundaries: a collision names both (section 3.4)
  const std::string hint = "Regions, curves, domains and boundaries share one namespace; rename one of them.";  // AUTHOR TEXT
  {
    json d = channel();
    at(d, "/boundaries/channel") = J(R"j({"of": "channel"})j");
    check_one("a boundary named like a region", d,
              W("boundary.name_collision", "/boundaries/channel", "name 'channel' is used by a region and a boundary")  // AUTHOR TEXT
                  .with_hint(hint)
                  .with_details(J(R"j({"name": "channel", "first": {"kind": "region", "at": "/regions/0/name"},
                                      "second": {"kind": "boundary", "at": "/boundaries/channel"}})j")));
  }
  {
    json d = channel();
    d["domains"] = J(R"j({"channel": ["channel"]})j");
    check_one("a domain named like a region", d,
              W("boundary.name_collision", "/domains/channel", "name 'channel' is used by a region and a domain")
                  .with_details(J(R"j({"first": {"kind": "region", "at": "/regions/0/name"}, "second": {"kind": "domain", "at": "/domains/channel"}})j")));
  }
  {
    json d = channel();
    d["domains"] = J(R"j({"walls": ["channel"]})j");
    check_one("a domain named like a boundary", d,
              W("boundary.name_collision", "/boundaries/walls", "name 'walls' is used by a domain and a boundary")
                  .with_details(J(R"j({"first": {"kind": "domain", "at": "/domains/walls"}, "second": {"kind": "boundary", "at": "/boundaries/walls"}})j")));
  }
  {
    json d = channel();
    d["regions"].push_back(d["regions"][0]);
    check_one("two regions with one name", d,
              W("boundary.name_collision", "/regions/1/name", "name 'channel' is used by two regions")
                  .with_details(J(R"j({"first": {"kind": "region", "at": "/regions/0/name"}, "second": {"kind": "region", "at": "/regions/1/name"}})j")));
  }
  {
    json d = load("curves_planar.json");
    d["curves"][0]["name"] = "slab";
    d["boundaries"]["section"]["curve"] = "slab";  // the dropped curve keeps its name, so the boundary that names it is not reported
    check_one("a curve named like a region", d,
              W("boundary.name_collision", "/curves/0/name", "name 'slab' is used by a region and a curve")
                  .with_details(J(R"j({"first": {"kind": "region", "at": "/regions/0/name"}, "second": {"kind": "curve", "at": "/curves/0/name"}})j")));
  }
  {
    // a disabled region keeps its name in the namespace, as in the electromagnetic parser
    json d = channel();
    json off = d["regions"][0];
    off["name"] = "spare";
    off["enabled"] = false;
    d["regions"].push_back(off);
    clean("a disabled region", d);
    d["boundaries"]["spare"] = J(R"j({"of": "channel"})j");
    check_one("a boundary named like a disabled region", d, W("boundary.name_collision", "/boundaries/spare"));
  }

  // ---- the reserved region names air and infinity_shell, with the electromagnetic parser's text
  for (const char* name : {"air", "infinity_shell"}) {
    json em = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "symmetry": "axisymmetric",
                    "regions": [{"name": "x", "shape": {"rect": {"r": [0, 1], "z": [0, 1]}}}]})j");
    em["regions"][0]["name"] = name;
    json mod = em;
    mod["physics"] = J(R"j({"h": {"type": "heat.transfer"}})j");
    same_as_twin(std::string("the region name ") + name, mod, em);
    const Triple t = em_error(em);
    CHECK(t.message == std::string("'") + name + "' is a reserved region name", "the reference text is the electromagnetic parser's");
  }
  {
    json em = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "symmetry": "axisymmetric",
                    "curves": [{"name": "air", "path": {"segment": [[0, 0], [1, 1]]}}]})j");
    json mod = em;
    mod["curves"][0].erase("material");
    mod["physics"] = J(R"j({"h": {"type": "heat.transfer"}})j");
    same_as_twin("the curve name air", mod, em);
  }

  // ---- unknown names in except and union (boundary.unknown_name); the valid names are the document's boundaries
  const Strs bnames = {"inlet", "outlet", "walls"};
  {
    json d = channel();
    at(d, "/boundaries/walls/except") = J(R"j(["inlet", "nope"])j");
    check_one("an unknown name in except", d,
              W("boundary.unknown_name", "/boundaries/walls/except/1", "unknown boundary 'nope'")  // AUTHOR TEXT
                  .with_hint("The boundaries of this document are: inlet, outlet, walls.")
                  .with_details(json{{"name", "nope"}, {"valid", jarr(bnames)}}));
  }
  {
    json d = channel();
    at(d, "/boundaries/u") = J(R"j({"union": ["inlet", "nope"]})j");
    check_one("an unknown name in union", d, W("boundary.unknown_name", "/boundaries/u/union/1", "unknown boundary 'nope'").with_hint_has("The boundaries of this document are: inlet, outlet, u, walls."));
  }
  {
    json d = channel();
    at(d, "/physics/flow/boundary_conditions/nope") = J(R"j({"kind": "wall"})j");
    check_one("an unknown boundary condition key", d,
              W("boundary.unknown_name", "/physics/flow/boundary_conditions/nope", "unknown boundary 'nope'").with_hint("The boundaries of this document are: inlet, outlet, walls."));
  }
  {
    json d = channel();
    at(d, "/boundaries/walls/of") = "chanel";
    check_one("an unknown domain in of", d,
              W("domain.unknown_name", "/boundaries/walls/of", "unknown domain or region 'chanel'")  // AUTHOR TEXT
                  .with_hint("The regions and domains of this document are: channel.")
                  .with_details(J(R"j({"name": "chanel", "valid": ["channel"]})j")));
  }
  {
    json d = channel();
    at(d, "/physics/flow/domains") = J(R"j(["channel", "pond"])j");
    check_one("an unknown domain in an instance", d, W("domain.unknown_name", "/physics/flow/domains/1", "unknown domain or region 'pond'"));
  }
  {
    json d = channel();
    d["domains"] = J(R"j({"wet": ["channel", "pond"]})j");
    check_one("an unknown region in a domain", d, W("domain.unknown_name", "/domains/wet/1", "unknown domain or region 'pond'"));
  }
  {
    json d = channel();
    at(d, "/outputs/T") = J(R"j({"kind": "maximum", "of": "u_mag", "in": "pond"})j");
    check_one("an unknown domain in an output", d, W("domain.unknown_name", "/outputs/T/in", "unknown domain or region 'pond'"));
  }
  {
    json d = channel();
    d["physics"]["flow"]["sources"] = J(R"j([{"kind": "body_force", "in": "pond", "force_N_m3": [0, 1]}])j");
    check_one("an unknown domain in a source", d, W("domain.unknown_name", "/physics/flow/sources/0/in", "unknown domain or region 'pond'"));
  }

  // ---- reserved names (section 5.2): axis never written, exterior only second in between, remaining only a condition key
  const std::string axis_msg = "'axis' is the part of the boundary on r = 0; it is created automatically and is never written";  // AUTHOR TEXT
  const std::string ext_msg = "'exterior' can only be the second entry of 'between'";                                              // AUTHOR TEXT
  const std::string rem_msg = "'remaining' can only be a key of 'boundary_conditions'";                                            // AUTHOR TEXT
  {
    json d = channel();
    at(d, "/boundaries/axis") = J(R"j({"of": "channel"})j");
    check_one("a boundary named axis", d,
              W("boundary.reserved_name", "/boundaries/axis", axis_msg).with_hint("Remove it: every physics applies its one valid axis condition there."));
  }
  {
    json d = channel();
    at(d, "/boundaries/exterior") = J(R"j({"of": "channel"})j");
    check_one("a boundary named exterior", d,
              W("boundary.reserved_name", "/boundaries/exterior", ext_msg).with_hint("Use it as the second entry of a 'between' selector, as in {\"between\": [A, \"exterior\"]}."));
  }
  {
    json d = channel();
    at(d, "/boundaries/remaining") = J(R"j({"of": "channel"})j");
    check_one("a boundary named remaining", d,
              W("boundary.reserved_name", "/boundaries/remaining", rem_msg).with_hint("Use it as a key of 'boundary_conditions'."));
  }
  {
    json d = channel();
    at(d, "/boundaries/b") = J(R"j({"between": ["exterior", "channel"]})j");
    check_one("exterior first in between", d, W("boundary.reserved_name", "/boundaries/b/between/0", ext_msg));
  }
  {
    json d = channel();
    at(d, "/boundaries/walls/except") = J(R"j(["inlet", "remaining"])j");
    check_one("remaining in except", d, W("boundary.reserved_name", "/boundaries/walls/except/1", rem_msg));
  }
  {
    json d = channel();
    at(d, "/physics/flow/boundary_conditions/axis") = J(R"j({"kind": "wall"})j");
    check_one("an axis condition", d, W("boundary.reserved_name", "/physics/flow/boundary_conditions/axis", axis_msg));
  }
  {
    json d = channel();
    at(d, "/physics/flow/boundary_conditions/exterior") = J(R"j({"kind": "wall"})j");
    check_one("an exterior condition", d, W("boundary.reserved_name", "/physics/flow/boundary_conditions/exterior", ext_msg));
  }
  {
    // remaining is the condition key it is meant to be
    json d = channel();
    d["physics"]["flow"]["boundary_conditions"] = J(R"j({"inlet": {"kind": "inlet", "velocity_m_s": [1, 0]}, "remaining": {"kind": "wall"}})j");
    clean("remaining as a condition key", d);
  }

  // ---- the reserved names are names of nothing else: regions, curves and domains share the namespace of the boundaries
  // (sections 3.4 and 5.2).  AUTHOR TEXT.  A reference to the refused name adds no finding of its own.
  {
    const std::string hint = "The names axis, exterior and remaining belong to the boundary grammar: 'axis' is the boundary on r = 0, 'exterior' the "
                             "outside of the model in a 'between' selector, and 'remaining' the boundary that no condition names. Rename the ";
    for (const char* name : {"axis", "exterior", "remaining"}) {
      json d = channel();
      d["regions"].push_back(d["regions"][0]);
      d["regions"][1]["name"] = name;
      at(d, "/boundaries/extra") = json{{"of", name}};
      check_one(std::string("a region named ") + name, d,
                W("boundary.reserved_name", "/regions/1/name", std::string("'") + name + "' is a reserved name and cannot be the name of a region")
                    .with_hint(hint + "region.")
                    .with_details(json{{"name", name}, {"kind", "region"}}));
      d = channel();
      d["domains"] = json::object();
      d["domains"][name] = J(R"j(["channel"])j");
      at(d, "/physics/flow/domains") = json::array({name});
      check_one(std::string("a domain named ") + name, d,
                W("boundary.reserved_name", std::string("/domains/") + name, std::string("'") + name + "' is a reserved name and cannot be the name of a domain")
                    .with_hint(hint + "domain."));
      d = load("curves_planar.json");
      d["curves"][0]["name"] = name;
      at(d, "/boundaries/section/curve") = name;
      check_one(std::string("a curve named ") + name, d,
                W("boundary.reserved_name", "/curves/0/name", std::string("'") + name + "' is a reserved name and cannot be the name of a curve").with_hint(hint + "curve."));
    }
    // where a domain is expected, a reserved name is refused as such, with the texts of the boundary grammar
    json d = channel();
    at(d, "/boundaries/extra") = J(R"j({"of": "axis"})j");
    check_one("of: axis", d, W("boundary.reserved_name", "/boundaries/extra/of", axis_msg));
    d = channel();
    at(d, "/physics/flow/sources") = J(R"j([{"kind": "body_force", "force_N_m3": [0, 0], "in": "exterior"}])j");
    check_one("a source in exterior", d, W("boundary.reserved_name", "/physics/flow/sources/0/in", ext_msg));
    d = channel();
    at(d, "/physics/flow/domains") = J(R"j(["remaining"])j");
    check_one("an instance in remaining", d, W("boundary.reserved_name", "/physics/flow/domains/0", rem_msg));
    d = channel();
    d["domains"] = J(R"j({"all": ["channel", "exterior"]})j");
    check_one("a domain that lists exterior", d, W("boundary.reserved_name", "/domains/all/1", ext_msg));
  }
  // ---- a domain that lists a disabled region loses it, and the resolved model says so (section 3.4)
  {
    json d = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "symmetry": "planar",
      "regions": [{"name": "a", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}},
                  {"name": "b", "shape": {"rect": {"x": [1, 2], "y": [0, 1]}}, "enabled": false},
                  {"name": "c", "shape": {"rect": {"x": [2, 3], "y": [0, 1]}}, "enabled": "1-1+1"}],
      "domains": {"all": ["a", "b", "c"]},
      "physics": {"h": {"type": "heat.transfer", "domains": ["all"]}}})j");
    const Parsed p = parse(d);
    CHECK(!p.f.has_errors(), "a domain with a disabled region parses");
    CHECK(p.m.disabled_regions == Strs({"b"}) && p.m.regions.size() == 2 && p.m.regions[0].name == "a" && p.m.regions[1].name == "c", "enabled: false and an expression that is not zero");
    CHECK(p.m.domains.count("all") && p.m.domains.at("all").regions == Strs({"a", "c"}) && p.m.domains.at("all").disabled == Strs({"b"}),
          "the domain loses 'b' and the resolved model says so");
    CHECK(p.m.instances.at("h").domains_written && p.m.instances.at("h").domains == Strs({"all"}), "the instance's domains are as written");
  }
}

void check_curve_keys_in_a_model_document() {
  clean("curves_planar", load("curves_planar.json"));
  clean("curves_axisymmetric", load("curves_axisymmetric.json"));
  {
    const Parsed p = parse(load("curves_planar.json"));
    CHECK(p.m.curves.size() == 3 && p.m.curves[0].name == "mid" && p.m.curves[1].name == "cut" && p.m.curves[2].name == "zigzag", "the three curves");
    if (p.m.curves.size() == 3) {
      CHECK(close(p.m.curves[0].mesh_size, 0.001) && close(p.m.curves[1].surface_size, 0.0005), "mesh_size and surface_size are read");
      CHECK(p.m.curves[0].path.kind == axmp::spec::Path::Kind::Segment && p.m.curves[1].path.kind == axmp::spec::Path::Kind::Arc &&
                p.m.curves[2].path.kind == axmp::spec::Path::Kind::Polyline,
            "a segment, an arc and a polyline");
    }
  }
  // material and thickness are refused with the model's curve keys
  const std::set<std::string> curve_keys = {"name", "path", "mesh_size", "surface_size", "notes"};
  for (const char* key : {"material", "thickness"}) {
    json d = load("curves_planar.json");
    d["curves"][0][key] = key == std::string("material") ? json("superconductor") : json(0.002);
    check_one(std::string("a curve with '") + key + "'", d, W("", "/curves/0", std::string("unknown field '") + key + "'").with_hint(only_list(curve_keys)));
  }
  json d = load("curves_axisymmetric.json");
  d["curves"][1]["bogus"] = 1;
  check_one("an unknown curve key", d, W("", "/curves/1", "unknown field 'bogus'").with_hint(only_list(curve_keys)));
  d = load("curves_axisymmetric.json");
  d["curves"][0].erase("name");
  check_one("a curve without a name", d, W("", "/curves/0", "missing required field 'name'"));
  d = load("curves_axisymmetric.json");
  d["curves"][0]["path"] = J(R"j({"helix": 1})j");
  same_as_twin("an unknown path type", d, [&] {
    json em = d;
    em.erase("physics"); em.erase("boundaries"); em.erase("outputs");
    return em;
  }());
}

void check_selector_grammar() {
  const json base = J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "sel", "symmetry": "planar",
    "parameters": {"L": 2.0, "H": 1.0},
    "regions": [{"name": "a", "shape": {"rect": {"x": [0, "L"], "y": [0, "H"]}}},
                {"name": "b", "shape": {"rect": {"x": ["L", "2*L"], "y": [0, "H"]}}}],
    "curves": [{"name": "mid", "path": {"segment": [["L/2", 0], ["L/2", "H"]]}}],
    "boundaries": {
      "b_of":      {"of": "a", "where": {"x": 0}},
      "b_range":   {"of": "a", "where": {"x": [0, "L"], "y": "H"}, "tolerance_m": 1e-6},
      "b_between": {"between": ["a", "b"]},
      "b_ext":     {"between": ["a", "exterior"]},
      "b_curve":   {"curve": "mid"},
      "b_union":   {"union": ["b_of", "b_between", {"of": "b"}]},
      "b_diff":    {"difference": [{"of": "a"}, "b_of"]},
      "b_ref":     "b_of",
      "b_except":  {"of": "a", "except": ["b_of", "b_ext"]}},
    "physics": {"heat": {"type": "heat.transfer"}}})j");
  const Parsed p = parse(base);
  CHECK(!p.f.has_errors(), "the six selector kinds and both filters parse");
  for (const auto& e : errors_of(p.f)) std::printf("    [%s] %s [at %s]\n", e.code().c_str(), e.message().c_str(), e.where().c_str());
  const auto& B = p.m.boundaries;
  if (B.size() == 9) {
    CHECK(B.at("b_of").kind == "of" && B.at("b_of").name == "a" && B.at("b_of").where.at("x") == std::vector<double>({0.0}), "of + an equality");
    CHECK(B.at("b_range").where.at("x") == std::vector<double>({0.0, 2.0}) && B.at("b_range").where.at("y") == std::vector<double>({1.0}) &&
              B.at("b_range").tolerance_m == 1e-6,
          "a range, an equality by expression, and the tolerance");
    CHECK(B.at("b_between").kind == "between" && B.at("b_between").between == Strs({"a", "b"}), "between");
    CHECK(B.at("b_ext").between == Strs({"a", "exterior"}), "between with exterior");
    CHECK(B.at("b_curve").kind == "curve" && B.at("b_curve").name == "mid", "curve");
    CHECK(B.at("b_union").kind == "union" && B.at("b_union").operands.size() == 3 && B.at("b_union").operands[0].kind == "ref" &&
              B.at("b_union").operands[0].name == "b_of" && B.at("b_union").operands[2].kind == "of" && B.at("b_union").operands[2].name == "b",
          "union of names and selectors");
    CHECK(B.at("b_diff").kind == "difference" && B.at("b_diff").operands.size() == 2 && B.at("b_diff").operands[0].kind == "of" && B.at("b_diff").operands[1].kind == "ref",
          "difference");
    CHECK(B.at("b_ref").kind == "ref" && B.at("b_ref").name == "b_of", "a bare name of another boundary");
    CHECK(B.at("b_except").except == Strs({"b_of", "b_ext"}), "except");
    CHECK(B.at("b_of").written == J(R"j({"of": "a", "where": {"x": 0}})j"), "the selector as written is kept");
  } else {
    fail("nine boundaries expected, got " + std::to_string(B.size()));
  }

  // the group selector, and of / between / curve on a document with mesh.file: deferred, not refused
  {
    json d = base;
    d["boundaries"] = J(R"j({"g": {"group": "left"}, "o": {"of": "a", "where": {"x": 0}}, "t": {"between": ["a", "b"]}, "u": {"union": ["g", "g"]}})j");
    d["mesh"] = J(R"j({"file": "m.msh", "info_file": "m.json"})j");
    const Parsed q = parse(d);
    CHECK(!q.f.has_errors(), "no error for selectors on an imported mesh");
    const auto def = of_kind(q.f, M::Finding::Kind::deferred);
    CHECK(def.size() == 2, "two deferred findings (of, between): got " + std::to_string(def.size()));
    if (def.size() == 2) {
      compare_error("deferred of", def[0], W("boundary.imported_mesh_selector", "/boundaries/o",  // AUTHOR TEXT (code, message, hint)
                                                "boundary 'o' uses 'of' on an imported mesh, where only 'group', 'union' and 'difference' are available")
                                              .with_hint("Select a group of the mesh's info file with {\"group\": NAME}."));
      compare_error("deferred between", def[1], W("boundary.imported_mesh_selector", "/boundaries/t",
                                                 "boundary 't' uses 'between' on an imported mesh, where only 'group', 'union' and 'difference' are available"));
    }
    CHECK(q.m.boundaries.count("g") && q.m.boundaries.at("g").kind == "group" && q.m.boundaries.at("g").name == "left", "the group selector");
  }
  // errors of the selector grammar (AUTHOR TEXT: the code and the messages)
  {
    json d = base;
    d["boundaries"] = J(R"j({"x": {"of": "a", "curve": "mid"}})j");
    check_one("two selector keys", d, W("boundary.selector_grammar", "/boundaries/x", "a boundary selector needs exactly one of of, between, curve, group, union, difference"));
    d["boundaries"] = J(R"j({"x": {"where": {"x": 0}}})j");
    check_one("no selector key", d, W("boundary.selector_grammar", "/boundaries/x", "a boundary selector needs exactly one of of, between, curve, group, union, difference"));
    d["boundaries"] = J(R"j({"x": {"curve": "mid", "where": {"x": 0}}})j");
    check_one("where on a curve", d, W("boundary.selector_grammar", "/boundaries/x/where", "'where' applies to 'of' and 'between' only"));
    d["boundaries"] = J(R"j({"x": {"group": "g", "except": ["x"]}})j");
    check_one("except on a group", d, W("boundary.selector_grammar", "/boundaries/x/except", "'except' applies to 'of' and 'between' only"));
    d["boundaries"] = J(R"j({"x": {"of": "a", "where": {"z": 0}}})j");
    check_one("a coordinate that the symmetry does not have", d,
              W("boundary.selector_grammar", "/boundaries/x/where/z", "unknown coordinate 'z' in 'where': the coordinates of a planar model are x, y"));
    d["boundaries"] = J(R"j({"x": {"of": "a", "bogus": 1}})j");
    check_one("an unknown selector key", d,
              W("", "/boundaries/x", "unknown field 'bogus'")
                  .with_hint(only_list({"of", "between", "curve", "group", "union", "difference", "where", "except", "tolerance_m"})));
    d["boundaries"] = J(R"j({"x": {"curve": "nothing"}})j");
    check_one("an unknown curve", d, W("boundary.unknown_name", "/boundaries/x/curve", "unknown curve 'nothing'").with_hint("The curves of this document are: mid."));
    d["boundaries"] = J(R"j({"x": "nothing"})j");
    check_one("an unknown boundary as a bare name", d, W("boundary.unknown_name", "/boundaries/x", "unknown boundary 'nothing'"));
  }
  // coordinates of the other symmetries
  {
    json d = pipe();
    d["boundaries"]["bad"] = J(R"j({"of": "pipe", "where": {"x": 0}})j");
    check_one("x in an axisymmetric model", d, W("boundary.selector_grammar", "/boundaries/bad/where/x", "unknown coordinate 'x' in 'where': the coordinates of an axisymmetric model are r, z"));
    d = busbar();
    d["boundaries"]["bad"] = J(R"j({"of": "bar", "where": {"w": 0}})j");
    check_one("w in a 3d model", d, W("boundary.selector_grammar", "/boundaries/bad/where/w", "unknown coordinate 'w' in 'where': the coordinates of a 3d model are x, y, z"));
  }
}

void check_unknown_type_option_kind() {
  const std::string F = "fluid.incompressible";
  // ---- type
  {
    json d = channel();
    d["physics"]["flow"]["type"] = "laminar_flow";
    check_one("a COMSOL type name", d,
              W("physics.unknown_type", "/physics/flow/type", "unknown physics type 'laminar_flow'")  // AUTHOR TEXT
                  .with_hint("Use 'fluid.incompressible' (the name here for 'laminar_flow'). The physics types are: " + join(type_names()) + ".")
                  .with_details(json{{"type", "laminar_flow"}, {"use", F}, {"valid", jarr(type_names())}, {"near", json::array()}}));
    d["physics"]["flow"]["type"] = "fluid.incompressibl";
    check_one("a near type name", d,
              W("physics.unknown_type", "/physics/flow/type", "unknown physics type 'fluid.incompressibl'")
                  .with_hint("Did you mean 'fluid.incompressible'? The physics types are: " + join(type_names()) + ".")
                  .with_details(json{{"near", json::array({F})}}));
    d["physics"]["flow"]["type"] = "plasma";
    check_one("an unknown type", d, W("physics.unknown_type", "/physics/flow/type", "unknown physics type 'plasma'").with_hint("The physics types are: " + join(type_names()) + "."));
    d["physics"]["flow"].erase("type");
    check_one("a missing type", d, W("", "/physics/flow", "missing required field 'type'"));
  }
  // the types that are absent or name-only in this version parse
  for (const char* t : {"chem.reactor", "electrochem.porous_electrode", "species.transport", "solid.linear", "current.conduction"}) {
    json d = channel();
    d["physics"] = json::object();
    d["physics"]["x"] = json{{"type", t}};
    d.erase("outputs");
    clean(std::string("the type ") + t, d);
  }
  // ---- options
  {
    json d = channel();
    d["physics"]["flow"]["options"]["swirl"] = true;
    check_one("the swirl option", d,
              W("physics.unknown_option", "/physics/flow/options/swirl", "unknown option 'swirl' for 'fluid.incompressible'")  // AUTHOR TEXT
                  .with_hint("There is no 'swirl' option: an axisymmetric flow always carries the three components (u_r, u_z, u_theta), and a "
                             "swirl-free problem is posed with zero theta data. " + list_sentence("options", F, option_names(F)))
                  .with_details(json{{"option", "swirl"}, {"valid", jarr(option_names(F))}}));
    d = channel();
    d["physics"]["flow"]["options"]["equaton"] = "stokes";
    check_one("a near option", d,
              W("physics.unknown_option", "/physics/flow/options/equaton", "unknown option 'equaton' for 'fluid.incompressible'")
                  .with_hint("Did you mean 'equation'? " + list_sentence("options", F, option_names(F)))
                  .with_details(json{{"near", json::array({"equation"})}}));
    d = channel();
    d["physics"]["flow"]["options"]["equation"] = "euler";
    check_one("an unknown option value", d,
              W("physics.unknown_option", "/physics/flow/options/equation", "'euler' is not a value of the option 'equation' for 'fluid.incompressible'")
                  .with_hint("The values of 'equation' are: stokes, navier_stokes.")
                  .with_details(J(R"j({"option": "equation", "value": "euler", "valid": ["stokes", "navier_stokes"]})j")));
    d = channel();
    d["physics"]["flow"]["options"]["order"] = 3;
    check_one("an unknown integer option value", d,
              W("physics.unknown_option", "/physics/flow/options/order", "'3' is not a value of the option 'order' for 'fluid.incompressible'")
                  .with_details(J(R"j({"option": "order", "value": "3", "valid": ["2"]})j")));
    // a numeric option may be an expression over the parameters; an integer option is written as a number
    d = plate();
    d["parameters"]["T_bath"] = 300.0;
    d["physics"]["cell"]["options"]["T_K"] = "T_bath";
    if (clean("a numeric option written as an expression", d)) {
      const auto& o = parse(d).m.instances.at("cell").options.at("T_K");
      CHECK(o.written && close(o.value.get<double>(), 300.0), "its value is evaluated");
    }
  }
  // ---- condition kinds
  {
    json d = channel();
    d["physics"]["flow"]["boundary_conditions"]["walls"]["kind"] = "no_slip";
    check_one("a COMSOL kind name", d,
              W("physics.unknown_condition_kind", "/physics/flow/boundary_conditions/walls/kind", "unknown condition kind 'no_slip' for 'fluid.incompressible'")  // AUTHOR TEXT
                  .with_hint("Use 'wall' (the name here for 'no_slip'). " + list_sentence("condition kinds", F, kind_names(F)))
                  .with_details(json{{"kind", "no_slip"}, {"use", "wall"}, {"valid", jarr(kind_names(F))}, {"near", json::array()}}));
    d["physics"]["flow"]["boundary_conditions"]["walls"]["kind"] = "outlt";
    check_one("a near kind", d,
              W("physics.unknown_condition_kind", "/physics/flow/boundary_conditions/walls/kind", "unknown condition kind 'outlt' for 'fluid.incompressible'")
                  .with_hint("Did you mean 'outlet'? " + list_sentence("condition kinds", F, kind_names(F)))
                  .with_details(json{{"near", json::array({"outlet"})}}));
    d["physics"]["flow"]["boundary_conditions"]["walls"].erase("kind");
    check_one("a missing kind", d, W("", "/physics/flow/boundary_conditions/walls", "missing required field 'kind'"));
    // a kind of another type
    d = channel();
    d["physics"]["flow"]["boundary_conditions"]["walls"]["kind"] = "temperature";
    check_one("a kind of another type", d, W("physics.unknown_condition_kind", "/physics/flow/boundary_conditions/walls/kind", "unknown condition kind 'temperature' for 'fluid.incompressible'"));
    // the kinds of the absent types are known
    d = busbar();
    clean("the busbar's kinds of absent types", d);
  }
  // ---- sources
  {
    json d = channel();
    d["physics"]["flow"]["sources"] = J(R"j([{"kind": "gravity", "in": "channel", "force_N_m3": [0, 1]}])j");
    check_one("an unknown source kind", d,
              W("physics.unknown_source_kind", "/physics/flow/sources/0/kind", "unknown source kind 'gravity' for 'fluid.incompressible'")  // AUTHOR TEXT
                  .with_hint("The source kinds of 'fluid.incompressible' are: " + join(source_names(F)) + ".")
                  .with_details(json{{"kind", "gravity"}, {"valid", jarr(source_names(F))}}));
    d["physics"]["flow"]["sources"] = J(R"j([{"in": "channel"}])j");
    check_one("a source without a kind", d, W("", "/physics/flow/sources/0", "missing required field 'kind'"));
    d["physics"]["flow"]["sources"] = J(R"j([{"kind": "body_force", "in": "channel", "force_N_m3": [0, "-rho*9.81"]}])j");
    clean("a body force", d);
  }
  // ---- keys of an instance, data keys of a condition
  {
    json d = channel();
    d["physics"]["flow"]["bogus"] = 1;
    check_one("an unknown instance key", d, W("", "/physics/flow", "unknown field 'bogus'").with_hint(only_list({"type", "domains", "options", "boundary_conditions", "sources", "initial", "notes"})));
    d = channel();
    d["physics"]["flow"]["boundary_conditions"]["outlet"]["bogus"] = 1;
    check_one("an unknown data key", d, W("", "/physics/flow/boundary_conditions/outlet", "unknown field 'bogus'").with_hint(only_list({"kind", "pressure_Pa"})));
    d = channel();
    d["physics"]["flow"]["boundary_conditions"]["inlet"].erase("velocity_m_s");
    check_one("a missing data key", d, W("", "/physics/flow/boundary_conditions/inlet", "missing required field 'velocity_m_s'"));
    d = channel();
    d["physics"]["flow"]["initial"] = J(R"j({"velocity_m_s": [0, 0]})j");
    d["physics"]["flow"]["notes"] = "free text";
    if (clean("initial and notes", d)) CHECK(parse(d).m.instances.at("flow").notes == "free text" && parse(d).m.instances.at("flow").initial == J(R"j({"velocity_m_s": [0, 0]})j"), "initial and notes are kept as written");
  }
}

void check_per_depth_keys() {
  // ---- planar: current_A is refused, current_A_m is the spelling; the note's text (section 3.3)
  {
    json d = plate();
    at(d, "/physics/cell/boundary_conditions/anode") = J(R"j({"kind": "electrode", "reactions": ["plate"], "current_A": 0.1})j");
    check_one("current_A in a planar model", d,
              W("model.per_depth_key", "/physics/cell/boundary_conditions/anode/current_A",
                "'current_A' is a total current; this model is planar, so results and inputs are per metre of depth. Write 'current_A_m' [A/m]")
                  .with_details(J(R"j({"use": "current_A_m", "symmetry": "planar"})j")));
    at(d, "/physics/cell/boundary_conditions/anode") = J(R"j({"kind": "electrode", "reactions": ["plate"], "current_A_m": 0.1})j");
    if (clean("current_A_m in a planar model", d))
      CHECK(datum_of(cond(parse(d).m.instances.at("cell"), "anode"), "current_A_m").values == std::vector<double>({0.1}), "the key is kept as written");
  }
  // ---- axisymmetric and 3D: the other direction (AUTHOR TEXT: the note gives the planar direction only)
  {
    json d = plate_axisymmetric();
    at(d, "/physics/cell/boundary_conditions/anode") = J(R"j({"kind": "electrode", "reactions": ["plate"], "current_A_m": 0.1})j");
    check_one("current_A_m in an axisymmetric model", d,
              W("model.per_depth_key", "/physics/cell/boundary_conditions/anode/current_A_m",
                "'current_A_m' is a current per metre of depth; this model is axisymmetric, so inputs are 3D totals. Write 'current_A' [A]")
                  .with_details(J(R"j({"use": "current_A", "symmetry": "axisymmetric"})j")));
    at(d, "/physics/cell/boundary_conditions/anode") = J(R"j({"kind": "electrode", "reactions": ["plate"], "current_A": 0.1})j");
    clean("current_A in an axisymmetric model", d);
  }
  // ---- a heat source: P_W and P_W_m
  {
    json d = heated();
    at(d, "/physics/heat/sources") = J(R"j([{"kind": "heat_source", "in": "cavity", "P_W": 1.0}])j");
    check_one("P_W in a planar model", d,
              W("model.per_depth_key", "/physics/heat/sources/0/P_W",
                "'P_W' is a total power; this model is planar, so results and inputs are per metre of depth. Write 'P_W_m' [W/m]")
                  .with_details(J(R"j({"use": "P_W_m", "symmetry": "planar"})j")));
    at(d, "/physics/heat/sources") = J(R"j([{"kind": "heat_source", "in": "cavity", "P_W_m": 1.0}])j");
    clean("P_W_m in a planar model", d);
    json b = busbar();
    at(b, "/physics/heat/sources") = J(R"j([{"kind": "heat_source", "in": "metal", "P_W_m": 1.0}])j");
    check_one("P_W_m in a 3d model", b,
              W("model.per_depth_key", "/physics/heat/sources/0/P_W_m",
                "'P_W_m' is a power per metre of depth; this model is 3d, so inputs are 3D totals. Write 'P_W' [W]")
                  .with_details(J(R"j({"use": "P_W", "symmetry": "3d"})j")));
    at(b, "/physics/heat/sources") = J(R"j([{"kind": "heat_source", "in": "metal", "P_W": 1.0}])j");
    clean("P_W in a 3d model", b);
  }
}

void check_field_expressions() {
  const std::string vel = "/physics/flow/boundary_conditions/inlet/velocity_m_s";
  // ---- a coordinate that the model does not have
  {
    json d = channel();
    at(d, vel) = J(R"j(["6*z", 0])j");
    check_one("z in a planar inlet", d,
              W("expr.unknown_variable", vel + "/0")
                  .with_message_has("unknown name 'z'")
                  .with_hint_has("x, y")
                  .with_details(J(R"j({"name": "z", "variables": ["x", "y"]})j")));
  }
  // ---- the length of a vector: the components of the geometry
  {
    json d = pipe();
    at(d, vel) = J(R"j([0, "2*(1-(r/R)^2)"])j");
    check_one("a two-component axisymmetric velocity", d,
              W("expr.vector_length", vel, "'velocity_m_s' has 2 components, and this axisymmetric model needs 3 (r, z, theta)")  // AUTHOR TEXT (message)
                  .with_hint("add the swirl component: [u_r, u_z, 0]")                                                    // the note, section 7
                  .with_details(J(R"j({"key": "velocity_m_s", "found": 2, "expected": 3})j")));
    d = channel();
    at(d, vel) = J(R"j([1, 0, 0])j");
    check_one("a three-component planar velocity", d,
              W("expr.vector_length", vel, "'velocity_m_s' has 3 components, and this planar model needs 2 (x, y)")
                  .with_hint("write one expression per component, in the order (x, y).")
                  .with_details(J(R"j({"found": 3, "expected": 2})j")));
    json b = busbar();
    at(b, "/physics/heat/sources") = J(R"j([{"kind": "prescribed_advection", "velocity_m_s": [0, 1]}])j");
    check_one("a two-component 3d prescribed velocity", b,
              W("expr.vector_length", "/physics/heat/sources/0/velocity_m_s", "'velocity_m_s' has 2 components, and this 3d model needs 3 (x, y, z)")
                  .with_hint("write one expression per component, in the order (x, y, z)."));
    // the velocity of a prescribed advection is meridional: two components in axisymmetric geometry
    json a = pipe();
    a["materials"]["water"]["thermal"] = J(R"j({"k_W_mK": 0.6, "cp_J_kgK": 4182})j");
    at(a, "/physics/heat") = J(R"j({"type": "heat.transfer", "sources": [{"kind": "prescribed_advection", "velocity_m_s": [0, "1-(r/R)^2"]}]})j");
    clean("two components for a prescribed advection in axisymmetric geometry", a);
    at(a, "/physics/heat/sources/0/velocity_m_s") = J(R"j([0, "1-(r/R)^2", 0])j");
    check_one("three components for it", a,
              W("expr.vector_length", "/physics/heat/sources/0/velocity_m_s", "'velocity_m_s' has 3 components, and this axisymmetric model needs 2 (r, z)")
                  .with_hint("write one expression per component, in the order (r, z)."));
    // a planar prescribed velocity has two
    json c = channel();
    at(c, "/physics/heat") = J(R"j({"type": "heat.transfer", "sources": [{"kind": "prescribed_advection", "velocity_m_s": ["U_mean", 0]}]})j");
    clean("two components for a planar prescribed advection", c);
    // the gravity of a buoyancy term is meridional too
    json h = heated();
    at(h, "/couplings/1/gravity_m_s2") = J(R"j([0, -9.81, 0])j");
    check_one("a three-component gravity in a planar model", h, W("expr.vector_length", "/couplings/1/gravity_m_s2", "'gravity_m_s2' has 3 components, and this planar model needs 2 (x, y)"));
  }
  // ---- the swirl hint is that of every axisymmetric vector written with two components (section 7), named after its key
  {
    json d = pipe();
    at(d, "/physics/flow/sources") = J(R"j([{"kind": "body_force", "force_N_m3": [0, 1]}])j");
    check_one("a two-component axisymmetric body force", d,
              W("expr.vector_length", "/physics/flow/sources/0/force_N_m3", "'force_N_m3' has 2 components, and this axisymmetric model needs 3 (r, z, theta)")
                  .with_hint("add the swirl component: [force_r, force_z, 0]"));  // AUTHOR TEXT (the component names)
    at(d, "/physics/flow/sources") = J(R"j([{"kind": "body_force", "force_N_m3": [0, 1, 0, 0]}])j");
    check_one("a four-component axisymmetric body force", d,
              W("expr.vector_length", "/physics/flow/sources/0/force_N_m3").with_hint("write one expression per component, in the order (r, z, theta)."));
  }
  // ---- the time t is a variable where a transient step solves the instance (section 7), after the coordinates
  {
    const std::string hot = "/physics/heat/boundary_conditions/hot/T_K";
    const json transient = J(R"j([{"id": "run", "type": "transient", "time": {"t_end_s": 1.0}}])j");
    // a transient step over every instance: T_hot + t at t = 2 is 303.15 + 2, by hand
    json d = heated();
    d["study"] = transient;
    at(d, hot) = "T_hot + t";
    if (clean("t in a transient step", d)) {
      const Parsed p = parse(d);
      const M::Instance& h = p.m.instances.at("heat");
      const M::Datum& T = datum_of(cond(h, "hot"), "T_K");
      const double pt[3] = {0.0, 0.5, 2.0};
      CHECK(h.in_transient_step && p.m.instances.at("flow").in_transient_step, "both instances are solved in the transient step");
      CHECK(T.variables == Strs({"x", "y", "t"}), "the variables of the datum: the coordinates, then the time");
      CHECK(T.exprs.size() == 1 && close(T.eval(pt).at(0), 303.15 + 2.0) && T.exprs[0].eval_d(pt, 2) == 1.0, "T_hot + t at t = 2, and its exact time derivative");
    }
    // the same datum in the stationary step of the document: t is no variable, and the hint says where it is one
    d = heated();
    at(d, hot) = "T_hot + t";
    check_one("t in a stationary step", d,
              W("expr.unknown_variable", hot)
                  .with_message_has("unknown name 't'")
                  .with_hint_has("The time t is a variable only for a physics that a transient step of the study solves.")  // AUTHOR TEXT
                  .with_details(J(R"j({"name": "t", "variables": ["x", "y"]})j")));
    {
      const Parsed p = parse(heated());
      CHECK(!p.m.instances.at("heat").in_transient_step, "no transient step: the instance is not marked");
      CHECK(datum_of(cond(p.m.instances.at("heat"), "hot"), "T_K").variables == Strs({"x", "y"}), "no transient step: the coordinates only");
    }
    // no study at all is the stationary default step
    d = heated();
    d.erase("study");
    at(d, hot) = "T_hot + t";
    check_one("t without a study", d, W("expr.unknown_variable", hot).with_details(J(R"j({"name": "t", "variables": ["x", "y"]})j")));
    // a transient step that lists its instances: t is a variable of those only
    d = heated();
    d["study"].push_back(J(R"j({"id": "ramp", "type": "transient", "physics": ["flow"], "initial_from": "steady", "time": {"t_end_s": 1.0}})j"));
    at(d, hot) = "T_hot + t";
    check_one("t in an instance that the transient step does not solve", d, W("expr.unknown_variable", hot));
    d["study"][1]["physics"] = J(R"j(["heat"])j");
    clean("t in the instance that the transient step solves", d);
    CHECK(!parse(d).m.instances.at("flow").in_transient_step && parse(d).m.instances.at("heat").in_transient_step, "the listed instance only");
    // a parameter named t is refused where t is a variable (code, the datum's pointer, details, the wording of the context)
    json c = channel();
    c["study"] = transient;
    at(c, "/parameters/t") = 1.0;
    check_one("a parameter named t in a transient step", c,
              W("expr.parameter_shadows_variable", vel + "/0", "parameter 't' has the name of the time of a transient step; rename the parameter")  // AUTHOR TEXT
                  .with_details(J(R"j({"parameter": "t", "variables": ["x", "y", "t"]})j")));
    // and is an ordinary parameter where it is not: a stationary document may use it
    c = channel();
    at(c, "/parameters/t") = 1.0;
    at(c, vel) = J(R"j(["t*y", 0])j");
    if (clean("a parameter named t in a stationary document", c)) {
      const double pt[2] = {0.0, 0.25};
      CHECK(close(datum_of(cond(parse(c).m.instances.at("flow"), "inlet"), "velocity_m_s").eval(pt).at(0), 0.25), "t is the parameter: 1 * 0.25");
    }
    // a coordinate is named before the time when both are shadowed
    c = channel();
    c["study"] = transient;
    at(c, "/parameters/t") = 1.0;
    at(c, "/parameters/y") = 1.0;
    check_one("parameters named y and t in a transient step", c,
              W("expr.parameter_shadows_variable", vel + "/0", "parameter 'y' has the name of a coordinate of this planar model; rename the parameter"));
  }
  // ---- a parameter with the name of a coordinate (section 7); the datum's own pointer, once per datum
  {
    json d = channel();
    at(d, "/parameters/y") = 2.0;
    check_one("a parameter named y in a planar model", d,
              W("expr.parameter_shadows_variable", vel + "/0", "parameter 'y' has the name of a coordinate of this planar model; rename the parameter")
                  .with_details(J(R"j({"parameter": "y", "variables": ["x", "y"]})j")));
    json a = pipe();
    at(a, "/parameters/z") = 2.0;
    check_one("a parameter named z in an axisymmetric model", a,
              W("expr.parameter_shadows_variable", vel + "/0", "parameter 'z' has the name of a coordinate of this axisymmetric model; rename the parameter")  // AUTHOR TEXT
                  .with_details(J(R"j({"parameter": "z", "variables": ["r", "z"]})j")));
    json b = busbar();
    at(b, "/parameters/x") = 2.0;
    check_one("a parameter named x in a 3d model", b,
              W("expr.parameter_shadows_variable", "/physics/current/boundary_conditions/terminal/V_V", "parameter 'x' has the name of a coordinate of this 3d model; rename the parameter")
                  .with_details(J(R"j({"parameter": "x", "variables": ["x", "y", "z"]})j")));
    // a parameter that is not a coordinate is fine, and so is a coordinate-named parameter where no field expression is compiled
    json ok = channel();
    at(ok, "/parameters/yy") = 2.0;
    clean("a parameter that is not a coordinate", ok);
  }
  // ---- a number that is not a field expression is evaluated over the parameters only
  {
    json d = channel();
    at(d, "/physics/flow/boundary_conditions/outlet/pressure_Pa") = "rho*9.81*H";
    if (clean("an expression over the parameters", d))
      CHECK(close(datum_of(cond(parse(d).m.instances.at("flow"), "outlet"), "pressure_Pa").values.at(0), 998.2 * 9.81 * 0.01), "pressure_Pa = rho g H");
  }
  // ---- the compiled expression is derived exactly: the divergence of the inlet velocity is d/dx(0) + d/dy(...)
  {
    const Parsed p = parse(channel());
    const M::Datum& v = datum_of(cond(p.m.instances.at("flow"), "inlet"), "velocity_m_s");
    if (v.exprs.size() == 2) {
      const double pt[2] = {0.0, 0.0025};
      const double H = 0.01, U = 0.01;
      // d/dy [6 U y (H - y)/H^2] = 6 U (H - 2 y)/H^2, from hand differentiation
      CHECK(close(v.exprs[0].eval_d(pt, 1), 6.0 * U * (H - 2.0 * 0.0025) / (H * H), 1e-11) && v.exprs[1].eval_d(pt, 0) == 0.0, "the compiled expression has exact derivatives");
    }
  }
}

void check_couplings_grammar() {
  const std::string unknown_hint = "The coupling terms are: " + join(term_names()) + ".";
  {
    json d = heated();
    at(d, "/couplings/0/term") = "convection";
    check_one("an unknown term", d,
              W("coupling.unknown_term", "/couplings/0/term", "unknown coupling term 'convection'")  // AUTHOR TEXT
                  .with_hint(unknown_hint)
                  .with_details(json{{"term", "convection"}, {"valid", jarr(term_names())}}));
  }
  {
    // advection from heat to flow: the term goes from fluid.* to heat.transfer or species.transport
    json d = heated();
    at(d, "/couplings/0/from") = "heat";
    at(d, "/couplings/0/to") = "flow";
    check_one("a term between types it does not connect", d,
              W("coupling.type_mismatch", "/couplings/0", "the term 'advection' does not connect 'heat' (heat.transfer) to 'flow' (fluid.incompressible)")  // AUTHOR TEXT
                  .with_hint("'advection' goes from fluid.* to heat.transfer or species.transport.")
                  .with_details(J(R"j({"term": "advection", "from": "heat", "to": "flow", "from_type": "heat.transfer", "to_type": "fluid.incompressible",
                                      "producers": ["fluid.*"], "consumers": ["heat.transfer", "species.transport"]})j")));
  }
  {
    json d = heated();
    at(d, "/couplings/0/on") = "hot";
    check_one("on for a domain term", d,
              W("coupling.wrong_selection_key", "/couplings/0/on", "the term 'advection' acts in domains, so it takes 'in' and not 'on'")  // AUTHOR TEXT
                  .with_hint("Write \"in\": DOMAIN to restrict where the term acts, or remove 'on'.")
                  .with_details(J(R"j({"term": "advection", "support": "domain", "key": "in"})j")));
  }
  {
    json d = heated();
    at(d, "/couplings/0/in") = "cavity";
    if (clean("in for a domain term", d)) CHECK(parse(d).m.couplings[0].in == "cavity", "in is parsed");
    at(d, "/couplings/0/in") = "pond";
    check_one("in names an unknown domain", d, W("domain.unknown_name", "/couplings/0/in", "unknown domain or region 'pond'"));
  }
  {
    json d = heated();
    at(d, "/couplings/0/from") = "stream";
    check_one("an unknown instance", d,
              W("coupling.unknown_physics", "/couplings/0/from", "unknown physics 'stream'")  // AUTHOR TEXT
                  .with_hint("The physics of this document are: flow, heat.")
                  .with_details(J(R"j({"physics": "stream", "valid": ["flow", "heat"]})j")));
  }
  {
    json d = heated();
    at(d, "/couplings/0/bogus") = 1;
    check_one("an unknown coupling key", d, W("", "/couplings/0", "unknown field 'bogus'").with_hint(only_list({"term", "from", "to", "in", "on"})));
    d = heated();
    at(d, "/couplings/1/bogus") = 1;
    check_one("an unknown buoyancy key", d, W("", "/couplings/1", "unknown field 'bogus'").with_hint(only_list({"term", "from", "to", "in", "on", "T_ref_K", "gravity_m_s2"})));
    d = heated();
    d["couplings"][1].erase("T_ref_K");
    check_one("a missing term data key", d, W("", "/couplings/1", "missing required field 'T_ref_K'"));
    d = heated();
    d["couplings"][0].erase("to");
    check_one("a missing to", d, W("", "/couplings/0", "missing required field 'to'"));
    d = heated();
    d["couplings"] = J(R"j({"term": "advection"})j");
    check_one("couplings that is not a list", d, W("", "/couplings", "couplings must be a list"));
  }
}

void check_study_grammar() {
  // ---- types
  {
    json d = heated();
    for (const char* t : {"eigen", "frequency"}) {
      at(d, "/study/0/type") = t;
      check_one(std::string("study type ") + t, d,
                W("study.type_reserved", "/study/0/type", std::string("study type '") + t + "' is reserved")  // AUTHOR TEXT
                    .with_hint("The electromagnetic eigenmode and frequency analyses stay in the 'cavity' and 'fullwave' sections; a model document has "
                               "the study types \"stationary\" and \"transient\"."));
    }
    at(d, "/study/0/type") = "steady";
    check_one("an unknown study type", d, W("", "/study/0/type", "unknown study type 'steady'").with_hint("Use \"stationary\" or \"transient\"."));  // AUTHOR TEXT
    d = heated();
    d["study"][0].erase("type");
    check_one("a missing study type", d, W("", "/study/0", "missing required field 'type'"));
    d = heated();
    d["study"][0].erase("id");
    check_one("a missing study id", d, W("", "/study/0", "missing required field 'id'"));
    d = heated();
    d["study"].push_back(d["study"][0]);
    check_one("two steps with one id", d, W("", "/study/1/id", "two study steps have the id 'steady'"));
    d = heated();
    at(d, "/study/0/continuation") = J(R"j({"scale": "nonsense", "factors": [0.5, 1.0]})j");
    check_one("an unknown continuation scale", d, W("", "/study/0/continuation/scale", "unknown continuation scale 'nonsense'"));
    at(d, "/study/0/continuation") = J(R"j({"scale": "buoyancy", "factors": [0.5, 2.0]})j");
    check_one("a continuation that does not end at 1", d, W("", "/study/0/continuation/factors", "the last factor of a continuation must be 1, the full coefficient"));
    d = heated();
    at(d, "/study/0/given") = J(R"j({"bogus": 1})j");
    check_one("a given that is not a field", d, W("", "/study/0/given/bogus", "'bogus' is not a field name"));
    d = heated();
    at(d, "/study/0/coupling") = J(R"j({"strategy": "monolithic", "tolerance": -1})j");
    check_one("a negative coupling tolerance", d, W("", "/study/0/coupling/tolerance", "tolerance must be positive"));
    d = heated();
    at(d, "/study/0/time") = J(R"j({"t_end_s": -1})j");
    check_one("a negative end time", d, W("", "/study/0/time/t_end_s", "t_end_s must be positive"));
  }
  // ---- values that no study can mean (AUTHOR TEXT)
  {
    json d = heated();
    at(d, "/study/0/continuation") = J(R"j({"scale": "buoyancy", "factors": []})j");
    check_one("a continuation without factors", d, W("", "/study/0/continuation/factors", "'factors' must list at least one factor"));
    d = plate();
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "continuation": {"ramp": "/physics/cell/boundary_conditions/anode/phi_s_V", "values": []}}])j");
    check_one("a ramp without values", d, W("", "/study/0/continuation/values", "'values' must list at least one value"));
    for (const char* n : {"-3", "0", "2.5"}) {
      d = heated();
      at(d, "/study/0/save") = json{{"n", json::parse(n)}};
      check_one(std::string("save.n = ") + n, d, W("", "/study/0/save/n", "n must be a positive integer"));
    }
    for (const char* n : {"-1", "0", "1.5"}) {
      d = heated();
      at(d, "/study/0/coupling/max_iterations") = json::parse(n);
      check_one(std::string("coupling.max_iterations = ") + n, d, W("", "/study/0/coupling/max_iterations", "max_iterations must be a positive integer"));
    }
    d = heated();
    at(d, "/study/0/coupling/max_iterations") = 25;
    if (clean("coupling.max_iterations = 25", d)) CHECK(parse(d).m.study[0].coupling.max_iterations == 25, "max_iterations is kept");
    for (const char* n : {"7", "0", "1.5"}) {
      d = heated();
      at(d, "/study/0/time") = json{{"t_end_s", 1.0}, {"order", json::parse(n)}};
      check_one(std::string("time.order = ") + n, d, W("", "/study/0/time/order", "order must be an integer from 1 to 5"));
    }
    // a transient step needs its end time (section 3.8: an end time has no sensible default)
    const std::string hint = "A transient step needs its end time, which has no sensible default: \"time\": {\"t_end_s\": ...}.";
    d = heated();
    at(d, "/study/0/type") = "transient";
    check_one("a transient step without time", d, W("", "/study/0", "missing required field 'time'").with_hint(hint));
    at(d, "/study/0/time") = J(R"j({"rtol": 1e-3})j");
    check_one("a transient step without an end time", d, W("", "/study/0/time", "missing required field 't_end_s'").with_hint(hint));
    at(d, "/study/0/time") = J(R"j({"t_end_s": 2.0})j");
    clean("a transient step with its end time", d);
  }
  // ---- physics, initial_from, the other keys
  {
    json d = heated();
    at(d, "/study/0/physics") = J(R"j(["flow", "nope"])j");
    check_one("an unknown physics in a step", d,
              W("study.unknown_physics", "/study/0/physics/1", "unknown physics 'nope'")  // AUTHOR TEXT
                  .with_hint("The physics of this document are: flow, heat.")
                  .with_details(J(R"j({"physics": "nope", "valid": ["flow", "heat"]})j")));
    d = heated();
    d["study"].push_back(J(R"j({"id": "ramp", "type": "transient", "physics": ["flow"], "initial_from": "nope", "time": {"t_end_s": 1.0}})j"));
    check_one("initial_from an unknown step", d,
              W("study.initial_from_unknown_step", "/study/1/initial_from", "'initial_from' names the step 'nope', which is not an earlier step")  // AUTHOR TEXT
                  .with_hint("The earlier steps of this study are: steady.")
                  .with_details(J(R"j({"initial_from": "nope", "earlier": ["steady"]})j")));
    d = heated();
    d["study"][0]["initial_from"] = "ramp";
    d["study"].push_back(J(R"j({"id": "ramp", "type": "transient", "physics": ["flow"], "time": {"t_end_s": 1.0}})j"));
    check_one("initial_from a later step", d,
              W("study.initial_from_unknown_step", "/study/0/initial_from", "'initial_from' names the step 'ramp', which is not an earlier step")
                  .with_hint("This is the first step, so it has no earlier step to start from.")
                  .with_details(J(R"j({"earlier": []})j")));
    d = heated();
    d["study"].push_back(J(R"j({"id": "ramp", "type": "transient", "physics": ["flow"], "initial_from": "steady",
                              "time": {"t_end_s": 10.0, "rtol": 1e-3, "first_step_s": 0.01, "order": 2}, "save": {"times_s": [1.0, 5.0, 10.0]}})j"));
    const Parsed p = parse(d);
    CHECK(!p.f.has_errors(), "the transient step of section 3.8");
    if (p.m.study.size() == 2) {
      const M::StudyStep& s = p.m.study[1];
      CHECK(s.type == "transient" && s.initial_from == "steady" && s.physics == Strs({"flow"}) && !s.implicit_default, "the second step");
      CHECK(s.time.at("t_end_s") == 10.0 && close(s.time.at("rtol"), 1e-3) && close(s.time.at("first_step_s"), 0.01) && s.time.at("order") == 2.0, "time");
      CHECK(s.save_times_s == std::vector<double>({1.0, 5.0, 10.0}) && s.save_n == 0, "save.times_s");
    }
    at(d, "/study/1/save") = J(R"j({"n": 51})j");
    const Parsed q = parse(d);
    CHECK(!q.f.has_errors() && q.m.study.size() == 2 && q.m.study[1].save_n == 51 && q.m.study[1].save_times_s.empty(), "save.n");
  }
  {
    json d = heated();
    d["study"][0]["time"] = J(R"j({"t_end": 1})j");
    check_one("an unknown time key", d, W("", "/study/0/time", "unknown field 't_end'").with_hint(only_list({"t_end_s", "rtol", "first_step_s", "order"})));
    d = heated();
    d["study"][0]["save"] = J(R"j({"times": [1]})j");
    check_one("an unknown save key", d, W("", "/study/0/save", "unknown field 'times'").with_hint(only_list({"times_s", "n"})));
    d = heated();
    d["study"][0]["bogus"] = 1;
    check_one("an unknown step key", d,
              W("", "/study/0", "unknown field 'bogus'")
                  .with_hint(only_list({"id", "type", "physics", "initial_from", "given", "coupling", "continuation", "solvers", "time", "save"})));
    d = heated();
    d["study"][0]["given"] = J(R"j({"T": "T_hot"})j");
    if (clean("given", d)) CHECK(close(parse(d).m.study[0].given.at("T"), 303.15), "given is evaluated over the parameters");
  }
  // ---- coupling strategy, continuation, solvers
  {
    json d = heated();
    at(d, "/study/0/coupling/strategy") = "newton";
    check_one("an unknown strategy", d, W("", "/study/0/coupling/strategy", "unknown coupling strategy 'newton'").with_hint("Use \"one_way\", \"gauss_seidel\", \"aitken\" or \"monolithic\"."));  // AUTHOR TEXT
    d = heated();
    at(d, "/study/0/continuation") = J(R"j({"scale": "buoyancy", "ramp": "/x", "factors": [1], "values": [1]})j");
    check_one("a continuation with scale and ramp", d, W("", "/study/0/continuation", "a continuation has exactly one of 'scale' and 'ramp'"));  // AUTHOR TEXT
    d = plate();
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "continuation": {"ramp": "/physics/cell/boundary_conditions/anode/phi_s_V", "values": [0.2, 0.4, "V_cell"]}}])j");
    const Parsed p = parse(d);
    CHECK(!p.f.has_errors() && p.m.study.size() == 1, "a ramp continuation parses (the pointer is checked by the validation)j");
    if (p.m.study.size() == 1)
      CHECK(p.m.study[0].continuation.ramp == "/physics/cell/boundary_conditions/anode/phi_s_V" && p.m.study[0].continuation.scale.empty() &&
                p.m.study[0].continuation.values == std::vector<double>({0.2, 0.4, 0.05}),
            "the ramp values are evaluated over the parameters (V_cell = 0.05)j");
    d = heated();
    at(d, "/study/0/solvers") = J(R"j({"nope": {"newton_rtol": 1e-12}})j");
    check_one("solvers of an unknown physics", d, W("study.unknown_physics", "/study/0/solvers/nope", "unknown physics 'nope'"));
    // the settings of an instance are the settings of its type's rows
    std::set<std::string> settings;
    for (const auto& r : M::registry().routes)
      for (const auto& s : r.solvers)
        if (s.physics == "fluid.incompressible") settings.insert(s.name);
    CHECK(!settings.empty(), "the registry lists solver settings of the flow type");
    d = heated();
    at(d, "/study/0/solvers") = J(R"j({"flow": {"newton_rtol": 1e-12, "line_search": true, "bogus": 1}})j");
    check_one("an unknown solver setting", d, W("", "/study/0/solvers/flow", "unknown field 'bogus'").with_hint(only_list(settings)));
    d = heated();
    at(d, "/study/0/solvers") = J(R"j({"flow": {"newton_rtol": 1e-12, "line_search": false}})j");
    if (clean("solver settings", d)) {
      const auto& s = parse(d).m.study[0].solvers;
      CHECK(s.count("flow") && s.at("flow").count("newton_rtol") && close(s.at("flow").at("newton_rtol").get<double>(), 1e-12) && s.at("flow").at("line_search") == false, "the settings are kept");
    }
  }
  // ---- the implicit default step lists every instance, in key order
  {
    json d = heated();
    d.erase("study");
    const Parsed q = parse(d);
    CHECK(!q.f.has_errors() && q.m.study.size() == 1 && q.m.study[0].implicit_default && q.m.study[0].id == "default" && q.m.study[0].physics == Strs({"flow", "heat"}),
          "no 'study': the default step over every instance");
  }
}

void check_outputs_grammar() {
  // ---- the kinds of section 10.3
  {
    json d = heated();
    at(d, "/outputs/I") = J(R"j({"kind": "integral", "of": "T", "in": "cavity"})j");
    at(d, "/outputs/A") = J(R"j({"kind": "average", "of": "T", "on": "hot"})j");
    at(d, "/outputs/Tmin") = J(R"j({"kind": "minimum", "of": "T", "in": "cavity"})j");
    at(d, "/outputs/X") = J(R"j({"kind": "extreme", "of": "T", "on": "cold"})j");
    at(d, "/outputs/pt") = J(R"j({"kind": "point", "of": "u_x", "at": [0.02, "Lc/2"]})j");
    at(d, "/outputs/Fx") = J(R"j({"kind": "force", "on": "hot", "component": "x"})j");
    at(d, "/outputs/Tq") = J(R"j({"kind": "torque", "on": "hot", "about": [0, 0]})j");
    at(d, "/outputs/Qc") = J(R"j({"kind": "flux", "of": "heat_conductive", "on": "cold"})j");
    const Parsed p = parse(d);
    CHECK(!p.f.has_errors(), "every kind of output parses");
    for (const auto& e : errors_of(p.f)) std::printf("    [%s] %s [at %s]\n", e.code().c_str(), e.message().c_str(), e.where().c_str());
    if (p.m.outputs.count("pt")) {
      CHECK(p.m.outputs.at("pt").at.size() == 2 && p.m.outputs.at("pt").at[0] == 0.02 && close(p.m.outputs.at("pt").at[1], 0.023), "at is evaluated: Lc/2");
      CHECK(p.m.outputs.at("Fx").component == "x" && p.m.outputs.at("Fx").kind == "force" && p.m.outputs.at("I").in == "cavity" && p.m.outputs.at("A").on == "hot", "on, in and component");
    }
    // "physics" disambiguates: T is a quantity of heat only
    at(d, "/outputs/I/physics") = "heat";
    CHECK(!parse(d).f.has_errors(), "physics names an instance");
    at(d, "/outputs/I/physics") = "flow";
    check_one("physics names an instance whose type has no such quantity", d,
              W("output.unknown_quantity", "/outputs/I/of", "unknown quantity 'T' on a planar model")
                  .with_hint("The quantities of this document on a planar model are: " + join(quantity_names("fluid.incompressible", "planar")) + ".")
                  .with_details(json{{"quantity", "T"}, {"geometry", "planar"}, {"valid", jarr(quantity_names("fluid.incompressible", "planar"))}}));
    at(d, "/outputs/I/physics") = "stream";
    check_one("physics names no instance", d,
              W("output.unknown_physics", "/outputs/I/physics", "unknown physics 'stream'").with_hint("The physics of this document are: flow, heat."));  // AUTHOR TEXT
  }
  // ---- an output name is an identifier (section 3.9); an expression output beside it is not disturbed.  AUTHOR TEXT
  {
    for (const char* name : {"p-in", "1bad", "a b"}) {
      json d = heated();
      d["outputs"][name] = J(R"j({"kind": "average", "of": "T", "in": "cavity"})j");
      d["outputs"]["twice"] = J(R"j({"kind": "expression", "expr": "2*T_avg"})j");
      d["outputs"]["T_avg"] = J(R"j({"kind": "average", "of": "T", "in": "cavity"})j");
      check_one(std::string("an output named ") + name, d,
                W("output.grammar", std::string("/outputs/") + name, std::string("'") + name + "' is not a valid output name")
                    .with_hint("An output name is an identifier, so that an expression output can use it: letters, digits and underscores, not starting "
                               "with a digit ([A-Za-z_][A-Za-z0-9_]*).")
                    .with_details(json{{"name", name}}));
    }
    json d = heated();
    d["outputs"]["_T2"] = J(R"j({"kind": "average", "of": "T", "in": "cavity"})j");
    d["outputs"]["Twice"] = J(R"j({"kind": "expression", "expr": "2*_T2"})j");
    clean("identifiers with capitals, digits and a leading underscore", d);
  }
  // ---- unknown quantity: the names of the geometry class (section 10.3)
  {
    json d = pipe();
    at(d, "/outputs/ux") = J(R"j({"kind": "maximum", "of": "u_x", "in": "pipe"})j");
    const Strs names = quantity_names("fluid.incompressible", "axisymmetric");
    CHECK(std::find(names.begin(), names.end(), "u_r") != names.end() && std::find(names.begin(), names.end(), "u_x") == names.end(), "the registry: u_x is not an axisymmetric quantity");
    check_one("u_x on an axisymmetric model", d,
              W("output.unknown_quantity", "/outputs/ux/of", "unknown quantity 'u_x' on an axisymmetric model")  // AUTHOR TEXT
                  .with_hint("The quantities of this document on an axisymmetric model are: " + join(names) + ".")
                  .with_details(json{{"quantity", "u_x"}, {"geometry", "axisymmetric"}, {"valid", jarr(names)}}));
    at(d, "/outputs/ux/of") = "u_r";
    clean("u_r on an axisymmetric model", d);
    // a free-form expression in `of` is not offered
    d = heated();
    at(d, "/outputs/prod") = J(R"j({"kind": "maximum", "of": "u_x*T", "in": "cavity"})j");
    check_one("a free-form expression in of", d, W("output.unknown_quantity", "/outputs/prod/of", "unknown quantity 'u_x*T' on a planar model"));
  }
  {
    json d = channel();
    at(d, "/outputs/Q_out/on") = "outlt";
    check_one("an unknown boundary", d,
              W("output.unknown_boundary", "/outputs/Q_out/on", "unknown boundary 'outlt'")  // AUTHOR TEXT
                  .with_hint("The boundaries of this document are: inlet, outlet, walls.")
                  .with_details(J(R"j({"boundary": "outlt", "valid": ["inlet", "outlet", "walls"]})j")));
  }
  // ---- the grammar of each kind (AUTHOR TEXT)
  {
    json d = channel();
    d["outputs"] = J(R"j({"Q": {"kind": "flux", "of": "volume"}})j");
    check_one("flux without on", d, W("output.grammar", "/outputs/Q", "output 'Q' of kind 'flux' needs 'of' and 'on'"));
    d["outputs"] = J(R"j({"Q": {"kind": "maximum", "of": "p"}})j");
    check_one("maximum without on or in", d, W("output.grammar", "/outputs/Q", "output 'Q' of kind 'maximum' needs 'of' and one of 'on' and 'in'"));
    d["outputs"] = J(R"j({"Q": {"kind": "maximum", "of": "p", "on": "inlet", "in": "channel"}})j");
    check_one("maximum with on and in", d, W("output.grammar", "/outputs/Q", "output 'Q' of kind 'maximum' needs 'of' and one of 'on' and 'in'"));
    d["outputs"] = J(R"j({"Q": {"kind": "point", "of": "p"}})j");
    check_one("point without at", d, W("output.grammar", "/outputs/Q", "output 'Q' of kind 'point' needs 'of' and 'at'"));
    d["outputs"] = J(R"j({"Q": {"kind": "force"}})j");
    check_one("force without on", d, W("output.grammar", "/outputs/Q", "output 'Q' of kind 'force' needs 'on'"));
    d["outputs"] = J(R"j({"Q": {"kind": "expression"}})j");
    check_one("expression without expr", d, W("output.grammar", "/outputs/Q", "output 'Q' of kind 'expression' needs 'expr'"));
    d["outputs"] = J(R"j({"Q": {"kind": "point", "of": "p", "at": [1, 2, 3]}})j");
    check_one("a point of three coordinates in a planar model", d, W("expr.vector_length", "/outputs/Q/at", "'at' has 3 components, and this planar model needs 2 (x, y)"));
    d["outputs"] = J(R"j({"Q": {"kind": "median", "of": "p", "on": "inlet"}})j");
    check_one("an unknown kind", d,
              W("output.unknown_kind", "/outputs/Q/kind", "unknown output kind 'median'")  // AUTHOR TEXT
                  .with_hint("The kinds are: integral, average, maximum, minimum, extreme, point, flux, force, torque, electrode_current, expression.")
                  .with_details(J(R"j({"kind": "median", "valid": ["integral", "average", "maximum", "minimum", "extreme", "point", "flux", "force", "torque", "electrode_current", "expression"]})j")));
    d["outputs"] = J(R"j({"Q": {"on": "inlet"}})j");
    check_one("an output without a kind", d, W("", "/outputs/Q", "missing required field 'kind'"));
    d["outputs"] = J(R"j({"Q": {"kind": "flux", "of": "volume", "on": "inlet", "bogus": 1}})j");
    check_one("an unknown output key", d,
              W("", "/outputs/Q", "unknown field 'bogus'")
                  .with_hint(only_list({"kind", "of", "on", "in", "at", "physics", "component", "about", "expr", "T_datum_K", "over", "notes"})));
  }
  // ---- expression outputs: cycles, unknown names, shadowing
  {
    json d = channel();
    d["outputs"] = J(R"j({"a": {"kind": "expression", "expr": "b + 1"}, "b": {"kind": "expression", "expr": "a*2"}})j");
    check_one("an expression cycle", d,
              W("output.cycle", "/outputs/a/expr", "the expression outputs form a cycle: a -> b -> a")  // AUTHOR TEXT
                  .with_hint("An expression output may use other outputs, but not itself, directly or through others.")
                  .with_details(J(R"j({"cycle": ["a", "b", "a"]})j")));
    d["outputs"] = J(R"j({"a": {"kind": "expression", "expr": "a + 1"}})j");
    check_one("an expression that reads itself", d, W("output.cycle", "/outputs/a/expr", "the expression outputs form a cycle: a -> a").with_details(J(R"j({"cycle": ["a", "a"]})j")));
    d["outputs"] = J(R"j({"a": {"kind": "expression", "expr": "b + 1"}, "b": {"kind": "expression", "expr": "c*2"}, "c": {"kind": "expression", "expr": "H"}})j");
    const Parsed p = parse(d);
    const auto pos = [&](const char* n) { return std::find(p.m.output_order.begin(), p.m.output_order.end(), n) - p.m.output_order.begin(); };
    CHECK(!p.f.has_errors() && p.m.output_order.size() == 3 && pos("c") < pos("b") && pos("b") < pos("a"), "a chain is evaluated in dependency order");
    CHECK(p.m.outputs.at("c").depends_on.empty() && p.m.outputs.at("b").depends_on == Strs({"c"}), "a parameter is not a dependency");
    d["outputs"] = J(R"j({"a": {"kind": "expression", "expr": "nothing + 1"}})j");
    check_one("an unknown name in an expression", d,
              W("expr.unknown_variable", "/outputs/a/expr").with_message_has("unknown name 'nothing'").with_details(J(R"j({"name": "nothing", "variables": ["a"]})j")));
    d = channel();
    at(d, "/parameters/Q_in") = 1.0;
    check_one("a parameter named like an output", d,
              W("expr.parameter_shadows_variable", "/outputs/dp/expr", "parameter 'Q_in' has the name of the output 'Q_in'; rename the parameter")  // AUTHOR TEXT
                  .with_details(J(R"j({"parameter": "Q_in"})j")));
  }
}

void check_reactions_grammar() {
  {
    json d = plate();
    d["reactions"] = J(R"j({"species": ["A", "B"],
        "mechanism": [{"reactants": {"A": 1}, "products": {"B": 1}, "A": 0.7, "Ea_J_mol": 0.0}],
        "electrode": {"plate": {"kind": "butler_volmer", "i0_A_m2": 10.0, "E_eq_V": 0.0, "alpha_a": 0.5, "alpha_c": 0.5},
                      "lin": {"kind": "linear", "i0_A_m2": 1.0, "E_eq_V": 0.1, "alpha_a": 0.5, "alpha_c": 0.5},
                      "ta": {"kind": "tafel_anodic", "i0_A_m2": 1.0, "E_eq_V": 0.0, "slope_V_decade": 0.12},
                      "tc": {"kind": "tafel_cathodic", "i0_A_m2": 1.0, "E_eq_V": 0.0, "slope_V_decade": 0.12}}})j");
    const Parsed p = parse(d);
    CHECK(!p.f.has_errors(), "species, a mechanism and the four electrode reaction kinds");
    for (const auto& e : errors_of(p.f)) std::printf("    [%s] %s [at %s]\n", e.code().c_str(), e.message().c_str(), e.where().c_str());
    CHECK(p.m.reactions.species == Strs({"A", "B"}) && p.m.reactions.mechanism.is_array() && p.m.reactions.mechanism.size() == 1, "species and mechanism");
    if (p.m.reactions.electrode.size() == 4) {
      CHECK(p.m.reactions.electrode.at("plate").kind == "butler_volmer" && p.m.reactions.electrode.at("ta").kind == "tafel_anodic" &&
                close(p.m.reactions.electrode.at("ta").slope_V_decade, 0.12) && close(p.m.reactions.electrode.at("lin").E_eq_V, 0.1),
            "the reactions are read");
    }
  }
  {
    json d = plate();
    at(d, "/reactions/electrode/plate/kind") = "tafel";
    check_one("an unknown reaction kind", d,
              W("reaction.unknown_kind", "/reactions/electrode/plate/kind", "unknown electrode reaction kind 'tafel'")  // AUTHOR TEXT
                  .with_hint("The kinds are: butler_volmer, linear, tafel_anodic, tafel_cathodic.")
                  .with_details(J(R"j({"kind": "tafel", "valid": ["butler_volmer", "linear", "tafel_anodic", "tafel_cathodic"]})j")));
    d = plate();
    at(d, "/reactions/electrode/plate/kind") = "tafel_anodic";
    check_one("a tafel reaction without a slope", d, W("", "/reactions/electrode/plate", "missing required field 'slope_V_decade'"));
    d = plate();
    d["reactions"]["electrode"]["plate"].erase("i0_A_m2");
    check_one("a reaction without i0", d, W("", "/reactions/electrode/plate", "missing required field 'i0_A_m2'"));
    d = plate();
    at(d, "/reactions/electrode/plate/bogus") = 1;
    check_one("an unknown reaction key", d, W("", "/reactions/electrode/plate", "unknown field 'bogus'").with_hint(only_list({"kind", "i0_A_m2", "E_eq_V", "alpha_a", "alpha_c", "slope_V_decade", "notes"})));
    d = plate();
    at(d, "/reactions/bogus") = 1;
    check_one("an unknown reactions key", d, W("", "/reactions", "unknown field 'bogus'").with_hint(only_list({"species", "mechanism", "electrode"})));
    d = plate();
    d["reactions"]["electrode"]["plate"].erase("kind");
    check_one("a reaction without a kind", d, W("", "/reactions/electrode/plate", "missing required field 'kind'"));
  }
  // an electrode condition names reactions from reactions.electrode only
  {
    json d = plate();
    at(d, "/physics/cell/boundary_conditions/anode/reactions") = J(R"j(["plate", "nope"])j");
    check_one("an unknown electrode reaction", d,
              W("reaction.unknown_name", "/physics/cell/boundary_conditions/anode/reactions/1", "unknown electrode reaction 'nope'")  // AUTHOR TEXT
                  .with_hint("The electrode reactions of this document are: plate.")
                  .with_details(J(R"j({"name": "nope", "valid": ["plate"]})j")));
    d.erase("reactions");
    check_one("an electrode condition in a document without reactions", d,
              W("reaction.unknown_name", "/physics/cell/boundary_conditions/anode/reactions/0", "unknown electrode reaction 'plate'")
                  .with_hint("This document has no electrode reactions: declare them under reactions.electrode.")
                  .with_details(J(R"j({"valid": []})j"))
                  .with_count(2));
    // a failed reactions section does not make every electrode reaction unknown
    d = plate();
    d["reactions"] = 5;
    check_one("reactions that is not an object", d, W("", "/reactions", "reactions must be an object"));
  }
}

// =========================================================================================== the region and curve loops
// The model path re-implements the loops over regions and curves of parse_design.  On every axisymmetric template and
// example, and on the curves of the two fixtures, the shapes and paths must equal those of the electromagnetic parser.
void compare_geometry(const std::string& what, const json& em_doc) {
  axmp::spec::Design d;
  try {
    d = axmp::spec::parse_design(em_doc);
  } catch (const std::exception& e) {
    fail(what + ": the electromagnetic parser refused the document: " + e.what());
    return;
  }
  json mod = json::object();
  mod["schema"] = em_doc.value("schema", "axionmultiphysics.design/1");
  mod["name"] = em_doc.value("name", "x");
  mod["symmetry"] = em_doc.value("symmetry", "axisymmetric");
  for (const char* k : {"parameters", "materials", "regions", "curves"})
    if (em_doc.contains(k)) mod[k] = em_doc[k];
  if (mod.contains("curves"))
    for (auto& c : mod["curves"]) {
      c.erase("material");
      c.erase("thickness");
    }
  mod["physics"] = J(R"j({"h": {"type": "heat.transfer"}})j");
  const Parsed p = parse(mod);
  for (const auto& e : errors_of(p.f)) fail(what + ": error [" + e.code() + "] " + e.message() + " [at " + e.where() + "]");
  if (p.f.has_errors()) return;
  const M::ResolvedModel& m = p.m;
  CHECK(m.disabled_regions == d.disabled_regions, what + ": disabled regions");
  if (m.regions.size() != d.regions.size()) {
    fail(what + ": " + std::to_string(m.regions.size()) + " regions, the electromagnetic parser has " + std::to_string(d.regions.size()));
    return;
  }
  for (size_t i = 0; i < d.regions.size(); ++i) {
    const auto& a = m.regions[i];
    const auto& b = d.regions[i];
    CHECK(a.name == b.name && a.material == b.material && a.mesh_size == b.mesh_size && a.surface_size == b.surface_size, what + ": region " + b.name + ": name, material, sizes");
    CHECK(a.shape.resolved() == b.shape.resolved(), what + ": region " + b.name + ": the resolved shape");
  }
  if (m.curves.size() != d.curves.size()) {
    fail(what + ": curves " + std::to_string(m.curves.size()) + " against " + std::to_string(d.curves.size()));
    return;
  }
  for (size_t i = 0; i < d.curves.size(); ++i) {
    CHECK(m.curves[i].name == d.curves[i].name && m.curves[i].mesh_size == d.curves[i].mesh_size && m.curves[i].surface_size == d.curves[i].surface_size,
          what + ": curve " + d.curves[i].name);
    CHECK(m.curves[i].path.resolved() == d.curves[i].path.resolved(), what + ": curve " + d.curves[i].name + ": the resolved path");
  }
  CHECK(m.parameters == d.parameters, what + ": parameters");
}

void check_region_loop_parity() {
  int n = 0;
  for (const auto& t : axmp::templates::list()) {
    const std::string name = t.at("name").get<std::string>();
    const json em = axmp::templates::make(name);
    if (em.value("symmetry", "axisymmetric") != "axisymmetric") continue;
    compare_geometry("template " + name, em);
    ++n;
  }
  CHECK(n >= 3, "at least three axisymmetric templates were compared: " + std::to_string(n));
#ifdef AXMP_SOURCE_DIR
  // the examples, the benchmark designs and the G0 designs: every axisymmetric document that is not a model document
  int m = 0;
  for (const char* dir : {"/examples", "/benchmarks/designs", "/tests/g0/designs"}) {
    const std::string path = std::string(AXMP_SOURCE_DIR) + dir;
    if (!std::filesystem::is_directory(path)) continue;
    std::vector<std::filesystem::path> files;
    for (const auto& f : std::filesystem::directory_iterator(path))
      if (f.path().extension() == ".json") files.push_back(f.path());
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
      std::ifstream in(f);
      json em;
      try {
        in >> em;
      } catch (...) {
        continue;
      }
      if (!em.is_object() || em.contains("physics") || em.value("symmetry", "axisymmetric") != "axisymmetric" || !em.contains("name")) continue;
      compare_geometry(std::string(dir + 1) + "/" + f.filename().string(), em);
      ++m;
    }
  }
  CHECK(m >= 4, "at least four axisymmetric examples, benchmarks and G0 designs were compared: " + std::to_string(m));
#endif
  // the curves of the two fixtures: the EM twin of the axisymmetric one is the document itself; of the planar one, the
  // cross-section design (the same planar shapes and paths)
  {
    json em = load("curves_axisymmetric.json");
    for (const char* k : {"physics", "boundaries", "outputs"}) em.erase(k);
    compare_geometry("curves_axisymmetric", em);
  }
  {
    json mod = load("curves_planar.json");
    json em = mod;
    for (const char* k : {"physics", "boundaries", "outputs"}) em.erase(k);
    em["symmetry"] = "xsec";
    em["cavity"] = json::object();
    em["fullwave"] = J(R"j({"modes": 6})j");
    axmp::spec::Design d;
    try {
      d = axmp::spec::parse_design(em);
    } catch (const std::exception& e) {
      fail(std::string("curves_planar: the cross-section twin was refused: ") + e.what());
      return;
    }
    const Parsed p = parse(mod);
    CHECK(!p.f.has_errors() && p.m.regions.size() == d.regions.size() && p.m.curves.size() == d.curves.size(), "curves_planar: counts");
    if (p.m.regions.size() == d.regions.size() && p.m.curves.size() == d.curves.size()) {
      for (size_t i = 0; i < d.regions.size(); ++i) CHECK(p.m.regions[i].shape.resolved() == d.regions[i].shape.resolved(), "curves_planar: region " + d.regions[i].name);
      for (size_t i = 0; i < d.curves.size(); ++i) CHECK(p.m.curves[i].path.resolved() == d.curves[i].path.resolved(), "curves_planar: curve " + d.curves[i].name);
    }
  }
}

// The refusals of the region loop are the electromagnetic parser's (same message, pointer and hint).
void check_region_errors_match_the_em_parser() {
  const json em0 = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "symmetry": "axisymmetric", "parameters": {"R": 1.0},
    "materials": {"m": {"kind": "linear", "mu_r": 2}},
    "regions": [{"name": "a", "shape": {"rect": {"r": [0, "R"], "z": [0, 1]}}, "material": "m"}]})j");
  struct Mut {
    const char* what;
    std::function<void(json&)> f;
  };
  const std::vector<Mut> muts = {
      {"an unknown region key", [](json& r) { r["bogus"] = 1; }},
      {"an unknown material", [](json& r) { r["material"] = "nope"; }},
      {"a negative mesh_size", [](json& r) { r["mesh_size"] = -1; }},
      {"a negative surface_size", [](json& r) { r["surface_size"] = -1; }},
      {"a region without a shape", [](json& r) { r.erase("shape"); }},
      {"a region without a name", [](json& r) { r.erase("name"); }},
      {"a shape with two keys", [](json& r) { r["shape"] = J(R"j({"rect": {"r": [0, 1], "z": [0, 1]}, "disk": {}})j"); }},
      {"a rect with a reversed range", [](json& r) { r["shape"] = J(R"j({"rect": {"r": [1, 0], "z": [0, 1]}})j"); }},
      {"an expression that does not evaluate", [](json& r) { r["mesh_size"] = "nope"; }},
  };
  for (const auto& mu : muts) {
    json em = em0;
    mu.f(em["regions"][0]);
    json mod = em;
    mod["materials"]["m"]["kind"] = "linear";
    mod["physics"] = J(R"j({"h": {"type": "heat.transfer"}})j");
    same_as_twin(mu.what, mod, em);
  }
  // a rect with a missing side: nlohmann's out_of_range escapes the shape parser; the model path turns it into an
  // axmp::Error that carries the pointer of the entry
  {
    json mod = em0;
    mod["regions"][0]["shape"] = J(R"j({"rect": {"r": [0, 1]}})j");
    mod["physics"] = J(R"j({"h": {"type": "heat.transfer"}})j");
    check_one("a rect without z", mod, W("", "/regions/0").with_message_has("key 'z' not found"));
  }
}

void check_matrix_documents_are_well_formed() {
  int n = 0;
  for (const M::MatrixCase& c : M::matrix_cases()) {
    const json doc = M::minimal_document(c);
    if (doc.is_null()) continue;
    ++n;
    const Parsed p = parse(doc);
    const auto errs = errors_of(p.f);
    if (c.code == "model.per_depth_key") {
      CHECK(errs.size() == 1 && errs[0].code() == "model.per_depth_key", "the matrix document of " + c.id + " gives exactly model.per_depth_key");
    } else {
      for (const auto& e : errs) fail("the matrix document of " + c.id + ": [" + e.code() + "] " + e.message() + " [at " + e.where() + "]");
    }
  }
  CHECK(n > 50, "many matrix documents were parsed: " + std::to_string(n));
}

// ============================================================================================ further checks of the grammar
// AUTHOR TEXT throughout: the note gives no text for these refusals, and the texts below are the specification.

// the order of the layers: the materials come before the mesh, and each layer's findings are in the order of the document
void check_materials_are_read_before_the_mesh() {
  json d = heated();
  at(d, "/materials/bad") = J(R"j({"bogus": 1})j");
  at(d, "/mesh/order") = 2;
  const auto errs = errors_of(parse(d).f);
  CHECK(errs.size() == 2, "a bad material and a bad mesh.order: two findings, " + std::to_string(errs.size()) + " found");
  if (errs.size() == 2) {
    CHECK(errs[0].where() == "/materials/bad" && errs[0].message() == "unknown field 'bogus'", "the material is reported first: " + errs[0].where());
    CHECK(errs[1].where() == "/mesh/order" && errs[1].code() == "model.mesh_order", "the mesh.order is reported second: " + errs[1].where());
  }
}

void check_mesh_counts_have_an_upper_bound() {
  for (const char* key : {"refinements", "threads"}) {
    for (const double bad : {1e300, 1e10, -1.0, 2.5}) {
      json d = heated();
      at(d, std::string("/mesh/") + key) = bad;
      check_one(std::string("mesh.") + key + " = " + std::to_string(bad), d,
                W("", std::string("/mesh/") + key, std::string("mesh.") + key + " must be a non-negative integer"));
    }
    json d = heated();
    at(d, std::string("/mesh/") + key) = 3;
    clean(std::string("mesh.") + key + " = 3", d);
  }
}

// a boundary may not name itself, directly or through others
void check_boundary_cycles() {
  const auto cycle = [&](const std::string& what, const char* boundaries, const Strs& expected) {
    json d = heated();
    const json add = J(boundaries);
    for (auto it = add.begin(); it != add.end(); ++it) d["boundaries"][it.key()] = it.value();
    const auto errs = errors_of(parse(d).f);
    CHECK(errs.size() == expected.size(), what + ": " + std::to_string(errs.size()) + " finding(s), want " + std::to_string(expected.size()));
    for (size_t i = 0; i < errs.size() && i < expected.size(); ++i) {
      CHECK(errs[i].code() == "boundary.cycle", what + ": the code is boundary.cycle, not '" + errs[i].code() + "'");
      CHECK(errs[i].message() == "the boundaries form a cycle: " + expected[i], what + ": message '" + errs[i].message() + "'");
      const std::string first = expected[i].substr(0, expected[i].find(' '));
      CHECK(errs[i].where() == "/boundaries/" + first, what + ": pointer '" + errs[i].where() + "'");
    }
    if (!errs.empty())
      CHECK(errs[0].hint() ==
                "A boundary may name other boundaries, in a name, a 'union', a 'difference' or an 'except', but not itself, directly or through others.",
            what + ": hint");
  };
  cycle("two names that name each other", R"j({"a": "b", "b": "a"})j", {"a -> b -> a"});
  cycle("a name that names itself", R"j({"a": "a"})j", {"a -> a"});
  cycle("an except that names its boundary", R"j({"a": {"of": "cavity", "except": ["a"]}})j", {"a -> a"});
  cycle("a union that names its boundary", R"j({"a": {"union": ["hot", "a"]}})j", {"a -> a"});
  cycle("a difference through another boundary", R"j({"a": {"difference": ["hot", "b"]}, "b": {"union": ["cold", "a"]}})j", {"a -> b -> a"});
  cycle("three boundaries", R"j({"a": "b", "b": "c", "c": "a"})j", {"a -> b -> c -> a"});
  // a chain that leads into a cycle is not a cycle of its own, and two cycles are two findings
  cycle("a chain into a cycle", R"j({"a": "b", "b": "a", "z": "a", "y": "z"})j", {"a -> b -> a"});
  cycle("two cycles", R"j({"a": "b", "b": "a", "c": "d", "d": "c"})j", {"a -> b -> a", "c -> d -> c"});
  cycle("no cycle: a diamond", R"j({"a": "hot", "b": {"union": ["a", "cold"]}, "c": {"difference": ["b", "a"]}})j", {});
  // the cycle is one finding: the names in it are still names of the document, and the conditions that use them are checked
  {
    json d = heated();
    d["boundaries"]["a"] = "b";
    d["boundaries"]["b"] = "a";
    at(d, "/physics/heat/boundary_conditions/a") = J(R"j({"kind": "insulation"})j");
    check_one("a condition on a boundary of a cycle adds nothing", d, W("boundary.cycle", "/boundaries/a", "the boundaries form a cycle: a -> b -> a"));
  }
}

// 'group' is the name of a group of an imported mesh
void check_group_needs_a_mesh_file() {
  json d = heated();
  d["boundaries"]["g"] = J(R"j({"group": "left"})j");
  check_one("a group without mesh.file", d,
            W("boundary.selector_grammar", "/boundaries/g/group",
              "boundary 'g' selects a 'group', which is a named group of an imported mesh, and this document has no 'mesh.file'"));
  at(d, "/mesh/file") = "mesh.msh";
  const auto errs = errors_of(parse(d).f);
  for (const auto& e : errs) CHECK(e.where().find("/boundaries/g") == std::string::npos, "a group with mesh.file is not refused: " + e.message());
  // the file is the presence of the key, not a value read from a section that has another error: mesh.order is refused as
  // the one error, and the group is not refused as well
  d = heated();
  d["boundaries"]["g"] = J(R"j({"group": "left"})j");
  at(d, "/mesh/file") = "mesh.msh";
  at(d, "/mesh/order") = 2;
  {
    const auto both = errors_of(parse(d).f);
    CHECK(both.size() == 1 && both[0].code() == "model.mesh_order", "mesh.file and mesh.order: only the order is refused, " + std::to_string(both.size()) + " found");
  }
  // a mesh section that is not an object is reported as that, once
  d = heated();
  d["boundaries"]["g"] = J(R"j({"group": "left"})j");
  d["mesh"] = 5;
  const auto bad = errors_of(parse(d).f);
  CHECK(bad.size() == 1 && bad[0].where() == "/mesh", "a mesh that is not an object: only that is reported");
}

// the type of a solver setting is the registry's
void check_solver_setting_types() {
  const auto solver = [&](const std::string& what, const char* settings, const char* setting, const char* type, const char* dflt, const char* sentence) {
    json d = heated();
    at(d, "/study/0/solvers") = json{{"flow", J(settings)}};
    check_one(what, d,
              W("study.solver_value", std::string("/study/0/solvers/flow/") + setting,
                std::string("the solver setting '") + setting + "' of 'flow' must be " + sentence)
                  .with_hint(std::string("The default is ") + dflt + ".")
                  .with_details(json{{"physics", "flow"}, {"setting", setting}, {"type", type}}));
  };
  // the settings and their types are the registry's: the test reads them, so that a change of the registry is followed
  const M::SolverSetting* rtol = nullptr;
  const M::SolverSetting* maxit = nullptr;
  const M::SolverSetting* search = nullptr;
  for (const auto& r : M::registry().routes)
    for (const auto& st : r.solvers)
      if (st.physics == "fluid.incompressible") {
        if (st.name == "newton_rtol") rtol = &st;
        if (st.name == "newton_maxit") maxit = &st;
        if (st.name == "line_search") search = &st;
      }
  CHECK(rtol && rtol->type == "number" && maxit && maxit->type == "integer" && search && search->type == "boolean", "the registry types the settings of the flow");
  if (!(rtol && maxit && search)) return;
  const std::string rd = rtol->default_value.dump(), md = maxit->default_value.dump(), sd = search->default_value.dump();
  solver("an object for a number", R"j({"newton_rtol": {}})j", "newton_rtol", "number", rd.c_str(), "a number");
  solver("a string for a number", R"j({"newton_rtol": "abc"})j", "newton_rtol", "number", rd.c_str(), "a number");
  solver("null for a number", R"j({"newton_rtol": null})j", "newton_rtol", "number", rd.c_str(), "a number");
  solver("a boolean for a number", R"j({"newton_rtol": true})j", "newton_rtol", "number", rd.c_str(), "a number");
  solver("a fraction for an integer", R"j({"newton_maxit": 2.5})j", "newton_maxit", "integer", md.c_str(), "a whole number");
  solver("a string for an integer", R"j({"newton_maxit": "30"})j", "newton_maxit", "integer", md.c_str(), "a whole number");
  solver("a number for a boolean", R"j({"line_search": 5})j", "line_search", "boolean", sd.c_str(), "true or false");
  solver("a string for a boolean", R"j({"line_search": "true"})j", "line_search", "boolean", sd.c_str(), "true or false");
  json d = heated();
  at(d, "/study/0/solvers") = J(R"j({"flow": {"newton_rtol": 1e-10, "newton_maxit": 40.0, "line_search": false, "n_picard": 2}})j");
  clean("the right types", d);
  at(d, "/study/0/solvers") = J(R"j({"flow": {"newton_rtol": 1, "newton_maxit": 40, "line_search": true}})j");
  clean("an integer for a number", d);
}

// a list that is empty reads as a sentence
void check_empty_lists_read_as_sentences() {
  const json base = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "symmetry": "planar",
                            "regions": [{"name": "a", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}}], "physics": {}})j");
  const Strs none;
  {
    json d = base;
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "physics": ["flow"]}])j");
    check_one("a study step that solves a physics when there is none", d,
              W("study.unknown_physics", "/study/0/physics/0", "unknown physics 'flow'").with_hint("This document defines no physics."));
    d = base;
    d["study"] = J(R"j([{"id": "s", "type": "stationary", "solvers": {"flow": {}}}])j");
    check_one("solver settings of a physics when there is none", d,
              W("study.unknown_physics", "/study/0/solvers/flow", "unknown physics 'flow'").with_hint("This document defines no physics."));
    d = base;
    d["couplings"] = J(R"j([{"term": "advection", "from": "flow", "to": "heat"}])j");
    check_one("a coupling between physics when there is none", d,
              W("coupling.unknown_physics", "/couplings/0/from", "unknown physics 'flow'").with_hint("This document defines no physics."));
    d = base;
    d["outputs"] = J(R"j({"o": {"kind": "average", "of": "T", "in": "a", "physics": "heat"}})j");
    check_one("an output of a physics when there is none", d,
              W("output.unknown_physics", "/outputs/o/physics", "unknown physics 'heat'").with_hint("This document defines no physics."));
    d = base;
    d["physics"] = J(R"j({"heat": {"type": "heat.transfer"}})j");
    d["outputs"] = J(R"j({"o": {"kind": "flux", "of": "heat", "on": "nowhere"}})j");
    check_one("an output on a boundary when there is none", d,
              W("output.unknown_boundary", "/outputs/o/on", "unknown boundary 'nowhere'").with_hint("This document defines no boundaries."));
  }
  // the builders themselves, for the lists that no small document empties
  CHECK(E::output_unknown_quantity("q", "planar", "/w", none).hint() == "This document has no quantities on a planar model.", "no quantities: " + E::output_unknown_quantity("q", "planar", "/w", none).hint());
  CHECK(E::list_of("options", "x.y", none) == "'x.y' has no options.", "no options: " + E::list_of("options", "x.y", none));
  CHECK(E::list_of("options", "x.y", Strs{"a", "b"}) == "The options of 'x.y' are: a, b.", "the list of options is as it was");
  CHECK(E::study_unknown_physics("p", "/w", Strs{"a", "b"}).hint() == "The physics of this document are: a, b.", "the list of physics is as it was");
}

// one parameter that shadows a variable is one finding for each set of variables
void check_a_shadowing_parameter_is_reported_once() {
  json d = heated();
  at(d, "/parameters/x") = 0.5;
  at(d, "/physics/flow/boundary_conditions/hot") = J(R"j({"kind": "inlet", "velocity_m_s": [0, 1]})j");
  {
    // the inlet of the flow and the two temperatures of the heat each have an expression: one cause, one finding
    const auto errs = errors_of(parse(d).f);
    size_t n = 0;
    for (const auto& e : errs) n += e.code() == "expr.parameter_shadows_variable";
    CHECK(n == 1 && errs.size() == 1, "the parameter x clashes in three conditions: one finding, " + std::to_string(errs.size()) + " found");
    if (errs.size() == 1) CHECK(errs[0].where() == "/physics/flow/boundary_conditions/hot/velocity_m_s/0", "the first is reported: " + errs[0].where());
  }
  // a transient step makes t a variable of the heat, so the clash is found with the variables (x, y) in the flow and
  // (x, y, t) in the heat: the message is the same, so it is one cause and one finding
  at(d, "/study") = J(R"j([{"id": "s", "type": "transient", "physics": ["heat"], "time": {"t_end_s": 1.0}}])j");
  {
    const auto errs = errors_of(parse(d).f);
    CHECK(errs.size() == 1 && errs[0].code() == "expr.parameter_shadows_variable",
          "one finding for a parameter that clashes with the variables of two physics, " + std::to_string(errs.size()) + " found");
    if (errs.size() == 1) CHECK(errs[0].where() == "/physics/flow/boundary_conditions/hot/velocity_m_s/0", "the first is reported: " + errs[0].where());
  }
  // only the first clashing variable of a set is named, so a second parameter is named once the first is renamed
  json e = heated();
  at(e, "/parameters/x") = 0.5;
  at(e, "/parameters/y") = 0.5;
  {
    const auto errs = errors_of(parse(e).f);
    CHECK(errs.size() == 1, "the first clashing variable of a set is the one named");
  }
}

// a boundary that is defined with a reserved name is refused where it is defined, and not once more wherever it is named
void check_a_reserved_boundary_is_refused_once() {
  for (const char* name : {"axis", "exterior", "remaining"}) {
    const std::string at_def = std::string("/boundaries/") + name;
    const auto defined = [&] {
      json d = heated();
      at(d, at_def) = J(R"j({"of": "cavity"})j");
      return d;
    };
    const auto one = [&](const std::string& what, const json& d) {
      const auto errs = errors_of(parse(d).f);
      std::string all;
      for (const auto& e : errs) all += "\n    [" + e.code() + "] " + e.where();
      CHECK(errs.size() == 1 && errs[0].code() == "boundary.reserved_name" && errs[0].where() == at_def,
            std::string(name) + ", " + what + ": one finding at its definition, " + std::to_string(errs.size()) + " found" + all);
    };
    json d = defined();
    one("defined only", d);
    d = defined();
    at(d, "/boundaries/ref") = name;
    one("named as the whole selector of another boundary", d);
    d = defined();
    at(d, "/boundaries/u") = json{{"union", json::array({name, "hot"})}};
    one("a member of a union", d);
    d = defined();
    at(d, "/boundaries/w") = json{{"of", "cavity"}, {"except", json::array({name})}};
    one("an entry of except", d);
    d = defined();
    at(d, "/outputs/Qx") = json{{"kind", "flux"}, {"of", "heat"}, {"on", name}};
    one("the boundary of an output", d);
    // a condition key and a periodic partner
    d = defined();
    at(d, std::string("/physics/flow/boundary_conditions/") + name) = J(R"j({"kind": "wall"})j");
    one("a condition key", d);
    d = defined();
    at(d, "/physics/flow/boundary_conditions/hot") = json{{"kind", "periodic"}, {"partner", name}, {"translation_m", J("[0.046, 0]")}};
    one("a periodic partner", d);
  }
}

// a reserved name is reported as one wherever it is used as the name of a boundary
void check_reserved_names_in_every_context() {
  for (const char* name : {"axis", "exterior", "remaining"}) {
    json d = heated();
    at(d, "/outputs/Qx") = json{{"kind", "flux"}, {"of", "heat"}, {"on", name}};
    check_one(std::string("an output on '") + name + "'", d, W("boundary.reserved_name", "/outputs/Qx/on", E::reserved_name(name, "").message()));
    d = heated();
    at(d, "/physics/flow/boundary_conditions/hot") = json{{"kind", "periodic"}, {"partner", name}, {"translation_m", J("[0.046, 0]")}};
    check_one(std::string("a periodic partner '") + name + "'", d,
              W("boundary.reserved_name", "/physics/flow/boundary_conditions/hot/partner", E::reserved_name(name, "").message()));
  }
  // a name that is not reserved is an unknown boundary, as it was
  json d = heated();
  at(d, "/outputs/Qx") = J(R"j({"kind": "flux", "of": "heat", "on": "elsewhere"})j");
  check_one("an unknown boundary of an output", d, W("output.unknown_boundary", "/outputs/Qx/on", "unknown boundary 'elsewhere'"));
  // a boundary that was refused for its name where it is defined adds nothing where it is used
  d = heated();
  at(d, "/boundaries/axis") = J(R"j({"of": "cavity", "where": {"y": 0}})j");
  at(d, "/outputs/Qx") = J(R"j({"kind": "flux", "of": "heat", "on": "axis"})j");
  check_one("a boundary named axis, used by an output", d, W("boundary.reserved_name", "/boundaries/axis"));
}

// the vector of a periodic translation has the components of the plane or the space of the model
void check_number_vector_length() {
  const auto periodic = [&](const char* sym, const char* translation) {
    json d = heated();
    d["symmetry"] = sym;
    at(d, "/physics/flow/boundary_conditions/hot") = json{{"kind", "periodic"}, {"partner", "cold"}, {"translation_m", J(translation)}};
    return d;
  };
  check_one("three components in a planar model", periodic("planar", "[0.046, 0, 0]"),
            W("expr.vector_length", "/physics/flow/boundary_conditions/hot/translation_m", "'translation_m' has 3 components, and this planar model needs 2 (x, y)")
                .with_details(J(R"j({"key": "translation_m", "found": 3, "expected": 2})j")));
  check_one("one component in a planar model", periodic("planar", "[0.046]"),
            W("expr.vector_length", "/physics/flow/boundary_conditions/hot/translation_m", "'translation_m' has 1 components, and this planar model needs 2 (x, y)"));
  const Parsed ok = parse(periodic("planar", "[0.046, 0]"));
  CHECK(!ok.f.has_errors(), "two components in a planar model");
}

// the loose corners of the study and of the selectors
void check_study_and_selector_values() {
  const auto step = [&](const char* text) {
    json d = heated();
    at(d, "/study") = J(text);
    return d;
  };
  check_one("an empty step id", step(R"j([{"id": "", "type": "stationary"}])j"), W("", "/study/0/id", "the id of a study step must not be empty"));
  check_one("an empty physics list", step(R"j([{"id": "s", "type": "stationary", "physics": []}])j"),
            W("", "/study/0/physics", "a study step must solve at least one physics"));
  check_one("save with both n and times_s", step(R"j([{"id": "s", "type": "stationary", "save": {"n": 5, "times_s": [1.0]}}])j"),
            W("", "/study/0/save", "'save' takes 'times_s' or 'n', not both"));
  check_one("unsorted times", step(R"j([{"id": "s", "type": "stationary", "save": {"times_s": [1.0, 5.0, 2.0]}}])j"),
            W("", "/study/0/save/times_s/2", "'times_s' must be strictly increasing"));
  check_one("repeated times", step(R"j([{"id": "s", "type": "stationary", "save": {"times_s": [1.0, 1.0]}}])j"),
            W("", "/study/0/save/times_s/1", "'times_s' must be strictly increasing"));
  check_one("a negative time", step(R"j([{"id": "s", "type": "stationary", "save": {"times_s": [-1.0, 2.0]}}])j"),
            W("", "/study/0/save/times_s/0", "a time of 'times_s' must not be negative"));
  check_one("no times", step(R"j([{"id": "s", "type": "stationary", "save": {"times_s": []}}])j"), W("", "/study/0/save/times_s", "'times_s' must list at least one time"));
  clean("sorted times from zero", step(R"j([{"id": "s", "type": "stationary", "save": {"times_s": [0.0, 1.0, 5.0]}}])j"));
  clean("n alone", step(R"j([{"id": "s", "type": "stationary", "save": {"n": 5}}])j"));

  for (const char* tol : {"-1e-9", "0"}) {
    json d = heated();
    at(d, "/boundaries/hot") = json::parse(std::string(R"j({"of": "cavity", "where": {"x": 0}, "tolerance_m": )j") + tol + "}");
    check_one(std::string("tolerance_m = ") + tol, d, W("boundary.selector_grammar", "/boundaries/hot/tolerance_m", "'tolerance_m' must be a positive length in metres"));
  }
  {
    json d = heated();
    at(d, "/boundaries/hot/tolerance_m") = 1e-6;
    clean("a positive tolerance", d);
  }
  {
    json d = heated();
    at(d, "/physics/heat/domains") = json::array();
    check_one("an instance with no domains", d, W("", "/physics/heat/domains", "domains must list at least one region or domain"));
    d = heated();
    at(d, "/domains") = J(R"j({"none": []})j");
    check_one("a domain with no regions", d, W("", "/domains/none", "a domain must list at least one region"));
  }
}

// the keys of an output: what the kind takes, the components, 'over', and the names of the grammar
void check_output_keys() {
  {
    json d = heated();
    at(d, "/outputs/dp") = J(R"j({"kind": "expression", "expr": "Q_hot", "of": "T"})j");
    check_one("of on an expression output", d,
              W("output.grammar", "/outputs/dp/of", "output 'dp' of kind 'expression' does not take 'of'")
                  .with_hint("The keys of an output of kind 'expression' are: kind, over, notes, expr.")
                  .with_details(J(R"j({"name": "dp", "kind": "expression", "key": "of"})j")));
    d = heated();
    at(d, "/outputs/Q_hot/expr") = "1";
    check_one("expr on a flux output", d, W("output.grammar", "/outputs/Q_hot/expr", "output 'Q_hot' of kind 'flux' does not take 'expr'"));
    d = heated();
    at(d, "/outputs/Q_hot/in") = "cavity";
    check_one("in on a flux output", d, W("output.grammar", "/outputs/Q_hot/in", "output 'Q_hot' of kind 'flux' does not take 'in'"));
    d = heated();
    at(d, "/outputs/F") = J(R"j({"kind": "force", "on": "hot", "of": "p"})j");
    check_one("of on a force output", d, W("output.grammar", "/outputs/F/of", "output 'F' of kind 'force' does not take 'of'"));
    d = heated();
    at(d, "/outputs/pt") = J(R"j({"kind": "point", "of": "T", "at": [0.01, 0.01], "on": "hot"})j");
    check_one("on on a point output", d, W("output.grammar", "/outputs/pt/on", "output 'pt' of kind 'point' does not take 'on'"));
    d = heated();
    at(d, "/outputs/Qh") = J(R"j({"kind": "flux", "of": "heat", "on": "hot", "T_datum_K": 293.15})j");
    clean("T_datum_K on a flux output", d);
    d = heated();
    at(d, "/outputs/dp") = J(R"j({"kind": "expression", "expr": "Q_hot", "physics": "heat"})j");
    check_one("physics on an expression output", d, W("output.grammar", "/outputs/dp/physics", "output 'dp' of kind 'expression' does not take 'physics'"));
  }
  // component: the components of a vector in the geometry; a torque has the one about the normal of the plane, or the three
  const auto force = [&](json d, const char* kind, const char* component) {
    at(d, "/outputs/Fq") = json{{"kind", kind}, {"on", "hot"}, {"component", component}};
    return d;
  };
  for (const char* c : {"x", "y"}) clean(std::string("force component ") + c + ", planar", force(heated(), "force", c));
  clean("torque component z, planar", force(heated(), "torque", "z"));
  for (const char* c : {"q", "z", "r", "X", ""})
    check_one(std::string("force component '") + c + "', planar", force(heated(), "force", c),
              W("output.grammar", "/outputs/Fq/component", std::string("unknown component '") + c + "' of the force in a planar model")
                  .with_hint("The components of a force in a planar model are: x, y."));
  check_one("torque component x, planar", force(heated(), "torque", "x"),
            W("output.grammar", "/outputs/Fq/component", "unknown component 'x' of the torque in a planar model").with_hint("The components of a torque in a planar model are: z."));
  {
    json a = pipe();
    at(a, "/outputs/Fz") = J(R"j({"kind": "force", "on": "wall", "component": "z"})j");
    clean("force component z, axisymmetric", a);
    at(a, "/outputs/Fz/component") = "x";
    check_one("force component x, axisymmetric", a, W("output.grammar", "/outputs/Fz/component", "unknown component 'x' of the force in an axisymmetric model")
                                                       .with_hint("The components of a force in an axisymmetric model are: z."));
    // the flow engines of an axisymmetric model return the axial force and the axial torque only
    for (const char* c : {"r", "theta"}) {
      at(a, "/outputs/Fz/component") = c;
      check_one(std::string("force component ") + c + ", axisymmetric", a,
                W("output.grammar", "/outputs/Fz/component", std::string("unknown component '") + c + "' of the force in an axisymmetric model")
                    .with_hint("The components of a force in an axisymmetric model are: z."));
    }
    at(a, "/outputs/Fz") = J(R"j({"kind": "torque", "on": "wall", "component": "z"})j");
    clean("torque component z, axisymmetric", a);
    at(a, "/outputs/Fz/component") = "r";
    check_one("torque component r, axisymmetric", a,
              W("output.grammar", "/outputs/Fz/component", "unknown component 'r' of the torque in an axisymmetric model")
                  .with_hint("The components of a torque in an axisymmetric model are: z."));
  }
  // in 3D a force and a torque have the three Cartesian components, and neither has the cylindrical ones
  {
    json b = busbar();
    for (const char* kind : {"force", "torque"}) {
      for (const char* c : {"x", "y", "z"}) {
        at(b, "/outputs/Fb") = json{{"kind", kind}, {"on", "skin"}, {"component", c}};
        clean(std::string(kind) + " component " + c + ", 3d", b);
      }
      for (const char* c : {"r", "theta"}) {
        at(b, "/outputs/Fb") = json{{"kind", kind}, {"on", "skin"}, {"component", c}};
        check_one(std::string(kind) + " component " + c + ", 3d", b,
                  W("output.grammar", "/outputs/Fb/component", std::string("unknown component '") + c + "' of the " + kind + " in a 3d model")
                      .with_hint(std::string("The components of a ") + kind + " in a 3d model are: x, y, z."));
      }
    }
  }
  // the keys that belong to a force or a torque are refused on the other kinds, the integral family among them
  for (const char* kind : {"integral", "average", "maximum", "minimum", "extreme"}) {
    for (const char* key : {"component", "at", "about"}) {
      json b = busbar();
      at(b, "/outputs/Zq") = json{{"kind", kind}, {"of", "T"}, {"in", "metal"}};
      at(b, std::string("/outputs/Zq/") + key) = std::string(key) == "component" ? json("z") : J("[0, 0, 0]");
      check_one(std::string(key) + " on an output of kind " + kind, b,
                W("output.grammar", std::string("/outputs/Zq/") + key, std::string("output 'Zq' of kind '") + kind + "' does not take '" + key + "'"));
    }
  }
  for (const char* key : {"component", "of", "in"}) {
    json b = busbar();
    at(b, "/outputs/Ie") = J(R"j({"kind": "electrode_current", "on": "terminal"})j");
    at(b, std::string("/outputs/Ie/") + key) = std::string(key) == "component" ? json("z") : json(std::string(key) == "of" ? "T" : "metal");
    check_one(std::string(key) + " on an electrode_current output", b,
              W("output.grammar", std::string("/outputs/Ie/") + key, std::string("output 'Ie' of kind 'electrode_current' does not take '") + key + "'"));
  }
  // over
  {
    json d = heated();
    at(d, "/outputs/Q_hot/over") = "time";
    clean("over time", d);
    at(d, "/outputs/Q_hot/over") = "space";
    check_one("over space", d, W("output.grammar", "/outputs/Q_hot/over", "output 'Q_hot' has an unknown 'over'")
                                    .with_hint("The only series is over the time of a transient step: \"over\": \"time\"."));
    at(d, "/outputs/Q_hot/over") = J(R"j({"a": 1})j");
    check_one("over an object", d, W("", "/outputs/Q_hot/over", "expected a string"));
  }
  // the names of the grammar: its constants and its functions (the note, section 7, and the reference evaluator's list)
  for (const char* name : {"pi", "e", "mu0"}) {
    json d = heated();
    at(d, std::string("/outputs/") + name) = J(R"j({"kind": "average", "of": "T", "in": "cavity"})j");
    check_one(std::string("an output named ") + name, d,
              W("output.grammar", std::string("/outputs/") + name, std::string("'") + name + "' cannot be the name of an output: it is a constant of the expression grammar"));
  }
  for (const char* name : {"sqrt", "exp", "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "abs", "pow", "min", "max"}) {
    json d = heated();
    at(d, std::string("/outputs/") + name) = J(R"j({"kind": "average", "of": "T", "in": "cavity"})j");
    check_one(std::string("an output named ") + name, d,
              W("output.grammar", std::string("/outputs/") + name, std::string("'") + name + "' cannot be the name of an output: it is a function of the expression grammar"));
  }
  // the grammar is asked for the arity message of the name, so a name that only occurs in the message of an unknown function
  // ("unknown function 'argument'") is not a function
  for (const char* name : {"argument", "arguments", "n_arguments", "takes", "argument_1"}) {
    json d = heated();
    at(d, std::string("/outputs/") + name) = J(R"j({"kind": "average", "of": "T", "in": "cavity"})j");
    clean(std::string("an output named ") + name, d);
  }
  for (const char* name : {"Pi", "E", "e1", "sqrt2", "mu1", "pi_", "minimum", "T_avg"}) {
    json d = heated();
    at(d, std::string("/outputs/") + name) = J(R"j({"kind": "average", "of": "T", "in": "cavity"})j");
    clean(std::string("an output named ") + name, d);
  }
}

}  // namespace

int main() {
  check_four_documents_parse();
  check_is_model_document_looks_only_at_the_key();
  check_findings_sink();
  check_grammar_errors_are_collected_per_section();
  check_symmetry_rules();
  check_identity();
  check_mesh_keys();
  check_em_sections_are_refused();
  check_names();
  check_curve_keys_in_a_model_document();
  check_selector_grammar();
  check_unknown_type_option_kind();
  check_per_depth_keys();
  check_field_expressions();
  check_couplings_grammar();
  check_study_grammar();
  check_outputs_grammar();
  check_reactions_grammar();
  check_region_loop_parity();
  check_region_errors_match_the_em_parser();
  check_matrix_documents_are_well_formed();
  check_materials_are_read_before_the_mesh();
  check_mesh_counts_have_an_upper_bound();
  check_boundary_cycles();
  check_group_needs_a_mesh_file();
  check_solver_setting_types();
  check_empty_lists_read_as_sentences();
  check_a_shadowing_parameter_is_reported_once();
  check_a_reserved_boundary_is_refused_once();
  check_reserved_names_in_every_context();
  check_number_vector_length();
  check_study_and_selector_values();
  check_output_keys();
  if (failures != 0) {
    std::printf("%d model-document check(s) failed\n", failures);
    return 1;
  }
  std::puts("model document parser (four documents, collection, names, selectors, catalogue checks, expressions, study, outputs): passed");
  return 0;
}
