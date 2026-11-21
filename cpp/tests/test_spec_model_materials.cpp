// The materials of a model document (design note section 6, the material split of section 6.2): parse_model_material and
// its four steps, the model's key lists, the new blocks and keys, the kind rule, the two ambiguity refusals, the property
// laws (section 6.3), and the pointers of every error.  Built in every build.
//
// INDEPENDENT REFERENCES.
//   * The unchanged electromagnetic parser, parse_design, on the "twin" of each material: the same node in an
//     electromagnetic document.  Where the model path applies the electromagnetic grammar (E_Pa > 0, -1 < nu < 0.5, the kind
//     names, the rf model, the [[T, k]] table), its message, pointer and hint must equal the twin's, and no text is typed
//     here for them.  The twin also supplies the values that the electromagnetic Material holds.
//   * axmp::eval_expr, the evaluator of the design grammar, for an expression that does not evaluate.
//   * Hand arithmetic for the values: 2 * 0.5e-3 = 1e-3, 5.998e7 / (1 + 3.9e-3 (T - 293.15)) is not evaluated (a law is
//     parsed here, not evaluated).
//
// TEXTS.  A text that the note gives is copied from it (section 6.2: the expansion refusal).  Where the note gives no text
// the text below is the test author's and is the specification (marked AUTHOR TEXT).
//
// THE API THE TESTS USE (see test_spec_model.cpp for the findings and the parse functions):
//   axmp::spec::parse_model(json) / parse_model_collect(json, Findings&) -> ResolvedModel, with
//   ResolvedModel::materials: std::map<std::string, MaterialProps>; MaterialProps { spec::Material em; Property density, mu_Pa_s,
//   beta_1_K, kappa_S_m, electric_sigma_S_m, k_W_mK, cp_J_kgK, alpha_1_K; bool has_fluid, has_electrolyte, has_electric,
//   has_species; json species; }; Property { bool given; double value; std::shared_ptr<const Law> law; is_law() };
//   Law { kind, of, expr, x, y, interpolation, range, out_of_range, source, allow_c0 };
//   axmp::spec::detail::parse_model_material(const std::string& name, const detail::Node& node) -> MaterialProps, declared
//   in spec_model.hpp (with a forward declaration of detail::Node), throwing the first axmp::Error of the material.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/expr.hpp"
#include "axionmultiphysics/spec.hpp"
#include "axionmultiphysics/spec_model.hpp"
#include "../src/model_errors.hpp"
#include "../src/spec_internal.hpp"

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

// ======================================================================================================= helpers
std::string join(const Strs& v, const std::string& sep = ", ") {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
  return s;
}
std::string only_list(const std::set<std::string>& keys) { return "Allowed fields here: " + join(Strs(keys.begin(), keys.end())) + "."; }
bool close(double a, double b, double rel = 1e-14) { return std::fabs(a - b) <= rel * std::max(std::fabs(a), std::fabs(b)); }
json J(const char* s) { return json::parse(s); }

// A model document of one region made of the material `name`, and the electromagnetic twin of it (the same node, with
// "kind" added where the node has none: the electromagnetic grammar requires it).
json model_doc(const std::string& name, const json& material, const json& parameters = json::object()) {
  json d = J(R"j({"schema": "axionmultiphysics.design/1", "name": "t", "symmetry": "axisymmetric",
                  "regions": [{"name": "r", "shape": {"rect": {"r": [0, 1], "z": [0, 1]}}}],
                  "physics": {"h": {"type": "heat.transfer"}}})j");
  d["regions"][0]["material"] = name;
  d["materials"] = json::object();
  d["materials"][name] = material;
  if (!parameters.empty()) d["parameters"] = parameters;
  return d;
}
json em_doc(const std::string& name, json material, const json& parameters = json::object()) {
  if (!material.contains("kind")) material["kind"] = "vacuum";
  json d = model_doc(name, material, parameters);
  d.erase("physics");
  return d;
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
std::vector<Error> errors_of(const M::Findings& f) {
  std::vector<Error> out;
  for (const auto& x : f.all())
    if (x.kind == M::Finding::Kind::error) out.push_back(x.error);
  return out;
}
json details_of(const Error& e) { return e.details().empty() ? json(nullptr) : json::parse(e.details()); }
bool subset(const json& have, const json& want) {
  if (want.is_object()) {
    if (!have.is_object()) return false;
    for (auto it = want.begin(); it != want.end(); ++it)
      if (!have.contains(it.key()) || !subset(have.at(it.key()), it.value())) return false;
    return true;
  }
  return have == want;
}

struct Want {
  std::string code, where, message, message_has, hint, hint_has;
  json details = nullptr;
  size_t count = 1;
  Want& with_hint(std::string h) { hint = std::move(h); return *this; }
  Want& with_hint_has(std::string h) { hint_has = std::move(h); return *this; }
  Want& with_message_has(std::string m) { message_has = std::move(m); return *this; }
  Want& with_details(json d) { details = std::move(d); return *this; }
};
Want W(std::string code, std::string where, std::string message = "") {
  Want w;
  w.code = std::move(code);
  w.where = std::move(where);
  w.message = std::move(message);
  return w;
}
void compare_error(const std::string& what, const Error& e, const Want& w) {
  if (e.code() != w.code) fail(what + ": code '" + e.code() + "', want '" + w.code + "' (" + e.what() + ")");
  if (e.where() != w.where) fail(what + ": pointer '" + e.where() + "', want '" + w.where + "'");
  if (!w.message.empty() && e.message() != w.message) fail(what + ": message\n    '" + e.message() + "'\n  want\n    '" + w.message + "'");
  if (!w.message_has.empty() && e.message().find(w.message_has) == std::string::npos) fail(what + ": message '" + e.message() + "' lacks '" + w.message_has + "'");
  if (!w.hint.empty() && e.hint() != w.hint) fail(what + ": hint\n    '" + e.hint() + "'\n  want\n    '" + w.hint + "'");
  if (!w.hint_has.empty() && e.hint().find(w.hint_has) == std::string::npos) fail(what + ": hint '" + e.hint() + "' lacks '" + w.hint_has + "'");
  if (!w.details.is_null()) {
    const json d = details_of(e);
    if (!subset(d, w.details)) fail(what + ": details " + d.dump() + " do not contain " + w.details.dump());
  }
}
// exactly one error, parse_model throws it and nothing else
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
    if (std::string(t.what()) != errs.front().what() || t.code() != errs.front().code()) fail(what + ": parse_model threw '" + t.what() + "', not the first error");
  } catch (const std::exception& t) {
    fail(what + ": parse_model threw a bare exception: " + t.what());
  }
}
bool clean(const std::string& what, const json& doc) {
  const auto errs = errors_of(parse(doc).f);
  for (const auto& e : errs) fail(what + ": unexpected error [" + e.code() + "] " + e.what() + " [at " + e.where() + "]");
  return errs.empty();
}

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
// the model path refuses the material as the electromagnetic parser refuses its twin: same message, pointer and hint.  The
// twin is the same node unless `em_material` is given (a node without the keys that only a model document has).
void same_as_twin(const std::string& what, const std::string& name, const json& material, const json& em_material = json(nullptr),
                  const json& parameters = json::object()) {
  const Triple t = em_error(em_doc(name, em_material.is_null() ? material : em_material, parameters));
  if (!t.threw) {
    fail(what + ": the electromagnetic twin was not refused, so there is no reference");
    return;
  }
  const auto errs = errors_of(parse(model_doc(name, material, parameters)).f);
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
const M::MaterialProps* material_of(const Parsed& p, const std::string& name) {
  const auto it = p.m.materials.find(name);
  return it == p.m.materials.end() ? nullptr : &it->second;
}

// the key lists of the model document's materials (section 6.2 step 1)
const std::set<std::string> kMaterialKeys = {"kind", "mu_r", "eps_r", "tan_delta", "sigma_S_m", "bh", "B_sat", "B", "H", "density", "warm", "cost_per_kg", "rf",
                                              "mechanical", "thermal", "fluid", "electrolyte", "electric", "species"};
const std::set<std::string> kThermalKeys = {"k_W_mK", "cp_J_kgK", "notes"};
const std::set<std::string> kMechanicalKeys = {"E_Pa", "nu", "alpha", "alpha_1_K", "notes"};
// AUTHOR TEXT: the new blocks accept a free-text 'notes' like thermal and mechanical.
const std::set<std::string> kFluidKeys = {"mu_Pa_s", "beta_1_K", "notes"};
const std::set<std::string> kElectrolyteKeys = {"kappa_S_m", "notes"};
const std::set<std::string> kElectricKeys = {"sigma_S_m", "notes"};

// ============================================================================================================ checks

void check_unknown_key_at_each_level_lists_the_model_keys() {
  // the reference: the electromagnetic lists, which the model path must never print
  {
    json em = em_doc("m", J(R"j({"kind": "linear", "bogus": 1})j"));
    const Triple t = em_error(em);
    CHECK(t.hint == "Allowed fields here: B, B_sat, H, bh, cost_per_kg, density, eps_r, kind, mechanical, mu_r, rf, sigma_S_m, tan_delta, thermal, warm.",
          "the electromagnetic material list is what it was (the reference of this test): " + t.hint);
  }
  check_one("an unknown material key", model_doc("m", J(R"j({"kind": "linear", "bogus": 1})j")),
            W("", "/materials/m", "unknown field 'bogus'").with_hint(only_list(kMaterialKeys)));
  check_one("an unknown thermal key", model_doc("m", J(R"j({"thermal": {"k_W_mK": 1.0, "bogus": 1}})j")),
            W("", "/materials/m/thermal", "unknown field 'bogus'").with_hint(only_list(kThermalKeys)));
  check_one("an unknown mechanical key", model_doc("m", J(R"j({"mechanical": {"E_Pa": 1e9, "nu": 0.3, "bogus": 1}})j")),
            W("", "/materials/m/mechanical", "unknown field 'bogus'").with_hint(only_list(kMechanicalKeys)));
  check_one("an unknown fluid key", model_doc("m", J(R"j({"fluid": {"mu_Pa_s": 1e-3, "bogus": 1}})j")),
            W("", "/materials/m/fluid", "unknown field 'bogus'").with_hint(only_list(kFluidKeys)));
  check_one("an unknown electrolyte key", model_doc("m", J(R"j({"electrolyte": {"kappa_S_m": 1.0, "bogus": 1}})j")),
            W("", "/materials/m/electrolyte", "unknown field 'bogus'").with_hint(only_list(kElectrolyteKeys)));
  check_one("an unknown electric key", model_doc("m", J(R"j({"electric": {"sigma_S_m": 1.0, "bogus": 1}})j")),
            W("", "/materials/m/electric", "unknown field 'bogus'").with_hint(only_list(kElectricKeys)));
  // the keys that the electromagnetic grammar does not know are accepted in a model document, and refused by the
  // electromagnetic parser with its own list (so that no `only` list of that parser was widened)
  const json all_new = J(R"j({"kind": "linear",
      "thermal": {"k_W_mK": 0.6, "cp_J_kgK": 4182.0, "notes": "n"},
      "mechanical": {"E_Pa": 2e9, "nu": 0.3, "alpha_1_K": 7e-5, "notes": "n"},
      "fluid": {"mu_Pa_s": 1e-3, "beta_1_K": 2e-4, "notes": "n"}, "electrolyte": {"kappa_S_m": 1.0, "notes": "n"},
      "electric": {"sigma_S_m": 1e6, "notes": "n"}, "species": {"A": {"D_m2_s": 1e-9, "z": 1}}})j");
  clean("every new block and key", model_doc("m", all_new));
  for (const char* where : {"fluid", "electrolyte", "electric", "species"}) {
    json node = J(R"j({"kind": "linear"})j");
    node[where] = all_new[where];
    const Triple t = em_error(em_doc("m", node));
    CHECK(t.threw && t.message == std::string("unknown field '") + where + "'" && t.where == "/materials/m",
          std::string("the electromagnetic parser still refuses the block '") + where + "' with its own text: " + t.message);
  }
  {
    const Triple t = em_error(em_doc("m", J(R"j({"kind": "linear", "thermal": {"k_W_mK": 1.0, "cp_J_kgK": 2.0}})j")));
    CHECK(t.threw && t.message == "unknown field 'cp_J_kgK'" && t.where == "/materials/m/thermal" && t.hint == "Allowed fields here: k_W_mK, notes.",
          "the electromagnetic parser still refuses thermal.cp_J_kgK with its own list: " + t.hint);
    const Triple u = em_error(em_doc("m", J(R"j({"kind": "linear", "mechanical": {"E_Pa": 1e9, "nu": 0.3, "alpha_1_K": 1e-5}})j")));
    CHECK(u.threw && u.message == "unknown field 'alpha_1_K'" && u.where == "/materials/m/mechanical", "and mechanical.alpha_1_K");
  }
  const Parsed p = parse(model_doc("m", all_new));
  if (const M::MaterialProps* m = material_of(p, "m")) {
    CHECK(m->has_fluid && m->has_electrolyte && m->has_electric && m->has_species, "the blocks are recorded");
    CHECK(m->species == J(R"j({"A": {"D_m2_s": 1e-9, "z": 1}})j"), "the species block is kept as written");
    CHECK(close(m->mu_Pa_s.value, 1e-3) && close(m->beta_1_K.value, 2e-4) && close(m->kappa_S_m.value, 1.0) && close(m->electric_sigma_S_m.value, 1e6) &&
              close(m->cp_J_kgK.value, 4182.0) && close(m->k_W_mK.value, 0.6) && close(m->alpha_1_K.value, 7e-5),
          "the new data");
    CHECK(m->mu_Pa_s.given && m->k_W_mK.given && !m->k_W_mK.is_law(), "given, not a law");
    CHECK(m->em.mech.given && close(m->em.mech.E_Pa, 2e9) && close(m->em.mech.nu, 0.3) && m->em.mech.alpha == 0.0, "the electromagnetic part keeps E and nu, and alpha stays 0");
    CHECK(m->em.therm.given && close(m->em.therm.k_const, 0.6), "the electromagnetic thermal part keeps k");
  } else {
    fail("the material 'm' is missing");
  }
}

void check_error_inside_a_new_block_has_the_users_pointer() {
  const json water = J(R"j({"fluid": {"mu_Pa_s": -1.0}})j");
  check_one("a negative viscosity", model_doc("water", water),
            W("", "/materials/water/fluid/mu_Pa_s", "mu_Pa_s (dynamic viscosity) must be positive"));  // AUTHOR TEXT
  check_one("a zero conductivity of an electrolyte", model_doc("bath", J(R"j({"electrolyte": {"kappa_S_m": 0}})j")),
            W("", "/materials/bath/electrolyte/kappa_S_m", "kappa_S_m (electrolyte conductivity) must be positive"));  // AUTHOR TEXT
  check_one("a negative electric conductivity", model_doc("m", J(R"j({"electric": {"sigma_S_m": -5}})j")),
            W("", "/materials/m/electric/sigma_S_m", "sigma_S_m (electric conductivity) must be positive"));  // AUTHOR TEXT
  check_one("a negative specific heat", model_doc("m", J(R"j({"thermal": {"k_W_mK": 1.0, "cp_J_kgK": -2.0}})j")),
            W("", "/materials/m/thermal/cp_J_kgK", "cp_J_kgK (specific heat) must be positive"));  // AUTHOR TEXT
  // a number that is an expression over the parameters is evaluated, with the user's pointer when it does not evaluate
  {
    const json d = model_doc("water", J(R"j({"fluid": {"mu_Pa_s": "2*mu_half"}})j"), J(R"j({"mu_half": 0.5e-3})j"));
    if (clean("an expression over the parameters", d)) CHECK(close(material_of(parse(d), "water")->mu_Pa_s.value, 1e-3), "2 * 0.5e-3 = 1e-3");
    json bad = model_doc("water", J(R"j({"fluid": {"mu_Pa_s": "nope"}})j"));
    std::string msg, hint;
    try {
      axmp::eval_expr("nope", axmp::ParamMap{}, "/materials/water/fluid/mu_Pa_s");
    } catch (const Error& e) {
      msg = e.message();
      hint = e.hint();
    }
    check_one("an expression that does not evaluate", bad, W("", "/materials/water/fluid/mu_Pa_s", msg).with_hint(hint));
    check_one("a string that is not a number", model_doc("water", J(R"j({"fluid": {"mu_Pa_s": [1]}})j")),
              W("", "/materials/water/fluid/mu_Pa_s", "expected a number or an expression string"));
  }
  check_one("a block that is not an object", model_doc("m", J(R"j({"fluid": 5})j")), W("", "/materials/m/fluid", "expected an object"));
  // an error in one block does not hide an error in another: the first of the material is the one reported, once
  check_one("two errors in one material give one", model_doc("m", J(R"j({"electric": {"sigma_S_m": -1}, "fluid": {"mu_Pa_s": -1}})j")),
            W("", "/materials/m/electric/sigma_S_m", "sigma_S_m (electric conductivity) must be positive"));
}

void check_blocks_without_their_main_key_are_incomplete() {
  // AUTHOR TEXT: a block needs its main key, as the thermal block needs k_W_mK
  struct Case {
    const char* block;
    const char* key;
    const char* unit;
    const char* material;
  };
  const Case cases[] = {{"thermal", "k_W_mK", "W/(m K)", R"j({"thermal": {"cp_J_kgK": 385.0}})j"},
                        {"fluid", "mu_Pa_s", "Pa s", R"j({"fluid": {"beta_1_K": 2e-4}})j"},
                        {"electrolyte", "kappa_S_m", "S/m", R"j({"electrolyte": {"notes": "n"}})j"},
                        {"electric", "sigma_S_m", "S/m", R"j({"electric": {}})j"}};
  for (const Case& c : cases) {
    check_one(std::string("a ") + c.block + " block without " + c.key, model_doc("m", J(c.material)),
              W("material.block_incomplete", std::string("/materials/m/") + c.block,
                std::string("the '") + c.block + "' block of material 'm' has no '" + c.key + "'")
                  .with_hint(std::string("A '") + c.block + "' block needs '" + c.key + "' (" + c.unit + "); add it.")
                  .with_details(json{{"material", "m"}, {"block", c.block}, {"key", c.key}}));
  }
}

void check_thermal_cp_without_k_is_refused() {
  // the electromagnetic parser's refusal is nlohmann's out_of_range (not an axmp::Error): the reference, so that the
  // model path's refusal is known to be its own
  {
    const Triple t = em_error(em_doc("m", J(R"j({"kind": "linear", "thermal": {}})j")));
    CHECK(t.threw && t.message.rfind("(bare) ", 0) == 0, "the electromagnetic twin throws a bare exception for a thermal block without k_W_mK: " + t.message);
  }
  const json doc = model_doc("copper_like", J(R"j({"thermal": {"cp_J_kgK": 385.0}})j"));
  const Parsed p = parse(doc);
  const auto errs = errors_of(p.f);
  CHECK(errs.size() == 1, "one error");
  if (errs.size() == 1) {
    CHECK(errs[0].code() == "material.block_incomplete" && errs[0].where() == "/materials/copper_like/thermal", "material.block_incomplete at the thermal block");
    CHECK(errs[0].message().find("k_W_mK") != std::string::npos, "the message names the missing key");
  }
  try {
    axmp::spec::parse_model(doc);
    fail("parse_model did not throw");
  } catch (const Error& e) {
    CHECK(e.code() == "material.block_incomplete", "parse_model throws an axmp::Error with the code");
  } catch (const std::exception& e) {
    fail(std::string("parse_model threw a bare exception: ") + e.what());
  }
  // with k_W_mK, with the heat capacity or without it, the block is complete; k as a law counts as given
  clean("thermal with k and cp", model_doc("m", J(R"j({"thermal": {"k_W_mK": 400.0, "cp_J_kgK": 385.0}})j")));
  clean("thermal with k only", model_doc("m", J(R"j({"thermal": {"k_W_mK": 400.0}})j")));
  clean("thermal with k as a law", model_doc("m", J(R"j({"thermal": {"k_W_mK": {"law": "expression", "of": ["T"], "expr": "400 - 0.05*T", "range": {"T": [250, 500]}}, "cp_J_kgK": 385.0}})j")));
}

// a law in mechanical.E_Pa or nu is read by the model path, which also checks the block's own requirements
void check_mechanical_laws() {
  auto JS = [](const std::string& t) { return J(t.c_str()); };
  const std::string law = R"j({"law": "expression", "of": ["T"], "expr": "2e11 - 1e7*T", "range": {"T": [250, 500]}})j";
  {
    const Parsed p = parse(model_doc("m", JS(std::string(R"j({"mechanical": {"E_Pa": )j") + law + R"j(, "nu": 0.3, "alpha": 1e-3}})j")));
    CHECK(!p.f.has_errors(), "E_Pa as a law");
    if (const M::MaterialProps* m = material_of(p, "m")) {
      CHECK(m->E_Pa.given && m->E_Pa.is_law() && m->nu.given && !m->nu.is_law() && close(m->nu.value, 0.3), "E_Pa is a law, nu a number");
      CHECK(close(m->em.mech.alpha, 1e-3), "alpha is kept");
    }
  }
  {
    const Parsed p = parse(model_doc("m", JS(std::string(R"j({"mechanical": {"E_Pa": 2e11, "nu": )j") + law + R"j(}})j")));
    CHECK(!p.f.has_errors(), "nu as a law");
  }
  // a missing key beside a law has the text that the electromagnetic parser gives the same block without the law
  same_as_twin("a mechanical block with a law and no nu", "m", JS(std::string(R"j({"mechanical": {"E_Pa": )j") + law + R"j(}})j"),
               J(R"j({"mechanical": {"E_Pa": 2e11}})j"));
  same_as_twin("a mechanical block with a law and no E_Pa", "m", JS(std::string(R"j({"mechanical": {"nu": )j") + law + R"j(}})j"),
               J(R"j({"mechanical": {"nu": 0.3}})j"));
  check_one("a mechanical block with a law and no nu: the pointer and no code", model_doc("m", JS(std::string(R"j({"mechanical": {"E_Pa": )j") + law + R"j(}})j")),
            W("", "/materials/m/mechanical", "missing required field 'nu'"));
  check_one("a mechanical block with a law and both expansion spellings",
            model_doc("m", JS(std::string(R"j({"mechanical": {"E_Pa": )j") + law + R"j(, "nu": 0.3, "alpha": 1e-3, "alpha_1_K": 1e-5}})j")),
            W("material.expansion_ambiguous", "/materials/m/mechanical"));
  check_one("an out-of-range nu beside a law", model_doc("m", JS(std::string(R"j({"mechanical": {"E_Pa": )j") + law + R"j(, "nu": 0.7}})j")),
            W("", "/materials/m/mechanical/nu", "Poisson's ratio nu = 0.7 must satisfy -1 < nu < 0.5"));
  // the range of Poisson's ratio applies to each value of a table law as it does to a number
  {
    const auto nu_table = [&](const char* y) {
      return model_doc("m", JS(std::string(R"j({"mechanical": {"E_Pa": )j") + law + R"j(, "nu": {"law": "table", "of": ["T"], "x": [300, 400], "y": )j" + y +
                                R"j(, "range": {"T": [300, 400]}}}})j"));
    };
    {
      const Parsed p = parse(nu_table("[0.3, 0.49]"));
      CHECK(!p.f.has_errors(), "a table of nu inside (-1, 0.5)");
    }
    check_one("a table of nu above 0.5", nu_table("[0.3, 0.7]"),
              W("law.table_invalid", "/materials/m/mechanical/nu/y/1", "Poisson's ratio must satisfy -1 < nu < 0.5; the table gives y = 0.7 at index 1"));
    check_one("a table of nu below -1", nu_table("[-2, 0.3]"),
              W("law.table_invalid", "/materials/m/mechanical/nu/y/0", "Poisson's ratio must satisfy -1 < nu < 0.5; the table gives y = -2 at index 0"));
    check_one("a table of nu at 0.5", nu_table("[0.3, 0.5]"),
              W("law.table_invalid", "/materials/m/mechanical/nu/y/1").with_hint("nu = 0.5 is incompressible and makes the displacement formulation singular; -1 < nu < 0.5."));
    check_one("a table of nu at -1", nu_table("[-1, 0.3]"), W("law.table_invalid", "/materials/m/mechanical/nu/y/0"));
  }
  // a law whose range lists a field that it does not read
  check_one("a law range with a field outside 'of'",
            model_doc("m", J(R"j({"mechanical": {"E_Pa": {"law": "expression", "of": ["T"], "expr": "2e11", "range": {"T": [1, 2], "p": [0, 1]}}, "nu": 0.3}})j")),
            W("", "/materials/m/mechanical/E_Pa/range/p", "the range of this law lists 'p', which its 'of' does not"));
  // thermal.notes is checked when the conductivity is a law
  check_one("thermal notes that are not a string beside a conductivity law",
            model_doc("m", JS(std::string(R"j({"thermal": {"k_W_mK": )j") + law + R"j(, "notes": 5}})j")), W("", "/materials/m/thermal/notes", "expected a string"));
}

void check_kind_absent_is_vacuum() {
  {
    const Parsed p = parse(model_doc("m", J(R"j({"density": 998.2})j")));
    CHECK(!p.f.has_errors(), "a material without kind");
    if (const M::MaterialProps* m = material_of(p, "m")) CHECK(m->em.kind == axmp::spec::MaterialKind::Vacuum && close(m->em.density, 998.2), "the kind is vacuum");
  }
  {
    const Parsed p = parse(model_doc("m", J(R"j({})j")));
    CHECK(!p.f.has_errors() && material_of(p, "m") && material_of(p, "m")->em.kind == axmp::spec::MaterialKind::Vacuum, "an empty material is vacuum");
  }
  // a kind that is written is read as the electromagnetic parser reads it, and refused as it refuses
  same_as_twin("an unknown kind", "m", J(R"j({"kind": "plasma"})j"));
  same_as_twin("a number as a kind", "m", J(R"j({"kind": 3})j"));
  {
    const Parsed p = parse(model_doc("m", J(R"j({"kind": "linear", "mu_r": 4})j")));
    CHECK(material_of(p, "m") && material_of(p, "m")->em.kind == axmp::spec::MaterialKind::Linear && close(material_of(p, "m")->em.mu_r, 4.0), "a written kind");
  }
  // the electromagnetic keys mean what they mean there: the twin's Material and the model's agree
  const json node = J(R"j({"kind": "linear", "mu_r": 2.5, "eps_r": 3.0, "tan_delta": 0.01, "density": 1000.0, "cost_per_kg": 5.0, "warm": true,
      "thermal": {"k_W_mK": [[100.0, 1.0], [300.0, 2.0]]}, "mechanical": {"E_Pa": 2e11, "nu": 0.3, "alpha": 1e-3},
      "rf": {"model": "skin", "T_K": 4, "rho_ohm_m": 1.7e-8}})j");
  axmp::spec::Design twin = axmp::spec::parse_design(em_doc("m", node));
  const Parsed p = parse(model_doc("m", node));
  const M::MaterialProps* m = material_of(p, "m");
  const axmp::spec::Material& e = twin.materials.at("m");
  if (m && !p.f.has_errors()) {
    CHECK(m->em.kind == e.kind && m->em.mu_r == e.mu_r && m->em.eps_r == e.eps_r && m->em.tan_delta == e.tan_delta && m->em.density == e.density &&
              m->em.cost_per_kg == e.cost_per_kg && m->em.warm == e.warm,
          "the electromagnetic scalars equal the twin's");
    CHECK(m->em.mech.given == e.mech.given && m->em.mech.E_Pa == e.mech.E_Pa && m->em.mech.nu == e.mech.nu && m->em.mech.alpha == e.mech.alpha, "mechanical equals the twin's");
    CHECK(m->em.therm.given == e.therm.given && m->em.therm.k_const == e.therm.k_const && m->em.therm.kT_T == e.therm.kT_T && m->em.therm.kT_k == e.therm.kT_k, "thermal equals the twin's");
    CHECK(m->em.rf.model == e.rf.model, "rf equals the twin's");
  } else {
    fail("the material of the electromagnetic twin was not parsed by the model path");
  }
  // E_Pa and nu: the electromagnetic text
  same_as_twin("E_Pa that is not positive", "m", J(R"j({"kind": "linear", "mechanical": {"E_Pa": -1.0, "nu": 0.3}})j"));
  same_as_twin("nu out of range", "m", J(R"j({"kind": "linear", "mechanical": {"E_Pa": 1e9, "nu": 0.5}})j"));
  same_as_twin("mu_r below one", "m", J(R"j({"mu_r": 0.5})j"));
}

void check_E_Pa_nonpositive_gives_the_em_text() {
  // with alpha_1_K beside it, so that the filtered copy is not the whole block
  const json node = J(R"j({"mechanical": {"E_Pa": 0.0, "nu": 0.3, "alpha_1_K": 1e-5}})j");
  const json em_node = J(R"j({"kind": "vacuum", "mechanical": {"E_Pa": 0.0, "nu": 0.3}})j");
  const Triple t = em_error(em_doc("m", em_node));
  CHECK(t.threw && t.where == "/materials/m/mechanical/E_Pa" && t.message == "E_Pa (Young's modulus) must be positive", "the reference: " + t.message);
  const auto errs = errors_of(parse(model_doc("m", node)).f);
  CHECK(errs.size() == 1, "one error");
  if (errs.size() == 1) {
    CHECK(errs[0].message() == t.message && errs[0].where() == t.where && errs[0].hint() == t.hint, "message(), where() and hint() equal those parse_design raises for the twin");
    CHECK(errs[0].code().empty(), "and it has no code, as the twin has none");
  }
}

void check_alpha_1_K_needs_E_and_nu() {
  // the existing "missing required field" text, through the filtered copy
  same_as_twin("alpha_1_K without E_Pa", "m", J(R"j({"mechanical": {"nu": 0.3, "alpha_1_K": 1e-5}})j"), J(R"j({"mechanical": {"nu": 0.3}})j"));
  const auto e1 = errors_of(parse(model_doc("m", J(R"j({"mechanical": {"alpha_1_K": 1e-5}})j"))).f);
  CHECK(e1.size() == 1 && e1[0].message() == "missing required field 'E_Pa'" && e1[0].where() == "/materials/m/mechanical" && e1[0].code().empty(),
        "alpha_1_K alone: 'E_Pa' is missing, at the mechanical block");
  const auto e2 = errors_of(parse(model_doc("m", J(R"j({"mechanical": {"E_Pa": 1e9, "alpha_1_K": 1e-5}})j"))).f);
  CHECK(e2.size() == 1 && e2[0].message() == "missing required field 'nu'" && e2[0].where() == "/materials/m/mechanical", "alpha_1_K with E_Pa only: 'nu' is missing");
  const Parsed ok = parse(model_doc("m", J(R"j({"mechanical": {"E_Pa": 1e9, "nu": 0.3, "alpha_1_K": 1.7e-5}})j")));
  CHECK(!ok.f.has_errors() && material_of(ok, "m") && close(material_of(ok, "m")->alpha_1_K.value, 1.7e-5) && material_of(ok, "m")->alpha_1_K.given, "with E_Pa and nu it is read");
}

void check_expansion_ambiguous() {
  // the note, section 6.2: the whole sentence is the message
  const std::string text =
      "`alpha` is the total contraction strain used by `mechanics.cooldown`; `alpha_1_K` is the expansion coefficient per kelvin used by the "
      "`thermal_expansion` coupling. Keep the one this document uses.";
  check_one("alpha and alpha_1_K", model_doc("m", J(R"j({"mechanical": {"E_Pa": 1e9, "nu": 0.3, "alpha": 1e-3, "alpha_1_K": 1e-5}})j")),
            W("material.expansion_ambiguous", "/materials/m/mechanical", text));
  clean("alpha alone", model_doc("m", J(R"j({"mechanical": {"E_Pa": 1e9, "nu": 0.3, "alpha": 1e-3}})j")));
  clean("alpha_1_K alone", model_doc("m", J(R"j({"mechanical": {"E_Pa": 1e9, "nu": 0.3, "alpha_1_K": 1e-5}})j")));
  // alpha_1_K as a law is also the per-kelvin meaning
  check_one("alpha and a law for alpha_1_K", model_doc("m", J(R"j({"mechanical": {"E_Pa": 1e9, "nu": 0.3, "alpha": 1e-3,
      "alpha_1_K": {"law": "expression", "of": ["T"], "expr": "1e-5 + 1e-8*T", "range": {"T": [200, 400]}}}})j")),
            W("material.expansion_ambiguous", "/materials/m/mechanical", text));
}

void check_conductivity_ambiguous() {
  // sigma_S_m at the material level (a full-wave medium) and electric.sigma_S_m: equal constants are accepted
  clean("equal constants", model_doc("m", J(R"j({"kind": "linear", "sigma_S_m": 1e6, "electric": {"sigma_S_m": 1e6}})j")));
  clean("equal after evaluation", model_doc("m", J(R"j({"kind": "linear", "sigma_S_m": "s0", "electric": {"sigma_S_m": "2*s0/2"}})j"), J(R"j({"s0": 1e6})j")));
  clean("one of them", model_doc("m", J(R"j({"kind": "linear", "sigma_S_m": 1e6})j")));
  const std::string msg = "material 'm' gives sigma_S_m both at the material level and in its 'electric' block, with different values";  // AUTHOR TEXT
  check_one("different values", model_doc("m", J(R"j({"kind": "linear", "sigma_S_m": 2e6, "electric": {"sigma_S_m": 1e6}})j")),
            W("material.conductivity_ambiguous", "/materials/m/electric/sigma_S_m", msg)
                .with_hint("Keep one of them, or make them equal: the material-level key is the conductivity of a full-wave medium and 'electric.sigma_S_m' that "
                           "of 'current.conduction'.")
                .with_details(J(R"j({"material": "m", "material_level": 2000000.0, "electric": 1000000.0})j")));
  check_one("a law and a constant", model_doc("m", J(R"j({"kind": "linear", "sigma_S_m": 1e6, "electric": {"sigma_S_m":
      {"law": "expression", "of": ["T"], "expr": "1e6", "range": {"T": [250, 500]}}}})j")),
            W("material.conductivity_ambiguous", "/materials/m/electric/sigma_S_m", msg));
}

void check_law_grammar() {
  const std::string P = "/materials/m/electric/sigma_S_m";
  auto with_law = [](const json& law) {
    json m = json::object();
    m["electric"] = json::object();
    m["electric"]["sigma_S_m"] = law;
    return model_doc("m", m, J(R"j({"T0": 293.15})j"));
  };
  const json expr_law = J(R"j({"law": "expression", "of": ["T"], "expr": "5.998e7/(1 + 3.9e-3*(T - T0))", "range": {"T": [250.0, 500.0]},
                               "source": "linear resistivity model"})j");

  // ---- an expression law compiles with its `of` fields as variables, and the parameters as constants
  {
    const Parsed p = parse(with_law(expr_law));
    CHECK(!p.f.has_errors(), "an expression law");
    if (const M::MaterialProps* m = material_of(p, "m")) {
      CHECK(m->electric_sigma_S_m.given && m->electric_sigma_S_m.is_law(), "the property is a law");
      if (m->electric_sigma_S_m.is_law()) {
        const M::Law& l = *m->electric_sigma_S_m.law;
        CHECK(l.kind == "expression" && l.of == Strs({"T"}) && l.expr == "5.998e7/(1 + 3.9e-3*(T - T0))" && l.range.at("T") == std::make_pair(250.0, 500.0) &&
                  l.out_of_range == "error" && !l.allow_c0 && l.source == "linear resistivity model",
              "kind, of, expr, range, the default out_of_range, allow_c0 and source");
      }
    }
    json nosrc = expr_law;
    nosrc.erase("source");
    const Parsed q = parse(with_law(nosrc));
    CHECK(!q.f.has_errors() && material_of(q, "m")->electric_sigma_S_m.law->source.empty(), "the source is optional at this stage");
    json two = expr_law;
    two["of"] = J(R"j(["T", "p"])j");
    two["range"] = J(R"j({"T": [250.0, 500.0], "p": [1e5, 2e5]})j");
    two["expr"] = "1e6*(1 + 1e-9*p)/(1 + 3.9e-3*(T - T0))";
    clean("a law of two fields", with_law(two));
    json sp = expr_law;
    sp["of"] = J(R"j(["c_Cu"])j");
    sp["range"] = J(R"j({"c_Cu": [0, 1]})j");
    sp["expr"] = "1e6*c_Cu";
    clean("a law of a species concentration", with_law(sp));
    json cl = expr_law;
    cl["out_of_range"] = "clamp";
    cl["allow_c0"] = true;
    const Parsed r = parse(with_law(cl));
    CHECK(!r.f.has_errors() && material_of(r, "m")->electric_sigma_S_m.law->out_of_range == "clamp" && material_of(r, "m")->electric_sigma_S_m.law->allow_c0, "out_of_range clamp and allow_c0");
  }
  // ---- the expression's own refusals, with the context (the deferrals of the field-expression tests)
  {
    json bad = expr_law;
    bad["expr"] = "5.998e7/(1 + 3.9e-3*(Q - T0))";
    check_one("an unknown name in a law", with_law(bad),
              W("expr.unknown_variable", P + "/expr").with_message_has("unknown name 'Q'").with_details(J(R"j({"name": "Q", "variables": ["T"]})j")));
    json shadow = model_doc("m", J(R"j({"electric": {"sigma_S_m": {"law": "expression", "of": ["T"], "expr": "1e6*T", "range": {"T": [250, 500]}}}})j"), J(R"j({"T": 300.0})j"));
    check_one("a parameter with the name of a field that the law reads", shadow,
              W("expr.parameter_shadows_variable", P + "/expr", "parameter 'T' has the name of the field 'T' that this law reads; rename the parameter")
                  .with_details(J(R"j({"parameter": "T", "variables": ["T"]})j")));
  }
  // ---- the range is required (law.range_missing), at the pointer where it belongs
  {
    json nr = expr_law;
    nr.erase("range");
    check_one("a law without a range", with_law(nr),
              W("law.range_missing", P + "/range", "the law of 'sigma_S_m' needs a validity range for 'T'")  // AUTHOR TEXT
                  .with_hint("Add \"range\": {\"T\": [T_min, T_max]} with the interval over which the law is valid.")
                  .with_details(J(R"j({"property": "sigma_S_m", "field": "T"})j")));
    json half = expr_law;
    half["of"] = J(R"j(["T", "p"])j");
    half["expr"] = "1e6*(1 + 1e-9*p)";
    check_one("a range that lacks a field", with_law(half), W("law.range_missing", P + "/range", "the law of 'sigma_S_m' needs a validity range for 'p'").with_details(J(R"j({"field": "p"})j")));
    json inverted = expr_law;
    inverted["range"]["T"] = J("[500.0, 250.0]");
    check_one("a range with min above max", with_law(inverted), W("", P + "/range/T", "range must satisfy min < max (got 500, 250)"));
  }
  // ---- the other refusals of the grammar (AUTHOR TEXT)
  {
    json k = expr_law;
    k["law"] = "poly";
    check_one("an unknown law", with_law(k), W("law.unknown_kind", P + "/law", "unknown law 'poly'").with_hint("Use \"expression\" or \"table\"."));
    json f = expr_law;
    f["of"] = J(R"j(["Q"])j");
    f["range"] = J(R"j({"Q": [0, 1]})j");
    f["expr"] = "1e6";
    check_one("a field that no physics publishes", with_law(f),
              W("law.unknown_field", P + "/of/0", "'Q' is not a field that a law can read")
                  .with_hint("The fields are T, p, B_mag and c_<species>.")
                  .with_details(J(R"j({"field": "Q", "valid": ["T", "p", "B_mag", "c_<species>"]})j")));
    json oor = expr_law;
    oor["out_of_range"] = "extrapolate";
    check_one("an unknown out_of_range", with_law(oor), W("law.unknown_option", P + "/out_of_range", "unknown out_of_range 'extrapolate'").with_hint("Use \"error\" or \"clamp\"."));
    json noof = expr_law;
    noof.erase("of");
    check_one("a law without 'of'", with_law(noof), W("", P, "missing required field 'of'"));
    json noexpr = expr_law;
    noexpr.erase("expr");
    check_one("an expression law without 'expr'", with_law(noexpr), W("", P, "missing required field 'expr'"));
    json extra = expr_law;
    extra["x"] = J("[1, 2]");
    check_one("a table key in an expression law", with_law(extra),
              W("", P, "unknown field 'x'").with_hint(only_list({"law", "of", "expr", "range", "out_of_range", "source", "allow_c0"})));
  }
  // ---- a table law (the example of section 6.3)
  const json table = J(R"j({"law": "table", "of": ["T"], "x": [280, 300, 320], "y": [1.43e-3, 8.54e-4, 5.77e-4],
                            "interpolation": "monotone_cubic", "range": {"T": [273.16, 373.15]}, "source": "CRC Handbook, water at 1 atm"})j");
  auto with_table = [](const json& law) {
    json m = json::object();
    m["fluid"] = json::object();
    m["fluid"]["mu_Pa_s"] = law;
    return model_doc("water", m);
  };
  {
    const Parsed p = parse(with_table(table));
    CHECK(!p.f.has_errors(), "a table law");
    if (const M::MaterialProps* m = material_of(p, "water")) {
      CHECK(m->has_fluid && m->mu_Pa_s.is_law(), "the viscosity is a law");
      if (m->mu_Pa_s.is_law()) {
        const M::Law& l = *m->mu_Pa_s.law;
        CHECK(l.kind == "table" && l.of == Strs({"T"}) && l.x == std::vector<double>({280, 300, 320}) && l.y == std::vector<double>({1.43e-3, 8.54e-4, 5.77e-4}) &&
                  l.interpolation == "monotone_cubic" && l.range.at("T") == std::make_pair(273.16, 373.15) && l.out_of_range == "error" && l.source == "CRC Handbook, water at 1 atm",
              "kind, of, x, y, interpolation, range and source");
      }
    }
    json dflt = table;
    dflt.erase("interpolation");
    const Parsed q = parse(with_table(dflt));
    CHECK(!q.f.has_errors() && material_of(q, "water")->mu_Pa_s.law->interpolation == "monotone_cubic", "monotone_cubic is the default");
    json lin = table;
    lin["interpolation"] = "linear";
    lin["allow_c0"] = true;
    const Parsed r = parse(with_table(lin));
    CHECK(!r.f.has_errors() && material_of(r, "water")->mu_Pa_s.law->interpolation == "linear" && material_of(r, "water")->mu_Pa_s.law->allow_c0, "linear, allow_c0");
  }
  {
    json d = table;
    d["x"] = J("[280, 320, 300]");
    check_one("a table whose x does not increase", with_table(d), W("law.table_invalid", "/materials/water/fluid/mu_Pa_s/x", "the 'x' of a table law must be strictly increasing"));  // AUTHOR TEXT
    d = table;
    d["x"] = J("[280, 300, 300]");
    check_one("a table with a repeated x", with_table(d), W("law.table_invalid", "/materials/water/fluid/mu_Pa_s/x"));
    d = table;
    d["y"] = J("[1.43e-3, 8.54e-4]");
    check_one("a table whose lengths differ", with_table(d), W("law.table_invalid", "/materials/water/fluid/mu_Pa_s/y", "the table law has 3 values of 'x' and 2 of 'y'"));
    d = table;
    d["x"] = J("[280]");
    d["y"] = J("[1.43e-3]");
    check_one("a table of one point", with_table(d), W("law.table_invalid", "/materials/water/fluid/mu_Pa_s/x", "a table law needs at least two points"));
    d = table;
    d["of"] = J(R"j(["T", "p"])j");
    d["range"] = J(R"j({"T": [273.16, 373.15], "p": [1e5, 2e5]})j");
    check_one("a table that reads two fields", with_table(d), W("law.table_invalid", "/materials/water/fluid/mu_Pa_s/of", "a table law reads one field, and 'of' lists 2"));
    d = table;
    d["interpolation"] = "cubic";
    check_one("an unknown interpolation", with_table(d),
              W("law.unknown_option", "/materials/water/fluid/mu_Pa_s/interpolation", "unknown interpolation 'cubic'").with_hint("Use \"monotone_cubic\" or \"linear\"."));
    d = table;
    d.erase("range");
    check_one("a table without a range", with_table(d), W("law.range_missing", "/materials/water/fluid/mu_Pa_s/range", "the law of 'mu_Pa_s' needs a validity range for 'T'"));
    d = table;
    d.erase("x");
    check_one("a table without x", with_table(d), W("", "/materials/water/fluid/mu_Pa_s", "missing required field 'x'"));
    d = table;
    d["expr"] = "1";
    check_one("an expression key in a table law", with_table(d),
              W("", "/materials/water/fluid/mu_Pa_s", "unknown field 'expr'").with_hint(only_list({"law", "of", "x", "y", "interpolation", "range", "out_of_range", "source", "allow_c0"})));
  }
  // ---- every property of the blocks may be a law
  {
    const json l = J(R"j({"law": "expression", "of": ["T"], "expr": "1 + 0*T", "range": {"T": [200, 400]}})j");
    auto law_at = [&](const char* block, const char* key, const char* extra) {
      json m = J(extra);
      m[block][key] = l;
      return model_doc("m", m);
    };
    struct Prop {
      const char* block;
      const char* key;
      const char* extra;
      std::function<const M::Property&(const M::MaterialProps&)> get;
    };
    const std::vector<Prop> props = {
        {"fluid", "mu_Pa_s", "{\"fluid\": {}}", [](const M::MaterialProps& m) -> const M::Property& { return m.mu_Pa_s; }},
        {"fluid", "beta_1_K", "{\"fluid\": {\"mu_Pa_s\": 1e-3}}", [](const M::MaterialProps& m) -> const M::Property& { return m.beta_1_K; }},
        {"electrolyte", "kappa_S_m", "{\"electrolyte\": {}}", [](const M::MaterialProps& m) -> const M::Property& { return m.kappa_S_m; }},
        {"electric", "sigma_S_m", "{\"electric\": {}}", [](const M::MaterialProps& m) -> const M::Property& { return m.electric_sigma_S_m; }},
        {"thermal", "k_W_mK", "{\"thermal\": {}}", [](const M::MaterialProps& m) -> const M::Property& { return m.k_W_mK; }},
        {"thermal", "cp_J_kgK", "{\"thermal\": {\"k_W_mK\": 1.0}}", [](const M::MaterialProps& m) -> const M::Property& { return m.cp_J_kgK; }},
        {"mechanical", "alpha_1_K", "{\"mechanical\": {\"E_Pa\": 1e9, \"nu\": 0.3}}", [](const M::MaterialProps& m) -> const M::Property& { return m.alpha_1_K; }},
    };
    for (const Prop& pr : props) {
      const json d = law_at(pr.block, pr.key, pr.extra);
      const Parsed p = parse(d);
      for (const auto& e : errors_of(p.f)) fail(std::string("a law for ") + pr.block + "." + pr.key + ": [" + e.code() + "] " + e.message() + " [at " + e.where() + "]");
      if (const M::MaterialProps* m = material_of(p, "m")) CHECK(pr.get(*m).is_law() && pr.get(*m).given, std::string("the law of ") + pr.block + "." + pr.key + " is recorded");
    }
    // a law in mechanical.E_Pa or nu is a property of the model: check_mechanical_laws
  }
  // ---- the density is a property of section 6.1's table too: a number, an expression or a law
  {
    const json dl = J(R"j({"law": "expression", "of": ["T"], "expr": "1000 - 0.1*T", "range": {"T": [250, 500]}})j");
    // "iron" is the kind whose density has a default that is not zero (the twin without a density shows it)
    for (const char* kind : {"", "linear", "iron"}) {
      json m = json{{"density", dl}};
      if (*kind) m["kind"] = kind;
      json twin = m;
      twin.erase("density");
      if (std::string(kind) == "iron") {
        twin["kind"] = "iron";
        CHECK(axmp::spec::parse_design(em_doc("m", twin)).materials.at("m").density > 0.0, "the electromagnetic parser gives iron a default density");
      } else {
        CHECK(!em_error(em_doc("m", twin)).threw, std::string("the twin without a density is valid, kind '") + kind + "'");
      }
      const Parsed p = parse(model_doc("m", m));
      for (const auto& e : errors_of(p.f)) fail(std::string("a density law, kind '") + kind + "': [" + e.code() + "] " + e.message() + " [at " + e.where() + "]");
      if (const M::MaterialProps* mp = material_of(p, "m")) {
        CHECK(mp->density.given && mp->density.is_law() && mp->density.law->expr == "1000 - 0.1*T", "the density is the law");
        CHECK(mp->em.density == 0.0, "the electromagnetic density is not given beside a law");
      }
    }
    // a number or an expression is read by the electromagnetic parser and mirrored; an absent density is not given
    const Parsed n = parse(model_doc("m", J(R"j({"density": "2*rho_half"})j"), J(R"j({"rho_half": 499.1})j")));
    CHECK(!n.f.has_errors(), "a density expression");
    if (const M::MaterialProps* mp = material_of(n, "m"))
      CHECK(mp->density.given && !mp->density.is_law() && close(mp->density.value, 998.2) && close(mp->em.density, 998.2), "2 * 499.1 = 998.2, in both places");
    const Parsed a = parse(model_doc("m", J(R"j({})j")));
    if (const M::MaterialProps* mp = material_of(a, "m")) CHECK(!mp->density.given && !mp->density.is_law(), "no density: not given");
    // the errors of a density law carry the user's pointer
    json nr = dl;
    nr.erase("range");
    check_one("a density law without a range", model_doc("m", json{{"density", nr}}),
              W("law.range_missing", "/materials/m/density/range", "the law of 'density' needs a validity range for 'T'").with_details(J(R"j({"property": "density", "field": "T"})j")));
    // a density that is neither is refused as the electromagnetic parser refuses it
    same_as_twin("a density that is a list", "m", J(R"j({"density": [1000]})j"));
  }
  // ---- a field listed twice in 'of' is refused where it is listed (AUTHOR TEXT)
  {
    json dup = expr_law;
    dup["of"] = J(R"j(["T", "T"])j");
    check_one("a field listed twice", with_law(dup), W("", P + "/of/1", "'of' lists the field 'T' twice").with_hint("Name each field that the law reads once."));
  }
  // ---- a law written for k_W_mK removes the block from the electromagnetic copy whole: its electromagnetic form is "not given"
  {
    const json m = J(R"j({"thermal": {"k_W_mK": {"law": "expression", "of": ["T"], "expr": "400 - 0.05*T", "range": {"T": [250, 500]}}, "cp_J_kgK": 385.0}})j");
    const Parsed p = parse(model_doc("m", m));
    CHECK(!p.f.has_errors(), "a thermal block with k as a law and cp");
    if (const M::MaterialProps* mp = material_of(p, "m")) {
      CHECK(!mp->em.therm.given, "the electromagnetic thermal part is not given");
      CHECK(mp->k_W_mK.is_law() && mp->k_W_mK.law->kind == "expression" && close(mp->cp_J_kgK.value, 385.0), "k is the law, cp the number");
    }
  }
  // ---- the [[T, k]] spelling of thermal.k_W_mK is the equivalent linear, clamped law
  {
    const json m = J(R"j({"thermal": {"k_W_mK": [[273.0, 0.5], [323.0, 0.6], [373.0, 0.7]]}})j");
    const Parsed p = parse(model_doc("m", m));
    CHECK(!p.f.has_errors(), "the [[T, k]] table");
    if (const M::MaterialProps* mp = material_of(p, "m")) {
      CHECK(mp->em.therm.given && mp->em.therm.kT_T == std::vector<double>({273.0, 323.0, 373.0}) && mp->em.therm.kT_k == std::vector<double>({0.5, 0.6, 0.7}), "the electromagnetic form is read as before");
      if (mp->k_W_mK.is_law()) {
        const M::Law& l = *mp->k_W_mK.law;
        CHECK(l.kind == "table" && l.of == Strs({"T"}) && l.x == std::vector<double>({273.0, 323.0, 373.0}) && l.y == std::vector<double>({0.5, 0.6, 0.7}) && l.interpolation == "linear" &&
                  l.out_of_range == "clamp" && l.range.at("T") == std::make_pair(273.0, 373.0),
              "the equivalent law: a linear table, clamped, over the tabulated range");
      } else {
        fail("k_W_mK as a [[T, k]] table is not a law");
      }
    }
    same_as_twin("a k table that does not increase", "m", J(R"j({"thermal": {"k_W_mK": [[300.0, 0.5], [273.0, 0.6]]}})j"));
    same_as_twin("a k table of one pair", "m", J(R"j({"thermal": {"k_W_mK": [[300.0, 0.5]]}})j"));
  }
}

void check_filtered_copy_keeps_pointers() {
  // an error inside rf of a model document has the pointer it has in an electromagnetic one, beside new blocks
  same_as_twin("rf field of another model", "m", J(R"j({"kind": "linear", "rf": {"model": "skin", "Rs_ohm": 1.0}})j"));
  same_as_twin("rf field beside new blocks", "m",
               J(R"j({"kind": "linear", "fluid": {"mu_Pa_s": 1e-3}, "thermal": {"k_W_mK": 1.0, "cp_J_kgK": 2.0}, "rf": {"model": "skin", "Rs_ohm": 1.0}})j"),
               J(R"j({"kind": "linear", "thermal": {"k_W_mK": 1.0}, "rf": {"model": "skin", "Rs_ohm": 1.0}})j"));
  same_as_twin("an unknown rf model", "m", J(R"j({"kind": "linear", "electric": {"sigma_S_m": 1.0}, "rf": {"model": "magic"}})j"),
               J(R"j({"kind": "linear", "rf": {"model": "magic"}})j"));
  same_as_twin("E_Pa beside alpha_1_K", "m", J(R"j({"mechanical": {"E_Pa": -1.0, "nu": 0.3, "alpha_1_K": 1e-5}})j"),
               J(R"j({"mechanical": {"E_Pa": -1.0, "nu": 0.3}})j"));
  same_as_twin("a bad k beside cp", "m", J(R"j({"thermal": {"k_W_mK": -1.0, "cp_J_kgK": 2.0}})j"), J(R"j({"thermal": {"k_W_mK": -1.0}})j"));
  // the pointer of the model path's own checks is the same kind of pointer
  check_one("mu_r that is a string that does not evaluate", model_doc("m", J(R"j({"kind": "linear", "fluid": {"mu_Pa_s": 1e-3}, "mu_r": "nope"})j")),
            W("", "/materials/m/mu_r").with_message_has("unknown name 'nope'"));
}

void check_parse_model_material_directly() {
  const axmp::ParamMap params = {{"k0", 0.6}};
  const json good = J(R"j({"density": 998.2, "fluid": {"mu_Pa_s": 1.002e-3}, "thermal": {"k_W_mK": "k0", "cp_J_kgK": 4182.0}})j");
  const axmp::spec::detail::Node node{good, "/materials/water", &params};
  try {
    const M::MaterialProps m = axmp::spec::detail::parse_model_material("water", node);
    CHECK(m.em.name == "water" && m.em.kind == axmp::spec::MaterialKind::Vacuum && close(m.em.density, 998.2), "the electromagnetic part: name, kind vacuum, density");
    CHECK(m.has_fluid && close(m.mu_Pa_s.value, 1.002e-3) && close(m.k_W_mK.value, 0.6) && close(m.cp_J_kgK.value, 4182.0) && m.em.therm.given && close(m.em.therm.k_const, 0.6),
          "the new data and the filtered copy's");
  } catch (const std::exception& e) {
    fail(std::string("parse_model_material threw on a valid material: ") + e.what());
  }
  const json bad = J(R"j({"fluid": {"mu_Pa_s": -1}, "bogus": 1})j");
  const axmp::spec::detail::Node bad_node{bad, "/materials/water", &params};
  try {
    axmp::spec::detail::parse_model_material("water", bad_node);
    fail("parse_model_material did not throw on an invalid material");
  } catch (const Error& e) {
    // step 1, the key check, comes before the parsing of the new data
    CHECK(e.message() == "unknown field 'bogus'" && e.where() == "/materials/water", "the first error is the key check's, with the node's own path: " + e.message());
  } catch (const std::exception& e) {
    fail(std::string("parse_model_material threw a bare exception: ") + e.what());
  }
  const json bad2 = J(R"j({"fluid": {"mu_Pa_s": -1}})j");
  const axmp::spec::detail::Node bad2_node{bad2, "/some/where/water", &params};
  try {
    axmp::spec::detail::parse_model_material("water", bad2_node);
    fail("parse_model_material did not throw on a negative viscosity");
  } catch (const Error& e) {
    CHECK(e.where() == "/some/where/water/fluid/mu_Pa_s", "the pointers are the node's path, extended: " + e.where());
  }
}

// ---- the order of the four steps: the key check, then the new data, then the electromagnetic parser on the filtered copy
void check_the_four_steps_run_in_order() {
  // an unknown key and an error in the new data: the unknown key (step 1)
  check_one("step 1 before step 2", model_doc("m", J(R"j({"bogus": 1, "fluid": {"mu_Pa_s": -1}})j")), W("", "/materials/m", "unknown field 'bogus'").with_hint(only_list(kMaterialKeys)));
  // an error in the new data and an error that only the electromagnetic parser sees: the new data (step 2)
  check_one("step 2 before step 4", model_doc("m", J(R"j({"mu_r": 0.5, "fluid": {"mu_Pa_s": -1}})j")),
            W("", "/materials/m/fluid/mu_Pa_s", "mu_Pa_s (dynamic viscosity) must be positive"));
  // and when the new data are right, step 4 reports the electromagnetic error
  check_one("step 4", model_doc("m", J(R"j({"mu_r": 0.5, "fluid": {"mu_Pa_s": 1e-3}})j")), W("", "/materials/m/mu_r", "mu_r must be >= 1"));
}

// ---- positive tables, the names of fields, and the built-in materials
void check_positive_laws_and_field_names() {
  const std::string expr_law = R"j({"law": "expression", "of": ["T"], "expr": "2e11 - 1e7*T", "range": {"T": [250, 500]}})j";
  auto JS = [](const std::string& t) { return J(t.c_str()); };
  // E_Pa must be positive beside a law for nu: the text of the electromagnetic parser, at its pointer
  same_as_twin("E_Pa that is not positive beside a law for nu", "m", JS(std::string(R"j({"mechanical": {"E_Pa": -1, "nu": )j") + expr_law + "}}"),
               J(R"j({"mechanical": {"E_Pa": -1, "nu": 0.3}})j"));
  same_as_twin("E_Pa of zero beside a law for nu", "m", JS(std::string(R"j({"mechanical": {"E_Pa": 0, "nu": )j") + expr_law + "}}"),
               J(R"j({"mechanical": {"E_Pa": 0, "nu": 0.3}})j"));

  // a table law whose y is not positive, for each property that must be positive (AUTHOR TEXT: the property's own
  // positivity sentence, then the value and its index)
  struct Prop {
    const char* block;
    const char* key;
    const char* sentence;
    const char* extra;  // what the block needs besides the property
  };
  const Prop props[] = {
      {"fluid", "mu_Pa_s", "mu_Pa_s (dynamic viscosity) must be positive", ""},
      {"electrolyte", "kappa_S_m", "kappa_S_m (electrolyte conductivity) must be positive", ""},
      {"electric", "sigma_S_m", "sigma_S_m (electric conductivity) must be positive", ""},
      {"thermal", "k_W_mK", "k_W_mK must be positive", ""},
      {"thermal", "cp_J_kgK", "cp_J_kgK (specific heat) must be positive", R"j("k_W_mK": 1.0,)j"},
      {"mechanical", "E_Pa", "E_Pa (Young's modulus) must be positive", R"j("nu": 0.3,)j"},
  };
  for (const Prop& pr : props) {
    for (const char* bad : {"-1.0", "0.0"}) {
      const std::string table = std::string(R"j({"law": "table", "of": ["T"], "x": [280, 300, 320], "y": [1e-3, )j") + bad + R"j(, 5e-4], "range": {"T": [280, 320]}})j";
      const std::string block = std::string("{\"") + pr.block + "\": {" + pr.extra + "\"" + pr.key + "\": " + table + "}}";
      check_one(std::string("a table of ") + pr.key + " with y = " + bad, model_doc("m", JS(block)),
                W("law.table_invalid", std::string("/materials/m/") + pr.block + "/" + pr.key + "/y/1",
                  std::string(pr.sentence) + "; the table gives y = " + (std::string(bad) == "0.0" ? "0" : "-1") + " at index 1"));
    }
    const std::string good = std::string(R"j({"law": "table", "of": ["T"], "x": [280, 300, 320], "y": [1e-3, 8e-4, 5e-4], "range": {"T": [280, 320]}})j");
    clean(std::string("a positive table of ") + pr.key, model_doc("m", JS(std::string("{\"") + pr.block + "\": {" + pr.extra + "\"" + pr.key + "\": " + good + "}}")));
  }
  // a table of a property that has no sign rule is not refused for a negative y
  clean("a table of beta_1_K with a negative y",
        model_doc("m", J(R"j({"fluid": {"mu_Pa_s": 1e-3, "beta_1_K": {"law": "table", "of": ["T"], "x": [280, 300], "y": [-1e-4, 2e-4], "range": {"T": [280, 300]}}}})j")));

  // the field c_<name> names a species, and the name is an identifier
  const std::string base = R"j({"fluid": {"mu_Pa_s": {"law": "table", "of": ["FIELD"], "x": [0, 1], "y": [1e-3, 2e-3], "range": {"FIELD": [0, 1]}}}})j";
  auto with_field = [&](const std::string& f) {
    std::string t = base;
    for (size_t at; (at = t.find("FIELD")) != std::string::npos;) t.replace(at, 5, f);
    return model_doc("m", JS(t));
  };
  clean("a field c_Cu2", with_field("c_Cu2"));
  clean("a field c_cu_2", with_field("c_cu_2"));
  for (const char* f : {"c_a b", "c_a-b", "c_1.5", "c_Cu(2)"})
    check_one(std::string("the field '") + f + "' is not a name", with_field(f),
              W("law.unknown_field", "/materials/m/fluid/mu_Pa_s/of/0", std::string("'") + f + "' is not a field that a law can read"));
}

// ---- a built-in material and the document material with the same data have the same Properties
void check_builtin_materials_mirror_their_properties() {
  const Parsed none = parse(model_doc("m", J(R"j({})j")));
  const auto builtins = axmp::spec::builtin_materials();
  for (const auto& kv : builtins) {
    const auto it = none.m.materials.find(kv.first);
    if (it == none.m.materials.end()) {
      fail("the built-in material '" + kv.first + "' is not in the model");
      continue;
    }
    const M::MaterialProps& mp = it->second;
    const axmp::spec::Material& em = kv.second;
    const std::string who = "built-in '" + kv.first + "': ";
    CHECK(mp.density.given == (em.density != 0.0) && (!mp.density.given || close(mp.density.value, em.density)), who + "the density");
    CHECK(mp.E_Pa.given == em.mech.given && mp.nu.given == em.mech.given && (!em.mech.given || (close(mp.E_Pa.value, em.mech.E_Pa) && close(mp.nu.value, em.mech.nu))),
          who + "E_Pa and nu");
    CHECK(mp.k_W_mK.given == em.therm.given && (!em.therm.given || (!mp.k_W_mK.is_law() && close(mp.k_W_mK.value, em.therm.k_const))), who + "k_W_mK");
  }
  CHECK(none.m.materials.at("copper").E_Pa.given && none.m.materials.at("copper").k_W_mK.given, "copper has E_Pa and k_W_mK");
  CHECK(!none.m.materials.at("vacuum").E_Pa.given && !none.m.materials.at("vacuum").k_W_mK.given && !none.m.materials.at("vacuum").density.given, "vacuum has none");

  // the equivalent document material: same flags and values
  struct Twin {
    const char* builtin;
    const char* doc;
  };
  const Twin twins[] = {
      {"winding", R"j({"kind": "vacuum", "mechanical": {"E_Pa": 80.0e9, "nu": 0.30, "alpha": 3.0e-3}, "thermal": {"k_W_mK": 1.0}})j"},
      {"nb", R"j({"kind": "superconductor", "density": 8570.0, "mechanical": {"E_Pa": 105.0e9, "nu": 0.40, "alpha": 1.43e-3}, "thermal": {"k_W_mK": 5.0}})j"},
  };
  for (const Twin& t : twins) {
    const json d = model_doc("docm", J(t.doc));
    const Parsed p = parse(d);
    for (const auto& e : errors_of(p.f)) fail(std::string("the document material equivalent to '") + t.builtin + "': " + e.message());
    const M::MaterialProps* a = material_of(p, t.builtin);
    const M::MaterialProps* b = material_of(p, "docm");
    if (!a || !b) {
      fail(std::string("the twin of '") + t.builtin + "' is missing");
      continue;
    }
    const auto same = [&](const char* name, const M::Property& x, const M::Property& y) {
      CHECK(x.given == y.given && x.is_law() == y.is_law() && (!x.given || x.value == y.value), std::string(t.builtin) + ": the property " + name + " is the same for the built-in and the document material");
    };
    same("density", a->density, b->density);
    same("E_Pa", a->E_Pa, b->E_Pa);
    same("nu", a->nu, b->nu);
    same("k_W_mK", a->k_W_mK, b->k_W_mK);
    CHECK(a->E_Pa.given && a->k_W_mK.given, std::string(t.builtin) + ": the built-in mirrors E_Pa and k_W_mK");
  }
}

}  // namespace

int main() {
  check_unknown_key_at_each_level_lists_the_model_keys();
  check_error_inside_a_new_block_has_the_users_pointer();
  check_blocks_without_their_main_key_are_incomplete();
  check_thermal_cp_without_k_is_refused();
  check_kind_absent_is_vacuum();
  check_E_Pa_nonpositive_gives_the_em_text();
  check_alpha_1_K_needs_E_and_nu();
  check_expansion_ambiguous();
  check_conductivity_ambiguous();
  check_law_grammar();
  check_mechanical_laws();
  check_filtered_copy_keeps_pointers();
  check_parse_model_material_directly();
  check_the_four_steps_run_in_order();
  check_positive_laws_and_field_names();
  check_builtin_materials_mirror_their_properties();
  if (failures != 0) {
    std::printf("%d model-material check(s) failed\n", failures);
    return 1;
  }
  std::puts("model materials (the split, the model's key lists, the new blocks, the kind rule, the ambiguities, laws, pointers): passed");
  return 0;
}
