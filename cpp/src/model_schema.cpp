// add_model_branch (model_schema.hpp).
#include "axionmultiphysics/model_schema.hpp"

#include <set>
#include <string>
#include <vector>

#include "axionmultiphysics/physics_registry.hpp"

namespace axmp::model {
namespace {

using json = nlohmann::json;

const json kNum = {{"oneOf", {{{"type", "number"}}, {{"type", "string"}, {"description", "expression over parameters"}}}}};
const json kStr = {{"type", "string"}};

// An object with the given properties and no others.
json obj(json props, json required = json::array()) {
  json o = {{"type", "object"}, {"properties", std::move(props)}, {"additionalProperties", false}};
  if (!required.empty()) o["required"] = std::move(required);
  return o;
}

json datum_schema(const DataKey& k) {
  if (k.type == "boolean") return {{"type", "boolean"}};
  if (k.type == "names") return {{"type", "array"}, {"items", kStr}};
  if (k.type == "boundary") return kStr;
  if (k.type == "number") return kNum;
  // number_vector, vector, meridional_vector: one number or expression per component
  return {{"type", "array"}, {"items", kNum}, {"minItems", 1}};
}

// The keys a kind of condition, source or coupling term takes: for the base object (all keys of all kinds, loosely
// typed by the first kind that has them) and, per kind, what is required, forbidden and typed.
struct KeySet {
  std::vector<std::string> names;
  std::set<std::string> seen;
  void add(const std::string& n) {
    if (seen.insert(n).second) names.push_back(n);
  }
};

// One per-kind rule: `selector_key` equals `value` => the keys of `keys` are typed and required, any other key of
// `all` is forbidden.
json kind_rule(const std::string& selector_key, const std::string& value, const std::vector<DataKey>& keys,
               const KeySet& all, const std::set<std::string>& always) {
  json props = json::object();
  json required = json::array();
  json any_of = json::array();
  std::set<std::string> allowed = always;
  for (const DataKey& k : keys) {
    allowed.insert(k.name);
    props[k.name] = datum_schema(k);
    if (!k.per_depth_twin.empty()) {
      allowed.insert(k.per_depth_twin);
      props[k.per_depth_twin] = datum_schema(k);
    }
    if (!k.required) continue;
    if (k.per_depth_twin.empty()) required.push_back(k.name);
    else any_of.push_back(json{{"required", {k.name}}}), any_of.push_back(json{{"required", {k.per_depth_twin}}});
  }
  json forbidden = json::array();
  for (const std::string& n : all.names)
    if (!allowed.count(n)) forbidden.push_back(json{{"required", {n}}});
  json then = {{"properties", props}};
  if (!required.empty()) then["required"] = required;
  if (!any_of.empty()) then["allOf"] = json::array({json{{"anyOf", any_of}}});
  if (!forbidden.empty()) then["not"] = {{"anyOf", forbidden}};
  return {{"if", {{"properties", {{selector_key, {{"const", value}}}}}, {"required", {selector_key}}}}, {"then", then}};
}

// A condition or a source object of one physics type: the kinds offered, each with its keys.
json kinds_schema(const std::vector<ConditionKind>& kinds, const std::set<std::string>& always, bool is_source) {
  KeySet all;
  json base = json::object();
  json names = json::array();
  for (const ConditionKind& c : kinds) {
    names.push_back(c.name);
    for (const DataKey& k : c.data) {
      for (const std::string& n : {k.name, k.per_depth_twin}) {
        if (n.empty() || all.seen.count(n)) continue;
        all.add(n);
        base[n] = datum_schema(k);
      }
    }
  }
  for (const std::string& n : always) {
    all.add(n);
    base[n] = kStr;
  }
  base["kind"] = {{"enum", names}};
  json o = {{"type", "object"}, {"required", {"kind"}}, {"properties", base}, {"additionalProperties", false}};
  json rules = json::array();
  for (const ConditionKind& c : kinds) rules.push_back(kind_rule("kind", c.name, c.data, all, [&] {
    std::set<std::string> a = always;
    a.insert("kind");
    return a;
  }()));
  if (!rules.empty()) o["allOf"] = rules;
  (void)is_source;
  return o;
}

json option_schema(const OptionSpec& o) {
  if (o.type == "enum") return {{"enum", o.values}};
  if (o.type == "integer") {
    json values = json::array();
    for (const std::string& v : o.values) values.push_back(std::stoi(v));
    return {{"enum", values}};
  }
  return kNum;
}

json physics_schema(const Registry& reg) {
  json type_names = json::array();
  json options_all = json::object();
  for (const PhysicsType& t : reg.types) {
    type_names.push_back(t.name);
    for (const OptionSpec& o : t.options)
      if (!options_all.contains(o.name)) options_all[o.name] = json::object();
  }
  json inst = obj({{"type", {{"enum", type_names}}},
                   {"domains", {{"type", "array"}, {"items", kStr}, {"minItems", 1}}},
                   {"options", {{"type", "object"}}},
                   {"boundary_conditions", {{"type", "object"}, {"additionalProperties", {{"type", "object"}, {"required", {"kind"}}}}}},
                   {"sources", {{"type", "array"}}},
                   {"initial", {{"type", "object"}}},
                   {"notes", kStr}},
                  {"type"});
  json rules = json::array();
  for (const PhysicsType& t : reg.types) {
    json opts = json::object();
    for (const OptionSpec& o : t.options) opts[o.name] = option_schema(o);
    json then = {{"properties",
                  {{"options", obj(opts)},
                   {"boundary_conditions", {{"type", "object"}, {"additionalProperties", kinds_schema(t.condition_kinds, {"notes"}, false)}}},
                   {"sources", {{"type", "array"}, {"items", kinds_schema(t.source_kinds, {"in", "notes"}, true)}}}}}};
    rules.push_back({{"if", {{"properties", {{"type", {{"const", t.name}}}}}, {"required", {"type"}}}}, {"then", then}});
  }
  inst["allOf"] = rules;
  return {{"type", "object"}, {"additionalProperties", inst}, {"x-status", "beta"}};
}

json couplings_schema(const Registry& reg) {
  KeySet all;
  json base = {{"term", {{"enum", json::array()}}}, {"from", kStr}, {"to", kStr}, {"in", kStr}, {"on", kStr}};
  for (const char* n : {"term", "from", "to", "in", "on"}) all.add(n);
  for (const CouplingTermType& t : reg.terms) {
    base["term"]["enum"].push_back(t.name);
    for (const DataKey& k : t.data) {
      if (all.seen.count(k.name)) continue;
      all.add(k.name);
      base[k.name] = datum_schema(k);
    }
  }
  json item = {{"type", "object"}, {"required", {"term", "from", "to"}}, {"properties", base}, {"additionalProperties", false}};
  json rules = json::array();
  for (const CouplingTermType& t : reg.terms)
    rules.push_back(kind_rule("term", t.name, t.data, all, {"term", "from", "to", "in", "on"}));
  item["allOf"] = rules;
  return {{"type", "array"}, {"items", item}, {"x-status", "beta"}};
}

json selector_ref() { return {{"$ref", "#/$defs/model_selector"}}; }

json selector_def() {
  json where = {{"type", "object"},
                {"additionalProperties", {{"anyOf", {kNum, json{{"type", "array"}, {"items", kNum}, {"minItems", 2}, {"maxItems", 2}}}}}}};
  json operand = {{"anyOf", {kStr, selector_ref()}}};
  json s = obj({{"of", kStr},
                {"between", {{"type", "array"}, {"items", kStr}, {"minItems", 2}, {"maxItems", 2}}},
                {"curve", kStr},
                {"group", kStr},
                {"union", {{"type", "array"}, {"items", operand}, {"minItems", 1}}},
                {"difference", {{"type", "array"}, {"items", operand}, {"minItems", 1}}},
                {"where", where},
                {"except", {{"type", "array"}, {"items", kStr}}},
                {"tolerance_m", kNum}});
  s["oneOf"] = {{{"required", {"of"}}}, {{"required", {"between"}}}, {{"required", {"curve"}}},
                {{"required", {"group"}}}, {{"required", {"union"}}}, {{"required", {"difference"}}}};
  return s;
}

json outputs_schema() {
  static const std::vector<std::string> kinds = {"integral", "average", "maximum", "minimum", "extreme", "point",
                                                 "flux", "force", "torque", "electrode_current", "expression"};
  json out = obj({{"kind", {{"enum", kinds}}}, {"of", kStr}, {"on", kStr}, {"in", kStr}, {"at", {{"type", "array"}, {"items", kNum}}},
                  {"physics", kStr}, {"component", kStr}, {"about", {{"type", "array"}, {"items", kNum}}}, {"expr", kStr},
                  {"T_datum_K", kNum}, {"over", {{"enum", {"time"}}}}, {"notes", kStr}},
                 {"kind"});
  auto rule = [](std::vector<std::string> ks, std::vector<std::string> need) {
    json k = json::array();
    for (const std::string& s : ks) k.push_back(s);
    return json{{"if", {{"properties", {{"kind", {{"enum", k}}}}}, {"required", {"kind"}}}}, {"then", {{"required", need}}}};
  };
  out["allOf"] = {rule({"flux"}, {"of", "on"}), rule({"integral", "average", "maximum", "minimum", "extreme"}, {"of"}),
                  rule({"point"}, {"of", "at"}), rule({"force", "torque", "electrode_current"}, {"on"}),
                  rule({"expression"}, {"expr"})};
  return {{"type", "object"}, {"additionalProperties", out}, {"x-status", "beta"}};
}

json study_schema() {
  json step = obj({{"id", kStr},
                   {"type", {{"enum", {"stationary", "transient"}}}},
                   {"physics", {{"type", "array"}, {"items", kStr}, {"minItems", 1}}},
                   {"initial_from", kStr},
                   {"given", {{"type", "object"}, {"additionalProperties", kNum}}},
                   {"coupling", obj({{"strategy", {{"enum", {"one_way", "gauss_seidel", "aitken", "monolithic"}}}},
                                     {"tolerance", kNum}, {"max_iterations", kNum}})},
                   {"continuation", obj({{"scale", {{"enum", {"viscosity", "buoyancy"}}}},
                                         {"factors", {{"type", "array"}, {"items", kNum}, {"minItems", 1}}},
                                         {"ramp", kStr},
                                         {"values", {{"type", "array"}, {"items", kNum}, {"minItems", 1}}}})},
                   {"solvers", {{"type", "object"}, {"additionalProperties", {{"type", "object"}}}}},
                   {"time", obj({{"t_end_s", kNum}, {"rtol", kNum}, {"first_step_s", kNum}, {"order", kNum}})},
                   {"save", obj({{"times_s", {{"type", "array"}, {"items", kNum}, {"minItems", 1}}}, {"n", kNum}})}},
                  {"id", "type"});
  return {{"type", "array"}, {"items", step}, {"x-status", "beta"}};
}

json reactions_schema() {
  json r = obj({{"kind", {{"enum", {"butler_volmer", "linear", "tafel_anodic", "tafel_cathodic"}}}}, {"i0_A_m2", kNum},
                {"E_eq_V", kNum}, {"alpha_a", kNum}, {"alpha_c", kNum}, {"slope_V_decade", kNum}, {"notes", kStr}},
               {"kind", "i0_A_m2"});
  return {{"type", "object"},
          {"properties", {{"species", {{"type", "array"}, {"items", kStr}}},
                          {"mechanism", {{"type", "array"}}},
                          {"electrode", {{"type", "object"}, {"additionalProperties", r}}}}},
          {"additionalProperties", false},
          {"x-status", "beta"}};
}

}  // namespace

json add_model_branch(const json& em) {
  const Registry& reg = registry();
  const json& props = em.at("properties");

  // the model's own sections
  json domains = {{"type", "object"}, {"additionalProperties", {{"type", "array"}, {"items", kStr}, {"minItems", 1}}}, {"x-status", "beta"}};
  json boundaries = {{"type", "object"}, {"additionalProperties", {{"anyOf", {kStr, selector_ref()}}}}, {"x-status", "beta"}};

  // the shared sections of a model document
  json region_props = props.at("regions").at("items").at("properties");
  json mat_props = json::object();
  for (const char* k : {"kind", "mu_r", "eps_r", "tan_delta", "sigma_S_m", "bh", "B_sat", "B", "H", "warm", "cost_per_kg", "rf"})
    mat_props[k] = props.at("materials").at("additionalProperties").at("properties").at(k);
  const json prop = {{"anyOf", {{{"type", "number"}}, {{"type", "string"}}, {{"type", "object"}}}}};   // a number, an expression or a law
  mat_props["density"] = prop;
  mat_props["species"] = {{"type", "object"}};
  mat_props["thermal"] = obj({{"k_W_mK", prop}, {"cp_J_kgK", prop}, {"notes", kStr}});
  mat_props["mechanical"] = obj({{"E_Pa", prop}, {"nu", prop}, {"alpha", prop}, {"alpha_1_K", prop}, {"notes", kStr}});
  mat_props["fluid"] = obj({{"mu_Pa_s", prop}, {"beta_1_K", prop}, {"notes", kStr}});
  mat_props["electrolyte"] = obj({{"kappa_S_m", prop}, {"notes", kStr}});
  mat_props["electric"] = obj({{"sigma_S_m", prop}, {"notes", kStr}});
  json model_region = obj(region_props, {"name"});
  json curve_props = json::object();
  for (const char* k : {"name", "path", "mesh_size", "surface_size", "notes"}) curve_props[k] = props.at("curves").at("items").at("properties").at(k);
  json model_mesh = obj({{"size", kNum}, {"surface_size", kNum}, {"geometry_order", kNum}, {"refinements", kNum},
                         {"exterior", {{"enum", {"none"}}}}, {"file", kStr}, {"info_file", kStr}, {"threads", kNum}});

  // the permissive union at the root
  json loose_mesh = props.at("mesh");
  loose_mesh.erase("additionalProperties");
  loose_mesh["properties"]["exterior"] = kStr;
  json loose_region = {{"type", "object"}, {"properties", region_props}};
  json root_props = props;
  root_props["regions"] = {{"type", "array"}, {"items", loose_region}};
  root_props["materials"] = {{"type", "object"}, {"additionalProperties", {{"type", "object"}}}};
  root_props["mesh"] = loose_mesh;
  root_props["domains"] = domains;
  root_props["boundaries"] = boundaries;
  root_props["reactions"] = reactions_schema();
  root_props["physics"] = physics_schema(reg);
  root_props["couplings"] = couplings_schema(reg);
  root_props["study"] = study_schema();
  root_props["outputs"] = outputs_schema();

  static const std::vector<std::string> em_sections = {"windings", "circuits", "superconductors", "pickup", "background", "fullwave",
                                                       "cavity", "hts", "electrostatics", "mechanics", "thermal", "benchmarks",
                                                       "scan", "cost", "conductors"};
  static const std::vector<std::string> model_sections = {"domains", "boundaries", "reactions", "physics", "couplings", "study", "outputs"};
  json no_em = json::array(), no_model = json::array();
  for (const std::string& k : em_sections) no_em.push_back(json{{"required", {k}}});
  for (const std::string& k : model_sections) no_model.push_back(json{{"required", {k}}});

  json then = {{"properties",
                {{"symmetry", {{"enum", {"axisymmetric", "3d", "planar"}}}},
                 {"regions", {{"type", "array"}, {"items", model_region}}},
                 {"curves", {{"type", "array"}, {"items", obj(curve_props, {"name", "path"})}}},
                 {"materials", {{"type", "object"}, {"additionalProperties", obj(mat_props)}}},
                 {"mesh", model_mesh}}},
               {"required", {"name"}},
               {"not", {{"anyOf", no_em}}}};
  json els = {{"properties", props}, {"required", em.at("required")}, {"not", {{"anyOf", no_model}}}};

  json out = em;
  out.erase("required");
  out["properties"] = root_props;
  out["$defs"] = {{"model_selector", selector_def()}};
  out["if"] = {{"required", {"physics"}}};
  out["then"] = then;
  out["else"] = els;
  return out;
}

}  // namespace axmp::model
