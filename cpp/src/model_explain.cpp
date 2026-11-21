// explain_model and patch_document (model_explain.hpp).
//
// The listing is built from the validation report: the resolved model, the regions of each instance, the coupling
// strategy of each step and the route of each group.  Name-keyed sections are printed in sorted key order (the parsed
// model does not keep the order of the keys of a JSON object), so the listing does not depend on it.
#include "axionmultiphysics/model_explain.hpp"

#include <algorithm>
#include <cstdio>
#include <set>
#include <tuple>

#include "axionmultiphysics/model_validate.hpp"
#include "axionmultiphysics/physics_registry.hpp"
#include "axionmultiphysics/spec.hpp"
#include "model_errors.hpp"

namespace axmp::model {
namespace {

using json = nlohmann::json;

// ------------------------------------------------------------------------------------------------ formatting
std::string num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%g", v);
  return buf;
}

// A value as the listing prints it: a number with %g, a boolean as true or false, a string as written, a list in
// parentheses.
std::string show(const json& v) {
  if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
  if (v.is_string()) return v.get<std::string>();
  if (v.is_number()) return num(v.get<double>());
  if (v.is_array()) {
    std::string s = "(";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + show(v[i]);
    return s + ")";
  }
  return v.dump();
}

std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
  return s;
}

constexpr size_t kTagColumn = 94;
const std::string kCont(14, ' ');
const std::string kSub(16, ' ');

// The content padded to the tag column, then the tag; two spaces at least.
std::string tagged(const std::string& content, const std::string& tag) {
  const size_t pad = content.size() >= kTagColumn ? 2 : std::max<size_t>(kTagColumn - content.size(), 2);
  return content + std::string(pad, ' ') + tag;
}

std::string labelled(const std::string& label, const std::string& content) {
  return "  " + label + std::string(label.size() < 10 ? 10 - label.size() : 0, ' ') + "  " + content;
}

// ------------------------------------------------------------------------------------------------ selectors
std::string filters(const Selector& s) {
  std::string t;
  if (!s.where.empty()) {
    t += " where ";
    bool first = true;
    for (const auto& [k, v] : s.where) {
      if (!first) t += " and ";
      first = false;
      if (v.size() == 1) t += k + " = " + num(v[0]);
      else if (v.size() == 2) t += k + " in [" + num(v[0]) + ", " + num(v[1]) + "]";
      else t += k;
    }
  }
  if (s.tolerance_m > 0) t += " within " + num(s.tolerance_m) + " m";
  if (!s.except.empty()) t += " except " + join(s.except, ", ");
  return t;
}

std::string selector_text(const Selector& s) {
  std::string base;
  if (s.kind == "of") base = "of " + s.name;
  else if (s.kind == "between") base = "between " + join(s.between, ", ");
  else if (s.kind == "curve") base = "curve " + s.name;
  else if (s.kind == "group") base = "group " + s.name;
  else if (s.kind == "ref") base = s.name;
  else if (s.kind == "union" || s.kind == "difference") {
    std::vector<std::string> parts;
    for (const Selector& o : s.operands) parts.push_back(selector_text(o));
    base = s.kind + " of (" + join(parts, "; ") + ")";
  } else base = s.kind;
  return base + filters(s);
}

// ------------------------------------------------------------------------------------------------ engine forms
std::string engine_form(const Instance& inst, const PhysicsType& type, const Condition& c) {
  if (type.status == "absent") return "not implemented yet";
  const std::string& t = inst.type;
  const std::string& k = c.kind;
  if (t == "fluid.incompressible") {
    if (k == "wall") return "dirichlet 0";
    if (k == "inlet") return "dirichlet";
    if (k == "outlet") {
      auto it = inst.options.find("viscous_form");
      const bool laplacian = it != inst.options.end() && it->second.value.is_string() && it->second.value.get<std::string>() == "laplacian";
      return laplacian ? "do_nothing: mu du/dn - p n = 0" : "do_nothing: 2 mu eps(u) n - p n = 0";
    }
  } else if (t == "heat.transfer") {
    if (k == "temperature") return "dirichlet";
    if (k == "insulation") return "natural: k dT/dn = 0";
    if (k == "convection") return "robin: -k dT/dn = h (T - T_ext)";
  } else if (t == "electrochem.current_distribution") {
    if (k == "electrode") return "kinetic: kappa dphi/dn = i(eta)";
    if (k == "insulation") return "natural: kappa dphi/dn = 0";
  }
  return "see `axmp physics --type " + t + "`";
}

// ------------------------------------------------------------------------------------------------ materials
struct Block {
  const char* name;
  std::vector<std::tuple<const Property MaterialProps::*, const char*, const char*, const char*>> keys;   // member, key, symbol, unit
};

const std::vector<Block>& blocks() {
  static const std::vector<Block> b = {
      {"fluid", {{&MaterialProps::mu_Pa_s, "mu_Pa_s", "mu", "Pa s"}, {&MaterialProps::beta_1_K, "beta_1_K", "beta", "1/K"}}},
      {"thermal", {{&MaterialProps::k_W_mK, "k_W_mK", "k", "W/(m K)"}, {&MaterialProps::cp_J_kgK, "cp_J_kgK", "cp", "J/(kg K)"}}},
      {"electrolyte", {{&MaterialProps::kappa_S_m, "kappa_S_m", "kappa", "S/m"}}},
      {"electric", {{&MaterialProps::electric_sigma_S_m, "sigma_S_m", "sigma", "S/m"}}},
      {"mechanical", {{&MaterialProps::E_Pa, "E_Pa", "E", "Pa"}, {&MaterialProps::nu, "nu", "nu", ""}, {&MaterialProps::alpha_1_K, "alpha_1_K", "alpha", "1/K"}}},
  };
  return b;
}

json law_json(const Law& l) { return {{"law", l.kind}, {"of", l.of}}; }

// ------------------------------------------------------------------------------------------------ the listing
struct Listing {
  std::vector<std::string> lines;
  json j = json::object();
};

std::string coupling_data(const std::map<std::string, Datum>& data) {
  std::vector<std::string> items;
  for (const auto& [k, d] : data) items.push_back(k + "=" + show(d.written));
  return join(items, " ");
}

// `skip`: a key that is listed elsewhere (the reactions of an electrode condition)
json data_json(const std::map<std::string, Datum>& data, const std::string& skip = "") {
  json o = json::object();
  for (const auto& [k, d] : data)
    if (k != skip) o[k] = d.written;
  return o;
}

// The single-type row that the instance of a one-way row takes its solver settings from: the row of its type alone in
// this geometry and study type that its options select (the lookup that validation makes).
const Route* own_row_of(const Registry& reg, const ResolvedModel& m, const StudyStep& st, const Instance& in) {
  auto text = [](const json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    if (v.is_number()) return num(v.get<double>());
    return v.dump();
  };
  auto has = [](const std::vector<std::string>& v, const std::string& x) { return std::find(v.begin(), v.end(), x) != v.end(); };
  for (const Route& r : reg.routes) {
    if (r.physics != std::vector<std::string>{in.type} || !r.couplings.empty()) continue;
    if (!r.geometries.empty() && !has(r.geometries, m.symmetry)) continue;
    if (!has(r.study_types, st.type)) continue;
    bool selected = true;
    for (const auto& [option, value] : r.when) {
      const auto it = in.options.find(option);
      selected = selected && it != in.options.end() && text(it->second.value) == value;
    }
    if (selected) return &r;
  }
  return nullptr;
}

const StepReport* step_report(const ValidationReport& rep, const std::string& id) {
  for (const StepReport& s : rep.steps)
    if (s.id == id) return &s;
  return nullptr;
}

Listing build(const json& doc) {
  if (!doc.is_object())
    throw Error("expected an object", "", "explain describes a model document: a JSON object with a \"physics\" section.");
  if (!spec::is_model_document(doc))
    throw Error("explain describes a model document (a design with a \"physics\" section); this is an electromagnetic design", "",
                "Use `axmp resolve` to see an electromagnetic design with its parameters evaluated.");
  const ValidationReport rep = validate_model(doc);
  if (!rep.errors.empty()) throw rep.errors.front();
  const ResolvedModel& m = rep.model;
  const Registry& reg = registry();

  Listing out;
  std::vector<std::string>& t = out.lines;
  json& j = out.j;
  j["schema"] = "axionmultiphysics.explain/1";
  j["name"] = m.name;
  if (!m.description.empty()) j["description"] = m.description;
  j["status"] = "beta";
  j["symmetry"] = m.symmetry;

  const bool geometry_free = m.regions.empty() && m.mesh.file.empty();
  std::string geo = geometry_free ? "no geometry" : m.symmetry == "planar" ? "planar, bounded; results per metre of depth" : m.symmetry + ", bounded";
  t.push_back(m.name + "  (model document; " + geo + ")");

  // parameters
  {
    std::vector<std::string> items;
    json jp = json::object();
    for (const auto& [k, v] : m.parameters) {
      items.push_back(k + "=" + num(v));
      jp[k] = v;
    }
    if (!items.empty()) t.push_back(labelled("parameters", join(items, "  ")));
    j["parameters"] = jp;
  }

  // materials (those the document writes)
  {
    std::vector<std::string> rows;
    json jm = json::object();
    const json written = doc.contains("materials") && doc["materials"].is_object() ? doc["materials"] : json::object();
    for (const auto& [name, props] : m.materials) {
      if (!written.contains(name)) continue;
      std::vector<std::string> parts;
      json one = json::object();
      auto prop_text = [](const Property& p, const char* sym, const char* unit, bool is_density) {
        if (p.is_law()) return std::string(sym) + " law " + p.law->kind + " of " + join(p.law->of, ", ");
        return std::string(sym) + " " + num(p.value) + (*unit ? std::string(" ") + unit : std::string()) + (is_density ? "" : "");
      };
      if (props.density.given) {
        parts.push_back(prop_text(props.density, "density", "kg/m^3", true));
        one["density"] = props.density.is_law() ? law_json(*props.density.law) : json(props.density.value);
      }
      for (const Block& b : blocks()) {
        std::vector<std::string> items;
        json jb = json::object();
        for (const auto& [member, key, sym, unit] : b.keys) {
          const Property& p = props.*member;
          if (!p.given) continue;
          items.push_back(prop_text(p, sym, unit, false));
          jb[key] = p.is_law() ? law_json(*p.law) : json(p.value);
        }
        if (items.empty()) continue;
        parts.push_back(std::string(b.name) + "(" + join(items, ", ") + ")");
        one[b.name] = jb;
      }
      rows.push_back(name + ": " + join(parts, ", "));
      jm[name] = one;
    }
    for (size_t i = 0; i < rows.size(); ++i) t.push_back(i == 0 ? labelled("materials", rows[i]) : kCont + rows[i]);
    j["materials"] = jm;
  }

  // regions and domains
  {
    std::vector<std::string> items;
    json jr = json::array();
    for (const auto& r : m.regions) {
      items.push_back(r.name + " [" + r.material + "]");
      jr.push_back({{"name", r.name}, {"material", r.material}});
    }
    if (!items.empty()) t.push_back(labelled("regions", join(items, "  ")));
    j["regions"] = jr;
    if (doc.contains("domains") && doc["domains"].is_object() && !doc["domains"].empty()) {
      std::vector<std::string> ds;
      json jd = json::object();
      for (const auto& [name, d] : m.domains) {
        ds.push_back(name + " = " + join(d.regions, ", "));
        jd[name] = d.regions;
      }
      t.push_back(labelled("domains", join(ds, "  ")));
      j["domains"] = jd;
    }
  }

  // boundaries
  {
    std::map<std::string, std::vector<std::string>> users;   // instances are in sorted id order
    for (const auto& [id, inst] : m.instances)
      for (const auto& [key, c] : inst.conditions) users[key].push_back(id);
    const std::string deferred = "deferred: resolved by `axmp boundaries`";
    std::vector<std::string> rows;
    json jb = json::object();
    for (const auto& [name, sel] : m.boundaries) {
      const std::vector<std::string>& u = users[name];
      rows.push_back(name + "  " + selector_text(sel) + "  used by " + (u.empty() ? std::string("nothing") : join(u, ", ")) + "  " + deferred);
      jb[name] = {{"selector", sel.written}, {"resolution", "deferred"}, {"used_by", u}};
    }
    auto rem = users.find("remaining");
    if (rem != users.end()) {
      rows.push_back("remaining  every boundary no entry names  used by " + join(rem->second, ", ") + "  " + deferred);
      j["remaining"] = {{"used_by", rem->second}};
    }
    for (size_t i = 0; i < rows.size(); ++i) t.push_back(i == 0 ? labelled("boundaries", rows[i]) : kCont + rows[i]);
    j["boundaries"] = jb;
  }

  // reactions
  if (m.reactions.given && !m.reactions.electrode.empty()) {
    std::vector<std::string> rows;
    json je = json::object();
    for (const auto& [name, r] : m.reactions.electrode) {
      std::string row = "electrode " + name + ": " + r.kind + " i0_A_m2=" + num(r.i0_A_m2) + " E_eq_V=" + num(r.E_eq_V) +
                        " alpha_a=" + num(r.alpha_a) + " alpha_c=" + num(r.alpha_c);
      json o = {{"kind", r.kind}, {"i0_A_m2", r.i0_A_m2}, {"E_eq_V", r.E_eq_V}, {"alpha_a", r.alpha_a}, {"alpha_c", r.alpha_c}};
      if (r.slope_V_decade != 0.0) {
        row += " slope_V_decade=" + num(r.slope_V_decade);
        o["slope_V_decade"] = r.slope_V_decade;
      }
      rows.push_back(row);
      je[name] = o;
    }
    for (size_t i = 0; i < rows.size(); ++i) t.push_back(i == 0 ? labelled("reactions", rows[i]) : kCont + rows[i]);
    j["reactions"] = {{"electrode", je}};
  }

  // physics
  {
    json jp = json::object();
    bool first = true;
    for (const auto& [id, inst] : m.instances) {
      const PhysicsType* type = reg.type(inst.type);
      const std::string status = type ? type->status : "absent";
      const std::vector<std::string>& regions = rep.regions.at(id);
      const std::string head = id + "  [" + inst.type + ", " + status + "]" + (regions.empty() ? std::string() : "  on " + join(regions, ", "));
      t.push_back(first ? labelled("physics", head) : kCont + head);
      first = false;
      json jo = json::object();
      for (const auto& [name, o] : inst.options) {
        t.push_back(tagged(kSub + name + " " + show(o.value), o.written ? "[written]" : "[default]"));
        jo[name] = {{"value", o.value}, {"origin", o.written ? "written" : "default"}};
      }
      json jc = json::object();
      for (const auto& [key, c] : inst.conditions) {
        const std::string form = type ? engine_form(inst, *type, c) : "not implemented yet";
        std::vector<std::string> data;
        for (const auto& [k, d] : c.data)
          if (c.reactions.empty() || k != "reactions") data.push_back(k + "=" + show(d.written));
        if (!c.reactions.empty()) data.push_back("reactions=" + join(c.reactions, ", "));
        t.push_back(kSub + key + "  " + c.kind + "  " + (data.empty() ? std::string() : join(data, " ") + "  ") + "-> " + form);
        json one = {{"kind", c.kind}, {"data", data_json(c.data, c.reactions.empty() ? "" : "reactions")}, {"engine_form", form}};
        if (!c.reactions.empty()) one["reactions"] = c.reactions;
        jc[key] = one;
      }
      for (const SourceTerm& s : inst.sources) {
        const std::string d = coupling_data(s.data);
        t.push_back(kSub + "source  " + s.kind + (s.in.empty() ? "" : "  in " + s.in) + (d.empty() ? "" : "  " + d));
      }
      jp[id] = {{"type", inst.type}, {"status", status}, {"domains", regions}, {"options", jo}, {"conditions", jc}};
    }
    j["physics"] = jp;
  }

  // couplings
  if (!m.couplings.empty()) {
    json jc = json::array();
    for (size_t i = 0; i < m.couplings.size(); ++i) {
      const Coupling& c = m.couplings[i];
      std::string d = coupling_data(c.data);
      if (!c.in.empty()) d += (d.empty() ? "" : " ") + std::string("in=") + c.in;
      if (!c.on.empty()) d += (d.empty() ? "" : " ") + std::string("on=") + c.on;
      const std::string row = c.term + "  " + c.from + " -> " + c.to + (d.empty() ? "" : "  " + d);
      t.push_back(i == 0 ? labelled("couplings", row) : kCont + row);
      json one = {{"term", c.term}, {"from", c.from}, {"to", c.to}, {"data", data_json(c.data)}};
      if (!c.in.empty()) one["in"] = c.in;
      if (!c.on.empty()) one["on"] = c.on;
      jc.push_back(one);
    }
    j["couplings"] = jc;
  }

  // mesh
  if (!geometry_free) {
    std::string s = m.mesh.size > 0 ? "size " + num(m.mesh.size) + " m" : std::string("size automatic");
    if (m.mesh.surface_size > 0) s += ", surface size " + num(m.mesh.surface_size) + " m";
    s += rep.geometry_order > 0 ? ", geometry order " + std::to_string(rep.geometry_order) : std::string(", geometry order from the mesh file");
    const std::string line = labelled("mesh", s);
    t.push_back(m.mesh.geometry_order_written ? line : tagged(line, "[geometry_order: default]"));
    json jm = {{"geometry_order", rep.geometry_order}, {"geometry_order_origin", m.mesh.geometry_order_written ? "written" : "default"}};
    if (m.mesh.size > 0) jm["size"] = m.mesh.size;
    if (m.mesh.surface_size > 0) jm["surface_size"] = m.mesh.surface_size;
    j["mesh"] = jm;
  }

  // study
  {
    json js = json::array();
    bool first = true;
    for (const StudyStep& st : m.study) {
      const StepReport* sr = step_report(rep, st.id);
      std::string line = st.id + "  " + st.type + " over [" + join(st.physics, ", ") + "]";
      if (st.implicit_default) line = tagged(first ? labelled("study", line) : kCont + line, "[implicit]");
      else line = first ? labelled("study", line) : kCont + line;
      t.push_back(line);
      first = false;
      json one = {{"id", st.id}, {"type", st.type}, {"physics", st.physics}, {"origin", st.implicit_default ? "implicit" : "written"}};
      if (st.physics.size() > 1 && sr) {
        std::string txt = "coupling " + sr->strategy;
        json jc = {{"strategy", sr->strategy}};
        if (st.coupling.tolerance != 0.0) {
          txt += ", tolerance " + num(st.coupling.tolerance);
          jc["tolerance"] = st.coupling.tolerance;
        }
        if (st.coupling.max_iterations != 0) {
          txt += ", max_iterations " + std::to_string(st.coupling.max_iterations);
          jc["max_iterations"] = st.coupling.max_iterations;
        }
        const bool strategy_default = st.coupling.strategy.empty();
        t.push_back(strategy_default ? tagged(kCont + txt, "[coupling: default]") : kCont + txt);
        jc["strategy_origin"] = strategy_default ? "default" : "written";
        one["coupling"] = jc;
      }
      if (st.continuation.given) {
        std::vector<std::string> vals;
        for (double v : st.continuation.values) vals.push_back(num(v));
        if (!st.continuation.scale.empty()) {
          t.push_back(kCont + "continuation scale " + st.continuation.scale + ", factors " + join(vals, ", "));
          one["continuation"] = {{"scale", st.continuation.scale}, {"values", st.continuation.values}};
        } else {
          t.push_back(kCont + "continuation ramp " + st.continuation.ramp + ", values " + join(vals, ", "));
          one["continuation"] = {{"ramp", st.continuation.ramp}, {"values", st.continuation.values}};
        }
      }
      // what a step writes beyond its physics: the model records the keys that were written, and no default of them
      if (!st.initial_from.empty()) {
        t.push_back(tagged(kCont + "initial_from " + st.initial_from, "[written]"));
        one["initial_from"] = st.initial_from;
      }
      if (!st.given.empty()) {
        std::vector<std::string> items;
        for (const auto& [k, v] : st.given) items.push_back(k + "=" + num(v));
        t.push_back(tagged(kCont + "given " + join(items, ", "), "[written]"));
        one["given"] = st.given;
      }
      if (!st.time.empty()) {
        std::vector<std::string> items;
        json jt = json::object();
        for (const char* k : {"t_end_s", "rtol", "first_step_s", "order"}) {
          const auto w = st.time.find(k);
          if (w == st.time.end()) continue;
          items.push_back(std::string(k) + " " + num(w->second));
          const json value = std::string(k) == "order" ? json(static_cast<int>(w->second)) : json(w->second);
          jt[k] = {{"value", value}, {"origin", "written"}};
        }
        t.push_back(tagged(kCont + "time " + join(items, ", "), "[written]"));
        one["time"] = jt;
      }
      if (!st.save_times_s.empty() || st.save_n != 0) {
        if (!st.save_times_s.empty()) {
          std::vector<std::string> items;
          for (double v : st.save_times_s) items.push_back(num(v));
          t.push_back(tagged(kCont + "save times_s " + join(items, ", "), "[written]"));
          one["save"] = {{"times_s", st.save_times_s}};
        } else {
          t.push_back(tagged(kCont + "save n " + std::to_string(st.save_n), "[written]"));
          one["save"] = {{"n", st.save_n}};
        }
      }
      json jsol = json::object();
      if (sr) {
        for (const std::string& id : st.physics) {
          auto inst = m.instances.find(id);
          if (inst == m.instances.end()) continue;
          const Route* row = nullptr;
          for (const RouteUse& u : sr->routes)
            if (std::find(u.physics.begin(), u.physics.end(), id) != u.physics.end()) row = reg.route(u.route);
          if (!row) continue;
          if (std::find(row->solvers_of_own_row.begin(), row->solvers_of_own_row.end(), inst->second.type) != row->solvers_of_own_row.end())
            if (const Route* own = own_row_of(reg, m, st, inst->second)) row = own;
          std::vector<std::string> items;
          std::vector<std::string> overridden;
          json one_solver = json::object();
          auto written = st.solvers.find(id);
          for (const SolverSetting& s : row->solvers) {
            if (s.physics != inst->second.type) continue;
            json value = s.default_value;
            bool was_written = false;
            if (written != st.solvers.end()) {
              auto w = written->second.find(s.name);
              if (w != written->second.end()) {
                value = w->second;
                was_written = true;
                overridden.push_back(s.name);
              }
            }
            items.push_back(s.name + " " + show(value));
            one_solver[s.name] = {{"value", value}, {"origin", was_written ? "written" : "default"}};
          }
          if (items.empty()) continue;
          const std::string tag = overridden.empty() ? "[defaults]" : "[defaults; written: " + join(overridden, ", ") + "]";
          t.push_back(tagged(kCont + id + ": " + join(items, ", "), tag));
          jsol[id] = one_solver;
        }
      }
      one["solvers"] = jsol;
      js.push_back(one);
    }
    j["study"] = js;
  }

  // outputs
  if (!m.output_order.empty()) t.push_back(labelled("outputs", join(m.output_order, "  ")));
  j["outputs"] = m.output_order;

  // routes
  {
    json jr = json::array();
    for (const StudyStep& st : m.study) {
      const StepReport* sr = step_report(rep, st.id);
      if (!sr) continue;
      for (const RouteUse& u : sr->routes) {
        std::vector<std::string> types;
        for (const std::string& id : u.physics) types.push_back(m.instances.at(id).type);
        const Route* row = u.route.empty() ? nullptr : reg.route(u.route);
        std::string line = (m.study.size() > 1 ? "step " + st.id + ": " : std::string()) + join(u.physics, " + ") + ": " +
                           join(types, " + ") + " / " + st.type + (geometry_free ? std::string() : " / " + m.symmetry) + " -> ";
        json one = {{"step", st.id}, {"physics", u.physics}, {"route", row ? json(u.route) : json(nullptr)},
                    {"implemented", row ? row->implemented : false}};
        if (row) {
          line += u.route + (row->implemented ? "" : " (not implemented yet)");
          one["needs"] = {{"mesher", row->needs.mesher}, {"fluids", row->needs.fluids}, {"mpi", row->needs.mpi},
                          {"engine_3d", row->needs.engine_3d}, {"max_ranks", row->needs.max_ranks}};
        } else {
          line += "no route in this version";
        }
        t.push_back(labelled("route", line));
        if (row) {
          std::vector<std::string> needs;
          if (row->needs.mesher) needs.push_back("the mesher");
          if (row->needs.fluids) needs.push_back("an MPI build with the fluids library");
          else if (row->needs.mpi) needs.push_back("an MPI build");
          if (row->needs.engine_3d) needs.push_back("the 3D engine");
          needs.push_back(row->needs.max_ranks == 1 ? "one rank for the solve" : "any number of ranks for the solve");
          t.push_back(labelled("needs", join(needs, "; ")));
        }
        jr.push_back(one);
      }
    }
    j["routes"] = jr;
  }
  for (std::string& line : t) line.erase(line.find_last_not_of(' ') + 1);
  return out;
}

// ------------------------------------------------------------------------------------------------ patch
using Tokens = std::vector<std::string>;

bool parse_pointer(const std::string& p, Tokens& out) {
  out.clear();
  if (p.empty()) return true;
  if (p[0] != '/') return false;
  size_t from = 1;
  for (;;) {
    const size_t to = p.find('/', from);
    std::string tok = p.substr(from, to == std::string::npos ? std::string::npos : to - from);
    std::string un;
    for (size_t i = 0; i < tok.size(); ++i) {
      if (tok[i] == '~' && i + 1 < tok.size() && tok[i + 1] == '1') { un += '/'; ++i; }
      else if (tok[i] == '~' && i + 1 < tok.size() && tok[i + 1] == '0') { un += '~'; ++i; }
      else if (tok[i] == '~') return false;
      else un += tok[i];
    }
    out.push_back(un);
    if (to == std::string::npos) break;
    from = to + 1;
  }
  return true;
}

// The array index of a token: digits without a leading zero; -1 when it is not one.
long array_index(const std::string& tok) {
  if (tok.empty() || tok.size() > 9) return -1;
  if (tok.size() > 1 && tok[0] == '0') return -1;
  for (char c : tok)
    if (c < '0' || c > '9') return -1;
  return std::stol(tok);
}

struct PatchFailure {
  std::string reason;
};

// The value at the tokens, or null with `reason` set.
json* resolve(json& doc, const Tokens& toks, size_t upto, const std::string& path, std::string& reason, bool for_parent) {
  json* cur = &doc;
  for (size_t i = 0; i < upto; ++i) {
    const std::string& tok = toks[i];
    if (cur->is_object()) {
      auto it = cur->find(tok);
      if (it == cur->end()) { reason = for_parent ? "the parent of '" + path + "' does not exist" : "the path '" + path + "' does not exist"; return nullptr; }
      cur = &*it;
    } else if (cur->is_array()) {
      const long k = array_index(tok);
      if (k < 0) { reason = for_parent ? "the parent of '" + path + "' does not exist" : "the path '" + path + "' does not exist"; return nullptr; }
      if (static_cast<size_t>(k) >= cur->size()) { reason = "the index in '" + path + "' is out of range"; return nullptr; }
      cur = &(*cur)[static_cast<size_t>(k)];
    } else {
      reason = for_parent ? "the parent of '" + path + "' does not exist" : "the path '" + path + "' does not exist";
      return nullptr;
    }
  }
  return cur;
}

// get: the value at `path`; sets `reason` and returns false when there is none.
bool get_at(json& doc, const std::string& path, json& value, std::string& reason) {
  Tokens toks;
  if (!parse_pointer(path, toks)) { reason = "the path '" + path + "' does not exist"; return false; }
  json* p = resolve(doc, toks, toks.size(), path, reason, false);
  if (!p) return false;
  value = *p;
  return true;
}

bool add_at(json& doc, const std::string& path, const json& value, std::string& reason) {
  Tokens toks;
  if (!parse_pointer(path, toks)) { reason = "the parent of '" + path + "' does not exist"; return false; }
  if (toks.empty()) { doc = value; return true; }
  json* parent = resolve(doc, toks, toks.size() - 1, path, reason, true);
  if (!parent) return false;
  const std::string& last = toks.back();
  if (parent->is_object()) {
    (*parent)[last] = value;
    return true;
  }
  if (parent->is_array()) {
    if (last == "-") { parent->push_back(value); return true; }
    const long k = array_index(last);
    if (k < 0) { reason = "the parent of '" + path + "' does not exist"; return false; }
    if (static_cast<size_t>(k) > parent->size()) { reason = "the index in '" + path + "' is out of range"; return false; }
    parent->insert(parent->begin() + k, value);
    return true;
  }
  reason = "the parent of '" + path + "' does not exist";
  return false;
}

bool remove_at(json& doc, const std::string& path, std::string& reason) {
  Tokens toks;
  if (!parse_pointer(path, toks) || toks.empty()) { reason = "the path '" + path + "' does not exist"; return false; }
  json* parent = resolve(doc, toks, toks.size() - 1, path, reason, false);
  if (!parent) return false;
  const std::string& last = toks.back();
  if (parent->is_object()) {
    if (!parent->contains(last)) { reason = "the path '" + path + "' does not exist"; return false; }
    parent->erase(last);
    return true;
  }
  if (parent->is_array()) {
    const long k = array_index(last);
    if (k < 0) { reason = "the path '" + path + "' does not exist"; return false; }
    if (static_cast<size_t>(k) >= parent->size()) { reason = "the index in '" + path + "' is out of range"; return false; }
    parent->erase(parent->begin() + k);
    return true;
  }
  reason = "the path '" + path + "' does not exist";
  return false;
}

std::string kind_of(const json& v) {
  if (v.is_object()) return "an object";
  if (v.is_array()) return "a list";
  if (v.is_string()) return "a string";
  if (v.is_number()) return "a number";
  if (v.is_boolean()) return "a boolean";
  return "null";
}

void apply(json& doc, const json& op, std::string& reason) {
  const std::string kind = op["op"].get<std::string>();
  const std::string path = op["path"].get<std::string>();
  static const std::set<std::string> known = {"add", "remove", "replace", "move", "copy", "test"};
  if (!known.count(kind)) { reason = "unknown op '" + kind + "'; use one of add, remove, replace, move, copy, test"; return; }
  json value;
  if (kind == "add" || kind == "replace" || kind == "test") {
    if (!op.contains("value")) { reason = "the operation '" + kind + "' needs a \"value\""; return; }
    value = op["value"];
  }
  std::string from;
  if (kind == "move" || kind == "copy") {
    if (!op.contains("from") || !op["from"].is_string()) { reason = "the operation '" + kind + "' needs a \"from\""; return; }
    from = op["from"].get<std::string>();
  }
  bool ok = true;
  if (kind == "add") ok = add_at(doc, path, value, reason);
  else if (kind == "remove") ok = remove_at(doc, path, reason);
  else if (kind == "replace") {
    json old;
    ok = get_at(doc, path, old, reason);
    if (ok) {
      Tokens toks;
      parse_pointer(path, toks);
      if (toks.empty()) doc = value;
      else ok = remove_at(doc, path, reason) && add_at(doc, path, value, reason);
    }
  } else if (kind == "test") {
    json cur;
    ok = get_at(doc, path, cur, reason);
    if (ok && cur != value) { reason = "the value at '" + path + "' is not the expected one"; ok = false; }
  } else {
    json cur;
    if (kind == "move") {
      Tokens ft, pt;
      if (parse_pointer(from, ft) && parse_pointer(path, pt) && ft.size() < pt.size() && std::equal(ft.begin(), ft.end(), pt.begin())) {
        reason = "a value cannot be moved into one of its own children ('" + from + "' is a prefix of '" + path + "')";
        return;
      }
    }
    ok = get_at(doc, from, cur, reason);
    if (ok && kind == "move") ok = remove_at(doc, from, reason);
    if (ok) ok = add_at(doc, path, cur, reason);
  }
  if (ok) reason.clear();
  else if (reason.empty()) reason = "the operation could not be applied";
}

}  // namespace

std::string explain_model(const json& doc, bool as_json) {
  const Listing l = build(doc);
  if (as_json) return l.j.dump(2);
  std::string s;
  for (const std::string& line : l.lines) s += line + "\n";
  return s;
}

json patch_document(const json& doc, const json& ops) {
  if (!ops.is_array()) throw msg::patch_not_a_list(kind_of(ops));
  json work = doc;
  for (size_t i = 0; i < ops.size(); ++i) {
    const json& op = ops[i];
    const json op_name = op.is_object() && op.contains("op") && op["op"].is_string() ? op["op"] : json(nullptr);
    const json path = op.is_object() && op.contains("path") && op["path"].is_string() ? op["path"] : json(nullptr);
    if (!op.is_object() || op_name.is_null() || path.is_null())
      throw msg::patch_operation_failed(i, op_name, path, "the operation must be an object with \"op\" and \"path\"");
    std::string reason;
    apply(work, op, reason);
    if (!reason.empty()) throw msg::patch_operation_failed(i, op_name, path, reason);
  }
  if (spec::is_model_document(work)) {
    const ValidationReport rep = validate_model(work);
    if (!rep.errors.empty()) throw rep.errors.front();
  } else {
    spec::parse_design(work);
  }
  return work;
}

}  // namespace axmp::model
