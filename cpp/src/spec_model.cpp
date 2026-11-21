// The parser of the model document: see cpp/include/axionmultiphysics/spec_model.hpp.
//
// THE RULES OF COLLECTION.
//   1. One function per section, in the order of the layers: identity and schema, parameters, symmetry, regions, curves,
//      materials, mesh, domains, boundaries, reactions, physics, couplings, study, outputs.  Findings are added in that
//      order, and within a section in the order of the document (a JSON object iterates in key order, an array by index).
//   2. An entry is a material, a region, a curve, a domain, a boundary, an instance, one boundary condition, one source,
//      one coupling, one study step, one output, one reaction.  Within an entry the first error only is recorded, and the
//      entry is then dropped from the partial model; the other entries are checked.  The fields of an instance (its keys, its
//      type, its domains, its options, its notes) are independent of one another and each records its own first error; its
//      conditions, sources and options are entries of their own.  A dropped entry keeps its name (the
//      names a document refers to are collected from its keys and from the names read before the error), so that a
//      reference to it is not reported as unknown, and an instance keeps its type when the type is valid.
//   3. A section that is not of the JSON type its grammar needs is reported once and is unusable as a whole.  A section is
//      skipped, with no finding of its own, when what it needs is unusable: every section that evaluates an expression
//      when "parameters" has an error; the sections that need the symmetry when "symmetry" has an error; the domains and the
//      boundaries when the regions or the curves are unusable; the physics when the regions, curves, materials, domains or
//      boundaries are unusable; the couplings, the study and the outputs when the physics is unusable or skipped.
//   4. A check that needs a name list that is unusable is skipped as well: the materials of a region when "materials" is not
//      an object, the domains of a name when "domains" is not an object, the electrode reactions of a condition when
//      "reactions" is not an object.  An instance whose type is missing or unknown has no catalogue entry, so its options,
//      conditions and sources, the types at the ends of a coupling and the quantities of the outputs are not checked.
//   5. Within an entry the checks run in this order: the keys, the required keys, the kind or type, the data keys, the values.
//   6. A cause that shows in several entries is one finding: a parameter that has the name of a variable is reported once for each
//      set of variables, and the boundaries that name one another in a cycle are one finding, at the first of them.
//
// THE TIME.  The time t is a variable of the field expressions of an instance that a transient step of the study solves
// (design note 7), after the coordinates.  The physics is read before the study, so the transient steps and the
// instances each one lists are read from the "study" array as written, before the instances: a step that is not an
// object or whose type is not "transient" contributes nothing, and a transient step whose "physics" is absent or is not a
// list of names solves every instance, as the study section reads it.  The study section reports what is wrong with a
// step; this scan reports nothing.
#include "axionmultiphysics/spec_model.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <sstream>

#include "axionmultiphysics/physics_registry.hpp"
#include "model_errors.hpp"
#include "spec_internal.hpp"

namespace axmp::spec {

using detail::Node;
namespace C = axmp::model::codes;
namespace E = axmp::model::msg;
namespace M = axmp::model;

namespace {

using Strs = std::vector<std::string>;

// ================================================================================================= small helpers

// A name of the expression grammar: what an expression can refer to.
bool is_identifier(const std::string& s) {
  if (s.empty() || (s[0] >= '0' && s[0] <= '9')) return false;
  for (char c : s)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) return false;
  return true;
}
// "constant" or "function" when `name` is a word of the expression grammar, else "".  The grammar is asked, not copied: a
// constant compiles to a value, and a function called with no arguments is refused for its arity (a message that says
// "name() takes ..." or "name() needs ...") and not as unknown (a message that only quotes the name).
std::string grammar_word(const std::string& name) {
  try {
    FieldExpr::compile(name, ParamMap{}, {}, "");
    return "constant";
  } catch (const Error&) {
  }
  try {
    FieldExpr::compile(name + "()", ParamMap{}, {}, "");
  } catch (const Error& e) {
    const std::string m = e.message();
    if (m.find(": " + name + "() ") != std::string::npos) return "function";
  }
  return "";
}
bool whole(double x) { return x == std::floor(x) && std::fabs(x) < 1e9; }

bool valid_id(const std::string& id) {
  if (id.empty() || !(id[0] >= 'a' && id[0] <= 'z')) return false;
  for (char c : id)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
  return true;
}

Strs coordinates(const std::string& symmetry) {
  if (symmetry == "planar") return {"x", "y"};
  if (symmetry == "axisymmetric") return {"r", "z"};
  return {"x", "y", "z"};
}
std::string paren(const Strs& names) { return "(" + E::join(names) + ")"; }

// What the symmetry says about the components of a vector datum.
struct VectorShape {
  size_t n;
  std::string names;
};
VectorShape vector_shape(const std::string& type, const std::string& symmetry) {
  if (symmetry == "planar") return {2, "(x, y)"};
  if (symmetry == "3d") return {3, "(x, y, z)"};
  if (type == "meridional_vector") return {2, "(r, z)"};
  return {3, "(r, z, theta)"};
}

template <class T>
std::string text_of(const T& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

const M::ConditionKind* find_kind(const std::vector<M::ConditionKind>& v, const std::string& name) {
  for (const auto& k : v)
    if (k.name == name) return &k;
  return nullptr;
}
// The kind that `name` is another vocabulary's word for, or "".
std::string alias_of(const std::vector<M::ConditionKind>& v, const std::string& name) {
  for (const auto& k : v)
    if (std::find(k.aliases.begin(), k.aliases.end(), name) != k.aliases.end()) return k.name;
  return "";
}
Strs kind_names(const std::vector<M::ConditionKind>& v) {
  Strs out;
  for (const auto& k : v) out.push_back(k.name);
  return out;
}

// A FieldExpr that names something unknown is the model path's expr.unknown_variable.
std::string unknown_name_of(const std::string& message) {
  static const std::string lead = "unknown name '";
  const size_t at = message.rfind(lead);
  if (at == std::string::npos) return "";
  const size_t from = at + lead.size();
  const size_t to = message.find('\'', from);
  return to == std::string::npos ? "" : message.substr(from, to - from);
}
// `time_note` is added to the hint when the unknown name is t: why the time is not a variable of this context.
FieldExpr compile_field(const std::string& text, const ParamMap& params, const Strs& variables, const std::string& where,
                        const std::string& context, const std::string& time_note = "") {
  try {
    return FieldExpr::compile(text, params, variables, where, context);
  } catch (const Error& e) {
    if (e.code().empty()) {
      const std::string name = unknown_name_of(e.message());
      if (!name.empty()) throw E::unknown_variable(e, name, variables, name == "t" ? time_note : std::string());
    }
    throw;
  }
}

// ================================================================================================ the laws and materials

// A field that a law reads or a step fixes: T, p, B_mag, or the concentration c_<species> of a species whose name is an
// identifier, so that an expression can name it.
bool is_field_name(const std::string& f) {
  return f == "T" || f == "p" || f == "B_mag" || (f.size() > 2 && f.compare(0, 2, "c_") == 0 && is_identifier(f.substr(2)));
}

// `positive_text`, when it is given, says that the property must be positive: a table whose y is not is refused with it.
M::Law parse_law(const Node& n, const std::string& property, const char* positive_text = nullptr) {
  if (!n.has("law")) throw Error("missing required field 'law'", n.path);
  const std::string kind = n.at("law").str();
  if (kind != "expression" && kind != "table") throw E::law_unknown_kind(kind, n.path + "/law");
  if (kind == "expression") {
    n.only({"law", "of", "expr", "range", "out_of_range", "source", "allow_c0"});
  } else {
    n.only({"law", "of", "x", "y", "interpolation", "range", "out_of_range", "source", "allow_c0"});
  }
  M::Law law;
  law.kind = kind;
  if (kind == "expression") law.interpolation.clear();

  if (!n.has("of")) throw Error("missing required field 'of'", n.path);
  const Node of = n.at("of");
  if (!of.j.is_array() || of.j.empty()) throw Error("'of' must be a list of the fields the law reads", of.path, "Example: [\"T\"].");
  for (size_t i = 0; i < of.j.size(); ++i) {
    const Node f = of.at(i);
    const std::string field = f.str();
    if (std::find(law.of.begin(), law.of.end(), field) != law.of.end())
      throw Error("'of' lists the field '" + field + "' twice", f.path, "Name each field that the law reads once.");
    law.of.push_back(field);
    if (!is_field_name(field)) throw E::law_unknown_field(field, f.path);
  }
  if (kind == "table" && law.of.size() != 1)
    throw E::table_invalid("a table law reads one field, and 'of' lists " + std::to_string(law.of.size()), of.path);

  if (kind == "expression") {
    if (!n.has("expr")) throw Error("missing required field 'expr'", n.path);
  } else {
    if (!n.has("x")) throw Error("missing required field 'x'", n.path);
    if (!n.has("y")) throw Error("missing required field 'y'", n.path);
    for (const char* key : {"x", "y"}) {
      const Node a = n.at(key);
      if (!a.j.is_array()) throw Error(std::string("'") + key + "' must be a list of numbers", a.path);
      std::vector<double>& dst = std::string(key) == "x" ? law.x : law.y;
      for (size_t i = 0; i < a.j.size(); ++i) dst.push_back(a.at(i).num());
    }
  }

  law.out_of_range = n.str("out_of_range", "error");
  if (law.out_of_range != "error" && law.out_of_range != "clamp")
    throw E::law_unknown_option("out_of_range", law.out_of_range, n.path + "/out_of_range", "Use \"error\" or \"clamp\".");
  if (kind == "table") {
    law.interpolation = n.str("interpolation", "monotone_cubic");
    if (law.interpolation != "monotone_cubic" && law.interpolation != "linear")
      throw E::law_unknown_option("interpolation", law.interpolation, n.path + "/interpolation", "Use \"monotone_cubic\" or \"linear\".");
  }

  const std::string range_path = n.path + "/range";
  for (const std::string& f : law.of) {
    if (!n.has("range") || !n.j.at("range").contains(f)) throw E::range_missing(property, f, range_path);
    law.range[f] = n.at("range").at(f).range();
  }
  if (n.has("range") && n.j.at("range").is_object())
    for (auto it = n.j.at("range").begin(); it != n.j.at("range").end(); ++it)
      if (std::find(law.of.begin(), law.of.end(), it.key()) == law.of.end())
        throw Error("the range of this law lists '" + it.key() + "', which its 'of' does not", range_path + "/" + it.key(),
                    "Remove the entry, or add '" + it.key() + "' to 'of'.");
  law.source = n.str("source", "");
  law.allow_c0 = n.boolean("allow_c0", false);

  if (kind == "table") {
    if (law.x.size() != law.y.size())
      throw E::table_invalid("the table law has " + std::to_string(law.x.size()) + " values of 'x' and " + std::to_string(law.y.size()) + " of 'y'",
                             n.path + "/y");
    if (law.x.size() < 2) throw E::table_invalid("a table law needs at least two points", n.path + "/x");
    for (size_t i = 1; i < law.x.size(); ++i)
      if (!(law.x[i] > law.x[i - 1])) throw E::table_invalid("the 'x' of a table law must be strictly increasing", n.path + "/x");
    if (positive_text)
      for (size_t i = 0; i < law.y.size(); ++i)
        if (!(law.y[i] > 0.0))
          throw E::table_invalid(std::string(positive_text) + "; the table gives y = " + text_of(law.y[i]) + " at index " + std::to_string(i),
                                 n.path + "/y/" + std::to_string(i));
  } else {
    law.expr = n.at("expr").str();
    // compiled to check it: the grammar, the names and the shadowing of a field by a parameter
    compile_field(law.expr, *n.params, law.of, n.path + "/expr", "the field '{name}' that this law reads");
  }
  return law;
}

// A property of a block: a law, or a number or an expression that must be positive when `positive_text` is given.
M::Property read_property(const Node& block, const char* key, const char* positive_text) {
  const Node n = block.at(key);
  M::Property p;
  p.given = true;
  if (n.j.is_object()) {
    p.law = std::make_shared<const M::Law>(parse_law(n, key, positive_text));
    return p;
  }
  p.value = n.num();
  if (positive_text && !(p.value > 0.0)) throw Error(positive_text, n.path);
  return p;
}

// True when E_Pa or nu of a mechanical block is written as a law.
bool mechanical_has_law(const Node& b) {
  for (const char* key : {"E_Pa", "nu"})
    if (b.has(key) && b.j.at(key).is_object()) return true;
  return false;
}

// The properties that the electromagnetic Material holds and the model's Properties mirror: the density, E_Pa and nu, and
// the conductivity k_W_mK, each as the Material has it unless the model path already read it as a law.  A material of the
// document and a built-in one are mirrored by this one function, so that the two give the same Properties.
void mirror_electromagnetic_properties(M::MaterialProps& mp, bool density_written, bool has_mechanical, bool has_thermal) {
  if (!(mp.E_Pa.is_law() || mp.nu.is_law()) && has_mechanical) {
    mp.E_Pa = {true, mp.em.mech.E_Pa, nullptr};
    mp.nu = {true, mp.em.mech.nu, nullptr};
  }
  if (mp.density.is_law()) mp.em.density = 0.0;  // not the default of the kind: the electromagnetic form is "not given"
  else if (density_written || mp.em.density != 0.0) mp.density = {true, mp.em.density, nullptr};
  if (has_thermal && !mp.k_W_mK.is_law()) {
    mp.k_W_mK.given = true;
    const ThermalProps& t = mp.em.therm;
    if (t.kT_T.empty()) {
      mp.k_W_mK.value = t.k_const;
    } else {
      // the [[T, k]] spelling: the equivalent law, linear and clamped over the tabulated range
      auto law = std::make_shared<M::Law>();
      law->kind = "table";
      law->of = {"T"};
      law->x = t.kT_T;
      law->y = t.kT_k;
      law->interpolation = "linear";
      law->out_of_range = "clamp";
      law->range["T"] = {t.kT_T.front(), t.kT_T.back()};
      mp.k_W_mK.value = t.kT_k.front();
      mp.k_W_mK.law = law;
    }
  }
}

}  // namespace

namespace detail {

M::MaterialProps parse_model_material(const std::string& name, const Node& n) {
  const json& j = n.j;
  // 1. its own key check, with the lists of a model document, at the material and at each block
  n.only({"kind", "mu_r", "eps_r", "tan_delta", "sigma_S_m", "bh", "B_sat", "B", "H", "density", "warm", "cost_per_kg", "rf",
          "mechanical", "thermal", "fluid", "electrolyte", "electric", "species"});
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string& key = it.key();
    if (key == "thermal") n.at(key).only({"k_W_mK", "cp_J_kgK", "notes"});
    else if (key == "mechanical") n.at(key).only({"E_Pa", "nu", "alpha", "alpha_1_K", "notes"});
    else if (key == "fluid") n.at(key).only({"mu_Pa_s", "beta_1_K", "notes"});
    else if (key == "electrolyte") n.at(key).only({"kappa_S_m", "notes"});
    else if (key == "electric") n.at(key).only({"sigma_S_m", "notes"});
    else if (key == "species" && !it.value().is_object()) throw Error("expected an object", n.path + "/species");
  }

  // 2. the new data, from the user's node at its own path
  M::MaterialProps mp;
  double mechanical_alpha = 0.0;
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string& key = it.key();
    const Node b = n.at(key);
    if (key == "density") {
      // a density law is the model's; a number or an expression is read by the unchanged parser below
      if (b.j.is_object()) mp.density = read_property(n, "density", nullptr);
    } else if (key == "electric") {
      if (!b.has("sigma_S_m")) throw E::block_incomplete(name, "electric", "sigma_S_m", "S/m", b.path);
      mp.has_electric = true;
      mp.electric_sigma_S_m = read_property(b, "sigma_S_m", "sigma_S_m (electric conductivity) must be positive");
      if (n.has("sigma_S_m")) {
        const double level = n.at("sigma_S_m").num();
        const M::Property& e = mp.electric_sigma_S_m;
        if (e.is_law() || e.value != level)
          throw E::conductivity_ambiguous(name, level, e.is_law() ? json("law") : json(e.value), b.path + "/sigma_S_m");
      }
    } else if (key == "electrolyte") {
      if (!b.has("kappa_S_m")) throw E::block_incomplete(name, "electrolyte", "kappa_S_m", "S/m", b.path);
      mp.has_electrolyte = true;
      mp.kappa_S_m = read_property(b, "kappa_S_m", "kappa_S_m (electrolyte conductivity) must be positive");
    } else if (key == "fluid") {
      if (!b.has("mu_Pa_s")) throw E::block_incomplete(name, "fluid", "mu_Pa_s", "Pa s", b.path);
      mp.has_fluid = true;
      mp.mu_Pa_s = read_property(b, "mu_Pa_s", "mu_Pa_s (dynamic viscosity) must be positive");
      if (b.has("beta_1_K")) mp.beta_1_K = read_property(b, "beta_1_K", nullptr);
    } else if (key == "thermal") {
      if (!b.has("k_W_mK")) throw E::block_incomplete(name, "thermal", "k_W_mK", "W/(m K)", b.path);
      if (b.j.at("k_W_mK").is_object()) mp.k_W_mK = read_property(b, "k_W_mK", "k_W_mK must be positive");
      b.str("notes", "");
      if (b.has("cp_J_kgK")) mp.cp_J_kgK = read_property(b, "cp_J_kgK", "cp_J_kgK (specific heat) must be positive");
    } else if (key == "mechanical") {
      if (b.has("alpha") && b.has("alpha_1_K")) throw E::expansion_ambiguous(b.path);
      if (b.has("alpha_1_K")) mp.alpha_1_K = read_property(b, "alpha_1_K", nullptr);
      if (mechanical_has_law(b)) {
        // a block with a law is read here whole, as a thermal block with a conductivity law is; a missing key has the
        // text that the unchanged parser gives it in a block without a law
        for (const char* req : {"E_Pa", "nu"})
          if (!b.has(req)) throw Error(std::string("missing required field '") + req + "'", b.path);
        mp.E_Pa = read_property(b, "E_Pa", "E_Pa (Young's modulus) must be positive");
        mp.nu = read_property(b, "nu", nullptr);
        if (mp.nu.is_law() && mp.nu.law->kind == "table") {
          // the values of a table are numbers, so the range that a number must be in applies to each of them
          const std::vector<double>& ys = mp.nu.law->y;
          for (size_t i = 0; i < ys.size(); ++i)
            if (!(ys[i] > -1.0 && ys[i] < 0.5))
              throw E::table_invalid("Poisson's ratio must satisfy -1 < nu < 0.5; the table gives y = " + text_of(ys[i]) + " at index " + std::to_string(i),
                                     b.path + "/nu/y/" + std::to_string(i),
                                     "nu = 0.5 is incompressible and makes the displacement formulation singular; -1 < nu < 0.5.");
        }
        if (!mp.nu.is_law() && !(mp.nu.value > -1.0 && mp.nu.value < 0.5))
          throw Error(strcat("Poisson's ratio nu = ", mp.nu.value, " must satisfy -1 < nu < 0.5"), b.path + "/nu",
                      "nu = 0.5 is incompressible and makes the displacement formulation singular.");
        mechanical_alpha = b.num("alpha", 0.0);
        b.str("notes", "");
      }
    } else if (key == "species") {
      mp.has_species = true;
      mp.species = it.value();
    }
  }

  // 3. the filtered copy: the new blocks and keys removed, a property written as a law removed (with its block, when it
  // is in one), "kind" inserted when absent.  Removing keys moves no other key, so the copy's pointers are the user's
  // pointers.
  json copy = j;
  if (mp.density.is_law()) copy.erase("density");
  for (const char* block : {"fluid", "electrolyte", "electric", "species"}) copy.erase(block);
  if (copy.contains("thermal") && copy["thermal"].is_object()) {
    if (copy["thermal"].contains("k_W_mK") && copy["thermal"]["k_W_mK"].is_object()) copy.erase("thermal");
    else copy["thermal"].erase("cp_J_kgK");
  }
  if (copy.contains("mechanical") && copy["mechanical"].is_object()) {
    if (mp.E_Pa.is_law() || mp.nu.is_law()) copy.erase("mechanical");
    else copy["mechanical"].erase("alpha_1_K");
  }
  if (!copy.contains("kind")) copy["kind"] = "vacuum";

  // 4. the unchanged parser on what is left, with the same path
  try {
    mp.em = parse_material(name, Node{copy, n.path, n.params});
  } catch (const json::exception& e) {
    throw Error(e.what(), n.path);
  }
  if (mp.E_Pa.is_law() || mp.nu.is_law()) mp.em.mech.alpha = mechanical_alpha;  // the block is the model's: the electromagnetic Material has none of it but the contraction strain
  mirror_electromagnetic_properties(mp, n.has("density"), n.has("mechanical"), n.has("thermal"));
  return mp;
}

}  // namespace detail

namespace {

// ================================================================================================= the parser

struct Claim {
  std::string kind, at;
};

class Parser {
 public:
  Parser(const json& doc, M::Findings& out) : doc_(doc), out_(out) {}
  M::ResolvedModel run();

 private:
  const json& doc_;
  M::Findings& out_;
  M::ResolvedModel m_;
  ParamMap params_;

  // what each section found usable
  bool params_ok_ = true, symmetry_ok_ = true, regions_ok_ = true, curves_ok_ = true, materials_ok_ = true;
  bool domains_ok_ = true, boundaries_ok_ = true, reactions_ok_ = true, physics_usable_ = false;
  // false when an entry failed before its name could be read: the names of the section are then not all known, and a
  // reference to a region or a curve cannot be called unknown
  bool region_names_complete_ = true, curve_names_complete_ = true, study_ids_complete_ = true;
  bool has_mesh_file_ = false;
  // the instances that a transient step solves, from the study as written (see THE TIME above)
  std::set<std::string> transient_ids_;
  bool transient_all_ = false;
  bool time_variable_ = false;  // t is a variable of the field expressions being read

  // the names of the document: those that a reference may name, collected as the sections are read
  std::set<std::string> material_names_;
  bool material_names_known_ = true;
  std::map<std::string, Claim> space_;                        // regions, curves, domains and boundaries
  std::map<std::string, Strs> boundary_refs_;  // the boundaries that each boundary names, in the order they are written
  std::set<std::string> region_names_, disabled_names_, curve_names_, domain_names_, boundary_names_, electrode_names_;
  Strs region_order_, curve_order_, domain_order_, boundary_order_, instance_ids_, output_names_;

  const std::string& sym() const { return m_.symmetry; }

  void add(const Error& e) {
    if (e.code() == C::expr_parameter_shadows_variable && !shadow_first_report(e)) return;
    out_.add(M::Finding::Kind::error, e);
  }
  // A parameter that has the name of a variable is one cause, however many expressions the clash is found in (and
  // whether the variables there are (x, y) or (x, y, t)): it is reported once for each (parameter, message).  True when
  // this is the first report of it.
  bool shadow_first_report(const Error& e) {
    try {
      const json d = json::parse(e.details());
      return shadows_reported_.insert(d.at("parameter").dump() + "\n" + e.message()).second;
    } catch (const std::exception&) {
      return true;
    }
  }
  std::set<std::string> shadows_reported_;
  template <class F>
  bool guard(const std::string& entry_path, F&& f) {
    try {
      f();
      return true;
    } catch (const Error& e) {
      add(e);
    } catch (const json::exception& e) {
      add(Error(e.what(), entry_path));
    }
    return false;
  }
  Node node(const std::string& key) const { return Node{doc_.at(key), "/" + key, &params_}; }
  // the name of a failed entry, when it can still be read: it stays a name of the document
  static void keep_name(const Node& n, std::set<std::string>& names, Strs& order, bool& complete);

  // sections
  void top_level();
  void parameters();
  void symmetry();
  void regions();
  void curves();
  void materials();
  void mesh();
  void domains();
  void boundaries();
  void reactions();
  void scan_transient_steps();
  bool in_transient_step(const std::string& id) const { return transient_all_ || transient_ids_.count(id) != 0; }
  void physics();
  void couplings();
  void study();
  void outputs();

  // names
  void claim_name(const std::string& name, const char* kind, const std::string& at);
  void unique_geometry_name(const std::string& name, const std::string& path, const char* kind);
  Strs valid_domains() const;
  void check_domain(const std::string& name, const std::string& at) const;
  void check_boundary(const std::string& name, const std::string& at) const;
  static void check_reserved(const std::string& name, const std::string& at);

  // boundaries and selectors
  M::Selector selector(const Node& n, const std::string& boundary);

  // physics
  void instance(const std::string& id, const Node& n);
  void option(const M::PhysicsType& ty, M::Instance& inst, const Node& on, const std::string& key);
  void condition(const M::PhysicsType& ty, M::Instance& inst, const std::string& key, const Node& cn);
  void source(const M::PhysicsType& ty, M::Instance& inst, const Node& sn);
  // the data keys of a condition, a source or a coupling term: allowed keys, per-depth spelling, required keys, values
  void data_keys(const std::vector<M::DataKey>& keys, const Node& n, const std::set<std::string>& extra,
                 std::map<std::string, M::Datum>& data, std::vector<std::string>* reactions);
  M::Datum datum(const M::DataKey& k, const Node& v, std::vector<std::string>* reactions);
  FieldExpr compile_component(const Node& c, const Strs& variables, const std::string& context) const;

  // study and outputs
  void study_step(const Node& n, Strs& earlier);
  Strs solver_settings(const std::string& type) const;
  bool vocabulary(const std::string& physics_id, Strs& names) const;
  void output(const std::string& name, const Node& n, std::map<std::string, Strs>& deps);
  std::vector<double> point(const Node& v, const char* key) const;
};

M::ResolvedModel Parser::run() {
  if (!doc_.is_object()) {
    add(Error("a model document must be a JSON object", "", "Write {\"schema\": \"axionmultiphysics.design/1\", \"name\": ..., \"physics\": {...}}."));
    return m_;
  }
  // the names of the materials are known before the regions are read, because a region names its material
  for (const auto& kv : builtin_materials()) material_names_.insert(kv.first);
  if (doc_.contains("materials")) {
    if (doc_["materials"].is_object()) {
      for (auto it = doc_["materials"].begin(); it != doc_["materials"].end(); ++it) material_names_.insert(it.key());
    } else {
      material_names_known_ = false;
    }
  }
  top_level();
  parameters();
  symmetry();
  if (!params_ok_) return m_;  // every later section evaluates an expression

  if (symmetry_ok_) {
    regions();
    curves();
  } else {
    regions_ok_ = curves_ok_ = false;
  }
  materials();
  mesh();
  if (symmetry_ok_ && regions_ok_ && curves_ok_) {
    domains();
    boundaries();
  } else {
    domains_ok_ = boundaries_ok_ = false;
  }
  reactions();
  if (symmetry_ok_ && regions_ok_ && curves_ok_ && materials_ok_ && domains_ok_ && boundaries_ok_) {
    physics();
    if (physics_usable_) {
      couplings();
      study();
      outputs();
    }
  }
  return m_;
}

// ----------------------------------------------------------------------------------------------- identity
void Parser::top_level() {
  static const Strs kEmSections = {"windings", "circuits", "superconductors", "pickup", "background", "fullwave", "cavity", "hts",
                                   "electrostatics", "mechanics", "thermal", "benchmarks", "scan", "cost", "conductors"};
  json rest = doc_;
  for (auto it = doc_.begin(); it != doc_.end(); ++it) {
    if (std::find(kEmSections.begin(), kEmSections.end(), it.key()) != kEmSections.end()) {
      add(E::em_section_not_combinable(it.key()));
      rest.erase(it.key());
    }
  }
  const ParamMap none{};
  guard("", [&] {
    Node{rest, "", &none}
        .only({"schema", "name", "description", "notes", "symmetry", "parameters", "regions", "curves", "materials", "domains", "boundaries",
               "reactions", "mesh", "physics", "couplings", "study", "outputs"});
  });
  const Node root{doc_, "", &none};
  guard("", [&] {
    m_.schema = root.str("schema", "axionmultiphysics.design/1");
    if (m_.schema != "axionmultiphysics.design/1")
      throw Error("unsupported schema '" + m_.schema + "'", "/schema", "This version reads \"axionmultiphysics.design/1\".");
  });
  guard("", [&] { m_.name = root.req_str("name"); });
  guard("", [&] { m_.description = root.str("description", ""); });
  guard("", [&] { m_.notes = root.str("notes", ""); });
}

void Parser::parameters() {
  if (!doc_.contains("parameters")) return;
  try {
    const json& p = doc_.at("parameters");
    if (!p.is_object()) throw Error("parameters must be an object of name: value", "/parameters");
    std::map<std::string, std::string> defs;
    for (auto it = p.begin(); it != p.end(); ++it) {
      if (it.value().is_number()) {
        std::ostringstream os;
        os.precision(17);
        os << it.value().get<double>();
        defs[it.key()] = os.str();
      } else if (it.value().is_string()) {
        defs[it.key()] = it.value().get<std::string>();
      } else {
        throw Error("parameter values must be numbers or expression strings", "/parameters/" + it.key());
      }
    }
    params_ = resolve_params(defs, "/parameters");
    m_.parameters = params_;
  } catch (const Error& e) {
    add(e);
    params_ok_ = false;
  }
}

void Parser::symmetry() {
  try {
    const ParamMap none{};
    const std::string s = Node{doc_, "", &none}.str("symmetry", "axisymmetric");
    if (s == "xsec") throw E::symmetry_not_offered(s);
    if (s != "axisymmetric" && s != "3d" && s != "planar") throw E::symmetry_unknown();
    m_.symmetry = s;
  } catch (const Error& e) {
    add(e);
    symmetry_ok_ = false;
  }
}

// ----------------------------------------------------------------------------------------------- names
void Parser::claim_name(const std::string& name, const char* kind, const std::string& at) {
  const auto it = space_.find(name);
  if (it != space_.end()) throw E::name_collision(name, it->second.kind, it->second.at, kind, at);
  space_[name] = Claim{kind, at};
}

// the names that the solvers and the mesher use for what no region describes, as in the electromagnetic parser
void Parser::unique_geometry_name(const std::string& name, const std::string& path, const char* kind) {
  if (name.empty()) throw Error("names must be non-empty", path);
  for (const char* rsv : {"air", "infinity_shell"}) {
    if (name == rsv) {
      throw Error("'" + name + "' is a reserved region name", path,
                  std::string("The solvers use \"air\" for the background outside every named region "
                              "and \"infinity_shell\" for the 3D open-boundary shell; a region of that "
                              "name would be treated as vacuum. Rename it (e.g. \"") + name + "_region\").");
    }
  }
  if (name == "axis" || name == "exterior" || name == "remaining") throw E::reserved_name_taken(name, kind, path);
  claim_name(name, kind, path);
}

Strs Parser::valid_domains() const {
  Strs out;
  for (const std::string& r : region_order_)
    if (!disabled_names_.count(r)) out.push_back(r);
  out.insert(out.end(), domain_order_.begin(), domain_order_.end());
  return out;
}
void Parser::check_domain(const std::string& name, const std::string& at) const {
  if (!domains_ok_ || !region_names_complete_) return;  // the name may be that of a domain, or of a region that could not be read
  if (region_names_.count(name) || domain_names_.count(name)) return;  // also a name that was refused where it is defined
  check_reserved(name, at);
  throw E::unknown_domain(name, at, valid_domains());
}
void Parser::check_boundary(const std::string& name, const std::string& at) const {
  if (boundary_names_.count(name)) return;  // also a name that was refused where it is defined
  check_reserved(name, at);
  throw E::unknown_boundary(name, at, boundary_order_);
}
void Parser::check_reserved(const std::string& name, const std::string& at) {
  if (name == "axis" || name == "exterior" || name == "remaining") throw E::reserved_name(name, at);
}

// ----------------------------------------------------------------------------------------------- regions and curves
void Parser::keep_name(const Node& n, std::set<std::string>& names, Strs& order, bool& complete) {
  if (n.j.is_object() && n.j.contains("name") && n.j["name"].is_string()) {
    const std::string name = n.j["name"].get<std::string>();
    if (names.insert(name).second) order.push_back(name);
  } else {
    complete = false;
  }
}

void Parser::regions() {
  if (!doc_.contains("regions")) return;
  const Node rs = node("regions");
  if (!rs.j.is_array()) {
    add(Error("regions must be a list", rs.path));
    regions_ok_ = false;
    return;
  }
  const bool is3d = sym() == "3d";
  detail::PlanarGuard planar(sym() == "planar");
  for (size_t i = 0; i < rs.j.size(); ++i) {
    const Node n = rs.at(i);
    bool named = false;
    const bool ok = guard(n.path, [&] {
      n.only({"name", "shape", "solid", "material", "mesh_size", "surface_size", "enabled", "notes"});
      Region g;
      g.name = n.req_str("name");
      named = true;
      if (!region_names_.count(g.name)) {
        region_names_.insert(g.name);
        region_order_.push_back(g.name);
      }
      unique_geometry_name(g.name, n.path + "/name", "region");
      // the optional parametric switch: the region exists only if `enabled` evaluates to nonzero
      if (n.has("enabled")) {
        const Node en = n.at("enabled");
        const bool on = en.j.is_boolean() ? en.j.get<bool>() : en.num() != 0.0;
        if (!on) {
          disabled_names_.insert(g.name);
          m_.disabled_regions.push_back(g.name);
          return;
        }
      }
      if (is3d) {
        if (!n.has("solid")) throw Error("3D regions need a 'solid'", n.path, "e.g. {\"cylinder\": {...}} or {\"revolve\": {\"profile\": SHAPE}}.");
        g.solid = detail::parse_solid(n.at("solid"));
      } else {
        if (!n.has("shape"))
          throw Error(sym() == "planar" ? "planar regions need a 'shape' in the (x, y) plane"
                                        : "axisymmetric regions need a 'shape' in the (r, z) half-plane",
                      n.path);
        g.shape = detail::parse_shape(n.at("shape"));
      }
      g.material = n.str("material", "vacuum");
      if (material_names_known_ && !material_names_.count(g.material)) {
        std::string list;
        for (const auto& name : material_names_) list += (list.empty() ? "" : ", ") + name;
        throw Error("unknown material '" + g.material + "'", n.path + "/material", "Define it under \"materials\" or use one of: " + list + ".");
      }
      g.mesh_size = n.num("mesh_size", 0.0);
      if (g.mesh_size < 0) throw Error("mesh_size must be > 0 (or absent for automatic)", n.path + "/mesh_size");
      g.surface_size = n.num("surface_size", 0.0);
      if (g.surface_size < 0)
        throw Error("surface_size must be > 0 (or absent for automatic)", n.path + "/surface_size",
                    "surface_size refines the mesh at and near this region's own boundary.");
      m_.regions.push_back(std::move(g));
    });
    if (!ok && !named) keep_name(n, region_names_, region_order_, region_names_complete_);
  }
}

void Parser::curves() {
  if (!doc_.contains("curves")) return;
  const Node cs = node("curves");
  if (!cs.j.is_array()) {
    add(Error("curves must be a list", cs.path));
    curves_ok_ = false;
    return;
  }
  detail::PlanarGuard planar(sym() == "planar");
  for (size_t i = 0; i < cs.j.size(); ++i) {
    const Node n = cs.at(i);
    bool named = false;
    const bool ok = guard(n.path, [&] {
      // a curve of a model document is a marker: it has no material and no thickness
      n.only({"name", "path", "mesh_size", "surface_size", "notes"});
      Curve c;
      c.name = n.req_str("name");
      named = true;
      if (!curve_names_.count(c.name)) {
        curve_names_.insert(c.name);
        curve_order_.push_back(c.name);
      }
      unique_geometry_name(c.name, n.path + "/name", "curve");
      if (!n.has("path")) throw Error("missing required field 'path'", n.path);
      c.path = detail::parse_path(n.at("path"));
      c.mesh_size = n.num("mesh_size", 0.0);
      if (c.mesh_size < 0) throw Error("mesh_size must be > 0 (or absent for automatic)", n.path + "/mesh_size");
      c.surface_size = n.num("surface_size", 0.0);
      if (c.surface_size < 0)
        throw Error("surface_size must be > 0 (or absent for automatic)", n.path + "/surface_size",
                    "surface_size refines the mesh at and near this sheet.");
      m_.curves.push_back(std::move(c));
    });
    if (!ok && !named) keep_name(n, curve_names_, curve_order_, curve_names_complete_);
  }
}

// ----------------------------------------------------------------------------------------------- materials
void Parser::materials() {
  for (const auto& kv : builtin_materials()) {
    M::MaterialProps mp;
    mp.em = kv.second;
    mirror_electromagnetic_properties(mp, false, mp.em.mech.given, mp.em.therm.given);
    m_.materials[kv.first] = std::move(mp);
  }
  if (!doc_.contains("materials")) return;
  const Node ms = node("materials");
  if (!ms.j.is_object()) {
    add(Error("materials must be an object of name: definition", ms.path));
    materials_ok_ = false;
    return;
  }
  for (auto it = ms.j.begin(); it != ms.j.end(); ++it) {
    const Node n = ms.at(it.key());
    guard(n.path, [&] { m_.materials[it.key()] = detail::parse_model_material(it.key(), n); });
  }
}

// ----------------------------------------------------------------------------------------------- mesh
void Parser::mesh() {
  if (!doc_.contains("mesh")) return;
  const Node mn = node("mesh");
  has_mesh_file_ = mn.has("file");
  guard(mn.path, [&] {
    if (!mn.j.is_object()) throw Error("expected an object", mn.path);
    static const std::set<std::string> allowed = {"size", "surface_size", "geometry_order", "refinements", "exterior", "file", "info_file", "threads"};
    for (auto it = mn.j.begin(); it != mn.j.end(); ++it) {
      if (it.key() == "order") throw E::mesh_order();
      if (it.key() == "adapt") throw E::mesh_key_not_offered("adapt");
      if (!allowed.count(it.key())) mn.only({"size", "surface_size", "geometry_order", "refinements", "exterior", "file", "info_file", "threads"});
    }
    M::MeshKeys& k = m_.mesh;
    k.size = mn.num("size", 0.0);
    if (mn.has("size") && !(k.size > 0)) throw Error("mesh.size must be > 0 (or absent for automatic)", mn.path + "/size");
    k.surface_size = mn.num("surface_size", 0.0);
    if (mn.has("surface_size") && !(k.surface_size > 0)) throw Error("mesh.surface_size must be > 0 (or absent for automatic)", mn.path + "/surface_size");
    if (mn.has("geometry_order")) {
      if (mn.has("file")) throw E::geometry_order_from_file();
      const double g = mn.at("geometry_order").num();
      if (g != 1.0 && g != 2.0) throw Error("mesh.geometry_order must be 1 or 2", mn.path + "/geometry_order");
      k.geometry_order = static_cast<int>(g);
      k.geometry_order_written = true;
    }
    if (mn.has("refinements")) {
      const double r = mn.at("refinements").num();
      if (!(r >= 0) || !whole(r)) throw Error("mesh.refinements must be a non-negative integer", mn.path + "/refinements");
      k.refinements = static_cast<int>(r);
    }
    if (mn.has("exterior")) {
      k.exterior = mn.at("exterior").str();
      if (k.exterior == "open") throw E::exterior_not_offered();
      if (k.exterior != "none") throw Error("mesh.exterior must be \"none\"", mn.path + "/exterior", "Only \"none\" is offered in a model document.");
    }
    k.file = mn.str("file", "");
    k.info_file = mn.str("info_file", "");
    if (mn.has("threads")) {
      const double t = mn.at("threads").num();
      if (!(t >= 0) || !whole(t)) throw Error("mesh.threads must be a non-negative integer", mn.path + "/threads");
      k.threads = static_cast<int>(t);
    }
  });
}

// ----------------------------------------------------------------------------------------------- domains
void Parser::domains() {
  if (!doc_.contains("domains")) return;
  const Node ds = node("domains");
  if (!ds.j.is_object()) {
    add(Error("domains must be an object of name: list of regions", ds.path));
    domains_ok_ = false;
    return;
  }
  for (auto it = ds.j.begin(); it != ds.j.end(); ++it) {
    domain_names_.insert(it.key());
    domain_order_.push_back(it.key());
  }
  for (auto it = ds.j.begin(); it != ds.j.end(); ++it) {
    const Node n = ds.at(it.key());
    guard(n.path, [&] {
      if (it.key() == "axis" || it.key() == "exterior" || it.key() == "remaining") throw E::reserved_name_taken(it.key(), "domain", n.path);
      claim_name(it.key(), "domain", n.path);
      if (!n.j.is_array()) throw Error("a domain is a list of region names", n.path);
      if (n.j.empty()) throw Error("a domain must list at least one region", n.path);
      M::Domain d;
      for (size_t i = 0; i < n.j.size(); ++i) {
        const Node mem = n.at(i);
        const std::string r = mem.str();
        if (!region_names_.count(r) && region_names_complete_) {
          if (domain_names_.count(r)) throw Error("'" + r + "' is a domain; a domain lists regions", mem.path);
          check_reserved(r, mem.path);
          throw E::unknown_domain(r, mem.path, valid_domains());
        }
        (disabled_names_.count(r) ? d.disabled : d.regions).push_back(r);
      }
      m_.domains[it.key()] = std::move(d);
    });
  }
}

// ----------------------------------------------------------------------------------------------- boundaries
void Parser::boundaries() {
  if (!doc_.contains("boundaries")) return;
  const Node bs = node("boundaries");
  if (!bs.j.is_object()) {
    add(Error("boundaries must be an object of name: selector", bs.path));
    boundaries_ok_ = false;
    return;
  }
  for (auto it = bs.j.begin(); it != bs.j.end(); ++it) {
    boundary_names_.insert(it.key());
    boundary_order_.push_back(it.key());
  }
  for (auto it = bs.j.begin(); it != bs.j.end(); ++it) {
    const Node n = bs.at(it.key());
    guard(n.path, [&] {
      check_reserved(it.key(), n.path);
      claim_name(it.key(), "boundary", n.path);
      m_.boundaries[it.key()] = selector(n, it.key());
    });
  }
  // the cycles of the boundaries that name boundaries, one finding for each, named from its first member in the order of the
  // document (the keys of an object are in key order)
  std::set<std::string> reported;
  for (const std::string& start : boundary_order_) {
    if (reported.count(start)) continue;
    Strs path;
    std::set<std::string> visited = {start};
    std::function<bool(const std::string&)> walk = [&](const std::string& u) {
      path.push_back(u);
      const auto refs = boundary_refs_.find(u);
      if (refs != boundary_refs_.end()) {
        for (const std::string& v : refs->second) {
          if (v == start) {
            path.push_back(v);
            return true;
          }
          if (visited.count(v)) continue;
          visited.insert(v);
          if (walk(v)) return true;
        }
      }
      path.pop_back();
      return false;
    };
    if (walk(start)) {
      add(E::boundary_cycle(path, "/boundaries/" + start));
      reported.insert(path.begin(), path.end());
    }
  }
}

M::Selector Parser::selector(const Node& n, const std::string& boundary) {
  M::Selector s;
  s.written = n.j;
  if (n.j.is_string()) {
    const std::string name = n.j.get<std::string>();
    check_boundary(name, n.path);  // a reserved name that is no defined boundary is refused there
    boundary_refs_[boundary].push_back(name);
    s.kind = "ref";
    s.name = name;
    return s;
  }
  if (!n.j.is_object()) throw Error("a boundary is a selector object or the name of another boundary", n.path);
  n.only({"of", "between", "curve", "group", "union", "difference", "where", "except", "tolerance_m"});
  Strs kinds;
  for (const char* k : {"of", "between", "curve", "group", "union", "difference"})
    if (n.has(k)) kinds.push_back(k);
  if (kinds.size() != 1) throw E::selector_needs_one(n.path);
  s.kind = kinds[0];
  const bool filtered = s.kind == "of" || s.kind == "between";
  if (n.has("where") && !filtered) throw E::selector_grammar("'where' applies to 'of' and 'between' only", n.path + "/where");
  if (n.has("except") && !filtered) throw E::selector_grammar("'except' applies to 'of' and 'between' only", n.path + "/except");

  if (s.kind == "of") {
    s.name = n.at("of").str();
    check_domain(s.name, n.path + "/of");
  } else if (s.kind == "between") {
    const Node b = n.at("between");
    if (!b.j.is_array() || b.j.size() != 2) throw Error("'between' lists two names: [A, B]", b.path);
    for (size_t i = 0; i < 2; ++i) {
      const std::string name = b.at(i).str();
      s.between.push_back(name);
      if (i == 1 && name == "exterior") continue;
      if (name == "exterior" || name == "axis" || name == "remaining") throw E::reserved_name(name, b.at(i).path);
      check_domain(name, b.at(i).path);
    }
  } else if (s.kind == "curve") {
    s.name = n.at("curve").str();
    if (curve_names_complete_ && !curve_names_.count(s.name)) throw E::unknown_curve(s.name, n.path + "/curve", curve_order_);
  } else if (s.kind == "group") {
    s.name = n.at("group").str();
    // the groups of a document are those of the mesh it imports; a mesh section that is not an object was reported as such
    if (!has_mesh_file_ && !(doc_.contains("mesh") && !doc_["mesh"].is_object())) throw E::group_needs_mesh_file(boundary, n.path + "/group");
  } else {
    const Node ops = n.at(s.kind);
    if (!ops.j.is_array() || ops.j.empty()) throw Error("'" + s.kind + "' needs a list of selectors or boundary names", ops.path);
    for (size_t i = 0; i < ops.j.size(); ++i) s.operands.push_back(selector(ops.at(i), boundary));
  }

  if (n.has("where")) {
    const Node w = n.at("where");
    if (!w.j.is_object()) throw Error("'where' must be an object of coordinate: value or [min, max]", w.path);
    const Strs coords = coordinates(sym());
    for (auto it = w.j.begin(); it != w.j.end(); ++it) {
      if (std::find(coords.begin(), coords.end(), it.key()) == coords.end())
        throw E::unknown_coordinate(it.key(), sym(), coords, w.path + "/" + it.key());
      const Node c = w.at(it.key());
      if (c.j.is_array()) {
        const auto r = c.range();
        s.where[it.key()] = {r.first, r.second};
      } else {
        s.where[it.key()] = {c.num()};
      }
    }
  }
  if (n.has("except")) {
    const Node ex = n.at("except");
    if (!ex.j.is_array()) throw Error("'except' is a list of boundary names", ex.path);
    for (size_t i = 0; i < ex.j.size(); ++i) {
      const std::string name = ex.at(i).str();
      check_boundary(name, ex.at(i).path);
      boundary_refs_[boundary].push_back(name);
      s.except.push_back(name);
    }
  }
  if (n.has("tolerance_m")) {
    s.tolerance_m = n.at("tolerance_m").num();
    if (!(s.tolerance_m > 0.0)) throw E::selector_grammar("'tolerance_m' must be a positive length in metres", n.path + "/tolerance_m");
  }
  if (has_mesh_file_ && (s.kind == "of" || s.kind == "between" || s.kind == "curve"))
    out_.add(M::Finding::Kind::deferred, E::imported_mesh_selector(boundary, s.kind, n.path));
  return s;
}

// ----------------------------------------------------------------------------------------------- reactions
void Parser::reactions() {
  if (!doc_.contains("reactions")) return;
  const Node rn = node("reactions");
  if (!rn.j.is_object()) {
    add(Error("reactions must be an object", rn.path));
    reactions_ok_ = false;
    return;
  }
  M::Reactions& R = m_.reactions;
  R.given = true;
  guard(rn.path, [&] { rn.only({"species", "mechanism", "electrode"}); });
  if (rn.has("species")) {
    guard(rn.path + "/species", [&] {
      const Node sp = rn.at("species");
      if (!sp.j.is_array()) throw Error("species must be a list of names", sp.path);
      for (size_t i = 0; i < sp.j.size(); ++i) R.species.push_back(sp.at(i).str());
    });
  }
  if (rn.has("mechanism")) {
    guard(rn.path + "/mechanism", [&] {
      const Node mc = rn.at("mechanism");
      if (!mc.j.is_array()) throw Error("mechanism must be a list of reaction steps", mc.path);
      R.mechanism = mc.j;
    });
  }
  if (!rn.has("electrode")) return;
  const Node en = rn.at("electrode");
  if (!en.j.is_object()) {
    add(Error("electrode must be an object of name: reaction", en.path));
    reactions_ok_ = false;
    return;
  }
  for (auto it = en.j.begin(); it != en.j.end(); ++it) electrode_names_.insert(it.key());
  static const Strs kinds = {"butler_volmer", "linear", "tafel_anodic", "tafel_cathodic"};
  for (auto it = en.j.begin(); it != en.j.end(); ++it) {
    const Node n = en.at(it.key());
    guard(n.path, [&] {
      n.only({"kind", "i0_A_m2", "E_eq_V", "alpha_a", "alpha_c", "slope_V_decade", "notes"});
      M::ElectrodeReaction r;
      r.kind = n.req_str("kind");
      r.i0_A_m2 = n.req_num("i0_A_m2");
      if (std::find(kinds.begin(), kinds.end(), r.kind) == kinds.end()) throw E::reaction_unknown_kind(r.kind, n.path + "/kind", kinds);
      const bool tafel = r.kind == "tafel_anodic" || r.kind == "tafel_cathodic";
      r.slope_V_decade = tafel ? n.req_num("slope_V_decade") : n.num("slope_V_decade", 0.0);
      r.E_eq_V = n.num("E_eq_V", 0.0);
      r.alpha_a = n.num("alpha_a", 0.5);
      r.alpha_c = n.num("alpha_c", 0.5);
      R.electrode[it.key()] = r;
    });
  }
}

// ----------------------------------------------------------------------------------------------- physics
void Parser::physics() {
  if (!doc_.contains("physics")) {
    physics_usable_ = true;
    return;
  }
  const Node ps = node("physics");
  if (!ps.j.is_object()) {
    add(Error("physics must be an object of id: definition", ps.path));
    return;
  }
  physics_usable_ = true;
  scan_transient_steps();
  for (auto it = ps.j.begin(); it != ps.j.end(); ++it) instance_ids_.push_back(it.key());
  for (auto it = ps.j.begin(); it != ps.j.end(); ++it) instance(it.key(), ps.at(it.key()));
}

void Parser::scan_transient_steps() {
  if (!doc_.contains("study") || !doc_["study"].is_array()) return;  // no study: the one default step is stationary
  for (const json& step : doc_["study"]) {
    if (!step.is_object() || !step.contains("type") || step["type"] != "transient") continue;
    bool listed = step.contains("physics") && step["physics"].is_array();
    if (listed)
      for (const json& id : step["physics"]) listed = listed && id.is_string();
    if (!listed) {
      transient_all_ = true;
      continue;
    }
    for (const json& id : step["physics"]) transient_ids_.insert(id.get<std::string>());
  }
}

void Parser::instance(const std::string& id, const Node& n) {
  M::Instance inst;
  inst.id = id;
  inst.in_transient_step = in_transient_step(id);
  time_variable_ = inst.in_transient_step;
  const M::PhysicsType* ty = nullptr;
  if (n.j.is_object() && n.j.contains("type") && n.j["type"].is_string()) {
    ty = M::registry().type(n.j["type"].get<std::string>());
    if (ty) inst.type = ty->name;
  }
  // the identity of the instance: a refused id drops the instance, which keeps its name and its type
  bool dropped = !guard(n.path, [&] {
    if (!valid_id(id)) throw E::invalid_id(id, n.path);
    const Strs& reserved = M::registry().reserved_ids;
    if (std::find(reserved.begin(), reserved.end(), id) != reserved.end()) throw E::reserved_id(id, n.path, reserved);
  });
  if (!dropped && !n.j.is_object()) {
    add(Error("expected an object", n.path));
    dropped = true;
  }
  if (dropped) {
    m_.instances[id] = inst;
    return;
  }

  guard(n.path, [&] { n.only({"type", "domains", "options", "boundary_conditions", "sources", "initial", "notes"}); });
  guard(n.path + "/type", [&] {
    if (!n.has("type")) throw Error("missing required field 'type'", n.path);
    const Node tn = n.at("type");
    const std::string t = tn.str();
    if (M::registry().type(t)) return;
    Strs valid;
    std::string use;
    for (const auto& pt : M::registry().types) {
      valid.push_back(pt.name);
      if (use.empty() && std::find(pt.aliases.begin(), pt.aliases.end(), t) != pt.aliases.end()) use = pt.name;
    }
    throw E::unknown_catalogue_name(C::physics_unknown_type, "unknown physics type '" + t + "'", "type", t, tn.path, valid, use,
                                    "The physics types are: " + E::join(valid) + ".");
  });

  if (n.has("domains")) {
    inst.domains_written = true;
    guard(n.path + "/domains", [&] {
      const Node dn = n.at("domains");
      if (!dn.j.is_array()) throw Error("domains must be a list of region and domain names", dn.path);
      if (dn.j.empty()) throw Error("domains must list at least one region or domain", dn.path, "Name where the physics lives, or remove 'domains' for the default.");
      for (size_t i = 0; i < dn.j.size(); ++i) {
        const Node d = dn.at(i);
        check_domain(d.str(), d.path);
        inst.domains.push_back(d.str());
      }
    });
  }
  if (ty) {
    for (const auto& o : ty->options) {
      if (o.default_value.empty()) continue;
      json v = o.default_value;
      if (o.type == "integer") v = std::stoi(o.default_value);
      else if (o.type == "number") v = std::stod(o.default_value);
      inst.options[o.name] = M::OptionValue{v, false};
    }
    if (n.has("options")) {
      const Node on = n.at("options");
      if (!on.j.is_object()) {
        add(Error("expected an object", on.path));
      } else {
        for (auto it = on.j.begin(); it != on.j.end(); ++it) guard(on.path + "/" + it.key(), [&] { option(*ty, inst, on, it.key()); });
      }
    }
    if (n.has("boundary_conditions")) {
      const Node bn = n.at("boundary_conditions");
      if (!bn.j.is_object()) {
        add(Error("expected an object", bn.path));
      } else {
        for (auto it = bn.j.begin(); it != bn.j.end(); ++it) condition(*ty, inst, it.key(), bn.at(it.key()));
      }
    }
    if (n.has("sources")) {
      const Node sn = n.at("sources");
      if (!sn.j.is_array()) {
        add(Error("sources must be a list", sn.path));
      } else {
        for (size_t i = 0; i < sn.j.size(); ++i) source(*ty, inst, sn.at(i));
      }
    }
  }
  if (n.has("initial")) {
    if (n.j["initial"].is_object()) inst.initial = n.j["initial"];
    else add(Error("expected an object", n.path + "/initial"));
  }
  guard(n.path + "/notes", [&] { inst.notes = n.str("notes", ""); });
  m_.instances[id] = std::move(inst);
}

void Parser::option(const M::PhysicsType& ty, M::Instance& inst, const Node& on, const std::string& key) {
  const M::OptionSpec* spec = nullptr;
  Strs valid;
  for (const auto& o : ty.options) {
    valid.push_back(o.name);
    if (o.name == key) spec = &o;
  }
  const Node v = on.at(key);
  if (!spec) {
    std::string lead;
    if (key == "swirl" && ty.name == "fluid.incompressible")
      lead = "There is no 'swirl' option: an axisymmetric flow always carries the three components (u_r, u_z, u_theta), and a "
             "swirl-free problem is posed with zero theta data. ";
    throw E::unknown_catalogue_name(C::physics_unknown_option, "unknown option '" + key + "' for '" + ty.name + "'", "option", key, v.path,
                                    valid, "", E::list_of("options", ty.name, valid), lead);
  }
  json value;
  if (spec->type == "enum") {
    const std::string text = v.j.is_string() ? v.j.get<std::string>() : v.j.dump();
    if (std::find(spec->values.begin(), spec->values.end(), text) == spec->values.end() || !v.j.is_string())
      throw E::unknown_option_value(ty.name, key, text, v.path, spec->values);
    value = text;
  } else if (spec->type == "integer") {
    const double x = v.num();
    const bool is_whole = whole(x);
    const std::string text = is_whole ? std::to_string(static_cast<long long>(x)) : text_of(x);
    if (!is_whole || std::find(spec->values.begin(), spec->values.end(), text) == spec->values.end())
      throw E::unknown_option_value(ty.name, key, text, v.path, spec->values);
    value = static_cast<long long>(x);
  } else {
    value = v.num();
  }
  inst.options[key] = M::OptionValue{value, true};
}

FieldExpr Parser::compile_component(const Node& c, const Strs& variables, const std::string& context) const {
  if (!c.j.is_number() && !c.j.is_string()) throw Error("expected a number or an expression string", c.path, "Example: 2.5 or \"6*y*(H-y)/H^2\".");
  const std::string text = c.j.is_string() ? c.j.get<std::string>() : c.j.dump();
  return compile_field(text, params_, variables, c.path, context,
                       time_variable_ ? "" : "The time t is a variable only for a physics that a transient step of the study solves.");
}

M::Datum Parser::datum(const M::DataKey& k, const Node& v, std::vector<std::string>* reactions) {
  M::Datum d;
  d.written = v.j;
  d.field_expression = k.field_expression;
  // the variables: the coordinates, then the time when a transient step solves the instance.  The compiler names the
  // first variable that a parameter shadows, a coordinate before the time, and the context says which of the two it is.
  Strs vars = coordinates(sym());
  bool coordinate_shadowed = false;
  for (const std::string& c : vars) coordinate_shadowed = coordinate_shadowed || params_.count(c) != 0;
  if (time_variable_) vars.push_back("t");
  const std::string context = coordinate_shadowed || !time_variable_ ? E::coordinate_phrase(sym()) : E::time_phrase();
  if (k.field_expression) d.variables = vars;
  if (k.type == "number") {
    if (k.field_expression) d.exprs.push_back(compile_component(v, vars, context));
    else d.values.push_back(v.num());
  } else if (k.type == "vector" || k.type == "meridional_vector") {
    if (!v.j.is_array()) throw Error("'" + k.name + "' must be a list of components", v.path, "Write one number or expression per component.");
    const VectorShape sh = vector_shape(k.type, sym());
    if (v.j.size() != sh.n)
      throw E::vector_length(k.name, v.j.size(), sh.n, sym(), sh.names,
                             k.type == "vector" && sym() == "axisymmetric" && v.j.size() == 2, v.path);
    for (size_t i = 0; i < v.j.size(); ++i) {
      if (k.field_expression) d.exprs.push_back(compile_component(v.at(i), vars, context));
      else d.values.push_back(v.at(i).num());
    }
  } else if (k.type == "boolean") {
    if (!v.j.is_boolean()) throw Error("expected true or false", v.path);
    d.values.push_back(v.j.get<bool>() ? 1.0 : 0.0);
  } else if (k.type == "names") {
    if (!v.j.is_array()) throw Error("'" + k.name + "' must be a list of names", v.path);
    for (size_t i = 0; i < v.j.size(); ++i) {
      const Node e = v.at(i);
      const std::string name = e.str();
      if (reactions) {
        if (reactions_ok_ && !electrode_names_.count(name))
          throw E::reaction_unknown_name(name, e.path, Strs(electrode_names_.begin(), electrode_names_.end()));
        reactions->push_back(name);
      }
    }
  } else if (k.type == "boundary") {
    check_boundary(v.str(), v.path);
  } else if (k.type == "number_vector") {
    if (!v.j.is_array()) throw Error("'" + k.name + "' must be a list of numbers", v.path);
    // a displacement in the plane or the space of the model: (x, y), (r, z) or (x, y, z)
    const VectorShape sh = vector_shape("meridional_vector", sym());
    if (v.j.size() != sh.n) throw E::vector_length(k.name, v.j.size(), sh.n, sym(), sh.names, false, v.path);
    for (size_t i = 0; i < v.j.size(); ++i) d.values.push_back(v.at(i).num());
  }
  return d;
}

void Parser::data_keys(const std::vector<M::DataKey>& keys, const Node& n, const std::set<std::string>& extra,
                       std::map<std::string, M::Datum>& data, std::vector<std::string>* reactions) {
  std::set<std::string> allowed = extra;
  allowed.insert("kind");
  for (const auto& k : keys) {
    allowed.insert(k.name);
    if (!k.per_depth_twin.empty()) allowed.insert(k.per_depth_twin);
  }
  for (auto it = n.j.begin(); it != n.j.end(); ++it)
    if (!allowed.count(it.key()))
      throw Error("unknown field '" + it.key() + "'", n.path, "Allowed fields here: " + E::join(Strs(allowed.begin(), allowed.end())) + ".");
  // the spelling per metre of depth belongs to a planar model, and the total to the others
  for (const auto& k : keys) {
    if (k.per_depth_twin.empty()) continue;
    if (sym() == "planar" && n.has(k.name)) throw E::per_depth(k.name, k.per_depth_twin, k.unit, sym(), n.path + "/" + k.name);
    if (sym() != "planar" && n.has(k.per_depth_twin)) throw E::per_depth(k.per_depth_twin, k.name, k.unit, sym(), n.path + "/" + k.per_depth_twin);
  }
  for (const auto& k : keys)
    if (k.required && !n.has(k.name) && !(!k.per_depth_twin.empty() && n.has(k.per_depth_twin)))
      throw Error("missing required field '" + k.name + "'", n.path);
  for (auto it = n.j.begin(); it != n.j.end(); ++it) {
    if (extra.count(it.key()) || it.key() == "kind") continue;
    for (const auto& k : keys) {
      if (k.name != it.key() && k.per_depth_twin != it.key()) continue;
      data[it.key()] = datum(k, n.at(it.key()), k.name == "reactions" ? reactions : nullptr);
      break;
    }
  }
}

void Parser::condition(const M::PhysicsType& ty, M::Instance& inst, const std::string& key, const Node& cn) {
  guard(cn.path, [&] {
    if (key == "axis" || key == "exterior") {
      // a boundary of that name was refused where it is defined; the key is not refused a second time
      if (!boundary_names_.count(key)) throw E::reserved_name(key, cn.path);
    } else if (key != "remaining") {
      check_boundary(key, cn.path);
    }
    if (!cn.j.is_object()) throw Error("expected an object", cn.path);
    if (!cn.has("kind")) throw Error("missing required field 'kind'", cn.path);
    const Node kn = cn.at("kind");
    const std::string kind = kn.str();
    const M::ConditionKind* ck = find_kind(ty.condition_kinds, kind);
    if (!ck) {
      const Strs valid = kind_names(ty.condition_kinds);
      throw E::unknown_catalogue_name(C::physics_unknown_condition_kind, "unknown condition kind '" + kind + "' for '" + ty.name + "'", "kind",
                                      kind, kn.path, valid, alias_of(ty.condition_kinds, kind), E::list_of("condition kinds", ty.name, valid));
    }
    M::Condition c;
    c.boundary = key;
    c.kind = kind;
    c.written = cn.j;
    data_keys(ck->data, cn, {}, c.data, &c.reactions);
    inst.conditions[key] = std::move(c);
  });
}

void Parser::source(const M::PhysicsType& ty, M::Instance& inst, const Node& sn) {
  guard(sn.path, [&] {
    if (!sn.j.is_object()) throw Error("expected an object", sn.path);
    if (!sn.has("kind")) throw Error("missing required field 'kind'", sn.path);
    const Node kn = sn.at("kind");
    const std::string kind = kn.str();
    const M::ConditionKind* sk = find_kind(ty.source_kinds, kind);
    if (!sk) {
      const Strs valid = kind_names(ty.source_kinds);
      throw E::unknown_catalogue_name(C::physics_unknown_source_kind, "unknown source kind '" + kind + "' for '" + ty.name + "'", "kind", kind,
                                      kn.path, valid, alias_of(ty.source_kinds, kind), E::list_of("source kinds", ty.name, valid));
    }
    M::SourceTerm s;
    s.kind = kind;
    s.written = sn.j;
    data_keys(sk->data, sn, {"in"}, s.data, nullptr);
    if (sn.has("in")) {
      s.in = sn.at("in").str();
      check_domain(s.in, sn.path + "/in");
    }
    inst.sources.push_back(std::move(s));
  });
}

// ----------------------------------------------------------------------------------------------- couplings
void Parser::couplings() {
  if (!doc_.contains("couplings")) return;
  const Node cs = node("couplings");
  if (!cs.j.is_array()) {
    add(Error("couplings must be a list", cs.path));
    return;
  }
  const M::Registry& R = M::registry();
  for (size_t i = 0; i < cs.j.size(); ++i) {
    const Node n = cs.at(i);
    guard(n.path, [&] {
      if (!n.j.is_object()) throw Error("expected an object", n.path);
      if (!n.has("term")) throw Error("missing required field 'term'", n.path);
      const Node tn = n.at("term");
      const std::string term = tn.str();
      const M::CouplingTermType* tt = R.term(term);
      if (!tt) {
        Strs valid;
        for (const auto& t : R.terms) valid.push_back(t.name);
        throw E::unknown_term(term, tn.path, valid);
      }
      std::set<std::string> allowed = {"term", "from", "to", "in", "on"};
      for (const auto& k : tt->data) allowed.insert(k.name);
      for (auto it = n.j.begin(); it != n.j.end(); ++it)
        if (!allowed.count(it.key()))
          throw Error("unknown field '" + it.key() + "'", n.path, "Allowed fields here: " + E::join(Strs(allowed.begin(), allowed.end())) + ".");
      for (const char* key : {"from", "to"})
        if (!n.has(key)) throw Error(std::string("missing required field '") + key + "'", n.path);
      for (const auto& k : tt->data)
        if (k.required && !n.has(k.name)) throw Error("missing required field '" + k.name + "'", n.path);
      M::Coupling c;
      c.index = static_cast<int>(i);
      c.term = term;
      c.from = n.at("from").str();
      c.to = n.at("to").str();
      for (const char* end : {"from", "to"}) {
        const std::string& id = std::string(end) == "from" ? c.from : c.to;
        if (std::find(instance_ids_.begin(), instance_ids_.end(), id) == instance_ids_.end())
          throw E::coupling_unknown_physics(id, n.path + "/" + end, instance_ids_);
      }
      const std::string from_type = m_.instances.at(c.from).type, to_type = m_.instances.at(c.to).type;
      if (!from_type.empty() && !to_type.empty() && (!M::type_matches(tt->producers, from_type) || !M::type_matches(tt->consumers, to_type)))
        throw E::type_mismatch(term, c.from, c.to, from_type, to_type, tt->producers, tt->consumers, n.path);
      const bool domain = tt->support == "domain";
      if (n.has(domain ? "on" : "in")) throw E::wrong_selection_key(term, tt->support, domain ? "in" : "on", n.path + (domain ? "/on" : "/in"));
      if (n.has("in")) {
        c.in = n.at("in").str();
        check_domain(c.in, n.path + "/in");
      }
      if (n.has("on")) {
        c.on = n.at("on").str();
        check_boundary(c.on, n.path + "/on");
      }
      time_variable_ = in_transient_step(c.to);  // the term is evaluated where its consumer is solved
      for (const auto& k : tt->data) c.data[k.name] = datum(k, n.at(k.name), nullptr);
      m_.couplings.push_back(std::move(c));
    });
  }
}

// ----------------------------------------------------------------------------------------------- study
// The setting `name` of the solvers of `type`, or null.
static const M::SolverSetting* find_solver_setting(const std::string& type, const std::string& name) {
  for (const auto& r : M::registry().routes)
    for (const auto& s : r.solvers)
      if (s.physics == type && s.name == name) return &s;
  return nullptr;
}

Strs Parser::solver_settings(const std::string& type) const {
  std::set<std::string> names;
  for (const auto& r : M::registry().routes)
    for (const auto& s : r.solvers)
      if (s.physics == type) names.insert(s.name);
  return Strs(names.begin(), names.end());
}

void Parser::study() {
  if (!doc_.contains("study")) {
    M::StudyStep step;
    step.id = "default";
    step.type = "stationary";
    step.physics = instance_ids_;
    step.implicit_default = true;
    m_.study.push_back(std::move(step));
    return;
  }
  const Node ss = node("study");
  if (!ss.j.is_array()) {
    add(Error("study must be a list", ss.path));
    return;
  }
  Strs ids;
  for (size_t i = 0; i < ss.j.size(); ++i) {
    const size_t before = m_.study.size();
    study_step(ss.at(i), ids);
    if (m_.study.size() > before) m_.study.back().index = static_cast<int>(i);
  }
}

void Parser::study_step(const Node& n, Strs& ids) {
  const Strs earlier = ids;
  // the id of a step is a name of the document even when the step fails: a later 'initial_from' may name it
  if (n.j.is_object() && n.j.contains("id") && n.j["id"].is_string()) ids.push_back(n.j["id"].get<std::string>());
  else study_ids_complete_ = false;
  const bool complete_before = study_ids_complete_;
  guard(n.path, [&] {
    n.only({"id", "type", "physics", "initial_from", "given", "coupling", "continuation", "solvers", "time", "save"});
    M::StudyStep step;
    step.id = n.req_str("id");
    if (step.id.empty()) throw Error("the id of a study step must not be empty", n.path + "/id", "Give each step its own name, such as \"steady\".");
    if (std::find(earlier.begin(), earlier.end(), step.id) != earlier.end())
      throw Error("two study steps have the id '" + step.id + "'", n.path + "/id", "Give each step its own id.");
    if (!n.has("type")) throw Error("missing required field 'type'", n.path);
    const Node tn = n.at("type");
    step.type = tn.str();
    if (step.type == "eigen" || step.type == "frequency") throw E::study_type_reserved(step.type, tn.path);
    if (step.type != "stationary" && step.type != "transient") throw E::study_type_unknown(step.type, tn.path);
    const auto known = [&](const std::string& id) { return std::find(instance_ids_.begin(), instance_ids_.end(), id) != instance_ids_.end(); };
    if (n.has("physics")) {
      const Node pn = n.at("physics");
      if (!pn.j.is_array()) throw Error("physics must be a list of physics ids", pn.path);
      if (pn.j.empty())
        throw Error("a study step must solve at least one physics", pn.path, "List the physics ids of the step, or remove 'physics' to solve every physics.");
      for (size_t i = 0; i < pn.j.size(); ++i) {
        const std::string id = pn.at(i).str();
        if (!known(id)) throw E::study_unknown_physics(id, pn.at(i).path, instance_ids_);
        step.physics.push_back(id);
      }
    } else {
      step.physics = instance_ids_;
    }
    if (n.has("initial_from")) {
      step.initial_from = n.at("initial_from").str();
      if (complete_before && std::find(earlier.begin(), earlier.end(), step.initial_from) == earlier.end())
        throw E::initial_from_unknown_step(step.initial_from, n.path + "/initial_from", earlier);
    }
    if (n.has("given")) {
      const Node g = n.at("given");
      if (!g.j.is_object()) throw Error("expected an object", g.path);
      for (auto it = g.j.begin(); it != g.j.end(); ++it) {
        if (!is_field_name(it.key())) throw Error("'" + it.key() + "' is not a field name", g.path + "/" + it.key(), "'given' fixes a field, such as T or p, to a number.");
        step.given[it.key()] = g.at(it.key()).num();
      }
    }
    if (n.has("coupling")) {
      const Node c = n.at("coupling");
      c.only({"strategy", "tolerance", "max_iterations"});
      step.coupling.given = true;
      if (c.has("strategy")) {
        step.coupling.strategy = c.at("strategy").str();
        static const Strs ok = {"one_way", "gauss_seidel", "aitken", "monolithic"};
        if (std::find(ok.begin(), ok.end(), step.coupling.strategy) == ok.end()) throw E::unknown_strategy(step.coupling.strategy, c.path + "/strategy");
      }
      step.coupling.tolerance = c.num("tolerance", 0.0);
      if (c.has("tolerance") && !(step.coupling.tolerance > 0.0)) throw Error("tolerance must be positive", c.path + "/tolerance");
      if (c.has("max_iterations")) {
        const double count = c.at("max_iterations").num();
        if (!(count >= 1.0) || !whole(count)) throw Error("max_iterations must be a positive integer", c.path + "/max_iterations");
        step.coupling.max_iterations = static_cast<int>(count);
      }
    }
    if (n.has("continuation")) {
      const Node c = n.at("continuation");
      c.only({"scale", "factors", "ramp", "values"});
      if (c.has("scale") == c.has("ramp")) throw Error("a continuation has exactly one of 'scale' and 'ramp'", c.path);
      const char* list = c.has("scale") ? "factors" : "values";
      step.continuation.given = true;
      if (c.has("scale")) {
        step.continuation.scale = c.at("scale").str();
        if (step.continuation.scale != "viscosity" && step.continuation.scale != "buoyancy")
          throw Error("unknown continuation scale '" + step.continuation.scale + "'", c.path + "/scale", "Use \"viscosity\" or \"buoyancy\".");
      }
      else step.continuation.ramp = c.at("ramp").str();
      if (!c.has(list)) throw Error(std::string("missing required field '") + list + "'", c.path);
      const Node l = c.at(list);
      if (!l.j.is_array()) throw Error(std::string("'") + list + "' must be a list of numbers", l.path);
      if (l.j.empty())
        throw Error(std::string("'") + list + "' must list at least one " + (c.has("scale") ? "factor" : "value"), l.path,
                    c.has("scale") ? "The last factor is 1, the full coefficient: for example [0.1, 1.0]."
                                   : "The last value is the value that the document gives the field.");
      for (size_t i = 0; i < l.j.size(); ++i) step.continuation.values.push_back(l.at(i).num());
      if (c.has("scale") && step.continuation.values.back() != 1.0)
        throw Error("the last factor of a continuation must be 1, the full coefficient", l.path, "End the list with 1.0.");
    }
    if (n.has("solvers")) {
      const Node sv = n.at("solvers");
      if (!sv.j.is_object()) throw Error("expected an object", sv.path);
      for (auto it = sv.j.begin(); it != sv.j.end(); ++it) {
        const Node s = sv.at(it.key());
        if (!known(it.key())) throw E::study_unknown_physics(it.key(), s.path, instance_ids_);
        if (!s.j.is_object()) throw Error("expected an object", s.path);
        const std::string type = m_.instances.at(it.key()).type;
        if (!type.empty()) {
          const Strs allowed = solver_settings(type);
          if (allowed.empty() && s.j.size() > 0)
            throw Error("physics '" + it.key() + "' (" + type + ") takes no solver settings", s.path, "Remove the entry for '" + it.key() + "' from 'solvers'.");
          for (auto st = s.j.begin(); st != s.j.end(); ++st)
            if (std::find(allowed.begin(), allowed.end(), st.key()) == allowed.end())
              throw Error("unknown field '" + st.key() + "'", s.path, "Allowed fields here: " + E::join(allowed) + ".");
          for (auto st = s.j.begin(); st != s.j.end(); ++st) {
            const M::SolverSetting* setting = find_solver_setting(type, st.key());
            const json& x = st.value();
            const bool ok = !setting || (setting->type == "boolean" ? x.is_boolean()
                                         : setting->type == "integer" ? x.is_number() && whole(x.get<double>())
                                                                      : x.is_number());
            if (!ok) throw E::solver_value(it.key(), st.key(), setting->type, x, setting->default_value, s.path + "/" + st.key());
          }
        }
        for (auto st = s.j.begin(); st != s.j.end(); ++st) step.solvers[it.key()][st.key()] = st.value();
      }
    }
    if (n.has("time")) {
      const Node t = n.at("time");
      t.only({"t_end_s", "rtol", "first_step_s", "order"});
      for (auto it = t.j.begin(); it != t.j.end(); ++it) {
        step.time[it.key()] = t.at(it.key()).num();
        if ((it.key() == "t_end_s" || it.key() == "first_step_s" || it.key() == "rtol") && !(step.time[it.key()] > 0.0))
          throw Error(it.key() + " must be positive", t.path + "/" + it.key());
        // the orders that the time integrators have; the orders of a route are the validation's to check
        if (it.key() == "order" && !(whole(step.time["order"]) && step.time["order"] >= 1.0 && step.time["order"] <= 5.0))
          throw Error("order must be an integer from 1 to 5", t.path + "/order", "It is the order of the time integration (BDF).");
      }
    }
    if (step.type == "transient" && !step.time.count("t_end_s")) {
      const std::string hint = "A transient step needs its end time, which has no sensible default: \"time\": {\"t_end_s\": ...}.";
      if (!n.has("time")) throw Error("missing required field 'time'", n.path, hint);
      throw Error("missing required field 't_end_s'", n.path + "/time", hint);
    }
    if (n.has("save")) {
      const Node sv = n.at("save");
      sv.only({"times_s", "n"});
      if (sv.has("times_s") && sv.has("n"))
        throw Error("'save' takes 'times_s' or 'n', not both", sv.path, "Write the times, as in {\"times_s\": [1.0, 5.0]}, or their number, as in {\"n\": 51}.");
      if (sv.has("times_s")) {
        const Node ts = sv.at("times_s");
        if (!ts.j.is_array()) throw Error("'times_s' must be a list of times", ts.path);
        if (ts.j.empty()) throw Error("'times_s' must list at least one time", ts.path);
        for (size_t i = 0; i < ts.j.size(); ++i) {
          const double t = ts.at(i).num();
          if (!(t >= 0.0)) throw Error("a time of 'times_s' must not be negative", ts.at(i).path);
          if (i > 0 && !(t > step.save_times_s.back()))
            throw Error("'times_s' must be strictly increasing", ts.at(i).path, "List the output times in increasing order, each once.");
          step.save_times_s.push_back(t);
        }
      }
      if (sv.has("n")) {
        const double count = sv.at("n").num();
        if (!(count >= 1.0) || !whole(count)) throw Error("n must be a positive integer", sv.path + "/n", "It is the number of output times, as in {\"n\": 51}.");
        step.save_n = static_cast<int>(count);
      }
    }
    m_.study.push_back(std::move(step));
  });
}

// ----------------------------------------------------------------------------------------------- outputs
// The quantity names of the instances' types in this geometry, in the registry's order, each once.  False when an instance
// has no type that the catalogue knows, so that the names cannot be listed.
bool Parser::vocabulary(const std::string& physics_id, Strs& names) const {
  Strs types;
  if (!physics_id.empty()) {
    const auto it = m_.instances.find(physics_id);
    if (it == m_.instances.end() || it->second.type.empty()) return false;
    types.push_back(it->second.type);
  } else {
    for (const auto& kv : m_.instances) {
      if (kv.second.type.empty()) return false;
      types.push_back(kv.second.type);
    }
  }
  for (const std::string& t : types) {
    const M::PhysicsType* ty = M::registry().type(t);
    if (!ty || !ty->quantities.count(sym())) continue;
    for (const auto& q : ty->quantities.at(sym()))
      if (std::find(names.begin(), names.end(), q.name) == names.end()) names.push_back(q.name);
  }
  return true;
}

// The components that a force or a torque can have in a geometry: the components of a vector there, and for a torque the one
// about the normal of the plane, or the three of the space. The flow engines of an axisymmetric model return the axial
// force and the axial torque only, so those are the only components there.
Strs component_names(const std::string& kind, const std::string& symmetry) {
  if (kind == "torque") return symmetry == "3d" ? Strs{"x", "y", "z"} : Strs{"z"};
  if (symmetry == "planar") return {"x", "y"};
  if (symmetry == "3d") return {"x", "y", "z"};
  return {"z"};
}

std::vector<double> Parser::point(const Node& v, const char* key) const {
  if (!v.j.is_array()) throw Error(std::string("'") + key + "' must be a list of coordinates", v.path);
  const size_t want = sym() == "3d" ? 3 : 2;
  if (v.j.size() != want) {
    const Strs c = coordinates(sym());
    throw E::vector_length(key, v.j.size(), want, sym(), paren(c), false, v.path);
  }
  std::vector<double> out;
  for (size_t i = 0; i < v.j.size(); ++i) out.push_back(v.at(i).num());
  return out;
}

void Parser::outputs() {
  if (!doc_.contains("outputs")) return;
  const Node os = node("outputs");
  if (!os.j.is_object()) {
    add(Error("outputs must be an object of name: definition", os.path));
    return;
  }
  for (auto it = os.j.begin(); it != os.j.end(); ++it) output_names_.push_back(it.key());
  std::map<std::string, Strs> deps;  // the expression outputs that were read: the outputs each reads
  for (auto it = os.j.begin(); it != os.j.end(); ++it) output(it.key(), os.at(it.key()), deps);

  // the cycles of the expression outputs, one finding for each, named from its smallest member
  std::set<std::string> reported;
  for (const auto& kv : deps) {
    const std::string start = kv.first;
    if (reported.count(start)) continue;
    Strs path;
    std::set<std::string> visited = {start};
    std::function<bool(const std::string&)> walk = [&](const std::string& u) {
      path.push_back(u);
      for (const std::string& v : deps.at(u)) {
        if (v == start) {
          path.push_back(v);
          return true;
        }
        if (visited.count(v) || !deps.count(v)) continue;
        visited.insert(v);
        if (walk(v)) return true;
      }
      path.pop_back();
      return false;
    };
    if (walk(start)) {
      add(E::output_cycle(path, "/outputs/" + start + "/expr"));
      reported.insert(path.begin(), path.end());
    }
  }
  // the order of evaluation: an expression output after every output it reads
  std::set<std::string> done;
  std::function<void(const std::string&)> emit = [&](const std::string& name) {
    if (!done.insert(name).second) return;
    const auto d = deps.find(name);
    if (d != deps.end())
      for (const std::string& v : d->second)
        if (m_.outputs.count(v)) emit(v);
    m_.output_order.push_back(name);
  };
  for (const auto& kv : m_.outputs) emit(kv.first);
}

void Parser::output(const std::string& name, const Node& n, std::map<std::string, Strs>& deps) {
  guard(n.path, [&] {
    if (!is_identifier(name)) throw E::output_invalid_name(name, n.path);
    if (const std::string word = grammar_word(name); !word.empty()) throw E::output_name_is_grammar(name, word, n.path);
    if (!n.j.is_object()) throw Error("expected an object", n.path);
    n.only({"kind", "of", "on", "in", "at", "physics", "component", "about", "expr", "T_datum_K", "over", "notes"});
    if (!n.has("kind")) throw Error("missing required field 'kind'", n.path);
    const Node kn = n.at("kind");
    M::Output o;
    o.name = name;
    o.kind = kn.str();
    o.written = n.j;
    static const Strs kinds = {"integral", "average", "maximum", "minimum", "extreme", "point", "flux", "force", "torque", "electrode_current", "expression"};
    if (std::find(kinds.begin(), kinds.end(), o.kind) == kinds.end()) throw E::output_unknown_kind(o.kind, kn.path, kinds);
    const bool of = n.has("of"), on = n.has("on"), in = n.has("in"), at = n.has("at"), expr = n.has("expr");
    const auto needs = [&](const std::string& what) { throw E::output_grammar(name, o.kind, what, n.path); };
    if (o.kind == "flux") {
      if (!(of && on)) needs("'of' and 'on'");
    } else if (o.kind == "integral" || o.kind == "average" || o.kind == "maximum" || o.kind == "minimum" || o.kind == "extreme") {
      if (!(of && (on != in))) needs("'of' and one of 'on' and 'in'");
    } else if (o.kind == "point") {
      if (!(of && at)) needs("'of' and 'at'");
    } else if (o.kind == "force" || o.kind == "torque" || o.kind == "electrode_current") {
      if (!on) needs("'on'");
    } else if (!expr) {
      needs("'expr'");
    }
    // the keys that the kind takes: a key that belongs to another kind is refused where it is written
    {
      Strs takes = {"kind", "over", "notes"};
      const auto take = [&](std::initializer_list<const char*> keys) { for (const char* k : keys) takes.push_back(k); };
      if (o.kind == "flux") take({"of", "on", "physics", "T_datum_K"});
      else if (o.kind == "point") take({"of", "at", "physics"});
      else if (o.kind == "force" || o.kind == "torque") take({"on", "physics", "component", "about"});
      else if (o.kind == "electrode_current") take({"on", "physics"});
      else if (o.kind == "expression") take({"expr"});
      else take({"of", "on", "in", "physics"});
      for (auto it = n.j.begin(); it != n.j.end(); ++it)
        if (std::find(takes.begin(), takes.end(), it.key()) == takes.end()) throw E::output_foreign_key(name, o.kind, it.key(), takes, n.path + "/" + it.key());
    }
    if (n.has("physics")) {
      const Node pn = n.at("physics");
      o.physics = pn.str();
      if (std::find(instance_ids_.begin(), instance_ids_.end(), o.physics) == instance_ids_.end())
        throw E::output_unknown_physics(o.physics, pn.path, instance_ids_);
    }
    if (of) {
      const Node qn = n.at("of");
      o.of = qn.str();
      Strs names;
      if (vocabulary(o.physics, names) && std::find(names.begin(), names.end(), o.of) == names.end())
        throw E::output_unknown_quantity(o.of, sym(), qn.path, names);
    }
    if (on) {
      const Node bn = n.at("on");
      o.on = bn.str();
      if (!boundary_names_.count(o.on)) {
        check_reserved(o.on, bn.path);
        throw E::output_unknown_boundary(o.on, bn.path, boundary_order_);
      }
    }
    if (in) {
      const Node dn = n.at("in");
      o.in = dn.str();
      check_domain(o.in, dn.path);
    }
    if (at) o.at = point(n.at("at"), "at");
    if (n.has("component")) {
      const Node cn = n.at("component");
      o.component = cn.str();
      const Strs valid = component_names(o.kind, sym());
      if (std::find(valid.begin(), valid.end(), o.component) == valid.end()) throw E::output_unknown_component(name, o.component, o.kind, sym(), valid, cn.path);
    }
    if (n.has("over") && n.at("over").str() != "time") throw E::output_over(name, n.path + "/over");
    if (n.has("about")) point(n.at("about"), "about");
    if (n.has("T_datum_K")) n.at("T_datum_K").num();
    if (expr) {
      const Node en = n.at("expr");
      o.expr = en.str();
      const FieldExpr f = compile_field(o.expr, params_, output_names_, en.path, "the output '{name}'");
      deps[name] = f.used_variables();
      o.depends_on = f.used_variables();
    }
    m_.outputs[name] = std::move(o);
  });
}

}  // namespace

bool is_model_document(const json& doc) { return doc.is_object() && doc.contains("physics"); }

M::ResolvedModel parse_model_collect(const json& doc, M::Findings& out) {
  Parser p(doc, out);
  return p.run();
}

M::ResolvedModel parse_model(const json& doc) {
  M::Findings f;
  M::ResolvedModel m = parse_model_collect(doc, f);
  if (f.has_errors()) throw Error(f.first_error());
  return m;
}

std::vector<std::string> serves_model_documents() { return M::msg::serves_model_documents(); }

void require_em_document(const json& doc, const std::string& operation) {
  if (is_model_document(doc)) throw M::msg::em_only_operation(doc, json(operation));
}

void refuse_unserved_model_document(const json& doc, const std::string& operation) {
  if (is_model_document(doc)) throw M::msg::operation_not_implemented(doc, operation);
}

json resolved_model_geometry(const M::ResolvedModel& m) {
  auto kind_name = [](MaterialKind k) -> const char* {
    switch (k) {
      case MaterialKind::Vacuum: return "vacuum";
      case MaterialKind::Linear: return "linear";
      case MaterialKind::Iron: return "iron";
      case MaterialKind::Superconductor: return "superconductor";
      case MaterialKind::Conductor: return "conductor";
    }
    return "vacuum";
  };
  const bool is3d = m.symmetry == "3d";
  json regions = json::array(), curves = json::array(), materials = json::object();
  for (const auto& g : m.regions) {
    json r = {{"name", g.name}, {"material", g.material}};
    if (auto it = m.materials.find(g.material); it != m.materials.end()) r["kind"] = kind_name(it->second.em.kind);
    if (is3d) r["solid"] = g.solid.resolved();
    else r["shape"] = g.shape.resolved();
    if (g.mesh_size > 0) r["mesh_size"] = g.mesh_size;
    if (g.surface_size > 0) r["surface_size"] = g.surface_size;
    regions.push_back(r);
  }
  for (const auto& c : m.curves) curves.push_back({{"name", c.name}, {"path", c.path.resolved()}});
  for (const auto& [name, mp] : m.materials) {
    const Material& e = mp.em;
    materials[name] = {{"kind", kind_name(e.kind)}, {"mu_r", e.mu_r}, {"eps_r", e.eps_r}, {"tan_delta", e.tan_delta},
                       {"sigma_S_m", e.sigma_S_m}, {"rf_model", e.rf.model}};
  }
  const json mesh = {{"size", m.mesh.size}, {"surface_size", m.mesh.surface_size}, {"geometry_order", m.mesh.geometry_order},
                     {"refinements", m.mesh.refinements}, {"exterior", m.mesh.exterior}};
  return json{{"name", m.name}, {"symmetry", m.symmetry}, {"parameters", m.parameters}, {"regions", regions},
              {"curves", curves}, {"disabled_regions", m.disabled_regions}, {"materials", materials}, {"mesh", mesh}};
}

namespace {
// RFC 6901: '~' is written ~0 and '/' is written ~1.
std::string pointer_token(const std::string& key) {
  std::string out;
  for (char c : key) {
    if (c == '~') out += "~0";
    else if (c == '/') out += "~1";
    else out += c;
  }
  return out;
}

// The SAX reader of scan_duplicate_keys: one frame per open object or array; it stops at the first repeated key.
class DuplicateKeyScan : public nlohmann::json_sax<json> {
 public:
  bool found = false;
  std::string key_found, pointer;

  bool null() override { return value(); }
  bool boolean(bool) override { return value(); }
  bool number_integer(number_integer_t) override { return value(); }
  bool number_unsigned(number_unsigned_t) override { return value(); }
  bool number_float(number_float_t, const string_t&) override { return value(); }
  bool string(string_t&) override { return value(); }
  bool binary(binary_t&) override { return value(); }
  bool start_object(std::size_t) override {
    frames_.push_back(Frame{true, {}, {}, 0});
    return true;
  }
  bool key(string_t& k) override {
    Frame& f = frames_.back();
    if (!f.keys.insert(k).second) {
      found = true;
      key_found = k;
      pointer.clear();
      for (size_t i = 0; i + 1 < frames_.size(); ++i)
        pointer += "/" + (frames_[i].object ? pointer_token(frames_[i].key) : std::to_string(frames_[i].index));
      pointer += "/" + pointer_token(k);
      return false;
    }
    f.key = k;
    return true;
  }
  bool end_object() override {
    frames_.pop_back();
    return value();
  }
  bool start_array(std::size_t) override {
    frames_.push_back(Frame{false, {}, {}, 0});
    return true;
  }
  bool end_array() override {
    frames_.pop_back();
    return value();
  }
  bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override { return false; }

 private:
  struct Frame {
    bool object;
    std::set<std::string> keys;
    std::string key;  // object: the key of the value being read
    size_t index;     // array: the index of the value being read
  };
  std::vector<Frame> frames_;
  // A value ended: in an array, the next value has the next index.
  bool value() {
    if (!frames_.empty() && !frames_.back().object) ++frames_.back().index;
    return true;
  }
};
}  // namespace

void scan_duplicate_keys(const std::string& text, const std::string& origin) {
  DuplicateKeyScan scan;
  json::sax_parse(text, &scan);
  if (scan.found) throw M::msg::duplicate_key(scan.key_found, scan.pointer, origin);
}

}  // namespace axmp::spec
