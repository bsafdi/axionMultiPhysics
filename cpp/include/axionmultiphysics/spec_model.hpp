// The model document (beta): a design that has a "physics" section.  It is read by its own parser, parse_model, which
// shares no state with the parser of an electromagnetic design (parse_design in spec.hpp) and leaves that parser's grammar,
// messages and lists untouched.
//
// The parser has one function per section, in the order of the layers of a model document: identity and schema,
// parameters, symmetry, regions and curves, materials, mesh, domains, boundaries, reactions, physics, couplings, study,
// outputs.  parse_model_collect records every grammar error it finds in a findings sink, so that one run reports every
// defect that does not hide another; parse_model throws the first.  What each section needs of the earlier ones, and
// what one defect skips, is the rule of collection in spec_model.cpp.
//
// What is read here is the grammar of the document and the names it uses against the physics catalogue
// (physics_registry.hpp): the types, options, condition kinds and data keys, never the cells of the route table, which are
// the validation's.  Every expression is evaluated and every default is explicit in the result.
#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/expr.hpp"
#include "axionmultiphysics/field_expr.hpp"
#include "axionmultiphysics/spec.hpp"

namespace axmp::spec::detail {
struct Node;  // spec_internal.hpp
}

namespace axmp::model {

using json = nlohmann::json;

class Findings;  // model_errors.hpp (internal): the sink of the model path

// A property law (design note 6.3): an expression or a table of the published fields.
struct Law {
  std::string kind;                                   // "expression" | "table"
  std::vector<std::string> of;                        // the fields it reads: T, p, B_mag, c_<species>
  std::string expr;                                   // expression law: the text as written
  std::vector<double> x, y;                           // table law
  std::string interpolation = "monotone_cubic";       // table law: "monotone_cubic" | "linear"
  std::map<std::string, std::pair<double, double>> range;   // the validity interval per field
  std::string out_of_range = "error";                 // "error" | "clamp"
  std::string source;                                 // provenance, free text
  bool allow_c0 = false;                              // a linear table inside a Newton iteration
};

// A property of a material block: absent, a constant (a number or an expression, evaluated) or a law.
struct Property {
  bool given = false;
  double value = 0.0;
  std::shared_ptr<const Law> law;
  bool is_law() const { return law != nullptr; }
};

// A material of a model document: the electromagnetic Material that the unchanged electromagnetic parser makes of what
// it reads, and the data that only a model document has.
struct MaterialProps {
  spec::Material em;
  Property density;                    // also in em.density unless it is a law (em.density is then 0, "not given")
  Property mu_Pa_s, beta_1_K;          // fluid
  Property kappa_S_m;                  // electrolyte
  Property electric_sigma_S_m;         // electric
  Property k_W_mK, cp_J_kgK;           // thermal (k_W_mK is also in em.therm unless it is a law)
  Property alpha_1_K;                  // mechanical expansion coefficient per kelvin
  Property E_Pa, nu;                   // mechanical (also in em.mech unless one is a law)
  bool has_fluid = false, has_electrolyte = false, has_electric = false, has_species = false;
  json species;                        // the species block as written
};

// A domain: a named list of regions.  A disabled region is dropped from `regions` and listed in `disabled`.
struct Domain {
  std::vector<std::string> regions;
  std::vector<std::string> disabled;
};

// A boundary selector (design note 5.2), not resolved: resolution needs the geometry.
struct Selector {
  std::string kind;                    // of | between | curve | group | union | difference | ref
  std::string name;                    // of: the domain; curve: the curve; group: the group; ref: the boundary
  std::vector<std::string> between;
  std::vector<Selector> operands;      // union, difference
  std::map<std::string, std::vector<double>> where;   // one value: equality, two: a closed range
  double tolerance_m = 0.0;            // 0 when absent
  std::vector<std::string> except;
  json written;                        // the selector as written
};

struct ElectrodeReaction {
  std::string kind;                    // butler_volmer | linear | tafel_anodic | tafel_cathodic
  double i0_A_m2 = 0.0, E_eq_V = 0.0, alpha_a = 0.5, alpha_c = 0.5, slope_V_decade = 0.0;
};

struct Reactions {
  bool given = false;
  std::vector<std::string> species;
  json mechanism;                      // null when absent; the steps are the chemistry library's to read
  std::map<std::string, ElectrodeReaction> electrode;
};

struct MeshKeys {
  double size = 0.0, surface_size = 0.0;     // 0: automatic
  int geometry_order = 0;                    // 0: absent
  bool geometry_order_written = false;
  int refinements = 0;
  std::string exterior = "none";
  std::string file, info_file;
  int threads = 0;
};

// A number or a vector of data of a condition, a source or a coupling term.  A field expression key has one compiled
// expression per component (a numeric component is compiled too); every other key has its evaluated constants.
//
// The variables of a field expression are the coordinates of the symmetry (x y | r z | x y z) and, when a transient
// step of the study solves the instance that owns the datum, the time t after them.  `variables` is that list: the
// layout of the point that eval takes.
struct Datum {
  json written;
  bool field_expression = false;
  std::vector<std::string> variables;  // field expression: the names, in the order of the point
  std::vector<FieldExpr> exprs;
  std::vector<double> values;
  // every component at the point, which has one value per entry of `variables`
  std::vector<double> eval(const double* point) const {
    if (!field_expression) return values;
    std::vector<double> out;
    out.reserve(exprs.size());
    for (const FieldExpr& e : exprs) out.push_back(e.eval(point));
    return out;
  }
};

struct OptionValue {
  json value;                          // a string, an integer or a double
  bool written = false;                // false: the registry's default
};

struct Condition {
  std::string boundary;                // the key: a boundary name or "remaining"
  std::string kind;
  std::map<std::string, Datum> data;   // by key as written (current_A_m stays current_A_m)
  std::vector<std::string> reactions;  // electrode: the names in reactions.electrode
  json written;
};

struct SourceTerm {
  std::string kind;
  std::string in;
  std::map<std::string, Datum> data;
  json written;
};

struct Instance {
  std::string id, type;
  std::vector<std::string> domains;
  bool domains_written = false;
  bool in_transient_step = false;      // a transient step of the study solves it: t is a variable of its field expressions
  std::map<std::string, OptionValue> options;   // every option of the type that has a default, and every one written
  std::map<std::string, Condition> conditions;  // key: a boundary name or "remaining"
  std::vector<SourceTerm> sources;
  json initial;                        // as written, null when absent
  std::string notes;
};

struct Coupling {
  int index = -1;                      // the position of the entry in "couplings" as written
  std::string term, from, to, in, on;
  std::map<std::string, Datum> data;
};

struct CouplingSettings {
  bool given = false;
  std::string strategy;
  double tolerance = 0.0;
  int max_iterations = 0;
};

struct Continuation {
  bool given = false;
  std::string scale, ramp;
  std::vector<double> values;          // the factors of a scale, or the values of a ramp, evaluated
};

struct StudyStep {
  int index = -1;                      // the position of the step in "study" as written; -1: the implicit default step
  std::string id, type;                // stationary | transient
  std::vector<std::string> physics;
  bool implicit_default = false;       // the step of a document without "study"
  std::string initial_from;
  std::map<std::string, double> given;
  CouplingSettings coupling;
  Continuation continuation;
  std::map<std::string, std::map<std::string, json>> solvers;   // per instance id: setting -> value as written
  std::map<std::string, double> time;  // t_end_s, rtol, first_step_s, order
  std::vector<double> save_times_s;
  int save_n = 0;
};

struct Output {
  std::string name, kind, of, on, in, physics, component, expr;
  std::vector<double> at;
  std::vector<std::string> depends_on;  // expression outputs: the outputs it reads, by name
  json written;
};

struct ResolvedModel {
  std::string schema, name, description, notes;
  std::string symmetry = "axisymmetric";
  ParamMap parameters;
  std::vector<spec::Region> regions;               // enabled, in painting order
  std::vector<std::string> disabled_regions;
  std::vector<spec::Curve> curves;                 // markers: no material, no thickness
  std::map<std::string, MaterialProps> materials;  // the built-in ones, overridden by the document's
  std::map<std::string, Domain> domains;
  std::map<std::string, Selector> boundaries;
  Reactions reactions;
  MeshKeys mesh;
  std::map<std::string, Instance> instances;
  std::vector<StudyStep> study;
  std::vector<Coupling> couplings;
  std::map<std::string, Output> outputs;
  std::vector<std::string> output_order;           // every output; an expression output after every output it reads
};

}  // namespace axmp::model

namespace axmp::spec {

// A document is a model document when it is an object with the key "physics" (whatever its value).
bool is_model_document(const json& doc);

// Every finding of the grammar of the document, section by section, in `out`; the model holds what could be read.
model::ResolvedModel parse_model_collect(const json& doc, model::Findings& out);

// The model of a document without a grammar error; the first error (an axmp::Error) otherwise.
model::ResolvedModel parse_model(const json& doc);

// The operations of this version that take a model document, in the order in which every refusal of a model document
// lists them under details.serves_model_documents (the command names: "validate", "resolve", "explain", "patch", "boundaries", "mesh").
std::vector<std::string> serves_model_documents();

// Refuses a model document at an operation that works on electromagnetic designs only: an axmp::Error with the code
// model.em_only_operation, at /physics, whose message and details.operation name `operation`.  Returns, and looks at
// nothing else, when the document has no key "physics".  Every electromagnetic-only entry point calls it before it
// parses; the guard at the top of parse_design is the backstop for one that does not (details.operation is null there).
void require_em_document(const json& doc, const std::string& operation);

// Refuses a model document at an operation that will take one in a later version and does not in this one (solving): an
// axmp::Error with the code route.not_implemented, at /physics, naming `operation`.  Returns when the
// document has no key "physics".
void refuse_unserved_model_document(const json& doc, const std::string& operation);

// The resolved geometry of a model document in the shape of resolved_geometry (spec.hpp): its top-level keys, regions,
// curves, disabled regions and materials (their electromagnetic kind and constants); "mesh" carries the keys of a model
// document (size, surface_size, geometry_order, refinements, exterior).
json resolved_model_geometry(const model::ResolvedModel& m);

// Refuses a JSON text in which one object has a key twice: an axmp::Error with the code json.duplicate_key at the
// RFC 6901 pointer of the second occurrence ('~' written ~0, '/' written ~1).  `origin` names the text in the message
// (the file path).  Returns when no object repeats a key.  The text must be valid JSON (it was parsed before).
void scan_duplicate_keys(const std::string& text, const std::string& origin);

namespace detail {
// One material of a model document: its own key check, the new data, the filtered copy for the unchanged parse_material
// (design note 6.2).  Throws the first axmp::Error of the material, with pointers below node.path.
model::MaterialProps parse_model_material(const std::string& name, const Node& node);
}  // namespace detail

}  // namespace axmp::spec
