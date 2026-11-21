// The physics catalogue of the model document (beta): the physics types, the coupling terms and the route table that a
// model document is validated against, compiled into libaxionmultiphysics in every build.
//
// The registry is METADATA only.  It names no engine type and links no engine; validation reads it and never needs a
// factory, which is what makes a document portable between builds (the sequential build validates every document an MPI
// build does).  The factories that build an engine for a route are registered at run time by the programs that link
// the engine (register_route_factory); a route without one is unavailable, never an error in the document.
//
// It is built on first use, in a function-local static of registry(): a program that never asks for it (every
// electromagnetic command) never builds it.
//
// Conventions of the tables:
//   * A route row is for ONE geometry class ("planar", "axisymmetric", "3d"; none for a geometry-free row).
//   * cell_types are the instance types whose condition kinds and option values are cells of the row's matrix; on the
//     coupled rows (a flow and a heat instance) that is the heat instance only: the flow instance's cells belong to
//     the flow rows.
//   * condition_kinds[type] lists the kinds that are offered on the row for at least one value of the row's options;
//     not_offered[type][kind] holds why a kind is offered for none of them: "engine_refuses", or
//     "not_exposed: <reason>" with the roadmap item in roadmap[type][kind].  not_offered_with[type]["<option>=<value>"]
//     [kind] holds the refusals that depend on an option value (the heat kinds by advective_form): "engine_refuses",
//     or "layer_rule: <rule name>".
//   * condition_kinds, not_offered, not_offered_with, roadmap and options are given for the cell types;
//     material_blocks, uniform, uniform_when and one_expression for every type of the row.
//   * solvers lists the keys of study[].solvers, each with the type whose instance takes it; a type named in
//     solvers_of_own_row takes the settings of the single-type row that its instance's options select in the same
//     geometry (the flow instance of the one-way row: the Stokes or the steady Navier-Stokes row).
//   * options[type] has one rule per option that applies on the row.  A rule whose forced value is set admits that
//     value only; otherwise `offered` lists the values the row offers, out of the option's values.  An option named in
//     `when` selects the row (equation = stokes or navier_stokes) and is not a cell.
//   * Option values are written as text in the catalogue: an integer option has the values "1", "2", and a document
//     writes the number.  A numeric option (T_K) has no list of values.
//   * Data-key types: "number", "boolean", "names" (a list of names), "boundary" (a boundary name), "number_vector"
//     (a list of numbers), and the vector types of field expressions: "vector" has the geometry's components, (x, y)
//     planar, (r, z, theta) axisymmetric (the swirl component is always written), (x, y, z) in 3D; "meridional_vector"
//     has (x, y), (r, z) and (x, y, z).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace axmp::model {

// The classes of a cell of the matrix test: offered (validation and the engine accept); engine_refuses (the engine
// refuses it, and validation refuses it by name first); engine_cannot_express (the engine's interface has one slot for
// what the document gives region by region, or no slot for an option of which it has one value); not_exposed (the
// engine could, the document path does not expose it yet, for a stated reason); layer_rule (a rule of this layer,
// stricter than the engine on purpose).
enum class CellValue { offered, engine_refuses, engine_cannot_express, not_exposed, layer_rule };
const char* to_string(CellValue c);

// The factor on the facet angle theta_max in the bound of the through-flow check of `insulation`,
// int |u . n| dS / int |u| dS <= max(1e-10, kThetaMaxFactor * theta_max), theta_max the largest angle between the
// normals of two boundary faces that share a vertex.  Written here once; no check reads it before the heat routes are
// implemented.
constexpr double kThetaMaxFactor = 1.0;

// What a route factory receives and returns.  Declared here so that the factory signature is final; defined by the
// model path and the adapters.
class Solver;
struct ResolvedModel;
struct MeshedModel;
struct StudyStep;
using RouteFactory = std::unique_ptr<Solver> (*)(const ResolvedModel&, const MeshedModel&, const StudyStep&);

// Registers the factory of the route `route_name`.  Any name is accepted (the factories are keyed by name, not checked
// against the table) and registering the same pair again changes nothing; a second factory for one name replaces the
// first.
void register_route_factory(const std::string& route_name, RouteFactory make);
// The factory registered for `route_name`, or nullptr.
RouteFactory route_factory(const std::string& route_name);

// An output or viewer quantity of a physics type in one geometry class.
struct Quantity {
  std::string name, unit, convention;
};

// A data key of a condition kind, a source kind or a coupling term.
struct DataKey {
  std::string name, type, unit;
  bool field_expression = false;  // a field expression of the coordinates (and the time of a transient step)
  bool rampable = false;          // may be the target of a ramp continuation (study.continuation.ramp)
  bool required = true;
  std::string per_depth_twin;     // the spelling of this key in a planar model (per metre of depth), or ""
};

// A boundary-condition kind, or (the same shape) a source kind.
struct ConditionKind {
  std::string name;
  std::vector<DataKey> data;
  std::vector<std::string> aliases;  // vocabulary of other codes that names this kind; never a second spelling
};

struct OptionSpec {
  std::string name, type;                      // type: "enum", "integer", "number"
  std::vector<std::string> values;             // the allowed values (empty for "number")
  std::string default_value;                   // "" when the option has no default
  std::map<std::string, int> max_geometry_order;   // per value: the highest element-geometry order it allows
};

struct PhysicsType {
  std::string name, family, title, status;     // status: "beta" | "absent"
  std::vector<std::string> aliases;
  std::vector<std::string> provides;           // the fields it publishes
  std::map<std::string, std::vector<Quantity>> quantities;   // per geometry class: "planar", "axisymmetric", "3d"
  std::vector<ConditionKind> condition_kinds;
  std::vector<ConditionKind> source_kinds;
  std::vector<OptionSpec> options;
  std::vector<std::string> not_implemented;    // non-empty while the status is beta
  std::string absent_phrase;                   // an absent type's phrase in the "not implemented" inventory
  std::string docs, validation;
};

struct CouplingTermType {
  std::string name, title, status;             // status: "beta" | "absent"
  std::string support;                         // "domain" (selected with "in") | "boundary" (with "on")
  std::vector<std::string> producers, consumers;   // type names; "fluid.*" stands for every type of the family
  std::vector<std::string> reads;              // the fields of the producer it reads
  std::vector<DataKey> data;
};

struct Requirements {
  bool mesher = true;                          // false only for geometry-free rows
  bool mpi = false, fluids = false, engine_3d = false;
  int max_ranks = 0;                           // 0: any number of ranks
};

struct OptionRule {
  std::string name;
  std::vector<std::string> offered;            // the values the row offers (empty when forced, or for a number)
  std::string forced;                          // the only value the row admits, or ""
  std::string default_value;                   // the value used when the document gives none
};

struct SolverSetting {
  std::string name, type;                      // type: "number", "integer", "boolean"
  nlohmann::json default_value;
  std::string physics;                         // the type whose instance takes it: study[].solvers["<that id>"]
};

// A property that must be one constant over the instance only under a condition of the document: an option value of
// the instance and, when `output` is not empty, a declared output of that kind on it (the viscosity of a force output
// with the Laplacian viscous form).
struct ConditionalUniform {
  std::string property;                        // "fluid.mu_Pa_s"
  std::string option;                          // "<option>=<value>": "viscous_form=laplacian"
  std::string output;                          // the output kind that imposes it ("force"), or "" for any document
  std::string why;                             // "engine_refuses: <where>" | "layer_rule: <rule name>"
};

struct Route {
  std::string name;                            // the key of the factory
  std::vector<std::string> physics;            // the instance types of the row
  std::vector<std::string> cell_types;         // the types whose kinds and options are cells of the row
  std::vector<std::string> couplings;          // the coupling terms of the row
  std::vector<std::string> geometries;         // one geometry class; empty for a geometry-free row
  std::vector<std::string> study_types;
  std::vector<std::string> strategies;         // the coupling strategies offered (rows with two instances)
  std::map<std::string, std::string> when;     // option values that select this row: {"equation": "stokes"}
  std::map<std::string, std::vector<std::string>> when_sources;   // per type: the source kinds whose presence
                                                                  // selects this row (prescribed_advection); a row
                                                                  // without them is for instances without them
  std::string engine;
  std::string status;                          // "not_implemented" until the row has a factory and a validation record
  bool implemented = false;                    // the same fact as a boolean: true exactly when status is "implemented"
  std::string default_strategy;
  Requirements needs;                          // "requires" in the JSON (the word is reserved in C++20)
  std::map<std::string, std::vector<std::string>> condition_kinds;
  std::map<std::string, std::map<std::string, std::string>> not_offered;
  std::map<std::string, std::map<std::string, std::map<std::string, std::string>>> not_offered_with;
  std::map<std::string, std::map<std::string, std::string>> roadmap;
  std::string axisymmetric_integrals;          // "includes_2pi" | "per_radian" | ""
  std::string force_reaction;                  // "physical_reaction" | "reaction: walls at rest only with the
                                               // laplacian form" | "" (no force output on this row)
  std::map<std::string, std::vector<OptionRule>> options;
  std::map<std::string, std::vector<std::string>> material_blocks;   // per type: the blocks every region needs
  std::map<std::string, std::vector<std::string>> uniform;           // one constant over the instance
  std::map<std::string, std::vector<ConditionalUniform>> uniform_when;   // per type: one constant under a condition
  std::map<std::string, std::vector<std::string>> one_expression;    // one value or expression in every region
  std::vector<std::string> element_types;      // the element geometries the engine takes
  bool equal_domains = false;                  // every instance of the row has the same domain
  std::vector<std::string> rampable;           // data keys with a warm start, in this geometry's spelling
  std::vector<std::string> limits;             // human-readable, echoed into results
  std::vector<std::string> layer_rules;        // the rules of this layer that the row applies
  std::vector<SolverSetting> solvers;          // the keys of study[].solvers for the row, each for the instance of
                                               // one type (SolverSetting::physics), with their defaults
  std::vector<std::string> solvers_of_own_row; // the types whose instance takes the solver settings of the row of its
                                               // own type that its options select in this geometry, not this row's
};

// A rule of this layer that is stricter than the engine on purpose.
struct LayerRule {
  std::string name, description;
};

// True when `type` is one of the type names of `patterns` ("fluid.*" stands for every type of the family).
inline bool type_matches(const std::vector<std::string>& patterns, const std::string& type) {
  for (const std::string& p : patterns) {
    if (p == type) return true;
    if (p.size() > 1 && p.compare(p.size() - 2, 2, ".*") == 0 && type.compare(0, p.size() - 1, p, 0, p.size() - 1) == 0) return true;
  }
  return false;
}

struct Registry {
  std::vector<PhysicsType> types;
  std::vector<CouplingTermType> terms;
  std::vector<Route> routes;
  std::vector<std::string> reserved_ids;       // instance ids a model document may not use
  std::vector<LayerRule> layer_rules;
  const PhysicsType* type(const std::string& name) const;        // nullptr when absent
  const CouplingTermType* term(const std::string& name) const;
  const Route* route(const std::string& name) const;
};

// The registry, built on the first call; every call returns the same object.
const Registry& registry();

// The catalogue as JSON, schema "axionmultiphysics.physics_catalogue/1" (axmp physics --json).  `routes` adds the
// route table and the layer rules.  A non-empty `type` restricts the catalogue to that type, the rows whose physics
// contain it and the terms it produces or consumes; an unknown name, or an alias, is an axmp::Error whose hint names
// the types.  The JSON does not depend on the build.
nlohmann::json catalogue_json(bool routes, const std::string& type);

}  // namespace axmp::model
