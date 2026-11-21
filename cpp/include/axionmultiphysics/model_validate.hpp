// Validation of a model document (beta): the grammar of parse_model_collect, then the checks that need the route table
// (physics_registry.hpp) and the coupling graph (model_graph.hpp), into one list of findings of four kinds (design note
// 14.2): errors (the document is wrong), unavailable (the document is fine and this version or build cannot solve it),
// deferred (checks that need the geometry kernel or an imported mesh file, not run here: the resolution of the named
// boundaries, the partition rule of each instance, the order and the element types of a mesh file; each is reported
// with the code of the check that runs later) and warnings.
//
// The route-level checks run for every group of instances whose own grammar is clean: an instance with a grammar
// finding under /physics/<id>, or at an end of an entry of "couplings" that has one (and every instance of its group),
// gets no route-level finding, so that one defect gives one finding.  Grammar findings come first, in the order of parse_model_collect; route-level findings follow.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_graph.hpp"
#include "axionmultiphysics/model_selection.hpp"
#include "axionmultiphysics/physics_registry.hpp"
#include "axionmultiphysics/spec_model.hpp"

namespace axmp::model {

// A group of instances of one study step and the route row it takes.  A group is a set of instances connected by the
// coupling terms of the step (property edges do not join groups); an instance with no term is a group of one.
struct RouteUse {
  std::vector<std::string> physics;  // the instance ids of the group, in the order of the step's physics list
  std::string route;                 // the row's name; "" when no row matches
};

struct StepReport {
  std::string id;
  CouplingGraph graph;
  std::vector<RouteUse> routes;      // one per group, in the order of each group's first instance in the step
  std::string strategy;              // the coupling strategy of the step: the one written, else "one_way" when every
                                     // component of the graph has one member, else the default strategy of the route
                                     // of the two-way group; "" when the step needs one and has none
                                     // (study.two_way_needs_strategy)
};

struct ValidationReport {
  std::vector<Error> errors, unavailable, deferred, warnings;   // each in the order found
  ResolvedModel model;                                          // what could be read
  std::map<std::string, std::vector<std::string>> regions;      // per instance id: its regions, in the order of the
                                                                // document's regions (the default domain of 3.6 when
                                                                // "domains" is not written)
  std::vector<StepReport> steps;                                // one per study step that parsed, in study order
  int geometry_order = 0;                                       // the geometry order of the master mesh: written, else
                                                                // the default of 3.3; 0 with mesh.file (read from the
                                                                // file)
  bool ok() const { return errors.empty(); }
};

// What a validation may ask of the installation.  The defaults ask nothing, so that explain, patch and the unit tests do
// not depend on the environment; the entry points that serve a user (axmp validate, boundaries and mesh, and their Python
// counterparts) turn the probe on.
struct ValidateOptions {
  bool probe_mesher = false;   // run the mesher's probe (mesher_status, model_mesh.hpp) for a document that needs the mesher
  std::string mesher;          // a path that the probe tries first; "": its search
};

// Validates a model document: parse_model_collect, then the route-level checks.  Never throws for a defect of the
// document; every defect is a finding.  With `probe_mesher`, a document that has an instance with a geometry and no imported
// mesh gets the unavailable finding build.requires_mesher, first in `unavailable`, when the mesher does not run.
ValidationReport validate_model(const nlohmann::json& doc, const ValidateOptions& options = {});

// The full check, run on the resolved boundaries (the table and resolution of the mesher's resolve_only run) before a mesh
// is made or a model is solved (design note 5.5): the empty selections, the partition rule of every instance that has a
// geometry (uncovered, covered twice, interior, outside its domain), the partial matches (warnings), and the outputs on
// resolved boundaries (an output outside its physics' domain, a force on an interior section, a force with the Laplacian
// viscous form on a boundary that carries a condition other than a wall at rest).  Its errors and warnings are added to the
// report, and the deferred findings it ran (the resolution of the boundaries, the partition rule, the force rule on a
// boundary that is not a key) are removed from it, so that `deferred` is empty for a document the mesher made.
// `document` is the file name the hints name; "" writes DOCUMENT.
void check_boundaries(ValidationReport& report, const selection::Table& table, const selection::Resolution& resolution,
                      const std::string& document = "");

// The report that `axmp validate` prints for a model document and _core.validate returns: "ok", "name", "parameters"
// (evaluated) and the four lists "errors", "unavailable", "deferred", "warnings", each finding being the command line's
// error JSON without "ok".  With an error, "ok" is false and the first error's fields are also at the top level; with
// `strict` and no error, the first unavailable finding is treated the same way.
nlohmann::json report_json(const ValidationReport& report, bool strict);

// The codes of the model path, sorted by code: a list of {"code", "kinds", "summary", "status"}.  "kinds" lists the
// finding kinds the code is raised as (error, unavailable, deferred, warning); "status" is "raised" for a code that this
// version raises and "reserved" for one that the route table names for a later version.  The reference page
// docs/reference/error_codes.md is generated from it.
nlohmann::json error_codes();

}  // namespace axmp::model
