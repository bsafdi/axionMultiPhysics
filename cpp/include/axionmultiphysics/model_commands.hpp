// The commands of a model document that need the geometry (beta): `axmp boundaries`, `axmp mesh`, and the Python objects
// that return what they print (interface design note 5.4, 5.5, 5.6).
//
// Every entry point validates the document with the mesher's probe, refuses what this version does not mesh (a 3D model,
// a model on an imported mesh), runs the mesher's resolve_only pass, then the full check on the resolved boundaries
// (check_boundaries, model_validate.hpp), and only then meshes.  The command line and the Python binding call the same
// functions, so that they cannot differ.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_mesh.hpp"
#include "axionmultiphysics/model_selection.hpp"
#include "axionmultiphysics/model_validate.hpp"

namespace axmp::model {

struct CommandOptions {
  std::string workdir;    // where the mesher's files are written; "": a temporary directory, removed before the call returns
  std::string mesher;     // a path that the probe and the runs try first
  int verbose = 0;
  bool mesh = false;      // boundaries: also mesh the model, to count the faces of every boundary
  std::string document;   // the file name the hints name ("" writes DOCUMENT)
};

// The table of the named boundaries of a resolved model document, as the JSON that `axmp boundaries --json` adds to the
// validation report: "symmetry", "boundaries" (one row per boundary name) and "edges" (every boundary entity, sorted by
// its point).  No master number and no geometry kernel tag appears in it.  `faces` (tag -> the number of boundary elements
// of the master mesh on that entity) adds "faces" to every row and every edge; null leaves it out.
nlohmann::json boundary_table(const ValidationReport& report, const selection::Table& table, const selection::Resolution& resolution,
                              const std::map<int, int>* faces = nullptr);

// `axmp boundaries`: the validation report with the table added.  `report["ok"]` is false when something refuses the
// command, and `error` is then the refusal (the first error of the report, or build.requires_mesher, a route refusal of
// a 3D or imported-mesh document, mesh.generation_failed); the first error's fields are also at the top level of the
// report, as report_json writes them.  An electromagnetic design is an axmp::Error with no code.
struct BoundariesResult {
  nlohmann::json report;
  std::optional<Error> error;
};
BoundariesResult boundaries_report(const nlohmann::json& document, const CommandOptions& options);

// The text of `axmp boundaries` for a report without an error: every boundary with its edges, measure, adjacent regions
// and what uses it, then the edges that no boundary names.  It prints no master number.
std::string boundaries_text(const nlohmann::json& report);

// `axmp mesh` and `_core.mesh` of a model document: the JSON that the command prints.  Every refusal is an axmp::Error.
// A document without a geometry gets {"ok", "name", "symmetry", "note"} and no file.
nlohmann::json mesh_command(const nlohmann::json& document, const CommandOptions& options);

// The meshed model of `axmp.mesh`, with what the Python object needs beside it: the table of the boundaries (without the
// face counts), the regions' areas, and the regions' names for `instance`.
struct MeshedBundle {
  MeshedModel model;
  ValidationReport report;
  nlohmann::json boundaries;   // the "boundaries" of boundary_table
  nlohmann::json regions;      // {region: {"area_m2"[, "volume_m3"]}}, enabled regions in document order
  std::string symmetry;
  // The instance mesh of the regions that `names` stand for (region or domain names): every enabled region when `names` is
  // empty and `all` is true.  An unknown name, a disabled region or an empty list is a std::invalid_argument.
  InstanceMesh instance(const std::vector<std::string>& names, bool all) const;
};
std::unique_ptr<MeshedBundle> mesh_bundle(const nlohmann::json& document, const CommandOptions& options);

}  // namespace axmp::model
