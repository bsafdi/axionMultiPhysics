// The case list of the matrix test and a minimal model document per case.  Internal: not installed; compiled into the
// library and included by the tests as "../src/model_matrix.hpp".
//
// matrix_cases() is a function of the registry (axionmultiphysics/physics_registry.hpp): every condition kind, option
// value, uniform property and one-expression datum of every route row is one case with its class on that row, plus the
// cases that pin an engine fact (the element geometry, the element-geometry order, two MPI ranks, the sparse boundary
// numbering of a master mesh).  What the registry does not hold is written here: the engine column, i.e. which engine
// a case is handed to, on which of the matrix test's meshes, through which call, and the engine's refusal text or the
// member whose type the test asserts.  Those are the note's reading of the engines; the matrix test confirms them.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/physics_registry.hpp"

namespace axmp::model {

struct MatrixCase {
  std::string id;                   // stable: "<row or engine>/<physics>/<axis>=<value>@<meshes joined by +, or none>/<ranks>"
  std::string row;                  // the route row; empty for an engine fact that is not a row (the sparse mesh)
  std::string engine;               // StokesSolver, NavierStokesSolver, TransientNavierStokes, BoussinesqSolver,
                                    // transport::ScalarTransport, chem::CurrentDistribution
  std::string physics;              // the type the cell is about
  std::string axis;                 // condition_kind | option_value | uniform | one_expression | element_type |
                                    // geometry_order | ranks | study_type | mesh_numbering
  std::string value;                // "slip", "order=1", "density", "quadrilateral", "2", ...
  CellValue cls;                    // the class on this row
  std::vector<std::string> meshes;  // engine meshes it runs on: base, sparse, nodes, quad; empty: no engine column
  int ranks = 1;                    // 1 or 2
  std::string call;                 // constructor | add_bc | finalize | first_step | layout | solve
  std::string engine_text;          // a substring of the engine's refusal (engine_refuses)
  std::string code;                 // the code validation must give (every class but offered)
  std::string member;               // engine_cannot_express: the member the type assertion names, or the member
                                    // whose absence it asserts
  std::string reason, roadmap;      // not_exposed and layer_rule
};

// Every case, in a fixed order; ids are unique.
std::vector<MatrixCase> matrix_cases();

// A model document that writes the case's variation on the base document of its row, or null when the case has no
// validation column (an engine fact that is not a row, or a rank count).  Deterministic.  The base document of a row
// is a unit square (axisymmetric: with its left side on r = 0; 3D: a unit cube) of one region, whose material carries
// the row's blocks, with the boundaries left, right, bottom, top (3D: and front, back) selected by `where`, and the
// row's instances, couplings and study type.  The variation is the condition kind on one boundary (top; right in an
// axisymmetric model), the option value, two regions with different values of a uniform or one-expression datum,
// mesh.geometry_order 2 with the option that needs straight elements, or an imported mesh file.  Apart from the
// variation the document writes only what its row offers, with the advective form of the cell where the cell names
// one (a kind "<kind>@<form>" or an advective_form value), so that a refused cell is refused for its own reason only.
nlohmann::json minimal_document(const MatrixCase& c);

}  // namespace axmp::model
