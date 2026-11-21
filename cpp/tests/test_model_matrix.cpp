// The catalogue of physics types, terms and routes (cpp/include/axionmultiphysics/physics_registry.hpp) and the case
// list of the matrix test (cpp/src/model_matrix.hpp), against the design note's tables.
//
// Built in every build; no engine of the fluids library is linked (the engine half of the flow and heat rows is
// test_fluids_model_matrix.cpp).  What this file asserts:
//
//   1. the 19 route rows, their requirements, their layer rules and solver settings, and that the catalogue says
//      "not implemented" without naming when a row will be;
//   2. the physics types, the absent types, the terms, the reserved ids and the Q26 constant;
//   3. one class per cell: every condition kind, option value, uniform property and one-expression datum of every
//      2D row has exactly one case, and the cases agree with the rows;
//   4. the classes themselves, against the test author's own transcription of the note (below);
//   5. the meshes and rank counts of the cases that are not condition kinds (sparse, nodes, quadrilateral, 2 ranks);
//   6. a minimal document for every cell that has a validation column;
//   7. the engine column of the chemistry rows, against chem::CurrentDistribution built here on rectangles;
//   8. the validation column: validate_model (model_validate.hpp) accepts every offered cell's document and refuses every
//      other one with the case's code, and that code only.
//
// INDEPENDENT REFERENCE.  Everything in namespace `note` is transcribed by hand from the interface design note
// (the interface design note, revision 5).  It departs from the note where an engine disagrees, and each departure is
// marked at its table: the geometry-order rule of the 2D current-distribution rows is an engine cell, not a layer
// rule; the rank limit and the skew convection of the transient rows, and the form cells of the Boussinesq rows, follow
// the engines.  The section of the note is cited at every table.  Nothing in it is generated from the registry.  A cell
// whose class the note does not state is not in the transcription, with one exception, marked where it is made: the
// heat kinds of the two Boussinesq rows, which the note leaves to the matrix test and which are transcribed from the
// engine.  The engine half corrects the table where an engine disagrees, on both sides.
//
// Codes the note gives are copied from it.  The note gives no text for a message here: this file asserts codes and
// classes, not message text, except the engine refusals that the note quotes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/chem/current_distribution.hpp"
#include "axionmultiphysics/model_validate.hpp"
#include "axionmultiphysics/physics_registry.hpp"
#include "../src/model_matrix.hpp"

using nlohmann::json;
using axmp::model::CellValue;
using axmp::model::MatrixCase;

static int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

namespace {

using Strs = std::vector<std::string>;

// ================================================================================================ the transcription
namespace note {

// ---- the route rows.  Section 8.5 (the table of 13 lines; section 8.1 makes a row one geometry class, so the six
// lines that name two geometry classes are two rows each: 13 + 6 = 19).  The row names are this test's choice and
// are the specification: <what>_<geometry class>, geometry class in {planar, axisymmetric, 3d}.
struct Row {
  const char* name;
  const char* geometry;         // "" for the 0D reactor
  Strs physics;                 // 8.5 "physics" column
  Strs cell_types;              // the types whose condition kinds and options are cells of this row
  Strs couplings;               // 8.5 "couplings"
  const char* study;            // 8.5 "study"
  const char* engine;           // 8.5 "engine": exact, or a substring when marked
  bool engine_is_substring;
  bool fluids;                  // 8.5 "needs" / 16: the fluids library
  int max_ranks;                // 8.5 "needs": "any ranks" 0, "one rank" 1, "one process" 1
  bool mesher;                  // 8.5 head: "Every row with a geometry also needs the mesher"; the reactor has none
};
const Strs F = {"fluid.incompressible"}, H = {"heat.transfer"}, FH = {"fluid.incompressible", "heat.transfer"},
           CD = {"electrochem.current_distribution"}, CR = {"chem.reactor"};
const std::vector<Row> rows = {
    {"stokes_planar", "planar", F, F, {}, "stationary", "StokesSolver", false, true, 0, true},
    {"stokes_axisymmetric", "axisymmetric", F, F, {}, "stationary", "StokesSolver", false, true, 0, true},
    {"navier_stokes_steady_planar", "planar", F, F, {}, "stationary", "NavierStokesSolver", false, true, 0, true},
    {"navier_stokes_steady_axisymmetric", "axisymmetric", F, F, {}, "stationary", "NavierStokesSolver", false, true,
     0, true},
    // 8.5 row 5 gives the transient rows "one rank".  The engine has no such limit: TransientNavierStokes takes a mesh
    // partitioned over two ranks, projects the initial velocity and steps, and its velocity is the one-rank velocity
    // (the engine half compares the two).  The one-rank check of NavierStokesSolver, "needs one MPI rank (the free-DOF
    // state is rank-local)", is in its free-DOF functions (state, residual, jacobian_action, jacobian, dual_norm,
    // full_residual), none of which a time step calls.  So the rows take any number of ranks.
    {"navier_stokes_transient_planar", "planar", F, F, {}, "transient", "TransientNavierStokes", false, true, 0, true},
    {"navier_stokes_transient_axisymmetric", "axisymmetric", F, F, {}, "transient", "TransientNavierStokes", false,
     true, 0, true},
    {"stokes_3d", "3d", F, F, {}, "stationary", "Stokes", true, true, 0, true},
    {"heat_planar", "planar", H, H, {}, "stationary", "transport::ScalarTransport", false, true, 1, true},
    {"heat_axisymmetric", "axisymmetric", H, H, {}, "stationary", "transport::ScalarTransport", false, true, 1, true},
    {"heat_prescribed_advection_planar", "planar", H, H, {}, "stationary", "ScalarTransport", true, true, 1, true},
    {"heat_prescribed_advection_axisymmetric", "axisymmetric", H, H, {}, "stationary", "ScalarTransport", true, true,
     1, true},
    {"flow_then_heat_planar", "planar", FH, H, {"advection"}, "stationary", "ScalarTransport", true, true, 1, true},
    {"flow_then_heat_axisymmetric", "axisymmetric", FH, H, {"advection"}, "stationary", "ScalarTransport", true, true,
     1, true},
    {"boussinesq_planar", "planar", FH, H, {"advection", "buoyancy"}, "stationary", "BoussinesqSolver", false, true, 1,
     true},
    {"boussinesq_axisymmetric", "axisymmetric", FH, H, {"advection", "buoyancy"}, "stationary", "BoussinesqSolver", false,
     true, 1, true},
    {"current_distribution_planar", "planar", CD, CD, {}, "stationary", "chem::CurrentDistribution", false, false, 1,
     true},
    {"current_distribution_axisymmetric", "axisymmetric", CD, CD, {}, "stationary", "chem::CurrentDistribution", false,
     false, 1, true},
    {"current_distribution_3d", "3d", CD, CD, {}, "stationary", "chem::CurrentDistribution3D", false, false, 0, true},
    {"chem_reactor", "", CR, CR, {}, "transient", "", true, false, 0, false},
};

// ---- 8.3, the flow table.  Columns: Stokes planar, Stokes axisymmetric, Navier-Stokes planar, Navier-Stokes
// axisymmetric, 3D Stokes.  y = yes (offered); n = no (the engine refuses it: engine_refuses); l = later (the engine
// could, the document path does not expose it yet: not_exposed).  The `axis` line of the note is "never written" and is
// not a cell.  `slip` and `symmetry` share a line of the note.
struct FlowLine { const char* kind; const char* cells; };
const FlowLine flow_table[] = {
    {"wall", "yyyyy"},      {"inlet", "yyyyy"}, {"outlet", "yyyyy"}, {"open", "yyyyy"},
    {"traction", "yyyyy"},  {"slip", "ynnny"},  {"symmetry", "ynnny"}, {"periodic", "lnnnl"},
};
const char* const flow_columns[5] = {"stokes_planar", "stokes_axisymmetric", "navier_stokes_steady_planar",
                                     "navier_stokes_steady_axisymmetric", "stokes_3d"};
// 8.5 row 5: the transient rows are "as Navier-Stokes" (the engine half checks them against
// TransientNavierStokes's constructor).
const char* const transient_rows[2] = {"navier_stokes_transient_planar", "navier_stokes_transient_axisymmetric"};
const int transient_column = 2;

// ---- 8.3, the heat table.  Columns: the forms `convective` and `conservative_divergence` (one column in the note),
// then `conservative_flux`.  y offered; n the engine refuses; l a rule of this layer (the note: "no, as a rule of this
// layer": `outflow` with `conservative_flux`, which the engine takes as Zero and only the Python binding refuses).
struct HeatLine { const char* kind; const char* cells; };
const HeatLine heat_table[] = {
    {"temperature", "yy"}, {"heat_flux", "yn"}, {"total_flux", "ny"},
    {"convection", "yy"},  {"insulation", "yy"}, {"outflow", "yl"},
};
// The rows on which the form is selectable (8.5 rows 7, 8 and 9: "kinds by advective_form"); the cell value is then
// "<kind>@<form>".  Row 7, the row without advection, is one of them.  Its "without advection the three forms coincide
// and the default is used" is about the operator, not about the kinds: the transport operator takes each form without
// a velocity (finalize asks for more only "with a callable velocity": "the conservative forms with a callable velocity
// need div b") and checks the flux kinds against the form in add_bc whether or not there is a velocity ("a total-flux
// datum needs the conservative_flux form"; "a diffusive-flux datum on a natural part is refused with
// conservative_flux").  On the Boussinesq rows (8.5 row 10, "advective_form forced to convective") the cell value is
// the kind alone and the column is the first.
const Strs form_rows = {"heat_planar", "heat_axisymmetric", "heat_prescribed_advection_planar",
                        "heat_prescribed_advection_axisymmetric", "flow_then_heat_planar", "flow_then_heat_axisymmetric"};
// The two Boussinesq rows: 8.5 row 10 says "the heat kinds it accepts are fixed by the matrix test", so the note states
// no class for them.  The engine fixes them: BoussinesqSolver builds its transport operator in the convective form
// (boussinesq.cpp: "to.form = transport::AdvectiveForm::Convective;") and hands it every condition of
// BoussinesqOptions::t_bcs, so the kinds are those of the first column.  temperature, heat_flux, convection, insulation
// and outflow are accepted (the engine half constructs and solves each of them on the cavity), and total_flux is
// refused ("a total-flux datum needs the conservative_flux form").
const Strs plain_heat_rows = {"boussinesq_planar", "boussinesq_axisymmetric"};
const Strs forms = {"convective", "conservative_divergence", "conservative_flux"};

// ---- 8.5 "blocks; uniform; other limits": uniform properties (the engine takes one number) and one-expression data
// (the engine takes one function of position), per row.  16 rule 2: both are class engine_cannot_express, with codes
// physics.property_not_uniform_over_domain and heat.data_not_regionwise.  Members are those 16 rule 2 names
// (`NavierStokesOptions::rho`, `BoussinesqOptions::cp`, `Velocity::scale`, `ScalarTransport::set_diffusivity`,
// `ScalarTransport::set_source`, and the two `BoussinesqOptions` members that hold a ScalarData).
struct Prop { const char* name; const char* member; };
struct RowProps { const char* row; std::vector<Prop> uniform, one_expression; };
const std::vector<RowProps> props = {
    {"navier_stokes_steady_planar", {{"density", "NavierStokesOptions::rho"}}, {}},
    {"navier_stokes_steady_axisymmetric", {{"density", "NavierStokesOptions::rho"}}, {}},
    {"navier_stokes_transient_planar", {{"density", ""}}, {}},
    {"navier_stokes_transient_axisymmetric", {{"density", ""}}, {}},
    {"heat_planar", {}, {{"thermal.k_W_mK", "ScalarTransport::set_diffusivity"}, {"heat_source", "ScalarTransport::set_source"}}},
    {"heat_axisymmetric", {}, {{"thermal.k_W_mK", "ScalarTransport::set_diffusivity"}, {"heat_source", "ScalarTransport::set_source"}}},
    {"heat_prescribed_advection_planar",
     {{"density", "Velocity::scale"}, {"thermal.cp_J_kgK", "Velocity::scale"}},
     {{"thermal.k_W_mK", "ScalarTransport::set_diffusivity"}, {"heat_source", "ScalarTransport::set_source"},
      {"prescribed_advection.velocity", ""}}},
    {"heat_prescribed_advection_axisymmetric",
     {{"density", "Velocity::scale"}, {"thermal.cp_J_kgK", "Velocity::scale"}},
     {{"thermal.k_W_mK", "ScalarTransport::set_diffusivity"}, {"heat_source", "ScalarTransport::set_source"},
      {"prescribed_advection.velocity", ""}}},
    {"flow_then_heat_planar",
     {{"density", "Velocity::scale"}, {"thermal.cp_J_kgK", "Velocity::scale"}},
     {{"thermal.k_W_mK", "ScalarTransport::set_diffusivity"}, {"heat_source", "ScalarTransport::set_source"}}},
    {"flow_then_heat_axisymmetric",
     {{"density", "Velocity::scale"}, {"thermal.cp_J_kgK", "Velocity::scale"}},
     {{"thermal.k_W_mK", "ScalarTransport::set_diffusivity"}, {"heat_source", "ScalarTransport::set_source"}}},
    {"boussinesq_planar",
     {{"density", "NavierStokesOptions::rho"}, {"thermal.cp_J_kgK", "BoussinesqOptions::cp"},
      {"fluid.beta_1_K", "BoussinesqOptions::beta_g"}},
     {{"thermal.k_W_mK", "BoussinesqOptions::k"}, {"heat_source", "BoussinesqOptions::heat_source"}}},
    {"boussinesq_axisymmetric",
     {{"density", "NavierStokesOptions::rho"}, {"thermal.cp_J_kgK", "BoussinesqOptions::cp"},
      {"fluid.beta_1_K", "BoussinesqOptions::beta_g"}},
     {{"thermal.k_W_mK", "BoussinesqOptions::k"}, {"heat_source", "BoussinesqOptions::heat_source"}}},
};

// ---- 16 rule 2, the six rules of this layer ("rules of this layer that are stricter than the engine on purpose"),
// and the rows each applies to (8.3, 8.5, 3.3).  A rule is a name listed in Route::layer_rules.
struct LayerRule { const char* name; Strs rows; };
const std::vector<LayerRule> layer_rules = {
    // 8.3: `insulation` on a boundary the advecting velocity crosses (heat.insulation_on_throughflow): every row with
    // advection (a prescribed velocity, a solved flow, the Boussinesq flow)
    {"insulation_on_throughflow",
     {"heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric", "flow_then_heat_planar",
      "flow_then_heat_axisymmetric", "boussinesq_planar", "boussinesq_axisymmetric"}},
    // 8.3: `outflow` with `conservative_flux`: the rows on which the form is selectable (cells below), the rows
    // without advection among them (the Python binding, which the rule mirrors, refuses the name whatever the velocity)
    {"outflow_with_conservative_flux",
     {"heat_planar", "heat_axisymmetric", "heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric",
      "flow_then_heat_planar", "flow_then_heat_axisymmetric"}},
    // 8.3 "Force outputs depend on the route": a force output with the Laplacian form on a Navier-Stokes row (walls at
    // rest only, output.force_needs_stress_form) and the uniform-viscosity requirement kept there
    {"force_with_laplacian_form",
     {"navier_stokes_steady_planar", "navier_stokes_steady_axisymmetric", "navier_stokes_transient_planar",
      "navier_stokes_transient_axisymmetric"}},
    // 3.3: the per-depth keys (current_A_m / P_W_m in a planar model, current_A / P_W otherwise): every row that has
    // such an input (a heat source, an electrode current)
    {"per_depth_keys",
     {"heat_planar", "heat_axisymmetric", "heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric",
      "flow_then_heat_planar", "flow_then_heat_axisymmetric", "boussinesq_planar", "boussinesq_axisymmetric",
      "current_distribution_planar", "current_distribution_axisymmetric", "current_distribution_3d"}},
    // 3.3: the geometry order may not exceed the field order.  On the two 2D current-distribution rows it is an
    // engine_refuses cell instead (the engine refuses curved edge nodes with P1); the flow rows have field
    // order 2 only.  Listed on every other row that has a geometry.
    {"geometry_order_exceeds_field_order",
     {"stokes_planar", "stokes_axisymmetric", "navier_stokes_steady_planar", "navier_stokes_steady_axisymmetric",
      "navier_stokes_transient_planar", "navier_stokes_transient_axisymmetric", "stokes_3d", "heat_planar",
      "heat_axisymmetric", "heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric",
      "flow_then_heat_planar", "flow_then_heat_axisymmetric", "boussinesq_planar", "boussinesq_axisymmetric",
      "current_distribution_3d"}},
    // 8.5 rows 8, 9, 10: uniform density and cp in a stagnant part of an advected instance
    {"uniform_capacity_in_stagnant_part",
     {"heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric", "flow_then_heat_planar",
      "flow_then_heat_axisymmetric", "boussinesq_planar", "boussinesq_axisymmetric"}},
};

// ---- the physics types and their condition kinds (8.2, 8.3, 4.4)
struct TypeRef { const char* name; const char* status; Strs kinds; };
const std::vector<TypeRef> types = {
    {"fluid.incompressible", "beta", {"wall", "inlet", "outlet", "open", "traction", "slip", "symmetry", "periodic"}},
    {"heat.transfer", "beta", {"temperature", "heat_flux", "total_flux", "convection", "insulation", "outflow"}},
    {"electrochem.current_distribution", "beta", {"electrode", "insulation", "normal_current"}},
    {"chem.reactor", "beta", {}},
    {"electrochem.porous_electrode", "beta", {}},  // name only
    {"current.conduction", "absent", {"potential", "ground", "insulation"}},   // 4.4: the busbar's conditions
    {"solid.linear", "absent", {"fixed", "free"}},                              // 4.4
    {"species.transport", "absent", {}},                                        // 8.2: name only
};

}  // namespace note

// ================================================================================================ helpers
const axmp::model::Registry& reg() { return axmp::model::registry(); }

std::string to_str(CellValue c) { return axmp::model::to_string(c); }

const axmp::model::Route* find_row(const std::string& name) {
  for (const auto& r : reg().routes)
    if (r.name == name) return &r;
  return nullptr;
}

const note::Row* find_note_row(const std::string& name) {
  for (const auto& r : note::rows)
    if (name == r.name) return &r;
  return nullptr;
}

bool has(const Strs& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

std::vector<const MatrixCase*> cases_of(const std::vector<MatrixCase>& all, const std::string& row, const std::string& axis,
                                        const std::string& value) {
  std::vector<const MatrixCase*> out;
  for (const auto& c : all)
    if (c.row == row && c.axis == axis && c.value == value) out.push_back(&c);
  return out;
}

bool is_form_row(const std::string& row) { return has(note::form_rows, row); }

const std::set<std::string> kCalls = {"constructor", "add_bc", "finalize", "first_step", "layout", "solve"};

// ================================================================================================ the transcription as cells
struct Ref {
  std::string row, axis, value;
  CellValue cls;
  std::string code;     // the code validation gives; "" for an offered cell
  std::string call;     // pinned when not empty (engine_refuses)
  std::string text;     // a substring of the engine's refusal, pinned when not empty (the note quotes it)
  std::string member;   // pinned when not empty (engine_cannot_express)
};

std::vector<Ref> reference_cells() {
  std::vector<Ref> out;
  using note::flow_table;
  // flow kinds (8.3)
  auto flow_cell = [&](const std::string& row, const note::FlowLine& l, char c) {
    Ref r;
    r.row = row; r.axis = "condition_kind"; r.value = l.kind;
    if (c == 'y') { r.cls = CellValue::offered; }
    else {
      r.cls = (c == 'n') ? CellValue::engine_refuses : CellValue::not_exposed;
      r.code = "fluid.condition_not_offered";
    }
    out.push_back(r);
  };
  for (const auto& l : flow_table) {
    for (int col = 0; col < 5; ++col) flow_cell(note::flow_columns[col], l, l.cells[col]);
    for (const char* t : note::transient_rows) flow_cell(t, l, l.cells[note::transient_column]);
  }
  // heat kinds (8.3)
  auto heat_cell = [&](const std::string& row, const std::string& value, char c, const std::string& kind) {
    Ref r;
    r.row = row; r.axis = "condition_kind"; r.value = value;
    if (c == 'y') r.cls = CellValue::offered;
    else {
      r.cls = (c == 'n') ? CellValue::engine_refuses : CellValue::layer_rule;
      r.code = "heat.condition_not_offered";
      // scalar_transport.cpp: the flux kinds by form are checked in add_bc ("a total-flux datum needs the
      // conservative_flux form").  BoussinesqSolver calls add_bc of its transport operator inside its own constructor
      // (boussinesq.cpp: "for (const auto& tb : opt.t_bcs) tr_->add_bc(tb.attrs, tb.kind, tb.data);"), so on its rows the
      // refusal comes out of the constructor.
      if (c == 'n') r.call = row.rfind("boussinesq_", 0) == 0 ? "constructor" : "add_bc";
    }
    (void)kind;
    out.push_back(r);
  };
  for (const auto& l : note::heat_table) {
    for (const auto& row : note::plain_heat_rows) heat_cell(row, l.kind, l.cells[0], l.kind);
    for (const auto& row : note::form_rows) {
      heat_cell(row, std::string(l.kind) + "@convective", l.cells[0], l.kind);
      heat_cell(row, std::string(l.kind) + "@conservative_divergence", l.cells[0], l.kind);
      heat_cell(row, std::string(l.kind) + "@conservative_flux", l.cells[1], l.kind);
    }
  }
  // properties (8.5, 16 rule 2)
  for (const auto& rp : note::props) {
    for (const auto& p : rp.uniform) {
      Ref r;
      r.row = rp.row; r.axis = "uniform"; r.value = p.name; r.cls = CellValue::engine_cannot_express;
      r.code = "physics.property_not_uniform_over_domain"; r.member = p.member;
      out.push_back(r);
    }
    for (const auto& p : rp.one_expression) {
      Ref r;
      r.row = rp.row; r.axis = "one_expression"; r.value = p.name; r.cls = CellValue::engine_cannot_express;
      r.code = (std::string(p.name) == "prescribed_advection.velocity") ? "heat.prescribed_advection_partial"
                                                                         : "heat.data_not_regionwise";
      r.member = p.member;
      out.push_back(r);
    }
  }
  // option values (8.3, 8.5)
  auto opt = [&](const std::string& row, const std::string& value, CellValue c, const std::string& code = "",
                 const std::string& call = "", const std::string& text = "") {
    Ref r;
    r.row = row; r.axis = "option_value"; r.value = value; r.cls = c; r.code = code; r.call = call; r.text = text;
    out.push_back(r);
  };
  for (const char* row : {"stokes_planar", "stokes_axisymmetric"}) {
    opt(row, "viscous_form=stress", CellValue::offered);
    opt(row, "viscous_form=laplacian", CellValue::offered);
    opt(row, "order=2", CellValue::offered);
  }
  for (const char* row : {"navier_stokes_steady_planar", "navier_stokes_steady_axisymmetric"}) {
    opt(row, "viscous_form=stress", CellValue::offered);
    opt(row, "viscous_form=laplacian", CellValue::offered);
    opt(row, "convection=standard", CellValue::offered);
    opt(row, "convection=skew", CellValue::offered);
    opt(row, "order=2", CellValue::offered);
  }
  // 8.5 row 5 makes the transient rows "as Navier-Stokes", and 8.3 gives convection the values standard and skew.  The
  // time stepper does not take the skew form: the constructor of TransientNavierStokes refuses it ("transient
  // Navier-Stokes: the skew convective form is not offered"), so the cell is an engine refusal on the transient rows.
  // The note names no code for a refused flow option; fluid.option_not_offered is this file's, after
  // heat.option_not_offered.
  for (const char* row : note::transient_rows) {
    opt(row, "viscous_form=stress", CellValue::offered);
    opt(row, "viscous_form=laplacian", CellValue::offered);
    opt(row, "convection=standard", CellValue::offered);
    opt(row, "convection=skew", CellValue::engine_refuses, "fluid.option_not_offered", "constructor",
        "the skew convective form is not offered");
    opt(row, "order=2", CellValue::offered);
  }
  const Strs heat_rows = {"heat_planar", "heat_axisymmetric", "heat_prescribed_advection_planar",
                          "heat_prescribed_advection_axisymmetric", "flow_then_heat_planar",
                          "flow_then_heat_axisymmetric"};
  for (const auto& row : heat_rows) {
    opt(row, "order=1", CellValue::offered);   // 8.3: "order (1 or 2; ... 2 only on the Boussinesq row)"
    opt(row, "order=2", CellValue::offered);
    opt(row, "stabilization=none", CellValue::offered);
  }
  // 8.5 row 7: "supg" refused where there is no velocity (heat.supg_needs_advection); 16 rule 2 lists the code.
  // scalar_transport.cpp checks "SUPG needs a velocity" in finalize.
  opt("heat_planar", "stabilization=supg", CellValue::engine_refuses, "heat.supg_needs_advection", "finalize",
      "SUPG needs a velocity");
  opt("heat_axisymmetric", "stabilization=supg", CellValue::engine_refuses, "heat.supg_needs_advection", "finalize",
      "SUPG needs a velocity");
  for (const char* row : {"heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric",
                          "flow_then_heat_planar", "flow_then_heat_axisymmetric", "boussinesq_planar",
                          "boussinesq_axisymmetric"})
    opt(row, "stabilization=supg", CellValue::offered);   // on straight elements (the base mesh); see the nodes cells
  for (const auto& row : note::form_rows)
    for (const auto& f : note::forms) opt(row, "advective_form=" + f, CellValue::offered);
  for (const char* row : {"boussinesq_planar", "boussinesq_axisymmetric"}) {
    // 8.3 and 8.5 row 10: the P1 temperature is refused by BoussinesqSolver's constructor
    // ("the temperature is P2 (the coupled solver refuses P1 T)"); the uncoupled heat rows take P1.
    opt(row, "order=1", CellValue::engine_refuses, "heat.option_not_offered", "constructor", "refuses P1 T");
    opt(row, "order=2", CellValue::offered);
    opt(row, "stabilization=none", CellValue::offered);
    // 8.5 row 10: "advective_form forced to convective (the engine's only energy form)".  The engine cannot be asked
    // for another: BoussinesqOptions has no member for the form, and the constructor of BoussinesqSolver sets the
    // convective form on its transport operator.  So the other two values are engine_cannot_express, with the code the
    // note gives a refused heat option of this row (heat.option_not_offered).
    opt(row, "advective_form=convective", CellValue::offered);
    for (const char* f : {"conservative_divergence", "conservative_flux"}) {
      Ref r;
      r.row = row; r.axis = "option_value"; r.value = std::string("advective_form=") + f;
      r.cls = CellValue::engine_cannot_express; r.code = "heat.option_not_offered";
      r.member = "BoussinesqOptions: no advective form";
      out.push_back(r);
    }
  }
  for (const char* row : {"current_distribution_planar", "current_distribution_axisymmetric"}) {
    opt(row, "kinetics=primary", CellValue::offered);
    opt(row, "kinetics=secondary", CellValue::offered);
    opt(row, "order=1", CellValue::offered);   // 8.3: order 1 or 2 (P1 on straight elements only: the geometry cell)
    opt(row, "order=2", CellValue::offered);
  }
  // current distribution kinds (8.3, last table)
  for (const char* row : {"current_distribution_planar", "current_distribution_axisymmetric"})
    for (const char* k : {"electrode", "insulation", "normal_current"}) {
      Ref r;
      r.row = row; r.axis = "condition_kind"; r.value = k; r.cls = CellValue::offered;
      out.push_back(r);
    }
  // "Refused: every electrode galvanostatic (electrochem.potential_level_free)".  current_distribution.cpp: the
  // engine refuses it in the lazy layout that solve runs ("no grounded electrode").
  for (const char* row : {"current_distribution_planar", "current_distribution_axisymmetric"}) {
    Ref r;
    r.row = row; r.axis = "condition_kind"; r.value = "electrode:every_galvanostatic"; r.cls = CellValue::engine_refuses;
    r.code = "electrochem.potential_level_free"; r.call = "layout"; r.text = "no grounded electrode";
    out.push_back(r);
    // Correction of 16 rule 2: on the two 2D current-distribution rows the geometry order above the field
    // order is an engine refusal; the engine's constructor refuses curved edge nodes with P1.
    Ref g;
    g.row = row; g.axis = "geometry_order"; g.value = "order=1"; g.cls = CellValue::engine_refuses;
    g.code = "model.geometry_order_exceeds_field_order"; g.call = "constructor";
    g.text = "curved edge nodes need order 2";
    out.push_back(g);
  }
  // element types (8.5 row 7; 8.5 "The two engine refusals that are not about data"; 16 rule 2)
  for (const auto& row : heat_rows) {
    Ref r;
    r.row = row; r.axis = "element_type"; r.value = "quadrilateral"; r.cls = CellValue::engine_refuses;
    r.code = "model.element_type_not_offered"; r.call = "constructor"; r.text = "only triangular meshes are offered";
    out.push_back(r);
  }
  for (const char* row : {"boussinesq_planar", "boussinesq_axisymmetric"}) {
    Ref r;
    r.row = row; r.axis = "element_type"; r.value = "quadrilateral"; r.cls = CellValue::engine_refuses;
    r.code = "model.element_type_not_offered"; r.call = "constructor"; r.text = "only triangular meshes are offered";
    out.push_back(r);
  }
  // "the flow engines accept quadrilaterals"
  for (const char* row : {"stokes_planar", "stokes_axisymmetric", "navier_stokes_steady_planar",
                          "navier_stokes_steady_axisymmetric", "navier_stokes_transient_planar",
                          "navier_stokes_transient_axisymmetric"}) {
    Ref r;
    r.row = row; r.axis = "element_type"; r.value = "quadrilateral"; r.cls = CellValue::offered;
    out.push_back(r);
  }
  // the nodes cell (16 rule 2, "the mesh with nodes"; 3.3): supg is refused by the constructor on a mesh with nodes,
  // on the heat rows with advection and the Boussinesq row
  for (const char* row : {"heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric",
                          "flow_then_heat_planar", "flow_then_heat_axisymmetric", "boussinesq_planar",
                          "boussinesq_axisymmetric"}) {
    Ref r;
    r.row = row; r.axis = "geometry_order"; r.value = "supg"; r.cls = CellValue::engine_refuses;
    r.code = "model.geometry_order_not_offered"; r.call = "constructor"; r.text = "isoparametric";
    out.push_back(r);
  }
  // study types (3.8: "no stationary route" for the reactor; "the default study is refused with the step to write").
  // Class layer_rule: no engine has a stationary reactor solve, so there is no engine to compare with, and the
  // refusal is this layer's decision not to adapt the default step to the route (3.8: "D11's default is not adapted";
  // "an end time has no sensible default").  It is not one of the six rules of 16 rule 2, which are about cells an
  // engine accepts.
  { Ref r; r.row = "chem_reactor"; r.axis = "study_type"; r.value = "stationary"; r.cls = CellValue::layer_rule;
    r.code = "study.type_not_offered"; out.push_back(r); }
  { Ref r; r.row = "chem_reactor"; r.axis = "study_type"; r.value = "transient"; r.cls = CellValue::offered;
    out.push_back(r); }
  return out;
}

// ================================================================================================ 1: rows
void check_rows_and_requirements() {
  CHECK(reg().routes.size() == 19, "the registry has %zu route rows, the note's table gives 13 lines = 19 rows",
        reg().routes.size());
  std::set<std::string> names;
  for (const auto& r : reg().routes) names.insert(r.name);
  CHECK(names.size() == reg().routes.size(), "route names are not unique");
  for (const auto& nr : note::rows) {
    const auto* r = find_row(nr.name);
    CHECK(r != nullptr, "row %s missing", nr.name);
    if (!r) continue;
    CHECK(r->physics == nr.physics, "%s: physics", nr.name);
    CHECK(r->cell_types == nr.cell_types, "%s: cell_types", nr.name);
    CHECK(r->couplings == nr.couplings, "%s: couplings", nr.name);
    const Strs geoms = std::string(nr.geometry).empty() ? Strs{} : Strs{nr.geometry};
    CHECK(r->geometries == geoms, "%s: geometries (one geometry class per row, 8.1)", nr.name);
    CHECK(r->study_types == Strs{nr.study}, "%s: study_types", nr.name);
    if (nr.engine_is_substring) CHECK(r->engine.find(nr.engine) != std::string::npos, "%s: engine '%s'", nr.name, r->engine.c_str());
    else CHECK(r->engine == nr.engine, "%s: engine '%s' != '%s'", nr.name, r->engine.c_str(), nr.engine);
    CHECK(r->status == "not_implemented", "%s: status '%s' (every row is not implemented in this version)", nr.name, r->status.c_str());
    CHECK(r->needs.fluids == nr.fluids, "%s: needs.fluids", nr.name);
    CHECK(r->needs.max_ranks == nr.max_ranks, "%s: needs.max_ranks %d != %d", nr.name, r->needs.max_ranks, nr.max_ranks);
    CHECK(r->needs.mesher == nr.mesher, "%s: needs.mesher", nr.name);
  }
  // 8.5 / 16: the 2D current distribution solves in every build (no MPI); the 3D one needs an MPI build
  CHECK(!find_row("current_distribution_planar")->needs.mpi && !find_row("current_distribution_axisymmetric")->needs.mpi,
        "2D current distribution must not need MPI");
  CHECK(find_row("current_distribution_3d")->needs.mpi, "3D current distribution needs MPI");
  CHECK(!find_row("chem_reactor")->needs.mesher, "chem.reactor has mesher = false (8.1: geometry-free routes)");
  // 8.5: option values that select a row
  CHECK(find_row("stokes_planar")->when == (std::map<std::string, std::string>{{"equation", "stokes"}}), "when: stokes planar");
  CHECK(find_row("stokes_3d")->when == (std::map<std::string, std::string>{{"equation", "stokes"}}), "when: stokes 3d");
  CHECK(find_row("navier_stokes_steady_planar")->when == (std::map<std::string, std::string>{{"equation", "navier_stokes"}}),
        "when: navier_stokes steady planar");
  CHECK(find_row("navier_stokes_transient_axisymmetric")->when ==
            (std::map<std::string, std::string>{{"equation", "navier_stokes"}}), "when: navier_stokes transient");
  // 8.5: strategies of the coupled rows
  for (const char* n : {"flow_then_heat_planar", "flow_then_heat_axisymmetric"}) {
    CHECK(find_row(n)->strategies == Strs{"one_way"} && find_row(n)->default_strategy == "one_way", "%s: one-way strategy", n);
    CHECK(find_row(n)->equal_domains, "%s: equal domains (8.5)", n);
  }
  for (const char* n : {"boussinesq_planar", "boussinesq_axisymmetric"}) {
    CHECK(find_row(n)->strategies == Strs{"monolithic"} && find_row(n)->default_strategy == "monolithic",
          "%s: monolithic strategy (8.5 row 10)", n);
    CHECK(find_row(n)->equal_domains, "%s: equal domains (8.5)", n);
  }
  CHECK(!find_row("heat_planar")->equal_domains && !find_row("stokes_planar")->equal_domains, "equal_domains is set on the coupled rows only");
  // 8.1 / 8.5 last column: force and axisymmetric-integral conventions (10.3)
  for (const char* n : {"stokes_planar", "stokes_axisymmetric", "stokes_3d"})
    CHECK(find_row(n)->force_reaction == "physical_reaction", "%s: force_reaction '%s'", n, find_row(n)->force_reaction.c_str());
  for (const char* n : {"navier_stokes_steady_planar", "navier_stokes_steady_axisymmetric", "navier_stokes_transient_planar",
                        "navier_stokes_transient_axisymmetric"})
    CHECK(find_row(n)->force_reaction == "reaction: walls at rest only with the laplacian form", "%s: force_reaction '%s'", n,
          find_row(n)->force_reaction.c_str());
  for (const char* n : {"heat_planar", "chem_reactor", "current_distribution_planar"})
    CHECK(find_row(n)->force_reaction.empty(), "%s: no force output on this row", n);
  for (const char* n : {"stokes_axisymmetric", "navier_stokes_steady_axisymmetric", "navier_stokes_transient_axisymmetric"})
    CHECK(find_row(n)->axisymmetric_integrals == "includes_2pi", "%s: integrals include 2 pi (8.5)", n);
  for (const char* n : {"heat_axisymmetric", "heat_prescribed_advection_axisymmetric", "current_distribution_axisymmetric"})
    CHECK(find_row(n)->axisymmetric_integrals == "per_radian", "%s: integrals per radian (8.5)", n);
  for (const char* n : {"stokes_planar", "heat_planar", "current_distribution_planar", "stokes_3d"})
    CHECK(find_row(n)->axisymmetric_integrals.empty(), "%s: no axisymmetric integrals", n);
  // 8.5 row 11: "rampable: one electrode's phi_s_V or current" (3.8), spelled per geometry class (3.3)
  CHECK((std::set<std::string>(find_row("current_distribution_planar")->rampable.begin(),
                               find_row("current_distribution_planar")->rampable.end()) ==
         (std::set<std::string>{"phi_s_V", "current_A_m"})), "rampable keys, planar");
  CHECK((std::set<std::string>(find_row("current_distribution_axisymmetric")->rampable.begin(),
                               find_row("current_distribution_axisymmetric")->rampable.end()) ==
         (std::set<std::string>{"phi_s_V", "current_A"})), "rampable keys, axisymmetric");
  CHECK(find_row("stokes_planar")->rampable.empty(), "no rampable key on a flow row");
  // 8.5 blocks
  for (const char* n : {"stokes_planar", "stokes_axisymmetric", "stokes_3d"}) {
    auto it = find_row(n)->material_blocks.find("fluid.incompressible");
    CHECK(it != find_row(n)->material_blocks.end() && it->second == Strs{"fluid"}, "%s: blocks: fluid", n);
  }
  for (const char* n : {"navier_stokes_steady_planar", "navier_stokes_steady_axisymmetric"}) {
    auto it = find_row(n)->material_blocks.find("fluid.incompressible");
    CHECK(it != find_row(n)->material_blocks.end() && has(it->second, "fluid") && has(it->second, "density"),
          "%s: blocks: fluid, density", n);
  }
  for (const char* n : {"heat_planar", "heat_axisymmetric"}) {
    auto it = find_row(n)->material_blocks.find("heat.transfer");
    CHECK(it != find_row(n)->material_blocks.end() && it->second == Strs{"thermal"}, "%s: blocks: thermal", n);
  }
  for (const char* n : {"current_distribution_planar", "current_distribution_axisymmetric", "current_distribution_3d"}) {
    auto it = find_row(n)->material_blocks.find("electrochem.current_distribution");
    CHECK(it != find_row(n)->material_blocks.end() && it->second == Strs{"electrolyte"}, "%s: blocks: electrolyte", n);
  }
  // 8.5: every row carries a human-readable limits list in the rows where the note states limits
  for (const char* n : {"heat_planar", "boussinesq_planar", "flow_then_heat_planar", "current_distribution_planar"})
    CHECK(!find_row(n)->limits.empty(), "%s: limits (echoed into results)", n);
}

// What the catalogue says of a row's implementation: the boolean "implemented" beside the status, and nothing about
// when.  No name of a development milestone (the letter U and a number, with an optional letter) appears in the
// catalogue's JSON, from which every user-facing form of the catalogue (axmp physics, axionmultiphysics.physics(), the
// reference page) is made.
void check_catalogue_says_not_implemented_yet() {
  const json cat = axmp::model::catalogue_json(true, "");
  CHECK(cat.contains("routes") && cat["routes"].size() == 19, "the catalogue with routes lists the 19 rows");
  for (const auto& r : cat.value("routes", json::array())) {
    const std::string n = r.value("name", "");
    CHECK(!r.contains("arrives_with"), "%s: the route names the milestone that implements it ('arrives_with')", n.c_str());
    CHECK(r.contains("implemented") && r["implemented"].is_boolean() && !r["implemented"].get<bool>(),
          "%s: \"implemented\": false (no route is implemented in this version)", n.c_str());
    CHECK(r.value("status", "") == "not_implemented", "%s: status", n.c_str());
  }
  const std::regex milestone("\\bU(1[0-2]|[1-9])[a-d]?\\b");
  for (const auto& [routes, type] : std::vector<std::pair<bool, std::string>>{{true, ""}, {false, ""}, {true, "heat.transfer"}}) {
    const std::string text = axmp::model::catalogue_json(routes, type).dump();
    std::smatch m;
    const bool found = std::regex_search(text, m, milestone);
    CHECK(!found, "the catalogue (routes %d, type '%s') names the milestone '%s'", (int)routes, type.c_str(),
          found ? m.str(0).c_str() : "");
  }
}

// 3.8, 6 item 7: solver settings per instance and their defaults
void check_solver_settings() {
  for (const char* n : {"navier_stokes_steady_planar", "navier_stokes_steady_axisymmetric"}) {
    const auto* r = find_row(n);
    std::map<std::string, json> s;
    for (const auto& k : r->solvers) s[k.name] = k.default_value;
    CHECK(s.count("newton_rtol") && s["newton_rtol"] == json(1e-12), "%s: newton_rtol default 1e-12", n);
    CHECK(s.count("n_picard") && s["n_picard"] == json(3), "%s: n_picard default 3", n);
    CHECK(s.count("line_search"), "%s: line_search", n);
    CHECK(s.count("residual_floor_factor") && s["residual_floor_factor"] == json(4),
          "%s: residual_floor_factor default 4", n);
  }
  for (const char* n : {"boussinesq_planar", "boussinesq_axisymmetric"}) {
    const auto* r = find_row(n);
    bool floor = false, rtol = false;
    for (const auto& k : r->solvers) {
      floor = floor || k.name == "residual_floor_factor";
      rtol = rtol || k.name == "newton_rtol";
    }
    CHECK(rtol, "%s: newton_rtol", n);
    CHECK(!floor, "%s: the Boussinesq solver ignores residual_floor_factor (plan section 6, item 7)", n);
  }
  CHECK(find_row("stokes_planar")->solvers.empty(), "a Stokes row has no nonlinear solver settings");
}

// ================================================================================================ 2: types, terms
void check_types() {
  const auto& R = reg();
  for (const auto& t : note::types) {
    const auto* p = R.type(t.name);
    CHECK(p != nullptr, "type %s missing", t.name);
    if (!p) continue;
    CHECK(p->status == t.status, "%s: status %s", t.name, p->status.c_str());
    CHECK(p->family == std::string(t.name).substr(0, std::string(t.name).find('.')), "%s: family '%s'", t.name,
          p->family.c_str());
    CHECK(!p->title.empty(), "%s: title", t.name);
    std::set<std::string> have;
    for (const auto& k : p->condition_kinds) have.insert(k.name);
    CHECK(have == std::set<std::string>(t.kinds.begin(), t.kinds.end()), "%s: condition kinds differ from the note", t.name);
  }
  CHECK(R.types.size() == note::types.size(), "the registry has %zu types, the note lists %zu (8.2 and 4.4)", R.types.size(),
        note::types.size());
  // 8.2: provides
  CHECK(has(R.type("fluid.incompressible")->provides, "u") && has(R.type("fluid.incompressible")->provides, "p"),
        "fluid provides u and p");
  CHECK(has(R.type("heat.transfer")->provides, "T"), "heat provides T");
  CHECK(has(R.type("electrochem.current_distribution")->provides, "phi"), "current distribution provides phi");
  // 8.1, 10.3: quantities per geometry class
  for (const char* t : {"fluid.incompressible", "heat.transfer", "electrochem.current_distribution"})
    for (const char* g : {"planar", "axisymmetric"}) {
      auto it = R.type(t)->quantities.find(g);
      CHECK(it != R.type(t)->quantities.end() && !it->second.empty(), "%s: quantities for %s", t, g);
    }
  CHECK(R.type("fluid.incompressible")->quantities.count("3d") == 1, "fluid: quantities for 3d");
  // 17: a beta type lists what is not implemented; 8.2: an absent type names its inventory phrase
  for (const auto& t : R.types) {
    if (t.status == "beta") CHECK(!t.not_implemented.empty(), "%s is beta and lists nothing as not implemented (17)", t.name.c_str());
    if (t.status == "absent") CHECK(!t.absent_phrase.empty(), "%s is absent and has no inventory phrase", t.name.c_str());
  }
  // 4.4: the phrases of the busbar's inventory
  CHECK(R.type("current.conduction")->absent_phrase == "physics type 'current.conduction'", "phrase of current.conduction: '%s'",
        R.type("current.conduction")->absent_phrase.c_str());
  CHECK(R.type("solid.linear")->absent_phrase == "'solid.linear' as a physics instance", "phrase of solid.linear: '%s'",
        R.type("solid.linear")->absent_phrase.c_str());
  // 4.4: what the busbar writes: the heat sink and thermal expansion terms' types publish fields and quantities
  CHECK(!R.type("current.conduction")->provides.empty() && !R.type("solid.linear")->provides.empty(),
        "absent types name the fields they publish (4.4)");
  CHECK(!R.type("current.conduction")->quantities.empty() && !R.type("solid.linear")->quantities.empty(),
        "absent types name the quantities the busbar's outputs use (4.4)");
  // 3.6: reserved ids
  const Strs reserved = {"em", "mechanics", "thermal", "electrostatics", "hts", "fullwave", "cavity", "mesh_motion"};
  CHECK(std::set<std::string>(R.reserved_ids.begin(), R.reserved_ids.end()) ==
            std::set<std::string>(reserved.begin(), reserved.end()),
        "reserved instance ids");
  // 8.1 / 3.3 data keys
  auto key = [&](const char* type, const char* kind, const char* name) -> const axmp::model::DataKey* {
    for (const auto& k : R.type(type)->condition_kinds)
      if (k.name == kind)
        for (const auto& d : k.data)
          if (d.name == name) return &d;
    return nullptr;
  };
  const auto* v = key("fluid.incompressible", "inlet", "velocity_m_s");
  CHECK(v && v->field_expression, "inlet velocity_m_s is a field expression (8.3)");
  const auto* p0 = key("fluid.incompressible", "outlet", "pressure_Pa");
  CHECK(p0 != nullptr, "outlet pressure_Pa");
  CHECK(key("fluid.incompressible", "traction", "traction_Pa") != nullptr, "traction_Pa");
  CHECK(key("fluid.incompressible", "wall", "velocity_m_s") && !key("fluid.incompressible", "wall", "velocity_m_s")->required,
        "wall velocity_m_s is optional (a moving wall)");
  for (const char* d : {"partner", "translation_m", "pressure_drop_Pa", "flow_rate"})
    CHECK(key("fluid.incompressible", "periodic", d) != nullptr, "periodic data key %s (8.3)", d);
  CHECK(key("heat.transfer", "temperature", "T_K") && key("heat.transfer", "temperature", "T_K")->field_expression, "T_K");
  CHECK(key("heat.transfer", "heat_flux", "q_W_m2") && key("heat.transfer", "total_flux", "q_W_m2"), "q_W_m2");
  CHECK(key("heat.transfer", "convection", "h_W_m2K") && key("heat.transfer", "convection", "T_ext_K"), "convection data");
  CHECK(key("electrochem.current_distribution", "normal_current", "i_A_m2") != nullptr, "i_A_m2");
  const auto* ca = key("electrochem.current_distribution", "electrode", "current_A");
  CHECK(ca && ca->per_depth_twin == "current_A_m", "electrode current_A has the per-depth twin current_A_m (3.3)");
  CHECK(ca && ca->rampable, "electrode current_A is rampable (3.8)");
  const auto* ps = key("electrochem.current_distribution", "electrode", "phi_s_V");
  CHECK(ps && ps->rampable, "electrode phi_s_V is rampable (3.8)");
  CHECK(key("electrochem.current_distribution", "electrode", "reactions") != nullptr, "electrode reactions");
  CHECK(key("electrochem.current_distribution", "electrode", "E_eq_V") != nullptr, "electrode E_eq_V (primary)");
  // aliases (8.1): COMSOL vocabulary is discoverable and is not a second spelling
  bool no_slip = false;
  for (const auto& k : R.type("fluid.incompressible")->condition_kinds)
    if (k.name == "wall") no_slip = has(k.aliases, "no_slip");
  CHECK(no_slip, "no_slip is an alias of wall");
  CHECK(has(R.type("fluid.incompressible")->aliases, "laminar_flow"), "laminar_flow is an alias of fluid.incompressible");
  // sources (8.3): the heat source with its per-depth twin, the prescribed advection, the body force of 3.6
  auto source = [&](const char* type, const char* kind) -> const axmp::model::ConditionKind* {
    for (const auto& k : R.type(type)->source_kinds)
      if (k.name == kind) return &k;
    return nullptr;
  };
  const auto* hs = source("heat.transfer", "heat_source");
  CHECK(hs != nullptr, "source kind heat_source");
  if (hs) {
    bool twin = false;
    for (const auto& d : hs->data)
      if (d.name == "P_W") twin = d.per_depth_twin == "P_W_m";
    CHECK(twin, "heat_source: P_W has the per-depth twin P_W_m");
  }
  const auto* pa = source("heat.transfer", "prescribed_advection");
  CHECK(pa != nullptr, "source kind prescribed_advection");
  CHECK(source("fluid.incompressible", "body_force") != nullptr, "source kind body_force (3.6)");
  // options (8.3), including max_geometry_order (3.3)
  auto option = [&](const char* type, const char* name) -> const axmp::model::OptionSpec* {
    for (const auto& o : R.type(type)->options)
      if (o.name == name) return &o;
    return nullptr;
  };
  const auto* stab = option("heat.transfer", "stabilization");
  CHECK(stab && stab->values == Strs({"none", "supg"}) && stab->default_value == "none", "heat stabilization: none, supg; default none");
  CHECK(stab && stab->max_geometry_order.count("supg") && stab->max_geometry_order.at("supg") == 1,
        "stabilization supg declares max_geometry_order 1 (3.3)");
  const auto* adv = option("heat.transfer", "advective_form");
  CHECK(adv && std::set<std::string>(adv->values.begin(), adv->values.end()) ==
                   std::set<std::string>(note::forms.begin(), note::forms.end()) && adv->default_value == "convective",
        "heat advective_form values and default");
  const auto* eq = option("fluid.incompressible", "equation");
  CHECK(eq && std::set<std::string>(eq->values.begin(), eq->values.end()) == (std::set<std::string>{"stokes", "navier_stokes"}),
        "fluid equation values");
  const auto* vf = option("fluid.incompressible", "viscous_form");
  CHECK(vf && vf->default_value == "stress" && std::set<std::string>(vf->values.begin(), vf->values.end()) ==
                                                   (std::set<std::string>{"stress", "laplacian"}),
        "fluid viscous_form: stress (default), laplacian (3.8 / Q19)");
  const auto* cv = option("fluid.incompressible", "convection");
  CHECK(cv && cv->default_value == "standard" && has(cv->values, "skew"), "fluid convection: standard (default), skew");
  CHECK(option("fluid.incompressible", "swirl") == nullptr, "there is no swirl option (8.3, Q23)");
  const auto* ho = option("heat.transfer", "order");
  CHECK(ho && ho->values == Strs({"1", "2"}) && ho->default_value == "2", "heat order: 1 or 2, default 2");
  const auto* fo = option("fluid.incompressible", "order");
  CHECK(fo && fo->values == Strs({"2"}), "fluid order: 2");
  const auto* co = option("electrochem.current_distribution", "order");
  CHECK(co && co->values == Strs({"1", "2"}), "current distribution order: 1 or 2");
  const auto* ck = option("electrochem.current_distribution", "kinetics");
  CHECK(ck && ck->values == Strs({"primary", "secondary"}), "kinetics: primary, secondary");
  const auto* ct = option("electrochem.current_distribution", "T_K");
  CHECK(ct && ct->values.empty(), "T_K is a number, not an enumeration");
}

void check_terms() {
  const auto& R = reg();
  const auto* adv = R.term("advection");
  CHECK(adv && adv->status == "beta" && adv->support == "domain", "advection: beta, domain support (8.4)");
  if (adv) {
    CHECK(adv->producers == Strs{"fluid.*"}, "advection producers: fluid.*");
    CHECK(has(adv->consumers, "heat.transfer") && has(adv->consumers, "species.transport"), "advection consumers (8.4)");
    CHECK(adv->data.empty(), "advection has no data keys");
  }
  const auto* bu = R.term("buoyancy");
  CHECK(bu && bu->status == "beta" && bu->support == "domain", "buoyancy: beta, domain support");
  if (bu) {
    CHECK(bu->producers == Strs{"heat.transfer"} && bu->consumers == Strs{"fluid.incompressible"}, "buoyancy: heat -> fluid");
    std::set<std::string> d;
    for (const auto& k : bu->data) d.insert(k.name);
    CHECK(d == (std::set<std::string>{"T_ref_K", "gravity_m_s2"}), "buoyancy data keys (8.4)");
  }
  const auto* jh = R.term("joule_heating");
  CHECK(jh && jh->status == "absent" && jh->support == "domain" && has(jh->producers, "current.conduction") &&
            has(jh->consumers, "heat.transfer"),
        "joule_heating: absent, domain, current.conduction -> heat.transfer");
  const auto* te = R.term("thermal_expansion");
  CHECK(te && te->status == "absent" && te->support == "domain" && has(te->producers, "heat.transfer") &&
            has(te->consumers, "solid.linear"),
        "thermal_expansion: absent, domain, heat.transfer -> solid.linear");
  if (te) {
    bool tref = false;
    for (const auto& k : te->data) tref = tref || k.name == "T_ref_K";
    CHECK(tref, "thermal_expansion data T_ref_K");
  }
  CHECK(R.term("no_such_term") == nullptr, "term() of an unknown name is null");
  // the Q26 constant is written once and not yet used
  static_assert(axmp::model::kThetaMaxFactor == 1.0, "Q26: the factor on theta_max is 1");
  CHECK(axmp::model::kThetaMaxFactor == 1.0, "Q26 factor");
}

// ================================================================================================ 3: one class per cell
void check_every_2d_cell_has_one_value() {
  const auto all = axmp::model::matrix_cases();
  // ids are unique and follow "<row or engine>/<physics>/<axis>=<value>@<meshes>/<ranks>"
  std::set<std::string> ids;
  for (const auto& c : all) {
    CHECK(ids.insert(c.id).second, "duplicate case id %s", c.id.c_str());
    const std::string head = (c.row.empty() ? c.engine : c.row) + "/" + c.physics + "/" + c.axis + "=" + c.value;
    CHECK(c.id.rfind(head, 0) == 0, "id '%s' does not start with '%s'", c.id.c_str(), head.c_str());
    CHECK(c.id.size() > 2 && c.id.compare(c.id.size() - 2, 2, "/" + std::to_string(c.ranks)) == 0,
          "id '%s' does not end with the rank count", c.id.c_str());
    CHECK(c.ranks == 1 || c.ranks == 2, "%s: ranks %d", c.id.c_str(), c.ranks);
    CHECK(c.cls == CellValue::offered || c.cls == CellValue::engine_refuses || c.cls == CellValue::engine_cannot_express ||
              c.cls == CellValue::not_exposed || c.cls == CellValue::layer_rule,
          "%s: class", c.id.c_str());
  }
  for (const auto& r : reg().routes) {
    if (r.geometries == Strs{"3d"} || r.geometries.empty()) continue;  // 3D rows are confirmed when their engines are; the reactor has no cells here
    for (const auto& t : r.cell_types) {
      const auto* ty = reg().type(t);
      CHECK(ty != nullptr, "%s: cell type %s is not in the registry", r.name.c_str(), t.c_str());
      if (!ty) continue;
      // forms: the value of a heat kind carries the form on the rows where the form is selectable
      Strs forms;
      if (auto fit = r.options.find(t); fit != r.options.end())
        for (const auto& rule : fit->second)
          if (rule.name == "advective_form" && rule.forced.empty()) forms = rule.offered;
      for (const auto& k : ty->condition_kinds) {
        if (forms.empty()) {
          CHECK(cases_of(all, r.name, "condition_kind", k.name).size() == 1, "%s: condition kind %s has %zu cases, want 1",
                r.name.c_str(), k.name.c_str(), cases_of(all, r.name, "condition_kind", k.name).size());
        } else {
          for (const auto& f : forms)
            CHECK(cases_of(all, r.name, "condition_kind", k.name + "@" + f).size() == 1,
                  "%s: condition kind %s@%s has %zu cases, want 1", r.name.c_str(), k.name.c_str(), f.c_str(),
                  cases_of(all, r.name, "condition_kind", k.name + "@" + f).size());
        }
      }
      // option values: every option rule of the row that does not select the row gives one cell per value of the
      // option; a cell is offered exactly when the value is in the rule's offered list, or is the forced value
      auto it = r.options.find(t);
      if (it != r.options.end())
        for (const auto& rule : it->second) {
          if (r.when.count(rule.name)) continue;
          const axmp::model::OptionSpec* spec = nullptr;
          for (const auto& o : ty->options)
            if (o.name == rule.name) spec = &o;
          CHECK(spec != nullptr, "%s: option rule %s is not an option of %s", r.name.c_str(), rule.name.c_str(), t.c_str());
          if (!spec) continue;
          if (!rule.forced.empty()) {
            // a forced option: one cell per value of the option; the forced value is offered and every other value
            // is one the engine cannot express
            for (const auto& v : spec->values) {
              const auto cs = cases_of(all, r.name, "option_value", rule.name + "=" + v);
              CHECK(cs.size() == 1, "%s: forced option %s=%s has %zu cases, want 1", r.name.c_str(), rule.name.c_str(), v.c_str(), cs.size());
              if (cs.size() == 1)
                CHECK(cs[0]->cls == (v == rule.forced ? CellValue::offered : CellValue::engine_cannot_express),
                      "%s: forced option %s=%s: class %s", r.name.c_str(), rule.name.c_str(), v.c_str(), to_str(cs[0]->cls).c_str());
            }
            continue;
          }
          for (const auto& v : spec->values) {
            const auto cs = cases_of(all, r.name, "option_value", rule.name + "=" + v);
            CHECK(cs.size() == 1, "%s: option %s=%s has %zu cases, want 1", r.name.c_str(), rule.name.c_str(), v.c_str(), cs.size());
            if (cs.size() == 1)
              CHECK((cs[0]->cls == CellValue::offered) == has(rule.offered, v),
                    "%s: option %s=%s: class %s but the row's offered list says %s", r.name.c_str(), rule.name.c_str(), v.c_str(),
                    to_str(cs[0]->cls).c_str(), has(rule.offered, v) ? "offered" : "not offered");
          }
        }
    }
    // uniform and one-expression data: exactly one cell each, class engine_cannot_express
    std::set<std::string> uni, one;
    for (const auto& kv : r.uniform) uni.insert(kv.second.begin(), kv.second.end());
    for (const auto& kv : r.one_expression) one.insert(kv.second.begin(), kv.second.end());
    for (const auto& p : uni) {
      const auto cs = cases_of(all, r.name, "uniform", p);
      CHECK(cs.size() == 1, "%s: uniform %s has %zu cases", r.name.c_str(), p.c_str(), cs.size());
      if (cs.size() == 1) CHECK(cs[0]->cls == CellValue::engine_cannot_express, "%s: uniform %s class", r.name.c_str(), p.c_str());
    }
    for (const auto& p : one) {
      const auto cs = cases_of(all, r.name, "one_expression", p);
      CHECK(cs.size() == 1, "%s: one-expression %s has %zu cases", r.name.c_str(), p.c_str(), cs.size());
      if (cs.size() == 1) CHECK(cs[0]->cls == CellValue::engine_cannot_express, "%s: one-expression %s class", r.name.c_str(), p.c_str());
    }
    // the converse: every case of the row names something the row declares
    for (const auto& c : all) {
      if (c.row != r.name) continue;
      CHECK(has(r.cell_types, c.physics) || c.axis == "uniform" || c.axis == "one_expression" || c.axis == "ranks",
            "%s: case %s is about a type the row has no cells for", r.name.c_str(), c.id.c_str());
      if (c.axis == "uniform") {
        bool listed = false;
        for (const auto& kv : r.uniform) listed = listed || has(kv.second, c.value);
        CHECK(listed, "%s: case %s is not a listed uniform property", r.name.c_str(), c.id.c_str());
      }
      if (c.axis == "one_expression") {
        bool listed = false;
        for (const auto& kv : r.one_expression) listed = listed || has(kv.second, c.value);
        CHECK(listed, "%s: case %s is not a listed one-expression datum", r.name.c_str(), c.id.c_str());
      }
    }
    // the route's own lists agree with the cells (8.1: condition_kinds = the OFFERED kinds, not_offered = kind -> why)
    for (const auto& t : r.cell_types) {
      const auto* ty = reg().type(t);
      if (!ty) continue;
      const auto offered_it = r.condition_kinds.find(t);
      const auto not_it = r.not_offered.find(t);
      for (const auto& k : ty->condition_kinds) {
        const bool listed_offered = offered_it != r.condition_kinds.end() && has(offered_it->second, k.name);
        const bool listed_refused = not_it != r.not_offered.end() && not_it->second.count(k.name);
        CHECK(listed_offered != listed_refused, "%s: kind %s must be in exactly one of condition_kinds and not_offered",
              r.name.c_str(), k.name.c_str());
        // classes of the cells of this kind on this row
        std::vector<CellValue> cl;
        for (const auto& c : all)
          if (c.row == r.name && c.axis == "condition_kind" && c.physics == t &&
              (c.value == k.name || c.value.rfind(k.name + "@", 0) == 0))
            cl.push_back(c.cls);
        if (cl.empty()) continue;
        const bool any_offered = std::count(cl.begin(), cl.end(), CellValue::offered) > 0;
        CHECK(listed_offered == any_offered, "%s: kind %s: route lists and cases disagree on 'offered'", r.name.c_str(), k.name.c_str());
        if (!any_offered && listed_refused) {
          const std::string why = not_it->second.at(k.name);
          if (cl[0] == CellValue::engine_refuses) CHECK(why == "engine_refuses", "%s: %s: not_offered says '%s'", r.name.c_str(), k.name.c_str(), why.c_str());
          if (cl[0] == CellValue::not_exposed) CHECK(why.rfind("not_exposed: ", 0) == 0 && why.size() > 13, "%s: %s: not_offered says '%s'", r.name.c_str(), k.name.c_str(), why.c_str());
        }
      }
    }
  }
}

// ================================================================================================ 4: against the note
void check_cells_match_the_note() {
  const auto all = axmp::model::matrix_cases();
  const auto refs = reference_cells();
  std::set<std::string> seen;
  for (const auto& r : refs) {
    seen.insert(r.row + "|" + r.axis + "|" + r.value);
    const auto cs = cases_of(all, r.row, r.axis, r.value);
    if (cs.size() != 1) {
      CHECK(false, "cell %s %s=%s: %zu cases, want exactly 1", r.row.c_str(), r.axis.c_str(), r.value.c_str(), cs.size());
      continue;
    }
    const auto& c = *cs[0];
    CHECK(c.cls == r.cls, "cell %s %s=%s: class %s, the note says %s", r.row.c_str(), r.axis.c_str(), r.value.c_str(),
          to_str(c.cls).c_str(), to_str(r.cls).c_str());
    if (r.cls != CellValue::offered)
      CHECK(!c.code.empty(), "cell %s %s=%s: a refused cell names the code validation gives", r.row.c_str(), r.axis.c_str(),
            r.value.c_str());
    else
      CHECK(c.code.empty(), "cell %s %s=%s: an offered cell has no refusal code (has '%s')", r.row.c_str(), r.axis.c_str(),
            r.value.c_str(), c.code.c_str());
    if (!r.code.empty())
      CHECK(c.code == r.code, "cell %s %s=%s: code '%s', the note says '%s'", r.row.c_str(), r.axis.c_str(), r.value.c_str(),
            c.code.c_str(), r.code.c_str());
    if (!r.call.empty())
      CHECK(c.call == r.call, "cell %s %s=%s: call '%s', want '%s'", r.row.c_str(), r.axis.c_str(), r.value.c_str(), c.call.c_str(),
            r.call.c_str());
    if (!r.text.empty())
      CHECK(c.engine_text.find(r.text) != std::string::npos, "cell %s %s=%s: engine_text '%s' lacks '%s'", r.row.c_str(),
            r.axis.c_str(), r.value.c_str(), c.engine_text.c_str(), r.text.c_str());
    if (!r.member.empty())
      CHECK(c.member == r.member, "cell %s %s=%s: member '%s', want '%s'", r.row.c_str(), r.axis.c_str(), r.value.c_str(),
            c.member.c_str(), r.member.c_str());
  }
  // The kind cells of the flow, heat and 2D chemistry rows are fully transcribed: nothing else may be there.
  for (const auto& c : all) {
    if (c.row.empty() || c.axis != "condition_kind") continue;
    const note::Row* nr = find_note_row(c.row);
    if (!nr) { CHECK(false, "case %s is on an unknown row", c.id.c_str()); continue; }
    if (std::string(nr->geometry) == "3d" && c.row == "current_distribution_3d") continue;
    CHECK(seen.count(c.row + "|" + c.axis + "|" + c.value) == 1, "case %s is not in the transcription of 8.3", c.id.c_str());
  }
  // every row is known to the transcription
  for (const auto& c : all)
    if (!c.row.empty()) CHECK(find_note_row(c.row) != nullptr, "case %s: unknown row '%s'", c.id.c_str(), c.row.c_str());
  // the number of kind cells: flow 8 kinds x (5 columns + 2 transient rows) + heat 6 kinds x (2 plain-row cells +
  // 6 form rows x 3 forms) + chemistry 3 kinds x 2 rows (the extra "electrode:every_galvanostatic" is not a kind)
  size_t kinds = 0;
  for (const auto& c : all)
    if (!c.row.empty() && c.axis == "condition_kind" && c.value.find(':') == std::string::npos &&
        c.row != "current_distribution_3d")
      ++kinds;
  CHECK(kinds == 8 * 7 + 6 * (2 + 6 * 3) + 3 * 2, "%zu condition-kind cells, the note's tables give %d", kinds, 8 * 7 + 6 * 20 + 6);
}

// Option rules and row-selection data of every row with a geometry, 2D and 3D, against 8.3 and 8.5: convection is a
// Navier-Stokes option only (standard and skew on the steady rows; standard alone on the transient rows, whose engine
// refuses the skew form), the advection source selects the prescribed-advection rows, and no other row has a source
// selector.  The 3D rows (8.5 rows 6 and 12) then option by option, from the option lists of 8.3, which name no
// geometry class: fluid.incompressible has equation (which selects the row), viscous_form ("stress" default,
// "laplacian") and order (2), and convection on Navier-Stokes only; electrochem.current_distribution has kinetics
// ("primary", "secondary"; "secondary" default), order (1 or 2) and T_K (298.15).  So each 3D row has the rules of its
// planar twin.
void check_option_rules_and_row_selectors() {
  for (const auto& r : axmp::model::registry().routes) {
    if (r.geometries.empty()) continue;
    const bool ns = r.name.rfind("navier_stokes", 0) == 0;
    const bool stokes = r.name.rfind("stokes", 0) == 0;
    bool has_convection = false;
    for (const auto& kv : r.options)
      for (const auto& o : kv.second)
        if (o.name == "convection") {
          has_convection = true;
          const bool transient = r.name.rfind("navier_stokes_transient", 0) == 0;
          const Strs want = transient ? Strs{"standard"} : Strs{"standard", "skew"};
          CHECK(o.offered == want && o.default_value == "standard", "%s: convection offers %s, default standard",
                r.name.c_str(), transient ? "standard only (the time stepper refuses skew)" : "standard and skew");
        }
    if (ns) CHECK(has_convection, "%s: Navier-Stokes has the convection option (8.3)", r.name.c_str());
    if (stokes) CHECK(!has_convection, "%s: Stokes has no convection option (8.3)", r.name.c_str());
    const bool prescribed = r.name.rfind("heat_prescribed_advection", 0) == 0;
    if (prescribed) {
      auto it = r.when_sources.find("heat.transfer");
      CHECK(it != r.when_sources.end() && it->second == std::vector<std::string>{"prescribed_advection"},
            "%s: selected by the prescribed_advection source (8.5 row 8)", r.name.c_str());
    } else {
      CHECK(r.when_sources.empty(), "%s: no source selector (only the prescribed-advection rows have one)", r.name.c_str());
    }
  }
  // the 3D rows, option by option
  struct Want { const char* name; Strs offered; const char* forced; const char* dflt; };
  struct Row3d { const char* row; const char* type; const char* twin; std::vector<Want> options; };
  const std::vector<Row3d> rows3d = {
      {"stokes_3d", "fluid.incompressible", "stokes_planar",
       {{"equation", {"stokes"}, "", ""}, {"viscous_form", {"stress", "laplacian"}, "", "stress"}, {"order", {"2"}, "", "2"}}},
      {"current_distribution_3d", "electrochem.current_distribution", "current_distribution_planar",
       {{"kinetics", {"primary", "secondary"}, "", "secondary"}, {"order", {"1", "2"}, "", "2"}, {"T_K", {}, "", "298.15"}}},
  };
  auto rules = [](const axmp::model::Route& r, const std::string& type) {
    std::map<std::string, const axmp::model::OptionRule*> m;
    auto it = r.options.find(type);
    if (it != r.options.end())
      for (const auto& o : it->second) m[o.name] = &o;
    return m;
  };
  for (const auto& w : rows3d) {
    const auto* r = find_row(w.row);
    const auto* twin = find_row(w.twin);
    CHECK(r && twin, "%s or %s missing", w.row, w.twin);
    if (!r || !twin) continue;
    CHECK(r->options.size() == 1 && r->options.count(w.type) == 1, "%s: option rules for %s only", w.row, w.type);
    const auto got = rules(*r, w.type);
    CHECK(got.size() == w.options.size(), "%s: %zu option rules, 8.3 gives %zu", w.row, got.size(), w.options.size());
    for (const auto& o : w.options) {
      auto it = got.find(o.name);
      CHECK(it != got.end(), "%s: no rule for option %s (8.3)", w.row, o.name);
      if (it == got.end()) continue;
      CHECK(it->second->offered == o.offered, "%s: option %s offers the values 8.3 gives", w.row, o.name);
      CHECK(it->second->forced == o.forced, "%s: option %s is forced to '%s'", w.row, o.name, it->second->forced.c_str());
      CHECK(it->second->default_value == o.dflt, "%s: option %s has the default '%s', 8.3 gives '%s'", w.row, o.name,
            it->second->default_value.c_str(), o.dflt);
    }
    // the planar twin has the same rules (8.3 names no geometry class)
    const auto tw = rules(*twin, w.type);
    CHECK(tw.size() == got.size(), "%s: %zu option rules, its planar twin %s has %zu", w.row, got.size(), w.twin, tw.size());
    for (const auto& [name, o] : got) {
      auto it = tw.find(name);
      CHECK(it != tw.end() && it->second->offered == o->offered && it->second->forced == o->forced &&
                it->second->default_value == o->default_value,
            "%s: option %s differs from the rule of %s", w.row, name.c_str(), w.twin);
    }
  }
  CHECK(find_row("stokes_3d")->when == (std::map<std::string, std::string>{{"equation", "stokes"}}),
        "stokes_3d is selected by equation = stokes");
  CHECK(find_row("current_distribution_3d")->when.empty(), "current_distribution_3d has no selecting option");
}

void check_not_exposed_cells_carry_a_reason_and_a_roadmap_item() {
  const auto all = axmp::model::matrix_cases();
  int periodic = 0;
  for (const auto& c : all) {
    if (c.cls != CellValue::not_exposed) continue;
    CHECK(!c.reason.empty(), "%s: not_exposed without a reason", c.id.c_str());
    CHECK(!c.roadmap.empty(), "%s: not_exposed without a roadmap item", c.id.c_str());
    CHECK(c.meshes.empty(), "%s: a not_exposed cell asserts nothing of the engine (16 rule 2)", c.id.c_str());
    if (c.value == "periodic") {
      ++periodic;
      CHECK(c.roadmap == "periodic flow", "%s: roadmap '%s' (Q24 and 21.1: the roadmap row is 'periodic flow')", c.id.c_str(),
            c.roadmap.c_str());
      CHECK(c.row == "stokes_planar" || c.row == "stokes_3d", "%s: periodic is 'later' on the planar and 3D Stokes rows only (Q24)",
            c.id.c_str());
    }
  }
  CHECK(periodic == 2, "%d not_exposed periodic cells, the note has 2 (planar and 3D Stokes)", periodic);
  for (const auto& c : all)
    if (c.cls == CellValue::layer_rule) {
      CHECK(!c.reason.empty(), "%s: layer_rule without a reason", c.id.c_str());
      CHECK(c.meshes.empty(), "%s: a layer rule is not compared with the engine", c.id.c_str());
    }
}

void check_engine_refuses_cells_name_the_call_and_the_text() {
  const auto all = axmp::model::matrix_cases();
  int n = 0;
  for (const auto& c : all) {
    if (c.cls != CellValue::engine_refuses) continue;
    ++n;
    CHECK(kCalls.count(c.call) == 1, "%s: call '%s'", c.id.c_str(), c.call.c_str());
    CHECK(!c.engine_text.empty(), "%s: engine_refuses without the engine's text", c.id.c_str());
    CHECK(!c.engine.empty(), "%s: no engine", c.id.c_str());
    CHECK(!c.code.empty(), "%s: no validation code", c.id.c_str());
  }
  CHECK(n > 40, "only %d engine_refuses cells", n);
  // every offered cell that has an engine column names the call it was checked through (16 rule 2: through finalize
  // where the engine has one)
  for (const auto& c : all)
    if (c.cls == CellValue::offered && !c.meshes.empty() && c.axis != "ranks")
      CHECK(kCalls.count(c.call) == 1, "%s: offered cell with call '%s'", c.id.c_str(), c.call.c_str());
  for (const auto& c : all)
    if (c.engine == "transport::ScalarTransport" && c.cls == CellValue::offered && !c.meshes.empty())
      CHECK(c.call == "finalize", "%s: the transport operator checks its conditions in finalize, so an offered cell goes through it (call '%s')",
            c.id.c_str(), c.call.c_str());
}

void check_engine_cannot_express_cells_name_the_member() {
  const auto all = axmp::model::matrix_cases();
  int n = 0;
  for (const auto& c : all) {
    if (c.cls != CellValue::engine_cannot_express) continue;
    ++n;
    CHECK(!c.member.empty(), "%s: engine_cannot_express without the member the type assertion names", c.id.c_str());
    CHECK(c.meshes.empty(), "%s: there is no problem to hand over, so no engine column", c.id.c_str());
    CHECK(c.code == "physics.property_not_uniform_over_domain" || c.code == "heat.data_not_regionwise" ||
              c.code == "heat.prescribed_advection_partial" || c.code == "heat.option_not_offered",
          "%s: code '%s'", c.id.c_str(), c.code.c_str());
    CHECK((c.axis == "uniform") == (c.code == "physics.property_not_uniform_over_domain"), "%s: axis and code", c.id.c_str());
    CHECK((c.axis == "option_value") == (c.code == "heat.option_not_offered"), "%s: axis and code", c.id.c_str());
  }
  // uniform: NS steady 2, transient 2, prescribed 2 x 2, one-way 2 x 2, Boussinesq 2 x 3 = 2+2+4+4+6; one expression:
  // heat 2 x 2, prescribed 2 x 3, one-way 2 x 2, Boussinesq 2 x 2 = 4+6+4+4; the advective forms the Boussinesq solver
  // does not have: 2 rows x 2 forms
  CHECK(n == (4 + 4 + 4 + 6) + (4 + 6 + 4 + 4) + 4, "%d engine_cannot_express cells, the note lists %d", n, 18 + 18 + 4);
}

void check_layer_rules_are_listed() {
  std::set<std::string> known;
  for (const auto& lr : note::layer_rules) known.insert(lr.name);
  CHECK(known.size() == 6, "the note lists six layer rules");
  for (const auto& lr : note::layer_rules) {
    std::set<std::string> have;
    for (const auto& r : reg().routes)
      if (has(r.layer_rules, lr.name)) have.insert(r.name);
    CHECK(have == std::set<std::string>(lr.rows.begin(), lr.rows.end()), "layer rule %s is listed on %zu rows, the note applies it to %zu",
          lr.name, have.size(), lr.rows.size());
  }
  for (const auto& r : reg().routes)
    for (const auto& n : r.layer_rules) CHECK(known.count(n) == 1, "%s: unknown layer rule '%s'", r.name.c_str(), n.c_str());
  // the registry gives each cell its class per row the same geometry-order rule is a layer rule on the
  // flow and 3D rows and an engine_refuses cell on the 2D current-distribution rows
  for (const char* n : {"current_distribution_planar", "current_distribution_axisymmetric"})
    CHECK(!has(find_row(n)->layer_rules, "geometry_order_exceeds_field_order"), "%s: the geometry order is an engine cell here", n);
  // outflow with conservative_flux is a cell of class layer_rule on the rows that select the form
  const auto all = axmp::model::matrix_cases();
  for (const auto& row : note::form_rows) {
    const auto cs = cases_of(all, row, "condition_kind", "outflow@conservative_flux");
    CHECK(cs.size() == 1 && cs[0]->cls == CellValue::layer_rule, "%s: outflow@conservative_flux is a layer rule", row.c_str());
  }
}

void check_beta_types_list_what_is_not_implemented() {
  for (const auto& t : reg().types)
    if (t.status == "beta") {
      CHECK(!t.not_implemented.empty(), "%s: beta with an empty not_implemented list", t.name.c_str());
      for (const auto& s : t.not_implemented) CHECK(!s.empty(), "%s: empty not_implemented entry", t.name.c_str());
    }
  for (const auto& t : reg().terms)
    if (t.status == "beta") CHECK(!t.title.empty(), "term %s: title", t.name.c_str());
  for (const char* n : {"fluid.incompressible", "heat.transfer", "electrochem.current_distribution", "chem.reactor",
                        "electrochem.porous_electrode"})
    CHECK(reg().type(n) && reg().type(n)->status == "beta", "%s is beta (8.2)", n);
  for (const char* n : {"current.conduction", "solid.linear", "species.transport"})
    CHECK(reg().type(n) && reg().type(n)->status == "absent", "%s is absent (8.2)", n);
}

void check_route_requirements() {
  // 8.5 "needs" column and 16: stated once, in the row.  Covered with the rows above; here the flow rows' unavailable
  // finding needs the fluids flag and every heat row needs it too (16 rule 6: heat.transfer needs an MPI build)
  for (const auto& r : reg().routes) {
    const bool heat_or_flow = has(r.physics, "heat.transfer") || has(r.physics, "fluid.incompressible");
    CHECK(r.needs.fluids == heat_or_flow, "%s: needs.fluids %d", r.name.c_str(), (int)r.needs.fluids);
    if (r.geometries.empty()) CHECK(!r.needs.mesher, "%s: geometry-free, so no mesher", r.name.c_str());
    else CHECK(r.needs.mesher, "%s: a row with a geometry needs the mesher (8.5)", r.name.c_str());
  }
}

void check_no_factory_is_registered() {
  using axmp::model::RouteFactory;
  for (const auto& r : reg().routes)
    CHECK(axmp::model::route_factory(r.name) == nullptr, "%s: a factory is registered; none is registered in this version", r.name.c_str());
  CHECK(axmp::model::route_factory("no_such_route") == nullptr, "unknown route");
  // register_route_factory is idempotent.  A function of another signature stands in: only the pointer matters.
  static void (*const f)() = [] {};
  RouteFactory dummy = reinterpret_cast<RouteFactory>(f);
  axmp::model::register_route_factory("test_dummy_route_of_the_registry_test", dummy);
  axmp::model::register_route_factory("test_dummy_route_of_the_registry_test", dummy);
  CHECK(axmp::model::route_factory("test_dummy_route_of_the_registry_test") == dummy, "registered factory not returned");
  CHECK(axmp::model::route_factory("stokes_planar") == nullptr, "registering one name must not register another");
  // the registry is built on first use and is one object
  CHECK(&axmp::model::registry() == &axmp::model::registry(), "registry() returns one object");
}

// ================================================================================================ 5: meshes and ranks
void check_case_meshes_and_ranks() {
  const auto all = axmp::model::matrix_cases();
  const std::set<std::string> mesh_names = {"base", "sparse", "nodes", "quad"};
  for (const auto& c : all) {
    for (const auto& m : c.meshes) CHECK(mesh_names.count(m) == 1, "%s: unknown mesh '%s'", c.id.c_str(), m.c_str());
    if (!c.meshes.empty())
      CHECK(c.cls == CellValue::offered || c.cls == CellValue::engine_refuses,
            "%s: only offered and engine_refuses cells have an engine column", c.id.c_str());
    if (c.axis == "ranks") CHECK(c.ranks == 2, "%s: a rank cell is a two-rank cell", c.id.c_str());
    else CHECK(c.ranks == 1, "%s: only rank cells run on 2 ranks", c.id.c_str());
    // The column is asserted here from the row's geometry, not from the generator's own bookkeeping: an offered or
    // refused cell of a fluids engine or of the transport operator on a planar or axisymmetric row, and every
    // sparse-mesh cell of those engines, is handed to its engine (a case without meshes is skipped by the engine
    // programs without a failure).
    const bool fluids_engine = c.engine == "StokesSolver" || c.engine == "NavierStokesSolver" ||
                               c.engine == "TransientNavierStokes" || c.engine == "BoussinesqSolver" ||
                               c.engine == "transport::ScalarTransport";
    if (fluids_engine && (c.cls == CellValue::offered || c.cls == CellValue::engine_refuses)) {
      const auto* row = c.row.empty() ? nullptr : find_row(c.row);
      const bool two_d = row && (row->geometries == Strs{"planar"} || row->geometries == Strs{"axisymmetric"});
      if (two_d || c.axis == "mesh_numbering")
        CHECK(!c.meshes.empty(), "%s: a cell of a fluids or transport engine on a 2D row has no engine column", c.id.c_str());
    }
    // 3D rows are not confirmed against an engine yet and the reactor is not an engine: no engine column in this version
    if (c.row == "stokes_3d" || c.row == "current_distribution_3d" || c.row == "chem_reactor")
      CHECK(c.meshes.empty(), "%s: no engine column in this version", c.id.c_str());
  }
  // 16 rule 2, "the sparse mesh": not rows of a route.  Every flow engine accepts it; the transport operator refuses
  // it in finalize with "boundary attribute 1 has no condition", and the Boussinesq solver with the same text in its
  // constructor, which finalizes its transport operator (boussinesq.cpp: "tr_->finalize();" inside
  // BoussinesqSolver::BoussinesqSolver).
  struct Sparse { const char* engine; CellValue cls; const char* call; };
  const Sparse sparse[] = {{"StokesSolver", CellValue::offered, "constructor"},
                           {"NavierStokesSolver", CellValue::offered, "constructor"},
                           {"TransientNavierStokes", CellValue::offered, "constructor"},
                           {"BoussinesqSolver", CellValue::engine_refuses, "constructor"},
                           {"transport::ScalarTransport", CellValue::engine_refuses, "finalize"}};
  for (const auto& s : sparse) {
    int n = 0;
    for (const auto& c : all)
      if (c.row.empty() && c.engine == s.engine && c.meshes == Strs{"sparse"}) {
        ++n;
        CHECK(c.cls == s.cls, "sparse mesh on %s: class %s", s.engine, to_str(c.cls).c_str());
        if (s.cls == CellValue::engine_refuses) {
          CHECK(c.call == s.call, "sparse mesh on %s: call '%s', want '%s'", s.engine, c.call.c_str(), s.call);
          CHECK(c.engine_text.find("boundary attribute 1 has no condition") != std::string::npos,
                "sparse mesh on %s: text '%s'", s.engine, c.engine_text.c_str());
        }
      }
    CHECK(n == 1, "%d sparse-mesh cells on %s, want 1", n, s.engine);
  }
  // the nodes cells (16 rule 2): supg refused by the constructor on a mesh with nodes, accepted on the base mesh;
  // without supg both meshes are accepted
  for (const char* row : {"heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric", "flow_then_heat_planar",
                          "flow_then_heat_axisymmetric", "boussinesq_planar", "boussinesq_axisymmetric"}) {
    const auto n = cases_of(all, row, "geometry_order", "supg");
    CHECK(n.size() == 1 && n[0]->meshes == Strs{"nodes"} && n[0]->cls == CellValue::engine_refuses, "%s: supg on nodes", row);
    const auto s = cases_of(all, row, "option_value", "stabilization=supg");
    CHECK(s.size() == 1 && s[0]->meshes == Strs{"base"} && s[0]->cls == CellValue::offered, "%s: supg on the base mesh", row);
    const auto g = cases_of(all, row, "option_value", "stabilization=none");
    CHECK(g.size() == 1 && std::set<std::string>(g[0]->meshes.begin(), g[0]->meshes.end()) ==
                               (std::set<std::string>{"base", "nodes"}),
          "%s: without supg both meshes are accepted", row);
  }
  // the quadrilateral cells: the heat family refuses them in the constructor; every flow engine accepts them
  for (const auto& c : all)
    if (c.axis == "element_type" && c.value == "quadrilateral" && c.row != "") {
      CHECK(c.meshes == Strs{"quad"}, "%s: the quadrilateral cell runs on the quad mesh", c.id.c_str());
    }
  // two-rank cells: the flow engines solve on any number of ranks, the stationary ones through their solve and the
  // time stepper through its first time steps; the transport operator and the Boussinesq solver are one-rank (8.5
  // "needs"; 16)
  for (const char* row : {"stokes_planar", "stokes_axisymmetric", "navier_stokes_steady_planar",
                          "navier_stokes_steady_axisymmetric", "navier_stokes_transient_planar",
                          "navier_stokes_transient_axisymmetric"}) {
    const auto r = cases_of(all, row, "ranks", "2");
    CHECK(r.size() == 1 && r[0]->ranks == 2 && r[0]->cls == CellValue::offered && !r[0]->meshes.empty(), "%s: offered on 2 ranks", row);
    const std::string call = std::string(row).rfind("navier_stokes_transient", 0) == 0 ? "first_step" : "solve";
    if (r.size() == 1) CHECK(r[0]->call == call, "%s: call '%s', want '%s'", row, r[0]->call.c_str(), call.c_str());
  }
  for (const auto& nr : note::rows) {
    if (nr.max_ranks != 1) continue;
    const std::string name = nr.name;
    if (name.rfind("current_distribution", 0) == 0 || name == "chem_reactor") continue;  // one process by construction
    const auto r = cases_of(all, name, "ranks", "2");
    CHECK(r.size() == 1, "%s: %zu two-rank cells, want 1", nr.name, r.size());
    if (r.size() != 1) continue;
    CHECK(r[0]->ranks == 2 && r[0]->cls == CellValue::engine_refuses && r[0]->code == "route.single_rank",
          "%s: the two-rank cell is an engine refusal with route.single_rank (class %s, code '%s')", nr.name,
          to_str(r[0]->cls).c_str(), r[0]->code.c_str());
    // the transport operator refuses more than one rank in its constructor ("ScalarTransport needs one MPI rank")
    if (name.rfind("heat_", 0) == 0 || name.rfind("flow_then_heat", 0) == 0)
      CHECK(r[0]->call == "constructor", "%s: call '%s'", nr.name, r[0]->call.c_str());
  }
  // the rank cells are all of this list; none is on a row that takes any number of ranks except the offered six
  int rank_cells = 0;
  for (const auto& c : all)
    if (c.axis == "ranks") ++rank_cells;
  CHECK(rank_cells == 6 + 8, "%d rank cells: 6 flow rows (offered) and 8 one-rank heat rows (refused)", rank_cells);
}

// ================================================================================================ 6: documents
std::string scalar_text(const json& v) { return v.is_string() ? v.get<std::string>() : v.dump(); }

const json* instance_of_type(const json& doc, const std::string& type) {
  if (!doc.contains("physics") || !doc["physics"].is_object()) return nullptr;
  for (const auto& kv : doc["physics"].items())
    if (kv.value().is_object() && kv.value().value("type", "") == type) return &kv.value();
  return nullptr;
}

void check_every_validation_cell_has_a_document() {
  const auto all = axmp::model::matrix_cases();
  int with = 0;
  for (const auto& c : all) {
    const json doc = axmp::model::minimal_document(c);
    const bool want = !c.row.empty() && c.axis != "ranks";
    if (!want) {
      CHECK(doc.is_null(), "%s: no validation column, so no document", c.id.c_str());
      continue;
    }
    ++with;
    CHECK(doc.is_object(), "%s: no document", c.id.c_str());
    if (!doc.is_object()) continue;
    CHECK(doc == axmp::model::minimal_document(c), "%s: the document is not deterministic", c.id.c_str());
    CHECK(doc.value("schema", "") == "axionmultiphysics.design/1", "%s: schema string", c.id.c_str());
    CHECK(doc.contains("physics") && doc["physics"].is_object() && !doc["physics"].empty(), "%s: physics section", c.id.c_str());
    const auto* nr = find_note_row(c.row);
    if (!nr) continue;
    const std::string geom = nr->geometry;
    if (!geom.empty()) CHECK(doc.value("symmetry", "") == geom, "%s: symmetry '%s' != '%s'", c.id.c_str(), doc.value("symmetry", "").c_str(), geom.c_str());
    // every type of the row is an instance of the document (the coupled rows carry both instances)
    for (const auto& t : nr->physics)
      CHECK(instance_of_type(doc, t) != nullptr, "%s: no instance of %s", c.id.c_str(), t.c_str());
    for (const auto& co : nr->couplings) {
      bool found = false;
      if (doc.contains("couplings"))
        for (const auto& e : doc["couplings"]) found = found || e.value("term", "") == co;
      CHECK(found, "%s: coupling term %s is missing", c.id.c_str(), co.c_str());
    }
    const json* inst = instance_of_type(doc, c.physics);
    if (c.axis == "condition_kind") {
      std::string kind = c.value.substr(0, c.value.find_first_of("@:"));
      CHECK(doc.dump().find("\"kind\":\"" + kind + "\"") != std::string::npos, "%s: no boundary condition of kind %s", c.id.c_str(), kind.c_str());
      const auto at = c.value.find('@');
      if (at != std::string::npos && inst)
        CHECK(inst->contains("options") && scalar_text((*inst)["options"].value("advective_form", json(""))) == c.value.substr(at + 1),
              "%s: advective_form option", c.id.c_str());
    } else if (c.axis == "option_value" && inst) {
      const auto eq = c.value.find('=');
      const std::string name = c.value.substr(0, eq), value = c.value.substr(eq + 1);
      CHECK(inst->contains("options") && (*inst)["options"].contains(name) && scalar_text((*inst)["options"][name]) == value,
            "%s: option %s is not %s in the document", c.id.c_str(), name.c_str(), value.c_str());
    } else if (c.axis == "uniform" || c.axis == "one_expression") {
      CHECK(doc.contains("regions") && doc["regions"].is_array() && doc["regions"].size() >= 2,
            "%s: two regions with different values", c.id.c_str());
    } else if (c.axis == "element_type") {
      CHECK(doc.contains("mesh") && doc["mesh"].is_object() && doc["mesh"].contains("file"), "%s: a mesh.file", c.id.c_str());
    } else if (c.axis == "geometry_order") {
      CHECK(doc.contains("mesh") && doc["mesh"].value("geometry_order", 0) == 2, "%s: mesh.geometry_order 2", c.id.c_str());
      if (c.value == "supg" && inst)
        CHECK(inst->contains("options") && scalar_text((*inst)["options"].value("stabilization", json(""))) == "supg", "%s: supg", c.id.c_str());
      if (c.value == "order=1" && inst)
        CHECK(inst->contains("options") && scalar_text((*inst)["options"].value("order", json(0))) == "1", "%s: order 1", c.id.c_str());
    } else if (c.axis == "study_type") {
      CHECK(doc.contains("study") && doc["study"].is_array() && !doc["study"].empty() &&
                doc["study"][0].value("type", "") == c.value,
            "%s: study type %s", c.id.c_str(), c.value.c_str());
    }
  }
  CHECK(with > 150, "only %d cells have a validation column", with);
}

// ================================================================================================ 6b: what a document says
// A document is the base problem of its row with exactly the variation of its cell (cpp/src/model_matrix.hpp,
// minimal_document).  Its deviations are read here from the document and the row's own lists: a kind or an option value
// the row does not offer, a geometry order the field order or SUPG forbids, a property or source that differs between
// the regions, a study type the row has no route for, no grounded electrode, a per-depth key in the wrong spelling, an
// insulated boundary that the advecting velocity crosses.  An offered cell's document has none; a refused cell's
// document has exactly its own, of its own class.  The geometry is that of the base documents: the unit rectangle (a
// unit box in 3D), with a side on r = 0 for the axisymmetric rows, and no boundary on the axis.

// the value of option `name` on an instance: the document's, else the default of the type
std::string instance_option(const json& inst, const std::string& type, const std::string& name) {
  if (inst.contains("options") && inst["options"].contains(name)) return scalar_text(inst["options"][name]);
  if (const auto* t = reg().type(type))
    for (const auto& o : t->options)
      if (o.name == name) return o.default_value;
  return "";
}

// the row whose lists an instance of `type` is read against: the row itself for its cell types, else (the flow
// instance of a coupled row) the stationary single-type row of the same geometry that the instance's options select
const axmp::model::Route* row_for_instance(const axmp::model::Route& row, const std::string& type, const json& inst) {
  if (has(row.cell_types, type)) return &row;
  for (const auto& r : reg().routes) {
    if (r.physics != Strs{type} || r.geometries != row.geometries || !has(r.study_types, "stationary")) continue;
    bool selected = true;
    for (const auto& [k, v] : r.when) selected = selected && instance_option(inst, type, k) == v;
    if (selected) return &r;
  }
  return nullptr;
}

// the regions an `in` selector names: a region, a domain of the document, or (absent) every region
std::set<std::string> regions_named(const json& doc, const json& in) {
  std::set<std::string> out;
  if (in.is_null()) {
    for (const auto& r : doc["regions"]) out.insert(r.value("name", ""));
    return out;
  }
  const std::string n = in.get<std::string>();
  if (doc.contains("domains") && doc["domains"].contains(n))
    for (const auto& m : doc["domains"][n]) out.insert(m.get<std::string>());
  else
    out.insert(n);
  return out;
}

// the properties of a material, flattened: {"density": "1.0", "thermal.k_W_mK": "1.0", ...}
std::map<std::string, std::string> flat_properties(const json& m) {
  std::map<std::string, std::string> out;
  for (const auto& kv : m.items()) {
    if (kv.value().is_object())
      for (const auto& sub : kv.value().items()) out[kv.key() + "." + sub.key()] = sub.value().dump();
    else
      out[kv.key()] = kv.value().dump();
  }
  return out;
}

// the deviations of a document from its row: key -> the row's reason (or a description)
std::map<std::string, std::string> deviations(const MatrixCase& c, const json& doc) {
  std::map<std::string, std::string> dev;
  const auto* row = find_row(c.row);
  if (!row) return dev;
  const std::string g = row->geometries.empty() ? "" : row->geometries[0];
  // ---- study types (no study: the default stationary step)
  if (doc.contains("study"))
    for (const auto& st : doc["study"])
      if (!has(row->study_types, st.value("type", ""))) dev["study:" + st.value("type", "")] = "no route of this study type";
  if (!doc.contains("study") && !has(row->study_types, "stationary")) dev["study:stationary"] = "the default step";
  if (g.empty()) return dev;
  // ---- the properties that differ between the regions
  std::map<std::string, std::string> region_material;
  for (const auto& r : doc["regions"]) region_material[r.value("name", "")] = r.value("material", "");
  std::map<std::string, std::map<std::string, std::string>> props_of;   // region -> its properties
  std::set<std::string> props;
  for (const auto& [rn, mat] : region_material) {
    if (doc["materials"].contains(mat)) props_of[rn] = flat_properties(doc["materials"][mat]);
    for (const auto& kv : props_of[rn]) props.insert(kv.first);
  }
  for (const auto& p : props) {
    std::set<std::string> vs;
    for (const auto& [rn, f] : props_of) vs.insert(f.count(p) ? f.at(p) : "(absent)");
    if (vs.size() > 1) dev["regions:" + p] = "differs between the regions";
  }
  // ---- instances
  int geometry_order = 1;
  if (doc.contains("mesh") && doc["mesh"].contains("geometry_order")) geometry_order = doc["mesh"]["geometry_order"].get<int>();
  std::set<std::string> crossed;  // the boundaries the advecting velocity crosses
  bool any_grounded = false, any_electrode = false;
  for (const auto& kv : doc["physics"].items()) {
    const json& inst = kv.value();
    const std::string type = inst.value("type", "");
    const auto* rr = row_for_instance(*row, type, inst);
    if (!rr) {
      dev["instance:" + type] = "no row for the instance";
      continue;
    }
    // options: offered, forced or row-selecting values
    if (inst.contains("options"))
      for (const auto& o : inst["options"].items()) {
        const std::string v = scalar_text(o.value());
        if (rr->when.count(o.key())) {
          if (rr->when.at(o.key()) != v) dev["option:" + type + ":" + o.key() + "=" + v] = "selects another row";
          continue;
        }
        const axmp::model::OptionRule* rule = nullptr;
        if (auto it = rr->options.find(type); it != rr->options.end())
          for (const auto& x : it->second)
            if (x.name == o.key()) rule = &x;
        const bool number = rule && rule->offered.empty() && rule->forced.empty();
        if (!rule || (!rule->forced.empty() && rule->forced != v) || (rule->forced.empty() && !number && !has(rule->offered, v)))
          dev["option:" + type + ":" + o.key() + "=" + v] = "not offered";
      }
    // the geometry order, against the field order and SUPG
    const std::string order = instance_option(inst, type, "order");
    if (!order.empty() && geometry_order > std::stoi(order)) dev["geometry_order:order=" + order] = "above the field order";
    if (geometry_order > 1 && instance_option(inst, type, "stabilization") == "supg") dev["geometry_order:supg"] = "SUPG on curved elements";
    // condition kinds, against the row's lists for the instance's advective form
    const std::string form = instance_option(inst, type, "advective_form");
    if (inst.contains("boundary_conditions"))
      for (const auto& b : inst["boundary_conditions"].items()) {
        const std::string k = b.value().value("kind", "");
        std::string why;
        if (auto w = rr->not_offered_with.find(type); w != rr->not_offered_with.end() && !form.empty())
          if (auto wf = w->second.find("advective_form=" + form); wf != w->second.end() && wf->second.count(k)) why = wf->second.at(k);
        if (why.empty())
          if (auto n = rr->not_offered.find(type); n != rr->not_offered.end() && n->second.count(k)) why = n->second.at(k);
        if (why.empty())
          if (auto ok = rr->condition_kinds.find(type); ok == rr->condition_kinds.end() || !has(ok->second, k)) why = "not listed";
        if (!why.empty()) dev["kind:" + type + ":" + k] = why;
        // a solved flow crosses every boundary but a wall at rest, a slip and a symmetry boundary (8.3, the structural
        // branch of insulation_on_throughflow)
        if (type == "fluid.incompressible" && has(row->couplings, "advection")) {
          const bool tangential = (k == "wall" && !b.value().contains("velocity_m_s")) || k == "slip" || k == "symmetry";
          if (!tangential) crossed.insert(b.key());
        }
        // electrodes: grounded (a potential, or primary) or galvanostatic, with the current in the geometry's spelling
        if (k == "electrode") {
          any_electrode = true;
          any_grounded = any_grounded || b.value().contains("phi_s_V") || b.value().value("primary", false);
          if (b.value().contains(g == "planar" ? "current_A" : "current_A_m"))
            dev["layer:per_depth_keys"] = "the current key of another geometry class";
        }
      }
    // sources: a heat source that differs between the regions; a prescribed velocity over part of the instance
    if (inst.contains("sources")) {
      std::map<std::string, double> q;
      std::set<std::string> advected;
      for (const auto& rm : region_material) q[rm.first] = 0.0;
      for (const auto& s : inst["sources"]) {
        const json in = s.contains("in") ? s["in"] : json();
        for (const auto& rn : regions_named(doc, in)) {
          if (s.value("kind", "") == "heat_source") q[rn] += s.value("q_W_m3", 0.0);
          if (s.value("kind", "") == "prescribed_advection") advected.insert(rn);
        }
        if (s.value("kind", "") == "prescribed_advection" && s.contains("velocity_m_s")) {
          // a constant velocity crosses the sides it is not parallel to
          const json& u = s["velocity_m_s"];
          const bool along0 = u.size() > 0 && u[0].is_number() && u[0].get<double>() != 0.0;
          const bool along1 = u.size() > 1 && u[1].is_number() && u[1].get<double>() != 0.0;
          if (g == "planar" && along0) {
            crossed.insert("left");
            crossed.insert("right");
          }
          if (g == "axisymmetric" && along0) crossed.insert("right");
          if ((g == "planar" || g == "axisymmetric") && along1) {
            crossed.insert("bottom");
            crossed.insert("top");
          }
        }
      }
      std::set<double> qs;
      for (const auto& kq : q) qs.insert(kq.second);
      if (qs.size() > 1) dev["regions:heat_source"] = "differs between the regions";
      if (!advected.empty() && advected.size() != region_material.size())
        dev["regions:prescribed_advection.velocity"] = "covers part of the instance";
    }
  }
  if (any_electrode && !any_grounded) dev["electrode:every_galvanostatic"] = "no grounded electrode";
  for (const auto& kv : doc["physics"].items())
    if (kv.value().value("type", "") == "heat.transfer" && kv.value().contains("boundary_conditions"))
      for (const auto& b : kv.value()["boundary_conditions"].items())
        if (b.value().value("kind", "") == "insulation" && crossed.count(b.key()))
          dev["layer:insulation_on_throughflow"] = "insulation on '" + b.key() + "'";
  return dev;
}

// the one deviation a refused cell's document has ("" for an offered cell, whose document has none; the element-type
// cells write a mesh file, which validation reports deferred, and nothing else)
std::string expected_deviation(const MatrixCase& c) {
  if (c.cls == CellValue::offered || c.axis == "element_type") return "";
  if (c.axis == "condition_kind") {
    if (c.value.find(':') != std::string::npos) return c.value;   // electrode:every_galvanostatic
    return "kind:" + c.physics + ":" + c.value.substr(0, c.value.find('@'));
  }
  if (c.axis == "option_value") return "option:" + c.physics + ":" + c.value;
  if (c.axis == "geometry_order") return "geometry_order:" + c.value;
  if (c.axis == "uniform" || c.axis == "one_expression") return "regions:" + c.value;
  if (c.axis == "study_type") return "study:" + c.value;
  return "(no document expected for axis " + c.axis + ")";
}

void check_documents_express_their_cells() {
  const auto all = axmp::model::matrix_cases();
  int n = 0;
  for (const auto& c : all) {
    const json doc = axmp::model::minimal_document(c);
    if (!doc.is_object() || !doc.contains("physics") || !doc["physics"].is_object()) continue;
    ++n;
    const auto* row = find_row(c.row);
    if (!row) continue;
    const std::string g = row->geometries.empty() ? "" : row->geometries[0];
    // ---- the deviations
    const auto dev = deviations(c, doc);
    const std::string want = expected_deviation(c);
    std::string got;
    for (const auto& kv : dev) got += (got.empty() ? "" : ", ") + kv.first + " (" + kv.second + ")";
    if (want.empty()) {
      CHECK(dev.empty(), "%s: the document of an offered cell writes what its row refuses: %s", c.id.c_str(), got.c_str());
    } else {
      CHECK(dev.size() == 1 && dev.count(want) == 1, "%s: the document must deviate by '%s' only; it deviates by: %s",
            c.id.c_str(), want.c_str(), got.empty() ? "nothing" : got.c_str());
      // a refused kind is refused by the row for the cell's own class
      if (dev.count(want) && c.axis == "condition_kind" && c.value.find(':') == std::string::npos) {
        const std::string& why = dev.at(want);
        const bool cls_ok = (c.cls == CellValue::engine_refuses && why == "engine_refuses") ||
                            (c.cls == CellValue::not_exposed && why.rfind("not_exposed: ", 0) == 0) ||
                            (c.cls == CellValue::layer_rule && why.rfind("layer_rule: ", 0) == 0);
        CHECK(cls_ok, "%s: the row refuses the kind with '%s', the cell's class is %s", c.id.c_str(), why.c_str(),
              to_str(c.cls).c_str());
      }
    }
    // every electrode galvanostatic: each electrode carries the current, in the spelling of the geometry class
    if (c.value == "electrode:every_galvanostatic") {
      const json* inst = instance_of_type(doc, c.physics);
      int electrodes = 0;
      if (inst)
        for (const auto& b : (*inst)["boundary_conditions"].items())
          if (b.value().value("kind", "") == "electrode") {
            ++electrodes;
            CHECK(!b.value().contains("phi_s_V") && !b.value().value("primary", false) &&
                      b.value().contains(g == "planar" ? "current_A_m" : "current_A"),
                  "%s: electrode '%s' is not galvanostatic: %s", c.id.c_str(), b.key().c_str(), b.value().dump().c_str());
          }
      CHECK(electrodes >= 2, "%s: %d electrodes, want at least two galvanostatic ones", c.id.c_str(), electrodes);
    }
    if (g.empty()) continue;
    // ---- the geometry: the unit rectangle (a unit box in 3D), with a side on r = 0 in axisymmetric geometry
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    bool on_axis = false;
    std::set<std::string> region_names;
    for (const auto& r : doc["regions"]) {
      region_names.insert(r.value("name", ""));
      std::vector<std::pair<double, double>> ext;
      if (g == "3d") {
        const json& b = r["solid"]["box"];
        for (int i = 0; i < 3; ++i) ext.push_back({b["min"][i].get<double>(), b["max"][i].get<double>()});
      } else {
        const json& rect = r["shape"]["rect"];
        for (const char* a : g == "planar" ? std::vector<const char*>{"x", "y"} : std::vector<const char*>{"r", "z"}) {
          CHECK(rect.contains(a), "%s: region '%s' has no %s range", c.id.c_str(), r.value("name", "").c_str(), a);
          ext.push_back(rect.contains(a) ? std::make_pair(rect[a][0].get<double>(), rect[a][1].get<double>())
                                         : std::make_pair(0.0, 0.0));
        }
        if (g == "axisymmetric" && ext[0].first == 0.0) on_axis = true;
      }
      for (size_t i = 0; i < ext.size(); ++i) lo[i] = std::min(lo[i], ext[i].first), hi[i] = std::max(hi[i], ext[i].second);
    }
    const int dim = g == "3d" ? 3 : 2;
    for (int i = 0; i < dim; ++i)
      CHECK(lo[i] == 0.0 && hi[i] == 1.0, "%s: the regions span [%g, %g] along axis %d, want the unit interval", c.id.c_str(),
            lo[i], hi[i], i);
    if (g == "axisymmetric") CHECK(on_axis, "%s: no region has a side on r = 0", c.id.c_str());
    // ---- the boundaries: by `where` on a coordinate of the geometry class (or a group of an imported mesh); the
    // axis is never written
    std::set<std::string> boundary_names;
    for (const auto& b : doc["boundaries"].items()) {
      boundary_names.insert(b.key());
      const json& s = b.value();
      if (s.contains("group")) continue;
      const std::string of = s.value("of", "");
      CHECK(region_names.count(of) || (doc.contains("domains") && doc["domains"].contains(of)),
            "%s: boundary '%s' selects the unknown region or domain '%s'", c.id.c_str(), b.key().c_str(), of.c_str());
      const std::string coords = g == "planar" ? "xy" : g == "axisymmetric" ? "rz" : "xyz";
      for (const auto& w : s["where"].items()) {
        CHECK(w.key().size() == 1 && coords.find(w.key()) != std::string::npos, "%s: boundary '%s' selects on '%s'",
              c.id.c_str(), b.key().c_str(), w.key().c_str());
        CHECK(!(g == "axisymmetric" && w.key() == "r" && w.value().get<double>() == 0.0),
              "%s: boundary '%s' is the axis, which is never written", c.id.c_str(), b.key().c_str());
      }
    }
    for (const auto& kv : doc["physics"].items())
      if (kv.value().contains("boundary_conditions"))
        for (const auto& b : kv.value()["boundary_conditions"].items()) {
          CHECK(boundary_names.count(b.key()) == 1, "%s: instance '%s' has a condition on the unknown boundary '%s'",
                c.id.c_str(), kv.key().c_str(), b.key().c_str());
          if (b.value().contains("partner"))
            CHECK(boundary_names.count(b.value()["partner"].get<std::string>()) == 1, "%s: unknown periodic partner",
                  c.id.c_str());
        }
    // ---- the couplings are the row's, and only those
    std::multiset<std::string> terms;
    if (doc.contains("couplings"))
      for (const auto& e : doc["couplings"]) terms.insert(e.value("term", ""));
    CHECK(terms == std::multiset<std::string>(row->couplings.begin(), row->couplings.end()), "%s: couplings", c.id.c_str());
  }
  CHECK(n > 150, "only %d documents", n);
}

// ================================================================================================ 6c: solver settings
// 3.8: study[].solvers is per instance ({"flow": {...}}).  Every setting names the type of its instance, a type of
// the row.  The one-way rows' flow instance takes the settings of its own row (Stokes or steady Navier-Stokes, chosen
// by its equation: their limit says so), and the transport operator's linear solve takes none.  The monolithic
// Boussinesq step writes its Newton settings on the flow instance (3.8's example).
void check_solver_settings_per_instance() {
  for (const auto& r : reg().routes) {
    for (const auto& s : r.solvers)
      CHECK(has(r.physics, s.physics), "%s: solver setting %s is for '%s', not a type of the row", r.name.c_str(),
            s.name.c_str(), s.physics.c_str());
    for (const auto& t : r.solvers_of_own_row) {
      CHECK(has(r.physics, t) && !has(r.cell_types, t), "%s: %s takes the settings of its own row, so it is a type of the "
            "row that is not a cell type", r.name.c_str(), t.c_str());
      for (const auto& s : r.solvers)
        CHECK(s.physics != t, "%s: %s takes its own row's settings and has the setting %s here too", r.name.c_str(),
              t.c_str(), s.name.c_str());
    }
    if (has(r.physics, "heat.transfer"))
      for (const auto& s : r.solvers)
        CHECK(s.physics != "heat.transfer", "%s: the transport operator's linear solve takes no setting (%s)", r.name.c_str(),
              s.name.c_str());
  }
  for (const char* n : {"flow_then_heat_planar", "flow_then_heat_axisymmetric"}) {
    CHECK(find_row(n)->solvers.empty(), "%s: the one-way row has no solver setting of its own", n);
    CHECK(find_row(n)->solvers_of_own_row == Strs{"fluid.incompressible"}, "%s: the flow instance takes its row's settings", n);
  }
  for (const char* n : {"boussinesq_planar", "boussinesq_axisymmetric", "navier_stokes_steady_planar",
                        "navier_stokes_steady_axisymmetric", "navier_stokes_transient_planar"}) {
    CHECK(!find_row(n)->solvers.empty(), "%s: Newton settings", n);
    for (const auto& s : find_row(n)->solvers)
      CHECK(s.physics == "fluid.incompressible", "%s: %s is for '%s', want the flow instance", n, s.name.c_str(), s.physics.c_str());
    CHECK(find_row(n)->solvers_of_own_row.empty(), "%s: takes its own settings", n);
  }
  for (const char* n : {"current_distribution_planar", "current_distribution_axisymmetric"})
    for (const auto& s : find_row(n)->solvers)
      CHECK(s.physics == "electrochem.current_distribution", "%s: %s is for '%s'", n, s.name.c_str(), s.physics.c_str());
}

// ================================================================================================ 6d: conditional uniform
// 8.3, "Force outputs depend on the route": on the Stokes rows "those rows list fluid.mu_Pa_s as uniform when
// viscous_form is "laplacian" and a force output is declared" (the engine's physical reaction with the Laplacian form
// needs a constant viscosity); "the uniform-viscosity requirement is kept on the Navier-Stokes rows with the Laplacian
// form too, as a rule of this layer".  No other row has a conditional uniform property.
void check_conditional_uniform_viscosity() {
  for (const auto& r : reg().routes) {
    const bool stokes = r.name.rfind("stokes_", 0) == 0, ns = r.name.rfind("navier_stokes_", 0) == 0;
    size_t n = 0;
    for (const auto& kv : r.uniform_when) n += kv.second.size();
    if (!stokes && !ns) {
      CHECK(n == 0, "%s: %zu conditional uniform properties, the note gives none", r.name.c_str(), n);
      continue;
    }
    const auto it = r.uniform_when.find("fluid.incompressible");
    CHECK(n == 1 && it != r.uniform_when.end(), "%s: the conditional uniform viscosity", r.name.c_str());
    if (n != 1 || it == r.uniform_when.end()) continue;
    const auto& u = it->second.at(0);
    CHECK(u.property == "fluid.mu_Pa_s" && u.option == "viscous_form=laplacian" && u.output == "force",
          "%s: conditional uniform '%s' with '%s' and output '%s'", r.name.c_str(), u.property.c_str(), u.option.c_str(),
          u.output.c_str());
    if (stokes)
      CHECK(u.why.rfind("engine_refuses", 0) == 0, "%s: the Stokes engine refuses it (why '%s')", r.name.c_str(), u.why.c_str());
    else
      CHECK(u.why == "layer_rule: force_with_laplacian_form", "%s: a rule of this layer (why '%s')", r.name.c_str(),
            u.why.c_str());
    // the unconditional uniform list is not where it goes (8.5: "none uniform" on the Stokes rows)
    for (const auto& kv : r.uniform)
      CHECK(!has(kv.second, "fluid.mu_Pa_s"), "%s: the viscosity is uniform only under the condition", r.name.c_str());
  }
}

// ================================================================================================ 7: chemistry engine
using namespace axmp::chem;

// A structured rectangle [x0, x0 + Lx] x [0, Ly] of nx x ny squares, each split in two triangles.  Boundary attributes:
// 1 the left side, 2 the right, 3 the bottom, 4 the top.  Written here, from the geometry; it is not the registry's.
MeshInput rect(int nx, int ny, double Lx, double Ly, double x0 = 0.0) {
  MeshInput m;
  auto id = [&](int i, int j) { return i * (ny + 1) + j; };
  for (int i = 0; i <= nx; ++i)
    for (int j = 0; j <= ny; ++j) {
      m.vertices.push_back(x0 + Lx * i / nx);
      m.vertices.push_back(Ly * j / ny);
    }
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j) {
      const int a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      m.triangles.insert(m.triangles.end(), {a, b, c, a, c, d});
      m.element_attr.insert(m.element_attr.end(), {1, 1});
    }
  for (int j = 0; j < ny; ++j) {
    m.boundary.insert(m.boundary.end(), {id(0, j), id(0, j + 1), id(nx, j), id(nx, j + 1)});
    m.boundary_attr.insert(m.boundary_attr.end(), {1, 2});
  }
  for (int i = 0; i < nx; ++i) {
    m.boundary.insert(m.boundary.end(), {id(i, 0), id(i + 1, 0), id(i, ny), id(i + 1, ny)});
    m.boundary_attr.insert(m.boundary_attr.end(), {3, 4});
  }
  return m;
}

// the same mesh with the mid-edge node of the first bottom edge moved by 10 % of the edge length in -y (a curved
// boundary edge: isoparametric P2)
MeshInput rect_with_edge_node(int nx, int ny, double Lx, double Ly) {
  MeshInput m = rect(nx, ny, Lx, Ly);
  const int a = 0, b = (ny + 1);  // vertices (0,0) and (1,0)
  m.edge_node_vertices = {a, b};
  const double h = Lx / nx;
  m.edge_node_xy = {0.5 * h, -0.1 * h};
  return m;
}

struct Outcome {
  bool threw = false;
  std::string where;   // "constructor" | "add_electrode" | "solve"
  std::string text;
  bool converged = false;
};

const double kappa = 1.0;
const double Lx = 0.01, Ly = 0.005;

Reaction bv_anode() { return Reaction{Reaction::Kind::BV, 20.0, 0.05, 0.3, 0.7, 0.0}; }
Reaction bv_cathode() { return Reaction{Reaction::Kind::BV, 5.0, -0.02, 0.6, 0.4, 0.0}; }

// the electrode pair used by the scenarios: planar, the left and right sides; axisymmetric, the bottom and top (the
// left side is the axis r = 0, which carries no condition)
struct Pair { int a, c; };
Pair pair_for(bool axi) { return axi ? Pair{3, 4} : Pair{1, 2}; }

// the 3D total current that a current density i produces on the electrode of attribute attr (axisymmetric), or per
// metre of depth (planar)
double total_current(bool axi, double i) { return axi ? i * M_PI * Lx * Lx : i * Ly; }

enum class Scenario { potentiostatic_bv, galvanostatic_beside_potentiostatic, primary, normal_current, insulation, every_galvanostatic };

Outcome run_scenario(Scenario sc, int order, bool axi, bool edge_nodes) {
  Outcome out;
  std::unique_ptr<CurrentDistribution> cd;
  try {
    MeshInput m = edge_nodes ? rect_with_edge_node(6, 3, Lx, Ly) : rect(6, 3, Lx, Ly);
    cd.reset(new CurrentDistribution(m, order, axi, 2 * order + 4, 2 * order + 4));
  } catch (const std::exception& e) {
    out.threw = true; out.where = "constructor"; out.text = e.what();
    return out;
  }
  try {
    const Pair p = pair_for(axi);
    cd->set_conductivity_by_attribute({{1, kappa}});
    ElectrodeSpec a, c;
    a.name = "anode"; a.attrs = {p.a}; c.name = "cathode"; c.attrs = {p.c};
    switch (sc) {
      case Scenario::potentiostatic_bv:
        a.reactions = {bv_anode()}; a.phi_s = 0.6;
        c.reactions = {bv_cathode()}; c.phi_s = 0.0;
        cd->add_electrode(a); cd->add_electrode(c);
        break;
      case Scenario::galvanostatic_beside_potentiostatic:
        a.reactions = {bv_anode()}; a.galvanostatic = true; a.current = total_current(axi, 1.0);  // i = 1 A/m^2
        c.reactions = {bv_cathode()}; c.phi_s = 0.0;
        cd->add_electrode(a); cd->add_electrode(c);
        break;
      case Scenario::primary:
        a.primary = true; a.E_eq = 0.05; a.phi_s = 0.3;
        c.primary = true; c.E_eq = -0.02; c.phi_s = 0.0;
        cd->add_electrode(a); cd->add_electrode(c);
        break;
      case Scenario::normal_current: {
        // one grounded primary electrode on the bottom (attr 3) and a prescribed normal current into the domain
        // through the top (attr 4): the insulator flux q of the engine
        ElectrodeSpec g;
        g.name = "ground"; g.attrs = {3}; g.primary = true; g.E_eq = 0.0; g.phi_s = 0.0;
        cd->add_electrode(g);
        const Eigen::MatrixXd pts = cd->boundary_points({4});
        cd->set_insulator_flux(4, Eigen::VectorXd::Constant(pts.rows(), 0.7));
        break;
      }
      case Scenario::insulation:
        // primary on the two electrode sides, nothing on the others: no call for `insulation`
        a.primary = true; a.E_eq = 0.0; a.phi_s = 0.3;
        c.primary = true; c.E_eq = 0.0; c.phi_s = 0.0;
        cd->add_electrode(a); cd->add_electrode(c);
        break;
      case Scenario::every_galvanostatic:
        a.reactions = {bv_anode()}; a.galvanostatic = true; a.current = total_current(axi, 1.0);
        c.reactions = {bv_cathode()}; c.galvanostatic = true; c.current = -total_current(axi, 1.0);
        cd->add_electrode(a); cd->add_electrode(c);
        break;
    }
  } catch (const std::exception& e) {
    out.threw = true; out.where = "add_electrode"; out.text = e.what();
    return out;
  }
  try {
    const SolveReport rep = cd->solve(NewtonOptions{});
    out.converged = rep.converged;
  } catch (const std::exception& e) {
    out.threw = true; out.where = "solve"; out.text = e.what();
    return out;
  }
  // checks that need the solved problem
  if (sc == Scenario::normal_current) {
    // conservation: the whole prescribed flux 0.7 A/m^2 over the top side leaves through the grounded electrode.
    // The reference is the area of the top side (planar: Lx per metre of depth; axisymmetric: per radian, r dr: Lx^2/2).
    // Tolerance 1e-9 relative: a direct (LDL^T) solve of a linear problem with about 100 unknowns; the residual of the
    // balance is round-off times the condition number of the Laplacian, 1e-13 observed on the unit rectangle.
    const double area = axi ? 0.5 * Lx * Lx : Lx;
    double I = 0.0;
    for (int d : cd->electrode_dofs(0)) I += cd->residual_full(0)(d);
    CHECK(std::abs(std::abs(I) - 0.7 * area) < 1e-9 * 0.7 * area, "normal_current: flux through the ground %.15g, prescribed %.15g",
          std::abs(I), 0.7 * area);
  }
  if (sc == Scenario::insulation && !edge_nodes) {
    // the field of two primary electrodes is linear between them and exactly representable (P1 and P2), so the
    // normal derivative on the unassigned sides is zero to round-off: the unassigned attribute is an insulator.
    // Insulating attributes: planar 3 and 4, axisymmetric 2 (the cylinder wall; the axis, attr 1, is natural too).
    const std::vector<int> ins = axi ? std::vector<int>{2, 1} : std::vector<int>{3, 4};
    const double scale = 0.3 / (axi ? Ly : Lx);   // |grad phi| of the linear field
    for (int attr : ins) {
      const auto q = cd->boundary_quadrature({attr}, 4);
      double worst = 0.0;
      for (int k = 0; k < q.w.size(); ++k) worst = std::max(worst, std::abs(q.grad(k, 0) * q.normal(k, 0) + q.grad(k, 1) * q.normal(k, 1)));
      CHECK(worst < 1e-9 * scale, "insulation: normal derivative %.3g on the unassigned attribute %d (scale %.3g)", worst, attr, scale);
    }
  }
  return out;
}

void check_chemistry_cells_against_current_distribution() {
  const auto all = axmp::model::matrix_cases();
  struct Mesh { const char* tag; int order; bool edge; };
  const Mesh meshes[] = {{"P1 straight", 1, false}, {"P2 straight", 2, false}, {"P2 with edge nodes", 2, true}};
  for (bool axi : {false, true}) {
    const char* row = axi ? "current_distribution_axisymmetric" : "current_distribution_planar";
    for (const auto& m : meshes) {
      // accepted: construct, add_electrode, solve converges
      struct Acc { Scenario sc; const char* name; };
      const Acc accepted[] = {{Scenario::potentiostatic_bv, "electrode potentiostatic"},
                              {Scenario::galvanostatic_beside_potentiostatic, "electrode galvanostatic beside a potentiostatic one"},
                              {Scenario::primary, "electrode primary"},
                              {Scenario::normal_current, "normal_current through set_insulator_flux"},
                              {Scenario::insulation, "insulation (no call)"}};
      for (const auto& a : accepted) {
        const Outcome o = run_scenario(a.sc, m.order, axi, m.edge);
        CHECK(!o.threw, "%s, %s, %s: the engine refused it in %s: %s", row, m.tag, a.name, o.where.c_str(), o.text.c_str());
        CHECK(o.converged, "%s, %s, %s: not converged", row, m.tag, a.name);
      }
      // refused: every electrode galvanostatic, in the lazy layout that solve runs
      {
        const Outcome o = run_scenario(Scenario::every_galvanostatic, m.order, axi, m.edge);
        CHECK(o.threw && o.where == "solve" && o.text.find("no grounded electrode") != std::string::npos,
              "%s, %s: every electrode galvanostatic: threw=%d at %s: '%s'", row, m.tag, (int)o.threw, o.where.c_str(), o.text.c_str());
      }
    }
    // refused: P1 with edge nodes, in the constructor
    {
      const Outcome o = run_scenario(Scenario::primary, 1, axi, true);
      CHECK(o.threw && o.where == "constructor" && o.text.find("curved edge nodes need order 2") != std::string::npos,
            "%s: P1 with edge nodes: threw=%d at %s: '%s'", row, (int)o.threw, o.where.c_str(), o.text.c_str());
    }
    // the cells of the case list that have an engine column on this row are exactly these, with these classes and calls
    std::map<std::string, std::pair<CellValue, std::string>> want = {
        {"condition_kind=electrode", {CellValue::offered, ""}},
        {"condition_kind=insulation", {CellValue::offered, ""}},
        {"condition_kind=normal_current", {CellValue::offered, ""}},
        {"condition_kind=electrode:every_galvanostatic", {CellValue::engine_refuses, "layout"}},
        {"geometry_order=order=1", {CellValue::engine_refuses, "constructor"}},
        {"option_value=order=1", {CellValue::offered, ""}},
        {"option_value=order=2", {CellValue::offered, ""}},
        {"option_value=kinetics=primary", {CellValue::offered, ""}},
        {"option_value=kinetics=secondary", {CellValue::offered, ""}},
    };
    std::map<std::string, int> found;
    for (const auto& c : all) {
      if (c.row != row || c.meshes.empty()) continue;
      CHECK(c.engine == "chem::CurrentDistribution", "%s: engine '%s'", c.id.c_str(), c.engine.c_str());
      const std::string key = c.axis + "=" + c.value;
      auto it = want.find(key);
      CHECK(it != want.end(), "%s: an engine cell the chemistry column does not know", c.id.c_str());
      if (it == want.end()) continue;
      ++found[key];
      CHECK(c.cls == it->second.first, "%s: class %s", c.id.c_str(), to_str(c.cls).c_str());
      if (!it->second.second.empty()) CHECK(c.call == it->second.second, "%s: call '%s', the engine refuses it in '%s'", c.id.c_str(), c.call.c_str(), it->second.second.c_str());
    }
    for (const auto& kv : want) CHECK(found[kv.first] == 1, "%s: %d cases of %s with an engine column, want 1", row, found[kv.first], kv.first.c_str());
    // the P1 / geometry cell runs on the mesh with edge nodes, the others on straight meshes
    const auto g = cases_of(all, row, "geometry_order", "order=1");
    CHECK(g.size() == 1 && has(g[0]->meshes, "nodes"), "%s: the geometry-order cell runs on the mesh with nodes", row);
  }
}

// The engine column driven by the case list: every chemistry case with an engine column is run on each mesh it lists
// ("base": the straight rectangle; "nodes": the same with one curved edge node), at the field order its document
// writes (2, or the order of an order cell), through the scenarios that express it, and the engine's answer must be
// the case's class: an offered case is constructed, laid out and solved to convergence; a refused case is refused
// in its call (the lazy layout runs in solve) with its text.
void check_chemistry_cases_on_their_meshes() {
  const auto all = axmp::model::matrix_cases();
  int runs = 0;
  for (const auto& c : all) {
    if (c.engine != "chem::CurrentDistribution" || c.meshes.empty()) continue;
    const bool axi = c.row == "current_distribution_axisymmetric";
    CHECK(axi || c.row == "current_distribution_planar", "%s: a chemistry engine cell on the row '%s'", c.id.c_str(), c.row.c_str());
    std::vector<Scenario> sc;
    int order = 2;
    const std::string key = c.axis + "=" + c.value;
    if (key == "condition_kind=electrode")
      sc = {Scenario::potentiostatic_bv, Scenario::galvanostatic_beside_potentiostatic, Scenario::primary};
    else if (key == "condition_kind=insulation") sc = {Scenario::insulation};
    else if (key == "condition_kind=normal_current") sc = {Scenario::normal_current};
    else if (key == "condition_kind=electrode:every_galvanostatic") sc = {Scenario::every_galvanostatic};
    else if (key == "option_value=kinetics=primary") sc = {Scenario::primary};
    else if (key == "option_value=kinetics=secondary") sc = {Scenario::potentiostatic_bv, Scenario::galvanostatic_beside_potentiostatic};
    else if (key == "option_value=order=1") {
      sc = {Scenario::potentiostatic_bv, Scenario::primary};
      order = 1;
    }
    else if (key == "option_value=order=2") {
      sc = {Scenario::potentiostatic_bv, Scenario::primary};
      order = 2;
    }
    else if (key == "geometry_order=order=1") {
      sc = {Scenario::primary};
      order = 1;
    }
    else {
      CHECK(false, "%s: a chemistry cell that the engine column cannot run", c.id.c_str());
      continue;
    }
    for (const auto& m : c.meshes) {
      CHECK(m == "base" || m == "nodes", "%s: the chemistry engine takes the base mesh and the mesh with nodes, not '%s'",
            c.id.c_str(), m.c_str());
      if (m != "base" && m != "nodes") continue;
      for (Scenario s : sc) {
        ++runs;
        const Outcome o = run_scenario(s, order, axi, m == "nodes");
        if (c.cls == CellValue::offered) {
          CHECK(!o.threw && o.converged, "%s on the %s mesh, P%d: offered, but the engine %s%s%s", c.id.c_str(), m.c_str(),
                order, o.threw ? "refused it in " : "did not converge", o.threw ? o.where.c_str() : "",
                o.threw ? (": " + o.text).c_str() : "");
        } else if (c.cls == CellValue::engine_refuses) {
          const std::string where = c.call == "layout" ? "solve" : c.call;
          CHECK(o.threw && o.where == where && o.text.find(c.engine_text) != std::string::npos,
                "%s on the %s mesh, P%d: refused in %s with '%s', but threw=%d at '%s': '%s'", c.id.c_str(), m.c_str(), order,
                c.call.c_str(), c.engine_text.c_str(), (int)o.threw, o.where.c_str(), o.text.c_str());
        } else {
          CHECK(false, "%s: class %s has no engine column", c.id.c_str(), to_str(c.cls).c_str());
        }
      }
    }
  }
  CHECK(runs >= 30, "only %d chemistry engine runs from the case list", runs);
}

// ================================================================================================ the validation column
// For every case with a document: an offered cell is accepted (no error); every other class is refused with the case's
// code and no other error, since the document varies its row's base document in the cell only.  An element-type cell
// is on an imported mesh, whose element types are known when the file is read: validation defers the check (a
// `deferred` finding with the case's code) and reports no error.  The water-and-oil heat cells of sections 8.5 and 18
// (two densities, two heat capacities, on the prescribed-advection and one-way rows) carry the hint of section 8.5.
void check_validation_column() {
  int documents = 0, checked = 0, water_oil = 0, threw = 0;
  for (const auto& c : axmp::model::matrix_cases()) {
    const json doc = axmp::model::minimal_document(c);
    if (!doc.is_object()) continue;
    ++documents;
    axmp::model::ValidationReport r;
    try {
      r = axmp::model::validate_model(doc);
    } catch (const std::exception& e) {
      if (++threw <= 3) CHECK(false, "%s: validate_model threw: %s", c.id.c_str(), e.what());
      continue;
    }
    ++checked;
    std::string errors;
    for (const auto& e : r.errors) errors += e.code() + " at " + e.where() + "; ";
    auto deferred_with = [&](const std::string& code) {
      for (const auto& e : r.deferred)
        if (e.code() == code) return true;
      return false;
    };
    if (c.cls == CellValue::offered) {
      CHECK(r.errors.empty(), "%s: offered, and validation refuses it: %s", c.id.c_str(), errors.c_str());
      if (c.axis == "element_type")
        CHECK(!deferred_with("model.element_type_not_offered"), "%s: an offered element type defers no element-type check", c.id.c_str());
      continue;
    }
    CHECK(!c.code.empty(), "%s: a refused cell names its code", c.id.c_str());
    if (c.axis == "element_type") {
      CHECK(r.errors.empty(), "%s: the element types of an imported mesh are not an error before the file is read: %s", c.id.c_str(),
            errors.c_str());
      CHECK(deferred_with(c.code), "%s: the element-type check is deferred with %s", c.id.c_str(), c.code.c_str());
      continue;
    }
    CHECK(!r.errors.empty(), "%s: %s, and validation accepts it", c.id.c_str(), axmp::model::to_string(c.cls));
    for (const auto& e : r.errors)
      CHECK(e.code() == c.code, "%s: refused with %s at %s, expected %s only", c.id.c_str(), e.code().c_str(), e.where().c_str(),
            c.code.c_str());
    const bool heat_row = c.row.rfind("heat_prescribed_advection_", 0) == 0 || c.row.rfind("flow_then_heat_", 0) == 0;
    if (c.axis == "uniform" && heat_row && c.physics == "heat.transfer" && (c.value == "density" || c.value == "thermal.cp_J_kgK")) {
      ++water_oil;
      for (const auto& e : r.errors) {
        CHECK(e.where() == "/physics/heat/domains", "%s: where %s", c.id.c_str(), e.where().c_str());
        CHECK(e.hint().find("he transport operator advects with one capacity rho cp per solve") != std::string::npos,
              "%s: the hint of section 8.5, got: %s", c.id.c_str(), e.hint().c_str());
        const json d = e.details().empty() ? json() : json::parse(e.details());
        CHECK(d.is_object() && d.value("property", "") == c.value && d.value("route", "") == c.row, "%s: details %s", c.id.c_str(),
              e.details().c_str());
      }
    }
  }
  CHECK(documents > 0 && checked == documents, "validation column: %d of %d documents validated", checked, documents);
  CHECK(water_oil == 8, "validation column: %d water-and-oil heat cells (two properties, two rows, two geometries: 8)", water_oil);
}

}  // namespace

int main() {
  check_rows_and_requirements();
  check_catalogue_says_not_implemented_yet();
  check_solver_settings();
  check_types();
  check_terms();
  check_beta_types_list_what_is_not_implemented();
  check_route_requirements();
  check_every_2d_cell_has_one_value();
  check_cells_match_the_note();
  check_option_rules_and_row_selectors();
  check_not_exposed_cells_carry_a_reason_and_a_roadmap_item();
  check_engine_refuses_cells_name_the_call_and_the_text();
  check_engine_cannot_express_cells_name_the_member();
  check_layer_rules_are_listed();
  check_case_meshes_and_ranks();
  check_every_validation_cell_has_a_document();
  check_documents_express_their_cells();
  check_solver_settings_per_instance();
  check_conditional_uniform_viscosity();
  check_chemistry_cells_against_current_distribution();
  check_chemistry_cases_on_their_meshes();
  check_validation_column();
  check_no_factory_is_registered();   // last: it registers a dummy factory
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
