// The findings sink of the model path, its codes and the builders of its message texts.  Internal: not installed.
//
// Every text that the parser and the validation of a model document write with a code is built here, so that the texts
// of one family are in one place and a wording is changed once.  The codes are the constants of namespace codes, each
// with a row in code_table(); no other source of the model path writes a code as a literal.  The texts that the
// electromagnetic parser also has (a missing field, an unknown field with its allowed list, an unknown material) are not
// here: the model path raises those through the same helpers as that parser, so that they are the same text.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_selection.hpp"

namespace axmp::model {

// A finding of the model path.  Only an `error` makes a document invalid; the other kinds say what the validation that
// follows the parse will do (unavailable: a route that this build does not have; deferred: a check that needs the mesh or
// the geometry; warning: a document that is valid but probably not what was meant).
struct Finding {
  enum class Kind { error, unavailable, deferred, warning } kind;
  Error error;
};

// The one sink of the model path, in the order in which findings are added.
class Findings {
 public:
  void add(Finding::Kind kind, Error error) { all_.push_back(Finding{kind, std::move(error)}); }
  bool has_errors() const {
    for (const Finding& f : all_)
      if (f.kind == Finding::Kind::error) return true;
    return false;
  }
  // The first finding of kind error; the caller has checked has_errors().
  const Error& first_error() const {
    for (const Finding& f : all_)
      if (f.kind == Finding::Kind::error) return f.error;
    throw Error("no error was recorded", "", "Check has_errors() first.");
  }
  const std::vector<Finding>& all() const { return all_; }

 private:
  std::vector<Finding> all_;
};

namespace codes {
inline constexpr const char* em_sections_not_combinable = "model.em_sections_not_combinable";
inline constexpr const char* symmetry_not_offered = "model.symmetry_not_offered";
inline constexpr const char* mesh_order = "model.mesh_order";
inline constexpr const char* geometry_order_from_file = "model.geometry_order_from_file";
inline constexpr const char* exterior_not_offered = "model.exterior_not_offered";
inline constexpr const char* mesh_key_not_offered = "model.mesh_key_not_offered";
inline constexpr const char* per_depth_key = "model.per_depth_key";
inline constexpr const char* name_collision = "boundary.name_collision";
inline constexpr const char* boundary_unknown_name = "boundary.unknown_name";
inline constexpr const char* boundary_reserved_name = "boundary.reserved_name";
inline constexpr const char* boundary_selector_grammar = "boundary.selector_grammar";
inline constexpr const char* boundary_cycle = "boundary.cycle";
inline constexpr const char* imported_mesh_selector = "boundary.imported_mesh_selector";
inline constexpr const char* domain_unknown_name = "domain.unknown_name";
inline constexpr const char* physics_invalid_id = "physics.invalid_id";
inline constexpr const char* physics_reserved_id = "physics.reserved_id";
inline constexpr const char* physics_unknown_type = "physics.unknown_type";
inline constexpr const char* physics_unknown_condition_kind = "physics.unknown_condition_kind";
inline constexpr const char* physics_unknown_option = "physics.unknown_option";
inline constexpr const char* physics_unknown_source_kind = "physics.unknown_source_kind";
inline constexpr const char* expr_unknown_variable = "expr.unknown_variable";
inline constexpr const char* expr_vector_length = "expr.vector_length";
inline constexpr const char* expr_parameter_shadows_variable = "expr.parameter_shadows_variable";
inline constexpr const char* coupling_unknown_term = "coupling.unknown_term";
inline constexpr const char* coupling_unknown_physics = "coupling.unknown_physics";
inline constexpr const char* coupling_type_mismatch = "coupling.type_mismatch";
inline constexpr const char* coupling_wrong_selection_key = "coupling.wrong_selection_key";
inline constexpr const char* study_type_reserved = "study.type_reserved";
inline constexpr const char* study_unknown_physics = "study.unknown_physics";
inline constexpr const char* study_initial_from_unknown_step = "study.initial_from_unknown_step";
inline constexpr const char* study_solver_value = "study.solver_value";
inline constexpr const char* output_unknown_kind = "output.unknown_kind";
inline constexpr const char* output_grammar = "output.grammar";
inline constexpr const char* output_unknown_quantity = "output.unknown_quantity";
inline constexpr const char* output_unknown_physics = "output.unknown_physics";
inline constexpr const char* output_unknown_boundary = "output.unknown_boundary";
inline constexpr const char* output_cycle = "output.cycle";
inline constexpr const char* reaction_unknown_kind = "reaction.unknown_kind";
inline constexpr const char* reaction_unknown_name = "reaction.unknown_name";
inline constexpr const char* material_block_incomplete = "material.block_incomplete";
inline constexpr const char* material_expansion_ambiguous = "material.expansion_ambiguous";
inline constexpr const char* material_conductivity_ambiguous = "material.conductivity_ambiguous";
inline constexpr const char* law_range_missing = "law.range_missing";
inline constexpr const char* law_unknown_kind = "law.unknown_kind";
inline constexpr const char* law_unknown_field = "law.unknown_field";
inline constexpr const char* law_unknown_option = "law.unknown_option";
inline constexpr const char* law_table_invalid = "law.table_invalid";
// the route-level checks of the validation
inline constexpr const char* geometry_order_exceeds_field_order = "model.geometry_order_exceeds_field_order";
inline constexpr const char* geometry_order_not_offered = "model.geometry_order_not_offered";
inline constexpr const char* geometry_order_lowered = "mesh.geometry_order_lowered";
inline constexpr const char* element_type_not_offered = "model.element_type_not_offered";
inline constexpr const char* beta_feature = "model.beta_feature";
inline constexpr const char* route_not_implemented = "route.not_implemented";
inline constexpr const char* route_law_not_offered = "route.law_not_offered";
inline constexpr const char* route_domains_must_coincide = "route.domains_must_coincide";
inline constexpr const char* physics_material_block_missing = "physics.material_block_missing";
inline constexpr const char* physics_domains_overlap = "physics.domains_overlap";
inline constexpr const char* physics_property_not_uniform = "physics.property_not_uniform_over_domain";
inline constexpr const char* fluid_condition_not_offered = "fluid.condition_not_offered";
inline constexpr const char* fluid_option_not_offered = "fluid.option_not_offered";
inline constexpr const char* fluid_outlet_stress_form = "fluid.outlet_stress_form";
inline constexpr const char* fluid_pseudo_traction = "fluid.pseudo_traction_on_curved_boundary";
inline constexpr const char* fluid_gravity_direction = "fluid.gravity_direction";
inline constexpr const char* fluid_buoyancy_needs_beta = "fluid.buoyancy_needs_beta";
inline constexpr const char* heat_condition_not_offered = "heat.condition_not_offered";
inline constexpr const char* heat_option_not_offered = "heat.option_not_offered";
inline constexpr const char* heat_supg_needs_advection = "heat.supg_needs_advection";
inline constexpr const char* heat_prescribed_advection_partial = "heat.prescribed_advection_partial";
inline constexpr const char* heat_data_not_regionwise = "heat.data_not_regionwise";
inline constexpr const char* heat_two_advections = "heat.two_advections";
inline constexpr const char* electrochem_potential_level_free = "electrochem.potential_level_free";
inline constexpr const char* output_force_needs_stress_form = "output.force_needs_stress_form";
inline constexpr const char* study_type_not_offered = "study.type_not_offered";
inline constexpr const char* study_strategy_not_offered = "study.strategy_not_offered";
inline constexpr const char* study_two_way_needs_strategy = "study.two_way_needs_strategy";
inline constexpr const char* study_field_not_rampable = "study.field_not_rampable";
inline constexpr const char* study_ramp_end_mismatch = "study.ramp_end_mismatch";
inline constexpr const char* coupling_empty_overlap = "coupling.empty_overlap";
inline constexpr const char* coupling_restriction_not_offered = "coupling.restriction_not_offered";
inline constexpr const char* coupling_uncoupled_pair = "coupling.uncoupled_pair";
inline constexpr const char* law_field_not_provided = "law.field_not_provided";
inline constexpr const char* coupling_selection_outside_support = "coupling.selection_outside_support";
inline constexpr const char* physics_source_outside_domain = "physics.source_outside_domain";
inline constexpr const char* output_force_not_offered = "output.force_not_offered";
inline constexpr const char* study_scale_not_applicable = "study.scale_not_applicable";
// the checks that need the resolved boundaries: reported as deferred by the validation, errors of the check on the
// resolved boundaries
inline constexpr const char* boundary_empty_selection = "boundary.empty_selection";
inline constexpr const char* physics_boundary_uncovered = "physics.boundary_uncovered";
// the other refusals of the partition rule
inline constexpr const char* physics_boundary_double = "physics.boundary_double";
inline constexpr const char* physics_boundary_interior = "physics.boundary_interior";
// the other findings of the check on the resolved boundaries
inline constexpr const char* physics_boundary_outside_domain = "physics.boundary_outside_domain";
inline constexpr const char* boundary_partial_match = "boundary.partial_match";
inline constexpr const char* boundary_between_overlap = "boundary.between_overlap";
inline constexpr const char* output_boundary_not_in_domain = "output.boundary_not_in_domain";
inline constexpr const char* output_force_on_interior = "output.force_on_interior";
inline constexpr const char* physics_option_not_offered = "physics.option_not_offered";
inline constexpr const char* route_single_rank = "route.single_rank";
inline constexpr const char* route_table_out_of_date = "internal.route_table_out_of_date";
// the internal error of the meshed model and of the exporters of an instance mesh (model_mesh.hpp)
inline constexpr const char* mesh_export = "internal.mesh_export";
// meshing a model document: an installation whose mesher does not run, a failed mesher run, a shape the bounded modes
// cannot mesh
inline constexpr const char* requires_mesher = "build.requires_mesher";
inline constexpr const char* requires_fluids = "build.requires_fluids";
inline constexpr const char* mesh_generation_failed = "mesh.generation_failed";
inline constexpr const char* shape_not_offered = "model.shape_not_offered";
// a 2D curved mesh with a tangled element, refused where the file is read (mesh_quality::require_untangled), on the model
// path and for electromagnetic designs alike
inline constexpr const char* mesh_tangled = "mesh.tangled";
inline constexpr const char* patch_operation_failed = "patch.operation_failed";
// the switch between the two parsers: the guard of the electromagnetic parser, the refusal of a model document by an
// operation on electromagnetic designs, a key repeated in a model document's file
inline constexpr const char* section_needs_physics = "model.section_needs_physics";
inline constexpr const char* em_only_operation = "model.em_only_operation";
inline constexpr const char* duplicate_key = "json.duplicate_key";
// an arc path of a curve: more than one turn, or below the axis of an axisymmetric design
inline constexpr const char* arc_span = "geometry.arc_span";
inline constexpr const char* arc_degenerate = "geometry.arc_degenerate";
inline constexpr const char* arc_negative_r = "geometry.arc_negative_r";
}  // namespace codes

// One row of the table of codes: the finding kinds it is raised as (space separated, out of error, unavailable,
// deferred, warning), one sentence, and whether this version raises it or only names it.
struct CodeInfo {
  const char* code;
  const char* kinds;
  const char* summary;
  bool reserved;
};

// The table of the codes of the model path, one row per constant of namespace codes.  The reference page
// docs/reference/error_codes.md is generated from it.
inline std::vector<CodeInfo> code_table() {
  namespace c = codes;
  return {
      {c::em_sections_not_combinable, "error", "An electromagnetic analysis section in a document that has a physics section.", false},
      {c::symmetry_not_offered, "error", "The symmetry \"xsec\" in a model document.", false},
      {c::mesh_order, "error", "The key mesh.order in a model document, where the field order is an option of each physics and the geometry order is mesh.geometry_order.", false},
      {c::geometry_order_from_file, "error", "The key mesh.geometry_order beside mesh.file: the geometry order of an imported mesh is the file's.", false},
      {c::exterior_not_offered, "error", "The value \"open\" of mesh.exterior in a model document.", false},
      {c::mesh_key_not_offered, "error", "A mesh key of an electromagnetic design that a model document does not offer (adapt).", false},
      {c::per_depth_key, "error", "An absolute input in the spelling of the other geometry: a total in a planar model, or a quantity per metre of depth in an axisymmetric or 3D one.", false},
      {c::name_collision, "error", "One name used by two of the regions, curves, domains and boundaries, which share a namespace.", false},
      {c::boundary_unknown_name, "error", "A boundary or a curve that the document does not define.", false},
      {c::boundary_reserved_name, "error", "One of the reserved names axis, exterior and remaining used where it has no meaning.", false},
      {c::boundary_selector_grammar, "error", "A boundary selector that is not one of the selector forms, or a filter on a form that takes none.", false},
      {c::boundary_cycle, "error", "Boundaries that name one another in a cycle (in a name, a union, a difference or an except).", false},
      {c::imported_mesh_selector, "deferred", "A geometric boundary selector on an imported mesh, whose boundaries are the groups of its info file; resolved when the file is read.", false},
      {c::domain_unknown_name, "error", "A region or a domain that the document does not define.", false},
      {c::physics_invalid_id, "error", "A physics id that is not a lowercase identifier.", false},
      {c::physics_reserved_id, "error", "A physics id that is kept for an electromagnetic section or a later physics type.", false},
      {c::physics_unknown_type, "error", "A physics type that is not in the catalogue; the hint gives the name to use for another code's name, or the near matches.", false},
      {c::physics_unknown_condition_kind, "error", "A condition kind that the physics type does not have.", false},
      {c::physics_unknown_option, "error", "An option, or a value of an option, that the physics type does not have.", false},
      {c::physics_unknown_source_kind, "error", "A source kind that the physics type does not have.", false},
      {c::expr_unknown_variable, "error", "An expression that names something that is not a variable of its context, a parameter or a constant; also the time t in a datum of an instance that a stationary step solves.", false},
      {c::expr_vector_length, "error", "A vector datum or a point with the wrong number of components for the symmetry.", false},
      {c::expr_parameter_shadows_variable, "error", "A parameter that has the name of a variable of a field expression.", false},
      {c::coupling_unknown_term, "error", "A coupling term that is not in the catalogue.", false},
      {c::coupling_unknown_physics, "error", "A coupling whose from or to is not a physics of the document.", false},
      {c::coupling_type_mismatch, "error", "A coupling term between physics types that it does not connect.", false},
      {c::coupling_wrong_selection_key, "error", "The key in on a boundary term, or on on a domain term.", false},
      {c::study_type_reserved, "error", "The study types eigen and frequency, which are kept for the electromagnetic analyses.", false},
      {c::study_unknown_physics, "error", "A study step that names a physics the document does not have.", false},
      {c::study_initial_from_unknown_step, "error", "An initial_from that does not name an earlier step.", false},
      {c::study_solver_value, "error", "A solver setting of a study step whose value has the wrong type: not true or false, not a whole number, or not a number.", false},
      {c::output_unknown_kind, "error", "An output kind that does not exist.", false},
      {c::output_grammar, "error", "An output whose name is not an identifier, or that lacks a key its kind needs.", false},
      {c::output_unknown_quantity, "error", "A quantity that no physics of the document publishes in this geometry.", false},
      {c::output_unknown_physics, "error", "An output that names a physics the document does not have.", false},
      {c::output_unknown_boundary, "error", "An output on a boundary that the document does not define.", false},
      {c::output_cycle, "error", "Expression outputs that read one another in a cycle.", false},
      {c::reaction_unknown_kind, "error", "An electrode reaction kind that does not exist.", false},
      {c::reaction_unknown_name, "error", "An electrode condition that names a reaction the document does not declare.", false},
      {c::material_block_incomplete, "error", "A material block without the key it needs (a thermal block without k_W_mK, a fluid block without mu_Pa_s).", false},
      {c::material_expansion_ambiguous, "error", "A material with both mechanical.alpha (the total contraction strain) and mechanical.alpha_1_K (the coefficient per kelvin).", false},
      {c::material_conductivity_ambiguous, "error", "A material with sigma_S_m at the material level and in its electric block, with different values.", false},
      {c::law_range_missing, "error", "A property law without the validity range of a field it reads.", false},
      {c::law_unknown_kind, "error", "A property law that is neither an expression nor a table.", false},
      {c::law_unknown_field, "error", "A property law of a field that no physics publishes.", false},
      {c::law_unknown_option, "error", "An interpolation or out_of_range value that a property law does not have.", false},
      {c::law_table_invalid, "error", "A table law with too few points, lists of different length, abscissae that do not increase, more than one field, or a tabulated value outside the range its property allows.", false},
      {c::geometry_order_exceeds_field_order, "error deferred", "A geometry order above the field order of an instance; deferred to the reading of the file for an imported mesh.", false},
      {c::geometry_order_not_offered, "error deferred", "Geometry order 2 with an option value that needs straight elements (SUPG of a heat instance); deferred to the reading of the file for an imported mesh.", false},
      {c::geometry_order_lowered, "warning", "The default geometry order is lowered to 1 by a P1 instance, or by an option that needs straight elements, in a model with curved boundaries.", false},
      {c::element_type_not_offered, "deferred", "An imported mesh on a route whose engine takes triangles only; checked when the file is read.", false},
      {c::beta_feature, "warning", "A physics type, a coupling term or a route of the document is beta.", false},
      {c::route_not_implemented, "unavailable", "The document is valid and this version cannot solve it: its route is not implemented yet, or no route exists for its combination of physics, coupling terms, geometry and study type; also the refusal of solving a model document, and of meshing a 3D model document, one on an imported mesh, or (in Python) one with no geometry, which this version does not do.", false},
      {c::route_law_not_offered, "error", "A property law of a field in a property that a route of this version takes as a constant.", false},
      {c::route_domains_must_coincide, "error", "Two instances of a route that solves them on one mesh have different domains.", false},
      {c::physics_material_block_missing, "error", "A region in a physics whose material lacks the data the route needs; also an instance that no region can take part in.", false},
      {c::physics_domains_overlap, "error", "Two instances of one family on a shared region.", false},
      {c::physics_property_not_uniform, "error", "A property that the route takes as one number differs between the regions of the instance.", false},
      {c::fluid_condition_not_offered, "error", "A flow condition kind that the route does not offer: the engine refuses it there, or the document path does not expose it yet.", false},
      {c::fluid_option_not_offered, "error", "A flow option value that the engine of the route refuses (the skew convective form in a transient step).", false},
      {c::fluid_outlet_stress_form, "warning", "An outlet or open boundary with the stress viscous form: zero total traction, which a developed profile does not satisfy.", false},
      {c::fluid_pseudo_traction, "warning", "An outlet, traction or slip condition on a curved boundary with the Laplacian viscous form imposes a pseudo-traction.", false},
      {c::fluid_gravity_direction, "error", "A buoyancy term whose gravity is not along the vertical axis, on the Boussinesq route.", false},
      {c::fluid_buoyancy_needs_beta, "error", "A region of a flow with the buoyancy term whose material has no fluid.beta_1_K.", false},
      {c::heat_condition_not_offered, "error", "A heat condition kind that the route's engine refuses, or that the advective form excludes.", false},
      {c::heat_option_not_offered, "error", "A heat option value that the route's engine refuses or cannot express (a P1 temperature or a conservative form on the Boussinesq route).", false},
      {c::heat_supg_needs_advection, "error", "SUPG stabilization on a heat instance that nothing advects.", false},
      {c::heat_prescribed_advection_partial, "error", "A prescribed velocity that covers part of the heat instance, or a second one.", false},
      {c::heat_data_not_regionwise, "error", "A conductivity or a heat source that differs from region to region on a route that takes one expression over the instance.", false},
      {c::heat_two_advections, "error", "A heat instance with both a prescribed velocity and an advection term.", false},
      {c::electrochem_potential_level_free, "error", "Every electrode is galvanostatic, so the potential level is free.", false},
      {c::output_force_needs_stress_form, "error deferred", "A force or torque with the Laplacian viscous form on a Navier-Stokes route, on a boundary that is not a wall at rest; deferred to the resolution of the boundaries when the boundary is not a key of the instance's conditions.", false},
      {c::output_force_not_offered, "error", "A force or torque of a flow on a route whose engine exposes no reaction (the one-way flow-then-heat and the Boussinesq routes); also a force or torque that belongs to no flow (it names a physics that is not a flow, or the document has none).", false},
      {c::study_type_not_offered, "error", "A step of a study type for which the physics has no route (a stationary step over a reactor).", false},
      {c::study_strategy_not_offered, "error", "A coupling strategy that the route does not offer; also one_way written for a two-way group.", false},
      {c::study_scale_not_applicable, "error", "A continuation scale with nothing to scale in its step: viscosity without a steady Navier-Stokes flow (no flow, a Stokes or Boussinesq flow, or a transient step), or buoyancy without a buoyancy term.", false},
      {c::study_two_way_needs_strategy, "error", "A two-way group of physics in a step without a coupling strategy, on no route that has a default.", false},
      {c::study_field_not_rampable, "error", "A ramp of a field that no engine can warm-start, or of a physics that the step does not solve.", false},
      {c::study_ramp_end_mismatch, "error", "A ramp whose last value is not the value the document gives the field.", false},
      {c::coupling_empty_overlap, "error", "A domain term whose producer and consumer share no region.", false},
      {c::coupling_restriction_not_offered, "error", "A term restricted to part of the shared domain on a route that cannot restrict it.", false},
      {c::coupling_uncoupled_pair, "warning", "Two physics act on a common region with no coupling term between them.", false},
      {c::law_field_not_provided, "error", "A property law reads a field that no physics of the step provides on the region and that the step does not give.", false},
      {c::coupling_selection_outside_support, "error", "The key in of a domain term names a region outside the shared domain of its two physics.", false},
      {c::physics_source_outside_domain, "error", "The key in of a source names a region outside the domain of its physics.", false},
      {c::boundary_empty_selection, "error deferred", "A named boundary that selects nothing: an error of the check on the resolved boundaries, which `axmp boundaries` and `axmp mesh` run; `axmp validate` does not resolve the boundaries (that needs the mesher, or the info file of an imported mesh) and reports it as deferred.", false},
      {c::physics_boundary_uncovered, "error deferred", "Part of the boundary of the domain of a physics has no condition: an error of the check on the resolved boundaries, which `axmp boundaries` and `axmp mesh` run (the partition rule: every part of that boundary carries exactly one condition); `axmp validate` reports it as deferred.", false},
      {c::physics_boundary_double, "error", "Two conditions of a physics cover the same part of the boundary of its domain (the partition rule).", false},
      {c::physics_boundary_interior, "error", "A condition of a physics on a boundary that is interior to its domain, also when only part of it is (the partition rule).", false},
      {c::physics_boundary_outside_domain, "error", "A condition of a physics on a boundary that has nothing on the boundary of its domain and nothing inside it (the partition rule).", false},
      {c::boundary_partial_match, "warning", "A where filter leaves out an edge that satisfies it in part, in a boundary that selects other edges: the filter keeps an edge only when the whole edge satisfies it.", false},
      {c::boundary_between_overlap, "error", "A between selector whose two domains share a region, where the direction of its normal is undefined.", false},
      {c::output_boundary_not_in_domain, "error", "An output of a physics on a boundary that lies outside the domain of that physics.", false},
      {c::output_force_on_interior, "error", "A force or torque on a boundary interior to the domain of its flow, where the engine has no reaction.", false},
      {c::physics_option_not_offered, "error", "An option value that a route refuses, for a physics that is neither a flow nor a heat instance; no route of this version refuses one.", true},
      {c::patch_operation_failed, "error", "An operation of a patch (RFC 6902) that cannot be applied: a path that does not exist, a test that does not hold, an unknown operation or a malformed one; the document is unchanged.", false},
      {c::route_single_rank, "unavailable", "A route that runs on one MPI rank, asked to run on several.", true},
      {c::route_table_out_of_date, "error", "An engine refused what validation accepted: a defect of the route table, not of the document.", true},
      {c::mesh_export, "error", "The meshed model or an exporter of an instance mesh met what a valid document never produces (a boundary face of an instance without an atom, mesher files that disagree with the library's resolution of the boundaries, a curved mesh for a P1 chemistry field or for a route that needs straight elements): a defect of the program, not of the document.", false},
      {c::mesh_tangled, "error", "A 2D mesh with curved (order-2) elements of which one is tangled (its Jacobian determinant changes sign, vanishes or is inverted against the other elements of its region), refused where the mesh file is read, before anything is solved; electromagnetic designs and imported mesh files raise it too.", false},
      {c::requires_fluids, "unavailable", "An instance mesh handed to the fluids engines (im.fluids_mesh()) in a build without the fluids module: a wheel from PyPI or the sequential source build; the MPI source build has it.", false},
      {c::requires_mesher, "unavailable", "A model document with a geometry in an installation whose mesher does not run: a wheel installed without the mesh extra, or a mesher whose libraries are missing; the validation reports it, and the commands that mesh refuse the document. `Design.export_gdml` refuses with it a mesher that cannot be run or loaded.", false},
      {c::mesh_generation_failed, "error", "The mesher ran and failed on the geometry of a model document: the message carries the mesher's error and the hint the path of its log.", false},
      {c::shape_not_offered, "error", "A contour with an elliptical segment in an enabled region of a model document: the bounded meshes take lines and circular arcs.", false},
      {c::section_needs_physics, "error", "A section of a model document (domains, boundaries, reactions, couplings, study or outputs) in a document without a physics section.", false},
      {c::em_only_operation, "error", "A model document given to an operation that works on electromagnetic designs only (sensitivities, optimization, the electromagnetic models, the 3D conversion).", false},
      {c::duplicate_key, "error", "A key repeated in one object of a model document's file, whose value JSON would silently replace.", false},
      {c::arc_span, "error", "An arc path whose angles span more than one turn (2 pi radians), or are not finite; the angles of an arc are in radians.", false},
      {c::arc_degenerate, "error", "An arc path with a radius that is not positive, or an arc shorter than 1e-7 m (radius times angle span), which the CAD kernel cannot build.", false},
      {c::arc_negative_r, "error", "An arc path that reaches r < 0, in any electromagnetic design but a cross-section (xsec) one (a planar design on an imported mesh included) and in an axisymmetric model document: it crosses the symmetry axis or lies beyond it.", false},
  };
}

// The builders of the message texts.
namespace msg {

using Strs = std::vector<std::string>;
using json = nlohmann::json;

inline std::string join(const Strs& v, const std::string& sep = ", ") {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
  return s;
}
inline json jarr(const Strs& v) {
  json a = json::array();
  for (const std::string& s : v) a.push_back(s);
  return a;
}
inline std::string on_geometry(const std::string& symmetry) {  // "on a planar geometry", "on an axisymmetric geometry"
  return std::string(symmetry == "axisymmetric" ? "on an " : "on a ") + symmetry + " geometry";
}
inline std::string article_model(const std::string& symmetry) {  // "a planar model", "an axisymmetric model", "a 3d model"
  return std::string(symmetry == "axisymmetric" ? "an " : "a ") + symmetry + " model";
}
inline Error coded(std::string what, std::string where, std::string hint, const char* code, const json& details = nullptr) {
  return Error(std::move(what), std::move(where), std::move(hint), code, details.is_null() ? std::string() : details.dump());
}

// ---------------------------------------------------------------------------------------------------- top level
inline Error em_section_not_combinable(const std::string& section) {
  return coded("'" + section + "' cannot be combined with 'physics' yet", "/" + section,
               "electromagnetic analyses and the physics section are solved by different engines in this version. Keep the "
               "electromagnetic design in its own document. Coupling them (deformation feedback into the electromagnetic "
               "figures of merit) is planned; `axmp physics --routes` lists what can be combined.",
               codes::em_sections_not_combinable);
}
inline Error symmetry_not_offered(const std::string& symmetry) {
  return coded("symmetry '" + symmetry + "' is not offered in a model document", "/symmetry",
               "Use \"planar\" (x, y), \"axisymmetric\" (r, z) or \"3d\". The cross-section analysis of a cavity belongs to an "
               "electromagnetic design.",
               codes::symmetry_not_offered);
}
inline Error symmetry_unknown() { return Error("symmetry must be 'planar', 'axisymmetric' or '3d'", "/symmetry"); }

// ---------------------------------------------------------------------------------------------------- mesh
inline Error mesh_order() {
  return coded("`mesh.order` is the field order of an electromagnetic solve. In a model document write the field order per "
               "physics (`/physics/flow/options/order`) and the geometry order in `mesh.geometry_order`",
               "/mesh/order", "Remove `mesh.order`; write `options.order` on a physics instance and `mesh.geometry_order` in `mesh`.",
               codes::mesh_order);
}
inline Error geometry_order_from_file() {
  return coded("the geometry order of an imported mesh is read from the file; remove `mesh.geometry_order`",
               "/mesh/geometry_order", "", codes::geometry_order_from_file);
}
inline Error exterior_not_offered() {
  return coded("mesh.exterior \"open\" is not offered in a model document", "/mesh/exterior",
               "Only \"none\" is offered: the union of the enabled regions is the whole domain, with no air and no infinity "
               "shell.",
               codes::exterior_not_offered);
}
inline Error mesh_key_not_offered(const std::string& key) {
  return coded("mesh." + key + " is not offered in a model document", "/mesh/" + key,
               "Use \"refinements\" for uniform refinement, or a smaller \"size\".", codes::mesh_key_not_offered);
}

// ---------------------------------------------------------------------------------------------------- names
inline Error name_collision(const std::string& name, const std::string& first_kind, const std::string& first_at,
                            const std::string& second_kind, const std::string& second_at) {
  std::string who;
  if (first_kind == second_kind) {
    who = "two " + first_kind + "s";
  } else {
    who = "a " + first_kind + " and a " + second_kind;
  }
  return coded("name '" + name + "' is used by " + who, second_at,
               "Regions, curves, domains and boundaries share one namespace; rename one of them.", codes::name_collision,
               {{"name", name}, {"first", {{"kind", first_kind}, {"at", first_at}}}, {"second", {{"kind", second_kind}, {"at", second_at}}}});
}
inline Error invalid_id(const std::string& id, const std::string& where) {
  return coded("'" + id + "' is not a valid physics id", where,
               "An id starts with a lowercase letter and continues with lowercase letters, digits and underscores "
               "([a-z][a-z0-9_]*).",
               codes::physics_invalid_id, {{"id", id}});
}
inline Error reserved_id(const std::string& id, const std::string& where, const Strs& reserved) {
  return coded("'" + id + "' is a reserved physics id", where,
               "The ids " + join(reserved) + " are kept for the electromagnetic sections and for later physics types; choose "
               "another id.",
               codes::physics_reserved_id, {{"id", id}, {"reserved", jarr(reserved)}});
}
inline Error unknown_boundary(const std::string& name, const std::string& where, const Strs& valid) {
  return coded("unknown boundary '" + name + "'", where,
               valid.empty() ? std::string("This document defines no boundaries.") : "The boundaries of this document are: " + join(valid) + ".",
               codes::boundary_unknown_name, {{"name", name}, {"valid", jarr(valid)}});
}
inline Error unknown_curve(const std::string& name, const std::string& where, const Strs& valid) {
  return coded("unknown curve '" + name + "'", where,
               valid.empty() ? std::string("This document defines no curves.") : "The curves of this document are: " + join(valid) + ".",
               codes::boundary_unknown_name, {{"name", name}, {"valid", jarr(valid)}});
}
inline Error unknown_domain(const std::string& name, const std::string& where, const Strs& valid) {
  return coded("unknown domain or region '" + name + "'", where,
               valid.empty() ? std::string("This document defines no regions or domains.")
                             : "The regions and domains of this document are: " + join(valid) + ".",
               codes::domain_unknown_name, {{"name", name}, {"valid", jarr(valid)}});
}
inline Error reserved_name(const std::string& name, const std::string& where) {
  if (name == "axis")
    return coded("'axis' is the part of the boundary on r = 0; it is created automatically and is never written", where,
                 "Remove it: every physics applies its one valid axis condition there.", codes::boundary_reserved_name);
  if (name == "exterior")
    return coded("'exterior' can only be the second entry of 'between'", where,
                 "Use it as the second entry of a 'between' selector, as in {\"between\": [A, \"exterior\"]}.",
                 codes::boundary_reserved_name);
  return coded("'remaining' can only be a key of 'boundary_conditions'", where, "Use it as a key of 'boundary_conditions'.",
               codes::boundary_reserved_name);
}
// A region, a curve or a domain that takes one of the names the boundary grammar keeps for itself.
inline Error reserved_name_taken(const std::string& name, const std::string& kind, const std::string& where) {
  return coded("'" + name + "' is a reserved name and cannot be the name of a " + kind, where,
               "The names axis, exterior and remaining belong to the boundary grammar: 'axis' is the boundary on r = 0, "
               "'exterior' the outside of the model in a 'between' selector, and 'remaining' the boundary that no condition "
               "names. Rename the " + kind + ".",
               codes::boundary_reserved_name, {{"name", name}, {"kind", kind}});
}

// ---------------------------------------------------------------------------------------------------- selectors
inline Error selector_grammar(const std::string& what, const std::string& where) {
  return coded(what, where, "", codes::boundary_selector_grammar);
}
inline Error boundary_cycle(const Strs& cycle, const std::string& where) {
  return coded("the boundaries form a cycle: " + join(cycle, " -> "), where,
               "A boundary may name other boundaries, in a name, a 'union', a 'difference' or an 'except', but not itself, "
               "directly or through others.",
               codes::boundary_cycle, {{"cycle", jarr(cycle)}});
}
inline Error group_needs_mesh_file(const std::string& boundary, const std::string& where) {
  return selector_grammar("boundary '" + boundary + "' selects a 'group', which is a named group of an imported mesh, and this document "
                          "has no 'mesh.file'",
                          where);
}
inline Error selector_needs_one(const std::string& where) {
  return selector_grammar("a boundary selector needs exactly one of of, between, curve, group, union, difference", where);
}
inline Error unknown_coordinate(const std::string& coord, const std::string& symmetry, const Strs& coords, const std::string& where) {
  return selector_grammar("unknown coordinate '" + coord + "' in 'where': the coordinates of " + article_model(symmetry) + " are " +
                              join(coords),
                          where);
}
inline Error imported_mesh_selector(const std::string& boundary, const std::string& key, const std::string& where) {
  return coded("boundary '" + boundary + "' uses '" + key + "' on an imported mesh, where only 'group', 'union' and "
               "'difference' are available",
               where,
               key == "of" ? std::string("Select a group of the mesh's info file with {\"group\": NAME}.")
                           : "Select a group of the mesh's info file with {\"group\": NAME}; the geometry of an imported mesh "
                             "is not described by the document.",
               codes::imported_mesh_selector);
}

// ---------------------------------------------------------------------------------------------------- catalogue
inline int edit_distance(const std::string& a, const std::string& b) {
  std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) prev[j] = static_cast<int>(j);
  for (size_t i = 1; i <= a.size(); ++i) {
    cur[0] = static_cast<int>(i);
    for (size_t j = 1; j <= b.size(); ++j)
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
    std::swap(prev, cur);
  }
  return prev[b.size()];
}
// The names of `valid` within two edits of `written` (not equal), nearest first, then in the order of the list.
inline Strs near_names(const std::string& written, const Strs& valid) {
  std::vector<std::pair<int, size_t>> hits;
  for (size_t i = 0; i < valid.size(); ++i) {
    const int d = edit_distance(written, valid[i]);
    if (d > 0 && d <= 2) hits.push_back({d, i});
  }
  std::stable_sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  Strs out;
  for (const auto& h : hits) out.push_back(valid[h.second]);
  return out;
}
inline std::string did_you_mean(const Strs& near) {
  std::string s;
  for (size_t i = 0; i < near.size(); ++i) {
    s += (i == 0 ? "" : (i + 1 == near.size() ? " or " : ", ")) + ("'" + near[i] + "'");
  }
  return "Did you mean " + s + "? ";
}
// An unknown name of a catalogue list: the hint says the alias to use, or the near matches, and then lists the names.
// `use` is the name an alias stands for ("" when the name is not an alias).
inline Error unknown_catalogue_name(const char* code, const std::string& what, const std::string& key, const std::string& written,
                                    const std::string& where, const Strs& valid, const std::string& use,
                                    const std::string& list_sentence, const std::string& lead = "") {
  const Strs near = near_names(written, valid);
  std::string hint;
  if (!use.empty()) {
    hint = "Use '" + use + "' (the name here for '" + written + "'). ";
  } else if (!near.empty()) {
    hint = did_you_mean(near);
  }
  json details = {{key, written}, {"valid", jarr(valid)}, {"near", jarr(near)}};
  if (!use.empty()) details["use"] = use;
  return coded(what, where, lead + hint + list_sentence, code, details);
}
inline std::string list_of(const std::string& plural, const std::string& owner, const Strs& v) {
  if (v.empty()) return "'" + owner + "' has no " + plural + ".";
  return "The " + plural + " of '" + owner + "' are: " + join(v) + ".";
}
inline Error unknown_option_value(const std::string& type, const std::string& option, const std::string& value_text, const std::string& where,
                                  const Strs& valid) {
  return coded("'" + value_text + "' is not a value of the option '" + option + "' for '" + type + "'", where,
               "The values of '" + option + "' are: " + join(valid) + ".", codes::physics_unknown_option,
               {{"option", option}, {"value", value_text}, {"valid", jarr(valid)}});
}

// ---------------------------------------------------------------------------------------------------- per depth
inline Error per_depth(const std::string& key, const std::string& use, const std::string& unit_total, const std::string& symmetry,
                       const std::string& where) {
  const std::string noun = unit_total == "A" ? "current" : unit_total == "W" ? "power" : "quantity";
  const bool planar = symmetry == "planar";
  std::string what, hint;
  if (planar) {
    what = "'" + key + "' is a total " + noun + "; this model is planar, so results and inputs are per metre of depth. Write '" +
           use + "' [" + unit_total + "/m]";
    hint = "In a planar model every absolute input is written per metre of depth, with the suffix _m.";
  } else {
    what = "'" + key + "' is a " + noun + " per metre of depth; this model is " + symmetry + ", so inputs are 3D totals. Write '" +
           use + "' [" + unit_total + "]";
    hint = "Only a planar model takes the per-metre spelling; here the input is the 3D total.";
  }
  return coded(what, where, hint, codes::per_depth_key, {{"use", use}, {"symmetry", symmetry}});
}

// ---------------------------------------------------------------------------------------------------- expressions
inline std::string coordinate_phrase(const std::string& symmetry) { return "a coordinate of this " + symmetry + " model"; }
inline std::string time_phrase() { return "the time of a transient step"; }

// `swirl_hint`: an axisymmetric vector written with its two meridional components.  The hint is the note's for a
// velocity, and names the components after the key for another vector (force_N_m3: [force_r, force_z, 0]).
inline Error vector_length(const std::string& key, size_t found, size_t expected, const std::string& symmetry,
                           const std::string& component_names, bool swirl_hint, const std::string& where) {
  std::string hint = "write one expression per component, in the order " + component_names + ".";
  if (swirl_hint) {
    const std::string stem = key.substr(0, key.find('_'));
    hint = stem == "velocity" ? std::string("add the swirl component: [u_r, u_z, 0]")
                              : "add the swirl component: [" + stem + "_r, " + stem + "_z, 0]";
  }
  return coded("'" + key + "' has " + std::to_string(found) + " components, and this " + symmetry + " model needs " +
                   std::to_string(expected) + " " + component_names,
               where, hint, codes::expr_vector_length, {{"key", key}, {"found", found}, {"expected", expected}});
}
// An expression that names something that is neither a variable of its context nor a parameter.  `note` is a sentence
// that the context adds (why the time is not a variable here).
inline Error unknown_variable(const Error& from, const std::string& name, const Strs& variables, const std::string& note = "") {
  return coded(from.message(), from.where(),
               "The variables here are " + (variables.empty() ? std::string("none") : join(variables)) +
                   "; the parameters of the document and the constants pi, e and mu0 may be used too." + (note.empty() ? "" : " " + note),
               codes::expr_unknown_variable, {{"name", name}, {"variables", jarr(variables)}});
}

// ---------------------------------------------------------------------------------------------------- couplings
inline Error unknown_term(const std::string& term, const std::string& where, const Strs& valid) {
  return coded("unknown coupling term '" + term + "'", where, "The coupling terms are: " + join(valid) + ".",
               codes::coupling_unknown_term, {{"term", term}, {"valid", jarr(valid)}});
}
// The sentence that lists the physics of the document, or says that it has none.
inline std::string physics_sentence(const Strs& valid) {
  return valid.empty() ? std::string("This document defines no physics.") : "The physics of this document are: " + join(valid) + ".";
}
inline Error coupling_unknown_physics(const std::string& id, const std::string& where, const Strs& valid) {
  return coded("unknown physics '" + id + "'", where, physics_sentence(valid),
               codes::coupling_unknown_physics, {{"physics", id}, {"valid", jarr(valid)}});
}
inline Error type_mismatch(const std::string& term, const std::string& from, const std::string& to, const std::string& from_type,
                           const std::string& to_type, const Strs& producers, const Strs& consumers, const std::string& where) {
  return coded("the term '" + term + "' does not connect '" + from + "' (" + from_type + ") to '" + to + "' (" + to_type + ")", where,
               "'" + term + "' goes from " + join(producers, " or ") + " to " + join(consumers, " or ") + ".",
               codes::coupling_type_mismatch,
               {{"term", term}, {"from", from}, {"to", to}, {"from_type", from_type}, {"to_type", to_type},
                {"producers", jarr(producers)}, {"consumers", jarr(consumers)}});
}
inline Error wrong_selection_key(const std::string& term, const std::string& support, const std::string& key, const std::string& where) {
  const bool domain = support == "domain";
  return coded("the term '" + term + "' acts " + (domain ? "in domains" : "on boundaries") + ", so it takes '" + key +
                   "' and not '" + (domain ? "on" : "in") + "'",
               where,
               domain ? std::string("Write \"in\": DOMAIN to restrict where the term acts, or remove 'on'.")
                      : std::string("Write \"on\": BOUNDARY to say where the term acts, or remove 'in'."),
               codes::coupling_wrong_selection_key, {{"term", term}, {"support", support}, {"key", key}});
}

// ---------------------------------------------------------------------------------------------------- study
inline Error study_type_reserved(const std::string& type, const std::string& where) {
  return coded("study type '" + type + "' is reserved", where,
               "The electromagnetic eigenmode and frequency analyses stay in the 'cavity' and 'fullwave' sections; a model "
               "document has the study types \"stationary\" and \"transient\".",
               codes::study_type_reserved);
}
inline Error study_type_unknown(const std::string& type, const std::string& where) {
  return Error("unknown study type '" + type + "'", where, "Use \"stationary\" or \"transient\".");
}
inline Error study_unknown_physics(const std::string& id, const std::string& where, const Strs& valid) {
  return coded("unknown physics '" + id + "'", where, physics_sentence(valid),
               codes::study_unknown_physics, {{"physics", id}, {"valid", jarr(valid)}});
}
inline Error initial_from_unknown_step(const std::string& id, const std::string& where, const Strs& earlier) {
  return coded("'initial_from' names the step '" + id + "', which is not an earlier step", where,
               earlier.empty() ? std::string("This is the first step, so it has no earlier step to start from.")
                               : "The earlier steps of this study are: " + join(earlier) + ".",
               codes::study_initial_from_unknown_step, {{"initial_from", id}, {"earlier", jarr(earlier)}});
}
inline Error solver_value(const std::string& physics, const std::string& setting, const std::string& type, const json& found,
                          const json& default_value, const std::string& where) {
  const std::string what = type == "boolean" ? "true or false" : type == "integer" ? "a whole number" : "a number";
  return coded("the solver setting '" + setting + "' of '" + physics + "' must be " + what, where,
               "The default is " + default_value.dump() + ".", codes::study_solver_value,
               {{"physics", physics}, {"setting", setting}, {"type", type}, {"found", found}});
}
inline Error unknown_strategy(const std::string& s, const std::string& where) {
  return Error("unknown coupling strategy '" + s + "'", where, "Use \"one_way\", \"gauss_seidel\", \"aitken\" or \"monolithic\".");
}

// ---------------------------------------------------------------------------------------------------- outputs
inline Error output_unknown_kind(const std::string& kind, const std::string& where, const Strs& valid) {
  return coded("unknown output kind '" + kind + "'", where, "The kinds are: " + join(valid) + ".", codes::output_unknown_kind,
               {{"kind", kind}, {"valid", jarr(valid)}});
}
inline Error output_invalid_name(const std::string& name, const std::string& where) {
  return coded("'" + name + "' is not a valid output name", where,
               "An output name is an identifier, so that an expression output can use it: letters, digits and underscores, "
               "not starting with a digit ([A-Za-z_][A-Za-z0-9_]*).",
               codes::output_grammar, {{"name", name}});
}
inline Error output_grammar(const std::string& name, const std::string& kind, const std::string& needs, const std::string& where) {
  return coded("output '" + name + "' of kind '" + kind + "' needs " + needs, where, "", codes::output_grammar);
}
inline Error output_unknown_quantity(const std::string& q, const std::string& symmetry, const std::string& where, const Strs& valid) {
  return coded("unknown quantity '" + q + "' on " + article_model(symmetry), where,
               valid.empty() ? "This document has no quantities on " + article_model(symmetry) + "."
                             : "The quantities of this document on " + article_model(symmetry) + " are: " + join(valid) + ".",
               codes::output_unknown_quantity, {{"quantity", q}, {"geometry", symmetry}, {"valid", jarr(valid)}});
}
inline Error output_unknown_physics(const std::string& id, const std::string& where, const Strs& valid) {
  return coded("unknown physics '" + id + "'", where, physics_sentence(valid),
               codes::output_unknown_physics, {{"physics", id}, {"valid", jarr(valid)}});
}
inline Error output_unknown_boundary(const std::string& b, const std::string& where, const Strs& valid) {
  return coded("unknown boundary '" + b + "'", where,
               valid.empty() ? std::string("This document defines no boundaries.") : "The boundaries of this document are: " + join(valid) + ".",
               codes::output_unknown_boundary, {{"boundary", b}, {"valid", jarr(valid)}});
}
// A key that the kind of the output does not take.
inline Error output_foreign_key(const std::string& name, const std::string& kind, const std::string& key, const Strs& takes,
                                const std::string& where) {
  return coded("output '" + name + "' of kind '" + kind + "' does not take '" + key + "'", where,
               "The keys of an output of kind '" + kind + "' are: " + join(takes) + ".", codes::output_grammar,
               {{"name", name}, {"kind", kind}, {"key", key}, {"valid", jarr(takes)}});
}
inline Error output_over(const std::string& name, const std::string& where) {
  return coded("output '" + name + "' has an unknown 'over'", where,
               "The only series is over the time of a transient step: \"over\": \"time\".", codes::output_grammar, {{"name", name}});
}
inline Error output_unknown_component(const std::string& name, const std::string& component, const std::string& kind,
                                      const std::string& symmetry, const Strs& valid, const std::string& where) {
  return coded("unknown component '" + component + "' of the " + kind + " in " + article_model(symmetry), where,
               "The components of a " + kind + " in " + article_model(symmetry) + " are: " + join(valid) + ".", codes::output_grammar,
               {{"name", name}, {"component", component}, {"geometry", symmetry}, {"valid", jarr(valid)}});
}
// `what` is "constant" or "function".
inline Error output_name_is_grammar(const std::string& name, const std::string& what, const std::string& where) {
  return coded("'" + name + "' cannot be the name of an output: it is " + (what == "function" ? "a function" : "a constant") +
                   " of the expression grammar",
               where,
               "An expression output uses the other outputs by name, so an output must not be named like " +
                   (what == "function" ? std::string("a function") : std::string("a constant")) + " of the grammar. Rename it.",
               codes::output_grammar, {{"name", name}});
}
inline Error output_cycle(const Strs& cycle, const std::string& where) {
  return coded("the expression outputs form a cycle: " + join(cycle, " -> "), where,
               "An expression output may use other outputs, but not itself, directly or through others.", codes::output_cycle,
               {{"cycle", jarr(cycle)}});
}

// ---------------------------------------------------------------------------------------------------- reactions
inline Error reaction_unknown_kind(const std::string& kind, const std::string& where, const Strs& valid) {
  return coded("unknown electrode reaction kind '" + kind + "'", where, "The kinds are: " + join(valid) + ".",
               codes::reaction_unknown_kind, {{"kind", kind}, {"valid", jarr(valid)}});
}
inline Error reaction_unknown_name(const std::string& name, const std::string& where, const Strs& valid) {
  return coded("unknown electrode reaction '" + name + "'", where,
               valid.empty() ? std::string("This document has no electrode reactions: declare them under reactions.electrode.")
                             : "The electrode reactions of this document are: " + join(valid) + ".",
               codes::reaction_unknown_name, {{"name", name}, {"valid", jarr(valid)}});
}

// ---------------------------------------------------------------------------------------------------- materials
inline Error block_incomplete(const std::string& material, const std::string& block, const std::string& key, const std::string& unit,
                              const std::string& where) {
  return coded("the '" + block + "' block of material '" + material + "' has no '" + key + "'", where,
               "A '" + block + "' block needs '" + key + "' (" + unit + "); add it.", codes::material_block_incomplete,
               {{"material", material}, {"block", block}, {"key", key}});
}
inline Error expansion_ambiguous(const std::string& where) {
  return coded("`alpha` is the total contraction strain used by `mechanics.cooldown`; `alpha_1_K` is the expansion coefficient "
               "per kelvin used by the `thermal_expansion` coupling. Keep the one this document uses.",
               where, "", codes::material_expansion_ambiguous);
}
inline Error conductivity_ambiguous(const std::string& material, const json& material_level, const json& electric, const std::string& where) {
  return coded("material '" + material + "' gives sigma_S_m both at the material level and in its 'electric' block, with different "
               "values",
               where,
               "Keep one of them, or make them equal: the material-level key is the conductivity of a full-wave medium and "
               "'electric.sigma_S_m' that of 'current.conduction'.",
               codes::material_conductivity_ambiguous, {{"material", material}, {"material_level", material_level}, {"electric", electric}});
}
inline Error range_missing(const std::string& property, const std::string& field, const std::string& where) {
  return coded("the law of '" + property + "' needs a validity range for '" + field + "'", where,
               "Add \"range\": {\"" + field + "\": [" + field + "_min, " + field + "_max]} with the interval over which the law is valid.",
               codes::law_range_missing, {{"property", property}, {"field", field}});
}
inline Error law_unknown_kind(const std::string& kind, const std::string& where) {
  return coded("unknown law '" + kind + "'", where, "Use \"expression\" or \"table\".", codes::law_unknown_kind);
}
inline Error law_unknown_field(const std::string& field, const std::string& where) {
  return coded("'" + field + "' is not a field that a law can read", where, "The fields are T, p, B_mag and c_<species>.",
               codes::law_unknown_field, {{"field", field}, {"valid", Strs{"T", "p", "B_mag", "c_<species>"}}});
}
inline Error law_unknown_option(const std::string& option, const std::string& value, const std::string& where, const std::string& hint) {
  return coded("unknown " + option + " '" + value + "'", where, hint, codes::law_unknown_option);
}
inline Error table_invalid(const std::string& what, const std::string& where, const std::string& hint = "") {
  return coded(what, where, hint, codes::law_table_invalid);
}

// ==================================================================================================== validation
// The texts of the route-level checks (model_validate.cpp).

// A number as the shortest decimal text that reads back as the same double; an integral value has no decimal point.
inline std::string num_text(double x) {
  char buf[64];
  if (x == 0.0 || !(x - x == 0.0)) {  // zero, or not finite
    std::snprintf(buf, sizeof buf, "%g", x);
    return buf;
  }
  int digits = 1;  // the fewest significant digits that read back as x
  for (; digits < 17; ++digits) {
    std::snprintf(buf, sizeof buf, "%.*e", digits - 1, x);
    if (std::strtod(buf, nullptr) == x) break;
  }
  std::snprintf(buf, sizeof buf, "%.*e", digits - 1, x);
  const std::string sci = buf;
  const int exponent = std::atoi(sci.c_str() + sci.find('e') + 1);
  if (exponent < -5 || exponent > 15) {
    std::snprintf(buf, sizeof buf, "%.*g", digits, x);
    return buf;
  }
  std::snprintf(buf, sizeof buf, "%.*f", std::max(0, digits - 1 - exponent), x);
  return buf;
}
inline std::string quoted(const Strs& v) {  // 'a', 'b', 'c'
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + ("'" + v[i] + "'");
  return s;
}
inline std::string quoted_and(const Strs& v) {  // 'a', 'b' and 'c'
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i == 0 ? "" : (i + 1 == v.size() ? " and " : ", ")) + ("'" + v[i] + "'");
  return s;
}
// A value of a document as text: a string as written, a number in its shortest form.
inline std::string value_text(const json& v) {
  if (v.is_string()) return v.get<std::string>();
  if (v.is_number()) return num_text(v.get<double>());
  return v.dump();
}

// ---------------------------------------------------------------------------------------------------- routes
// No row: the inventory of what is missing (design note 4.4).
inline Error route_inventory(const Strs& types, const Strs& terms, const std::string& geometry, const std::string& study,
                             const Strs& missing) {
  std::string what = "no route for {" + join(types) + "}";
  if (!terms.empty()) what += " with {" + join(terms) + "}";
  if (!geometry.empty()) what += " " + on_geometry(geometry);
  if (study != "stationary") what += " in a " + study + " step";
  // one missing piece is "it", unless the piece is itself a list of types or of terms
  const bool several = missing.size() != 1 || missing[0].rfind("physics types ", 0) == 0 || missing[0].rfind("coupling terms ", 0) == 0;
  return coded(what, "/physics",
               "not implemented in this version: " + join(missing, "; ") + ". The document does not need to change when " +
                   (several ? "they are" : "it is") + ". `axmp physics --routes` lists every combination that can be solved.",
               codes::route_not_implemented,
               {{"physics", jarr(types)}, {"couplings", jarr(terms)}, {"geometry", geometry.empty() ? json(nullptr) : json(geometry)},
                {"study", study}, {"missing", jarr(missing)}});
}
// A row that is not implemented yet.  `members`: (id, type) of each instance of the group.
inline Error route_not_implemented(const std::vector<std::pair<std::string, std::string>>& members, const std::string& route,
                                   const std::string& engine, const json& needs, const json& missing_in_this_build) {
  std::string who;
  Strs ids;
  for (size_t i = 0; i < members.size(); ++i) {
    who += (i == 0 ? "" : (i + 1 == members.size() ? " and " : ", ")) + ("'" + members[i].first + "' (" + members[i].second + ")");
    ids.push_back(members[i].first);
  }
  return coded("physics " + who + (members.size() == 1 ? " takes" : " take") + " the route '" + route + "', which is not implemented yet",
               "/physics",
               "The document is valid; this version validates it and cannot solve it yet. `axmp physics --routes` lists every "
               "route and whether it is implemented.",
               codes::route_not_implemented,
               {{"route", route}, {"physics", jarr(ids)}, {"engine", engine}, {"requires", needs}, {"implemented", false},
                {"missing_in_this_build", missing_in_this_build}});
}
inline Error study_type_not_offered(const std::string& id, const std::string& type, const std::string& study, const Strs& offered,
                                    const std::string& where, bool written) {
  std::string hint;
  if (study == "stationary") {
    const std::string step = "{\"id\": \"run\", \"type\": \"transient\", \"time\": {\"t_end_s\": ...}}";
    hint = written ? "use a transient step: " + step : "add \"study\": [" + step + "]";
  } else {
    hint = "use a stationary step: {\"id\": \"run\", \"type\": \"stationary\"}";
  }
  return coded("physics '" + id + "' (" + type + ") has no " + study + " solve", where, hint, codes::study_type_not_offered,
               {{"physics", id}, {"type", type}, {"offered", jarr(offered)}});
}

// ---------------------------------------------------------------------------------------------------- condition kinds
// `why`: "engine_refuses" or "not_exposed" (then with the reason and the roadmap item of the row).
inline Error flow_condition_not_offered(const std::string& id, const std::string& kind, const std::string& where, const std::string& route,
                                        const std::string& engine, const std::string& geometry, const std::string& equation,
                                        const Strs& offered, const std::string& why, const std::string& reason,
                                        const std::string& roadmap) {
  json details = {{"kind", kind}, {"route", route}, {"geometry", geometry}, {"equation", equation}, {"offered", jarr(offered)}, {"why", why}};
  std::string hint;
  if (why == "not_exposed") {
    hint = "The engine could take it, but the document path does not expose it yet: " + reason + "." +
           (roadmap.empty() ? "" : " Roadmap item: '" + roadmap + "'.") + " The kinds offered on the route '" + route + "' are: " +
           join(offered) + ".";
    details["reason"] = reason;
    details["roadmap"] = roadmap;
  } else {
    hint = "The engine of the route '" + route + "' (" + engine + ") refuses it. The kinds offered there are: " + join(offered) + ".";
  }
  return coded("physics '" + id + "': the condition kind '" + kind + "' is not offered on this route", where, hint,
               codes::fluid_condition_not_offered, details);
}
// `form`: the advective form that excludes the kind, or "" when the route offers the kind with no form.  `why`:
// "engine_refuses" or "layer_rule" (then with the rule's name).
inline Error heat_condition_not_offered(const std::string& id, const std::string& kind, const std::string& where, const std::string& route,
                                        const std::string& engine, const std::string& geometry, const std::string& form,
                                        const Strs& offered, const std::string& why, const std::string& rule) {
  json details = {{"kind", kind}, {"route", route}, {"geometry", geometry}, {"offered", jarr(offered)}, {"why", why}};
  if (!rule.empty()) details["rule"] = rule;
  if (form.empty())
    return coded("physics '" + id + "': the condition kind '" + kind + "' is not offered on this route", where,
                 "The engine of the route '" + route + "' (" + engine + ") refuses it. The kinds offered there are: " + join(offered) + ".",
                 codes::heat_condition_not_offered, details);
  details["advective_form"] = form;
  const std::string others = "or the convective or conservative_divergence form.";
  std::string hint;
  if (kind == "outflow") {
    hint = "With the " + form + " form a zero flux is a zero total flux, which would block the advected heat. Use total_flux "
           "(q_W_m2 is the total flux into the domain), " + others;
  } else if (kind == "heat_flux") {
    hint = "With the " + form + " form the natural datum is the total flux. Use total_flux (q_W_m2 is the total flux into the "
           "domain), " + others;
  } else if (kind == "total_flux") {
    hint = "The " + form + " form's natural datum is the conductive flux. Use heat_flux (q_W_m2 is the conductive flux into the "
           "domain), or \"advective_form\": \"conservative_flux\".";
  } else {
    hint = "The kinds offered with this form are: " + join(offered) + ".";
  }
  return coded("physics '" + id + "': the condition kind '" + kind + "' is not offered with \"advective_form\": \"" + form + "\"", where,
               hint, codes::heat_condition_not_offered, details);
}
inline Error potential_level_free(const std::string& id, const Strs& electrodes) {
  return coded("physics '" + id + "': every electrode is galvanostatic, so the potential level is free",
               "/physics/" + id + "/boundary_conditions",
               "Give at least one electrode a potential (phi_s_V); the engine refuses a problem whose electrodes are all "
               "galvanostatic.",
               codes::electrochem_potential_level_free, {{"electrodes", jarr(electrodes)}});
}

// ---------------------------------------------------------------------------------------------------- options
// An option value that the row does not offer.  `value` is the value as the document writes it.
inline Error option_not_offered(const std::string& family, const std::string& id, const std::string& option, const json& value,
                                const json& offered, const Strs& offered_text, const std::string& route, const std::string& engine) {
  const std::string where = "/physics/" + id + "/options/" + option;
  const json details = {{"option", option}, {"value", value}, {"offered", offered}, {"route", route}, {"engine", engine}};
  const std::string v = value_text(value);
  if (family == "heat" && option == "order")
    return coded("physics '" + id + "': the temperature order " + v + " is not offered on this route", where,
                 "the monolithic Boussinesq engine refuses a P1 temperature (\"the temperature is P2 (the coupled solver refuses "
                 "P1 T)\"); it solves Taylor-Hood P2/P1 velocity and pressure with a P2 temperature. Use \"order\": 2, or drop the "
                 "buoyancy coupling (the conduction and advection routes offer P1).",
                 codes::heat_option_not_offered, details);
  if (family == "heat" && option == "advective_form")
    return coded("physics '" + id + "': the advective form '" + v + "' is not offered on this route", where,
                 "The monolithic Boussinesq engine has the convective form only. Use \"advective_form\": \"convective\" (the "
                 "default), or drop the buoyancy coupling (the conduction and advection routes offer the three forms).",
                 codes::heat_option_not_offered, details);
  if (family == "fluid" && option == "convection")
    return coded("physics '" + id + "': the convective form '" + v + "' is not offered on this route", where,
                 "The transient Navier-Stokes engine (" + engine + ") takes the standard convective form only. Use "
                 "\"convection\": \"standard\" (the default).",
                 codes::fluid_option_not_offered, details);
  return coded("physics '" + id + "': the value " + v + " of the option '" + option + "' is not offered on this route", where,
               "The route '" + route + "' (" + engine + ") offers: " + join(offered_text) + ".",
               family == "fluid" ? codes::fluid_option_not_offered
                                 : family == "heat" ? codes::heat_option_not_offered : codes::physics_option_not_offered,
               details);
}
inline Error supg_needs_advection(const std::string& id, const std::string& route) {
  return coded("physics '" + id + "': \"stabilization\": \"supg\" needs an advecting velocity, and nothing advects this instance",
               "/physics/" + id + "/options/stabilization",
               "SUPG stabilizes advection, and the transport operator refuses it without a velocity. Use \"stabilization\": "
               "\"none\", or advect the instance (an advection term from a flow, or a prescribed_advection source).",
               codes::heat_supg_needs_advection, {{"option", "stabilization"}, {"value", "supg"}, {"route", route}});
}

// ---------------------------------------------------------------------------------------------------- uniform data
// Two regions of an instance with their materials and the values of a property.
struct TwoValues {
  std::string region_a, material_a, region_b, material_b;
  double a = 0.0, b = 0.0;
};
// `property`: density | thermal.cp_J_kgK | fluid.mu_Pa_s | fluid.beta_1_K.  `reason`: which engine sentence and remedy
// follow the two values: navier_stokes | transient | boussinesq | transport | stokes_force | layer_force.  `output` is
// the force output that imposes the requirement (the two force reasons).
inline Error property_not_uniform(const std::string& id, const std::string& property, const std::string& route, const TwoValues& v,
                                  const json& regions, const std::string& reason, const std::string& output) {
  std::string label = property, unit;
  if (property == "density") label = "density", unit = "kg/m^3";
  else if (property.rfind("thermal.cp", 0) == 0) label = "heat capacity", unit = "J/(kg K)";
  else if (property.rfind("fluid.mu", 0) == 0) label = "viscosity", unit = "Pa s";
  else if (property.rfind("fluid.beta", 0) == 0) label = "thermal expansion coefficient", unit = "1/K";
  const std::string values = "'" + v.region_a + "' (" + v.material_a + ") has " + label + " " + num_text(v.a) + " " + unit + " and '" +
                             v.region_b + "' (" + v.material_b + ") has " + num_text(v.b) + " " + unit + ".";
  std::string tail;
  if (reason == "navier_stokes")
    tail = " The steady Navier-Stokes engine takes one density per solve. Use one physics instance per fluid, or equation "
           "\"stokes\" (which does not use the density).";
  else if (reason == "transient")
    tail = " The transient Navier-Stokes engine takes one density per solve. Use one physics instance per fluid.";
  else if (reason == "boussinesq")
    tail = " The Boussinesq engine takes one " + label + " per solve. Use materials that agree in " + property + " over the instance.";
  else if (reason == "transport")
    tail = " The transport operator advects with one capacity rho cp per solve. Use materials that agree in density and "
           "thermal.cp_J_kgK over the instance.";
  else if (reason == "stokes_force")
    tail = " With the Laplacian viscous form the engine's physical reaction is defined for one viscosity, and the force output '" +
           output + "' needs it. Use \"viscous_form\": \"stress\", or remove the force output.";
  else
    tail = " With the Laplacian viscous form and a viscosity that varies, the engine's reaction is not a force, and the force "
           "output '" + output + "' needs one viscosity. Use \"viscous_form\": \"stress\", or remove the force output.";
  json details = {{"property", property}, {"route", route}, {"regions", regions}};
  if (!output.empty()) details["output"] = output;
  return coded("physics '" + id + "': the " + label + " must be the same in every region of its domain on this route",
               "/physics/" + id + "/domains", values + tail, codes::physics_property_not_uniform, details);
}
inline std::string one_function_sentence(bool boussinesq, const std::string& what) {
  return boussinesq ? " The Boussinesq engine takes the " + what + " as one function of position and is not told the region."
                    : " The transport operator takes the " + what + " as one function of position in this version and is not told the region.";
}
inline Error conductivity_not_regionwise(const std::string& id, const std::string& route, bool boussinesq, const TwoValues& v,
                                         const json& regions) {
  return coded("physics '" + id + "': the conductivity must be the same in every region of its domain on this route",
               "/physics/" + id + "/domains",
               "'" + v.region_a + "' (" + v.material_a + ") has k = " + num_text(v.a) + " W/(m K) and '" + v.region_b + "' (" +
                   v.material_b + ") has " + num_text(v.b) + "." + one_function_sentence(boussinesq, "conductivity") +
                   " Give the regions of the instance one conductivity.",
               codes::heat_data_not_regionwise, {{"data", "thermal.k_W_mK"}, {"route", route}, {"regions", regions}});
}
// `text_a`, `text_b`: what each of the two regions has ("q_W_m3 = 1e3*x", "2e3*x", "no heat source").
inline Error heat_source_not_regionwise(const std::string& id, const std::string& route, bool boussinesq, const std::string& region_a,
                                        const std::string& text_a, const std::string& region_b, const std::string& text_b,
                                        const json& regions) {
  return coded("physics '" + id + "': the heat source must be the same in every region of its domain on this route",
               "/physics/" + id + "/sources",
               "'" + region_a + "' has " + text_a + " and '" + region_b + "' has " + text_b + "." +
                   one_function_sentence(boussinesq, "heat source") + " Write one source over the whole instance.",
               codes::heat_data_not_regionwise, {{"data", "heat_source"}, {"route", route}, {"regions", regions}});
}
inline Error prescribed_advection_partial(const std::string& id, size_t source, const Strs& regions, const std::string& in,
                                          bool capacities_agree) {
  const std::string names = quoted_and(regions);
  return coded("a prescribed velocity covers the whole heat instance on this route (the transport operator advects with one velocity "
               "field and one capacity rho cp per solve)",
               "/physics/" + id + "/sources/" + std::to_string(source) + "/in",
               "Write one field expression over " + names + " that goes to zero continuously where nothing moves (the expression "
               "grammar has no step and no conditional: `max(0, ...)` and the smooth functions are what there is), which this "
               "route then needs " + names + " to agree on in `density` and `thermal.cp_J_kgK`.",
               codes::heat_prescribed_advection_partial, {{"regions", jarr(regions)}, {"in", in}, {"capacities_agree", capacities_agree}});
}
inline Error prescribed_advection_twice(const std::string& id, const std::vector<size_t>& sources) {
  return coded("physics '" + id + "' has " + (sources.size() == 2 ? std::string("two") : std::to_string(sources.size())) +
                   " prescribed_advection sources; this route takes one",
               "/physics/" + id + "/sources/" + std::to_string(sources.at(1)),
               "The transport operator advects with one velocity field per solve. Write one field expression over the whole "
               "instance.",
               codes::heat_prescribed_advection_partial, {{"sources", sources}});
}
inline Error two_advections(const std::string& id, size_t source, const std::string& from, int coupling) {
  return coded("physics '" + id + "' is advected twice: by the prescribed_advection source and by the advection term from '" + from + "'",
               "/physics/" + id + "/sources/" + std::to_string(source),
               "The transport operator advects with one velocity per solve. Keep the advection term to use the solved flow, or the "
               "prescribed_advection source to use a given velocity.",
               codes::heat_two_advections, {{"physics", id}, {"source", source}, {"term_from", from}, {"coupling", coupling}});
}

// ---------------------------------------------------------------------------------------------------- domains
inline Error domains_must_coincide(const std::string& a, const std::string& b, const std::string& route, const Strs& regions_a,
                                   const Strs& regions_b) {
  return coded("physics '" + a + "' and '" + b + "' must have the same domain on this route", "/physics/" + b + "/domains",
               "The route '" + route + "' solves both on one mesh with no interpolation: '" + a + "' is on " + quoted(regions_a) +
                   " and '" + b + "' on " + quoted(regions_b) + ". Give them the same domains.",
               codes::route_domains_must_coincide, {{"route", route}, {"physics", {{a, jarr(regions_a)}, {b, jarr(regions_b)}}}});
}
// `missing`: the registry's names of what the material lacks ("fluid", "density", "thermal.cp_J_kgK").
inline std::string missing_blocks_text(const Strs& missing) {
  std::string s;
  for (size_t i = 0; i < missing.size(); ++i) {
    const bool block = missing[i].find('.') == std::string::npos && missing[i] != "density";
    s += (i == 0 ? "no " : (i + 1 == missing.size() ? " and no " : ", no ")) + ("`" + missing[i] + "`") + (block ? " block" : "");
  }
  return s;
}
inline Error material_block_missing(const std::string& region, const std::string& material, const std::string& id, const std::string& type,
                                    const Strs& missing) {
  return coded("region '" + region + "' is in physics '" + id + "' (" + type + "), but its material '" + material + "' has " +
                   missing_blocks_text(missing),
               "/physics/" + id + "/domains",
               "Add the missing data to material '" + material + "', or remove '" + region + "' from the domains of '" + id + "'.",
               codes::physics_material_block_missing,
               {{"region", region}, {"material", material}, {"physics", id}, {"missing", jarr(missing)}});
}
// An instance without "domains" that no enabled region can take part in.
inline Error no_region_for_physics(const std::string& id, const std::string& type, const Strs& needs) {
  return coded("physics '" + id + "' (" + type + ") has no region: no enabled region has a material with what its route needs",
               "/physics/" + id,
               "Without \"domains\" an instance takes every enabled region whose material carries " + join(needs) +
                   ". Add the data to a material, or write \"domains\".",
               codes::physics_material_block_missing, {{"physics", id}, {"missing", jarr(needs)}});
}
inline Error domains_overlap(const std::string& a, const std::string& b, const std::string& family, const Strs& regions) {
  return coded("physics '" + a + "' and '" + b + "' (family " + family + ") share the region" + (regions.size() == 1 ? " " : "s ") +
                   quoted_and(regions),
               "/physics/" + b + "/domains", "Two instances of one family may not overlap: give them disjoint domains, or use one instance.",
               codes::physics_domains_overlap, {{"physics", jarr(Strs{a, b})}, {"family", family}, {"regions", jarr(regions)}});
}
inline Error gravity_direction(int coupling, const json& written, const std::string& route, const std::string& axis) {
  return coded("the buoyancy term needs gravity along the " + axis + " axis on this route",
               "/couplings/" + std::to_string(coupling) + "/gravity_m_s2",
               "The Boussinesq engine takes gravity along the vertical axis only (y in a planar model, z in an axisymmetric one), "
               "pointing up or down: write [0, g].",
               codes::fluid_gravity_direction, {{"gravity_m_s2", written}, {"route", route}, {"axis", axis}});
}
inline Error buoyancy_needs_beta(const std::string& region, const std::string& material, const std::string& id) {
  return coded("region '" + region + "' is in physics '" + id + "' with the buoyancy term, but its material '" + material +
                   "' has no `fluid.beta_1_K`",
               "/materials/" + material + "/fluid",
               "The Boussinesq buoyancy term needs the thermal expansion coefficient of the fluid: add \"beta_1_K\" to the fluid "
               "block of '" + material + "'.",
               codes::fluid_buoyancy_needs_beta, {{"region", region}, {"material", material}, {"physics", id}});
}

// ---------------------------------------------------------------------------------------------------- geometry order
// `straight_only`: the engine sentence of an instance whose P1 engine takes straight elements only.
inline Error geometry_order_exceeds(int geometry_order, int field_order, const std::string& id, bool straight_only) {
  return coded("mesh.geometry_order " + std::to_string(geometry_order) + " exceeds the field order " + std::to_string(field_order) +
                   " of physics '" + id + "'",
               "/mesh/geometry_order",
               std::string("Element geometry of higher order than the fields breaks the exactness of rigid motions") +
                   (straight_only ? ", and the P1 current-distribution engine takes straight elements only" : "") + ". Use order " +
                   std::to_string(geometry_order) + " for '" + id + "', or geometry order " + std::to_string(field_order) + ".",
               codes::geometry_order_exceeds_field_order,
               {{"geometry_order", geometry_order}, {"field_order", field_order}, {"physics", id}});
}
inline Error geometry_order_not_offered(int geometry_order, const std::string& id, const std::string& option, const std::string& value) {
  return coded("mesh.geometry_order " + std::to_string(geometry_order) + " is not offered with \"" + option + "\": \"" + value +
                   "\" of physics '" + id + "': the transport operator does not offer SUPG on isoparametric (curved) elements (the "
                   "mapped Hessian is not implemented)",
               "/mesh/geometry_order", "Use geometry order 1, or \"" + option + "\": \"none\".", codes::geometry_order_not_offered,
               {{"physics", id}, {"option", option}, {"value", value},
                {"engine_text", "SUPG on isoparametric (curved) elements is not offered"}});
}
// `option` empty: the instance has field order 1; otherwise the option value that needs straight elements.
inline Error geometry_order_lowered(const std::string& id, const std::string& option, const std::string& value) {
  if (option.empty())
    return coded("the geometry order is lowered to 1 because physics '" + id + "' has field order 1, and this model has curved "
                 "boundaries",
                 "/mesh",
                 "On straight elements a curved boundary is a polygon, and every instance of the model is solved on this geometry. "
                 "Use order 2 for '" + id + "' to keep curved elements, or write \"geometry_order\": 1 in \"mesh\" to state it.",
                 codes::geometry_order_lowered, {{"geometry_order", 1}, {"physics", id}, {"cause", "order=1"}});
  return coded("the geometry order is lowered to 1 because physics '" + id + "' has \"" + option + "\": \"" + value +
                   "\", and this model has curved boundaries",
               "/mesh",
               "SUPG needs straight elements. On straight elements a curved boundary is a polygon, and every instance of the model, "
               "a flow included, is solved on this geometry. Use \"" + option + "\": \"none\" to keep curved elements, or write "
               "\"geometry_order\": 1 in \"mesh\" to state it.",
               codes::geometry_order_lowered, {{"geometry_order", 1}, {"physics", id}, {"cause", option + "=" + value}});
}
// The checks of an imported mesh, which need the file.
inline Error deferred_field_order(const std::string& file, const std::string& id, int field_order) {
  return coded("the geometry order of the imported mesh '" + file + "' is checked against the field order " + std::to_string(field_order) +
                   " of physics '" + id + "' when the file is read",
               "/mesh/file", "An order-2 mesh is refused then: re-export it at order 1, or use order 2 for '" + id + "'.",
               codes::geometry_order_exceeds_field_order, {{"physics", id}, {"field_order", field_order}, {"file", file}});
}
inline Error deferred_straight_elements(const std::string& file, const std::string& id, const std::string& option, const std::string& value) {
  return coded("the geometry order of the imported mesh '" + file + "' is checked against \"" + option + "\": \"" + value +
                   "\" of physics '" + id + "' when the file is read",
               "/mesh/file", "An order-2 mesh is refused then: re-export it at order 1, or use \"" + option + "\": \"none\".",
               codes::geometry_order_not_offered, {{"physics", id}, {"option", option}, {"value", value}, {"file", file}});
}
inline Error deferred_element_types(const std::string& file, const std::string& id, const std::string& route, const Strs& accepts) {
  return coded("the element types of the imported mesh '" + file + "' are checked against physics '" + id + "' when the file is read",
               "/mesh/file", "The route '" + route + "' takes triangles only: a quadrilateral or mixed mesh is refused then.",
               codes::element_type_not_offered, {{"physics", id}, {"route", route}, {"accepts", jarr(accepts)}, {"file", file}});
}

// ---------------------------------------------------------------------------------------------------- study
inline Error ramp_end_mismatch(const std::string& where, const std::string& ramp, double last, double document) {
  return coded("the ramp ends at " + num_text(last) + " and the document says " + num_text(document), where,
               "The last value of a ramp is the value the document gives the field (" + ramp + "), so that the document states the "
               "model that is finally solved. End the ramp at " + num_text(document) + ", or give the field the value " + num_text(last) + ".",
               codes::study_ramp_end_mismatch, {{"ramp", ramp}, {"last", last}, {"document", document}});
}
inline Error field_not_rampable(const std::string& where, const std::string& ramp, const Strs& rampable) {
  return coded("'" + ramp + "' is not a rampable field", where,
               "A ramp steps a field that the engine can warm-start: in this version one electrode's phi_s_V or current. " +
                   (rampable.empty() ? std::string("This document has no rampable field.")
                                     : "The rampable fields of this document are: " + join(rampable) + "."),
               codes::study_field_not_rampable, {{"ramp", ramp}, {"rampable", jarr(rampable)}});
}
// A ramp of a field that is rampable, of a physics that another step solves and this one does not.
inline Error ramp_of_unsolved_physics(const std::string& where, const std::string& ramp, const std::string& id, const std::string& step,
                                      const Strs& rampable) {
  return coded("'" + ramp + "' is not a rampable field of step '" + step + "', which does not solve physics '" + id + "'", where,
               "A ramp steps a field of a physics that its own step solves, each stage starting from the previous solution. Move "
               "the continuation to a step that solves '" + id + "', or add '" + id + "' to the physics of step '" + step + "'.",
               codes::study_field_not_rampable, {{"ramp", ramp}, {"rampable", jarr(rampable)}, {"step", step}, {"physics", id}});
}
// "one_way" written for a step whose graph has a two-way group.  `offered`: the strategies of the group's route, or
// empty when the group has no route.
inline Error one_way_on_two_way_group(const std::string& where, const std::string& step, const Strs& group, const Strs& offered) {
  Strs two_way;
  for (const std::string& s : offered)
    if (s != "one_way") two_way.push_back(s);
  return coded("the coupling strategy 'one_way' is not offered for the two-way group " + quoted_and(group) + " of step '" + step + "'", where,
               "A two-way group is never run as one unlabelled sweep. Write " +
                   (two_way.empty() ? std::string("gauss_seidel, aitken or monolithic (no route lists the strategies of this group yet)")
                                    : "one of the strategies of its route: " + join(two_way)) +
                   ".",
               codes::study_strategy_not_offered, {{"strategy", "one_way"}, {"step", step}, {"group", jarr(group)}, {"offered", jarr(two_way)}});
}
// A continuation scale whose coefficient no physics of the step has.  `has_fluid` is true for a viscosity scale in a step
// that solves a fluid.incompressible physics that has no viscosity schedule: only the steady Navier-Stokes engine has one
// (the Stokes engine has none; the Boussinesq engine scales the buoyancy; the time stepper runs no continuation).
inline Error scale_not_applicable(const std::string& where, const std::string& scale, const std::string& step, bool has_fluid = false) {
  const bool buoyancy = scale == "buoyancy";
  const std::string needs = buoyancy ? "coupling term buoyancy"
                                     : has_fluid ? "a stationary fluid.incompressible physics with equation navier_stokes and no buoyancy term"
                                                 : "physics type fluid.incompressible";
  return coded("the continuation scale '" + scale + "' has nothing to scale in step '" + step + "': " +
                   (buoyancy ? "no buoyancy term couples two physics of the step"
                    : has_fluid ? "no fluid.incompressible physics of the step has a viscosity schedule"
                                : "the step solves no fluid.incompressible physics"),
               where,
               std::string("A scale multiplies one coefficient of the flow in stages: \"viscosity\" the viscosity of a "
                           "fluid.incompressible physics, \"buoyancy\" the buoyancy term. ") +
                   (buoyancy ? "Add the buoyancy term to \"couplings\" and both of its physics to the step, or remove the continuation."
                    : has_fluid ? "Only the steady Navier-Stokes engine has a viscosity schedule: write \"equation\": \"navier_stokes\" in a stationary "
                                  "step without the buoyancy term (the Boussinesq engine scales \"buoyancy\"), or remove the continuation."
                                : "Add the flow to the physics of the step, or remove the continuation."),
               codes::study_scale_not_applicable, {{"scale", scale}, {"step", step}, {"needs", needs}});
}
inline Error strategy_not_offered(const std::string& where, const std::string& strategy, const std::string& route, const Strs& offered,
                                  const std::string& default_strategy, const std::string& step) {
  return coded("the coupling strategy '" + strategy + "' is not offered on the route '" + route + "'; this route offers: " + join(offered),
               where, "Remove \"strategy\" to use the route's default (" + default_strategy + "), or write one of the offered strategies.",
               codes::study_strategy_not_offered, {{"strategy", strategy}, {"route", route}, {"offered", jarr(offered)}, {"step", step}});
}
inline Error two_way_needs_strategy(const std::string& where, const std::string& step, const Strs& group) {
  return coded("physics " + quoted_and(group) + " form a two-way group in step '" + step + "', which needs a coupling strategy", where,
               "Write \"coupling\": {\"strategy\": ...} in the step: gauss_seidel, aitken or monolithic, as the route of the group "
               "offers. A two-way group is never run as one unlabelled sweep.",
               codes::study_two_way_needs_strategy, {{"step", step}, {"group", jarr(group)}});
}
// The time t in a datum of an instance that a stationary step solves as well as a transient one.
inline Error time_in_stationary_step(const std::string& where, const std::string& step, const std::string& id) {
  return coded("'t' is not a variable in the stationary step '" + step + "', which solves physics '" + id + "'", where,
               "The time t is a variable only in a transient step. Solve '" + id + "' in transient steps only, or write the datum "
               "without t.",
               codes::expr_unknown_variable, {{"name", "t"}, {"step", step}, {"physics", id}});
}

// ---------------------------------------------------------------------------------------------------- laws
inline std::string property_label(const std::string& property) {
  if (property == "density") return "density";
  if (property.rfind("fluid.mu", 0) == 0) return "viscosity";
  if (property.rfind("fluid.beta", 0) == 0) return "thermal expansion coefficient";
  if (property.rfind("thermal.k", 0) == 0) return "conductivity";
  if (property.rfind("thermal.cp", 0) == 0) return "heat capacity";
  if (property.rfind("electrolyte.", 0) == 0) return "electrolyte conductivity";
  if (property.rfind("electric.", 0) == 0) return "electric conductivity";
  return property;
}
inline Error law_not_offered(const std::string& id, const std::string& property, const std::string& material, const Strs& fields,
                             const std::string& route, const std::string& where) {
  return coded("physics '" + id + "': the " + property_label(property) + " of '" + material + "' is a law of " + join(fields, " and ") +
                   ", and this route takes a constant",
               where,
               "The engine of the route '" + route + "' takes " + property + " as one number per region; a law of a field is not "
               "offered on any route of this version. Write a number or an expression over the parameters.",
               codes::route_law_not_offered, {{"property", property}, {"material", material}, {"route", route}, {"physics", id}});
}
// `providers`: the physics types that provide the field.
inline Error law_field_not_provided(const std::string& property, const std::string& material, const std::string& field,
                                    const std::string& step, const Strs& regions, const Strs& providers, const std::string& where) {
  const std::string on = quoted_and(regions);
  const std::string who = providers.empty() ? "no physics type of this version provides " + field
                                            : join(providers, " or ") + " provides " + field;
  return coded("the law of " + property + " in material '" + material + "' reads " + field + ", which no physics of step '" + step +
                   "' provides on " + on,
               where,
               "Add a physics that provides " + field + " on " + on + " to the step (" + who + "), or give " + field +
                   " a fixed value in the step's \"given\": {\"" + field + "\": ...}.",
               codes::law_field_not_provided,
               {{"property", property}, {"material", material}, {"field", field}, {"step", step}, {"regions", jarr(regions)},
                {"providers", jarr(providers)}});
}

// ---------------------------------------------------------------------------------------------------- outputs
// `output_kind`: force | torque.  `condition`: the kind of the condition on the boundary; `moving`: a wall with a velocity.
inline std::string laplacian_force_message(const std::string& name, const std::string& output_kind, const std::string& boundary) {
  return "output '" + name + "': a " + output_kind + " on '" + boundary + "' is not available with the Laplacian viscous form on this route";
}
inline std::string laplacian_force_reason() {
  return "The Navier-Stokes engine returns, for the Laplacian form, the pseudo-traction mu du/dn - p n, which is the force only on "
         "walls at rest. Use \"viscous_form\": \"stress\" (the physical form), or \"equation\": \"stokes\" if the flow is creeping.";
}
inline Error force_needs_stress_form(const std::string& name, const std::string& output_kind, const std::string& boundary,
                                     const std::string& condition, bool moving, const std::string& route) {
  const std::string what = moving ? "'" + boundary + "' is a moving wall." : "'" + boundary + "' carries the condition '" + condition + "'.";
  return coded(laplacian_force_message(name, output_kind, boundary), "/outputs/" + name, what + " " + laplacian_force_reason(),
               codes::output_force_needs_stress_form,
               {{"boundary", boundary}, {"condition", moving ? std::string("wall with velocity_m_s") : condition}, {"route", route},
                {"accepted_on", "wall boundaries at rest"}});
}

// The same rule for a boundary that is not a key of the instance's conditions: what it carries is known when the
// boundaries are resolved.
inline Error deferred_force_condition(const std::string& name, const std::string& output_kind, const std::string& boundary,
                                      const std::string& id, const std::string& route) {
  return coded("output '" + name + "': whether a " + output_kind + " on '" + boundary + "' is available with the Laplacian viscous form "
               "is checked when the boundaries are resolved",
               "/outputs/" + name,
               "'" + boundary + "' is not a key of the boundary conditions of '" + id + "', so the conditions on it are not known "
               "before that. With the Laplacian form the Navier-Stokes engine returns the pseudo-traction mu du/dn - p n, which is "
               "the force only on walls at rest: the output is refused then if '" + boundary + "' has any other part. With "
               "\"viscous_form\": \"stress\" it is available on every boundary.",
               codes::output_force_needs_stress_form,
               {{"boundary", boundary}, {"physics", id}, {"route", route}, {"accepted_on", "wall boundaries at rest"}});
}
// A force or torque of a flow on a row whose engine exposes no reaction.
inline Error force_not_offered(const std::string& name, const std::string& output_kind, const std::string& boundary, const std::string& id,
                               const std::string& route, const std::string& engine) {
  return coded("output '" + name + "': a " + output_kind + " on '" + boundary + "' is not offered on the route '" + route + "'",
               "/outputs/" + name,
               "The engine of this route (" + engine + ") exposes no reaction, so the route has no force and no torque output. "
               "Remove the output, or compute it in a document that solves '" + id + "' alone: the Stokes and Navier-Stokes routes "
               "have force outputs.",
               codes::output_force_not_offered,
               {{"output", name}, {"kind", output_kind}, {"boundary", boundary}, {"physics", id}, {"route", route}});
}
// A force or a torque that belongs to no flow: it names a physics that is not a flow (`type` is its type), or, with `id`
// empty, the document has no flow.
inline Error force_without_flow(const std::string& name, const std::string& output_kind, const std::string& boundary, const std::string& id,
                                const std::string& type) {
  return coded("output '" + name + "': a " + output_kind + " on '" + boundary + "' needs a flow, and " +
                   (id.empty() ? std::string("the document has no fluid.incompressible physics")
                               : "the physics '" + id + "' is a " + type + ", not a fluid.incompressible"),
               "/outputs/" + name,
               "A force or a torque is the reaction of a flow solve on a boundary. " +
                   (id.empty() ? std::string("Add a fluid.incompressible physics that has conditions on the boundary, or remove the output.")
                               : "Write the id of a fluid.incompressible physics in \"physics\", or remove the output."),
               codes::output_force_not_offered, {{"output", name}, {"kind", output_kind}, {"boundary", boundary}, {"physics", id}, {"type", type}});
}

// ---------------------------------------------------------------------------------------------------- deferred
// The resolution of the named boundaries: it needs the mesher, or the info file of an imported mesh (`file`).
inline Error deferred_boundary_resolution(const Strs& boundaries, const std::string& file) {
  const bool one = boundaries.size() == 1;
  const std::string when = file.empty() ? "when the geometry is built" : "when the imported mesh '" + file + "' is read";
  return coded(std::string("the named ") + (one ? "boundary " : "boundaries ") + quoted_and(boundaries) + (one ? " is" : " are") +
                   " resolved " + when,
               "/boundaries",
               std::string("This validation reads the structure of the document only. ") +
                   (file.empty() ? "The mesher finds the boundary entities that each selector names"
                                 : "The groups of the mesh's info file give the entities that each selector names") +
                   "; a boundary that selects nothing is refused then.",
               codes::boundary_empty_selection, json{{"boundaries", jarr(boundaries)}, {"file", file.empty() ? json(nullptr) : json(file)}});
}
// The partition rule of one instance (design note 5.5).  `keys`: the keys of its boundary conditions.
inline Error deferred_partition(const std::string& id, const Strs& keys, const std::string& file) {
  const std::string when = file.empty() ? "when the geometry is built" : "when the imported mesh '" + file + "' is read";
  return coded("physics '" + id + "': whether its boundary conditions cover the boundary of its domain exactly once is checked " + when,
               "/physics/" + id + "/boundary_conditions",
               "Every part of the boundary of the domain of '" + id + "' takes exactly one condition, and a boundary interior to the "
               "domain takes none. This validation reads the structure of the document only; an uncovered part, a part covered "
               "twice and a condition on an interior boundary are refused then.",
               codes::physics_boundary_uncovered,
               {{"physics", id}, {"conditions", jarr(keys)},
                {"checks", jarr(Strs{codes::physics_boundary_uncovered, codes::physics_boundary_double, codes::physics_boundary_interior})}});
}

// ---------------------------------------------------------------------------------------------------- resolved boundaries
// The texts of the checks that need the resolved boundaries (design note 5.2, 5.5, 10.3): the empty selection, a partly
// matching entity, the partition rule of an instance and the outputs on a boundary.  They take the diagnostics of
// selection::resolve, selection::partition and selection::place, the entity table for the counts, measures, first
// points and adjacent regions, and the document's names.  A count is "1 edge" or "N edges"; a measure is the sum of
// the lengths in ascending tag order, in metres; the first point is the point of the entity whose point is the
// smallest lexicographically, written "(x, y) = (...)" (planar) or "(r, z) = (...)" (axisymmetric); the adjacent pairs
// are the distinct side pairs of the entities, each in region-pair order (document order, "exterior" last), the pairs in
// that order.  cpp/tests/test_model_selection.cpp gives every text.  check_boundaries (model_validate.hpp) raises them.

inline std::string curve_hint() { return "To select part of an edge, add a curve along that part and use {\"curve\": NAME}."; }
inline std::string edges_text(size_t n) { return n == 1 ? std::string("1 edge") : std::to_string(n) + " edges"; }
// The coordinates of a table, in component order.
inline Strs coordinate_names(const std::string& geometry) {
  if (geometry == "axisymmetric") return {"r", "z"};
  if (geometry == "3d") return {"x", "y", "z"};
  return {"x", "y"};
}
// The entities of `tags`, in ascending tag order; no tag, or a tag that is not in the table, is a defect of the caller.
inline std::vector<const selection::Entity*> entities_of(const selection::Table& table, std::vector<int> tags) {
  if (tags.empty()) throw std::logic_error("a finding on the resolved boundaries without an entity");
  std::sort(tags.begin(), tags.end());
  std::vector<const selection::Entity*> out;
  for (int tag : tags) {
    const auto it = std::find_if(table.entities.begin(), table.entities.end(), [tag](const selection::Entity& e) { return e.tag == tag; });
    if (it == table.entities.end()) throw std::logic_error("no entity " + std::to_string(tag) + " in the entity table");
    out.push_back(&*it);
  }
  return out;
}
inline double measure_m(const std::vector<const selection::Entity*>& es) {
  double m = 0.0;
  for (const selection::Entity* e : es) m += e->length_m;
  return m;
}
inline const selection::Entity& first_entity(const std::vector<const selection::Entity*>& es) {
  return **std::min_element(es.begin(), es.end(), [](const selection::Entity* a, const selection::Entity* b) {
    return std::lexicographical_compare(a->point, a->point + 3, b->point, b->point + 3);
  });
}
inline json first_point(const selection::Table& table, const std::vector<const selection::Entity*>& es) {
  const selection::Entity& e = first_entity(es);
  json p = json::array();
  for (size_t c = 0; c < coordinate_names(table.geometry).size(); ++c) p.push_back(e.point[c]);
  return p;
}
inline std::string first_point_text(const selection::Table& table, const std::vector<const selection::Entity*>& es) {
  const selection::Entity& e = first_entity(es);
  const Strs names = coordinate_names(table.geometry);
  Strs values;
  for (size_t c = 0; c < names.size(); ++c) values.push_back(num_text(e.point[c]));
  return "(" + join(names) + ") = (" + join(values) + ")";
}
// The distinct side pairs of the entities, each in region-pair order, the pairs in that order.
inline std::vector<std::pair<std::string, std::string>> side_pairs(const selection::Table& table,
                                                                   const std::vector<const selection::Entity*>& es) {
  const auto rank = [&table](const std::string& s) {
    const auto it = std::find(table.regions.begin(), table.regions.end(), s);
    if (it != table.regions.end()) return std::make_pair(size_t(it - table.regions.begin()), s);
    return std::make_pair(table.regions.size() + (s == "exterior" ? 0 : 1), s);
  };
  std::vector<std::pair<std::pair<size_t, std::string>, std::pair<size_t, std::string>>> keyed;
  for (const selection::Entity* e : es) {
    auto a = rank(e->side[0]), b = rank(e->side[1]);
    if (b < a) std::swap(a, b);
    if (std::find(keyed.begin(), keyed.end(), std::make_pair(a, b)) == keyed.end()) keyed.emplace_back(a, b);
  }
  std::sort(keyed.begin(), keyed.end());
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto& [a, b] : keyed) out.emplace_back(a.second, b.second);
  return out;
}
// "between 'r' and 's'", "between 'r' and the exterior", "inside 'r'", joined with " and ".
inline std::string pairs_text(const std::vector<std::pair<std::string, std::string>>& pairs) {
  Strs parts;
  for (const auto& [r, s] : pairs)
    parts.push_back(r == s ? "inside '" + r + "'" : "between '" + r + "' and " + (s == "exterior" ? std::string("the exterior") : "'" + s + "'"));
  return join(parts, " and ");
}
inline json pairs_json(const std::vector<std::pair<std::string, std::string>>& pairs) {
  json a = json::array();
  for (const auto& [r, s] : pairs) a.push_back(json::array({r, s}));
  return a;
}
// Where an interior selection lies, for the instance `id` (the regions are its own).
inline std::string region_sentence(const std::string& key, const std::string& id, const std::vector<std::pair<std::string, std::string>>& pairs) {
  if (pairs.size() == 1 && pairs[0].first == pairs[0].second)
    return "'" + key + "' lies inside '" + pairs[0].first + "', which belongs to '" + id + "'";
  if (pairs.size() == 1) return "'" + key + "' lies " + pairs_text(pairs) + ", and both regions belong to '" + id + "'";
  return "'" + key + "' lies " + pairs_text(pairs) + ", and these regions belong to '" + id + "'";
}
// The domain of an of or between as the document writes it: 'D', or 'A' and 'B' for a list; the exterior.
inline std::string domain_text(const json& d) {
  if (d.is_string()) return d.get<std::string>() == "exterior" ? std::string("the exterior") : "'" + d.get<std::string>() + "'";
  if (d.is_array()) {
    Strs v;
    for (const json& r : d) v.push_back(r.is_string() ? r.get<std::string>() : r.dump());
    return quoted_and(v);
  }
  return d.dump();
}
// The constraints of a where filter in coordinate order: "x = v", "x in [a, b]", joined with " and "; `all_equal` tells
// whether every constraint is an equality.
inline std::string constraints_text(const std::string& geometry, const json& where, bool& all_equal) {
  Strs keys = coordinate_names(geometry), parts;
  for (auto it = where.begin(); it != where.end(); ++it)
    if (std::find(keys.begin(), keys.end(), it.key()) == keys.end()) keys.push_back(it.key());
  all_equal = true;
  for (const std::string& c : keys) {
    if (!where.contains(c)) continue;
    const json& v = where.at(c);
    if (v.is_array() && v.size() == 2) {
      parts.push_back(c + " in [" + value_text(v[0]) + ", " + value_text(v[1]) + "]");
      all_equal = false;
    } else {
      parts.push_back(c + " = " + value_text(v.is_array() && v.size() == 1 ? v[0] : v));
    }
  }
  return join(parts, " and ");
}
// The candidates of an empty selection as text: "x = v (L m)" for a line, "x in [a, b], y in [c, d] (L m)" for a box.
inline std::string candidates_text(const std::string& geometry, const json& candidates) {
  Strs items;
  for (const json& c : candidates) {
    std::string item;
    if (c.contains("where")) {
      for (auto it = c.at("where").begin(); it != c.at("where").end(); ++it) item = it.key() + " = " + value_text(it.value());
    } else if (c.contains("bbox")) {
      Strs parts;
      for (const std::string& k : coordinate_names(geometry))
        if (c.at("bbox").contains(k))
          parts.push_back(k + " in [" + value_text(c.at("bbox").at(k)[0]) + ", " + value_text(c.at("bbox").at(k)[1]) + "]");
      item = join(parts);
    }
    items.push_back(item + " (" + value_text(c.value("measure_m", json(0.0))) + " m)");
  }
  return join(items);
}
// "Did you mean {"c": v}, or is the geometry shifted?": c is the first equality coordinate with a candidate line in that
// coordinate, v the value of the nearest such line (the smaller of two at the same distance).
inline std::string did_you_mean_shift(const std::string& geometry, const json& where, const json& candidates) {
  for (const std::string& c : coordinate_names(geometry)) {
    if (!where.contains(c)) continue;
    const json& w = where.at(c);
    if (w.is_array() && w.size() == 2) continue;
    const double v = (w.is_array() ? w[0] : w).get<double>();
    bool found = false;
    double best = 0.0;
    for (const json& cand : candidates) {
      if (!cand.contains("where") || !cand.at("where").contains(c)) continue;
      const double x = cand.at("where").at(c).get<double>();
      const double d = std::fabs(x - v), db = std::fabs(best - v);
      if (!found || d < db || (d == db && x < best)) best = x, found = true;
    }
    if (found) return "Did you mean {\"" + c + "\": " + num_text(best) + "}, or is the geometry shifted?";
  }
  return "Is the geometry shifted?";
}

// The filtered selectors of a boundary as the document writes it (model_selection.hpp, Resolution): its of and between
// selectors with a where filter that add entities to it, each with its JSON pointer below /boundaries/<name>.
inline void filtered_selectors(const json& s, const std::string& pointer, std::vector<std::pair<std::string, const json*>>& out) {
  if (!s.is_object()) return;
  if ((s.contains("of") || s.contains("between")) && s.contains("where")) {
    out.emplace_back(pointer, &s);
  } else if (s.contains("union") && s.at("union").is_array()) {
    for (size_t i = 0; i < s.at("union").size(); ++i) filtered_selectors(s.at("union")[i], pointer + "/union/" + std::to_string(i), out);
  } else if (s.contains("difference") && s.at("difference").is_array() && !s.at("difference").empty()) {
    filtered_selectors(s.at("difference")[0], pointer + "/difference/0", out);
  }
}
// The one filtered selector of a written boundary, or null; `where` receives the pointer of its where filter, else that
// of the boundary.
inline const json* one_filtered_selector(const std::string& boundary, const json& written, std::string& where) {
  std::vector<std::pair<std::string, const json*>> leaves;
  filtered_selectors(written, "", leaves);
  where = "/boundaries/" + boundary + (leaves.size() == 1 ? leaves[0].first + "/where" : "");
  return leaves.size() == 1 ? leaves[0].second : nullptr;
}

// boundary.empty_selection, the error.  `written`: the selector as the document writes it; `empty`:
// Resolution::empty of the boundary.
inline Error empty_selection(const std::string& boundary, const json& written, const selection::Table& table, const json& empty) {
  const json candidates = empty.is_object() && empty.contains("candidates") ? empty.at("candidates") : json::array();
  const json partial = empty.is_object() && empty.contains("partial") ? empty.at("partial") : json::array();
  const bool axis_only = empty.is_object() && empty.value("axis_only", false);
  std::string pointer;
  const json* leaf = one_filtered_selector(boundary, written, pointer);
  json where = empty.is_object() && empty.contains("where") ? empty.at("where") : json(nullptr);
  if (!where.is_object() && leaf) where = leaf->at("where");
  const bool one = leaf && where.is_object();  // the constraints and the domains of one filtered selector
  const bool between = one && leaf->contains("between") && leaf->at("between").is_array() && leaf->at("between").size() == 2;
  const std::string owner = between ? "between " + domain_text(leaf->at("between")[0]) + " and " + domain_text(leaf->at("between")[1])
                            : one && leaf->contains("of") ? "of " + domain_text(leaf->at("of"))
                                                          : std::string();
  std::string hint;
  bool all_equal = true;
  const std::string constraints = one ? constraints_text(table.geometry, where, all_equal) : std::string();
  const std::string on = all_equal ? "on " : "within ";
  if (axis_only) {
    hint = "'" + boundary + "' lies only on the axis r = 0, which is reserved: no named boundary includes it, and every physics applies "
           "its axis condition there automatically.";
  } else if (!partial.empty()) {
    const auto es = entities_of(table, partial.get<std::vector<int>>());
    hint = edges_text(es.size()) + (one ? " " + owner : std::string()) + (es.size() == 1 ? " lies" : " lie") + " partly " +
           (one ? on + constraints : "within a where filter of '" + boundary + "'") + " (" + num_text(measure_m(es)) + " m, first at " +
           first_point_text(table, es) + "); a where filter keeps an edge only when the whole edge satisfies it. " + curve_hint();
  } else if (one && !candidates.empty()) {
    hint = (between ? "no edge " : "no boundary edge ") + owner + " lies " + on + constraints + ". Edges " + owner + ": " +
           candidates_text(table.geometry, candidates) + ". " + did_you_mean_shift(table.geometry, where, candidates) + " " + curve_hint();
  } else {
    hint = "'" + boundary + "' selects no boundary edge of this geometry. `axmp boundaries` lists every boundary edge and the boundaries "
           "that select it.";
  }
  return coded("boundary '" + boundary + "' selects nothing", pointer, hint,
               codes::boundary_empty_selection,
               {{"selector", written}, {"candidates", candidates}, {"partial", partial}, {"axis_only", axis_only}});
}
// boundary.partial_match (a warning): `tags`, the partial matches of a non-empty selection.
inline Error partial_match(const std::string& boundary, const json& written, const selection::Table& table, const std::vector<int>& tags) {
  const auto es = entities_of(table, tags);
  const std::string count = edges_text(es.size());
  std::string pointer;
  one_filtered_selector(boundary, written, pointer);
  return coded("boundary '" + boundary + "' leaves out " + count + (es.size() == 1 ? " that satisfies" : " that satisfy") +
                   " its where filter in part",
               pointer,
               count + ", " + num_text(measure_m(es)) + " m, first at " + first_point_text(table, es) + ": a where filter keeps an edge "
               "only when the whole edge satisfies it. " + curve_hint(),
               codes::boundary_partial_match,
               {{"boundary", boundary}, {"entities", es.size()}, {"measure_m", measure_m(es)}, {"first_point", first_point(table, es)}});
}
// boundary.between_overlap: `pointer` is the selector's "between" (a validation check: no table).
inline Error between_overlap(const std::string& boundary, const std::string& pointer, const std::string& first, const std::string& second,
                             const Strs& shared) {
  return coded("boundary '" + boundary + "': the two domains of 'between' share the " + (shared.size() == 1 ? "region " : "regions ") +
                   quoted_and(shared),
               pointer,
               "the normal of a 'between' selection points from the first domain to the second, which is undefined on an edge whose two "
               "sides lie in both. Give two domains without a common region, or use {\"of\": DOMAIN}.",
               codes::boundary_between_overlap, {{"boundary", boundary}, {"between", jarr(Strs{first, second})}, {"shared", jarr(shared)}});
}
// physics.boundary_uncovered: `domain`, the instance's domain as one name for the example; `keys`, its condition keys;
// `remaining_kind`, the kind the example gives "remaining"; `document`, the file as the user named it, or "".
inline Error boundary_uncovered(const std::string& id, const std::string& domain, const Strs& keys, const std::string& remaining_kind,
                                const std::string& document, const selection::Table& table, const std::vector<int>& tags) {
  const auto es = entities_of(table, tags);
  const auto pairs = side_pairs(table, es);
  Strs except;
  for (const std::string& k : keys) except.push_back("\"" + k + "\"");
  const std::string example = "\"walls\": {\"of\": " + (domain.empty() ? std::string("DOMAIN") : "\"" + domain + "\"") +
                              (keys.empty() ? std::string() : ", \"except\": [" + join(except) + "]") + "}";
  return coded("physics '" + id + "': part of the boundary of its domain has no condition", "/physics/" + id + "/boundary_conditions",
               "uncovered: " + edges_text(es.size()) + ", " + num_text(measure_m(es)) + " m, " + pairs_text(pairs) + ", first at " +
                   first_point_text(table, es) + ". Name them (for example " + example + ") and give them a condition, or add "
                   "\"remaining\": {\"kind\": \"" + remaining_kind + "\"} to this physics. `axmp boundaries " +
                   (document.empty() ? std::string("DOCUMENT") : document) + "` lists every boundary.",
               codes::physics_boundary_uncovered,
               {{"physics", id}, {"entities", es.size()}, {"measure_m", measure_m(es)}, {"first_point", first_point(table, es)},
                {"adjacent", pairs_json(pairs)}, {"conditions", jarr(keys)}});
}
// physics.boundary_double: one finding per pair of keys, `first` before `second` in the order of the keys.
inline Error boundary_double(const std::string& id, const std::string& first, const std::string& second, const selection::Table& table,
                             const std::vector<int>& tags) {
  const auto es = entities_of(table, tags);
  const std::string where = es.size() == 1 ? "1 edge at " + first_point_text(table, es)
                                           : edges_text(es.size()) + ", " + num_text(measure_m(es)) + " m, first at " + first_point_text(table, es);
  return coded("physics '" + id + "': boundaries '" + first + "' and '" + second + "' both cover " + where,
               "/physics/" + id + "/boundary_conditions/" + second,
               "one condition per boundary face. Remove one, or exclude the overlap: {\"of\": ..., \"except\": [\"" + first + "\"]}.",
               codes::physics_boundary_double,
               {{"physics", id}, {"keys", jarr(Strs{first, second})}, {"entities", es.size()}, {"measure_m", measure_m(es)},
                {"first_point", first_point(table, es)}});
}
// physics.boundary_interior: `tags`, the key's entities with both sides in the instance; `whole`, every entity of the key.
inline Error boundary_interior(const std::string& id, const std::string& key, bool whole, const selection::Table& table,
                               const std::vector<int>& tags) {
  const auto es = entities_of(table, tags);
  const auto pairs = side_pairs(table, es);
  const json details = {{"physics", id}, {"boundary", key}, {"whole", whole}, {"entities", es.size()}, {"measure_m", measure_m(es)},
                        {"first_point", first_point(table, es)}, {"adjacent", pairs_json(pairs)}};
  const std::string where = "/physics/" + id + "/boundary_conditions/" + key;
  if (whole)
    return coded("physics '" + id + "': boundary '" + key + "' is not on the boundary of its domain", where,
                 region_sentence(key, id, pairs) + ". An interior boundary can be used in outputs, not as a boundary condition.",
                 codes::physics_boundary_interior, details);
  return coded("physics '" + id + "': part of boundary '" + key + "' is interior to its domain", where,
               edges_text(es.size()) + " of '" + key + "', " + num_text(measure_m(es)) + " m, first at " + first_point_text(table, es) +
                   (es.size() == 1 ? ", lies " : ", lie ") + pairs_text(pairs) + ", inside the domain of '" + id + "'. A condition acts on "
                   "all of its boundary, and an interior boundary can be used in outputs, not as a boundary condition: exclude the "
                   "interior part with \"except\", or key a boundary without it.",
               codes::physics_boundary_interior, details);
}
// physics.boundary_outside_domain: `regions`, the instance's regions; `tags`, the key's selection.
inline Error boundary_outside_domain(const std::string& id, const std::string& key, const Strs& regions, const selection::Table& table,
                                     const std::vector<int>& tags) {
  const auto pairs = side_pairs(table, entities_of(table, tags));
  return coded("physics '" + id + "': boundary '" + key + "' lies outside its domain", "/physics/" + id + "/boundary_conditions/" + key,
               "'" + key + "' lies " + pairs_text(pairs) + ", and the domain of '" + id + "' is " + quoted_and(regions) + ". A condition key "
               "names a boundary on the boundary of the domain: key another boundary, or change the domains of '" + id + "'.",
               codes::physics_boundary_outside_domain,
               {{"physics", id}, {"boundary", key}, {"adjacent", pairs_json(pairs)}, {"domain", jarr(regions)}});
}
// output.force_on_interior: `output_kind`, force | torque; `tags`, the selection's interior entities.
inline Error force_on_interior(const std::string& output, const std::string& output_kind, const std::string& boundary, const std::string& id,
                               const selection::Table& table, const std::vector<int>& tags) {
  const auto pairs = side_pairs(table, entities_of(table, tags));
  return coded("output '" + output + "': a " + output_kind + " on '" + boundary + "' is not available: '" + boundary +
                   "' is interior to the domain of '" + id + "'",
               "/outputs/" + output,
               "a force or a torque is the engine's reaction on the boundary of the domain, and " + region_sentence(boundary, id, pairs) +
                   ". Through an interior section, use a flux, an integral, an average, a maximum or a minimum.",
               codes::output_force_on_interior,
               {{"output", output}, {"kind", output_kind}, {"boundary", boundary}, {"physics", id}, {"adjacent", pairs_json(pairs)}});
}
// output.boundary_not_in_domain: `regions`, the instance's regions; `tags`, the selection.
inline Error boundary_not_in_domain(const std::string& output, const std::string& boundary, const std::string& id, const Strs& regions,
                                    const selection::Table& table, const std::vector<int>& tags) {
  const auto pairs = side_pairs(table, entities_of(table, tags));
  return coded("output '" + output + "': boundary '" + boundary + "' is not in the domain of '" + id + "'", "/outputs/" + output,
               "'" + boundary + "' lies " + pairs_text(pairs) + ", and the domain of '" + id + "' is " + quoted_and(regions) +
                   ". An output of '" + id + "' is computed on a boundary or a section of its domain.",
               codes::output_boundary_not_in_domain,
               {{"output", output}, {"boundary", boundary}, {"physics", id}, {"adjacent", pairs_json(pairs)}, {"domain", jarr(regions)}});
}
// output.force_needs_stress_form, resolved: a force on a boundary that is not a key of the instance, part of which a key
// `key` covers whose condition is not a wall at rest (`condition`, its kind; `moving`, a wall with a velocity).
inline Error force_needs_stress_form_on_part(const std::string& output, const std::string& output_kind, const std::string& boundary,
                                             const std::string& key, const std::string& condition, bool moving, const std::string& route) {
  const std::string what = moving ? "part of '" + boundary + "' is the moving wall '" + key + "'."
                                  : "part of '" + boundary + "' carries the condition '" + condition + "' of '" + key + "'.";
  return coded(laplacian_force_message(output, output_kind, boundary), "/outputs/" + output, what + " " + laplacian_force_reason(),
               codes::output_force_needs_stress_form,
               {{"boundary", boundary}, {"key", key}, {"condition", moving ? std::string("wall with velocity_m_s") : condition},
                {"route", route}, {"accepted_on", "wall boundaries at rest"}});
}

// ---------------------------------------------------------------------------------------------------- warnings
// `feature`: physics_type | coupling_term | route.  `validation`: the validation page of a type, or "".
inline Error beta_feature(const std::string& feature, const std::string& name, const std::string& where, const std::string& validation) {
  const std::string what = feature == "physics_type" ? "physics type" : feature == "coupling_term" ? "coupling term" : "route";
  const std::string hint = validation.empty() ? std::string("Beta: no published benchmark value is compared.")
                                              : "Beta: verified against manufactured, closed-form and cross-code solutions; no published "
                                                "benchmark value is compared. See " + validation + ".";
  return coded(what + " '" + name + "' is beta", where, hint, codes::beta_feature, {{"feature", feature}, {"name", name}});
}
inline Error outlet_stress_form(const std::string& id, const std::string& boundary, const std::string& kind) {
  return coded("physics '" + id + "': the " + (kind == "outlet" ? std::string("outlet") : "open boundary") + " '" + boundary +
                   "' imposes zero total traction with the stress viscous form, which a fully developed profile does not satisfy",
               "/physics/" + id + "/boundary_conditions/" + boundary,
               "The tangential traction of a developed profile is mu du_t/dn, so the flow bends near the outlet and the mean outlet "
               "pressure is not exactly the value written. Use \"viscous_form\": \"laplacian\" when the outlet is straight and normal "
               "to the flow, or a longer outlet section. The developed-outflow condition is roadmap item F2b.",
               codes::fluid_outlet_stress_form, {{"boundary", boundary}, {"kind", kind}, {"viscous_form", "stress"}});
}
inline Error pseudo_traction(const std::string& id, const std::string& boundary, const std::string& kind) {
  return coded("physics '" + id + "': the " + kind + " condition on '" + boundary + "', a curved boundary, imposes a pseudo-traction "
               "with the Laplacian viscous form, not the physical traction",
               "/physics/" + id + "/boundary_conditions/" + boundary,
               "With \"viscous_form\": \"laplacian\" the natural condition is mu du/dn - p n, which equals the Cauchy traction only on "
               "straight boundaries. Use \"viscous_form\": \"stress\" for a physical traction on a curved boundary.",
               codes::fluid_pseudo_traction, {{"boundary", boundary}, {"kind", kind}, {"viscous_form", "laplacian"}});
}
inline Error uncoupled_pair(const std::string& a, const std::string& b, const Strs& regions, const Strs& candidates) {
  return coded("physics '" + a + "' and '" + b + "' both act on " + quoted_and(regions) + ", and no coupling term connects them: they are "
               "solved uncoupled",
               "/couplings",
               "This is valid (forced convection without buoyancy, for example), and nothing is inferred. The terms that can connect "
               "them are: " + join(candidates) + ". Add one to \"couplings\" if they interact.",
               codes::coupling_uncoupled_pair, {{"physics", jarr(Strs{a, b})}, {"regions", jarr(regions)}, {"candidates", jarr(candidates)}});
}

// ---------------------------------------------------------------------------------------------------- where a term acts
inline Error empty_overlap(int coupling, const std::string& term, const std::string& from, const std::string& to, const Strs& from_regions,
                           const Strs& to_regions) {
  return coded("coupling '" + term + "' from '" + from + "' to '" + to + "': no shared region", "/couplings/" + std::to_string(coupling),
               "'" + from + "' is on " + quoted(from_regions) + " and '" + to + "' on " + quoted(to_regions) +
                   "; a domain term acts where both are. Give them a common region, or remove the term.",
               codes::coupling_empty_overlap,
               {{"term", term}, {"from", from}, {"to", to}, {"from_regions", jarr(from_regions)}, {"to_regions", jarr(to_regions)}});
}
inline Error restriction_not_offered(int coupling, const std::string& term, const std::string& from, const std::string& to,
                                     const std::string& route, const std::string& in, const Strs& in_regions, const Strs& shared) {
  return coded("coupling '" + term + "' from '" + from + "' to '" + to + "': the route '" + route + "' cannot restrict the term to '" + in + "'",
               "/couplings/" + std::to_string(coupling) + "/in",
               "On this route the term acts on the whole shared domain (" + quoted(shared) + "). Remove \"in\", or make it the whole "
               "shared domain.",
               codes::coupling_restriction_not_offered, {{"term", term}, {"route", route}, {"in", jarr(in_regions)}, {"shared", jarr(shared)}});
}

// What the `in` of a term or a source names beyond the domain it must stay inside: some regions of it (the others are in),
// or all of them.
inline std::string in_names_outside(const std::string& in, const Strs& in_regions, const Strs& outside) {
  if (outside.size() < in_regions.size()) return "'" + in + "' also names " + quoted_and(outside);
  if (in_regions.size() == 1 && in_regions[0] == in) return "'" + in + "' is not in it";
  return "none of the regions of '" + in + "' (" + quoted_and(outside) + ") is in it";
}
// `in` of a domain term with a region outside the shared domain of its two instances.
inline Error in_outside_shared_domain(int coupling, const std::string& term, const std::string& from, const std::string& to,
                                      const std::string& in, const Strs& in_regions, const Strs& shared, const Strs& outside) {
  return coded("coupling '" + term + "' from '" + from + "' to '" + to + "': \"in\": \"" + in + "\" is not inside the shared domain",
               "/couplings/" + std::to_string(coupling) + "/in",
               "A domain term acts where both physics are, here " + quoted(shared) + ", and \"in\" restricts it to a part of that; " +
                   in_names_outside(in, in_regions, outside) + ". Remove \"in\", or name regions of the shared domain.",
               codes::coupling_selection_outside_support,
               {{"term", term}, {"from", from}, {"to", to}, {"in", jarr(in_regions)}, {"shared", jarr(shared)}, {"outside", jarr(outside)}});
}
// `in` of a source with a region outside the domain of its instance.
inline Error source_outside_domain(const std::string& id, size_t source, const std::string& kind, const std::string& in,
                                   const Strs& in_regions, const Strs& regions, const Strs& outside) {
  return coded("physics '" + id + "': \"in\": \"" + in + "\" of its " + kind + " source is not inside the domain of the physics",
               "/physics/" + id + "/sources/" + std::to_string(source) + "/in",
               "'" + id + "' is on " + quoted(regions) + ", and " + in_names_outside(in, in_regions, outside) +
                   ". Remove \"in\" for a source over the whole instance, or name regions of its domain.",
               codes::physics_source_outside_domain,
               {{"physics", id}, {"source", source}, {"kind", kind}, {"in", jarr(in_regions)}, {"regions", jarr(regions)},
                {"outside", jarr(outside)}});
}

// ---------------------------------------------------------------------------------------------------- patch
// An operation of a patch that cannot be applied.  `op` and `path` are as written (null when the operation has none).
inline Error patch_operation_failed(size_t index, const json& op, const json& path, const std::string& reason) {
  return coded("patch operation " + std::to_string(index) + " failed: " + reason, "/" + std::to_string(index),
               "The document is unchanged. Operations are RFC 6902 (JSON Patch); paths are JSON pointers like the \"where\" of an error.",
               codes::patch_operation_failed, {{"index", index}, {"op", op}, {"path", path}});
}
inline Error patch_not_a_list(const std::string& got) {
  return coded("patch operations must be a list; got " + got, "",
               "The document is unchanged. Operations are RFC 6902 (JSON Patch); paths are JSON pointers like the \"where\" of an error.",
               codes::patch_operation_failed);
}

// ---------------------------------------------------------------------------------------------------- the switch
// The operations of this version that take a model document, in the order listed under details.serves_model_documents.
inline Strs serves_model_documents() { return {"validate", "resolve", "explain", "patch", "boundaries", "mesh"}; }
inline std::string serves_tail() { return "In this version a model document is taken by: " + join(serves_model_documents()) + "."; }
// "'<name>'" of a document with a string name, else "" (the callers then say "this document" or "the document").
inline std::string document_name(const json& doc) {
  if (doc.is_object() && doc.contains("name") && doc["name"].is_string()) return "'" + doc["name"].get<std::string>() + "'";
  return "";
}
inline Error section_needs_physics(const std::string& key) {
  return coded("'" + key + "' belongs to a model document, and this document has no 'physics' section", "/" + key,
               "add \"physics\": {...} (run `axmp physics` for the available types), or remove '" + key + "'.",
               codes::section_needs_physics);
}
// `operation` null: the backstop of parse_design, which cannot name its caller.
inline Error em_only_operation(const json& doc, const json& operation) {
  const std::string op = operation.is_string() ? operation.get<std::string>() : std::string();
  const std::string who = operation.is_string() ? "'" + op + "'" : std::string("this operation");
  const std::string name = document_name(doc);
  const std::string what = name.empty() ? std::string("this document is a model document") : name + " is a model document";
  std::string hint;
  if (op == "sensitivities" || op == "gradient")
    hint = "adjoint sensitivities are an electromagnetic capability. ";
  else if (op == "optimize" || op == "global_optimize" || op == "optimize_design" || op == "optimize_global")
    hint = "this version optimizes electromagnetic designs. ";
  else
    hint = "this operation needs an electromagnetic design (a document without 'physics'). ";
  return coded(who + " works on electromagnetic designs, and " + what + " (it has a 'physics' section)", "/physics",
               hint + serves_tail(), codes::em_only_operation,
               json{{"operation", operation}, {"serves_model_documents", jarr(serves_model_documents())}});
}
inline Error operation_not_implemented(const json& doc, const std::string& operation) {
  const std::string name = document_name(doc);
  const std::string who = name.empty() ? std::string("the document has a 'physics' section") : name + " has a 'physics' section";
  return coded("'" + operation + "' of a model document is not implemented yet (" + who + ")", "/physics",
               "solving a model document is not implemented in this version; `axmp validate` reports what the "
               "document needs. " + serves_tail(),
               codes::route_not_implemented,
               json{{"operation", operation}, {"implemented", false}, {"serves_model_documents", jarr(serves_model_documents())}});
}
inline Error duplicate_key(const std::string& key, const std::string& pointer, const std::string& origin) {
  return coded("duplicate key '" + key + "' in '" + origin + "'", pointer,
               "a model document may not repeat a key in an object: JSON keeps only the last value, silently. Keep one of "
               "the two.",
               codes::duplicate_key, json{{"key", key}});
}

// ---------------------------------------------------------------------------------------------------- the meshed model
// The internal errors of the meshed model and of the exporters of an instance mesh (model_mesh.hpp): a valid document
// never reaches them.  `regions`: the instance's regions.
inline std::string report_it() {
  return "This is a defect of axionMultiPhysics, not of the document; please report it with the document.";
}
inline Error mesh_export_nodes_for_p1(const Strs& regions, int geometry_order) {
  return coded("chemistry mesh export: the instance mesh of " + quoted_and(regions) + " has nodes (geometry order " +
                   std::to_string(geometry_order) + "), and the field order is 1",
               "",
               "A P1 chemistry field is solved on straight elements, and a curved mesh is not straightened silently; the "
               "validation lowers the geometry order of such a model to 1. " + report_it(),
               codes::mesh_export, {{"regions", jarr(regions)}, {"geometry_order", geometry_order}, {"field_order", 1}});
}
inline Error mesh_export_not_straight(const std::string& route, const Strs& regions, int geometry_order) {
  return coded("route '" + route + "' needs straight elements, and the instance mesh of " + quoted_and(regions) +
                   " has nodes (geometry order " + std::to_string(geometry_order) + ")",
               "",
               "The validation lowers the geometry order of a model to 1 when one of its routes needs straight elements. " +
                   report_it(),
               codes::mesh_export, {{"route", route}, {"regions", jarr(regions)}, {"geometry_order", geometry_order}});
}
// A boundary face of an instance mesh with no boundary element of the master mesh on it.
inline Error mesh_export_missing_atom(const Strs& regions, const double p[2], const double q[2]) {
  return coded("the instance mesh of " + quoted_and(regions) + " has a boundary face, from (" + num_text(p[0]) + ", " +
                   num_text(p[1]) + ") to (" + num_text(q[0]) + ", " + num_text(q[1]) +
                   "), with no boundary element of the master mesh on it: its atom is missing",
               "",
               "Every region interface and every exterior edge of a meshed model carries an atom, so the mesh file is "
               "incomplete. " + report_it(),
               codes::mesh_export, {{"regions", jarr(regions)}, {"from", {p[0], p[1]}}, {"to", {q[0], q[1]}}});
}
// The mesher's files and the library disagree (`what` says how).
inline Error mesh_export_files(const std::string& what, const std::string& file) {
  return coded("the mesher's files do not match the model: " + what, "",
               "A mesh file and its info file come from one run of axmp-mesher on this document; the mismatch is in '" + file +
                   "'. " + report_it(),
               codes::mesh_export, {{"file", file}});
}

// ---------------------------------------------------------------------------------------------------- meshing a model document
// The mesher's probe failed (`path`: the program probed, "" when none was found; `reason`: the probe's account).  The Linux
// libraries that gmsh's wheel needs are named when the loader's message names a library other than gmsh's own.
inline bool names_other_library(const std::string& reason) {
  const std::string key = "shared libraries: ";
  const size_t at = reason.find(key);
  if (at == std::string::npos) return false;
  const std::string lib = reason.substr(at + key.size(), reason.find(':', at + key.size()) - (at + key.size()));
  return !lib.empty() && lib.rfind("libgmsh", 0) != 0;
}
inline Error requires_mesher(const std::string& path, const std::string& reason) {
  std::string hint = "A wheel installed without gmsh: `pip install \"axionmultiphysics[mesh]\"`. A source build: configure with "
                     "-DAXMP_BUILD_MESHER=ON and the gmsh SDK. A mesher installed elsewhere: $AXMP_MESHER (or --mesher) names it.";
  if (names_other_library(reason))
    hint += " The Linux wheel of gmsh needs a few X11/OpenGL libraries even without a display (Debian/Ubuntu: "
            "`sudo apt-get install libglu1-mesa libgl1 libxcursor1 libxft2 libxinerama1`).";
  return coded("the mesher does not run in this installation, and the geometry of this document needs it", "/regions", hint,
               codes::requires_mesher, {{"mesher", path.empty() ? json(nullptr) : json(path)}, {"reason", reason}});
}
// im.fluids_mesh() in a build that has no fluids module.
inline Error requires_fluids() {
  return coded("the fluids engines are not in this build of axionMultiPhysics (they need an MPI build)", "",
               "A wheel from PyPI and the sequential source build have no fluids module; install the MPI source build "
               "(docs/guides/install.md, \"MPI source build\").",
               codes::requires_fluids, json());
}
// A mesher run that failed: `why`, its error; `log`, its log; `status`, its exit status; `tail`, the log's last lines.
inline Error mesh_generation_failed(const std::string& why, const std::string& log, int status, const std::string& tail) {
  std::string hint = "The mesher's log is '" + log + "' (exit status " + std::to_string(status) + ")";
  hint += tail.empty() ? "; it is empty." : "; its last lines (the directory may be removed with this error):\n" + tail;
  return coded("mesh generation failed: " + why, "/regions", hint, codes::mesh_generation_failed,
               {{"reason", why}, {"log", log}, {"exit_status", status}});
}
// A contour with an elliptical segment: `pointer`, the segment; `segment`, its index in the contour.
inline Error shape_not_offered(const std::string& region, const std::string& pointer, size_t segment) {
  return coded("region '" + region + "': a contour with an elliptical segment cannot be meshed in a model document", pointer,
               "The elliptical segment of a contour becomes a spline curve, which the mesher of a model document does not take in "
               "this version. Build the region from \"rect\", \"polygon\" and \"disk\" (a circle) with \"union\" and "
               "\"difference\", or use lines only.",
               codes::shape_not_offered, {{"region", region}, {"segment", segment}});
}
// A model document that the bounded meshes do not take in this version (route.not_implemented): `operation`, mesh | boundaries.
inline Error mesh_3d_not_implemented(const std::string& operation) {
  return coded("'" + operation + "' of a 3D model document: bounded 3D meshes are not implemented yet", "/symmetry",
               "Planar and axisymmetric model documents are meshed in this version; `axmp validate` reports what a 3D document needs.",
               codes::route_not_implemented, json{{"operation", operation}, {"implemented", false}, {"symmetry", "3d"}});
}
inline Error mesh_no_geometry_not_implemented() {
  return coded("'axmp.mesh' of a model document with no geometry: there is no mesh to return", "/regions",
               "The document has no regions and no imported mesh, so there is nothing to mesh; `axmp mesh` prints a note for such a "
               "document. `axmp validate` reports what the document needs.",
               codes::route_not_implemented, json{{"operation", "mesh"}, {"implemented", false}, {"geometry", false}});
}
inline Error mesh_imported_not_implemented(const std::string& operation, const std::string& file) {
  return coded("'" + operation + "' of a model document on the imported mesh '" + file + "' is not implemented yet", "/mesh/file",
               "This version meshes the regions of a model document; reading an imported mesh as a meshed model comes later. "
               "`axmp validate` reports what the document needs.",
               codes::route_not_implemented, json{{"operation", operation}, {"implemented", false}, {"mesh_file", file}});
}
// The refusals of an electromagnetic design (no code, as `explain` refuses one).
inline Error boundaries_needs_model_document() {
  return Error("boundaries lists the named boundaries of a model document (a design with a \"physics\" section); this is an "
               "electromagnetic design",
               "", "`axmp resolve` prints the resolved geometry of an electromagnetic design.");
}
inline Error mesh_needs_model_document() {
  return Error("axmp.mesh meshes a model document (a design with a \"physics\" section); this is an electromagnetic design", "",
               "`axmp mesh` meshes an electromagnetic design from the command line.");
}

// ---------------------------------------------------------------------------------------------------- arc paths
// `where` is the pointer of the arc's angles.  An angle pair that looks like degrees (a span of at most 360) is converted
// in the hint.
inline Error arc_span(double a0, double a1, const std::string& where) {
  char buf[256];
  std::string what, hint = "The angles of an arc are in radians, and an arc is at most one turn (2 pi).";
  if (!std::isfinite(a0) || !std::isfinite(a1)) {
    what = "the angles of an arc must be finite numbers";
  } else {
    std::snprintf(buf, sizeof buf, "the angles of an arc span %.6g radians (from %.6g to %.6g), which is more than one turn (2 pi)",
                  std::fabs(a1 - a0), a0, a1);
    what = buf;
    if (std::fabs(a1 - a0) <= 360.0) {
      std::snprintf(buf, sizeof buf, " If %.6g and %.6g are degrees, write them in radians: [%.17g, %.17g].", a0, a1,
                    a0 * M_PI / 180.0, a1 * M_PI / 180.0);
      hint += buf;
    }
  }
  return coded(what, where, hint, codes::arc_span);
}
inline Error arc_radius(double radius, const std::string& where) {
  char buf[120];
  std::snprintf(buf, sizeof buf, "the radius of an arc must be positive, not %.6g", radius);
  return coded(buf, where, "Give the radius of the arc in metres, greater than 0.", codes::arc_degenerate);
}
inline Error arc_length(double length, const std::string& where) {
  char buf[200];
  std::snprintf(buf, sizeof buf, "the arc is %.6g m long (radius times the span of its angles), not longer than 1e-7 m", length);
  return coded(buf, where,
               "The mesher cannot build an arc of 1e-7 m or less (the tolerance of the CAD kernel). Give an arc longer than "
               "1e-7 m, with the angles in radians, or remove it.",
               codes::arc_degenerate);
}
// `where` is the pointer of the arc itself.
inline Error arc_negative_r(double min_r, const std::string& where) {
  char buf[200];
  std::snprintf(buf, sizeof buf, "the arc reaches r = %.6g m, below the symmetry axis (r >= 0)", min_r);
  return coded(buf, where,
               "Axisymmetric geometry lives in the half-plane r >= 0. Split the arc where it meets the axis, keeping the part at "
               "r >= 0, or move the centre or reduce the radius so that the arc stays off the axis.",
               codes::arc_negative_r);
}

}  // namespace msg
}  // namespace axmp::model
