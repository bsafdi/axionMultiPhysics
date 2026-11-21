// The physics catalogue: see cpp/include/axionmultiphysics/physics_registry.hpp.
//
// The tables below are a reading of the engines of the fluids and chemistry libraries (their conditions, options,
// refusals and conventions), with one cell class per row: on the planar and axisymmetric current-distribution rows a
// geometry order above the field order is an engine refusal, elsewhere a rule of this layer.  They are not trusted:
// the matrix test (cpp/tests/test_model_matrix.cpp and the engine half for the fluids library) checks every cell
// against the engines, and a cell is corrected from that test, never from memory.
#include "axionmultiphysics/physics_registry.hpp"

#include <algorithm>
#include <mutex>
#include <set>
#include <utility>

#include "axionmultiphysics/core.hpp"
#include "model_errors.hpp"

namespace axmp::model {

using json = nlohmann::json;
using Strs = std::vector<std::string>;

const char* to_string(CellValue c) {
  switch (c) {
    case CellValue::offered: return "offered";
    case CellValue::engine_refuses: return "engine_refuses";
    case CellValue::engine_cannot_express: return "engine_cannot_express";
    case CellValue::not_exposed: return "not_exposed";
    case CellValue::layer_rule: return "layer_rule";
  }
  return "";
}

// ------------------------------------------------------------------------------------------------ route factories
namespace {
struct Factories {
  std::mutex m;
  std::map<std::string, RouteFactory> f;
};
Factories& factories() {
  static Factories f;
  return f;
}
}  // namespace

void register_route_factory(const std::string& route_name, RouteFactory make) {
  Factories& F = factories();
  std::lock_guard<std::mutex> lock(F.m);
  F.f[route_name] = make;
}

RouteFactory route_factory(const std::string& route_name) {
  Factories& F = factories();
  std::lock_guard<std::mutex> lock(F.m);
  auto it = F.f.find(route_name);
  return it == F.f.end() ? nullptr : it->second;
}

// ------------------------------------------------------------------------------------------------ lookups
const PhysicsType* Registry::type(const std::string& name) const {
  for (const auto& t : types)
    if (t.name == name) return &t;
  return nullptr;
}
const CouplingTermType* Registry::term(const std::string& name) const {
  for (const auto& t : terms)
    if (t.name == name) return &t;
  return nullptr;
}
const Route* Registry::route(const std::string& name) const {
  for (const auto& r : routes)
    if (r.name == name) return &r;
  return nullptr;
}

namespace {

// ------------------------------------------------------------------------------------------------ builders
// A data key, required unless marked optional: key("velocity_m_s", "vector", "m/s").expr().optional().
struct KeyBuilder {
  DataKey d;
  KeyBuilder& expr() { d.field_expression = true; return *this; }
  KeyBuilder& optional() { d.required = false; return *this; }
  KeyBuilder& rampable() { d.rampable = true; return *this; }
  KeyBuilder& twin(const char* planar_spelling) { d.per_depth_twin = planar_spelling; return *this; }
  operator DataKey() const { return d; }
};
KeyBuilder key(const char* name, const char* type, const char* unit) {
  KeyBuilder k;
  k.d.name = name;
  k.d.type = type;
  k.d.unit = unit;
  return k;
}
ConditionKind kind(const char* name, std::vector<DataKey> data = {}, Strs aliases = {}) {
  ConditionKind k;
  k.name = name;
  k.data = std::move(data);
  k.aliases = std::move(aliases);
  return k;
}
OptionSpec option(const char* name, const char* type, Strs values, const char* dflt,
                  std::map<std::string, int> max_geometry_order = {}) {
  OptionSpec o;
  o.name = name;
  o.type = type;
  o.values = std::move(values);
  o.default_value = dflt;
  o.max_geometry_order = std::move(max_geometry_order);
  return o;
}
OptionRule offered(const char* name, Strs values, const char* dflt) {
  OptionRule r;
  r.name = name;
  r.offered = std::move(values);
  r.default_value = dflt;
  return r;
}
OptionRule forced(const char* name, const char* value) {
  OptionRule r;
  r.name = name;
  r.forced = value;
  r.default_value = value;
  return r;
}
SolverSetting setting(const char* physics, const char* name, const char* type, json dflt) {
  SolverSetting s;
  s.name = name;
  s.type = type;
  s.default_value = std::move(dflt);
  s.physics = physics;
  return s;
}
Quantity q(const char* name, const char* unit, const char* convention = "") { return Quantity{name, unit, convention}; }

constexpr const char* F = "fluid.incompressible";
constexpr const char* H = "heat.transfer";
constexpr const char* CD = "electrochem.current_distribution";
constexpr const char* CR = "chem.reactor";

// ------------------------------------------------------------------------------------------------ types
PhysicsType fluid_type() {
  PhysicsType t;
  t.name = F;
  t.title = "Incompressible flow (Stokes and Navier-Stokes)";
  t.status = "beta";
  t.aliases = {"laminar_flow", "creeping_flow"};
  t.provides = {"u", "p"};
  const char* tr = "traction by the fluid on the wall along t = n rotated +90 deg";
  t.quantities["planar"] = {q("u", "m/s", "velocity (u_x, u_y)"), q("u_x", "m/s"), q("u_y", "m/s"),
                            q("u_mag", "m/s"), q("p", "Pa"), q("vorticity", "1/s", "du_y/dx - du_x/dy"),
                            q("wall_shear", "Pa", tr), q("wall_shear_mag", "Pa"),
                            q("volume", "m^2/s", "flux only: per metre of depth, positive out of the domain"),
                            q("mass", "kg/(m s)", "flux only: per metre of depth, positive out of the domain")};
  t.quantities["axisymmetric"] = {q("u", "m/s", "velocity (u_r, u_z, u_theta)"), q("u_r", "m/s"), q("u_z", "m/s"),
                                  q("u_theta", "m/s"), q("u_mag", "m/s"), q("p", "Pa"),
                                  q("vorticity", "1/s", "azimuthal component du_r/dz - du_z/dr"),
                                  q("wall_shear", "Pa", tr), q("wall_shear_mag", "Pa", "includes the swirl part"),
                                  q("wall_shear_theta", "Pa", "the swirl part of the wall traction"),
                                  q("volume", "m^3/s", "flux only: 3D total, positive out of the domain"),
                                  q("mass", "kg/s", "flux only: 3D total, positive out of the domain")};
  t.quantities["3d"] = {q("u", "m/s", "velocity (u_x, u_y, u_z)"), q("u_x", "m/s"), q("u_y", "m/s"), q("u_z", "m/s"),
                        q("u_mag", "m/s"), q("p", "Pa"), q("vorticity", "1/s", "magnitude of curl u"),
                        q("wall_shear_mag", "Pa", "magnitude only (no tangent frame in 3D)"),
                        q("volume", "m^3/s", "flux only: positive out of the domain"),
                        q("mass", "kg/s", "flux only: positive out of the domain")};
  t.condition_kinds = {
      kind("wall", {key("velocity_m_s", "vector", "m/s").expr().optional()}, {"no_slip"}),
      kind("inlet", {key("velocity_m_s", "vector", "m/s").expr()}),
      kind("outlet", {key("pressure_Pa", "number", "Pa").optional()}),
      kind("open", {}, {"open_boundary"}),
      kind("traction", {key("traction_Pa", "vector", "Pa").expr()}, {"boundary_stress"}),
      kind("slip"),
      kind("symmetry"),
      kind("periodic",
           {key("partner", "boundary", ""), key("translation_m", "number_vector", "m"),
            key("pressure_drop_Pa", "number", "Pa").optional(),
            key("flow_rate", "number", "m^2/s planar (per metre of depth), m^3/s in 3D").optional()},
           {"periodic_flow_condition"}),
  };
  t.source_kinds = {kind("body_force", {key("force_N_m3", "vector", "N/m^3").expr()}, {"volume_force"})};
  t.options = {option("equation", "enum", {"stokes", "navier_stokes"}, ""),
               option("viscous_form", "enum", {"stress", "laplacian"}, "stress"),
               option("convection", "enum", {"standard", "skew"}, "standard"),
               option("order", "integer", {"2"}, "2")};
  t.not_implemented = {
      "solving through the model document: every route of this type is not implemented in this version",
      "periodic conditions through the document (they need a periodic instance mesh; roadmap item 'periodic flow')",
      "slip and symmetry conditions with Navier-Stokes, and in axisymmetric geometry",
      "the swirl-free axisymmetric formulation: an axisymmetric flow has the three components (u_r, u_z, u_theta)",
      "inflow by flow rate, backflow stabilization and the directional do-nothing outlet (roadmap item F2b)",
      "inductionless MHD, the equal-order (GLS) discretization and turbulence models as options",
      "a force on a moving wall with the Laplacian viscous form on the Navier-Stokes rows"};
  t.docs = "docs/guides/fluids.md";
  t.validation = "docs/validation/fluids.md";
  return t;
}

PhysicsType heat_type() {
  PhysicsType t;
  t.name = H;
  t.title = "Heat transfer in solids and fluids";
  t.status = "beta";
  t.aliases = {"heat_transfer_in_solids", "heat_transfer_in_fluids"};
  t.provides = {"T"};
  for (const char* g : {"planar", "axisymmetric", "3d"}) {
    const bool per_depth = std::string(g) == "planar";
    t.quantities[g] = {
        q("T", "K"), q("heat_flux_n", "W/m^2", "conductive flux -k grad T . n, positive out of the domain"),
        q("heat", per_depth ? "W/m" : "W",
          per_depth ? "flux only: conductive plus advected, per metre of depth, positive out of the domain"
                    : "flux only: conductive plus advected, 3D total, positive out of the domain"),
        q("heat_conductive", per_depth ? "W/m" : "W",
          per_depth ? "flux only: conductive part, per metre of depth, positive out of the domain"
                    : "flux only: conductive part, 3D total, positive out of the domain")};
  }
  t.condition_kinds = {
      kind("temperature", {key("T_K", "number", "K").expr()}),
      kind("heat_flux", {key("q_W_m2", "number", "W/m^2")}),
      kind("total_flux", {key("q_W_m2", "number", "W/m^2")}),
      kind("convection", {key("h_W_m2K", "number", "W/(m^2 K)"), key("T_ext_K", "number", "K")},
           {"convective_heat_flux"}),
      kind("insulation", {}, {"thermal_insulation"}),
      kind("outflow"),
  };
  t.source_kinds = {
      kind("heat_source", {key("q_W_m3", "number", "W/m^3").expr().optional(),
                           key("P_W", "number", "W").optional().twin("P_W_m")}),
      kind("prescribed_advection", {key("velocity_m_s", "meridional_vector", "m/s").expr()},
           {"velocity_field"}),
  };
  t.options = {option("order", "integer", {"1", "2"}, "2"),
               option("stabilization", "enum", {"none", "supg"}, "none", {{"supg", 1}}),
               option("advective_form", "enum", {"convective", "conservative_flux", "conservative_divergence"},
                      "convective")};
  t.not_implemented = {
      "solving through the model document: every route of this type is not implemented in this version",
      "a transient heat solve",
      "heat transfer in 3D",
      "a conductivity or a heat source that differs from region to region (one expression over the instance until "
      "the transport operator takes data per region)",
      "a heat capacity rho cp that differs between the regions of an advected instance (conjugate heat transfer)",
      "SUPG on curved (isoparametric) elements",
      "quadrilateral elements",
      "the heat-to-heat interface term and thermal contact"};
  t.docs = "docs/guides/fluids.md";
  t.validation = "docs/validation/fluids.md";
  return t;
}

PhysicsType current_distribution_type() {
  PhysicsType t;
  t.name = CD;
  t.title = "Primary and secondary current distribution";
  t.status = "beta";
  t.aliases = {"primary_current_distribution", "secondary_current_distribution"};
  t.provides = {"phi"};
  for (const char* g : {"planar", "axisymmetric", "3d"})
    t.quantities[g] = {q("phi_l", "V", "electrolyte potential"),
                       q("i_mag", "A/m^2", "magnitude of the current density"),
                       q("i_n", "A/m^2", "normal current density, positive out of the electrolyte"),
                       q("overpotential", "V", "eta = phi_s - phi_l - E_eq, anodic positive")};
  t.condition_kinds = {
      kind("electrode",
           {key("reactions", "names", "").optional(),
            key("phi_s_V", "number", "V").optional().rampable(),
            key("current_A", "number", "A").optional().rampable().twin("current_A_m"),
            key("primary", "boolean", "").optional(), key("E_eq_V", "number", "V").optional()},
           {"electrode_surface"}),
      kind("insulation"),
      kind("normal_current", {key("i_A_m2", "number", "A/m^2")}, {"normal_current_density"}),
  };
  t.options = {option("kinetics", "enum", {"primary", "secondary"}, "secondary"),
               option("order", "integer", {"1", "2"}, "2"), option("T_K", "number", {}, "298.15")};
  t.not_implemented = {
      "solving through the model document: every route of this type is not implemented in this version",
      "every electrode galvanostatic: the potential level is then free",
      "P1 elements on curved (isoparametric) geometry",
      "concentration-dependent kinetics and species transport in the electrolyte"};
  t.docs = "docs/guides/chemistry.md";
  t.validation = "docs/validation/chemistry.md";
  return t;
}

PhysicsType reactor_type() {
  PhysicsType t;
  t.name = CR;
  t.title = "Reaction engineering (0D reactors)";
  t.status = "beta";
  t.aliases = {"reaction_engineering"};
  t.provides = {"c_<species>"};
  t.not_implemented = {
      "solving through the model document: the route of this type is not implemented in this version",
      "the reactor's own data (volume, feed, initial composition) in the document grammar",
      "a stationary solve: a reactor is solved in time only"};
  t.docs = "docs/guides/chemistry.md";
  t.validation = "docs/validation/chemistry.md";
  return t;
}

PhysicsType porous_electrode_type() {
  PhysicsType t;
  t.name = "electrochem.porous_electrode";
  t.title = "Porous electrode (1D)";
  t.status = "beta";
  t.not_implemented = {"everything but the name: the type has no condition kinds, options or route yet"};
  t.docs = "docs/guides/chemistry.md";
  t.validation = "docs/validation/chemistry.md";
  return t;
}

// Absent types: known to the catalogue, so that a document naming them parses and is reported unavailable with the
// "not implemented" inventory, and with no route row.
PhysicsType conduction_type() {
  PhysicsType t;
  t.name = "current.conduction";
  t.title = "Electric currents (conduction)";
  t.status = "absent";
  t.aliases = {"electric_currents"};
  t.provides = {"V"};
  for (const char* g : {"planar", "axisymmetric", "3d"}) {
    const bool per_depth = std::string(g) == "planar";
    t.quantities[g] = {q("V", "V", "electric potential"), q("J_mag", "A/m^2", "magnitude of the current density"),
                       q("joule_power_density", "W/m^3"),
                       q("charge", per_depth ? "A/m" : "A",
                         per_depth ? "flux only: current per metre of depth, positive out of the domain"
                                   : "flux only: current, positive out of the domain")};
  }
  t.condition_kinds = {kind("potential", {key("V_V", "number", "V").expr()}, {"electric_potential"}),
                       kind("ground"), kind("insulation", {}, {"electric_insulation"})};
  t.not_implemented = {"the type: it has no engine in this version"};
  t.absent_phrase = "physics type 'current.conduction'";
  return t;
}

PhysicsType solid_type() {
  PhysicsType t;
  t.name = "solid.linear";
  t.title = "Linear elastic solid";
  t.status = "absent";
  t.aliases = {"solid_mechanics"};
  t.provides = {"displacement"};
  for (const char* g : {"planar", "axisymmetric", "3d"})
    t.quantities[g] = {q("displacement_mag", "m"), q("von_mises", "Pa", "von Mises equivalent stress")};
  t.condition_kinds = {kind("fixed", {}, {"fixed_constraint"}), kind("free")};
  t.not_implemented = {"the type as a physics instance: elasticity runs only behind the electromagnetic 'mechanics' "
                       "section in this version"};
  t.absent_phrase = "'solid.linear' as a physics instance";
  return t;
}

PhysicsType species_type() {
  PhysicsType t;
  t.name = "species.transport";
  t.title = "Transport of diluted species";
  t.status = "absent";
  t.not_implemented = {"the type: it has no engine in this version"};
  t.absent_phrase = "physics type 'species.transport'";
  return t;
}

// ------------------------------------------------------------------------------------------------ terms
std::vector<CouplingTermType> terms() {
  CouplingTermType adv, buo, joule, exp;
  adv.name = "advection";
  adv.title = "Advection of a scalar by a solved velocity";
  adv.status = "beta";
  adv.support = "domain";
  adv.producers = {"fluid.*"};
  adv.consumers = {H, "species.transport"};
  adv.reads = {"u"};
  buo.name = "buoyancy";
  buo.title = "Boussinesq buoyancy of a temperature field";
  buo.status = "beta";
  buo.support = "domain";
  buo.producers = {H};
  buo.consumers = {F};
  buo.reads = {"T"};
  buo.data = {key("T_ref_K", "number", "K"), key("gravity_m_s2", "meridional_vector", "m/s^2")};
  joule.name = "joule_heating";
  joule.title = "Joule heating by an electric current";
  joule.status = "absent";
  joule.support = "domain";
  joule.producers = {"current.conduction"};
  joule.consumers = {H};
  joule.reads = {"V"};
  exp.name = "thermal_expansion";
  exp.title = "Thermal expansion of a solid";
  exp.status = "absent";
  exp.support = "domain";
  exp.producers = {H};
  exp.consumers = {"solid.linear"};
  exp.reads = {"T"};
  exp.data = {key("T_ref_K", "number", "K")};
  return {adv, buo, joule, exp};
}

// ------------------------------------------------------------------------------------------------ layer rules
std::vector<LayerRule> layer_rules() {
  return {
      {"insulation_on_throughflow",
       "insulation is refused on a boundary that the advecting velocity crosses (heat.insulation_on_throughflow); the "
       "engine takes a zero flux anywhere"},
      {"outflow_with_conservative_flux",
       "outflow is refused with the conservative_flux form, where a zero total flux would block the advected heat (on a "
       "row without advection too, so that the name means the same on every heat row); the engine takes it as a zero "
       "flux and only the Python binding refuses the name, which this layer mirrors"},
      {"force_with_laplacian_form",
       "with the Laplacian viscous form a force output is accepted on walls at rest only "
       "(output.force_needs_stress_form), and needs one viscosity over the instance"},
      {"per_depth_keys",
       "an absolute input is written per metre of depth in a planar model (current_A_m, P_W_m) and as the 3D total "
       "otherwise (current_A, P_W) (model.per_depth_key)"},
      {"geometry_order_exceeds_field_order",
       "mesh.geometry_order may not exceed the smallest field order of the instances "
       "(model.geometry_order_exceeds_field_order)"},
      {"uniform_capacity_in_stagnant_part",
       "density and thermal.cp_J_kgK are one constant over an advected heat instance, in its stagnant parts as well, "
       "although a steady solve needs the capacity only where the velocity is not zero"},
  };
}

// ------------------------------------------------------------------------------------------------ routes
constexpr const char* kPeriodicReason =
    "the engine needs a topologically periodic mesh, and the instance mesh has no boundary element on an interior "
    "face; exposing it takes a periodic instance mesh, the partition rule's exception for the partner and matching "
    "nodes on the two sides";

Strs flow_kinds_all() { return {"wall", "inlet", "outlet", "open", "traction", "slip", "symmetry"}; }
Strs flow_kinds_convective() { return {"wall", "inlet", "outlet", "open", "traction"}; }

// The settings of the steady Navier-Stokes Newton solve (NonlinearOptions), on the flow instance.
std::vector<SolverSetting> navier_stokes_solvers() {
  return {setting(F, "newton_rtol", "number", 1e-12), setting(F, "newton_maxit", "integer", 30),
          setting(F, "n_picard", "integer", 3), setting(F, "line_search", "boolean", true),
          setting(F, "residual_floor_factor", "number", 4.0)};
}

Route base_route(const std::string& name, const char* geometry, Strs physics, const char* study, const char* engine) {
  Route r;
  r.name = name;
  r.physics = physics;
  r.cell_types = std::move(physics);
  if (*geometry) r.geometries = {geometry};
  r.study_types = {study};
  r.engine = engine;
  r.status = "not_implemented";
  r.implemented = false;
  return r;
}

enum class FlowRow { stokes, navier_stokes, transient };

Route flow_row(FlowRow kindof, const char* geometry) {
  const std::string g = geometry;
  const bool axi = g == "axisymmetric", three = g == "3d";
  std::string name;
  const char* engine = "";
  const char* study = "stationary";
  if (kindof == FlowRow::stokes) {
    name = "stokes_" + g;
    engine = three ? "StokesSolver (3D, distributed)" : "StokesSolver";
  } else if (kindof == FlowRow::navier_stokes) {
    name = "navier_stokes_steady_" + g;
    engine = "NavierStokesSolver";
  } else {
    name = "navier_stokes_transient_" + g;
    engine = "TransientNavierStokes";
    study = "transient";
  }
  Route r = base_route(name, geometry, {F}, study, engine);
  r.needs.mpi = true;
  r.needs.fluids = true;
  r.needs.engine_3d = three;
  r.needs.max_ranks = 0;   // every flow engine solves on a partitioned mesh, the time stepper included
  const bool stokes = kindof == FlowRow::stokes;
  r.when = {{"equation", stokes ? "stokes" : "navier_stokes"}};
  if (stokes && !axi) {
    r.condition_kinds[F] = flow_kinds_all();
    r.not_offered[F]["periodic"] = std::string("not_exposed: ") + kPeriodicReason;
    r.roadmap[F]["periodic"] = "periodic flow";
  } else {
    r.condition_kinds[F] = flow_kinds_convective();
    for (const char* k : {"slip", "symmetry", "periodic"}) r.not_offered[F][k] = "engine_refuses";
  }
  r.options[F].push_back(offered("equation", {stokes ? "stokes" : "navier_stokes"}, ""));
  r.options[F].push_back(offered("viscous_form", {"stress", "laplacian"}, "stress"));
  // the steady solver takes both convective forms; the time stepper takes the standard form only (its constructor
  // refuses the skew form)
  if (kindof == FlowRow::navier_stokes)
    r.options[F].push_back(offered("convection", {"standard", "skew"}, "standard"));
  if (kindof == FlowRow::transient) r.options[F].push_back(offered("convection", {"standard"}, "standard"));
  r.options[F].push_back(offered("order", {"2"}, "2"));
  r.material_blocks[F] = stokes ? Strs{"fluid"} : Strs{"fluid", "density"};
  if (!stokes) r.uniform[F] = {"density"};
  r.element_types = three ? Strs{"tetrahedron"} : Strs{"triangle", "quadrilateral"};
  r.axisymmetric_integrals = axi ? "includes_2pi" : "";
  r.force_reaction = stokes ? "physical_reaction" : "reaction: walls at rest only with the laplacian form";
  if (!stokes) r.layer_rules.push_back("force_with_laplacian_form");
  r.layer_rules.push_back("geometry_order_exceeds_field_order");
  if (kindof == FlowRow::navier_stokes) r.solvers = navier_stokes_solvers();
  if (kindof == FlowRow::transient)
    r.solvers = {setting(F, "newton_rtol", "number", 1e-12), setting(F, "newton_maxit", "integer", 30)};
  // a force output with the Laplacian form needs one viscosity: StokesSolver::physical_reaction refuses a variable
  // viscosity there; on the Navier-Stokes rows the requirement is kept as a rule of this layer
  ConditionalUniform mu;
  mu.property = "fluid.mu_Pa_s";
  mu.option = "viscous_form=laplacian";
  mu.output = "force";
  mu.why = stokes ? "engine_refuses: StokesSolver::physical_reaction" : "layer_rule: force_with_laplacian_form";
  r.uniform_when[F] = {mu};
  // limits
  if (stokes)
    r.limits.push_back("a force output with the Laplacian viscous form needs one viscosity over the instance (the "
                       "engine's physical reaction is defined for a constant viscosity only)");
  else
    r.limits.push_back("density is one constant over the instance: the engine takes one density per solve");
  if (!stokes)
    r.limits.push_back("a force output is the engine's reaction of the solved form: the physical force with the stress "
                       "form; with the Laplacian form the force on walls at rest only");
  if (stokes && !axi)
    r.limits.push_back("periodic conditions are not exposed yet (roadmap item 'periodic flow')");
  else if (stokes)
    r.limits.push_back("no slip, symmetry or periodic condition in axisymmetric geometry");
  else
    r.limits.push_back("no slip, symmetry or periodic condition with convection");
  if (axi)
    r.limits.push_back("an axisymmetric flow has the three velocity components (u_r, u_z, u_theta); forces and torques "
                       "are 3D totals (the engine includes 2 pi)");
  if (kindof == FlowRow::transient)
    r.limits.push_back("convection skew is refused: the time stepper takes the standard convective form only");
  if (kindof == FlowRow::transient)
    r.limits.push_back("any number of MPI ranks for the time steps; the point evaluation of a stored state needs one "
                       "rank");
  else if (!three)
    r.limits.push_back("any number of MPI ranks for the solve; outputs and point sampling on rank 0 from the saved "
                       "solution");
  if (three) r.limits.push_back("distributed over the MPI ranks; every cell of this row is confirmed against the "
                                "engine when the row is implemented");
  return r;
}

void heat_kinds_by_form(Route& r) {
  r.condition_kinds[H] = {"temperature", "heat_flux", "total_flux", "convection", "insulation", "outflow"};
  r.not_offered_with[H]["advective_form=convective"]["total_flux"] = "engine_refuses";
  r.not_offered_with[H]["advective_form=conservative_divergence"]["total_flux"] = "engine_refuses";
  r.not_offered_with[H]["advective_form=conservative_flux"]["heat_flux"] = "engine_refuses";
  r.not_offered_with[H]["advective_form=conservative_flux"]["outflow"] = "layer_rule: outflow_with_conservative_flux";
}
void heat_kinds_convective_only(Route& r) {
  r.condition_kinds[H] = {"temperature", "heat_flux", "convection", "insulation", "outflow"};
  r.not_offered[H]["total_flux"] = "engine_refuses";
}

enum class HeatRow { conduction, prescribed, one_way, boussinesq };

Route heat_row(HeatRow kindof, const char* geometry) {
  const std::string g = geometry;
  const bool axi = g == "axisymmetric";
  std::string name;
  Strs physics = {H}, couplings;
  const char* engine = "transport::ScalarTransport";
  switch (kindof) {
    case HeatRow::conduction: name = "heat_" + g; break;
    case HeatRow::prescribed:
      name = "heat_prescribed_advection_" + g;
      engine = "transport::ScalarTransport with a callable velocity";
      break;
    case HeatRow::one_way:
      name = "flow_then_heat_" + g;
      physics = {F, H};
      couplings = {"advection"};
      engine = "the flow row's engine (StokesSolver or NavierStokesSolver), then transport::ScalarTransport with the "
               "finite-element velocity";
      break;
    case HeatRow::boussinesq:
      name = "boussinesq_" + g;
      physics = {F, H};
      couplings = {"advection", "buoyancy"};
      engine = "BoussinesqSolver";
      break;
  }
  Route r = base_route(name, geometry, physics, "stationary", engine);
  r.cell_types = {H};
  r.couplings = couplings;
  r.needs.mpi = true;
  r.needs.fluids = true;
  r.needs.max_ranks = 1;
  const bool advected = kindof != HeatRow::conduction;
  // the transport operator takes the three advective forms with or without a velocity, and checks the flux kinds
  // against the form in add_bc either way; the Boussinesq solver has the convective form only
  const bool form_selectable = kindof != HeatRow::boussinesq;
  if (form_selectable) heat_kinds_by_form(r);
  else heat_kinds_convective_only(r);
  r.options[H].push_back(offered("order", kindof == HeatRow::boussinesq ? Strs{"2"} : Strs{"1", "2"}, "2"));
  r.options[H].push_back(offered("stabilization", advected ? Strs{"none", "supg"} : Strs{"none"}, "none"));
  if (form_selectable)
    r.options[H].push_back(
        offered("advective_form", {"convective", "conservative_divergence", "conservative_flux"}, "convective"));
  else
    r.options[H].push_back(forced("advective_form", "convective"));
  if (kindof == HeatRow::prescribed) r.when_sources[H] = {"prescribed_advection"};
  // blocks, uniform, one expression
  switch (kindof) {
    case HeatRow::conduction:
      r.material_blocks[H] = {"thermal"};
      r.one_expression[H] = {"thermal.k_W_mK", "heat_source"};
      break;
    case HeatRow::prescribed:
      r.material_blocks[H] = {"thermal", "thermal.cp_J_kgK", "density"};
      r.uniform[H] = {"density", "thermal.cp_J_kgK"};
      r.one_expression[H] = {"thermal.k_W_mK", "heat_source", "prescribed_advection.velocity"};
      break;
    case HeatRow::one_way:
      r.material_blocks[F] = {"fluid"};
      r.material_blocks[H] = {"thermal", "thermal.cp_J_kgK", "density"};
      r.uniform[H] = {"density", "thermal.cp_J_kgK"};
      r.one_expression[H] = {"thermal.k_W_mK", "heat_source"};
      break;
    case HeatRow::boussinesq:
      r.material_blocks[F] = {"fluid", "density", "fluid.beta_1_K"};
      r.material_blocks[H] = {"thermal", "thermal.cp_J_kgK"};
      r.uniform[F] = {"density", "fluid.beta_1_K"};
      r.uniform[H] = {"thermal.cp_J_kgK"};
      r.one_expression[H] = {"thermal.k_W_mK", "heat_source"};
      break;
  }
  if (kindof == HeatRow::one_way) {
    r.strategies = {"one_way"};
    r.default_strategy = "one_way";
    r.equal_domains = true;
    // the flow instance takes the settings of its own row (none for Stokes, the Newton settings for Navier-Stokes);
    // the transport operator's linear solve takes none
    r.solvers_of_own_row = {F};
  }
  if (kindof == HeatRow::boussinesq) {
    r.strategies = {"monolithic"};
    r.default_strategy = "monolithic";
    r.equal_domains = true;
    r.when = {{"equation", "navier_stokes"}};
    // one monolithic Newton solve, written on the flow instance (BoussinesqOptions::ns.nonlinear)
    r.solvers = {setting(F, "newton_rtol", "number", 1e-12), setting(F, "newton_maxit", "integer", 30),
                 setting(F, "line_search", "boolean", true)};
  }
  r.element_types = {"triangle"};
  r.axisymmetric_integrals = axi ? "per_radian" : "";
  if (advected) r.layer_rules.push_back("insulation_on_throughflow");
  if (form_selectable) r.layer_rules.push_back("outflow_with_conservative_flux");
  r.layer_rules.push_back("per_depth_keys");
  r.layer_rules.push_back("geometry_order_exceeds_field_order");
  if (advected) r.layer_rules.push_back("uniform_capacity_in_stagnant_part");
  // limits
  r.limits.push_back("one MPI rank");
  r.limits.push_back("triangles only (the transport operator's constructor refuses other elements)");
  if (kindof == HeatRow::boussinesq)
    r.limits.push_back("thermal.k_W_mK and the heat source are one expression over the instance (the engine's options "
                       "take one function of position)");
  else
    r.limits.push_back("thermal.k_W_mK and the heat source are one expression over the instance until the transport "
                       "operator takes data per region");
  switch (kindof) {
    case HeatRow::conduction:
      r.limits.push_back("stabilization supg is refused: there is no velocity for it");
      r.limits.push_back("without advection the three advective forms are one operator, and the condition kinds "
                         "still depend on advective_form (the transport operator checks them against the form)");
      break;
    case HeatRow::prescribed:
      r.limits.push_back("one prescribed_advection source covering the whole instance (one velocity per solve), and no "
                         "flow instance");
      r.limits.push_back("density and thermal.cp_J_kgK are one constant over the instance (one advective capacity per "
                         "solve)");
      r.limits.push_back("stabilization supg needs straight elements (geometry order 1)");
      r.limits.push_back("the condition kinds depend on advective_form");
      break;
    case HeatRow::one_way:
      r.limits.push_back("the flow instance is checked against the Stokes or steady Navier-Stokes row of the same "
                         "geometry that its equation selects, and solved first; its solver settings are those of "
                         "that row");
      r.limits.push_back("the flow and heat instances have the same domain (one mesh, no interpolation)");
      r.limits.push_back("density and thermal.cp_J_kgK are one constant over the instance, on a Stokes flow too");
      r.limits.push_back("stabilization supg needs straight elements, for the flow as well (one master mesh)");
      r.limits.push_back("triangles only for the flow as well (one master mesh)");
      r.limits.push_back("the condition kinds depend on advective_form");
      r.limits.push_back("no force output on this row");
      break;
    case HeatRow::boussinesq:
      r.limits.push_back("steady only; one monolithic Newton solve of velocity, pressure and temperature");
      r.limits.push_back("the flow instance is checked against the steady Navier-Stokes row of the same geometry");
      r.limits.push_back("gravity along the vertical axis only (y planar, z axisymmetric); T_ref_K is required");
      r.limits.push_back("advective_form is forced to convective (the engine's only energy form)");
      r.limits.push_back("the temperature is P2: heat order 1 is refused");
      r.limits.push_back("density, thermal.cp_J_kgK and fluid.beta_1_K are one constant over the instance");
      r.limits.push_back("the flow and heat instances have the same domain");
      r.limits.push_back("stabilization supg needs straight elements");
      r.limits.push_back("no force output on this row (the engine exposes no reaction)");
      break;
  }
  if (axi)
    r.limits.push_back("the engine's integrals are per radian; the results are 3D totals (times 2 pi)");
  return r;
}

Route current_distribution_row(const char* geometry) {
  const std::string g = geometry;
  const bool three = g == "3d", axi = g == "axisymmetric";
  Route r = base_route("current_distribution_" + g, geometry, {CD}, "stationary",
                       three ? "chem::CurrentDistribution3D" : "chem::CurrentDistribution");
  r.needs.mpi = three;
  r.needs.engine_3d = three;
  r.needs.max_ranks = three ? 0 : 1;
  r.condition_kinds[CD] = {"electrode", "insulation", "normal_current"};
  r.options[CD] = {offered("kinetics", {"primary", "secondary"}, "secondary"), offered("order", {"1", "2"}, "2"),
                   offered("T_K", {}, "298.15")};
  r.material_blocks[CD] = {"electrolyte"};
  r.element_types = three ? Strs{"tetrahedron"} : Strs{"triangle"};
  r.axisymmetric_integrals = axi ? "per_radian" : "";
  r.rampable = {"phi_s_V", g == "planar" ? "current_A_m" : "current_A"};
  r.layer_rules.push_back("per_depth_keys");
  if (three) r.layer_rules.push_back("geometry_order_exceeds_field_order");
  r.solvers = {setting(CD, "newton_rtol", "number", 1e-12), setting(CD, "newton_maxit", "integer", 60)};
  r.limits.push_back("at least one electrode with a given potential: with every electrode galvanostatic the potential "
                     "level is free (electrochem.potential_level_free)");
  if (three) {
    r.limits.push_back("distributed over the MPI ranks; every cell of this row is confirmed against the engine when "
                       "the row is implemented");
  } else {
    r.limits.push_back("one process");
    r.limits.push_back("order 1 on straight elements only: the engine refuses curved edge nodes with P1");
  }
  if (g == "planar")
    r.limits.push_back("a galvanostatic current is per metre of depth (current_A_m)");
  else
    r.limits.push_back("a galvanostatic current is the 3D total (current_A)");
  if (axi) r.limits.push_back("the engine's integrals are per radian; the results are 3D totals (times 2 pi)");
  r.limits.push_back("rampable: one electrode's phi_s_V or current, through a warm-started continuation");
  return r;
}

Route reactor_row() {
  Route r = base_route("chem_reactor", "", {CR}, "transient",
                       "chem reactor problems (batch, CSTR, semibatch, constant-pressure gas, adiabatic liquid, plug "
                       "flow) with the time core");
  r.needs.mesher = false;
  // max_ranks stays 0, any number of ranks: the reactor problems and the time core make no MPI call and test no rank
  // count, so nothing refuses an MPI launch, and every rank would integrate the same system to the same numbers.  The
  // solve is one process's work; running it once under such a launch is the route adapter's to arrange.
  r.limits.push_back("no geometry and no mesher");
  r.limits.push_back("no stationary route: the default study is refused with the transient step to write "
                     "(study.type_not_offered)");
  return r;
}

Registry build() {
  Registry R;
  R.types = {fluid_type(),          heat_type(),       current_distribution_type(), reactor_type(),
             porous_electrode_type(), conduction_type(), solid_type(),                species_type()};
  for (auto& t : R.types) t.family = t.name.substr(0, t.name.find('.'));
  R.terms = terms();
  R.layer_rules = layer_rules();
  R.reserved_ids = {"em", "mechanics", "thermal", "electrostatics", "hts", "fullwave", "cavity", "mesh_motion"};
  for (const char* g : {"planar", "axisymmetric"}) R.routes.push_back(flow_row(FlowRow::stokes, g));
  for (const char* g : {"planar", "axisymmetric"}) R.routes.push_back(flow_row(FlowRow::navier_stokes, g));
  for (const char* g : {"planar", "axisymmetric"}) R.routes.push_back(flow_row(FlowRow::transient, g));
  R.routes.push_back(flow_row(FlowRow::stokes, "3d"));
  for (HeatRow h : {HeatRow::conduction, HeatRow::prescribed, HeatRow::one_way, HeatRow::boussinesq})
    for (const char* g : {"planar", "axisymmetric"}) R.routes.push_back(heat_row(h, g));
  for (const char* g : {"planar", "axisymmetric", "3d"}) R.routes.push_back(current_distribution_row(g));
  R.routes.push_back(reactor_row());
  return R;
}

// ------------------------------------------------------------------------------------------------ JSON
json key_json(const DataKey& d) {
  return {{"name", d.name},
          {"type", d.type},
          {"unit", d.unit},
          {"field_expression", d.field_expression},
          {"rampable", d.rampable},
          {"required", d.required},
          {"per_depth_twin", d.per_depth_twin}};
}
json keys_json(const std::vector<DataKey>& v) {
  json a = json::array();
  for (const auto& d : v) a.push_back(key_json(d));
  return a;
}
json kinds_json(const std::vector<ConditionKind>& v) {
  json a = json::array();
  for (const auto& k : v) a.push_back({{"name", k.name}, {"aliases", k.aliases}, {"data", keys_json(k.data)}});
  return a;
}
json type_json(const PhysicsType& t) {
  json quantities = json::object();
  for (const auto& [g, qs] : t.quantities) {
    json a = json::array();
    for (const auto& x : qs) a.push_back({{"name", x.name}, {"unit", x.unit}, {"convention", x.convention}});
    quantities[g] = a;
  }
  json options = json::array();
  for (const auto& o : t.options) {
    json mgo = json::object();
    for (const auto& [v, n] : o.max_geometry_order) mgo[v] = n;
    options.push_back({{"name", o.name},
                       {"type", o.type},
                       {"values", o.values},
                       {"default", o.default_value},
                       {"max_geometry_order", mgo}});
  }
  return {{"name", t.name},
          {"family", t.family},
          {"title", t.title},
          {"status", t.status},
          {"aliases", t.aliases},
          {"provides", t.provides},
          {"quantities", quantities},
          {"condition_kinds", kinds_json(t.condition_kinds)},
          {"source_kinds", kinds_json(t.source_kinds)},
          {"options", options},
          {"not_implemented", t.not_implemented},
          {"absent_phrase", t.absent_phrase},
          {"docs", t.docs},
          {"validation", t.validation}};
}
json term_json(const CouplingTermType& t) {
  return {{"name", t.name},           {"title", t.title},         {"status", t.status},
          {"support", t.support},     {"producers", t.producers}, {"consumers", t.consumers},
          {"reads", t.reads},         {"data", keys_json(t.data)}};
}
json route_json(const Route& r) {
  json options = json::object();
  for (const auto& [t, rules] : r.options) {
    json a = json::array();
    for (const auto& x : rules)
      a.push_back({{"name", x.name}, {"offered", x.offered}, {"forced", x.forced}, {"default", x.default_value}});
    options[t] = a;
  }
  json solvers = json::array();
  for (const auto& s : r.solvers)
    solvers.push_back({{"name", s.name}, {"type", s.type}, {"default", s.default_value}, {"physics", s.physics}});
  json uniform_when = json::object();
  for (const auto& [t, conds] : r.uniform_when) {
    json a = json::array();
    for (const auto& c : conds)
      a.push_back({{"property", c.property}, {"option", c.option}, {"output", c.output}, {"why", c.why}});
    uniform_when[t] = a;
  }
  auto obj = [](const auto& m) { return m.empty() ? json::object() : json(m); };
  return {{"name", r.name},
          {"physics", r.physics},
          {"cell_types", r.cell_types},
          {"couplings", r.couplings},
          {"geometries", r.geometries},
          {"study_types", r.study_types},
          {"strategies", r.strategies},
          {"when", obj(r.when)},
          {"when_sources", obj(r.when_sources)},
          {"engine", r.engine},
          {"status", r.status},
          {"implemented", r.implemented},
          {"default_strategy", r.default_strategy},
          {"requires",
           {{"mesher", r.needs.mesher},
            {"mpi", r.needs.mpi},
            {"fluids", r.needs.fluids},
            {"engine_3d", r.needs.engine_3d},
            {"max_ranks", r.needs.max_ranks}}},
          {"condition_kinds", obj(r.condition_kinds)},
          {"not_offered", obj(r.not_offered)},
          {"not_offered_with", obj(r.not_offered_with)},
          {"roadmap", obj(r.roadmap)},
          {"axisymmetric_integrals", r.axisymmetric_integrals},
          {"force_reaction", r.force_reaction},
          {"options", options},
          {"material_blocks", obj(r.material_blocks)},
          {"uniform", obj(r.uniform)},
          {"uniform_when", uniform_when},
          {"one_expression", obj(r.one_expression)},
          {"element_types", r.element_types},
          {"equal_domains", r.equal_domains},
          {"rampable", r.rampable},
          {"limits", r.limits},
          {"layer_rules", r.layer_rules},
          {"solvers", solvers},
          {"solvers_of_own_row", r.solvers_of_own_row}};
}

// "fluid.*" matches every type of the family "fluid"; any other pattern matches its own name.
bool matches(const std::string& pattern, const std::string& type) {
  if (pattern.size() >= 2 && pattern.compare(pattern.size() - 2, 2, ".*") == 0)
    return type.compare(0, pattern.size() - 1, pattern, 0, pattern.size() - 1) == 0;
  return pattern == type;
}

}  // namespace

const Registry& registry() {
  static const Registry R = build();
  return R;
}

json catalogue_json(bool routes, const std::string& type) {
  const Registry& R = registry();
  if (!type.empty() && R.type(type) == nullptr) {
    std::string names;
    for (const auto& t : R.types) names += (names.empty() ? "" : ", ") + t.name;
    std::string alias_of;
    for (const auto& t : R.types)
      if (std::find(t.aliases.begin(), t.aliases.end(), type) != t.aliases.end()) alias_of = t.name;
    const std::string hint = alias_of.empty()
                                 ? "The physics types are: " + names + "."
                                 : "'" + type + "' is another code's name for '" + alias_of +
                                       "'; write '" + alias_of + "'. The physics types are: " + names + ".";
    json details = {{"type", type}, {"known", json::array()}};
    for (const auto& t : R.types) details["known"].push_back(t.name);
    if (!alias_of.empty()) details["alias_of"] = alias_of;
    throw Error("unknown physics type '" + type + "'", "", hint, codes::physics_unknown_type, details.dump());
  }
  json out = {{"schema", "axionmultiphysics.physics_catalogue/1"}, {"status", "beta"}};
  json types = json::array(), terms_out = json::array(), routes_out = json::array();
  for (const auto& t : R.types)
    if (type.empty() || t.name == type) types.push_back(type_json(t));
  for (const auto& t : R.terms) {
    bool keep = type.empty();
    for (const auto& p : t.producers) keep = keep || matches(p, type);
    for (const auto& c : t.consumers) keep = keep || matches(c, type);
    if (keep) terms_out.push_back(term_json(t));
  }
  out["types"] = types;
  out["terms"] = terms_out;
  out["reserved_ids"] = R.reserved_ids;
  if (routes) {
    for (const auto& r : R.routes)
      if (type.empty() || std::find(r.physics.begin(), r.physics.end(), type) != r.physics.end())
        routes_out.push_back(route_json(r));
    out["routes"] = routes_out;
    json rules = json::array();
    for (const auto& l : R.layer_rules) rules.push_back({{"name", l.name}, {"description", l.description}});
    out["layer_rules"] = rules;
  }
  return out;
}

}  // namespace axmp::model
