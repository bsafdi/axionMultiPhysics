// What the graph and the validation of a model document share: the properties of a material by name, which of them an
// instance reads in a step, and the regions a name stands for.  Internal: not installed.
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "axionmultiphysics/physics_registry.hpp"
#include "axionmultiphysics/spec_model.hpp"

namespace axmp::model::detail {

// A property of a material: its block ("" for the material level) and key, and where MaterialProps holds it.
struct PropertyRef {
  const char* block;
  const char* key;
  Property MaterialProps::* member;
  // the name the route table uses: "density", "fluid.mu_Pa_s"
  std::string name() const { return *block ? std::string(block) + "." + key : std::string(key); }
  // its JSON pointer in material `material`
  std::string pointer(const std::string& material) const {
    return "/materials/" + material + (*block ? std::string("/") + block : std::string()) + "/" + key;
  }
};

// Every property a physics reads, in the order of the property edges of a material.
inline const std::vector<PropertyRef>& properties() {
  static const std::vector<PropertyRef> all = {
      {"", "density", &MaterialProps::density},
      {"fluid", "mu_Pa_s", &MaterialProps::mu_Pa_s},
      {"fluid", "beta_1_K", &MaterialProps::beta_1_K},
      {"electrolyte", "kappa_S_m", &MaterialProps::kappa_S_m},
      {"electric", "sigma_S_m", &MaterialProps::electric_sigma_S_m},
      {"thermal", "k_W_mK", &MaterialProps::k_W_mK},
      {"thermal", "cp_J_kgK", &MaterialProps::cp_J_kgK},
      {"mechanical", "alpha_1_K", &MaterialProps::alpha_1_K},
      {"mechanical", "E_Pa", &MaterialProps::E_Pa},
      {"mechanical", "nu", &MaterialProps::nu},
  };
  return all;
}
inline const PropertyRef* property(const std::string& name) {
  for (const PropertyRef& p : properties())
    if (p.name() == name) return &p;
  return nullptr;
}

inline bool contains(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

// True when a coupling of `term` whose consumer is `id` has both ends in the step.
inline bool consumes(const ResolvedModel& m, const StudyStep& step, const std::string& id, const std::string& term) {
  for (const Coupling& c : m.couplings)
    if (c.term == term && c.to == id && contains(step.physics, c.from) && contains(step.physics, c.to)) return true;
  return false;
}
inline bool has_source(const Instance& inst, const std::string& kind) {
  for (const SourceTerm& s : inst.sources)
    if (s.kind == kind) return true;
  return false;
}

// Whether instance `inst` reads property `p` in `step` (design note 6.1): a flow its viscosity, and its density with
// convection, in time or under buoyancy; a heat instance its conductivity, and density and heat capacity when it is
// advected or solved in time; a current distribution the electrolyte conductivity; a conduction the electric
// conductivity; a solid its moduli; the buoyancy and thermal-expansion terms the coefficients of their consumer.
inline bool reads(const ResolvedModel& m, const StudyStep& step, const Instance& inst, const PropertyRef& p) {
  const std::string block = p.block, key = p.key;
  const bool transient = step.type == "transient";
  if (inst.type == "fluid.incompressible") {
    if (block == "fluid" && key == "mu_Pa_s") return true;
    const bool buoyant = consumes(m, step, inst.id, "buoyancy");
    if (block == "fluid" && key == "beta_1_K") return buoyant;
    if (key == "density") {
      const auto eq = inst.options.find("equation");
      const bool convective = eq != inst.options.end() && eq->second.value == "navier_stokes";
      return convective || transient || buoyant;
    }
    return false;
  }
  if (inst.type == "heat.transfer") {
    if (block == "thermal" && key == "k_W_mK") return true;
    const bool advected = consumes(m, step, inst.id, "advection") || has_source(inst, "prescribed_advection");
    if (key == "density" || (block == "thermal" && key == "cp_J_kgK")) return advected || transient;
    return false;
  }
  if (inst.type == "electrochem.current_distribution") return block == "electrolyte";
  if (inst.type == "current.conduction") return block == "electric";
  if (inst.type == "solid.linear") {
    if (block != "mechanical") return false;
    if (key == "alpha_1_K") return consumes(m, step, inst.id, "thermal_expansion");
    return true;
  }
  return false;
}

// Whether the type publishes the field ("c_<species>" stands for every concentration).
inline bool provides(const PhysicsType& t, const std::string& field) {
  for (const std::string& f : t.provides) {
    if (f == field) return true;
    if (f == "c_<species>" && field.rfind("c_", 0) == 0) return true;
  }
  return false;
}

// The enabled regions that `name` (a region or a domain) stands for, in the order of the document's regions.
inline std::vector<std::string> regions_of(const ResolvedModel& m, const std::vector<std::string>& names) {
  std::vector<std::string> out;
  for (const auto& r : m.regions) {
    bool in = false;
    for (const std::string& name : names) {
      if (name == r.name) in = true;
      const auto d = m.domains.find(name);
      if (d != m.domains.end() && contains(d->second.regions, r.name)) in = true;
    }
    if (in) out.push_back(r.name);
  }
  return out;
}

inline std::vector<std::string> intersection(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  std::vector<std::string> out;
  for (const std::string& x : a)
    if (contains(b, x)) out.push_back(x);
  return out;
}

}  // namespace axmp::model::detail
