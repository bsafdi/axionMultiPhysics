// SPDX-License-Identifier: BSD-3-Clause
// The mesher input of a 3D design (see mesher_input_3d.hpp).
#include "axionmultiphysics/mesher_input_3d.hpp"

#include <algorithm>

#include "axionmultiphysics/axi.hpp"
#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/units.hpp"

namespace axmp {

using json = nlohmann::json;

json mesher_input_3d(const spec::Design& d, const SolveOptions& o, std::map<std::string, int>* attr_of_region,
                     std::map<int, std::string>* region_of_attr) {
  AXMP_REQUIRE(d.curves.empty(), "zero-thickness curves (sheets) are not supported in 3D designs", "/curves",
              "Model sheets as thin regions.");

  json in;
  in["dim"] = 3;
  in["domain_radius"] = d.mesh.domain_radius;  // <= 0: automatic in the mesher
  in["infinity_shell"] = d.fullwave.rf_only ? 0.0 : 1.25;                  // exact open boundary: shell [Rb, 1.25 Rb] mapped to infinity
  in["shell_attr"] = kShellAttr;
  in["size"] = d.mesh.size;
  in["symmetry"]=d.mesh.symmetry;
  if(!d.mesh.domain_solid.is_null()) in["domain_solid"]=d.mesh.domain_solid;
  if (d.fullwave.enabled && !d.fullwave.frequencies.empty()) {
    // full wave: the shell is a complex-scaled map (absorbing layer), resolved with more element layers;
    // at most lambda/8 per element at the highest frequency
    in["shell_layers"] = 4;
    const double fmax = *std::max_element(d.fullwave.frequencies.begin(), d.fullwave.frequencies.end());
    in["max_size"] = units::c / fmax / 8;
  }
  in["surface_size"] = d.mesh.surface_size;
  in["order"] = d.mesh.geometry_order > 0 ? d.mesh.geometry_order : std::min(d.mesh.order, 2);
  in["outer_attr"] = axi::kOuterAttr;
  in["air_attr"] = axi::kAirAttr;
  json regs = json::array(), refine = json::array();
  int attr = 2;
  for (auto& g : d.regions) {
    json r = {{"name", g.name}, {"attr", attr}, {"solid", g.solid.resolved()}};
    if (g.mesh_size > 0) r["size"] = g.mesh_size;
    if (g.surface_size > 0) r["surface_size"] = g.surface_size;  // per-region surface size (axmp-mesher mesh3d)
    regs.push_back(r);
    if (attr_of_region) (*attr_of_region)[g.name] = attr;
    if (region_of_attr) (*region_of_attr)[attr] = g.name;
    auto k = d.materials.at(g.material).kind;
    bool wind = std::any_of(d.windings.begin(), d.windings.end(), [&](auto& w) { return w.region == g.name; });
    bool port = d.pickup && d.pickup->port == g.name;
    if (d.mesh.remove_pec && (k==spec::MaterialKind::Conductor || k==spec::MaterialKind::Superconductor))
      regs.back()["remove_pec"]=true;
    if (k != spec::MaterialKind::Vacuum || wind || port) refine.push_back(g.name);
    ++attr;
  }
  if (region_of_attr) {
    (*region_of_attr)[axi::kAirAttr] = "air";
    (*region_of_attr)[kShellAttr] = "infinity_shell";
  }
  in["regions"] = regs;
  in["refine_regions"] = refine;
  if (!o.size_field_file.empty()) in["size_field_file"] = o.size_field_file;
  const std::string base = o.workdir + "/mesh3d";
  in["output"] = base + ".msh";
  in["info"] = base + "_info.json";
  in["threads"] = 1;  // deterministic meshes (multi-threaded HXT is not reproducible)
  if(d.source.contains("mesh"))
    for(const char* key:{"far_size","surface_layer","threads","algorithm3d","curvature_points","extend_from_boundary",
                         "optimize_netgen"})
      if(d.source["mesh"].contains(key)) in[key]=d.source["mesh"][key];

  return in;
}

}  // namespace axmp
