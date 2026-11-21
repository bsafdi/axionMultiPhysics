// The input of axmp-mesher for a 3D design: what the 3D engine (Model3D) writes as mesh3d_in.json. It is made here, in
// every build, so that a build without the 3D engine, and every rank of a parallel program, can have it without
// running the engine.
#pragma once

#include <map>
#include <string>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/model.hpp"
#include "axionmultiphysics/spec.hpp"

namespace axmp {

// The attribute of the infinity shell in the 3D mesh.
inline constexpr int kShellAttr = 898;

// The mesher input of the 3D design `d` (output and info paths in o.workdir, the size field file of `o`). Refuses a
// design with sheets. `attr_of_region` and `region_of_attr` (either may be null) receive the attribute of each region
// (2, 3, ... in document order; the background `air` and `infinity_shell` only in region_of_attr).
nlohmann::json mesher_input_3d(const spec::Design& d, const SolveOptions& o, std::map<std::string, int>* attr_of_region,
                               std::map<int, std::string>* region_of_attr);

}  // namespace axmp
