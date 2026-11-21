// SPDX-License-Identifier: BSD-3-Clause
// The Geant4 data sets of release 11.4 (names and environment variables of its G4DatasetDefinitions.cmake) and which
// of them the offered physics lists read.
#pragma once

#include <string>
#include <vector>

namespace axtr {

struct DatasetStatus {
  std::string name;
  std::string env;
  std::string path;  // where Geant4 finds the set (the variable, GEANT4_DATA_DIR, or the installation); empty when nowhere
  bool ok = false;   // that is an existing directory
  bool required = false;
};

// Status of every data set. `required` marks those the physics list `physics_list` needs.
std::vector<DatasetStatus> dataset_status(const std::string& physics_list);

}  // namespace axtr
