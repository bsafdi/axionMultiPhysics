// SPDX-License-Identifier: BSD-3-Clause
#include "datasets.hpp"

#include <cstdlib>
#include <filesystem>
#include <system_error>

#include "G4EnvironmentUtils.hh"

namespace axtr {

namespace {
struct Def {
  const char* name;
  const char* env;
  bool required_by_ftfp_bert;
};
// Which sets a list needs was established from the source of Geant4 11.4.3 and from file opens. A 20-event muon run
// opens files of G4EMLOW, G4PARTICLEXS, PhotonEvaporation and G4ENSDFSTATE only, and nothing else for FTFP_BERT. The
// others are listed, not required: G4PII is read only for PIXE (off in the standard EM physics), G4SAIDDATA is
// compiled but never instantiated, G4ABLA is reached from Bertini only when G4CASCADE_USE_ABLA is set, and a 200-event
// pi+ job of 1 to 5 GeV (hadronic interactions) gave byte-identical output with all three pointing at an empty
// directory. Neutron HP, optical surfaces, INCL, channeling and radioactive decay belong to other lists.
constexpr Def kDefs[] = {
    {"G4NDL", "G4NEUTRONHPDATA", false},
    {"G4EMLOW", "G4LEDATA", true},
    {"PhotonEvaporation", "G4LEVELGAMMADATA", true},
    {"RadioactiveDecay", "G4RADIOACTIVEDATA", false},
    {"G4PARTICLEXS", "G4PARTICLEXSDATA", true},
    {"G4PII", "G4PIIDATA", false},
    {"RealSurface", "G4REALSURFACEDATA", false},
    {"G4SAIDDATA", "G4SAIDXSDATA", false},
    {"G4ABLA", "G4ABLADATA", false},
    {"G4INCL", "G4INCLDATA", false},
    {"G4ENSDFSTATE", "G4ENSDFSTATEDATA", true},
    {"G4CHANNELING", "G4CHANNELINGDATA", false},
};
}  // namespace

std::vector<DatasetStatus> dataset_status(const std::string& physics_list) {
  std::vector<DatasetStatus> out;
  for (const Def& d : kDefs) {
    DatasetStatus s;
    s.name = d.name;
    s.env = d.env;
    // Geant4's own lookup: the variable of the data set, else GEANT4_DATA_DIR, else the installation's data directory
    const char* v = G4FindDataDir(d.env);
    if (v) s.path = v;
    std::error_code ec;
    s.ok = !s.path.empty() && std::filesystem::is_directory(s.path, ec);
    // the same sets are required for the three offered lists
    s.required = (physics_list == "FTFP_BERT" || physics_list == "FTFP_BERT_EMZ" || physics_list == "QGSP_BIC") &&
                 d.required_by_ftfp_bert;
    out.push_back(s);
  }
  return out;
}

}  // namespace axtr
