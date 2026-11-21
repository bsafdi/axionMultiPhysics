// SPDX-License-Identifier: BSD-3-Clause
// axmp-transport: particle transport with Geant4, as a separate program.
//
// This program links Geant4 (and so the LGPL-3 CLHEP bundled in it) and nothing of axionMultiPhysics; the package
// exchanges files with it (job in, NPY arrays and a record out), as it does with axmp-mesher and gmsh.
//
// usage: axmp-transport --check
//        axmp-transport --nist NAME [NAME ...]
//        axmp-transport --geometry solids.json
//        axmp-transport [--threads N] job.json
// exit status: 0 success; 2 usage, invalid job, invalid solids file or unknown NIST material; 3 refused at set-up (an
// overlap of the geometry); 4 failure during the run.
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4IonisParamMat.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4SystemOfUnits.hh"
#include "G4VModularPhysicsList.hh"
#include "G4Version.hh"

#include "datasets.hpp"
#include "geometry.hpp"
#include "job.hpp"
#include "run.hpp"
#include "runtime.hpp"
#include "transport.hpp"

namespace {

using nlohmann::json;

void usage(std::ostream& os) {
  os << "usage: axmp-transport --check\n"
        "       axmp-transport --nist NAME [NAME ...]\n"
        "       axmp-transport --geometry solids.json\n"
        "       axmp-transport [--threads N] job.json\n"
        "--check prints one JSON document describing this build and its Geant4 data sets on stdout.\n"
        "--nist prints Geant4's own data of NIST materials (density, I-value, state, temperature, pressure,\n"
        "mass fractions) as one JSON document on stdout.\n"
        "--geometry builds a solids list (docs/guides/solids.md) in Geant4, checks it for overlaps and prints\n"
        "the volumes and masses of its volumes as one JSON document on stdout.\n"
        "The job file is described in docs/guides/particles.md. Exit status: 0 success; 2 usage, invalid job or\n"
        "invalid solids file; 3 refused at set-up (an overlap of the geometry); 4 failure during the run.\n";
}

// The particles of Geant4's table once FTFP_BERT has constructed them (the general ions excluded): name -> PDG code,
// mass [eV], charge [e].
json particle_table() {
  G4PhysListFactory factory;
  factory.SetVerbose(0);
  G4VModularPhysicsList* physics = factory.GetReferencePhysList("FTFP_BERT");
  physics->SetVerboseLevel(0);
  physics->ConstructParticle();
  json out = json::object();
  auto* it = G4ParticleTable::GetParticleTable()->GetIterator();
  it->reset();
  while ((*it)()) {
    const G4ParticleDefinition* p = it->value();
    if (p->IsGeneralIon() || p->GetParticleName() == "GenericIon") continue;
    out[p->GetParticleName()] = {{"pdg", p->GetPDGEncoding()},
                                 {"mass_eV", p->GetPDGMass() / eV},
                                 {"charge", p->GetPDGCharge() / eplus}};
  }
  return out;
}

const char* state_name(G4State s) {
  if (s == kStateSolid) return "solid";
  if (s == kStateLiquid) return "liquid";
  if (s == kStateGas) return "gas";
  return "undefined";
}

// Geant4's own data of NIST materials, each value converted once from Geant4's units (nlohmann::json writes the
// shortest text that reads back as the same double).
int nist(const std::vector<std::string>& names) {
  auto* nm = G4NistManager::Instance();
  json mats = json::object();
  for (const auto& name : names) {
    const G4Material* mat = nm->FindOrBuildMaterial(name);
    if (!mat) {
      std::cerr << "axmp-transport: error: Geant4's NIST tables have no material '" << name << "'" << std::endl;
      return axtr::kUsage;
    }
    json elements = json::object();
    const G4ElementVector* ev = mat->GetElementVector();
    const G4double* fraction = mat->GetFractionVector();
    for (std::size_t k = 0; k < mat->GetNumberOfElements(); ++k) {
      const std::string sym = (*ev)[k]->GetSymbol();
      elements[sym] = (elements.contains(sym) ? elements[sym].get<double>() : 0.0) + fraction[k];
    }
    mats[name] = {{"density_kg_m3", mat->GetDensity() / (kg / m3)},
                  {"mean_excitation_eV", mat->GetIonisation()->GetMeanExcitationEnergy() / eV},
                  {"state", state_name(mat->GetState())},
                  {"T_K", mat->GetTemperature() / kelvin},
                  {"pressure_Pa", mat->GetPressure() / hep_pascal},
                  {"elements", elements}};
  }
  std::cout << json({{"schema", "axionmultiphysics.transport-nist/1"}, {"materials", mats}}).dump(1) << std::endl;
  return axtr::kOk;
}

int check() {
  json sets = json::array();
  bool ok = true;
  for (const auto& d : axtr::dataset_status("FTFP_BERT")) {
    sets.push_back({{"name", d.name},
                    {"env", d.env},
                    {"path", d.path},
                    {"status", d.ok ? "ok" : "missing"},
                    {"required", d.required}});
    if (d.required && !d.ok) ok = false;
  }
  const std::string version = axtr::geant4_version();
#ifdef G4MULTITHREADED
  const bool multithreaded = true;
#else
  const bool multithreaded = false;
#endif
  json doc = {{"schema", "axionmultiphysics.transport-check/1"},
              {"program", "axmp-transport"},
              {"program_version", AXMP_TRANSPORT_VERSION},
              {"geant4_version", version},
              {"geant4_tested", version == "11.4.3"},
              {"multithreaded", multithreaded},
              {"physics_lists", axtr::offered_physics_lists()},
              {"modes", axtr::offered_modes()},
              {"geometry", true},
              {"nist", true},
              {"stepper", axtr::kStepper},
              {"particles", particle_table()},
              {"datasets", sets},
              {"ok", ok && multithreaded}};
  std::cout << doc.dump(1) << std::endl;
  return axtr::kOk;
}

}  // namespace

int main(int argc, char** argv) {
  axtr::process_start();
  axtr::route_geant4_output_to_stderr();

  bool do_check = false;
  bool threads_given = false;
  int threads = 1;
  std::string job_path;
  std::string geometry_path;
  bool do_nist = false;
  std::vector<std::string> nist_names;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--nist") {
      do_nist = true;
      while (i + 1 < argc && argv[i + 1][0] != '-') nist_names.push_back(argv[++i]);
      if (nist_names.empty()) {
        std::cerr << "axmp-transport: error: --nist needs at least one material name\n";
        return axtr::kUsage;
      }
    } else if (a == "--check") {
      do_check = true;
    } else if (a == "-h" || a == "--help") {
      usage(std::cout);
      return axtr::kOk;
    } else if (a == "--threads") {
      if (i + 1 >= argc) {
        std::cerr << "axmp-transport: error: --threads needs a number\n";
        return axtr::kUsage;
      }
      const std::string v = argv[++i];
      char* end = nullptr;
      const long n = std::strtol(v.c_str(), &end, 10);
      if (v.empty() || *end != '\0' || n < 1 || n > 4096) {
        std::cerr << "axmp-transport: error: --threads needs an integer from 1 to 4096, not '" << v << "'\n";
        return axtr::kUsage;
      }
      threads = static_cast<int>(n);
      threads_given = true;
    } else if (a == "--geometry") {
      if (i + 1 >= argc || !geometry_path.empty()) {
        std::cerr << "axmp-transport: error: --geometry needs one solids file\n";
        return axtr::kUsage;
      }
      geometry_path = argv[++i];
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "axmp-transport: error: unknown option '" << a << "'\n";
      usage(std::cerr);
      return axtr::kUsage;
    } else if (job_path.empty()) {
      job_path = a;
    } else {
      std::cerr << "axmp-transport: error: more than one job file\n";
      return axtr::kUsage;
    }
  }
  if (do_nist) {
    if (do_check || !job_path.empty() || !geometry_path.empty()) {
      std::cerr << "axmp-transport: error: --nist takes material names only\n";
      return axtr::kUsage;
    }
    return nist(nist_names);
  }
  if (do_check) {
    if (!job_path.empty() || !geometry_path.empty()) {
      std::cerr << "axmp-transport: error: --check takes no job file\n";
      return axtr::kUsage;
    }
    return check();
  }
  if (!geometry_path.empty()) {
    if (!job_path.empty() || threads_given) {
      std::cerr << "axmp-transport: error: --geometry takes no job file and no --threads\n";
      return axtr::kUsage;
    }
    return axtr::run_geometry(geometry_path);
  }
  if (job_path.empty()) {
    usage(std::cerr);
    return axtr::kUsage;
  }
  axtr::Job job;
  try {
    job = axtr::load_job(job_path);
  } catch (const axtr::JobError& e) {
    std::cerr << "axmp-transport: error: " << e.what() << std::endl;
    return axtr::kUsage;
  }
  if (job.mode == "transport") return axtr::run_transport(job, threads);
  if (job.mode == "probe") return axtr::run_probe(job);
  return axtr::run_job(job, threads);
}
