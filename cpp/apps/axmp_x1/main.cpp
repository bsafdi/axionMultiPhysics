// SPDX-License-Identifier: BSD-3-Clause
//
// axmp-x1: a plain Geant4 application that reads a GDML file and runs a gun in it with a given physics list, cuts,
// uniform field and per-event seeds. It is a cross-check of axmp-transport: it is built from Geant4's reference
// documentation and headers, shares no source with that program, and links nothing of this package (Geant4 with its
// GDML reader, and the header-only nlohmann/json).
//
//   axmp-x1 x1.json
//
// The job file holds exactly: gdml, out, physics_list, production_cut_m, integrator {delta_chord_m, delta_one_step_m,
// delta_intersection_m, epsilon_min, epsilon_max, min_step_m}, field (null or {"type": "uniform", "B_T": [3]}),
// source (a gun, see below), chunk_seed, events, threads (1), regions (names), volume_regions (volume name -> index
// into regions). Relative paths are taken against the directory of the job file.
//
// Written to <out>: volumes.json, materials.json, first_flat.npy, edep_eV.npy, hits_event.npy, hits_region.npy,
// hits_x_m.npy.
//
// Exit status: 0 done; 2 invalid job file or invalid GDML; 4 failure during the run.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "CLHEP/Random/MixMaxRng.h"
#include "G4ChordFinder.hh"
#include "G4Event.hh"
#include "G4FieldManager.hh"
#include "G4GDMLParser.hh"
#include "G4IonisParamMat.hh"
#include "G4GeometryTolerance.hh"
#include "G4LogicalVolume.hh"
#include "G4Navigator.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4RunManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4StateManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4UniformMagField.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VExceptionHandler.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4coutDestination.hh"
#include "G4ios.hh"
#include "Randomize.hh"

#include "npy.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr int kInvalid = 2;
constexpr int kRunFailed = 4;

struct Invalid : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------------------------- the job file

struct Integrator {
  double delta_chord_m, delta_one_step_m, delta_intersection_m, epsilon_min, epsilon_max, min_step_m;
};

struct Source {
  json particle;  // a name or a PDG code
  bool has_energy = false;
  double energy_eV = 0.0, momentum_eV_c = 0.0;
  G4ThreeVector at;  // internal units
  bool isotropic = false;
  G4ThreeVector direction{0, 0, 1};
  double spot_sigma_m = 0.0;
};

struct Job {
  fs::path gdml, out;
  std::string physics_list;
  double production_cut_m = 0.0;
  Integrator integrator{};
  bool has_field = false;
  double field_T[3] = {0, 0, 0};
  Source source;
  std::uint64_t chunk_seed = 0;
  std::uint64_t events = 0;
  int threads = 1;
  std::vector<std::string> regions;
  std::map<std::string, int> volume_regions;
};

void only_keys(const json& j, const std::set<std::string>& allowed, const std::string& where) {
  if (!j.is_object()) throw Invalid(where + " must be an object");
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!allowed.count(it.key())) throw Invalid("unknown key '" + it.key() + "' in " + where);
  for (const auto& k : allowed)
    if (!j.contains(k)) throw Invalid("missing key '" + k + "' in " + where);
}

double number(const json& j, const std::string& what) {
  if (!j.is_number()) throw Invalid(what + " must be a number");
  return j.get<double>();
}

std::array<double, 3> vec3(const json& j, const std::string& what) {
  if (!j.is_array() || j.size() != 3) throw Invalid(what + " must be three numbers");
  return {number(j[0], what), number(j[1], what), number(j[2], what)};
}

Job read_job(const fs::path& file) {
  std::ifstream in(file);
  if (!in) throw Invalid("cannot open " + file.string());
  json j;
  try {
    j = json::parse(in);
  } catch (const std::exception& e) {
    throw Invalid(std::string("not valid JSON: ") + e.what());
  }
  only_keys(j, {"gdml", "out", "physics_list", "production_cut_m", "integrator", "field", "source", "chunk_seed", "events",
                "threads", "regions", "volume_regions"},
            "the job");
  const fs::path base = file.parent_path();
  auto path = [&](const std::string& s) { return fs::path(s).is_absolute() ? fs::path(s) : base / s; };
  Job job;
  job.gdml = path(j.at("gdml").get<std::string>());
  job.out = path(j.at("out").get<std::string>());
  job.physics_list = j.at("physics_list").get<std::string>();
  job.production_cut_m = number(j.at("production_cut_m"), "production_cut_m");
  if (!(job.production_cut_m > 0.0)) throw Invalid("production_cut_m must be positive");
  const json& ig = j.at("integrator");
  only_keys(ig, {"delta_chord_m", "delta_one_step_m", "delta_intersection_m", "epsilon_min", "epsilon_max", "min_step_m"},
            "integrator");
  job.integrator = {number(ig.at("delta_chord_m"), "delta_chord_m"), number(ig.at("delta_one_step_m"), "delta_one_step_m"),
                    number(ig.at("delta_intersection_m"), "delta_intersection_m"),
                    number(ig.at("epsilon_min"), "epsilon_min"), number(ig.at("epsilon_max"), "epsilon_max"),
                    number(ig.at("min_step_m"), "min_step_m")};
  const json& f = j.at("field");
  if (!f.is_null()) {
    only_keys(f, {"type", "B_T"}, "field");
    if (f.at("type") != "uniform") throw Invalid("field.type must be \"uniform\"");
    const auto b = vec3(f.at("B_T"), "field.B_T");
    job.has_field = true;
    for (int k = 0; k < 3; ++k) job.field_T[k] = b[k];
  }
  const json& s = j.at("source");
  if (!s.is_object() || s.value("type", "") != "gun") throw Invalid("source.type must be \"gun\"");
  for (auto it = s.begin(); it != s.end(); ++it)
    if (!std::set<std::string>{"type", "particle", "energy_eV", "momentum_eV_c", "at_m", "direction", "spot_sigma_m"}.count(it.key()))
      throw Invalid("unknown key '" + it.key() + "' in source");
  Source& src = job.source;
  src.particle = s.at("particle");
  if (!src.particle.is_string() && !src.particle.is_number_integer()) throw Invalid("source.particle must be a name or a PDG code");
  if (s.contains("energy_eV") == s.contains("momentum_eV_c")) throw Invalid("source needs exactly one of energy_eV and momentum_eV_c");
  src.has_energy = s.contains("energy_eV");
  src.energy_eV = src.has_energy ? number(s.at("energy_eV"), "energy_eV") : 0.0;
  src.momentum_eV_c = src.has_energy ? 0.0 : number(s.at("momentum_eV_c"), "momentum_eV_c");
  if (!((src.has_energy ? src.energy_eV : src.momentum_eV_c) > 0.0)) throw Invalid("source energy or momentum must be positive");
  const auto at = vec3(s.at("at_m"), "at_m");
  src.at = G4ThreeVector(at[0] * m, at[1] * m, at[2] * m);
  const json& d = s.at("direction");
  if (d.is_string()) {
    if (d != "isotropic") throw Invalid("source.direction must be a vector or \"isotropic\"");
    src.isotropic = true;
  } else {
    const auto v = vec3(d, "direction");
    src.direction = G4ThreeVector(v[0], v[1], v[2]);
    if (!(src.direction.mag2() > 0.0)) throw Invalid("source.direction must not be zero");
    src.direction = src.direction.unit();
  }
  src.spot_sigma_m = s.contains("spot_sigma_m") ? number(s.at("spot_sigma_m"), "spot_sigma_m") : 0.0;
  if (src.spot_sigma_m < 0.0 || (src.isotropic && src.spot_sigma_m != 0.0)) throw Invalid("invalid spot_sigma_m");
  const json& cs = j.at("chunk_seed");
  if (!cs.is_number_unsigned()) throw Invalid("chunk_seed must be a non-negative integer");
  job.chunk_seed = cs.get<std::uint64_t>();
  const json& ev = j.at("events");
  if (!ev.is_number_unsigned() || ev.get<std::uint64_t>() == 0) throw Invalid("events must be a positive integer");
  job.events = ev.get<std::uint64_t>();
  const json& th = j.at("threads");
  if (!th.is_number_integer() || th.get<int>() != 1) throw Invalid("threads must be 1: this program runs the serial manager");
  job.threads = 1;
  job.regions = j.at("regions").get<std::vector<std::string>>();
  const json& vr = j.at("volume_regions");
  if (!vr.is_object()) throw Invalid("volume_regions must be an object");
  for (auto it = vr.begin(); it != vr.end(); ++it) {
    const int idx = it.value().get<int>();
    if (idx < 0 || idx >= static_cast<int>(job.regions.size())) throw Invalid("volume_regions['" + it.key() + "'] is not a region index");
    job.volume_regions[it.key()] = idx;
  }
  return job;
}

// ---------------------------------------------------------------------------------------------- run state

// The words that seed event `event` of a chunk with seed `chunk_seed`: outputs 2e and 2e+1 of the SplitMix64 sequence
// that starts at the state chunk_seed (Steele, Lea, Flood 2014), each split into its low and high 32 bits, low first.
std::uint64_t splitmix_output(std::uint64_t state, std::uint64_t n) {
  std::uint64_t z = state + n * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

struct State {
  Job job;
  std::map<const G4LogicalVolume*, int> region_of;  // by logical volume
  std::vector<double> first_flat;
  std::vector<double> edep_eV;  // events x regions
  std::vector<std::uint64_t> hit_event;
  std::vector<std::int32_t> hit_region;
  std::vector<double> hit_x_m;  // 3 per hit
  std::vector<char> transported;
  G4VPhysicalVolume* world = nullptr;
};
State g_state;

int region_index(const G4LogicalVolume* lv) {
  const auto it = g_state.region_of.find(lv);
  return it == g_state.region_of.end() ? -1 : it->second;
}

// ---------------------------------------------------------------------------------------------- Geant4 classes

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    G4GDMLParser parser;
    parser.SetOverlapCheck(false);
    parser.Read(g_state.job.gdml.string(), false);  // no schema validation: that would read the schema from the network
    G4VPhysicalVolume* world = parser.GetWorldVolume();
    g_state.world = world;

    const Integrator& ig = g_state.job.integrator;
    auto* store = G4LogicalVolumeStore::GetInstance();
    for (G4LogicalVolume* lv : *store) {
      const auto it = g_state.job.volume_regions.find(lv->GetName());
      if (it != g_state.job.volume_regions.end()) g_state.region_of[lv] = it->second;
    }
    // Every volume gets a field manager of its own, so that no volume takes its mother's; the volumes of a region get
    // the uniform field, the others (the world and the voids) none.
    for (G4LogicalVolume* lv : *store) {
      G4FieldManager* manager = nullptr;
      if (g_state.job.has_field && g_state.region_of.count(lv)) {
        const double* b = g_state.job.field_T;
        auto* field = new G4UniformMagField(G4ThreeVector(b[0] * tesla, b[1] * tesla, b[2] * tesla));
        auto* chord = new G4ChordFinder(field, ig.min_step_m * m, nullptr, G4ChordFinder::kTemplatedStepperType);
        chord->SetDeltaChord(ig.delta_chord_m * m);
        manager = new G4FieldManager(field, chord);
        manager->SetDeltaOneStep(ig.delta_one_step_m * m);
        manager->SetDeltaIntersection(ig.delta_intersection_m * m);
        if (!manager->SetMaximumEpsilonStep(ig.epsilon_max) || !manager->SetMinimumEpsilonStep(ig.epsilon_min))
          throw Invalid("epsilon_min and epsilon_max are not accepted by Geant4");
      } else {
        manager = new G4FieldManager();
      }
      lv->SetFieldManager(manager, false);
    }
    return world;
  }
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : gun_(1) {
    const Source& s = g_state.job.source;
    auto* table = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particle = s.particle.is_string() ? table->FindParticle(s.particle.get<std::string>())
                                                            : table->FindParticle(s.particle.get<int>());
    if (!particle) throw Invalid("unknown particle " + s.particle.dump());
    gun_.SetParticleDefinition(particle);
    if (s.has_energy) gun_.SetParticleEnergy(s.energy_eV * eV);
    else gun_.SetParticleMomentum(s.momentum_eV_c * eV);
  }

  void GeneratePrimaries(G4Event* event) override {
    const auto index = static_cast<std::uint64_t>(event->GetEventID());
    const std::uint64_t seed = g_state.job.chunk_seed;
    const std::uint64_t o0 = splitmix_output(seed, 2 * index + 1);
    const std::uint64_t o1 = splitmix_output(seed, 2 * index + 2);
    words_[0] = static_cast<long>(o0 & 0xFFFFFFFFULL);
    words_[1] = static_cast<long>(o0 >> 32);
    words_[2] = static_cast<long>(o1 & 0xFFFFFFFFULL);
    words_[3] = static_cast<long>(o1 >> 32);
    G4Random::setTheSeeds(words_, 4);  // before anything is drawn
    const double u0 = G4UniformRand();
    if (index >= g_state.first_flat.size()) throw std::runtime_error("an event index beyond the events of the run");
    g_state.first_flat[index] = u0;

    const Source& s = g_state.job.source;
    G4ThreeVector direction = s.direction;
    G4ThreeVector position = s.at;
    if (s.isotropic) {
      const double u1 = G4UniformRand();
      const double u2 = G4UniformRand();
      const double c = 1.0 - 2.0 * u1;
      const double phi = CLHEP::twopi * u2;
      const double sn = std::sqrt(std::max(0.0, 1.0 - c * c));
      direction = G4ThreeVector(sn * std::cos(phi), sn * std::sin(phi), c);
    } else if (s.spot_sigma_m > 0.0) {
      const double u1 = G4UniformRand();
      const double u2 = G4UniformRand();
      const double rho = s.spot_sigma_m * std::sqrt(-2.0 * std::log(1.0 - u1)) * m;
      const double psi = CLHEP::twopi * u2;
      G4ThreeVector e1 = direction.cross(G4ThreeVector(0, 0, 1));
      e1 = e1.mag2() > 0.0 ? e1.unit() : G4ThreeVector(1, 0, 0);
      const G4ThreeVector e2 = direction.cross(e1);
      position = s.at + rho * (std::cos(psi) * e1 + std::sin(psi) * e2);
    }
    gun_.SetParticleMomentumDirection(direction);
    gun_.SetParticlePosition(position);
    gun_.GeneratePrimaryVertex(event);
  }

 private:
  G4ParticleGun gun_;
  long words_[4] = {0, 0, 0, 0};  // the engine keeps the pointer it is given
};

// Deposits go to the region of the step's pre-step volume. A step that begins on the boundary of a volume of a region
// is a hit at its first point, unless the track's previous step was no longer than the surface tolerance (a repeat
// of the same crossing). A track leaves the domain (stopped and killed) when its step ends at the world's outer
// boundary, or when a step that began in the world or a void, longer than the surface tolerance, ends and a point a
// little way along it (the smaller of half the step and 1 um) lies in the world or a void, as a navigator without
// history reports it. A step that merely ends in the world or a void ends nothing: Geant4 can locate a zero-length
// step there at an interface.
class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const double tolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
    const G4Track* track = step->GetTrack();
    const double length = step->GetStepLength();
    const bool has_previous = track->GetTrackID() == last_track_;
    const bool previous_long = has_previous && last_length_ > tolerance;
    last_track_ = track->GetTrackID();
    last_length_ = length;

    const G4StepPoint* pre = step->GetPreStepPoint();
    const G4VPhysicalVolume* pre_volume = pre->GetPhysicalVolume();
    const int region = pre_volume ? region_index(pre_volume->GetLogicalVolume()) : -1;
    const auto event = static_cast<std::uint64_t>(G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID());
    if (region >= 0) {
      g_state.edep_eV[event * g_state.job.regions.size() + region] += step->GetTotalEnergyDeposit() / eV;
      if (pre->GetStepStatus() == fGeomBoundary && previous_long) {
        const G4ThreeVector x = pre->GetPosition();
        g_state.hit_event.push_back(event);
        g_state.hit_region.push_back(region);
        g_state.hit_x_m.push_back(x.x() / m);
        g_state.hit_x_m.push_back(x.y() / m);
        g_state.hit_x_m.push_back(x.z() / m);
      }
    }
    if (step->GetPostStepPoint()->GetStepStatus() == fWorldBoundary) {
      return;  // Geant4 itself kills a track that leaves the world
    }
    const G4TrackStatus status = track->GetTrackStatus();
    if (region < 0 && length > tolerance && (status == fAlive || status == fStopButAlive)) {
      const double reach = std::min(0.5 * length, 1.0 * um);
      const G4ThreeVector probe = pre->GetPosition() + reach * pre->GetMomentumDirection();
      if (!navigator_) {
        navigator_ = std::make_unique<G4Navigator>();
        navigator_->SetWorldVolume(g_state.world);
      }
      const G4VPhysicalVolume* found = navigator_->LocateGlobalPointAndSetup(probe, nullptr, false, true);
      if (!found || region_index(found->GetLogicalVolume()) < 0) step->GetTrack()->SetTrackStatus(fStopAndKill);
    }
  }

 private:
  G4int last_track_ = -1;
  double last_length_ = 0.0;
  std::unique_ptr<G4Navigator> navigator_;
};

class Events : public G4UserEventAction {
 public:
  void EndOfEventAction(const G4Event* event) override {
    g_state.transported[static_cast<std::size_t>(event->GetEventID())] = 1;
  }
};

class Actions : public G4VUserActionInitialization {
 public:
  void Build() const override {
    SetUserAction(new Gun);
    SetUserAction(new Stepping);
    SetUserAction(new Events);
  }
};

// Output of Geant4 goes to stderr (stdout stays empty); a fatal exception ends the program with status 4.
class ToStderr : public G4coutDestination {
 public:
  G4int ReceiveG4cout(const G4String& s) override {
    std::cerr << s << std::flush;
    return 0;
  }
  G4int ReceiveG4cerr(const G4String& s) override {
    std::cerr << s << std::flush;
    return 0;
  }
};

class Handler : public G4VExceptionHandler {
 public:
  G4bool Notify(const char* origin, const char* code, G4ExceptionSeverity severity, const char* description) override {
    std::cerr << "axmp-x1: Geant4 " << (severity == FatalException || severity == FatalErrorInArgument ? "fatal exception " : "exception ")
              << code << " in " << origin << ": " << description << std::endl;
    if (severity == FatalException || severity == FatalErrorInArgument || severity == RunMustBeAborted ||
        severity == EventMustBeAborted) {
      std::_Exit(kRunFailed);
    }
    return false;  // warnings continue
  }
};

// ---------------------------------------------------------------------------------------------- reports

std::string state_name(const G4Material* mat) {
  switch (mat->GetState()) {
    case kStateSolid: return "solid";
    case kStateLiquid: return "liquid";
    case kStateGas: return "gas";
    default: return "undefined";
  }
}

void write_text(const fs::path& file, const std::string& text) {
  std::ofstream f(file, std::ios::binary | std::ios::trunc);
  f << text;
  f.flush();
  if (!f) throw std::runtime_error("cannot write " + file.string());
}

void write_geometry_reports(const fs::path& out) {
  json volumes = json::object();
  json materials = json::array();
  std::vector<const G4Material*> seen;
  for (const G4LogicalVolume* lv : *G4LogicalVolumeStore::GetInstance()) {
    volumes[lv->GetName()] = lv->GetSolid()->GetCubicVolume() / m3;
    const G4Material* mat = lv->GetMaterial();
    if (std::find(seen.begin(), seen.end(), mat) != seen.end()) continue;
    seen.push_back(mat);
    json elements = json::object();
    const G4ElementVector* ev = mat->GetElementVector();
    const G4double* fraction = mat->GetFractionVector();
    for (std::size_t k = 0; k < mat->GetNumberOfElements(); ++k) {
      const std::string symbol = (*ev)[k]->GetSymbol();
      elements[symbol] = (elements.contains(symbol) ? elements[symbol].get<double>() : 0.0) + fraction[k];
    }
    materials.push_back({{"g4_name", mat->GetName()},
                         {"density_kg_m3", mat->GetDensity() / (kg / m3)},
                         {"mean_excitation_eV", mat->GetIonisation()->GetMeanExcitationEnergy() / eV},
                         {"state", state_name(mat)},
                         {"T_K", mat->GetTemperature() / kelvin},
                         {"pressure_Pa", mat->GetPressure() / hep_pascal},
                         {"elements", elements}});
  }
  write_text(out / "volumes.json", volumes.dump(1) + "\n");
  write_text(out / "materials.json", materials.dump(1) + "\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    std::cerr << "usage: axmp-x1 x1.json\n"
                 "Reads the GDML file named by the job, runs the gun with the physics list, cuts, uniform field and\n"
                 "per-event seeds of the job, and writes volumes.json, materials.json and the event arrays to its out\n"
                 "directory. Exit status: 0 done; 2 invalid job or GDML; 4 failure during the run.\n";
    return argc == 2 ? 0 : kInvalid;
  }
  try {
    g_state.job = read_job(argv[1]);
  } catch (const std::exception& e) {
    std::cerr << "axmp-x1: error: invalid job file: " << e.what() << std::endl;
    return kInvalid;
  }
  const Job& job = g_state.job;
  try {
    fs::create_directories(job.out);
    static ToStderr destination;
    G4iosSetDestination(&destination);
    static Handler handler;
    G4StateManager::GetStateManager()->SetExceptionHandler(&handler);

    G4Random::setTheEngine(new CLHEP::MixMaxRng);
    std::unique_ptr<G4RunManager> manager(G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
    G4PhysListFactory factory;
    factory.SetVerbose(0);
    G4VModularPhysicsList* physics = factory.GetReferencePhysList(job.physics_list);
    if (!physics) {
      std::cerr << "axmp-x1: error: unknown physics list '" << job.physics_list << "'" << std::endl;
      return kInvalid;
    }
    physics->SetVerboseLevel(0);
    physics->SetDefaultCutValue(job.production_cut_m * m);
    manager->SetVerboseLevel(0);
    manager->SetUserInitialization(new Detector);
    manager->SetUserInitialization(physics);
    manager->SetUserInitialization(new Actions);
    g_state.first_flat.assign(job.events, 0.0);
    g_state.edep_eV.assign(job.events * job.regions.size(), 0.0);
    g_state.transported.assign(job.events, 0);
    manager->Initialize();
    write_geometry_reports(job.out);
    manager->BeamOn(static_cast<G4int>(job.events));
    for (std::uint64_t i = 0; i < job.events; ++i)
      if (!g_state.transported[i]) throw std::runtime_error("event " + std::to_string(i) + " was not transported");

    const std::size_t n = job.events, r = job.regions.size(), hits = g_state.hit_event.size();
    x1::write_npy(job.out / "first_flat.npy", "<f8", {n}, g_state.first_flat.data(), 8 * n);
    x1::write_npy(job.out / "edep_eV.npy", "<f8", {n, r}, g_state.edep_eV.data(), 8 * n * r);
    x1::write_npy(job.out / "hits_event.npy", "<u8", {hits}, g_state.hit_event.data(), 8 * hits);
    x1::write_npy(job.out / "hits_region.npy", "<i4", {hits}, g_state.hit_region.data(), 4 * hits);
    x1::write_npy(job.out / "hits_x_m.npy", "<f8", {hits, 3}, g_state.hit_x_m.data(), 24 * hits);
    manager.reset();
  } catch (const Invalid& e) {
    std::cerr << "axmp-x1: error: " << e.what() << std::endl;
    return kInvalid;
  } catch (const std::exception& e) {
    std::cerr << "axmp-x1: error: " << e.what() << std::endl;
    return kRunFailed;
  }
  return 0;
}
