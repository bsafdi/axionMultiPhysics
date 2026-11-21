// SPDX-License-Identifier: BSD-3-Clause
#include "hello.hpp"

#include <limits>
#include <sstream>

#include "G4Box.hh"
#include "G4Event.hh"
#include "G4FieldManager.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4MagneticField.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4TransportationManager.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

#include "runtime.hpp"
#include "seeding.hpp"

namespace axtr {

void ChunkArrays::reset(std::uint64_t n) {
  events = n;
  event.assign(n, std::numeric_limits<std::uint64_t>::max());  // a row nobody wrote stays recognisable
  seed_words.assign(4 * n, 0);
  first_flat.assign(n, 0.0);
  ekin_eV.assign(n, 0.0);
  edep_eV.assign(n, 0.0);
  n_steps.assign(n, 0);
  n_tracks.assign(n, 0);
  field_calls.assign(n, 0);
}

SharedRun& shared_run() {
  static SharedRun s;
  return s;
}

namespace {

// Per-thread tallies of the event being transported on that thread.
struct EventTally {
  double edep_eV = 0.0;
  std::uint64_t steps = 0;
  std::uint64_t tracks = 0;
  std::uint64_t field_calls = 0;
};
thread_local EventTally t_tally;

// A uniform field that counts how often it is evaluated (once per call of GetFieldValue, by this thread).
class CountingUniformField : public G4MagneticField {
 public:
  explicit CountingUniformField(const G4ThreeVector& b) : b_(b) {}
  void GetFieldValue(const G4double[4], G4double* field) const override {
    ++t_tally.field_calls;
    field[0] = b_.x();
    field[1] = b_.y();
    field[2] = b_.z();
  }

 private:
  G4ThreeVector b_;
};

constexpr const char* kTargetName = "target";

class Detector : public G4VUserDetectorConstruction {
 public:
  explicit Detector(const HelloSpec& hello) : hello_(hello) {}

  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* silicon = nist->FindOrBuildMaterial("G4_Si");
    auto* world_solid = new G4Box("world", 1.0 * m, 1.0 * m, 1.0 * m);
    auto* world_lv = new G4LogicalVolume(world_solid, vacuum, "world");
    auto* world = new G4PVPlacement(nullptr, G4ThreeVector(), world_lv, "world", nullptr, false, 0, true);
    auto* target_solid = new G4Box(kTargetName, 0.5 * m, 0.5 * m, 0.5 * hello_.target_thickness_m * m);
    auto* target_lv = new G4LogicalVolume(target_solid, silicon, kTargetName);
    new G4PVPlacement(nullptr, G4ThreeVector(), target_lv, kTargetName, world_lv, false, 0, true);
    return world;
  }

  void ConstructSDandField() override {
    if (!hello_.has_field()) return;  // no field is attached, so no field call can be counted
    static thread_local CountingUniformField field(
        G4ThreeVector(hello_.field_T[0] * tesla, hello_.field_T[1] * tesla, hello_.field_T[2] * tesla));
    auto* manager = G4TransportationManager::GetTransportationManager()->GetFieldManager();
    manager->SetDetectorField(&field);
    manager->CreateChordFinder(&field);
  }

 private:
  HelloSpec hello_;
};

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction {
 public:
  PrimaryGenerator() : gun_(1) {
    const auto& hello = shared_run().hello;
    auto* particle = G4ParticleTable::GetParticleTable()->FindParticle(hello.particle);
    if (!particle) {
      std::ostringstream msg;
      msg << "unknown particle '" << hello.particle << "'";
      G4Exception("PrimaryGenerator", "axtr0001", FatalException, msg.str().c_str());
    }
    gun_.SetParticleDefinition(particle);
    gun_.SetParticleMomentumDirection(G4ThreeVector(0.0, 0.0, 1.0));
    gun_.SetParticlePosition(G4ThreeVector(0.0, 0.0, -0.5 * m));
  }

  void GeneratePrimaries(G4Event* event) override {
    // Seeding comes first: the words of (chunk seed, event index) are installed before anything is drawn. The array
    // is a member because MixMaxRng::setSeeds keeps the pointer it is given.
    const auto index = static_cast<std::uint64_t>(event->GetEventID());
    SharedRun& run = shared_run();
    event_seed_words(run.chunk_seed.load(std::memory_order_relaxed), index, words_);
    G4Random::setTheSeeds(words_, 4);
    const double u0 = G4UniformRand();

    ChunkArrays& rows = *run.arrays;
    if (index >= rows.events) {
      G4Exception("PrimaryGenerator::GeneratePrimaries", "axtr0002", FatalException,
                  "an event index beyond the events of the chunk");
    }
    const double e_eV = run.hello.energy_lo_eV + (run.hello.energy_hi_eV - run.hello.energy_lo_eV) * u0;
    for (int k = 0; k < 4; ++k) rows.seed_words[4 * index + k] = static_cast<std::uint32_t>(words_[k]);
    rows.first_flat[index] = u0;
    rows.ekin_eV[index] = e_eV;
    gun_.SetParticleEnergy(e_eV * eV);
    gun_.GeneratePrimaryVertex(event);
  }

 private:
  G4ParticleGun gun_;
  long words_[4] = {0, 0, 0, 0};
};

class EventAction : public G4UserEventAction {
 public:
  void BeginOfEventAction(const G4Event*) override {
    t_tally = EventTally{};
    SharedRun& run = shared_run();
    if (run.first_event_s.load(std::memory_order_relaxed) < 0.0) {
      double expected = -1.0;
      const double now = std::chrono::duration<double>(Clock::now() - process_start()).count();
      run.first_event_s.compare_exchange_strong(expected, now);
    }
  }

  void EndOfEventAction(const G4Event* event) override {
    ChunkArrays& rows = *shared_run().arrays;
    const auto index = static_cast<std::uint64_t>(event->GetEventID());
    rows.event[index] = index;
    rows.edep_eV[index] = t_tally.edep_eV;
    rows.n_steps[index] = t_tally.steps;
    rows.n_tracks[index] = t_tally.tracks;
    rows.field_calls[index] = t_tally.field_calls;
  }
};

class TrackingAction : public G4UserTrackingAction {
 public:
  void PreUserTrackingAction(const G4Track*) override { ++t_tally.tracks; }
};

class SteppingAction : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    ++t_tally.steps;
    if (!target_) target_ = G4LogicalVolumeStore::GetInstance()->GetVolume(kTargetName);
    const G4VPhysicalVolume* volume = step->GetPreStepPoint()->GetPhysicalVolume();
    if (volume && volume->GetLogicalVolume() == target_) t_tally.edep_eV += step->GetTotalEnergyDeposit() / eV;
  }

 private:
  G4LogicalVolume* target_ = nullptr;
};

class Actions : public G4VUserActionInitialization {
 public:
  void BuildForMaster() const override { install_exception_handler(); }
  void Build() const override {
    route_geant4_output_to_stderr();
    install_exception_handler();
    SetUserAction(new PrimaryGenerator);
    SetUserAction(new EventAction);
    SetUserAction(new TrackingAction);
    SetUserAction(new SteppingAction);
  }
};

}  // namespace

G4VUserDetectorConstruction* make_hello_detector(const HelloSpec& hello) { return new Detector(hello); }
G4VUserActionInitialization* make_hello_actions() { return new Actions; }

}  // namespace axtr
