// SPDX-License-Identifier: BSD-3-Clause
#include "transport.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "G4ChordFinder.hh"
#include "G4Event.hh"
#include "G4FieldManager.hh"
#include "G4GeometryTolerance.hh"
#include "G4IntegrationDriver.hh"
#include "G4LogicalVolume.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4MTRunManager.hh"
#include "G4Navigator.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4PhysicalConstants.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4TDormandPrince45.hh"
#include "G4TaskRunManager.hh"
#include "G4Track.hh"
#include "G4Transportation.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VSolid.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"
#include "CLHEP/Random/MixMaxRng.h"

#include "datasets.hpp"
#include "fields.hpp"
#include "geometry.hpp"
#include "npy.hpp"
#include "run.hpp"
#include "runtime.hpp"
#include "seeding.hpp"

namespace axtr {

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

int error(int code, const std::string& what) {
  std::cerr << "axmp-transport: error: " << what << std::endl;
  return code;
}

void write_text(const fs::path& file, const std::string& text) {
  std::ofstream f(file, std::ios::binary | std::ios::trunc);
  f << text;
  f.flush();
  if (!f) throw std::runtime_error("cannot write " + file.string());
}

double seconds(Clock::time_point from, Clock::time_point to) { return std::chrono::duration<double>(to - from).count(); }

// ============================================================================================ shared state
struct HitRecord {
  std::int32_t track, parent, pdg, region;
  double x[3], p[3], t_s, ekin_eV;
};

// One row per event of a chunk, written only by the thread that runs the event (no lock), except the tallies, which
// are integer sums (order-independent) under a mutex.
struct Rows {
  std::uint64_t n = 0;
  std::size_t R = 0;
  std::vector<std::uint64_t> event, n_steps, n_tracks, field_calls, outside_calls, looping_killed, abandoned_killed;
  std::vector<std::uint32_t> seed_words;
  std::vector<double> first_flat, primary, edep, exit_eV, looping_eV, abandoned_eV;
  std::vector<std::vector<std::vector<HitRecord>>> hits;  // [scorer][event]
  std::vector<std::vector<std::uint32_t>> hits_dropped;   // [scorer][event]
  std::vector<std::vector<std::int64_t>> tally;           // [tally][bin]
  std::mutex tally_mutex;

  void reset(std::uint64_t events, std::size_t regions, std::size_t scorers, const std::vector<TallySpec>& tallies) {
    n = events;
    R = regions;
    event.assign(n, std::numeric_limits<std::uint64_t>::max());
    n_steps.assign(n, 0);
    n_tracks.assign(n, 0);
    field_calls.assign(n, 0);
    outside_calls.assign(n, 0);
    looping_killed.assign(n, 0);
    abandoned_killed.assign(n, 0);
    seed_words.assign(4 * n, 0);
    first_flat.assign(n, 0.0);
    primary.assign(7 * n, 0.0);
    edep.assign(R * n, 0.0);
    exit_eV.assign(n, 0.0);
    looping_eV.assign(n, 0.0);
    abandoned_eV.assign(n, 0.0);
    hits.assign(scorers, {});
    hits_dropped.assign(scorers, {});
    for (std::size_t s = 0; s < scorers; ++s) {
      hits[s].assign(n, {});
      hits_dropped[s].assign(n, 0);
    }
    tally.assign(tallies.size(), {});
    for (std::size_t t = 0; t < tallies.size(); ++t) tally[t].assign(tallies[t].size(), 0);
  }
};

struct Shared {
  const Job* job = nullptr;
  SolidsSpec spec;
  BuiltGeometry geo;
  std::unordered_map<const G4LogicalVolume*, int> region_of;   // -1: the world and the voids
  std::vector<std::shared_ptr<const MapPatch>> patch_of_field; // per job field; null for the analytic fields
  std::vector<int> field_of_region;                            // -1: field-free
  std::vector<std::vector<int>> scorers_of_region, tallies_of_region;
  std::vector<std::set<std::string>> scorer_particles;
  const G4ParticleDefinition* particle = nullptr;
  double mass_eV = 0.0, ekin_eV = 0.0, p_eV_c = 0.0;
  std::atomic<std::uint64_t> chunk_seed{0};
  Rows* rows = nullptr;
  std::atomic<double> first_event_s{-1.0};
};
Shared& shared() {
  static Shared s;
  return s;
}

// Per-thread tallies of the event being transported on that thread.
struct EventState {
  std::vector<double> edep;
  double exit_eV = 0.0, looping_eV = 0.0, abandoned_eV = 0.0;
  std::uint64_t steps = 0, tracks = 0, looping = 0, abandoned = 0;
  std::vector<std::vector<HitRecord>> hits;
  std::vector<std::uint32_t> dropped;
  std::vector<std::unordered_map<std::uint64_t, double>> tally;
  const G4Transportation* transportation = nullptr;
  double killed_at_start = 0.0;
  int prev_track = 0;     // the track of the previous step on this thread, and that step's length
  double prev_len = -1.0;
};
thread_local EventState t_ev;

// A navigator of this thread that locates points without the history of any track.
G4Navigator& probe_navigator() {
  static thread_local std::unique_ptr<G4Navigator> nav;
  if (!nav) {
    nav = std::make_unique<G4Navigator>();
    nav->SetWorldVolume(shared().geo.world);
  }
  return *nav;
}

int region_index(const SolidsSpec& spec, const std::string& name) {
  for (std::size_t r = 0; r < spec.regions.size(); ++r)
    if (spec.regions[r] == name) return static_cast<int>(r);
  return -1;
}

// Loads the maps the job's fields name and assigns each region its field. Throws std::runtime_error.
void prepare_fields(Shared& S) {
  const auto& fields = S.job->transport.fields;
  S.field_of_region.assign(S.spec.regions.size(), -1);
  S.patch_of_field.assign(fields.size(), nullptr);
  std::map<std::pair<std::string, std::string>, std::shared_ptr<const MapPatch>> loaded;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const FieldSpec& f = fields[i];
    if (f.type == "map") {
      const auto key = std::make_pair(f.path.string(), f.patch);
      if (!loaded.count(key)) loaded[key] = load_patch(f.path, f.patch);
      S.patch_of_field[i] = loaded[key];
    }
    for (const auto& r : f.regions) {
      const int k = region_index(S.spec, r);
      if (k < 0) throw std::runtime_error("fields[" + std::to_string(i) + "]: '" + r + "' is not a region of the solids");
      if (S.field_of_region[k] >= 0)
        throw std::runtime_error("region '" + r + "' is served by fields[" + std::to_string(S.field_of_region[k]) +
                                 "] and fields[" + std::to_string(i) + "]");
      S.field_of_region[k] = static_cast<int>(i);
    }
  }
}

G4MagneticField* make_field(const Shared& S, const BuiltVolume& v) {
  if (v.region < 0 || S.field_of_region[v.region] < 0) return nullptr;
  const int i = S.field_of_region[v.region];
  const FieldSpec& f = S.job->transport.fields[i];
  if (f.type == "uniform")
    return new CountingUniformField(G4ThreeVector(f.B_T[0] * tesla, f.B_T[1] * tesla, f.B_T[2] * tesla));
  const G4ThreeVector offset(v.offset_m[0] * m, v.offset_m[1] * m, v.offset_m[2] * m);
  return new VolumeField(f, S.patch_of_field[i], v.solid, offset, S.job->transport.outside, v.spec->name);
}

// ============================================================================================ Geant4 user classes
class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    Shared& S = shared();
    S.geo = build_geometry(S.spec);
    for (const BuiltVolume& v : S.geo.volumes) S.region_of[v.logical] = v.region;
    return S.geo.world;
  }

  // Every logical volume gets a field manager of its own (a field-free one where no field serves its region), so
  // that no volume inherits its mother's field, and every field manager the same pinned integrator.
  void ConstructSDandField() override {
    const Shared& S = shared();
    const IntegratorSpec& in = S.job->transport.integrator;
    for (const BuiltVolume& v : S.geo.volumes) {
      G4MagneticField* field = make_field(S, v);
      G4FieldManager* manager = nullptr;
      // Geant4's field-manager store owns every G4FieldManager and deletes it at the end of the process; the fields
      // and chord finders live as long as the process
      if (!field) {
        manager = new G4FieldManager();
      } else {
        manager = new G4FieldManager(field);
        auto* chord = new G4ChordFinder(field, in.min_step_m * m, nullptr, G4ChordFinder::kTemplatedStepperType);
        using Driver = G4IntegrationDriver<G4TDormandPrince45<G4Mag_UsualEqRhs, 6>>;
        if (!dynamic_cast<Driver*>(chord->GetIntegrationDriver())) {
          std::cerr << "axmp-transport: error: Geant4 did not build the integration driver " << kStepper << std::endl;
          std::_Exit(kRefused);
        }
        chord->SetDeltaChord(in.delta_chord_m * m);
        manager->SetChordFinder(chord);
        manager->SetDeltaOneStep(in.delta_one_step_m * m);
        manager->SetDeltaIntersection(in.delta_intersection_m * m);
        // the order keeps eps_min <= eps_max at every step (Geant4 refuses the other)
        if (in.epsilon_max >= manager->GetMinimumEpsilonStep()) {
          manager->SetMaximumEpsilonStep(in.epsilon_max);
          manager->SetMinimumEpsilonStep(in.epsilon_min);
        } else {
          manager->SetMinimumEpsilonStep(in.epsilon_min);
          manager->SetMaximumEpsilonStep(in.epsilon_max);
        }
        if (manager->GetMinimumEpsilonStep() != in.epsilon_min || manager->GetMaximumEpsilonStep() != in.epsilon_max) {
          std::cerr << "axmp-transport: error: Geant4 refuses the epsilons [" << in.epsilon_min << ", "
                    << in.epsilon_max << "]" << std::endl;
          std::_Exit(kRefused);
        }
      }
      v.logical->SetFieldManager(manager, false);
    }
  }
};

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction {
 public:
  PrimaryGenerator() : gun_(1) {
    const Shared& S = shared();
    gun_.SetParticleDefinition(const_cast<G4ParticleDefinition*>(S.particle));
    gun_.SetParticleEnergy(S.ekin_eV * eV);
    gun_.SetParticleTime(0.0);
  }

  void GeneratePrimaries(G4Event* event) override {
    // Seeding comes first: the words of (chunk seed, event index) are installed before anything is drawn. The array
    // is a member because MixMaxRng::setSeeds keeps the pointer it is given.
    const auto index = static_cast<std::uint64_t>(event->GetEventID());
    Shared& S = shared();
    event_seed_words(S.chunk_seed.load(std::memory_order_relaxed), index, words_);
    G4Random::setTheSeeds(words_, 4);
    const double u0 = G4UniformRand();
    Rows& rows = *S.rows;
    if (index >= rows.n)
      G4Exception("PrimaryGenerator::GeneratePrimaries", "axtr0002", FatalException,
                  "an event index beyond the events of the chunk");
    const SourceSpec& src = S.job->transport.source;
    double pos[3] = {src.at_m[0], src.at_m[1], src.at_m[2]};
    double d[3] = {src.direction[0], src.direction[1], src.direction[2]};
    if (src.isotropic) {
      const double u1 = G4UniformRand(), u2 = G4UniformRand();
      const double c = 1.0 - 2.0 * u1;
      const double phi = CLHEP::twopi * u2;
      const double s = std::sqrt(std::max(0.0, 1.0 - c * c));
      d[0] = s * std::cos(phi);
      d[1] = s * std::sin(phi);
      d[2] = c;
    } else if (src.spot_sigma_m > 0.0) {
      const double u1 = G4UniformRand(), u2 = G4UniformRand();
      const double rho = src.spot_sigma_m * std::sqrt(-2.0 * std::log(1.0 - u1));
      const double psi = CLHEP::twopi * u2;
      // e1 = unit(d x z) (x when d is along z), e2 = d x e1
      double e1[3] = {d[1], -d[0], 0.0};
      const double n1 = std::hypot(e1[0], e1[1]);
      if (n1 > 0.0) {
        e1[0] /= n1;
        e1[1] /= n1;
      } else {
        e1[0] = 1.0;
        e1[1] = 0.0;
      }
      const double e2[3] = {d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0]};
      for (int k = 0; k < 3; ++k) pos[k] += rho * (std::cos(psi) * e1[k] + std::sin(psi) * e2[k]);
    }
    for (int k = 0; k < 4; ++k) rows.seed_words[4 * index + k] = static_cast<std::uint32_t>(words_[k]);
    rows.first_flat[index] = u0;
    double* prim = &rows.primary[7 * index];
    for (int k = 0; k < 3; ++k) {
      prim[k] = pos[k];
      prim[3 + k] = S.p_eV_c * d[k];
    }
    prim[6] = S.ekin_eV;
    const G4ThreeVector vertex(pos[0] * m, pos[1] * m, pos[2] * m);
    if (S.geo.world->GetLogicalVolume()->GetSolid()->Inside(vertex) != kInside) {
      std::ostringstream msg;
      msg << "the primary vertex (" << pos[0] << ", " << pos[1] << ", " << pos[2]
          << ") m drawn from the spot lies outside the world: the spot is wider than the solids list";
      G4Exception("PrimaryGenerator::GeneratePrimaries", "axtr0003", FatalException, msg.str().c_str());
    }
    gun_.SetParticlePosition(vertex);
    gun_.SetParticleMomentumDirection(G4ThreeVector(d[0], d[1], d[2]));
    gun_.GeneratePrimaryVertex(event);
  }

 private:
  G4ParticleGun gun_;
  long words_[4] = {0, 0, 0, 0};
};

class EventAction : public G4UserEventAction {
 public:
  void BeginOfEventAction(const G4Event*) override {
    const Shared& S = shared();
    EventState& e = t_ev;
    e.edep.assign(S.spec.regions.size(), 0.0);
    e.exit_eV = e.looping_eV = e.abandoned_eV = 0.0;
    e.steps = e.tracks = e.looping = e.abandoned = 0;
    take_abandoned_track();
    const std::size_t ns = S.job->transport.hits.size();
    e.hits.resize(ns);
    for (auto& h : e.hits) h.clear();
    e.dropped.assign(ns, 0);
    e.tally.resize(S.job->transport.tallies.size());
    for (auto& t : e.tally) t.clear();
    field_counters() = FieldCounters{};
    Shared& W = shared();
    if (W.first_event_s.load(std::memory_order_relaxed) < 0.0) {
      double expected = -1.0;
      W.first_event_s.compare_exchange_strong(expected, seconds(process_start(), Clock::now()));
    }
  }

  void EndOfEventAction(const G4Event* event) override {
    Shared& S = shared();
    Rows& rows = *S.rows;
    EventState& e = t_ev;
    const auto i = static_cast<std::uint64_t>(event->GetEventID());
    rows.event[i] = i;
    for (std::size_t r = 0; r < rows.R; ++r) rows.edep[rows.R * i + r] = e.edep[r];
    rows.exit_eV[i] = e.exit_eV;
    rows.n_steps[i] = e.steps;
    rows.n_tracks[i] = e.tracks;
    rows.field_calls[i] = field_counters().calls;
    rows.outside_calls[i] = field_counters().outside;
    rows.looping_killed[i] = e.looping;
    rows.looping_eV[i] = e.looping_eV;
    rows.abandoned_killed[i] = e.abandoned;
    rows.abandoned_eV[i] = e.abandoned_eV;
    for (std::size_t s = 0; s < e.hits.size(); ++s) {
      rows.hits[s][i] = e.hits[s];
      rows.hits_dropped[s][i] = e.dropped[s];
    }
    // each bin's event total rounded to the quantum, then integer-added: the sum does not depend on the order
    const double q = S.job->transport.quantum_eV;
    if (!e.tally.empty()) {
      std::lock_guard<std::mutex> lock(rows.tally_mutex);
      for (std::size_t t = 0; t < e.tally.size(); ++t)
        for (const auto& [bin, eV] : e.tally[t]) rows.tally[t][bin] += static_cast<std::int64_t>(std::llround(eV / q));
    }
  }
};

class TrackingAction : public G4UserTrackingAction {
 public:
  void PreUserTrackingAction(const G4Track* track) override {
    ++t_ev.tracks;
    const G4ParticleDefinition* def = track->GetDefinition();
    auto it = transportation_.find(def);
    if (it == transportation_.end()) {
      const G4Transportation* found = nullptr;
      if (const G4ProcessManager* pm = def->GetProcessManager()) {
        const G4ProcessVector* list = pm->GetProcessList();
        for (std::size_t k = 0; list && k < list->size(); ++k)
          if (auto* tr = dynamic_cast<const G4Transportation*>((*list)[k])) found = tr;
      }
      it = transportation_.emplace(def, found).first;
    }
    t_ev.transportation = it->second;
    t_ev.killed_at_start = it->second ? it->second->GetSumEnergyKilled() : 0.0;
  }

  // Geant4's transportation kills a track that loops in a field without progress and adds its kinetic energy to its
  // running sum: a rise of that sum during the track is a looping track killed.
  void PostUserTrackingAction(const G4Track*) override {
    if (!t_ev.transportation) return;
    const double killed = t_ev.transportation->GetSumEnergyKilled() - t_ev.killed_at_start;
    if (killed > 0.0) {
      ++t_ev.looping;
      t_ev.looping_eV += killed / eV;
    }
  }

 private:
  std::unordered_map<const G4ParticleDefinition*, const G4Transportation*> transportation_;
};

std::size_t tally_bin(const TallySpec& t, const G4ThreeVector& mid_mm, bool& inside) {
  double c[3];
  if (t.cylindrical) {
    c[0] = std::hypot(mid_mm.x(), mid_mm.y()) / m;
    c[1] = mid_mm.z() / m;
  } else {
    c[0] = mid_mm.x() / m;
    c[1] = mid_mm.y() / m;
    c[2] = mid_mm.z() / m;
  }
  std::size_t flat = 0;
  inside = true;
  for (int a = 0; a < t.axes; ++a) {
    if (!(c[a] >= t.lo[a] && c[a] <= t.hi[a])) {
      inside = false;
      return 0;
    }
    long long i = static_cast<long long>(std::floor((c[a] - t.lo[a]) / (t.hi[a] - t.lo[a]) * t.bins[a]));
    i = std::min<long long>(std::max<long long>(i, 0), t.bins[a] - 1);
    flat = flat * static_cast<std::size_t>(t.bins[a]) + static_cast<std::size_t>(i);
  }
  return flat;
}

class SteppingAction : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const Shared& S = shared();
    EventState& e = t_ev;
    ++e.steps;
    const G4StepPoint* pre = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();
    G4Track* track = step->GetTrack();
    const G4VPhysicalVolume* pv = pre->GetPhysicalVolume();
    const int region = pv ? S.region_of.at(pv->GetLogicalVolume()) : -1;
    const double length = step->GetStepLength();
    // After a zero-length step back across a face it has just crossed, Geant4 can place a track in the mother volume
    // (the world) for one step, then return it to the volume it was in: a step that follows a step of the same track
    // no longer than the surface tolerance enters nothing.
    const double tolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
    const bool after_zero_length = track->GetCurrentStepNumber() > 1 && e.prev_track == track->GetTrackID() &&
                                   e.prev_len <= tolerance;
    e.prev_track = track->GetTrackID();
    e.prev_len = length;
    if (region >= 0) {
      const double edep = step->GetTotalEnergyDeposit() / eV;
      e.edep[region] += edep;
      if (edep != 0.0)
        for (int t : S.tallies_of_region[region]) {
          bool inside = false;
          const std::size_t bin =
              tally_bin(S.job->transport.tallies[t], 0.5 * (pre->GetPosition() + post->GetPosition()), inside);
          if (inside) e.tally[t][bin] += edep;
        }
      // a hit: the step enters a volume of the scorer's regions (its pre-step point lies on the boundary)
      if (pre->GetStepStatus() == fGeomBoundary && !after_zero_length)
        for (int s : S.scorers_of_region[region]) {
          const HitsSpec& h = S.job->transport.hits[s];
          if (h.primary_only && track->GetParentID() != 0) continue;
          if (!h.all_particles && !S.scorer_particles[s].count(track->GetDefinition()->GetParticleName())) continue;
          if (e.hits[s].size() >= h.max_per_event) {
            ++e.dropped[s];
            continue;
          }
          HitRecord rec;
          rec.track = track->GetTrackID();
          rec.parent = track->GetParentID();
          rec.pdg = track->GetDefinition()->GetPDGEncoding();
          rec.region = region;
          const G4ThreeVector x = pre->GetPosition(), p = pre->GetMomentum();
          rec.x[0] = x.x() / m, rec.x[1] = x.y() / m, rec.x[2] = x.z() / m;
          rec.p[0] = p.x() / eV, rec.p[1] = p.y() / eV, rec.p[2] = p.z() / eV;
          rec.t_s = pre->GetGlobalTime() / CLHEP::second;
          rec.ekin_eV = pre->GetKineticEnergy() / eV;
          e.hits[s].push_back(rec);
        }
    }
    // The world and the voids end the transport's domain. A track leaves it when it leaves the world, or at the end
    // of a step longer than the surface tolerance that started in the world or a void and whose first micrometre (at
    // most half the step) lies in the world or a void by a locate without history. A step that merely ends in the world is no
    // exit: at an interface between two regions Geant4 can report the world for one zero-length step.
    // a track Geant4's navigator abandoned as stuck during this step (GeomNav0003) ends here, counted with its energy
    if (take_abandoned_track()) {
      ++e.abandoned;
      e.abandoned_eV += post->GetKineticEnergy() / eV;
      track->SetTrackStatus(fStopAndKill);
      return;
    }
    if (post->GetStepStatus() == fWorldBoundary) {
      e.exit_eV += post->GetKineticEnergy() / eV;
      return;
    }
    const G4TrackStatus status = track->GetTrackStatus();
    if (region < 0 && length > tolerance && (status == fAlive || status == fStopButAlive)) {
      const G4ThreeVector probe =
          pre->GetPosition() + std::min(0.5 * length, 1.0 * um) * pre->GetMomentumDirection();
      const G4VPhysicalVolume* at = probe_navigator().LocateGlobalPointAndSetup(probe, nullptr, false, true);
      if (!at || S.region_of.at(at->GetLogicalVolume()) < 0) {
        e.exit_eV += post->GetKineticEnergy() / eV;
        track->SetTrackStatus(fStopAndKill);
      }
    }
  }
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

// ============================================================================================ outputs
json integrator_json(const IntegratorSpec& in) {
  return {{"delta_chord_m", in.delta_chord_m},       {"delta_one_step_m", in.delta_one_step_m},
          {"delta_intersection_m", in.delta_intersection_m}, {"epsilon_min", in.epsilon_min},
          {"epsilon_max", in.epsilon_max},           {"min_step_m", in.min_step_m},
          {"stepper", kStepper}};
}

void write_chunk(const Job& job, const ChunkSpec& chunk, const Rows& a, int threads, double run_s,
                 const std::vector<std::string>& regions) {
  const TransportSpec& tr = job.transport;
  // each writer has a temporary directory of its own (an orphaned program of a killed driver may write the same
  // chunk); the complete chunk is renamed into place
  const std::string stem = "chunk_" + std::to_string(chunk.index);
  const std::string pid = std::to_string(static_cast<long long>(::getpid()));
  const fs::path final_dir = job.out / stem;
  const fs::path tmp_dir = job.out / (stem + "." + pid + ".tmp");
  fs::remove_all(tmp_dir);
  fs::create_directories(tmp_dir);
  const std::size_t n = a.n;
  json arrays = json::object();
  auto put = [&](const std::string& name, const char* descr, std::vector<std::size_t> shape, const void* data,
                 std::size_t bytes) {
    const std::string file = name + ".npy";
    const std::string sha = write_npy(tmp_dir / file, descr, shape, data, bytes);
    arrays[name] = {{"file", file}, {"dtype", descr}, {"shape", shape}, {"sha256", sha}};
  };
  put("event", "<u8", {n}, a.event.data(), 8 * n);
  put("seed_words", "<u4", {n, 4}, a.seed_words.data(), 16 * n);
  put("first_flat", "<f8", {n}, a.first_flat.data(), 8 * n);
  put("primary", "<f8", {n, 7}, a.primary.data(), 56 * n);
  put("edep_eV", "<f8", {n, a.R}, a.edep.data(), 8 * n * a.R);
  put("exit_eV", "<f8", {n}, a.exit_eV.data(), 8 * n);
  put("n_steps", "<u8", {n}, a.n_steps.data(), 8 * n);
  put("n_tracks", "<u8", {n}, a.n_tracks.data(), 8 * n);
  put("field_calls", "<u8", {n}, a.field_calls.data(), 8 * n);
  put("outside_calls", "<u8", {n}, a.outside_calls.data(), 8 * n);
  put("looping_killed", "<u8", {n}, a.looping_killed.data(), 8 * n);
  put("looping_eV", "<f8", {n}, a.looping_eV.data(), 8 * n);
  put("abandoned_killed", "<u8", {n}, a.abandoned_killed.data(), 8 * n);
  put("abandoned_eV", "<f8", {n}, a.abandoned_eV.data(), 8 * n);
  std::uint64_t dropped_total = 0;
  for (std::size_t s = 0; s < tr.hits.size(); ++s) {
    const std::string h = "hits_" + tr.hits[s].name + "_";
    std::vector<std::uint32_t> count(n), dropped(a.hits_dropped[s]);
    std::vector<std::uint64_t> ev;
    std::vector<std::int32_t> track, parent, pdg, region;
    std::vector<double> x, p, t, ekin;
    for (std::size_t i = 0; i < n; ++i) {
      count[i] = static_cast<std::uint32_t>(a.hits[s][i].size());
      dropped_total += dropped[i];
      for (const HitRecord& r : a.hits[s][i]) {
        ev.push_back(i);
        track.push_back(r.track);
        parent.push_back(r.parent);
        pdg.push_back(r.pdg);
        region.push_back(r.region);
        x.insert(x.end(), r.x, r.x + 3);
        p.insert(p.end(), r.p, r.p + 3);
        t.push_back(r.t_s);
        ekin.push_back(r.ekin_eV);
      }
    }
    const std::size_t mrec = ev.size();
    put(h + "count", "<u4", {n}, count.data(), 4 * n);
    put(h + "dropped", "<u4", {n}, dropped.data(), 4 * n);
    put(h + "event", "<u8", {mrec}, ev.data(), 8 * mrec);
    put(h + "track", "<i4", {mrec}, track.data(), 4 * mrec);
    put(h + "parent", "<i4", {mrec}, parent.data(), 4 * mrec);
    put(h + "pdg", "<i4", {mrec}, pdg.data(), 4 * mrec);
    put(h + "region", "<i4", {mrec}, region.data(), 4 * mrec);
    put(h + "x_m", "<f8", {mrec, 3}, x.data(), 24 * mrec);
    put(h + "p_eV_c", "<f8", {mrec, 3}, p.data(), 24 * mrec);
    put(h + "t_s", "<f8", {mrec}, t.data(), 8 * mrec);
    put(h + "ekin_eV", "<f8", {mrec}, ekin.data(), 8 * mrec);
  }
  for (std::size_t k = 0; k < tr.tallies.size(); ++k) {
    const TallySpec& g = tr.tallies[k];
    std::vector<std::size_t> shape;
    for (int ax = 0; ax < g.axes; ++ax) shape.push_back(static_cast<std::size_t>(g.bins[ax]));
    put("tally_" + g.name, "<i8", shape, a.tally[k].data(), 8 * a.tally[k].size());
  }
  std::uint64_t fc = 0, oc = 0, lk = 0, ak = 0;
  double le = 0.0, ae = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    fc += a.field_calls[i];
    oc += a.outside_calls[i];
    lk += a.looping_killed[i];
    le += a.looping_eV[i];
    ak += a.abandoned_killed[i];
    ae += a.abandoned_eV[i];
  }
  json manifest = {{"schema", "axionmultiphysics.transport/1"},
                   {"mode", "transport"},
                   {"chunk", chunk.index},
                   {"chunk_seed", chunk.seed},
                   {"events", chunk.events},
                   {"threads", threads},
                   {"geant4_version", geant4_version()},
                   {"program_version", AXMP_TRANSPORT_VERSION},
                   {"physics_list", job.physics_list},
                   {"production_cut_m", tr.production_cut_m},
                   {"integrator", integrator_json(tr.integrator)},
                   {"quantum_eV", tr.quantum_eV},
                   {"outside", tr.outside == Outside::zero ? "zero" : "refuse"},
                   {"regions", regions},
                   {"arrays", arrays},
                   {"counters",
                    {{"field_calls", fc},
                     {"outside_calls", oc},
                     {"looping_killed", lk},
                     {"looping_eV", le},
                     {"abandoned_killed", ak},
                     {"abandoned_eV", ae},
                     {"hits_dropped", dropped_total}}},
                   {"timing", {{"run_s", run_s}}}};
  write_text(tmp_dir / "manifest.json", manifest.dump(1) + "\n");
  std::error_code ec;
  fs::rename(tmp_dir, final_dir, ec);
  if (ec) {
    // a chunk directory is there (an earlier run's, or another writer's): moved aside, then replaced
    const fs::path old = job.out / (stem + "." + pid + ".old");
    std::error_code ignore;
    fs::remove_all(old, ignore);
    fs::rename(final_dir, old, ignore);
    fs::rename(tmp_dir, final_dir, ec);
    fs::remove_all(old, ignore);
    if (ec) {
      // another writer placed the same chunk in between: its arrays are this chunk's (they depend on its seed only)
      fs::remove_all(tmp_dir, ignore);
      if (!fs::exists(final_dir / "manifest.json")) throw std::runtime_error("cannot rename " + tmp_dir.string());
    }
  }
}

// Reads the solids, resolves the source's particle (after the physics list filled the particle table), and loads the
// maps. Returns 0 or an exit status.
int load_inputs(Shared& S, const Job& job) {
  S.job = &job;
  try {
    S.spec = load_solids(job.transport.solids);
  } catch (const GeometryError& e) {
    return error(kUsage, "invalid solids file " + job.transport.solids.string() + ": " + e.what());
  }
  try {
    prepare_fields(S);
  } catch (const std::exception& e) {
    return error(kUsage, e.what());
  }
  if (job.mode == "transport") {
    // a patch is right about one cell beyond its region; the boundary crossings read the field within
    // deltaIntersection of the boundary, so it may be at most a tenth of the finest attached spacing
    double finest = 0.0;
    for (const auto& patch : S.patch_of_field)
      if (patch)
        for (int a = 0; a < patch->dims; ++a)
          finest = finest > 0.0 ? std::min(finest, patch->spacing[a]) : patch->spacing[a];
    const double dint = job.transport.integrator.delta_intersection_m;
    if (finest > 0.0 && dint > 0.1 * finest * (1.0 + 1e-12)) {
      std::ostringstream msg;
      msg << "integrator.delta_intersection_m " << dint << " m exceeds 0.1 of the finest spacing of the attached "
          << "map patches (" << finest << " m)";
      return error(kUsage, msg.str());
    }
  }
  const auto& tr = job.transport;
  const std::size_t R = S.spec.regions.size();
  S.scorers_of_region.assign(R, {});
  S.tallies_of_region.assign(R, {});
  S.scorer_particles.assign(tr.hits.size(), {});
  for (std::size_t s = 0; s < tr.hits.size(); ++s) {
    for (const auto& r : tr.hits[s].regions) {
      const int k = region_index(S.spec, r);
      if (k < 0) return error(kUsage, "hits '" + tr.hits[s].name + "': '" + r + "' is not a region of the solids");
      S.scorers_of_region[k].push_back(static_cast<int>(s));
    }
    S.scorer_particles[s].insert(tr.hits[s].particles.begin(), tr.hits[s].particles.end());
  }
  for (std::size_t t = 0; t < tr.tallies.size(); ++t)
    for (const auto& r : tr.tallies[t].regions) {
      const int k = region_index(S.spec, r);
      if (k < 0) return error(kUsage, "tally '" + tr.tallies[t].name + "': '" + r + "' is not a region of the solids");
      S.tallies_of_region[k].push_back(static_cast<int>(t));
    }
  return kOk;
}

}  // namespace

// ============================================================================================ transport
int run_transport(const Job& job, int threads) {
#ifndef G4MULTITHREADED
  return error(kRefused, "this Geant4 was built without multithreading (GEANT4_BUILD_MULTITHREADED=OFF)");
#else
  for (const auto& d : dataset_status(job.physics_list)) {
    if (d.required && !d.ok)
      return error(kRefused, "the Geant4 data set " + d.name + " is required by " + job.physics_list + " but " + d.env +
                                 (d.path.empty() ? " is not set" : " does not name a directory (" + d.path + ")") +
                                 "; source geant4.sh of the installation or set the variable");
  }
  Shared& S = shared();
  if (const int rc = load_inputs(S, job)) return rc;
  const TransportSpec& tr = job.transport;

  install_exception_handler();
  G4Random::setTheEngine(new CLHEP::MixMaxRng);
  std::unique_ptr<G4RunManager> manager(
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::TaskingOnly, nullptr, true, threads));
  auto* tasking = dynamic_cast<G4TaskRunManager*>(manager.get());
  if (!tasking) return error(kRefused, "Geant4 did not provide the tasking run manager");
  if (job.debug_seed_once_per_communication)
    G4MTRunManager::SetSeedOncePerCommunication(*job.debug_seed_once_per_communication);
  const int seed_mode = G4MTRunManager::SeedOncePerCommunication();
  if (seed_mode != 0)
    return error(kRefused, "G4MTRunManager::SeedOncePerCommunication() is " + std::to_string(seed_mode) +
                               ", not 0 (SeedOncePerCommunication must be 0)");

  G4PhysListFactory factory;
  factory.SetVerbose(0);
  G4VModularPhysicsList* physics = factory.GetReferencePhysList(job.physics_list);
  if (!physics) return error(kUsage, "unknown physics list '" + job.physics_list + "'");
  physics->SetVerboseLevel(0);
  physics->SetDefaultCutValue(tr.production_cut_m * m);
  manager->SetVerboseLevel(0);
  manager->SetUserInitialization(new Detector);
  manager->SetUserInitialization(physics);
  // the particle table is filled now; the workers' primary generators need the particle
  auto* table = G4ParticleTable::GetParticleTable();
  const SourceSpec& src = tr.source;
  S.particle = src.particle.empty() ? table->FindParticle(src.pdg) : table->FindParticle(src.particle);
  if (!S.particle)
    return error(kUsage, src.particle.empty() ? "no particle of Geant4's table has the PDG code " + std::to_string(src.pdg)
                                              : "unknown particle '" + src.particle + "' (source.particle)");
  S.mass_eV = S.particle->GetPDGMass() / eV;
  if (src.by_momentum) {
    const double p = src.value_eV;
    S.p_eV_c = p;
    S.ekin_eV = p * p / (std::sqrt(p * p + S.mass_eV * S.mass_eV) + S.mass_eV);
  } else {
    S.ekin_eV = src.value_eV;
    S.p_eV_c = std::sqrt(S.ekin_eV * (S.ekin_eV + 2.0 * S.mass_eV));
  }
  manager->SetUserInitialization(new Actions);
  try {
    manager->Initialize();
  } catch (const GeometryError& e) {
    return error(kUsage, e.what());
  }
  {
    // a primary vertex outside the world has no volume to start in: refused before anything is written
    const G4ThreeVector at(src.at_m[0] * m, src.at_m[1] * m, src.at_m[2] * m);
    if (S.geo.world->GetLogicalVolume()->GetSolid()->Inside(at) != kInside) {
      std::ostringstream msg;
      msg << "source.at_m (" << src.at_m[0] << ", " << src.at_m[1] << ", " << src.at_m[2]
          << ") m lies outside the world of the solids list (or on its boundary): the primary has no volume to "
             "start in";
      return error(kUsage, msg.str());
    }
  }
  const int actual_threads = tasking->GetNumberOfThreads();
  try {
    fs::create_directories(job.out);
    fs::create_directories(job.record.parent_path());
  } catch (const std::exception& e) {
    return error(kRunFailed, std::string("cannot create the output directory: ") + e.what());
  }

  Rows rows;
  S.rows = &rows;
  std::uint64_t field_calls_total = 0, outside_total = 0, looping_total = 0, abandoned_total = 0;
  std::vector<std::uint64_t> done;
  for (const ChunkSpec& chunk : job.chunks) {
    if (std::find(tr.fail_chunks.begin(), tr.fail_chunks.end(), chunk.index) != tr.fail_chunks.end())
      return error(kRunFailed, "chunk " + std::to_string(chunk.index) + " fails (debug.fail_chunks)");
    rows.reset(chunk.events, S.spec.regions.size(), tr.hits.size(), tr.tallies);
    S.chunk_seed.store(chunk.seed);
    const auto t0 = Clock::now();
    manager->BeamOn(static_cast<G4int>(chunk.events));
    const double run_s = seconds(t0, Clock::now());
    for (std::uint64_t i = 0; i < chunk.events; ++i)
      if (rows.event[i] != i)
        return error(kRunFailed, "chunk " + std::to_string(chunk.index) + ": event " + std::to_string(i) +
                                     " was not transported");
    for (std::uint64_t i = 0; i < chunk.events; ++i) {
      field_calls_total += rows.field_calls[i];
      outside_total += rows.outside_calls[i];
      looping_total += rows.looping_killed[i];
      abandoned_total += rows.abandoned_killed[i];
    }
    try {
      write_chunk(job, chunk, rows, actual_threads, run_s, S.spec.regions);
    } catch (const std::exception& e) {
      return error(kRunFailed, std::string("cannot write chunk ") + std::to_string(chunk.index) + ": " + e.what());
    }
    done.push_back(chunk.index);
  }
  json record = {{"schema", "axionmultiphysics.transport-record/1"},
                 {"status", "ok"},
                 {"mode", "transport"},
                 {"threads", actual_threads},
                 {"chunks", done},
                 {"seed_once_per_communication", seed_mode},
                 {"geant4_version", geant4_version()},
                 {"field_calls_total", field_calls_total},
                 {"outside_calls_total", outside_total},
                 {"looping_killed_total", looping_total},
                 {"abandoned_killed_total", abandoned_total},
                 {"timing", {{"startup_s", S.first_event_s.load()}, {"total_s", seconds(process_start(), Clock::now())}}}};
  try {
    write_text(job.record, record.dump(1) + "\n");
  } catch (const std::exception& e) {
    return error(kRunFailed, e.what());
  }
  S.rows = nullptr;
  manager.reset();
  return kOk;
#endif
}

// ============================================================================================ probe
int run_probe(const Job& job) {
  Shared& S = shared();
  if (const int rc = load_inputs(S, job)) return rc;
  NpyDoubles pts;
  try {
    pts = read_npy_doubles(job.transport.probe_points);
  } catch (const std::exception& e) {
    return error(kUsage, e.what());
  }
  if (pts.shape.size() != 2 || pts.shape[1] != 3) return error(kUsage, "the probe points must have shape (n, 3)");
  try {
    S.geo = build_geometry(S.spec);
  } catch (const GeometryError& e) {
    return error(kUsage, e.what());
  }
  std::unordered_map<const G4VPhysicalVolume*, std::pair<int, G4MagneticField*>> field_of;
  for (const BuiltVolume& v : S.geo.volumes) field_of[v.physical] = {v.region, make_field(S, v)};
  G4Navigator nav;
  nav.SetWorldVolume(S.geo.world);
  const std::size_t n = pts.shape[0];
  std::vector<double> B(3 * n, 0.0);
  std::vector<std::int32_t> region(n, -1);
  for (std::size_t i = 0; i < n; ++i) {
    const double* p = &pts.data[3 * i];
    const G4ThreeVector x(p[0] * m, p[1] * m, p[2] * m);
    const G4VPhysicalVolume* pv = nav.LocateGlobalPointAndSetup(x, nullptr, false, true);
    if (!pv) continue;
    const auto& [r, field] = field_of.at(pv);
    region[i] = r;
    if (r < 0 || !field) continue;
    double b[3];
    if (auto* vf = dynamic_cast<VolumeField*>(field)) {
      vf->value(p, b, true);  // the point lies in the volume that holds it
    } else {
      const G4double point[4] = {x.x(), x.y(), x.z(), 0.0};
      G4double g[6] = {0, 0, 0, 0, 0, 0};
      field->GetFieldValue(point, g);
      for (int k = 0; k < 3; ++k) b[k] = g[k] / tesla;
    }
    for (int k = 0; k < 3; ++k) B[3 * i + k] = b[k];
  }
  const fs::path dir = job.out / "probe";
  try {
    fs::create_directories(dir);
    json arrays = json::object();
    arrays["B_T"] = {{"file", "B_T.npy"}, {"sha256", write_npy(dir / "B_T.npy", "<f8", {n, 3}, B.data(), 24 * n)}};
    arrays["region"] = {{"file", "region.npy"},
                        {"sha256", write_npy(dir / "region.npy", "<i4", {n}, region.data(), 4 * n)}};
    json man = {{"schema", "axionmultiphysics.transport-probe/1"},
                {"points", n},
                {"regions", S.spec.regions},
                {"outside", job.transport.outside == Outside::zero ? "zero" : "refuse"},
                {"outside_calls", field_counters().outside},
                {"arrays", arrays}};
    write_text(dir / "manifest.json", man.dump(1) + "\n");
  } catch (const std::exception& e) {
    return error(kRunFailed, e.what());
  }
  return kOk;
}

}  // namespace axtr
