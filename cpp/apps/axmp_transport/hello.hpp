// SPDX-License-Identifier: BSD-3-Clause
// The hello mode: muons through a silicon slab in an optional uniform field. It exists to exercise the whole chain
// (job, seeding, tasking run manager, per-event records, outputs) with a physical setup small enough to check by hand.
#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "job.hpp"

class G4VUserActionInitialization;
class G4VUserDetectorConstruction;

namespace axtr {

// One row per event of a chunk, indexed by the event's index in the chunk. A worker thread writes only the rows of
// the events it runs, so the rows need no lock.
struct ChunkArrays {
  std::uint64_t events = 0;
  std::vector<std::uint64_t> event;
  std::vector<std::uint32_t> seed_words;  // events x 4
  std::vector<double> first_flat;
  std::vector<double> ekin_eV;
  std::vector<double> edep_eV;
  std::vector<std::uint64_t> n_steps;
  std::vector<std::uint64_t> n_tracks;
  std::vector<std::uint64_t> field_calls;

  void reset(std::uint64_t n);
};

// What the worker threads read: set by the master thread between two BeamOn calls, never during one.
struct SharedRun {
  HelloSpec hello;
  std::atomic<std::uint64_t> chunk_seed{0};
  ChunkArrays* arrays = nullptr;
  // Seconds from the start of the process to the first event's BeginOfEventAction, negative until it has begun.
  std::atomic<double> first_event_s{-1.0};
};
SharedRun& shared_run();

G4VUserDetectorConstruction* make_hello_detector(const HelloSpec& hello);
G4VUserActionInitialization* make_hello_actions();

}  // namespace axtr
