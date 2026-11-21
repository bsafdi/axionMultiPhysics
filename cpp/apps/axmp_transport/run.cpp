// SPDX-License-Identifier: BSD-3-Clause
#include "run.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "G4MTRunManager.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4RunManagerFactory.hh"
#include "G4TaskRunManager.hh"
#include "G4Version.hh"
#include "G4VModularPhysicsList.hh"
#include "Randomize.hh"
#include "CLHEP/Random/MixMaxRng.h"

#include "datasets.hpp"
#include "hello.hpp"
#include "npy.hpp"
#include "seeding.hpp"

namespace axtr {

using nlohmann::json;
namespace fs = std::filesystem;

std::string geant4_version() {
  return std::to_string(G4VERSION_NUMBER / 100) + "." + std::to_string((G4VERSION_NUMBER / 10) % 10) + "." +
         std::to_string(G4VERSION_NUMBER % 10);
}

namespace {

int error(int code, const std::string& what) {
  std::cerr << "axmp-transport: error: " << what << std::endl;
  return code;
}

double seconds(Clock::time_point from, Clock::time_point to) { return std::chrono::duration<double>(to - from).count(); }

void write_text(const fs::path& file, const std::string& text) {
  std::ofstream f(file, std::ios::binary | std::ios::trunc);
  f << text;
  f.flush();
  if (!f) throw std::runtime_error("cannot write " + file.string());
}

json hello_json(const HelloSpec& h) {
  return {{"particle", h.particle},
          {"energy_eV", {h.energy_lo_eV, h.energy_hi_eV}},
          {"field_T", {h.field_T[0], h.field_T[1], h.field_T[2]}},
          {"target_thickness_m", h.target_thickness_m}};
}

// Writes chunk_<k>.tmp/ with one NPY file per array and the manifest, then renames it to chunk_<k>/ (replacing one).
void write_chunk(const Job& job, const ChunkSpec& chunk, const ChunkArrays& a, int threads, double run_s) {
  const fs::path final_dir = job.out / ("chunk_" + std::to_string(chunk.index));
  const fs::path tmp_dir = job.out / ("chunk_" + std::to_string(chunk.index) + ".tmp");
  fs::remove_all(tmp_dir);
  fs::create_directories(tmp_dir);
  const std::size_t n = a.events;
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
  put("ekin_eV", "<f8", {n}, a.ekin_eV.data(), 8 * n);
  put("edep_eV", "<f8", {n}, a.edep_eV.data(), 8 * n);
  put("n_steps", "<u8", {n}, a.n_steps.data(), 8 * n);
  put("n_tracks", "<u8", {n}, a.n_tracks.data(), 8 * n);
  put("field_calls", "<u8", {n}, a.field_calls.data(), 8 * n);
  json manifest = {{"schema", "axionmultiphysics.transport/1"},
                   {"mode", job.mode},
                   {"chunk", chunk.index},
                   {"chunk_seed", chunk.seed},
                   {"events", chunk.events},
                   {"arrays", arrays},
                   {"geant4_version", geant4_version()},
                   {"program_version", AXMP_TRANSPORT_VERSION},
                   {"threads", threads},
                   {"physics_list", job.physics_list},
                   {"hello", hello_json(job.hello)},
                   {"timing", {{"run_s", run_s}}}};
  write_text(tmp_dir / "manifest.json", manifest.dump(1) + "\n");
  fs::remove_all(final_dir);
  fs::rename(tmp_dir, final_dir);
}

}  // namespace

int run_job(const Job& job, int threads) {
#ifndef G4MULTITHREADED
  return error(kRefused, "this Geant4 was built without multithreading (GEANT4_BUILD_MULTITHREADED=OFF)");
#else
  // Data sets: refuse before anything starts when one the physics list reads is not there.
  for (const auto& d : dataset_status(job.physics_list)) {
    if (d.required && !d.ok)
      return error(kRefused, "the Geant4 data set " + d.name + " is required by " + job.physics_list + " but " + d.env +
                                 (d.path.empty() ? " is not set" : " does not name a directory (" + d.path + ")") +
                                 "; source geant4.sh of the installation or set the variable");
  }

  install_exception_handler();
  // Master engine: MixMax. Every worker engine is made from it, and every event is reseeded at its start.
  G4Random::setTheEngine(new CLHEP::MixMaxRng);

  std::unique_ptr<G4RunManager> manager(
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::TaskingOnly, nullptr, true, threads));
  auto* tasking = dynamic_cast<G4TaskRunManager*>(manager.get());
  if (!tasking) return error(kRefused, "Geant4 did not provide the tasking run manager");

  if (job.debug_seed_once_per_communication)
    G4MTRunManager::SetSeedOncePerCommunication(*job.debug_seed_once_per_communication);
  const int seed_mode = G4MTRunManager::SeedOncePerCommunication();
  if (seed_mode != 0) {
    return error(kRefused, "G4MTRunManager::SeedOncePerCommunication() is " + std::to_string(seed_mode) +
                               ", not 0: the events of a chunk would depend on the thread that ran them, and the "
                               "per-event seeds this program installs would be shared (SeedOncePerCommunication must be 0)");
  }

  SharedRun& shared = shared_run();
  shared.hello = job.hello;
  ChunkArrays arrays;
  shared.arrays = &arrays;

  G4PhysListFactory factory;
  factory.SetVerbose(0);
  G4VModularPhysicsList* physics = factory.GetReferencePhysList(job.physics_list);
  if (!physics) return error(kUsage, "unknown physics list '" + job.physics_list + "'");
  physics->SetVerboseLevel(0);
  manager->SetVerboseLevel(0);
  manager->SetUserInitialization(make_hello_detector(job.hello));
  manager->SetUserInitialization(physics);
  // The particle table is filled when the physics list is set. Initialize() starts the worker threads, whose primary
  // generators need the particle, so an unknown name is refused here, as an invalid job.
  if (!G4ParticleTable::GetParticleTable()->FindParticle(job.hello.particle))
    return error(kUsage, "unknown particle '" + job.hello.particle + "' (hello.particle)");
  manager->SetUserInitialization(make_hello_actions());
  manager->Initialize();
  const int actual_threads = tasking->GetNumberOfThreads();

  try {
    fs::create_directories(job.out);
    fs::create_directories(job.record.parent_path());
  } catch (const std::exception& e) {
    return error(kRunFailed, std::string("cannot create the output directory: ") + e.what());
  }

  std::uint64_t field_calls_total = 0;
  std::vector<std::uint64_t> done;
  for (const ChunkSpec& chunk : job.chunks) {
    arrays.reset(chunk.events);
    shared.chunk_seed.store(chunk.seed);
    const auto t0 = Clock::now();
    manager->BeamOn(static_cast<G4int>(chunk.events));
    const double run_s = seconds(t0, Clock::now());
    for (std::uint64_t i = 0; i < chunk.events; ++i)
      if (arrays.event[i] != i)
        return error(kRunFailed, "chunk " + std::to_string(chunk.index) + ": event " + std::to_string(i) +
                                     " was not transported");
    for (std::uint64_t v : arrays.field_calls) field_calls_total += v;
    try {
      write_chunk(job, chunk, arrays, actual_threads, run_s);
    } catch (const std::exception& e) {
      return error(kRunFailed, std::string("cannot write chunk ") + std::to_string(chunk.index) + ": " + e.what());
    }
    done.push_back(chunk.index);
  }

  const double startup = shared.first_event_s.load();
  const double total = seconds(process_start(), Clock::now());
  json record = {{"schema", "axionmultiphysics.transport-record/1"},
                 {"status", "ok"},
                 {"threads", actual_threads},
                 {"chunks", done},
                 {"seed_once_per_communication", seed_mode},
                 {"geant4_version", geant4_version()},
                 {"field_calls_total", field_calls_total},
                 {"timing", {{"startup_s", startup}, {"total_s", total}}}};
  try {
    write_text(job.record, record.dump(1) + "\n");
  } catch (const std::exception& e) {
    return error(kRunFailed, e.what());
  }
  shared.arrays = nullptr;
  manager.reset();
  return kOk;
#endif
}

}  // namespace axtr
