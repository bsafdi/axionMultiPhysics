// SPDX-License-Identifier: BSD-3-Clause
// The job file of axmp-transport: parsed and validated strictly (an unknown or missing key, a wrong type or a value
// out of range is an error that names it) before Geant4 is touched.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace axtr {

struct JobError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ChunkSpec {
  std::uint64_t index = 0;
  std::uint64_t seed = 0;
  std::uint64_t events = 0;
};

struct HelloSpec {
  std::string particle = "mu-";
  double energy_lo_eV = 5.0e8;
  double energy_hi_eV = 1.5e9;
  double field_T[3] = {0.0, 0.0, 0.0};
  double target_thickness_m = 0.01;
  bool has_field() const { return field_T[0] != 0.0 || field_T[1] != 0.0 || field_T[2] != 0.0; }
};

// A field attached to the volumes of some regions (the job names every region explicitly).
struct FieldSpec {
  std::string type;  // uniform, current_loop, map
  double B_T[3] = {0.0, 0.0, 0.0};
  double radius_m = 0.0, current_A = 0.0, z_m = 0.0;
  std::filesystem::path path;  // map directory, absolute
  std::string patch;
  std::vector<std::string> regions;
};

// The accuracy parameters of the field integration, in m (the epsilons are relative).
struct IntegratorSpec {
  double delta_chord_m = 2.5e-4, delta_one_step_m = 1e-5, delta_intersection_m = 1e-6;
  double epsilon_min = 5e-5, epsilon_max = 1e-3, min_step_m = 1e-5;
};

struct SourceSpec {
  std::string particle;   // a name of Geant4's particle table, or empty when given by PDG code
  int pdg = 0;
  bool by_momentum = false;
  double value_eV = 0.0;  // kinetic energy (eV) or momentum (eV/c)
  double at_m[3] = {0.0, 0.0, 0.0};
  bool isotropic = false;
  double direction[3] = {0.0, 0.0, 1.0};  // normalised
  double spot_sigma_m = 0.0;
};

struct TallySpec {
  std::string name;
  bool cylindrical = true;
  int axes = 2;                     // 2 (r, z) or 3 (x, y, z)
  double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
  int bins[3] = {1, 1, 1};
  std::vector<std::string> regions;
  std::size_t size() const { return static_cast<std::size_t>(bins[0]) * bins[1] * (axes == 3 ? bins[2] : 1); }
};

struct HitsSpec {
  std::string name;
  std::vector<std::string> regions;
  bool primary_only = false;
  bool all_particles = true;
  std::vector<std::string> particles;
  std::uint32_t max_per_event = 100;
};

enum class Outside { refuse, zero };

struct TransportSpec {
  double production_cut_m = 7e-4;
  std::filesystem::path solids;  // absolute
  std::vector<FieldSpec> fields;
  Outside outside = Outside::refuse;
  IntegratorSpec integrator;
  SourceSpec source;
  double quantum_eV = 1e-3;
  std::vector<TallySpec> tallies;
  std::vector<HitsSpec> hits;
  std::vector<std::uint64_t> fail_chunks;  // debug: the process ends with status 4 on reaching one of these
  std::filesystem::path probe_points;      // probe mode: an (n, 3) float64 NPY file of points in m
};

struct Job {
  std::string mode;
  std::filesystem::path out;     // absolute
  std::filesystem::path record;  // absolute
  std::vector<ChunkSpec> chunks;
  std::string physics_list = "FTFP_BERT";
  HelloSpec hello;
  TransportSpec transport;       // modes transport and probe
  std::optional<int> debug_seed_once_per_communication;
};

inline constexpr const char* kJobSchema = "axionmultiphysics.transport-job/1";

// The physics lists and modes this program offers.
const std::vector<std::string>& offered_physics_lists();
const std::vector<std::string>& offered_modes();

// Reads and validates the job file. Relative paths in it resolve against the directory of the job file.
// Throws JobError.
Job load_job(const std::filesystem::path& file);

}  // namespace axtr
