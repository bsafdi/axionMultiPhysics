// SPDX-License-Identifier: BSD-3-Clause
#include "job.hpp"

#include <cctype>
#include <cmath>
#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

namespace axtr {

using nlohmann::json;
namespace fs = std::filesystem;

const std::vector<std::string>& offered_physics_lists() {
  static const std::vector<std::string> v = {"FTFP_BERT", "FTFP_BERT_EMZ", "QGSP_BIC"};
  return v;
}

const std::vector<std::string>& offered_modes() {
  static const std::vector<std::string> v = {"hello", "transport", "probe"};
  return v;
}

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& what) {
  throw JobError((where.empty() ? std::string() : where + ": ") + what);
}

const json& object_at(const json& j, const std::string& where) {
  if (!j.is_object()) fail(where, "must be an object");
  return j;
}

void check_keys(const json& obj, const std::string& where, const std::set<std::string>& required,
                const std::set<std::string>& optional) {
  for (auto it = obj.begin(); it != obj.end(); ++it)
    if (!required.count(it.key()) && !optional.count(it.key()))
      fail(where, "unknown key '" + it.key() + "'");
  for (const auto& k : required)
    if (!obj.contains(k)) fail(where, "missing key '" + k + "'");
}

std::string string_at(const json& j, const std::string& where) {
  if (!j.is_string()) fail(where, "must be a string");
  return j.get<std::string>();
}

std::uint64_t unsigned_at(const json& j, const std::string& where, std::uint64_t lo, std::uint64_t hi) {
  if (!j.is_number_unsigned()) fail(where, "must be a non-negative integer (a JSON integer below 2^64)");
  const std::uint64_t v = j.get<std::uint64_t>();
  if (v < lo || v > hi) fail(where, "is out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
  return v;
}

double finite_at(const json& j, const std::string& where) {
  if (!j.is_number()) fail(where, "must be a number");
  const double v = j.get<double>();
  if (!std::isfinite(v)) fail(where, "must be finite");
  return v;
}

fs::path path_at(const json& j, const std::string& where, const fs::path& base) {
  const std::string s = string_at(j, where);
  if (s.empty()) fail(where, "must not be empty");
  const fs::path p(s);
  return (p.is_absolute() ? p : base / p).lexically_normal();
}

double positive_at(const json& j, const std::string& where) {
  const double v = finite_at(j, where);
  if (!(v > 0.0)) fail(where, "must be positive");
  return v;
}

void vector3_at(const json& j, const std::string& where, double out[3]) {
  if (!j.is_array() || j.size() != 3) fail(where, "must be an array of three numbers");
  for (int k = 0; k < 3; ++k) out[k] = finite_at(j[k], where + "[" + std::to_string(k) + "]");
}

std::vector<std::string> names_at(const json& j, const std::string& where) {
  if (!j.is_array() || j.empty()) fail(where, "must be a non-empty array of names");
  std::vector<std::string> out;
  for (std::size_t i = 0; i < j.size(); ++i) {
    out.push_back(string_at(j[i], where + "[" + std::to_string(i) + "]"));
    if (out.back().empty()) fail(where, "holds an empty name");
  }
  return out;
}

std::string name_at(const json& j, const std::string& where) {
  const std::string s = string_at(j, where);
  bool ok = !s.empty() && (std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_');
  for (char c : s) ok = ok && (std::isalnum(static_cast<unsigned char>(c)) || c == '_');
  if (!ok) fail(where, "'" + s + "' is not a name [A-Za-z_][A-Za-z0-9_]*");
  return s;
}

std::vector<FieldSpec> fields_at(const json& j, const fs::path& base) {
  if (!j.is_array()) fail("fields", "must be an array");
  std::vector<FieldSpec> out;
  for (std::size_t i = 0; i < j.size(); ++i) {
    const std::string w = "fields[" + std::to_string(i) + "]";
    const json& f = object_at(j[i], w);
    if (!f.contains("type")) fail(w, "missing key 'type'");
    FieldSpec s;
    s.type = string_at(f["type"], w + ".type");
    if (s.type == "uniform") {
      check_keys(f, w, {"type", "B_T", "regions"}, {});
      vector3_at(f["B_T"], w + ".B_T", s.B_T);
    } else if (s.type == "current_loop") {
      check_keys(f, w, {"type", "radius_m", "current_A", "z_m", "regions"}, {});
      s.radius_m = positive_at(f["radius_m"], w + ".radius_m");
      s.current_A = finite_at(f["current_A"], w + ".current_A");
      s.z_m = finite_at(f["z_m"], w + ".z_m");
    } else if (s.type == "map") {
      check_keys(f, w, {"type", "path", "patch", "regions"}, {});
      s.path = path_at(f["path"], w + ".path", base);
      s.patch = string_at(f["patch"], w + ".patch");
      if (s.patch.empty()) fail(w + ".patch", "must not be empty");
    } else {
      fail(w + ".type", "'" + s.type + "' is not uniform, current_loop or map");
    }
    s.regions = names_at(f["regions"], w + ".regions");
    out.push_back(s);
  }
  return out;
}

Outside outside_at(const json& j) {
  const std::string s = string_at(j, "outside");
  if (s == "refuse") return Outside::refuse;
  if (s == "zero") return Outside::zero;
  fail("outside", "must be \"refuse\" or \"zero\", not '" + s + "'");
}

IntegratorSpec integrator_at(const json& j) {
  const json& o = object_at(j, "integrator");
  check_keys(o, "integrator", {},
             {"delta_chord_m", "delta_one_step_m", "delta_intersection_m", "epsilon_min", "epsilon_max",
              "min_step_m"});
  IntegratorSpec s;
  if (o.contains("delta_chord_m")) s.delta_chord_m = positive_at(o["delta_chord_m"], "integrator.delta_chord_m");
  if (o.contains("delta_one_step_m"))
    s.delta_one_step_m = positive_at(o["delta_one_step_m"], "integrator.delta_one_step_m");
  if (o.contains("delta_intersection_m"))
    s.delta_intersection_m = positive_at(o["delta_intersection_m"], "integrator.delta_intersection_m");
  if (o.contains("epsilon_min")) s.epsilon_min = positive_at(o["epsilon_min"], "integrator.epsilon_min");
  if (o.contains("epsilon_max")) s.epsilon_max = positive_at(o["epsilon_max"], "integrator.epsilon_max");
  if (o.contains("min_step_m")) s.min_step_m = positive_at(o["min_step_m"], "integrator.min_step_m");
  if (!(s.epsilon_min <= s.epsilon_max && s.epsilon_max <= 1e-2))
    fail("integrator", "needs epsilon_min <= epsilon_max <= 1e-2");
  return s;
}

SourceSpec source_at(const json& j) {
  const json& o = object_at(j, "source");
  check_keys(o, "source", {"type", "particle", "at_m", "direction"},
             {"energy_eV", "momentum_eV_c", "spot_sigma_m"});
  SourceSpec s;
  if (string_at(o["type"], "source.type") != "gun") fail("source.type", "must be \"gun\"");
  if (o["particle"].is_number_integer()) {
    s.pdg = o["particle"].get<int>();
  } else {
    s.particle = string_at(o["particle"], "source.particle");
    if (s.particle.empty()) fail("source.particle", "must not be empty");
  }
  if (o.contains("energy_eV") == o.contains("momentum_eV_c"))
    fail("source", "needs exactly one of energy_eV and momentum_eV_c");
  s.by_momentum = o.contains("momentum_eV_c");
  s.value_eV = positive_at(o[s.by_momentum ? "momentum_eV_c" : "energy_eV"],
                           s.by_momentum ? "source.momentum_eV_c" : "source.energy_eV");
  vector3_at(o["at_m"], "source.at_m", s.at_m);
  if (o["direction"].is_string()) {
    if (o["direction"].get<std::string>() != "isotropic")
      fail("source.direction", "must be three numbers or \"isotropic\"");
    s.isotropic = true;
  } else {
    vector3_at(o["direction"], "source.direction", s.direction);
    const double n = std::sqrt(s.direction[0] * s.direction[0] + s.direction[1] * s.direction[1] +
                               s.direction[2] * s.direction[2]);
    if (!(n > 0.0)) fail("source.direction", "must not be the zero vector");
    for (double& d : s.direction) d /= n;
  }
  if (o.contains("spot_sigma_m")) {
    s.spot_sigma_m = finite_at(o["spot_sigma_m"], "source.spot_sigma_m");
    if (s.spot_sigma_m < 0.0) fail("source.spot_sigma_m", "must not be negative");
  }
  if (s.isotropic && s.spot_sigma_m != 0.0) fail("source", "an isotropic source is a point (spot_sigma_m 0)");
  return s;
}

TallySpec tally_at(const json& j, const std::string& w) {
  const json& o = object_at(j, w);
  check_keys(o, w, {"name", "grid", "regions"}, {});
  TallySpec s;
  s.name = name_at(o["name"], w + ".name");
  const json& g = object_at(o["grid"], w + ".grid");
  if (!g.contains("type")) fail(w + ".grid", "missing key 'type'");
  const std::string type = string_at(g["type"], w + ".grid.type");
  std::vector<std::string> axes;
  if (type == "cylindrical") {
    check_keys(g, w + ".grid", {"type", "r_m", "z_m", "bins"}, {});
    axes = {"r_m", "z_m"};
  } else if (type == "cartesian") {
    check_keys(g, w + ".grid", {"type", "x_m", "y_m", "z_m", "bins"}, {});
    axes = {"x_m", "y_m", "z_m"};
    s.cylindrical = false;
  } else {
    fail(w + ".grid.type", "must be cylindrical or cartesian");
  }
  s.axes = static_cast<int>(axes.size());
  const json& b = g["bins"];
  if (!b.is_array() || b.size() != axes.size()) fail(w + ".grid.bins", "needs one count per axis");
  for (std::size_t a = 0; a < axes.size(); ++a) {
    const json& r = g[axes[a]];
    const std::string aw = w + ".grid." + axes[a];
    if (!r.is_array() || r.size() != 2) fail(aw, "must be [lo, hi]");
    s.lo[a] = finite_at(r[0], aw + "[0]");
    s.hi[a] = finite_at(r[1], aw + "[1]");
    if (!(s.lo[a] < s.hi[a])) fail(aw, "needs lo < hi");
    s.bins[a] = static_cast<int>(unsigned_at(b[a], w + ".grid.bins", 1, 100000000));
  }
  if (s.size() > 400000000ULL) fail(w + ".grid.bins", "has too many bins");
  s.regions = names_at(o["regions"], w + ".regions");
  return s;
}

HitsSpec hits_at(const json& j, const std::string& w) {
  const json& o = object_at(j, w);
  check_keys(o, w, {"name", "regions"}, {"track", "particles", "max_per_event"});
  HitsSpec s;
  s.name = name_at(o["name"], w + ".name");
  s.regions = names_at(o["regions"], w + ".regions");
  if (o.contains("track")) {
    const std::string t = string_at(o["track"], w + ".track");
    if (t != "primary" && t != "any") fail(w + ".track", "must be \"primary\" or \"any\"");
    s.primary_only = t == "primary";
  }
  if (o.contains("particles") && !o["particles"].is_null()) {
    s.all_particles = false;
    s.particles = names_at(o["particles"], w + ".particles");
  }
  if (o.contains("max_per_event"))
    s.max_per_event = static_cast<std::uint32_t>(unsigned_at(o["max_per_event"], w + ".max_per_event", 1, 1000000000));
  return s;
}

}  // namespace

Job load_job(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) fail("", "cannot read the job file " + file.string());
  json j;
  try {
    j = json::parse(in);
  } catch (const json::exception& e) {
    fail("", "the job file " + file.string() + " is not valid JSON: " + e.what());
  }
  const fs::path base = fs::absolute(file).parent_path();
  object_at(j, "job");
  if (!j.contains("mode")) fail("job", "missing key 'mode'");
  Job job;
  job.mode = string_at(j["mode"], "mode");
  {
    bool known = false;
    for (const auto& m : offered_modes()) known = known || m == job.mode;
    if (!known) fail("mode", "unknown mode '" + job.mode + "'");
  }
  if (job.mode == "hello") {
    check_keys(j, "job", {"schema", "mode", "out", "chunks"}, {"physics_list", "hello", "record", "debug"});
  } else if (job.mode == "transport") {
    check_keys(j, "job", {"schema", "mode", "out", "chunks", "solids", "source"},
               {"record", "physics_list", "production_cut_m", "fields", "outside", "integrator", "quantum_eV",
                "tallies", "hits", "debug"});
  } else {
    check_keys(j, "job", {"schema", "mode", "out", "solids", "fields", "probe"}, {"outside", "record"});
  }
  if (string_at(j["schema"], "schema") != kJobSchema)
    fail("schema", std::string("must be \"") + kJobSchema + "\"");
  job.out = path_at(j["out"], "out", base);
  job.record = j.contains("record") ? path_at(j["record"], "record", base) : job.out / "record.json";

  if (j.contains("physics_list")) {
    job.physics_list = string_at(j["physics_list"], "physics_list");
    bool known = false;
    for (const auto& m : offered_physics_lists()) known = known || m == job.physics_list;
    if (!known) fail("physics_list", "unknown physics list '" + job.physics_list + "'");
  }

  TransportSpec& tr = job.transport;
  if (job.mode != "hello") {
    tr.solids = path_at(j["solids"], "solids", base);
    if (j.contains("fields")) tr.fields = fields_at(j["fields"], base);
    if (j.contains("outside")) tr.outside = outside_at(j["outside"]);
  }
  if (job.mode == "probe") {
    const json& p = object_at(j["probe"], "probe");
    check_keys(p, "probe", {"points"}, {});
    tr.probe_points = path_at(p["points"], "probe.points", base);
    return job;
  }
  if (job.mode == "transport") {
    if (j.contains("production_cut_m")) tr.production_cut_m = positive_at(j["production_cut_m"], "production_cut_m");
    if (j.contains("integrator")) tr.integrator = integrator_at(j["integrator"]);
    tr.source = source_at(j["source"]);
    if (j.contains("quantum_eV")) tr.quantum_eV = positive_at(j["quantum_eV"], "quantum_eV");
    std::set<std::string> names;
    if (j.contains("tallies")) {
      if (!j["tallies"].is_array()) fail("tallies", "must be an array");
      for (std::size_t i = 0; i < j["tallies"].size(); ++i) {
        tr.tallies.push_back(tally_at(j["tallies"][i], "tallies[" + std::to_string(i) + "]"));
        if (!names.insert("t:" + tr.tallies.back().name).second)
          fail("tallies", "the name '" + tr.tallies.back().name + "' is used twice");
      }
    }
    if (j.contains("hits")) {
      if (!j["hits"].is_array()) fail("hits", "must be an array");
      for (std::size_t i = 0; i < j["hits"].size(); ++i) {
        tr.hits.push_back(hits_at(j["hits"][i], "hits[" + std::to_string(i) + "]"));
        if (!names.insert("h:" + tr.hits.back().name).second)
          fail("hits", "the name '" + tr.hits.back().name + "' is used twice");
      }
    }
  }

  const json& chunks = j["chunks"];
  if (!chunks.is_array() || chunks.empty()) fail("chunks", "must be a non-empty array");
  std::set<std::uint64_t> seen;
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    const std::string w = "chunks[" + std::to_string(i) + "]";
    const json& c = object_at(chunks[i], w);
    check_keys(c, w, {"index", "seed", "events"}, {});
    ChunkSpec s;
    s.index = unsigned_at(c["index"], w + ".index", 0, 0x7FFFFFFFULL);
    s.seed = unsigned_at(c["seed"], w + ".seed", 0, UINT64_MAX);
    s.events = unsigned_at(c["events"], w + ".events", 1, 0x7FFFFFFFULL);
    if (!seen.insert(s.index).second) fail(w + ".index", "repeats the index of an earlier chunk");
    job.chunks.push_back(s);
  }

  if (job.mode == "hello" && j.contains("hello")) {
    const json& h = object_at(j["hello"], "hello");
    check_keys(h, "hello", {}, {"particle", "energy_eV", "field_T", "target_thickness_m"});
    if (h.contains("particle")) {
      job.hello.particle = string_at(h["particle"], "hello.particle");
      if (job.hello.particle.empty()) fail("hello.particle", "must not be empty");
    }
    if (h.contains("energy_eV")) {
      const json& e = h["energy_eV"];
      if (!e.is_array() || e.size() != 2) fail("hello.energy_eV", "must be an array [low, high] of two numbers");
      job.hello.energy_lo_eV = finite_at(e[0], "hello.energy_eV[0]");
      job.hello.energy_hi_eV = finite_at(e[1], "hello.energy_eV[1]");
      if (!(job.hello.energy_lo_eV > 0.0) || job.hello.energy_hi_eV < job.hello.energy_lo_eV)
        fail("hello.energy_eV", "needs 0 < low <= high");
    }
    if (h.contains("field_T")) {
      const json& f = h["field_T"];
      if (!f.is_array() || f.size() != 3) fail("hello.field_T", "must be an array of three numbers (tesla)");
      for (int k = 0; k < 3; ++k) job.hello.field_T[k] = finite_at(f[k], "hello.field_T[" + std::to_string(k) + "]");
    }
    if (h.contains("target_thickness_m")) {
      job.hello.target_thickness_m = finite_at(h["target_thickness_m"], "hello.target_thickness_m");
      if (!(job.hello.target_thickness_m > 0.0 && job.hello.target_thickness_m <= 0.9))
        fail("hello.target_thickness_m", "must lie in (0, 0.9] metres");
    }
  }

  if (j.contains("debug")) {
    const json& d = object_at(j["debug"], "debug");
    check_keys(d, "debug", {}, {"seed_once_per_communication", "fail_chunks"});
    if (d.contains("seed_once_per_communication"))
      job.debug_seed_once_per_communication =
          static_cast<int>(unsigned_at(d["seed_once_per_communication"], "debug.seed_once_per_communication", 0, 2));
    if (d.contains("fail_chunks")) {
      const json& f = d["fail_chunks"];
      if (!f.is_array()) fail("debug.fail_chunks", "must be an array of chunk indices");
      for (std::size_t i = 0; i < f.size(); ++i)
        tr.fail_chunks.push_back(unsigned_at(f[i], "debug.fail_chunks[" + std::to_string(i) + "]", 0, UINT64_MAX));
    }
  }
  return job;
}

}  // namespace axtr
