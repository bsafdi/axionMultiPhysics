// Tangled curved elements on the model path (model_mesh.hpp). A model document whose thin curved region tangles at
// geometry order 2 (a ring 0.498 < r < 0.5 m at mesh size 0.05 m: 63 of its 126 triangles fold over without the
// mesher's check, gmsh 4.15.2) is meshed into a mesh with no tangled element, or refused as any failed mesher run is
// (the same message prefix and code); with gmsh's optimisation switched off (AXMP_MESHER_HO=none) it is refused, with
// the mesher's text. read_meshed_model refuses a tangled mesh file whoever wrote it (code mesh.tangled), and reads the
// same file untangled. Returns 77 (CMake's SKIP_RETURN_CODE) when there is no axmp-mesher.
//
// INDEPENDENT REFERENCE. det J of every 6-node triangle from the node coordinates of the MSH 2.2 file (the P2 shape
// functions in gmsh's node order) on the lattice (i/12, j/12), i + j <= 12; an element is tangled when det J is not of
// one strict sign there, or has the sign opposite to the majority of the elements of its region (the field-map
// builder's definition, cpp/src/fieldmap.cpp; tests/test_mesher_tangled.py is the same in Python).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/axi.hpp"
#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_mesh.hpp"
#include "axionmultiphysics/model_validate.hpp"

using nlohmann::json;
namespace M = axmp::model;
namespace fs = std::filesystem;

static int failures = 0;
static void fail(const std::string& what) {
  std::printf("FAIL: %s\n", what.c_str());
  std::fflush(stdout);
  ++failures;
}
#define CHECK(cond, what)                 \
  do {                                    \
    if (!(cond)) fail(std::string(what)); \
  } while (0)

namespace {

struct TempDir {
  std::string path;
  TempDir() {
    std::string t = (fs::temp_directory_path() / "axmp_model_mesh_tangled_XXXXXX").string();
    std::vector<char> buf(t.begin(), t.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) throw std::runtime_error("mkdtemp failed");
    path = buf.data();
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

json read_json(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  return json::parse(in);
}
std::string data_path(const std::string& name) {
#ifdef AXMP_SOURCE_DIR
  return std::string(AXMP_SOURCE_DIR) + "/cpp/tests/data/model/u2/" + name;
#else
  return "cpp/tests/data/model/u2/" + name;
#endif
}
M::ValidationReport validated(const json& doc, const std::string& what) {
  M::ValidationReport rep = M::validate_model(doc);
  if (!rep.ok()) throw std::runtime_error(what + ": the document does not validate: " + rep.errors.at(0).what());
  return rep;
}
void run(const char* name, const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception& e) {
    fail(std::string(name) + ": unexpected exception: " + e.what());
  }
}

// ---- an MSH 2.2 ASCII file: the lines, the nodes, the 6-node triangles and the physical names of dimension 2
struct Msh {
  std::vector<std::string> lines;
  std::map<long, std::pair<double, double>> node;
  std::map<long, size_t> node_line;
  std::vector<std::vector<long>> tri;  // gmsh order v0 v1 v2 e01 e12 e20
  std::vector<int> phys;
  std::map<int, std::string> name;
};
Msh read_msh(const std::string& path) {
  Msh m;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  for (std::string l; std::getline(in, l);) m.lines.push_back(l);
  auto at = [&](const std::string& s) {
    for (size_t i = 0; i < m.lines.size(); ++i)
      if (m.lines[i] == s) return i;
    throw std::runtime_error(path + ": no " + s);
  };
  {
    const size_t i = at("$PhysicalNames");
    const int n = std::stoi(m.lines[i + 1]);
    for (int k = 0; k < n; ++k) {
      std::istringstream s(m.lines[i + 2 + k]);
      int dim = 0, tag = 0;
      std::string rest;
      s >> dim >> tag;
      std::getline(s, rest);
      const size_t a = rest.find('"'), b = rest.rfind('"');
      if (dim == 2 && a != std::string::npos && b > a) m.name[tag] = rest.substr(a + 1, b - a - 1);
    }
  }
  {
    const size_t i = at("$Nodes");
    const long n = std::stol(m.lines[i + 1]);
    for (long k = 0; k < n; ++k) {
      std::istringstream s(m.lines[i + 2 + k]);
      long t = 0;
      double x = 0, y = 0;
      s >> t >> x >> y;
      m.node[t] = {x, y};
      m.node_line[t] = i + 2 + k;
    }
  }
  {
    const size_t i = at("$Elements");
    const long n = std::stol(m.lines[i + 1]);
    for (long k = 0; k < n; ++k) {
      std::istringstream s(m.lines[i + 2 + k]);
      long t = 0;
      int type = 0, nt = 0;
      s >> t >> type >> nt;
      std::vector<long> tags(nt);
      for (auto& v : tags) s >> v;
      if (type != 9) continue;
      std::vector<long> c(6);
      for (auto& v : c) s >> v;
      m.tri.push_back(c);
      m.phys.push_back(static_cast<int>(tags.at(0)));
    }
  }
  return m;
}
void write_msh(const Msh& m, const std::string& path) {
  std::ofstream out(path);
  for (size_t i = 0; i < m.lines.size(); ++i) out << m.lines[i] << (i + 1 < m.lines.size() ? "\n" : "");
}

// min and max of det J over the lattice, per triangle
void det_range(const Msh& m, std::vector<double>& mn, std::vector<double>& mx) {
  mn.assign(m.tri.size(), 1e300);
  mx.assign(m.tri.size(), -1e300);
  const int n = 12;
  for (size_t e = 0; e < m.tri.size(); ++e) {
    double P[6][2];
    for (int k = 0; k < 6; ++k) P[k][0] = m.node.at(m.tri[e][k]).first, P[k][1] = m.node.at(m.tri[e][k]).second;
    for (int i = 0; i <= n; ++i)
      for (int j = 0; i + j <= n; ++j) {
        const double u = double(i) / n, v = double(j) / n, w = 1 - u - v;
        const double dN[6][2] = {{-(4 * w - 1), -(4 * w - 1)}, {4 * u - 1, 0}, {0, 4 * v - 1},
                                 {4 * (w - u), -4 * u}, {4 * v, 4 * u}, {-4 * v, 4 * (w - v)}};
        double J[2][2] = {{0, 0}, {0, 0}};
        for (int k = 0; k < 6; ++k)
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) J[a][b] += P[k][a] * dN[k][b];
        const double d = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        mn[e] = std::min(mn[e], d);
        mx[e] = std::max(mx[e], d);
      }
  }
}
int tangled_count(const Msh& m) {
  std::vector<double> mn, mx;
  det_range(m, mn, mx);
  std::map<int, std::pair<int, int>> sign;
  for (size_t e = 0; e < mn.size(); ++e) {
    if (mn[e] > 0) ++sign[m.phys[e]].first;
    else if (mx[e] < 0) ++sign[m.phys[e]].second;
  }
  int bad = 0;
  for (size_t e = 0; e < mn.size(); ++e) {
    const auto& s = sign[m.phys[e]];
    if (!(s.second > s.first ? mx[e] < 0 : mn[e] > 0)) ++bad;
  }
  return bad;
}
// Moves the mid-node of the first edge of one triangle of `region` (an edge shared with another triangle of the region)
// beyond the triangle's opposite vertex.
void tangle(Msh& m, const std::string& region) {
  int attr = -1;
  for (const auto& [t, n] : m.name)
    if (n == region) attr = t;
  if (attr < 0) throw std::runtime_error("no region " + region);
  std::map<std::pair<long, long>, int> owners;
  auto key = [](long a, long b) { return std::make_pair(std::min(a, b), std::max(a, b)); };
  for (size_t e = 0; e < m.tri.size(); ++e)
    if (m.phys[e] == attr)
      for (int k = 0; k < 3; ++k) ++owners[key(m.tri[e][k], m.tri[e][(k + 1) % 3])];
  for (size_t e = 0; e < m.tri.size(); ++e) {
    if (m.phys[e] != attr || owners[key(m.tri[e][0], m.tri[e][1])] != 2) continue;
    const auto mid = m.node.at(m.tri[e][3]), v2 = m.node.at(m.tri[e][2]);
    const double x = v2.first + 0.5 * (v2.first - mid.first), y = v2.second + 0.5 * (v2.second - mid.second);
    m.node[m.tri[e][3]] = {x, y};
    char buf[128];
    std::snprintf(buf, sizeof buf, "%ld %.17g %.17g 0", m.tri[e][3], x, y);
    m.lines[m.node_line.at(m.tri[e][3])] = buf;
    return;
  }
  throw std::runtime_error("no interior edge in " + region);
}

bool has_count(const std::string& text, int k) {
  return std::regex_search(text, std::regex("(^|[^0-9.])" + std::to_string(k) + "([^0-9.]|$)"));
}

// the gmsh of this mesher, from the info file of a valid mesh (a refusal's info file need not name it)
std::string g_gmsh_version;
void probe_gmsh_version() {
  const M::ValidationReport rep = validated(read_json(data_path("annulus.json")), "annulus.json");
  TempDir dir;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  g_gmsh_version = M::mesh_model(rep, opt).info.value("gmsh_version", std::string());
}

json thin_ring_doc() {
  json d = read_json(data_path("annulus.json"));
  d["name"] = "thin_ring";
  d["parameters"]["R1"] = 0.498;
  d["parameters"]["R2"] = 0.5;
  d["mesh"] = {{"size", 0.05}, {"geometry_order", 2}};
  return d;
}

// ---- the thin ring: repaired (the mesh has no tangled element, the info file records it) or refused as any failed
// mesher run
void check_thin_ring_meshed() {
  const M::ValidationReport rep = validated(thin_ring_doc(), "thin ring");
  TempDir dir;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  try {
    const M::MeshedModel m = M::mesh_model(rep, opt);
    const Msh f = read_msh(m.mesh_file);
    CHECK(tangled_count(f) == 0, "thin ring: the mesh read by the model path has " + std::to_string(tangled_count(f)) +
                                     " tangled elements");
    CHECK(m.info.contains("high_order") && m.info["high_order"].value("tangled_before", 0) >= 1 &&
              m.info["high_order"].value("tangled_after", -1) == 0,
          "thin ring: the info file records the repair: " + m.info.value("high_order", json()).dump());
    CHECK(m.geometry_order == 2, "thin ring: the repaired mesh is curved");
  } catch (const axmp::Error& e) {
    CHECK(g_gmsh_version != "4.15.2",
          std::string("thin ring: refused on the pinned gmsh, which repairs it: ") + e.what());
    CHECK(e.message().rfind("mesh generation failed: ", 0) == 0, "thin ring: refused, the message " + e.message());
  }
}

// a mesher that fails for another reason: exit status 1, {"ok": false, "error": ...}
std::string failing_mesher(const std::string& dir) {
  const std::string script = dir + "/failing-mesher";
  std::ofstream(script) << "#!/bin/sh\n"
                           "info=$(sed -n 's/.*\"info\": *\"\\([^\"]*\\)\".*/\\1/p' \"$1\" | head -n 1)\n"
                           "printf '{\"ok\": false, \"error\": \"some other failure\"}' > \"$info\"\n"
                           "echo 'axmp-mesher: some other failure' >&2\n"
                           "exit 1\n";
  fs::permissions(script, fs::perms::owner_all);
  return script;
}

// ---- with the optimisation off the mesher refuses the thin ring; the model path reports it as any failed run
void check_thin_ring_refused() {
  const M::ValidationReport rep = validated(thin_ring_doc(), "thin ring");
  TempDir dir, other;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  std::string code_tangled, code_other = "(none)";
  setenv("AXMP_MESHER_HO", "none", 1);
  try {
    M::mesh_model(rep, opt);
    fail("thin ring, AXMP_MESHER_HO=none: no error");
  } catch (const axmp::Error& e) {
    const json info = read_json(dir.path + "/mesh_info.json");
    CHECK(info.value("ok", true) == false, "thin ring, AXMP_MESHER_HO=none: the info file says ok: " + info.dump());
    const std::string text = info.value("error", std::string());
    const int k = info.contains("high_order") ? info["high_order"].value("tangled_after", -1) : -1;
    CHECK(k >= 1, "thin ring, AXMP_MESHER_HO=none: the record: " + info.value("high_order", json()).dump());
    if (g_gmsh_version == "4.15.2") CHECK(k == 63, "thin ring: tangled " + std::to_string(k));
    CHECK(e.message() == "mesh generation failed: " + text, "thin ring: not the mesher's error: " + e.message());
    CHECK(has_count(text, k) && text.find("'ring'") != std::string::npos && text.find("order 1") != std::string::npos,
          "thin ring: the mesher's error names the count, the region and the remedy: " + text);
    CHECK(!fs::exists(dir.path + "/mesh.msh"), "thin ring: a refused mesh file exists");
    code_tangled = e.code();
  }
  unsetenv("AXMP_MESHER_HO");
  M::MeshOptions o2;
  o2.workdir = other.path;
  o2.mesher = failing_mesher(other.path);
  try {
    M::mesh_model(rep, o2);
    fail("a failing mesher: no error");
  } catch (const axmp::Error& e) {
    CHECK(e.message() == "mesh generation failed: some other failure", "a failing mesher: " + e.message());
    code_other = e.code();
  }
  CHECK(code_tangled == code_other, "the refusal's code '" + code_tangled + "' is not that of another failed run '" +
                                        code_other + "'");
}

// ---- read_meshed_model refuses a tangled file (mesh.tangled) and reads it untangled
void check_read_refuses_a_tangled_file() {
  const M::ValidationReport rep = validated(read_json(data_path("annulus.json")), "annulus.json");
  TempDir dir;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  const M::MeshedModel m = M::mesh_model(rep, opt);
  Msh f = read_msh(m.mesh_file);
  CHECK(tangled_count(f) == 0, "annulus: the mesher's mesh is tangled");
  const std::string copy = dir.path + "/copy.msh", bad = dir.path + "/tangled.msh";
  write_msh(f, copy);
  const M::MeshedModel ok = M::read_meshed_model(rep, copy, m.info_file, 0);
  CHECK(ok.geometry_order == 2, "annulus: the copy is read, curved");
  tangle(f, "ring");
  write_msh(f, bad);
  const int k = tangled_count(read_msh(bad));
  CHECK(k >= 1 && k <= 2, "annulus: the hand tangling tangles 1 or 2 elements, not " + std::to_string(k));
  try {
    M::read_meshed_model(rep, bad, m.info_file, 0);
    fail("annulus: a tangled mesh file is read");
  } catch (const axmp::Error& e) {
    CHECK(e.code() == "mesh.tangled", "annulus: the code is '" + e.code() + "'");
    CHECK(e.message().rfind(std::to_string(k) + " of ", 0) == 0,
          "annulus: the message starts with the count " + std::to_string(k) + ": " + e.message());
    CHECK(e.message().find("'ring'") != std::string::npos, "annulus: the message names the region: " + e.message());
    CHECK(e.message().find("tangled.msh") != std::string::npos || e.hint().find("tangled.msh") != std::string::npos,
          "annulus: the error names the file: " + std::string(e.what()));
    CHECK(e.hint().find("order 1") != std::string::npos &&
              (e.hint().find("efine") != std::string::npos),
          "annulus: the hint gives the remedy: " + e.hint());
  }
  // with refinements: the check is on the file, before any refinement
  try {
    M::read_meshed_model(rep, bad, m.info_file, 1);
    fail("annulus: a tangled mesh file is read with one refinement");
  } catch (const axmp::Error& e) {
    CHECK(e.code() == "mesh.tangled", "annulus, one refinement: the code is '" + e.code() + "'");
  }
}

}  // namespace

int main() {
  try {
    axmp::axi::find_mesher();
  } catch (const std::exception&) {
    std::printf("SKIPPED: no axmp-mesher in this build\n");
    return 77;
  }
  unsetenv("AXMP_MESHER_HO");
  unsetenv("AXMP_MESHER_HO_ISOLATE");
  run("the gmsh version of this mesher", probe_gmsh_version);
  std::printf("gmsh %s\n", g_gmsh_version.c_str());
  run("the thin ring meshed: repaired, or refused as a failed run", check_thin_ring_meshed);
  run("the thin ring with the optimisation off: refused with the mesher's text", check_thin_ring_refused);
  run("read_meshed_model refuses a tangled file", check_read_refuses_a_tangled_file);
  if (failures) {
    std::printf("%d FAILURES\n", failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
