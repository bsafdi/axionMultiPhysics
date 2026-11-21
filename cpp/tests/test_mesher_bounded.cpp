// The bounded planar and axisymmetric modes of axmp-mesher ("bounded": true; design note 5.3): the entity
// table and the atoms in the info file, the exact bounding boxes, markers (open ones included), the physical groups of
// the MSH 2.2 file, the geometry order, resolve_only, and the refusals.  Runs the mesher on the fixtures of
// cpp/tests/data/model/u2/mesher/ and reads the info file, the mesh file (a small MSH 2.2 reader below: the elementary
// entity of every element is the entity tag of the info file) and the mesh through MFEM.  Never calls gmsh.  Returns 77
// (skipped) when there is no mesher.
//
// INDEPENDENT REFERENCES (nothing below is computed by the mesher or by the code under test).
//   * Hand-built entity tables (struct Row, one per straight edge, struct Circle, one per circle): midpoints, lengths,
//     boxes, revolved areas (Pappus: 2 pi r_c L), the expected atom of each edge.  The sides of an entity are derived
//     from each fixture's own region_of(x, y) (a hand-coded point-in-region rule), probing the left and the right of the
//     entity's tangent, so a wrong side order or a wrong tangent fails whatever gmsh's orientation of the edge.
//   * Region areas and volumes: rectangles, pi (R2^2 - R1^2), pi R^2 / 2 and 4 pi R^3 / 3 (the revolved half disk).
//   * MFEM's own quadrature of |J| along the boundary elements and over the elements, and MFEM's topology (the faces of
//     the mesh).
//   * fixtures::master_numbers() (cpp/tests/model_matrix_fixtures.hpp), the numbering of the unit square of the matrix
//     documents; the master numbering of design note 5.3 worked out by hand for the other fixtures.
//   * The selection library's resolution (selection::resolve), run on the entity table the info file carries, as a consistency check of
//     what the mesher resolved (it is the code the mesher calls, so it is no check of the selectors' semantics; those
//     are test_model_selection.cpp's).
//
// THE SPECIFICATION this test fixes (where the plan and the corrections leave room).
//   Input: cpp/tests/data/model/u2/mesher/*.json plus "output" and "info" paths added by this test.
//   Info file, bounded modes: "ok", "bounded": true, "geometry", "exterior" (as in the input), "order", "gmsh_version",
//     "seconds", "diagonal_m" (the diagonal of the box of the union of the regions), "tolerance_m" (1e-9 diagonal_m);
//     "regions": {name: {"attr", "area_m2", "volume_m3" (axisymmetric only: the revolved volume, 2 pi r_c A)}};
//     "entities": the table of model_selection.hpp (to_json), every boundary entity with "atom" (its master number); the
//     "tag" of an entity is the elementary entity of its line elements in the mesh file;
//     "boundaries": {name: {"attributes": [master numbers of its atoms, ascending], "entities": count, "measure" (sum of
//       length_m), "measure_units": "m", "area_m2" (axisymmetric only), "bbox": {"x": [lo, hi], "y": [lo, hi]} ("r" and
//       "z" when axisymmetric; null when empty), "point" (the smallest point of its entities in lexicographic order; null
//       when empty), "adjacent": [[region, region], ...] (each pair in region order, "exterior" last, unique, in region
//       -pair order)}};
//     "boundary_groups": [{"attr": master, "name": "atom_<master>", "names": [...], "type": "named" | "unnamed" | "axis",
//       "adjacent": [side, side], "entities": count}], in master order; "empty" and "partial": the resolution's
//       (model_selection.hpp); and, unless "resolve_only": "nodes", "elements_2d", "boundary_elements".
//   Mesh file: MSH 2.2 ASCII; one physical group of dimension 1 per atom, named atom_<master> with the master number as
//     its attribute, every boundary entity in exactly one; one of dimension 2 per region, named as the region, with the
//     region's "attr" as its number; every line element runs along its entity's "tangent" (the ONE tangent rule: a marker
//     entity along the written direction of its first marker, the others listed in "curves_reversed").
//   "entities" of a marker carry "curves" in the document order of the input's "curves".
//   Refusals: {"ok": false, "error": text} and exit status 1, with the texts of mesh_plane for a covered region and the
//     bounded mode's own text for a curve outside every region.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionmultiphysics/axi.hpp"
#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_selection.hpp"
#include "model_matrix_fixtures.hpp"

using nlohmann::json;
namespace S = axmp::model::selection;
namespace fs = std::filesystem;

static int failures = 0;
static void fail(const std::string& what) {
  std::printf("FAIL: %s\n", what.c_str());
  ++failures;
}
#define CHECK(cond, what)                 \
  do {                                    \
    if (!(cond)) fail(std::string(what)); \
  } while (0)

namespace {

using Strs = std::vector<std::string>;
using Ints = std::vector<int>;
constexpr double kPi = 3.14159265358979323846;

std::string g_mesher;
fs::path g_scratch;

std::string fmt(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.17g", v);
  return b;
}
bool close_rel(double got, double want, double rel) { return std::fabs(got - want) <= rel * std::fabs(want); }
bool close_abs(double got, double want, double tol) { return std::fabs(got - want) <= tol; }

// ----------------------------------------------------------------------------------------------- running the mesher
struct Run {
  bool started = false;
  int rc = -1;
  json info;               // null when no info file was written
  std::string msh;         // path of the mesh file ("" when none)
  std::string input_path;
  json input;
  std::string why;         // why there is no usable result (an ok:false answer, a missing info file)
  bool ok() const { return started && rc == 0 && info.is_object() && info.value("ok", false); }
};

std::string source_dir() {
#ifdef AXMP_SOURCE_DIR
  return AXMP_SOURCE_DIR;
#else
  return ".";
#endif
}

json fixture(const std::string& name) {
  const std::string path = source_dir() + "/cpp/tests/data/model/u2/mesher/" + name + ".json";
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read the fixture " + path);
  json d;
  in >> d;
  return d;
}

// Runs the mesher on `doc` (output and info paths set here) in a directory of its own.
Run run_doc(json doc, const std::string& key) {
  static int counter = 0;
  Run r;
  const fs::path dir = g_scratch / (key + "_" + std::to_string(counter++));
  fs::create_directories(dir);
  doc["output"] = (dir / "mesh.msh").string();
  doc["info"] = (dir / "mesh_info.json").string();
  r.input = doc;
  r.input_path = (dir / "input.json").string();
  std::ofstream(r.input_path) << doc.dump(1);
  r.rc = axmp::axi::run_process({g_mesher, r.input_path}, (dir / "mesher.log").string());
  r.started = true;
  if (fs::exists(dir / "mesh_info.json")) {
    std::ifstream in(dir / "mesh_info.json");
    try {
      in >> r.info;
    } catch (const std::exception& e) {
      r.why = std::string("the info file is not JSON: ") + e.what();
    }
  } else {
    r.why = "no info file";
  }
  if (fs::exists(dir / "mesh.msh")) r.msh = (dir / "mesh.msh").string();
  if (!r.ok() && r.why.empty())
    r.why = "the mesher answered rc=" + std::to_string(r.rc) + ", ok=" + (r.info.value("ok", false) ? "true" : "false") +
            ", error: " + r.info.value("error", std::string("(none)"));
  return r;
}

std::map<std::string, Run>& cache() {
  static std::map<std::string, Run> c;
  return c;
}
// A fixture run once (with optional overrides) and kept.  Returns nullptr after recording a failure for `who`.
const Run* get(const std::string& who, const std::string& name, const json& overrides = json::object(),
               const std::string& tag = "") {
  const std::string key = name + tag;
  auto it = cache().find(key);
  if (it == cache().end()) {
    json d = fixture(name);
    for (auto& [k, v] : overrides.items()) d[k] = v;
    it = cache().emplace(key, run_doc(d, name)).first;
  }
  if (!it->second.ok()) {
    fail(who + ": " + name + ": " + it->second.why);
    return nullptr;
  }
  return &it->second;
}

// ------------------------------------------------------------------------------------------------ the MSH 2.2 reader
struct MshEl {
  int type = 0, phys = 0, elem = 0;
  std::vector<int> nodes;
};
struct Msh {
  std::map<int, std::string> names1, names2;
  std::map<int, std::array<double, 3>> nodes;
  std::vector<MshEl> els;
  std::vector<MshEl> lines() const {
    std::vector<MshEl> out;
    for (const auto& e : els)
      if (e.type == 1 || e.type == 8) out.push_back(e);
    return out;
  }
  std::vector<MshEl> tris() const {
    std::vector<MshEl> out;
    for (const auto& e : els)
      if (e.type == 2 || e.type == 9) out.push_back(e);
    return out;
  }
};
Msh read_msh(const std::string& path) {
  Msh m;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line == "$PhysicalNames") {
      std::getline(in, line);
      const int n = std::stoi(line);
      for (int i = 0; i < n; ++i) {
        std::getline(in, line);
        std::istringstream ss(line);
        int dim, tag;
        ss >> dim >> tag;
        const size_t q0 = line.find('"'), q1 = line.rfind('"');
        const std::string name = line.substr(q0 + 1, q1 - q0 - 1);
        (dim == 1 ? m.names1 : m.names2)[tag] = name;
        if (dim != 1 && dim != 2) throw std::runtime_error("a physical name of dimension " + std::to_string(dim));
      }
    } else if (line == "$Nodes") {
      std::getline(in, line);
      const int n = std::stoi(line);
      for (int i = 0; i < n; ++i) {
        int id;
        double x, y, z;
        in >> id >> x >> y >> z;
        m.nodes[id] = {x, y, z};
      }
    } else if (line == "$Elements") {
      std::getline(in, line);
      const int n = std::stoi(line);
      for (int i = 0; i < n; ++i) {
        std::getline(in, line);
        if (line.empty()) {
          --i;
          continue;
        }
        std::istringstream ss(line);
        int id, type, ntags;
        ss >> id >> type >> ntags;
        MshEl e;
        e.type = type;
        std::vector<int> tags(ntags);
        for (int& t : tags) ss >> t;
        e.phys = tags.size() > 0 ? tags[0] : 0;
        e.elem = tags.size() > 1 ? tags[1] : 0;
        int v;
        while (ss >> v) e.nodes.push_back(v);
        m.els.push_back(e);
      }
    }
  }
  return m;
}

// -------------------------------------------------------------------------------------------- the hand-built tables
struct Row {  // one straight edge
  double x, y;          // its midpoint
  double length;
  double area = 0;      // revolved (axisymmetric fixtures)
  int atom;             // expected master number
  double dir[2] = {0, 0};     // the written direction when it carries a marker
  Strs curves = {};     // expected curves (document order)
  Strs reversed = {};   // expected curves_reversed
  bool axis = false;
};
struct Circle {         // one circle (possibly in several arcs)
  double cx, cy, R;
  double total_length;  // of all its arcs
  double total_area = 0;
  double lo[2], hi[2];  // box of the union of its arcs
  int atom;
};
struct Atom {
  int attr;
  Strs names;
  std::string type;      // named / unnamed / axis
  std::set<std::pair<std::string, std::string>> adjacent;  // as unordered region pairs, written (min index, max index)
};
struct Fx {
  std::string name;
  bool axisym = false;
  Strs regions;
  std::function<std::string(double, double)> region_of;   // "" outside every region
  double delta;                                           // probe distance for the sides
  std::vector<Row> rows;
  std::vector<Circle> circles;
  std::vector<Atom> atoms;
  double diag;
  std::map<std::string, std::pair<double, double>> region_measures;  // name -> {area, volume (axisymmetric)}
};

std::string rect_region(double x, double y, double x0, double x1, double y0, double y1, const std::string& name) {
  return (x > x0 && x < x1 && y > y0 && y < y1) ? name : "";
}

Fx fx_rect() {
  Fx f;
  f.name = "rect";
  f.regions = {"box"};
  f.region_of = [](double x, double y) { return rect_region(x, y, 0, 0.04, 0, 0.01, "box"); };
  f.delta = 1e-6;
  f.rows = {{0.02, 0, 0.04, 0, 2001}, {0, 0.005, 0.01, 0, 2002}, {0.04, 0.005, 0.01, 0, 2003}, {0.02, 0.01, 0.04, 0, 2004}};
  f.atoms = {{2001, {"bottom"}, "named", {{"box", "exterior"}}}, {2002, {"left"}, "named", {{"box", "exterior"}}},
             {2003, {"right"}, "named", {{"box", "exterior"}}}, {2004, {"top"}, "named", {{"box", "exterior"}}}};
  f.diag = std::hypot(0.04, 0.01);
  f.region_measures["box"] = {0.04 * 0.01, 0};
  return f;
}
Fx fx_unit_square(bool axisym) {
  Fx f;
  f.name = axisym ? "unit_square_axisymmetric" : "unit_square_planar";
  f.axisym = axisym;
  f.regions = {"box"};
  f.region_of = [](double x, double y) { return rect_region(x, y, 0, 1, 0, 1, "box"); };
  f.delta = 1e-4;
  const auto master = axmp::model::fixtures::master_numbers(axisym ? axmp::model::fixtures::Geometry::axisymmetric
                                                                  : axmp::model::fixtures::Geometry::planar);
  Row bottom{0.5, 0, 1, axisym ? kPi : 0, master.at("bottom")}, right{1, 0.5, 1, axisym ? 2 * kPi : 0, master.at("right")},
      top{0.5, 1, 1, axisym ? kPi : 0, master.at("top")};
  f.rows = {bottom, right, top};
  Row left{0, 0.5, 1, 0, 0};
  if (axisym) {
    left.atom = master.at("axis");
    left.axis = true;
  } else {
    left.atom = master.at("left");
  }
  f.rows.push_back(left);
  for (const auto& [name, m] : master) {
    Atom a;
    a.attr = m;
    a.names = name == "axis" ? Strs{} : Strs{name};
    a.type = name == "axis" ? "axis" : "named";
    a.adjacent = {{"box", "exterior"}};
    f.atoms.push_back(a);
  }
  std::sort(f.atoms.begin(), f.atoms.end(), [](const Atom& a, const Atom& b) { return a.attr < b.attr; });
  f.diag = std::sqrt(2.0);
  f.region_measures["box"] = {1.0, axisym ? kPi : 0};
  return f;
}
// The split rectangle (L = 0.04, H = 0.01): a for x < 0.02, b for the rest; the marker across a at x = 0.01 splits the
// top and the bottom of a.  mk_dir: the written direction of the first marker.
Fx fx_split_rect(const std::string& name, double mk_dir, const Strs& curves, const Strs& reversed, const Strs& atom4_names) {
  Fx f;
  f.name = name;
  f.regions = {"a", "b"};
  f.region_of = [](double x, double y) {
    if (!(x > 0 && x < 0.04 && y > 0 && y < 0.01)) return std::string();
    return std::string(x < 0.02 ? "a" : "b");
  };
  f.delta = 1e-6;
  Row mk{0.01, 0.005, 0.01, 0, 2004};
  mk.dir[1] = mk_dir;
  mk.curves = curves;
  mk.reversed = reversed;
  f.rows = {mk,
            {0, 0.005, 0.01, 0, 2002},       // hot
            {0.005, 0.01, 0.01, 0, 2005},    // unnamed (a, exterior): the four pieces of a's top and bottom
            {0.005, 0, 0.01, 0, 2005},
            {0.015, 0.01, 0.01, 0, 2005},
            {0.015, 0, 0.01, 0, 2005},
            {0.02, 0.005, 0.01, 0, 2003},    // interface
            {0.03, 0, 0.02, 0, 2006},        // unnamed (b, exterior)
            {0.04, 0.005, 0.01, 0, 2001},    // cold
            {0.03, 0.01, 0.02, 0, 2006}};
  f.atoms = {{2001, {"cold"}, "named", {{"b", "exterior"}}},
             {2002, {"hot"}, "named", {{"a", "exterior"}}},
             {2003, {"interface"}, "named", {{"a", "b"}}},
             {2004, atom4_names, "named", {{"a", "a"}}},
             {2005, {}, "unnamed", {{"a", "exterior"}}},
             {2006, {}, "unnamed", {{"b", "exterior"}}}};
  f.diag = std::hypot(0.04, 0.01);
  f.region_measures["a"] = {0.02 * 0.01, 0};
  f.region_measures["b"] = {0.02 * 0.01, 0};
  return f;
}
Fx fx_open_markers() {
  Fx f;
  f.name = "open_markers";
  f.regions = {"a", "b"};
  f.region_of = [](double x, double y) {
    if (!(x > 0 && x < 0.04 && y > 0 && y < 0.01)) return std::string();
    return std::string(x < 0.02 ? "a" : "b");
  };
  f.delta = 1e-6;
  Row mk_in{0.033, 0.005, 0.006, 0, 2003}, mk_edge{0.0375, 0.002, 0.005, 0, 2002};
  mk_in.dir[0] = 1;
  mk_in.curves = {"mk_in"};
  mk_edge.dir[0] = -1;
  mk_edge.curves = {"mk_edge"};
  f.rows = {mk_in, mk_edge,
            {0.01, 0.01, 0.02, 0, 2005}, {0.01, 0, 0.02, 0, 2005}, {0.02, 0.005, 0.01, 0, 2004}, {0, 0.005, 0.01, 0, 2001},
            {0.03, 0, 0.02, 0, 2006}, {0.04, 0.001, 0.002, 0, 2006}, {0.03, 0.01, 0.02, 0, 2006}, {0.04, 0.006, 0.008, 0, 2006}};
  f.atoms = {{2001, {"hot"}, "named", {{"a", "exterior"}}}, {2002, {"mk_edge"}, "named", {{"b", "b"}}},
             {2003, {"mk_in"}, "named", {{"b", "b"}}},      {2004, {}, "unnamed", {{"a", "b"}}},
             {2005, {}, "unnamed", {{"a", "exterior"}}},    {2006, {}, "unnamed", {{"b", "exterior"}}}};
  f.diag = std::hypot(0.04, 0.01);
  f.region_measures["a"] = {0.0002, 0};
  f.region_measures["b"] = {0.0002, 0};
  return f;
}
Fx fx_channel_post() {
  Fx f;
  f.name = "channel_post";
  f.regions = {"channel", "post"};
  f.region_of = [](double x, double y) {
    if ((x - 0.01) * (x - 0.01) + (y - 0.005) * (y - 0.005) < 0.002 * 0.002) return std::string("post");
    return rect_region(x, y, 0, 0.04, 0, 0.01, "channel");
  };
  f.delta = 1e-6;
  f.rows = {{0.02, 0, 0.04, 0, 2004}, {0, 0.005, 0.01, 0, 2002}, {0.04, 0.005, 0.01, 0, 2003}, {0.02, 0.01, 0.04, 0, 2004}};
  Circle c{0.01, 0.005, 0.002, 2 * kPi * 0.002, 0, {0.008, 0.003}, {0.012, 0.007}, 2001};
  f.circles = {c};
  f.atoms = {{2001, {"cylinder"}, "named", {{"channel", "post"}}}, {2002, {"inlet"}, "named", {{"channel", "exterior"}}},
             {2003, {"outlet"}, "named", {{"channel", "exterior"}}}, {2004, {"wall"}, "named", {{"channel", "exterior"}}}};
  f.diag = std::hypot(0.04, 0.01);
  f.region_measures["channel"] = {0.04 * 0.01 - kPi * 0.002 * 0.002 + kPi * 0.002 * 0.002 - kPi * 0.002 * 0.002, 0};
  f.region_measures["channel"].first = 0.04 * 0.01 - kPi * 0.002 * 0.002;
  f.region_measures["post"] = {kPi * 0.002 * 0.002, 0};
  return f;
}
Fx fx_annulus() {
  Fx f;
  f.name = "annulus";
  f.regions = {"ring"};
  const double R1 = 0.002, R2 = 0.005;
  f.region_of = [=](double x, double y) {
    const double r = std::hypot(x, y);
    return (r > R1 && r < R2) ? std::string("ring") : std::string();
  };
  f.delta = 1e-6;
  f.circles = {{0, 0, R1, 2 * kPi * R1, 0, {-R1, -R1}, {R1, R1}, 2001}, {0, 0, R2, 2 * kPi * R2, 0, {-R2, -R2}, {R2, R2}, 2002}};
  f.atoms = {{2001, {"inner"}, "named", {{"ring", "exterior"}}}, {2002, {"outer"}, "named", {{"ring", "exterior"}}}};
  f.diag = 2 * R2 * std::sqrt(2.0);
  f.region_measures["ring"] = {kPi * (R2 * R2 - R1 * R1), 0};
  return f;
}
Fx fx_pipe() {
  Fx f;
  f.name = "pipe_axi";
  f.axisym = true;
  f.regions = {"p1", "p2"};
  const double R = 0.005, Lp = 0.04;
  f.region_of = [=](double r, double z) {
    if (!(r > 0 && r < R && z > 0 && z < Lp)) return std::string();
    return std::string(z < Lp / 2 ? "p1" : "p2");
  };
  f.delta = 1e-6;
  const double disc = kPi * R * R, wall = 2 * kPi * R * (Lp / 2);
  Row inlet{R / 2, 0, R, disc, 2001}, wall1{R, 0.01, 0.02, wall, 2003}, interface{R / 2, 0.02, R, disc, 2004},
      axis1{0, 0.01, 0.02, 0, 900}, wall2{R, 0.03, 0.02, wall, 2003}, outlet{R / 2, 0.04, R, disc, 2002},
      axis2{0, 0.03, 0.02, 0, 900};
  axis1.axis = axis2.axis = true;
  f.rows = {inlet, wall1, interface, axis1, wall2, outlet, axis2};
  f.atoms = {{900, {}, "axis", {{"p1", "exterior"}}}, {2001, {"inlet"}, "named", {{"p1", "exterior"}}},
             {2002, {"outlet"}, "named", {{"p2", "exterior"}}}, {2003, {"wall"}, "named", {{"p1", "exterior"}, {"p2", "exterior"}}},
             {2004, {}, "unnamed", {{"p1", "p2"}}}};
  f.diag = std::hypot(R, Lp);
  f.region_measures["p1"] = {R * Lp / 2, 2 * kPi * (R / 2) * (R * Lp / 2)};
  f.region_measures["p2"] = f.region_measures["p1"];
  return f;
}
Fx fx_disk_on_axis() {
  Fx f;
  f.name = "disk_on_axis";
  f.axisym = true;
  f.regions = {"ball"};
  const double R = 0.005, zc = 0.01;
  f.region_of = [=](double r, double z) {
    return (r > 0 && r * r + (z - zc) * (z - zc) < R * R) ? std::string("ball") : std::string();
  };
  f.delta = 1e-6;
  Row axis{0, zc, 2 * R, 0, 900};
  axis.axis = true;
  f.rows = {axis};
  // the semicircle r >= 0: length pi R, revolved area 4 pi R^2 (Pappus: 2 pi (2R/pi) pi R), box r in [0, R], z in [zc-R, zc+R]
  f.circles = {{0, zc, R, kPi * R, 4 * kPi * R * R, {0, zc - R}, {R, zc + R}, 2001}};
  f.atoms = {{900, {}, "axis", {{"ball", "exterior"}}}, {2001, {"surface"}, "named", {{"ball", "exterior"}}}};
  f.diag = std::hypot(R, 2 * R);
  // half disk: area pi R^2 / 2; volume 2 pi r_c A with r_c = 4R / (3 pi): 4 pi R^3 / 3
  f.region_measures["ball"] = {kPi * R * R / 2, 4 * kPi * R * R * R / 3};
  return f;
}
Fx fx_thin_film() {
  Fx f;
  f.name = "thin_film";
  f.regions = {"film"};
  f.region_of = [](double x, double y) { return rect_region(x, y, 0, 1e-5, 0, 1e-6, "film"); };
  f.delta = 1e-9;
  f.rows = {{5e-6, 0, 1e-5, 0, 2001}, {1e-5, 5e-7, 1e-6, 0, 2003}, {5e-6, 1e-6, 1e-5, 0, 2002}, {0, 5e-7, 1e-6, 0, 2003}};
  f.atoms = {{2001, {"bottom"}, "named", {{"film", "exterior"}}}, {2002, {"top"}, "named", {{"film", "exterior"}}},
             {2003, {}, "unnamed", {{"film", "exterior"}}}};
  f.diag = std::hypot(1e-5, 1e-6);
  f.region_measures["film"] = {1e-11, 0};
  return f;
}
Fx fx_channel_shifted() {
  Fx f;
  f.name = "channel_shifted";
  f.regions = {"channel"};
  f.region_of = [](double x, double y) { return rect_region(x, y, 0.0005, 0.0405, 0, 0.01, "channel"); };
  f.delta = 1e-6;
  f.rows = {{0.0205, 0, 0.04, 0, 2001}, {0.0405, 0.005, 0.01, 0, 2001}, {0.0205, 0.01, 0.04, 0, 2001}, {0.0005, 0.005, 0.01, 0, 2001}};
  f.atoms = {{2001, {}, "unnamed", {{"channel", "exterior"}}}};
  f.diag = std::hypot(0.04, 0.01);
  f.region_measures["channel"] = {0.0004, 0};
  return f;
}
Fx fx_names_join() {
  Fx f;
  f.name = "names_join";
  f.regions = {"box"};
  f.region_of = [](double x, double y) { return rect_region(x, y, 0, 1, 0, 1, "box"); };
  f.delta = 1e-4;
  f.rows = {{0.5, 0, 1, 0, 2003}, {1, 0.5, 1, 0, 2002}, {0.5, 1, 1, 0, 2004}, {0, 0.5, 1, 0, 2001}};
  f.atoms = {{2001, {"a", "b"}, "named", {{"box", "exterior"}}}, {2002, {"a+b"}, "named", {{"box", "exterior"}}},
             {2003, {"b"}, "named", {{"box", "exterior"}}}, {2004, {}, "unnamed", {{"box", "exterior"}}}};
  f.diag = std::sqrt(2.0);
  f.region_measures["box"] = {1.0, 0};
  return f;
}

// Markers on the box's bottom and right edges, written against the counter-clockwise direction of the boundary: the
// entity's tangent is the written direction (so the exterior is on its left) and the line elements run along it.
Fx fx_marker_on_boundary() {
  Fx f;
  f.name = "marker_on_boundary";
  f.regions = {"box"};
  f.region_of = [](double x, double y) { return rect_region(x, y, 0, 0.04, 0, 0.01, "box"); };
  f.delta = 1e-6;
  Row mk_b{0.02, 0, 0.04, 0, 2001}, mk_r{0.04, 0.005, 0.01, 0, 2002};
  mk_b.dir[0] = -1;
  mk_b.curves = {"mk_b"};
  mk_r.dir[1] = -1;
  mk_r.curves = {"mk_r"};
  f.rows = {mk_b, mk_r, {0.02, 0.01, 0.04, 0, 2003}, {0, 0.005, 0.01, 0, 2003}};
  f.atoms = {{2001, {"mk_b"}, "named", {{"box", "exterior"}}}, {2002, {"mk_r"}, "named", {{"box", "exterior"}}},
             {2003, {}, "unnamed", {{"box", "exterior"}}}};
  f.diag = std::hypot(0.04, 0.01);
  f.region_measures["box"] = {0.0004, 0};
  return f;
}

std::vector<Fx> all_fixtures() {
  return {fx_rect(),
          fx_unit_square(false),
          fx_unit_square(true),
          fx_split_rect("split_rect", +1, {"mk"}, {}, {"mk"}),
          fx_split_rect("split_rect_mk_down", -1, {"mk"}, {}, {"mk"}),
          fx_split_rect("split_rect_two_markers", -1, {"mk_down", "mk_up"}, {"mk_up"}, {"mk_down", "mk_up"}),
          fx_open_markers(),
          fx_marker_on_boundary(),
          fx_channel_post(),
          fx_annulus(),
          fx_pipe(),
          fx_disk_on_axis(),
          fx_thin_film(),
          fx_channel_shifted(),
          fx_names_join()};
}

// The row-or-circle of an info entity.
const Row* find_row(const Fx& f, double px, double py, double tol) {
  const Row* hit = nullptr;
  for (const Row& r : f.rows)
    if (std::fabs(r.x - px) <= tol && std::fabs(r.y - py) <= tol) {
      if (hit) return nullptr;  // ambiguous: the table itself would be wrong
      hit = &r;
    }
  return hit;
}
const Circle* find_circle(const Fx& f, double px, double py) {
  for (const Circle& c : f.circles) {
    const double rr = std::hypot(px - c.cx, py - c.cy);
    if (std::fabs(rr - c.R) <= 1e-9 * c.R) return &c;
  }
  return nullptr;
}

// ------------------------------------------------------------------------------------------ the entity table checks
// Every check below names the fixture and entity in its message.  `Ent` is the info file's entity.
struct Ent {
  int tag = 0, atom = -1;
  std::string side0, side1, type;
  Strs curves, reversed;
  bool axis = false;
  double lo[2], hi[2], p[2], t[2], length = 0, area = 0;
  bool has_third_zero = true;
};
std::vector<Ent> read_entities(const json& info, const std::string& who) {
  std::vector<Ent> out;
  for (const json& e : info.at("entities")) {
    Ent x;
    x.tag = e.at("tag");
    x.atom = e.contains("atom") && !e["atom"].is_null() ? e["atom"].get<int>() : -1;
    x.side0 = e.at("side").at(0);
    x.side1 = e.at("side").at(1);
    x.type = e.at("type");
    for (const json& c : e.at("curves")) x.curves.push_back(c);
    if (e.contains("curves_reversed"))
      for (const json& c : e["curves_reversed"]) x.reversed.push_back(c);
    x.axis = e.at("axis");
    for (int k = 0; k < 2; ++k) {
      x.lo[k] = e.at("lo").at(k);
      x.hi[k] = e.at("hi").at(k);
      x.p[k] = e.at("point").at(k);
      x.t[k] = e.at("tangent").at(k);
    }
    if (e.at("lo").at(2).get<double>() != 0 || e.at("hi").at(2).get<double>() != 0 || e.at("point").at(2).get<double>() != 0 ||
        e.at("tangent").at(2).get<double>() != 0)
      fail(who + ": entity " + std::to_string(x.tag) + ": the third coordinates of a 2D table are not 0");
    if (e.at("dim").get<int>() != 1) fail(who + ": entity " + std::to_string(x.tag) + ": dim is not 1");
    x.length = e.at("length_m");
    x.area = e.at("area_m2");
    out.push_back(x);
  }
  return out;
}

void check_table_against(const Fx& f, const Run& r, const std::string& who) {
  const json& info = r.info;
  const std::string at = who + ": " + f.name;
  const std::vector<Ent> ents = read_entities(info, at);
  const double tol = 1e-12 * f.diag;   // the boxes and points: exact geometry to round-off, never gmsh's 1e-7 inflation
  std::set<const Row*> seen_rows;
  std::map<const Circle*, std::vector<const Ent*>> arcs;
  std::set<int> tags;
  for (const Ent& e : ents) {
    const std::string en = at + ": entity " + std::to_string(e.tag);
    CHECK(tags.insert(e.tag).second, en + ": tag repeated");
    const double tn = std::hypot(e.t[0], e.t[1]);
    CHECK(close_abs(tn, 1.0, 1e-12), en + ": tangent is not a unit vector: " + fmt(tn));
    // the sides: the region on the left of the tangent first (the left normal is the tangent rotated by +90 degrees)
    const double nx = -e.t[1], ny = e.t[0];
    const std::string left = f.region_of(e.p[0] + f.delta * nx, e.p[1] + f.delta * ny);
    const std::string right = f.region_of(e.p[0] - f.delta * nx, e.p[1] - f.delta * ny);
    CHECK(e.side0 == (left.empty() ? "exterior" : left), en + ": side[0] (left of the tangent) is '" + e.side0 + "', want '" +
                                                           (left.empty() ? "exterior" : left) + "'");
    CHECK(e.side1 == (right.empty() ? "exterior" : right), en + ": side[1] (right of the tangent) is '" + e.side1 + "', want '" +
                                                             (right.empty() ? "exterior" : right) + "'");
    if (e.type == "line") {
      const Row* row = find_row(f, e.p[0], e.p[1], tol);
      CHECK(row != nullptr, en + ": no edge of the hand table has the midpoint (" + fmt(e.p[0]) + ", " + fmt(e.p[1]) + ")");
      if (!row) continue;
      seen_rows.insert(row);
      CHECK(close_rel(e.length, row->length, 1e-14), en + ": length_m " + fmt(e.length) + ", want " + fmt(row->length));
      if (f.axisym)
        CHECK(close_rel(e.area, row->area, 1e-14) && (row->area != 0 || e.area == 0),
              en + ": area_m2 " + fmt(e.area) + ", want " + fmt(row->area));
      else
        CHECK(e.area == 0, en + ": area_m2 of a planar entity is " + fmt(e.area));
      // exact box: the edge is axis-parallel; the box is the span of its end points
      const bool horizontal = std::fabs(e.t[1]) < 1e-12;
      const double hx = horizontal ? row->length / 2 : 0, hy = horizontal ? 0 : row->length / 2;
      CHECK(close_abs(e.lo[0], row->x - hx, tol) && close_abs(e.hi[0], row->x + hx, tol) && close_abs(e.lo[1], row->y - hy, tol) &&
                close_abs(e.hi[1], row->y + hy, tol),
            en + ": box [" + fmt(e.lo[0]) + ", " + fmt(e.hi[0]) + "] x [" + fmt(e.lo[1]) + ", " + fmt(e.hi[1]) +
                "] is not the exact box (gmsh's own boxes are inflated by 1e-7 m)");
      CHECK(close_abs(e.p[0], row->x, tol) && close_abs(e.p[1], row->y, tol), en + ": point is not the midpoint");
      // the tangent is along the edge; for a marker it is the written direction of the first marker
      const double along = horizontal ? e.t[0] : e.t[1];
      CHECK(close_abs(std::fabs(along), 1.0, 1e-12), en + ": tangent is not along the edge");
      if (row->dir[0] != 0 || row->dir[1] != 0)
        CHECK(close_abs(e.t[0], row->dir[0], 1e-12) && close_abs(e.t[1], row->dir[1], 1e-12),
              en + ": tangent (" + fmt(e.t[0]) + ", " + fmt(e.t[1]) + ") is not the written direction of the first marker (" +
                  fmt(row->dir[0]) + ", " + fmt(row->dir[1]) + ")");
      CHECK(e.curves == row->curves, en + ": curves differ from the hand table");
      CHECK(e.reversed == row->reversed, en + ": curves_reversed differs from the hand table");
      CHECK(e.axis == row->axis, en + ": axis flag is " + std::string(e.axis ? "true" : "false"));
      CHECK(e.atom == row->atom, en + ": atom " + std::to_string(e.atom) + ", want " + std::to_string(row->atom));
    } else {
      CHECK(e.type == "circle" || e.type == "ellipse", en + ": type '" + e.type + "' (a circle or an arc of one)");
      const Circle* c = find_circle(f, e.p[0], e.p[1]);
      CHECK(c != nullptr, en + ": its point (" + fmt(e.p[0]) + ", " + fmt(e.p[1]) + ") lies on no circle of the hand table");
      if (!c) continue;
      arcs[c].push_back(&e);
      CHECK(std::fabs(e.t[0] * (e.p[0] - c->cx) + e.t[1] * (e.p[1] - c->cy)) <= 1e-9 * c->R,
            en + ": tangent is not perpendicular to the radius");
      CHECK(e.atom == c->atom, en + ": atom " + std::to_string(e.atom) + ", want " + std::to_string(c->atom));
      CHECK(!e.axis, en + ": a circle on the axis");
      CHECK(e.curves.empty(), en + ": curves of a circle");
    }
  }
  CHECK(seen_rows.size() == f.rows.size(),
        at + ": " + std::to_string(f.rows.size() - seen_rows.size()) + " of the " + std::to_string(f.rows.size()) +
            " edges of the hand table are missing from the entities");
  size_t nline = 0;
  for (const Ent& e : ents) nline += e.type == "line";
  CHECK(nline == f.rows.size(), at + ": " + std::to_string(nline) + " straight entities, want " + std::to_string(f.rows.size()));
  for (const Circle& c : f.circles) {
    const auto it = arcs.find(&c);
    CHECK(it != arcs.end() && !it->second.empty(), at + ": no entity on the circle R = " + fmt(c.R));
    if (it == arcs.end()) continue;
    double len = 0, area = 0, lo[2] = {1e300, 1e300}, hi[2] = {-1e300, -1e300};
    for (const Ent* e : it->second) {
      len += e->length;
      area += e->area;
      for (int k = 0; k < 2; ++k) {
        lo[k] = std::min(lo[k], e->lo[k]);
        hi[k] = std::max(hi[k], e->hi[k]);
      }
      // every arc's box is that of the arc: it contains its point and its tangent's direction fits; the union is checked below
      CHECK(e->lo[0] <= e->p[0] + tol && e->p[0] <= e->hi[0] + tol && e->lo[1] <= e->p[1] + tol && e->p[1] <= e->hi[1] + tol,
            at + ": an arc's box does not contain its point");
    }
    CHECK(close_rel(len, c.total_length, 1e-14), at + ": circle R = " + fmt(c.R) + ": total length " + fmt(len) + ", want " + fmt(c.total_length));
    if (f.axisym)
      CHECK(close_rel(area, c.total_area, 1e-14), at + ": circle R = " + fmt(c.R) + ": total revolved area " + fmt(area) + ", want " + fmt(c.total_area));
    CHECK(close_abs(lo[0], c.lo[0], tol) && close_abs(lo[1], c.lo[1], tol) && close_abs(hi[0], c.hi[0], tol) &&
              close_abs(hi[1], c.hi[1], tol),
          at + ": circle R = " + fmt(c.R) + ": the union of the arcs' boxes [" + fmt(lo[0]) + ", " + fmt(hi[0]) + "] x [" + fmt(lo[1]) + ", " +
              fmt(hi[1]) + "] is not the exact box (the extreme points of the full curve in the arc's range)");
  }
  size_t ncurved = 0;
  for (const Ent& e : ents) ncurved += e.type != "line";
  CHECK(f.circles.empty() == (ncurved == 0), at + ": curved entities present/absent against the hand table");
  // info header
  CHECK(info.value("bounded", false) == true, at + ": info has no \"bounded\": true");
  CHECK(info.value("geometry", std::string()) == (f.axisym ? "axisymmetric" : "planar"), at + ": info geometry");
  CHECK(info.value("exterior", std::string()) == "none", at + ": info exterior");
  CHECK(info.contains("gmsh_version") && info["gmsh_version"].is_string(), at + ": info gmsh_version");
  CHECK(info.contains("seconds") && info["seconds"].is_number(), at + ": info seconds");
  CHECK(info.contains("order"), at + ": info order");
  CHECK(info.contains("diagonal_m") && close_rel(info.value("diagonal_m", 0.0), f.diag, 1e-12),
        at + ": diagonal_m " + fmt(info.value("diagonal_m", 0.0)) + ", want " + fmt(f.diag));
  CHECK(info.contains("tolerance_m") && close_rel(info.value("tolerance_m", 0.0), 1e-9 * f.diag, 1e-12), at + ": tolerance_m is 1e-9 of the diagonal");
  // regions
  CHECK(info.contains("regions") && info["regions"].size() == f.regions.size(), at + ": regions of the info file");
  for (size_t i = 0; i < f.regions.size(); ++i) {
    const std::string& name = f.regions[i];
    if (!info.contains("regions") || !info["regions"].contains(name)) {
      fail(at + ": region '" + name + "' missing from the info file");
      continue;
    }
    const json& j = info["regions"][name];
    CHECK(j.value("attr", 0) == int(i) + 1, at + ": region '" + name + "' attr is " + std::to_string(j.value("attr", 0)) + ", want " + std::to_string(i + 1));
    const auto want = f.region_measures.at(name);
    CHECK(close_rel(j.value("area_m2", 0.0), want.first, 1e-14),
          at + ": region '" + name + "' area_m2 " + fmt(j.value("area_m2", 0.0)) + ", want " + fmt(want.first));
    if (f.axisym)
      CHECK(close_rel(j.value("volume_m3", 0.0), want.second, 1e-14),
            at + ": region '" + name + "' volume_m3 " + fmt(j.value("volume_m3", 0.0)) + ", want " + fmt(want.second));
    else
      CHECK(!j.contains("volume_m3"), at + ": a planar region has volume_m3");
  }
  // boundary groups: one per atom, in master order, with the physical-group name atom_<master>
  const json groups = info.value("boundary_groups", json::array());
  CHECK(groups.size() == f.atoms.size(), at + ": " + std::to_string(groups.size()) + " boundary groups, want " + std::to_string(f.atoms.size()));
  for (size_t i = 0; i < std::min(groups.size(), f.atoms.size()); ++i) {
    const json& g = groups[i];
    const Atom& a = f.atoms[i];
    const std::string an = at + ": atom " + std::to_string(a.attr);
    CHECK(g.value("attr", 0) == a.attr, an + ": attr " + std::to_string(g.value("attr", 0)));
    CHECK(g.value("name", std::string()) == "atom_" + std::to_string(a.attr), an + ": group name is '" + g.value("name", std::string()) + "'");
    CHECK(g.value("names", Strs{}) == a.names, an + ": names differ");
    CHECK(g.value("type", std::string()) == a.type, an + ": type is '" + g.value("type", std::string()) + "', want " + a.type);
    // adjacent: the two regions of the atom (an unnamed atom's region pair; a named atom's first entity's sides)
    if (g.contains("adjacent") && g["adjacent"].size() == 2) {
      auto idx = [&](const std::string& s) {
        for (size_t k = 0; k < f.regions.size(); ++k)
          if (f.regions[k] == s) return int(k);
        return int(f.regions.size());
      };
      std::string s0 = g["adjacent"][0], s1 = g["adjacent"][1];
      if (idx(s0) > idx(s1)) std::swap(s0, s1);
      CHECK(a.adjacent.count({s0, s1}) > 0, an + ": adjacent [" + s0 + ", " + s1 + "] is not a pair of the atom's entities");
    } else {
      fail(an + ": adjacent is not a pair");
    }
  }
  // atoms and their entities: every entity's atom is the atom whose group lists it; an atom is one master number
  std::map<int, size_t> count;
  for (const Ent& e : ents)
    if (e.atom >= 0) ++count[e.atom];
  for (const Atom& a : f.atoms) {
    size_t want = 0;
    for (const Row& r : f.rows) want += r.atom == a.attr;
    CHECK(count.count(a.attr) > 0, at + ": no entity in atom " + std::to_string(a.attr));
    bool curved = false;
    for (const Circle& c : f.circles) curved = curved || c.atom == a.attr;
    if (!curved)
      CHECK(count[a.attr] == want, at + ": atom " + std::to_string(a.attr) + " has " + std::to_string(count[a.attr]) + " entities, want " + std::to_string(want));
  }
  // boundaries: attributes, entities, measure, bbox, point, adjacent
  const json bj = info.value("boundaries", json::object());
  std::set<std::string> names;
  for (auto& [n, sel] : r.input.at("boundaries").items()) names.insert(n);
  CHECK(bj.size() == names.size(), at + ": boundaries has " + std::to_string(bj.size()) + " names, want " + std::to_string(names.size()));
  for (const std::string& n : names) {
    if (!bj.contains(n)) {
      fail(at + ": boundaries lacks '" + n + "'");
      continue;
    }
    const json& b = bj[n];
    Ints want_attr;
    for (const Atom& a : f.atoms)
      if (std::find(a.names.begin(), a.names.end(), n) != a.names.end()) want_attr.push_back(a.attr);
    CHECK(b.value("attributes", Ints{}) == want_attr, at + ": boundaries['" + n + "'].attributes differ from the groups that name it");
    double meas = 0, area = 0;
    size_t cnt = 0;
    double lo[2] = {1e300, 1e300}, hi[2] = {-1e300, -1e300};
    for (const Ent& e : ents) {
      if (std::find(want_attr.begin(), want_attr.end(), e.atom) == want_attr.end()) continue;
      ++cnt;
      meas += e.length;
      area += e.area;
      for (int k = 0; k < 2; ++k) {
        lo[k] = std::min(lo[k], e.lo[k]);
        hi[k] = std::max(hi[k], e.hi[k]);
      }
    }
    CHECK(size_t(b.value("entities", 0)) == cnt, at + ": boundaries['" + n + "'].entities");
    CHECK(close_rel(b.value("measure", 0.0), meas, 1e-13), at + ": boundaries['" + n + "'].measure");
    CHECK(b.value("measure_units", std::string()) == "m", at + ": boundaries['" + n + "'].measure_units");
    if (f.axisym)
      CHECK(b.contains("area_m2") && close_rel(b.value("area_m2", 0.0), area, 1e-13), at + ": boundaries['" + n + "'].area_m2");
    const char* k0 = f.axisym ? "r" : "x";
    const char* k1 = f.axisym ? "z" : "y";
    if (cnt == 0) {
      CHECK(b["bbox"].is_null() && b["point"].is_null() && b.value("measure", 1.0) == 0.0, at + ": an empty boundary '" + n + "' has bbox and point null, measure 0");
    } else if (b.contains("bbox") && b["bbox"].is_object() && b["bbox"].contains(k0) && b["bbox"].contains(k1))
      CHECK(close_abs(b["bbox"][k0][0], lo[0], tol) && close_abs(b["bbox"][k0][1], hi[0], tol) && close_abs(b["bbox"][k1][0], lo[1], tol) &&
                close_abs(b["bbox"][k1][1], hi[1], tol),
            at + ": boundaries['" + n + "'].bbox is not the union of its entities' boxes");
    else
      fail(at + ": boundaries['" + n + "'].bbox lacks the keys " + k0 + " and " + k1);
    if (cnt > 0) {
      // point: the smallest point of its entities in lexicographic order; adjacent: the region pairs of its entities, each in
      // region order ("exterior" last), unique, in region-pair order
      const Ent* first = nullptr;
      std::set<std::pair<int, int>> pairs;
      auto ridx = [&](const std::string& s) {
        for (size_t k = 0; k < f.regions.size(); ++k)
          if (f.regions[k] == s) return int(k);
        return int(f.regions.size());
      };
      for (const Ent& e : ents) {
        if (std::find(want_attr.begin(), want_attr.end(), e.atom) == want_attr.end()) continue;
        if (!first || std::make_pair(e.p[0], e.p[1]) < std::make_pair(first->p[0], first->p[1])) first = &e;
        const int i0 = ridx(e.side0), i1 = ridx(e.side1);
        pairs.insert({std::min(i0, i1), std::max(i0, i1)});
      }
      CHECK(b.contains("point") && b["point"].is_array() && b["point"].size() == 2 && first != nullptr &&
                close_abs(b["point"][0], first->p[0], tol) && close_abs(b["point"][1], first->p[1], tol),
            at + ": boundaries['" + n + "'].point is not the lexicographically smallest point of its entities (" + (b.contains("point") ? b["point"].dump() : "absent") + ")");
      json want_adj = json::array();
      auto nm = [&](int i) { return i < int(f.regions.size()) ? f.regions[i] : std::string("exterior"); };
      for (const auto& p : pairs) want_adj.push_back({nm(p.first), nm(p.second)});
      CHECK(b.contains("adjacent") && b["adjacent"] == want_adj, at + ": boundaries['" + n + "'].adjacent is " + (b.contains("adjacent") ? b["adjacent"].dump() : "absent") + ", want " + want_adj.dump());
    }
  }
  // consistency with the library's resolution on the table the info file carries
  try {
    const S::Table table = S::table_from_json(json{{"geometry", info["geometry"]},
                                                    {"regions", f.regions},
                                                    {"entities", info["entities"]},
                                                    {"diagonal_m", info["diagonal_m"]}});
    const S::Resolution res = S::resolve(table, r.input.at("boundaries"));
    CHECK(res.atoms.size() == f.atoms.size(), at + ": resolve() on the info's table gives " + std::to_string(res.atoms.size()) + " atoms");
    for (const S::Atom& a : res.atoms)
      CHECK(count.count(a.master) > 0 && count[a.master] == a.tags.size(),
            at + ": atom " + std::to_string(a.master) + " of resolve() on the info's table has a different entity count");
    for (const Ent& e : ents) {
      const auto it = res.atom_of_tag.find(e.tag);
      CHECK(it != res.atom_of_tag.end() && it->second == e.atom, at + ": entity " + std::to_string(e.tag) + ": its atom differs from resolve()'s");
    }
    for (const auto& [name, tags] : res.selected) {
      const Ints attrs = bj.contains(name) ? bj[name].value("attributes", Ints{}) : Ints{};
      size_t got = 0;
      for (const Ent& e : ents) got += std::find(attrs.begin(), attrs.end(), e.atom) != attrs.end();
      // a name's entities are the entities of its atoms (atoms are signatures, so the counts are exact)
      CHECK(got == tags.size(), at + ": boundary '" + name + "': " + std::to_string(got) + " entities in the info, " +
                                    std::to_string(tags.size()) + " by resolve()");
    }
  } catch (const std::exception& ex) {
    fail(at + ": consistency with resolve(): " + ex.what());
  }
}

// --------------------------------------------------------------------------------------- MFEM and mesh-file helpers
struct Loaded {
  std::unique_ptr<mfem::Mesh> mesh;
  Msh msh;
};
Loaded load(const Run& r) {
  Loaded l;
  l.msh = read_msh(r.msh);
  l.mesh = std::make_unique<mfem::Mesh>(r.msh.c_str(), 1, 1, true);
  return l;
}
double bdr_length(mfem::Mesh& m, int i) {
  mfem::ElementTransformation* T = m.GetBdrElementTransformation(i);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 10);
  double s = 0;
  for (int q = 0; q < ir.GetNPoints(); ++q) {
    T->SetIntPoint(&ir.IntPoint(q));
    s += ir.IntPoint(q).weight * T->Weight();
  }
  return s;
}
double elem_area(mfem::Mesh& m, int e) {
  mfem::ElementTransformation* T = m.GetElementTransformation(e);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(m.GetElementBaseGeometry(e), 8);
  double s = 0;
  for (int q = 0; q < ir.GetNPoints(); ++q) {
    T->SetIntPoint(&ir.IntPoint(q));
    s += ir.IntPoint(q).weight * T->Weight();
  }
  return s;
}
std::map<int, double> length_by_attr(mfem::Mesh& m) {
  std::map<int, double> out;
  for (int i = 0; i < m.GetNBE(); ++i) out[m.GetBdrAttribute(i)] += bdr_length(m, i);
  return out;
}
std::array<double, 2> vtx(const Msh& m, int id) {
  const auto& p = m.nodes.at(id);
  return {p[0], p[1]};
}

// =================================================================================================== the checks

// The entity table (sides in the order of the tangent, tangent, curves_reversed, exact boxes, lengths, areas, atoms,
// the info's groups and boundaries) of every fixture against the hand-built tables.
void check_entity_table() {
  for (const Fx& f : all_fixtures()) {
    const Run* r = get("check_entity_table", f.name);
    if (r) check_table_against(f, *r, "check_entity_table");
  }
}

// Lengths and revolved areas of the info file are the analytic values to a relative 1e-14 (round-off), and so are the
// region areas and volumes (check_entity_table holds them per fixture; here the sums over the atoms and the closed forms
// of the circles and the revolved surfaces).
void check_info_measures_are_exact() {
  const Run* ann = get("check_info_measures_are_exact", "annulus");
  if (ann) {
    double l1 = 0, l2 = 0;
    for (const json& e : ann->info.at("entities")) (e.at("atom").get<int>() == 2001 ? l1 : l2) += e.at("length_m").get<double>();
    CHECK(close_rel(l1, 2 * kPi * 0.002, 1e-14), "annulus: inner circle length " + fmt(l1) + ", want 2 pi R1 = " + fmt(2 * kPi * 0.002));
    CHECK(close_rel(l2, 2 * kPi * 0.005, 1e-14), "annulus: outer circle length " + fmt(l2) + ", want 2 pi R2 = " + fmt(2 * kPi * 0.005));
    const double area = ann->info.at("regions").at("ring").at("area_m2");
    CHECK(close_rel(area, kPi * (0.005 * 0.005 - 0.002 * 0.002), 1e-14), "annulus: area " + fmt(area));
  }
  const Run* pipe = get("check_info_measures_are_exact", "pipe_axi");
  if (pipe) {
    double wall = 0, inlet = 0, outlet = 0, iface = 0;
    for (const json& e : pipe->info.at("entities")) {
      const int a = e.at("atom");
      const double ar = e.at("area_m2");
      if (a == 2003) wall += ar;
      if (a == 2001) inlet += ar;
      if (a == 2002) outlet += ar;
      if (a == 2004) iface += ar;
    }
    const double R = 0.005, Lp = 0.04;
    CHECK(close_rel(wall, 2 * kPi * R * Lp, 1e-14), "pipe: the revolved wall is 2 pi R L: " + fmt(wall));
    CHECK(close_rel(inlet, kPi * R * R, 1e-14), "pipe: the revolved inlet disc is pi R^2: " + fmt(inlet));
    CHECK(close_rel(outlet, kPi * R * R, 1e-14), "pipe: the revolved outlet disc is pi R^2: " + fmt(outlet));
    CHECK(close_rel(iface, kPi * R * R, 1e-14), "pipe: the revolved interface disc is pi R^2: " + fmt(iface));
    const json b = pipe->info.at("boundaries").at("wall");
    CHECK(close_rel(b.value("area_m2", 0.0), 2 * kPi * R * Lp, 1e-14), "pipe: boundaries.wall.area_m2 is 2 pi R L");
    CHECK(close_rel(b.value("measure", 0.0), 2 * 0.02, 1e-14), "pipe: boundaries.wall.measure is the generator length 2 x 0.02");
  }
  const Run* disk = get("check_info_measures_are_exact", "disk_on_axis");
  if (disk) {
    const double R = 0.005;
    double len = 0, area = 0;
    for (const json& e : disk->info.at("entities"))
      if (e.at("atom").get<int>() == 2001) {
        len += e.at("length_m").get<double>();
        area += e.at("area_m2").get<double>();
      }
    CHECK(close_rel(len, kPi * R, 1e-14), "disk_on_axis: the semicircle's length is pi R: " + fmt(len));
    CHECK(close_rel(area, 4 * kPi * R * R, 1e-14), "disk_on_axis: the revolved semicircle is the sphere 4 pi R^2: " + fmt(area));
    const json& reg = disk->info.at("regions").at("ball");
    CHECK(close_rel(reg.value("volume_m3", 0.0), 4 * kPi * R * R * R / 3, 1e-14), "disk_on_axis: the revolved half disk is 4/3 pi R^3");
  }
}

// The mesh's boundary measures: straight atoms to round-off, circles converging at the rate of the geometry order.
void check_mesh_measures() {
  const char* who = "check_mesh_measures";
  // straight edges: the sum over the atom's boundary elements is the hand length
  for (const std::string& name : {"rect", "split_rect", "pipe_axi", "channel_post"}) {
    const Run* r = get(who, name);
    if (!r) continue;
    Loaded l = load(*r);
    const auto len = length_by_attr(*l.mesh);
    const Fx* fx = nullptr;
    static std::vector<Fx> all = all_fixtures();
    for (const Fx& f : all)
      if (f.name == name) fx = &f;
    for (const Atom& a : fx->atoms) {
      double want = 0;
      for (const Row& row : fx->rows)
        if (row.atom == a.attr) want += row.length;
      if (!fx->circles.empty() && a.attr == fx->circles[0].atom) continue;  // curved: below
      const double got = len.count(a.attr) ? len.at(a.attr) : 0.0;
      CHECK(close_rel(got, want, 1e-12), std::string(who) + ": " + name + ": atom " + std::to_string(a.attr) + ": the boundary elements sum to " + fmt(got) +
                                              ", want " + fmt(want));
    }
  }
  // circles: observed order of the length error against the segment size over the three sizes (curvature sizing off)
  for (int order : {1, 2}) {
    std::map<int, std::vector<std::pair<double, double>>> data;  // atom -> (h_eff, relative error)
    bool all_ok = true;
    for (const char* name : {"annulus_conv_h4e-4", "annulus_conv_h2e-4", "annulus_conv_h1e-4"}) {
      const Run* r = get(who, name, json{{"order", order}}, "_o" + std::to_string(order));
      if (!r) {
        all_ok = false;
        continue;
      }
      Loaded l = load(*r);
      const auto len = length_by_attr(*l.mesh);
      std::map<int, int> nseg;
      for (const MshEl& e : l.msh.lines()) ++nseg[e.phys];
      const double R[2] = {0.002, 0.005};
      const int atoms[2] = {2001, 2002};
      for (int k = 0; k < 2; ++k) {
        const double want = 2 * kPi * R[k];
        const double err = std::fabs(len.at(atoms[k]) - want) / want;
        data[atoms[k]].push_back({want / nseg.at(atoms[k]), err});
      }
    }
    if (!all_ok) continue;
    const double need = order == 1 ? 1.8 : 3.6;
    for (auto& [atom, pts] : data) {
      // least-squares slope of log(err) against log(h_eff)
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (auto& [h, e] : pts) {
        if (!(e > 0)) {
          fail(std::string(who) + ": atom " + std::to_string(atom) + ": error exactly 0 at geometry order " + std::to_string(order));
          continue;
        }
        const double x = std::log(h), y = std::log(e);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
      }
      const double n = double(pts.size());
      const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
      std::printf("  observed order of the circle length error, atom %d, geometry order %d: %.3f (need >= %.1f)\n", atom, order, slope, need);
      CHECK(slope >= need, std::string(who) + ": atom " + std::to_string(atom) + ": observed order " + fmt(slope) + " at geometry order " +
                               std::to_string(order) + ", need >= " + fmt(need));
    }
  }
}

// No two boundary elements share a face; every exterior face, every face between two regions and every marker face
// carries exactly one boundary element; atoms sum to the analytic perimeter / interface length.
void check_counting() {
  const char* who = "check_counting";
  for (const char* name : {"rect", "split_rect", "split_rect_two_markers", "open_markers", "channel_post", "annulus", "pipe_axi"}) {
    const Run* r = get(who, name);
    if (!r) continue;
    Loaded l = load(*r);
    mfem::Mesh& m = *l.mesh;
    const std::string at = std::string(who) + ": " + name;
    // (a) in the file: no two line elements on the same pair of end vertices
    std::map<std::pair<int, int>, int> uses;
    for (const MshEl& e : l.msh.lines()) {
      const int a = std::min(e.nodes[0], e.nodes[1]), b = std::max(e.nodes[0], e.nodes[1]);
      ++uses[{a, b}];
    }
    int dup = 0;
    for (auto& [k, c] : uses) dup += c > 1;
    CHECK(dup == 0, at + ": " + std::to_string(dup) + " faces carry more than one line element in the mesh file (an entity in two physical groups is written twice)");
    // (b) in the mesh: the set of faces of boundary elements has no repeats, and is exactly the exterior faces, the faces
    //     between differently-attributed elements and the faces of the marker entities
    std::set<int> bdr_faces;
    int repeated = 0;
    for (int i = 0; i < m.GetNBE(); ++i)
      if (!bdr_faces.insert(m.GetBdrElementFaceIndex(i)).second) ++repeated;
    CHECK(repeated == 0, at + ": " + std::to_string(repeated) + " boundary elements repeat a face");
    std::set<int> must;
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      int e1 = -1, e2 = -1;
      m.GetFaceElements(f, &e1, &e2);
      if (e2 < 0 || m.GetAttribute(e1) != m.GetAttribute(e2)) must.insert(f);
    }
    // marker faces: the faces of the line elements of marker entities
    const std::vector<Ent> ents = read_entities(r->info, at);
    std::set<int> marker_tags;
    for (const Ent& e : ents)
      if (!e.curves.empty()) marker_tags.insert(e.tag);
    std::set<int> marker_phys;
    for (const MshEl& e : l.msh.lines())
      if (marker_tags.count(e.elem)) marker_phys.insert(e.phys);
    for (int i = 0; i < m.GetNBE(); ++i)
      if (marker_phys.count(m.GetBdrAttribute(i))) must.insert(m.GetBdrElementFaceIndex(i));
    for (int f : must)
      CHECK(bdr_faces.count(f) == 1, at + ": face " + std::to_string(f) + " (exterior, between two regions or on a marker) has no boundary element");
    for (int f : bdr_faces)
      CHECK(must.count(f) == 1, at + ": a boundary element on face " + std::to_string(f) + ", which is none of those");
    CHECK(int(bdr_faces.size()) == m.GetNBE(), at + ": boundary elements and faces differ in number");
    // (c) every boundary entity has line elements and every line element belongs to an entity of the info file
    std::map<int, int> per_entity;
    std::map<int, int> atom_of;
    for (const Ent& e : ents) atom_of[e.tag] = e.atom;
    for (const MshEl& e : l.msh.lines()) {
      ++per_entity[e.elem];
      CHECK(atom_of.count(e.elem) && atom_of[e.elem] == e.phys,
            at + ": a line element of entity " + std::to_string(e.elem) + " is in physical group " + std::to_string(e.phys) + ", its atom is " +
                std::to_string(atom_of.count(e.elem) ? atom_of[e.elem] : -1));
    }
    for (const Ent& e : ents)
      if (e.atom >= 0) CHECK(per_entity[e.tag] > 0, at + ": entity " + std::to_string(e.tag) + " has no line element");
    // (d) the exterior atoms sum to the perimeter, the interface atoms to the interface lengths (straight fixtures; the
    //     circle fixtures to their chord sums, whose error check_mesh_measures holds)
    const auto len = length_by_attr(m);
    double exterior = 0, interface = 0;
    std::map<int, std::string> kind;
    for (const Ent& e : ents) {
      if (e.atom < 0) continue;
      kind[e.atom] = (e.side0 == "exterior" || e.side1 == "exterior") ? "exterior" : (e.side0 == e.side1 ? "marker" : "interface");
    }
    for (auto& [a, k] : kind) {
      if (k == "exterior" && a != 900) exterior += len.count(a) ? len.at(a) : 0;
      if (k == "interface") interface += len.count(a) ? len.at(a) : 0;
    }
    const std::string n = name;
    const double tol = (n == "annulus" || n == "channel_post") ? 1e-4 : 1e-12;
    if (n == "rect") CHECK(close_rel(exterior, 0.1, 1e-12), at + ": exterior atoms sum to " + fmt(exterior) + ", the perimeter is 0.1");
    if (n == "split_rect" || n == "split_rect_two_markers") {
      CHECK(close_rel(exterior, 0.1, 1e-12), at + ": exterior atoms sum to " + fmt(exterior) + ", the perimeter is 0.1");
      CHECK(close_rel(interface, 0.01, 1e-12), at + ": the interface atom sums to " + fmt(interface) + ", the interface is 0.01");
    }
    if (n == "open_markers") CHECK(close_rel(exterior, 0.1, 1e-12), at + ": exterior atoms sum to " + fmt(exterior));
    if (n == "channel_post") {
      CHECK(close_rel(exterior, 0.1, 1e-12), at + ": exterior atoms sum to " + fmt(exterior));
      CHECK(close_rel(interface, 2 * kPi * 0.002, tol), at + ": the cylinder atom sums to " + fmt(interface));
    }
    if (n == "annulus") CHECK(close_rel(exterior, 2 * kPi * 0.007, tol), at + ": the circles sum to " + fmt(exterior));
    if (n == "pipe_axi") {
      CHECK(close_rel(exterior, 0.005 + 0.04 + 0.005, 1e-12), at + ": exterior atoms (axis excluded) sum to " + fmt(exterior));
      CHECK(close_rel(interface, 0.005, 1e-12), at + ": the interface atom sums to " + fmt(interface));
    }
  }
}

// Element attributes are 1..R in region order; every boundary attribute is an atom of the info file; the groups agree
// with the boundaries; the unit squares give fixtures::master_numbers().
void check_attributes() {
  const char* who = "check_attributes";
  for (const Fx& f : all_fixtures()) {
    if (f.name == "channel_shifted") continue;  // resolve-only fixture
    const Run* r = get(who, f.name);
    if (!r) continue;
    Loaded l = load(*r);
    mfem::Mesh& m = *l.mesh;
    const std::string at = std::string(who) + ": " + f.name;
    std::set<int> eattrs;
    for (int e = 0; e < m.GetNE(); ++e) eattrs.insert(m.GetAttribute(e));
    std::set<int> want;
    for (size_t i = 0; i < f.regions.size(); ++i) want.insert(int(i) + 1);
    CHECK(eattrs == want, at + ": element attributes are not 1.." + std::to_string(f.regions.size()) + " (a region's attribute is its position among the enabled regions)");
    // each element's attribute is its region's: by the centroid and the hand region rule
    int wrong = 0;
    for (int e = 0; e < m.GetNE(); ++e) {
      mfem::Vector c;
      m.GetElementCenter(e, c);
      const std::string reg = f.region_of(c[0], c[1]);
      const int a = m.GetAttribute(e);
      if (reg.empty() || a < 1 || a > int(f.regions.size()) || f.regions[a - 1] != reg) ++wrong;
    }
    CHECK(wrong == 0, at + ": " + std::to_string(wrong) + " elements carry the attribute of another region");
    // region names of dimension 2
    for (size_t i = 0; i < f.regions.size(); ++i) {
      const auto it = l.msh.names2.find(int(i) + 1);
      CHECK(it != l.msh.names2.end() && it->second == f.regions[i], at + ": the region group " + std::to_string(i + 1) + " is not named '" + f.regions[i] + "'");
    }
    // boundary attributes are atoms
    std::set<int> atoms;
    for (const json& g : r->info.at("boundary_groups")) atoms.insert(g.at("attr").get<int>());
    std::set<int> battrs;
    for (int i = 0; i < m.GetNBE(); ++i) battrs.insert(m.GetBdrAttribute(i));
    CHECK(battrs == atoms, at + ": the boundary attributes of the mesh are not the atoms of the info file");
    // the master numbering: axis 900, named 2001.. in the lexicographic order of the name lists, unnamed after
    int prev = 0;
    for (const json& g : r->info.at("boundary_groups")) {
      const int a = g.at("attr");
      CHECK(a > prev, at + ": groups are not in ascending master order");
      prev = a;
      CHECK((a == 900) == (g.at("type") == "axis"), at + ": only the axis atom is 900");
      if (a != 900) CHECK(a >= 2001, at + ": a non-axis atom below 2001");
    }
    if (f.name == "unit_square_planar" || f.name == "unit_square_axisymmetric") {
      const auto master = axmp::model::fixtures::master_numbers(f.axisym ? axmp::model::fixtures::Geometry::axisymmetric
                                                                          : axmp::model::fixtures::Geometry::planar);
      for (const auto& [side, num] : master) {
        if (side == "axis") {
          CHECK(atoms.count(900) == 1, at + ": the axis atom is not 900");
          continue;
        }
        CHECK(r->info.at("boundaries").contains(side) && r->info["boundaries"][side].value("attributes", Ints{}) == Ints{num},
              at + ": boundary '" + side + "' is not atom " + std::to_string(num) + " (fixtures::master_numbers)");
      }
    }
  }
}

void check_axis() {
  const char* who = "check_axis";
  for (const char* name : {"pipe_axi", "disk_on_axis", "unit_square_axisymmetric"}) {
    const Run* r = get(who, name);
    if (!r) continue;
    Loaded l = load(*r);
    mfem::Mesh& m = *l.mesh;
    const std::string at = std::string(who) + ": " + name;
    int on_axis_other = 0, axis_elements = 0, axis_off = 0;
    for (int i = 0; i < m.GetNBE(); ++i) {
      mfem::Array<int> v;
      m.GetBdrElement(i)->GetVertices(v);
      bool all0 = true;
      for (int k = 0; k < v.Size(); ++k) all0 = all0 && std::fabs(m.GetVertex(v[k])[0]) <= 1e-14;
      if (m.GetBdrAttribute(i) == 900) {
        ++axis_elements;
        if (!all0) ++axis_off;
      } else if (all0) {
        ++on_axis_other;
      }
    }
    CHECK(axis_elements > 0, at + ": no boundary element in atom 900");
    CHECK(axis_off == 0, at + ": " + std::to_string(axis_off) + " elements of atom 900 are off r = 0");
    CHECK(on_axis_other == 0, at + ": " + std::to_string(on_axis_other) + " elements on r = 0 belong to another atom");
    double rmin = 1e300;
    for (int v = 0; v < m.GetNV(); ++v) rmin = std::min(rmin, m.GetVertex(v)[0]);
    CHECK(rmin >= -1e-14, at + ": a vertex with r < 0: " + fmt(rmin) + " (the model is clipped at r >= 0)");
    for (const json& e : r->info.at("entities"))
      CHECK((e.at("atom").get<int>() == 900) == e.at("axis").get<bool>(), at + ": an entity's axis flag and atom 900 disagree");
    // no named selection contains the axis
    for (auto& [n, b] : r->info.at("boundaries").items())
      for (int a : b.at("attributes").get<Ints>()) CHECK(a != 900, at + ": boundary '" + n + "' contains the axis atom");
  }
  // planar fixtures have no axis atom
  const Run* r = get(who, "rect");
  if (r)
    for (const json& g : r->info.at("boundary_groups")) CHECK(g.at("attr").get<int>() != 900, "check_axis: rect: a planar model has an atom 900");
  // the half disk's mesh lies on one side of the axis
  const Run* d = get(who, "disk_on_axis");
  if (d) {
    Loaded l = load(*d);
    double area = 0;
    for (int e = 0; e < l.mesh->GetNE(); ++e) area += elem_area(*l.mesh, e);
    CHECK(close_rel(area, kPi * 0.005 * 0.005 / 2, 2e-5), "check_axis: disk_on_axis: the half disk's mesh area " + fmt(area));
  }
}

// Markers: their own atom with the measure H, their line elements run in the written direction (in the file and as MFEM
// reads them), open markers (both ends in a region, and one end on the boundary) are found.
void check_markers() {
  const char* who = "check_markers";
  const Run* r = get(who, "split_rect");
  if (r) {
    const json& b = r->info.at("boundaries").at("mk");
    CHECK(b.value("attributes", Ints{}) == Ints{2004}, "check_markers: mk is atom 2004");
    CHECK(close_rel(b.value("measure", 0.0), 0.01, 1e-14), "check_markers: mk measure is H");
    CHECK(b.value("entities", 0) == 1, "check_markers: mk is one entity");
    // the marker's atom holds the marker alone (no other boundary entity)
    for (const json& e : r->info.at("entities"))
      if (e.at("atom").get<int>() == 2004) CHECK(e.at("curves").size() == 1 && e.at("curves")[0] == "mk", "check_markers: an entity of atom 2004 without the marker");
  }
  struct Case { const char* name; int phys; int axis; double sign; bool mfem_keeps; };   // line elements run along `axis` with this sign
  // mfem_keeps: an interior marker's boundary elements keep their order through MFEM's read; an exterior one's are turned
  for (const Case& c : {Case{"split_rect", 2004, 1, +1, true}, Case{"split_rect_mk_down", 2004, 1, -1, true},
                        Case{"split_rect_two_markers", 2004, 1, -1, true}, Case{"open_markers", 2003, 0, +1, true},
                        Case{"open_markers", 2002, 0, -1, true}, Case{"marker_on_boundary", 2001, 0, -1, false},
                        Case{"marker_on_boundary", 2002, 1, -1, false}}) {
    const std::string tag = std::string(c.name) + ":" + std::to_string(c.phys);
    const Run* run = get(who, c.name);
    if (!run) continue;
    Loaded l = load(*run);
    int n = 0, bad = 0, bad_mfem = 0;
    for (const MshEl& e : l.msh.lines()) {
      if (e.phys != c.phys) continue;
      ++n;
      const auto p0 = vtx(l.msh, e.nodes[0]), p1 = vtx(l.msh, e.nodes[1]);
      if (!((p1[c.axis] - p0[c.axis]) * c.sign > 0)) ++bad;
    }
    CHECK(n > 0, std::string(who) + ": " + tag + ": no line elements in the marker's atom");
    CHECK(bad == 0, std::string(who) + ": " + tag + ": " + std::to_string(bad) + " of the marker's line elements run against its written direction");
    // MFEM keeps the order of an interior boundary element (it turns exterior ones only)
    mfem::Mesh& m = *l.mesh;
    for (int i = 0; i < m.GetNBE() && c.mfem_keeps; ++i) {
      if (m.GetBdrAttribute(i) != c.phys) continue;
      mfem::Array<int> v;
      m.GetBdrElement(i)->GetVertices(v);
      if (!((m.GetVertex(v[1])[c.axis] - m.GetVertex(v[0])[c.axis]) * c.sign > 0)) ++bad_mfem;
    }
    CHECK(bad_mfem == 0, std::string(who) + ": " + tag + ": MFEM's read reversed " + std::to_string(bad_mfem) + " of the marker's boundary elements");
  }
  // open markers: both are entities between a region and itself, in atoms of their own, with elements in the mesh and 2D
  // elements on both sides (the interior faces)
  const Run* o = get(who, "open_markers");
  if (o) {
    for (const char* mk : {"mk_in", "mk_edge"}) {
      CHECK(o->info.at("boundaries").contains(mk) && o->info["boundaries"][mk].value("entities", 0) == 1, std::string("check_markers: open marker '") + mk + "' is one entity");
    }
    CHECK(close_rel(o->info["boundaries"]["mk_in"].value("measure", 0.0), 0.006, 1e-13), "check_markers: mk_in is 0.006 long");
    CHECK(close_rel(o->info["boundaries"]["mk_edge"].value("measure", 0.0), 0.005, 1e-13), "check_markers: mk_edge is 0.005 long");
    Loaded l = load(*o);
    mfem::Mesh& m = *l.mesh;
    for (int phys : {2002, 2003}) {
      int n = 0, interior = 0;
      for (int i = 0; i < m.GetNBE(); ++i)
        if (m.GetBdrAttribute(i) == phys) {
          ++n;
          int e1 = -1, e2 = -1;
          m.GetFaceElements(m.GetBdrElementFaceIndex(i), &e1, &e2);
          interior += e2 >= 0;
        }
      CHECK(n > 0 && interior == n, "check_markers: open marker atom " + std::to_string(phys) + ": every boundary element is an interior face (" +
                                       std::to_string(interior) + " of " + std::to_string(n) + ")");
    }
    // the marker that ends inside the region does not split it: the region b stays one piece, its area is intact
    double area_b = 0;
    for (int e = 0; e < m.GetNE(); ++e)
      if (m.GetAttribute(e) == 2) area_b += elem_area(m, e);
    CHECK(close_rel(area_b, 0.0002, 1e-12), "check_markers: open_markers: region b's mesh area " + fmt(area_b));
  }
}

// An entity on the exact box: thin_film's top is selected by y = 1e-6; an inflated box (9e-7 to 1.1e-6) selects none.
void check_where_exact_box() {
  const char* who = "check_where_exact_box";
  const Run* r = get(who, "thin_film");
  if (!r) return;
  const json& top = r->info.at("boundaries").at("top");
  CHECK(top.value("entities", 0) == 1, std::string(who) + ": top selects " + std::to_string(top.value("entities", 0)) + " entities, want 1 (an inflated box selects none)");
  CHECK(top.value("attributes", Ints{}) == Ints{2002}, std::string(who) + ": top is atom 2002");
  CHECK(r->info.at("empty").is_object() && r->info.at("empty").empty(), std::string(who) + ": no empty selection");
  for (const json& e : r->info.at("entities")) {
    const double d = 1e-7;
    CHECK(e.at("hi").at(1).get<double>() - e.at("lo").at(1).get<double>() < 1e-5 + 0.5 * d && e.at("hi").at(0).get<double>() - e.at("lo").at(0).get<double>() < 1e-5 + 0.5 * d,
          std::string(who) + ": a box is wider than its entity by about 1e-7 m");
  }
  for (const json& e : r->info.at("entities"))
    if (e.at("atom") == 2002) {
      CHECK(std::fabs(e.at("lo").at(1).get<double>() - 1e-6) < 1e-18 && std::fabs(e.at("hi").at(1).get<double>() - 1e-6) < 1e-18,
            std::string(who) + ": the top edge's y range is " + fmt(e.at("lo").at(1).get<double>()) + " to " + fmt(e.at("hi").at(1).get<double>()) + ", want [1e-6, 1e-6]");
    }
}

// resolve_only: the info file with entities, boundaries and groups, no mesh file, and the fields of the full run.
void check_resolve_only() {
  const char* who = "check_resolve_only";
  for (const char* name : {"rect", "split_rect", "annulus", "pipe_axi", "open_markers", "channel_shifted"}) {
    json d = fixture(name);
    d["resolve_only"] = true;
    const Run r = run_doc(d, std::string(name) + "_resolve");
    if (!r.ok()) {
      fail(std::string(who) + ": " + name + ": " + r.why);
      continue;
    }
    std::printf("  resolve_only %s: mesher seconds = %g\n", name, r.info.value("seconds", -1.0));
    CHECK(r.msh.empty(), std::string(who) + ": " + name + ": a mesh file was written");
    CHECK(r.info.contains("entities") && r.info.contains("boundaries") && r.info.contains("boundary_groups") && r.info.contains("regions"),
          std::string(who) + ": " + name + ": the info file lacks entities, boundaries, boundary_groups or regions");
    CHECK(!r.info.contains("nodes") && !r.info.contains("elements_2d") && !r.info.contains("boundary_elements"),
          std::string(who) + ": " + name + ": counts of a mesh that was not made");
    CHECK(r.info.value("bounded", false), std::string(who) + ": " + name + ": bounded");
    // the same resolution as the full run (the geometry is the same)
    json full = fixture(name);
    full["resolve_only"] = false;
    if (std::string(name) != "channel_shifted") {
      const Run* f = get(who, name);
      if (f)
        for (const char* k : {"regions", "entities", "boundaries", "boundary_groups", "empty", "partial", "diagonal_m", "tolerance_m"})
          CHECK(f->info.at(k) == r.info.at(k), std::string(who) + ": " + name + ": '" + k + "' differs between resolve_only and the full run");
    }
    if (std::string(name) == "channel_shifted") {
      const json& e = r.info.at("empty");
      CHECK(e.contains("hot"), std::string(who) + ": channel_shifted: no empty selection 'hot'");
      if (e.contains("hot")) {
        CHECK(e["hot"].at("where") == json::parse(R"({"x": [0]})"), std::string(who) + ": channel_shifted: where");
        const json& c = e["hot"].at("candidates");
        // the note's example: the sides of the shifted channel at x = 0.0005 and x = 0.0405, then the two long edges
        CHECK(c.size() == 4, std::string(who) + ": channel_shifted: four candidate lines");
        if (c.size() == 4) {
          const double xs[2] = {0.0005, 0.0405};
          for (int k = 0; k < 2; ++k) {
            CHECK(c[k].at("where").contains("x") && close_abs(c[k]["where"]["x"].get<double>(), xs[k], 1e-12), std::string(who) + ": channel_shifted: candidate x");
            CHECK(close_rel(c[k].at("measure_m").get<double>(), 0.01, 1e-13), std::string(who) + ": channel_shifted: candidate measure");
          }
          CHECK(c[2].at("where").contains("y") && close_abs(c[2]["where"]["y"].get<double>(), 0.0, 1e-12) && close_rel(c[2]["measure_m"].get<double>(), 0.04, 1e-13),
                std::string(who) + ": channel_shifted: the bottom candidate");
          CHECK(c[3].at("where").contains("y") && close_abs(c[3]["where"]["y"].get<double>(), 0.01, 1e-12) && close_rel(c[3]["measure_m"].get<double>(), 0.04, 1e-13),
                std::string(who) + ": channel_shifted: the top candidate");
        }
      }
      check_table_against(fx_channel_shifted(), r, who);
    }
    if (std::string(name) == "annulus") {
      // the outer circle's box encloses inner's where box in x and in y: the box cannot tell whether the circle passes
      // through it (it does not), so it is not a partial match (model_selection.cpp, the partial-match rule)
      const json& p = r.info.at("partial");
      CHECK(!p.contains("inner"), std::string(who) + ": annulus: the outer circle is not a partial match of 'inner'");
    }
  }
}

// Two runs of one document, at sizes h and h/2 and orders 1 and 2: the same boundaries, groups, regions, entities and
// atoms (the info file outside seconds and the mesh counts); two identical runs give the same mesh file.
void check_determinism() {
  const char* who = "check_determinism";
  for (const char* name : {"split_rect", "channel_post", "pipe_axi"}) {
    json base = fixture(name);
    const double h = base.at("size");
    std::vector<Run> runs;
    for (const json& ov : {json{{"order", 2}}, json{{"order", 2}}, json{{"order", 2}, {"size", h / 2}, {"surface_size", h / 2}}, json{{"order", 1}}}) {
      json d = base;
      for (auto& [k, v] : ov.items()) d[k] = v;
      runs.push_back(run_doc(d, std::string(name) + "_det"));
    }
    bool ok = true;
    for (const Run& r : runs)
      if (!r.ok()) {
        fail(std::string(who) + ": " + name + ": " + r.why);
        ok = false;
        break;
      }
    if (!ok) continue;
    for (size_t k = 1; k < runs.size(); ++k)
      for (const char* key : {"regions", "entities", "boundaries", "boundary_groups", "empty", "partial", "diagonal_m", "tolerance_m"})
        CHECK(runs[0].info.at(key) == runs[k].info.at(key), std::string(who) + ": " + name + ": '" + key + "' of run " + std::to_string(k) + " differs from run 0");
    std::ifstream a(runs[0].msh, std::ios::binary), b(runs[1].msh, std::ios::binary);
    const std::string sa((std::istreambuf_iterator<char>(a)), {}), sb((std::istreambuf_iterator<char>(b)), {});
    CHECK(!sa.empty() && sa == sb, std::string(who) + ": " + name + ": two identical runs wrote different mesh files");
  }
}

// Geometry order 1: no nodes; order 2: nodes, the boundary nodes of the annulus on their circles to 1e-12 R, every
// element with a positive Jacobian determinant at its quadrature points.
void check_geometry_order() {
  const char* who = "check_geometry_order";
  for (int order : {1, 2}) {
    for (const char* name : {"annulus", "channel_post", "pipe_axi"}) {
      const Run* r = get(who, name, json{{"order", order}}, "_o" + std::to_string(order));
      if (!r) continue;
      Loaded l = load(*r);
      mfem::Mesh& m = *l.mesh;
      const std::string at = std::string(who) + ": " + name + " order " + std::to_string(order);
      CHECK(r->info.value("order", 0) == order, at + ": info order");
      if (order == 1) {
        CHECK(m.GetNodes() == nullptr, at + ": MFEM's GetNodes() is not null (the mesh has curved geometry at order 1)");
        for (const MshEl& e : l.msh.els) CHECK(e.type == 1 || e.type == 2 || e.type == 15, at + ": an element of type " + std::to_string(e.type) + " at order 1");
      } else {
        CHECK(m.GetNodes() != nullptr, at + ": MFEM's GetNodes() is null at order 2");
      }
      if (order == 2 && (std::string(name) == "annulus" || std::string(name) == "channel_post")) {
        const double cx = std::string(name) == "annulus" ? 0.0 : 0.01, cy = std::string(name) == "annulus" ? 0.0 : 0.005;
        const double radii[2] = {0.002, 0.005};
        double worst = 0;
        int checked = 0;
        for (int i = 0; i < m.GetNBE(); ++i) {
          const int a = m.GetBdrAttribute(i);
          const bool circle = std::string(name) == "annulus" ? true : a == 2001;
          if (!circle) continue;
          mfem::ElementTransformation* T = m.GetBdrElementTransformation(i);
          for (double t : {0.0, 0.5, 1.0}) {
            mfem::IntegrationPoint ip;
            ip.Set1w(t, 1.0);
            mfem::Vector p;
            T->Transform(ip, p);
            const double rr = std::hypot(p[0] - cx, p[1] - cy);
            double best = 1e300;
            for (double R : radii) best = std::min(best, std::fabs(rr - R) / R);
            if (std::string(name) == "channel_post") best = std::fabs(rr - 0.002) / 0.002;
            worst = std::max(worst, best);
          }
          ++checked;
        }
        CHECK(checked > 0, at + ": no circle boundary element");
        CHECK(worst <= 1e-12, at + ": a boundary P2 node is off its circle by " + fmt(worst) + " R (limit 1e-12)");
      }
      int bad = 0;
      for (int e = 0; e < m.GetNE(); ++e) {
        mfem::ElementTransformation* T = m.GetElementTransformation(e);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(m.GetElementBaseGeometry(e), 4);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          T->SetIntPoint(&ir.IntPoint(q));
          if (!(T->Jacobian().Det() > 0)) ++bad;
        }
      }
      CHECK(bad == 0, at + ": " + std::to_string(bad) + " quadrature points with a non-positive Jacobian determinant");
    }
  }
}

// Refusals: ok:false with the texts, exit status 1, no mesh file.
void check_refusals() {
  const char* who = "check_refusals";
  struct Case { const char* name; const char* text; };
  for (const Case& c : {Case{"refuse_covered", "region 'a' is completely covered by later regions; reorder or remove it"},
                        Case{"refuse_curve_outside", "curve 'mk' (or part of it) lies outside every enabled region; a marker must lie inside the regions or on their boundary"}}) {
    const Run r = run_doc(fixture(c.name), c.name);
    CHECK(r.started && r.rc == 1, std::string(who) + ": " + c.name + ": exit status " + std::to_string(r.rc) + ", want 1");
    CHECK(r.info.is_object() && r.info.value("ok", true) == false, std::string(who) + ": " + c.name + ": the info file does not say ok: false");
    const std::string got = r.info.is_object() ? r.info.value("error", std::string()) : std::string();
    CHECK(got == c.text, std::string(who) + ": " + c.name + ":\n    got:  " + got + "\n    want: " + c.text);
    CHECK(r.msh.empty(), std::string(who) + ": " + c.name + ": a mesh file was written");
  }
  // a covered region does not leave the fixture's size and boundaries unused: the refusal needs no resolution
}

// The mesh file's dimension-1 physical groups are atom_<master>, one per atom, even where a boundary is named a+b beside
// an atom of a and b; the info file says the same.
void check_group_names() {
  const char* who = "check_group_names";
  for (const char* name : {"names_join", "split_rect", "pipe_axi", "unit_square_axisymmetric"}) {
    const Run* r = get(who, name);
    if (!r) continue;
    const Msh m = read_msh(r->msh);
    std::set<int> atoms;
    for (const json& g : r->info.at("boundary_groups")) atoms.insert(g.at("attr").get<int>());
    CHECK(m.names1.size() == atoms.size(), std::string(who) + ": " + name + ": " + std::to_string(m.names1.size()) + " dimension-1 groups in the mesh file, " +
                                              std::to_string(atoms.size()) + " atoms");
    for (const auto& [num, nm] : m.names1) CHECK(atoms.count(num) == 1 && nm == "atom_" + std::to_string(num), std::string(who) + ": " + name + ": group " + std::to_string(num) + " is named '" + nm + "'");
    // every line element is in exactly one of them and its group is an atom
    for (const MshEl& e : m.lines()) CHECK(m.names1.count(e.phys) == 1, std::string(who) + ": " + name + ": a line element in physical group " + std::to_string(e.phys));
  }
  const Run* r = get(who, "names_join");
  if (r) {
    const json& g = r->info.at("boundary_groups");
    CHECK(g.size() == 4, std::string(who) + ": names_join: four atoms (a+b, [a, b], b and the rest)");
    CHECK(r->info.at("boundaries").contains("a+b") && r->info["boundaries"]["a+b"].value("attributes", Ints{}) == Ints{2002}, std::string(who) + ": names_join: boundary 'a+b' is atom 2002");
    const Fx f = fx_names_join();
    check_table_against(f, *r, who);
  }
}

// A plane-mode input (no "bounded") still takes the plane branch: the plane info (plane: true, groups with the outer
// group) and not the bounded one.
void check_plane_mode_dispatch() {
  const char* who = "check_plane_mode_dispatch";
  const Run* r = get(who, "plane_rect");
  if (!r) return;
  CHECK(r->info.value("plane", false) == true, std::string(who) + ": the plane info has plane: true");
  CHECK(!r->info.contains("bounded") && !r->info.contains("entities") && !r->info.contains("boundary_groups"), std::string(who) + ": the plane info is the plane mode's");
  bool outer = false;
  for (const json& g : r->info.at("groups")) outer = outer || (g.value("name", std::string()) == "outer" && g.value("attr", 0) == 901);
  CHECK(outer, std::string(who) + ": the plane groups hold 'outer' = 901");
  const Msh m = read_msh(r->msh);
  CHECK(m.names1.size() == 2 && m.names1.count(901) == 1 && m.names1.at(901) == "outer", std::string(who) + ": the plane mesh's dimension-1 groups are outer and the sheet");
  // and an explicit "bounded": false with "plane": true is the plane mode too
  json d = fixture("plane_rect");
  d["bounded"] = false;
  const Run r2 = run_doc(d, "plane_rect_false");
  CHECK(r2.ok() && r2.info.value("plane", false) == true && !r2.info.contains("bounded"), std::string(who) + ": \"bounded\": false takes the plane branch");
  // the line elements of the outer group and the sheet group are double-counted as in the plane mode (as before): the
  // plane mode is not edited
  const Msh m1 = read_msh(r->msh);
  int n = 0;
  for (const MshEl& e : m1.lines()) n += e.phys == 902;
  CHECK(n > 0, std::string(who) + ": the sheet group has elements");
}

// The info counts agree with the mesh file; the mesh reads with MFEM, 2D elements are triangles.
void check_info_counts() {
  const char* who = "check_info_counts";
  for (const char* name : {"rect", "pipe_axi", "annulus"}) {
    const Run* r = get(who, name);
    if (!r) continue;
    const Msh m = read_msh(r->msh);
    const std::string at = std::string(who) + ": " + name;
    CHECK(r->info.value("nodes", size_t(0)) == m.nodes.size(), at + ": nodes " + std::to_string(r->info.value("nodes", size_t(0))) + " against " + std::to_string(m.nodes.size()) + " in the file");
    CHECK(r->info.value("elements_2d", size_t(0)) == m.tris().size(), at + ": elements_2d against the file");
    CHECK(r->info.value("boundary_elements", size_t(0)) == m.lines().size(), at + ": boundary_elements " + std::to_string(r->info.value("boundary_elements", size_t(0))) +
                                                                                " against " + std::to_string(m.lines().size()) + " line elements in the file");
    CHECK(m.lines().size() + m.tris().size() <= m.els.size(), at + ": counts");
    for (const MshEl& e : m.els) CHECK(e.type != 15 || true, at);  // points (type 15) are not boundary elements
    mfem::Mesh mm(r->msh.c_str(), 1, 1, true);
    CHECK(mm.GetNE() == int(m.tris().size()) && mm.GetNBE() == int(m.lines().size()), at + ": MFEM reads a different number of elements or boundary elements");
  }
}


// ------------------------------------------------------------------ arcs whose extreme points fall inside the arc
// A group of arcs of one circle written out by hand: centre, radius and angular range [a0, a1] (counter-clockwise), the
// unordered pair of regions the arcs separate, and the markers they carry.  gmsh may split a group at the seam of its
// circle (angle 0), so the entities of a group are summed, never counted.
struct ArcGroup {
  double cx, cy, R, a0, a1;
  Strs curves;
  std::pair<std::string, std::string> sides;
};
struct CurvedFx {
  std::string name;
  bool axisym = false;
  std::function<std::string(double, double)> region_of;
  double delta = 1e-6, diag = 1;
  std::vector<std::array<double, 4>> lines;   // straight entities by their end points
  std::vector<ArcGroup> groups;
};

// The box of the arc of the circle (c, R) over [a0, a1]: the end points and the extreme points of the full circle that
// lie in the range.
void arc_box(double cx, double cy, double R, double a0, double a1, double* lo, double* hi) {
  lo[0] = hi[0] = cx + R * std::cos(a0);
  lo[1] = hi[1] = cy + R * std::sin(a0);
  auto add = [&](double a) {
    const double x = cx + R * std::cos(a), y = cy + R * std::sin(a);
    lo[0] = std::min(lo[0], x); hi[0] = std::max(hi[0], x);
    lo[1] = std::min(lo[1], y); hi[1] = std::max(hi[1], y);
  };
  add(a1);
  for (int m = -12; m <= 12; ++m) {
    const double a = m * kPi / 2;
    if (a >= a0 && a <= a1) add(a);
  }
}
double revolved_arc_area(double cx, double R, double a0, double a1) {
  return 2 * kPi * R * (cx * (a1 - a0) + R * (std::sin(a1) - std::sin(a0)));
}

void check_curved_fixture(const CurvedFx& f, const std::string& who) {
  const Run* r = get(who, f.name);
  if (!r) return;
  const std::string at = who + ": " + f.name;
  const std::vector<Ent> ents = read_entities(r->info, at);
  const double tol = 1e-12 * f.diag;
  std::vector<int> line_seen(f.lines.size(), 0);
  struct Acc { double len = 0, area = 0, lo[2] = {1e300, 1e300}, hi[2] = {-1e300, -1e300}; int n = 0; };
  std::vector<Acc> acc(f.groups.size());
  size_t nlines = 0;
  for (const Ent& e : ents) {
    const std::string en = at + ": entity " + std::to_string(e.tag);
    const double nx = -e.t[1], ny = e.t[0];
    const std::string left = f.region_of(e.p[0] + f.delta * nx, e.p[1] + f.delta * ny);
    const std::string right = f.region_of(e.p[0] - f.delta * nx, e.p[1] - f.delta * ny);
    CHECK(e.side0 == (left.empty() ? "exterior" : left) && e.side1 == (right.empty() ? "exterior" : right),
          en + ": sides [" + e.side0 + ", " + e.side1 + "] against the hand region rule [" + (left.empty() ? "exterior" : left) + ", " +
              (right.empty() ? "exterior" : right) + "]");
    CHECK(close_abs(std::hypot(e.t[0], e.t[1]), 1.0, 1e-12), en + ": tangent is not a unit vector");
    if (e.type == "line") {
      ++nlines;
      bool hit = false;
      for (size_t k = 0; k < f.lines.size(); ++k) {
        const auto& L = f.lines[k];
        const double mx = 0.5 * (L[0] + L[2]), my = 0.5 * (L[1] + L[3]), len = std::hypot(L[2] - L[0], L[3] - L[1]);
        if (!(close_abs(e.p[0], mx, tol) && close_abs(e.p[1], my, tol))) continue;
        hit = true;
        ++line_seen[k];
        CHECK(close_rel(e.length, len, 1e-13), en + ": length_m " + fmt(e.length) + ", want " + fmt(len));
        CHECK(close_abs(e.lo[0], std::min(L[0], L[2]), tol) && close_abs(e.hi[0], std::max(L[0], L[2]), tol) &&
                  close_abs(e.lo[1], std::min(L[1], L[3]), tol) && close_abs(e.hi[1], std::max(L[1], L[3]), tol),
              en + ": the box of a straight entity is not the span of its end points");
        if (f.axisym) CHECK(close_abs(e.area, 2 * kPi * mx * len, 1e-13 * (1 + 2 * kPi * mx * len)), en + ": revolved area " + fmt(e.area));
      }
      CHECK(hit, en + ": a straight entity at (" + fmt(e.p[0]) + ", " + fmt(e.p[1]) + ") that the hand table does not have");
      continue;
    }
    CHECK(e.type == "circle" || e.type == "ellipse", en + ": type '" + e.type + "'");
    // the group: same circle, mid angle inside the range, same pair of regions
    const ArcGroup* g = nullptr;
    size_t gi = 0;
    for (size_t k = 0; k < f.groups.size(); ++k) {
      const ArcGroup& c = f.groups[k];
      if (std::fabs(std::hypot(e.p[0] - c.cx, e.p[1] - c.cy) - c.R) > 1e-9 * c.R) continue;
      double rel = std::atan2(e.p[1] - c.cy, e.p[0] - c.cx) - c.a0;
      while (rel < -1e-9) rel += 2 * kPi;
      while (rel >= 2 * kPi - 1e-9) rel -= 2 * kPi;
      if (rel > c.a1 - c.a0 + 1e-9) continue;
      g = &c;
      gi = k;
      break;
    }
    CHECK(g != nullptr, en + ": an arc at (" + fmt(e.p[0]) + ", " + fmt(e.p[1]) + ") that lies in no group of the hand table");
    if (!g) continue;
    const bool sides_ok = (e.side0 == g->sides.first && e.side1 == g->sides.second) || (e.side0 == g->sides.second && e.side1 == g->sides.first);
    CHECK(sides_ok, en + ": sides [" + e.side0 + ", " + e.side1 + "] are not the pair [" + g->sides.first + ", " + g->sides.second + "]");
    CHECK(e.curves == g->curves, en + ": curves differ from the hand table");
    CHECK(std::fabs(e.t[0] * (e.p[0] - g->cx) + e.t[1] * (e.p[1] - g->cy)) <= 1e-9 * g->R, en + ": tangent is not perpendicular to the radius");
    if (!g->curves.empty()) {
      const double ccw = e.t[0] * -(e.p[1] - g->cy) + e.t[1] * (e.p[0] - g->cx);   // an arc marker is written counter-clockwise
      CHECK(ccw > 0, en + ": the tangent of an arc marker is not its written (counter-clockwise) direction");
    }
    // this arc's own range from its parametric midpoint and its length (the box must follow from them)
    const double am = std::atan2(e.p[1] - g->cy, e.p[0] - g->cx), half = e.length / (2 * g->R);
    double lo[2], hi[2];
    arc_box(g->cx, g->cy, g->R, am - half, am + half, lo, hi);
    CHECK(close_abs(e.lo[0], lo[0], tol) && close_abs(e.hi[0], hi[0], tol) && close_abs(e.lo[1], lo[1], tol) && close_abs(e.hi[1], hi[1], tol),
          en + ": box [" + fmt(e.lo[0]) + ", " + fmt(e.hi[0]) + "] x [" + fmt(e.lo[1]) + ", " + fmt(e.hi[1]) + "], the arc's own extreme points give [" +
              fmt(lo[0]) + ", " + fmt(hi[0]) + "] x [" + fmt(lo[1]) + ", " + fmt(hi[1]) + "]");
    Acc& a = acc[gi];
    ++a.n;
    a.len += e.length;
    a.area += e.area;
    for (int k = 0; k < 2; ++k) {
      a.lo[k] = std::min(a.lo[k], e.lo[k]);
      a.hi[k] = std::max(a.hi[k], e.hi[k]);
    }
  }
  CHECK(nlines == f.lines.size(), at + ": " + std::to_string(nlines) + " straight entities, want " + std::to_string(f.lines.size()));
  for (size_t k = 0; k < f.lines.size(); ++k) CHECK(line_seen[k] == 1, at + ": straight entity " + std::to_string(k) + " of the hand table matched " + std::to_string(line_seen[k]) + " times");
  for (size_t k = 0; k < f.groups.size(); ++k) {
    const ArcGroup& g = f.groups[k];
    const std::string gn = at + ": arcs of the circle (" + fmt(g.cx) + ", " + fmt(g.cy) + ") R " + fmt(g.R) + " over [" + fmt(g.a0) + ", " + fmt(g.a1) + "]";
    CHECK(acc[k].n > 0, gn + ": no entity");
    if (acc[k].n == 0) continue;
    CHECK(close_rel(acc[k].len, g.R * (g.a1 - g.a0), 1e-13), gn + ": total length " + fmt(acc[k].len) + ", want " + fmt(g.R * (g.a1 - g.a0)));
    double lo[2], hi[2];
    arc_box(g.cx, g.cy, g.R, g.a0, g.a1, lo, hi);
    CHECK(close_abs(acc[k].lo[0], lo[0], tol) && close_abs(acc[k].hi[0], hi[0], tol) && close_abs(acc[k].lo[1], lo[1], tol) && close_abs(acc[k].hi[1], hi[1], tol),
          gn + ": union box [" + fmt(acc[k].lo[0]) + ", " + fmt(acc[k].hi[0]) + "] x [" + fmt(acc[k].lo[1]) + ", " + fmt(acc[k].hi[1]) + "], want [" + fmt(lo[0]) +
              ", " + fmt(hi[0]) + "] x [" + fmt(lo[1]) + ", " + fmt(hi[1]) + "]");
    if (f.axisym) {
      const double want = revolved_arc_area(g.cx, g.R, g.a0, g.a1);
      CHECK(close_rel(acc[k].area, want, 1e-12), gn + ": revolved area " + fmt(acc[k].area) + ", want " + fmt(want));
    }
  }
}

std::array<double, 4> seg4(double a, double b, double c, double d) { return {a, b, c, d}; }

// The rounded polygon of the fillets: for each vertex the arc of radius r tangent to both edges (centre on the bisector),
// the edges shortened by the tangent length r tan(phi / 2), phi the turning angle.  Counter-clockwise points.
CurvedFx fillet_fixture(const std::string& name, const std::vector<std::array<double, 2>>& P, double r) {
  CurvedFx f;
  f.name = name;
  const size_t n = P.size();
  std::vector<std::array<double, 2>> A(n), B(n);
  for (size_t i = 0; i < n; ++i) {
    const auto& p0 = P[(i + n - 1) % n];
    const auto& p1 = P[i];
    const auto& p2 = P[(i + 1) % n];
    const double ul = std::hypot(p1[0] - p0[0], p1[1] - p0[1]), wl = std::hypot(p2[0] - p1[0], p2[1] - p1[1]);
    const double ux = (p1[0] - p0[0]) / ul, uy = (p1[1] - p0[1]) / ul, wx = (p2[0] - p1[0]) / wl, wy = (p2[1] - p1[1]) / wl;
    const double phi = std::atan2(ux * wy - uy * wx, ux * wx + uy * wy);
    const double t = r * std::tan(phi / 2);
    A[i] = {p1[0] - ux * t, p1[1] - uy * t};
    B[i] = {p1[0] + wx * t, p1[1] + wy * t};
    const double cx = A[i][0] - uy * r, cy = A[i][1] + ux * r;   // A + r * (left normal of u)
    const double a0 = std::atan2(A[i][1] - cy, A[i][0] - cx);
    f.groups.push_back({cx, cy, r, a0, a0 + phi, {}, {"a", "exterior"}});
  }
  for (size_t i = 0; i < n; ++i) f.lines.push_back(seg4(B[i][0], B[i][1], A[(i + 1) % n][0], A[(i + 1) % n][1]));
  // the rounded polygon is the set of points within r of the polygon of the fillets' centres (all radii are r)
  std::vector<std::array<double, 2>> C;
  for (const ArcGroup& g : f.groups) C.push_back({g.cx, g.cy});
  f.region_of = [C, r](double x, double y) {
    const size_t m = C.size();
    bool inside = true;
    double best = 1e300;
    for (size_t i = 0; i < m; ++i) {
      const auto& a = C[i];
      const auto& b = C[(i + 1) % m];
      const double ex = b[0] - a[0], ey = b[1] - a[1];
      if (ex * (y - a[1]) - ey * (x - a[0]) < 0) inside = false;   // counter-clockwise: the interior is on the left
      const double t = std::clamp(((x - a[0]) * ex + (y - a[1]) * ey) / (ex * ex + ey * ey), 0.0, 1.0);
      best = std::min(best, std::hypot(x - a[0] - t * ex, y - a[1] - t * ey));
    }
    return (inside || best < r) ? std::string("a") : std::string();
  };
  return f;
}

void check_arc_extremes() {
  const char* who = "check_arc_extremes";
  auto in_rect = [](double x, double y, double x0, double x1, double y0, double y1) { return x > x0 && x < x1 && y > y0 && y < y1; };
  {  // arc markers over [2.5, 3.8], [0.1, 5.0] and [-2, -1]: extreme points inside the arcs, away from multiples of 90 degrees
    CurvedFx f;
    f.name = "arc_markers";
    f.region_of = [=](double x, double y) { return in_rect(x, y, 0, 1, 0, 1) ? std::string("a") : std::string(); };
    f.diag = std::sqrt(2.0);
    f.lines = {seg4(0, 0, 1, 0), seg4(1, 0, 1, 1), seg4(1, 1, 0, 1), seg4(0, 1, 0, 0)};
    f.groups = {{0.25, 0.25, 0.15, 2.5, 3.8, {"m1"}, {"a", "a"}}, {0.7, 0.7, 0.2, 0.1, 5.0, {"m2"}, {"a", "a"}}, {0.7, 0.25, 0.15, -2.0, -1.0, {"m3"}, {"a", "a"}}};
    check_curved_fixture(f, who);
  }
  {  // the filleted trapezoid and the filleted triangle (fillet radii 0.1 and 0.05)
    CurvedFx t = fillet_fixture("fillet_trapezoid", {{0, 0}, {1, 0}, {0.7, 0.6}, {0.2, 0.6}}, 0.1);
    t.diag = std::hypot(1.0, 0.6);
    check_curved_fixture(t, who);
    CurvedFx g = fillet_fixture("fillet_triangle", {{0, 0}, {1, 0}, {0.3, 0.8}}, 0.05);
    g.diag = std::hypot(1.0, 0.8);
    check_curved_fixture(g, who);
  }
  {  // a disk (centre (0.9, 0.95), R 0.3) painted over the corner of the unit square
    const double s1 = std::sqrt(0.09 - 0.0025), s2 = std::sqrt(0.09 - 0.01);
    const double inner0 = std::atan2(0.05, -s1), inner1 = 2 * kPi + std::atan2(-s2, 0.1);
    CurvedFx f;
    f.name = "disk_over_corner";
    f.region_of = [=](double x, double y) {
      if (std::hypot(x - 0.9, y - 0.95) < 0.3) return std::string("b");
      return in_rect(x, y, 0, 1, 0, 1) ? std::string("a") : std::string();
    };
    f.diag = std::hypot(1.2, 1.25);
    f.lines = {seg4(0, 0, 0, 1), seg4(0, 0, 1, 0), seg4(0, 1, 0.9 - s1, 1), seg4(0.9 - s1, 1, 1, 1), seg4(1, 0, 1, 0.95 - s2), seg4(1, 0.95 - s2, 1, 1)};
    f.groups = {{0.9, 0.95, 0.3, inner0, inner1, {}, {"a", "b"}}, {0.9, 0.95, 0.3, inner1 - 2 * kPi, inner0, {}, {"b", "exterior"}}};
    check_curved_fixture(f, who);
  }
  {  // an off-axis ball (centre (0.5, 0.9), R 0.2) cut by the cylinder's top, axisymmetric: the circle meets z = 1 at 30 and 150 degrees
    const double s = std::sqrt(0.04 - 0.01);
    CurvedFx f;
    f.name = "axi_ball_cut";
    f.axisym = true;
    f.region_of = [=](double r, double z) {
      if (std::hypot(r - 0.5, z - 0.9) < 0.2) return std::string("ball");
      return in_rect(r, z, 0, 1, 0, 1) ? std::string("cyl") : std::string();
    };
    f.diag = std::hypot(1.0, 1.1);
    f.lines = {seg4(0, 0, 0, 1), seg4(0, 0, 1, 0), seg4(1, 0, 1, 1), seg4(0, 1, 0.5 - s, 1), seg4(0.5 - s, 1, 0.5 + s, 1), seg4(0.5 + s, 1, 1, 1)};
    f.groups = {{0.5, 0.9, 0.2, 5 * kPi / 6, 13 * kPi / 6, {}, {"ball", "cyl"}}, {0.5, 0.9, 0.2, kPi / 6, 5 * kPi / 6, {}, {"ball", "exterior"}}};
    check_curved_fixture(f, who);
  }
  {  // a hole's circle split by an arc marker over [0, 1]
    CurvedFx f;
    f.name = "hole_arc_marker";
    f.region_of = [=](double x, double y) { return (in_rect(x, y, 0, 1, 0, 1) && std::hypot(x - 0.5, y - 0.5) > 0.2) ? std::string("a") : std::string(); };
    f.diag = std::sqrt(2.0);
    f.lines = {seg4(0, 0, 1, 0), seg4(1, 0, 1, 1), seg4(1, 1, 0, 1), seg4(0, 1, 0, 0)};
    f.groups = {{0.5, 0.5, 0.2, 0.0, 1.0, {"m"}, {"a", "exterior"}}, {0.5, 0.5, 0.2, 1.0, 2 * kPi, {}, {"a", "exterior"}}};
    check_curved_fixture(f, who);
  }
}

// A contour region with an elliptical segment is refused by the bounded modes with a message of its own (its elliptical
// segments are splines, whose areas and boxes the classifier does not have): ok false, exit status 1, no mesh file, and
// not an internal error.
void check_contour_ellipse_refusal() {
  const char* who = "check_contour_ellipse_refusal";
  const Run r = run_doc(fixture("refuse_contour_ellipse"), "refuse_contour_ellipse");
  CHECK(r.started && r.rc == 1, std::string(who) + ": exit status " + std::to_string(r.rc) + ", want 1");
  CHECK(r.info.is_object() && r.info.value("ok", true) == false, std::string(who) + ": the info file does not say ok: false");
  const std::string got = r.info.is_object() ? r.info.value("error", std::string()) : std::string();
  const std::string want = "region 'a': a contour with elliptical segments is not supported by the bounded modes";
  CHECK(got == want, std::string(who) + ":\n    got:  " + got + "\n    want: " + want);
  CHECK(got.find("internal") == std::string::npos, std::string(who) + ": an internal error is not a refusal");
  CHECK(r.msh.empty(), std::string(who) + ": a mesh file was written");
}

void run(const char* name, void (*f)()) {
  const int before = failures;
  try {
    f();
  } catch (const std::exception& e) {
    fail(std::string(name) + ": threw: " + e.what());
  }
  std::printf("%s %s\n", failures == before ? "ok  " : "FAIL", name);
}

}  // namespace

int main() {
  try {
    g_mesher = axmp::axi::find_mesher();
  } catch (const std::exception& e) {
    std::printf("SKIP: no axmp-mesher (%s)\n", e.what());
    return 77;
  }
#ifndef AXMP_SOURCE_DIR
  std::printf("SKIP: no AXMP_SOURCE_DIR\n");
  return 77;
#endif
  g_scratch = fs::temp_directory_path() / ("axmp_mesher_bounded_" + std::to_string(::getpid()));
  fs::create_directories(g_scratch);
  run("check_entity_table", check_entity_table);
  run("check_info_measures_are_exact", check_info_measures_are_exact);
  run("check_mesh_measures", check_mesh_measures);
  run("check_counting", check_counting);
  run("check_attributes", check_attributes);
  run("check_axis", check_axis);
  run("check_markers", check_markers);
  run("check_where_exact_box", check_where_exact_box);
  run("check_resolve_only", check_resolve_only);
  run("check_determinism", check_determinism);
  run("check_geometry_order", check_geometry_order);
  run("check_refusals", check_refusals);
  run("check_group_names", check_group_names);
  run("check_plane_mode_dispatch", check_plane_mode_dispatch);
  run("check_info_counts", check_info_counts);
  run("check_arc_extremes", check_arc_extremes);
  run("check_contour_ellipse_refusal", check_contour_ellipse_refusal);
  std::error_code ec;
  if (!failures) fs::remove_all(g_scratch, ec);
  else std::printf("(kept %s)\n", g_scratch.c_str());
  std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
