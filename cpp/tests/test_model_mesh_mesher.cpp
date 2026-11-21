// The meshed model and the instance mesh on meshes made by axmp-mesher from model documents (model_mesh.hpp): the
// probe of the real mesher, then the mesher-made cases (mesh_model, read_meshed_model, resolve_boundaries).  The
// hand-built cases, which run without the mesher, are in test_model_mesh.cpp.  Built in every build; returns 77 (CMake's
// SKIP_RETURN_CODE) when no axmp-mesher is found.
//
// Mesh-dependent counts are never asserted (another gmsh build may mesh differently): only geometry (lengths, normals,
// positions), the numbering (which depends on names and regions only) and determinism within one build.
//
// INDEPENDENT REFERENCES.  The hand atoms of split_rect and pipe_axi (written out in test_model_mesh.cpp's fixtures);
// analytic lengths H, L, Lz and 2 pi R1; the orientation of design note 5.2 (+H for between [a, b], -H for the marker
// written bottom to top, +H top to bottom), evaluated from the instance mesh's face vertices; fixtures::dense_attributes()
// and fixtures::master_numbers() (model_matrix_fixtures.hpp) for the unit square of the matrix documents.
#include <algorithm>
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionmultiphysics/axi.hpp"
#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_mesh.hpp"
#include "axionmultiphysics/model_selection.hpp"
#include "axionmultiphysics/model_validate.hpp"
#include "../src/model_matrix.hpp"
#include "model_matrix_fixtures.hpp"

using nlohmann::json;
namespace M = axmp::model;
namespace S = axmp::model::selection;
namespace fx = axmp::model::fixtures;

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

using Strs = std::vector<std::string>;
using Ints = std::vector<int>;

std::string ints_text(const Ints& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "]";
}
std::string num(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.17g", v);
  return b;
}
void check_eq(const std::string& what, const Ints& got, const Ints& want) {
  if (got != want) fail(what + ": got " + ints_text(got) + ", want " + ints_text(want));
}
void check_eq(const std::string& what, const std::string& got, const std::string& want) {
  if (got != want) fail(what + ": got '" + got + "', want '" + want + "'");
}
void check_eq(const std::string& what, int got, int want) {
  if (got != want) fail(what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}
void check_near(const std::string& what, double got, double want, double tol) {
  if (!(std::fabs(got - want) <= tol))
    fail(what + ": got " + num(got) + ", want " + num(want) + " (difference " + num(got - want) + ", tolerance " + num(tol) + ")");
}

void run(const char* name, const std::function<void()>& body) {
  const int before = failures;
  try {
    body();
  } catch (const axmp::Error& e) {
    fail(std::string(name) + ": threw axmp::Error: " + e.what() + (e.code().empty() ? "" : " [code " + e.code() + "]"));
  } catch (const std::exception& e) {
    fail(std::string(name) + ": threw: " + e.what());
  }
  std::printf("%s %s\n", failures == before ? "ok  " : "FAIL", name);
  std::fflush(stdout);
}

constexpr double kL = 0.04, kH = 0.01, kLz = 0.04, kR1 = 0.002;
const double kPi = std::acos(-1.0);

struct TempDir {
  std::string path;
  TempDir() {
    std::string t = (std::filesystem::temp_directory_path() / "axmp_model_mesh_mesher_XXXXXX").string();
    std::vector<char> buf(t.begin(), t.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) throw std::runtime_error("mkdtemp failed");
    path = buf.data();
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

json read_json(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  return json::parse(in);
}
std::string read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
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
// split_rect.json, with the marker written top to bottom when `down`.
json split_rect_doc(bool down) {
  json d = read_json(data_path("split_rect.json"));
  if (down) d["curves"][0]["path"]["segment"] = json::array({json::array({"L/4", "H"}), json::array({"L/4", 0})});
  return d;
}

void face_ends(const mfem::Mesh& m, int f, double p[2], double q[2]) {
  mfem::Array<int> v;
  m.GetFaceVertices(f, v);
  const double *a = m.GetVertex(v[0]), *b = m.GetVertex(v[1]);
  p[0] = a[0], p[1] = a[1], q[0] = b[0], q[1] = b[1];
}
void bdr_ends(const mfem::Mesh& m, int be, double p[2], double q[2]) {
  mfem::Array<int> v;
  m.GetBdrElementVertices(be, v);
  const double *a = m.GetVertex(v[0]), *b = m.GetVertex(v[1]);
  p[0] = a[0], p[1] = a[1], q[0] = b[0], q[1] = b[1];
}

// The atoms of a resolution by master number: {kind, names, side pair}.
struct AtomView {
  std::string kind;
  Strs names;
};
std::map<int, AtomView> atoms_of(const S::Resolution& r) {
  std::map<int, AtomView> out;
  for (const S::Atom& a : r.atoms) out[a.master] = {a.kind, a.names};
  return out;
}
void check_atoms(const std::string& what, const S::Resolution& r, const std::map<int, AtomView>& want) {
  const auto got = atoms_of(r);
  CHECK(got.size() == want.size(), what + ": " + std::to_string(got.size()) + " atoms, want " + std::to_string(want.size()));
  for (const auto& [master, w] : want) {
    auto it = got.find(master);
    CHECK(it != got.end() && it->second.kind == w.kind && it->second.names == w.names,
          what + ": atom " + std::to_string(master) + " is " + w.kind);
  }
}
const std::map<int, AtomView> kSplitRectAtoms = {{2001, {"named", {"cold"}}},   {2002, {"named", {"hot"}}},
                                                  {2003, {"named", {"interface"}}}, {2004, {"named", {"mk"}}},
                                                  {2005, {"unnamed", {}}},          {2006, {"unnamed", {}}}};

// The contract of design note 5.4, from MFEM's topology: every boundary element on a face with one adjacent element, and
// every such face with exactly one boundary element.
void check_contract(const std::string& what, const M::InstanceMesh& im) {
  const mfem::Mesh& m = *im.mesh;
  std::vector<int> count(m.GetNumFaces(), 0);
  for (int be = 0; be < m.GetNBE(); ++be) ++count[m.GetBdrElementFaceIndex(be)];
  for (int f = 0; f < m.GetNumFaces(); ++f) {
    int e1 = -1, e2 = -1;
    m.GetFaceElements(f, &e1, &e2);
    CHECK(count[f] == (e2 < 0 ? 1 : 0), what + ": face " + std::to_string(f) + " carries " + std::to_string(count[f]) +
                                            " boundary elements (" + (e2 < 0 ? "one" : "two") + " adjacent elements)");
  }
}

// sum over a face set of sign * (the face's integral of n . e_x) and of the face lengths
std::pair<double, double> flux_and_length(const mfem::Mesh& m, const M::FaceSet& fs) {
  double flux = 0.0, len = 0.0;
  for (size_t i = 0; i < fs.faces.size() && i < fs.sign.size(); ++i) {
    double p[2], q[2];
    face_ends(m, fs.faces[i], p, q);
    flux += fs.sign[i] * (q[1] - p[1]);
    len += std::hypot(q[0] - p[0], q[1] - p[1]);
  }
  return {flux, len};
}
double sum(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s;
}
double weighted_normal(const M::FacePoints& fp, int c) {
  double s = 0.0;
  for (size_t j = 0; j < fp.weight.size() && 2 * j + 1 < fp.normal.size(); ++j) s += fp.weight[j] * fp.normal[2 * j + c];
  return s;
}

// ---- the probe of the mesher of this build (the program loads: status 2)
void check_probe(const std::string& found) {
  const M::MesherStatus s = M::mesher_status();
  CHECK(s.available, "the mesher of this build is available: " + s.why);
  CHECK(!s.path.empty() && std::filesystem::equivalent(s.path, found), "the probe names the mesher found: " + s.path);
  CHECK(s.why.empty(), "available: why is empty");
}

// ---- split_rect meshed: the numbering, the face sets' orientation, the face points, the read, determinism
void check_split_rect() {
  for (bool down : {false, true}) {
    const std::string tag = down ? "split_rect (marker top to bottom)" : "split_rect";
    const M::ValidationReport rep = validated(split_rect_doc(down), tag);
    TempDir dir;
    M::MeshOptions opt;
    opt.workdir = dir.path;
    const M::MeshedModel m = M::mesh_model(rep, opt);
    check_eq(tag + ": symmetry", m.symmetry, "planar");
    check_eq(tag + ": geometry_order", m.geometry_order, rep.geometry_order);
    check_eq(tag + ": mesh_file", m.mesh_file, dir.path + "/mesh.msh");
    check_eq(tag + ": info_file", m.info_file, dir.path + "/mesh_info.json");
    CHECK(std::filesystem::exists(m.mesh_file) && std::filesystem::exists(m.info_file), tag + ": the files are on disk");
    CHECK(m.region_attribute == (std::map<std::string, int>{{"a", 1}, {"b", 2}}), tag + ": region_attribute");
    CHECK(m.table.regions == (Strs{"a", "b"}), tag + ": the table's regions");
    check_atoms(tag, m.resolution, kSplitRectAtoms);
    if (!m.master) {
      fail(tag + ": no master");
      continue;
    }
    CHECK((m.master->GetNodes() != nullptr) == (rep.geometry_order == 2), tag + ": the master has nodes at geometry order 2 only");
    check_eq(tag + ": one boundary_entity per boundary element", static_cast<int>(m.boundary_entity.size()), m.master->GetNBE());
    for (int be = 0; be < m.master->GetNBE() && be < static_cast<int>(m.boundary_entity.size()); ++be) {
      auto it = m.resolution.atom_of_tag.find(m.boundary_entity[be]);
      CHECK(it != m.resolution.atom_of_tag.end() && it->second == m.master->GetBdrAttribute(be),
            tag + ": master boundary element " + std::to_string(be) + " carries the atom of its entity");
    }
    const M::InstanceMesh im = m.instance_mesh({"a", "b"});
    check_contract(tag + " [a, b]", im);
    CHECK(im.master_attribute == (std::map<int, int>{{1, 2001}, {2, 2002}, {3, 2005}, {4, 2006}}), tag + " [a, b]: master_attribute");
    CHECK(im.attributes == (std::map<std::string, Ints>{{"cold", {1}}, {"hot", {2}}, {"interface", {}}, {"mk", {}}}),
          tag + " [a, b]: attributes");
    check_eq(tag + " [a, b]: remaining_attributes", im.remaining_attributes, Ints{3, 4});
    // the face sets: lengths H, orientation of design note 5.2 (signs from the master)
    const auto [mk_flux, mk_len] = flux_and_length(*im.mesh, im.section("mk"));
    check_near(tag + " [a, b]: section('mk') covers H", mk_len, kH, 1e-12 * kH);
    check_near(tag + " [a, b]: section('mk'): the integral of n . e_x", mk_flux, down ? kH : -kH, 1e-12 * kH);
    const auto [if_flux, if_len] = flux_and_length(*im.mesh, im.section("interface"));
    check_near(tag + " [a, b]: section('interface') covers H", if_len, kH, 1e-12 * kH);
    check_near(tag + " [a, b]: section('interface'): the integral of n . e_x", if_flux, kH, 1e-12 * kH);
    // face points
    const M::FacePoints hot = im.face_points("hot", 4);
    check_near(tag + " [a, b]: face_points('hot'): the weights sum to H", sum(hot.weight), kH, 1e-12 * kH);
    check_near(tag + " [a, b]: face_points('hot'): n . e_x outward", weighted_normal(hot, 0), -kH, 1e-12 * kH);
    check_near(tag + " [a, b]: face_points('mk'): n . e_x", weighted_normal(im.face_points("mk", 4), 0), down ? kH : -kH, 1e-12 * kH);
    double rem = 0.0;
    for (int a : im.remaining_attributes) rem += sum(im.face_points(a, 4).weight);
    check_near(tag + " [a, b]: the unnamed atoms (top and bottom) have the length 2L", rem, 2 * kL, 1e-12 * kL);
    // [a]: the interface is a boundary carrying its atom
    const M::InstanceMesh a = m.instance_mesh({"a"});
    check_contract(tag + " [a]", a);
    CHECK(a.master_attribute == (std::map<int, int>{{1, 2002}, {2, 2003}, {3, 2005}}), tag + " [a]: master_attribute");
    check_near(tag + " [a]: face_points('interface'): outward from a", weighted_normal(a.face_points("interface", 4), 0), kH, 1e-12 * kH);
    // the files read again give the same model and the same instance maps
    const M::MeshedModel r = M::read_meshed_model(rep, m.mesh_file, m.info_file, 0);
    CHECK(r.boundary_entity == m.boundary_entity, tag + ": read_meshed_model: boundary_entity");
    CHECK(S::to_json(r.table) == S::to_json(m.table), tag + ": read_meshed_model: the table");
    const M::InstanceMesh ir = r.instance_mesh({"a", "b"});
    CHECK(ir.master_attribute == im.master_attribute && ir.attributes == im.attributes &&
              ir.remaining_attributes == im.remaining_attributes,
          tag + ": read_meshed_model: the instance maps");
    if (!down) {
      // determinism within one build: a second run into another directory gives the same mesh file, byte for byte
      TempDir dir2;
      M::MeshOptions opt2;
      opt2.workdir = dir2.path;
      const M::MeshedModel m2 = M::mesh_model(rep, opt2);
      CHECK(read_bytes(m.mesh_file) == read_bytes(m2.mesh_file), tag + ": two runs give byte-identical mesh files");
      json i1 = read_json(m.info_file), i2 = read_json(m2.info_file);
      i1.erase("seconds");
      i2.erase("seconds");
      CHECK(i1 == i2, tag + ": two runs give the same info file but for its timing");
    }
  }
}

// ---- pipe_axi meshed: the axis atom
void check_pipe_axi() {
  const M::ValidationReport rep = validated(read_json(data_path("pipe_axi.json")), "pipe_axi.json");
  TempDir dir;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  const M::MeshedModel m = M::mesh_model(rep, opt);
  check_atoms("pipe_axi", m.resolution,
              {{900, {"axis", {}}}, {2001, {"named", {"inlet"}}}, {2002, {"named", {"outlet"}}}, {2003, {"named", {"wall"}}},
               {2004, {"unnamed", {}}}});
  const M::InstanceMesh im = m.instance_mesh({"p1", "p2"});
  check_contract("pipe_axi [p1, p2]", im);
  check_eq("pipe_axi [p1, p2]: axis_attributes", im.axis_attributes, Ints{1});
  CHECK(im.master_attribute == (std::map<int, int>{{1, 900}, {2, 2001}, {3, 2002}, {4, 2003}}), "pipe_axi [p1, p2]: master_attribute");
  const mfem::Mesh& mm = *im.mesh;
  const double tol = 1e-12 * kLz;
  std::vector<int> axis_on(mm.GetNumFaces(), 0);
  double axis_len = 0.0;
  for (int be = 0; be < mm.GetNBE(); ++be) {
    double p[2], q[2];
    bdr_ends(mm, be, p, q);
    if (mm.GetBdrAttribute(be) == 1) {
      CHECK(std::fabs(p[0]) <= tol && std::fabs(q[0]) <= tol, "pipe_axi [p1, p2]: an axis boundary element lies on r = 0");
      axis_len += std::fabs(q[1] - p[1]);
      ++axis_on[mm.GetBdrElementFaceIndex(be)];
    }
    CHECK(!(std::fabs(p[1] - kLz / 2) <= tol && std::fabs(q[1] - kLz / 2) <= tol), "pipe_axi [p1, p2]: no boundary element on the interface");
  }
  for (int f = 0; f < mm.GetNumFaces(); ++f) {
    double p[2], q[2];
    face_ends(mm, f, p, q);
    if (std::fabs(p[0]) <= tol && std::fabs(q[0]) <= tol)
      CHECK(axis_on[f] == 1, "pipe_axi [p1, p2]: the face " + std::to_string(f) + " on r = 0 carries the axis atom");
  }
  check_near("pipe_axi [p1, p2]: the axis atom covers r = 0", axis_len, kLz, 1e-12 * kLz);
}

// ---- channel_post meshed: a disk post; the instance of the channel has no element of the post, and 'cylinder' (the
// circle of radius R about (L/4, H/2)) is a boundary of it, with normals into the post
void check_channel_post() {
  const M::ValidationReport rep = validated(read_json(data_path("channel_post.json")), "channel_post.json");
  TempDir dir;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  const M::MeshedModel m = M::mesh_model(rep, opt);
  check_atoms("channel_post", m.resolution,
              {{2001, {"named", {"cylinder"}}}, {2002, {"named", {"inlet"}}}, {2003, {"named", {"outlet"}}}, {2004, {"named", {"walls"}}}});
  const M::InstanceMesh im = m.instance_mesh({"channel"});
  check_contract("channel_post [channel]", im);
  CHECK(im.master_attribute == (std::map<int, int>{{1, 2001}, {2, 2002}, {3, 2003}, {4, 2004}}), "channel_post [channel]: master_attribute");
  CHECK(im.attributes.count("cylinder") && im.attributes.at("cylinder") == Ints{1}, "channel_post [channel]: the cylinder atom is on the boundary");
  for (int e = 0; e < im.mesh->GetNE(); ++e)
    CHECK(im.mesh->GetAttribute(e) == 1, "channel_post [channel]: element " + std::to_string(e) + " is not of the post");
  const double cx = kL / 4, cy = kH / 2, R = 0.002;
  for (int be = 0; be < im.mesh->GetNBE(); ++be) {
    if (im.mesh->GetBdrAttribute(be) != 1) continue;
    double p[2], q[2];
    bdr_ends(*im.mesh, be, p, q);
    CHECK(std::fabs(std::hypot(p[0] - cx, p[1] - cy) - R) <= 1e-12 * R && std::fabs(std::hypot(q[0] - cx, q[1] - cy) - R) <= 1e-12 * R,
          "channel_post [channel]: a 'cylinder' boundary element has its vertices on the post's circle");
  }
  const M::FacePoints fp = im.face_points("cylinder", 4);
  check_near("channel_post [channel]: face_points('cylinder'): the weights sum to 2 pi R", sum(fp.weight), 2 * kPi * R, 1e-5 * 2 * kPi * R);
  for (size_t j = 0; j < fp.weight.size() && 2 * j + 1 < fp.x.size(); ++j) {
    const double dx = fp.x[2 * j] - cx, dy = fp.x[2 * j + 1] - cy, r = std::hypot(dx, dy);
    CHECK((fp.normal[2 * j] * dx + fp.normal[2 * j + 1] * dy) / r < -0.999,
          "channel_post [channel]: the normal of 'cylinder' points out of the channel, into the post");
  }
}

// ---- the annulus at geometry order 2: face points on the curved circle, the chemistry exporter's edge nodes
void check_annulus() {
  const M::ValidationReport rep = validated(read_json(data_path("annulus.json")), "annulus.json");
  TempDir dir;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  const M::MeshedModel m = M::mesh_model(rep, opt);
  const M::InstanceMesh im = m.instance_mesh({"ring"});
  check_eq("annulus: geometry_order", im.geometry_order, 2);
  const M::FacePoints fp = im.face_points("inner", 4);
  // the P2 circle falls short of 2 pi R1 by about alpha^4 / 60 (alpha the half-angle of a segment; below 5e-6 for the
  // 24 segments the curvature sizing gives at least), and its points lie within R1 alpha^4 / 32 inside the circle
  check_near("annulus: face_points('inner'): the weights sum to 2 pi R1", sum(fp.weight), 2 * kPi * kR1, 1e-5 * 2 * kPi * kR1);
  for (size_t j = 0; j < fp.weight.size() && 2 * j + 1 < fp.x.size(); ++j) {
    const double r = std::hypot(fp.x[2 * j], fp.x[2 * j + 1]);
    CHECK(r >= kR1 * (1 - 2e-5) && r <= kR1 * (1 + 1e-12), "annulus: a point of 'inner' at r = " + num(r) + " lies on the P2 circle");
    const double radial = (fp.normal[2 * j] * fp.x[2 * j] + fp.normal[2 * j + 1] * fp.x[2 * j + 1]) / r;
    CHECK(radial < -0.999, "annulus: the normal of 'inner' points out of the ring, to the centre");
  }
  const axmp::chem::MeshInput mi = M::chem_mesh_input(im, 2);
  CHECK(!mi.edge_node_xy.empty(), "annulus: chem_mesh_input gives the curved edge nodes");
  for (size_t i = 0; 2 * i + 1 < mi.edge_node_xy.size(); ++i) {
    const double r = std::hypot(mi.edge_node_xy[2 * i], mi.edge_node_xy[2 * i + 1]);
    const double R = std::fabs(r - kR1) < std::fabs(r - 0.005) ? kR1 : 0.005;
    CHECK(std::fabs(r - R) <= 1e-12 * R, "annulus: chem edge node " + std::to_string(i) + " lies on its circle (r = " + num(r) + ")");
  }
  try {
    M::require_straight(im, "heat_planar");
    fail("annulus: require_straight accepted a curved instance mesh");
  } catch (const axmp::Error& e) {
    check_eq("annulus: require_straight: code", e.code(), "internal.mesh_export");
  }
}

// ---- the unit square of the matrix documents, meshed by the mesher: its instance numbering is the fixtures'
void check_matrix_square() {
  for (fx::Geometry g : {fx::Geometry::planar, fx::Geometry::axisymmetric}) {
    const std::string row = std::string("heat_") + fx::to_string(g);
    json doc;
    for (const M::MatrixCase& c : M::matrix_cases()) {
      if (c.row != row || c.axis != "condition_kind") continue;
      json d = M::minimal_document(c);
      if (d.is_null() || !M::validate_model(d).ok() || d.at("regions").size() != 1) continue;
      doc = d;
      break;
    }
    if (doc.is_null()) {
      fail(row + ": no matrix document of one region validates");
      continue;
    }
    const M::ValidationReport rep = M::validate_model(doc);
    TempDir dir;
    M::MeshOptions opt;
    opt.workdir = dir.path;
    const M::MeshedModel m = M::mesh_model(rep, opt);
    const M::InstanceMesh im = m.instance_mesh({"box"});
    const auto dense = fx::dense_attributes(g);
    const auto master = fx::master_numbers(g);
    for (const std::string& side : fx::side_names(g)) {
      const int a = dense.at(side);
      if (side == "axis") {
        check_eq(row + ": the axis attribute", im.axis_attributes, Ints{a});
      } else {
        CHECK(im.attributes.count(side) && im.attributes.at(side) == Ints{a},
              row + ": attributes['" + side + "'] is fixtures::dense_attributes()'s " + std::to_string(a));
      }
      CHECK(im.master_attribute.count(a) && im.master_attribute.at(a) == master.at(side),
            row + ": the master number of '" + side + "' is fixtures::master_numbers()'s " + std::to_string(master.at(side)));
    }
  }
}

// ---- the resolve_only run
void check_resolve_only() {
  const M::ValidationReport rep = validated(split_rect_doc(false), "split_rect.json");
  TempDir dir;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  const S::Resolution r = M::resolve_boundaries(rep, opt);
  check_atoms("resolve_boundaries(split_rect)", r, kSplitRectAtoms);
  Strs names;
  for (const auto& [n, tags] : r.selected) names.push_back(n);
  CHECK(names == (Strs{"cold", "hot", "interface", "mk"}), "resolve_boundaries(split_rect): every boundary name is resolved");
  CHECK(!std::filesystem::exists(dir.path + "/mesh.msh"), "resolve_boundaries meshes nothing");
}

// ---- a name with boundary faces and interior faces on one instance: the boundary faces are not dropped
// ring_a is the whole boundary of a.  On [a, b] its faces are a's outer faces (outward) and the interface (carrying the
// section's sign); on [a] all of them are boundary faces.  Divergence theorem on a = [0, L/2] x [0, H]: the weights sum
// to the perimeter L + 2H, the weighted normal sums to 0, and the weighted x . n sums to 2 area(a) = L H.
void check_ring_a() {
  json doc = split_rect_doc(false);
  doc["boundaries"]["ring_a"] = {{"of", "a"}};
  const M::ValidationReport rep = validated(doc, "split_rect with ring_a");
  TempDir dir;
  M::MeshOptions opt;
  opt.workdir = dir.path;
  const M::MeshedModel m = M::mesh_model(rep, opt);
  for (const Strs& regions : {Strs{"a", "b"}, Strs{"a"}}) {
    const std::string tag = std::string("ring_a on ") + (regions.size() == 2 ? "[a, b]" : "[a]");
    const M::InstanceMesh im = m.instance_mesh(regions);
    const M::FacePoints fp = im.face_points("ring_a", 4);
    double wx = 0.0;
    for (size_t j = 0; j < fp.weight.size(); ++j)
      wx += fp.weight[j] * (fp.x[2 * j] * fp.normal[2 * j] + fp.x[2 * j + 1] * fp.normal[2 * j + 1]);
    check_near(tag + ": the weights sum to L + 2H", sum(fp.weight), kL + 2 * kH, 1e-12 * kL);
    check_near(tag + ": the weighted normal sums to 0 (x)", weighted_normal(fp, 0), 0.0, 1e-12 * kL);
    check_near(tag + ": the weighted normal sums to 0 (y)", weighted_normal(fp, 1), 0.0, 1e-12 * kL);
    check_near(tag + ": the weighted x . n sums to L H", wx, kL * kH, 1e-12 * kL * kH);
  }
  check_near("ring_a on [a, b]: its section is the interface, of length H", m.instance_mesh({"a", "b"}).section("ring_a").length_m, kH, 1e-12 * kH);
}

// ---- a mesher that fails: with a temporary working directory the error still carries the log's text
void check_failing_mesher() {
  TempDir scripts;
  const std::string script = scripts.path + "/fake-mesher";
  {
    std::ofstream f(script);
    f << "#!/bin/sh\n"
         "echo 'first line of the log' >&2\n"
         "i=0; while [ $i -lt 60 ]; do echo \"filler $i\" >&2; i=$((i+1)); done\n"
         "echo 'the reason: gmsh could not mesh the contour' >&2\n"
         "exit 3\n";
  }
  std::filesystem::permissions(script, std::filesystem::perms::owner_all);
  const M::ValidationReport rep = validated(split_rect_doc(false), "split_rect.json");
  M::MeshOptions opt;   // workdir empty: a temporary directory, removed when the call ends
  opt.mesher = script;
  for (const char* which : {"mesh_model", "resolve_boundaries"}) {
    try {
      if (std::string(which) == "mesh_model") M::mesh_model(rep, opt);
      else M::resolve_boundaries(rep, opt);
      fail(std::string(which) + ": a failing mesher did not throw");
    } catch (const axmp::Error& e) {
      const std::string h = e.hint();
      CHECK(e.message().find("mesh generation failed") == 0, std::string(which) + ": the message");
      CHECK(h.find("the reason: gmsh could not mesh the contour") != std::string::npos, std::string(which) + ": the hint has the log's last line: " + h);
      CHECK(h.find("exit status 3") != std::string::npos, std::string(which) + ": the hint has the exit status");
      CHECK(h.find("first line of the log") == std::string::npos, std::string(which) + ": the hint has only the log's tail");
    }
  }
}

}  // namespace

int main() {
  run("a failing mesher: the error carries the log's tail", check_failing_mesher);   // needs no real mesher
  std::string found;
  try {
    found = axmp::axi::find_mesher();
  } catch (const std::exception&) {
    if (failures) return 1;
    std::printf("SKIPPED: no axmp-mesher in this build\n");
    return 77;
  }
  run("the probe of this build's mesher", [&] { check_probe(found); });
  run("split_rect meshed: numbering, orientation, face points, the read, determinism", check_split_rect);
  run("pipe_axi meshed: the axis atom", check_pipe_axi);
  run("channel_post meshed: the post is not in the channel's mesh", check_channel_post);
  run("the annulus meshed at geometry order 2: face points and edge nodes", check_annulus);
  run("the unit square of the matrix documents: the fixtures' numbering", check_matrix_square);
  run("the resolve_only run", check_resolve_only);
  run("ring_a: a name with boundary and interior faces on one instance", check_ring_a);
  if (failures) {
    std::printf("%d FAILURES\n", failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
