// The meshed model and the instance mesh (cpp/include/axionmultiphysics/model_mesh.hpp): the contract of design note
// 5.4 (the elements of the instance's regions, exactly the faces with one adjacent element as boundary elements, the
// axis kept, no interior face), the dense boundary numbering and its four maps, the maps to the master, the face sets
// of sections and their orientation (design note 5.2), the face-point instrument, the nodes rule and the two exporters'
// guards, the writer of the mesher's bounded input, the mesher probe, and the reader of the mesher's files.  Built and
// run in every build: no mesher.  The master meshes are made here, from arrays, with their master numbers and entity
// tables written by hand; the mesher-made cases are in test_model_mesh_mesher.cpp.
//
// INDEPENDENT REFERENCES.
//   * The contract of design note 5.4 evaluated with MFEM's own topology on the instance mesh (FaceIsInterior,
//     GetFaceElements, GetBdrElementFaceIndex), which does not use the library's maps.
//   * The atoms present on an instance's boundary computed here from the MASTER mesh's topology (a master boundary
//     element whose face has exactly one adjacent element in the instance's regions), and, separately, from
//     selection::partition (model_selection.hpp); numbered by fixtures::instance_numbering() (model_matrix_fixtures.hpp), written from design
//     note 5.4 by another author.
//   * Hand tables: every entity of every fixture with its sides, written direction and atom; every expected map below.
//   * Analytic lengths and normals: a face set's integral of n . e_x is computed here from the instance mesh's face
//     vertices and the set's signs (the normal of a face written from v0 to v1 is (t_y, -t_x) / |t|), and checked
//     against +H (between [a, b], normal from a to b), -H (the marker written bottom to top: tangent +e_y, left normal
//     -e_x) and +H (the marker written top to bottom).
//   * The circle: the exact length of the P2 (quadratic) interpolant of an arc in closed form, the P2 radial deviation
//     bound R alpha^4 / 32 at the half-angle alpha of a boundary edge (derived in the comment of
//     check_face_points_annulus), and the circle itself for the P2 edge nodes.
//
// CONVENTIONS (model_mesh.hpp).  Dense numbering: the master numbers present on the instance mesh's boundary, sorted,
// numbered 1..n.  A face's normal is MFEM's, out of its first element.  A section's sign is +1 when the selection's
// normal is the face's normal.  Coordinates: planar (x, y), axisymmetric (r, z).
//
// The marker is the curve 'mk_line' and the boundary that selects it 'mk' (a model document cannot give a curve and a
// boundary one name: they share a namespace).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_mesh.hpp"
#include "axionmultiphysics/model_selection.hpp"
#include "axionmultiphysics/model_validate.hpp"
#include "axionmultiphysics/spec_model.hpp"
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
std::string strs_text(const Strs& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + ("'" + v[i] + "'");
  return s + "]";
}
std::string map_text(const std::map<int, int>& m) {
  std::string s = "{";
  bool first = true;
  for (const auto& [k, v] : m) {
    s += (first ? "" : ", ") + std::to_string(k) + ": " + std::to_string(v);
    first = false;
  }
  return s + "}";
}
std::string names_map_text(const std::map<std::string, Ints>& m) {
  std::string s = "{";
  bool first = true;
  for (const auto& [k, v] : m) {
    s += (first ? "" : ", ") + k + ": " + ints_text(v);
    first = false;
  }
  return s + "}";
}
std::string num(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.17g", v);
  return b;
}
void check_eq(const std::string& what, const Ints& got, const Ints& want) {
  if (got != want) fail(what + ": got " + ints_text(got) + ", want " + ints_text(want));
}
void check_eq(const std::string& what, const Strs& got, const Strs& want) {
  if (got != want) fail(what + ": got " + strs_text(got) + ", want " + strs_text(want));
}
void check_eq(const std::string& what, int got, int want) {
  if (got != want) fail(what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}
void check_eq(const std::string& what, const std::string& got, const std::string& want) {
  if (got != want) fail(what + "\n    got:  " + got + "\n    want: " + want);
}
void check_json(const std::string& what, const json& got, const json& want) {
  if (got != want) fail(what + "\n    got:  " + got.dump() + "\n    want: " + want.dump());
}
// |got - want| <= tol
void check_near(const std::string& what, double got, double want, double tol) {
  if (!(std::fabs(got - want) <= tol))
    fail(what + ": got " + num(got) + ", want " + num(want) + " (difference " + num(got - want) + ", tolerance " + num(tol) +
         ")");
}
bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// Runs one check; an exception (the library's "not implemented yet" included) is one failure naming the check.
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

// Expects `body` to throw an exception of type E; returns its what() ("" and a failure otherwise).
template <class E>
std::string expect_throw(const std::string& what, const std::function<void()>& body) {
  try {
    body();
  } catch (const E& e) {
    return e.what();
  } catch (const std::exception& e) {
    fail(what + ": threw the wrong type: " + e.what());
    return "";
  }
  fail(what + ": did not throw");
  return "";
}
// Expects an axmp::Error with the code `code`; returns it.
axmp::Error expect_error(const std::string& what, const std::string& code, const std::function<void()>& body) {
  try {
    body();
  } catch (const axmp::Error& e) {
    check_eq(what + ": code", e.code(), code);
    return e;
  } catch (const std::exception& e) {
    fail(what + ": threw something other than axmp::Error: " + e.what());
    return axmp::Error("");
  }
  fail(what + ": did not throw");
  return axmp::Error("");
}

constexpr double kL = 0.04, kH = 0.01;           // split_rect, channel_post, rect: L x H
constexpr double kR = 0.005, kLz = 0.04;         // pipe_axi: r in [0, R], z in [0, Lz], split at Lz / 2
constexpr double kR1 = 0.002, kR2 = 0.005;       // annulus
constexpr int kTheta = 32, kRadial = 3;          // annulus: segments per circle, cells across the ring
const double kPi = std::acos(-1.0);

// ------------------------------------------------------------------------------------------- hand-built masters
// A master mesh as arrays: triangles with their region attributes, boundary segments written in the direction given,
// with their master numbers (written by hand) and the tag of their entity.
struct Arrays {
  std::vector<double> xy;
  Ints tri, tri_attr;
  Ints seg, seg_attr, seg_entity;
};

// The (nx + 1) x (ny + 1) grid of spacing h from the origin, x fastest.
struct Grid {
  int nx, ny;
  double h;
  int v(int i, int j) const { return j * (nx + 1) + i; }
  double at(int i) const { return i * h; }
};

// Every cell of the grid with a nonzero attribute, as two counter-clockwise triangles; cells row by row, i fastest, so
// that of two cells side by side the left one has the lower element indices.
Arrays grid_arrays(const Grid& g, const std::function<int(int, int)>& cell_attr) {
  Arrays a;
  for (int j = 0; j <= g.ny; ++j)
    for (int i = 0; i <= g.nx; ++i) {
      a.xy.push_back(g.at(i));
      a.xy.push_back(g.at(j));
    }
  for (int j = 0; j < g.ny; ++j)
    for (int i = 0; i < g.nx; ++i) {
      const int attr = cell_attr(i, j);
      if (attr == 0) continue;
      const int v00 = g.v(i, j), v10 = g.v(i + 1, j), v11 = g.v(i + 1, j + 1), v01 = g.v(i, j + 1);
      a.tri.insert(a.tri.end(), {v00, v10, v11, v00, v11, v01});
      a.tri_attr.insert(a.tri_attr.end(), {attr, attr});
    }
  return a;
}

// The boundary segments of the grid line from (i0, j0) to (i1, j1) (axis-parallel), written in that direction.
void add_line(Arrays& a, const Grid& g, int i0, int j0, int i1, int j1, int master, int entity) {
  const int di = (i1 > i0) - (i1 < i0), dj = (j1 > j0) - (j1 < j0);
  if ((di != 0) == (dj != 0)) throw std::logic_error("add_line: an axis-parallel line of at least one segment");
  for (int i = i0, j = j0; i != i1 || j != j1; i += di, j += dj) {
    a.seg.push_back(g.v(i, j));
    a.seg.push_back(g.v(i + di, j + dj));
    a.seg_attr.push_back(master);
    a.seg_entity.push_back(entity);
  }
}

// The moved P2 edge nodes of a curved master: the node of every edge whose two vertices lie on the circle of radius R
// about the origin (R in `circles`) is put on that circle, on the bisector of the edge.
void curve_edges(mfem::Mesh& m, const std::vector<double>& circles) {
  mfem::GridFunction* nodes = m.GetNodes();
  const mfem::FiniteElementSpace* fes = nodes->FESpace();
  for (int ed = 0; ed < m.GetNEdges(); ++ed) {
    mfem::Array<int> v;
    m.GetEdgeVertices(ed, v);
    const double *p = m.GetVertex(v[0]), *q = m.GetVertex(v[1]);
    for (double R : circles) {
      if (std::fabs(std::hypot(p[0], p[1]) - R) > 1e-12 * R || std::fabs(std::hypot(q[0], q[1]) - R) > 1e-12 * R) continue;
      const double th = std::atan2(p[1] + q[1], p[0] + q[0]);
      mfem::Array<int> d;
      fes->GetEdgeInteriorDofs(ed, d);
      if (d.Size() != 1) throw std::logic_error("curve_edges: one P2 node per edge");
      (*nodes)(fes->DofToVDof(d[0], 0)) = R * std::cos(th);
      (*nodes)(fes->DofToVDof(d[0], 1)) = R * std::sin(th);
    }
  }
  m.NodesUpdated();
}

std::unique_ptr<mfem::Mesh> to_mesh(const Arrays& a, int order, const std::vector<double>& circles = {}) {
  const int nv = static_cast<int>(a.xy.size() / 2), ne = static_cast<int>(a.tri_attr.size()),
            nb = static_cast<int>(a.seg_attr.size());
  auto m = std::make_unique<mfem::Mesh>(2, nv, ne, nb, 2);
  for (int v = 0; v < nv; ++v) m->AddVertex(a.xy[2 * v], a.xy[2 * v + 1]);
  for (int e = 0; e < ne; ++e) m->AddTriangle(a.tri[3 * e], a.tri[3 * e + 1], a.tri[3 * e + 2], a.tri_attr[e]);
  for (int b = 0; b < nb; ++b) m->AddBdrSegment(a.seg[2 * b], a.seg[2 * b + 1], a.seg_attr[b]);
  m->FinalizeTriMesh(1, 0, true);
  if (order == 2) {
    m->SetCurvature(2, false, 2, mfem::Ordering::byVDIM);
    if (!circles.empty()) curve_edges(*m, circles);
  }
  return m;
}

// A straight entity of the table written from (x0, y0) to (x1, y1) (axis-parallel): exact box, midpoint, unit tangent.
S::Entity line_entity(int tag, const std::string& s0, const std::string& s1, double x0, double y0, double x1, double y1,
                      const Strs& curves = {}, bool axis = false) {
  S::Entity e;
  e.tag = tag;
  e.side[0] = s0;
  e.side[1] = s1;
  e.curves = curves;
  e.axis = axis;
  e.type = "line";
  e.lo[0] = std::min(x0, x1);
  e.hi[0] = std::max(x0, x1);
  e.lo[1] = std::min(y0, y1);
  e.hi[1] = std::max(y0, y1);
  e.point[0] = 0.5 * (x0 + x1);
  e.point[1] = 0.5 * (y0 + y1);
  e.tangent[0] = x1 > x0 ? 1.0 : (x1 < x0 ? -1.0 : 0.0);
  e.tangent[1] = y1 > y0 ? 1.0 : (y1 < y0 ? -1.0 : 0.0);
  e.length_m = std::fabs(x1 - x0) + std::fabs(y1 - y0);
  return e;
}

// A fixture: the master's arrays, its entity table, the selectors in the mesher's input form, the region attributes,
// and the atoms by hand: master number -> {kind, names, tags}.
struct HandAtom {
  std::string kind;
  Strs names;
  Ints tags;
};
struct Fixture {
  std::string name, symmetry;
  Arrays arrays;
  S::Table table;
  json selectors;
  std::map<std::string, int> regions;
  std::map<int, HandAtom> atoms;
  std::vector<double> circles;   // the circles the curved master's boundary nodes lie on
};

// The meshed model of a fixture, built in memory: the resolution is selection::resolve's (checked against the hand atoms in
// check_fixtures).
M::MeshedModel model_of(const Fixture& f, int order) {
  M::MeshedModel m;
  m.symmetry = f.symmetry;
  m.geometry_order = order;
  m.master = to_mesh(f.arrays, order, f.circles);
  m.table = f.table;
  m.resolution = S::resolve(f.table, f.selectors);
  m.region_attribute = f.regions;
  m.boundary_entity = f.arrays.seg_entity;
  return m;
}

// split_rect: [0, L] x [0, H], n cells per H; the region a (x < L/2, attribute 1) and b (attribute 2); the marker
// 'mk_line' across a at x = L/4, written bottom to top (or, mk_down, top to bottom).  Boundaries: hot (of a, x = 0),
// cold (of b, x = L), interface (between a and b), mk (the marker); top and bottom are left unnamed.
//
//      8 (a|ext) <-  7 (a|ext) <-   6 (b|ext) <-
//   +-------------+-------------+--------------------------+
//   |             ^             ^                          ^
//   1 (a|ext)    10 (a|a) mk   9 (a|b)                     5 (b|ext)
//   v             |             |                          |
//   +-------------+-------------+--------------------------+
//      2 (a|ext) ->  3 (a|ext) ->   4 (b|ext) ->
//   x = 0        L/4           L/2                         L
//
// Atoms (by hand): 2001 [cold] {5}, 2002 [hot] {1}, 2003 [interface] {9}, 2004 [mk] {10}, 2005 unnamed (a, exterior)
// {2, 3, 7, 8}, 2006 unnamed (b, exterior) {4, 6}.
Fixture split_rect(int n, bool mk_down = false) {
  const Grid g{4 * n, n, kH / n};
  Fixture f;
  f.name = std::string("split_rect") + (mk_down ? " (marker written top to bottom)" : "");
  f.symmetry = "planar";
  f.regions = {{"a", 1}, {"b", 2}};
  f.arrays = grid_arrays(g, [n](int i, int) { return i < 2 * n ? 1 : 2; });
  const int q = n, half = 2 * n, all = 4 * n, top = n;
  struct Line {
    int tag, i0, j0, i1, j1, master;
    const char *s0, *s1;
  };
  std::vector<Line> lines = {{1, 0, top, 0, 0, 2002, "a", "exterior"},          {2, 0, 0, q, 0, 2005, "a", "exterior"},
                             {3, q, 0, half, 0, 2005, "a", "exterior"},         {4, half, 0, all, 0, 2006, "b", "exterior"},
                             {5, all, 0, all, top, 2001, "b", "exterior"},      {6, all, top, half, top, 2006, "b", "exterior"},
                             {7, half, top, q, top, 2005, "a", "exterior"},     {8, q, top, 0, top, 2005, "a", "exterior"},
                             {9, half, 0, half, top, 2003, "a", "b"}};
  lines.push_back(mk_down ? Line{10, q, top, q, 0, 2004, "a", "a"} : Line{10, q, 0, q, top, 2004, "a", "a"});
  f.table.geometry = "planar";
  f.table.regions = {"a", "b"};
  f.table.diagonal_m = std::hypot(kL, kH);
  for (const Line& l : lines) {
    add_line(f.arrays, g, l.i0, l.j0, l.i1, l.j1, l.master, l.tag);
    f.table.entities.push_back(line_entity(l.tag, l.s0, l.s1, g.at(l.i0), g.at(l.j0), g.at(l.i1), g.at(l.j1),
                                           l.tag == 10 ? Strs{"mk_line"} : Strs{}));
  }
  f.selectors = {{"hot", {{"of", {"a"}}, {"where", {{"x", {0.0}}}}}},
                 {"cold", {{"of", {"b"}}, {"where", {{"x", {g.at(all)}}}}}},
                 {"interface", {{"between", {{"a"}, {"b"}}}}},
                 {"mk", {{"curve", "mk_line"}}}};
  f.atoms = {{2001, {"named", {"cold"}, {5}}},        {2002, {"named", {"hot"}, {1}}},
             {2003, {"named", {"interface"}, {9}}},   {2004, {"named", {"mk"}, {10}}},
             {2005, {"unnamed", {}, {2, 3, 7, 8}}},  {2006, {"unnamed", {}, {4, 6}}}};
  return f;
}

// pipe_axi: r in [0, R], z in [0, Lz], n cells per R; p1 (z < Lz/2, attribute 1) and p2 (attribute 2).  Boundaries:
// inlet (of p1, z = 0), outlet (of p2, z = Lz), wall (of [p1, p2], r = R).  The axis r = 0 is two entities.
//
//   6 (p2|ext) <-                  Atoms (by hand): 900 axis {1, 2}, 2001 [inlet] {3}, 2002 [outlet] {6},
//   +------+                       2003 [wall] {4, 5}, 2004 unnamed (p1, p2) {7}.
//   2 axis |5 (p2|ext) ^
//   +--7-->+   (p2|p1)
//   1 axis |4 (p1|ext) ^
//   +------+
//   3 (p1|ext) ->
Fixture pipe_axi(int n) {
  const Grid g{n, 8 * n, kR / n};
  Fixture f;
  f.name = "pipe_axi";
  f.symmetry = "axisymmetric";
  f.regions = {{"p1", 1}, {"p2", 2}};
  f.arrays = grid_arrays(g, [n](int, int j) { return j < 4 * n ? 1 : 2; });
  const int mid = 4 * n, top = 8 * n;
  struct Line {
    int tag, i0, j0, i1, j1, master;
    const char *s0, *s1;
    bool axis;
  };
  const std::vector<Line> lines = {{1, 0, mid, 0, 0, 900, "p1", "exterior", true},
                                   {2, 0, top, 0, mid, 900, "p2", "exterior", true},
                                   {3, 0, 0, n, 0, 2001, "p1", "exterior", false},
                                   {4, n, 0, n, mid, 2003, "p1", "exterior", false},
                                   {5, n, mid, n, top, 2003, "p2", "exterior", false},
                                   {6, n, top, 0, top, 2002, "p2", "exterior", false},
                                   {7, 0, mid, n, mid, 2004, "p2", "p1", false}};
  f.table.geometry = "axisymmetric";
  f.table.regions = {"p1", "p2"};
  f.table.diagonal_m = std::hypot(kR, kLz);
  for (const Line& l : lines) {
    add_line(f.arrays, g, l.i0, l.j0, l.i1, l.j1, l.master, l.tag);
    f.table.entities.push_back(line_entity(l.tag, l.s0, l.s1, g.at(l.i0), g.at(l.j0), g.at(l.i1), g.at(l.j1), {}, l.axis));
  }
  f.selectors = {{"inlet", {{"of", {"p1"}}, {"where", {{"z", {0.0}}}}}},
                 {"outlet", {{"of", {"p2"}}, {"where", {{"z", {g.at(top)}}}}}},
                 {"wall", {{"of", json::array({"p1", "p2"})}, {"where", {{"r", {g.at(n)}}}}}}};
  f.atoms = {{900, {"axis", {}, {1, 2}}},          {2001, {"named", {"inlet"}, {3}}}, {2002, {"named", {"outlet"}, {6}}},
             {2003, {"named", {"wall"}, {4, 5}}}, {2004, {"unnamed", {}, {7}}}};
  return f;
}

// channel_post: the channel [0, L] x [0, H] (attribute 1) with a square post (attribute 2) of side H/2 centred at
// (L/4, H/2) (a square in place of the disk of the mesher's fixture: the contract does not depend on the shape).
// n is a multiple of 4.  Boundaries: inlet (x = 0), outlet (x = L), cylinder (between channel and post), walls (of
// channel except the three others).  Atoms (by hand): 2001 [cylinder] {5, 6, 7, 8}, 2002 [inlet] {1}, 2003 [outlet]
// {3}, 2004 [walls] {2, 4}.
Fixture channel_post(int n) {
  const Grid g{4 * n, n, kH / n};
  const int px0 = 3 * n / 4, px1 = 5 * n / 4, py0 = n / 4, py1 = 3 * n / 4;
  Fixture f;
  f.name = "channel_post";
  f.symmetry = "planar";
  f.regions = {{"channel", 1}, {"post", 2}};
  f.arrays = grid_arrays(g, [=](int i, int j) { return (i >= px0 && i < px1 && j >= py0 && j < py1) ? 2 : 1; });
  struct Line {
    int tag, i0, j0, i1, j1, master;
    const char *s0, *s1;
  };
  const std::vector<Line> lines = {{1, 0, n, 0, 0, 2002, "channel", "exterior"},       {2, 0, 0, 4 * n, 0, 2004, "channel", "exterior"},
                                   {3, 4 * n, 0, 4 * n, n, 2003, "channel", "exterior"}, {4, 4 * n, n, 0, n, 2004, "channel", "exterior"},
                                   {5, px0, py0, px1, py0, 2001, "post", "channel"},     {6, px1, py0, px1, py1, 2001, "post", "channel"},
                                   {7, px1, py1, px0, py1, 2001, "post", "channel"},     {8, px0, py1, px0, py0, 2001, "post", "channel"}};
  f.table.geometry = "planar";
  f.table.regions = {"channel", "post"};
  f.table.diagonal_m = std::hypot(kL, kH);
  for (const Line& l : lines) {
    add_line(f.arrays, g, l.i0, l.j0, l.i1, l.j1, l.master, l.tag);
    f.table.entities.push_back(line_entity(l.tag, l.s0, l.s1, g.at(l.i0), g.at(l.j0), g.at(l.i1), g.at(l.j1)));
  }
  f.selectors = {{"inlet", {{"of", {"channel"}}, {"where", {{"x", {0.0}}}}}},
                 {"outlet", {{"of", {"channel"}}, {"where", {{"x", {g.at(4 * n)}}}}}},
                 {"cylinder", {{"between", {{"channel"}, {"post"}}}}},
                 {"walls", {{"of", {"channel"}}, {"except", {"inlet", "outlet", "cylinder"}}}}};
  f.atoms = {{2001, {"named", {"cylinder"}, {5, 6, 7, 8}}},
             {2002, {"named", {"inlet"}, {1}}},
             {2003, {"named", {"outlet"}, {3}}},
             {2004, {"named", {"walls"}, {2, 4}}}};
  return f;
}

// rect: [0, L] x [0, H], one region 'box'; left, right, bottom, top by where.  Atoms: 2001 [bottom] {2}, 2002 [left]
// {1}, 2003 [right] {3}, 2004 [top] {4}.
Fixture rect(int n) {
  const Grid g{4 * n, n, kH / n};
  Fixture f;
  f.name = "rect";
  f.symmetry = "planar";
  f.regions = {{"box", 1}};
  f.arrays = grid_arrays(g, [](int, int) { return 1; });
  add_line(f.arrays, g, 0, n, 0, 0, 2002, 1);
  add_line(f.arrays, g, 0, 0, 4 * n, 0, 2001, 2);
  add_line(f.arrays, g, 4 * n, 0, 4 * n, n, 2003, 3);
  add_line(f.arrays, g, 4 * n, n, 0, n, 2004, 4);
  f.table.geometry = "planar";
  f.table.regions = {"box"};
  f.table.diagonal_m = std::hypot(kL, kH);
  f.table.entities = {line_entity(1, "box", "exterior", 0, g.at(n), 0, 0),
                      line_entity(2, "box", "exterior", 0, 0, g.at(4 * n), 0),
                      line_entity(3, "box", "exterior", g.at(4 * n), 0, g.at(4 * n), g.at(n)),
                      line_entity(4, "box", "exterior", g.at(4 * n), g.at(n), 0, g.at(n))};
  f.selectors = {{"left", {{"of", {"box"}}, {"where", {{"x", {0.0}}}}}},
                 {"right", {{"of", {"box"}}, {"where", {{"x", {g.at(4 * n)}}}}}},
                 {"bottom", {{"of", {"box"}}, {"where", {{"y", {0.0}}}}}},
                 {"top", {{"of", {"box"}}, {"where", {{"y", {g.at(n)}}}}}}};
  f.atoms = {{2001, {"named", {"bottom"}, {2}}},
             {2002, {"named", {"left"}, {1}}},
             {2003, {"named", {"right"}, {3}}},
             {2004, {"named", {"top"}, {4}}}};
  return f;
}

// annulus: the ring R1 < r < R2 about the origin (region 'ring'), kRadial cells across, kTheta segments per circle;
// the inner circle (entity 1, counter-clockwise: the exterior on its left) and the outer (entity 2, counter-clockwise:
// the ring on its left).  inner = between [ring, exterior] within the box of the inner circle; outer = between [ring,
// exterior] except inner.  Atoms: 2001 [inner] {1}, 2002 [outer] {2}.  At order 2 the edge nodes of the circles lie on
// them, at the angle bisector of the edge.
Fixture annulus() {
  Fixture f;
  f.name = "annulus";
  f.symmetry = "planar";
  f.regions = {{"ring", 1}};
  f.circles = {kR1, kR2};
  Arrays& a = f.arrays;
  auto v = [](int k, int j) { return k * kTheta + (j % kTheta); };
  for (int k = 0; k <= kRadial; ++k) {
    const double r = kR1 + k * (kR2 - kR1) / kRadial;
    for (int j = 0; j < kTheta; ++j) {
      const double th = 2.0 * kPi * j / kTheta;
      a.xy.push_back(r * std::cos(th));
      a.xy.push_back(r * std::sin(th));
    }
  }
  for (int k = 0; k < kRadial; ++k)
    for (int j = 0; j < kTheta; ++j) {
      a.tri.insert(a.tri.end(), {v(k, j), v(k + 1, j), v(k + 1, j + 1), v(k, j), v(k + 1, j + 1), v(k, j + 1)});
      a.tri_attr.insert(a.tri_attr.end(), {1, 1});
    }
  for (int j = 0; j < kTheta; ++j) {
    a.seg.insert(a.seg.end(), {v(0, j), v(0, j + 1)});
    a.seg_attr.push_back(2001);
    a.seg_entity.push_back(1);
  }
  for (int j = 0; j < kTheta; ++j) {
    a.seg.insert(a.seg.end(), {v(kRadial, j), v(kRadial, j + 1)});
    a.seg_attr.push_back(2002);
    a.seg_entity.push_back(2);
  }
  auto circle = [](int tag, const char* s0, const char* s1, double R) {
    S::Entity e;
    e.tag = tag;
    e.side[0] = s0;
    e.side[1] = s1;
    e.type = "circle";
    e.lo[0] = e.lo[1] = -R;
    e.hi[0] = e.hi[1] = R;
    e.point[0] = -R;            // the parametric midpoint of a counter-clockwise circle from angle 0
    e.tangent[1] = -1.0;
    e.length_m = 2.0 * kPi * R;
    return e;
  };
  f.table.geometry = "planar";
  f.table.regions = {"ring"};
  f.table.diagonal_m = 2.0 * std::sqrt(2.0) * kR2;
  f.table.entities = {circle(1, "exterior", "ring", kR1), circle(2, "ring", "exterior", kR2)};
  f.selectors = {{"inner", {{"between", {{"ring"}, "exterior"}}, {"where", {{"x", {-kR1, kR1}}, {"y", {-kR1, kR1}}}}}},
                 {"outer", {{"between", {{"ring"}, "exterior"}}, {"except", {"inner"}}}}};
  f.atoms = {{2001, {"named", {"inner"}, {1}}}, {2002, {"named", {"outer"}, {2}}}};
  return f;
}

// ------------------------------------------------------------------------------------------- references from MFEM
std::set<int> attrs_of(const Fixture& f, const Strs& regions) {
  std::set<int> out;
  for (const auto& r : regions) out.insert(f.regions.at(r));
  return out;
}

// The master numbers on the boundary of the instance of the region attributes `attrs`, from the master's topology:
// every master boundary element whose face has exactly one adjacent element in those regions.
Ints present_atoms(const mfem::Mesh& master, const std::set<int>& attrs) {
  std::set<int> out;
  for (int be = 0; be < master.GetNBE(); ++be) {
    const int f = master.GetBdrElementFaceIndex(be);
    int e1 = -1, e2 = -1;
    master.GetFaceElements(f, &e1, &e2);
    const int in = (e1 >= 0 && attrs.count(master.GetAttribute(e1))) + (e2 >= 0 && attrs.count(master.GetAttribute(e2)));
    if (in == 1) out.insert(master.GetBdrAttribute(be));
  }
  return Ints(out.begin(), out.end());
}

// The coordinates of a vertex of a mesh.
std::pair<double, double> vertex(const mfem::Mesh& m, int v) {
  const double* c = m.GetVertex(v);
  return {c[0], c[1]};
}
// The end points of face f in the order of the face (v0, v1).
void face_ends(const mfem::Mesh& m, int f, double p[2], double q[2]) {
  mfem::Array<int> v;
  m.GetFaceVertices(f, v);
  const auto a = vertex(m, v[0]), b = vertex(m, v[1]);
  p[0] = a.first, p[1] = a.second, q[0] = b.first, q[1] = b.second;
}
// The end points of boundary element be.
void bdr_ends(const mfem::Mesh& m, int be, double p[2], double q[2]) {
  mfem::Array<int> v;
  m.GetBdrElementVertices(be, v);
  const auto a = vertex(m, v[0]), b = vertex(m, v[1]);
  p[0] = a.first, p[1] = a.second, q[0] = b.first, q[1] = b.second;
}
// The sorted vertex coordinates of an element (or a face), for comparing an instance entity with its master.
std::vector<std::pair<double, double>> element_points(const mfem::Mesh& m, int e) {
  mfem::Array<int> v;
  m.GetElementVertices(e, v);
  std::vector<std::pair<double, double>> out;
  for (int i = 0; i < v.Size(); ++i) out.push_back(vertex(m, v[i]));
  std::sort(out.begin(), out.end());
  return out;
}
std::vector<std::pair<double, double>> face_points_of(const mfem::Mesh& m, int f) {
  mfem::Array<int> v;
  m.GetFaceVertices(f, v);
  std::vector<std::pair<double, double>> out;
  for (int i = 0; i < v.Size(); ++i) out.push_back(vertex(m, v[i]));
  std::sort(out.begin(), out.end());
  return out;
}

// ----------------------------------------------------------------------------------------------- expectations
struct Expect {
  std::map<int, int> master;                  // instance attribute -> master number
  std::map<std::string, Ints> attributes;     // every name -> instance attributes
  Ints remaining, axis;
};

// The instance mesh of a fresh model of `f` (built and destroyed here, so the instance mesh must stand alone).
M::InstanceMesh instance_of(const Fixture& f, int order, const Strs& regions) {
  M::MeshedModel m = model_of(f, order);
  return m.instance_mesh(regions);
}

// ------------------------------------------------------------------------------------------------------ checks
// The fixtures are what the tables say: selection::resolve of every fixture gives the hand atoms, and every master
// boundary element carries the atom of its entity.
void check_fixtures() {
  for (const Fixture& f : {split_rect(4), split_rect(4, true), pipe_axi(4), channel_post(4), rect(4), annulus()}) {
    const S::Resolution res = S::resolve(f.table, f.selectors);
    std::map<int, HandAtom> got;
    for (const S::Atom& a : res.atoms) got[a.master] = {a.kind, a.names, a.tags};
    CHECK(got.size() == f.atoms.size(), f.name + ": the number of atoms");
    for (const auto& [master, want] : f.atoms) {
      auto it = got.find(master);
      if (it == got.end()) {
        fail(f.name + ": no atom " + std::to_string(master));
        continue;
      }
      check_eq(f.name + ": atom " + std::to_string(master) + " kind", it->second.kind, want.kind);
      check_eq(f.name + ": atom " + std::to_string(master) + " names", it->second.names, want.names);
      check_eq(f.name + ": atom " + std::to_string(master) + " tags", it->second.tags, want.tags);
    }
    for (size_t b = 0; b < f.arrays.seg_attr.size(); ++b)
      CHECK(res.atom_of_tag.count(f.arrays.seg_entity[b]) && res.atom_of_tag.at(f.arrays.seg_entity[b]) == f.arrays.seg_attr[b],
            f.name + ": boundary segment " + std::to_string(b) + " carries the atom of its entity");
  }
}

// The contract of design note 5.4 on an instance mesh, from MFEM's topology only: a plain mfem::Mesh; every boundary
// element lies on a face with exactly one adjacent element, and every such face carries exactly one boundary element;
// the elements are those of the instance's regions, each mapped to a master element with the same vertices and
// attribute; every face is mapped to a master face with the same vertices.
void check_contract(const std::string& what, const M::MeshedModel& mm, const M::InstanceMesh& im, const std::set<int>& attrs) {
  if (!im.mesh) {
    fail(what + ": no mesh");
    return;
  }
  const mfem::Mesh& m = *im.mesh;
  const mfem::Mesh& master = *mm.master;
  CHECK(dynamic_cast<const mfem::SubMesh*>(&m) == nullptr, what + ": the instance mesh is a plain mfem::Mesh, not a SubMesh");
  CHECK(m.Dimension() == 2 && m.SpaceDimension() == 2, what + ": a 2D mesh");
  std::vector<int> count(m.GetNumFaces(), 0);
  for (int be = 0; be < m.GetNBE(); ++be) {
    const int f = m.GetBdrElementFaceIndex(be);
    ++count[f];
    int e1 = -1, e2 = -1;
    m.GetFaceElements(f, &e1, &e2);
    CHECK(e1 >= 0 && e2 < 0 && !m.FaceIsInterior(f),
          what + ": boundary element " + std::to_string(be) + " lies on a face with two adjacent elements");
  }
  for (int f = 0; f < m.GetNumFaces(); ++f) {
    int e1 = -1, e2 = -1;
    m.GetFaceElements(f, &e1, &e2);
    if (e2 < 0)
      CHECK(count[f] == 1, what + ": the face " + std::to_string(f) + " with one adjacent element carries " +
                               std::to_string(count[f]) + " boundary elements, not 1");
    else
      CHECK(count[f] == 0, what + ": the interior face " + std::to_string(f) + " carries a boundary element");
  }
  // the elements of the regions, and nothing else
  int master_count = 0;
  for (int e = 0; e < master.GetNE(); ++e) master_count += attrs.count(master.GetAttribute(e)) ? 1 : 0;
  check_eq(what + ": the number of elements (those of the instance's regions in the master)", m.GetNE(), master_count);
  check_eq(what + ": element_to_master has one entry per element", static_cast<int>(im.element_to_master.size()), m.GetNE());
  check_eq(what + ": face_to_master has one entry per face", static_cast<int>(im.face_to_master.size()), m.GetNumFaces());
  if (static_cast<int>(im.element_to_master.size()) == m.GetNE()) {
    std::set<int> seen;
    for (int e = 0; e < m.GetNE(); ++e) {
      const int pe = im.element_to_master[e];
      if (pe < 0 || pe >= master.GetNE()) {
        fail(what + ": element_to_master[" + std::to_string(e) + "] is not a master element");
        continue;
      }
      seen.insert(pe);
      CHECK(attrs.count(m.GetAttribute(e)), what + ": element " + std::to_string(e) + " has a region attribute of the instance");
      CHECK(m.GetAttribute(e) == master.GetAttribute(pe), what + ": element " + std::to_string(e) + " keeps its master attribute");
      CHECK(element_points(m, e) == element_points(master, pe),
            what + ": element " + std::to_string(e) + " has the vertices of its master element");
    }
    check_eq(what + ": element_to_master is injective", static_cast<int>(seen.size()), m.GetNE());
  }
  if (static_cast<int>(im.face_to_master.size()) == m.GetNumFaces())
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      const int pf = im.face_to_master[f];
      if (pf < 0 || pf >= master.GetNumFaces()) {
        fail(what + ": face_to_master[" + std::to_string(f) + "] is not a master face");
        continue;
      }
      CHECK(face_points_of(m, f) == face_points_of(master, pf), what + ": face " + std::to_string(f) + " has the vertices of its master face");
    }
  // the element attributes are not renumbered: region_attribute is the master's restricted to the instance
  std::set<int> elem_attrs;
  for (int e = 0; e < m.GetNE(); ++e) elem_attrs.insert(m.GetAttribute(e));
  CHECK(elem_attrs == attrs, what + ": the element attributes are those of the instance's regions");
}

// The dense numbering and the four maps of an instance mesh against the hand expectation and the two references.
void check_numbering(const std::string& what, const Fixture& f, const M::MeshedModel& mm, const M::InstanceMesh& im,
                     const Strs& regions, const Expect& e) {
  if (!im.mesh) {
    fail(what + ": no mesh");
    return;
  }
  const mfem::Mesh& m = *im.mesh;
  const int n = static_cast<int>(e.master.size());
  Ints want_attrs;
  for (int a = 1; a <= n; ++a) want_attrs.push_back(a);
  check_eq(what + ": bdr_attributes is 1..n", Ints(m.bdr_attributes.begin(), m.bdr_attributes.end()), want_attrs);
  if (im.master_attribute != e.master)
    fail(what + ": master_attribute: got " + map_text(im.master_attribute) + ", want " + map_text(e.master));
  // reference 1: the atoms present, from the master's topology, numbered by fixtures::instance_numbering()
  const Ints present = present_atoms(*mm.master, attrs_of(f, regions));
  std::map<int, int> inverse;
  for (const auto& [inst, master] : im.master_attribute) inverse[master] = inst;
  if (fx::instance_numbering(present) != inverse)
    fail(what + ": the numbering is not fixtures::instance_numbering() of the atoms present " + ints_text(present) +
         ": master -> instance got " + map_text(inverse));
  CHECK(inverse.size() == im.master_attribute.size(), what + ": master_attribute is injective");
  // reference 2: selection::partition with every name as a key and "remaining", plus the axis atoms
  Strs keys;
  for (const auto& [name, tags] : mm.resolution.selected) keys.push_back(name);
  keys.push_back("remaining");
  const S::Partition part = S::partition(mm.table, mm.resolution, regions, keys);
  std::set<int> covered(part.axis_atoms.begin(), part.axis_atoms.end());
  for (const auto& [key, atoms] : part.atoms_of_key) covered.insert(atoms.begin(), atoms.end());
  check_eq(what + ": the atoms present equal selection::partition of the instance", Ints(covered.begin(), covered.end()), present);
  // every boundary element carries the instance attribute of the master boundary element on its face
  if (static_cast<int>(im.face_to_master.size()) == m.GetNumFaces()) {
    const mfem::Array<int> face_to_be = mm.master->GetFaceToBdrElMap();
    for (int be = 0; be < m.GetNBE(); ++be) {
      const int pf = im.face_to_master[m.GetBdrElementFaceIndex(be)];
      const int pbe = pf >= 0 && pf < face_to_be.Size() ? face_to_be[pf] : -1;
      if (pbe < 0) {
        fail(what + ": boundary element " + std::to_string(be) + " has no master boundary element on its face");
        continue;
      }
      const auto it = im.master_attribute.find(m.GetBdrAttribute(be));
      CHECK(it != im.master_attribute.end() && it->second == mm.master->GetBdrAttribute(pbe),
            what + ": boundary element " + std::to_string(be) + " carries the instance number of its master atom " +
                std::to_string(mm.master->GetBdrAttribute(pbe)));
    }
  }
  if (im.attributes != e.attributes)
    fail(what + ": attributes: got " + names_map_text(im.attributes) + ", want " + names_map_text(e.attributes));
  check_eq(what + ": remaining_attributes", im.remaining_attributes, e.remaining);
  check_eq(what + ": axis_attributes", im.axis_attributes, e.axis);
  // the named lists (which may overlap one another), the unnamed list and the axis list partition 1..n
  std::set<int> named;
  for (const auto& [name, attrs] : im.attributes) named.insert(attrs.begin(), attrs.end());
  std::set<int> all;
  for (int a : named) all.insert(a);
  for (int a : im.remaining_attributes) {
    CHECK(!named.count(a), what + ": remaining attribute " + std::to_string(a) + " is also a named one");
    all.insert(a);
  }
  for (int a : im.axis_attributes) {
    CHECK(!named.count(a), what + ": axis attribute " + std::to_string(a) + " is also a named one");
    CHECK(std::find(im.remaining_attributes.begin(), im.remaining_attributes.end(), a) == im.remaining_attributes.end(),
          what + ": axis attribute " + std::to_string(a) + " is also a remaining one");
    all.insert(a);
  }
  check_eq(what + ": the named, remaining and axis lists cover 1..n", Ints(all.begin(), all.end()), want_attrs);
  // names_of: the names of the atom (from the resolution's atoms)
  for (const auto& [inst, master] : im.master_attribute) {
    Strs want;
    for (const S::Atom& a : mm.resolution.atoms)
      if (a.master == master) want = a.names;
    check_eq(what + ": names_of(" + std::to_string(inst) + ")", im.names_of(inst), want);
  }
  // regions and their attributes
  Strs want_regions;
  for (const auto& r : mm.table.regions)
    if (std::find(regions.begin(), regions.end(), r) != regions.end()) want_regions.push_back(r);
  check_eq(what + ": regions, in the document order", im.regions, want_regions);
  std::map<std::string, int> want_ra;
  for (const auto& r : want_regions) want_ra[r] = f.regions.at(r);
  CHECK(im.region_attribute == want_ra, what + ": region_attribute");
}

// ---- the contract (design note 5.4) on split_rect for [a, b], [a] and [b]
void check_contract_split_rect() {
  const Fixture f = split_rect(4);
  const double xmk = kL / 4, xif = kL / 2;
  for (const Strs& regions : {Strs{"a", "b"}, Strs{"a"}, Strs{"b"}}) {
    const std::string what = "split_rect " + strs_text(regions);
    M::MeshedModel mm = model_of(f, 1);
    const M::InstanceMesh im = mm.instance_mesh(regions);
    check_contract(what, mm, im, attrs_of(f, regions));
    if (!im.mesh) continue;
    const mfem::Mesh& m = *im.mesh;
    double on_interface = 0.0, on_marker = 0.0;
    std::set<int> interface_attrs;
    for (int be = 0; be < m.GetNBE(); ++be) {
      double p[2], q[2];
      bdr_ends(m, be, p, q);
      if (p[0] == xif && q[0] == xif) {
        on_interface += std::fabs(q[1] - p[1]);
        interface_attrs.insert(m.GetBdrAttribute(be));
      }
      if (p[0] == xmk && q[0] == xmk) on_marker += std::fabs(q[1] - p[1]);
    }
    CHECK(on_marker == 0.0, what + ": no boundary element on the marker (interior to every instance that contains it)");
    if (regions.size() == 2) {
      CHECK(on_interface == 0.0, what + ": no boundary element on the interface");
    } else {
      check_near(what + ": the interface is a boundary of the instance, of length H", on_interface, kH, 1e-15 * kH);
      check_eq(what + ": the interface's boundary elements carry one attribute", static_cast<int>(interface_attrs.size()), 1);
      if (interface_attrs.size() == 1) {
        const int a = *interface_attrs.begin();
        check_eq(what + ": that attribute is attributes['interface']", im.attributes.count("interface") ? im.attributes.at("interface") : Ints{},
                 Ints{a});
        CHECK(im.master_attribute.count(a) && im.master_attribute.at(a) == 2003, what + ": the interface atom is 2003");
        check_eq(what + ": names_of the interface attribute", im.names_of(a), Strs{"interface"});
      }
    }
  }
  // the regions are taken as a set and kept in the document order
  M::MeshedModel mm = model_of(f, 1);
  check_eq("split_rect [b, a]: regions", mm.instance_mesh({"b", "a"}).regions, Strs{"a", "b"});
  // a master without the interface's boundary elements (a missing atom): the instance [a], whose boundary the interface
  // is, is refused loudly (MFEM's SubMesh would give that face a fabricated attribute); [a, b], where the interface is
  // interior, is not affected
  {
    Fixture holed = split_rect(4);
    Arrays kept;
    kept.xy = holed.arrays.xy, kept.tri = holed.arrays.tri, kept.tri_attr = holed.arrays.tri_attr;
    for (size_t b = 0; b < holed.arrays.seg_attr.size(); ++b) {
      if (holed.arrays.seg_entity[b] == 9) continue;
      kept.seg.insert(kept.seg.end(), {holed.arrays.seg[2 * b], holed.arrays.seg[2 * b + 1]});
      kept.seg_attr.push_back(holed.arrays.seg_attr[b]);
      kept.seg_entity.push_back(holed.arrays.seg_entity[b]);
    }
    holed.arrays = kept;
    M::MeshedModel hm = model_of(holed, 1);
    const axmp::Error e = expect_error("split_rect without the interface's boundary elements, instance [a]", "internal.mesh_export",
                                       [&] { hm.instance_mesh({"a"}); });
    CHECK(contains(e.message(), "atom"), "the missing-atom refusal says an atom is missing: " + e.message());
    try {
      check_eq("split_rect without the interface's boundary elements, instance [a, b]: bdr_attributes",
               hm.instance_mesh({"a", "b"}).mesh->bdr_attributes.Size(), 4);
    } catch (const std::exception& x) {
      fail(std::string("split_rect without the interface's boundary elements, instance [a, b]: threw: ") + x.what());
    }
  }
  // misuse
  const std::string e1 = expect_throw<std::invalid_argument>("instance_mesh({})", [&] { mm.instance_mesh({}); });
  const std::string e2 = expect_throw<std::invalid_argument>("instance_mesh({'zz'})", [&] { mm.instance_mesh({"a", "zz"}); });
  CHECK(e2.empty() || contains(e2, "zz"), "instance_mesh({'a', 'zz'}): the message names 'zz': " + e2);
}

// The hand expectations of split_rect, pipe_axi and channel_post.
Expect expect_split_rect(const Strs& regions) {
  if (regions == Strs{"a", "b"})
    return {{{1, 2001}, {2, 2002}, {3, 2005}, {4, 2006}}, {{"cold", {1}}, {"hot", {2}}, {"interface", {}}, {"mk", {}}}, {3, 4}, {}};
  if (regions == Strs{"a"})
    return {{{1, 2002}, {2, 2003}, {3, 2005}}, {{"cold", {}}, {"hot", {1}}, {"interface", {2}}, {"mk", {}}}, {3}, {}};
  return {{{1, 2001}, {2, 2003}, {3, 2006}}, {{"cold", {1}}, {"hot", {}}, {"interface", {2}}, {"mk", {}}}, {3}, {}};
}
Expect expect_pipe_axi(const Strs& regions) {
  if (regions == Strs{"p1", "p2"})
    return {{{1, 900}, {2, 2001}, {3, 2002}, {4, 2003}}, {{"inlet", {2}}, {"outlet", {3}}, {"wall", {4}}}, {}, {1}};
  if (regions == Strs{"p1"})
    return {{{1, 900}, {2, 2001}, {3, 2003}, {4, 2004}}, {{"inlet", {2}}, {"outlet", {}}, {"wall", {3}}}, {4}, {1}};
  return {{{1, 900}, {2, 2002}, {3, 2003}, {4, 2004}}, {{"inlet", {}}, {"outlet", {2}}, {"wall", {3}}}, {4}, {1}};
}

// The maps of an instance mesh that must not depend on the mesh size, the geometry order or the run.
struct Maps {
  std::map<std::string, Ints> attributes;
  Ints remaining, axis;
  std::map<int, int> master;
  std::map<std::string, int> region_attribute;
  Strs regions;
  bool operator==(const Maps& o) const {
    return attributes == o.attributes && remaining == o.remaining && axis == o.axis && master == o.master &&
           region_attribute == o.region_attribute && regions == o.regions;
  }
};
Maps maps_of(const M::InstanceMesh& im) {
  return {im.attributes, im.remaining_attributes, im.axis_attributes, im.master_attribute, im.region_attribute, im.regions};
}

// ---- dense numbering on split_rect and pipe_axi, every domain; stability across runs, sizes and geometry orders
void check_dense_numbering() {
  struct Case {
    std::function<Fixture(int)> make;
    std::vector<Strs> domains;
    std::function<Expect(const Strs&)> expect;
  };
  const std::vector<Case> cases = {
      {[](int n) { return split_rect(n); }, {{"a", "b"}, {"a"}, {"b"}}, expect_split_rect},
      {[](int n) { return pipe_axi(n); }, {{"p1", "p2"}, {"p1"}, {"p2"}}, expect_pipe_axi}};
  for (const Case& c : cases)
    for (const Strs& regions : c.domains) {
      const Fixture f4 = c.make(4);
      const std::string what = f4.name + " " + strs_text(regions);
      M::MeshedModel mm = model_of(f4, 1);
      const M::InstanceMesh im = mm.instance_mesh(regions);
      check_numbering(what, f4, mm, im, regions, c.expect(regions));
      // the same maps: a second run, size h/2, geometry order 2
      const Maps base = maps_of(im);
      CHECK(maps_of(instance_of(f4, 1, regions)) == base, what + ": the maps of a second run are identical");
      CHECK(maps_of(instance_of(c.make(8), 1, regions)) == base, what + ": the maps at size h/2 are identical");
      CHECK(maps_of(instance_of(f4, 2, regions)) == base, what + ": the maps at geometry order 2 are identical");
    }
  // [a] and [a, b] differ on the interface atom; split_rect's remaining list is not empty
  const Fixture f = split_rect(4);
  const M::InstanceMesh ab = instance_of(f, 1, {"a", "b"}), a = instance_of(f, 1, {"a"});
  CHECK(ab.attributes.count("interface") && ab.attributes.at("interface").empty(), "split_rect [a, b]: interface has no attribute");
  CHECK(a.attributes.count("interface") && a.attributes.at("interface").size() == 1, "split_rect [a]: interface has one attribute");
  CHECK(!ab.remaining_attributes.empty(), "split_rect [a, b]: top and bottom are unnamed, so remaining_attributes is not empty");
}

// ---- channel_post, domain [channel]: no element of post; the cylinder atom on the boundary
void check_channel_post() {
  const Fixture f = channel_post(4);
  M::MeshedModel mm = model_of(f, 1);
  const M::InstanceMesh im = mm.instance_mesh({"channel"});
  check_contract("channel_post [channel]", mm, im, {1});
  check_numbering("channel_post [channel]", f, mm, im, {"channel"},
                  {{{1, 2001}, {2, 2002}, {3, 2003}, {4, 2004}}, {{"cylinder", {1}}, {"inlet", {2}}, {"outlet", {3}}, {"walls", {4}}}, {}, {}});
  if (im.mesh) {
    for (int e = 0; e < im.mesh->GetNE(); ++e)
      CHECK(im.mesh->GetAttribute(e) == 1, "channel_post [channel]: element " + std::to_string(e) + " is not of the post");
    double cyl = 0.0;
    for (int be = 0; be < im.mesh->GetNBE(); ++be) {
      if (im.mesh->GetBdrAttribute(be) != 1) continue;
      double p[2], q[2];
      bdr_ends(*im.mesh, be, p, q);
      const double x0 = 0.75 * kH, x1 = 1.25 * kH, y0 = 0.25 * kH, y1 = 0.75 * kH, t = 1e-15;
      const bool on_square = (std::fabs(p[0] - x0) < t && std::fabs(q[0] - x0) < t) || (std::fabs(p[0] - x1) < t && std::fabs(q[0] - x1) < t) ||
                             (std::fabs(p[1] - y0) < t && std::fabs(q[1] - y0) < t) || (std::fabs(p[1] - y1) < t && std::fabs(q[1] - y1) < t);
      CHECK(on_square, "channel_post [channel]: a 'cylinder' boundary element lies on the post's boundary");
      cyl += std::hypot(q[0] - p[0], q[1] - p[1]);
    }
    check_near("channel_post [channel]: the length of 'cylinder' (the post's perimeter, 2 H)", cyl, 2 * kH, 1e-15);
  }
  // the post alone: its whole boundary is the cylinder atom
  M::MeshedModel mm2 = model_of(f, 1);
  const M::InstanceMesh post = mm2.instance_mesh({"post"});
  check_contract("channel_post [post]", mm2, post, {2});
  check_numbering("channel_post [post]", f, mm2, post, {"post"},
                  {{{1, 2001}}, {{"cylinder", {1}}, {"inlet", {}}, {"outlet", {}}, {"walls", {}}}, {}, {}});
}

// ---- pipe_axi: the axis atom on every face on r = 0, none on the interface
void check_axis() {
  const Fixture f = pipe_axi(4);
  for (const Strs& regions : {Strs{"p1", "p2"}, Strs{"p1"}, Strs{"p2"}}) {
    const std::string what = "pipe_axi " + strs_text(regions);
    M::MeshedModel mm = model_of(f, 2);
    const M::InstanceMesh im = mm.instance_mesh(regions);
    check_contract(what, mm, im, attrs_of(f, regions));
    check_eq(what + ": axis_attributes", im.axis_attributes, Ints{1});
    if (!im.mesh) continue;
    const mfem::Mesh& m = *im.mesh;
    std::vector<int> axis_on_face(m.GetNumFaces(), 0);
    double axis_length = 0.0, on_interface = 0.0;
    for (int be = 0; be < m.GetNBE(); ++be) {
      double p[2], q[2];
      bdr_ends(m, be, p, q);
      if (m.GetBdrAttribute(be) == 1) {
        CHECK(p[0] == 0.0 && q[0] == 0.0, what + ": an axis boundary element lies on r = 0");
        axis_length += std::fabs(q[1] - p[1]);
        ++axis_on_face[m.GetBdrElementFaceIndex(be)];
      }
      if (p[1] == kLz / 2 && q[1] == kLz / 2) on_interface += 1.0;
    }
    for (int fc = 0; fc < m.GetNumFaces(); ++fc) {
      double p[2], q[2];
      face_ends(m, fc, p, q);
      if (p[0] == 0.0 && q[0] == 0.0)
        CHECK(axis_on_face[fc] == 1, what + ": the face " + std::to_string(fc) + " on r = 0 carries one axis boundary element");
    }
    const double want = regions.size() == 2 ? kLz : kLz / 2;
    check_near(what + ": the axis boundary elements cover r = 0 of the instance", axis_length, want, 1e-15);
    if (regions.size() == 2) CHECK(on_interface == 0.0, what + ": no boundary element on the interface z = Lz/2");
    check_eq(what + ": names_of the axis attribute", im.names_of(1), Strs{});
  }
}

// ---- the nodes rule
void check_no_nodes() {
  for (const Fixture& f : {rect(4), annulus()}) {
    const Strs regions = {f.table.regions.at(0)};
    const M::InstanceMesh o1 = instance_of(f, 1, regions);
    CHECK(o1.mesh && o1.mesh->GetNodes() == nullptr, f.name + " at geometry order 1: the instance mesh has no nodes");
    check_eq(f.name + " at geometry order 1: geometry_order", o1.geometry_order, 1);
    const M::InstanceMesh o2 = instance_of(f, 2, regions);
    CHECK(o2.mesh && o2.mesh->GetNodes() != nullptr, f.name + " at geometry order 2: the instance mesh has nodes");
    check_eq(f.name + " at geometry order 2: geometry_order", o2.geometry_order, 2);
    if (o2.mesh && o2.mesh->GetNodes())
      check_eq(f.name + " at geometry order 2: the order of the nodes", o2.mesh->GetNodes()->FESpace()->GetMaxElementOrder(), 2);
  }
  // require_straight: refuses a mesh with nodes, accepts one without
  const M::InstanceMesh curved = instance_of(annulus(), 2, {"ring"}), straight = instance_of(annulus(), 1, {"ring"}),
                        box = instance_of(rect(4), 1, {"box"});
  const axmp::Error e = expect_error("require_straight on the curved annulus", "internal.mesh_export",
                                     [&] { M::require_straight(curved, "heat_planar"); });
  check_eq("require_straight: the message", e.message(),
           "route 'heat_planar' needs straight elements, and the instance mesh of 'ring' has nodes (geometry order 2)");
  CHECK(contains(e.hint(), "please report it"), "require_straight: the hint asks for a bug report: " + e.hint());
  try {
    M::require_straight(straight, "heat_planar");
    M::require_straight(box, "heat_planar");
  } catch (const std::exception& x) {
    fail(std::string("require_straight refused a mesh without nodes: ") + x.what());
  }
}

// The integral of n . e_x over a face set, from the instance mesh's face vertices: the face written from v0 to v1 has
// the normal (t_y, -t_x) / |t|, so its integral of n . e_x is t_y; times the sign.
double flux_x(const mfem::Mesh& m, const M::FaceSet& fs) {
  double s = 0.0;
  for (size_t i = 0; i < fs.faces.size() && i < fs.sign.size(); ++i) {
    double p[2], q[2];
    face_ends(m, fs.faces[i], p, q);
    s += fs.sign[i] * (q[1] - p[1]);
  }
  return s;
}
double faces_length(const mfem::Mesh& m, const M::FaceSet& fs) {
  double s = 0.0;
  for (int f : fs.faces) {
    double p[2], q[2];
    face_ends(m, f, p, q);
    s += std::hypot(q[0] - p[0], q[1] - p[1]);
  }
  return s;
}

void check_face_set(const std::string& what, const M::InstanceMesh& im, const M::FaceSet& fs, double x_line, double want_flux) {
  if (!im.mesh) return;
  const mfem::Mesh& m = *im.mesh;
  check_eq(what + ": one sign per face", static_cast<int>(fs.sign.size()), static_cast<int>(fs.faces.size()));
  CHECK(!fs.faces.empty(), what + ": has faces");
  CHECK(std::is_sorted(fs.faces.begin(), fs.faces.end()) &&
            std::adjacent_find(fs.faces.begin(), fs.faces.end()) == fs.faces.end(),
        what + ": the faces are ascending and distinct");
  for (int s : fs.sign) CHECK(s == 1 || s == -1, what + ": a sign is +1 or -1");
  for (int f : fs.faces) {
    double p[2], q[2];
    face_ends(m, f, p, q);
    CHECK(p[0] == x_line && q[0] == x_line, what + ": face " + std::to_string(f) + " lies on x = " + num(x_line));
    int e1 = -1, e2 = -1;
    m.GetFaceElements(f, &e1, &e2);
    CHECK(e2 >= 0, what + ": face " + std::to_string(f) + " is interior to the instance");
  }
  check_near(what + ": the faces cover H", faces_length(m, fs), kH, 1e-15 * kH);
  check_near(what + ": length_m", fs.length_m, kH, 1e-14 * kH);
  check_near(what + ": the integral of n . e_x", flux_x(m, fs), want_flux, 1e-15 * kH);
}

// ---- sections: lengths and the orientation of design note 5.2, signs from the master
void check_sections() {
  for (bool down : {false, true}) {
    const Fixture f = split_rect(4, down);
    const std::string tag = down ? " (marker top to bottom)" : " (marker bottom to top)";
    const double want_mk = down ? kH : -kH;
    for (const Strs& regions : {Strs{"a", "b"}, Strs{"a"}}) {
      const std::string what = "split_rect " + strs_text(regions) + tag;
      // the model is destroyed before the instance mesh is used (instance_of): the instance mesh stands alone
      const M::InstanceMesh im = instance_of(f, 1, regions);
      const M::FaceSet mk = im.section("mk");
      check_eq(what + ": section('mk').name", mk.name, "mk");
      check_face_set(what + ": section('mk')", im, mk, kL / 4, want_mk);
      if (regions.size() == 2) {
        const M::FaceSet itf = im.section("interface");
        check_eq(what + ": section('interface').name", itf.name, "interface");
        check_face_set(what + ": section('interface')", im, itf, kL / 2, kH);
      } else {
        expect_throw<std::invalid_argument>(what + ": section('interface') (a boundary of [a], not interior)",
                                            [&] { im.section("interface"); });
      }
      expect_throw<std::invalid_argument>(what + ": section('hot') (on the boundary)", [&] { im.section("hot"); });
      const std::string e = expect_throw<std::invalid_argument>(what + ": section('nope')", [&] { im.section("nope"); });
      CHECK(e.empty() || contains(e, "nope"), what + ": section('nope') names it: " + e);
    }
  }
  // at geometry order 2 (straight nodes) the same
  const M::InstanceMesh im2 = instance_of(split_rect(4), 2, {"a", "b"});
  check_face_set("split_rect [a, b] at geometry order 2: section('mk')", im2, im2.section("mk"), kL / 4, -kH);
  check_face_set("split_rect [a, b] at geometry order 2: section('interface')", im2, im2.section("interface"), kL / 2, kH);
}

// The arrays of a FacePoints have consistent sizes; returns q (points per face), 0 when inconsistent.
int points_per_face(const std::string& what, const M::FacePoints& fp) {
  const size_t m = fp.face.size();
  if (fp.element.size() != m || m == 0) {
    fail(what + ": face and element have " + std::to_string(fp.face.size()) + " and " + std::to_string(fp.element.size()) +
         " entries (at least one face)");
    return 0;
  }
  const size_t q = fp.weight.size() / m;
  if (q == 0 || fp.weight.size() != m * q || fp.ref.size() != 2 * m * q || fp.normal.size() != 2 * m * q || fp.x.size() != 2 * m * q) {
    fail(what + ": the arrays are not (m), (m, q, 2), (m, q), (m, q, 2), (m, q, 2)");
    return 0;
  }
  return static_cast<int>(q);
}

// Every face point: the face is ascending, its element is the face's first element (the only one on the boundary),
// the reference point maps through the element's transformation to the physical point, the normal is a unit vector,
// perpendicular to a straight face, and is the face's normal times `sign_of(face)` (+1 on the boundary).
void check_points(const std::string& what, const M::InstanceMesh& im, const M::FacePoints& fp, bool straight,
                  const std::function<int(int)>& sign_of, double scale) {
  const int q = points_per_face(what, fp);
  if (!q || !im.mesh) return;
  mfem::Mesh& m = *im.mesh;
  CHECK(std::is_sorted(fp.face.begin(), fp.face.end()) && std::adjacent_find(fp.face.begin(), fp.face.end()) == fp.face.end(),
        what + ": the faces are ascending and distinct");
  for (size_t i = 0; i < fp.face.size(); ++i) {
    const int f = fp.face[i];
    int e1 = -1, e2 = -1;
    m.GetFaceElements(f, &e1, &e2);
    CHECK(fp.element[i] == e1, what + ": the element of face " + std::to_string(f) + " is its first element");
    double p[2], qq[2];
    face_ends(m, f, p, qq);
    const double tx = qq[0] - p[0], ty = qq[1] - p[1], len = std::hypot(tx, ty);
    const double fnx = ty / len, fny = -tx / len;
    mfem::ElementTransformation* T = m.GetElementTransformation(fp.element[i]);
    for (int k = 0; k < q; ++k) {
      const size_t j = i * q + k;
      mfem::IntegrationPoint ip;
      ip.Set2(fp.ref[2 * j], fp.ref[2 * j + 1]);
      mfem::Vector X(2);
      T->Transform(ip, X);
      CHECK(std::fabs(X(0) - fp.x[2 * j]) <= 1e-14 * scale && std::fabs(X(1) - fp.x[2 * j + 1]) <= 1e-14 * scale,
            what + ": the reference point " + std::to_string(k) + " of face " + std::to_string(f) +
                " maps to the physical point through the element's transformation");
      const double nx = fp.normal[2 * j], ny = fp.normal[2 * j + 1];
      CHECK(std::fabs(std::hypot(nx, ny) - 1.0) <= 1e-14, what + ": a unit normal");
      CHECK(fp.weight[j] > 0.0, what + ": a positive weight");
      if (straight) {
        // on the face's segment, and the face's normal times its sign
        const double cross = (fp.x[2 * j] - p[0]) * ty - (fp.x[2 * j + 1] - p[1]) * tx;
        CHECK(std::fabs(cross) <= 1e-15 * scale * scale, what + ": the point " + std::to_string(k) + " of face " +
                                                            std::to_string(f) + " lies on the face's line");
        const int s = sign_of(f);
        CHECK(std::fabs(nx - s * fnx) <= 1e-14 && std::fabs(ny - s * fny) <= 1e-14,
              what + ": the normal of face " + std::to_string(f) + " is " + (s > 0 ? "+" : "-") + "its face normal");
      }
    }
  }
}

double sum(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s;
}
// sum over points of w n_c (c = 0, 1), or of w |n_c|
double weighted_normal(const M::FacePoints& fp, int c, bool absolute = false) {
  double s = 0.0;
  for (size_t j = 0; j < fp.weight.size() && 2 * j + 1 < fp.normal.size(); ++j)
    s += fp.weight[j] * (absolute ? std::fabs(fp.normal[2 * j + c]) : fp.normal[2 * j + c]);
  return s;
}

// ---- face points on split_rect [a, b] (order 4)
void check_face_points() {
  const Fixture f = split_rect(4);
  const M::InstanceMesh im = instance_of(f, 1, {"a", "b"});
  if (!im.mesh) {
    fail("split_rect [a, b]: no mesh");
    return;
  }
  const int n = im.mesh->bdr_attributes.Size();
  auto outward = [](int) { return 1; };
  // named: the sections interface and mk, the boundary hot
  for (const auto& [name, want_nx] : std::vector<std::pair<std::string, double>>{{"interface", kH}, {"mk", -kH}, {"hot", -kH}}) {
    const std::string what = "split_rect [a, b]: face_points('" + name + "', 4)";
    const M::FacePoints fp = im.face_points(name, 4);
    std::function<int(int)> sign_of = outward;
    if (name != "hot") {
      const M::FaceSet fs = im.section(name);
      check_eq(what + ": the faces of the section", fp.face, fs.faces);
      sign_of = [fs](int face) {
        for (size_t i = 0; i < fs.faces.size(); ++i)
          if (fs.faces[i] == face) return fs.sign[i];
        return 0;
      };
    }
    check_points(what, im, fp, true, sign_of, kL);
    check_near(what + ": the weights sum to H", sum(fp.weight), kH, 1e-14 * kH);
    check_near(what + ": the integral of n . e_x", weighted_normal(fp, 0), want_nx, 1e-14 * kH);
    check_near(what + ": the integral of n . e_y", weighted_normal(fp, 1), 0.0, 1e-14 * kH);
  }
  // the quadrature: exact for degree 4 on hot (x = 0, y in [0, H]): the integral of y^4 is H^5 / 5
  {
    const M::FacePoints fp = im.face_points("hot", 4);
    double s = 0.0;
    for (size_t j = 0; j < fp.weight.size() && 2 * j + 1 < fp.x.size(); ++j) {
      CHECK(std::fabs(fp.x[2 * j]) <= 1e-15 * kL, "face_points('hot'): the points lie on x = 0");
      s += fp.weight[j] * std::pow(fp.x[2 * j + 1], 4);
    }
    check_near("face_points('hot', 4): the integral of y^4 is H^5 / 5 (exact at degree 4)", s, std::pow(kH, 5) / 5,
               1e-13 * std::pow(kH, 5));
  }
  // by attribute: the unnamed atoms, which have no name (top and bottom of a, and of b: L each)
  CHECK(im.remaining_attributes.size() == 2, "split_rect [a, b]: two unnamed atoms");
  double total = 0.0, ny = 0.0, any = 0.0;
  for (int a : im.remaining_attributes) {
    const std::string what = "split_rect [a, b]: face_points(attribute " + std::to_string(a) + ", 4)";
    const M::FacePoints fp = im.face_points(a, 4);
    check_points(what, im, fp, true, outward, kL);
    check_near(what + ": the weights sum to L (top and bottom of one region)", sum(fp.weight), kL, 1e-14 * kL);
    total += sum(fp.weight);
    ny += weighted_normal(fp, 1);
    any += weighted_normal(fp, 1, true);
    for (size_t j = 0; j < fp.weight.size() && 2 * j + 1 < fp.x.size(); ++j)
      CHECK(fp.x[2 * j + 1] == 0.0 || std::fabs(fp.x[2 * j + 1] - kH) <= 1e-15 * kH,
            what + ": the points lie on y = 0 or y = H");
  }
  check_near("split_rect [a, b]: the unnamed atoms' weights sum to 2L", total, 2 * kL, 1e-14 * kL);
  check_near("split_rect [a, b]: the unnamed atoms' integral of n . e_y is 0", ny, 0.0, 1e-14 * kL);
  check_near("split_rect [a, b]: the unnamed atoms' integral of |n . e_y| is 2L", any, 2 * kL, 1e-14 * kL);
  // the attribute overload over attributes['hot'] gives the arrays of the named call
  if (im.attributes.count("hot") && im.attributes.at("hot").size() == 1) {
    const M::FacePoints by_name = im.face_points("hot", 4), by_attr = im.face_points(im.attributes.at("hot")[0], 4);
    CHECK(by_name.face == by_attr.face && by_name.element == by_attr.element && by_name.ref == by_attr.ref &&
              by_name.weight == by_attr.weight && by_name.normal == by_attr.normal && by_name.x == by_attr.x,
          "split_rect [a, b]: face_points(attributes['hot'][0]) equals face_points('hot')");
  }
  // every attribute 1..n is reachable, and nothing else: 0, n + 1 and master numbers are refused
  for (int a = 1; a <= n; ++a) CHECK(!im.face_points(a, 2).face.empty(), "face_points(attribute " + std::to_string(a) + ") has faces");
  for (int bad : {0, n + 1, 2001, 2005, 2006}) {
    const std::string e = expect_throw<std::invalid_argument>("face_points(attribute " + std::to_string(bad) + ")",
                                                              [&] { im.face_points(bad, 4); });
    CHECK(e.empty() || contains(e, std::to_string(bad)), "face_points(attribute " + std::to_string(bad) + "): the message names it: " + e);
  }
  expect_throw<std::invalid_argument>("names_of(0)", [&] { im.names_of(0); });
  expect_throw<std::invalid_argument>("names_of(2001)", [&] { im.names_of(2001); });
  expect_throw<std::invalid_argument>("face_points('nope')", [&] { im.face_points("nope", 4); });
  // a name with no face on the instance
  const M::InstanceMesh a = instance_of(f, 1, {"a"});
  const std::string e = expect_throw<std::invalid_argument>("split_rect [a]: face_points('cold') (on b only)",
                                                            [&] { a.face_points("cold", 4); });
  CHECK(e.empty() || contains(e, "cold"), "split_rect [a]: face_points('cold') names it: " + e);
  // the interface as a boundary of [a] and of [b]: outward normals, +H and -H
  check_near("split_rect [a]: face_points('interface'): n . e_x outward from a", weighted_normal(a.face_points("interface", 4), 0), kH,
             1e-14 * kH);
  const M::InstanceMesh b = instance_of(f, 1, {"b"});
  check_near("split_rect [b]: face_points('interface'): n . e_x outward from b", weighted_normal(b.face_points("interface", 4), 0),
             -kH, 1e-14 * kH);
  // the reversed marker
  const M::InstanceMesh down = instance_of(split_rect(4, true), 1, {"a", "b"});
  check_near("split_rect [a, b], marker top to bottom: face_points('mk'): n . e_x", weighted_normal(down.face_points("mk", 4), 0), kH,
             1e-14 * kH);
}

// ---- face points on the curved annulus (geometry order 2)
//
// The P2 edge on a circle of radius R through the end points at -alpha and +alpha (about the edge's bisector) and the
// node at 0: x(s) = R [1 + (cos alpha - 1) s^2], y(s) = R sin(alpha) s, s in [-1, 1].  Its radius is
// R (1 - alpha^4 s^2 (1 - s^2) / 8 + O(alpha^6)), at most alpha^4 / 32 below R; its length is the closed form
// sqrt(a^2 + b^2) + (a^2 / b) asinh(b / a), a = R sin alpha, b = 2 R (1 - cos alpha), which falls short of 2 alpha R by
// about alpha^4 / 60 relative (1.54e-6 for 32 segments).  A quadrature of degree 4 integrates the length to about
// 1.3e-9 relative (3-point Gauss, measured), far inside the 1e-8 asserted.
double p2_arc_length(double R, double alpha) {
  const double a = R * std::sin(alpha), b = 2.0 * R * (1.0 - std::cos(alpha));
  return std::sqrt(a * a + b * b) + (a * a / b) * std::asinh(b / a);
}

void check_face_points_annulus() {
  const M::InstanceMesh im = instance_of(annulus(), 2, {"ring"});
  const double alpha = kPi / kTheta;
  for (const auto& [name, R, sign] : std::vector<std::tuple<std::string, double, double>>{{"inner", kR1, -1.0}, {"outer", kR2, 1.0}}) {
    const std::string what = "annulus at order 2: face_points('" + name + "', 4)";
    const M::FacePoints fp = im.face_points(name, 4);
    check_points(what, im, fp, false, [](int) { return 1; }, kR2);
    const double exact = kTheta * p2_arc_length(R, alpha);
    check_near(what + ": the weights sum to the length of the P2 circle", sum(fp.weight), exact, 1e-8 * exact);
    check_near(what + ": the P2 circle's length is 2 pi R to its geometric accuracy", exact, 2 * kPi * R, 2e-6 * 2 * kPi * R);
    const double lo = R * (1.0 - 1.1 * std::pow(alpha, 4) / 32), hi = R * (1.0 + 1e-14);
    for (size_t j = 0; j < fp.weight.size() && 2 * j + 1 < fp.x.size(); ++j) {
      const double r = std::hypot(fp.x[2 * j], fp.x[2 * j + 1]);
      CHECK(r >= lo && r <= hi, what + ": the point at radius " + num(r) + " lies on the P2 circle within R alpha^4 / 32 of R = " + num(R));
      // outward from the ring: towards the centre on the inner circle, away from it on the outer
      const double radial = (fp.normal[2 * j] * fp.x[2 * j] + fp.normal[2 * j + 1] * fp.x[2 * j + 1]) / r;
      CHECK(sign * radial > 0.999, what + ": the normal points out of the ring (radial component " + num(radial) + ")");
    }
  }
}

// ---- the chemistry exporter
void check_chem_exporter() {
  // the curved annulus, P2: the edge nodes of the circles, on them; the arrays are the instance mesh's
  const M::InstanceMesh im = instance_of(annulus(), 2, {"ring"});
  if (!im.mesh) {
    fail("annulus: no mesh");
    return;
  }
  const axmp::chem::MeshInput mi = M::chem_mesh_input(im, 2);
  const mfem::Mesh& m = *im.mesh;
  check_eq("chem_mesh_input: one vertex pair per edge node", static_cast<int>(mi.edge_node_vertices.size()),
           static_cast<int>(mi.edge_node_xy.size()));
  check_eq("chem_mesh_input: the edge nodes are those of the two circles", static_cast<int>(mi.edge_node_xy.size() / 2), 2 * kTheta);
  std::set<std::pair<int, int>> edges;
  for (int ed = 0; ed < m.GetNEdges(); ++ed) {
    mfem::Array<int> v;
    m.GetEdgeVertices(ed, v);
    edges.insert({std::min(v[0], v[1]), std::max(v[0], v[1])});
  }
  for (size_t i = 0; 2 * i + 1 < mi.edge_node_xy.size() && 2 * i + 1 < mi.edge_node_vertices.size(); ++i) {
    const int a = mi.edge_node_vertices[2 * i], b = mi.edge_node_vertices[2 * i + 1];
    CHECK(edges.count({std::min(a, b), std::max(a, b)}), "chem_mesh_input: edge node " + std::to_string(i) + " names an edge");
    const double r = std::hypot(mi.edge_node_xy[2 * i], mi.edge_node_xy[2 * i + 1]);
    const double R = std::fabs(r - kR1) < std::fabs(r - kR2) ? kR1 : kR2;
    CHECK(std::fabs(r - R) <= 1e-12 * R, "chem_mesh_input: edge node " + std::to_string(i) + " lies on its circle (r = " + num(r) + ")");
    const auto pa = vertex(m, a), pb = vertex(m, b);
    CHECK(std::fabs(std::hypot(pa.first, pa.second) - R) <= 1e-12 * R && std::fabs(std::hypot(pb.first, pb.second) - R) <= 1e-12 * R,
          "chem_mesh_input: the edge of node " + std::to_string(i) + " has its two vertices on that circle");
  }
  // vertices, triangles, attributes and boundary: the instance mesh's, in its order and numbering
  CHECK(static_cast<int>(mi.vertices.size()) == 2 * m.GetNV(), "chem_mesh_input: the vertices of the instance mesh");
  for (int v = 0; v < m.GetNV() && 2 * v + 1 < static_cast<int>(mi.vertices.size()); ++v)
    CHECK(mi.vertices[2 * v] == m.GetVertex(v)[0] && mi.vertices[2 * v + 1] == m.GetVertex(v)[1],
          "chem_mesh_input: vertex " + std::to_string(v));
  CHECK(static_cast<int>(mi.triangles.size()) == 3 * m.GetNE() && static_cast<int>(mi.element_attr.size()) == m.GetNE(),
        "chem_mesh_input: the triangles of the instance mesh");
  for (int e = 0; e < m.GetNE() && 3 * e + 2 < static_cast<int>(mi.triangles.size()); ++e) {
    mfem::Array<int> v;
    m.GetElementVertices(e, v);
    CHECK(mi.triangles[3 * e] == v[0] && mi.triangles[3 * e + 1] == v[1] && mi.triangles[3 * e + 2] == v[2] &&
              mi.element_attr[e] == m.GetAttribute(e),
          "chem_mesh_input: triangle " + std::to_string(e));
  }
  CHECK(static_cast<int>(mi.boundary.size()) == 2 * m.GetNBE() && static_cast<int>(mi.boundary_attr.size()) == m.GetNBE(),
        "chem_mesh_input: the boundary segments of the instance mesh");
  for (int be = 0; be < m.GetNBE() && 2 * be + 1 < static_cast<int>(mi.boundary.size()); ++be) {
    mfem::Array<int> v;
    m.GetBdrElementVertices(be, v);
    CHECK(mi.boundary[2 * be] == v[0] && mi.boundary[2 * be + 1] == v[1] && mi.boundary_attr[be] == m.GetBdrAttribute(be),
          "chem_mesh_input: boundary segment " + std::to_string(be) + " in the instance numbering");
  }
  // P1 on a mesh with nodes: the internal error, never straightened
  const axmp::Error e = expect_error("chem_mesh_input(curved annulus, 1)", "internal.mesh_export", [&] { M::chem_mesh_input(im, 1); });
  check_eq("chem_mesh_input(curved, 1): the message", e.message(),
           "chemistry mesh export: the instance mesh of 'ring' has nodes (geometry order 2), and the field order is 1");
  CHECK(contains(e.hint(), "please report it"), "chem_mesh_input(curved, 1): the hint asks for a bug report: " + e.hint());
  // a straight mesh: no edge node at field order 1 or 2; with straight nodes (geometry order 2), none either
  const M::InstanceMesh box1 = instance_of(rect(4), 1, {"box"}), box2 = instance_of(rect(4), 2, {"box"});
  for (int order : {1, 2}) {
    const axmp::chem::MeshInput s = M::chem_mesh_input(box1, order);
    CHECK(s.edge_node_vertices.empty() && s.edge_node_xy.empty(),
          "chem_mesh_input(straight mesh, " + std::to_string(order) + "): no edge node");
    CHECK(s.triangles.size() == 3 * static_cast<size_t>(box1.mesh ? box1.mesh->GetNE() : 0),
          "chem_mesh_input(straight mesh, " + std::to_string(order) + "): the triangles");
  }
  const axmp::chem::MeshInput s2 = M::chem_mesh_input(box2, 2);
  CHECK(s2.edge_node_vertices.empty() && s2.edge_node_xy.empty(),
        "chem_mesh_input(straight mesh with nodes, 2): no edge node (every node is at its chord midpoint)");
  expect_error("chem_mesh_input(straight mesh with nodes, 1)", "internal.mesh_export", [&] { M::chem_mesh_input(box2, 1); });
}

// ------------------------------------------------------------------------------------------ the mesher input
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

void check_mesher_input() {
  const json doc = read_json(data_path("split_rect.json"));
  const M::ValidationReport rep = validated(doc, "split_rect.json");
  const std::string dir = "/work/dir";
  const json in = M::bounded_mesher_input(rep, false, dir);
  const json geo = axmp::spec::resolved_model_geometry(rep.model);
  // the regions: enabled, in document order, attributes 1..R, the resolved shapes; the size only where written
  json shape_a, shape_b, path_mk;
  for (const json& r : geo.at("regions")) {
    if (r.at("name") == "a") shape_a = r.at("shape");
    if (r.at("name") == "b") shape_b = r.at("shape");
  }
  path_mk = geo.at("curves").at(0).at("path");
  check_json("split_rect: the resolved shape of a is numbers", shape_a.at("rect").at("r"), json::array({0.0, kL / 2}));
  const json want = {
      {"bounded", true},
      {"geometry", "planar"},
      {"exterior", "none"},
      {"regions", json::array({{{"name", "a"}, {"attr", 1}, {"shape", shape_a}, {"size", kH / 16}},
                               {{"name", "b"}, {"attr", 2}, {"shape", shape_b}}})},
      {"curves", json::array({{{"name", "mk_line"}, {"path", path_mk}}})},
      {"boundaries",
       {{"hot", {{"of", {"a"}}, {"where", {{"x", {0.0}}}}}},
        {"cold", {{"of", {"b"}}, {"where", {{"x", {kL}}}}}},
        {"interface", {{"between", {{"a"}, {"b"}}}}},
        {"mk", {{"curve", "mk_line"}}}}},
      {"size", kH / 8},
      {"surface_size", kH / 8},
      {"curvature_points", 24},
      {"order", rep.geometry_order},
      {"resolve_only", false},
      {"output", dir + "/mesh.msh"},
      {"info", dir + "/mesh_info.json"},
      {"verbose", 0}};
  check_json("bounded_mesher_input(split_rect.json)", in, want);
  check_eq("split_rect.json: the report's geometry order (the default: 2)", rep.geometry_order, 2);
  check_json("bounded_mesher_input(split_rect.json, resolve_only)", M::bounded_mesher_input(rep, true, dir).at("resolve_only"), true);

  // the written geometry order and surface size; the default size (1/40 of the diagonal of the enabled regions)
  {
    json d = doc;
    d["mesh"] = {{"geometry_order", 1}, {"surface_size", 0.0005}};
    const M::ValidationReport r = validated(d, "split_rect.json, geometry order 1, no size");
    const json i = M::bounded_mesher_input(r, false, dir);
    check_json("geometry order 1: order", i.at("order"), 1);
    const double diag = std::sqrt(kL * kL + kH * kH);   // [0, L] x [0, H]: the disabled region c is not a region
    check_near("no mesh.size: size = 1/40 of the diagonal of the enabled regions", i.at("size").get<double>(), diag / 40,
               1e-15 * diag);
    check_json("surface_size as written", i.at("surface_size"), 0.0005);
  }
  // domains expanded (a disabled region dropped), names as {"ref": NAME}, where ranges, tolerance_m and except
  {
    json d = doc;
    d["boundaries"]["ends"] = {{"between", json::array({"slab_all", "exterior"})}};
    d["boundaries"]["both"] = {{"union", json::array({"hot", {{"of", "b"}, {"where", {{"y", "H"}, {"x", json::array({"L/2", "L"})}}},
                                                             {"tolerance_m", 1e-6}, {"except", {"cold"}}}})}};
    d["boundaries"]["alias"] = "hot";
    const M::ValidationReport r = validated(d, "split_rect.json with ends, both and alias");
    const json b = M::bounded_mesher_input(r, false, dir).at("boundaries");
    check_json("between [slab_all, exterior] (c disabled)", b.at("ends"), json({{"between", json::array({json::array({"a", "b"}), "exterior"})}}));
    check_json("union of a name and a filtered of", b.at("both"),
               json({{"union", json::array({json({{"ref", "hot"}}), json({{"of", {"b"}}, {"where", {{"y", {kH}}, {"x", {kL / 2, kL}}}},
                                                                         {"tolerance_m", 1e-6}, {"except", {"cold"}}})})}}));
    check_json("a boundary written as a name", b.at("alias"), json({{"ref", "hot"}}));
  }
  // axisymmetric
  {
    const json p = read_json(data_path("pipe_axi.json"));
    const M::ValidationReport r = validated(p, "pipe_axi.json");
    const json i = M::bounded_mesher_input(r, false, dir);
    check_json("pipe_axi.json: geometry", i.at("geometry"), "axisymmetric");
    check_json("pipe_axi.json: the regions' names and attributes",
               json::array({i.at("regions").at(0).at("name"), i.at("regions").at(0).at("attr"), i.at("regions").at(1).at("name"),
                            i.at("regions").at(1).at("attr")}),
               json::array({"p1", 1, "p2", 2}));
    check_json("pipe_axi.json: the boundaries", i.at("boundaries"),
               json({{"inlet", {{"of", {"p1"}}, {"where", {{"z", {0.0}}}}}},
                     {"outlet", {{"of", {"p2"}}, {"where", {{"z", {kLz}}}}}},
                     {"wall", {{"of", json::array({"p1", "p2"})}, {"where", {{"r", {kR}}}}}}}));
  }
}

// ---------------------------------------------------------------------------------------------------- the probe
struct TempDir {
  std::string path;
  TempDir() {
    std::string t = (std::filesystem::temp_directory_path() / "axmp_model_mesh_XXXXXX").string();
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
void write_script(const std::string& path, const std::string& body) {
  {
    std::ofstream out(path, std::ios::trunc);
    out << "#!/bin/sh\n" << body << "\n";
  }
  chmod(path.c_str(), 0755);
}

void check_probe() {
  TempDir tmp;
  const std::string a = tmp.path + "/mesher_a", b = tmp.path + "/mesher_b", c = tmp.path + "/mesher_c";
  const std::string loader =
      "axmp-mesher: error while loading shared libraries: libgmsh.so.4.15: cannot open shared object file: No such file or directory";
  write_script(a, "echo '" + loader + "' >&2\nexit 127");
  write_script(b, "echo 'usage: axmp-mesher input.json' >&2\nexit 2");
  write_script(c, "exit 0");
  const char* old = std::getenv("AXMP_MESHER");
  const std::string saved = old ? old : "";
  // AXMP_MESHER names a program that the loader cannot start (status 127): unavailable, with the loader's message
  setenv("AXMP_MESHER", a.c_str(), 1);
  const M::MesherStatus s1 = M::mesher_status();
  CHECK(!s1.available, "probe of a program that exits 127: not available");
  CHECK(!s1.path.empty() && std::filesystem::equivalent(s1.path, a), "probe of a program that exits 127: its path: " + s1.path);
  CHECK(contains(s1.why, "libgmsh.so.4.15: cannot open shared object file"), "probe of a program that exits 127: why: " + s1.why);
  // cached per path: the same path gives the same answer, even after the program changed
  write_script(a, "exit 2");
  const M::MesherStatus s2 = M::mesher_status();
  CHECK(!s2.available && s2.why == s1.why, "the probe is cached per path");
  // exit status 2 (the usage text): available
  setenv("AXMP_MESHER", b.c_str(), 1);
  const M::MesherStatus s3 = M::mesher_status();
  CHECK(s3.available, "probe of a program that exits 2: available");
  CHECK(s3.why.empty(), "probe of a program that exits 2: why is empty: " + s3.why);
  CHECK(!s3.path.empty() && std::filesystem::equivalent(s3.path, b), "probe of a program that exits 2: its path: " + s3.path);
  // any other status: not available, and why says which
  const M::MesherStatus s4 = M::mesher_status(c);
  CHECK(!s4.available, "probe of a program that exits 0: not available");
  CHECK(contains(s4.why, "status 0"), "probe of a program that exits 0: why names the status: " + s4.why);
  if (old) setenv("AXMP_MESHER", saved.c_str(), 1);
  else unsetenv("AXMP_MESHER");
}

// ---------------------------------------------------------------------------------------- reading the mesher's files
// The master of split_rect written as the mesher writes it: MSH 2.2 ASCII, the line elements entity by entity in the
// written direction with the physical tag of their atom and the elementary tag of their entity, the triangles with
// the region attribute; and the info file of the plan's interface section 3.2 (the fields the reader reads, and others).
void write_msh(const Fixture& f, const std::string& path) {
  const Arrays& a = f.arrays;
  std::ofstream out(path);
  out.precision(17);
  out << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
  std::set<int> atoms(a.seg_attr.begin(), a.seg_attr.end());
  out << "$PhysicalNames\n" << atoms.size() + f.regions.size() << "\n";
  for (int at : atoms) out << "1 " << at << " \"atom_" << at << "\"\n";
  for (const auto& [name, attr] : f.regions) out << "2 " << attr << " \"" << name << "\"\n";
  out << "$EndPhysicalNames\n$Nodes\n" << a.xy.size() / 2 << "\n";
  for (size_t v = 0; v < a.xy.size() / 2; ++v) out << v + 1 << " " << a.xy[2 * v] << " " << a.xy[2 * v + 1] << " 0\n";
  out << "$EndNodes\n$Elements\n" << a.seg_attr.size() + a.tri_attr.size() << "\n";
  int id = 1;
  for (size_t b = 0; b < a.seg_attr.size(); ++b)
    out << id++ << " 1 2 " << a.seg_attr[b] << " " << a.seg_entity[b] << " " << a.seg[2 * b] + 1 << " " << a.seg[2 * b + 1] + 1 << "\n";
  for (size_t e = 0; e < a.tri_attr.size(); ++e)
    out << id++ << " 2 2 " << a.tri_attr[e] << " " << 100 + a.tri_attr[e] << " " << a.tri[3 * e] + 1 << " " << a.tri[3 * e + 1] + 1
        << " " << a.tri[3 * e + 2] + 1 << "\n";
  out << "$EndElements\n";
}
void write_info(const Fixture& f, const std::string& path) {
  json info = {{"ok", true}, {"bounded", true}, {"geometry", f.table.geometry}, {"exterior", "none"}, {"order", 1},
               {"gmsh_version", "hand-written"}, {"seconds", 0.0}, {"diagonal_m", f.table.diagonal_m},
               {"tolerance_m", 1e-9 * f.table.diagonal_m}};
  json regions = json::object();
  for (const auto& [name, attr] : f.regions) regions[name] = {{"attr", attr}, {"area_m2", 0.5 * kL * kH}};
  info["regions"] = regions;
  json entities = S::to_json(f.table).at("entities");
  const S::Resolution res = S::resolve(f.table, f.selectors);
  for (json& e : entities) e["atom"] = res.atom_of_tag.at(e.at("tag").get<int>());
  info["entities"] = entities;
  json groups = json::array();
  for (const auto& [master, atom] : f.atoms) {
    json adjacent = json::array();
    for (const S::Atom& a : res.atoms)
      if (a.master == master) adjacent = json::array({a.side[0], a.side[1]});
    groups.push_back({{"attr", master}, {"name", "atom_" + std::to_string(master)}, {"names", atom.names}, {"type", atom.kind},
                      {"adjacent", adjacent}});
  }
  info["boundary_groups"] = groups;
  info["empty"] = json::object();
  info["partial"] = json::object();
  std::ofstream(path) << info.dump(1);
}

void check_read_meshed_model() {
  TempDir tmp;
  const json doc = read_json(data_path("split_rect.json"));
  for (bool down : {false, true}) {
    const Fixture f = split_rect(4, down);
    json d = doc;
    if (down) d["curves"][0]["path"]["segment"] = json::array({json::array({"L/4", "H"}), json::array({"L/4", 0})});
    const M::ValidationReport rep = validated(d, "split_rect.json");
    const std::string msh = tmp.path + "/mesh.msh", info = tmp.path + "/mesh_info.json";
    write_msh(f, msh);
    write_info(f, info);
    for (int refinements : {0, 1}) {
      const std::string what = std::string("read_meshed_model(split_rect") + (down ? ", marker top to bottom" : "") +
                               ", refinements " + std::to_string(refinements) + ")";
      M::MeshedModel m = M::read_meshed_model(rep, msh, info, refinements);
      check_eq(what + ": symmetry", m.symmetry, "planar");
      check_eq(what + ": mesh_file", m.mesh_file, msh);
      check_eq(what + ": info_file", m.info_file, info);
      check_eq(what + ": geometry_order", m.geometry_order, 1);
      CHECK(m.info.is_object() && m.info.value("ok", false), what + ": info is the info file");
      CHECK(m.region_attribute == f.regions, what + ": region_attribute");
      check_json(what + ": the table", S::to_json(m.table), S::to_json(f.table));
      const S::Resolution want = S::resolve(f.table, f.selectors);
      CHECK(m.resolution.orientation == want.orientation && m.resolution.atom_of_tag == want.atom_of_tag &&
                m.resolution.selected == want.selected,
            what + ": the resolution recomputed from the table and the document's selectors");
      if (!m.master) {
        fail(what + ": no master");
        continue;
      }
      check_eq(what + ": the number of boundary elements", m.master->GetNBE(), static_cast<int>(f.arrays.seg_attr.size()) << refinements);
      check_eq(what + ": one boundary_entity per boundary element", static_cast<int>(m.boundary_entity.size()), m.master->GetNBE());
      for (int be = 0; be < m.master->GetNBE() && be < static_cast<int>(m.boundary_entity.size()); ++be) {
        const int tag = m.boundary_entity[be];
        const S::Entity* ent = nullptr;
        for (const S::Entity& e : m.table.entities)
          if (e.tag == tag) ent = &e;
        double p[2], q[2];
        bdr_ends(*m.master, be, p, q);
        const double t = 1e-15;
        CHECK(ent && p[0] >= ent->lo[0] - t && q[0] >= ent->lo[0] - t && p[0] <= ent->hi[0] + t && q[0] <= ent->hi[0] + t &&
                  p[1] >= ent->lo[1] - t && q[1] >= ent->lo[1] - t && p[1] <= ent->hi[1] + t && q[1] <= ent->hi[1] + t,
              what + ": boundary element " + std::to_string(be) + " lies on its entity " + std::to_string(tag));
        CHECK(want.atom_of_tag.count(tag) && m.master->GetBdrAttribute(be) == want.atom_of_tag.at(tag),
              what + ": boundary element " + std::to_string(be) + " carries the atom of its entity");
      }
      const M::InstanceMesh im = m.instance_mesh({"a", "b"});
      check_face_set(what + ": section('mk')", im, im.section("mk"), kL / 4, down ? kH : -kH);
      check_face_set(what + ": section('interface')", im, im.section("interface"), kL / 2, kH);
      CHECK(maps_of(im) == maps_of(instance_of(f, 1, {"a", "b"})), what + ": the instance maps equal those of the model built in memory");
    }
  }
}

}  // namespace

int main() {
  run("the fixtures' hand atoms are selection::resolve's", check_fixtures);
  run("the contract of design note 5.4 on split_rect", check_contract_split_rect);
  run("dense numbering and the four maps", check_dense_numbering);
  run("channel_post: the post is not in the channel's mesh", check_channel_post);
  run("pipe_axi: the axis atom", check_axis);
  run("no nodes at geometry order 1; require_straight", check_no_nodes);
  run("sections: lengths and orientation, signs from the master", check_sections);
  run("face points on split_rect", check_face_points);
  run("face points on the curved annulus", check_face_points_annulus);
  run("the chemistry exporter", check_chem_exporter);
  run("the mesher's bounded input", check_mesher_input);
  run("the mesher probe", check_probe);
  run("reading the mesher's files", check_read_meshed_model);
  if (failures) {
    std::printf("%d FAILURES\n", failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
