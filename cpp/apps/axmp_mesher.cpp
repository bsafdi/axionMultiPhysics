// axmp-mesher: build and mesh an axisymmetric (r, z) geometry with gmsh/OpenCASCADE.
//
// This program links the gmsh SDK (GPL) and runs as a separate process; the
// axionMultiPhysics library only exchanges files with it, so the library itself links no GPL code.
//
// usage: axmp-mesher [--export-regions] input.json
// "plane": true meshes the (x, y) cross-section of a z-uniform cavity instead (mesh_plane()).
// input:  {"domain_radius": Rb, "domain_center_z": z0,
//          "regions": [{"name","attr","shape","size","surface_size"}...],
//          "curves": [{"name","attr","path","size","surface_size"}...],
//          "refine_regions": [names], "surface_size": h_s, "size": h_dev,
//          "far_size": h_far, "order": 2, "output": "mesh.msh", "info": "mesh.json",
//          "axis_attr": 900, "outer_attr": 901}
// The half-disk domain is centred on the axis at z = domain_center_z (default 0).
// Axisymmetric path only: a region or curve with its own "surface_size" gets a dedicated refinement
// field at its own boundary, and the rest follow the global "surface_size" within "refine_regions".
// The 3D routine uses the global "surface_size" for every refined region.
// Regions are listed in priority order: where shapes overlap, the later wins.
// Unclaimed space inside the half-disk becomes attribute "air_attr".
//
// "bounded": true meshes the regions of a model document (mesh_bounded_2d()): the domain is the union of the regions,
// with no air, no half disk and no infinity shell.
// input:  {"bounded": true, "geometry": "planar" | "axisymmetric", "exterior": "none",
//          "regions": [{"name","attr" (1, 2, ... in order),"shape","size","surface_size"}...],
//          "curves": [{"name","path","surface_size"}...], "boundaries": {name: selector (model_selection.hpp)},
//          "size": h, "surface_size": h_s, "curvature_points": 24 (0: off), "order": 1 | 2,
//          "resolve_only": false, "output": "mesh.msh", "info": "mesh.json"}
// Shapes and paths take (x, y) as "r" and "z" in both geometries; "axisymmetric" clips the regions at r >= 0; a
// contour with elliptical segments is refused. A marker's written direction: a segment or a polyline from point to
// point, an arc or a rect path counter-clockwise.
// The info file holds the entity table ("entities", each boundary entity with its "atom"), "regions", "boundaries",
// "boundary_groups", "empty" and "partial", and, unless "resolve_only", the counts of the mesh. In the mesh file
// (MSH 2.2) every atom is one physical group of dimension 1, named atom_<master> with its master number, and every
// region one of dimension 2 with its attribute; the line elements of an entity run along its "tangent".
//
// At geometry order 2 the three 2D modes check every element for tangling (untangle_2d). A mesh with a tangled element
// is repaired by gmsh's high-order optimisation, in the curved-mesh process, or refused; only then does the info file
// have "high_order": {"checked", "tangled_before", "optimised", "tangled_after"}.
#include <gmsh.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <dirent.h>
#include <fcntl.h>
#include <link.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/personality.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/mesh_quality.hpp"
#include "axionmultiphysics/model_selection.hpp"

using json = nlohmann::json;
namespace occ = gmsh::model::occ;
using DimTags = std::vector<std::pair<int, int>>;

#ifdef __linux__
// Whether this process is the curved-mesh process (run_curved_process): AXMP_MESHER_STAGE is "curved <pid>", the pid
// of the axmp-mesher that started it. Decided once, possibly before main() (sysconf below); the meshing child forked
// from that process inherits the answer.
static bool curved_stage() {
  static int stage = -1;
  if (stage < 0) {
    const char* v = std::getenv("AXMP_MESHER_STAGE");
    stage = v && std::strncmp(v, "curved ", 7) == 0 && std::atol(v + 7) == static_cast<long>(getppid()) ? 1 : 0;
  }
  return stage == 1;
}
#endif

#if defined(__linux__) && defined(__GLIBC__)
// The CPU count that the curved-mesh process sees. libgomp (at start), OpenBLAS and
// OpenCASCADE (both inside libgmsh) size arrays by these counts, and on cluster nodes with 128 and with 256 CPUs that
// gave the curved meshes of STEP and surface-dump inputs different heaps. These definitions take precedence over
// glibc's for every library. In the curved-mesh process they answer a fixed 1024: at least the possible CPUs of common
// compute nodes (typically at most 512), so that indexing by a CPU number stays in range, libgomp's affinity mask fits
// at the first try, and code that branches on more than one CPU takes the branch it takes on a large node. Elsewhere
// they answer as glibc does. Limits: on a machine with more than 1024 possible CPUs libgomp's
// first affinity query fails and its arrays depend on the node again; glibc's internal callers (malloc's arena limit,
// unused on one thread) are not covered; OpenCASCADE's default thread pool now has 1023 slots, which would start that
// many threads if its parallel booleans were enabled (gmsh's Geometry.OCCParallel; no axionMultiPhysics input sets it). glibc
// only: __sysconf is glibc's.
constexpr long kCurvedCpus = 1024;
extern "C" long int __sysconf(int name) __THROW;

extern "C" long int sysconf(int name) __THROW {
  if ((name == _SC_NPROCESSORS_CONF || name == _SC_NPROCESSORS_ONLN) && curved_stage()) return kCurvedCpus;
  return __sysconf(name);
}

extern "C" int get_nprocs_conf() __THROW {
  return static_cast<int>(curved_stage() ? kCurvedCpus : __sysconf(_SC_NPROCESSORS_CONF));
}

extern "C" int get_nprocs() __THROW {
  return static_cast<int>(curved_stage() ? kCurvedCpus : __sysconf(_SC_NPROCESSORS_ONLN));
}
#endif

#ifdef __linux__
// Whether this process runs from the copy of axmp-mesher in memory (run_curved_process): /proc/self/exe, which gmsh
// copies into its context, then reads "/memfd:axmp-mesher (deleted)" wherever axmp-mesher is installed. Allocates
// nothing.
static bool running_from_memory() {
  char path[64];
  const ssize_t n = readlink("/proc/self/exe", path, sizeof path - 1);
  return n > 0 && std::strncmp(path, "/memfd:", 7) == 0;
}
#endif

namespace {

struct Fail : std::runtime_error {
  using std::runtime_error::runtime_error;
};

double nrm(double x, double y) { return std::sqrt(x * x + y * y); }

// Names written with their double quotes replaced, per dimension: the written name -> the real one.
std::map<std::pair<int, std::string>, std::string> g_replaced_names;

std::string name_collision(const std::string& a, const std::string& b, int dim) {
  return "the names '" + a + "' and '" + b + "' (physical groups of dimension " + std::to_string(dim) +
         ") would be the same in the mesh file, where a double quote is written as a single quote (MFEM's gmsh "
         "reader cannot read one); rename one of them";
}

// A physical group named so that MFEM's gmsh reader can read the mesh back. That reader takes a name up to the next
// double quote, so a region name with one would make every load of the mesh abort (`axmp mesh` too, which reads the
// mesh back to count invalid curved elements). Such quotes are written as single quotes. The mesh info keeps the real names, and
// axionMultiPhysics reads none back from the mesh file. gmsh keeps one name per dimension, so a written name that another group
// of that dimension already has, or gets later, is refused. Names without a double quote are passed on as they are,
// with no extra work unless a name with one came before.
void add_physical_group(int dim, const std::vector<int>& tags, int attr, const std::string& name) {
  if (name.find('"') == std::string::npos) {
    if (!g_replaced_names.empty()) {
      const auto it = g_replaced_names.find({dim, name});
      if (it != g_replaced_names.end()) throw Fail(name_collision(it->second, name, dim));
    }
    gmsh::model::addPhysicalGroup(dim, tags, attr, name);
    return;
  }
  std::string readable = name;
  std::replace(readable.begin(), readable.end(), '"', '\'');
  const auto it = g_replaced_names.find({dim, readable});
  if (it != g_replaced_names.end()) throw Fail(name_collision(it->second, name, dim));
  DimTags groups;
  gmsh::model::getPhysicalGroups(groups, dim);
  for (const auto& g : groups) {
    std::string other;
    gmsh::model::getPhysicalName(g.first, g.second, other);
    if (other == readable) throw Fail(name_collision(other, name, dim));
  }
  g_replaced_names[{dim, readable}] = name;
  gmsh::model::addPhysicalGroup(dim, tags, attr, readable);
}

// Wire through points with a per-vertex fillet radius (closed).  `fillet` holds one radius per
// point; a zero radius leaves that corner sharp.
int polygon_wire(const std::vector<std::pair<double, double>>& P, const std::vector<double>& fillet) {
  const size_t n = P.size();
  std::vector<int> curves;
  const bool any = std::any_of(fillet.begin(), fillet.end(), [](double f) { return f > 0; });
  if (!any) {
    std::vector<int> pts;
    for (auto& p : P) pts.push_back(occ::addPoint(p.first, p.second, 0));
    for (size_t i = 0; i < n; ++i) curves.push_back(occ::addLine(pts[i], pts[(i + 1) % n]));
    return occ::addWire(curves, -1, true);
  }
  // Tangent points and arc centres at each vertex.
  struct Corner { double ax, ay, bx, by, cx, cy; bool round; };
  std::vector<Corner> C(n);
  std::vector<double> d(n, 0.0), elen(n, 0.0);  // tangent offset at each vertex, length of edge i -> i+1
  for (size_t i = 0; i < n; ++i) {
    auto [px, py] = P[(i + n - 1) % n];
    auto [vx, vy] = P[i];
    auto [qx, qy] = P[(i + 1) % n];
    double ux = px - vx, uy = py - vy, wx = qx - vx, wy = qy - vy;
    double lu = nrm(ux, uy), lw = nrm(wx, wy);
    if (!(lu > 0) || !(lw > 0)) throw Fail("polygon has coincident points");
    elen[i] = lw;
    ux /= lu; uy /= lu; wx /= lw; wy /= lw;
    double cosang = std::clamp(ux * wx + uy * wy, -1.0, 1.0);
    double ang = std::acos(cosang);  // interior angle between edges
    const double f = i < fillet.size() ? fillet[i] : 0.0;
    if (f <= 0 || ang < 1e-6 || std::fabs(M_PI - ang) < 1e-6) {
      C[i] = {vx, vy, vx, vy, 0, 0, false};
      continue;
    }
    d[i] = f / std::tan(ang / 2);  // distance from vertex to tangent points
    double bxv = ux + wx, byv = uy + wy, lb = nrm(bxv, byv);
    double dc = f / std::sin(ang / 2);
    C[i] = {vx + d[i] * ux, vy + d[i] * uy, vx + d[i] * wx, vy + d[i] * wy,
            vx + dc * bxv / lb, vy + dc * byv / lb, true};
  }
  // Neighbouring fillets must not overrun the edge they share.
  for (size_t i = 0; i < n; ++i) {
    const size_t j = (i + 1) % n;
    if (d[i] + d[j] > elen[i] * (1 + 1e-9))
      throw Fail("polygon fillets at vertices " + std::to_string(i) + " and " + std::to_string(j) +
                 " overlap on the edge between them (needs " + std::to_string(d[i] + d[j]) + " m of " +
                 std::to_string(elen[i]) + " m)");
  }
  // Points are deduplicated by coordinate, so that a fillet ending exactly where the next corner
  // begins (or on the next vertex) yields one point instead of two coincident ones.  The tolerance
  // is OpenCASCADE's own Precision::Confusion (1e-7 m): below it OCC would merge the points itself
  // and the zero-length edge between them would fail.
  double scale = 0;
  for (size_t i = 0; i < n; ++i) scale = std::max(scale, elen[i]);
  const double tol = std::max(1e-7, 1e-9 * scale);
  std::map<std::pair<long long, long long>, int> seen;
  auto point = [&](double x, double y) {
    const auto key = std::make_pair(std::llround(x / tol), std::llround(y / tol));
    auto it = seen.find(key);
    if (it != seen.end()) return it->second;
    return seen[key] = occ::addPoint(x, y, 0);
  };
  std::vector<int> a(n), b(n);
  for (size_t i = 0; i < n; ++i) {
    a[i] = point(C[i].ax, C[i].ay);
    b[i] = C[i].round ? point(C[i].bx, C[i].by) : a[i];
    // a fillet so small that its two tangent points coincide is no fillet at all
    if (C[i].round && b[i] == a[i]) C[i].round = false;
  }
  for (size_t i = 0; i < n; ++i) {
    if (C[i].round) {
      int c = occ::addPoint(C[i].cx, C[i].cy, 0);
      curves.push_back(occ::addCircleArc(a[i], c, b[i]));
    }
    const size_t j = (i + 1) % n;
    if (b[i] != a[j]) curves.push_back(occ::addLine(b[i], a[j]));
  }
  if (curves.size() < 3) throw Fail("polygon fillets collapsed the outline to fewer than three edges");
  return occ::addWire(curves, -1, true);
}

// Closed wire of a resolved "contour" shape (spec.cpp): lines and axis-aligned elliptical arcs, point(t) =
// center + (ar cos t, az sin t).  Each arc is one exact rational quadratic B-spline (spans of at most pi/2: control
// points E(t_a), the intersection of the end tangents c + (ar cos t_m, az sin t_m)/cos(D/2) and E(t_b), middle
// weight cos(D/2); the affine image of the exact circle representation), so the wall is smooth and the curved
// (second-order) mesh nodes are placed on the true surface.  Consecutive segments share their end points; the loop
// is closed by a line back to the start unless the last segment already ends there.
int contour_wire(const json& b) {
  const double r0 = b.at("start")[0].get<double>(), z0 = b.at("start")[1].get<double>();
  double scale = std::max(std::fabs(r0), std::fabs(z0));
  for (auto& sg : b.at("segments")) {
    if (sg.contains("line")) scale = std::max({scale, std::fabs(sg["line"][0].get<double>()), std::fabs(sg["line"][1].get<double>())});
    else scale = std::max({scale, std::fabs(sg["ellipse"]["center"][0].get<double>()) + sg["ellipse"]["axes"][0].get<double>(),
                           std::fabs(sg["ellipse"]["center"][1].get<double>()) + sg["ellipse"]["axes"][1].get<double>()});
  }
  const double tol = std::max(1e-7, 1e-9 * scale);  // OpenCASCADE's confusion tolerance (as spec.cpp)
  const int p_start = occ::addPoint(r0, z0, 0);
  int cur = p_start;
  double cr = r0, cz = z0;
  std::vector<int> curves;
  const size_t nseg = b.at("segments").size();
  auto end_point = [&](double r, double z, bool last) {
    if (last && std::hypot(r - r0, z - z0) <= tol) return p_start;
    return occ::addPoint(r, z, 0);
  };
  for (size_t i = 0; i < nseg; ++i) {
    const json& sg = b.at("segments")[i];
    const bool last = i + 1 == nseg;
    if (sg.contains("line")) {
      const double r = sg["line"][0].get<double>(), z = sg["line"][1].get<double>();
      const int p = end_point(r, z, last);
      curves.push_back(occ::addLine(cur, p));
      cur = p;
      cr = r;
      cz = z;
      continue;
    }
    const json& e = sg.at("ellipse");
    const double xc = e["center"][0].get<double>(), yc = e["center"][1].get<double>();
    const double ar = e["axes"][0].get<double>(), az = e["axes"][1].get<double>();
    const double t0 = e["t"][0].get<double>(), t1 = e["t"][1].get<double>();
    const int n = std::max(1, static_cast<int>(std::ceil(std::fabs(t1 - t0) / (M_PI / 2) - 1e-12)));
    const double D = (t1 - t0) / n, w = std::cos(0.5 * D);
    std::vector<int> pts{cur};
    std::vector<double> wts{1.0};
    std::vector<double> knots;
    std::vector<int> mult;
    for (int k = 0; k < n; ++k) {
      const double ta = t0 + k * D, tm = ta + 0.5 * D, tb = ta + D;
      pts.push_back(occ::addPoint(xc + ar * std::cos(tm) / w, yc + az * std::sin(tm) / w, 0));
      wts.push_back(w);
      const double re = xc + ar * std::cos(tb), ze = yc + az * std::sin(tb);
      pts.push_back(k + 1 == n ? end_point(re, ze, last) : occ::addPoint(re, ze, 0));
      wts.push_back(1.0);
      if (k + 1 == n) { cr = re; cz = ze; }
    }
    for (int k = 0; k <= n; ++k) {
      knots.push_back(k);
      mult.push_back(k == 0 || k == n ? 3 : 2);
    }
    curves.push_back(occ::addBSpline(pts, -1, 2, wts, knots, mult));
    cur = pts.back();
  }
  if (cur != p_start) {
    if (std::hypot(cr - r0, cz - z0) <= tol) throw Fail("contour: internal: end point not merged with the start");
    curves.push_back(occ::addLine(cur, p_start));
  }
  return occ::addWire(curves, -1, true);
}

// Build an OCC surface for a resolved shape; returns its dim-tags.
DimTags build_shape(const json& s) {
  if (!s.is_object() || s.size() != 1) throw Fail("bad shape: " + s.dump());
  const std::string k = s.begin().key();
  const json& b = s.begin().value();
  if (k == "rect") {
    double r0 = b["r"][0], r1 = b["r"][1], z0 = b["z"][0], z1 = b["z"][1];
    double f = b.value("fillet", 0.0);
    return {{2, occ::addRectangle(r0, z0, 0, r1 - r0, z1 - z0, -1, f)}};
  }
  if (k == "frame") {
    double r0 = b["r"][0], r1 = b["r"][1], z0 = b["z"][0], z1 = b["z"][1];
    double f = b.value("fillet", 0.0), t = b["thickness"];
    int outer = occ::addRectangle(r0, z0, 0, r1 - r0, z1 - z0, -1, f);
    int inner = occ::addRectangle(r0 + t, z0 + t, 0, r1 - r0 - 2 * t, z1 - z0 - 2 * t, -1, std::max(f - t, 0.0));
    DimTags out;
    std::vector<DimTags> m;
    occ::cut({{2, outer}}, {{2, inner}}, out, m);
    return out;
  }
  if (k == "polygon") {
    std::vector<std::pair<double, double>> P;
    std::vector<double> F;
    const double f0 = b.value("fillet", 0.0);
    for (auto& p : b["points"]) {
      P.push_back({p[0].get<double>(), p[1].get<double>()});
      F.push_back(p.size() > 2 ? p[2].get<double>() : f0);  // [r, z] or [r, z, fillet]
    }
    if (b.contains("fillets")) {  // parallel list of per-vertex radii wins
      const json& fl = b.at("fillets");
      if (fl.size() != P.size()) throw Fail("'fillets' must have one radius per polygon point");
      for (size_t i = 0; i < P.size(); ++i) F[i] = fl[i].get<double>();
    }
    int w = polygon_wire(P, F);
    return {{2, occ::addPlaneSurface({w})}};
  }
  if (k == "disk") {
    double cr = b["center"][0], cz = b["center"][1], R = b["radius"];
    return {{2, occ::addDisk(cr, cz, 0, R, R)}};
  }
  if (k == "contour") return {{2, occ::addPlaneSurface({contour_wire(b)})}};
  if (k == "union" || k == "difference") {
    DimTags first = build_shape(b[0]);
    DimTags rest;
    for (size_t i = 1; i < b.size(); ++i) {
      DimTags c = build_shape(b[i]);
      rest.insert(rest.end(), c.begin(), c.end());
    }
    DimTags out;
    std::vector<DimTags> m;
    if (k == "union") occ::fuse(first, rest, out, m);
    else occ::cut(first, rest, out, m);
    return out;
  }
  throw Fail("unknown shape type " + k);
}

DimTags build_path(const json& s) {
  const std::string k = s.begin().key();
  const json& b = s.begin().value();
  DimTags out;
  auto line_through = [&](const std::vector<std::pair<double, double>>& P, bool closed) {
    std::vector<int> pts;
    for (auto& p : P) pts.push_back(occ::addPoint(p.first, p.second, 0));
    size_t n = P.size();
    for (size_t i = 0; i + 1 < n; ++i) out.push_back({1, occ::addLine(pts[i], pts[i + 1])});
    if (closed) out.push_back({1, occ::addLine(pts[n - 1], pts[0])});
  };
  if (k == "segment") {
    line_through({{b[0][0], b[0][1]}, {b[1][0], b[1][1]}}, false);
  } else if (k == "polyline") {
    std::vector<std::pair<double, double>> P;
    for (auto& p : b["points"]) P.push_back({p[0].get<double>(), p[1].get<double>()});
    line_through(P, b.value("closed", false));
  } else if (k == "rect") {
    double r0 = b["r"][0], r1 = b["r"][1], z0 = b["z"][0], z1 = b["z"][1];
    double f = b.value("fillet", 0.0);
    int sfc = occ::addRectangle(r0, z0, 0, r1 - r0, z1 - z0, -1, f);
    DimTags bd;
    gmsh::model::occ::synchronize();
    gmsh::model::getBoundary({{2, sfc}}, bd, false, false, false);
    for (auto& e : bd) out.push_back({1, std::abs(e.second)});
    // keep the curves, drop the surface
    occ::remove({{2, sfc}}, false);
  } else if (k == "arc") {
    double cr = b["center"][0], cz = b["center"][1], R = b["radius"];
    double a0 = b["angles"][0], a1 = b["angles"][1];
    out.push_back({1, occ::addCircle(cr, cz, 0, R, -1, a0, a1)});
  } else {
    throw Fail("unknown path type " + k);
  }
  return out;
}

// Rotation taking +z to the unit vector (ax, ay, az): returns (axis, angle) for occ::rotate.
void z_to_axis(double ax, double ay, double az, double& rx, double& ry, double& rz, double& ang) {
  double n = std::sqrt(ax * ax + ay * ay + az * az);
  ax /= n; ay /= n; az /= n;
  // axis = z x a, angle = acos(z . a)
  rx = -ay; ry = ax; rz = 0;
  double s = std::sqrt(rx * rx + ry * ry);
  ang = std::acos(std::clamp(az, -1.0, 1.0));
  if (s < 1e-12) { rx = 1; ry = 0; rz = 0; ang = az > 0 ? 0.0 : M_PI; }
}

DimTags build_solid(const json& s) {
  if (!s.is_object()) throw Fail("bad solid: " + s.dump());
  std::string k;
  for (auto it = s.begin(); it != s.end(); ++it) if (it.key() != "rotate" && it.key() != "translate") k = it.key();
  const json& b = s.at(k);
  auto v = [](const json& a, int i) { return a[i].get<double>(); };
  DimTags out;
  if (k == "box") {
    const json &mn = b["min"], &mx = b["max"];
    out = {{3, occ::addBox(v(mn, 0), v(mn, 1), v(mn, 2), v(mx, 0) - v(mn, 0), v(mx, 1) - v(mn, 1), v(mx, 2) - v(mn, 2))}};
  } else if (k == "cylinder") {
    const json &p = b["base"], &a = b["axis"];
    out = {{3, occ::addCylinder(v(p, 0), v(p, 1), v(p, 2), v(a, 0), v(a, 1), v(a, 2), b["radius"].get<double>())}};
  } else if (k == "cone") {
    const json &p = b["base"], &a = b["axis"];
    out = {{3, occ::addCone(v(p, 0), v(p, 1), v(p, 2), v(a, 0), v(a, 1), v(a, 2), b["r1"].get<double>(), b["r2"].get<double>())}};
  } else if (k == "sphere") {
    const json& c = b["center"];
    out = {{3, occ::addSphere(v(c, 0), v(c, 1), v(c, 2), b["radius"].get<double>())}};
  } else if (k == "torus") {
    const json &c = b["center"], &a = b["axis"];
    out = {{3, occ::addTorus(v(c, 0), v(c, 1), v(c, 2), b["major"].get<double>(), b["minor"].get<double>(), -1,
                             2 * M_PI, {v(a, 0), v(a, 1), v(a, 2)})}};
  } else if (k == "revolve") {
    // profile in (r, z) built in the x-y plane, rotated to the x-z plane, revolved about +z,
    // then mapped to the requested axis and centre.
    DimTags prof = build_shape(b["profile"]);
    occ::rotate(prof, 0, 0, 0, 1, 0, 0, M_PI / 2);  // (x, y, 0) -> (x, 0, y)
    double a0 = b["angles"][0].get<double>() * M_PI / 180, a1 = b["angles"][1].get<double>() * M_PI / 180;
    if (a0 != 0) occ::rotate(prof, 0, 0, 0, 0, 0, 1, a0);
    DimTags rv;
    occ::revolve(prof, 0, 0, 0, 0, 0, 1, a1 - a0, rv);
    for (auto& e : rv) if (e.first == 3) out.push_back(e);
    occ::remove(prof, true);
    const json &c = b["center"], &a = b["axis"];
    double rx, ry, rz, ang;
    z_to_axis(v(a, 0), v(a, 1), v(a, 2), rx, ry, rz, ang);
    if (ang != 0) occ::rotate(out, 0, 0, 0, rx, ry, rz, ang);
    if (v(c, 0) != 0 || v(c, 1) != 0 || v(c, 2) != 0) occ::translate(out, v(c, 0), v(c, 1), v(c, 2));
  } else if (k == "extrude") {
    std::vector<int> pts, lines;
    double z0 = b["z"][0], z1 = b["z"][1];
    for (auto& q : b["polygon"]) pts.push_back(occ::addPoint(q[0].get<double>(), q[1].get<double>(), z0));
    for (size_t i = 0; i < pts.size(); ++i) lines.push_back(occ::addLine(pts[i], pts[(i + 1) % pts.size()]));
    int w = occ::addCurveLoop(lines);
    int f = occ::addPlaneSurface({w});
    DimTags ex;
    occ::extrude({{2, f}}, 0, 0, z1 - z0, ex);
    for (auto& e : ex) if (e.first == 3) out.push_back(e);
  } else if (k == "wedge_frame") {
    // Boolean-free polyhedral frame (for example the coil cases of a toroidal magnet): the region
    // {(x, z) inside "polygon" and outside the optional "hole", -(tan_a x - c_neg) <= y <= tan_a x - c_pos} in local coordinates
    // (x, y, z); both side faces are planes and every band face is a planar trapezoid.  Built directly from
    // points, lines and plane surfaces; both band edges are parallel to y, so every band face is planar (an extrusion clipped by two half-space boxes costs minutes of OCC
    // booleans per solid at ~130 polygon vertices).  Use the generic "rotate"/"translate" keys to place it.
    const double ta = b.at("tan_a").get<double>(), cp = b.at("c_pos").get<double>(), cn = b.at("c_neg").get<double>();
    auto ring = [&](const json& poly, std::vector<int>& top, std::vector<int>& bot, std::vector<int>& ltop,
                    std::vector<int>& lbot, std::vector<int>& lv) {
      for (auto& q : poly) {
        const double x = q[0].get<double>(), z = q[1].get<double>(), wp = ta * x - cp, wn = ta * x - cn;
        if (!(wp + wn > 0)) throw Fail("wedge_frame: non-positive width at x = " + std::to_string(x));
        top.push_back(occ::addPoint(x, wp, z));
        bot.push_back(occ::addPoint(x, -wn, z));
      }
      const size_t n = top.size();
      for (size_t i = 0; i < n; ++i) {
        ltop.push_back(occ::addLine(top[i], top[(i + 1) % n]));
        lbot.push_back(occ::addLine(bot[i], bot[(i + 1) % n]));
        lv.push_back(occ::addLine(top[i], bot[i]));
      }
    };
    std::vector<int> ot, ob, olt, olb, olv, ht, hb, hlt, hlb, hlv;
    ring(b.at("polygon"), ot, ob, olt, olb, olv);
    const bool hole = b.contains("hole");
    if (hole) ring(b.at("hole"), ht, hb, hlt, hlb, hlv);
    std::vector<int> faces;
    {
      std::vector<int> w1 = {occ::addCurveLoop(olt)}, w2 = {occ::addCurveLoop(olb)};
      if (hole) { w1.push_back(occ::addCurveLoop(hlt)); w2.push_back(occ::addCurveLoop(hlb)); }
      faces.push_back(occ::addPlaneSurface(w1));
      faces.push_back(occ::addPlaneSurface(w2));
    }
    auto band = [&](const std::vector<int>& lt, const std::vector<int>& lb, const std::vector<int>& lv) {
      const size_t n = lt.size();
      for (size_t i = 0; i < n; ++i)
        faces.push_back(occ::addPlaneSurface({occ::addCurveLoop({lt[i], lv[(i + 1) % n], lb[i], lv[i]})}));
    };
    band(olt, olb, olv);
    if (hole) band(hlt, hlb, hlv);
    out = {{3, occ::addVolume({occ::addSurfaceLoop(faces)})}};
  } else if (k == "step") {
    DimTags imp;
    occ::importShapes(b["file"].get<std::string>(), imp, true);
    for (auto& e : imp) if (e.first == 3) out.push_back(e);
    double sc = b.value("scale", 1.0);
    if (sc != 1.0) occ::dilate(out, 0, 0, 0, sc, sc, sc);
    if (out.empty()) throw Fail("STEP file contains no solids: " + b["file"].get<std::string>());
  } else if (k == "union" || k == "difference" || k == "intersection") {
    DimTags first = build_solid(b[0]), rest;
    for (size_t i = 1; i < b.size(); ++i) {
      DimTags c = build_solid(b[i]);
      rest.insert(rest.end(), c.begin(), c.end());
    }
    std::vector<DimTags> m;
    if (k == "union") occ::fuse(first, rest, out, m);
    else if (k == "difference") occ::cut(first, rest, out, m);
    else occ::intersect(first, rest, out, m);
  } else {
    throw Fail("unknown solid " + k);
  }
  if (s.contains("rotate")) {
    const json& r = s["rotate"];
    const json& ax = r["axis"];
    const json o = r.value("origin", json::array({0, 0, 0}));
    occ::rotate(out, v(o, 0), v(o, 1), v(o, 2), v(ax, 0), v(ax, 1), v(ax, 2), r["angle_deg"].get<double>() * M_PI / 180);
  }
  if (s.contains("translate")) {
    const json& t = s["translate"];
    occ::translate(out, v(t, 0), v(t, 1), v(t, 2));
  }
  return out;
}

// Adaptive refinement: a target element size on the elements of a previous mesh, read from the binary
// file in["size_field_file"] (int32 nodes per element (3: triangles, 4: tetrahedra), int64 count, then per
// element the node coordinates (x, y, z per node) and the target size at each node, float64). It becomes a
// list-based post-processing view used as a PostView size field (combined with the others by Min).
// Returns the field tag (-1 without a file) and lowers *hmin to the smallest requested size.
int size_field_from_file(const json& in, double* hmin) {
  const std::string path = in.value("size_field_file", std::string());
  if (path.empty()) return -1;
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read size field file " + path);
  int32_t nn = 0;
  int64_t ne = 0;
  f.read(reinterpret_cast<char*>(&nn), sizeof nn);
  f.read(reinterpret_cast<char*>(&ne), sizeof ne);
  if (nn != 3 && nn != 4) throw std::runtime_error("size field: 3 or 4 nodes per element expected");
  std::vector<double> rec(4 * nn), data;
  data.reserve(static_cast<size_t>(ne) * 4 * nn);
  for (int64_t e = 0; e < ne; ++e) {
    f.read(reinterpret_cast<char*>(rec.data()), sizeof(double) * rec.size());
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < nn; ++k) data.push_back(rec[3 * k + c]);  // x1..xn, y1..yn, z1..zn
    for (int k = 0; k < nn; ++k) {
      data.push_back(rec[3 * nn + k]);
      *hmin = std::min(*hmin, rec[3 * nn + k]);
    }
  }
  if (!f) throw std::runtime_error("size field file " + path + " is truncated");
  const int view = gmsh::view::add("adaptive size");
  gmsh::view::addListData(view, nn == 3 ? "ST" : "SS", static_cast<int>(ne), data);
  const int fld = gmsh::model::mesh::field::add("PostView");
  gmsh::model::mesh::field::setNumber(fld, "ViewTag", view);
  return fld;
}

// Curved tetrahedra with min det J <= 1e-3 max |det J| (gmsh's Jacobian bounds; the solve's own check samples det J
// at quadrature points with the same threshold, see curved_element_invalid in model3d.cpp).
std::size_t invalid_curved_tets() {
  std::vector<int> types;
  std::vector<std::vector<std::size_t>> tags, nodes;
  gmsh::model::mesh::getElements(types, tags, nodes, 3);
  std::size_t bad = 0;
  for (const auto& t : tags) {
    if (t.empty()) continue;
    std::vector<double> lo, hi;
    gmsh::model::mesh::getElementQualities(t, lo, "minDetJac");
    gmsh::model::mesh::getElementQualities(t, hi, "maxDetJac");
    for (std::size_t i = 0; i < lo.size(); ++i)
      if (!(lo[i] > 1e-3 * std::fabs(hi[i]))) ++bad;
  }
  return bad;
}

// Largest curvature (1/m) over the model's curves, sampled at 9 points per curve; 0 if every curve is straight.
double max_curve_curvature() {
  DimTags curves;
  gmsh::model::getEntities(curves, 1);
  double kmax = 0;
  for (const auto& cv : curves) {
    std::vector<double> pmin, pmax;
    gmsh::model::getParametrizationBounds(1, cv.second, pmin, pmax);
    std::vector<double> ts;
    for (int i = 0; i <= 8; ++i) ts.push_back(pmin[0] + (pmax[0] - pmin[0]) * i / 8.0);
    std::vector<double> k;
    gmsh::model::getCurvature(1, cv.second, ts, k);
    for (double v : k)
      if (std::isfinite(v)) kmax = std::max(kmax, std::fabs(v));
  }
  return kmax;
}

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::string(v) != "0";
}

// Threads of this process (Linux: the entries of /proc/self/task); 0 where that cannot be counted.
int thread_count() {
  int n = 0;
#ifdef __linux__
  if (DIR* d = opendir("/proc/self/task")) {
    while (const dirent* e = readdir(d))
      if (e->d_name[0] != '.') ++n;
    closedir(d);
  }
#endif
  return n;
}

// How the curved steps' process was started (main, run_curved_process): "on" in the curved-mesh process, which has a
// fixed command line, environment and file names; "off: <why>" where this process does them itself; "n/a" otherwise.
std::string g_canonical = "n/a";
// --export-regions was given (a plain bool: nothing is allocated before the mesh is made)
bool g_export_regions = false;

// Says on stderr that a curved mesh is not reproducible, and why (mesh info "high_order_isolation" and
// "high_order_canonical" record the same).
void warn_not_reproducible(const std::string& why) {
  std::cerr << "axmp-mesher: WARNING: this curved (order-2) mesh is not reproducible (" << why
            << "): the same input can give a different mesh in another run or another output directory\n";
}

// "off" when this process runs with address-space randomization disabled (Linux personality ADDR_NO_RANDOMIZE).
std::string address_randomization() {
#ifdef __linux__
  const int p = personality(0xffffffff);
  if (p != -1) return (p & ADDR_NO_RANDOMIZE) ? "off" : "on";
#endif
  return "unknown";
}

// AXMP_MESHER_HO: elastic, full, auto (the default, also when empty) or none.
std::string high_order_mode() {
  const char* v = std::getenv("AXMP_MESHER_HO");
  const std::string ho = v && *v ? v : "auto";
  if (ho != "elastic" && ho != "full" && ho != "none" && ho != "auto")
    throw Fail("AXMP_MESHER_HO must be elastic, full, auto or none, not '" + ho + "'");
  return ho;
}

// AXMP_MESHER_HO_ISOLATE: "1" or "" (the default: the curved steps run where they are reproducible) or "0" (in this
// process).
std::string isolate_mode() {
  const char* v = std::getenv("AXMP_MESHER_HO_ISOLATE");
  const std::string value = v ? v : "";
  if (value != "" && value != "0" && value != "1")
    throw Fail("AXMP_MESHER_HO_ISOLATE must be 1 (the default: reproducible curved meshes), 0 (one process) or empty, "
               "not '" + value + "'");
  return value;
}

// ------------------------------------------------------------------------------------------------- 2D curved elements
// A 2D element of geometry order 2 is tangled when its Jacobian determinant is not of one strict sign over the element
// (it changes sign or vanishes somewhere in it), or is of the sign opposite to that of the majority of the elements of
// its gmsh surface. det J of a 6-node triangle is a quadratic polynomial of the reference coordinates, so its range
// over the element is exact (mesh_quality::det_range_p2, the library's check on read): a fold between sample points is
// not missed. The majority is taken per surface, the unit that gmsh meshes in one winding and that the 2D modes turn
// counter-clockwise only after the check: a surface meshed clockwise has det J < 0 throughout, and one region can hold
// surfaces of both windings (a union with a polygon listed clockwise). Only the sign counts, not a quality threshold:
// valid meshes of thin curved regions have elements with min det J / max det J of a few 1e-3
// (examples/pickup_capacitance.json), which must be written as they are.
struct TangleScan {
  std::size_t checked = 0;  // the 6-node triangles
  std::size_t tangled = 0;
  std::string region;       // the region of the first tangled element
  double at[2] = {0, 0};    // the image of its reference centre
};

// The real name of a physical group of dimension 2 (add_physical_group writes double quotes as single quotes).
std::string region_name(int group) {
  std::string name;
  gmsh::model::getPhysicalName(2, group, name);
  const auto it = g_replaced_names.find({2, name});
  if (it != g_replaced_names.end()) return it->second;
  return name.empty() ? "attribute " + std::to_string(group) : name;
}

TangleScan scan_tangled_2d() {
  namespace mq = axmp::mesh_quality;
  // derivatives (d/du, d/dv) of the six P2 shape functions in gmsh's node order (vertices 0, 1, 2, then the mid-edge
  // nodes of the edges 01, 12, 20) at the six points where det_range_p2 takes det J
  double dN[6][12];
  for (int q = 0; q < 6; ++q) {
    const double u = mq::kP2Points[q][0], v = mq::kP2Points[q][1], w = 1 - u - v;
    const double d[12] = {1 - 4 * w, 1 - 4 * w, 4 * u - 1, 0, 0, 4 * v - 1, 4 * (w - u), -4 * u, 4 * v, 4 * u,
                          -4 * v, 4 * (w - v)};
    std::copy(d, d + 12, dN[q]);
  }
  std::vector<std::size_t> node_tags;
  std::vector<double> xyz, uvw;
  gmsh::model::mesh::getNodes(node_tags, xyz, uvw, -1, -1, true, false);
  std::size_t max_tag = 0;
  for (const std::size_t t : node_tags) max_tag = std::max(max_tag, t);
  std::vector<std::size_t> index(max_tag + 1, 0);
  for (std::size_t k = 0; k < node_tags.size(); ++k) index[node_tags[k]] = k;
  struct Element {
    int surface, region;
    double lo, hi, x, y;
  };
  std::vector<Element> all;
  std::map<int, std::array<std::size_t, 2>> signs;  // surface -> elements with det J > 0, < 0 throughout
  DimTags surfaces;
  gmsh::model::getEntities(surfaces, 2);
  for (const auto& s : surfaces) {
    std::vector<int> groups;
    gmsh::model::getPhysicalGroupsForEntity(2, s.second, groups);
    const int region = groups.empty() ? -s.second : groups.front();  // a surface in no group is named by its tag
    std::vector<std::size_t> tags, nodes;
    gmsh::model::mesh::getElementsByType(9, tags, nodes, s.second);
    for (std::size_t e = 0; e < tags.size(); ++e) {
      double P[6][2];
      for (int k = 0; k < 6; ++k) {
        const std::size_t at = index[nodes[6 * e + k]];
        P[k][0] = xyz[3 * at];
        P[k][1] = xyz[3 * at + 1];
      }
      double q[6];
      for (int p = 0; p < 6; ++p) {
        double J[2][2] = {{0, 0}, {0, 0}};
        for (int k = 0; k < 6; ++k)
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) J[a][b] += P[k][a] * dN[p][2 * k + b];
        q[p] = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      }
      Element el{s.second, region, 0, 0, 0, 0};
      mq::det_range_p2(q, el.lo, el.hi);
      // the reference centre: P2 shape functions -1/9 at the vertices, 4/9 at the mid-edge nodes
      el.x = (4 * (P[3][0] + P[4][0] + P[5][0]) - (P[0][0] + P[1][0] + P[2][0])) / 9;
      el.y = (4 * (P[3][1] + P[4][1] + P[5][1]) - (P[0][1] + P[1][1] + P[2][1])) / 9;
      if (el.lo > 0) ++signs[s.second][0];
      else if (el.hi < 0) ++signs[s.second][1];
      all.push_back(el);
    }
  }
  TangleScan scan;
  scan.checked = all.size();
  for (const Element& el : all) {
    const auto& s = signs[el.surface];
    if (s[1] > s[0] ? el.hi < 0 : el.lo > 0) continue;
    if (scan.tangled++ == 0) {
      scan.region = el.region > 0 ? region_name(el.region) : "surface " + std::to_string(-el.region);
      scan.at[0] = el.x;
      scan.at[1] = el.y;
    }
  }
  return scan;
}

#ifdef __linux__
// Thrown by a 2D mode whose mesh has a tangled element to repair, outside the curved-mesh process: main() makes the
// whole mesh again in that process (run_curved_process), where the repair gives the same mesh in every run.
struct RepairInCurvedProcess {};
#endif

// The input flag with which main() hands a 2D input to the curved-mesh process (curved_isolated); it changes nothing
// else.
const char* const kTangledRepair = "repair_tangled_in_curved_process";

// The record of the check of a refused 2D mesh, for its info file (main).
json g_refused_high_order;

// The check of a 2D mesh just raised to order 2 (the axisymmetric open mode, "plane" and "bounded"). No tangled
// element: returns null, and nothing else runs, so the mesh is the one made without the check. Otherwise gmsh's
// high-order optimisation runs (the elastic pass, then the patch optimizer if an element is still tangled; the nodes of
// curved boundaries stay on the CAD), unless AXMP_MESHER_HO=none, and the check runs again: returns the record
// {"checked", "tangled_before", "optimised", "tangled_after": 0}, or throws Fail with the count, the region and the
// remedy (the record in g_refused_high_order). Both gmsh passes order their work by memory address (mesh3d), so on
// Linux the repair runs only in the curved-mesh process (RepairInCurvedProcess), unless AXMP_MESHER_HO_ISOLATE=0.
json untangle_2d() {
  const std::string ho = high_order_mode();
  const std::string isolate = isolate_mode();
  const TangleScan before = scan_tangled_2d();
  if (before.tangled == 0) return json();
  const bool optimise = ho != "none";
  TangleScan after = before;
  if (optimise) {
    if (g_canonical != "on") {
#ifdef __linux__
      if (isolate != "0" && !curved_stage()) throw RepairInCurvedProcess();  // never handed on twice
      warn_not_reproducible(isolate == "0" ? "tangled elements repaired in this process: AXMP_MESHER_HO_ISOLATE=0"
                                           : "tangled elements repaired outside the canonical curved-mesh process");
#else
      warn_not_reproducible("tangled elements repaired without a curved-mesh process, which needs Linux");
#endif
    } else if (address_randomization() == "on") {
      warn_not_reproducible("address randomization on");
    }
    // gmsh reports a patch it cannot bring to its target quality as an error, which the API's error mode would throw
    // from inside the optimizer (mesh3d); what the passes achieve is checked below instead
    double abort_mode = 0;
    gmsh::option::getNumber("General.AbortOnError", abort_mode);
    gmsh::option::setNumber("General.AbortOnError", 0);
    gmsh::option::setNumber("Mesh.HighOrderOptimize", 2);
    gmsh::model::mesh::optimize("HighOrderElastic");
    after = scan_tangled_2d();
    if (after.tangled) {
      gmsh::model::mesh::optimize("HighOrder");
      after = scan_tangled_2d();
    }
    gmsh::option::setNumber("General.AbortOnError", abort_mode);
  }
  const json record = {{"checked", before.checked}, {"tangled_before", before.tangled}, {"optimised", optimise},
                       {"tangled_after", after.tangled}};
  if (after.tangled == 0) return record;
  g_refused_high_order = record;
  char where[96];
  std::snprintf(where, sizeof where, "(%.6g, %.6g)", after.at[0], after.at[1]);
  throw Fail(std::to_string(after.tangled) + " of " + std::to_string(after.checked) + " curved elements are tangled " +
             (optimise ? "after gmsh's high-order optimisation"
                       : "(gmsh's high-order optimisation is switched off: AXMP_MESHER_HO=none)") +
             ": their Jacobian determinant changes sign, vanishes, or is inverted against the other elements of their "
             "region; the first is in region '" + after.region + "' near " + where +
             ". The region is much thinner than the mesh size at geometry order 2: refine the mesh in or around region '" +
             after.region + "' (a smaller surface_size or mesh_size there), or use geometry order 1 (straight-sided "
             "elements).");
}

template <class T>
void put_vector(std::ofstream& f, const std::vector<T>& v) {
  const std::uint64_t n = v.size();
  f.write(reinterpret_cast<const char*>(&n), sizeof n);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

template <class T>
std::vector<T> get_vector(std::ifstream& f) {
  std::uint64_t n = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof n);
  std::vector<T> v(f ? n : 0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  return v;
}

// Curved-mesh handoff from the meshing child to the parent that runs gmsh's HighOrder patch optimizer (mesh3d): every
// entity's nodes (tags, coordinates, parametric coordinates) and elements, in gmsh's entity order. Returns false,
// without a usable file, if a node on a curve or surface has no parametric coordinates: re-created without them, it
// could no longer slide along the CAD in the optimizer.
bool save_curved_mesh(const std::string& path) {
  DimTags ents;
  gmsh::model::getEntities(ents);
  std::ofstream f(path, std::ios::binary);
  const std::uint64_t count = ents.size();
  f.write(reinterpret_cast<const char*>(&count), sizeof count);
  for (const auto& e : ents) {
    std::vector<std::size_t> nodes;
    std::vector<double> xyz, uvw;
    gmsh::model::mesh::getNodes(nodes, xyz, uvw, e.first, e.second, false, true);
    const std::size_t per_node = (e.first == 1 || e.first == 2) ? static_cast<std::size_t>(e.first) : 0;
    if (uvw.size() != per_node * nodes.size()) return false;
    std::vector<int> types;
    std::vector<std::vector<std::size_t>> elements, element_nodes;
    gmsh::model::mesh::getElements(types, elements, element_nodes, e.first, e.second);
    put_vector(f, std::vector<int>{e.first, e.second});
    put_vector(f, nodes);
    put_vector(f, xyz);
    put_vector(f, uvw);
    put_vector(f, types);
    for (std::size_t i = 0; i < types.size(); ++i) {
      put_vector(f, elements[i]);
      put_vector(f, element_nodes[i]);
    }
  }
  f.close();
  if (!f) throw Fail("cannot write the curved-mesh handoff file '" + path + "'; is the output directory writable "
                     "and not full?");
  return true;
}

// Re-creates the mesh written by save_curved_mesh on the (unmeshed) geometry of this process: all nodes first, as
// MEdgeVertex / MFaceVertex with their parametric coordinates on curves and surfaces, then the elements.
void load_curved_mesh(const std::string& path) {
  struct EntityMesh {
    std::vector<int> dim_tag;
    std::vector<std::size_t> nodes;
    std::vector<double> xyz;
    std::vector<double> uvw;
    std::vector<int> types;
    std::vector<std::vector<std::size_t>> elements;
    std::vector<std::vector<std::size_t>> element_nodes;
  };
  std::ifstream f(path, std::ios::binary);
  std::uint64_t count = 0;
  f.read(reinterpret_cast<char*>(&count), sizeof count);
  std::vector<EntityMesh> all(f ? count : 0);
  for (auto& m : all) {
    m.dim_tag = get_vector<int>(f);
    m.nodes = get_vector<std::size_t>(f);
    m.xyz = get_vector<double>(f);
    m.uvw = get_vector<double>(f);
    m.types = get_vector<int>(f);
    for (std::size_t i = 0; i < m.types.size(); ++i) {
      m.elements.push_back(get_vector<std::size_t>(f));
      m.element_nodes.push_back(get_vector<std::size_t>(f));
    }
    if (m.dim_tag.size() != 2) f.setstate(std::ios::failbit);
  }
  if (!f || all.empty())
    throw Fail("cannot read the curved-mesh handoff file '" + path + "' written by the meshing process");
  for (const auto& m : all) gmsh::model::mesh::addNodes(m.dim_tag[0], m.dim_tag[1], m.nodes, m.xyz, m.uvw);
  // The volume elements are allocated in a fixed pseudo-random order, then put back in their order in the volume
  // (reorderElements), so the patch optimizer sees them in a mixed address order, as in one process, yet the same in
  // every run. In file order (fresh memory in load order) gmsh's first pass failed on both designs where it can:
  // curved_probe_shell.json left 4 invalid tets and core30 (curved_elements_study.py) all 2418, where shuffled orders
  // (two seeds each) and 6 of 7 in-process runs cleared them in that pass.
  const std::uint64_t seed = 1;
  for (const auto& m : all)
    for (std::size_t i = 0; i < m.types.size(); ++i) {
      if (m.dim_tag[0] != 3 || m.elements[i].empty()) {
        gmsh::model::mesh::addElementsByType(m.dim_tag[1], m.types[i], m.elements[i], m.element_nodes[i]);
        continue;
      }
      const std::size_t n = m.elements[i].size();
      const std::size_t per = m.element_nodes[i].size() / n;
      std::vector<std::size_t> perm(n);
      for (std::size_t k = 0; k < n; ++k) perm[k] = k;
      std::uint64_t state = seed;
      for (std::size_t k = n; k > 1; --k) {  // Fisher-Yates with splitmix64 (the same on every platform)
        state += 0x9e3779b97f4a7c15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z = z ^ (z >> 31);
        std::swap(perm[k - 1], perm[z % k]);
      }
      std::vector<std::size_t> tags(n), nodes(n * per), ordering(n);
      for (std::size_t k = 0; k < n; ++k) {
        tags[k] = m.elements[i][perm[k]];
        std::copy(m.element_nodes[i].begin() + per * perm[k], m.element_nodes[i].begin() + per * (perm[k] + 1),
                  nodes.begin() + per * k);
        ordering[perm[k]] = k;
      }
      gmsh::model::mesh::addElementsByType(m.dim_tag[1], m.types[i], tags, nodes);
      gmsh::model::mesh::reorderElements(m.types[i], m.dim_tag[1], ordering);
    }
}

// Node coordinates and parametric coordinates of the whole mesh, entity by entity, to undo an optimizer pass.
struct EntityNodes {
  std::size_t per_node = 0;
  std::vector<std::size_t> tags;
  std::vector<double> xyz;
  std::vector<double> uvw;
};

std::vector<EntityNodes> save_node_positions() {
  DimTags ents;
  gmsh::model::getEntities(ents);
  std::vector<EntityNodes> all(ents.size());
  for (std::size_t k = 0; k < ents.size(); ++k) {
    const int dim = ents[k].first;
    all[k].per_node = (dim == 1 || dim == 2) ? static_cast<std::size_t>(dim) : 0;
    gmsh::model::mesh::getNodes(all[k].tags, all[k].xyz, all[k].uvw, dim, ents[k].second, false, true);
    if (all[k].uvw.size() == all[k].per_node * all[k].tags.size()) continue;
    // getNodes leaves out the nodes without parametric coordinates: take each node's own (setNode ignores the zeros
    // put in for the others)
    all[k].uvw.assign(all[k].per_node * all[k].tags.size(), 0.0);
    for (std::size_t i = 0; i < all[k].tags.size(); ++i) {
      std::vector<double> xyz, uvw;
      int node_dim = 0;
      int node_tag = 0;
      gmsh::model::mesh::getNode(all[k].tags[i], xyz, uvw, node_dim, node_tag);
      for (std::size_t j = 0; j < uvw.size() && j < all[k].per_node; ++j) all[k].uvw[all[k].per_node * i + j] = uvw[j];
    }
  }
  return all;
}

void restore_node_positions(const std::vector<EntityNodes>& all) {
  for (const auto& e : all)
    for (std::size_t i = 0; i < e.tags.size(); ++i) {
      const std::vector<double> xyz(e.xyz.begin() + 3 * i, e.xyz.begin() + 3 * i + 3);
      const std::vector<double> uvw(e.uvw.begin() + e.per_node * i, e.uvw.begin() + e.per_node * (i + 1));
      gmsh::model::mesh::setNode(e.tags[i], xyz, uvw);
    }
}

// The partition of a 3D geometry (--export-regions): the faces with the regions on their two sides, the volumes with
// their oriented shells, and one surface triangulation that every face and every volume shares, made to a chord
// tolerance. The infinity shell is not a volume of it.

double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The smallest distance found from p to the surface of face `tag` by Newton's method on the squared distance, from the
// parameters q; `best` is a distance already known (an upper bound of the true one). OpenCASCADE's projection of a point
// on a cone can land on the wrong point, which only gives a larger distance than the true one.
double surface_descent(int tag, const std::array<double, 3>& p, std::array<double, 2> q, double best) {
  std::vector<double> x, d1, d2;
  std::array<double, 3> r{};
  auto at = [&](const std::array<double, 2>& uv) {
    gmsh::model::getValue(2, tag, {uv[0], uv[1]}, x);
    if (x.size() < 3) return false;
    for (int k = 0; k < 3; ++k) r[k] = x[k] - p[k];
    return std::isfinite(r[0] + r[1] + r[2]);
  };
  if (!at(q)) return best;
  double f = std::sqrt(dot3(r.data(), r.data()));
  best = std::min(best, f);
  for (int it = 0; it < 30; ++it) {
    gmsh::model::getDerivative(2, tag, {q[0], q[1]}, d1);
    gmsh::model::getSecondDerivative(2, tag, {q[0], q[1]}, d2);
    if (d1.size() < 6 || d2.size() < 9) break;
    const double* su = &d1[0];
    const double* sv = &d1[3];
    const double g0 = dot3(r.data(), su), g1 = dot3(r.data(), sv);
    double h00 = dot3(su, su) + dot3(r.data(), &d2[0]);
    double h11 = dot3(sv, sv) + dot3(r.data(), &d2[3]);
    double h01 = dot3(su, sv) + dot3(r.data(), &d2[6]);
    double det = h00 * h11 - h01 * h01;
    if (!(h00 > 0 && det > 1e-12 * h00 * h11)) {  // not a minimum: the Gauss-Newton matrix, damped
      h00 = dot3(su, su);
      h11 = dot3(sv, sv);
      h01 = dot3(su, sv);
      const double damp = 1e-6 * (h00 + h11) + 1e-300;
      h00 += damp;
      h11 += damp;
      det = h00 * h11 - h01 * h01;
    }
    if (!(det > 0)) break;
    const double du = -(h11 * g0 - h01 * g1) / det, dv = -(h00 * g1 - h01 * g0) / det;
    bool moved = false;
    double t = 1;
    for (int k = 0; k < 12; ++k, t *= 0.5) {
      const std::array<double, 2> trial = {q[0] + t * du, q[1] + t * dv};
      if (!at(trial)) continue;
      const double fn = std::sqrt(dot3(r.data(), r.data()));
      if (fn <= f) {
        q = trial;
        const double step = t * t * (du * du * dot3(su, su) + dv * dv * dot3(sv, sv));
        f = fn;
        best = std::min(best, f);
        moved = step > 1e-24;
        break;
      }
    }
    if (!moved) break;
  }
  return best;
}

// The partition of the 3D geometry mesh3d has built (the fragmented solids, `vol_owner` the region of each volume, -1
// for the air; `excluded` the volumes of the infinity shell). The triangulation is the export's own: it depends on the
// geometry and on the chord tolerance only.
json export_partition_3d(const json& in, const json& regions, const std::map<int, int>& vol_owner,
                         const std::set<int>& excluded, double Rb, const json& center,
                         const std::chrono::steady_clock::time_point& t0) {
  const auto sq = [](double a) { return a * a; };
  using Tri = std::array<int, 3>;
  using Vec = std::array<double, 3>;
  // ----- volumes and faces
  DimTags all3, all2;
  gmsh::model::getEntities(all3, 3);
  gmsh::model::getEntities(all2, 2);
  std::vector<int> vtags;
  std::map<int, std::string> vregion;
  std::map<int, std::vector<int>> face_vols;  // face tag -> its volumes of the partition
  std::map<int, std::vector<int>> vol_faces;
  for (const auto& e : all3) {
    if (excluded.count(e.second)) continue;
    vtags.push_back(e.second);
    const auto it = vol_owner.find(e.second);
    const int ow = it == vol_owner.end() ? -1 : it->second;
    vregion[e.second] = ow < 0 ? std::string("air") : regions[ow].at("name").get<std::string>();
    DimTags bd;
    gmsh::model::getBoundary({{3, e.second}}, bd, false, false, false);
    for (const auto& s : bd) {
      face_vols[std::abs(s.second)].push_back(e.second);
      vol_faces[e.second].push_back(std::abs(s.second));
    }
  }
  std::sort(vtags.begin(), vtags.end());
  std::vector<int> ftags;
  for (const auto& kv : face_vols) ftags.push_back(kv.first);
  if (ftags.empty()) throw Fail("the geometry has no volume to export");
  // the boxes of OpenCASCADE, the diagonal of their union and the chord tolerance
  std::map<int, std::array<double, 6>> box;
  double ulo[3] = {1e300, 1e300, 1e300}, uhi[3] = {-1e300, -1e300, -1e300};
  for (int f : ftags) {
    auto& b = box[f];
    occ::getBoundingBox(2, f, b[0], b[1], b[2], b[3], b[4], b[5]);
    for (int k = 0; k < 3; ++k) {
      ulo[k] = std::min(ulo[k], b[k]);
      uhi[k] = std::max(uhi[k], b[3 + k]);
    }
  }
  const double diagonal = std::sqrt(sq(uhi[0] - ulo[0]) + sq(uhi[1] - ulo[1]) + sq(uhi[2] - ulo[2]));
  const double chord = in.contains("chord_tol_m") ? in.at("chord_tol_m").get<double>()
                                                   : std::min(1e-3, 1e-4 * diagonal);
  if (!(chord > 0) || !std::isfinite(chord)) throw Fail("chord_tol_m must be a positive number of metres");
  const double tolerance = 1e-9 * diagonal;
  std::map<int, std::string> ftype;
  for (int f : ftags) {
    std::string type;
    gmsh::model::getType(2, f, type);
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char ch) {
      return ch == ' ' ? '_' : static_cast<char>(std::tolower(ch));
    });
    ftype[f] = type;
  }

  // ----- sizes: the largest curvature of each face and curve gives the size at which a chord is within the tolerance
  const double target = 0.7 * chord;  // the centroid and edge midpoints within it keep the interior within the chord
  auto curvature_of = [&](int dim, int tag) {
    std::vector<double> lo, hi, par, kmax, kmin, dmax, dmin, kappa;
    double k = 0;
    try {
      gmsh::model::getParametrizationBounds(dim, tag, lo, hi);
      const int n = dim == 2 ? 7 : 9;
      for (int i = 0; i < n; ++i) {
        const double s = static_cast<double>(i) / (n - 1);
        if (dim == 1) {
          par.push_back(lo[0] + s * (hi[0] - lo[0]));
        } else {
          for (int j = 0; j < n; ++j) {
            par.push_back(lo[0] + s * (hi[0] - lo[0]));
            par.push_back(lo[1] + static_cast<double>(j) / (n - 1) * (hi[1] - lo[1]));
          }
        }
      }
      if (dim == 2) {
        gmsh::model::getPrincipalCurvatures(tag, par, kmax, kmin, dmax, dmin);
        for (size_t i = 0; i < kmax.size(); ++i) {
          if (std::isfinite(kmax[i])) k = std::max(k, std::fabs(kmax[i]));
          if (i < kmin.size() && std::isfinite(kmin[i])) k = std::max(k, std::fabs(kmin[i]));
        }
      } else {
        gmsh::model::getCurvature(1, tag, par, kappa);
        for (double v : kappa)
          if (std::isfinite(v)) k = std::max(k, std::fabs(v));
      }
    } catch (const std::exception&) {
      k = -1;  // not available
    }
    return k;
  };
  std::vector<int> curved;  // the faces that are not planes
  std::map<int, double> hface, hcurve;
  std::set<int> curve_set;
  for (int f : ftags) {
    DimTags bd;
    gmsh::model::getBoundary({{2, f}}, bd, false, false, false);
    for (const auto& c : bd) curve_set.insert(std::abs(c.second));
    if (ftype[f] == "plane") continue;
    curved.push_back(f);
    const double k = curvature_of(2, f);
    hface[f] = k > 0 ? std::min(std::sqrt(6.0 * target / k), diagonal) : diagonal / 50;
  }
  for (int c : curve_set) {
    const double k = curvature_of(1, c);
    if (k > 0) hcurve[c] = std::min(std::sqrt(8.0 * target / k), diagonal);
    else if (k < 0) hcurve[c] = diagonal / 50;
  }
  // the faces only the infinity shell bounds are meshed too (the mesher meshes the whole model): coarsely
  std::vector<int> coarse;
  for (const auto& e : all2)
    if (!face_vols.count(e.second)) coarse.push_back(e.second);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);
  gmsh::option::setNumber("Mesh.MeshSizeMin", 0);
  gmsh::option::setNumber("Mesh.MeshSizeMax", diagonal);
  std::vector<int> field_ids;
  auto set_fields = [&]() {
    for (int id : field_ids) gmsh::model::mesh::field::remove(id);
    field_ids.clear();
    std::vector<double> mins;
    char expr[64];
    auto add = [&](double h, const char* list, const std::vector<double>& tags) {
      const int fc = gmsh::model::mesh::field::add("MathEval");
      std::snprintf(expr, sizeof expr, "%.17g", h);
      gmsh::model::mesh::field::setString(fc, "F", expr);
      const int fr = gmsh::model::mesh::field::add("Restrict");
      gmsh::model::mesh::field::setNumber(fr, "InField", fc);
      gmsh::model::mesh::field::setNumbers(fr, list, tags);
      field_ids.push_back(fc);
      field_ids.push_back(fr);
      mins.push_back(fr);
    };
    for (const auto& kv : hface) add(kv.second, "SurfacesList", {static_cast<double>(kv.first)});
    for (const auto& kv : hcurve) add(kv.second, "CurvesList", {static_cast<double>(kv.first)});
    if (!coarse.empty()) {
      std::vector<double> tags(coarse.begin(), coarse.end());
      add(2 * Rb, "SurfacesList", tags);
    }
    if (mins.empty()) return;
    const int fm = gmsh::model::mesh::field::add("Min");
    gmsh::model::mesh::field::setNumbers(fm, "FieldsList", mins);
    gmsh::model::mesh::field::setAsBackgroundMesh(fm);
    field_ids.push_back(fm);
  };

  // ----- the mesh, refined face by face until every curved face meets the tolerance
  std::vector<size_t> ntags;
  std::vector<double> ncoord, npar;
  std::vector<int> node_index;  // gmsh node tag -> row of ncoord / 3
  std::map<int, std::vector<Tri>> tris;
  std::map<int, double> sagitta;
  const int max_passes = 12;
  for (int pass = 0;; ++pass) {
    set_fields();
    gmsh::model::mesh::clear();
    gmsh::model::mesh::generate(2);
    gmsh::model::mesh::getNodes(ntags, ncoord, npar, -1, -1, false, false);
    size_t top = 0;
    for (size_t t : ntags) top = std::max(top, t);
    node_index.assign(top + 1, -1);
    for (size_t i = 0; i < ntags.size(); ++i) node_index[ntags[i]] = static_cast<int>(i);
    tris.clear();
    for (int f : ftags) {
      std::vector<int> types;
      std::vector<std::vector<std::size_t>> etags, enodes;
      gmsh::model::mesh::getElements(types, etags, enodes, 2, f);
      auto& T = tris[f];
      for (size_t k = 0; k < types.size(); ++k) {
        if (types[k] != 2) throw Fail("the surface mesh of face " + std::to_string(f) + " has an element that is not a triangle");
        for (size_t i = 0; i + 2 < enodes[k].size(); i += 3)
          T.push_back({node_index.at(enodes[k][i]), node_index.at(enodes[k][i + 1]), node_index.at(enodes[k][i + 2])});
      }
    }
    // the distance of the centroids and edge midpoints from each curved face
    bool all_ok = true;
    for (int f : ftags) sagitta[f] = 0.0;
    for (int f : curved) {
      const auto& T = tris.at(f);
      std::map<size_t, std::array<double, 2>> uv;
      {
        std::vector<size_t> ft;
        std::vector<double> fc, fp;
        gmsh::model::mesh::getNodes(ft, fc, fp, 2, f, true, true);
        for (size_t i = 0; i < ft.size() && 2 * i + 1 < fp.size(); ++i) uv[ft[i]] = {fp[2 * i], fp[2 * i + 1]};
      }
      std::vector<double> pts;
      std::vector<std::array<int, 4>> owner_node;  // per point: the triangle's nodes (a, b, c) and its index
      for (size_t k = 0; k < T.size(); ++k) {
        const double* a = &ncoord[3 * T[k][0]];
        const double* b = &ncoord[3 * T[k][1]];
        const double* c = &ncoord[3 * T[k][2]];
        for (int m = 0; m < 4; ++m)
          for (int d = 0; d < 3; ++d)
            pts.push_back(m == 0 ? (a[d] + b[d] + c[d]) / 3.0
                                 : 0.5 * ((m == 1 ? a[d] : m == 2 ? b[d] : c[d]) + (m == 1 ? b[d] : m == 2 ? c[d] : a[d])));
        for (int m = 0; m < 4; ++m) owner_node.push_back({T[k][0], T[k][1], T[k][2], static_cast<int>(k)});
      }
      std::vector<double> closest, cpar;
      gmsh::model::getClosestPoint(2, f, pts, closest, cpar);
      const size_t np = pts.size() / 3;
      std::vector<double> d1(np);
      for (size_t i = 0; i < np; ++i)
        d1[i] = std::sqrt(sq(pts[3 * i] - closest[3 * i]) + sq(pts[3 * i + 1] - closest[3 * i + 1]) +
                          sq(pts[3 * i + 2] - closest[3 * i + 2]));
      std::vector<size_t> order(np);
      for (size_t i = 0; i < np; ++i) order[i] = i;
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return d1[a] > d1[b] || (d1[a] == d1[b] && a < b); });
      double best = 0;
      for (size_t i : order) {
        if (d1[i] <= best * (1 + 1e-7)) break;
        double ex = d1[i];
        const Vec p = {pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]};
        for (int v = 0; v < 3; ++v) {  // from the parameters of the nodes of its triangle
          const auto it = uv.find(ntags[owner_node[i][v]]);
          if (it != uv.end()) ex = std::min(ex, surface_descent(f, p, it->second, ex));
        }
        best = std::max(best, ex);
      }
      sagitta[f] = best;
      if (best > target) {
        all_ok = false;
        hface[f] *= std::max(0.5, 0.95 * std::sqrt(target / best));
      }
    }
    if (all_ok) break;
    if (pass + 1 >= max_passes) {
      std::string which;
      for (int f : curved)
        if (sagitta[f] > target) which += (which.empty() ? "" : ", ") + std::to_string(f);
      throw Fail("the surface triangulation does not meet chord_tol_m on face(s) " + which);
    }
  }

  // ----- one node list: the nodes the faces use, in the order of gmsh's node tags
  std::vector<int> used(ntags.size(), 0);
  for (const auto& kv : tris)
    for (const Tri& t : kv.second)
      for (int v : t) used[v] = 1;
  std::vector<std::pair<size_t, int>> by_tag;
  for (size_t i = 0; i < ntags.size(); ++i)
    if (used[i]) by_tag.push_back({ntags[i], static_cast<int>(i)});
  std::sort(by_tag.begin(), by_tag.end());
  std::vector<int> renum(ntags.size(), -1);
  std::vector<Vec> X;
  for (const auto& tg : by_tag) {
    renum[tg.second] = static_cast<int>(X.size());
    X.push_back({ncoord[3 * tg.second], ncoord[3 * tg.second + 1], ncoord[3 * tg.second + 2]});
  }
  for (auto& kv : tris)
    for (Tri& t : kv.second)
      for (int& v : t) v = renum[v];

  // ----- each face: its sides, a point and the normal there. The side a volume lies on is a fact of topology: a
  // volume's oriented boundary has the face with + where the volume is on the side the face's normal points out of.
  // The triangles of a face keep gmsh's orientation, which is the face's (the shells below are checked to close).
  struct FaceRecord {
    int vol_a = -1, vol_b = -1;  // the volumes the triangles' normal points out of and into (-1: the exterior)
    Vec point{}, normal{};
  };
  std::map<int, FaceRecord> rec;
  {
    std::map<int, std::pair<int, int>> plus_minus;
    for (const auto& e : all3) {
      DimTags bd;
      gmsh::model::getBoundary({{3, e.second}}, bd, false, true, false);
      for (const auto& sf : bd) {
        auto& pm = plus_minus.try_emplace(std::abs(sf.second), std::make_pair(-1, -1)).first->second;
        (sf.second > 0 ? pm.first : pm.second) = e.second;
      }
    }
    for (int f : ftags) {
      const auto& pm = plus_minus.at(f);
      rec[f].vol_a = (pm.first > 0 && !excluded.count(pm.first)) ? pm.first : -1;
      rec[f].vol_b = (pm.second > 0 && !excluded.count(pm.second)) ? pm.second : -1;
    }
  }
  struct Shell {
    std::vector<int> faces;
    double volume;
  };
  auto signed_volume = [&](const std::vector<Tri>& T) {
    double s6 = 0;
    for (const Tri& t : T) {
      const Vec &a = X[t[0]], &b = X[t[1]], &c = X[t[2]];
      s6 += a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    return s6 / 6.0;
  };
  // the shells of volume v with the faces' present orientation (closed when `check`)
  auto shells_of = [&](int v, bool check) {
    std::vector<int> fs = vol_faces.at(v);
    std::sort(fs.begin(), fs.end());
    fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
    // A shell is a closed manifold surface: faces belong to one shell when an edge of the shell pairs them, the
    // directed edge of one with the reverse edge of the other (where more than two faces meet at an edge, the pairing
    // prefers the faces whose other side is the same volume).
    std::map<int, int> parent;
    for (int f : fs) parent[f] = f;
    std::function<int(int)> root = [&](int x) { return parent.at(x) == x ? x : parent[x] = root(parent.at(x)); };
    std::map<std::pair<int, int>, std::pair<std::vector<int>, std::vector<int>>> edge_uses;  // (lo, hi) -> forward, backward
    for (int f : fs) {
      const bool plus = rec.at(f).vol_a == v;
      for (const Tri& t0 : tris.at(f)) {
        Tri t = t0;
        if (!plus) std::swap(t[1], t[2]);
        for (int e = 0; e < 3; ++e) {
          const int a = t[e], b = t[(e + 1) % 3];
          auto& u = edge_uses[{std::min(a, b), std::max(a, b)}];
          (a < b ? u.first : u.second).push_back(f);
        }
      }
    }
    auto other_side = [&](int f) { return rec.at(f).vol_a == v ? rec.at(f).vol_b : rec.at(f).vol_a; };
    for (auto& kv : edge_uses) {
      std::vector<int> fw = kv.second.first, bw = kv.second.second;
      std::sort(fw.begin(), fw.end());
      std::sort(bw.begin(), bw.end());
      std::vector<char> used(bw.size(), 0);
      for (int f : fw) {
        int pick = -1;
        for (size_t k = 0; k < bw.size() && pick < 0; ++k)
          if (!used[k] && other_side(bw[k]) == other_side(f)) pick = static_cast<int>(k);
        for (size_t k = 0; k < bw.size() && pick < 0; ++k)
          if (!used[k]) pick = static_cast<int>(k);
        if (pick < 0) break;
        used[pick] = 1;
        parent[root(f)] = root(bw[pick]);
      }
    }
    std::map<int, std::vector<int>> shells;  // root -> signed faces
    for (int f : fs) shells[root(f)].push_back(rec.at(f).vol_a == v ? f : -f);
    std::vector<Shell> list;
    for (auto& kv : shells) {
      std::vector<Tri> T;
      for (int s : kv.second) {
        for (Tri t : tris.at(std::abs(s))) {
          if (s < 0) std::swap(t[1], t[2]);
          T.push_back(t);
        }
      }
      // closed: every directed edge once, with its reverse
      std::map<std::pair<int, int>, int> dir;
      for (const Tri& t : T)
        for (int e = 0; e < 3; ++e) ++dir[{t[e], t[(e + 1) % 3]}];
      for (const auto& d : dir)
        if (check && d.second != 1)
          throw Fail("the surface of volume " + std::to_string(v) + " is not a manifold: an edge is used by more than "
                     "two of its triangles, so its surface touches itself along an edge");
        else if (check && !dir.count({d.first.second, d.first.first}))
          throw Fail("the surface of volume " + std::to_string(v) + " is not a closed manifold: an edge of its "
                     "triangles has no neighbour (its surface touches itself, or the triangulation does not close)");
      list.push_back({kv.second, signed_volume(T)});
    }
    return list;
  };

  // The triangles of a face keep gmsh's orientation, which is not the volumes' (a plane's normal does not follow the
  // face's orientation). Each face is turned or not so that every volume's boundary, its faces taken with the signs
  // of the oriented boundary, is a closed surface: across an edge that two faces of a volume share, the volume's
  // outward triangles run in opposite directions. That fixes the faces of a connected set up to a common turn, which
  // the volumes' shells decide (the shell that encloses the most is an outer shell: its volume is positive).
  std::map<int, std::vector<std::pair<int, int>>> parity_edges;  // face -> (face, parity) with o_f o_g = parity
  {
    std::map<int, std::map<int, int>> use_sign;  // volume -> face -> sign of the oriented boundary
    for (const auto& e : all3) {
      DimTags bd;
      gmsh::model::getBoundary({{3, e.second}}, bd, false, true, false);
      for (const auto& sf : bd) use_sign[e.second][std::abs(sf.second)] = sf.second > 0 ? 1 : -1;
    }
    for (int v : vtags) {
      std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> uses;  // edge -> (face, direction)
      for (const auto& fs : use_sign.at(v)) {
        if (!tris.count(fs.first)) continue;
        for (const Tri& t : tris.at(fs.first))
          for (int e = 0; e < 3; ++e) {
            const int a = t[e], b = t[(e + 1) % 3];
            uses[{std::min(a, b), std::max(a, b)}].push_back({fs.first, a < b ? 1 : -1});
          }
      }
      for (const auto& kv : uses) {
        if (kv.second.size() != 2 || kv.second[0].first == kv.second[1].first) continue;
        const auto &x = kv.second[0], &y = kv.second[1];
        const int parity = -use_sign.at(v).at(x.first) * use_sign.at(v).at(y.first) * x.second * y.second;
        parity_edges[x.first].push_back({y.first, parity});
        parity_edges[y.first].push_back({x.first, parity});
      }
    }
  }
  std::map<int, int> turned, component;  // face -> +1 / -1, face -> component
  {
    int ncomp = 0;
    for (int f0 : ftags) {
      if (turned.count(f0)) continue;
      turned[f0] = 1;
      component[f0] = ncomp;
      std::vector<int> stack = {f0};
      while (!stack.empty()) {
        const int f = stack.back();
        stack.pop_back();
        for (const auto& [g, parity] : parity_edges[f]) {
          const int want = turned.at(f) * parity;
          if (!turned.count(g)) {
            turned[g] = want;
            component[g] = ncomp;
            stack.push_back(g);
          } else if (turned.at(g) != want) {
            throw Fail("the orientation of the faces around face " + std::to_string(g) + " is inconsistent");
          }
        }
      }
      ++ncomp;
    }
    for (int f : ftags)
      if (turned.at(f) < 0)
        for (Tri& t : tris.at(f)) std::swap(t[1], t[2]);
    // What the orientation of a connected set decides is the sign of every shell in it. A volume with one shell has an
    // outer shell only: positive. For the others the volumes' own volumes decide: the shells of a volume enclose its
    // volume (the outer shell positive, a cavity negative), which the exact volume of OpenCASCADE states. The sets are
    // taken from the one that encloses the most, each turned if that brings the shells of its volumes closer to them.
    struct ShellRef {
      int volume, comp;
      double w;
    };
    std::vector<ShellRef> refs;
    std::map<int, double> vmass, biggest;
    for (int v : vtags) {
      double m = 0;
      occ::getMass(3, v, m);
      vmass[v] = m;
      for (const Shell& sh : shells_of(v, false)) {
        const int comp = component.at(std::abs(sh.faces.front()));
        refs.push_back({v, comp, sh.volume});
        biggest[comp] = std::max(biggest[comp], std::fabs(sh.volume));
      }
    }
    std::map<int, int> decision;
    {
      std::map<int, int> count;
      for (const auto& r : refs) ++count[r.volume];
      for (const auto& r : refs)
        if (count[r.volume] == 1) decision.emplace(r.comp, r.w > 0 ? 1 : -1);
    }
    for (const auto& kv : decision)
      if (kv.second < 0)
        for (auto& r : refs)
          if (r.comp == kv.first) r.w = -r.w;
    std::vector<int> pending;
    for (const auto& kv : biggest)
      if (!decision.count(kv.first)) pending.push_back(kv.first);
    std::sort(pending.begin(), pending.end(), [&](int a, int b) { return biggest[a] > biggest[b] || (biggest[a] == biggest[b] && a < b); });
    auto misfit = [&](int comp) {
      std::map<int, double> total;
      for (const auto& r : refs) total[r.volume] += r.w;
      double bad = 0;
      for (const auto& r : refs)
        if (r.comp == comp) bad += std::fabs(vmass.at(r.volume) - total[r.volume]);
      return bad;
    };
    for (int comp : pending) {
      const double keep = misfit(comp);
      for (auto& r : refs)
        if (r.comp == comp) r.w = -r.w;
      const double turn = misfit(comp);
      if (turn < keep) decision[comp] = -1;
      else {
        decision[comp] = 1;
        for (auto& r : refs)
          if (r.comp == comp) r.w = -r.w;
      }
    }
    for (int f : ftags)
      if (decision.count(component.at(f)) && decision.at(component.at(f)) < 0)
        for (Tri& t : tris.at(f)) std::swap(t[1], t[2]);
  }
  auto cross_dot = [&](const Vec& a, const Vec& b, const Vec& c, const Vec& n) {  // (b-a) x (c-a) . n
    const Vec u = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const Vec x = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
    return dot3(x.data(), n.data());
  };
  for (int f : ftags) {
    auto& T = tris.at(f);
    // a point of the face and the surface's normal there: a node inside the face, near the middle of its parameters,
    // else the surface point under the centroid of one of its largest triangles
    struct Candidate {
      std::array<double, 2> uv;
      int holder;  // a triangle that holds the point (-1: the one that has the node)
      int node;    // the node, as renumbered, or -1
    };
    std::vector<Candidate> cands;
    {
      std::vector<size_t> ft;
      std::vector<double> fc, fp;
      gmsh::model::mesh::getNodes(ft, fc, fp, 2, f, false, true);
      if (!ft.empty() && fp.size() >= 2 * ft.size()) {
        double um = 0, vm = 0;
        for (size_t i = 0; i < ft.size(); ++i) {
          um += fp[2 * i];
          vm += fp[2 * i + 1];
        }
        um /= ft.size();
        vm /= ft.size();
        std::vector<size_t> idx(ft.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
          const double da = sq(fp[2 * a] - um) + sq(fp[2 * a + 1] - vm), db = sq(fp[2 * b] - um) + sq(fp[2 * b + 1] - vm);
          return da < db || (da == db && ft[a] < ft[b]);
        });
        for (size_t i : idx) {
          if (cands.size() >= 50) break;
          if (ft[i] < node_index.size() && node_index[ft[i]] >= 0 && renum[node_index[ft[i]]] >= 0)
            cands.push_back({{fp[2 * i], fp[2 * i + 1]}, -1, renum[node_index[ft[i]]]});
        }
      }
      if (cands.empty()) {
        std::vector<double> area(T.size());
        std::vector<size_t> idx(T.size());
        for (size_t k = 0; k < T.size(); ++k) {
          const Vec &a = X[T[k][0]], &b = X[T[k][1]], &c = X[T[k][2]];
          const Vec u = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
          const Vec x = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
          area[k] = dot3(x.data(), x.data());
          idx[k] = k;
        }
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return area[a] > area[b] || (area[a] == area[b] && a < b); });
        for (size_t k : idx) {
          if (cands.size() >= 20) break;
          std::vector<double> cp, cq;
          const Vec &a = X[T[k][0]], &b = X[T[k][1]], &c = X[T[k][2]];
          gmsh::model::getClosestPoint(2, f, {(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3}, cp, cq);
          if (cq.size() < 2) continue;
          // the triangles' polygonal edge can cut into a hole of the face: the point must lie in the trimmed face
          bool in_face = true;
          try {
            in_face = gmsh::model::isInside(2, f, {cq[0], cq[1]}, true) > 0;
          } catch (const std::exception&) {
          }
          if (in_face) cands.push_back({{cq[0], cq[1]}, static_cast<int>(k), -1});
        }
      }
      if (cands.empty()) {  // a node of the face's boundary lies on the face
        std::vector<size_t> ft;
        std::vector<double> fc, fp;
        gmsh::model::mesh::getNodes(ft, fc, fp, 2, f, true, true);
        for (size_t i = 0; i < ft.size() && cands.size() < 20 && fp.size() >= 2 * ft.size(); ++i)
          if (ft[i] < node_index.size() && node_index[ft[i]] >= 0 && renum[node_index[ft[i]]] >= 0)
            cands.push_back({{fp[2 * i], fp[2 * i + 1]}, -1, renum[node_index[ft[i]]]});
      }
    }
    bool placed = false;
    for (const Candidate& cd : cands) {
      if (placed) break;
      std::vector<double> val, nrm_v;
      gmsh::model::getValue(2, f, {cd.uv[0], cd.uv[1]}, val);
      gmsh::model::getNormal(f, {cd.uv[0], cd.uv[1]}, nrm_v);
      if (val.size() < 3 || nrm_v.size() < 3) continue;
      const double nl = std::sqrt(dot3(nrm_v.data(), nrm_v.data()));
      if (!(nl > 0) || !std::isfinite(nl + val[0] + val[1] + val[2])) continue;
      Vec n = {nrm_v[0] / nl, nrm_v[1] / nl, nrm_v[2] / nl};
      int holder = cd.holder;
      for (size_t k = 0; k < T.size() && holder < 0; ++k)
        if (T[k][0] == cd.node || T[k][1] == cd.node || T[k][2] == cd.node) holder = static_cast<int>(k);
      if (holder < 0) continue;
      const double along = cross_dot(X[T[holder][0]], X[T[holder][1]], X[T[holder][2]], n);
      if (along == 0) continue;
      if (along < 0) n = {-n[0], -n[1], -n[2]};  // turned like the triangles
      rec[f].point = {val[0], val[1], val[2]};
      rec[f].normal = n;
      placed = true;
    }
    if (!placed) throw Fail("cannot find a point of face " + std::to_string(f));
  }

  // ----- the volumes: shells of signed faces, outer shell first
  json jvolumes = json::array();
  std::map<std::string, double> region_volume;
  for (int v : vtags) {
    std::vector<Shell> list = shells_of(v, true);
    std::stable_sort(list.begin(), list.end(), [](const Shell& a, const Shell& b) { return a.volume > b.volume; });
    if (list.empty() || !(list[0].volume > 0) || (list.size() > 1 && list[1].volume >= 0))
    {
      std::string shells_text;
      for (const auto& sh : list)
        shells_text += " [" + std::to_string(sh.faces.size()) + " faces, volume " + std::to_string(sh.volume) + "]";
      throw Fail("volume " + std::to_string(v) + " does not have exactly one outer shell (its shells:" + shells_text + ")");
    }
    std::stable_sort(list.begin() + 1, list.end(), [](const Shell& a, const Shell& b) {
      int ma = 1 << 30, mb = 1 << 30;
      for (int s : a.faces) ma = std::min(ma, std::abs(s));
      for (int s : b.faces) mb = std::min(mb, std::abs(s));
      return ma < mb;
    });
    json jshells = json::array();
    for (auto& sh : list) {
      std::sort(sh.faces.begin(), sh.faces.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
      jshells.push_back(sh.faces);
    }
    double vol = 0;
    occ::getMass(3, v, vol);
    region_volume[vregion.at(v)] += vol;
    jvolumes.push_back({{"tag", v}, {"region", vregion.at(v)}, {"shells", jshells}, {"volume_m3", vol}});
  }

  // ----- the faces
  json jentities = json::array(), jfaces = json::array();
  for (int f : ftags) {
    const auto& T = tris.at(f);
    const FaceRecord& r = rec.at(f);
    // the face's box: OpenCASCADE's, cut to the nodes' box (plus 4 chord tolerances) where it is loose (a torus: its
    // box is the surface's, much larger than the face's), except at the extremes of the union of the boxes
    std::array<double, 6> b = box.at(f);
    if (ftype[f] != "plane" && ftype[f] != "cylinder" && ftype[f] != "cone" && ftype[f] != "sphere") {
      double nlo[3] = {1e300, 1e300, 1e300}, nhi[3] = {-1e300, -1e300, -1e300};
      for (const Tri& t : T)
        for (int v : t)
          for (int k = 0; k < 3; ++k) {
            nlo[k] = std::min(nlo[k], X[v][k]);
            nhi[k] = std::max(nhi[k], X[v][k]);
          }
      for (int k = 0; k < 3; ++k) {
        if (b[k] != ulo[k]) b[k] = std::max(b[k], nlo[k] - 4 * chord);
        if (b[3 + k] != uhi[k]) b[3 + k] = std::min(b[3 + k], nhi[k] + 4 * chord);
      }
    }
    double area = 0;
    occ::getMass(2, f, area);
    auto name = [&](int v) { return v < 0 ? std::string("exterior") : vregion.at(v); };
    jentities.push_back({{"tag", f},
                         {"dim", 2},
                         {"side", {name(r.vol_a), name(r.vol_b)}},
                         {"curves", json::array()},
                         {"axis", false},
                         {"type", ftype.at(f)},
                         {"lo", {b[0], b[1], b[2]}},
                         {"hi", {b[3], b[4], b[5]}},
                         {"point", {r.point[0], r.point[1], r.point[2]}},
                         {"normal", {r.normal[0], r.normal[1], r.normal[2]}},
                         {"area_m2", area}});
    json jt = json::array();
    for (const Tri& t : T) jt.push_back({t[0], t[1], t[2]});
    jfaces.push_back({{"tag", f}, {"triangles", jt}, {"sagitta_m", ftype.at(f) == "plane" ? 0.0 : sagitta.at(f)}});
  }
  json jnodes = json::array();
  for (const Vec& x : X) jnodes.push_back({x[0], x[1], x[2]});
  json jregions = json::object();
  jregions["air"] = {{"attr", in.value("air_attr", 1)}, {"volume_m3", region_volume["air"]}};
  for (const auto& rg : regions) jregions[rg.at("name").get<std::string>()] = {
      {"attr", rg.at("attr").get<int>()}, {"volume_m3", region_volume[rg.at("name").get<std::string>()]}};
  std::string version;
  gmsh::option::getString("General.Version", version);
  return {{"ok", true},
          {"export_regions", true},
          {"geometry", "3d"},
          {"gmsh_version", version},
          {"domain_radius", Rb},
          {"domain_center", center},
          {"diagonal_m", diagonal},
          {"tolerance_m", tolerance},
          {"chord_tol_m", chord},
          {"regions", jregions},
          {"entities", jentities},
          {"volumes", jvolumes},
          {"triangulation", {{"nodes", jnodes}, {"faces", jfaces}}},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
}

int mesh3d(json in, json& info) {
  auto t0 = std::chrono::steady_clock::now();
  const json c = in.value("domain_center", json::array({0, 0, 0}));
  const int outer_attr = in.value("outer_attr", 901), air_attr = in.value("air_attr", 1);
  DimTags objects;
  std::vector<int> owner;
  const json& regions = in.value("regions", json::array());
  for (size_t i = 0; i < regions.size(); ++i) {
    DimTags s = build_solid(regions[i].at("solid"));
    for (auto& e : s) {
      objects.push_back(e);
      owner.push_back(static_cast<int>(i));
    }
  }
  // automatic extents: device bounding box -> domain radius and element sizes
  double bx0 = 1e300, by0 = 1e300, bz0 = 1e300, bx1 = -1e300, by1 = -1e300, bz1 = -1e300;
  for (auto& e : objects) {
    double a0, a1, a2, a3, a4, a5;
    occ::getBoundingBox(e.first, e.second, a0, a1, a2, a3, a4, a5);
    bx0 = std::min(bx0, a0); by0 = std::min(by0, a1); bz0 = std::min(bz0, a2);
    bx1 = std::max(bx1, a3); by1 = std::max(by1, a4); bz1 = std::max(bz1, a5);
  }
  double Rdev = 0;
  for (double xx : {bx0, bx1})
    for (double yy : {by0, by1})
      for (double zz : {bz0, bz1})
        Rdev = std::max(Rdev, std::sqrt((xx - c[0].get<double>()) * (xx - c[0].get<double>()) +
                                        (yy - c[1].get<double>()) * (yy - c[1].get<double>()) +
                                        (zz - c[2].get<double>()) * (zz - c[2].get<double>())));
  const double Ldev = std::max({bx1 - bx0, by1 - by0, bz1 - bz0});
  // with the infinity shell the open boundary is exact, so the air region can be small
  if (in.value("domain_radius", 0.0) <= 0) in["domain_radius"] = (in.value("infinity_shell", 0.0) > 1.0 ? 1.6 : 4.0) * Rdev;
  if (in.value("size", 0.0) <= 0) in["size"] = Ldev / 20.0;
  if (in.value("surface_size", 0.0) <= 0) in["surface_size"] = in["size"].get<double>() / 3.0;
  if (in.value("far_size", 0.0) <= 0) in["far_size"] = in["domain_radius"].get<double>() / 6.0;
  in["device_radius"] = Rdev;
  const double Rb = in.at("domain_radius");
  // optional "infinity shell" [Rb, Rs]: the solver maps it onto [Rb, infinity) (exact open boundary)
  const json symmetry=in.value("symmetry",json::array());
  const bool custom_domain=in.contains("domain_solid");
  const double shell_factor = (custom_domain || !symmetry.empty()) ? 0 : in.value("infinity_shell", 0.0);
  const double Rs = shell_factor > 1.0 ? shell_factor * Rb : Rb;
  const int shell_attr = in.value("shell_attr", 898);
  int shell_tag = -1;
  if (Rs > Rb) {
    int outer_ball = occ::addSphere(c[0], c[1], c[2], Rs);
    int inner_ball = occ::addSphere(c[0], c[1], c[2], Rb);
    DimTags cut;
    std::vector<DimTags> cmap;
    occ::cut({{3, outer_ball}}, {{3, inner_ball}}, cut, cmap);
    if (cut.size() != 1) throw Fail("could not build the infinity shell");
    shell_tag = cut[0].second;
  }
  DimTags domain=custom_domain ? build_solid(in.at("domain_solid")) :
                                DimTags{{3,occ::addSphere(c[0],c[1],c[2],Rb)}};
  auto clip = [&](const DimTags& input) {
    if (symmetry.empty() || input.empty()) return input;
    double lo[3]={-2*Rs,-2*Rs,-2*Rs},hi[3]={2*Rs,2*Rs,2*Rs};
    for (const auto& p:symmetry) {
      const std::string axis=p.at("axis"); const int a=axis=="x"?0:(axis=="y"?1:2);
      if (p.value("side",1.0)>0) lo[a]=0; else hi[a]=0;
    }
    int box=occ::addBox(lo[0],lo[1],lo[2],hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);
    DimTags out; std::vector<DimTags> map;
    occ::intersect(input,{{3,box}},out,map);
    return out;
  };
  std::set<int> active_regions;
  if (!symmetry.empty()) {
    DimTags kept; std::vector<int> kept_owner;
    for (size_t i=0;i<objects.size();++i) for (const auto& e:clip({objects[i]})) {
      if (e.first==3) {kept.push_back(e);kept_owner.push_back(owner[i]);active_regions.insert(owner[i]);}
    }
    objects=kept;owner=kept_owner;
    domain=clip(domain);
  } else for (int i:owner) active_regions.insert(i);
  for (const auto& e:domain) {objects.insert(objects.begin(),e);owner.insert(owner.begin(),-1);}
  if (domain.empty()) throw Fail("computational domain is empty after symmetry clipping");
  if (shell_tag >= 0) {
    objects.push_back({3, shell_tag});
    owner.push_back(-2);  // the shell
  }
  DimTags frag;
  std::vector<DimTags> fmap;
  occ::fragment(objects, {}, frag, fmap);
  occ::synchronize();
  std::map<int, int> vol_owner;
  std::set<int> in_domain;
  std::set<int> shell_vols;
  for (size_t k = 0; k < fmap.size(); ++k) {
    for (auto& e : fmap[k]) {
      if (e.first != 3) continue;
      if (owner[k] == -2) {
        shell_vols.insert(e.second);
        continue;
      }
      if (owner[k] < 0) in_domain.insert(e.second);
      auto it = vol_owner.find(e.second);
      if (owner[k] >= 0 && (it == vol_owner.end() || it->second < owner[k])) vol_owner[e.second] = owner[k];
      else if (owner[k] < 0 && it == vol_owner.end()) vol_owner[e.second] = -1;
    }
  }
  DimTags all3;
  gmsh::model::getEntities(all3, 3);
  for (auto& e : all3) {
    if (shell_vols.count(e.second) && !in_domain.count(e.second)) continue;
    if (!in_domain.count(e.second)) throw Fail("a region extends outside the computational domain; increase mesh.domain_radius");
    if (!vol_owner.count(e.second)) vol_owner[e.second] = -1;
  }
  // COMSOL export (docs/comsol_export.md): the fragmented (conformal) OCC solids as STEP, before any PEC
  // volume is removed, with the region that owns each solid; "geometry_only" stops before meshing.
  if (in.contains("step_output")) {
    const std::string step = in.at("step_output").get<std::string>();
    gmsh::write(step);
    json solids = json::array();
    for (auto& e : all3) {
      const bool shell = shell_vols.count(e.second) && !in_domain.count(e.second);
      const int ow = vol_owner.count(e.second) ? vol_owner.at(e.second) : -1;
      double lo[3], hi[3], cm[3];
      occ::getBoundingBox(3, e.second, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
      occ::getCenterOfMass(3, e.second, cm[0], cm[1], cm[2]);
      double vol = 0;
      occ::getMass(3, e.second, vol);
      solids.push_back({{"volume_tag", e.second},
                        {"region", shell ? std::string("infinity_shell") : (ow < 0 ? std::string("air") : regions[ow].at("name").get<std::string>())},
                        {"attr", shell ? shell_attr : (ow < 0 ? air_attr : regions[ow].at("attr").get<int>())},
                        {"volume_m3", vol}, {"center_of_mass", {cm[0], cm[1], cm[2]}},
                        {"bbox", {lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]}}});
    }
    info["step"] = {{"file", step}, {"solids", solids}};
    if (in.value("geometry_only", false)) {
      info["ok"] = true;
      info["geometry_only"] = true;
      info["domain_radius"] = Rb;
      info["shell_outer_radius"] = Rs;
      info["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return 0;
    }
  }
  std::map<int, std::vector<int>> by_attr;
  std::map<int, std::string> attr_name;
  std::set<int> removed_volumes;
  for (auto& [tag, ow] : vol_owner) {
    if (shell_vols.count(tag) && !in_domain.count(tag)) continue;
    int attr = ow < 0 ? air_attr : regions[ow].at("attr").get<int>();
    by_attr[attr].push_back(tag);
    attr_name[attr] = ow < 0 ? "air" : regions[ow].at("name").get<std::string>();
    if (ow>=0 && regions[ow].value("remove_pec",false)) removed_volumes.insert(tag);
  }
  std::vector<int> shell_list;
  for (int t : shell_vols) {
    if (in_domain.count(t)) continue;
    auto it = vol_owner.find(t);
    if (it != vol_owner.end() && it->second >= 0)
      throw Fail("region '" + regions[it->second].at("name").get<std::string>() +
                 "' extends into the infinity shell; increase mesh.domain_radius");
    shell_list.push_back(t);
  }
  if (!shell_list.empty()) {
    by_attr[shell_attr] = shell_list;
    attr_name[shell_attr] = "infinity_shell";
  }
  // Preserve connected PEC-body identity before deleting its volume. This is
  // required for terminals on holes, including distinct disconnected bodies
  // belonging to the same CAD region. Face contacts merge conductor bodies.
  std::map<int,std::vector<int>> surface_volumes, region_surfaces;
  std::map<int,int> parent;
  for (const auto& e:all3) {
    DimTags bd; gmsh::model::getBoundary({e},bd,false,false,false);
    for (const auto& s:bd) surface_volumes[std::abs(s.second)].push_back(e.second);
    if (removed_volumes.count(e.second)) parent[e.second]=e.second;
  }
  auto root=[&](int x) {while(parent.at(x)!=x) x=parent.at(x);return x;};
  for (const auto& [s,vs]:surface_volumes) {
    int first=-1;
    for (int v:vs) if (removed_volumes.count(v)) {
      if (first<0) first=v; else parent[root(v)]=root(first);
    }
  }
  std::map<int,int> body_ids;
  for (int v:removed_volumes) if (!body_ids.count(root(v))) body_ids[root(v)]=static_cast<int>(body_ids.size());
  for (const auto& [attr,vs]:by_attr) {
    std::set<int> surfaces;
    for (int v:vs) {DimTags bd;gmsh::model::getBoundary({{3,v}},bd,false,false,false);
      for (const auto& s:bd) surfaces.insert(std::abs(s.second));}
    region_surfaces[attr]=std::vector<int>(surfaces.begin(),surfaces.end());
  }
  json groups = json::array(), boundary_groups=json::array(), conductors=json::array();
  for (auto& [attr, tags] : by_attr) {
    std::vector<int> kept;
    for (int t:tags) if (!removed_volumes.count(t)) kept.push_back(t);
    if (!kept.empty()) {
      add_physical_group(3, kept, attr, attr_name[attr]);
      groups.push_back({{"dim", 3}, {"attr", attr}, {"name", attr_name[attr]}, {"entities", kept.size()}});
    }
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    if (active_regions.count(static_cast<int>(i)) && !by_attr.count(regions[i].at("attr").get<int>()))
      throw Fail("region '" + regions[i].at("name").get<std::string>() + "' is completely covered by later regions");
  }
  std::map<int,std::vector<int>> boundary_tags;
  std::map<int,json> boundary_meta;
  boundary_meta[outer_attr]={{"attr",outer_attr},{"name","outer"},{"type","pec"}};
  for (size_t i=0;i<symmetry.size();++i) {
    const int attr=910+static_cast<int>(i);
    boundary_meta[attr]={{"attr",attr},{"name",symmetry[i]["name"]},{"type",symmetry[i]["type"]},
                         {"axis",symmetry[i]["axis"]},{"side",symmetry[i].value("side",1.0)}};
  }
  for (const auto& [s,vs]:surface_volumes) {
    int alive=0,dead=-1;
    for (int v:vs) {if (removed_volumes.count(v)) dead=v;else ++alive;}
    if (alive!=1) continue;
    int attr=outer_attr;
    if (dead>=0) {
      const int bid=body_ids.at(root(dead));attr=1000+bid;
      const int ow=vol_owner.at(dead);
      boundary_meta[attr]={{"attr",attr},{"name","pec_body_"+std::to_string(bid)},
                           {"type","pec"},{"body_id",bid},{"conductor",regions[ow]["name"]}};
    } else {
      double lo[3],hi[3];occ::getBoundingBox(2,s,lo[0],lo[1],lo[2],hi[0],hi[1],hi[2]);
      for (size_t i=0;i<symmetry.size();++i) {
        const std::string ax=symmetry[i]["axis"];const int a=ax=="x"?0:(ax=="y"?1:2);
        if (std::max(std::fabs(lo[a]),std::fabs(hi[a]))<std::max(2e-6,Rs*1e-8)) attr=910+static_cast<int>(i);
      }
    }
    boundary_tags[attr].push_back(s);
  }
  if (g_export_regions) {
    std::set<int> shell_only;
    for (int t : shell_vols)
      if (!in_domain.count(t)) shell_only.insert(t);
    info = export_partition_3d(in, regions, vol_owner, shell_only, Rb, c, t0);
    return 0;
  }
  if (!removed_volumes.empty()) {
    DimTags remove;for(int v:removed_volumes) remove.push_back({3,v});
    occ::remove(remove,true);occ::synchronize();
  }
  for (const auto& [attr,tags]:boundary_tags) {
    const json& meta=boundary_meta.at(attr);
    add_physical_group(2,tags,attr,meta["name"].get<std::string>());
    boundary_groups.push_back(meta);
    groups.push_back({{"dim",2},{"attr",attr},{"name",meta["name"]},{"entities",tags.size()}});
  }
  for (const auto& [r,bid]:body_ids) {
    std::set<std::string> names;int count=0;
    for (int v:removed_volumes) if(root(v)==r) {++count;names.insert(regions[vol_owner.at(v)]["name"].get<std::string>());}
    conductors.push_back({{"body_id",bid},{"regions",names},{"removed_volumes",count}});
  }

  // sizes
  // max_size: an upper bound on every element size (full wave: a fraction of the shortest wavelength)
  const double h_cap = in.value("max_size", 0.0) > 0 ? in.value("max_size", 0.0) : 1e300;
  const double h_dev = std::min(in.at("size").get<double>(), h_cap), h_s = std::min(in.value("surface_size", h_dev), h_cap),
               h_far = std::min(in.value("far_size", Rb / 6), h_cap);
  const double r_dev = in.value("device_radius", Rb / 3);
  // extend_from_boundary = 1 grades the volume from the boundary sizes (e.g. curvature-refined curved walls; without
  // it a finely meshed convex wall meets a coarse interior in one layer of flat elements, e.g. at an SRF iris)
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", static_cast<int>(in.value("extend_from_boundary", 0.0)));
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", in.value("curvature_points", 16));
  gmsh::option::setNumber("Mesh.MeshSizeMax", h_far);
  double mesh_size_min = 0, min_curve_radius = 0;
  {  // the floor must not clip the smallest requested size (per-region sizes included)
    double hmin = std::min(h_s, h_dev);
    for (auto& g : regions)
      for (const char* key : {"size", "surface_size"}) {
        const double v = g.value(key, 0.0);
        if (v > 0) hmin = std::min(hmin, v);
      }
    mesh_size_min = hmin * 0.25;
    const double kmax = max_curve_curvature();
    min_curve_radius = kmax > 0 ? 1 / kmax : 0;
    // experimental option: nor may it clip curvature sizing on the smallest curve radius
    if (env_flag("AXMP_MESHER_CURVATURE_FLOOR") && kmax > 0)
      mesh_size_min = std::min(mesh_size_min, 2 * M_PI / (kmax * in.value("curvature_points", 16.0)));
    gmsh::option::setNumber("Mesh.MeshSizeMin", mesh_size_min);
  }
  std::vector<double> fields;
  int ff = gmsh::model::mesh::field::add("MathEval");
  char expr[320];
  std::snprintf(expr, sizeof expr, "max(%.17g, %.17g + 0.3*(sqrt((x-%.17g)^2+(y-%.17g)^2+(z-%.17g)^2) - %.17g))", h_dev,
                h_dev, c[0].get<double>(), c[1].get<double>(), c[2].get<double>(), r_dev);
  gmsh::model::mesh::field::setString(ff, "F", expr);
  fields.push_back(ff);
  if (!shell_list.empty()) {  // element layers across the shell (the mapped exterior field is smooth there)
    std::vector<double> tags(shell_list.begin(), shell_list.end());
    int fc = gmsh::model::mesh::field::add("MathEval");
    const double layers = std::max(1.0, in.value("shell_layers", 2.0));
    std::snprintf(expr, sizeof expr, "%.17g", std::min(h_far, (Rs - Rb) / layers));
    gmsh::model::mesh::field::setString(fc, "F", expr);
    int fr = gmsh::model::mesh::field::add("Restrict");
    gmsh::model::mesh::field::setNumber(fr, "InField", fc);
    gmsh::model::mesh::field::setNumbers(fr, "VolumesList", tags);
    fields.push_back(fr);
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    double h = regions[i].value("size", 0.0);
    if (h <= 0) continue;
    std::vector<double> tags;
    for (int t : by_attr[regions[i].at("attr").get<int>()]) if(!removed_volumes.count(t)) tags.push_back(t);
    if (tags.empty()) continue;
    int fc = gmsh::model::mesh::field::add("MathEval");
    std::snprintf(expr, sizeof expr, "%.17g", h);
    gmsh::model::mesh::field::setString(fc, "F", expr);
    int fr = gmsh::model::mesh::field::add("Restrict");
    gmsh::model::mesh::field::setNumber(fr, "InField", fc);
    gmsh::model::mesh::field::setNumbers(fr, "VolumesList", tags);
    fields.push_back(fr);
  }
  // Distance + Threshold refinement towards a set of surfaces with a given target size.  SizeMax
  // must not undercut the device size: a Threshold field returns SizeMax at every point beyond
  // DistMax and this one is not Restricted to the region.
  const double layer3 = in.value("surface_layer", 4.0 * h_dev);
  auto refine_towards3 = [&](const std::vector<double>& surfs, double hmin) {
    if (surfs.empty() || hmin <= 0) return;
    int fd = gmsh::model::mesh::field::add("Distance");
    gmsh::model::mesh::field::setNumbers(fd, "SurfacesList", surfs);
    int ft = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(ft, "InField", fd);
    gmsh::model::mesh::field::setNumber(ft, "SizeMin", hmin);
    gmsh::model::mesh::field::setNumber(ft, "SizeMax", std::max(h_dev, hmin));
    gmsh::model::mesh::field::setNumber(ft, "DistMin", 0.5 * hmin);
    gmsh::model::mesh::field::setNumber(ft, "DistMax", std::max(layer3, 4.0 * hmin));
    fields.push_back(ft);
  };
  auto boundary_surfaces = [&](const std::vector<int>& vols) {
    std::vector<double> out;
    std::set<int> valid;DimTags faces;gmsh::model::getEntities(faces,2);
    for(const auto& f:faces) valid.insert(f.second);
    for(const auto& [s,vs]:surface_volumes) if(valid.count(s))
      for(int v:vs) if(std::find(vols.begin(),vols.end(),v)!=vols.end()) {out.push_back(s);break;}
    return out;
  };
  // per-region surface_size: its own boundary refined to its own size
  std::set<std::string> own_surface;
  for (size_t i = 0; i < regions.size(); ++i) {
    const double hr = regions[i].value("surface_size", 0.0);
    if (hr <= 0) continue;
    own_surface.insert(regions[i].at("name").get<std::string>());
    refine_towards3(boundary_surfaces(by_attr[regions[i].at("attr").get<int>()]), hr);
  }
  const std::vector<std::string> refine_list = in.value("refine_regions", std::vector<std::string>{});
  std::set<std::string> refine(refine_list.begin(), refine_list.end());
  std::vector<double> rsurf;
  for (auto& [attr, tags] : by_attr) {
    if (!refine.count(attr_name[attr]) || own_surface.count(attr_name[attr])) continue;
    for (double t : boundary_surfaces(tags)) rsurf.push_back(t);
  }
  refine_towards3(rsurf, h_s);
  {
    // adaptive cycles: the floor of cycle 0 (per-region sizes included), lowered to the smallest adapted size
    double hmin = mesh_size_min;
    const int fa = size_field_from_file(in, &hmin);
    if (fa >= 0) {
      fields.push_back(fa);
      gmsh::option::setNumber("Mesh.MeshSizeMin", hmin);
    }
  }
  int fmin = gmsh::model::mesh::field::add("Min");
  gmsh::model::mesh::field::setNumbers(fmin, "FieldsList", fields);
  gmsh::model::mesh::field::setAsBackgroundMesh(fmin);
  gmsh::option::setNumber("Mesh.Algorithm", 6);
  gmsh::option::setNumber("Mesh.Algorithm3D", in.value("algorithm3d", 10));
  const int threads = in.value("threads", 4);  // General.NumThreads, set where meshing starts (see below)
  gmsh::option::setNumber("Mesh.Optimize", 1);
  if (in.value("optimize_netgen", false)) gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);
  const char* dump = std::getenv("AXMP_MESHER_SURFACE_DUMP");
  if (dump && *dump) {  // debugging aid: surface tags -> bbox
    // Written through nlohmann::json so that region names are escaped and bounding boxes keep full precision.
    auto write_dump = [&]() {
      json list = json::array();
      DimTags faces;
      gmsh::model::getEntities(faces, 2);
      for (const auto& fc : faces) {
        double a0, a1, a2, a3, a4, a5;
        gmsh::model::getBoundingBox(2, fc.second, a0, a1, a2, a3, a4, a5);
        std::vector<int> up, down;
        gmsh::model::getAdjacencies(2, fc.second, up, down);
        std::string names;
        for (int v : up) {
          auto it = vol_owner.find(v);
          names += (it == vol_owner.end() ? std::string("?") : it->second < 0 ? std::string("air")
                    : regions[it->second].at("name").get<std::string>()) + ";";
        }
        list.push_back({{"tag", fc.second}, {"bbox", {a0, a1, a2, a3, a4, a5}}, {"volumes", names}});
      }
      std::ofstream f(dump);
      if (f) f << list.dump();
      if (!f) std::cerr << "axmp-mesher: warning: AXMP_MESHER_SURFACE_DUMP: cannot write '" << dump << "'; no surface dump\n";
    };
    // In a forked child where that is safe (one thread): the dump's queries allocate in this process, and done here
    // they changed the curved mesh of the same input.
    pid_t pid = -1;
    if (g_canonical == "on" || thread_count() <= 1) {
      std::cout.flush();
      std::fflush(nullptr);
      pid = fork();
    }
    if (pid == 0) {
      try {
        write_dump();
      } catch (const std::exception& e) {
        std::cerr << "axmp-mesher: warning: AXMP_MESHER_SURFACE_DUMP: " << e.what() << "; no surface dump\n";
      }
      std::_Exit(0);
    }
    if (pid > 0) {
      int wstatus = 0;
      while (waitpid(pid, &wstatus, 0) < 0 && errno == EINTR) {
      }
    } else {
      write_dump();
    }
  }
  // AXMP_MESHER_OPTIMIZE=netgen: gmsh's Netgen optimizer (face/edge swaps and smoothing) on the straight-sided tets,
  // before the order is raised. The default optimizer leaves slivers between near-coplanar nodes of curved boundaries
  // (the 5 mm slit of a CORE-type slotted pickup: width 30 nm, aspect 1e5-1e6), on which the port conduction PCG and
  // AMS break down.
  const char* optimize_env = std::getenv("AXMP_MESHER_OPTIMIZE");
  const std::string optimize = optimize_env && *optimize_env ? optimize_env : "default";
  if (optimize != "default" && optimize != "netgen")
    throw Fail("AXMP_MESHER_OPTIMIZE must be netgen or empty, not '" + optimize + "'");
  int order = in.value("order", 2);
  const std::string ho = high_order_mode();
  const std::string output = in.at("output").get<std::string>();
  json ho_info = json::object();
  json ho_passes = json::array();
  std::string patch_optimizer = "not_run";
  // straight-sided tets, raised to order 2 (nodes of curved boundaries on the CAD)
  auto mesh_order2 = [&]() {
    gmsh::model::mesh::generate(3);
    if (optimize == "netgen") gmsh::model::mesh::optimize("Netgen");
    if (order > 1) {
      gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
      gmsh::model::mesh::setOrder(std::min(order, 2));
    }
  };
  // untangle curved elements near thin curved features: gmsh's elastic pass
  auto curve = [&]() {
    if (order > 1) {
      gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
      gmsh::option::setNumber("Mesh.HighOrderOptimize", 2);
      if (ho != "none") gmsh::model::mesh::optimize("HighOrderElastic");
    }
  };
  // auto: the Jacobian-targeted patch optimizer only where the elastic pass leaves invalid elements (it removes every
  // one on the probe shell and core30 of benchmarks/curved_elements_study.py, at 2-5x the elastic pass's meshing time)
  auto count_after_elastic = [&]() {
    const std::size_t after_elastic = ho == "auto" ? invalid_curved_tets() : 0;
    if (ho == "auto") ho_info["invalid_after_elastic"] = after_elastic;
    return ho == "full" || (ho == "auto" && after_elastic > 0);
  };
  auto run_patch_optimizer = [&]() {
    // gmsh reports an optimizer that cannot bring every patch to its target quality ("Failed to reach critical
    // value") as an error, and in the API's error mode that throws from inside the optimizer, where nothing can
    // catch it: the mesher aborted (SIGABRT) on benchmarks/designs/curved_probe_shell.json. Log it instead, keep
    // what the optimizer achieved, and count what is left; the solve straightens any remaining invalid element.
    double abort_mode = 0;
    gmsh::option::getNumber("General.AbortOnError", abort_mode);
    gmsh::option::setNumber("General.AbortOnError", 0);
    // Where invalid elements remain after gmsh's pass, up to two more passes from the improved mesh with patches of 2x
    // and 4x the layers: gmsh's own adaptive strategy grows a failed patch the same way (adaptBlobLayerFact = 2,
    // maxAdaptBlob = 3 in HighOrderMeshOptimizer.h), but its API runs only the fixed-patch strategy. A pass that does
    // not lower the count is undone. Each pass re-optimizes every element outside gmsh's quality band and costs up to
    // about as much as the first, with no time limit (a time limit would make the mesh depend on the machine). With the
    // volume elements loaded in file order they rescued both designs on which gmsh's pass failed:
    // curved_probe_shell.json 6 -> 4 -> 0 (12 layers), core30 2418 -> 2418 -> 2418 (12, undone) -> 0 (24), at 3x the
    // first pass's time. The shuffled load (load_curved_mesh) is
    // a fixed draw that behaves like in-process ones, not a tuned one: over seeds, core30 cleared in gmsh's pass 5 of 5
    // times and the full-size shell 5 of 7 (the other two needed the 12-layer pass), so these
    // passes remain needed.
    double layers = 6;
    gmsh::option::getNumber("Mesh.HighOrderNumLayers", layers);
    std::size_t left = 0;
    for (int pass = 0; pass < 3; ++pass) {
      if (pass > 0 && left == 0) break;
      std::vector<EntityNodes> before;
      if (pass > 0) before = save_node_positions();
      const double pass_layers = layers * (1 << pass);
      gmsh::option::setNumber("Mesh.HighOrderNumLayers", pass_layers);
      const auto start = std::chrono::steady_clock::now();
      gmsh::model::mesh::optimize("HighOrder");
      gmsh::option::setNumber("Mesh.HighOrderNumLayers", layers);
      const std::size_t now = invalid_curved_tets();
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      const bool kept = pass == 0 || now < left;
      if (kept)
        left = now;
      else
        restore_node_positions(before);
      ho_passes.push_back({{"layers", pass_layers}, {"invalid_after", now}, {"kept", kept}, {"seconds", seconds}});
      if (pass > 0)
        std::cerr << "axmp-mesher: note: extra HighOrder pass with " << pass_layers << " layers (" << seconds
                  << " s): " << now << " invalid curved tets, " << (kept ? "kept" : "undone") << "\n";
    }
    gmsh::option::setNumber("General.AbortOnError", abort_mode);
    ho_info["invalid_after_high_order"] = left;
  };
  auto write_mesh = [&]() {
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::option::setNumber("Mesh.Binary", 1);
    gmsh::write(output);
  };
  // element quality: gamma (inscribed/circumscribed radius ratio of the straight element, 1 =
  // regular) and the scaled Jacobian of the curved element; poorly shaped wall elements spoil pointwise wall fields
  auto mesh_quality = [&]() {
    json q = json::object();
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 3);
    std::vector<std::size_t> all;
    for (auto& vv : etags) all.insert(all.end(), vv.begin(), vv.end());
    if (all.empty()) return q;
    std::vector<double> g, sj;
    gmsh::model::mesh::getElementQualities(all, g, "gamma");
    gmsh::model::mesh::getElementQualities(all, sj, "minSJ");
    std::size_t g01 = 0;
    for (double x : g) g01 += x < 0.1;
    q = {{"gamma_min", *std::min_element(g.begin(), g.end())}, {"gamma_below_0.1", g01},
         {"minSJ_min", *std::min_element(sj.begin(), sj.end())}};
    return q;
  };
  auto count_3d = [&]() {
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 3);
    std::size_t n = 0;
    for (auto& vv : etags) n += vv.size();
    return n;
  };
  // Reproducible curved meshes. gmsh orders two order-2 steps by the addresses of the mesh
  // objects (gmsh 4.15.2): the HighOrder patch optimizer keeps each patch in a std::set<MElement *> and numbers its
  // unknowns in that order (contrib/MeshOptimizer/MeshOptimizer.cpp:46, Patch.cpp:81), and the elastic pass sorts the
  // surface triangles it smooths by address (HighOrderMeshElasticAnalogy.cpp:185). Their floating-point results, and
  // the curved mesh, therefore depend on where generate() and setOrder() left the objects, which differs from run to
  // run even on one thread with ASLR off; the straight mesh and setOrder's nodes do not. So a forked child meshes and
  // raises the order, and hands the mesh over (save_curved_mesh); this process, which has not meshed, re-creates it,
  // all nodes and then the elements, the volume elements in a fixed shuffled order (load_curved_mesh), and runs the
  // elastic pass and the patch optimizer. Most of the elements then take fresh memory in a fixed order. What remained
  // with ASLR on: 1 run in 146 differed on the test design and 2 in 81 on a coarse shell, none in
  // 88 with ASLR off, and this process's memory after the geometry differed between runs only with ASLR on; main()
  // therefore re-executes the mesher once with address randomization off (Linux; AXMP_MESHER_ASLR=on keeps it).
  // Whatever this process allocated before the load still decides which free chunks the elements reuse, so that must
  // not depend on the caller either: a one-character longer output path changed the mesh (run_curved_process), and on
  // Linux this process is the curved-mesh process, with a fixed command line, environment and file names.
  // Reproducible for one build of axmp-mesher and gmsh, "threads": 1 (multi-threaded HXT is not reproducible,
  // model3d.cpp), on Linux/glibc. AXMP_MESHER_HO_ISOLATE=0 runs everything in one process (not reproducible).
  const std::string isolate_value = isolate_mode();
  std::string isolation = "n/a";
  int process_threads = 0;
  if (order > 1 && ho != "none") {
    isolation = "isolated";
    process_threads = thread_count();
    if (isolate_value == "0")
      isolation = "off: AXMP_MESHER_HO_ISOLATE=0";
    else if (process_threads > 1)  // forking a multi-threaded process is unsafe; this one should have one thread here
      isolation = "off: " + std::to_string(process_threads) + " threads before the fork";
    if (isolation != "isolated")
      warn_not_reproducible("isolation " + isolation);
    else if (g_canonical != "on")
      warn_not_reproducible("curved-mesh process " + g_canonical);
    else if (address_randomization() == "on")
      warn_not_reproducible("address randomization on");
#ifdef __linux__
    if (isolation == "isolated" && g_canonical == "on" && !running_from_memory())
      std::cerr << "axmp-mesher: warning: the curved-mesh process runs from axmp-mesher's file, not from a copy in memory "
                   "(mesh info \"high_order_executable\"), so curved meshes of STEP solids and of the surface dump can "
                   "depend on where axmp-mesher is installed\n";
#endif
  }
  std::size_t nel = 0;
  json quality = json::object();
  if (isolation != "isolated") {
    gmsh::option::setNumber("General.NumThreads", threads);
    mesh_order2();
    curve();
    if (order > 1 && count_after_elastic()) {
      run_patch_optimizer();
      patch_optimizer = "in_process";
    }
    write_mesh();
    nel = count_3d();
    quality = mesh_quality();
  } else {
    const std::string handoff = output + ".handoff";
    const std::string report = output + ".handoff.json";
    std::remove(handoff.c_str());  // from a run that was killed
    std::remove(report.c_str());
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    const pid_t parent = getpid();
    const pid_t pid = fork();
    if (pid < 0)
      throw Fail(std::string("cannot fork the meshing process (") + std::strerror(errno) + "); set "
                 "AXMP_MESHER_HO_ISOLATE=0 to mesh in one process (curved meshes then differ from run to run)");
    if (pid == 0) {
#ifdef __linux__
      // the child dies with this process (a kill of axmp-mesher's pid alone would otherwise leave it meshing)
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      if (getppid() != parent) std::_Exit(1);
#endif
      json status = json::object();
      status["pid"] = static_cast<long long>(getpid());
      int rc = 0;
      try {
        gmsh::option::setNumber("General.NumThreads", threads);
        mesh_order2();
        if (save_curved_mesh(handoff)) {
          status["next"] = "curve";
        } else {
          // nodes without parametric coordinates cannot be handed over: curve here (not reproducible)
          std::remove(handoff.c_str());
          curve();
          if (count_after_elastic()) {
            run_patch_optimizer();
            status["patch_optimizer"] = "in_process";
          }
          write_mesh();
          status["quality"] = mesh_quality();  // this process holds the mesh; the parent has none
          status["next"] = "done";
          status["isolation"] = "off: nodes without parametric coordinates";
          status["high_order_counts"] = ho_info;
          status["high_order_passes"] = ho_passes;
        }
        status["elements_3d"] = count_3d();
      } catch (const std::exception& e) {
        status["error"] = e.what();
        rc = 1;
      } catch (...) {
        status["error"] = "unknown exception";
        rc = 1;
      }
      std::ofstream(report) << status.dump(-1, ' ', false, json::error_handler_t::replace);
      std::cout.flush();
      std::fflush(nullptr);  // gmsh's terminal output ("verbose")
      std::_Exit(rc);  // no exit handlers: gmsh's state belongs to the parent
    }
    int wstatus = 0;
    while (waitpid(pid, &wstatus, 0) < 0 && errno == EINTR) {
    }
    json status;
    try {
      std::ifstream f(report);
      status = json::parse(f);
    } catch (const std::exception&) {
      status = json();
    }
    std::remove(report.c_str());
    const std::string how_far = "to see how far gmsh got, add \"verbose\": 1 to the axmp-mesher input file (axmp "
                                "writes it as mesh3d_in.json in the mesh directory) and run axmp-mesher on it";
    if (!status.is_object()) {
      std::remove(handoff.c_str());
      if (WIFSIGNALED(wstatus))
        throw Fail("the meshing process (forked by axmp-mesher) was killed by signal " +
                   std::to_string(WTERMSIG(wstatus)) + " (9: out of memory or a job limit; 11: a crash in gmsh); " +
                   how_far);
      throw Fail("the meshing process (forked by axmp-mesher) exited with status " +
                 std::to_string(WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1) + " without a report; " + how_far);
    }
    if (status.value("pid", 0LL) != static_cast<long long>(pid)) {
      std::remove(handoff.c_str());
      throw Fail("the meshing report '" + report + "' comes from another process (pid " +
                 std::to_string(status.value("pid", 0LL)) + ", expected " + std::to_string(pid) +
                 "); is another axmp-mesher writing the same output? Use one output directory per run");
    }
    if (status.contains("error")) {
      std::remove(handoff.c_str());
      throw Fail(status["error"].get<std::string>());
    }
    nel = status.value("elements_3d", std::size_t(0));
    if (status.value("next", std::string()) == "curve") {
      try {
        load_curved_mesh(handoff);
      } catch (...) {
        std::remove(handoff.c_str());
        throw;
      }
      std::remove(handoff.c_str());
      gmsh::option::setNumber("General.NumThreads", threads);
      curve();
      if (count_after_elastic()) {
        run_patch_optimizer();
        patch_optimizer = "isolated";
      }
      write_mesh();
      quality = mesh_quality();
    } else {
      quality = status.value("quality", json::object());
      ho_info = status.value("high_order_counts", json::object());
      ho_passes = status.value("high_order_passes", json::array());
      patch_optimizer = status.value("patch_optimizer", std::string("not_run"));
      isolation = status.value("isolation", isolation);
      if (isolation != "isolated") warn_not_reproducible("isolation " + isolation);
    }
  }
  std::string ver;
  gmsh::option::getString("General.Version", ver);
  const json step_info = info.is_object() ? info.value("step", json()) : json();
  info = {{"ok", true}, {"gmsh_version", ver}, {"elements_3d", nel}, {"groups", groups},
          {"boundary_groups",boundary_groups},{"pec_conductors",conductors},
          {"removed_pec_volumes",removed_volumes.size()},{"symmetry",symmetry},
          {"domain_radius", Rb}, {"device_radius", Rdev}, {"size", h_dev}, {"surface_size", h_s},
          {"shell_inner_radius", Rb}, {"shell_outer_radius", Rs}, {"shell_attr", shell_list.empty() ? -1 : shell_attr},
          {"domain_center", c}, {"mesh_size_min", mesh_size_min}, {"min_curve_radius", min_curve_radius},
          {"high_order_optimize", order > 1 ? ho : std::string("n/a")}, {"high_order_counts", ho_info},
          {"high_order_patch_optimizer", patch_optimizer}, {"high_order_passes", ho_passes},
          {"high_order_isolation", isolation}, {"address_randomization", address_randomization()},
          {"high_order_canonical", isolation == "n/a" ? std::string("n/a") : g_canonical},
          {"high_order_threads_before_fork", process_threads},
          {"tet_optimize", optimize}, {"quality", quality},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
  if (!step_info.is_null()) info["step"] = step_info;
  return 0;
}

// Cross-section ("plane": true) meshing for z-uniform cavities ("symmetry": "xsec"): the domain is the union of
// the region shapes in the (x, y) plane (no half-disk, no outer sphere, no clipping at x < 0). Regions are painted
// in priority order as in the axisymmetric path; curves (sheets) are embedded. The boundary of the union gets
// physical group outer_attr ("outer": the cavity wall). Sizes: "size" everywhere (no far-field growth),
// per-region size/surface_size, and "surface_size" at the outer wall, at the boundaries of "refine_regions" and
// at curves without their own surface_size.
void mesh_plane(const json& in, json& info) {
  auto t0 = std::chrono::steady_clock::now();
  const int outer_attr = in.value("outer_attr", 901);
  const json& regions = in.value("regions", json::array());
  const json& curves = in.value("curves", json::array());
  if (regions.empty()) throw Fail("a cross-section design needs at least one region (the cavity)");
  DimTags objects;
  std::vector<int> owner;
  for (size_t i = 0; i < regions.size(); ++i) {
    DimTags s = build_shape(regions[i].at("shape"));
    for (auto& e : s) {
      objects.push_back(e);
      owner.push_back(static_cast<int>(i));
    }
  }
  const int nreg = static_cast<int>(regions.size());
  for (size_t i = 0; i < curves.size(); ++i) {
    DimTags s = build_path(curves[i].at("path"));
    for (auto& e : s) {
      objects.push_back(e);
      owner.push_back(nreg + static_cast<int>(i));
    }
  }
  DimTags frag;
  std::vector<DimTags> fmap;
  if (objects.size() > 1) {
    occ::fragment(objects, {}, frag, fmap);
  } else {  // a single shape: nothing to fragment (OCC returns no map then)
    fmap = {objects};
  }
  occ::synchronize();
  std::map<int, int> surf_owner;
  std::map<int, std::set<int>> curve_items;
  for (size_t k = 0; k < fmap.size(); ++k) {
    for (auto& e : fmap[k]) {
      if (e.first == 2 && owner[k] < nreg) {
        auto it = surf_owner.find(e.second);
        if (it == surf_owner.end() || it->second < owner[k]) surf_owner[e.second] = owner[k];
      } else if (e.first == 1 && owner[k] >= nreg) {
        curve_items[e.second].insert(owner[k] - nreg);
      }
    }
  }
  DimTags all2;
  gmsh::model::getEntities(all2, 2);
  for (auto& e : all2)
    if (!surf_owner.count(e.second)) throw Fail("internal: a surface of the cross-section has no region");
  // curves must lie in the domain: the midpoint of every curve piece is inside (or on) some surface
  for (auto& [tag, items] : curve_items) {
    std::vector<double> pmin, pmax, xyz;
    gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
    gmsh::model::getValue(1, tag, {0.5 * (pmin[0] + pmax[0])}, xyz);
    bool inside = false;
    for (auto& e : all2)
      if (gmsh::model::isInside(2, e.second, xyz, false) > 0) { inside = true; break; }
    if (!inside)
      throw Fail("curve '" + curves[*items.begin()].at("name").get<std::string>() +
                 "' (or part of it) lies outside every region; a cross-section sheet must lie inside the cavity or "
                 "on its boundary");
  }
  std::map<int, std::vector<int>> by_attr;
  std::map<int, std::string> attr_name;
  for (auto& [tag, ow] : surf_owner) {
    const int attr = regions[ow].at("attr").get<int>();
    by_attr[attr].push_back(tag);
    attr_name[attr] = regions[ow].at("name").get<std::string>();
  }
  json groups = json::array();
  for (auto& [attr, tags] : by_attr) {
    add_physical_group(2, tags, attr, attr_name[attr]);
    groups.push_back({{"dim", 2}, {"attr", attr}, {"name", attr_name[attr]}, {"entities", tags.size()}});
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    if (!by_attr.count(regions[i].at("attr").get<int>()))
      throw Fail("region '" + regions[i].at("name").get<std::string>() +
                 "' is completely covered by later regions; reorder or remove it");
  }
  DimTags bnd;
  gmsh::model::getBoundary(all2, bnd, true, false, false);
  std::vector<int> outer;
  std::vector<double> outer_d;
  for (auto& e : bnd) {
    outer.push_back(std::abs(e.second));
    outer_d.push_back(std::abs(e.second));
  }
  gmsh::model::addPhysicalGroup(1, outer, outer_attr, "outer");
  groups.push_back({{"dim", 1}, {"attr", outer_attr}, {"name", "outer"}});
  for (size_t i = 0; i < curves.size(); ++i) {
    std::vector<int> tags;
    for (auto& [tag, items] : curve_items)
      if (items.count(static_cast<int>(i))) tags.push_back(tag);
    const int attr = curves[i].at("attr");
    add_physical_group(1, tags, attr, curves[i].at("name").get<std::string>());
    groups.push_back({{"dim", 1}, {"attr", attr}, {"name", curves[i].at("name")}, {"entities", tags.size()}});
  }
  // ------------------------------------------------------------ mesh size
  const double h_dev = in.at("size");
  const double h_s = in.value("surface_size", h_dev);
  double hmin = std::min(h_s, h_dev);
  for (const json* list : {&regions, &curves})
    for (auto& g : *list)
      for (const char* key : {"size", "surface_size"}) {
        const double v = g.value(key, 0.0);
        if (v > 0) hmin = std::min(hmin, v);
      }
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", in.value("curvature_points", 24));
  gmsh::option::setNumber("Mesh.MeshSizeMax", h_dev);
  gmsh::option::setNumber("Mesh.MeshSizeMin", hmin * 0.25);
  std::vector<double> fields;
  char expr[64];
  {
    const int fc = gmsh::model::mesh::field::add("MathEval");
    std::snprintf(expr, sizeof expr, "%.17g", h_dev);
    gmsh::model::mesh::field::setString(fc, "F", expr);
    fields.push_back(fc);
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    const double h = regions[i].value("size", 0.0);
    if (h <= 0) continue;
    std::vector<double> tags;
    for (int t : by_attr[regions[i].at("attr").get<int>()]) tags.push_back(t);
    const int fc = gmsh::model::mesh::field::add("MathEval");
    std::snprintf(expr, sizeof expr, "%.17g", h);
    gmsh::model::mesh::field::setString(fc, "F", expr);
    const int fr = gmsh::model::mesh::field::add("Restrict");
    gmsh::model::mesh::field::setNumber(fr, "InField", fc);
    gmsh::model::mesh::field::setNumbers(fr, "SurfacesList", tags);
    fields.push_back(fr);
  }
  const double layer = in.value("surface_layer", 4.0 * h_dev);
  auto refine_towards = [&](const std::vector<double>& cs, double hm, double hx) {
    if (cs.empty() || hm <= 0 || hm >= hx) return;
    const int fd = gmsh::model::mesh::field::add("Distance");
    gmsh::model::mesh::field::setNumbers(fd, "CurvesList", cs);
    gmsh::model::mesh::field::setNumber(fd, "Sampling", 200);
    const int ft = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(ft, "InField", fd);
    gmsh::model::mesh::field::setNumber(ft, "SizeMin", hm);
    gmsh::model::mesh::field::setNumber(ft, "SizeMax", hx);
    gmsh::model::mesh::field::setNumber(ft, "DistMin", 0.5 * hm);
    gmsh::model::mesh::field::setNumber(ft, "DistMax", std::max(layer, 4.0 * hm));
    fields.push_back(ft);
  };
  for (size_t i = 0; i < regions.size(); ++i) {
    const double hr = regions[i].value("surface_size", 0.0);
    if (hr <= 0) continue;
    DimTags sd, cb;
    for (int t : by_attr[regions[i].at("attr").get<int>()]) sd.push_back({2, t});
    gmsh::model::getBoundary(sd, cb, false, false, false);
    std::vector<double> cs;
    for (auto& e : cb) cs.push_back(std::abs(e.second));
    refine_towards(cs, hr, std::max(h_dev, hr));
  }
  std::vector<double> refine_curves = outer_d;
  const std::vector<std::string> refine_list = in.value("refine_regions", std::vector<std::string>{});
  const std::set<std::string> refine(refine_list.begin(), refine_list.end());
  for (auto& [attr, tags] : by_attr) {
    if (!refine.count(attr_name[attr])) continue;
    DimTags sd, cb;
    for (int t : tags) sd.push_back({2, t});
    gmsh::model::getBoundary(sd, cb, false, false, false);
    for (auto& e : cb) refine_curves.push_back(std::abs(e.second));
  }
  std::map<int, std::vector<double>> own_curves;
  for (auto& [tag, items] : curve_items) {
    int own = -1;
    for (int it : items)
      if (curves[it].value("surface_size", 0.0) > 0) { own = it; break; }
    if (own >= 0) own_curves[own].push_back(tag);
    else refine_curves.push_back(tag);
  }
  for (auto& [item, cs] : own_curves) refine_towards(cs, curves[item].value("surface_size", 0.0), h_dev);
  refine_towards(refine_curves, h_s, h_dev);
  const int fmin = gmsh::model::mesh::field::add("Min");
  gmsh::model::mesh::field::setNumbers(fmin, "FieldsList", fields);
  gmsh::model::mesh::field::setAsBackgroundMesh(fmin);
  gmsh::option::setNumber("Mesh.Algorithm", 6);
  gmsh::model::mesh::generate(2);
  const int order = in.value("order", 2);
  if (order > 1) {
    gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
    gmsh::model::mesh::setOrder(std::min(order, 2));
  }
  const json high_order = order > 1 ? untangle_2d() : json();
  for (auto& e : all2) {  // counter-clockwise in (x, y)
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 2, e.second);
    if (et.empty() || enodes[0].size() < 3) continue;
    std::vector<double> c[3], pc;
    int dimn, tagn;
    for (int q = 0; q < 3; ++q) gmsh::model::mesh::getNode(enodes[0][q], c[q], pc, dimn, tagn);
    const double area = (c[1][0] - c[0][0]) * (c[2][1] - c[0][1]) - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1]);
    if (area < 0) gmsh::model::mesh::reverse({{2, e.second}});
  }
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.Binary", 0);
  gmsh::write(in.at("output").get<std::string>());
  std::vector<std::size_t> ntags;
  std::vector<double> ncoord, npar;
  gmsh::model::mesh::getNodes(ntags, ncoord, npar);
  std::vector<int> et;
  std::vector<std::vector<std::size_t>> etags, enodes;
  gmsh::model::mesh::getElements(et, etags, enodes, 2);
  size_t nel = 0;
  for (auto& v : etags) nel += v.size();
  std::string ver;
  gmsh::option::getString("General.Version", ver);
  info = {{"ok", true}, {"plane", true}, {"gmsh_version", ver}, {"nodes", ntags.size()}, {"elements_2d", nel},
          {"groups", groups},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
  if (!high_order.is_null()) info["high_order"] = high_order;
}

// ------------------------------------------------------------------------------------------------- bounded modes
// "bounded": true. The domain is the union of the regions (later regions paint over earlier ones), with no air, no
// half disk and no infinity shell; "axisymmetric" clips every region at r >= 0. Every curve entity of the fragmented
// geometry goes into an entity table (axionmultiphysics/model_selection.hpp), the named boundaries are resolved on it
// (selection::resolve), and every atom of the resolution becomes one physical group of dimension 1, named
// atom_<master>, so that each boundary entity is in exactly one group. The info file carries the table, the atoms and
// the boundaries; with "resolve_only" nothing is meshed.

namespace selection = axmp::model::selection;

// Gauss-Legendre points and weights on [-1, 1].
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 1.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int k = 1; k <= n; ++k) {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2 * k - 1) * z * p1 - (k - 1) * p2) / k;
      }
      dp = n * (z * p0 - p1) / (z * z - 1);
      const double dz = p0 / dp;
      z -= dz;
      if (std::fabs(dz) < 1e-15) break;
    }
    x[i] = z;
    w[i] = 2 / ((1 - z * z) * dp * dp);
  }
}

// The integral of x dy along a curve in the direction of its parametrization (16 intervals of 8 Gauss points: exact
// for lines, and to about 1e-14 for a full circle; it only has to tell +area from -area).
double integral_x_dy(int tag) {
  std::vector<double> pmin, pmax, gx, gw;
  gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
  gauss_legendre(8, gx, gw);
  const int intervals = 16;
  const double h = (pmax[0] - pmin[0]) / intervals;
  std::vector<double> ts, ws, xyz, d;
  for (int k = 0; k < intervals; ++k)
    for (size_t q = 0; q < gx.size(); ++q) {
      ts.push_back(pmin[0] + h * (k + 0.5 * (gx[q] + 1)));
      ws.push_back(0.5 * h * gw[q]);
    }
  gmsh::model::getValue(1, tag, ts, xyz);
  gmsh::model::getDerivative(1, tag, ts, d);
  double s = 0;
  for (size_t i = 0; i < ts.size(); ++i) s += ws[i] * xyz[3 * i] * d[3 * i + 1];
  return s;
}

// The exact bounding box of a curve of the shape grammar (never gmsh's getBoundingBox, which OCC's tolerance inflates
// by about 1e-7 m): the end points (its vertices) and, for an arc of a circle or an ellipse, the points of the full
// curve where x or y is extreme that lie in the arc's parameter range. Such a curve is parametrized by the angle,
// P(t) = C + a cos(t) X + b sin(t) Y, so each coordinate's derivative is alpha cos(t) + beta sin(t). alpha and beta
// come from the first derivatives at two parameters of the arc (gmsh's first derivative is OCC's exact one; its
// second derivative is a finite difference, good to about 1e-7 only), and each stationary parameter is refined by a
// Newton step on the first derivative.
void exact_box(int tag, const std::string& type, double lo[3], double hi[3]) {
  for (int c = 0; c < 3; ++c) {
    lo[c] = c < 2 ? 1e300 : 0.0;
    hi[c] = c < 2 ? -1e300 : 0.0;
  }
  auto add = [&](const double* p) {
    for (int c = 0; c < 2; ++c) {
      lo[c] = std::min(lo[c], p[c]);
      hi[c] = std::max(hi[c], p[c]);
    }
  };
  DimTags ends;
  gmsh::model::getBoundary({{1, tag}}, ends, false, false, false);
  for (const auto& v : ends) {
    std::vector<double> xyz;
    gmsh::model::getValue(0, std::abs(v.second), {}, xyz);
    add(xyz.data());
  }
  std::vector<double> pmin, pmax;
  gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
  if (ends.empty()) {  // a closed curve without a vertex: its value at the start
    std::vector<double> xyz;
    gmsh::model::getValue(1, tag, {pmin[0]}, xyz);
    add(xyz.data());
  }
  if (type == "line") return;
  const double tm = 0.5 * (pmin[0] + pmax[0]), eps = 1e-12 * (pmax[0] - pmin[0]);
  // two parameters at most pi/2 apart (a full circle spans 2 pi), so that the 2 x 2 system is well posed
  const double h = std::min(0.25 * (pmax[0] - pmin[0]), 0.25 * M_PI), ta = tm - h, tb = tm + h, det = std::sin(tb - ta);
  std::vector<double> d1;
  gmsh::model::getDerivative(1, tag, {ta, tb}, d1);
  double alpha[2], beta[2];
  std::vector<double> cand;
  std::vector<int> coord;
  for (int c = 0; c < 2; ++c) {
    // alpha cos(ta) + beta sin(ta) = x'(ta), alpha cos(tb) + beta sin(tb) = x'(tb)
    alpha[c] = (d1[c] * std::sin(tb) - d1[3 + c] * std::sin(ta)) / det;
    beta[c] = (d1[3 + c] * std::cos(ta) - d1[c] * std::cos(tb)) / det;
    if (alpha[c] == 0 && beta[c] == 0) continue;
    const double t1 = std::atan2(alpha[c], -beta[c]);
    for (const double t0 : {t1, t1 + M_PI}) {
      double t = t0 + 2 * M_PI * std::ceil((pmin[0] - eps - t0) / (2 * M_PI));
      for (; t <= pmax[0] + eps; t += 2 * M_PI) {
        cand.push_back(std::clamp(t, pmin[0], pmax[0]));
        coord.push_back(c);
      }
    }
  }
  if (cand.empty()) return;
  std::vector<double> xyz, d;
  gmsh::model::getDerivative(1, tag, cand, d);
  for (size_t i = 0; i < cand.size(); ++i) {  // Newton: x''(t) = -alpha sin(t) + beta cos(t)
    const int c = coord[i];
    const double slope = -alpha[c] * std::sin(cand[i]) + beta[c] * std::cos(cand[i]);
    if (slope != 0) cand[i] = std::clamp(cand[i] - d[3 * i + c] / slope, pmin[0], pmax[0]);
  }
  gmsh::model::getValue(1, tag, cand, xyz);
  gmsh::model::getDerivative(1, tag, cand, d);
  for (size_t i = 0; i < cand.size(); ++i) {
    // the parametrization is the one assumed above: the coordinate is stationary there (or the point is an end)
    const double speed = std::hypot(d[3 * i], d[3 * i + 1]);
    const bool at_end = cand[i] == pmin[0] || cand[i] == pmax[0];
    if (!at_end && std::fabs(d[3 * i + coord[i]]) > 1e-9 * speed)
      throw Fail("internal: curve " + std::to_string(tag) + " is not parametrized by the angle; its box is unknown");
    add(&xyz[3 * i]);
  }
}

// Whether a shape has a contour with an elliptical segment (contour_wire makes it a rational B-spline, whose box,
// length and area the bounded modes do not compute exactly).
bool has_elliptical_contour(const json& s) {
  if (!s.is_object() || s.size() != 1) return false;
  const std::string k = s.begin().key();
  const json& b = s.begin().value();
  if (k == "contour") {
    for (const auto& sg : b.value("segments", json::array()))
      if (sg.contains("ellipse")) return true;
    return false;
  }
  if (k == "union" || k == "difference")
    for (const auto& x : b)
      if (has_elliptical_contour(x)) return true;
  return false;
}

// An axisymmetric region's shape clipped at r >= 0 (a shape that reaches r < 0, such as a disk centred on the axis).
DimTags clip_at_axis(const DimTags& shape, const std::string& name) {
  double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
  for (const auto& e : shape) {
    double a, b, c, d, f, g;
    occ::getBoundingBox(e.first, e.second, a, b, c, d, f, g);
    x0 = std::min(x0, a);
    y0 = std::min(y0, b);
    x1 = std::max(x1, d);
    y1 = std::max(y1, f);
  }
  if (x0 >= 0) return shape;
  const std::string outside = "region '" + name + "' lies at r < 0 only; an axisymmetric region must reach r > 0";
  if (x1 <= 0) throw Fail(outside);
  const double w = std::max(x1 - x0, y1 - y0);
  const int half = occ::addRectangle(0, y0 - w, 0, x1 + w, y1 - y0 + 2 * w);
  DimTags out, surfaces;
  std::vector<DimTags> m;
  occ::intersect(shape, {{2, half}}, out, m);
  for (const auto& e : out)
    if (e.first == 2) surfaces.push_back(e);
  if (surfaces.empty()) throw Fail(outside);
  return surfaces;
}

// The written direction of piece `j` of a curve's path (build_path's order) at the point p: a segment and a polyline
// run from point to point, an arc and a rect path counter-clockwise (increasing angle; a rect path is convex, so the
// counter-clockwise tangent at p has a positive component along the rotated radius from the rectangle's centre).
std::array<double, 2> written_direction(const json& path, size_t j, const double* p) {
  const std::string k = path.begin().key();
  const json& b = path.begin().value();
  if (k == "segment")
    return {b[1][0].get<double>() - b[0][0].get<double>(), b[1][1].get<double>() - b[0][1].get<double>()};
  if (k == "polyline") {
    const json& P = b.at("points");
    const size_t n = P.size(), a = j, c = (j + 1) % n;
    return {P[c][0].get<double>() - P[a][0].get<double>(), P[c][1].get<double>() - P[a][1].get<double>()};
  }
  double cx, cy;
  if (k == "arc") {
    cx = b["center"][0];
    cy = b["center"][1];
  } else if (k == "rect") {
    cx = 0.5 * (b["r"][0].get<double>() + b["r"][1].get<double>());
    cy = 0.5 * (b["z"][0].get<double>() + b["z"][1].get<double>());
  } else {
    throw Fail("unknown path type " + k);
  }
  return {-(p[1] - cy), p[0] - cx};
}

// ------------------------------------------------------------ the partition: the entity table and the faces
//
// The classification of the fragmented 2D geometry that the bounded modes and the export mode share: for every curve
// its type and exact parameters (a line, or the conic c + a cos t u + b sin t v with v = u turned by +90 degrees), its
// end vertices, the surfaces on its sides, and for every surface its loops of signed curve tags.

// An elliptical segment of a contour (axis-aligned: point(t) = center + (axes[0] cos t, axes[1] sin t)). contour_wire
// makes it a rational B-spline, whose parameter is not the angle, so the table takes the segment's own numbers.
struct ContourEllipse {
  double center[2], axes[2];
};

void collect_contour_ellipses(const json& s, std::vector<ContourEllipse>& out) {
  if (!s.is_object() || s.size() != 1) return;
  const std::string k = s.begin().key();
  const json& b = s.begin().value();
  if (k == "contour") {
    for (const auto& sg : b.value("segments", json::array()))
      if (sg.contains("ellipse")) {
        const json& e = sg["ellipse"];
        out.push_back({{e["center"][0].get<double>(), e["center"][1].get<double>()},
                       {e["axes"][0].get<double>(), e["axes"][1].get<double>()}});
      }
  } else if (k == "union" || k == "difference") {
    for (const auto& x : b) collect_contour_ellipses(x, out);
  }
}

// A circle or an ellipse: c + a cos t u + b sin t v, a >= b > 0, u a unit vector, v = u turned by +90 degrees; `ccw`:
// the parameter of the curve it describes turns counter-clockwise as the curve's own parameter grows.
struct Conic {
  bool line = true;
  double c[2] = {0, 0}, a = 0, b = 0, u[2] = {1, 0};
  bool ccw = true;
};

struct CurveKind {
  std::string type;  // "line", "circle", "ellipse"
  Conic g;           // in the sense of the curve's own parametrization
  bool spline = false;
};

// Fits the conic P(t) = C + A cos t + B sin t, the parametrization of an OpenCASCADE circle or ellipse, to a curve
// whose parameter is that angle: A and B from the first derivatives at two parameters (exact in OpenCASCADE), C from
// the values; then three other points of the curve must lie on it. False when the curve is not such a conic.
bool fit_conic(int tag, double pmin, double pmax, Conic& g) {
  const double span = pmax - pmin, tm = 0.5 * (pmin + pmax);
  const double h = std::min(0.25 * span, 0.25 * M_PI), ta = tm - h, tb = tm + h, det = std::sin(tb - ta);
  std::vector<double> d, p;
  gmsh::model::getDerivative(1, tag, {ta, tb}, d);
  double A[2], B[2];
  for (int k = 0; k < 2; ++k) {
    A[k] = (d[k] * std::cos(tb) - d[3 + k] * std::cos(ta)) / det;
    B[k] = (d[k] * std::sin(tb) - d[3 + k] * std::sin(ta)) / det;
  }
  gmsh::model::getValue(1, tag, {ta, tm, tb}, p);
  const double ts[3] = {ta, tm, tb};
  double C[2] = {0, 0};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 2; ++k) C[k] += (p[3 * i + k] - A[k] * std::cos(ts[i]) - B[k] * std::sin(ts[i])) / 3.0;
  const double scale = std::hypot(A[0], A[1]) + std::hypot(B[0], B[1]);
  const double check[5] = {0.0, 0.2, 0.55, 0.8, 1.0};
  std::vector<double> tc;
  for (const double s : check) tc.push_back(pmin + s * span);
  gmsh::model::getValue(1, tag, tc, p);
  for (size_t i = 0; i < tc.size(); ++i)
    for (int k = 0; k < 2; ++k)
      if (std::fabs(p[3 * i + k] - (C[k] + A[k] * std::cos(tc[i]) + B[k] * std::sin(tc[i]))) > 1e-9 * scale) return false;
  const double AA = A[0] * A[0] + A[1] * A[1], BB = B[0] * B[0] + B[1] * B[1], AB = A[0] * B[0] + A[1] * B[1];
  const double t1 = 0.5 * std::atan2(2 * AB, AA - BB);
  double U[2], V[2];
  for (int k = 0; k < 2; ++k) {
    U[k] = A[k] * std::cos(t1) + B[k] * std::sin(t1);
    V[k] = -A[k] * std::sin(t1) + B[k] * std::cos(t1);
  }
  double a = std::hypot(U[0], U[1]), b = std::hypot(V[0], V[1]);
  const double* major = U;
  if (b > a) {
    std::swap(a, b);
    major = V;
  }
  g.line = false;
  g.c[0] = C[0];
  g.c[1] = C[1];
  if (std::fabs(a - b) <= 1e-12 * a) {  // a circle: one radius, the major axis is free
    g.a = g.b = 0.5 * (a + b);
    g.u[0] = 1.0;
    g.u[1] = 0.0;
  } else {
    g.a = a;
    g.b = b;
    g.u[0] = major[0] / a;
    g.u[1] = major[1] / a;
  }
  g.ccw = A[0] * B[1] - A[1] * B[0] > 0;
  return true;
}

// Whether a curve is a straight segment (parametrized along its length, as OpenCASCADE lines are).
bool is_straight(int tag, double pmin, double pmax) {
  const double span = pmax - pmin;
  std::vector<double> p, d;
  gmsh::model::getValue(1, tag, {pmin, pmin + 0.5 * span, pmax}, p);
  gmsh::model::getDerivative(1, tag, {pmin + 0.25 * span, pmin + 0.75 * span}, d);
  const double ex = p[6] - p[0], ey = p[7] - p[1], L = std::hypot(ex, ey);
  if (L == 0) return false;
  if (std::fabs((p[3] - p[0]) * ey - (p[4] - p[1]) * ex) > 1e-10 * L * L) return false;
  for (int i = 0; i < 2; ++i) {
    const double dx = d[3 * i], dy = d[3 * i + 1], s = std::hypot(dx, dy);
    if (s == 0 || std::fabs(dx * ey - dy * ex) > 1e-10 * s * L || dx * ex + dy * ey <= 0) return false;
  }
  return true;
}

// The type of a curve and its exact parameters. A Line, Circle or Ellipse is itself; a TrimmedCurve (what the frames,
// fillets and booleans of OpenCASCADE leave) is resolved to its basis, a line or a conic; the rational B-spline of a
// contour's elliptical segment is that segment's ellipse (the input's numbers).
CurveKind classify_curve(int tag, const std::vector<ContourEllipse>& contours) {
  std::string gtype;
  gmsh::model::getType(1, tag, gtype);
  std::vector<double> pmin, pmax;
  gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
  CurveKind k;
  if (gtype == "Line") {
    k.type = "line";
    return k;
  }
  if (gtype == "Circle" || gtype == "Ellipse" || gtype == "TrimmedCurve") {
    if (gtype == "TrimmedCurve" && is_straight(tag, pmin[0], pmax[0])) {
      k.type = "line";
      return k;
    }
    if (fit_conic(tag, pmin[0], pmax[0], k.g)) {
      k.type = gtype == "Circle" ? "circle" : gtype == "Ellipse" ? "ellipse" : k.g.a == k.g.b ? "circle" : "ellipse";
      return k;
    }
  } else if (gtype.rfind("BSpline", 0) == 0 && !contours.empty()) {
    const double s[3] = {0.2, 0.5, 0.8};
    std::vector<double> ts, p, d;
    for (const double x : s) ts.push_back(pmin[0] + x * (pmax[0] - pmin[0]));
    gmsh::model::getValue(1, tag, ts, p);
    gmsh::model::getDerivative(1, tag, {ts[1]}, d);
    for (const auto& e : contours) {
      bool on = true;
      for (int i = 0; i < 3; ++i) {
        const double x = (p[3 * i] - e.center[0]) / e.axes[0], y = (p[3 * i + 1] - e.center[1]) / e.axes[1];
        if (std::fabs(x * x + y * y - 1.0) > 1e-9) on = false;
      }
      if (!on) continue;
      k.type = "ellipse";
      k.spline = true;
      k.g.line = false;
      k.g.c[0] = e.center[0];
      k.g.c[1] = e.center[1];
      k.g.a = std::max(e.axes[0], e.axes[1]);
      k.g.b = std::min(e.axes[0], e.axes[1]);
      k.g.u[0] = e.axes[0] >= e.axes[1] ? 1.0 : 0.0;
      k.g.u[1] = e.axes[0] >= e.axes[1] ? 0.0 : 1.0;
      k.g.ccw = (p[3] - e.center[0]) * d[1] - (p[4] - e.center[1]) * d[0] > 0;
      return k;
    }
  }
  throw Fail("internal: curve " + std::to_string(tag) + " is of type '" + gtype +
             "'; the geometry kernel's curves are lines, arcs of circles and ellipses (trimmed ones resolved to those) and a "
             "contour's elliptical B-splines, written as ellipses; any other curve is refused");
}

// The angle t of a point on the conic, c + a cos t u + b sin t v.
double conic_angle(const Conic& g, const double* x) {
  const double dx = x[0] - g.c[0], dy = x[1] - g.c[1];
  return std::atan2((-dx * g.u[1] + dy * g.u[0]) / g.b, (dx * g.u[0] + dy * g.u[1]) / g.a);
}

// The exact box of a conic arc from `start` to `end`, turning counter-clockwise iff `ccw` (for the elliptical
// B-splines of a contour, whose box exact_box cannot take from the derivatives).
void conic_box(const Conic& g, bool ccw, const double* start, const double* end, double lo[3], double hi[3]) {
  for (int c = 0; c < 3; ++c) {
    lo[c] = c < 2 ? 1e300 : 0.0;
    hi[c] = c < 2 ? -1e300 : 0.0;
  }
  auto add = [&](const double* p) {
    for (int c = 0; c < 2; ++c) {
      lo[c] = std::min(lo[c], p[c]);
      hi[c] = std::max(hi[c], p[c]);
    }
  };
  add(start);
  add(end);
  const double sense = ccw ? 1.0 : -1.0, t0 = conic_angle(g, start);
  const double v[2] = {-g.u[1], g.u[0]};
  const double sweep = std::fmod(std::fmod(sense * (conic_angle(g, end) - t0), 2 * M_PI) + 2 * M_PI, 2 * M_PI);
  for (int c = 0; c < 2; ++c) {
    const double t1 = std::atan2(g.b * v[c], g.a * g.u[c]);
    for (const double t : {t1, t1 + M_PI}) {
      const double delta = std::fmod(std::fmod(sense * (t - t0), 2 * M_PI) + 2 * M_PI, 2 * M_PI);
      if (delta <= sweep) {
        const double q[2] = {g.c[0] + g.a * std::cos(t) * g.u[0] + g.b * std::sin(t) * v[0],
                             g.c[1] + g.a * std::cos(t) * g.u[1] + g.b * std::sin(t) * v[1]};
        add(q);
      }
    }
  }
}

// The integrals of x0 dx1 and x0^2 dx1 along a table entity from `start` to `end` (the terms of Green's theorem for
// the area and, times pi, the revolved volume of the face on its left).
void green_integrals(const Conic& g, bool ccw, bool closed, const double* start, const double* end, double& g1,
                     double& g2) {
  if (g.line) {
    g1 = 0.5 * (start[0] + end[0]) * (end[1] - start[1]);
    g2 = (start[0] * start[0] + start[0] * end[0] + end[0] * end[0]) * (end[1] - start[1]) / 3.0;
    return;
  }
  static std::vector<double> gx, gw;
  if (gx.empty()) gauss_legendre(24, gx, gw);
  const double sense = ccw ? 1.0 : -1.0, t0 = conic_angle(g, start);
  const double sweep =
      closed ? 2 * M_PI : std::fmod(std::fmod(sense * (conic_angle(g, end) - t0), 2 * M_PI) + 2 * M_PI, 2 * M_PI);
  const double v[2] = {-g.u[1], g.u[0]}, dtds = sense * sweep;
  g1 = g2 = 0;
  const int panels = 16;
  for (int k = 0; k < panels; ++k)
    for (size_t q = 0; q < gx.size(); ++q) {
      const double s = (k + 0.5 * (gx[q] + 1.0)) / panels, w = 0.5 * gw[q] / panels, t = t0 + sense * s * sweep;
      const double x = g.c[0] + g.a * std::cos(t) * g.u[0] + g.b * std::sin(t) * v[0];
      const double dz = -g.a * std::sin(t) * g.u[1] + g.b * std::cos(t) * v[1];
      g1 += w * x * dz * dtds;
      g2 += w * x * x * dz * dtds;
    }
}

struct PartitionInput {
  std::string geometry;
  std::vector<std::string> names;  // the regions (the export mode appends the background, "air")
  int nreg = 0;                    // the regions of the input: owner[k] >= nreg is a curve (index owner[k] - nreg)
  const json* curves = nullptr;
  std::map<int, int> surf_region;                     // surface -> index in names
  std::map<int, std::vector<size_t>> curve_objects;   // curve entity -> its curve objects, input order
  std::vector<int> owner;
  std::vector<size_t> part;
  std::vector<ContourEllipse> contours;
  bool export_mode = false;  // the export: a curve outside every surface (clipped at r < 0) is left out of the table
};

struct Partition {
  selection::Table table;
  std::set<int> reversed;           // entities whose line elements run against their parametrization
  std::vector<json> extras;         // per entity of the table: start, end and the arc keys
  json faces = json::array();
  std::vector<double> area, volume;  // per name: from the loops (Green's theorem on the exact curves)
};

Partition build_partition(const PartitionInput& pin) {
  Partition out;
  selection::Table& table = out.table;
  const bool axisym = pin.geometry == "axisymmetric";
  const std::vector<std::string>& names = pin.names;
  const int nreg = pin.nreg;
  const json& curves = *pin.curves;
  auto curve_name = [&](int c) { return curves[c].at("name").get<std::string>(); };

  // the curve types the classifier knows (lines and arcs), before anything is computed from the curves
  DimTags all1;
  gmsh::model::getEntities(all1, 1);
  std::sort(all1.begin(), all1.end());
  std::map<int, CurveKind> kind;
  for (const auto& c1 : all1) kind[c1.second] = classify_curve(c1.second, pin.contours);

  // the end vertices of every curve: its first and last in the sense of its parametrization (matched to the curve's
  // values at its parameter bounds, which only decides which is which), with the coordinates of the vertices themselves
  struct Ends {
    int vs = 0, ve = 0;  // vertex tags (a negative number for a closed curve without a vertex)
    double s[2] = {0, 0}, e[2] = {0, 0};
    bool closed = false;
  };
  std::map<int, Ends> ends;
  for (const auto& c1 : all1) {
    const int tag = c1.second;
    DimTags vb;
    gmsh::model::getBoundary({{1, tag}}, vb, false, false, false);
    std::vector<int> vt;
    for (const auto& v : vb)
      if (std::find(vt.begin(), vt.end(), std::abs(v.second)) == vt.end()) vt.push_back(std::abs(v.second));
    std::vector<double> pmin, pmax, p;
    gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
    gmsh::model::getValue(1, tag, {pmin[0], pmax[0]}, p);
    Ends en;
    auto xy = [&](int v, double* x) {
      std::vector<double> q;
      gmsh::model::getValue(0, v, {}, q);
      x[0] = q[0];
      x[1] = q[1];
    };
    if (vt.empty()) {  // a closed curve without a vertex: its value at the start
      en.vs = en.ve = -tag;
      en.s[0] = en.e[0] = p[0];
      en.s[1] = en.e[1] = p[1];
      en.closed = true;
    } else if (vt.size() == 1) {
      en.vs = en.ve = vt[0];
      xy(vt[0], en.s);
      en.e[0] = en.s[0];
      en.e[1] = en.s[1];
      en.closed = true;
    } else {
      double x0[2], x1[2];
      xy(vt[0], x0);
      xy(vt[1], x1);
      const bool first = std::hypot(x0[0] - p[0], x0[1] - p[1]) + std::hypot(x1[0] - p[3], x1[1] - p[4]) <=
                         std::hypot(x1[0] - p[0], x1[1] - p[1]) + std::hypot(x0[0] - p[3], x0[1] - p[4]);
      en.vs = first ? vt[0] : vt[1];
      en.ve = first ? vt[1] : vt[0];
      for (int k = 0; k < 2; ++k) {
        en.s[k] = first ? x0[k] : x1[k];
        en.e[k] = first ? x1[k] : x0[k];
      }
    }
    ends[tag] = en;
  }

  // The surfaces on either side of every curve. A surface's oriented boundary has the surface on its left when the
  // integral of x dy along it is +area, on its right when it is -area; uses[curve] holds (region, +1 when that region
  // is on the left of the curve's parametric direction, else -1). A curve inside a surface (a marker that ends in a
  // region) has no upward adjacency: it is embedded in that surface.
  std::map<int, std::vector<std::pair<int, int>>> uses;
  std::map<int, std::pair<DimTags, int>> face_boundary;  // surface -> (its oriented boundary, sense)
  for (const auto& [f, r] : pin.surf_region) {
    DimTags bd;
    gmsh::model::getBoundary({{2, f}}, bd, false, true, false);
    double s = 0, area = 0;
    bool spline = false;
    for (const auto& b : bd) {
      s += (b.second > 0 ? 1 : -1) * integral_x_dy(std::abs(b.second));
      spline = spline || kind.at(std::abs(b.second)).spline;
    }
    occ::getMass(2, f, area);
    // OpenCASCADE's mass of a face bounded by a contour's B-spline is off by a few parts in 1e3: only the sign counts
    int sense;
    if (spline) sense = std::fabs(std::fabs(s) - area) <= 0.05 * area ? (s > 0 ? 1 : -1) : 0;
    else sense = std::fabs(s - area) <= 1e-6 * area ? 1 : std::fabs(s + area) <= 1e-6 * area ? -1 : 0;
    if (sense == 0) throw Fail("internal: the boundary of surface " + std::to_string(f) + " does not enclose its area");
    face_boundary[f] = {bd, sense};
    for (const auto& b : bd) uses[std::abs(b.second)].push_back({r, sense * (b.second > 0 ? 1 : -1)});
  }
  std::map<int, int> embedded_in;
  for (const auto& [f, r] : pin.surf_region) {
    DimTags emb;
    gmsh::model::mesh::getEmbedded(2, f, emb);
    for (const auto& e : emb)
      if (e.first == 1) embedded_in[e.second] = r;
  }

  // ------------------------------------------------------------ the entity table
  table.geometry = pin.geometry;
  table.regions = names;
  struct Geom {
    Conic g;
    bool ccw = true;  // the sense of the arc from start to end
    double s[2], e[2];
    int vs, ve;
    bool closed;
  };
  std::map<int, Geom> geom;  // table orientation: start first along the tangent
  for (const auto& c1 : all1) {
    const int tag = c1.second;
    selection::Entity e;
    e.tag = tag;
    e.dim = 1;
    e.type = kind.at(tag).type;
    std::vector<double> pmin, pmax, p, d;
    gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
    const double tm = 0.5 * (pmin[0] + pmax[0]);
    gmsh::model::getValue(1, tag, {tm}, p);
    gmsh::model::getDerivative(1, tag, {tm}, d);
    const double speed = std::hypot(d[0], d[1]);
    double t[2] = {d[0] / speed, d[1] / speed};
    // sides with respect to the parametric direction
    std::string left, right;
    const auto u = uses.find(tag);
    const auto mk = pin.curve_objects.find(tag);
    if (u != uses.end() && u->second.size() == 2 && u->second[0].second == -u->second[1].second) {
      const bool first_left = u->second[0].second > 0;
      left = names[u->second[first_left ? 0 : 1].first];
      right = names[u->second[first_left ? 1 : 0].first];
    } else if (u != uses.end() && u->second.size() == 1) {
      left = u->second[0].second > 0 ? names[u->second[0].first] : "exterior";
      right = u->second[0].second > 0 ? "exterior" : names[u->second[0].first];
    } else if (u == uses.end() && embedded_in.count(tag)) {
      left = right = names[embedded_in[tag]];
    } else if (u == uses.end() && pin.export_mode) {
      continue;  // what is left of a curve clipped at r < 0
    } else if (u == uses.end() && mk != pin.curve_objects.end()) {
      throw Fail("curve '" + curve_name(pin.owner[mk->second.front()] - nreg) +
                 "' (or part of it) lies outside every enabled region; a marker must lie inside the regions or on "
                 "their boundary");
    } else {
      throw Fail("internal: curve " + std::to_string(tag) + " has an unexpected set of surfaces on its sides");
    }
    // markers: the tangent is the written direction of the first (in document order)
    bool reversed = false;
    if (mk != pin.curve_objects.end()) {
      std::map<int, int> written;  // curve index -> +1 when written along the parametrization, -1 against it
      for (const size_t k : mk->second) {
        const int ci = pin.owner[k] - nreg;
        if (written.count(ci)) continue;
        const auto w = written_direction(curves[ci].at("path"), pin.part[k], p.data());
        const double along = w[0] * d[0] + w[1] * d[1];
        if (std::fabs(along) <= 1e-9 * speed * std::hypot(w[0], w[1]))
          throw Fail("internal: curve " + std::to_string(tag) + " is not along the path of '" + curve_name(ci) + "'");
        written[ci] = along > 0 ? 1 : -1;
      }
      const int first = written.begin()->second;
      for (const auto& [ci, s] : written) {
        e.curves.push_back(curve_name(ci));
        if (s != first) e.curves_reversed.push_back(curve_name(ci));
      }
      if (first < 0) {
        t[0] = -t[0];
        t[1] = -t[1];
        std::swap(left, right);
        out.reversed.insert(tag);
        reversed = true;
      }
    }
    e.side[0] = left;
    e.side[1] = right;
    e.point[0] = p[0];
    e.point[1] = p[1];
    e.tangent[0] = t[0];
    e.tangent[1] = t[1];
    const CurveKind& ck = kind.at(tag);
    const Ends& en = ends.at(tag);
    Geom gm;
    gm.g = ck.g;
    gm.closed = en.closed;
    gm.ccw = reversed ? !ck.g.ccw : ck.g.ccw;
    for (int k = 0; k < 2; ++k) {
      gm.s[k] = reversed ? en.e[k] : en.s[k];
      gm.e[k] = reversed ? en.s[k] : en.e[k];
    }
    gm.vs = reversed ? en.ve : en.vs;
    gm.ve = reversed ? en.vs : en.ve;
    geom[tag] = gm;
    if (ck.spline) conic_box(ck.g, gm.ccw, gm.s, gm.e, e.lo, e.hi);
    else exact_box(tag, e.type, e.lo, e.hi);
    occ::getMass(1, tag, e.length_m);
    table.entities.push_back(e);
  }
  double lo[2] = {1e300, 1e300}, hi[2] = {-1e300, -1e300};
  for (const auto& e : table.entities)
    for (int c = 0; c < 2; ++c) {
      lo[c] = std::min(lo[c], e.lo[c]);
      hi[c] = std::max(hi[c], e.hi[c]);
    }
  table.diagonal_m = std::hypot(hi[0] - lo[0], hi[1] - lo[1]);
  const double tol = 1e-9 * table.diagonal_m;
  if (axisym) {
    for (auto& e : table.entities) {
      // the axis: an entity on the domain boundary with every point (sampled, never by box) at r = 0
      if ((e.side[0] == "exterior") != (e.side[1] == "exterior")) {
        std::vector<double> pmin, pmax, xyz;
        gmsh::model::getParametrizationBounds(1, e.tag, pmin, pmax);
        double rmax = 0;
        for (const double s : {0.0, 0.25, 0.5, 0.75, 1.0}) {
          gmsh::model::getValue(1, e.tag, {pmin[0] + s * (pmax[0] - pmin[0])}, xyz);
          rmax = std::max(rmax, std::fabs(xyz[0]));
        }
        e.axis = rmax < tol;
      }
      if (!e.axis) {  // Pappus: the revolved area is 2 pi r_c L
        double cx, cy, cz;
        occ::getCenterOfMass(1, e.tag, cx, cy, cz);
        e.area_m2 = 2 * M_PI * cx * e.length_m;
      }
    }
    // a vertex on the axis has r = 0.0 exactly, in every entity that has it
    for (auto& [tag, gm] : geom)
      for (double* x : {gm.s, gm.e})
        if (std::fabs(x[0]) <= tol) x[0] = 0.0;
  }

  // the exact parameters of every entity
  for (const auto& e : table.entities) {
    const Geom& gm = geom.at(e.tag);
    json x = json::object();
    x["start"] = json::array({gm.s[0], gm.s[1], 0.0});
    x["end"] = json::array({gm.e[0], gm.e[1], 0.0});
    if (!gm.g.line) {
      x["center"] = json::array({gm.g.c[0], gm.g.c[1], 0.0});
      x["radii"] = json::array({gm.g.a, gm.g.b});
      x["major_axis"] = json::array({gm.g.u[0], gm.g.u[1], 0.0});
      x["ccw"] = gm.ccw;
    }
    out.extras.push_back(x);
  }

  // ------------------------------------------------------------ the faces: loops of signed entity tags
  out.area.assign(names.size(), 0.0);
  out.volume.assign(names.size(), 0.0);
  for (const auto& [f, r] : pin.surf_region) {
    struct Item {
      int tag;   // table sign applied: +tag runs the entity from start to end
      int sign;
      int from, to;  // vertex tags in the direction of the traversal
    };
    const auto& [bd, sense] = face_boundary.at(f);
    std::vector<Item> seq;
    for (const auto& b : bd) {
      const int tag = std::abs(b.second);
      if (!geom.count(tag)) continue;
      const bool rev = out.reversed.count(tag) > 0;
      const int param_sign = sense * (b.second > 0 ? 1 : -1);  // along the parametrization
      const int table_sign = rev ? -param_sign : param_sign;
      const Geom& gm = geom.at(tag);
      seq.push_back({tag, table_sign, table_sign > 0 ? gm.vs : gm.ve, table_sign > 0 ? gm.ve : gm.vs});
    }
    if (sense < 0) std::reverse(seq.begin(), seq.end());  // the wire's order, now in the sense of the traversal
    std::vector<bool> used(seq.size(), false);
    std::vector<std::vector<Item>> loops;
    for (size_t i = 0; i < seq.size(); ++i) {
      if (used[i]) continue;
      std::vector<Item> loop{seq[i]};
      used[i] = true;
      size_t at = i;
      while (loop.back().to != loop.front().from) {
        size_t next = seq.size();
        for (size_t o = 1; o <= seq.size(); ++o) {
          const size_t j = (at + o) % seq.size();
          if (!used[j] && seq[j].from == loop.back().to) {
            next = j;
            break;
          }
        }
        if (next == seq.size()) throw Fail("internal: the boundary of surface " + std::to_string(f) + " does not close");
        used[next] = true;
        loop.push_back(seq[next]);
        at = next;
      }
      loops.push_back(loop);
    }
    // measures by Green's theorem; the outer loop is the one with the largest area, counter-clockwise
    std::vector<double> la, lv;
    for (const auto& loop : loops) {
      double a = 0, v = 0;
      for (const Item& it : loop) {
        const Geom& gm = geom.at(it.tag);
        double g1, g2;
        green_integrals(gm.g, gm.ccw, gm.closed, gm.s, gm.e, g1, g2);
        a += it.sign * g1;
        v += it.sign * M_PI * g2;
      }
      la.push_back(a);
      lv.push_back(v);
    }
    std::vector<size_t> order(loops.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    auto min_tag = [&](size_t i) {
      int m = 1 << 30;
      for (const Item& it : loops[i]) m = std::min(m, it.tag);
      return m;
    };
    size_t outer = 0;
    for (size_t i = 1; i < loops.size(); ++i)
      if (la[i] > la[outer]) outer = i;
    if (!(la[outer] > 0)) throw Fail("internal: surface " + std::to_string(f) + " has no counter-clockwise loop");
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      if ((a == outer) != (b == outer)) return a == outer;
      return min_tag(a) < min_tag(b);
    });
    json jl = json::array();
    for (const size_t i : order) {
      auto& loop = loops[i];
      std::rotate(loop.begin(),
                  std::min_element(loop.begin(), loop.end(), [](const Item& a, const Item& b) { return a.tag < b.tag; }),
                  loop.end());
      json l = json::array();
      for (const Item& it : loop) l.push_back(it.sign * it.tag);
      jl.push_back(l);
      out.area[r] += la[i];
      out.volume[r] += lv[i];
    }
    out.faces.push_back({{"tag", f}, {"region", names[r]}, {"loops", jl}});
  }
  return out;
}

// The bounded planar and axisymmetric modes (header of this file).
void mesh_bounded_2d(const json& in, json& info) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string geometry = in.value("geometry", std::string("planar"));
  if (geometry != "planar" && geometry != "axisymmetric")
    throw Fail("the bounded mode meshes a planar or an axisymmetric geometry, not '" + geometry + "'");
  const bool axisym = geometry == "axisymmetric";
  const std::string exterior = in.value("exterior", std::string("none"));
  if (exterior != "none") throw Fail("the bounded mode takes the exterior \"none\" only, not '" + exterior + "'");
  const json regions = in.value("regions", json::array());
  const json curves = in.value("curves", json::array());
  const json boundaries = in.value("boundaries", json::object());
  if (regions.empty()) throw Fail("the bounded mode needs at least one region (the domain is their union)");
  const int order = in.value("order", 2);
  if (order != 1 && order != 2) throw Fail("the geometry order is 1 or 2, not " + std::to_string(order));
  const bool resolve_only = in.value("resolve_only", false);
  const int nreg = static_cast<int>(regions.size());
  std::vector<std::string> names;
  for (int i = 0; i < nreg; ++i) {
    names.push_back(regions[i].at("name").get<std::string>());
    if (regions[i].value("attr", i + 1) != i + 1)
      throw Fail("region '" + names.back() + "': the attributes of the regions are 1, 2, ... in their order");
    if (has_elliptical_contour(regions[i].at("shape")))
      throw Fail("region '" + names.back() +
                 "': a contour with elliptical segments is not supported by the bounded modes");
  }
  auto curve_name = [&](int c) { return curves[c].at("name").get<std::string>(); };

  // ------------------------------------------------------------ geometry: regions, then curves, fragmented
  DimTags objects;
  std::vector<int> owner;    // region index, or nreg + curve index
  std::vector<size_t> part;  // a curve's piece: its index in build_path's list
  for (int i = 0; i < nreg; ++i) {
    DimTags s = build_shape(regions[i].at("shape"));
    if (axisym) s = clip_at_axis(s, names[i]);
    for (const auto& e : s) {
      objects.push_back(e);
      owner.push_back(i);
      part.push_back(0);
    }
  }
  for (size_t c = 0; c < curves.size(); ++c) {
    const DimTags s = build_path(curves[c].at("path"));
    for (size_t j = 0; j < s.size(); ++j) {
      objects.push_back(s[j]);
      owner.push_back(nreg + static_cast<int>(c));
      part.push_back(j);
    }
  }
  DimTags frag;
  std::vector<DimTags> fmap;
  if (objects.size() > 1) {
    occ::fragment(objects, {}, frag, fmap);
  } else {  // a single shape: nothing to fragment (OCC returns no map then)
    fmap = {objects};
  }
  occ::synchronize();

  // the region of every surface: the last region that contains it
  std::map<int, int> surf_region;
  std::map<int, std::vector<size_t>> curve_objects;  // curve entity -> the curve objects it comes from, input order
  for (size_t k = 0; k < fmap.size(); ++k)
    for (const auto& e : fmap[k]) {
      if (e.first == 2 && owner[k] < nreg) {
        const auto it = surf_region.find(e.second);
        if (it == surf_region.end() || it->second < owner[k]) surf_region[e.second] = owner[k];
      } else if (e.first == 1 && owner[k] >= nreg) {
        curve_objects[e.second].push_back(k);
      }
    }
  DimTags all2;
  gmsh::model::getEntities(all2, 2);
  for (const auto& e : all2)
    if (!surf_region.count(e.second)) throw Fail("internal: a surface of the bounded model has no region");
  std::vector<std::vector<int>> region_surfaces(nreg);
  for (const auto& [tag, r] : surf_region) region_surfaces[r].push_back(tag);
  for (int i = 0; i < nreg; ++i)
    if (region_surfaces[i].empty())
      throw Fail("region '" + names[i] + "' is completely covered by later regions; reorder or remove it");

  PartitionInput pin;
  pin.geometry = geometry;
  pin.names = names;
  pin.nreg = nreg;
  pin.curves = &curves;
  pin.surf_region = surf_region;
  pin.curve_objects = curve_objects;
  pin.owner = owner;
  pin.part = part;
  Partition partition = build_partition(pin);
  selection::Table& table = partition.table;
  const std::set<int>& reversed = partition.reversed;
  const double tol = 1e-9 * table.diagonal_m;
  json jregions = json::object();
  for (int i = 0; i < nreg; ++i) {
    double area = 0, volume = 0;
    for (const int f : region_surfaces[i]) {
      double a, cx, cy, cz;
      occ::getMass(2, f, a);
      area += a;
      if (axisym) {
        occ::getCenterOfMass(2, f, cx, cy, cz);
        volume += 2 * M_PI * cx * a;
      }
    }
    json r = {{"attr", i + 1}, {"area_m2", area}};
    if (axisym) r["volume_m3"] = volume;
    jregions[names[i]] = r;
  }

  // ------------------------------------------------------------ the resolution and the info file
  const selection::Resolution res = selection::resolve(table, boundaries);
  std::map<int, const selection::Entity*> by_tag;
  for (const auto& e : table.entities) by_tag[e.tag] = &e;
  json jentities = selection::to_json(table).at("entities");
  for (size_t i = 0; i < jentities.size(); ++i) {
    json& j = jentities[i];
    const auto it = res.atom_of_tag.find(j.at("tag").get<int>());
    if (it != res.atom_of_tag.end()) j["atom"] = it->second;
    j.update(partition.extras[i]);
  }
  auto side_index = [&](const std::string& s) {
    const auto it = std::find(names.begin(), names.end(), s);
    return static_cast<int>(it - names.begin());  // "exterior": nreg, after every region
  };
  auto side_name = [&](int i) { return i < nreg ? names[i] : std::string("exterior"); };
  const char* k0 = axisym ? "r" : "x";
  const char* k1 = axisym ? "z" : "y";
  json jboundaries = json::object();
  for (const auto& [name, tags] : res.selected) {
    std::vector<int> attributes;
    for (const auto& a : res.atoms)
      if (std::find(a.names.begin(), a.names.end(), name) != a.names.end()) attributes.push_back(a.master);
    std::sort(attributes.begin(), attributes.end());
    double measure = 0, area = 0, blo[2] = {1e300, 1e300}, bhi[2] = {-1e300, -1e300};
    const selection::Entity* first = nullptr;
    std::set<std::pair<int, int>> adjacent;
    for (const int tag : tags) {
      const selection::Entity& e = *by_tag.at(tag);
      measure += e.length_m;
      area += e.area_m2;
      for (int c = 0; c < 2; ++c) {
        blo[c] = std::min(blo[c], e.lo[c]);
        bhi[c] = std::max(bhi[c], e.hi[c]);
      }
      if (!first || std::make_pair(e.point[0], e.point[1]) < std::make_pair(first->point[0], first->point[1]))
        first = &e;
      const int a = side_index(e.side[0]), b = side_index(e.side[1]);
      adjacent.insert({std::min(a, b), std::max(a, b)});
    }
    json b = {{"attributes", attributes}, {"entities", tags.size()}, {"measure", measure}, {"measure_units", "m"}};
    if (axisym) b["area_m2"] = area;
    b["bbox"] = tags.empty() ? json(nullptr) : json{{k0, {blo[0], bhi[0]}}, {k1, {blo[1], bhi[1]}}};
    b["point"] = tags.empty() ? json(nullptr) : json{first->point[0], first->point[1]};
    json adj = json::array();
    for (const auto& [a, c] : adjacent) adj.push_back({side_name(a), side_name(c)});
    b["adjacent"] = adj;
    jboundaries[name] = b;
  }
  json jgroups = json::array();
  for (const auto& a : res.atoms) {
    size_t n = a.tags.size();
    jgroups.push_back({{"attr", a.master},
                       {"name", "atom_" + std::to_string(a.master)},
                       {"names", a.names},
                       {"type", a.kind},
                       {"adjacent", {a.side[0], a.side[1]}},
                       {"entities", n}});
  }
  json jempty = json::object(), jpartial = json::object();
  for (const auto& [name, v] : res.empty) jempty[name] = v;
  for (const auto& [name, v] : res.partial) jpartial[name] = v;
  std::string version;
  gmsh::option::getString("General.Version", version);
  info = {{"ok", true},
          {"bounded", true},
          {"geometry", geometry},
          {"exterior", exterior},
          {"order", order},
          {"gmsh_version", version},
          {"diagonal_m", table.diagonal_m},
          {"tolerance_m", tol},
          {"regions", jregions},
          {"entities", jentities},
          {"faces", partition.faces},
          {"boundaries", jboundaries},
          {"boundary_groups", jgroups},
          {"empty", jempty},
          {"partial", jpartial}};
  if (resolve_only) {
    info["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return;
  }

  // ------------------------------------------------------------ physical groups: one per region, one per atom
  for (int i = 0; i < nreg; ++i) add_physical_group(2, region_surfaces[i], i + 1, names[i]);
  for (const auto& a : res.atoms) add_physical_group(1, a.tags, a.master, "atom_" + std::to_string(a.master));

  // ------------------------------------------------------------ mesh size (as mesh_plane, with no far field)
  const double h_dev = in.at("size");
  const double h_s = in.value("surface_size", h_dev);
  double hmin = std::min(h_s, h_dev);
  for (const json* list : {&regions, &curves})
    for (const auto& g : *list)
      for (const char* key : {"size", "surface_size"}) {
        const double v = g.value(key, 0.0);
        if (v > 0) hmin = std::min(hmin, v);
      }
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", in.value("curvature_points", 24));
  gmsh::option::setNumber("Mesh.MeshSizeMax", h_dev);
  gmsh::option::setNumber("Mesh.MeshSizeMin", hmin * 0.25);
  std::vector<double> fields;
  auto constant_field = [&](double h) {
    char expr[64];
    std::snprintf(expr, sizeof expr, "%.17g", h);
    const int fc = gmsh::model::mesh::field::add("MathEval");
    gmsh::model::mesh::field::setString(fc, "F", expr);
    return fc;
  };
  fields.push_back(constant_field(h_dev));
  for (int i = 0; i < nreg; ++i) {
    const double h = regions[i].value("size", 0.0);
    if (h <= 0) continue;
    const std::vector<double> tags(region_surfaces[i].begin(), region_surfaces[i].end());
    const int fr = gmsh::model::mesh::field::add("Restrict");
    gmsh::model::mesh::field::setNumber(fr, "InField", constant_field(h));
    gmsh::model::mesh::field::setNumbers(fr, "SurfacesList", tags);
    fields.push_back(fr);
  }
  const double layer = in.value("surface_layer", 4.0 * h_dev);
  auto refine_towards = [&](const std::vector<double>& cs, double hm, double hx) {
    if (cs.empty() || hm <= 0 || hm >= hx) return;
    const int fd = gmsh::model::mesh::field::add("Distance");
    gmsh::model::mesh::field::setNumbers(fd, "CurvesList", cs);
    gmsh::model::mesh::field::setNumber(fd, "Sampling", 200);
    const int ft = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(ft, "InField", fd);
    gmsh::model::mesh::field::setNumber(ft, "SizeMin", hm);
    gmsh::model::mesh::field::setNumber(ft, "SizeMax", hx);
    gmsh::model::mesh::field::setNumber(ft, "DistMin", 0.5 * hm);
    gmsh::model::mesh::field::setNumber(ft, "DistMax", std::max(layer, 4.0 * hm));
    fields.push_back(ft);
  };
  for (int i = 0; i < nreg; ++i) {
    const double hr = regions[i].value("surface_size", 0.0);
    if (hr <= 0) continue;
    DimTags sd, cb;
    for (const int f : region_surfaces[i]) sd.push_back({2, f});
    gmsh::model::getBoundary(sd, cb, false, false, false);
    std::vector<double> cs;
    for (const auto& e : cb) cs.push_back(std::abs(e.second));
    refine_towards(cs, hr, std::max(h_dev, hr));
  }
  // the global surface_size at the domain boundary and at the markers without a surface_size of their own
  std::vector<double> refine_curves;
  std::map<int, std::vector<double>> own_curves;
  for (const auto& e : table.entities) {
    if (e.side[0] == "exterior" || e.side[1] == "exterior") refine_curves.push_back(e.tag);
    const auto mk = curve_objects.find(e.tag);
    if (mk == curve_objects.end()) continue;
    int own = -1;
    for (const size_t k : mk->second)
      if (curves[owner[k] - nreg].value("surface_size", 0.0) > 0) {
        own = owner[k] - nreg;
        break;
      }
    if (own >= 0) own_curves[own].push_back(e.tag);
    else if (e.side[0] != "exterior" && e.side[1] != "exterior") refine_curves.push_back(e.tag);
  }
  for (const auto& [ci, cs] : own_curves) refine_towards(cs, curves[ci].value("surface_size", 0.0), h_dev);
  refine_towards(refine_curves, h_s, h_dev);
  const int fmin = gmsh::model::mesh::field::add("Min");
  gmsh::model::mesh::field::setNumbers(fmin, "FieldsList", fields);
  gmsh::model::mesh::field::setAsBackgroundMesh(fmin);
  gmsh::option::setNumber("Mesh.Algorithm", 6);
  gmsh::model::mesh::generate(2);
  if (order == 2) {
    gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
    gmsh::model::mesh::setOrder(2);
  }
  const json high_order = order == 2 ? untangle_2d() : json();
  for (const auto& e : all2) {  // counter-clockwise in (x, y)
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 2, e.second);
    if (et.empty() || enodes[0].size() < 3) continue;
    std::vector<double> c[3], pc;
    int dimn, tagn;
    for (int q = 0; q < 3; ++q) gmsh::model::mesh::getNode(enodes[0][q], c[q], pc, dimn, tagn);
    const double area = (c[1][0] - c[0][0]) * (c[2][1] - c[0][1]) - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1]);
    if (area < 0) gmsh::model::mesh::reverse({{2, e.second}});
  }
  // the line elements of an entity run along its tangent: those of a marker written against the parametrization
  // are turned
  if (!reversed.empty()) {
    DimTags r;
    for (const int tag : reversed) r.push_back({1, tag});
    gmsh::model::mesh::reverse(r);
  }
  std::size_t boundary_elements = 0;
  for (const auto& [tag, master] : res.atom_of_tag) {
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 1, tag);
    if (et.empty() || etags[0].empty()) throw Fail("internal: curve " + std::to_string(tag) + " has no line element");
    for (const auto& v : etags) boundary_elements += v.size();
    // check the first element against the parametrization
    std::vector<double> a, b, pc, par, d;
    int dimn, tagn;
    gmsh::model::mesh::getNode(enodes[0][0], a, pc, dimn, tagn);
    gmsh::model::mesh::getNode(enodes[0][1], b, pc, dimn, tagn);
    gmsh::model::getParametrization(1, tag, {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.0}, par);
    gmsh::model::getDerivative(1, tag, par, d);
    const double along = (b[0] - a[0]) * d[0] + (b[1] - a[1]) * d[1];
    if ((reversed.count(tag) ? -along : along) <= 0)
      throw Fail("internal: the line elements of curve " + std::to_string(tag) + " do not run along its tangent");
  }
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.Binary", 0);
  gmsh::write(in.at("output").get<std::string>());
  std::vector<std::size_t> ntags;
  std::vector<double> ncoord, npar;
  gmsh::model::mesh::getNodes(ntags, ncoord, npar);
  std::vector<int> et;
  std::vector<std::vector<std::size_t>> etags, enodes;
  gmsh::model::mesh::getElements(et, etags, enodes, 2);
  std::size_t nel = 0;
  for (const auto& v : etags) nel += v.size();
  info["nodes"] = ntags.size();
  info["elements_2d"] = nel;
  info["boundary_elements"] = boundary_elements;
  if (!high_order.is_null()) info["high_order"] = high_order;
  info["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The partition of the axisymmetric electromagnetic geometry (--export-regions): the regions, with the background
// "air" last, the entity table with the exact curve parameters, and the faces. Region areas and volumes come from the
// loops (Green's theorem on the curve parameters), not from OpenCASCADE's mass, which is not exact on a face bounded
// by a B-spline. `owner`, `fmap` and `surf_owner` are the construction's: the owner of each input object (-1 for the
// domain, the region index, or nreg + the curve index), what each became, and the owner of each surface (-1: air).
json export_partition(const json& in, const std::vector<int>& owner, const std::vector<DimTags>& fmap,
                      const std::map<int, int>& surf_owner, const std::chrono::steady_clock::time_point& t0) {
  const json regions = in.value("regions", json::array());
  const json curves = in.value("curves", json::array());
  const int nreg = static_cast<int>(regions.size());
  PartitionInput pin;
  pin.geometry = "axisymmetric";
  pin.nreg = nreg;
  pin.curves = &curves;
  pin.export_mode = true;
  for (const auto& r : regions) {
    pin.names.push_back(r.at("name").get<std::string>());
    collect_contour_ellipses(r.at("shape"), pin.contours);
  }
  pin.names.push_back("air");
  for (const auto& [tag, ow] : surf_owner) pin.surf_region[tag] = ow < 0 ? nreg : ow;
  std::map<int, size_t> pieces;  // the next piece of each curve's path
  pin.owner = owner;
  for (size_t k = 0; k < owner.size(); ++k) pin.part.push_back(owner[k] >= nreg ? pieces[owner[k]]++ : 0);
  for (size_t k = 0; k < fmap.size(); ++k)
    for (const auto& e : fmap[k])
      if (e.first == 1 && owner[k] >= nreg) pin.curve_objects[e.second].push_back(k);
  const Partition partition = build_partition(pin);
  json jentities = selection::to_json(partition.table).at("entities");
  for (size_t i = 0; i < jentities.size(); ++i) jentities[i].update(partition.extras[i]);
  json jregions = json::object();
  for (int i = 0; i <= nreg; ++i) {
    const int attr = i < nreg ? regions[i].at("attr").get<int>() : in.value("air_attr", 1);
    jregions[pin.names[i]] = {{"attr", attr}, {"area_m2", partition.area[i]}, {"volume_m3", partition.volume[i]}};
  }
  std::string version;
  gmsh::option::getString("General.Version", version);
  return {{"ok", true},
          {"export_regions", true},
          {"geometry", "axisymmetric"},
          {"gmsh_version", version},
          {"domain_radius", in.at("domain_radius")},
          {"domain_center_z", in.value("domain_center_z", 0.0)},
          {"diagonal_m", partition.table.diagonal_m},
          {"tolerance_m", 1e-9 * partition.table.diagonal_m},
          {"regions", jregions},
          {"entities", jentities},
          {"faces", partition.faces},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
}

// Whether the input is meshed in the curved-mesh process: a 3D input for which mesh3d will raise the order and run the
// curved steps in an isolated parent (its own test, before meshing), or a 2D input that main() handed over because its
// mesh has a tangled element to repair (kTangledRepair, untangle_2d).
bool curved_isolated(const json& in) {
  if (!in.is_object() || in.value("geometry_only", false)) return false;
  const bool mesh3d_input = !in.value("plane", false) && in.value("dim", 2) == 3;
  if (!mesh3d_input && !in.value(kTangledRepair, false)) return false;
  const char* ho = std::getenv("AXMP_MESHER_HO");
  const char* isolate = std::getenv("AXMP_MESHER_HO_ISOLATE");
  return in.value("order", 2) > 1 && !(ho && std::string(ho) == "none") && !(isolate && std::string(isolate) == "0");
}

#ifdef __linux__
// The curved-mesh process. In the isolated parent (mesh3d), the free chunks that its allocations
// before the load leave behind decide where some elements land, and with them the curved mesh. Some of those
// allocations came from outside the input:
// - the output path: with one input and one build, the coarse shell of tests/test_mesh3d_reproducible.py gave one
//   curved mesh in mesh directories up to 63 characters long and another from 64, where the report path
//   output + ".handoff.json" moves from glibc's 96-byte chunks to the 112-byte ones of gmsh's order-2 tetrahedra;
// - for inputs with STEP solids, "step_output" or the surface dump: the executable's path,
//   which gmsh copies at gmsh::initialize and adds to PATH and PYTHONPATH with setenv(), whose array grows with the
//   environment's size; whether stdin, stdout and stderr are regular files (a constructor in libgmsh, probably its
//   static libgfortran, fstat()s fds 0-2 before main and buffers the regular files); and
//   arrays that libraries size by the CPU count.
// So the curved steps run in a fresh process:
// - started as `axmp-mesher input.json` from a copy of this executable in memory (/proc/self/exe then reads
//   "/memfd:axmp-mesher (deleted)"); an executable that finds its libraries through $ORIGIN gets its run path,
//   expanded, in LD_LIBRARY_PATH; if the copy cannot be executed, the process runs from the file and warns;
// - in a private work directory next to the output (curved_names), where input.json is the input with every path
//   replaced by a fixed name, each a symbolic link to the user's file (mesh.msh -> the output, ...), and where the
//   process writes its mesh info (info.json), which run_curved_process copies out before it removes the directory;
// - with a fixed environment of ten variables (curved_env), and with gmsh's -noenv (main);
// - with stdin /dev/null and stdout and stderr pipes, which this process relays (relay_output), and a fixed CPU count
//   (sysconf above). (main also makes stdout's buffer static, which a piped stdout does not need.)
// The dump is written by a short-lived child (mesh3d). Still in the process's history, and so able to change the
// curved mesh: the input itself, including "verbose" and "step_output", and the build.
// A lock file next to the output, locked while the process runs, makes a second axmp-mesher writing the same output
// stop with a message. Each run writes into and removes only what it created.
const char* const kCurvedStage = "AXMP_MESHER_STAGE";
const char* const kCurvedInput = "input.json";
const char* const kCurvedInfo = "info.json";
const char* const kCurvedNames = "names.json";  // fixed name -> the user's path, read only to word an error

// A signal (SIGTERM, SIGINT, SIGHUP) that came while the curved-mesh process ran: passed on to it, and raised again in
// this process once it has cleaned up (main).
volatile sig_atomic_t g_stop_signal = 0;
pid_t g_curved_pid = 0;

extern "C" void forward_stop(int sig) {
  g_stop_signal = sig;
  if (g_curved_pid > 0) kill(g_curved_pid, sig);
}

// JSON as text, with any byte that is not UTF-8 replaced (paths need not be UTF-8; dump() would throw).
std::string json_text(const json& j, int indent) {
  return j.dump(indent, ' ', false, json::error_handler_t::replace);
}

// The input as the curved-mesh process reads it: each path replaced by a fixed name (keeping its extension, by which
// gmsh picks a format); `links` gets each name and the absolute path it stands for. The files of STEP solids, in the
// regions and in "domain_solid" alike, and the size field file must exist (Fail names the missing one).
json curved_input(const json& in, std::vector<std::pair<std::string, std::string>>& links) {
  json c = in;
  auto fixed = [&](const std::string& name, const std::string& path) {
    const std::string local = name + std::filesystem::path(path).extension().string();
    links.push_back({local, std::filesystem::absolute(path).string()});
    return local;
  };
  auto existing = [&](const std::string& name, const std::string& path, const std::string& what) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
      throw Fail("the " + what + " '" + path + "' does not exist (absolute path '" +
                 std::filesystem::absolute(path).string() + "')");
    return fixed(name, path);
  };
  c["output"] = fixed("mesh", in.at("output").get<std::string>());
  c["info"] = kCurvedInfo;
  if (in.contains("step_output")) c["step_output"] = fixed("geometry", in.at("step_output").get<std::string>());
  if (!in.value("size_field_file", std::string()).empty())
    c["size_field_file"] = existing("size_field", in.at("size_field_file").get<std::string>(), "size field file");
  int k = 0;
  std::function<void(json&)> walk = [&](json& j) {  // the files of "step" solids, anywhere in the input
    if (j.is_object() && j.contains("step") && j["step"].is_object() && j["step"].contains("file") &&
        j["step"]["file"].is_string())
      j["step"]["file"] = existing("input_" + std::to_string(k++), j["step"]["file"].get<std::string>(), "STEP file");
    if (j.is_structured())
      for (auto& v : j) walk(v);
  };
  walk(c);
  return c;
}

// The executable's run path (DT_RPATH, DT_RUNPATH) with $ORIGIN expanded to its directory, if it has $ORIGIN: the
// curved-mesh process gets it in LD_LIBRARY_PATH, since from memory $ORIGIN cannot be resolved (the wheel's mesher,
// CMakeLists.txt). Empty otherwise.
std::string expanded_origin_path() {
  const char* strings = nullptr;
  for (const ElfW(Dyn)* d = _DYNAMIC; d->d_tag != DT_NULL; ++d)
    if (d->d_tag == DT_STRTAB) strings = reinterpret_cast<const char*>(d->d_un.d_ptr);
  if (!strings) return "";
  char exe[4096];
  const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (n <= 0) return "";
  exe[n] = '\0';
  const std::string dir = std::filesystem::path(exe).parent_path().string();
  std::string out;
  for (const ElfW(Dyn)* d = _DYNAMIC; d->d_tag != DT_NULL; ++d) {
    if (d->d_tag != DT_RPATH && d->d_tag != DT_RUNPATH) continue;
    std::string path = strings + d->d_un.d_val;
    if (path.find("$ORIGIN") == std::string::npos && path.find("${ORIGIN}") == std::string::npos) continue;
    for (const std::string token : {"${ORIGIN}", "$ORIGIN"})
      for (std::size_t at = path.find(token); at != std::string::npos; at = path.find(token, at + dir.size()))
        path.replace(at, token.size(), dir);
    out += (out.empty() ? "" : ":") + path;
  }
  return out;
}

// The curved-mesh process's environment, sorted, always the same ten variables: the six AXMP_MESHER_* switches (empty
// where the caller set none; AXMP_MESHER_SURFACE_DUMP made absolute), the stage marker, LD_LIBRARY_PATH (the expanded
// $ORIGIN run path, then the caller's; empty if neither), and OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. The last
// two make it single-threaded at the fork: libgmsh links OpenBLAS statically, which starts a thread per CPU when it is
// loaded unless told otherwise (without them, 4 threads in a 4-CPU job, and no isolation; the caller's environment
// may set neither). LD_PRELOAD, locale, OpenMP placement, HOME, malloc tunables, SLURM's variables and the rest are left
// out.
std::vector<std::string> curved_env() {
  std::vector<std::string> env;
  for (const char* name : {"AXMP_MESHER_ASLR", "AXMP_MESHER_CURVATURE_FLOOR", "AXMP_MESHER_HO", "AXMP_MESHER_HO_ISOLATE",
                           "AXMP_MESHER_OPTIMIZE", "AXMP_MESHER_SURFACE_DUMP", "LD_LIBRARY_PATH"}) {
    const char* v = std::getenv(name);
    std::string value = v ? v : "";
    if (std::string(name) == "AXMP_MESHER_SURFACE_DUMP" && !value.empty())
      value = std::filesystem::absolute(value).string();
    if (std::string(name) == "LD_LIBRARY_PATH") {
      const std::string origin = expanded_origin_path();
      if (!origin.empty()) value = value.empty() ? origin : origin + ":" + value;
    }
    env.push_back(std::string(name) + "=" + value);
  }
  env.push_back(std::string(kCurvedStage) + "=curved " + std::to_string(static_cast<long>(getpid())));
  env.push_back("OPENBLAS_NUM_THREADS=1");
  env.push_back("OMP_NUM_THREADS=1");
  std::sort(env.begin(), env.end());
  return env;
}

// A copy of this executable in memory, to start the curved-mesh process from: its /proc/self/exe then reads
// "/memfd:axmp-mesher (deleted)" wherever axmp-mesher is installed. -1, with the reason, if that is not possible.
int executable_in_memory(std::string& why) {
  const int src = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
  const int fd = src < 0 ? -1 : memfd_create("axmp-mesher", MFD_CLOEXEC);
  if (fd < 0) {
    why = std::string("memfd_create or /proc/self/exe: ") + std::strerror(errno);
    if (src >= 0) close(src);
    return -1;
  }
  char buf[1 << 16];
  ssize_t n = 0;
  while ((n = read(src, buf, sizeof buf)) > 0)
    if (write(fd, buf, static_cast<size_t>(n)) != n) n = -1;
  const int err = errno;
  close(src);
  if (n < 0) {
    why = std::string("copying the executable: ") + std::strerror(err);
    close(fd);
    return -1;
  }
  return fd;
}

// The names next to `output` of its lock file and work directory: <output>.axmp-mesher.lock and
// <output>.axmp-mesher.XXXXXX, or for an output file name longer than 200 bytes its first 200 and a hash of all of it
// in place of the name, so that both stay within the file system's 255 bytes whatever output name gmsh accepts.
std::string curved_stem(const std::string& output) {
  const std::filesystem::path p(output);
  const std::string name = p.filename().string();
  if (name.size() <= 200) return output;
  std::uint64_t h = 1469598103934665603ull;  // FNV-1a
  for (unsigned char ch : name) {
    h ^= ch;
    h *= 1099511628211ull;
  }
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
  return (p.parent_path() / (name.substr(0, 200) + "-" + hex)).string();
}

// Takes the lock file `lock_path` (flock), so that two axmp-meshers writing one output do not run at once. Returns its
// descriptor; -1 with `error` set if another axmp-mesher holds it or the file cannot be used; -2 if the file system
// cannot lock (then nothing is checked). `created`: this run created the file, wrote "host <name> pid <n>" in it, and
// removes it at the end. A file it did not create is only locked: never written, followed (a symbolic link is
// refused) or removed, and it must be a regular file of this user.
int lock_output(const std::string& lock_path, const std::string& output, bool& created, std::string& error) {
  for (int attempt = 0; attempt < 8; ++attempt) {
    created = false;
    int fd = open(lock_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd >= 0) {
      created = true;
    } else if (errno == EEXIST) {
      fd = open(lock_path.c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
      if (fd < 0 && errno == ENOENT) continue;  // removed since: try again
    }
    if (fd < 0) {
      const int err = errno;
      if (err == ELOOP)
        error = "'" + lock_path + "' is a symbolic link; axmp-mesher locks the output '" + output + "' with a regular "
                "file of that name, and does not follow links there. Remove it";
      else
        error = "cannot create the lock file '" + lock_path + "' (" + std::strerror(err) + "); is the output "
                "directory writable?";
      return -1;
    }
    struct stat held;
    if (fstat(fd, &held) != 0 || !S_ISREG(held.st_mode) || held.st_uid != geteuid()) {
      close(fd);
      error = "'" + lock_path + "' is not a regular file of yours; axmp-mesher locks the output '" + output +
              "' with a file of that name. Remove it, or use another output name";
      return -1;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
      const int err = errno;
      if (err == EWOULDBLOCK) {
        char holder[160] = {0};
        const ssize_t got = pread(fd, holder, sizeof holder - 1, 0);
        if (got > 0) holder[got] = '\0';
        for (char& ch : holder)
          if (ch == '\n') ch = '\0';
        close(fd);
        error = "another axmp-mesher (" + std::string(got > 0 ? holder : "host and pid unknown") + ") is writing '" +
                output + "' now; use one output directory per run, or wait for it to finish";
        return -1;
      }
      if (created) unlink(lock_path.c_str());
      close(fd);
      return -2;
    }
    struct stat now;
    if (lstat(lock_path.c_str(), &now) == 0 && S_ISREG(now.st_mode) && held.st_ino == now.st_ino &&
        held.st_dev == now.st_dev) {
      if (created) {
        char host[128] = {0};
        gethostname(host, sizeof host - 1);
        const std::string owner = "host " + std::string(host) + " pid " + std::to_string(static_cast<long>(getpid())) +
                                  "\n";
        if (pwrite(fd, owner.data(), owner.size(), 0) < 0) {
          // the lock holds without the note
        }
      }
      return fd;
    }
    close(fd);  // the file was removed and made again meanwhile: lock the new one
  }
  error = "cannot take the lock '" + lock_path + "' (it keeps being replaced)";
  return -1;
}

// What one run made next to the output, removed when this goes out of scope (also on an exception): its work
// directory, and its lock (the file only if the run created it).
struct CurvedRunFiles {
  std::string lock_path;
  int lock_fd = -1;
  bool lock_created = false;
  std::string work;
  ~CurvedRunFiles() {
    std::error_code ec;
    if (!work.empty()) std::filesystem::remove_all(work, ec);
    if (lock_fd >= 0) {
      if (lock_created) unlink(lock_path.c_str());
      close(lock_fd);
    }
  }
};

// Passes the curved-mesh process's output on to this process's stdout and stderr until both pipes close. That process
// writes to pipes and reads /dev/null, whatever this one has: with stdin, stdout or stderr a regular file, the curved
// mesh of a STEP input differed even with nothing written.
void relay_output(int out_fd, int err_fd) {
  std::signal(SIGPIPE, SIG_IGN);  // a closed reader must not end this process before the mesher does
  struct pollfd fds[2] = {{out_fd, POLLIN, 0}, {err_fd, POLLIN, 0}};
  const int to[2] = {STDOUT_FILENO, STDERR_FILENO};
  bool writable[2] = {true, true};
  int open_count = 2;
  char buf[1 << 16];
  while (open_count > 0) {
    if (poll(fds, 2, -1) < 0) {
      if (errno == EINTR) continue;
      break;
    }
    for (int i = 0; i < 2; ++i) {
      if (fds[i].fd < 0 || !(fds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      const ssize_t n = read(fds[i].fd, buf, sizeof buf);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        close(fds[i].fd);
        fds[i].fd = -1;
        --open_count;
        continue;
      }
      for (ssize_t done = 0; writable[i] && done < n;) {
        const ssize_t w = write(to[i], buf + done, static_cast<size_t>(n - done));
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) writable[i] = false;  // keep reading, so that the mesher does not block
        else done += w;
      }
    }
  }
}

// In the curved-mesh process, an error message with the fixed names replaced by the user's paths (names.json).
std::string user_paths(std::string message) {
  json names;
  try {
    std::ifstream f(kCurvedNames);
    names = json::parse(f);
  } catch (const std::exception&) {
    return message;
  }
  std::vector<std::pair<std::string, std::string>> pairs;
  for (auto it = names.begin(); it != names.end(); ++it) pairs.push_back({it.key(), it.value().get<std::string>()});
  std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  std::string out;  // one pass, longest name first, so that no replacement is replaced again
  for (std::size_t i = 0; i < message.size();) {
    bool hit = false;
    for (const auto& [local, path] : pairs)
      if (message.compare(i, local.size(), local) == 0) {
        out += path;
        i += local.size();
        hit = true;
        break;
      }
    if (!hit) out += message[i++];
  }
  return out;
}

// Makes the curved mesh of `in` in the curved-mesh process and waits for it. Returns false, with the reason, if that
// process cannot be set up or started; main() then stops with that reason (`optout_helps`: whether meshing in one
// process could do better, which a full file system, for one, rules out). Otherwise returns true and sets `rc` to the
// exit status for main(); the mesh is at the output path and its info, also on failure, at the input's "info" path.
bool run_curved_process(const json& in, int& rc, std::string& why, bool& optout_helps) {
  namespace fs = std::filesystem;
  optout_helps = true;
  const std::string info_path = in.value("info", std::string("mesh_info.json"));
  auto fail = [&](const std::string& error) {  // an error in the input or in its use: reported, rc 1
    std::cerr << "axmp-mesher: " << error << "\n";
    std::ofstream(info_path) << json_text(json({{"ok", false}, {"error", error}}), 2);
    rc = 1;
    return true;
  };
  if (!in.contains("output") || !in.at("output").is_string()) return fail("the input has no \"output\" path");
  const std::string output = in.at("output").get<std::string>();
  std::error_code ec;
  const fs::path output_dir = fs::absolute(output).parent_path();
  if (!fs::is_directory(output_dir, ec))
    return fail("the output directory '" + output_dir.string() + "' does not exist; create it first");
  std::vector<std::pair<std::string, std::string>> links;
  json canonical;
  try {
    canonical = curved_input(in, links);
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  const std::string stem = curved_stem(output);
  CurvedRunFiles files;
  files.lock_path = stem + ".axmp-mesher.lock";
  std::string lock_error;
  files.lock_fd = lock_output(files.lock_path, output, files.lock_created, lock_error);
  if (files.lock_fd == -1) return fail(lock_error);
  auto full = [&](int err) {
    if (err == ENOSPC || err == EDQUOT) optout_helps = false;
  };
  std::string work_template = stem + ".axmp-mesher.XXXXXX";
  if (!mkdtemp(work_template.data())) {
    full(errno);
    why = "cannot create a work directory next to the output '" + output + "' (" + std::strerror(errno) + ")";
    return false;
  }
  files.work = work_template;
  const std::string& work = files.work;
  json names = json::object();
  for (const auto& [name, target] : links) {
    fs::create_symlink(target, fs::path(work) / name, ec);
    if (ec) {
      full(ec.value());
      why = "cannot make the link '" + name + "' to '" + target + "' in its work directory (" + ec.message() + ")";
      return false;
    }
    names[name] = target;
  }
  // the handoff files are in the work directory, which is gone when an error is read
  const std::string handoff = links.front().first + ".handoff";
  names[handoff + ".json"] = handoff + ".json (in the work directory next to '" + output + "')";
  names[handoff] = handoff + " (in the work directory next to '" + output + "')";
  {
    std::ofstream f(fs::path(work) / kCurvedInput);
    f << canonical.dump(1);
    std::ofstream g(fs::path(work) / kCurvedNames);
    g << json_text(names, 1);
    if (!f || !g) {
      full(errno);
      why = "cannot write in its work directory next to the output '" + output + "'";
      return false;
    }
  }
  std::string exe_why;
  const int exe_fd = executable_in_memory(exe_why);
  const std::vector<std::string> env = curved_env();
  std::vector<char*> envp;
  for (const auto& s : env) envp.push_back(const_cast<char*>(s.c_str()));
  envp.push_back(nullptr);
  std::string arg0 = "axmp-mesher";
  std::string arg1 = kCurvedInput;
  char* args[] = {arg0.data(), arg1.data(), nullptr};
  // status_pipe: the child writes {1, errno} there if the copy in memory cannot be executed (it then tries the file),
  // and {2, errno} if the file cannot either (closed on a successful exec); out_pipe and err_pipe: the process's stdout
  // and stderr (relay_output), and /dev/null its stdin
  int status_pipe[2];
  int out_pipe[2];
  int err_pipe[2];
  const int null_fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
  if (null_fd < 0 || pipe2(status_pipe, O_CLOEXEC) != 0 || pipe2(out_pipe, O_CLOEXEC) != 0 ||
      pipe2(err_pipe, O_CLOEXEC) != 0) {
    if (exe_fd >= 0) close(exe_fd);
    why = std::string("cannot open /dev/null or make pipes (") + std::strerror(errno) + ")";
    return false;
  }
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  const pid_t parent = getpid();
  const pid_t pid = fork();
  if (pid < 0) {
    for (int fd : {status_pipe[0], status_pipe[1], out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1], null_fd})
      close(fd);
    if (exe_fd >= 0) close(exe_fd);
    why = std::string("cannot fork (") + std::strerror(errno) + ")";
    return false;
  }
  if (pid == 0) {
    close(status_pipe[0]);
    prctl(PR_SET_PDEATHSIG, SIGKILL);  // it (and its meshing child) die with this process
    if (getppid() != parent) std::_Exit(1);
    int report[2] = {0, 0};
    if (dup2(null_fd, STDIN_FILENO) < 0 || dup2(out_pipe[1], STDOUT_FILENO) < 0 ||
        dup2(err_pipe[1], STDERR_FILENO) < 0 || chdir(work.c_str()) != 0) {
      report[0] = 2;
      report[1] = errno;
    } else {
      if (exe_fd >= 0) {
        fexecve(exe_fd, args, envp.data());
        report[0] = 1;  // the copy in memory cannot be executed (vm.memfd_noexec, SELinux, ...): the file
        report[1] = errno;
        if (write(status_pipe[1], report, sizeof report) != static_cast<ssize_t>(sizeof report)) std::_Exit(126);
      }
      execve("/proc/self/exe", args, envp.data());
      report[0] = 2;
      report[1] = errno;
    }
    if (write(status_pipe[1], report, sizeof report) != static_cast<ssize_t>(sizeof report)) std::_Exit(126);
    std::_Exit(127);
  }
  g_curved_pid = pid;
  for (int sig : {SIGTERM, SIGINT, SIGHUP}) {  // passed on (forward_stop); not if the caller ignores them
    struct sigaction old = {};
    sigaction(sig, nullptr, &old);
    if (old.sa_handler == SIG_IGN) continue;
    struct sigaction act = {};
    act.sa_handler = forward_stop;
    sigemptyset(&act.sa_mask);
    sigaction(sig, &act, nullptr);
  }
  close(status_pipe[1]);
  close(out_pipe[1]);
  close(err_pipe[1]);
  close(null_fd);
  if (exe_fd >= 0) close(exe_fd);
  int reports[4] = {0, 0, 0, 0};
  std::size_t got = 0;
  while (got < sizeof reports) {
    const ssize_t n = read(status_pipe[0], reinterpret_cast<char*>(reports) + got, sizeof reports - got);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    got += static_cast<std::size_t>(n);
  }
  close(status_pipe[0]);
  relay_output(out_pipe[0], err_pipe[0]);
  int wstatus = 0;
  while (waitpid(pid, &wstatus, 0) < 0 && errno == EINTR) {
  }
  g_curved_pid = 0;
  std::string executable = exe_fd >= 0 ? "in memory" : "file (" + exe_why + ")";
  for (std::size_t r = 0; r + 1 < got / sizeof(int); r += 2) {
    if (reports[r] == 1) executable = std::string("file (the copy in memory cannot be executed: ") +
                                      std::strerror(reports[r + 1]) + ")";
    if (reports[r] == 2) {
      why = "cannot start it for the output '" + output + "' (" + std::strerror(reports[r + 1]) + ")";
      return false;
    }
  }
  json info;
  try {
    std::ifstream f(fs::path(work) / kCurvedInfo);
    info = json::parse(f);
  } catch (const std::exception&) {
    info = json();
  }
  if (!info.is_object()) {
    std::string what;
    if (g_stop_signal)
      what = "was stopped by signal " + std::to_string(g_stop_signal) + ", sent to axmp-mesher";
    else if (WIFSIGNALED(wstatus))
      what = "was killed by signal " + std::to_string(WTERMSIG(wstatus)) + " (9: out of memory or a job limit; 11: a "
             "crash in gmsh)";
    else
      what = "exited with status " + std::to_string(WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1) +
             " without its mesh info";
    info = {{"ok", false}, {"error", "the curved-mesh process for the output '" + output + "' " + what}};
    std::cerr << "axmp-mesher: " << info["error"].get<std::string>() << "\n";
  }
  if (info.contains("step") && info["step"].is_object() && in.contains("step_output"))
    info["step"]["file"] = in.at("step_output");
  if (info.contains("high_order_canonical")) info["high_order_executable"] = executable;
  std::ofstream(info_path) << json_text(info, 2);
  rc = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : 1;
  if (rc == 0 && !info.value("ok", false)) rc = 1;
  return true;
}

// Makes the mesh of `in` in the curved-mesh process (run_curved_process) and returns the exit status for main(). If
// that process cannot be started, the mesher stops with the reason: a curved mesh made here would depend on the output
// path again.
int mesh_in_curved_process(const json& in) {
  int rc = 0;
  std::string why;
  bool optout_helps = true;
  bool started = false;
  try {
    started = run_curved_process(in, rc, why, optout_helps);
  } catch (const std::exception& e) {  // what the run made is removed already (CurvedRunFiles)
    why = e.what();
    optout_helps = false;
  }
  if (g_stop_signal) {  // stopped by a signal: raise it again, now that the run has cleaned up
    const int sig = g_stop_signal;
    std::signal(sig, SIG_DFL);
    std::raise(sig);
  }
  if (started) return rc;
  std::string error = "cannot start the curved-mesh process (" + why + ")";
  if (optout_helps)
    error += "; set AXMP_MESHER_HO_ISOLATE=0 to mesh in one process (curved meshes then differ from run to run and "
             "between output directories)";
  std::cerr << "axmp-mesher: " << error << "\n";
  const json failed = {{"ok", false}, {"error", error}};
  std::ofstream(in.value("info", std::string("mesh_info.json"))) << json_text(failed, 2);
  return 1;
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef __linux__
  // stdin, stdout and stderr open (/dev/null where a caller closed one), so that no file of the mesher's lands on
  // fds 0-2
  for (int fd = 0; fd <= 2; ++fd) {
    if (fcntl(fd, F_GETFD) != -1 || errno != EBADF) continue;
    const int null_fd = open("/dev/null", O_RDWR);
    if (null_fd >= 0 && null_fd != fd) {
      dup2(null_fd, fd);
      close(null_fd);
    }
  }
#endif
  // stdout's buffer is static, so that it is not allocated at the first write (not needed for the curved-mesh
  // process, whose stdout is a pipe)
  static char stdout_buffer[8192];
  setvbuf(stdout, stdout_buffer, isatty(STDOUT_FILENO) ? _IOLBF : _IOFBF, sizeof stdout_buffer);
  const bool export_regions = argc == 3 && std::string(argv[1]) == "--export-regions";
  if (argc != 2 && !export_regions) {
    std::cerr << "usage: axmp-mesher [--export-regions] input.json\n";
    return 2;
  }
  const char* aslr_env = std::getenv("AXMP_MESHER_ASLR");
  const std::string aslr = aslr_env ? aslr_env : "";
  if (aslr != "" && aslr != "off" && aslr != "on") {
    std::cerr << "axmp-mesher: AXMP_MESHER_ASLR must be off (the default), on or empty, not '" << aslr << "'\n";
    return 2;
  }
#ifdef __linux__
  // Re-run once with address-space randomization off (what `setarch -R` does). The curved-element steps order by
  // address (mesh3d), and with randomization on, the memory the geometry leaves behind, and with it a curved mesh,
  // still differed in about 3 runs in 140 (none in 88 with it off). AXMP_MESHER_ASLR=on keeps randomization.
  const int persona = personality(0xffffffff);
  if (aslr != "on" && persona != -1 && !(persona & ADDR_NO_RANDOMIZE) &&
      personality(static_cast<unsigned long>(persona) | ADDR_NO_RANDOMIZE) != -1) {
    execv("/proc/self/exe", argv);
    personality(static_cast<unsigned long>(persona));  // exec failed: carry on with randomization
  }
#endif
  // Settings that would change meshes behind the input's back: gmsh's ~/.gmshrc and ~/.gmsh-options are not read
  // (gmsh::initialize below), and PETSc's options, which gmsh's elastic pass reads for its linear solve, are cleared.
  unsetenv("PETSC_OPTIONS");
  if (export_regions) argv[1] = argv[2];  // the input file; the flag is not needed again
  auto t0 = std::chrono::steady_clock::now();
  json in;
  try {
    std::ifstream f(argv[1]);
    in = json::parse(f);
  } catch (std::exception& e) {
    std::cerr << "axmp-mesher: cannot read input: " << e.what() << "\n";
    return 2;
  }
  const bool export_3d = export_regions && in.value("dim", 2) == 3 && !in.value("bounded", false) && !in.value("plane", false);
  if (export_regions && (in.value("bounded", false) || in.value("plane", false) ||
                         (export_3d && !in.value("symmetry", json::array()).empty()))) {
    std::string error = "--export-regions exports the regions of an electromagnetic input only";
    if (in.value("bounded", false))
      error += "; the info file of a bounded input carries its partition (resolve_only)";
    if (export_3d)
      error = "--export-regions does not export a model reduced by symmetry planes (the input's symmetry): the partition would be part of the device";
    std::cerr << "axmp-mesher: " << error << "\n";
    const json failed = {{"ok", false}, {"error", error}};
    std::ofstream(in.value("info", std::string("mesh_info.json"))) << failed.dump(2, ' ', false, json::error_handler_t::replace);
    return 1;
  }
  // A curved 3D mesh is made by the curved-mesh process (run_curved_process), unless this is it. If that process
  // cannot be started, the mesher stops: a curved mesh made here would depend on the output path again.
  // AXMP_MESHER_HO_ISOLATE=0 is the way to mesh in one process.
  g_export_regions = export_regions;
  if (!export_regions && curved_isolated(in)) {
#ifdef __linux__
    if (curved_stage() && std::string(argv[1]) == kCurvedInput) {
      g_canonical = "on";
    } else {
      return mesh_in_curved_process(in);
    }
#else
    g_canonical = "off: Linux only";
#endif
  }
  json info;
  try {
    if (g_canonical == "on") {
      // gmsh's -noenv: no setenv() of PATH and PYTHONPATH at startup (whose array grows with the environment). The
      // API's option parser does not know -noenv and would warn, so "-v 0" comes first and the verbosity is reset.
      static char a0[] = "axmp-mesher";
      static char a1[] = "-v";
      static char a2[] = "0";
      static char a3[] = "-noenv";
      static char* gmsh_argv[] = {a0, a1, a2, a3, nullptr};
      gmsh::initialize(4, gmsh_argv, false);
      gmsh::option::setNumber("General.Verbosity", 5);
    } else {
      gmsh::initialize(0, nullptr, false);
    }
    gmsh::option::setNumber("General.Terminal", in.value("verbose", 0));
    gmsh::option::setNumber("General.NumThreads", 1);
    gmsh::model::add("axmp");
    if (in.value("bounded", false)) {  // the regions of a model document, no air
      mesh_bounded_2d(in, info);
      gmsh::finalize();
      std::ofstream(in.value("info", std::string("mesh_info.json")))
          << info.dump(2, ' ', false, json::error_handler_t::replace);
      return 0;
    }
    if (in.value("plane", false)) {  // (x, y) cross-section of a z-uniform cavity
      mesh_plane(in, info);
      gmsh::finalize();
      std::ofstream(in.value("info", std::string("mesh_info.json")))
          << info.dump(2, ' ', false, json::error_handler_t::replace);
      return 0;
    }
    if (in.value("dim", 2) == 3) {
      mesh3d(in, info);
      gmsh::finalize();
      std::ofstream(in.value("info", std::string("mesh_info.json")))
          << info.dump(2, ' ', false, json::error_handler_t::replace);
      return 0;
    }

    const double Rb = in.at("domain_radius");
    const double z0 = in.value("domain_center_z", 0.0);  // centre of the outer semicircle on the axis
    const int axis_attr = in.value("axis_attr", 900), outer_attr = in.value("outer_attr", 901);
    const int air_attr = in.value("air_attr", 1);

    // Half disk r >= 0, centred at (0, z0).
    int disk = occ::addDisk(0, z0, 0, Rb, Rb);
    int box = occ::addRectangle(0, z0 - Rb, 0, Rb, 2 * Rb);
    DimTags half;
    std::vector<DimTags> hm;
    occ::intersect({{2, disk}}, {{2, box}}, half, hm);

    // Inputs: [half, region shapes..., curves...]; remember which input slots
    // belong to which item.
    DimTags objects = half;
    std::vector<int> owner;  // owner[k] = item index for objects[k]; -1 domain
    owner.assign(objects.size(), -1);
    const json& regions = in.value("regions", json::array());
    const json& curves = in.value("curves", json::array());
    for (size_t i = 0; i < regions.size(); ++i) {
      DimTags s = build_shape(regions[i].at("shape"));
      for (auto& e : s) {
        objects.push_back(e);
        owner.push_back(static_cast<int>(i));
      }
    }
    const int nreg = static_cast<int>(regions.size());
    for (size_t i = 0; i < curves.size(); ++i) {
      DimTags s = build_path(curves[i].at("path"));
      for (auto& e : s) {
        objects.push_back(e);
        owner.push_back(nreg + static_cast<int>(i));
      }
    }

    DimTags frag;
    std::vector<DimTags> fmap;
    occ::fragment(objects, {}, frag, fmap);
    occ::synchronize();

    // Owner of each resulting surface: highest-priority region containing it.
    std::map<int, int> surf_owner;  // surface tag -> region index or -1 (air)
    std::set<int> in_domain;
    std::map<int, std::set<int>> curve_items;  // curve tag -> curve item indices
    for (size_t k = 0; k < fmap.size(); ++k) {
      for (auto& e : fmap[k]) {
        if (e.first == 2) {
          if (owner[k] < 0) in_domain.insert(e.second);
          if (owner[k] >= 0 && owner[k] < nreg) {
            auto it = surf_owner.find(e.second);
            if (it == surf_owner.end() || it->second < owner[k]) surf_owner[e.second] = owner[k];
          } else if (!surf_owner.count(e.second)) {
            surf_owner[e.second] = -1;
          }
        } else if (e.first == 1 && owner[k] >= nreg) {
          curve_items[e.second].insert(owner[k] - nreg);
        }
      }
    }
    DimTags all2;
    gmsh::model::getEntities(all2, 2);
    DimTags drop;
    for (auto& e : all2) {
      if (!in_domain.count(e.second)) {
        double cx, cy, cz;
        occ::getCenterOfMass(2, e.second, cx, cy, cz);
        if (cx < 0) {  // piece at r < 0 (e.g. a disk centred on the axis): clip
          drop.push_back(e);
          surf_owner.erase(e.second);
          continue;
        }
        throw Fail("a region extends outside the computational domain (radius " + std::to_string(Rb) +
                   "); increase mesh.domain_radius or check the geometry");
      }
      if (!surf_owner.count(e.second)) surf_owner[e.second] = -1;
    }
    if (!drop.empty()) {
      occ::remove(drop, true);
      occ::synchronize();
      gmsh::model::getEntities(all2, 2);
    }

    // Physical groups for regions.
    std::map<int, std::vector<int>> by_attr;
    std::map<int, std::string> attr_name;
    for (auto& [tag, ow] : surf_owner) {
      int attr = ow < 0 ? air_attr : regions[ow].at("attr").get<int>();
      by_attr[attr].push_back(tag);
      attr_name[attr] = ow < 0 ? "air" : regions[ow].at("name").get<std::string>();
    }
    json groups = json::array();
    for (auto& [attr, tags] : by_attr) {
      add_physical_group(2, tags, attr, attr_name[attr]);
      groups.push_back({{"dim", 2}, {"attr", attr}, {"name", attr_name[attr]}, {"entities", tags.size()}});
    }
    for (size_t i = 0; i < regions.size(); ++i) {
      int attr = regions[i].at("attr");
      if (!by_attr.count(attr)) {
        throw Fail("region '" + regions[i].at("name").get<std::string>() +
                   "' is completely covered by later regions; reorder or remove it");
      }
    }

    // Boundary of the domain: axis (r ~ 0) and outer arc.
    DimTags bnd;
    gmsh::model::getBoundary(all2, bnd, true, false, false);
    std::vector<int> axis, outer;
    for (auto& e : bnd) {
      // classify by points on the curve: OCC bounding boxes are inflated by the geometric tolerance, so a
      // bounding-box test never recognizes the axis (it would be tagged "outer")
      const int tag = std::abs(e.second);
      std::vector<double> pmin, pmax;
      gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
      double rmax = 0;
      for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        std::vector<double> xyz;
        gmsh::model::getValue(1, tag, {pmin[0] + t * (pmax[0] - pmin[0])}, xyz);
        rmax = std::max(rmax, std::fabs(xyz[0]));
      }
      if (rmax < 1e-9 * Rb) axis.push_back(tag);
      else outer.push_back(tag);
    }
    gmsh::model::addPhysicalGroup(1, axis, axis_attr, "axis");
    gmsh::model::addPhysicalGroup(1, outer, outer_attr, "outer");
    groups.push_back({{"dim", 1}, {"attr", axis_attr}, {"name", "axis"}});
    groups.push_back({{"dim", 1}, {"attr", outer_attr}, {"name", "outer"}});
    for (size_t i = 0; i < curves.size(); ++i) {
      std::vector<int> tags;
      for (auto& [tag, items] : curve_items) {
        if (items.count(static_cast<int>(i))) tags.push_back(tag);
      }
      int attr = curves[i].at("attr");
      add_physical_group(1, tags, attr, curves[i].at("name").get<std::string>());
      groups.push_back({{"dim", 1}, {"attr", attr}, {"name", curves[i].at("name")}, {"entities", tags.size()}});
    }

    if (export_regions) {
      info = export_partition(in, owner, fmap, surf_owner, t0);
      gmsh::finalize();
      std::ofstream(in.value("info", std::string("mesh_info.json")))
          << info.dump(2, ' ', false, json::error_handler_t::replace);
      return 0;
    }

    // ------------------------------------------------------------ mesh size
    const double h_dev = in.at("size");
    const double h_s = in.value("surface_size", h_dev);
    const double h_far = in.value("far_size", Rb / 8);
    const double r_dev = in.value("device_radius", Rb / 4);
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", in.value("curvature_points", 24));
    gmsh::option::setNumber("Mesh.MeshSizeMax", h_far);
    double floor_2d = 0;
    {  // the floor must not clip the smallest requested size (per-region/curve sizes included)
      double hmin = std::min(h_s, h_dev);
      for (auto& g : regions) {
        for (const char* key : {"size", "surface_size"}) {
          const double v = g.value(key, 0.0);
          if (v > 0) hmin = std::min(hmin, v);
        }
      }
      for (auto& c : curves) {
        for (const char* key : {"size", "surface_size"}) {
          const double v = c.value(key, 0.0);
          if (v > 0) hmin = std::min(hmin, v);
        }
      }
      floor_2d = hmin * 0.25;
      gmsh::option::setNumber("Mesh.MeshSizeMin", floor_2d);
    }
    std::vector<double> fields;
    // far-field growth from the device, measured from the domain centre (0, z0)
    int ff = gmsh::model::mesh::field::add("MathEval");
    char expr[320];
    std::snprintf(expr, sizeof expr, "max(%.17g, %.17g + 0.25*(sqrt(x^2+(y-(%.17g))^2) - %.17g))", h_dev, h_dev, z0,
                  r_dev);
    gmsh::model::mesh::field::setString(ff, "F", expr);
    fields.push_back(ff);
    // per-region sizes
    for (size_t i = 0; i < regions.size(); ++i) {
      double h = regions[i].value("size", 0.0);
      if (h <= 0) continue;
      int attr = regions[i].at("attr");
      std::vector<double> tags;
      for (int t : by_attr[attr]) tags.push_back(t);
      int fc = gmsh::model::mesh::field::add("MathEval");
      std::snprintf(expr, sizeof expr, "%.17g", h);
      gmsh::model::mesh::field::setString(fc, "F", expr);
      int fr = gmsh::model::mesh::field::add("Restrict");
      gmsh::model::mesh::field::setNumber(fr, "InField", fc);
      gmsh::model::mesh::field::setNumbers(fr, "SurfacesList", tags);
      fields.push_back(fr);
    }
    // Distance + Threshold refinement towards a set of curves with a given target size.
    const double layer = in.value("surface_layer", 6.0 * h_dev);
    auto refine_towards = [&](const std::vector<double>& cs, double hmin, double hmax) {
      if (cs.empty() || hmin <= 0) return;
      int fd = gmsh::model::mesh::field::add("Distance");
      gmsh::model::mesh::field::setNumbers(fd, "CurvesList", cs);
      gmsh::model::mesh::field::setNumber(fd, "Sampling", 200);
      int ft = gmsh::model::mesh::field::add("Threshold");
      gmsh::model::mesh::field::setNumber(ft, "InField", fd);
      gmsh::model::mesh::field::setNumber(ft, "SizeMin", hmin);
      gmsh::model::mesh::field::setNumber(ft, "SizeMax", std::max(hmax, hmin));
      gmsh::model::mesh::field::setNumber(ft, "DistMin", 0.5 * hmin);
      gmsh::model::mesh::field::setNumber(ft, "DistMax", std::max(layer, 4.0 * hmin));
      fields.push_back(ft);
    };
    // per-region surface_size: its own boundary refined to its own size
    for (size_t i = 0; i < regions.size(); ++i) {
      const double hr = regions[i].value("surface_size", 0.0);
      if (hr <= 0) continue;
      DimTags sd, cb;
      for (int t : by_attr[regions[i].at("attr").get<int>()]) sd.push_back({2, t});
      gmsh::model::getBoundary(sd, cb, false, false, false);
      std::vector<double> cs;
      for (auto& e : cb) cs.push_back(std::abs(e.second));
      // SizeMax must not undercut the device size: a Threshold field returns SizeMax at every
      // point beyond DistMax, and this one is not Restricted to the region
      refine_towards(cs, hr, std::max(h_dev, hr));
    }
    // refinement towards boundaries of selected regions and curves without their own surface_size
    std::vector<double> refine_curves;
    const std::vector<std::string> refine_list = in.value("refine_regions", std::vector<std::string>{});
    std::set<std::string> refine(refine_list.begin(), refine_list.end());
    for (auto& [attr, tags] : by_attr) {
      if (!refine.count(attr_name[attr])) continue;
      DimTags sd, cb;
      for (int t : tags) sd.push_back({2, t});
      gmsh::model::getBoundary(sd, cb, false, false, false);
      for (auto& e : cb) refine_curves.push_back(std::abs(e.second));
    }
    // per-curve surface_size (sheets); the rest follow the global surface_size
    std::map<int, double> curve_hs;  // curve item index -> its own surface size
    for (size_t i = 0; i < curves.size(); ++i) {
      const double hc = curves[i].value("surface_size", 0.0);
      if (hc > 0) curve_hs[static_cast<int>(i)] = hc;
    }
    std::map<int, std::vector<double>> own_curves;  // item index -> OCC curve tags
    for (auto& [tag, items] : curve_items) {
      int own = -1;
      for (int it : items) if (curve_hs.count(it)) { own = it; break; }
      if (own >= 0) own_curves[own].push_back(tag);
      else refine_curves.push_back(tag);
    }
    for (auto& [item, cs] : own_curves) refine_towards(cs, curve_hs[item], h_dev);
    refine_towards(refine_curves, h_s, h_dev);
    {
      // adaptive cycles: the floor of cycle 0 (per-region and curve sizes included), lowered to the smallest adapted size
      double hmin = floor_2d;
      const int fa = size_field_from_file(in, &hmin);
      if (fa >= 0) {
        fields.push_back(fa);
        gmsh::option::setNumber("Mesh.MeshSizeMin", hmin);
      }
    }
    int fmin = gmsh::model::mesh::field::add("Min");
    gmsh::model::mesh::field::setNumbers(fmin, "FieldsList", fields);
    gmsh::model::mesh::field::setAsBackgroundMesh(fmin);

    gmsh::option::setNumber("Mesh.Algorithm", 6);
    gmsh::model::mesh::generate(2);
    int order = in.value("order", 2);
    if (order > 1) {
      gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
      gmsh::model::mesh::setOrder(std::min(order, 2));
    }
    const json high_order = order > 1 ? untangle_2d() : json();
    // Orient every surface mesh counter-clockwise in (r, z).
    for (auto& e : all2) {
      std::vector<int> et;
      std::vector<std::vector<std::size_t>> etags, enodes;
      gmsh::model::mesh::getElements(et, etags, enodes, 2, e.second);
      if (et.empty() || enodes[0].size() < 3) continue;
      std::vector<double> c[3], pc;
      int dimn, tagn;
      for (int q = 0; q < 3; ++q) gmsh::model::mesh::getNode(enodes[0][q], c[q], pc, dimn, tagn);
      double area = (c[1][0] - c[0][0]) * (c[2][1] - c[0][1]) - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1]);
      if (area < 0) gmsh::model::mesh::reverse({{2, e.second}});
    }

    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::option::setNumber("Mesh.Binary", 0);
    gmsh::write(in.at("output").get<std::string>());

    std::vector<std::size_t> ntags;
    std::vector<double> ncoord, npar;
    gmsh::model::mesh::getNodes(ntags, ncoord, npar);
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 2);
    size_t nel = 0;
    for (auto& v : etags) nel += v.size();
    std::string ver;
    gmsh::option::getString("General.Version", ver);
    info = {{"ok", true},
            {"gmsh_version", ver},
            {"nodes", ntags.size()},
            {"elements_2d", nel},
            {"groups", groups},
            {"domain_radius", Rb},
            {"domain_center_z", z0},
            {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
    if (!high_order.is_null()) info["high_order"] = high_order;
    gmsh::finalize();
  }
#ifdef __linux__
  catch (const RepairInCurvedProcess&) {
    // a 2D mesh with a tangled element to repair: made again, repaired and written by the curved-mesh process
    gmsh::finalize();
    json flagged = in;
    flagged[kTangledRepair] = true;
    return mesh_in_curved_process(flagged);
  }
#endif
  catch (std::exception& e) {
    std::string error = e.what();
#ifdef __linux__
    if (g_canonical == "on") error = user_paths(error);  // the user's file names, not the work directory's
#endif
    info = {{"ok", false}, {"error", error}};
    if (!g_refused_high_order.is_null()) info["high_order"] = g_refused_high_order;  // a refused 2D mesh's record
    std::cerr << "axmp-mesher: " << error << "\n";
  }
  std::ofstream(in.value("info", std::string("mesh_info.json")))
      << info.dump(2, ' ', false, json::error_handler_t::replace);
  return info.value("ok", false) ? 0 : 1;
}
