// Element quality for solver-failure messages: the worst element(s) of a mesh, where they are, in which region,
// and how to fix them in the design. Computed only when a solve breaks down or stalls (no cost to a successful
// solve). MPI-free; the 3D engine gathers the per-rank summaries (model3d_impl.hpp, worst_elements_note). Also the
// check that refuses a tangled 2D curved mesh where a mesh file is read (require_untangled).
#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mfem {
class Mesh;
}

namespace axmp::mesh_quality {

// Thresholds: a curved element is distorted at min det J <= 1e-3 max |det J| (the 3D engine's curved-element test,
// model3d.cpp curved_element_invalid); an element is flat (a sliver) at a Jacobian aspect ratio above 1e3 (the 3D
// engine's mesh warning).
inline constexpr double kDistortedDetRatio = 1e-3;
inline constexpr double kFlatAspect = 1e3;

struct ElementQuality {
  long long index = -1;              // element number (the serial mesh's in 3D MPI runs)
  int attribute = 0;                 // element attribute (region)
  std::array<double, 3> centroid{};  // image of the reference centre (m; z = 0 in 2D, where x = r and y = z)
  // largest Jacobian aspect ratio sigma_max / sigma_min over the sample points, measured against the perfect
  // element as Mesh::GetElementSize does (1 for an equilateral simplex; infinite where det J <= 0)
  double aspect = 1;
  double det_ratio = 1;              // min det J / max |det J| over the sample points (<= 0: inverted or degenerate)
  bool curved = false;               // high-order geometry
  // Affine elements (straight simplices) have a constant Jacobian, sampled at the centre. Others (curved, or
  // straight quadrilaterals, prisms, hexahedra) are sampled at the centre, the reference vertices and edge midpoints
  // (where a P2 element's det J usually has its minimum) and the order-6 quadrature points.
  bool affine = true;
  bool inverted() const { return !(det_ratio > 0); }
};

// strict weak order "a is worse than b": inverted first (most negative det ratio first), then by aspect ratio,
// then by distortion; ties by the lower element number (deterministic across ranks)
bool worse(const ElementQuality& a, const ElementQuality& b);

ElementQuality measure(mfem::Mesh& m, int e);

struct Summary {
  std::vector<ElementQuality> worst;  // worst first
  long long elements = 0;
  long long inverted = 0;             // det J <= 0 at some sample point
  long long distorted = 0;            // not affine, not inverted, min det J <= 1e-3 max |det J| (JSON
                                      // "distorted_curved": curved, or straight but not a simplex)
  long long flat = 0;                 // aspect ratio > 1e3 (inverted elements excluded)
  int dim = 3;
};

// The `keep` worst elements of a mesh. global_index maps a local element to its global (serial) number.
Summary scan(mfem::Mesh& m, int keep, const std::vector<int>* global_index = nullptr);
// Combine two summaries of disjoint element sets (counts add, the worst `keep` survive).
void merge(Summary& into, const Summary& other, int keep);

// fixed-size records for gathering across MPI ranks
inline constexpr int kRecordDoubles = 9;
void pack(const ElementQuality& q, double* rec);
ElementQuality unpack(const double* rec);

using RegionName = std::function<std::string(int attribute)>;

struct DescribeOptions {
  bool three_d = true;                     // 3D or 2D remedies
  std::string coords_2d = "(r, z)";        // 2D coordinate names: "(r, z)" axisymmetric, "(x, y)" planar
  std::string imported_file;               // the mesh was read from mesh.file: the remedy is about that file
  std::string listed_in = "the results";   // where mesh.info.worst_elements can be read; empty: nowhere (a thrown error)
};

// The text appended to a solver-failure message: the worst element (number, region, centroid, measures), the
// counts, and what to change in the design (for an imported mesh: in the tool that wrote it). Starts with a space.
std::string describe(const Summary& s, const RegionName& region, const DescribeOptions& o = {});
nlohmann::json to_json(const Summary& s, const RegionName& region);

// ---- tangled curved elements of a 2D mesh, refused where a mesh file is read
// An element of a 2D mesh with nodes of order 2 or more is tangled when its Jacobian determinant is not of one strict
// sign over the element (it changes sign or vanishes somewhere in it), or is of the sign opposite to that of the
// majority of the elements of its region (attribute; the elements of a mesh read by MFEM are oriented, so a region's
// majority is that of each of its parts). For a 6-node triangle det J is a quadratic polynomial of the reference
// coordinates, and its range over the element is exact (det_range_p2); other elements are sampled on the lattice
// (i/12, j/12) of the reference element (i + j <= 12 on a triangle), the field-map builder's. axmp-mesher checks the
// meshes it makes with the same definition. A tangled element has no inverse map and folds the solution over itself;
// it appears where a curved region is much thinner than the mesh size.

// The reference points (u, v) at which det_range_p2 takes the values of a quadratic: the vertices of the reference
// triangle, then the midpoints of its edges (0, 1), (1, 2), (2, 0).
inline constexpr double kP2Points[6][2] = {{0, 0}, {1, 0}, {0, 1}, {0.5, 0}, {0.5, 0.5}, {0, 0.5}};

// The exact minimum and maximum over the reference triangle (u, v >= 0, u + v <= 1) of the quadratic polynomial whose
// values at kP2Points are q. Its coefficients in 1, u, v, u^2, uv, v^2 follow from the six values; the extremes lie at
// the vertices, at the stationary point of an edge inside that edge, or at the interior stationary point inside the
// triangle. Header-only: axmp-mesher uses it too.
inline void det_range_p2(const double q[6], double& lo, double& hi) {
  const double a = q[0], b = 4 * q[3] - 3 * q[0] - q[1], c = 4 * q[5] - 3 * q[0] - q[2];
  const double d = 2 * (q[0] + q[1] - 2 * q[3]), f = 2 * (q[0] + q[2] - 2 * q[5]), e = 4 * (q[0] - q[3] + q[4] - q[5]);
  lo = std::min(q[0], std::min(q[1], q[2]));
  hi = std::max(q[0], std::max(q[1], q[2]));
  auto take = [&](double value) {
    lo = std::min(lo, value);
    hi = std::max(hi, value);
  };
  // A t^2 + B t + C on an edge, at its stationary point when that lies inside the edge
  auto edge = [&](double A, double B, double C) {
    if (A == 0) return;
    const double t = -B / (2 * A);
    if (t > 0 && t < 1) take((A * t + B) * t + C);
  };
  edge(d, b, a);                                  // v = 0, u = t
  edge(f, c, a);                                  // u = 0, v = t
  edge(d - e + f, c - b - 2 * d + e, a + b + d);  // u = 1 - t, v = t
  const double m = 4 * d * f - e * e;             // the interior stationary point: grad = 0
  if (m != 0) {
    const double u = (e * c - 2 * f * b) / m, v = (e * b - 2 * d * c) / m;
    if (u > 0 && v > 0 && u + v < 1) take(a + b * u + c * v + d * u * u + e * u * v + f * v * v);
  }
}

struct UntangledOptions {
  std::string where = "/mesh";             // the JSON pointer of the error
  std::string coords_2d = "(r, z)";        // the coordinate names of the location: "(r, z)" or "(x, y)"
  std::string order_one = "mesh.order 1";  // the design setting that makes the geometry order 1 (the remedy)
  bool imported = false;                   // the file is the design's mesh.file: the remedy is about that file
};

// Throws Error with the code mesh.tangled when an element of `m` is tangled: the count, the file, and the region and
// location of the first tangled element in the message, the remedy in the hint. Returns at once for a mesh without
// nodes or with nodes of order 1, and for a mesh that is not 2D in the plane (curved 3D meshes have their own check:
// the 3D solve straightens invalid curved tetrahedra). Call it on the mesh as read, before any refinement. MPI-free.
void require_untangled(mfem::Mesh& m, const RegionName& region, const std::string& file,
                       const UntangledOptions& o = {});

}  // namespace axmp::mesh_quality
