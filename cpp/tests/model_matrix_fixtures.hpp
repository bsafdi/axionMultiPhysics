// The meshes of the engine half of the matrix test and the reference instance numbering they are numbered by.
//
// Serial mfem::Mesh objects only: this header uses no library of the fluids module, no partitioned mesh and no MPI
// call, so that a test built in every build can include it (tests/test_packaging.py keeps the sequential build free of
// MPI tokens outside the 3D engine and the fluids module).  The quadrilateral mesh and the partitioned meshes are built
// in cpp/tests/test_fluids_model_matrix.cpp, which links the fluids library.
//
// INDEPENDENT REFERENCE.  instance_numbering() is written from section 5.4 of the interface design note ("The atoms
// present on the instance mesh's boundary elements are sorted by master attribute (so axis, 900, comes first where it
// exists) and numbered 1, 2, ..., n with no hole"), and the master numbers of the fixtures from its section 5.3 ("axis
// keeps 900; atoms get 2001, 2002, ... in the lexicographic order of their sorted name lists").  It shares no code with
// the library's own instance mesh, which is compared with it.
//
// The fixtures are the unit square of the matrix documents (cpp/src/model_matrix.cpp, minimal_document): planar, the
// sides left (x = 0), right (x = 1), bottom (y = 0) and top (y = 1); axisymmetric, the same square in the (r, z) half
// plane with the side r = 0 on the axis and the sides right (r = 1), bottom (z = 0) and top (z = 1).  Each side is one
// named boundary of one entity, hence one atom.
//
//   base_mesh    triangles; boundary attributes renumbered from the master numbers by instance_numbering(), so
//                exactly 1..n; no nodes grid function (GetNodes() == nullptr), as an instance mesh of geometry order 1
//   sparse_mesh  the same triangles with the master numbers left on (2001 to 2004 planar; 900, 2001 to 2003
//                axisymmetric): the numbering the dense renumbering exists to remove
//   nodes_mesh   the base mesh after SetCurvature(2): the same straight triangles and attributes, with a nodes grid
//                function (what the transport operator reads as "curved")
//
// The triangulation cuts each cell of an n x n grid along the diagonal that points away from the nearest corner of the
// square, so that no triangle has its three vertices on the boundary (the Taylor-Hood mesh condition that the document
// path checks before a flow solve).
#pragma once

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace axmp::model::fixtures {

// ---------------------------------------------------------------------------------------------- the instance numbering
// The instance attribute of every master attribute present on the boundary of an instance mesh: the distinct master
// attributes sorted ascending and numbered 1, 2, ..., n.  The input may repeat attributes and come in any order (it is
// the attribute of every boundary element); a master attribute below 1 is not one.
inline std::map<int, int> instance_numbering(const std::vector<int>& master_attributes) {
  std::vector<int> present(master_attributes);
  for (int a : present)
    if (a < 1) throw std::invalid_argument("instance_numbering: a master attribute is a positive integer");
  std::sort(present.begin(), present.end());
  present.erase(std::unique(present.begin(), present.end()), present.end());
  std::map<int, int> out;
  int next = 1;
  for (int a : present) out[a] = next++;
  return out;
}

// --------------------------------------------------------------------------------------------------------- the sides
enum class Geometry { planar, axisymmetric };

inline const char* to_string(Geometry g) { return g == Geometry::planar ? "planar" : "axisymmetric"; }

// The master number of the axis (section 5.3).
constexpr int kAxisMaster = 900;
// The first master number of a named atom (section 5.3).
constexpr int kFirstAtomMaster = 2001;
// Cells per side of the square.
constexpr int kCells = 4;

// The boundary names of the square, as the matrix documents write them; "axis" is the side r = 0, which a document
// never names.
inline std::vector<std::string> side_names(Geometry g) {
  if (g == Geometry::planar) return {"left", "right", "bottom", "top"};
  return {"axis", "right", "bottom", "top"};
}

// The master number of every side: the axis 900, the named atoms 2001, 2002, ... in the lexicographic order of their
// names (one name each).
inline std::map<std::string, int> master_numbers(Geometry g) {
  std::vector<std::string> named;
  for (const auto& s : side_names(g))
    if (s != "axis") named.push_back(s);
  std::sort(named.begin(), named.end());
  std::map<std::string, int> out;
  int next = kFirstAtomMaster;
  for (const auto& s : named) out[s] = next++;
  if (g == Geometry::axisymmetric) out["axis"] = kAxisMaster;
  return out;
}

// The side of the square a boundary segment from p to q lies on, or "" when it lies on none.
inline std::string side_of_segment(Geometry g, const double* p, const double* q) {
  const double tol = 1e-12;
  auto on = [&](int coord, double value) {
    return std::fabs(p[coord] - value) <= tol && std::fabs(q[coord] - value) <= tol;
  };
  if (on(0, 0.0)) return g == Geometry::planar ? "left" : "axis";
  if (on(0, 1.0)) return "right";
  if (on(1, 0.0)) return "bottom";
  if (on(1, 1.0)) return "top";
  return "";
}

// ------------------------------------------------------------------------------------------------------ the fixtures
struct Fixture {
  std::string name;                        // base, sparse, nodes (quad: in the fluids test)
  Geometry geometry = Geometry::planar;
  mfem::Mesh mesh;                         // serial
  std::map<std::string, int> master;       // side -> master number (section 5.3)
  std::map<std::string, int> attribute;    // side -> the boundary attribute it carries on this mesh

  // The attribute of a side of this mesh; an unknown side is a defect of the test.
  int attr(const std::string& side) const {
    auto it = attribute.find(side);
    if (it == attribute.end()) throw std::logic_error("fixture " + name + ": no side '" + side + "'");
    return it->second;
  }
  // The attributes of several sides.
  std::vector<int> attrs(const std::vector<std::string>& sides) const {
    std::vector<int> out;
    for (const auto& s : sides) out.push_back(attr(s));
    return out;
  }
};

namespace detail {

// The vertices of the (n + 1) x (n + 1) grid of the unit square, x fastest.
inline int vertex(int n, int i, int j) { return j * (n + 1) + i; }

// The square meshed with the master numbers on its boundary: triangles, or (quads = true) the cells themselves.
// Returned as arrays so that the quadrilateral fixture of the fluids test is built from the same grid.
struct Arrays {
  std::vector<double> xy;          // vertex coordinates, interleaved
  std::vector<int> elements;       // 3 or 4 vertices per element, counter-clockwise
  std::vector<int> bdr_segments;   // 2 vertices per segment
  std::vector<int> bdr_attributes; // the master number of the side of each segment
};

inline Arrays square_arrays(Geometry g, int n, bool quads) {
  if (n < 2) throw std::invalid_argument("fixtures: at least 2 cells per side");
  Arrays a;
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n; ++i) {
      a.xy.push_back(static_cast<double>(i) / n);
      a.xy.push_back(static_cast<double>(j) / n);
    }
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const int v00 = vertex(n, i, j), v10 = vertex(n, i + 1, j), v11 = vertex(n, i + 1, j + 1),
                v01 = vertex(n, i, j + 1);
      if (quads) {
        a.elements.insert(a.elements.end(), {v00, v10, v11, v01});
        continue;
      }
      // the diagonal points away from the nearest corner: through (i, j)-(i+1, j+1) in the lower-left and upper-right
      // quarters, through (i+1, j)-(i, j+1) in the two others, so a corner cell keeps an interior vertex in both
      // of its triangles
      const bool lower = 2 * j + 1 < n, left = 2 * i + 1 < n;
      if (lower == left) {
        a.elements.insert(a.elements.end(), {v00, v10, v11});
        a.elements.insert(a.elements.end(), {v00, v11, v01});
      } else {
        a.elements.insert(a.elements.end(), {v00, v10, v01});
        a.elements.insert(a.elements.end(), {v10, v11, v01});
      }
    }
  const std::map<std::string, int> master = master_numbers(g);
  auto segment = [&](int p, int q) {
    const std::string side = side_of_segment(g, &a.xy[2 * p], &a.xy[2 * q]);
    a.bdr_segments.push_back(p);
    a.bdr_segments.push_back(q);
    a.bdr_attributes.push_back(master.at(side));
  };
  for (int i = 0; i < n; ++i) segment(vertex(n, i, 0), vertex(n, i + 1, 0));          // bottom
  for (int j = 0; j < n; ++j) segment(vertex(n, n, j), vertex(n, n, j + 1));          // right
  for (int i = n; i > 0; --i) segment(vertex(n, i, n), vertex(n, i - 1, n));          // top
  for (int j = n; j > 0; --j) segment(vertex(n, 0, j), vertex(n, 0, j - 1));          // left or axis
  return a;
}

inline mfem::Mesh triangle_mesh(const Arrays& a) {
  const int nv = static_cast<int>(a.xy.size() / 2), ne = static_cast<int>(a.elements.size() / 3),
            nb = static_cast<int>(a.bdr_attributes.size());
  mfem::Mesh m(2, nv, ne, nb, 2);
  for (int v = 0; v < nv; ++v) m.AddVertex(a.xy[2 * v], a.xy[2 * v + 1]);
  for (int e = 0; e < ne; ++e) m.AddTriangle(a.elements[3 * e], a.elements[3 * e + 1], a.elements[3 * e + 2], 1);
  for (int b = 0; b < nb; ++b) m.AddBdrSegment(a.bdr_segments[2 * b], a.bdr_segments[2 * b + 1], a.bdr_attributes[b]);
  m.FinalizeTriMesh(1, 0, true);
  return m;
}

// Every boundary attribute of `m` replaced through `map` (master -> instance).
inline void renumber_boundary(mfem::Mesh& m, const std::map<int, int>& map) {
  for (int be = 0; be < m.GetNBE(); ++be) m.SetBdrAttribute(be, map.at(m.GetBdrAttribute(be)));
  m.SetAttributes();
}

}  // namespace detail

// The instance numbering of the square's master numbers.
inline std::map<int, int> square_numbering(Geometry g) {
  std::vector<int> masters;
  for (const auto& [side, m] : master_numbers(g)) masters.push_back(m);
  return instance_numbering(masters);
}

// The side -> attribute maps of the square: with the master numbers, and renumbered densely.
inline std::map<std::string, int> sparse_attributes(Geometry g) { return master_numbers(g); }
inline std::map<std::string, int> dense_attributes(Geometry g) {
  const auto map = square_numbering(g);
  std::map<std::string, int> out;
  for (const auto& [side, m] : master_numbers(g)) out[side] = map.at(m);
  return out;
}

inline Fixture sparse_mesh(Geometry g, int n = kCells) {
  Fixture f;
  f.name = "sparse";
  f.geometry = g;
  f.mesh = detail::triangle_mesh(detail::square_arrays(g, n, false));
  f.master = master_numbers(g);
  f.attribute = sparse_attributes(g);
  return f;
}

inline Fixture base_mesh(Geometry g, int n = kCells) {
  Fixture f = sparse_mesh(g, n);
  f.name = "base";
  detail::renumber_boundary(f.mesh, square_numbering(g));
  f.attribute = dense_attributes(g);
  return f;
}

inline Fixture nodes_mesh(Geometry g, int n = kCells) {
  Fixture f = base_mesh(g, n);
  f.name = "nodes";
  f.mesh.SetCurvature(2);
  return f;
}

}  // namespace axmp::model::fixtures
