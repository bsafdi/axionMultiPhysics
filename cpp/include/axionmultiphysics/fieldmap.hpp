// Field maps for particle transport (beta): the numerics behind axionmultiphysics.fieldmap.
//
// A map is one patch per region (or per set of adjacent regions across which nothing jumps), each a regular grid
// holding the region's recovered static field B* (B_h averaged at the nodes of a continuous H1 space of the solution's
// order over the patch's own elements). Nodes outside the patch take the nearest patch element's polynomial, never a
// neighbour's value. The error of a grid is measured on fixed chords: E = RMS over chords of
// |int (B_map - B*) dl| / max(int |B*| dl, 0.1 L_c B_patch), and the indicator H the same with B* - B_h.
// docs/guides/field_maps.md is the user's description; the Python module orchestrates the refinement loop and writes
// the directory format.
//
// Everything here is deterministic and independent of the number of OpenMP threads: work is split per chord and per
// grid node, every result goes to its own slot, and every sum runs in a fixed order.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace axmp {

class AxiModel;

namespace fieldmap {

using json = nlohmann::json;

// ------------------------------------------------------------------------------------------- scrambled Halton
// The sequence of the map format: bases 2, 3, 5, 7, 11, 13; per dimension and digit a permutation of the digits drawn
// by a Fisher-Yates shuffle from one splitmix64 stream of the seed; point k (k >= 1) is
// sum_j perm[d][j][digit_j(k)] b^-(j+1), accumulated as v += p * f, f /= b, over every digit of the table, then
// min(v, 1 - 2^-53). The arithmetic is plain double precision in that order (this file is compiled without
// floating-point contraction), so the points are bitwise the same on every platform.
class Halton {
 public:
  explicit Halton(uint64_t seed, int dims = 6);
  void point(uint64_t k, double* u) const;  // point k >= 1, dims coordinates in [0, 1)
  int dims() const { return dims_; }

 private:
  int dims_;
  std::vector<std::vector<std::vector<int>>> perm_;  // [dimension][digit][digit value]
};
std::vector<double> halton(uint64_t seed, long n, int dims);  // points 1..n, row-major (n, dims)
uint64_t family_seed(uint64_t seed, uint64_t index);           // seed ^ (index * 0x9E3779B97F4A7C15) mod 2^64

// ------------------------------------------------------------------------------------------- polynomial bases
// Lagrange basis of the complete polynomials of degree p on a reference simplex, built from the node positions of an
// MFEM element (its nodal H1 basis is this one). Evaluation is thread-safe, unlike MFEM's CalcShape.
class Lagrange {
 public:
  Lagrange() = default;
  Lagrange(int dim, const std::vector<double>& nodes);  // nodes: n x dim reference coordinates
  int n = 0, dim = 0, degree = 0;
  void eval(const double* xi, double* N, double* dN = nullptr) const;  // dN: n x dim (d N_i / d xi_a at i*dim+a)

 private:
  std::vector<int> exps_;  // n x dim exponents of the monomials
  std::vector<double> C_;  // N_i = sum_j C_[j*n+i] m_j
};

// ------------------------------------------------------------------------------------------- a solved field
// The static field of a solved design, element by element: what the builder needs from an engine.
struct RegionLaw {
  enum Kind { Linear, Nonlinear, Superconductor, Excluded } kind = Linear;
  double mu_r = 1.0;         // Linear
  std::string material;      // Nonlinear: the material's name
};

class Source {
 public:
  virtual ~Source() = default;
  int dim = 2;       // 2: axisymmetric (r, z); 3: Cartesian
  int order = 1;     // order of the solution, and of the recovery space
  int ne = 0;
  std::vector<int> attr;                        // element attribute
  std::map<int, std::string> region_of_attr;    // the implicit background region is "air"
  std::map<std::string, RegionLaw> law;         // static constitutive law per region
  std::map<int, std::string> curve_of_attr;     // face attribute of each curve (axisymmetric)
  int axis_attr = -1;                           // face attribute of the symmetry axis (axisymmetric)
  // geometry: ng nodes per element (vertices first), their reference coordinates, element areas or volumes
  int ng = 0;
  std::vector<double> gref, gx, vol;            // gref: ng x dim; gx: ne x ng x dim
  // topology: for element e and local face j (opposite vertex j), the neighbour (-1: none) and the face attribute
  std::vector<int> nbr, fattr;
  // the recovery space (continuous H1 of the solution's order)
  int nd = 0, ndofs = 0;
  std::vector<double> nref;                     // nd x dim reference coordinates of its nodes
  std::vector<int> dofs;                        // ne x nd global node of each element node
  std::vector<double> dof_x;                    // ndofs x dim physical coordinates (axis nodes at r = 0 exactly)
  double rtol = 0;                              // |r| below this is on the axis
  long solution_dofs = 0;
  // B_h at reference point xi of element e, physical point x and Jacobian J (J[a*dim+b] = d x_a / d xi_b);
  // native components ((B_r, B_z, B_phi) or (B_x, B_y, B_z)). Thread-safe.
  virtual void Bh(int e, const double* xi, const double* x, const double* J, double B[3]) const = 0;
  // the static solution's degrees of freedom (for the manifest's hash)
  virtual std::vector<double> solution() const = 0;
};

std::unique_ptr<Source> axi_source(const AxiModel& m);

// ------------------------------------------------------------------------------------------- the builder
class Mapper;

// A grid: origin, spacing and node count per axis (node i at origin + i * spacing).
struct Grid {
  int dim = 2;
  double origin[3] = {0, 0, 0}, spacing[3] = {1, 1, 1};
  long n[3] = {1, 1, 1};
  long nodes() const { return n[0] * n[1] * (dim == 3 ? n[2] : 1); }
  void interp(const double* B, const double* x, double out[3]) const;  // multilinear, cell clipped to the grid
};

// One family of chords of a patch: start points, directions, segments in the patch, and the integrals of B*, |B*| and
// B_h along them (fixed for every grid level).
struct Chords {
  std::string family;
  int dim = 2;
  long attempts = 0;
  std::vector<double> start, direction, length, I_star, I_abs, I_h;  // n x dim, n x dim, n, n x 3, n, n x 3
  std::vector<double> segments;                                      // m x 2 (s0, s1)
  std::vector<int64_t> segment_chord;                                // m
  long size() const { return static_cast<long>(length.size()); }
  // exact integrals of a multilinear map along the segments: pieces between grid-line crossings, 2-point
  // Gauss-Legendre (exact for a polynomial of degree 3 along a line)
  std::vector<double> map_integrals(const Grid& g, const double* B) const;
};

// The statistic of a family against a map: E, its 95 % quantile and maximum, and H.
struct Statistic {
  long n = 0;
  double E = 0, E_q95 = 0, E_max = 0, H = 0;
};
Statistic statistic(const Chords& c, const std::vector<double>& I_map, double B_patch);

// A patch: its elements (one component of a set of regions), recovered field and geometry.
class Patch {
 public:
  const Mapper& M;
  std::vector<std::string> regions;
  int component = 0;
  std::vector<int> elems;
  std::vector<char> in;          // per mesh element
  double h_el = 0, B_patch = 0;
  double box[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  bool touches_axis = false, has_jump = false;
  std::vector<double> bstar;     // ne x nd x 3 (patch elements only)

  Patch(const Mapper& m, std::vector<std::string> regions, int component, std::vector<int> elems);
  ~Patch();
  json info() const;
  // B* in element e at reference point xi (r = 0 handled by the caller)
  void eval_star(int e, const double* xi, double B[3]) const;
  // lowest-index patch element containing x (reference coordinates within 1e-12), its reference point; -1 if none
  int locate(const double* x, double* xi) const;
  // the recovered field at points (n x dim), NaN where no patch element contains a point
  std::vector<double> recovered(const std::vector<double>& pts) const;
  // the grid's nodal values (n x 3): B* at nodes in or on the patch, the nearest element's polynomial at active nodes
  // outside it, 0 at inactive nodes
  std::vector<double> fill(const Grid& g) const;
  // chords of a family: "bulk", "layer", "axis" or "track" (with a box: lo, hi per axis)
  Chords chords(const std::string& family, long N, uint64_t seed, const std::vector<double>& box) const;
  // whether a box (lo, hi per axis) meets the bounding box of a patch element
  bool box_meets(const std::vector<double>& box) const;
  // pointwise report at n points by element: |B_map - B*| / B_patch and |B* - B_h| / B_patch quantiles
  json pointwise(const Grid& g, const double* B, long n, uint64_t seed) const;

  struct Impl;
  std::unique_ptr<Impl> p_;
};

class Mapper {
 public:
  explicit Mapper(std::unique_ptr<Source> src);
  ~Mapper();
  const Source& source() const { return *src_; }
  int dim() const { return src_->dim; }
  // the regions of the mesh with their laws: [{"name", "law", "mu_r"?, "material"?}]
  json regions() const;
  // The patches of a selection of regions (empty: every region but the superconducting and excluded ones):
  // merged (if `merge`) across shared faces with equal laws and no curve, split at curves into connected components.
  // Refuses an unknown or superconducting region (fieldmap.invalid_option) and a curve with the same component on both
  // sides (fieldmap.curve_in_region). Returns {"patches": [{"regions", "component", "components", "curves",
  // "elements"}], "zero_field_regions": [...]}.
  json decompose(const std::vector<std::string>& regions, bool merge) const;
  // the patch of `regions` (one group of decompose) and its component, recovered
  std::unique_ptr<Patch> patch(const std::vector<std::string>& regions, int component) const;
  // the largest |B*| over the nodal values on the outer boundary of the union of the patches (faces whose other side
  // is not a patch element, except the axis and superconducting regions)
  double boundary_max(const std::vector<const Patch*>& patches) const;
  // normal-component mismatch at the midpoints of the faces shared by two patches, relative to max(B_patch):
  // {"points", "normal_mismatch_rms", "normal_mismatch_max"}, or null when they share no face
  json interface(const Patch& a, const Grid& ga, const double* Ba, const Patch& b, const Grid& gb,
                 const double* Bb) const;

  struct Impl;
  std::unique_ptr<Impl> p_;

 private:
  std::unique_ptr<Source> src_;
};

}  // namespace fieldmap
}  // namespace axmp
