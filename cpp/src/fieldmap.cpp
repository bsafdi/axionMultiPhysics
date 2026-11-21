// Field maps for particle transport (beta): see axionmultiphysics/fieldmap.hpp and docs/guides/field_maps.md.
//
// This file is compiled without floating-point contraction (CMakeLists.txt): the scrambled-Halton points are part of
// the map format and must be bitwise the same everywhere, and the chord arithmetic then does not depend on whether
// the compiler fuses a multiply and an add.
#include "axionmultiphysics/fieldmap.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "axionmultiphysics/axi.hpp"
#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model.hpp"
#include "axionmultiphysics/omp_shim.hpp"
#include "axionmultiphysics/spec.hpp"

namespace axmp::fieldmap {

namespace {

constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;
constexpr int kBases[6] = {2, 3, 5, 7, 11, 13};
constexpr double kInside = 1e-12;   // a point is in an element when its barycentric coordinates are >= -kInside
constexpr double kFloor = 0.1;      // denominator floor, fraction of L_c B_patch
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += kGolden);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// the smallest n with b^n >= 2^53, plus one: every digit a double can resolve, and one more
int halton_digits(int b) {
  int n = 0;
  uint64_t p = 1;
  while (p < (uint64_t(1) << 53)) {
    p *= static_cast<uint64_t>(b);
    ++n;
  }
  return n + 1;
}

Error fm_error(const std::string& code, const std::string& what, const std::string& hint) {
  return Error(what, "", hint, "fieldmap." + code);
}

inline double norm3(const double* a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

// distance from p to the segment [a, b] (2D)
double seg_dist(const double* p, const double* a, const double* b) {
  const double dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
  double t = L2 > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2 : 0.0;
  t = std::clamp(t, 0.0, 1.0);
  const double qx = a[0] + t * dx - p[0], qy = a[1] + t * dy - p[1];
  return std::sqrt(qx * qx + qy * qy);
}

// distance from p to the triangle (a, b, c) (3D; Ericson, Real-Time Collision Detection, 5.1.5)
double tri_dist(const double* p, const double* a, const double* b, const double* c) {
  auto sub = [](const double* u, const double* v, double* w) {
    for (int k = 0; k < 3; ++k) w[k] = u[k] - v[k];
  };
  auto dot = [](const double* u, const double* v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; };
  double ab[3], ac[3], ap[3], q[3];
  sub(b, a, ab);
  sub(c, a, ac);
  sub(p, a, ap);
  auto at = [&](double v, double w) {
    for (int k = 0; k < 3; ++k) q[k] = a[k] + v * ab[k] + w * ac[k];
    double d[3];
    sub(p, q, d);
    return std::sqrt(dot(d, d));
  };
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return at(0, 0);
  double bp[3];
  sub(p, b, bp);
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return at(1, 0);
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return at(d1 / (d1 - d3), 0);
  double cp[3];
  sub(p, c, cp);
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return at(0, 1);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return at(0, d2 / (d2 - d6));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return at(1 - w, w);
  }
  const double den = 1.0 / (va + vb + vc);
  return at(vb * den, vc * den);
}

// Gauss-Kronrod 7-15 (QUADPACK qk15): Kronrod abscissae and weights, and the weights of the embedded Gauss rule at the
// odd abscissae (1, 3, 5) and the centre
constexpr double kXgk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                            0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                            0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                            0.207784955007898467600689403773245, 0.000000000000000000000000000000000};
constexpr double kWgk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                            0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                            0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                            0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
constexpr double kWg[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                           0.381830050505118944950369775488975, 0.417959183673469387755102040816327};

}  // namespace

// ============================================================================================ scrambled Halton
Halton::Halton(uint64_t seed, int dims) : dims_(dims) {
  AXMP_REQUIRE(dims >= 1 && dims <= 6, "internal: the scrambled-Halton sequence has 1 to 6 dimensions");
  uint64_t state = seed;
  perm_.resize(dims);
  for (int d = 0; d < dims; ++d) {
    const int b = kBases[d];
    perm_[d].resize(halton_digits(b));
    for (auto& p : perm_[d]) {
      p.resize(b);
      std::iota(p.begin(), p.end(), 0);
      for (int i = b - 1; i > 0; --i) {
        const int k = static_cast<int>(splitmix64(state) % static_cast<uint64_t>(i + 1));
        std::swap(p[i], p[k]);
      }
    }
  }
}

void Halton::point(uint64_t k, double* u) const {
  static const double one_minus = 1.0 - std::ldexp(1.0, -53);
  for (int d = 0; d < dims_; ++d) {
    const int b = kBases[d];
    uint64_t m = k;
    double f = 1.0 / b, v = 0.0;
    for (const auto& p : perm_[d]) {
      const int dig = static_cast<int>(m % static_cast<uint64_t>(b));
      m /= static_cast<uint64_t>(b);
      const double term = static_cast<double>(p[dig]) * f;
      v += term;
      f /= b;
    }
    u[d] = std::min(v, one_minus);
  }
}

std::vector<double> halton(uint64_t seed, long n, int dims) {
  Halton h(seed, dims);
  std::vector<double> out(static_cast<size_t>(std::max(0L, n)) * dims);
  for (long k = 0; k < n; ++k) h.point(static_cast<uint64_t>(k + 1), &out[static_cast<size_t>(k) * dims]);
  return out;
}

uint64_t family_seed(uint64_t seed, uint64_t index) { return seed ^ (index * kGolden); }

// ============================================================================================ Lagrange bases
Lagrange::Lagrange(int d, const std::vector<double>& nodes) : dim(d) {
  n = static_cast<int>(nodes.size()) / d;
  // the degree whose complete polynomial space has n members
  for (degree = 0;; ++degree) {
    long cnt = 1;
    for (int a = 1; a <= d; ++a) cnt = cnt * (degree + a) / a;
    if (cnt == n) break;
    AXMP_REQUIRE(cnt < n, strcat("internal: ", n, " nodes do not span a complete polynomial space in ", d, "D"));
  }
  for (int a = 0; a <= degree; ++a)
    for (int b = 0; a + b <= degree; ++b) {
      if (d == 2) {
        exps_.insert(exps_.end(), {a, b});
      } else {
        for (int c = 0; a + b + c <= degree; ++c) exps_.insert(exps_.end(), {a, b, c});
      }
    }
  mfem::DenseMatrix V(n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double m = 1;
      for (int a = 0; a < d; ++a) m *= std::pow(nodes[i * d + a], exps_[j * d + a]);
      V(i, j) = m;
    }
  mfem::DenseMatrixInverse inv(V);
  mfem::DenseMatrix Ci(n);
  inv.GetInverseMatrix(Ci);
  C_.resize(static_cast<size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) C_[j * n + i] = Ci(j, i);
}

void Lagrange::eval(const double* xi, double* N, double* dN) const {
  double pw[3][8];
  for (int a = 0; a < dim; ++a) {
    pw[a][0] = 1;
    for (int e = 1; e <= degree; ++e) pw[a][e] = pw[a][e - 1] * xi[a];
  }
  for (int i = 0; i < n; ++i) N[i] = 0;
  if (dN)
    for (int i = 0; i < n * dim; ++i) dN[i] = 0;
  for (int j = 0; j < n; ++j) {
    const int* ex = &exps_[j * dim];
    double m = 1;
    for (int a = 0; a < dim; ++a) m *= pw[a][ex[a]];
    const double* c = &C_[static_cast<size_t>(j) * n];
    for (int i = 0; i < n; ++i) N[i] += c[i] * m;
    if (!dN) continue;
    for (int a = 0; a < dim; ++a) {
      if (ex[a] == 0) continue;
      double dm = ex[a] * pw[a][ex[a] - 1];
      for (int b = 0; b < dim; ++b)
        if (b != a) dm *= pw[b][ex[b]];
      for (int i = 0; i < n; ++i) dN[i * dim + a] += c[i] * dm;
    }
  }
}

// ============================================================================================ grids
void Grid::interp(const double* B, const double* x, double out[3]) const {
  long idx[3] = {0, 0, 0};
  double fr[3] = {0, 0, 0};
  for (int a = 0; a < dim; ++a) {
    const double t = (x[a] - origin[a]) / spacing[a];
    long i = static_cast<long>(std::floor(t));
    i = std::clamp(i, 0L, n[a] - 2);
    idx[a] = i;
    fr[a] = t - static_cast<double>(i);
  }
  out[0] = out[1] = out[2] = 0;
  const int nc = 1 << dim;
  for (int c = 0; c < nc; ++c) {
    double w = 1;
    long o = 0;
    for (int a = 0; a < dim; ++a) {
      const int bit = (c >> a) & 1;
      w *= bit ? fr[a] : 1.0 - fr[a];
      o = o * n[a] + idx[a] + bit;
    }
    for (int k = 0; k < 3; ++k) out[k] += w * B[o * 3 + k];
  }
}

// ============================================================================================ the mapper's geometry
namespace {

// A uniform grid of bins over element bounding boxes; each bin lists its elements in increasing index.
struct Bins {
  int dim = 2;
  double lo[3] = {0, 0, 0}, h[3] = {1, 1, 1};
  int n[3] = {1, 1, 1};
  std::vector<int> start, ids;
  bool empty() const { return ids.empty(); }
  int cell(int a, double x) const {
    const double t = std::floor((x - lo[a]) / h[a]);
    if (!(t >= 0)) return 0;
    return t >= n[a] ? n[a] - 1 : static_cast<int>(t);
  }
  int index(const int* c) const { return dim == 2 ? c[1] * n[0] + c[0] : (c[2] * n[1] + c[1]) * n[0] + c[0]; }
  void build(int d, const std::vector<int>& elems, const std::vector<double>& bb, double target) {
    dim = d;
    start.clear();
    ids.clear();
    if (elems.empty()) return;
    double hi[3];
    for (int a = 0; a < d; ++a) {
      lo[a] = std::numeric_limits<double>::infinity();
      hi[a] = -lo[a];
    }
    for (int e : elems)
      for (int a = 0; a < d; ++a) {
        lo[a] = std::min(lo[a], bb[(static_cast<size_t>(e) * d + a) * 2]);
        hi[a] = std::max(hi[a], bb[(static_cast<size_t>(e) * d + a) * 2 + 1]);
      }
    long total = 1;
    for (int a = 0; a < d; ++a) {
      const double L = std::max(hi[a] - lo[a], 1e-300);
      n[a] = static_cast<int>(std::clamp(std::ceil(L / std::max(target, 1e-300)), 1.0, d == 2 ? 4096.0 : 256.0));
      h[a] = L / n[a];
      total *= n[a];
    }
    for (int a = d; a < 3; ++a) n[a] = 1;
    std::vector<int> count(total + 1, 0);
    auto range = [&](int e, int* c0, int* c1) {
      for (int a = 0; a < d; ++a) {
        c0[a] = cell(a, bb[(static_cast<size_t>(e) * d + a) * 2]);
        c1[a] = cell(a, bb[(static_cast<size_t>(e) * d + a) * 2 + 1]);
      }
      for (int a = d; a < 3; ++a) c0[a] = c1[a] = 0;
    };
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 1) {
        start.assign(total + 1, 0);
        for (long i = 0; i < total; ++i) start[i + 1] = start[i] + count[i];
        ids.assign(start[total], -1);
        std::fill(count.begin(), count.end(), 0);
      }
      for (int e : elems) {
        int c0[3], c1[3], c[3];
        range(e, c0, c1);
        for (c[2] = c0[2]; c[2] <= c1[2]; ++c[2])
          for (c[1] = c0[1]; c[1] <= c1[1]; ++c[1])
            for (c[0] = c0[0]; c[0] <= c1[0]; ++c[0]) {
              const int k = index(c);
              if (pass == 0) ++count[k];
              else ids[start[k] + count[k]++] = e;
            }
      }
    }
  }
};

}  // namespace

struct Mapper::Impl {
  const Source& S;
  int dim, nv, ng;
  Lagrange G, F;                          // geometry and recovery bases
  std::vector<double> bb;                 // ne x dim x (lo, hi)
  std::vector<double> hsz;                // element size: sqrt(2 area) or (6 volume)^(1/3)
  std::vector<char> affine;
  std::vector<double> x0, Ja, Jai;        // affine (vertex) map per element: x = x0 + Ja xi; Jai = Ja^-1
  std::vector<std::vector<int>> fpoly;    // per local face: geometry nodes (2D: polyline in order; 3D: triangles)
  std::vector<std::vector<int>> frnodes;  // per local face: recovery nodes on it
  std::vector<std::array<double, 3>> fmid;  // per local face: reference midpoint
  std::vector<char> domain;               // element in the physical domain (not an excluded region)
  std::vector<int> reg;                   // region index of each element
  std::vector<std::string> rname;
  std::vector<RegionLaw> rlaw;
  Bins all;                               // every domain element

  explicit Impl(const Source& s);

  const double* X(int e) const { return &S.gx[static_cast<size_t>(e) * ng * dim]; }
  // physical point and Jacobian (J[a*dim+b] = d x_a / d xi_b) at a reference point
  void map(int e, const double* xi, double* x, double* J = nullptr) const {
    double Gs[64], dG[192];
    G.eval(xi, Gs, J ? dG : nullptr);
    const double* P = X(e);
    for (int a = 0; a < dim; ++a) x[a] = 0;
    if (J)
      for (int i = 0; i < dim * dim; ++i) J[i] = 0;
    for (int k = 0; k < ng; ++k) {
      for (int a = 0; a < dim; ++a) {
        x[a] += Gs[k] * P[k * dim + a];
        if (J)
          for (int b = 0; b < dim; ++b) J[a * dim + b] += dG[k * dim + b] * P[k * dim + a];
      }
    }
  }
  static void solve(int d, const double* J, const double* r, double* out) {
    if (d == 2) {
      const double det = J[0] * J[3] - J[1] * J[2];
      out[0] = (J[3] * r[0] - J[1] * r[1]) / det;
      out[1] = (-J[2] * r[0] + J[0] * r[1]) / det;
      return;
    }
    const double c00 = J[4] * J[8] - J[5] * J[7], c01 = J[5] * J[6] - J[3] * J[8], c02 = J[3] * J[7] - J[4] * J[6];
    const double det = J[0] * c00 + J[1] * c01 + J[2] * c02;
    const double c10 = J[2] * J[7] - J[1] * J[8], c11 = J[0] * J[8] - J[2] * J[6], c12 = J[1] * J[6] - J[0] * J[7];
    const double c20 = J[1] * J[5] - J[2] * J[4], c21 = J[2] * J[3] - J[0] * J[5], c22 = J[0] * J[4] - J[1] * J[3];
    out[0] = (c00 * r[0] + c10 * r[1] + c20 * r[2]) / det;
    out[1] = (c01 * r[0] + c11 * r[1] + c21 * r[2]) / det;
    out[2] = (c02 * r[0] + c12 * r[1] + c22 * r[2]) / det;
  }
  // reference coordinates of x in element e, without projection (outside points get the polynomial's continuation);
  // false when Newton's method did not converge (curved elements only)
  bool ref(int e, const double* x, double* xi, const double* guess = nullptr) const {
    const double* o = &x0[static_cast<size_t>(e) * dim];
    const double* Ai = &Jai[static_cast<size_t>(e) * dim * dim];
    double d[3];
    for (int a = 0; a < dim; ++a) d[a] = x[a] - o[a];
    if (affine[e] || !guess) {
      for (int a = 0; a < dim; ++a) {
        xi[a] = 0;
        for (int b = 0; b < dim; ++b) xi[a] += Ai[a * dim + b] * d[b];
      }
      if (affine[e]) return true;
    } else {
      for (int a = 0; a < dim; ++a) xi[a] = guess[a];
    }
    double y[3], J[9], r[3], dx[3];
    for (int it = 0; it < 30; ++it) {
      map(e, xi, y, J);
      for (int a = 0; a < dim; ++a) r[a] = x[a] - y[a];
      solve(dim, J, r, dx);
      double m = 0;
      for (int a = 0; a < dim; ++a) {
        xi[a] += dx[a];
        m = std::max(m, std::fabs(dx[a]));
      }
      if (!(m == m) || m > 1e6) return false;
      if (m < 1e-15) return true;
    }
    return true;  // converged to round-off in practice; the inside test decides
  }
  double lmin(const double* xi) const {
    double s = 0, m = std::numeric_limits<double>::infinity();
    for (int a = 0; a < dim; ++a) {
      s += xi[a];
      m = std::min(m, xi[a]);
    }
    return std::min(m, 1.0 - s);
  }
  // barycentric coordinate of local face j (opposite vertex j)
  double lam(const double* xi, int j) const {
    if (j > 0) return xi[j - 1];
    double s = 0;
    for (int a = 0; a < dim; ++a) s += xi[a];
    return 1.0 - s;
  }
  bool in_bbox(int e, const double* x) const {
    for (int a = 0; a < dim; ++a) {
      const double* b = &bb[(static_cast<size_t>(e) * dim + a) * 2];
      if (x[a] < b[0] || x[a] > b[1]) return false;
    }
    return true;
  }
  bool contains(int e, const double* x, double* xi, double tol = kInside) const {
    if (!in_bbox(e, x)) return false;
    if (!ref(e, x, xi)) return false;
    return lmin(xi) >= -tol;
  }
  int locate_in(const Bins& B, const double* x, double* xi, const std::vector<char>* mask = nullptr) const {
    if (B.empty()) return -1;
    int c[3] = {0, 0, 0};
    for (int a = 0; a < dim; ++a) {
      if (x[a] < B.lo[a] || x[a] > B.lo[a] + B.n[a] * B.h[a]) return -1;
      c[a] = B.cell(a, x[a]);
    }
    const int k = B.index(c);
    for (int i = B.start[k]; i < B.start[k + 1]; ++i) {
      const int e = B.ids[i];
      if (mask && !(*mask)[e]) continue;
      if (contains(e, x, xi)) return e;
    }
    return -1;
  }
  // distance from x to the boundary polygon of element e (straight segments or flat triangles through its nodes)
  double poly_dist(int e, const double* x) const {
    const double* P = X(e);
    double best = std::numeric_limits<double>::infinity();
    for (int j = 0; j < nv; ++j) {
      const auto& f = fpoly[j];
      if (dim == 2) {
        for (size_t k = 0; k + 1 < f.size(); ++k) best = std::min(best, seg_dist(x, &P[f[k] * 2], &P[f[k + 1] * 2]));
      } else {
        for (size_t k = 0; k + 2 < f.size(); k += 3)
          best = std::min(best, tri_dist(x, &P[f[k] * 3], &P[f[k + 1] * 3], &P[f[k + 2] * 3]));
      }
    }
    return best;
  }
  // where the line p + t v leaves element e after t = s (s <= t <= smax), and through which local face (-1: it
  // reaches smax inside)
  double exit(int e, const double* p, const double* v, double s, double smax, int& face) const;
  int region_of(int e) const { return reg[e]; }
  const RegionLaw& law_of(int e) const { return rlaw[reg[e]]; }
};

Mapper::Impl::Impl(const Source& s) : S(s), dim(s.dim), nv(s.dim + 1), ng(s.ng) {
  const int ne = S.ne;
  G = Lagrange(dim, S.gref);
  F = Lagrange(dim, S.nref);
  AXMP_REQUIRE(G.n == ng && ng >= nv && F.n == S.nd, "internal: field-map element bases do not match the mesh");
  // reference vertices must come first
  for (int k = 0; k < nv; ++k)
    for (int a = 0; a < dim; ++a)
      AXMP_REQUIRE(S.gref[k * dim + a] == (k == a + 1 ? 1.0 : 0.0), "internal: geometry nodes must list the vertices first");
  // regions
  std::map<std::string, int> rid;
  reg.assign(ne, -1);
  domain.assign(ne, 1);
  for (int e = 0; e < ne; ++e) {
    auto it = S.region_of_attr.find(S.attr[e]);
    const std::string name = it == S.region_of_attr.end() ? strcat("attr", S.attr[e]) : it->second;
    auto r = rid.find(name);
    if (r == rid.end()) {
      r = rid.emplace(name, static_cast<int>(rname.size())).first;
      rname.push_back(name);
      auto lw = S.law.find(name);
      rlaw.push_back(lw == S.law.end() ? RegionLaw{} : lw->second);
    }
    reg[e] = r->second;
    domain[e] = rlaw[r->second].kind != RegionLaw::Excluded;
  }
  // faces of the reference element: nodes on each, polylines / triangles
  auto on_face = [&](const double* xi, int j) { return std::fabs(lam(xi, j)) < 1e-12; };
  fpoly.assign(nv, {});
  frnodes.assign(nv, {});
  fmid.assign(nv, {0, 0, 0});
  for (int j = 0; j < nv; ++j) {
    std::vector<int> nodes;
    for (int k = 0; k < ng; ++k)
      if (on_face(&S.gref[k * dim], j)) nodes.push_back(k);
    std::vector<int> verts;
    for (int k = 0; k < nv; ++k)
      if (k != j) verts.push_back(k);
    for (int k : verts)
      for (int a = 0; a < dim; ++a) fmid[j][a] += S.gref[k * dim + a] / dim;
    if (dim == 2) {
      const double* a0 = &S.gref[verts[0] * 2];
      std::sort(nodes.begin(), nodes.end(), [&](int u, int w) {
        const double du = std::hypot(S.gref[u * 2] - a0[0], S.gref[u * 2 + 1] - a0[1]);
        const double dw = std::hypot(S.gref[w * 2] - a0[0], S.gref[w * 2 + 1] - a0[1]);
        return du < dw;
      });
      fpoly[j] = nodes;
    } else {
      // flat triangles through the face's nodes: the face itself (straight), or the four of a quadratic face
      auto mid = [&](int u, int w) {
        for (int k : nodes) {
          bool ok = true;
          for (int a = 0; a < 3; ++a)
            ok = ok && std::fabs(S.gref[k * 3 + a] - 0.5 * (S.gref[u * 3 + a] + S.gref[w * 3 + a])) < 1e-12;
          if (ok) return k;
        }
        return -1;
      };
      const int a = verts[0], b = verts[1], c = verts[2];
      const int ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      if (nodes.size() == 6 && ab >= 0 && bc >= 0 && ca >= 0)
        fpoly[j] = {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca};
      else
        fpoly[j] = {a, b, c};
    }
    for (int k = 0; k < S.nd; ++k)
      if (on_face(&S.nref[k * dim], j)) frnodes[j].push_back(k);
  }
  // per-element geometry
  bb.assign(static_cast<size_t>(ne) * dim * 2, 0.0);
  hsz.assign(ne, 0.0);
  affine.assign(ne, 0);
  x0.assign(static_cast<size_t>(ne) * dim, 0.0);
  Ja.assign(static_cast<size_t>(ne) * dim * dim, 0.0);
  Jai.assign(static_cast<size_t>(ne) * dim * dim, 0.0);
  // boundary sample points of curved elements (for their true bounding box)
  std::vector<std::vector<double>> samples;
  {
    const int m = 16;
    for (int j = 0; j < nv; ++j) {
      std::vector<int> verts;
      for (int k = 0; k < nv; ++k)
        if (k != j) verts.push_back(k);
      if (dim == 2) {
        for (int i = 0; i <= m; ++i) {
          const double t = static_cast<double>(i) / m;
          std::vector<double> xi(2);
          for (int a = 0; a < 2; ++a) xi[a] = (1 - t) * S.gref[verts[0] * 2 + a] + t * S.gref[verts[1] * 2 + a];
          samples.push_back(xi);
        }
      } else {
        for (int i = 0; i <= m; ++i)
          for (int k = 0; i + k <= m; ++k) {
            const double t = static_cast<double>(i) / m, w = static_cast<double>(k) / m;
            std::vector<double> xi(3);
            for (int a = 0; a < 3; ++a)
              xi[a] = (1 - t - w) * S.gref[verts[0] * 3 + a] + t * S.gref[verts[1] * 3 + a] + w * S.gref[verts[2] * 3 + a];
            samples.push_back(xi);
          }
      }
    }
  }
  for (int e = 0; e < ne; ++e) {
    const double* P = X(e);
    double* o = &x0[static_cast<size_t>(e) * dim];
    double* A = &Ja[static_cast<size_t>(e) * dim * dim];
    for (int a = 0; a < dim; ++a) {
      o[a] = P[a];
      for (int b = 0; b < dim; ++b) A[a * dim + b] = P[(b + 1) * dim + a] - P[a];
    }
    // inverse of the vertex map
    double* Ai = &Jai[static_cast<size_t>(e) * dim * dim];
    for (int b = 0; b < dim; ++b) {
      double unit[3] = {0, 0, 0}, col[3];
      unit[b] = 1;
      solve(dim, A, unit, col);
      for (int a = 0; a < dim; ++a) Ai[a * dim + b] = col[a];
    }
    hsz[e] = dim == 2 ? std::sqrt(2 * S.vol[e]) : std::cbrt(6 * S.vol[e]);
    // affine: every geometry node on the vertex map
    double ext = 0;
    double lo[3], hi[3];
    for (int a = 0; a < dim; ++a) {
      lo[a] = std::numeric_limits<double>::infinity();
      hi[a] = -lo[a];
    }
    for (int k = 0; k < ng; ++k)
      for (int a = 0; a < dim; ++a) {
        lo[a] = std::min(lo[a], P[k * dim + a]);
        hi[a] = std::max(hi[a], P[k * dim + a]);
      }
    for (int a = 0; a < dim; ++a) ext = std::max(ext, hi[a] - lo[a]);
    bool aff = true;
    for (int k = nv; k < ng && aff; ++k) {
      for (int a = 0; a < dim; ++a) {
        double y = o[a];
        for (int b = 0; b < dim; ++b) y += A[a * dim + b] * S.gref[k * dim + b];
        if (std::fabs(y - P[k * dim + a]) > 1e-12 * ext) aff = false;
      }
    }
    affine[e] = aff;
    if (!aff) {  // the curved boundary can bulge beyond its nodes: sample it
      for (const auto& xi : samples) {
        double y[3];
        map(e, xi.data(), y);
        for (int a = 0; a < dim; ++a) {
          lo[a] = std::min(lo[a], y[a]);
          hi[a] = std::max(hi[a], y[a]);
        }
      }
    }
    const double pad = (aff ? 1e-12 : 1e-3) * ext;
    for (int a = 0; a < dim; ++a) {
      bb[(static_cast<size_t>(e) * dim + a) * 2] = lo[a] - pad;
      bb[(static_cast<size_t>(e) * dim + a) * 2 + 1] = hi[a] + pad;
    }
  }
  std::vector<int> dom;
  double hsum = 0;
  for (int e = 0; e < ne; ++e)
    if (domain[e]) {
      dom.push_back(e);
      hsum += hsz[e];
    }
  all.build(dim, dom, bb, dom.empty() ? 1.0 : 1.5 * hsum / dom.size());
}

double Mapper::Impl::exit(int e, const double* p, const double* v, double s, double smax, int& face) const {
  face = -1;
  // barycentric coordinates along the line under the vertex map: lambda_j(t) = a_j + b_j t
  const double* o = &x0[static_cast<size_t>(e) * dim];
  const double* Ai = &Jai[static_cast<size_t>(e) * dim * dim];
  double xp[3] = {0, 0, 0}, dv[3] = {0, 0, 0};
  for (int a = 0; a < dim; ++a)
    for (int b = 0; b < dim; ++b) {
      xp[a] += Ai[a * dim + b] * (p[b] - o[b]);
      dv[a] += Ai[a * dim + b] * v[b];
    }
  double best = std::numeric_limits<double>::infinity();
  int bj = -1;
  for (int j = 0; j < nv; ++j) {
    const double aj = lam(xp, j);
    const double bj_ = j > 0 ? dv[j - 1] : -(dv[0] + dv[1] + (dim == 3 ? dv[2] : 0.0));
    if (bj_ < 0) {
      const double t = -aj / bj_;
      if (t < best) {
        best = t;
        bj = j;
      }
    }
  }
  if (affine[e]) {
    if (best >= smax) return smax;
    face = bj;
    return std::max(best, s);
  }
  // Curved: the first zero after s of the inside measure m(t) = min_j lambda_j(xi(p + t v)). At s, where the line
  // entered, m is zero up to round-off, so a bracket cannot start there: the line is sampled from s in steps of an
  // eighth of the vertex map's piece (at least h/256, at most h/8) up to the first sample outside, and the exit is
  // refined between the last sample inside and that one (Illinois from a clearly positive end, bisection when the
  // inside end is s itself). The first sample outside also ends the piece when the line leaves through a concave face
  // and comes back in: the re-entry is a later step of the walk. An excursion shorter than a step (a line within a
  // sliver of tangency to the face) is not resolved.
  double xi[3] = {0, 0, 0}, guess[3];
  int jm = -1;
  auto m = [&](double t, int& jmin) {
    double x[3];
    for (int a = 0; a < dim; ++a) x[a] = p[a] + t * v[a];
    if (!ref(e, x, xi, guess)) {
      ref(e, x, xi);  // the vertex map's coordinates
    }
    for (int a = 0; a < dim; ++a) guess[a] = xi[a];
    double mm = std::numeric_limits<double>::infinity();
    jmin = -1;
    for (int j = 0; j < nv; ++j) {
      const double l = lam(xi, j);
      if (l < mm) {
        mm = l;
        jmin = j;
      }
    }
    return mm;
  };
  {
    double x[3];
    for (int a = 0; a < dim; ++a) x[a] = p[a] + s * v[a];
    ref(e, x, guess);  // Newton from the vertex map
  }
  const double h = hsz[e];
  double lo = s, flo = m(s, jm);
  if (flo < -1e-9) {  // not in e at s (the caller came through a corner): leave at once
    face = jm;
    return s;
  }
  const double span = std::isfinite(best) && best > s ? best - s : h;
  const double dt = std::clamp(span / 8, h / 256, h / 8);
  double hi = -1, fhi = 0;
  int jhi = -1;
  for (long k = 1;; ++k) {
    const double t = std::min(s + static_cast<double>(k) * dt, smax);
    const double ft = m(t, jm);
    if (ft < 0) {
      hi = t;
      fhi = ft;
      jhi = jm;
      break;
    }
    lo = t;
    flo = ft;
    if (t >= smax) return smax;  // inside up to smax
  }
  int side = 0;
  for (int it = 0; it < 200 && hi - lo > 1e-15 * (h + std::fabs(hi)); ++it) {
    double t = 0.5 * (lo + hi);
    if (flo > 1e-12) {  // Illinois (regula falsi with halving of a retained end's value)
      t = (lo * fhi - hi * flo) / (fhi - flo);
      if (!(t > lo && t < hi)) t = 0.5 * (lo + hi);
    }
    const double ft = m(t, jm);
    if (ft >= 0) {
      lo = t;
      flo = ft;
      if (side == -1) fhi *= 0.5;
      side = -1;
    } else {
      hi = t;
      fhi = ft;
      jhi = jm;
      if (side == 1) flo *= 0.5;
      side = 1;
    }
  }
  face = jhi;
  return std::max(lo, s);
}

// ============================================================================================ the mapper
Mapper::Mapper(std::unique_ptr<Source> src) : src_(std::move(src)) { p_ = std::make_unique<Impl>(*src_); }
Mapper::~Mapper() = default;

namespace {
const char* law_name(RegionLaw::Kind k) {
  switch (k) {
    case RegionLaw::Linear: return "linear";
    case RegionLaw::Nonlinear: return "nonlinear";
    case RegionLaw::Superconductor: return "superconductor";
    default: return "excluded";
  }
}
bool same_law(const RegionLaw& a, const RegionLaw& b) {
  if (a.kind != b.kind) return false;
  if (a.kind == RegionLaw::Linear) return std::fabs(a.mu_r - b.mu_r) <= 1e-6 * std::max(std::fabs(a.mu_r), std::fabs(b.mu_r));
  if (a.kind == RegionLaw::Nonlinear) return a.material == b.material;
  return false;
}
}  // namespace

json Mapper::regions() const {
  json out = json::array();
  const Impl& I = *p_;
  std::vector<int> order(I.rname.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return I.rname[a] < I.rname[b]; });
  for (int r : order) {
    json j = {{"name", I.rname[r]}, {"law", law_name(I.rlaw[r].kind)}};
    if (I.rlaw[r].kind == RegionLaw::Linear) j["mu_r"] = I.rlaw[r].mu_r;
    if (I.rlaw[r].kind == RegionLaw::Nonlinear) j["material"] = I.rlaw[r].material;
    out.push_back(j);
  }
  return out;
}

namespace {
// connected components of the elements of a set of regions, without crossing curve faces; numbered in the order of
// each component's lowest element index
std::vector<int> components(const Mapper::Impl& I, const Source& S, const std::vector<char>& in_group, int& nc) {
  const int ne = S.ne, nv = I.nv;
  std::vector<int> comp(ne, -1);
  nc = 0;
  std::vector<int> stack;
  for (int e0 = 0; e0 < ne; ++e0) {
    if (!in_group[e0] || comp[e0] >= 0) continue;
    comp[e0] = nc;
    stack.assign(1, e0);
    while (!stack.empty()) {
      const int e = stack.back();
      stack.pop_back();
      for (int j = 0; j < nv; ++j) {
        const int n = S.nbr[e * nv + j];
        if (n < 0 || !in_group[n] || comp[n] >= 0) continue;
        if (S.curve_of_attr.count(S.fattr[e * nv + j])) continue;
        comp[n] = nc;
        stack.push_back(n);
      }
    }
    ++nc;
  }
  return comp;
}

std::string joined(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "+" : "") + v[i];
  return s;
}
}  // namespace

json Mapper::decompose(const std::vector<std::string>& selection, bool merge) const {
  const Impl& I = *p_;
  const Source& S = *src_;
  const int nr = static_cast<int>(I.rname.size()), nv = I.nv;
  std::vector<char> sel(nr, 0);
  json zero = json::array();
  {
    std::vector<std::string> sc;
    for (int r = 0; r < nr; ++r)
      if (I.rlaw[r].kind == RegionLaw::Superconductor) sc.push_back(I.rname[r]);
    std::sort(sc.begin(), sc.end());
    for (auto& n : sc) zero.push_back({{"region", n}, {"reason", "superconductor"}});
  }
  if (selection.empty()) {
    for (int r = 0; r < nr; ++r)
      sel[r] = I.rlaw[r].kind == RegionLaw::Linear || I.rlaw[r].kind == RegionLaw::Nonlinear;
  } else {
    for (const auto& name : selection) {
      auto it = std::find(I.rname.begin(), I.rname.end(), name);
      std::vector<std::string> known;
      for (int r = 0; r < nr; ++r)
        if (I.rlaw[r].kind == RegionLaw::Linear || I.rlaw[r].kind == RegionLaw::Nonlinear) known.push_back(I.rname[r]);
      std::sort(known.begin(), known.end());
      std::string list;
      for (auto& k : known) list += (list.empty() ? "" : ", ") + k;
      if (it == I.rname.end() || I.rlaw[it - I.rname.begin()].kind == RegionLaw::Excluded)
        throw fm_error("invalid_option", strcat("regions: no region named '", name, "' in the solved mesh"),
                       strcat("Name regions of the design (mapped regions: ", list, "), or pass regions=None."));
      const int r = static_cast<int>(it - I.rname.begin());
      if (I.rlaw[r].kind == RegionLaw::Superconductor)
        throw fm_error("invalid_option",
                       strcat("regions: '", name, "' is an ideal superconductor: B = 0 there and it gets no patch"),
                       "Leave it out of `regions`; the manifest lists it under zero_field_regions.");
      sel[r] = 1;
    }
  }
  // merge adjacent regions with equal laws and no curve on their shared faces
  std::vector<int> parent(nr);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int a) { return parent[a] == a ? a : parent[a] = find(parent[a]); };
  if (merge) {
    std::map<std::pair<int, int>, bool> pair_curve;  // (A, B) share a face; true if a curve lies on one
    for (int e = 0; e < S.ne; ++e) {
      const int ra = I.reg[e];
      if (!sel[ra]) continue;
      for (int j = 0; j < nv; ++j) {
        const int n = S.nbr[e * nv + j];
        if (n < 0) continue;
        const int rb = I.reg[n];
        if (rb == ra || !sel[rb]) continue;
        const bool curve = S.curve_of_attr.count(S.fattr[e * nv + j]) > 0;
        auto key = std::make_pair(std::min(ra, rb), std::max(ra, rb));
        auto it = pair_curve.find(key);
        if (it == pair_curve.end()) pair_curve[key] = curve;
        else it->second = it->second || curve;
      }
    }
    for (auto& [key, curve] : pair_curve)
      if (!curve && same_law(I.rlaw[key.first], I.rlaw[key.second])) {
        const int a = find(key.first), b = find(key.second);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
      }
  }
  std::map<int, std::vector<std::string>> groups;
  for (int r = 0; r < nr; ++r)
    if (sel[r]) groups[find(r)].push_back(I.rname[r]);
  std::vector<std::vector<std::string>> glist;
  for (auto& [root, names] : groups) {
    std::sort(names.begin(), names.end());
    glist.push_back(names);
  }
  std::sort(glist.begin(), glist.end(), [](const auto& a, const auto& b) { return joined(a) < joined(b); });
  json patches = json::array();
  for (const auto& names : glist) {
    std::vector<char> in_group(S.ne, 0);
    for (int e = 0; e < S.ne; ++e) in_group[e] = std::find(names.begin(), names.end(), I.rname[I.reg[e]]) != names.end();
    int nc = 0;
    const std::vector<int> comp = components(I, S, in_group, nc);
    std::vector<std::set<std::string>> curves(nc);
    std::vector<long> count(nc, 0);
    for (int e = 0; e < S.ne; ++e) {
      if (comp[e] < 0) continue;
      ++count[comp[e]];
      for (int j = 0; j < nv; ++j) {
        auto c = S.curve_of_attr.find(S.fattr[e * nv + j]);
        if (c == S.curve_of_attr.end()) continue;
        const int n = S.nbr[e * nv + j];
        if (n >= 0 && comp[n] == comp[e])
          throw fm_error("curve_in_region",
                         strcat("curve '", c->second, "' lies inside region '", joined(names),
                                "' with the same part of the region on both of its sides: no patch can hold a field "
                                "jump that ends inside it"),
                         strcat("Close the curve so that it separates the region into parts (each becomes a patch), "
                                "or leave region '", joined(names), "' out of `regions`."));
        curves[comp[e]].insert(c->second);
      }
    }
    for (int k = 0; k < nc; ++k) {
      json cj = json::array();
      for (auto& c : curves[k]) cj.push_back(c);
      patches.push_back({{"regions", names}, {"component", k}, {"components", nc}, {"curves", cj},
                         {"elements", count[k]}});
    }
  }
  return {{"patches", patches}, {"zero_field_regions", zero}};
}

// ============================================================================================ patches
struct Patch::Impl {
  Bins loc;                      // patch elements
  bool near_ready = false;
  Bins near;                     // patch elements with a vertex on the patch boundary
  bool jump_ready = false;
  Bins jbins;                    // jump-boundary pieces
  std::vector<double> jgeo;      // 2D: segments (4 per piece); 3D: triangles (9 per piece)
  std::vector<double> jbb;       // their bounding boxes (dim x 2)
};

Patch::Patch(const Mapper& m, std::vector<std::string> regs, int comp, std::vector<int> el)
    : M(m), regions(std::move(regs)), component(comp), elems(std::move(el)), p_(std::make_unique<Impl>()) {
  const Mapper::Impl& I = *M.p_;
  const Source& S = M.source();
  const int dim = S.dim, nv = I.nv, nd = S.nd;
  in.assign(S.ne, 0);
  for (int e : elems) in[e] = 1;
  // h_el: the median element size, element floor(n/2) of the sorted sizes
  std::vector<double> sz;
  for (int e : elems) sz.push_back(I.hsz[e]);
  std::sort(sz.begin(), sz.end());
  h_el = sz[sz.size() / 2];
  // box of the element nodes
  for (int a = 0; a < dim; ++a) {
    box[a][0] = std::numeric_limits<double>::infinity();
    box[a][1] = -box[a][0];
  }
  for (int e : elems) {
    const double* P = I.X(e);
    for (int k = 0; k < S.ng; ++k)
      for (int a = 0; a < dim; ++a) {
        box[a][0] = std::min(box[a][0], P[k * dim + a]);
        box[a][1] = std::max(box[a][1], P[k * dim + a]);
      }
    for (int j = 0; j < nv; ++j) {
      const int n = S.nbr[e * nv + j];
      if (dim == 2 && n < 0 && S.fattr[e * nv + j] == S.axis_attr) touches_axis = true;
      if (n >= 0 && I.domain[n] && !in[n]) has_jump = true;
    }
  }
  if (dim == 2) box[0][0] = touches_axis ? 0.0 : std::max(box[0][0], 0.0);
  // curved elements whose Jacobian determinant is not positive on a lattice of reference points (nodes and the
  // points between them) are tangled: neither the inverse map nor a chord walk is defined in them
  {
    const int n = dim == 2 ? 12 : 8;
    std::vector<double> lat;
    if (dim == 2) {
      for (int i = 0; i <= n; ++i)
        for (int j = 0; i + j <= n; ++j) lat.insert(lat.end(), {double(i) / n, double(j) / n});
    } else {
      for (int i = 0; i <= n; ++i)
        for (int j = 0; i + j <= n; ++j)
          for (int k = 0; i + j + k <= n; ++k) lat.insert(lat.end(), {double(i) / n, double(j) / n, double(k) / n});
    }
    const long nl = static_cast<long>(lat.size()) / dim, ne_ = static_cast<long>(elems.size());
    std::vector<double> dmin(static_cast<size_t>(ne_), std::numeric_limits<double>::infinity()), dmax(dmin.size(), -std::numeric_limits<double>::infinity());
#pragma omp parallel for schedule(dynamic, 64)
    for (long i = 0; i < ne_; ++i) {
      const int e = elems[i];
      if (I.affine[e]) continue;
      double x[3], J[9];
      for (long q = 0; q < nl; ++q) {
        I.map(e, &lat[q * dim], x, J);
        const double det = dim == 2 ? J[0] * J[3] - J[1] * J[2]
                                    : J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) +
                                          J[2] * (J[3] * J[7] - J[4] * J[6]);
        dmin[i] = std::min(dmin[i], det);
        dmax[i] = std::max(dmax[i], det);
      }
    }
    // the orientation does not matter, only that it is one sign: an element is tangled when its sign changes or
    // vanishes inside it, or when it has the opposite sign to the majority of the patch's curved elements
    long npos = 0, nneg = 0;
    for (long i = 0; i < ne_; ++i) {
      if (I.affine[elems[i]]) continue;
      if (dmin[i] > 0) ++npos;
      else if (dmax[i] < 0) ++nneg;
    }
    const int major = nneg > npos ? -1 : 1;
    long bad = 0, first = -1;
    for (long i = 0; i < ne_; ++i) {
      if (dmin[i] == std::numeric_limits<double>::infinity()) continue;  // affine
      const bool ok = major > 0 ? dmin[i] > 0 : dmax[i] < 0;
      if (!ok) {
        ++bad;
        if (first < 0) first = i;
      }
    }
    if (bad) {
      const bool changes = !(dmin[first] > 0) && !(dmax[first] < 0);
      throw fm_error("tangled_mesh",
                     strcat(bad, " of ", ne_, " curved elements of the region", regions.size() > 1 ? "s " : " ",
                            joined(regions), " have a Jacobian determinant that changes sign, vanishes or is inverted "
                            "against the other elements (element ", elems[first], ", determinant from ", dmin[first],
                            " to ", dmax[first], changes ? ", sign change" : ", inverted", "): the mesh is tangled there"),
                     "The mesh is tangled: curved elements fold over where a curved region is much thinner than the "
                     "mesh size at geometry order 2. Refine the mesh in or around that region, or use geometry order "
                     "1. No map was written.");
    }
  }
  // recovery: B_h at the nodes of the H1 space, averaged over the patch's elements; on the axis (kind 1) B_r = B_phi = 0
  // and B_z is the mean of the limits of the elements with an edge on the axis
  const long nel = static_cast<long>(elems.size());
  std::vector<double> nodal(static_cast<size_t>(nel) * nd * 3);
#pragma omp parallel for schedule(dynamic, 64)
  for (long i = 0; i < nel; ++i) {
    const int e = elems[i];
    for (int k = 0; k < nd; ++k) {
      const double* xi = &S.nref[k * dim];
      double x[3], J[9];
      I.map(e, xi, x, J);
      if (dim == 2 && std::fabs(x[0]) < S.rtol) x[0] = 0.0;
      S.Bh(e, xi, x, J, &nodal[(static_cast<size_t>(i) * nd + k) * 3]);
    }
  }
  std::vector<double> sum(static_cast<size_t>(S.ndofs) * 3, 0.0), sax(sum.size(), 0.0), sall(sum.size(), 0.0);
  std::vector<int> cnt(S.ndofs, 0), cax(S.ndofs, 0);
  for (long i = 0; i < nel; ++i) {
    const int e = elems[i];
    bool axis_edge = false;
    if (dim == 2)
      for (int j = 0; j < nv; ++j) axis_edge = axis_edge || (S.nbr[e * nv + j] < 0 && S.fattr[e * nv + j] == S.axis_attr);
    for (int k = 0; k < nd; ++k) {
      const int g = S.dofs[static_cast<size_t>(e) * nd + k];
      const double* B = &nodal[(static_cast<size_t>(i) * nd + k) * 3];
      if (dim == 2 && S.dof_x[static_cast<size_t>(g) * 2] == 0.0) {
        sall[g * 3 + 1] += B[1];
        ++cnt[g];
        if (axis_edge) {
          sax[g * 3 + 1] += B[1];
          ++cax[g];
        }
        continue;
      }
      for (int c = 0; c < 3; ++c) sum[g * 3 + c] += B[c];
      ++cnt[g];
    }
  }
  bstar.assign(static_cast<size_t>(S.ne) * nd * 3, 0.0);
  B_patch = 0;
  for (int e : elems)
    for (int k = 0; k < nd; ++k) {
      const int g = S.dofs[static_cast<size_t>(e) * nd + k];
      double* B = &bstar[(static_cast<size_t>(e) * nd + k) * 3];
      if (dim == 2 && S.dof_x[static_cast<size_t>(g) * 2] == 0.0) {
        B[0] = B[2] = 0.0;
        B[1] = cax[g] > 0 ? sax[g * 3 + 1] / cax[g] : sall[g * 3 + 1] / cnt[g];
      } else {
        for (int c = 0; c < 3; ++c) B[c] = sum[g * 3 + c] / cnt[g];
      }
      B_patch = std::max(B_patch, norm3(B));
    }
  // locator
  double hs = 0;
  for (int e : elems) hs += I.hsz[e];
  p_->loc.build(dim, elems, I.bb, 1.5 * hs / std::max<long>(1, nel));
}

Patch::~Patch() = default;

json Patch::info() const {
  const int dim = M.dim();
  json b = json::array();
  for (int a = 0; a < dim; ++a) b.push_back({box[a][0], box[a][1]});
  return {{"regions", regions}, {"component", component}, {"elements", elems.size()}, {"h_el", h_el},
          {"box", b}, {"touches_axis", touches_axis}, {"has_jump", has_jump}, {"B_patch", B_patch}};
}

void Patch::eval_star(int e, const double* xi, double B[3]) const {
  const Mapper::Impl& I = *M.p_;
  const int nd = M.source().nd;
  double N[64];
  I.F.eval(xi, N);
  B[0] = B[1] = B[2] = 0;
  const double* v = &bstar[static_cast<size_t>(e) * nd * 3];
  for (int i = 0; i < nd; ++i)
    for (int c = 0; c < 3; ++c) B[c] += N[i] * v[i * 3 + c];
}

int Patch::locate(const double* x, double* xi) const { return M.p_->locate_in(p_->loc, x, xi); }

std::vector<double> Patch::recovered(const std::vector<double>& pts) const {
  const int dim = M.dim();
  const long n = static_cast<long>(pts.size()) / dim;
  std::vector<double> out(static_cast<size_t>(n) * 3, kNaN);
#pragma omp parallel for schedule(dynamic, 256)
  for (long i = 0; i < n; ++i) {
    double xi[3];
    const double* x = &pts[static_cast<size_t>(i) * dim];
    const int e = locate(x, xi);
    if (e < 0) continue;
    double* B = &out[static_cast<size_t>(i) * 3];
    eval_star(e, xi, B);
    if (dim == 2 && x[0] == 0.0) B[0] = B[2] = 0.0;
  }
  return out;
}

bool Patch::box_meets(const std::vector<double>& bx) const {
  const Mapper::Impl& I = *M.p_;
  const int dim = M.dim();
  for (int e : elems) {
    bool ok = true;
    for (int a = 0; a < dim && ok; ++a) {
      const double* b = &I.bb[(static_cast<size_t>(e) * dim + a) * 2];
      ok = b[1] >= bx[2 * a] && b[0] <= bx[2 * a + 1];
    }
    if (ok) return true;
  }
  return false;
}

std::vector<double> Patch::fill(const Grid& g) const {
  const Mapper::Impl& I = *M.p_;
  const Source& S = M.source();
  const int dim = S.dim, nv = I.nv;
  // cells that meet the bounding box of a patch element
  long nc[3] = {1, 1, 1};
  for (int a = 0; a < dim; ++a) nc[a] = g.n[a] - 1;
  std::vector<char> cell(static_cast<size_t>(nc[0] * nc[1] * nc[2]), 0);
  for (int e : elems) {
    long c0[3] = {0, 0, 0}, c1[3] = {0, 0, 0};
    for (int a = 0; a < dim; ++a) {
      const double* b = &I.bb[(static_cast<size_t>(e) * dim + a) * 2];
      c0[a] = std::clamp(static_cast<long>(std::floor((b[0] - g.origin[a]) / g.spacing[a])), 0L, nc[a] - 1);
      c1[a] = std::clamp(static_cast<long>(std::floor((b[1] - g.origin[a]) / g.spacing[a])), 0L, nc[a] - 1);
    }
    for (long i = c0[0]; i <= c1[0]; ++i)
      for (long j = c0[1]; j <= c1[1]; ++j)
        for (long k = c0[2]; k <= c1[2]; ++k) cell[(i * nc[1] + j) * nc[2] + k] = 1;
  }
  // candidates of the nearest-element search: patch elements with a vertex on the patch boundary
  if (!p_->near_ready) {
    struct H {
      size_t operator()(const std::array<double, 3>& a) const {
        return std::hash<double>()(a[0]) ^ (std::hash<double>()(a[1]) * 31) ^ (std::hash<double>()(a[2]) * 131);
      }
    };
    std::unordered_set<std::array<double, 3>, H> bverts;
    for (int e : elems)
      for (int j = 0; j < nv; ++j) {
        const int n = S.nbr[e * nv + j];
        if (n >= 0 && in[n]) continue;
        const double* P = I.X(e);
        for (int k = 0; k < nv; ++k) {
          if (k == j) continue;
          std::array<double, 3> v = {0, 0, 0};
          for (int a = 0; a < dim; ++a) v[a] = P[k * dim + a];
          bverts.insert(v);
        }
      }
    std::vector<int> cand;
    for (int e : elems) {
      const double* P = I.X(e);
      for (int k = 0; k < nv; ++k) {
        std::array<double, 3> v = {0, 0, 0};
        for (int a = 0; a < dim; ++a) v[a] = P[k * dim + a];
        if (bverts.count(v)) {
          cand.push_back(e);
          break;
        }
      }
    }
    const_cast<Patch*>(this)->p_->near.build(dim, cand, I.bb, h_el);
    const_cast<Patch*>(this)->p_->near_ready = true;
  }
  const Bins& NB = p_->near;
  const long total = g.nodes();
  std::vector<double> out(static_cast<size_t>(total) * 3, 0.0);
  const long n1 = dim == 3 ? g.n[1] * g.n[2] : g.n[1], n2 = dim == 3 ? g.n[2] : 1;
#pragma omp parallel for schedule(dynamic, 1024)
  for (long o = 0; o < total; ++o) {
    long idx[3] = {o / n1, (o / n2) % g.n[1], dim == 3 ? o % n2 : 0};
    bool active = false;
    for (int c = 0; c < (1 << dim) && !active; ++c) {
      long ci[3] = {0, 0, 0};
      bool ok = true;
      for (int a = 0; a < dim; ++a) {
        ci[a] = idx[a] - ((c >> a) & 1);
        ok = ok && ci[a] >= 0 && ci[a] < nc[a];
      }
      if (ok && cell[(ci[0] * nc[1] + ci[1]) * nc[2] + ci[2]]) active = true;
    }
    if (!active) continue;
    double x[3] = {0, 0, 0}, xi[3];
    for (int a = 0; a < dim; ++a) x[a] = g.origin[a] + static_cast<double>(idx[a]) * g.spacing[a];
    double* B = &out[static_cast<size_t>(o) * 3];
    int e = locate(x, xi);
    if (e < 0) {
      // the nearest patch element by the distance to its node polygon; ties to the lowest index
      int c[3] = {0, 0, 0};
      for (int a = 0; a < dim; ++a) c[a] = NB.cell(a, x[a]);
      double best = std::numeric_limits<double>::infinity();
      int eb = -1;
      const int rmax = std::max({NB.n[0], NB.n[1], NB.n[2]});
      double hmin = std::min(NB.h[0], NB.h[1]);
      if (dim == 3) hmin = std::min(hmin, NB.h[2]);
      for (int rad = 0; rad <= rmax; ++rad) {
        int q[3];
        const int z0 = dim == 3 ? std::max(0, c[2] - rad) : 0, z1 = dim == 3 ? std::min(NB.n[2] - 1, c[2] + rad) : 0;
        for (q[2] = z0; q[2] <= z1; ++q[2])
          for (q[1] = std::max(0, c[1] - rad); q[1] <= std::min(NB.n[1] - 1, c[1] + rad); ++q[1])
            for (q[0] = std::max(0, c[0] - rad); q[0] <= std::min(NB.n[0] - 1, c[0] + rad); ++q[0]) {
              int ring = std::max(std::abs(q[0] - c[0]), std::abs(q[1] - c[1]));
              if (dim == 3) ring = std::max(ring, std::abs(q[2] - c[2]));
              if (ring != rad) continue;
              const int k = NB.index(q);
              for (int i = NB.start[k]; i < NB.start[k + 1]; ++i) {
                const int ec = NB.ids[i];
                const double d = I.poly_dist(ec, x);
                if (d < best || (d == best && ec < eb)) {
                  best = d;
                  eb = ec;
                }
              }
            }
        if (eb >= 0 && best < rad * hmin) break;
      }
      e = eb;
      if (e < 0) continue;
      double g0[3];
      I.ref(e, x, g0);
      if (!I.ref(e, x, xi, g0))
        for (int a = 0; a < dim; ++a) xi[a] = g0[a];
    }
    eval_star(e, xi, B);
  }
  if (dim == 2 && touches_axis)
    for (long j = 0; j < g.n[1]; ++j) out[static_cast<size_t>(j) * 3] = out[static_cast<size_t>(j) * 3 + 2] = 0.0;
  return out;
}

// ============================================================================================ chords
namespace {
struct Piece {
  int e;
  double s0, s1;
};
}  // namespace

Chords Patch::chords(const std::string& family, long N, uint64_t seed, const std::vector<double>& bx) const {
  const Mapper::Impl& I = *M.p_;
  const Source& S = M.source();
  const int dim = S.dim, nv = I.nv;
  const double Lnom = 8.0 * h_el;
  const bool layer = family == "layer", axis = family == "axis", track = family == "track";
  AXMP_REQUIRE(family == "bulk" || layer || axis || track, "internal: unknown chord family " + family);
  // the jump boundary (layer family): straight pieces through the nodes of the faces whose other side is a domain
  // element outside the patch
  if (layer && !p_->jump_ready) {
    Impl& J = *const_cast<Patch*>(this)->p_;
    std::vector<int> ids;
    for (int e : elems) {
      const double* P = I.X(e);
      for (int j = 0; j < nv; ++j) {
        const int n = S.nbr[e * nv + j];
        if (n < 0 || !I.domain[n] || in[n]) continue;
        const auto& f = I.fpoly[j];
        if (dim == 2) {
          for (size_t k = 0; k + 1 < f.size(); ++k)
            J.jgeo.insert(J.jgeo.end(), {P[f[k] * 2], P[f[k] * 2 + 1], P[f[k + 1] * 2], P[f[k + 1] * 2 + 1]});
        } else {
          for (size_t k = 0; k + 2 < f.size(); k += 3)
            for (int q = 0; q < 3; ++q)
              for (int a = 0; a < 3; ++a) J.jgeo.push_back(P[f[k + q] * 3 + a]);
        }
      }
    }
    const int per = dim == 2 ? 4 : 9, npts = dim == 2 ? 2 : 3;
    const long np = static_cast<long>(J.jgeo.size()) / per;
    J.jbb.assign(static_cast<size_t>(np) * dim * 2, 0.0);
    for (long i = 0; i < np; ++i) {
      ids.push_back(static_cast<int>(i));
      for (int a = 0; a < dim; ++a) {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (int q = 0; q < npts; ++q) {
          lo = std::min(lo, J.jgeo[i * per + q * dim + a]);
          hi = std::max(hi, J.jgeo[i * per + q * dim + a]);
        }
        J.jbb[(i * dim + a) * 2] = lo;
        J.jbb[(i * dim + a) * 2 + 1] = hi;
      }
    }
    J.jbins.build(dim, ids, J.jbb, h_el);
    J.jump_ready = true;
  }
  // distance from x to the jump boundary, capped at cap
  auto jump_dist = [&](const double* x, double cap) {
    const Impl& J = *p_;
    double d = cap;
    if (J.jbins.empty()) return d;
    int c0[3] = {0, 0, 0}, c1[3] = {0, 0, 0};
    for (int a = 0; a < dim; ++a) {
      c0[a] = J.jbins.cell(a, x[a] - cap);
      c1[a] = J.jbins.cell(a, x[a] + cap);
    }
    int q[3];
    for (q[2] = c0[2]; q[2] <= c1[2]; ++q[2])
      for (q[1] = c0[1]; q[1] <= c1[1]; ++q[1])
        for (q[0] = c0[0]; q[0] <= c1[0]; ++q[0]) {
          const int k = J.jbins.index(q);
          for (int i = J.jbins.start[k]; i < J.jbins.start[k + 1]; ++i) {
            const double* g = &J.jgeo[static_cast<size_t>(J.jbins.ids[i]) * (dim == 2 ? 4 : 9)];
            d = std::min(d, dim == 2 ? seg_dist(x, g, g + 2) : tri_dist(x, g, g + 3, g + 6));
          }
        }
    return d;
  };
  // candidate elements and their weights
  std::vector<int> cand;
  std::vector<double> cdf;
  double acc = 0;
  for (int e : elems) {
    const double* P = I.X(e);
    bool ok = true;
    if (layer) {
      ok = false;
      for (int k = 0; k < nv && !ok; ++k) ok = jump_dist(&P[k * dim], 2 * h_el + 2 * I.hsz[e]) <= h_el + I.hsz[e];
    } else if (axis) {
      double rmin = std::numeric_limits<double>::infinity();
      for (int k = 0; k < S.ng; ++k) rmin = std::min(rmin, P[k * dim]);
      ok = rmin < h_el;
    } else if (track) {
      for (int a = 0; a < dim && ok; ++a) {
        const double* b = &I.bb[(static_cast<size_t>(e) * dim + a) * 2];
        ok = b[1] >= bx[2 * a] && b[0] <= bx[2 * a + 1];
      }
    }
    if (!ok) continue;
    cand.push_back(e);
    acc += S.vol[e];
    cdf.push_back(acc);
  }
  Chords C;
  C.family = family;
  C.dim = dim;
  if (cand.empty() || N <= 0) return C;
  const double total = cdf.back();
  const Halton H(seed, 6);
  // the parameter interval of the track box along a line (slab method)
  auto box_range = [&](const double* p, const double* v, double& t0, double& t1) {
    t0 = -std::numeric_limits<double>::infinity();
    t1 = std::numeric_limits<double>::infinity();
    for (int a = 0; a < dim; ++a) {
      if (v[a] == 0) {
        if (p[a] < bx[2 * a] || p[a] > bx[2 * a + 1]) t1 = -1;
        continue;
      }
      double u0 = (bx[2 * a] - p[a]) / v[a], u1 = (bx[2 * a + 1] - p[a]) / v[a];
      if (u0 > u1) std::swap(u0, u1);
      t0 = std::max(t0, u0);
      t1 = std::min(t1, u1);
    }
  };
  struct Cand {
    int e;
    double p[3], v[3];
  };
  // Walk the mesh along p + s v from element c.e, s in [0, Lnom]: the pieces in each element. Returns false when the
  // walk cannot be completed (no element found past an interior face, or no progress): never a shorter chord.
  auto walk = [&](const Cand& c, std::vector<Piece>& out) {
    out.clear();
    int e = c.e;
    double s = 0;
    int stall = 0;
    // consecutive hand-overs that do not advance the chord parameter: two elements that each report an exit at the
    // same s would pass the chord back and forth for ever
    int idle = 0;
    double best = 0;
    for (int step = 0; step < 1000000; ++step) {
      int face = -1;
      const double se = I.exit(e, c.p, c.v, s, Lnom, face);
      if (se > s) out.push_back({e, s, se});
      if (face < 0 || se >= Lnom) return true;  // the nominal length is reached inside e
      s = std::max(s, se);
      // the element past s: the neighbour across the exit face, else any domain element holding a point just beyond
      // s (the line crossed a vertex or an edge); the probe moves further out while it still finds e
      const int n = S.nbr[e * nv + face];
      int next = -1;
      double used = 0;
      for (int k = 0; k < 7 && next < 0; ++k) {
        used = 1e-9 * I.hsz[e] * std::pow(10.0, k);
        double probe[3], xi[3];
        for (int a = 0; a < dim; ++a) probe[a] = c.p[a] + (s + used) * c.v[a];
        if (n >= 0 && I.domain[n] && I.contains(n, probe, xi, 1e-9)) next = n;
        else next = I.locate_in(I.all, probe, xi);
        if (next == e) next = -1;
      }
      if (next < 0) {
        double probe[3], xi[3];
        for (int a = 0; a < dim; ++a) probe[a] = c.p[a] + (s + used) * c.v[a];
        if (I.contains(e, probe, xi, 1e-9)) {  // e still holds the line: its exit came early; continue in e
          if (++stall > 16) return false;
          const double s1 = std::min(s + used, Lnom);
          out.push_back({e, s, s1});
          s = s1;
          if (s >= Lnom) return true;
          continue;
        }
        // nothing beyond s: the line leaves the physical domain, which is complete only through its boundary
        return n < 0 || !I.domain[n];
      }
      stall = 0;
      if (s > best + 1e-12 * std::max(Lnom, std::fabs(best))) {
        best = s;
        idle = 0;
      } else if (++idle > 32) {
        return false;
      }
      e = next;
    }
    return false;
  };
  // B*, |B*| and B_h along a piece, adaptive Gauss-Kronrod to tol (absolute, on each of the three integrals)
  auto integrate = [&](const Cand& c, int e, double a, double b, double tol, double* out) {
    double guess[3];
    bool have = false;
    auto f = [&](double s, double* y) {
      double x[3], xi[3], J[9];
      for (int k = 0; k < dim; ++k) x[k] = c.p[k] + s * c.v[k];
      if (!(have && !I.affine[e] && I.ref(e, x, xi, guess))) I.ref(e, x, xi);
      for (int k = 0; k < dim; ++k) guess[k] = xi[k];
      have = true;
      double Bs[3], Bh[3], xm[3];
      eval_star(e, xi, Bs);
      I.map(e, xi, xm, J);
      if (dim == 2 && std::fabs(x[0]) < S.rtol) x[0] = 0.0;
      S.Bh(e, xi, x, J, Bh);
      y[0] = Bs[0];
      y[1] = Bs[1];
      y[2] = Bs[2];
      y[3] = norm3(Bs);
      y[4] = Bh[0];
      y[5] = Bh[1];
      y[6] = Bh[2];
    };
    // Globally adaptive Gauss-Kronrod 7-15: split the interval with the largest error estimate |K15 - G7| until
    // the sum meets tol. Bounded: at most kMaxIntervals intervals, and it stops when splitting no longer lowers the
    // error (the estimate is then round-off, as for B_h in a screened region, where it is a small difference of
    // the background and the reaction field; a jump in B_h's 1/r term at an axis vertex halves it per split).
    struct Iv {
      double lo, hi, err, K[7];
    };
    auto gk = [&](double lo, double hi) {
      Iv v{lo, hi, 0, {0, 0, 0, 0, 0, 0, 0}};
      const double mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
      double Gs[7] = {0, 0, 0, 0, 0, 0, 0}, y[7], z[7];
      f(mid, y);
      for (int q = 0; q < 7; ++q) {
        v.K[q] += kWgk[7] * y[q];
        Gs[q] += kWg[3] * y[q];
      }
      for (int i = 0; i < 7; ++i) {
        f(mid - half * kXgk[i], y);
        f(mid + half * kXgk[i], z);
        for (int q = 0; q < 7; ++q) {
          v.K[q] += kWgk[i] * (y[q] + z[q]);
          if (i % 2 == 1) Gs[q] += kWg[i / 2] * (y[q] + z[q]);
        }
      }
      double d[7];
      for (int q = 0; q < 7; ++q) {
        d[q] = (v.K[q] - Gs[q]) * half;
        v.K[q] *= half;
      }
      v.err = std::max({norm3(d), std::fabs(d[3]), norm3(d + 4)});
      return v;
    };
    constexpr int kMaxIntervals = 64, kStalls = 3;
    std::vector<Iv> iv = {gk(a, b)};
    double total = iv[0].err;
    int stalls = 0;
    while (total > tol && static_cast<int>(iv.size()) < kMaxIntervals && stalls < kStalls) {
      size_t k = 0;
      for (size_t i = 1; i < iv.size(); ++i)
        if (iv[i].err > iv[k].err) k = i;
      const Iv parent = iv[k];
      const double mid = 0.5 * (parent.lo + parent.hi);
      if (!(mid > parent.lo && mid < parent.hi)) break;
      iv[k] = gk(parent.lo, mid);
      iv.push_back(gk(mid, parent.hi));
      stalls = iv[k].err + iv.back().err > 0.75 * parent.err ? stalls + 1 : 0;
      total = 0;
      for (const Iv& v : iv) total += v.err;
    }
    std::sort(iv.begin(), iv.end(), [](const Iv& u, const Iv& w) { return u.lo < w.lo; });
    for (const Iv& v : iv)
      for (int q = 0; q < 7; ++q) out[q] += v.K[q];
  };
  struct Traced {
    std::vector<double> seg;  // (s0, s1) pairs
    double L = 0, I[7] = {0, 0, 0, 0, 0, 0, 0};
    bool complete = true;
  };
  auto trace = [&](const Cand& c, Traced& T) {
    thread_local std::vector<Piece> pieces;
    if (!walk(c, pieces)) {
      T.complete = false;
      return;
    }
    double t0 = 0, t1 = Lnom;
    if (track) {
      double b0, b1;
      box_range(c.p, c.v, b0, b1);
      t0 = std::max(t0, b0);
      t1 = std::min(t1, b1);
    }
    std::vector<Piece> keep;
    for (const auto& pc : pieces) {
      if (!in[pc.e]) continue;
      const double a = std::max(pc.s0, t0), b = std::min(pc.s1, t1);
      if (b > a) keep.push_back({pc.e, a, b});
    }
    for (const auto& pc : keep) {
      if (!T.seg.empty() && T.seg.back() == pc.s0) T.seg.back() = pc.s1;
      else T.seg.insert(T.seg.end(), {pc.s0, pc.s1});
    }
    for (size_t k = 0; k < T.seg.size(); k += 2) T.L += T.seg[k + 1] - T.seg[k];
    for (const auto& pc : keep) integrate(c, pc.e, pc.s0, pc.s1, 1e-11 * B_patch * (pc.s1 - pc.s0), T.I);
  };
  // attempts in order; each consumes the next point of the sequence; chords with L_c = 0 are dropped
  uint64_t k = 0;
  const uint64_t max_attempts = std::max<uint64_t>(1u << 24, static_cast<uint64_t>(N) * 4096);
  while (C.size() < N) {
    const long want = N - C.size();
    std::vector<Cand> batch;
    while (static_cast<long>(batch.size()) < std::max<long>(want + want / 8 + 16, 64) && k < max_attempts) {
      double u[6];
      H.point(++k, u);
      long i = static_cast<long>(std::upper_bound(cdf.begin(), cdf.end(), u[0] * total) - cdf.begin());
      i = std::min<long>(i, static_cast<long>(cand.size()) - 1);
      Cand c{};
      c.e = cand[i];
      double xi[3];
      if (dim == 2) {
        double s = u[1], t = u[2];
        if (s + t > 1) {
          s = 1 - s;
          t = 1 - t;
        }
        xi[0] = s;
        xi[1] = t;
      } else {
        double s = u[1], t = u[2], w = u[3];
        if (s + t > 1) {
          s = 1 - s;
          t = 1 - t;
        }
        if (t + w > 1) {
          const double tt = t;
          t = 1 - w;
          w = 1 - s - tt;
        } else if (s + t + w > 1) {
          const double ss = s;
          s = 1 - t - w;
          w = ss + t + w - 1;
        }
        xi[0] = s;
        xi[1] = t;
        xi[2] = w;
      }
      I.map(c.e, xi, c.p);
      if (layer && jump_dist(c.p, 2 * h_el) > h_el) continue;
      if (axis && c.p[0] >= h_el) continue;
      if (track) {
        bool inb = true;
        for (int a = 0; a < dim; ++a) inb = inb && c.p[a] >= bx[2 * a] && c.p[a] <= bx[2 * a + 1];
        if (!inb) continue;
      }
      if (dim == 2) {
        const double th = 2.0 * M_PI * u[3];
        c.v[0] = std::cos(th);
        c.v[1] = std::sin(th);
        c.v[2] = 0;
      } else {
        const double ct = 2.0 * u[4] - 1.0, ph = 2.0 * M_PI * u[5];
        const double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
        c.v[0] = st * std::cos(ph);
        c.v[1] = st * std::sin(ph);
        c.v[2] = ct;
      }
      batch.push_back(c);
    }
    if (batch.empty()) break;
    std::vector<Traced> res(batch.size());
#pragma omp parallel for schedule(dynamic, 16)
    for (long b = 0; b < static_cast<long>(batch.size()); ++b) trace(batch[b], res[b]);
    for (size_t b = 0; b < batch.size() && C.size() < N; ++b) {
      if (!res[b].complete) {
        const Cand& c = batch[b];
        std::string at, dir;
        for (int a = 0; a < dim; ++a) {
          at += strcat(a ? ", " : "", c.p[a]);
          dir += strcat(a ? ", " : "", c.v[a]);
        }
        throw fm_error("chord_incomplete",
                       strcat("the walk of chord ", C.size(), " of family ", family, " (patch of regions ",
                              joined(regions), ", component ", component, "; start (", at, "), direction (", dir,
                              ")) through the mesh could not be completed"),
                       "This is an error of the map builder, not of the design: report it with the design. No map "
                       "was written.");
      }
      if (!(res[b].L > 0)) continue;
      const long id = C.size();
      for (int a = 0; a < dim; ++a) {
        C.start.push_back(batch[b].p[a]);
        C.direction.push_back(batch[b].v[a]);
      }
      for (size_t q = 0; q < res[b].seg.size(); q += 2) {
        C.segments.insert(C.segments.end(), {res[b].seg[q], res[b].seg[q + 1]});
        C.segment_chord.push_back(id);
      }
      C.length.push_back(res[b].L);
      C.I_star.insert(C.I_star.end(), {res[b].I[0], res[b].I[1], res[b].I[2]});
      C.I_abs.push_back(res[b].I[3]);
      C.I_h.insert(C.I_h.end(), {res[b].I[4], res[b].I[5], res[b].I[6]});
    }
    if (k >= max_attempts) break;
  }
  C.attempts = static_cast<long>(k);
  return C;
}

std::vector<double> Chords::map_integrals(const Grid& g, const double* B) const {
  const long nch = size(), m = static_cast<long>(segment_chord.size());
  std::vector<double> out(static_cast<size_t>(nch) * 3, 0.0);
  // the first segment of each chord (segments are sorted by chord)
  std::vector<long> first(nch + 1, m);
  for (long i = m - 1; i >= 0; --i) first[segment_chord[i]] = i;
  const double gx = 1.0 / std::sqrt(3.0);
#pragma omp parallel for schedule(dynamic, 64)
  for (long c = 0; c < nch; ++c) {
    const double* p = &start[static_cast<size_t>(c) * dim];
    const double* v = &direction[static_cast<size_t>(c) * dim];
    std::vector<double> cuts;
    double acc[3] = {0, 0, 0};
    for (long s = first[c]; s < m && segment_chord[s] == c; ++s) {
      const double s0 = segments[2 * s], s1 = segments[2 * s + 1];
      cuts.assign({s0, s1});
      for (int a = 0; a < dim; ++a) {
        if (v[a] == 0) continue;
        const double x0 = p[a] + s0 * v[a], x1 = p[a] + s1 * v[a];
        const long lo = static_cast<long>(std::ceil((std::min(x0, x1) - g.origin[a]) / g.spacing[a]));
        const long hi = static_cast<long>(std::floor((std::max(x0, x1) - g.origin[a]) / g.spacing[a]));
        for (long k = lo; k <= hi; ++k) {
          const double t = (g.origin[a] + static_cast<double>(k) * g.spacing[a] - p[a]) / v[a];
          if (t > s0 && t < s1) cuts.push_back(t);
        }
      }
      std::sort(cuts.begin(), cuts.end());
      for (size_t q = 0; q + 1 < cuts.size(); ++q) {
        const double a = cuts[q], b = cuts[q + 1];
        if (!(b > a)) continue;
        const double mid = 0.5 * (b + a), half = 0.5 * (b - a);
        for (int r = -1; r <= 1; r += 2) {
          const double sq = mid + r * half * gx;
          double x[3] = {0, 0, 0}, val[3];
          for (int k = 0; k < dim; ++k) x[k] = p[k] + sq * v[k];
          g.interp(B, x, val);
          for (int k = 0; k < 3; ++k) acc[k] += half * val[k];
        }
      }
    }
    for (int k = 0; k < 3; ++k) out[static_cast<size_t>(c) * 3 + k] = acc[k];
  }
  return out;
}

Statistic statistic(const Chords& c, const std::vector<double>& I_map, double B_patch) {
  Statistic st;
  const long n = c.size();
  st.n = n;
  if (n == 0) return st;
  std::vector<double> e(n);
  double se = 0, sh = 0;
  for (long i = 0; i < n; ++i) {
    const double den = std::max(c.I_abs[i], kFloor * c.length[i] * B_patch);
    double d1[3], d2[3];
    for (int k = 0; k < 3; ++k) {
      d1[k] = I_map[i * 3 + k] - c.I_star[i * 3 + k];
      d2[k] = c.I_star[i * 3 + k] - c.I_h[i * 3 + k];
    }
    e[i] = norm3(d1) / den;
    const double h = norm3(d2) / den;
    se += e[i] * e[i];
    sh += h * h;
  }
  st.E = std::sqrt(se / n);
  st.H = std::sqrt(sh / n);
  std::sort(e.begin(), e.end());
  st.E_q95 = e[static_cast<size_t>(std::floor(0.95 * static_cast<double>(n - 1)))];
  st.E_max = e.back();
  return st;
}

json Patch::pointwise(const Grid& g, const double* B, long n, uint64_t seed) const {
  const Mapper::Impl& I = *M.p_;
  const Source& S = M.source();
  const int dim = S.dim;
  std::vector<double> cdf;
  double acc = 0;
  for (int e : elems) {
    acc += S.vol[e];
    cdf.push_back(acc);
  }
  const Halton H(seed, 6);
  std::vector<double> a(n), b(n);
#pragma omp parallel for schedule(dynamic, 1024)
  for (long k = 0; k < n; ++k) {
    double u[6], xi[3], x[3], J[9];
    H.point(static_cast<uint64_t>(k + 1), u);
    long i = static_cast<long>(std::upper_bound(cdf.begin(), cdf.end(), u[0] * acc) - cdf.begin());
    i = std::min<long>(i, static_cast<long>(elems.size()) - 1);
    const int e = elems[i];
    if (dim == 2) {
      double s = u[1], t = u[2];
      if (s + t > 1) {
        s = 1 - s;
        t = 1 - t;
      }
      xi[0] = s;
      xi[1] = t;
    } else {
      double s = u[1], t = u[2], w = u[3];
      if (s + t > 1) {
        s = 1 - s;
        t = 1 - t;
      }
      if (t + w > 1) {
        const double tt = t;
        t = 1 - w;
        w = 1 - s - tt;
      } else if (s + t + w > 1) {
        const double ss = s;
        s = 1 - t - w;
        w = ss + t + w - 1;
      }
      xi[0] = s;
      xi[1] = t;
      xi[2] = w;
    }
    I.map(e, xi, x, J);
    if (dim == 2 && std::fabs(x[0]) < S.rtol) x[0] = 0.0;
    double Bs[3], Bm[3], Bh[3], d[3];
    eval_star(e, xi, Bs);
    if (dim == 2 && x[0] == 0.0) Bs[0] = Bs[2] = 0.0;
    g.interp(B, x, Bm);
    S.Bh(e, xi, x, J, Bh);
    for (int c = 0; c < 3; ++c) d[c] = Bm[c] - Bs[c];
    a[k] = norm3(d) / B_patch;
    for (int c = 0; c < 3; ++c) d[c] = Bs[c] - Bh[c];
    b[k] = norm3(d) / B_patch;
  }
  auto q = [&](std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    const double m = static_cast<double>(v.size() - 1);
    return json{{"q50", v[static_cast<size_t>(std::floor(0.5 * m))]},
                {"q99", v[static_cast<size_t>(std::floor(0.99 * m))]},
                {"max", v.back()}};
  };
  return {{"n", n}, {"map_vs_Bstar", q(a)}, {"indicator", q(b)}};
}

double Mapper::boundary_max(const std::vector<const Patch*>& patches) const {
  const Impl& I = *p_;
  const Source& S = *src_;
  const int nv = I.nv, nd = S.nd;
  std::vector<char> mapped(S.ne, 0);
  for (const Patch* p : patches)
    for (int e : p->elems) mapped[e] = 1;
  double mx = 0;
  for (const Patch* p : patches)
    for (int e : p->elems)
      for (int j = 0; j < nv; ++j) {
        const int n = S.nbr[e * nv + j];
        if (n >= 0 && mapped[n]) continue;
        if (n < 0 && S.dim == 2 && S.fattr[e * nv + j] == S.axis_attr) continue;
        if (n >= 0 && I.law_of(n).kind == RegionLaw::Superconductor) continue;
        for (int k : I.frnodes[j]) mx = std::max(mx, norm3(&p->bstar[(static_cast<size_t>(e) * nd + k) * 3]));
      }
  return mx;
}

json Mapper::interface(const Patch& a, const Grid& ga, const double* Ba, const Patch& b, const Grid& gb,
                       const double* Bb) const {
  const Impl& I = *p_;
  const Source& S = *src_;
  const int dim = S.dim, nv = I.nv;
  const double scale = std::max(a.B_patch, b.B_patch);
  long count = 0;
  double s2 = 0, mx = 0;
  for (int e : a.elems)
    for (int j = 0; j < nv; ++j) {
      const int n = S.nbr[e * nv + j];
      if (n < 0 || !b.in[n]) continue;
      double x[3], J[9], nrm[3] = {0, 0, 0};
      I.map(e, I.fmid[j].data(), x, J);
      std::vector<int> verts;
      for (int k = 0; k < nv; ++k)
        if (k != j) verts.push_back(k);
      // tangents of the reference face, mapped
      double t[2][3] = {{0, 0, 0}, {0, 0, 0}};
      for (int q = 0; q < dim - 1; ++q)
        for (int c = 0; c < dim; ++c) {
          const double dr = S.gref[verts[q + 1] * dim + c] - S.gref[verts[0] * dim + c];
          for (int r = 0; r < dim; ++r) t[q][r] += J[r * dim + c] * dr;
        }
      if (dim == 2) {
        nrm[0] = t[0][1];
        nrm[1] = -t[0][0];
      } else {
        nrm[0] = t[0][1] * t[1][2] - t[0][2] * t[1][1];
        nrm[1] = t[0][2] * t[1][0] - t[0][0] * t[1][2];
        nrm[2] = t[0][0] * t[1][1] - t[0][1] * t[1][0];
      }
      const double nn = norm3(nrm);
      double va[3], vb[3];
      ga.interp(Ba, x, va);
      gb.interp(Bb, x, vb);
      double dn = 0;
      for (int c = 0; c < dim; ++c) dn += nrm[c] / nn * (va[c] - vb[c]);
      const double m = scale > 0 ? std::fabs(dn) / scale : 0.0;
      ++count;
      s2 += m * m;
      mx = std::max(mx, m);
    }
  if (count == 0) return nullptr;
  return {{"points", count}, {"normal_mismatch_rms", std::sqrt(s2 / count)}, {"normal_mismatch_max", mx}};
}

std::unique_ptr<Patch> Mapper::patch(const std::vector<std::string>& regs, int component) const {
  const Impl& I = *p_;
  const Source& S = *src_;
  std::vector<char> in_group(S.ne, 0);
  for (int e = 0; e < S.ne; ++e) in_group[e] = std::find(regs.begin(), regs.end(), I.rname[I.reg[e]]) != regs.end();
  int nc = 0;
  const std::vector<int> comp = components(I, S, in_group, nc);
  AXMP_REQUIRE(component >= 0 && component < nc,
               strcat("internal: the regions ", joined(regs), " have ", nc, " component(s), not ", component + 1));
  std::vector<int> elems;
  for (int e = 0; e < S.ne; ++e)
    if (comp[e] == component) elems.push_back(e);
  return std::make_unique<Patch>(*this, regs, component, std::move(elems));
}

// ============================================================================================ the axisymmetric source
namespace {

class AxiSource : public Source {
 public:
  explicit AxiSource(const AxiModel& m) : m_(m) {
    AXMP_REQUIRE(m.solved(), "solve the design before exporting a field map");
    const axi::Space& sp = m.space();
    const mfem::Mesh& mesh = sp.mesh;
    auto& mm = const_cast<mfem::Mesh&>(mesh);
    dim = 2;
    order = sp.order;
    ne = mesh.GetNE();
    attr.resize(ne);
    for (int e = 0; e < ne; ++e) attr[e] = mesh.GetAttribute(e);
    region_of_attr = m.mesh_bundle().region_of_attr;
    axis_attr = axi::kAxisAttr;
    const spec::Design& d = m.design();
    // the curves across which the static field jumps: superconducting sheets and thin magnetic layers; a sheet of a
    // normal conductor is transparent to the static field and is not an interface
    for (auto& [name, a] : m.mesh_bundle().attr_of_curve) {
      const spec::Curve* c = d.find_curve(name);
      if (!c) continue;
      const auto mat = d.materials.find(c->material);
      const bool sc = mat != d.materials.end() && mat->second.kind == spec::MaterialKind::Superconductor;
      if (sc || c->thin_layer()) curve_of_attr[a] = name;
    }
    for (auto& [a, name] : region_of_attr) {
      RegionLaw L;
      const spec::Material* mat = nullptr;
      if (d.find_region(name)) mat = &d.material_of_region(name);
      if (mat) {
        switch (mat->kind) {
          case spec::MaterialKind::Linear: L.mu_r = mat->mu_r; break;
          case spec::MaterialKind::Iron: L.kind = RegionLaw::Nonlinear; L.material = mat->name; break;
          case spec::MaterialKind::Superconductor: L.kind = RegionLaw::Superconductor; break;
          default: break;  // vacuum and normal conductors: mu_0
        }
      }
      law[name] = L;
    }
    // geometry
    const mfem::GridFunction* nodes = mesh.GetNodes();
    static mfem::Linear2DFiniteElement p1;
    const mfem::FiniteElement* gfe = nodes ? nodes->FESpace()->GetFE(0) : &p1;
    ng = gfe->GetDof();
    const mfem::IntegrationRule& gn = gfe->GetNodes();
    for (int k = 0; k < ng; ++k) gref.insert(gref.end(), {gn.IntPoint(k).x, gn.IntPoint(k).y});
    gx.resize(static_cast<size_t>(ne) * ng * 2);
    mfem::Array<int> vd;
    for (int e = 0; e < ne; ++e) {
      if (nodes) {
        nodes->FESpace()->GetElementVDofs(e, vd);
        for (int k = 0; k < ng; ++k) {
          gx[(static_cast<size_t>(e) * ng + k) * 2] = (*nodes)(vd[k]);
          gx[(static_cast<size_t>(e) * ng + k) * 2 + 1] = (*nodes)(vd[ng + k]);
        }
      } else {
        mesh.GetElementVertices(e, vd);
        for (int k = 0; k < 3; ++k) {
          gx[(static_cast<size_t>(e) * 3 + k) * 2] = mesh.GetVertex(vd[k])[0];
          gx[(static_cast<size_t>(e) * 3 + k) * 2 + 1] = mesh.GetVertex(vd[k])[1];
        }
      }
    }
    vol.resize(ne);
    for (int e = 0; e < ne; ++e) vol[e] = mm.GetElementVolume(e);
    // topology: local face j is the edge opposite vertex j
    nbr.assign(static_cast<size_t>(ne) * 3, -1);
    fattr.assign(static_cast<size_t>(ne) * 3, 0);
    mfem::Array<int> ev, edges, ori, fv;
    for (int e = 0; e < ne; ++e) {
      mesh.GetElementVertices(e, ev);
      mesh.GetElementEdges(e, edges, ori);
      for (int f : edges) {
        mesh.GetFaceVertices(f, fv);
        int j = -1;
        for (int k = 0; k < 3; ++k)
          if (ev[k] != fv[0] && ev[k] != fv[1]) j = k;
        int e1, e2;
        mesh.GetFaceElements(f, &e1, &e2);
        nbr[e * 3 + j] = e1 == e ? e2 : e1;
        fattr[e * 3 + j] = sp.face_bdr_attr[f];
      }
    }
    // recovery space: the solution's H1 space
    const mfem::FiniteElement* fe = sp.fes->GetFE(0);
    nd = fe->GetDof();
    const mfem::IntegrationRule& nn = fe->GetNodes();
    for (int k = 0; k < nd; ++k) nref.insert(nref.end(), {nn.IntPoint(k).x, nn.IntPoint(k).y});
    field_ = Lagrange(2, nref);
    ndofs = sp.ndof();
    dofs.resize(static_cast<size_t>(ne) * nd);
    for (int e = 0; e < ne; ++e) {
      const auto& d0 = sp.edofs(e);
      for (int k = 0; k < nd; ++k) dofs[static_cast<size_t>(e) * nd + k] = d0[k];
    }
    dof_x.resize(static_cast<size_t>(ndofs) * 2);
    for (int i = 0; i < ndofs; ++i) {
      dof_x[i * 2] = sp.r[i];
      dof_x[i * 2 + 1] = sp.z[i];
    }
    rtol = sp.rtol;
    solution_dofs = ndofs;
  }
  void Bh(int e, const double* xi, const double* x, const double* J, double B[3]) const override {
    thread_local axi::PointEval pe;
    const int n = field_.n;
    pe.N.resize(n);
    pe.dN.resize(2 * n);
    double dref[128];
    field_.eval(xi, pe.N.data(), dref);
    const double det = J[0] * J[3] - J[1] * J[2], id = 1.0 / det;
    for (int i = 0; i < n; ++i) {
      const double a = dref[2 * i], b = dref[2 * i + 1];
      pe.dN[2 * i] = (J[3] * a - J[2] * b) * id;
      pe.dN[2 * i + 1] = (-J[1] * a + J[0] * b) * id;
    }
    pe.nd = n;
    pe.r = std::fabs(x[0]) < rtol ? 0.0 : x[0];
    pe.z = x[1];
    pe.w = 0;
    pe.ip = nullptr;
    const AxiModel::Sample s = m_.eval_pe(e, pe);
    B[0] = s.Br;
    B[1] = s.Bz;
    B[2] = s.Bphi;
  }
  std::vector<double> solution() const override {
    std::vector<double> v;
    if (m_.poloidal_lab()) {
      const mfem::Vector& u = m_.poloidal_lab_solution().u;
      v.insert(v.end(), u.GetData(), u.GetData() + u.Size());
    }
    if (m_.toroidal_lab()) {
      const auto& h = m_.toroidal_lab_solution();
      v.insert(v.end(), h.unit.GetData(), h.unit.GetData() + h.unit.Size());
      v.insert(v.end(), h.G_elem.begin(), h.G_elem.end());
      v.insert(v.end(), h.La.begin(), h.La.end());
      v.insert(v.end(), h.Lb.begin(), h.Lb.end());
    }
    return v;
  }

 private:
  const AxiModel& m_;
  Lagrange field_;
};

}  // namespace

std::unique_ptr<Source> axi_source(const AxiModel& m) { return std::make_unique<AxiSource>(m); }

}  // namespace axmp::fieldmap
