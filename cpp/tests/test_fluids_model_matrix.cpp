// The engine half of the matrix test: every flow and heat cell of the route table (the case list of
// cpp/src/model_matrix.hpp) handed to the engine of its row, on the meshes that the document path hands an engine.
//
// Built in MPI builds only (it links the fluids library).  Two runs of one program:
//   test_fluids_model_matrix               one rank: every case with ranks == 1
//   test_fluids_model_matrix_2ranks        the same program under the MPI launcher with --ranks 2: the cases with
//                                          ranks == 2
// The chemistry cells, whose engine is in the core library, are the every-build half (cpp/tests/test_model_matrix.cpp).
//
// What this file asserts:
//   check_reference_numbering   the instance numbering of model_matrix_fixtures.hpp, on its own;
//   check_fixtures              the four meshes: base (triangles, boundary attributes exactly 1..n, no nodes), sparse
//                               (the master numbers), nodes (the base with a nodes grid function, unchanged geometry),
//                               quad (quadrilaterals numbered as the base);
//   check_coverage_contract     the coverage rule below, on a synthetic set of visits;
//   engine_cannot_express_types the members of the engines' interfaces that hold one value per solve, by their type
//                               (compile time), and that no per-region slot exists beside them (detection idiom);
//   the cells                   each case handed to its engine through exercise(): flow_offered_cells_accepted,
//                               flow_engine_refusals, heat_offered_cells_accepted_through_finalize, heat_engine_refusals,
//                               sparse_mesh_cells, nodes_mesh_cells, quad_mesh_cells, rank_cells;
//   check_every_engine_cell_is_exercised
//                               every case that names a flow or heat engine and meshes is handed to its engine exactly
//                               once per entry of its meshes, in the run of its rank count, and nothing else is.
//
// INDEPENDENT REFERENCES.  The engines (StokesSolver, NavierStokesSolver, TransientNavierStokes, BoussinesqSolver,
// transport::ScalarTransport), reached through a translation from condition kinds to engine conditions written in this
// file, which shares no code with the adapters of the model path; the instance numbering of section 5.4 of the
// interface design note, written again in model_matrix_fixtures.hpp; the engines' headers for the type assertions
// (file and line at each).
#include <mpi.h>

#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fluids/boussinesq.hpp"
#include "axionmultiphysics/fluids/mesh_checks.hpp"
#include "axionmultiphysics/fluids/meshes.hpp"
#include "axionmultiphysics/fluids/navier_stokes.hpp"
#include "axionmultiphysics/fluids/stokes.hpp"
#include "axionmultiphysics/fluids/transient_ns.hpp"
#include "axionmultiphysics/physics_registry.hpp"
#include "axionmultiphysics/transport/scalar_transport.hpp"
#include "../src/model_matrix.hpp"
#include "model_matrix_fixtures.hpp"

namespace fx = axmp::model::fixtures;
namespace fl = axmp::fluids;
namespace tr = axmp::transport;
using axmp::model::CellValue;
using axmp::model::MatrixCase;

static int failures = 0;
static int g_rank = 0, g_size = 1;
#define CHECK(cond, ...)                         \
  do {                                           \
    if (!(cond)) {                               \
      std::printf("FAIL [rank %d]: ", g_rank);   \
      std::printf(__VA_ARGS__);                  \
      std::printf("\n");                         \
      ++failures;                                \
    }                                            \
  } while (0)

namespace {

using Strs = std::vector<std::string>;

bool has(const Strs& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

const std::vector<MatrixCase>& cases() {
  static const std::vector<MatrixCase> all = axmp::model::matrix_cases();
  return all;
}

// The engines of the fluids library, which this program builds; the chemistry engine is the other half's.
bool flow_engine(const std::string& e) {
  return e == "StokesSolver" || e == "NavierStokesSolver" || e == "TransientNavierStokes";
}
bool heat_engine(const std::string& e) { return e == "transport::ScalarTransport" || e == "BoussinesqSolver"; }
bool in_scope(const MatrixCase& c) { return (flow_engine(c.engine) || heat_engine(c.engine)) && !c.meshes.empty(); }

const std::set<std::string> kCalls = {"constructor", "add_bc", "finalize", "first_step", "layout", "solve"};

// ================================================================================================ the fixtures
// The quadrilateral mesh: the cells of the base grid themselves, built with the fluids library's mesh builder (4
// vertices per element, fluids/meshes.hpp) and numbered by the same rule as the base mesh.  No mesher mode produces
// it; an imported mesh can.
fx::Fixture quad_mesh(fx::Geometry g, int n = fx::kCells) {
  const fx::detail::Arrays a = fx::detail::square_arrays(g, n, true);
  fx::Fixture f;
  f.name = "quad";
  f.geometry = g;
  f.mesh = fl::mesh_from_arrays(a.xy, a.elements, 4, a.bdr_segments, a.bdr_attributes);
  fx::detail::renumber_boundary(f.mesh, fx::square_numbering(g));
  f.master = fx::master_numbers(g);
  f.attribute = fx::dense_attributes(g);
  return f;
}

// The fixture of a mesh name of the case list: base, sparse, nodes, quad.
fx::Fixture fixture(const std::string& name, fx::Geometry g) {
  if (name == "base") return fx::base_mesh(g);
  if (name == "sparse") return fx::sparse_mesh(g);
  if (name == "nodes") return fx::nodes_mesh(g);
  if (name == "quad") return quad_mesh(g);
  throw std::logic_error("no fixture '" + name + "'");
}

// The partitioned mesh the engines of the fluids library take, over every rank of `comm`: of this run (one rank, or
// the two of the --ranks 2 run), or of one rank alone for a one-rank reference.  The transport operator takes it as an
// mfem::Mesh: on one rank it is the serial mesh's twin, and on two the operator's rank check sees the partition
// (fluids::require_single_rank returns silently on a serial mesh).
std::unique_ptr<mfem::ParMesh> partitioned(fx::Fixture& f, MPI_Comm comm) {
  return std::make_unique<mfem::ParMesh>(comm, f.mesh);
}

// The geometry class of a case's row ("planar" or "axisymmetric"); the sparse cases, which are not rows, run on both.
std::vector<fx::Geometry> geometries_of(const MatrixCase& c) {
  if (c.row.empty()) return {fx::Geometry::planar, fx::Geometry::axisymmetric};
  const auto* r = axmp::model::registry().route(c.row);
  if (!r || r->geometries.empty()) throw std::logic_error(c.id + ": no geometry class");
  return {r->geometries[0] == "axisymmetric" ? fx::Geometry::axisymmetric : fx::Geometry::planar};
}

// ================================================================================================ handing a case over
// What a cell's problem reports while it runs: the engine call it is about to make, so that a refusal is attributed to
// the call it came from, and the problem's own checks (a solve that must converge), which are never read as a refusal.
struct CellCheck : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Probe {
 public:
  // The problem is about to make `call` (constructor, add_bc, finalize, first_step, layout, solve).
  void enter(const std::string& call) {
    if (!kCalls.count(call)) throw std::logic_error("Probe::enter: unknown call '" + call + "'");
    current_ = call;
    entered_.push_back(call);
  }
  const std::string& current() const { return current_; }
  bool entered(const std::string& call) const { return has(entered_, call); }
  std::string calls() const {
    std::string s;
    for (const auto& c : entered_) s += (s.empty() ? "" : ", ") + c;
    return s.empty() ? std::string("none") : s;
  }
  // A check of the problem itself; throws CellCheck when it fails.
  void require(bool ok, const std::string& what) const {
    if (!ok) throw CellCheck(what);
  }

 private:
  std::string current_;
  Strs entered_;
};

using Problem = std::function<void(Probe&)>;
using Visits = std::map<std::pair<std::string, std::string>, int>;   // (case id, mesh) -> times handed over
Visits g_visits;

// Hands case `c` to its engine on the mesh named `mesh` through `problem`, checks the outcome against the class of the
// case and records the visit.
//   offered          the problem runs to its end without an exception, and it entered the case's call;
//   engine_refuses   the engine throws while the problem is in the case's call, and the message contains the case's
//                    engine_text.
// A CellCheck from the problem fails the cell whatever its class.  Every rank of the run calls it for every case of
// the run.
void exercise(const MatrixCase& c, const std::string& mesh, const Problem& problem) {
  CHECK(in_scope(c) && has(c.meshes, mesh) && c.ranks == g_size,
        "%s: handed to its engine on mesh '%s' at %d rank(s), which the case does not list", c.id.c_str(), mesh.c_str(),
        g_size);
  ++g_visits[{c.id, mesh}];
  Probe p;
  bool threw = false, own = false;
  std::string what;
  try {
    problem(p);
  } catch (const CellCheck& e) {
    own = true;
    what = e.what();
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  if (own) {
    CHECK(false, "%s on %s: the problem's own check failed (in '%s'): %s", c.id.c_str(), mesh.c_str(),
          p.current().c_str(), what.c_str());
    return;
  }
  switch (c.cls) {
    case CellValue::offered:
      CHECK(!threw, "%s on %s: offered, but the engine refused it in '%s': %s", c.id.c_str(), mesh.c_str(),
            p.current().c_str(), what.c_str());
      CHECK(threw || p.entered(c.call), "%s on %s: offered through '%s', but the problem never made that call (calls: %s)",
            c.id.c_str(), mesh.c_str(), c.call.c_str(), p.calls().c_str());
      break;
    case CellValue::engine_refuses:
      CHECK(threw, "%s on %s: the case says the engine refuses it in '%s' (\"%s\"), but the engine accepted it (calls: %s)",
            c.id.c_str(), mesh.c_str(), c.call.c_str(), c.engine_text.c_str(), p.calls().c_str());
      if (threw) {
        CHECK(p.current() == c.call, "%s on %s: refused in '%s', the case names '%s': %s", c.id.c_str(), mesh.c_str(),
              p.current().c_str(), c.call.c_str(), what.c_str());
        CHECK(what.find(c.engine_text) != std::string::npos, "%s on %s: the refusal \"%s\" does not contain \"%s\"",
              c.id.c_str(), mesh.c_str(), what.c_str(), c.engine_text.c_str());
      }
      break;
    default:
      CHECK(false, "%s: class %s has no engine column", c.id.c_str(), axmp::model::to_string(c.cls));
  }
}

// ================================================================================================ the coverage rule
// One visit of the run: a case of the case list, on one of its meshes.
struct Visit {
  const MatrixCase* c;
  std::string mesh;
};

// The groups of cells this program is written in, one per visit.  Rank cells are the two-rank run's; the sparse,
// nodes and quadrilateral meshes have a group each; on the base mesh the group is by engine family and class.
enum class Group { flow_offered, flow_refusals, heat_offered, heat_refusals, sparse, nodes, quad, ranks };

const char* to_string(Group g) {
  switch (g) {
    case Group::flow_offered: return "flow_offered_cells_accepted";
    case Group::flow_refusals: return "flow_engine_refusals";
    case Group::heat_offered: return "heat_offered_cells_accepted_through_finalize";
    case Group::heat_refusals: return "heat_engine_refusals";
    case Group::sparse: return "sparse_mesh_cells";
    case Group::nodes: return "nodes_mesh_cells";
    case Group::quad: return "quad_mesh_cells";
    case Group::ranks: return "rank_cells";
  }
  return "";
}

Group group_of(const MatrixCase& c, const std::string& mesh) {
  if (c.ranks == 2) return Group::ranks;
  if (mesh == "sparse") return Group::sparse;
  if (mesh == "nodes") return Group::nodes;
  if (mesh == "quad") return Group::quad;
  if (mesh != "base") throw std::logic_error(c.id + ": unknown mesh '" + mesh + "'");
  const bool offered = c.cls == CellValue::offered;
  if (flow_engine(c.engine)) return offered ? Group::flow_offered : Group::flow_refusals;
  return offered ? Group::heat_offered : Group::heat_refusals;
}

// Every visit a run on `ranks` ranks must make: each case that names a flow or heat engine and meshes, once per entry
// of its meshes, when its rank count is the run's.
std::vector<Visit> expected_visits(const std::vector<MatrixCase>& all, int ranks) {
  std::vector<Visit> out;
  for (const auto& c : all)
    if (in_scope(c) && c.ranks == ranks)
      for (const auto& m : c.meshes) out.push_back({&c, m});
  return out;
}

std::vector<Visit> visits_of(Group g, int ranks) {
  std::vector<Visit> out;
  for (const auto& v : expected_visits(cases(), ranks))
    if (group_of(*v.c, v.mesh) == g) out.push_back(v);
  return out;
}

// What is wrong with `visits` against `expected`: a visit never made, made more than once, or made but not expected.
Strs coverage_failures(const std::vector<Visit>& expected, const Visits& visits) {
  Strs out;
  std::set<std::pair<std::string, std::string>> want;
  for (const auto& v : expected) want.insert({v.c->id, v.mesh});
  for (const auto& key : want) {
    auto it = visits.find(key);
    const int n = it == visits.end() ? 0 : it->second;
    if (n == 0) out.push_back("never handed to its engine: " + key.first + " on " + key.second);
    else if (n > 1) out.push_back("handed to its engine " + std::to_string(n) + " times: " + key.first + " on " + key.second);
  }
  for (const auto& [key, n] : visits)
    if (n > 0 && !want.count(key))
      out.push_back("handed to its engine but not a visit of this run: " + key.first + " on " + key.second);
  return out;
}

// ================================================================================================ the checks
void check_reference_numbering() {
  // section 5.4: sorted by master attribute, numbered 1..n with no hole; the axis (900) first where it exists
  {
    const auto m = fx::instance_numbering({900, 2001, 2003});
    CHECK((m == std::map<int, int>{{900, 1}, {2001, 2}, {2003, 3}}), "numbering of {900, 2001, 2003}: want 900->1, 2001->2, 2003->3");
  }
  // the input is the attribute of every boundary element: repeats and any order
  {
    const auto m = fx::instance_numbering({2003, 2001, 2003, 900, 2001, 2001});
    CHECK((m == std::map<int, int>{{900, 1}, {2001, 2}, {2003, 3}}), "numbering with repeats and out of order");
  }
  CHECK((fx::instance_numbering({2004}) == std::map<int, int>{{2004, 1}}), "one atom is numbered 1");
  CHECK(fx::instance_numbering({}).empty(), "no boundary, no numbering");
  // no hole: the values are exactly 1..n whatever the gaps in the master numbers
  {
    const auto m = fx::instance_numbering({2009, 2001, 901, 900, 2100, 3});
    std::vector<int> values;
    for (const auto& [k, v] : m) values.push_back(v);
    CHECK((values == std::vector<int>{1, 2, 3, 4, 5, 6}), "the instance attributes are 1..n in the order of the master numbers");
    CHECK(m.at(3) == 1 && m.at(900) == 2 && m.at(2100) == 6, "sorted by master attribute");
  }
  bool refused = false;
  try {
    fx::instance_numbering({2001, 0});
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused, "a master attribute below 1 is refused");
  // the master numbers of the square (section 5.3): axis 900; the named atoms 2001, ... in the lexicographic order of
  // their names (bottom < left < right < top)
  CHECK((fx::master_numbers(fx::Geometry::planar) ==
         std::map<std::string, int>{{"bottom", 2001}, {"left", 2002}, {"right", 2003}, {"top", 2004}}),
        "planar master numbers");
  CHECK((fx::master_numbers(fx::Geometry::axisymmetric) ==
         std::map<std::string, int>{{"axis", 900}, {"bottom", 2001}, {"right", 2002}, {"top", 2003}}),
        "axisymmetric master numbers");
  CHECK((fx::dense_attributes(fx::Geometry::planar) ==
         std::map<std::string, int>{{"bottom", 1}, {"left", 2}, {"right", 3}, {"top", 4}}),
        "planar instance attributes");
  CHECK((fx::dense_attributes(fx::Geometry::axisymmetric) ==
         std::map<std::string, int>{{"axis", 1}, {"bottom", 2}, {"right", 3}, {"top", 4}}),
        "axisymmetric instance attributes: the axis first");
}

// The sorted distinct boundary attributes of a mesh, from its boundary elements (not from bdr_attributes, which a
// stale SetAttributes would leave behind) and from bdr_attributes; both must agree.
std::vector<int> boundary_attributes(const mfem::Mesh& m) {
  std::set<int> s;
  for (int be = 0; be < m.GetNBE(); ++be) s.insert(m.GetBdrAttribute(be));
  std::vector<int> listed(m.bdr_attributes.begin(), m.bdr_attributes.end());
  std::sort(listed.begin(), listed.end());
  std::vector<int> from_elements(s.begin(), s.end());
  CHECK(listed == from_elements, "bdr_attributes disagrees with the boundary elements");
  return from_elements;
}

// Every boundary element of `f` lies on the side whose attribute it carries, and each side has n elements.
void check_sides(const fx::Fixture& f, int n) {
  std::map<std::string, int> count;
  mfem::Array<int> vs;
  for (int be = 0; be < f.mesh.GetNBE(); ++be) {
    f.mesh.GetBdrElementVertices(be, vs);
    const std::string side = fx::side_of_segment(f.geometry, f.mesh.GetVertex(vs[0]), f.mesh.GetVertex(vs[1]));
    CHECK(!side.empty(), "%s (%s): boundary element %d is on no side", f.name.c_str(), fx::to_string(f.geometry), be);
    if (side.empty()) continue;
    ++count[side];
    CHECK(f.mesh.GetBdrAttribute(be) == f.attr(side), "%s (%s): boundary element %d on '%s' has attribute %d, want %d",
          f.name.c_str(), fx::to_string(f.geometry), be, side.c_str(), f.mesh.GetBdrAttribute(be), f.attr(side));
  }
  for (const auto& s : fx::side_names(f.geometry))
    CHECK(count[s] == n, "%s (%s): %d boundary elements on '%s', want %d", f.name.c_str(), fx::to_string(f.geometry),
          count[s], s.c_str(), n);
}

void check_fixtures() {
  const int n = fx::kCells;
  for (fx::Geometry g : {fx::Geometry::planar, fx::Geometry::axisymmetric}) {
    const char* gn = fx::to_string(g);
    std::vector<int> dense;
    for (const auto& [side, a] : fx::dense_attributes(g)) dense.push_back(a);
    std::sort(dense.begin(), dense.end());
    std::vector<int> masters;
    for (const auto& [side, a] : fx::master_numbers(g)) masters.push_back(a);
    std::sort(masters.begin(), masters.end());
    CHECK((dense == std::vector<int>{1, 2, 3, 4}), "%s: four sides numbered 1..4", gn);

    // base: triangles only, boundary attributes exactly 1..n, no nodes
    const fx::Fixture base = fx::base_mesh(g);
    CHECK(base.mesh.GetNE() == 2 * n * n, "base (%s): %d elements, want %d", gn, base.mesh.GetNE(), 2 * n * n);
    bool tri = true;
    for (int e = 0; e < base.mesh.GetNE(); ++e) tri = tri && base.mesh.GetElementGeometry(e) == mfem::Geometry::TRIANGLE;
    CHECK(tri, "base (%s): triangles only", gn);
    CHECK(base.mesh.GetNodes() == nullptr, "base (%s): no nodes grid function (an instance mesh of geometry order 1)", gn);
    CHECK(boundary_attributes(base.mesh) == dense, "base (%s): the boundary attributes are exactly 1..n", gn);
    CHECK(base.attribute == fx::dense_attributes(g), "base (%s): its side map is the instance numbering", gn);
    check_sides(base, n);
    // the Taylor-Hood mesh condition the document path checks before a flow solve
    const fl::InfSupMeshReport th = fl::check_taylor_hood_mesh(base.mesh);
    CHECK(th.ok() && th.checked_elements == 2 * n * n, "base (%s): %zu triangles have all their vertices on the boundary",
          gn, th.all_boundary_vertices.size());

    // sparse: the same triangles with the master numbers left on
    const fx::Fixture sparse = fx::sparse_mesh(g);
    CHECK(sparse.mesh.GetNE() == base.mesh.GetNE() && sparse.mesh.GetNV() == base.mesh.GetNV(), "sparse (%s): the base's elements", gn);
    CHECK(boundary_attributes(sparse.mesh) == masters, "sparse (%s): the boundary attributes are the master numbers", gn);
    CHECK(sparse.mesh.GetNodes() == nullptr, "sparse (%s): no nodes", gn);
    check_sides(sparse, n);
    if (g == fx::Geometry::axisymmetric)
      CHECK(sparse.attr("axis") == fx::kAxisMaster, "sparse (%s): the axis keeps 900", gn);

    // nodes: the base with a nodes grid function; the vertices, the attributes and the (straight) geometry unchanged
    fx::Fixture nodes = fx::nodes_mesh(g);
    CHECK(nodes.mesh.GetNodes() != nullptr, "nodes (%s): a nodes grid function", gn);
    CHECK(nodes.mesh.GetNV() == base.mesh.GetNV() && nodes.mesh.GetNE() == base.mesh.GetNE(), "nodes (%s): the base's mesh", gn);
    double dv = 0;
    for (int v = 0; v < base.mesh.GetNV(); ++v)
      for (int k = 0; k < 2; ++k) dv = std::max(dv, std::fabs(nodes.mesh.GetVertex(v)[k] - base.mesh.GetVertex(v)[k]));
    CHECK(dv == 0.0, "nodes (%s): the vertices moved by %g", gn, dv);
    CHECK(boundary_attributes(nodes.mesh) == dense, "nodes (%s): the base's boundary attributes", gn);
    check_sides(nodes, n);
    // straight: the element map sends the reference vertices to the vertices and the edge midpoints to the chord
    // midpoints (a P2 interpolation of an affine map is the map)
    double dg = 0;
    {
      mfem::Mesh& m = nodes.mesh;
      const double ref[6][2] = {{0, 0}, {1, 0}, {0, 1}, {0.5, 0}, {0.5, 0.5}, {0, 0.5}};
      mfem::Array<int> ev;
      for (int e = 0; e < m.GetNE(); ++e) {
        m.GetElementVertices(e, ev);
        mfem::ElementTransformation* T = m.GetElementTransformation(e);
        const double* p[3] = {m.GetVertex(ev[0]), m.GetVertex(ev[1]), m.GetVertex(ev[2])};
        for (int q = 0; q < 6; ++q) {
          mfem::IntegrationPoint ip;
          ip.Set2(ref[q][0], ref[q][1]);
          double x[2];
          mfem::Vector xv(x, 2);
          T->Transform(ip, xv);
          const double l1 = ref[q][0], l2 = ref[q][1], l0 = 1 - l1 - l2;
          for (int k = 0; k < 2; ++k) dg = std::max(dg, std::fabs(x[k] - (l0 * p[0][k] + l1 * p[1][k] + l2 * p[2][k])));
        }
      }
    }
    CHECK(dg < 1e-14, "nodes (%s): the element map is not the straight one (%g)", gn, dg);

    // quad: every element a quadrilateral, numbered as the base
    const fx::Fixture quad = quad_mesh(g);
    CHECK(quad.mesh.GetNE() == n * n, "quad (%s): %d elements, want %d", gn, quad.mesh.GetNE(), n * n);
    bool all_quad = true;
    for (int e = 0; e < quad.mesh.GetNE(); ++e)
      all_quad = all_quad && quad.mesh.GetElementGeometry(e) == mfem::Geometry::SQUARE;
    CHECK(all_quad, "quad (%s): every element a quadrilateral", gn);
    CHECK(quad.mesh.GetNodes() == nullptr, "quad (%s): no nodes", gn);
    CHECK(boundary_attributes(quad.mesh) == dense, "quad (%s): the boundary attributes are exactly 1..n", gn);
    CHECK(quad.attribute == base.attribute, "quad (%s): numbered by the base's rule", gn);
    check_sides(quad, n);
  }
}

void check_coverage_contract() {
  // the rule itself, on synthetic visits of the one-rank and the two-rank run
  for (int ranks : {1, 2}) {
    const auto want = expected_visits(cases(), ranks);
    CHECK(!want.empty(), "the %d-rank run has no visit to make", ranks);
    if (want.empty()) continue;
    Visits complete;
    for (const auto& v : want) complete[{v.c->id, v.mesh}] = 1;
    CHECK(coverage_failures(want, complete).empty(), "%d ranks: a complete set of visits is reported incomplete", ranks);
    // one visit left out: on the two-rank run, the transient Navier-Stokes cell
    std::pair<std::string, std::string> dropped = {want.front().c->id, want.front().mesh};
    for (const auto& v : want)
      if (v.c->engine == "TransientNavierStokes") dropped = {v.c->id, v.mesh};
    Visits missing = complete;
    missing.erase(dropped);
    const Strs m = coverage_failures(want, missing);
    CHECK(m.size() == 1 && m[0].find(dropped.first) != std::string::npos && m[0].find("never") != std::string::npos,
          "%d ranks: a visit left out (%s) is not reported", ranks, dropped.first.c_str());
    Visits twice = complete;
    ++twice[{want.back().c->id, want.back().mesh}];
    const Strs t = coverage_failures(want, twice);
    CHECK(t.size() == 1 && t[0].find("2 times") != std::string::npos, "%d ranks: a visit made twice is not reported", ranks);
    Visits extra = complete;
    extra[{want.front().c->id, "quad_that_the_case_does_not_list"}] = 1;
    CHECK(coverage_failures(want, extra).size() == 1, "%d ranks: a visit outside the case list is not reported", ranks);
  }
  // every visit belongs to one group, and the two runs together cover every case with a flow or heat engine column
  size_t n1 = 0, n2 = 0;
  for (const auto& c : cases())
    if (in_scope(c)) (c.ranks == 1 ? n1 : n2) += c.meshes.size();
  CHECK(expected_visits(cases(), 1).size() == n1 && expected_visits(cases(), 2).size() == n2,
        "the visits of the two runs are not the engine columns of the case list");
  for (const auto& c : cases())
    if (in_scope(c)) CHECK(c.ranks == 1 || c.ranks == 2, "%s: ranks %d", c.id.c_str(), c.ranks);
}

// ================================================================================================ engine_cannot_express
// A member detector: has_member_NAME<T>::value is true when T has a member called NAME of any kind (data member,
// member function, overloaded or not, static, type).  It works by ambiguity: Derived inherits NAME from T and from
// Fallback, so &Derived::NAME is ill-formed exactly when T has one, and the first overload drops out.
#define AXMP_MEMBER_DETECTOR(NAME)                                                          \
  template <class T>                                                                        \
  struct has_member_##NAME {                                                                \
    struct Fallback {                                                                       \
      int NAME;                                                                             \
    };                                                                                      \
    struct Derived : T, Fallback {};                                                        \
    template <class U, U>                                                                   \
    struct Check;                                                                           \
    template <class U>                                                                      \
    static std::false_type test(Check<int Fallback::*, &U::NAME>*);                         \
    template <class U>                                                                      \
    static std::true_type test(...);                                                        \
    static constexpr bool value = decltype(test<Derived>(nullptr))::value;                  \
  }

// the per-region slots that would make a one-value member region-wise
AXMP_MEMBER_DETECTOR(set_diffusivity_by_element);
AXMP_MEMBER_DETECTOR(set_source_by_element);
AXMP_MEMBER_DETECTOR(set_capacity_by_element);
AXMP_MEMBER_DETECTOR(set_velocity_by_element);
AXMP_MEMBER_DETECTOR(scale_by_element);
AXMP_MEMBER_DETECTOR(fn_by_element);
AXMP_MEMBER_DETECTOR(rho_by_element);
AXMP_MEMBER_DETECTOR(cp_by_element);
AXMP_MEMBER_DETECTOR(beta_g_by_element);
AXMP_MEMBER_DETECTOR(k_by_element);
AXMP_MEMBER_DETECTOR(heat_source_by_element);
// a gravity vector beside the vertical sign
AXMP_MEMBER_DETECTOR(gravity);
AXMP_MEMBER_DETECTOR(gravity_m_s2);
AXMP_MEMBER_DETECTOR(gravity_vector);
AXMP_MEMBER_DETECTOR(gravity_direction);
AXMP_MEMBER_DETECTOR(e_up);
// an advective form of the coupled energy equation
AXMP_MEMBER_DETECTOR(form);
AXMP_MEMBER_DETECTOR(advective_form);
AXMP_MEMBER_DETECTOR(energy_form);
// positive controls: members that exist
AXMP_MEMBER_DETECTOR(set_diffusivity);
AXMP_MEMBER_DETECTOR(k);
AXMP_MEMBER_DETECTOR(scale);

// The detector sees members that exist, of both kinds (a member function, a data member).
static_assert(has_member_set_diffusivity<tr::ScalarTransport>::value, "the member detector misses a member function");
static_assert(has_member_k<fl::BoussinesqOptions>::value, "the member detector misses a data member");
static_assert(has_member_scale<tr::Velocity>::value, "the member detector misses a data member");
static_assert(has_member_energy_form<fl::BoussinesqDiagnostics>::value, "the member detector misses a string member");
static_assert(has_member_form<tr::TransportOptions>::value, "the member detector misses an enum member");

// ---- the data of the transport operator are functions of position only (transport/scalar_transport.hpp l.44 to 50):
// a batch callback is given points, never the element they lie in
static_assert(std::is_same_v<tr::BatchFn, std::function<void(const double*, int, double*)>>,
              "scalar_transport.hpp l.44: BatchFn is no longer a function of position only");
static_assert(std::is_same_v<tr::BatchVec, std::function<void(const double*, int, int, double*)>>,
              "scalar_transport.hpp l.45: BatchVec is no longer a function of position only");
static_assert(std::is_same_v<decltype(tr::ScalarData::value), double> &&
                  std::is_same_v<decltype(tr::ScalarData::fn), tr::BatchFn>,
              "scalar_transport.hpp l.47 to 50: ScalarData is no longer one number or one function of position");

// ---- thermal.k_W_mK, one expression on the transport rows: ScalarTransport::set_diffusivity takes one ScalarData
// (scalar_transport.hpp l.101)
static_assert(std::is_same_v<decltype(&tr::ScalarTransport::set_diffusivity),
                             void (tr::ScalarTransport::*)(const tr::ScalarData&, const tr::BatchVec&)>,
              "scalar_transport.hpp l.101: set_diffusivity changed: re-read the one-expression cells of thermal.k_W_mK");
static_assert(!has_member_set_diffusivity_by_element<tr::ScalarTransport>::value,
              "ScalarTransport has set_diffusivity_by_element: the conductivity is region-wise on the transport rows; "
              "change their thermal.k_W_mK cells to per region and delete this detector");
// ---- the heat source, one expression on the transport rows: set_source takes one ScalarData (l.103)
static_assert(std::is_same_v<decltype(&tr::ScalarTransport::set_source),
                             void (tr::ScalarTransport::*)(const tr::ScalarData&)>,
              "scalar_transport.hpp l.103: set_source changed: re-read the one-expression cells of heat_source");
static_assert(!has_member_set_source_by_element<tr::ScalarTransport>::value,
              "ScalarTransport has set_source_by_element: the heat source is region-wise on the transport rows; change "
              "their heat_source cells to per region and delete this detector");
// ---- one capacity per solve: set_capacity takes one double (l.99)
static_assert(std::is_same_v<decltype(&tr::ScalarTransport::set_capacity), void (tr::ScalarTransport::*)(double)>,
              "scalar_transport.hpp l.99: set_capacity no longer takes one number");
static_assert(!has_member_set_capacity_by_element<tr::ScalarTransport>::value,
              "ScalarTransport has a per-element capacity: re-read the uniform density and cp cells of the advected "
              "heat rows");
// ---- density and thermal.cp_J_kgK, uniform on the advected transport rows: the advective velocity is
// beta = scale * (b or u_h) with one double scale (l.62)
static_assert(std::is_same_v<decltype(tr::Velocity::scale), double>,
              "scalar_transport.hpp l.62: Velocity::scale is no longer one number");
static_assert(!has_member_scale_by_element<tr::Velocity>::value,
              "Velocity has a per-element scale: re-read the uniform density and cp cells of the advected heat rows");
// ---- prescribed_advection.velocity, one expression covering the instance: one callable velocity per operator
// (Velocity::fn, l.59; set_velocity, l.102)
static_assert(std::is_same_v<decltype(tr::Velocity::fn), tr::BatchVec>,
              "scalar_transport.hpp l.59: Velocity::fn is no longer one function of position");
static_assert(std::is_same_v<decltype(&tr::ScalarTransport::set_velocity),
                             void (tr::ScalarTransport::*)(const tr::Velocity&)>,
              "scalar_transport.hpp l.102: set_velocity no longer takes one velocity");
static_assert(!has_member_fn_by_element<tr::Velocity>::value && !has_member_set_velocity_by_element<tr::ScalarTransport>::value,
              "the transport operator has a per-element velocity: re-read the prescribed_advection.velocity cells");
// ---- density, uniform on the Navier-Stokes rows and the Boussinesq rows: NavierStokesOptions::rho is one double
// (navier_stokes.hpp l.72)
static_assert(std::is_same_v<decltype(fl::NavierStokesOptions::rho), double>,
              "navier_stokes.hpp l.72: NavierStokesOptions::rho is no longer one number");
static_assert(!has_member_rho_by_element<fl::NavierStokesOptions>::value,
              "NavierStokesOptions has a per-element density: re-read the uniform density cells");
// ---- the Boussinesq rows: cp and beta g are one double each, k and the heat source one ScalarData each
// (boussinesq.hpp l.33, 34, 38, 40)
static_assert(std::is_same_v<decltype(fl::BoussinesqOptions::cp), double>, "boussinesq.hpp l.33: cp is no longer one number");
static_assert(std::is_same_v<decltype(fl::BoussinesqOptions::beta_g), double>,
              "boussinesq.hpp l.34: beta_g is no longer one number");
static_assert(std::is_same_v<decltype(fl::BoussinesqOptions::k), tr::ScalarData>,
              "boussinesq.hpp l.38: k is no longer one ScalarData");
static_assert(std::is_same_v<decltype(fl::BoussinesqOptions::heat_source), tr::ScalarData>,
              "boussinesq.hpp l.40: heat_source is no longer one ScalarData");
static_assert(!has_member_cp_by_element<fl::BoussinesqOptions>::value && !has_member_beta_g_by_element<fl::BoussinesqOptions>::value,
              "BoussinesqOptions has a per-element cp or beta g: re-read the uniform cells of the Boussinesq rows");
static_assert(!has_member_k_by_element<fl::BoussinesqOptions>::value &&
                  !has_member_heat_source_by_element<fl::BoussinesqOptions>::value,
              "BoussinesqOptions has a per-element k or heat source: re-read the one-expression cells of the Boussinesq "
              "rows");
// ---- gravity along the vertical axis only (fluid.gravity_direction): the buoyancy is up_sign * beta_g along e_up,
// e_up = e_y planar or e_z axisymmetric, an int sign and a double (boussinesq.hpp l.34 and 37), and no gravity vector
static_assert(std::is_same_v<decltype(fl::BoussinesqOptions::up_sign), int>,
              "boussinesq.hpp l.37: up_sign is no longer an int sign");
static_assert(!has_member_gravity<fl::BoussinesqOptions>::value && !has_member_gravity_m_s2<fl::BoussinesqOptions>::value &&
                  !has_member_gravity_vector<fl::BoussinesqOptions>::value &&
                  !has_member_gravity_direction<fl::BoussinesqOptions>::value &&
                  !has_member_e_up<fl::BoussinesqOptions>::value,
              "BoussinesqOptions has a gravity vector: re-read the cell behind fluid.gravity_direction");
// ---- the energy equation of the Boussinesq solver has one advective form: BoussinesqOptions has no form member
// (boussinesq.hpp l.30 to 52) and the constructor sets the convective form (boussinesq.cpp l.255)
static_assert(!has_member_form<fl::BoussinesqOptions>::value && !has_member_advective_form<fl::BoussinesqOptions>::value &&
                  !has_member_energy_form<fl::BoussinesqOptions>::value,
              "BoussinesqOptions has an advective form: re-read the advective_form rule of the Boussinesq rows");

// The members the assertions above are about, by the names the cases give them (MatrixCase::member).
const std::set<std::string> kAssertedMembers = {
    "NavierStokesOptions::rho",       "Velocity::scale",          "Velocity::fn",
    "ScalarTransport::set_capacity",  "ScalarTransport::set_diffusivity", "ScalarTransport::set_source",
    "BoussinesqOptions::cp",          "BoussinesqOptions::beta_g", "BoussinesqOptions::k",
    "BoussinesqOptions::heat_source", "BoussinesqOptions::up_sign",
    "BoussinesqOptions: no advective form",   // the absence of a member, asserted by the detectors above
};

// The engine_cannot_express cells: there is no problem to hand over (the engine's interface has one slot, or none
// for an option of which the engine has one value), so the cell is its member's type, or its member's absence,
// asserted above at compile time.  Here: every such cell of a flow or heat engine names a member that is asserted.
void engine_cannot_express_types() {
  int n = 0;
  for (const auto& c : cases()) {
    if (c.cls != CellValue::engine_cannot_express || !(flow_engine(c.engine) || heat_engine(c.engine))) continue;
    ++n;
    CHECK(kAssertedMembers.count(c.member) == 1, "%s: member '%s' has no type assertion in this file", c.id.c_str(),
          c.member.c_str());
    CHECK(c.meshes.empty(), "%s: an engine_cannot_express cell has no problem to hand over", c.id.c_str());
  }
  CHECK(n > 0, "no engine_cannot_express cell names a flow or heat engine");
}

// ================================================================================================ the translation
// From a model document to engine calls, written here and shared with nothing of the model path.  The problem of a
// case is the document of its cell (cpp/src/model_matrix.hpp, minimal_document): the boundary conditions of each
// instance are read from it by boundary name, a name is the side of the fixture with that name, and a kind becomes the
// engine condition of the two tables below.  The engine therefore sees the boundary assignment that validation sees.
// The side r = 0 of an axisymmetric model, which a document never writes, is added as the engine's axis condition.
// Numbers are the document's; the two inflow profiles of the base documents are this file's functions.
//
//   fluid.incompressible (fluids/stokes.hpp, StokesBoundary)        heat.transfer (transport/scalar_transport.hpp, BcKind)
//     wall, inlet                     dirichlet                       temperature              Dirichlet
//     outlet with pressure 0, open    do_nothing                      heat_flux                Flux
//     outlet with pressure p0 != 0    traction, -p0 n                 total_flux               TotalFlux
//     traction                        traction                        convection               Robin
//     slip, symmetry                  slip, symmetry                  insulation, outflow      Zero
//     periodic                        periodic (pressure drop)        the side r = 0           Axis
//     the side r = 0                  axis
using json = nlohmann::json;

constexpr const char* kFluid = "fluid.incompressible";
constexpr const char* kHeat = "heat.transfer";

// How often each line of the tables was used, so that a line that no cell reaches is reported.
std::map<std::string, int> g_translated;
void used(const std::string& line) { ++g_translated[line]; }

bool has_instance(const json& doc, const std::string& type) {
  for (const auto& kv : doc.at("physics").items())
    if (kv.value().value("type", "") == type) return true;
  return false;
}
const json& instance_of(const json& doc, const std::string& type) {
  for (const auto& kv : doc.at("physics").items())
    if (kv.value().value("type", "") == type) return kv.value();
  throw std::logic_error("the document has no instance of " + type);
}
json& instance_of(json& doc, const std::string& type) {
  for (auto& kv : doc.at("physics").items())
    if (kv.value().value("type", "") == type) return kv.value();
  throw std::logic_error("the document has no instance of " + type);
}

// The value of an option of an instance, as text: the document's, else the default of its type.
std::string option_of(const json& inst, const std::string& name) {
  if (inst.contains("options") && inst["options"].contains(name)) {
    const json& v = inst["options"][name];
    return v.is_string() ? v.get<std::string>() : v.dump();
  }
  if (const auto* t = axmp::model::registry().type(inst.at("type").get<std::string>()))
    for (const auto& o : t->options)
      if (o.name == name) return o.default_value;
  throw std::logic_error("no option '" + name + "' on an instance of " + inst.value("type", "?"));
}

// The one material of an engine case (the cases with two regions have no engine column).
const json& material_of(const json& doc) {
  if (!doc.at("materials").contains("medium")) throw std::logic_error("an engine case has the one material 'medium'");
  return doc["materials"]["medium"];
}

std::string sci(double x) {
  char b[40];
  std::snprintf(b, sizeof b, "%.3e", x);
  return b;
}

int vdim_of(fx::Geometry g) { return g == fx::Geometry::planar ? 2 : 3; }

// The boundary that carries the condition kind of a kind cell (cpp/src/model_matrix.hpp).
std::string variable_side(fx::Geometry g) { return g == fx::Geometry::axisymmetric ? "right" : "top"; }

// The outward unit normal of a side of the square, with the components of the flow formulation.
mfem::Vector normal_of(const std::string& side, int vdim) {
  mfem::Vector n(vdim);
  n = 0.0;
  if (side == "left" || side == "axis") n(0) = -1.0;
  else if (side == "right") n(0) = 1.0;
  else if (side == "bottom") n(1) = -1.0;
  else if (side == "top") n(1) = 1.0;
  else throw std::logic_error("no normal for the side '" + side + "'");
  return n;
}

// One component of a vector datum: a number, or one of the inflow profiles of the base documents (planar, through
// x = 0; axisymmetric, through z = 0).
std::function<double(double, double)> component(const json& v) {
  if (v.is_number()) {
    const double a = v.get<double>();
    return [a](double, double) { return a; };
  }
  const std::string s = v.get<std::string>();
  if (s == "4*y*(1-y)") return [](double, double y) { return 4.0 * y * (1.0 - y); };
  if (s == "2*(1-r^2)") return [](double r, double) { return 2.0 * (1.0 - r * r); };
  throw std::logic_error("the translation has no function for the expression '" + s + "'");
}

// ---------------------------------------------------------------------------------------------- flow conditions
struct FlowBoundary {
  fl::StokesBoundary bc;
  std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned;   // the coefficients bc points to
  mfem::VectorCoefficient* keep(mfem::VectorCoefficient* c) {
    owned.emplace_back(c);
    return c;
  }
};

// A vector datum with the components of the formulation (2 planar; 3 axisymmetric, which solves the swirl): the
// engine refuses any other count.
mfem::VectorCoefficient* vector_data(const json& v, int vdim, FlowBoundary& out) {
  if (!v.is_array() || static_cast<int>(v.size()) != vdim)
    throw std::logic_error("a vector datum of the document has " + std::to_string(v.size()) + " components, the "
                           "formulation has " + std::to_string(vdim));
  std::vector<std::function<double(double, double)>> comp;
  for (const auto& e : v) comp.push_back(component(e));
  return out.keep(new mfem::VectorFunctionCoefficient(vdim, [comp](const mfem::Vector& x, mfem::Vector& u) {
    for (size_t k = 0; k < comp.size(); ++k) u(static_cast<int>(k)) = comp[k](x(0), x(1));
  }));
}

void translate_flow(const json& conditions, const fx::Fixture& f, FlowBoundary& out) {
  const int vdim = vdim_of(f.geometry);
  for (const auto& kv : conditions.items()) {
    const std::string side = kv.key();
    const json& c = kv.value();
    const int a = f.attr(side);
    const std::string kind = c.at("kind").get<std::string>();
    if (kind == "wall" || kind == "inlet") {
      if (c.contains("velocity_m_s")) out.bc.dirichlet(a, vector_data(c["velocity_m_s"], vdim, out));
      else out.bc.dirichlet(a);
      used("flow " + kind + " -> dirichlet");
    } else if (kind == "outlet") {
      const double p0 = c.at("pressure_Pa").get<double>();
      if (p0 == 0.0) {
        out.bc.do_nothing(a);
        used("flow outlet (pressure 0) -> do_nothing");
      } else {
        mfem::Vector t = normal_of(side, vdim);   // a straight side: the traction -p0 n is one vector
        t *= -p0;
        out.bc.traction(a, out.keep(new mfem::VectorConstantCoefficient(t)));
        used("flow outlet (pressure not 0) -> traction");
      }
    } else if (kind == "open") {
      out.bc.do_nothing(a);
      used("flow open -> do_nothing");
    } else if (kind == "traction") {
      out.bc.traction(a, vector_data(c.at("traction_Pa"), vdim, out));
      used("flow traction -> traction");
    } else if (kind == "slip") {
      out.bc.slip(a);
      used("flow slip -> slip");
    } else if (kind == "symmetry") {
      out.bc.symmetry(a);
      used("flow symmetry -> symmetry");
    } else if (kind == "periodic") {
      out.bc.periodic(a, f.attr(c.at("partner").get<std::string>()), c.at("translation_m").get<std::vector<double>>(),
                      fl::PeriodicDrive::PressureDrop, c.at("pressure_drop_Pa").get<double>());
      used("flow periodic -> periodic");
    } else {
      throw std::logic_error("the translation has no engine condition for the flow kind '" + kind + "'");
    }
  }
  if (f.geometry == fx::Geometry::axisymmetric) {
    out.bc.axis(f.attr("axis"));
    used("flow r = 0 -> axis");
  }
}

// ---------------------------------------------------------------------------------------------- heat conditions
struct HeatCondition {
  std::vector<int> attrs;
  tr::BcKind kind;
  tr::BcData data;
};

std::vector<HeatCondition> translate_heat(const json& conditions, const fx::Fixture& f) {
  std::vector<HeatCondition> out;
  for (const auto& kv : conditions.items()) {
    const json& c = kv.value();
    const std::string kind = c.at("kind").get<std::string>();
    HeatCondition h;
    std::string name;
    h.attrs = {f.attr(kv.key())};
    if (kind == "temperature") {
      h.kind = tr::BcKind::Dirichlet;
      name = "Dirichlet";
      h.data.g.value = c.at("T_K").get<double>();
    } else if (kind == "heat_flux") {
      h.kind = tr::BcKind::Flux;
      name = "Flux";
      h.data.g.value = c.at("q_W_m2").get<double>();
    } else if (kind == "total_flux") {
      h.kind = tr::BcKind::TotalFlux;
      name = "TotalFlux";
      h.data.g.value = c.at("q_W_m2").get<double>();
    } else if (kind == "convection") {
      h.kind = tr::BcKind::Robin;
      name = "Robin";
      h.data.h.value = c.at("h_W_m2K").get<double>();
      h.data.ext.value = c.at("T_ext_K").get<double>();
    } else if (kind == "insulation" || kind == "outflow") {
      h.kind = tr::BcKind::Zero;
      name = "Zero";
    } else {
      throw std::logic_error("the translation has no engine condition for the heat kind '" + kind + "'");
    }
    used("heat " + kind + " -> " + name);
    out.push_back(h);
  }
  if (f.geometry == fx::Geometry::axisymmetric) {
    // a zero-flux condition on r = 0 is refused by the operator: the axis has its own kind
    out.push_back({{f.attr("axis")}, tr::BcKind::Axis, {}});
    used("heat r = 0 -> Axis");
  }
  return out;
}

// ---------------------------------------------------------------------------------------------- the flow engines
fl::StokesOptions stokes_options(const json& flow, fx::Geometry g, mfem::Coefficient& mu) {
  fl::StokesOptions so;
  so.nu = &mu;
  so.geometry = g == fx::Geometry::axisymmetric ? fl::Geometry::Axisymmetric : fl::Geometry::Planar;
  const std::string form = option_of(flow, "viscous_form");
  if (form == "stress") so.form = fl::ViscousForm::SymmetricStress;
  else if (form == "laplacian") so.form = fl::ViscousForm::Laplacian;
  else throw std::logic_error("no engine value for viscous_form '" + form + "'");
  so.velocity_order = std::stoi(option_of(flow, "order"));
  return so;
}

fl::NavierStokesOptions navier_stokes_options(const json& doc, fx::Geometry g, mfem::Coefficient& mu) {
  const json& flow = instance_of(doc, kFluid);
  fl::NavierStokesOptions o;
  o.stokes = stokes_options(flow, g, mu);
  o.rho = material_of(doc).value("density", 1.0);
  const std::string conv = option_of(flow, "convection");
  if (conv == "standard") o.convection = fl::ConvectiveForm::Standard;
  else if (conv == "skew") o.convection = fl::ConvectiveForm::Skew;
  else throw std::logic_error("no engine value for convection '" + conv + "'");
  return o;
}

double kinetic_energy(const mfem::ParGridFunction& u) {
  mfem::Vector z(u.VectorDim());
  z = 0.0;
  mfem::VectorConstantCoefficient zero(z);
  const double l2 = u.ComputeL2Error(zero);
  return 0.5 * l2 * l2;
}

// The L2 norm of a field.
double l2_norm(const mfem::ParGridFunction& u) {
  mfem::Vector z(u.VectorDim());
  z = 0.0;
  mfem::VectorConstantCoefficient zero_vector(z);
  mfem::ConstantCoefficient zero_scalar(0.0);
  return u.VectorDim() > 1 ? u.ComputeL2Error(zero_vector) : u.ComputeL2Error(zero_scalar);
}

// A flow handed to its engine and solved; kept alive while a transport operator reads its velocity.
struct FlowRun {
  mfem::ConstantCoefficient mu;
  FlowBoundary boundary;
  std::unique_ptr<fl::StokesSolver> stokes;
  std::unique_ptr<fl::NavierStokesSolver> ns;
  std::unique_ptr<fl::TransientNavierStokes> transient;
  double energy = -1.0;   // the kinetic energy of the solved velocity (after the second step, for the time stepper)
  double pressure_norm = -1.0;   // the L2 norm of the solved pressure (same state)
  mfem::ParFiniteElementSpace& space() { return stokes ? stokes->velocity_space() : ns->velocity_space(); }
  const mfem::ParGridFunction& velocity() const { return stokes ? stokes->velocity() : ns->velocity(); }
};

// The flow instance of `doc` handed to `engine` on the mesh `pm` of fixture `f`: the constructor, then a solve that
// must converge (one time step for the transient engine), so that a refusal of the assembly or of the solve is seen
// too.
void run_flow(Probe& p, const std::string& engine, const json& doc, const fx::Fixture& f, mfem::ParMesh& pm, FlowRun& run) {
  const json& flow = instance_of(doc, kFluid);
  run.mu.constant = material_of(doc).at("fluid").at("mu_Pa_s").get<double>();
  translate_flow(flow.at("boundary_conditions"), f, run.boundary);
  if (engine == "StokesSolver") {
    const fl::StokesOptions so = stokes_options(flow, f.geometry, run.mu);
    p.enter("constructor");
    run.stokes = std::make_unique<fl::StokesSolver>(pm, so, run.boundary.bc);
    p.enter("solve");
    const fl::StokesDiagnostics& d = run.stokes->solve();
    p.require(d.converged, "the Stokes solve did not converge (relative residual " + sci(d.rel_residual) + ")");
    run.energy = kinetic_energy(run.stokes->velocity());
    run.pressure_norm = l2_norm(run.stokes->pressure());
    return;
  }
  const fl::NavierStokesOptions o = navier_stokes_options(doc, f.geometry, run.mu);
  if (engine == "NavierStokesSolver") {
    p.enter("constructor");
    run.ns = std::make_unique<fl::NavierStokesSolver>(pm, o, run.boundary.bc);
    p.enter("solve");
    const fl::NavierStokesDiagnostics& d = run.ns->solve();
    p.require(d.converged, "the Navier-Stokes solve did not converge: " + d.failure);
    run.energy = kinetic_energy(run.ns->velocity());
    run.pressure_norm = l2_norm(run.ns->pressure());
    return;
  }
  if (engine != "TransientNavierStokes") throw std::logic_error("no flow engine '" + engine + "'");
  fl::TransientOptions t;
  t.ns = o;
  p.enter("constructor");
  run.transient = std::make_unique<fl::TransientNavierStokes>(pm, t, run.boundary.bc, [](double, double, double) {});
  // the first time steps: the fluid at rest, projected onto the discretely divergence-free fields with the boundary
  // data, then the BDF1 starting step and one BDF2 step, each of the document's end time
  p.enter("first_step");
  mfem::Vector z(vdim_of(f.geometry));
  z = 0.0;
  mfem::VectorConstantCoefficient at_rest(z);
  p.require(run.transient->initialize(0.0, at_rest, true), "the projection of the initial velocity failed");
  for (int step = 1; step <= 2; ++step) {
    const fl::StepResult r = run.transient->advance(0.1);
    p.require(r.converged, "time step " + std::to_string(step) + " did not converge: " + r.failure);
    p.require(r.order == step, "time step " + std::to_string(step) + " has the BDF order " + std::to_string(r.order));
  }
  run.energy = kinetic_energy(run.transient->core().velocity_field());
  run.pressure_norm = l2_norm(run.transient->core().pressure_field());
}

// ---------------------------------------------------------------------------------------------- the heat engines
const json* advection_source(const json& heat) {
  if (heat.contains("sources"))
    for (const auto& s : heat["sources"])
      if (s.value("kind", "") == "prescribed_advection") return &s;
  return nullptr;
}

tr::AdvectiveForm form_of(const json& heat) {
  const std::string form = option_of(heat, "advective_form");
  if (form == "convective") return tr::AdvectiveForm::Convective;
  if (form == "conservative_divergence") return tr::AdvectiveForm::ConservativeDivergence;
  if (form == "conservative_flux") return tr::AdvectiveForm::ConservativeFlux;
  throw std::logic_error("no engine value for advective_form '" + form + "'");
}

// The heat instance of `doc` handed to the transport operator: the constructor, the data, every condition through
// add_bc, finalize, and a solve that must converge.  The velocity is the document's: none (conduction), the callable
// of a prescribed_advection source with its divergence, or the finite-element velocity of `flow`, solved on the same
// mesh object.  Returns the temperature.
mfem::Vector run_transport(Probe& p, const json& doc, const fx::Fixture& f, mfem::ParMesh& pm, FlowRun* flow) {
  const json& heat = instance_of(doc, kHeat);
  const json& mat = material_of(doc);
  tr::TransportOptions to;
  to.geometry = f.geometry == fx::Geometry::axisymmetric ? tr::Geometry::Axisymmetric : tr::Geometry::Planar;
  to.order = std::stoi(option_of(heat, "order"));
  to.supg = option_of(heat, "stabilization") == "supg";
  to.form = form_of(heat);
  p.enter("constructor");
  tr::ScalarTransport op(pm, to);
  tr::ScalarData k;
  k.value = mat.at("thermal").at("k_W_mK").get<double>();
  op.set_diffusivity(k);
  const json* adv = advection_source(heat);
  if (adv || flow) {
    const double capacity = mat.at("density").get<double>() * mat.at("thermal").at("cp_J_kgK").get<double>();
    op.set_capacity(capacity);
    tr::Velocity v;
    v.scale = capacity;
    if (adv) {
      const double bx = adv->at("velocity_m_s").at(0).get<double>(), by = adv->at("velocity_m_s").at(1).get<double>();
      v.fn = [bx, by](const double*, int n, int ncomp, double* out) {
        for (int i = 0; i < n; ++i) {
          for (int k2 = 0; k2 < ncomp; ++k2) out[ncomp * i + k2] = 0.0;
          out[ncomp * i] = bx;
          out[ncomp * i + 1] = by;
        }
      };
      v.div_fn = [](const double*, int n, double* out) {   // a constant velocity: div b = 0
        for (int i = 0; i < n; ++i) out[i] = 0.0;
      };
    } else {
      v.fes = &flow->space();
    }
    op.set_velocity(v);
  }
  for (const auto& h : translate_heat(heat.at("boundary_conditions"), f)) {
    p.enter("add_bc");
    op.add_bc(h.attrs, h.kind, h.data);
  }
  p.enter("finalize");
  op.finalize();
  p.enter("solve");
  mfem::Vector T;
  const tr::SolveDiagnostics d = op.solve(T, flow ? &flow->velocity() : nullptr, fl::LUBackend::EigenLU, 1e-10);
  p.require(d.converged, "the transport solve did not converge (relative residual " + sci(d.relative_residual) +
                             "): " + d.failure);
  return T;
}

const json& coupling_of(const json& doc, const std::string& term) {
  if (doc.contains("couplings"))
    for (const auto& c : doc["couplings"])
      if (c.value("term", "") == term) return c;
  throw std::logic_error("the document has no coupling term '" + term + "'");
}

// The flow and heat instances of a Boussinesq document handed to BoussinesqSolver, whose constructor builds its
// transport operator, adds the heat conditions and finalizes it: the constructor, then a solve that must converge.
void run_boussinesq(Probe& p, const json& doc, const fx::Fixture& f, mfem::ParMesh& pm) {
  const json& heat = instance_of(doc, kHeat);
  const json& mat = material_of(doc);
  mfem::ConstantCoefficient mu(mat.at("fluid").at("mu_Pa_s").get<double>());
  FlowBoundary boundary;
  translate_flow(instance_of(doc, kFluid).at("boundary_conditions"), f, boundary);
  fl::BoussinesqOptions b;
  const fl::NavierStokesOptions o = navier_stokes_options(doc, f.geometry, mu);
  b.ns.stokes = o.stokes;
  b.ns.rho = o.rho;
  b.ns.convection = o.convection;
  b.cp = mat.at("thermal").at("cp_J_kgK").get<double>();
  // gravity along the vertical axis (y planar, z axisymmetric: the second coordinate): beta g and the sign of "up"
  const json& buoyancy = coupling_of(doc, "buoyancy");
  const json& gravity = buoyancy.at("gravity_m_s2");
  if (gravity.at(0).get<double>() != 0.0) throw std::logic_error("BoussinesqOptions takes gravity along the vertical axis only");
  const double gz = gravity.at(1).get<double>();
  b.beta_g = mat.at("fluid").at("beta_1_K").get<double>() * std::fabs(gz);
  b.up_sign = gz < 0 ? 1 : -1;
  b.t_ref_set = true;
  b.t_ref = buoyancy.at("T_ref_K").get<double>();
  b.k.value = mat.at("thermal").at("k_W_mK").get<double>();
  b.t_order = std::stoi(option_of(heat, "order"));
  b.supg = option_of(heat, "stabilization") == "supg";
  // BoussinesqOptions has no member for the advective form (asserted above): only the convective form can be handed over
  if (form_of(heat) != tr::AdvectiveForm::Convective)
    throw std::logic_error("BoussinesqOptions has no advective form to set: the document asks for another than convective");
  for (const auto& h : translate_heat(heat.at("boundary_conditions"), f)) b.t_bcs.push_back({h.attrs, h.kind, h.data});
  p.enter("constructor");
  fl::BoussinesqSolver solver(pm, b, boundary.bc);
  p.require(solver.transport().options().form == tr::AdvectiveForm::Convective &&
                solver.diagnostics().energy_form == "convective",
            "the energy equation of the Boussinesq solver is not in the convective form (diagnostics: '" +
                solver.diagnostics().energy_form + "')");
  p.enter("solve");
  const fl::BoussinesqDiagnostics& d = solver.solve();
  p.require(d.converged, "the Boussinesq solve did not converge: " + d.failure);
}

// ---------------------------------------------------------------------------------------------- a case
// The row whose base problem a case runs: its own, or, for the sparse-mesh cases (which are not rows), the simplest
// row of the engine in the geometry class.
std::string row_of(const MatrixCase& c, fx::Geometry g) {
  if (!c.row.empty()) return c.row;
  const std::string gs = fx::to_string(g);
  if (c.engine == "StokesSolver") return "stokes_" + gs;
  if (c.engine == "NavierStokesSolver") return "navier_stokes_steady_" + gs;
  if (c.engine == "TransientNavierStokes") return "navier_stokes_transient_" + gs;
  if (c.engine == "BoussinesqSolver") return "boussinesq_" + gs;
  if (c.engine == "transport::ScalarTransport") return "heat_" + gs;
  throw std::logic_error(c.id + ": no row for the engine '" + c.engine + "'");
}

// The document of a case: its own, or, for a case without one (a rank count, the sparse mesh), the base document of
// the row, which is the document of the row's cell for the default field order.
json document_of(const MatrixCase& c, fx::Geometry g) {
  json doc = axmp::model::minimal_document(c);
  if (doc.is_null()) {
    const std::string row = row_of(c, g);
    for (const auto& b : cases())
      if (b.row == row && b.axis == "option_value" && b.value == "order=2") doc = axmp::model::minimal_document(b);
  }
  if (!doc.is_object()) throw std::logic_error(c.id + ": no document to translate");
  return doc;
}

struct Outcome {
  double energy = -1.0;   // flow engines: the kinetic energy of the solved velocity
  double pressure_norm = -1.0;   // flow engines: the L2 norm of the solved pressure
  mfem::Vector T;         // the transport operator: the temperature
};

// `doc` handed to `engine` on fixture `f`, partitioned over `comm`.
Outcome run_document(Probe& p, const std::string& engine, const json& doc, fx::Fixture f, MPI_Comm comm) {
  const std::unique_ptr<mfem::ParMesh> pm = partitioned(f, comm);
  Outcome out;
  if (flow_engine(engine)) {
    FlowRun run;
    run_flow(p, engine, doc, f, *pm, run);
    out.energy = run.energy;
    out.pressure_norm = run.pressure_norm;
  } else if (engine == "BoussinesqSolver") {
    run_boussinesq(p, doc, f, *pm);
  } else if (engine == "transport::ScalarTransport") {
    // a flow instance beside the heat instance is solved first, by the engine its equation selects, on the same mesh
    std::unique_ptr<FlowRun> flow;
    if (has_instance(doc, kFluid)) {
      flow = std::make_unique<FlowRun>();
      const std::string equation = option_of(instance_of(doc, kFluid), "equation");
      run_flow(p, equation == "stokes" ? "StokesSolver" : "NavierStokesSolver", doc, f, *pm, *flow);
    }
    out.T = run_transport(p, doc, f, *pm, flow.get());
  } else {
    throw std::logic_error("no engine '" + engine + "' in this program");
  }
  return out;
}

// Runs `one` for every geometry class of the case.  A case the engine refuses must be refused in each: the refusals
// before the last are checked here, the last is left to exercise().
void for_each_geometry(const MatrixCase& c, Probe& p, const std::function<void(fx::Geometry)>& one) {
  const std::vector<fx::Geometry> gs = geometries_of(c);
  for (size_t i = 0; i < gs.size(); ++i) {
    if (c.cls != CellValue::engine_refuses || i + 1 == gs.size()) {
      one(gs[i]);
      continue;
    }
    bool threw = false;
    std::string what;
    try {
      one(gs[i]);
    } catch (const CellCheck&) {
      throw;
    } catch (const std::exception& e) {
      threw = true;
      what = e.what();
    }
    const std::string g = fx::to_string(gs[i]);
    p.require(threw, g + ": the engine accepted what the case says it refuses");
    p.require(p.current() == c.call, g + ": refused in '" + p.current() + "', the case names '" + c.call + "': " + what);
    p.require(what.find(c.engine_text) != std::string::npos, g + ": the refusal \"" + what + "\" does not contain \"" +
                                                                 c.engine_text + "\"");
  }
}

// The problem of a case on a mesh: its document handed to its engine, and what else the cell is about.
void run_case(Probe& p, const MatrixCase& c, const std::string& mesh) {
  for_each_geometry(c, p, [&](fx::Geometry g) {
    const json doc = document_of(c, g);
    const fx::Fixture f = fixture(mesh, g);
    const Outcome out = run_document(p, c.engine, doc, f, MPI_COMM_WORLD);

    // an outlet is two engine conditions: the cell is handed over with the pressure 0 (above) and with a pressure
    if (flow_engine(c.engine) && c.axis == "condition_kind" && c.value == "outlet") {
      json with_pressure = doc;
      instance_of(with_pressure, kFluid)["boundary_conditions"][variable_side(g)]["pressure_Pa"] = 1.0;
      run_document(p, c.engine, with_pressure, f, MPI_COMM_WORLD);
    }
    // the one-way rows take the flow of either stationary flow row: the base cell is handed over with a Navier-Stokes
    // flow as well (the documents write the Stokes flow)
    if (c.engine == "transport::ScalarTransport" && has_instance(doc, kFluid) && c.axis == "option_value" &&
        c.value == "order=2") {
      json with_convection = doc;
      instance_of(with_convection, kFluid)["options"]["equation"] = "navier_stokes";
      run_document(p, c.engine, with_convection, f, MPI_COMM_WORLD);
    }
    // without a velocity the three advective forms are one operator: the temperature of a conservative form is that
    // of the convective form.  Both are direct solves of the same matrix, so they differ by round-off (measured: 0);
    // 1e-9 K on temperatures of 300 K is 3e-12 relative, and a form that kept an advective term would differ by kelvins.
    if (c.engine == "transport::ScalarTransport" && c.cls == CellValue::offered && c.axis == "option_value" &&
        c.value.rfind("advective_form=", 0) == 0 && !has_instance(doc, kFluid) &&
        !advection_source(instance_of(doc, kHeat)) && option_of(instance_of(doc, kHeat), "advective_form") != "convective") {
      json convective = doc;
      instance_of(convective, kHeat)["options"]["advective_form"] = "convective";
      Probe reference;
      const Outcome ref = run_document(reference, c.engine, convective, f, MPI_COMM_WORLD);
      p.require(ref.T.Size() == out.T.Size() && out.T.Size() > 0, "no temperature to compare");
      double d = 0.0;
      for (int i = 0; i < out.T.Size(); ++i) d = std::max(d, std::fabs(out.T(i) - ref.T(i)));
      if (g_rank == 0) std::printf("  %s: differs from the convective form by %.2e K at most\n", c.id.c_str(), d);
      p.require(d <= 1e-9, "without a velocity the form " + c.value + " gives another temperature than the convective "
                               "form (max difference " + sci(d) + " K)");
    }
    // two ranks: the flow that the engine solves on the partitioned mesh is the flow it solves on one rank.  Rank 0
    // solves the same document on a mesh of its own.  Both are the same discrete problem solved by the same direct
    // method, so the kinetic energies differ by round-off (measured: 0 to 6e-16 relative on the six flow rows, the
    // two time steps of the transient rows included; the pressure norms differ by 2.0e-16 to 2.54e-15 relative,
    // measured on cm2 and cf1).  The bound is the loosest convergence gate of the engines, the
    // relative residual 1e-10 of the Stokes solve; a contribution lost at the partition boundary would change the
    // energy by a fraction of its value.  The pressure is compared by its L2 norm on the same footing (a field that
    // the velocity does not see, such as a shift of the pressure on one rank, leaves the energy unchanged).
    if (g_size == 2 && flow_engine(c.engine) && c.cls == CellValue::offered && g_rank == 0) {
      Probe reference;
      const Outcome one = run_document(reference, c.engine, doc, f, MPI_COMM_SELF);
      p.require(one.energy > 0 && std::fabs(out.energy - one.energy) <= 1e-10 * one.energy,
                "kinetic energy on 2 ranks " + sci(out.energy) + ", on 1 rank " + sci(one.energy));
      p.require(one.pressure_norm > 0 && std::fabs(out.pressure_norm - one.pressure_norm) <= 1e-10 * one.pressure_norm,
                "pressure norm on 2 ranks " + sci(out.pressure_norm) + ", on 1 rank " + sci(one.pressure_norm));
      std::printf("  %s: kinetic energy and pressure norm on 2 ranks and on 1 rank differ by %.2e and %.2e (relative)\n",
                  c.id.c_str(), std::fabs(out.energy - one.energy) / one.energy,
                  std::fabs(out.pressure_norm - one.pressure_norm) / one.pressure_norm);
    }
  });
}

// ================================================================================================ the cells
// Every visit of a group handed to its engine.
size_t hand_over(Group g) {
  const std::vector<Visit> visits = visits_of(g, g_size);
  for (const auto& v : visits) {
    const MatrixCase& c = *v.c;
    const std::string mesh = v.mesh;
    exercise(c, mesh, [&c, &mesh](Probe& p) { run_case(p, c, mesh); });
  }
  if (g_rank == 0) std::printf("%s: %zu cells handed to their engines\n", to_string(g), visits.size());
  return visits.size();
}

void require_translated(Group g, const Strs& lines) {
  for (const auto& l : lines)
    CHECK(g_translated[l] > 0, "%s: no cell reached the translation '%s'", to_string(g), l.c_str());
}

// Every offered flow cell on the base mesh: StokesSolver (planar, axisymmetric), NavierStokesSolver (planar,
// axisymmetric with swirl) and TransientNavierStokes (planar, axisymmetric) take each offered kind and option value,
// and solve.
void flow_offered_cells_accepted() {
  hand_over(Group::flow_offered);
  require_translated(Group::flow_offered,
                     {"flow wall -> dirichlet", "flow inlet -> dirichlet", "flow outlet (pressure 0) -> do_nothing",
                      "flow outlet (pressure not 0) -> traction", "flow open -> do_nothing", "flow traction -> traction",
                      "flow slip -> slip", "flow symmetry -> symmetry", "flow r = 0 -> axis"});
}

// Every refused flow cell on the base mesh: slip, symmetry and periodic in axisymmetric geometry and with convection,
// and the option values an engine refuses, each in the call and with the text its case names.
void flow_engine_refusals() {
  hand_over(Group::flow_refusals);
  require_translated(Group::flow_refusals, {"flow periodic -> periodic"});
}

// Every offered heat cell on the base mesh: the transport operator through finalize and a solve (conduction; a
// callable velocity with its divergence; the finite-element velocity of a flow solved on the same mesh), and
// BoussinesqSolver through its constructor and a solve.
void heat_offered_cells_accepted_through_finalize() {
  hand_over(Group::heat_offered);
  require_translated(Group::heat_offered,
                     {"heat temperature -> Dirichlet", "heat heat_flux -> Flux", "heat total_flux -> TotalFlux",
                      "heat convection -> Robin", "heat insulation -> Zero", "heat outflow -> Zero", "heat r = 0 -> Axis"});
}

// Every refused heat cell on the base mesh: the flux kinds against the advective form (add_bc; inside the constructor
// of BoussinesqSolver on its rows), SUPG without a velocity (finalize), a P1 temperature in the Boussinesq solver.
void heat_engine_refusals() { hand_over(Group::heat_refusals); }

// The sparse mesh (the master numbers left on), planar and axisymmetric: the flow engines take it with conditions on
// the attributes present; the transport operator and BoussinesqSolver refuse it, because they count attributes from 1.
void sparse_mesh_cells() { hand_over(Group::sparse); }

// The mesh with a nodes grid function: SUPG is refused by the constructor of the transport operator (and of
// BoussinesqSolver, which builds one); without SUPG the mesh is accepted.
void nodes_mesh_cells() { hand_over(Group::nodes); }

// The quadrilateral mesh: accepted by the flow engines, refused by the constructor of the transport operator (and of
// BoussinesqSolver).
void quad_mesh_cells() { hand_over(Group::quad); }

// Two ranks: the flow engines solve, and the solution is the one-rank solution; the transport operator and
// BoussinesqSolver refuse a mesh partitioned over two ranks.
void rank_cells() { hand_over(Group::ranks); }

// ================================================================================================ coverage
void check_every_engine_cell_is_exercised() {
  const auto want = expected_visits(cases(), g_size);
  const Strs m = coverage_failures(want, g_visits);
  std::map<std::string, int> per_group;
  for (const auto& v : want) {
    auto it = g_visits.find({v.c->id, v.mesh});
    if (it == g_visits.end() || it->second != 1) ++per_group[to_string(group_of(*v.c, v.mesh))];
  }
  for (const auto& [g, k] : per_group)
    CHECK(false, "%d visit(s) of %s are missing or repeated", k, g.c_str());
  const size_t shown = std::min<size_t>(m.size(), 20);
  for (size_t i = 0; i < shown; ++i) CHECK(false, "%s", m[i].c_str());
  CHECK(m.size() == shown, "... and %zu more coverage failures", m.size() - shown);
  if (g_rank == 0)
    std::printf("coverage (%d rank%s): %zu visits expected, %zu made, %zu failures\n", g_size, g_size == 1 ? "" : "s",
                want.size(), g_visits.size(), m.size());
}

}  // namespace

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  g_rank = mfem::Mpi::WorldRank();
  g_size = mfem::Mpi::WorldSize();
  int ranks = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--ranks") == 0 && i + 1 < argc) {
      ranks = std::atoi(argv[++i]);
    } else {
      if (g_rank == 0) std::fprintf(stderr, "usage: test_fluids_model_matrix [--ranks 1|2]\n");
      return 2;
    }
  }
  CHECK(ranks == g_size, "started on %d rank(s) with --ranks %d: the two-rank cells run under the MPI launcher on 2 ranks",
        g_size, ranks);
  CHECK(ranks == 1 || ranks == 2, "--ranks %d: the case list has one-rank and two-rank cells only", ranks);
  if (ranks == g_size && (ranks == 1 || ranks == 2)) {
    try {
      if (g_size == 1) {
        check_reference_numbering();
        check_fixtures();
        check_coverage_contract();
        engine_cannot_express_types();
        flow_offered_cells_accepted();
        flow_engine_refusals();
        heat_offered_cells_accepted_through_finalize();
        heat_engine_refusals();
        sparse_mesh_cells();
        nodes_mesh_cells();
        quad_mesh_cells();
      } else {
        rank_cells();
      }
      check_every_engine_cell_is_exercised();
    } catch (const std::exception& e) {
      CHECK(false, "uncaught exception: %s", e.what());
    }
  }
  int total = 0;
  MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) {
    if (total) std::printf("test_fluids_model_matrix (%d rank%s): %d FAILURES\n", g_size, g_size == 1 ? "" : "s", total);
    else std::printf("test_fluids_model_matrix (%d rank%s): OK\n", g_size, g_size == 1 ? "" : "s");
  }
  return total ? 1 : 0;
}
