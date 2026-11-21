// The selection semantics of the named boundaries of a model document (cpp/include/axionmultiphysics/model_selection.hpp):
// selector evaluation on an entity table, the atoms and their master numbers, the partition rule of a physics instance,
// the placement of a selection with respect to an instance, the table's JSON form, and the message builders of the
// checks that use them (cpp/src/model_errors.hpp).  Built and run in every build: no mesher, no geometry kernel.
//
// INDEPENDENT REFERENCES.
//   * Design note sections 5.2 (selectors, where, orientation), 5.3 (atoms, master numbering), 5.5 (the partition rule
//     and its three texts), 3.6 ("remaining") and 10.3 (force outputs), transcribed by hand below.
//   * Hand-drawn entity tables of the fixtures of the mesher's bounded modes (a rectangle split into two regions with a
//     marker, a channel with a post, an annulus, the axisymmetric pipe split in two, a thin film), each entity written
//     out with its sides, box, midpoint, tangent and length; every selection, orientation, atom, master number and
//     partition below is worked out by hand from those tables.
//   * fixtures::master_numbers() (cpp/tests/model_matrix_fixtures.hpp), the numbering of the unit square of the
//     matrix documents.
//
// CONVENTIONS (model_selection.hpp).
//   side[0] is the region on the left of the entity's tangent (where the tangent rotated by +90 degrees points),
//   side[1] the region on its right.  Orientation +1: the selection's normal points from side[0] into side[1], i.e. it
//   is n = (t_y, -t_x); -1: it points the other way.  The tests turn an orientation o into the normal o * (t_y, -t_x) and
//   check its direction geometrically: a marker written bottom to top (t = +e_y) has the normal -e_x, written top to
//   bottom +e_x; between [a, b] across the vertical interface of the split rectangle points along +e_x.
//
// TEXTS.  The texts of design note 5.2 (the empty selection of the shifted channel) and 5.5 (the three partition
// refusals) are copied verbatim, joined into one line where the note wraps them.  Every other text is the test
// author's (AUTHOR TEXT) and is the specification.  Numbers are written as the shortest decimal that reads back as the
// same double (msg::num_text).  A count is "1 edge" / "N edges"; a measure is the sum of the lengths in ascending tag
// order; "first at" is the point of the entity whose point is lexicographically smallest; adjacent pairs are written in
// region-pair order (document order of the regions, "exterior" last).
//
// THE API (namespace axmp::model): selection::resolve, partition, place, to_json, table_from_json; msg::empty_selection,
// partial_match, between_overlap, boundary_uncovered, boundary_double, boundary_interior, boundary_outside_domain,
// force_on_interior, boundary_not_in_domain, force_needs_stress_form_on_part.  Each finding is an axmp::Error:
// message(), where(), hint(), code(), details() (JSON text).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_selection.hpp"
#include "../src/model_errors.hpp"
#include "model_matrix_fixtures.hpp"

using nlohmann::json;
using axmp::Error;
namespace S = axmp::model::selection;
namespace msg = axmp::model::msg;

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

json J(const char* s) { return json::parse(s); }

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
void check_eq(const std::string& what, const std::string& got, const std::string& want) {
  if (got != want) fail(what + "\n    got:  " + got + "\n    want: " + want);
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
void check_json(const std::string& what, const json& got, const json& want) {
  if (got != want) fail(what + "\n    got:  " + got.dump() + "\n    want: " + want.dump());
}

// ------------------------------------------------------------------------------------------------- entity tables
// A straight, axis-parallel entity written from (x0, y0) to (x1, y1): box from the end points, midpoint, unit tangent
// along the written direction; the length is given (hand value).
S::Entity seg(int tag, const std::string& s0, const std::string& s1, double x0, double y0, double x1, double y1, double length,
              const Strs& curves = {}, bool axis = false) {
  if (x0 != x1 && y0 != y1) throw std::logic_error("seg: the test writes axis-parallel segments only");
  S::Entity e;
  e.tag = tag;
  e.dim = 1;
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
  e.length_m = length;
  return e;
}
// A circular arc or circle, every field written out.
S::Entity arc(int tag, const std::string& s0, const std::string& s1, double lox, double loy, double hix, double hiy, double px,
              double py, double tx, double ty, double length) {
  S::Entity e;
  e.tag = tag;
  e.dim = 1;
  e.side[0] = s0;
  e.side[1] = s1;
  e.type = "circle";
  e.lo[0] = lox, e.lo[1] = loy, e.hi[0] = hix, e.hi[1] = hiy;
  e.point[0] = px, e.point[1] = py;
  e.tangent[0] = tx, e.tangent[1] = ty;
  e.length_m = length;
  return e;
}

// The split rectangle: x in [0, L], y in [0, H], L = 0.04, H = 0.01; region a for x < L/2, b for x > L/2; the marker mk
// across a at x = L/4, written bottom to top (top to bottom when mk_down), which splits the bottom and the top of a.
// Every exterior edge runs counter-clockwise (region at side[0]).  With rim, the right edge x = L carries the marker
// rim written top to bottom, so its tangent is -e_y and its left side is the exterior.  Diagonal hypot(0.04, 0.01).
//
//        8 (a|ext)   7 (a|ext)     6 (b|ext)
//    +-----<-----+-----<-----+---------<---------+  y = 0.01
//    |           ^           ^                   ^
//  1 v (a|ext)  10 mk (a|a)  9 interface (a|b)   5 (b|ext)
//    |           |           |                   |
//    +----->-----+----->-----+--------->---------+  y = 0
//   x=0  2 (a|ext) 0.01 3 (a|ext) 0.02  4 (b|ext)  0.04
S::Table split_rect(bool mk_down = false, bool rim = false) {
  S::Table t;
  t.geometry = "planar";
  t.regions = {"a", "b"};
  t.diagonal_m = 0.04123105625617661;
  t.entities = {
      seg(1, "a", "exterior", 0, 0.01, 0, 0, 0.01),
      seg(2, "a", "exterior", 0, 0, 0.01, 0, 0.01),
      seg(3, "a", "exterior", 0.01, 0, 0.02, 0, 0.01),
      seg(4, "b", "exterior", 0.02, 0, 0.04, 0, 0.02),
      rim ? seg(5, "exterior", "b", 0.04, 0.01, 0.04, 0, 0.01, {"rim"}) : seg(5, "b", "exterior", 0.04, 0, 0.04, 0.01, 0.01),
      seg(6, "b", "exterior", 0.04, 0.01, 0.02, 0.01, 0.02),
      seg(7, "a", "exterior", 0.02, 0.01, 0.01, 0.01, 0.01),
      seg(8, "a", "exterior", 0.01, 0.01, 0, 0.01, 0.01),
      seg(9, "a", "b", 0.02, 0, 0.02, 0.01, 0.01),
      mk_down ? seg(10, "a", "a", 0.01, 0.01, 0.01, 0, 0.01, {"mk"}) : seg(10, "a", "a", 0.01, 0, 0.01, 0.01, 0.01, {"mk"}),
  };
  return t;
}
// The base selectors of the split rectangle: hot (x = 0), cold (x = L), the interface and the marker; top and bottom
// are left unnamed.
json split_rect_selectors() {
  return J(R"({"hot":       {"of": ["a"], "where": {"x": [0]}},
               "cold":      {"of": ["b"], "where": {"x": [0.04]}},
               "interface": {"between": [["a"], ["b"]]},
               "mk":        {"curve": "mk"}})");
}

// The channel of note 4.1 (L = 0.04, H = 0.01), its left side at x0 and its right side at x1: 1 inlet side, 2 bottom,
// 3 outlet side, 4 top, counter-clockwise.  The note's empty-selection example shifts it to x0 = 0.0005, x1 = 0.0405.
S::Table channel(double x0 = 0.0, double x1 = 0.04) {
  S::Table t;
  t.geometry = "planar";
  t.regions = {"channel"};
  t.diagonal_m = 0.04123105625617661;
  t.entities = {
      seg(1, "channel", "exterior", x0, 0.01, x0, 0, 0.01),
      seg(2, "channel", "exterior", x0, 0, x1, 0, 0.04),
      seg(3, "channel", "exterior", x1, 0, x1, 0.01, 0.01),
      seg(4, "channel", "exterior", x1, 0.01, x0, 0.01, 0.04),
  };
  return t;
}
// The channel with a post: a disk of radius 0.002 centred at (0.01, 0.005), a region of its own after the channel.  Its
// circle is two counter-clockwise arcs, 5 the upper (midpoint (0.01, 0.007), tangent -e_x) and 6 the lower (midpoint
// (0.01, 0.003), tangent +e_x); the post is on their left, so side = [post, channel].  Each arc is pi R long.
S::Table channel_post() {
  S::Table t = channel();
  t.regions = {"channel", "post"};
  t.entities.push_back(arc(5, "post", "channel", 0.008, 0.005, 0.012, 0.007, 0.01, 0.007, -1, 0, 0.006283185307179587));
  t.entities.push_back(arc(6, "post", "channel", 0.008, 0.003, 0.012, 0.005, 0.01, 0.003, 1, 0, 0.006283185307179587));
  return t;
}
// The annulus R1 = 0.002, R2 = 0.005 centred at the origin: 1 the outer circle, 2 the inner one, each one closed
// counter-clockwise curve whose parametric midpoint is at angle pi, where the tangent is -e_y.  The ring is on the left
// of the outer circle and on the right of the inner one.  Lengths 2 pi R.
S::Table annulus() {
  S::Table t;
  t.geometry = "planar";
  t.regions = {"ring"};
  t.diagonal_m = 0.01414213562373095;
  t.entities = {
      arc(1, "ring", "exterior", -0.005, -0.005, 0.005, 0.005, -0.005, 0, 0, -1, 0.031415926535897934),
      arc(2, "exterior", "ring", -0.002, -0.002, 0.002, 0.002, -0.002, 0, 0, -1, 0.012566370614359173),
  };
  return t;
}
// The axisymmetric pipe: r in [0, 0.005], z in [0, 0.04], p1 for z < 0.02 and p2 above.  1 and 7 lie on the axis; 2 is
// the inlet (z = 0), 6 the outlet (z = 0.04), 3 and 5 the wall (r = R), 4 the interface z = 0.02, written towards the
// axis so that p1 is on its left.  With axis_marker the axis entity of p1 also carries the marker axm.
S::Table pipe(bool axis_marker = false) {
  S::Table t;
  t.geometry = "axisymmetric";
  t.regions = {"p1", "p2"};
  t.diagonal_m = 0.04031128874149275;
  t.entities = {
      seg(1, "p1", "exterior", 0, 0.02, 0, 0, 0.02, axis_marker ? Strs{"axm"} : Strs{}, true),
      seg(2, "p1", "exterior", 0, 0, 0.005, 0, 0.005),
      seg(3, "p1", "exterior", 0.005, 0, 0.005, 0.02, 0.02),
      seg(4, "p1", "p2", 0.005, 0.02, 0, 0.02, 0.005),
      seg(5, "p2", "exterior", 0.005, 0.02, 0.005, 0.04, 0.02),
      seg(6, "p2", "exterior", 0.005, 0.04, 0, 0.04, 0.005),
      seg(7, "p2", "exterior", 0, 0.04, 0, 0.02, 0.02, {}, true),
  };
  // revolved areas, 2 pi r_c L (Pappus)
  t.entities[1].area_m2 = 7.853981633974484e-05;
  t.entities[2].area_m2 = 0.0006283185307179587;
  return t;
}
json pipe_selectors() {
  return J(R"({"inlet":  {"of": ["p1"], "where": {"z": [0]}},
               "outlet": {"of": ["p2"], "where": {"z": [0.04]}},
               "wall":   {"of": ["p1", "p2"], "where": {"r": [0.005]}}})");
}
// The thin film: 1e-5 m by 1e-6 m (1 bottom, 2 right, 3 top, 4 left).  With inflate, every box is widened by 1e-7 m on
// each side, as gmsh's getBoundingBox reports it.
S::Table film(bool inflate = false) {
  S::Table t;
  t.geometry = "planar";
  t.regions = {"film"};
  t.diagonal_m = 1.0049875621120892e-05;
  t.entities = {
      seg(1, "film", "exterior", 0, 0, 1e-5, 0, 1e-5),
      seg(2, "film", "exterior", 1e-5, 0, 1e-5, 1e-6, 1e-6),
      seg(3, "film", "exterior", 1e-5, 1e-6, 0, 1e-6, 1e-5),
      seg(4, "film", "exterior", 0, 1e-6, 0, 0, 1e-6),
  };
  if (inflate)
    for (S::Entity& e : t.entities)
      for (int c = 0; c < 2; ++c) e.lo[c] -= 1e-7, e.hi[c] += 1e-7;
  return t;
}
// The unit square of the matrix documents, region "box": 11 the side x = 0 (r = 0, the axis, when axisymmetric),
// 12 bottom, 13 right, 14 top.
S::Table square(bool axisymmetric) {
  S::Table t;
  t.geometry = axisymmetric ? "axisymmetric" : "planar";
  t.regions = {"box"};
  t.diagonal_m = 1.4142135623730951;
  t.entities = {
      seg(11, "box", "exterior", 0, 1, 0, 0, 1, {}, axisymmetric),
      seg(12, "box", "exterior", 0, 0, 1, 0, 1),
      seg(13, "box", "exterior", 1, 0, 1, 1, 1),
      seg(14, "box", "exterior", 1, 1, 0, 1, 1),
  };
  return t;
}

const S::Entity& entity(const S::Table& t, int tag) {
  for (const S::Entity& e : t.entities)
    if (e.tag == tag) return e;
  throw std::logic_error("no entity " + std::to_string(tag));
}
// The normal of the selection `name` on entity `tag`: orientation x (t_y, -t_x).
std::pair<double, double> normal(const S::Table& t, const S::Resolution& r, const std::string& name, int tag) {
  const auto it = r.orientation.find({name, tag});
  if (it == r.orientation.end()) throw std::logic_error("no orientation of (" + name + ", " + std::to_string(tag) + ")");
  const S::Entity& e = entity(t, tag);
  return {it->second * e.tangent[1], -it->second * e.tangent[0]};
}
Ints sel(const S::Resolution& r, const std::string& name) {
  const auto it = r.selected.find(name);
  if (it == r.selected.end()) throw std::logic_error("no selection '" + name + "'");
  return it->second;
}
int orient(const S::Resolution& r, const std::string& name, int tag) {
  const auto it = r.orientation.find({name, tag});
  return it == r.orientation.end() ? 0 : it->second;
}
json empty_of(const S::Resolution& r, const std::string& name) {
  const auto it = r.empty.find(name);
  return it == r.empty.end() ? json(nullptr) : it->second;
}
json details(const Error& e) { return e.details().empty() ? json(nullptr) : json::parse(e.details()); }

// One expected atom.
struct A {
  int master;
  std::string kind;
  Strs names;
  std::string s0, s1;
  Ints tags;
};
void check_atoms(const std::string& what, const S::Resolution& r, const std::vector<A>& want) {
  check_eq(what + ": number of atoms", int(r.atoms.size()), int(want.size()));
  for (size_t i = 0; i < std::min(r.atoms.size(), want.size()); ++i) {
    const S::Atom& a = r.atoms[i];
    const A& w = want[i];
    const std::string at = what + ": atom " + std::to_string(i);
    check_eq(at + " master", a.master, w.master);
    check_eq(at + " kind", a.kind, w.kind);
    check_eq(at + " names", a.names, w.names);
    check_eq(at + " side", a.side[0] + "|" + a.side[1], w.s0 + "|" + w.s1);
    check_eq(at + " tags", a.tags, w.tags);
    for (int t : w.tags) {
      const auto it = r.atom_of_tag.find(t);
      check_eq(at + " atom_of_tag[" + std::to_string(t) + "]", it == r.atom_of_tag.end() ? -1 : it->second, w.master);
    }
  }
  size_t n = 0;
  for (const A& w : want) n += w.tags.size();
  check_eq(what + ": atom_of_tag has every boundary entity", int(r.atom_of_tag.size()), int(n));
}

// ======================================================================================================== the checks

// of D: exactly one side in D; never an axis entity, never a marker with both sides in D.  Orientation outward from D.
void check_of_selects_one_sided_entities() {
  const S::Table t = split_rect();
  const S::Resolution r = S::resolve(t, J(R"({"a_of": {"of": ["a"]}, "b_of": {"of": ["b"]}, "ab_of": {"of": ["a", "b"]}})"));
  // a: its three outer sides (1; 2, 3; 7, 8) and the interface 9; not the marker 10 (a|a); nothing of b
  check_eq("of a", sel(r, "a_of"), Ints{1, 2, 3, 7, 8, 9});
  check_eq("of b", sel(r, "b_of"), Ints{4, 5, 6, 9});
  check_eq("of [a, b]: the outer boundary only", sel(r, "ab_of"), Ints{1, 2, 3, 4, 5, 6, 7, 8});
  for (int k : {1, 2, 3, 4, 5, 6, 7, 8}) check_eq("of [a, b] outward on " + std::to_string(k), orient(r, "ab_of", k), +1);
  check_eq("of a on the interface: from a (side[0]) to b", orient(r, "a_of", 9), +1);
  check_eq("of b on the interface: from b (side[1]) to a", orient(r, "b_of", 9), -1);
  const auto n9 = normal(t, r, "b_of", 9);
  CHECK(n9.first == -1.0 && n9.second == 0.0, "of b on the interface: the outward normal of b is -e_x");

  // the right side written top to bottom (tangent -e_y, exterior on its left): of b is still outward, +e_x
  const S::Table tr = split_rect(false, true);
  const S::Resolution rr = S::resolve(tr, J(R"({"b_of": {"of": ["b"]}})"));
  check_eq("of b, right side reversed", sel(rr, "b_of"), Ints{4, 5, 6, 9});
  check_eq("of b on the reversed right side: from b (side[1]) outward", orient(rr, "b_of", 5), -1);
  const auto n5 = normal(tr, rr, "b_of", 5);
  CHECK(n5.first == 1.0 && n5.second == 0.0, "of b on x = L: the outward normal is +e_x");

  // the pipe: of never returns an axis entity
  const S::Table p = pipe();
  const S::Resolution rp = S::resolve(p, J(R"({"p1_of": {"of": ["p1"]}, "all_of": {"of": ["p1", "p2"]}})"));
  check_eq("of p1: no axis entity (1)", sel(rp, "p1_of"), Ints{2, 3, 4});
  check_eq("of [p1, p2]: no axis entity (1, 7), no interface", sel(rp, "all_of"), Ints{2, 3, 5, 6});
  check_eq("of p1 on the interface (p1 = side[0])", orient(rp, "p1_of", 4), +1);
}

// between [A, B]: one side in A, the other in B; symmetric in its result, oriented from A to B; never the axis.
void check_between() {
  const S::Table t = split_rect();
  const S::Resolution r = S::resolve(t, J(R"({"ab":  {"between": [["a"], ["b"]]},
                                              "ba":  {"between": [["b"], ["a"]]},
                                              "ae":  {"between": [["a"], "exterior"]},
                                              "abe": {"between": [["a", "b"], "exterior"]},
                                              "ov":  {"between": [["a", "b"], ["b"]]}})"));
  check_eq("between [a, b]", sel(r, "ab"), Ints{9});
  check_eq("between [b, a]", sel(r, "ba"), Ints{9});
  check_eq("between [a, b]: from a (side[0]) to b", orient(r, "ab", 9), +1);
  check_eq("between [b, a]: from b (side[1]) to a", orient(r, "ba", 9), -1);
  const auto n = normal(t, r, "ab", 9);
  CHECK(n.first == 1.0 && n.second == 0.0, "between [a, b] across x = L/2 points along +e_x");
  check_eq("between [a, exterior]", sel(r, "ae"), Ints{1, 2, 3, 7, 8});
  check_eq("between [[a, b], exterior]", sel(r, "abe"), Ints{1, 2, 3, 4, 5, 6, 7, 8});
  for (int k : {1, 2, 3, 4, 5, 6, 7, 8}) check_eq("between [[a, b], exterior] outward on " + std::to_string(k), orient(r, "abe", k), +1);
  // a shared region: the resolution's overlap diagnostic (validation refuses it with boundary.between_overlap)
  const auto ov = r.overlap.find("ov");
  CHECK(ov != r.overlap.end(), "between [[a, b], [b]]: an overlap diagnostic");
  if (ov != r.overlap.end()) check_eq("between [[a, b], [b]]: the shared regions", ov->second, Strs{"b"});
  for (const char* k : {"ab", "ba", "ae", "abe"}) CHECK(!r.overlap.count(k), std::string("no overlap diagnostic for ") + k);

  // the annulus: the inner circle has the exterior on its left; between [ring, exterior] points out of the ring on both
  const S::Table an = annulus();
  const S::Resolution ra = S::resolve(an, J(R"({"inner": {"between": [["ring"], "exterior"],
                                                          "where": {"x": [-0.002, 0.002], "y": [-0.002, 0.002]}},
                                                "outer": {"between": [["ring"], "exterior"], "except": ["inner"]}})"));
  check_eq("annulus inner", sel(ra, "inner"), Ints{2});
  check_eq("annulus outer", sel(ra, "outer"), Ints{1});
  check_eq("annulus inner: from the ring (side[1]) to the exterior (side[0])", orient(ra, "inner", 2), -1);
  check_eq("annulus outer: from the ring (side[0]) to the exterior", orient(ra, "outer", 1), +1);
  const auto ni = normal(an, ra, "inner", 2), no = normal(an, ra, "outer", 1);
  // at the midpoint (-R, 0) the radial unit vector is -e_x: n . e_r = -n_x
  CHECK(-ni.first == -1.0 && ni.second == 0.0, "annulus inner: the normal points towards the axis of the hole");
  CHECK(-no.first == 1.0 && no.second == 0.0, "annulus outer: the normal points away from the centre");

  // the pipe: between [D, exterior] excludes the axis
  const S::Resolution rp = S::resolve(pipe(), J(R"({"skin": {"between": [["p1", "p2"], "exterior"]}})"));
  check_eq("pipe between [[p1, p2], exterior]: no axis entity", sel(rp, "skin"), Ints{2, 3, 5, 6});

  // AUTHOR TEXT: the builder of boundary.between_overlap (the validation check that raises it is not in this unit)
  const Error e1 = msg::between_overlap("band", "/boundaries/band/between", "wet", "metal", {"bar"});
  check_eq("between_overlap message", e1.message(), "boundary 'band': the two domains of 'between' share the region 'bar'");
  check_eq("between_overlap where", e1.where(), "/boundaries/band/between");
  check_eq("between_overlap hint", e1.hint(),
           "the normal of a 'between' selection points from the first domain to the second, which is undefined on an edge "
           "whose two sides lie in both. Give two domains without a common region, or use {\"of\": DOMAIN}.");
  check_eq("between_overlap code", e1.code(), "boundary.between_overlap");
  check_json("between_overlap details", details(e1), J(R"({"boundary": "band", "between": ["wet", "metal"], "shared": ["bar"]})"));
  const Error e2 = msg::between_overlap("band", "/boundaries/band/union/1/between", "wet", "metal", {"bar", "bolt"});
  check_eq("between_overlap message, two regions", e2.message(),
           "boundary 'band': the two domains of 'between' share the regions 'bar' and 'bolt'");
  check_eq("between_overlap where, nested", e2.where(), "/boundaries/band/union/1/between");
  check_json("between_overlap details.shared", details(e2)["shared"], J(R"(["bar", "bolt"])"));
}

// curve NAME: the marker's entities on the boundary and inside a region, never the axis; the normal is the written
// tangent rotated by +90 degrees.
void check_curve_marker() {
  const S::Table t = split_rect();
  const S::Resolution r = S::resolve(t, J(R"({"mk": {"curve": "mk"}})"));
  check_eq("curve mk (inside a)", sel(r, "mk"), Ints{10});
  check_eq("curve mk: the left normal of the written tangent is the side[1] -> side[0] direction", orient(r, "mk", 10), -1);
  const auto n = normal(t, r, "mk", 10);
  CHECK(n.first == -1.0 && n.second == 0.0, "mk written bottom to top (tangent +e_y): normal -e_x (n . e_x = -1)");

  const S::Table td = split_rect(true);
  const S::Resolution rd = S::resolve(td, J(R"({"mk": {"curve": "mk"}})"));
  check_eq("curve mk written top to bottom", sel(rd, "mk"), Ints{10});
  const auto nd = normal(td, rd, "mk", 10);
  CHECK(nd.first == 1.0 && nd.second == 0.0, "mk written top to bottom (tangent -e_y): normal +e_x (the sign reverses)");

  // a marker on the boundary of the domain (the right side of b, written top to bottom)
  const S::Table tr = split_rect(false, true);
  const S::Resolution rr = S::resolve(tr, J(R"({"rim": {"curve": "rim"}})"));
  check_eq("curve rim (on the boundary)", sel(rr, "rim"), Ints{5});
  const auto nr = normal(tr, rr, "rim", 5);
  CHECK(nr.first == 1.0 && nr.second == 0.0, "rim written top to bottom: normal +e_x");

  // a marker that lies only on r = 0 selects nothing; the axis entity stays in the axis atom
  const S::Table p = pipe(true);
  json sp = pipe_selectors();
  sp["axm"] = J(R"({"curve": "axm"})");
  const S::Resolution rp = S::resolve(p, sp);
  check_eq("curve axm on the axis", sel(rp, "axm"), Ints{});
  check_json("curve axm: the empty diagnostic", empty_of(rp, "axm"),
             J(R"({"where": null, "candidates": [], "partial": [], "axis_only": true})"));
  check_atoms("pipe with a marker on the axis", rp,
              {{900, "axis", {}, "p1", "exterior", {1, 7}},
               {2001, "named", {"inlet"}, "p1", "exterior", {2}},
               {2002, "named", {"outlet"}, "p2", "exterior", {6}},
               {2003, "named", {"wall"}, "p1", "exterior", {3, 5}},
               {2004, "unnamed", {}, "p1", "p2", {4}}});
  // AUTHOR TEXT
  const Error e = msg::empty_selection("axm", J(R"({"curve": "axm"})"), p, empty_of(rp, "axm"));
  check_eq("axis-only message", e.message(), "boundary 'axm' selects nothing");
  check_eq("axis-only where", e.where(), "/boundaries/axm");
  check_eq("axis-only hint", e.hint(),
           "'axm' lies only on the axis r = 0, which is reserved: no named boundary includes it, and every physics applies "
           "its axis condition there automatically.");
  check_eq("axis-only code", e.code(), "boundary.empty_selection");
  check_json("axis-only details", details(e),
             J(R"({"selector": {"curve": "axm"}, "candidates": [], "partial": [], "axis_only": true})"));
}

// where: the whole exact box, within the tolerance; tolerance_m overrides it; a partly matching entity is recorded.
void check_where_uses_the_whole_box() {
  const S::Table t = split_rect();  // tolerance 1e-9 x 0.04123105625617661 = 4.1e-11 m
  const S::Resolution r = S::resolve(t, J(R"({
      "eq_in":      {"of": ["a"], "where": {"x": [1e-11]}},
      "eq_out":     {"of": ["a"], "where": {"x": [1e-10]}},
      "eq_tol":     {"of": ["a"], "where": {"x": [1e-10]}, "tolerance_m": 1e-9},
      "rng_in":     {"of": ["b"], "where": {"y": [0], "x": [0.02000000001, 0.04]}},
      "rng_out":    {"of": ["b"], "where": {"y": [0], "x": [0.0200000001, 0.04]}},
      "half":       {"of": ["a", "b"], "where": {"y": [0], "x": [0, 0.03]}},
      "half_empty": {"of": ["b"], "where": {"y": [0], "x": [0.025, 0.035]}}})"));
  check_eq("x = 1e-11 holds x = 0 within 4.1e-11", sel(r, "eq_in"), Ints{1});
  check_eq("x = 1e-10 does not", sel(r, "eq_out"), Ints{});
  check_eq("tolerance_m 1e-9 overrides", sel(r, "eq_tol"), Ints{1});
  check_eq("a range whose end is 1e-11 inside the box's end", sel(r, "rng_in"), Ints{4});
  check_eq("a range whose end is 1e-10 inside the box's end", sel(r, "rng_out"), Ints{});
  check_json("an equality is never partly satisfied", empty_of(r, "eq_out")["partial"], json::array());
  // the bottom of b (x in [0.02, 0.04]) overlaps [0.0200000001, 0.04] by 0.0199999999 > tol
  check_json("the bottom of b, partly inside the range", empty_of(r, "rng_out")["partial"], J("[4]"));
  // half: the bottoms of a (2, 3) lie in x in [0, 0.03]; the bottom of b (4) is half inside: not selected, recorded
  check_eq("half: the whole boxes", sel(r, "half"), Ints{2, 3});
  CHECK(r.partial.count("half"), "half: a partial diagnostic");
  if (r.partial.count("half")) check_json("half: partial", r.partial.at("half"), J(R"({"tags": [4]})"));
  CHECK(!r.empty.count("half"), "half is not empty");
  for (const char* k : {"eq_in", "eq_tol", "rng_in"}) CHECK(!r.partial.count(k), std::string("no partial diagnostic for ") + k);
  // half_empty: nothing whole, the bottom of b partly; candidates: the edges of b by line (x = 0.02 the interface,
  // x = 0.04, y = 0, y = 0.01)
  check_eq("half_empty", sel(r, "half_empty"), Ints{});
  check_json("half_empty: the empty diagnostic", empty_of(r, "half_empty"),
             J(R"({"where": {"x": [0.025, 0.035], "y": [0]},
                   "candidates": [{"where": {"x": 0.02}, "measure_m": 0.01}, {"where": {"x": 0.04}, "measure_m": 0.01},
                                  {"where": {"y": 0}, "measure_m": 0.02}, {"where": {"y": 0.01}, "measure_m": 0.02}],
                   "partial": [4], "axis_only": false})"));

  // AUTHOR TEXT: boundary.partial_match (warning) and the empty selection with a partial match
  const json w_half = J(R"({"of": ["a", "b"], "where": {"y": 0, "x": [0, 0.03]}})");
  const Error p = msg::partial_match("half", w_half, t, {4});
  check_eq("partial_match message", p.message(), "boundary 'half' leaves out 1 edge that satisfies its where filter in part");
  check_eq("partial_match where", p.where(), "/boundaries/half/where");
  check_eq("partial_match hint", p.hint(),
           "1 edge, 0.02 m, first at (x, y) = (0.03, 0): a where filter keeps an edge only when the whole edge satisfies "
           "it. To select part of an edge, add a curve along that part and use {\"curve\": NAME}.");
  check_eq("partial_match code", p.code(), "boundary.partial_match");
  check_json("partial_match details", details(p),
             J(R"({"boundary": "half", "entities": 1, "measure_m": 0.02, "first_point": [0.03, 0]})"));
  const json w_he = J(R"({"of": "b", "where": {"y": 0, "x": [0.025, 0.035]}})");
  const Error e = msg::empty_selection("half_empty", w_he, t, empty_of(r, "half_empty"));
  check_eq("empty with a partial match: message", e.message(), "boundary 'half_empty' selects nothing");
  check_eq("empty with a partial match: where", e.where(), "/boundaries/half_empty/where");
  check_eq("empty with a partial match: hint", e.hint(),
           "1 edge of 'b' lies partly within x in [0.025, 0.035] and y = 0 (0.02 m, first at (x, y) = (0.03, 0)); a where "
           "filter keeps an edge only when the whole edge satisfies it. To select part of an edge, add a curve along that "
           "part and use {\"curve\": NAME}.");
  check_json("empty with a partial match: details.partial", details(e)["partial"], J("[4]"));

  // the exact box against gmsh's inflated one (1e-7 m) on a 1e-5 m model, whose tolerance is 1.0e-14 m
  const json top = J(R"({"top": {"of": ["film"], "where": {"y": [1e-6]}}})");
  check_eq("thin film: the exact box passes y = 1e-6", sel(S::resolve(film(), top), "top"), Ints{3});
  check_eq("thin film: an inflated box fails it", sel(S::resolve(film(true), top), "top"), Ints{});
  check_eq("thin film: tolerance_m 2e-7 accepts the inflated box",
           sel(S::resolve(film(true), J(R"({"top": {"of": ["film"], "where": {"y": [1e-6]}, "tolerance_m": 2e-7}})")), "top"),
           Ints{3});
}

// except, union, difference, ref, nested; orientation through them.
void check_set_operations() {
  const S::Table t = split_rect();
  const S::Resolution r = S::resolve(t, J(R"({
      "left":       {"of": ["a"], "where": {"x": [0]}},
      "bottom_all": {"of": ["a", "b"], "where": {"y": [0]}},
      "skin":       {"of": ["a", "b"]},
      "u":          {"union": [{"ref": "left"}, {"ref": "bottom_all"}]},
      "d":          {"difference": [{"ref": "skin"}, {"ref": "left"}, {"ref": "bottom_all"}]},
      "e":          {"of": ["a", "b"], "except": ["left", "bottom_all"]},
      "n":          {"difference": [{"union": [{"ref": "u"}, {"curve": "mk"}]}, {"of": ["b"], "where": {"y": [0]}}]},
      "r":          {"ref": "n"},
      "exb":        {"between": [["a"], "exterior"], "except": ["left"]},
      "uf":         {"union": [{"between": [["b"], ["a"]]}, {"between": [["a"], ["b"]]}]}})"));
  check_eq("left", sel(r, "left"), Ints{1});
  check_eq("bottom_all", sel(r, "bottom_all"), Ints{2, 3, 4});
  check_eq("skin", sel(r, "skin"), Ints{1, 2, 3, 4, 5, 6, 7, 8});
  check_eq("union", sel(r, "u"), Ints{1, 2, 3, 4});
  check_eq("difference of three", sel(r, "d"), Ints{5, 6, 7, 8});
  check_eq("except of two", sel(r, "e"), Ints{5, 6, 7, 8});
  check_eq("nested difference of a union", sel(r, "n"), Ints{1, 2, 3, 10});
  check_eq("ref", sel(r, "r"), Ints{1, 2, 3, 10});
  check_eq("except on between", sel(r, "exb"), Ints{2, 3, 7, 8});
  check_eq("union of two betweens", sel(r, "uf"), Ints{9});
  check_eq("ref keeps the marker's orientation", orient(r, "r", 10), -1);
  for (int k : {1, 2, 3}) check_eq("ref keeps the outward orientation on " + std::to_string(k), orient(r, "r", k), +1);
  check_eq("union: the first operand that selects the entity orients it (between [b, a])", orient(r, "uf", 9), -1);
  check_eq("difference: the first operand orients", orient(r, "d", 5), +1);

  // the note's walls: of channel except inlet, outlet, cylinder
  const S::Resolution rc = S::resolve(channel_post(), J(R"({
      "inlet":    {"of": ["channel"], "where": {"x": [0]}},
      "outlet":   {"of": ["channel"], "where": {"x": [0.04]}},
      "cylinder": {"between": [["channel"], ["post"]]},
      "walls":    {"of": ["channel"], "except": ["inlet", "outlet", "cylinder"]}})"));
  check_eq("channel inlet", sel(rc, "inlet"), Ints{1});
  check_eq("channel outlet", sel(rc, "outlet"), Ints{3});
  check_eq("channel cylinder", sel(rc, "cylinder"), Ints{5, 6});
  check_eq("channel walls", sel(rc, "walls"), Ints{2, 4});
  for (int k : {5, 6}) check_eq("cylinder: from the channel (side[1]) to the post", orient(rc, "cylinder", k), -1);
  const auto n5 = normal(channel_post(), rc, "cylinder", 5);
  CHECK(n5.first == 0.0 && n5.second == -1.0, "cylinder at the top of the post points down, into the post");
}

// The note's example of an empty selection: the channel shifted by 0.0005 in x, "inlet" at x = 0.
void check_empty_selection_candidates() {
  const S::Table t = channel(0.0005, 0.0405);
  const S::Resolution r = S::resolve(t, J(R"({"inlet": {"of": ["channel"], "where": {"x": [0]}},
                                              "far":   {"of": ["channel"], "where": {"x": [0.05, 0.06]}}})"));
  check_eq("shifted inlet", sel(r, "inlet"), Ints{});
  const json cands = J(R"([{"where": {"x": 0.0005}, "measure_m": 0.01}, {"where": {"x": 0.0405}, "measure_m": 0.01},
                           {"where": {"y": 0}, "measure_m": 0.04}, {"where": {"y": 0.01}, "measure_m": 0.04}])");
  check_json("shifted inlet: the empty diagnostic", empty_of(r, "inlet"),
             json{{"where", J(R"({"x": [0]})")}, {"candidates", cands}, {"partial", json::array()}, {"axis_only", false}});
  // NOTE TEXT (design note 5.2), with details as the note shows them plus "partial" and "axis_only"
  const json written = J(R"({"of": "channel", "where": {"x": 0}})");
  const Error e = msg::empty_selection("inlet", written, t, empty_of(r, "inlet"));
  check_eq("note: message", e.message(), "boundary 'inlet' selects nothing");
  check_eq("note: where", e.where(), "/boundaries/inlet/where");
  check_eq("note: hint", e.hint(),
           "no boundary edge of 'channel' lies on x = 0. Edges of 'channel': x = 0.0005 (0.01 m), x = 0.0405 (0.01 m), "
           "y = 0 (0.04 m), y = 0.01 (0.04 m). Did you mean {\"x\": 0.0005}, or is the geometry shifted? To select part of "
           "an edge, add a curve along that part and use {\"curve\": NAME}.");
  check_eq("note: code", e.code(), "boundary.empty_selection");
  check_json("note: details", details(e),
             json{{"selector", written}, {"candidates", cands}, {"partial", json::array()}, {"axis_only", false}});
  // AUTHOR TEXT: a range (no equality to suggest a value for)
  const Error f = msg::empty_selection("far", J(R"({"of": "channel", "where": {"x": [0.05, 0.06]}})"), t, empty_of(r, "far"));
  check_eq("range: hint", f.hint(),
           "no boundary edge of 'channel' lies within x in [0.05, 0.06]. Edges of 'channel': x = 0.0005 (0.01 m), x = 0.0405 "
           "(0.01 m), y = 0 (0.04 m), y = 0.01 (0.04 m). Is the geometry shifted? To select part of an edge, add a curve "
           "along that part and use {\"curve\": NAME}.");

  // between, and curved candidates (no coordinate in which the box is flat): written as boxes, sorted by point
  const S::Table cp = channel_post();
  const S::Resolution rc = S::resolve(cp, J(R"({"cyl_x": {"between": [["channel"], ["post"]], "where": {"x": [0.02]}},
                                                "none":  {"between": [["post"], "exterior"]}})"));
  check_eq("cyl_x", sel(rc, "cyl_x"), Ints{});
  const json arcs = J(R"([{"bbox": {"x": [0.008, 0.012], "y": [0.003, 0.005]}, "measure_m": 0.006283185307179587},
                          {"bbox": {"x": [0.008, 0.012], "y": [0.005, 0.007]}, "measure_m": 0.006283185307179587}])");
  check_json("cyl_x: candidates", empty_of(rc, "cyl_x")["candidates"], arcs);
  const Error c = msg::empty_selection("cyl_x", J(R"({"between": ["channel", "post"], "where": {"x": 0.02}})"), cp,
                                       empty_of(rc, "cyl_x"));
  check_eq("between: hint", c.hint(),
           "no edge between 'channel' and 'post' lies on x = 0.02. Edges between 'channel' and 'post': x in [0.008, 0.012], "
           "y in [0.003, 0.005] (0.006283185307179587 m), x in [0.008, 0.012], y in [0.005, 0.007] (0.006283185307179587 m). "
           "Is the geometry shifted? To select part of an edge, add a curve along that part and use {\"curve\": NAME}.");
  // a selection empty without a where filter
  check_eq("none", sel(rc, "none"), Ints{});
  check_json("none: the empty diagnostic", empty_of(rc, "none"),
             J(R"({"where": null, "candidates": [], "partial": [], "axis_only": false})"));
  const Error d = msg::empty_selection("none", J(R"({"between": ["post", "exterior"]})"), cp, empty_of(rc, "none"));
  check_eq("no where: where", d.where(), "/boundaries/none");
  check_eq("no where: hint", d.hint(),
           "'none' selects no boundary edge of this geometry. `axmp boundaries` lists every boundary edge and the boundaries "
           "that select it.");
}

// Atoms are membership signatures; unnamed entities form one atom per region pair.
void check_atoms_are_signatures() {
  const S::Table cp = channel_post();
  // inlet and skin overlap: atoms [cylinder], [inlet, skin], [skin]
  const S::Resolution r1 = S::resolve(cp, J(R"({"inlet":    {"of": ["channel"], "where": {"x": [0]}},
                                                "skin":     {"of": ["channel", "post"]},
                                                "cylinder": {"between": [["channel"], ["post"]]}})"));
  check_eq("skin", sel(r1, "skin"), Ints{1, 2, 3, 4});
  check_atoms("inlet and skin", r1,
              {{2001, "named", {"cylinder"}, "post", "channel", {5, 6}},
               {2002, "named", {"inlet", "skin"}, "channel", "exterior", {1}},
               {2003, "named", {"skin"}, "channel", "exterior", {2, 3, 4}}});
  // the note's channel with a post: except shapes the atoms
  const S::Resolution r2 = S::resolve(cp, J(R"({"inlet":    {"of": ["channel"], "where": {"x": [0]}},
                                                "outlet":   {"of": ["channel"], "where": {"x": [0.04]}},
                                                "cylinder": {"between": [["channel"], ["post"]]},
                                                "walls":    {"of": ["channel"], "except": ["inlet", "outlet", "cylinder"]}})"));
  check_atoms("channel with walls", r2,
              {{2001, "named", {"cylinder"}, "post", "channel", {5, 6}},
               {2002, "named", {"inlet"}, "channel", "exterior", {1}},
               {2003, "named", {"outlet"}, "channel", "exterior", {3}},
               {2004, "named", {"walls"}, "channel", "exterior", {2, 4}}});
  // unnamed: one atom per region pair, the pair in region order (the arcs have the post at side[0]), exterior last
  const S::Resolution r3 = S::resolve(cp, J(R"({"inlet":  {"of": ["channel"], "where": {"x": [0]}},
                                                "outlet": {"of": ["channel"], "where": {"x": [0.04]}}})"));
  check_atoms("channel, unnamed pairs", r3,
              {{2001, "named", {"inlet"}, "channel", "exterior", {1}},
               {2002, "named", {"outlet"}, "channel", "exterior", {3}},
               {2003, "unnamed", {}, "channel", "post", {5, 6}},
               {2004, "unnamed", {}, "channel", "exterior", {2, 4}}});
  // a marker that no boundary names is the unnamed atom of its pair (a, a), which sorts first
  const S::Resolution r4 = S::resolve(split_rect(), J(R"({"hot":  {"of": ["a"], "where": {"x": [0]}},
                                                          "cold": {"of": ["b"], "where": {"x": [0.04]}}})"));
  check_atoms("split rectangle, hot and cold", r4,
              {{2001, "named", {"cold"}, "b", "exterior", {5}},
               {2002, "named", {"hot"}, "a", "exterior", {1}},
               {2003, "unnamed", {}, "a", "a", {10}},
               {2004, "unnamed", {}, "a", "b", {9}},
               {2005, "unnamed", {}, "a", "exterior", {2, 3, 7, 8}},
               {2006, "unnamed", {}, "b", "exterior", {4, 6}}});
}

// Master numbers: axis 900, named 2001... by sorted name lists, unnamed after them by region pair.
void check_master_numbering() {
  namespace F = axmp::model::fixtures;
  // the unit square of the matrix documents
  for (bool axi : {false, true}) {
    const S::Table t = square(axi);
    const json sq = axi ? J(R"({"right": {"of": ["box"], "where": {"r": [1]}}, "bottom": {"of": ["box"], "where": {"z": [0]}},
                               "top": {"of": ["box"], "where": {"z": [1]}}})")
                        : J(R"({"left": {"of": ["box"], "where": {"x": [0]}}, "right": {"of": ["box"], "where": {"x": [1]}},
                               "bottom": {"of": ["box"], "where": {"y": [0]}}, "top": {"of": ["box"], "where": {"y": [1]}}})");
    const S::Resolution r = S::resolve(t, sq);
    const auto want = F::master_numbers(axi ? F::Geometry::axisymmetric : F::Geometry::planar);
    const std::string g = axi ? "axisymmetric square" : "planar square";
    std::map<std::string, int> got;
    for (const S::Atom& a : r.atoms) got[a.kind == "axis" ? "axis" : (a.names.size() == 1 ? a.names[0] : "?")] = a.master;
    CHECK(got == want, g + ": the master numbers of fixtures::master_numbers()");
    if (axi)
      check_atoms(g, r,
                  {{900, "axis", {}, "box", "exterior", {11}},
                   {2001, "named", {"bottom"}, "box", "exterior", {12}},
                   {2002, "named", {"right"}, "box", "exterior", {13}},
                   {2003, "named", {"top"}, "box", "exterior", {14}}});
    else
      check_atoms(g, r,
                  {{2001, "named", {"bottom"}, "box", "exterior", {12}},
                   {2002, "named", {"left"}, "box", "exterior", {11}},
                   {2003, "named", {"right"}, "box", "exterior", {13}},
                   {2004, "named", {"top"}, "box", "exterior", {14}}});
  }
  // lists of several names: [cold, skin] < [hot, skin] < [interface] < [mk] < [skin]
  json s = split_rect_selectors();
  s["skin"] = J(R"({"of": ["a", "b"]})");
  check_atoms("split rectangle with skin", S::resolve(split_rect(), s),
              {{2001, "named", {"cold", "skin"}, "b", "exterior", {5}},
               {2002, "named", {"hot", "skin"}, "a", "exterior", {1}},
               {2003, "named", {"interface"}, "a", "b", {9}},
               {2004, "named", {"mk"}, "a", "a", {10}},
               {2005, "named", {"skin"}, "a", "exterior", {2, 3, 4, 6, 7, 8}}});
  // the base split rectangle: the named atoms first, then (a, exterior) and (b, exterior)
  check_atoms("split rectangle", S::resolve(split_rect(), split_rect_selectors()),
              {{2001, "named", {"cold"}, "b", "exterior", {5}},
               {2002, "named", {"hot"}, "a", "exterior", {1}},
               {2003, "named", {"interface"}, "a", "b", {9}},
               {2004, "named", {"mk"}, "a", "a", {10}},
               {2005, "unnamed", {}, "a", "exterior", {2, 3, 7, 8}},
               {2006, "unnamed", {}, "b", "exterior", {4, 6}}});
  // the pipe: the axis first, the interface unnamed
  check_atoms("pipe", S::resolve(pipe(), pipe_selectors()),
              {{900, "axis", {}, "p1", "exterior", {1, 7}},
               {2001, "named", {"inlet"}, "p1", "exterior", {2}},
               {2002, "named", {"outlet"}, "p2", "exterior", {6}},
               {2003, "named", {"wall"}, "p1", "exterior", {3, 5}},
               {2004, "unnamed", {}, "p1", "p2", {4}}});
}

// The numbering depends on names and regions only: a shuffled, retagged table gives the same atoms.
void check_numbering_ignores_tags_and_entity_order() {
  json s = split_rect_selectors();
  s["skin"] = J(R"({"of": ["a", "b"]})");
  s["bottom_all"] = J(R"({"of": ["a", "b"], "where": {"y": [0]}})");
  const std::vector<std::pair<S::Table, json>> cases = {{split_rect(), s}, {pipe(), pipe_selectors()}, {channel_post(),
      J(R"({"inlet": {"of": ["channel"], "where": {"x": [0]}}, "outlet": {"of": ["channel"], "where": {"x": [0.04]}}})")}};
  int n = 0;
  for (const auto& [t, sels] : cases) {
    const std::string what = "case " + std::to_string(n++);
    S::Table u = t;
    std::reverse(u.entities.begin(), u.entities.end());
    std::rotate(u.entities.begin(), u.entities.begin() + 3, u.entities.end());
    std::map<int, int> to;  // old tag -> new tag
    for (S::Entity& e : u.entities) {
      to[e.tag] = 1000 - 37 * e.tag;
      e.tag = to[e.tag];
    }
    auto mapped = [&](const Ints& v) {
      Ints w;
      for (int x : v) w.push_back(to.at(x));
      std::sort(w.begin(), w.end());
      return w;
    };
    const S::Resolution a = S::resolve(t, sels), b = S::resolve(u, sels);
    check_eq(what + ": number of atoms", int(b.atoms.size()), int(a.atoms.size()));
    for (size_t i = 0; i < std::min(a.atoms.size(), b.atoms.size()); ++i) {
      const std::string at = what + ": atom " + std::to_string(i);
      check_eq(at + " master", b.atoms[i].master, a.atoms[i].master);
      check_eq(at + " kind", b.atoms[i].kind, a.atoms[i].kind);
      check_eq(at + " names", b.atoms[i].names, a.atoms[i].names);
      check_eq(at + " tags", b.atoms[i].tags, mapped(a.atoms[i].tags));
      if (a.atoms[i].kind == "unnamed")
        check_eq(at + " side", b.atoms[i].side[0] + "|" + b.atoms[i].side[1], a.atoms[i].side[0] + "|" + a.atoms[i].side[1]);
    }
    for (const auto& [name, tags] : a.selected) {
      check_eq(what + ": selection " + name, b.selected.count(name) ? b.selected.at(name) : Ints{-1}, mapped(tags));
      for (int k : tags) check_eq(what + ": orientation " + name, orient(b, name, to.at(k)), orient(a, name, k));
    }
    for (const auto& [tag, atom] : a.atom_of_tag) {
      const auto it = b.atom_of_tag.find(to.at(tag));
      check_eq(what + ": atom_of_tag", it == b.atom_of_tag.end() ? -1 : it->second, atom);
    }
  }
}

// The partition rule: the three refusals of note 5.5, the partly interior key, the key outside the domain.
void check_partition_refusals() {
  // ---- uncovered: the note's channel without walls
  const S::Table ch = channel();
  const json chs = J(R"({"inlet":  {"of": ["channel"], "where": {"x": [0]}},
                         "outlet": {"of": ["channel"], "where": {"x": [0.04]}},
                         "west":   {"of": ["channel"], "where": {"x": [0]}},
                         "walls":  {"of": ["channel"], "except": ["inlet", "outlet"]}})");
  const S::Resolution rc = S::resolve(ch, chs);
  const S::Partition pu = S::partition(ch, rc, {"channel"}, {"inlet", "outlet"});
  check_eq("uncovered: the bottom and the top", pu.uncovered, Ints{2, 4});
  CHECK(pu.doubly.empty() && pu.interior_keys_tags.empty() && pu.outside_keys.empty() && pu.outside_part.empty(),
        "uncovered: no other finding");
  // NOTE TEXT (design note 5.5)
  const Error u = msg::boundary_uncovered("flow", "channel", {"inlet", "outlet"}, "wall", "channel.json", ch, pu.uncovered);
  check_eq("uncovered message", u.message(), "physics 'flow': part of the boundary of its domain has no condition");
  check_eq("uncovered where", u.where(), "/physics/flow/boundary_conditions");
  check_eq("uncovered hint", u.hint(),
           "uncovered: 2 edges, 0.08 m, between 'channel' and the exterior, first at (x, y) = (0.02, 0). Name them (for "
           "example \"walls\": {\"of\": \"channel\", \"except\": [\"inlet\", \"outlet\"]}) and give them a condition, or add "
           "\"remaining\": {\"kind\": \"wall\"} to this physics. `axmp boundaries channel.json` lists every boundary.");
  check_eq("uncovered code", u.code(), "physics.boundary_uncovered");
  check_json("uncovered details", details(u),
             J(R"({"physics": "flow", "entities": 2, "measure_m": 0.08, "first_point": [0.02, 0],
                   "adjacent": [["channel", "exterior"]], "conditions": ["inlet", "outlet"]})"));
  // AUTHOR TEXT: two pairs, no file name; heat on b keyed by cold only: 4 and 6 (b|ext) and the interface 9 (a|b)
  const S::Table sr = split_rect();
  const S::Resolution rs = S::resolve(sr, split_rect_selectors());
  const S::Partition pb = S::partition(sr, rs, {"b"}, {"cold"});
  check_eq("uncovered on b", pb.uncovered, Ints{4, 6, 9});
  const Error u2 = msg::boundary_uncovered("heat", "b", {"cold"}, "insulation", "", sr, pb.uncovered);
  check_eq("uncovered hint, two pairs", u2.hint(),
           "uncovered: 3 edges, 0.05 m, between 'a' and 'b' and between 'b' and the exterior, first at (x, y) = (0.02, "
           "0.005). Name them (for example \"walls\": {\"of\": \"b\", \"except\": [\"cold\"]}) and give them a condition, or "
           "add \"remaining\": {\"kind\": \"insulation\"} to this physics. `axmp boundaries DOCUMENT` lists every boundary.");
  check_json("uncovered details, two pairs", details(u2)["adjacent"], J(R"([["a", "b"], ["b", "exterior"]])"));
  // AUTHOR TEXT: axisymmetric, one edge; the pipe's p1 keyed by inlet and wall: the interface is uncovered, the part of
  // wall on p2 is taken nowhere
  const S::Table pp = pipe();
  const S::Resolution rpp = S::resolve(pp, pipe_selectors());
  const S::Partition p1 = S::partition(pp, rpp, {"p1"}, {"inlet", "wall"});
  check_eq("pipe p1: uncovered interface", p1.uncovered, Ints{4});
  check_eq("pipe p1: the axis atom", p1.axis_atoms, Ints{900});
  CHECK(p1.outside_part.count("wall"), "pipe p1: wall has a part off the instance");
  if (p1.outside_part.count("wall")) check_eq("pipe p1: wall's part on p2", p1.outside_part.at("wall"), Ints{5});
  const Error u3 = msg::boundary_uncovered("flow", "p1", {"inlet", "wall"}, "wall", "pipe.json", pp, p1.uncovered);
  check_eq("uncovered hint, axisymmetric", u3.hint(),
           "uncovered: 1 edge, 0.005 m, between 'p1' and 'p2', first at (r, z) = (0.0025, 0.02). Name them (for example "
           "\"walls\": {\"of\": \"p1\", \"except\": [\"inlet\", \"wall\"]}) and give them a condition, or add \"remaining\": "
           "{\"kind\": \"wall\"} to this physics. `axmp boundaries pipe.json` lists every boundary.");
  check_json("uncovered details, axisymmetric", details(u3)["first_point"], J("[0.0025, 0.02]"));

  // ---- double: the note's inlet and west
  const S::Partition pd = S::partition(ch, rc, {"channel"}, {"inlet", "west", "outlet", "walls"});
  CHECK(pd.uncovered.empty(), "double: nothing uncovered");
  check_eq("double: entity 1", pd.doubly.count(1) ? pd.doubly.at(1) : Strs{}, Strs{"inlet", "west"});
  check_eq("double: one entity", int(pd.doubly.size()), 1);
  const Error d = msg::boundary_double("flow", "inlet", "west", ch, {1});
  check_eq("double message", d.message(), "physics 'flow': boundaries 'inlet' and 'west' both cover 1 edge at (x, y) = (0, 0.005)");
  check_eq("double where", d.where(), "/physics/flow/boundary_conditions/west");
  check_eq("double hint", d.hint(), "one condition per boundary face. Remove one, or exclude the overlap: {\"of\": ..., \"except\": [\"inlet\"]}.");
  check_eq("double code", d.code(), "physics.boundary_double");
  check_json("double details", details(d),
             J(R"({"physics": "flow", "keys": ["inlet", "west"], "entities": 1, "measure_m": 0.01, "first_point": [0, 0.005]})"));
  // AUTHOR TEXT: several edges
  const Error d2 = msg::boundary_double("heat", "skin", "bottom_all", sr, {2, 3, 4});
  check_eq("double message, three edges", d2.message(),
           "physics 'heat': boundaries 'skin' and 'bottom_all' both cover 3 edges, 0.04 m, first at (x, y) = (0.005, 0)");
  check_eq("double where, three edges", d2.where(), "/physics/heat/boundary_conditions/bottom_all");

  // ---- interior, whole: the note's cylinder in a flow over the channel and the post
  const S::Table cp = channel_post();
  const S::Resolution rcp = S::resolve(cp, J(R"({"inlet":    {"of": ["channel"], "where": {"x": [0]}},
                                                 "outlet":   {"of": ["channel"], "where": {"x": [0.04]}},
                                                 "cylinder": {"between": [["channel"], ["post"]]},
                                                 "walls":    {"of": ["channel"], "except": ["inlet", "outlet", "cylinder"]}})"));
  const S::Partition pi = S::partition(cp, rcp, {"channel", "post"}, {"inlet", "outlet", "walls", "cylinder"});
  CHECK(pi.interior_keys_tags.count("cylinder"), "interior: cylinder");
  if (pi.interior_keys_tags.count("cylinder")) check_eq("interior: cylinder's entities", pi.interior_keys_tags.at("cylinder"), Ints{5, 6});
  CHECK(pi.uncovered.empty() && pi.doubly.empty() && pi.outside_keys.empty(), "interior: no other finding");
  // NOTE TEXT (design note 5.5)
  const Error i = msg::boundary_interior("flow", "cylinder", true, cp, {5, 6});
  check_eq("interior message", i.message(), "physics 'flow': boundary 'cylinder' is not on the boundary of its domain");
  check_eq("interior where", i.where(), "/physics/flow/boundary_conditions/cylinder");
  check_eq("interior hint", i.hint(),
           "'cylinder' lies between 'channel' and 'post', and both regions belong to 'flow'. An interior boundary can be "
           "used in outputs, not as a boundary condition.");
  check_eq("interior code", i.code(), "physics.boundary_interior");
  check_json("interior details", details(i),
             J(R"({"physics": "flow", "boundary": "cylinder", "whole": true, "entities": 2, "measure_m": 0.012566370614359173,
                   "first_point": [0.01, 0.003], "adjacent": [["channel", "post"]]})"));
  // AUTHOR TEXT: a marker inside one region
  const S::Partition pm = S::partition(sr, rs, {"a"}, {"hot", "mk", "remaining"});
  CHECK(pm.interior_keys_tags.count("mk"), "interior: the marker inside a");
  if (pm.interior_keys_tags.count("mk")) check_eq("interior: mk", pm.interior_keys_tags.at("mk"), Ints{10});
  const Error im = msg::boundary_interior("flow", "mk", true, sr, {10});
  check_eq("interior hint, marker", im.hint(),
           "'mk' lies inside 'a', which belongs to 'flow'. An interior boundary can be used in outputs, not as a boundary "
           "condition.");

  // ---- interior, partial: of a keyed by an instance over a and b (its interface part is interior)
  json sa = split_rect_selectors();
  sa["a_skin"] = J(R"({"of": ["a"]})");
  const S::Resolution rsa = S::resolve(sr, sa);
  const S::Partition pp5 = S::partition(sr, rsa, {"a", "b"}, {"a_skin", "cold", "remaining"});
  CHECK(pp5.interior_keys_tags.count("a_skin"), "partly interior: a_skin");
  if (pp5.interior_keys_tags.count("a_skin")) check_eq("partly interior: the interface", pp5.interior_keys_tags.at("a_skin"), Ints{9});
  // AUTHOR TEXT
  const Error ip = msg::boundary_interior("heat", "a_skin", false, sr, {9});
  check_eq("partly interior message", ip.message(), "physics 'heat': part of boundary 'a_skin' is interior to its domain");
  check_eq("partly interior where", ip.where(), "/physics/heat/boundary_conditions/a_skin");
  check_eq("partly interior hint", ip.hint(),
           "1 edge of 'a_skin', 0.01 m, first at (x, y) = (0.02, 0.005), lies between 'a' and 'b', inside the domain of "
           "'heat'. A condition acts on all of its boundary, and an interior boundary can be used in outputs, not as a "
           "boundary condition: exclude the interior part with \"except\", or key a boundary without it.");
  check_eq("partly interior code", ip.code(), "physics.boundary_interior");
  check_json("partly interior details", details(ip),
             J(R"({"physics": "heat", "boundary": "a_skin", "whole": false, "entities": 1, "measure_m": 0.01,
                   "first_point": [0.02, 0.005], "adjacent": [["a", "b"]]})"));

  // ---- outside the domain: hot keyed by an instance on b; a key whose selection is empty is not "outside"
  json so = split_rect_selectors();
  so["nothing"] = J(R"({"of": ["a"], "where": {"x": [0.5]}})");
  const S::Resolution rso = S::resolve(sr, so);
  const S::Partition po = S::partition(sr, rso, {"b"}, {"hot", "cold", "nothing", "remaining"});
  check_eq("outside: hot", po.outside_keys, Strs{"hot"});
  CHECK(po.outside_part.empty(), "outside: hot is not also listed under outside_part");
  CHECK(po.interior_keys_tags.empty() && po.uncovered.empty() && po.doubly.empty(), "outside: no other finding");
  check_eq("outside: the empty key covers nothing", po.atoms_of_key.count("nothing") ? po.atoms_of_key.at("nothing") : Ints{-1}, Ints{});
  // AUTHOR TEXT
  const Error o = msg::boundary_outside_domain("heat", "hot", {"b"}, sr, {1});
  check_eq("outside message", o.message(), "physics 'heat': boundary 'hot' lies outside its domain");
  check_eq("outside where", o.where(), "/physics/heat/boundary_conditions/hot");
  check_eq("outside hint", o.hint(),
           "'hot' lies between 'a' and the exterior, and the domain of 'heat' is 'b'. A condition key names a boundary on "
           "the boundary of the domain: key another boundary, or change the domains of 'heat'.");
  check_eq("outside code", o.code(), "physics.boundary_outside_domain");
  check_json("outside details", details(o),
             J(R"({"physics": "heat", "boundary": "hot", "adjacent": [["a", "exterior"]], "domain": ["b"]})"));
}

// A key part of whose selection lies off its instance is taken where it lies on the domain boundary.
void check_key_partly_outside_is_taken_where_it_lies() {
  const S::Table t = split_rect();
  const S::Resolution r = S::resolve(t, J(R"({"ab_skin": {"of": ["a", "b"]}, "interface": {"between": [["a"], ["b"]]}})"));
  check_atoms("ab_skin", r,
              {{2001, "named", {"ab_skin"}, "a", "exterior", {1, 2, 3, 4, 5, 6, 7, 8}},
               {2002, "named", {"interface"}, "a", "b", {9}},
               {2003, "unnamed", {}, "a", "a", {10}}});
  const S::Partition p = S::partition(t, r, {"a"}, {"ab_skin", "interface"});
  CHECK(p.uncovered.empty() && p.doubly.empty() && p.interior_keys_tags.empty() && p.outside_keys.empty(), "no finding");
  CHECK(p.outside_part.count("ab_skin"), "ab_skin has a part off the instance");
  if (p.outside_part.count("ab_skin")) check_eq("ab_skin's part on b", p.outside_part.at("ab_skin"), Ints{4, 5, 6});
  check_eq("ab_skin covers its atom", p.atoms_of_key.count("ab_skin") ? p.atoms_of_key.at("ab_skin") : Ints{}, Ints{2001});
  check_eq("interface covers its atom", p.atoms_of_key.count("interface") ? p.atoms_of_key.at("interface") : Ints{}, Ints{2002});
  const S::Placement pl = S::place(t, r, p, {"a"}, "ab_skin");
  check_eq("placement of ab_skin: boundary", pl.boundary, Ints{1, 2, 3, 7, 8});
  check_eq("placement of ab_skin: interior", pl.interior, Ints{});
  check_eq("placement of ab_skin: outside", pl.outside, Ints{4, 5, 6});
  check_eq("placement of ab_skin: keys", pl.keys, Strs{"ab_skin"});
}

// "remaining": the rest of the domain boundary, named atoms included; never the axis, never an interior entity.
void check_remaining() {
  const S::Table t = split_rect();
  const S::Resolution r = S::resolve(t, split_rect_selectors());
  auto aok = [](const S::Partition& p, const std::string& k) { return p.atoms_of_key.count(k) ? p.atoms_of_key.at(k) : Ints{-1}; };
  const S::Partition pab = S::partition(t, r, {"a", "b"}, {"hot", "cold", "remaining"});
  check_eq("[a, b]: remaining is the unnamed outer atoms, not the interface or the marker", aok(pab, "remaining"), Ints{2005, 2006});
  check_eq("[a, b]: hot", aok(pab, "hot"), Ints{2002});
  check_eq("[a, b]: cold", aok(pab, "cold"), Ints{2001});
  CHECK(pab.uncovered.empty() && pab.doubly.empty() && pab.interior_keys_tags.empty() && pab.outside_keys.empty() &&
            pab.outside_part.empty() && pab.axis_atoms.empty(),
        "[a, b]: no finding");
  const S::Partition pa = S::partition(t, r, {"a"}, {"hot", "remaining"});
  check_eq("[a]: remaining includes the interface atom, which no key of a names", aok(pa, "remaining"), Ints{2003, 2005});
  CHECK(pa.uncovered.empty() && pa.doubly.empty() && pa.interior_keys_tags.empty(), "[a]: no finding");
  // the pipe: never the axis
  const S::Table p = pipe();
  const S::Resolution rp = S::resolve(p, pipe_selectors());
  const S::Partition pp = S::partition(p, rp, {"p1", "p2"}, {"inlet", "outlet", "remaining"});
  check_eq("pipe: remaining is the wall, not the axis", aok(pp, "remaining"), Ints{2003});
  check_eq("pipe: the axis atom", pp.axis_atoms, Ints{900});
  CHECK(pp.uncovered.empty() && pp.doubly.empty() && pp.interior_keys_tags.empty(), "pipe: no finding");
  const S::Partition pp2 = S::partition(p, rp, {"p2"}, {"outlet", "remaining"});
  check_eq("pipe p2: remaining is its wall and the interface", aok(pp2, "remaining"), Ints{2003, 2004});
  // everything keyed: remaining is empty, and that is no finding
  json s = split_rect_selectors();
  s["skin"] = J(R"({"of": ["a", "b"]})");
  const S::Resolution rk = S::resolve(t, s);
  const S::Partition pk = S::partition(t, rk, {"a", "b"}, {"skin", "remaining"});
  check_eq("everything keyed: remaining", aok(pk, "remaining"), Ints{});
  CHECK(pk.uncovered.empty() && pk.doubly.empty(), "everything keyed: no finding");
}

// Overlapping boundaries keyed by two instances are accepted; by one, refused.
void check_overlap_across_instances() {
  const S::Table t = split_rect();
  json s = split_rect_selectors();
  s["skin"] = J(R"({"of": ["a", "b"]})");
  const S::Resolution r = S::resolve(t, s);
  const S::Partition heat = S::partition(t, r, {"a", "b"}, {"skin"});
  const S::Partition flow = S::partition(t, r, {"a", "b"}, {"hot", "cold", "remaining"});
  CHECK(heat.uncovered.empty() && heat.doubly.empty(), "heat keyed by skin: accepted");
  CHECK(flow.uncovered.empty() && flow.doubly.empty(), "flow keyed by hot, cold and remaining: accepted");
  const S::Partition both = S::partition(t, r, {"a", "b"}, {"skin", "hot", "remaining"});
  check_eq("one instance keyed by skin and hot", both.doubly.count(1) ? both.doubly.at(1) : Strs{}, Strs{"skin", "hot"});
  check_eq("one instance keyed by skin and hot: one entity twice", int(both.doubly.size()), 1);
  const auto rem = both.atoms_of_key.find("remaining");
  check_eq("one instance keyed by skin and hot: remaining", rem == both.atoms_of_key.end() ? Ints{-1} : rem->second, Ints{});
}

// Outputs: a force on an interior selection, an output off the domain, the Laplacian-form force rule decided from the
// keys that cover the boundary.
void check_force_outputs() {
  const S::Table t = split_rect();
  const S::Resolution r = S::resolve(t, split_rect_selectors());
  const S::Partition pa = S::partition(t, r, {"a"}, {"hot", "interface", "remaining"});
  const S::Placement mk = S::place(t, r, pa, {"a"}, "mk");
  check_eq("mk in a flow on a: interior", mk.interior, Ints{10});
  check_eq("mk in a flow on a: boundary", mk.boundary, Ints{});
  check_eq("mk in a flow on a: keys", mk.keys, Strs{});
  const S::Partition pab = S::partition(t, r, {"a", "b"}, {"hot", "cold", "remaining"});
  const S::Placement itf = S::place(t, r, pab, {"a", "b"}, "interface");
  check_eq("the interface in a flow on [a, b]: interior", itf.interior, Ints{9});
  // AUTHOR TEXT
  const Error f = msg::force_on_interior("drag", "force", "mk", "flow", t, mk.interior);
  check_eq("force_on_interior message", f.message(), "output 'drag': a force on 'mk' is not available: 'mk' is interior to the domain of 'flow'");
  check_eq("force_on_interior where", f.where(), "/outputs/drag");
  check_eq("force_on_interior hint", f.hint(),
           "a force or a torque is the engine's reaction on the boundary of the domain, and 'mk' lies inside 'a', which "
           "belongs to 'flow'. Through an interior section, use a flux, an integral, an average, a maximum or a minimum.");
  check_eq("force_on_interior code", f.code(), "output.force_on_interior");
  check_json("force_on_interior details", details(f),
             J(R"({"output": "drag", "kind": "force", "boundary": "mk", "physics": "flow", "adjacent": [["a", "a"]]})"));

  // outside the domain: hot for a heat instance on b
  const S::Partition pb = S::partition(t, r, {"b"}, {"cold", "remaining"});
  const S::Placement hot = S::place(t, r, pb, {"b"}, "hot");
  check_eq("hot for an instance on b: outside", hot.outside, Ints{1});
  CHECK(hot.boundary.empty() && hot.interior.empty() && hot.keys.empty(), "hot for an instance on b: nothing on or in b");
  // AUTHOR TEXT
  const Error o = msg::boundary_not_in_domain("q_hot", "hot", "heat", {"b"}, t, hot.outside);
  check_eq("not_in_domain message", o.message(), "output 'q_hot': boundary 'hot' is not in the domain of 'heat'");
  check_eq("not_in_domain where", o.where(), "/outputs/q_hot");
  check_eq("not_in_domain hint", o.hint(),
           "'hot' lies between 'a' and the exterior, and the domain of 'heat' is 'b'. An output of 'heat' is computed on a "
           "boundary or a section of its domain.");
  check_eq("not_in_domain code", o.code(), "output.boundary_not_in_domain");
  check_json("not_in_domain details", details(o),
             J(R"({"output": "q_hot", "boundary": "hot", "physics": "heat", "adjacent": [["a", "exterior"]], "domain": ["b"]})"));

  // the Laplacian-form rule on a boundary that is not a key: the keys that cover it decide (walls at rest pass)
  const S::Table ch = channel();
  const S::Resolution rc = S::resolve(ch, J(R"({"inlet":  {"of": ["channel"], "where": {"x": [0]}},
                                               "outlet": {"of": ["channel"], "where": {"x": [0.04]}},
                                               "walls":  {"of": ["channel"], "except": ["inlet", "outlet"]},
                                               "skin":   {"of": ["channel"]},
                                               "bottom": {"of": ["channel"], "where": {"y": [0]}}})"));
  const S::Partition pc = S::partition(ch, rc, {"channel"}, {"inlet", "outlet", "walls"});
  const S::Placement skin = S::place(ch, rc, pc, {"channel"}, "skin");
  check_eq("skin: on the domain boundary", skin.boundary, Ints{1, 2, 3, 4});
  check_eq("skin: covered by inlet, outlet and walls (an inlet part: refused)", skin.keys, Strs{"inlet", "outlet", "walls"});
  check_eq("bottom: covered by walls only (a wall at rest: accepted)", S::place(ch, rc, pc, {"channel"}, "bottom").keys, Strs{"walls"});
  const S::Partition pr = S::partition(ch, rc, {"channel"}, {"inlet", "outlet", "remaining"});
  check_eq("bottom: covered by remaining", S::place(ch, rc, pr, {"channel"}, "bottom").keys, Strs{"remaining"});
  // AUTHOR TEXT
  const Error l = msg::force_needs_stress_form_on_part("F_skin", "force", "skin", "inlet", "inlet", false, "navier_stokes_steady_planar");
  check_eq("Laplacian rule message", l.message(), "output 'F_skin': a force on 'skin' is not available with the Laplacian viscous form on this route");
  check_eq("Laplacian rule where", l.where(), "/outputs/F_skin");
  check_eq("Laplacian rule hint", l.hint(),
           "part of 'skin' carries the condition 'inlet' of 'inlet'. The Navier-Stokes engine returns, for the Laplacian "
           "form, the pseudo-traction mu du/dn - p n, which is the force only on walls at rest. Use \"viscous_form\": "
           "\"stress\" (the physical form), or \"equation\": \"stokes\" if the flow is creeping.");
  check_eq("Laplacian rule code", l.code(), "output.force_needs_stress_form");
  check_json("Laplacian rule details", details(l),
             J(R"({"boundary": "skin", "key": "inlet", "condition": "inlet", "route": "navier_stokes_steady_planar",
                   "accepted_on": "wall boundaries at rest"})"));
  const Error m = msg::force_needs_stress_form_on_part("T_top", "torque", "top_all", "lid", "wall", true, "navier_stokes_steady_planar");
  check_eq("Laplacian rule, moving wall: message", m.message(),
           "output 'T_top': a torque on 'top_all' is not available with the Laplacian viscous form on this route");
  check_eq("Laplacian rule, moving wall: hint", m.hint(),
           "part of 'top_all' is the moving wall 'lid'. The Navier-Stokes engine returns, for the Laplacian form, the "
           "pseudo-traction mu du/dn - p n, which is the force only on walls at rest. Use \"viscous_form\": \"stress\" (the "
           "physical form), or \"equation\": \"stokes\" if the flow is creeping.");
  check_json("Laplacian rule, moving wall: condition", details(m)["condition"], json("wall with velocity_m_s"));
}

// to_json and table_from_json: the keys, and every double bit for bit (through the JSON text too).
void check_table_json_round_trip() {
  S::Table t = pipe(true);
  S::Entity odd = arc(42, "p1", "exterior", 0.1 + 0.2, -0.0, 1e-300, 4.9406564584124654e-324, 0.006283185307179587,
                      1.0 / 3.0, 0.7071067811865476, -0.7071067811865475, 2.220446049250313e-16);
  odd.lo[2] = -1.7976931348623157e308, odd.hi[2] = 123456789.123456789, odd.point[2] = 5e-324;
  odd.area_m2 = 3.141592653589793e-12;
  odd.type = "ellipse";
  odd.dim = 1;
  t.entities.push_back(odd);
  t.diagonal_m = 0.1 + 0.7;
  const json j = S::to_json(t);
  std::set<std::string> top, ent;
  for (auto it = j.begin(); it != j.end(); ++it) top.insert(it.key());
  CHECK(top == (std::set<std::string>{"geometry", "regions", "entities", "diagonal_m"}), "the table's keys: " + j.dump().substr(0, 200));
  if (j.contains("entities") && j["entities"].is_array() && !j["entities"].empty())
    for (auto it = j["entities"][0].begin(); it != j["entities"][0].end(); ++it) ent.insert(it.key());
  CHECK(ent == (std::set<std::string>{"tag", "dim", "side", "curves", "axis", "type", "lo", "hi", "point", "tangent", "length_m", "area_m2"}),
        "an entity's keys");
  json jx = json::parse(j.dump());
  for (json& e : jx["entities"]) e["atom"] = 2001;  // the info file adds each entity's atom
  for (const json& src : {j, jx}) {
    const S::Table u = S::table_from_json(src);
    check_eq("geometry", u.geometry, t.geometry);
    check_eq("regions", u.regions, t.regions);
    CHECK(std::memcmp(&u.diagonal_m, &t.diagonal_m, sizeof(double)) == 0, "diagonal_m bit for bit");
    check_eq("entities", int(u.entities.size()), int(t.entities.size()));
    for (size_t k = 0; k < std::min(u.entities.size(), t.entities.size()); ++k) {
      const S::Entity &a = t.entities[k], &b = u.entities[k];
      const std::string at = "entity " + std::to_string(a.tag);
      check_eq(at + " tag", b.tag, a.tag);
      check_eq(at + " dim", b.dim, a.dim);
      check_eq(at + " side", b.side[0] + "|" + b.side[1], a.side[0] + "|" + a.side[1]);
      check_eq(at + " curves", b.curves, a.curves);
      CHECK(b.axis == a.axis, at + " axis");
      check_eq(at + " type", b.type, a.type);
      CHECK(std::memcmp(a.lo, b.lo, sizeof a.lo) == 0 && std::memcmp(a.hi, b.hi, sizeof a.hi) == 0, at + " box bit for bit");
      CHECK(std::memcmp(a.point, b.point, sizeof a.point) == 0, at + " point bit for bit");
      CHECK(std::memcmp(a.tangent, b.tangent, sizeof a.tangent) == 0, at + " tangent bit for bit");
      CHECK(std::memcmp(&a.length_m, &b.length_m, sizeof(double)) == 0 && std::memcmp(&a.area_m2, &b.area_m2, sizeof(double)) == 0,
            at + " measures bit for bit");
    }
  }
}

// IMPLEMENTER TEXT.  An entity that carries two markers written in opposite directions.  The table has one tangent per
// entity, the written direction of the first of its markers, and orientation is keyed by (name, tag): a marker written
// against the entity's tangent is listed in Entity::curves_reversed, and its sign is the opposite of a marker written
// along it.  The normal of a marker depends only on the direction in which it is written,
// never on which marker gave the entity its tangent.  Hand values: a marker written bottom to top (+e_y) has the normal
// -e_x, one written top to bottom (-e_y) the normal +e_x (the tangent rotated by +90 degrees).
void check_two_markers_written_in_opposite_directions() {
  // entity 10 of the split rectangle carries mk, written bottom to top, and down, written top to bottom; once with the
  // entity's tangent along mk (+e_y), once along down (-e_y)
  for (bool tangent_down : {false, true}) {
    S::Table t = split_rect(tangent_down);
    S::Entity& e10 = t.entities[9];
    CHECK(e10.tag == 10, "entity 10 is the marker");
    e10.curves = tangent_down ? Strs{"down", "mk"} : Strs{"mk", "down"};  // the first marker gives the tangent
    e10.curves_reversed = {tangent_down ? "mk" : "down"};
    const std::string what = tangent_down ? "tangent -e_y" : "tangent +e_y";
    const json sels = J(R"({"mk": {"curve": "mk"}, "down": {"curve": "down"},
                            "mk_first": {"union": [{"curve": "mk"}, {"curve": "down"}]},
                            "down_first": {"union": [{"ref": "down"}, {"ref": "mk"}]}})");
    const std::vector<S::Table> tables = {t, S::table_from_json(json::parse(S::to_json(t).dump()))};
    for (size_t k = 0; k < tables.size(); ++k) {
      const S::Table& u = tables[k];
      const std::string w = what + (k == 0 ? "" : ", after the JSON round trip");
      const S::Resolution r = S::resolve(u, sels);
      for (const char* k : {"mk", "down", "mk_first", "down_first"}) check_eq(w + ": " + k, sel(r, k), Ints{10});
      check_eq(w + ": mk", orient(r, "mk", 10), tangent_down ? +1 : -1);
      check_eq(w + ": down", orient(r, "down", 10), tangent_down ? -1 : +1);
      const auto nm = normal(u, r, "mk", 10), nd = normal(u, r, "down", 10);
      CHECK(nm.first == -1.0 && nm.second == 0.0, w + ": mk written bottom to top has the normal -e_x");
      CHECK(nd.first == 1.0 && nd.second == 0.0, w + ": down written top to bottom has the normal +e_x");
      const auto n1 = normal(u, r, "mk_first", 10), n2 = normal(u, r, "down_first", 10);
      CHECK(n1.first == -1.0 && n1.second == 0.0, w + ": a union orients by its first operand (mk)");
      CHECK(n2.first == 1.0 && n2.second == 0.0, w + ": a union orients by its first operand (down)");
      check_atoms(w, r,
                  {{2001, "named", {"down", "down_first", "mk", "mk_first"}, "a", "a", {10}},
                   {2002, "unnamed", {}, "a", "b", {9}},
                   {2003, "unnamed", {}, "a", "exterior", {1, 2, 3, 7, 8}},
                   {2004, "unnamed", {}, "b", "exterior", {4, 5, 6}}});
    }
    // the JSON form: the key only on the entity that has a reversed marker
    const json j = S::to_json(t);
    for (const json& e : j["entities"]) {
      const bool has = e.contains("curves_reversed");
      CHECK(has == (e["tag"] == 10), what + ": curves_reversed only on entity 10, tag " + e["tag"].dump());
      if (has) check_json(what + ": curves_reversed", e["curves_reversed"], json::array({tangent_down ? "mk" : "down"}));
    }
  }
}

// IMPLEMENTER TEXT.  The partial matches and the candidates of a where filter on an of or between nested in a union or a
// difference (model_selection.hpp, Resolution): every filtered selector that adds entities to the boundary reports its
// partial matches, unless the boundary selects them anyway or a later operand of a difference removes them; the
// candidates and the where are those of the one filtered selector of an empty boundary whose filter keeps nothing.  On
// the channel: the bottom 2 (x in [0, 0.04], 0.04 m, point (0.02, 0)) lies partly in x in [0, 0.03]; the top 4 (point
// (0.02, 0.01)) partly in y = 0.01, x in [0.01, 0.05].  The candidates are grouped by line within the tolerance.
void check_partial_matches_of_nested_filters() {
  const S::Table t = channel();
  const S::Resolution r = S::resolve(t, J(R"({
      "inlet":         {"of": ["channel"], "where": {"x": [0]}},
      "nested":        {"union": [{"of": ["channel"], "where": {"y": [0], "x": [0, 0.03]}}, {"ref": "inlet"}]},
      "nested_empty":  {"union": [{"of": ["channel"], "where": {"y": [0], "x": [0, 0.03]}}]},
      "in_difference": {"difference": [{"of": ["channel"], "where": {"x": [0, 0.03]}}, {"ref": "inlet"}]},
      "removed":       {"difference": [{"of": ["channel"], "where": {"y": [0], "x": [0, 0.03]}}, {"of": ["channel"], "where": {"y": [0]}}]},
      "subtracted":    {"difference": [{"of": ["channel"], "where": {"x": [0]}}, {"of": ["channel"], "where": {"y": [0], "x": [0, 0.03]}}]},
      "covered":       {"union": [{"of": ["channel"], "where": {"y": [0], "x": [0, 0.03]}}, {"of": ["channel"], "where": {"y": [0]}}]},
      "two":           {"union": [{"of": ["channel"], "where": {"y": [0], "x": [0, 0.03]}},
                                  {"of": ["channel"], "where": {"y": [0.01], "x": [0.01, 0.05]}}]},
      "via_ref":       {"union": [{"ref": "two"}, {"ref": "inlet"}]}})"));
  const json lines = J(R"([{"where": {"x": 0}, "measure_m": 0.01}, {"where": {"x": 0.04}, "measure_m": 0.01},
                           {"where": {"y": 0}, "measure_m": 0.04}, {"where": {"y": 0.01}, "measure_m": 0.04}])");
  auto partial_of = [&r](const std::string& k) { return r.partial.count(k) ? r.partial.at(k) : json(nullptr); };
  // a union operand: the bottom is left out of a non-empty boundary
  check_eq("nested", sel(r, "nested"), Ints{1});
  check_json("nested: partial", partial_of("nested"), J(R"({"tags": [2]})"));
  // the only operand of a union: the empty diagnostic of that filter
  check_eq("nested_empty", sel(r, "nested_empty"), Ints{});
  check_json("nested_empty: the empty diagnostic", empty_of(r, "nested_empty"),
             json{{"where", J(R"({"x": [0, 0.03], "y": [0]})")}, {"candidates", lines}, {"partial", J("[2]")}, {"axis_only", false}});
  // the first operand of a difference that removes everything its filter kept: partial matches, no candidates
  check_eq("in_difference", sel(r, "in_difference"), Ints{});
  check_json("in_difference: the empty diagnostic", empty_of(r, "in_difference"),
             J(R"({"where": null, "candidates": [], "partial": [2, 4], "axis_only": false})"));
  // a partial match that a later operand of the difference removes is not reported
  check_json("removed: the empty diagnostic", empty_of(r, "removed"),
             json{{"where", J(R"({"x": [0, 0.03], "y": [0]})")}, {"candidates", lines}, {"partial", json::array()}, {"axis_only", false}});
  // a filter on a subtracted operand adds nothing, so it reports nothing (its partial match, the bottom, is not selected)
  check_eq("subtracted", sel(r, "subtracted"), Ints{1});
  CHECK(!r.partial.count("subtracted"), "subtracted: no partial diagnostic");
  // a partial match that another operand selects is not left out
  check_eq("covered", sel(r, "covered"), Ints{2});
  CHECK(!r.partial.count("covered"), "covered: no partial diagnostic");
  // two filters: their partial matches, and neither where nor candidates
  check_json("two: the empty diagnostic", empty_of(r, "two"), J(R"({"where": null, "candidates": [], "partial": [2, 4], "axis_only": false})"));
  // through ref the boundary named reports its own
  check_eq("via_ref", sel(r, "via_ref"), Ints{1});
  CHECK(!r.partial.count("via_ref"), "via_ref: no partial diagnostic");

  // the texts and pointers of the builders, the document writing a name as a string operand
  const std::string tail = "a where filter keeps an edge only when the whole edge satisfies it. To select part of an edge, add a curve "
                           "along that part and use {\"curve\": NAME}.";
  const Error p = msg::partial_match("nested", J(R"({"union": [{"of": "channel", "where": {"y": 0, "x": [0, 0.03]}}, "inlet"]})"), t, {2});
  check_eq("nested: partial_match where", p.where(), "/boundaries/nested/union/0/where");
  check_eq("nested: partial_match hint", p.hint(), "1 edge, 0.04 m, first at (x, y) = (0.02, 0): " + tail);
  const Error e = msg::empty_selection("nested_empty", J(R"({"union": [{"of": "channel", "where": {"y": 0, "x": [0, 0.03]}}]})"), t,
                                       empty_of(r, "nested_empty"));
  check_eq("nested_empty: where", e.where(), "/boundaries/nested_empty/union/0/where");
  check_eq("nested_empty: hint", e.hint(),
           "1 edge of 'channel' lies partly within x in [0, 0.03] and y = 0 (0.04 m, first at (x, y) = (0.02, 0)); " + tail);
  const Error d = msg::empty_selection("in_difference", J(R"({"difference": [{"of": "channel", "where": {"x": [0, 0.03]}}, "inlet"]})"), t,
                                       empty_of(r, "in_difference"));
  check_eq("in_difference: where", d.where(), "/boundaries/in_difference/difference/0/where");
  check_eq("in_difference: hint", d.hint(), "2 edges of 'channel' lie partly within x in [0, 0.03] (0.08 m, first at (x, y) = (0.02, 0)); " + tail);
  const Error w = msg::empty_selection("two", J(R"({"union": [{"of": "channel", "where": {"y": 0, "x": [0, 0.03]}},
                                                              {"of": "channel", "where": {"y": 0.01, "x": [0.01, 0.05]}}]})"),
                                       t, empty_of(r, "two"));
  check_eq("two: where", w.where(), "/boundaries/two");
  check_eq("two: hint", w.hint(), "2 edges lie partly within a where filter of 'two' (0.08 m, first at (x, y) = (0.02, 0)); " + tail);

  // candidates: two collinear pieces of the bottom whose y differs by 1e-19 (far below the tolerance, 4.1e-11) are one line
  S::Table u = channel();
  u.entities[1] = seg(21, "channel", "exterior", 0, 0, 0.02, 0, 0.02);
  u.entities.push_back(seg(22, "channel", "exterior", 0.02, 1e-19, 0.04, 1e-19, 0.02));
  const S::Resolution ru = S::resolve(u, J(R"({"far": {"of": ["channel"], "where": {"x": [0.5]}}})"));
  check_json("candidates within the tolerance", empty_of(ru, "far")["candidates"], lines);
}

// AUTHOR TEXT (the rule).  An edge that a where filter does not keep is a partial match when the filter keeps part of it:
//   * a straight edge (type "line"), exactly: the segment point -/+ tangent * length / 2 has a part longer than the
//     tolerance inside the filter's box (the box of the range constraints; an equality constraint that the edge's box
//     does not satisfy makes no partial match, as before: an equality is never partly satisfied);
//   * a curved edge (a circle or an ellipse arc), by its box: the box overlaps every unsatisfied range by more than the
//     tolerance, unless the filter's box lies strictly inside the edge's box in every coordinate of the geometry (each
//     coordinate constrained by a range, lo - tol > the box's lo and hi + tol < the box's hi): the box cannot tell
//     whether the curve passes through such a filter, and a closed curve runs around it (the outer circle of the annulus
//     around a box drawn on the inner circle), so it is not reported.
// The annulus selects its inner circle by the circle's box, the natural way to write it: no partial match.  A filter
// that sticks out of the outer circle's box on one side still reports it, and a coordinate that the filter leaves free
// never encloses.  The lines of a triangle: the hypotenuse is reported where it crosses the filter's box (also a box
// inside its own box) and not where it misses it (also a box that overlaps its box).
void check_partial_match_of_an_enclosed_filter() {
  const S::Table an = annulus();
  const S::Resolution r = S::resolve(an, J(R"({
      "inner":      {"between": [["ring"], "exterior"], "where": {"x": [-0.002, 0.002], "y": [-0.002, 0.002]}},
      "west_inner": {"between": [["ring"], "exterior"], "where": {"x": [-0.002, 0.0055], "y": [-0.002, 0.002]}},
      "band":       {"between": [["ring"], "exterior"], "where": {"y": [-0.001, 0.001]}}})"));
  check_eq("annulus inner", sel(r, "inner"), Ints{2});
  CHECK(!r.partial.count("inner"), "annulus inner: the outer circle's box encloses the filter's box: no partial match");
  check_eq("west_inner", sel(r, "west_inner"), Ints{2});
  CHECK(r.partial.count("west_inner"), "west_inner: the filter sticks out of the outer circle's box in x: a partial match");
  if (r.partial.count("west_inner")) check_json("west_inner: partial", r.partial.at("west_inner"), J(R"({"tags": [1]})"));
  check_eq("band", sel(r, "band"), Ints{});
  check_json("band: x is free, so no box encloses the filter: both circles in part", empty_of(r, "band")["partial"], J("[1, 2]"));

  // a triangle: the bottom 1, the right edge 2 and the hypotenuse 3 from (0.04, 0.01) back to (0, 0), y = x / 4
  S::Table t;
  t.geometry = "planar";
  t.regions = {"tri"};
  t.diagonal_m = 0.04123105625617661;
  t.entities = {seg(1, "tri", "exterior", 0, 0, 0.04, 0, 0.04), seg(2, "tri", "exterior", 0.04, 0, 0.04, 0.01, 0.01)};
  S::Entity h;
  h.tag = 3;
  h.dim = 1;
  h.side[0] = "tri";
  h.side[1] = "exterior";
  h.type = "line";
  h.lo[0] = 0, h.lo[1] = 0, h.hi[0] = 0.04, h.hi[1] = 0.01;
  h.point[0] = 0.02, h.point[1] = 0.005;
  h.tangent[0] = -0.9701425001453319, h.tangent[1] = -0.24253562503633297;
  h.length_m = 0.04123105625617661;
  t.entities.push_back(h);
  // the hypotenuse is y = x / 4
  const S::Resolution rt = S::resolve(t, J(R"({
      "bottom_part": {"of": ["tri"], "where": {"x": [0.01, 0.02], "y": [0, 0.002]}},
      "inside":      {"of": ["tri"], "where": {"x": [0.01, 0.02], "y": [0.001, 0.002]}},
      "crossing":    {"of": ["tri"], "where": {"x": [0.015, 0.025], "y": [0.003, 0.006]}},
      "corner":      {"of": ["tri"], "where": {"x": [0.03, 0.05], "y": [0, 0.004]}},
      "x_eq":        {"of": ["tri"], "where": {"x": [0.02]}},
      "cooled":      {"union": [{"of": ["tri"], "where": {"y": [0]}},
                                {"of": ["tri"], "where": {"x": [0.015, 0.025], "y": [0.003, 0.006]}}]}})"));
  // x in [0.01, 0.02]: the hypotenuse has y in [0.0025, 0.005], above y = 0.002: it misses both boxes; the bottom (y = 0)
  // runs past x = 0.02 inside y in [0, 0.002]
  check_json("bottom_part: the bottom in part; the hypotenuse misses the box", empty_of(rt, "bottom_part")["partial"], J("[1]"));
  check_json("inside: the hypotenuse misses the box inside its own box", empty_of(rt, "inside")["partial"], json::array());
  // x in [0.015, 0.025]: the hypotenuse has y in [0.00375, 0.00625], through y in [0.003, 0.006]: a partial match, also
  // though the filter's box lies inside its box
  check_json("crossing: the hypotenuse crosses the box inside its own box", empty_of(rt, "crossing")["partial"], J("[3]"));
  // x in [0.03, 0.04]: the hypotenuse has y in [0.0075, 0.01], above the box, which overlaps its box; the bottom and the
  // right edge (x = 0.04, y in [0, 0.01]) run into the box
  check_json("corner: the bottom and the right edge in part; the hypotenuse misses the box", empty_of(rt, "corner")["partial"],
             J("[1, 2]"));
  // an equality is never partly satisfied: the hypotenuse crosses x = 0.02 at one point
  check_json("x_eq: no partial match of an equality", empty_of(rt, "x_eq")["partial"], json::array());
  // a boundary that selects the bottom and means the part of the hypotenuse in the box as well: the warning's case
  check_eq("cooled", sel(rt, "cooled"), Ints{1});
  CHECK(rt.partial.count("cooled"), "cooled: the hypotenuse is a partial match of the second operand");
  if (rt.partial.count("cooled")) check_json("cooled: partial", rt.partial.at("cooled"), J(R"({"tags": [3]})"));
}

// A straight edge given by its end points: box, midpoint, unit tangent (from (x0, y0) to (x1, y1)) and length.
S::Entity line(int tag, const std::string& s0, const std::string& s1, double x0, double y0, double x1, double y1, bool axis = false) {
  S::Entity e;
  e.tag = tag;
  e.dim = 1;
  e.side[0] = s0;
  e.side[1] = s1;
  e.axis = axis;
  e.type = "line";
  e.lo[0] = std::min(x0, x1), e.hi[0] = std::max(x0, x1);
  e.lo[1] = std::min(y0, y1), e.hi[1] = std::max(y0, y1);
  e.point[0] = 0.5 * (x0 + x1), e.point[1] = 0.5 * (y0 + y1);
  e.length_m = std::hypot(x1 - x0, y1 - y0);
  e.tangent[0] = (x1 - x0) / e.length_m, e.tangent[1] = (y1 - y0) / e.length_m;
  return e;
}
// A polygon region "box", counter-clockwise, its edges tagged 1, 2, ... in order; the diagonal of its bounding box.
S::Table polygon_table(const std::vector<std::pair<double, double>>& v, const std::string& geometry = "planar") {
  S::Table t;
  t.geometry = geometry;
  t.regions = {"box"};
  double lx = 1e300, ly = 1e300, hx = -1e300, hy = -1e300;
  for (size_t i = 0; i < v.size(); ++i) {
    const auto [x0, y0] = v[i];
    const auto [x1, y1] = v[(i + 1) % v.size()];
    t.entities.push_back(line(static_cast<int>(i + 1), "box", "exterior", x0, y0, x1, y1,
                              geometry == "axisymmetric" && x0 == 0.0 && x1 == 0.0));
    lx = std::min(lx, x0), ly = std::min(ly, y0), hx = std::max(hx, x0), hy = std::max(hy, y0);
  }
  t.diagonal_m = std::hypot(hx - lx, hy - ly);
  return t;
}

// AUTHOR TEXT (the rule at the filter's bounds).  For a straight edge, a constraint that the edge's box satisfies does
// not cut its segment; every other constraint cuts it at the filter's own bounds (not widened by the tolerance); the
// edge is a partial match when the part left is longer than the tolerance.  So an edge that only touches the filter's
// box (it ends on one of its faces, or meets one of its corners) is not a partial match, in whatever document it lies,
// and an edge that enters the box by less than the tolerance (tolerance_m) is not one either, while one that enters by
// clearly more is.  Each case is a union of an edge kept for sure and the filter, so that the selection is not empty.
//   rectangle 0.04 x 0.01: 1 bottom, 2 right (x = 0.04), 3 top, 4 left;
//   trapezoid: 1 bottom, 2 right (x = 0.04, y in [0, 0.01]), 3 the slanted top from (0.04, 0.01) to (0, 0.02)
//     (y = 0.02 - x / 4), 4 left;
//   "diamond": 1 bottom (0, 0)-(0.02, 0), 2 the 45-degree edge to (0.04, 0.02), 3 top, 4 left;
//   the trapezoid in (r, z), its left edge on the axis.
void check_line_partial_matches_at_the_filter_bounds() {
  const S::Table rect = polygon_table({{0, 0}, {0.04, 0}, {0.04, 0.01}, {0, 0.01}});
  const S::Table trap = polygon_table({{0, 0}, {0.04, 0}, {0.04, 0.01}, {0, 0.02}});
  const S::Table diam = polygon_table({{0, 0}, {0.02, 0}, {0.04, 0.02}, {0, 0.02}});
  const S::Table axi = polygon_table({{0, 0}, {0.04, 0}, {0.04, 0.01}, {0, 0.02}}, "axisymmetric");
  const auto partial_of = [](const S::Resolution& r, const std::string& k) { return r.partial.count(k) ? r.partial.at(k) : json(nullptr); };
  const json touching = J(R"({
      "end_on_face":  {"union": [{"of": ["box"], "where": {"x": [0]}}, {"of": ["box"], "where": {"x": [0.04, 0.05]}}]},
      "around_right": {"union": [{"of": ["box"], "where": {"x": [0]}}, {"of": ["box"], "where": {"x": [0.04, 1], "y": [0, 0.01]}}]}})");
  // the bottom and the top of the rectangle end on the face x = 0.04
  const S::Resolution rr = S::resolve(rect, touching);
  check_eq("rectangle end_on_face", sel(rr, "end_on_face"), Ints{2, 4});
  check_json("rectangle end_on_face: no partial match", partial_of(rr, "end_on_face"), nullptr);
  check_json("rectangle around_right: no partial match", partial_of(rr, "around_right"), nullptr);
  // the same bottom edge in the trapezoid, and its slanted top, end on the same face: the same answer
  const S::Resolution rt = S::resolve(trap, touching);
  check_eq("trapezoid end_on_face", sel(rt, "end_on_face"), Ints{2, 4});
  check_json("trapezoid end_on_face: no partial match (the slant and the bottom end on the face)", partial_of(rt, "end_on_face"), nullptr);
  check_json("trapezoid around_right: no partial match (a box drawn exactly around the right edge)",
             partial_of(rt, "around_right"), nullptr);
  // a corner of the box on the slant: (0.02, 0.015) is the slant's midpoint
  const S::Resolution rc = S::resolve(trap, J(R"({"corner": {"union": [{"of": ["box"], "where": {"y": [0]}},
                                                                  {"of": ["box"], "where": {"x": [0.02, 0.03], "y": [0.015, 0.02]}}]}})"));
  check_eq("trapezoid corner", sel(rc, "corner"), Ints{1});
  check_json("trapezoid corner: the box's corner on the slant is no partial match", partial_of(rc, "corner"), nullptr);
  // the 45-degree edge and the top end at (0.04, 0.02) on the face x = 0.04
  const S::Resolution rd = S::resolve(diam, J(R"({"end_on_face": {"union": [{"of": ["box"], "where": {"x": [0]}},
                                                                       {"of": ["box"], "where": {"x": [0.04, 0.05]}}]}})"));
  check_eq("diamond end_on_face", sel(rd, "end_on_face"), Ints{4});
  check_json("diamond end_on_face: no partial match", partial_of(rd, "end_on_face"), nullptr);
  // axisymmetric: the slant ends on r = 0.04
  const S::Resolution ra = S::resolve(axi, J(R"({"end_on_face": {"union": [{"of": ["box"], "where": {"z": [0]}},
                                                                      {"of": ["box"], "where": {"r": [0.04, 1]}}]}})"));
  check_eq("axisymmetric end_on_face", sel(ra, "end_on_face"), Ints{1, 2});
  check_json("axisymmetric end_on_face: no partial match", partial_of(ra, "end_on_face"), nullptr);

  // tolerance_m 1e-6: the bottom and the top enter the box x >= 0.04 - p by p
  const S::Resolution rtol = S::resolve(rect, J(R"({
      "in_by_half_um": {"of": ["box"], "where": {"x": [0.0399995, 0.05]}, "tolerance_m": 1e-6},
      "in_by_2_um":    {"of": ["box"], "where": {"x": [0.039998, 0.05]}, "tolerance_m": 1e-6}})"));
  check_eq("in_by_half_um", sel(rtol, "in_by_half_um"), Ints{2});
  check_json("in_by_half_um: an entry shorter than tolerance_m is no partial match", partial_of(rtol, "in_by_half_um"), nullptr);
  check_eq("in_by_2_um", sel(rtol, "in_by_2_um"), Ints{2});
  check_json("in_by_2_um: an entry longer than tolerance_m is a partial match", partial_of(rtol, "in_by_2_um"), J(R"({"tags": [1, 3]})"));

  // an edge within the tolerance of a satisfied bound is cut by the other constraints only: the right edge (x = 0.04) is
  // within tolerance_m of x >= 0.0400005, so the filter's y range alone decides, and it keeps y in [0, 0.005] of it
  const S::Resolution rbound = S::resolve(rect, J(R"({
      "edge_within_tol": {"of": ["box"], "where": {"x": [0.0400005, 0.05], "y": [0, 0.005]}, "tolerance_m": 1e-6}})"));
  check_json("edge_within_tol: the right edge is in part inside the box", empty_of(rbound, "edge_within_tol")["partial"], J("[2]"));

  // the whole segment counts: a box at the far end of the triangle's hypotenuse (y = x / 4, x in [0.035, 0.04] there,
  // y in [0.00875, 0.01]) and of its right edge (x = 0.04, y in [0.008, 0.01] inside the box)
  S::Table tri;
  tri.geometry = "planar";
  tri.regions = {"tri"};
  tri.diagonal_m = std::hypot(0.04, 0.01);
  tri.entities = {line(1, "tri", "exterior", 0, 0, 0.04, 0), line(2, "tri", "exterior", 0.04, 0, 0.04, 0.01),
                  line(3, "tri", "exterior", 0.04, 0.01, 0, 0)};
  const S::Resolution re = S::resolve(tri, J(R"({"end_box": {"of": ["tri"], "where": {"x": [0.035, 0.05], "y": [0.008, 0.02]}}})"));
  check_json("end_box: the right edge and the hypotenuse enter near their ends", empty_of(re, "end_box")["partial"], J("[2, 3]"));
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
  run("check_of_selects_one_sided_entities", check_of_selects_one_sided_entities);
  run("check_between", check_between);
  run("check_curve_marker", check_curve_marker);
  run("check_where_uses_the_whole_box", check_where_uses_the_whole_box);
  run("check_set_operations", check_set_operations);
  run("check_empty_selection_candidates", check_empty_selection_candidates);
  run("check_atoms_are_signatures", check_atoms_are_signatures);
  run("check_master_numbering", check_master_numbering);
  run("check_numbering_ignores_tags_and_entity_order", check_numbering_ignores_tags_and_entity_order);
  run("check_partition_refusals", check_partition_refusals);
  run("check_key_partly_outside_is_taken_where_it_lies", check_key_partly_outside_is_taken_where_it_lies);
  run("check_remaining", check_remaining);
  run("check_overlap_across_instances", check_overlap_across_instances);
  run("check_force_outputs", check_force_outputs);
  run("check_table_json_round_trip", check_table_json_round_trip);
  run("check_two_markers_written_in_opposite_directions", check_two_markers_written_in_opposite_directions);
  run("check_partial_matches_of_nested_filters", check_partial_matches_of_nested_filters);
  run("check_partial_match_of_an_enclosed_filter", check_partial_match_of_an_enclosed_filter);
  run("check_line_partial_matches_at_the_filter_bounds", check_line_partial_matches_at_the_filter_bounds);
  std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
