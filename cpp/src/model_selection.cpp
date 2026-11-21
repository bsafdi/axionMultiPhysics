// The named boundaries of a model document resolved on an entity table (model_selection.hpp).
//
// The standard library and nlohmann/json only, and no namespace-scope object with a constructor: axmp-mesher compiles
// this file, and every electromagnetic mesh run starts that program.  A selection is held as a map from the index of an
// entity in Table::entities to its orientation; tags appear only in what the functions return.
#include "axionmultiphysics/model_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace axmp::model::selection {

namespace {

using json = nlohmann::json;
using Strs = std::vector<std::string>;
using Sel = std::map<std::size_t, int>;  // entity index -> orientation (+1 / -1, model_selection.hpp CONVENTIONS)

bool is_exterior(const std::string& s) { return s == "exterior"; }

bool contains(const Strs& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

// The entities that take part in a selection, an atom or a partition (model_selection.hpp).
bool is_boundary_entity(const Entity& e) {
  return e.side[0] != e.side[1] || is_exterior(e.side[0]) || is_exterior(e.side[1]) || !e.curves.empty() || e.axis;
}

// The coordinates of a `where` filter, in component order.
Strs coordinate_names(const std::string& geometry) {
  if (geometry == "axisymmetric") return {"r", "z"};
  if (geometry == "3d") return {"x", "y", "z"};
  return {"x", "y"};
}
int component(const std::string& geometry, const std::string& name) {
  const Strs names = coordinate_names(geometry);
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end())
    throw std::logic_error("selection: '" + name + "' is not a coordinate of a " + geometry + " table (resolve takes grammar-clean selectors)");
  return int(it - names.begin());
}

// A region list of the input form; a single name is accepted as a list of one.
Strs domain_list(const json& d) {
  Strs out;
  if (d.is_string()) {
    out.push_back(d.get<std::string>());
  } else if (d.is_array()) {
    for (const json& r : d) out.push_back(r.get<std::string>());
  } else {
    throw std::logic_error("selection: a domain is a region name or a list of them (resolve takes grammar-clean selectors)");
  }
  return out;
}

// The order of region pairs: the regions in Table::regions order, then "exterior", then any other name.
std::size_t region_rank(const Table& t, const std::string& s) {
  const auto it = std::find(t.regions.begin(), t.regions.end(), s);
  if (it != t.regions.end()) return std::size_t(it - t.regions.begin());
  return is_exterior(s) ? t.regions.size() : t.regions.size() + 1;
}
using PairKey = std::tuple<std::size_t, std::string, std::size_t, std::string>;
PairKey pair_key(const Table& t, const Entity& e) {
  PairKey a{region_rank(t, e.side[0]), e.side[0], region_rank(t, e.side[1]), e.side[1]};
  if (std::tie(std::get<2>(a), std::get<3>(a)) < std::tie(std::get<0>(a), std::get<1>(a)))
    a = PairKey{std::get<2>(a), std::get<3>(a), std::get<0>(a), std::get<1>(a)};
  return a;
}

// One constraint of a `where` filter; an equality has lo == hi.
struct Constraint {
  int c = 0;
  double lo = 0.0, hi = 0.0;
  bool equality = false;
};
std::vector<Constraint> constraints_of(const Table& t, const json& where) {
  std::vector<Constraint> out;
  for (auto it = where.begin(); it != where.end(); ++it) {
    Constraint k;
    k.c = component(t.geometry, it.key());
    const json& v = it.value();
    if (v.is_number()) {
      k.lo = k.hi = v.get<double>();
      k.equality = true;
    } else if (v.is_array() && v.size() == 1) {
      k.lo = k.hi = v[0].get<double>();
      k.equality = true;
    } else if (v.is_array() && v.size() == 2) {
      k.lo = v[0].get<double>();
      k.hi = v[1].get<double>();
    } else {
      throw std::logic_error("selection: a where value is [v] or [lo, hi] (resolve takes grammar-clean selectors)");
    }
    out.push_back(k);
  }
  return out;
}
// The whole box satisfies the constraint within tol: v - tol <= lo[c] and hi[c] <= v + tol for an equality (lo == hi
// == v), lo - tol <= lo[c] and hi[c] <= hi + tol for a range.
bool satisfies(const Entity& e, const Constraint& k, double tol) { return k.lo - tol <= e.lo[k.c] && e.hi[k.c] <= k.hi + tol; }
// A range that the box does not satisfy, and overlaps by more than tol; an equality is never partly satisfied.
bool overlaps(const Entity& e, const Constraint& k, double tol) {
  return !k.equality && std::min(e.hi[k.c], k.hi) - std::max(e.lo[k.c], k.lo) > tol;
}
bool kept(const Entity& e, const std::vector<Constraint>& ks, double tol) {
  for (const Constraint& k : ks)
    if (!satisfies(e, k, tol)) return false;
  return true;
}
// The filter's box lies strictly inside the entity's box in every coordinate of the geometry (each one constrained by a
// range): for a curved entity the box cannot tell whether it passes through the filter's box, since a closed curve runs
// around it.
bool encloses(const Entity& e, const std::vector<Constraint>& ks, double tol, int dim) {
  for (int c = 0; c < dim; ++c) {
    const Constraint* k = nullptr;
    for (const Constraint& x : ks)
      if (x.c == c) k = &x;
    if (!k || k->equality || !(k->lo - tol > e.lo[c] && k->hi + tol < e.hi[c])) return false;
  }
  return true;
}
// A straight entity, decided exactly: its segment (the midpoint -/+ tangent * length / 2) is cut at the bounds of every
// constraint that the entity's box does not already satisfy, and the part that remains is longer than tol.  A constraint
// that the box satisfies (an edge that lies in the filter's range, or within tol of it) does not cut the segment, so an
// edge that only touches the filter's box is not in part inside it.
bool segment_enters(const Entity& e, const std::vector<Constraint>& ks, double tol, int dim) {
  double norm = 0.0;
  for (int c = 0; c < dim; ++c) norm += e.tangent[c] * e.tangent[c];
  norm = std::sqrt(norm);
  if (!(norm > 0.0) || !(e.length_m > 0.0)) return false;
  double t0 = 0.0, t1 = 1.0;
  for (const Constraint& k : ks) {
    if (satisfies(e, k, tol)) continue;
    const double half = 0.5 * e.length_m * e.tangent[k.c] / norm;
    const double a = e.point[k.c] - half, d = 2.0 * half;
    if (d == 0.0) {
      if (a < k.lo || a > k.hi) return false;
      continue;
    }
    double u0 = (k.lo - a) / d, u1 = (k.hi - a) / d;
    if (u0 > u1) std::swap(u0, u1);
    t0 = std::max(t0, u0);
    t1 = std::min(t1, u1);
    if (t1 <= t0) return false;
  }
  return (t1 - t0) * e.length_m > tol;
}
// Not kept, and the filter keeps part of it.  A straight entity: its segment enters the filter's box (segment_enters),
// when no equality is left unsatisfied (an equality is never partly satisfied).  Any other entity: by its box, every
// constraint is either satisfied or overlapped by more than tol, unless the entity's box encloses the filter's box
// (encloses).
bool partly(const Entity& e, const std::vector<Constraint>& ks, double tol, int dim) {
  if (e.type == "line") {
    bool some = false;
    for (const Constraint& k : ks) {
      if (satisfies(e, k, tol)) continue;
      if (k.equality) return false;
      some = true;
    }
    return some && segment_enters(e, ks, tol, dim);
  }
  if (encloses(e, ks, tol, dim)) return false;
  bool some = false;
  for (const Constraint& k : ks) {
    if (satisfies(e, k, tol)) continue;
    if (!overlaps(e, k, tol)) return false;
    some = true;
  }
  return some;
}

// The evaluation of the selectors of one document on one table, with memoization of the named boundaries.  With
// allow_axis, of, between and curve also take axis entities: only to tell a selection that is empty because the axis,
// which no named boundary includes, is all it would select.
class Evaluator {
 public:
  Evaluator(const Table& t, const json& selectors, bool allow_axis) : t_(t), selectors_(selectors), allow_axis_(allow_axis) {}

  const Sel& named(const std::string& name) {
    const auto done = memo_.find(name);
    if (done != memo_.end()) return done->second;
    if (!selectors_.is_object() || !selectors_.contains(name))
      throw std::logic_error("selection: the boundary '" + name + "' has no selector (resolve takes grammar-clean selectors)");
    if (!active_.insert(name).second)
      throw std::logic_error("selection: the boundary '" + name + "' names itself through a cycle (resolve takes grammar-clean selectors)");
    Sel s = eval(selectors_.at(name), name);
    active_.erase(name);
    return memo_.emplace(name, std::move(s)).first->second;
  }

  // A selector with its filters: the base, then except, then where.
  Sel eval(const json& node, const std::string& owner) {
    Sel s = unfiltered(node, owner);
    if (node.contains("where")) {
      const std::vector<Constraint> ks = constraints_of(t_, node.at("where"));
      const double tol = tolerance(node);
      for (auto it = s.begin(); it != s.end();) it = kept(t_.entities[it->first], ks, tol) ? std::next(it) : s.erase(it);
    }
    return s;
  }
  // The base minus except, before the where filter.
  Sel unfiltered(const json& node, const std::string& owner) {
    Sel s = base(node, owner);
    if (node.contains("except"))
      for (const json& n : node.at("except"))
        for (const auto& [i, o] : named(n.get<std::string>())) s.erase(i);
    return s;
  }
  double tolerance(const json& node) const {
    return node.contains("tolerance_m") ? node.at("tolerance_m").get<double>() : 1e-9 * t_.diagonal_m;
  }

  // The partial matches of a selector: the entities that a filtered of or between in a position that adds entities to
  // the selection (the selector itself, an operand of a union, the first operand of a difference, at any depth; not
  // through ref, whose boundary reports its own) matches in part, less those that a later operand of an enclosing
  // difference removes.  The caller removes those that the boundary selects through another operand.  `leaves` receives
  // those filtered of and between nodes, in written order.
  std::set<std::size_t> partial_matches(const json& node, const std::string& owner, std::vector<const json*>& leaves) {
    std::set<std::size_t> p;
    if (!node.is_object()) return p;
    if ((node.contains("of") || node.contains("between")) && node.contains("where")) {
      leaves.push_back(&node);
      const std::vector<Constraint> ks = constraints_of(t_, node.at("where"));
      const double tol = tolerance(node);
      for (const auto& [i, o] : unfiltered(node, owner))
        if (!kept(t_.entities[i], ks, tol) && partly(t_.entities[i], ks, tol, t_.geometry == "3d" ? 3 : 2)) p.insert(i);
    } else if (node.contains("union")) {
      for (const json& op : node.at("union"))
        for (std::size_t i : partial_matches(op, owner, leaves)) p.insert(i);
    } else if (node.contains("difference") && node.at("difference").is_array() && !node.at("difference").empty()) {
      const json& ops = node.at("difference");
      p = partial_matches(ops[0], owner, leaves);
      for (std::size_t k = 1; k < ops.size(); ++k)
        for (const auto& [i, o] : eval(ops[k], owner)) p.erase(i);  // removed by the difference in any case
    }
    return p;
  }

  // Per boundary, the regions that the two domains of one of its between selectors share.
  std::map<std::string, std::set<std::string>> shared;

 private:
  bool eligible(const Entity& e) const { return is_boundary_entity(e) && (allow_axis_ || !e.axis); }

  Sel base(const json& node, const std::string& owner) {
    if (!node.is_object()) throw std::logic_error("selection: a selector is an object (resolve takes grammar-clean selectors)");
    Sel s;
    if (node.contains("of")) {
      // exactly one side in D; outward from D
      const Strs d = domain_list(node.at("of"));
      for (std::size_t i = 0; i < t_.entities.size(); ++i) {
        const Entity& e = t_.entities[i];
        if (!eligible(e)) continue;
        const bool in0 = contains(d, e.side[0]), in1 = contains(d, e.side[1]);
        if (in0 != in1) s[i] = in0 ? +1 : -1;
      }
    } else if (node.contains("between")) {
      // one side in A and the other in B; from A to B
      const json& ab = node.at("between");
      if (!ab.is_array() || ab.size() != 2)
        throw std::logic_error("selection: between takes two domains (resolve takes grammar-clean selectors)");
      const Strs a = domain_list(ab[0]), b = domain_list(ab[1]);
      for (const std::string& r : a)
        if (contains(b, r)) shared[owner].insert(r);
      for (std::size_t i = 0; i < t_.entities.size(); ++i) {
        const Entity& e = t_.entities[i];
        if (!eligible(e)) continue;
        if (contains(a, e.side[0]) && contains(b, e.side[1]))
          s[i] = +1;
        else if (contains(b, e.side[0]) && contains(a, e.side[1]))
          s[i] = -1;
      }
    } else if (node.contains("curve")) {
      // the marker's entities; the written tangent rotated by +90 degrees: -1 along the entity's tangent, +1 against it
      const std::string m = node.at("curve").get<std::string>();
      for (std::size_t i = 0; i < t_.entities.size(); ++i) {
        const Entity& e = t_.entities[i];
        if (!eligible(e) || !contains(e.curves, m)) continue;
        s[i] = contains(e.curves_reversed, m) ? +1 : -1;
      }
    } else if (node.contains("union")) {
      // the first operand, in written order, that selects an entity orients it
      for (const json& op : node.at("union"))
        for (const auto& [i, o] : eval(op, owner)) s.emplace(i, o);
    } else if (node.contains("difference")) {
      const json& ops = node.at("difference");
      if (!ops.is_array() || ops.empty())
        throw std::logic_error("selection: difference takes a list of selectors (resolve takes grammar-clean selectors)");
      s = eval(ops[0], owner);
      for (std::size_t k = 1; k < ops.size(); ++k)
        for (const auto& [i, o] : eval(ops[k], owner)) s.erase(i);
    } else if (node.contains("ref")) {
      s = named(node.at("ref").get<std::string>());
    } else {
      throw std::logic_error("selection: a selector is of, between, curve, union, difference or ref (resolve takes grammar-clean selectors)");
    }
    return s;
  }

  const Table& t_;
  const json& selectors_;
  const bool allow_axis_;
  std::map<std::string, Sel> memo_;
  std::set<std::string> active_;
};

bool lexicographically_before(const Entity& a, const Entity& b) {
  return std::tie(a.point[0], a.point[1], a.point[2], a.tag) < std::tie(b.point[0], b.point[1], b.point[2], b.tag);
}

// The candidates of an empty selection (model_selection.hpp, Resolution::empty): the entities of `pre`, grouped by the
// line on which they lie flat, then the others as boxes.
json candidates(const Table& t, const Sel& pre, double tol) {
  const Strs names = coordinate_names(t.geometry);
  std::vector<std::vector<const Entity*>> flat(names.size());  // per coordinate, the entities flat in it first
  std::vector<const Entity*> boxes;
  for (const auto& [i, o] : pre) {
    const Entity& e = t.entities[i];
    int c = 0;
    while (c < int(names.size()) && !(e.hi[c] - e.lo[c] <= tol)) ++c;
    (c < int(names.size()) ? flat[c] : boxes).push_back(&e);
  }
  const auto by_tag = [](const Entity* a, const Entity* b) { return a->tag < b->tag; };
  json out = json::array();
  for (std::size_t c = 0; c < names.size(); ++c) {
    // one line per run of values within tol of the run's first (smallest) value
    std::sort(flat[c].begin(), flat[c].end(),
              [c](const Entity* a, const Entity* b) { return std::tie(a->lo[c], a->tag) < std::tie(b->lo[c], b->tag); });
    for (std::size_t k = 0; k < flat[c].size();) {
      const double value = flat[c][k]->lo[c];
      std::vector<const Entity*> es;
      for (; k < flat[c].size() && flat[c][k]->lo[c] - value <= tol; ++k) es.push_back(flat[c][k]);
      std::sort(es.begin(), es.end(), by_tag);
      double measure = 0.0;
      for (const Entity* e : es) measure += e->length_m;
      json where = json::object();
      where[names[c]] = value;
      out.push_back(json{{"where", where}, {"measure_m", measure}});
    }
  }
  std::sort(boxes.begin(), boxes.end(), [](const Entity* a, const Entity* b) { return lexicographically_before(*a, *b); });
  for (const Entity* e : boxes) {
    json box = json::object();
    for (std::size_t c = 0; c < names.size(); ++c) {
      json range = json::array();
      range.push_back(e->lo[c]);
      range.push_back(e->hi[c]);
      box[names[c]] = range;
    }
    out.push_back(json{{"bbox", box}, {"measure_m", e->length_m}});
  }
  return out;
}

std::vector<int> tags_of(const Table& t, const Sel& s) {
  std::vector<int> v;
  for (const auto& [i, o] : s) v.push_back(t.entities[i].tag);
  std::sort(v.begin(), v.end());
  return v;
}

json strings(const Strs& v) {
  json a = json::array();
  for (const std::string& s : v) a.push_back(s);
  return a;
}
json triple(const double (&v)[3]) {
  json a = json::array();
  for (double x : v) a.push_back(x);
  return a;
}
void read_triple(const json& j, const char* key, double (&v)[3]) {
  if (!j.contains(key)) return;
  const json& a = j.at(key);
  for (std::size_t c = 0; c < 3 && c < a.size(); ++c) v[c] = a[c].get<double>();
}
Strs read_strings(const json& j, const char* key) {
  Strs out;
  if (j.contains(key))
    for (const json& s : j.at(key)) out.push_back(s.get<std::string>());
  return out;
}

// The number of sides of `e` in the regions I.
int sides_in(const Entity& e, const Strs& regions) { return int(contains(regions, e.side[0])) + int(contains(regions, e.side[1])); }

const Entity* find_tag(const Table& t, int tag) {
  for (const Entity& e : t.entities)
    if (e.tag == tag) return &e;
  return nullptr;
}

}  // namespace

Resolution resolve(const Table& t, const nlohmann::json& selectors) {
  Resolution r;
  Evaluator ev(t, selectors, false), with_axis(t, selectors, true);
  std::map<std::string, Sel> chosen;
  if (selectors.is_object())
    for (auto it = selectors.begin(); it != selectors.end(); ++it) chosen[it.key()] = ev.named(it.key());

  // the selections, their orientations and the diagnostics of the top-level where filter
  for (const auto& [name, s] : chosen) {
    r.selected[name] = tags_of(t, s);
    for (const auto& [i, o] : s) r.orientation[{name, t.entities[i].tag}] = o;

    const json& node = selectors.at(name);
    std::vector<const json*> leaves;
    std::vector<int> part;
    for (std::size_t i : ev.partial_matches(node, name, leaves))
      if (!s.count(i)) part.push_back(t.entities[i].tag);
    std::sort(part.begin(), part.end());
    if (s.empty()) {
      // the where and the candidates of the one filtered of or between, when the boundary has one and its filter left
      // nothing
      const json* leaf = leaves.size() == 1 && ev.eval(*leaves[0], name).empty() ? leaves[0] : nullptr;
      json e = json::object();
      e["where"] = leaf ? leaf->at("where") : json(nullptr);
      e["candidates"] = leaf ? candidates(t, ev.unfiltered(*leaf, name), ev.tolerance(*leaf)) : json::array();
      e["partial"] = part;
      e["axis_only"] = !with_axis.named(name).empty();
      r.empty[name] = e;
    } else if (!part.empty()) {
      r.partial[name] = json{{"tags", part}};
    }
  }
  for (const auto& [name, regions] : ev.shared) {
    if (regions.empty() || !chosen.count(name)) continue;
    Strs v;
    for (const std::string& reg : t.regions)
      if (regions.count(reg)) v.push_back(reg);
    for (const std::string& reg : regions)
      if (!contains(v, reg)) v.push_back(reg);
    r.overlap[name] = v;
  }

  // the atoms: the membership signature of every boundary entity
  std::map<std::size_t, Strs> names_of;
  for (const auto& [name, s] : chosen)  // in name order, so each signature is sorted
    for (const auto& [i, o] : s) names_of[i].push_back(name);
  struct Build {
    std::string side[2];
    std::vector<int> tags;
  };
  Build axis;
  std::map<Strs, Build> named;
  std::map<PairKey, Build> unnamed;
  for (std::size_t i = 0; i < t.entities.size(); ++i) {
    const Entity& e = t.entities[i];
    if (!is_boundary_entity(e)) continue;
    Build* b = nullptr;
    std::string side0 = e.side[0], side1 = e.side[1];  // named and axis: the sides of the first entity in table order
    if (e.axis) {
      b = &axis;
    } else if (names_of.count(i)) {
      b = &named[names_of.at(i)];
    } else {  // unnamed: the region pair, in pair order
      const PairKey k = pair_key(t, e);
      b = &unnamed[k];
      side0 = std::get<1>(k), side1 = std::get<3>(k);
    }
    if (b->tags.empty()) b->side[0] = side0, b->side[1] = side1;
    b->tags.push_back(e.tag);
  }
  auto add = [&r](int master, const char* kind, const Strs& names, Build& b) {
    Atom a;
    a.master = master;
    a.kind = kind;
    a.names = names;
    a.side[0] = b.side[0], a.side[1] = b.side[1];
    std::sort(b.tags.begin(), b.tags.end());
    a.tags = b.tags;
    for (int tag : a.tags) r.atom_of_tag[tag] = master;
    r.atoms.push_back(a);
  };
  if (!axis.tags.empty()) add(900, "axis", {}, axis);
  int master = 2001;
  for (auto& [names, b] : named) add(master++, "named", names, b);
  for (auto& [k, b] : unnamed) add(master++, "unnamed", {}, b);
  return r;
}

Partition partition(const Table& t, const Resolution& r, const std::vector<std::string>& regions, const std::vector<std::string>& keys) {
  Partition p;
  const auto atom_of = [&r](int tag) {
    const auto it = r.atom_of_tag.find(tag);
    return it == r.atom_of_tag.end() ? 0 : it->second;
  };
  // the domain boundary: the boundary entities, not on the axis, with exactly one side in I
  std::set<int> domain_boundary, axis_atoms;
  for (const Entity& e : t.entities) {
    if (!is_boundary_entity(e)) continue;
    const int n = sides_in(e, regions);
    if (e.axis) {
      if (n > 0) axis_atoms.insert(atom_of(e.tag));
    } else if (n == 1) {
      domain_boundary.insert(e.tag);
    }
  }
  p.axis_atoms.assign(axis_atoms.begin(), axis_atoms.end());

  std::map<int, Strs> covering;  // domain-boundary tag -> the keys other than "remaining" that cover it, in K order
  std::map<std::string, std::vector<int>> covered;
  for (const std::string& k : keys) {
    if (k == "remaining") continue;
    const auto it = r.selected.find(k);
    if (it == r.selected.end()) continue;
    std::vector<int> on, inside, off;
    for (int tag : it->second) {
      const Entity* e = find_tag(t, tag);
      if (!e || e->axis) continue;
      const int n = sides_in(*e, regions);
      (n == 1 ? on : n == 2 ? inside : off).push_back(tag);
    }
    for (int tag : on) covering[tag].push_back(k);
    covered[k] = on;
    if (!inside.empty()) p.interior_keys_tags[k] = inside;
    if (!it->second.empty() && on.empty() && inside.empty())
      p.outside_keys.push_back(k);
    else if (!off.empty())
      p.outside_part[k] = off;
  }
  const bool remaining = contains(keys, "remaining");
  for (int tag : domain_boundary) {
    if (covering.count(tag)) continue;
    (remaining ? covered["remaining"] : p.uncovered).push_back(tag);
  }
  for (const auto& [tag, ks] : covering)
    if (ks.size() >= 2) p.doubly[tag] = ks;
  for (const std::string& k : keys) {
    std::set<int> atoms;
    for (int tag : covered[k]) atoms.insert(atom_of(tag));
    p.atoms_of_key[k].assign(atoms.begin(), atoms.end());
  }
  return p;
}

Placement place(const Table& t, const Resolution& r, const Partition& p, const std::vector<std::string>& regions, const std::string& name) {
  Placement pl;
  const auto it = r.selected.find(name);
  if (it == r.selected.end()) return pl;
  std::set<int> atoms;  // the atoms of its domain-boundary entities
  for (int tag : it->second) {
    const Entity* e = find_tag(t, tag);
    if (!e) continue;
    const int n = sides_in(*e, regions);
    (n == 1 ? pl.boundary : n == 2 ? pl.interior : pl.outside).push_back(tag);
    if (n == 1) {
      const auto a = r.atom_of_tag.find(tag);
      if (a != r.atom_of_tag.end()) atoms.insert(a->second);
    }
  }
  // A key covers either every domain-boundary entity of an atom or none of them: the entities of an atom are selected
  // by the same names, and "remaining" covers what no other key covers.
  for (const auto& [k, ka] : p.atoms_of_key)  // in key order
    for (int a : ka)
      if (atoms.count(a)) {
        pl.keys.push_back(k);
        break;
      }
  return pl;
}

nlohmann::json to_json(const Table& t) {
  json entities = json::array();
  for (const Entity& e : t.entities) {
    json side = json::array();
    side.push_back(e.side[0]);
    side.push_back(e.side[1]);
    json j = json::object();
    j["tag"] = e.tag;
    j["dim"] = e.dim;
    j["side"] = side;
    j["curves"] = strings(e.curves);
    if (!e.curves_reversed.empty()) j["curves_reversed"] = strings(e.curves_reversed);
    j["axis"] = e.axis;
    j["type"] = e.type;
    j["lo"] = triple(e.lo);
    j["hi"] = triple(e.hi);
    j["point"] = triple(e.point);
    j["tangent"] = triple(e.tangent);
    j["length_m"] = e.length_m;
    j["area_m2"] = e.area_m2;
    entities.push_back(j);
  }
  json out = json::object();
  out["geometry"] = t.geometry;
  out["regions"] = strings(t.regions);
  out["entities"] = entities;
  out["diagonal_m"] = t.diagonal_m;
  return out;
}

Table table_from_json(const nlohmann::json& j) {
  Table t;
  t.geometry = j.value("geometry", std::string());
  t.regions = read_strings(j, "regions");
  t.diagonal_m = j.value("diagonal_m", 0.0);
  if (j.contains("entities"))
    for (const json& ej : j.at("entities")) {
      Entity e;
      e.tag = ej.value("tag", 0);
      e.dim = ej.value("dim", 1);
      if (ej.contains("side") && ej.at("side").size() == 2) {
        e.side[0] = ej.at("side")[0].get<std::string>();
        e.side[1] = ej.at("side")[1].get<std::string>();
      }
      e.curves = read_strings(ej, "curves");
      e.curves_reversed = read_strings(ej, "curves_reversed");
      e.axis = ej.value("axis", false);
      e.type = ej.value("type", std::string());
      read_triple(ej, "lo", e.lo);
      read_triple(ej, "hi", e.hi);
      read_triple(ej, "point", e.point);
      read_triple(ej, "tangent", e.tangent);
      e.length_m = ej.value("length_m", 0.0);
      e.area_m2 = ej.value("area_m2", 0.0);
      t.entities.push_back(e);
    }
  return t;
}

}  // namespace axmp::model::selection
