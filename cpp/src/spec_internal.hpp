// Internal to the specification parser: the helpers that the parser of a design and the parser of a model
// document share. Not installed.
#pragma once

#include <initializer_list>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/expr.hpp"
#include "axionmultiphysics/spec.hpp"

namespace axmp::spec::detail {

// True while a "symmetry": "xsec" design is parsed: shapes and paths live in the (x, y) cross-section plane
// (no r >= 0 restriction; rect also accepts "x"/"y" for "r"/"z"; disks may be placed by pivot, arm and angle).
extern thread_local bool g_planar;
struct PlanarGuard {
  explicit PlanarGuard(bool on) { g_planar = on; }
  ~PlanarGuard() { g_planar = false; }
};

// Reads values from a JSON node while tracking its path for error messages.
struct Node {
  const json& j;
  std::string path;
  const ParamMap* params;

  Node at(const std::string& key) const {
    return Node{j.at(key), path + "/" + key, params};
  }
  Node at(size_t i) const { return Node{j.at(i), strcat(path, "/", i), params}; }
  bool has(const std::string& key) const { return j.is_object() && j.contains(key); }

  double num() const {
    if (j.is_number()) return j.get<double>();
    if (j.is_string()) return eval_expr(j.get<std::string>(), *params, path);
    throw Error("expected a number or an expression string", path, "Example: 2.5 or \"0.6*R\".");
  }
  double num(const std::string& key, double def) const { return has(key) ? at(key).num() : def; }
  double req_num(const std::string& key) const {
    if (!has(key)) throw Error("missing required field '" + key + "'", path);
    return at(key).num();
  }
  std::string str() const {
    if (!j.is_string()) throw Error("expected a string", path);
    return j.get<std::string>();
  }
  std::string str(const std::string& key, const std::string& def) const {
    return has(key) ? at(key).str() : def;
  }
  std::string req_str(const std::string& key) const {
    if (!has(key)) throw Error("missing required field '" + key + "'", path);
    return at(key).str();
  }
  bool boolean(const std::string& key, bool def) const {
    if (!has(key)) return def;
    if (!j.at(key).is_boolean()) throw Error("expected true or false", path + "/" + key);
    return j.at(key).get<bool>();
  }
  Vec2 vec2() const {
    if (!j.is_array() || j.size() != 2) throw Error("expected a point [r, z]", path);
    return Vec2{at(size_t(0)).num(), at(size_t(1)).num()};
  }
  std::pair<double, double> range() const {
    if (!j.is_array() || j.size() != 2) throw Error("expected a range [min, max]", path);
    double a = at(size_t(0)).num(), b = at(size_t(1)).num();
    if (!(b > a)) throw Error(strcat("range must satisfy min < max (got ", a, ", ", b, ")"), path);
    return {a, b};
  }
  void only(std::initializer_list<const char*> allowed) const {
    if (!j.is_object()) throw Error("expected an object", path);
    std::set<std::string> ok(allowed.begin(), allowed.end());
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (!ok.count(it.key())) {
        std::string list;
        for (auto& a : ok) list += (list.empty() ? "" : ", ") + a;
        throw Error("unknown field '" + it.key() + "'", path, "Allowed fields here: " + list + ".");
      }
    }
  }
};

void check_nonneg_r(double r, const std::string& path);
void check_polygon_fillets(const std::vector<Vec2>& P, const std::vector<double>& f, const std::string& path);
Shape parse_shape(const Node& n);
Path parse_path(const Node& n);
Vec3 vec3(const Node& n);
Solid parse_solid(const Node& n);
Constraint parse_constraint(const Node& n);
void parse_mechanical(const Node& n, MechanicalProps& m);
void parse_thermal_material(const Node& n, ThermalProps& t);
Material parse_material(const std::string& name, const Node& n);
Conductor parse_conductor(const std::string& name, const Node& n, const Conductor& base);

}  // namespace axmp::spec::detail
