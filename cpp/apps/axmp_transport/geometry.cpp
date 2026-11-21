// SPDX-License-Identifier: BSD-3-Clause
#include "geometry.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4ExceptionSeverity.hh"
#include "G4GenericPolycone.hh"
#include "G4IonisParamMat.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Orb.hh"
#include "G4PVPlacement.hh"
#include "G4PhysicalConstants.hh"
#include "G4StateManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Torus.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#include "G4Tubs.hh"
#include "G4VExceptionHandler.hh"

#include "runtime.hpp"

namespace axtr {

namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& where, const std::string& what) {
  throw GeometryError((where.empty() ? std::string() : where + ": ") + what);
}

void check_keys(const json& obj, const std::string& where, const std::set<std::string>& required,
                const std::set<std::string>& optional) {
  if (!obj.is_object()) fail(where, "must be an object");
  for (auto it = obj.begin(); it != obj.end(); ++it)
    if (!required.count(it.key()) && !optional.count(it.key())) fail(where, "unknown key '" + it.key() + "'");
  for (const auto& k : required)
    if (!obj.contains(k)) fail(where, "missing key '" + k + "'");
}

std::string string_at(const json& j, const std::string& where) {
  if (!j.is_string()) fail(where, "must be a string");
  return j.get<std::string>();
}

double finite_at(const json& j, const std::string& where) {
  if (!j.is_number()) fail(where, "must be a number");
  const double v = j.get<double>();
  if (!std::isfinite(v)) fail(where, "must be finite");
  return v;
}

double positive_at(const json& j, const std::string& where) {
  const double v = finite_at(j, where);
  if (!(v > 0.0)) fail(where, "must be positive");
  return v;
}

double nonnegative_at(const json& j, const std::string& where) {
  const double v = finite_at(j, where);
  if (!(v >= 0.0)) fail(where, "must not be negative");
  return v;
}

double orient(const std::pair<double, double>& a, const std::pair<double, double>& b,
              const std::pair<double, double>& c) {
  return (b.first - a.first) * (c.second - a.second) - (b.second - a.second) * (c.first - a.first);
}

using Pt = std::pair<double, double>;

// Whether the segments ab and cd lie on one line and share a stretch of it (more than a point).
bool run_back(const Pt& a, const Pt& b, const Pt& c, const Pt& d) {
  if (orient(a, b, c) != 0 || orient(a, b, d) != 0 || orient(c, d, a) != 0 || orient(c, d, b) != 0) return false;
  const bool by_r = std::abs(a.first - b.first) >= std::abs(a.second - b.second);
  auto key = [&](const Pt& p) { return by_r ? p.first : p.second; };
  return std::min(std::max(key(a), key(b)), std::max(key(c), key(d))) >
         std::max(std::min(key(a), key(b)), std::min(key(c), key(d)));
}

// The one outline this program refuses before Geant4 sees it: two edges that are not neighbours and run along each
// other for a stretch (the outline doubles back over itself), which Geant4 builds into a solid of overlapping
// boundary. Everything else that makes a polycone unbuildable (almost no area, crossing edges, fewer than three
// points left once Geant4 merges the vertices closer than its length tolerance) is left to Geant4, whose refusal
// ends the run as an invalid file (see build_geometry).
void check_polycone(const std::vector<Pt>& p, const std::string& where) {
  const std::size_t n = p.size();
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 2; j < n; ++j) {
      if (i == 0 && j == n - 1) continue;  // neighbours through the closing edge
      if (run_back(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]))
        fail(where, "runs back over itself along edges " + std::to_string(i) + " and " + std::to_string(j));
    }
}

// A tessellated solid must be what Geant4 would otherwise only warn about: a closed surface, every edge used once in
// each direction by the triangles, that encloses a positive volume (so that the triangles turn outwards).
void check_tessellated(const std::vector<std::array<double, 3>>& v, const std::vector<std::array<int, 3>>& tri,
                       const std::string& where) {
  std::map<std::pair<int, int>, int> directed;
  double six_volume = 0.0;
  for (const auto& t : tri) {
    if (t[0] == t[1] || t[1] == t[2] || t[2] == t[0]) fail(where, "has a triangle with a repeated vertex");
    for (int e = 0; e < 3; ++e) ++directed[{t[e], t[(e + 1) % 3]}];
    const auto &a = v[t[0]], &b = v[t[1]], &c = v[t[2]];
    six_volume += a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) +
                  a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  for (const auto& [edge, count] : directed) {
    if (count != 1)
      fail(where, "is not closed: the edge " + std::to_string(edge.first) + "-" + std::to_string(edge.second) +
                      " is used " + std::to_string(count) + " times in one direction");
    if (!directed.count({edge.second, edge.first}))
      fail(where, "is not closed: the edge " + std::to_string(edge.first) + "-" + std::to_string(edge.second) +
                      " is not used in the other direction");
  }
  if (!(six_volume > 0.0)) fail(where, "encloses no positive volume (its triangles must turn outwards)");
}

MaterialSpec read_material(const json& m, const std::string& where) {
  check_keys(m, where, {"name", "gdml_name", "density_kg_m3", "composition"},
             {"mean_excitation_eV", "state", "T_K", "pressure_Pa"});
  MaterialSpec out;
  out.name = string_at(m.at("name"), where + ".name");
  if (out.name.empty() || out.name.rfind("G4_", 0) == 0)
    fail(where + ".name", "'" + out.name + "' is empty or a NIST name (G4_...)");
  out.gdml_name = string_at(m.at("gdml_name"), where + ".gdml_name");
  if (out.gdml_name.empty() || out.gdml_name.rfind("G4_", 0) == 0)
    fail(where + ".gdml_name", "'" + out.gdml_name + "' is empty or a NIST name (G4_...)");
  out.density_kg_m3 = positive_at(m.at("density_kg_m3"), where + ".density_kg_m3");
  const json& c = m.at("composition");
  const std::string cw = where + ".composition";
  if (!c.is_object()) fail(cw, "must be an object");
  if (c.contains("nist")) {
    check_keys(c, cw, {"nist"}, {});
    out.nist = string_at(c.at("nist"), cw + ".nist");
    if (out.nist.size() <= 3 || out.nist.rfind("G4_", 0) != 0) fail(cw + ".nist", "'" + out.nist + "' is not G4_...");
  } else {
    check_keys(c, cw, {"elements", "by"}, {});
    const std::string by = string_at(c.at("by"), cw + ".by");
    const json& els = c.at("elements");
    if (!els.is_object() || els.empty()) fail(cw + ".elements", "must be a non-empty object");
    for (auto it = els.begin(); it != els.end(); ++it) {
      const std::string ew = cw + ".elements." + it.key();
      if (by == "mass") {
        out.mass_fractions.emplace_back(it.key(), positive_at(it.value(), ew));
      } else if (by == "atoms") {
        if (!it.value().is_number_integer() || it.value().get<long long>() < 1 ||
            it.value().get<long long>() > 1000000000LL)
          fail(ew, "must be a positive integer");
        out.atoms.emplace_back(it.key(), static_cast<int>(it.value().get<long long>()));
      } else {
        fail(cw + ".by", "must be 'mass' or 'atoms', not '" + by + "'");
      }
    }
  }
  if (m.contains("mean_excitation_eV"))
    out.mean_excitation_eV = positive_at(m.at("mean_excitation_eV"), where + ".mean_excitation_eV");
  if (m.contains("T_K")) out.T_K = positive_at(m.at("T_K"), where + ".T_K");
  if (m.contains("pressure_Pa")) out.pressure_Pa = positive_at(m.at("pressure_Pa"), where + ".pressure_Pa");
  if (m.contains("state")) {
    out.state = string_at(m.at("state"), where + ".state");
    if (out.state != "solid" && out.state != "liquid" && out.state != "gas")
      fail(where + ".state", "must be solid, liquid or gas, not '" + out.state + "'");
  }
  return out;
}

void read_vector3(const json& j, const std::string& where, double out[3]) {
  if (!j.is_array() || j.size() != 3) fail(where, "must be a list of three numbers");
  for (int k = 0; k < 3; ++k) out[k] = finite_at(j[k], where + "[" + std::to_string(k) + "]");
}

SolidSpec read_solid(const json& s, const std::string& where, bool three_d) {
  check_keys(s, where, {"name", "gdml_name", "kind", "params", "mother", "position_m", "material"},
             {"rotation_deg", "region", "discretized", "sagitta_m"});
  SolidSpec out;
  out.name = string_at(s.at("name"), where + ".name");
  if (out.name.empty()) fail(where + ".name", "must not be empty");
  const std::string w = "solid '" + out.name + "'";
  out.gdml_name = string_at(s.at("gdml_name"), w + ".gdml_name");
  if (out.gdml_name.empty()) fail(w + ".gdml_name", "must not be empty");
  out.kind = string_at(s.at("kind"), w + ".kind");
  if (!s.at("mother").is_null()) {
    out.mother = string_at(s.at("mother"), w + ".mother");
    if (out.mother.empty()) fail(w + ".mother", "must not be empty");
  }
  read_vector3(s.at("position_m"), w + ".position_m", out.position_m);
  if (s.contains("rotation_deg")) {
    double rot[3];
    read_vector3(s.at("rotation_deg"), w + ".rotation_deg", rot);
    if (rot[0] != 0.0 || rot[1] != 0.0 || rot[2] != 0.0)
      fail(w + ".rotation_deg", "this version places solids without rotation (a list of this version has none)");
  }
  if (s.contains("region") && !s.at("region").is_null()) out.region = string_at(s.at("region"), w + ".region");
  out.material = string_at(s.at("material"), w + ".material");
  const json& p = s.at("params");
  const std::string pw = w + ".params";
  if (out.kind == "tube") {
    check_keys(p, pw, {"rmin_m", "rmax_m", "z_m", "startphi_deg", "deltaphi_deg"}, {});
    out.rmin_m = nonnegative_at(p.at("rmin_m"), pw + ".rmin_m");
    out.rmax_m = positive_at(p.at("rmax_m"), pw + ".rmax_m");
    out.z_m = positive_at(p.at("z_m"), pw + ".z_m");
    if (!(out.rmin_m < out.rmax_m)) fail(pw, "rmin_m must be below rmax_m");
  } else if (out.kind == "torus") {
    check_keys(p, pw, {"rmin_m", "rmax_m", "rtor_m", "startphi_deg", "deltaphi_deg"}, {});
    out.rmin_m = nonnegative_at(p.at("rmin_m"), pw + ".rmin_m");
    out.rmax_m = positive_at(p.at("rmax_m"), pw + ".rmax_m");
    out.rtor_m = positive_at(p.at("rtor_m"), pw + ".rtor_m");
    if (!(out.rmin_m < out.rmax_m) || !(out.rmax_m < out.rtor_m))
      fail(pw, "the radii must satisfy rmin_m < rmax_m < rtor_m");
  } else if (out.kind == "orb") {
    check_keys(p, pw, {"r_m"}, {});
    out.r_m = positive_at(p.at("r_m"), pw + ".r_m");
  } else if (out.kind == "genericPolycone") {
    check_keys(p, pw, {"startphi_deg", "deltaphi_deg", "rz_m"}, {});
    const json& rz = p.at("rz_m");
    if (!rz.is_array() || rz.size() < 3) fail(pw + ".rz_m", "must be a list of at least three [r, z] points");
    for (std::size_t i = 0; i < rz.size(); ++i) {
      const std::string iw = pw + ".rz_m[" + std::to_string(i) + "]";
      if (!rz[i].is_array() || rz[i].size() != 2) fail(iw, "must be [r, z]");
      out.rz_m.emplace_back(nonnegative_at(rz[i][0], iw + "[0]"), finite_at(rz[i][1], iw + "[1]"));
    }
    check_polycone(out.rz_m, pw + ".rz_m");
  } else if (out.kind == "tessellated" && three_d) {
    check_keys(p, pw, {"vertices_m", "triangles"}, {});
    const json& vs = p.at("vertices_m");
    const json& ts = p.at("triangles");
    if (!vs.is_array() || vs.size() < 4) fail(pw + ".vertices_m", "must be a list of at least four [x, y, z] points");
    if (!ts.is_array() || ts.size() < 4) fail(pw + ".triangles", "must be a list of at least four triples of indices");
    for (std::size_t i = 0; i < vs.size(); ++i) {
      const std::string iw = pw + ".vertices_m[" + std::to_string(i) + "]";
      if (!vs[i].is_array() || vs[i].size() != 3) fail(iw, "must be [x, y, z]");
      out.vertices_m.push_back({finite_at(vs[i][0], iw + "[0]"), finite_at(vs[i][1], iw + "[1]"),
                                finite_at(vs[i][2], iw + "[2]")});
    }
    for (std::size_t i = 0; i < ts.size(); ++i) {
      const std::string iw = pw + ".triangles[" + std::to_string(i) + "]";
      if (!ts[i].is_array() || ts[i].size() != 3) fail(iw, "must be three vertex indices");
      std::array<int, 3> t{};
      for (int k = 0; k < 3; ++k) {
        if (!ts[i][k].is_number_integer()) fail(iw, "must be three integer vertex indices");
        const long long v = ts[i][k].get<long long>();
        if (v < 0 || v >= static_cast<long long>(vs.size())) fail(iw, "names a vertex the list does not have");
        t[k] = static_cast<int>(v);
      }
      out.triangles.push_back(t);
    }
    check_tessellated(out.vertices_m, out.triangles, w);
  } else {
    fail(w + ".kind", std::string("'") + out.kind + "' is not tube, torus, orb, genericPolycone" +
                          (three_d ? " or tessellated" : ""));
  }
  if (p.contains("startphi_deg")) {
    out.startphi_deg = finite_at(p.at("startphi_deg"), pw + ".startphi_deg");
    out.deltaphi_deg = positive_at(p.at("deltaphi_deg"), pw + ".deltaphi_deg");
  }
  return out;
}

// What Geant4 raises as a fatal exception while the geometry is built (a solid it refuses to construct).
struct GeantFatal : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Records the warnings Geant4 raises while it is installed (the overlap check reports through them) and passes every
// exception on to the program's handler. While ``throw_fatal`` is set, a fatal exception is thrown as a GeantFatal
// instead, carrying Geant4's origin, code and message, so that the construction of a solid Geant4 refuses ends the
// geometry run as an invalid file instead of a fatal error.
class RecordingHandler : public G4VExceptionHandler {
 public:
  struct Entry {
    std::string code, description;
  };
  std::vector<Entry> warnings;
  bool throw_fatal = false;

  G4bool Notify(const char* origin, const char* code, G4ExceptionSeverity severity, const char* description) override {
    if (severity == JustWarning) warnings.push_back({code ? code : "", description ? description : ""});
    if (throw_fatal && (severity == FatalException || severity == FatalErrorInArgument))
      throw GeantFatal(std::string(origin ? origin : "") + " [" + (code ? code : "") + "]: " +
                       (description ? description : ""));
    return exception_handler()->Notify(origin, code, severity, description);
  }
};

}  // namespace

SolidsSpec load_solids(const std::filesystem::path& file) {
  std::ifstream in(file);
  if (!in) fail(file.string(), "cannot be read");
  json doc;
  try {
    doc = json::parse(in);
  } catch (const json::parse_error& e) {
    fail(file.string(), std::string("is not JSON: ") + e.what());
  }
  check_keys(doc, "", {"schema", "geometry", "units", "world", "materials", "solids"},
             {"generator", "chord_tol_m", "regions", "warnings"});
  if (string_at(doc.at("schema"), "schema") != kSolidsSchema)
    fail("schema", "must be '" + std::string(kSolidsSchema) + "'");
  const std::string geometry = string_at(doc.at("geometry"), "geometry");
  if (geometry != "axisymmetric" && geometry != "3d")
    fail("geometry", "must be 'axisymmetric' or '3d', not '" + geometry + "'");
  const json& units = doc.at("units");
  if (!units.is_object()) fail("units", "must be an object");
  const std::map<std::string, std::string> want = {{"length", "m"}, {"angle", "deg"}, {"density", "kg/m3"}};
  for (const auto& [k, v] : want)
    if (!units.contains(k) || !units.at(k).is_string() || units.at(k).get<std::string>() != v)
      fail("units." + k, "must be '" + v + "'");

  SolidsSpec spec;
  spec.world = string_at(doc.at("world"), "world");
  const json& mats = doc.at("materials");
  if (!mats.is_array()) fail("materials", "must be a list");
  std::set<std::string> mat_names, mat_gdml;
  for (std::size_t i = 0; i < mats.size(); ++i) {
    spec.materials.push_back(read_material(mats[i], "materials[" + std::to_string(i) + "]"));
    if (!mat_names.insert(spec.materials.back().name).second)
      fail("materials", "'" + spec.materials.back().name + "' is defined twice");
    if (!mat_gdml.insert(spec.materials.back().gdml_name).second)
      fail("materials", "the GDML name '" + spec.materials.back().gdml_name + "' is used twice");
  }
  const json& solids = doc.at("solids");
  if (!solids.is_array() || solids.empty()) fail("solids", "must be a non-empty list");
  std::set<std::string> seen, seen_gdml;
  for (std::size_t i = 0; i < solids.size(); ++i) {
    SolidSpec s = read_solid(solids[i], "solids[" + std::to_string(i) + "]", geometry == "3d");
    const std::string w = "solid '" + s.name + "'";
    if (i == 0) {
      if (s.name != spec.world || !s.mother.empty()) fail(w, "the first solid must be the world, without a mother");
    } else if (s.mother.empty()) {
      fail(w, "has no mother (only the world has none)");
    } else if (!seen.count(s.mother)) {
      fail(w + ".mother", "'" + s.mother + "' is not a solid defined before it");
    }
    if (!seen.insert(s.name).second) fail(w, "is defined twice");
    if (!seen_gdml.insert(s.gdml_name).second) fail(w + ".gdml_name", "'" + s.gdml_name + "' is used twice");
    if (s.material.rfind("G4_", 0) != 0 && !mat_names.count(s.material))
      fail(w + ".material", "'" + s.material + "' is neither a material of the file nor a NIST material (G4_...)");
    spec.solids.push_back(std::move(s));
  }
  // the order of the regions: the file's "regions" object, read with its key order kept
  if (doc.contains("regions")) {
    std::ifstream again(file);
    const nlohmann::ordered_json od = nlohmann::ordered_json::parse(again);
    if (!od.at("regions").is_object()) fail("regions", "must be an object");
    for (auto it = od.at("regions").begin(); it != od.at("regions").end(); ++it) spec.regions.push_back(it.key());
  }
  for (const SolidSpec& s : spec.solids) {
    if (!s.region) continue;
    bool known = false;
    for (const auto& r : spec.regions) known = known || r == *s.region;
    if (!known) {
      if (doc.contains("regions")) fail("solid '" + s.name + "'.region", "'" + *s.region + "' is not in 'regions'");
      spec.regions.push_back(*s.region);
    }
  }
  return spec;
}

BuiltGeometry build_geometry(const SolidsSpec& spec) {
  auto* nist = G4NistManager::Instance();
  BuiltGeometry out;
  std::map<std::string, G4Material*> materials;
  auto nist_material = [&](const std::string& name) {
    G4Material* m = nist->FindOrBuildMaterial(name);
    if (!m) throw GeometryError("Geant4's NIST tables have no material '" + name + "'");
    return m;
  };
  auto element = [&](const std::string& material, const std::string& symbol) {
    G4Element* e = nist->FindOrBuildElement(symbol);
    if (!e)
      throw GeometryError("material '" + material + "': Geant4's NIST tables have no element '" + symbol + "'");
    return e;
  };
  for (const MaterialSpec& m : spec.materials) {
    G4State state = kStateUndefined;
    if (m.state == "solid") state = kStateSolid;
    if (m.state == "liquid") state = kStateLiquid;
    if (m.state == "gas") state = kStateGas;
    const G4double T = m.T_K ? *m.T_K * kelvin : NTP_Temperature;
    const G4double P = m.pressure_Pa ? *m.pressure_Pa * hep_pascal : CLHEP::STP_Pressure;
    const G4double rho = m.density_kg_m3 * (kg / m3);
    const G4int n = !m.nist.empty() ? 1 : static_cast<G4int>(m.mass_fractions.size() + m.atoms.size());
    auto* mat = new G4Material(m.gdml_name, rho, n, state, T, P);
    if (!m.nist.empty()) {
      mat->AddMaterial(nist_material(m.nist), 1.0);
    } else {
      for (const auto& [sym, w] : m.mass_fractions) mat->AddElement(element(m.name, sym), w);
      for (const auto& [sym, k] : m.atoms) mat->AddElement(element(m.name, sym), k);
    }
    if (m.mean_excitation_eV) mat->GetIonisation()->SetMeanExcitationEnergy(*m.mean_excitation_eV * eV);
    materials[m.name] = mat;
    out.materials.emplace_back(m.name, mat);
  }

  std::map<std::string, std::size_t> index;
  for (const SolidSpec& s : spec.solids) {
    const std::string sname = s.gdml_name + "_solid";
    G4VSolid* solid = nullptr;
    const G4double sphi = s.startphi_deg * deg, dphi = s.deltaphi_deg * deg;
    if (s.kind == "tube") {
      solid = new G4Tubs(sname, s.rmin_m * m, s.rmax_m * m, s.z_m * (0.5 * m), sphi, dphi);
    } else if (s.kind == "torus") {
      solid = new G4Torus(sname, s.rmin_m * m, s.rmax_m * m, s.rtor_m * m, sphi, dphi);
    } else if (s.kind == "orb") {
      solid = new G4Orb(sname, s.r_m * m);
    } else if (s.kind == "tessellated") {
      auto* tess = new G4TessellatedSolid(sname);
      auto point = [&](int i) {
        return G4ThreeVector(s.vertices_m[i][0] * m, s.vertices_m[i][1] * m, s.vertices_m[i][2] * m);
      };
      for (const auto& t : s.triangles) tess->AddFacet(new G4TriangularFacet(point(t[0]), point(t[1]), point(t[2]), ABSOLUTE));
      tess->SetSolidClosed(true);
      solid = tess;
    } else {
      std::vector<G4double> r, z;
      for (const auto& [ri, zi] : s.rz_m) {
        r.push_back(ri * m);
        z.push_back(zi * m);
      }
      solid = new G4GenericPolycone(sname, sphi, dphi, static_cast<G4int>(r.size()), r.data(), z.data());
    }
    G4Material* material = nullptr;
    if (s.material.rfind("G4_", 0) == 0) {
      material = nist_material(s.material);
      bool listed = false;
      for (const auto& [name, mat] : out.materials) listed = listed || mat == material;
      if (!listed) out.materials.emplace_back(s.material, material);
    } else {
      material = materials.at(s.material);
    }
    auto* logical = new G4LogicalVolume(solid, material, s.gdml_name);
    BuiltVolume v;
    v.spec = &s;
    if (s.region)
      for (std::size_t r = 0; r < spec.regions.size(); ++r)
        if (spec.regions[r] == *s.region) v.region = static_cast<int>(r);
    v.solid = solid;
    v.logical = logical;
    if (s.mother.empty()) {
      v.physical = new G4PVPlacement(nullptr, G4ThreeVector(), logical, s.gdml_name + "_PV", nullptr, false, 0, false);
      out.world = v.physical;
    } else {
      BuiltVolume& mother = out.volumes.at(index.at(s.mother));
      for (int k = 0; k < 3; ++k) v.offset_m[k] = mother.offset_m[k] + s.position_m[k];
      const G4ThreeVector pos(s.position_m[0] * m, s.position_m[1] * m, s.position_m[2] * m);
      v.physical = new G4PVPlacement(nullptr, pos, logical, s.gdml_name + "_pv", mother.logical, false, 0, false);
      mother.daughters.push_back(out.volumes.size());
    }
    index[s.name] = out.volumes.size();
    out.volumes.push_back(v);
  }
  return out;
}

int run_geometry(const std::filesystem::path& file) {
  SolidsSpec spec;
  try {
    spec = load_solids(file);
  } catch (const GeometryError& e) {
    std::cerr << "axmp-transport: error: invalid solids file " << file << ": " << e.what() << std::endl;
    return kUsage;
  }
  exception_handler();  // the program's handler exists before ours replaces it
  RecordingHandler recorder;
  G4StateManager::GetStateManager()->SetExceptionHandler(&recorder);

  BuiltGeometry geo;
  recorder.throw_fatal = true;
  try {
    geo = build_geometry(spec);
  } catch (const GeometryError& e) {
    std::cerr << "axmp-transport: error: " << e.what() << std::endl;
    G4StateManager::GetStateManager()->SetExceptionHandler(exception_handler());
    return kUsage;
  } catch (const GeantFatal& e) {
    std::cerr << "axmp-transport: error: invalid solids file " << file << ": Geant4 refuses to build the geometry: "
              << e.what() << std::endl;
    G4StateManager::GetStateManager()->SetExceptionHandler(exception_handler());
    return kUsage;
  }
  recorder.throw_fatal = false;

  // Geant4's overlap check on every placement, once the whole tree is built (each placement is checked against its
  // mother and all its siblings)
  json overlaps = json::array();
  for (const BuiltVolume& v : geo.volumes) {
    if (v.spec->mother.empty()) continue;
    const std::size_t before = recorder.warnings.size();
    const bool overlap = static_cast<G4PVPlacement*>(v.physical)->CheckOverlaps(1000, 0.0, true, 1);
    std::string message;
    for (std::size_t k = before; k < recorder.warnings.size(); ++k)
      if (recorder.warnings[k].code == "GeomVol1002")
        message += (message.empty() ? "" : "\n") + recorder.warnings[k].description;
    // a check that could not sample the solid's surface reports no overlap but a warning: it did not check
    if (overlap || !message.empty())
      overlaps.push_back({{"volume", v.spec->name}, {"mother", v.spec->mother}, {"message", message}});
  }
  G4StateManager::GetStateManager()->SetExceptionHandler(exception_handler());

  json volumes = json::array();
  for (const BuiltVolume& v : geo.volumes) {
    G4double own = v.solid->GetCubicVolume();
    for (std::size_t d : v.daughters) own -= geo.volumes[d].solid->GetCubicVolume();
    const double solid_m3 = v.solid->GetCubicVolume() / m3;
    const double own_m3 = own / m3;
    const double rho = v.logical->GetMaterial()->GetDensity() / (kg / m3);
    volumes.push_back({{"name", v.spec->name},
                       {"g4_name", v.logical->GetName()},
                       {"placement", v.physical->GetName()},
                       {"region", v.spec->region ? json(*v.spec->region) : json(nullptr)},
                       {"material", v.spec->material},
                       {"solid_volume_m3", solid_m3},
                       {"volume_m3", own_m3},
                       {"density_kg_m3", rho},
                       {"mass_kg", own_m3 * rho}});
  }
  // the materials as Geant4 built them: what a GDML application reading the exported file must reproduce
  json materials = json::array();
  for (const auto& [name, mat] : geo.materials) {
    json elements = json::object();
    const G4ElementVector* ev = mat->GetElementVector();
    const G4double* fraction = mat->GetFractionVector();
    for (std::size_t k = 0; k < mat->GetNumberOfElements(); ++k) {
      const std::string sym = (*ev)[k]->GetSymbol();
      elements[sym] = (elements.contains(sym) ? elements[sym].get<double>() : 0.0) + fraction[k];
    }
    const char* state = "undefined";
    if (mat->GetState() == kStateSolid) state = "solid";
    if (mat->GetState() == kStateLiquid) state = "liquid";
    if (mat->GetState() == kStateGas) state = "gas";
    materials.push_back({{"name", name},
                         {"g4_name", mat->GetName()},
                         {"density_kg_m3", mat->GetDensity() / (kg / m3)},
                         {"mean_excitation_eV", mat->GetIonisation()->GetMeanExcitationEnergy() / eV},
                         {"state", state},
                         {"T_K", mat->GetTemperature() / kelvin},
                         {"pressure_Pa", mat->GetPressure() / hep_pascal},
                         {"elements", elements}});
  }
  json out;
  if (overlaps.empty()) {
    out = {{"ok", true}, {"volumes", volumes}, {"materials", materials}, {"overlaps", overlaps}};
  } else {
    std::ostringstream msg;
    msg << "Geant4's overlap check refuses " << overlaps.size() << " placement(s):";
    for (const auto& o : overlaps)
      msg << " " << o.at("volume").get<std::string>() << " (in " << o.at("mother").get<std::string>() << ")";
    out = {{"ok", false},
           {"code", "particle.geometry_overlap"},
           {"error", msg.str()},
           {"overlaps", overlaps},
           {"volumes", volumes},
           {"materials", materials}};
  }
  std::cout << out.dump(1) << std::endl;
  return overlaps.empty() ? kOk : kRefused;
}

}  // namespace axtr
