// The commands of a model document that need the geometry (model_commands.hpp).
#include "axionmultiphysics/model_commands.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>

#include "error_json.hpp"
#include "model_errors.hpp"
#include "model_reads.hpp"

namespace axmp::model {

using nlohmann::json;
using Strs = std::vector<std::string>;

namespace {

// ------------------------------------------------------------------------------------------- the table
// The partition of every instance that has a geometry and conditions, for the part of a key that an instance takes.
struct Taken {
  Strs regions;
  selection::Partition partition;
};

std::map<std::string, Taken> partitions_of(const ValidationReport& report, const selection::Table& table, const selection::Resolution& res) {
  std::map<std::string, Taken> out;
  for (const auto& [id, in] : report.model.instances) {
    const auto regs = report.regions.find(id);
    if (in.conditions.empty() || regs == report.regions.end() || regs->second.empty()) continue;
    Strs keys;
    for (const auto& [key, c] : in.conditions) keys.push_back(key);
    out[id] = Taken{regs->second, selection::partition(table, res, regs->second, keys)};
  }
  return out;
}

const selection::Entity* find_entity(const selection::Table& t, int tag) {
  for (const selection::Entity& e : t.entities)
    if (e.tag == tag) return &e;
  return nullptr;
}

json box_json(const Strs& names, const double lo[3], const double hi[3]) {
  json b = json::object();
  for (size_t c = 0; c < names.size(); ++c) b[names[c]] = json::array({lo[c], hi[c]});
  return b;
}
json point_json(size_t dim, const double p[3]) {
  json a = json::array();
  for (size_t c = 0; c < dim; ++c) a.push_back(p[c]);
  return a;
}

}  // namespace

json boundary_table(const ValidationReport& report, const selection::Table& table, const selection::Resolution& res,
                    const std::map<int, int>* faces) {
  const ResolvedModel& m = report.model;
  const Strs coords = msg::coordinate_names(table.geometry);
  const bool axisym = table.geometry == "axisymmetric";
  const auto faces_of = [&](int tag) {
    const auto it = faces->find(tag);
    return it == faces->end() ? 0 : it->second;
  };
  const std::map<std::string, Taken> taken = partitions_of(report, table, res);

  json rows = json::object();
  for (const auto& [name, sel] : m.boundaries) {
    const auto found = res.selected.find(name);
    std::vector<int> tags = found == res.selected.end() ? std::vector<int>() : found->second;
    std::sort(tags.begin(), tags.end());
    std::vector<const selection::Entity*> es;
    for (int t : tags)
      if (const selection::Entity* e = find_entity(table, t)) es.push_back(e);
    json row = json::object();
    row["selector"] = sel.written;
    row["entities"] = es.size();
    double measure = 0.0, area = 0.0, lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    int nfaces = 0;
    for (const selection::Entity* e : es) {
      measure += e->length_m;
      area += e->area_m2;
      for (size_t c = 0; c < coords.size(); ++c) {
        lo[c] = std::min(lo[c], e->lo[c]);
        hi[c] = std::max(hi[c], e->hi[c]);
      }
      if (faces) nfaces += faces_of(e->tag);
    }
    row["measure"] = measure;
    row["measure_units"] = "m";
    if (axisym) row["area_m2"] = area;
    if (es.empty()) {
      row["bbox"] = nullptr;
      row["point"] = nullptr;
      row["adjacent"] = json::array();
    } else {
      row["bbox"] = box_json(coords, lo, hi);
      row["point"] = point_json(coords.size(), msg::first_entity(es).point);
      row["adjacent"] = msg::pairs_json(msg::side_pairs(table, es));
    }
    json used = json::array();
    for (const auto& [id, in] : m.instances) {
      const auto c = in.conditions.find(name);
      const auto part = taken.find(id);
      if (c == in.conditions.end() || part == taken.end()) continue;
      const selection::Placement pl = selection::place(table, res, part->second.partition, part->second.regions, name);
      double mine = 0.0;
      for (int t : pl.boundary)
        if (const selection::Entity* e = find_entity(table, t)) mine += e->length_m;
      used.push_back({{"physics", id}, {"condition", c->second.kind}, {"entities", pl.boundary.size()}, {"measure", mine}});
    }
    for (const auto& [oname, o] : m.outputs)
      if (o.on == name) used.push_back({{"output", oname}, {"kind", o.kind}});
    row["used_by"] = used;
    if (faces) row["faces"] = nfaces;
    rows[name] = row;
  }

  std::map<int, const selection::Atom*> atom_of;
  for (const selection::Atom& a : res.atoms) atom_of[a.master] = &a;
  std::vector<std::pair<std::vector<double>, json>> edges;
  for (const selection::Entity& e : table.entities) {
    const auto at = res.atom_of_tag.find(e.tag);
    if (at == res.atom_of_tag.end()) continue;
    json j = json::object();
    j["adjacent"] = msg::pairs_json(msg::side_pairs(table, {&e}))[0];
    j["type"] = e.type;
    j["bbox"] = box_json(coords, e.lo, e.hi);
    j["point"] = point_json(coords.size(), e.point);
    j["measure"] = e.length_m;
    j["measure_units"] = "m";
    if (axisym) j["area_m2"] = e.area_m2;
    Strs names;
    const auto atom = atom_of.find(at->second);
    if (atom != atom_of.end()) names = atom->second->names;
    std::sort(names.begin(), names.end());
    j["names"] = names;
    j["axis"] = e.axis;
    if (faces) j["faces"] = faces_of(e.tag);
    std::vector<double> key(e.point, e.point + coords.size());
    key.push_back(e.tag);
    edges.emplace_back(key, j);
  }
  std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  json list = json::array();
  for (const auto& [k, j] : edges) list.push_back(j);
  return json{{"symmetry", table.geometry}, {"boundaries", rows}, {"edges", list}};
}

namespace {

// ------------------------------------------------------------------------------------------- the common steps
struct Prepared {
  ValidationReport report;
  std::optional<Error> refusal;     // what stops the command; a report error, a refusal or a failed run
  bool geometry = true;             // false: nothing to mesh
  bool resolved = false;
  BoundaryResolution resolution;
};

MeshOptions mesh_options(const CommandOptions& o) {
  MeshOptions m;
  m.workdir = o.workdir;
  m.mesher = o.mesher;
  m.verbose = o.verbose;
  return m;
}

// Validation with the probe, the refusals of this version, the resolve_only run and the full check.
Prepared prepare(const json& document, const CommandOptions& o, const std::string& operation) {
  Prepared p;
  ValidateOptions vo;
  vo.probe_mesher = true;
  vo.mesher = o.mesher;
  p.report = validate_model(document, vo);
  if (!p.report.ok()) {
    p.refusal = p.report.errors.front();
    return p;
  }
  const ResolvedModel& m = p.report.model;
  if (m.symmetry == "3d") {
    p.refusal = msg::mesh_3d_not_implemented(operation);
    return p;
  }
  if (!m.mesh.file.empty()) {
    p.refusal = msg::mesh_imported_not_implemented(operation, m.mesh.file);
    return p;
  }
  if (m.regions.empty()) {
    p.geometry = false;
    return p;
  }
  for (const Error& u : p.report.unavailable)
    if (u.code() == codes::requires_mesher) {
      p.refusal = u;
      return p;
    }
  try {
    p.resolution = resolve_boundary_table(p.report, mesh_options(o));
    p.resolved = true;
  } catch (const Error& e) {
    if (e.code() == codes::mesh_generation_failed) p.report.errors.push_back(e);
    p.refusal = e;
    return p;
  }
  check_boundaries(p.report, p.resolution.table, p.resolution.resolution, o.document);
  if (!p.report.ok()) p.refusal = p.report.errors.front();
  return p;
}

// The atoms of two resolutions agree (master number, kind and names): the meshing run resolved what the resolve_only run did.
void same_atoms(const selection::Resolution& a, const selection::Resolution& b, const std::string& info_file) {
  bool same = a.atoms.size() == b.atoms.size();
  for (size_t i = 0; same && i < a.atoms.size(); ++i)
    same = a.atoms[i].master == b.atoms[i].master && a.atoms[i].kind == b.atoms[i].kind && a.atoms[i].names == b.atoms[i].names &&
           a.atoms[i].tags.size() == b.atoms[i].tags.size();
  if (!same) throw msg::mesh_export_files("the meshing run resolved the boundaries into other atoms than the resolve_only run", info_file);
}

std::map<int, int> face_counts(const MeshedModel& m) {
  std::map<int, int> counts;
  for (int tag : m.boundary_entity) ++counts[tag];
  return counts;
}

// The meshing run after a full check without errors.
MeshedModel run_mesh(const Prepared& p, const CommandOptions& o) {
  MeshedModel m = mesh_model(p.report, mesh_options(o));
  same_atoms(p.resolution.resolution, m.resolution, m.info_file);
  return m;
}

json regions_json(const MeshedModel& m, const std::string& symmetry) {
  json regions = json::object();
  std::vector<std::pair<int, std::string>> order;
  for (const auto& [name, attr] : m.region_attribute) order.emplace_back(attr, name);
  std::sort(order.begin(), order.end());
  for (const auto& [attr, name] : order) {
    const json& r = m.info.at("regions").at(name);
    json row = {{"area_m2", r.value("area_m2", 0.0)}};
    if (symmetry == "axisymmetric") row["volume_m3"] = r.value("volume_m3", 0.0);
    regions[name] = row;
  }
  return regions;
}

// The report as `axmp boundaries --json` prints it: the validation report, the first error's fields at the top level when
// the command is refused, and the table when the boundaries were resolved.
json refused_report(const Prepared& p) {
  json out = report_json(p.report, false);
  if (p.refusal) {
    json top = error_json(*p.refusal);
    top.erase("ok");
    out["ok"] = false;
    for (auto it = top.begin(); it != top.end(); ++it) out[it.key()] = it.value();
  }
  return out;
}

}  // namespace

// ------------------------------------------------------------------------------------------- axmp boundaries
BoundariesResult boundaries_report(const json& document, const CommandOptions& options) {
  if (!spec::is_model_document(document)) throw msg::boundaries_needs_model_document();
  Prepared p = prepare(document, options, "boundaries");
  BoundariesResult out;
  std::unique_ptr<MeshedModel> meshed;
  if (!p.refusal && p.geometry && options.mesh) {
    try {
      meshed = std::make_unique<MeshedModel>(run_mesh(p, options));
    } catch (const Error& e) {
      if (e.code() == codes::mesh_generation_failed) p.report.errors.push_back(e);
      p.refusal = e;
    }
  }
  out.report = refused_report(p);
  out.error = p.refusal;
  if (p.resolved || !p.geometry) {
    json table = p.resolved ? json() : json{{"symmetry", p.report.model.symmetry}, {"boundaries", json::object()}, {"edges", json::array()}};
    std::map<int, int> faces;
    if (meshed) faces = face_counts(*meshed);
    if (p.resolved) table = boundary_table(p.report, p.resolution.table, p.resolution.resolution, meshed ? &faces : nullptr);
    for (auto it = table.begin(); it != table.end(); ++it) out.report[it.key()] = it.value();
  }
  return out;
}

std::string boundaries_text(const json& report) {
  std::ostringstream os;
  const std::string name = report.value("name", std::string());
  os << "Boundaries of '" << name << "' (" << report.value("symmetry", std::string()) << ")\n";
  const json& rows = report.at("boundaries");
  if (rows.empty()) os << "  no named boundary\n";
  for (auto it = rows.begin(); it != rows.end(); ++it) {
    const json& r = it.value();
    os << "\n  " << it.key() << ": " << msg::edges_text(r.at("entities").get<size_t>()) << ", " << msg::num_text(r.at("measure").get<double>())
       << " m";
    if (r.contains("area_m2")) os << ", " << msg::num_text(r.at("area_m2").get<double>()) << " m2";
    if (r.contains("faces")) os << ", " << r.at("faces").get<int>() << " faces";
    os << "\n    selector: " << r.at("selector").dump() << "\n    adjacent: ";
    const json& adj = r.at("adjacent");
    if (adj.empty()) os << "none";
    for (size_t i = 0; i < adj.size(); ++i) os << (i ? ", " : "") << adj[i][0].get<std::string>() << " | " << adj[i][1].get<std::string>();
    os << "\n    used by: ";
    const json& used = r.at("used_by");
    if (used.empty()) os << "nothing";
    for (size_t i = 0; i < used.size(); ++i) {
      os << (i ? ", " : "");
      if (used[i].contains("physics")) os << used[i].at("physics").get<std::string>() << " (" << used[i].at("condition").get<std::string>() << ")";
      else os << "output " << used[i].at("output").get<std::string>() << " (" << used[i].at("kind").get<std::string>() << ")";
    }
    os << "\n";
  }
  const json& edges = report.at("edges");
  os << "\nEdges that no boundary names:\n";
  bool any = false;
  for (const json& e : edges) {
    if (!e.at("names").empty()) continue;
    any = true;
    os << "  " << e.at("type").get<std::string>() << " at " << e.at("point").dump() << ", " << msg::num_text(e.at("measure").get<double>()) << " m, "
       << e.at("adjacent")[0].get<std::string>() << " | " << e.at("adjacent")[1].get<std::string>() << (e.at("axis").get<bool>() ? " (axis)" : "") << "\n";
  }
  if (!any) os << "  none\n";
  return os.str();
}

// ------------------------------------------------------------------------------------------- axmp mesh
json mesh_command(const json& document, const CommandOptions& options) {
  // an earlier run's files in the directory are removed first, so that no refusal leaves a mesh behind
  // (never for a document on an imported mesh, whose own files may have these names)
  const bool imported = document.is_object() && document.contains("mesh") && document.at("mesh").is_object() && document.at("mesh").contains("file");
  if (!options.workdir.empty() && spec::is_model_document(document) && !imported) {
    std::error_code ec;
    for (const char* f : {"mesh.msh", "mesh_info.json", "mesh_in.json"}) std::filesystem::remove(options.workdir + "/" + f, ec);
  }
  const Prepared p = prepare(document, options, "mesh");
  if (p.refusal) throw *p.refusal;
  json warnings = json::array();
  for (const Error& w : p.report.warnings) {
    json j = error_json(w);
    j.erase("ok");
    warnings.push_back(j);
  }
  if (!p.geometry)
    return json{{"ok", true},
                {"name", p.report.model.name},
                {"symmetry", p.report.model.symmetry},
                {"note", "this model document has no geometry (no regions and no imported mesh): there is nothing to mesh"},
                {"warnings", warnings}};
  const MeshedModel m = run_mesh(p, options);
  const json table = boundary_table(p.report, p.resolution.table, p.resolution.resolution);
  json boundaries = json::object();
  for (auto it = table.at("boundaries").begin(); it != table.at("boundaries").end(); ++it) {
    json row = {{"measure", it.value().at("measure")}, {"measure_units", "m"}};
    if (it.value().contains("area_m2")) row["area_m2"] = it.value().at("area_m2");
    boundaries[it.key()] = row;
  }
  const std::string dir = options.workdir;
  const auto path = [&](const char* file) { return dir.empty() ? json(nullptr) : json(dir + "/" + file); };
  return json{{"ok", true},
              {"name", p.report.model.name},
              {"symmetry", m.symmetry},
              {"geometry_order", m.geometry_order},
              {"mesh", path("mesh.msh")},
              {"mesher_input", path("mesh_in.json")},
              {"mesher_info", path("mesh_info.json")},
              {"boundaries", boundaries},
              {"warnings", warnings}};
}

// ------------------------------------------------------------------------------------------- axmp.mesh
std::unique_ptr<MeshedBundle> mesh_bundle(const json& document, const CommandOptions& options) {
  if (!spec::is_model_document(document)) throw msg::mesh_needs_model_document();
  Prepared p = prepare(document, options, "mesh");
  if (p.refusal) throw *p.refusal;
  if (!p.geometry) throw msg::mesh_no_geometry_not_implemented();
  auto b = std::make_unique<MeshedBundle>();
  b->model = run_mesh(p, options);
  b->symmetry = b->model.symmetry;
  b->boundaries = boundary_table(p.report, p.resolution.table, p.resolution.resolution).at("boundaries");
  b->regions = regions_json(b->model, b->symmetry);
  b->report = std::move(p.report);
  return b;
}

InstanceMesh MeshedBundle::instance(const Strs& names, bool all) const {
  const ResolvedModel& m = report.model;
  Strs wanted;
  if (all && names.empty()) {
    for (const spec::Region& r : m.regions) wanted.push_back(r.name);
  } else {
    if (names.empty()) throw std::invalid_argument("instance: an empty list of regions; name at least one region or domain");
    for (const std::string& n : names) {
      const bool region = std::any_of(m.regions.begin(), m.regions.end(), [&](const spec::Region& r) { return r.name == n; });
      const bool domain = m.domains.count(n) > 0;
      if (!region && !domain) {
        Strs known;
        for (const spec::Region& r : m.regions) known.push_back(r.name);
        for (const auto& [d, v] : m.domains) known.push_back(d);
        throw std::invalid_argument("instance: '" + n + "' is not an enabled region or a domain of the model (known: " + msg::quoted(known) + ")");
      }
    }
    wanted = detail::regions_of(m, names);
    if (wanted.empty()) throw std::invalid_argument("instance: the names stand for no enabled region");
  }
  return model.instance_mesh(wanted);
}

}  // namespace axmp::model
