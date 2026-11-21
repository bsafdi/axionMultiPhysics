// The meshed model of a model document and the instance mesh of a physics instance (model_mesh.hpp).
//
// THE INSTANCE MESH.  MFEM's SubMesh of the master's elements of the instance's regions is copied into a plain
// mfem::Mesh (its nodes included), together with the SubMesh's maps to the master, and the SubMesh goes.  The SubMesh
// gives every face with one adjacent element a boundary element, and also transfers the master's boundary elements on
// faces interior to it; RemoveInternalBoundaries drops the latter, so that the boundary elements are exactly the faces
// with one adjacent element.  Each takes the master number of the master's boundary element on the same face (found
// through the parent edge map), never the SubMesh's attribute: the SubMesh gives a face that has no master boundary
// element the fabricated attribute max + 1, and such a face is refused instead (a missing atom).  The master numbers
// present are then numbered densely, in ascending order.
//
// SECTIONS.  The faces of a named boundary interior to an instance take their signs from the MASTER's boundary element
// on the face: MFEM turns only the boundary elements of exterior faces when it reads or finalizes a mesh, so a boundary
// element on an interior face keeps the direction in which the mesh file writes it, which is its entity's tangent
// (model_selection.hpp, CONVENTIONS), and the resolution's orientation of (name, entity) is relative to that
// direction.  The SubMesh re-creates its interior boundary elements from its own faces, in the faces' vertex order, so
// its boundary elements cannot give the sign.
#include "axionmultiphysics/model_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "axionmultiphysics/axi.hpp"
#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/mesh_quality.hpp"
#include "axionmultiphysics/spec_model.hpp"
#include "model_errors.hpp"

namespace axmp::model {

using nlohmann::json;
using Strs = std::vector<std::string>;

namespace {

std::string regions_text(const Strs& regions) { return msg::quoted_and(regions); }

// ------------------------------------------------------------------------------------------- the mesher's input
void check_report(const ValidationReport& report, const char* who) {
  if (!report.ok())
    throw std::invalid_argument(std::string(who) + ": the document has errors; mesh a document that validates");
  const ResolvedModel& m = report.model;
  if (!m.mesh.file.empty())
    throw std::invalid_argument(std::string(who) + ": the document imports the mesh '" + m.mesh.file +
                                "', which the mesher does not make");
  if (m.symmetry != "planar" && m.symmetry != "axisymmetric")
    throw std::invalid_argument(std::string(who) + ": the mesher's bounded modes mesh planar and axisymmetric models, not '" +
                                m.symmetry + "'");
  if (m.regions.empty()) throw std::invalid_argument(std::string(who) + ": the document has no enabled region");
}

// The enabled regions of `name`, a domain or a region, in the document order of the regions (none for a disabled
// region).
json regions_of(const ResolvedModel& m, const std::string& name) {
  json out = json::array();
  const auto d = m.domains.find(name);
  for (const spec::Region& r : m.regions) {
    const bool in = d != m.domains.end()
                        ? std::find(d->second.regions.begin(), d->second.regions.end(), r.name) != d->second.regions.end()
                        : r.name == name;
    if (in) out.push_back(r.name);
  }
  return out;
}

// A selector of the document in the mesher's input form (model_selection.hpp).
json selector_input(const ResolvedModel& m, const Selector& s, const std::string& boundary) {
  if (s.kind == "ref") return json{{"ref", s.name}};
  json out = json::object();
  if (s.kind == "of") {
    out["of"] = regions_of(m, s.name);
  } else if (s.kind == "between") {
    const json second = s.between.at(1) == "exterior" ? json("exterior") : regions_of(m, s.between.at(1));
    out["between"] = json::array({regions_of(m, s.between.at(0)), second});
  } else if (s.kind == "curve") {
    out["curve"] = s.name;
  } else if (s.kind == "union" || s.kind == "difference") {
    json ops = json::array();
    for (const Selector& op : s.operands) ops.push_back(selector_input(m, op, boundary));
    out[s.kind] = ops;
  } else {
    throw std::invalid_argument("bounded_mesher_input: the boundary '" + boundary + "' selects by '" + s.kind +
                                "', which only an imported mesh offers");
  }
  if (s.kind == "of" || s.kind == "between") {
    const bool written = s.written.is_object();
    if (written && s.written.contains("where")) {
      json w = json::object();
      for (const auto& [c, v] : s.where) w[c] = v;
      out["where"] = w;
    }
    if (written && s.written.contains("tolerance_m")) out["tolerance_m"] = s.tolerance_m;
    if (written && s.written.contains("except")) out["except"] = s.except;
  }
  return out;
}

json boundaries_input(const ResolvedModel& m) {
  json b = json::object();
  for (const auto& [name, sel] : m.boundaries) b[name] = selector_input(m, sel, name);
  return b;
}

// ------------------------------------------------------------------------------------------- running the mesher
// The working directory of a run: the one given (created if missing), else a temporary one removed at the end.
struct WorkDir {
  std::string path;
  bool temporary = false;
  explicit WorkDir(const std::string& dir) {
    std::error_code ec;
    if (!dir.empty()) {
      std::filesystem::create_directories(dir, ec);
      AXMP_REQUIRE(!ec, "cannot create the working directory '" + dir + "': " + ec.message());
      path = dir;
      return;
    }
    std::string t = (std::filesystem::temp_directory_path(ec) / "axmp_mesh_XXXXXX").string();
    std::vector<char> buf(t.begin(), t.end());
    buf.push_back('\0');
    AXMP_REQUIRE(mkdtemp(buf.data()) != nullptr, "cannot create a temporary working directory for the mesher");
    path = buf.data();
    temporary = true;
  }
  ~WorkDir() {
    if (!temporary) return;
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  WorkDir(const WorkDir&) = delete;
  WorkDir& operator=(const WorkDir&) = delete;
};

std::string read_text(const std::string& path, size_t limit = std::string::npos) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  std::string s = ss.str();
  if (s.size() > limit) s = s.substr(0, limit) + " ...";
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}

// The last lines of a text file (at most `lines` lines and `bytes` bytes), for an error hint: a temporary working
// directory is gone by the time the error is read, and the log holds the reason when the mesher left no info file.
std::string tail_text(const std::string& path, size_t lines = 40, size_t bytes = 4000) {
  std::string s = read_text(path);
  size_t cut = 0, seen = 0;
  for (size_t i = s.size(); i-- > 0;)
    if (s[i] == '\n' && ++seen == lines) {
      cut = i + 1;
      break;
    }
  s = s.substr(cut);
  if (s.size() > bytes) s = "..." + s.substr(s.size() - bytes);
  return s;
}

// The JSON object of a file; null when the file is missing or is not a JSON object.
json read_object(const std::string& path) {
  std::ifstream in(path);
  if (!in) return nullptr;
  json j = json::parse(in, nullptr, false);
  return j.is_object() ? j : json(nullptr);
}

// Writes the input next to the files, runs the mesher on it and returns its info file; a run that fails is an
// axmp::Error with the mesher's own message.
json run_mesher(const json& in, const std::string& dir, const std::string& hint) {
  const std::string input = dir + "/mesh_in.json", log = dir + "/mesh_mesher.log";
  const std::string info_path = in.at("info").get<std::string>(), output = in.at("output").get<std::string>();
  std::error_code ec;
  std::filesystem::remove(info_path, ec);   // no stale file is read as this run's
  std::filesystem::remove(output, ec);
  {
    std::ofstream f(input);
    AXMP_REQUIRE(f, "cannot write the mesher's input '" + input + "'");
    f << in.dump(2) << "\n";
  }
  const std::string mesher = axi::find_mesher(hint);
  const int rc = axi::run_process({mesher, input}, log);
  const json info = read_object(info_path);
  if (rc != 0 || !info.is_object() || !info.value("ok", false)) {
    std::string why = "the mesher wrote no info file";
    if (info.is_object()) why = info.contains("error") && info["error"].is_string() ? info["error"].get<std::string>() : "no reason given";
    throw msg::mesh_generation_failed(why, log, rc, tail_text(log));
  }
  return info;
}

// ------------------------------------------------------------------------------------------- reading the files
// The entity table of an info file (its "regions" object ordered by attribute), and the region attributes.
selection::Table table_of(const json& info, const std::string& file, std::map<std::string, int>& region_attribute) {
  if (!info.contains("regions") || !info["regions"].is_object() || !info.contains("entities") || !info["entities"].is_array())
    throw msg::mesh_export_files("the info file has no \"regions\" object or no \"entities\" list", file);
  std::vector<std::pair<int, std::string>> regs;
  for (auto it = info["regions"].begin(); it != info["regions"].end(); ++it) {
    if (!it.value().is_object() || !it.value().contains("attr") || !it.value()["attr"].is_number_integer())
      throw msg::mesh_export_files("the region '" + it.key() + "' of the info file has no integer \"attr\"", file);
    regs.emplace_back(it.value()["attr"].get<int>(), it.key());
  }
  std::sort(regs.begin(), regs.end());
  Strs names;
  region_attribute.clear();
  for (const auto& [attr, name] : regs) {
    names.push_back(name);
    region_attribute[name] = attr;
  }
  const json t = {{"geometry", info.value("geometry", std::string())},
                  {"regions", names},
                  {"entities", info["entities"]},
                  {"diagonal_m", info.value("diagonal_m", 0.0)}};
  return selection::table_from_json(t);
}

// The atoms of the resolution equal the info file's "boundary_groups" (master number, type and names).
void check_groups(const selection::Resolution& r, const json& info, const std::string& file) {
  using Group = std::pair<std::string, Strs>;
  std::map<int, Group> lib, mesher;
  for (const selection::Atom& a : r.atoms) lib[a.master] = {a.kind, a.names};
  if (!info.contains("boundary_groups") || !info["boundary_groups"].is_array())
    throw msg::mesh_export_files("the info file has no \"boundary_groups\" list", file);
  for (const json& g : info["boundary_groups"]) {
    if (!g.is_object() || !g.contains("attr") || !g["attr"].is_number_integer())
      throw msg::mesh_export_files("a boundary group of the info file has no integer \"attr\"", file);
    Strs names;
    if (g.contains("names") && g["names"].is_array())
      for (const json& n : g["names"]) names.push_back(n.is_string() ? n.get<std::string>() : n.dump());
    std::sort(names.begin(), names.end());
    mesher[g["attr"].get<int>()] = {g.value("type", std::string()), names};
  }
  if (lib == mesher) return;
  auto text = [](const Group& g) { return g.first + " " + msg::quoted(g.second); };
  std::set<int> masters;
  for (const auto& [k, v] : lib) masters.insert(k);
  for (const auto& [k, v] : mesher) masters.insert(k);
  for (int k : masters) {
    const auto a = lib.find(k), b = mesher.find(k);
    if (a != lib.end() && b != mesher.end() && a->second == b->second) continue;
    const std::string ours = a == lib.end() ? std::string("absent") : text(a->second);
    const std::string theirs = b == mesher.end() ? std::string("absent") : text(b->second);
    throw msg::mesh_export_files("the atom " + std::to_string(k) + " is " + ours + " in the library's resolution of the boundaries and " +
                                     theirs + " in the info file's boundary groups",
                                 file);
  }
}

// gmsh element types of dimension 1: the lines of 2 to 11 nodes.
bool is_line_type(int type) { return type == 1 || type == 8 || type == 26 || type == 27 || type == 28 || (type >= 62 && type <= 66); }

// The elementary (entity) tag of every element of dimension 1 of an ASCII MSH 2 file, in the order of the file, which
// is the order in which MFEM makes them boundary elements of a 2D mesh.
std::vector<int> msh_line_entities(const std::string& path) {
  std::ifstream in(path);
  AXMP_REQUIRE(in, "cannot read the mesh file '" + path + "'");
  auto trim = [](std::string& s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
  };
  std::string line;
  while (std::getline(in, line)) {
    trim(line);
    if (line == "$MeshFormat") {
      std::getline(in, line);
      std::istringstream ss(line);
      std::string version;
      int file_type = -1;
      ss >> version >> file_type;
      if (version.rfind("2.", 0) != 0 || file_type != 0)
        throw msg::mesh_export_files("the mesh file is not an ASCII MSH 2 file (version " + version + ", file type " +
                                         std::to_string(file_type) + ")",
                                     path);
    } else if (line == "$Elements") {
      std::getline(in, line);
      const long n = std::strtol(line.c_str(), nullptr, 10);
      std::vector<int> out;
      for (long i = 0; i < n; ++i) {
        if (!std::getline(in, line)) throw msg::mesh_export_files("the $Elements section of the mesh file is cut short", path);
        std::istringstream ss(line);
        long id = 0;
        int type = 0, ntags = 0;
        ss >> id >> type >> ntags;
        std::vector<int> tags(std::max(ntags, 0));
        for (int& t : tags) ss >> t;
        if (!ss) throw msg::mesh_export_files("the element line '" + line + "' of the mesh file cannot be read", path);
        if (!is_line_type(type)) continue;
        if (ntags < 2) throw msg::mesh_export_files("the line element " + std::to_string(id) + " of the mesh file has no elementary tag", path);
        out.push_back(tags[1]);
      }
      return out;
    }
  }
  throw msg::mesh_export_files("the mesh file has no $Elements section", path);
}

// One uniform refinement of the master, carrying the entity of each boundary element to its two children.  MFEM
// splits boundary element i into the elements 2i (from its first vertex) and 2i + 1 (to its second), and keeps the
// vertex numbers of the coarse mesh; both are checked.
void refine(mfem::Mesh& mesh, std::vector<int>& entity, const std::string& file) {
  const int nbe = mesh.GetNBE();
  std::vector<std::pair<int, int>> ends(nbe);
  for (int be = 0; be < nbe; ++be) {
    mfem::Array<int> v;
    mesh.GetBdrElementVertices(be, v);
    ends[be] = {v[0], v[1]};
  }
  mesh.UniformRefinement();
  if (mesh.GetNBE() != 2 * nbe)
    throw msg::mesh_export_files("a uniform refinement of the master mesh gives " + std::to_string(mesh.GetNBE()) +
                                     " boundary elements from " + std::to_string(nbe),
                                 file);
  std::vector<int> out(2 * nbe);
  for (int be = 0; be < nbe; ++be) {
    mfem::Array<int> a, b;
    mesh.GetBdrElementVertices(2 * be, a);
    mesh.GetBdrElementVertices(2 * be + 1, b);
    const bool ok = (a[0] == ends[be].first || a[1] == ends[be].first) && (b[0] == ends[be].second || b[1] == ends[be].second);
    if (!ok)
      throw msg::mesh_export_files("a uniform refinement of the master mesh does not number the children of boundary element " +
                                       std::to_string(be) + " as 2i and 2i + 1",
                                   file);
    out[2 * be] = out[2 * be + 1] = entity[be];
  }
  entity.swap(out);
}

// ------------------------------------------------------------------------------------------- faces
// The length of a face on the mesh's geometry (curved or straight).
double face_length(const mfem::Mesh& mesh, int f) {
  mfem::IsoparametricTransformation T;
  mesh.GetFaceTransformation(f, &T);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, mesh.GetNodes() ? 16 : 1);
  double s = 0.0;
  for (int k = 0; k < ir.GetNPoints(); ++k) {
    const mfem::IntegrationPoint& ip = ir.IntPoint(k);
    T.SetIntPoint(&ip);
    s += ip.weight * T.Weight();
  }
  return s;
}

// The face points of the faces (face, sign), in that order: per face its first element, and per point of the
// segment rule of `order` the reference point in that element, the weight times the face Jacobian, the unit normal
// times the sign (MFEM's face normal points out of the first element) and the physical point.
FacePoints points_on(const mfem::Mesh& mesh, const std::vector<std::pair<int, int>>& faces, int order) {
  if (order < 0) throw std::invalid_argument("face_points: the order of the rule is negative (" + std::to_string(order) + ")");
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, order);
  FacePoints fp;
  mfem::FaceElementTransformations T;
  mfem::IsoparametricTransformation T1, T2;
  mfem::Vector X(2), nor(2);
  for (const auto& [f, sign] : faces) {
    mesh.GetFaceElementTransformations(f, T, T1, T2);
    fp.face.push_back(f);
    fp.element.push_back(T.Elem1No);
    for (int k = 0; k < ir.GetNPoints(); ++k) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(k);
      T.SetAllIntPoints(&ip);
      const mfem::IntegrationPoint& eip = T.GetElement1IntPoint();
      fp.ref.push_back(eip.x);
      fp.ref.push_back(eip.y);
      fp.weight.push_back(ip.weight * T.Weight());
      mfem::CalcOrtho(T.Jacobian(), nor);
      const double len = nor.Norml2();
      fp.normal.push_back(sign * nor(0) / len);
      fp.normal.push_back(sign * nor(1) / len);
      T.Transform(ip, X);
      fp.x.push_back(X(0));
      fp.x.push_back(X(1));
    }
  }
  return fp;
}

void face_ends(const mfem::Mesh& mesh, int f, double p[2], double q[2]) {
  mfem::Array<int> v;
  mesh.GetFaceVertices(f, v);
  const double *a = mesh.GetVertex(v[0]), *b = mesh.GetVertex(v[1]);
  p[0] = a[0], p[1] = a[1], q[0] = b[0], q[1] = b[1];
}

const mfem::Mesh& mesh_of(const InstanceMesh& im, const char* who) {
  if (!im.mesh) throw std::invalid_argument(std::string(who) + ": the instance mesh has no mesh");
  return *im.mesh;
}

}  // namespace

// ------------------------------------------------------------------------------------------- the probe
MesherStatus mesher_status(const std::string& hint) {
  MesherStatus st;
  try {
    st.path = axi::find_mesher(hint);
  } catch (const Error& e) {
    st.why = e.what();
    return st;
  }
  static std::mutex mu;
  static std::map<std::string, MesherStatus> cache;
  const std::lock_guard<std::mutex> lock(mu);
  if (const auto it = cache.find(st.path); it != cache.end()) return it->second;
  std::error_code ec;
  std::string t = (std::filesystem::temp_directory_path(ec) / "axmp_mesher_probe_XXXXXX").string();
  std::vector<char> buf(t.begin(), t.end());
  buf.push_back('\0');
  const int fd = mkstemp(buf.data());
  if (fd < 0) {
    st.why = "cannot create the log file of the probe of '" + st.path + "'";
    return st;
  }
  close(fd);
  const std::string log = buf.data();
  try {
    const int rc = axi::run_process({st.path}, log);
    const std::string text = read_text(log, 4000);
    if (rc == 2) {
      st.available = true;   // the usage text: the program loaded
    } else if (rc == 127) {
      st.why = text.empty() ? "'" + st.path + "' could not be loaded (exit status 127)" : text;
    } else {
      st.why = "'" + st.path + "' " + (rc < 0 ? std::string("was stopped by a signal") : "exited with status " + std::to_string(rc)) +
               " when run without arguments" + (text.empty() ? std::string() : ": " + text);
    }
  } catch (const Error& e) {
    st.why = e.message();
  }
  std::filesystem::remove(log, ec);
  cache[st.path] = st;
  return st;
}

// ------------------------------------------------------------------------------------------- the mesher's input
nlohmann::json bounded_mesher_input(const ValidationReport& report, bool resolve_only, const std::string& workdir) {
  check_report(report, "bounded_mesher_input");
  const ResolvedModel& m = report.model;
  json regions = json::array();
  double lo[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  double hi[2] = {-lo[0], -lo[1]};
  int attr = 1;
  for (const spec::Region& g : m.regions) {
    json r = {{"name", g.name}, {"attr", attr++}, {"shape", g.shape.resolved()}};
    if (g.mesh_size > 0) r["size"] = g.mesh_size;
    if (g.surface_size > 0) r["surface_size"] = g.surface_size;
    regions.push_back(r);
    double r0 = 0, r1 = 0, z0 = 0, z1 = 0;
    g.shape.bbox(r0, r1, z0, z1);
    lo[0] = std::min(lo[0], r0), hi[0] = std::max(hi[0], r1);
    lo[1] = std::min(lo[1], z0), hi[1] = std::max(hi[1], z1);
  }
  json curves = json::array();
  for (const spec::Curve& c : m.curves) {
    json cj = {{"name", c.name}, {"path", c.path.resolved()}};
    if (c.mesh_size > 0) cj["size"] = c.mesh_size;
    if (c.surface_size > 0) cj["surface_size"] = c.surface_size;
    curves.push_back(cj);
  }
  // the default size: 1/40 of the diagonal of the bounding box of the enabled regions
  const double size = m.mesh.size > 0 ? m.mesh.size : std::hypot(hi[0] - lo[0], hi[1] - lo[1]) / 40.0;
  const double surface_size = m.mesh.surface_size > 0 ? m.mesh.surface_size : size;
  return json{{"bounded", true},
              {"geometry", m.symmetry},
              {"exterior", m.mesh.exterior},
              {"regions", regions},
              {"curves", curves},
              {"boundaries", boundaries_input(m)},
              {"size", size},
              {"surface_size", surface_size},
              {"curvature_points", 24},
              {"order", report.geometry_order},
              {"resolve_only", resolve_only},
              {"output", workdir + "/mesh.msh"},
              {"info", workdir + "/mesh_info.json"},
              {"verbose", 0}};
}

BoundaryResolution resolve_boundary_table(const ValidationReport& report, const MeshOptions& options) {
  check_report(report, "resolve_boundaries");
  const WorkDir dir(options.workdir);
  json in = bounded_mesher_input(report, true, dir.path);
  if (options.verbose > 1) in["verbose"] = 1;
  const std::string info_file = in.at("info").get<std::string>();
  const json info = run_mesher(in, dir.path, options.mesher);
  BoundaryResolution out;
  out.table = table_of(info, info_file, out.region_attribute);
  out.resolution = selection::resolve(out.table, in.at("boundaries"));
  check_groups(out.resolution, info, info_file);
  out.info = info;
  return out;
}

selection::Resolution resolve_boundaries(const ValidationReport& report, const MeshOptions& options) {
  return resolve_boundary_table(report, options).resolution;
}

// ------------------------------------------------------------------------------------------- the meshed model
MeshedModel mesh_model(const ValidationReport& report, const MeshOptions& options) {
  check_report(report, "mesh_model");
  const WorkDir dir(options.workdir);
  json in = bounded_mesher_input(report, false, dir.path);
  if (options.verbose > 1) in["verbose"] = 1;
  run_mesher(in, dir.path, options.mesher);
  MeshedModel m = read_meshed_model(report, in.at("output").get<std::string>(), in.at("info").get<std::string>(),
                                    report.model.mesh.refinements);
  if (dir.temporary) {   // the files are removed with the directory
    m.mesh_file.clear();
    m.info_file.clear();
  }
  return m;
}

MeshedModel read_meshed_model(const ValidationReport& report, const std::string& mesh_file, const std::string& info_file,
                              int refinements) {
  if (refinements < 0) throw std::invalid_argument("read_meshed_model: a negative number of refinements");
  MeshedModel m;
  m.mesh_file = mesh_file;
  m.info_file = info_file;
  m.info = read_object(info_file);
  AXMP_REQUIRE(m.info.is_object(), "cannot read the mesher's info file '" + info_file + "' (missing, or not a JSON object)");
  if (!m.info.value("ok", false)) throw msg::mesh_export_files("the info file is not that of a successful run (\"ok\" is not true)", info_file);
  m.table = table_of(m.info, info_file, m.region_attribute);
  m.symmetry = m.table.geometry;
  {
    std::ifstream probe(mesh_file);
    AXMP_REQUIRE(probe, "cannot read the mesh file '" + mesh_file + "'");
  }
  m.boundary_entity = msh_line_entities(mesh_file);
  m.master = std::make_unique<mfem::Mesh>(mesh_file.c_str(), 1, 1, true);
  if (m.master->Dimension() != 2) throw msg::mesh_export_files("the mesh file is not a 2D mesh", mesh_file);
  {
    // a tangled curved element is refused whoever wrote the file (the mesher repairs or refuses one itself)
    mesh_quality::UntangledOptions untangled;
    untangled.coords_2d = m.symmetry == "axisymmetric" ? "(r, z)" : "(x, y)";
    untangled.order_one = "mesh.geometry_order 1";
    mesh_quality::require_untangled(
        *m.master,
        [&](int a) {
          for (const auto& [name, attr] : m.region_attribute)
            if (attr == a) return name;
          return strcat("attribute ", a);
        },
        mesh_file, untangled);
  }
  if (static_cast<int>(m.boundary_entity.size()) != m.master->GetNBE())
    throw msg::mesh_export_files("the mesh file has " + std::to_string(m.boundary_entity.size()) + " line elements and MFEM reads " +
                                     std::to_string(m.master->GetNBE()) + " boundary elements",
                                 mesh_file);
  for (int k = 0; k < refinements; ++k) refine(*m.master, m.boundary_entity, mesh_file);
  m.geometry_order = m.master->GetNodes() ? m.master->GetNodes()->FESpace()->GetMaxElementOrder() : 1;
  // the resolution of the document's selectors on the mesher's table (the info file carries each entity's tangent, but the reader has no mesher input to compare it with,
  // so a file left over from other markers or geometry is not detected here)
  m.resolution = selection::resolve(m.table, boundaries_input(report.model));
  check_groups(m.resolution, m.info, info_file);
  for (int be = 0; be < m.master->GetNBE(); ++be) {
    const int tag = m.boundary_entity[be];
    const auto it = m.resolution.atom_of_tag.find(tag);
    if (it == m.resolution.atom_of_tag.end() || it->second != m.master->GetBdrAttribute(be))
      throw msg::mesh_export_files("the boundary element " + std::to_string(be) + " of the master mesh carries the attribute " +
                                       std::to_string(m.master->GetBdrAttribute(be)) + ", and its entity " + std::to_string(tag) +
                                       (it == m.resolution.atom_of_tag.end() ? std::string(" is no boundary entity")
                                                                             : " has the atom " + std::to_string(it->second)),
                                   mesh_file);
  }
  return m;
}

// ------------------------------------------------------------------------------------------- the instance mesh
InstanceMesh MeshedModel::instance_mesh(const std::vector<std::string>& regions) const {
  if (regions.empty()) throw std::invalid_argument("instance_mesh: no region is given; name at least one enabled region");
  for (const std::string& r : regions)
    if (!region_attribute.count(r)) {
      Strs known;
      for (const auto& [name, attr] : region_attribute) known.push_back(name);
      throw std::invalid_argument("instance_mesh: '" + r + "' is not an enabled region of the model (its regions: " +
                                  msg::quoted(known) + ")");
    }
  if (!master) throw std::invalid_argument("instance_mesh: the meshed model has no master mesh");
  if (static_cast<int>(boundary_entity.size()) != master->GetNBE())
    throw msg::mesh_export_files("the master mesh has " + std::to_string(master->GetNBE()) + " boundary elements and boundary_entity " +
                                     std::to_string(boundary_entity.size()) + " entries",
                                 mesh_file);

  InstanceMesh im;
  // the regions as a set, in the document order (their attributes are 1..R in that order)
  std::set<std::pair<int, std::string>> ordered;
  for (const std::string& r : regions) ordered.insert({region_attribute.at(r), r});
  mfem::Array<int> attrs;
  for (const auto& [attr, name] : ordered) {
    im.regions.push_back(name);
    im.region_attribute[name] = attr;
    attrs.Append(attr);
  }

  const mfem::Array<int> master_face_to_be = master->GetFaceToBdrElMap();
  std::vector<int> vertex_to_master;
  {
    const mfem::SubMesh sub = mfem::SubMesh::CreateFromDomain(*master, attrs);
    const mfem::Array<int>& pe = sub.GetParentElementIDMap();
    const mfem::Array<int>& pf = sub.GetParentEdgeIDMap();   // in 2D the faces are the edges
    const mfem::Array<int>& pv = sub.GetParentVertexIDMap();
    im.element_to_master.assign(pe.begin(), pe.end());
    im.face_to_master.assign(pf.begin(), pf.end());
    vertex_to_master.assign(pv.begin(), pv.end());
    im.mesh = std::make_unique<mfem::Mesh>(sub, true);
  }
  mfem::Mesh& mesh = *im.mesh;
  if (static_cast<int>(im.face_to_master.size()) != mesh.GetNumFaces())
    throw msg::mesh_export_files("the face map of the instance mesh of " + regions_text(im.regions) + " does not cover its faces", mesh_file);
  mesh.RemoveInternalBoundaries();

  // the master number of each boundary element: that of the master's boundary element on its face
  std::vector<int> be_master(mesh.GetNBE());
  std::set<int> present;
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    const int f = mesh.GetBdrElementFaceIndex(be);
    const int pbe = master_face_to_be[im.face_to_master[f]];
    if (pbe < 0) {
      double p[2], q[2];
      face_ends(mesh, f, p, q);
      throw msg::mesh_export_missing_atom(im.regions, p, q);
    }
    be_master[be] = master->GetBdrAttribute(pbe);
    present.insert(be_master[be]);
  }
  // dense numbering: the master numbers present, ascending, numbered 1..n
  std::map<int, int> dense;
  for (int mnum : present) {
    const int a = static_cast<int>(dense.size()) + 1;
    dense[mnum] = a;
    im.master_attribute[a] = mnum;
  }
  for (int be = 0; be < mesh.GetNBE(); ++be) mesh.SetBdrAttribute(be, dense.at(be_master[be]));
  mesh.SetAttributes();
  im.geometry_order = mesh.GetNodes() ? mesh.GetNodes()->FESpace()->GetMaxElementOrder() : 1;

  // the names of the atoms present
  std::map<int, const selection::Atom*> atom_of;
  for (const selection::Atom& a : resolution.atoms) atom_of[a.master] = &a;
  for (const auto& [name, tags] : resolution.selected) {
    im.attributes[name] = {};
    im.interior_faces[name].name = name;
  }
  for (const auto& [a, mnum] : im.master_attribute) {
    const auto it = atom_of.find(mnum);
    if (it == atom_of.end())
      throw msg::mesh_export_files("the master number " + std::to_string(mnum) + " on the boundary of the instance mesh of " +
                                       regions_text(im.regions) + " is the number of no atom of the resolution",
                                   mesh_file);
    const selection::Atom& atom = *it->second;
    Strs names = atom.names;
    std::sort(names.begin(), names.end());
    im.attribute_names[a] = names;
    if (atom.kind == "axis") im.axis_attributes.push_back(a);
    else if (atom.kind == "unnamed") im.remaining_attributes.push_back(a);
    for (const std::string& name : names) im.attributes[name].push_back(a);
  }

  // the sections: the faces interior to the instance that carry a master boundary element of a named entity, signed by
  // the direction of that element times the orientation of (name, entity)
  std::map<int, Strs> names_of_tag;
  for (const auto& [name, tags] : resolution.selected)
    for (int t : tags) names_of_tag[t].push_back(name);
  for (int f = 0; f < mesh.GetNumFaces(); ++f) {
    if (!mesh.FaceIsInterior(f)) continue;
    const int pbe = master_face_to_be[im.face_to_master[f]];
    if (pbe < 0) continue;
    const int tag = boundary_entity[pbe];
    const auto nt = names_of_tag.find(tag);
    if (nt == names_of_tag.end()) continue;
    mfem::Array<int> fv, bv;
    mesh.GetFaceVertices(f, fv);
    master->GetBdrElementVertices(pbe, bv);
    const int a = vertex_to_master[fv[0]], b = vertex_to_master[fv[1]];
    int along = 0;   // +1: the face runs in the direction in which the master's boundary element is written
    if (a == bv[0] && b == bv[1]) along = 1;
    else if (a == bv[1] && b == bv[0]) along = -1;
    else
      throw msg::mesh_export_files("the face " + std::to_string(f) + " of the instance mesh of " + regions_text(im.regions) +
                                       " and the master's boundary element on it have different vertices",
                                   mesh_file);
    for (const std::string& name : nt->second) {
      const auto o = resolution.orientation.find({name, tag});
      if (o == resolution.orientation.end())
        throw msg::mesh_export_files("the resolution gives no orientation to the entity " + std::to_string(tag) + " of '" + name + "'",
                                     mesh_file);
      FaceSet& fs = im.interior_faces[name];
      fs.faces.push_back(f);
      fs.sign.push_back(o->second * along);
    }
  }
  for (auto& [name, fs] : im.interior_faces) {
    fs.length_m = 0.0;
    for (int f : fs.faces) fs.length_m += face_length(mesh, f);
  }
  return im;
}

FaceSet InstanceMesh::section(const std::string& name) const {
  if (!attributes.count(name)) throw std::invalid_argument("section: '" + name + "' is not a boundary of the model");
  const auto it = interior_faces.find(name);
  if (it == interior_faces.end() || it->second.faces.empty())
    throw std::invalid_argument("section: the boundary '" + name + "' has no face interior to the instance mesh of " +
                                regions_text(regions) + " (a section is a boundary with the instance on both sides)");
  return it->second;
}

FacePoints InstanceMesh::face_points(const std::string& name, int order) const {
  const mfem::Mesh& m = mesh_of(*this, "face_points");
  const auto attrs = attributes.find(name);
  if (attrs == attributes.end()) throw std::invalid_argument("face_points: '" + name + "' is not a boundary of the model");
  std::map<int, int> faces;   // face -> sign, ascending
  for (int be = 0; be < m.GetNBE(); ++be)
    if (std::find(attrs->second.begin(), attrs->second.end(), m.GetBdrAttribute(be)) != attrs->second.end())
      faces[m.GetBdrElementFaceIndex(be)] = 1;   // outward
  if (const auto it = interior_faces.find(name); it != interior_faces.end())
    for (size_t i = 0; i < it->second.faces.size(); ++i) faces[it->second.faces[i]] = it->second.sign[i];
  if (faces.empty())
    throw std::invalid_argument("face_points: the boundary '" + name + "' has no face on the instance mesh of " + regions_text(regions));
  return points_on(m, std::vector<std::pair<int, int>>(faces.begin(), faces.end()), order);
}

FacePoints InstanceMesh::face_points(int attribute, int order) const {
  const mfem::Mesh& m = mesh_of(*this, "face_points");
  if (!master_attribute.count(attribute))
    throw std::invalid_argument("face_points: " + std::to_string(attribute) + " is not a boundary attribute of the instance mesh of " +
                                regions_text(regions) + " (its attributes are 1 to " + std::to_string(master_attribute.size()) + ")");
  std::vector<std::pair<int, int>> faces;
  for (int be = 0; be < m.GetNBE(); ++be)
    if (m.GetBdrAttribute(be) == attribute) faces.emplace_back(m.GetBdrElementFaceIndex(be), 1);   // outward
  std::sort(faces.begin(), faces.end());
  return points_on(m, faces, order);
}

std::vector<std::string> InstanceMesh::names_of(int instance_attribute) const {
  if (!master_attribute.count(instance_attribute))
    throw std::invalid_argument("names_of: " + std::to_string(instance_attribute) + " is not a boundary attribute of the instance mesh of " +
                                regions_text(regions) + " (its attributes are 1 to " + std::to_string(master_attribute.size()) + ")");
  const auto it = attribute_names.find(instance_attribute);
  return it == attribute_names.end() ? std::vector<std::string>{} : it->second;
}

// ------------------------------------------------------------------------------------------- the exporters
chem::MeshInput chem_mesh_input(const InstanceMesh& im, int field_order) {
  const mfem::Mesh& m = mesh_of(im, "chem_mesh_input");
  if (field_order != 1 && field_order != 2)
    throw std::invalid_argument("chem_mesh_input: the field order is 1 or 2, not " + std::to_string(field_order));
  if (m.GetNodes() && field_order == 1) throw msg::mesh_export_nodes_for_p1(im.regions, im.geometry_order);
  chem::MeshInput in;
  for (int v = 0; v < m.GetNV(); ++v) {
    in.vertices.push_back(m.GetVertex(v)[0]);
    in.vertices.push_back(m.GetVertex(v)[1]);
  }
  for (int e = 0; e < m.GetNE(); ++e) {
    if (m.GetElementGeometry(e) != mfem::Geometry::TRIANGLE)
      throw std::invalid_argument("chem_mesh_input: element " + std::to_string(e) + " of the instance mesh is not a triangle");
    mfem::Array<int> v;
    m.GetElementVertices(e, v);
    in.triangles.insert(in.triangles.end(), {v[0], v[1], v[2]});
    in.element_attr.push_back(m.GetAttribute(e));
  }
  for (int be = 0; be < m.GetNBE(); ++be) {
    mfem::Array<int> v;
    m.GetBdrElementVertices(be, v);
    in.boundary.insert(in.boundary.end(), {v[0], v[1]});
    in.boundary_attr.push_back(m.GetBdrAttribute(be));
  }
  if (field_order == 2 && m.GetNodes()) {
    // the P2 node of an edge is its point at the parameter 1/2; listed when it is off the chord's midpoint
    mfem::IsoparametricTransformation T;
    mfem::IntegrationPoint mid;
    mid.Set1w(0.5, 1.0);
    mfem::Vector X(2);
    for (int ed = 0; ed < m.GetNEdges(); ++ed) {
      mfem::Array<int> v;
      m.GetEdgeVertices(ed, v);
      const double *p = m.GetVertex(v[0]), *q = m.GetVertex(v[1]);
      m.GetEdgeTransformation(ed, &T);
      T.Transform(mid, X);
      const double off = std::hypot(X(0) - 0.5 * (p[0] + q[0]), X(1) - 0.5 * (p[1] + q[1]));
      if (off <= 1e-12 * std::hypot(q[0] - p[0], q[1] - p[1])) continue;
      in.edge_node_vertices.insert(in.edge_node_vertices.end(), {v[0], v[1]});
      in.edge_node_xy.insert(in.edge_node_xy.end(), {X(0), X(1)});
    }
  }
  return in;
}

void require_straight(const InstanceMesh& im, const std::string& route) {
  const mfem::Mesh& m = mesh_of(im, "require_straight");
  if (!m.GetNodes()) return;
  throw msg::mesh_export_not_straight(route, im.regions, im.geometry_order);
}

}  // namespace axmp::model
