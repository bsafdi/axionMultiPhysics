// axmp: command-line interface to axionMultiPhysics.
//
//   axmp solve design.json [-o DIR] [--order P] [-v N]   mesh + solve, write DIR/results.json
//                                                       (3D designs: srun/mpirun -n N axmp solve ... runs on N MPI ranks)
//     --no-self-force: skip the per-winding vacuum solve that removes the self-force floor
//     --status-file F: write the summary to F instead of DIR/axmp_status.json
//   axmp sensitivities design.json [--param P]... [--objective Q]...   adjoint or finite-difference shape sensitivities
//   axmp validate design.json [--strict] [--mesher M]             check a design, print errors as JSON (a model document: the
//                                                       report of every finding; --strict fails on an unavailable one)
//   axmp mesh design.json [-o DIR] [--mesher M] [-v N]   generate the mesh only (DIR/mesh.msh or mesh3d.msh)
//   axmp resolve design.json                             print the geometry with every expression evaluated
//   axmp explain design.json [--json]                    list the resolved model of a model document (beta)
//   axmp patch design.json ops.json [-o OUT]             apply an RFC 6902 patch, validate the result (all or nothing)
//   axmp schema                                          print the design JSON Schema
//   axmp physics [--json] [--type T] [--routes] [--all]  print the physics catalogue of model documents (beta)
//   axmp requirement [--benchmark NAME]                  print F_req for the built-in benchmarks
//   axmp render scene.json -o out.png [--view section|3d]  render a field image from a scene file
//   axmp templates                                       list parametric templates
//   axmp template NAME [--set key=value ...]             print a design from a template
//
// Errors are reported on stdout as {"ok": false, "error": ..., "where": ..., "hint": ...}
// so that agents can act on them; an error that has a code or details adds "code" and "details" as well.
// `axmp solve` also writes that summary (or the success summary) to
// DIR/axmp_status.json, or to the file named by --status-file, from rank 0: MPI launchers may split or interleave long stdout lines (srun forwards output in
// 1 KiB messages), and Design.solve(mpi_ranks=N) reads the file first.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <unistd.h>

#include "axionmultiphysics/adjoint.hpp"
#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/device.hpp"
#include "axionmultiphysics/model.hpp"
#include "axionmultiphysics/model3d.hpp"
#include "axionmultiphysics/model_commands.hpp"
#include "axionmultiphysics/model_validate.hpp"
#include "axionmultiphysics/physics_registry.hpp"
#include "axionmultiphysics/xsec.hpp"
#include "axionmultiphysics/fw2d.hpp"
#include "axionmultiphysics/render.hpp"
#include "axionmultiphysics/scan.hpp"
#include "axionmultiphysics/spec.hpp"
#include "axionmultiphysics/model_explain.hpp"
#include "axionmultiphysics/spec_model.hpp"
#include "axionmultiphysics/templates.hpp"
#include "../src/error_json.hpp"

using json = nlohmann::json;

namespace {
using axmp::mpi_rank;
// Rank and size of this process as set by the launcher (srun / mpirun), before or without MPI initialization.
int launcher_rank(int* size) {
  auto env_int = [](std::initializer_list<const char*> names, int dflt) {
    for (const char* n : names)
      if (const char* v = std::getenv(n)) return std::atoi(v);
    return dflt;
  };
  if (size) *size = env_int({"OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_STEP_NUM_TASKS"}, 1);
  return env_int({"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"}, 0);
}
// DIR/axmp_status.json of the running `axmp solve` (empty before its options are read)
std::string& status_file() {
  static std::string path;
  return path;
}

// Print the summary on stdout (rank 0; others on stderr) and write it to the status file (rank 0, atomically).
// Text that quotes the input (a parse error shows the bytes it stopped at) may hold invalid UTF-8: dump it with
// replacement characters instead of throwing out of the error handler.
std::string dump_text(const json& j) { return j.dump(2, ' ', false, json::error_handler_t::replace); }

void emit(const json& j, bool root) {
  (root ? std::cout : std::cerr) << dump_text(j) << "\n";
  if (!root || launcher_rank(nullptr) != 0 || status_file().empty()) return;
  try {
    const std::filesystem::path path(status_file());
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    const std::string tmp = status_file() + ".tmp" + std::to_string(::getpid());
    std::ofstream f(tmp);
    f << dump_text(j) << "\n";
    f.close();  // a write error at close (a full quota) shows here, before the rename
    std::error_code ec;
    if (!f) {
      std::filesystem::remove(tmp, ec);
      return;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) std::filesystem::remove(tmp, ec);
  } catch (const std::exception&) {  // the stdout copy remains
  }
}

// `text_out`: the file text, for a caller that scans it later (the operations of a patch); `scan`: scan now when the
// document is a model document (an operations file is scanned by its caller, which knows what it is applied to).
json read_json(const std::string& path, std::string* text_out = nullptr, bool scan = true) {
  std::ifstream f(path);
  if (!f) throw axmp::Error("cannot open '" + path + "'");
  // Read once: the scan below needs the same text, and a pipe cannot be read a second time. A read error throws as
  // json::parse(f) does (it reads the stream buffer the same way).
  const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  json doc;
  try {
    doc = json::parse(text);
  } catch (json::parse_error& e) {
    throw axmp::Error(std::string("invalid JSON in '") + path + "': " + e.what(), "",
                     "Check for trailing commas and unquoted names.");
  }
  if (text_out) *text_out = text;
  if (scan && axmp::spec::is_model_document(doc))  // a key repeated in a model document is refused (JSON keeps the last)
    axmp::spec::scan_duplicate_keys(text, path);
  return doc;
}

int usage() {
  std::cerr << "usage:\n"
               "  axmp solve design.json [-o DIR] [--order P] [-v N] [--scene] [--vtu] [--save] [--no-self-force]\n"
               "                         [--force-samples N] [--status-file F]\n"
               "  axmp sensitivities design.json [--param P]... [--objective Q]... [--method adjoint|fd_morph|fd_remesh]\n"
               "                                 [--step S] [--benchmark NAME] [-o DIR] [--order P]\n"
               "  axmp validate design.json [--strict] [--mesher M]\n"
               "  axmp mesh design.json [-o DIR] [--mesher M] [-v N]\n"
               "  axmp boundaries design.json [--json] [--mesh] [-o DIR] [--mesher M]\n"
               "  axmp resolve design.json\n"
               "  axmp explain design.json [--json]\n"
               "  axmp patch design.json ops.json [-o OUT]\n"
               "  axmp schema\n"
               "  axmp physics [--json] [--type T] [--routes] [--all]\n"
               "  axmp requirement [--benchmark NAME]\n"
               "  axmp render scene.json -o out.png [--view section|3d] [--quantity Q] [--log] [--theme dark]\n"
               "  axmp templates\n"
               "  axmp template NAME [--set key=value ...]\n";
  return 2;
}

// ---- axmp physics: the catalogue as text (the JSON form is axmp physics --json)
std::string joined(const json& a, const char* sep = ", ") {
  std::string s;
  for (const auto& x : a) s += (s.empty() ? "" : sep) + (x.is_string() ? x.get<std::string>() : x.dump());
  return s;
}
std::string padded(const std::string& s, size_t n) {
  return s.size() >= n ? s + " " : s + std::string(n - s.size(), ' ');
}

void print_catalogue(const json& cat, bool all) {
  std::cout << "axionMultiPhysics physics catalogue (" << cat["schema"].get<std::string>()
            << "): everything here is beta\n\nphysics types\n";
  for (const auto& t : cat["types"]) {
    std::cout << "  " << padded(t["name"].get<std::string>(), 36) << padded(t["status"].get<std::string>(), 8)
              << t["title"].get<std::string>() << "\n";
    if (!all) continue;
    if (!t["aliases"].empty()) std::cout << "      aliases: " << joined(t["aliases"]) << "\n";
    if (!t["provides"].empty()) std::cout << "      provides: " << joined(t["provides"]) << "\n";
    for (const auto& o : t["options"]) {
      const std::string dflt = o["default"].get<std::string>();
      std::cout << "      option " << o["name"].get<std::string>() << " ("
                << (o["values"].empty() ? o["type"].get<std::string>() : joined(o["values"], " | "))
                << (dflt.empty() ? std::string() : "; default " + dflt) << ")\n";
    }
    for (const char* which : {"condition_kinds", "source_kinds"})
      for (const auto& k : t[which]) {
        std::vector<std::string> keys;
        for (const auto& d : k["data"])
          keys.push_back(d["name"].get<std::string>() + (d["required"].get<bool>() ? "" : " (optional)"));
        std::cout << "      " << (std::string(which) == "condition_kinds" ? "kind " : "source ")
                  << k["name"].get<std::string>() << (keys.empty() ? std::string() : ": " + joined(json(keys)))
                  << "\n";
      }
    for (const auto& [g, qs] : t["quantities"].items()) {
      std::vector<std::string> names;
      for (const auto& q : qs) names.push_back(q["name"].get<std::string>());
      std::cout << "      quantities (" << g << "): " << joined(json(names)) << "\n";
    }
    for (const auto& n : t["not_implemented"]) std::cout << "      not implemented: " << n.get<std::string>() << "\n";
  }
  std::cout << "\ncoupling terms\n";
  for (const auto& t : cat["terms"])
    std::cout << "  " << padded(t["name"].get<std::string>(), 20) << padded(t["status"].get<std::string>(), 8)
              << joined(t["producers"]) << " -> "
              << joined(t["consumers"]) << " (" << t["support"].get<std::string>() << ")\n";
  if (cat.contains("routes")) {
    std::cout << "\nroutes\n";
    for (const auto& r : cat["routes"]) {
      const std::string geometry = r["geometries"].empty() ? "no geometry" : joined(r["geometries"]);
      std::cout << "  " << padded(r["name"].get<std::string>(), 40)
                << (r["implemented"].get<bool>() ? "implemented: " : "not implemented yet: ") << joined(r["physics"], " + ")
                << (r["couplings"].empty() ? std::string() : " with " + joined(r["couplings"], " + ")) << "; "
                << geometry << "; " << joined(r["study_types"]) << "\n";
      if (!all) continue;
      const json& q = r["requires"];
      std::vector<std::string> needs;
      if (q["mesher"].get<bool>()) needs.push_back("the mesher");
      if (q["mpi"].get<bool>()) needs.push_back("an MPI build");
      if (q["fluids"].get<bool>()) needs.push_back("the fluids library");
      const int ranks = q["max_ranks"].get<int>();
      needs.push_back(ranks == 0 ? std::string("any number of ranks") : "at most " + std::to_string(ranks) + " rank");
      std::cout << "      engine: " << r["engine"].get<std::string>() << "\n";
      std::cout << "      needs: " << joined(json(needs)) << "\n";
      for (const auto& [t, kinds] : r["condition_kinds"].items())
        std::cout << "      " << t << " kinds offered: " << joined(kinds) << "\n";
      for (const auto& [t, kinds] : r["not_offered"].items())
        for (const auto& [k, why] : kinds.items())
          std::cout << "      " << t << " kind " << k << ": " << why.get<std::string>() << "\n";
      for (const auto& l : r["limits"]) std::cout << "      limit: " << l.get<std::string>() << "\n";
    }
  }
  std::cout << "\nreserved instance ids: " << joined(cat["reserved_ids"]) << "\n";
}

// axmp physics [--json] [--type T] [--routes] [--all]; returns the exit status (2: a usage error)
int physics_command(const std::vector<std::string>& a) {
  bool as_json = false, routes = false, all = false;
  std::string type;
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i] == "--json") as_json = true;
    else if (a[i] == "--routes") routes = true;
    else if (a[i] == "--all") all = true;
    else if (a[i] == "--type" && i + 1 < a.size()) type = a[++i];
    else return usage();
  }
  const json cat = axmp::model::catalogue_json(routes, type);
  if (as_json) std::cout << cat.dump(2) << "\n";
  else print_catalogue(cat, all);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> a(argv + 1, argv + argc);
  if (a.empty()) return usage();
  try {
    axmp::configure_device();  // $AXMP_DEVICE: hypre on a GPU in a -DAXMP_GPU build; MFEM always on the CPU
    const std::string cmd = a[0];
    if (cmd == "schema") {
      std::cout << axmp::spec::design_schema().dump(2) << "\n";
      return 0;
    }
    if (cmd == "physics") return physics_command(a);
    if (cmd == "requirement") {
      std::vector<std::string> names = {"optimistic", "conservative", "conservative_solenoid"};
      for (size_t i = 1; i + 1 < a.size(); ++i) if (a[i] == "--benchmark") names = {a[i + 1]};
      json out = json::array();
      axmp::spec::ScanSpec s;
      for (auto& n : names) {
        auto b = axmp::spec::builtin_benchmark(n);
        auto r = axmp::scan::F_required(b, s);
        out.push_back({{"benchmark", n}, {"F_req_SI_T4_H2_m8", r.F_SI}, {"F_req_natural_T4_m10", r.F_nat},
                       {"F_req_eV-2", r.F_nat_eV}, {"g_min_GeV-1", r.g_min_GeV}});
      }
      std::cout << out.dump(2) << "\n";
      return 0;
    }
    if (cmd == "render") {
      // axmp render scene.json -o out.png [--view section|3d] [--quantity Q] [--log] [--theme dark]
      if (a.size() < 2) return usage();
      json scene = read_json(a[1]);
      json opt = json::object();
      std::string out = "render.png";
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) out = a[++i];
        else if (a[i] == "--view" && i + 1 < a.size()) opt["view"] = a[++i];
        else if (a[i] == "--quantity" && i + 1 < a.size()) opt["quantity"] = a[++i];
        else if (a[i] == "--theme" && i + 1 < a.size()) opt["theme"] = a[++i];
        else if (a[i] == "--log") opt["log_scale"] = true;
        else return usage();
      }
      axmp::render::write_png(out, scene, axmp::render::options_from_json(opt));
      std::cout << json{{"ok", true}, {"png", out}}.dump() << "\n";
      return 0;
    }
    if (cmd == "templates") {
      std::cout << axmp::templates::list().dump(2) << "\n";
      return 0;
    }
    if (cmd == "template") {
      if (a.size() < 2) return usage();
      json ov = json::object();
      for (size_t i = 2; i + 1 < a.size(); ++i) {
        if (a[i] != "--set") return usage();
        std::string kv = a[++i];
        auto eq = kv.find('=');
        if (eq == std::string::npos) return usage();
        std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        try {
          size_t pos;
          double x = std::stod(v, &pos);
          if (pos == v.size()) ov[k] = x;
          else ov[k] = v;
        } catch (...) {
          ov[k] = v;
        }
      }
      std::cout << axmp::templates::make(a[1], ov).dump(2) << "\n";
      return 0;
    }
    if (cmd == "explain") {
      // axmp explain [--json] design.json  (the flag before or after the file)
      bool as_json = false;
      std::string file;
      for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] == "--json") as_json = true;
        else if (!a[i].empty() && a[i][0] == '-') return usage();
        else if (file.empty()) file = a[i];
        else return usage();
      }
      if (file.empty()) return usage();
      std::cout << axmp::model::explain_model(read_json(file), as_json);
      if (as_json) std::cout << "\n";
      return 0;
    }
    if (cmd == "patch") {
      // axmp patch design.json ops.json [-o OUT]
      std::vector<std::string> files;
      std::string out;
      for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) out = a[++i];
        else if (!a[i].empty() && a[i][0] == '-') return usage();
        else files.push_back(a[i]);
      }
      if (files.size() != 2) return usage();
      const json input = read_json(files[0]);
      std::string ops_text;
      const json ops = read_json(files[1], &ops_text, false);
      // a key repeated in the operations is refused as in the document (the pointer is into the operations file) when
      // the document, or what the patch makes of it, is a model document
      if (axmp::spec::is_model_document(input)) axmp::spec::scan_duplicate_keys(ops_text, files[1]);
      json patched = axmp::model::patch_document(input, ops);
      if (!axmp::spec::is_model_document(input) && axmp::spec::is_model_document(patched))
        axmp::spec::scan_duplicate_keys(ops_text, files[1]);
      if (out.empty()) {
        std::cout << patched.dump(2) << "\n";
      } else {
        std::ofstream f(out);
        if (!f) throw axmp::Error("cannot write '" + out + "'");
        f << patched.dump(2) << "\n";
        f.close();
        std::cout << json{{"ok", true}, {"name", patched.value("name", std::string())}, {"written", out}}.dump() << "\n";
      }
      return 0;
    }
    if (a.size() < 2) return usage();
    json doc = read_json(a[1]);
    if (cmd == "validate") {
      if (axmp::spec::is_model_document(doc)) {
        bool strict = false;
        axmp::model::ValidateOptions probe;
        for (size_t i = 2; i < a.size(); ++i) {
          if (a[i] == "--strict") strict = true;
          else if (a[i] == "--mesher") {
            if (i + 1 >= a.size()) return usage();
            probe.mesher = a[++i];
          }
        }
        probe.probe_mesher = true;
        const json report = axmp::model::report_json(axmp::model::validate_model(doc, probe), strict);
        std::cout << report.dump(2) << "\n";
        return report["ok"].get<bool>() ? 0 : 1;
      }
      auto d = axmp::spec::parse_design(doc);
      std::cout << json{{"ok", true}, {"name", d.name}, {"regions", d.regions.size()}, {"curves", d.curves.size()},
                        {"windings", d.windings.size()}}
                       .dump(2)
                << "\n";
      return 0;
    }
    if (cmd == "boundaries") {
      axmp::model::CommandOptions o;
      bool as_json = false;
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "--json") as_json = true;
        else if (a[i] == "--mesh") o.mesh = true;
        else if (a[i] == "-o" && i + 1 < a.size()) o.workdir = a[++i];
        else if (a[i] == "--mesher" && i + 1 < a.size()) o.mesher = a[++i];
        else return usage();
      }
      o.document = a[1];
      const axmp::model::BoundariesResult r = axmp::model::boundaries_report(doc, o);
      if (as_json || r.error) std::cout << r.report.dump(2) << "\n";
      else std::cout << axmp::model::boundaries_text(r.report);
      return r.error ? 1 : 0;
    }
    if (cmd == "resolve") {
      if (axmp::spec::is_model_document(doc)) {
        std::cout << axmp::spec::resolved_model_geometry(axmp::spec::parse_model(doc)).dump(2) << "\n";
        return 0;
      }
      std::cout << axmp::spec::resolved_geometry(axmp::spec::parse_design(doc)).dump(2) << "\n";
      return 0;
    }
    if (cmd == "mesh") {
      axmp::SolveOptions o;
      o.workdir = "axmp_mesh";
      o.verbose = 0;
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) o.workdir = a[++i];
        else if (a[i] == "--mesher" && i + 1 < a.size()) o.mesher = a[++i];
        else if (a[i] == "-v" && i + 1 < a.size()) o.verbose = std::stoi(a[++i]);
        else return usage();
      }
      axmp::log_level() = o.verbose;
      if (axmp::spec::is_model_document(doc)) {
        axmp::model::CommandOptions co;
        co.workdir = o.workdir;
        co.mesher = o.mesher;
        co.verbose = o.verbose;
        co.document = a[1];
        std::cout << axmp::model::mesh_command(doc, co).dump(2) << "\n";
        return 0;
      }
      json out = axmp::mesh_any(doc, o);
      if (mpi_rank() == 0) std::cout << out.dump(2) << "\n";
      return 0;
    }
    if (cmd == "solve") {
      axmp::SolveOptions o;
      o.workdir = "axmp_run";
      bool want_scene = false, want_vtu = false, want_save = false;
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) o.workdir = a[++i];
        else if (a[i] == "--order" && i + 1 < a.size()) o.order = std::stoi(a[++i]);
        else if (a[i] == "-v" && i + 1 < a.size()) o.verbose = std::stoi(a[++i]);
        else if (a[i] == "--mesher" && i + 1 < a.size()) o.mesher = a[++i];
        else if (a[i] == "--scene") want_scene = true;
        else if (a[i] == "--vtu") want_vtu = true;
        else if (a[i] == "--save") want_save = true;
        else if (a[i] == "--no-self-force") o.winding_self_force = false;
        else if (a[i] == "--force-samples" && i + 1 < a.size()) o.force_samples = std::stoi(a[++i]);
        else if (a[i] == "--status-file" && i + 1 < a.size()) status_file() = a[++i];
        else return usage();
      }
      axmp::log_level() = o.verbose;
      if (status_file().empty()) status_file() = o.workdir + "/axmp_status.json";
      axmp::spec::refuse_unserved_model_document(doc, "solve");
      if (doc.value("symmetry", std::string("axisymmetric")) != "3d" && doc.value("symmetry", std::string()) != "xsec" &&
          doc.contains("mesh") && doc["mesh"].contains("file")) {  // 2D full wave on an imported mesh
        if (launcher_rank(nullptr) > 0) return 0;
        axmp::Fw2DModel model(axmp::spec::parse_design(doc));
        model.run(o);
        const json& res = model.results();
        emit(json{{"ok", true}, {"results", o.workdir + "/results.json"}, {"warnings", res["warnings"]}}, true);
        return 0;
      }
      if (doc.value("symmetry", std::string("axisymmetric")) == "xsec") {  // z-uniform cavity cross-section
        if (launcher_rank(nullptr) > 0) return 0;
        axmp::XsecModel model(axmp::spec::parse_design(doc));
        model.run(o);
        const json& res = model.results();  // also written to workdir/results.json
        emit(json{{"ok", true}, {"results", o.workdir + "/results.json"}, {"warnings", res["warnings"]}}, true);
        return 0;
      }
      if (doc.value("symmetry", std::string("axisymmetric")) == "3d") {
        // MPI: every rank solves its part (srun/mpirun -n N axmp solve ...); rank 0 writes the outputs
        axmp::Model3D model(axmp::spec::parse_design(doc));
        model.run(o);
        const json& res = model.results();
        const bool root = mpi_rank() == 0;
        if (root) std::ofstream(o.workdir + "/results.json") << res.dump(2);
        if (want_scene) {
          json sc = model.scene();  // gathered fields: identical on every rank
          if (root) std::ofstream(o.workdir + "/scene.json") << sc.dump();
        }
        if (want_vtu) model.write_paraview(o.workdir + "/paraview");  // collective, one piece per rank
        if (want_save) model.save(o.workdir + "/fields");            // rank 0 writes the gathered fields
        if (root)
          emit(json{{"ok", true}, {"results", o.workdir + "/results.json"}, {"coupling", res["coupling"]},
                    {"benchmarks", res["benchmarks"]}, {"warnings", res["warnings"]}}, true);
        return 0;
      }
      // axisymmetric designs run on one process: under a multi-rank launch only rank 0 works (the others would
      // write the same files)
      int nlaunch = 1;
      if (const int r = launcher_rank(&nlaunch); r > 0) {
        std::cerr << json{{"ok", true}, {"idle_rank", r},
                          {"note", "axisymmetric designs run on rank 0 only; this rank did nothing"}}.dump() << "\n";
        return 0;
      }
      axmp::AxiModel model(axmp::spec::parse_design(doc));
      model.run(o);
      json res = model.results();
      if (nlaunch > 1)
        res["warnings"].push_back("axisymmetric designs run on a single process: " + std::to_string(nlaunch - 1) +
                                  " of the launched ranks stayed idle");
      std::ofstream(o.workdir + "/results.json") << res.dump(2);
      if (want_scene) std::ofstream(o.workdir + "/scene.json") << model.scene().dump();
      if (want_vtu) model.write_vtu(o.workdir + "/fields.vtu");
      json summary = {{"ok", true}, {"results", o.workdir + "/results.json"}, {"coupling", res["coupling"]},
                      {"benchmarks", res["benchmarks"]}, {"warnings", res["warnings"]}};
      emit(summary, true);
      return 0;
    }
    if (cmd == "sensitivities") {
      axmp::sens::Options so;
      so.solve.workdir = "axmp_run";
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) so.solve.workdir = a[++i];
        else if (a[i] == "--order" && i + 1 < a.size()) so.solve.order = std::stoi(a[++i]);
        else if (a[i] == "--param" && i + 1 < a.size()) so.params.push_back(a[++i]);
        else if (a[i] == "--objective" && i + 1 < a.size()) so.objectives.push_back(a[++i]);
        else if (a[i] == "--method" && i + 1 < a.size()) so.method = a[++i];
        else if (a[i] == "--step" && i + 1 < a.size()) so.step = std::stod(a[++i]);
        else if (a[i] == "--benchmark" && i + 1 < a.size()) so.benchmark = a[++i];
        else if (a[i] == "--mesher" && i + 1 < a.size()) so.solve.mesher = a[++i];
        else if (a[i] == "-v" && i + 1 < a.size()) so.solve.verbose = std::stoi(a[++i]);
        else return usage();
      }
      axmp::log_level() = so.solve.verbose;
      axmp::spec::require_em_document(doc, "sensitivities");
      AXMP_REQUIRE(doc.value("symmetry", std::string("axisymmetric")) != "3d",
                  "sensitivities are implemented for axisymmetric designs", "/symmetry",
                  "Differentiate the axisymmetric model and cross-check the optimum in 3D.");
      axmp::AxiModel model(axmp::spec::parse_design(doc));
      axmp::SolveOptions o = so.solve;
      model.run(o);
      json out = axmp::sens::sensitivities(model, so);
      std::ofstream(o.workdir + "/sensitivities.json") << out.dump(2);
      out["file"] = o.workdir + "/sensitivities.json";
      out["diagnostics"].erase("morph");
      std::cout << out.dump(2) << "\n";
      return 0;
    }
    return usage();
  } catch (axmp::Error& e) {
    // MPI: the JSON error goes to stdout from rank 0 only (other ranks report on stderr)
    emit(axmp::error_json(e), mpi_rank() == 0);
    return 1;
  } catch (std::exception& e) {
    emit(json{{"ok", false}, {"error", e.what()}}, mpi_rank() == 0);
    return 1;
  }
}
