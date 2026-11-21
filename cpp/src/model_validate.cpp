// Validation of a model document: see axionmultiphysics/model_validate.hpp.
//
// THE ORDER OF THE CHECKS.  The grammar first (parse_model_collect).  Then, for what the grammar left usable:
//   1. the options without a default that an instance must write;
//   2. per study step, the groups of instances that the coupling terms connect, and the route row of each group
//      (match_route); the regions of every instance (its "domains", else the default of design note 3.6 on its route);
//   3. the beta warnings (one per type, per term, per route);
//   4. per step: the coupling graph, the checks of each group on its row, the strategy, the continuation, the laws
//      whose field nothing provides, the terms that act nowhere, the uncoupled pairs;
//   5. per document: two instances of one family on a shared region, an "in" that reaches outside what it restricts,
//      the checks that wait for the resolved boundaries (reported as deferred), and the geometry order.
// A finding is reported once, whatever the number of steps that meet it.
//
// WHAT A GRAMMAR FINDING SKIPS.  An instance with a finding under /physics/<id>, or at an end of a coupling entry with
// a finding, takes part in no route-level check, and neither does any instance of its group.  The ends of such an entry
// are read from the document as written ("from" and "to"; a name that is no instance stands for the instances whose id
// is near it); when an end cannot be read, no instance is checked.  A region whose material has a finding under
// /materials/<name> takes part in no material check.  A finding in the regions, curves or domains skips every
// route-level check (the geometry is not known), one in the mesh the geometry-order rule.
//
// WHAT IS DEFERRED.  Validation reads the structure of the document, not the geometry: the resolution of the named
// boundaries, the partition rule of each instance (design note 5.5), a force output on a boundary that is not a key of
// its instance's conditions, and the checks of an imported mesh are reported as deferred findings, with the code of
// the check that runs later.
//
// CURVED BOUNDARIES are decided before any mesh exists, by a conservative syntactic test: the model has curved
// boundaries when a shape is a disk, has an elliptic arc or a rounded corner, when a curve is an arc, or when a solid
// is anything but a box or an extrusion; a named boundary is then curved unless its selector fixes a coordinate
// (`where` with one value).
#include "axionmultiphysics/model_validate.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <set>

#include "axionmultiphysics/model_mesh.hpp"
#include "error_json.hpp"
#include "model_errors.hpp"
#include "model_reads.hpp"

namespace axmp::model {

namespace {

using Strs = std::vector<std::string>;
namespace E = msg;
using detail::contains;
using detail::intersection;
using Kind = Finding::Kind;

constexpr const char* kFluid = "fluid.incompressible";
constexpr const char* kHeat = "heat.transfer";
constexpr const char* kCurrentDistribution = "electrochem.current_distribution";

bool under(const std::string& where, const std::string& prefix) {
  return where.compare(0, prefix.size(), prefix) == 0 && (where.size() == prefix.size() || where[prefix.size()] == '/');
}
// The segment of a JSON pointer after `prefix` ("/physics/flow/options" after "/physics/" is "flow").
std::string segment_after(const std::string& where, const std::string& prefix) {
  const size_t end = where.find('/', prefix.size());
  return where.substr(prefix.size(), end == std::string::npos ? std::string::npos : end - prefix.size());
}

std::string option_text(const json& v) {
  if (v.is_string()) return v.get<std::string>();
  if (v.is_number_integer()) return std::to_string(v.get<long long>());
  if (v.is_number()) return E::num_text(v.get<double>());
  return v.dump();
}
std::string option_of(const Instance& inst, const std::string& name) {
  const auto it = inst.options.find(name);
  return it == inst.options.end() ? std::string() : option_text(it->second.value);
}

// ------------------------------------------------------------------------------------------------ geometry
bool curved_shape(const spec::Shape& s) {
  using K = spec::Shape::Kind;
  if (s.kind == K::Disk) return true;
  if (s.fillet > 0.0) return true;
  for (double f : s.fillets)
    if (f > 0.0) return true;
  for (const auto& seg : s.segments)
    if (seg.ellipse) return true;
  for (const auto& c : s.children)
    if (curved_shape(c)) return true;
  return false;
}
bool curved_solid(const spec::Solid& s) {
  using K = spec::Solid::Kind;
  if (s.kind == K::Box || s.kind == K::Extrude || s.kind == K::Imported) return false;
  if (s.kind == K::Union || s.kind == K::Difference || s.kind == K::Intersection) {
    for (const auto& c : s.children)
      if (curved_solid(c)) return true;
    return false;
  }
  return true;
}
bool curved_model(const ResolvedModel& m) {
  const bool is3d = m.symmetry == "3d";
  for (const auto& r : m.regions)
    if (is3d ? curved_solid(r.solid) : curved_shape(r.shape)) return true;
  for (const auto& c : m.curves)
    if (c.path.kind == spec::Path::Kind::Arc || c.path.fillet > 0.0) return true;
  return false;
}
// A named boundary of a model with curved boundaries is curved unless its selector fixes a coordinate.
bool curved_boundary(const ResolvedModel& m, const std::string& name) {
  const auto it = m.boundaries.find(name);
  if (it == m.boundaries.end()) return true;
  for (const auto& w : it->second.where)
    if (w.second.size() == 1) return false;
  return true;
}

// ------------------------------------------------------------------------------------------------ materials
bool has_block(const MaterialProps& mp, const std::string& name) {
  if (name == "fluid") return mp.has_fluid;
  if (name == "electrolyte") return mp.has_electrolyte;
  if (name == "electric") return mp.has_electric;
  if (name == "thermal") return mp.k_W_mK.given;
  if (name == "mechanical") return mp.E_Pa.given;
  if (name == "species") return mp.has_species;
  const detail::PropertyRef* p = detail::property(name);
  return p ? (mp.*(p->member)).given : true;
}
// The block every route of a type needs, for an instance that takes no row.
Strs type_blocks(const std::string& type) {
  if (type == kFluid) return {"fluid"};
  if (type == kHeat) return {"thermal"};
  if (type == kCurrentDistribution) return {"electrolyte"};
  if (type == "current.conduction") return {"electric"};
  if (type == "solid.linear") return {"mechanical"};
  if (type == "species.transport") return {"species"};
  return {};
}
constexpr const char* kBeta = "fluid.beta_1_K";

// ------------------------------------------------------------------------------------------------ the validation
struct Group {
  Strs ids;                                   // in the order of the step's physics list
  bool bad = false;                           // an instance of it has a grammar finding
  const Route* row = nullptr;
  std::map<std::string, const Route*> own;    // per instance: the row whose cells are its own (a flow instance of a
                                              // coupled row: the Stokes or Navier-Stokes row its options select)
  Strs terms;                                 // the terms between its members in the step, sorted, each once
  Strs other_studies;                         // the study types for which the combination has a row, when this one has none
};

bool geometry_free_type(const Registry& reg, const std::string& type) {
  bool any = false;
  for (const Route& r : reg.routes)
    if (contains(r.physics, type)) {
      if (!r.geometries.empty()) return false;
      any = true;
    }
  return any;
}

class Validation {
 public:
  Validation(const json& doc, ValidationReport& rep, Findings& out, const ValidateOptions& options)
      : doc_(doc), rep_(rep), m_(rep.model), out_(out), R_(registry()), options_(options) {}
  void run();

 private:
  const json& doc_;                           // the document as written
  ValidationReport& rep_;
  const ResolvedModel& m_;
  Findings& out_;
  const Registry& R_;
  const ValidateOptions options_;
  std::set<std::string> bad_instances_, bad_materials_, said_;
  bool mesh_bad_ = false;
  std::vector<std::vector<Group>> groups_;    // per step
  std::map<std::string, std::pair<size_t, size_t>> home_;   // per instance: the first (step, group) that solves it

  void add(Kind kind, const Error& e) {
    if (said_.insert(std::string(1, static_cast<char>('0' + static_cast<int>(kind))) + e.code() + "|" + e.where() + "|" + e.message()).second)
      out_.add(kind, e);
  }
  void error(const Error& e) { add(Kind::error, e); }

  const Instance& inst(const std::string& id) const { return m_.instances.at(id); }
  const Strs& regions(const std::string& id) const {
    static const Strs none;
    const auto it = rep_.regions.find(id);
    return it == rep_.regions.end() ? none : it->second;
  }
  const MaterialProps* material_of(const std::string& region, std::string* name = nullptr) const {
    for (const auto& r : m_.regions)
      if (r.name == region) {
        if (name) *name = r.material;
        const auto it = m_.materials.find(r.material);
        return it == m_.materials.end() || bad_materials_.count(r.material) ? nullptr : &it->second;
      }
    return nullptr;
  }
  bool geometry_free(const std::string& type) const { return geometry_free_type(R_, type); }
  std::string step_pointer(const StudyStep& s) const { return s.index < 0 ? std::string("/study") : "/study/" + std::to_string(s.index); }

  // An instance that some step solves in a group without a grammar finding.
  bool usable(const std::string& id) const {
    const auto h = home_.find(id);
    return h != home_.end() && !groups_[h->second.first][h->second.second].bad;
  }

  bool scan_grammar_findings();
  void mark_coupling_ends();
  void required_options();
  void make_groups();
  void match_route(const StudyStep& step, Group& g) const;
  const Route* own_row(const StudyStep& step, const Instance& in) const;
  void resolve_regions();
  Strs needs(const Group* g, const Instance& in) const;
  void beta_warnings();
  void report_route(const StudyStep& step, const Group& g);
  void check_group(const StudyStep& step, const Group& g);
  void check_materials(const Group& g, const Instance& in);
  void check_conditions(const Group& g, const Instance& in);
  void check_options(const Group& g, const Instance& in);
  void check_uniform(const StudyStep& step, const Group& g, const Instance& in);
  void check_one_expression(const StudyStep& step, const Group& g, const Instance& in);
  void check_laws(const StudyStep& step, const Group& g, const Instance& in);
  void check_flow_outputs_and_warnings(const Group& g, const Instance& in);
  bool differing(const Instance& in, const detail::PropertyRef& p, E::TwoValues& tv, json& values) const;
  bool force_output_of(const Instance& in, const Output& o) const;
  Strs force_outputs(const Instance& in, bool on_condition_only) const;
  void check_time(const StudyStep& step);
  void check_strategy(const StudyStep& step, const std::vector<Group>& groups, StepReport& sr);
  void check_continuation(const StudyStep& step, const std::vector<Group>& groups);
  void check_law_fields(const StudyStep& step);
  void check_graph(const StudyStep& step, const std::vector<Group>& groups, const CouplingGraph& graph);
  void check_overlap();
  void check_selections();
  void check_mesher();
  void check_shapes();
  void check_between_overlap();
  void between_overlap_of(const std::string& name, const Selector& s, const std::string& pointer);
  void check_force_owners();
  void check_deferred();
  void check_geometry_order();
};

// The grammar findings decide what the route-level checks may look at.  False when nothing can be checked.
bool Validation::scan_grammar_findings() {
  for (const Finding& f : out_.all()) {
    if (f.kind != Kind::error) continue;
    const std::string& w = f.error.where();
    if (under(w, "/regions") || under(w, "/curves") || under(w, "/domains") || under(w, "/symmetry") || under(w, "/parameters")) return false;
    if (w == "/physics" || w == "/couplings" || w == "/study") return false;
    if (under(w, "/mesh")) mesh_bad_ = true;
    if (w.rfind("/physics/", 0) == 0) bad_instances_.insert(segment_after(w, "/physics/"));
    if (w.rfind("/materials/", 0) == 0) bad_materials_.insert(segment_after(w, "/materials/"));
  }
  return true;
}

// An entry of "couplings" that the grammar dropped leaves its two instances without the term, so they are not checked.
// The ends are read from the document as written.
void Validation::mark_coupling_ends() {
  Strs ids;
  for (const auto& [id, in] : m_.instances) ids.push_back(id);
  const auto all = [&] { bad_instances_.insert(ids.begin(), ids.end()); };
  for (const Finding& f : out_.all()) {
    if (f.kind != Kind::error || f.error.where().rfind("/couplings/", 0) != 0) continue;
    const std::string position = segment_after(f.error.where(), "/couplings/");
    const json* entry = nullptr;
    if (!position.empty() && position.size() < 9 && position.find_first_not_of("0123456789") == std::string::npos && doc_.is_object()) {
      const auto list = doc_.find("couplings");
      const size_t k = std::stoul(position);
      if (list != doc_.end() && list->is_array() && k < list->size() && (*list)[k].is_object()) entry = &(*list)[k];
    }
    if (!entry) {
      all();
      continue;
    }
    for (const char* end : {"from", "to"}) {
      const auto written = entry->find(end);
      if (written == entry->end() || !written->is_string()) {
        all();
        continue;
      }
      const std::string name = written->get<std::string>();
      if (m_.instances.count(name)) {
        bad_instances_.insert(name);
        continue;
      }
      // a misspelt end may be any instance within two edits that the term can have at that end: an instance of another
      // type cannot be the one that was meant, so its checks are kept (all the near ones when the term is not known)
      const auto term = entry->find("term");
      const CouplingTermType* tt = term != entry->end() && term->is_string() ? R_.term(term->get<std::string>()) : nullptr;
      for (const std::string& near : E::near_names(name, ids)) {
        if (tt && !type_matches(std::string(end) == "from" ? tt->producers : tt->consumers, inst(near).type)) continue;
        bad_instances_.insert(near);
      }
    }
  }
}

// An option without a default has to be written: the route cannot be chosen without it.
void Validation::required_options() {
  for (const auto& [id, in] : m_.instances) {
    if (bad_instances_.count(id)) continue;
    const PhysicsType* ty = R_.type(in.type);
    if (!ty) {
      bad_instances_.insert(id);
      continue;
    }
    for (const OptionSpec& o : ty->options) {
      if (!o.default_value.empty() || o.type != "enum" || in.options.count(o.name)) continue;
      error(Error("missing required field '" + o.name + "'", "/physics/" + id + "/options",
                  "The option '" + o.name + "' of '" + in.type + "' has no default: write one of " + E::join(o.values) + "."));
      bad_instances_.insert(id);
    }
  }
}

// The groups of each step: the instances that its coupling terms connect.
void Validation::make_groups() {
  for (size_t s = 0; s < m_.study.size(); ++s) {
    const StudyStep& step = m_.study[s];
    Strs nodes;
    for (const std::string& id : step.physics)
      if (!contains(nodes, id) && m_.instances.count(id)) nodes.push_back(id);
    std::map<std::string, std::string> parent;
    for (const std::string& id : nodes) parent[id] = id;
    const std::function<std::string(const std::string&)> find = [&](const std::string& x) {
      return parent[x] == x ? x : parent[x] = find(parent[x]);
    };
    for (const Coupling& c : m_.couplings)
      if (parent.count(c.from) && parent.count(c.to)) parent[find(c.from)] = find(c.to);
    std::vector<Group> groups;
    for (const std::string& id : nodes) {
      const std::string root = find(id);
      Group* g = nullptr;
      for (Group& x : groups)
        if (find(x.ids[0]) == root) g = &x;
      if (!g) {
        groups.emplace_back();
        g = &groups.back();
      }
      g->ids.push_back(id);
      if (bad_instances_.count(id)) g->bad = true;
    }
    for (size_t k = 0; k < groups.size(); ++k) {
      Group& g = groups[k];
      for (const Coupling& c : m_.couplings)
        if (contains(g.ids, c.from) && contains(g.ids, c.to) && !contains(g.terms, c.term)) g.terms.push_back(c.term);
      std::sort(g.terms.begin(), g.terms.end());
      if (!g.bad) match_route(step, g);
      for (const std::string& id : g.ids)
        if (!home_.count(id)) home_[id] = {s, k};
    }
    groups_.push_back(std::move(groups));
  }
}

// The row of a group: the instance types, the terms between them, the geometry class, the study type and the option
// values that select a row (`when`); among the rows that agree on those, the one whose selecting sources
// (`when_sources`) the instances have.
void Validation::match_route(const StudyStep& step, Group& g) const {
  Strs types;
  for (const std::string& id : g.ids) types.push_back(inst(id).type);
  std::sort(types.begin(), types.end());
  // the source kinds that select a row, per type
  std::map<std::string, Strs> selecting;
  for (const Route& r : R_.routes)
    for (const auto& [t, kinds] : r.when_sources)
      for (const std::string& k : kinds)
        if (!contains(selecting[t], k)) selecting[t].push_back(k);
  std::vector<const Route*> candidates;
  for (const Route& r : R_.routes) {
    Strs row_types = r.physics, row_terms = r.couplings;
    std::sort(row_types.begin(), row_types.end());
    std::sort(row_terms.begin(), row_terms.end());
    if (row_types != types || row_terms != g.terms) continue;
    if (!r.geometries.empty() && !contains(r.geometries, m_.symmetry)) continue;
    bool selected = true;
    for (const auto& [option, value] : r.when) {
      bool found = false;
      for (const std::string& id : g.ids) {
        const auto o = inst(id).options.find(option);
        if (o == inst(id).options.end()) continue;
        found = true;
        selected = selected && option_text(o->second.value) == value;
      }
      selected = selected && found;
    }
    if (!selected) continue;
    if (!contains(r.study_types, step.type)) {
      for (const std::string& st : r.study_types)
        if (!contains(g.other_studies, st)) g.other_studies.push_back(st);
      continue;
    }
    candidates.push_back(&r);
  }
  const auto has_sources = [&](const Route& r) {
    for (const std::string& id : g.ids) {
      const Instance& in = inst(id);
      const auto listed = r.when_sources.find(in.type);
      const Strs& want = listed == r.when_sources.end() ? Strs() : listed->second;
      for (const std::string& k : selecting[in.type])
        if (detail::has_source(in, k) != contains(want, k)) return false;
    }
    return true;
  };
  for (const Route* r : candidates)
    if (!g.row && has_sources(*r)) g.row = r;
  // a selecting source on a combination that has no row for it: the row without the source, which then refuses it
  for (const Route* r : candidates)
    if (!g.row && r->when_sources.empty()) g.row = r;
  if (!g.row) return;
  g.other_studies.clear();
  for (const std::string& id : g.ids) {
    const Instance& in = inst(id);
    g.own[id] = g.row;
    if (g.row->physics.size() > 1 && !contains(g.row->cell_types, in.type))
      if (const Route* own = own_row(step, in)) g.own[id] = own;
  }
}

// The single-type row that an instance of a coupled row is checked against: the row of its type alone in this
// geometry and study type that its options select.
const Route* Validation::own_row(const StudyStep& step, const Instance& in) const {
  for (const Route& r : R_.routes) {
    if (r.physics != Strs{in.type} || !r.couplings.empty()) continue;
    if (!r.geometries.empty() && !contains(r.geometries, m_.symmetry)) continue;
    if (!contains(r.study_types, step.type)) continue;
    bool selected = true;
    for (const auto& [option, value] : r.when) selected = selected && option_of(in, option) == value;
    if (selected) return &r;
  }
  return nullptr;
}

// What the material of a region of the instance must carry: the blocks of its row and of its own row.
Strs Validation::needs(const Group* g, const Instance& in) const {
  if (!g || !g->row) return type_blocks(in.type);
  Strs out;
  for (const Route* r : {g->row, g->own.at(in.id)}) {
    const auto it = r->material_blocks.find(in.type);
    if (it == r->material_blocks.end()) continue;
    for (const std::string& b : it->second)
      if (!contains(out, b)) out.push_back(b);
  }
  return out;
}

// The regions of every instance: its "domains", else every enabled region whose material carries what its route needs
// (design note 3.6).  The expansion coefficient of the buoyancy term is not part of that default: a fluid without it is
// reported by name instead of being left out silently.
void Validation::resolve_regions() {
  for (const auto& [id, in] : m_.instances) {
    Strs& out = rep_.regions[id];
    if (geometry_free(in.type)) continue;
    if (in.domains_written) {
      out = detail::regions_of(m_, in.domains);
      continue;
    }
    const auto h = home_.find(id);
    const Group* g = h == home_.end() ? nullptr : &groups_[h->second.first][h->second.second];
    const bool usable = g && !g->bad;
    Strs need;
    for (const std::string& b : needs(usable ? g : nullptr, in))
      if (b != kBeta) need.push_back(b);
    for (const auto& r : m_.regions) {
      const auto mat = m_.materials.find(r.material);
      bool carries = mat != m_.materials.end();
      if (carries && !bad_materials_.count(r.material))
        for (const std::string& b : need) carries = carries && has_block(mat->second, b);
      if (carries) out.push_back(r.name);
    }
    if (out.empty() && usable && !m_.regions.empty() && m_.mesh.file.empty()) error(E::no_region_for_physics(id, in.type, need));
  }
}

// One warning per beta type, then per beta term, then per route (design note 17).
void Validation::beta_warnings() {
  Strs seen;
  for (const auto& [id, in] : m_.instances) {
    const PhysicsType* ty = R_.type(in.type);
    if (bad_instances_.count(id) || !ty || ty->status != "beta" || contains(seen, in.type)) continue;
    seen.push_back(in.type);
    add(Kind::warning, E::beta_feature("physics_type", in.type, "/physics/" + id + "/type", ty->validation));
  }
  seen.clear();
  for (const Coupling& c : m_.couplings) {
    const CouplingTermType* tt = R_.term(c.term);
    if (!tt || tt->status != "beta" || contains(seen, c.term)) continue;
    seen.push_back(c.term);
    add(Kind::warning, E::beta_feature("coupling_term", c.term, "/couplings/" + std::to_string(c.index) + "/term", ""));
  }
  seen.clear();
  for (const auto& groups : groups_)
    for (const Group& g : groups) {
      if (g.bad || !g.row || contains(seen, g.row->name)) continue;
      seen.push_back(g.row->name);
      add(Kind::warning, E::beta_feature("route", g.row->name, "/physics", ""));
    }
}

// The finding that says what the group can be solved with: a row that is not implemented yet, a stationary step over
// a combination that is solved in time only, or the inventory of what is missing (design note 4.4).
void Validation::report_route(const StudyStep& step, const Group& g) {
  if (g.row) {
    if (g.row->implemented) return;
    std::vector<std::pair<std::string, std::string>> members;
    for (const std::string& id : g.ids) members.emplace_back(id, inst(id).type);
    const Requirements& q = g.row->needs;
    json missing = json::array();
#ifdef AXMP_NO_MPI
    if (q.mpi) missing.push_back("mpi");
    if (q.fluids) missing.push_back("fluids");
#endif
    add(Kind::unavailable,
        E::route_not_implemented(members, g.row->name, g.row->engine,
                                 {{"mesher", q.mesher}, {"mpi", q.mpi}, {"fluids", q.fluids}, {"engine_3d", q.engine_3d}, {"max_ranks", q.max_ranks}},
                                 missing));
    return;
  }
  // no stationary solve at all (a reactor): the document has to say the transient step, which has no default
  if (!g.other_studies.empty() && step.type == "stationary") {
    const Instance& in = inst(g.ids[0]);
    error(E::study_type_not_offered(in.id, in.type, step.type, g.other_studies, step.index < 0 ? "/study" : step_pointer(step) + "/type",
                                    step.index >= 0));
    return;
  }
  // the inventory: absent types named as types, then absent terms, then what else is missing, by type
  Strs types, absent_types, rest, absent_terms;
  for (const std::string& id : g.ids)
    if (!contains(types, inst(id).type)) types.push_back(inst(id).type);
  std::sort(types.begin(), types.end());
  const std::string geometry = m_.symmetry;
  bool with_geometry = false;
  for (const std::string& t : types) {
    const PhysicsType* ty = R_.type(t);
    if (!geometry_free(t)) with_geometry = true;
    if (ty && ty->status == "absent") {
      if (ty->absent_phrase == "physics type '" + t + "'") absent_types.push_back(t);
      else rest.push_back(ty->absent_phrase);
      continue;
    }
    bool on_geometry = false, any = false;
    for (const Route& r : R_.routes)
      if (contains(r.physics, t)) {
        any = true;
        on_geometry = on_geometry || r.geometries.empty() || contains(r.geometries, geometry);
      }
    if (!any) rest.push_back("a route for '" + t + "'");
    else if (!on_geometry) rest.push_back("'" + t + "' " + E::on_geometry(geometry));
  }
  for (const std::string& t : g.terms) {
    const CouplingTermType* tt = R_.term(t);
    if (tt && tt->status == "absent") absent_terms.push_back(t);
  }
  Strs missing;
  if (!absent_types.empty()) missing.push_back(std::string("physics type") + (absent_types.size() > 1 ? "s " : " ") + E::quoted_and(absent_types));
  if (!absent_terms.empty()) missing.push_back(std::string("coupling term") + (absent_terms.size() > 1 ? "s " : " ") + E::quoted_and(absent_terms));
  missing.insert(missing.end(), rest.begin(), rest.end());
  if (!g.other_studies.empty()) missing.push_back("a " + step.type + " solve of this combination");
  if (missing.empty()) missing.push_back("this combination of physics, coupling terms and options");
  add(Kind::unavailable, E::route_inventory(types, g.terms, with_geometry ? geometry : std::string(), step.type, missing));
}

// ------------------------------------------------------------------------------------------------ a group on its row
void Validation::check_group(const StudyStep& step, const Group& g) {
  const Route& row = *g.row;
  for (const std::string& id : g.ids) {
    const Instance& in = inst(id);
    check_materials(g, in);
    check_conditions(g, in);
    check_options(g, in);
    check_uniform(step, g, in);
    check_one_expression(step, g, in);
    check_laws(step, g, in);
    if (in.type == kFluid) check_flow_outputs_and_warnings(g, in);
  }
  // one mesh for every instance of the row
  if (row.equal_domains)
    for (size_t i = 1; i < g.ids.size(); ++i) {
      Strs a = regions(g.ids[0]), b = regions(g.ids[i]);
      Strs sa = a, sb = b;
      std::sort(sa.begin(), sa.end());
      std::sort(sb.begin(), sb.end());
      if (sa != sb) error(E::domains_must_coincide(g.ids[0], g.ids[i], row.name, a, b));
    }
  for (const Coupling& c : m_.couplings) {
    if (!contains(g.ids, c.from) || !contains(g.ids, c.to)) continue;
    const Strs shared = intersection(regions(c.from), regions(c.to));
    // a term restricted to part of the shared domain, on a row that solves the two instances on one mesh
    // (an "in" that also reaches outside the shared domain has a second finding for that)
    if (row.equal_domains && !c.in.empty()) {
      const Strs acts = intersection(shared, detail::regions_of(m_, {c.in}));
      if (!acts.empty() && acts.size() < shared.size()) error(E::restriction_not_offered(c.index, c.term, c.from, c.to, row.name, c.in, acts, shared));
    }
    // gravity along the vertical axis, either way
    if (c.term == "buoyancy") {
      const auto d = c.data.find("gravity_m_s2");
      if (d != c.data.end() && !d->second.values.empty() && d->second.values[0] != 0.0)
        error(E::gravity_direction(c.index, d->second.written, row.name, m_.symmetry == "axisymmetric" ? "z" : "y"));
    }
    // a prescribed velocity and a solved one on the same instance
    if (c.term == "advection") {
      const Instance& to = inst(c.to);
      for (size_t i = 0; i < to.sources.size(); ++i)
        if (to.sources[i].kind == "prescribed_advection") {
          error(E::two_advections(c.to, i, c.from, c.index));
          break;
        }
    }
  }
}

// The blocks of the regions an instance names, and the expansion coefficient of a buoyant flow in every region.
void Validation::check_materials(const Group& g, const Instance& in) {
  const Strs need = needs(&g, in);
  Strs told;
  for (const std::string& region : regions(in.id)) {
    std::string material;
    const MaterialProps* mp = material_of(region, &material);
    if (!mp) continue;
    Strs missing;
    for (const std::string& b : need)
      if (b != kBeta && !has_block(*mp, b)) missing.push_back(b);
    if (!missing.empty()) {
      if (in.domains_written) error(E::material_block_missing(region, material, in.id, in.type, missing));
      continue;
    }
    if (contains(need, kBeta) && !has_block(*mp, kBeta) && !contains(told, material)) {
      told.push_back(material);
      error(E::buoyancy_needs_beta(region, material, in.id));
    }
  }
}

void Validation::check_conditions(const Group& g, const Instance& in) {
  const Route& row = g.row->condition_kinds.count(in.type) ? *g.row : *g.own.at(in.id);
  const auto offered_it = row.condition_kinds.find(in.type);
  if (offered_it == row.condition_kinds.end()) return;
  const Strs& offered = offered_it->second;
  const std::string geometry = m_.symmetry;
  const std::string base = "/physics/" + in.id + "/boundary_conditions/";
  for (const auto& [key, c] : in.conditions) {
    const std::string where = base + key + "/kind";
    if (!contains(offered, c.kind)) {
      std::string why = "engine_refuses", reason, roadmap;
      const auto n = row.not_offered.find(in.type);
      if (n != row.not_offered.end() && n->second.count(c.kind)) why = n->second.at(c.kind);
      const std::string lead = "not_exposed: ";
      if (why.rfind(lead, 0) == 0) {
        reason = why.substr(lead.size());
        why = "not_exposed";
        const auto rm = row.roadmap.find(in.type);
        if (rm != row.roadmap.end() && rm->second.count(c.kind)) roadmap = rm->second.at(c.kind);
      }
      if (in.type == kHeat)
        error(E::heat_condition_not_offered(in.id, c.kind, where, row.name, row.engine, geometry, "", offered, why, ""));
      else
        error(E::flow_condition_not_offered(in.id, c.kind, where, row.name, row.engine, geometry, option_of(in, "equation"), offered, why,
                                            reason, roadmap));
      continue;
    }
    // the kinds that an option value excludes (the heat kinds by advective form)
    const auto with = row.not_offered_with.find(in.type);
    if (with == row.not_offered_with.end()) continue;
    for (const auto& [option_value, refused] : with->second) {
      const size_t eq = option_value.find('=');
      const std::string option = option_value.substr(0, eq), value = option_value.substr(eq + 1);
      if (option_of(in, option) != value || !refused.count(c.kind)) continue;
      Strs left;
      for (const std::string& k : offered)
        if (!refused.count(k)) left.push_back(k);
      std::string why = refused.at(c.kind), rule;
      const std::string lead = "layer_rule: ";
      if (why.rfind(lead, 0) == 0) {
        rule = why.substr(lead.size());
        why = "layer_rule";
      }
      error(E::heat_condition_not_offered(in.id, c.kind, where, row.name, row.engine, geometry, value, left, why, rule));
    }
  }
  // every electrode galvanostatic: the potential level is free
  if (in.type == kCurrentDistribution) {
    Strs electrodes;
    bool potential = false;
    for (const auto& [key, c] : in.conditions) {
      if (c.kind != "electrode") continue;
      electrodes.push_back(key);
      if (!c.data.count("current_A") && !c.data.count("current_A_m")) potential = true;
    }
    if (!electrodes.empty() && !potential) error(E::potential_level_free(in.id, electrodes));
  }
}

void Validation::check_options(const Group& g, const Instance& in) {
  const Route& row = g.row->options.count(in.type) ? *g.row : *g.own.at(in.id);
  const auto rules = row.options.find(in.type);
  const PhysicsType* ty = R_.type(in.type);
  if (rules == row.options.end() || !ty) return;
  for (const OptionRule& rule : rules->second) {
    if (row.when.count(rule.name) || !in.options.count(rule.name)) continue;
    const Strs offered = rule.forced.empty() ? rule.offered : Strs{rule.forced};
    if (offered.empty()) continue;  // a number
    const json& value = in.options.at(rule.name).value;
    const std::string text = option_text(value);
    if (contains(offered, text)) continue;
    if (in.type == kHeat && rule.name == "stabilization" && text == "supg") {
      error(E::supg_needs_advection(in.id, row.name));
      continue;
    }
    bool integer = false;
    for (const OptionSpec& o : ty->options)
      if (o.name == rule.name) integer = o.type == "integer";
    json offered_json = json::array();
    for (const std::string& v : offered) offered_json.push_back(integer ? json(std::stoi(v)) : json(v));
    error(E::option_not_offered(ty->family, in.id, rule.name, value, offered_json, offered, row.name, row.engine));
  }
}

// The values of a property over the regions of an instance, and the first two regions that differ.  False when they
// agree; a region whose material does not give the property as a constant is left out (it has its own finding).
bool Validation::differing(const Instance& in, const detail::PropertyRef& p, E::TwoValues& tv, json& values) const {
  values = json::object();
  bool have = false, differ = false;
  for (const std::string& region : regions(in.id)) {
    std::string material;
    const MaterialProps* mp = material_of(region, &material);
    if (!mp) continue;
    const Property& v = mp->*(p.member);
    if (!v.given || v.is_law()) continue;
    values[region] = v.value;
    if (!have) {
      have = true;
      tv.region_a = region;
      tv.material_a = material;
      tv.a = v.value;
    } else if (!differ && v.value != tv.a) {
      differ = true;
      tv.region_b = region;
      tv.material_b = material;
      tv.b = v.value;
    }
  }
  return differ;
}

// The force and torque outputs of a flow instance, by name: those that name it, and those that name no physics when
// the boundary carries one of its conditions (`on_condition_only`) or in any case.
// An output that names no physics on a boundary that carries a condition of another flow instance, and none of this
// one, is that instance's.
bool Validation::force_output_of(const Instance& in, const Output& o) const {
  if (o.kind != "force" && o.kind != "torque") return false;
  if (!o.physics.empty()) return o.physics == in.id;
  if (in.conditions.count(o.on)) return true;
  for (const auto& [id, other] : m_.instances)
    if (id != in.id && other.type == kFluid && other.conditions.count(o.on)) return false;
  return true;
}
Strs Validation::force_outputs(const Instance& in, bool on_condition_only) const {
  Strs out;
  for (const auto& [name, o] : m_.outputs) {
    if (!force_output_of(in, o)) continue;
    if (on_condition_only && !in.conditions.count(o.on)) continue;
    out.push_back(name);
  }
  return out;
}

void Validation::check_uniform(const StudyStep&, const Group& g, const Instance& in) {
  const Route& row = *g.row;
  const Route& own = *g.own.at(in.id);
  const bool boussinesq = contains(row.couplings, "buoyancy");
  std::vector<std::pair<std::string, const Route*>> props;
  for (const Route* r : {&row, &own}) {
    const auto it = r->uniform.find(in.type);
    if (it == r->uniform.end()) continue;
    for (const std::string& p : it->second) {
      bool listed = false;
      for (const auto& x : props) listed = listed || x.first == p;
      if (!listed) props.emplace_back(p, r);
    }
  }
  for (const auto& [name, from] : props) {
    const detail::PropertyRef* p = detail::property(name);
    E::TwoValues tv;
    json values;
    if (!p || !differing(in, *p, tv, values)) continue;
    std::string reason = "transport";
    if (boussinesq) reason = "boussinesq";
    else if (in.type == kFluid) reason = contains(from->study_types, "transient") ? "transient" : "navier_stokes";
    error(E::property_not_uniform(in.id, name, from->name, tv, values, reason, ""));
  }
  // one constant under a condition: the viscosity of a force output with the Laplacian form
  const auto when = own.uniform_when.find(in.type);
  if (when == own.uniform_when.end()) return;
  for (const ConditionalUniform& cu : when->second) {
    const size_t eq = cu.option.find('=');
    if (option_of(in, cu.option.substr(0, eq)) != cu.option.substr(eq + 1)) continue;
    std::string output;
    if (!cu.output.empty()) {
      // a row without force outputs refuses the output itself
      const Strs outputs = row.force_reaction.empty() ? Strs() : force_outputs(in, false);
      if (outputs.empty()) continue;
      output = outputs[0];
    }
    const detail::PropertyRef* p = detail::property(cu.property);
    E::TwoValues tv;
    json values;
    if (!p || !differing(in, *p, tv, values)) continue;
    error(E::property_not_uniform(in.id, cu.property, own.name, tv, values, cu.why.rfind("engine_refuses", 0) == 0 ? "stokes_force" : "layer_force",
                                  output));
  }
}

// The data that are one expression over the instance: the conductivity, the heat source, the prescribed velocity.
void Validation::check_one_expression(const StudyStep&, const Group& g, const Instance& in) {
  const Route& row = *g.row;
  const auto list = row.one_expression.find(in.type);
  if (list == row.one_expression.end()) return;
  const bool boussinesq = contains(row.couplings, "buoyancy");
  const Strs& mine = regions(in.id);
  for (const std::string& datum : list->second) {
    if (datum == "heat_source") {
      // per region: the sources that cover it, as written without their "in"
      std::vector<json> covering(mine.size(), json::array());
      for (const SourceTerm& s : in.sources) {
        if (s.kind != "heat_source") continue;
        json written = s.written;
        written.erase("in");
        written.erase("kind");
        written.erase("notes");
        const Strs where = s.in.empty() ? mine : detail::regions_of(m_, {s.in});
        for (size_t i = 0; i < mine.size(); ++i)
          if (contains(where, mine[i])) covering[i].push_back(written);
      }
      size_t other = 0;
      for (size_t i = 1; i < mine.size() && other == 0; ++i)
        if (covering[i] != covering[0]) other = i;
      if (other == 0) continue;
      // what a region has, for the text and for the details
      const auto value = [&](size_t i) -> json {
        if (covering[i].empty()) return nullptr;
        if (covering[i].size() == 1 && covering[i][0].size() == 1) return covering[i][0].begin().value();
        return covering[i];
      };
      const auto key = [&](size_t i) { return covering[i].size() == 1 && covering[i][0].size() == 1 ? covering[i][0].begin().key() : std::string(); };
      const auto text = [&](size_t i, bool with_key) -> std::string {
        if (covering[i].empty()) return "no heat source";
        if (key(i).empty()) return std::to_string(covering[i].size()) + " heat sources";
        return (with_key ? key(i) + " = " : std::string()) + E::value_text(value(i));
      };
      json values = json::object();
      for (size_t i = 0; i < mine.size(); ++i) values[mine[i]] = value(i);
      error(E::heat_source_not_regionwise(in.id, row.name, boussinesq, mine[0], text(0, true), mine[other],
                                          text(other, key(other) != key(0) || key(0).empty()), values));
    } else if (datum == "prescribed_advection.velocity") {
      std::vector<size_t> sources;
      for (size_t i = 0; i < in.sources.size(); ++i)
        if (in.sources[i].kind == "prescribed_advection") sources.push_back(i);
      if (sources.size() > 1) {
        error(E::prescribed_advection_twice(in.id, sources));
        continue;
      }
      if (sources.empty() || in.sources[sources[0]].in.empty()) continue;
      const SourceTerm& s = in.sources[sources[0]];
      // the part of the instance that the source covers (an "in" that reaches outside it has its own finding, and one that
      // covers none of the instance only that finding)
      const size_t covered = intersection(mine, detail::regions_of(m_, {s.in})).size();
      if (covered == mine.size() || covered == 0) continue;
      bool agree = true;
      for (const char* name : {"density", "thermal.cp_J_kgK"}) {
        E::TwoValues tv;
        json values;
        const detail::PropertyRef* p = detail::property(name);
        agree = agree && p && !differing(in, *p, tv, values);
      }
      error(E::prescribed_advection_partial(in.id, sources[0], mine, s.in, agree));
    } else if (const detail::PropertyRef* p = detail::property(datum)) {
      E::TwoValues tv;
      json values;
      if (differing(in, *p, tv, values)) error(E::conductivity_not_regionwise(in.id, row.name, boussinesq, tv, values));
    }
  }
}

// A law of a field in a property that the instance reads: no route of this version takes one.
void Validation::check_laws(const StudyStep& step, const Group& g, const Instance& in) {
  const Route& own = *g.own.at(in.id);
  for (const detail::PropertyRef& p : detail::properties()) {
    if (!detail::reads(m_, step, in, p)) continue;
    for (const std::string& region : regions(in.id)) {
      std::string material;
      const MaterialProps* mp = material_of(region, &material);
      if (!mp || !(mp->*(p.member)).is_law()) continue;
      error(E::law_not_offered(in.id, p.name(), material, (mp->*(p.member)).law->of, own.name, p.pointer(material)));
    }
  }
}

// The force outputs of a flow: none on a row whose engine exposes no reaction (the one-way and Boussinesq rows), and
// the force rule of the Navier-Stokes rows; then the two warnings that keep the viscous forms honest.
void Validation::check_flow_outputs_and_warnings(const Group& g, const Instance& in) {
  const Route& own = *g.own.at(in.id);
  const std::string form = option_of(in, "viscous_form");
  if (g.row->force_reaction.empty()) {
    for (const std::string& name : force_outputs(in, false)) {
      const Output& o = m_.outputs.at(name);
      error(E::force_not_offered(name, o.kind, o.on, in.id, g.row->name, g.row->engine));
    }
  } else if (form == "laplacian" && contains(own.layer_rules, "force_with_laplacian_form")) {
    for (const std::string& name : force_outputs(in, false)) {
      const Output& o = m_.outputs.at(name);
      // a boundary that is not a key of the instance's conditions: what it carries is known with the geometry
      if (!in.conditions.count(o.on)) {
        add(Kind::deferred, E::deferred_force_condition(name, o.kind, o.on, in.id, own.name));
        continue;
      }
      const Condition& c = in.conditions.at(o.on);
      const bool moving = c.kind == "wall" && c.data.count("velocity_m_s");
      if (c.kind == "wall" && !moving) continue;
      // a kind that the row refuses has its own finding
      const auto offered = own.condition_kinds.find(in.type);
      if (offered != own.condition_kinds.end() && !contains(offered->second, c.kind)) continue;
      error(E::force_needs_stress_form(name, o.kind, o.on, c.kind, moving, own.name));
    }
  }
  const bool curved = curved_model(m_);
  for (const auto& [key, c] : in.conditions) {
    if (form == "stress" && (c.kind == "outlet" || c.kind == "open")) add(Kind::warning, E::outlet_stress_form(in.id, key, c.kind));
    if (form == "laplacian" && curved && (c.kind == "outlet" || c.kind == "traction" || c.kind == "slip") && curved_boundary(m_, key))
      add(Kind::warning, E::pseudo_traction(in.id, key, c.kind));
  }
}

// ------------------------------------------------------------------------------------------------ a step
// The time is a variable of an instance's data when a transient step solves it; a stationary step that solves the same
// instance has no time to give them.
void Validation::check_time(const StudyStep& step) {
  if (step.type != "stationary") return;
  const auto scan = [&](const std::map<std::string, Datum>& data, const std::string& base, const std::string& id) {
    for (const auto& [key, d] : data) {
      if (!d.field_expression) continue;
      for (size_t i = 0; i < d.exprs.size(); ++i) {
        if (!contains(d.exprs[i].used_variables(), "t")) continue;
        error(E::time_in_stationary_step(base + key + (d.written.is_array() ? "/" + std::to_string(i) : std::string()), step.id, id));
      }
    }
  };
  for (const std::string& id : step.physics) {
    const auto it = m_.instances.find(id);
    if (it == m_.instances.end() || bad_instances_.count(id) || !it->second.in_transient_step) continue;
    const Instance& in = it->second;
    for (const auto& [key, c] : in.conditions) scan(c.data, "/physics/" + id + "/boundary_conditions/" + key + "/", id);
    for (size_t i = 0; i < in.sources.size(); ++i) scan(in.sources[i].data, "/physics/" + id + "/sources/" + std::to_string(i) + "/", id);
  }
  for (const Coupling& c : m_.couplings)
    if (contains(step.physics, c.from) && contains(step.physics, c.to)) scan(c.data, "/couplings/" + std::to_string(c.index) + "/", c.to);
}

// The strategy of a step: the one written, checked against each coupled row; else one_way on an acyclic graph, else
// the default of the route of the two-way group.
void Validation::check_strategy(const StudyStep& step, const std::vector<Group>& groups, StepReport& sr) {
  const std::string& written = step.coupling.strategy;
  if (!written.empty()) {
    sr.strategy = written;
    for (const Group& g : groups)
      if (!g.bad && g.row && !g.row->strategies.empty() && !contains(g.row->strategies, written))
        error(E::strategy_not_offered(step_pointer(step) + "/coupling/strategy", written, g.row->name, g.row->strategies,
                                      g.row->default_strategy, step.id));
    // a two-way group never runs as one unlabelled sweep, on a route or on none
    if (written == "one_way")
      for (const Strs& component : sr.graph.components) {
        if (component.size() < 2) continue;
        const Group* of = nullptr;
        for (const Group& g : groups)
          if (contains(g.ids, component[0])) of = &g;
        if (of && of->bad) continue;
        const Strs offered = of && of->row ? of->row->strategies : Strs();
        if (!offered.empty() && !contains(offered, written)) continue;  // refused above, with the route
        error(E::one_way_on_two_way_group(step_pointer(step) + "/coupling/strategy", step.id, component, offered));
      }
    return;
  }
  sr.strategy = "one_way";
  for (const Strs& component : sr.graph.components) {
    if (component.size() < 2) continue;
    const Group* of = nullptr;
    for (const Group& g : groups)
      if (contains(g.ids, component[0])) of = &g;
    if (of && of->bad) continue;
    if (of && of->row && !of->row->default_strategy.empty()) {
      sr.strategy = of->row->default_strategy;
      continue;
    }
    sr.strategy.clear();
    error(E::two_way_needs_strategy(step_pointer(step), step.id, component));
    return;
  }
}

// A scale: the step has the coefficient it multiplies.  A ramp: a pointer to one rampable field of an instance that the
// step solves, on the row it takes there, and a last value that is the document's.
void Validation::check_continuation(const StudyStep& step, const std::vector<Group>& groups) {
  const std::string& ramp = step.continuation.ramp;
  const std::string& scale = step.continuation.scale;
  if (ramp.empty() && scale.empty()) return;
  for (const Group& g : groups)
    if (g.bad) return;
  if (!scale.empty()) {
    bool applies = false, fluid = false;
    for (const Group& g : groups) {
      if (scale == "buoyancy") {
        applies = applies || contains(g.terms, "buoyancy");
        continue;
      }
      for (const std::string& id : g.ids) {
        if (inst(id).type != kFluid) continue;
        fluid = true;
        // only the steady Navier-Stokes engine has a viscosity schedule: not Stokes, not the Boussinesq row (it scales the
        // buoyancy), not the time stepper
        applies = applies || (step.type == "stationary" && option_of(inst(id), "equation") == "navier_stokes" && !contains(g.terms, "buoyancy"));
      }
    }
    if (!applies) error(E::scale_not_applicable(step_pointer(step) + "/continuation/scale", scale, step.id, fluid));
    return;
  }
  Strs rampable;          // the rampable pointers of the document
  std::string elsewhere;  // the instance of the pointer, when it is rampable in another step only
  double document = 0.0;
  bool found = false;
  for (const auto& [id, in] : m_.instances) {
    // the row of the instance in this step, else in the first step that solves it
    const Group* here = nullptr;
    for (const Group& g : groups)
      if (contains(g.ids, id)) here = &g;
    const Group* of = here;
    if (!of) {
      const auto h = home_.find(id);
      if (h == home_.end()) continue;
      of = &groups_[h->second.first][h->second.second];
    }
    if (of->bad || !of->row) continue;
    for (const auto& [key, c] : in.conditions) {
      if (c.kind != "electrode") continue;
      for (const std::string& field : of->row->rampable) {
        const auto d = c.data.find(field);
        if (d == c.data.end() || d->second.values.empty()) continue;
        rampable.push_back("/physics/" + id + "/boundary_conditions/" + key + "/" + field);
        if (rampable.back() != ramp) continue;
        if (here) {
          found = true;
          document = d->second.values[0];
        } else {
          elsewhere = id;
        }
      }
    }
  }
  const std::string base = step_pointer(step) + "/continuation/";
  if (!found) {
    error(elsewhere.empty() ? E::field_not_rampable(base + "ramp", ramp, rampable)
                            : E::ramp_of_unsolved_physics(base + "ramp", ramp, elsewhere, step.id, rampable));
    return;
  }
  const std::vector<double>& values = step.continuation.values;
  if (values.empty()) return;
  const double last = values.back();
  if (std::fabs(last - document) > 1e-12 * std::max(std::fabs(last), std::fabs(document)))
    error(E::ramp_end_mismatch(base + "values/" + std::to_string(values.size() - 1), ramp, last, document));
}

// A law reads a field: an instance of the step provides it on the region, or the step gives it a value.
void Validation::check_law_fields(const StudyStep& step) {
  for (const auto& [material, props] : m_.materials) {
    if (bad_materials_.count(material)) continue;
    Strs of_material;
    for (const auto& r : m_.regions)
      if (r.material == material) of_material.push_back(r.name);
    if (of_material.empty()) continue;
    for (const detail::PropertyRef& p : detail::properties()) {
      const Property& value = props.*(p.member);
      if (!value.is_law()) continue;
      for (const std::string& field : value.law->of) {
        if (step.given.count(field)) continue;
        Strs uncovered;
        for (const std::string& consumer : step.physics) {
          const auto ci = m_.instances.find(consumer);
          if (ci == m_.instances.end() || bad_instances_.count(consumer) || !detail::reads(m_, step, ci->second, p)) continue;
          for (const std::string& region : intersection(of_material, regions(consumer))) {
            bool covered = false;
            for (const std::string& producer : step.physics) {
              const auto pi = m_.instances.find(producer);
              const PhysicsType* pt = pi == m_.instances.end() ? nullptr : R_.type(pi->second.type);
              covered = covered || (pt && detail::provides(*pt, field) && contains(regions(producer), region));
            }
            if (!covered && !contains(uncovered, region)) uncovered.push_back(region);
          }
        }
        if (uncovered.empty()) continue;
        Strs providers;
        for (const PhysicsType& t : R_.types)
          if (detail::provides(t, field)) providers.push_back(t.name);
        error(E::law_field_not_provided(p.name(), material, field, step.id, detail::regions_of(m_, uncovered), providers, p.pointer(material)));
      }
    }
  }
}

// The terms that act nowhere, and the pairs that share a region with no term between them.
void Validation::check_graph(const StudyStep&, const std::vector<Group>& groups, const CouplingGraph& graph) {
  const auto bad = [&](const std::string& id) {
    for (const Group& g : groups)
      if (contains(g.ids, id)) return g.bad;
    return true;
  };
  for (const GraphEdge& e : graph.edges) {
    if (e.kind != "term" || !e.acts_in.empty() || bad(e.from) || bad(e.to)) continue;
    if (geometry_free(inst(e.from).type) || geometry_free(inst(e.to).type)) continue;
    // (an "in" outside a shared domain that is not empty has its own finding)
    if (!intersection(regions(e.from), regions(e.to)).empty()) continue;
    error(E::empty_overlap(e.coupling, e.term, e.from, e.to, regions(e.from), regions(e.to)));
  }
  for (const UncoupledPair& p : graph.uncoupled)
    if (!bad(p.a) && !bad(p.b)) add(Kind::warning, E::uncoupled_pair(p.a, p.b, p.regions, p.candidates));
}

// ------------------------------------------------------------------------------------------------ the document
// Two instances of one family on a shared region.
void Validation::check_overlap() {
  for (auto a = m_.instances.begin(); a != m_.instances.end(); ++a)
    for (auto b = std::next(a); b != m_.instances.end(); ++b) {
      if (bad_instances_.count(a->first) || bad_instances_.count(b->first)) continue;
      const PhysicsType* ta = R_.type(a->second.type);
      const PhysicsType* tb = R_.type(b->second.type);
      if (!ta || !tb || ta->family != tb->family) continue;
      const Strs shared = intersection(regions(a->first), regions(b->first));
      if (!shared.empty()) error(E::domains_overlap(a->first, b->first, ta->family, shared));
    }
}

// "in" restricts: a domain term to a part of the shared domain of its two instances, a source to a part of its instance.
void Validation::check_selections() {
  const auto outside_of = [](const Strs& named, const Strs& within) {
    Strs out;
    for (const std::string& r : named)
      if (!contains(within, r)) out.push_back(r);
    return out;
  };
  for (const Coupling& c : m_.couplings) {
    if (c.in.empty() || !usable(c.from) || !usable(c.to) || geometry_free(inst(c.from).type) || geometry_free(inst(c.to).type)) continue;
    const Strs shared = intersection(regions(c.from), regions(c.to));
    if (shared.empty()) continue;  // coupling.empty_overlap
    const Strs named = detail::regions_of(m_, {c.in});
    const Strs outside = outside_of(named, shared);
    if (!outside.empty()) error(E::in_outside_shared_domain(c.index, c.term, c.from, c.to, c.in, named, shared, outside));
  }
  for (const auto& [id, in] : m_.instances) {
    if (!usable(id) || geometry_free(in.type) || regions(id).empty()) continue;
    for (size_t i = 0; i < in.sources.size(); ++i) {
      const SourceTerm& s = in.sources[i];
      if (s.in.empty()) continue;
      const Strs named = detail::regions_of(m_, {s.in});
      const Strs outside = outside_of(named, regions(id));
      if (!outside.empty()) error(E::source_outside_domain(id, i, s.kind, s.in, named, regions(id), outside));
    }
  }
}

// The mesher's probe, for a document that has an instance with a geometry and no imported mesh: the unavailable finding
// build.requires_mesher when the mesher does not run.
void Validation::check_mesher() {
  if (!options_.probe_mesher || !m_.mesh.file.empty() || m_.regions.empty()) return;
  bool geometry = false;
  for (const auto& [id, in] : m_.instances)
    geometry = geometry || (R_.type(in.type) && !geometry_free(in.type));
  if (!geometry) return;
  const MesherStatus status = mesher_status(options_.mesher);
  if (!status.available) add(Kind::unavailable, E::requires_mesher(status.path, status.why));
}

// The first elliptical segment of a contour in the shape written at `pointer`, depth first through union and difference
// in written order: its pointer and its index in the contour.
bool find_elliptical_segment(const json& shape, const std::string& pointer, std::string& where, size_t& index) {
  if (!shape.is_object()) return false;
  for (const char* op : {"union", "difference"}) {
    if (!shape.contains(op) || !shape.at(op).is_array()) continue;
    for (size_t k = 0; k < shape.at(op).size(); ++k)
      if (find_elliptical_segment(shape.at(op)[k], pointer + "/" + op + "/" + std::to_string(k), where, index)) return true;
  }
  if (shape.contains("contour") && shape.at("contour").is_object() && shape.at("contour").contains("segments") &&
      shape.at("contour").at("segments").is_array()) {
    const json& segments = shape.at("contour").at("segments");
    for (size_t k = 0; k < segments.size(); ++k)
      if (segments[k].is_object() && segments[k].contains("ellipse")) {
        where = pointer + "/contour/segments/" + std::to_string(k);
        index = k;
        return true;
      }
  }
  return false;
}

// The bounded meshes take lines and circular arcs: a contour's elliptical segment is a spline curve.  One finding per
// enabled region, at the first such segment.
void Validation::check_shapes() {
  if (m_.symmetry == "3d" || !m_.mesh.file.empty() || !doc_.is_object() || !doc_.contains("regions") || !doc_.at("regions").is_array()) return;
  const json& regions = doc_.at("regions");
  for (size_t i = 0; i < regions.size(); ++i) {
    const json& r = regions[i];
    if (!r.is_object() || !r.contains("name") || !r.at("name").is_string() || !r.contains("shape")) continue;
    const std::string name = r.at("name").get<std::string>();
    const bool enabled = std::any_of(m_.regions.begin(), m_.regions.end(), [&](const spec::Region& g) { return g.name == name; });
    std::string where;
    size_t index = 0;
    if (enabled && find_elliptical_segment(r.at("shape"), "/regions/" + std::to_string(i) + "/shape", where, index))
      error(E::shape_not_offered(name, where, index));
  }
}

// A between selector whose two domains share a region has no direction for its normal (at any depth of a union or a
// difference).
void Validation::between_overlap_of(const std::string& name, const Selector& s, const std::string& pointer) {
  if (s.kind == "between" && s.between.size() == 2 && s.between[1] != "exterior") {
    const Strs a = detail::regions_of(m_, {s.between[0]}), b = detail::regions_of(m_, {s.between[1]});
    Strs shared;
    for (const spec::Region& r : m_.regions)
      if (contains(a, r.name) && contains(b, r.name)) shared.push_back(r.name);
    if (!shared.empty()) error(E::between_overlap(name, pointer + "/between", s.between[0], s.between[1], shared));
  }
  if (s.kind == "union" || s.kind == "difference")
    for (size_t k = 0; k < s.operands.size(); ++k)
      between_overlap_of(name, s.operands[k], pointer + "/" + s.kind + "/" + std::to_string(k));
}
void Validation::check_between_overlap() {
  for (const auto& [name, s] : m_.boundaries) between_overlap_of(name, s, "/boundaries/" + name);
}

// A force or a torque is the reaction of a flow: one that names a physics that is not a flow, or that is written in a
// document with no flow, belongs to nothing.  (The flows' own rules are checked with their instances.)
void Validation::check_force_owners() {
  // a physics that has a finding of its own (an unknown type is one) is reported by that finding alone, and while one has,
  // whether the document has a flow is not known
  bool any_flow = false;
  const bool undecided = !bad_instances_.empty();
  for (const auto& [id, in] : m_.instances) any_flow = any_flow || in.type == kFluid;
  for (const auto& [name, o] : m_.outputs) {
    if (o.kind != "force" && o.kind != "torque") continue;
    if (!o.physics.empty()) {
      const auto it = m_.instances.find(o.physics);
      if (it == m_.instances.end() || bad_instances_.count(o.physics)) continue;
      if (it->second.type != kFluid) error(E::force_without_flow(name, o.kind, o.on, o.physics, it->second.type));
    } else if (!any_flow && !undecided) {
      error(E::force_without_flow(name, o.kind, o.on, "", ""));
    }
  }
}

// What waits for the resolved boundaries: which entities each named boundary selects, and the partition rule of every
// instance that has a boundary (design note 5.5).
void Validation::check_deferred() {
  Strs names;
  // a member of a cycle of names is refused and not resolved later (the cycle is one finding, at its first member); a boundary
  // whose selector is refused is not in the model at all
  std::set<std::string> refused;
  for (const Finding& f : out_.all()) {
    if (f.kind != Kind::error || f.error.code() != codes::boundary_cycle) continue;
    try {
      const json details = json::parse(f.error.details());
      for (const json& member : details.at("cycle")) refused.insert(member.get<std::string>());
    } catch (const std::exception&) {
    }
  }
  for (const auto& [name, selector] : m_.boundaries)
    if (!refused.count(name)) names.push_back(name);
  if (!names.empty()) add(Kind::deferred, E::deferred_boundary_resolution(names, m_.mesh.file));
  for (const auto& [id, in] : m_.instances) {
    if (bad_instances_.count(id) || !R_.type(in.type) || geometry_free(in.type)) continue;
    Strs keys;
    for (const auto& [key, c] : in.conditions) keys.push_back(key);
    add(Kind::deferred, E::deferred_partition(id, keys, m_.mesh.file));
  }
}

// The geometry order (design note 3.3): it may not exceed the smallest field order, nor what an option value that
// needs straight elements allows; the default is the largest order that every instance takes.  With an imported mesh
// the order is the file's, and the rules are checks that run when the file is read.
void Validation::check_geometry_order() {
  struct Limit {
    std::string id, option, value;  // option empty: the field order
    int order;
  };
  std::vector<Limit> limits;
  for (const auto& [id, in] : m_.instances) {
    const PhysicsType* ty = R_.type(in.type);
    if (bad_instances_.count(id) || !ty || geometry_free(in.type)) continue;
    const auto order = in.options.find("order");
    if (order != in.options.end() && order->second.value.is_number()) limits.push_back({id, "", "", order->second.value.get<int>()});
    for (const OptionSpec& o : ty->options) {
      const std::string value = option_of(in, o.name);
      const auto max = o.max_geometry_order.find(value);
      if (max != o.max_geometry_order.end()) limits.push_back({id, o.name, value, max->second});
    }
  }
  const MeshKeys& mesh = m_.mesh;
  if (!mesh.file.empty()) {
    rep_.geometry_order = 0;
    for (const Limit& l : limits) {
      if (l.order >= 2) continue;
      if (l.option.empty()) add(Kind::deferred, E::deferred_field_order(mesh.file, l.id, l.order));
      else add(Kind::deferred, E::deferred_straight_elements(mesh.file, l.id, l.option, l.value));
    }
    // the element types of the file, on a row whose engine takes triangles only
    for (const auto& groups : groups_)
      for (const Group& g : groups) {
        if (g.bad || !g.row || !contains(g.row->element_types, "triangle") || contains(g.row->element_types, "quadrilateral")) continue;
        std::string id = g.ids[0];
        for (const std::string& x : g.ids)
          if (contains(g.row->cell_types, inst(x).type)) {
            id = x;
            break;
          }
        add(Kind::deferred, E::deferred_element_types(mesh.file, id, g.row->name, g.row->element_types));
      }
    return;
  }
  if (mesh.geometry_order_written) {
    rep_.geometry_order = mesh.geometry_order;
    for (const Limit& l : limits) {
      if (l.order >= mesh.geometry_order) continue;
      if (l.option.empty())
        error(E::geometry_order_exceeds(mesh.geometry_order, l.order, l.id,
                                        inst(l.id).type == kCurrentDistribution && m_.symmetry != "3d"));
      else
        error(E::geometry_order_not_offered(mesh.geometry_order, l.id, l.option, l.value));
    }
    return;
  }
  // the default: the largest order every instance takes; the instance named is one with field order 1, else the first
  // whose option needs straight elements
  int order = 2;
  for (const Limit& l : limits) order = std::min(order, l.order);
  const Limit* cause = nullptr;
  for (const Limit& l : limits)
    if (l.order == order && l.option.empty() && !cause) cause = &l;
  for (const Limit& l : limits)
    if (l.order == order && !cause) cause = &l;
  rep_.geometry_order = order;
  if (cause && order < 2 && curved_model(m_)) add(Kind::warning, E::geometry_order_lowered(cause->id, cause->option, cause->value));
}

void Validation::run() {
  if (m_.instances.empty() || m_.study.empty()) return;
  if (!scan_grammar_findings()) return;
  check_mesher();
  mark_coupling_ends();
  required_options();
  make_groups();
  resolve_regions();
  beta_warnings();
  for (size_t s = 0; s < m_.study.size(); ++s) {
    const StudyStep& step = m_.study[s];
    const std::vector<Group>& groups = groups_[s];
    StepReport sr;
    sr.id = step.id;
    StudyStep solved = step;
    solved.physics.clear();
    for (const Group& g : groups) solved.physics.insert(solved.physics.end(), g.ids.begin(), g.ids.end());
    // the nodes in the order of the step's list
    Strs nodes;
    for (const std::string& id : step.physics)
      if (contains(solved.physics, id) && !contains(nodes, id)) nodes.push_back(id);
    solved.physics = nodes;
    sr.graph = coupling_graph(m_, solved, rep_.regions);
    for (const Group& g : groups) sr.routes.push_back(RouteUse{g.ids, g.row ? g.row->name : std::string()});
    check_time(step);
    for (const Group& g : groups) {
      if (g.bad) continue;
      report_route(step, g);
      if (g.row) check_group(step, g);
    }
    check_strategy(step, groups, sr);
    check_continuation(step, groups);
    check_law_fields(solved);
    check_graph(step, groups, sr.graph);
    rep_.steps.push_back(std::move(sr));
  }
  check_overlap();
  check_selections();
  check_between_overlap();
  check_shapes();
  check_force_owners();
  check_deferred();
  if (!mesh_bad_) check_geometry_order();
}

}  // namespace

ValidationReport validate_model(const nlohmann::json& doc, const ValidateOptions& options) {
  ValidationReport rep;
  Findings found;
  try {
    rep.model = spec::parse_model_collect(doc, found);
    Validation(doc, rep, found, options).run();
  } catch (const Error& e) {
    found.add(Kind::error, e);
  } catch (const std::exception& e) {
    found.add(Kind::error, Error(std::string("the validation of the model document failed: ") + e.what(), "",
                                 "This is a defect of the validation, not of the document; please report it."));
  }
  for (const Finding& f : found.all()) {
    switch (f.kind) {
      case Kind::error: rep.errors.push_back(f.error); break;
      case Kind::unavailable: rep.unavailable.push_back(f.error); break;
      case Kind::deferred: rep.deferred.push_back(f.error); break;
      case Kind::warning: rep.warnings.push_back(f.error); break;
    }
  }
  return rep;
}

// ------------------------------------------------------------------------------------------------ the full check
namespace {

// What the check knows of one instance with a geometry: its regions, the keys of its conditions (in the order of the map,
// "remaining" among them) and the partition of its domain boundary.
struct Part {
  Strs regions, keys;
  selection::Partition partition;
};

// The kind that the example of the uncovered hint gives to "remaining": a wall for a flow, else an insulation.
std::string remaining_example(const PhysicsType* type) {
  if (type)
    for (const char* kind : {"wall", "insulation"})
      for (const ConditionKind& k : type->condition_kinds)
        if (k.name == kind) return kind;
  return type && !type->condition_kinds.empty() ? type->condition_kinds.front().name : std::string("insulation");
}

}  // namespace

void check_boundaries(ValidationReport& report, const selection::Table& table, const selection::Resolution& resolution,
                      const std::string& document) {
  const ResolvedModel& m = report.model;
  const Registry& reg = registry();
  std::vector<Error> errors, warnings;
  std::set<std::string> empty;
  const auto selected = [&](const std::string& name) -> const std::vector<int>& {
    static const std::vector<int> none;
    const auto it = resolution.selected.find(name);
    return it == resolution.selected.end() ? none : it->second;
  };
  // the empty selections
  for (const auto& [name, sel] : m.boundaries) {
    if (!selected(name).empty()) continue;
    empty.insert(name);
    const auto e = resolution.empty.find(name);
    errors.push_back(E::empty_selection(name, sel.written, table, e == resolution.empty.end() ? json(nullptr) : e->second));
  }
  // the partition rule of each instance with a geometry (an instance with a key that selects nothing is left to that finding)
  std::map<std::string, Part> parts;
  for (const auto& [id, in] : m.instances) {
    const PhysicsType* type = reg.type(in.type);
    const auto regs = report.regions.find(id);
    if (!type || geometry_free_type(reg, in.type) || regs == report.regions.end() || regs->second.empty()) continue;
    Part part;
    part.regions = regs->second;
    bool blocked = false;
    for (const auto& [key, c] : in.conditions) {
      part.keys.push_back(key);
      blocked = blocked || empty.count(key);
    }
    if (blocked) continue;
    part.partition = selection::partition(table, resolution, part.regions, part.keys);
    const selection::Partition& p = part.partition;
    for (const std::string& key : p.outside_keys) errors.push_back(E::boundary_outside_domain(id, key, part.regions, table, selected(key)));
    for (const std::string& key : part.keys) {
      const auto it = p.interior_keys_tags.find(key);
      if (it != p.interior_keys_tags.end() && !it->second.empty())
        errors.push_back(E::boundary_interior(id, key, it->second.size() == selected(key).size(), table, it->second));
    }
    std::map<std::pair<size_t, size_t>, std::vector<int>> pairs;
    for (const auto& [tag, keys] : p.doubly)
      for (size_t a = 0; a < keys.size(); ++a)
        for (size_t b = a + 1; b < keys.size(); ++b) {
          const auto at = [&](const std::string& k) { return size_t(std::find(part.keys.begin(), part.keys.end(), k) - part.keys.begin()); };
          pairs[{at(keys[a]), at(keys[b])}].push_back(tag);
        }
    for (const auto& [pair, tags] : pairs)
      errors.push_back(E::boundary_double(id, part.keys[pair.first], part.keys[pair.second], table, tags));
    if (!p.uncovered.empty()) {
      Strs named;
      for (const std::string& key : part.keys)
        if (key != "remaining") named.push_back(key);
      const std::string domain = in.domains_written && in.domains.size() == 1 ? in.domains[0]
                                 : part.regions.size() == 1                    ? part.regions[0]
                                                                               : std::string();
      errors.push_back(E::boundary_uncovered(id, domain, named, remaining_example(type), document, table, p.uncovered));
    }
    parts[id] = std::move(part);
  }
  // the outputs on the resolved boundaries: the instance an output belongs to is the physics it names, else the first
  // instance whose type has its quantity (a force: a flow) and that the selection touches
  const auto owner = [&](const Output& o, const std::string& boundary) -> const Instance* {
    const Instance* first = nullptr;
    for (const auto& [id, in] : m.instances) {
      if (!parts.count(id)) continue;
      bool mine = false;
      if (!o.physics.empty()) {
        mine = o.physics == id;
      } else if (o.kind == "force" || o.kind == "torque") {
        mine = in.type == kFluid;
      } else if (o.kind == "electrode_current") {
        mine = in.type == kCurrentDistribution;
      } else if (const PhysicsType* ty = reg.type(in.type); ty && ty->quantities.count(m.symmetry)) {
        for (const Quantity& q : ty->quantities.at(m.symmetry)) mine = mine || q.name == o.of;
      }
      if (!mine) continue;
      if (!first) first = &in;
      const Part& part = parts.at(id);
      const selection::Placement pl = selection::place(table, resolution, part.partition, part.regions, boundary);
      if (!pl.boundary.empty() || !pl.interior.empty()) return &in;
    }
    return first;
  };
  // the force outputs that validation could not decide: with the Laplacian form, a force on a boundary that is not a key
  std::map<std::string, Error> force_rule;
  for (const Error& d : report.deferred)
    if (d.code() == codes::output_force_needs_stress_form && d.where().rfind("/outputs/", 0) == 0) force_rule.emplace(d.where().substr(9), d);
  for (const auto& [name, o] : m.outputs) {
    if (o.on.empty() || empty.count(o.on) || !m.boundaries.count(o.on)) continue;
    const Instance* in = owner(o, o.on);
    if (!in) continue;
    const Part& part = parts.at(in->id);
    const selection::Placement pl = selection::place(table, resolution, part.partition, part.regions, o.on);
    if (pl.boundary.empty() && pl.interior.empty()) {
      errors.push_back(E::boundary_not_in_domain(name, o.on, in->id, part.regions, table, selected(o.on)));
      continue;
    }
    const bool force = o.kind == "force" || o.kind == "torque";
    if (force && !pl.interior.empty()) {
      errors.push_back(E::force_on_interior(name, o.kind, o.on, in->id, table, pl.interior));
      continue;
    }
    const auto rule = force_rule.find(name);
    if (!force || rule == force_rule.end() || in->conditions.count(o.on)) continue;
    std::string route;
    try {
      route = json::parse(rule->second.details()).value("route", std::string());
    } catch (const std::exception&) {
    }
    for (const std::string& key : pl.keys) {
      const auto c = in->conditions.find(key);
      if (c == in->conditions.end()) continue;
      const bool moving = c->second.kind == "wall" && c->second.data.count("velocity_m_s");
      if (c->second.kind == "wall" && !moving) continue;
      errors.push_back(E::force_needs_stress_form_on_part(name, o.kind, o.on, key, c->second.kind, moving, route));
      break;
    }
  }
  // the partial matches
  for (const auto& [name, sel] : m.boundaries) {
    const auto it = resolution.partial.find(name);
    if (it == resolution.partial.end() || !it->second.is_object() || !it->second.contains("tags")) continue;
    warnings.push_back(E::partial_match(name, sel.written, table, it->second.at("tags").get<std::vector<int>>()));
  }
  // what this check ran is no longer deferred
  std::vector<Error> still;
  for (const Error& d : report.deferred)
    if (d.code() != codes::boundary_empty_selection && d.code() != codes::physics_boundary_uncovered &&
        d.code() != codes::output_force_needs_stress_form)
      still.push_back(d);
  report.deferred = std::move(still);
  report.errors.insert(report.errors.end(), errors.begin(), errors.end());
  report.warnings.insert(report.warnings.end(), warnings.begin(), warnings.end());
}

nlohmann::json report_json(const ValidationReport& report, bool strict) {
  auto finding = [](const Error& e) {
    json j = error_json(e);
    j.erase("ok");
    return j;
  };
  auto list = [&](const std::vector<Error>& v) {
    json a = json::array();
    for (const Error& e : v) a.push_back(finding(e));
    return a;
  };
  json out = {{"ok", true}, {"name", report.model.name}, {"parameters", report.model.parameters},
              {"errors", list(report.errors)}, {"unavailable", list(report.unavailable)},
              {"deferred", list(report.deferred)}, {"warnings", list(report.warnings)}};
  const Error* first = !report.errors.empty() ? &report.errors.front()
                       : (strict && !report.unavailable.empty() ? &report.unavailable.front() : nullptr);
  if (first) {
    out["ok"] = false;
    const json top = finding(*first);
    for (auto it = top.begin(); it != top.end(); ++it) out[it.key()] = it.value();
  }
  return out;
}

nlohmann::json error_codes() {
  std::vector<CodeInfo> table = code_table();
  std::sort(table.begin(), table.end(), [](const CodeInfo& a, const CodeInfo& b) { return std::string(a.code) < std::string(b.code); });
  json out = json::array();
  for (const CodeInfo& c : table) {
    json kinds = json::array();
    const std::string all = c.kinds;
    size_t from = 0;
    while (from < all.size()) {
      const size_t to = all.find(' ', from);
      kinds.push_back(all.substr(from, to == std::string::npos ? std::string::npos : to - from));
      from = to == std::string::npos ? all.size() : to + 1;
    }
    out.push_back({{"code", c.code}, {"kinds", kinds}, {"summary", c.summary}, {"status", c.reserved ? "reserved" : "raised"}});
  }
  return out;
}

}  // namespace axmp::model
