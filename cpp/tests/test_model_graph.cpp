// The coupling graph of a model document (cpp/include/axionmultiphysics/model_graph.hpp): strongly connected components
// in topological order on hand-constructed graphs, then the graph that validate_model (model_validate.hpp) builds for
// the documents of the interface design note (section 3.7: where a term acts, property dependences, the graph and its
// strongly connected components; section 4.4: the busbar's loop).  Built in every build; nothing here needs the mesher
// or an engine.
//
// INDEPENDENT REFERENCES.
//   * The components and their order are derived by hand from each graph below, written next to it; nothing is
//     computed by a second algorithm.  The order rule is the header's: topological, ties broken by the position in
//     `nodes` of a component's first member, members in the order of `nodes`.
//   * The documents of section 4 of the note, verbatim in cpp/tests/data/model/, and variants written here.
//   * Section 3.7 for what a term edge and a property edge are, and for the warning on an uncoupled pair; section 8.4
//     for the terms between two types (the candidates).
//
// TEXTS.  The note gives no text for coupling.empty_overlap (only "no shared region", which the message carries),
// coupling.restriction_not_offered or coupling.uncoupled_pair: the texts here are the test author's (AUTHOR TEXT) and
// are the specification.
//
// THE API THE TESTS USE (namespace axmp::model):
//   strongly_connected_components(nodes, edges) -> vector<vector<string>>       (model_graph.hpp)
//   validate_model(json) -> ValidationReport { errors, unavailable, deferred, warnings, model, regions, steps,
//                                              geometry_order }                    (model_validate.hpp)
//   StepReport { id, graph, routes, strategy }; CouplingGraph { step, nodes, edges, components, uncoupled };
//   GraphEdge { from, to, kind, term, coupling, property, material, field, acts_in }; UncoupledPair { a, b, regions,
//   candidates }; RouteUse { physics, route }.
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/model_graph.hpp"
#include "axionmultiphysics/model_validate.hpp"

using nlohmann::json;
using axmp::Error;
namespace M = axmp::model;

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
using Comps = std::vector<Strs>;
using Edges = std::vector<std::pair<std::string, std::string>>;

std::string show(const Strs& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + v[i];
  return s + "]";
}
std::string show(const Comps& c) {
  std::string s = "[";
  for (size_t i = 0; i < c.size(); ++i) s += (i ? ", " : "") + show(c[i]);
  return s + "]";
}

json J(const char* s) { return json::parse(s); }

json load(const std::string& name) {
#ifdef AXMP_SOURCE_DIR
  const std::string path = std::string(AXMP_SOURCE_DIR) + "/cpp/tests/data/model/" + name;
#else
  const std::string path = "cpp/tests/data/model/" + name;
#endif
  std::ifstream in(path);
  if (!in) {
    fail("cannot read " + path);
    return json::object();
  }
  json d;
  in >> d;
  return d;
}

json details_of(const Error& e) { return e.details().empty() ? json(nullptr) : json::parse(e.details()); }

std::vector<Error> with_code(const std::vector<Error>& v, const std::string& code) {
  std::vector<Error> out;
  for (const auto& e : v)
    if (e.code() == code) out.push_back(e);
  return out;
}

// Runs one check; a check that throws (the stub of the implementation does) is one failure, and the others still run.
void run(const char* name, const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception& e) {
    fail(std::string(name) + " threw: " + e.what());
  }
}

const M::StepReport* step_of(const M::ValidationReport& r, const std::string& id) {
  for (const auto& s : r.steps)
    if (s.id == id) return &s;
  fail("no step '" + id + "' in the report");
  return nullptr;
}

const M::GraphEdge* edge(const M::CouplingGraph& g, const std::string& from, const std::string& to, const std::string& kind) {
  for (const auto& e : g.edges)
    if (e.from == from && e.to == to && e.kind == kind) return &e;
  return nullptr;
}

// ================================================================================================ hand-made graphs
// Each case: nodes, edges, and the components derived by hand (the derivation is in the comment).
struct GraphCase {
  const char* name;
  Strs nodes;
  Edges edges;
  Comps expected;
};

void check_components_of_hand_made_graphs() {
  const std::vector<GraphCase> cases = {
      // one node, no edge
      {"single", {"a"}, {}, {{"a"}}},
      // a -> b: two components; a must come before b although b is listed first
      {"one_way_listed_backwards", {"b", "a"}, {{"a", "b"}}, {{"a"}, {"b"}}},
      // a -> b with the nodes in order
      {"one_way", {"a", "b"}, {{"a", "b"}}, {{"a"}, {"b"}}},
      // a <-> b: one component, members in the order of nodes
      {"two_cycle", {"b", "a"}, {{"a", "b"}, {"b", "a"}}, {{"b", "a"}}},
      // a -> b <- c: three singletons (one WEAK component).  Ready at first: {a} (position 0) and {c} (position 2);
      // a first, then c, then b.
      {"converging", {"a", "b", "c"}, {{"a", "b"}, {"c", "b"}}, {{"a"}, {"c"}, {"b"}}},
      // a self-loop is a component of one
      {"self_loop", {"a", "b"}, {{"a", "a"}, {"a", "b"}}, {{"a"}, {"b"}}},
      // diamond a -> b, a -> c, b -> d, c -> d with nodes d, c, b, a.  Ready: {a}; then {c} (position 1) before {b}
      // (position 2); then {d}.
      {"diamond", {"d", "c", "b", "a"}, {{"a", "b"}, {"a", "c"}, {"b", "d"}, {"c", "d"}}, {{"a"}, {"c"}, {"b"}, {"d"}}},
      // x -> y and p <-> q, nodes x, y, p, q.  Components {x} (0), {y} (1), {p, q} (2).  Ready: {x}, {p, q}: x; then
      // {y} (1) before {p, q} (2).
      {"tie_break", {"x", "y", "p", "q"}, {{"x", "y"}, {"p", "q"}, {"q", "p"}}, {{"x"}, {"y"}, {"p", "q"}}},
      // a -> b -> c -> a (cycle), c -> d, d <-> e (cycle), f -> a.  Components {a, b, c}, {d, e}, {f}.  f precedes
      // {a, b, c}, which precedes {d, e}.  Ready at first: {f} only.
      {"two_cycles_and_a_feeder",
       {"a", "b", "c", "d", "e", "f"},
       {{"a", "b"}, {"b", "c"}, {"c", "a"}, {"c", "d"}, {"d", "e"}, {"e", "d"}, {"f", "a"}},
       {{"f"}, {"a", "b", "c"}, {"d", "e"}}},
      // n0 -> n1 -> n2 -> n0; n2 -> n3; n3 -> n4 -> n7 -> n6 -> n5 -> n3.  Components {n0, n1, n2} and
      // {n3, n4, n5, n6, n7} (every one of n3..n7 is on the cycle n3 n4 n7 n6 n5); the first precedes the second.
      {"long_cycle",
       {"n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"},
       {{"n0", "n1"}, {"n1", "n2"}, {"n2", "n0"}, {"n2", "n3"}, {"n3", "n4"}, {"n4", "n7"}, {"n7", "n6"}, {"n6", "n5"},
        {"n5", "n3"}},
       {{"n0", "n1", "n2"}, {"n3", "n4", "n5", "n6", "n7"}}},
      // three isolated nodes keep their order
      {"isolated", {"c", "a", "b"}, {}, {{"c"}, {"a"}, {"b"}}},
      // a chain listed in reverse: c -> b -> a with nodes a, b, c gives c, b, a
      {"reverse_chain", {"a", "b", "c"}, {{"c", "b"}, {"b", "a"}}, {{"c"}, {"b"}, {"a"}}},
      // two edges between one pair in the same direction are one dependence
      {"parallel_edges", {"a", "b"}, {{"a", "b"}, {"a", "b"}}, {{"a"}, {"b"}}},
  };
  for (const auto& c : cases) {
    Comps got;
    try {
      got = M::strongly_connected_components(c.nodes, c.edges);
    } catch (const std::exception& e) {
      fail(std::string("strongly_connected_components(") + c.name + ") threw: " + e.what());
      continue;
    }
    CHECK(got == c.expected, std::string(c.name) + ": components " + show(got) + ", expected " + show(c.expected));
  }
  // an edge to a node that is not listed
  bool threw = false;
  try {
    M::strongly_connected_components({"a"}, {{"a", "z"}});
  } catch (const std::invalid_argument&) {
    threw = true;
  } catch (const std::exception& e) {
    fail(std::string("unknown node: threw another exception: ") + e.what());
    threw = true;
  }
  CHECK(threw, "an edge naming a node that is not in `nodes` must be a std::invalid_argument");
}

// ================================================================================================ documents
// The channel of section 4.1: one instance, no term.
void check_single_node() {
  const auto r = M::validate_model(load("channel.json"));
  CHECK(r.errors.empty(), "channel: no error");
  CHECK(r.steps.size() == 1, "channel: one (implicit) step");
  const M::StepReport* s = step_of(r, "default");
  if (!s) return;
  CHECK(s->graph.step == "default", "channel: the graph's step is 'default'");
  CHECK(s->graph.nodes == Strs{"flow"}, "channel: nodes " + show(s->graph.nodes));
  CHECK(s->graph.edges.empty(), "channel: no edge");
  CHECK(s->graph.components == Comps{{"flow"}}, "channel: components " + show(s->graph.components));
  CHECK(s->graph.uncoupled.empty(), "channel: no uncoupled pair");
  CHECK(s->strategy == "one_way", "channel: an acyclic graph defaults to one_way, got '" + s->strategy + "'");
  CHECK(r.regions.count("flow") && r.regions.at("flow") == Strs{"channel"}, "channel: flow on [channel] (the default domain)");
}

// The heated cavity with the advection term alone: two singleton components, flow first.  This is the case that tells
// strongly from weakly connected components (one weak component).
json one_way_cavity() {
  json d = load("heated_cavity.json");
  d["couplings"] = J(R"j([{"term": "advection", "from": "flow", "to": "heat"}])j");
  d.erase("study");
  return d;
}

void check_one_way_pair() {
  const auto r = M::validate_model(one_way_cavity());
  CHECK(r.errors.empty(), "one-way pair: no error");
  const M::StepReport* s = step_of(r, "default");
  if (!s) return;
  CHECK(s->graph.nodes == (Strs{"flow", "heat"}), "one-way pair: nodes " + show(s->graph.nodes));
  CHECK(s->graph.components == (Comps{{"flow"}, {"heat"}}),
        "one-way pair: two singleton components, flow first; got " + show(s->graph.components));
  CHECK(s->graph.edges.size() == 1, "one-way pair: one edge");
  const M::GraphEdge* e = edge(s->graph, "flow", "heat", "term");
  CHECK(e != nullptr, "one-way pair: the term edge flow -> heat");
  if (e) {
    CHECK(e->term == "advection" && e->coupling == 0, "one-way pair: the edge is the advection term, couplings/0");
    CHECK(e->acts_in == Strs{"cavity"}, "one-way pair: advection acts in [cavity], got " + show(e->acts_in));
  }
  CHECK(s->strategy == "one_way", "one-way pair: the default strategy is one_way, got '" + s->strategy + "'");
  CHECK(s->routes.size() == 1 && s->routes[0].physics == (Strs{"flow", "heat"}) &&
            s->routes[0].route == "flow_then_heat_planar",
        "one-way pair: one group {flow, heat} on the route flow_then_heat_planar");
}

// The heated cavity of section 4.2: advection and buoyancy close a two-cycle.
void check_two_cycle_from_two_terms() {
  const auto r = M::validate_model(load("heated_cavity.json"));
  CHECK(r.errors.empty(), "cavity: no error");
  const M::StepReport* s = step_of(r, "steady");
  if (!s) return;
  CHECK(s->graph.components == (Comps{{"flow", "heat"}}), "cavity: one two-way group {flow, heat}; got " + show(s->graph.components));
  CHECK(s->graph.edges.size() == 2, "cavity: two edges");
  const M::GraphEdge* adv = edge(s->graph, "flow", "heat", "term");
  const M::GraphEdge* buo = edge(s->graph, "heat", "flow", "term");
  CHECK(adv && adv->term == "advection" && adv->coupling == 0, "cavity: advection flow -> heat, couplings/0");
  CHECK(buo && buo->term == "buoyancy" && buo->coupling == 1, "cavity: buoyancy heat -> flow, couplings/1");
  if (adv && buo) CHECK(s->graph.edges[0].term == "advection" && s->graph.edges[1].term == "buoyancy", "cavity: edges in the order of couplings");
  CHECK(s->strategy == "monolithic", "cavity: the written strategy monolithic, got '" + s->strategy + "'");
  CHECK(s->routes.size() == 1 && s->routes[0].route == "boussinesq_planar", "cavity: the route boussinesq_planar");
  // without the coupling block, the two-way group takes the route's default strategy
  json d = load("heated_cavity.json");
  d["study"][0].erase("coupling");
  const auto r2 = M::validate_model(d);
  CHECK(r2.errors.empty(), "cavity without a coupling block: no error (the route has a default strategy)");
  if (const M::StepReport* s2 = step_of(r2, "steady")) CHECK(s2->strategy == "monolithic", "cavity: the route's default strategy monolithic");
}

// The busbar of section 4.4: joule_heating current -> heat, thermal_expansion heat -> solid, and the conductivity law
// of copper (sigma of T) gives the property edge heat -> current, which closes {current, heat}; solid is downstream.
void check_two_cycle_closed_by_a_property_law() {
  const auto r = M::validate_model(load("busbar.json"));
  CHECK(r.errors.empty(), "busbar: no error");
  const M::StepReport* s = step_of(r, "steady");
  if (!s) return;
  CHECK(s->graph.nodes == (Strs{"current", "heat", "solid"}), "busbar: nodes " + show(s->graph.nodes));
  CHECK(s->graph.components == (Comps{{"current", "heat"}, {"solid"}}),
        "busbar: {current, heat} then {solid}; got " + show(s->graph.components));
  CHECK(s->graph.edges.size() == 3, "busbar: two term edges and one property edge");
  const M::GraphEdge* j = edge(s->graph, "current", "heat", "term");
  const M::GraphEdge* x = edge(s->graph, "heat", "solid", "term");
  const M::GraphEdge* p = edge(s->graph, "heat", "current", "property");
  CHECK(j && j->term == "joule_heating" && j->acts_in == (Strs{"bar", "bolt"}), "busbar: joule_heating current -> heat in [bar, bolt]");
  CHECK(x && x->term == "thermal_expansion" && x->acts_in == (Strs{"bar", "bolt"}), "busbar: thermal_expansion heat -> solid in [bar, bolt]");
  CHECK(p != nullptr, "busbar: the property edge heat -> current (kind property), not current -> heat");
  CHECK(edge(s->graph, "current", "heat", "property") == nullptr, "busbar: no reversed property edge");
  if (p) {
    CHECK(p->property == "electric.sigma_S_m" && p->material == "copper" && p->field == "T",
          "busbar: the property edge is copper's electric.sigma_S_m, a law of T");
    CHECK(p->acts_in == Strs{"bar"}, "busbar: the law acts in [bar] only (titanium's sigma is a constant), got " + show(p->acts_in));
    CHECK(p->term.empty() && p->coupling == -1, "busbar: a property edge has no term");
  }
  if (s->graph.edges.size() == 3)
    CHECK(s->graph.edges[0].kind == "term" && s->graph.edges[1].kind == "term" && s->graph.edges[2].kind == "property",
          "busbar: term edges first, in the order of couplings, then the property edge");
  CHECK(s->strategy == "gauss_seidel", "busbar: the written strategy gauss_seidel");
  CHECK(s->graph.uncoupled.empty(), "busbar: current and solid share the metal but no catalogue term connects their types");
  CHECK(r.regions.count("current") && r.regions.at("current") == (Strs{"bar", "bolt"}), "busbar: current on the domain metal = [bar, bolt]");
}

// Flow, heat and an electric current on one region: advection and buoyancy (flow <-> heat), Joule heating
// (current -> heat) and a conductivity law of T (heat -> current): one component of three.
json three_members() {
  return J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "three", "symmetry": "planar",
    "materials": {"melt": {"density": 7000.0, "fluid": {"mu_Pa_s": 0.005, "beta_1_K": 1e-4},
                           "thermal": {"k_W_mK": 30.0, "cp_J_kgK": 800.0},
                           "electric": {"sigma_S_m": {"law": "expression", "of": ["T"], "expr": "1e6/(1 + 1e-3*(T - 300))",
                                                      "range": {"T": [250.0, 1500.0]}, "source": "test"}}}},
    "regions": [{"name": "pool", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}, "material": "melt"}],
    "boundaries": {"left": {"of": "pool", "where": {"x": 0}}, "right": {"of": "pool", "where": {"x": 1}}},
    "physics": {
      "flow": {"type": "fluid.incompressible", "options": {"equation": "navier_stokes"},
               "boundary_conditions": {"remaining": {"kind": "wall"}}},
      "heat": {"type": "heat.transfer",
               "boundary_conditions": {"left": {"kind": "temperature", "T_K": 310}, "right": {"kind": "temperature", "T_K": 300},
                                       "remaining": {"kind": "insulation"}}},
      "current": {"type": "current.conduction",
                  "boundary_conditions": {"left": {"kind": "potential", "V_V": 0.01}, "right": {"kind": "ground"},
                                          "remaining": {"kind": "insulation"}}}
    },
    "couplings": [
      {"term": "advection", "from": "flow", "to": "heat"},
      {"term": "buoyancy", "from": "heat", "to": "flow", "T_ref_K": 305, "gravity_m_s2": [0, -9.81]},
      {"term": "joule_heating", "from": "current", "to": "heat"}
    ],
    "study": [{"id": "s", "type": "stationary", "coupling": {"strategy": "gauss_seidel"}}]
  })j");
}

void check_three_member_component() {
  const auto r = M::validate_model(three_members());
  CHECK(r.errors.empty(), "three members: no error (the combination has no route: unavailable, not wrong)");
  const M::StepReport* s = step_of(r, "s");
  if (!s) return;
  CHECK(s->graph.nodes == (Strs{"current", "flow", "heat"}), "three members: nodes in the order of the instance ids " + show(s->graph.nodes));
  CHECK(s->graph.components == (Comps{{"current", "flow", "heat"}}), "three members: one component of three; got " + show(s->graph.components));
  CHECK(s->graph.edges.size() == 4, "three members: three term edges and one property edge");
  CHECK(edge(s->graph, "heat", "current", "property") != nullptr, "three members: the property edge heat -> current");
  CHECK(edge(s->graph, "current", "heat", "term") != nullptr, "three members: joule_heating current -> heat");
  CHECK(with_code(r.unavailable, "route.not_implemented").size() == 1, "three members: one route.not_implemented (no row)");
  CHECK(s->routes.size() == 1 && s->routes[0].route.empty() && s->routes[0].physics == (Strs{"current", "flow", "heat"}),
        "three members: one group of three with no row");
}

// ------------------------------------------------------------------------------------------------ where a term acts
// A planar document with two regions of one fluid with a thermal block, 'a' and 'b', a flow and a heat instance with
// the given domains, and the advection term (with `in` when not empty).
json two_region_pair(const json& flow_domains, const json& heat_domains, const std::string& in) {
  json d = J(R"j({
    "schema": "axionmultiphysics.design/1", "name": "pair", "symmetry": "planar",
    "materials": {"water": {"density": 998.2, "fluid": {"mu_Pa_s": 1.002e-3}, "thermal": {"k_W_mK": 0.6, "cp_J_kgK": 4182.0}}},
    "regions": [{"name": "a", "shape": {"rect": {"x": [0, 1], "y": [0, 1]}}, "material": "water"},
                {"name": "b", "shape": {"rect": {"x": [1, 2], "y": [0, 1]}}, "material": "water"}],
    "boundaries": {"inlet": {"of": "a", "where": {"x": 0}}, "outlet": {"of": "b", "where": {"x": 2}}},
    "physics": {
      "flow": {"type": "fluid.incompressible", "options": {"equation": "stokes"},
               "boundary_conditions": {"inlet": {"kind": "inlet", "velocity_m_s": ["4*y*(1-y)", 0]},
                                       "outlet": {"kind": "outlet"}, "remaining": {"kind": "wall"}}},
      "heat": {"type": "heat.transfer",
               "boundary_conditions": {"inlet": {"kind": "temperature", "T_K": 300}, "outlet": {"kind": "outflow"},
                                       "remaining": {"kind": "temperature", "T_K": 310}}}
    },
    "couplings": [{"term": "advection", "from": "flow", "to": "heat"}]
  })j");
  if (!flow_domains.is_null()) d["physics"]["flow"]["domains"] = flow_domains;
  if (!heat_domains.is_null()) d["physics"]["heat"]["domains"] = heat_domains;
  if (!in.empty()) d["couplings"][0]["in"] = in;
  return d;
}

void check_acts_in() {
  // the intersection of the two domains
  {
    const auto r = M::validate_model(two_region_pair(J(R"(["a"])"), J(R"(["a", "b"])"), ""));
    if (const M::StepReport* s = step_of(r, "default")) {
      const M::GraphEdge* e = edge(s->graph, "flow", "heat", "term");
      CHECK(e && e->acts_in == Strs{"a"}, "acts_in: the intersection of [a] and [a, b] is [a]");
    }
    // the one-way row needs equal domains (checked in test_model_validate.cpp); here only that the term acts in [a]
    CHECK(with_code(r.errors, "coupling.empty_overlap").empty(), "acts_in: a non-empty intersection is not an empty overlap");
  }
  // `in` equal to the whole intersection: accepted on an equal-domains row
  {
    // both default domains are [a, b]; "in" naming both regions (through a domain) is the whole shared domain
    json d = two_region_pair(nullptr, nullptr, "");
    d["domains"] = J(R"({"both": ["a", "b"]})");
    d["couplings"][0]["in"] = "both";
    const auto r2 = M::validate_model(d);
    CHECK(r2.errors.empty(), "acts_in: \"in\" equal to the shared domain is accepted");
    if (const M::StepReport* s = step_of(r2, "default")) {
      const M::GraphEdge* e = edge(s->graph, "flow", "heat", "term");
      CHECK(e && e->acts_in == (Strs{"a", "b"}), "acts_in: in = both gives [a, b]");
    }
  }
  // no shared region: coupling.empty_overlap (AUTHOR TEXT; "no shared region" is the note's, section 3.7)
  {
    const auto r = M::validate_model(two_region_pair(J(R"(["a"])"), J(R"(["b"])"), ""));
    const auto e = with_code(r.errors, "coupling.empty_overlap");
    CHECK(e.size() == 1, "empty overlap: one coupling.empty_overlap");
    if (e.size() == 1) {
      CHECK(e[0].message() == "coupling 'advection' from 'flow' to 'heat': no shared region", "empty overlap: message: " + e[0].message());
      CHECK(e[0].where() == "/couplings/0", "empty overlap: where " + e[0].where());
      CHECK(e[0].hint() == "'flow' is on 'a' and 'heat' on 'b'; a domain term acts where both are. Give them a common region, "
                           "or remove the term.",
            "empty overlap: hint: " + e[0].hint());
      CHECK(details_of(e[0]) == J(R"({"term": "advection", "from": "flow", "to": "heat", "from_regions": ["a"], "to_regions": ["b"]})"),
            "empty overlap: details " + e[0].details());
    }
  }
  // a strict subset on a row that cannot restrict the term: coupling.restriction_not_offered (AUTHOR TEXT)
  {
    const auto r = M::validate_model(two_region_pair(nullptr, nullptr, "a"));
    const auto e = with_code(r.errors, "coupling.restriction_not_offered");
    CHECK(e.size() == 1, "restriction: one coupling.restriction_not_offered");
    if (e.size() == 1) {
      CHECK(e[0].message() == "coupling 'advection' from 'flow' to 'heat': the route 'flow_then_heat_planar' cannot restrict the term to 'a'",
            "restriction: message: " + e[0].message());
      CHECK(e[0].where() == "/couplings/0/in", "restriction: where " + e[0].where());
      CHECK(e[0].hint() == "On this route the term acts on the whole shared domain ('a', 'b'). Remove \"in\", or make it the whole "
                           "shared domain.",
            "restriction: hint: " + e[0].hint());
      CHECK(details_of(e[0]) == J(R"({"term": "advection", "route": "flow_then_heat_planar", "in": ["a"], "shared": ["a", "b"]})"),
            "restriction: details " + e[0].details());
    }
    CHECK(r.errors.size() == 1, "restriction: the only error");
    // the graph says where the term was asked to act: the part of the shared domain that "in" names
    if (const M::StepReport* s = step_of(r, "default")) {
      const M::GraphEdge* g = edge(s->graph, "flow", "heat", "term");
      CHECK(g && g->acts_in == Strs{"a"}, "restriction: acts_in is [a], the part that \"in\" names");
    }
    // the pointer is the entry's position in the document, also after an entry that the grammar dropped
    json d = two_region_pair(nullptr, nullptr, "a");
    d["couplings"] = J(R"j([{"term": "advection", "from": "nobody", "to": "nowhere"}, {"term": "advection", "from": "flow", "to": "heat", "in": "a"}])j");
    const auto e2 = with_code(M::validate_model(d).errors, "coupling.restriction_not_offered");
    CHECK(e2.size() == 1 && e2[0].where() == "/couplings/1/in", "restriction: the pointer after a dropped entry is /couplings/1/in");
  }
  // "in" restricts: a region outside the shared domain is refused (section 3.7: "checked to be a subset").  IMPLEMENTER TEXT.
  {
    json d = two_region_pair(J(R"(["a"])"), J(R"(["a"])"), "both");
    d["domains"] = J(R"({"both": ["a", "b"]})");
    const auto r = M::validate_model(d);
    const auto e = with_code(r.errors, "coupling.selection_outside_support");
    CHECK(e.size() == 1, "in beyond the shared domain: one coupling.selection_outside_support");
    if (e.size() == 1) {
      CHECK(e[0].message() == "coupling 'advection' from 'flow' to 'heat': \"in\": \"both\" is not inside the shared domain",
            "in beyond the shared domain: message: " + e[0].message());
      CHECK(e[0].where() == "/couplings/0/in", "in beyond the shared domain: where " + e[0].where());
      CHECK(e[0].hint() == "A domain term acts where both physics are, here 'a', and \"in\" restricts it to a part of that; 'both' also "
                           "names 'b'. Remove \"in\", or name regions of the shared domain.",
            "in beyond the shared domain: hint: " + e[0].hint());
      CHECK(details_of(e[0]) ==
                J(R"({"term": "advection", "from": "flow", "to": "heat", "in": ["a", "b"], "shared": ["a"], "outside": ["b"]})"),
            "in beyond the shared domain: details " + e[0].details());
    }
    CHECK(r.errors.size() == 1, "in beyond the shared domain: the only error (no restriction, no empty overlap)");
    // "in" wholly outside a shared domain that is not empty: the same finding, and not "no shared region"
    json out = two_region_pair(J(R"(["a"])"), J(R"(["a"])"), "b");
    const auto r2 = M::validate_model(out);
    CHECK(r2.errors.size() == 1 && with_code(r2.errors, "coupling.selection_outside_support").size() == 1,
          "in outside the shared domain: one coupling.selection_outside_support");
    CHECK(with_code(r2.errors, "coupling.empty_overlap").empty(), "in outside the shared domain: the two domains do share a region");
  }
  // "in" with a part of the shared domain and a region outside it: two independent defects, both reported (the term
  // cannot be restricted to 'a' on this row, and 'c' is no part of the shared domain)
  {
    json d = two_region_pair(J(R"(["a", "b"])"), J(R"(["a", "b"])"), "ac");
    d["regions"].push_back(J(R"({"name": "c", "shape": {"rect": {"x": [2, 3], "y": [0, 1]}}, "material": "water"})"));
    d["boundaries"]["outlet"] = J(R"({"of": "b", "where": {"x": 2}})");
    d["domains"] = J(R"({"ac": ["a", "c"]})");
    const auto r = M::validate_model(d);
    const auto outside = with_code(r.errors, "coupling.selection_outside_support");
    const auto restricted = with_code(r.errors, "coupling.restriction_not_offered");
    CHECK(r.errors.size() == 2 && outside.size() == 1 && restricted.size() == 1, "in across the shared domain: the two findings");
    if (outside.size() == 1)
      CHECK(details_of(outside[0])["outside"] == J(R"(["c"])") && details_of(outside[0])["shared"] == J(R"(["a", "b"])"),
            "in across the shared domain: 'c' is outside [a, b]: " + outside[0].details());
    if (restricted.size() == 1)
      CHECK(details_of(restricted[0])["in"] == J(R"(["a"])"), "in across the shared domain: the term would act in [a]: " + restricted[0].details());
  }
}

// ------------------------------------------------------------------------------------------------ uncoupled pairs
// "in" of a term is checked against the shared domain only for a group without a finding, and its hint reads the same
// whether some or all of the regions it names are outside.
void check_in_of_a_term_in_a_group_with_a_finding() {
  json d = two_region_pair(J(R"(["a"])"), J(R"(["a"])"), "b");
  const auto r = M::validate_model(d);
  const auto e = with_code(r.errors, "coupling.selection_outside_support");
  CHECK(e.size() == 1, "in outside the shared domain: refused, got " + std::to_string(e.size()));
  if (e.size() == 1)
    CHECK(e[0].hint() == "A domain term acts where both physics are, here 'a', and \"in\" restricts it to a part of that; 'b' is not in it. "
                         "Remove \"in\", or name regions of the shared domain.",
          "in disjoint from the shared domain: hint: " + e[0].hint());
  d["physics"]["heat"]["boundary_conditions"]["inlet"]["kind"] = "nonsense";
  const auto bad = M::validate_model(d);
  CHECK(!bad.errors.empty() && with_code(bad.errors, "coupling.selection_outside_support").empty(),
        "in outside the shared domain of a group with a finding: not refused again, got " + std::to_string(bad.errors.size()) + " errors");
}

void check_uncoupled_pair() {
  // flow and heat on one region with no term: a warning naming the candidate terms (section 3.7: "not an error: forced
  // convection without buoyancy is a valid model")
  json d = load("heated_cavity.json");
  d.erase("couplings");
  d.erase("study");
  const auto r = M::validate_model(d);
  CHECK(r.errors.empty(), "uncoupled: no error");
  if (const M::StepReport* s = step_of(r, "default")) {
    CHECK(s->graph.edges.empty(), "uncoupled: no edge");
    CHECK(s->graph.components == (Comps{{"flow"}, {"heat"}}), "uncoupled: two components " + show(s->graph.components));
    CHECK(s->graph.uncoupled.size() == 1, "uncoupled: one pair");
    if (s->graph.uncoupled.size() == 1) {
      const auto& p = s->graph.uncoupled[0];
      CHECK(p.a == "flow" && p.b == "heat", "uncoupled: the pair (flow, heat)");
      CHECK(p.regions == Strs{"cavity"}, "uncoupled: on [cavity]");
      CHECK(p.candidates == (Strs{"advection", "buoyancy"}), "uncoupled: candidates [advection, buoyancy], got " + show(p.candidates));
    }
    CHECK(s->routes.size() == 2, "uncoupled: two groups of one");
  }
  const auto w = with_code(r.warnings, "coupling.uncoupled_pair");
  CHECK(w.size() == 1, "uncoupled: one coupling.uncoupled_pair warning");
  if (w.size() == 1) {
    // AUTHOR TEXT
    CHECK(w[0].message() == "physics 'flow' and 'heat' both act on 'cavity', and no coupling term connects them: they are solved "
                            "uncoupled",
          "uncoupled: message: " + w[0].message());
    CHECK(w[0].where() == "/couplings", "uncoupled: where " + w[0].where());
    CHECK(w[0].hint() == "This is valid (forced convection without buoyancy, for example), and nothing is inferred. The terms that "
                         "can connect them are: advection, buoyancy. Add one to \"couplings\" if they interact.",
          "uncoupled: hint: " + w[0].hint());
    CHECK(details_of(w[0]) == J(R"({"physics": ["flow", "heat"], "regions": ["cavity"], "candidates": ["advection", "buoyancy"]})"),
          "uncoupled: details " + w[0].details());
  }
  // a pair whose types no catalogue term connects is not a candidate pair: a flow and a current distribution
  json fc = load("channel.json");
  fc["materials"]["water"]["electrolyte"] = J(R"({"kappa_S_m": 1.0})");
  fc["physics"]["cell"] = J(R"({"type": "electrochem.current_distribution", "options": {"kinetics": "primary"},
                                 "boundary_conditions": {"inlet": {"kind": "electrode", "phi_s_V": 0.1, "primary": true, "E_eq_V": 0.0},
                                                         "outlet": {"kind": "electrode", "phi_s_V": 0.0, "primary": true, "E_eq_V": 0.0},
                                                         "remaining": {"kind": "insulation"}}})");
  const auto r2 = M::validate_model(fc);
  CHECK(with_code(r2.warnings, "coupling.uncoupled_pair").empty(), "uncoupled: no warning for a flow and a current distribution");
  // two instances on disjoint regions, with no term, are not a pair
  json disjoint = two_region_pair(J(R"(["a"])"), J(R"(["b"])"), "");
  disjoint.erase("couplings");
  const auto r3 = M::validate_model(disjoint);
  CHECK(with_code(r3.warnings, "coupling.uncoupled_pair").empty(), "uncoupled: no warning for disjoint domains");
  // a pair connected by a property law alone is coupled: the busbar of section 4.4 without its terms keeps the edge
  // heat -> current of copper's sigma(T), so (current, heat) is no pair; (heat, solid) has no edge left and is one
  json laws = load("busbar.json");
  laws.erase("couplings");
  laws.erase("study");
  const auto r5 = M::validate_model(laws);
  if (const M::StepReport* s = step_of(r5, "default")) {
    CHECK(edge(s->graph, "heat", "current", "property") != nullptr, "property pair: the law's edge heat -> current is in the graph");
    CHECK(s->graph.uncoupled.size() == 1 && s->graph.uncoupled[0].a == "heat" && s->graph.uncoupled[0].b == "solid",
          "property pair: the only uncoupled pair is (heat, solid)");
  }
  const auto w5 = with_code(r5.warnings, "coupling.uncoupled_pair");
  CHECK(w5.size() == 1 && details_of(w5[0])["physics"] == J(R"(["heat", "solid"])"),
        "property pair: one warning, for (heat, solid); a pair that a law connects is not uncoupled");
  // a pair connected by one term in one direction is coupled
  const auto r4 = M::validate_model(one_way_cavity());
  CHECK(with_code(r4.warnings, "coupling.uncoupled_pair").empty(), "uncoupled: one term is a coupling");
}

}  // namespace

int main() {
  run("check_components_of_hand_made_graphs", check_components_of_hand_made_graphs);
  run("check_single_node", check_single_node);
  run("check_one_way_pair", check_one_way_pair);
  run("check_two_cycle_from_two_terms", check_two_cycle_from_two_terms);
  run("check_two_cycle_closed_by_a_property_law", check_two_cycle_closed_by_a_property_law);
  run("check_three_member_component", check_three_member_component);
  run("check_acts_in", check_acts_in);
  run("check_in_of_a_term_in_a_group_with_a_finding", check_in_of_a_term_in_a_group_with_a_finding);
  run("check_uncoupled_pair", check_uncoupled_pair);
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("test_model_graph: OK\n");
  return 0;
}
