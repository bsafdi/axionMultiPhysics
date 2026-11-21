// The coupling graph of a model document: see axionmultiphysics/model_graph.hpp.
#include "axionmultiphysics/model_graph.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "axionmultiphysics/physics_registry.hpp"
#include "model_reads.hpp"

namespace axmp::model {

using Strs = std::vector<std::string>;

// Tarjan's algorithm for the components, then a topological order of the condensation in which, among the components
// whose predecessors are all placed, the one whose first member is earliest in `nodes` comes first.
std::vector<Strs> strongly_connected_components(const Strs& nodes, const std::vector<std::pair<std::string, std::string>>& edges) {
  const int n = static_cast<int>(nodes.size());
  std::map<std::string, int> position;
  for (int i = 0; i < n; ++i) position.emplace(nodes[i], i);
  std::vector<std::vector<int>> next(n);
  for (const auto& e : edges) {
    const auto a = position.find(e.first), b = position.find(e.second);
    if (a == position.end() || b == position.end())
      throw std::invalid_argument("strongly_connected_components: the edge " + e.first + " -> " + e.second +
                                  " names a node that is not in the list of nodes");
    next[a->second].push_back(b->second);
  }

  std::vector<int> index(n, -1), low(n, 0), component(n, -1), stack;
  std::vector<char> on_stack(n, 0);
  int counter = 0, ncomp = 0;
  std::function<void(int)> visit = [&](int v) {
    index[v] = low[v] = counter++;
    stack.push_back(v);
    on_stack[v] = 1;
    for (int w : next[v]) {
      if (index[w] < 0) {
        visit(w);
        low[v] = std::min(low[v], low[w]);
      } else if (on_stack[w]) {
        low[v] = std::min(low[v], index[w]);
      }
    }
    if (low[v] == index[v]) {
      for (;;) {
        const int w = stack.back();
        stack.pop_back();
        on_stack[w] = 0;
        component[w] = ncomp;
        if (w == v) break;
      }
      ++ncomp;
    }
  };
  for (int v = 0; v < n; ++v)
    if (index[v] < 0) visit(v);

  // the members of each component in the order of `nodes`; members[c][0] is its first member
  std::vector<std::vector<int>> members(ncomp);
  for (int v = 0; v < n; ++v) members[component[v]].push_back(v);
  // the condensation: an edge between two components, counted once
  std::vector<std::vector<int>> out(ncomp);
  std::vector<int> waiting(ncomp, 0);
  for (int v = 0; v < n; ++v)
    for (int w : next[v]) {
      const int a = component[v], b = component[w];
      if (a == b || std::find(out[a].begin(), out[a].end(), b) != out[a].end()) continue;
      out[a].push_back(b);
      ++waiting[b];
    }
  std::vector<Strs> result;
  std::vector<char> placed(ncomp, 0);
  for (int k = 0; k < ncomp; ++k) {
    int best = -1;
    for (int c = 0; c < ncomp; ++c)
      if (!placed[c] && waiting[c] == 0 && (best < 0 || members[c][0] < members[best][0])) best = c;
    // a condensation is acyclic, so a component with no unplaced predecessor always exists
    placed[best] = 1;
    for (int b : out[best]) --waiting[b];
    Strs names;
    for (int v : members[best]) names.push_back(nodes[v]);
    result.push_back(std::move(names));
  }
  return result;
}

CouplingGraph coupling_graph(const ResolvedModel& model, const StudyStep& step, const std::map<std::string, Strs>& regions) {
  using detail::contains;
  const Registry& R = registry();
  CouplingGraph g;
  g.step = step.id;
  g.nodes = step.physics;
  const auto regions_of_instance = [&](const std::string& id) -> Strs {
    const auto it = regions.find(id);
    return it == regions.end() ? Strs() : it->second;
  };

  // the coupling terms whose two ends the step solves, in the order of "couplings"
  for (const Coupling& c : model.couplings) {
    if (!contains(g.nodes, c.from) || !contains(g.nodes, c.to)) continue;
    GraphEdge e;
    e.from = c.from;
    e.to = c.to;
    e.kind = "term";
    e.term = c.term;
    e.coupling = c.index;
    e.acts_in = detail::intersection(regions_of_instance(c.from), regions_of_instance(c.to));
    if (!c.in.empty()) e.acts_in = detail::intersection(e.acts_in, detail::regions_of(model, {c.in}));
    g.edges.push_back(std::move(e));
  }

  // the property dependences: a law of a field, from the instance that provides the field on a region of the material
  // to every other instance that reads the property there.  A field that the step gives a fixed value is no dependence.
  for (const auto& [material, props] : model.materials) {
    Strs of_material;
    for (const auto& r : model.regions)
      if (r.material == material) of_material.push_back(r.name);
    if (of_material.empty()) continue;
    for (const detail::PropertyRef& p : detail::properties()) {
      const Property& value = props.*(p.member);
      if (!value.is_law()) continue;
      for (const std::string& field : value.law->of) {
        if (step.given.count(field)) continue;
        for (const std::string& producer : g.nodes) {
          const auto pi = model.instances.find(producer);
          const PhysicsType* pt = pi == model.instances.end() ? nullptr : R.type(pi->second.type);
          if (!pt || !detail::provides(*pt, field)) continue;
          const Strs where = detail::intersection(of_material, regions_of_instance(producer));
          for (const std::string& consumer : g.nodes) {
            if (consumer == producer) continue;
            const auto ci = model.instances.find(consumer);
            if (ci == model.instances.end() || !detail::reads(model, step, ci->second, p)) continue;
            const Strs acts = detail::intersection(where, regions_of_instance(consumer));
            if (acts.empty()) continue;
            GraphEdge e;
            e.from = producer;
            e.to = consumer;
            e.kind = "property";
            e.property = p.name();
            e.material = material;
            e.field = field;
            e.acts_in = acts;
            g.edges.push_back(std::move(e));
          }
        }
      }
    }
  }

  std::vector<std::pair<std::string, std::string>> pairs;
  for (const GraphEdge& e : g.edges) pairs.emplace_back(e.from, e.to);
  g.components = strongly_connected_components(g.nodes, pairs);

  // the pairs that share a region with no edge between them, for which the catalogue has a term
  Strs sorted = g.nodes;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 0; i < sorted.size(); ++i)
    for (size_t j = i + 1; j < sorted.size(); ++j) {
      const std::string& a = sorted[i];
      const std::string& b = sorted[j];
      if (a == b) continue;
      bool connected = false;
      for (const GraphEdge& e : g.edges) connected = connected || (e.from == a && e.to == b) || (e.from == b && e.to == a);
      if (connected) continue;
      const Strs shared = detail::intersection(regions_of_instance(a), regions_of_instance(b));
      if (shared.empty()) continue;
      const auto ia = model.instances.find(a), ib = model.instances.find(b);
      if (ia == model.instances.end() || ib == model.instances.end()) continue;
      UncoupledPair pair;
      for (const CouplingTermType& t : R.terms)
        if ((type_matches(t.producers, ia->second.type) && type_matches(t.consumers, ib->second.type)) ||
            (type_matches(t.producers, ib->second.type) && type_matches(t.consumers, ia->second.type)))
          pair.candidates.push_back(t.name);
      if (pair.candidates.empty()) continue;
      pair.a = a;
      pair.b = b;
      pair.regions = shared;
      g.uncoupled.push_back(std::move(pair));
    }
  return g;
}

}  // namespace axmp::model
