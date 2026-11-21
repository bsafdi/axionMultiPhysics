// The coupling graph of a model document (beta): the instances of one study step as nodes, the coupling terms between
// them and the property dependences derived from the materials' laws as directed edges, its strongly connected
// components in topological order, and the pairs of instances that share a region with no edge between them.
//
// A component with more than one member is a two-way group, solved with the step's coupling strategy; the components
// are solved in the order given here.  An edge goes from the instance that provides a field (the producer) to the
// instance that receives a load, a coefficient or a property built from it (the consumer).
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "axionmultiphysics/spec_model.hpp"

namespace axmp::model {

// One directed edge of the coupling graph.
struct GraphEdge {
  std::string from, to;              // the producer and the consumer (instance ids)
  std::string kind;                  // "term": a coupling term of the document; "property": a law of a field
  std::string term;                  // kind "term": the term's name; "" otherwise
  int coupling = -1;                 // kind "term": the index of the entry in "couplings"; -1 otherwise
  std::string property;              // kind "property": the property written as a law ("electric.sigma_S_m")
  std::string material;              // kind "property": the material whose law it is
  std::string field;                 // kind "property": the field the law reads, which `from` provides ("T")
  std::vector<std::string> acts_in;  // the regions where the edge acts, in the order of the document's regions
};

// Two instances of a step that act on a common region, with no edge between them (in either direction), whose types
// the catalogue has at least one coupling term for.
struct UncoupledPair {
  std::string a, b;                     // instance ids, a < b
  std::vector<std::string> regions;     // the regions both act on, in the order of the document's regions
  std::vector<std::string> candidates;  // the catalogue's terms between their types (either direction), in the
                                        // catalogue's order
};

struct CouplingGraph {
  std::string step;                                  // the step's id
  std::vector<std::string> nodes;                    // the step's instances, in the order of its physics list
  std::vector<GraphEdge> edges;                      // the term edges in the order of "couplings", then the
                                                     // property edges in the order of the materials (by name), then
                                                     // of their properties
  std::vector<std::vector<std::string>> components;  // strongly connected components, in the order of
                                                     // strongly_connected_components(nodes, edges)
  std::vector<UncoupledPair> uncoupled;              // ordered by (a, b)
};

// The strongly connected components of a directed graph, in topological order: every edge between two components goes
// from an earlier one to a later one.  Where no edge orders two components, the one whose first member comes first in
// `nodes` comes first (repeatedly take, among the components all of whose predecessors are placed, the one whose first
// member has the smallest position in `nodes`).  Within a component the members are in the order of `nodes`.  A node
// with an edge to itself is a component of one.  An edge that names a node not in `nodes` is a std::invalid_argument.
std::vector<std::vector<std::string>> strongly_connected_components(
    const std::vector<std::string>& nodes, const std::vector<std::pair<std::string, std::string>>& edges);

// The coupling graph of `step` of `model`.  `regions` gives the regions of every instance (validate_model resolves the
// default domains of design note 3.6 on the route each instance takes).  A term edge acts on the intersection of its
// producer's and its consumer's regions, or on the regions of its "in" when written.  A property edge goes from the
// instance of the step that provides the field a law reads, on a region whose material has the law, to every other
// instance of the step that reads the property on that region; an instance never gets an edge from itself.
CouplingGraph coupling_graph(const ResolvedModel& model, const StudyStep& step,
                             const std::map<std::string, std::vector<std::string>>& regions);

}  // namespace axmp::model
