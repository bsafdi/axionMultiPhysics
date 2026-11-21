// The named boundaries of a model document resolved on an entity table (beta): the evaluation of the selectors, the
// atoms and their master numbers, and the partition rule of a physics instance (interface design note 5.2, 5.3, 5.5).
//
// Pure code on a table of the geometry's entities (the edges of a 2D model): no mesh, no geometry kernel.  The mesher
// classifies its fragmented geometry into a Table and calls resolve(); the library calls partition() and place() on the
// table and the resolution that the mesher wrote into its info file.  This header and its source use the standard library
// and nlohmann/json only, because axmp-mesher compiles them too; the source has no namespace-scope object with a
// constructor, since every electromagnetic mesh run starts that program.
//
// CONVENTIONS.
//   Coordinates: planar (x, y), axisymmetric (r, z) as components 0 and 1; 3D (x, y, z).
//   Direction of an entity: `tangent` is the direction in which the entity's line elements are written in the mesh file;
//     the mesher writes them along it.  For an entity without a marker it is the direction of the entity's
//     parametrization.  For an entity that carries markers it is the direction in which the first of `curves` is
//     written, and every other marker written against it is listed in `curves_reversed` (so the list is empty for an
//     entity with one marker).  side[0] is the region on the left of the tangent (the side into which the tangent rotated
//     by +90 degrees points), side[1] the region on its right; a counter-clockwise exterior boundary has its region at
//     side[0] and "exterior" at side[1].  A marker inside one region has side[0] == side[1].
//   Orientation of a selected entity: +1 when the normal of the selection (note 5.2) points from side[0] into side[1],
//     that is, when it is the tangent rotated by -90 degrees, (t_y, -t_x) (the normal MFEM computes for a boundary
//     segment written along the tangent); -1 when it points from side[1] into side[0].  Per selector:
//       of D               outward from D: +1 when side[0] is in D;
//       between [A, B]     from the side in A to the side in B: +1 when side[0] is in A;
//       curve NAME         the tangent as written rotated by +90 degrees (to the left of the written direction): -1,
//                          and +1 for a marker in curves_reversed (written against the entity's tangent);
//       ref NAME           as the boundary NAME;  except: as the selector it filters;
//       difference         as its first operand;  union: as the first operand, in written order, that selects the entity.
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace axmp::model::selection {

struct Entity {
  int tag = 0;                        // the mesher's entity tag; meaningful within one info file only
  int dim = 1;                        // 1: an edge of a 2D model (2: a face of a 3D one)
  std::string side[2];                // the regions on its two sides, "exterior" for none: side[0] on the left of the
                                      // tangent, side[1] on its right; side[0] == side[1] for a marker inside one region
  std::vector<std::string> curves;    // the curve markers it carries (names of "curves")
  std::vector<std::string> curves_reversed;   // those of `curves` written against `tangent` (never the first; see
                                              // CONVENTIONS)
  bool axis = false;                  // axisymmetric: every point on r = 0 (within the tolerance)
  std::string type;                   // "line", "circle", "ellipse"; "mesh" for an entity of an imported mesh
  double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};   // the exact bounding box (never one inflated by a geometry kernel)
  double point[3] = {0, 0, 0};        // the parametric midpoint
  double tangent[3] = {0, 0, 0};      // at the midpoint: the direction of its line elements (see CONVENTIONS)
  double length_m = 0.0, area_m2 = 0.0;          // area_m2: revolved (axisymmetric), 0 planar
};

struct Table {
  std::string geometry;               // "planar" | "axisymmetric" | "3d"
  std::vector<std::string> regions;   // enabled regions, document order
  std::vector<Entity> entities;
  double diagonal_m = 0.0;            // of the union of the regions; the default where-tolerance is 1e-9 of it
};

// The boundary entities are the entities with two different sides, the entities with a side "exterior", the entities
// that carry a marker, and the axis entities; any other entity of the table takes part in nothing below.
//
// A selector in the mesher's input form: domains expanded to region lists, expressions evaluated, a name of another
// boundary written {"ref": NAME}.
//   {"of": [R...]}                      the boundary entities with exactly one side in [R...]; never an axis entity
//   {"between": [[R...], [R...]]}       one side in A and the other in B; B may be the string "exterior"; never an
//   {"between": [[R...], "exterior"]}   axis entity
//   {"curve": NAME}                     the entities that carry the marker NAME, whatever their sides; never an axis
//                                       entity (a marker that lies only on r = 0 selects nothing)
//   {"union": [S...]}, {"difference": [S, T...]}, {"ref": NAME}     set operations on selector objects
//   of and between take the filters
//     "where": {"x": [v] | [lo, hi], ...}   keep an entity whose WHOLE box satisfies every constraint within the tolerance:
//                                           v - tol <= lo[c] and hi[c] <= v + tol (equality), lo - tol <= lo[c] and
//                                           hi[c] <= hi + tol (closed range)
//     "tolerance_m": t                      the tolerance; else 1e-9 x Table::diagonal_m
//     "except": [NAME...]                   remove the entities of the named boundaries
//   An entity of the selection before its where filter (except applied) that is not kept is a PARTIAL match when every
//   constraint is either satisfied or overlaps its interval by more than the tolerance (min(hi[c], hi) - max(lo[c], lo) >
//   tol), and at least one is of the second kind.  An equality is never partly satisfied.  A straight entity (type "line") is decided exactly: it is a partial match when no equality is left unsatisfied and the
//   segment (the midpoint -/+ tangent * length / 2), cut at the bounds of every constraint that the edge's box does not
//   already satisfy, has a part longer than the tolerance left (a constraint that the box satisfies does not cut it, so an
//   edge that only touches the filter's box is not in part inside it).  A curved entity is decided by its box, with one exception: an entity whose box
//   encloses the filter's box strictly in every coordinate of the geometry (each coordinate constrained by a range, with
//   lo - tol > the box's lo and hi + tol < the box's hi) is not a partial match, because the box cannot tell whether the
//   curve passes through the filter's box and a closed curve runs around it (the outer circle of an annulus around a filter
//   drawn on its inner circle).
struct Atom {
  int master = 0;                     // 900 axis; 2001, 2002, ...
  std::string kind;                   // "axis" | "named" | "unnamed"
  std::vector<std::string> names;     // sorted; empty for unnamed and axis
  std::string side[2];                // unnamed: its region pair in pair order; named and axis: the sides of its first
                                      // entity in the order of Table::entities
  std::vector<int> tags;              // sorted
};

// The atom of a boundary entity is its membership signature, the sorted list of the boundary names that select it; an
// entity that no name selects has the unnamed atom of its region pair; every axis entity is in the axis atom.  Master
// numbers: the axis 900; the named atoms 2001, 2002, ... in the lexicographic order of their sorted name lists (compared
// as lists of strings); then the unnamed atoms in the order of their region pair (i, j), i <= j, the indices those of
// Table::regions and "exterior" after every region.  The numbering depends on the names and the regions only.
struct Resolution {
  std::map<std::string, std::vector<int>> selected;          // every boundary name -> its entity tags, sorted (may be
                                                             // empty)
  std::map<std::pair<std::string, int>, int> orientation;   // (name, tag) -> +1 / -1 for every selected entity
                                                             // (CONVENTIONS)
  // The FILTERED SELECTORS of a boundary: its of and between selectors with a where filter that add entities to it, that
  // is, the selector itself, an operand of a union and the first operand of a difference, at any depth (not through
  // ref: the boundary named reports its own).  The PARTIAL MATCHES of a boundary: the partial matches of its filtered
  // selectors that it does not select and that no later operand of an enclosing difference removes.
  //
  // Per empty selection: {"where": the where filter in input form, or null;
  //                       "candidates": [...] (below); "partial": [tags]; "axis_only": bool}.
  //   where and candidates: when the boundary has exactly one filtered selector and its where filter keeps nothing, that
  //     filter, and the entities of that selector before the filter, grouped by line: the entities whose box has
  //     hi[c] - lo[c] <= tol in coordinate c (the first such c) and whose lo[c] lie within tol of the group's first
  //     (smallest) value form one group, {"where": {"<c>": that value}, "measure_m": <the sum of their lengths, in
  //     ascending tag order>}, the groups sorted by coordinate, then by value; then every other entity as
  //     {"bbox": {"<c>": [lo[c], hi[c]], ...}, "measure_m": length}, sorted by point.  Otherwise null and [].
  //   partial: the boundary's partial matches, sorted.
  //   axis_only: the selection is empty and would not be if axis entities could be selected.
  std::map<std::string, nlohmann::json> empty;
  // Per non-empty selection with partial matches: {"tags": [tags, sorted]}.
  std::map<std::string, nlohmann::json> partial;
  // Per between whose two domains share regions (refused at validation): the shared regions, in Table::regions order.
  std::map<std::string, std::vector<std::string>> overlap;
  std::vector<Atom> atoms;                                   // in master order
  std::map<int, int> atom_of_tag;                            // every boundary entity -> its atom's master number
};

// `selectors`: an object, boundary name -> selector in the input form.  resolve is called on documents without grammar
// errors only (a cycle of names is refused by the parser).
Resolution resolve(const Table&, const nlohmann::json& selectors);

// The partition rule of one instance with the regions I and the condition keys K (boundary names and "remaining").
// Its DOMAIN BOUNDARY is the set of boundary entities, not on the axis, with exactly one side in I.  A key covers the
// domain-boundary entities its boundary selects; "remaining" covers the domain boundary minus every other key of the
// instance (named atoms that no key of this instance names included; never the axis).
struct Partition {
  std::map<std::string, std::vector<int>> atoms_of_key;   // every key of K -> the master atoms of the entities it
                                                          // covers, sorted ("remaining" resolved; may be empty)
  std::vector<int> axis_atoms;                            // the axis atoms with an entity that has a side in I
  std::vector<int> uncovered;                             // domain-boundary entities that no key covers, sorted
  std::map<std::string, std::vector<int>> interior_keys_tags;   // key -> its selected entities with both sides in I
                                                                // (physics.boundary_interior, also when only part of
                                                                // the key is interior)
  std::vector<std::string> outside_keys;                  // keys with a non-empty selection that has nothing on the
                                                          // domain boundary and nothing inside (in K order)
  std::map<std::string, std::vector<int>> outside_part;   // key (not in outside_keys) -> its selected entities with no
                                                          // side in I: taken nowhere by this instance
  std::map<int, std::vector<std::string>> doubly;         // tag -> the keys covering it, when two or more (in K order)
};
Partition partition(const Table&, const Resolution&, const std::vector<std::string>& regions,
                    const std::vector<std::string>& keys);

// Where the selection `name` lies with respect to an instance (its regions, and the partition of its keys): for the
// output checks (a force on an interior selection, an output outside the domain, the Laplacian-form force rule on a
// boundary that is not a key).
struct Placement {
  std::vector<int> boundary;          // its entities on the domain boundary, sorted
  std::vector<int> interior;          // its entities with both sides in the instance, sorted
  std::vector<int> outside;           // its entities with no side in the instance, sorted
  std::vector<std::string> keys;      // the keys of the partition that cover one of its domain-boundary entities, sorted
};
Placement place(const Table&, const Resolution&, const Partition&, const std::vector<std::string>& regions,
                const std::string& name);

// The table as JSON and back; every double survives bit for bit.  The keys: "geometry", "regions", "entities",
// "diagonal_m"; per entity "tag", "dim", "side" [2], "curves", "axis", "type", "lo", "hi", "point", "tangent" [3 each],
// "length_m", "area_m2", and "curves_reversed" only for an entity that has one.  table_from_json ignores keys it does
// not know (the info file adds "atom" to each entity).
nlohmann::json to_json(const Table&);
Table table_from_json(const nlohmann::json&);

}  // namespace axmp::model::selection
