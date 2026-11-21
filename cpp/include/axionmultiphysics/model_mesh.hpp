// The meshed model of a model document and the instance mesh of a physics instance (beta; interface design note 5.4).
//
// The mesher meshes the union of the enabled regions into one MASTER mesh whose boundary elements carry the master
// numbers of the atoms (model_selection.hpp: the axis 900, the named atoms 2001, 2002, ..., then the unnamed atoms).
// No engine receives the master mesh.  An engine receives the INSTANCE MESH of the regions of a physics instance: a plain
// serial mfem::Mesh of the elements of those regions, whose boundary elements are exactly its faces with one adjacent
// element (the domain boundary of the instance and its axis faces), numbered densely 1..n.
//
// CONVENTIONS.
//   Dense numbering: the atoms present on the instance mesh's boundary elements, sorted by master number (so the axis,
//     900, comes first where it is present), numbered 1, 2, ..., n.  Element (region) attributes are not renumbered:
//     they are the master's, 1..R in the document order of the enabled regions.
//   Faces: the faces of the instance mesh in MFEM's numbering (in 2D its edges).  The NORMAL OF A FACE is the normal
//     MFEM computes for it: for a face written from vertex v0 to v1 (tangent t), (t_y, -t_x) / |t|, which points out of
//     the face's first element (Elem1 of GetFaceElements).
//   Orientation of a selection (design note 5.2): on the instance's boundary the normal points out of the instance; on
//     a face interior to the instance (a SECTION) it is the orientation of the selection: from A to B for between [A, B],
//     outward from D for of D, the written tangent rotated by +90 degrees for a curve (model_selection.hpp,
//     CONVENTIONS).  The sign of an interior face is taken from the MASTER mesh's boundary element on that face (the
//     direction in which the mesh file writes it, which MFEM keeps for interior boundary elements) and the resolution's
//     orientation of its entity, never from the instance mesh's own boundary elements or faces, whose vertex order is
//     MFEM's.
//
// Errors.  A defect of the document is an axmp::Error with a code (model_errors.hpp); a mesher run that fails is
// mesh.generation_failed.  A misuse of this interface (an
// unknown region or boundary name, an attribute that is not on the instance mesh) is a std::invalid_argument, which the
// Python binding raises as ValueError.  An exporter asked for what a valid document never asks for raises the internal
// error internal.mesh_export (an axmp::Error), which asks for a bug report.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionmultiphysics/chem/current_distribution.hpp"
#include "axionmultiphysics/model_selection.hpp"
#include "axionmultiphysics/model_validate.hpp"

namespace axmp::model {

struct MeshOptions {
  std::string workdir;    // where the mesher's input, mesh, info file and log are written (created if missing); "":
                          // a temporary directory, removed before the call returns (mesh_file and info_file are then "")
  std::string mesher;     // a path that axi::find_mesher tries first; "" for its search
  int verbose = 0;
};

// Whether the mesher program loads: it is run with no argument; exit status 2 (its usage text) means it loaded.
// Any other status, or no executable, means it is not available, and `why` says why: the loader's message from the
// probe's log for status 127, else the status and the log.  `path` is the resolved path ("" when none was found).  The
// result is cached per resolved path for the life of the process.
struct MesherStatus {
  bool available = false;
  std::string path, why;  // why: "" when available
};
MesherStatus mesher_status(const std::string& hint = "");

// The input of the mesher's bounded mode for a validated model document (report.ok()):
//   "bounded": true, "geometry": the symmetry ("planar" | "axisymmetric"), "exterior": mesh.exterior,
//   "regions": the enabled regions in document order, each {"name", "attr" (1..R in that order), "shape" (the resolved
//     shape, as resolved_model_geometry writes it), "size" and "surface_size" (each only when the region gives it)},
//   "curves": the curves in document order, each {"name", "path" (resolved)} plus "size" / "surface_size" when given,
//   "boundaries": every boundary name -> its selector in the mesher's input form (model_selection.hpp): every domain
//     name expanded to its enabled regions (in document order), a region name to a list of one, "where" values as
//     [v] or [lo, hi], "tolerance_m" and "except" only when written, a name written {"ref": NAME},
//   "size": mesh.size, else 1/40 of the diagonal of the bounding box of the enabled regions,
//   "surface_size": mesh.surface_size, else "size",  "curvature_points": 24,  "order": report.geometry_order,
//   "resolve_only", "output": workdir + "/mesh.msh", "info": workdir + "/mesh_info.json", "verbose": 0.
nlohmann::json bounded_mesher_input(const ValidationReport& report, bool resolve_only, const std::string& workdir);

// The resolve_only run of the mesher: the entity table of the geometry, the resolution of the document's named boundaries
// on it, the regions' attributes and the info file as the mesher wrote it.
struct BoundaryResolution {
  selection::Table table;
  selection::Resolution resolution;
  std::map<std::string, int> region_attribute;   // every enabled region -> its element attribute (1..R)
  nlohmann::json info;
};
BoundaryResolution resolve_boundary_table(const ValidationReport& report, const MeshOptions& options);

// The resolution of the document's named boundaries on its geometry (resolve_boundary_table's resolution).
selection::Resolution resolve_boundaries(const ValidationReport& report, const MeshOptions& options);

// The faces of a selection interior to an instance (a section), with the orientation of design note 5.2.
struct FaceSet {
  std::string name;
  std::vector<int> faces;   // instance-mesh face indices, ascending
  std::vector<int> sign;    // per face: +1 when the selection's normal is the face's normal (CONVENTIONS), else -1
  double length_m = 0.0;    // the sum of the lengths of the faces on the instance mesh (its geometry, curved or not)
};

// Quadrature on the faces of a boundary or a section of an instance (the flux instrument).  m faces, in ascending
// order of their instance-mesh face index; q points per face, those of mfem::IntRules.Get(Geometry::SEGMENT, order)
// (exact for polynomials of degree `order` in the face's parameter); arrays row major, face slowest, component fastest.
struct FacePoints {
  std::vector<int> face;        // (m): the instance-mesh face index, ascending
  std::vector<int> element;     // (m): the adjacent element of the face in the instance mesh: its only element on the
                                // boundary, its first element (Elem1) on a section
  std::vector<double> ref;      // (m, q, 2): the reference coordinates of the points in that element
  std::vector<double> weight;   // (m, q): the quadrature weight times the face Jacobian (they sum to the face's length)
  std::vector<double> normal;   // (m, q, 2): the unit normal with the orientation of CONVENTIONS
  std::vector<double> x;        // (m, q, 2): the physical points
};

// The mesh of one physics instance.  It owns everything it needs: no pointer or reference to the MeshedModel or to its
// master mesh (its section and face_points work after the MeshedModel is destroyed).
struct InstanceMesh {
  std::vector<std::string> regions;                       // the instance's regions, in the document order
  std::unique_ptr<mfem::Mesh> mesh;                       // a plain serial mfem::Mesh (not an mfem::SubMesh)
  int geometry_order = 1;                                 // 1: mesh->GetNodes() == nullptr; 2: nodes of order 2
  std::map<std::string, std::vector<int>> attributes;     // every boundary name of the model -> the instance attributes
                                                          // of its atoms on this instance's boundary, ascending (empty
                                                          // when it has none: interior, or off the instance)
  std::vector<int> remaining_attributes, axis_attributes; // the unnamed atoms; the axis atom (ascending)
  std::map<int, int> master_attribute;                    // every instance attribute 1..n -> its master number
  std::map<std::string, int> region_attribute;            // the instance's regions -> their element attributes
  std::vector<int> element_to_master;                     // per instance element: the master element index
  std::vector<int> face_to_master;                        // per instance face: the master face index
  // What section, face_points and names_of read, filled by MeshedModel::instance_mesh: every boundary name of the model
  // -> its faces interior to this instance with their signs (CONVENTIONS; no face when it has none), and every instance
  // attribute -> the boundary names of its atom, sorted.
  std::map<std::string, FaceSet> interior_faces;
  std::map<int, std::vector<std::string>> attribute_names;

  // The faces of `name` interior to this instance (both sides in it), signed as in CONVENTIONS.  An unknown name, or a
  // name with no interior face on this instance, is a std::invalid_argument.
  FaceSet section(const std::string& name) const;
  // The faces of `name` on this instance, on its boundary (outward normals) and interior to it (the section's normals).
  // An unknown name, or a name with no face on this instance, is a std::invalid_argument.
  FacePoints face_points(const std::string& name, int order) const;
  // The faces of one boundary attribute of this instance (1..n), outward normals: the way to reach an unnamed atom
  // (remaining_attributes) or the axis.  An attribute that is not on this instance mesh (0, n + 1, a master number) is
  // a std::invalid_argument.
  FacePoints face_points(int attribute, int order) const;
  // The boundary names of the atom of an instance attribute, sorted (empty for an unnamed atom and the axis); an
  // attribute that is not on this instance mesh is a std::invalid_argument.
  std::vector<std::string> names_of(int instance_attribute) const;
};

// The master mesh with its table and resolution.  physics_registry.hpp forward-declares it (RouteFactory).
struct MeshedModel {
  std::string symmetry, mesh_file, info_file;             // symmetry: "planar" | "axisymmetric"
  int geometry_order = 2;                                 // 1: the master has no nodes
  std::unique_ptr<mfem::Mesh> master;                     // refinements applied; boundary attributes: master numbers
  selection::Table table;
  selection::Resolution resolution;
  std::map<std::string, int> region_attribute;            // every enabled region -> its element attribute (1..R)
  std::vector<int> boundary_entity;                       // per boundary element of `master`: the tag of its entity in
                                                          // `table` (the elementary tag the mesh file gives it); after
                                                          // refinements, each child has its parent's
  nlohmann::json info;                                    // the info file as read (null when built in memory)

  // The instance mesh of `regions` (names of enabled regions, any order, at least one).  An empty list or an unknown
  // name is a std::invalid_argument.  A boundary face of the instance with no master boundary element (a missing atom;
  // MFEM's SubMesh would give it the fabricated attribute max + 1) is the internal error internal.mesh_export, whose
  // message names the face and says that an atom is missing.
  InstanceMesh instance_mesh(const std::vector<std::string>& regions) const;
};

// Runs the mesher on bounded_mesher_input(report, false, options.workdir), then read_meshed_model on its files with the
// document's mesh.refinements.
MeshedModel mesh_model(const ValidationReport& report, const MeshOptions& options);

// Reads the mesher's files: the master mesh (MSH 2.2) with `refinements` uniform refinements, boundary_entity from the
// elementary tags of its dimension-1 elements (in the order MFEM reads them), the table from the info file ("geometry",
// "regions" {name: {"attr", ...}}, "entities", "diagonal_m"), and the resolution recomputed with selection::resolve on
// that table and the selectors of bounded_mesher_input(report, ...); an atom of the recomputed resolution that differs
// from the info file's "boundary_groups", or a boundary element whose attribute is not the atom of its entity, is the
// internal error internal.mesh_export.  The report gives the selectors: every rank of a parallel run has it.
MeshedModel read_meshed_model(const ValidationReport& report, const std::string& mesh_file, const std::string& info_file,
                              int refinements);

// The vertex, triangle and edge-node arrays of chem::MeshInput for an instance mesh, in the instance numbering: the
// vertices, triangles, element attributes, boundary segments and their attributes of im.mesh, in its order; for
// field_order 2, the P2 edge nodes that differ from their edge's chord midpoint by more than 1e-12 times the edge's
// length (none on a straight mesh).  An instance mesh with nodes and field_order 1 is the internal error
// internal.mesh_export (a curved mesh is not straightened for a P1 field), with the message
//   chemistry mesh export: the instance mesh of <regions> has nodes (geometry order 2), and the field order is 1
// (<regions>: 'a', or 'a' and 'b', or 'a', 'b' and 'c') and a hint that ends by asking for a bug report ("... please
// report it with the document.").
chem::MeshInput chem_mesh_input(const InstanceMesh& im, int field_order);

// Returns when the instance mesh has no nodes; otherwise the internal error internal.mesh_export, for a route whose
// options declare max_geometry_order 1 (the caller decides that; this function does not read the table), with the
// message
//   route '<route>' needs straight elements, and the instance mesh of <regions> has nodes (geometry order 2)
// and a hint that asks for a bug report ("... please report it with the document.").
void require_straight(const InstanceMesh& im, const std::string& route);

}  // namespace axmp::model
