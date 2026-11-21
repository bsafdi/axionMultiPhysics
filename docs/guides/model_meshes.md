# Meshes of model documents (beta)

A planar or axisymmetric [model document](../design_format.md#model-documents-beta) with regions can be meshed in this
version, and its named boundaries can be listed and checked. Nothing solves it yet: this is the geometry half of the
model path. 3D model documents and model documents on an imported mesh (`mesh.file`) are refused with
`route.not_implemented`.

The mesher is the separate program `axmp-mesher` (it runs gmsh). Where it does not run (a wheel installed without the
`mesh` extra, or without the X11/OpenGL libraries gmsh needs), `axmp validate` lists the unavailable finding
`build.requires_mesher` first, with the install hint, and the commands below refuse the document with it. A document without a
geometry (a reactor) or on an imported mesh needs no mesher.

## A split rectangle

```json
{
  "schema": "axionmultiphysics.design/1", "name": "split_rect", "symmetry": "planar",
  "parameters": {"L": 0.04, "H": 0.01},
  "materials": {"slab": {"density": 998, "thermal": {"k_W_mK": 0.6, "cp_J_kgK": 4180}}},
  "regions": [
    {"name": "a", "shape": {"rect": {"x": [0, "L/2"], "y": [0, "H"]}}, "material": "slab"},
    {"name": "b", "shape": {"rect": {"x": ["L/2", "L"], "y": [0, "H"]}}, "material": "slab"}],
  "curves": [{"name": "mk_line", "path": {"segment": [["L/4", 0], ["L/4", "H"]]}}],
  "boundaries": {"hot": {"of": "a", "where": {"x": 0}}, "cold": {"of": "b", "where": {"x": "L"}},
                 "interface": {"between": ["a", "b"]}, "mk": {"curve": "mk_line"}},
  "physics": {"heat": {"type": "heat.transfer", "domains": ["a", "b"],
    "boundary_conditions": {"hot": {"kind": "temperature", "T_K": 310}, "cold": {"kind": "temperature", "T_K": 290},
                            "remaining": {"kind": "insulation"}}}},
  "outputs": {"Q_mk": {"kind": "flux", "of": "heat", "on": "mk"}},
  "mesh": {"size": "H/8"}
}
```

## `axmp boundaries`

```
axmp boundaries split_rect.json            # a text table
axmp boundaries split_rect.json --json     # the same as JSON
axmp boundaries split_rect.json --json --mesh -o run   # also mesh into run/ and count the faces
```

It validates the document (with the mesher probe), runs the mesher's resolve pass, then the **full check**: the checks
that `axmp validate` can only defer because they need the geometry. They are the empty selections
(`boundary.empty_selection`), the partition rule of every physics (`physics.boundary_uncovered`,
`physics.boundary_double`, `physics.boundary_interior`, `physics.boundary_outside_domain`), the partial matches
(`boundary.partial_match`, a warning), and the outputs on resolved boundaries (`output.boundary_not_in_domain`,
`output.force_on_interior`, `output.force_needs_stress_form`). After it the report's `deferred` list is empty.

The JSON is the validation report with three keys added:

- `symmetry`;
- `boundaries`: one row per boundary name with `selector` (as written), `entities`, `measure` and `measure_units`
  (`"m"`, the length), `area_m2` (the revolved area, axisymmetric only), `bbox` (exact, keyed by the coordinate names),
  `point`, `adjacent` (the region pairs, `exterior` last) and `used_by`: the conditions keyed by the boundary (physics,
  kind, and **the part the physics takes**: a key is taken where its boundary lies on the domain boundary of that
  physics) then the outputs on it;
- `edges`: every boundary edge, sorted by its point, with `adjacent`, `type`, `bbox`, `point`, `measure`, `names` (the
  boundaries that select it; empty for an edge no boundary names, which `"remaining"` covers) and `axis`.

With `--mesh` every row and edge also has `faces`, its number of boundary elements on the mesh (with the document's
`mesh.refinements`). No master number or geometry-kernel tag appears anywhere.

With an error the command prints the report with `ok` false and the first error's fields at the top level, the table
included when the boundaries could be resolved (the hints send you to it), and exits with status 1.

## `axmp mesh`

```
axmp mesh split_rect.json -o run
```

removes the `mesh.msh`, `mesh_info.json` and `mesh_in.json` of an earlier run in `run/` before it does anything else, so no
refusal leaves an earlier mesh behind, then runs the same checks before it meshes. When one fails, **no mesh is written**;
if the check that failed needs the geometry, the mesher's resolve pass has left its `mesh_in.json` (with `"resolve_only":
true`), `mesh_info.json` and `mesh_mesher.log` in `run/`. When all pass it writes `run/mesh_in.json` (the mesher's
input), `run/mesh.msh`, `run/mesh_info.json` (which lists the atoms and their master numbers) and `run/mesh_mesher.log`.
It prints `ok`, `name`, `symmetry`, `geometry_order`, the three paths, `boundaries` (`measure`, `measure_units` and, for
an axisymmetric model, `area_m2`) and the `warnings`. A failed mesher run is `mesh.generation_failed`, with the mesher's
error in the message and the log's path in the hint. The geometry order is `mesh.geometry_order`, else the largest
order every physics takes (`mesh.geometry_order_lowered` warns when that is 1 for a curved model). `axmp validate` takes `--mesher` too, for its probe.

## Python

```python
import axionmultiphysics as axmp

d = axmp.load("split_rect.json")
table = d.boundaries()                 # the --json report; raises the first error
m = axmp.mesh(d, workdir="run")        # MeshedModel; without workdir a temporary directory is used and removed
m.symmetry, m.geometry_order, m.regions   # regions: {"a": {"area_m2": ...}, ...} ("volume_m3" too when axisymmetric)
m.boundaries                           # the table's rows, without face counts
im = m.instance("a")                   # a region or domain name, a list of names, or None for every region
im.attributes["hot"]                   # instance boundary attributes of a boundary: list[int], 1..n
im.remaining_attributes, im.axis_attributes
im.vertices(), im.elements(), im.boundary(), im.boundary_attributes(), im.element_attributes()
fp = im.face_points("mk", order=4)     # points, weights and normals for a flux integral
```

`face_points(name=None, *, order, attribute=None)` takes exactly one of a boundary (or section) name and an instance
attribute, which is the way to the edges no boundary names and to the axis. It returns NumPy arrays `faces` (m,),
`elements` (m,), `ref` (m, q, 2), `weights` (m, q) (the face Jacobian is included, so they sum to the length), `normals`
(m, q, 2) and `points` (m, q, 2). The normals are outward on the boundary of the instance; on a section (a boundary
interior to the instance) they have the orientation of the selection: from A to B for `between`, outward from D for `of`,
the written direction of a curve marker turned by +90 degrees. `section(name)` gives the faces of an interior boundary with
their signs and length. The instance mesh owns its data: it works after the `MeshedModel` is gone. Misuse (an unknown name,
a boundary with no face on the instance, an attribute that is not on it, both or neither argument) is a `ValueError`; an
electromagnetic design given to `Design.boundaries()` or `axmp.mesh` is refused with a message and no code.

`axmp.mesh` meshes into its working directory every time and reads what that run wrote.

## Handing a mesh to the engines

`im.fluids_mesh()` and `im.chem_mesh(order=2)` give an instance mesh to the fluids and chemistry engines
(`m.fluids_mesh(domain=None)` and `m.chem_mesh(domain=None, order=2)` are shorthands for `m.instance(domain)` followed by
the call). The engines take no document yet (nothing solves a model document in this version), so the conditions are
yours to write, on the instance's own attribute numbers: `im.attributes[name]`, `im.remaining_attributes` and
`im.axis_attributes`. A master number is not a boundary attribute of the instance mesh, and both engine families refuse
one.

```python
import numpy as np
import axionmultiphysics as axmp
from axionmultiphysics import fluids

# Plane Poiseuille flow through a channel of height H (a model document with a geometry and its named boundaries)
L, H, U, nu = 0.04, 0.01, 0.01, 1e-6
d = {"schema": "axionmultiphysics.design/1", "name": "channel", "symmetry": "planar",
     "parameters": {"L": L, "H": H},
     "materials": {"water": {"density": 998.2, "fluid": {"mu_Pa_s": 0.001}}},
     "regions": [{"name": "channel", "shape": {"rect": {"x": [0, "L"], "y": [0, "H"]}}, "material": "water"}],
     "boundaries": {"inlet": {"of": "channel", "where": {"x": 0}}, "outlet": {"of": "channel", "where": {"x": "L"}},
                    "walls": {"of": "channel", "except": ["inlet", "outlet"]}},
     "physics": {"flow": {"type": "fluid.incompressible", "options": {"equation": "stokes", "viscous_form": "laplacian"},
                          "boundary_conditions": {"inlet": {"kind": "inlet", "velocity_m_s": ["6*0.01*y*(H-y)/H^2", 0]},
                                                  "outlet": {"kind": "outlet", "pressure_Pa": 0},
                                                  "walls": {"kind": "wall"}}}},
     "mesh": {"size": "H/8"}}
m = axmp.mesh(d)
im = m.instance()
fm = im.fluids_mesh()                       # a fluids.Mesh: a copy of the instance mesh
bcs = [{"attribute": a, "kind": "dirichlet", "value": lambda x, y: (6 * U * y * (H - y) / H**2, 0.0)}
       for a in im.attributes["inlet"]]
bcs += [{"attribute": a, "kind": "do_nothing"} for a in im.attributes["outlet"]]
bcs += [{"attribute": a, "kind": "dirichlet", "value": (0.0, 0.0)} for a in im.attributes["walls"]]
sol = fluids.solve_stokes(fm, nu, bcs=bcs, form="laplacian")    # the Laplacian form admits the developed outflow
```

`fluids_mesh()` returns an `axionmultiphysics.fluids.Mesh` with the instance mesh's vertices, triangles, boundary segments
and boundary attributes in the same order, and its nodes when the mesh has them (`Mesh.has_nodes`: true at geometry order 2,
false at geometry order 1, where an advection-dominated heat solve may use `supg=True`). It is a copy and owns its data.
In a build without the fluids module (a wheel from PyPI, the sequential source build) it raises `AxionMultiPhysicsError`
with the code `build.requires_fluids`; the [MPI source build](install.md) has it.

`chem_mesh(order)` returns a `chem.TriMesh` for `chem.CurrentDistribution`. `order` is the field order of the current
distribution, 1 or 2; the mesh records the instance's `geometry_order` and its `edge_nodes()` are the P2 nodes that lie off
their chord (on the circle of an annulus, none on straight edges). A mesh that has nodes is never straightened: `chem_mesh(1)` on one,
`CurrentDistribution(mesh, order=1)` and `CurrentDistribution(mesh, order=2, curved=False)` on a mesh that records geometry
order 2 all raise a `ValueError` that names the remedy, as does `chem.meshes.red_refine` of a curved bridge mesh (refine the
mesh by a smaller `mesh.size`). A model whose physics
takes a P1 field has its geometry order lowered to 1 by the validation, so its instance mesh has no nodes.

```python
import axionmultiphysics as axmp
from axionmultiphysics import chem

d = {"schema": "axionmultiphysics.design/1", "name": "cell", "symmetry": "planar",
     "materials": {"bath": {"electrolyte": {"kappa_S_m": 1.0}}},
     "regions": [{"name": "bulk", "shape": {"rect": {"x": [0, 0.01], "y": [0, 0.02]}}, "material": "bath"}],
     "boundaries": {"anode": {"of": "bulk", "where": {"x": 0}}, "cathode": {"of": "bulk", "where": {"x": 0.01}}},
     "reactions": {"electrode": {"plate": {"kind": "linear", "i0_A_m2": 10, "E_eq_V": 0.0}}},
     "physics": {"cell": {"type": "electrochem.current_distribution", "options": {"kinetics": "secondary", "order": 2},
                          "boundary_conditions": {
                              "anode": {"kind": "electrode", "reactions": ["plate"], "phi_s_V": 0.05},
                              "cathode": {"kind": "electrode", "reactions": ["plate"], "phi_s_V": 0.0},
                              "remaining": {"kind": "insulation"}}}},
     "mesh": {"size": 0.001}}
m = axmp.mesh(d)
im = m.instance()
cd = chem.CurrentDistribution(m.chem_mesh(), order=2)
cd.set_conductivity({a: 1.0 for a in im.region_attribute.values()})
cd.add_electrode("anode", im.attributes["anode"], primary=True, phi_s=0.05)
cd.add_electrode("cathode", im.attributes["cathode"], primary=True, phi_s=0.0)
print(cd.solve()["converged"])      # the other two sides are insulating by default
```

The potential of the parallel-plate cell is linear, so the current per metre of depth is `kappa * dV * Hc / gap` to
round-off. The tolerances of the engine tests that run through this bridge (the closed forms of Poiseuille flow, the
two-layer flow, Hagen-Poiseuille flow in a pipe, a heat slab and this cell) are recorded per processor model in
`cpp/tests/data/model/u2/cases/`, written by `scripts/model_bridge_tolerances.py --record`.

## Orientation and the direction of markers

The orientation rules are in [named boundaries](../design_format.md#named-boundaries-atoms-and-the-partition-rule). An `arc`
marker path is read counter-clockwise from its first angle to its second (the validation requires the first to be the
smaller), so its normal, the left of the tangent, points to the arc's centre. A `rect` path has no written order, so its
orientation is not yours to choose: use `between` or `of` for an oriented flux through a closed curve. A `segment` or
`polyline` runs as written.

## Limits of this version

- A `contour` with an elliptical segment is refused at validation (`model.shape_not_offered`): use `disk`, `rect`, `polygon`,
  `union` and `difference`.
- 3D model documents and model documents on an imported mesh are refused with `route.not_implemented`.
- A `where` filter on a straight edge is decided exactly (the segment is cut at the filter's bounds; an edge that only touches the box is not reported). On a curved edge it is decided by the edge's bounding box, and a curved
  edge whose box encloses the filter's box strictly in every coordinate is not reported as a partial match, so a curve that
  crosses such a box while another edge of the boundary is selected is not warned about. An empty selection still lists the
  candidates.
- `axmp.mesh` raises `route.not_implemented` for a 3D document (at `/symmetry`), a document on an imported mesh (at `/mesh/file`) and a document with no geometry (at `/regions`; `axmp mesh` prints a note for the last).
