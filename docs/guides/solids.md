# Solids for particle transport (beta)

Geant4 transports particles through a tree of solids, each placed in its mother. `axionmultiphysics.solids` builds
that tree from the mesher's description of an axisymmetric partition (the regions as disjoint faces in the (r, z)
half-plane, bounded by curves whose exact parameters the mesher states) or of a [3D partition](#3d-designs) (the
volumes of a 3D design, bounded by triangulated faces). It writes the tree as `solids.json` (schema
`axionmultiphysics.solids/1`), which `axmp-transport` instantiates, or as GDML for any Geant4 application. Per region it
reports the volume and the mass of the exact solid of revolution and of the solids that represent it. Nothing here
needs gmsh or Geant4.

!!! warning "Beta"
    The module, the format of `solids.json` and the codes may change between minor versions. The partition comes from a
    hand-built table, from the mesher's info file of a bounded axisymmetric mode, or from an axisymmetric or 3D
    electromagnetic design ([`Design.export_gdml`](#electromagnetic-designs), [3D designs](#3d-designs)); the route
    from a model document to the partition comes later.

```python
from axionmultiphysics import solids

doc = solids.from_partition(partition, materials, region_materials)   # chord_tol_m=None
solids.write(doc, "solids.json")
solids.write_gdml(doc, "design.gdml")
```

- `partition`: the mesher's info file of a bounded axisymmetric mode, or the file of `axmp-mesher --export-regions` (a
  JSON object), with the extended entity table described below; for a 3D design, the file of `--export-regions` on its
  3D mesher input ([3D designs](#3d-designs)).
- `materials`: the document's materials, `{name: {"density": kg/m3, "particle": {...}}}`. Other keys of a material
  (its electromagnetic or thermal properties) are ignored.
- `region_materials`: `{region: material name}` for every region that has a face.
- `chord_tol_m`: the chord tolerance of every region, in m; by default each region has its own (below). A 3D partition
  takes none (`ValueError`): its triangulation is the mesher's, made to the `chord_tol_m` given to the export.

`solids.default_chord_tol(size_m)` is min(1 µm, 1e-5 × size); `solids.VOLUME_WARN` is 1e-4; `solids.SCHEMA` is
`"axionmultiphysics.solids/1"`. Errors are `AxionMultiPhysicsError` with a `code` and a `hint`; the volume warning is a
`solids.SolidsWarning`.

## The partition

The conversion reads, from the info file:

- `geometry`: `"axisymmetric"` (a planar partition is refused with `particle.planar_not_offered`);
- `regions`: `{name: {"attr": ...}}`; the regions are reported in the order of `attr`;
- `entities`: the curves, each with `tag`, `type` (`line`, `circle` or `ellipse`), `side` ([the region on the left of
  the curve's direction, the region on its right], `"exterior"` outside the domain), and the extension below;
- `faces`: `[{"tag", "region", "loops": [[signed tags], ...]}]`, every final face of the partition with its outer loop
  first. A loop keeps its face on its left (the outer loop counter-clockwise in (r, z), the inner loops clockwise); a
  positive tag runs the entity from `start` to `end`, a negative one from `end` to `start`.

Each entity carries `start` and `end` ([r, z, 0], in m): the coordinates of its end vertices, `start` first along the
curve. Entities that meet share the coordinates of their common vertex bit for bit, a point on the axis has r exactly
0, and a closed curve has `start == end`. An arc (`circle` or `ellipse`) also has `center`, `radii` ([a, b], a ≥ b;
equal for a circle, also for the circular arcs that gmsh reports as ellipses), `major_axis` (the unit vector of the
semi-axis a) and `ccw` (true when the curve turns counter-clockwise from `start` to `end`): the curve is
c + a cos t u + b sin t v, with v the major axis turned by +90°.

Every other key of the info file (the regions' measures, the atoms, the boundaries, the timings) is ignored, and the
output depends only on the geometry: not on the order of the entities, faces or inner loops, the tags, or where a
loop starts. A table that does not describe closed faces is refused with `particle.partition_invalid`: no `faces`; a
face that names an entity or a region the table does not have; an entity type other than line, circle or ellipse (an
imported mesh has no exact curves); a missing `start` or `end`; a loop that does not close (consecutive curves must
share their vertex bit for bit); a face that runs a curve with its region on the wrong side; an outer loop that is not
counter-clockwise.

## Electromagnetic designs

`Design.export_gdml(path, composition, *, solids=None, chord_tol_m=None, workdir=None, mesher=None)` (beta) writes the
GDML of an axisymmetric electromagnetic design, and returns the solids document:

```python
import axionmultiphysics as axmp

design = axmp.template("core")
composition = {"winding": {"nist": "G4_Cu", "density": 8960.0}, "nb": {"nist": "G4_Nb"},
               "vacuum": {"nist": "G4_Galactic", "density": 1e-22}, "air": {"nist": "G4_AIR", "density": 1.205}}
doc = design.export_gdml("core.gdml", composition, solids="solids.json")
```

The route is the design's mesher input (the one `axmp mesh` writes), `axmp-mesher --export-regions` on it, and
`solids.from_partition` on the partition it writes. The mesher's export builds the geometry of the electromagnetic mesh
(the half disk, the regions in order, the curves, the fragmentation) and writes, instead of a mesh, the partition of
that geometry to the `info` path of its input: `regions` (the design's regions and the background `air`, each with its
`attr`, its `area_m2` and its `volume_m3`, computed from the exact curves), `entities` (every curve with the keys above;
`side` names a region, `air` for the background, or `exterior` beyond the outer arc and the axis) and `faces`. The tags
are those of the mesh of the same input. A contour's elliptical segment, a B-spline in gmsh, is an `ellipse` with the
contour's own centre and axes. The mesher exits with status 1 and `{"ok": false, "error": ...}` in that file when the
construction fails, and refuses inputs of the bounded and plane modes (use the bounded mode's own info file with
`resolve_only`: it carries the same keys). A 3D input has its own partition ([3D designs](#3d-designs)).

- `composition`: `{material: block}`, the block of [Materials](#materials) with an optional `density` in kg/m3. Each
  region takes its design `material`; the background region takes the material named `air` (the design's vacuum
  material of that name, or the design's own definition of `air`). Entries for materials no region uses are ignored;
  sheets (`curves`) have no volume and need no entry; disabled regions are not in the partition.
- The density of a material is the design's, as the electromagnetic parser resolves it (iron 7874, copper 8960 and nb
  8570 are built in; an expression is evaluated), when it is positive; otherwise the `density` of the block. Two
  different positive values are `particle.composition_invalid` (naming the material and both values); none is
  `particle.density_missing`; a used material without a block is `particle.composition_missing`.
- `solids`: a path; the solids document is also written there as `solids.json`. `chord_tol_m` is passed to the
  conversion, and so are its warnings.
- `workdir`: keeps `regions_in.json` (the mesher input) and `regions.json` (the partition, the mesher's `info`) there.
  Without it a temporary directory is used and removed; nothing is written in the current directory.
- `mesher`: the program (default: `$AXMP_MESHER`, else the package's).

Refused, with nothing written: a model document (`model.em_only_operation`); a 3D design with `mesh.symmetry`, a design on an imported mesh (`mesh.file`) and a mesher without the export mode
(`particle.partition_unavailable`); a mesher that cannot be run or
loaded, as the model path reports it (`build.requires_mesher`, with the loader's message); a planar or cross-section
design (`particle.planar_not_offered`); a mesher run that fails (`mesh.generation_failed`, the message carrying the
mesher's error and the last lines of its log in the hint; with `workdir` the log stays there as `regions_mesher.log`).
The input is made by the code that `axmp mesh` runs, which also meshes the design once into a temporary directory.

## 3D designs

`Design.export_gdml` also exports a 3D electromagnetic design (`"symmetry": "3d"`), with the same arguments and the same
composition rules. Every volume becomes a **`tessellated`** solid: a closed triangulated surface, which Geant4 builds as a
`G4TessellatedSolid`. The route is the design's 3D mesher input (the `mesh3d_in.json` that `axmp mesh` writes, made by
a library function that the 3D engine also uses, compiled into every build: the sequential build exports 3D designs
too, and the engine is not run), `axmp-mesher --export-regions` on it and `solids.from_partition` on the partition. Each
rank of a parallel program (`mpirun -n 4 python script.py`) exports on its own, with the same files; nothing is
collective. The mesher runs only with the export flag, never for an electromagnetic mesh. The engine's own refusals of a
design (sheets, in a 3D design) are reported as themselves. A design reduced by symmetry planes (`mesh.symmetry`) is
refused with `particle.partition_unavailable`: a reduced model is not the device.

The mesher builds the 3D geometry as for the electromagnetic mesh (the solids, the domain sphere, the **infinity shell**,
the fragmentation, the region of each volume by priority) and writes, instead of a mesh, the 3D partition to the `info`
path of its input. The infinity shell is not part of it: the transport's domain ends at the domain sphere, the inner
radius `domain_radius`. The conductors that the electromagnetic mesh removes (`remove_pec`) are volumes of the partition.
The file has `geometry` `"3d"`, `domain_radius`, `domain_center`, `diagonal_m` (of the union of the faces' boxes),
`tolerance_m` (1e-9 × `diagonal_m`), `chord_tol_m` and:

- `regions`: `{name: {"attr", "volume_m3"}}` for the design's regions and the background `air`; `volume_m3` is the exact
  volume (OpenCASCADE's).
- `entities`: one record per face, in the order of `tag`, with `dim` 2, `side` ([A, B]: the face's triangles' normal points
  from the region A into the region B; `air` for the background, **`exterior`** beyond the domain sphere, where the
  infinity shell was), `curves` (`[]`), `axis` (false), `type` (gmsh's surface type in lower case: `plane`, `cylinder`,
  `cone`, `sphere`, `torus`, `bspline_surface`, ...), `lo` and `hi` (a box that holds the face), `point` and `normal` (a
  point of the face and the unit normal of the surface there, turned like the triangles) and `area_m2`.
- `volumes`: `[{"tag", "region", "shells", "volume_m3"}]`, every volume but the shell's. `tag` is the volume's tag in the
  mesh of the same input. `shells` lists signed face tags, the outer shell first: `+tag` where the volume is the face's
  `side[0]` (the triangles' normal points out of it), `-tag` where it is `side[1]`. A shell with its faces turned by their
  signs is a closed manifold surface oriented outwards (every edge used once in each direction in the shell); a cavity
  is an inner shell. Faces belong to one shell when an edge of the shell pairs them, so the air around two boxes that
  meet along an edge has a shell for each. A domain without regions exports the air alone (one volume).
- `triangulation`: `{"nodes": [[x, y, z], ...], "faces": [{"tag", "triangles", "sagitta_m"}, ...]}`. One node list serves
  the whole partition; each face is triangulated once and listed once (the volumes on its two sides use the same
  triangles, one with the reversed orientation, and the faces that meet at a curve share its nodes, so that coordinates
  agree bit for bit with no matching tolerance). `sagitta_m` is the largest distance of a triangle's centroid or edge
  midpoint from the face (0 for a plane).

**The chord tolerance.** `chord_tol_m` is the largest distance a triangle may lie from its face. It is the input's key
`chord_tol_m` (the `chord_tol_m` of `Design.export_gdml`), by default `min(1 mm, 1e-4 × diagonal_m)`. The triangulation is
the export's own, made for that tolerance, and depends on the geometry and the tolerance only, not on the sizes of the
electromagnetic mesh. A region's triangulated volume differs from its exact one only under the triangles of curved
faces (reported per region, and warned about above `VOLUME_WARN`: lower `chord_tol_m`); planar regions are exact to
rounding.

**The solids.** One solid per volume, of kind `tessellated` with `params` `{"vertices_m": [[x, y, z], ...], "triangles":
[[i, j, k], ...]}`: the triangles of the volume's outer shell, counter-clockwise seen from outside, with its own vertex
list copied bit for bit from the partition's nodes, so that solids that touch have the same facets on the face between
them. Position and rotation are zero (the vertices are absolute); `discretized` is true and `sagitta_m` the largest of
the outer shell's faces. Names are `<region>_<k>` in a geometric order. The volumes in a cavity of a volume (inside one of
its inner shells, directly or as a cluster of volumes touching across faces of their outer shells) are that volume's
daughters; a volume in no cavity is a daughter of the world, a vacuum `orb` around every node. In the report,
`volume_solids_m3` is the volume of the region's solids minus their daughters', which is the triangulated volume of the
region.

**Thin layers.** The triangles are chords of the exact surfaces, so a layer thinner than about the chord tolerance
cannot be represented: the chords of its two surfaces cross. The conversion refuses such a partition with
`particle.partition_invalid` (the message names the region, the hint says to lower `chord_tol_m` well below the layer's
thickness) when a daughter's triangulated surface is not inside its mother's, when two siblings overlap, or when a
solid's own triangles cross each other. Triangles that two solids share exactly are contact, not overlap. The mesher
still exports the partition at any tolerance (its sides are facts of the topology), and the same partition converts
at a finer `chord_tol_m`. The checks use a sweep over the triangles' boxes, exact piercing tests of triangle edges
against triangles (the tolerance is `tolerance_m`) and ray casting for containment; on the window-frame dipole
(29 000 triangles) they take about 2 s. Pairs of triangles are found with a uniform 3D grid, so the time grows about as
the number of triangles; a solid that is star-shaped about the centre of its box (a ball, a box, a cylinder: every
triangle faces away from the centre and their solid angles add up to 4 pi) cannot cross itself and skips the
self-crossing test. A fine `chord_tol_m` costs mostly in triangles, not in the checks: the count of a curved face grows as
1 / `chord_tol_m` (a sphere of 0.5 m radius: 64 000 triangles at 1.5e-4 m, 425 000 at 3.3e-5 m, converted in 1.7 s and
8.9 s with the self-crossing test forced on, 0.2 s and 1.1 s without), and the export, the JSON files and Geant4's
tessellated solids grow with it.

**GDML.** A tessellated solid is a `<tessellated name="<gdml_name>_solid" lunit="m">` of `<triangular vertex1= vertex2=
vertex3= type="ABSOLUTE"/>` in the list's order and orientation; its vertices are `<position unit="m">` entries of
`<define>` named `<gdml_name>_v<k>`, written with 17 digits. `axmp-transport --geometry` builds a `G4TessellatedSolid` of
`G4TriangularFacet` (`ABSOLUTE`, lengths in m) set closed, after checking that the triangles close (every edge once in
each direction) and enclose a positive volume: Geant4 only warns about an open solid, the program refuses it as an
invalid file (exit status 2, the solid named).

Every 3D volume is tessellated, including the primitives: a native solid that touches a tessellated one across a curved
face would overlap it by the sagitta, which Geant4's overlap check (a tolerance of zero) refuses.

## The solids

Each face becomes one solid, a full revolution about the z axis:

1. **Notches at the axis are filled.** The solid of a face is the revolution of its outer loop with every notch at
   the axis filled: of the parts of the loop between consecutive points on the axis, the one that runs upwards is
   kept and the others are replaced by the axis. Whatever lies in a notch (a face, or a hole of the domain) becomes a
   daughter of the solid, as whatever lies in a hole does. This is what makes a sphere on the axis exact: the face
   around it is a full tube, and the sphere is placed in it.
2. **The kind of solid.** One closed circle off the axis is a `torus`; one half circle centred on the axis, its ends
   on the axis, is an `orb`; straight edges that make a rectangle (after merging collinear edges) a `tube`; anything
   else a `genericPolycone`, exact when its edges are straight and **discretized** when it has arcs. A circle split
   into several curves, an ellipse and a fillet are polycones.
3. **Holes of the domain are vacuum.** The boundary curves of the domain, followed with the domain on their left,
   form closed chains (or chains from the axis to the axis, closed along it). A chain that turns clockwise encloses a
   part of the exterior: it becomes a vacuum solid `void_<k>` (`G4_Galactic`), so that a hole is never filled with
   the material around it.
4. **The mother** of a face or a void is the smallest filled loop that contains it; one that nothing contains is a
   daughter of the world. Siblings do not overlap and every daughter lies inside its mother by construction, provided
   no two curves come closer than about the chord tolerance (the chords of a curve stray from it by up to its
   sagitta). The conversion checks it: when a layer is thinner than that, so that a face or a void would lose its
   place in the tree, a daughter would leave its mother, two siblings would overlap or a polycone would cross itself,
   it refuses the partition with `particle.partition_invalid`, whose hint is to lower `chord_tol_m` well below the
   layer's thickness. It never writes a list without a face or with overlapping solids. Curves that touch
   convert when the discretized chords stay on their own side (two circles or ellipses touching from outside, a
   torus on the edge of a box, an ellipse inscribed in a box). Where a discretized curve is concave towards the other
   solid (a shape touching the inside of an ellipse, say), the chords pass through that solid by about 0.8 of the
   chord tolerance at every tolerance, so lowering `chord_tol_m` does not help: the partition is refused, and the
   curves must be separated there by a few chord tolerances. A face whose outline passes twice through one point (a
   pinch: a notch whose tip touches the face's own boundary) is refused too, naming the region and the point, because
   whether Geant4 builds such a polycone depends on rounding in its crossing test; separate the touching
   boundaries. The checks work to 1e-12 of the partition's diagonal but never above 5e-13 m (half of
   Geant4's overlap tolerance), so that a partition admitted here is not refused by Geant4's overlap check.
5. **The world** is a vacuum tube (`G4_Galactic`) centred at the origin, enclosing the partition with a margin of
   1e-3 of its diagonal.
6. **Names.** The solids of a region are `<region>_0`, `<region>_1`, ... in a geometric order (their bounding box,
   their area, then their vertices); the voids `void_0`, `void_1`, ... (skipping a name that the solids of a region
   called `void` already have); the world `world`. Any region or material name is accepted: the names that GDML and
   Geant4 use are mapped from them (`gdml_name`, below).

**Discretization.** Every arc of a polycone is cut into chords once, from its curve, and every solid that has it uses
the same points, so neighbours share their vertices bit for bit and leave no gap or sliver. The end points are the
partition's; the points between are at equal steps of the curve parameter. A circular arc of radius R and sweep θ is
cut into the fewest equal angles N whose sagitta R (1 - cos(θ/2N)) is within the tolerance; an elliptical arc uses the
same rule with its major semi-axis, whose sagitta bounds the true one. The tolerance `chord_tol` of a region is
min(1 µm, 1e-5 × the diagonal of the box of its curves); a curve between two regions uses the smaller of their two;
`chord_tol_m=` sets every region's. Tori and orbs are never discretized.

## `solids.json`

```json
{"schema": "axionmultiphysics.solids/1", "generator": "axionmultiphysics <version>", "geometry": "axisymmetric",
 "units": {"length": "m", "angle": "deg", "density": "kg/m3", "volume": "m3", "mass": "kg"},
 "chord_tol_m": null, "world": "world", "materials": [...], "solids": [...], "regions": {...}, "warnings": []}
```

The file holds no time, host name or path, and the same partition gives the same bytes on every run (strict JSON,
UTF-8, numbers as the shortest text that reads back as the same double).

- `materials`: the materials that some region uses, sorted by name: `{"name", "gdml_name", "density_kg_m3",
  "composition"}` and, when given, `mean_excitation_eV`, `state`, `T_K`, `pressure_Pa`. `composition` is `{"nist": "G4_..."}` or
  `{"elements": {symbol: value}, "by": "mass" | "atoms"}`.
- `solids`: every solid after its mother (the world first, children in name order), each `{"name", "gdml_name",
  "kind", "params", "mother", "position_m", "rotation_deg", "region", "material", "discretized", "sagitta_m"}`.
  `region` and `material` are the document's names.
  `position_m` is relative to the mother; `rotation_deg` is [0, 0, 0] for an axisymmetric design. `region` is null for
  the world and the voids, whose material is `G4_Galactic`. `sagitta_m` is the largest sagitta of the chords in the
  solid (0 when none).
- The parameters of each kind are named as GDML's attributes, with a unit suffix:

  | kind | params | frame |
  |---|---|---|
  | `tube` | `rmin_m`, `rmax_m`, `z_m` (the full length), `startphi_deg`, `deltaphi_deg` | its centre |
  | `torus` | `rmin_m` (0), `rmax_m`, `rtor_m`, `startphi_deg`, `deltaphi_deg` | the centre of its circle |
  | `orb` | `r_m` | its centre |
  | `genericPolycone` | `startphi_deg`, `deltaphi_deg`, `rz_m` ([[r, z], ...], counter-clockwise) | the origin: its points are absolute |
  | `tessellated` (3D lists) | `vertices_m` ([[x, y, z], ...]), `triangles` ([[i, j, k], ...], counter-clockwise seen from outside) | the origin: its vertices are absolute |

  Values that the partition states (radii, vertices) are copied bit for bit.
- `regions`: per region with faces, in document order: `material`, `density_kg_m3`, `solids` (its solids' names),
  `volume_m3`, `volume_solids_m3`, `volume_rel_diff`, `mass_kg`, `mass_solids_kg`, `mass_rel_diff`,
  `sagitta_max_m`, `chord_tol_m`.
- `warnings`: one entry `{"code": "particle.volume_deficit", "region", ...}` per region above the threshold.

## The volume and mass report

`volume_m3` is the volume of the exact solid of revolution of the region's faces: by Pappus' theorem, π times the
closed integral of r² dz around every loop of every face, evaluated in closed form on lines and arcs from the curve
parameters. `volume_solids_m3` is what the solids hold: for each of the region's solids, its volume minus its
daughters' (the closed forms Geant4 uses: a tube, a torus, a sphere, and the sum of cone frusta of a polycone).
`volume_rel_diff` is `volume_solids_m3 / volume_m3 - 1`; it is zero to rounding for exact solids, and for discretized
ones the chords move the volume by at most 2π r_max s L (s the sagitta, L the length of the curved boundary). The
masses are the volumes times the material's `density`. When `|volume_rel_diff|` or `|mass_rel_diff|` exceeds
`VOLUME_WARN` (1e-4), the region is warned about (`SolidsWarning`, message `particle.volume_deficit: region
'<name>' ...`) and listed in `warnings`: lower `chord_tol_m`.

## Materials

A material in the transport's domain needs a `particle` block and a density; nothing is ever treated as vacuum
(vacuum is written explicitly, as `{"nist": "G4_Galactic"}`):

```json
"iron":  {"density": 7874.0, "particle": {"nist": "G4_Fe"}},
"steel": {"density": 7930.0, "particle": {"elements": {"Fe": 0.70, "Cr": 0.19, "Ni": 0.11}, "by": "mass"}},
"pe":    {"density": 940.0,  "particle": {"elements": {"C": 1, "H": 2}, "by": "atoms", "mean_excitation_eV": 57.4}}
```

- `nist`: the name of a NIST material of Geant4 (`G4_...`); the material is that composition at the document's
  density.
- `elements` with `by: "mass"`: mass fractions, positive and summing to 1 within 1e-9 (they are not renormalized);
  with `by: "atoms"`: positive integer atom counts. Elements are given by their symbols, from H to Bh (Z = 1 to
  107, the elements Geant4's NIST tables build); a heavier element is refused, so that no exported file names an element
  Geant4 cannot build.
- Optional: `mean_excitation_eV`, `state` (`solid`, `liquid` or `gas`), `T_K`, `pressure_Pa` (positive).
- **NIST compositions are resolved before a run.** `from_partition` writes a NIST composition as a material of one
  component (the NIST material) at the document's density, and Geant4 would then compute its mean excitation energy
  from the elements by Bragg additivity and set its state from the density, instead of using NIST's values (water
  69.0 eV, solid, instead of 78.0 eV, liquid; NaI 400 eV instead of 452 eV). The transport therefore resolves every
  NIST composition against the Geant4 it runs with (`particles.resolve_materials`, which reads
  `axmp-transport --nist`; the driver runs it before every run and writes the resolved list to the run's
  `exchange/solids.json`):
    - a material whose block is exactly `{"nist": X}` at X's own density (within 1e-9, relative) becomes **the NIST
      material itself**: its solids name `X`, the GDML file references `X`, and the material record is dropped when
      no solid uses it (the region keeps its document material name in `regions`);
    - any other NIST composition keeps its one component and takes X's `mean_excitation_eV` and `state` where the
      document gives none (written as `<MEE>` and `state=` in GDML, set the same way natively);
    - a density more than 1 % from NIST's warns `particle.density_differs_from_nist` (the document's density is
      used).

  The residual of the second case, measured with Geant4 11.4.3's `G4EmCalculator` (total dE/dx at NIST's density
  against the NIST material itself):

  | material | 1 GeV/c muons | 100 GeV/c muons | 10 MeV protons | 1 MeV electrons |
  |---|---|---|---|---|
  | G4_WATER | +1.01 % | 0.00 % | +0.58 % | +0.61 % |
  | G4_SODIUM_IODIDE | +1.41 % | +0.06 % | +1.23 % | +0.45 % |
  | G4_CESIUM_IODIDE | +1.69 % | +0.04 % | +1.61 % | +0.40 % |
  | G4_BGO | +1.99 % | +0.10 % | 0 | +1.18 % |
  | G4_POLYSTYRENE | +0.82 % | 0 | +1.05 % | +0.79 % |
  | G4_KAPTON | +1.02 % | 0 | +0.81 % | +0.98 % |
  | G4_AIR, G4_PbWO4, G4_lAr | within 0.09 % | within 0.09 % | within 0.09 % | within 0.09 % |
  | G4_Fe, G4_Si | 0 | 0 | +0.16 %, +0.21 % | 0 |

  It comes from data Geant4 looks up by the NIST material's name (the density-effect table, ICRU 90 water, proton
  stopping tables), which only the NIST material itself carries. Where it matters, use the NIST density, or give an
  explicit element composition with `mean_excitation_eV`.
- A document material may not have a name that starts with `G4_`: that prefix always names a NIST material.
- The density is the material's `density` in kg/m3 and is always the one used. (Comparing it with the NIST density of
  a NIST material is a check of the transport set-up, where Geant4's tables are.)

## GDML

`write_gdml(doc, path)` writes the same list as GDML (`define`, `materials`, `solids`, `structure`, `setup`), so that
any Geant4 application can read it. Three rules make Geant4 read from it the same numbers as `axmp-transport` builds
from `solids.json`:

1. **17 significant digits.** Every number is written with 17 significant digits (`%.17g`), which reads back as the
   same double (Geant4's own writer uses 15, which does not). Lengths are in `m` and angles in `deg`, units Geant4
   multiplies exactly as the native path does; a density is `<D value=... unit="kg/m3"/>` with the document's value.
2. **NIST by name, never defined.** The file defines no element, no isotope and no material with a NIST name: Geant4's
   reader prefers a file's definition to its NIST table, which would change the material's data. Every document
   material is a material of the file, named by its `gdml_name`, with its density and its components: one
   `<fraction n="1" ref="G4_Fe"/>` for a NIST composition, `<fraction n=... ref="Fe"/>` by mass,
   `<composite n=... ref="H"/>` by atoms (Geant4's NIST manager resolves the elements from their symbols), after
   `T`, `P`, `MEE` and `D`. The world and the voids reference `G4_Galactic`; so does every solid whose material a
   resolved list names directly (`G4_WATER` at NIST's density, say): its volume's `materialref` is the NIST name.
3. **Rotations** are written as the document's angles, x then y then z (none in an axisymmetric design).

One `volume` per solid, named by the solid's `gdml_name`, is defined after its daughters (the world last); each
daughter is a `physvol` with a `position` in m. The volume's solid, placement and position are named after it, with
`_solid`, `_pv` and `_pos`.

**Names in the file.** A GDML `name` is an XML ID: ASCII letters, digits, `_`, `.` and `-`, not starting with a digit,
unique across the file; and Geant4's reader cuts a name where it contains `0x`. A document may name its regions and
materials as it likes (`316L steel`, `1st`, `world`), so each solid and each material carries a `gdml_name`: its own
name when that is valid, contains no `0x`, does not start with `G4_` and collides with no other name of the file
(including the `_solid`, `_pv` and `_pos` names, and `Default`, the name of the file's `setup`); otherwise the name with every other character replaced by `_`, a `_`
in front of a leading digit, `0x` written `0_x`, and `_1`, `_2`, ... appended until it is unique. Solids are named
before materials, each set in name order, so the mapping depends only on the names, never on the order of the input.
`axmp-transport` gives its Geant4 volumes and materials the same names, and its placements the names Geant4's GDML
reader gives them (`<gdml_name>_pv`, and `<gdml_name>_PV` for the world's), so a GDML application and the native path
name everything alike; `solids.json` keeps the document's names beside them. With these names the file follows the GDML
schema of Geant4 11.4.3 (the schema is not part of this repository).

## The geometry mode of `axmp-transport`

```bash
axmp-transport --geometry solids.json
```

builds the list in Geant4 (see [Particle transport](particles.md) for building the program): the materials through
Geant4's NIST manager as above (a NIST composition is a material of the document's density with one component, the
NIST material; a solid whose material is a NIST name uses that NIST material), one `G4Tubs`, `G4Torus`, `G4Orb`, `G4GenericPolycone` or (3D lists) `G4TessellatedSolid` per solid, one logical volume per solid placed
in its mother's, with the lengths read in metres and the angles in degrees, and every volume and material named by its
`gdml_name`. It then runs Geant4's overlap check (`CheckOverlaps`, 1000 points per placement, a tolerance of zero) on
every placement and prints one JSON document on stdout:

```json
{"ok": true,
 "volumes": [{"name", "g4_name", "placement", "region", "material", "solid_volume_m3", "volume_m3", "density_kg_m3", "mass_kg"}],
 "materials": [{"name", "g4_name", "density_kg_m3", "mean_excitation_eV", "state", "T_K", "pressure_Pa",
                "elements"}],
 "overlaps": []}
```

`name`, `region` and `material` are the names of the solids list, `g4_name` the name the Geant4 volume or material
holds (the `gdml_name`; a NIST material keeps its name) and `placement` the name of the volume's placement. `solid_volume_m3` is Geant4's volume of the solid,
`volume_m3` that minus its daughters', `density_kg_m3` the density of its Geant4 material and `mass_kg` their product.
They agree with the report's `volume_solids_m3` and `mass_solids_kg` within 1e-9. `materials` lists every material the
volumes use (`G4_Galactic` included) as Geant4 built it: its density, mean excitation energy (the document's
`mean_excitation_eV`, or the one Geant4 computes from the elements), state (`solid`, `liquid` or `gas`; Geant4 makes a
material of undefined state solid above 10 kg/m3 and a gas below), temperature and pressure (293.15 K and 101325 Pa
unless the document gives them) and `elements`, the mass fraction of each element. These are what a GDML application
reading the exported file must reproduce. Every value is converted from Geant4's units once, on output.

An overlap refuses the geometry: exit status 3 and `{"ok": false, "code": "particle.geometry_overlap", "error": ...,
"overlaps": [{"volume", "mother", "message"}]}`. A file that is not a valid solids list is refused with exit status 2,
a message on stderr and nothing on stdout; that includes a polycone that Geant4 refuses to build (an outline of
almost no area, edges that cross, or fewer than three points left after Geant4 merges the vertices closer than its
length tolerance, 1e-12 m, in both R and Z; Geant4's own message is on stderr) or one whose outline runs back over
itself along a stretch of an edge (the program's own check). `--check` reports `"geometry": true` for a program
that has this mode.

## Codes

| code | kind | raised for |
|---|---|---|
| `particle.planar_not_offered` | error | A planar partition: transport is offered for axisymmetric and 3D designs. |
| `particle.partition_unavailable` | error | `Design.export_gdml` on a design whose partition cannot be exported in this version or with this mesher: a 3D design reduced by symmetry planes (`mesh.symmetry`), an imported mesh (`mesh.file`), or an `axmp-mesher` without `--export-regions`. |
| `particle.partition_invalid` | error | A table that does not describe closed faces with exact curves (see [The partition](#the-partition)), or a layer thinner than about the chord tolerance, whose solids would overlap, cross themselves or lose their place in the tree (lower `chord_tol_m`). |
| `particle.composition_missing` | error | A region without a material, a material that is not defined, or a material without a `particle` block. |
| `particle.density_missing` | error | A material used by a region without a positive, finite `density`. |
| `particle.composition_invalid` | error | A `particle` block that is not one of the forms above: an unknown key, both or neither of `nist` and `elements`, a `nist` name without `G4_`, `by` missing or not `mass`/`atoms`, an unknown element symbol or one heavier than Bh (Z = 107), mass fractions that are not positive or do not sum to 1, atom counts that are not positive integers, a `state` other than solid, liquid or gas, a non-positive `mean_excitation_eV`, `T_K` or `pressure_Pa`, or a document material named `G4_...`; for `Design.export_gdml` also a composition density that differs from the design's. |
| `particle.volume_deficit` | warning | A region whose solids differ from the exact solid of revolution by more than `VOLUME_WARN` in volume or mass. |
| `particle.geometry_overlap` | error | `axmp-transport --geometry`: Geant4's overlap check found a placement that leaves its mother or enters a sibling. |
| `particle.density_differs_from_nist` | warning | `particles.resolve_materials`: a NIST composition whose density differs from NIST's by more than 1 %. |

The messages name the region, the material, the element or the key concerned.

## Limits of this version

- Axisymmetric designs (full revolutions only) and 3D designs without symmetry planes; model documents come later.
- A 3D design is exported as tessellated solids only (native solids for primitives are not offered), to the chord
  tolerance of the export. A layer thinner than the chord tolerance is refused by the conversion (see "Thin layers" in
  [3D designs](#3d-designs)); a very thin layer needs a fine `chord_tol_m` and a large triangulation.
- The partition must carry the extended entity table (exact curve parameters and face loops). B-splines are not read:
  the mesher writes a contour's elliptical arc as an `ellipse`.
- `Design.export_gdml` has no `mesh.file` route, and it meshes an axisymmetric design once to obtain the mesher's input
  (a 3D design is not meshed: its mesher input is written without the engine).
- Elements up to Z = 107 only (Geant4's NIST tables build no heavier one).
