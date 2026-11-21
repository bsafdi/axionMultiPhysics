# Field maps for particle transport (beta)

A particle tracker needs the static magnetic field at arbitrary points, many millions of times, without the finite
element solve. `Results.export_field_map` writes the field of a solved design as a **field map**: one regular grid
(a *patch*) per region, each holding that region's own recovered field, refined until a fixed set of line integrals
through the region agrees with the field to a stated tolerance. `axionmultiphysics.fieldmap` reads, evaluates and
checks such maps. The format is `axionmultiphysics.fieldmap/1`.

Field maps are **beta**: their interface may change. The error control was calibrated on solved axisymmetric fields
(a permeable sphere, superconducting and thin-layer sheets, iron cores, solenoids, a knee-law toroid) and checked on
one 3D solve; the transport that reads the maps is not part of this version.

## Exporting a map

```python
import axionmultiphysics as axmp

res = axmp.Design(doc).solve(workdir="runs/magnet")
fm = res.export_field_map("runs/magnet/fieldmap", tol=1e-4,
                          track_volumes=[{"name": "bore", "r": [0.0, 0.2], "z": [-0.5, 0.5]}])
print([(p.name, p.shape, p.record["accepted"]) for p in fm.patches])   # a patch not accepted also warns
B = fm.evaluate([[0.05, 0.0, 0.1]], patch="air+winding")   # Cartesian (x, y, z) in m -> (Bx, By, Bz) in T
```

| option | default | meaning |
|---|---|---|
| `regions` | `None` | the regions to map; `None` maps every region except ideal superconductors |
| `tol` | `1e-4` | the RMS error of the map's chord integrals each patch must reach (`0 < tol < 1`) |
| `merge` | `True` | adjacent regions with the same constitutive law and no curve between them share one patch |
| `track_volumes` | `None` | boxes whose own chord families are reported: `{"name", "r": [lo, hi], "z": [lo, hi]}` (3D: `"x", "y", "z"`) |
| `iterate_on_track_volumes` | `False` | refine until the track volumes, too, meet the tolerance |
| `outside` | `"refuse"` | what the evaluator does outside a patch's box: `"refuse"` or `"zero"` |
| `max_patch_MB`, `max_total_MB` | `512.0`, `2048.0` | caps on one patch's array and on all of them (MB = 2^20 bytes) |

Axisymmetric results give **kind 1** patches: (B_r, B_z, B_φ) on an (r, z) grid, interpolated bilinearly;
`Results3D` gives **kind 3** patches: (B_x, B_y, B_z) on a Cartesian grid, interpolated trilinearly. Planar and
cross-section results are not mapped. A refusal writes nothing: neither `path` nor `path.tmp` exists afterwards.

## What a patch holds

**One patch per region.** Each region is served by its own grid, which covers the region's bounding box. Adjacent
regions whose constitutive laws are equal (linear with μ_r equal within 1e-6 relative, such as air, windings and normal
conductors, or the same nonlinear material) and whose shared faces carry no curve merge into one patch, named by its
regions joined by `+` (`"air+winding"`). Ideal superconducting regions get no patch (B = 0 there) and are listed under
`zero_field_regions`.

**Curves.** A superconducting sheet or a thin magnetic layer inside a region is an interface: the field jumps across
it. A closed one splits the region into parts, each a patch of its own named `<name>#k`. An open one, which has the
same part of the region on both of its sides, is refused with `fieldmap.curve_in_region`: no patch can hold a jump that
ends inside it. Close the curve, or leave the region out. A sheet of a normal conductor (copper, say) is transparent to
the static field: it is not an interface and is ignored (no split, no refusal, not listed in `curves`).

**The recovered field B\*.** Within a patch the map represents B\*, the finite element field B_h averaged at the nodes
of a continuous space of the solution's order over the **patch's own elements only**, so averaging never crosses an
interface with a jump. On the axis (kind 1, r = 0) B\* has B_r = B_φ = 0, and B_z is the mean of the limits
B_z = 2 ∂A_φ/∂r of the elements that have an edge on the axis.

**Nodes.** A grid node inside or on a patch element takes B\* there (one-sided at the patch boundary). A node outside
the patch whose grid cells reach the bounding box of a patch element takes the polynomial of the **nearest patch
element**, continued to the node (never a neighbour's value), so that the stepper's trial points just outside the
region see the region's own field continued. Other nodes hold 0. Kind-1 axis nodes hold B_r = B_φ = 0. The patch's box
is the box of its elements' nodes, so a region whose faces lie on the box (a rectangular core) has no grid beyond
them: points just outside are served by clamping (below).

**How far the continuation can be trusted.** Where a region's boundary lies inside the box (a curved interface, a
sheet), the continued polynomial reaches about **one cell of the final level** past the boundary, and beyond that
the grid holds zeros, which are interpolated without any refusal. In the sphere example (tolerance 1e-3) the air
patch evaluated inside the ball, against the exact exterior field continued, differs from it by at most 0.8 % of
B_patch at 0.25 final cell past the boundary and by up to 27 % at 1 cell (median 1.1 %); at 2 cells the error is
large (median 14 %) and at 4 cells most points return 0. The depth shrinks as the map is refined, and the
`margin` (below) does **not** extend it: the margin applies only where a region face is the box face. A stepper that
visits points beyond a curved interface must keep them within a fraction of a final-level cell.

**Field-free patches.** A patch whose B_patch (below) is at most 1e-10 of the largest B_patch of the map, such as the
inside of a closed superconducting sheet, is written as zeros (`field_free: true`) and not iterated on.

## Error control

**Grids.** Level 0 of a patch has `n = max(9, ceil(L / h_el) + 1)` nodes along each axis of its box (length L), with
h_el the median element size of the patch (sqrt(2 × area) in 2D, (6 × volume)^(1/3) in 3D); each level halves the
spacing (`n → 2n − 1`).

**Chords.** Each patch has fixed families of straight chords of nominal length 8 h_el, clipped to the patch (and, for
kind 1, to r ≥ 0): `bulk` (16384, start points spread over all elements), `layer` (4096, start points within h_el of
the patch's *jump boundary*: faces whose other side is another region or another part across a curve; not the outer
boundary of the domain, not the axis), `axis` (4096, kind-1 patches that touch r = 0, start points with r < h_el) and
`track_<name>` (4096 per track volume that meets the patch, clipped to the box). Kind-1 chords are lines in the (r, z)
half-plane; kind-3 chords have isotropic directions. Start points are drawn element by element (probability ∝ area or
volume) from a scrambled-Halton sequence whose seed is a constant of the format (`chord_seed`); the same chords serve
every level.

**The statistic.** With B_patch the largest |B\*| over the B\* nodal values of the patch's elements, per chord c

    e_c = |∫_c (B_map − B*) dl| / max( ∫_c |B*| dl , 0.1 L_c B_patch ),
    n_c = |∫_c (B*  − B_h) dl| / the same denominator,

and, per family, E = RMS over chords of e_c (also its 95 % quantile and maximum) and H = RMS of n_c. The integrals of
the map are exact (multilinear along a line); those of B\*, |B\*| and B_h use adaptive Gauss-Kronrod quadrature on the
pieces between element boundaries. A chord is its intersection with the patch, found by walking it element by
element (curved elements included): nothing of it is ever left out. An excursion shorter than the sampling step of
the walk's point location (about 1e-5 m of closest approach to a curved interface) is not resolved: a chord that
barely grazes the interface is counted as staying in the patch, which in the measured cases added up to a few mm
(0.5-1 % of the chord) of the neighbouring region. A walk that cannot be completed (no element past a face, or
more than a few tens of consecutive hand-overs without advancing along the chord) stops the export with
`fieldmap.chord_incomplete` and writes nothing, rather than shortening the chord. Before any chord is drawn, a patch
with an element whose Jacobian determinant is not of one sign at the nodes or the points between them (a tangled
element) is refused with `fieldmap.tangled_mesh`. This is a second guard: the mesher repairs or refuses a 2D mesh with
a tangled element, and the solve refuses one it reads (`mesh.tangled`, see "Curved elements in 2D" in the
[design format](../design_format.md)), so a solved design does not reach it.

**Acceptance.** A patch is accepted when E ≤ tol_eff = max(tol, 0.1 H) in every acceptance family (`bulk`, `layer`,
`axis`; the track families too with `iterate_on_track_volumes`). The floor 0.1 H keeps the map from resolving the field
ten times better than the solve represents it: when it acts, the family's record has `floor_active: true` and a
`note` with both numbers ("the map tolerance was raised to a tenth of the FE indicator ...; a finer or higher-order
solve lowers H"). H is an indicator of the solve's inter-element inconsistency, not evidence of its accuracy.

**Guard and caps.** From level 2 on, a family still above its tol_eff whose E fell by less than a factor 3.0 over the
last two halvings stops the patch with the warning `fieldmap.not_converging` (written, not accepted). A patch whose
next level would exceed `max_patch_MB` is written at the finest level that fits with `fieldmap.cap_reached`; a patch
whose coarsest level does not fit is refused (`fieldmap.patch_too_large`), and so are patches that together exceed
`max_total_MB` (`fieldmap.total_cap_exceeded`). A patch that is written but not accepted also raises a
`fieldmap.FieldMapWarning` (a `UserWarning`) whose message starts with the code and names the patch, so that a script
that does not read `accepted` still hears of it.

**What the accuracy is for.** A momentum scale inherits δ(∫B dl)/∫B dl. E is an RMS over a patch: along tracks
confined to a small part of it the map error can be ten to twenty times larger, which is what track volumes measure.
That the finite element field itself is accurate enough is a question for the solve (a finer mesh or a higher order),
not for the map.

## Outside the maps

`FieldMap.evaluate(points, patch=name)` interpolates the named patch's grid (a map with one patch needs no name). The
evaluator never decides which region a point is in: the transport does, by the volume it is tracking in. Each patch
record carries a `margin` (its level-0 spacing per axis; for `write_map`, the written spacing): a point outside the
patch's box by at most the margin along every axis is **clamped** onto the box and gets exactly the value of the box's
nearest point (a stepper's trial point just outside a volume whose faces lie on the box). A point farther out raises
`fieldmap.outside_map` when the map was written with `outside="refuse"`; with `outside="zero"` it gets B = 0 and is
counted in `FieldMap.outside_calls`.
An export with `outside="zero"` records `boundary_ratio`, the largest |B\*| on the outer boundary of the mapped domain
relative to the largest B_patch, and warns with `fieldmap.outside_field_not_small` above 1e-3.

## The directory format

```
path/
  manifest.json
  patch_0/B.npy                       # shape (n_r, n_z, 3) or (n_x, n_y, n_z, 3), little-endian float64, C order, T
  patch_0/chords/<family>/start.npy   # and direction, segments, segment_chord, length, I_star, I_abs, I_h, I_map
  patch_1/...
```

Patches appear in the manifest sorted by name, in directories `patch_<k>` in that order; node i of axis a sits at
`origin[a] + i * spacing[a]` (m). The manifest records the schema, units (`m`, `T`), `tol`, `outside`, the chord
constants (`chord_seed`, `chords`), the control constants (`control`), the source (`fe_solution`: symmetry, design
name and SHA-256 of the design document, SHA-256 of the static solution, order, elements, DOFs, version and
`build_info` without the thread count, which `timing` records, and without the build's directory and file names),
`zero_field_regions`, `track_volumes`, `warnings`, the patches and `timing`. Each patch record has its file and
SHA-256, kind and frame (kind 3: a 4 × 4 placement transform, global = T local), regions, part (`component`) and
bounding curves, box, h_el, grid (`origin`, `spacing`, `shape`), the history of E per level, `B_patch`, `field_free`,
`accepted`, `stop_reason` (`accepted`, `not_converging`, `cap` or `field_free`), one record per chord family (`n`, `E`,
`E_q95`, `E_max`, `H`, `tol_eff`, `floor_active`, `gain_last`, `gain_two`, `acceptance`), a pointwise report at 1e5
points (|B_map − B\*| and |B\* − B_h| relative to B_patch: 50 %, 99 %, maximum) and the normal-component mismatch
with each neighbouring patch at the midpoints of their shared faces. The same results exported twice, with any number
of threads, give identical arrays and manifests apart from `timing`; no string of a manifest outside `timing`
is a local path.

`fieldmap.write_map(path, patches, outside=...)` writes the same format from nodal values given by the caller (an
analytic field), with `source: {"type": "nodal_values"}`; `fieldmap.load(path)` reads either.

## Checking a map against its solve

- `fieldmap.recovered_field(results, map, patch, points)`: B\* of a patch at points in its native coordinates
  ((r, z) or (x, y, z)), NaN outside its elements.
- `fieldmap.chord_error(results, map, seed=None, patch=None)`: E and H of every family recomputed from the map read
  back from disk; with the manifest's seed it reproduces the report, with another seed it gives an independent
  estimate. Agreement within a factor 2.5 was measured on smooth cases (a sphere, sheets, a solenoid, a knee-law
  toroid); next to the singular corners of a saturated iron core E differed by up to 3.24 (the core) and 7.0 (the air
  beside it) between seeds, because a few millimetre chords at the corners hold most of E² there.
- `fieldmap.decide(E, tol_eff)` is the refinement rule itself; `fieldmap.halton(seed, n)` and
  `fieldmap.family_seed(seed, index)` give the chord sequences.

## Error codes

| code | when |
|---|---|
| `fieldmap.invalid_option` | a bad argument (tolerance, regions, caps, track volume, `outside`, a patch of `write_map`), an unknown or superconducting region in `regions`, a multi-patch `evaluate` without `patch`, planar or cross-section results |
| `fieldmap.curve_in_region` | a curve with the same part of its region on both sides (an open sheet) |
| `fieldmap.track_volume_empty` | a track volume that meets no mapped patch |
| `fieldmap.patch_too_large` | a patch whose coarsest grid exceeds `max_patch_MB` |
| `fieldmap.total_cap_exceeded` | the written patches together exceed `max_total_MB` |
| `fieldmap.outside_map` | `evaluate` at a point outside the patch's box by more than its `margin`, with `outside="refuse"` |
| `fieldmap.tangled_mesh` | a patch with tangled curved elements (Jacobian determinant changing sign, vanishing or inverted; a curved region much thinner than the mesh size at geometry order 2); refine the mesh or use geometry order 1; nothing is written. A second guard: a solve refuses such a mesh first (`mesh.tangled`) |
| `fieldmap.chord_incomplete` | a chord could not be walked through the mesh (an error of the builder: report it); nothing is written |
| `fieldmap.not_converging` | warning in the manifest: the two-halving guard stopped a patch |
| `fieldmap.cap_reached` | warning in the manifest: a patch was written at the largest level that fits its cap |
| `fieldmap.outside_field_not_small` | warning in the manifest: with `outside="zero"`, the field on the mapped domain's boundary exceeds 1e-3 of the largest B_patch |

Each error is an `AxionMultiPhysicsError` with `code`, a message naming the region, curve, patch or option, and a
`hint`.

## Limitations

- Bilinear and trilinear maps are not divergence-free; that matters only for tracking over very many turns.
- The patch statistic is an RMS: a localized error is diluted by the patch's volume (track volumes report it).
- The indicator H under-reports the finite element error (by up to about 14 in the measured cases).
- Thin curved 3D shells waste most of their box and may reach the cap.
- Next to singular corners (re-entrant iron, saturated cores) E is carried by a few chords and varies between chord
  seeds by factors of 3 to 7; it is reported, not gated.
- Not in this version (described in the design): excluding the element around a re-entrant corner from the pointwise
  statistics, and copying the design's own convergence evidence (order 2 against order 3) into the manifest. The
  pointwise report includes the corner elements.
- A placement (`transform`) is a rotation and a translation; `write_map` refuses a scale, shear or reflection.
