# Changelog

All notable changes to axionMultiPhysics are recorded here. Versions follow [semantic versioning](https://semver.org/);
while the version is 0.x, minor versions may change the API. Modules marked **beta** are usable but less validated,
and their interfaces may change.

## Unreleased

## 0.2.0 (2026-11-20)

### Highlights

- **Model documents (beta):** one physics-based input format for fluids, heat, electrochemistry and coupled physics,
  with a physics catalogue, validation, `axmp explain`, `axmp patch` and a JSON Schema. Planar and axisymmetric model
  documents can be meshed; no model document can be solved yet.
- **Bounded 2D meshes and named boundaries (beta):** `axmp boundaries` and `axmp mesh` (`axmp.mesh(d)` in Python) check
  the named boundaries of a planar or axisymmetric model document and mesh it, and an instance mesh can be handed to the
  fluids and chemistry engines. Model documents on an imported mesh are not meshed yet (`route.not_implemented`).
- **Export to Geant4 (beta):** field maps of solved designs (`Results.export_field_map`), solids and GDML of
  axisymmetric and 3D electromagnetic designs (`Design.export_gdml`, `axionmultiphysics.solids`), and their check in
  Geant4 (`axmp-transport --geometry`).
- **Particle transport (beta):** `axmp-transport`, a separate Geant4 program built on request
  (`-DAXMP_BUILD_TRANSPORT=ON`), driven from Python by `axionmultiphysics.particles` (reproducible runs in chunks,
  estimates with statistical errors); demonstrated on a solved solenoid with a tracker, with a cross-check that compares
  volumes, materials, geantino crossings and electron deposits with an independent Geant4 application reading the
  exported GDML. Its limits are listed under "Known limitations".
- **Fluids (beta), accuracy:** every sparse LU walks one recorded retry chain and is accepted on its caller's own
  accuracy gate (`lu_retry`, `lu_stage`, `lu_attempts`); the initial projection, the final Newton step and the mass
  solves are gated, and the default convection quadrature is exact.
- **Meshes and geometry:** tangled curved elements of 2D meshes are repaired by the mesher or refused (`mesh.tangled`),
  and arc paths that span more than a turn, are degenerate or reach r < 0 are refused at validation (`geometry.arc_*`).
- **Error codes:** errors carry `code`, `where`, `hint` and `details` in Python, on the command line and in the MCP
  server.

### Upgrading from 0.1.0

- Electromagnetic designs: what they compute is unchanged (the new compatibility gate G0 compares the text of
  `axmp validate`, `axmp resolve` and `axmp template` for every example, benchmark and test design, and the files and
  numbers of 14 listed solves and meshes byte for byte on the reference platform; see "Validation and compatibility"),
  except that a 2D mesh of geometry order 2 with tangled curved elements is now repaired by the mesher or refused
  (`mesh.tangled`), and that these `arc` paths are now refused at validation: angles that span more than one turn
  (`geometry.arc_span`), a radius that is not positive or an arc of 1e-7 m or less (`geometry.arc_degenerate`), and, in
  every electromagnetic design except a cross-section (`xsec`) one, planar designs on an imported mesh included, an arc
  that reaches r < 0 (`geometry.arc_negative_r`). The spans and the arcs through the axis aborted `axmp mesh`; an arc
  reaching r < 0 in a planar design on an imported mesh, where no mesher runs, was accepted and is now refused (see
  Fixed).
- Fluids (beta): the default `quad_order_convection` is now the exact degree for the element shape and the geometry
  order, so computed solutions change on planar quadrilateral meshes and on curved (isoparametric) meshes, and are
  unchanged on straight triangles. `converged` of a steady Navier-Stokes or inductionless-MHD Newton solve also covers a
  stop at the round-off floor of the residual (`stopped_at_floor`), and a final direct solve or a mass solve that misses
  its bound makes a solve report `converged = false`. A direct solve that escalates ends where the retry chain accepts
  it (in a STRUMPACK build, STRUMPACK with MC64 job 2 instead of Eigen's LU); first configurations are unchanged.
  `lu_fallbacks`, `lu_fallback_backend` and `lu_matching_retry` are kept as derived aliases for one release.
- Chemistry (beta): an exception raised by a problem's own right-hand side now propagates to the caller.
- Fluids (beta): the Eigen direct solver no longer crashes on large 3D systems (64-bit indices above 46 340 unknowns,
  and an out-of-memory error reported on every rank). `axmp` no longer aborts on a design file or file name that is not
  valid UTF-8, and a configure that enables Fortran checks the Fortran compiler.
- The validation pages report the 2026-11-17 and 2026-11-18 reruns of the fluids and chemistry cases (see "Validation
  pages" under Changed).

### Added

#### Model documents (beta)

- **The physics catalogue of model documents (beta):** `axmp physics [--json] [--type T] [--routes] [--all]` and
  `axionmultiphysics.physics(routes=False, type=None)` print the physics types (`fluid.incompressible`, `heat.transfer`,
  `electrochem.current_distribution`, `chem.reactor`, `electrochem.porous_electrode`, and the absent
  `current.conduction`, `solid.linear`, `species.transport`) with their options, condition kinds, sources and quantities
  per geometry class, the coupling terms, the reserved instance ids and, with `--routes`, the route table of 19 rows,
  one per combination of physics, couplings, geometry class and study type that an engine of this version solves. The
  JSON has the schema string `axionmultiphysics.physics_catalogue/1` and is the same in every build; the reference page
  `docs/reference/physics_catalogue.md` is generated from it. No command solves a model document yet, and every route is
  marked "not implemented yet" (`"implemented": false` in the JSON). What a planar or axisymmetric flow or heat route
  offers and refuses is checked against its engine by a test of the MPI build. The registry is built on first use, so a
  command that does not ask for it never builds it.
- **The parser of the model document (beta)** in the C++ core (`axionmultiphysics/spec_model.hpp`): `is_model_document`
  (an object with the key `physics`), `parse_model_collect`, which reads each section of a document (parameters,
  symmetry, regions and curves, materials, mesh, domains, boundaries, reactions, physics, couplings, study, outputs) and
  reports every grammar error that does not hide another, and `parse_model`, which raises the first. The names a
  document uses are checked against the physics catalogue (types, options, condition kinds, data keys, sources, coupling
  terms, quantities per geometry class), the sections of an electromagnetic design are refused beside `physics`
  (`model.em_sections_not_combinable`), the materials take the new blocks `fluid`, `electrolyte`, `electric` and
  `species`, `thermal.cp_J_kgK`, `mechanical.alpha_1_K` and property laws (including `density`, `mechanical.E_Pa` and
  `nu`), and the boundary, source, coupling and law expressions of a document are compiled (`initial` is kept as written
  until the catalogue has initial keys). The time `t` is a variable of the field expressions of a physics that a
  transient step of the study solves, after the coordinates, and a parameter named `t` is refused there. The names
  `axis`, `exterior` and `remaining` are reserved for the boundary grammar and refused as the name of a region, a curve
  or a domain; an output name is an identifier that is not a constant or a function of the expression grammar, and an
  output takes the keys of its kind (`component` of the geometry, `over` is `"time"`); a transient step needs
  `time.t_end_s`; a boundary that names itself, directly or through others, is one `boundary.cycle`; a `group` selector
  needs `mesh.file`; the values of `study[].solvers` have the type of the catalogue's setting (`study.solver_value`);
  the built-in materials give the same properties as a document material with their data. A parameter that shadows a
  variable is reported once for each set of variables. The parser of an electromagnetic design is unchanged;
  `axmp validate`, `axmp resolve`, `axmp explain` and `axmp patch` take a model document (see the switch below).
- **The switch between the two parsers (beta).** A document with a top-level `physics` key is a model document:
  `axmp validate` prints its validation report (`ok`, `name`, `parameters`, and every finding under `errors`,
  `unavailable`, `deferred`, `warnings`, with its `code` when it has one; with an error, `ok` is false, the first
  error's fields are also at the top level and the exit status is 1; `--strict` treats the first `unavailable` finding
  the same way), and `axmp resolve` prints its resolved geometry in the shape it has for an electromagnetic design
  (`mesh` carries the keys of a model document). `_core.validate(design, strict=False, all_errors=False)` and
  `Design.validate(strict=False, all_errors=False)` return that report, raise the first error (with `strict`, also the
  first unavailable finding) unless `all_errors`; `_core.resolve` and `Design.resolved_params()` take a model document
  as well. Solving one (`axmp solve`, `_core.run_3d`, `Design.solve`, the MCP tools `solve_design` and `sweep`), and
  meshing a 3D one or one on an imported mesh (`axmp mesh`, `_core.mesh`), are refused with `route.not_implemented` and
  `details.operation`; a planar or axisymmetric one is meshed (see "Bounded 2D meshes and named boundaries"). The
  operations that work on electromagnetic designs only refuse a model document by name, with `model.em_only_operation`
  (`details.operation`, `details.serves_model_documents`): `axmp sensitivities`, the classes `_core.Model`, `Model3D`,
  `XsecModel`, `Fw2DModel`, `Design.gradient`, `Design.optimize`, `axmp.optimize`, `axmp.global_optimize`,
  `sensitivity.optimize`, `convert.to_3d`, the COMSOL export (`axmp export-comsol`, `comsol.export_comsol`, for a
  design, a design `.json` or the `design.json` of a run directory) and the MCP tools `make_3d`, `set_fullwave`,
  `set_hts`, `sensitivities`, `optimize_design`, `optimize_global`, `optimize` and `export_comsol`, before anything is
  changed or started; `axmp export-comsol` prints an error that has a code (that refusal, or
  `model.section_needs_physics`) with the fields of the MCP tools' error payload (`code`, `where`, `hint`, `details`;
  its `error` is the whole message), and an error without a code as before. The parser of an electromagnetic design
  refuses a section of a model document (`domains`, `boundaries`, `reactions`, `couplings`, `study`, `outputs`) in a
  document without `physics` (`model.section_needs_physics`), before an unknown key. `axmp` and `axmp.load` refuse a
  model document whose file repeats a key in an object (`json.duplicate_key`, at the pointer of the second occurrence;
  `_core.scan_duplicate_keys(text, origin)`; `axmp` scans the text it parsed, so a file read from a pipe such as
  `/dev/stdin` is scanned too); an electromagnetic file keeps the last value, as before. An electromagnetic design
  without those six keys gives the output it gave (before, such a key was an unknown field).
- **Validation of model documents (beta)** in the C++ core (`axionmultiphysics/model_validate.hpp`,
  `axionmultiphysics/model_graph.hpp`). `validate_model` reads the grammar of a document and then checks it against the
  route table: the route of each group of coupled physics, or the inventory of what is missing when the combination has
  none; the condition kinds, option values, material data and uniform properties that the route's engine takes; property
  laws; the coupling strategy and continuation of each study step; the geometry order. It reports every finding it can
  in one pass, of four kinds: errors, unavailable (the document is valid and this version cannot solve it), deferred
  (checks that wait for the resolved geometry or for an imported mesh file: which entities each named boundary selects,
  whether the conditions of each physics cover the boundary of its domain exactly once, the element types and the
  geometry order of a mesh file) and warnings. The coupling graph of a step has the coupling terms and the dependences
  that property laws create as edges, and its strongly connected components in the order in which they are solved.
  `axmp validate` and `Design.validate()` print or return its report. The codes of the findings are one table, which
  `axionmultiphysics._core.error_codes()` returns and from which the reference page `docs/reference/error_codes.md` is
  generated. Among its rules: a force or a torque output must belong to a flow (an output that names a physics that is
  not a fluid.incompressible, or that is written in a document without one, is `output.force_not_offered`); in an
  axisymmetric model the `component` of a force or torque is `z` only (the flow engines return the axial force and
  torque); a continuation scale `"viscosity"` needs a stationary Navier-Stokes flow without the buoyancy term (the
  Stokes and Boussinesq engines and the transient scheme have no viscosity schedule: `study.scale_not_applicable`); a
  table law for Poisson's ratio is held to -1 < nu < 0.5 at every value; a misspelt end of a coupling leaves out of the
  route checks only the instances that the term could have at that end; a parameter that clashes with the variables of
  two physics is one finding; `boundary.reserved_name` is reported once, where the boundary is defined; the deferred
  resolution of the named boundaries leaves out those already refused. An output may be named `argument` or
  `n_arguments` (neither is a function of the expression grammar).
- **Explain, patch and schema (beta).** `axmp explain design.json [--json]` and `Design.explain(json=False)` list the
  resolved model of a model document: the parameters, materials, regions and domains, the boundaries with their
  selectors (the resolution to entities is deferred to `axmp boundaries`), each physics with every option tagged
  `[default]` or `[written]`, its boundary conditions and the form the engine gives each, the coupling terms, the mesh,
  the study with the coupling strategy and the solver defaults of each step, the outputs, and the route each group of
  physics takes with what it needs. The JSON form has the schema string `axionmultiphysics.explain/1`. A step's `time`,
  `save`, `initial_from` and `given` are listed, tagged `[written]`, and are in the JSON study entry; a document without
  geometry has no symmetry in its route line, no dangling `on` for an instance without regions and no empty
  `parameters`, `regions` or `outputs` line; `axmp explain` of a JSON array says "expected an object". A document that
  validation refuses is refused with its first error. `axmp patch design.json ops.json [-o OUT]` and `Design.patch(ops)`
  apply an RFC 6902 JSON Patch to a copy and validate the result, for model documents and electromagnetic designs alike:
  either every operation applies and the result validates, or the document is unchanged and the error is either the
  failed operation (the new code `patch.operation_failed`, `where` `/<index>`) or, for a result that does not validate,
  the validation's own error. `axmp schema` now describes model documents too: `if` the key `physics` is present the
  model rules apply (generated from the physics catalogue; the seven model sections carry `"x-status": "beta"`), `else`
  the electromagnetic rules, which are unchanged; the required keys are in the branches, so a geometry-free model
  document needs no `regions`. The patched document has its keys in sorted order. The page [design
  format](docs/design_format.md) has a section on model documents (beta): what they contain and what they can do in this
  version (parse, validate, explain, patch, schema, and mesh a planar or axisymmetric one; not solve).

#### Bounded 2D meshes and named boundaries (beta)

- **Bounded 2D meshes and named boundaries of model documents (beta).** `axmp boundaries` (and `Design.boundaries()`)
  lists the named boundaries of a planar or axisymmetric model document with the checks on them: the partition rule of
  every physics, empty selections, partial matches and the outputs on resolved boundaries, which `axmp validate` can
  only defer. `axmp mesh`, `_core.mesh` and the new `axmp.mesh(d)` mesh such a document (`MeshedModel`, `InstanceMesh`
  with `face_points` for flux integrals), after the same checks. New findings: `build.requires_mesher` (unavailable,
  when the mesher does not run), `mesh.generation_failed`, `model.shape_not_offered` (a contour with an elliptical
  segment), `boundary.between_overlap` (at validation), and the raised `physics.boundary_double`,
  `physics.boundary_interior`, `physics.boundary_outside_domain`, `boundary.partial_match`,
  `output.boundary_not_in_domain`, `output.force_on_interior` and `internal.mesh_export`. `axmp validate` now probes the
  mesher. A `where` filter whose box lies strictly inside an edge's box in every coordinate is not a partial match of
  that edge. 3D model documents and model documents on an imported mesh are not meshed yet (`route.not_implemented`),
  and neither is a document with no geometry by `axmp.mesh` (`axmp mesh` prints a note). The partial-match rule decides
  a straight edge exactly (the segment against the filter's box; an edge that only touches it is not reported) and keeps
  the enclosure exception for curved edges. `axmp validate` takes `--mesher M` for its probe, and `axmp mesh` removes an
  earlier mesh in `-o` before it starts. When the mesher fails, the error's hint ends with the last lines of the
  mesher's log, so the reason survives when the working directory was a temporary one that is removed. See [Meshes of
  model documents](docs/guides/model_meshes.md).
- **Instance meshes for the fluids and chemistry engines (beta).** `InstanceMesh.fluids_mesh()` returns a `fluids.Mesh`
  that is a copy of the instance mesh (new `fluids.Mesh.from_instance` and read-only `Mesh.has_nodes`), and
  `InstanceMesh.chem_mesh(order=2)` a `chem.TriMesh` subclass that records `geometry_order` and gives the exported P2
  `edge_nodes()`; `MeshedModel.fluids_mesh(domain=None)` and `MeshedModel.chem_mesh(domain=None, order=2)` are
  shorthands. In a build without the fluids module `fluids_mesh()` raises the new unavailable code
  `build.requires_fluids`. A field of order 1 on a mesh with nodes is refused with a `ValueError` that names the remedy,
  also by `chem.CurrentDistribution(mesh, order=1)` and `(mesh, order=2, curved=False)` on a mesh that records geometry
  order 2, and by `chem.meshes.red_refine` of a curved bridge mesh. Engine acceptance tests run through the bridge (heat
  slab, flows through two regions, the axisymmetric pipe with its axis, Poiseuille flow with both viscous forms, the
  parallel-plate cell) against closed forms, with tolerances recorded per processor model. See [Meshes of model
  documents](docs/guides/model_meshes.md#handing-a-mesh-to-the-engines).

#### Export to Geant4 (beta)

- **Field maps for particle transport (beta).** `Results.export_field_map(path, ...)` (and `Results3D.export_field_map`)
  writes the static field of a solved design as per-region patches in the directory format
  `axionmultiphysics.fieldmap/1`: one grid per region (adjacent regions with the same law share one; a closed
  superconducting sheet or thin magnetic layer splits its region, an open one is refused, a conductor sheet is no
  interface), each holding the region's recovered field and its continuation beyond the region, refined until the RMS
  error of the map's integrals along fixed chords meets `tol` (or a tenth of the FE indicator), with a two-halving
  guard, caps, track volumes, an outside rule with a clamping margin of the level-0 grid spacing, and a
  `FieldMapWarning` for a patch that is not accepted. `axionmultiphysics.fieldmap` reads (`load`), writes from nodal
  values (`write_map`), evaluates (`FieldMap.evaluate`) and checks maps (`recovered_field`, `chord_error`). See [Field
  maps for particle transport](docs/guides/field_maps.md).
- **Solids for particle transport (beta).** `axionmultiphysics.solids.from_partition` turns an axisymmetric partition
  (the mesher's entity table extended by the exact curve parameters and the oriented loops of every face) into the
  solids list `axionmultiphysics.solids/1`: one tube, torus, orb or generic polycone per face, revolved by a full turn
  with its notches at the axis filled, placed in the face that contains it; holes of the domain are vacuum daughters;
  arcs that are not native are cut into chords once per curve, within min(1 um, 1e-5 x the region's size) by default, so
  that neighbours share every vertex. Per region it reports the exact (Pappus) volume and mass against those of the
  solids, with a `particle.volume_deficit` warning above 1e-4. `solids.write` writes the list as JSON and
  `solids.write_gdml` as GDML (17 significant digits; NIST materials and elements referenced by name and never defined;
  any region or material name is mapped to a valid, unique GDML name, carried as `gdml_name`). A layer thinner than
  about the chord tolerance, whose solids would overlap or lose their place in the tree, is refused
  (`particle.partition_invalid`, with the hint to lower `chord_tol_m`; a discretized curve that is concave towards a
  solid it touches, and a face pinched at a point, are refused). Materials need a `particle` block (`nist`, or
  `elements` by mass or by atoms, up to Z = 107) and a density; the new codes are `particle.planar_not_offered`,
  `particle.partition_invalid`, `particle.composition_missing`, `particle.density_missing` and
  `particle.composition_invalid`. `axmp-transport --geometry solids.json` builds the list in Geant4 under the GDML
  names, runs Geant4's overlap check on every placement (an overlap is `particle.geometry_overlap`) and prints each
  volume's Geant4 volume and mass and each material as Geant4 built it (density, mean excitation energy, state,
  temperature, pressure, elements) and the name of each placement, as Geant4's GDML reader names it; `--check` reports
  `"geometry": true`. See [Solids for particle transport](docs/guides/solids.md).
- **GDML of an axisymmetric electromagnetic design (beta).**
  `Design.export_gdml(path, composition, *, solids=None, chord_tol_m=None, workdir=None, mesher=None)` writes the GDML
  (and optionally `solids.json`) of an axisymmetric electromagnetic design and returns the solids document: the design's
  mesher input, `axmp-mesher --export-regions`, then `solids.from_partition`. `axmp-mesher --export-regions input.json`
  runs the construction of the electromagnetic mesh and writes the exact partition of the geometry (regions with exact
  areas and volumes, the entities with the curve parameters, the faces with their loops) to the input's `info` path
  instead of a mesh. The entity table of every 2D mode that writes one (the bounded planar and axisymmetric modes, and
  the export) has `start` and `end` per entity, `center`, `radii`, `major_axis` and `ccw` per arc, and `faces`. The
  mesher resolves the trimmed curves that framed, filleted and Boolean shapes leave to their lines and arcs, so the
  bounded modes mesh such shapes too. New code `particle.partition_unavailable`. See [Solids for particle
  transport](docs/guides/solids.md#electromagnetic-designs).
- **GDML of a 3D electromagnetic design (beta).** `Design.export_gdml` also exports a 3D design, in every build (the 3D
  mesher input is made by a library function, `_core.mesher_input_3d`, that the 3D engine also uses; no engine run, each
  MPI rank exports on its own; a design with `mesh.symmetry` is refused with `particle.partition_unavailable`).
  `axmp-mesher --export-regions` accepts the 3D mesher input and writes the 3D partition: the faces as entities of `dim`
  2 with their two side regions (`exterior` beyond the domain sphere, the infinity shell is not part of it), the
  `volumes` with signed shells, and one surface triangulation shared by every face, made to `chord_tol_m` (default
  `min(1 mm, 1e-4 x diagonal_m)`; the new key of the input, and the `chord_tol_m` of `export_gdml`).
  `solids.from_partition` turns it into one `tessellated` solid per volume (daughters by cavity, a vacuum orb as the
  world), `solids.write_gdml` writes `tessellated` solids of `triangular` facets, and `axmp-transport --geometry` builds
  them as `G4TessellatedSolid`, refusing a solid whose triangles do not close or enclose no positive volume (Geant4 only
  warns). The conversion refuses a partition whose triangulated solids do not nest, overlap or cross themselves (a layer
  thinner than the chord tolerance: `particle.partition_invalid`, lower `chord_tol_m`). A domain without regions exports
  the air. The mesh of a 3D design without the flag is unchanged. See [Solids for particle
  transport](docs/guides/solids.md#3d-designs).

#### Particle transport (beta)

- **Particle transport toolchain (beta).** `axmp-transport`, a separate program that links Geant4 11.2 to 11.4 (tested
  with 11.4.3) and exchanges files with the rest of the package. It is built only with `-DAXMP_BUILD_TRANSPORT=ON`,
  which is off by default and forced off in wheels; the configure refuses a Geant4 outside that range or built without
  multithreading. It has `--check`, a hello-world mode (muons through a silicon slab in a field) and the transport modes
  below, with per-event seeds that depend only on the chunk seed and the event index, so that results are bitwise
  identical for any number of threads. See [Particle transport](docs/guides/particles.md).
- **Particle transport driver and the transport modes (beta).** `axionmultiphysics.particles` plans a run into chunks
  with seeds derived from one master seed, runs `axmp-transport` on them (a pool of processes, a failed one retried
  once), merges the longest complete prefix in (chunk, event) order, extends a run bitwise equal to a direct run, stops
  on a relative error evaluated on prefixes, and estimates the outputs (per-event deposits, hits and the primary; means,
  standard deviations, quantiles, fractions with Clopper-Pearson intervals, counts, exact 95 % limits after zero events,
  histograms, tallies with exact 128-bit batch sums, hit records) as `Measured` values with their statistical errors
  (batch means over 20 batches). `particles.transport(design, results, ...)` transports particles through a solved
  design's field map (`solids=` is required: it does not run the partition export itself yet; `Design.export_gdml`
  returns a solids document). `axmp-transport` has the modes `transport` (a gun through a solids list, a field object
  and a field manager per logical volume with Geant4's templated Dormand-Prince 4(5) and `G4IntegrationDriver` pinned,
  uniform, current-loop and field-map fields, the world and voids ending the domain) and `probe` (the fields at given
  points), `--nist` (Geant4's own NIST material data) and, in `--check`, the physics lists `FTFP_BERT`, `FTFP_BERT_EMZ`
  and `QGSP_BIC`, the stepper and the particle table. NIST compositions are resolved against the program's tables before
  every run: at NIST's density the solids use the NIST material itself (its GDML references it by name), otherwise the
  one-component material carries NIST's I-value and state. The command line
  `python -m axionmultiphysics.particles {execute,merge,extend}` exits with status 5 for a partial result (2 for a usage
  error). One driver works on a workdir at a time (an exclusive lock; `events.workdir_busy`); seeds lie in [0, 2^64).
  The world and the voids end the transport's domain only where a track really is in them, never at an interface between
  two regions where Geant4 reports the world for one zero-length step, and such a step records no hit. The program also
  refuses a deltaIntersection above a tenth of the finest map spacing. A field map inside the workdir is named relative
  to it in the job files (a copied workdir reads its own map), and `run.json` records each map's manifest SHA-256, which
  `execute`, `extend` and `run` check before any program starts (`events.field_map_changed`). A track Geant4's navigator
  abandons as stuck is ended and counted per event (`abandoned_killed`, `abandoned_eV`; warning
  `particle.track_abandoned`) instead of ending the process. Each program writes a chunk through a temporary directory
  of its own. A gun outside every region is refused (`particle.source_invalid`), and the program refuses a primary
  vertex outside the world (exit status 2). See [Particle transport](docs/guides/particles.md) and [its
  validation](docs/validation/particles.md).
- **The process rule for Geant4 and Pythia.** AGENTS.md ("Dependencies") and the licensing table of the install guide
  state it, and `tests/test_process_rule.py` checks that the library, the Python module and `axmp` link none of them and
  that importing the package imports no Geant4 or Pythia binding.

#### Errors and expressions

- `axmp::Error` takes two optional fields, a machine-readable `code` and `details` (the JSON text of an object); neither
  enters the message.
- `AxionMultiPhysicsError` in Python has the attributes `code`, `where`, `hint` (strings, empty by default) and
  `details` (a dict, or `None`; `None` too when the details have no members or are not strict JSON, which is when the
  command line prints none). An error of `Design.solve(mpi_ranks=N)` has the four attributes as rank 0 reported them;
  its message is unchanged.
- The command line and the MCP server add `code` and `details` to an error's JSON when the error has them.
- `AxionMultiPhysicsError` has a docstring that names its attributes `code`, `where`, `hint` and `details`.
- Error details nested deeper than 64 levels (the details object is level 1) are reported as no details, on the command
  line, in the MCP payload and in Python alike, instead of being read recursively without bound.
- The errors that an electromagnetic design could already meet in 0.1.0 have no code, so no existing output, exception
  type or message changes. Codes are carried by the findings of model documents (`docs/reference/error_codes.md`), by
  the refusal of a model document's section in a document without `physics` (`model.section_needs_physics`, see the
  switch above), by the new refusals of arc paths and tangled meshes (`geometry.arc_span`, `geometry.arc_degenerate`,
  `geometry.arc_negative_r`, `mesh.tangled`; see Fixed) and by the errors of the new modules (`build.requires_*`,
  `fieldmap.*`, `particle.*`, `events.*`; their guides list them).
- Compiled field expressions in the C++ core (`axionmultiphysics/field_expr.hpp`). `FieldExpr::compile` parses an
  expression of the design grammar once, with free variables of the context (the coordinates, the time of a transient
  step, a field a property law reads), and evaluates it at one point or at many, with exact first derivatives by forward
  mode over the compiled tree. A text in which no variable occurs gives the same double as the design-parameter
  evaluator `eval_expr` gives, bit for bit, and is refused for the same reason; a parameter that has the name of a
  variable is refused where the expression is compiled, rather than shadowed in silence. Evaluation is re-entrant, so
  one compiled expression serves the several threads of an assembly loop. Where a first derivative is a convention
  rather than a formula — the kink of `abs`, the argument `min` and `max` select on a tie, a power that is constant in
  the operand being differentiated — the header states the convention. No key of an electromagnetic design reaches it,
  so nothing an electromagnetic design can contain has changed; the parser of model documents compiles their expressions
  with it.
- `FieldExpr::compile` takes an optional last argument, a context phrase for the message that refuses a parameter named
  like a variable ("parameter 'y' has the name of a coordinate of this planar model; rename the parameter"); that
  refusal has the code `expr.parameter_shadows_variable` and the details `{"parameter", "variables"}`. Without the
  argument the message is the one it was.

#### C++ API

- C++: `AxiModel::elasticity()` and `AxiModel::elastic_solution()` give the elasticity problem and its solution from the
  last solve with mechanics enabled (`elasticity()` is `nullptr`, and `elastic_solution()` throws `std::logic_error`,
  when there is none).

#### Validation and compatibility

- The compatibility gate G0 (`tests/g0/`, `tests/test_g0.py`, `docs/developer/g0.md`, `.github/workflows/g0.yml`)
  records what an electromagnetic design observes of axionMultiPhysics (the text and exit status of `axmp validate`,
  `axmp resolve` and `axmp template` for every design under `examples/`, `benchmarks/designs/` and `tests/g0/designs/`,
  and, for a list of solves and meshes, the files and numbers they write) and compares every later build with the
  record: tier A the text, tier B the bytes (on one pinned platform), tier C the numbers on other platforms. It is how a
  change to the code that electromagnetic designs execute is shown not to change their behaviour. For a user nothing
  changes; for a contributor a declared change of a message or a number is a separate, argued update of the record. The
  workflow runs on the release tag `v*` and on pull requests in the public repository.

### Changed

#### Fluids (beta)

- **Fluids (beta): one recorded retry chain for every direct solve.** Every sparse LU of the fluids solvers (Stokes 2D
  and 3D, the steady and transient Navier-Stokes steps and the initial projection, Boussinesq, GLS, scalar transport,
  and the dual-norm mass solves) walks one chain of configurations and accepts a configuration on the caller's own
  accuracy gate: the projection on its gate (`projection_residual <= 1e-13`), every Navier-Stokes step on its relative
  residual at the final step's bound `max(newton_rtol, 1e-11)`, Stokes at `direct_tol`, the mass solves at `1e-10`;
  Boussinesq and GLS steps (`max(newton_rtol, 1e-11)`) and the scalar transport Jacobian (`max(1e-2 rtol, 1e-11)`) get
  thresholds of their own, recorded (their LU had no gate). Default chains: STRUMPACK with MC64 matching (job 5), then
  STRUMPACK with MC64 job 2, then Eigen's LU up to 3e5 unknowns (measured: job 2 accepted every 2D system that job 5
  missed, 34 of 34 on an AMD EPYC 7662 and 30 of 30 projections on an Intel Xeon Gold 6330, among them the six planar
  `k = 5` Womersley starters, in 0.31 s to 0.33 s per factorization against 4.0 s to 4.3 s for Eigen's LU; the
  configuration without matching rescued none of them that job 2 did not, and threw ZERO_PIVOT on all six starters); in
  `solve_stokes_3d` STRUMPACK without matching, then with it (the former matching retry; with `lu_matching=True`: with
  matching, then without), then Eigen up to 5e4 unknowns; Eigen first: no retry. The fixed 1e-6 retry threshold of the
  time steps is removed (a semi-implicit step solve at 1e-7 passed it although the step's gate is 1e-11), the steady
  strategies are on the chain too, and a rejected configuration no longer leaves its error text in `failure`. New
  keyword `lu_retry="auto" | "off"` on every entry point (`"off"`: the first configuration alone, judged and recorded
  like any other, its solution returned for the caller's gate to decide; in `solve_stokes_3d` that is STRUMPACK without
  matching alone, which does not reproduce an older run whose matching retry fired: the default chain does). New
  diagnostics: `lu_stage` (the configuration that produced the solution; `lu_backend` is now its backend, `solver` stays
  the configured one), `lu_attempts` (one record per configuration tried, with its outcome, reason and measure),
  `lu_backward_error`, the counts of the call (`lu_solves`, `lu_escalated`, `lu_escalated_final`, `lu_attempts_total`,
  `lu_rejected`, `lu_failed`, `lu_accepted_by`, `lu_rejected_seconds`, `mass_lu_escalated`), `projection_lu_attempts`
  and `lu_kernel_probe`; history entries carry `lu_attempts` and `lu_stage`, and `TransientNavierStokes.stats` sums the
  counts over the steps (the projection's attempts are no longer booked to the first step). `lu_fallbacks` now counts
  the solves of the call that escalated (it was a counter over the solver's lifetime) and is kept, with
  `lu_fallback_backend` and the 3D `lu_matching_retry` / `lu_matching_retry_reason`, as a derived alias for one release
  (`lu_fallback_backend` now names a stage, e.g. `strumpack:none` where it read `strumpack-no-matching`;
  `lu_matching_retry_reason` is the first configuration's own reason, without the prefix "matching-free factorization
  failed: "). The kernel probe recognises a correctly bordered system at small length units (the square of side 1e-6
  with the mean-value multiplier was refused). The initial projection with the mean-value multiplier now passes on the
  squares of side 1e6, on `1e6 x 1e4` and on `1e7 x 1e4` (N = 32), which failed: their first STRUMPACK configuration is
  accurate on the projection's gate (5e-17) but missed the 1e-6 floor, and on `1e7 x 1e4` (N = 32) the first
  configuration is refused on the gate (0.051 on an EPYC 7662, 0.060 on a Xeon Gold 6330) and STRUMPACK with MC64 job 2
  passes (5.8e-17); a projection that no configuration passes raises an error that lists the attempts and names the
  other pressure treatment (the pinned pressure, `nullspace="pin"`, after the multiplier: it passed on the measured
  domains where the multiplier was refused, but with Eigen's LU (`lu="eigen"`) it is refused itself on `1e9 x 1e4` (N =
  32 on a Xeon Gold 6330, N = 16 on an EPYC), where the multiplier passes; the default STRUMPACK chain accepts the pin
  there with MC64 job 2). Results change only where a solve escalates (first configurations are unchanged); the
  published records, solved with Eigen alone, keep their values (they gain the new fields). In a STRUMPACK build the
  solves that escalated now end in STRUMPACK with MC64 job 2 instead of Eigen's LU: the six planar `k = 5` Womersley
  starters (`lu_accepted_by`, `lu_attempts` and the factor times change; replayed on an EPYC 7662, the accepted
  solutions agree with Eigen's to 1.1e-15). The cost of gating every step, measured on the published fluids cases in
  quick mode with STRUMPACK (the 72 cases that solve in the calling process; 15 664 direct solves of nonlinear and
  linear solves, 144 initial projections): no solve of a problem with a solution escalates, and no projection or mass
  solve does; the only escalations are those of the two negative controls whose systems are singular by design (the
  P1/P1 GLS variants without PSPG: 80 intermediate and 2 final step solves that no configuration accepts, 1.6 s of
  rejected configurations in a 2.8 s case; the unstable P1/P1 Stokes pair: 2 solves). The 3D cases solve in MPI workers
  and were not counted: their first two configurations are the former matching retry, unchanged (their records differ
  only in the text of `lu_matching_retry_reason`, now the first configuration's own reason). The documentation names the
  backend, the pressure treatment and the processor for every measured number, and the remedy it names for a refused
  projection is the pinned pressure (`nullspace="pin"`): on `1e6 x 1e4` (N = 16) the refused projection (4.4e-12, Xeon)
  is Eigen's LU with the mean-value multiplier, the last configuration of the former chain; STRUMPACK's first
  configuration with the multiplier passes there (5.3e-17 on an EPYC 7662, 5.4e-17 on a Xeon Gold 6330), and both
  backends give about 5e-17 with the pinned pressure.
- The initial Leray projection of `TransientNavierStokes.initialize` is gated on the backward error of its saddle solve
  against `1e-13`, at every `newton_rtol`, and recorded as `projection_residual` and `projection_converged` in the
  diagnostics (the fixed 1e-6 acceptance and the discarded tolerance are gone). `projection_residual` is, for the
  momentum and for the continuity block, the norm of the residual over the norm of the sizes of the terms of that block,
  and the larger of the two: 4e-17 to 6e-17 for an accurate solve whatever the mesh, the aspect ratio of the domain, the
  length unit and `newton_rtol` are (measured with Eigen's LU on 578 to 132 098 velocity dofs, aspect ratios up to 1e5
  and lengths from 1e-6 to 1e6, on an AMD EPYC 7452), so neither a long channel, nor a change of length unit, nor a
  tight `newton_rtol` turns an accurate projection into a refused one. STRUMPACK returns measurably inaccurate
  projections on some scaled or stretched domains, and which ones depends on the processor; the gate refuses those that
  miss the bound, and the retry chain (above) tries its next configuration. A projection that no configuration passes
  makes `initialize` return `False`, and `advance`, `step_trial`, `adapt` and `snapshot` then raise an error naming the
  projection, its attempts and the other pressure treatment instead of stepping from an invalid state.
- Every Navier-Stokes history entry carries `lu_residual`; the diagnostics carry `lu_residual_max` and
  `lu_target_missed`, and each transient step result and `stats` carry `lu_residual` / `lu_residual_max`. A final step
  whose direct solve is above `max(newton_rtol, 1e-11)` now makes the solve report `converged = false` (intermediate
  steps are recorded, not enforced; the C++ option `NonlinearOptions::final_solve_residual_max` overrides the bound).
- Every dual-norm mass solve is checked: `mass_lu_residual_max` is reported (Navier-Stokes, transient, scalar transport,
  Boussinesq, GLS) and a value above `1e-10` makes the solve report `converged = false`. GLS also refuses a
  triangulation whose mass matrix cannot be Cholesky-factorized (for example a vertex that no triangle uses) when it is
  built.
- `quad_order_convection` whose rule does not integrate the convective form exactly is refused (`axmp::Error`, naming
  `NavierStokesOptions.quad_order_convection`). The default rule is now the exact degree for the element shape and the
  geometric order `q` of the mesh: on triangles `3p + q - 2` (planar) / `3p + 2q - 2` (axisymmetric), which is unchanged
  on straight triangles (`3p - 1`, `3p`); on quadrilaterals `3p + q - 1` / `3p + 2q - 1` per direction. **Computed
  solutions change on planar quadrilateral meshes and on curved (isoparametric) meshes**, where the earlier default
  (`3p - 1` planar, `3p` axisymmetric, whatever the element) under-integrated the convective term: on straight planar
  quadrilaterals the Q2 rule goes from 3 x 3 to 4 x 4 Gauss points (degree 5 to 6), and a lid-driven cavity at Re = 100
  moves by 6.8e-3 of its largest velocity on 6 x 6 quadrilaterals and by 1.6e-3 on 12 x 12 (velocities at the element
  centres). On straight axisymmetric quadrilaterals nothing changes: the exact degree is 7, and the earlier degree 6
  selects the same 4 x 4 Gauss rule. C++: `ConvectionIntegrator::exact_degree(element, transformation, axisymmetric)`
  and `rule_is_exact(...)` replace `quad_order(int)` (see Removed), and an integrator built with a rule that is not
  exact refuses to assemble unless its constructor is given `allow_under_integration = true`. No published validation
  case changes with the default rule: the Navier-Stokes cases run on straight triangles, except
  `analytic_mhd_annulus_variable_b` (isoparametric P2), whose density is zero, so that its convective term vanishes on
  any rule.
- The multi-rank comparison of the C++ Navier-Stokes tests is registered as the ctest `test_fluids_navier_stokes_mpi4`,
  which fails if the launcher does not start its four ranks as one MPI job.
- The stop of a steady solve at the round-off floor of its residual (above) and these gates are independent: a floor
  stop passes the gates on the final direct solve and the mass solves, and a floor stop whose final direct solve misses
  its bound reports `stopped_at_floor = true` with `converged = false`.
- Documentation: the STRUMPACK failure on large transient systems is described as measured (deterministic and
  matrix-dependent, not occasional); the cavity cases state the Picard count the runner uses (`n_picard = 0`).
- Documentation: the guide to the fluids module says that the transient Navier-Stokes time stepper takes its steps on
  any number of MPI ranks (the model-matrix test runs them on two), while point evaluation of a stored state needs one.

#### Validation pages

- Validation pages: `docs/validation/fluids.md` now reports 79 of the 81 cases run in full, and all 79 meet all their
  criteria (the 0.1.0 page: 67 run in full, 66 meeting them); the twelve cases that were left in quick mode (the seven
  3D Stokes cases, the four Womersley cases and `mms_stokes_2d_periodic`) were run in full on 2026-11-17 in three builds
  (serial STRUMPACK, MPI STRUMPACK, Eigen fallback), and `analytic_hartmann_2d` was run again on 2026-11-18 after the
  change of its stopping rule (see the floor stop under Fixed). The page says which numbers are generated and which were
  corrected by hand, and that the 2026-11-17 Eigen runs of six 3D Stokes cases crashed (the Eigen entry under Fixed
  addresses that crash; those runs predate it). `docs/validation/chemistry.md` reports all 39 cases run in full and
  meeting all their criteria (the 0.1.0 page: 37 of 39), with the two 3D current-distribution cases run in full on
  2026-11-17 in the same three builds. The raw results of these runs are not published.

### Fixed

#### Meshes and geometry

- **Tangled curved elements in 2D meshes.** At geometry order 2, a curved region much thinner than the mesh size (a 2 mm
  shell on a sphere of radius 0.5 m at a 0.05 m mesh size: 32 of its 64 triangles) gave triangles that fold over, and
  `axmp-mesher` wrote them with `"ok": true`; the solve then ran on them. The mesher now checks every element of a 2D
  mesh of geometry order 2 (axisymmetric, cross-section, and the bounded meshes of model documents): an element is
  tangled when its Jacobian determinant is not of one strict sign over the element (computed exactly for 6-node
  triangles, whose det J is quadratic), or is inverted against the other elements meshed in one winding with it (the
  majority per gmsh surface, so a region whose parts gmsh meshes in opposite windings is not tangled). A tangled mesh is
  repaired by gmsh's high-order optimisation, with the nodes of curved boundaries kept on the curves, in the curved-mesh
  process of the 3D path, so that a repaired mesh is the same in every run. The mesh info then has
  `"high_order": {"checked", "tangled_before", "optimised", "tangled_after"}`. A mesh still tangled is refused with the
  count, the region and the remedy (refine the mesh in or around that region, or use geometry order 1), and
  `Design.solve` and `axmp mesh` report it as a failed mesher run. `AXMP_MESHER_HO=none` switches the repair off in 2D
  too. Meshes with no tangled element are written as before, byte for byte, without the new key.
- **The library refuses a tangled 2D curved mesh when it reads one** (the axisymmetric and cross-section solves, an
  imported 2D `mesh.file`, the meshed model of a model document), whoever wrote the file: the new code `mesh.tangled`,
  with the count, the file, the region and location of the first tangled element, and the remedy in the hint. It is
  raised before anything is solved, for electromagnetic designs too. Valid meshes are read and solved as before. The
  field-map builder's own refusal (`fieldmap.tangled_mesh`) stays as a second guard.
- **Arc paths that abort the mesher are refused at validation.** An `arc` path whose angles span more than one turn
  (typically degrees written where radians are expected, such as `[-60, 60]`) now fails `axmp validate`, `axmp mesh`,
  `Design.validate` and `Design.solve` with `geometry.arc_span` at `<path>/arc/angles`, and the hint gives the values in
  radians. A radius that is not positive, or an arc of 1e-7 m or less, is refused with `geometry.arc_degenerate`. In
  every electromagnetic design except a cross-section (`xsec`) one (planar designs on an imported mesh included) and in
  an axisymmetric model document, an arc that reaches r < 0 (a centre on the axis, or an arc that crosses it) is refused
  with `geometry.arc_negative_r` at `<path>/arc`. The hint of `geometry.arc_span` prints the converted angles with 17
  digits, so that the suggestion can be pasted. The readers of a 2D mesh (the axisymmetric space, the model's boundary
  faces and the 2D full-wave model on an imported mesh) now stop with an error naming the boundary element when it is no
  face of the mesh, instead of writing at index -1. Both used to abort `axmp mesh` of an electromagnetic design (signal
  6) or fail late in a model document's mesh. A full turn is still accepted, with a slack of 1e-12 for rounding.

#### Fluids (beta), chemistry (beta)

- **Fluids (beta):** the Eigen backend of the fluids direct solver no longer crashes on large systems. Its 32-bit factor
  offsets overflowed (a segmentation fault at the full 3D Stokes refinement); above 46 340 unknowns, or 2^28 matrix
  entries, it uses 64-bit indices, and when it runs out of memory the error names the sizes and the remedies on every
  rank. It runs on rank 0 alone and is limited by that rank's memory (documented, with measured 3D sizes).
- **Fluids (beta):** a steady Navier-Stokes or inductionless-MHD solve with `strategy="newton"` whose step stops
  reducing the residual at the round-off floor of its own residual evaluation now stops there and is reported as
  converged, instead of running its whole iteration budget and failing with "Newton budget exhausted". The floor is
  measured, not assumed: it is the change of the residual, in the relative dual norm the gate uses, when every entry of
  the state is moved by one unit in the last place with alternating sign. Three conditions must hold, so that the stop
  neither cuts a converging iteration short nor loosens the meaning of `converged`: the last step reduced the residual
  by nothing at all, the residual is within `residual_floor_factor` of the measured floor, and it is still within a
  decade of `newton_rtol`. The diagnostics carry the floor (`residual_floor`) next to the new flag `stopped_at_floor`,
  so such a stop is never silent, and the new keyword `solve_navier_stokes(..., residual_floor_factor=...)` sets how
  close to the floor a residual must be (default 4; 0 switches the rule off). `newton_rtol` and every other gate are
  unchanged, and `converged` still bounds `relative_residual`: by `newton_rtol`, or by ten times `newton_rtol` when
  `stopped_at_floor` is set. This changes what `converged` means for a default solve in two ways a caller can see: a
  floor stop accepts the iterate it fired on, so it can report a slightly larger residual than the full budget would
  have reached (measured on the case below with STRUMPACK: 1.14e-12 at the stop, against 9.97e-13 at the twenty-second
  step without the rule, which met `newton_rtol` outright), and the floor is a round-off-level measurement that depends
  on the number of MPI ranks, so a solve sitting at the floor may stop a step earlier or later, or not at all, on
  another rank count. The `strategy="picard"` and `"ptc"` branches, the time steps, `solve_gls` and `solve_boussinesq`
  are unchanged and ignore the new field. The floor grows like h^(-3/2) on a traction-driven problem, which is why the
  hydrodynamic (Ha = 0) level of the verification case `analytic_hartmann_2d` stagnated at a relative residual of
  1.0e-12 against `newton_rtol` 1e-12 and the case's solver criterion failed; the case now meets all of its criteria and
  gained a closed-form comparison of the flow rate on that level, two patch checks on it, a gate on the largest unscaled
  Euclidean residual of its solves, and a check that every one of its solves reported that residual.
- **Chemistry (beta), time core:** the starting-step estimate no longer swallows every exception. Only a derivative that
  is not finite at the trial point (the new `ConsistentDerivativeError`, thrown by `consistent_derivative`) leads to the
  smaller starting step; an exception raised by the problem's own right-hand side, such as a Python exception or
  `KeyboardInterrupt` from a callback, propagates to the caller. `consistent_derivative` now raises
  `ConsistentDerivativeError` for a non-finite derivative at every call site (before it returned a non-finite vector),
  so a non-finite right-hand side is reported where it happens instead of as a later Newton failure. A run whose
  right-hand side is not finite at the trial point now starts from `h0 * 1e-2` and so takes a different step sequence
  than before, with unchanged accuracy; results of runs that did not hit that path are unchanged.

#### Command line, build, installation and C++ API

- **Command line:** `axmp` no longer aborts with a core dump (exit 134) when the text of an error is not valid UTF-8.
  This covers a design file that is not valid UTF-8, a parse error at a non-ASCII character of a valid UTF-8 file (for
  example a typographic quote pasted from a word processor, where the parse error quotes only the first byte of the
  character), and a file or directory name that is not UTF-8. The error quotes the bytes it stopped at, and printing it
  as JSON threw out of the error handler; the invalid bytes are now printed as the replacement character, and the error
  is reported as `invalid JSON in 'FILE': ...` (or `cannot open '...'`) with exit 1 on stdout. An error of `solve` that
  comes after its options are read (for example an output directory name that is not UTF-8) is also in its status file.
  Every output for valid input is unchanged.
- **Build:** a configure with `-DAXMP_STRUMPACK_PREFIX` (an MPI STRUMPACK, which enables Fortran) now stops when CMake's
  GNU Fortran compiler is of an older GCC major version than the GNU C++ compiler, with a message that names both
  compilers. Before, the configure succeeded and the link failed much later with undefined `std::filesystem` symbols
  from the older libstdc++. Configure with `-DCMAKE_Fortran_COMPILER=` set to the `gfortran` or `mpifort` of the C++
  compiler's GCC, in a new build directory or with `cmake --fresh` (CMake 3.24 or newer), because CMake caches the first
  Fortran compiler it finds.
- **Installation guide:** the troubleshooting entry for a crash in a STRUMPACK solve after
  `OpenBLAS warning: precompiled NUM_THREADS exceeded` now covers the crash that `OPENBLAS_NUM_THREADS=1` does not
  prevent. STRUMPACK calls OpenBLAS from all of its OpenMP threads at once, and an OpenBLAS build has a fixed number of
  work buffers, the number its warning names (96 in an OpenBLAS 0.3.24 that crashed intermittently with 128 threads).
  Set `OMP_NUM_THREADS` to at most that number, or use an OpenBLAS built with a larger `NUM_THREADS`.
- C++: `AxiModel::set_mesh()`, `mesh()` and every solve now discard the mechanics and thermal problems of an earlier
  solve (they refer to the replaced space); before, `sample_mech()` after `set_mesh()` without a new solve read a
  dangling space.

### Removed

- Fluids (beta), C++: `ConvectionIntegrator::quad_order(int)` (it knew only the velocity order); `exact_degree` and
  `rule_is_exact` replace it (see the fluids entries under Changed).
- Fluids (beta): the fixed 1e-6 acceptance of the initial projection and the fixed 1e-6 retry threshold of the time
  steps, replaced by the accuracy gates and the retry chain (see Changed).

### Known limitations (beta)

- **Particle transport through 3D solids is not validated.** `axmp-transport --geometry` builds and checks the
  triangulated solids of a 3D design (volumes, masses, overlaps), but no transport through them is tested, and the 3D
  dipole of the demonstration skips.
- **The route from `Design.export_gdml` to `particles.transport(solids=...)` has no test**, in 2D or 3D: the
  demonstration's solids are written by hand from the same regions.
- **NIST compositions away from NIST's density.** Such a material is built as one component that carries NIST's mean
  excitation energy and state, but not the data Geant4 looks up by the NIST material's name (the density-effect table,
  ICRU 90 water, proton stopping tables): its total dE/dx differs from the NIST material's by up to about 2 % (+1.01 %
  for water, +1.99 % for BGO with 1 GeV/c muons; table in [Solids for particle
  transport](docs/guides/solids.md#materials)). Use NIST's density, or an explicit element composition with
  `mean_excitation_eV`.
- **Hits of shallow excursions.** An excursion into a neighbouring volume shallower than deltaChord (0.25 mm) is
  collapsed by Geant4 to one zero-length step: its entry is recorded as a hit and its return is not (measured: 11 such
  pairs among 174 375 hits of 5000 protons crossing slabs at grazing angles, none at normal incidence).
- **PyPI timing.** The 0.2.0 wheels follow on PyPI after the release; until `pip index versions axionmultiphysics` lists
  0.2.0, `pip install axionmultiphysics` installs 0.1.0, whose API lacks the 0.2.0 features. Install 0.2.0 from source
  ([installation](docs/guides/install.md#build-and-installation)).
- **Platforms.** 0.2.0 was built and tested on Linux x86_64 only; macOS was not built or tested for this release.
- Further limits of the beta features are listed in their guides ("Limits of this version"), for example one gun per run
  and statistical errors only in particle transport.

## 0.1.0: initial public release

The first public release. It contains the following capabilities.

### Lumped-element detectors

- Axisymmetric magnetoquasistatic engine: the poloidal (ψ = rA_φ) and toroidal (G = rH_φ) problems on high-order
  H1 elements with an exact open boundary (Dirichlet-to-Neumann map); nonlinear iron (B–H curves); ideal
  superconductors with fluxoid or net-current constraints; stranded windings, persistent circuits and imposed
  background fields; thin magnetic layers (shields that are not meshed through their thickness).
- Parametric templates `toroid`, `solenoid` and `core`. Designs are JSON documents with parameter expressions,
  checked against a JSON Schema; errors name the JSON path and give a hint.
- Results: the signal coupling Ψ, the pickup inductance L_p, the scan figure of merit 𝓕 = Ψ⁴/L_p², the stored
  energy, peak fields and superconductor surface fields (with sharp-corner detection), forces (net and distributed,
  winding self-force, a Newton's-third-law closure check), and for each noise benchmark 𝓕/𝓕_req, the amplitude
  scale needed, a cost metric and the cost-optimal size. Every result carries diagnostics and consistency checks.
- Conductor critical surfaces (NbTi, Nb3Sn, REBCO) with critical-current and temperature margins; the HTS critical
  state (E–J power law) for azimuthal screening currents.
- Electrostatics: capacitance matrices and readout-gap capacitance, axisymmetric and 3D.
- 3D H(curl) magnetoquasistatic engine with an infinity-shell open boundary (MPI builds).
- Full wave beyond magnetoquasistatics, axisymmetric (both m = 0 families, exact outgoing boundary) and 3D (MPI
  builds): readout impedance Z(ω), Ψ(ω), radiation Q, eigenmodes and self-resonances; S/Z/Y parameters and
  Touchstone v1 export with passivity and reciprocity checks; 3D current ports and multiport impedance matrices.
  **Beta:** the surface-resistance models of RF walls behind the perturbative surface losses, and in 3D the
  prescribed static source, the imposed background field, perfectly matched layers, the error indicator, the Gauss-law guard on the ungauged paths,
  the adaptive rational-fit frequency sweep, the opt-in tree-cotree gauge, the eigenmodes, closed RF-only models
  (symmetry surfaces, removed PEC volumes, lumped gaps, direct factorization) and the energy-matrix, field-export and
  reactive-load diagnostics.
- Readout circuits (modified nodal analysis with the noise of every element), multiport networks, finite-band scans
  with the exact network noise, reach curves, and multi-band and multipole scans implementing the methods of
  arXiv:2211.00008 and arXiv:2302.14084 (`axionmultiphysics.multipole`, **beta**).
- Shape sensitivities by a discrete adjoint on a morphed mesh (axisymmetric designs with linear materials);
  gradient-based (L-BFGS-B, SLSQP), global (differential evolution) and derivative-free (bounded Powell)
  optimization.
- Mechanics (axisymmetric linear elasticity with bonded contact, cooldown loads) and steady heat conduction with a
  temperature-dependent conductivity; 3D versions in MPI builds.
- Torsional dynamics of axisymmetric structures and an analytic torsional-waveguide mobility (**beta**).

### Other detector types (beta)

- Cavity haloscopes: axisymmetric, cross-section and 3D eigenmodes, form and geometry factors, Q0 from
  surface-resistance models, tuning sweeps with mode tracking; haloscope sensitivity (scan rate, optimal coupling,
  exclusion) and the parameters of published haloscope runs.
- Coupling of a cavity mode to a coaxial probe on the axis of a pillbox (`axionmultiphysics.cavity_coupling`).
- SRF heterodyne cavities: elliptical-cell geometry, degenerate pump and signal mode pairs, their figures of merit,
  and the sizing of a scan.
- dc SQUID readout chains: added noise and the reach of a detector read out by them.

### Fluids and chemistry (beta)

- `axionmultiphysics.fluids` (MPI builds only): Taylor–Hood Stokes (planar, axisymmetric with swirl, distributed 3D),
  steady Navier–Stokes, stabilized P1/P1 (planar), inductionless MHD (2.5D, one MPI rank), transient Navier–Stokes
  (BDF1/BDF2), and scalar transport with Boussinesq convection (`axionmultiphysics.transport`, one MPI rank). Meshes are
  given as arrays; there is no design-document, viewer or MCP integration.
- `axionmultiphysics.chem`: stiff chemical kinetics and 0D reactors with axionMultiPhysics's own time integrators,
  chemical equilibrium as an index-1 DAE; primary and secondary current distributions (planar and axisymmetric; 3D in
  MPI builds); a 1D porous electrode with linearized kinetics.
- Importing `axionmultiphysics.fluids`, `axionmultiphysics.chem` or `axionmultiphysics.transport` emits a one-time
  `axionmultiphysics.BetaWarning`. User guides: docs/guides/fluids.md and docs/guides/chemistry.md.
- Verification of both modules as case definitions with acceptance criteria and lean reference numbers, run by
  `benchmarks/fluids/run.py` and `benchmarks/chem/run.py` (list, show, run, report); the optional cross-code
  reference generators (NGSolve, FEniCSx) are in `tools/fluids_mms`. The measured results, including what is not
  verified, are on docs/validation/fluids.md and docs/validation/chemistry.md.

### Interfaces

- Python API, the `axmp` command line, a Jupyter widget, single-file HTML export, a live viewer server, VTU export
  for ParaView and a deterministic PNG renderer.
- An MCP server with 36 tools for AI assistants, an agent skill for detector design and a project MCP configuration.
- COMSOL export (mesh, Java model, STEP, physics settings) and import of `.mph` models (**beta**), both without a
  COMSOL installation.

### Validation

- Tests against closed-form solutions, manufactured solutions and independent codes, and comparisons with published
  detectors and experiments, each quantity with a verdict (agrees, disagrees or not decidable): ABRACADABRA-10cm,
  DMRadio-50L, DMRadio-m3, DMRadio-GUT, DMRadio-Core, HAYSTAC, RADES, CAPP, QUAX and the COMSOL models of
  arXiv:2211.00008 (shielded scenarios). The documentation's validation pages describe the checks and their
  tolerances.

### Builds and packaging

- Wheels on PyPI (`pip install "axionmultiphysics[mesh]"`) for manylinux_2_28 x86_64 and macOS 11+ on x86_64 and arm64,
  CPython 3.10–3.14, and the equivalent sequential source build (`scripts/build_deps_wheel.sh`, then
  `AXIONMULTIPHYSICS_DEPS=$PWD/third_party/install-seq pip install .` in a clone): everything except the 3D engine and
  the fluids module. Source builds with MPI add both; OpenMP, STRUMPACK and CUDA (hypre in the 3D engine) are optional.
- Pinned dependencies: MFEM 4.10, hypre 2.33, METIS 5.1, Eigen 3.4, nlohmann/json 3.12 and the gmsh 4.15 SDK. gmsh
  is used only as a separate program and is not redistributed in the wheels.
