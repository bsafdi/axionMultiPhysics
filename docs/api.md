# API reference

Lumped-element detectors, cavity haloscopes and SRF heterodyne cavities search for axion dark matter. Here we list the public objects of the `axionmultiphysics` Python package, grouped by topic.

## Errors

Every error the package raises is an `AxionMultiPhysicsError`. Besides its text it has four attributes: `where` (a
JSON pointer into the document, or an empty string), `hint` (what to change), `code` (a stable dotted identifier,
or an empty string) and `details` (a dict of structured facts, or `None`; `None` also when the details are nested
deeper than 64 levels, the details object being level 1, as on the command line and in the MCP reply). The command
line prints `code` and `details` in its error JSON only when they are not empty (see [the command line](cli.md)),
and the MCP server's failure reply carries the same four fields.

## The physics catalogue (beta)

`axionmultiphysics.physics(routes=False, type=None)` returns the physics catalogue of model documents as a dict, the
JSON that `axmp physics --json` prints (schema `axionmultiphysics.physics_catalogue/1`): the physics types with their
options, condition kinds, sources and quantities, the coupling terms, the reserved instance ids and, with
`routes=True`, the route table and the layer rules. Everything in it is beta, and no route is implemented in this
version. The [reference page](reference/physics_catalogue.md) is generated from it.

The codes that the findings about a model document carry are listed in the
[error codes of the model document](reference/error_codes.md), a page generated from the table of codes in the core
(`axionmultiphysics._core.error_codes()` returns it as JSON text). `Design.validate(strict=False, all_errors=False)` of a
model document (one with a top-level `physics` key) returns its validation report (`ok`, `name`, `parameters`,
`errors`, `unavailable`, `deferred`, `warnings`), raising the first error (with `strict`, also the first unavailable
finding) unless `all_errors`; `axmp.load` refuses a model document whose file repeats a key (`json.duplicate_key`).
`Design.boundaries(mesh=False, all_errors=False)` returns the table of the named boundaries of a model document as `axmp
boundaries --json` prints it (the validation report with `symmetry`, `boundaries` and `edges` added; `mesh=True` adds
`faces`), raising the first error unless `all_errors`; `axmp.mesh(d, workdir=None, mesher=None)` meshes a model document
(a `Design` or a dict) and returns a `MeshedModel` with `symmetry`, `geometry_order`, `regions`, `boundaries`,
`mesh_file`, `info_file` and `instance(domain=None)`, which returns the `InstanceMesh` of a region or domain name, a list
of names or every region: its `attributes`, `remaining_attributes`, `axis_attributes`, `master_attribute`,
`region_attribute`, the arrays `vertices()`, `elements()`, `element_attributes()`, `boundary()` and
`boundary_attributes()`, `section(name)` and `face_points(name=None, *, order, attribute=None)`, the quadrature on the
faces of a boundary, a section or one instance attribute (the way to the edges no boundary names and to the axis), with the
points, weights and normals as NumPy arrays. Both need the mesher: without one that runs they raise
`build.requires_mesher`; a 3D document, a document on an imported mesh and (for `axmp.mesh`) a document with no geometry raise `route.not_implemented`; a failed mesher run is `mesh.generation_failed`; misuse of an instance mesh (an unknown name, an
attribute that is not on it) is a `ValueError`; an electromagnetic design is refused without a code. An `InstanceMesh` also has `fluids_mesh()` (a `fluids.Mesh` copy of it; MPI builds, else `build.requires_fluids`) and `chem_mesh(order=2)` (a `chem.TriMesh`; `MeshedModel.fluids_mesh(domain=None)` and `MeshedModel.chem_mesh(domain=None, order=2)` are shorthands). See
[meshes of model documents](guides/model_meshes.md).
`Design.solve` refuses a model document with `route.not_implemented`, and the electromagnetic-only operations
(`Design.gradient`, `Design.optimize`, `axmp.optimize`, `axmp.global_optimize`, `sensitivity.optimize`, `convert.to_3d`,
`comsol.export_comsol`, the model classes of `_core`) refuse it with `model.em_only_operation`; both name the
operation in `details.operation`.
`Design.explain()` and `Design.patch()` read a model document too.

`Design.explain(json=False)` returns the resolved model of a model document as the text that `axmp explain` prints, or
with `json=True` as the dict of schema `axionmultiphysics.explain/1`. It raises `AxionMultiPhysicsError` for a
document that fails validation (the first error) and for an electromagnetic design.

`Design.patch(ops)` applies an RFC 6902 JSON Patch (a list of operations; paths are RFC 6901 JSON pointers) atomically
and returns the design: either every operation applies and the result validates, or `AxionMultiPhysicsError` is raised
and `design.doc` is unchanged. The keys of the patched document are in sorted order (the order of the keys of a JSON
object carries no meaning). An operation that cannot be applied raises with the code `patch.operation_failed` and
`where` `/<index>`; a result that does not validate raises the validation's own error. It works on electromagnetic
designs too.

::: axionmultiphysics.design.physics
    options:
      heading_level: 3
      show_root_full_path: false

## Designs and solves

axionMultiPhysics provides functions and classes for creating designs from templates or files, solving them and accessing the results.

::: axionmultiphysics.design.template
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.design.templates
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.design.load
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.design.Design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.results.Results
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.results.Results3D
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.results.requirement
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.global_opt.optimize
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.global_opt.global_optimize
    options:
      heading_level: 3
      show_root_full_path: false

## Lumped-element scan and reach

`scan.multipole_scan` (with `scan.reach_multiband` and `scan.reach_2211`, module `axionmultiphysics.multipole`) is beta.

::: axionmultiphysics.scan.reach
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.scan.g_qcd
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.scan.reach_fullwave
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.scan.finite_band
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.scan.multipole_scan
    options:
      heading_level: 3
      show_root_full_path: false

## Readout circuit and SQUID chains

The dc SQUID chains (`axionmultiphysics.squid`) are beta.

::: axionmultiphysics.circuit.series_readout
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.circuit.Readout
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.squid.chain
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.squid.Chain
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.squid.benchmark
    options:
      heading_level: 3
      show_root_full_path: false

## Cavity haloscopes

Everything in this section is beta, as are the related modules `axionmultiphysics.cavity_xsec`,
`axionmultiphysics.cavity_coupling` and `axionmultiphysics.haloscope_presets`.

::: axionmultiphysics.cavity.pillbox_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity.coax_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity.pillbox_modes
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity.surface_resistance
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity_track.sweep
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity_track.track
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity_track.Tracking
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.Cavity
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.Receiver
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.scan_rate
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.beta_opt
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.g_limit_scan
    options:
      heading_level: 3
      show_root_full_path: false

## SRF heterodyne

Beta: `axionmultiphysics.srf` and `axionmultiphysics.heterodyne`.

::: axionmultiphysics.srf.elliptical_cell_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.srf.tune_degeneracy
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.srf.find_mode
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.srf.figures_of_merit
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.heterodyne.SRFGeometry
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.heterodyne.scan_rate
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.heterodyne.required_volume
    options:
      heading_level: 3
      show_root_full_path: false

## Touchstone (S-, Y- and Z-parameters)

::: axionmultiphysics.touchstone.write_touchstone
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.touchstone.read_touchstone
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.touchstone.port_parameters
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.touchstone.z_to_s
    options:
      heading_level: 3
      show_root_full_path: false

## COMSOL

The import (`axionmultiphysics.comsol_import`) is beta; the export is not.

::: axionmultiphysics.comsol.export_comsol
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.comsol_import.import_comsol
    options:
      heading_level: 3
      show_root_full_path: false

## Field maps (beta)

Per-region maps of the static field for particle transport, written by `Results.export_field_map` (see the
[field-map guide](guides/field_maps.md)).

::: axionmultiphysics.fieldmap.load
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.fieldmap.write_map
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.fieldmap.FieldMap
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.fieldmap.Patch
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.fieldmap.recovered_field
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.fieldmap.chord_error
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.fieldmap.decide
    options:
      heading_level: 3
      show_root_full_path: false

## Fluids and chemistry

The fluids (`axionmultiphysics.fluids`, `axionmultiphysics.transport`) and chemistry (`axionmultiphysics.chem`) modules
are beta. The [fluids](guides/fluids.md) and [chemistry](guides/chemistry.md) guides list their entry points, and
`help(...)` on each entry point documents its arguments.

## Solids for particle transport (beta)

The solids list of an axisymmetric partition and its GDML export (see the [solids guide](guides/solids.md)).

::: axionmultiphysics.Design.export_gdml
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.solids.from_partition
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.solids.write
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.solids.write_gdml
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.solids.default_chord_tol
    options:
      heading_level: 3
      show_root_full_path: false

## Particle transport (beta)

The driver of `axmp-transport`: plans, runs and merges transport runs and estimates their outputs (see the
[particle transport guide](guides/particles.md)).

::: axionmultiphysics.particles.plan
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.execute
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.merge
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.run
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.extend
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.load
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.transport
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.resolve_materials
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.capabilities
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.ParticleResults
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.Measured
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.estimate_mean
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.estimate_fraction
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.estimate_count
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.batch_means
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.derive_seed
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.chunk_seed
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.particles.event_words
    options:
      heading_level: 3
      show_root_full_path: false
