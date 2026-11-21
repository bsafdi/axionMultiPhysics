# Command line: `axmp`

`axmp` turns a detector design into a mesh, solved fields and result files.
[Installing axionMultiPhysics](guides/install.md) adds it to your `PATH`. The console
script runs the compiled command line shipped with the package
(`axionmultiphysics/bin/axmp`, or `build/axmp` in a source checkout); set `AXMP_BIN` to
select a different binary. `export-comsol`, `import-comsol` and
`export-touchstone` are handled in Python before the binary is looked for.

| Command | What it does |
|---|---|
| `axmp solve design.json [-o DIR] [--order P] [-v N] [--mesher M] [--scene] [--vtu] [--save] [--no-self-force] [--force-samples N] [--status-file F]` | Solve and write `DIR/results.json`; optionally export a viewer scene (`--scene`), ParaView files (`--vtu`) or saved 3D fields (`--save`). The summary printed on stdout (success, or an error raised after the options are read; not an error in reading the design file) is also written by rank 0 to `DIR/axmp_status.json` (or `F`). |
| `axmp sensitivities design.json [--param P]... [--objective Q]... [--method adjoint\|fd_morph\|fd_remesh] [--step S] [--benchmark NAME] [-o DIR] [--order P] [--mesher M] [-v N]` | Compute axisymmetric [shape sensitivities](optimization.md). |
| `axmp validate design.json [--strict] [--mesher M]` | Check the design. A model document (beta; a document with `physics`) prints its validation report, every finding at once (see below); `--strict` makes an `unavailable` finding fail as an error does. |
| `axmp mesh design.json [-o DIR] [--mesher M] [-v N]` | Write `DIR/mesh.msh` or `DIR/mesh3d.msh`. A model document (beta) is meshed after its full check: see below. |
| `axmp boundaries design.json [--json] [--mesh] [-o DIR] [--mesher M]` | List the named boundaries of a model document (beta) with the checks on them: validates with the mesher probe, runs the mesher's resolve pass and the full check (the partition rule of every physics, empty selections, partial matches, the outputs on resolved boundaries), and prints a table. `--json` prints the validation report with `symmetry`, `boundaries` (per name: `selector`, `entities`, `measure`, `bbox`, `point`, `adjacent`, `used_by`) and `edges` added, and `deferred` empty; `--mesh` meshes into `-o DIR` (else a temporary directory) and adds `faces`. An error prints that report with `ok` false and exit status 1. An electromagnetic design is refused (use `axmp resolve`). |
| `axmp resolve design.json` | Print the geometry with expressions evaluated (a model document too). |
| `axmp explain design.json [--json]` | List the resolved model of a model document (beta): every default explicit and tagged `[default]` or `[written]`, each physics with its boundary conditions and the form the engine gives them, the coupling terms, the study with its coupling strategy and solver defaults and the `time`, `save`, `initial_from` and `given` that a step writes (tagged `[written]`; nothing is listed for a key that was not written), the route each group of physics takes and what it needs. `--json` prints the same as `axionmultiphysics.explain/1`. A document that validation refuses is refused with its first error (exit status 1); an electromagnetic design is refused with the hint to use `axmp resolve`, and a file that is not a JSON object with "expected an object". The flag may come before or after the file; wrong arguments exit with status 2. |
| `axmp patch design.json ops.json [-o OUT]` | Apply an [RFC 6902](https://www.rfc-editor.org/rfc/rfc6902) JSON Patch (a list of `add`, `remove`, `replace`, `move`, `copy`, `test` operations with [RFC 6901](https://www.rfc-editor.org/rfc/rfc6901) paths) to a copy of the document and validate the result: a model document with the model validation, an electromagnetic design with its parser. All or nothing: either every operation applies and the result validates, or nothing is written and the error is printed (exit status 1). An operation that cannot be applied has the code `patch.operation_failed` and `where` set to `/<index>` of the operation. Without `-o` the patched document is printed; with `-o` it is written to `OUT` and `{"ok": true, "name": ..., "written": OUT}` is printed. The input file is never changed. The patched document has its keys in sorted order. |
| `axmp schema` | Print the [design format](design_format.md) JSON Schema: one schema for electromagnetic designs and, when the key `physics` is present, for model documents (beta; the model sections are marked `"x-status": "beta"`). |
| `axmp physics [--json] [--type T] [--routes] [--all]` | Print the physics catalogue of model documents (beta): the physics types, coupling terms and, with `--routes`, the route table ([reference page](reference/physics_catalogue.md)). `--type T` restricts it to one type, `--all` prints the details, `--json` the machine form (`axionmultiphysics.physics_catalogue/1`, details included). Needs no design file. An unknown type is an error (exit status 1) whose hint lists the types. |
| `axmp requirement [--benchmark NAME]` | Print the scan requirement 𝓕_req for the built-in noise benchmarks (`optimistic`, `conservative`, `conservative_solenoid`; all three by default). |
| `axmp render scene.json -o out.png [--view section\|3d] [--quantity Q] [--log] [--theme dark]` | Write a [PNG snapshot](render.md). |
| `axmp templates` | List parametric templates. |
| `axmp template NAME [--set key=value ...]` | Print a design from a template. |
| `axmp export-comsol SOURCE [-o OUT] [--format mphtxt\|java\|step] ...` | [Export a design to COMSOL](comsol_export.md). |
| `axmp import-comsol model.mph [-o design.json] ...` | [Import a COMSOL model](comsol_import.md). |
| `axmp export-touchstone results.json [-o OUT.sNp] [--kind S\|Z\|Y] [--z0 50] [--dense] [--surface-losses]` | Write full-wave port parameters as [Touchstone v1](physics/touchstone.md) (time convention e^{+jωt}). Exit status 0: written; 3: written but not passive; 2: error (message on stderr). |

```bash
axmp template toroid --set R=2.0 --set H=2.0 > toroid.json
axmp validate toroid.json
axmp solve toroid.json -o runs/toroid --scene
axmp render runs/toroid/scene.json -o toroid.png --view 3d --quantity Bmag
```

This writes the detector's fields and coupling results to
`runs/toroid/results.json`, a viewer scene to `runs/toroid/scene.json`, and a
magnetic-field snapshot to `toroid.png`.

`axmp solve` picks the solver from the design: axisymmetric designs (the
default), z-uniform cavity cross-sections (`"symmetry": "xsec"`), 2D full-wave
models on an imported mesh (`mesh.file`), and 3D designs (`"symmetry": "3d"`).
Run 3D designs with an MPI-enabled build under an MPI launcher
(`mpirun -n N axmp solve ...` or your scheduler's equivalent). Axisymmetric,
cross-section and 2D imported-mesh designs run on one process; under a
multi-rank launch only rank 0 works (an axisymmetric result then carries a
warning that the other ranks stayed idle).

`axmp physics` is beta and describes the model document that this version is building towards: no route of the
catalogue is implemented yet, and every row is printed as "not implemented yet" and has `"implemented": false` in the
JSON. The catalogue is the same in every build. `axmp explain` and `axmp patch` are beta too; neither solves anything.

A model document (beta: a document with a top-level `physics` section) is taken by `axmp validate`, `axmp resolve`,
`axmp explain`, `axmp patch`, `axmp boundaries` and `axmp mesh`. `axmp validate` prints the report

```json
{"ok": true, "name": "channel", "parameters": {"L": 0.04},
 "errors": [], "unavailable": [{"error": "...", "where": "/physics", "hint": "...", "code": "route.not_implemented", "details": {}}],
 "deferred": [], "warnings": []}
```

with every finding of the document, each one the error JSON of the command line without `ok`. With an error, `ok` is
false, the first error's `error`, `where`, `hint` (and `code` and `details` when it has them) are also at the top
level, and the exit status is 1; with `--strict` and no error, the first `unavailable` finding (the document is valid
and this version cannot solve it) is treated the same way. `axmp resolve` prints the resolved geometry with the keys
it has for an electromagnetic design (`mesh` carries the mesh keys of a model document); a grammar error is printed
as the first error. `axmp solve` refuses a model document with the code `route.not_implemented`
(solving one is not implemented in this version); `axmp mesh` meshes a planar or axisymmetric one (below) and refuses a 3D
one or one on an imported mesh with the same code; `axmp sensitivities` and `axmp export-comsol` (a
design `.json` or the `design.json` of a run directory) refuse it with `model.em_only_operation`. Both codes carry
`details.operation` (the command; `export_comsol` for `axmp export-comsol`) and `details.serves_model_documents`
(`["validate", "resolve", "explain", "patch", "boundaries", "mesh"]`). An electromagnetic design with a top-level `domains`, `boundaries`, `reactions`,
`couplings`, `study` or `outputs` is refused with `model.section_needs_physics`. `axmp mesh model.json [-o DIR] [--mesher M]` of a model document validates it with the mesher probe, runs the resolve pass
and the full check (so an uncovered boundary is refused before any mesh is made), then meshes, writing `mesh_in.json`,
`mesh.msh`, `mesh_info.json` and `mesh_mesher.log` to `DIR` (default `axmp_mesh`) and printing `ok`, `name`, `symmetry`,
`geometry_order`, the three paths, the `boundaries` with their `measure` (and `area_m2` when axisymmetric) and the
`warnings`; a document with no geometry prints a `note`. A document with a geometry in an installation whose mesher does not
run is `build.requires_mesher` (an `unavailable` finding of `axmp validate`, first in its list), a failed mesher run is
`mesh.generation_failed`. See [meshes of model documents](guides/model_meshes.md). A model document whose file repeats
a key in an object is refused with `json.duplicate_key` at the JSON pointer of the second occurrence (JSON would keep
the last value silently); an electromagnetic design file is read as before. The codes are listed in [error
codes](reference/error_codes.md).

Run `axmp` without arguments for usage, or use `axmp export-comsol --help`,
`axmp import-comsol --help` and `axmp export-touchstone --help` for their options.
Solver and validation errors are JSON with an `error` message and, when
available, `where` and `hint` fields; an error that has a machine-readable
`code` or structured `details` (a JSON object) prints each of them as a key of
its own, so an error may print one, both or neither (details with no members are
left out, and so are details nested deeper than 64 levels). The errors of an
electromagnetic design have no code (except `model.section_needs_physics`, a
section of a model document in a document without `physics`, the
`geometry.arc_*` refusals of an `arc` path, and `mesh.tangled`, a 2D curved
mesh with a tangled element); most of those of
a model document have one ([error codes](reference/error_codes.md) names the
findings without one). Under MPI
launchers that split or interleave long stdout lines, read the summary from the
status file rather than from stdout; `Design.solve(mpi_ranks=N)` does that, with
a status file of its own per launch
(`--status-file`).
