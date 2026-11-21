# Particle transport (beta)

Charged particles through a design's solids and its solved magnetic field: muons through a tracker in a solenoid's
bore, electrons showering in iron, geantinos counting an acceptance. Geant4 runs in **`axmp-transport`**, a separate
program; the package drives it from Python with **`axionmultiphysics.particles`**, which plans a run into chunks of
events, starts the program on them, merges the chunks in a fixed order and estimates the requested outputs with their
statistical errors. The package's library, its Python module and the `axmp` command never link Geant4: they exchange
files with `axmp-transport`, as they do with `axmp-mesher` and gmsh.

!!! warning "Beta"
    The job format, the output arrays, the driver's interface and the command line may change between minor
    versions. Every number is Monte Carlo with **statistical errors only**: the dependence on the physics list and the
    approximations of Geant4's models are not estimated. Statistical error is not the total error. Nothing here is
    validated against measured data; [the validation page](../validation/particles.md) lists what is checked against
    closed forms and published formulas.

This product includes software developed by Members of the Geant4 Collaboration ( http://cern.ch/geant4 ).

## A first run

```python
import axionmultiphysics as axmp
from axionmultiphysics import particles, solids

res = axmp.Design(doc).solve(workdir="runs/solenoid")
fm = res.export_field_map("runs/solenoid/fieldmap", tol=1e-4)          # docs/guides/field_maps.md
sol = solids.from_partition(partition, materials, region_materials)     # docs/guides/solids.md
r = particles.transport(axmp.Design(doc), res, solids=sol, field_map="runs/solenoid/fieldmap",
                        sources=[{"type": "gun", "particle": "mu+", "momentum_eV_c": 2e8,
                                  "at_m": [0, 0, 0], "direction": [1, 0, 0]}],
                        outputs={"trk": {"hits": {"in": ["layer_0", "layer_1"], "track": "primary"}},
                                 "E": {"deposit": {"in": "layer_0"}}, "Em": {"mean": {"of": "E"}}},
                        events=10000, workdir="runs/solenoid/muons", seed=1)
print(r["Em"])            # e.g. "1.234e3 +- 0.012e3 (10000 events)": a Measured, value and statistical error
hits = r.hits("trk")      # chunk, event, track, parent, pdg, region, x_m, p_eV_c, t_s, ekin_eV
```

`particles.transport(design, results, ...)` is the route from a solved design: its regions are the solids (passed as
`solids=`: this version does not run the partition export for the call, which without `solids=` raises
`particle.partition_unavailable`; the solids document that `Design.export_gdml` returns can be passed, a route that
the validation does not cover yet), its field the map (exported from `results` into `<workdir>/fieldmap` when
`field_map` is not given; every patch serves its record's regions). Everything else goes to `particles.run`, which is
`particles.plan` + `particles.execute` + `particles.merge` on a work directory and works with any solids list and
any fields:

```python
r = particles.run("runs/w", solids=sol, fields=[{"type": "uniform", "B_T": [0, 0, 1.5]}],
                  sources=[...], outputs={...}, events=100000, chunk_events=1000, jobs=8, seed=7)
r = particles.extend("runs/w", events=200000)    # bitwise equal to a direct run of 200 000 events
r = particles.load("runs/w")                     # the merged result from disk, nothing run again
```

`particles.capabilities()` says what this installation has (never raises): the program (`found`, `path`, its
`--check` document, `ok`), whether the mesher exports partitions for the API route (`partition_export`, false in this
version for both geometries) and the event generator (none in this version).

## The driver: `axionmultiphysics.particles`

```python
particles.plan(workdir, *, solids, sources, outputs, events, fields=(), seed=1, chunk_events=1000, jobs=1,
               physics_list="FTFP_BERT", production_cut_m=7e-4, integrator=None, outside="refuse",
               quantum_eV=1e-3, stop=None, threads=1, program=None) -> dict          # run.json's content
particles.execute(workdir, *, workers=1)
particles.merge(workdir, *, allow_failed_chunks=False) -> ParticleResults
particles.run(workdir, **plan_kwargs, workers=None, allow_failed_chunks=False) -> ParticleResults
particles.extend(workdir, *, events, workers=1, chunk_events=None, allow_failed_chunks=False) -> ParticleResults
particles.load(workdir) -> ParticleResults
particles.transport(design, results, *, sources, outputs, events, composition=None, solids=None, field_map=None,
                    map_tol=1e-4, track_volumes=None, workdir=None, **run_kwargs) -> ParticleResults
particles.resolve_materials(doc, *, program=None) -> dict
```

The module is pure Python (numpy, scipy); importing it imports nothing of Geant4. It finds the program as `program=`,
else `$AXMP_TRANSPORT`, else `axmp-transport` next to `axmp`, else on `PATH`; without one, `plan` raises
`build.requires_program` (build it with `-DAXMP_BUILD_TRANSPORT=ON` and set `AXMP_TRANSPORT`).

**`plan`** validates everything before it writes anything, and refuses with a named code (below): the event count
(`events` is required: a run never defaults to a number of events), the seed (an integer in [0, 2^64),
`events.invalid` otherwise: two seeds equal modulo 2^64 would be one run recorded as two), the source, the physics
list, the outputs, the fields and their coverage of the solids, the integrator, a tally that could overflow, and the
geometry (Geant4's overlap check, `particle.geometry_overlap`). A numeric argument of the wrong type or range
(`jobs`, `threads`, `workers`, `production_cut_m`, `quantum_eV`, a seed that is not an integer) and a `debug` that
is not an object raise a plain `ValueError` instead. It resolves the NIST compositions of the solids against Geant4's tables
(`resolve_materials`, below), then writes the work directory:

```
<workdir>/run.json                the plan, updated by execute, merge and extend
<workdir>/exchange/solids.json    the resolved solids list both Geant4 routes read
<workdir>/exchange/job_<j>.json   one job file per block of chunks; exchange/commands.txt: their command lines
<workdir>/out/chunk_<k>/          the program's chunks; out/record_<j>.json
<workdir>/results.json, merged/   written by merge
<workdir>/logs/                   the program's output per process, with its exit status
<workdir>/.lock                   the workdir's lock
```

**One driver per workdir.** `plan`, `execute`, `merge` and `extend` (and `run` and `transport`) hold an exclusive
lock on the workdir (`fcntl.flock` on `<workdir>/.lock`) while they work; another process that tries meanwhile is
refused with `events.workdir_busy` and changes nothing (`load` only reads and takes no lock).

**Field maps of a run.** The job files name a map that lies inside the workdir (the one `transport()` exports into
`<workdir>/fieldmap`) relative to it (`../fieldmap`), so a copied or moved workdir reads its own map; every other
map by its real absolute path, so a workdir reached through a symbolic link finds it either way. `run.json` records
each map's path relative to the workdir and the SHA-256 of its manifest (`fields[i].manifest_sha256`; the manifest
lists every patch array's checksum). `execute`, `extend` and `run` compare it, where the job files point, before any
program starts, and refuse a map that changed or vanished with `events.field_map_changed`, naming the map and
changing nothing: an extension would otherwise mix two fields.

`workdir` must be new, empty, or hold a planned run (which is planned again: its chunks are removed;
`events.workdir_not_empty` otherwise). The run is cut into `ceil(events / chunk_events)` chunks (the last one short
when `events` is not a multiple), and the chunks into `jobs` contiguous blocks of near-equal length, one job file each
(fewer when there are fewer chunks). `threads` is the `--threads` of every process.

**`execute`** starts the job files, `workers` processes at a time (`run` defaults `workers` to the number of job
files). A process that fails (non-zero exit status) has its incomplete chunks run again once, in one new process; a
chunk still incomplete is `failed`. A chunk counts as complete once its manifest exists (the program writes it last
and then renames the chunk's directory); the merge checks every array against its checksum.

**`merge`** reads every planned chunk: it is complete when its manifest's `chunk`, `chunk_seed` and `events` are the
plan's (otherwise the merge is refused, `events.chunk_mismatch`) and every array matches its SHA-256 (otherwise the
chunk counts as missing). The merged sample is the **longest complete prefix** of chunks 0, 1, 2, ... (with
`allow_failed_chunks=True`, every complete chunk). A prefix of a sequence of independent chunks is an unbiased sample
whatever made the others fail, and it is what an extension continues. Fewer events than requested make a
**partial** result: `partial` true, `missing_chunks` (the planned chunks not merged), warning `events.partial`.
Events are merged in (chunk, event) order, so the result does not depend on `jobs`, `workers` or `threads`.

**`extend(workdir, events=N2)`** (N2 at least the events requested so far, `events.invalid` otherwise; the same
`chunk_events`, `events.chunk_events_mismatch` otherwise) reuses every complete full chunk (its checksums checked),
runs a short last chunk again in full and replaces it, runs the new chunks, and merges: the result equals, bit for
bit, a direct run of N2 events (arrays, tallies and `results.json`, timings aside).

**Stop rules.** `stop={"output": NAME, "rel_error": r}` (NAME an aggregate output) merges the smallest prefix of
K >= 20 chunks for which the output on chunks 0..K-1 has `stat_error / |value| <= r`, or the planned chunks. It is
evaluated on prefixes of complete chunks in index order, so the result does not depend on the number of workers;
chunks beyond K that completed are kept on disk as `beyond_stop` and not merged. `execute` evaluates the rule each
time a job file's process exits, and then stops the others and starts no more: the rule saves compute only at the
boundaries of job files. With the default `jobs=1` every chunk runs; plan `jobs` well above `workers` (for example a
few chunks per job file) when the rule is meant to end the run early. `run.json`'s `stop` records
`reason` (`rel_error` or `events`), `chunks` and `events`.

**Command line.**

```bash
python -m axionmultiphysics.particles {execute,merge,extend} WORKDIR [--workers N] [--allow-failed-chunks] [--events N]
```

prints one JSON document, `{"status": "complete" | "partial", "events", "events_requested", "missing_chunks"}`, and
exits 0 when complete, with **exit status 5** when partial, 1 when refused (`{"error", "code"}` on stdout), 2 for a
usage error (a malformed command, `--workers` or `--events` that is not a positive integer, or `--events` given to
`execute` or `merge`; nothing is changed). `--allow-failed-chunks` applies to the merge that `execute`, `merge` and
`extend` end with (the Python `execute` only runs and takes no such option). A scheduler can run the commands of `exchange/commands.txt` itself (one per job file) and then
`merge`.

### Sources

Exactly one source in this version, a gun:

```json
{"type": "gun", "particle": "mu+", "momentum_eV_c": 1e9, "at_m": [0, 0, -0.5], "direction": [0, 0, 1],
 "spot_sigma_m": 0.001}
```

- `particle`: a name of Geant4's particle table (`axmp-transport --check` lists it) or a PDG code; a name it lacks is
  refused with `particle.unknown_particle`, whose message lists candidates (`muon` -> `mu-`, `mu+`; `electron` ->
  `e-`; `pion` -> `pi+`, `pi-`, `pi0`; close spellings);
- exactly one of `energy_eV` (kinetic) and `momentum_eV_c`, positive; a momentum is converted to kinetic energy with
  the mass of Geant4's particle table;
- `direction`: a non-zero vector (normalised) or `"isotropic"` (uniform on the sphere; a point source);
- `spot_sigma_m` (default 0): a Gaussian spot, sigma per transverse coordinate, in the plane through `at_m`
  perpendicular to `direction`;
- `at_m` must lie in a region of the solids (their surfaces included): `plan` locates it in the solids list and
  refuses a point beyond the world or in the world or a void, where the primary would have no volume to start in or
  would leave the domain at once (`particle.source_invalid`). The program refuses a job file whose `at_m` lies
  outside the world (exit status 2, nothing written), and ends the process (status 4, with a message) if a spot
  draws a vertex beyond the world.

Every event draws first `u0 = G4UniformRand()` (recorded as `first_flat`), then, for an isotropic direction,
`u1, u2`: cos(theta) = 1 - 2 u1, phi = 2 pi u2; for a spot, `u1, u2`: rho = sigma sqrt(-2 ln(1 - u1)), psi = 2 pi u2,
the offset rho (cos(psi) e1 + sin(psi) e2) with e1 the unit vector of d x z (x when d is along z) and e2 = d x e1.
Any other refusal of the source is `particle.source_invalid`.

### Outputs

`outputs` is `{name: spec}` (names `[A-Za-z_][A-Za-z0-9_]*`), each spec with exactly one kind:

| kind | spec | value |
|---|---|---|
| per-event variable | `{"deposit": {"in": REGION or [REGIONS]}}` | the energy deposited per event, eV |
| per-event variable | `{"hit": {"in": REGION, "of": "x" \| "y" \| "z" \| "r" \| "phi" \| "t" \| "E_kin", "track": "primary" \| "any"}}` | at the event's first hit in the region (default track: the primary); undefined (NaN) when there is none; m, rad, s, eV |
| per-event variable | `{"primary": {"of": "x" \| "y" \| "z" \| "px" \| "py" \| "pz" \| "E_kin"}}` | the primary as generated; m, eV/c, eV |
| aggregate | `{"mean": {"of": VAR, "where": W}}`, `{"std": ...}`, `{"quantile": {"of": VAR, "q": q, "where": W}}` | `Measured` |
| aggregate | `{"fraction": {"where": W}}`, `{"count": {"where": W}}` | `Measured` (over all events) |
| histogram | `{"histogram": {"of": VAR, "bins": n, "range": [lo, hi], "where": W}}` | `r.histogram(name)` |
| tally | `{"tally": {"grid": G, "in": [REGIONS]}}` | `r.tally(name)` |
| hits | `{"hits": {"in": [REGIONS], "track": "any" \| "primary", "particles": [NAMES], "max_per_event": n, "keep_events": m}}` | `r.hits(name)` |

- `W` (optional) is `{VAR: {"gt" | "ge" | "lt" | "le": x, "defined": true | false}, ...}` (all must hold) or
  `{"any": [W, ...]}`. An undefined value satisfies only `{"defined": false}`.
- A grid `G` is `{"type": "cylindrical", "r_m": [r0, r1], "z_m": [z0, z1], "bins": [nr, nz]}` (full rings) or
  `{"type": "cartesian", "x_m": [..], "y_m": [..], "z_m": [..], "bins": [nx, ny, nz]}`. A step's deposit goes to
  the bin of the step's midpoint; a deposit outside the grid or outside `in` (default: every region) is not tallied.
- A hit is one record per step that **enters** a volume of the listed regions (its pre-step point lies on the
  volume's boundary, and the track's previous step is longer than Geant4's surface tolerance, 1e-12 m), taken at
  that point: track and parent IDs, PDG code, region (index in the solids' `regions`),
  position, momentum, global time since the primary vertex, kinetic energy. At most `max_per_event` records per
  event (default 100; the rest are counted as dropped); with `keep_events`, only the first m events in (chunk, event)
  order keep records. At some interfaces Geant4 takes a zero-length step back and forth across a face; the step
  after it records nothing, so no point is recorded twice. An excursion into a neighbouring volume shallower than
  deltaChord (0.25 mm) is collapsed by Geant4 to such a step: its entry is recorded and its return is not (measured:
  11 such pairs among 174 375 hits of 5000 protons crossing slabs of 1 T and 0.2 T at grazing angles, none at
  normal incidence). `results.json` gives `records`, `dropped`, `events_with_records` and
  `events_without_records`.
- Anything else is refused with `particle.output_invalid`, naming the output.

### Results and estimators

```python
r[name] -> Measured        # an aggregate; KeyError for other names
r.histogram(name)          # edges (numpy.linspace(lo, hi, bins + 1)), counts per [lo_i, hi_i), errors, underflow,
                           # overflow, undefined
r.tally(name)              # edges, edep_eV (mean deposit per event per bin), stat_error, q_dep_J_m3, volume_m3,
                           # events, quantum_eV, lo, hi (the exact 128-bit batch sums), batch_events
r.events(names=None)       # chunk, event and the per-event variables, numpy arrays in (chunk, event) order
r.hits(name)               # chunk, event, track, parent, pdg, region, x_m, p_eV_c, t_s, ekin_eV
r.record, r.outputs, r.n_events, r.events_requested, r.partial, r.missing_chunks, r.warnings, r.workdir
```

A **`Measured`** is a float carrying `stat_error`, `error` (the estimator), `events`, `events_defined`, `units`,
`mc` (true), and `upper_limit`, `cl`, `interval` where they apply. Arithmetic on it returns a plain float, so an error
is never carried along silently. It prints its value to the decimal place its error supports (the PDG rule: the
three leading digits of the error decide two significant digits for 100-354, one for 355-949, and 950-999 round up
to 1000 with two), as `"0.9981 +- 0.0021 (2000 events)"` or `"(1.235 +- 0.031)e-7 (2000 events)"`.

| output | value | statistical error (`error`) |
|---|---|---|
| `mean` | the mean of the selected defined values | s / sqrt(n), s with ddof 1 (`sem`); none below n = 2 |
| `fraction` | k / N over all merged events | the **Clopper-Pearson** interval at 68.27 % (`interval`) and half its width (`clopper_pearson`) |
| `count` | k | sqrt(k) (`poisson`) |
| zero counts | 0 | none: the exact one-sided 95 % limit, 1 - 0.05^(1/N) for a fraction, -ln 0.05 = 3.0 for a count (`upper_limit`, `cl`); printed `"< 0.26 (95 % CL, 10 events)"`, never `0 +- 0` |
| no merged event | NaN | none, and no limit (`reason`: no events), for every aggregate |
| `std`, `quantile`, tallies | on all merged events | **batch means** (`batch_means`) |

Batch means: chunk k belongs to batch k mod 20; v_b is the statistic on batch b's events (a tally: its bin sum times
the quantum over the batch's events), w_b = n_b / N, and sigma^2 = B / (B - 1) sum w_b^2 (v_b - vbar)^2 with
vbar = sum w_b v_b and B = 20. Below 20 merged chunks there is no such error (`stat_error` None, warning
`stat.too_few_chunks`). The functions `estimate_mean(values)`, `estimate_fraction(k, n)`, `estimate_count(k)` and
`batch_means(values, weights)` are the same estimators for your own arrays.

**Tallies are exact.** Every event's deposit in a bin is rounded to `quantum_eV` (default 1e-3 eV) and added as an
integer, per chunk in 64 bits and per batch in 128 bits (two limbs, `merged/tally_<name>_lo.npy` unsigned and
`_hi.npy` signed, exact value hi x 2^64 + lo): the sums do not depend on the order of the threads or the chunks, and
none can wrap. A plan whose chunk could reach 2^63 quanta (`chunk_events` x (kinetic plus rest energy) /
`quantum_eV`) is refused with `particle.tally_overflow`.

`results.json` (schema `axionmultiphysics.particles-results/1`) carries `beta`, `mc` (the sentence "Monte Carlo
(beta): N events, seed S; statistical errors only; physics-list dependence not estimated"), `seed`, `events`,
`partial`, `missing_chunks` and per output its record: aggregates `{"value", "stat_error", "error", "events",
"events_defined", "units", "mc": true, "beta": true}` (plus `interval`, `upper_limit`, `cl`), histograms their
arrays, tallies the file names of their batch arrays, hits their counts.

`run.json` (schema `axionmultiphysics.particles-run/1`) records `beta`, `seed`, `stage` and `instance`,
`chunk_events`, `events_requested`, every chunk (`index`, `seed`, `events`, `job`, `status`: `planned`, `complete`,
`failed` or `beyond_stop`, `attempts`, `sha256` of its manifest), the program (path, versions, data sets, command
lines), the physics list, `production_cut_m`, `integrator` (with `stepper`), `quantum_eV`, `outside`, `fields` (as
given, map paths relative to the work directory, plus each map's `tol` and its patches' error records), `sources`,
`outputs`, `stop`, `materials_resolved`, `warnings`, `partial`, `missing_chunks` and `events`. Outside the program's
record and `timing` it holds no time, host or absolute path.

### Seeds

One master `seed` (default 1, always recorded) gives every chunk its seed:
`chunk_seed(seed, k) = derive_seed(seed, "transport", "transport", k)`, where `derive_seed(seed, *keys)` starts from
z = seed mod 2^64 and for each key sets z = mix(((z XOR h(key)) + 0x9E3779B97F4A7C15) mod 2^64), mix being
SplitMix64's output function, h(int) the int mod 2^64 and h(str) the first 8 bytes (little-endian) of the SHA-256 of
its UTF-8. It is injective in the last key, so the chunks of one run have distinct seeds. Every event of a chunk is
seeded from (chunk seed, event index) as the program describes below (`particles.event_words` gives its four words).

### Fields

`fields` is a list; each entry serves the regions it names (default: every region for an analytic field, the patch
record's regions for a map patch):

| entry | field |
|---|---|
| `{"type": "uniform", "B_T": [bx, by, bz], "regions": [...]}` | Geant4's `G4UniformMagField` |
| `{"type": "current_loop", "radius_m": a, "current_A": I, "z_m": z0, "regions": [...]}` | a circular filament on the z axis in the plane z = z0, counter-clockwise seen from +z for I > 0 (closed form with complete elliptic integrals) |
| `{"type": "map", "path": dir, "patch": name, "regions": [...]}` | one patch of an `axionmultiphysics.fieldmap/1` map ([Field maps](field_maps.md)) |
| `{"type": "map", "path": dir}` | every patch of a map, each serving its record's `regions`; the map's `zero_field_regions` are field-free |

A region served twice is refused (`particle.field_ambiguous`), an unknown region or a malformed entry with
`particle.field_invalid` (also a `write_map` patch named without `regions`).

**One field per volume.** Every logical volume gets a field object and a field manager of its own, a field-free one
where no field serves its region, so that no volume inherits its mother's field (Geant4 hands a mother's field
manager to daughters that have none). Every field manager uses the same integrator: Geant4's templated Dormand-Prince
4(5) stepper with `G4IntegrationDriver` (`G4ChordFinder`'s `kTemplatedStepperType`; recorded as
`"stepper": "G4TDormandPrince45+G4IntegrationDriver"`), never the interpolation, B-field, FSAL or QSS drivers. A step
near a boundary is integrated past it and the boundary is then located by re-integrating from the step's start, so
only the field within about `deltaIntersection` of the boundary enters the trajectory. A patch is correct about one
grid cell beyond its region (its margin), so `delta_intersection_m` must be at most 0.1 of the finest spacing of the
attached patches (refused by the driver with `particle.integrator_invalid`, and by the program for any job file
it is handed: exit status 2, nothing written). Measured on a field jump at a sphere (1 T
inside, 0.2 T outside): per-volume patches correct to 1 um beyond the boundary give the exact piecewise circles to
2e-6 m; a field that is zero at once beyond the boundary, a single point-located field, the interpolation driver or a
deltaIntersection of 0.1 mm give 2e-4 to 6e-3 m. That accuracy holds for a track that enters a volume deeper than
about deltaChord (0.25 mm): Geant4 can miss a volume that a chord of the step enters by less, so a track that grazes
the sphere (entering by 1e-7 to 5e-5 m) leaves it with errors of up to 2.5e-3 m (0.5 GeV/c) and 7.6e-3 m
(0.1 GeV/c), equivalent to an error of at most deltaChord in its impact parameter; entries of 1 to 3 mm are exact to
7e-6 m and 5e-5 m. A track started within a micrometre of an interface and parallel to it is the same case.

`integrator` takes any of `delta_chord_m` (2.5e-4), `delta_one_step_m` (1e-5), `delta_intersection_m` (1e-6),
`epsilon_min` (5e-5), `epsilon_max` (1e-3) and `min_step_m` (1e-5), Geant4's defaults; positive, with
`epsilon_min <= epsilon_max <= 1e-2`. A `delta_intersection_m` above 0.1 of the thinnest tube of the solids warns
`particle.delta_intersection_not_small`.

**Outside the maps.** At set-up the driver checks that each region served by a patch lies in the patch's box
enlarged by its margin (kind 1: the solids' (r, z) extent; kind 3: the corners of the solids' bounding cylinder in
the patch frame), and that, when `fields` is not empty, every region is served or field-free by a map's
`zero_field_regions`; otherwise it refuses with `particle.field_outside_map`, naming the region and the patch. With
`outside="zero"` the run proceeds instead: unserved regions are field-free, a field call outside a patch's box plus
margin at a point inside the volume returns 0 and is counted (`outside_calls`), and the result warns
`particle.field_outside_map` with the total count and each map's `boundary_ratio`. During the run, a stepper's
trial point beyond a patch's box outside the volume (tens of mm beyond a volume whose face is the box face) takes the
value of the box's nearest point and is not counted: such values do not enter the trajectory.

**The domain.** The world and the voids (solids without a region) end the transport's domain: a track that is in
them leaves it, its kinetic energy added to the event's `exit_eV`. A track is in them when it leaves the world, or at
the end of a step longer than the surface tolerance that started in the world or a void and whose first micrometre
(at most half the step) lies in the world or a void by a locate of Geant4's navigator without history. A step that
merely ends in the world is no exit: at an interface between two regions Geant4 can report the world (the mother of
both) for one zero-length step and return the track on the next.

### Physics

`physics_list` is one of `FTFP_BERT` (the default), `FTFP_BERT_EMZ` (Geant4's most precise electromagnetic option)
and `QGSP_BIC` (`particle.physics_list_unknown` otherwise, naming them; a high-precision neutron list `*_HP` is
refused with `particle.physics_list_unavailable`). `production_cut_m` (default 0.7 mm) is one production cut for all
particles and regions. Geant4 kills particles that loop in a field without progress; their number and energy are
recorded per event, and the result warns `particle.looping_killed` when their energy exceeds 1e-6 of the deposited
energy. A track that Geant4's navigator abandons as stuck (not moving for 25 steps, `GeomNav0003`; seen for a track
that snakes along an interface where the field reverses, which a solved map does not hold but adjacent uniform
fields of opposite sign do) is ended where it stands instead of the event being aborted: the event and its chunk
complete, the track's number and kinetic energy are recorded per event (`abandoned_killed`, `abandoned_eV`; not in
`exit_eV`), and the result warns `particle.track_abandoned`.

### Materials: NIST compositions

`particles.resolve_materials(doc)` returns the solids document with its NIST compositions resolved against the
Geant4 the program is built with (`axmp-transport --nist`); the driver runs it before every run and records
`materials_resolved` (per material `as`: `nist` or `one_component`, and NIST's data). A material whose block is
exactly `{"nist": X}` at X's own density (within 1e-9) becomes the NIST material itself; any other keeps one
component and takes X's mean excitation energy and state where the document gives none; a density more than 1 %
from NIST's warns `particle.density_differs_from_nist`. The residual of the second case and the reasons are in
[Solids for particle transport](solids.md#materials).

### Codes

| code | kind | raised for |
|---|---|---|
| `build.requires_program` | error | `axmp-transport` is not found (or not executable). |
| `events.count_missing` | error | No `events`: a run never defaults to a number of events. |
| `events.invalid` | error | `events` or `chunk_events` not a positive integer; a seed outside [0, 2^64); an extension below the events requested so far; a work directory without a planned run. |
| `events.workdir_not_empty` | error | `plan` into a directory that holds other files and no `run.json`. |
| `events.workdir_busy` | error | `plan`, `execute`, `merge` or `extend` on a workdir whose lock another process holds (the command line: exit 1 and the code). |
| `events.chunk_events_mismatch` | error | `extend` with a `chunk_events` other than the plan's. |
| `events.chunk_mismatch` | error | A chunk whose manifest names another chunk, seed or number of events than the plan. |
| `events.partial` | warning | Fewer events merged than requested (exit status 5 on the command line). |
| `stat.too_few_chunks` | warning | Fewer than 20 merged chunks: `std`, `quantile` and tallies carry no statistical error. |
| `particle.unknown_particle` | error | A particle Geant4's table lacks; the message lists candidates. |
| `particle.source_invalid` | error | Not exactly one gun, an unknown key, both or neither of `energy_eV` and `momentum_eV_c`, a non-positive energy, a zero direction, a negative spot, a spot with an isotropic direction, a position outside every region of the solids. |
| `particle.output_invalid` | error | An output (or a `stop`) that is not one of the forms above. |
| `particle.field_invalid` | error | A malformed field, an unknown region, a map patch without regions, an `outside` other than `refuse` or `zero`. |
| `particle.field_ambiguous` | error | A region served by two fields; `transport()` given both `fields=` and `field_map=`. |
| `particle.field_outside_map` | error, or warning with `outside="zero"` | A region beyond its patch's box plus margin, or a region served by no field; with `outside="zero"`, the count of calls outside the maps. |
| `particle.integrator_invalid` | error | An unknown or non-positive integrator value, `epsilon_min > epsilon_max` or `epsilon_max > 1e-2`, a `delta_intersection_m` above 0.1 of the finest map spacing. |
| `particle.delta_intersection_not_small` | warning | `delta_intersection_m` above 0.1 of the thinnest tube of the solids. |
| `particle.physics_list_unknown` | error | A physics list other than the three offered. |
| `particle.physics_list_unavailable` | error | A high-precision neutron list (`*_HP`): not offered in this version. |
| `particle.tally_overflow` | error | A chunk whose tally could reach 2^63 quanta. |
| `particle.partition_unavailable` | error | `transport()` without `solids=`: this version does not run the design's partition export for it (`capabilities()["partition_export"]` is false). |
| `particle.density_differs_from_nist` | warning | A NIST composition more than 1 % from NIST's density. |
| `particle.looping_killed` | warning | Looping tracks killed carrying more than 1e-6 of the deposited energy. |
| `particle.track_abandoned` | warning | Tracks that Geant4's navigator abandoned as stuck (`GeomNav0003`) were ended; their number and energy are in the arrays `abandoned_killed` and `abandoned_eV`. |
| `events.field_map_changed` | error | `execute`, `extend` or `run` on a workdir whose field map changed or vanished since the plan (its manifest's SHA-256). |

The codes of the solids list (`particle.partition_invalid`, `particle.geometry_overlap`, ...) are in
[Solids for particle transport](solids.md#codes). Each error is an `AxionMultiPhysicsError` with `code` and `hint`,
whose message names the offending item and ends with `. Hint: ...`; each warning a `particles.ParticleWarning` whose
message starts with its code.

### Limits of this version

- One gun per run; no event generator yet.
- The transport is validated with axisymmetric solids lists only. 3D lists (`tessellated` solids from
  `Design.export_gdml` of a 3D design) are built and checked by `axmp-transport --geometry` (volumes, masses,
  overlaps), but no transport through them is validated yet (the 3D dipole of the demonstration skips).
- `transport()` needs `solids=`: it does not run the partition export itself (`capabilities()["partition_export"]`).
- Statistical errors only; the physics-list dependence is not estimated (compare `FTFP_BERT` with `QGSP_BIC` or
  `FTFP_BERT_EMZ` by hand where it matters). No energy-balance check yet.
- Reproducible bit for bit on one build, compiler and platform; another build agrees statistically only.

## The program `axmp-transport`

`axmp-transport` reads a job file, transports events, and writes NPY arrays and a run record. Its modes: `transport`
(the driver's), `probe` (the fields at given points) and `hello` (muons through a silicon slab: a check of the
toolchain). `axmp-transport --geometry solids.json` builds a solids list in Geant4 and checks it for overlaps (see
[Solids for particle transport](solids.md)); `axmp-transport --nist NAME [NAME ...]` prints Geant4's own data of NIST
materials.

### Licences

The source of `axmp-transport` is BSD-3-Clause. The running program links Geant4 (Geant4 Software License) and the
LGPL-3 CLHEP that every Geant4 build contains. For that reason it is a separate program, it is built only on request
(`-DAXMP_BUILD_TRANSPORT=ON`, off by default and always off when building a wheel), and no wheel contains Geant4 or
CLHEP code. Geant4's data sets are distributed by CERN and are never redistributed or downloaded by this package. The
full rule is in [Installation: licensing of each wheel](../install.md#licensing-of-each-wheel).

### Building it

You need a Geant4 from 11.2 up to, not including, 11.5, built multithreaded (`GEANT4_BUILD_MULTITHREADED=ON`), with
its data sets installed (Geant4's `GEANT4_INSTALL_DATA=ON`, or an existing installation). The 12 data sets that
Geant4 11.4 installs by default are 1.65 GB compressed and about 2.2 GB unpacked; Geant4 finds them under
`<prefix>/share/Geant4/data`, or where `GEANT4_DATA_DIR` or the variable of a data set points. 11.4.3 is the version the
project is tested with; another version in the range is accepted and reported as untested by `--check`.

```bash
cmake -S . -B build -DAXMP_BUILD_TRANSPORT=ON -DGeant4_DIR=<geant4 prefix>/lib64/cmake/Geant4
cmake --build build --target axmp-transport
```

The configure refuses a Geant4 outside the range or without multithreading and says what to change. (Geant4's own
version file checks only the lower end of the range that `find_package` asks for, so the upper end is checked
explicitly.)

### Checking an installation

```bash
axmp-transport --check
```

prints one JSON document on stdout (Geant4's banner and messages go to stderr): the program and Geant4 versions,
whether that Geant4 is the tested one, the physics lists and modes offered, and the Geant4 data sets with where each
was found, whether it is `ok` or `missing`, and whether the offered physics list requires it (FTFP_BERT reads G4EMLOW, G4PARTICLEXS, PhotonEvaporation and
G4ENSDFSTATE). `ok` is true when every
required data set is present. `"geometry": true` says that the program has the geometry mode, `"nist": true` that it
has `--nist`; `modes` lists `hello`, `transport` and `probe`, `physics_lists` the three offered, `stepper` the pinned
integration driver, and `particles` Geant4's particle table once `FTFP_BERT` has built it (name: PDG code, mass in
eV, charge; ions left out).

```bash
axmp-transport --nist G4_WATER G4_SODIUM_IODIDE
```

prints `{"schema": "axionmultiphysics.transport-nist/1", "materials": {NAME: {"density_kg_m3",
"mean_excitation_eV", "state", "T_K", "pressure_Pa", "elements": {symbol: mass fraction}}}}`, each value converted
once from Geant4's units and written so that it reads back as the same double; a name Geant4's NIST manager does not
know ends it with exit status 2 (named on stderr, nothing on stdout).

### Running a job

```bash
axmp-transport [--threads N] job.json
```

The run manager is always Geant4's tasking run manager with `N` threads (default 1), also for `N = 1`. Relative paths
in the job resolve against the directory of the job file. Unknown or missing keys, wrong types and values out of
range are refused before Geant4 starts, and so is a `delta_intersection_m` above 0.1 of the finest spacing of the
map patches the job attaches (exit status 2, nothing written). A job file of the mode `transport` (schema
`axionmultiphysics.transport-job/1`), as the driver writes it:

```json
{"schema": "axionmultiphysics.transport-job/1", "mode": "transport", "out": "../out", "record": "../out/record_0.json",
 "chunks": [{"index": 0, "seed": 1234, "events": 1000}],
 "physics_list": "FTFP_BERT", "production_cut_m": 0.0007,
 "solids": "solids.json",
 "fields": [{"type": "map", "path": "/data/run1/fieldmap", "patch": "air", "regions": ["air"]}],
 "outside": "refuse",
 "integrator": {"delta_chord_m": 0.00025, "delta_one_step_m": 1e-05, "delta_intersection_m": 1e-06,
                "epsilon_min": 5e-05, "epsilon_max": 0.001, "min_step_m": 1e-05},
 "source": {"type": "gun", "particle": "mu+", "momentum_eV_c": 1e9, "at_m": [0, 0, 0], "direction": [0, 0, 1],
            "spot_sigma_m": 0.0},
 "quantum_eV": 0.001,
 "tallies": [{"name": "q", "grid": {"type": "cylindrical", "r_m": [0, 0.5], "z_m": [-1, 1], "bins": [10, 20]},
              "regions": ["iron"]}],
 "hits": [{"name": "tracker", "regions": ["layer_0", "layer_1"], "track": "any", "particles": null,
           "max_per_event": 100}]}
```

`chunks` is a list of `{index, seed, events}` run in the listed order in one process (the geometry and the physics
are built once), one `BeamOn` each; `seed` is an integer in [0, 2^64). Every region is named explicitly (the driver
resolves the defaults). `fields`, `tallies` and `hits` may be empty. `debug` is for tests only (`fail_chunks`: the
process ends with status 4 on reaching one of these chunks, before running it).

**Mode `probe`.** `{"schema", "mode": "probe", "out", "solids", "fields", "outside", "probe": {"points":
"points.npy"}}` with an `(n, 3)` float64 array of points in m: each point is located in its volume (a navigator of
the built geometry) and that volume's field object is evaluated as the stepper would. It writes
`<out>/probe/B_T.npy` (`(n, 3)`, tesla), `<out>/probe/region.npy` (`(n,)` int32, the index in the solids' `regions`,
-1 for the world and the voids, whose field is 0) and `<out>/probe/manifest.json` (`outside_calls`). Map patches are
evaluated as `FieldMap.evaluate` evaluates them (bilinear in (r, z) for kind 1 with B_x = B_r cos(phi) - B_phi
sin(phi), B_y = B_r sin(phi) + B_phi cos(phi); trilinear for kind 3 in the patch frame, rotated back; clamped within
the margin), within 1e-12 of the field's scale. A point beyond its patch's box plus margin ends the probe with exit
status 4 and `fieldmap.outside_map` on stderr (`outside: "refuse"`) or gets 0 and is counted (`"zero"`).

**Mode `hello`.** Muons through a silicon slab in an optional uniform field: a check of the toolchain, the seeding
and the reproducibility.

```json
{"schema": "axionmultiphysics.transport-job/1", "mode": "hello", "out": "out",
 "chunks": [{"index": 0, "seed": 1, "events": 200}],
 "hello": {"particle": "mu-", "energy_eV": [5.0e8, 1.5e9], "field_T": [0.0, 1.0, 0.0], "target_thickness_m": 0.01}}
```

A world box of half-lengths 1 m of `G4_Galactic`; a box of half-lengths 0.5 m by 0.5 m by half the thickness of
`G4_Si` at the origin. The gun starts at (0, 0, -0.5 m) along +z with kinetic energy `low + (high - low) u`, `u`
being the first random number of the event. A non-zero `field_T` attaches a uniform field to the whole world. Its
arrays are `event`, `seed_words`, `first_flat`, `ekin_eV`, `edep_eV` (in the target), `n_steps`, `n_tracks` and
`field_calls`.

**Exit status.** 0 success. 2 usage or invalid job. 3 refused at set-up: a required data set is missing, Geant4 is
not multithreaded, `SeedOncePerCommunication` is not 0, or Geant4 does not build the pinned integration driver. 4
failure during the run: any Geant4 exception that is not a warning, including one that would abort a single event,
ends the process with this status and writes no output for the chunk in progress (events are never dropped); so does
a field call outside a map at a point inside its volume with `outside: "refuse"` (`fieldmap.outside_map`). An error
is one line on stderr that starts with `axmp-transport: error:`. Nothing is written under `out` when the status is 2
or 3.

### Chunk outputs

For every chunk `k`, `<out>/chunk_<k>/` holds one NPY file per array and `manifest.json`; the directory is written as
`chunk_<k>.<pid>.tmp` (a directory of the writing process's own, so that a program left running by a killed driver
cannot collide with a new one) and renamed when complete, replacing an earlier `chunk_<k>` (a chunk that did not
finish is never renamed). Row `i` of every per-event array is event `i` of the chunk, whatever order the threads finished in.
Mode `transport`:

| array | dtype, shape | content |
|---|---|---|
| `event` | `<u8` (n,) | event index |
| `seed_words` | `<u4` (n, 4) | the four words that seeded the event |
| `first_flat` | `<f8` (n,) | u0, the first random number of the event |
| `primary` | `<f8` (n, 7) | x, y, z [m], px, py, pz [eV/c], E_kin [eV] of the primary as generated |
| `edep_eV` | `<f8` (n, R) | energy deposited per region (R regions, the solids' `regions` order) |
| `exit_eV` | `<f8` (n,) | kinetic energy of tracks entering the world or a void |
| `n_steps`, `n_tracks`, `field_calls`, `outside_calls`, `looping_killed` | `<u8` (n,) | per event |
| `looping_eV` | `<f8` (n,) | kinetic energy of the looping tracks killed |
| `abandoned_killed` | `<u8` (n,) | tracks Geant4's navigator abandoned as stuck (`GeomNav0003`), ended by the program |
| `abandoned_eV` | `<f8` (n,) | their kinetic energy (booked here, not in `exit_eV`) |
| `hits_<name>_count`, `hits_<name>_dropped` | `<u4` (n,) | records kept and dropped per event |
| `hits_<name>_event` | `<u8` (m,) | the hit records, sorted by event, then in the event's step order |
| `hits_<name>_track`, `_parent`, `_pdg`, `_region` | `<i4` (m,) | |
| `hits_<name>_x_m`, `hits_<name>_p_eV_c` | `<f8` (m, 3) | the pre-step point's position; momentum |
| `hits_<name>_t_s`, `hits_<name>_ekin_eV` | `<f8` (m,) | global time since the primary vertex; kinetic energy |
| `tally_<name>` | `<i8` (grid bins) | sum over the chunk's events of each event's bin total rounded to the quantum |

`manifest.json` (schema `axionmultiphysics.transport/1`) records `chunk`, `chunk_seed`, `events`, `threads`, the
versions, `physics_list`, `production_cut_m`, `integrator` (with `stepper`), `quantum_eV`, `outside`, `regions`,
`arrays` (each array's file, dtype, shape and SHA-256), `counters` (`field_calls`, `outside_calls`,
`looping_killed`, `looping_eV`, `abandoned_killed`, `abandoned_eV`, `hits_dropped`) and `timing`. The record (`record`, default `<out>/record.json`)
records the status, the mode, the threads, the chunks run, `seed_once_per_communication` (always 0), the totals of
field calls, outside calls, looping tracks killed and abandoned tracks, and `timing`: `startup_s`, the time from the start of the
process to the first event (Geant4's initialisation, physics tables and thread start; about 1 s for `FTFP_BERT` on
the cluster's nodes, whatever the number of threads), and `total_s`.

### Seeding and reproducibility

The seed of an event depends only on the chunk seed and the event's index in the chunk. The four 32-bit words are the
outputs `2e` and `2e+1` of the SplitMix64 sequence that starts at the chunk seed (the outputs of the generator of
Steele, Lea and Flood, 2014), low half first. They are installed with `G4Random::setTheSeeds(words, 4)` as the first
statement of the primary generator's `GeneratePrimaries`, on a MixMax engine, in every mode. Geant4 seeds the event
itself first; the words overwrite that, so the chunk size is not limited by Geant4's refill of its seeds after 10 000
events. The program refuses to start when Geant4's `SeedOncePerCommunication` is not 0, the mode in which the seeds of
events would depend on the thread that ran them.

An event is therefore the same whatever `--threads` is, whichever chunks ran before it in the process, and whichever
position it has in a longer chunk with the same seed: the arrays of a chunk are bitwise identical in all those runs
(the manifests differ in `threads` and `timing` only). This holds for one build, compiler and platform; another build
agrees statistically only.
