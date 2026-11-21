"""Design documents: load, build from templates, edit, validate, solve."""
from __future__ import annotations

import copy
import json
import json as _json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _core

if TYPE_CHECKING:          # annotations only (results imports design)
    from .results import Results

# One line of the pretty-printed JSON `axmp` writes (nlohmann dump(2)): an optional key, then a bracket, a closing
# bracket or a scalar, an optional comma. Strings hold no raw newlines, so every line of the document matches.
_STR = r'"(?:[^"\\]|\\.)*"'
_JSON_LINE = re.compile(r"^\s*(?:" + _STR + r"\s*:\s*)?(?:[{\[]|[}\]],?|(?:" + _STR +
                        r"|-?\d[\d.eE+\-]*|true|false|null|\{\}|\[\])\s*,?)\s*$")


def _is_axmp_result(obj: Any) -> bool:
    """`axmp solve`'s summary: {"ok": true, "results": "<path>", ...} or {"ok": false, "error": "...", ...}."""
    if not isinstance(obj, dict) or not isinstance(obj.get("ok"), bool):
        return False
    return isinstance(obj.get("results" if obj["ok"] else "error"), str)


def _pretty_document(lines: list[str], first: int) -> str | None:
    """The pretty-printed JSON document that opens at lines[first] ("{" at column 0), skipping foreign lines.

    nlohmann's dump(2) puts one key or value per line, indented by two spaces per level of nesting. A line belongs
    to the document only if it has that form (_JSON_LINE) and exactly the indentation its depth requires; other
    lines (warnings) are skipped. The document is rejected (None) rather than accepted without a member when a
    skipped line could be one of its own lines displaced or damaged by a foreign fragment spliced into it (srun
    forwards output in 1 KiB pieces): a line of JSON form at the wrong indentation, or a line at the required
    indentation that starts like a member (a quote). None also if the document does not close.
    """
    if lines[first] != "{":
        return None
    kept = ["{"]
    depth = 1
    for line in lines[first + 1:]:
        body = line.lstrip(" ")
        closing = body[:1] in ("}", "]")
        indent = 2 * (depth - 1 if closing else depth)
        json_form = bool(_JSON_LINE.match(line))
        if not json_form or len(line) - len(body) != indent:
            if json_form or (len(line) - len(body) == indent and body[:1] == '"'):
                return None
            continue
        kept.append(line)
        if closing:
            depth -= 1
        if body.rstrip().endswith(("{", "[")):
            depth += 1
        if depth == 0:
            return "\n".join(kept)
    return None


def _parse_axmp_output(stdout: str) -> dict | None:
    """The summary object of an `axmp solve` run from its stdout, or None.

    MPI launchers and transports (UCX, OpenMPI, PMIx, srun) may print warnings into the same stream: before, after,
    or, from other ranks, between the lines of rank 0's JSON. Warnings may contain
    braces and even small JSON objects, so an object is accepted only if it opens with "{" at column 0 and is the
    summary (_is_axmp_result); errors printed by every rank before MPI starts give several copies, and the first is
    taken. Pass 1 decodes the raw stream at each such line. Pass 2 rebuilds each pretty-printed document without the
    foreign lines inside it (_pretty_document). A summary cut short, or with a member damaged by a spliced
    fragment, is not accepted. `axmp solve` also writes the summary to DIR/axmp_status.json, which Design.solve reads
    first (_read_axmp_status); this parser is the fallback.
    """
    dec = json.JSONDecoder()
    for m in re.finditer(r"(?m)^\{", stdout):
        try:
            obj = dec.raw_decode(stdout, m.start())[0]
        except ValueError:
            continue
        if _is_axmp_result(obj):
            return obj
    # split at newlines only: str.splitlines would also split at U+2028 or U+0085 inside strings (raw UTF-8)
    lines = [line[:-1] if line.endswith("\r") else line for line in stdout.split("\n")]
    for k in range(len(lines)):
        doc = _pretty_document(lines, k)
        if doc is None:
            continue
        try:
            obj = json.loads(doc)
        except ValueError:
            continue
        if _is_axmp_result(obj):
            return obj
    return None


def _read_axmp_status(path: Path) -> dict | None:
    """The summary `axmp solve` wrote to DIR/axmp_status.json (rank 0, atomically), or None."""
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return obj if _is_axmp_result(obj) else None


def schema() -> dict:
    """JSON Schema of the design format (axionmultiphysics.design/1)."""
    return json.loads(_core.schema())


def physics(routes: bool = False, type: str | None = None) -> dict:  # noqa: A002 (the name of the CLI flag)
    """The physics catalogue of model documents (beta), as ``axmp physics --json`` prints it.

    The catalogue lists the physics types (their options, condition kinds, sources, quantities per geometry class and
    what is not implemented), the coupling terms and the reserved instance ids; ``routes=True`` adds the route table,
    one row per combination of physics, couplings, geometry class and study type that an engine of this version
    solves, with what each row offers and needs, and the rules of this layer. Every entry is beta, and no route is
    implemented yet (every row has ``"implemented": False`` and ``"status": "not_implemented"``). The catalogue is the
    same in every build. The schema string is ``axionmultiphysics.physics_catalogue/1``.

    Parameters
    ----------
    routes
        Add the route table (``"routes"``) and the layer rules (``"layer_rules"``).
    type
        Restrict the catalogue to one physics type, the rows whose physics contain it and the coupling terms it
        produces or consumes.

    Returns
    -------
    dict
        The catalogue: ``schema``, ``status``, ``types``, ``terms``, ``reserved_ids`` and, with ``routes=True``,
        ``routes`` and ``layer_rules``.

    Raises
    ------
    AxionMultiPhysicsError
        ``type`` is not the name of a physics type (an alias is refused too); the hint names the types.
    """
    return json.loads(_core.physics(bool(routes), type or ""))


def templates() -> list[dict]:
    """Parametric templates with their parameters (default, units, meaning)."""
    return json.loads(_core.templates())


def template(name: str, **params: Any) -> "Design":
    """Instantiate a parametric template, overriding parameter defaults.

    Values may be numbers or expression strings over other parameters,
    e.g. ``template("core", Rc=1.5, b="2*Rc")``.
    """
    return Design(json.loads(_core.template(name, json.dumps(params))))


def load(path: str | Path) -> "Design":
    """Load a design from a JSON file.  A model document (one with "physics") that repeats a key in an object is
    refused (``json.duplicate_key``): JSON would keep only the last value."""
    text = Path(path).read_text()
    doc = json.loads(text)
    if isinstance(doc, dict) and "physics" in doc:
        _core.scan_duplicate_keys(text, str(path))
    return Design(doc)


def _require_em_document(design: Any, operation: str) -> None:
    """Refuses a model document (``model.em_only_operation``, naming ``operation``) at an operation that works on
    electromagnetic designs only; returns for any other document."""
    doc = design.doc if isinstance(design, Design) else design
    if isinstance(doc, dict) and "physics" in doc:
        _core._require_em_document(json.dumps(doc), operation)


def _mesher_unavailable(program: str, reason: str) -> Exception:
    """``build.requires_mesher``: the mesher does not run, with the loader's own account of why (as the model path)."""
    hint = ("A wheel installed without gmsh: `pip install \"axionmultiphysics[mesh]\"`. A source build: configure with "
            "-DAXMP_BUILD_MESHER=ON and the gmsh SDK. A mesher installed elsewhere: $AXMP_MESHER (or mesher=) names it.")
    exc = _core.AxionMultiPhysicsError(f"the mesher '{program}' does not run in this installation, and the export "
                                       f"needs it: {reason}. Hint: {hint}")
    exc.code, exc.hint = "build.requires_mesher", hint
    return exc


def _log_tail(path: Path, lines: int = 40, size: int = 4000) -> str:
    """The last lines of a log (at most ``lines`` lines and ``size`` characters); empty when it cannot be read."""
    try:
        text = Path(path).read_text("utf-8", errors="replace")
    except OSError:
        return ""
    text = "\n".join(text.splitlines()[-lines:])
    return "..." + text[-size:] if len(text) > size else text


def _mesh_failed(why: str, log: Path, status: int | None, kept: bool) -> Exception:
    """``mesh.generation_failed`` with the log's last lines in the hint (a temporary directory is gone when the error
    is read); ``kept``: the log stays where it is."""
    tail = _log_tail(log)
    where = f"The mesher's log is '{log}'" if kept else "The mesher's log was in a temporary directory, now removed"
    hint = where + (f" (exit status {status})" if status is not None else "")
    hint += (f"; its last lines:\n{tail}" if tail else "; it is empty.")
    exc = _core.AxionMultiPhysicsError(f"mesh generation failed: {why}. Hint: {hint}")
    exc.code, exc.hint = "mesh.generation_failed", hint
    return exc


class Design:
    """A design document (dict) with helpers.

    The document is the single source of truth; this class never hides state.
    ``d.doc`` is the underlying dict and may be edited directly.
    """

    def __init__(self, doc: dict):
        if isinstance(doc, Design):
            doc = doc.doc
        self.doc: dict = copy.deepcopy(doc)

    # ------------------------------------------------------------ basics
    @property
    def name(self) -> str:
        return self.doc.get("name", "design")

    @property
    def params(self) -> dict:
        """Parameter definitions (numbers or expressions)."""
        return self.doc.setdefault("parameters", {})

    def resolved_params(self) -> dict:
        """Numerical values of all parameters (expressions evaluated)."""
        return json.loads(_core.validate(json.dumps(self.doc)))["parameters"]

    def set(self, **params: Any) -> "Design":
        """Change existing parameters in place; returns self.

        Raises KeyError for unknown names (use ``add_param`` to create one).
        """
        for k, v in params.items():
            if k not in self.params:
                known = ", ".join(sorted(self.params))
                raise KeyError(f"unknown parameter '{k}'; known parameters: {known}")
            self.params[k] = v
        return self

    def add_param(self, name: str, value: Any) -> "Design":
        self.params[name] = value
        return self

    def with_params(self, **params: Any) -> "Design":
        """A modified copy."""
        return Design(self.doc).set(**params)

    def copy(self) -> "Design":
        return Design(self.doc)

    def validate(self, strict: bool = False, all_errors: bool = False) -> dict:
        """Parse and check the design in C++; raises AxionMultiPhysicsError with a hint on problems.

        A model document (one with "physics") returns the validation report: ``ok``, ``name``, ``parameters`` and the
        findings ``errors``, ``unavailable``, ``deferred``, ``warnings``.  It raises the first error, and with
        ``strict`` the first unavailable finding; ``all_errors=True`` returns the report without raising.  An
        electromagnetic design ignores both flags.
        """
        if "physics" in self.doc:
            return json.loads(_core.validate(json.dumps(self.doc), bool(strict), bool(all_errors)))
        return json.loads(_core.validate(json.dumps(self.doc)))

    def boundaries(self, mesh: bool = False, all_errors: bool = False) -> dict:
        """The named boundaries of a model document, as ``axmp boundaries --json`` prints them (beta).

        Validates the document, runs the mesher's ``resolve_only`` pass and the full check (the partition rule of every
        physics, the empty selections, the partial matches, the outputs on resolved boundaries), and returns the
        validation report with ``symmetry``, ``boundaries`` (one row per name: ``selector``, ``entities``, ``measure``,
        ``bbox``, ``point``, ``adjacent``, ``used_by``) and ``edges`` added. ``mesh=True`` also meshes the model and
        adds the number of boundary elements, ``faces``, to every row and edge. Raises AxionMultiPhysicsError with the
        first error (also ``build.requires_mesher`` when the mesher does not run, and ``mesh.generation_failed``);
        ``all_errors=True`` returns the report instead of raising. Needs the mesher; an electromagnetic design is
        refused.
        """
        return _json.loads(_core.boundaries(_json.dumps(self.doc), bool(mesh), "", "", bool(all_errors)))

    def explain(self, json: bool = False) -> "str | dict":  # noqa: A002 (the name of the CLI flag)
        """The resolved model of a model document (beta), as ``axmp explain`` prints it.

        Every default is listed and tagged, each physics with its boundary conditions and the form the engine gives
        them, the coupling terms, the study, the route each group of physics takes and what it needs. Returns the text,
        or with ``json=True`` the dict of schema ``axionmultiphysics.explain/1``. Raises AxionMultiPhysicsError for a
        document that fails validation (the first error) and for an electromagnetic design.
        """
        text = _core.explain(_json.dumps(self.doc), json)
        return _json.loads(text) if json else text

    def patch(self, ops: Any) -> "Design":
        """Apply an RFC 6902 JSON Patch (a list of operations) atomically; returns self.

        Either every operation applies and the result validates, or AxionMultiPhysicsError is raised (code
        ``patch.operation_failed`` with the index of the operation in ``where``, or the validation's own error) and
        the document is unchanged. Paths are JSON pointers (RFC 6901). Works on electromagnetic designs too.
        """
        self.doc = _json.loads(_core.patch(_json.dumps(self.doc), _json.dumps(ops)))
        return self

    # ------------------------------------------------------------ io
    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.doc, indent=indent)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_json())
        return p

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Design({self.name!r}; {ps})"

    # ------------------------------------------------------------ solve
    def solve(self, workdir: str | Path | None = None, order: int = 0, verbose: int = 1,
              threads: int | None = None, mpi_ranks: int = 1, launcher: str | None = None,
              self_force: bool = True, force_samples: int = 200) -> "Results":
        """Mesh and solve; returns Results.

        workdir defaults to ``runs/<name>``.  ``order`` overrides mesh.order.
        3D designs: ``mpi_ranks > 1`` runs ``axmp solve --save`` on that many MPI ranks (``launcher``: e.g.
        "srun" or "mpirun"; default: $AXMP_MPI_LAUNCHER, else srun inside a SLURM allocation when srun is on PATH,
        else mpirun, else mpiexec) and
        loads the saved solution, so sampling and scenes work as after a single-process solve.

        ``self_force`` estimates (and, for linear designs, removes) the numerical self-force of each
        powered azimuthal winding, at the price of one extra linear solve per such winding
        (axisymmetric designs only).

        ``force_samples`` is the number of samples per body in ``forces.distributions``.  The merge
        that produces them preserves the integral, so the net forces do not depend on it; it only
        sets how finely the load is resolved and how big ``results.json`` gets (about 160 bytes per
        sample per body).  0 keeps the net forces and the closure check but writes no arrays.
        """
        from .results import Results

        if "physics" in self.doc:  # a model document: solving it is not implemented in this version
            _core._refuse_unserved_model_document(json.dumps(self.doc), "solve")
        if threads:
            from . import set_threads

            set_threads(threads)
        wd = Path(workdir) if workdir else Path("runs") / self.name
        if self.doc.get("symmetry", "axisymmetric") == "xsec":  # z-uniform cavity cross-section (axionmultiphysics.cavity_xsec)
            from .cavity_xsec import XsecResults

            if mpi_ranks > 1:
                raise ValueError("mpi_ranks > 1 is for 3D designs; the cross-section solver runs in one process")
            modelx = _core.XsecModel(json.dumps(self.doc))
            modelx.run(str(wd), int(order), int(verbose), "")
            return XsecResults(modelx, self, wd)
        if self.doc.get("symmetry", "axisymmetric") in ("axisymmetric", "planar") and \
                "file" in (self.doc.get("mesh") or {}):  # 2D full wave on an imported mesh (COMSOL 2D import)
            from .fw2d import Fw2DResults

            if mpi_ranks > 1:
                raise ValueError("mpi_ranks > 1 is for 3D designs; the 2D imported-mesh solver runs in one process")
            model2 = _core.Fw2DModel(json.dumps(self.doc))
            model2.run(str(wd), int(order), int(verbose))
            return Fw2DResults(model2, self, wd)
        if self.doc.get("symmetry", "axisymmetric") == "3d":
            from .results import Results3D

            if mpi_ranks > 1:
                return self._solve_mpi(wd, order, verbose, mpi_ranks, launcher, threads)
            model3 = _core.Model3D(json.dumps(self.doc))
            model3.run(str(wd), int(order), int(verbose), "")
            return Results3D(model3, self, wd)
        if mpi_ranks > 1:
            raise ValueError("mpi_ranks > 1 is for 3D designs; the axisymmetric engine runs in one process")
        model = _core.Model(json.dumps(self.doc))
        model.run(str(wd), int(order), int(verbose), "", bool(self_force), int(force_samples))
        return Results(model, self, wd)

    # ------------------------------------------------------------ particle transport
    def export_gdml(self, path: str | Path, composition: dict, *, solids: str | Path | None = None,
                    chord_tol_m: float | None = None, workdir: str | Path | None = None,
                    mesher: str | Path | None = None) -> dict:
        """Writes the geometry of an axisymmetric or 3D electromagnetic design as GDML for particle transport (beta).

        The route is the design's mesher input (the one ``axmp mesh`` writes), ``axmp-mesher --export-regions`` on it
        (the exact partition of the model: its regions, the background ``air`` included, with their curves) and
        :func:`axionmultiphysics.solids.from_partition` on that partition. Returns the solids document
        (``axionmultiphysics.solids/1``) and writes the GDML at ``path`` and, when ``solids`` is a path, the document
        as ``solids.json`` there.

        ``composition`` maps each material name the regions use to a ``particle`` block (``nist``, or ``elements`` and
        ``by``; optional ``mean_excitation_eV``, ``state``, ``T_K``, ``pressure_Pa``) with an optional ``density``
        (kg/m3). A region takes its design material, the background region the material named ``air``. A material's
        density is the design's (as the electromagnetic parser resolves it: the built-in iron, copper and nb values,
        expressions evaluated) when it has one, else the block's ``density``; two different positive values are
        ``particle.composition_invalid``, none ``particle.density_missing``, and a used material without a block
        ``particle.composition_missing``. Blocks of materials no region uses are ignored.

        For a 3D design the mesher input is the 3D engine's (``mesh3d_in.json``, made by the same library function in
        every build, without running the engine: each rank of a parallel program exports on its own), the
        partition the mesher's 3D one (one triangulation of every face, to the chord tolerance) and the solids are
        tessellated; ``chord_tol_m`` then goes to the mesher (default: min(1 mm, 1e-4 x the diagonal of the
        geometry)). A design reduced by symmetry planes (``mesh.symmetry``) is refused.

        ``chord_tol_m`` is the conversion's chord tolerance. ``workdir``, when given, keeps ``regions_in.json`` (the
        mesher input) and ``regions.json`` (the partition); without it nothing is left behind. ``mesher`` is the
        program (default: ``$AXMP_MESHER``, else the one of the package). Refused: a model document
        (``model.em_only_operation``); a 3D design with ``mesh.symmetry``, an
        imported mesh (``mesh.file``) and a mesher without the export mode (``particle.partition_unavailable``); a mesher that cannot be run or loaded (``build.requires_mesher``,
        with the loader's message); a planar design (``particle.planar_not_offered``); a failed mesher run
        (``mesh.generation_failed``, the last lines of the mesher's log in the hint; with ``workdir`` the log is kept
        there as ``regions_mesher.log``). Nothing is written then.
        """
        import os
        import subprocess
        import tempfile

        from . import solids as _solids
        from ._beta import warn_beta
        from ._paths import mesher_hint, mesher_path

        warn_beta("Design.export_gdml")
        _require_em_document(self, "export_gdml")
        symmetry = self.doc.get("symmetry", "axisymmetric")
        three_d = symmetry == "3d"
        if three_d:
            if (self.doc.get("mesh") or {}).get("symmetry"):
                raise _solids._error("partition_unavailable", "the partition of a 3D design reduced by symmetry "
                                     "planes (mesh.symmetry) cannot be exported: the model is part of the device",
                                     "remove mesh.symmetry from the design you export (the planes are for the solver)")
            if chord_tol_m is not None and (not isinstance(chord_tol_m, (int, float)) or isinstance(chord_tol_m, bool)
                                            or not chord_tol_m > 0):
                raise ValueError(f"chord_tol_m must be a positive number of metres, not {chord_tol_m!r}")
        if symmetry in ("xsec", "planar"):
            raise _solids._error("planar_not_offered", f"a {symmetry} design has no solids: particle transport is "
                                 "offered for axisymmetric and 3D designs",
                                 "describe the design as axisymmetric, or transport it in a 3D design")
        if "file" in (self.doc.get("mesh") or {}):
            raise _solids._error("partition_unavailable", "the partition of a design on an imported mesh "
                                 "(mesh.file) cannot be exported: an imported mesh has no exact curves",
                                 "describe the geometry with regions instead of mesh.file")
        if not isinstance(composition, dict):
            raise _solids._error("composition_invalid", "the composition is not a mapping from material names to "
                                 "particle blocks", "pass {material: {\"nist\": ...}} (docs/guides/solids.md)")
        program = str(mesher) if mesher is not None else mesher_path()
        if program is None:
            exc = _core.AxionMultiPhysicsError("axmp-mesher was not found. Hint: " + mesher_hint())
            exc.code = "build.requires_mesher"
            raise exc
        # a mesher without the export mode answers two arguments with its usage; one that does not load, or does not
        # run, is the model path's build.requires_mesher (exit status 127 is the loader's)
        try:
            probe = subprocess.run([program, "--export-regions", os.devnull], capture_output=True, text=True)
        except OSError as e:
            raise _mesher_unavailable(program, f"it cannot be run ({e.strerror or e})") from None
        if probe.returncode == 127:
            text = probe.stderr.strip()[-4000:]
            raise _mesher_unavailable(program, text or "it could not be loaded (exit status 127)")
        if probe.returncode == 2 and "usage:" in probe.stderr and "--export-regions" not in probe.stderr:
            raise _solids._error("partition_unavailable", f"the mesher {program} has no export mode "
                                 "(--export-regions)", "use an axmp-mesher of this version of the package "
                                 "(set AXMP_MESHER or pass mesher=)")
        if probe.returncode != 2:
            how = "was stopped by a signal" if probe.returncode < 0 else f"exited with status {probe.returncode}"
            raise _mesher_unavailable(program, f"it {how} when run with the export flag and no input"
                                      + (f": {probe.stderr.strip()[-4000:]}" if probe.stderr.strip() else ""))
        out_path = Path(path)
        materials_of = {r["name"]: r.get("material", "vacuum") for r in self.doc.get("regions", [])}
        densities = json.loads(_core._material_densities(json.dumps(self.doc)))
        with tempfile.TemporaryDirectory(prefix="axmp-export-") as tmp:
            wd = Path(workdir) if workdir is not None else Path(tmp) / "work"
            wd.mkdir(parents=True, exist_ok=True)
            if three_d:
                # the library function the 3D engine writes its mesher input with: no engine run, nothing collective
                inp = json.loads(_core.mesher_input_3d(json.dumps(self.doc)))
                if chord_tol_m is not None:
                    inp["chord_tol_m"] = float(chord_tol_m)
            else:
                try:  # axmp mesh's own code writes the mesher input (and the design's mesh, which is discarded)
                    _core.mesh(json.dumps(self.doc), str(Path(tmp) / "mesh"), 0, program)
                except _core.AxionMultiPhysicsError as e:
                    if str(e).startswith("mesh generation failed"):
                        why = str(e)[len("mesh generation failed: "):].split(" [at ")[0]
                        raise _mesh_failed(why, Path(tmp) / "mesh" / "mesh_mesher.log", None, False) from None
                    raise
                inp = json.loads((Path(tmp) / "mesh" / "mesh_in.json").read_text("utf-8"))
            inp["output"], inp["info"] = str(wd / "regions.msh"), str(wd / "regions.json")
            (wd / "regions_in.json").write_text(json.dumps(inp, indent=1), "utf-8")
            (wd / "regions.json").unlink(missing_ok=True)
            log = wd / "regions_mesher.log"
            with open(log, "w", encoding="utf-8") as lf:
                run = subprocess.run([program, "--export-regions", str(wd / "regions_in.json")], stdout=lf,
                                     stderr=subprocess.STDOUT)
            part = None
            if (wd / "regions.json").is_file():
                try:
                    part = json.loads((wd / "regions.json").read_text("utf-8"))
                except json.JSONDecodeError:
                    part = None
            if run.returncode != 0 or not isinstance(part, dict) or part.get("ok") is not True:
                why = (part or {}).get("error") if isinstance(part, dict) else None
                raise _mesh_failed(why or "the mesher wrote no partition", log, run.returncode, workdir is not None)
            region_materials = {name: ("air" if name == "air" else materials_of.get(name, "vacuum"))
                                for name in part["regions"]}
            materials = {}
            for m in sorted(set(region_materials.values())):
                if m not in composition:
                    continue  # the conversion refuses it: particle.composition_missing
                block = dict(composition[m]) if isinstance(composition[m], dict) else composition[m]
                rho_c = block.pop("density", None) if isinstance(block, dict) else None
                rho_d = float(densities.get(m, 0.0))
                if rho_d > 0.0:
                    if isinstance(rho_c, (int, float)) and not isinstance(rho_c, bool) and rho_c > 0.0 \
                            and rho_c != rho_d:
                        raise _solids._error("composition_invalid", f"material '{m}': the composition gives the "
                                             f"density {rho_c:g} kg/m3 and the design {rho_d:g} kg/m3",
                                             "give one density: drop it from the composition, or make them equal")
                    rho = rho_d
                else:
                    rho = rho_c if rho_c is not None else 0.0
                materials[m] = {"density": rho, "particle": block}
            doc = _solids.from_partition(part, materials, region_materials,
                                         chord_tol_m=None if three_d else chord_tol_m)
        _solids.write_gdml(doc, out_path)
        if solids is not None:
            _solids.write(doc, solids)
        return doc

    # ------------------------------------------------------------ sensitivities
    def gradient(self, params: Any = None, objectives: Any = None, method: str = "adjoint", step: float = 1e-4,
                 steps: dict | None = None, benchmark: str = "", workdir: str | Path | None = None,
                 order: int = 0, verbose: int = 0) -> dict:
        """Solve and return the derivatives of the figures of merit with respect to parameters.

        Shorthand for ``self.solve(...).sensitivities(...)``; see
        :meth:`axionmultiphysics.Results.sensitivities` for the arguments and the returned dict.
        """
        _require_em_document(self, "gradient")
        res = self.solve(workdir=workdir, order=order, verbose=verbose)
        return res.sensitivities(params=params, objectives=objectives, method=method, step=step, steps=steps,
                                 benchmark=benchmark, order=order)

    def optimize(self, objective: str = "cost", params: dict | None = None, **kw) -> dict:
        """Optimize this design; see :func:`axionmultiphysics.optimize` (``method="global"``: differential evolution)."""
        from .global_opt import optimize as _opt

        return _opt(self, objective=objective, params=params, **kw)

    def _solve_mpi(self, wd: Path, order: int, verbose: int, nranks: int, launcher: str | None,
                   threads: int | None = None) -> "Results":
        import os
        import shlex
        import shutil
        import subprocess
        import uuid

        from ._paths import axmp_path, default_mpi_launcher
        from .results import Results3D

        exe = axmp_path()
        if exe is None:
            raise RuntimeError("cannot find the axmp executable (build target axmp, or set AXMP_EXE)")
        launcher = launcher or default_mpi_launcher()
        wd.mkdir(parents=True, exist_ok=True)
        dpath = wd / "design.json"
        dpath.write_text(json.dumps(self.doc, indent=1))
        # rank 0's summary goes to a status file of this launch only, so that concurrent solves in one workdir do not
        # read each other's
        status = wd / f"axmp_status.{uuid.uuid4().hex}.json"
        cmd = shlex.split(launcher) + ["-n", str(nranks), exe, "solve", str(dpath), "-o", str(wd), "-v", str(verbose),
                                       "--save", "--status-file", str(status)]
        if order:
            cmd += ["--order", str(order)]
        # single node: keep the shared-memory transport (avoids InfiniBand/UCX setup); several nodes need the
        # network transports, so drop the single-process settings of this Python session
        multi_node = int(os.environ.get("SLURM_NNODES", "1")) > 1
        env = {k: v for k, v in os.environ.items() if not (multi_node and k in ("OMPI_MCA_pml", "OMPI_MCA_btl"))}
        if "SLURM_MEM_PER_NODE" in env:  # srun refuses mutually exclusive SLURM_MEM_PER_* settings
            env.pop("SLURM_MEM_PER_CPU", None)
            env.pop("SLURM_MEM_PER_GPU", None)
        env.pop("SLURM_TRES_PER_TASK", None)  # may contradict SLURM_CPUS_PER_TASK in a nested srun
        env["SLURM_EXPORT_ENV"] = "ALL"  # srun passes this environment to the ranks (sites may default to NONE)
        if threads:
            env["OMP_NUM_THREADS"] = str(int(threads))
        timeout = float(os.environ.get("AXMP_MPI_TIMEOUT", "0")) or None  # seconds; unset: no limit
        log = wd / "mpi_solve.log"
        # fields/ of an earlier run must not pass for this run's (exit 0 is accepted below only with fields/ written)
        if (wd / "fields").is_dir():
            shutil.rmtree(wd / "fields")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired as ex:
            log.write_text(str(ex.stdout or "") + "\n" + str(ex.stderr or ""))
            raise _core.AxionMultiPhysicsError(f"MPI solve exceeded AXMP_MPI_TIMEOUT = {timeout} s; see {log}")
        log.write_text(proc.stdout + "\n" + proc.stderr)
        # rank 0's summary: this launch's status file, else stdout among any launcher/transport warnings
        # (_parse_axmp_output). Exit 0 with fields/ written (by this launch: an older one was removed) is a success even
        # if neither can be read (a split line must not turn a finished solve into a failure).
        out = _read_axmp_status(status) or _parse_axmp_output(proc.stdout)
        try:
            status.replace(wd / "axmp_status.json")  # the name `axmp solve` uses by default
        except OSError:
            pass
        fields = wd / "fields"
        if out is None and proc.returncode == 0 and fields.is_dir():
            out = {"ok": True, "results": str(wd / "results.json")}
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        stderr_tail = f". Last lines of stderr:\n{tail}" if tail else ""
        if out is None:
            raise _core.AxionMultiPhysicsError(f"MPI solve failed (exit {proc.returncode}); see {log}{stderr_tail}")
        if not out.get("ok", False):
            err = out.get("error", "MPI solve failed")
            if out.get("hint"):
                err += f" (hint: {out['hint']})"
            exc = _core.AxionMultiPhysicsError(err)
            for key in ("where", "hint", "code"):  # rank 0's fields, as the exception of an in-process solve has them
                if isinstance(out.get(key), str):
                    setattr(exc, key, out[key])
            if isinstance(out.get("details"), dict) and out["details"]:
                exc.details = out["details"]
            raise exc
        if proc.returncode != 0:
            # rank 0 finished (it writes the summary after saving the fields), then the launch failed: another rank,
            # or MPI/UCX teardown. Not accepted silently.
            saved = (f"; rank 0 reported success and saved the solution in {fields} (axionmultiphysics._core.Model3D.load); check "
                     f"the log before using it") if fields.is_dir() else ""
            raise _core.AxionMultiPhysicsError(f"MPI solve failed (exit {proc.returncode}) after rank 0 reported success; see "
                                      f"{log}{saved}{stderr_tail}")
        return Results3D(_core.Model3D.load(str(fields)), self, wd)
