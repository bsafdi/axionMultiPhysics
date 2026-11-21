"""The particle-transport driver: plans a run into chunks and job files, runs ``axmp-transport`` on them, merges the
chunks in a fixed order and estimates the outputs.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from . import _spec
from ._common import (BATCHES, E_CHARGE, INSTANCE, INTEGRATOR, PRODUCTION_CUT_M, QUANTUM_EV, SCHEMA_CHUNK,
                      SCHEMA_JOB, SCHEMA_RESULTS, SCHEMA_RUN, STAGE, error, real, warn)
from ._stats import Measured, _mean, batch_means, estimate_count, estimate_fraction

PROGRAM = "axmp-transport"
PARTITION_EXPORT = {"axisymmetric": False, "3d": False}
_M64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_TIMEOUT_TOOL = 600


# ============================================================================================ seeds
def _mix(z: int) -> int:
    """SplitMix64's output function (Steele, Lea and Flood 2014; Vigna's public-domain splitmix64.c)."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _key(k) -> int:
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("a seed key is an int or a str, not a bool")
    if isinstance(k, (int, np.integer)):
        return int(k) & _M64
    if isinstance(k, str):
        return int.from_bytes(hashlib.sha256(k.encode("utf-8")).digest()[:8], "little")
    raise TypeError(f"a seed key is an int or a str, not {type(k).__name__}")


def derive_seed(seed: int, *keys) -> int:
    """z = seed mod 2^64, then for each key z = mix(((z XOR h(key)) + 0x9E3779B97F4A7C15) mod 2^64), with mix
    SplitMix64's output function, h(int) = the int mod 2^64 and h(str) the first 8 bytes (little-endian) of the SHA-256
    of its UTF-8. Injective in the last key for fixed earlier keys."""
    z = _key(seed)
    for k in keys:
        z = _mix(((z ^ _key(k)) + _GOLDEN) & _M64)
    return z


def chunk_seed(seed: int, chunk: int, stage: str = STAGE, instance: str = INSTANCE) -> int:
    """The seed of chunk ``chunk`` of a run: derive_seed(seed, stage, instance, chunk)."""
    return derive_seed(seed, stage, instance, chunk)


def event_words(chunk_seed_: int, event: int) -> tuple[int, int, int, int]:
    """The four 32-bit words that seed event ``event`` of a chunk: outputs 2e and 2e+1 of the SplitMix64 sequence that
    starts at the chunk seed, each split low half first (the program's rule)."""
    s = _key(chunk_seed_)
    o0 = _mix((s + (2 * int(event) + 1) * _GOLDEN) & _M64)
    o1 = _mix((s + (2 * int(event) + 2) * _GOLDEN) & _M64)
    return o0 & 0xFFFFFFFF, o0 >> 32, o1 & 0xFFFFFFFF, o1 >> 32


# ============================================================================================ the program
def _executable(p) -> bool:
    return bool(p) and os.path.isfile(p) and os.access(p, os.X_OK)


def find_program(program=None) -> str | None:
    """``program``, else $AXMP_TRANSPORT, else axmp-transport next to axmp, else on PATH; None when not found."""
    if program is not None:
        p = os.fspath(program)
        return p if _executable(p) else None
    env = os.environ.get("AXMP_TRANSPORT")
    if env:
        return env if _executable(env) else None
    from .. import _paths
    axmp = _paths.axmp_path()
    if axmp:
        cand = os.path.join(os.path.dirname(axmp), PROGRAM)
        if _executable(cand):
            return cand
    return shutil.which(PROGRAM)


def require_program(program=None) -> str:
    exe = find_program(program)
    if exe is None:
        where = f"'{program}' is not an executable file" if program is not None else "it was not found"
        raise error("build.requires_program", f"particle transport needs the separate program {PROGRAM}: {where}",
                    "build it with cmake -DAXMP_BUILD_TRANSPORT=ON (it needs Geant4) and set AXMP_TRANSPORT to it, "
                    "or pass program=")
    return os.path.abspath(exe)


_CHECKS: dict = {}


def _tool(exe: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([exe, *args], capture_output=True, text=True, timeout=_TIMEOUT_TOOL)


def check(exe: str) -> dict:
    """The program's ``--check`` document (cached per executable and modification time)."""
    st = os.stat(exe)
    key = (exe, st.st_mtime_ns, st.st_size)
    if key not in _CHECKS:
        r = _tool(exe, ["--check"])
        if r.returncode != 0:
            raise error("build.requires_program", f"{exe} --check failed (exit status {r.returncode}): "
                        f"{r.stderr.strip()[-500:]}", "check the Geant4 installation the program was built with")
        _CHECKS[key] = json.loads(r.stdout)
    return _CHECKS[key]


def capabilities() -> dict:
    """What this installation can do for particle transport; never raises.

    ``transport``: ``found``, ``path``, ``check`` (the program's ``--check`` document, or None) and ``ok``;
    ``partition_export``: whether ``transport()`` exports a design's partition by itself (the API route without
    ``solids=``), per geometry (``axisymmetric``, ``3d``); ``generator``: the event generator (not in this version)."""
    out = {"transport": {"found": False, "path": None, "check": None, "ok": False},
           "partition_export": dict(PARTITION_EXPORT),
           "generator": {"found": False, "reason": "no event generator in this version: sources are particle guns"}}
    try:
        exe = find_program()
        if exe:
            out["transport"].update(found=True, path=os.path.abspath(exe))
            c = check(os.path.abspath(exe))
            out["transport"].update(check=c, ok=bool(c.get("ok")))
    except Exception as e:  # noqa: BLE001  (capabilities report, they do not raise)
        out["transport"]["error"] = str(e)
    return out


def nist_data(exe: str, names) -> dict:
    names = sorted(set(names))
    if not names:
        return {}
    r = _tool(exe, ["--nist", *names])
    if r.returncode != 0:
        raise error("particle.composition_invalid", f"Geant4's NIST tables refuse: {r.stderr.strip()[-500:]}",
                    "name NIST materials of Geant4 (G4_WATER, G4_Fe, ...)")
    return json.loads(r.stdout)["materials"]


# ============================================================================================ materials
_OPTIONS = ("mean_excitation_eV", "state", "T_K", "pressure_Pa")


def _resolve(doc: dict, exe: str | None) -> tuple[dict, dict, list]:
    out = copy.deepcopy(doc)
    nist_mats = [m for m in out["materials"] if "nist" in m["composition"]]
    if not nist_mats:
        return out, {}, []
    if exe is None:
        exe = require_program(None)
    data = nist_data(exe, [m["composition"]["nist"] for m in nist_mats])
    record, warns, replace = {}, [], {}
    for m in nist_mats:
        X = m["composition"]["nist"]
        d = data[X]
        rel = abs(m["density_kg_m3"] / d["density_kg_m3"] - 1.0)
        if rel > 0.01:
            w = warn("particle.density_differs_from_nist", f"material '{m['name']}' ({X}) has density "
                     f"{m['density_kg_m3']:g} kg/m3, {100 * rel:.3g} % from NIST's {d['density_kg_m3']:g} kg/m3; the "
                     "document's density is used, with NIST's composition, I-value and state", stacklevel=5)
            warns.append(w)
            out.setdefault("warnings", []).append({"code": "particle.density_differs_from_nist",
                                                   "material": m["name"], "nist": X, "rel_diff": rel})
        rec = {"as": None, "nist": X, **d, "document_density_kg_m3": m["density_kg_m3"]}
        if rel <= 1e-9 and not any(k in m for k in _OPTIONS):
            rec["as"] = "nist"
            replace[m["name"]] = X
        else:
            carried = []
            if "mean_excitation_eV" not in m:
                m["mean_excitation_eV"] = d["mean_excitation_eV"]
                carried.append("mean_excitation_eV")
            if "state" not in m and d.get("state") in ("solid", "liquid", "gas"):
                m["state"] = d["state"]
                carried.append("state")
            rec["as"] = "one_component"
            rec["carried"] = carried
        record[m["name"]] = rec
    for s in out["solids"]:
        if s["material"] in replace:
            s["material"] = replace[s["material"]]
    used = {s["material"] for s in out["solids"]}
    out["materials"] = [m for m in out["materials"] if m["name"] in used]
    return out, record, warns


def _load_solids(solids) -> dict:
    if isinstance(solids, Mapping):
        doc = copy.deepcopy(dict(solids))
    else:
        doc = json.loads(Path(solids).read_text())
    if doc.get("schema") != "axionmultiphysics.solids/1" or not isinstance(doc.get("regions"), Mapping):
        raise error("particle.partition_invalid", "solids is not a solids document (axionmultiphysics.solids/1 with "
                    "its 'regions')", "pass the document of axionmultiphysics.solids.from_partition or its file")
    return doc


def resolve_materials(doc, *, program=None) -> dict:
    """The solids document with its NIST compositions resolved against Geant4's own tables (``axmp-transport
    --nist``); the input is not changed.

    A material whose particle block is exactly ``{"nist": X}`` at X's density (within 1e-9, relative) becomes the NIST
    material itself: its solids name X, and the material record is dropped when no solid uses it. Any other NIST
    composition stays one component and takes X's ``mean_excitation_eV`` and ``state`` where the document gives none.
    A density more than 1 % from NIST's warns ``particle.density_differs_from_nist``."""
    doc = _load_solids(doc)
    needs = any("nist" in m["composition"] for m in doc["materials"])
    out, _, _ = _resolve(doc, require_program(program) if needs else None)
    return out


# ============================================================================================ files
def _write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _read_run(wd: Path) -> dict:
    f = wd / "run.json"
    if not f.is_file():
        raise error("events.invalid", f"no planned run in {wd} (run.json missing)",
                    "plan the run first (particles.plan or particles.run)")
    return json.loads(f.read_text())


_HELD: dict[str, list] = {}       # real path of a workdir -> [file descriptor of its lock, depth]
LOCK_FILE = ".lock"


class _Lock:
    """An exclusive lock on a workdir (``fcntl.flock`` on ``<workdir>/.lock``), held by one process at a time and
    re-entrant within it; another process meets ``events.workdir_busy``. The descriptor is not inherited by the
    programs the driver starts."""

    def __init__(self, wd: Path):
        self.wd = Path(wd)
        self.key = os.path.realpath(self.wd)

    def __enter__(self):
        if self.key in _HELD:
            _HELD[self.key][1] += 1
            return self
        self.wd.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.wd / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                os.close(fd)
                raise error("events.workdir_busy", f"another driver is working on {self.wd} (its lock "
                            f"{self.wd / LOCK_FILE} is held)", "wait for it to finish; one process works on a "
                            "workdir at a time") from None
            # a file system without locks: the workdir is used unprotected
        _HELD[self.key] = [fd, 1]
        return self

    def __exit__(self, *exc):
        held = _HELD[self.key]
        held[1] -= 1
        if held[1] == 0:
            del _HELD[self.key]
            try:
                fcntl.flock(held[0], fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(held[0])
        return False


def _check_maps(wd: Path, run: dict) -> None:
    """Refuses (``events.field_map_changed``) when a field map the run reads is no longer the map it was planned with:
    the SHA-256 of its manifest, which lists every patch array's checksum, is compared where the job files point."""
    files = run.get("job_files") or []
    job = json.loads((wd / files[0]["file"]).read_text()) if files else {"fields": []}
    for rec in run.get("fields", []):
        sha = rec.get("manifest_sha256")
        if not sha:
            continue
        named = os.path.normpath(os.path.join(wd, rec["path"]))
        where = {os.path.normpath(os.path.join(wd, "exchange", job["fields"][j]["path"]))
                 for j in rec.get("job_fields", []) if j < len(job["fields"])} or {named}
        for loc in sorted(where):
            try:
                now = _spec.manifest_sha256(loc)
            except OSError:
                now = None
            if now != sha:
                state = "is missing" if now is None else "changed"
                raise error("events.field_map_changed", f"the field map {named} (read at {loc}) {state} since the run was "
                            "planned: its chunks would mix two fields",
                            "restore the map, or plan the run again (a new workdir) with the map as it is now")


def _npy(path: Path, arr: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(arr), allow_pickle=False)


def _plain(x):
    """NaN and infinities as None (results.json is strict JSON)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ============================================================================================ plan
def _check_count(events, chunk_events):
    if events is None:
        raise error("events.count_missing", "the number of events is required (events=...)",
                    "pass events=N; a run never defaults to a number of events")
    if not _spec.positive_int(events):
        raise error("events.invalid", f"events={events!r} is not a positive integer", "pass events=N with N >= 1")
    if not _spec.positive_int(chunk_events):
        raise error("events.invalid", f"chunk_events={chunk_events!r} is not a positive integer",
                    f"pass chunk_events=N with N >= 1 (default {1000})")


def _blocks(n: int, parts: int) -> list[list[int]]:
    parts = max(1, min(parts, n))
    base, extra = divmod(n, parts)
    out, k = [], 0
    for j in range(parts):
        m = base + (1 if j < extra else 0)
        out.append(list(range(k, k + m)))
        k += m
    return out


def plan(workdir, *, solids, sources, outputs, events, fields=(), seed=1, chunk_events=1000, jobs=1,
         physics_list="FTFP_BERT", production_cut_m=PRODUCTION_CUT_M, integrator=None, outside="refuse",
         quantum_eV=QUANTUM_EV, stop=None, threads=1, program=None, debug=None) -> dict:
    """Plans a transport run into ``workdir`` and returns ``run.json``'s content. Nothing runs (see ``execute``).

    Validates everything first and refuses with a named code before writing anything: the event count
    (``events.count_missing``, ``events.invalid``), the source, the physics list, the outputs, the fields and their
    coverage of the solids, the integrator, a tally that could overflow, and the geometry (Geant4's overlap check,
    ``particle.geometry_overlap``). Resolves NIST compositions against the program's tables. Writes
    ``exchange/solids.json`` (the resolved list), one job file per block of chunks and ``exchange/commands.txt``."""
    return _plan(workdir, solids=solids, sources=sources, outputs=outputs, events=events, fields=fields, seed=seed,
                 chunk_events=chunk_events, jobs=jobs, physics_list=physics_list, production_cut_m=production_cut_m,
                 integrator=integrator, outside=outside, quantum_eV=quantum_eV, stop=stop, threads=threads,
                 program=program, debug=debug)


def _plan(workdir, *, solids, sources, outputs, events, fields, seed, chunk_events, jobs, physics_list,
          production_cut_m, integrator, outside, quantum_eV, stop, threads, program, debug, allow=(),
          validate_only=False) -> dict | None:
    from .. import solids as solids_mod

    _check_count(events, chunk_events)
    for name, v in (("jobs", jobs), ("threads", threads)):
        if not _spec.positive_int(v):
            raise ValueError(f"{name}={v!r} is not a positive integer")
    if not _spec.is_int(seed):
        raise ValueError(f"seed={seed!r} is not an integer")
    if not 0 <= int(seed) < 2 ** 64:
        raise error("events.invalid", f"seed={seed} lies outside [0, 2^64)", "a seed is one of 2^64 values: two seeds "
                    "that agree modulo 2^64 would be one run recorded as two")
    for name, v in (("production_cut_m", production_cut_m), ("quantum_eV", quantum_eV)):
        if not real(v) or v <= 0:
            raise ValueError(f"{name}={v!r} is not a positive number")
    if outside not in ("refuse", "zero"):
        raise error("particle.field_invalid", f"outside={outside!r}", "outside is \"refuse\" (the default) or \"zero\"")
    plist = _spec.physics_list_of(physics_list)
    if debug is not None and not isinstance(debug, Mapping):
        raise ValueError("debug must be an object")
    wd = Path(os.path.abspath(os.fspath(workdir)))
    if wd.exists() and (not wd.is_dir() or (not (wd / "run.json").is_file()
                                            and any(p.name not in allow + (LOCK_FILE,) for p in wd.iterdir()))):
        raise error("events.workdir_not_empty", f"the work directory {wd} holds files of something else",
                    "give a new or empty directory (or one that holds a planned run, which is planned again)")
    exe = require_program(program)
    chk = check(exe)
    table = chk.get("particles", {})
    doc = _load_solids(solids)
    with_warnings: list[dict] = []
    doc, mats_record, w = _resolve(doc, exe)
    with_warnings += w
    regions = list(doc["regions"])
    source, ekin, mass = _spec.source_of(sources, table)
    _spec.check_source_position(source, doc)
    O = _spec.outputs_of(outputs, regions, table)
    exchange = wd / "exchange"
    F = _spec.fields_of(fields, doc, regions, outside, exchange, wd)
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        integ = _spec.integrator_of(integrator, F.spacing_min, doc)
    for c in caught:
        with_warnings.append({"code": str(c.message).split(":")[0], "message": str(c.message)})
        _w.warn_explicit(c.message, c.category, c.filename, c.lineno)
    if O.tallies and chunk_events * (ekin + mass) / quantum_eV >= 2.0 ** 63:
        raise error("particle.tally_overflow", f"a chunk of {chunk_events} events of {ekin + mass:.6g} eV (kinetic "
                    f"plus rest energy) holds up to {chunk_events * (ekin + mass) / quantum_eV:.4g} quanta of "
                    f"{quantum_eV:g} eV, beyond the 2^63 of a tally's 64-bit bin",
                    "raise quantum_eV or lower chunk_events")
    stop_rec = None
    if stop is not None:
        if (not isinstance(stop, Mapping) or set(stop) != {"output", "rel_error"} or stop["output"] not in O.aggregates
                or not real(stop["rel_error"]) or stop["rel_error"] <= 0):
            raise error("particle.output_invalid", f"stop={stop!r}: needs {{\"output\": an aggregate output, "
                        "\"rel_error\": r > 0}", "stop on a mean, std, quantile, fraction or count output")
        stop_rec = {"output": stop["output"], "rel_error": float(stop["rel_error"]), "reason": None, "chunks": None,
                    "events": None}
    if validate_only:
        return None
    # Geant4's overlap check of the resolved list, before anything is written
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "solids.json"
        solids_mod.write(doc, f)
        r = _tool(exe, ["--geometry", str(f)])
    if r.returncode == 3:
        try:
            g = json.loads(r.stdout)
        except ValueError:
            g = {}
        raise error(g.get("code", "particle.geometry_overlap"), g.get("error", r.stderr.strip()[-500:]),
                    "the solids overlap in Geant4: lower chord_tol_m of the conversion, or separate the regions")
    if r.returncode != 0:
        raise error("particle.partition_invalid", f"{PROGRAM} refuses the solids list: {r.stderr.strip()[-800:]}",
                    "the list must come from axionmultiphysics.solids.from_partition")

    # the plan
    n_chunks = -(-events // chunk_events)
    blocks = _blocks(n_chunks, int(jobs))
    chunks = []
    for j, b in enumerate(blocks):
        for k in b:
            n = chunk_events if k < n_chunks - 1 else events - chunk_events * (n_chunks - 1)
            chunks.append({"index": k, "seed": chunk_seed(seed, k), "events": n, "job": j, "status": "planned",
                           "attempts": 0, "sha256": None})
    job_base = {"schema": SCHEMA_JOB, "mode": "transport", "out": "../out", "record": None, "chunks": None,
                "physics_list": plist, "production_cut_m": float(production_cut_m), "solids": "solids.json",
                "fields": F.job, "outside": outside, "integrator": {k: integ[k] for k in INTEGRATOR},
                "source": source, "quantum_eV": float(quantum_eV),
                "tallies": [{"name": n, "grid": t["grid"], "regions": t["regions"]} for n, t in O.tallies.items()],
                "hits": O.scorers}
    if debug is not None:
        job_base["debug"] = _spec.as_json(dict(debug))
    # write, holding the workdir's lock
    with _Lock(wd):
        if wd.exists() and (wd / "run.json").is_file():
            for name in ("exchange", "out", "merged", "logs"):
                shutil.rmtree(wd / name, ignore_errors=True)
            (wd / "results.json").unlink(missing_ok=True)
        exchange.mkdir(parents=True, exist_ok=True)
        (wd / "out").mkdir(exist_ok=True)
        solids_mod.write(doc, exchange / "solids.json")
        job_files, commands = [], []
        for j, b in enumerate(blocks):
            name = f"job_{j}.json"
            _write_job(exchange / name, job_base, [c for c in chunks if c["job"] == j], j)
            job_files.append({"index": j, "file": f"exchange/{name}", "chunks": b})
            commands.append(_command(exe, threads, exchange / name))
        (exchange / "commands.txt").write_text("".join(c + "\n" for c in commands))
        record = {
            "schema": SCHEMA_RUN, "beta": True, "seed": int(seed), "stage": STAGE, "instance": INSTANCE,
            "chunk_events": int(chunk_events), "events_requested": int(events), "jobs": len(blocks),
            "threads": int(threads), "chunks": chunks, "job_files": job_files, "regions": regions,
            "programs": {PROGRAM: {"path": exe, "program_version": chk.get("program_version"),
                                   "geant4_version": chk.get("geant4_version"),
                                   "geant4_tested": chk.get("geant4_tested"), "datasets": chk.get("datasets"),
                                   "commands": commands}},
            "physics_list": plist, "production_cut_m": float(production_cut_m),
            "integrator": _spec.stepper_record(integ),
            "quantum_eV": float(quantum_eV), "outside": outside, "fields": F.record, "field_maps": F.maps,
            "sources": _spec.as_json(list(sources)), "outputs": _spec.as_json(dict(outputs)), "stop": stop_rec,
            "materials_resolved": mats_record, "warnings": with_warnings, "plan_warnings": with_warnings,
            "partial": False, "missing_chunks": [], "events": 0,
            "debug": job_base.get("debug"),
        }
        _write_json(wd / "run.json", record)
        return record


def _write_job(path: Path, base: dict, chunks: list[dict], j) -> None:
    job = dict(base)
    job["record"] = f"../out/record_{j}.json"
    job["chunks"] = [{"index": c["index"], "seed": c["seed"], "events": c["events"]} for c in chunks]
    path.write_text(json.dumps(job, indent=1, allow_nan=False) + "\n")


def _command(exe: str, threads: int, job: Path) -> str:
    return " ".join(shlex.quote(a) for a in (exe, "--threads", str(threads), str(job)))


# ============================================================================================ chunks on disk
class _Chunk:
    """One chunk directory read and checked against the plan: ``state`` is "complete", "missing", "corrupt" (an
    array's checksum differs) or "mismatch" (the manifest belongs to another chunk, seed or size)."""

    def __init__(self, wd: Path, plan_chunk: dict, load: tuple = (), verify: bool = True):
        self.k = plan_chunk["index"]
        self.dir = wd / "out" / f"chunk_{self.k}"
        self.arrays: dict[str, np.ndarray] = {}
        self.manifest = None
        self.sha256 = None
        self.message = ""
        mf = self.dir / "manifest.json"
        if not mf.is_file():
            self.state = "missing"
            return
        raw = mf.read_bytes()
        try:
            man = json.loads(raw)
        except ValueError:
            self.state, self.message = "corrupt", "manifest.json is not JSON"
            return
        self.manifest = man
        want = (plan_chunk["index"], plan_chunk["seed"], plan_chunk["events"])
        got = (man.get("chunk"), man.get("chunk_seed"), man.get("events"))
        if man.get("schema") != SCHEMA_CHUNK or got != want:
            self.state = "mismatch"
            self.message = (f"chunk {self.k}: its manifest says chunk {got[0]}, seed {got[1]}, {got[2]} events; the "
                            f"plan says chunk {want[0]}, seed {want[1]}, {want[2]} events")
            return
        for name, e in man.get("arrays", {}).items():
            f = self.dir / e["file"]
            if not verify and name not in load and "*" not in load:
                continue
            if not f.is_file():
                self.state, self.message = "corrupt", f"chunk {self.k}: {e['file']} is missing"
                return
            data = f.read_bytes()
            if hashlib.sha256(data).hexdigest() != e["sha256"]:
                self.state, self.message = "corrupt", f"chunk {self.k}: {e['file']} does not match its checksum"
                return
            if name in load or "*" in load:
                self.arrays[name] = np.load(io.BytesIO(data), allow_pickle=False)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.state = "complete"


def _variables_of_chunk(c: _Chunk, O: _spec.Outputs, regions: list[str]) -> dict[str, np.ndarray]:
    a = c.arrays
    n = int(c.manifest["events"])
    out = {}
    for name, v in O.variables.items():
        if v["kind"] == "deposit":
            E = a["edep_eV"]
            idx = [regions.index(r) for r in v["regions"]]
            x = E[:, idx[0]].astype(float, copy=True)
            for i in idx[1:]:
                x = x + E[:, i]
            out[name] = x
        elif v["kind"] == "primary":
            out[name] = a["primary"][:, v["column"]].astype(float, copy=True)
        else:
            s = v["scorer"]
            cnt = a[f"hits_{s}_count"].astype(np.int64)
            first = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
            has = cnt > 0
            x = np.full(n, np.nan)
            if np.any(has):
                i = first[has]
                pos = a[f"hits_{s}_x_m"]
                of = v["of"]
                if of in ("x", "y", "z"):
                    x[has] = pos[i, "xyz".index(of)]
                elif of == "r":
                    x[has] = np.hypot(pos[i, 0], pos[i, 1])
                elif of == "phi":
                    x[has] = np.arctan2(pos[i, 1], pos[i, 0])
                elif of == "t":
                    x[has] = a[f"hits_{s}_t_s"][i]
                else:
                    x[has] = a[f"hits_{s}_ekin_eV"][i]
            out[name] = x
    return out


def _needed_arrays(O: _spec.Outputs) -> set:
    need = {"edep_eV", "primary", "outside_calls", "looping_eV", "looping_killed", "abandoned_killed", "abandoned_eV"}
    for v in O.variables.values():
        if v["kind"] == "hit":
            s = v["scorer"]
            need |= {f"hits_{s}_count", f"hits_{s}_x_m", f"hits_{s}_t_s", f"hits_{s}_ekin_eV"}
    for t in O.tallies:
        need.add(f"tally_{t}")
    for h in O.hits:
        need |= {f"hits_{h}_{k}" for k in _HIT_FIELDS + ("count", "dropped")}
    return need


def _where(W, vals: dict, n: int) -> np.ndarray:
    if W is None:
        return np.ones(n, bool)
    if "any" in W:
        m = np.zeros(n, bool)
        for w in W["any"]:
            m |= _where(w, vals, n)
        return m
    m = np.ones(n, bool)
    for var, cond in W.items():
        x = vals[var]
        defined = ~np.isnan(x)
        with np.errstate(invalid="ignore"):
            for op, t in cond.items():
                if op == "defined":
                    m &= defined if t else ~defined
                elif op == "gt":
                    m &= defined & (x > t)
                elif op == "ge":
                    m &= defined & (x >= t)
                elif op == "lt":
                    m &= defined & (x < t)
                else:
                    m &= defined & (x <= t)
    return m


def _aggregate(name: str, a: dict, vals: dict, batch: np.ndarray, n_chunks: int, O: _spec.Outputs) -> Measured:
    N = len(batch)
    sel = _where(a.get("where"), vals, N)
    kind = a["kind"]
    if kind == "fraction":
        return estimate_fraction(int(sel.sum()), N) if N else Measured(float("nan"), events=0, units="1",
                                                                       reason="no events")
    if kind == "count":
        if N == 0:
            return Measured(float("nan"), events=0, events_defined=0, units="1", reason="no events")
        return estimate_count(int(sel.sum()), events=N)
    x = vals[a["of"]]
    keep = sel & ~np.isnan(x)
    units = a["units"]
    if kind == "mean":
        y = np.where(keep, x, np.nan)
        return _mean(y, N, units)
    xs = x[keep]
    n = len(xs)
    stat = (lambda z: float(np.std(z, ddof=1))) if kind == "std" else (lambda z: float(np.quantile(z, a["q"])))
    need = 2 if kind == "std" else 1
    if n < need:
        return Measured(float("nan"), error="batch_means", events=N, events_defined=n, units=units,
                        reason="too few events where the variable is defined and the selection holds")
    v = stat(xs)
    if n_chunks < BATCHES:
        return Measured(v, error="batch_means", events=N, events_defined=n, units=units,
                        reason=f"fewer than {BATCHES} chunks for batch means")
    vb, nb = [], []
    for b in range(BATCHES):
        z = x[keep & (batch == b)]
        if len(z) < need:
            return Measured(v, error="batch_means", events=N, events_defined=n, units=units,
                            reason=f"batch {b} has too few selected events")
        vb.append(stat(z))
        nb.append(int(np.sum(batch == b)))
    _, s = batch_means(vb, nb)
    return Measured(v, stat_error=s, error="batch_means", events=N, events_defined=n, units=units)


def _outputs_of_run(run: dict):
    return _spec.outputs_of(run["outputs"], run["regions"], _spec.AnyParticle())


# ============================================================================================ stop rule
class _NotComplete(Exception):
    """A chunk of the prefix fails its checksums: the prefix ends before it."""


def _stop_K(stop: dict, O: _spec.Outputs, run: dict, wd: Path, prefix: list, cache: dict, start: int = BATCHES):
    """The smallest K >= 20 among the complete prefix ``prefix`` (chunk indices 0..len-1) at which the stop output's
    relative error meets its target, or None."""
    a = O.aggregates[stop["output"]]
    for K in range(max(start, BATCHES), len(prefix) + 1):
        try:
            vals, batch = _prefix_values(O, run, wd, prefix[:K], cache)
        except _NotComplete:
            return None
        m = _aggregate(stop["output"], a, vals, batch, K, O)
        if m.stat_error is not None and float(m) != 0.0 and m.stat_error / abs(float(m)) <= stop["rel_error"]:
            return K
    return None


def _prefix_values(O, run, wd, ks, cache):
    regions = run["regions"]
    parts, batch = [], []
    for k in ks:
        if k not in cache:
            c = _Chunk(wd, run["chunks"][k], load=tuple(_needed_arrays(O)))
            if c.state != "complete":
                raise _NotComplete(k)
            cache[k] = (_variables_of_chunk(c, O, regions), int(c.manifest["events"]))
        v, n = cache[k]
        parts.append(v)
        batch.append(np.full(n, k % BATCHES, dtype=np.int64))
    vals = {name: np.concatenate([p[name] for p in parts]) if parts else np.zeros(0) for name in O.variables}
    return vals, np.concatenate(batch) if batch else np.zeros(0, np.int64)


# ============================================================================================ execute
def execute(workdir, *, workers=1) -> None:
    """Runs the planned job files with ``axmp-transport``, ``workers`` processes at a time. A process that fails has
    its incomplete chunks rerun once, in one new process; a chunk still incomplete is ``failed``. With a stop rule the
    run ends at the first complete prefix of chunks that meets it (whatever the number of workers). Chunks are
    checked against their checksums as they complete; ``run.json`` is updated after every process."""
    wd = Path(os.path.abspath(os.fspath(workdir)))
    _read_run(wd)
    if not _spec.positive_int(workers):
        raise ValueError(f"workers={workers!r} is not a positive integer")
    with _Lock(wd):
        _execute(wd, workers)


def _execute(wd: Path, workers: int) -> None:
    run = _read_run(wd)
    _check_maps(wd, run)
    exe = run["programs"][PROGRAM]["path"]
    if not _executable(exe):
        raise error("build.requires_program", f"the run's program {exe} is not an executable file any more",
                    "rebuild it, or plan the run again with program=")
    O = _outputs_of_run(run)
    chunks = run["chunks"]
    # a chunk is complete once its manifest (written last, then the directory renamed) names it; the arrays are
    # checked against their checksums by the merge, which reads them anyway
    complete = {c["index"]: _Chunk(wd, c, verify=False).state == "complete" for c in chunks}
    for c in chunks:
        if complete[c["index"]]:
            c["status"] = "complete"
    stop = run.get("stop")
    cache: dict = {}
    K_found = None
    checked = BATCHES

    def prefix():
        p = []
        for c in chunks:
            if not complete[c["index"]]:
                break
            p.append(c["index"])
        return p

    def evaluate_stop():
        nonlocal K_found, checked
        if stop is None or K_found is not None:
            return
        p = prefix()
        if len(p) >= checked:
            K = _stop_K(stop, O, run, wd, p, cache, start=checked)
            if K is not None:
                K_found = K
            else:
                checked = len(p) + 1

    evaluate_stop()
    queue = []
    exchange = wd / "exchange"
    for jf in run["job_files"]:
        inc = [k for k in jf["chunks"] if not complete[k]]
        if not inc:
            continue
        file = wd / jf["file"]
        if inc != jf["chunks"]:
            file = _variant(wd, jf, inc, "rest")
        queue.append({"job": jf["index"], "file": file, "chunks": inc, "retry": False})
    logs = wd / "logs"
    logs.mkdir(exist_ok=True)
    running = []
    threads = run["threads"]
    t0 = time.time()
    while (queue and K_found is None) or running:
        while queue and K_found is None and len(running) < workers:
            task = queue.pop(0)
            for k in task["chunks"]:
                chunks[k]["attempts"] += 1
            log = open(logs / f"{Path(task['file']).stem}_a{chunks[task['chunks'][0]]['attempts']}.log", "w")
            p = subprocess.Popen([exe, "--threads", str(threads), str(task["file"])], stdout=log,
                                 stderr=subprocess.STDOUT, cwd=exchange)
            running.append((p, task, log))
        finished = []
        while not finished:
            for item in running:
                if item[0].poll() is not None:
                    finished.append(item)
            if not finished:
                time.sleep(0.02)
        for item in finished:
            running.remove(item)
            p, task, log = item
            log.close()
            with open(log.name, "a") as f:
                f.write(f"\n[particles] exit status {p.returncode}\n")
            inc = []
            for k in task["chunks"]:
                c = _Chunk(wd, chunks[k], verify=False)
                complete[k] = c.state == "complete"
                if complete[k]:
                    chunks[k]["status"], chunks[k]["sha256"] = "complete", c.sha256
                else:
                    inc.append(k)
            if inc:
                if not task["retry"] and K_found is None:
                    jf = run["job_files"][task["job"]]
                    queue.insert(0, {"job": task["job"], "file": _variant(wd, jf, inc, "retry"), "chunks": inc,
                                     "retry": True})
                else:
                    for k in inc:
                        chunks[k]["status"] = "failed"
            evaluate_stop()
        if K_found is not None:
            for p, task, log in running:
                p.terminate()
            for p, task, log in running:
                p.wait()
                log.close()
                for k in task["chunks"]:
                    c = _Chunk(wd, chunks[k], verify=False)
                    complete[k] = c.state == "complete"
                    if complete[k]:
                        chunks[k]["status"], chunks[k]["sha256"] = "complete", c.sha256
            running = []
            queue = []
        _write_json(wd / "run.json", run)
    if stop is not None:
        _apply_stop(run, K_found)
    run.setdefault("timing", {})["execute_s"] = time.time() - t0
    _write_json(wd / "run.json", run)


def _apply_stop(run: dict, K):
    stop = run["stop"]
    if K is None:
        stop.update(reason="events", chunks=None, events=None)
        for c in run["chunks"]:
            if c["status"] == "beyond_stop":
                c["status"] = "complete"
        return
    stop.update(reason="rel_error", chunks=K, events=sum(c["events"] for c in run["chunks"][:K]))
    for c in run["chunks"][K:]:
        if c["status"] == "complete":
            c["status"] = "beyond_stop"


def _variant(wd: Path, jf: dict, chunks: list[int], what: str) -> Path:
    base = json.loads((wd / jf["file"]).read_text())
    keep = set(chunks)
    base["chunks"] = [c for c in base["chunks"] if c["index"] in keep]
    i = 0
    while True:
        f = wd / "exchange" / f"job_{jf['index']}_{what}{i}.json"
        if not f.exists():
            break
        i += 1
    f.write_text(json.dumps(base, indent=1, allow_nan=False) + "\n")
    return f


# ============================================================================================ merge
def merge(workdir, *, allow_failed_chunks=False) -> "ParticleResults":
    """Merges the complete chunks in (chunk, event) order and estimates the outputs; writes ``results.json`` and
    ``merged/``. Without ``allow_failed_chunks`` the sample is the longest complete prefix of the planned chunks;
    with it, every complete chunk. Fewer events than requested make a partial result (``events.partial``). A chunk
    whose manifest disagrees with the plan is refused (``events.chunk_mismatch``); one whose arrays do not match
    their checksums is missing."""
    wd = Path(os.path.abspath(os.fspath(workdir)))
    _read_run(wd)
    with _Lock(wd):
        return _merge(wd, allow_failed_chunks)


def _merge(wd: Path, allow_failed_chunks: bool) -> "ParticleResults":
    run = _read_run(wd)
    O = _outputs_of_run(run)
    regions = run["regions"]
    need = tuple(_needed_arrays(O))
    loaded: dict[int, _Chunk] = {}
    states = {}
    for c in run["chunks"]:
        ch = _Chunk(wd, c, load=need)
        if ch.state == "mismatch":
            raise error("events.chunk_mismatch", ch.message, "the chunk directory belongs to another plan: move it "
                        "away (or plan the run again) and execute")
        states[c["index"]] = ch.state
        if ch.state == "complete":
            loaded[c["index"]] = ch
            c["sha256"] = ch.sha256
            if c["status"] in ("planned", "failed"):
                c["status"] = "complete"
        elif c["status"] in ("complete", "beyond_stop"):
            c["status"] = "failed"
    prefix = []
    for c in run["chunks"]:
        if states[c["index"]] != "complete":
            break
        prefix.append(c["index"])
    stop = run.get("stop")
    K = None
    if stop is not None:
        K = _stop_K(stop, O, run, wd, prefix, {k: (_variables_of_chunk(loaded[k], O, regions),
                                                  int(loaded[k].manifest["events"])) for k in prefix})
        _apply_stop(run, K)
    if K is not None:
        merged = prefix[:K]
        planned = [c["index"] for c in run["chunks"][:K]]
    else:
        merged = sorted(loaded) if allow_failed_chunks else prefix
        planned = [c["index"] for c in run["chunks"]]
    merged_set = set(merged)
    missing = [k for k in planned if k not in merged_set]
    res = _estimate(wd, run, O, [loaded[k] for k in merged], missing)
    return res


def _estimate(wd: Path, run: dict, O: _spec.Outputs, chunks: list[_Chunk], missing: list[int]) -> "ParticleResults":
    regions = run["regions"]
    N = sum(int(c.manifest["events"]) for c in chunks)
    requested = run["events_requested"]
    stopped = run.get("stop") is not None and run["stop"].get("reason") == "rel_error"
    partial = bool(missing) or (N < requested and not stopped)
    per = [_variables_of_chunk(c, O, regions) for c in chunks]
    vals = {name: (np.concatenate([p[name] for p in per]) if per else np.zeros(0)) for name in O.variables}
    batch = np.concatenate([np.full(int(c.manifest["events"]), c.k % BATCHES, np.int64) for c in chunks]) \
        if chunks else np.zeros(0, np.int64)
    warnings_out = [w for w in run.get("plan_warnings", [])]
    if partial:
        warnings_out.append(warn("events.partial", f"{N} of {run['events_requested']} events merged; missing chunks "
                                 f"{missing}", stacklevel=4))
    n_chunks = len(chunks)
    needs_batches = any(a["kind"] in ("std", "quantile") for a in O.aggregates.values()) or bool(O.tallies)
    if needs_batches and n_chunks < BATCHES:
        warnings_out.append(warn("stat.too_few_chunks", f"{n_chunks} chunks merged, fewer than the {BATCHES} that "
                                 "batch means need: std, quantile and tally outputs carry no statistical error",
                                 stacklevel=4))
    outputs = {}
    measured = {}
    for name, v in O.variables.items():
        outputs[name] = {"kind": v["kind"], "units": v["units"]}
    for name, a in O.aggregates.items():
        m = _aggregate(name, a, vals, batch, n_chunks, O)
        measured[name] = m
        outputs[name] = dict(m.as_dict(), kind=a["kind"])
    for name, h in O.histograms.items():
        x = vals[h["of"]]
        sel = _where(h.get("where"), vals, N)
        undefined = int(np.sum(sel & np.isnan(x)))
        xs = x[sel & ~np.isnan(x)]
        edges = np.linspace(h["range"][0], h["range"][1], h["bins"] + 1)
        idx = np.searchsorted(edges, xs, side="right") - 1
        inside = (idx >= 0) & (idx < h["bins"])
        counts = np.bincount(idx[inside], minlength=h["bins"]).astype(np.int64)
        outputs[name] = {"kind": "histogram", "edges": edges.tolist(), "counts": counts.tolist(),
                         "errors": np.sqrt(counts).tolist(), "underflow": int(np.sum(xs < edges[0])),
                         "overflow": int(np.sum(xs >= edges[-1])), "undefined": undefined, "units": h["units"],
                         "events": N, "mc": True, "beta": True}
    merged_dir = wd / "merged"
    if O.tallies:
        merged_dir.mkdir(exist_ok=True)
    for name, t in O.tallies.items():
        bins = tuple(t["grid"]["bins"])
        lo = np.zeros((BATCHES, *bins), np.uint64)
        hi = np.zeros((BATCHES, *bins), np.int64)
        ne = np.zeros(BATCHES, np.int64)
        for c in chunks:
            T = c.arrays[f"tally_{name}"].astype(np.int64).reshape(bins)
            b = c.k % BATCHES
            s = lo[b] + T.view(np.uint64)                # wraps modulo 2^64
            carry = (s < lo[b]).astype(np.int64)
            hi[b] += carry - (T < 0).astype(np.int64)
            lo[b] = s
            ne[b] += int(c.manifest["events"])
        _npy(merged_dir / f"tally_{name}_lo.npy", lo.astype("<u8"))
        _npy(merged_dir / f"tally_{name}_hi.npy", hi.astype("<i8"))
        _npy(merged_dir / f"tally_{name}_events.npy", ne.astype("<i8"))
        outputs[name] = {"kind": "tally", "lo": f"merged/tally_{name}_lo.npy", "hi": f"merged/tally_{name}_hi.npy",
                         "batch_events": f"merged/tally_{name}_events.npy", "grid": t["grid"],
                         "regions": t["regions"], "quantum_eV": run["quantum_eV"], "events": N, "units": "eV",
                         "mc": True, "beta": True}
    hits_cache = {}
    for name, h in O.hits.items():
        d = _hits_of(chunks, name, h)
        hits_cache[name] = d
        outputs[name] = {"kind": "hits", "records": int(len(d["event"])), "dropped": d["_dropped"],
                         "events_with_records": d["_with"], "events_without_records": N - d["_with"],
                         "keep_events": h["keep_events"], "max_per_event": h["max_per_event"]}
    # counters
    outside_calls = int(sum(int(np.sum(c.arrays["outside_calls"])) for c in chunks))
    looping_eV = float(sum(float(np.sum(c.arrays["looping_eV"])) for c in chunks))
    looping_n = int(sum(int(np.sum(c.arrays["looping_killed"])) for c in chunks))
    deposit = float(sum(float(np.sum(c.arrays["edep_eV"])) for c in chunks))
    abandoned_n = int(sum(int(np.sum(c.arrays.get("abandoned_killed", 0))) for c in chunks))
    abandoned_eV = float(sum(float(np.sum(c.arrays.get("abandoned_eV", 0.0))) for c in chunks))
    if run["outside"] == "zero":
        ratios = {p: m.get("boundary_ratio") for p, m in run.get("field_maps", {}).items()}
        warnings_out.append(warn("particle.field_outside_map", f"{outside_calls} field calls fell outside the maps "
                                 "and returned B = 0 (outside=\"zero\"); unserved regions are field-free; "
                                 f"boundary_ratio of the maps: {ratios}", stacklevel=4))
    if abandoned_n:
        warnings_out.append(warn("particle.track_abandoned", f"{abandoned_n} tracks that Geant4's navigator abandoned as "
                                 f"stuck (GeomNav0003) were ended, carrying {abandoned_eV:.6g} eV (arrays "
                                 "abandoned_killed and abandoned_eV); such a track usually sits at an interface where "
                                 "the field reverses", stacklevel=4))
    if looping_eV > 1e-6 * deposit:
        warnings_out.append(warn("particle.looping_killed", f"{looping_n} looping tracks were killed in the field, "
                                 f"carrying {looping_eV:.6g} eV, more than 1e-6 of the {deposit:.6g} eV deposited",
                                 stacklevel=4))
    S = run["seed"]
    results = {"schema": SCHEMA_RESULTS, "beta": True,
               "mc": (f"Monte Carlo (beta): {N} events, seed {S}; statistical errors only; physics-list dependence "
                      "not estimated"),
               "seed": S, "events": N, "events_requested": requested, "partial": partial, "missing_chunks": missing,
               "merged_chunks": [c.k for c in chunks],
               "counters": {"outside_calls": outside_calls, "looping_killed": looping_n, "looping_eV": looping_eV,
                            "abandoned_killed": abandoned_n, "abandoned_eV": abandoned_eV},
               "outputs": outputs, "warnings": warnings_out}
    _write_json(wd / "results.json", _json_clean(results))
    run.update(partial=partial, missing_chunks=missing, events=N, warnings=warnings_out)
    _write_json(wd / "run.json", run)
    r = ParticleResults(wd, run, results, O)
    r._vals = vals
    r._index = (np.concatenate([np.full(int(c.manifest["events"]), c.k, np.int64) for c in chunks])
                if chunks else np.zeros(0, np.int64),
                np.concatenate([np.arange(int(c.manifest["events"]), dtype=np.int64) for c in chunks])
                if chunks else np.zeros(0, np.int64))
    r._hits = hits_cache
    return r


def _json_clean(x):
    if isinstance(x, dict):
        return {k: _json_clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_clean(v) for v in x]
    return _plain(x)


_HIT_FIELDS = ("event", "track", "parent", "pdg", "region", "x_m", "p_eV_c", "t_s", "ekin_eV")


def _hits_of(chunks: list[_Chunk], name: str, h: dict) -> dict:
    """The records of a hits output over the merged chunks, in (chunk, event, step) order, with keep_events applied."""
    parts = {k: [] for k in _HIT_FIELDS}
    chunk_col = []
    dropped, with_rec, seen = 0, 0, 0
    keep = h.get("keep_events")
    for c in chunks:
        a = c.arrays
        n = int(c.manifest["events"])
        cnt = a[f"hits_{name}_count"].astype(np.int64)
        dropped += int(np.sum(a[f"hits_{name}_dropped"]))
        ev = a[f"hits_{name}_event"].astype(np.int64)
        if keep is None:
            sel = np.ones(len(ev), bool)
            with_rec += int(np.sum(cnt > 0))
        else:
            room = max(0, keep - seen)
            sel = ev < room
            with_rec += int(np.sum(cnt[:min(room, n)] > 0))
        seen += n
        for k in _HIT_FIELDS:
            parts[k].append(a[f"hits_{name}_{k}"][sel])
        chunk_col.append(np.full(int(sel.sum()), c.k, np.int64))
    out = {"chunk": np.concatenate(chunk_col) if chunk_col else np.zeros(0, np.int64)}
    for k in _HIT_FIELDS:
        if parts[k]:
            out[k] = np.concatenate(parts[k])
        else:
            out[k] = np.zeros((0, 3) if k in ("x_m", "p_eV_c") else 0)
    out["_dropped"], out["_with"] = dropped, with_rec
    return out


# ============================================================================================ results
class ParticleResults:
    """The merged result of a transport run (beta): ``res[name]`` (a ``Measured`` for an aggregate output),
    ``histogram(name)``, ``tally(name)``, ``events(names=None)``, ``hits(name)``; ``record`` (run.json), ``outputs``
    (results.json "outputs"), ``n_events``, ``events_requested``, ``partial``, ``missing_chunks``, ``warnings`` and
    ``workdir``. Statistical errors only: the physics-list dependence is not estimated."""

    def __init__(self, workdir: Path, record: dict, results: dict, O: _spec.Outputs):
        self.workdir = Path(workdir)
        self.record = record
        self.results = results
        self.outputs = results["outputs"]
        self.n_events = results["events"]
        self.events_requested = results["events_requested"]
        self.partial = results["partial"]
        self.missing_chunks = list(results["missing_chunks"])
        self.warnings = list(results["warnings"])
        self._O = O
        self._vals = None
        self._index = None
        self._hits: dict = {}

    def __repr__(self) -> str:
        return f"<ParticleResults {self.results['mc']}{' (partial)' if self.partial else ''}>"

    def __getitem__(self, name: str) -> Measured:
        o = self.outputs.get(name)
        if o is None or o.get("kind") not in _spec.AGGREGATE_KINDS:
            raise KeyError(f"{name!r} is not an aggregate output (mean, std, quantile, fraction, count) of this run")
        return Measured.from_dict(o)

    def _chunks(self, load) -> list[_Chunk]:
        out = []
        for k in self.results["merged_chunks"]:
            c = _Chunk(self.workdir, self.record["chunks"][k], load=load)
            if c.state != "complete":
                raise error("events.chunk_mismatch", f"chunk {k} changed on disk since the merge ({c.state}: "
                            f"{c.message})", "merge the run again")
            out.append(c)
        return out

    def _load_events(self):
        if self._vals is not None:
            return
        chunks = self._chunks(tuple(_needed_arrays(self._O)))
        per = [_variables_of_chunk(c, self._O, self.record["regions"]) for c in chunks]
        self._vals = {n: (np.concatenate([p[n] for p in per]) if per else np.zeros(0)) for n in self._O.variables}
        self._index = (np.concatenate([np.full(int(c.manifest["events"]), c.k, np.int64) for c in chunks])
                       if chunks else np.zeros(0, np.int64),
                       np.concatenate([np.arange(int(c.manifest["events"]), dtype=np.int64) for c in chunks])
                       if chunks else np.zeros(0, np.int64))

    def events(self, names=None) -> dict:
        """The per-event variables of the merged events, in (chunk, event) order: ``chunk``, ``event`` and each
        variable (NaN where undefined)."""
        self._load_events()
        if names is None:
            names = list(self._O.variables)
        out = {"chunk": self._index[0], "event": self._index[1]}
        for n in names:
            if n not in self._vals:
                raise KeyError(f"{n!r} is not a per-event variable (deposit, hit, primary) of this run")
            out[n] = self._vals[n]
        return out

    def hits(self, name: str) -> dict:
        """The records of a hits output: ``chunk``, ``event``, ``track``, ``parent``, ``pdg``, ``region`` (index in
        the solids' regions), ``x_m`` (n, 3), ``p_eV_c`` (n, 3), ``t_s``, ``ekin_eV``."""
        if name not in self._O.hits:
            raise KeyError(f"{name!r} is not a hits output of this run")
        if name not in self._hits:
            keys = [f"hits_{name}_{k}" for k in _HIT_FIELDS] + [f"hits_{name}_count", f"hits_{name}_dropped"]
            self._hits[name] = _hits_of(self._chunks(tuple(keys)), name, self._O.hits[name])
        return {k: v for k, v in self._hits[name].items() if not k.startswith("_")}

    def histogram(self, name: str) -> dict:
        o = self.outputs.get(name)
        if o is None or o.get("kind") != "histogram":
            raise KeyError(f"{name!r} is not a histogram output of this run")
        return {"edges": np.asarray(o["edges"], float), "counts": np.asarray(o["counts"], np.int64),
                "errors": np.asarray(o["errors"], float), "underflow": o["underflow"], "overflow": o["overflow"],
                "undefined": o["undefined"]}

    def tally(self, name: str) -> dict:
        """The L1 tally: ``edges`` (per axis), ``edep_eV`` (mean deposit per event per bin), ``stat_error`` (batch
        means; None below 20 chunks), ``q_dep_J_m3``, ``volume_m3``, ``events``, ``quantum_eV``, ``lo`` and ``hi``
        (the exact 128-bit batch sums hi x 2^64 + lo, quanta) and ``batch_events``."""
        o = self.outputs.get(name)
        if o is None or o.get("kind") != "tally":
            raise KeyError(f"{name!r} is not a tally output of this run")
        lo = np.load(self.workdir / o["lo"])
        hi = np.load(self.workdir / o["hi"])
        nb = np.load(self.workdir / o["batch_events"])
        q = float(o["quantum_eV"])
        N = int(nb.sum())
        g = o["grid"]
        shape = lo.shape[1:]
        exact = (hi.astype(object) * (1 << 64)) + lo.astype(object)          # Python integers, exact
        total = exact.sum(axis=0)
        flat_total = np.array([float(int(x)) for x in np.ravel(total)], dtype=float).reshape(shape)
        edep = flat_total * q / N if N else np.full(shape, np.nan)
        n_chunks = len(self.results["merged_chunks"])
        stat = None
        if n_chunks >= BATCHES and np.all(nb > 0):
            vb = np.array([[float(int(x)) for x in np.ravel(exact[b])] for b in range(BATCHES)], dtype=float)
            vb = vb * q / nb[:, None].astype(float)
            w = nb.astype(float) / nb.sum()
            vbar = np.sum(w[:, None] * vb, axis=0)
            stat = np.sqrt(BATCHES / (BATCHES - 1) * np.sum((w[:, None] ** 2) * (vb - vbar) ** 2, axis=0)).reshape(shape)
        if g["type"] == "cylindrical":
            er = np.linspace(g["r_m"][0], g["r_m"][1], g["bins"][0] + 1)
            ez = np.linspace(g["z_m"][0], g["z_m"][1], g["bins"][1] + 1)
            vol = math.pi * np.outer(np.diff(er * er), np.diff(ez))
            edges = [er, ez]
        else:
            edges = [np.linspace(g[a][0], g[a][1], g["bins"][i] + 1) for i, a in enumerate(("x_m", "y_m", "z_m"))]
            d = [np.diff(e) for e in edges]
            vol = d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]
        return {"edges": edges, "edep_eV": edep, "stat_error": stat, "q_dep_J_m3": edep * E_CHARGE / vol,
                "volume_m3": vol, "events": N, "quantum_eV": q, "lo": lo, "hi": hi, "batch_events": nb}


def load(workdir) -> ParticleResults:
    """The merged result of a run from its ``results.json`` and chunks (nothing is run or merged again)."""
    wd = Path(os.path.abspath(os.fspath(workdir)))
    run = _read_run(wd)
    f = wd / "results.json"
    if not f.is_file():
        raise error("events.invalid", f"{wd} has no results.json", "merge the run first (particles.merge)")
    return ParticleResults(wd, run, json.loads(f.read_text()), _outputs_of_run(run))


# ============================================================================================ run, extend, transport
def run(workdir, *, workers=None, allow_failed_chunks=False, **plan_kwargs) -> ParticleResults:
    """``plan`` + ``execute`` + ``merge``; ``workers`` defaults to the number of job files."""
    if workers is not None and not _spec.positive_int(workers):
        raise ValueError(f"workers={workers!r} is not a positive integer")
    rec = plan(workdir, **plan_kwargs)
    with _Lock(Path(os.path.abspath(os.fspath(workdir)))):
        execute(workdir, workers=rec["jobs"] if workers is None else workers)
        return merge(workdir, allow_failed_chunks=allow_failed_chunks)


def extend(workdir, *, events, workers=1, chunk_events=None, allow_failed_chunks=False) -> ParticleResults:
    """Extends a run to ``events`` (at least the events requested so far): complete full chunks are reused (their
    checksums checked), a short last chunk is run again in full and replaced, new chunks are run; the merged result
    equals, bit for bit, a direct run of ``events``."""
    wd = Path(os.path.abspath(os.fspath(workdir)))
    _read_run(wd)
    if not _spec.positive_int(workers):
        raise ValueError(f"workers={workers!r} is not a positive integer")
    with _Lock(wd):
        return _extend(wd, events, workers, chunk_events, allow_failed_chunks)


def _extend(wd: Path, events, workers, chunk_events, allow_failed_chunks) -> ParticleResults:
    run_rec = _read_run(wd)
    ce = run_rec["chunk_events"]
    if chunk_events is not None and chunk_events != ce:
        raise error("events.chunk_events_mismatch", f"chunk_events={chunk_events!r}, but the run was planned with "
                    f"{ce}", "an extension keeps the plan's chunk size (a chunk's events depend on it)")
    if not _spec.positive_int(events) or events < run_rec["events_requested"]:
        raise error("events.invalid", f"events={events!r}: an extension needs an integer of at least the "
                    f"{run_rec['events_requested']} events requested so far", "pass a larger events=")
    _check_maps(wd, run_rec)
    n_chunks = -(-events // ce)
    old = {c["index"]: c for c in run_rec["chunks"]}
    chunks, rerun = [], []
    for k in range(n_chunks):
        n = ce if k < n_chunks - 1 else events - ce * (n_chunks - 1)
        c = dict(old[k]) if k in old else {"index": k, "seed": chunk_seed(run_rec["seed"], k), "events": n,
                                            "job": None, "status": "planned", "attempts": 0, "sha256": None}
        if c["events"] != n:
            shutil.rmtree(wd / "out" / f"chunk_{k}", ignore_errors=True)
            c.update(events=n, status="planned", attempts=0, sha256=None)
        if c["status"] != "complete" or _Chunk(wd, c).state != "complete":
            if c["status"] == "beyond_stop" and _Chunk(wd, c).state == "complete":
                c["status"] = "complete"
            else:
                c["status"] = "planned"
                rerun.append(k)
        chunks.append(c)
    base = json.loads((wd / run_rec["job_files"][0]["file"]).read_text())
    exe = run_rec["programs"][PROGRAM]["path"]
    j0 = len(run_rec["job_files"])
    for b in (_blocks(len(rerun), max(1, int(workers))) if rerun else []):
        j = j0
        j0 += 1
        ks = [rerun[i] for i in b]
        for k in ks:
            chunks[k]["job"] = j
        name = f"job_{j}.json"
        _write_job(wd / "exchange" / name, base, [chunks[k] for k in ks], j)
        run_rec["job_files"].append({"index": j, "file": f"exchange/{name}", "chunks": ks})
        cmd = _command(exe, run_rec["threads"], wd / "exchange" / name)
        run_rec["programs"][PROGRAM]["commands"].append(cmd)
        with open(wd / "exchange" / "commands.txt", "a") as f:
            f.write(cmd + "\n")
    for jf in run_rec["job_files"]:
        jf["chunks"] = [k for k in jf["chunks"] if k < n_chunks and chunks[k]["job"] == jf["index"]]
    run_rec["chunks"] = chunks
    run_rec["events_requested"] = int(events)
    if run_rec.get("stop") is not None:
        run_rec["stop"].update(reason=None, chunks=None, events=None)
    _write_json(wd / "run.json", run_rec)
    execute(wd, workers=workers)
    return merge(wd, allow_failed_chunks=allow_failed_chunks)


def transport(design, results, *, sources, outputs, events, composition=None, solids=None, field_map=None,
              map_tol=1e-4, track_volumes=None, workdir=None, **run_kwargs) -> ParticleResults:
    """Transports particles through a solved design (beta): its regions as solids, its field as a map.

    ``solids``: a solids document (or its file) of the design's regions, built from an exported partition (the
    document ``Design.export_gdml`` returns, or ``solids.from_partition`` of the partition ``axmp-mesher
    --export-regions`` writes). This version does not export the partition by itself
    (``capabilities()["partition_export"]`` is false), so without ``solids`` the call raises
    ``particle.partition_unavailable``. ``field_map``: a map directory; without it the map is exported
    from ``results`` (``Results.export_field_map``, ``tol=map_tol``, ``track_volumes``) into ``<workdir>/fieldmap``.
    Every patch of the map serves its record's regions. ``composition`` maps the design's material names to particle
    blocks (used when the partition is exported). Other keywords go to ``run``."""
    doc = getattr(design, "doc", design) or {}
    sym = doc.get("symmetry", "axisymmetric") if isinstance(doc, Mapping) else "axisymmetric"
    geometry = "3d" if sym == "3d" else "axisymmetric"
    if solids is None:
        # PARTITION_EXPORT says whether this function may export the partition itself; in this version it never does
        raise error("particle.partition_unavailable",
                    f"transport() takes the regions of {'a 3D' if geometry == '3d' else 'an axisymmetric'} design as "
                    f"solids= and does not export its partition by itself",
                    "pass solids= built from an exported partition: the solids document that "
                    "Design.export_gdml(path, composition) returns, or axionmultiphysics.solids.from_partition of the "
                    "partition that axmp-mesher --export-regions writes")
    wd = Path(os.path.abspath(os.fspath(workdir))) if workdir is not None else \
        Path(tempfile.mkdtemp(prefix="axmp-particles-"))
    kw = dict(run_kwargs)
    workers = kw.pop("workers", None)
    allow_failed = kw.pop("allow_failed_chunks", False)
    if workers is not None and not _spec.positive_int(workers):
        raise ValueError(f"workers={workers!r} is not a positive integer")
    if field_map is not None and "fields" in kw:
        raise error("particle.field_ambiguous", "transport() was given both fields= and field_map=",
                    "pass the map as field_map= (every patch serves its regions) or the fields as fields=, not both")
    export = field_map is None and "fields" not in kw
    if export and results is None:
        raise ValueError("transport needs results (a solved design) or field_map=")
    args = dict(fields=[] if export else [{"type": "map", "path": os.fspath(field_map)}], seed=1, chunk_events=1000,
                jobs=1, physics_list="FTFP_BERT", production_cut_m=PRODUCTION_CUT_M, integrator=None,
                outside="refuse", quantum_eV=QUANTUM_EV, stop=None, threads=1, program=None, debug=None)
    args.update(kw)
    # the call is validated before anything is written or exported (the map's own checks follow its export)
    _plan(wd, solids=solids, sources=sources, outputs=outputs, events=events, allow=("fieldmap",),
          validate_only=True, **args)
    with _Lock(wd):
        if export:
            results.export_field_map(wd / "fieldmap", tol=map_tol, track_volumes=track_volumes)
            args["fields"] = [{"type": "map", "path": os.fspath(wd / "fieldmap")}]
        rec = _plan(wd, solids=solids, sources=sources, outputs=outputs, events=events, allow=("fieldmap",), **args)
        execute(wd, workers=rec["jobs"] if workers is None else workers)
        return merge(wd, allow_failed_chunks=allow_failed)
