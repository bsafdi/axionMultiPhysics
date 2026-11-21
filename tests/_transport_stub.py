"""A stand-in for ``axmp-transport`` in the Geant4-free tests of the driver (``tests/test_particles_driver.py``).

It implements the program's documented file contract (docs/guides/particles.md): ``--check``, ``--geometry``,
``--nist`` (for ``G4_Galactic`` and ``G4_WATER`` only) and the ``transport`` mode, writing chunk directories
in the format ``axionmultiphysics.transport/1``. Its "physics" is a deterministic function of the event's seed words,
so every reproducibility property the driver must keep (order, grouping, extension, prefixes) is visible bit for bit,
and runs in milliseconds. It is not a model of Geant4.

Per event e of a chunk with seed s: u = the first word pair of ``event_words(s, e)`` as a double in [0, 1) is
``first_flat``; the primary deposits u x E_kin in the first region of the job's solids and the rest leaves the
domain (``exit_eV``); every tally adds llround(u E_kin / quantum) to bin (e mod bins) of its grid; every hits scorer
writes floor(4 u) + 1 records (capped by ``max_per_event``, the rest counted as dropped), record j at x = (u, e, j)
m, t = j ns, kinetic energy E_kin (1 - j / 10), track j + 1, pdg of the primary; ``outside_calls`` is 1 per event
when the job's ``outside`` is "zero". ``debug.fail_chunks`` makes the process exit 4 on reaching such a chunk; the
environment variable ``AXMP_STUB_FAIL_ONCE`` (a path) makes it fail once on such a chunk and create the file, so
the driver's retry then succeeds. ``debug.stub_looping_eV`` (the stand-in's own key) sets ``looping_eV`` per event;
``debug.stub_sleep_s`` makes it wait before its first chunk, after writing ``<out>/.stub_running_<pid>``. A field map
is read, as the program reads it, at the path the job gives, resolved textually against the job file's directory;
its field scales the deposits by 1 + mean |B| / 100 per patch, so a run shows which map it read.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _particles_ref as ref  # noqa: E402

VERSION = "0.0.0-stub"
PARTICLES = {
    "mu-": (13, 105.6583755e6, -1.0), "mu+": (-13, 105.6583755e6, 1.0), "e-": (11, 0.51099895e6, -1.0),
    "e+": (-11, 0.51099895e6, 1.0), "gamma": (22, 0.0, 0.0), "proton": (2212, 938.27208816e6, 1.0),
    "anti_proton": (-2212, 938.27208816e6, -1.0), "neutron": (2112, 939.56542052e6, 0.0),
    "pi+": (211, 139.57039e6, 1.0), "pi-": (-211, 139.57039e6, -1.0), "pi0": (111, 134.9768e6, 0.0),
    "kaon+": (321, 493.677e6, 1.0), "kaon-": (-321, 493.677e6, -1.0), "kaon0L": (130, 497.611e6, 0.0),
    "kaon0S": (310, 497.611e6, 0.0), "geantino": (0, 0.0, 0.0), "chargedgeantino": (0, 0.0, 1.0),
}
NIST = {
    "G4_Galactic": {"density_kg_m3": 1e-22, "mean_excitation_eV": 21.8, "state": "gas", "T_K": 2.73,
                    "pressure_Pa": 3e-18, "elements": {"H": 1.0}},
    "G4_WATER": {"density_kg_m3": 1000.0, "mean_excitation_eV": 78.0, "state": "liquid", "T_K": 293.15,
                 "pressure_Pa": 101325.0, "elements": {"H": 0.111894, "O": 0.888106}},
}


def npy_bytes(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, np.ascontiguousarray(a), allow_pickle=False)
    return b.getvalue()


def check() -> int:
    print(json.dumps({
        "schema": "axionmultiphysics.transport-check/1", "program": "axmp-transport", "program_version": VERSION,
        "geant4_version": "11.4.3", "geant4_tested": True, "multithreaded": True,
        "physics_lists": ["FTFP_BERT", "FTFP_BERT_EMZ", "QGSP_BIC"], "modes": ["hello", "transport", "probe"],
        "geometry": True, "nist": True, "stepper": "G4TDormandPrince45+G4IntegrationDriver",
        "datasets": [{"name": "G4EMLOW", "env": "G4LEDATA", "path": "/stub", "status": "ok", "required": True},
                     {"name": "G4NDL", "env": "G4NEUTRONHPDATA", "path": "/stub", "status": "missing",
                      "required": False}],
        "particles": {k: {"pdg": v[0], "mass_eV": v[1], "charge": v[2]} for k, v in PARTICLES.items()},
        "ok": True}))
    return 0


def nist(names) -> int:
    bad = [n for n in names if n not in NIST]
    if bad:
        print(f"axmp-transport: error: unknown NIST material '{bad[0]}'", file=sys.stderr)
        return 2
    print(json.dumps({"schema": "axionmultiphysics.transport-nist/1", "materials": {n: NIST[n] for n in names}}))
    return 0


def geometry(path) -> int:
    json.loads(Path(path).read_text())
    print(json.dumps({"ok": True, "volumes": [], "materials": [], "overlaps": []}))
    return 0


def kinetic(src) -> tuple[float, float]:
    p = src["particle"]
    if isinstance(p, int):
        p = next(k for k, v in PARTICLES.items() if v[0] == p)
    m = PARTICLES[p][1]
    if "energy_eV" in src:
        return float(src["energy_eV"]), m
    pc = float(src["momentum_eV_c"])
    return math.hypot(pc, m) - m, m


def run(job_path: Path, threads: int) -> int:
    job = json.loads(job_path.read_text())
    if job.get("schema") != "axionmultiphysics.transport-job/1" or job.get("mode") != "transport":
        print("axmp-transport: error: not a transport job", file=sys.stderr)
        return 2
    base = job_path.parent
    field_factor = 1.0
    for f in job.get("fields", []):
        if f.get("type") == "map":       # resolved as the program does: textually against the job file's directory
            mp = os.path.normpath(os.path.join(os.path.abspath(str(base)), f["path"]))
            if not os.path.isfile(os.path.join(mp, "manifest.json")):
                print(f"axmp-transport: error: the field map {mp}: manifest.json cannot be read", file=sys.stderr)
                return 2
            man = json.loads(Path(mp, "manifest.json").read_text())
            for rec in man["patches"]:     # the field changes the deposits: 1 + mean |B| / 100
                field_factor *= 1.0 + float(np.abs(np.load(os.path.join(mp, rec["file"]))).mean()) / 100.0
    out = (base / job["out"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    nap = float(job.get("debug", {}).get("stub_sleep_s", 0.0))
    if nap:
        (out / f".stub_running_{os.getpid()}").write_text("running\n")
        import time
        time.sleep(nap)
    solids = json.loads((base / job["solids"]).read_text())
    regions = list(solids["regions"])
    R = len(regions)
    src = job["source"]
    ekin, mass = kinetic(src)
    pdg = PARTICLES[src["particle"]][0] if isinstance(src["particle"], str) else int(src["particle"])
    q = float(job.get("quantum_eV", 1e-3))
    fail = set(job.get("debug", {}).get("fail_chunks", []))
    fail_once = os.environ.get("AXMP_STUB_FAIL_ONCE")
    for c in job["chunks"]:
        k, seed, n = int(c["index"]), int(c["seed"]), int(c["events"])
        if k in fail:
            if not fail_once or not Path(fail_once).exists():
                if fail_once:
                    Path(fail_once).write_text("failed once\n")
                print(f"axmp-transport: error: chunk {k} failed (debug)", file=sys.stderr)
                return 4
        words = np.array([ref.event_words(seed, e) for e in range(n)], dtype="<u4").reshape(n, 4)
        u = np.array([(int(w[0]) | (int(w[1]) << 32)) / 2.0 ** 64 for w in words], dtype="<f8")
        arrays = {
            "event": np.arange(n, dtype="<u8"), "seed_words": words, "first_flat": u,
            "primary": np.column_stack([np.zeros((n, 3)), np.zeros((n, 2)), np.full(n, math.sqrt(ekin * (ekin + 2 * mass))),
                                        np.full(n, ekin)]).astype("<f8"),
        }
        edep = np.zeros((n, R), dtype="<f8")
        edep[:, 0] = u * ekin * field_factor
        arrays["edep_eV"] = edep
        arrays["exit_eV"] = (ekin - u * ekin).astype("<f8")
        for name in ("n_steps", "n_tracks", "field_calls", "looping_killed"):
            arrays[name] = np.ones(n, dtype="<u8")
        arrays["outside_calls"] = np.full(n, 1 if job.get("outside") == "zero" else 0, dtype="<u8")
        arrays["looping_eV"] = np.full(n, float(job.get("debug", {}).get("stub_looping_eV", 0.0)), dtype="<f8")
        for t in job.get("tallies", []):
            g = t["grid"]
            bins = tuple(int(b) for b in g["bins"])
            T = np.zeros(int(np.prod(bins)), dtype=np.int64)
            for e in range(n):
                T[e % T.size] += int(round(float(u[e]) * ekin / q))
            arrays[f"tally_{t['name']}"] = T.reshape(bins).astype("<i8")
        for h in job.get("hits", []):
            cap = int(h.get("max_per_event", 100))
            rec = {"event": [], "track": [], "parent": [], "pdg": [], "region": [], "x_m": [], "p_eV_c": [],
                   "t_s": [], "ekin_eV": []}
            cnt, drop = np.zeros(n, "<u4"), np.zeros(n, "<u4")
            reg = regions.index(h["regions"][0])
            for e in range(n):
                m = int(4 * u[e]) + 1
                cnt[e], drop[e] = min(m, cap), max(0, m - cap)
                for j in range(min(m, cap)):
                    rec["event"].append(e)
                    rec["track"].append(j + 1)
                    rec["parent"].append(j)
                    rec["pdg"].append(pdg)
                    rec["region"].append(reg)
                    rec["x_m"].append([float(u[e]), float(e), float(j)])
                    rec["p_eV_c"].append([0.0, 0.0, 1.0])
                    rec["t_s"].append(j * 1e-9)
                    rec["ekin_eV"].append(ekin * (1 - j / 10))
            nm = h["name"]
            arrays[f"hits_{nm}_count"], arrays[f"hits_{nm}_dropped"] = cnt, drop
            arrays[f"hits_{nm}_event"] = np.array(rec["event"], "<u8")
            for key in ("track", "parent", "pdg", "region"):
                arrays[f"hits_{nm}_{key}"] = np.array(rec[key], "<i4")
            for key in ("x_m", "p_eV_c"):
                arrays[f"hits_{nm}_{key}"] = np.array(rec[key], "<f8").reshape(-1, 3)
            for key in ("t_s", "ekin_eV"):
                arrays[f"hits_{nm}_{key}"] = np.array(rec[key], "<f8")
        tmp = out / f"chunk_{k}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        listing = {}
        for name, a in arrays.items():
            data = npy_bytes(a)
            (tmp / f"{name}.npy").write_bytes(data)
            listing[name] = {"file": f"{name}.npy", "dtype": a.dtype.str, "shape": list(a.shape),
                             "sha256": hashlib.sha256(data).hexdigest()}
        man = {"schema": "axionmultiphysics.transport/1", "mode": "transport", "chunk": k, "chunk_seed": seed,
               "events": n, "threads": threads, "geant4_version": "11.4.3", "program_version": VERSION,
               "physics_list": job.get("physics_list", "FTFP_BERT"), "production_cut_m": job.get("production_cut_m"),
               "integrator": dict(job.get("integrator", {}), stepper="G4TDormandPrince45+G4IntegrationDriver"),
               "quantum_eV": q, "outside": job.get("outside", "refuse"), "regions": regions, "arrays": listing,
               "counters": {"field_calls": n, "outside_calls": int(arrays["outside_calls"].sum()),
                            "looping_killed": n, "looping_eV": 0.0, "hits_dropped": 0},
               "timing": {"total_s": 0.0}}
        (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
        final = out / f"chunk_{k}"
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    rec_path = (base / job.get("record", str(Path(job["out"]) / "record.json"))).resolve()
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps({"schema": "axionmultiphysics.transport-record/1", "status": "ok",
                                    "mode": "transport", "threads": threads,
                                    "chunks": [c["index"] for c in job["chunks"]]}))
    return 0


def main(argv) -> int:
    if argv[:1] == ["--check"]:
        return check()
    if argv[:1] == ["--nist"]:
        return nist(argv[1:])
    if argv[:1] == ["--geometry"]:
        return geometry(argv[1])
    threads = 1
    if argv[:1] == ["--threads"]:
        threads, argv = int(argv[1]), argv[2:]
    if len(argv) != 1:
        print("axmp-transport: error: usage", file=sys.stderr)
        return 2
    return run(Path(argv[0]), threads)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
