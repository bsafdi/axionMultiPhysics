"""X1 (design note §13.6): a plain Geant4 application, written from Geant4's documentation by an agent other than the
implementer, reads the GDML file that ``axionmultiphysics.solids.write_gdml`` writes from the run's resolved solids
list, and runs the same physics list, cuts, field, source and seeds as ``axmp-transport``.

Geometry, tier D: the volumes of the solids and the boundary-crossing points of 1e4 geantinos are bitwise equal, and
so are the materials as built (density, mean excitation energy, state, temperature, pressure, element fractions:
the resolution of NIST compositions keeps them equal). Deposits, tier B: per-region means agree within
4 sigma; their bitwise equality is reported (it becomes the gate only if it is measured to hold on the gate's build).

The application is found through ``AXMP_X1_APP`` and needs a Geant4 built with GDML (xerces-c). Its command line:
``$AXMP_X1_APP x1.json``, with

    {"gdml": PATH, "out": DIR, "physics_list": NAME, "production_cut_m": X, "integrator": {...} (the job's keys),
     "field": null | {"type": "uniform", "B_T": [3]}, "source": (the job's gun), "chunk_seed": S, "events": N,
     "threads": 1, "regions": [names], "volume_regions": {GDML volume name: region index}}

It seeds each event with the program's four words at the top of GeneratePrimaries, draws u0 first, samples the gun
as docs/guides/particles.md states, and writes to DIR: ``volumes.json`` ({volume name: G4VSolid::GetCubicVolume in m3}),
``materials.json`` (as ``axmp-transport --geometry``'s ``materials``), ``first_flat.npy``, ``edep_eV.npy`` ((N, R)),
``hits_event.npy`` (<u8), ``hits_region.npy`` (<i4), ``hits_x_m.npy`` ((m, 3) <f8): the pre-step point of every
step that enters a volume of a region, in the event's step order.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from axionmultiphysics import particles  # noqa: E402
from axionmultiphysics import solids as solids_mod  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _particles_ref as ref  # noqa: E402
import _solids_ref as sref  # noqa: E402

pytestmark = pytest.mark.px_transport


@pytest.fixture(scope="module")
def programs():
    exe = os.environ.get("AXMP_TRANSPORT")
    app = os.environ.get("AXMP_X1_APP")
    if not exe:
        pytest.skip("axmp-transport is not built here: set AXMP_TRANSPORT")
    if not app:
        pytest.skip("X1 needs the standalone GDML application (AXMP_X1_APP) and a Geant4 built with GDML")
    for p in (exe, app):
        if not (Path(p).is_file() and os.access(p, os.X_OK)):
            pytest.fail(f"{p} is not an executable file")
    return exe, app


@pytest.fixture(scope="module")
def case(programs, tmp_path_factory):
    exe, app = programs
    base = tmp_path_factory.mktemp("x1")
    fx = sref.native()
    table, mats, rm = fx.args()
    mats["water"] = {"density": 1000.0, "particle": {"nist": "G4_WATER"}}
    mats["water998"] = {"density": 998.2, "particle": {"nist": "G4_WATER"}}
    rm["pin"], rm["cone"] = "water", "water998"
    doc = particles.resolve_materials(solids_mod.from_partition(table, mats, rm), program=exe)
    solids_mod.write_gdml(doc, base / "x1.gdml")
    regions = list(doc["regions"])
    volume_regions = {s["gdml_name"]: regions.index(s["region"]) for s in doc["solids"] if s.get("region")}
    return exe, app, base, doc, regions, volume_regions


def run_app(app, base, name, doc_case, source, events, seed, field=None):
    exe, _, _, doc, regions, volume_regions = doc_case
    cfg = {"gdml": str(base / "x1.gdml"), "out": str(base / name), "physics_list": "FTFP_BERT",
           "production_cut_m": 7e-4, "integrator": dict(particles.INTEGRATOR), "field": field, "source": source,
           "chunk_seed": particles.chunk_seed(seed, 0), "events": events, "threads": 1, "regions": regions,
           "volume_regions": volume_regions}
    (base / f"{name}.json").write_text(json.dumps(cfg))
    r = subprocess.run([app, str(base / f"{name}.json")], capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stderr[-3000:]
    return base / name


def run_native(exe, base, name, doc, source, events, seed, field=None):
    regions = list(doc["regions"])
    return particles.run(base / name, program=exe, solids=doc, events=events, chunk_events=events, seed=seed,
                         sources=[source], fields=[field] if field else [],
                         outputs={**{f"E_{i}": {"deposit": {"in": r}} for i, r in enumerate(regions)},
                                  **{f"m_{i}": {"mean": {"of": f"E_{i}"}} for i, r in enumerate(regions)},
                                  "all": {"hits": {"in": regions, "max_per_event": 1000}}})


def test_x1_volumes_and_materials_are_bitwise_equal(case):
    exe, app, base, doc, regions, vr = case
    d = run_app(app, base, "geo", case, {"type": "gun", "particle": "geantino", "energy_eV": 1e6, "at_m": [0, 0, 0],
                                         "direction": "isotropic", "spot_sigma_m": 0.0}, 1, 1)
    (base / "solids.json").write_text(json.dumps(doc))
    g = json.loads(subprocess.run([exe, "--geometry", str(base / "solids.json")], capture_output=True, text=True,
                                  timeout=600).stdout)
    app_vol = json.loads((d / "volumes.json").read_text())
    for v in g["volumes"]:
        assert app_vol[v["g4_name"]] == v["solid_volume_m3"], v["g4_name"]
    app_mat = {m["g4_name"]: m for m in json.loads((d / "materials.json").read_text())}
    for m in g["materials"]:
        a = app_mat[m["g4_name"]]
        for key in ("density_kg_m3", "mean_excitation_eV", "state", "T_K", "pressure_Pa", "elements"):
            assert a[key] == m[key], (m["g4_name"], key)


def test_x1_geantino_crossings_are_bitwise_equal(case):
    exe, app, base, doc, regions, vr = case
    src = {"type": "gun", "particle": "geantino", "energy_eV": 1e6, "at_m": [0.05, 0.02, 0.1],
           "direction": "isotropic", "spot_sigma_m": 0.0}
    d = run_app(app, base, "geantinos", case, src, 10000, 2)
    res = run_native(exe, base, "native_geantinos", doc, src, 10000, 2)
    h = res.hits("all")
    for key, name in (("event", "hits_event.npy"), ("region", "hits_region.npy"), ("x_m", "hits_x_m.npy")):
        a = np.load(d / name)
        assert a.tobytes() == np.ascontiguousarray(h[key]).astype(a.dtype).tobytes(), key
    assert np.load(d / "first_flat.npy").tobytes() == np.load(base / "native_geantinos" / "out" / "chunk_0" /
                                                              "first_flat.npy").tobytes()


def test_x1_deposits_agree_and_their_bitwise_equality_is_reported(case):
    exe, app, base, doc, regions, vr = case
    src = {"type": "gun", "particle": "e-", "energy_eV": 2e8, "at_m": [0.3, 0.0, -0.3], "direction": [0, 0, 1],
           "spot_sigma_m": 0.0}
    field = {"type": "uniform", "B_T": [0.0, 0.3, 0.0]}
    d = run_app(app, base, "showers", case, src, 2000, 3, field)
    res = run_native(exe, base, "native_showers", doc, src, 2000, 3, field)
    app_E = np.load(d / "edep_eV.npy")
    nat = np.column_stack([res.events()[f"E_{i}"] for i in range(len(regions))])
    print("X1 deposits bitwise equal:", app_E.tobytes() == nat.tobytes())
    for i, r in enumerate(regions):
        a, b = app_E[:, i], nat[:, i]
        se = math.hypot(a.std(ddof=1), b.std(ddof=1)) / math.sqrt(len(a))
        assert abs(a.mean() - b.mean()) <= 4 * se + 1e-9 * abs(b.mean()), r
