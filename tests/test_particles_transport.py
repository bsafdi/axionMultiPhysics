"""Particle transport through solids and field maps with ``axmp-transport`` and its driver (beta).

The design note's rows (§13.1 F1, F2, F4, F5; §13.2 P5; §13.7 S1 to S4) and two rules docs/guides/particles.md adds to
the note (a field jump at a curved boundary; NIST compounds resolved against Geant4's tables), on the real program:

- the program's ``--check``, ``--nist`` and ``probe`` mode (the map evaluation in Geant4 against
  ``axionmultiphysics.fieldmap``'s, the frames, the units, the outside rule);
- tracking against exact trajectories: helices in uniform fields (Geant4's and maps of both kinds), the canonical
  angular momentum in a current loop's field (Geant4's own field class and a map filled from the closed form),
  piecewise circles across a field jump at a sphere, a spectrometer of 20 planes against its exact crossings;
- multiple scattering against Highland's formula; resolution against Gluckstern's (PDG 2024);
- reproducibility bit for bit (threads, jobs, a chunk alone or after others, extension, a stop rule, beyond 10 000
  events, MixMax references), seeds statistically independent, the 128-bit tallies, zero counts.

Needs the program: ``AXMP_TRANSPORT`` (every test skips without it; with it set to something that is not an
executable every test fails). Tests that need the field-map module of the package import it inside. S3 (1e7
events) needs ``AXMP_SLOW=1`` as well. Tolerances: tier A rows assert the note's ceiling; the value measured on the
reference platform is printed so the reviewer can tighten them to ten times it.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from axionmultiphysics import particles  # noqa: E402
from axionmultiphysics import solids as solids_mod  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _particles_ref as ref  # noqa: E402
import test_transport_hello as hello  # noqa: E402  (the hello mode's recorded MixMax references)

pytestmark = pytest.mark.px_transport

SLOW = bool(os.environ.get("AXMP_SLOW"))
TIMEOUT = 3600
C = ref.C_LIGHT


# ============================================================================================ helpers
@pytest.fixture(scope="module")
def program() -> str:
    exe = os.environ.get("AXMP_TRANSPORT")
    if not exe:
        pytest.skip("axmp-transport is not built here: set AXMP_TRANSPORT to it (cmake -DAXMP_BUILD_TRANSPORT=ON)")
    if not (Path(exe).is_file() and os.access(exe, os.X_OK)):
        pytest.fail(f"AXMP_TRANSPORT={exe} is not an executable file")
    return exe


def doc_of(P, regions, materials, region_materials, **kw):
    return solids_mod.from_partition(P.table(regions), materials, region_materials, **kw)


def vacuum_slabs(z_edges, names, rmax, materials=None, region_materials=None):
    P, regions = ref.slabs(z_edges, names, rmax)
    mats = materials or {"vac": ref.VACUUM}
    rm = region_materials or {r: "vac" for r in regions}
    return doc_of(P, regions, mats, rm)


def run(wd, program, **kw):
    kw.setdefault("seed", 1)
    return particles.run(wd, program=program, **kw)


def gun(particle, at, direction, **e):
    g = {"type": "gun", "particle": particle, "at_m": list(at), "direction": direction}
    g.update(e)
    return g


def primary_hits(res, name):
    """The hits of the primary (track 1) of each event, in (chunk, event, step) order."""
    h = res.hits(name)
    keep = h["track"] == 1
    return {k: v[keep] for k, v in h.items()}


def code_of(excinfo):
    return getattr(excinfo.value, "code", None)


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in ("timing", "programs", "threads")}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def chunk_arrays(wd, k):
    d = Path(wd) / "out" / f"chunk_{k}"
    man = json.loads((d / "manifest.json").read_text())
    return {n: np.load(d / e["file"]) for n, e in man["arrays"].items()}, man


def assert_chunks_bitwise(a, b, ks, what):
    for k in ks:
        x, mx = chunk_arrays(a, k)
        y, my = chunk_arrays(b, k)
        assert sorted(x) == sorted(y), what
        for n in x:
            if x[n].tobytes() != y[n].tobytes():
                pytest.fail(f"{what}: chunk {k} array '{n}' differs")
        assert strip(mx) == strip(my), f"{what}: chunk {k} manifest differs outside threads/timing"


def run_job(program, job_dir, job, threads=1, extra=()):
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(job, indent=1))
    return subprocess.run([program, "--threads", str(threads), *extra, str(job_dir / "job.json")], capture_output=True,
                          text=True, timeout=TIMEOUT, cwd=job_dir)


def write_map(path, patches, outside="refuse"):
    from axionmultiphysics import fieldmap
    return fieldmap.write_map(path, patches, outside=outside)


# ============================================================================================ the program
def test_check_offers_the_transport_modes_physics_lists_and_particle_table(program):
    r = subprocess.run([program, "--check"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    c = json.loads(r.stdout)
    assert {"hello", "transport", "probe"} <= set(c["modes"])
    assert c["physics_lists"] == ["FTFP_BERT", "FTFP_BERT_EMZ", "QGSP_BIC"]
    assert c["nist"] is True and c["geometry"] is True
    assert c["stepper"] == "G4TDormandPrince45+G4IntegrationDriver"
    t = c["particles"]
    assert t["mu-"]["pdg"] == 13 and t["mu+"]["pdg"] == -13 and t["proton"]["pdg"] == 2212
    assert t["mu-"]["mass_eV"] == pytest.approx(ref.M_MU_EV, rel=1e-7) and t["mu-"]["charge"] == -1.0
    assert t["e+"]["charge"] == 1.0 and t["geantino"]["mass_eV"] == 0.0 and "chargedgeantino" in t


def test_nist_reports_geant4s_own_material_data(program):
    names = ["G4_WATER", "G4_SODIUM_IODIDE", "G4_Fe", "G4_AIR", "G4_Galactic"]
    r = subprocess.run([program, "--nist", *names], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert out["schema"] == "axionmultiphysics.transport-nist/1" and sorted(out["materials"]) == sorted(names)
    w = out["materials"]["G4_WATER"]
    assert w["mean_excitation_eV"] == pytest.approx(78.0, rel=1e-12) and w["state"] == "liquid"
    assert w["density_kg_m3"] == pytest.approx(1000.0, rel=1e-12)
    assert w["elements"]["H"] == pytest.approx(2 * 1.008 / 18.015, rel=2e-3) and sum(w["elements"].values()) == \
        pytest.approx(1.0, abs=1e-12)
    n = out["materials"]["G4_SODIUM_IODIDE"]
    assert (n["mean_excitation_eV"], n["state"]) == (pytest.approx(452.0, rel=1e-12), "solid")
    assert n["density_kg_m3"] == pytest.approx(3667.0, rel=1e-12)
    assert out["materials"]["G4_Fe"]["mean_excitation_eV"] == pytest.approx(286.0, rel=1e-12)
    assert out["materials"]["G4_AIR"]["state"] == "gas"
    r = subprocess.run([program, "--nist", "G4_WATER", "G4_UNOBTAINIUM"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 2 and "G4_UNOBTAINIUM" in r.stderr and r.stdout.strip() == ""


def test_resolved_nist_compositions_are_built_as_resolved(program, tmp_path):
    """Decision (ii): at NIST's density the solids use the NIST material itself; at another density one component
    carrying NIST's I-value and state. The native path builds exactly what the list says (``--geometry`` reports the
    materials as built), so the GDML route reading the same list builds the same."""
    nist = json.loads(subprocess.run([program, "--nist", "G4_WATER", "G4_SODIUM_IODIDE"], capture_output=True,
                                     text=True, timeout=600).stdout)["materials"]
    P, regions = ref.slabs([-0.5, 0.0, 0.1, 0.2, 0.5], ["gas", "w1000", "w998", "nai"], 0.3)
    mats = {"vac": ref.VACUUM, "water": {"density": 1000.0, "particle": {"nist": "G4_WATER"}},
            "water998": {"density": 998.2, "particle": {"nist": "G4_WATER"}},
            "nai": {"density": 3667.0, "particle": {"nist": "G4_SODIUM_IODIDE"}}}
    doc = doc_of(P, regions, mats, {"gas": "vac", "w1000": "water", "w998": "water998", "nai": "nai"})
    r = particles.resolve_materials(doc, program=program)
    (tmp_path / "solids.json").write_text(json.dumps(r))
    g = subprocess.run([program, "--geometry", str(tmp_path / "solids.json")], capture_output=True, text=True,
                       timeout=600)
    assert g.returncode == 0, g.stderr[-2000:]
    out = json.loads(g.stdout)
    by_region = {v["region"]: v for v in out["volumes"] if v["region"]}
    mat = {m["g4_name"]: m for m in out["materials"]}
    assert by_region["w1000"]["material"] == "G4_WATER" and by_region["nai"]["material"] == "G4_SODIUM_IODIDE"
    assert mat["G4_WATER"]["mean_excitation_eV"] == nist["G4_WATER"]["mean_excitation_eV"]
    w998 = mat[next(m["gdml_name"] for m in r["materials"] if m["name"] == "water998")]
    assert w998["mean_excitation_eV"] == nist["G4_WATER"]["mean_excitation_eV"]      # bitwise: the list's double
    assert w998["state"] == "liquid" and w998["density_kg_m3"] == pytest.approx(998.2, rel=1e-12)


# ============================================================================================ probe: maps in Geant4
def _probe(program, wd, doc, fields, points, outside="refuse"):
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "solids.json").write_text(json.dumps(doc))
    np.save(wd / "points.npy", np.asarray(points, dtype="<f8"))
    job = {"schema": "axionmultiphysics.transport-job/1", "mode": "probe", "out": "out", "solids": "solids.json",
           "fields": fields, "outside": outside, "probe": {"points": "points.npy"}}
    r = run_job(program, wd, job)
    if r.returncode != 0:
        return r, None, None, None
    d = wd / "out" / "probe"
    return r, np.load(d / "B_T.npy"), np.load(d / "region.npy"), json.loads((d / "manifest.json").read_text())


def test_maps_are_evaluated_in_geant4_as_the_package_evaluates_them(program, tmp_path):
    """Kind 1 with B_r, B_z and B_phi all varying (the cos/sin projection), kind 3 with a rotated and translated frame,
    a uniform field and the current loop: every probe point equals ``FieldMap.evaluate`` (or the closed form) within
    1e-12 of the field's scale, so the tesla factor, the component order, the axes and the frames are right."""
    from axionmultiphysics import fieldmap
    rr = np.linspace(0.0, 0.5, 26)
    zz = np.linspace(-0.4, 0.4, 41)
    R, Z = np.meshgrid(rr, zz, indexing="ij")
    B1 = np.stack([0.3 * R * Z + 0.01, 1.2 - 0.5 * R ** 2 + 0.2 * Z, 0.7 * R - 0.1 * Z], axis=-1)
    th = math.radians(30.0)
    T = np.eye(4)
    T[:2, :2] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    T[:3, 3] = [0.01, -0.02, 0.03]
    g = np.linspace(-0.6, 0.6, 31)
    X, Y, Zc = np.meshgrid(g, g, np.linspace(-0.5, 0.5, 26), indexing="ij")
    B3 = np.stack([0.2 + 0.3 * X - 0.1 * Zc, -0.4 + 0.2 * Y * X, 0.9 + 0.05 * Zc], axis=-1)
    m = write_map(tmp_path / "map", [
        {"name": "axi", "kind": 1, "origin": [0.0, -0.4], "spacing": [0.02, 0.02], "B": B1},
        {"name": "cart", "kind": 3, "origin": [-0.6, -0.6, -0.5], "spacing": [0.04, 0.04, 0.04], "B": B3,
         "transform": T.tolist()}])
    doc = vacuum_slabs([-0.4, -0.1, 0.0, 0.1, 0.4], ["a", "b", "c", "d"], 0.45)
    fields = [{"type": "map", "path": str(tmp_path / "map"), "patch": "axi", "regions": ["a"]},
              {"type": "map", "path": str(tmp_path / "map"), "patch": "cart", "regions": ["b"]},
              {"type": "uniform", "B_T": [0.1, -0.2, 0.3], "regions": ["c"]},
              {"type": "current_loop", "radius_m": 0.3, "current_A": 2.0e5, "z_m": 0.25, "regions": ["d"]}]
    rng = np.random.default_rng(5)
    pts = []
    for zlo, zhi in ((-0.39, -0.11), (-0.099, -0.001), (0.001, 0.099), (0.11, 0.39)):
        r_ = 0.44 * np.sqrt(rng.random(400))
        ph = 2 * math.pi * rng.random(400)
        pts.append(np.column_stack([r_ * np.cos(ph), r_ * np.sin(ph), rng.uniform(zlo, zhi, 400)]))
    pts.append([[0.0, 0.0, -0.2], [0.0, 0.0, 0.3], [2.0, 0.0, 0.0]])          # on the axis, and in the world
    pts = np.vstack(pts)
    r, B, reg, man = _probe(program, tmp_path / "p", doc, fields, pts)
    assert r.returncode == 0, r.stderr[-3000:]
    regions = list(doc["regions"])
    want = np.zeros_like(pts)
    for i, p in enumerate(pts):
        if reg[i] < 0:
            continue
        name = regions[reg[i]]
        if name == "a":
            want[i] = m.evaluate([p], patch="axi")[0]
        elif name == "b":
            want[i] = m.evaluate([p], patch="cart")[0]
        elif name == "c":
            want[i] = [0.1, -0.2, 0.3]
        else:
            rp = math.hypot(p[0], p[1])
            br, bz = ref.loop_field(rp, p[2], 0.3, 2.0e5, 0.25)
            want[i] = [br * p[0] / rp if rp > 0 else 0.0, br * p[1] / rp if rp > 0 else 0.0, bz]
    assert reg[-1] == -1 and np.all(B[-1] == 0.0)
    assert [regions[k] for k in reg[-3:-1]] == ["a", "d"]
    scale = np.max(np.abs(want), axis=0).max()
    err = np.max(np.abs(B - want))
    print(f"probe: max |B_program - B_reference| = {err:.3e} T (scale {scale:.3f} T)")
    assert err <= 1e-12 * scale, err
    for name in "abcd":
        assert np.sum(reg == regions.index(name)) >= 300, name


def test_a_probe_point_outside_its_patch_obeys_outside(program, tmp_path):
    B = np.zeros((11, 11, 3))
    B[..., 1] = 0.5
    write_map(tmp_path / "map", [{"name": "p", "kind": 1, "origin": [0.0, -0.1], "spacing": [0.01, 0.02], "B": B}])
    doc = vacuum_slabs([-0.5, 0.5], ["a"], 0.4)
    f = [{"type": "map", "path": str(tmp_path / "map"), "patch": "p", "regions": ["a"]}]
    inside_margin = [[0.105, 0.0, 0.05], [0.0, 0.0, 0.115]]        # within one spacing of the box: clamped
    beyond = [[0.3, 0.0, 0.0], [0.0, 0.0, 0.3], [0.05, 0.0, -0.4]]
    r, Bp, reg, man = _probe(program, tmp_path / "z", doc, f, inside_margin + beyond, outside="zero")
    assert r.returncode == 0, r.stderr[-2000:]
    assert np.allclose(Bp[:2], [[0, 0, 0.5]] * 2, rtol=0, atol=1e-15) and np.all(Bp[2:] == 0.0)
    assert man["outside_calls"] == 3
    r, *_ = _probe(program, tmp_path / "r", doc, f, inside_margin + beyond, outside="refuse")
    assert r.returncode == 4 and "fieldmap.outside_map" in r.stderr


def test_the_program_refuses_a_delta_intersection_above_a_tenth_of_the_finest_map_spacing(program, tmp_path):
    """The rule of the field-jump decision (deltaIntersection at most 0.1 of the finest spacing of any attached
    patch) holds for any job file the program is handed, not only for the driver's: 0.0021 m against patches of
    0.04 m and 0.02 m spacing is refused with exit status 2, a message naming the setting, and nothing written;
    0.002 m runs."""
    B = np.zeros((11, 21, 3))
    B[..., 1] = 0.3
    write_map(tmp_path / "m", [{"name": "coarse", "kind": 1, "origin": [0.0, -0.4], "spacing": [0.04, 0.04], "B": B},
                               {"name": "fine", "kind": 1, "origin": [0.0, -0.2], "spacing": [0.02, 0.02], "B": B}])
    doc = vacuum_slabs([-0.38, 0.0, 0.18], ["a", "b"], 0.2)      # inside both patches' boxes
    for dint, ok in ((0.0021, False), (0.002, True)):
        job, _ = geantino_job("out", [(0, 1, 2)])
        job["source"]["at_m"] = [0.0, 0.0, -0.3]
        job["fields"] = [{"type": "map", "path": str(tmp_path / "m"), "patch": "coarse", "regions": ["a"]},
                         {"type": "map", "path": str(tmp_path / "m"), "patch": "fine", "regions": ["b"]}]
        job["integrator"] = dict(particles.INTEGRATOR, delta_intersection_m=dint)
        d = tmp_path / f"j{dint}"
        d.mkdir()
        (d / "solids.json").write_text(json.dumps(doc))
        r = run_job(program, d, job)
        if ok:
            assert r.returncode == 0, r.stderr[-2000:]
        else:
            assert r.returncode == 2, (r.returncode, r.stderr[-2000:])
            assert "delta_intersection" in r.stderr or "deltaIntersection" in r.stderr
            assert not (d / "out" / "chunk_0").exists()


# ============================================================================================ F1: helices
P_T, P_Z = 1.0, 1.0                                      # GeV/c
R_F1 = P_T / (ref.GEV_PER_TM * 1.0)                      # 3.33564 m in 1 T
PITCH = 2 * math.pi * R_F1 * P_Z / P_T                   # 20.958 m
T_PLANE = 1e-3


def f1_planes_along_z():
    edges = [-1.0]
    names = []
    for k in range(1, 11):
        edges += [k * PITCH - T_PLANE, k * PITCH]
        names += ["vac", "plane"]
    edges.append(10 * PITCH + 1.0)
    names.append("vac")
    return vacuum_slabs(edges, names, 7.0)


def f1_shells_along_r():
    edges = [0.0]
    names = []
    for k in range(1, 11):
        edges += [21.0 * k - T_PLANE, 21.0 * k]
        names += ["vac", "plane"]
    edges.append(211.0)
    names.append("vac")
    P, regions = ref.shells(edges, names, -4.0, 4.0)
    return doc_of(P, regions, {"vac": ref.VACUUM}, {r: "vac" for r in regions})


def f1_error(res, axis):
    """Largest distance between a hit and the exact helix at the same surface, relative to R, and the largest
    relative change of |p|."""
    h = primary_hits(res, "planes")
    p0 = math.hypot(P_T, P_Z) * 1e9
    err = []
    for x in h["x_m"]:
        if axis == "z":
            s = x[2] * math.hypot(P_T, P_Z) / P_Z
            exact = ref.helix_z(s, P_T, P_Z, 1.0)
        else:
            rk = math.hypot(x[0], x[1])
            s = ref.first_root(lambda s: math.hypot(*ref.helix_x(s, P_Z, P_T, 1.0)[:2]) - rk, 0.0, 400.0, n=20000)
            exact = ref.helix_x(s, P_Z, P_T, 1.0)
        err.append(np.linalg.norm(x - exact) / R_F1)
    dp = np.max(np.abs(np.linalg.norm(h["p_eV_c"], axis=1) / p0 - 1))
    return max(err), dp, len(h["x_m"])


@pytest.mark.parametrize("field", ["geant4_uniform", "map_kind1"])
def test_f1_a_helix_along_z_closes_on_itself_for_ten_turns(program, tmp_path, field):
    if field == "geant4_uniform":
        fields = [{"type": "uniform", "B_T": [0.0, 0.0, 1.0]}]
    else:
        B = np.zeros((2, 2, 3))
        B[..., 1] = 1.0
        write_map(tmp_path / "m", [{"name": "u", "kind": 1, "origin": [0.0, -2.0], "spacing": [8.0, 215.0], "B": B}])
        fields = [{"type": "map", "path": str(tmp_path / "m"), "patch": "u", "regions": ["vac", "plane"]}]
    d = np.array([P_T, 0.0, P_Z]) / math.hypot(P_T, P_Z)
    res = run(tmp_path / "w", program, solids=f1_planes_along_z(), fields=fields, events=1, chunk_events=1,
              sources=[gun("mu+", (0, 0, 0), d.tolist(), momentum_eV_c=math.hypot(P_T, P_Z) * 1e9)],
              outputs={"planes": {"hits": {"in": ["plane"], "track": "primary"}}})
    err, dp, n = f1_error(res, "z")
    print(f"F1 along z ({field}): max |dx| / R = {err:.3e}, max ||p|/p - 1| = {dp:.1e}")
    assert n == 10
    assert err <= 1e-4 and dp <= 1e-9


@pytest.mark.parametrize("field", ["geant4_uniform", "map_kind3"])
def test_f1_a_helix_about_x(program, tmp_path, field):
    if field == "geant4_uniform":
        fields = [{"type": "uniform", "B_T": [1.0, 0.0, 0.0]}]
    else:
        B = np.zeros((2, 2, 2, 3))
        B[..., 0] = 1.0
        write_map(tmp_path / "m", [{"name": "u", "kind": 3, "origin": [-212.0, -212.0, -5.0],
                                    "spacing": [424.0, 424.0, 10.0], "B": B}])
        fields = [{"type": "map", "path": str(tmp_path / "m"), "patch": "u", "regions": ["vac", "plane"]}]
    d = np.array([P_Z, 0.0, P_T]) / math.hypot(P_T, P_Z)          # along B: P_Z; across: P_T
    res = run(tmp_path / "w", program, solids=f1_shells_along_r(), fields=fields, events=1, chunk_events=1,
              sources=[gun("mu+", (0, 0, 0), d.tolist(), momentum_eV_c=math.hypot(P_T, P_Z) * 1e9)],
              outputs={"planes": {"hits": {"in": ["plane"], "track": "primary"}}})
    err, dp, n = f1_error(res, "x")
    print(f"F1 about x ({field}): max |dx| / R = {err:.3e}, max ||p|/p - 1| = {dp:.1e}")
    assert n == 10
    assert err <= 1e-4 and dp <= 1e-9


def test_f1_the_error_is_controlled_by_the_integrator(program, tmp_path):
    """The note: halving deltaOneStep must reduce the error. The accuracy parameters (deltaOneStep, deltaIntersection,
    deltaChord, the epsilons) at 4, 2 and 1 times Geant4's defaults: the error falls at each halving (measured with
    a scratch Geant4 program on this geometry: 3.5e-8, 6.2e-9 and 1.1e-9 of R) and stays below the ceiling."""
    errs = {}
    for f in (4, 2, 1):
        integ = {k: v * f for k, v in particles.INTEGRATOR.items() if k in ("delta_one_step_m",
                                                                           "delta_intersection_m", "delta_chord_m")}
        integ["epsilon_max"] = min(1e-2, 1e-3 * f)
        integ["epsilon_min"] = min(integ["epsilon_max"], 5e-5 * f)
        d = np.array([P_T, 0.0, P_Z]) / math.hypot(P_T, P_Z)
        res = run(tmp_path / f"w{f}", program, solids=f1_planes_along_z(), events=1, chunk_events=1,
                  fields=[{"type": "uniform", "B_T": [0.0, 0.0, 1.0]}], integrator=integ,
                  sources=[gun("mu+", (0, 0, 0), d.tolist(), momentum_eV_c=math.hypot(P_T, P_Z) * 1e9)],
                  outputs={"planes": {"hits": {"in": ["plane"], "track": "primary"}}})
        errs[f] = f1_error(res, "z")[0]
    print("F1 error by integrator scale:", errs)
    assert errs[4] > errs[2] > errs[1] and errs[1] <= 1e-4


# ============================================================================================ F2: canonical angular momentum
A_LOOP, I_LOOP = 0.5, 1.0e6
X0_P = np.array([0.2, 0.05, -1.5])
D0_P = np.array([0.05, 0.08, 1.0]) / np.linalg.norm([0.05, 0.08, 1.0])
P0_P = 100e6                          # eV/c, proton
PLANES_F2 = (-1.0, -0.5, 0.0, 0.5, 1.0)


def f2_doc():
    edges, names = [-1.6], []
    for z in PLANES_F2:
        edges += [z - 1e-4, z]
        names += ["vac", "plane"]
    edges.append(1.6)
    names.append("vac")
    return vacuum_slabs(edges, names, 1.5)


def f2_dense_doc():
    """13 planes of 1 mm between z = -0.3 and 0.3 m, close to the loop's plane (the review's case)."""
    edges, names = [-0.45], []
    for z in np.linspace(-0.3, 0.3, 13):
        edges += [float(z), float(z) + 1e-3]
        names += ["vac", "plane"]
    edges.append(0.45)
    names.append("vac")
    return vacuum_slabs(edges, names, 0.45)


def f2_run(program, wd, fields, x0=X0_P, d0=D0_P, integrator=None, doc=None):
    return run(wd, program, solids=doc or f2_doc(), fields=fields, events=1, chunk_events=1, integrator=integrator,
               sources=[gun("proton", x0, list(d0), momentum_eV_c=P0_P)],
               outputs={"planes": {"hits": {"in": ["plane"], "track": "primary"}}})


def p_phi_change(res, x0=X0_P, d0=D0_P):
    h = primary_hits(res, "planes")
    A = lambda r, z: ref.loop_Aphi(r, z, A_LOOP, I_LOOP)        # noqa: E731
    P = ref.canonical_p_phi(h["x_m"], h["p_eV_c"], 1.0, A)
    P0 = ref.canonical_p_phi(np.asarray(x0), P0_P * np.asarray(d0), 1.0, A)[0]
    scale = max(np.max(np.hypot(h["x_m"][:, 0], h["x_m"][:, 1]) * np.linalg.norm(h["p_eV_c"], axis=1)),
                math.hypot(*x0[:2]) * P0_P)
    return np.abs(P - P0), scale, h


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# Protons of 100 MeV/c in the loop's field: two through the five planes of f2_doc, and the review's through 13 planes
# close to the loop's plane, where the default integrator is least accurate
F2_TRACKS = [(X0_P, D0_P, f2_doc), (np.array([-0.12, 0.15, -1.5]), _unit([-0.05, 0.1, 1.0]), f2_doc),
             (np.array([0.2, 0.0, -0.4]), _unit([0.3, 0.2, 1.0]), f2_dense_doc)]
F2_DEFAULT = 2.5e-5       # ten times the largest measured at Geant4's defaults (2.3e-6), below one deltaOneStep / r
F2_CEILING = 1e-6         # the note's ceiling, held with the tighter integrator (measured up to 1.8e-7)


def loop_field_entry():
    return [{"type": "current_loop", "radius_m": A_LOOP, "current_A": I_LOOP, "z_m": 0.0}]


def test_f2_the_canonical_angular_momentum_is_conserved_in_geant4s_loop_field(program, tmp_path):
    """At Geant4's default integrator, which the transport pins, |dP_phi| / max r|p| is the integrator's accuracy and
    depends on the track and its planes: about 2e-7 for the first two, 2.3e-6 for the review's (13 planes near the
    loop's plane); each step is held to deltaOneStep = 1e-5 m, 5e-5 of r = 0.2 m. The tolerance is ten times the
    largest measured, 2.5e-5; the note's ceiling of 1e-6 is held with a tighter integrator (next test)."""
    worst = 0.0
    for i, (x0, d0, doc) in enumerate(F2_TRACKS):
        res = f2_run(program, tmp_path / f"w{i}", loop_field_entry(), x0, d0, doc=doc())
        dP, scale, h = p_phi_change(res, x0, d0)
        worst = max(worst, dP.max() / scale)
        print(f"F2 analytic, track {i}, defaults: max |dP_phi| / max r|p| = {dP.max() / scale:.3e} over {len(dP)} "
              "planes")
        assert len(dP) >= 3
        assert np.allclose(np.linalg.norm(h["p_eV_c"], axis=1), P0_P, rtol=1e-9)    # kinetic energy
    assert worst <= F2_DEFAULT


def test_f2_with_a_tighter_integrator_the_note_ceiling_holds(program, tmp_path):
    """deltaOneStep 1e-6 m, deltaIntersection 1e-7 m, epsilon 5e-6 to 1e-4: every track within the note's ceiling,
    1e-6 of max r|p| (measured 5e-8 to 1.8e-7, the review's track 1.8e-7), and a track that exceeds the ceiling at
    the defaults improves at least 5 times (the review's: 13 times): the change of P_phi is the integrator's, which
    a sign, a projection or a unit error would not be."""
    for i, (x0, d0, doc) in enumerate(F2_TRACKS):
        dP_d, scale_d, _ = p_phi_change(f2_run(program, tmp_path / f"d{i}", loop_field_entry(), x0, d0, doc=doc()),
                                        x0, d0)
        dP, scale, h = p_phi_change(f2_run(program, tmp_path / f"t{i}", loop_field_entry(), x0, d0, TIGHT, doc()),
                                    x0, d0)
        print(f"F2 analytic, track {i}: tight {dP.max() / scale:.3e}, defaults {dP_d.max() / scale_d:.3e}")
        assert dP.max() <= F2_CEILING * scale
        if dP_d.max() / scale_d > F2_CEILING:
            assert dP.max() / scale * 5 <= dP_d.max() / scale_d


def test_f2_through_a_map_the_change_is_bounded_by_the_map_error_along_the_track(program, tmp_path):
    """|dP_phi| <= |q| c integral of r |B_map - B_exact| dl (the note's bound; no psi_map is assumed), evaluated here
    along the exact-field trajectory, plus the analytic case's ceiling."""
    from axionmultiphysics import fieldmap
    rr = np.arange(151) * 0.01
    zz = -1.605 + np.arange(322) * 0.01                     # no node on the wire (0.5, 0)
    R, Z = np.meshgrid(rr, zz, indexing="ij")
    br, bz = ref.loop_field(R, Z, A_LOOP, I_LOOP)
    write_map(tmp_path / "m", [{"name": "loop", "kind": 1, "origin": [0.0, -1.605], "spacing": [0.01, 0.01],
                                "B": np.stack([br, bz, np.zeros_like(br)], axis=-1)}])
    fm = fieldmap.load(tmp_path / "m")
    res = f2_run(program, tmp_path / "w", [{"type": "map", "path": str(tmp_path / "m"), "patch": "loop",
                                            "regions": ["vac", "plane"]}])
    dP, scale, h = p_phi_change(res)

    def exact_B(x):
        r = math.hypot(x[0], x[1])
        b_r, b_z = ref.loop_field(r, x[2], A_LOOP, I_LOOP)
        return np.array([b_r * x[0] / r, b_r * x[1] / r, b_z]) if r > 0 else np.array([0.0, 0.0, float(b_z)])

    k = C / P0_P

    def rhs(s, y):
        B = exact_B(y[:3])
        dB = np.linalg.norm(fm.evaluate([y[:3]])[0] - B)
        return np.concatenate([y[3:6], k * np.cross(y[3:6], B), [math.hypot(y[0], y[1]) * dB]])

    events = [(lambda s, y, z=z: y[2] - (z - 1e-4)) for z in PLANES_F2]
    events[-1].terminal = True                               # the map ends at z = 1.605
    sol = integrate.solve_ivp(rhs, (0.0, 6.0), np.concatenate([X0_P, D0_P, [0.0]]), method="DOP853", rtol=1e-10,
                              atol=1e-12, events=events)
    bound = np.array([C * sol.y_events[i][0][6] for i in range(len(PLANES_F2))]) + F2_DEFAULT * scale
    print("F2 map: |dP_phi| =", dP, " bound =", bound)
    assert len(dP) == len(PLANES_F2)
    assert np.all(dP <= bound)
    assert np.all(bound < 0.05 * scale), "the bound is too loose to test anything"


# ============================================================================================ a field jump at a curved boundary
A_BALL, H_GRID = 0.25, 0.005


def jump_maps(path):
    """Two patches holding their region's uniform B_z up to two cells beyond the region and 0 beyond: what a map
    patch holds beyond its continuation."""
    ri = np.arange(54) * H_GRID
    zi = -0.265 + np.arange(107) * H_GRID
    R, Z = np.meshgrid(ri, zi, indexing="ij")
    Bi = np.zeros((*R.shape, 3))
    Bi[..., 1] = np.where(np.hypot(R, Z) <= A_BALL + 2 * H_GRID, 1.0, 0.0)
    ro = np.arange(122) * H_GRID
    zo = -0.305 + np.arange(123) * H_GRID
    R, Z = np.meshgrid(ro, zo, indexing="ij")
    Bo = np.zeros((*R.shape, 3))
    Bo[..., 1] = np.where(np.hypot(R, Z) >= A_BALL - 2 * H_GRID, 0.2, 0.0)
    write_map(path, [{"name": "inner", "kind": 1, "origin": [0.0, -0.265], "spacing": [H_GRID, H_GRID], "B": Bi},
                     {"name": "outer", "kind": 1, "origin": [0.0, -0.305], "spacing": [H_GRID, H_GRID], "B": Bo}])


@pytest.mark.parametrize("p_gev", [0.5, 0.1])
def test_a_field_jump_at_a_curved_boundary_follows_the_exact_trajectory(program, tmp_path, p_gev):
    """Per-volume patches, each correct only up to two cells beyond its region, and Geant4's
    default integration driver give the exact piecewise circles (measured with a scratch Geant4 program on this
    configuration: 2.6e-6 m at 0.5 GeV/c and 1.1e-5 to 1.5e-5 m at 0.1 GeV/c). The tolerance, 1e-4 m and 1e-4 rad, is
    about ten times that; the interpolation driver, one point-located field, no continuation or a deltaIntersection
    of 0.1 mm give 2.3e-4 to 6e-3 m."""
    jump_maps(tmp_path / "m")
    P, regions = ref.ball_in_cylinder(A_BALL, 0.6, 0.601, 0.3)
    doc = doc_of(P, regions, {"vac": ref.VACUUM}, {r: "vac" for r in regions})
    fields = [{"type": "map", "path": str(tmp_path / "m"), "patch": "inner", "regions": ["ball"]},
              {"type": "map", "path": str(tmp_path / "m"), "patch": "outer", "regions": ["outer", "exit"]}]
    res = run(tmp_path / "w", program, solids=doc, fields=fields, events=60, chunk_events=60, seed=3,
              sources=[gun("mu+", (-0.4, 0.0, 0.0), [1.0, 0.0, 0.0], momentum_eV_c=p_gev * 1e9, spot_sigma_m=0.08)],
              outputs={"x0": {"primary": {"of": "x"}}, "y0": {"primary": {"of": "y"}}, "z0": {"primary": {"of": "z"}},
                       "out": {"hits": {"in": ["exit"], "track": "primary", "max_per_event": 1}}})
    ev = res.events()
    h = res.hits("out")
    first = {}
    for i in range(len(h["event"])):
        first.setdefault(int(h["event"][i]), i)
    errs, angs, crossed = [], [], 0
    for e in range(60):
        z0 = ev["z0"][e]
        if abs(z0) > 0.29:
            continue
        assert ev["x0"][e] == -0.4
        a_eff = math.sqrt(A_BALL ** 2 - z0 ** 2) if abs(z0) < A_BALL else 1e-9
        x, y, dx, dy = ref.circles_across_sphere(ev["y0"][e], p_gev, a_eff, 1.0, 0.2, 0.6, -0.4)
        i = first[e]
        errs.append(math.hypot(h["x_m"][i][0] - x, h["x_m"][i][1] - y))
        angs.append(abs(math.atan2(h["p_eV_c"][i][1], h["p_eV_c"][i][0]) - math.atan2(dy, dx)))
        assert abs(h["x_m"][i][2] - z0) <= 1e-9
        crossed += abs(ev["y0"][e]) < a_eff
    print(f"jump, p = {p_gev}: max exit error {max(errs):.2e} m, {max(angs):.2e} rad; {crossed} of {len(errs)} "
          "tracks cross the ball")
    assert crossed >= 20
    assert max(errs) <= 1e-4 and max(angs) <= 1e-4


# ============================================================================================ F4: spectrometer
B_F4, P_F4, L_F4, N_F4, EPS_F4 = 1.5, 10.0, 2.0, 20, 100e-6
Z_F4 = np.arange(N_F4) * L_F4 / (N_F4 - 1)


def f4_doc(material):
    edges, names = [-0.2], []
    for z in Z_F4:
        edges += [z, z + 1e-4]
        names += ["gap", "plane"]
    edges.append(2.2)
    names.append("gap")
    P, regions = ref.slabs(edges, names, 0.5)
    return doc_of(P, regions, {"m": material}, {r: "m" for r in regions})


def f4_run(program, wd, material, events, seed=11, integrator=None):
    return run(wd, program, solids=f4_doc(material), events=events, chunk_events=max(1, events // 20), seed=seed,
               fields=[{"type": "uniform", "B_T": [B_F4, 0.0, 0.0]}], integrator=integrator,
               sources=[gun("mu+", (0, 0, -0.1), [0.0, 0.0, 1.0], momentum_eV_c=P_F4 * 1e9)],
               outputs={"planes": {"hits": {"in": ["plane"], "track": "primary", "max_per_event": 40}}})


def f4_exact_y(z):
    R = P_F4 / (ref.GEV_PER_TM * B_F4)
    th = np.arcsin((z + 0.1) / R)
    return R * (1 - np.cos(th))


def f4_tracks(res):
    h = primary_hits(res, "planes")
    keys = h["chunk"].astype(np.int64) * 10 ** 9 + h["event"].astype(np.int64)
    out = []
    for k in np.unique(keys):
        sel = keys == k
        if sel.sum() == N_F4:
            out.append(h["x_m"][sel])
    return out


TIGHT = {"delta_one_step_m": 1e-6, "delta_intersection_m": 1e-7, "epsilon_min": 5e-6, "epsilon_max": 1e-4}


def f4_vacuum(program, wd, integrator=None):
    res = f4_run(program, wd, ref.VACUUM, 1, integrator=integrator)
    (x,) = f4_tracks(res)
    assert np.allclose(x[:, 2], Z_F4, rtol=0, atol=1e-9)
    assert np.max(np.abs(x[:, 0])) <= 1e-9
    err = float(np.max(np.abs(x[:, 1] - f4_exact_y(x[:, 2]))))
    zc, yc, R = ref.fit_circle(x[:, 2], x[:, 1])
    return err, ref.GEV_PER_TM * B_F4 * R / P_F4 - 1


def test_f4_spectrometer_hits_lie_on_the_exact_circle(program, tmp_path):
    """Vacuum, Geant4's default integrator (the one the transport pins): the 20 hits against the exact crossings
    (field scale, lever arm, units of the hits) and the momentum from a circle fit to them. The hits drift smoothly
    from the circle by the integrator's own error (measured 5.9e-6 m over the 2 m, a fit bias of 4.8e-5), so the
    position keeps its ceiling of 1e-5 m and the momentum scale is held to the design's requirement on a
    spectrometer's scale bias, a tenth of its resolution: 3e-4 (design note §5.3, UC1; this resolution is 0.30 %)."""
    err, bias = f4_vacuum(program, tmp_path / "v")
    print(f"F4 vacuum, default integrator: max |y - y_exact| = {err:.2e} m (sagitta 22.5 mm); p_fit / p - 1 = "
          f"{bias:.2e}")
    assert err <= 1e-5
    assert abs(bias) <= 3e-4


def test_f4_with_a_tighter_integrator_the_scale_bias_is_the_integrators(program, tmp_path):
    """The same spectrometer with deltaOneStep 1e-6 m, deltaIntersection 1e-7 m and epsilon 5e-6 to 1e-4: the hits
    come within 2e-6 m of the exact crossings and the fitted momentum within 1e-5 (measured 2.1e-7 m and 7.2e-7), and
    the position error is at least 5 times below the default's (measured 28 times). So the default's bias is the
    integrator's, which a field scale, a lever arm or a unit error would not be."""
    err_d, _ = f4_vacuum(program, tmp_path / "default")
    err, bias = f4_vacuum(program, tmp_path / "tight", integrator=TIGHT)
    print(f"F4 vacuum, tight integrator: max |y - y_exact| = {err:.2e} m (default {err_d:.2e} m); p_fit / p - 1 = "
          f"{bias:.2e}")
    assert err <= 2e-6 and abs(bias) <= 1e-5
    assert err * 5 <= err_d


def test_f4_resolution_of_the_planes_is_gluckstern(program, tmp_path):
    """1e4 smearings of the vacuum hits by eps = 100 um: the relative spread of the fitted momentum agrees with the
    exact least-squares value within 2.8 % (4 sigma of a spread from 1e4 samples), which is within 1 % of
    Gluckstern's 0.30 % (PDG 2024 eq. 35.61)."""
    res = f4_run(program, tmp_path / "v", ref.VACUUM, 1)
    (x,) = f4_tracks(res)
    z, y = x[:, 2], x[:, 1]
    rng = np.random.default_rng(4242)
    c0 = ref.fit_parabola(z, y)[2]
    cs = np.array([ref.fit_parabola(z, y + rng.normal(0, EPS_F4, N_F4))[2] for _ in range(10000)])
    rel = np.std(cs / c0, ddof=1)
    lsq = ref.lsq_curvature_sigma(z, EPS_F4) / (2 * c0)
    g = ref.gluckstern_dp_over_p(EPS_F4, P_F4, B_F4, L_F4, N_F4)
    print(f"F4: measured {rel:.4%}, least squares {lsq:.4%}, Gluckstern {g:.4%}")
    assert lsq == pytest.approx(g, rel=0.01)
    assert rel == pytest.approx(lsq, rel=4 / math.sqrt(2 * 10000))


def test_f4_multiple_scattering_in_air_adds_the_pdg_term(program, tmp_path):
    """2 m of air: the fitted curvature of unsmeared hits spreads by the scattering term alone. Highland's theta_0
    (PDG 2024 eq. 34.16) is the width of a Gaussian fitted to the central 98 % of the projected angles (Lynch and
    Dahl 1991), not a standard deviation, which single scatterings inflate (measured: 0.144 % against a core of
    0.108 %); so the statistic is the same Gaussian fit to the central 98 % of the curvatures (the P5 estimator).
    Reference: PDG 2024 eq. 35.63 with Highland's logarithm at the whole length, 0.117 % (0.14 % without it).
    Tolerance (tier B): the note's 15 % (the formula is approximate: labelled so) plus 4 sigma of the core estimator
    from 1e4 events, 1.35 / sqrt(2N) each (bootstrap on a measured sample: 0.88 %; pure Gaussian: 0.95 %)."""
    N = 10000
    res = f4_run(program, tmp_path / "a", ref.AIR, N)
    tracks = f4_tracks(res)
    assert len(tracks) >= 0.99 * N
    c = np.array([ref.fit_parabola(x[:, 2], x[:, 1])[2] for x in tracks])
    core = ref.gaussian_core_sigma(c) / abs(np.mean(c))
    want = ref.ms_dp_over_p(P_F4, B_F4, L_F4, ref.X0_AIR_M)
    print(f"F4 air: scattering term (Gaussian core of the central 98 %) {core:.4%}, standard deviation "
          f"{np.std(c, ddof=1) / abs(np.mean(c)):.4%}, against PDG {want:.4%} with Highland's logarithm "
          f"({ref.ms_dp_over_p(P_F4, B_F4, L_F4, ref.X0_AIR_M, highland_log=False):.4%} without; approximate)")
    assert core == pytest.approx(want, rel=0.15 + 4 * 1.35 / math.sqrt(2 * len(c)))


# ============================================================================================ F5: outside the maps
def test_f5_a_track_leaving_its_map_is_refused_or_counted_never_silently_zero(program, tmp_path):
    B = np.zeros((11, 11, 3))
    B[..., 1] = 1.0                                                # B_z = 1 T for z <= 0 (+ one spacing, 0.05)
    write_map(tmp_path / "m", [{"name": "half", "kind": 1, "origin": [0.0, -0.5], "spacing": [0.05, 0.05], "B": B}])
    doc = vacuum_slabs([-0.5, 0.4, 0.4001, 0.5], ["a", "det", "a"], 0.5)
    f = [{"type": "map", "path": str(tmp_path / "m"), "patch": "half", "regions": ["a", "det"]}]
    kwargs = dict(solids=doc, fields=f, events=1, chunk_events=1,
                  sources=[gun("mu+", (0, 0, -0.45), [0.2, 0.0, 0.5], momentum_eV_c=math.hypot(0.2, 0.5) * 1e9)],
                  outputs={"d": {"hits": {"in": ["det"], "track": "primary"}}})
    with pytest.raises(Exception) as ei:
        run(tmp_path / "refuse", program, **kwargs)
    assert code_of(ei) == "particle.field_outside_map" and "half" in str(ei.value)
    with pytest.warns(particles.ParticleWarning, match=r"^particle\.field_outside_map"):
        res = run(tmp_path / "zero", program, outside="zero", **kwargs)
    ev_calls = chunk_arrays(tmp_path / "zero", 0)[0]["outside_calls"]
    assert int(ev_calls.sum()) > 0
    w = [x for x in res.record["warnings"] if x["code"] == "particle.field_outside_map"]
    assert w and str(int(ev_calls.sum())) in w[0]["message"]
    # the field ends at the box plus its margin: helix up to z = 0.05, straight after. A field that ends inside a
    # volume is a jump the integrator straddles (millirad errors, measured), so the direction is compared at 3e-2 rad,
    # against a turn of 0.3 rad up to z = 0.05 and a difference of 0.2 rad had the field continued
    h = primary_hits(res, "d")
    p = h["p_eV_c"][0]
    th_end = (0.05 + 0.45) * 0.2 / 0.5 / (0.2 / ref.GEV_PER_TM)          # turned angle up to z = 0.05
    d_exact = np.array([math.cos(th_end), -math.sin(th_end)])
    d_g4 = p[:2] / np.linalg.norm(p[:2])
    assert np.linalg.norm(d_g4 - d_exact) <= 3e-2
    th_cont = (0.4 + 0.45) * 0.2 / 0.5 / (0.2 / ref.GEV_PER_TM)
    assert np.linalg.norm(d_g4 - [math.cos(th_cont), -math.sin(th_cont)]) > 0.1


def test_f5_a_region_without_a_field_inherits_nothing_from_its_mother(program, tmp_path):
    """A daughter region served by no field (outside "zero") inside a mother with a uniform field: the track is
    straight inside the daughter (Geant4 hands a mother's field manager to daughters that have none)."""
    from _solids_ref import Partition
    Pt = Partition()
    pts = {"a": (0.0, -0.5), "b": (1.0, -0.5), "c": (1.0, 0.99), "d": (0.0, 0.99), "e": (0.0, 0.5), "f": (0.3, 0.5),
           "g": (0.3, 0.0), "h": (0.0, 0.0), "i": (1.0, 1.0), "j": (0.0, 1.0)}
    L = Pt.line
    ab, bc, cd, de = L(pts["a"], pts["b"]), L(pts["b"], pts["c"]), L(pts["c"], pts["d"]), L(pts["d"], pts["e"])
    ef, fg, gh, ha = L(pts["e"], pts["f"]), L(pts["f"], pts["g"]), L(pts["g"], pts["h"]), L(pts["h"], pts["a"])
    he = L(pts["h"], pts["e"])
    ci, ij, jd = L(pts["c"], pts["i"]), L(pts["i"], pts["j"]), L(pts["j"], pts["d"])
    Pt.face("outer", [ab, bc, cd, de, ef, fg, gh, ha])
    Pt.face("inner", [-gh, -fg, -ef, -he])
    Pt.face("det", [ci, ij, jd, -cd])
    regions = ["outer", "inner", "det"]
    doc = doc_of(Pt, regions, {"vac": ref.VACUUM}, {r: "vac" for r in regions})
    p = 0.5
    with pytest.warns(particles.ParticleWarning, match=r"^particle\.field_outside_map"):
        res = run(tmp_path / "w", program, solids=doc, outside="zero", events=1, chunk_events=1,
                  fields=[{"type": "uniform", "B_T": [0.5, 0.0, 0.0], "regions": ["outer", "det"]}],
                  sources=[gun("mu+", (0, 0, -0.4), [0.0, 0.0, 1.0], momentum_eV_c=p * 1e9)],
                  outputs={"d": {"hits": {"in": ["det"], "track": "primary"}}})
    h = primary_hits(res, "d")
    R = p / (ref.GEV_PER_TM * 0.5)
    # (y, z) plane; mu+ along +z in B along +x turns towards +y. Arc from z = -0.4 to 0, straight through the
    # daughter (z from 0 to 0.5), arc again up to the detector's face at z = 0.99.
    th = math.asin(0.4 / R)
    y1 = R * (1 - math.cos(th))
    y2 = y1 + 0.5 * math.tan(th)
    cy, cz = y2 + R * math.cos(th), 0.5 - R * math.sin(th)        # centre of the second arc
    y_exact = cy - math.sqrt(R * R - (0.99 - cz) ** 2)
    y_inherited = R - math.sqrt(R * R - (0.99 + 0.4) ** 2)          # one arc: the mother's field inside the daughter
    print(f"F5 daughter: y = {h['x_m'][0][1]:.6f} m, exact {y_exact:.6f}, with an inherited field {y_inherited:.6f}")
    assert h["x_m"][0][1] == pytest.approx(y_exact, abs=2e-5)
    assert abs(y_exact - y_inherited) > 0.1


# ============================================================================================ interfaces are not exits
SLAB_EDGES = [round(-0.2 + 0.01 * k, 6) for k in range(41)]
SLAB_NAMES = [("a" if k % 2 == 0 else "b") for k in range(40)]


def shell_doc(edges, names, materials, region_materials, rmax=0.5, t=1e-3):
    P, regions = ref.slabs_in_a_shell(edges, names, rmax, t)
    return doc_of(P, regions, materials, {r: region_materials(r) for r in regions}), regions


def tracks_of(h, sel):
    """Hit records of selected rows grouped by (chunk, event, track), each in step order."""
    out = {}
    for i in np.nonzero(sel)[0]:
        out.setdefault((int(h["chunk"][i]), int(h["event"][i]), int(h["track"][i])), []).append(i)
    return out


@pytest.mark.parametrize("b_field", ["field_and_none", "field_and_weaker_field"])
def test_every_proton_reaches_the_outer_surface_through_forty_interfaces(program, tmp_path, b_field):
    """Protons of 30 MeV/c, isotropic, on helices in vacuum through 40 slabs of 1 cm (1 T in every second slab, and
    0 T or 0.2 T in the others), inside a shell of 1 mm: an interface between two regions is not the domain's
    boundary, so every primary must reach the shell. A track that Geant4 places in the world for one zero-length step
    at such an interface was ended there and booked as leaving (measured in review: 50 and 56 of 5000 primaries,
    1 %; with 1000 events here, 10 +- 3 expected from that defect, none from a correct transport). Also no
    zero-length re-entry may record a second hit at one point (21 and 29 such repeats measured on the defect)."""
    other = [] if b_field == "field_and_none" else [{"type": "uniform", "B_T": [0.2, 0.0, 0.0], "regions": ["b"]}]
    doc, regions = shell_doc(SLAB_EDGES, SLAB_NAMES, {"vac": ref.VACUUM}, lambda r: "vac")
    res = run(tmp_path / "w", program, solids=doc, events=1000, chunk_events=500, seed=5,
              fields=[{"type": "uniform", "B_T": [1.0, 0.0, 0.0], "regions": ["a", "shell"]}] + other,
              outside="zero" if not other else "refuse",
              sources=[gun("proton", (0.0, 0.0, 0.0005), "isotropic", momentum_eV_c=30e6)],
              outputs={"all": {"hits": {"in": ["a", "b", "shell"], "track": "primary", "max_per_event": 100000}}})
    h = res.hits("all")
    shell = regions.index("shell")
    reached = {(int(c), int(e)) for c, e, r in zip(h["chunk"], h["event"], h["region"]) if r == shell}
    lost = [(c, e) for c in (0, 1) for e in range(500) if (c, e) not in reached]
    ex = np.concatenate([chunk_arrays(tmp_path / "w", k)[0]["exit_eV"] for k in (0, 1)])
    m_p = json.loads(subprocess.run([program, "--check"], capture_output=True, text=True,
                                    timeout=600).stdout)["particles"]["proton"]["mass_eV"]   # Geant4's mass
    ekin = math.hypot(30e6, m_p) - m_p
    dups = sum(int(np.sum(np.linalg.norm(np.diff(h["x_m"][rows], axis=0), axis=1) <= 1e-12))
               for rows in tracks_of(h, np.ones(len(h["event"]), bool)).values())
    print(f"{b_field}: {len(lost)} of 1000 primaries never reached the shell; {dups} hits repeat the track's previous "
          f"point; {len(h['event'])} hits")
    assert not lost and dups == 0, (f"primaries ended inside the domain: {lost[:10]}; hits repeated at one point "
                                    f"(zero-length re-entries): {dups}")
    assert np.allclose(ex, ekin, rtol=1e-12)


def test_no_shower_track_is_ended_at_an_interface_inside_the_domain(program, tmp_path):
    """200 electrons of 200 MeV into 20 iron plates of 2 mm with 2 mm vacuum gaps, 1 T everywhere, in a shell of
    1 mm. Per event, the energy booked as leaving the domain (exit_eV) cannot exceed the net kinetic energy carried
    into the shell (every entry into the shell, less every return from it into the inside, from the hits of all
    tracks): a track ended at an interior interface adds to exit_eV without crossing the shell (measured in review:
    769 tracks in 2000 events, 0.38 per event, so about 77 here; none from a correct transport)."""
    edges = [-0.01] + [round(0.002 * k, 6) for k in range(41)] + [0.5]
    names = ["front"] + [("fe" if k % 2 == 0 else "gap") for k in range(40)] + ["rear"]
    doc, regions = shell_doc(edges, names, {"vac": ref.VACUUM, "iron": ref.IRON},
                             lambda r: "iron" if r == "fe" else "vac")
    res = run(tmp_path / "w", program, solids=doc, events=200, chunk_events=100, seed=7,
              fields=[{"type": "uniform", "B_T": [1.0, 0.0, 0.0]}],
              sources=[gun("e-", (0.0, 0.0, -0.005), [0.0, 0.0, 1.0], momentum_eV_c=200e6)],
              outputs={"all": {"hits": {"in": regions, "track": "any", "max_per_event": 10 ** 6}}})
    h = res.hits("all")
    assert int(json.loads((tmp_path / "w" / "results.json").read_text())["outputs"]["all"]["dropped"]) == 0
    shell = regions.index("shell")
    net = {}
    for (c, e, t), rows in tracks_of(h, np.ones(len(h["event"]), bool)).items():
        for j, i in enumerate(rows):
            if h["region"][i] == shell:
                net[(c, e)] = net.get((c, e), 0.0) + h["ekin_eV"][i]
            elif j > 0 and h["region"][rows[j - 1]] == shell:
                net[(c, e)] = net.get((c, e), 0.0) - h["ekin_eV"][i]
    bad = []
    for c in (0, 1):
        ex = chunk_arrays(tmp_path / "w", c)[0]["exit_eV"]
        for e in range(100):
            if ex[e] > net.get((c, e), 0.0) + 1.0:
                bad.append((c, e, ex[e] - net.get((c, e), 0.0)))
    print(f"shower: {len(bad)} of 200 events book more energy as leaving than crosses the shell")
    assert not bad, f"energy booked as leaving without crossing the shell: {bad[:5]}"


def test_a_track_geant4_abandons_as_stuck_is_counted_and_the_chunk_completes(program, tmp_path):
    """A mu+ of 0.1 GeV/c sent along an interface where B_x reverses (-0.2 T below, +0.2 T above), crossing it at
    1 mrad, snakes along it until Geant4 abandons it as stuck (GeomNav0003, "not moving for 25 steps"). Such a track
    is ended and counted per event, as a looping track is (arrays ``abandoned_killed`` and ``abandoned_eV``; the
    warning ``particle.track_abandoned``), and the chunk completes: three identical events, all merged, each
    abandoning its primary, whose energy is booked there and not as leaving."""
    R = 0.1 / (ref.GEV_PER_TM * 0.2)
    z0 = R * (1 - math.cos(1e-3))
    P, regions = ref.slabs([-0.2, 0.0, 0.2], ["lo", "hi"], 0.5)
    doc = doc_of(P, regions, {"vac": ref.VACUUM}, {r: "vac" for r in regions})
    with pytest.warns(particles.ParticleWarning, match=r"^particle\.track_abandoned"):
        res = run(tmp_path / "w", program, solids=doc, events=3, chunk_events=3, seed=1,
                  fields=[{"type": "uniform", "B_T": [-0.2, 0.0, 0.0], "regions": ["lo"]},
                          {"type": "uniform", "B_T": [0.2, 0.0, 0.0], "regions": ["hi"]}],
                  sources=[gun("mu+", (0.0, -0.3, z0), [0.0, 1.0, 0.0], momentum_eV_c=0.1e9)],
                  outputs={"h": {"hits": {"in": ["lo", "hi"], "track": "primary", "max_per_event": 100000}}})
    assert res.partial is False and res.n_events == 3
    x, man = chunk_arrays(tmp_path / "w", 0)
    m = json.loads(subprocess.run([program, "--check"], capture_output=True, text=True,
                                  timeout=600).stdout)["particles"]["mu+"]["mass_eV"]
    ekin = math.hypot(0.1e9, m) - m
    print(f"stuck: abandoned per event {x['abandoned_killed'].tolist()}, abandoned_eV {x['abandoned_eV'].tolist()}, "
          f"exit_eV {x['exit_eV'].tolist()}, hits {len(res.hits('h')['event'])}")
    assert x["abandoned_killed"].dtype.str == "<u8" and x["abandoned_eV"].dtype.str == "<f8"
    assert np.all(x["abandoned_killed"] >= 1)
    assert np.allclose(x["abandoned_eV"], ekin, rtol=1e-9) and np.all(x["exit_eV"] == 0.0)
    assert int(man["counters"]["abandoned_killed"]) == int(x["abandoned_killed"].sum())
    assert any(w["code"] == "particle.track_abandoned" for w in res.record["warnings"])


def test_the_program_refuses_a_primary_vertex_outside_the_world(program, tmp_path):
    """A job file whose gun lies beyond the world (here 0.1 m below it) is refused by the program itself: exit status
    2, a message naming the position, nothing written (no Geant4 crash); a gun inside the world runs."""
    for z, ok in ((-0.6, False), (-0.4, True)):
        job, doc = geantino_job("out", [(0, 1, 3)])
        job["source"]["at_m"] = [0.0, 0.0, z]
        d = tmp_path / f"z{z}"
        d.mkdir()
        (d / "solids.json").write_text(json.dumps(doc))
        r = run_job(program, d, job)
        if ok:
            assert r.returncode == 0, r.stderr[-2000:]
        else:
            assert r.returncode == 2, (r.returncode, r.stderr[-2000:])
            assert "outside the world" in r.stderr
            assert not (d / "out" / "chunk_0").exists()


# ============================================================================================ P5: multiple scattering
@pytest.mark.parametrize("layers", [(("fe", 0.01),), (("fe", 0.005), ("pb", 0.0025))])
def test_p5_multiple_scattering_matches_highland(program, tmp_path, layers):
    edges, names = [-0.2, 0.0], ["vac"]
    z = 0.0
    for m, t in layers:
        names.append(m)
        z += t
        edges.append(z)
    edges += [z + 1e-3, z + 1.1e-3, 0.1]
    names += ["vac", "plane", "vac"]
    P, regions = ref.slabs(edges, names, 0.5)
    mats = {"vac": ref.VACUUM, "iron": ref.IRON, "lead": ref.LEAD}
    rm = {"vac": "vac", "plane": "vac", "fe": "iron", "pb": "lead"}
    doc = doc_of(P, regions, mats, {r: rm[r] for r in regions})
    N = 10000
    res = run(tmp_path / "w", program, solids=doc, events=N, chunk_events=500, seed=17,
              sources=[gun("mu-", (0, 0, -0.1), [0.0, 0.0, 1.0], momentum_eV_c=1e9)],
              outputs={"px": {"hit": {"in": "plane", "of": "x"}}, "pl": {"hits": {"in": ["plane"],
                                                                                    "track": "primary",
                                                                                    "max_per_event": 1}}})
    h = primary_hits(res, "pl")
    th = np.arctan2(h["p_eV_c"][:, 0], h["p_eV_c"][:, 2])
    assert len(th) >= 0.995 * N
    sigma = ref.gaussian_core_sigma(th)
    x_X0 = sum(t / (ref.X0_FE_M if m == "fe" else ref.X0_PB_M) for m, t in layers)
    want = ref.highland(1e9, ref.M_MU_EV, x_X0)
    print(f"P5 {layers}: theta_0 = {sigma * 1e3:.3f} mrad, Highland {want * 1e3:.3f} mrad (x/X0 = {x_X0:.3f})")
    if len(layers) == 1:
        assert want == pytest.approx(10.1e-3, rel=0.01)
    assert sigma == pytest.approx(want, rel=0.10)


# ============================================================================================ S1: bit for bit
def busy_doc():
    P, regions = ref.slabs([-0.2, 0.0, 0.02, 0.1, 0.101, 0.2], ["vac", "iron", "vac", "layer", "vac"], 0.3)
    return doc_of(P, regions, {"vac": ref.VACUUM, "iron": ref.IRON},
                  {"vac": "vac", "iron": "iron", "layer": "vac"})


GRID_FE = {"type": "cylindrical", "r_m": [0.0, 0.3], "z_m": [0.0, 0.02], "bins": [6, 4]}
BUSY_OUT = {"E": {"deposit": {"in": "iron"}}, "Em": {"mean": {"of": "E"}}, "Es": {"std": {"of": "E"}},
            "q": {"tally": {"grid": GRID_FE, "in": ["iron"]}},
            "trk": {"hits": {"in": ["layer"], "max_per_event": 3}}}


def busy(wd, program, **kw):
    k = dict(solids=busy_doc(), events=1000, chunk_events=200, seed=7, outputs=copy.deepcopy(BUSY_OUT),
             fields=[{"type": "uniform", "B_T": [1.0, 0.0, 0.0]}],
             sources=[gun("e-", (0, 0, -0.1), [0.0, 0.0, 1.0], energy_eV=2e8)])
    k.update(kw)
    return run(wd, program, **k)


@pytest.fixture(scope="module")
def busy_runs(program, tmp_path_factory):
    base = tmp_path_factory.mktemp("busy")
    out = {"A": busy(base / "A", program), "B": busy(base / "B", program, threads=4),
           "C": busy(base / "C", program, jobs=3, workers=3), "D": busy(base / "D", program)}
    busy(base / "F", program, events=700)
    out["F"] = particles.extend(base / "F", events=1000, workers=2)
    out["base"] = base
    return out


def results_of(wd):
    return strip(json.loads((Path(wd) / "results.json").read_text()))


@pytest.mark.parametrize("other", ["B", "C", "D", "F"])
def test_s1_merged_results_are_bitwise_equal(busy_runs, other):
    """Same seed twice (D); 1 against 4 threads (B); 1 against 3 processes (C); a run extended from a partial
    chunk (F): every chunk array, the merged tallies and results.json, bit for bit."""
    base = busy_runs["base"]
    assert_chunks_bitwise(base / "A", base / other, range(5), f"A vs {other}")
    assert results_of(base / "A") == results_of(base / other)
    for n in ("lo", "hi", "events"):
        a = np.load(base / "A" / "merged" / f"tally_q_{n}.npy")
        b = np.load(base / other / "merged" / f"tally_q_{n}.npy")
        assert a.tobytes() == b.tobytes(), n


def test_s1_the_busy_case_exercises_what_it_claims(busy_runs):
    a = busy_runs["A"]
    x, man = chunk_arrays(busy_runs["base"] / "A", 0)
    assert x["hits_trk_dropped"].sum() > 0, "no event reached the per-event limit"
    assert np.all(x["hits_trk_count"] <= 3)
    assert x["n_tracks"].max() > 10 and x["field_calls"].sum() > 0
    assert np.array_equal(x["event"], np.arange(200))
    assert x["seed_words"].tolist() == [ref.event_words(ref.chunk_seed(7, 0), e) for e in range(200)]
    assert man["integrator"]["stepper"] == "G4TDormandPrince45+G4IntegrationDriver"
    assert float(a["Em"]) > 0 and a["Es"].stat_error is None          # 5 chunks: too few for batch means
    # the tally against the per-event deposits: sum of rounded bin totals within half a quantum per touched bin
    t = a.tally("q")
    E = a.events()["E"]
    assert abs(float(np.sum(t["edep_eV"])) * 1000 - E.sum()) <= 1000 * 24 * 0.5e-3 + 1e-9 * E.sum()


def test_s1_a_chunk_alone_equals_the_same_chunk_after_others(busy_runs, program):
    base = busy_runs["base"]
    job = json.loads(next((base / "A" / "exchange").glob("job_*.json")).read_text())
    job["chunks"] = [c for c in job["chunks"] if c["index"] == 3]
    job["out"] = str(base / "alone" / "out")
    job["record"] = str(base / "alone" / "out" / "record.json")
    for key in ("solids",):
        job[key] = str((base / "A" / "exchange" / job[key]).resolve())
    r = run_job(program, base / "alone", job, threads=2)
    assert r.returncode == 0, r.stderr[-2000:]
    x, _ = chunk_arrays(base / "alone", 3)
    y, _ = chunk_arrays(base / "A", 3)
    for n in y:
        assert x[n].tobytes() == y[n].tobytes(), n


def geantino_job(out, chunks, direction=(0.0, 0.0, 1.0)):
    P, regions = ref.slabs([-0.5, 0.0, 0.5], ["vac", "vac2"], 0.5)
    return {"schema": "axionmultiphysics.transport-job/1", "mode": "transport", "out": str(out),
            "chunks": [{"index": k, "seed": s, "events": n} for k, s, n in chunks], "physics_list": "FTFP_BERT",
            "production_cut_m": 7e-4, "solids": "solids.json", "fields": [], "outside": "refuse",
            "integrator": dict(particles.INTEGRATOR),
            "source": {"type": "gun", "particle": "geantino", "energy_eV": 1e9, "at_m": [0.0, 0.0, -0.4],
                       "direction": list(direction), "spot_sigma_m": 0.0},
            "quantum_eV": 1e-3, "tallies": [], "hits": []}, doc_of(P, regions, {"vac": ref.VACUUM},
                                                                    {"vac": "vac", "vac2": "vac"})


def test_s1_the_first_draw_equals_the_mixmax_reference_also_beyond_10000_events(program, tmp_path):
    """The reseed sits at the top of GeneratePrimaries in this mode too: the first number of each event equals the hello mode's
    references (MixMax seeded with the event's words, recorded outside the repository), including events after
    Geant4's own refill of its seeds at 10 000; the 25 000-event chunk is bitwise equal for 1 and 4 threads."""
    long_seed = 0xAB54A98CEB1F0AD2
    chunks = [(0, 1, 200), (1, 0xFEDCBA9876543210, 200), (2, 0xDEADBEEF, 200), (3, long_seed, 25000)]
    for threads, name in ((1, "t1"), (4, "t4")):
        job, doc = geantino_job("out", chunks)
        (tmp_path / name).mkdir()
        (tmp_path / name / "solids.json").write_text(json.dumps(doc))
        r = run_job(program, tmp_path / name, job, threads=threads)
        assert r.returncode == 0, r.stderr[-2000:]
    seeds = {0: 1, 1: 0xFEDCBA9876543210, 2: 0xDEADBEEF, 3: long_seed}
    checked = 0
    for k, s in seeds.items():
        a, _ = chunk_arrays(tmp_path / "t1", k)
        b, _ = chunk_arrays(tmp_path / "t4", k)
        for n in a:
            assert a[n].tobytes() == b[n].tobytes(), (k, n)
        for (cs, e), bits in hello.FIRST_FLAT.items():
            if cs == s:
                assert a["first_flat"][e].view(np.uint64) == bits, (hex(cs), e)
                assert a["seed_words"][e].tolist() == ref.event_words(cs, e)
                checked += 1
    assert checked == len(hello.FIRST_FLAT)


def test_s1_a_stop_rule_gives_the_same_result_with_1_and_4_workers(program, tmp_path):
    out = {}
    for w in (1, 4):
        r = busy(tmp_path / f"w{w}", program, events=1600, chunk_events=20, jobs=8, workers=w,
                 stop={"output": "Em", "rel_error": 0.02}, outputs={"E": {"deposit": {"in": "iron"}},
                                                                     "Em": {"mean": {"of": "E"}}})
        out[w] = (r.record["stop"], results_of(tmp_path / f"w{w}"))
    assert out[1] == out[4]
    st = out[1][0]
    assert st["chunks"] >= 20 and st["events"] == 20 * st["chunks"]


def test_s1_a_tally_beyond_2_to_the_63_quanta_equals_its_exact_integer(program, tmp_path):
    """81 one-event chunks of a 1 GeV electron showering in 20 cm of lead (35 X0); a quantum that puts the primary's
    total energy at 2^63 / 1.1 quanta: batch 0 holds chunks 0, 20, 40, 60 and 80, so its sum exceeds 2^63 whenever the
    shower deposits more than a quarter of its energy, and must equal the exact sum of the chunks."""
    P, regions = ref.slabs([-0.1, 0.0, 0.2, 0.3], ["vac", "lead", "vac"], 0.3)
    doc = doc_of(P, regions, {"vac": ref.VACUUM, "lead": ref.LEAD}, {"vac": "vac", "lead": "lead"})
    E = 1e9
    q = (E + 0.51099895e6) / 2.0 ** 63 * 1.1
    grid = {"type": "cylindrical", "r_m": [0.0, 0.3], "z_m": [0.0, 0.2], "bins": [1, 1]}
    run(tmp_path / "w", program, solids=doc, events=81, chunk_events=1, quantum_eV=q, jobs=2, workers=2, seed=9,
        sources=[gun("e-", (0, 0, -0.05), [0.0, 0.0, 1.0], energy_eV=E)],
        outputs={"q": {"tally": {"grid": grid, "in": ["lead"]}}})
    lo = np.load(tmp_path / "w" / "merged" / "tally_q_lo.npy")
    hi = np.load(tmp_path / "w" / "merged" / "tally_q_hi.npy")
    exact = {}
    for k in range(81):
        exact[k % 20] = exact.get(k % 20, 0) + int(chunk_arrays(tmp_path / "w", k)[0]["tally_q"].reshape(-1)[0])
    got = {b: (int(hi[b].reshape(-1)[0]) << 64) + int(lo[b].reshape(-1)[0]) for b in range(20)}
    assert got == exact
    assert max(got.values()) > 2 ** 63, "the case does not pass 2^63"


# ============================================================================================ S2, S3, S4
def test_s2_independent_seeds_agree_within_their_errors(program, tmp_path):
    P, regions = ref.slabs([-0.1, 0.0, 0.01, 0.1], ["vac", "iron", "vac"], 0.3)
    doc = doc_of(P, regions, {"vac": ref.VACUUM, "iron": ref.IRON}, {"vac": "vac", "iron": "iron"})
    m, s = [], []
    for seed in range(1, 11):
        r = run(tmp_path / f"s{seed}", program, solids=doc, events=2000, chunk_events=100, seed=seed,
                sources=[gun("mu-", (0, 0, -0.05), [0.0, 0.0, 1.0], momentum_eV_c=1e9)],
                outputs={"E": {"deposit": {"in": "iron"}}, "Em": {"mean": {"of": "E"}}})
        m.append(float(r["Em"]))
        s.append(r["Em"].stat_error)
    m, s = np.array(m), np.array(s)
    w = 1 / s ** 2
    mbar = np.sum(w * m) / np.sum(w)
    chi2 = float(np.sum(((m - mbar) / s) ** 2))
    lo, hi = stats.chi2.cdf(chi2, 9), stats.chi2.sf(chi2, 9)
    print(f"S2: chi2 = {chi2:.2f} for 9 dof; P(<) = {lo:.3g}, P(>) = {hi:.3g}")
    assert lo > 1e-4 and hi > 1e-4
    assert len(set(m.tolist())) == 10


def cone_doc():
    r_disk = math.tan(math.radians(10.0))
    P, regions = ref.slabs([-0.5, 1.0, 1.0 + 1e-6, 1.5], ["vac", "disk", "vac"], r_disk)
    return doc_of(P, regions, {"vac": ref.VACUUM}, {"vac": "vac", "disk": "vac"})


@pytest.mark.skipif(not SLOW, reason="S3 transports 1e7 geantinos (minutes on 8 cores); set AXMP_SLOW=1")
def test_s3_the_estimators_cover_the_acceptance_of_a_cone(program, tmp_path):
    """One run of 1000 chunks of 1e4 isotropic geantinos (8 processes); each chunk an experiment: the 1 sigma
    interval of `mean` covers (1 - cos 10 deg) / 2 in 68.3 +- 5.9 % of the chunks, Clopper-Pearson's in at least
    62.4 %."""
    truth = (1 - math.cos(math.radians(10.0))) / 2
    res = run(tmp_path / "w", program, solids=cone_doc(), events=10 ** 7, chunk_events=10 ** 4, jobs=8, workers=8,
              seed=2024, sources=[gun("geantino", (0, 0, 0), "isotropic", energy_eV=1e6)],
              outputs={"hz": {"hit": {"in": "disk", "of": "z"}}, "acc": {"fraction": {"where": {"hz": {"defined": True}}}}})
    ev = res.events(["hz"])
    hit = ~np.isnan(ev["hz"])
    cov_m = cov_cp = 0
    for k in range(1000):
        x = hit[ev["chunk"] == k].astype(float)
        mm = particles.estimate_mean(x)
        cov_m += abs(float(mm) - truth) <= mm.stat_error
        f = particles.estimate_fraction(int(x.sum()), len(x))
        cov_cp += f.interval[0] <= truth <= f.interval[1]
    print(f"S3: coverage mean {cov_m / 10:.1f} %, Clopper-Pearson {cov_cp / 10:.1f} %; acceptance {float(res['acc']):.5e}")
    assert abs(cov_m / 1000 - 0.683) <= 0.059
    assert cov_cp / 1000 >= 0.683 - 0.059
    assert abs(float(res["acc"]) - truth) <= 4 * res["acc"].stat_error


def test_s4_no_hit_is_an_upper_limit(program, tmp_path):
    res = run(tmp_path / "w", program, solids=cone_doc(), events=1000, chunk_events=100,
              sources=[gun("geantino", (0, 0, 0), [0.0, 0.0, -1.0], energy_eV=1e6)],
              outputs={"hz": {"hit": {"in": "disk", "of": "z"}}, "acc": {"fraction": {"where": {"hz": {"defined": True}}}}})
    a = res["acc"]
    assert float(a) == 0.0 and a.stat_error is None
    assert a.upper_limit == pytest.approx(ref.zero_fraction_limit(1000), rel=1e-12)
    assert str(a).startswith("< ")


# ============================================================================================ sources and lists
def test_the_gun_samples_its_spot_and_isotropic_directions(program, tmp_path):
    res = run(tmp_path / "spot", program, solids=cone_doc(), events=4000, chunk_events=1000,
              sources=[gun("geantino", (0.01, -0.02, 0.0), [0.0, 0.0, 1.0], energy_eV=1e6, spot_sigma_m=0.003)],
              outputs={"x": {"primary": {"of": "x"}}, "y": {"primary": {"of": "y"}}, "z": {"primary": {"of": "z"}}})
    ev = res.events()
    assert np.all(ev["z"] == 0.0)
    for v, mu in ((ev["x"], 0.01), (ev["y"], -0.02)):
        assert stats.kstest((v - mu) / 0.003, "norm").pvalue > 1e-4
    res = run(tmp_path / "iso", program, solids=cone_doc(), events=4000, chunk_events=1000,
              sources=[gun("geantino", (0, 0, 0), "isotropic", energy_eV=1e6)],
              outputs={"px": {"primary": {"of": "px"}}, "py": {"primary": {"of": "py"}},
                       "pz": {"primary": {"of": "pz"}}})
    ev = res.events()
    p = np.column_stack([ev["px"], ev["py"], ev["pz"]])
    pn = np.linalg.norm(p, axis=1)
    assert np.allclose(pn, 1e6, rtol=1e-12)
    assert stats.kstest((p[:, 2] / pn + 1) / 2, "uniform").pvalue > 1e-4
    assert stats.kstest((np.arctan2(p[:, 1], p[:, 0]) + math.pi) / (2 * math.pi), "uniform").pvalue > 1e-4


def test_momentum_and_kinetic_energy_are_converted_with_geant4s_mass(program, tmp_path):
    """A gun given a momentum gets E_kin = hypot(p, m) - m with the mass of Geant4's particle table, which the program
    reports (``--check``): Geant4 11.4.3's muon is 105.6583715 MeV, PDG 2024's 105.6583755 MeV, 3.8e-8 apart, which
    moves E_kin by 1.6e-9 at 2.5 GeV/c. The conversion is checked to 1e-12 against the program's own mass, and that
    mass against PDG's within 1e-7 (the table's mass is the muon's, in eV)."""
    chk = json.loads(subprocess.run([program, "--check"], capture_output=True, text=True, timeout=600).stdout)
    m = chk["particles"]["mu+"]["mass_eV"]
    assert m == pytest.approx(ref.M_MU_EV, rel=1e-7)
    res = run(tmp_path / "w", program, solids=cone_doc(), events=1, chunk_events=1,
              sources=[gun("mu+", (0, 0, 0), [0.0, 0.0, 1.0], momentum_eV_c=2.5e9)],
              outputs={"E": {"primary": {"of": "E_kin"}}, "pz": {"primary": {"of": "pz"}}})
    ev = res.events()
    assert ev["pz"][0] == pytest.approx(2.5e9, rel=1e-12)
    assert ev["E"][0] == pytest.approx(math.hypot(2.5e9, m) - m, rel=1e-12)


@pytest.mark.parametrize("plist", ["FTFP_BERT", "FTFP_BERT_EMZ", "QGSP_BIC"])
def test_each_offered_physics_list_runs_and_is_recorded(program, tmp_path, plist):
    res = run(tmp_path / "w", program, solids=busy_doc(), events=20, chunk_events=10, physics_list=plist,
              sources=[gun("mu-", (0, 0, -0.1), [0.0, 0.0, 1.0], momentum_eV_c=1e9)],
              outputs={"E": {"deposit": {"in": "iron"}}, "Em": {"mean": {"of": "E"}}})
    assert chunk_arrays(tmp_path / "w", 0)[1]["physics_list"] == plist
    assert res.record["physics_list"] == plist
    assert 5e6 < float(res["Em"]) < 5e7        # about 13 MeV in 1 cm of iron... 2 cm here: tens of MeV


def test_the_record_carries_the_programs_the_maps_and_the_honesty_sentence(program, tmp_path):
    B = np.zeros((2, 2, 3))
    B[..., 1] = 0.5
    write_map(tmp_path / "m", [{"name": "u", "kind": 1, "origin": [0.0, -0.3], "spacing": [0.4, 0.6], "B": B}])
    res = busy(tmp_path / "w", program, events=40, chunk_events=20,
               fields=[{"type": "map", "path": str(tmp_path / "m"), "patch": "u",
                        "regions": ["vac", "iron", "layer"]}])
    rec = res.record
    prog = rec["programs"]["axmp-transport"]
    assert prog["geant4_version"] == "11.4.3" and prog["geant4_tested"] is True and prog["program_version"]
    assert rec["fields"][0]["patch"] == "u"
    assert rec["integrator"]["stepper"] == "G4TDormandPrince45+G4IntegrationDriver"
    out = json.loads((tmp_path / "w" / "results.json").read_text())
    assert out["mc"].startswith("Monte Carlo (beta): 40 events, seed 7;")
    lp = chunk_arrays(tmp_path / "w", 0)[0]
    for name in ("looping_killed", "looping_eV", "exit_eV", "outside_calls", "primary", "edep_eV"):
        assert name in lp, name
    assert lp["edep_eV"].shape == (20, len(busy_doc()["regions"]))
