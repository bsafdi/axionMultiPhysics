"""The first demonstration of the particle transport (design note §14: "a solved solenoid and a 3D iron dipole ->
muons -> tracker -> momentum resolution and scale"), on the API route ``particles.transport(design, results, ...)``.

- **Solved solenoid** (runs now): a solenoid (R 0.5 m, H 1 m, winding 0.05 m, 2e5 ampere-turns, domain radius 1.5 m)
  with eight air layers 2 mm thick at r = 0.10 to 0.38 m as the tracker, solved by the package, its field exported
  as a map (``Results.export_field_map``) with a track volume over the tracker. The mesher does not yet export the
  design's partition, so the partition is written here by hand with the same regions; mu+ at 0.2 GeV/c leave the
  centre along +x and cross the layers in the bore field (about 0.18 T; radius of curvature about 3.7 m; sagitta over
  the tracker about 3 mm). Checks: the hits follow the map's own trajectory (tier A); the momentum scale against the
  exact closed-form field is within the map-plus-solve error measured along the path; the resolution from 50 um
  smearing is the least-squares value; the Gaussian core (central 98 %) of the scattering term in air agrees with PDG 2024
  eq. 35.63 (with Highland's logarithm) within 30 % (the formula is for many planes; eight here: labelled approximate).
- **3D iron dipole** (skips until ``transport()`` exports a 3D design's partition itself and ``capabilities()`` says
  so): a window-frame H dipole of iron (mu_r 1000), two loop coils, a 0.1 m gap; tracker planes upstream and
  downstream; mu+ at 2 GeV/c; momentum from the change of direction against the field integral of the map.

Needs ``AXMP_TRANSPORT`` and a build that solves designs (the mesher). The numbers printed here go to
docs/validation/particles.md. The map tolerance is 1e-3 (1e-4 with ``AXMP_SLOW=1``).

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import axionmultiphysics as axmp
from axionmultiphysics import particles  # noqa: E402
from axionmultiphysics import solids as solids_mod  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _particles_ref as ref  # noqa: E402

pytestmark = pytest.mark.px_transport

SLOW = bool(os.environ.get("AXMP_SLOW"))
SOL = dict(R=0.5, H=1.0, tw=0.05, NI=2.0e5)
LAYERS = [0.10 + 0.04 * i for i in range(8)]
T_LAYER = 0.002
P_DEMO = 0.2                      # GeV/c
EPS = 50e-6


@pytest.fixture(scope="module")
def program():
    exe = os.environ.get("AXMP_TRANSPORT")
    if not exe:
        pytest.skip("axmp-transport is not built here: set AXMP_TRANSPORT")
    if not (Path(exe).is_file() and os.access(exe, os.X_OK)):
        pytest.fail(f"AXMP_TRANSPORT={exe} is not an executable file")
    from axionmultiphysics import _paths
    if _paths.mesher_path() is None:
        pytest.skip("the demonstration solves a design: needs axmp-mesher (AXMP_MESHER)")
    return exe


def solenoid_design():
    regions = [{"name": "winding", "material": "winding", "shape": {"rect": {"r": [0.5, 0.55], "z": [-0.5, 0.5]}}}]
    for i, r in enumerate(LAYERS):
        regions.append({"name": f"layer_{i}", "material": "air",
                        "shape": {"rect": {"r": [r, r + T_LAYER], "z": [-0.4, 0.4]}}})
    return {"schema": "axionmultiphysics.design/1", "name": "px4_solenoid_spectrometer", "regions": regions,
            "windings": [{"name": "magnet", "region": "winding", "type": "azimuthal", "ampere_turns": SOL["NI"]}],
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02, "domain_radius": 1.5}}


def solenoid_solids(air_block):
    rects = [("winding", 0.5, 0.55, -0.5, 0.5)] + [(f"layer_{i}", r, r + T_LAYER, -0.4, 0.4)
                                                    for i, r in enumerate(LAYERS)]
    P, regions = ref.rect_holes_in_half_disk(1.5, rects)
    mats = {"gas": air_block, "copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}}}
    rm = {r: ("copper" if r == "winding" else "gas") for r in regions}
    return solids_mod.from_partition(P.table(regions), mats, rm)


@pytest.fixture(scope="module")
def solenoid(program, tmp_path_factory):
    base = tmp_path_factory.mktemp("solenoid")
    design = axmp.Design(solenoid_design())
    res = design.solve(workdir=str(base / "solve"), verbose=0)
    tol = 1e-4 if SLOW else 1e-3
    res.export_field_map(base / "map", tol=tol, track_volumes=[{"name": "tracker", "r": [0.0, 0.42],
                                                                "z": [-0.05, 0.05]}])
    return program, base, design, res


def demo_run(sol, name, air_block, events, seed=1):
    program, base, design, res = sol
    return particles.transport(design, res, solids=solenoid_solids(air_block), field_map=base / "map",
                               sources=[{"type": "gun", "particle": "mu+", "momentum_eV_c": P_DEMO * 1e9,
                                         "at_m": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]}],
                               outputs={"trk": {"hits": {"in": [f"layer_{i}" for i in range(8)], "track": "primary",
                                                         "max_per_event": 20}}},
                               events=events, chunk_events=max(1, events // 20), seed=seed,
                               workdir=base / name, program=program)


def exact_bz(x):
    r = math.hypot(x[0], x[1])
    br, bz = ref.solenoid_field(r, x[2], SOL["R"], SOL["H"], SOL["tw"], SOL["NI"])
    return np.array([float(br) * x[0] / r if r else 0.0, float(br) * x[1] / r if r else 0.0, float(bz)])


def layer_events():
    ev = [(lambda s, y, r=r: math.hypot(y[0], y[1]) - r) for r in LAYERS]
    ev[-1].terminal = True              # the path ends at the last layer (beyond it lies the winding)
    return ev


def reference_hits(field):
    hits, sol = ref.rk_trajectory(field, [0.0, 0.0, 0.0], [P_DEMO * 1e9, 0.0, 0.0], 1.0, 0.6, layer_events(),
                                  rtol=1e-11, atol=1e-13)
    hits.sort(key=lambda h: math.hypot(h[1][0], h[1][1]))
    return np.array([h[1][:3] for h in hits]), sol


def hits_of_event(res, k=0, e=0):
    h = res.hits("trk")
    sel = (h["chunk"] == k) & (h["event"] == e) & (h["track"] == 1)
    x = h["x_m"][sel]
    return x[np.argsort(np.hypot(x[:, 0], x[:, 1]))]


def test_the_demo_hits_follow_the_maps_trajectory_and_the_scale_is_the_maps(solenoid):
    from axionmultiphysics import fieldmap
    program, base, design, res = solenoid
    r = demo_run(solenoid, "vacuum", ref.VACUUM, 1)
    g4 = hits_of_event(r)
    assert len(g4) == 8
    fm = fieldmap.load(base / "map")
    (patch,) = [p.name for p in fm.patches]
    mapped, _ = reference_hits(lambda x: fm.evaluate([x], patch=patch)[0])
    exact, sol = reference_hits(exact_bz)
    err = np.max(np.linalg.norm(g4 - mapped, axis=1))
    print(f"demo: hits against the map's own trajectory: max {err:.2e} m")
    assert err <= 2e-5
    # the map plus the solve, against the closed form, along the path through the tracker
    s = np.linspace(0.0, sol.t[-1], 400)
    path = np.array([np.interp(s, sol.t, sol.y[i]) for i in range(3)]).T
    bm = np.array([fm.evaluate([p], patch=patch)[0][2] for p in path])
    be = np.array([exact_bz(p)[2] for p in path])
    beta = float(np.trapezoid(np.abs(bm - be), s) / np.trapezoid(np.abs(be), s))
    R_g4 = ref.fit_circle(g4[:, 0], g4[:, 1])[2]
    R_ex = ref.fit_circle(exact[:, 0], exact[:, 1])[2]
    sel = (np.hypot(path[:, 0], path[:, 1]) >= LAYERS[0]) & (np.hypot(path[:, 0], path[:, 1]) <= LAYERS[-1])
    B_eff = float(np.mean(be[sel]))
    scale = ref.GEV_PER_TM * B_eff * R_g4 / P_DEMO
    print(f"demo: momentum scale {scale:.6f} with the path-averaged exact field {B_eff:.5f} T; "
          f"R(map) / R(exact) - 1 = {R_g4 / R_ex - 1:.2e}; map and solve against the closed form: {beta:.2e}")
    assert abs(R_g4 / R_ex - 1) <= 2 * beta + 2e-5
    assert abs(scale - 1) <= 0.02, "the circle fit with the path-averaged field is a 2 % model at most here"


def test_the_demo_resolution_is_the_least_squares_value(solenoid):
    r = demo_run(solenoid, "vacuum_res", ref.VACUUM, 1)
    x = hits_of_event(r)
    rng = np.random.default_rng(77)
    c0 = ref.fit_parabola(x[:, 0], x[:, 1])[2]
    cs = np.array([ref.fit_parabola(x[:, 0], x[:, 1] + rng.normal(0, EPS, len(x)))[2] for _ in range(10000)])
    rel = np.std(cs / c0, ddof=1)
    lsq = ref.lsq_curvature_sigma(x[:, 0], EPS) / abs(2 * c0)
    B_eff = 2 * abs(c0) * P_DEMO / ref.GEV_PER_TM
    g = ref.gluckstern_dp_over_p(EPS, P_DEMO, B_eff, LAYERS[-1] - LAYERS[0], len(x))
    print(f"demo: resolution {rel:.3%} (least squares {lsq:.3%}, Gluckstern {g:.3%}) at 50 um, 8 layers")
    assert rel == pytest.approx(lsq, rel=4 / math.sqrt(2 * 10000))
    assert lsq == pytest.approx(g, rel=0.05)


def test_the_demo_scattering_term_in_air(solenoid):
    N = 4000
    r = demo_run(solenoid, "air", ref.AIR, N, seed=3)
    h = r.hits("trk")
    prim = h["track"] == 1
    key = h["chunk"][prim].astype(np.int64) * 10 ** 9 + h["event"][prim].astype(np.int64)
    xs = h["x_m"][prim]
    cs = []
    for k in np.unique(key):
        x = xs[key == k]
        if len(x) == 8:
            cs.append(ref.fit_parabola(x[:, 0], x[:, 1])[2])
    cs = np.array(cs)
    assert len(cs) >= 0.95 * N
    # Highland's width is that of a Gaussian fitted to the central 98 % (as in the spectrometer row F4)
    rel = ref.gaussian_core_sigma(cs) / abs(np.mean(cs))
    B_eff = 2 * abs(np.mean(cs)) * P_DEMO / ref.GEV_PER_TM
    beta = P_DEMO / math.hypot(P_DEMO, ref.M_MU_EV * 1e-9)
    want = ref.ms_dp_over_p(P_DEMO, B_eff, LAYERS[-1] - LAYERS[0], ref.X0_AIR_M, beta)
    print(f"demo: scattering term in air {rel:.3%} (Gaussian core of the central 98 %; standard deviation "
          f"{np.std(cs, ddof=1) / abs(np.mean(cs)):.3%}) against PDG {want:.3%} (approximate)")
    assert rel == pytest.approx(want, rel=0.30)


# ============================================================================================ the 3D dipole
def dipole_design():
    def box(lo, hi):
        return {"box": {"min": lo, "max": hi}}
    yoke = {"difference": [box([-0.35, -0.3, -0.2], [0.35, 0.3, 0.2]), box([-0.25, -0.2, -0.3], [0.25, 0.2, 0.3])]}
    regions = [{"name": "yoke", "material": "iron", "solid": yoke},
               {"name": "pole_top", "material": "iron", "solid": box([-0.1, 0.05, -0.2], [0.1, 0.2, 0.2])},
               {"name": "pole_bottom", "material": "iron", "solid": box([-0.1, -0.2, -0.2], [0.1, -0.05, 0.2])}]
    for name, y0, y1 in (("coil_top", 0.06, 0.14), ("coil_bottom", -0.14, -0.06)):
        regions.append({"name": name, "material": "winding",
                        "solid": {"difference": [box([-0.18, y0, -0.28], [0.18, y1, 0.28]),
                                                 box([-0.11, y0 - 0.01, -0.21], [0.11, y1 + 0.01, 0.21])]}})
    for i, z in enumerate((-1.2, -1.0, -0.8, 0.8, 1.0, 1.2)):
        regions.append({"name": f"plane_{i}", "material": "air", "solid": box([-0.4, -0.4, z], [0.4, 0.4, z + 0.002])})
    return {"schema": "axionmultiphysics.design/1", "name": "px4_iron_dipole", "symmetry": "3d", "regions": regions,
            "materials": {"iron": {"kind": "linear", "mu_r": 1000}},
            "windings": [{"name": n, "region": n, "type": "loop", "ampere_turns": 2.0e4, "axis": {"direction": [0, 1, 0]}}
                         for n in ("coil_top", "coil_bottom")],
            "mesh": {"order": 2, "size": 0.12, "surface_size": 0.03, "domain_radius": 2.2}}


def test_the_iron_dipole_measures_momentum_from_the_bend(tmp_path):
    caps = particles.capabilities()
    if not caps["partition_export"]["3d"]:
        pytest.skip("transport() does not export a 3D partition by itself yet "
                    "(capabilities()['partition_export']['3d'])")
    if not caps["transport"]["ok"]:
        pytest.skip("axmp-transport is not available")
    if not axmp.build_info().get("engine_3d"):
        pytest.skip("a 3D solve needs the MPI build")
    if not SLOW:
        pytest.skip("a 3D solve and a kind-3 map (minutes); set AXMP_SLOW=1")
    design = axmp.Design(dipole_design())
    res = design.solve(workdir=str(tmp_path / "solve"), verbose=0)
    comp = {"iron": {"nist": "G4_Fe", "density": 7874.0}, "winding": {"nist": "G4_Cu", "density": 8960.0},
            "air": {"nist": "G4_Galactic", "density": 1e-22}}
    P = 2.0
    r = particles.transport(design, res, composition=comp, map_tol=1e-3,
                            track_volumes=[{"name": "gap", "x": [-0.05, 0.05], "y": [-0.03, 0.03], "z": [-1.5, 1.5]}],
                            sources=[{"type": "gun", "particle": "mu+", "momentum_eV_c": P * 1e9,
                                      "at_m": [0.0, 0.0, -1.5], "direction": [0.0, 0.0, 1.0]}],
                            outputs={"trk": {"hits": {"in": [f"plane_{i}" for i in range(6)], "track": "primary"}}},
                            events=1, chunk_events=1, workdir=tmp_path / "run")
    h = r.hits("trk")
    x = h["x_m"][np.argsort(h["x_m"][:, 2])]
    assert len(x) == 6
    up = np.polyfit(x[:3, 2], x[:3, 0], 1)[0]
    down = np.polyfit(x[3:, 2], x[3:, 0], 1)[0]
    dtheta = abs(math.atan(down) - math.atan(up))
    zz = np.linspace(-1.5, 1.5, 3001)
    By = np.asarray(res.sample(np.zeros_like(zz), np.zeros_like(zz), zz)["By"], float)     # the solved field
    p_reco = ref.GEV_PER_TM * float(np.trapezoid(By, zz)) / dtheta
    print(f"dipole: deflection {dtheta * 1e3:.3f} mrad, field integral {np.trapezoid(By, zz):.5f} T m, "
          f"p = {p_reco:.5f} GeV/c for {P} GeV/c")
    assert p_reco == pytest.approx(P, rel=5e-3)

