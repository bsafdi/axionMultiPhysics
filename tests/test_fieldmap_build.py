"""Field maps built from solved axisymmetric designs (``Results.export_field_map``; axionmultiphysics.fieldmap, beta).

The particle-transport design note, section 5 (field maps) and its validation plan, with the corrections measured on
solved fields: per-region recovery, the extension rule, curves, chords, the RMS statistic, acceptance, caps and the
manifest. The designs are those measurements': a mu_r = 1000 sphere in a uniform 0.1 T field, a closed
superconducting spherical sheet, an open superconducting cap, a thin mu_r = 2e4 layer on the sphere, a saturated iron
core in a coil, a knee-law iron block in a toroid, and a coarse solenoid (R 0.5, H 1, 4600 elements).

References, all independent of the package (tests/_fieldmap_ref.py): the exact fields of the sphere, the superconducting
sphere (mu = 0), the thick spherical shell, the thick solenoid (Derby-Olbert, checked against loop superposition) and
the knee-law toroid (B = knee(NI / (2 pi r)) inside the iron); the test's own multilinear interpolation and exact chord
integrals of the written arrays; the recovered field B* only through ``fieldmap.recovered_field`` and the FE field B_h
only through ``Results.sample`` (the existing API). ``test_exact_fields_describe_the_solved_designs`` checks that the
closed forms describe the package's solutions of these designs.

What is checked and why the tolerances hold:
- the statistic is recomputed from the written map and the written per-chord integrals with the 10 % floor; the
  integrals of the map along a chord are exact (multilinear along a line is a polynomial per cell), so they agree to
  1e-10 of the denominator; E, H, q95 and the maximum are arithmetic of recorded numbers (1e-12 relative);
- the recorded integrals of B* are checked against ``recovered_field`` with Gauss-Legendre on pieces of h_el / 8
  (B* is continuous, so 1e-8 of the denominator), those of B_h against ``Results.sample`` relative to H itself (B_h
  jumps between elements by about H, and the pieces do not follow the elements);
- the triangle bound: RMS over chords of map vs exact <= 1.05 (RMS of B* vs exact + E) (the slack covers the
  quadrature of the exact field); an under-reported E fails it;
- the recovery bound: RMS of B* vs exact <= 1.5 RMS of B_h vs exact + 1e-4 (healthy, measured: <= 1.20 on these exact
  cases; recovery across an interface, across a sheet, or the axis value from the nearest off-axis node: 7 to 1700
  times; the additive term is for the knee-law iron, whose B_h is exact);
- the convergence rule: in every acceptance family E(l) / E(l + 2) >= 3.0 (healthy, measured: >= 4.17; planted defects
  <= 2.14), where E(l + 2) is above 1e-3 tol_eff (below that, E is at the level of quadrature and round-off);
- acceptance: the written level meets tol_eff = max(tol, 0.1 H) in every acceptance family, and replaying the rule on
  the recorded history (tests/_fieldmap_ref.decide_rule) stops where the builder stopped;
- the report reproduces: ``fieldmap.chord_error`` with the manifest's seed gives the reported E to 1e-12; with another
  seed E agrees within a factor 2.5 (measured <= 1.82 at N = 16384 / 4096 with the 10 % floor).

The default suite exports at tol = 1e-3 (levels 0 to 3) and checks every rule; the measured cases at the default
tol = 1e-4, with all chords, need AXMP_SLOW=1.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core
from axionmultiphysics import fieldmap as fmod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fieldmap_ref as ref  # noqa: E402

SLOW = bool(os.environ.get("AXMP_SLOW"))
slow = pytest.mark.skipif(not SLOW, reason="slow (the measured cases at tol 1e-4, every chord); set AXMP_SLOW=1")

B0 = 0.1
A_SPHERE = 0.5
DOMAIN = 1.5
EDGE = 1e-3          # points closer than this to a closed-form boundary are not judged (the mesh's boundary is a
                     # quadratic approximation of the circle)
# the constants of the format, stated here independently of the package's own (test_fieldmap_format.py checks those)
FLOOR = 0.1
TOL_EFF_FACTOR = 0.1
GUARD = 3.0
MIN_NODES = 9
LENGTH_H_EL = 8.0
FIELD_FREE_RATIO = 1e-10
CHORD_SEED = 0x5DEECE66D
CHORD_KEYS = ("start", "direction", "segments", "segment_chord", "length", "I_star", "I_abs", "I_h", "I_map")


# ============================================================================================ designs
def _arc(radius, a0, a1):
    return {"arc": {"center": [0, 0], "radius": radius, "angles": [a0, a1]}}


def sphere_doc(material="fe1000"):
    """A sphere of radius 0.5 m (mu_r = 1000 by default) in a uniform 0.1 T field."""
    return {"schema": "axionmultiphysics.design/1", "name": "fieldmap_iron_sphere",
            "materials": {"fe1000": {"kind": "linear", "mu_r": 1000}},
            "regions": [{"name": "ball", "material": material, "shape": {"contour": {
                "start": [0, -A_SPHERE],
                "segments": [{"ellipse": {"center": [0, 0], "axes": [A_SPHERE, A_SPHERE], "t": [-math.pi / 2, math.pi / 2]}},
                             {"line": [0, -A_SPHERE]}]}}}],
            "background": {"type": "uniform", "Bz": B0},
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02, "domain_radius": DOMAIN}}


def sheet_doc(open_cap=False):
    """A closed superconducting sheet on the sphere, or an open cap (0 to 90 degrees)."""
    a0 = 0.0 if open_cap else -math.pi / 2
    return {"schema": "axionmultiphysics.design/1", "name": "fieldmap_sc_cap" if open_cap else "fieldmap_sc_sphere",
            "regions": [], "background": {"type": "uniform", "Bz": B0},
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.01, "domain_radius": DOMAIN},
            "curves": [{"name": "shell", "material": "superconductor", "path": _arc(A_SPHERE, a0, math.pi / 2),
                        "surface_size": 0.01}]}


LAYER_D, LAYER_MU = 0.002, 2e4


def layer_doc():
    """A thin mu_r = 2e4 layer, 2 mm, on the sphere."""
    return {"schema": "axionmultiphysics.design/1", "name": "fieldmap_layer_sphere", "regions": [],
            "background": {"type": "uniform", "Bz": B0},
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.01, "domain_radius": DOMAIN},
            "materials": {"mumetal": {"kind": "linear", "mu_r": LAYER_MU}},
            "curves": [{"name": "shell", "material": "mumetal", "thickness": LAYER_D,
                        "path": _arc(A_SPHERE, -math.pi / 2, math.pi / 2), "surface_size": 0.01}]}


def core_doc(refined=False):
    """An iron_knee core r < 0.4, |z| < 0.3 in a 3e6 At coil r 0.5-0.6, |z| < 0.2 (saturated, about 6 T)."""
    mesh = {"order": 2, "size": 0.03, "surface_size": 0.01}
    if refined:
        mesh["refinements"] = 1
    return {"schema": "axionmultiphysics.design/1", "name": "fieldmap_iron_core",
            "regions": [{"name": "coil", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.2, 0.2]}}, "material": "winding"},
                        {"name": "core", "shape": {"rect": {"r": [0.0, 0.4], "z": [-0.3, 0.3]}}, "material": "iron_knee"}],
            "windings": [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 3e6}], "mesh": mesh}


SOL = dict(R=0.5, H=1.0, tw=0.05, NI=2e5)


def solenoid_doc():
    """A thick solenoid R 0.5, H 1, winding 0.05, 2e5 At (4600 elements)."""
    return {"schema": "axionmultiphysics.design/1", "name": "fieldmap_solenoid",
            "parameters": {"R": SOL["R"], "H": SOL["H"], "tw": SOL["tw"], "NI": SOL["NI"]},
            "regions": [{"name": "winding", "material": "winding",
                         "shape": {"rect": {"r": ["R", "R+tw"], "z": ["-H/2", "H/2"]}}}],
            "windings": [{"name": "magnet", "region": "winding", "type": "azimuthal", "ampere_turns": "NI"}],
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02, "domain_radius": 1.5}}


KNEE = dict(NI=13500.0, mur=1000.0, Bs=2.0, fe=(1.25, 1.45, -0.2, 0.2))


def toroid_doc():
    """A poloidal (toroidal-field) winding with an iron_knee block straddling the knee radius 1.35 m."""
    return {"schema": "axionmultiphysics.design/1", "name": "fieldmap_toroid_knee",
            "parameters": {"R": 2.0, "H": 1.0, "tw": 0.02, "NI": KNEE["NI"], "y": 0.5},
            "regions": [{"name": "winding", "material": "winding",
                         "shape": {"frame": {"r": ["y*R-tw", "R+tw"], "thickness": "tw", "z": ["-H/2-tw", "H/2+tw"]}}},
                        {"name": "fe", "material": "iron_knee", "shape": {"rect": {"r": [1.25, 1.45], "z": [-0.2, 0.2]}}}],
            "windings": [{"name": "toroid", "region": "winding", "type": "poloidal", "ampere_turns": "NI"}],
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.01}}


# ============================================================================================ exact fields (native)
def _rz(q):
    q = np.atleast_2d(np.asarray(q, dtype=float))
    return q[:, 0], q[:, 1]


def dipole_field(K, Bin_factor):
    """(outside, inside) native fields of a sphere-like body of radius 0.5 in B0 (K in units of 0.5^3)."""
    def outside(q):
        r, z = _rz(q)
        br, bz = ref.dipole_exterior(B0, K, A_SPHERE, r, z)
        return np.column_stack([br, bz, np.zeros_like(br)])

    def inside(q):
        r, _ = _rz(q)
        return np.column_stack([np.zeros_like(r), np.full_like(r, Bin_factor * B0), np.zeros_like(r)])
    return outside, inside


def sphere_exact(mu):
    return dipole_field(*ref.sphere_K_Bin(mu))


def layer_exact():
    a1, a2 = A_SPHERE - LAYER_D / 2, A_SPHERE + LAYER_D / 2
    K2, fin = ref.shell_K_Bin(a1, a2, LAYER_MU)
    return dipole_field(K2 * (a2 / A_SPHERE) ** 3, fin)


def solenoid_exact(q):
    r, z = _rz(q)
    br, bz = ref.solenoid_B(SOL["R"], SOL["H"], SOL["tw"], SOL["NI"], r, z, nodes=16)
    return np.column_stack([br, bz, np.zeros_like(br)])


def knee_exact(sign):
    mu0 = ref.MU0

    def f(q):
        r, _ = _rz(q)
        H = KNEE["NI"] / (2 * math.pi * r)
        Hk = KNEE["Bs"] / (mu0 * KNEE["mur"])
        B = np.where(H <= Hk, mu0 * KNEE["mur"] * H, KNEE["Bs"] + mu0 * (H - Hk))
        return np.column_stack([np.zeros_like(r), np.zeros_like(r), sign * B])
    return f


# ============================================================================================ helpers
def without_run_and_paths(info):
    """``build_info()`` as a manifest records it: without ``threads`` (a property of the run, kept under ``timing``) and
    without what names a directory of the machine (``build_dir``, ``git_dirty_files``, at every depth)."""
    return {k: (without_run_and_paths(v) if isinstance(v, dict) else v) for k, v in info.items()
            if k not in ("threads", "build_dir", "git_dirty_files")}


def local_paths(obj, roots):
    """Every string of a manifest that is an absolute path or names one of ``roots`` (the map's directory, the
    solve's work directory, the source tree, the home directory)."""
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += local_paths(v, roots)
    elif isinstance(obj, list):
        for v in obj:
            out += local_paths(v, roots)
    elif isinstance(obj, str):
        if os.path.isabs(obj) or any(r in obj for r in roots):
            out.append(obj)
    return out


def solve(doc, wd):
    return axmp.Design(doc).solve(workdir=str(wd), verbose=0)


def manifest(path):
    return json.loads((Path(path) / "manifest.json").read_text())


def precs(m):
    return {p["name"]: p for p in m["patches"]}


def array(path, prec):
    return np.load(Path(path) / prec["file"])


def chords(path, prec, fam):
    d = Path(path) / prec["dir"] / "chords" / fam
    return {k: np.load(d / f"{k}.npy") for k in CHORD_KEYS}


def nodes(prec):
    axes = [prec["origin"][a] + prec["spacing"][a] * np.arange(prec["shape"][a]) for a in range(len(prec["shape"]))]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack(grids, axis=-1)


def nothing_written(path):
    return not Path(path).exists() and not Path(str(path) + ".tmp").exists()


def code_of(excinfo):
    return getattr(excinfo.value, "code", None)


def acceptance_families(prec):
    return {f: v for f, v in prec["families"].items() if v["acceptance"]}


def den(C, B_patch):
    return np.maximum(C["I_abs"], FLOOR * C["length"] * B_patch)


def subset(C, n):
    """The first n chords of a family (all segments of those chords)."""
    if n is None or n >= len(C["length"]):
        return C
    keep = C["segment_chord"] < n
    out = {k: C[k][:n] for k in ("start", "direction", "length", "I_star", "I_abs", "I_h", "I_map")}
    out["segments"] = C["segments"][keep]
    out["segment_chord"] = C["segment_chord"][keep]
    return out


def exact_integrals(C, exact, h_el, sub=8, npts=6):
    pts, w, c = ref.chord_points(C["start"], C["direction"], C["segments"], C["segment_chord"], h_el / sub, npts)
    return ref.integrate(exact(pts), w, c, len(C["length"]))


def check_statistic_recomputed(path, prec, fam, tol):
    """The report reproduces, independently: the map's chord integrals from the written array, and the reported numbers from the
    recorded per-chord integrals with the 10 % floor."""
    C = chords(path, prec, fam)
    rec = prec["families"][fam]
    n = len(C["length"])
    assert rec["n"] == n
    B_patch = prec["B_patch"]
    d = den(C, B_patch)
    B = array(path, prec)
    I_map = ref.map_chord_integrals(prec["origin"], prec["spacing"], B, C["start"], C["direction"],
                                    C["segments"], C["segment_chord"], n)
    assert np.max(np.linalg.norm(I_map - C["I_map"], axis=1) / d) <= 1e-10, f"{prec['name']}/{fam}: I_map"
    e = ref.per_chord(C["I_map"], C["I_star"], C["I_abs"], C["length"], B_patch)
    h = ref.per_chord(C["I_star"], C["I_h"], C["I_abs"], C["length"], B_patch)
    es = np.sort(e)
    assert rec["E"] == pytest.approx(ref.rms(e), rel=1e-12)
    assert rec["E_q95"] == pytest.approx(es[int(0.95 * (n - 1))], rel=1e-12)
    assert rec["E_max"] == pytest.approx(es[-1], rel=1e-12)
    assert rec["H"] == pytest.approx(ref.rms(h), rel=1e-12)
    assert rec["H"] > 0.0
    assert rec["tol_eff"] == pytest.approx(max(tol, TOL_EFF_FACTOR * rec["H"]), rel=1e-14)
    assert rec["floor_active"] == (TOL_EFF_FACTOR * rec["H"] > tol)
    assert ref.rms(ref.per_chord(I_map, C["I_star"], C["I_abs"], C["length"], B_patch)) == pytest.approx(rec["E"], rel=1e-8)
    return C


def star_integrals(res, path, prec, C, pieces):
    """Integrals of B* and |B*| along the chords C: 3-point Gauss-Legendre on pieces of h_el / pieces, from
    ``recovered_field``. The pieces do not follow the elements, across whose faces B* has kinks, so the error of the
    rule falls like the square of the piece (measured on the sphere: 8.7e-6 of the denominator at h_el / 8, 7.5e-7 at
    h_el / 32, 3.1e-8 at h_el / 128); the reference is therefore converged and its own error estimated."""
    n = len(C["length"])
    pts, w, c = ref.chord_points(C["start"], C["direction"], C["segments"], C["segment_chord"], prec["h_el"] / pieces, 3)
    Bs = np.asarray(fmod.recovered_field(res, path, prec["name"], pts))
    assert np.all(np.isfinite(Bs)), "every chord point lies in the patch"
    return ref.integrate(Bs, w, c, n), ref.integrate(np.linalg.norm(Bs, axis=1), w, c, n)[:, 0]


def check_star_integrals(res, path, prec, fam, n_check=16):
    """The recorded integrals of B* and |B*| against a converged reference: pieces of h_el / 4096, whose error is below
    a third of its difference from pieces of h_el / 2048 (second-order convergence), required to be under 2e-9 of the
    denominator (measured: <= 2.3e-9 between h_el / 1024 and h_el / 2048, <= 4.4e-10 from the builder, on the sphere).
    The recorded values must agree with it to 1e-8 of the denominator, five times the reference's certified error."""
    C = subset(chords(path, prec, fam), n_check)
    d = den(C, prec["B_patch"])
    I1, A1 = star_integrals(res, path, prec, C, 2048)
    I2, A2 = star_integrals(res, path, prec, C, 4096)
    est = max(np.max(np.linalg.norm(I1 - I2, axis=1) / d), np.max(np.abs(A1 - A2) / d))
    assert est <= 2e-9, f"{prec['name']}/{fam}: the reference itself is not converged ({est:.1e})"
    assert np.max(np.linalg.norm(I2 - C["I_star"], axis=1) / d) <= 1e-8
    assert np.max(np.abs(A2 - C["I_abs"]) / d) <= 1e-8
    return C, d


def check_integrals_of_the_fields(res, path, prec, fam, n_check=16):
    """The recorded integrals of B* (and |B*|) against ``recovered_field``, and those of B_h against ``Results.sample``."""
    C, d = check_star_integrals(res, path, prec, fam, n_check)
    n = len(C["length"])
    h_el = prec["h_el"]
    pts, w, c = ref.chord_points(C["start"], C["direction"], C["segments"], C["segment_chord"], h_el / 32, 3)
    s = res.sample(pts[:, 0], pts[:, 1])
    Bh = np.column_stack([s["Br"], s["Bz"], s["Bphi"]])
    I_h = ref.integrate(Bh, w, c, n)
    diff = ref.rms(np.linalg.norm(I_h - C["I_h"], axis=1) / d)
    H_here = ref.rms(np.linalg.norm(C["I_star"] - I_h, axis=1) / d)
    assert diff <= 0.2 * H_here, f"recorded B_h integrals differ from B_h by {diff:.2e} (H here {H_here:.2e})"


def check_f3b_accuracy(path, prec, exact, n_chords=None, tol_v=1e-4, sub=8):
    """The triangle bound and the recovery bound in every family against the exact field (Gauss-Legendre, 6 points on pieces of h_el / sub)."""
    out = {}
    for fam in prec["families"]:
        C = subset(chords(path, prec, fam), n_chords)
        n = len(C["length"])
        B_patch = prec["B_patch"]
        I_ex = exact_integrals(C, exact, prec["h_el"], sub=sub)
        I_map = ref.map_chord_integrals(prec["origin"], prec["spacing"], array(path, prec), C["start"], C["direction"],
                                        C["segments"], C["segment_chord"], n)
        a = ref.rms(ref.per_chord(I_map, I_ex, C["I_abs"], C["length"], B_patch))
        b = ref.rms(ref.per_chord(C["I_star"], I_ex, C["I_abs"], C["length"], B_patch))
        hh = ref.rms(ref.per_chord(C["I_h"], I_ex, C["I_abs"], C["length"], B_patch))
        e = ref.rms(ref.per_chord(C["I_map"], C["I_star"], C["I_abs"], C["length"], B_patch))
        out[fam] = (a, b, hh, e)
        assert a <= 1.05 * (b + e), f"{prec['name']}/{fam} triangle bound: map vs exact {a:.3e} > 1.05 (B* {b:.3e} + E {e:.3e})"
        assert b <= 1.5 * hh + tol_v, f"{prec['name']}/{fam} recovery bound: B* vs exact {b:.3e}, B_h vs exact {hh:.3e}"
    return out


def check_acceptance(prec, tol):
    """The convergence rule and acceptance, and the replay of the rule on the recorded history."""
    fams = acceptance_families(prec)
    assert {"bulk"} <= set(fams)
    hist = prec["levels"]
    L = prec["level"]
    assert [h["level"] for h in hist] == list(range(L + 1))
    assert list(hist[-1]["shape"]) == list(prec["shape"])
    tol_eff = {f: fams[f]["tol_eff"] for f in fams}
    for f in fams:
        assert fams[f]["E"] == hist[L]["E"][f]
        assert tol_eff[f] == pytest.approx(max(tol, TOL_EFF_FACTOR * fams[f]["H"]), rel=1e-14)
    for l in range(L + 1):
        got = ref.decide_rule({f: [h["E"][f] for h in hist[:l + 1]] for f in fams}, tol_eff)
        if l < L:
            assert got["status"] == "refine", f"{prec['name']}: the rule stops at level {l}, the builder went on"
    assert got["status"] == "accepted", f"{prec['name']}: {got} ({prec['stop_reason']})"
    assert prec["accepted"] is True and prec["stop_reason"] == "accepted"
    for f in fams:                                                       # acceptance
        assert fams[f]["E"] <= tol_eff[f]
        for l in range(L - 1):                                           # the convergence rule
            e0, e2 = hist[l]["E"][f], hist[l + 2]["E"][f]
            if e2 > 1e-3 * tol_eff[f]:
                assert e0 / e2 >= GUARD, f"{prec['name']}/{f}: E({l})/E({l + 2}) = {e0 / e2:.2f}"
        if L >= 1:
            assert fams[f]["gain_last"] == pytest.approx(hist[L - 1]["E"][f] / hist[L]["E"][f], rel=1e-12)
        if L >= 2:
            assert fams[f]["gain_two"] == pytest.approx(hist[L - 2]["E"][f] / hist[L]["E"][f], rel=1e-12)


def check_grid_rule(prec):
    """Level-0 spacing from the median element size (at least 9 nodes per axis), halving per level, nested grids."""
    box = prec["box"]
    h_el = prec["h_el"]
    for a in range(len(box)):
        L = box[a][1] - box[a][0]
        n0 = max(MIN_NODES, math.ceil(L / h_el) + 1)
        shapes = [h["shape"][a] for h in prec["levels"]]
        assert shapes[0] == n0
        assert all(shapes[k + 1] == 2 * shapes[k] - 1 for k in range(len(shapes) - 1))
        assert prec["origin"][a] == box[a][0]
        assert prec["spacing"][a] == pytest.approx(L / (prec["shape"][a] - 1), rel=1e-14)
        assert prec["levels"][-1]["spacing"][a] == prec["spacing"][a]


def recorded_intervals(C):
    """Per chord: the recorded in-patch length and the number of separate intervals (touching segments merged)."""
    n = len(C["length"])
    count = np.zeros(n, dtype=np.int64)
    last_end = np.full(n, -np.inf)
    for (s0, s1), c in zip(C["segments"], C["segment_chord"]):
        if s0 > last_end[c] + 1e-9 * (1.0 + abs(s0)):
            count[c] += 1
        last_end[c] = max(last_end[c], s1)
    return np.bincount(C["segment_chord"], weights=C["segments"][:, 1] - C["segments"][:, 0], minlength=n), count


def grazing(C, radii, L):
    """The smallest |n . v| over the chord's crossings of the boundary spheres inside [0, L], 0 for a line that passes
    within 1e-3 of a radius of tangency (where the mesh's quadratic boundary and the exact sphere may differ in
    whether the line crosses at all)."""
    g = np.full(len(C["length"]), np.inf)
    st, dr = C["start"], C["direction"]
    for rad in radii:
        roots = ref._sphere_roots(st, dr, rad)
        for k in (0, 1):
            ok = np.isfinite(roots[:, k]) & (roots[:, k] > 0) & (roots[:, k] < L)
            x = st + np.nan_to_num(roots[:, k])[:, None] * dr
            g = np.where(ok, np.minimum(g, np.abs(np.sum(x * dr, axis=1)) / rad), g)
        b = np.sum(st * dr, axis=1)
        closest = np.sqrt(np.maximum(np.sum(st * st, axis=1) - b * b, 0.0))
        g = np.where((np.abs(closest - rad) < 1e-3 * rad) & (-b > 0) & (-b < L), 0.0, g)
    return g


def check_chords_complete(path, prec, inside, radii, half_plane=False, box_of=lambda fam: None):
    """Chord completeness: the recorded segments of every chord cover its exact intersection with the region over
    [0, 8 h_el] (closed-form roots of the boundary spheres, the axis and the track box; tests/_fieldmap_ref). Away from
    grazing incidence (|n . v| >= 0.1 at every crossing) the lengths agree to 1e-3 of the nominal length and the
    numbers of separate intervals are equal; the mesh's curved boundary is a quadratic approximation of the sphere, which
    moved lengths by <= 1e-4 m in the measurements (2D: <= 1.5e-6 at q99). Grazing chords, where a sliver of the region
    can differ between the mesh and the sphere, agree to 2 % of the nominal length. A walk that stops early (the
    measured defect: 0.5 % of 2D and 3-9 % of 3D chords, shortened by 0.01 to 0.9 m) or a chord cut to its first
    interval fails."""
    L = LENGTH_H_EL * prec["h_el"]
    for fam in prec["families"]:
        C = chords(path, prec, fam)
        exact, n_exact = ref.exact_chord_intervals(C["start"], C["direction"], L, inside(fam), radii, half_plane,
                                                    box_of(fam))[:2]
        got, n_got = recorded_intervals(C)
        g = grazing(C, radii, L)
        clean = g >= 0.1
        dL = np.abs(got - exact)
        bad = np.flatnonzero((clean & ((dL > 1e-3 * L) | (n_got != n_exact))) | (~clean & (dL > 0.02 * L)))
        assert bad.size == 0, (f"{prec['name']}/{fam}: {bad.size} of {len(exact)} chords do not cover their exact "
                               f"intersection; first {bad[:5].tolist()}: recorded {got[bad[:5]].round(5).tolist()} m "
                               f"in {n_got[bad[:5]].tolist()} intervals, exact {exact[bad[:5]].round(5).tolist()} m in "
                               f"{n_exact[bad[:5]].tolist()}")


def sphere_region(name, box=None):
    """The exact region of a sphere-case patch in the (r, z) half-plane: the ball, or the air between the sphere and
    the domain's circle; intersected with the track box when given."""
    def inside(q):
        rho = np.hypot(q[:, 0], q[:, 1])
        ok = (q[:, 0] >= 0) & ((rho <= A_SPHERE) if name == "ball" else ((rho >= A_SPHERE) & (rho <= DOMAIN)))
        if box is not None:
            ok &= (q[:, 0] >= box[0][0]) & (q[:, 0] <= box[0][1]) & (q[:, 1] >= box[1][0]) & (q[:, 1] <= box[1][1])
        return ok
    return inside


def in_sphere_region(r, z, inside):
    rho = np.hypot(r, z)
    if inside:
        return rho < A_SPHERE - EDGE
    return (rho > A_SPHERE + EDGE) & (rho < DOMAIN - EDGE)


# ============================================================================================ the references describe the solves
def _region_points(n, inside, rng):
    """Random points of the sphere's inside or outside (within the domain), away from the surfaces."""
    pts = []
    while sum(len(p) for p in pts) < n:
        q = rng.uniform([0.0, -DOMAIN], [DOMAIN, DOMAIN], (4 * n, 2))
        pts.append(q[in_sphere_region(q[:, 0], q[:, 1], inside) & (np.hypot(*q.T) > 0.0)])
    return np.concatenate(pts)[:n]


def _bh(res, q):
    s = res.sample(q[:, 0], q[:, 1])
    return np.column_stack([s["Br"], s["Bz"], s["Bphi"]])


def test_exact_fields_describe_the_solved_designs(sphere_res, sheet_res, solenoid_res, core_res):
    """Not a field-map test: the closed forms used below agree with the package's FE solutions of the same designs to
    the FE error (pointwise, relative to the largest field), so a sign, unit or convention slip in a reference cannot
    pass for a map defect. The core has no closed form; its solve is checked for the saturation the case needs."""
    rng = np.random.default_rng(7)
    out, ins = sphere_exact(1000.0)
    for (_, res), fields in ((sphere_res, (out, ins)), (sheet_res, sphere_exact(0.0))):
        for inside, f in ((False, fields[0]), (True, fields[1])):
            q = _region_points(4000, inside, rng)
            err = np.linalg.norm(_bh(res, q) - f(q), axis=1)
            assert np.quantile(err, 0.99) <= 3e-3 * 0.3 and np.max(err) <= 0.03 * 0.3
    _, res = solenoid_res
    q = np.column_stack([rng.uniform(0.0, 1.4, 4000), rng.uniform(-1.4, 1.4, 4000)])
    q = q[np.hypot(*q.T) < 1.45]
    want = solenoid_exact(q)
    err = np.linalg.norm(_bh(res, q) - want, axis=1)
    scale = np.max(np.linalg.norm(want, axis=1))
    assert np.quantile(err, 0.99) <= 3e-3 * scale and np.max(err) <= 0.03 * scale
    _, res = core_res
    assert 4.0 < float(np.max(np.abs(_bh(res, np.array([[0.0, 0.0], [0.2, 0.0]]))[:, 1]))) < 8.0


# ============================================================================================ the sphere, tol 1e-3
TRACK = [{"name": "equator", "r": [0.4, 0.7], "z": [-0.15, 0.15]}]


@pytest.fixture(scope="module")
def sphere_res(tmp_path_factory):
    wd = tmp_path_factory.mktemp("sphere")
    return wd, solve(sphere_doc(), wd / "solve")


@pytest.fixture(scope="module")
def sphere(sphere_res):
    wd, res = sphere_res
    path = wd / "map"
    fm = res.export_field_map(path, tol=1e-3, track_volumes=TRACK)
    out, ins = sphere_exact(1000.0)
    return SimpleNamespace(res=res, path=path, fm=fm, m=manifest(path), wd=wd,
                           exact={"air": out, "ball": ins})


def test_manifest_records_provenance_units_and_the_constants(sphere):
    m = sphere.m
    assert m["schema"] == "axionmultiphysics.fieldmap/1"
    assert m["units"] == {"length": "m", "B": "T"}
    assert m["tol"] == 1e-3
    assert m["outside"] == "refuse" and "boundary_ratio" not in m
    assert m["chord_seed"] == CHORD_SEED
    assert m["chords"] == {"bulk": 16384, "layer": 4096, "axis": 4096, "track": 4096, "length_h_el": 8.0,
                           "floor": 0.1, "sequence": "scrambled-halton", "bases": [2, 3, 5, 7, 11, 13]}
    assert m["control"] == {"tol_eff_factor": 0.1, "guard_two_halvings": 3.0, "field_free_ratio": 1e-10,
                            "min_nodes": 9, "max_patch_MB": 512.0, "max_total_MB": 2048.0,
                            "iterate_on_track_volumes": False, "merge": True}
    src = m["source"]
    assert src["type"] == "fe_solution" and src["symmetry"] == "axisymmetric"
    assert src["design"] == "fieldmap_iron_sphere"
    text = json.dumps(sphere.res.design.doc, sort_keys=True, separators=(",", ":"))
    assert src["design_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert len(src["solution_sha256"]) == 64
    assert src["order"] == sphere.res.data["mesh"]["order"]
    assert src["elements"] == sphere.res.data["mesh"]["elements"]
    assert src["version"] == axmp.__version__
    assert src["build_info"] == without_run_and_paths(axmp.build_info())
    assert m["timing"]["threads"] == axmp.build_info()["threads"]
    roots = [str(sphere.path), str(sphere.res.workdir), str(Path(__file__).resolve().parents[1]), str(Path.home())]
    assert local_paths({k: v for k, v in m.items() if k != "timing"}, roots) == [], "a user file names no local path"
    assert m["zero_field_regions"] == []
    assert m["track_volumes"] == TRACK
    assert isinstance(m["warnings"], list)
    assert isinstance(sphere.fm, fmod.FieldMap)


def test_one_patch_per_region_with_its_own_grid(sphere):
    P = precs(sphere.m)
    assert set(P) == {"air", "ball"}
    for name, box in (("air", [[0.0, DOMAIN], [-DOMAIN, DOMAIN]]), ("ball", [[0.0, A_SPHERE], [-A_SPHERE, A_SPHERE]])):
        p = P[name]
        assert p["regions"] == [name] and p["component"] is None and p["curves"] == []
        assert p["kind"] == 1 and p["frame"] == {"type": "axisymmetric", "axis": "z"}
        assert p["touches_axis"] is True and p["box"][0][0] == 0.0
        np.testing.assert_allclose(np.asarray(p["box"]), box, atol=2e-4)
        assert 0.02 < p["h_el"] < 0.07
        assert p["field_free"] is False
        check_grid_rule(p)
    # per-patch spacing: the ball (uniform field) is accepted at level 0, the air needs finer grids
    assert P["ball"]["level"] == 0 and P["air"]["level"] >= 2
    assert P["air"]["spacing"][0] < 0.5 * P["ball"]["spacing"][0]


def test_written_arrays(sphere):
    for p in sphere.m["patches"]:
        raw = (sphere.path / p["file"]).read_bytes()
        assert p["sha256"] == hashlib.sha256(raw).hexdigest()
        B = array(sphere.path, p)
        assert B.dtype.str == "<f8" and B.flags["C_CONTIGUOUS"]
        assert B.shape == tuple(p["shape"]) + (3,)
        assert p["bytes"] == B.size * 8
        assert np.all(np.isfinite(B))


def test_in_region_nodes_hold_the_patch_recovered_field(sphere):
    """Nodes inside the served region, and nodes on its boundary, take B*, the patch's own recovered field (sections
    5.2-5.3): wherever ``recovered_field`` finds a patch element (one-sided by construction) the map holds exactly
    that value. Every node of the closed-form region is such a node."""
    for p in sphere.m["patches"]:
        X = nodes(p).reshape(-1, 2)
        B = array(sphere.path, p).reshape(-1, 3)
        Bs = np.asarray(fmod.recovered_field(sphere.res, sphere.path, p["name"], X))
        found = np.all(np.isfinite(Bs), axis=1)
        inside = in_sphere_region(X[:, 0], X[:, 1], p["name"] == "ball")
        assert np.all(found[inside])
        rho = np.hypot(X[:, 0], X[:, 1])
        assert np.any(found & (np.abs(rho - A_SPHERE) < 0.25 * min(p["spacing"]))), "nodes at the interface are judged"
        assert np.max(np.abs(B[found] - Bs[found])) <= 1e-12 * p["B_patch"]
        # and B* is the FE field's region: within the FE error of the exact field
        want = sphere.exact[p["name"]](X[inside])
        assert np.max(np.linalg.norm(Bs[inside] - want, axis=1)) <= 0.03 * p["B_patch"]


def test_axis_rule(sphere):
    """On r = 0: B_r = B_phi = 0 exactly in the map and in B*, and B_z is the edge-element limit (close to exact)."""
    for p in sphere.m["patches"]:
        B = array(sphere.path, p)
        assert np.all(B[0, :, 0] == 0.0) and np.all(B[0, :, 2] == 0.0)
        z = p["origin"][1] + p["spacing"][1] * np.arange(p["shape"][1])
        inside = np.abs(z) < A_SPHERE - EDGE if p["name"] == "ball" else (np.abs(z) > A_SPHERE + EDGE) & (np.abs(z) < DOMAIN - EDGE)
        q = np.column_stack([np.zeros(inside.sum()), z[inside]])
        Bs = np.asarray(fmod.recovered_field(sphere.res, sphere.path, p["name"], q))
        assert np.all(Bs[:, 0] == 0.0) and np.all(Bs[:, 2] == 0.0)
        np.testing.assert_allclose(Bs[:, 1], sphere.exact[p["name"]](q)[:, 1], atol=1e-2 * p["B_patch"])


def test_extension_nodes_hold_the_regions_own_field_continued(sphere):
    """Air-patch nodes inside the ball hold the air's own field extended from the nearest air element, never the
    ball's: they follow the exterior field's analytic continuation (the dipole formula inside the sphere), which near
    the equator differs from the ball's field by more than half of B_patch. A node within one map cell of the surface
    belongs to a cell that reaches into the air, so it is filled; deeper nodes are filled (by the same rule) or hold 0
    when no cell around them touches an air element's bounding box; nodes far from every air element hold 0."""
    p = precs(sphere.m)["air"]
    X = nodes(p).reshape(-1, 2)
    B = array(sphere.path, p).reshape(-1, 3)
    rho = np.hypot(X[:, 0], X[:, 1])
    depth = A_SPHERE - rho
    h = min(p["spacing"])
    first = (depth > EDGE) & (depth < h)
    assert first.sum() > 50
    cont = sphere.exact["air"](X[first])
    err = np.linalg.norm(B[first] - cont, axis=1)
    assert np.max(err) <= 0.05 * p["B_patch"], f"extension error {np.max(err) / p['B_patch']:.3f} of B_patch"
    ball = sphere.exact["ball"](X[first])
    assert np.max(np.linalg.norm(ball - cont, axis=1)) >= 0.5 * p["B_patch"], "the test can tell the two apart"
    deeper = (depth >= h) & (depth < p["h_el"])
    filled = np.any(B[deeper] != 0.0, axis=1)
    err = np.linalg.norm(B[deeper][filled] - sphere.exact["air"](X[deeper][filled]), axis=1)
    assert err.size == 0 or np.max(err) <= 0.05 * p["B_patch"]
    far = depth > 2.5 * p["h_el"] + 2 * max(p["spacing"])
    assert far.sum() > 50 and np.all(B[far] == 0.0)


def test_B_patch_excludes_extrapolated_nodes(sphere):
    """B_patch is the largest |B*| over the patch's own B* nodal values: within 1 % of the exact maximum over the
    region (0.2994 T at the poles, 3 mu / (mu + 2) B0 in the ball), not the larger values of extension nodes (+6 % to
    +21 % in the measurements)."""
    want = 3 * 1000.0 / 1002.0 * B0
    for p in sphere.m["patches"]:
        assert p["B_patch"] == pytest.approx(want, rel=0.01), p["name"]


def test_chord_families_and_their_geometry(sphere):
    for p in sphere.m["patches"]:
        fams = p["families"]
        assert set(fams) == {"bulk", "layer", "axis", "track_equator"}
        assert {f: fams[f]["n"] for f in fams} == {"bulk": 16384, "layer": 4096, "axis": 4096, "track_equator": 4096}
        assert {f: fams[f]["acceptance"] for f in fams} == {"bulk": True, "layer": True, "axis": True,
                                                             "track_equator": False}
        h_el = p["h_el"]
        inside = p["name"] == "ball"
        for fam in fams:
            C = chords(sphere.path, p, fam)
            st, dr, sg, sc = C["start"], C["direction"], C["segments"], C["segment_chord"]
            assert st.shape == (fams[fam]["n"], 2) and dr.shape == st.shape
            np.testing.assert_allclose(np.linalg.norm(dr, axis=1), 1.0, atol=1e-14)
            assert np.all(np.diff(sc) >= 0) and np.all(sg[:, 1] > sg[:, 0])
            assert np.all(sg[:, 0] >= 0.0) and np.all(sg[:, 1] <= LENGTH_H_EL * h_el * (1 + 1e-12))
            np.testing.assert_allclose(np.bincount(sc, weights=sg[:, 1] - sg[:, 0], minlength=len(st)), C["length"],
                                       rtol=1e-12)
            rho0 = np.hypot(st[:, 0], st[:, 1])
            assert np.all(st[:, 0] >= 0.0)
            assert np.all(rho0 <= A_SPHERE + EDGE) if inside else np.all(rho0 >= A_SPHERE - EDGE)
            for s in (sg[:, 0], sg[:, 1], 0.5 * (sg[:, 0] + sg[:, 1])):
                x = st[sc] + s[:, None] * dr[sc]
                assert np.all(x[:, 0] >= -1e-12), "chords are clipped at r = 0"
                rho = np.hypot(x[:, 0], x[:, 1])
                if inside:
                    assert np.all(rho <= A_SPHERE + EDGE)
                else:
                    assert np.all((rho >= A_SPHERE - EDGE) & (rho <= DOMAIN + EDGE))
                if fam == "track_equator":
                    assert np.all((x[:, 0] >= 0.4 - 1e-12) & (x[:, 0] <= 0.7 + 1e-12) & (np.abs(x[:, 1]) <= 0.15 + 1e-12))
            if fam == "layer":
                assert np.all(np.abs(rho0 - A_SPHERE) <= h_el + EDGE), "layer starts lie within h_el of the interface"
            if fam == "axis":
                assert np.all(st[:, 0] < h_el)
                ends = st[sc] + sg[:, 1:2] * dr[sc]
                starts = st[sc] + sg[:, 0:1] * dr[sc]
                assert np.sum(np.abs(ends[:, 0]) < 1e-9) + np.sum(np.abs(starts[:, 0]) < 1e-9) > 100
            if fam == "track_equator":
                assert np.all((st[:, 0] >= 0.4) & (st[:, 0] <= 0.7) & (np.abs(st[:, 1]) <= 0.15))


def test_chords_cover_the_exact_intersection_with_the_region(sphere):
    """Every chord's recorded segments are its whole intersection with the region (curved elements on the sphere)."""
    box = {"track_equator": [TRACK[0]["r"], TRACK[0]["z"]]}
    for p in sphere.m["patches"]:
        check_chords_complete(sphere.path, p, lambda fam, n=p["name"]: sphere_region(n, box.get(fam)),
                              (A_SPHERE, DOMAIN), half_plane=True, box_of=box.get)


def test_bulk_chords_come_from_the_manifest_seed(sphere):
    """Chord i of the bulk family uses point i + 1 of the family's sequence: its direction is (cos, sin) of 2 pi u_3.
    The same chords serve every level (a chord set regenerated per level would not start the sequence here)."""
    u = ref.halton(ref.family_seed(CHORD_SEED, 0), 16384)
    th = 2 * np.pi * u[:, 3]
    for p in sphere.m["patches"]:
        C = chords(sphere.path, p, "bulk")
        np.testing.assert_allclose(C["direction"], np.column_stack([np.cos(th), np.sin(th)]), atol=1e-14)


def test_reported_statistic_is_recomputed_from_the_written_map(sphere):
    """The report reproduces, independently of the builder's arithmetic; the 10 % floor and B_patch in the denominator."""
    for p in sphere.m["patches"]:
        for fam in p["families"]:
            check_statistic_recomputed(sphere.path, p, fam, 1e-3)
            check_integrals_of_the_fields(sphere.res, sphere.path, p, fam)


def test_the_floor_is_active_somewhere(sphere):
    """Next to the mu_r = 1000 sphere's equator the exterior field nearly vanishes: some chords' denominators are the
    floor, so a wrong floor changes the reported E (the recomputation above catches it)."""
    p = precs(sphere.m)["air"]
    C = chords(sphere.path, p, "layer")
    assert np.mean(C["I_abs"] < FLOOR * C["length"] * p["B_patch"]) > 0.01


def test_chord_error_reproduces_the_report_and_another_seed_agrees(sphere):
    """The report reproduces through the package, from the map read back from disk; an independent seed within 2.5."""
    again = fmod.chord_error(sphere.res, sphere.path)
    other = fmod.chord_error(sphere.res, sphere.path, seed=CHORD_SEED + 1)
    for p in sphere.m["patches"]:
        for fam, rec in p["families"].items():
            assert again[p["name"]][fam]["E"] == pytest.approx(rec["E"], rel=1e-12)
            assert again[p["name"]][fam]["H"] == pytest.approx(rec["H"], rel=1e-12)
            assert other[p["name"]][fam]["n"] == rec["n"]
            if rec["acceptance"]:
                ratio = other[p["name"]][fam]["E"] / rec["E"]
                assert 1 / 2.5 <= ratio <= 2.5, f"{p['name']}/{fam}: seed spread {ratio:.2f}"


def test_sphere_map_accuracy_and_acceptance(sphere):
    for p in sphere.m["patches"]:
        check_acceptance(p, 1e-3)
        check_f3b_accuracy(sphere.path, p, sphere.exact[p["name"]])


def test_track_volume_is_reported_not_iterated(sphere):
    for p in sphere.m["patches"]:
        t = p["families"]["track_equator"]
        assert t["acceptance"] is False
        for k in ("E", "E_q95", "E_max", "H"):
            assert np.isfinite(t[k]) and t[k] >= 0
        assert t["E"] <= t["E_max"]


def test_pointwise_and_interface_reports(sphere):
    P = precs(sphere.m)
    for p in P.values():
        pw = p["pointwise"]
        assert pw["n"] == 100000
        for k in ("map_vs_Bstar", "indicator"):
            assert 0 <= pw[k]["q50"] <= pw[k]["q99"] <= pw[k]["max"] < 1
        (iface,) = p["interfaces"]
        assert iface["with"] == ({"air", "ball"} - {p["name"]}).pop()
        assert iface["points"] > 0
        assert 0 <= iface["normal_mismatch_rms"] <= iface["normal_mismatch_max"] <= 0.05


def _all_files(path):
    return sorted(str(f.relative_to(path)) for f in Path(path).rglob("*") if f.is_file())


def test_export_is_bitwise_reproducible(sphere):
    """The same results exported again, with 1 and with 4 threads: identical arrays (every NPY file, chords included)
    and identical manifests apart from ``timing``."""
    paths = []
    before = _core.get_threads()
    for threads in (1, 4):
        _core.set_threads(threads)
        try:
            pth = sphere.wd / f"again_{threads}"
            sphere.res.export_field_map(pth, tol=1e-3, track_volumes=TRACK)
            paths.append(pth)
        finally:
            _core.set_threads(before)
    m0 = dict(sphere.m)
    m0.pop("timing", None)
    for pth in paths:
        assert _all_files(pth) == _all_files(sphere.path)
        for f in _all_files(pth):
            if f.endswith(".npy"):
                assert (pth / f).read_bytes() == (sphere.path / f).read_bytes(), f
        m = manifest(pth)
        assert m["timing"]["threads"] == int(pth.name.split("_")[1])
        m.pop("timing", None)
        assert m == m0


def test_iterating_on_a_track_volume_makes_it_an_acceptance_family(sphere):
    pth = sphere.wd / "iterate"
    sphere.res.export_field_map(pth, tol=1e-3, track_volumes=TRACK, iterate_on_track_volumes=True)
    m = manifest(pth)
    assert m["control"]["iterate_on_track_volumes"] is True
    base = precs(sphere.m)
    for p in m["patches"]:
        t = p["families"]["track_equator"]
        assert t["acceptance"] is True and t["E"] <= t["tol_eff"]
        check_acceptance(p, 1e-3)
        assert p["level"] >= base[p["name"]]["level"]


def test_a_subset_of_regions(sphere_res):
    wd, res = sphere_res
    pth = wd / "ball_only"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res.export_field_map(pth, regions=["ball"], tol=1e-3)
    assert not [w for w in caught if "fieldmap." in str(w.message)], "an accepted export warns about nothing"
    m = manifest(pth)
    assert [p["name"] for p in m["patches"]] == ["ball"]
    (p,) = m["patches"]
    assert "layer" in p["families"], "the interface with an unmapped region is a jump boundary"


def test_caps(sphere_res):
    """A patch that reaches its cap is written at the finest level that fits, not accepted, with a warning (at
    tol = 1e-9 the air patch needs more than the 1 MB cap; the ball may meet its tol_eff, raised by 0.1 H, first)."""
    wd, res = sphere_res
    cap = 1.0
    pth = wd / "capped"
    with pytest.warns(fmod.FieldMapWarning, match=r"fieldmap\.cap_reached.*\bair\b"):
        res.export_field_map(pth, tol=1e-9, max_patch_MB=cap)
    m = manifest(pth)
    codes = {(w["code"], w.get("patch")) for w in m["warnings"]}
    P = precs(m)
    assert P["air"]["stop_reason"] == "cap"
    for p in m["patches"]:
        assert p["bytes"] <= cap * 2 ** 20
        if p["stop_reason"] == "cap":
            assert p["accepted"] is False
            nxt = np.prod([2 * n - 1 for n in p["shape"]]) * 3 * 8
            assert nxt > cap * 2 ** 20
            assert ("fieldmap.cap_reached", p["name"]) in codes
        else:
            assert p["stop_reason"] == "accepted"
    # the coarsest grid of the air patch does not fit 20 kB
    pth = wd / "too_large"
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        res.export_field_map(pth, tol=1e-3, max_patch_MB=0.02)
    assert code_of(e) == "fieldmap.patch_too_large" and "air" in str(e.value)
    assert nothing_written(pth)
    # the accepted patches together exceed the run's cap
    pth = wd / "total"
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        res.export_field_map(pth, tol=1e-3, max_total_MB=0.5)
    assert code_of(e) == "fieldmap.total_cap_exceeded"
    assert nothing_written(pth)


def test_a_patch_that_does_not_converge_is_written_with_a_warning(sphere_res, monkeypatch):
    """The two-halving guard stopping a patch gives the manifest warning and a FieldMapWarning (a plain UserWarning
    does not satisfy the class check). The guard's verdict is forced: no solved case of this size stalls."""
    wd, res = sphere_res
    monkeypatch.setattr(fmod, "decide", lambda *a, **k: {"status": "not_converging", "family": "bulk", "gain": 1.1})
    pth = wd / "stalled"
    with pytest.warns(fmod.FieldMapWarning, match=r"fieldmap\.not_converging.*\bair\b"):
        res.export_field_map(pth, tol=1e-3)
    m = manifest(pth)
    assert {(w["code"], w["patch"]) for w in m["warnings"]} == {("fieldmap.not_converging", "air"),
                                                                ("fieldmap.not_converging", "ball")}
    assert all(p["stop_reason"] == "not_converging" and p["accepted"] is False for p in m["patches"])


class _FailingWalk:
    """A patch whose chords cannot be walked: the C++ error of the walk, as the core raises it."""

    def __init__(self, patch):
        self._patch = patch

    def __getattr__(self, name):
        return getattr(self._patch, name)

    def chords(self, *args):
        exc = _core.AxionMultiPhysicsError("the walk of chord 7 of family bulk could not be completed. Hint: report it")
        exc.code = "fieldmap.chord_incomplete"
        exc.hint = "report it"
        raise exc


class _FailingMapper:
    def __init__(self, mapper):
        self._mapper = mapper

    def __getattr__(self, name):
        return getattr(self._mapper, name)

    def patch(self, *args):
        return _FailingWalk(self._mapper.patch(*args))


def test_a_chord_that_cannot_be_walked_stops_the_export_and_writes_nothing(sphere_res, monkeypatch):
    """The core's error is re-raised naming the patch and the family, with its hint, and no directory is left. The
    walk's own failure cannot be provoked on a valid mesh (a tangled one is refused earlier), so the core is stood in
    for at its Python boundary."""
    wd, res = sphere_res
    fmod._mapper(res)
    monkeypatch.setattr(res, "_fieldmap_mapper", _FailingMapper(res._fieldmap_mapper))
    pth = wd / "unwalkable"
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        res.export_field_map(pth, tol=1e-3)
    assert code_of(e) == "fieldmap.chord_incomplete"
    assert "patch 'air'" in str(e.value) and "family bulk" in str(e.value)
    assert e.value.hint
    assert nothing_written(pth)


def shell_doc(thickness):
    """An iron (mu_r = 1000) shell of the given thickness between the sphere of radius 0.5 m and a smaller one, as a
    region, on the sphere's mesh (order 2, 0.05 m on the surface)."""
    a1 = A_SPHERE - thickness
    d = sphere_doc()
    d["name"] = "fieldmap_iron_shell"
    d["regions"] = [{"name": "shell", "material": "fe1000", "shape": {"contour": {"start": [0, -A_SPHERE], "segments": [
        {"ellipse": {"center": [0, 0], "axes": [A_SPHERE, A_SPHERE], "t": [-math.pi / 2, math.pi / 2]}},
        {"line": [0, a1]},
        {"ellipse": {"center": [0, 0], "axes": [a1, a1], "t": [math.pi / 2, -math.pi / 2]}},
        {"line": [0, -A_SPHERE]}]}}}]
    d["mesh"]["surface_size"] = 0.05
    return d


def test_a_shell_that_tangles_at_order_2_is_mapped_on_the_repaired_mesh(tmp_path):
    """A 2 mm shell on a 0.05 m surface mesh at geometry order 2: as meshed, half of its curved elements fold over
    (det J changes sign), and a walk through them cycles. The mesher repairs such a mesh before the solve, or refuses
    it, and the library refuses a tangled mesh when it reads one, so the export never meets a tangled patch (its own
    refusal, fieldmap.tangled_mesh, stays as a second guard): the solve's mesh has no tangled element, and the export
    completes with a patch for the shell. (At tol 1e-3 the shell's patch stops not_converging at level 2 on the pinned
    gmsh, so its acceptance is not asserted here.)"""
    from test_mesher_tangled import PINNED_GMSH, mesher_gmsh_version, tangled_count

    try:
        res = solve(shell_doc(0.002), tmp_path / "solve")
    except axmp.AxionMultiPhysicsError as e:
        assert mesher_gmsh_version() != PINNED_GMSH, f"the pinned gmsh repairs this mesh: {e}"
        pytest.skip(f"this gmsh leaves the shell tangled; the mesher refuses it: {e}")
    info = json.loads((tmp_path / "solve" / "mesh_info.json").read_text())
    ho = info.get("high_order", {})
    assert ho.get("tangled_before", 0) >= 1 and ho.get("tangled_after") == 0, ho
    assert tangled_count(tmp_path / "solve" / "mesh.msh") == 0
    pth = tmp_path / "map"
    res.export_field_map(pth, tol=1e-3)
    P = precs(manifest(pth))
    assert "shell" in P, sorted(P)


def test_a_valid_thin_curved_shell_is_mapped(tmp_path):
    """The same shell at 3 mm (aspect 16) keeps every element's det J positive: it is mapped and accepted, not
    refused as tangled."""
    res = solve(shell_doc(0.003), tmp_path / "solve")
    pth = tmp_path / "map"
    res.export_field_map(pth, tol=1e-3)
    P = precs(manifest(pth))
    assert "shell" in P and P["shell"]["accepted"] is True


def test_a_thin_magnetic_layer_is_an_interface(tmp_path):
    """A thin mu_r = 2e4 layer on the sphere splits the air into two patches whose jump boundary is the layer; a layer
    that were not an interface would leave one patch with no layer family."""
    res = solve(layer_doc(), tmp_path / "solve")
    pth = tmp_path / "map"
    res.export_field_map(pth, tol=1e-3)
    outer, inner = _split(manifest(pth))
    assert outer["curves"] == ["shell"] and inner["curves"] == ["shell"]
    assert "layer" in outer["families"]
    assert inner["field_free"] is False and inner["accepted"] is True


@pytest.mark.parametrize("kw, code", [
    ({"tol": 0.0}, "fieldmap.invalid_option"),
    ({"tol": 1.0}, "fieldmap.invalid_option"),
    ({"regions": ["nope"]}, "fieldmap.invalid_option"),
    ({"outside": "clamp"}, "fieldmap.invalid_option"),
    ({"max_patch_MB": 0.0}, "fieldmap.invalid_option"),
    ({"track_volumes": [{"name": "b", "r": [0.7, 0.4], "z": [0.0, 1.0]}]}, "fieldmap.invalid_option"),
    ({"track_volumes": [{"name": "b", "x": [0, 1], "y": [0, 1], "z": [0, 1]}]}, "fieldmap.invalid_option"),
    ({"track_volumes": [{"name": "a b", "r": [0.1, 0.2], "z": [0.0, 1.0]}]}, "fieldmap.invalid_option"),
    ({"track_volumes": [{"name": "b", "r": [0.1, 0.2], "z": [0, 1]}, {"name": "b", "r": [0.1, 0.3], "z": [0, 1]}]},
     "fieldmap.invalid_option"),
    ({"track_volumes": [{"name": "far", "r": [5.0, 6.0], "z": [0.0, 1.0]}]}, "fieldmap.track_volume_empty"),
])
def test_invalid_options_are_refused_before_anything_is_written(sphere_res, kw, code):
    wd, res = sphere_res
    pth = wd / "refused"
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        res.export_field_map(pth, **kw)
    assert code_of(e) == code
    assert e.value.hint
    assert nothing_written(pth)


# ============================================================================================ curves: a closed sheet
@pytest.fixture(scope="module")
def sheet_res(tmp_path_factory):
    wd = tmp_path_factory.mktemp("sheet")
    return wd, solve(sheet_doc(), wd / "solve")


@pytest.fixture(scope="module")
def closed_sheet(sheet_res):
    wd, res = sheet_res
    path = wd / "map"
    res.export_field_map(path, tol=1e-3)
    return SimpleNamespace(res=res, path=path, m=manifest(path))


def _split(m):
    P = list(m["patches"])
    assert len(P) == 2 and {p["name"] for p in P} == {"air#0", "air#1"}
    inner = [p for p in P if p["box"][0][1] <= A_SPHERE + EDGE and abs(p["box"][1][1]) <= A_SPHERE + EDGE]
    assert len(inner) == 1
    outer = [p for p in P if p is not inner[0]][0]
    return outer, inner[0]


def test_a_closed_sheet_splits_its_region_into_patches(closed_sheet):
    outer, inner = _split(closed_sheet.m)
    for p in (outer, inner):
        assert p["regions"] == ["air"] and p["component"] in (0, 1) and p["curves"] == ["shell"]
    assert {outer["component"], inner["component"]} == {0, 1}
    # the inside of the superconducting sheet is field-free: written as zero, reported, not iterated on
    assert inner["field_free"] is True and inner["stop_reason"] == "field_free" and inner["accepted"] is True
    assert inner["B_patch"] <= FIELD_FREE_RATIO * outer["B_patch"]
    assert inner["families"] == {} and inner["level"] == 0
    assert np.all(array(closed_sheet.path, inner) == 0.0)
    # the outside is an ordinary patch whose jump boundary is the sheet
    assert outer["field_free"] is False
    assert set(outer["families"]) == {"bulk", "layer", "axis"}
    C = chords(closed_sheet.path, outer, "layer")
    rho0 = np.hypot(C["start"][:, 0], C["start"][:, 1])
    assert np.all((rho0 >= A_SPHERE - EDGE) & (rho0 <= A_SPHERE + outer["h_el"] + EDGE))
    check_acceptance(outer, 1e-3)


def test_closed_sheet_chords_are_complete(closed_sheet):
    outer, _ = _split(closed_sheet.m)
    check_chords_complete(closed_sheet.path, outer, lambda fam: sphere_region("air"), (A_SPHERE, DOMAIN), half_plane=True)


def test_closed_sheet_map_accuracy(closed_sheet):
    """Outside a superconducting sphere B = B0 z-hat + the dipole with K = -1/2: recovery across the sheet (B* 39 to
    531 times B_h's error in the measurements) fails the recovery bound."""
    outer, _ = _split(closed_sheet.m)
    out, _ = sphere_exact(0.0)
    check_f3b_accuracy(closed_sheet.path, outer, out)
    for fam in outer["families"]:
        check_statistic_recomputed(closed_sheet.path, outer, fam, 1e-3)


@pytest.mark.parametrize("open_cap", [True, False])
def test_conductor_curves_are_not_interfaces(tmp_path, open_cap):
    """A curve of a normal conductor is transparent to the static field (the uniform 0.1 T on both sides of a copper
    cap): it is not a field jump. Only superconducting sheets and thin magnetic layers split a region or are refused
    when open; a conductor curve, open or closed, leaves the air one patch, with no curve, no layer family, holding
    B = (0, 0.1 T, 0) at every filled node."""
    doc = sheet_doc(open_cap=open_cap)
    doc["curves"][0]["material"] = "copper"
    res = solve(doc, tmp_path / "solve")
    pth = tmp_path / "map"
    res.export_field_map(pth, tol=1e-3)
    m = manifest(pth)
    assert [p["name"] for p in m["patches"]] == ["air"]
    (p,) = m["patches"]
    assert p["component"] is None and p["curves"] == [] and p["field_free"] is False
    assert set(p["families"]) == {"bulk", "axis"}
    check_acceptance(p, 1e-3)
    B = array(pth, p).reshape(-1, 3)
    filled = np.any(B != 0.0, axis=1)
    assert filled.mean() > 0.5
    np.testing.assert_allclose(B[filled], np.tile([0.0, B0, 0.0], (filled.sum(), 1)), rtol=0, atol=1e-12 * B0)


def test_an_open_sheet_is_refused(tmp_path):
    """An open cap has the same component on both sides; no patch can hold a jump that ends inside it."""
    res = solve(sheet_doc(open_cap=True), tmp_path / "solve")
    pth = tmp_path / "map"
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        res.export_field_map(pth, tol=1e-3)
    assert code_of(e) == "fieldmap.curve_in_region"
    assert "shell" in str(e.value) and "air" in str(e.value)
    assert e.value.hint
    assert nothing_written(pth)


# ============================================================================================ boundary nodes (the iron core)
@pytest.fixture(scope="module")
def core_res(tmp_path_factory):
    wd = tmp_path_factory.mktemp("core")
    return wd, solve(core_doc(), wd / "solve")


@pytest.fixture(scope="module")
def core(core_res):
    wd, res = core_res
    path = wd / "map"
    res.export_field_map(path, regions=["core"], tol=1e-4)
    return SimpleNamespace(res=res, path=path, m=manifest(path))


def test_boundary_nodes_hold_the_regions_own_value(core):
    """The core's patch box is the core itself, so its faces r = 0.4 and z = +-0.3 lie on grid lines. Those nodes take
    the core's own B* (one-sided), not the air's: equal to ``recovered_field`` of the core, and on the iron side of
    the physical jump of the tangential field."""
    (p,) = core.m["patches"]
    assert p["name"] == "core" and p["regions"] == ["core"]
    np.testing.assert_allclose(np.asarray(p["box"]), [[0.0, 0.4], [-0.3, 0.3]], atol=1e-9)
    X = nodes(p)
    B = array(core.path, p)
    on = np.zeros(X.shape[:2], bool)
    on[-1, :] = True
    on[:, 0] = True
    on[:, -1] = True
    q = X[on]
    Bm = B[on]
    Bs = np.asarray(fmod.recovered_field(core.res, core.path, "core", q))
    assert np.max(np.abs(Bm - Bs)) <= 1e-12 * p["B_patch"]
    eps = 1e-7
    nrm = np.zeros_like(q)
    nrm[np.isclose(q[:, 0], 0.4), 0] = 1.0
    nrm[np.isclose(q[:, 1], 0.3), 1] = 1.0
    nrm[np.isclose(q[:, 1], -0.3), 1] = -1.0
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1)[:, None]
    si = core.res.sample(*(q - eps * nrm).T)
    so = core.res.sample(*(q + eps * nrm).T)
    B_in = np.column_stack([si["Br"], si["Bz"], si["Bphi"]])
    B_out = np.column_stack([so["Br"], so["Bz"], so["Bphi"]])
    jump = np.linalg.norm(B_out - B_in, axis=1)
    big = jump > 0.05 * p["B_patch"]
    assert big.sum() >= 10
    assert np.all(np.linalg.norm(Bm[big] - B_in[big], axis=1) <= 0.25 * jump[big])


def test_trial_points_just_outside_the_core_are_clamped(core):
    """The core's faces r = 0.4 and z = +-0.3 are its patch box. A point just outside (a stepper's trial point) within
    one level-0 spacing is clamped onto the face and gets the face's value; beyond it, the outside rule refuses."""
    (p,) = core.m["patches"]
    assert p["margin"] == pytest.approx(p["levels"][0]["spacing"], rel=1e-15)
    pat = fmod.load(core.path).patch("core")
    m_r, m_z = p["margin"]
    for out, on in (((0.4 + 1e-6, 0.0), (0.4, 0.0)), ((0.4 + 0.9 * m_r, 0.1), (0.4, 0.1)),
                    ((0.2, 0.3 + 0.5 * m_z), (0.2, 0.3)), ((0.4 + 1e-9, -0.3 - 1e-9), (0.4, -0.3))):
        assert np.array_equal(np.asarray(pat.evaluate_native([out])), np.asarray(pat.evaluate_native([on]))), out
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        pat.evaluate_native([(0.4 + 1.01 * m_r, 0.0)])
    assert code_of(e) == "fieldmap.outside_map"


def test_core_patch_converges(core):
    (p,) = core.m["patches"]
    assert set(p["families"]) == {"bulk", "layer", "axis"}
    check_grid_rule(p)
    check_acceptance(p, 1e-4)
    for fam in p["families"]:
        check_statistic_recomputed(core.path, p, fam, 1e-4)


def test_the_core_axis_family_converges_where_a_vertex_rule_would_stall(core_res):
    """The axis rule takes B_z from elements with an edge on the axis only. Including the elements that touch the axis
    at one vertex makes the axis family's E level off near 5e-6 on this saturated core (measured: E falling 2.1, 1.5,
    1.4 per halving from level 2); at tol = 1e-6 the axis family's tol_eff (0.1 H, about 4e-6) lies below that level,
    so the healthy map must converge there (two-halving gains >= 11 measured, 3.0 required) and a stalled one is
    stopped by the guard."""
    wd, res = core_res
    pth = wd / "axis_tight"
    res.export_field_map(pth, regions=["core"], tol=1e-6)
    (p,) = manifest(pth)["patches"]
    ax = p["families"]["axis"]
    assert ax["acceptance"] is True and ax["tol_eff"] < 5e-6, "the axis family is driven below the stall level"
    check_acceptance(p, 1e-6)
    assert ax["E"] <= ax["tol_eff"]


# ============================================================================================ the axis and merging (solenoid)
@pytest.fixture(scope="module")
def solenoid_res(tmp_path_factory):
    wd = tmp_path_factory.mktemp("solenoid")
    return wd, solve(solenoid_doc(), wd / "solve")


@pytest.fixture(scope="module")
def solenoid(solenoid_res):
    wd, res = solenoid_res
    path = wd / "map"
    res.export_field_map(path, tol=1e-3)
    return SimpleNamespace(res=res, path=path, m=manifest(path), wd=wd)


def test_regions_without_a_jump_share_a_patch(solenoid):
    (p,) = solenoid.m["patches"]
    assert p["name"] == "air+winding" and p["regions"] == ["air", "winding"]
    assert set(p["families"]) == {"bulk", "axis"}, "no jump boundary: no layer family (outer boundary and axis excluded)"
    assert p["touches_axis"] is True
    check_acceptance(p, 1e-3)


def test_axis_rule_on_a_solenoid(solenoid):
    """B_z on the axis from the edge elements' own limits: the axis family's B* is as accurate as B_h (taking B_z from
    the nearest off-axis node made it 17 times worse on this solenoid in the measurements)."""
    (p,) = solenoid.m["patches"]
    B = array(solenoid.path, p)
    assert np.all(B[0, :, 0] == 0.0) and np.all(B[0, :, 2] == 0.0)
    z = p["origin"][1] + p["spacing"][1] * np.arange(p["shape"][1])
    zz = z[np.abs(z) < 1.4]
    q = np.column_stack([np.zeros_like(zz), zz])
    np.testing.assert_allclose(np.asarray(fmod.recovered_field(solenoid.res, solenoid.path, p["name"], q))[:, 1],
                               solenoid_exact(q)[:, 1], atol=2e-3 * p["B_patch"])
    acc = check_f3b_accuracy(solenoid.path, p, solenoid_exact, n_chords=512, sub=4)
    assert acc["axis"][1] <= 1.5 * acc["axis"][2] + 1e-4


def test_unmerged_regions_and_thin_patches(solenoid):
    pth = solenoid.wd / "unmerged"
    solenoid.res.export_field_map(pth, tol=1e-3, merge=False)
    m = manifest(pth)
    P = precs(m)
    assert set(P) == {"air", "winding"} and m["control"]["merge"] is False
    w = P["winding"]
    assert w["levels"][0]["shape"][0] == MIN_NODES, "a 5 cm winding still gets 9 nodes across"
    check_grid_rule(w)
    for p in P.values():
        assert "layer" in p["families"]
        check_acceptance(p, 1e-3)
    check_f3b_accuracy(pth, P["winding"], solenoid_exact, n_chords=256, sub=4)


# ============================================================================================ superconducting region, outside rule
def test_superconducting_region_and_the_outside_rule(tmp_path):
    """An ideal superconducting region gets no patch (B = 0 there); with outside: "zero" the manifest records the
    largest |B| on the mapped domain's outer boundary relative to B_patch and warns above 1e-3. Outside a
    superconducting sphere: max |B| over the air is 1.5 B0 at its equator; on the domain's arc rho = 1.5 m the exact
    maximum is computed here."""
    res = solve(sphere_doc(material="superconductor"), tmp_path / "solve")
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        res.export_field_map(tmp_path / "ball", regions=["ball"], tol=1e-3)
    assert code_of(e) == "fieldmap.invalid_option" and nothing_written(tmp_path / "ball")
    pth = tmp_path / "map"
    res.export_field_map(pth, tol=1e-3, outside="zero")
    m = manifest(pth)
    assert [p["name"] for p in m["patches"]] == ["air"]
    assert m["zero_field_regions"] == [{"region": "ball", "reason": "superconductor"}]
    assert m["outside"] == "zero"
    (p,) = m["patches"]
    assert p["B_patch"] == pytest.approx(1.5 * B0, rel=0.01)
    th = np.linspace(0.0, np.pi, 20001)
    out, _ = sphere_exact(0.0)
    b = out(np.column_stack([DOMAIN * np.sin(th), DOMAIN * np.cos(th)]))
    ratio = np.max(np.linalg.norm(b, axis=1)) / (1.5 * B0)
    assert m["boundary_ratio"] == pytest.approx(ratio, rel=0.02)
    assert any(w["code"] == "fieldmap.outside_field_not_small" for w in m["warnings"])
    assert "layer" in p["families"], "the superconductor's surface is a jump boundary"
    check_f3b_accuracy(pth, p, out)


# ============================================================================================ the measured cases at the default tol (AXMP_SLOW)
SLOW_CASES = ["sphere", "closed_sheet", "thin_layer", "solenoid", "knee_toroid", "core"]
# Cases with singular (re-entrant or convex iron) corners: there a handful of millimetre chords leaving the corner hold
# most of E^2 and the seed spread of E reached 3.9 in the measurements (3.2 on this core); it is reported, not gated.
# The gate of 2.5 applies to the exact cases, which have no singular corner (measured <= 1.82).
SINGULAR_CORNER = {"core"}


@slow
@pytest.mark.parametrize("case", SLOW_CASES)
def test_accuracy_and_reproducibility_at_the_default_tolerance(tmp_path, case):
    """Every rule above on the measured cases at tol = 1e-4, every chord (the solenoid: 4096 per
    family, for the cost of its exact field)."""
    kw = {}
    n_chords = None
    if case == "sphere":
        doc = sphere_doc()
        o, i = sphere_exact(1000.0)
        exact = {"air": o, "ball": i}
    elif case == "closed_sheet":
        doc = sheet_doc()
        o, _ = sphere_exact(0.0)
        exact = {"outer": o}
    elif case == "thin_layer":
        doc = layer_doc()
        o, i = layer_exact()
        exact = {"outer": o, "inner": i}
    elif case == "solenoid":
        doc = solenoid_doc()
        exact = {"air+winding": solenoid_exact}
        n_chords = 4096
    elif case == "knee_toroid":
        doc = toroid_doc()
        kw["regions"] = ["fe"]
        exact = None
    else:
        doc = core_doc()
        exact = {}
    res = solve(doc, tmp_path / "solve")
    if case == "knee_toroid":
        s = res.sample([1.35], [0.0])
        exact = {"fe": knee_exact(1.0 if s["Bphi"][0] >= 0 else -1.0)}
    pth = tmp_path / "map"
    res.export_field_map(pth, **kw)
    m = manifest(pth)
    assert m["tol"] == 1e-4
    P = precs(m)
    if case in ("closed_sheet", "thin_layer"):
        outer, inner = _split(m)
        P = {"outer": outer, "inner": inner}
        assert inner["field_free"] is (case == "closed_sheet")
    other = fmod.chord_error(res, pth, seed=CHORD_SEED + 1)
    for key, p in P.items():
        if p["field_free"]:
            continue
        check_grid_rule(p)
        check_acceptance(p, 1e-4)
        for fam in p["families"]:
            check_statistic_recomputed(pth, p, fam, 1e-4)
            ratio = other[p["name"]][fam]["E"] / p["families"][fam]["E"]
            assert math.isfinite(ratio) and ratio > 0
            if case in SINGULAR_CORNER:
                print(f"{case} {p['name']}/{fam}: E with another seed / reported = {ratio:.2f} (reported, not gated)")
            elif p["families"][fam]["acceptance"]:
                assert 1 / 2.5 <= ratio <= 2.5, f"{p['name']}/{fam}: seed spread {ratio:.2f}"
        if key in exact:
            check_f3b_accuracy(pth, p, exact[key], n_chords=n_chords, sub=4 if case == "solenoid" else 8)
