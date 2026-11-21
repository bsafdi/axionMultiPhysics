"""Field maps (axionmultiphysics.fieldmap, beta): the format, the evaluator and the constants, without a solve.

The particle-transport design note, section 5 (field maps) and its validation plan, with the corrections measured on
solved fields. This file holds the parts of the field-to-transport checks that need no particle transport (the map's
own path: nodal values written by ``write_map``, read by ``load``, interpolated by ``FieldMap.evaluate``): the
interpolation of closed-form fills, a helix in a uniform map, the canonical angular momentum of protons through a
current loop's map; the outside-the-map rule; the scrambled-Halton sequence of the format; the refinement decision
(``decide``: acceptance and the two-halving guard); the documentation. Every reference is independent of the package:
tests/_fieldmap_ref.py (closed forms with their sources, the sequence written from its specification, the test's own
multilinear interpolation and a Runge-Kutta tracker).

Tolerances:
- interpolation: the error of multilinear interpolation in a cell is at most sum_a (h_a^2 / 8) max |d^2 B / dx_a^2| per
  component (the tensor-product interpolation bound); the maximum over the cell is taken at its corners and centre,
  which can miss the true maximum by a little, hence the factor 1.25. The observed order of the pointwise error (its
  RMS over 1e4 random points, which unlike the maximum does not depend on where the worst point happens to fall in its
  cell) under halving is 2.0 +- 0.2.
- uniform fields and the quadrupole (linear in the coordinates) are reproduced exactly by multilinear interpolation:
  the error is round-off, 1e-14 of the field scale.
- the helix: a fourth-order Runge-Kutta track with 500 steps per turn has a phase error below 1e-7 over 10 turns; the
  radius, pitch and end point are checked to 1e-6 relative (the design note's ceiling for the Geant4 path is 1e-4).
- the loop: the canonical angular momentum P_phi = r p_phi + q psi is conserved along the exact field to the
  integrator's error (1e-9 of r |p|); through a map d(P_phi)/ds = 0.2998 q [u_z r dB_r - u_r r dB_z] with
  dB = B_map - B_exact, so |Delta P_phi| <= 0.2998 |q| int r |dB| ds plus the integrator's error; and |dB| itself stays
  within the interpolation bound at every point of the track.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import axionmultiphysics as axmp
from axionmultiphysics import fieldmap as fmod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fieldmap_ref as ref  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RNG_SEED = 12345


def _rng(k=0):
    return np.random.default_rng(RNG_SEED + k)


def _code(excinfo):
    return getattr(excinfo.value, "code", None)


# ============================================================================================ the references themselves
def test_reference_elliptic_integrals():
    # A&S table 17.1 / 17.5 values; K(0) = E(0) = pi/2
    K, E = ref.ellipke(np.array([0.0, 0.5, 0.9]))
    assert K[0] == pytest.approx(math.pi / 2, rel=1e-15) and E[0] == pytest.approx(math.pi / 2, rel=1e-15)
    assert K[1] == pytest.approx(1.8540746773013719, rel=1e-14)
    assert E[1] == pytest.approx(1.3506438810476755, rel=1e-14)
    assert K[2] == pytest.approx(2.5780921133481733, rel=1e-14)
    assert E[2] == pytest.approx(1.1047747327040733, rel=1e-14)


def test_reference_loop_is_the_curl_of_its_potential():
    a, current = 0.5, 1e5
    r = np.array([0.0, 0.05, 0.2, 0.3, 0.7, 1.1])
    z = np.array([0.3, -0.4, 0.1, 0.25, -0.6, 0.9])
    Br, Bz = ref.loop_B(a, current, r, z)
    # on the axis: mu0 I a^2 / (2 (a^2 + z^2)^(3/2))
    assert Bz[0] == pytest.approx(ref.MU0 * current * a * a / (2 * (a * a + z[0] ** 2) ** 1.5), rel=1e-14)
    assert Br[0] == 0.0
    d = 1e-5
    rr, zz = r[1:], z[1:]
    dpsi_dr = (ref.loop_psi(a, current, rr + d, zz) - ref.loop_psi(a, current, rr - d, zz)) / (2 * d)
    dpsi_dz = (ref.loop_psi(a, current, rr, zz + d) - ref.loop_psi(a, current, rr, zz - d)) / (2 * d)
    np.testing.assert_allclose(Bz[1:], dpsi_dr / rr, rtol=1e-7)
    np.testing.assert_allclose(Br[1:], -dpsi_dz / rr, rtol=1e-6, atol=1e-12)


def test_reference_solenoid_sheet_and_thick_winding():
    R, b, K_s = 0.5, 0.5, 2e5
    _, Bz0 = ref.sheet_B(R, b, K_s, 0.0, 0.0)
    assert float(Bz0) == pytest.approx(ref.MU0 * K_s * b / math.hypot(R, b), rel=1e-13)
    R, H, tw, NI = 0.5, 1.0, 0.05, 2e5
    r = np.array([0.0, 0.2, 0.3, 0.8, 0.1, 1.0])
    z = np.array([0.0, 0.3, -0.7, 0.2, 0.9, -1.2])
    Br, Bz = ref.solenoid_B(R, H, tw, NI, r, z)
    Br2, Bz2 = ref.solenoid_B_by_loops(R, H, tw, NI, r, z)
    scale = ref.MU0 * NI / H
    np.testing.assert_allclose(Br, Br2, atol=1e-9 * scale)
    np.testing.assert_allclose(Bz, Bz2, atol=1e-9 * scale)


def test_reference_sphere_shell_and_image_solution():
    # sphere: B_n and H_t continuous at the surface
    B0, a = 0.1, 0.5
    for mu in (1000.0, 2.5):
        K, f = ref.sphere_K_Bin(mu)
        th = np.linspace(0.1, 3.0, 7)
        r, z = a * np.sin(th), a * np.cos(th)
        Br, Bz = ref.dipole_exterior(B0, K, a, r, z)
        Bn_out = Br * np.sin(th) + Bz * np.cos(th)
        Bt_out = Br * np.cos(th) - Bz * np.sin(th)
        Bn_in = f * B0 * np.cos(th)
        Bt_in = -f * B0 * np.sin(th)
        np.testing.assert_allclose(Bn_out, Bn_in, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(Bt_out * mu, Bt_in, rtol=1e-12, atol=1e-15)
    # shell: mu = 1 is no shell; a vanishing inner radius is the solid sphere
    K, f = ref.shell_K_Bin(0.49, 0.51, 1.0)
    assert abs(K) < 1e-14 and f == pytest.approx(1.0, rel=1e-14)
    K, f = ref.shell_K_Bin(1e-4, 0.5, 50.0)
    K0, f0 = ref.sphere_K_Bin(50.0)
    assert K == pytest.approx(K0, rel=1e-9)
    # images: B_n and H_t continuous at z = 0
    rr = np.linspace(0.05, 1.2, 9)
    for mu in (1000.0, 3.0):
        ba = ref.loop_over_halfspace(0.4, 1e4, 0.2, mu, rr, 0.0, "air")
        bi = ref.loop_over_halfspace(0.4, 1e4, 0.2, mu, rr, 0.0, "iron")
        np.testing.assert_allclose(ba[1], bi[1], rtol=1e-12)
        np.testing.assert_allclose(ba[0], bi[0] / mu, rtol=1e-12, atol=1e-18)


def test_reference_tracker_reproduces_a_helix():
    def field(x):
        return np.array([0.0, 0.0, 1.0])
    R = 1.0 / ref.C_GEV
    n = 500
    xs, _ = ref.rk4_track(field, [0, 0, 0], [1, 0, 0], 1.0, 1.0, 2 * math.pi * R / n, n)
    assert np.linalg.norm(xs[-1]) < 1e-8 * R
    assert np.max(np.abs(np.hypot(xs[:, 0], xs[:, 1] + R) - R)) < 1e-9 * R


def test_reference_chord_integrals_of_a_map():
    """The test's exact chord integral of a multilinear map: a bilinear field is reproduced by its nodes, and its
    integral along a segment is a polynomial integral done here in closed form; a trilinear map against dense
    sampling of the same interpolant."""
    c = np.array([[0.3, -0.2, 0.7], [1.1, 0.4, -0.3], [0.05, 2.0, 0.1], [0.8, -0.6, 0.2]])   # f = c0 + c1 r + c2 z + c3 r z

    def f(r, z):
        return c[0] + np.multiply.outer(r, c[1]) + np.multiply.outer(z, c[2]) + np.multiply.outer(r * z, c[3])
    origin, spacing = (0.0, -1.0), (0.1, 0.25)
    R, Z = np.meshgrid(origin[0] + spacing[0] * np.arange(13), origin[1] + spacing[1] * np.arange(9), indexing="ij")
    B = f(R, Z)
    start = np.array([[0.13, -0.77], [1.0, 0.9], [0.5, 0.0]])
    th = np.array([0.3, 3.9, 1.5707963267948966])
    direction = np.column_stack([np.cos(th), np.sin(th)])
    seg = np.array([[0.0, 1.0], [0.1, 0.4], [0.6, 1.2], [0.0, 0.8]])
    seg_chord = np.array([0, 1, 1, 2])
    got = ref.map_chord_integrals(origin, spacing, B, start, direction, seg, seg_chord, 3)
    want = np.zeros((3, 3))
    for (s0, s1), k in zip(seg, seg_chord):
        p, v = start[k], direction[k]
        # integral of c0 + c1 (p0 + s v0) + c2 (p1 + s v1) + c3 (p0 + s v0)(p1 + s v1) ds
        m0, m1, m2 = s1 - s0, (s1 ** 2 - s0 ** 2) / 2, (s1 ** 3 - s0 ** 3) / 3
        want[k] += (c[0] * m0 + c[1] * (p[0] * m0 + v[0] * m1) + c[2] * (p[1] * m0 + v[1] * m1)
                    + c[3] * (p[0] * p[1] * m0 + (p[0] * v[1] + p[1] * v[0]) * m1 + v[0] * v[1] * m2))
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-14)
    rng = _rng(9)
    B3 = rng.normal(size=(5, 6, 7, 3))
    o3, h3 = (0.0, 0.0, 0.0), (0.3, 0.2, 0.25)
    st = np.array([[0.1, 0.1, 0.1]])
    v = np.array([[0.6, 0.5, 0.62]])
    v /= np.linalg.norm(v)
    sg = np.array([[0.0, 1.4]])
    got = ref.map_chord_integrals(o3, h3, B3, st, v, sg, np.array([0]), 1)
    s = (np.arange(400000) + 0.5) / 400000 * 1.4
    dense = ref.interp_native(o3, h3, B3, st + s[:, None] * v).sum(axis=0) * 1.4 / 400000
    np.testing.assert_allclose(got[0], dense, rtol=1e-8, atol=1e-10)


# ============================================================================================ constants of the format
def test_constants_of_the_format():
    assert fmod.SCHEMA == "axionmultiphysics.fieldmap/1"
    assert fmod.CHORD_SEED == 0x5DEECE66D
    assert fmod.HALTON_BASES == (2, 3, 5, 7, 11, 13)
    assert (fmod.N_BULK, fmod.N_LAYER, fmod.N_AXIS, fmod.N_TRACK) == (16384, 4096, 4096, 4096)
    assert fmod.CHORD_LENGTH_H_EL == 8.0
    assert fmod.FLOOR == 0.1
    assert fmod.TOL_EFF_FACTOR == 0.1
    assert fmod.GUARD_TWO_HALVINGS == 3.0
    assert fmod.FIELD_FREE_RATIO == 1e-10
    assert fmod.MIN_NODES == 9
    assert fmod.POINTWISE_POINTS == 100000


# ============================================================================================ scrambled Halton
@pytest.mark.parametrize("seed", [0x5DEECE66D, 0x5DEECE66E, 0, 2 ** 64 - 1])
def test_halton_sequence_is_the_specified_one_bitwise(seed):
    n = 1200
    got = np.asarray(fmod.halton(seed, n, 6))
    want = ref.halton(seed, n, 6)
    assert got.shape == (n, 6) and got.dtype == np.float64
    assert np.array_equal(got, want), "the sequence must be bitwise the specified one (no fused multiply-add)"
    assert np.all(got >= 0.0) and np.all(got < 1.0)


def test_halton_prefix_and_dims_are_consistent():
    a = np.asarray(fmod.halton(fmod.CHORD_SEED, 500, 6))
    b = np.asarray(fmod.halton(fmod.CHORD_SEED, 50, 4))
    assert np.array_equal(a[:50, :4], b)


def test_family_seed():
    for idx in (0, 1, 2, 3, 7):
        assert fmod.family_seed(fmod.CHORD_SEED, idx) == ref.family_seed(fmod.CHORD_SEED, idx)
    assert fmod.family_seed(fmod.CHORD_SEED, 0) == fmod.CHORD_SEED


# ============================================================================================ the refinement decision
_rule = ref.decide_rule


DECIDE_CASES = [
    ({"bulk": [1e-3]}, {"bulk": 1e-4}),
    ({"bulk": [1e-3, 2e-4, 5e-5]}, {"bulk": 1e-4}),
    ({"bulk": [1e-3, 5e-4, 4e-4]}, {"bulk": 1e-4}),
    ({"bulk": [0.75, 0.5, 0.25]}, {"bulk": 1e-4}),                       # exactly 3.0: not a stall
    ({"bulk": [0.75, 0.5, 0.2500001]}, {"bulk": 1e-4}),                  # just below 3.0
    ({"bulk": [1e-3, 5e-4, 4e-4], "axis": [1e-6, 9e-7, 8e-7]}, {"bulk": 1e-4, "axis": 1e-4}),
    ({"bulk": [1e-3, 2e-4, 1.1e-4], "axis": [1e-6, 9e-7, 8e-7]}, {"bulk": 1e-4, "axis": 1e-4}),  # stalls below tol
    ({"bulk": [2e-4, 1.5e-4, 9e-5], "layer": [1e-4, 8e-5, 7e-5]}, {"bulk": 1e-4, "layer": 1e-4}),  # accepted first
    ({"bulk": [1.0, 0.5, 0.4], "layer": [1.0, 0.6, 0.5]}, {"bulk": 1e-4, "layer": 1e-4}),         # smallest gain
    ({"bulk": [1.0, 0.5, 0.5], "layer": [1.0, 0.6, 0.5]}, {"bulk": 1e-4, "layer": 1e-4}),         # tie: alphabetical
    ({"bulk": [1.0, 0.9]}, {"bulk": 1e-4}),                              # the guard needs three grids
    ({"bulk": [1.0, 0.0]}, {"bulk": 1e-4}),
    ({"bulk": [1e-3, 2e-4, 1.5e-4], "layer": [3e-3, 1e-3, 3e-4]}, {"bulk": 1e-4, "layer": 2e-4}),
]


@pytest.mark.parametrize("E, tol_eff", DECIDE_CASES)
def test_decide_is_the_acceptance_rule_and_the_two_halving_guard(E, tol_eff):
    got = fmod.decide(E, tol_eff)
    want = _rule(E, tol_eff)
    assert got["status"] == want["status"]
    assert got["level"] == want["level"]
    assert got["family"] == want["family"]
    if want["gain"] is None:
        assert got["gain"] is None
    else:
        assert got["gain"] == pytest.approx(want["gain"], rel=1e-15)


def test_decide_refuses_ragged_histories():
    with pytest.raises((ValueError, axmp.AxionMultiPhysicsError)):
        fmod.decide({"bulk": [1.0, 0.5], "layer": [1.0]}, {"bulk": 1e-4, "layer": 1e-4})


# ============================================================================================ writing and reading
def _kind1_patch(name, r, z, fill):
    """Nodal values of a kind-1 patch on the grid r[0] + i h_r, z[0] + j h_z (r, z: (lo, hi, n))."""
    nr, nz = r[2], z[2]
    hr, hz = (r[1] - r[0]) / (nr - 1), (z[1] - z[0]) / (nz - 1)
    R, Z = np.meshgrid(r[0] + hr * np.arange(nr), z[0] + hz * np.arange(nz), indexing="ij")
    B = np.zeros((nr, nz, 3))
    br, bz = fill(R, Z)
    B[..., 0], B[..., 1] = br, bz
    return {"name": name, "kind": 1, "origin": [r[0], z[0]], "spacing": [hr, hz], "B": B}


def _kind3_patch(name, box, n, fill, transform=None):
    h = [(box[a][1] - box[a][0]) / (n[a] - 1) for a in range(3)]
    X, Y, Z = np.meshgrid(*[box[a][0] + h[a] * np.arange(n[a]) for a in range(3)], indexing="ij")
    B = np.stack(fill(X, Y, Z), axis=-1).astype(float)
    p = {"name": name, "kind": 3, "origin": [box[a][0] for a in range(3)], "spacing": h, "B": B}
    if transform is not None:
        p["transform"] = np.asarray(transform).tolist()
    return p


def _uniform(b):
    return lambda *X: tuple(np.full(X[0].shape, c) for c in b)


def test_write_map_and_load_round_trip(tmp_path):
    p1 = _kind1_patch("air", (0.0, 1.0, 11), (-1.0, 1.0, 21), lambda R, Z: (0.1 * R * Z, 0.3 + 0.0 * R))
    p1["B"][..., 2] = 0.02
    p3 = _kind3_patch("box", ((-1, 1), (-0.5, 0.5), (0, 2)), (5, 3, 9), _uniform((0.1, -0.2, 0.3)))
    fm = fmod.write_map(tmp_path / "m", [p1, p3])
    m = json.loads((tmp_path / "m" / "manifest.json").read_text())
    assert m["schema"] == "axionmultiphysics.fieldmap/1"
    assert m["units"] == {"length": "m", "B": "T"}
    assert m["outside"] == "refuse"
    assert m["source"]["type"] == "nodal_values"
    assert [p["name"] for p in m["patches"]] == ["air", "box"]
    for k, (rec, given) in enumerate(zip(m["patches"], (p1, p3))):
        assert rec["dir"] == f"patch_{k}"
        assert rec["file"] == f"patch_{k}/B.npy"
        assert rec["kind"] == given["kind"]
        assert rec["origin"] == pytest.approx(given["origin"], rel=0, abs=0)
        assert rec["spacing"] == pytest.approx(given["spacing"], rel=0, abs=0)
        assert rec["shape"] == list(given["B"].shape[:-1])
        raw = (tmp_path / "m" / rec["file"]).read_bytes()
        assert rec["sha256"] == hashlib.sha256(raw).hexdigest()
        assert rec["bytes"] == given["B"].size * 8
        arr = np.load(tmp_path / "m" / rec["file"])
        assert arr.dtype.str == "<f8" and arr.flags["C_CONTIGUOUS"]
        assert np.array_equal(arr, given["B"])
    assert m["patches"][0]["frame"] == {"type": "axisymmetric", "axis": "z"}
    assert m["patches"][1]["frame"]["type"] == "cartesian"
    assert np.array_equal(np.asarray(m["patches"][1]["frame"]["transform"]), np.eye(4))
    text = (tmp_path / "m" / "manifest.json").read_text()
    assert str(tmp_path) not in text and str(Path.home()) not in text, "a user file names no local path"
    again = fmod.load(tmp_path / "m")
    assert [p.name for p in again.patches] == ["air", "box"]
    assert np.array_equal(again.patch("air").B, p1["B"])
    assert again.patch("box").kind == 3 and tuple(again.patch("box").shape) == (5, 3, 9)
    assert again.manifest == m
    assert isinstance(fm, fmod.FieldMap)


def test_write_map_refuses_inconsistent_patches(tmp_path):
    good = _kind1_patch("a", (0.0, 1.0, 5), (-1.0, 1.0, 5), lambda R, Z: (0 * R, 0 * R + 1))
    bad_shape = dict(good, B=good["B"][..., :2])
    bad_spacing = dict(good, spacing=[0.0, 0.5])
    one_node = dict(good, B=good["B"][:1])
    bad_kind = dict(good, kind=2)
    for k, p in enumerate((bad_shape, bad_spacing, one_node, bad_kind)):
        with pytest.raises(axmp.AxionMultiPhysicsError) as e:
            fmod.write_map(tmp_path / f"bad{k}", [p])
        assert _code(e) == "fieldmap.invalid_option"
        assert not (tmp_path / f"bad{k}").exists()
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        fmod.write_map(tmp_path / "bad_outside", [good], outside="clamp")
    assert _code(e) == "fieldmap.invalid_option"


# ============================================================================================ interpolation of closed-form fills
def _loop_native(a, current):
    def f(q):
        q = np.atleast_2d(q)
        rr = np.abs(q[:, 0])
        br, bz = ref.loop_B(a, current, rr, q[:, 1])
        return np.column_stack([np.sign(q[:, 0]) * br, bz, np.zeros_like(bz)])
    return f


def _sheet_native(R, b, K_s):
    def f(q):
        q = np.atleast_2d(q)
        rr = np.abs(q[:, 0])
        br, bz = ref.sheet_B(R, b, K_s, rr, q[:, 1])
        return np.column_stack([np.sign(q[:, 0]) * br, bz, np.zeros_like(bz)])
    return f


def _interpolation_kind1(tmp_path, native, rbox, zbox, nr0, levels=4):
    rng = _rng(1)
    n = 10000
    r = rng.uniform(rbox[0], rbox[1], n)
    z = rng.uniform(zbox[0], zbox[1], n)
    phi = rng.uniform(-math.pi, math.pi, n)
    xyz = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    exact = native(np.column_stack([r, z]))
    exact_cart = np.column_stack([exact[:, 0] * np.cos(phi), exact[:, 0] * np.sin(phi), exact[:, 1]])
    errs = []
    for lev in range(levels):
        nr = (nr0 - 1) * 2 ** lev + 1
        nz = int(round((zbox[1] - zbox[0]) / (rbox[1] - rbox[0]))) * (nr - 1) + 1
        p = _kind1_patch("p", (rbox[0], rbox[1], nr), (zbox[0], zbox[1], nz), lambda R, Z: tuple(
            native(np.column_stack([R.ravel(), Z.ravel()]))[:, c].reshape(R.shape) for c in (0, 1)))
        fm = fmod.write_map(tmp_path / f"k1_{lev}", [p])
        got = np.asarray(fm.evaluate(xyz))
        err = np.linalg.norm(got - exact_cart, axis=1)
        bound = ref.second_derivative_bound(native, np.column_stack([r, z]), (p["origin"], p["spacing"]))
        allowed = 1.25 * np.linalg.norm(bound, axis=1) + 1e-14 * np.max(np.abs(exact))
        worst = int(np.argmax(err / allowed))
        assert np.all(err <= allowed), (f"level {lev}: error {err[worst]:.3e} > bound {allowed[worst]:.3e} at "
                                        f"r = {r[worst]:.4f}, z = {z[worst]:.4f}")
        native_got = np.asarray(fm.patch("p").evaluate_native(np.column_stack([r, z])))
        np.testing.assert_allclose(native_got[:, :2], exact[:, :2], atol=float(np.max(allowed)))
        errs.append(ref.rms(err))
    orders = [math.log2(errs[k] / errs[k + 1]) for k in range(levels - 1)]
    assert all(1.8 <= o <= 2.2 for o in orders), f"observed orders {orders}"


def test_interpolation_kind1_loop(tmp_path):
    _interpolation_kind1(tmp_path, _loop_native(0.5, 1e5), (0.0, 0.3), (-0.3, 0.3), 7)


def test_interpolation_kind1_finite_solenoid(tmp_path):
    _interpolation_kind1(tmp_path, _sheet_native(0.5, 0.5, 2e5), (0.0, 0.35), (-0.7, 0.7), 8)


def _loop_cart(a, current):
    def f(q):
        q = np.atleast_2d(q)
        return np.column_stack(ref.loop_B_cartesian(a, current, q[:, 0], q[:, 1], q[:, 2]))
    return f


def test_interpolation_kind3_loop(tmp_path):
    f = _loop_cart(0.5, 1e5)
    box = ((-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25))
    rng = _rng(2)
    pts = rng.uniform(-0.25, 0.25, (10000, 3))
    exact = f(pts)
    errs = []
    for lev, n in enumerate((6, 11, 21, 41)):
        p = _kind3_patch("p", box, (n, n, n), lambda X, Y, Z: tuple(
            f(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))[:, c].reshape(X.shape) for c in range(3)))
        fm = fmod.write_map(tmp_path / f"k3_{lev}", [p])
        got = np.asarray(fm.evaluate(pts))
        err = np.linalg.norm(got - exact, axis=1)
        bound = ref.second_derivative_bound(f, pts, (p["origin"], p["spacing"]))
        allowed = 1.25 * np.linalg.norm(bound, axis=1) + 1e-14 * np.max(np.abs(exact))
        assert np.all(err <= allowed), f"level {lev}: {np.max(err / allowed):.3f} times the bound"
        errs.append(ref.rms(err))
    orders = [math.log2(errs[k] / errs[k + 1]) for k in range(3)]
    assert all(1.8 <= o <= 2.2 for o in orders), f"observed orders {orders}"


def test_quadrupole_and_uniform_fields_are_exact(tmp_path):
    g = 2.0
    box = ((-0.5, 0.5), (-0.5, 0.5), (-1.0, 1.0))
    p = _kind3_patch("q", box, (5, 7, 9), lambda X, Y, Z: ref.quadrupole_B(g, X, Y, Z))
    fm = fmod.write_map(tmp_path / "quad", [p])
    pts = _rng(3).uniform(-0.5, 0.5, (5000, 3)) * np.array([1, 1, 2])
    pts[:100, :2] = 0.0                                   # the zero-field axis
    got = np.asarray(fm.evaluate(pts))
    want = np.column_stack(ref.quadrupole_B(g, pts[:, 0], pts[:, 1], pts[:, 2]))
    assert np.max(np.abs(got - want)) <= 1e-14 * g
    assert np.max(np.abs(got[:100])) <= 1e-14 * g


def test_placement_transform(tmp_path):
    th = math.radians(30.0)
    Rm = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]]) @ \
        np.array([[math.cos(0.7), -math.sin(0.7), 0], [math.sin(0.7), math.cos(0.7), 0], [0, 0, 1]])
    t = np.array([0.1, -0.2, 0.3])
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    f = _loop_cart(0.5, 1e5)
    box = ((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))
    p = _kind3_patch("p", box, (33, 33, 33), lambda X, Y, Z: tuple(
        f(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))[:, c].reshape(X.shape) for c in range(3)), T)
    fm = fmod.write_map(tmp_path / "placed", [p])
    local = _rng(4).uniform(-0.2, 0.2, (3000, 3))
    glob = local @ Rm.T + t
    want = f(local) @ Rm.T
    got = np.asarray(fm.evaluate(glob))
    bound = ref.second_derivative_bound(f, local, (p["origin"], p["spacing"]))
    allowed = 1.25 * np.linalg.norm(bound, axis=1) + 1e-13
    assert np.all(np.linalg.norm(got - want, axis=1) <= allowed)


def test_patches_on_both_sides_of_a_jump(tmp_path):
    """The rule of section 5.2 at the level of the format: two patches meet at the plane z = 0 of a permeable
    half-space (mu = 1000), each filled with its own side's field continued across the plane. Inside its own half,
    each patch is accurate to the interpolation bound up to the plane, including in the cells that straddle it; the
    evaluator never decides the region (the transport does)."""
    a, I, hgt, mu = 0.4, 1e4, 0.15, 1000.0

    def side(s):
        def f(q):
            q = np.atleast_2d(q)
            br, bz = ref.loop_over_halfspace(a, I, hgt, mu, np.abs(q[:, 0]), q[:, 1], s)
            return np.column_stack([np.sign(q[:, 0]) * br, bz, np.zeros_like(bz)])
        return f
    fa, fi = side("air"), side("iron")

    def fill(f):
        return lambda R, Z: tuple(f(np.column_stack([R.ravel(), Z.ravel()]))[:, c].reshape(R.shape) for c in (0, 1))
    # the grids' z lines do not fall on the plane: the cells around z = 0 straddle it
    pa = _kind1_patch("air", (0.0, 0.3, 61), (-0.0217, 0.0983, 25), fill(fa))
    pi = _kind1_patch("iron", (0.0, 0.3, 61), (-0.1017, 0.0183, 25), fill(fi))
    fm = fmod.write_map(tmp_path / "half", [pa, pi])
    rng = _rng(5)
    r = rng.uniform(0.0, 0.3, 4000)
    for name, f, zr in (("air", fa, (0.0, 0.0983)), ("iron", fi, (-0.1017, 0.0))):
        z = rng.uniform(*zr, 4000)
        z[:500] = np.sign(zr[0] + zr[1]) * rng.uniform(0.0, 0.005, 500)   # within one cell of the plane
        got = np.asarray(fm.patch(name).evaluate_native(np.column_stack([r, z])))
        want = f(np.column_stack([r, z]))
        p = pa if name == "air" else pi
        bound = ref.second_derivative_bound(f, np.column_stack([r, z]), (p["origin"], p["spacing"]))
        allowed = 1.25 * np.linalg.norm(bound, axis=1) + 1e-14 * np.max(np.abs(want))
        assert np.all(np.linalg.norm(got - want, axis=1) <= allowed), name
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        fm.evaluate(np.array([[0.1, 0.0, 0.01]]))           # two patches: the caller names the patch
    assert _code(e) == "fieldmap.invalid_option"


# ============================================================================================ outside the map
def test_outside_the_map_is_refused_by_default(tmp_path):
    """Beyond the clamping margin (one spacing of the grid, test below) a point is outside the map: refused."""
    p = _kind1_patch("u", (0.0, 1.0, 5), (-1.0, 1.0, 9), lambda R, Z: (0 * R, 0 * R + 0.5))
    fm = fmod.write_map(tmp_path / "refuse", [p])
    on_box = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.3], [0.6, 0.8, -1.0]])
    np.testing.assert_allclose(np.asarray(fm.evaluate(on_box))[:, 2], 0.5, rtol=1e-15)
    for bad in ([1.3, 0.0, 0.0], [0.0, 0.0, 1.3], [1.0, 1.0, 0.0]):
        with pytest.raises(axmp.AxionMultiPhysicsError) as e:
            fm.evaluate(np.array([bad]))
        assert _code(e) == "fieldmap.outside_map"
        assert e.value.hint


def test_outside_zero_returns_zero_and_counts(tmp_path):
    p = _kind3_patch("u", ((-1, 1), (-1, 1), (-1, 1)), (5, 5, 5), _uniform((0.0, 0.0, 2.0)))
    fm = fmod.write_map(tmp_path / "zero", [p], outside="zero")
    assert fm.manifest["outside"] == "zero"
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, -1.6], [1.0, 1.0, 1.0]])
    got = np.asarray(fm.evaluate(pts))
    np.testing.assert_array_equal(got[1], 0.0)
    np.testing.assert_array_equal(got[2], 0.0)
    np.testing.assert_allclose(got[[0, 3], 2], 2.0, rtol=1e-15)
    assert fm.outside_calls == 2
    fm.evaluate(pts[1:2])
    assert fm.outside_calls == 3


def test_points_within_one_cell_of_a_box_are_clamped(tmp_path):
    """A stepper's trial points fall just outside a volume, hence outside its patch's box when the region's faces lie
    on the box (every patch box is the region's node box). Within one grid spacing of the box along every axis (the
    patch's ``margin``) a point is clamped onto the box: its value is that of the nearest point of the box, exactly.
    Beyond the margin the outside rule applies."""
    p = _kind1_patch("u", (0.0, 1.0, 5), (-1.0, 1.0, 9), lambda R, Z: (0.1 * R * Z, 0.5 + 0.2 * R - 0.1 * Z))
    p["B"][..., 2] = 0.03
    fm = fmod.write_map(tmp_path / "clamp", [p])
    pat = fm.patch("u")
    assert pat.record["margin"] == pytest.approx([0.25, 0.25], rel=1e-15)
    pairs = [((1.0 + 1e-6, 0.3), (1.0, 0.3)), ((1.0 + 0.24, -0.7), (1.0, -0.7)), ((0.4, 1.0 + 0.2), (0.4, 1.0)),
             ((0.9, -1.0 - 0.25), (0.9, -1.0)), ((1.1, 1.1), (1.0, 1.0))]
    for out, on in pairs:
        assert np.array_equal(np.asarray(pat.evaluate_native([out])), np.asarray(pat.evaluate_native([on]))), out
    phi = 0.7
    a = np.asarray(fm.evaluate([[1.2 * math.cos(phi), 1.2 * math.sin(phi), 0.3]]))
    b = np.asarray(fm.evaluate([[math.cos(phi), math.sin(phi), 0.3]]))
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-16)
    for bad in ((1.0 + 0.26, 0.0), (0.5, -1.0 - 0.3)):
        with pytest.raises(axmp.AxionMultiPhysicsError) as e:
            pat.evaluate_native([bad])
        assert _code(e) == "fieldmap.outside_map"
    p3 = _kind3_patch("k", ((-1, 1), (-1, 1), (0, 2)), (5, 5, 9), lambda X, Y, Z: (X + Z, Y * Z, 0.5 + 0 * X))
    fm3 = fmod.write_map(tmp_path / "clamp3", [p3])
    assert fm3.patch("k").record["margin"] == pytest.approx([0.5, 0.5, 0.25], rel=1e-15)
    a = np.asarray(fm3.evaluate([[1.4, -1.2, 2.2]]))
    b = np.asarray(fm3.evaluate([[1.0, -1.0, 2.0]]))
    assert np.array_equal(a, b)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        fm3.evaluate([[1.6, 0.0, 1.0]])
    assert _code(e) == "fieldmap.outside_map"


def test_kind1_cartesian_components_with_an_azimuthal_field(tmp_path):
    """A kind-1 map with B_phi: B = B_r (cos phi, sin phi, 0) + B_phi (-sin phi, cos phi, 0) + B_z z-hat. A bilinear
    (B_r, B_z, B_phi) is reproduced exactly, so the Cartesian components agree to round-off; and the field of a current
    +I along +z (B_phi = mu0 I / (2 pi r), counterclockwise seen from +z: B_y > 0 on the +x axis) keeps its sense."""
    def fill(R, Z):
        return 0.2 * R * Z, 0.3 - 0.1 * Z
    p = _kind1_patch("phi", (0.0, 1.0, 11), (-1.0, 1.0, 21), fill)
    R, Z = np.meshgrid(np.linspace(0.0, 1.0, 11), np.linspace(-1.0, 1.0, 21), indexing="ij")
    p["B"][..., 2] = 0.05 * R + 0.02 * R * Z
    fm = fmod.write_map(tmp_path / "phi", [p])
    rng = _rng(11)
    r = rng.uniform(0.0, 1.0, 3000)
    z = rng.uniform(-1.0, 1.0, 3000)
    phi = rng.uniform(-math.pi, math.pi, 3000)
    br, bz, bp = 0.2 * r * z, 0.3 - 0.1 * z, 0.05 * r + 0.02 * r * z
    want = np.column_stack([br * np.cos(phi) - bp * np.sin(phi), br * np.sin(phi) + bp * np.cos(phi), bz])
    got = np.asarray(fm.evaluate(np.column_stack([r * np.cos(phi), r * np.sin(phi), z])))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)
    current = 1e4
    q = _kind1_patch("wire", (0.1, 1.1, 101), (-0.5, 0.5, 11), lambda R, Z: (0 * R, 0 * R))
    Rq = 0.1 + 0.01 * np.arange(101)
    q["B"][..., 2] = (ref.MU0 * current / (2 * math.pi * Rq))[:, None]
    fw = fmod.write_map(tmp_path / "wire", [q])
    b = np.asarray(fw.evaluate([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [-0.3, -0.4, 0.1]]))
    exact = ref.MU0 * current / (2 * math.pi * 0.5)
    np.testing.assert_allclose(b[0], [0.0, exact, 0.0], rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(b[1], [-exact, 0.0, 0.0], rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(b[2], [0.8 * exact, -0.6 * exact, 0.0], rtol=1e-3, atol=1e-12)


@pytest.mark.parametrize("which", ["scale", "shear", "reflection", "projective"])
def test_write_map_refuses_a_transform_that_is_not_a_placement(tmp_path, which):
    """A placement is a rotation and a translation: B is rotated by R and a point found by R^T (p - t). A scale or a
    shear has no such inverse, and a reflection would need the axial vector's extra sign: all refused."""
    T = np.eye(4)
    if which == "scale":
        T[:3, :3] *= 2.0
    elif which == "shear":
        T[0, 1] = 0.3
    elif which == "reflection":
        T[2, 2] = -1.0
    else:
        T[3, 0] = 0.1
    p = _kind3_patch("k", ((-1, 1), (-1, 1), (-1, 1)), (3, 3, 3), _uniform((0.0, 0.0, 1.0)), T)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        fmod.write_map(tmp_path / which, [p])
    assert _code(e) == "fieldmap.invalid_option"
    assert not (tmp_path / which).exists()
    th = 0.4
    ok = np.eye(4)
    ok[:2, :2] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    ok[:3, 3] = [0.1, 0.2, 0.3]
    fmod.write_map(tmp_path / f"{which}_rotation", [dict(p, transform=ok.tolist())])


def test_the_warning_class():
    assert issubclass(fmod.FieldMapWarning, UserWarning)


# ============================================================================================ a helix in a uniform map
P_T = 1.0      # GeV/c
R_HELIX = P_T / (ref.C_GEV * 1.0)            # 3.33564 m in 1 T
PITCH = 2 * math.pi * R_HELIX                 # p_z = p_T: 20.958 m


def test_helix_numbers():
    assert R_HELIX == pytest.approx(3.33564, rel=1e-5)
    assert PITCH == pytest.approx(20.958, rel=1e-4)


@pytest.mark.parametrize("case", ["kind3_z", "kind3_x", "kind1_z"])
def test_uniform_map_helix(tmp_path, case):
    if case == "kind3_x":
        b = (1.0, 0.0, 0.0)
        p = _kind3_patch("u", ((-5.0, 10 * PITCH + 5), (-8.0, 8.0), (-8.0, 8.0)), (9, 9, 9), _uniform(b))
    elif case == "kind3_z":
        b = (0.0, 0.0, 1.0)
        p = _kind3_patch("u", ((-8.0, 8.0), (-8.0, 8.0), (-5.0, 10 * PITCH + 5)), (9, 9, 9), _uniform(b))
    else:
        b = (0.0, 0.0, 1.0)
        p = _kind1_patch("u", (0.0, 8.0, 9), (-5.0, 10 * PITCH + 5, 9), lambda R, Z: (0 * R, 0 * R + 1.0))
    fm = fmod.write_map(tmp_path / case, [p])
    # the map holds the field exactly, in T, along the right axis
    if case == "kind3_x":
        pts = _rng(6).uniform([0, -6, -6], [10 * PITCH, 6, 6], (2000, 3))
    else:
        pts = _rng(6).uniform([-5, -5, 0], [5, 5, 10 * PITCH], (2000, 3))
    got = np.asarray(fm.evaluate(pts))
    assert np.max(np.abs(got - np.asarray(b))) <= 1e-14

    def field(x):
        return np.asarray(fm.evaluate(x[None, :]))[0]
    n_per_turn = 500
    ds = 2 * math.pi * R_HELIX * math.sqrt(2.0) / n_per_turn
    u0 = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)                    # p_T along x, p_z along z, p_T = p_z
    if case == "kind3_x":
        u0 = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)                # p_parallel along x, p_T along z
    xs, us = ref.rk4_track(field, [0.0, 0.0, 0.0], u0, P_T * math.sqrt(2.0), +1.0, ds, 10 * n_per_turn)
    axis = np.asarray(b)
    end = xs[-1]
    np.testing.assert_allclose(end, 10 * PITCH * axis, atol=1e-6 * 10 * PITCH)
    # mu+ in B along +axis turns clockwise seen from the tip of B: the centre of the circle
    u_perp = u0 - (u0 @ axis) * axis
    u_perp /= np.linalg.norm(u_perp)
    centre = R_HELIX * np.cross(u_perp, axis)
    rel = xs - centre
    rel_perp = rel - np.outer(rel @ axis, axis)
    np.testing.assert_allclose(np.linalg.norm(rel_perp, axis=1), R_HELIX, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(us, axis=1), 1.0, atol=1e-9)


# ============================================================================================ protons through a current loop's map
LOOP_A = 0.5
LOOP_I = 0.1 * 2 * LOOP_A / ref.MU0              # 0.1 T at the loop's centre
P_PROTON = 0.1                                   # GeV/c


def _psi_exact(x):
    return ref.loop_psi(LOOP_A, LOOP_I, np.hypot(x[:, 0], x[:, 1]), x[:, 2])


def _p_phi(xs, us):
    r = np.hypot(xs[:, 0], xs[:, 1])
    rs = np.where(r > 0, r, 1.0)
    e_phi = np.column_stack([-xs[:, 1] / rs, xs[:, 0] / rs, np.zeros(len(r))])
    return r * P_PROTON * np.sum(us * e_phi, axis=1) + ref.C_GEV * 1.0 * _psi_exact(xs)


def test_loop_canonical_momentum_through_a_kind1_map(tmp_path):
    native = _loop_native(LOOP_A, LOOP_I)
    p = _kind1_patch("loop", (0.0, 0.4, 41), (-1.2, 1.2, 241), lambda R, Z: tuple(
        native(np.column_stack([R.ravel(), Z.ravel()]))[:, c].reshape(R.shape) for c in (0, 1)))
    fm = fmod.write_map(tmp_path / "loop", [p])

    def exact(x):
        return np.column_stack(ref.loop_B_cartesian(LOOP_A, LOOP_I, x[None, 0], x[None, 1], x[None, 2]))[0]

    def mapped(x):
        return np.asarray(fm.evaluate(x[None, :]))[0]

    def stop(x):
        return x[2] > 1.1 or math.hypot(x[0], x[1]) > 0.38
    starts = [((0.10, 0.0, -1.1), (0.0, 0.02, 1.0)), ((0.05, 0.08, -1.1), (0.015, -0.01, 1.0)),
              ((-0.12, 0.03, -1.1), (-0.01, 0.02, 1.0))]
    ds = 0.002
    for x0, u0 in starts:
        xe, ue = ref.rk4_track(exact, x0, u0, P_PROTON, 1.0, ds, 3000, stop)
        pe = _p_phi(xe, ue)
        scale = np.max(np.hypot(xe[:, 0], xe[:, 1])) * P_PROTON
        assert np.max(np.abs(pe - pe[0])) <= 1e-9 * scale, "the integrator itself"
        xm, um = ref.rk4_track(mapped, x0, u0, P_PROTON, 1.0, ds, 3000, stop)
        assert xm[-1][2] > 1.0, "the track crosses the loop's plane"
        pm = _p_phi(xm, um)
        Bm = np.asarray(fm.evaluate(xm))
        Bx = np.column_stack(ref.loop_B_cartesian(LOOP_A, LOOP_I, xm[:, 0], xm[:, 1], xm[:, 2]))
        dB = np.linalg.norm(Bm - Bx, axis=1)
        r = np.hypot(xm[:, 0], xm[:, 1])
        # the map's error at every point of the track is at the interpolation level (the interpolation bound) ...
        bound = ref.second_derivative_bound(native, np.column_stack([r, xm[:, 2]]), (p["origin"], p["spacing"]))
        allowed = 1.25 * np.linalg.norm(bound, axis=1) + 1e-14
        assert np.all(dB <= allowed), f"map error {np.max(dB / allowed):.2f} times the interpolation bound"
        # ... and the canonical momentum moves by no more than the integral of the map's error
        integrand = r * dB
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * ds)])
        drift = np.abs(pm - pm[0])
        assert np.all(drift <= 1.01 * ref.C_GEV * cum + 1e-9 * scale + 1e-15)


# ============================================================================================ documentation
def test_the_guide_documents_the_format_and_every_code():
    guide = ROOT / "docs" / "guides" / "field_maps.md"
    assert guide.is_file(), "docs/guides/field_maps.md is the user page of field maps"
    text = guide.read_text()
    assert "field_maps.md" in (ROOT / "mkdocs.yml").read_text()
    for code in ("fieldmap.curve_in_region", "fieldmap.not_converging", "fieldmap.cap_reached",
                 "fieldmap.patch_too_large", "fieldmap.total_cap_exceeded", "fieldmap.outside_map",
                 "fieldmap.outside_field_not_small", "fieldmap.track_volume_empty", "fieldmap.invalid_option",
                 "fieldmap.chord_incomplete", "fieldmap.tangled_mesh", "FieldMapWarning", "margin"):
        assert code in text, code
    assert "axionmultiphysics.fieldmap/1" in text
    assert re.search(r"\bbeta\b", text, re.I)
    assert "export_field_map" in (ROOT / "llms.txt").read_text()
    # the notes since 0.1.0 ("Unreleased" and the release sections above 0.1.0) without their Highlights, so that the
    # entry itself must name the export
    notes = (ROOT / "CHANGELOG.md").read_text().split("## Unreleased", 1)[1].split("\n## 0.1.0", 1)[0]
    notes = re.sub(r"### Highlights\n.*?(?=\n### )", "", notes, flags=re.S)
    assert "export_field_map" in notes, "CHANGELOG.md (notes since 0.1.0, outside the Highlights) has no field-map entry"
