"""Field maps of a solved 3D design (``Results3D.export_field_map``, kind 3; axionmultiphysics.fieldmap, beta).

The particle-transport design note, section 5. The measurements that fixed the map constants were all axisymmetric;
before the kind-3 constants are relied on, the same checks run on one 3D solve: the mu_r = 1000 sphere (radius 0.5 m,
uniform 0.1 T) revolved by ``to_3d``, with its exact field. Kind-3 chords are 3D lines with isotropic directions
(cos(theta) = 2 u_4 - 1, phi = 2 pi u_5 of the family's sequence point); their integrals of the trilinear map are
exact per cell; the statistic and the tolerances are those of tests/test_fieldmap_build.py.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import axionmultiphysics as axmp
from axionmultiphysics import fieldmap as fmod
from axionmultiphysics.convert import to_3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fieldmap_ref as ref  # noqa: E402
import test_fieldmap_build as tb  # noqa: E402

pytestmark = [pytest.mark.skipif(not axmp.build_info()["engine_3d"],
                                 reason="needs the 3D engine (an MPI build; the wheels have none)"),
              pytest.mark.skipif(not os.environ.get("AXMP_SLOW"), reason="a 3D solve (minutes); set AXMP_SLOW=1")]


def _exact(name):
    K, f = ref.sphere_K_Bin(1000.0)

    def outside(q):
        q = np.atleast_2d(q)
        r = np.hypot(q[:, 0], q[:, 1])
        br, bz = ref.dipole_exterior(tb.B0, K, tb.A_SPHERE, r, q[:, 2])
        phi = np.arctan2(q[:, 1], q[:, 0])
        return np.column_stack([br * np.cos(phi), br * np.sin(phi), bz])

    def inside(q):
        q = np.atleast_2d(q)
        return np.column_stack([np.zeros(len(q)), np.zeros(len(q)), np.full(len(q), f * tb.B0)])
    return outside if name == "air" else inside


@pytest.fixture(scope="module")
def sphere3d(tmp_path_factory):
    wd = tmp_path_factory.mktemp("sphere3d")
    d3 = to_3d(axmp.Design(tb.sphere_doc()), mesh={"order": 2, "size": 0.15, "surface_size": 0.06})
    res = d3.solve(workdir=str(wd / "solve"), verbose=0)
    path = wd / "map"
    res.export_field_map(path, tol=1e-3)
    return res, path, json.loads((path / "manifest.json").read_text())


def test_kind3_patches_and_their_grids(sphere3d):
    res, path, m = sphere3d
    assert m["source"]["symmetry"] == "3d"
    P = tb.precs(m)
    assert set(P) == {"air", "ball"}
    for p in P.values():
        assert p["kind"] == 3
        assert p["frame"]["type"] == "cartesian"
        assert np.array_equal(np.asarray(p["frame"]["transform"]), np.eye(4))
        assert "axis" not in p["families"] and {"bulk", "layer"} <= set(p["families"])
        tb.check_grid_rule(p)
        B = tb.array(path, p)
        assert B.shape == tuple(p["shape"]) + (3,) and B.dtype.str == "<f8"


def test_kind3_chords_are_isotropic_3d_lines_from_the_manifest_seed(sphere3d):
    res, path, m = sphere3d
    u = ref.halton(ref.family_seed(tb.CHORD_SEED, 0), 16384)
    c = 2 * u[:, 4] - 1
    phi = 2 * np.pi * u[:, 5]
    s = np.sqrt(np.maximum(0.0, 1 - c * c))
    want = np.column_stack([s * np.cos(phi), s * np.sin(phi), c])
    for p in m["patches"]:
        C = tb.chords(path, p, "bulk")
        assert C["start"].shape == (16384, 3)
        np.testing.assert_allclose(C["direction"], want, atol=1e-14)
        rho = np.linalg.norm(C["start"], axis=1)
        if p["name"] == "ball":
            assert np.all(rho <= tb.A_SPHERE + 0.01)
        else:
            assert np.all(rho >= tb.A_SPHERE - 0.01)


def test_kind3_chords_cover_the_exact_intersection(sphere3d):
    """Every 3D chord's recorded segments cover its exact intersection with the region (ray-sphere roots for the ball
    and the domain sphere of radius 1.5 m; curved tetrahedra on both)."""
    res, path, m = sphere3d
    for p in m["patches"]:
        def inside(fam, name=p["name"]):
            def f(q):
                rho = np.linalg.norm(q, axis=1)
                return rho <= tb.A_SPHERE if name == "ball" else (rho >= tb.A_SPHERE) & (rho <= tb.DOMAIN)
            return f
        tb.check_chords_complete(path, p, inside, (tb.A_SPHERE, tb.DOMAIN))


def test_kind3_statistic_and_f3b(sphere3d):
    res, path, m = sphere3d
    for p in m["patches"]:
        tb.check_acceptance(p, 1e-3)
        for fam in p["families"]:
            tb.check_statistic_recomputed(path, p, fam, 1e-3)
        tb.check_f3b_accuracy(path, p, _exact(p["name"]))


def test_kind3_recovered_field_and_f3c(sphere3d):
    res, path, m = sphere3d
    for p in m["patches"]:
        for fam in p["families"]:
            tb.check_star_integrals(res, path, p, fam)
    again = fmod.chord_error(res, path)
    other = fmod.chord_error(res, path, seed=tb.CHORD_SEED + 1)
    for p in m["patches"]:
        for fam, rec in p["families"].items():
            assert again[p["name"]][fam]["E"] == pytest.approx(rec["E"], rel=1e-12)
            if rec["acceptance"]:
                assert 1 / 2.5 <= other[p["name"]][fam]["E"] / rec["E"] <= 2.5


def test_kind3_cartesian_evaluation_matches_the_written_array(sphere3d):
    res, path, m = sphere3d
    fm = fmod.load(path)
    p = tb.precs(m)["air"]
    pts = np.random.default_rng(3).uniform([0.6, -0.4, -0.4], [0.9, 0.4, 0.4], (2000, 3))
    want = ref.interp_native(p["origin"], p["spacing"], tb.array(path, p), pts)
    np.testing.assert_allclose(np.asarray(fm.evaluate(pts, patch="air")), want, rtol=0, atol=1e-14 * p["B_patch"])
    assert math.isfinite(float(np.max(np.abs(want))))
