"""Tangled curved elements in 2D meshes of geometry order 2: axmp-mesher and the library's mesh read.

A curved region much thinner than the mesh size (a 2 mm shell on a sphere of radius 0.5 m at a 0.05 m mesh size) gives
6-node triangles whose Jacobian determinant changes sign: half of the shell's triangles fold over. The behaviour fixed
here:

* The mesher checks every element of a 2D mesh of geometry order 2 (the axisymmetric "open" mode, "plane" and
  "bounded"). Only when an element is tangled does it run gmsh's high-order optimisation, then it checks again. A mesh
  with no tangled element is written unchanged (the .msh bytes and the info file of the mesher without the check; no
  new key).
* A repaired mesh is written with "ok": true and the info record "high_order": {"checked", "tangled_before",
  "optimised", "tangled_after"}; it keeps the geometry (the nodes of curved boundary edges stay on the curves) and it is
  the same in every run, whatever the output directory or the environment.
* A mesh still tangled after the optimisation, or tangled with the optimisation switched off (AXMP_MESHER_HO=none), is
  refused: exit status 1, "ok": false, an error naming the count, the region of the first tangled element and the
  remedy (refine the mesh in or around that region, or geometry order 1), the same "high_order" record, no mesh file.
* The library refuses a tangled curved mesh when it reads one, whoever wrote it (a mesher that reports success, or an
  imported file): code mesh.tangled, with the count, the region and the remedy, before anything is solved. The EM
  solve and `axmp mesh` report a mesher's refusal as they report any failed mesher run.

Tangled: det J of a 6-node triangle is not of one strict sign over the whole element (it changes sign or vanishes
somewhere in it), or has the sign opposite to the majority of the elements meshed in one orientation with it. det J
of a 6-node triangle is a quadratic polynomial of the reference coordinates, so its minimum over the triangle is
computed exactly (the vertices, the extremum of each edge, the interior stationary point), not sampled: a fold
between the points of the field-map builder's lattice (i/12, j/12) is tangled too. The orientation does not matter:
the majority is that of a gmsh surface (the unit gmsh meshes in one winding and the mesher turns counter-clockwise),
so a region meshed clockwise, or a region whose parts gmsh meshed in opposite windings (a union with a polygon listed
clockwise), is not tangled; and an element written with its nodes reversed in a file is read in the mesh's
orientation (MFEM orients the elements of a mesh it reads) and changes nothing.

INDEPENDENT REFERENCES. det J is computed here from the node coordinates of the MSH 2.2 file (the P2 shape functions in
gmsh's node order; its exact range from the monomial coefficients, checked against a 1/400 grid on random elements),
not by the code under test; the shells and rings are bounded by circles of known radius. Measured
with the mesher without the check, gmsh 4.15.2 (Lawrencium): the 2 mm shell has 32 of its 64 triangles tangled
(det J ratio down to -0.20), the 3 mm shell none (0.198); the plane and bounded 2 mm rings 63 of 126; the meshes
of the G0 axisymmetric designs none (the smallest ratio, 0.0024, in examples/pickup_capacitance.json, so a threshold
on the ratio instead of the sign would change that mesh). gmsh's optimisation repaired every 2D case tried (shells
from 1e-6 m to 2 mm, radii 0.05 and 0.5 m, a crescent, the rings), so the refusal is exercised with the optimisation
switched off.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from axionmultiphysics import _paths

ROOT = Path(__file__).resolve().parents[1]
BOUNDED_FIXTURES = ROOT / "cpp" / "tests" / "data" / "model" / "u2" / "mesher"
PINNED_GMSH = "4.15.2"
A = 0.5  # outer radius of the shells and rings (m)


def _mesher():
    m = _paths.mesher_path()
    if not m:
        pytest.skip("axmp-mesher not found")
    return m


def _gmsh_version(info):
    return str(info.get("gmsh_version", "")) if isinstance(info, dict) else ""


# ------------------------------------------------------------------------------------------------- independent det J
def read_msh22(path):
    """Physical names {(dim, tag): name}, nodes {tag: (x, y)}, and the 2D elements: tags, physical tags, node tags."""
    lines = Path(path).read_text().split("\n")
    names = {}
    if "$PhysicalNames" in lines:
        i = lines.index("$PhysicalNames")
        for ln in lines[i + 2:i + 2 + int(lines[i + 1])]:
            a = ln.split(None, 2)
            names[(int(a[0]), int(a[1]))] = a[2].strip().strip('"')
    i = lines.index("$Nodes")
    nodes = {}
    for ln in lines[i + 2:i + 2 + int(lines[i + 1])]:
        a = ln.split()
        nodes[int(a[0])] = (float(a[1]), float(a[2]))
    i = lines.index("$Elements")
    tags, phys, conn = [], [], []
    for ln in lines[i + 2:i + 2 + int(lines[i + 1])]:
        a = ln.split()
        if a[1] in ("2", "9"):  # 3- and 6-node triangles
            nt = int(a[2])
            conn.append([int(x) for x in a[3 + nt:]])
            tags.append(int(a[0]))
            phys.append(int(a[3]))
    return names, nodes, np.array(tags), np.array(phys), conn


def _dN(u, v):
    w = 1 - u - v
    return np.array([[-(4 * w - 1), -(4 * w - 1)], [4 * u - 1, 0], [0, 4 * v - 1], [4 * (w - u), -4 * u],
                     [4 * v, 4 * u], [-4 * v, 4 * (w - v)]])


LATTICE = [(i / 12, j / 12) for i in range(13) for j in range(13 - i)]
VERTICES = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
NODES6 = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
_MONOMIALS_INV = np.linalg.inv(np.array([[1, u, v, u * u, u * v, v * v] for u, v in NODES6]))


def det_at(P, u, v):
    """det J of the 6-node triangles P (n, 6, 2) at the reference point (u, v)."""
    J = np.einsum("nk,enj->ejk", _dN(u, v), P)
    return J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]


def sampled_range(P, points):
    d = np.stack([det_at(P, u, v) for u, v in points], 1)
    return d.min(1), d.max(1)


def exact_range(P):
    """Exact min and max of det J over each 6-node triangle. det J is quadratic in (u, v): its values at the six nodes
    give the coefficients of 1, u, v, u^2, uv, v^2; the extremes lie at the vertices, at the stationary point of each
    edge (inside it) or at the interior stationary point (inside the triangle)."""
    q = np.stack([det_at(P, u, v) for u, v in NODES6], 1)
    a, b, c, d, e, f = (q @ _MONOMIALS_INV.T).T
    cand = [q[:, 0], q[:, 1], q[:, 2]]

    def on_edge(A, B, C):  # A t^2 + B t + C at its stationary point when 0 < t < 1
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -B / (2 * A)
            return np.where((A != 0) & (t > 0) & (t < 1), A * t * t + B * t + C, np.nan)

    cand += [on_edge(d, b, a), on_edge(f, c, a), on_edge(d - e + f, -b + c - 2 * d + e, a + b + d)]  # v=0, u=0, u+v=1
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 4 * d * f - e * e
        u0, v0 = (-2 * f * b + e * c) / m, (-2 * d * c + e * b) / m
        inside = (m != 0) & (u0 > 0) & (v0 > 0) & (u0 + v0 < 1)
        cand.append(np.where(inside, a + b * u0 + c * v0 + d * u0 * u0 + e * u0 * v0 + f * v0 * v0, np.nan))
    C = np.stack(cand, 1)
    return np.nanmin(C, 1), np.nanmax(C, 1)


def mesh_triangles(path):
    names, nodes, tags, phys, conn = read_msh22(path)
    assert conn and all(len(c) == 6 for c in conn), f"{path}: not a mesh of 6-node triangles only"
    return np.array([[nodes[k] for k in c] for c in conn]), phys, names


def det_range(path):
    """Per 6-node triangle: the exact min and max of det J; also the physical tags, the node coordinates (n, 6, 2)
    and the physical names."""
    P, phys, names = mesh_triangles(path)
    mn, mx = exact_range(P)
    return mn, mx, phys, P, names


def tangled_mask(mn, mx, phys):
    """The definition of the module docstring, the majority taken per region (`phys`; in a file the mesher wrote every
    surface is counter-clockwise, so a region's majority is its surfaces'): the elements not of its strict sign."""
    bad = np.zeros(len(mn), bool)
    for g in np.unique(phys):
        k = phys == g
        pos, neg = int((mn[k] > 0).sum()), int((mx[k] < 0).sum())
        ok = mx[k] < 0 if neg > pos else mn[k] > 0
        bad[np.where(k)[0][~ok]] = True
    return bad


def tangled_count(path):
    mn, mx, phys, _, _ = det_range(path)
    return int(tangled_mask(mn, mx, phys).sum())


def curved_count(path):
    """Elements with a mid-edge node off the straight midpoint (relative 1e-9 of the edge length)."""
    _, _, _, P, _ = det_range(path)
    mid = 0.5 * (P[:, [0, 1, 2]] + P[:, [1, 2, 0]])
    edge = np.linalg.norm(P[:, [1, 2, 0]] - P[:, [0, 1, 2]], axis=2)
    off = np.linalg.norm(P[:, 3:6] - mid, axis=2)
    return int((off > 1e-9 * edge).any(axis=1).sum())


def region_tag(names, region):
    tags = [t for (d, t), n in names.items() if d == 2 and n == region]
    assert len(tags) == 1, (region, names)
    return tags[0]


def assert_ring_geometry_kept(path, region, a_in, a_out=A):
    """Every node of the region's elements lies in the ring a_in <= r <= a_out, and the mid-node of every edge whose two
    vertices lie on one of the circles lies on that circle (the optimisation curved the mesh, it did not straighten the
    boundary)."""
    mn, mx, phys, P, names = det_range(path)
    k = phys == region_tag(names, region)
    Q = P[k]
    r = np.hypot(Q[..., 0], Q[..., 1])
    tol = 1e-9 * a_out
    assert r.min() >= a_in - tol and r.max() <= a_out + tol, (r.min(), r.max())
    on_arc = 0
    for (i, j), m in (((0, 1), 3), ((1, 2), 4), ((2, 0), 5)):
        for R in (a_in, a_out):
            both = (np.abs(r[:, i] - R) < tol) & (np.abs(r[:, j] - R) < tol)
            # an edge on the symmetry axis (axisymmetric shells) has both ends on the axis, not on one circle
            assert np.all(np.abs(r[both, m] - R) < tol), (region, R, r[both, m])
            on_arc += int(both.sum())
    assert on_arc > 0, "no edge of the region lies on its circles"


# ------------------------------------------------------------------------------------------------- mesher inputs
def _contour_shell(t):
    a1 = A - t
    return {"contour": {"segments": [
        {"ellipse": {"axes": [A, A], "center": [0.0, 0.0], "t": [-math.pi / 2, math.pi / 2]}},
        {"line": [0.0, a1]},
        {"ellipse": {"axes": [a1, a1], "center": [0.0, 0.0], "t": [math.pi / 2, -math.pi / 2]}},
        {"line": [0.0, -A]}], "start": [0.0, -A]}}


def axi_input(regions, *, order=2, surface_size=0.05, size=0.05):
    """The axisymmetric (open-domain) input the library writes for the sphere documents of test_fieldmap_build.py."""
    return {"air_attr": 1, "axis_attr": 900, "curves": [], "device_radius": 0.7071067811865476, "domain_center_z": 0.0,
            "domain_radius": 1.5, "far_size": 0.04908738521234052, "order": order, "outer_attr": 901,
            "refine_regions": [], "regions": regions, "size": size, "surface_layer": 0.2,
            "surface_size": surface_size, "verbose": 0}


def axi_shell(t, *, order=2, own_surface_size=None):
    """An iron shell A - t < r < A on the half disk (the region 'shell'), mesh size 0.05 m."""
    reg = {"attr": 2, "name": "shell", "shape": _contour_shell(t)}
    if own_surface_size:
        reg["surface_size"] = own_surface_size
    return axi_input([reg], order=order)


def axi_sphere():
    ball = {"attr": 2, "name": "ball", "shape": {"contour": {"segments": [
        {"ellipse": {"axes": [A, A], "center": [0.0, 0.0], "t": [-math.pi / 2, math.pi / 2]}}, {"line": [0.0, -A]}],
        "start": [0.0, -A]}}}
    return axi_input([ball], surface_size=0.02)


def axi_ball_clockwise():
    """The sphere with its contour written clockwise: gmsh meshes the surface clockwise, so the region's elements all
    have det J < 0 until the mesher turns them (the region majority rule keeps them valid)."""
    ball = {"attr": 2, "name": "ball", "shape": {"contour": {"segments": [
        {"line": [0.0, A]}, {"ellipse": {"axes": [A, A], "center": [0.0, 0.0], "t": [math.pi / 2, -math.pi / 2]}}],
        "start": [0.0, -A]}}}
    return axi_input([ball], surface_size=0.02)


def axi_pickup_capacitance():
    """The input the library writes for examples/pickup_capacitance.json (a G0 design): valid, with the most distorted
    curved elements of the G0 meshes (det J ratio 0.0024)."""
    return {"air_attr": 1, "axis_attr": 900, "curves": [], "device_radius": 0.4242640687119285,
            "domain_center_z": 0.0, "domain_radius": 1.0606601717798212, "far_size": 0.03471002295436223, "order": 2,
            "outer_attr": 901, "refine_regions": ["can", "ring"], "regions": [
                {"attr": 2, "name": "can", "shape": {"difference": [
                    {"rect": {"fillet": 0.0, "r": [0.0, 0.3], "z": [-0.3, 0.3]}},
                    {"rect": {"fillet": 0.0, "r": [0.0, 0.295], "z": [-0.295, 0.295]}}]}},
                {"attr": 3, "name": "ring", "shape": {"disk": {"center": [0.15, 0.0], "radius": 0.01}}},
                {"attr": 4, "name": "support", "shape": {"rect": {"fillet": 0.0, "r": [0.13, 0.16999999999999998],
                                                                  "z": [-0.02, -0.01]}}}],
            "size": 0.02, "surface_layer": 0.08, "surface_size": 0.002, "verbose": 0}


def plane_ring(t, *, order=2):
    """A ring A - t < r < A (region 'ring') inside a vacuum disk of radius 0.6 m, the plane mode, mesh size 0.05 m."""
    return {"plane": True, "regions": [
        {"name": "vacuum", "attr": 1, "shape": {"disk": {"center": [0, 0], "radius": 0.6}}},
        {"name": "ring", "attr": 2, "shape": {"difference": [{"disk": {"center": [0, 0], "radius": A}},
                                                             {"disk": {"center": [0, 0], "radius": A - t}}]}}],
            "curves": [], "size": 0.05, "surface_size": 0.05, "order": order}


def bounded_annulus():
    """The existing fixture of the bounded mode (a ring 2-5 mm, geometry order 2)."""
    return json.loads((BOUNDED_FIXTURES / "annulus.json").read_text())


def bounded_ring(t, *, order=2):
    """annulus.json with the ring A - t < r < A (region 'ring', no air), mesh size 0.05 m."""
    d = bounded_annulus()
    d["regions"][0]["shape"]["difference"][0]["disk"]["radius"] = A
    d["regions"][0]["shape"]["difference"][1]["disk"]["radius"] = A - t
    d["boundaries"]["inner"]["where"] = {"x": [-(A - t), A - t], "y": [-(A - t), A - t]}
    d.update(size=0.05, surface_size=0.05, order=order)
    return d


# One region whose parts gmsh meshes in opposite windings: a polygon listed clockwise next to a disk or to a polygon
# listed counter-clockwise (disjoint or overlapping). Every element is valid; only the windings differ.
CCW_SQUARE = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]
CW_SQUARE = [[0.1, -0.2], [0.1, -0.15], [0.15, -0.15], [0.15, -0.2]]
CW_OVERLAPPING = [[0.25, 0.25], [0.25, 0.35], [0.35, 0.35], [0.35, 0.25]]
MIXED_WINDINGS = {
    "disk_and_cw_polygon": {"union": [{"disk": {"center": [0.3, 0.3], "radius": 0.05}},
                                      {"polygon": {"points": CW_SQUARE}}]},
    "ccw_and_cw_polygons": {"union": [{"polygon": {"points": CCW_SQUARE}}, {"polygon": {"points": CW_SQUARE}}]},
    "overlapping_ccw_and_cw_polygons": {"union": [{"polygon": {"points": CCW_SQUARE}},
                                                  {"polygon": {"points": CW_OVERLAPPING}}]},
}


def axi_mixed(shape):
    return axi_input([{"attr": 2, "name": "mixed", "shape": MIXED_WINDINGS[shape]}], surface_size=0.02)


def plane_mixed(shape):
    return {"plane": True, "regions": [
        {"name": "vacuum", "attr": 1, "shape": {"disk": {"center": [0, 0], "radius": 0.6}}},
        {"name": "mixed", "attr": 2, "shape": MIXED_WINDINGS[shape]}],
            "curves": [], "size": 0.05, "surface_size": 0.02, "order": 2}


def bounded_mixed(shape):
    return {"bounded": True, "geometry": "planar", "exterior": "none",
            "regions": [{"name": "mixed", "attr": 1, "shape": MIXED_WINDINGS[shape]}], "curves": [],
            "boundaries": {"all": {"of": ["mixed"]}}, "size": 0.02, "surface_size": 0.02, "curvature_points": 24,
            "order": 2, "resolve_only": False, "verbose": 0}


def run_mesher(inp, outdir, env=None):
    """Runs the mesher on `inp` with the mesh and info written into `outdir`; returns (exit status, info or None,
    mesh path, stderr)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = dict(inp, output=str(outdir / "mesh.msh"), info=str(outdir / "mesh_info.json"))
    (outdir / "in.json").write_text(json.dumps(d, indent=1))
    e = dict(os.environ)
    for k in ("AXMP_MESHER_HO", "AXMP_MESHER_HO_ISOLATE", "AXMP_MESHER_OPTIMIZE", "AXMP_MESHER_ASLR"):
        e.pop(k, None)
    e.update(env or {})
    p = subprocess.run([_mesher(), str(outdir / "in.json")], capture_output=True, text=True, env=e, timeout=600)
    info_path = outdir / "mesh_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else None
    return p.returncode, info, outdir / "mesh.msh", p.stderr


@functools.lru_cache(maxsize=None)
def mesher_gmsh_version():
    """The gmsh of this mesher, from the info file of a valid mesh (a refusal's info file need not name it)."""
    with tempfile.TemporaryDirectory() as d:
        _, info, _, err = run_mesher(dict(bounded_annulus(), order=1), d)
        assert info and info.get("ok"), err[-2000:]
        return _gmsh_version(info)


# The thin cases: (mode, input, tangled region, inner radius, tangled before the repair on the pinned gmsh)
TANGLED = {
    "axi_shell_2mm": (lambda: axi_shell(0.002), "shell", A - 0.002, 32),
    "plane_ring_2mm": (lambda: plane_ring(0.002), "ring", A - 0.002, 63),
    "bounded_ring_2mm": (lambda: bounded_ring(0.002), "ring", A - 0.002, 63),
}


# ------------------------------------------------------------------------------------------------- the mesher
@pytest.mark.parametrize("case", sorted(TANGLED))
def test_a_tangled_mesh_is_repaired_and_keeps_its_geometry(case, tmp_path):
    """The thin shell and rings: the mesh written with ok: true has no tangled element, its boundary nodes are still on
    the circles, and the info file says what happened. On the pinned gmsh the optimisation repairs all three; on another
    gmsh a refusal is also within the contract (then nothing is written)."""
    make, region, a_in, before = TANGLED[case]
    rc, info, msh, err = run_mesher(make(), tmp_path)
    assert info is not None, err[-2000:]
    if not info.get("ok"):
        assert mesher_gmsh_version() != PINNED_GMSH, f"refused on the pinned gmsh: {info.get('error')}"
        assert rc != 0 and not msh.exists()
        return
    assert rc == 0, err[-2000:]
    assert tangled_count(msh) == 0, f"{case}: a mesh written with ok: true is tangled"
    assert_ring_geometry_kept(msh, region, a_in)
    ho = info.get("high_order")
    assert isinstance(ho, dict), f"{case}: the info file does not record the repair: {sorted(info)}"
    assert set(ho) == {"checked", "tangled_before", "optimised", "tangled_after"}, ho
    assert ho["optimised"] is True and ho["tangled_after"] == 0, ho
    assert curved_count(msh) <= ho["checked"] <= info["elements_2d"], (ho, info["elements_2d"])
    assert 1 <= ho["tangled_before"] <= ho["checked"], ho
    if mesher_gmsh_version() == PINNED_GMSH:
        assert ho["tangled_before"] == before, ho


@pytest.mark.parametrize("case", sorted(TANGLED))
def test_a_mesh_left_tangled_is_refused_with_the_remedy(case, tmp_path):
    """With gmsh's optimisation switched off (AXMP_MESHER_HO=none) the thin cases stay tangled: refused, with an error
    that names the count, the region and the remedy, the record of the check, and no mesh file."""
    make, region, _, before = TANGLED[case]
    rc, info, msh, err = run_mesher(make(), tmp_path, env={"AXMP_MESHER_HO": "none"})
    assert rc == 1, (rc, err[-2000:])
    assert info is not None and info.get("ok") is False, info
    assert not msh.exists(), "a refused mesh was written"
    ho = info.get("high_order")
    assert isinstance(ho, dict) and ho.get("optimised") is False, info
    k = ho["tangled_after"]
    assert k == ho["tangled_before"] >= 1, ho
    if mesher_gmsh_version() == PINNED_GMSH:
        assert k == before, ho
    msg = info["error"]
    assert re.search(rf"(?<![\d.]){k}(?![\d.])", msg), f"the count {k} is not in: {msg}"
    assert "tangled" in msg and f"'{region}'" in msg, msg
    assert re.search(r"[Rr]efine the mesh in or around", msg) and "order 1" in msg, msg
    assert msg in err, "the error is not on stderr"


def test_the_remedy_named_in_the_refusal_works(tmp_path):
    """Both remedies of the message: the 2 mm shell meshed at geometry order 1, or with a finer surface size on the
    shell, is not tangled even with the optimisation switched off; nothing is recorded."""
    for name, inp in (("order1", axi_shell(0.002, order=1)), ("refined", axi_shell(0.002, own_surface_size=0.005))):
        rc, info, msh, err = run_mesher(inp, tmp_path / name, env={"AXMP_MESHER_HO": "none"})
        assert rc == 0 and info["ok"] is True, (name, info, err[-2000:])
        assert "high_order" not in info, (name, info["high_order"])
        if name == "refined":
            assert tangled_count(msh) == 0


def _masked_info_text(path):
    """The info file's text with the value of "seconds" blanked (G0 masks it too)."""
    return re.sub(r'("seconds": )[-+0-9.eE]+', r"\g<1>0", Path(path).read_text())


def _sha(b):
    return hashlib.sha256(b).hexdigest()


# Valid meshes and their references: sha256 of the .msh file and of the masked info text written by the mesher without
# the check, gmsh 4.15.2 on Lawrencium (G0 platforms lawrencium-x86_64-*; the same on the Xeon and the EPYC nodes).
UNCHANGED = {
    "axi_sphere_order2": (axi_sphere,
        "8742e815bf7cf7dfc940da2a7ea964679385dbc3e1ce42ea89b9a7c6541c704e",
        "8cba965f4214f50eb4a5c6ece23d3c7f5bfd2008dc86b44fa3f9c82a65b62e7a"),
    "axi_ball_clockwise_order2": (axi_ball_clockwise,
        "8f42b4ab0c77bb476be85073553d8fae19c9af684392bd38ec8cba79f35337fd",
        "dd2164ac2e461ccbf844139a9c513c7133134f135e377b939e226a971fff2c0f"),
    "axi_shell_3mm_order2": (lambda: axi_shell(0.003),
        "5e0efbaa2cca7be14a503205af0767d860ec142b353a1b1b85e534d18f0595c2",
        "6421fe7dfdaabfdc834ba2ad1374335844a805dc3f093abd4080468220d7377d"),
    "axi_shell_2mm_order1": (lambda: axi_shell(0.002, order=1),
        "c2be3f8dcbb0320de622e35f3f06ca20230e2bbb8363fbafa075d9d410973d8e",
        "cd4eaf6807cf1d7c062b83d8bf87a2f7dbeb08951219d176f371db6793933856"),
    "axi_pickup_capacitance_order2": (axi_pickup_capacitance,
        "3d5231f4e015bb017713d4ea63af18f0472f9d06cb5053b951b104824372b84a",
        "b3edd4701d717e78293aeaf9a468de2868da579fde13f86a446aad7a62b052ee"),
    "plane_ring_20mm_order2": (lambda: plane_ring(0.02),
        "8949597a1313599b52a223470980740de21626ef55a6b243c7f7ee8993da2835",
        "be70ed7c96a531bad21a33fba20ef3ae3a2745c05610aba2edf5cbb3926cdd04"),
    "plane_ring_2mm_order1": (lambda: plane_ring(0.002, order=1),
        "f8aaf29eaa546aadfccd014a44396352c89955d3c5f07db45609066b127c8465",
        "d8e75b214df09a396286fdb09c519d72eac486ddf31451d24e5d33fcfeca0bee"),
    "bounded_annulus_order2": (bounded_annulus,
        "74b828994824a093e1710ad10d99591ec2d8ef1f1542dba1ae4c40b48a3a963d",
        "012e2ff73a59ddd4443c591bb7db30abc330cfcc9563174da0785e645956eeab"),
    "axi_mixed_disk_and_cw_polygon_order2": (lambda: axi_mixed("disk_and_cw_polygon"),
        "bd49eac515e18ba9ef358e93a5198af70245694461bc75634251fb04b2d45367",
        "803566c0043a65d6132f71bd2cb70d2842a99ecdfb6e26e16a52e2c8b7c973ec"),
    "axi_mixed_ccw_and_cw_polygons_order2": (lambda: axi_mixed("ccw_and_cw_polygons"),
        "b0d034f7743651dc054178b9978e939e49c67616092c191b9a06ba541f52d7cf",
        "1e11ad894d10ae510467d6e66f9a37ff092d9f04202d1f9db29b96feaa5a6ed3"),
    "axi_mixed_overlapping_ccw_and_cw_polygons_order2": (lambda: axi_mixed("overlapping_ccw_and_cw_polygons"),
        "eb04fa0c806c0bbe7dd22aa4467e5010884c664fbc21a7a87ecc852bf72e5990",
        "d11cd0c8f56126d9894a8c6d3870f321585e3ac87a47fe72831de483c21b6bd3"),
    "plane_mixed_disk_and_cw_polygon_order2": (lambda: plane_mixed("disk_and_cw_polygon"),
        "7f6fcbb1ff3b4a40b247a0cdd62c1c4b864e768e789d599e5095fee8e55329ba",
        "5e4d9ac394c3990f267a2451fec96285e39ea09d28b101dcaf0040a8ff71d431"),
    "plane_mixed_ccw_and_cw_polygons_order2": (lambda: plane_mixed("ccw_and_cw_polygons"),
        "0d64f62e403c5608ad215cb95f2cac9ab9590c0e2c759a95d6a108d04911b7ba",
        "e67c7f2a2edcac3c71377886b7fccdca1be8bc4d0d644e4f2b302ea04a9174c6"),
    "bounded_mixed_disk_and_cw_polygon_order2": (lambda: bounded_mixed("disk_and_cw_polygon"),
        "c4d99dc95c592ac4199519f2d6ab47b1fd0493cdea2e9515fa2bcd1e22ff9395",
        "9f184245556912f30413fd7c62d8d81733670aaf10c2ea747c05a0e70144d043"),
    "bounded_mixed_ccw_and_cw_polygons_order2": (lambda: bounded_mixed("ccw_and_cw_polygons"),
        "ea6d3de4c8a5bd1c375bb7fe3addcd9699640a1405e9bcbfd47a6f4fda33885c",
        "de360ea8089dfb7f095f0f97d60030a02e520c0b4f5916ae3ebe352230963597"),
    "bounded_ring_2mm_order1": (lambda: bounded_ring(0.002, order=1),
        "02a3b3a5903888b563c2541cfde4b2cbabf0c517041f8b4eaed2398511050ef5",
        "ba23627c8087abbf820898d69fa79a151658379b0fd507a09111fc1742132642"),
}


@pytest.mark.parametrize("case", sorted(UNCHANGED))
def test_a_mesh_with_no_tangled_element_is_written_unchanged(case, tmp_path):
    """Valid order-2 meshes (thin, nearly degenerate, wound clockwise, one region of parts wound oppositely, the
    existing bounded fixture) and order-1 meshes of the thin cases: no record, nothing tangled, and on the platform of
    the references the bytes of the mesher without the check."""
    make, msh_sha, info_sha = UNCHANGED[case]
    inp = make()
    rc, info, msh, err = run_mesher(inp, tmp_path)
    assert rc == 0 and info["ok"] is True, (info, err[-2000:])
    assert "high_order" not in info, info["high_order"]
    if inp.get("order", 2) == 2:
        assert tangled_count(msh) == 0
    platform = os.environ.get("AXMP_G0_PLATFORM", "")
    if not (platform.startswith("lawrencium-x86_64") and _gmsh_version(info) == PINNED_GMSH):
        pytest.skip(f"the byte references are of gmsh {PINNED_GMSH} on lawrencium-x86_64 (here: "
                    f"'{platform}', gmsh {_gmsh_version(info)})")
    assert _sha(msh.read_bytes()) == msh_sha, f"{case}: the mesh file changed"
    assert _sha(_masked_info_text(tmp_path / "mesh_info.json").encode()) == info_sha, f"{case}: the info file changed"


def test_a_repaired_mesh_is_the_same_in_every_run(tmp_path):
    """The repair runs gmsh steps that order their work by memory address. The repaired 2 mm shell is the same file
    whatever the length of the output directory's name and the size of the environment (run in the mesher's own
    process, gmsh 4.15.2's passes give 7 different meshes in 8 such runs)."""
    shas, infos = set(), set()
    for k in range(3):
        out = tmp_path / ("d" * (7 * k + 1)) / "run"
        rc, info, msh, err = run_mesher(axi_shell(0.002), out, env={"AXMP_TEST_PADDING": "x" * (137 * (k + 1))})
        assert rc == 0 and info["ok"], (info, err[-2000:])
        assert info.get("high_order", {}).get("tangled_before", 0) >= 1 and tangled_count(msh) == 0, info
        shas.add(_sha(msh.read_bytes()))
        infos.add(_masked_info_text(out / "mesh_info.json"))
    assert len(shas) == 1, f"{len(shas)} different repaired meshes in 3 runs"
    assert len(infos) == 1, "the info files differ"


def test_order_one_is_not_checked_and_order_two_of_the_same_geometry_is(tmp_path):
    """The record appears with the repair only: absent at order 1, present at order 2 of the same 2 mm shell."""
    _, info1, _, _ = run_mesher(axi_shell(0.002, order=1), tmp_path / "o1")
    _, info2, _, _ = run_mesher(axi_shell(0.002), tmp_path / "o2")
    assert info1["ok"] and "high_order" not in info1
    assert info2.get("high_order", {}).get("tangled_before", 0) >= 1, info2


# ------------------------------------------------------------------------------------------------- hand-tangled files
def tangle_file(path, attr):
    """Moves the mid-node of one interior edge of the region `attr` (a 6-node triangle whose first edge is shared with
    another element of the same region) beyond the triangle's opposite vertex; returns the element's position in the
    file. Rewrites the MSH 2.2 file in place, node lines only."""
    lines = Path(path).read_text().split("\n")
    names, nodes, tags, phys, conn = read_msh22(path)
    owners = {}
    for e, (c, g) in enumerate(zip(conn, phys)):
        if len(c) == 6:
            for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[0])):
                owners.setdefault(frozenset((a, b)), []).append((e, g))
    for e, (c, g) in enumerate(zip(conn, phys)):
        if g != attr or len(c) != 6:
            continue
        own = owners[frozenset((c[0], c[1]))]
        if len(own) == 2 and all(o[1] == attr for o in own):
            m, v2 = np.array(nodes[c[3]]), np.array(nodes[c[2]])
            new = v2 + 0.5 * (v2 - m)
            i = lines.index("$Nodes")
            for j in range(i + 2, i + 2 + int(lines[i + 1])):
                if lines[j].split()[0] == str(c[3]):
                    lines[j] = f"{c[3]} {new[0]:.17g} {new[1]:.17g} 0"
                    break
            Path(path).write_text("\n".join(lines))
            return e
    raise AssertionError(f"no interior edge in the region {attr}")


STUB = """import json, subprocess, sys
sys.path[:0] = {paths!r}
import test_mesher_tangled as T
inp = json.load(open(sys.argv[1]))
rc = subprocess.run([{real!r}, sys.argv[1]]).returncode
if rc:
    sys.exit(rc)
names = T.read_msh22(inp["output"])[0]
T.tangle_file(inp["output"], T.region_tag(names, {region!r}))
"""


def _stub_mesher(tmp_path, region):
    """A mesher that runs the real one, then tangles one element of `region` and reports success (as an older mesher, a
    broken one or a hand edit would)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "stub_mesher.py"
    paths = [str(Path(__file__).resolve().parent)] + [str(Path(p).resolve()) for p in sys.path if p]
    script.write_text(STUB.format(paths=paths, real=_mesher(), region=region))
    exe = tmp_path / "stub-mesher"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    exe.chmod(0o755)
    return str(exe)


def test_the_hand_tangling_tangles(tmp_path):
    """The stand-in for a tangled file is tangled by the definition: the elements next to the moved node, and only in
    the chosen region."""
    rc, info, msh, err = run_mesher(axi_sphere(), tmp_path)
    assert rc == 0 and tangled_count(msh) == 0
    names = read_msh22(msh)[0]
    tangle_file(msh, region_tag(names, "ball"))
    mn, mx, phys, _, _ = det_range(msh)
    bad = tangled_mask(mn, mx, phys)
    assert 1 <= bad.sum() <= 2 and set(phys[bad]) == {region_tag(names, "ball")}, (bad.sum(), set(phys[bad]))


def _core():
    axmp = pytest.importorskip("axionmultiphysics")
    if not hasattr(axmp, "Design"):
        pytest.skip("axionmultiphysics without Design")
    try:
        axmp.build_info()
    except Exception:
        pytest.skip("axionmultiphysics._core not built")
    return axmp


def _code(exc):
    return getattr(exc, "code", None)


def sphere_doc(regions, *, order=2, surface_size=0.05):
    return {"schema": "axionmultiphysics.design/1", "name": "tangle_probe",
            "materials": {"fe1000": {"kind": "linear", "mu_r": 1000}},
            "regions": regions, "background": {"type": "uniform", "Bz": 0.1},
            "mesh": {"order": order, "size": 0.05, "surface_size": surface_size, "domain_radius": 1.5}}


def shell_doc(t, order=2):
    return sphere_doc([{"name": "shell", "material": "fe1000", "shape": {"contour": {"start": [0, -A], "segments": [
        {"ellipse": {"center": [0, 0], "axes": [A, A], "t": [-math.pi / 2, math.pi / 2]}}, {"line": [0, A - t]},
        {"ellipse": {"center": [0, 0], "axes": [A - t, A - t], "t": [math.pi / 2, -math.pi / 2]}},
        {"line": [0, -A]}]}}}], order=order)


def ball_doc():
    return sphere_doc([{"name": "ball", "material": "fe1000", "shape": {"contour": {"start": [0, -A], "segments": [
        {"ellipse": {"center": [0, 0], "axes": [A, A], "t": [-math.pi / 2, math.pi / 2]}}, {"line": [0, -A]}]}}}],
        surface_size=0.02)


def _assert_tangled_refusal(exc, count, region):
    """The library's refusal: code mesh.tangled; the message starts with the count ("<k> of <n> curved elements") and
    names the region; the hint gives the remedy."""
    msg = str(exc)
    assert _code(exc) == "mesh.tangled", (_code(exc), msg)
    assert re.match(rf"{count} of \d+ curved elements", msg), f"the message does not start with {count}: {msg}"
    assert f"'{region}'" in msg.split(" Hint: ")[0] and "tangled" in msg, msg
    hint = getattr(exc, "hint", "") or ""
    assert re.search(r"[Rr]efine", hint) and "order 1" in hint, hint


# ------------------------------------------------------------------------------------------------- the library
def test_the_em_solve_of_the_thin_shell_runs_on_the_repaired_mesh(tmp_path):
    """The 2 mm shell of the module docstring end to end: it solves, on a mesh with no tangled element, and the results
    carry the mesher's record."""
    axmp = _core()
    _mesher()
    res = axmp.Design(shell_doc(0.002)).solve(workdir=str(tmp_path), verbose=0)
    minfo = json.loads((tmp_path / "mesh_info.json").read_text())
    if not minfo.get("ok"):
        pytest.fail("unreachable: the solve returned on a refused mesh")
    assert tangled_count(tmp_path / "mesh.msh") == 0
    assert minfo["high_order"]["tangled_before"] >= 1 and minfo["high_order"]["tangled_after"] == 0
    assert res.data["mesh"]["info"]["mesher"]["high_order"] == minfo["high_order"]


def test_the_em_solve_reports_the_mesher_refusal(tmp_path, monkeypatch):
    """With the optimisation off the mesher refuses the 2 mm shell; the solve raises the error of a failed mesher run
    with the mesher's text (count, region, remedy) and writes no results."""
    axmp = _core()
    _mesher()
    monkeypatch.setenv("AXMP_MESHER_HO", "none")
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        axmp.Design(shell_doc(0.002)).solve(workdir=str(tmp_path), verbose=0)
    msg = str(e.value)
    assert msg.startswith("mesh generation failed: "), msg
    minfo = json.loads((tmp_path / "mesh_info.json").read_text())
    assert minfo["ok"] is False and minfo["error"] in msg, (minfo, msg)
    assert "'shell'" in msg and "order 1" in msg and re.search(r"[Rr]efine the mesh in or around", msg), msg
    assert not (tmp_path / "results.json").exists()


def test_axmp_mesh_reports_the_mesher_refusal_as_any_mesher_failure(tmp_path):
    """`axmp mesh` on the 2 mm shell with the optimisation off: exit status 1 and the JSON error of a failed mesher run
    (the same keys and location as a mesher that fails for another reason), carrying the mesher's text."""
    _mesher()
    exe = _paths.axmp_path()
    if not exe:
        pytest.skip("axmp executable not found")
    doc = tmp_path / "shell.json"
    doc.write_text(json.dumps(shell_doc(0.002)))
    env = dict(os.environ, AXMP_MESHER_HO="none")
    p = subprocess.run([exe, "mesh", str(doc), "-o", str(tmp_path / "m")], capture_output=True, text=True, env=env)
    assert p.returncode == 1, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    out = json.loads(p.stdout)
    minfo = json.loads((tmp_path / "m" / "mesh_info.json").read_text())
    assert out["ok"] is False and minfo["error"] in out["error"], (out, minfo)
    assert out["error"].startswith("mesh generation failed: ")
    # another failed mesher run: a mesher that exits 1 with its own error
    other = tmp_path / "failing-mesher"
    other.write_text("#!/bin/sh\n"
                     f"{sys.executable} -c \"import json,sys; d=json.load(open(sys.argv[1])); "
                     "json.dump({'ok': False, 'error': 'some other failure'}, open(d['info'], 'w'))\" \"$1\"\n"
                     "exit 1\n")
    other.chmod(0o755)
    q = subprocess.run([exe, "mesh", str(doc), "-o", str(tmp_path / "o"), "--mesher", str(other)], capture_output=True,
                       text=True, env=env)
    ref = json.loads(q.stdout)
    assert q.returncode == p.returncode and sorted(ref) == sorted(out), (ref, out)
    assert ref.get("where") == out.get("where") and ref.get("code") == out.get("code"), (ref, out)


def test_the_library_refuses_a_tangled_mesh_that_the_mesher_reported_as_good(tmp_path, monkeypatch):
    """The EM axisymmetric solve on a mesher that tangles one element of 'ball' and reports success: mesh.tangled with
    the count of the file (the independent det J) and the region, before anything is solved."""
    axmp = _core()
    _mesher()
    monkeypatch.setenv("AXMP_MESHER", _stub_mesher(tmp_path / "stub", "ball"))
    wd = tmp_path / "run"
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        axmp.Design(ball_doc()).solve(workdir=str(wd), verbose=0)
    assert json.loads((wd / "mesh_info.json").read_text())["ok"] is True
    _assert_tangled_refusal(e.value, tangled_count(wd / "mesh.msh"), "ball")
    assert not (wd / "results.json").exists()


def test_the_library_refuses_a_tangled_cross_section_mesh(tmp_path, monkeypatch):
    """The same on the plane mode (the cross-section cavity solver), on its disk region 'vacuum'."""
    axmp = _core()
    _mesher()
    from axionmultiphysics import cavity_xsec as cx

    d = cx.cylinder_rods_design(0.05, [], modes=2, h=0.05 / 8)
    monkeypatch.setenv("AXMP_MESHER", _stub_mesher(tmp_path / "stub", d.doc["regions"][0]["name"]))
    wd = tmp_path / "run"
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d.solve(str(wd), verbose=0)
    assert json.loads((wd / "mesh_info.json").read_text())["ok"] is True
    _assert_tangled_refusal(e.value, tangled_count(wd / "mesh.msh"), d.doc["regions"][0]["name"])


def _imported_plates(tmp_path, tangle):
    """The 2D full-wave design of test_comsol_import2d.py on an imported second-order mesh (region 'field', attribute
    2), with one element tangled by hand when `tangle`."""
    from axionmultiphysics.fw2d import mapped_mesh, write_mesh

    tmp_path.mkdir(parents=True, exist_ok=True)
    pts, tri, reg, edges, side = mapped_mesh(lambda u, v: (0.3 * u, 0.02 * v), 20, 2)
    attr = {0: 10002, 1: 10002, 2: 10002, 3: 10001}
    write_mesh(tmp_path / "m.msh", pts, tri, 2 + reg, edges, [attr[s] for s in side])
    (tmp_path / "info.json").write_text(json.dumps({"boundary_groups": [
        {"attr": 10001, "name": "lport1", "type": "port"}, {"attr": 10002, "name": "walls", "type": "pec"}]}))
    if tangle:
        tangle_file(tmp_path / "m.msh", 2)
    return {"schema": "axionmultiphysics.design/1", "name": "x", "symmetry": "planar",
            "regions": [{"name": "field", "material": "vacuum", "shape": {"imported": {}}}],
            "fullwave": {"boundary": "pec", "rf_only": True, "frequencies": [1e8], "thickness_m": 0.05},
            "pickup": {"type": "port", "port_boundary": "lport1"},
            "mesh": {"order": 2, "file": str(tmp_path / "m.msh"), "info_file": str(tmp_path / "info.json")}}


def test_the_library_refuses_a_tangled_imported_mesh(tmp_path):
    """An imported curved mesh (mesh.file) cannot bypass the check: refused with mesh.tangled; the same mesh untangled
    solves."""
    axmp = _core()
    if not hasattr(axmp._core, "Fw2DModel"):
        pytest.skip("axionmultiphysics._core without Fw2DModel")
    good = _imported_plates(tmp_path / "good", tangle=False)
    assert tangled_count(tmp_path / "good" / "m.msh") == 0
    axmp.Design(good).solve(workdir=str(tmp_path / "good" / "r"), verbose=0)
    bad = _imported_plates(tmp_path / "bad", tangle=True)
    n = tangled_count(tmp_path / "bad" / "m.msh")
    assert n >= 1
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        axmp.Design(bad).solve(workdir=str(tmp_path / "bad" / "r"), verbose=0)
    _assert_tangled_refusal(e.value, n, "field")
    assert not (tmp_path / "bad" / "r" / "results.json").exists()


def _numbers(x, path=""):
    """The numeric leaves of a results tree, timings left out."""
    if isinstance(x, dict):
        for k, v in x.items():
            if "second" not in k and "time" not in k:
                yield from _numbers(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _numbers(v, f"{path}/{i}")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def test_a_reversed_element_is_not_tangled_and_changes_nothing(tmp_path):
    """Orientation does not make an element tangled: in an imported mesh, one element written with its nodes in the
    reverse order (det J < 0 throughout, against the rest of its region in the file), or every element so written, is
    read in a consistent orientation (MFEM orients the elements of a mesh it reads), not refused, and the full-wave
    results are those of the original file (relative 1e-9; measured: equal)."""
    axmp = _core()
    if not hasattr(axmp._core, "Fw2DModel"):
        pytest.skip("axionmultiphysics._core without Fw2DModel")

    def reverse(path, which):
        lines = Path(path).read_text().split("\n")
        i = lines.index("$Elements")
        seen = 0
        for j in range(i + 2, i + 2 + int(lines[i + 1])):
            a = lines[j].split()
            if a[1] != "9":
                continue
            if which == "all" or seen == which:
                nt = int(a[2])
                v = a[3 + nt:]
                # gmsh order v0 v1 v2 e01 e12 e20 -> v0 v2 v1 e20 e12 e01
                a[3 + nt:] = [v[0], v[2], v[1], v[5], v[4], v[3]]
                lines[j] = " ".join(a)
            seen += 1
        Path(path).write_text("\n".join(lines))

    ref = axmp.Design(_imported_plates(tmp_path / "ref", tangle=False)).solve(workdir=str(tmp_path / "ref" / "r"),
                                                                              verbose=0)
    want = dict(_numbers(ref.data["fullwave"]))
    assert want
    for name, which in (("one", 7), ("all", "all")):
        d = _imported_plates(tmp_path / name, tangle=False)
        reverse(tmp_path / name / "m.msh", which)
        mn, mx, phys, _, _ = det_range(tmp_path / name / "m.msh")
        assert (mx < 0).sum() == (1 if which == 7 else len(mx))  # the stand-in is what it says
        got = dict(_numbers(axmp.Design(d).solve(workdir=str(tmp_path / name / "r"), verbose=0).data["fullwave"]))
        assert set(got) == set(want), name
        bad = [(k, want[k], got[k]) for k in want if abs(got[k] - want[k]) > 1e-9 * max(abs(want[k]), 1e-300)]
        assert not bad, (name, bad[:5])


def test_an_em_design_whose_region_mixes_windings_solves(tmp_path):
    """One iron region made of a disk and a polygon listed clockwise (gmsh meshes the two parts in opposite windings):
    the EM axisymmetric design meshes without a record and solves, as with the mesher without the check."""
    axmp = _core()
    _mesher()
    doc = sphere_doc([{"name": "mixed", "material": "fe1000", "shape": MIXED_WINDINGS["disk_and_cw_polygon"]}],
                     surface_size=0.02)
    res = axmp.Design(doc).solve(workdir=str(tmp_path), verbose=0)
    minfo = json.loads((tmp_path / "mesh_info.json").read_text())
    assert minfo["ok"] is True and "high_order" not in minfo, minfo.get("high_order")
    assert tangled_count(tmp_path / "mesh.msh") == 0
    assert (tmp_path / "results.json").exists() and res.data["mesh"]["info"]["mesher"]["ok"] is True


# A 6-node triangle folded between the lattice points: on the reference triangle the vertices and the
# mid-node of edge 20 stay, the mid-nodes of edges 01 and 12 move to these points. det J is positive at every point
# (i/12, j/12) (minimum 0.0024) and negative between them (exact minimum -0.0026, near (0.199, 0.801)); at its vertices
# it is at least 0.19. Moving the two mid-nodes 1.1 times as far makes it negative on the lattice too (-0.127) while
# the vertices stay positive (0.108).
FOLD_MIDS = {3: (0.9657945449427133, -0.4964975764648499), 4: (0.2972474467424332, 0.3907841135936919)}


def embed_fold(path, scale):
    """Folds the first corner triangle of a mesh file (two edges on the boundary, so no other element changes): its
    mid-nodes on those edges move to FOLD_MIDS (scaled from the straight positions) in the triangle's own affine frame.
    Returns the element's index among the file's triangles."""
    names, nodes, tags, phys, conn = read_msh22(path)
    count, mid = {}, {}
    for c in conn:
        for (a, b), m in (((c[0], c[1]), c[3]), ((c[1], c[2]), c[4]), ((c[2], c[0]), c[5])):
            count[frozenset((a, b))] = count.get(frozenset((a, b)), 0) + 1
            mid[frozenset((a, b))] = m
    for e, c in enumerate(conn):
        for r in range(3):  # a cyclic relabelling keeps the orientation
            w = c[:3][r:] + c[:3][:r]
            if count[frozenset((w[0], w[1]))] == 1 and count[frozenset((w[1], w[2]))] == 1:
                V = np.array([nodes[k] for k in w])
                A = np.column_stack([V[1] - V[0], V[2] - V[0]])
                new = {}
                for k, (a, b) in ((3, (w[0], w[1])), (4, (w[1], w[2]))):
                    straight = np.array(NODES6[k])
                    new[mid[frozenset((a, b))]] = V[0] + A @ (straight + scale * (np.array(FOLD_MIDS[k]) - straight))
                lines = Path(path).read_text().split("\n")
                i = lines.index("$Nodes")
                for j in range(i + 2, i + 2 + int(lines[i + 1])):
                    k = int(lines[j].split()[0])
                    if k in new:
                        lines[j] = f"{k} {new[k][0]:.17g} {new[k][1]:.17g} 0"
                Path(path).write_text("\n".join(lines))
                return e
    raise AssertionError("no corner triangle")


@pytest.mark.parametrize("scale, on_lattice", [(1.0, False), (1.1, True)], ids=["between_lattice_points", "deeper"])
def test_a_fold_inside_an_element_is_refused_on_read(scale, on_lattice, tmp_path):
    """One element of an imported curved mesh folds inside while det J stays positive at its three vertices; at scale 1
    also at every point of the lattice (i/12, j/12). The read refuses it in both cases (mesh.tangled, one element, the
    region 'field'): the check is exact for 6-node triangles. A check at the vertices only accepts both; one on the
    lattice accepts the first. (The mesher's own check cannot be given such an element: in every tangled mesh gmsh made
    in this file and in about a hundred other thin or curved geometries, each tangled element has det J <= 0 at a
    vertex.)"""
    axmp = _core()
    if not hasattr(axmp._core, "Fw2DModel"):
        pytest.skip("axionmultiphysics._core without Fw2DModel")
    d = _imported_plates(tmp_path, tangle=False)
    e = embed_fold(tmp_path / "m.msh", scale)
    P, phys, names = mesh_triangles(tmp_path / "m.msh")
    exact_lo, _ = exact_range(P)
    vertex_lo, _ = sampled_range(P, VERTICES)
    lattice_lo, _ = sampled_range(P, LATTICE)
    assert (vertex_lo > 0).all(), "a vertex is not positive"
    assert list(np.where(exact_lo <= 0)[0]) == [e], np.where(exact_lo <= 0)
    assert list(np.where(lattice_lo <= 0)[0]) == ([e] if on_lattice else []), np.where(lattice_lo <= 0)
    with pytest.raises(axmp.AxionMultiPhysicsError) as exc:
        axmp.Design(d).solve(workdir=str(tmp_path / "r"), verbose=0)
    _assert_tangled_refusal(exc.value, 1, "field")
    assert not (tmp_path / "r" / "results.json").exists()


def test_a_valid_thin_curved_shell_is_not_refused(tmp_path):
    """The 3 mm shell (det J ratio down to 0.198) and the sphere: valid curved meshes are read and solved (the
    check is on the sign, not on a quality threshold)."""
    axmp = _core()
    _mesher()
    for name, doc in (("shell3", shell_doc(0.003)), ("ball", ball_doc())):
        axmp.Design(doc).solve(workdir=str(tmp_path / name), verbose=0)
        minfo = json.loads((tmp_path / name / "mesh_info.json").read_text())
        assert "high_order" not in minfo and tangled_count(tmp_path / name / "mesh.msh") == 0
