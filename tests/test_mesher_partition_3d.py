"""The mesher's 3D partition: ``axmp-mesher --export-regions`` on the input of a 3D electromagnetic (EM) design.

The export builds the 3D EM geometry as ``axmp-mesher INPUT`` does (the solids, the domain sphere, the infinity shell,
the fragmentation, the region of each volume by priority) and writes, instead of a mesh, the partition of that
geometry to INPUT's ``info``: the faces as entities of ``dim`` 2 (each with its two side regions), the ``volumes``
(region and oriented shells of face tags) and one surface triangulation per face, shared by the two volumes the face
bounds. The infinity shell is not part of it.

Designs (``_partition3d_ref``): a window-frame iron dipole (frame, cylindrical poles cutting notches into it, ring
coils, two scintillator planes) and a design with every surface kind (a hollow box with a ball in its air pocket, a
torus, a cone, a block crossed by a rod). Both have closed forms for every region volume.

Independent references: the closed forms; the exact surfaces the faces lie on (planes fitted to the nodes; the
design's cylinders, spheres, tori and cones), from which the tests compute the triangles' distance to the face; the EM
mesh of the same input (the integral of det J over each region's P2 tetrahedra, exact for the cubic Jacobian, with the
bound of its curved boundary's deviation from the exact surfaces); records of the mesher before the export existed
(``tests/data/mesher_partition/reference_records_3d.json``).

Tolerances (measured on an implementation written for these tests, not the package's):
- nodes on their surface: ``tolerance_m`` (1e-9 x the diagonal) for curved faces (measured 3e-17 m), 1e-12 x the
  diagonal for the plane through a planar face's nodes (measured 4e-15 m);
- every triangle within ``chord_tol_m`` of its face (centroid, edge midpoints and seven interior points), plus
  ``tolerance_m``;
- a region's exact volume (``regions[*].volume_m3``) against the closed form: 1e-12 relative (OpenCASCADE integrates
  these analytic solids to rounding: measured at most 7e-16);
- against the EM mesh: twice the P2 bound plus 1e-12 of the volume (measured at most 0.67 of the bound);
- the triangulated volume against the exact one: the sum over the triangles of curved faces of the area times the
  largest sampled distance from the face, plus 1e-12 of the volume (a closed polyhedron whose vertices lie on the
  exact surface differs from the solid only under the triangles of curved faces).

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pytest
from scipy.spatial import cKDTree

import _partition3d_ref as R3
import _partition_ref as R
from _partition_ref import mp  # noqa: F401 (a fixture)

NAMES = list(R3.DESIGNS3)


def _export(mesher, inp, directory, name="x", chord=R3.TEST_CHORD):
    inp = copy.deepcopy(inp)
    if chord is not None:
        inp["chord_tol_m"] = chord
    return R.run_mesher(mesher, inp, directory, name, export=True)


@pytest.fixture(scope="module")
def runs(mp, tmp_path_factory):  # noqa: F811
    """Per design: the mesher input, the export (process, partition, mesh path) and the EM mesh (process, info, mesh)."""
    out = {}
    for name in NAMES:
        d = tmp_path_factory.mktemp(f"p3_{name}")
        inp = R3.mesher_input_3d(R3.design3(name))
        r, part, msh = _export(mp, inp, d / "export")
        em_r, em_info, em_msh = R.run_mesher(mp, inp, d / "em", "em")
        out[name] = {"input": inp, "r": r, "part": part, "msh": msh, "em_r": em_r, "em_info": em_info, "em_msh": em_msh}
    return out


def _part(runs, name):
    run = runs[name]
    assert run["r"].returncode == 0 and isinstance(run["part"], dict) and run["part"].get("ok") is True, (
        f"axmp-mesher --export-regions on a 3D input exited with {run['r'].returncode}: "
        f"{(run['part'] or {}).get('error')!r} {run['r'].stderr[-1500:]}")
    return run["part"]


def _em(runs, name):
    run = runs[name]
    assert run["em_r"].returncode == 0 and run["em_info"]["ok"] is True, run["em_r"].stderr[-1500:]
    return R3.MshBin(run["em_msh"])


def _face_surfaces(part, name):
    """{face tag: exact surface} from the face's nodes (a design surface within tolerance_m, else a plane within 1e-12 x
    the diagonal); asserts that every face lies on one."""
    X, tri, _ = R3.shells_of(part)
    curved = R3.curved_surfaces(name)
    out = {}
    for tag, T in tri.items():
        nodes = X[np.unique(T)]
        s = R3.surface_of(nodes, curved, part["tolerance_m"])
        if s is None:
            s = R3.surface_of(nodes, [], 1e-12 * part["diagonal_m"])
        assert s is not None, f"face {tag}: its nodes lie on no surface of the design and on no plane"
        out[tag] = s
    return out


# ============================================================================================ the file
@pytest.mark.parametrize("name", NAMES)
def test_the_3d_export_writes_the_partition_and_no_mesh(runs, name):
    part = _part(runs, name)
    run = runs[name]
    assert set(part) == R3.EXPORT3_KEYS, sorted(set(part) ^ R3.EXPORT3_KEYS)
    assert part["export_regions"] is True and part["geometry"] == "3d"
    assert not run["msh"].exists(), "the export wrote a mesh"
    em_info = run["em_info"]
    assert part["domain_radius"] == em_info["domain_radius"] == em_info["shell_inner_radius"]
    assert part["domain_center"] == [0, 0, 0] or np.array_equal(np.array(part["domain_center"], float), np.zeros(3))
    lo = np.min([e["lo"] for e in part["entities"]], axis=0)
    hi = np.max([e["hi"] for e in part["entities"]], axis=0)
    assert part["diagonal_m"] == pytest.approx(float(np.linalg.norm(hi - lo)), rel=1e-12)
    assert part["tolerance_m"] == pytest.approx(1e-9 * part["diagonal_m"], rel=1e-12)
    assert part["chord_tol_m"] == R3.TEST_CHORD
    design = R3.design3(name)
    want = {"air": R3.AIR_ATTR} | {r["name"]: 2 + k for k, r in enumerate(design["regions"])}
    assert set(part["regions"]) == set(want), sorted(part["regions"])
    for r, rec in part["regions"].items():
        assert set(rec) == {"attr", "volume_m3"} and rec["attr"] == want[r], (r, rec)


@pytest.mark.parametrize("name", NAMES)
def test_faces_are_entities_of_dim_2_with_their_sides(runs, name):
    part = _part(runs, name)
    tags = [e["tag"] for e in part["entities"]]
    assert tags == sorted(tags) and len(set(tags)) == len(tags)
    assert [f["tag"] for f in part["triangulation"]["faces"]] == tags
    names = set(part["regions"]) | {"exterior"}
    for e in part["entities"]:
        assert set(e) == R3.FACE_KEYS, (e["tag"], sorted(set(e) ^ R3.FACE_KEYS))
        assert e["dim"] == 2 and e["curves"] == [] and e["axis"] is False, e["tag"]
        assert isinstance(e["type"], str) and e["type"], e["tag"]
        assert len(e["side"]) == 2 and set(e["side"]) <= names and e["side"] != ["exterior", "exterior"], e
    for f in part["triangulation"]["faces"]:
        assert set(f) == {"tag", "triangles", "sagitta_m"}, sorted(f)
        assert len(f["triangles"]) > 0, f["tag"]


@pytest.mark.parametrize("name", NAMES)
def test_each_volume_is_closed_and_consistently_oriented(runs, name):
    """Watertight: each shell of a volume, its faces' triangles turned by the sign of their tag, is a closed surface
    oriented outwards (every edge used once in each direction); the outer shell (the first) encloses a positive
    volume and every other shell a negative one. Every face is used once with + by a volume of its side[0] and once
    with - by a volume of its side[1] (once only when the other side is the exterior)."""
    part = _part(runs, name)
    X, tri, vols = R3.shells_of(part)
    side = {e["tag"]: e["side"] for e in part["entities"]}
    uses = {t: [] for t in tri}
    regions_of = {v["tag"]: v["region"] for v in part["volumes"]}
    for v in part["volumes"]:
        assert v["region"] in part["regions"] and v["region"] != "infinity_shell", v["region"]
        assert v["shells"] and all(sh for sh in v["shells"]), v["tag"]
        seen = [abs(s) for sh in v["shells"] for s in sh]
        assert len(seen) == len(set(seen)), f"volume {v['tag']} uses a face twice"
        for sh in v["shells"]:
            for s in sh:
                uses[abs(s)].append((1 if s > 0 else -1, v["tag"]))
        for k, (sh, T) in enumerate(zip(v["shells"], vols[v["tag"]])):
            bad = R3.unbalanced_edges(T)
            assert bad == 0, f"volume {v['tag']} ({v['region']}), shell {k} {sh}: {bad} edges not closed"
            vol = R3.signed_volume(X, T)
            assert (vol > 0) if k == 0 else (vol < 0), (v["tag"], k, vol)
    for t, u in uses.items():
        s0, s1 = side[t]
        want = sorted([(1, s0)] * (s0 != "exterior") + [(-1, s1)] * (s1 != "exterior"))
        got = sorted((sg, regions_of[vt]) for sg, vt in u)
        assert got == want, f"face {t} (sides {side[t]}) is used as {got}"
    assert sorted(regions_of.values()) and set(regions_of.values()) == set(part["regions"])


@pytest.mark.parametrize("name", NAMES)
def test_faces_share_their_nodes_bit_for_bit(runs, name):
    """One node list for every face: the faces that meet at a curve or a corner refer to the same node (so the two
    volumes a face bounds, and the faces around an edge, have the same coordinates bit for bit). Every node is used, no
    two nodes are within tolerance_m of each other (a duplicated seam), no triangle is degenerate or listed twice."""
    part = _part(runs, name)
    X, tri, _ = R3.shells_of(part)
    assert X.ndim == 2 and X.shape[1] == 3 and np.isfinite(X).all()
    allt = np.concatenate(list(tri.values()))
    assert allt.min() >= 0 and allt.max() < len(X)
    assert np.array_equal(np.unique(allt), np.arange(len(X))), "a node of the triangulation is not used"
    pairs = cKDTree(X).query_pairs(part["tolerance_m"])
    assert not pairs, f"{len(pairs)} pairs of nodes closer than tolerance_m, e.g. {X[list(pairs)[0][0]]}"
    assert (allt[:, 0] != allt[:, 1]).all() and (allt[:, 1] != allt[:, 2]).all() and (allt[:, 0] != allt[:, 2]).all()
    assert (R3.triangle_areas(X, allt) > 0).all()
    key = np.sort(allt, axis=1)
    assert len(np.unique(key, axis=0)) == len(key), "a triangle is listed twice"
    # faces meet at their boundary curves: every node on the boundary of a face's triangulation (an edge used by one
    # of its triangles only) is a node of another face too (a closed face, the domain sphere, has no boundary)
    owner = {}
    for t, T in tri.items():
        for n in np.unique(T):
            owner.setdefault(int(n), set()).add(t)
    for t, T in tri.items():
        e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        uniq, cnt = np.unique(e, axis=0, return_counts=True)
        assert cnt.max() <= 2, f"face {t}: an edge of three triangles"
        lonely = [int(n) for n in np.unique(uniq[cnt == 1]) if len(owner[int(n)]) == 1]
        assert not lonely, f"face {t}: {len(lonely)} nodes of its boundary belong to no other face, e.g. {X[lonely[0]]}"


@pytest.mark.parametrize("name", NAMES)
def test_the_infinity_shell_is_excluded(runs, name):
    """No volume of the shell; the transport's domain ends at the shell's inner radius: every node within it, and the
    faces on the exterior side are exactly those on the domain sphere, which they cover."""
    part = _part(runs, name)
    msh = _em(runs, name)
    p, e, _ = R3.tet_volumes(msh)
    reg = msh.region_of(p)
    shell_tags = set(e[reg == "infinity_shell"].tolist())
    assert shell_tags, "the EM mesh has no infinity shell (the test's premise)"
    assert not shell_tags & {v["tag"] for v in part["volumes"]}
    assert "infinity_shell" not in part["regions"]
    X, tri, _ = R3.shells_of(part)
    c, Rd = R3.domain(name)
    assert part["domain_radius"] == Rd
    assert float(np.max(np.linalg.norm(X - c, axis=1))) <= Rd * (1 + 1e-12)
    sphere = R3.Sphere(c, Rd)
    area = 0.0
    for ent in part["entities"]:
        T = tri[ent["tag"]]
        on = float(np.max(sphere.distance(X[np.unique(T)]))) <= part["tolerance_m"]
        assert on == ("exterior" in ent["side"]), (ent["tag"], ent["side"], on)
        if on:
            assert ent["side"][0] == "air" or ent["side"][1] == "air", ent["side"]
            area += float(R3.triangle_areas(X, T).sum())
    full = 4 * math.pi * Rd ** 2
    assert full * (1 - 4 * part["chord_tol_m"] / Rd) <= area <= full, (area, full)


@pytest.mark.parametrize("name", NAMES)
def test_nodes_lie_on_their_faces_and_triangles_within_the_chord_tolerance(runs, name):
    part = _part(runs, name)
    X, tri, _ = R3.shells_of(part)
    surf = _face_surfaces(part, name)
    tol, chord = part["tolerance_m"], part["chord_tol_m"]
    sag = {f["tag"]: f["sagitta_m"] for f in part["triangulation"]["faces"]}
    for t, T in tri.items():
        s = surf[t]
        cp = s.distance(R3.chord_points(X, T).reshape(-1, 3))
        ip = s.distance(R3.interior_points(X, T).reshape(-1, 3))
        worst = max(float(cp.max()), float(ip.max()))
        assert worst <= chord + tol, f"face {t} ({s.kind}): a triangle is {worst:.3g} m from the face (chord {chord})"
        if s.kind == "plane":
            assert 0.0 <= sag[t] <= tol, (t, sag[t])
        else:
            # sagitta_m: the largest distance of a triangle's centroid or edge midpoint from the face
            assert abs(sag[t] - float(cp.max())) <= 1e-6 * sag[t] + tol, (t, sag[t], float(cp.max()))


@pytest.mark.parametrize("name", NAMES)
def test_the_face_records_describe_the_faces(runs, name):
    """type (plane, cylinder, sphere, torus, cone: gmsh's type in lower case), box (holds the face: its nodes, and the
    exact extent of a whole sphere or torus; beyond its nodes' box by at most 4 chord tolerances, the depth at which a
    face's extreme point can lie beyond its nearest nodes, plus 1e-6 m), point (on the face), normal (the surface's,
    turned from side[0] to side[1] like the triangles) and area (the exact face's: it differs from the triangulated area
    by at most what the chords of its boundary and of its curvature move)."""
    part = _part(runs, name)
    X, tri, _ = R3.shells_of(part)
    surf = _face_surfaces(part, name)
    tol, chord = part["tolerance_m"], part["chord_tol_m"]
    c, Rd = R3.domain(name)
    for e in part["entities"]:
        t, s = e["tag"], surf[e["tag"]]
        T = tri[t]
        assert e["type"] == s.kind, (t, e["type"], s.kind)
        nodes = X[np.unique(T)]
        lo, hi = np.array(e["lo"]), np.array(e["hi"])
        assert (lo <= nodes.min(axis=0) + tol).all() and (hi >= nodes.max(axis=0) - tol).all(), (t, lo, hi)
        slack = 4 * chord + 1e-6
        assert (lo >= nodes.min(axis=0) - slack).all() and (hi <= nodes.max(axis=0) + slack).all(), (
            t, e["type"], lo - nodes.min(axis=0), hi - nodes.max(axis=0))
        ext = _whole_surface_extent(s, nodes, tol)
        if ext is not None:
            assert (lo <= ext[0] + tol).all() and (hi >= ext[1] - tol).all(), (t, e["type"], lo, hi, ext)
        p = np.array(e["point"], float)
        assert float(s.distance(p[None])[0]) <= tol, (t, "point off the face's surface")
        d_tri = float(np.min(_point_triangle_distance(p, X, T)))
        assert d_tri <= chord + tol, (t, "point outside the face", d_tri)
        n = np.array(e["normal"], float)
        assert abs(np.linalg.norm(n) - 1.0) <= 1e-12, (t, n)
        ns = _surface_normal(s, p)
        assert np.linalg.norm(np.cross(n, ns)) <= 1e-9, (t, n, ns)
        k = int(np.argmin(_point_triangle_distance(p, X, T)))
        a, b, cc = X[T[k]]
        assert float(np.cross(b - a, cc - a) @ n) > 0, (t, "normal against the triangles' orientation")
        A_tri = float(R3.triangle_areas(X, T).sum())
        L = _boundary_length(X, T)
        rho = getattr(s, "rho", math.inf)
        allowance = L * chord + (2 * A_tri * chord / rho if math.isfinite(rho) else 0.0)
        assert abs(e["area_m2"] - A_tri) <= allowance + 1e-12 * e["area_m2"], (t, e["area_m2"], A_tri, allowance)


@pytest.mark.parametrize("name", NAMES)
def test_region_volumes_equal_the_em_mesh_s_and_the_closed_forms(runs, name):
    """Per region: the exact volume (``regions[*].volume_m3``, the sum of its volumes' ``volume_m3``) against the closed
    form and against the EM mesh of the same input; the volume tags are the EM mesh's elementary tags of the region,
    and each volume's ``volume_m3`` its tetrahedra's; the triangulated volume (outer shell minus inner shells) against
    the exact one within the chords' bound."""
    part = _part(runs, name)
    msh = _em(runs, name)
    p, e, v = R3.tet_volumes(msh)
    reg = msh.region_of(p)
    nd = msh.elements[11][2]
    curved = R3.curved_surfaces(name)
    tol_nodes = 1e-9 * part["diagonal_m"]
    exact = R3.exact_volumes(name)
    X, tri, vols = R3.shells_of(part)
    surf = _face_surfaces(part, name)
    assert set(exact) == set(part["regions"])
    for r, V in exact.items():
        Vp = part["regions"][r]["volume_m3"]
        assert abs(Vp / V - 1.0) <= 1e-12, f"{r}: exported {Vp!r}, closed form {V!r}"
        mine = [x for x in part["volumes"] if x["region"] == r]
        assert sum(x["volume_m3"] for x in mine) == pytest.approx(Vp, rel=1e-14), r
        m = reg == r
        assert {x["tag"] for x in mine} == set(e[m].tolist()), (r, [x["tag"] for x in mine], sorted(set(e[m].tolist())))
        for x in mine:
            mt = m & (e == x["tag"])
            B = R3.p2_boundary_bound(msh, nd[mt], curved, tol_nodes)
            assert abs(x["volume_m3"] - v[mt].sum()) <= 2 * B + 1e-12 * x["volume_m3"], (r, x["tag"])
        B = R3.p2_boundary_bound(msh, nd[m], curved, tol_nodes)
        Vem = float(v[m].sum())
        assert abs(Vp - Vem) <= 2 * B + 1e-12 * Vp, f"{r}: exported {Vp}, EM mesh {Vem}, bound {2 * B:.3g}"
        Vt, bound = 0.0, 0.0
        for x in mine:
            Vt += sum(R3.signed_volume(X, T) for T in vols[x["tag"]])
            for sh in x["shells"]:
                for s in sh:
                    srf = surf[abs(s)]
                    if srf.kind != "plane":
                        T = tri[abs(s)]
                        dev = srf.distance(np.concatenate([R3.chord_points(X, T), R3.interior_points(X, T)], 1)
                                           .reshape(-1, 3)).reshape(len(T), -1).max(axis=1)
                        bound += float(R3.triangle_areas(X, T) @ dev)
        assert abs(Vt - Vp) <= bound + 1e-12 * Vp, f"{r}: triangulated {Vt}, exact {Vp}, bound {bound:.3g}"


# ============================================================================================ what it depends on
def test_the_partition_depends_on_the_geometry_and_the_chord_tolerance_only(mp, runs, tmp_path):  # noqa: F811
    """The same file (``seconds`` aside) from a second run, and from inputs that differ only in what the EM mesh
    depends on: element sizes, refinement, curvature points, the order, PEC removal (the conductors removed from the
    EM mesh are volumes of the partition)."""
    name = "dipole"
    base = _part(runs, name)
    inp = runs[name]["input"]
    variants = {"again": inp}
    v = copy.deepcopy(inp)
    v.update(size=0.05, surface_size=0.02, refine_regions=[], curvature_points=24, order=1, far_size=0.3)
    variants["sizes"] = v
    v = copy.deepcopy(inp)
    v["regions"][3]["remove_pec"] = True
    v["regions"][3]["size"] = 0.02
    variants["remove_pec"] = v
    for key, inp2 in variants.items():
        r, part, _ = _export(mp, inp2, tmp_path / key)
        assert r.returncode == 0 and part and part.get("ok") is True, (key, r.stderr[-1500:])
        assert R.same_json(R.without_seconds(part), R.without_seconds(base)) == [], key


def test_the_chord_tolerance_controls_the_triangulation(mp, runs, tmp_path):  # noqa: F811
    """Without ``chord_tol_m`` the export uses min(1e-3, 1e-4 x diagonal_m); a finer tolerance gives a finer
    triangulation of the curved faces (planar regions keep their volume to rounding) and a smaller volume deficit."""
    name = "dipole"
    coarse = _part(runs, name)
    inp = runs[name]["input"]
    r, dflt, _ = _export(mp, inp, tmp_path / "default", chord=None)
    assert r.returncode == 0 and dflt and dflt.get("ok") is True, r.stderr[-1500:]
    assert dflt["chord_tol_m"] == min(1e-3, 1e-4 * dflt["diagonal_m"])
    assert dflt["diagonal_m"] == coarse["diagonal_m"]
    r, fine, _ = _export(mp, inp, tmp_path / "fine", chord=R3.TEST_CHORD / 4)
    assert r.returncode == 0 and fine and fine.get("ok") is True, r.stderr[-1500:]
    assert fine["chord_tol_m"] == R3.TEST_CHORD / 4
    for part in (dflt, fine):
        surf = _face_surfaces(part, name)
        X, tri, _ = R3.shells_of(part)
        for t, T in tri.items():
            d = surf[t].distance(R3.chord_points(X, T).reshape(-1, 3)).max()
            assert d <= part["chord_tol_m"] + part["tolerance_m"], (t, d, part["chord_tol_m"])

    def deficit(part, region):
        X, _, vols = R3.shells_of(part)
        Vt = sum(R3.signed_volume(X, T) for x in part["volumes"] if x["region"] == region for T in vols[x["tag"]])
        return part["regions"][region]["volume_m3"] - Vt

    for region in ("coil_top", "pole_top", "air"):
        assert abs(deficit(fine, region)) < abs(deficit(coarse, region)), region
    assert abs(deficit(fine, "tracker_up")) <= 1e-12 * fine["regions"]["tracker_up"]["volume_m3"]
    assert sum(len(f["triangles"]) for f in fine["triangulation"]["faces"]) > \
        sum(len(f["triangles"]) for f in coarse["triangulation"]["faces"])


def test_the_3d_export_refuses_what_is_not_a_device(mp, tmp_path):  # noqa: F811
    """A model reduced by symmetry planes is refused (the partition would be part of the device); the construction's
    own refusals reach the info file as in the EM mode. Exit status 1, ``ok`` false, no mesh."""
    inp = R3.mesher_input_3d(R3.design3("dipole"))
    sym = copy.deepcopy(inp)
    sym.update(symmetry=[{"name": "mirror", "axis": "x", "side": 1.0, "type": "pec"}], infinity_shell=0)
    out = copy.deepcopy(inp)
    out["regions"].append({"name": "far", "attr": 99, "solid": R3._box((1.05, -0.05, -0.05), (1.15, 0.05, 0.05))})
    for key, bad, words in (("symmetry", sym, ("symmetry",)), ("shell", out, ("infinity shell", "domain"))):
        r, info, msh = _export(mp, bad, tmp_path / key)
        assert r.returncode == 1, (key, r.returncode, r.stderr[-1000:])
        assert info is not None and info.get("ok") is False and isinstance(info.get("error"), str), (key, info)
        assert any(w in info["error"] for w in words), (key, info["error"])
        assert not msh.exists(), key


def _manifold_shells(part, vtag):
    """The shells of a volume as oriented triangle arrays, each asserted manifold: every edge used exactly once in each
    direction within the shell (so two cavities that meet along an edge are two shells, not one)."""
    _, _, vols = R3.shells_of(part)
    out = []
    for k, T in enumerate(vols[vtag]):
        e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        uniq, cnt = np.unique(e, axis=0, return_counts=True)
        assert cnt.max() == 1, f"volume {vtag}, shell {k}: a directed edge is used {cnt.max()} times"
        rev = {tuple(x) for x in uniq.tolist()}
        assert all((b, a) in rev for a, b in uniq.tolist()), f"volume {vtag}, shell {k}: an edge is not paired"
        out.append(T)
    return out


def test_volumes_that_touch_along_an_edge_export(mp, tmp_path):  # noqa: F811
    """Two boxes that share one edge: the export succeeds (as the EM mesh of the same input does); the air has three
    shells, the domain sphere and one cavity per box, each a closed manifold surface (shells are faces joined across
    the edges they pair, not across every shared curve); the conversion accepts it (touching along an edge is not an
    overlap)."""
    from axionmultiphysics import solids
    inp = R3.mesher_input_3d(R3.edge_touch_design())
    em_r, em_info, _ = R.run_mesher(mp, inp, tmp_path / "em", "em")
    assert em_r.returncode == 0 and em_info["ok"] is True, "the EM mesh of the input fails (the test's premise)"
    r, part, msh = _export(mp, inp, tmp_path / "x")
    assert r.returncode == 0 and part and part.get("ok") is True, (r.returncode, (part or {}).get("error"))
    assert not msh.exists()
    face_region = {}
    for e in part["entities"]:
        for sd in e["side"]:
            face_region.setdefault(e["tag"], set()).add(sd)
    for v in part["volumes"]:
        shells = _manifold_shells(part, v["tag"])
        if v["region"] == "air":
            assert len(shells) == 3, [len(sh) for sh in v["shells"]]
            cavities = [{r for f in sh for r in face_region[abs(f)]} - {"air"} for sh in v["shells"][1:]]
            assert sorted(map(sorted, cavities)) == [["a"], ["b"]], cavities
        else:
            assert len(shells) == 1, (v["region"], v["shells"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = solids.from_partition(part, R3.THIN_MATERIALS, {"a": "iron", "b": "iron", "air": "air"})
    by = {s["name"]: s for s in doc["solids"]}
    assert by["a_0"]["mother"] == by["b_0"]["mother"] == "air_0", {n: s["mother"] for n, s in by.items()}


def test_a_domain_without_regions_exports_the_air(mp, tmp_path):  # noqa: F811
    """An input without regions is a valid partition: the air of the domain sphere, one volume with one shell, every
    face between the air and the exterior; it converts to one solid in the world."""
    from axionmultiphysics import solids
    inp = R3.air_only_input(0.5)
    r, part, msh = _export(mp, inp, tmp_path / "x")
    assert r.returncode == 0 and part and part.get("ok") is True, (r.returncode, (part or {}).get("error"))
    assert set(part["regions"]) == {"air"} and part["regions"]["air"]["attr"] == R3.AIR_ATTR
    assert [v["region"] for v in part["volumes"]] == ["air"] and len(part["volumes"][0]["shells"]) == 1
    assert all(sorted(e["side"]) == ["air", "exterior"] for e in part["entities"])
    V = part["regions"]["air"]["volume_m3"]
    assert abs(V / (4.0 / 3.0 * math.pi * 0.5 ** 3) - 1.0) <= 1e-12, V
    _manifold_shells(part, part["volumes"][0]["tag"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = solids.from_partition(part, R3.THIN_MATERIALS, {"air": "air"})
    assert [s["name"] for s in doc["solids"]] == ["world", "air_0"] and doc["solids"][1]["mother"] == "world"


# ============================================================================================ EM meshing unchanged
@pytest.mark.parametrize("name", NAMES)
def test_em_meshes_3d_are_unchanged(runs, name):
    """``axmp-mesher INPUT`` without the flag writes the 3D EM mesh and info file of the mesher before the export
    existed, bit for bit (``seconds`` aside): the curved (order 2) meshes of the two designs, whose curved steps depend
    on the heap the construction leaves behind."""
    run = runs[name]
    assert run["em_r"].returncode == 0, run["em_r"].stderr[-1500:]
    rec = R3.records3()["em"][name]
    assert R3.mesher_input_3d(R3.design3(name)) == rec["input"], "the design or its mesher input changed"
    assert R.sha256(run["em_msh"]) == rec["mesh_sha256"]
    diffs = R.same_json(R3.strip_seconds(run["em_info"]), rec["info"])
    assert diffs == [], diffs[:10]


# ============================================================================================ helpers
def _whole_surface_extent(s, nodes, tol):
    """The exact box of a face that is a whole sphere or torus (its nodes reach every extreme of the surface within a
    quarter of the radius), else None."""
    if s.kind == "sphere":
        lo, hi = s.c - s.r, s.c + s.r
    elif s.kind == "torus" and abs(abs(s.a[2]) - 1.0) < 1e-12:
        lo = s.c - np.array([s.R + s.r, s.R + s.r, s.r])
        hi = s.c + np.array([s.R + s.r, s.R + s.r, s.r])
    else:
        return None
    if (nodes.min(axis=0) - lo > 0.25 * s.r).any() or (hi - nodes.max(axis=0) > 0.25 * s.r).any():
        return None
    return lo, hi


def _boundary_length(X, T):
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, cnt = np.unique(key, axis=0, return_counts=True)
    b = uniq[cnt == 1]
    return float(np.linalg.norm(X[b[:, 0]] - X[b[:, 1]], axis=1).sum())


def _surface_normal(s, p):
    """The unit normal of the exact surface at p (either sense)."""
    if s.kind == "plane":
        return s.n
    if s.kind == "sphere":
        d = p - s.c
    elif s.kind == "cylinder":
        d = p - s.b
        d = d - (d @ s.a) * s.a
    elif s.kind == "torus":
        d = p - s.c
        h = d @ s.a
        q = d - h * s.a
        ring = s.c + q / np.linalg.norm(q) * s.R
        d = p - ring
    elif s.kind == "cone":
        d = p - s.b
        sa = d @ s.a
        radial = d - sa * s.a
        u = radial / np.linalg.norm(radial)
        t = np.array([s.r2 - s.r1, s.h])
        t /= np.linalg.norm(t)
        d = t[1] * u - t[0] * s.a
    else:
        raise ValueError(s.kind)
    return d / np.linalg.norm(d)


def _point_triangle_distance(p, X, T):
    """The distance from p to each triangle of T (exact: the plane when the projection is inside, else the edges)."""
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    n = n / nn[:, None]
    dist_plane = np.einsum("ij,ij->i", p - a, n)
    q = p - dist_plane[:, None] * n
    inside = np.ones(len(T), bool)
    for u, w in ((a, b), (b, c), (c, a)):
        inside &= np.einsum("ij,ij->i", np.cross(w - u, q - u), n) >= 0
    out = np.abs(dist_plane)

    def seg(u, w):
        d = w - u
        t = np.clip(np.einsum("ij,ij->i", p - u, d) / np.einsum("ij,ij->i", d, d), 0, 1)
        return np.linalg.norm(u + t[:, None] * d - p, axis=1)

    edge = np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return np.where(inside, out, edge)
