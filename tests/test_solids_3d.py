"""Solids of a 3D partition: ``solids.from_partition`` on the mesher's 3D export (``geometry`` "3d") and on hand-built
3D partitions, ``solids.write`` and ``solids.write_gdml``.

Each volume of the partition becomes one tessellated solid, the triangles of its outer shell turned outwards, copied
bit for bit from the partition's node list (so that the solids on the two sides of a face have the same facets); the
volumes in its cavities (its inner shells) become its daughters; a volume in no cavity is a daughter of the world, a
vacuum solid. The infinity shell has no solid (the mesher does not export it).

Independent references: the partition itself (its shells, signed volumes and region volumes), a ray-casting test of
which outer shell contains a point inside each volume (the mother is the smallest such volume), the hand-built
partitions' closed forms, and Python's XML parser for the GDML file.

Tolerances: the report's ``volume_solids_m3`` against the partition's triangulated volume 1e-12 relative (the same
triangles, summed in another order); vertices and triangles bit for bit.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import json
import os
import random
import shutil
import subprocess
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import _partition3d_ref as R3
import _partition_ref as R
from _partition_ref import mp  # noqa: F401 (a fixture)

from axionmultiphysics import solids

NAMES = list(R3.DESIGNS3)


def _convert(part, materials, region_materials):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        doc = solids.from_partition(part, materials, region_materials)
    return doc, [x for x in w if issubclass(x.category, solids.SolidsWarning)]


@pytest.fixture(scope="module")
def exports(mp, tmp_path_factory):  # noqa: F811
    out = {}
    for name in NAMES:
        d = tmp_path_factory.mktemp(f"s3_{name}")
        inp = R3.mesher_input_3d(R3.design3(name))
        inp["chord_tol_m"] = R3.TEST_CHORD
        r, part, _ = R.run_mesher(mp, inp, d, "x", export=True)
        assert r.returncode == 0 and part and part.get("ok") is True, (
            f"the 3D export failed: {(part or {}).get('error')!r} {r.stderr[-1500:]}")
        doc, warns = _convert(part, R3.solids_materials3(name), R3.region_materials3(R3.design3(name)))
        out[name] = (part, doc, warns)
    return out


# ============================================================================================ references
_RAY = np.array([0.5773502691896258, 0.5773502691896257, 0.5773502691896259]) + np.array([1e-3, -2e-3, 3e-4])
_RAY = _RAY / np.linalg.norm(_RAY)


def _inside(X, T, p) -> bool:
    """Whether p lies inside the closed triangulated surface (ray casting along a fixed irrational direction)."""
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    e1, e2 = b - a, c - a
    h = np.cross(_RAY, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-300
    f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p - a
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ _RAY)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return bool(np.count_nonzero(hit) % 2)


def _point_inside(X, T):
    """A point just inside the closed outward surface T: off the centroid of its largest triangle, inwards."""
    areas = R3.triangle_areas(X, T)
    k = int(np.argmax(areas))
    a, b, c = X[T[k]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n)
    return (a + b + c) / 3.0 - 1e-6 * np.sqrt(areas[k]) * n


def _canonical(tri_xyz):
    """A triangle given by its three vertex coordinates (tuples of float.hex), rotated to start at its smallest vertex
    (orientation kept)."""
    k = min(range(3), key=lambda i: tri_xyz[i])
    return tuple(tri_xyz[k:] + tri_xyz[:k])


def _facets(X, T):
    return sorted(_canonical([tuple(float(v).hex() for v in X[i]) for i in t]) for t in T.tolist())


def _expected_tree(part):
    """{volume tag: mother volume tag or None (the world)} by geometry alone: the smallest outer shell (by volume) other
    than the volume's own that contains a point inside the volume."""
    X, _, vols = R3.shells_of(part)
    outer = {t: shells[0] for t, shells in vols.items()}
    size = {t: R3.signed_volume(X, T) for t, T in outer.items()}
    out = {}
    for t, T in outer.items():
        p = _point_inside(X, T)
        cands = [u for u in outer if u != t and size[u] > size[t] and _inside(X, outer[u], p)]
        out[t] = min(cands, key=lambda u: size[u]) if cands else None
    return out


def _solid_of_volume(part, doc):
    """{volume tag: solid} matched by facets: each tessellated solid's facets are exactly one volume's outer shell."""
    X, _, vols = R3.shells_of(part)
    by_facets = {tuple(_facets(X, shells[0])): t for t, shells in vols.items()}
    out = {}
    for s in doc["solids"][1:]:
        V = np.array(s["params"]["vertices_m"], float)
        T = np.array(s["params"]["triangles"], np.int64)
        key = tuple(_facets(V, T))
        assert key in by_facets, f"solid {s['name']}: its facets are no volume's outer shell"
        out[by_facets[key]] = s
    return out


# ============================================================================================ hand-built partitions
def _box_faces(lo, hi):
    """The six faces of a box as quads (4 corners each), counter-clockwise seen from outside."""
    x = [lo, hi]
    faces = []
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        for side in (0, 1):
            q = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = [0.0, 0.0, 0.0]
                p[k], p[i], p[j] = x[side][k], x[a][i], x[b][j]
                q.append(p)
            n = np.cross(np.subtract(q[1], q[0]), np.subtract(q[2], q[0]))
            if (n[k] > 0) != (side == 1):
                q = q[::-1]
            faces.append(q)
    return faces


def cube_partition():
    """A unit cube "a" in the air of a cube [-1, 2]^3, as the mesher would write it: two triangles per face, the outer
    shell's faces between air and the exterior."""
    nodes, index = [], {}

    def node(p):
        key = tuple(p)
        if key not in index:
            index[key] = len(nodes)
            nodes.append(list(p))
        return index[key]

    entities, tfaces, tag = [], [], 0
    shells = {"a": [], "air_outer": []}
    for (lo, hi, sides, key) in (((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), ["a", "air"], "a"),
                                 ((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), ["air", "exterior"], "air_outer")):
        for q in _box_faces(lo, hi):
            tag += 1
            ids = [node(p) for p in q]
            qa = np.array(q)
            n = np.cross(qa[1] - qa[0], qa[2] - qa[0])
            n = n / np.linalg.norm(n)
            entities.append({"tag": tag, "dim": 2, "side": sides, "curves": [], "axis": False, "type": "plane",
                             "lo": qa.min(axis=0).tolist(), "hi": qa.max(axis=0).tolist(),
                             "point": qa.mean(axis=0).tolist(), "normal": n.tolist(),
                             "area_m2": float(np.linalg.norm(np.cross(qa[1] - qa[0], qa[3] - qa[0])))})
            tfaces.append({"tag": tag, "triangles": [[ids[0], ids[1], ids[2]], [ids[0], ids[2], ids[3]]],
                           "sagitta_m": 0.0})
            shells[key].append(tag)
    diag = float(np.sqrt(27.0))
    return {"ok": True, "export_regions": True, "geometry": "3d", "gmsh_version": "4.15.2", "domain_radius": 1.0,
            "domain_center": [0.0, 0.0, 0.0], "diagonal_m": diag, "tolerance_m": 1e-9 * diag, "chord_tol_m": 1e-3,
            "regions": {"air": {"attr": 1, "volume_m3": 26.0}, "a": {"attr": 2, "volume_m3": 1.0}},
            "entities": entities,
            "volumes": [{"tag": 1, "region": "a", "shells": [shells["a"]], "volume_m3": 1.0},
                        {"tag": 2, "region": "air", "shells": [shells["air_outer"], [-t for t in shells["a"]]],
                         "volume_m3": 26.0}],
            "triangulation": {"nodes": nodes, "faces": tfaces}, "seconds": 0.0}


CUBE_MATERIALS = {"m": {"density": 1000.0, "particle": {"nist": "G4_WATER"}},
                  "air": {"density": 1.205, "particle": {"nist": "G4_AIR"}}}
CUBE_REGIONS = {"a": "m", "air": "air"}


def test_a_hand_built_3d_partition_converts():
    part = cube_partition()
    doc, warns = _convert(part, CUBE_MATERIALS, CUBE_REGIONS)
    assert doc["schema"] == solids.SCHEMA and doc["geometry"] == "3d" and doc["chord_tol_m"] == 1e-3
    assert not warns
    world = doc["solids"][0]
    assert world["name"] == doc["world"] and world["mother"] is None and world["region"] is None
    assert world["material"] == solids.VACUUM
    by = {s["name"]: s for s in doc["solids"]}
    assert set(by) == {doc["world"], "air_0", "a_0"}
    assert by["air_0"]["mother"] == doc["world"] and by["a_0"]["mother"] == "air_0"
    for n in ("air_0", "a_0"):
        s = by[n]
        assert s["kind"] == "tessellated" and set(s["params"]) == {"vertices_m", "triangles"}, s
        assert s["position_m"] == [0.0, 0.0, 0.0] and s["rotation_deg"] == [0.0, 0.0, 0.0]
        assert s["discretized"] is True and s["sagitta_m"] == 0.0
        V = np.array(s["params"]["vertices_m"], float)
        T = np.array(s["params"]["triangles"], np.int64)
        assert len(T) == 12 and R3.unbalanced_edges(T) == 0 and R3.signed_volume(V, T) > 0
    assert R3.signed_volume(np.array(by["a_0"]["params"]["vertices_m"]), np.array(by["a_0"]["params"]["triangles"])) == 1.0
    rep = doc["regions"]
    assert list(rep) == ["air", "a"]                       # document (attr) order
    assert rep["a"]["volume_m3"] == 1.0 and rep["a"]["volume_solids_m3"] == 1.0 and rep["a"]["volume_rel_diff"] == 0.0
    assert rep["air"]["volume_m3"] == 26.0 and rep["air"]["volume_solids_m3"] == pytest.approx(26.0, rel=1e-15)
    assert rep["a"]["mass_kg"] == 1000.0 and rep["a"]["sagitta_max_m"] == 0.0 and rep["a"]["chord_tol_m"] == 1e-3
    assert rep["air"]["solids"] == ["air_0"] and rep["a"]["solids"] == ["a_0"]


def _bad(mutate):
    part = cube_partition()
    mutate(part)
    return part


def _drop_triangle(p):
    p["triangulation"]["faces"][0]["triangles"].pop()


def _flip_face(p):
    f = p["triangulation"]["faces"][0]
    f["triangles"] = [t[::-1] for t in f["triangles"]]


def _flip_volume(p):
    for v in p["volumes"]:
        if v["region"] == "a":
            v["shells"] = [[-t for t in sh] for sh in v["shells"]]


def _wrong_sign(p):
    p["volumes"][1]["shells"][1][0] *= -1


INVALID = {
    "no_triangulation": lambda p: p.pop("triangulation"),
    "no_volumes": lambda p: p.pop("volumes"),
    "shell_not_closed": _drop_triangle,                    # a face's triangles leave a hole
    "face_turned": _flip_face,                             # one face's triangles turned: its edges no longer close
    "volume_inside_out": _flip_volume,                     # the outer shell encloses a negative volume
    "sign_against_sides": _wrong_sign,                     # the air uses a face of 'a' with the sign of side[0]
    "unknown_face": lambda p: p["volumes"][0]["shells"][0].append(99),
    "unknown_region": lambda p: p["volumes"][0].update(region="nowhere"),
    "index_out_of_range": lambda p: p["triangulation"]["faces"][0]["triangles"][0].__setitem__(0, 10 ** 6),
}


@pytest.mark.parametrize("case", list(INVALID))
def test_a_3d_partition_that_does_not_describe_closed_volumes_is_refused(case):
    _convert(cube_partition(), CUBE_MATERIALS, CUBE_REGIONS)        # the partition without the defect converts
    part = _bad(INVALID[case])
    with pytest.raises(Exception) as ei:
        _convert(part, CUBE_MATERIALS, CUBE_REGIONS)
    assert getattr(ei.value, "code", None) == "particle.partition_invalid", (case, ei.value)
    _assert_3d_hint(ei.value)


def _assert_3d_hint(e):
    """A refusal of a 3D partition hints at the 3D export, not at the axisymmetric (bounded-mode) table."""
    hint = getattr(e, "hint", "") or ""
    assert hint and "3D" in hint, hint
    assert "axisymmetric" not in hint and "bounded" not in hint, hint


def test_a_3d_partition_takes_no_chord_tolerance():
    """The triangulation of a 3D partition is the mesher's: the conversion's ``chord_tol_m`` is for axisymmetric
    partitions (``Design.export_gdml`` passes it to the mesher for a 3D design)."""
    with pytest.raises(ValueError, match="chord_tol_m"):
        solids.from_partition(cube_partition(), CUBE_MATERIALS, CUBE_REGIONS, chord_tol_m=1e-4)


def test_the_solids_of_a_3d_partition_depend_on_its_geometry_only():
    """Not on the order of the entities, the volumes, the triangulation's faces or a volume's inner shells."""
    part = cube_partition()
    doc, _ = _convert(part, CUBE_MATERIALS, CUBE_REGIONS)
    rnd = random.Random(7)
    for _ in range(3):
        p = copy.deepcopy(part)
        rnd.shuffle(p["entities"])
        rnd.shuffle(p["volumes"])
        rnd.shuffle(p["triangulation"]["faces"])
        for v in p["volumes"]:
            for sh in v["shells"]:
                rnd.shuffle(sh)
        assert _convert(p, CUBE_MATERIALS, CUBE_REGIONS)[0] == doc


def test_a_3d_partition_needs_compositions_like_an_axisymmetric_one():
    _convert(cube_partition(), CUBE_MATERIALS, CUBE_REGIONS)
    with pytest.raises(Exception) as ei:
        _convert(cube_partition(), {"m": CUBE_MATERIALS["m"]}, CUBE_REGIONS)
    assert getattr(ei.value, "code", None) == "particle.composition_missing"


# ============================================================================================ exported partitions
@pytest.mark.parametrize("name", NAMES)
def test_each_volume_becomes_a_tessellated_solid_of_its_outer_shell(exports, name):
    part, doc, _ = exports[name]
    assert doc["geometry"] == "3d" and doc["chord_tol_m"] == part["chord_tol_m"]
    assert doc["units"]["length"] == "m"
    world = doc["solids"][0]
    assert world["mother"] is None and world["region"] is None and world["material"] == solids.VACUUM
    assert len(doc["solids"]) == 1 + len(part["volumes"]), "one solid per volume, and the world"
    rm = R3.region_materials3(R3.design3(name))
    match = _solid_of_volume(part, doc)
    assert set(match) == {v["tag"] for v in part["volumes"]}
    X, tri, vols = R3.shells_of(part)
    sag = {f["tag"]: f["sagitta_m"] for f in part["triangulation"]["faces"]}
    for v in part["volumes"]:
        s = match[v["tag"]]
        assert s["kind"] == "tessellated" and set(s["params"]) == {"vertices_m", "triangles"}
        assert s["region"] == v["region"] and s["material"] == rm[v["region"]]
        assert s["name"].startswith(v["region"] + "_") and s["name"][len(v["region"]) + 1:].isdigit()
        assert s["position_m"] == [0.0, 0.0, 0.0] and s["rotation_deg"] == [0.0, 0.0, 0.0]
        assert s["discretized"] is True
        assert s["sagitta_m"] == max(sag[abs(f)] for f in v["shells"][0])
        V = np.array(s["params"]["vertices_m"], float)
        T = np.array(s["params"]["triangles"], np.int64)
        assert np.array_equal(np.unique(T), np.arange(len(V))), f"{s['name']}: a vertex is not used"
        assert R3.unbalanced_edges(T) == 0 and R3.signed_volume(V, T) > 0
        # the vertices are the partition's nodes, bit for bit
        known = {tuple(float(x).hex() for x in row) for row in X}
        assert all(tuple(float(x).hex() for x in row) in known for row in V), s["name"]
    names = [s["name"] for s in doc["solids"]]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name", NAMES)
def test_a_face_has_the_same_facets_in_the_solids_on_both_its_sides(exports, name):
    """Where two solids touch (siblings across a face of both their outer shells), their facets on that face are the
    same triangles, the same coordinates bit for bit, in opposite orientations."""
    part, doc, _ = exports[name]
    X, tri, _ = R3.shells_of(part)
    match = _solid_of_volume(part, doc)
    facets = {t: set(_facets(np.array(s["params"]["vertices_m"], float), np.array(s["params"]["triangles"])))
              for t, s in match.items()}
    outer_use = {}
    for v in part["volumes"]:
        for f in v["shells"][0]:
            outer_use.setdefault(abs(f), []).append((f > 0, v["tag"]))
    shared = 0
    for f, uses in outer_use.items():
        if len(uses) != 2:
            continue
        shared += 1
        (s1, v1), (s2, v2) = uses
        T = tri[f]
        mine = set(_facets(X, T if s1 else T[:, ::-1]))
        theirs = set(_facets(X, T if s2 else T[:, ::-1]))
        assert mine <= facets[v1] and theirs <= facets[v2], f
    assert shared > 0, "the design has no face between two outer shells (the test's premise)"


@pytest.mark.parametrize("name", NAMES)
def test_the_mother_of_a_solid_is_the_volume_whose_cavity_holds_it(exports, name):
    part, doc, _ = exports[name]
    match = _solid_of_volume(part, doc)
    want = _expected_tree(part)
    for t, m in want.items():
        assert match[t]["mother"] == (doc["world"] if m is None else match[m]["name"]), (match[t]["name"], m)
    if name == "shapes":
        by_region = {}
        for t, s in match.items():
            by_region.setdefault(s["region"], []).append(s)
        assert len(by_region["air"]) == 2 and len(by_region["rod"]) == 3, {k: len(v) for k, v in by_region.items()}
        pocket = [s for s in by_region["air"] if s["mother"] != doc["world"]]
        assert len(pocket) == 1 and pocket[0]["mother"] == by_region["can"][0]["name"]
        assert by_region["ball"][0]["mother"] == pocket[0]["name"]
    # every solid after its mother
    seen = set()
    for s in doc["solids"]:
        assert s["mother"] is None or s["mother"] in seen, s["name"]
        seen.add(s["name"])


@pytest.mark.parametrize("name", NAMES)
def test_the_volume_report_of_a_3d_partition(exports, name):
    part, doc, warns = exports[name]
    X, tri, vols = R3.shells_of(part)
    sag = {f["tag"]: f["sagitta_m"] for f in part["triangulation"]["faces"]}
    dens = R3.DESIGNS3[name][2]
    rm = R3.region_materials3(R3.design3(name))
    by = {s["name"]: s for s in doc["solids"]}
    order = sorted(part["regions"], key=lambda r: part["regions"][r]["attr"])
    assert list(doc["regions"]) == order
    warned = set()
    for w in warns:
        assert str(w.message).startswith("particle.volume_deficit"), w.message
        warned.add(str(w.message).split("'")[1])
    for r, rep in doc["regions"].items():
        mine = [v for v in part["volumes"] if v["region"] == r]
        Vt = sum(R3.signed_volume(X, T) for v in mine for T in vols[v["tag"]])
        assert rep["material"] == rm[r] and rep["density_kg_m3"] == dens[rm[r]]
        assert rep["volume_m3"] == part["regions"][r]["volume_m3"]
        assert rep["volume_solids_m3"] == pytest.approx(Vt, rel=1e-12), r
        own = 0.0
        for n in rep["solids"]:
            s = by[n]
            V = R3.signed_volume(np.array(s["params"]["vertices_m"]), np.array(s["params"]["triangles"]))
            for d in doc["solids"]:
                if d["mother"] == n:
                    V -= R3.signed_volume(np.array(d["params"]["vertices_m"]), np.array(d["params"]["triangles"]))
            own += V
        assert own == pytest.approx(rep["volume_solids_m3"], rel=1e-12), r
        assert rep["volume_rel_diff"] == pytest.approx(rep["volume_solids_m3"] / rep["volume_m3"] - 1.0, abs=1e-15)
        assert rep["mass_kg"] == pytest.approx(rep["volume_m3"] * rep["density_kg_m3"], rel=1e-15)
        assert rep["mass_solids_kg"] == pytest.approx(rep["volume_solids_m3"] * rep["density_kg_m3"], rel=1e-15)
        assert rep["sagitta_max_m"] == max(sag[abs(f)] for v in mine for sh in v["shells"] for f in sh), r
        assert rep["chord_tol_m"] == part["chord_tol_m"]
        assert (abs(rep["volume_rel_diff"]) > solids.VOLUME_WARN) == (r in warned), (r, rep["volume_rel_diff"])
    assert sorted(w["region"] for w in doc["warnings"]) == sorted(warned)


@pytest.mark.parametrize("name", NAMES)
def test_gdml_of_a_3d_partition(exports, name, tmp_path):
    """One ``tessellated`` solid per tessellated solid of the list, its facets ``triangular`` with ``type="ABSOLUTE"``
    and vertices defined as positions in m: the coordinates read back bit for bit and the triangles in the list's
    order and orientation. The same bytes twice; valid against Geant4's schema when one is given."""
    _, doc, _ = exports[name]
    path = tmp_path / "x.gdml"
    solids.write_gdml(doc, path)
    solids.write_gdml(doc, tmp_path / "y.gdml")
    assert path.read_bytes() == (tmp_path / "y.gdml").read_bytes()
    root = ET.parse(path).getroot()
    pos = {}
    for p in root.find("define").iter("position"):
        assert p.get("unit") == "m", p.attrib
        pos[p.get("name")] = tuple(float(p.get(k)) for k in ("x", "y", "z"))
    tess = {t.get("name"): t for t in root.find("solids").iter("tessellated")}
    for s in doc["solids"]:
        if s["kind"] != "tessellated":
            continue
        t = tess[s["gdml_name"] + "_solid"]
        assert t.get("lunit") == "m", t.attrib
        facets = list(t)
        assert all(f.tag == "triangular" and f.get("type") == "ABSOLUTE" for f in facets)
        V = s["params"]["vertices_m"]
        got = [[pos[f.get(f"vertex{k}")] for k in (1, 2, 3)] for f in facets]
        want = [[tuple(V[i]) for i in tr] for tr in s["params"]["triangles"]]
        assert got == want, s["name"]
    vols = {v.get("name") for v in root.iter("volume")}
    assert vols == {s["gdml_name"] for s in doc["solids"]}
    xsd = os.environ.get("AXMP_GDML_SCHEMA")
    if xsd and shutil.which("xmllint"):
        r = subprocess.run(["xmllint", "--noout", "--schema", xsd, str(path)], capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.parametrize("name", NAMES)
def test_solids_json_of_a_3d_partition_reads_back_and_is_reproducible(exports, name, tmp_path):
    part, doc, _ = exports[name]
    solids.write(doc, tmp_path / "a.json")
    again, _ = _convert(part, R3.solids_materials3(name), R3.region_materials3(R3.design3(name)))
    solids.write(again, tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert json.loads((tmp_path / "a.json").read_text("utf-8")) == doc


# ============================================================================================ thin layers
def _export3(mp, doc, directory, chord=None):  # noqa: F811
    inp = R3.mesher_input_3d(doc)
    if chord is not None:
        inp["chord_tol_m"] = chord
    r, part, msh = R.run_mesher(mp, inp, directory, "x", export=True)
    assert r.returncode == 0 and part and part.get("ok") is True, (
        f"the 3D export failed: exit {r.returncode}, {(part or {}).get('error')!r}")
    assert not msh.exists()
    return part


def _region_materials(doc):
    out = {r["name"]: r["material"] for r in doc["regions"]}
    out["air"] = "air"
    return out


def _refused_thin(part, doc, names):
    """The conversion refuses a partition whose triangulated solids would not nest or would overlap: the code, a 3D hint
    that says to lower chord_tol_m, a message that names a region involved; nothing is returned."""
    with pytest.raises(Exception) as ei:
        _convert(part, R3.THIN_MATERIALS, _region_materials(doc))
    e = ei.value
    assert getattr(e, "code", None) == "particle.partition_invalid", e
    _assert_3d_hint(e)
    assert "chord_tol_m" in e.hint, e.hint
    assert any(n in str(e) for n in names), str(e)


def _nested(doc):
    """Every tessellated daughter's vertices lie inside its tessellated mother's surface (own ray casting)."""
    by = {s["name"]: s for s in doc["solids"]}
    pairs = 0
    for s in doc["solids"]:
        m = by.get(s["mother"]) if s["mother"] else None
        if m is None or m["kind"] != "tessellated" or s["kind"] != "tessellated":
            continue
        X = np.array(m["params"]["vertices_m"], float)
        T = np.array(m["params"]["triangles"], np.int64)
        P = np.array(s["params"]["vertices_m"], float)
        inside = R3.inside_closed_surface(X, T, P)
        assert inside.all(), f"{np.count_nonzero(~inside)} of {len(P)} vertices of {s['name']} lie outside {m['name']}"
        pairs += 1
    return pairs


def test_a_thin_spherical_shell_is_refused_at_the_default_chord_tolerance(mp, tmp_path):  # noqa: F811
    """An iron shell 0.1 mm thick, default chord tolerance (3.5e-4 m): the mesher exports the partition, but the
    triangulated cavity would poke through the shell, so the conversion refuses it."""
    doc = R3.thin_sphere_design(1e-4, 1.0)
    part = _export3(mp, doc, tmp_path)
    assert part["chord_tol_m"] > 1e-4
    _refused_thin(part, doc, ("shell",))


def test_a_thin_spherical_shell_converts_at_a_chord_tolerance_below_its_thickness(mp, tmp_path):  # noqa: F811
    doc = R3.thin_sphere_design(1e-4, 0.15)
    part = _export3(mp, doc, tmp_path, chord=4e-5)
    doc_s, _ = _convert(part, R3.THIN_MATERIALS, _region_materials(doc))
    assert _nested(doc_s) == 2, "the cavity's air in the shell, the shell in the air"


def test_a_rod_in_a_tube_with_a_gap_below_the_chord_tolerance_is_refused(mp, tmp_path):  # noqa: F811
    """Siblings 0.2 mm apart, chord tolerance 1 mm: the chords of the tube's bore cut into the rod."""
    doc = R3.rod_in_tube_design(2e-4)
    part = _export3(mp, doc, tmp_path, chord=1e-3)
    _refused_thin(part, doc, ("rod", "tube"))


def test_a_thin_cylindrical_shell_exports_and_its_conversion_is_refused(mp, tmp_path):  # noqa: F811
    """An open iron cylinder 0.2 mm thick at the default chord tolerance: the partition is exported (its sides are
    topological, the triangulation is too coarse for this wall only), and the conversion refuses it, because the
    chords of its outer surface cross its bore."""
    doc = R3.thin_cylinder_design(2e-4, 1.0)
    part = _export3(mp, doc, tmp_path)
    assert {v["region"] for v in part["volumes"]} == {"air", "shield", "rod"}
    _refused_thin(part, doc, ("shield",))


def test_a_thin_cylindrical_shell_converts_at_a_fine_chord_tolerance(mp, tmp_path):  # noqa: F811
    doc = R3.thin_cylinder_design(2e-4, 0.15)
    part = _export3(mp, doc, tmp_path, chord=4e-5)
    doc_s, _ = _convert(part, R3.THIN_MATERIALS, _region_materials(doc))
    assert _nested(doc_s) >= 2
