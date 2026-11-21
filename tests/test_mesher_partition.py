"""The mesher's partition: face loops and exact curve parameters in the entity table, and ``--export-regions``.

The contract tested (``docs/guides/solids.md``, "The partition", documents the keys; ``solids.from_partition`` reads
them):

- In every 2D mode that writes an entity table (the bounded planar and axisymmetric modes, meshed or
  ``resolve_only``, and the export mode) each entity carries ``start`` and ``end`` (its end vertices, ``start`` first
  along ``tangent``, a common vertex bit for bit in every entity that has it, r exactly 0 on the axis, ``start ==
  end`` for a closed curve), and each circle or ellipse ``center``, ``radii`` [a, b] (a >= b, equal for a circle),
  ``major_axis`` and ``ccw``: the curve c + a cos t u + b sin t v (v = u turned by +90 degrees) from ``start`` to
  ``end``, t increasing iff ``ccw``. The table has ``faces``: every surface with its region and its loops of signed
  entity tags, the outer loop first, each keeping its face on its left.
- Outside these keys a bounded info file and its mesh are unchanged: equal to the records of the mesher before the
  keys existed (``tests/data/mesher_partition/reference_records.json``, made with its sequential build), and so is
  every EM mesh.
- ``axmp-mesher --export-regions INPUT`` on the input of the axisymmetric EM mode (``axmp mesh``'s ``mesh_in.json``)
  writes the partition to the input's ``info`` path and no mesh: ``ok``, ``export_regions``, ``geometry``,
  ``gmsh_version``, ``domain_radius``, ``domain_center_z``, ``diagonal_m``, ``tolerance_m``, ``regions`` (``air``
  and the design's regions, with ``attr``, ``area_m2``, ``volume_m3``), ``entities`` (no ``atom``), ``faces``,
  ``seconds``. Its tags are those of the mesh of the same input. A contour's elliptical arc (a B-spline in gmsh) is an
  ``ellipse`` with the contour's own centre and axes.

Independent references: gmsh's own points on its curves, read from the mesh of the same input (the interior nodes of
every edge, P2 mid-edge nodes included, are the curve evaluated at a parameter); the triangles of each face (their
physical and elementary tags, their boundary edges, their measures by a quadrature exact for P2 maps); the curves
evaluated from the table's parameters alone (``_partition_ref.Curve``: no gmsh and no ``solids``); OpenCASCADE's region
areas and Pappus volumes already in the info file (tested as exact for lines and arcs); the reference records.

Tolerances:
- interior mesh nodes and ``point`` on the described curve: 1e-12 x ``diagonal_m``. gmsh evaluates its curve (rounding,
  about 1e-16 of the size) and prints 16 significant digits (5e-17 relative), so the noise is near 1e-16; a radius or
  centre wrong by more than 1e-12 of the size fails;
- end vertices: ``tolerance_m`` (1e-9 x ``diagonal_m``), the mesher's own scale for "on the axis": OpenCASCADE computes
  some vertices by intersection and holds them only to its tolerances;
- positions along a curve: 1e-9 at the ends of the parameter range (the node is the vertex printed with 16 digits);
  joints between two edges exactly (one node, one computation);
- box against the parameters 1e-12 x ``diagonal_m``; length 1e-9 relative (OpenCASCADE's length is exact to rounding
  on lines and arcs, but held to its integration tolerance on a contour's B-spline: 2.6e-10 measured on the
  nose's half ellipse; the quadrature here is good to about 1e-14, and the points test the parameters far more
  tightly);
- region areas and volumes from the loops against the info file's: 1e-10 relative (OpenCASCADE's face mass is exact
  to rounding on lines and arcs; on a face bounded by a contour's B-spline it is not, 2.4e-3 measured on the
  nose, so the export must compute those measures from the curves);
- a face's or region's mesh measure against the exact one: 1e-12 relative plus 10 times the sum, over the boundary
  edges, of the gap between the P2 edge and the circle through its three nodes (times 2 pi r for volumes). That gap is
  the P2 boundary's error for a circle (O(h^5) per edge); for an ellipse the curve's departure from that circle over
  one edge is of the same order, hence the factor 10. Straight edges have no gap, so straight regions compare at
  1e-12.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import _partition_ref as R
from _partition_ref import mp  # noqa: F401 (a fixture)

BOUNDED = R.bounded_fixtures()
REFUSALS = R.refusal_fixtures()
DESIGNS = list(R.DESIGNS)
AXI_BOUNDED = [n for n in BOUNDED if R.fixture(n)["geometry"] == "axisymmetric"]
CASES = [f"bounded:{n}" for n in BOUNDED] + [f"em:{n}" for n in DESIGNS]
MODES = ("meshed", "resolve_only")
NODE_REL = 1e-12
S_END = 1e-9
GAP_FACTOR = 10.0


# ============================================================================================ the runs
_CACHE: dict = {}


def run_case(mp: str, case: str, base: Path):
    """(table, mesh, extra) of a case, computed once per module: a bounded fixture meshed (its info file is the table),
    or an EM design meshed by ``axmp mesh``'s code and exported from the same input. A failure is raised again at
    every use, so that each test fails for it."""
    if case not in _CACHE:
        try:
            _CACHE[case] = ("ok", _run_case(mp, case, base / case.replace(":", "_")))
        except Exception as e:  # noqa: BLE001 (re-raised in the tests)
            _CACHE[case] = ("error", e)
    status, value = _CACHE[case]
    if status == "error":
        raise value
    return value


def _run_case(mp: str, case: str, d: Path):
    kind, name = case.split(":")
    if kind == "bounded":
        inp = R.fixture(name)
        inp["resolve_only"] = False
        r, info, msh = R.run_mesher(mp, inp, d)
        assert r.returncode == 0 and info and info.get("ok") is True, f"{name}: {r.returncode} {r.stderr[-500:]}"
        return info, R.Msh(msh), {"input": inp}
    doc = R.design(name)
    inp, msh, _ = R.em_mesh(doc, d / "mesh")
    r, part, out = R.run_mesher(mp, inp, d / "export", "regions", export=True)
    assert r.returncode == 0, (f"{name}: axmp-mesher --export-regions exited with {r.returncode} (the export mode "
                               f"is missing?): {r.stderr[-500:]}")
    assert part is not None and part.get("ok") is True, f"{name}: {part}"
    return part, R.Msh(msh), {"input": inp, "unused_output": out, "doc": doc}


def _has_keys(table: dict) -> None:
    """The partition keys are there (the message of every test of this file on a mesher without them)."""
    assert "faces" in table, "the table has no 'faces' (the mesher does not write the partition)"
    for e in table["entities"]:
        assert "start" in e and "end" in e, f"entity {e['tag']} has no 'start'/'end'"


# ============================================================================================ (1) bounded: unchanged + keys
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", BOUNDED)
def test_bounded_info_files_are_unchanged_but_for_the_partition_keys(mp, tmp_path, name, mode):
    """Every accepted bounded fixture, meshed and ``resolve_only``: the info file has the partition keys, and without
    them (and ``seconds``) it is the reference record, key order and every number bit for bit; the mesh file is the
    record's byte for byte; ``resolve_only`` writes no mesh."""
    rec = R.records()["bounded"][name][mode]
    inp = R.fixture(name)
    inp["resolve_only"] = mode == "resolve_only"
    r, info, msh = R.run_mesher(mp, inp, tmp_path)
    assert r.returncode == 0 and info is not None and info.get("ok") is True, r.stderr[-1000:]
    _has_keys(info)
    for e in info["entities"]:
        if e["type"] in ("circle", "ellipse"):
            assert all(k in e for k in R.ARC_KEYS), f"arc entity {e['tag']} lacks {set(R.ARC_KEYS) - set(e)}"
        else:
            assert not any(k in e for k in R.ARC_KEYS), f"line entity {e['tag']} has arc keys"
    diff = R.same_json(R.strip_added(info), rec["info"])
    assert not diff, f"{name} ({mode}) differs from the reference info file outside the added keys:\n" + "\n".join(diff[:20])
    if mode == "meshed":
        assert R.sha256(msh) == rec["msh_sha256"], f"{name}: the mesh file differs from the reference"
    else:
        assert not msh.exists(), "resolve_only wrote a mesh"


@pytest.mark.parametrize("name", REFUSALS)
def test_bounded_refusals_are_unchanged(mp, tmp_path, name):
    rec = R.records()["refusals"][name]
    r, info, msh = R.run_mesher(mp, R.fixture(name), tmp_path)
    assert r.returncode == rec["status"] == 1
    assert R.same_json(info, rec["info"]) == []
    assert msh.exists() is rec["msh"] is False


# ============================================================================================ EM meshes unchanged
@pytest.mark.parametrize("name", list(R.RECORDED) + ["plane_rect"])
def test_em_meshes_are_unchanged(mp, tmp_path, name):
    """``axmp-mesher INPUT`` without the flag, on EM inputs (three axisymmetric designs through ``axmp mesh``'s code,
    and the plane mode's fixture): the reference mesh byte for byte and info file (``seconds`` aside). G0 tier B makes
    the same comparison for G0's designs on the reference node."""
    recs = R.records()
    if name == "plane_rect":
        rec = recs["plane"][name]
        r, info, msh = R.run_mesher(mp, R.fixture(name), tmp_path)
        assert r.returncode == 0
    else:
        rec = recs["em"][name]
        _, msh, info_path = R.em_mesh(R.design(name), tmp_path)
        info = json.loads(info_path.read_text("utf-8"))
    assert R.same_json(R.without_seconds(info), rec["info"]) == []
    assert R.sha256(msh) == rec["msh_sha256"], f"{name}: the mesh differs from the reference"


# ============================================================================================ (2) the keys on real meshes
def _vec3(v, what):
    assert isinstance(v, list) and len(v) == 3 and all(isinstance(x, float) for x in v), f"{what}: {v!r}"
    assert v[2] == 0.0, f"{what}: third component {v[2]!r}"


@pytest.mark.parametrize("case", CASES)
def test_entities_carry_their_vertices_and_arcs_their_parameters(mp, tmp_path_factory, case):
    table, _, _ = run_case(mp, case, tmp_path_factory.getbasetemp())
    _has_keys(table)
    for e in table["entities"]:
        at = f"entity {e['tag']}"
        _vec3(e["start"], at + " start")
        _vec3(e["end"], at + " end")
        if e["type"] == "line":
            assert not any(k in e for k in R.ARC_KEYS), f"{at}: a line with arc keys"
            assert e["start"] != e["end"], f"{at}: a line with start == end"
            continue
        assert e["type"] in ("circle", "ellipse"), f"{at}: type {e['type']!r}"
        _vec3(e["center"], at + " center")
        _vec3(e["major_axis"], at + " major_axis")
        a, b = e["radii"]
        assert isinstance(a, float) and isinstance(b, float) and a >= b > 0.0, f"{at}: radii {e['radii']}"
        assert abs(math.hypot(*e["major_axis"][:2]) - 1.0) <= 1e-15, f"{at}: major_axis {e['major_axis']}"
        assert isinstance(e["ccw"], bool), f"{at}: ccw {e['ccw']!r}"
        if e["type"] == "circle" or abs(a - b) <= 1e-12 * a:
            assert a == b, f"{at}: a circular arc with radii {e['radii']} (one double written twice)"
            assert e["major_axis"] == [1.0, 0.0, 0.0], f"{at}: a circle's major axis {e['major_axis']}"


@pytest.mark.parametrize("case", CASES)
def test_vertices_are_shared_bit_for_bit_and_the_axis_is_exact(mp, tmp_path_factory, case):
    """No two different vertex coordinates lie within ``tolerance_m`` of each other (a common vertex is one value);
    in the axisymmetric geometry every vertex of an axis entity has r = 0.0 and no
    vertex has 0 < |r| <= ``tolerance_m``; consecutive traversals of every loop share their vertex bit for bit."""
    table, _, _ = run_case(mp, case, tmp_path_factory.getbasetemp())
    _has_keys(table)
    tol = table["tolerance_m"]
    pts = sorted({tuple(e[k][:2]) for e in table["entities"] for k in ("start", "end")})
    P = np.array(pts)
    for i in range(len(P)):
        d = np.hypot(*(P[i + 1:] - P[i]).T)
        close = np.nonzero(d <= tol)[0]
        assert close.size == 0, f"vertices {pts[i]} and {pts[i + 1 + close[0]]} differ by {d[close[0]]:.3g} m"
    if table["geometry"] == "axisymmetric":
        for e in table["entities"]:
            if e["axis"]:
                assert e["start"][0] == 0.0 and e["end"][0] == 0.0, f"axis entity {e['tag']}: {e['start']} {e['end']}"
        bad = [p for p in pts if 0.0 < abs(p[0]) <= tol]
        assert not bad, f"vertices within tolerance_m of the axis but not on it: {bad[:5]}"
    ent = {e["tag"]: e for e in table["entities"]}
    for f in table["faces"]:
        for loop in f["loops"]:
            ends = [(ent[abs(s)]["start"], ent[abs(s)]["end"])[:: 1 if s > 0 else -1] for s in loop]
            for k, (p, q) in enumerate(ends):
                nxt = ends[(k + 1) % len(ends)][0]
                assert q == nxt, f"face {f['tag']}: the loop {loop} does not close bit for bit after {loop[k]}"


def _samples(table: dict, msh: R.Msh, cs: dict) -> dict:
    """gmsh's points on each entity: {tag: [(a, m, b, face or None), ...]} with node ids, from the boundary edges of
    every face's triangles (assigned among that face's loop entities: the one whose curve holds the mid node, inside
    its arc) and from the line elements (by elementary tag)."""
    diag = table["diagonal_m"]
    loops_of = {f["tag"]: {abs(s) for l in f["loops"] for s in l} for f in table["faces"]}
    tris_by_face: dict = {}
    for t in msh.triangles():
        tris_by_face.setdefault(t[2], []).append(t)
    out: dict = {e["tag"]: [] for e in table["entities"]}
    for ftag, tris in tris_by_face.items():
        assert ftag in loops_of, f"the mesh has a face {ftag} that 'faces' does not list"
        for a, b, m, _ in R.boundary_edges(msh, tris):
            mid = msh.p(m) if m is not None else 0.5 * (msh.p(a) + msh.p(b))
            cand = [t for t in loops_of[ftag]
                    if cs[t].distance(mid) <= NODE_REL * diag and 0.0 < cs[t].s(mid) < 1.0]
            assert len(cand) == 1, (f"face {ftag}: the boundary edge {msh.p(a)}-{msh.p(b)} lies on "
                                    f"{len(cand)} of its loop entities {sorted(loops_of[ftag])}")
            out[cand[0]].append((a, m, b, ftag))
    for typ, _, etag, nodes in msh.lines():
        assert etag in out, f"the mesh has line elements on curve {etag}, which 'entities' does not list"
        out[etag].append((nodes[0], nodes[2] if typ == 8 else None, nodes[1], None))
    return out


def _case_samples(mp, case, base):
    table, msh, extra = run_case(mp, case, base)
    _has_keys(table)
    if "samples" not in extra:
        cs = R.curves(table)
        extra["curves"] = cs
        extra["samples"] = _samples(table, msh, cs)
    return table, msh, extra["curves"], extra["samples"]


@pytest.mark.parametrize("case", CASES)
def test_the_curve_parameters_describe_gmsh_s_curves(mp, tmp_path_factory, case):
    """For every entity: gmsh's interior points on it (the P2 mid nodes of its mesh edges) lie on the described curve
    within 1e-12 x ``diagonal_m`` and inside the arc from ``start`` to ``end`` in the sense ``ccw``; its end vertices
    within ``tolerance_m``; ``point`` on the curve inside the arc; ``tangent`` the direction from ``start`` to ``end``
    there; ``lo``/``hi`` the exact box of the arc; ``length_m`` its length. Every entity has mesh points."""
    table, msh, cs, samples = _case_samples(mp, case, tmp_path_factory.getbasetemp())
    diag, tol = table["diagonal_m"], table["tolerance_m"]
    for e in table["entities"]:
        c, at = cs[e["tag"]], f"entity {e['tag']} ({e['type']})"
        for k in ("start", "end"):
            assert c.distance(e[k][:2]) <= tol, f"{at}: {k} {e[k]} is {c.distance(e[k][:2]):.3g} m off its curve"
        assert c.distance(e["point"][:2]) <= NODE_REL * diag, f"{at}: point off the curve"
        sp = c.s(e["point"][:2])
        assert 0.0 < sp < 1.0, f"{at}: point at s = {sp} (outside the arc from start to end)"
        dot = float(np.dot(e["tangent"][:2], c.tangent(sp)))
        assert dot >= 1.0 - 1e-9, f"{at}: tangent {e['tangent']} against the start->end direction (dot {dot})"
        box = c.box()
        got = (e["lo"][0], e["lo"][1], e["hi"][0], e["hi"][1])
        assert max(abs(x - y) for x, y in zip(box, got)) <= NODE_REL * diag, f"{at}: box {got} != {box}"
        assert abs(c.length() / e["length_m"] - 1.0) <= 1e-9, f"{at}: length {e['length_m']} != {c.length()}"
        mids = [m for _, m, _, _ in samples[e["tag"]] if m is not None]
        assert mids, f"{at}: no mesh point on it"
        for m in mids:
            x = msh.p(m)
            assert c.distance(x) <= NODE_REL * diag, f"{at}: mesh node {x} is {c.distance(x):.3g} m off the curve"
            assert 0.0 < c.s(x) < 1.0, f"{at}: mesh node {x} at s = {c.s(x)} (outside the arc)"
        for a, _, b, _ in samples[e["tag"]]:
            for n in (a, b):
                assert c.distance(msh.p(n)) <= tol, f"{at}: mesh vertex {msh.p(n)} off the curve"


def _tiles(c: R.Curve, edges: list, msh: R.Msh, at: str) -> int:
    """Asserts that the edges (a, b node pairs) tile the curve from s = 0 to 1 exactly once, every edge in one sense;
    returns that sense (+1 from start to end)."""
    steps = [c.step(msh.p(a), msh.p(b)) for a, _, b, _ in edges]
    assert all(s != 0.0 for s in steps), f"{at}: an edge of zero extent"
    sense = 1 if steps[0] > 0 else -1
    assert all((s > 0) == (sense > 0) for s in steps), f"{at}: its mesh edges run both ways"
    total = sum(abs(s) for s in steps)
    assert abs(total - 1.0) <= S_END, f"{at}: the mesh edges cover {total} of the arc from start to end"
    if not c.closed:
        iv = sorted((min(c.s(msh.p(a)), c.s(msh.p(b))), max(c.s(msh.p(a)), c.s(msh.p(b)))) for a, _, b, _ in edges)
        assert abs(iv[0][0]) <= S_END and abs(iv[-1][1] - 1.0) <= S_END, f"{at}: the edges span {iv[0][0]}..{iv[-1][1]}"
        for (_, hi), (lo, _) in zip(iv, iv[1:]):
            assert hi == lo, f"{at}: a gap or an overlap between mesh edges at s = {hi}, {lo}"
    else:
        count: dict = {}
        for a, _, b, _ in edges:
            count[a] = count.get(a, 0) + 1
            count[b] = count.get(b, 0) + 1
        assert set(count.values()) == {2}, f"{at}: the mesh edges of a closed curve do not form one cycle"
    return sense


@pytest.mark.parametrize("case", CASES)
def test_faces_are_the_mesh_s_faces_bounded_by_closed_oriented_loops(mp, tmp_path_factory, case):
    """``faces`` lists exactly the mesh's faces (elementary tags of the triangles), each with the region of its
    triangles; the outer loop is counter-clockwise and the inner loops clockwise (Green's theorem on the parameters);
    a face runs each entity of its loops once, with its region on the entity's left that way (``side[0]`` forward,
    ``side[1]`` backward); the boundary edges of the face's triangles lie on exactly its loop entities, tile each of them
    from ``start`` to ``end``, and run the way the loop runs it (the triangles are on the left of their edges); the
    face's mesh area equals its loops' area; per region the loops' area (and revolved volume) equal the info file's."""
    table, msh, cs, samples = _case_samples(mp, case, tmp_path_factory.getbasetemp())
    ent = {e["tag"]: e for e in table["entities"]}
    names2 = {t: n for (d, t), n in msh.names.items() if d == 2}
    tris = msh.triangles()
    by_face: dict = {}
    for k, t in enumerate(tris):
        by_face.setdefault(t[2], []).append(k)
    assert sorted(f["tag"] for f in table["faces"]) == sorted(by_face), "faces != the mesh's faces"
    assert [f["tag"] for f in table["faces"]] == sorted(f["tag"] for f in table["faces"]), "faces not in tag order"
    region_area: dict = {}
    region_vol: dict = {}
    for f in table["faces"]:
        at = f"face {f['tag']} ({f['region']})"
        assert {names2[tris[k][1]] for k in by_face[f["tag"]]} == {f["region"]}, f"{at}: its triangles' region"
        flat = [s for l in f["loops"] for s in l]
        assert len({abs(s) for s in flat}) == len(flat), f"{at}: an entity twice in its loops"
        for s in flat:
            side = ent[abs(s)]["side"][0 if s > 0 else 1]
            assert side == f["region"], f"{at}: runs entity {abs(s)} {'forward' if s > 0 else 'backward'}, left is {side}"
        meas = [R.loop_measures(l, cs) for l in f["loops"]]
        assert meas[0][0] > 0.0, f"{at}: the outer loop is not counter-clockwise"
        assert all(m[0] < 0.0 for m in meas[1:]), f"{at}: an inner loop is not clockwise"
        area = sum(m[0] for m in meas)
        region_area[f["region"]] = region_area.get(f["region"], 0.0) + area
        region_vol[f["region"]] = region_vol.get(f["region"], 0.0) + sum(m[1] for m in meas)
        sign = {abs(s): (1 if s > 0 else -1) for s in flat}
        mine: dict = {}
        for t, lst in samples.items():
            for smp in lst:
                if smp[3] == f["tag"]:
                    mine.setdefault(t, []).append(smp)
        assert set(mine) == set(sign), f"{at}: mesh boundary on {sorted(mine)}, loops name {sorted(sign)}"
        gap = 0.0
        for t, edges in mine.items():
            sense = _tiles(cs[t], edges, msh, f"{at}, entity {t}")
            assert sense == sign[t], f"{at}: runs entity {t} with sign {sign[t]}, its triangles lie the other way"
            if ent[t]["type"] != "line":
                gap += sum(R.p2_gap(msh.p(a), msh.p(m), msh.p(b)) for a, m, b, _ in edges if m is not None)
        a_mesh, _ = R.triangle_measures(msh, [tris[k] for k in by_face[f["tag"]]])
        A = float(a_mesh.sum())
        assert abs(A - area) <= 1e-12 * area + GAP_FACTOR * gap, f"{at}: mesh area {A}, loops {area} (gap {gap:.3g})"
    for r, rec in table["regions"].items():
        assert abs(region_area[r] / rec["area_m2"] - 1.0) <= 1e-10, f"region {r}: {region_area[r]} != {rec['area_m2']}"
        if table["geometry"] == "axisymmetric":
            assert abs(region_vol[r] / rec["volume_m3"] - 1.0) <= 1e-10, f"region {r}: volume {region_vol[r]}"
    # an entity is used by the faces of its sides: once on each side that is not the exterior, or by none (embedded)
    used: dict = {}
    for f in table["faces"]:
        for l in f["loops"]:
            for s in l:
                used.setdefault(abs(s), []).append(s)
    for e in table["entities"]:
        u = sorted(used.get(e["tag"], []))
        if not u:
            assert e["side"][0] == e["side"][1] != "exterior", f"entity {e['tag']} is in no loop but has sides {e['side']}"
            continue
        want = sorted(([e["tag"]] if e["side"][0] != "exterior" else []) + ([-e["tag"]] if e["side"][1] != "exterior" else []))
        assert u == want, f"entity {e['tag']} with sides {e['side']} is run {u} by the faces"


@pytest.mark.parametrize("case", CASES)
def test_line_elements_lie_on_their_entities(mp, tmp_path_factory, case):
    """The line elements of each entity (elementary tag; markers, the axis, atoms, the outer arc) tile its arc from
    ``start`` to ``end``; in the bounded modes they run along ``tangent``, so every element goes from start to end."""
    table, msh, cs, samples = _case_samples(mp, case, tmp_path_factory.getbasetemp())
    for t, lst in samples.items():
        edges = [x for x in lst if x[3] is None]
        if not edges:
            continue
        sense = _tiles(cs[t], edges, msh, f"entity {t}, line elements")
        if case.startswith("bounded:"):
            assert sense == 1, f"entity {t}: its line elements run from end to start (start is not first along tangent)"


FILLET_LOSS = 1.0 - math.pi / 4.0      # the area a fillet of radius 1 cuts off a right-angled corner


@pytest.mark.parametrize("case, region, area", [
    ("em:clockwise", "a", 0.04 - 4 * 0.04 ** 2 * FILLET_LOSS),              # square, four convex corners
    ("em:clockwise", "b", 0.12 - 4 * 0.05 ** 2 * FILLET_LOSS),              # L: five convex corners, one reflex
    ("em:markers", "b", 0.04 - 0.05 ** 2 * FILLET_LOSS)])                    # square, one corner under the arc sheet
def test_clockwise_arcs_and_a_reversed_arc_marker_give_the_exact_areas(mp, tmp_path_factory, case, region, area):
    """Polygons written clockwise make clockwise fillet arcs (``ccw`` false); the arc sheet drawn counter-clockwise
    over a clockwise fillet is an entity whose written direction runs against gmsh's parametrization. Each region's
    area from its loops (Green's theorem on the curve parameters) equals the closed form within 1e-12: an arc whose
    ``ccw`` is wrong is its complement, and the region's area is then off by tens of percent. The sheet's entity runs
    from (0.30, 0.35) to (0.25, 0.40), counter-clockwise, with region b on its left."""
    table, _, _ = run_case(mp, case, tmp_path_factory.getbasetemp())
    _has_keys(table)
    cs = R.curves(table)
    got = sum(R.loop_measures(l, cs)[0] for f in table["faces"] if f["region"] == region for l in f["loops"])
    assert abs(got / area - 1.0) <= 1e-12, f"region {region}: area {got} from the loops, closed form {area}"
    arcs = [e for e in table["entities"] if e["type"] != "line" and region in e["side"]]
    if case == "em:clockwise":
        assert any(e["ccw"] is False for e in arcs), "no clockwise arc on the clockwise polygon"
    else:
        sheet = [e for e in table["entities"] if "s2" in e["curves"]]
        assert len(sheet) == 1, sheet
        e, tol = sheet[0], table["tolerance_m"]
        assert e["type"] in ("circle", "ellipse") and e["ccw"] is True, e
        assert math.dist(e["start"][:2], (0.30, 0.35)) <= tol and math.dist(e["end"][:2], (0.25, 0.40)) <= tol, e
        assert e["side"] == ["b", "air"], e["side"]


# ============================================================================================ the export mode
def test_the_usage_names_the_export_mode(mp):
    import subprocess
    r = subprocess.run([mp], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "--export-regions" in r.stderr, r.stderr
    r = subprocess.run([mp, "--export", "x.json"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "--export-regions" in r.stderr, r.stderr


@pytest.mark.parametrize("name", DESIGNS)
def test_export_regions_writes_the_partition_and_no_mesh(mp, tmp_path_factory, name):
    table, msh, extra = run_case(mp, f"em:{name}", tmp_path_factory.getbasetemp())
    assert set(table) == R.EXPORT_KEYS, sorted(set(table) ^ R.EXPORT_KEYS)
    assert table["ok"] is True and table["export_regions"] is True and table["geometry"] == "axisymmetric"
    inp = extra["input"]
    assert table["domain_radius"] == inp["domain_radius"] and table["domain_center_z"] == inp["domain_center_z"]
    assert table["tolerance_m"] == 1e-9 * table["diagonal_m"]
    lo = np.min([e["lo"][:2] for e in table["entities"]], 0)
    hi = np.max([e["hi"][:2] for e in table["entities"]], 0)
    assert abs(table["diagonal_m"] / math.hypot(*(hi - lo)) - 1.0) <= 1e-15
    assert not extra["unused_output"].exists(), "the export mode wrote a mesh"
    want = {"air": inp["air_attr"]} | {r["name"]: r["attr"] for r in inp["regions"]}
    assert {k: v["attr"] for k, v in table["regions"].items()} == want
    for v in table["regions"].values():
        assert set(v) == {"attr", "area_m2", "volume_m3"} and v["area_m2"] > 0 and v["volume_m3"] > 0
    names = set(want) | {"exterior"}
    for e in table["entities"]:
        assert R.ENTITY_FORM <= set(e) and "atom" not in e, e
        assert set(e) <= R.ENTITY_FORM | set(R.ENTITY_KEYS_ADDED) | {"curves_reversed"}, sorted(e)
        assert set(e["side"]) <= names and e["dim"] == 1
        assert set(e["curves"]) <= {c["name"] for c in inp.get("curves", [])}
    # every curve of the mesh's line elements is an entity; the axis and outer groups are the axis and exterior arcs
    names1 = {t: n for (d, t), n in msh.names.items() if d == 1}
    ent = {e["tag"]: e for e in table["entities"]}
    for _, phys, etag, _ in msh.lines():
        e = ent[etag]
        if names1[phys] == "axis":
            assert e["axis"] is True and "exterior" in e["side"]
        elif names1[phys] == "outer":
            assert e["axis"] is False and "exterior" in e["side"] and e["type"] != "line"
        else:
            assert names1[phys] in e["curves"], (names1[phys], e)


@pytest.mark.parametrize("name", DESIGNS)
def test_export_regions_is_deterministic_and_depends_on_the_geometry_only(mp, tmp_path, name):
    """Two runs give the same partition (``seconds`` aside), and so does the input of the same design with other
    mesh sizes and geometry order 1."""
    inp, _, _ = R.em_mesh(R.design(name), tmp_path / "mesh")
    outs = []
    for k, scale in enumerate((1.0, 1.0, 0.5)):
        i = copy.deepcopy(inp)
        for key in ("size", "surface_size", "far_size", "surface_layer"):
            i[key] = i[key] * scale
        if scale != 1.0:
            i["order"] = 1
        r, part, _ = R.run_mesher(mp, i, tmp_path / f"r{k}", "regions", export=True)
        assert r.returncode == 0 and part and part["ok"] is True, r.stderr[-500:]
        outs.append(R.without_seconds(part))
    assert R.same_json(outs[0], outs[1]) == []
    assert R.same_json(outs[0], outs[2]) == []


def test_contour_ellipses_are_exported_with_the_input_s_parameters(mp, tmp_path_factory):
    """The nose's elliptical segment (axes 0.06 along r, 0.1 along z, a rational B-spline in gmsh) is an ``ellipse``
    whose centre and radii are the contour's numbers bit for bit, its major axis along z; its arc runs from (0.25, 0.3)
    to (0.25, 0.5) through r = 0.31."""
    table, _, _ = run_case(mp, "em:curved", tmp_path_factory.getbasetemp())
    _has_keys(table)
    c0, c1 = R.NOSE["center"]
    on = [e for e in table["entities"] if e["type"] != "line" and "nose" in e["side"]
          and e.get("center", [None, None])[:2] == [c0, c1]]
    assert on, "no arc entity of the nose with the contour's centre (bit for bit)"
    for e in on:
        assert e["type"] == "ellipse" and e["radii"] == [0.1, 0.06] and e["major_axis"] == [0.0, 1.0, 0.0], e
    total = sum(e["length_m"] for e in on)
    # half the ellipse's perimeter: the integral of sqrt((a sin t)^2 + (b cos t)^2) over a half turn
    t = np.linspace(-math.pi / 2, math.pi / 2, 200001)
    ref = np.trapezoid(np.hypot(0.06 * np.sin(t), 0.1 * np.cos(t)), t)
    assert abs(total / ref - 1.0) <= 1e-9
    ends = {tuple(e[k][:2]) for e in on for k in ("start", "end")}
    assert (0.25, 0.3) in ends and (0.25, 0.5) in ends


# 3D inputs are exported too (tests/test_mesher_partition_3d.py); the bounded and plane modes are refused
@pytest.mark.parametrize("which", ["bounded", "plane"])
def test_export_regions_refuses_the_inputs_of_the_other_modes(mp, tmp_path, which):
    inp = {"bounded": lambda: R.fixture("pipe_axi"), "plane": lambda: R.fixture("plane_rect")}[which]()
    r, info, msh = R.run_mesher(mp, inp, tmp_path, export=True)
    assert r.returncode == 1, (r.returncode, r.stderr[-500:])
    assert info is not None and info["ok"] is False and "--export-regions" in info["error"], info
    assert not msh.exists()
    if which == "bounded":
        assert "resolve_only" in info["error"]


def test_a_failed_export_run_is_reported_in_the_info_file(mp, tmp_path):
    """A region outside the domain (the domain radius cut to 0.3 m in the input): the construction's own refusal, in
    the info file, exit 1, no mesh."""
    inp, _, _ = R.em_mesh(R.design("yoke"), tmp_path / "mesh")
    inp["domain_radius"] = 0.3
    r, info, msh = R.run_mesher(mp, inp, tmp_path / "x", "regions", export=True)
    assert r.returncode == 1, (r.returncode, r.stderr[-500:])
    assert info is not None and info["ok"] is False and "outside the computational domain" in info["error"], info
    assert not msh.exists()


# ============================================================================================ (3) volumes
@pytest.mark.parametrize("case", [f"em:{n}" for n in DESIGNS] + [f"bounded:{n}" for n in AXI_BOUNDED])
def test_exported_region_volumes_equal_the_mesh_s(mp, tmp_path_factory, case):
    """``solids.from_partition`` accepts the partition, and each region's exact volume (``volume_m3``, from the curve
    parameters) equals the region's volume in the P2 mesh of the same input (the integral of 2 pi r over its
    triangles) within the P2 boundary's error, and the info file's OpenCASCADE volume within 1e-10."""
    from axionmultiphysics import solids

    table, msh, cs, _ = _case_samples(mp, case, tmp_path_factory.getbasetemp())
    regions = list(table["regions"])
    mats = {f"m_{r}": {"density": 1000.0, "particle": {"nist": "G4_WATER"}} for r in regions}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = solids.from_partition(table, mats, {r: f"m_{r}" for r in regions})
    assert set(doc["regions"]) == set(regions)
    vols = R.region_volumes(msh)
    for r in regions:
        V_mesh, gap = vols[r]
        V = doc["regions"][r]["volume_m3"]
        assert abs(V_mesh - V) <= 1e-12 * V + GAP_FACTOR * gap, f"region {r}: mesh {V_mesh}, solids {V} (gap {gap:.3g})"
        assert abs(table["regions"][r]["volume_m3"] / V - 1.0) <= 1e-10, f"region {r}: info file {table['regions'][r]}"
