"""``axmp boundaries``, ``Design.boundaries()`` and the full check of a model document (design note 5.5, 5.6, 14.2):
the resolved table of the named boundaries, and the checks on the resolved boundaries that ``axmp validate`` defers.

``axmp boundaries design.json [--json] [--mesh] [-o DIR] [--mesher M]`` validates the document with the mesher's probe,
runs the mesher's ``resolve_only`` pass (with ``--mesh``, a meshing run), runs the full check (the partition rule of
every instance, the empty selections, the partial matches, the force rules and the outputs on resolved boundaries) and
prints the table. ``--json`` prints the validation report with the table added (``symmetry``, ``boundaries``,
``edges``): after the full check its ``deferred`` list is empty. With an error the command prints that report with
``ok`` false and the first error's fields at the top level (as ``axmp validate`` does), and exits with status 1; the
table is in it when the boundaries could be resolved, since the hints of the partition rule send the user to this
command. ``Design.boundaries(mesh=False, all_errors=False)`` and ``_core.boundaries`` return the ``--json`` report and
raise the first error (``all_errors=True``: the report, without raising).

The table (``boundaries``): one row per named boundary, keyed by its name, with ``selector`` (as written),
``entities``, ``measure`` and ``measure_units`` (the length, "m"), ``area_m2`` (the revolved area; axisymmetric models
only), ``bbox`` (exact, per coordinate name), ``point`` (a point of the boundary), ``adjacent`` (the region pairs, in
pair order: document order, ``exterior`` last) and ``used_by``: the conditions that key it, ``{"physics", "condition"
(the kind), "entities", "measure"}`` with the part the instance takes (design note 5.5; a key is taken where its
boundary lies on the domain boundary of the instance), sorted by physics id, then the outputs on it,
``{"output", "kind"}``, sorted by output name. ``edges``: every boundary entity (every edge with an atom), sorted by
its point: ``adjacent`` (one pair), ``type``, ``bbox``, ``point``, ``measure``, ``measure_units`` (and ``area_m2``),
``names`` (the boundaries that select it, sorted) and ``axis``. With ``--mesh``, every row and every edge also has
``faces``, its number of boundary elements on the master mesh (with the document's refinements). Nothing in the table
carries a master number or a geometry kernel's tag (design note 12.1: no API hands out a master number).

Independent references: the analytic lengths, areas and boxes of the fixtures (``cpp/tests/data/model/u2``,
``channel.json``); the texts of design note 5.2 (the shifted channel) and 5.5 (the uncovered and doubly covered
channel), verbatim; the texts of the message builders of ``cpp/src/model_errors.hpp`` (tested on hand-made tables in
``cpp/tests/test_model_selection.cpp``) for the other refusals; the line elements of the mesh file, counted here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core

import model_mesh_support as S
from model_mesh_support import mesher  # noqa: F401 (a fixture)

L, H = 0.04, 0.01
REL = 1e-12
ROW_KEYS = {"selector", "entities", "measure", "measure_units", "bbox", "point", "adjacent", "used_by"}
EDGE_KEYS = {"adjacent", "type", "bbox", "point", "measure", "measure_units", "names", "axis"}


def _close(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= REL * scale


def _boundaries(tmp_path: Path, document: dict, *args, name: str = "doc.json") -> tuple[int, dict]:
    return S.run_json("boundaries", S.write(tmp_path, document, name), "--json", *args)


def _ok_report(tmp_path: Path, document: dict, *args) -> dict:
    status, rep = _boundaries(tmp_path, document, *args)
    assert status == 0, json.dumps(rep, indent=1)[:4000]
    assert rep["ok"] is True and rep["errors"] == [], rep["errors"]
    assert S.REPORT_KEYS | S.TABLE_KEYS <= set(rep) <= S.REPORT_KEYS | S.TABLE_KEYS, sorted(rep)
    return rep


def _refusal(tmp_path: Path, document: dict, *args) -> dict:
    """``axmp boundaries --json`` of a document with an error: exit 1, the report with ok false and the first error's
    fields at the top level; returns the report."""
    status, rep = _boundaries(tmp_path, document, *args)
    assert status == 1, json.dumps(rep, indent=1)[:4000]
    assert rep["ok"] is False and rep["errors"], rep
    first = rep["errors"][0]
    assert {k: rep[k] for k in first} == first
    return rep


def _check_box(row: dict, want: dict, scale: float) -> None:
    assert set(row["bbox"]) == set(want), row["bbox"]
    for c, (lo, hi) in want.items():
        got = row["bbox"][c]
        assert _close(got[0], lo, scale) and _close(got[1], hi, scale), f"bbox {c}: {got}, want {[lo, hi]}"
    names = list(want)
    for c, v in zip(names, row["point"]):
        assert row["bbox"][c][0] - REL * scale <= v <= row["bbox"][c][1] + REL * scale, f"point {row['point']} outside the box"


def _check_row(row: dict, *, selector, entities, measure, bbox, adjacent, used_by, scale, area=None) -> None:
    keys = ROW_KEYS | ({"area_m2"} if area is not None else set())
    assert set(row) == keys, sorted(row)
    assert row["selector"] == selector
    assert row["entities"] == entities
    assert row["measure_units"] == "m"
    assert _close(row["measure"], measure, scale), f"measure {row['measure']}, want {measure}"
    if area is not None:
        assert _close(row["area_m2"], area, scale * scale), f"area {row['area_m2']}, want {area}"
    _check_box(row, bbox, scale)
    assert row["adjacent"] == adjacent
    assert len(row["used_by"]) == len(used_by), row["used_by"]
    for got, want in zip(row["used_by"], used_by):
        assert set(got) == set(want), got
        for k, v in want.items():
            if k == "measure":
                assert _close(got[k], v, scale), f"used_by measure {got[k]}, want {v}"
            else:
                assert got[k] == v, f"used_by {k}: {got[k]!r}, want {v!r}"


def _check_edges(rep: dict) -> None:
    """Every edge has its keys; the measures of the edges of a name add up to the name's row; no master number."""
    for e in rep["edges"]:
        assert set(e) == EDGE_KEYS | ({"area_m2"} if rep["symmetry"] == "axisymmetric" else set()) | (
            {"faces"} if "faces" in e else set()), sorted(e)
    points = [tuple(e["point"]) for e in rep["edges"]]
    assert points == sorted(points), "the edges are sorted by their point"
    for name, row in rep["boundaries"].items():
        mine = [e for e in rep["edges"] if name in e["names"]]
        assert len(mine) == row["entities"], name
        assert _close(sum(e["measure"] for e in mine), row["measure"], row["measure"]), name
    assert S.keys_anywhere({k: rep[k] for k in ("boundaries", "edges")}, S.NUMBER_KEYS) == []


# ----------------------------------------------------------------------------------------------------------- tables
def test_the_table_of_the_split_rectangle(tmp_path, mesher):
    """split_rect: two regions, a marker across a written bottom to top, a disabled third region; the heat instance
    spans a and b. Rows against the analytic geometry; the interface and the marker are interior to the instance and
    used by no condition; the marker is used by the output Q_mk. Ten edges: the marker splits the top and the bottom of
    a, so six edges are named by no boundary (the remaining ones, 2 L in all). After the full check nothing is deferred,
    and the Python API returns the same report."""
    d = S.doc("split_rect")
    rep = _ok_report(tmp_path, d)
    assert rep["symmetry"] == "planar" and rep["name"] == "split_rect"
    assert rep["deferred"] == []
    assert set(rep["boundaries"]) == {"cold", "hot", "interface", "mk"}
    b = rep["boundaries"]
    heat = lambda m: [{"physics": "heat", "condition": "temperature", "entities": 1, "measure": m}]  # noqa: E731
    _check_row(b["hot"], selector={"of": "left", "where": {"x": 0}}, entities=1, measure=H,
               bbox={"x": [0, 0], "y": [0, H]}, adjacent=[["a", "exterior"]], used_by=heat(H), scale=L)
    _check_row(b["cold"], selector={"of": "b", "where": {"x": "L"}}, entities=1, measure=H,
               bbox={"x": [L, L], "y": [0, H]}, adjacent=[["b", "exterior"]], used_by=heat(H), scale=L)
    _check_row(b["interface"], selector={"between": ["left", "b"]}, entities=1, measure=H,
               bbox={"x": [L / 2, L / 2], "y": [0, H]}, adjacent=[["a", "b"]], used_by=[], scale=L)
    _check_row(b["mk"], selector={"curve": "mk_line"}, entities=1, measure=H,
               bbox={"x": [L / 4, L / 4], "y": [0, H]}, adjacent=[["a", "a"]],
               used_by=[{"output": "Q_mk", "kind": "flux"}], scale=L)
    assert len(rep["edges"]) == 10
    _check_edges(rep)
    unnamed = [e for e in rep["edges"] if not e["names"]]
    assert len(unnamed) == 6 and _close(sum(e["measure"] for e in unnamed), 2 * L, L)
    assert all(e["type"] == "line" and e["axis"] is False for e in rep["edges"])
    assert sorted(tuple(e["adjacent"]) for e in unnamed) == [("a", "exterior")] * 4 + [("b", "exterior")] * 2
    assert not [w for w in rep["warnings"] if w["code"] == "boundary.partial_match"]
    # the Python API: the same report
    assert axmp.Design(d).boundaries() == rep
    assert json.loads(_core.boundaries(json.dumps(d))) == rep


def test_the_table_of_the_axisymmetric_pipe(tmp_path, mesher):
    """pipe_axi: the rows carry the revolved area (pi R^2 for the ends, 2 pi R Lz for the wall over both regions); the
    axis r = 0 is in no row, and its two edges are listed with ``axis`` true and no name."""
    R, Lz = 0.005, 0.04
    rep = _ok_report(tmp_path, S.doc("pipe_axi"))
    assert rep["symmetry"] == "axisymmetric" and rep["deferred"] == []
    assert set(rep["boundaries"]) == {"inlet", "outlet", "wall"}
    b = rep["boundaries"]
    heat = lambda kind, n, m: [{"physics": "heat", "condition": kind, "entities": n, "measure": m}]  # noqa: E731
    _check_row(b["inlet"], selector={"of": "p1", "where": {"z": 0}}, entities=1, measure=R, area=math.pi * R * R,
               bbox={"r": [0, R], "z": [0, 0]}, adjacent=[["p1", "exterior"]], used_by=heat("temperature", 1, R), scale=Lz)
    _check_row(b["outlet"], selector={"of": "p2", "where": {"z": "Lz"}}, entities=1, measure=R, area=math.pi * R * R,
               bbox={"r": [0, R], "z": [Lz, Lz]}, adjacent=[["p2", "exterior"]], used_by=heat("temperature", 1, R),
               scale=Lz)
    _check_row(b["wall"], selector={"of": "pipe", "where": {"r": "R"}}, entities=2, measure=Lz, area=2 * math.pi * R * Lz,
               bbox={"r": [R, R], "z": [0, Lz]}, adjacent=[["p1", "exterior"], ["p2", "exterior"]],
               used_by=heat("insulation", 2, Lz), scale=Lz)
    _check_edges(rep)
    axis = [e for e in rep["edges"] if e["axis"]]
    assert len(axis) == 2 and all(e["names"] == [] and e["bbox"]["r"] == [0, 0] for e in axis)
    assert _close(sum(e["measure"] for e in axis), Lz, Lz) and all(e["area_m2"] == 0 for e in axis)
    inner = [e for e in rep["edges"] if e["adjacent"] == ["p1", "p2"]]
    assert len(inner) == 1 and inner[0]["names"] == [] and _close(inner[0]["area_m2"], math.pi * R * R, Lz * Lz)


def test_the_table_of_the_channel_lists_conditions_and_outputs(tmp_path, mesher):
    """channel.json (a flow, whose route is not implemented in this version: the command needs no route). "Used by"
    lists the condition with its kind, then the outputs on the boundary by name; the report keeps the route's
    unavailable finding, and the command exits 0."""
    rep = _ok_report(tmp_path, S.doc("channel"))
    assert [f["code"] for f in rep["unavailable"]] == ["route.not_implemented"]
    assert rep["deferred"] == []
    b = rep["boundaries"]
    flow = lambda kind, n, m: [{"physics": "flow", "condition": kind, "entities": n, "measure": m}]  # noqa: E731
    _check_row(b["inlet"], selector={"of": "channel", "where": {"x": 0}}, entities=1, measure=H,
               bbox={"x": [0, 0], "y": [0, H]}, adjacent=[["channel", "exterior"]], scale=L,
               used_by=flow("inlet", 1, H) + [{"output": "Q_in", "kind": "flux"}, {"output": "p_in", "kind": "average"}])
    _check_row(b["outlet"], selector={"of": "channel", "where": {"x": "L"}}, entities=1, measure=H,
               bbox={"x": [L, L], "y": [0, H]}, adjacent=[["channel", "exterior"]], scale=L,
               used_by=flow("outlet", 1, H) + [{"output": "Q_out", "kind": "flux"}, {"output": "p_out", "kind": "average"}])
    _check_row(b["walls"], selector={"of": "channel", "except": ["inlet", "outlet"]}, entities=2, measure=2 * L,
               bbox={"x": [0, L], "y": [0, H]}, adjacent=[["channel", "exterior"]], scale=L,
               used_by=flow("wall", 2, 2 * L) + [{"output": "F_walls", "kind": "force"},
                                                {"output": "tau_max", "kind": "maximum"}])
    _check_edges(rep)


def test_a_key_is_taken_where_it_lies_on_each_instance(tmp_path, mesher):
    """split_two: 'bottom' and 'top' span a and b and are keys of the instance on a and of the instance on b; each
    takes the half on its own domain, and the two overlapping uses are accepted. 'mid' is a boundary of both."""
    rep = _ok_report(tmp_path, S.doc("split_two"))
    assert rep["deferred"] == []
    b = rep["boundaries"]
    for name, y in (("bottom", 0.0), ("top", H)):
        _check_row(b[name], selector={"of": "both", "where": {"y": 0 if name == "bottom" else "H"}}, entities=2,
                   measure=L, bbox={"x": [0, L], "y": [y, y]}, adjacent=[["a", "exterior"], ["b", "exterior"]], scale=L,
                   used_by=[{"physics": "heat_a", "condition": "insulation", "entities": 1, "measure": L / 2},
                            {"physics": "heat_b", "condition": "insulation", "entities": 1, "measure": L / 2}])
    _check_row(b["mid"], selector={"between": ["a", "b"]}, entities=1, measure=H, bbox={"x": [L / 2, L / 2], "y": [0, H]},
               adjacent=[["a", "b"]], scale=L,
               used_by=[{"physics": "heat_a", "condition": "temperature", "entities": 1, "measure": H},
                        {"physics": "heat_b", "condition": "temperature", "entities": 1, "measure": H}])
    _check_edges(rep)


def test_mesh_adds_the_face_counts(tmp_path, mesher):
    """``--mesh -o DIR`` meshes the model into DIR and gives every row and every edge its number of boundary elements:
    the edges' counts add up to the line elements of the mesh file, and a row's count is the sum over its edges. With
    one refinement every count doubles (the counts are those of the master mesh as the instance meshes are made from).
    Without ``--mesh`` there is no count. ``Design.boundaries(mesh=True)`` gives the same report."""
    d = S.doc("split_rect")
    rep = _ok_report(tmp_path, d, "--mesh", "-o", tmp_path / "m")
    msh = tmp_path / "m" / "mesh.msh"
    assert msh.is_file() and (tmp_path / "m" / "mesh_info.json").is_file()
    assert all(isinstance(e.get("faces"), int) and e["faces"] > 0 for e in rep["edges"])
    assert sum(e["faces"] for e in rep["edges"]) == S.line_elements(msh)
    for name, row in rep["boundaries"].items():
        assert set(row) == ROW_KEYS | {"faces"}, sorted(row)
        assert row["faces"] == sum(e["faces"] for e in rep["edges"] if name in e["names"]), name
    _check_edges(rep)
    assert axmp.Design(d).boundaries(mesh=True) == rep
    plain = _ok_report(tmp_path, d)
    assert S.keys_anywhere(plain, {"faces"}) == []

    d["mesh"]["refinements"] = 1
    ref = _ok_report(tmp_path, d, "--mesh", "-o", tmp_path / "r")
    assert sum(e["faces"] for e in ref["edges"]) == 2 * S.line_elements(tmp_path / "r" / "mesh.msh")
    assert {n: r["faces"] for n, r in ref["boundaries"].items()} == {n: 2 * r["faces"] for n, r in rep["boundaries"].items()}


def test_the_text_table_is_the_default(tmp_path, mesher):
    """Without ``--json`` the table is text (not JSON) that names every boundary; it carries no master number."""
    r = S.run("boundaries", S.write(tmp_path, S.doc("split_rect")))
    assert r.returncode == 0, r.stdout + r.stderr
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.stdout)
    for name in ("hot", "cold", "interface", "mk"):
        assert name in r.stdout, name
    for number in ("2001", "2002", "2003", "2004", "2005", "2006"):
        assert number not in r.stdout, number


# ----------------------------------------------------------------------------------------------- deferred, resolved
def test_the_full_check_resolves_what_validate_defers(tmp_path, mesher):
    """``axmp validate`` defers the resolution of the boundaries and the partition rule (and a force on a boundary that
    is not a key); ``axmp boundaries`` runs those checks and removes the deferred findings: a force on a part of the
    walls is accepted (every part of it is a wall at rest)."""
    d = S.doc("channel")
    d["boundaries"]["floor"] = {"of": "channel", "where": {"y": 0}}
    d["outputs"]["F_floor"] = {"kind": "force", "on": "floor"}
    path = S.write(tmp_path, d)
    status, val = S.run_json("validate", path)
    assert status == 0, val
    deferred = {f["code"] for f in val["deferred"]}
    assert {"boundary.empty_selection", "physics.boundary_uncovered", "output.force_needs_stress_form"} <= deferred, deferred
    status, rep = S.run_json("boundaries", path, "--json")
    assert status == 0 and rep["ok"] is True, rep
    assert rep["errors"] == [] and rep["deferred"] == []
    assert rep["boundaries"]["floor"]["used_by"] == [{"output": "F_floor", "kind": "force"}]


def test_a_force_on_a_boundary_with_another_condition_is_refused_once_resolved(tmp_path, mesher):
    """The same rule when part of the boundary carries a condition that is not a wall at rest: deferred by
    ``axmp validate``, refused by the full check (output.force_needs_stress_form in its resolved form, naming the key
    and its condition)."""
    d = S.doc("channel")
    d["boundaries"]["left_all"] = {"of": "channel", "where": {"x": 0}}
    d["outputs"] = {"F_left": {"kind": "force", "on": "left_all"}}
    rep = _refusal(tmp_path, d)
    assert [f["code"] for f in rep["errors"]] == ["output.force_needs_stress_form"]
    S.assert_finding(
        rep["errors"][0], code="output.force_needs_stress_form", where="/outputs/F_left",
        message="output 'F_left': a force on 'left_all' is not available with the Laplacian viscous form on this route",
        hint=("part of 'left_all' carries the condition 'inlet' of 'inlet'. The Navier-Stokes engine returns, for the "
              "Laplacian form, the pseudo-traction mu du/dn - p n, which is the force only on walls at rest. Use "
              "\"viscous_form\": \"stress\" (the physical form), or \"equation\": \"stokes\" if the flow is creeping."),
        details={"boundary": "left_all", "key": "inlet", "condition": "inlet", "route": "navier_stokes_steady_planar",
                 "accepted_on": "wall boundaries at rest"})
    assert rep["deferred"] == []


# ------------------------------------------------------------------------------------------------------- refusals
def _shifted_channel() -> dict:
    """Design note 5.2's example: the channel drawn from x = 0.0005, 'inlet' still at x = 0 (the outlet follows the
    shift, and the outputs are removed, so that the empty inlet is the only finding)."""
    d = S.doc("channel")
    d["regions"][0]["shape"] = {"rect": {"x": [0.0005, 0.0405], "y": [0, "H"]}}
    d["boundaries"]["outlet"] = {"of": "channel", "where": {"x": 0.0405}}
    del d["outputs"]
    return d


SHIFTED = {
    "error": "boundary 'inlet' selects nothing",
    "where": "/boundaries/inlet/where",
    # design note 5.2, verbatim (joined into one line)
    "hint": ("no boundary edge of 'channel' lies on x = 0. Edges of 'channel': x = 0.0005 (0.01 m), x = 0.0405 (0.01 m), "
             "y = 0 (0.04 m), y = 0.01 (0.04 m). Did you mean {\"x\": 0.0005}, or is the geometry shifted? To select part "
             "of an edge, add a curve along that part and use {\"curve\": NAME}."),
    "code": "boundary.empty_selection",
    "details": {"selector": {"of": "channel", "where": {"x": 0}},
                "candidates": [{"where": {"x": 0.0005}, "measure_m": 0.01}, {"where": {"x": 0.0405}, "measure_m": 0.01},
                               {"where": {"y": 0}, "measure_m": 0.04}, {"where": {"y": 0.01}, "measure_m": 0.04}],
                "partial": [], "axis_only": False},
}


def test_an_empty_selection_is_refused_with_the_notes_text(tmp_path, mesher):
    d = _shifted_channel()
    rep = _refusal(tmp_path, d)
    assert rep["errors"] == [SHIFTED], json.dumps(rep["errors"], indent=1)
    S.assert_exception_is(S.raises(axmp.Design(d).boundaries), SHIFTED)
    S.assert_exception_is(S.raises(_core.boundaries, json.dumps(d)), SHIFTED)
    # all_errors: the report, not an exception
    assert axmp.Design(d).boundaries(all_errors=True)["errors"] == [SHIFTED]


def test_a_selection_of_the_axis_alone_is_empty_with_the_axis_hint(tmp_path, mesher):
    """The axis r = 0 is in no named selection: a where filter that keeps only axis edges selects nothing,
    and the hint says that the axis is reserved."""
    d = S.doc("pipe_axi")
    d["boundaries"]["axis_sel"] = {"of": "pipe", "where": {"r": 0}}
    rep = _refusal(tmp_path, d)
    f = rep["errors"][0]
    S.assert_finding(f, code="boundary.empty_selection", where="/boundaries/axis_sel/where",
                     message="boundary 'axis_sel' selects nothing",
                     hint=("'axis_sel' lies only on the axis r = 0, which is reserved: no named boundary includes it, and "
                           "every physics applies its axis condition there automatically."))
    assert f["details"]["axis_only"] is True and f["details"]["selector"] == {"of": "pipe", "where": {"r": 0}}
    assert [g["code"] for g in rep["errors"]] == ["boundary.empty_selection"]


def _uncovered_channel() -> dict:
    """Design note 5.5's first example: the channel with no condition (and no name) for its walls."""
    d = S.doc("channel")
    del d["boundaries"]["walls"]
    del d["physics"]["flow"]["boundary_conditions"]["walls"]
    del d["outputs"]
    return d


def _uncovered(document: str) -> dict:
    # design note 5.5, verbatim (joined into one line); the command names the document as the user gave it
    return {"error": "physics 'flow': part of the boundary of its domain has no condition",
            "where": "/physics/flow/boundary_conditions",
            "hint": ("uncovered: 2 edges, 0.08 m, between 'channel' and the exterior, first at (x, y) = (0.02, 0). Name them "
                     "(for example \"walls\": {\"of\": \"channel\", \"except\": [\"inlet\", \"outlet\"]}) and give them a "
                     "condition, or add \"remaining\": {\"kind\": \"wall\"} to this physics. `axmp boundaries " + document +
                     "` lists every boundary."),
            "code": "physics.boundary_uncovered",
            "details": {"physics": "flow", "entities": 2, "measure_m": 0.08, "first_point": [0.02, 0],
                        "adjacent": [["channel", "exterior"]], "conditions": ["inlet", "outlet"]}}


def test_an_uncovered_boundary_is_refused_with_the_notes_text(tmp_path, mesher):
    """The command names the document as given on its command line; the Python API has no file name and writes
    DOCUMENT. The table is in the report: the hint sends the user to it."""
    d = _uncovered_channel()
    S.write(tmp_path, d, "channel.json")
    status, rep = S.run_json("boundaries", "channel.json", "--json", cwd=tmp_path)
    assert status == 1 and rep["ok"] is False, rep
    assert rep["errors"] == [_uncovered("channel.json")], json.dumps(rep["errors"], indent=1)
    assert {k: rep[k] for k in rep["errors"][0]} == rep["errors"][0]
    assert set(rep["boundaries"]) == {"inlet", "outlet"} and len(rep["edges"]) == 4
    S.assert_exception_is(S.raises(axmp.Design(d).boundaries), _uncovered("DOCUMENT"))


DOUBLE = {
    # design note 5.5, verbatim
    "error": "physics 'flow': boundaries 'inlet' and 'west' both cover 1 edge at (x, y) = (0, 0.005)",
    "where": "/physics/flow/boundary_conditions/west",
    "hint": "one condition per boundary face. Remove one, or exclude the overlap: {\"of\": ..., \"except\": [\"inlet\"]}.",
    "code": "physics.boundary_double",
    "details": {"physics": "flow", "keys": ["inlet", "west"], "entities": 1, "measure_m": 0.01, "first_point": [0, 0.005]},
}


def test_a_doubly_covered_boundary_is_refused_with_the_notes_text(tmp_path, mesher):
    d = S.doc("channel")
    d["boundaries"]["west"] = {"of": "channel", "where": {"x": 0}}
    d["physics"]["flow"]["boundary_conditions"]["west"] = {"kind": "wall"}
    rep = _refusal(tmp_path, d)
    assert rep["errors"] == [DOUBLE], json.dumps(rep["errors"], indent=1)
    S.assert_exception_is(S.raises(axmp.Design(d).boundaries), DOUBLE)


def test_a_condition_on_an_interior_boundary_is_refused(tmp_path, mesher):
    """channel_post with the heat instance over the channel and the post: 'cylinder' is interior to it."""
    d = S.doc("channel_post")
    d["domains"]["fluid"] = ["channel", "post"]
    rep = _refusal(tmp_path, d)
    assert [f["code"] for f in rep["errors"]] == ["physics.boundary_interior"]
    f = rep["errors"][0]
    S.assert_finding(f, code="physics.boundary_interior", where="/physics/heat/boundary_conditions/cylinder",
                     message="physics 'heat': boundary 'cylinder' is not on the boundary of its domain",
                     hint=("'cylinder' lies between 'channel' and 'post', and both regions belong to 'heat'. An interior "
                           "boundary can be used in outputs, not as a boundary condition."))
    det = f["details"]
    assert {k: det[k] for k in ("physics", "boundary", "whole", "entities", "adjacent")} == {
        "physics": "heat", "boundary": "cylinder", "whole": True, "entities": 1, "adjacent": [["channel", "post"]]}
    assert abs(det["measure_m"] - 2 * math.pi * 0.002) <= 1e-12 * L
    S.assert_exception_is(S.raises(axmp.Design(d).boundaries), f)


def test_a_condition_outside_the_domain_is_refused(tmp_path, mesher):
    d = S.doc("split_two")
    d["physics"]["heat_a"]["boundary_conditions"]["right"] = {"kind": "insulation"}
    rep = _refusal(tmp_path, d)
    want = {"error": "physics 'heat_a': boundary 'right' lies outside its domain",
            "where": "/physics/heat_a/boundary_conditions/right",
            "hint": ("'right' lies between 'b' and the exterior, and the domain of 'heat_a' is 'a'. A condition key names a "
                     "boundary on the boundary of the domain: key another boundary, or change the domains of 'heat_a'."),
            "code": "physics.boundary_outside_domain",
            "details": {"physics": "heat_a", "boundary": "right", "adjacent": [["b", "exterior"]], "domain": ["a"]}}
    assert rep["errors"] == [want], json.dumps(rep["errors"], indent=1)
    S.assert_exception_is(S.raises(axmp.Design(d).boundaries), want)


def test_an_output_outside_its_physics_is_refused(tmp_path, mesher):
    d = S.doc("split_two")
    d["outputs"] = {"Q_r": {"kind": "flux", "of": "heat", "physics": "heat_a", "on": "right"}}
    rep = _refusal(tmp_path, d)
    want = {"error": "output 'Q_r': boundary 'right' is not in the domain of 'heat_a'",
            "where": "/outputs/Q_r",
            "hint": ("'right' lies between 'b' and the exterior, and the domain of 'heat_a' is 'a'. An output of 'heat_a' is "
                     "computed on a boundary or a section of its domain."),
            "code": "output.boundary_not_in_domain",
            "details": {"output": "Q_r", "boundary": "right", "physics": "heat_a", "adjacent": [["b", "exterior"]],
                        "domain": ["a"]}}
    assert rep["errors"] == [want], json.dumps(rep["errors"], indent=1)


def test_a_force_on_an_interior_section_is_refused(tmp_path, mesher):
    """A marker across the channel is interior to the flow: a force on it has no reaction (stress form, so that the
    Laplacian rule is not involved)."""
    d = S.doc("channel")
    d["physics"]["flow"]["options"]["viscous_form"] = "stress"
    d["curves"] = [{"name": "mid_line", "path": {"segment": [["L/2", 0], ["L/2", "H"]]}}]
    d["boundaries"]["mid"] = {"curve": "mid_line"}
    d["outputs"] = {"F_mid": {"kind": "force", "on": "mid"}}
    rep = _refusal(tmp_path, d)
    want = {"error": "output 'F_mid': a force on 'mid' is not available: 'mid' is interior to the domain of 'flow'",
            "where": "/outputs/F_mid",
            "hint": ("a force or a torque is the engine's reaction on the boundary of the domain, and 'mid' lies inside "
                     "'channel', which belongs to 'flow'. Through an interior section, use a flux, an integral, an average, a "
                     "maximum or a minimum."),
            "code": "output.force_on_interior",
            "details": {"output": "F_mid", "kind": "force", "boundary": "mid", "physics": "flow",
                        "adjacent": [["channel", "channel"]]}}
    assert rep["errors"] == [want], json.dumps(rep["errors"], indent=1)


BETWEEN_OVERLAP = {
    "error": "boundary 'bad': the two domains of 'between' share the region 'b'",
    "where": "/boundaries/bad/between",
    "hint": ("the normal of a 'between' selection points from the first domain to the second, which is undefined on an edge "
             "whose two sides lie in both. Give two domains without a common region, or use {\"of\": DOMAIN}."),
    "code": "boundary.between_overlap",
    "details": {"boundary": "bad", "between": ["slab_all", "b"], "shared": ["b"]},
}


def test_between_two_domains_with_a_common_region_is_refused_by_validate(tmp_path):
    """Refused at validation, which needs no mesher for it: here the mesher is a stub that cannot load
    (exit 127), and ``axmp validate`` reports the error and never runs the stub with an input."""
    d = S.doc("split_rect")
    d["boundaries"]["bad"] = {"between": ["slab_all", "b"]}
    log = tmp_path / "stub.log"
    stub = S.stub(tmp_path, "axmp-mesher", status=127, stderr=S.LOADER_GMSH, log=log)
    status, rep = S.run_json("validate", S.write(tmp_path, d), env=S.environment(AXMP_MESHER=str(stub)))
    assert status == 1 and rep["ok"] is False
    assert rep["errors"] == [BETWEEN_OVERLAP], json.dumps(rep["errors"], indent=1)
    assert all(a == "" for a in S.stub_runs(log)), S.stub_runs(log)
    S.assert_exception_is(S.raises(axmp.Design(d).validate), BETWEEN_OVERLAP)


def test_between_two_domains_with_a_common_region_is_refused_by_boundaries(tmp_path, mesher):
    d = S.doc("split_rect")
    d["boundaries"]["bad"] = {"between": ["slab_all", "b"]}
    rep = _refusal(tmp_path, d)
    assert rep["errors"] == [BETWEEN_OVERLAP]


PARTIAL = {
    "error": "boundary 'half' leaves out 2 edges that satisfy its where filter in part",
    "where": "/boundaries/half/where",
    "hint": ("2 edges, 0.08 m, first at (x, y) = (0.02, 0): a where filter keeps an edge only when the whole edge satisfies "
             "it. To select part of an edge, add a curve along that part and use {\"curve\": NAME}."),
    "code": "boundary.partial_match",
    "details": {"boundary": "half", "entities": 2, "measure_m": 0.08, "first_point": [0.02, 0]},
}


def test_a_partial_match_is_a_warning_with_the_marker_hint(tmp_path, mesher):
    """'half' keeps the inlet (x = 0) and leaves out the walls, whose boxes run past x = L/2: a warning, the command
    succeeds; the walls are not in 'half'."""
    d = S.doc("channel")
    d["boundaries"]["half"] = {"of": "channel", "where": {"x": [0, "L/2"]}}
    rep = _ok_report(tmp_path, d)
    partial = [w for w in rep["warnings"] if w["code"] == "boundary.partial_match"]
    assert partial == [PARTIAL], json.dumps(partial, indent=1)
    assert rep["boundaries"]["half"]["entities"] == 1 and _close(rep["boundaries"]["half"]["measure"], H, L)


def test_the_annulus_has_no_partial_match(tmp_path, mesher):
    """The annulus selects its inner circle by the circle's box: the outer circle's box encloses that box, and the
    outer circle does not pass through it. A curved edge whose box encloses the filter's box in every coordinate is not
    a partial match: its box cannot tell whether it passes through the filter's box (the hand-made table of the same
    annulus in cpp/tests/test_model_selection.cpp also checks a filter that sticks out of the box)."""
    rep = _ok_report(tmp_path, S.doc("annulus"))
    assert not [w for w in rep["warnings"] if w["code"] == "boundary.partial_match"], rep["warnings"]
    b = rep["boundaries"]
    assert b["inner"]["entities"] == 1 and _close(b["inner"]["measure"], 2 * math.pi * 0.002, 0.01)
    assert b["outer"]["entities"] == 1 and _close(b["outer"]["measure"], 2 * math.pi * 0.005, 0.01)
    assert all(e["type"] != "line" for e in rep["edges"])


def _trapezoid() -> dict:
    """A channel whose top runs straight from (0.04, 0.01) back to (0, 0.02): y = 0.02 - x / 4. 'cooled' is the bottom
    and, by a box on the middle of the top, the part of the top in that box."""
    d = S.doc("channel_post")
    d["name"] = "trapezoid"
    d["regions"] = [{"name": "channel", "material": "slab", "shape": {"polygon": [[0, 0], [0.04, 0], [0.04, 0.01], [0, 0.02]]}}]
    del d["domains"]
    d["boundaries"] = {"inlet": {"of": "channel", "where": {"x": 0}},
                       "outlet": {"of": "channel", "where": {"x": 0.04}},
                       "cooled": {"union": [{"of": "channel", "where": {"y": 0}},
                                            {"of": "channel", "where": {"x": [0.015, 0.025], "y": [0.012, 0.018]}}]},
                       "walls": {"of": "channel", "except": ["inlet", "outlet", "cooled"]}}
    d["physics"]["heat"]["domains"] = ["channel"]
    d["physics"]["heat"]["boundary_conditions"] = {"inlet": {"kind": "temperature", "T_K": 300.0},
                                                   "outlet": {"kind": "temperature", "T_K": 290.0},
                                                   "cooled": {"kind": "insulation"}, "walls": {"kind": "insulation"}}
    return d


def test_a_box_on_a_slanted_edge_is_a_partial_match(tmp_path, mesher):
    """A straight edge is decided exactly: the slanted top (y in [0.01375, 0.01625] for x in [0.015, 0.025]) passes
    through the second operand's box, which lies inside the top's own box. The box keeps no whole edge, so 'cooled' is
    the bottom alone, and the top, which the user meant in part, goes to 'walls': the warning says so."""
    rep = _ok_report(tmp_path, _trapezoid())
    b = rep["boundaries"]["cooled"]
    assert b["entities"] == 1 and _close(b["measure"], L, L), b
    partial = [w for w in rep["warnings"] if w["code"] == "boundary.partial_match"]
    assert len(partial) == 1, rep["warnings"]
    w = partial[0]
    S.assert_finding(w, code="boundary.partial_match", where="/boundaries/cooled",
                     message="boundary 'cooled' leaves out 1 edge that satisfies its where filter in part")
    top = math.hypot(0.04, 0.01)
    det = w["details"]
    assert det["boundary"] == "cooled" and det["entities"] == 1 and _close(det["measure_m"], top, L), det
    assert _close(det["first_point"][0], 0.02, L) and _close(det["first_point"][1], 0.015, L), det
    assert w["hint"].startswith("1 edge, ") and w["hint"].endswith(
        "a where filter keeps an edge only when the whole edge satisfies it. To select part of an edge, add a curve along "
        "that part and use {\"curve\": NAME}."), w["hint"]


RECT = [[0, 0], [0.04, 0], [0.04, 0.01], [0, 0.01]]
TRAP = [[0, 0], [0.04, 0], [0.04, 0.01], [0, 0.02]]   # the slanted top y = 0.02 - x / 4 from (0.04, 0.01) to (0, 0.02)
DIAM = [[0, 0], [0.02, 0], [0.04, 0.02], [0, 0.02]]   # a 45-degree edge from (0.02, 0) to (0.04, 0.02)


def _polygon_doc(polygon: list, probe: dict, symmetry: str = "planar") -> dict:
    """One polygon region 'channel', the boundary 'probe' and the rest of its boundary 'walls', a heat instance."""
    d = S.doc("channel_post")
    d["name"] = "polygon"
    d["symmetry"] = symmetry
    d["regions"] = [{"name": "channel", "material": "slab", "shape": {"polygon": polygon}}]
    del d["domains"]
    d["boundaries"] = {"probe": probe, "walls": {"of": "channel", "except": ["probe"]}}
    d["physics"]["heat"]["domains"] = ["channel"]
    d["physics"]["heat"]["boundary_conditions"] = {"probe": {"kind": "temperature", "T_K": 300.0},
                                                   "walls": {"kind": "insulation"}}
    d["mesh"] = {"size": 0.004}
    return d


def _probe(first: dict, where: dict, **extra) -> dict:
    """A union of an edge kept for sure and the filter, so that the selection is never empty."""
    return {"union": [{"of": "channel", "where": first}, {"of": "channel", "where": where, **extra}]}


def _partial(tmp_path: Path, d: dict, name: str) -> list:
    """The partial-match warnings of ``axmp boundaries``; the mesher's own resolution (``partial`` of its info file)
    must agree on 'probe' (the same selection code is compiled into both)."""
    out = tmp_path / (name + ".out")
    status, rep = _boundaries(tmp_path, d, "-o", out, name=name)
    assert status == 0 and rep["errors"] == [], json.dumps(rep, indent=1)[:3000]
    warnings = [w for w in rep["warnings"] if w["code"] == "boundary.partial_match"]
    mesher_partial = json.loads((out / "mesh_info.json").read_text()).get("partial", {})
    assert ("probe" in mesher_partial) == bool(warnings), (mesher_partial, warnings)
    return warnings


TOUCHING = {
    # the bottom and the slanted top end on the face x = 0.04
    "trapezoid_end_on_face": (TRAP, {"x": 0}, {"x": [0.04, 0.05]}, "planar"),
    # a box drawn exactly around the right edge; the slant ends at its corner (0.04, 0.01)
    "trapezoid_box_around_the_right_edge": (TRAP, {"x": 0}, {"x": [0.04, 1], "y": [0, 0.01]}, "planar"),
    # a corner of the box, (0.02, 0.015), lies on the slant
    "trapezoid_corner_on_the_slant": (TRAP, {"y": 0}, {"x": [0.02, 0.03], "y": [0.015, 0.02]}, "planar"),
    # the 45-degree edge and the top end at (0.04, 0.02), on the face x = 0.04
    "diamond_end_on_face": (DIAM, {"x": 0}, {"x": [0.04, 0.05]}, "planar"),
    # (r, z): the slant ends on r = 0.04
    "axisymmetric_end_on_face": (TRAP, {"z": 0}, {"r": [0.04, 1]}, "axisymmetric"),
    # the bottom and the top of a rectangle end on the face x = 0.04
    "rectangle_end_on_face": (RECT, {"x": 0}, {"x": [0.04, 0.05]}, "planar"),
}


@pytest.mark.parametrize("case", sorted(TOUCHING))
def test_an_edge_that_only_touches_the_filter_box_is_no_partial_match(tmp_path, mesher, case):
    """A straight edge that ends on a face of the filter's box, or meets one of its corners, has no part inside it:
    no warning (a constraint the edge's box satisfies does not cut the edge; the others cut it at the filter's own
    bounds; the part left must be longer than the tolerance). The library and the mesher decide alike."""
    polygon, first, where, symmetry = TOUCHING[case]
    d = _polygon_doc(polygon, _probe(first, where), symmetry)
    assert _partial(tmp_path, d, f"{case}.json") == []


def test_the_same_edge_is_decided_alike_in_two_documents(tmp_path, mesher):
    """The bottom edge (0, 0)-(0.04, 0) ends on the face x = 0.04 in a rectangle and in a trapezoid: the same answer
    in both (no partial match), whatever the rest of the geometry (the tolerance scales with its diagonal)."""
    for name, polygon in (("rect.json", RECT), ("trap.json", TRAP)):
        d = _polygon_doc(polygon, _probe({"x": 0}, {"x": [0.04, 0.05]}))
        status, rep = _boundaries(tmp_path, d, name=name)
        assert status == 0, rep
        bottom = [e for e in rep["edges"] if e["bbox"]["y"] == [0, 0]]
        assert len(bottom) == 1 and "probe" not in bottom[0]["names"], (name, bottom)
        assert [w for w in rep["warnings"] if w["code"] == "boundary.partial_match"] == [], (name, rep["warnings"])


def test_tolerance_m_is_a_tolerance_for_straight_edges(tmp_path, mesher):
    """With ``tolerance_m: 1e-6`` the bottom and the top of the rectangle entering the box x >= 0.04 - p by p = 0.5 um
    are within the tolerance: no warning. By p = 2 um they are partial matches: one warning for the two edges."""
    near = _polygon_doc(RECT, _probe({"x": 0}, {"x": [0.04 - 0.5e-6, 0.05]}, tolerance_m=1e-6))
    assert _partial(tmp_path, near, "near.json") == []
    far = _polygon_doc(RECT, _probe({"x": 0}, {"x": [0.04 - 2e-6, 0.05]}, tolerance_m=1e-6))
    w = _partial(tmp_path, far, "far.json")
    assert len(w) == 1 and w[0]["details"]["entities"] == 2, w
    assert _close(w[0]["details"]["measure_m"], 2 * L, L), w


@pytest.mark.parametrize("case", ["union", "difference", "nested"])
def test_between_overlap_is_found_inside_union_and_difference(tmp_path, case):
    """boundary.between_overlap at any depth of a union or a difference, with the pointer of the 'between'."""
    d = S.doc("split_rect")
    bad = {"between": ["slab_all", "b"]}
    if case == "union":
        d["boundaries"]["bad"] = {"union": [{"of": "b", "where": {"x": "L"}}, bad]}
        ptr = "/boundaries/bad/union/1/between"
    elif case == "difference":
        d["boundaries"]["bad"] = {"difference": [{"of": "b"}, bad]}
        ptr = "/boundaries/bad/difference/1/between"
    else:
        d["boundaries"]["bad"] = {"difference": [{"union": [{"of": "b"}, bad]}, "cold"]}
        ptr = "/boundaries/bad/difference/0/union/1/between"
    status, rep = S.run_json("validate", S.write(tmp_path, d))
    assert status == 1, rep
    assert [(f["code"], f["where"]) for f in rep["errors"]] == [("boundary.between_overlap", ptr)], rep["errors"]
    assert rep["errors"][0]["details"] == {"boundary": "bad", "between": ["slab_all", "b"], "shared": ["b"]}


def test_an_output_without_physics_belongs_to_the_instance_it_touches(tmp_path, mesher):
    """Two heat instances on a and b: a flux that names no physics, on b's right edge, is heat_b's (the instance of its
    quantity that the selection touches), so the full check accepts it; naming heat_a refuses it."""
    d = S.doc("split_two")
    d["outputs"] = {"Q_r": {"kind": "flux", "of": "heat", "on": "right"},
                    "Q_l": {"kind": "flux", "of": "heat", "on": "left"}}
    status, rep = S.run_json("boundaries", S.write(tmp_path, d), "--json")
    assert status == 0 and rep["errors"] == [], json.dumps(rep["errors"], indent=1)
    d["outputs"]["Q_r"]["physics"] = "heat_a"
    status, rep = S.run_json("boundaries", S.write(tmp_path, d), "--json")
    assert status == 1 and [f["code"] for f in rep["errors"]] == ["output.boundary_not_in_domain"], rep["errors"]


# ----------------------------------------------------------------------------------------- electromagnetic designs
EM_BOUNDARIES = ("boundaries lists the named boundaries of a model document (a design with a \"physics\" section); this is "
                 "an electromagnetic design")


def test_an_electromagnetic_design_is_refused_without_a_code(tmp_path):
    """As ``axmp explain`` refuses one: a message and a hint, no code; nothing is meshed."""
    status, out = S.run_json("boundaries", S.write(tmp_path, S.em_minimal(), "em.json"), "--json")
    assert status == 1
    assert set(out) == {"ok", "error", "where", "hint"} and out["ok"] is False, out
    assert out["error"] == EM_BOUNDARIES and out["where"] == "" and "`axmp resolve`" in out["hint"]
    r = S.run("boundaries", tmp_path / "em.json")
    assert r.returncode == 1 and json.loads(r.stdout) == out
    e = S.raises(axmp.Design(S.em_minimal()).boundaries)
    assert type(e) is _core.AxionMultiPhysicsError and e.code == "" and e.details is None
    assert str(e) == f"{EM_BOUNDARIES} Hint: {out['hint']}"
