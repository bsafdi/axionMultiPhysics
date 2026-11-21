"""``axmp mesh`` and ``_core.mesh`` of a model document, ``axmp.mesh(d)`` with its meshed model and instance meshes
(design note 5.4, 12.1), the refusals before meshing, the failed mesher run, and the shapes that
the bounded modes cannot mesh.

``axmp mesh model.json -o DIR [--mesher M] [-v N]`` of a model document validates it with the mesher's probe (errors
refuse; ``route.*`` unavailable findings do not; ``build.requires_mesher`` does), runs the ``resolve_only`` pass and the
full check, then meshes; it writes ``DIR/mesh_in.json``, ``DIR/mesh.msh``, ``DIR/mesh_info.json`` (and the mesher's log
``DIR/mesh_mesher.log``) and prints ``{"ok", "name", "symmetry", "geometry_order", "mesh", "mesher_input",
"mesher_info", "boundaries", "warnings"}``: the paths, each boundary's ``measure`` and ``measure_units`` (and
``area_m2`` in an axisymmetric model; the master numbers are in the info file only) and the warnings (validation's and
the full check's). ``_core.mesh`` returns the same JSON (one library function serves both). A refusal is the command
line's error JSON and exit status 1; ``_core.mesh`` and ``axmp.mesh`` raise it.

``axmp.mesh(d, workdir=None, mesher=None)`` returns a ``MeshedModel``: ``symmetry``, ``geometry_order``, ``regions``
(name -> ``{"area_m2"}``, with ``"volume_m3"`` when axisymmetric), ``boundaries`` (the table of ``Design.boundaries()``),
``mesh_file`` and ``info_file`` (None without a workdir), and ``instance(domain=None)`` (None: every enabled region; a
region or domain name, or a list of them). It has no attribute map and no ``info``. An ``InstanceMesh`` has ``regions``,
``geometry_order``, ``has_nodes``, ``attributes`` (name -> list of instance attributes), ``remaining_attributes``,
``axis_attributes``, ``master_attribute`` (instance attribute -> master number), ``region_attribute``, the arrays
``vertices()`` (nv, 2), ``elements()`` (ne, 3), ``element_attributes()`` (ne,), ``boundary()`` (nbe, 2),
``boundary_attributes()`` (nbe,), ``section(name)`` (``{"name", "faces", "sign", "length_m"}``) and
``face_points(name=None, *, order, attribute=None)``: a dict of NumPy arrays ``faces`` (m,), ``elements`` (m,), ``ref``
(m, q, 2), ``weights`` (m, q), ``normals`` (m, q, 2), ``points`` (m, q, 2), with exactly one of ``name`` and
``attribute``. A misuse (an unknown name, an attribute not on the instance mesh, both or neither of name and
attribute) is a ``ValueError``.

Independent references: the analytic geometry of the fixtures (lengths, areas, normals of the split rectangle: the
marker written bottom to top has the normal -e_x, the interface between a and b +e_x); the master numbers of design
note 5.3 worked out by hand (as in cpp/tests/test_model_mesh_mesher.cpp); the texts of design note 5.5; the element
types of the mesh file, read here.
"""
from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core

import model_mesh_support as S
from model_mesh_support import mesher  # noqa: F401 (a fixture)

L, H = 0.04, 0.01
MESH_KEYS = {"ok", "name", "symmetry", "geometry_order", "mesh", "mesher_input", "mesher_info", "boundaries", "warnings"}
FP_KEYS = {"faces", "elements", "ref", "weights", "normals", "points"}
MESH_FILES = ("mesh_in.json", "mesh.msh", "mesh_info.json")


def _no_mesh_files(directory: Path) -> None:
    for f in MESH_FILES:
        assert not (directory / f).exists(), f"{f} was written"


def _close(a, b, scale, rel=1e-12) -> bool:
    return abs(a - b) <= rel * scale


def _covered() -> dict:
    """Region 'a' lies under region 'b', painted after it: the mesher refuses it."""
    d = S.doc("split_rect")
    d["name"] = "covered"
    d["regions"] = [{"name": "a", "shape": {"rect": {"x": [0, "L/2"], "y": [0, "H"]}}, "material": "slab"},
                    {"name": "b", "shape": {"rect": {"x": [0, "L"], "y": [0, "H"]}}, "material": "slab"}]
    del d["curves"], d["outputs"], d["domains"]
    d["boundaries"] = {"hot": {"of": "b", "where": {"x": 0}}, "cold": {"of": "b", "where": {"x": "L"}}}
    d["physics"]["heat"]["domains"] = ["a", "b"]
    return d


COVERED_REASON = "region 'a' is completely covered by later regions; reorder or remove it"


def _marker_down() -> dict:
    d = S.doc("split_rect")
    d["curves"][0]["path"] = {"segment": [["L/4", "H"], ["L/4", 0]]}
    return d


# ------------------------------------------------------------------------------------------------------ axmp mesh
def test_mesh_writes_the_files_and_prints_the_summary(tmp_path, mesher):
    d = S.doc("split_rect")
    out_dir = tmp_path / "m"
    status, out = S.run_json("mesh", S.write(tmp_path, d), "-o", out_dir)
    assert status == 0, out
    assert set(out) == MESH_KEYS, sorted(out)
    assert out["ok"] is True and out["name"] == "split_rect" and out["symmetry"] == "planar" and out["geometry_order"] == 2
    assert Path(out["mesh"]) == out_dir / "mesh.msh"
    assert Path(out["mesher_input"]) == out_dir / "mesh_in.json"
    assert Path(out["mesher_info"]) == out_dir / "mesh_info.json"
    for f in MESH_FILES + ("mesh_mesher.log",):
        assert (out_dir / f).is_file(), f
    assert set(out["boundaries"]) == {"cold", "hot", "interface", "mk"}
    for name, row in out["boundaries"].items():
        assert set(row) == {"measure", "measure_units"} and row["measure_units"] == "m", row
        assert _close(row["measure"], H, L), (name, row)
    # the warnings: those of the validation (the beta labels), each a finding
    status, val = S.run_json("validate", S.write(tmp_path, d))
    assert [w["code"] for w in out["warnings"]] == [w["code"] for w in val["warnings"]]
    assert S.keys_anywhere(out, S.NUMBER_KEYS) == []
    # the master numbers are in the info file; the input is the bounded mode's, meshing
    info = json.loads((out_dir / "mesh_info.json").read_text())
    assert sorted(g["attr"] for g in info["boundary_groups"]) == [2001, 2002, 2003, 2004, 2005, 2006]
    inp = json.loads((out_dir / "mesh_in.json").read_text())
    assert inp["bounded"] is True and inp["resolve_only"] is False
    types = S.element_types(out_dir / "mesh.msh")
    assert 9 in types and 8 in types and not types & {1, 2}, types   # geometry order 2: 6-node triangles, 3-node lines


def test_mesh_of_the_axisymmetric_pipe_gives_the_revolved_areas(tmp_path, mesher):
    R, Lz = 0.005, 0.04
    status, out = S.run_json("mesh", S.write(tmp_path, S.doc("pipe_axi")), "-o", tmp_path / "m")
    assert status == 0, out
    assert out["symmetry"] == "axisymmetric"
    b = out["boundaries"]
    assert set(b) == {"inlet", "outlet", "wall"}
    assert all(set(row) == {"measure", "measure_units", "area_m2"} for row in b.values()), b
    assert _close(b["wall"]["measure"], Lz, Lz) and _close(b["wall"]["area_m2"], 2 * math.pi * R * Lz, Lz * Lz)
    assert _close(b["inlet"]["measure"], R, Lz) and _close(b["inlet"]["area_m2"], math.pi * R * R, Lz * Lz)


def test_core_mesh_returns_what_the_command_prints(tmp_path, mesher):
    """One library function serves both: the same JSON, but for the paths, and the same files. The
    channel's flow route is not implemented in this version, and that does not stop the meshing."""
    d = S.doc("channel")
    status, cli = S.run_json("mesh", S.write(tmp_path, d), "-o", tmp_path / "c")
    assert status == 0, cli
    py = json.loads(_core.mesh(json.dumps(d), str(tmp_path / "p"), 0))
    for k, f in (("mesh", "mesh.msh"), ("mesher_input", "mesh_in.json"), ("mesher_info", "mesh_info.json")):
        assert Path(cli[k]) == tmp_path / "c" / f and Path(py[k]) == tmp_path / "p" / f
        assert (tmp_path / "p" / f).is_file()
        cli[k] = py[k] = None
    assert py == cli


def test_a_p1_annulus_is_meshed_at_geometry_order_1(tmp_path, mesher):
    """A P1 current distribution lowers the default geometry order to 1 (mesh.geometry_order_lowered); with
    ``mesh.geometry_order: 2`` the document is refused before the mesher runs, and nothing is written."""
    d = S.doc("annulus_cell")
    status, out = S.run_json("mesh", S.write(tmp_path, d), "-o", tmp_path / "m")
    assert status == 0, out
    assert out["geometry_order"] == 1
    assert "mesh.geometry_order_lowered" in [w["code"] for w in out["warnings"]]
    types = S.element_types(tmp_path / "m" / "mesh.msh")
    assert 2 in types and 1 in types and not types & {8, 9}, types   # straight: 3-node triangles, 2-node lines
    d["mesh"]["geometry_order"] = 2
    status, out = S.run_json("mesh", S.write(tmp_path, d, "order2.json"), "-o", tmp_path / "n")
    assert status == 1 and out["ok"] is False and out["code"] == "model.geometry_order_exceeds_field_order", out
    _no_mesh_files(tmp_path / "n")


def _uncovered_channel() -> dict:
    d = S.doc("channel")
    del d["boundaries"]["walls"]
    del d["physics"]["flow"]["boundary_conditions"]["walls"]
    del d["outputs"]
    return d


def test_an_uncovered_boundary_is_refused_before_meshing(tmp_path, mesher):
    """The full check runs before meshing (design note 5.5): no mesh file is written, and the command, ``_core.mesh``
    and ``axmp.mesh`` refuse alike (the command names the document as given; the Python API writes DOCUMENT)."""
    d = _uncovered_channel()
    S.write(tmp_path, d, "channel.json")
    status, out = S.run_json("mesh", "channel.json", "-o", "m", cwd=tmp_path)
    assert status == 1 and out["ok"] is False, out
    assert out["code"] == "physics.boundary_uncovered" and out["where"] == "/physics/flow/boundary_conditions"
    assert out["error"] == "physics 'flow': part of the boundary of its domain has no condition"
    assert "`axmp boundaries channel.json` lists every boundary." in out["hint"]
    assert not (tmp_path / "m" / "mesh.msh").exists()
    want = {k: v for k, v in out.items() if k != "ok"}
    want["hint"] = want["hint"].replace("`axmp boundaries channel.json`", "`axmp boundaries DOCUMENT`")
    S.assert_exception_is(S.raises(_core.mesh, json.dumps(d), str(tmp_path / "p"), 0), want)
    assert not (tmp_path / "p" / "mesh.msh").exists()
    S.assert_exception_is(S.raises(axmp.mesh, axmp.Design(d)), want)


def test_a_geometry_free_model_prints_a_note(tmp_path):
    """A reactor has no geometry: nothing to mesh, ``ok`` with a note (as an electromagnetic design on an imported mesh
    prints one); no mesher is needed for it."""
    d = {"name": "r", "physics": {"rx": {"type": "chem.reactor"}},
         "study": [{"id": "run", "type": "transient", "time": {"t_end_s": 10}}]}
    status, out = S.run_json("mesh", S.write(tmp_path, d), "-o", tmp_path / "m")
    assert status == 0, out
    assert out["ok"] is True and out["name"] == "r" and "no geometry" in out["note"], out
    assert "mesh" not in out
    _no_mesh_files(tmp_path / "m")
    assert json.loads(_core.mesh(json.dumps(d), str(tmp_path / "p"), 0)) == out


def test_a_3d_model_is_refused_with_route_not_implemented(tmp_path, mesher):
    """Bounded 3D meshes are not implemented yet; ``axmp mesh`` and ``axmp boundaries`` refuse, writing
    nothing."""
    d = S.doc("busbar")
    path = S.write(tmp_path, d)
    status, out = S.run_json("mesh", path, "-o", tmp_path / "m")
    assert status == 1 and out["code"] == "route.not_implemented" and out["where"] == "/symmetry", out
    assert "bounded 3D meshes are not implemented yet" in out["error"]
    _no_mesh_files(tmp_path / "m")
    status, rep = S.run_json("boundaries", path, "--json", "-o", tmp_path / "b")
    assert status == 1 and rep["ok"] is False and rep["code"] == "route.not_implemented", rep
    _no_mesh_files(tmp_path / "b")
    e = S.raises(_core.mesh, json.dumps(d), str(tmp_path / "p"), 0)
    assert e.code == "route.not_implemented" and e.where == "/symmetry"


def test_a_model_on_an_imported_mesh_is_refused_in_this_version(tmp_path):
    """Model documents on ``mesh.file`` are meshed (read) in a later version: ``axmp mesh`` and ``axmp boundaries``
    refuse them with route.not_implemented at /mesh/file; they do not mesh the regions instead."""
    d = S.doc("channel")
    del d["regions"]
    d["mesh"] = {"file": "chan.msh", "info_file": "chan.json"}
    d["boundaries"] = {"inlet": {"group": "inlet"}, "outlet": {"group": "outlet"}, "walls": {"group": "walls"}}
    del d["outputs"]["F_walls"]
    path = S.write(tmp_path, d)
    for args in (("mesh", path, "-o", tmp_path / "m"), ("boundaries", path, "--json", "-o", tmp_path / "m")):
        status, out = S.run_json(*args)
        assert status == 1 and out["ok"] is False, out
        assert out["code"] == "route.not_implemented" and out["where"] == "/mesh/file", out
        assert "imported mesh" in out["error"], out
    _no_mesh_files(tmp_path / "m")


# ------------------------------------------------------------------------------------------- the mesher's failures
def test_a_failed_mesher_run_is_mesh_generation_failed(tmp_path, mesher):
    """The mesher's text in the message and in ``details.reason``, the log's path in the hint; through
    ``axmp mesh``, ``axmp boundaries``, ``_core.mesh``, ``axmp.mesh`` and ``Design.boundaries``."""
    d = _covered()
    path = S.write(tmp_path, d)
    status, out = S.run_json("mesh", path, "-o", tmp_path / "m")
    assert status == 1 and out["ok"] is False, out
    assert out["code"] == "mesh.generation_failed" and out["where"] == "/regions", out
    assert out["error"] == "mesh generation failed: " + COVERED_REASON
    assert out["details"]["reason"] == COVERED_REASON
    log = tmp_path / "m" / "mesh_mesher.log"
    assert str(log) in out["hint"] and log.is_file()
    assert not (tmp_path / "m" / "mesh.msh").exists()
    status, rep = S.run_json("boundaries", path, "--json", "-o", tmp_path / "b")
    assert status == 1 and rep["ok"] is False and rep["code"] == "mesh.generation_failed", rep
    assert rep["details"]["reason"] == COVERED_REASON and str(tmp_path / "b" / "mesh_mesher.log") in rep["hint"]
    for e in (S.raises(_core.mesh, json.dumps(d), str(tmp_path / "p"), 0), S.raises(axmp.mesh, axmp.Design(d)),
              S.raises(axmp.Design(d).boundaries)):
        assert type(e) is _core.AxionMultiPhysicsError and e.code == "mesh.generation_failed", e
        assert e.where == "/regions" and e.details["reason"] == COVERED_REASON


_TAMPER = """import json, subprocess, sys
rc = subprocess.call([{real!r}, *sys.argv[1:]])
if rc == 0 and len(sys.argv) > 1:
    inp = json.load(open(sys.argv[1]))
    if not inp.get("resolve_only"):
        info = json.load(open(inp["info"]))
        named = [g for g in info["boundary_groups"] if g["type"] == "named"]
        named[0]["names"], named[1]["names"] = named[1]["names"], named[0]["names"]
        json.dump(info, open(inp["info"], "w"))
sys.exit(rc)
"""


def test_mesher_files_that_disagree_with_the_resolution_are_an_internal_error(tmp_path, mesher):
    """A mesher whose info file names the atoms differently from the library's resolution (here a wrapper around the
    real mesher swaps the names of two atoms after the meshing run) is a defect of the program: internal.mesh_export,
    which asks for a bug report, through ``axmp mesh`` and ``axmp.mesh`` (so that code is raised in this version)."""
    script = tmp_path / "tamper.py"
    script.write_text(_TAMPER.format(real=mesher))
    wrapper = tmp_path / "tamper-mesher"
    wrapper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{script}' \"$@\"\n")
    wrapper.chmod(0o755)
    d = S.doc("split_rect")
    status, out = S.run_json("mesh", S.write(tmp_path, d), "-o", tmp_path / "m", "--mesher", wrapper)
    assert status == 1 and out["code"] == "internal.mesh_export", out
    assert "report" in out["hint"]
    e = S.raises(axmp.mesh, axmp.Design(d), mesher=str(wrapper))
    assert type(e) is _core.AxionMultiPhysicsError and e.code == "internal.mesh_export", e


_MOVE_MARKER = """import json, subprocess, sys
args = sys.argv[1:]
if args:
    inp = json.load(open(args[0]))
    if not inp.get("resolve_only"):
        for c in inp.get("curves", []):
            c["path"]["segment"] = [[x + {dx!r}, y] for x, y in c["path"]["segment"]]
        moved = args[0] + ".moved.json"
        json.dump(inp, open(moved, "w"))
        args = [moved] + args[1:]
sys.exit(subprocess.call([{real!r}, *args]))
"""


def test_a_meshing_run_that_resolves_other_atoms_than_the_resolve_pass_is_an_internal_error(tmp_path, mesher):
    """The meshing run must resolve the atoms that the resolve_only pass checked. A wrapper around the real mesher moves
    the marker of split_rect from a into b for the meshing run only: its files agree with each other and with the
    document's selectors, so only the comparison of the two runs' atoms (the unnamed atoms of a and b now have other
    entities) can refuse it, with internal.mesh_export."""
    script = tmp_path / "move_marker.py"
    script.write_text(_MOVE_MARKER.format(real=mesher, dx=L / 2))
    wrapper = tmp_path / "moving-mesher"
    wrapper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{script}' \"$@\"\n")
    wrapper.chmod(0o755)
    d = S.doc("split_rect")
    status, out = S.run_json("mesh", S.write(tmp_path, d), "-o", tmp_path / "m", "--mesher", wrapper)
    assert status == 1 and out.get("code") == "internal.mesh_export", out
    assert "report" in out["hint"]
    e = S.raises(axmp.mesh, axmp.Design(d), mesher=str(wrapper))
    assert type(e) is _core.AxionMultiPhysicsError and e.code == "internal.mesh_export", e


def test_mesh_prints_the_warnings_of_the_full_check(tmp_path, mesher):
    """``axmp mesh`` prints the validation's warnings, then the full check's (a partial match)."""
    d = S.doc("channel")
    d["boundaries"]["half"] = {"of": "channel", "where": {"x": [0, "L/2"]}}
    status, out = S.run_json("mesh", S.write(tmp_path, d), "-o", tmp_path / "m")
    assert status == 0, out
    codes = [w["code"] for w in out["warnings"]]
    assert codes[-1] == "boundary.partial_match" and codes.count("boundary.partial_match") == 1, codes
    _, val = S.run_json("validate", S.write(tmp_path, d))
    assert codes[:-1] == [w["code"] for w in val["warnings"]], (codes, val["warnings"])


def test_a_refused_mesh_leaves_no_mesh_in_its_directory(tmp_path, mesher):
    """A refusal after the resolve pass (an uncovered boundary) may leave that pass's files in -o DIR, but no mesh: the
    mesh of an earlier run in DIR is removed, and the input left there is the resolve pass's."""
    out_dir = tmp_path / "m"
    status, out = S.run_json("mesh", S.write(tmp_path, S.doc("split_rect")), "-o", out_dir)
    assert status == 0 and (out_dir / "mesh.msh").is_file(), out
    status, out = S.run_json("mesh", S.write(tmp_path, _uncovered_channel(), "u.json"), "-o", out_dir)
    assert status == 1 and out["code"] == "physics.boundary_uncovered", out
    assert not (out_dir / "mesh.msh").exists()
    if (out_dir / "mesh_in.json").exists():
        assert json.loads((out_dir / "mesh_in.json").read_text())["resolve_only"] is True


def _refused_before_the_resolve_pass(tmp_path: Path, case: str) -> tuple:
    """A document and the extra arguments of an ``axmp mesh`` refused before the resolve pass."""
    if case == "validation":
        d = S.doc("split_rect")
        d["regions"][0]["material"] = "no_such_material"
        return d, ()
    if case == "requires_mesher":
        stub = S.stub(tmp_path, "axmp-mesher", status=127, stderr=S.LOADER_GMSH)
        return S.doc("split_rect"), ("--mesher", stub)
    return S.doc("busbar"), ()   # 3D: refused before any mesher run


@pytest.mark.parametrize("case", ["validation", "requires_mesher", "3d"])
def test_a_refusal_before_the_resolve_pass_leaves_no_earlier_mesh(tmp_path, mesher, case):
    """Every ``axmp mesh`` of a model document starts by removing the mesh, the info file and the input of an earlier
    run in -o DIR: a refusal at validation, by build.requires_mesher or of a 3D document leaves none of them behind."""
    out_dir = tmp_path / "m"
    status, out = S.run_json("mesh", S.write(tmp_path, S.doc("split_rect")), "-o", out_dir)
    assert status == 0 and all((out_dir / f).is_file() for f in MESH_FILES), out
    d, extra = _refused_before_the_resolve_pass(tmp_path, case)
    status, out = S.run_json("mesh", S.write(tmp_path, d, "refused.json"), "-o", out_dir, *extra)
    assert status == 1 and out["ok"] is False, out
    _no_mesh_files(out_dir)


def test_face_points_takes_numpy_integers(tmp_path, mesher):
    """An attribute read from an array is a NumPy integer: it is taken as the int it is."""
    im = axmp.mesh(axmp.Design(S.doc("split_rect"))).instance()
    a = im.attributes["hot"][0]
    want = im.face_points(attribute=a, order=4)
    for n in (np.int64(a), np.int32(a), np.uint8(a), im.boundary_attributes()[im.boundary_attributes() == a][0]):
        got = im.face_points(attribute=n, order=4)
        assert all(np.array_equal(got[k], want[k]) for k in want), type(n)


def test_core_mesh_without_a_workdir_names_no_removed_file(mesher):
    """Without a workdir the files are meshed in a temporary directory that is removed: the paths are null, or name
    files that exist."""
    out = json.loads(_core.mesh(json.dumps(S.doc("split_rect")), "", 0))
    assert out["ok"] is True, out
    for k in ("mesh", "mesher_input", "mesher_info"):
        assert out[k] is None or Path(out[k]).is_file(), (k, out[k])


def test_axmp_mesh_of_a_model_without_a_geometry_is_refused():
    """``axmp.mesh`` returns a meshed model, and a document with no geometry (a reactor) has none: it raises the
    unavailable refusal of the documents it does not mesh, route.not_implemented, at /regions (the command prints a
    note instead)."""
    d = {"name": "r", "physics": {"rx": {"type": "chem.reactor"}},
         "study": [{"id": "run", "type": "transient", "time": {"t_end_s": 10}}]}
    e = S.raises(axmp.mesh, axmp.Design(d))
    assert type(e) is _core.AxionMultiPhysicsError, f"{type(e).__name__}: {e}"
    assert e.code == "route.not_implemented" and e.where == "/regions", (e.code, e.where)
    assert "no geometry" in str(e), str(e)


# ------------------------------------------------------------------------------------------------------ axmp.mesh
def test_axmp_mesh_returns_the_meshed_model(tmp_path, mesher):
    d = axmp.Design(S.doc("split_rect"))
    m = axmp.mesh(d, workdir=tmp_path / "w")
    assert type(m).__name__ == "MeshedModel" and type(m).__module__ == "axionmultiphysics.model_mesh"
    assert m.symmetry == "planar" and m.geometry_order == 2
    assert list(m.regions) == ["a", "b"]
    for r in ("a", "b"):
        assert set(m.regions[r]) == {"area_m2"} and _close(m.regions[r]["area_m2"], L / 2 * H, L * H), m.regions
    assert m.boundaries == d.boundaries()["boundaries"]
    assert m.mesh_file == str(tmp_path / "w" / "mesh.msh") and Path(m.mesh_file).is_file()
    assert m.info_file == str(tmp_path / "w" / "mesh_info.json") and Path(m.info_file).is_file()
    # no attribute map, no info file contents on the meshed model (design note 12.1)
    for name in ("info", "attributes", "master_attribute", "remaining_attributes", "axis_attributes"):
        assert not hasattr(m, name), name
    assert S.keys_anywhere(m.boundaries, S.NUMBER_KEYS) == []
    t = axmp.mesh(S.doc("split_rect"))   # a dict, no workdir: a temporary directory
    assert t.mesh_file is None and t.info_file is None and t.geometry_order == 2


SPLIT_AB = {"master_attribute": {1: 2001, 2: 2002, 3: 2005, 4: 2006},
            "attributes": {"cold": [1], "hot": [2], "interface": [], "mk": []},
            "remaining_attributes": [3, 4], "axis_attributes": [], "region_attribute": {"a": 1, "b": 2},
            "regions": ["a", "b"]}
SPLIT_A = {"master_attribute": {1: 2002, 2: 2003, 3: 2005},
           "attributes": {"cold": [], "hot": [1], "interface": [2], "mk": []},
           "remaining_attributes": [3], "axis_attributes": [], "region_attribute": {"a": 1}, "regions": ["a"]}


def _maps(im) -> dict:
    return {k: getattr(im, k) for k in SPLIT_AB}


def test_instance_meshes_and_their_maps(tmp_path, mesher):
    """The instance meshes of split_rect (spot checks of the maps that cpp/tests/test_model_mesh_mesher.cpp checks): every region by default, by region
    or domain name, by a list; the named lists, the remaining and the axis attributes partition 1..n; a disabled, an
    unknown or no region is a ValueError."""
    m = axmp.mesh(axmp.Design(S.doc("split_rect")))
    im = m.instance()
    assert type(im).__name__ == "InstanceMesh"
    assert _maps(im) == SPLIT_AB
    assert im.geometry_order == 2 and im.has_nodes is True
    for dom in (["a", "b"], "slab_all", ["left", "b"]):
        assert _maps(m.instance(dom)) == SPLIT_AB, dom
    assert _maps(m.instance("a")) == SPLIT_A and _maps(m.instance("left")) == SPLIT_A
    n = len(im.master_attribute)
    named = set().union(*map(set, im.attributes.values()))
    assert named | set(im.remaining_attributes) | set(im.axis_attributes) == set(range(1, n + 1))
    assert not named & set(im.remaining_attributes)
    for bad in ("nope", "c", []):
        with pytest.raises(ValueError):
            m.instance(bad)


def test_the_arrays_of_an_instance_mesh(tmp_path, mesher):
    """The arrays tie the maps to the mesh: the boundary attributes are 1..n, the element attributes the regions', the
    segments of 'hot' have the length H and the triangles of each region its area (straight edges: chords exact)."""
    im = axmp.mesh(axmp.Design(S.doc("split_rect"))).instance()
    V, E, EA = im.vertices(), im.elements(), im.element_attributes()
    B, BA = im.boundary(), im.boundary_attributes()
    assert V.ndim == 2 and V.shape[1] == 2 and V.dtype == np.float64
    assert E.shape == (len(EA), 3) and B.shape == (len(BA), 2)
    assert E.min() >= 0 and E.max() < len(V) and B.min() >= 0 and B.max() < len(V)
    assert set(EA.tolist()) == {1, 2} and set(BA.tolist()) == {1, 2, 3, 4}
    hot = B[BA == im.attributes["hot"][0]]
    assert _close(np.linalg.norm(V[hot[:, 1]] - V[hot[:, 0]], axis=1).sum(), H, L)
    for region, attr in im.region_attribute.items():
        t = V[E[EA == attr]]
        u, v = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()
        assert _close(area, L / 2 * H, L * H), region


def test_the_arrays_are_copies_that_outlive_the_meshes(tmp_path, mesher):
    """The NumPy arrays own a copy of the data: they survive the instance and the meshed model, and writing into one
    does not change the mesh."""
    m = axmp.mesh(S.doc("split_rect"))
    im = m.instance()
    arrays = {k: getattr(im, k)() for k in ("vertices", "elements", "element_attributes", "boundary", "boundary_attributes")}
    fp = im.face_points("mk", order=3)
    copies = {k: a.copy() for k, a in arrays.items()}
    fpc = {k: a.copy() for k, a in fp.items()}
    del m, im
    gc.collect()
    _ = [np.ones(5000) for _ in range(500)]
    other = axmp.mesh(S.doc("pipe_axi")).instance()
    assert all(np.array_equal(arrays[k], copies[k]) for k in arrays)
    assert all(np.array_equal(fp[k], fpc[k]) for k in fp)
    v = other.vertices()
    v[:] = -1.0
    assert other.vertices().min() >= 0.0


def test_face_points_and_sections(tmp_path, mesher):
    """The flux instrument: by name on the instance boundary (outward normals) and on a section (the
    orientation of design note 5.2), and by instance attribute (the way to an unnamed atom)."""
    m = axmp.mesh(axmp.Design(S.doc("split_rect")))
    im = m.instance()
    fp = im.face_points("hot", order=4)
    assert set(fp) == FP_KEYS
    k, q = len(fp["faces"]), fp["weights"].shape[1]
    assert k > 0 and fp["weights"].shape == (k, q)
    assert fp["ref"].shape == fp["normals"].shape == fp["points"].shape == (k, q, 2)
    assert fp["elements"].shape == (k,) and fp["elements"].min() >= 0 and fp["elements"].max() < len(im.elements())
    assert np.all(np.diff(fp["faces"]) > 0), "faces ascending"
    w, n, x = fp["weights"], fp["normals"], fp["points"]
    assert _close(w.sum(), H, H) and _close((w * n[..., 0]).sum(), -H, H) and abs((w * n[..., 1]).sum()) <= 1e-12 * H
    assert np.abs(x[..., 0]).max() <= 1e-12 * L
    # sections: the interface between a and b points from a to b (+e_x), the marker written bottom to top to its left
    flux = lambda f: (f["weights"] * f["normals"][..., 0]).sum()  # noqa: E731
    assert _close(flux(im.face_points("interface", order=4)), H, H)
    assert _close(flux(im.face_points("mk", order=4)), -H, H)
    sec = im.section("mk")
    assert set(sec) == {"name", "faces", "sign", "length_m"} and sec["name"] == "mk"
    assert _close(sec["length_m"], H, H) and set(np.asarray(sec["sign"]).tolist()) <= {-1, 1}
    assert np.array_equal(np.asarray(sec["faces"]), im.face_points("mk", order=4)["faces"])
    assert _close(im.section("interface")["length_m"], H, H)
    # by attribute: the same faces as by name; the unnamed atoms (top and bottom) have the length 2 L
    fa = im.face_points(attribute=im.attributes["hot"][0], order=4)
    assert np.array_equal(fa["faces"], fp["faces"]) and np.allclose(fa["normals"], fp["normals"], rtol=0, atol=1e-14)
    rem = sum(im.face_points(attribute=a, order=4)["weights"].sum() for a in im.remaining_attributes)
    assert _close(rem, 2 * L, L)
    # misuse: ValueError, no code
    n_attr = len(im.master_attribute)
    for call in (lambda: im.face_points(order=4), lambda: im.face_points("hot", order=4, attribute=2),
                 lambda: im.face_points(attribute=0, order=4), lambda: im.face_points(attribute=n_attr + 1, order=4),
                 lambda: im.face_points(attribute=2001, order=4), lambda: im.face_points("nope", order=4),
                 lambda: im.face_points(attribute=1.7, order=4), lambda: im.face_points(attribute=2**40, order=4),
                 lambda: m.instance("a").face_points("cold", order=4), lambda: im.section("hot"),
                 lambda: im.section("nope")):
        with pytest.raises(ValueError):
            call()
    # the instance mesh owns what it needs
    del m
    gc.collect()
    assert _close(im.face_points("hot", order=4)["weights"].sum(), H, H)


def test_the_axis_of_the_pipe_comes_first(tmp_path, mesher):
    R, Lz = 0.005, 0.04
    m = axmp.mesh(axmp.Design(S.doc("pipe_axi")))
    assert m.symmetry == "axisymmetric"
    assert set(m.regions["p1"]) == {"area_m2", "volume_m3"}
    assert _close(m.regions["p1"]["area_m2"], R * Lz / 2, R * Lz)
    assert _close(m.regions["p1"]["volume_m3"], math.pi * R * R * Lz / 2, R * R * Lz)
    im = m.instance("pipe")
    assert im.axis_attributes == [1] and im.master_attribute[1] == 900
    fp = im.face_points(attribute=1, order=2)
    assert np.abs(fp["points"][..., 0]).max() <= 1e-12 * Lz and _close(fp["weights"].sum(), Lz, Lz)


def test_a_reused_workdir_is_meshed_again(tmp_path, mesher):
    """No API of this change reads a workdir it did not just write (the stale-workdir refusal is left to the first
    one that does): meshing into a workdir of an earlier document gives the new document's mesh (the reversed marker's
    normal), and a failed run leaves no mesh file and no info file of the earlier run behind (the mesher may write its
    own info file of the failure)."""
    w = tmp_path / "w"
    up = axmp.mesh(axmp.Design(S.doc("split_rect")), workdir=w).instance().face_points("mk", order=4)
    assert _close((up["weights"] * up["normals"][..., 0]).sum(), -H, H)
    down = axmp.mesh(axmp.Design(_marker_down()), workdir=w).instance().face_points("mk", order=4)
    assert _close((down["weights"] * down["normals"][..., 0]).sum(), H, H)
    e = S.raises(axmp.mesh, axmp.Design(_covered()), workdir=w)
    assert getattr(e, "code", None) == "mesh.generation_failed", e
    assert not (w / "mesh.msh").exists()
    assert not (w / "mesh_info.json").exists() or json.loads((w / "mesh_info.json").read_text())["ok"] is False


EM_MESH = ("axmp.mesh meshes a model document (a design with a \"physics\" section); this is an electromagnetic design")


def test_axmp_mesh_refuses_an_electromagnetic_design_without_a_code():
    """As ``axmp explain`` refuses one. ``axmp mesh`` and ``_core.mesh`` of an electromagnetic design are unchanged (G0 tier A)."""
    e = S.raises(axmp.mesh, axmp.Design(S.em_minimal()))
    assert type(e) is _core.AxionMultiPhysicsError and e.code == "" and e.where == "" and e.details is None
    assert str(e).startswith(EM_MESH + " Hint: ") and "`axmp mesh`" in e.hint


def test_the_package_exports_mesh():
    assert "mesh" in axmp.__all__ and callable(axmp.mesh)
    from axionmultiphysics import model_mesh

    assert axmp.mesh is model_mesh.mesh


# ------------------------------------------------------------------------- shapes the bounded modes mesh since the
# partition: the curves OpenCASCADE leaves trimmed
def test_a_union_with_a_filleted_rectangle_meshes_in_the_bounded_mode(tmp_path, mesher):
    """The union of a rectangle with a filleted rectangle on its side (the way the electromagnetic templates build a
    cavity with a neck) leaves trimmed curves in the geometry kernel: the bounded mode resolves them to their lines and
    arcs. The union is the rectangle again: its area is 0.04 x 0.2 m."""
    d = S.doc("pipe_axi")
    d["regions"] = [{"name": "p1", "material": "water", "shape": {"union": [
        {"rect": {"r": [0.01, 0.05], "z": [0.0, 0.2]}},
        {"rect": {"r": [0.01, 0.02], "z": [0.05, 0.1], "fillet": 0.003}}]}}]
    d["domains"] = {"pipe": ["p1"]}
    d["boundaries"] = {"wall": {"of": "pipe", "where": {"r": 0.05}}}
    d["physics"]["heat"]["boundary_conditions"] = {"wall": {"kind": "temperature", "T_K": 300.0},
                                                   "remaining": {"kind": "insulation"}}
    d["mesh"] = {"size": 0.01}
    status, out = S.run_json("mesh", S.write(tmp_path, d), "-o", tmp_path / "m")
    assert status == 0, out
    info = json.loads(Path(out["mesher_info"]).read_text("utf-8"))
    assert abs(info["regions"]["p1"]["area_m2"] / (0.04 * 0.2) - 1.0) <= 1e-10, info["regions"]
    assert info["faces"] and all(e["type"] in ("line", "circle", "ellipse") for e in info["entities"])
    assert S.element_types(Path(out["mesh"]))


# ------------------------------------------------------------------------- shapes the bounded modes cannot mesh
def _d_shape(x0: float, y0: float, w: float, r: float) -> dict:
    """A contour of lines and one elliptical (here circular) arc: x0..x0+w along the bottom, a half circle of radius r
    to the top, and back: a B-spline curve to the geometry kernel."""
    return {"contour": {"start": [x0, y0],
                        "segments": [{"line": [x0 + w, y0]},
                                     {"ellipse": {"center": [x0 + w, y0 + r], "axes": [r, r],
                                                  "t": [-math.pi / 2, math.pi / 2]}},
                                     {"line": [x0, y0 + 2 * r]}]}}


def _shape_refusal(region: str, pointer: str) -> dict:
    return {"error": f"region '{region}': a contour with an elliptical segment cannot be meshed in a model document",
            "where": pointer, "code": "model.shape_not_offered", "details": {"region": region, "segment": 1}}


@pytest.mark.parametrize("case", ["direct", "union", "difference", "nested"])
def test_a_contour_with_an_elliptical_segment_is_refused_at_validation(tmp_path, case):
    """The bounded modes take lines and arcs of circles; a contour's elliptical segment becomes a B-spline curve, which
    the mesher refuses: validation refuses it first, also inside ``union`` and ``difference``, with the pointer of the segment. No mesher is needed."""
    d = S.doc("split_rect")
    rect_a, rect_b = d["regions"][0]["shape"], d["regions"][1]["shape"]
    if case == "direct":
        d["regions"][0]["shape"], region, ptr = _d_shape(0, 0, 0.015, 0.005), "a", "/regions/0/shape/contour/segments/1"
    elif case == "union":
        d["regions"][0]["shape"] = {"union": [rect_a, _d_shape(0, 0, 0.015, 0.005)]}
        region, ptr = "a", "/regions/0/shape/union/1/contour/segments/1"
    elif case == "difference":   # a hole in b
        d["regions"][1]["shape"] = {"difference": [rect_b, _d_shape(0.025, 0.002, 0.005, 0.003)]}
        region, ptr = "b", "/regions/1/shape/difference/1/contour/segments/1"
    else:
        d["regions"][0]["shape"] = {"difference": [{"union": [rect_a, _d_shape(0, 0, 0.015, 0.005)]},
                                                   {"rect": {"x": [0.001, 0.002], "y": [0.001, 0.002]}}]}
        region, ptr = "a", "/regions/0/shape/difference/0/union/1/contour/segments/1"
    status, rep = S.run_json("validate", S.write(tmp_path, d))
    assert status == 1 and rep["ok"] is False, rep
    f = rep["errors"][0]
    want = _shape_refusal(region, ptr)
    assert {k: f.get(k) for k in want} == want, f
    assert '"disk"' in f["hint"], f["hint"]
    assert [g["code"] for g in rep["errors"]] == ["model.shape_not_offered"]
    e = S.raises(axmp.Design(d).validate)
    assert (e.code, e.where, e.details) == (want["code"], want["where"], want["details"])


def test_an_elliptical_contour_in_a_disabled_region_or_an_electromagnetic_design_is_accepted(tmp_path):
    """A disabled region is not meshed; an electromagnetic design's contours are its own mesher's (unchanged)."""
    d = S.doc("split_rect")
    d["regions"][2]["shape"] = _d_shape(0.04, 0, 0.01, 0.005)
    status, rep = S.run_json("validate", S.write(tmp_path, d))
    assert status == 0 and rep["errors"] == [], rep["errors"]
    em = {"name": "e", "regions": [{"name": "x", "shape": _d_shape(0.1, 0, 0.2, 0.1)}]}
    status, out = S.run_json("validate", S.write(tmp_path, em, "em.json"))
    assert status == 0 and out["ok"] is True, out


def test_an_elliptical_contour_in_a_document_on_an_imported_mesh_is_not_refused(tmp_path):
    """A document on an imported mesh (``mesh.file``) is not meshed in the bounded modes: its region shapes are not
    refused for their elliptical segments."""
    d = S.doc("channel")
    d["regions"][0]["shape"] = _d_shape(0, 0, 0.015, 0.005)
    d["mesh"] = {"file": "chan.msh", "info_file": "chan.json"}
    d["boundaries"] = {"inlet": {"group": "inlet"}, "outlet": {"group": "outlet"}, "walls": {"group": "walls"}}
    status, rep = S.run_json("validate", S.write(tmp_path, d))
    assert "model.shape_not_offered" not in [f["code"] for f in rep["errors"]], rep["errors"]
    assert status == 0 and rep["errors"] == [], rep["errors"]


def test_a_model_on_an_imported_mesh_keeps_the_files_of_its_directory(tmp_path):
    """The command clears an earlier run's mesh.msh, mesh_info.json and mesh_in.json, but never for a document on an
    imported mesh, whose own files may have these names: it is refused and both files are untouched."""
    d = S.doc("channel")
    del d["regions"]
    d["mesh"] = {"file": "mesh.msh", "info_file": "mesh_info.json"}
    d["boundaries"] = {"inlet": {"group": "inlet"}, "outlet": {"group": "outlet"}, "walls": {"group": "walls"}}
    del d["outputs"]["F_walls"]
    S.write(tmp_path, d)
    files = {"mesh.msh": b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", "mesh_info.json": b'{"groups": []}\n'}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    status, out = S.run_json("mesh", "doc.json", "-o", ".", cwd=tmp_path)
    assert status == 1 and out["code"] == "route.not_implemented", out
    for name, data in files.items():
        assert (tmp_path / name).read_bytes() == data, name
