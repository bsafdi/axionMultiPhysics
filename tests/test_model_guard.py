"""A model document at the entry points of the electromagnetic path (design note 3.2): the guard at the top of
``parse_design``, the named refusal of every electromagnetic-only entry point, the generic refusal of the guard as a
backstop, and the refusal of solving, which will take a model document in a later version (meshing takes one in this
version: tests/test_model_mesh_cli.py).

Texts: ``tests/model_switch_texts.py`` (design note 3.2 verbatim where it gives a text; the specification where it gives
none). Independent references: the texts of 3.2; the entry-point table of 3.2 as it is on this tree (development plan
section 6 item 4); today's "unknown field" and "missing required field" texts, measured with the unchanged build and
pinned here; ``compose()`` of ``core.hpp`` for ``str(e)``.

The model document of every refusal is ``cpp/tests/data/model/channel.json`` (named 'channel', symmetry planar).
Needs the compiled core; the command-line tests need the ``axmp`` executable (``AXMP_BIN``, ``AXMP_EXE`` or a build
directory) and are skipped without it; the MCP tests need the ``mcp`` package.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import pickle
import subprocess
from pathlib import Path, PurePosixPath

import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core

import model_switch_texts as T

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = ROOT / "cpp" / "tests" / "data" / "model" / "channel.json"
ERROR = _core.AxionMultiPhysicsError

_spec = importlib.util.spec_from_file_location("g0_harness_model_guard", ROOT / "tests" / "g0" / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)


def _axmp() -> str:
    exe = g0.find_axmp()
    if exe is None:
        pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")
    return str(exe)


def _run(*args, timeout=300) -> subprocess.CompletedProcess:
    return subprocess.run([_axmp(), *map(str, args)], capture_output=True, text=True, timeout=timeout)


def _channel() -> dict:
    return json.loads(CHANNEL.read_text("utf-8"))


def _em_minimal() -> dict:
    return {"name": "m", "regions": [{"name": "x", "shape": {"rect": {"r": [0, 1], "z": [0, 1]}}}]}


def _write(tmp_path: Path, doc: dict, name: str = "doc.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p


def _raises(f, *a, **kw) -> BaseException:
    with pytest.raises(Exception) as ei:
        f(*a, **kw)
    return ei.value


# ----------------------------------------------------------------- the seven texts of the guard
@pytest.mark.parametrize("key", T.SECTIONS)
def test_seven_guard_texts(tmp_path, key):
    """Six of the seven: a section key of a model document without "physics", through ``axmp validate`` (stdout,
    exit status 1, no ``details`` key) and through the exception of ``_core.validate`` and ``Design.validate``.
    The seventh (``physics`` reaching an electromagnetic-only operation) is ``test_the_seventh_guard_text``."""
    doc = _em_minimal()
    doc[key] = {}
    want = T.section_needs_physics(key)
    T.assert_exception(_raises(_core.validate, json.dumps(doc)), want)
    T.assert_exception(_raises(axmp.Design(doc).validate), want)
    r = _run("validate", _write(tmp_path, doc))
    assert r.returncode == 1, r.stdout + r.stderr
    assert json.loads(r.stdout) == T.cli_json(want)


def test_the_seventh_guard_text(tmp_path):
    """``physics`` reaching an electromagnetic-only operation: ``axmp sensitivities`` prints the refusal of design note
    3.2 (message and pointer verbatim) and exits 1; the exception of a model class carries the same fields."""
    r = _run("sensitivities", CHANNEL, "-o", tmp_path / "run")
    assert r.returncode == 1, r.stdout + r.stderr
    assert json.loads(r.stdout) == T.cli_json(T.em_only("sensitivities"))
    assert not (tmp_path / "run" / "sensitivities.json").exists()
    T.assert_exception(_raises(_core.Model, CHANNEL.read_text("utf-8")), T.em_only("Model"))


def test_a_section_key_wins_over_an_unknown_key_and_the_first_section_is_named(tmp_path):
    """The two membership tests come before ``root.only``: with an unknown top-level key as well, the section key is
    reported; of two section keys the first in the order of the note's list (domains, boundaries, reactions,
    couplings, study, outputs) is named."""
    doc = _em_minimal()
    doc.update({"zzz_unknown": 1, "outputs": {}, "boundaries": {}})
    T.assert_exception(_raises(_core.validate, json.dumps(doc)), T.section_needs_physics("boundaries"))
    r = _run("validate", _write(tmp_path, doc))
    assert json.loads(r.stdout) == T.cli_json(T.section_needs_physics("boundaries"))


# ----------------------------------------------------------------- the entry points of the table of 3.2
def _mcp():
    pytest.importorskip("mcp", reason="the MCP server needs the 'mcp' package")
    from axionmultiphysics import mcp_server
    return mcp_server


def _mcp_workspace(tmp_path, monkeypatch):
    ms = _mcp()
    designs, runs = tmp_path / "designs", tmp_path / "runs"
    designs.mkdir()
    runs.mkdir()
    monkeypatch.setattr(ms, "WORKSPACE", tmp_path)
    monkeypatch.setattr(ms, "DESIGNS", designs)
    monkeypatch.setattr(ms, "RUNS", runs)
    monkeypatch.setattr(ms, "JOBDIR", tmp_path / "jobs")
    (designs / "channel.json").write_text(CHANNEL.read_text("utf-8"), encoding="utf-8")
    return ms


def _call_tool(ms, name: str):
    tools = {
        "make_3d": lambda: ms.make_3d("channel"),
        "set_fullwave": lambda: ms.set_fullwave("channel", f_min=1e3, f_max=1e4, points=3),
        "set_hts": lambda: ms.set_hts("channel"),
        "sensitivities": lambda: asyncio.run(ms.sensitivities("channel", None)),
        "optimize_design": lambda: asyncio.run(ms.optimize_design("channel", None)),
        "optimize_global": lambda: asyncio.run(ms.optimize_global("channel", {"L": [0.01, 0.05]}, None)),
        "optimize": lambda: asyncio.run(ms.optimize("channel", {"L": [0.01, 0.05]}, None)),
        # a tool of the modules: called through the server, which returns its error payload as JSON text
        "export_comsol": lambda: json.loads(asyncio.run(ms.server.call_tool(
            "export_comsol", {"name": "channel", "format": "java"})).content[0].text),
    }
    return tools[name]()


PYTHON_SITES = {
    # the four binding constructors, named by their Python class
    "Model3D": lambda d: _core.Model3D(json.dumps(d.doc)),
    "XsecModel": lambda d: _core.XsecModel(json.dumps(d.doc)),
    "Fw2DModel": lambda d: _core.Fw2DModel(json.dumps(d.doc)),
    "Model": lambda d: _core.Model(json.dumps(d.doc)),
    # the Python functions and methods, refused before anything else
    "gradient": lambda d: d.gradient(),
    "optimize": lambda d: d.optimize(objective="cost", params={"L": [0.01, 0.05]}),
    "global_optimize": lambda d: axmp.global_optimize(d, objective="cost", params={"L": [0.01, 0.05]}),
    "to_3d": lambda d: __import__("axionmultiphysics.convert", fromlist=["to_3d"]).to_3d(d),
    # the COMSOL export builds an electromagnetic COMSOL model from the resolved geometry (the file name is relative
    # to the test's directory)
    "export_comsol": lambda d: __import__("axionmultiphysics.comsol", fromlist=["export_comsol"]).export_comsol(
        d, "channel.java", format="java"),
}
MCP_TOOLS = ["make_3d", "set_fullwave", "set_hts", "sensitivities", "optimize_design", "optimize_global", "optimize",
             "export_comsol"]


@pytest.mark.parametrize("operation", sorted(PYTHON_SITES))
def test_em_only_entry_points_name_themselves(tmp_path, monkeypatch, operation):
    """Each electromagnetic-only entry point of the Python side refuses a model document with
    ``model.em_only_operation``, and ``details.operation`` names it (development plan N1): not the guard's generic
    text, and not an error of the operation itself (``to_3d`` would say "design is already 3D"). Nothing is written
    to the working directory."""
    monkeypatch.chdir(tmp_path)
    d = axmp.Design(_channel())
    T.assert_exception(_raises(PYTHON_SITES[operation], d), T.em_only(operation))
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_sensitivity_optimize_names_itself(tmp_path, monkeypatch):
    """``sensitivity.optimize``, called directly, refuses a model document with the coded refusal naming "optimize"
    before it evaluates anything (not the uncoded error of its first solve), for a Design and for a dict, and writes
    nothing."""
    from axionmultiphysics.sensitivity import optimize

    monkeypatch.chdir(tmp_path)
    for source in (axmp.Design(_channel()), _channel()):
        T.assert_exception(_raises(optimize, source, objective="cost", params={"L": [0.01, 0.05]}),
                           T.em_only("optimize"))
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_axmp_optimize_names_itself():
    """``axmp.optimize`` (the module function that ``Design.optimize`` calls) names "optimize" as well."""
    T.assert_exception(_raises(axmp.optimize, axmp.Design(_channel()), objective="cost", params={"L": [0.01, 0.05]}),
                       T.em_only("optimize"))


@pytest.mark.parametrize("tool", MCP_TOOLS)
def test_em_only_mcp_tools_name_themselves(tmp_path, monkeypatch, tool):
    """The electromagnetic MCP tools refuse a model document through the error payload, before they change or start
    anything, and ``details.operation`` is the tool's name."""
    ms = _mcp_workspace(tmp_path, monkeypatch)
    before = (tmp_path / "designs" / "channel.json").read_bytes()
    payload = _call_tool(ms, tool)
    T.assert_payload(payload, T.em_only(tool))
    assert (tmp_path / "designs" / "channel.json").read_bytes() == before
    assert sorted(p.name for p in (tmp_path / "designs").iterdir()) == ["channel.json"]
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == []


@pytest.mark.parametrize("fmt", ["java", "mphtxt", "step"])
def test_export_comsol_refuses_every_form_of_a_model_document(tmp_path, monkeypatch, capsys, fmt):
    """``export_comsol`` refuses a model document in every form of its source (a Design, a dict, a ``.json`` file as
    ``str`` or ``Path``, the ``design.json`` of a run directory) and in every format, with and without
    ``physics="fullwave"``, before it resolves, meshes or writes anything; ``axmp export-comsol`` (handled in Python)
    prints the same refusal as its error JSON, with the code and details, and exits 1."""
    from axionmultiphysics import _cli
    from axionmultiphysics.comsol import export_comsol

    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    run = src / "run"
    run.mkdir(parents=True)
    (src / "channel.json").write_text(CHANNEL.read_text("utf-8"), encoding="utf-8")
    (run / "design.json").write_text(CHANNEL.read_text("utf-8"), encoding="utf-8")
    want = T.em_only("export_comsol")
    out = tmp_path / "out" / f"channel.{fmt}"
    class _PathLike:   # any os.PathLike, not only pathlib.Path
        def __init__(self, p):
            self.p = str(p)

        def __fspath__(self):
            return self.p

    sources = [axmp.Design(_channel()), _channel(), str(src / "channel.json"), src / "channel.json", run,
               PurePosixPath(src / "channel.json"), _PathLike(src / "channel.json"), _PathLike(run)]
    for source in sources:
        for physics in (["none", "fullwave"] if fmt != "step" else ["none"]):
            T.assert_exception(_raises(export_comsol, source, out, format=fmt, physics=physics), want)
    for source in (src / "channel.json", run):
        assert _cli.main(["export-comsol", str(source), "-o", str(out), "--format", fmt]) == 1
        T.assert_payload(json.loads(capsys.readouterr().out), want)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src"]
    assert sorted(p.name for p in src.iterdir()) == ["channel.json", "run"]
    assert sorted(p.name for p in run.iterdir()) == ["design.json"]


def test_export_comsol_leaves_an_unreadable_run_directory_to_the_export(tmp_path, monkeypatch):
    """The refusal never raises for a path it cannot read: a run directory without permission (mode 000) is left to
    the export, which reports its own error (not an error of the refusal's file probes), and an os.DirEntry source
    is refused as a model document."""
    from axionmultiphysics import comsol
    from axionmultiphysics.comsol import _refuse_model_document

    monkeypatch.chdir(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "design.json").write_text(CHANNEL.read_text("utf-8"), encoding="utf-8")
    (tmp_path / "channel.json").write_text(CHANNEL.read_text("utf-8"), encoding="utf-8")
    with os.scandir(tmp_path) as it:
        entry = next(e for e in it if e.name == "channel.json")
    T.assert_exception(_raises(_refuse_model_document, entry), T.em_only("export_comsol"))
    if os.geteuid() == 0:
        pytest.skip("mode 000 does not stop root")
    locked.chmod(0)
    try:
        assert _refuse_model_document(locked) is None
        assert _refuse_model_document(str(locked)) is None
        assert _refuse_model_document(PurePosixPath(locked)) is None
        # the refusal raises nothing, so the export's own error is what the caller sees (it is not
        # model.em_only_operation)
        e = _raises(comsol.export_comsol, locked, tmp_path / "o" / "x.java", format="java")
        assert getattr(e, "code", "") != "model.em_only_operation"
    finally:
        locked.chmod(0o755)


def test_export_comsol_prints_an_electromagnetic_error_as_before(tmp_path, monkeypatch, capsys):
    """An error without a code keeps the error JSON that ``axmp export-comsol`` printed before: ``ok`` and the whole
    message, no ``where`` or ``hint`` keys although the message has both (an unknown field in the ``mesh`` of an
    electromagnetic design, raised by ``_core.resolve``)."""
    from axionmultiphysics import _cli

    monkeypatch.chdir(tmp_path)
    doc = _em_minimal()
    doc["mesh"] = {"zzz": 1}
    src = _write(tmp_path, doc, "em.json")
    e = _raises(_core.resolve, json.dumps(doc))
    assert e.code == "" and e.where and e.hint
    assert _cli.main(["export-comsol", str(src), "-o", str(tmp_path / "em.java"), "--format", "java"]) == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": str(e)}


def test_an_unnamed_model_document_is_called_this_document():
    doc = _channel()
    del doc["name"]
    T.assert_exception(_raises(lambda: _core.Model3D(json.dumps(doc))), T.em_only("Model3D", name=None))


def test_backstop_reports_a_generic_operation():
    """``_core._parse_design_for_tests`` calls ``parse_design`` with no ``require_em_document`` before it (it is listed
    in the manifest of call sites as a test site): a model document gets the guard's generic text, with
    ``details.operation`` null (development plan N1); the value of "physics" is not looked at."""
    assert hasattr(_core, "_parse_design_for_tests"), "the private binding _core._parse_design_for_tests is missing"
    T.assert_exception(_raises(_core._parse_design_for_tests, CHANNEL.read_text("utf-8")), T.em_only(None))
    doc = _em_minimal()
    doc["physics"] = 3
    T.assert_exception(_raises(_core._parse_design_for_tests, json.dumps(doc)), T.em_only(None, name="m"))
    # an electromagnetic design parses (the binding returns the validate dictionary of the design's name)
    assert json.loads(_core._parse_design_for_tests(json.dumps(_em_minimal())))["name"] == "m"


# ----------------------------------------------------------------- solving: not implemented yet
def test_solve_refuses_with_route_not_implemented(tmp_path):
    """``axmp solve``, ``_core.run_3d`` and ``Design.solve`` take a model document in a later version: until then they
    refuse it with ``route.not_implemented`` (development plan N2), not with the guard's ``model.em_only_operation``,
    and write no result. (``axmp mesh`` and ``_core.mesh`` take one in this version: tests/test_model_mesh_cli.py.)"""
    r = _run("solve", CHANNEL, "-o", tmp_path / "s")
    assert r.returncode == 1, r.stdout + r.stderr
    assert json.loads(r.stdout) == T.cli_json(T.not_implemented("solve"))
    # the summary file of every solve (Design.solve(mpi_ranks=N) reads it) carries the same refusal
    assert json.loads((tmp_path / "s" / "axmp_status.json").read_text()) == T.cli_json(T.not_implemented("solve"))
    assert not (tmp_path / "s" / "results.json").exists()

    text = CHANNEL.read_text("utf-8")
    T.assert_exception(_raises(_core.run_3d, text, str(tmp_path / "pr"), 0, 0), T.not_implemented("run_3d"))
    d = axmp.Design(_channel())
    T.assert_exception(_raises(d.solve, workdir=tmp_path / "ds"), T.not_implemented("solve"))
    # refused before the dispatch on mpi_ranks (no launcher is looked for)
    T.assert_exception(_raises(d.solve, workdir=tmp_path / "ds2", mpi_ranks=2, launcher="no-such-launcher"),
                       T.not_implemented("solve"))
    assert not (tmp_path / "ds" / "results.json").exists()


def test_mcp_solve_and_sweep_refuse_with_route_not_implemented(tmp_path, monkeypatch):
    """``solve_design`` and ``sweep`` get the refusal of ``Design.solve`` from the worker, with its code and details
    (they cross the worker's process boundary as the worker's error payload; the server adds the job id)."""
    ms = _mcp_workspace(tmp_path, monkeypatch)
    for out in (asyncio.run(ms.solve_design("channel", None)), asyncio.run(ms.sweep("channel", "L", [0.04], None))):
        out = dict(out)
        assert out.pop("job_id", None), out
        T.assert_payload(out, T.not_implemented("solve"))


def test_serves_model_documents_lists_this_version():
    """``details.serves_model_documents`` is ``["validate", "resolve", "explain", "patch", "boundaries", "mesh"]`` in every
    refusal of this version (``axmp physics`` takes no document and is not listed)."""
    d = axmp.Design(_channel())
    six = ["validate", "resolve", "explain", "patch", "boundaries", "mesh"]
    for f in (lambda: _core.Model(json.dumps(d.doc)), d.solve, lambda: _core._parse_design_for_tests(json.dumps(d.doc))):
        e = _raises(f)
        assert getattr(e, "details", None) and e.details["serves_model_documents"] == six, e


# ----------------------------------------------------------------- keys below the top level keep today's texts
# Today's texts (axmp validate of the unchanged build, the allowed lists of spec.cpp's `only` calls in std::set order).
MESH_ALLOWED = ("Allowed fields here: adapt, algorithm3d, curvature_points, domain_center_z, domain_radius, domain_solid, "
                "extend_from_boundary, far_size, file, geometry_order, info_file, optimize_netgen, order, refinements, "
                "remove_pec, size, surface_layer, surface_size, symmetry, threads.")


def _unknown_field_twin(doc_with_new_key: dict, doc_with_zzz: dict, key: str, tmp_path: Path):
    """The new key gives the text that an unknown key 'zzz' at the same place gives (today's text; its allowed list is
    not widened), through the exception and through ``axmp validate``."""
    e_new = _raises(_core.validate, json.dumps(doc_with_new_key))
    e_ref = _raises(_core.validate, json.dumps(doc_with_zzz))
    assert type(e_new) is ERROR and type(e_ref) is ERROR
    assert str(e_new) == str(e_ref).replace("unknown field 'zzz'", f"unknown field '{key}'", 1)
    assert e_new.code == "" and e_new.details is None
    assert key not in e_new.hint
    r = _run("validate", _write(tmp_path, doc_with_new_key, f"{key}.json"))
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert set(out) == {"ok", "error", "where", "hint"} and out["error"] == f"unknown field '{key}'"
    return out


def test_new_material_key_or_exterior_without_physics_gives_the_unknown_field_text(tmp_path):
    """Design note 3.2 item 1: ``mesh.exterior``, ``thermal.cp_J_kgK``, ``mechanical.alpha_1_K`` and the material
    blocks ``fluid``, ``electrolyte``, ``electric``, ``species`` get no dedicated refusal in an electromagnetic
    document: they keep today's "unknown field" text with the electromagnetic allowed list."""
    a, b = _em_minimal(), _em_minimal()
    a["mesh"], b["mesh"] = {"exterior": "none"}, {"zzz": "none"}
    out = _unknown_field_twin(a, b, "exterior", tmp_path)
    assert out == {"ok": False, "error": "unknown field 'exterior'", "where": "/mesh", "hint": MESH_ALLOWED}
    for block in ("fluid", "electrolyte", "electric", "species"):
        a, b = _em_minimal(), _em_minimal()
        a["materials"] = {"w": {"kind": "vacuum", block: {}}}
        b["materials"] = {"w": {"kind": "vacuum", "zzz": {}}}
        assert _unknown_field_twin(a, b, block, tmp_path)["where"] == "/materials/w"
    a, b = _em_minimal(), _em_minimal()
    a["materials"] = {"w": {"kind": "vacuum", "thermal": {"k_W_mK": 1.0, "cp_J_kgK": 1.0}}}
    b["materials"] = {"w": {"kind": "vacuum", "thermal": {"k_W_mK": 1.0, "zzz": 1.0}}}
    assert _unknown_field_twin(a, b, "cp_J_kgK", tmp_path)["where"] == "/materials/w/thermal"
    a, b = _em_minimal(), _em_minimal()
    a["materials"] = {"w": {"kind": "vacuum", "mechanical": {"alpha_1_K": 1e-5}}}
    b["materials"] = {"w": {"kind": "vacuum", "mechanical": {"zzz": 1e-5}}}
    assert _unknown_field_twin(a, b, "alpha_1_K", tmp_path)["where"] == "/materials/w/mechanical"


def test_material_without_kind_is_still_refused_without_physics(tmp_path):
    """A material without ``kind`` is a model document's material (design note 6.2) and stays an error of an
    electromagnetic document, with today's text."""
    doc = _em_minimal()
    doc["materials"] = {"w": {"density": 1000}}
    doc["regions"][0]["material"] = "w"
    want = {"ok": False, "error": "missing required field 'kind'", "hint": "", "where": "/materials/w"}
    r = _run("validate", _write(tmp_path, doc))
    assert r.returncode == 1 and json.loads(r.stdout) == want
    e = _raises(_core.validate, json.dumps(doc))
    assert type(e) is ERROR and str(e) == "missing required field 'kind' [at /materials/w]"
    assert (e.code, e.details) == ("", None)


# ----------------------------------------------------------------- what G0 does not hold
def test_corpus_is_unchanged(tmp_path, monkeypatch):
    """G0 tier A holds the corpus of refused electromagnetic documents (``tests/test_g0.py``). This adds what it does
    not: a model-document error with a code and details survives pickling (the MCP worker's process boundary), and a
    usage error still exits 2 for a model document, before any refusal of the document."""
    e = _raises(_core.Model, CHANNEL.read_text("utf-8"))
    back = pickle.loads(pickle.dumps(e))
    assert type(back) is ERROR and str(back) == str(e) and back.args == e.args
    assert (back.code, back.where, back.hint, back.details) == (e.code, e.where, e.hint, e.details)
    assert back.details == T.em_only("Model")["details"]
    for args in (["mesh", CHANNEL, "--bogus"], ["solve", CHANNEL, "--bogus"], ["sensitivities", CHANNEL, "--bogus"],
                 ["validate"], ["no-such-command", CHANNEL]):
        r = _run(*args)
        assert r.returncode == 2, (args, r.stdout, r.stderr)
        assert "usage:" in r.stderr
    # no corpus document has a top-level key of a model document (docs/developer/g0.md)
    corpus = json.loads((ROOT / "tests" / "g0" / "corpus.json").read_text("utf-8"))["documents"]
    for c in corpus:
        doc = c.get("doc")
        if isinstance(doc, dict):
            assert not ({"physics", *T.SECTIONS} & set(doc)), c["id"]
