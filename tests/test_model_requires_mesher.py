"""``build.requires_mesher``: a model document with a geometry in an installation whose mesher does not run (design note
14.2, 14.3).

The probe runs the mesher that ``axi::find_mesher`` finds (``--mesher M`` first, then ``$AXMP_MESHER``, then the
program's own directory) with no argument: exit status 2 (its usage text) means it loaded; anything else, or no
program, means it is not available. The probe is cached per resolved path for the life of a process. It runs only at
the entry points that serve a user: ``axmp validate``, ``_core.validate`` (``Design.validate``), ``axmp boundaries``,
``_core.boundaries`` (``Design.boundaries``), ``axmp mesh``, ``_core.mesh`` and ``_core.mesh_model`` (``axmp.mesh``);
never in ``explain``, ``patch`` or ``resolve`` (``validate_model``'s default runs no probe).

The finding is unavailable (``validate`` reports it, first in ``unavailable``; ``--strict`` raises it), only for a
document with an instance that has a geometry and no ``mesh.file``; ``boundaries``, ``mesh``, ``_core.mesh`` and
``axmp.mesh`` refuse with it and write nothing. Its fields: ``where`` "/regions"; ``details`` ``{"mesher": the path
probed (None when none was found), "reason": the probe's account, which holds the loader's message from the mesher's
standard error}``; the hint names both remedies (``pip install "axionmultiphysics[mesh]"`` for a wheel,
``-DAXMP_BUILD_MESHER=ON`` with the gmsh SDK for a source build, and ``$AXMP_MESHER``), and, when the loader's message
names a library other than gmsh's, the X11/OpenGL libraries that gmsh's Linux wheel needs, as the install page lists
them.

The wheel-like environment is made with stubs: shell scripts that print a loader's line and exit 127, as a mesher
whose ``libgmsh`` (or ``libGLU``) is missing does; each records its runs, so that a test can see that the probe ran,
and that nothing else did.

Independent references: the loader's message format; the library list of ``docs/guides/install.md``; the outputs of
``explain``, ``patch`` and ``resolve`` with a working mesher.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core

import model_mesh_support as S
from model_mesh_support import mesher  # noqa: F401 (a fixture)

MESSAGE = "the mesher does not run in this installation, and the geometry of this document needs it"
STUBS = {"gmsh": S.LOADER_GMSH, "glu": S.LOADER_GLU}


def _stub(tmp_path: Path, kind: str = "gmsh", status: int = 127) -> tuple[Path, Path]:
    log = tmp_path / f"stub-{kind}-{status}.log"
    d = tmp_path / f"stub-{kind}-{status}"
    d.mkdir(exist_ok=True)
    return S.stub(d, "axmp-mesher", status=status, stderr=STUBS[kind], log=log), log


def _check_finding(f: dict, stub: Path, kind: str) -> None:
    assert f["code"] == "build.requires_mesher" and f["where"] == "/regions", f
    assert f["error"] == MESSAGE
    assert f["details"]["mesher"] == str(stub), f["details"]
    assert STUBS[kind] in f["details"]["reason"], f["details"]["reason"]
    for remedy in ('pip install "axionmultiphysics[mesh]"', "-DAXMP_BUILD_MESHER=ON", "AXMP_MESHER"):
        assert remedy in f["hint"], (remedy, f["hint"])
    assert (S.GL_LIBRARIES in f["hint"]) == (kind == "glu"), f["hint"]


@pytest.mark.parametrize("kind", ["gmsh", "glu"])
def test_validate_reports_the_finding_with_the_install_hint(tmp_path, monkeypatch, kind):
    stub, log = _stub(tmp_path, kind)
    path = S.write(tmp_path, S.doc("channel"))
    env = S.environment(AXMP_MESHER=str(stub))
    status, rep = S.run_json("validate", path, env=env)
    assert status == 0 and rep["ok"] is True and rep["errors"] == [], rep
    assert [f["code"] for f in rep["unavailable"]] == ["build.requires_mesher", "route.not_implemented"]
    _check_finding(rep["unavailable"][0], stub, kind)
    # --strict raises it: the first unavailable finding
    status, strict = S.run_json("validate", path, "--strict", env=env)
    first = rep["unavailable"][0]
    assert status == 1 and strict["ok"] is False and {k: strict[k] for k in first} == first
    # in process: the same report, and Design.validate(strict=True) raises the finding
    monkeypatch.setenv("AXMP_MESHER", str(stub))
    assert json.loads(_core.validate(path.read_text("utf-8"))) == rep
    S.assert_exception_is(S.raises(axmp.load(path).validate, strict=True), first)
    assert set(S.stub_runs(log)) == {""}, "only the probe ran the stub"


def test_boundaries_and_mesh_refuse_and_write_nothing(tmp_path, monkeypatch):
    stub, log = _stub(tmp_path)
    d = S.doc("split_rect")
    path = S.write(tmp_path, d)
    env = S.environment(AXMP_MESHER=str(stub))
    status, rep = S.run_json("boundaries", path, "--json", "-o", tmp_path / "b", env=env)
    assert status == 1 and rep["ok"] is False and rep["code"] == "build.requires_mesher", rep
    first = next(f for f in rep["unavailable"] if f["code"] == "build.requires_mesher")
    _check_finding(first, stub, "gmsh")
    assert {k: rep[k] for k in first} == first
    assert "boundaries" not in rep
    status, out = S.run_json("mesh", path, "-o", tmp_path / "m", env=env)
    assert status == 1 and out == {"ok": False, **first}, out
    for sub in ("b", "m"):
        assert not (tmp_path / sub).exists() or not any((tmp_path / sub).iterdir()), f"{sub}: something was written"
    monkeypatch.setenv("AXMP_MESHER", str(stub))
    for call in (lambda: _core.mesh(json.dumps(d), str(tmp_path / "p"), 0), lambda: axmp.mesh(axmp.Design(d)),
                 lambda: axmp.Design(d).boundaries(), lambda: _core.boundaries(json.dumps(d))):
        S.assert_exception_is(S.raises(call), first)
    assert not (tmp_path / "p").exists() or not any((tmp_path / "p").iterdir())
    assert set(S.stub_runs(log)) == {""}, "only the probe ran the stub"


def test_the_mesher_option_comes_before_the_environment(tmp_path, mesher):
    """With a working mesher in ``$AXMP_MESHER``, ``--mesher`` names the stub, which is probed and refused."""
    stub, log = _stub(tmp_path)
    path = S.write(tmp_path, S.doc("split_rect"))
    status, rep = S.run_json("boundaries", path, "--json", "--mesher", stub)
    assert status == 1 and rep["code"] == "build.requires_mesher" and rep["details"]["mesher"] == str(stub), rep
    status, out = S.run_json("mesh", path, "-o", tmp_path / "m", "--mesher", stub)
    assert status == 1 and out["code"] == "build.requires_mesher" and out["details"]["mesher"] == str(stub), out
    e = S.raises(axmp.mesh, axmp.Design(S.doc("split_rect")), mesher=str(stub))
    assert e.code == "build.requires_mesher" and e.details["mesher"] == str(stub)


def test_validate_takes_the_mesher_option(tmp_path, mesher):
    """``axmp validate --mesher M`` probes M, as ``axmp boundaries`` and ``axmp mesh`` do (the hint names the option):
    a stub that cannot load in ``$AXMP_MESHER`` and a working ``--mesher`` give no finding; a working mesher in the
    environment and the stub given by ``--mesher`` give the finding, naming the stub."""
    stub, _ = _stub(tmp_path)
    path = S.write(tmp_path, S.doc("channel"))
    status, rep = S.run_json("validate", path, "--mesher", mesher, env=S.environment(AXMP_MESHER=str(stub)))
    assert status == 0 and [f["code"] for f in rep["unavailable"]] == ["route.not_implemented"], rep["unavailable"]
    status, rep = S.run_json("validate", path, "--mesher", stub)
    assert status == 0 and rep["unavailable"][0]["code"] == "build.requires_mesher", rep["unavailable"]
    assert rep["unavailable"][0]["details"]["mesher"] == str(stub)
    status, rep = S.run_json("validate", path, "--strict", "--mesher", stub)
    assert status == 1 and rep["code"] == "build.requires_mesher", rep


def test_validate_mesher_without_a_value_is_a_usage_error(tmp_path):
    """``axmp validate doc.json --mesher`` with no value is a usage error (exit status 2), as for ``axmp mesh``."""
    path = S.write(tmp_path, S.doc("channel"))
    for cmd in ("validate", "mesh"):
        r = S.run(cmd, path, "--mesher")
        assert r.returncode == 2 and "usage:" in r.stderr, (cmd, r.returncode, r.stdout, r.stderr)


def test_no_finding_without_a_geometry_to_mesh(tmp_path):
    """A reactor (no geometry) and a document on an imported mesh (``mesh.file``) need no mesher."""
    stub, _ = _stub(tmp_path)
    env = S.environment(AXMP_MESHER=str(stub))
    reactor = {"name": "r", "physics": {"rx": {"type": "chem.reactor"}},
               "study": [{"id": "run", "type": "transient", "time": {"t_end_s": 10}}]}
    imported = S.doc("channel")
    del imported["regions"]
    imported["mesh"] = {"file": "chan.msh", "info_file": "chan.json"}
    imported["boundaries"] = {"inlet": {"group": "inlet"}, "outlet": {"group": "outlet"}, "walls": {"group": "walls"}}
    del imported["outputs"]["F_walls"]
    for name, d in (("reactor", reactor), ("imported", imported)):
        status, rep = S.run_json("validate", S.write(tmp_path, d, f"{name}.json"), env=env)
        assert status == 0 and rep["errors"] == [], (name, rep["errors"])
        assert "build.requires_mesher" not in [f["code"] for f in rep["unavailable"]], (name, rep["unavailable"])


@pytest.mark.parametrize("status", [0, 1])
def test_a_program_that_does_not_print_the_usage_is_not_available(tmp_path, status):
    """Only exit status 2 means the mesher loaded; a program that exits 0 or 1 without arguments is not the mesher."""
    d = tmp_path / "other"
    d.mkdir()
    stub = S.stub(d, "axmp-mesher", status=status, stderr="not the mesher")
    status_, rep = S.run_json("validate", S.write(tmp_path, S.doc("channel")), env=S.environment(AXMP_MESHER=str(stub)))
    assert status_ == 0
    f = rep["unavailable"][0]
    assert f["code"] == "build.requires_mesher" and f["details"]["mesher"] == str(stub), rep["unavailable"]
    assert f"status {status}" in f["details"]["reason"] and "not the mesher" in f["details"]["reason"]


@pytest.mark.parametrize("kind", ["gmsh", "glu"])
def test_explain_patch_and_resolve_do_not_probe(tmp_path, monkeypatch, mesher, kind):
    """``explain``, ``patch`` and ``resolve`` print exactly what they print with the working mesher, and never run the stub (``validate_model`` runs no probe by default)."""
    stub, log = _stub(tmp_path, kind)
    path = S.write(tmp_path, S.doc("channel"), "channel.json")
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([{"op": "replace", "path": "/parameters/U_mean", "value": 0.02}]))
    for args in (("explain", path), ("explain", "--json", path), ("patch", path, ops), ("resolve", path)):
        real = S.run(*args)
        stubbed = S.run(*args, env=S.environment(AXMP_MESHER=str(stub)))
        assert real.returncode == 0, real.stdout + real.stderr
        assert (stubbed.returncode, stubbed.stdout) == (real.returncode, real.stdout), args
    d = axmp.Design(S.doc("channel"))
    text, js = d.explain(), d.explain(json=True)
    monkeypatch.setenv("AXMP_MESHER", str(stub))
    e = axmp.Design(S.doc("channel"))
    assert e.explain() == text and e.explain(json=True) == js
    e.patch([{"op": "replace", "path": "/parameters/U_mean", "value": 0.02}])
    assert S.stub_runs(log) == [], S.stub_runs(log)


def test_the_probe_runs_once_per_mesher_in_a_process(tmp_path, monkeypatch):
    stub, log = _stub(tmp_path)
    monkeypatch.setenv("AXMP_MESHER", str(stub))
    text = json.dumps(S.doc("channel"))
    for _ in range(2):
        rep = json.loads(_core.validate(text))
        assert rep["unavailable"][0]["code"] == "build.requires_mesher"
    S.raises(axmp.Design(S.doc("channel")).boundaries)
    assert S.stub_runs(log) == [""], S.stub_runs(log)
