"""The served entry points of a model document in this version: ``validate`` and ``resolve`` (design note 3.2 table,
12.2, 14.2; development plan N8, N15).

``axmp validate`` of a model document prints the JSON report (plan N8): ``ok``, ``name``, ``parameters`` (evaluated),
``errors``, ``unavailable``, ``deferred``, ``warnings``; each finding is the command line's error JSON without ``ok``
(``error``, ``where``, ``hint``, and ``code`` and ``details`` when it has them: most findings of a model document
have a code, some have none). With an error, the first error's fields are also at the top level, ``ok`` is false and
the exit status is 1. ``--strict`` treats the first ``unavailable`` finding the same way when there is no error. No
flag is added for the full list: it is the plain output. In Python, ``_core.validate`` and ``Design.validate()`` raise
the first error, ``Design.validate(strict=True)`` also the first unavailable finding, and
``Design.validate(all_errors=True)`` returns the report without raising. An electromagnetic document is unchanged by
all of this (its output, and the extra arguments ``axmp validate`` ignores today).

``axmp resolve`` of a model document prints the resolved geometry in today's shape (plan N15): today's top-level keys,
``mesh`` with the model's keys.

Independent references: the numbers written in ``channel.json``; the electromagnetic parser's resolution of the same
rectangle; the codes of the parser (``physics.unknown_type``, ``model.mesh_order``) for two independent defects;
the exit statuses of design note 12.2 (0 and 1 only: 3 and 4 need a solve).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core

import model_switch_texts as T

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = ROOT / "cpp" / "tests" / "data" / "model" / "channel.json"
ERROR = _core.AxionMultiPhysicsError
REPORT_KEYS = {"ok", "name", "parameters", "errors", "unavailable", "deferred", "warnings"}
FIRST_KEYS = {"error", "where", "hint", "code"}

_spec = importlib.util.spec_from_file_location("g0_harness_model_cli", ROOT / "tests" / "g0" / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)


def _axmp() -> str:
    exe = g0.find_axmp()
    if exe is None:
        pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")
    return str(exe)


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([_axmp(), *map(str, args)], capture_output=True, text=True, timeout=300)


def _channel() -> dict:
    return json.loads(CHANNEL.read_text("utf-8"))


def _mesher_loads() -> bool:
    """The mesher's probe, done here independently (the mesher that the core finds first, run with no argument, exits
    with 2): ``axmp validate`` lists build.requires_mesher exactly when it fails."""
    path = g0.mesher_exe()
    if not path or not os.access(path, os.X_OK):
        return False
    return subprocess.run([path], capture_output=True, timeout=60).returncode == 2


def _two_errors() -> dict:
    """channel with two independent defects in two sections: a misspelt physics type and the EM key mesh.order."""
    doc = _channel()
    doc["physics"]["flow"]["type"] = "fluid.incompressibl"
    doc["mesh"]["order"] = 2
    return doc


def _em() -> dict:
    return {"name": "m", "parameters": {"R": 2}, "regions": [{"name": "x", "shape": {"rect": {"r": [0, "R"], "z": [0, 1]}}}]}


def _write(tmp_path: Path, doc: dict, name: str) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p


def _check_finding(f: dict) -> None:
    assert FIRST_KEYS <= set(f) <= FIRST_KEYS | {"details"}, sorted(f)
    assert "ok" not in f and f["code"] and isinstance(f["error"], str)


def test_validate_prints_the_report():
    """A valid channel: exit 0, the report's seven keys, the evaluated parameters, no error, and the route of its
    flow reported as not implemented (every route of this version is, plan N11). ``_core.validate`` and
    ``Design.validate()`` return the same dictionary; ``Design.resolved_params()`` reads its parameters."""
    r = _run("validate", CHANNEL)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert set(rep) == REPORT_KEYS, sorted(rep)
    assert rep["ok"] is True and rep["name"] == "channel"
    assert rep["parameters"] == {"L": 0.04, "H": 0.01, "U_mean": 0.01, "rho": 998.2, "mu": 1.002e-3}
    assert rep["errors"] == []
    # validate_model's finding for channel (cpp/tests/test_model_validate.cpp, the case "channel"): exactly one; and
    # build.requires_mesher, first, exactly when the mesher's probe fails (tests/test_model_requires_mesher.py)
    codes = [f["code"] for f in rep["unavailable"]]
    assert [c for c in codes if c != "build.requires_mesher"] == ["route.not_implemented"]
    assert ("build.requires_mesher" in codes) == (not _mesher_loads()), codes
    assert "build.requires_mesher" not in codes[1:]
    nav = next(f for f in rep["unavailable"] if f["code"] == "route.not_implemented")
    assert nav["error"] == ("physics 'flow' (fluid.incompressible) takes the route 'navier_stokes_steady_planar', which "
                            "is not implemented yet")
    assert nav["where"] == "/physics" and nav["details"]["route"] == "navier_stokes_steady_planar"
    assert nav["details"]["implemented"] is False
    for kind in ("unavailable", "deferred", "warnings"):
        assert isinstance(rep[kind], list)
        for f in rep[kind]:
            _check_finding(f)
    assert json.loads(_core.validate(CHANNEL.read_text("utf-8"))) == rep
    assert axmp.Design(_channel()).validate() == rep
    assert axmp.load(CHANNEL).validate() == rep
    assert axmp.Design(_channel()).resolved_params() == rep["parameters"]


def test_validate_reports_every_error_and_the_first_at_the_top(tmp_path):
    """Two independent defects are both listed; the first one's fields are also at the top level; exit 1."""
    r = _run("validate", _write(tmp_path, _two_errors(), "bad.json"))
    assert r.returncode == 1, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert REPORT_KEYS | FIRST_KEYS <= set(rep) <= REPORT_KEYS | FIRST_KEYS | {"details"}, sorted(rep)
    assert rep["ok"] is False
    codes = [f["code"] for f in rep["errors"]]
    assert "physics.unknown_type" in codes and "model.mesh_order" in codes, codes
    for f in rep["errors"]:
        _check_finding(f)
    first = rep["errors"][0]
    assert {k: rep[k] for k in first} == first


def test_strict_raises_on_unavailable(tmp_path):
    """``--strict``: a document without errors and with an unavailable finding gives ``ok`` false, the first
    unavailable finding at the top level, exit 1; ``Design.validate(strict=True)`` raises it. For an
    electromagnetic document nothing changes: ``axmp validate em.json --strict`` prints what it prints without the
    flag, exit 0 (the flag is ignored there today), and ``Design.validate(strict=True)`` returns today's dictionary."""
    r = _run("validate", CHANNEL, "--strict")
    assert r.returncode == 1, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert rep["ok"] is False and rep["errors"] == []
    first = rep["unavailable"][0]
    # the first unavailable finding is build.requires_mesher exactly when the mesher's probe fails; the first other one
    # is the route's
    assert (first["code"] == "build.requires_mesher") == (not _mesher_loads()), first
    assert next(f for f in rep["unavailable"] if f["code"] != "build.requires_mesher")["code"] == "route.not_implemented"
    assert {k: rep[k] for k in first} == first
    with pytest.raises(ERROR) as ei:
        axmp.Design(_channel()).validate(strict=True)
    e = ei.value
    assert (e.code, e.where, e.hint) == (first["code"], first["where"], first["hint"])
    assert e.details == first.get("details")
    assert str(e) == T.composed(first)
    em = _write(tmp_path, _em(), "em.json")
    a, b = _run("validate", em), _run("validate", em, "--strict")
    assert (a.returncode, a.stdout) == (b.returncode, b.stdout) == (0, a.stdout)
    assert json.loads(a.stdout) == {"ok": True, "name": "m", "regions": 1, "curves": 0, "windings": 0}
    assert axmp.Design(_em()).validate(strict=True) == axmp.Design(_em()).validate() == \
        {"ok": True, "name": "m", "parameters": {"R": 2.0}}


def test_all_errors_returns_the_report():
    """``Design.validate()`` raises the first error (the report's ``errors[0]``); ``all_errors=True`` returns the report
    with every error and does not raise. An electromagnetic error is raised as today, with or without the flag."""
    doc = _two_errors()
    rep = axmp.Design(doc).validate(all_errors=True)
    assert set(rep) >= REPORT_KEYS and rep["ok"] is False and len(rep["errors"]) >= 2
    first = rep["errors"][0]
    for f in (axmp.Design(doc).validate, lambda: _core.validate(json.dumps(doc))):
        with pytest.raises(ERROR) as ei:
            f()
        e = ei.value
        assert (e.code, e.where, e.hint, str(e)) == (first["code"], first["where"], first["hint"], T.composed(first))
    ok = axmp.Design(_channel()).validate(all_errors=True)
    assert ok == axmp.Design(_channel()).validate()
    bad_em = _em()
    bad_em["zzz"] = 1
    for kw in ({}, {"all_errors": True}, {"strict": True}):
        with pytest.raises(ERROR) as ei:
            axmp.Design(bad_em).validate(**kw)
        assert str(ei.value).startswith("unknown field 'zzz'") and ei.value.code == ""


def test_resolve_of_a_model_document(tmp_path):
    """``axmp resolve`` and ``_core.resolve`` of a model document: today's top-level keys (those of an electromagnetic
    design without a pickup), ``mesh`` with the model's keys, the parameters evaluated, the rectangle resolved as the
    electromagnetic parser resolves it. A document with a grammar error is refused with its first error, as
    ``validate`` raises it."""
    r = _run("resolve", CHANNEL)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout)
    assert json.loads(_core.resolve(CHANNEL.read_text("utf-8"))) == out
    # the electromagnetic parser takes no planar design with a rect (planar needs mesh.file, and then 'imported'
    # shapes) and writes every resolved rect with the keys r and z: the axisymmetric rect is its resolution of the same
    # rectangle
    twin = {"name": "channel", "regions": [{"name": "channel", "shape": {"rect": {"r": [0, 0.04], "z": [0, 0.01]}}}]}
    ref = json.loads(_core.resolve(json.dumps(twin)))
    assert set(out) == set(ref), (sorted(out), sorted(ref))
    assert set(out["mesh"]) == {"size", "surface_size", "geometry_order", "refinements", "exterior"}
    assert out["name"] == "channel" and out["symmetry"] == "planar"
    assert out["parameters"] == {"L": 0.04, "H": 0.01, "U_mean": 0.01, "rho": 998.2, "mu": 1.002e-3}
    assert [g["name"] for g in out["regions"]] == ["channel"] and out["regions"][0]["material"] == "water"
    assert out["regions"][0]["shape"] == ref["regions"][0]["shape"]
    assert "water" in out["materials"] and "kind" in out["materials"]["water"]
    bad = _write(tmp_path, _two_errors(), "bad.json")
    r = _run("resolve", bad)
    assert r.returncode == 1
    first = json.loads(_run("validate", bad).stdout)["errors"][0]
    assert json.loads(r.stdout) == {"ok": False, **first}


def test_exit_statuses(tmp_path):
    """Design note 12.2: a model document gets 0 or 1 from ``validate`` and ``resolve`` (3 and 4 need a solve);
    ``usage()`` lists ``--strict``."""
    cases = [(["validate", CHANNEL], 0), (["resolve", CHANNEL], 0), (["validate", CHANNEL, "--strict"], 1),
             (["validate", _write(tmp_path, _two_errors(), "bad.json")], 1),
             (["resolve", _write(tmp_path, _two_errors(), "bad2.json")], 1)]
    for args, status in cases:
        r = _run(*args)
        assert r.returncode == status, (args, r.returncode, r.stdout, r.stderr)
    r = _run()
    assert r.returncode == 2 and "axmp validate design.json [--strict]" in r.stderr


@pytest.mark.parametrize("command", ["validate", "resolve"])
@pytest.mark.parametrize("raw", [b'{"name": "\xff\xfe", "regions": []}', b'{"name": "x", "k": "\xc3(" }',
                                 b"\xff\xfe{\x00}\x00",
                                 '{\u201cname\u201d: "x"}'.encode("utf-8")])
def test_a_parse_error_at_a_byte_that_is_not_ascii_is_an_error_not_an_abort(tmp_path, command, raw):
    """The parse error quotes the bytes it stopped at, and when it stops inside a multi-byte character (a file that is
    not valid UTF-8, or a valid one with a typographic quote pasted from a word processor) the quote ends in a
    truncated character; printing it as JSON used to throw out of the error handler, and ``axmp`` aborted (exit 134,
    a core dump). It is the error "invalid JSON" without a code, with exit status 1 and a stdout that is JSON, the bad
    bytes shown as U+FFFD (not dropped). (Reference: the exit status 1 and the error text of the same file with a
    plain syntax error.)"""
    bad = tmp_path / "bad.json"
    bad.write_bytes(raw)
    r = subprocess.run([_axmp(), command, str(bad)], capture_output=True, timeout=300)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert b"terminate" not in r.stderr and b"Abort" not in r.stderr
    out = json.loads(r.stdout.decode("utf-8"))
    assert out["ok"] is False and out["error"].startswith(f"invalid JSON in '{bad}'")
    assert "\ufffd" in out["error"], out["error"]
    plain = tmp_path / "plain.json"
    plain.write_text('{"name": "x",}', encoding="utf-8")
    ref = json.loads(subprocess.run([_axmp(), command, str(plain)], capture_output=True, text=True,
                                    timeout=300).stdout)
    assert ref["ok"] is False and ref["error"].startswith("invalid JSON in")


def test_a_name_that_is_not_utf8_is_an_error_and_the_status_file_has_the_same_text(tmp_path):
    """A directory name that is not UTF-8 reaches the error of ``solve`` (printed to stdout and copied to the status
    file); both must be JSON with the bad byte shown as U+FFFD, and the status file the same JSON as stdout."""
    design = ROOT / "examples" / "long_solenoid_coax.json"
    status = tmp_path / "st.json"
    out_dir = b"/proc/no\xffdir"
    r = subprocess.run([_axmp().encode(), b"solve", str(design).encode(), b"-o", out_dir, b"--status-file",
                        str(status).encode()], capture_output=True, timeout=300)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    out = json.loads(r.stdout.decode("utf-8"))
    assert out["ok"] is False and "\ufffd" in out["error"], out
    assert status.is_file(), "the status file was not written"
    assert json.loads(status.read_text("utf-8")) == out
