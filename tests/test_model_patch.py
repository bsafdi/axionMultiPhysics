"""``axmp patch`` and ``Design.patch()``: an RFC 6902 JSON Patch applied atomically, then validated.

Independent reference: RFC 6902 (operations add, remove, replace, move, copy, test; pointers of RFC 6901, "-" for the end
of an array) and section 12.1 of the interface design note: 'either every operation applies and the result validates, or
the document is unchanged and the error names the operation index'. The texts below are the test author's (the note
gives none) and are the specification of ``patch.operation_failed``:

* code ``patch.operation_failed``; ``where`` is ``/<index>``, the pointer of the operation in the list of operations
  (zero-based), not a pointer into the document; ``details`` is ``{"index", "op", "path"}`` (``op`` and ``path`` as
  written, ``null`` when the operation does not have them);
* message ``patch operation <index> failed: <reason>``; the reasons are the REASONS table below;
* hint ``The document is unchanged. Operations are RFC 6902 (JSON Patch); paths are JSON pointers like the "where" of
  an error.``

A patch whose operations all apply but whose result does not validate raises the validation error itself (its own code,
if any, and its own ``where`` in the patched document), and the document is unchanged. A patch is applied to a copy, one
operation at a time, so that a later operation sees the result of an earlier one.

This file needs the compiled core with the ``patch`` binding and the ``axmp`` executable of the same build.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

import axionmultiphysics as axmp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cpp" / "tests" / "data" / "model"
G0 = ROOT / "tests" / "g0"

_spec = importlib.util.spec_from_file_location("g0_harness_patch", G0 / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)

HINT = ('The document is unchanged. Operations are RFC 6902 (JSON Patch); paths are JSON pointers like the "where" of '
        "an error.")

OK_OP = {"op": "replace", "path": "/physics/flow/options/viscous_form", "value": "stress"}

# (operation, reason): every way an operation can fail on the channel document
REASONS = [
    ({"op": "replace", "path": "/physics/flow/nope/x", "value": 1}, "the path '/physics/flow/nope/x' does not exist"),
    ({"op": "remove", "path": "/physics/flow/nope"}, "the path '/physics/flow/nope' does not exist"),
    ({"op": "add", "path": "/physics/nope/x", "value": 1}, "the parent of '/physics/nope/x' does not exist"),
    ({"op": "test", "path": "/name", "value": "not the name"}, "the value at '/name' is not the expected one"),
    ({"op": "test", "path": "/nope", "value": 1}, "the path '/nope' does not exist"),
    ({"op": "move", "from": "/nope", "path": "/x"}, "the path '/nope' does not exist"),
    ({"op": "copy", "from": "/nope", "path": "/x"}, "the path '/nope' does not exist"),
    ({"op": "add", "path": "/regions/7", "value": {}}, "the index in '/regions/7' is out of range"),
    ({"op": "move", "from": "/regions/0", "path": "/regions/0/x"},
     "a value cannot be moved into one of its own children ('/regions/0' is a prefix of '/regions/0/x')"),
    ({"op": "move", "from": "/parameters", "path": "/parameters/x"},
     "a value cannot be moved into one of its own children ('/parameters' is a prefix of '/parameters/x')"),
    ({"op": "frobnicate", "path": "/name"}, "unknown op 'frobnicate'; use one of add, remove, replace, move, copy, test"),
    ({"path": "/name"}, 'the operation must be an object with "op" and "path"'),
    ({"op": "replace"}, 'the operation must be an object with "op" and "path"'),
    ("not an object", 'the operation must be an object with "op" and "path"'),
]


def _axmp() -> str:
    exe = g0.find_axmp()
    if exe is None:
        pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")
    return exe


def _run(*args):
    return subprocess.run([_axmp(), "patch", *args], capture_output=True, text=True, timeout=120)


def _channel() -> dict:
    return json.loads((DATA / "channel.json").read_text())


def _files(tmp_path, doc, ops):
    (tmp_path / "doc.json").write_text(json.dumps(doc))
    (tmp_path / "ops.json").write_text(json.dumps(ops))
    return str(tmp_path / "doc.json"), str(tmp_path / "ops.json")


# ----------------------------------------------------------------- atomicity and the index
def test_one_invalid_operation_leaves_the_document_unchanged_and_names_its_index(tmp_path):
    """Valid, invalid, valid: the error is for the operation 1; neither the first nor the third is applied, in Python and
    on the command line (no output file is written, the input file is untouched)."""
    doc = _channel()
    before = copy.deepcopy(doc)
    bad = {"op": "replace", "path": "/physics/flow/nope/x", "value": 1}
    ops = [OK_OP, bad, {"op": "replace", "path": "/name", "value": "renamed"}]

    d = axmp.Design(doc)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d.patch(ops)
    err = e.value
    assert err.code == "patch.operation_failed"
    assert err.where == "/1"
    assert err.hint == HINT
    assert err.details == {"index": 1, "op": "replace", "path": "/physics/flow/nope/x"}
    assert "patch operation 1 failed: the path '/physics/flow/nope/x' does not exist" in str(err)
    assert d.doc == before and doc == before          # not applied in place, not even the valid first operation

    dfile, ofile = _files(tmp_path, doc, ops)
    out_file = tmp_path / "out.json"
    r = _run(dfile, ofile, "-o", str(out_file))
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["code"] == "patch.operation_failed" and out["where"] == "/1" and out["hint"] == HINT
    assert out["details"] == {"index": 1, "op": "replace", "path": "/physics/flow/nope/x"}
    assert out["error"] == "patch operation 1 failed: the path '/physics/flow/nope/x' does not exist"
    assert not out_file.exists()
    assert json.loads(Path(dfile).read_text()) == before


@pytest.mark.parametrize("op,reason", REASONS, ids=[r[1][:40] for r in REASONS])
def test_every_failure_has_its_reason_and_the_document_stays(op, reason):
    doc = _channel()
    before = copy.deepcopy(doc)
    d = axmp.Design(doc)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d.patch([OK_OP, op])
    assert e.value.code == "patch.operation_failed"
    assert str(e.value).startswith(f"patch operation 1 failed: {reason}")
    assert e.value.where == "/1"
    assert e.value.details["index"] == 1
    if isinstance(op, dict):
        assert e.value.details["op"] == op.get("op") and e.value.details["path"] == op.get("path")
    else:
        assert e.value.details["op"] is None and e.value.details["path"] is None
    assert d.doc == before


def test_the_ops_must_be_a_list():
    d = axmp.Design(_channel())
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d.patch({"op": "replace", "path": "/name", "value": "x"})
    assert e.value.code == "patch.operation_failed"
    assert str(e.value).startswith("patch operations must be a list; got an object")
    assert e.value.where == ""


def test_a_later_operation_sees_the_result_of_an_earlier_one():
    """Applied one at a time to the copy: add an output, then edit its expression; and a test that holds only after
    the first operation."""
    d = axmp.Design(_channel())
    d.patch([
        {"op": "add", "path": "/outputs/Q_check", "value": {"kind": "expression", "expr": "Q_in"}},
        {"op": "test", "path": "/outputs/Q_check/expr", "value": "Q_in"},
        {"op": "replace", "path": "/outputs/Q_check/expr", "value": "Q_in + Q_out"},
    ])
    assert d.doc["outputs"]["Q_check"]["expr"] == "Q_in + Q_out"
    assert d.explain(json=True)["outputs"] == ["F_walls", "Q_in", "Q_out", "Q_check", "p_in", "p_out", "dp", "tau_max"]


# ----------------------------------------------------------------- success
def test_patch_applies_and_validates_a_model_document(tmp_path):
    doc = _channel()
    expected = copy.deepcopy(doc)
    expected["physics"]["flow"]["options"]["viscous_form"] = "stress"
    d = axmp.Design(doc)
    assert d.patch([OK_OP]) is d                      # chainable, as set() is
    assert d.doc == expected
    assert "viscous_form stress" in d.explain()

    dfile, ofile = _files(tmp_path, _channel(), [OK_OP])
    r = _run(dfile, ofile)                            # no -o: the patched document on stdout
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout) == expected
    out_file = tmp_path / "patched.json"
    r = _run(dfile, ofile, "-o", str(out_file))       # -o: the file is written, stdout says so
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout) == {"ok": True, "name": "channel", "written": str(out_file)}
    assert json.loads(out_file.read_text()) == expected
    assert json.loads(Path(dfile).read_text()) == _channel()      # the input file is never edited


def test_an_empty_patch_validates_and_returns_the_document():
    """The empty list of operations is a valid patch: the check that the document validates, and nothing else."""
    d = axmp.Design(_channel())
    d.patch([])
    assert d.doc == _channel()
    bad = _channel()
    bad["physics"]["flow"]["type"] = "fluid.bogus"
    with pytest.raises(axmp.AxionMultiPhysicsError):
        axmp.Design(bad).patch([])


def test_a_patch_whose_result_does_not_validate_is_refused_and_leaves_the_document(tmp_path):
    doc = _channel()
    before = copy.deepcopy(doc)
    ops = [{"op": "replace", "path": "/physics/flow/boundary_conditions/outlet/kind", "value": "nonexistent"}]
    d = axmp.Design(doc)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d.patch(ops)
    assert e.value.code == "physics.unknown_condition_kind"          # the validation's own error, not patch.operation_failed
    assert e.value.where == "/physics/flow/boundary_conditions/outlet/kind"
    assert d.doc == before
    dfile, ofile = _files(tmp_path, doc, ops)
    r = _run(dfile, ofile, "-o", str(tmp_path / "out.json"))
    assert r.returncode == 1 and json.loads(r.stdout)["code"] == "physics.unknown_condition_kind"
    assert not (tmp_path / "out.json").exists()


def test_paths_are_rfc_6901_pointers_with_tilde_escapes():
    """Section 12.1: the paths are JSON pointers. ~1 is '/', ~0 is '~' in a pointer; the error pointers of this library
    do not escape them (the pointer of a key 'a/b' is '/a/b'), so a patch must not read a pointer of an error back
    unescaped: it is given in RFC 6901 form. Here the key 'x/y~z' is added to the root and refused as an unknown
    field, whose message quotes the key as it is."""
    d = axmp.Design(_channel())
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d.patch([{"op": "add", "path": "/x~1y~0z", "value": 1}])
    assert "x/y~z" in str(e.value) and e.value.code != "patch.operation_failed"


def test_array_operations_follow_rfc_6902():
    """'-' appends, remove takes an element out, move removes then inserts, copy duplicates: the inlet velocity [1, 2]
    becomes [1, 2, 3] (not valid on its own), [1, 2] again, and then [2, 1] (a move of element 0 to index 1)."""
    doc = _channel()
    doc["physics"]["flow"]["boundary_conditions"]["inlet"]["velocity_m_s"] = [1, 2]
    v = "/physics/flow/boundary_conditions/inlet/velocity_m_s"
    d = axmp.Design(doc)
    d.patch([
        {"op": "add", "path": v + "/-", "value": 3},
        {"op": "test", "path": v, "value": [1, 2, 3]},
        {"op": "remove", "path": v + "/2"},
        {"op": "move", "from": v + "/0", "path": v + "/1"},
        {"op": "copy", "from": "/parameters/L", "path": "/parameters/L_again"},
    ])
    assert d.doc["physics"]["flow"]["boundary_conditions"]["inlet"]["velocity_m_s"] == [2, 1]
    assert d.doc["parameters"]["L_again"] == d.doc["parameters"]["L"]


# ----------------------------------------------------------------- electromagnetic documents
def test_patch_works_on_em_documents(tmp_path):
    """Section 12.1: 'It works on EM documents too.' A parameter edited, the result validated by the electromagnetic
    parser; a patch that makes the design invalid is refused with the parser's own message and leaves the design."""
    d = axmp.template("toroid")
    name = sorted(d.doc["parameters"])[0]
    old = d.doc["parameters"][name]
    new = old * 1.01 if isinstance(old, (int, float)) else old
    ops = [{"op": "replace", "path": f"/parameters/{name}", "value": new}]
    before = copy.deepcopy(d.doc)
    d.patch(ops)
    expected = copy.deepcopy(before)
    expected["parameters"][name] = new
    assert d.doc == expected
    d.validate()

    dfile, ofile = _files(tmp_path, before, ops)
    r = _run(dfile, ofile)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout) == expected

    d2 = axmp.template("toroid")
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d2.patch([{"op": "add", "path": "/bogus_key", "value": 1}])
    assert "bogus_key" in str(e.value) and e.value.code != "patch.operation_failed"
    assert d2.doc == before

    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d2.patch([{"op": "test", "path": "/name", "value": "nope"}])
    assert e.value.code == "patch.operation_failed" and e.value.where == "/0"


def test_a_move_into_its_own_child_is_refused_even_when_the_array_would_allow_it():
    """RFC 6902 section 4.4: "from" must not be a proper prefix of "path". The sequence that would otherwise copy a
    region, move the original into its own child and remove that child is refused at the move, and the document
    is unchanged."""
    d = axmp.Design(_channel())
    before = copy.deepcopy(d.doc)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d.patch([{"op": "copy", "from": "/regions/0", "path": "/regions/-"},
                 {"op": "move", "from": "/regions/1", "path": "/regions/1/x"},
                 {"op": "remove", "path": "/regions/1/x"}])
    assert e.value.code == "patch.operation_failed" and e.value.where == "/1"
    assert d.doc == before


def test_a_move_to_its_own_place_and_to_a_sibling_prefix_are_allowed():
    """The refusal is for a proper prefix only: moving a key onto itself, or to a path that merely starts with the same
    characters ('/parameters/L' to '/parameters/L2'), applies."""
    d = axmp.Design(_channel())
    d.patch([{"op": "move", "from": "/parameters/L", "path": "/parameters/L"},
             {"op": "move", "from": "/parameters/L", "path": "/parameters/L2"},
             {"op": "move", "from": "/parameters/L2", "path": "/parameters/L"}])
    assert d.doc["parameters"]["L"] == _channel()["parameters"]["L"]


def test_the_two_escapes_are_decoded_in_the_order_of_rfc_6901():
    """RFC 6901 section 4: '~1' becomes '/' first, then '~0' becomes '~'. So '/parameters/a~01b' addresses the key
    'a~1b' and '/parameters/a~1b' the key 'a/b': two different keys, which a decoder that turns '~0' into '~' first
    would take for the same one."""
    d = axmp.Design(_channel())
    d.patch([
        {"op": "add", "path": "/parameters/a~01b", "value": 1},
        {"op": "add", "path": "/parameters/a~1b", "value": 2},
        {"op": "test", "path": "/parameters/a~01b", "value": 1},
        {"op": "test", "path": "/parameters/a~1b", "value": 2},
        {"op": "remove", "path": "/parameters/a~01b"},
        {"op": "remove", "path": "/parameters/a~1b"},
    ])
    assert d.doc == _channel()
