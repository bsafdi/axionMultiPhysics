"""The structured fields of an error: the Python exception's ``code``, ``where``, ``hint`` and ``details``, the
command line's error JSON, and the MCP server's error payload.

Two halves.

* **Guards** (they pass as the package stands and must keep passing): the exception class, its base classes,
  ``str(e)``, ``e.args``, pickling, the exception type of every document of the compatibility corpus, the four
  keys of the command line's error JSON and its exit status, and the MCP payload of every corpus error against the
  message-only payload, which this file carries as its reference.
* **The new behaviour**: the four fields exist with their defaults, a translated
  error carries them, and the two new keys reach the command line and the MCP payload only when they are not
  empty.

No electromagnetic error has a code, so nothing in the library raises one; the fields are exercised through
``_core._error_for_tests``, a private entry point for exactly that (``cpp/tests/test_error_fields.cpp`` is the
C++ half).

Independent references: the message-only payload (pinned below as ``_err_from_the_message``), the regular
expression it uses read against the message that ``compose()`` builds, and the recorded corpus of refused
documents.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core

ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "tests" / "g0"

_spec = importlib.util.spec_from_file_location("g0_harness_errors", G0 / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)

ERROR = _core.AxionMultiPhysicsError

# Corpus documents that the JSON library itself refuses, so they arrive as plain RuntimeError and not as an
# axionMultiPhysics error. The translator must not take them over.
NOT_OUR_ERROR = {"windings_not_list", "material_thermal_missing_k"}

CODE = "boundary.empty_selection"
DETAILS = {"selector": {"of": "channel", "where": {"x": 0}}, "candidates": [{"where": {"x": 0.0005}}]}
WHAT = "boundary 'inlet' selects nothing"
WHERE = "/boundaries/inlet/where"
HINT = "no boundary edge of 'channel' lies on x = 0."


# ----------------------------------------------------------------- the reference payload
def _err_from_the_message(e: Exception) -> dict:
    """The payload read from the message text alone, as ``axionmultiphysics.mcp_server._err`` builds it in 0.1.0.
    The payload of an error without a code is this, key for key and in this order."""
    msg = str(e)
    out = {"ok": False, "error": msg}
    m = re.search(r"\[at ([^\]]*)\]", msg)
    if m:
        out["where"] = m.group(1)
    if "Hint:" in msg:
        out["hint"] = msg.split("Hint:", 1)[1].strip()
    return out


def _err():
    """The server's ``_err``; the MCP package is an extra, so a build without it skips."""
    pytest.importorskip("mcp", reason="the MCP server needs the 'mcp' package")
    from axionmultiphysics import mcp_server

    return mcp_server._err


def corpus_text(cid: str) -> str:
    d = next(x for x in g0.load_corpus() if x["id"] == cid)
    return d["text"] if "text" in d else json.dumps(d["doc"])


@functools.lru_cache(maxsize=1)
def corpus_errors():
    """[(id, exception)] for every document of the corpus that is refused."""
    out = []
    for d in g0.load_corpus():
        text = d["text"] if "text" in d else json.dumps(d["doc"])
        try:
            _core.validate(text)
        except Exception as e:  # noqa: BLE001 - the exception is the datum
            out.append((d["id"], e.with_traceback(None)))
    return tuple(out)


class Fake(Exception):
    """An exception that carries the four fields without going through the library."""

    def __init__(self, message, where="", hint="", code="", details=None):
        super().__init__(message)
        self.where, self.hint, self.code, self.details = where, hint, code, details


# ----------------------------------------------------------------- guards: the exception class
def test_the_corpus_is_refused_and_two_of_its_documents_are_not_our_error():
    errs = corpus_errors()
    assert len(errs) >= 70, f"only {len(errs)} corpus documents were refused"
    ids = {i for i, _ in errs}
    assert NOT_OUR_ERROR <= ids, f"the corpus no longer holds {NOT_OUR_ERROR - ids}"
    for cid, e in errs:
        if cid in NOT_OUR_ERROR:
            assert type(e) is RuntimeError, f"{cid}: {type(e)}"
        else:
            assert type(e) is ERROR, f"{cid}: {type(e)}"
        assert e.args == (str(e),), cid
        assert str(e), cid


def test_the_exception_class_keeps_its_name_module_and_base_classes():
    assert ERROR.__module__ == "axionmultiphysics._core"
    assert ERROR.__qualname__ == "AxionMultiPhysicsError"
    assert ERROR.__mro__ == (ERROR, Exception, BaseException, object)
    assert axmp.AxionMultiPhysicsError is ERROR
    if axmp.build_info().get("fluids"):
        fluids = pytest.importorskip("axionmultiphysics.fluids", exc_type=ImportError)
        assert fluids.FluidsError is ERROR


def test_the_message_carries_the_pointer_and_the_hint_exactly_once():
    """The precondition of the regular expression that the message-only payload and the MCP worker use: a message
    built by ``compose()`` holds at most one ``[at ...]`` and at most one ``Hint:``. A message with two of either
    would be read differently from the text than from the fields, and its payload would move."""
    for cid, e in corpus_errors():
        msg = str(e)
        assert msg.count("[at ") <= 1, cid
        assert msg.count("Hint:") <= 1, cid


def test_pickling_keeps_the_type_the_text_and_the_arguments():
    """The MCP server runs long jobs in a worker process; an error must survive the boundary unchanged."""
    for cid, e in corpus_errors():
        back = pickle.loads(pickle.dumps(e))
        assert type(back) is type(e), cid
        assert str(back) == str(e), cid
        assert back.args == e.args, cid


# ----------------------------------------------------------------- guards: the MCP payload
def test_the_mcp_payload_of_every_refused_document_is_the_message_only_payload():
    err = _err()
    for cid, e in corpus_errors():
        now, reference = err(e), _err_from_the_message(e)
        assert list(now.items()) == list(reference.items()), f"{cid}: {now} != {reference}"


def test_the_mcp_payload_reads_the_message_of_an_exception_without_the_fields():
    """Not every exception comes with the fields: other code paths raise ValueError or OSError, and the class can be
    raised from Python without them. Such an exception has no fields (or empty ones), so the pointer and the hint
    must still be read from the text."""
    err = _err()
    text = "MPI solve failed [at /regions/2/material] Hint: Give the region a material."
    for e in (ValueError(text), FileNotFoundError(text), ERROR(text), Fake(text)):
        got = err(e)
        assert got == {"ok": False, "error": text, "where": "/regions/2/material",
                       "hint": "Give the region a material."}, type(e).__name__
        assert list(got) == ["ok", "error", "where", "hint"], type(e).__name__


def test_the_mcp_payload_of_an_exception_without_a_message_pointer_has_two_keys():
    err = _err()
    assert err(ValueError("no design named 'x'")) == {"ok": False, "error": "no design named 'x'"}


def test_the_mcp_payload_omits_the_two_new_keys_when_the_error_has_none():
    """An empty field adds no key: that is what keeps the payload of every error without a code identical. It
    holds for the message-only payload as well, so this is a guard and not new behaviour."""
    err = _err()
    assert err(Fake("something went wrong [at /x] Hint: fix it")) == {
        "ok": False, "error": "something went wrong [at /x] Hint: fix it", "where": "/x", "hint": "fix it"}
    assert err(Fake("something went wrong", where="", hint="", code="", details={})) == {
        "ok": False, "error": "something went wrong"}
    assert err(Fake("something went wrong", code="", details=None)) == {
        "ok": False, "error": "something went wrong"}


# ----------------------------------------------------------------- guards: the command line
def _axmp():
    exe = g0.find_axmp()
    if exe is None:
        pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")
    return exe


def _validate(tmp_path, text, name="doc.json"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    r = subprocess.run([_axmp(), "validate", str(p)], capture_output=True, text=True, timeout=300)
    return r


def test_the_command_line_prints_four_keys_and_exits_1_for_a_refused_design(tmp_path):
    """No error of the library has a code, so the error JSON of every refused design is exactly four keys:
    ``code`` and ``details`` must not appear, not even empty."""
    ours = [cid for cid, _ in corpus_errors() if cid not in NOT_OUR_ERROR]
    assert len(ours) >= 12
    for n, cid in enumerate(ours[:12]):
        r = _validate(tmp_path, corpus_text(cid), name=f"{n}.json")
        assert r.returncode == 1, f"{cid}: exit {r.returncode}\n{r.stdout}"
        out = json.loads(r.stdout)
        assert set(out) == {"ok", "error", "where", "hint"}, f"{cid}: {sorted(out)}"
        assert out["ok"] is False, cid


def test_the_command_line_prints_two_keys_for_an_error_that_is_not_ours(tmp_path):
    """An exception that is not an axionMultiPhysics error is reported by the second catch clause, which prints
    neither a pointer nor a hint, and neither of the two new keys."""
    for n, cid in enumerate(sorted(NOT_OUR_ERROR)):
        r = _validate(tmp_path, corpus_text(cid), name=f"other{n}.json")
        assert r.returncode == 1, cid
        assert set(json.loads(r.stdout)) == {"ok", "error"}, cid


def test_the_command_line_exits_2_on_a_usage_error():
    r = subprocess.run([_axmp(), "no-such-command"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2
    assert "usage:" in (r.stdout + r.stderr)


# ----------------------------------------------------------------- the new behaviour
def test_the_exception_offers_the_four_fields_with_their_defaults():
    """Every instance has the four attributes, whatever raised it: code, where and hint are strings, empty when
    the error has none, and details is None. Code in the package raises the class directly, so the defaults
    belong to the class, not only to the instances the translator builds."""
    assert (ERROR.code, ERROR.where, ERROR.hint, ERROR.details) == ("", "", "", None)
    e = ERROR("plain message")
    assert (e.code, e.where, e.hint, e.details) == ("", "", "", None)
    assert str(e) == "plain message" and e.args == ("plain message",)


def test_a_refused_document_reports_the_pointer_and_the_hint_in_its_fields():
    """The fields of an error the library raises agree with the message ``compose()`` built from them, and no
    electromagnetic error has a code."""
    with_pointer = 0
    with_hint = 0
    for cid, e in corpus_errors():
        if cid in NOT_OUR_ERROR:
            continue
        msg = str(e)
        m = re.search(r"\[at ([^\]]*)\]", msg)
        assert e.where == (m.group(1) if m else ""), cid
        assert e.hint == (msg.split("Hint:", 1)[1].strip() if "Hint:" in msg else ""), cid
        assert e.code == "", f"{cid} now has the code {e.code!r}: that is a baseline change of its own"
        assert e.details is None, cid
        with_pointer += bool(e.where)
        with_hint += bool(e.hint)
    assert with_pointer >= 60 and with_hint >= 40, (with_pointer, with_hint)


def test_the_translator_sets_the_fields_of_an_error_that_has_them():
    with pytest.raises(ERROR) as excinfo:
        _core._error_for_tests(WHAT, WHERE, HINT, CODE, json.dumps(DETAILS))
    e = excinfo.value
    assert type(e) is ERROR
    assert e.code == CODE
    assert e.details == DETAILS
    assert e.where == WHERE
    assert e.hint == HINT
    composed = f"{WHAT} [at {WHERE}] Hint: {HINT}"
    assert str(e) == composed, "the code or the details reached the message"
    assert e.args == (composed,)


def test_the_translator_is_consulted_before_the_built_in_one():
    """nanobind's own translator for this exception type builds the instance from the message alone. If it ran
    first, the fields would be the class defaults, so a non-empty code on the instance is the proof of order.
    The attributes are set on the instance, which ``vars(e)`` shows."""
    with pytest.raises(ERROR) as excinfo:
        _core._error_for_tests(WHAT, WHERE, HINT, CODE, json.dumps(DETAILS))
    e = excinfo.value
    assert e.code == CODE, "the built-in translator built this instance: the fields were lost"
    assert {"code", "where", "hint", "details"} <= set(vars(e))


def test_a_solver_failure_arrives_as_the_same_class_with_its_fields():
    """``axmp::SolverFailure`` derives from the error class; the translator catches the base, so Python sees one
    exception type for the whole library."""
    with pytest.raises(ERROR) as excinfo:
        _core._error_for_tests(WHAT, WHERE, HINT, CODE, json.dumps(DETAILS), solver_failure=True)
    e = excinfo.value
    assert type(e) is ERROR
    assert (e.code, e.where, e.hint) == (CODE, WHERE, HINT)
    assert e.details == DETAILS


def test_an_error_without_a_code_arrives_with_the_defaults():
    with pytest.raises(ERROR) as excinfo:
        _core._error_for_tests(WHAT, WHERE, HINT)
    e = excinfo.value
    assert (e.code, e.details) == ("", None)
    assert (e.where, e.hint) == (WHERE, HINT)
    assert str(e) == f"{WHAT} [at {WHERE}] Hint: {HINT}"


def test_details_that_are_not_an_object_are_dropped_and_the_error_still_arrives():
    """Reporting an error must not fail while it is being reported: malformed details are lost, the rest is not."""
    for bad in ("not json", "[1, 2]", "7", '{"a":'):
        with pytest.raises(ERROR) as excinfo:
            _core._error_for_tests(WHAT, WHERE, HINT, CODE, bad)
        e = excinfo.value
        assert e.details is None, bad
        assert e.code == CODE, bad
        assert str(e) == f"{WHAT} [at {WHERE}] Hint: {HINT}", bad


def test_the_exception_has_details_exactly_when_the_command_line_prints_them():
    """The command line reads the details with a strict JSON parser and leaves out an object with no members. The
    exception must agree: ``NaN`` and ``Infinity`` are not JSON (Python's own reader accepts them, and a payload
    holding one is not a JSON reply), and ``{}`` is no details. A number too large for a double is refused by the
    command line's parser, where Python's reads an infinity."""
    err = _err()
    for bad in ('{"r": NaN}', '{"r": Infinity}', '{"r": -Infinity}', '{"a": {"b": [1, NaN]}}', '{"r": 1e999, "n": 2}',
                "{}", "{ }"):
        with pytest.raises(ERROR) as excinfo:
            _core._error_for_tests(WHAT, WHERE, HINT, CODE, bad)
        e = excinfo.value
        assert e.details is None, bad
        assert e.code == CODE, bad
        got = err(e)
        assert "details" not in got, bad
        json.dumps(got, allow_nan=False)


def test_the_exception_class_is_documented_with_its_attributes():
    doc = ERROR.__doc__
    assert doc, "AxionMultiPhysicsError has no docstring"
    for name in ("code", "where", "hint", "details"):
        assert name in doc, name


def _nested(levels):
    """A details object nested ``levels`` deep in all (the object itself is level 1)."""
    return '{"a":' * (levels - 1) + "[1]" + "}" * (levels - 1) if levels > 1 else "{}"


def _levels_cases():
    return ((2, True), (63, True), (64, True), (65, False), (200000, False))


def _raise_with_details(text):
    with pytest.raises(ERROR) as excinfo:
        _core._error_for_tests(WHAT, WHERE, HINT, CODE, text)
    return excinfo.value


def test_details_nested_beyond_a_fixed_depth_are_dropped():
    """Details are reported to a depth of 64 (the details object is level 1) and are None beyond it, so that no
    surface recurses without bound; the exception and the command line's payload read the same function."""
    for levels, kept in _levels_cases():
        text = _nested(levels)  # the array in the innermost object counts as a level too, so this is `levels` deep
        e = _raise_with_details(text)
        assert (e.details is not None) == kept, levels
        assert e.code == CODE, levels
    # an ordinary nested case is returned whole
    e = _raise_with_details('{"a": {"b": [1, {"c": [2, 3]}]}, "s": "[[{"}')
    assert e.details == {"a": {"b": [1, {"c": [2, 3]}]}, "s": "[[{"}
    # brackets inside a string, also after an escaped quote, are text and not nesting
    many = "[" * 200 + "{" * 200
    for text, want in (('{"s": "%s"}' % many, {"s": many}), ('{"s": "\\"%s"}' % many, {"s": '"' + many})):
        assert _raise_with_details(text).details == want
    # an escaped backslash ends no escape: the quote after it closes the string, so the brackets of the string that
    # follows are text too
    e = _raise_with_details('{"s": "\\\\", "t": "%s"}' % many)
    assert e.details == {"s": "\\", "t": many}
    # the depth falls again when a level closes: many siblings are not nesting
    e = _raise_with_details("{" + ", ".join('"k%d": [1]' % i for i in range(70)) + "}")
    assert e.details is not None and len(e.details) == 70
    e = _raise_with_details('{"l": [' + ", ".join('{"x": 1}' for _ in range(100)) + "]}")
    assert e.details is not None and len(e.details["l"]) == 100


def test_the_mcp_payload_drops_details_nested_beyond_a_fixed_depth():
    """The MCP payload reads the same function (where the mcp extra is installed)."""
    err = _err()
    for levels, kept in _levels_cases():
        e = _raise_with_details(_nested(levels))
        assert ("details" in err(e)) == kept, levels


def test_the_mcp_payload_carries_the_code_and_the_details_when_they_are_there():
    err = _err()
    with pytest.raises(ERROR) as excinfo:
        _core._error_for_tests(WHAT, WHERE, HINT, CODE, json.dumps(DETAILS))
    got = err(excinfo.value)
    assert got["ok"] is False
    assert got["error"] == str(excinfo.value), "the payload's text is the whole message, as it was"
    assert got["where"] == WHERE
    assert got["hint"] == HINT
    assert got["code"] == CODE
    assert got["details"] == DETAILS
    assert set(got) == {"ok", "error", "where", "hint", "code", "details"}
    json.dumps(got, allow_nan=False)  # every tool's reply must be plain JSON


def test_the_mcp_payload_prefers_the_fields_over_the_text():
    """Where the fields and the text disagree the fields win: they are what the error says, the text is a
    rendering of it. No message of the library disagrees (the test on the corpus above checks that), so no payload
    of an error the library raises moves."""
    err = _err()
    got = err(Fake("odd [at /wrong] Hint: no", where="/right", hint="yes", code="a.b", details={"k": 1}))
    assert got["where"] == "/right"
    assert got["hint"] == "yes"
    assert got["code"] == "a.b"
    assert got["details"] == {"k": 1}


def test_pickling_keeps_the_structured_fields():
    """The worker process reports a failure to the server; the fields must cross with the error."""
    with pytest.raises(ERROR) as excinfo:
        _core._error_for_tests(WHAT, WHERE, HINT, CODE, json.dumps(DETAILS))
    back = pickle.loads(pickle.dumps(excinfo.value))
    assert type(back) is ERROR
    assert str(back) == str(excinfo.value)
    assert back.args == excinfo.value.args
    assert (back.code, back.where, back.hint, back.details) == (CODE, WHERE, HINT, DETAILS)


def test_the_command_line_builds_its_error_json_in_one_place():
    """The two keys are printed when they are not empty, and the C++ test checks that function's output; this
    checks that the command line is the caller, so the keys cannot be forgotten on one path."""
    helper = ROOT / "cpp" / "src" / "error_json.hpp"
    assert helper.is_file(), f"{helper} does not exist"
    app = (ROOT / "cpp" / "apps" / "axmp.cpp").read_text("utf-8")
    body = app.split("catch (axmp::Error& e)", 1)
    assert len(body) == 2, "cpp/apps/axmp.cpp no longer catches axmp::Error by that spelling"
    caught = body[1].split("catch (", 1)[0]
    assert "error_json(" in caught, f"the error JSON is still built by hand:\n{caught}"


def test_every_solver_failure_rethrow_keeps_the_fields_through_with_hint_note():
    """A solver failure that is rethrown with a note must keep the code and the details of the original, which only
    ``with_hint_note`` does; no failure has a code yet, so no run can show the difference."""
    for name in ("model.cpp", "model3d_fullwave.cpp"):
        src = (ROOT / "cpp" / "src" / name).read_text("utf-8")
        blocks = src.split("catch (const SolverFailure&")[1:]
        assert blocks, f"{name} no longer catches SolverFailure by that spelling"
        for block in blocks:
            body = block.split("\n}", 1)[0]
            if "throw" in body:
                assert "with_hint_note(" in body, f"{name}: a SolverFailure is rethrown without with_hint_note:\n{body}"


def test_the_mcp_payload_ignores_a_code_or_details_of_the_wrong_type():
    """An unrelated exception may have attributes of these names (xml.etree.ElementTree.ParseError.code is an int)."""
    err = _err()
    assert err(Fake("x", code=3, details=[1, 2])) == {"ok": False, "error": "x"}
    assert err(Fake("x", code="a.b", details="text")) == {"ok": False, "error": "x", "code": "a.b"}
    for bad in ({"residual": float("nan")}, {"regions": {1, 2}}, {"r": float("inf")}):
        got = err(Fake("x", code="a.b", details=bad))
        assert got == {"ok": False, "error": "x", "code": "a.b"}, bad
        json.dumps(got, allow_nan=False)


def test_the_mcp_payload_is_built_for_an_exception_that_cannot_be_read():
    """The payload is built in the ``except`` branch of every tool: if building it raised, the tool would return
    nothing. An attribute that raises is treated as absent, and the text is then read from the message."""
    err = _err()

    class Odd(Exception):
        @property
        def details(self):
            raise RuntimeError("details unavailable")

        @property
        def where(self):
            raise KeyError("where")

        code = "a.b"

    assert err(Odd("something failed [at /x] Hint: retry")) == {
        "ok": False, "error": "something failed [at /x] Hint: retry", "where": "/x", "hint": "retry", "code": "a.b"}

    class Mute(Exception):
        def __str__(self):
            raise RuntimeError("no text")

    assert err(Mute()) == {"ok": False, "error": "Mute"}


def test_the_worker_writes_the_same_payload_as_the_server(tmp_path):
    """The long-running tools report through the worker, which must not keep a copy of the payload that drops the
    code and the details."""
    pytest.importorskip("mcp", reason="the MCP server needs the 'mcp' package")
    from axionmultiphysics import _mcp_worker, mcp_server

    e = Fake("failed [at /study/0] Hint: lower the step", where="/study/0", hint="lower the step",
             code="solve.not_converged", details={"residual": 3.2})
    out = tmp_path / "result.json"
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"kind": "boom", "result": str(out)}))

    def boom(_job):
        raise e

    old_argv, _mcp_worker.JOBS["boom"] = sys.argv, boom
    sys.argv = ["worker", str(job)]
    try:
        rc = _mcp_worker.main()
    finally:
        sys.argv = old_argv
        del _mcp_worker.JOBS["boom"]
    assert rc == 1
    assert json.loads(out.read_text()) == mcp_server._err(e)
    assert json.loads(out.read_text())["code"] == "solve.not_converged"
