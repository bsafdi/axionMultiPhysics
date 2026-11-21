"""A key repeated in one object of a JSON file (design note 15.2 row 15, Q7; development plan N14).

An electromagnetic document is read exactly as today: ``json::parse`` (and Python's ``json.loads`` in ``axmp.load``)
keeps the last of two values, silently. Only when the parsed document is a model document (it has the top-level key
"physics") is the text scanned a second time, and a repeated key is refused with ``json.duplicate_key`` at the RFC 6901
pointer of the second occurrence ('~' written ~0, '/' written ~1), by ``axmp``'s file reading and by ``axmp.load``.

Independent references: the last-value twin of each electromagnetic document (the same document written with the last
value only), whose ``axmp validate`` and ``axmp resolve`` output must be the duplicate's byte for byte; RFC 6901 for
the pointers; the texts of ``tests/model_switch_texts.py``. G0's corpus already records ``accepted_duplicate_key``
(``axmp validate`` of an electromagnetic document with a repeated key); no baseline changes here (plan N14).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core

import model_switch_texts as T

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = ROOT / "cpp" / "tests" / "data" / "model" / "channel.json"

_spec = importlib.util.spec_from_file_location("g0_harness_duplicate_key", ROOT / "tests" / "g0" / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)


def _axmp() -> str:
    exe = g0.find_axmp()
    if exe is None:
        pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")
    return str(exe)


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([_axmp(), *map(str, args)], capture_output=True, text=True, timeout=300)


# Electromagnetic documents with a repeated key, each with its last-value twin. The last one has a material named
# "physics": only a top-level key makes a model document.
EM_REGION = '"regions": [{"name": "x", "shape": {"rect": {"r": [0, "R"], "z": [0, 1]}}}]'
EM_CASES = {
    "top_level_name": (
        '{"name": "first", "parameters": {"R": 1}, ' + EM_REGION + ', "name": "second"}',
        '{"parameters": {"R": 1}, ' + EM_REGION + ', "name": "second"}'),
    "parameter": (
        '{"name": "p", "parameters": {"R": 1, "R": 2}, ' + EM_REGION + '}',
        '{"name": "p", "parameters": {"R": 2}, ' + EM_REGION + '}'),
    "nested_in_a_region": (
        '{"name": "q", "parameters": {"R": 1}, "regions": [{"name": "x", "name": "y", '
        '"shape": {"rect": {"r": [0, "R"], "z": [0, 1]}}}]}',
        '{"name": "q", "parameters": {"R": 1}, "regions": [{"name": "y", '
        '"shape": {"rect": {"r": [0, "R"], "z": [0, 1]}}}]}'),
    "material_named_physics": (
        '{"name": "m", "parameters": {"R": 1}, "materials": {"physics": {"kind": "linear", "mu_r": 2, "mu_r": 3}}, '
        '"regions": [{"name": "x", "material": "physics", "shape": {"rect": {"r": [0, "R"], "z": [0, 1]}}}]}',
        '{"name": "m", "parameters": {"R": 1}, "materials": {"physics": {"kind": "linear", "mu_r": 3}}, '
        '"regions": [{"name": "x", "material": "physics", "shape": {"rect": {"r": [0, "R"], "z": [0, 1]}}}]}'),
}


@pytest.mark.parametrize("case", sorted(EM_CASES))
def test_em_duplicate_key_behaves_as_today(tmp_path, case):
    """``axmp validate`` and ``axmp resolve`` print the twin's stdout byte for byte with the twin's exit status (0),
    and ``axmp.load`` returns the twin's document (the last value kept)."""
    dup_text, twin_text = EM_CASES[case]
    assert json.loads(dup_text) == json.loads(twin_text), "the twin is the last-value reading of the duplicate"
    dup, twin = tmp_path / "dup.json", tmp_path / "twin.json"
    dup.write_text(dup_text, encoding="utf-8")
    twin.write_text(twin_text, encoding="utf-8")
    for cmd in ("validate", "resolve"):
        a, b = _run(cmd, dup), _run(cmd, twin)
        assert (a.returncode, a.stdout) == (b.returncode, b.stdout), f"{cmd}: {a.stdout}\nvs\n{b.stdout}"
        assert a.returncode == 0, a.stdout + a.stderr
    d = axmp.load(dup)
    assert d.doc == json.loads(twin_text)
    assert d.validate() == axmp.load(twin).validate()


def _model_text_with(insert_at: str, fragment: str) -> str:
    """channel.json's text with ``fragment`` inserted right after the first occurrence of ``insert_at``."""
    text = CHANNEL.read_text("utf-8")
    i = text.index(insert_at) + len(insert_at)
    return text[:i] + fragment + text[i:]


MODEL_CASES = {
    # key, pointer of the second occurrence, the text
    "top_level": ("name", "/name", _model_text_with('"name": "channel",', ' "name": "channel",')),
    "physics_instance_key": ("type", "/physics/flow/type",
                             _model_text_with('"type": "fluid.incompressible",', ' "type": "fluid.incompressible",')),
    "key_with_a_slash": ("a/b", "/parameters/a~1b", _model_text_with('"parameters": {', '"a/b": 1, "a/b": 2, ')),
    "key_with_a_tilde": ("x~y", "/parameters/x~0y", _model_text_with('"parameters": {', '"x~y": 1, "x~y": 2, ')),
    "in_an_array": ("of", "/couplings/0/of",
                    _model_text_with('"physics": {', '').replace(
                        '"outputs": {', '"couplings": [{"of": 1, "of": 2}], "outputs": {', 1)),
}


@pytest.mark.parametrize("case", sorted(MODEL_CASES))
def test_model_duplicate_key_is_refused(tmp_path, case):
    """``axmp validate`` and ``axmp resolve`` (and ``axmp mesh``: the file is read before the command) print
    ``json.duplicate_key`` at the pointer of the second occurrence and exit 1; ``axmp.load`` raises the same error;
    the message names the file as it was given."""
    key, pointer, text = MODEL_CASES[case]
    p = tmp_path / "model.json"
    p.write_text(text, encoding="utf-8")
    assert "physics" in json.loads(text)
    want = T.duplicate_key(key, pointer, str(p))
    for cmd in ("validate", "resolve", "mesh", "explain"):
        r = _run(cmd, p)
        assert r.returncode == 1, f"{cmd}: {r.stdout}{r.stderr}"
        assert json.loads(r.stdout) == T.cli_json(want), f"{cmd}: {r.stdout}"
    ops = tmp_path / "ops.json"
    ops.write_text("[]", encoding="utf-8")
    r = _run("patch", p, ops)
    assert r.returncode == 1, f"patch: {r.stdout}{r.stderr}"
    assert json.loads(r.stdout) == T.cli_json(want), f"patch: {r.stdout}"
    with pytest.raises(Exception) as ei:
        axmp.load(p)
    T.assert_exception(ei.value, want)
    with pytest.raises(Exception) as ei:
        axmp.load(str(p))
    T.assert_exception(ei.value, want)


@pytest.mark.skipif(not Path("/dev/stdin").exists(), reason="no /dev/stdin on this platform")
def test_a_pipe_is_scanned_too(tmp_path):
    """A file that can be read only once (``cat model.json | axmp validate /dev/stdin``; a process substitution) is
    scanned as well: the text is read once, then parsed and scanned. An electromagnetic document from a pipe keeps the
    last value, as its twin read from a file does."""
    def piped(cmd: str, text: str) -> subprocess.CompletedProcess:
        return subprocess.run([_axmp(), cmd, "/dev/stdin"], input=text, capture_output=True, text=True, timeout=300)

    key, pointer, text = MODEL_CASES["top_level"]
    want = T.cli_json(T.duplicate_key(key, pointer, "/dev/stdin"))
    for cmd in ("validate", "resolve"):
        r = piped(cmd, text)
        assert r.returncode == 1, f"{cmd}: {r.stdout}{r.stderr}"
        assert json.loads(r.stdout) == want, f"{cmd}: {r.stdout}"
    dup_text, twin_text = EM_CASES["top_level_name"]
    twin = tmp_path / "twin.json"
    twin.write_text(twin_text, encoding="utf-8")
    for cmd in ("validate", "resolve"):
        a, b = piped(cmd, dup_text), _run(cmd, twin)
        assert (a.returncode, a.stdout) == (b.returncode, b.stdout) and a.returncode == 0, f"{cmd}: {a.stdout}{a.stderr}"


def test_the_scan_is_a_binding():
    """``_core.scan_duplicate_keys(text, origin)``: None for a text without a repeated key (the same key in two
    objects is no repetition), the error otherwise; the first repetition in document order is reported."""
    assert _core.scan_duplicate_keys('{"a": {"k": 1}, "b": {"k": 2}, "k": 3}', "t.json") is None
    with pytest.raises(Exception) as ei:
        _core.scan_duplicate_keys('{"b": {"k": 1, "k": 2}, "a": 1, "a": 2}', "t.json")
    T.assert_exception(ei.value, T.duplicate_key("k", "/b/k", "t.json"))


def test_a_key_repeated_in_the_patch_operations_is_refused_for_a_model_document(tmp_path):
    """JSON keeps the last value of a repeated key, silently. The operations file of ``axmp patch`` is scanned like the
    document when the document (or what the patch makes of it) is a model document; the pointer is into the operations
    file. For an electromagnetic design that stays an electromagnetic design the operations are read as before."""
    ops_text = ('[{"op": "replace", "path": "/physics/flow/options", '
                '"value": {"equation": "stokes", "equation": "navier_stokes"}}]')
    ops = tmp_path / "ops.json"
    ops.write_text(ops_text, encoding="utf-8")
    r = _run("patch", CHANNEL, ops)
    assert r.returncode == 1, r.stdout + r.stderr
    assert json.loads(r.stdout) == T.cli_json(T.duplicate_key("equation", "/0/value/equation", str(ops)))
    em_ops = tmp_path / "em_ops.json"
    em_ops.write_text('[{"op": "add", "path": "/notes", "value": "x", "value": "y"}]', encoding="utf-8")
    toroid = tmp_path / "toroid.json"
    toroid.write_text(json.dumps(axmp.template("toroid").doc), encoding="utf-8")
    r = _run("patch", toroid, em_ops)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["notes"] == "y"
