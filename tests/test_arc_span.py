"""An arc path whose angles span more than one turn is refused by validation.

An ``arc`` path is ``{"center": [r, z], "radius": R, "angles": [a0, a1]}`` with the angles in RADIANS. A span
``|a1 - a0|`` above 2 pi (degrees written where radians are meant, e.g. ``[-60, 60]``), and an angle that is not a
finite number, is refused with the code ``geometry.arc_span`` at ``<path>/arc/angles``, with a hint that the angles are
radians; when the span would be a sensible number of degrees (at most 360) the hint also gives the two values
converted to radians. A span of at most 2 pi is accepted (a full turn included). Reversed angles (a1 < a0) are refused today as a range
that does not increase, with no code; that stays as it is.

Every place a user can reach an arc path is covered: the curves of an electromagnetic design and the curves of a model
document, each through ``axmp validate``, ``axmp mesh`` and the Python entry points (``Design.validate``, and for
the model document ``axmp.mesh``). Every run that may abort the process is a subprocess, so an abort is a failure of
the test and not the end of the test session.

In an axisymmetric design an arc any of whose points has r < -1e-12 m is refused with the code
``geometry.arc_negative_r`` at ``<path>/arc`` (the section "the axis" below); a planar design has no axis.

Independent reference: 2 pi and the degree-to-radian conversion are computed here (``math``); the refused spans are
chosen on both sides of the bound by margins far larger than a rounding error, and the exact bound (2 pi, written two
ways) is checked as accepted.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import model_mesh_support as S
from model_mesh_support import mesher  # noqa: F401 (a fixture)

ROOT = Path(__file__).resolve().parents[1]
CODE = "geometry.arc_span"
TWO_PI = 2 * math.pi

# the spans that are refused: [a0, a1] as written in the document
REFUSED = [
    pytest.param([-60, 60], id="degrees_pm60"),
    pytest.param([0, 360], id="degrees_360"),
    pytest.param([0, 90], id="degrees_90"),
    pytest.param([0, TWO_PI + 0.01], id="just_over_a_turn"),
    pytest.param([-4, 4], id="symmetric_over"),
    pytest.param([0, 1000], id="not_degrees"),
]
# the spans that are accepted by validation
ACCEPTED = [
    pytest.param([0, 1.2], id="short"),
    pytest.param([-math.pi / 2, math.pi / 2], id="half_turn"),
    pytest.param([0, TWO_PI], id="full_turn"),
    pytest.param([-math.pi, math.pi], id="full_turn_centred"),
    pytest.param([5, 5 + 6.2], id="far_from_zero_short_span"),
]


# ------------------------------------------------------------------------------------------------ the documents
def em_design(angles, *, center=(2.0, 0.0), radius=0.75, symmetry=None) -> dict:
    """An electromagnetic design: a rectangle and an open arc sheet (curves) of the given angles. The default centre
    is off the axis (r = 2 > radius), so no point of the arc is at a negative r whatever the angles; ``center=(0, 0)``
    puts it on the axis."""
    d = {"schema": "axionmultiphysics.design/1", "name": "arc_span_em",
         "regions": [{"name": "box", "shape": {"rect": {"r": [0.0, 0.5], "z": [-0.2, 0.2]}}}],
         "curves": [{"name": "sheet",
                     "path": {"arc": {"center": list(center), "radius": radius, "angles": angles}},
                     "material": "superconductor", "mesh_size": 0.05}],
         "mesh": {"size": 0.1, "domain_radius": 1.5 + center[0]}}
    if symmetry == "planar":
        d["symmetry"] = "planar"
        d["regions"] = [{"name": "box", "shape": {"rect": {"x": [-0.5, 0.5], "y": [-0.2, 0.2]}}}]
    return d


def model_document(angles, *, parameters: dict | None = None, center=("R", "H/2"), radius="R/2") -> dict:
    """The model document of curves_axisymmetric.json with the angles of its arc curve ``cut`` replaced; the centre
    is at r = R, so that no point of the arc is at a negative r (``center=(0, "H/2")`` puts it on the axis)."""
    d = S.doc("curves_axisymmetric")
    d["materials"] = {"m": {"density": 1.0, "thermal": {"k_W_mK": 1.0, "cp_J_kgK": 1.0}}}
    for r in d["regions"]:
        r["material"] = "m"
    for c in d["curves"]:
        if c["name"] == "cut":
            c["path"]["arc"].update({"angles": angles, "center": list(center), "radius": radius})
    if parameters:
        d["parameters"].update(parameters)
    return d


def bad_pointer(document: dict, kind: str) -> str:
    """The JSON pointer of the ``angles`` of the arc that the document carries."""
    index = [c["name"] for c in document["curves"]].index("sheet" if kind == "em" else "cut")
    return f"/curves/{index}/path/arc/angles"


DOCUMENTS = {"em": em_design, "model": model_document}
# the arc on the axis: half-plane angles only (the documents that mesh)
ON_AXIS = {"em": lambda a: em_design(a, center=(0.0, 0.0)),
           "model": lambda a: model_document(a, center=(0, "H/2"))}


# ------------------------------------------------------------------------------------------------ the runs
def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def python(code: str, *args, timeout: int = 300) -> subprocess.CompletedProcess:
    """A Python program in its own process: an abort is a return code, not the end of the session."""
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True,
                          timeout=timeout, env=_env())


def cli(*args, timeout: int = 300) -> subprocess.CompletedProcess:
    return S.run(*args, timeout=timeout)


def _no_abort(r: subprocess.CompletedProcess, what: str) -> None:
    assert r.returncode >= 0, f"{what}: the process was killed by signal {-r.returncode}\nstderr: {r.stderr[-400:]}"
    assert "double free" not in r.stderr and "corruption" not in r.stderr, f"{what}: memory corruption\n{r.stderr[-400:]}"


def _error_json(r: subprocess.CompletedProcess, what: str) -> dict:
    """The error JSON of a refusal: exit status 1 and one object on stdout."""
    _no_abort(r, what)
    assert r.returncode == 1, f"{what}: status {r.returncode}\nstdout: {r.stdout[-400:]}\nstderr: {r.stderr[-400:]}"
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"{what}: stdout is not JSON ({e}): {r.stdout[-400:]}")


def _floats(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]


def check_refusal(f: dict, *, where: str, angles, what: str) -> None:
    """The finding: code, pointer, a hint that says radians, and the converted values when the numbers look like
    degrees. ``f`` has the keys ``error``, ``where``, ``hint``, ``code``."""
    assert f.get("code") == CODE, f"{what}: code {f.get('code')!r}"
    assert f.get("where") == where, f"{what}: where {f.get('where')!r}, expected {where!r}"
    assert "radian" in f.get("hint", "").lower(), f"{what}: the hint does not say radians: {f.get('hint')!r}"
    assert f.get("error"), f"{what}: no message"
    a0, a1 = angles
    if abs(a1 - a0) <= 360:
        have = _floats(f["hint"])
        for deg in (a0, a1):
            want = math.radians(deg)
            assert any(abs(h - want) < 2e-3 for h in have), (
                f"{what}: the hint does not suggest {want:.4f} rad for {deg}: {f['hint']!r}")


# ------------------------------------------------------------------------------------------------ validate
@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("angles", REFUSED)
def test_axmp_validate_refuses(tmp_path, kind, angles):
    d = DOCUMENTS[kind](angles)
    p = S.write(tmp_path, d)
    r = cli("validate", p)
    _no_abort(r, "axmp validate")
    out = json.loads(r.stdout)
    if kind == "em":
        assert r.returncode == 1, r.stdout
        f = out
    else:  # a model document's validation lists the findings; the exit status says whether any is an error
        assert r.returncode == 1, r.stdout
        f = out.get("errors", [out])[0] if "errors" in out else out
    check_refusal(f, where=bad_pointer(d, kind), angles=angles, what=f"axmp validate ({kind})")


@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("angles", ACCEPTED)
def test_axmp_validate_accepts(tmp_path, kind, angles):
    p = S.write(tmp_path, DOCUMENTS[kind](angles))
    r = cli("validate", p)
    _no_abort(r, "axmp validate")
    assert r.returncode == 0, f"{r.stdout[-600:]}\n{r.stderr[-300:]}"


@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("angles", REFUSED)
def test_python_design_validate_refuses(kind, angles):
    d = DOCUMENTS[kind](angles)
    code = (
        "import json, sys, axionmultiphysics as axmp\n"
        "d = axmp.Design(json.loads(sys.argv[1]))\n"
        "try:\n"
        "    d.validate()\n"
        "except axmp.AxionMultiPhysicsError as e:\n"
        "    print(json.dumps({'error': e.args[0], 'where': e.where, 'hint': e.hint, 'code': e.code}))\n"
        "else:\n"
        "    print(json.dumps({'raised': False}))\n")
    r = python(code, json.dumps(d))
    _no_abort(r, "Design.validate")
    assert r.returncode == 0, r.stderr[-600:]
    f = json.loads(r.stdout)
    assert f != {"raised": False}, "Design.validate accepted the document"
    check_refusal(f, where=bad_pointer(d, kind), angles=angles, what=f"Design.validate ({kind})")


@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("angles", ACCEPTED)
def test_python_design_validate_accepts(kind, angles):
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "axmp.Design(json.loads(sys.argv[1])).validate(); print('ok')\n")
    r = python(code, json.dumps(DOCUMENTS[kind](angles)))
    _no_abort(r, "Design.validate")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-600:]


def test_model_validate_report_lists_every_bad_arc():
    """The report of a model document (all_errors=True) has one finding per refused arc, each at its own pointer,
    and the rest of the document is not blamed."""
    d = model_document([-60, 60])
    d["curves"].append({"name": "second", "path": {"arc": {"center": [0.02, 0.05], "radius": 0.01, "angles": [0, 7]}}})
    d["curves"].append({"name": "fine", "path": {"arc": {"center": [0.02, 0.05], "radius": 0.01, "angles": [0, 1]}}})
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "r = axmp.Design(json.loads(sys.argv[1])).validate(all_errors=True)\n"
            "print(json.dumps(r['errors']))\n")
    r = python(code, json.dumps(d))
    _no_abort(r, "Design.validate(all_errors=True)")
    assert r.returncode == 0, r.stderr[-600:]
    errors = json.loads(r.stdout)
    arc = [e for e in errors if e.get("code") == CODE]
    names = [c["name"] for c in d["curves"]]
    assert sorted(e["where"] for e in arc) == sorted(
        f"/curves/{names.index(n)}/path/arc/angles" for n in ("cut", "second")), errors
    assert len(errors) == len(arc), f"another finding appeared: {errors}"


def test_angles_from_parameters_are_checked_after_evaluation():
    """The angles are expressions: the span is that of the evaluated values."""
    d = model_document(["A0", "A1"], parameters={"A0": -60, "A1": "30*2"})
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "r = axmp.Design(json.loads(sys.argv[1])).validate(all_errors=True)\n"
            "print(json.dumps(r['errors']))\n")
    r = python(code, json.dumps(d))
    _no_abort(r, "expression angles")
    errors = json.loads(r.stdout)
    assert [e["code"] for e in errors] == [CODE], errors
    ok = model_document(["A0", "A1"], parameters={"A0": 0.5, "A1": "2*pi - 0.5"})
    r = python(code, json.dumps(ok))
    assert json.loads(r.stdout) == [], r.stdout


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_angle_literals_are_refused(tmp_path, literal):
    """A document cannot carry a non-finite angle today (the JSON reader refuses these literals, and an expression
    that evaluates to a non-finite value is refused where it is evaluated). Guard: that stays so."""
    text = json.dumps(em_design([0, 1])).replace("[0, 1]", f"[0, {literal}]")
    assert literal in text
    p = tmp_path / "nf.json"
    p.write_text(text, "utf-8")
    r = cli("validate", p)
    _no_abort(r, "axmp validate")
    assert r.returncode != 0 and json.loads(r.stdout).get("ok") is not True, r.stdout


@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("expression", ["1/0", "log(0)", "0/0"])
def test_non_finite_angle_expressions_are_refused(kind, expression):
    d = DOCUMENTS[kind]([0, 1])
    if kind == "em":
        d["parameters"] = {"A": expression}
        d["curves"][0]["path"]["arc"]["angles"] = [0, "A"]
    else:
        d["parameters"]["A"] = expression
        for c in d["curves"]:
            if c["name"] == "cut":
                c["path"]["arc"]["angles"] = [0, "A"]
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "try:\n"
            "    r = axmp.Design(json.loads(sys.argv[1])).validate(all_errors=True)\n"
            "    print(json.dumps({'refused': bool(r.get('errors'))}))\n"
            "except axmp.AxionMultiPhysicsError:\n"
            "    print(json.dumps({'refused': True}))\n")
    r = python(code, json.dumps(d))
    _no_abort(r, "validate with a non-finite angle")
    assert json.loads(r.stdout) == {"refused": True}


# ------------------------------------------------------------------------------------------------ mesh
@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("angles", [[-60, 60], [0, 360], [0, TWO_PI + 0.01]])
def test_axmp_mesh_refuses_and_does_not_abort(tmp_path, mesher, kind, angles):
    d = ON_AXIS[kind](angles)
    p = S.write(tmp_path, d)
    out = tmp_path / "out"
    r = cli("mesh", p, "-o", out)
    f = _error_json(r, f"axmp mesh ({kind})")
    check_refusal(f, where=bad_pointer(d, kind), angles=angles, what=f"axmp mesh ({kind})")
    assert not (out / "mesh.msh").exists(), "a mesh was written for a refused document"


@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("angles", [[0, 1.2], [-1.2, 1.2]])
def test_axmp_mesh_still_meshes_a_valid_arc(tmp_path, mesher, kind, angles):
    p = S.write(tmp_path, ON_AXIS[kind](angles))
    out = tmp_path / "out"
    r = cli("mesh", p, "-o", out)
    _no_abort(r, f"axmp mesh ({kind})")
    assert r.returncode == 0, f"{r.stdout[-600:]}\n{r.stderr[-400:]}"
    assert (out / "mesh.msh").is_file()


@pytest.mark.parametrize("angles", [[-60, 60], [0, 360]])
def test_python_mesh_of_a_model_document_refuses(tmp_path, mesher, angles):
    """``axmp.mesh`` of a model document raises the refusal; the process does not abort."""
    d = ON_AXIS["model"](angles)
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "try:\n"
            "    axmp.mesh(axmp.Design(json.loads(sys.argv[1])), workdir=sys.argv[2])\n"
            "except axmp.AxionMultiPhysicsError as e:\n"
            "    print(json.dumps({'error': e.args[0], 'where': e.where, 'hint': e.hint, 'code': e.code}))\n"
            "else:\n"
            "    print(json.dumps({'meshed': True}))\n")
    env = S.environment(AXMP_MESHER=mesher)
    r = subprocess.run([sys.executable, "-c", code, json.dumps(d), str(tmp_path / "w")], capture_output=True,
                       text=True, timeout=300, env={**_env(), "AXMP_MESHER": env["AXMP_MESHER"]})
    _no_abort(r, "axmp.mesh")
    f = json.loads(r.stdout)
    assert "meshed" not in f, "axmp.mesh meshed a document with an arc of more than a turn"
    check_refusal(f, where=bad_pointer(d, "model"), angles=angles, what="axmp.mesh")


@pytest.mark.parametrize("angles", [[-60, 60], [0, 360]])
def test_python_design_solve_of_an_em_design_refuses(tmp_path, mesher, angles):
    """The command line's ``mesh`` of an electromagnetic design is the entry point; through Python the design is
    meshed by ``Design.solve``, which meshes first: that must refuse, not abort."""
    d = ON_AXIS["em"](angles)
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "d = axmp.Design(json.loads(sys.argv[1]))\n"
            "try:\n"
            "    d.solve(sys.argv[2], verbose=0)\n"
            "except axmp.AxionMultiPhysicsError as e:\n"
            "    print(json.dumps({'error': e.args[0], 'where': e.where, 'hint': e.hint, 'code': e.code}))\n"
            "else:\n"
            "    print(json.dumps({'meshed': True}))\n")
    r = subprocess.run([sys.executable, "-c", code, json.dumps(d), str(tmp_path / "w")], capture_output=True,
                       text=True, timeout=300, env={**_env(), "AXMP_MESHER": mesher})
    _no_abort(r, "Design.solve")
    f = json.loads(r.stdout)
    assert "meshed" not in f
    check_refusal(f, where=bad_pointer(d, "em"), angles=angles, what="Design.solve")


# ------------------------------------------------------------------------------------------------ the axis
# In an axisymmetric design an arc path is refused when any of its points has r < -1e-12 (the tolerance of the
# check on a point's r, in metres): code geometry.arc_negative_r at "<path>/arc", hint: the arc crosses the axis,
# split it at the axis or move the centre. The arc's smallest r is exact: c - R when pi (mod 2 pi) lies in the span
# (the extreme of the radial direction), else the smaller of the two end points. A span above 2 pi (or not finite)
# is the other code and is reported first. A planar design has no axis: negative x is allowed.
NEG = "geometry.arc_negative_r"
HALF = math.pi
# (id, centre r, radius, angles): refused
NEGATIVE_R = [
    pytest.param(0.0, 0.75, [0, TWO_PI - 0.5], id="through_the_axis_ends_positive"),
    pytest.param(0.595, 0.6, [HALF - 0.15, HALF + 0.15], id="extreme_at_pi_ends_positive"),
    pytest.param(0.595, 0.6, [3 * HALF - 0.15, 3 * HALF + 0.15], id="extreme_at_3pi_ends_positive"),
    pytest.param(0.75 - 1e-3, 0.75, [HALF / 2, 3 * HALF / 2], id="dips_1e-3"),
    pytest.param(0.75 - 1e-9, 0.75, [HALF / 2, 3 * HALF / 2], id="dips_1e-9"),
    pytest.param(0.1, 0.5, [1.5, 2.5], id="end_point_negative"),
    pytest.param(-1.0, 0.5, [0, 1], id="entirely_negative"),
]
NONNEGATIVE_R = [
    pytest.param(0.75, 0.75, [HALF / 2, 3 * HALF / 2], id="tangent_to_the_axis"),
    pytest.param(0.75, 0.75, [0, TWO_PI], id="tangent_full_circle"),
    pytest.param(0.75, 0.75, [HALF, 3 * HALF], id="tangent_from_pi_to_3pi"),
    pytest.param(0.75 - 1e-14, 0.75, [HALF / 2, 3 * HALF / 2], id="dips_1e-14_within_tolerance"),
    pytest.param(0.0, 0.75, [-HALF / 2, HALF / 2], id="right_half_plane"),
    pytest.param(0.0, 0.75, [0, HALF / 2], id="quarter_from_the_axis"),
]


def on_axis_document(kind: str, cr: float, radius: float, angles):
    if kind == "em":
        return em_design(angles, center=(cr, 0.0), radius=radius)
    return model_document(angles, center=(cr, "H/2"), radius=radius)


def check_negative_r(f: dict, *, kind: str, d: dict, what: str) -> None:
    index = [c["name"] for c in d["curves"]].index("sheet" if kind == "em" else "cut")
    assert f.get("code") == NEG, f"{what}: code {f.get('code')!r}: {f}"
    assert f.get("where") == f"/curves/{index}/path/arc", f"{what}: where {f.get('where')!r}"
    hint = f.get("hint", "").lower()
    assert "axis" in hint and "split" in hint and ("centre" in hint or "center" in hint), f"{what}: hint {hint!r}"
    assert f.get("error"), f"{what}: no message"


@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("cr, radius, angles", NEGATIVE_R)
def test_arc_below_the_axis_is_refused_by_validate(tmp_path, kind, cr, radius, angles):
    d = on_axis_document(kind, cr, radius, angles)
    r = cli("validate", S.write(tmp_path, d))
    _no_abort(r, "axmp validate")
    assert r.returncode == 1, r.stdout
    out = json.loads(r.stdout)
    f = out["errors"][0] if "errors" in out else out
    check_negative_r(f, kind=kind, d=d, what=f"axmp validate ({kind})")
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "try:\n"
            "    axmp.Design(json.loads(sys.argv[1])).validate()\n"
            "except axmp.AxionMultiPhysicsError as e:\n"
            "    print(json.dumps({'error': e.args[0], 'where': e.where, 'hint': e.hint, 'code': e.code}))\n"
            "else:\n"
            "    print(json.dumps({'raised': False}))\n")
    r = python(code, json.dumps(d))
    _no_abort(r, "Design.validate")
    g = json.loads(r.stdout)
    assert g != {"raised": False}, "Design.validate accepted an arc below the axis"
    check_negative_r(g, kind=kind, d=d, what=f"Design.validate ({kind})")


@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("cr, radius, angles", NONNEGATIVE_R)
def test_arc_on_or_above_the_axis_is_accepted_by_validate(tmp_path, kind, cr, radius, angles):
    d = on_axis_document(kind, cr, radius, angles)
    r = cli("validate", S.write(tmp_path, d))
    _no_abort(r, "axmp validate")
    assert r.returncode == 0, f"{r.stdout[-600:]}\n{r.stderr[-300:]}"


def test_span_is_reported_before_the_axis():
    """[-60, 60] about an axis-centred arc breaks both rules: one finding for the arc, the span code."""
    d = em_design([-60, 60], center=(0.0, 0.0))
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "try:\n"
            "    axmp.Design(json.loads(sys.argv[1])).validate()\n"
            "except axmp.AxionMultiPhysicsError as e:\n"
            "    print(e.code, e.where)\n")
    r = python(code, json.dumps(d))
    _no_abort(r, "validate")
    assert r.stdout.split() == [CODE, "/curves/0/path/arc/angles"], r.stdout


def test_model_report_has_one_finding_per_bad_arc_of_either_kind():
    d = model_document([0, TWO_PI - 0.5], center=(0, "H/2"))
    d["curves"].append({"name": "wide", "path": {"arc": {"center": [0.02, 0.05], "radius": 0.01, "angles": [0, 7]}}})
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "r = axmp.Design(json.loads(sys.argv[1])).validate(all_errors=True)\n"
            "print(json.dumps([[e.get('code'), e['where']] for e in r['errors']]))\n")
    r = python(code, json.dumps(d))
    _no_abort(r, "validate(all_errors=True)")
    names = [c["name"] for c in d["curves"]]
    assert sorted(json.loads(r.stdout)) == sorted([
        [NEG, f"/curves/{names.index('cut')}/path/arc"],
        [CODE, f"/curves/{names.index('wide')}/path/arc/angles"]]), r.stdout


def planar_model(angles) -> dict:
    d = S.doc("curves_planar")
    d["materials"] = {"m": {"density": 1.0, "thermal": {"k_W_mK": 1.0, "cp_J_kgK": 1.0}}}
    for r in d["regions"]:
        r["material"] = "m"
    for c in d["curves"]:
        if c["name"] == "cut":
            c["path"]["arc"].update({"center": [-0.05, "H/2"], "angles": angles})
    return d


@pytest.mark.parametrize("angles", [[-1.0, 5.0], [0, 4.0], [0.5, 6.2]])
def test_planar_model_allows_negative_x(tmp_path, angles):
    """No axis in a planar design: an arc centred at negative x validates, as long as its span is within a turn. (A
    planar electromagnetic design needs an imported mesh, so only the model document is reachable here.)"""
    r = cli("validate", S.write(tmp_path, planar_model(angles)))
    _no_abort(r, "axmp validate (planar model)")
    assert r.returncode == 0, f"{r.stdout[-500:]}"


def test_planar_model_still_refuses_a_span_of_more_than_a_turn(tmp_path):
    r = cli("validate", S.write(tmp_path, planar_model([-60, 60])))
    _no_abort(r, "axmp validate (planar model)")
    assert r.returncode == 1, r.stdout
    out = json.loads(r.stdout)
    assert (out["errors"][0] if "errors" in out else out).get("code") == CODE, r.stdout


def test_axmp_mesh_of_an_arc_through_the_axis_refuses_and_does_not_abort(tmp_path, mesher):
    """The arc about a centre at r = 0 with angles [0, 2 pi - 0.5] (span 5.78 rad, within a turn) aborts the mesher
    of an electromagnetic design; the model document's mesher stops it ("lies outside every enabled region")."""
    for kind in ("em", "model"):
        d = on_axis_document(kind, 0.0, 0.75 if kind == "em" else "R/2", [0, TWO_PI - 0.5])
        out = tmp_path / kind
        r = cli("mesh", S.write(tmp_path, d, f"{kind}.json"), "-o", out)
        f = _error_json(r, f"axmp mesh ({kind})")
        check_negative_r(f, kind=kind, d=d, what=f"axmp mesh ({kind})")
        assert not (out / "mesh.msh").exists()


def test_design_solve_of_an_arc_through_the_axis_refuses(tmp_path, mesher):
    d = em_design([0, TWO_PI - 0.5], center=(0.0, 0.0))
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "try:\n"
            "    axmp.Design(json.loads(sys.argv[1])).solve(sys.argv[2], verbose=0)\n"
            "except axmp.AxionMultiPhysicsError as e:\n"
            "    print(json.dumps({'error': e.args[0], 'where': e.where, 'hint': e.hint, 'code': e.code}))\n"
            "else:\n"
            "    print(json.dumps({'solved': True}))\n")
    r = subprocess.run([sys.executable, "-c", code, json.dumps(d), str(tmp_path / "w")], capture_output=True,
                       text=True, timeout=300, env={**_env(), "AXMP_MESHER": mesher})
    _no_abort(r, "Design.solve")
    check_negative_r(json.loads(r.stdout), kind="em", d=d, what="Design.solve")


# ------------------------------------------------------------------------------------------------ rounding, degenerate arcs
# The bound on the span allows rounding: a full turn written [a, a + 2 pi] is accepted for every start angle a (in
# double precision (a + 2 pi) - a is sometimes one unit in the last place above 2 pi). Only a span above
# 2 pi (1 + 1e-12) is refused.
DEGENERATE = "geometry.arc_degenerate"
SHORT = 1e-7  # an arc shorter than this (radius * |a1 - a0|, in metres) is refused, as the contour's ellipse arc is


def _validate_all(documents: list[dict]) -> list:
    """Design.validate (all_errors for a model document) of each document in one process: the list of the codes of
    the findings (None for no finding)."""
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "out = []\n"
            "for d in json.load(open(sys.argv[1])):\n"
            "    d = axmp.Design(d)\n"
            "    try:\n"
            "        r = d.validate(all_errors=True)\n"
            "        out.append([e.get('code') for e in r.get('errors', [])] or None)\n"
            "    except axmp.AxionMultiPhysicsError as e:\n"
            "        out.append([e.code])\n"
            "json.dump(out, sys.stdout)\n")
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "docs.json"
        f.write_text(json.dumps(documents), "utf-8")
        r = python(code, f, timeout=600)
    _no_abort(r, "validate of a list")
    assert r.returncode == 0, r.stderr[-600:]
    return json.loads(r.stdout)


def test_full_turn_sweep_of_start_angles_is_accepted():
    """[a, a + 2 pi] for 2001 start angles a in [-10, 10] (133 of them evaluate to a span one ulp above 2 pi)."""
    starts = [round(-10 + 0.01 * i, 2) for i in range(2001)] + [1.72, 1.8, 2.05]
    docs = [em_design([a, a + TWO_PI]) for a in starts]
    out = _validate_all(docs)
    bad = [a for a, o in zip(starts, out) if o is not None]
    assert not bad, f"full turns refused for the start angles {bad[:10]} ({len(bad)} in all)"


@pytest.mark.parametrize("kind", ["em", "model"])
def test_full_turn_from_1_72_is_accepted(tmp_path, kind):
    a = 1.72
    d = DOCUMENTS[kind]([a, a + TWO_PI])
    r = cli("validate", S.write(tmp_path, d))
    _no_abort(r, "axmp validate")
    assert r.returncode == 0, r.stdout[-600:]


def test_full_turn_with_expression_angles_is_accepted():
    """The natural parametric circle: angles ["t0", "t0 + 2*pi"] with t0 = 1.72 (and the other starts of the sweep)."""
    starts = [1.72, 1.8, 2.05, 0.0, -1.3, 4.1]
    docs = [model_document(["t0", "t0 + 2*pi"], parameters={"t0": a}) for a in starts]
    docs += [em_design(["t0", "t0 + 2*pi"]) | {"parameters": {"t0": a}} for a in starts]
    out = _validate_all(docs)
    assert out == [None] * len(docs), out


def test_span_slightly_above_a_turn_is_still_refused():
    """The slack is rounding only: 1e-6 rad above a turn is refused, 1e-13 relative above is accepted."""
    docs = [em_design([0, TWO_PI + 1e-6]), em_design([0, TWO_PI * (1 + 1e-13)])]
    assert _validate_all(docs) == [[CODE], None]


# the hint's suggested radians, pasted back, are accepted
@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("degrees", [[0, 360], [-180, 180], [90, 450], [-60, 60]])
def test_the_hint_suggestion_is_accepted_when_pasted_back(kind, degrees):
    d = DOCUMENTS[kind](degrees)
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "try:\n"
            "    axmp.Design(json.loads(sys.argv[1])).validate()\n"
            "except axmp.AxionMultiPhysicsError as e:\n"
            "    print(json.dumps({'code': e.code, 'hint': e.hint}))\n")
    r = python(code, json.dumps(d))
    _no_abort(r, "validate")
    f = json.loads(r.stdout)
    assert f["code"] == CODE, f
    pairs = re.findall(r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]", f["hint"])
    assert pairs, f"no [a0, a1] pair in the hint: {f['hint']!r}"
    pasted = [float(pairs[-1][0]), float(pairs[-1][1])]
    assert pasted == pytest.approx([math.radians(degrees[0]), math.radians(degrees[1])], abs=1e-9)
    out = _validate_all([DOCUMENTS[kind](pasted)])
    assert out == [None], f"the suggestion {pasted} (from {f['hint']!r}) is refused: {out}"


# a degenerate arc: radius <= 0 and length <= 1e-7 m
@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("radius", [0, -0.1])
def test_radius_not_positive_is_refused_as_degenerate(tmp_path, kind, radius):
    """Radius 0 or below: the reason is the radius (not the axis, not the span), at ``<path>/arc/radius``."""
    d = DOCUMENTS[kind]([-0.5, 0.5])
    arc = [c for c in d["curves"] if c["name"] == ("sheet" if kind == "em" else "cut")][0]["path"]["arc"]
    arc["radius"] = radius
    r = cli("validate", S.write(tmp_path, d))
    _no_abort(r, "axmp validate")
    assert r.returncode == 1, r.stdout
    out = json.loads(r.stdout)
    f = out["errors"][0] if "errors" in out else out
    idx = [c["name"] for c in d["curves"]].index("sheet" if kind == "em" else "cut")
    assert f.get("code") == DEGENERATE, f
    assert f.get("where") == f"/curves/{idx}/path/arc/radius", f
    assert "radius" in (f.get("error", "") + f.get("hint", "")).lower()


def test_negative_radius_about_an_axis_centre_is_not_blamed_on_the_axis():
    """A radius of -R about r = R/2 would 'reach' r < 0 by the formula: the finding is the radius."""
    d = model_document([-0.5, 0.5], center=("R/2", "H/2"), radius="-R")
    assert _validate_all([d]) == [[DEGENERATE]]


@pytest.mark.parametrize("kind", ["em", "model"])
@pytest.mark.parametrize("length", [1e-10, 1e-8, 5e-8, 1e-7])
def test_arc_shorter_than_the_kernel_tolerance_is_refused(tmp_path, kind, length):
    """radius * span <= 1e-7 m (the contour's limit for its ellipse arc), at ``<path>/arc/angles``."""
    radius = 0.1
    d = DOCUMENTS[kind]([0.0, length / radius])
    arc = [c for c in d["curves"] if c["name"] == ("sheet" if kind == "em" else "cut")][0]["path"]["arc"]
    arc["radius"] = radius if kind == "em" else 0.1
    r = cli("validate", S.write(tmp_path, d))
    _no_abort(r, "axmp validate")
    assert r.returncode == 1, r.stdout
    out = json.loads(r.stdout)
    f = out["errors"][0] if "errors" in out else out
    idx = [c["name"] for c in d["curves"]].index("sheet" if kind == "em" else "cut")
    assert f.get("code") == DEGENERATE, f
    assert f.get("where") == f"/curves/{idx}/path/arc/angles", f
    assert re.search(r"1e-0?7", f.get("error", "") + f.get("hint", "")), f"the limit is not named: {f}"


@pytest.mark.parametrize("kind", ["em", "model"])
def test_arc_just_above_the_kernel_tolerance_is_accepted(tmp_path, kind):
    d = DOCUMENTS[kind]([0.0, 2e-6])
    arc = [c for c in d["curves"] if c["name"] == ("sheet" if kind == "em" else "cut")][0]["path"]["arc"]
    arc["radius"] = 0.1  # length 2e-7 m
    r = cli("validate", S.write(tmp_path, d))
    _no_abort(r, "axmp validate")
    assert r.returncode == 0, r.stdout[-500:]


@pytest.mark.parametrize("span", [1e-9, 1e-11])
def test_axmp_mesh_of_a_tiny_arc_refuses_and_does_not_abort(tmp_path, mesher, span):
    """An arc of 1e-10 m passed validation and aborted ``axmp mesh``/``Design.solve`` (a boundary element that is no
    edge of the mesh: a write at index -1). Validation refuses it now: no run of the mesher."""
    d = em_design([0.0, span], center=(0.3, 0.0), radius=0.1)
    out = tmp_path / "out"
    r = cli("mesh", S.write(tmp_path, d), "-o", out)
    f = _error_json(r, "axmp mesh")
    assert f.get("code") == DEGENERATE and f.get("where") == "/curves/0/path/arc/angles", f
    assert not (out / "mesh.msh").exists()
    code = ("import json, sys, axionmultiphysics as axmp\n"
            "try:\n"
            "    axmp.Design(json.loads(sys.argv[1])).solve(sys.argv[2], verbose=0)\n"
            "except axmp.AxionMultiPhysicsError as e:\n"
            "    print(json.dumps({'code': e.code, 'where': e.where}))\n"
            "else:\n"
            "    print(json.dumps({'solved': True}))\n")
    r = subprocess.run([sys.executable, "-c", code, json.dumps(d), str(tmp_path / "w")], capture_output=True,
                       text=True, timeout=300, env={**_env(), "AXMP_MESHER": mesher})
    _no_abort(r, "Design.solve")
    assert json.loads(r.stdout) == {"code": DEGENERATE, "where": "/curves/0/path/arc/angles"}


# the axis rule and the cross-section designs ("xsec": (x, y), no axis)
def xsec_arc_design(angles, center=(-0.01, 0.0), radius=0.008) -> dict:
    from axionmultiphysics import cavity_xsec as cx
    d = cx.xsec_design([{"name": "cav", "shape": {"disk": {"center": [0, 0], "radius": 0.025}}, "material": "vacuum"}],
                       curves=[{"name": "sheet", "material": "superconductor",
                                "path": {"arc": {"center": list(center), "radius": radius, "angles": angles}}}],
                       allow_lossless_walls=True)
    return d.doc


@pytest.mark.parametrize("angles", [[0, 1.5], [-1.0, 1.0], [0, TWO_PI], [2.0, 5.0]])
@pytest.mark.parametrize("center", [(-0.01, 0.0), (0.0, 0.0), (-0.02, 0.01)])
def test_xsec_design_allows_an_arc_at_negative_x(tmp_path, center, angles):
    d = xsec_arc_design(angles, center)
    assert d["symmetry"] == "xsec"
    r = cli("validate", S.write(tmp_path, d))
    _no_abort(r, "axmp validate (xsec)")
    assert r.returncode == 0, r.stdout[-600:]


def test_xsec_design_still_refuses_a_span_of_more_than_a_turn(tmp_path):
    r = cli("validate", S.write(tmp_path, xsec_arc_design([-60, 60])))
    _no_abort(r, "axmp validate (xsec)")
    assert r.returncode == 1 and json.loads(r.stdout).get("code") == CODE, r.stdout


# ------------------------------------------------------------------------------------------------ other places
def test_no_other_arc_reader():
    """Every place of the sources that reads the angles of an arc path does it through the one parser, so one check
    covers every entry point: the key ``angles`` of an arc is read in exactly one function."""
    hits = []
    for f in sorted((ROOT / "cpp" / "src").glob("*.cpp")):
        text = f.read_text("utf-8")
        for m in re.finditer(r'at\("angles"\)\.range\(\)', text):
            line = text[:m.start()].count("\n") + 1
            hits.append((f.name, line))
    # the arc path of spec.cpp, and the revolve solid of the same file (degrees, with its own 360-degree check)
    assert {h[0] for h in hits} == {"spec.cpp"}, hits
    assert len(hits) == 2, hits
