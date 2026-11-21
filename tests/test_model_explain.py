"""``axmp explain`` and ``Design.explain()``: the resolved model of a model document, as text and as JSON.

Independent reference: the listing of section 4.1 of the interface design note (revision 5), adapted by hand for the four
documents of ``cpp/tests/data/model/`` and committed beside them as ``cpp/tests/data/model/explain/<name>.txt`` and
``<name>.json``. Nothing here is generated from the implementation. What the listing keeps of section 4.1: the header, one
labelled line per section (label column of 10 characters, two spaces, content), sub-lines indented 16 characters, the
tags ``[written]``, ``[default]``, ``[implicit]`` at one column. What the adaptation changes, and why:

* the boundary table prints the selector and ``deferred: resolved by `axmp boundaries```, not entity counts and lengths
  (the selection needs the mesher); it prints no milestone name;
* the options are one per line (name, value, tag), so that the tag column is the same everywhere;
* name-keyed sections (parameters, materials, boundaries, physics, conditions, data keys) are printed in sorted key
  order, because the parsed model does not keep the order of a JSON object's keys; ``outputs`` follow the model's own
  evaluation order (an expression output after the outputs it reads), ``regions``, ``couplings`` and ``study`` follow the
  document;
* numbers print with ``%g``; written expressions stay as written; values of options are typed by the registry;
* the route line names the registry row and says ``(not implemented yet)``; no route exists for the busbar, and the line
  says so (``no route in this version``).

The full text format is the specification the implementer was handed; see the module constants below for every rule that
a golden file does not exercise.

This file needs the compiled core (``_core``) with the ``explain`` binding and the ``axmp`` executable of the same build
(``AXMP_BIN``, ``AXMP_EXE`` or a build directory), as ``tests/test_physics_catalogue.py`` does.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

import axionmultiphysics as axmp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cpp" / "tests" / "data" / "model"
GOLD = DATA / "explain"
G0 = ROOT / "tests" / "g0"

_spec = importlib.util.spec_from_file_location("g0_harness_explain", G0 / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)

NAMES = ["channel", "heated_cavity", "plate_cell", "busbar"]
MILESTONE = re.compile(r"\bU(1[0-2]|[1-9])[a-d]?\b")
EXPLAIN_SCHEMA = "axionmultiphysics.explain/1"


def _axmp() -> str:
    exe = g0.find_axmp()
    if exe is None:
        pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")
    return exe


def _run(*args):
    return subprocess.run([_axmp(), *args], capture_output=True, text=True, timeout=120)


def _doc(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def _write(tmp_path: Path, doc: dict, name: str = "doc.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


# ----------------------------------------------------------------- the golden files
@pytest.mark.parametrize("name", NAMES)
def test_explain_matches_the_golden_file(name, tmp_path):
    """The text of ``axmp explain`` is the golden file byte for byte; ``axmp explain --json`` parses to the golden JSON;
    ``Design.explain()`` and ``Design.explain(json=True)`` give the same two. Exit status 0 even though no route of this
    version is implemented (an ``unavailable`` finding does not stop an explanation)."""
    gold_txt = (GOLD / f"{name}.txt").read_text()
    gold_json = json.loads((GOLD / f"{name}.json").read_text())
    path = str(DATA / f"{name}.json")

    r = _run("explain", path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout == gold_txt

    r = _run("explain", path, "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout) == gold_json

    d = axmp.Design(_doc(name))
    assert d.explain() == gold_txt
    assert d.explain(json=True) == gold_json


@pytest.mark.parametrize("name", NAMES)
def test_golden_json_carries_its_schema_string_and_names_no_milestone(name):
    gold_json = json.loads((GOLD / f"{name}.json").read_text())
    assert gold_json["schema"] == EXPLAIN_SCHEMA
    assert gold_json["status"] == "beta"
    assert not MILESTONE.search((GOLD / f"{name}.txt").read_text())
    assert not MILESTONE.search(json.dumps(gold_json))


def test_the_json_flag_may_come_before_the_file():
    """``axmp explain [--json] design.json``: the flag is accepted in either position."""
    path = str(DATA / "channel.json")
    a = _run("explain", path, "--json")
    b = _run("explain", "--json", path)
    assert a.returncode == 0 and b.returncode == 0
    assert json.loads(a.stdout) == json.loads(b.stdout)


def test_explain_json_is_the_core_text_of_the_same_function():
    """``_core.explain(doc_text, as_json)`` returns the text, or the JSON text; ``Design.explain`` wraps it."""
    from axionmultiphysics import _core

    text = json.dumps(_doc("channel"))
    assert _core.explain(text, False) == (GOLD / "channel.txt").read_text()
    assert json.loads(_core.explain(text, True)) == json.loads((GOLD / "channel.json").read_text())


# ----------------------------------------------------------------- what the listing shows, rule by rule
def test_every_default_is_explicit_and_tagged():
    """Section 4.1: 'every default explicit'. The options a document does not write are listed with the registry's
    default and the tag [default]; the written ones carry [written]."""
    txt = axmp.Design(_doc("channel")).explain()
    lines = {ln.split()[0]: ln for ln in txt.splitlines() if ln.startswith(" " * 16) and not "->" in ln}
    assert lines["convection"].rstrip().endswith("[default]") and "standard" in lines["convection"]
    assert lines["order"].rstrip().endswith("[default]")
    assert lines["equation"].rstrip().endswith("[written]")
    assert lines["viscous_form"].rstrip().endswith("[written]")
    # one tag column for all of them
    cols = {ln.rindex("[") for ln in lines.values()}
    assert len(cols) == 1


def test_the_engine_form_of_the_outlet_follows_the_viscous_form(tmp_path):
    """Section 4.1: with the Laplacian form the outlet is 'do_nothing: mu du/dn - p n = 0'; with the stress form
    (the default) it is 2 mu eps(u) n - p n = 0 (section 4.1's own explanation of the two forms)."""
    d = axmp.Design(_doc("channel"))
    assert "-> do_nothing: mu du/dn - p n = 0" in d.explain()
    d.patch([{"op": "remove", "path": "/physics/flow/options/viscous_form"}])
    txt = d.explain()
    assert "-> do_nothing: 2 mu eps(u) n - p n = 0" in txt
    assert re.search(r"viscous_form stress +\[default\]", txt)


def test_the_selector_text_and_the_deferred_resolution():
    txt = axmp.Design(_doc("channel")).explain()
    assert "walls  of channel except inlet, outlet  used by flow  deferred: resolved by `axmp boundaries`" in txt
    assert "outlet  of channel where x = 0.04  used by flow" in txt     # the expression 'L' evaluated, printed with %g
    j = axmp.Design(_doc("channel")).explain(json=True)
    assert j["boundaries"]["outlet"]["selector"] == {"of": "channel", "where": {"x": "L"}}     # as written
    assert j["boundaries"]["outlet"]["resolution"] == "deferred"


def test_an_unavailable_route_is_stated_not_hidden():
    j = axmp.Design(_doc("busbar")).explain(json=True)
    assert j["routes"] == [{"step": "steady", "physics": ["current", "heat", "solid"], "route": None, "implemented": False}]
    j = axmp.Design(_doc("channel")).explain(json=True)
    assert j["routes"][0]["route"] == "navier_stokes_steady_planar" and j["routes"][0]["implemented"] is False
    assert j["routes"][0]["needs"] == {"mesher": True, "fluids": True, "mpi": True, "engine_3d": False, "max_ranks": 0}


def test_the_listing_is_independent_of_the_order_of_the_keys_of_the_document():
    """The parsed model does not keep the document's key order, so the listing does not depend on it either."""
    doc = _doc("channel")
    rev = json.loads(json.dumps(doc), object_pairs_hook=lambda kv: dict(reversed(kv)))
    assert axmp.Design(rev).explain() == axmp.Design(doc).explain()


def test_explain_does_not_change_the_document():
    doc = _doc("heated_cavity")
    before = copy.deepcopy(doc)
    axmp.Design(doc).explain()
    assert doc == before


# ----------------------------------------------------------------- refusals
def test_explain_refuses_a_document_that_validation_refuses(tmp_path):
    """explain must not accept a document that validation refuses. The first error, as ``axmp validate``
    prints it: the four keys error, where, hint and code, exit status 1, nothing on stderr's account of the model."""
    doc = _doc("channel")
    doc["physics"]["flow"]["boundary_conditions"]["outlet"]["kind"] = "nonexistent"
    r = _run("explain", _write(tmp_path, doc))
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["code"] == "physics.unknown_condition_kind"
    assert out["where"] == "/physics/flow/boundary_conditions/outlet/kind"
    assert "nonexistent" in out["error"]
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        axmp.Design(doc).explain()
    assert e.value.code == "physics.unknown_condition_kind"
    assert e.value.where == "/physics/flow/boundary_conditions/outlet/kind"


def test_explain_of_an_electromagnetic_design_says_what_it_is_for(tmp_path):
    """explain is for model documents. An electromagnetic design gets a refusal (no code), with the command that
    describes it."""
    d = axmp.template("toroid")
    msg = "explain describes a model document (a design with a \"physics\" section); this is an electromagnetic design"
    hint = "Use `axmp resolve` to see an electromagnetic design with its parameters evaluated."
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        d.explain()
    assert msg in str(e.value)
    assert e.value.hint == hint
    r = _run("explain", _write(tmp_path, d.doc))
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["error"] == msg and out["hint"] == hint and "code" not in out


def test_explain_usage_errors_exit_2():
    path = str(DATA / "channel.json")
    assert _run("explain").returncode == 2
    assert _run("explain", path, path).returncode == 2
    assert _run("explain", path, "--bogus").returncode == 2
    assert _run("patch", path).returncode == 2


def test_usage_lists_explain_and_patch():
    """Section 12.2's two rows, in the usage's own style (section 15.2 row 10, declared)."""
    r = _run()
    text = r.stdout + r.stderr
    assert "  axmp explain design.json [--json]" in text
    assert "  axmp patch design.json ops.json [-o OUT]" in text


# ----------------------------------------------------------------- solver settings, needs and other defaults
NS_DEFAULTS = "newton_rtol 1e-12, newton_maxit 30, n_picard 3, line_search true, residual_floor_factor 4"


def _channel_with_heat() -> dict:
    """The channel with a heat instance that the flow drives one way (the advection coupling): the group's row is the
    one-way row; the flow instance takes the settings of its own row, the steady Navier-Stokes row."""
    d = _doc("channel")
    d["materials"]["water"]["thermal"] = {"k_W_mK": 0.6, "cp_J_kgK": 4180}
    d["physics"]["heat"] = {"type": "heat.transfer", "boundary_conditions": {
        "inlet": {"kind": "temperature", "T_K": 300}, "outlet": {"kind": "insulation"},
        "walls": {"kind": "temperature", "T_K": 310}}}
    d["couplings"] = [{"term": "advection", "from": "flow", "to": "heat"}]
    del d["outputs"]["F_walls"]
    return d


def _solver_lines(txt: str) -> list[str]:
    return [ln for ln in txt.splitlines() if re.search(r"\bnewton_rtol\b", ln)]


def test_a_one_way_flow_and_heat_step_lists_the_flow_solver_defaults():
    """The flow instance of a one-way step takes the settings of its own row; the listing states every one of them,
    tagged as defaults, and the heat instance (no settings) has no line. The coupling strategy is the default one."""
    d = axmp.Design(_channel_with_heat())
    lines = _solver_lines(d.explain())
    assert len(lines) == 1
    assert lines[0].startswith(" " * 14 + "flow: " + NS_DEFAULTS) and lines[0].rstrip().endswith("[defaults]")
    study = d.explain(json=True)["study"][0]
    assert study["coupling"]["strategy"] == "one_way" and study["coupling"]["strategy_origin"] == "default"
    assert study["solvers"] == {"flow": {
        "newton_rtol": {"value": 1e-12, "origin": "default"}, "newton_maxit": {"value": 30, "origin": "default"},
        "n_picard": {"value": 3, "origin": "default"}, "line_search": {"value": True, "origin": "default"},
        "residual_floor_factor": {"value": 4, "origin": "default"}}}


def test_written_solver_values_are_listed_as_written_on_a_one_way_step_and_on_a_single_row():
    doc = _channel_with_heat()
    doc["study"] = [{"id": "s", "type": "stationary", "physics": ["flow", "heat"],
                     "solvers": {"flow": {"newton_rtol": 1e-10, "newton_maxit": 50}}}]
    d = axmp.Design(doc)
    lines = _solver_lines(d.explain())
    assert len(lines) == 1
    assert lines[0].startswith(" " * 14 + "flow: newton_rtol 1e-10, newton_maxit 50, n_picard 3, line_search true, residual_floor_factor 4")
    assert lines[0].rstrip().endswith("[defaults; written: newton_rtol, newton_maxit]")
    sol = d.explain(json=True)["study"][0]["solvers"]["flow"]
    assert sol["newton_rtol"] == {"value": 1e-10, "origin": "written"}
    assert sol["newton_maxit"] == {"value": 50, "origin": "written"}
    assert sol["n_picard"] == {"value": 3, "origin": "default"}

    single = _doc("channel")
    single["study"] = [{"id": "s", "type": "stationary", "solvers": {"flow": {"line_search": False}}}]
    d = axmp.Design(single)
    lines = _solver_lines(d.explain())
    assert len(lines) == 1 and "line_search false" in lines[0] and lines[0].rstrip().endswith("[defaults; written: line_search]")
    assert d.explain(json=True)["study"][0]["solvers"]["flow"]["line_search"] == {"value": False, "origin": "written"}


def test_a_three_dimensional_current_distribution_says_what_it_needs():
    """Route current_distribution_3d needs an MPI build and the 3D engine (not the fluids library); the listing says so
    in the text and in the JSON, and the planar row of the same cell does not."""
    doc = _doc("plate_cell")
    doc["symmetry"] = "3d"
    doc["regions"] = [{"name": "bulk", "solid": {"box": {"min": [0, 0, 0], "max": ["gap", "Hc", "Hc"]}}, "material": "bath"}]
    d = axmp.Design(doc)
    needs = [ln for ln in d.explain().splitlines() if ln.startswith("  needs")]
    assert needs == ["  needs       the mesher; an MPI build; the 3D engine; any number of ranks for the solve"]
    route = d.explain(json=True)["routes"][0]
    assert route["route"] == "current_distribution_3d"
    assert route["needs"] == {"mesher": True, "fluids": False, "mpi": True, "engine_3d": True, "max_ranks": 0}
    planar = axmp.Design(_doc("plate_cell")).explain(json=True)["routes"][0]["needs"]
    assert planar["engine_3d"] is False and planar["mpi"] is False


def test_a_default_coupling_strategy_and_a_written_geometry_order_are_tagged_as_they_are():
    hc = _doc("heated_cavity")
    written = axmp.Design(hc).explain()
    assert any(ln.strip() == "coupling monolithic" for ln in written.splitlines())
    assert axmp.Design(hc).explain(json=True)["study"][0]["coupling"]["strategy_origin"] == "written"
    del hc["study"][0]["coupling"]
    d = axmp.Design(hc)
    lines = [ln for ln in d.explain().splitlines() if ln.strip().startswith("coupling monolithic")]
    assert len(lines) == 1 and lines[0].rstrip().endswith("[coupling: default]")
    assert d.explain(json=True)["study"][0]["coupling"] == {"strategy": "monolithic", "strategy_origin": "default"}

    ch = _doc("channel")
    ch["mesh"]["geometry_order"] = 1
    d = axmp.Design(ch)
    mesh = [ln for ln in d.explain().splitlines() if ln.startswith("  mesh")]
    assert len(mesh) == 1 and "geometry order 1" in mesh[0] and "[" not in mesh[0]
    assert d.explain(json=True)["mesh"]["geometry_order_origin"] == "written"


def test_a_geometry_free_document_lists_no_geometry_and_no_line_ends_in_blanks():
    doc = {"name": "r", "physics": {"rx": {"type": "chem.reactor"}},
           "study": [{"id": "run", "type": "transient", "time": {"t_end_s": 10}}]}
    txt = axmp.Design(doc).explain()
    assert txt.splitlines()[0] == "r  (model document; no geometry)"
    assert not any(ln.startswith("  mesh") for ln in txt.splitlines())
    assert all(ln == ln.rstrip() for ln in txt.splitlines())


def test_a_document_on_an_imported_mesh_without_regions_is_not_listed_as_geometry_free():
    """A model document with ``mesh.file`` and no ``regions`` takes its geometry from the file: the header does not say
    "no geometry", and the mesh line and the JSON mesh entry say that the geometry order comes from the mesh file."""
    doc = _doc("channel")
    del doc["regions"]
    doc["mesh"] = {"file": "chan.msh"}
    doc["boundaries"] = {"inlet": {"group": "inlet"}, "outlet": {"group": "outlet"}, "walls": {"group": "walls"}}
    del doc["outputs"]["F_walls"]
    d = axmp.Design(doc)
    lines = d.explain().splitlines()
    assert "no geometry" not in lines[0]
    mesh = [ln for ln in lines if ln.startswith("  mesh")]
    assert len(mesh) == 1 and "geometry order from the mesh file" in mesh[0]
    assert d.explain(json=True)["mesh"]["geometry_order"] == 0


# ----------------------------------------------------------------- the keys of a step that the document writes
def _step_lines(txt: str) -> list[str]:
    """The lines of the study block: from the line labelled ``study`` to the line of the next label."""
    lines = txt.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith("  study "))
    out = [lines[first]]
    for ln in lines[first + 1:]:
        if not ln.startswith(" " * 14):
            break
        out.append(ln)
    return out


def test_a_transient_step_lists_what_it_writes_and_tags_it_written():
    """The model keeps the keys that a step wrote (time, save, initial_from, given), and no default of them, so those
    are listed as written and the ones that were not written are not listed."""
    doc = _doc("channel")
    doc["study"] = [
        {"id": "steady", "type": "stationary"},
        {"id": "spinup", "type": "transient", "initial_from": "steady", "given": {"T": 300.0},
         "time": {"t_end_s": 5, "rtol": 1e-5, "first_step_s": 0.01, "order": 2}, "save": {"times_s": [0, 1, 5]}},
    ]
    d = axmp.Design(doc)
    block = _step_lines(d.explain())
    body = [ln.rstrip() for ln in block]
    assert any(ln.strip().startswith("initial_from steady") and ln.endswith("[written]") for ln in body), block
    assert any(ln.strip().startswith("given T=300") and ln.endswith("[written]") for ln in body), block
    assert any(re.match(r"\s+time t_end_s 5, rtol 1e-05, first_step_s 0.01, order 2\s+\[written\]$", ln) for ln in body), block
    assert any(re.match(r"\s+save times_s 0, 1, 5\s+\[written\]$", ln) for ln in body), block
    steps = d.explain(json=True)["study"]
    assert "time" not in steps[0] and "save" not in steps[0] and "initial_from" not in steps[0]
    assert steps[1]["initial_from"] == "steady"
    assert steps[1]["given"] == {"T": 300.0}
    assert steps[1]["time"] == {"t_end_s": {"value": 5.0, "origin": "written"},
                                "rtol": {"value": 1e-5, "origin": "written"},
                                "first_step_s": {"value": 0.01, "origin": "written"},
                                "order": {"value": 2.0, "origin": "written"}}
    assert type(steps[1]["time"]["order"]["value"]) is int
    assert steps[1]["save"] == {"times_s": [0.0, 1.0, 5.0]}


def test_a_step_that_writes_only_its_end_time_and_a_count_lists_those_alone():
    doc = _doc("channel")
    doc["study"] = [{"id": "run", "type": "transient", "time": {"t_end_s": 2.5}, "save": {"n": 11}}]
    d = axmp.Design(doc)
    block = [ln.rstrip() for ln in _step_lines(d.explain())]
    assert any(re.match(r"\s+time t_end_s 2.5\s+\[written\]$", ln) for ln in block), block
    assert any(re.match(r"\s+save n 11\s+\[written\]$", ln) for ln in block), block
    assert not any(re.match(r"\s+time .*(rtol|first_step_s|order)", ln) for ln in block), block
    step = d.explain(json=True)["study"][0]
    assert step["time"] == {"t_end_s": {"value": 2.5, "origin": "written"}}
    assert step["save"] == {"n": 11}


def test_a_stationary_step_that_writes_nothing_lists_no_such_lines():
    txt = axmp.Design(_doc("channel")).explain()
    assert not any(re.match(r"\s+(time|save|initial_from|given)\b", ln) for ln in txt.splitlines())


# ----------------------------------------------------------------- listing of a document without geometry
def test_a_geometry_free_document_lists_no_geometry_in_its_route_line_and_no_empty_labels():
    doc = {"name": "r", "physics": {"rx": {"type": "chem.reactor"}},
           "study": [{"id": "run", "type": "transient", "time": {"t_end_s": 10}}]}
    d = axmp.Design(doc)
    lines = d.explain().splitlines()
    route = [ln for ln in lines if ln.startswith("  route")]
    assert len(route) == 1 and re.search(r"chem\.reactor / transient -> ", route[0]), route
    assert "axisymmetric" not in route[0] and "planar" not in route[0]
    phys = [ln for ln in lines if ln.startswith("  physics")]
    assert len(phys) == 1 and not re.search(r"\bon\b", phys[0]) and phys[0].rstrip().endswith("[chem.reactor, beta]"), phys
    for label in ("parameters", "regions", "outputs"):
        assert not any(ln.strip() == label for ln in lines), (label, lines)
    # the JSON keeps its keys (an empty object or list)
    j = d.explain(json=True)
    assert j["parameters"] == {} and j["regions"] == [] and j["outputs"] == []


def test_explain_of_a_json_array_says_it_expects_an_object(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text("[1, 2]")
    r = _run("explain", str(p))
    assert r.returncode != 0
    assert "expected an object" in r.stderr + r.stdout
    assert "electromagnetic" not in r.stderr + r.stdout
