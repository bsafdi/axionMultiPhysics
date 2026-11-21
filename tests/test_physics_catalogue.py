"""The physics catalogue: ``axmp physics``, ``axionmultiphysics.physics()`` and the generated reference page.

The catalogue is the registry of physics types, coupling terms and route rows that the model document of the
interface design is validated against (cpp/include/axionmultiphysics/physics_registry.hpp). Its tables, and the
case list behind the matrix test, are asserted against the design note's tables in ``cpp/tests/test_model_matrix.cpp``;
this file asserts what a user can see: the command line, the Python function, the JSON of schema
``axionmultiphysics.physics_catalogue/1``, and the page ``docs/reference/physics_catalogue.md`` that is generated from it.

Independent references: section 8.5 of the interface design note (revision 5) transcribed below as
``SECTION_8_5`` (13 lines; the registry has one row per geometry class, so 19 rows), sections 4.4, 8.1, 8.2, 8.3 and
12.2 of the note for the rest. Nothing is generated from the registry.

This file needs the compiled core (``_core``) with the ``physics`` binding, and the ``axmp`` executable of the same
build (``AXMP_BIN``, ``AXMP_EXE`` or a build directory), as ``tests/test_error_fields.py`` does.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import axionmultiphysics as axmp

ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "tests" / "g0"

_spec = importlib.util.spec_from_file_location("g0_harness_catalogue", G0 / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)

SCHEMA = "axionmultiphysics.physics_catalogue/1"

# Section 8.5, the table's 13 lines: (physics, couplings, geometries, study type, rows the line gives). The row names
# are <what>_<geometry class>; they are the specification of the registry.
SECTION_8_5 = [
    (("fluid.incompressible",), (), ("planar",), "stationary", ["stokes_planar"]),
    (("fluid.incompressible",), (), ("axisymmetric",), "stationary", ["stokes_axisymmetric"]),
    (("fluid.incompressible",), (), ("planar",), "stationary", ["navier_stokes_steady_planar"]),
    (("fluid.incompressible",), (), ("axisymmetric",), "stationary", ["navier_stokes_steady_axisymmetric"]),
    (("fluid.incompressible",), (), ("planar", "axisymmetric"), "transient",
     ["navier_stokes_transient_planar", "navier_stokes_transient_axisymmetric"]),
    (("fluid.incompressible",), (), ("3d",), "stationary", ["stokes_3d"]),
    (("heat.transfer",), (), ("planar", "axisymmetric"), "stationary", ["heat_planar", "heat_axisymmetric"]),
    (("heat.transfer",), (), ("planar", "axisymmetric"), "stationary",
     ["heat_prescribed_advection_planar", "heat_prescribed_advection_axisymmetric"]),
    (("fluid.incompressible", "heat.transfer"), ("advection",), ("planar", "axisymmetric"), "stationary",
     ["flow_then_heat_planar", "flow_then_heat_axisymmetric"]),
    (("fluid.incompressible", "heat.transfer"), ("advection", "buoyancy"), ("planar", "axisymmetric"), "stationary",
     ["boussinesq_planar", "boussinesq_axisymmetric"]),
    (("electrochem.current_distribution",), (), ("planar", "axisymmetric"), "stationary",
     ["current_distribution_planar", "current_distribution_axisymmetric"]),
    (("electrochem.current_distribution",), (), ("3d",), "stationary", ["current_distribution_3d"]),
    (("chem.reactor",), (), (), "transient", ["chem_reactor"]),
]
ALL_ROWS = sorted(r for line in SECTION_8_5 for r in line[4])

# The name of a development milestone (the letter U and a number, with an optional letter): internal planning vocabulary
# that the catalogue, the command line and the pages a user reads never show. A route says whether it is implemented,
# not when it will be.
MILESTONE = re.compile(r"\bU(1[0-2]|[1-9])[a-d]?\b")

TYPES = {  # section 8.2 and 4.4: name -> status
    "fluid.incompressible": "beta",
    "heat.transfer": "beta",
    "electrochem.current_distribution": "beta",
    "chem.reactor": "beta",
    "electrochem.porous_electrode": "beta",
    "current.conduction": "absent",
    "solid.linear": "absent",
    "species.transport": "absent",
}
HEAT_ROWS = [r for r in ALL_ROWS if r.startswith(("heat_", "flow_then_heat", "boussinesq"))]
FLOW_ROWS = [r for r in ALL_ROWS if r.startswith(("stokes", "navier_stokes", "flow_then_heat", "boussinesq"))]


# ----------------------------------------------------------------- helpers
def _axmp() -> str:
    exe = g0.find_axmp()
    if exe is None:
        pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")
    return exe


def _run(*args, cwd=None):
    return subprocess.run([_axmp(), "physics", *args], capture_output=True, text=True, timeout=120, cwd=cwd)


def _routes(cat: dict) -> dict:
    return {r["name"]: r for r in cat["routes"]}


def _types(cat: dict) -> dict:
    return {t["name"]: t for t in cat["types"]}


# ----------------------------------------------------------------- the command line and Python agree
@pytest.mark.parametrize("flags,kwargs", [
    ((), {}),
    (("--routes",), {"routes": True}),
    (("--type", "heat.transfer"), {"type": "heat.transfer"}),
    (("--routes", "--type", "heat.transfer"), {"routes": True, "type": "heat.transfer"}),
])
def test_cli_and_python_give_the_same_catalogue(flags, kwargs):
    """``axmp physics --json [--routes] [--type T]`` and ``axmp.physics(routes=, type=)`` print the same JSON. The JSON
    does not depend on the build (validation never depends on the build, section 16 rule 1), so the executable and the
    Python module may come from different builds."""
    r = _run("--json", *flags)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout) == axmp.physics(**kwargs)


def test_all_is_implied_by_json():
    """Section 12.2: ``--all`` is implied in ``--json``."""
    a = _run("--json")
    b = _run("--json", "--all")
    assert a.returncode == 0 and b.returncode == 0
    assert json.loads(a.stdout) == json.loads(b.stdout)


def test_physics_is_a_function_beside_schema():
    """The plan: ``physics(routes=False, type=None)`` in design.py, exported next to ``schema``; no new module, so the
    name is not both a module and a function."""
    assert callable(axmp.physics) and not inspect.ismodule(axmp.physics)
    assert "physics" in axmp.__all__
    params = inspect.signature(axmp.physics).parameters
    assert list(params)[:2] == ["routes", "type"]
    assert params["routes"].default is False and params["type"].default is None


# ----------------------------------------------------------------- the JSON
def test_catalogue_schema_string():
    cat = axmp.physics()
    assert cat["schema"] == SCHEMA
    assert cat["status"] == "beta"
    assert "routes" not in cat, "the route table is printed only with routes=True (axmp physics --routes)"
    assert isinstance(cat["types"], list) and isinstance(cat["terms"], list)
    assert set(_types(cat)) == set(TYPES)
    for name, status in TYPES.items():
        assert _types(cat)[name]["status"] == status, name
    assert set(cat["reserved_ids"]) == {"em", "mechanics", "thermal", "electrostatics", "hts", "fullwave", "cavity",
                                        "mesh_motion"}
    with_routes = axmp.physics(routes=True)
    assert with_routes["schema"] == SCHEMA and len(with_routes["routes"]) == 19
    json.dumps(with_routes)  # plain JSON


def test_every_type_is_complete():
    cat = axmp.physics()
    for name, t in _types(cat).items():
        assert t["family"] == name.split(".")[0], name
        assert t["title"], name
        assert isinstance(t["aliases"], list) and isinstance(t["provides"], list), name
        assert isinstance(t["not_implemented"], list), name
        if t["status"] == "beta":
            assert t["not_implemented"], f"{name}: a beta type lists what is not implemented (section 17)"
        for kind in t["condition_kinds"]:
            assert kind["name"] and isinstance(kind["aliases"], list) and isinstance(kind["data"], list), (name, kind)
            for key in kind["data"]:
                assert key["name"] and "type" in key and "unit" in key, (name, kind["name"], key)
    # quantities per geometry class (8.1, 10.3)
    for name in ("fluid.incompressible", "heat.transfer", "electrochem.current_distribution"):
        q = _types(cat)[name]["quantities"]
        assert q["planar"] and q["axisymmetric"], name
        for quantity in q["planar"]:
            assert quantity["name"] and quantity["unit"] is not None, (name, quantity)
    assert _types(cat)["fluid.incompressible"]["quantities"]["3d"]


def test_condition_kinds_and_data_keys():
    t = _types(axmp.physics())
    kinds = {n: [k["name"] for k in t[n]["condition_kinds"]] for n in t}
    assert set(kinds["fluid.incompressible"]) == {"wall", "inlet", "outlet", "open", "traction", "slip", "symmetry",
                                                  "periodic"}
    assert set(kinds["heat.transfer"]) == {"temperature", "heat_flux", "total_flux", "convection", "insulation",
                                           "outflow"}
    assert set(kinds["electrochem.current_distribution"]) == {"electrode", "insulation", "normal_current"}
    assert set(kinds["current.conduction"]) == {"potential", "ground", "insulation"}   # the busbar, 4.4
    assert set(kinds["solid.linear"]) == {"fixed", "free"}
    data = {(n, k["name"]): {d["name"]: d for d in k["data"]} for n in t for k in t[n]["condition_kinds"]}
    assert data[("fluid.incompressible", "inlet")]["velocity_m_s"]["field_expression"] is True
    assert set(data[("fluid.incompressible", "periodic")]) >= {"partner", "translation_m", "pressure_drop_Pa", "flow_rate"}
    assert set(data[("heat.transfer", "convection")]) >= {"h_W_m2K", "T_ext_K"}
    electrode = data[("electrochem.current_distribution", "electrode")]
    assert electrode["current_A"]["per_depth_twin"] == "current_A_m"    # section 3.3
    assert electrode["current_A"]["rampable"] is True and electrode["phi_s_V"]["rampable"] is True   # section 3.8
    sources = {s["name"]: s for s in t["heat.transfer"]["source_kinds"]}
    assert {"heat_source", "prescribed_advection"} <= set(sources)
    assert {d["name"]: d for d in sources["heat_source"]["data"]}["P_W"]["per_depth_twin"] == "P_W_m"


def test_options_and_solver_settings():
    cat = axmp.physics(routes=True)
    opts = {n: {o["name"]: o for o in t["options"]} for n, t in _types(cat).items()}
    stab = opts["heat.transfer"]["stabilization"]
    assert stab["values"] == ["none", "supg"] and stab["default"] == "none"
    assert stab["max_geometry_order"] == {"supg": 1}                        # section 3.3
    assert set(opts["heat.transfer"]["advective_form"]["values"]) == {"convective", "conservative_flux",
                                                                      "conservative_divergence"}
    assert opts["fluid.incompressible"]["viscous_form"]["default"] == "stress"      # Q19
    assert opts["fluid.incompressible"]["convection"]["default"] == "standard"
    assert "swirl" not in opts["fluid.incompressible"]                              # Q23
    assert {"primary", "secondary"} == set(opts["electrochem.current_distribution"]["kinetics"]["values"])
    routes = _routes(cat)
    for name in ("navier_stokes_steady_planar", "navier_stokes_steady_axisymmetric"):
        solvers = {s["name"]: s for s in routes[name]["solvers"]}
        assert solvers["newton_rtol"]["default"] == 1e-12 and solvers["n_picard"]["default"] == 3
        assert solvers["residual_floor_factor"]["default"] == 4
    assert "residual_floor_factor" not in {s["name"] for s in routes["boussinesq_planar"]["solvers"]}


def test_terms():
    terms = {t["name"]: t for t in axmp.physics()["terms"]}
    assert set(terms) >= {"advection", "buoyancy", "joule_heating", "thermal_expansion"}
    assert terms["advection"]["status"] == "beta" and terms["advection"]["support"] == "domain"
    assert terms["buoyancy"]["status"] == "beta" and terms["buoyancy"]["support"] == "domain"
    assert {d["name"] for d in terms["buoyancy"]["data"]} == {"T_ref_K", "gravity_m_s2"}
    assert terms["joule_heating"]["status"] == "absent" and terms["thermal_expansion"]["status"] == "absent"
    assert terms["advection"]["producers"] == ["fluid.*"]
    assert "heat.transfer" in terms["advection"]["consumers"]


# ----------------------------------------------------------------- the route table
def test_routes_list_every_line_of_section_8_5():
    """The note's table has 13 lines; the registry has one row per geometry class, so six lines that name two classes
    are two rows each: 19 rows."""
    routes = _routes(axmp.physics(routes=True))
    assert sorted(routes) == ALL_ROWS and len(ALL_ROWS) == 19
    for physics, couplings, geometries, study, names in SECTION_8_5:
        got_geoms = []
        for n in names:
            r = routes[n]
            assert tuple(r["physics"]) == physics, n
            assert tuple(r["couplings"]) == couplings, n
            assert r["study_types"] == [study], n
            assert len(r["geometries"]) <= 1, f"{n}: a row is for one geometry class (section 8.1)"
            got_geoms += r["geometries"]
        assert tuple(sorted(got_geoms)) == tuple(sorted(geometries)), names


def test_route_rows_carry_the_fields_of_the_note():
    routes = _routes(axmp.physics(routes=True))
    for n, r in routes.items():
        assert {"name", "physics", "cell_types", "couplings", "geometries", "study_types", "strategies", "when", "engine",
                "status", "implemented", "default_strategy", "requires", "condition_kinds", "not_offered",
                "axisymmetric_integrals", "force_reaction", "options", "material_blocks", "uniform", "one_expression",
                "element_types", "equal_domains", "rampable", "limits", "layer_rules", "solvers"} <= set(r), n
        assert set(r["requires"]) == {"mesher", "mpi", "fluids", "engine_3d", "max_ranks"}, n
    # section 8.5 "needs" and section 16
    for n in HEAT_ROWS + [r for r in ALL_ROWS if r.startswith(("stokes", "navier"))]:
        assert routes[n]["requires"]["fluids"] is True, n
    for n in ("current_distribution_planar", "current_distribution_axisymmetric"):
        assert routes[n]["requires"]["fluids"] is False and routes[n]["requires"]["mpi"] is False, n
        assert routes[n]["requires"]["max_ranks"] == 1, n
    assert routes["current_distribution_3d"]["requires"]["mpi"] is True
    assert routes["chem_reactor"]["requires"]["mesher"] is False and routes["chem_reactor"]["geometries"] == []
    for n in HEAT_ROWS:
        assert routes[n]["requires"]["max_ranks"] == 1, n
    # the flow engines solve on a partitioned mesh, the time stepper included (8.5 says "one rank" for the transient
    # rows; the engine has no such limit, and the engine half of the matrix test steps it on two ranks)
    for n in ("stokes_planar", "stokes_axisymmetric", "navier_stokes_steady_planar", "stokes_3d",
              "navier_stokes_transient_planar", "navier_stokes_transient_axisymmetric"):
        assert routes[n]["requires"]["max_ranks"] == 0, n
    # selecting option values, strategies, equal domains
    assert routes["stokes_planar"]["when"] == {"equation": "stokes"}
    assert routes["navier_stokes_steady_planar"]["when"] == {"equation": "navier_stokes"}
    assert routes["flow_then_heat_planar"]["strategies"] == ["one_way"]
    assert routes["boussinesq_planar"]["strategies"] == ["monolithic"]
    assert routes["boussinesq_planar"]["default_strategy"] == "monolithic"
    assert routes["boussinesq_planar"]["equal_domains"] is True and routes["heat_planar"]["equal_domains"] is False
    # element types (8.5): the transport operator takes triangles only; the flow engines take quadrilaterals as well
    for n in HEAT_ROWS:
        assert routes[n]["element_types"] == ["triangle"], n
    for n in ("stokes_planar", "navier_stokes_steady_axisymmetric", "navier_stokes_transient_planar"):
        assert set(routes[n]["element_types"]) == {"triangle", "quadrilateral"}, n
    # one-expression and uniform data (8.5)
    assert set(routes["navier_stokes_steady_planar"]["uniform"]["fluid.incompressible"]) == {"density"}
    assert routes["stokes_planar"]["uniform"] == {} or all(not v for v in routes["stokes_planar"]["uniform"].values())
    one = {p for v in routes["heat_planar"]["one_expression"].values() for p in v}
    assert one == {"thermal.k_W_mK", "heat_source"}
    uni = {p for v in routes["boussinesq_planar"]["uniform"].values() for p in v}
    assert uni == {"density", "thermal.cp_J_kgK", "fluid.beta_1_K"}
    # conventions of the last column
    assert routes["stokes_axisymmetric"]["axisymmetric_integrals"] == "includes_2pi"
    assert routes["heat_axisymmetric"]["axisymmetric_integrals"] == "per_radian"
    assert routes["stokes_planar"]["force_reaction"] == "physical_reaction"
    assert routes["navier_stokes_steady_planar"]["force_reaction"] == "reaction: walls at rest only with the laplacian form"
    # layer rules (section 16 rule 2): six, each on the rows the C++ test lists
    rules = {x for r in routes.values() for x in r["layer_rules"]}
    assert rules == {"insulation_on_throughflow", "outflow_with_conservative_flux", "force_with_laplacian_form",
                     "per_depth_keys", "geometry_order_exceeds_field_order", "uniform_capacity_in_stagnant_part"}
    # forced option: the Boussinesq energy form is convective only (8.5 row 10)
    forced = {o["name"]: o["forced"] for o in routes["boussinesq_planar"]["options"]["heat.transfer"] if o["forced"]}
    assert forced == {"advective_form": "convective"}
    offered = {o["name"]: o["offered"] for o in routes["boussinesq_planar"]["options"]["heat.transfer"]}
    assert offered["order"] == ["2"]                                         # the engine refuses a P1 temperature


def test_every_route_is_not_implemented_yet():
    """No route of this version is implemented. A route says so with the boolean ``implemented`` beside its status;
    nothing in it says when it will be."""
    routes = _routes(axmp.physics(routes=True))
    for n, r in routes.items():
        assert r["status"] == "not_implemented", n
        assert r["implemented"] is False, n
        assert "arrives_with" not in r, f"{n}: the route names the milestone that implements it"


def _section(text: str, start: str, stop: str) -> str:
    """The text from the first line that contains ``start`` to the next line that starts with ``stop``."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if start in line:
            out = [line]
            for rest in lines[i + 1:]:
                if rest.startswith(stop):
                    break
                out.append(rest)
            return "\n".join(out)
    return ""


def _newest_changelog_section(changelog: str) -> str:
    """The newest CHANGELOG section that mentions the catalogue ("axmp physics"): "Unreleased" while its entries do,
    else the newest release section (an unrelated new entry under "Unreleased" does not matter)."""
    for part in changelog.split("\n## ")[1:]:
        title, _, body = part.partition("\n")
        if "axmp physics" in body:
            return body
    return ""


def test_no_milestone_name_reaches_the_user(tmp_path):
    """What a user reads about the catalogue says "not implemented yet" and names no development milestone: the JSON
    of ``axmp.physics()`` and ``axmp physics --json``, every text form of ``axmp physics``, the generated reference page,
    and the catalogue's entries in docs/cli.md, llms.txt and the CHANGELOG."""
    def read(p):
        return (ROOT / p).read_text("utf-8")
    texts = {
        "axmp.physics(routes=True)": json.dumps(axmp.physics(routes=True)),
        "axmp.physics(routes=True, type='heat.transfer')": json.dumps(axmp.physics(routes=True, type="heat.transfer")),
    }
    for flags in ((), ("--routes",), ("--routes", "--all"), ("--all",), ("--json", "--routes")):
        r = _run(*flags, cwd=tmp_path)
        assert r.returncode == 0, r.stdout + r.stderr
        texts["axmp physics " + " ".join(flags)] = r.stdout + r.stderr
    texts["docs/reference/physics_catalogue.md"] = read("docs/reference/physics_catalogue.md")
    texts["docs/cli.md"] = "\n".join(line for line in read("docs/cli.md").splitlines() if "physics" in line)
    texts["llms.txt"] = _section(read("llms.txt"), "axmp.physics(", "- ")
    changelog = read("CHANGELOG.md")
    unreleased = _newest_changelog_section(changelog)
    texts["CHANGELOG.md (newest section that names the catalogue)"] = _section(unreleased, "axmp physics", "- ")
    assert texts["llms.txt"] and texts["CHANGELOG.md (newest section that names the catalogue)"], "the catalogue's entries were not found"
    for where, text in texts.items():
        m = MILESTONE.search(text)
        assert m is None, f"{where} names the milestone {m.group(0)!r}: ...{text[max(0, m.start() - 80):m.end() + 40]}..."
        assert "arrives with" not in text and "arrives_with" not in text, where
    routes_text = texts["axmp physics --routes"]
    for n in ALL_ROWS:
        line = next((x for x in routes_text.splitlines() if x.strip().startswith(n + " ")), "")
        assert "not implemented yet" in line, f"axmp physics --routes, row {n}: {line!r}"


def test_type_filter():
    cat = axmp.physics(routes=True, type="heat.transfer")
    assert [t["name"] for t in cat["types"]] == ["heat.transfer"]
    assert sorted(r["name"] for r in cat["routes"]) == sorted(HEAT_ROWS) and len(HEAT_ROWS) == 8
    assert {t["name"] for t in cat["terms"]} == {"advection", "buoyancy", "joule_heating", "thermal_expansion"}
    flow = axmp.physics(routes=True, type="fluid.incompressible")
    assert sorted(r["name"] for r in flow["routes"]) == sorted(FLOW_ROWS) and len(FLOW_ROWS) == 11
    assert {t["name"] for t in flow["terms"]} == {"advection", "buoyancy"}     # fluid.* produces, fluid is a consumer
    reactor = axmp.physics(routes=True, type="chem.reactor")
    assert [r["name"] for r in reactor["routes"]] == ["chem_reactor"] and reactor["terms"] == []
    assert "routes" not in axmp.physics(type="heat.transfer")
    assert axmp.physics(type="heat.transfer")["schema"] == SCHEMA


def test_an_unknown_type_is_refused_with_the_known_names():
    for bad in ("no.such_type", "laminar_flow"):
        with pytest.raises(axmp.AxionMultiPhysicsError) as e:
            axmp.physics(type=bad)
        assert bad in str(e.value)
        assert "fluid.incompressible" in e.value.hint and "heat.transfer" in e.value.hint


def test_aliases_name_the_spelling_to_use():
    """Section 8.1: COMSOL vocabulary is discoverable without becoming a second spelling. The catalogue lists the
    alias on the entry it names; a document that writes it is refused with a hint."""
    t = _types(axmp.physics())
    assert "laminar_flow" in t["fluid.incompressible"]["aliases"]
    wall = {k["name"]: k for k in t["fluid.incompressible"]["condition_kinds"]}["wall"]
    assert "no_slip" in wall["aliases"]
    for name, entry in t.items():          # an alias is never also a name
        assert not set(entry["aliases"]) & set(t), name


def test_absent_types_are_known_and_absent():
    """The busbar of 4.4 needs ``current.conduction``, ``solid.linear`` and the terms ``joule_heating`` and
    ``thermal_expansion`` to parse; they are catalogue entries with no route row, and the phrases of the 4.4
    inventory."""
    cat = axmp.physics(routes=True)
    t = _types(cat)
    assert t["current.conduction"]["status"] == "absent" and t["solid.linear"]["status"] == "absent"
    assert t["current.conduction"]["absent_phrase"] == "physics type 'current.conduction'"
    assert t["solid.linear"]["absent_phrase"] == "'solid.linear' as a physics instance"
    assert t["species.transport"]["status"] == "absent"
    assert t["current.conduction"]["provides"] and t["solid.linear"]["provides"]
    routed = {p for r in cat["routes"] for p in r["physics"]}
    assert not routed & {"current.conduction", "solid.linear", "species.transport", "electrochem.porous_electrode"}
    assert {x["name"] for x in cat["terms"] if x["status"] == "absent"} == {"joule_heating", "thermal_expansion"}
    # the beta porous electrode is name-only
    assert t["electrochem.porous_electrode"]["status"] == "beta" and t["electrochem.porous_electrode"]["condition_kinds"] == []


# ----------------------------------------------------------------- the command line
def test_physics_needs_no_document(tmp_path):
    """The command is handled with the document-less commands: it runs from an empty directory, with no file."""
    r = _run(cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    for name in TYPES:
        assert name in r.stdout, name
    assert "beta" in r.stdout


def test_routes_are_printed_one_line_each(tmp_path):
    r = _run("--routes", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    for n in ALL_ROWS:
        assert n in r.stdout, n
    assert "not implemented" in r.stdout
    only = _run("--routes", "--type", "chem.reactor", cwd=tmp_path)
    assert only.returncode == 0 and "chem_reactor" in only.stdout and "stokes_planar" not in only.stdout


def test_usage_errors_exit_2_and_a_refused_type_exits_1(tmp_path):
    assert _run("--no-such-flag", cwd=tmp_path).returncode == 2
    assert _run("--type", cwd=tmp_path).returncode == 2
    r = _run("--type", "no.such_type", "--json", cwd=tmp_path)
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["ok"] is False and "no.such_type" in out["error"] and "fluid.incompressible" in out["hint"]
    usage = subprocess.run([_axmp()], capture_output=True, text=True, timeout=60)
    assert usage.returncode == 2 and "axmp physics" in (usage.stdout + usage.stderr)


# ----------------------------------------------------------------- the generated page
PAGE = ROOT / "docs" / "reference" / "physics_catalogue.md"
GENERATOR = ROOT / "scripts" / "docs" / "gen_physics_catalogue.py"


def _generate() -> str:
    env = dict(os.environ)
    home = str(Path(axmp.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (home, env.get("PYTHONPATH", "")) if p)
    r = subprocess.run([sys.executable, str(GENERATOR)], capture_output=True, text=True, timeout=300, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_reference_page_is_generated():
    """``scripts/docs/gen_physics_catalogue.py`` prints the page from ``axionmultiphysics.physics(routes=True)``; the
    committed ``docs/reference/physics_catalogue.md`` is its output byte for byte. A test fails when they differ
    (section 8.1: the capability tables are generated from the registry)."""
    assert GENERATOR.is_file(), "scripts/docs/gen_physics_catalogue.py"
    assert PAGE.is_file(), "docs/reference/physics_catalogue.md (regenerate with: python scripts/docs/gen_physics_catalogue.py > docs/reference/physics_catalogue.md)"
    text = _generate()
    assert PAGE.read_text("utf-8") == text, "the committed page is stale: regenerate it from the registry"
    assert text.startswith("# ") and "beta" in text.lower()
    for n in list(TYPES) + ALL_ROWS:
        assert n in text, f"the generated page does not mention {n}"
    for t in ("advection", "buoyancy", "joule_heating", "thermal_expansion"):
        assert t in text, t
    assert _generate() == text, "the generator is not deterministic"


def test_docs_changelog_llms_and_premerge_are_updated():
    """The plan's "Files to change": cli.md, api.md, mkdocs.yml, CHANGELOG.md (Unreleased), llms.txt, premerge.txt."""
    def read(p):
        return (ROOT / p).read_text("utf-8")
    assert "axmp physics" in read("docs/cli.md")
    assert "physics" in read("docs/api.md") and "axionmultiphysics.physics" in read("docs/api.md")
    assert "reference/physics_catalogue.md" in read("mkdocs.yml")
    assert "axmp physics" in read("llms.txt")
    changelog = read("CHANGELOG.md")
    unreleased = _newest_changelog_section(changelog)
    assert "axmp physics" in unreleased and "beta" in unreleased.lower(), "CHANGELOG.md, newest section that names axmp physics: marked beta"
    assert "tests/test_physics_catalogue.py" in read("tests/premerge.txt")
