"""The error codes of the model document (beta) and the page ``docs/reference/error_codes.md`` generated from them.

The codes of the model path live in one table, in ``cpp/src/model_errors.hpp`` (namespace ``codes``), which the binding
``_core.error_codes()`` returns as the JSON text of a list of objects, one per code, sorted by code::

    {"code": "heat.option_not_offered", "kinds": ["error"], "summary": "...", "status": "raised"}

``kinds`` lists the finding kinds the code is raised as ("error", "unavailable", "deferred", "warning"); ``status`` is
"raised" for a code this version raises and "reserved" for a code that this version names for a later one (the
two-rank refusal of a one-rank route, an engine that refuses what validation accepted, and an option refusal that no
route of this version has). ``summary`` is one sentence.
``scripts/docs/gen_error_codes.py`` prints the page from that list; the committed page is its output byte for byte.

Design note section 14.1: "The list is generated from the source into docs/reference/error_codes.md; a test asserts that
every code raised in the model path is documented." Independent references: the code lists of the development plan (the
codes of the parser and of the validation), the codes the parser added beyond that list (``boundary.reserved_name`` and the 17
others below), and the codes the C++ tests of the model path assert, read from their source here.

This file needs the compiled core (``_core``).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import axionmultiphysics as axmp

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reference" / "error_codes.md"
GENERATOR = ROOT / "scripts" / "docs" / "gen_error_codes.py"
ERRORS_HPP = ROOT / "cpp" / "src" / "model_errors.hpp"

# <area>.<problem>: one lowercase word, a dot, a lowercase identifier (design note 14.1).
CODE_SHAPE = re.compile(r"^[a-z]+\.[a-z][a-z0-9_]*$")
LITERAL = re.compile(r'"([a-z]+\.[a-z][a-z0-9_]*)"')
FILE_NAME = re.compile(r"\.(json|msh|md|hpp|cpp|py|txt)$")
KINDS = {"error", "unavailable", "deferred", "warning"}
AREAS = {"model", "mesh", "physics", "boundary", "domain", "material", "expr", "geometry", "law", "coupling", "study", "output",
         "reaction", "heat", "fluid", "electrochem", "route", "json", "patch", "build", "internal", "solve", "sweep"}

# The plan: the codes the parser raises.
PARSER = [
    "model.mesh_order", "model.geometry_order_from_file", "model.per_depth_key", "model.em_sections_not_combinable",
    "physics.unknown_type", "physics.unknown_option", "physics.unknown_condition_kind", "physics.reserved_id",
    "boundary.unknown_name", "boundary.name_collision", "material.expansion_ambiguous", "material.conductivity_ambiguous",
    "expr.parameter_shadows_variable", "expr.unknown_variable", "expr.vector_length", "law.range_missing",
    "coupling.unknown_term", "coupling.type_mismatch", "coupling.wrong_selection_key", "study.unknown_physics",
    "study.initial_from_unknown_step", "output.unknown_quantity", "output.unknown_boundary", "output.cycle",
    "model.symmetry_not_offered", "model.exterior_not_offered", "model.mesh_key_not_offered", "study.type_reserved",
    "material.block_incomplete",
]
# The codes the parser added beyond the plan's list.
PARSER_EXTRA = [
    "boundary.reserved_name", "boundary.selector_grammar", "boundary.imported_mesh_selector", "domain.unknown_name",
    "physics.invalid_id", "physics.unknown_source_kind", "coupling.unknown_physics", "output.unknown_kind",
    "output.grammar", "output.unknown_physics", "reaction.unknown_kind", "reaction.unknown_name", "law.unknown_kind",
    "law.unknown_field", "law.unknown_option", "law.table_invalid", "boundary.cycle", "study.solver_value",
]
# The plan: the codes the validation raises; then the ones its tests add: the transient rows' convective form
# (the engine half of the matrix test), the element types of an imported mesh (deferred in this version) and the beta
# warning of section 17.
VALIDATION = [
    "model.geometry_order_exceeds_field_order", "mesh.geometry_order_lowered", "model.geometry_order_not_offered",
    "heat.option_not_offered", "heat.supg_needs_advection", "heat.prescribed_advection_partial", "heat.data_not_regionwise",
    "heat.condition_not_offered", "heat.two_advections", "fluid.condition_not_offered", "fluid.outlet_stress_form",
    "fluid.pseudo_traction_on_curved_boundary", "fluid.gravity_direction", "fluid.buoyancy_needs_beta",
    "electrochem.potential_level_free", "output.force_needs_stress_form", "study.type_not_offered",
    "study.strategy_not_offered", "study.two_way_needs_strategy", "study.field_not_rampable", "study.ramp_end_mismatch",
    "physics.property_not_uniform_over_domain", "physics.material_block_missing", "physics.domains_overlap",
    "route.domains_must_coincide", "route.not_implemented", "route.law_not_offered", "coupling.empty_overlap",
    "coupling.restriction_not_offered", "coupling.uncoupled_pair", "law.field_not_provided",
    "fluid.option_not_offered", "model.element_type_not_offered", "model.beta_feature",
]
# The checks that the plan defers to the resolution of the boundaries (reported as deferred findings), and the codes
# of the checks of the design note that the plan's list has no code for: the key "in" as a subset, the routes without
# a force output, a continuation scale without its coefficient.
VALIDATION += [
    "boundary.empty_selection", "physics.boundary_uncovered", "coupling.selection_outside_support",
    "physics.source_outside_domain", "output.force_not_offered", "study.scale_not_applicable",
]
# The switch between the two parsers (design note 3.2, 15.2 row 15): the guard of the electromagnetic parser, the
# refusal of a model document at an electromagnetic-only operation, a key repeated in a model document's file.
SWITCH = ["model.section_needs_physics", "model.em_only_operation", "json.duplicate_key"]
# Codes that this version names for a later one (the matrix test's cases carry the first two; the last is the code of an
# option refusal that no route of this version has); listed, not raised.
RESERVED = ["route.single_rank", "internal.route_table_out_of_date", "physics.option_not_offered"]
# The checks on the resolved boundaries (design note 5.2, 5.5, 10.3), which `axmp boundaries`, `axmp mesh` and the
# Python API run (tests/test_model_boundaries.py): the two other refusals of the partition rule, a key with nothing on
# the boundary of its domain or inside it, an entity a where filter matches in part (a warning), between with two
# domains that share a region (at validation), an output on a boundary outside its physics' domain, a force on an
# interior section.
RESOLVED = ["physics.boundary_double", "physics.boundary_interior", "physics.boundary_outside_domain",
            "boundary.partial_match", "boundary.between_overlap", "output.boundary_not_in_domain", "output.force_on_interior"]
# Meshing a model document (tests/test_model_mesh_cli.py, tests/test_model_requires_mesher.py): an installation whose
# mesher does not run, a failed mesher run, a shape the bounded modes cannot mesh (a contour with an elliptical segment),
# and the internal error of the meshed model and its exporters (mesher files that disagree with the library's resolution
# of the boundaries, an instance face without an atom), which `axmp mesh` and `axmp.mesh` reach.
MESHING = ["build.requires_mesher", "mesh.generation_failed", "model.shape_not_offered", "internal.mesh_export"]
# The bridge to the engines (tests/test_fluids_model_bridge.py): im.fluids_mesh() in a build without the fluids module.
BRIDGE = ["build.requires_fluids"]

# The kinds that the tests of the model path fix for some codes.
KINDS_OF = {
    "mesh.tangled": {"error"},
    "route.not_implemented": {"unavailable"},
    "mesh.geometry_order_lowered": {"warning"},
    "coupling.uncoupled_pair": {"warning"},
    "fluid.outlet_stress_form": {"warning"},
    "fluid.pseudo_traction_on_curved_boundary": {"warning"},
    "model.beta_feature": {"warning"},
    "boundary.imported_mesh_selector": {"deferred"},
    "model.element_type_not_offered": {"deferred"},
    "model.geometry_order_exceeds_field_order": {"error", "deferred"},
    "model.geometry_order_not_offered": {"error", "deferred"},
    # deferred at validation, errors once the boundaries are resolved
    "boundary.empty_selection": {"deferred", "error"},
    "physics.boundary_uncovered": {"deferred", "error"},
    # errors of the full check only (the deferred finding of the partition rule carries physics.boundary_uncovered)
    "physics.boundary_double": {"error"},
    "physics.boundary_interior": {"error"},
    "physics.boundary_outside_domain": {"error"},
    "boundary.partial_match": {"warning"},
    "boundary.between_overlap": {"error"},
    "output.boundary_not_in_domain": {"error"},
    "output.force_on_interior": {"error"},
    "output.force_needs_stress_form": {"error", "deferred"},
    "output.force_not_offered": {"error"},
    "model.section_needs_physics": {"error"},
    "model.em_only_operation": {"error"},
    "json.duplicate_key": {"error"},
    "internal.mesh_export": {"error"},
    "build.requires_mesher": {"unavailable"},
    "build.requires_fluids": {"unavailable"},
    "mesh.generation_failed": {"error"},
    "model.shape_not_offered": {"error"},
}

# The C++ tests of the model path, whose code literals are the codes they assert.
CPP_TESTS = ["cpp/tests/test_spec_model.cpp", "cpp/tests/test_spec_model_materials.cpp", "cpp/tests/test_model_validate.cpp",
             "cpp/tests/test_model_graph.cpp", "cpp/tests/test_model_matrix.cpp", "cpp/tests/test_field_expr.cpp",
             "cpp/tests/test_model_switch.cpp", "cpp/tests/test_model_selection.cpp", "cpp/tests/test_model_mesh.cpp",
             "cpp/tests/test_model_mesh_mesher.cpp", "cpp/tests/test_model_mesh_tangled.cpp"]
# Literals of a code's shape in those tests that are not codes: document keys and a deliberately misspelt type name.
NOT_CODES = {"mesh.order", "mesh.adapt", "fluid.incompressibl"}
# Literals of a code's shape in the message builders of model_errors.hpp that are not codes: the stems of property names
# by which a builder chooses a label ("fluid.mu" of fluid.mu_Pa_s).
PROPERTY_STEMS = {"fluid.beta", "fluid.mu", "thermal.cp", "thermal.k"}


def _table() -> list[dict]:
    from axionmultiphysics import _core as core
    assert hasattr(core, "error_codes"), "_core.error_codes() is missing (cpp/python/bindings.cpp)"
    table = json.loads(core.error_codes())
    assert isinstance(table, list) and table, "_core.error_codes() returns the JSON text of a non-empty list"
    return table


def _codes() -> dict[str, dict]:
    return {e["code"]: e for e in _table()}


def _catalogue_names() -> set[str]:
    cat = axmp.physics(routes=True)
    names = set()
    for t in cat["types"]:
        names.add(t["name"])
        names.update(t.get("aliases", []))
    for t in cat["terms"]:
        names.add(t["name"])
    return names


def _header_codes() -> set[str]:
    text = ERRORS_HPP.read_text("utf-8")
    m = re.search(r"namespace codes \{(.*?)\}\s*//\s*namespace codes", text, re.S)
    assert m, "cpp/src/model_errors.hpp has no `namespace codes { ... }  // namespace codes` block"
    return set(re.findall(r'inline constexpr const char\* \w+ = "([^"]+)";', m.group(1)))


def _generate() -> str:
    env = dict(os.environ)
    home = str(Path(axmp.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (home, env.get("PYTHONPATH", "")) if p)
    r = subprocess.run([sys.executable, str(GENERATOR)], capture_output=True, text=True, timeout=300, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_codes_are_dotted_identifiers():
    """Every code of the table has the shape ``<area>.<problem>``, a known area, a kind, a summary and a status; the
    table is sorted and has no duplicate; it has every code of the parser and of the validation, raised, and the reserved ones."""
    table = _table()
    codes = [e["code"] for e in table]
    assert codes == sorted(codes), "the table is sorted by code"
    assert len(codes) == len(set(codes)), "no code is listed twice"
    for e in table:
        assert set(e) >= {"code", "kinds", "summary", "status"}, e
        assert CODE_SHAPE.match(e["code"]), f"{e['code']} is not <area>.<problem>"
        assert e["code"].split(".")[0] in AREAS, f"{e['code']}: unknown area"
        assert e["kinds"] and set(e["kinds"]) <= KINDS, f"{e['code']}: kinds {e['kinds']}"
        assert isinstance(e["summary"], str) and e["summary"].strip(), f"{e['code']}: no summary"
        assert e["status"] in ("raised", "reserved"), f"{e['code']}: status {e['status']}"
    by_code = {e["code"]: e for e in table}
    for c in PARSER + PARSER_EXTRA + VALIDATION + SWITCH + RESOLVED + MESHING + BRIDGE:
        assert c in by_code, f"{c} is not in the table"
        assert by_code[c]["status"] == "raised", f"{c}: status {by_code[c]['status']}"
    for c in RESERVED:
        assert c in by_code and by_code[c]["status"] == "reserved", f"{c} is listed as reserved"
    for c, kinds in KINDS_OF.items():
        assert set(by_code[c]["kinds"]) == kinds, f"{c}: kinds {by_code[c]['kinds']}, expected {sorted(kinds)}"


def test_error_codes_page_is_generated():
    """``scripts/docs/gen_error_codes.py`` prints the page from ``_core.error_codes()``; the committed page is its output
    byte for byte, it lists every code, and the docs navigation links it."""
    assert GENERATOR.is_file(), "scripts/docs/gen_error_codes.py"
    assert PAGE.is_file(), ("docs/reference/error_codes.md (regenerate with: python scripts/docs/gen_error_codes.py > "
                            "docs/reference/error_codes.md)")
    text = _generate()
    assert PAGE.read_text("utf-8") == text, "the committed page is stale: regenerate it"
    assert _generate() == text, "the generator is not deterministic"
    assert text.startswith("# ") and "beta" in text.lower()
    for e in _table():
        assert f"`{e['code']}`" in text, f"the page does not list {e['code']}"
    assert "reference/error_codes.md" in (ROOT / "mkdocs.yml").read_text("utf-8")


def test_every_code_asserted_in_a_test_is_listed():
    """Every code that a C++ test of the model path asserts is in the table (a code-shaped literal of those tests that
    is not a catalogue name, a file name or one of the document keys in NOT_CODES)."""
    table = _codes()
    names = _catalogue_names()
    missing = {}
    seen = set()
    for rel in CPP_TESTS:
        path = ROOT / rel
        assert path.is_file(), rel
        for lit in LITERAL.findall(path.read_text("utf-8")):
            if lit in names or lit in NOT_CODES or FILE_NAME.search(lit) or lit.split(".")[0] not in AREAS:
                continue
            seen.add(lit)
            if lit not in table:
                missing.setdefault(lit, []).append(rel)
    assert not missing, f"codes asserted in tests and missing from the table: {missing}"
    # the scan reads the tests of this change
    for c in ("heat.option_not_offered", "route.not_implemented", "coupling.uncoupled_pair", "model.beta_feature",
              "model.em_only_operation", "json.duplicate_key"):
        assert c in seen, f"the scan did not find {c} in the C++ tests"


def test_no_code_is_raised_outside_the_table():
    """The codes live in one place (design note risk 10; the plan's "texts in one place"): the table is the ``codes``
    namespace of ``cpp/src/model_errors.hpp``, and no other source of the model path writes a code literal. A
    code-shaped literal outside that header is a catalogue name (a physics type, an alias or a coupling term, such as
    ``fluid.incompressible``) or a file name."""
    table = _codes()
    header = _header_codes()
    assert header == set(table), (f"model_errors.hpp and the table differ: only in the header {sorted(header - set(table))}, "
                                  f"only in the table {sorted(set(table) - header)}")
    names = _catalogue_names()
    src = ROOT / "cpp" / "src"
    files = sorted(set(src.glob("spec_model*.cpp")) | set(src.glob("model_*.cpp")) | set(src.glob("model_*.hpp"))
                   | {src / "field_expr.cpp", src / "physics_registry.cpp"})
    files = [f for f in files if f.name != "model_errors.hpp"]
    for required in ("model_validate.cpp", "model_graph.cpp", "spec_model.cpp", "model_matrix.cpp"):
        assert any(f.name == required for f in files), f"cpp/src/{required} is not scanned"
    stray = {}
    for f in files:
        for lit in LITERAL.findall(f.read_text("utf-8")):
            if lit in names or FILE_NAME.search(lit):
                continue
            stray.setdefault(f.name, set()).add(lit)
    assert not stray, f"code literals outside cpp/src/model_errors.hpp (use the constants of namespace codes): {stray}"
    # model_errors.hpp itself, outside its `namespace codes` block: a message builder passes a constant, never a literal
    text = (src / "model_errors.hpp").read_text("utf-8")
    block = re.search(r"namespace codes \{.*?\}\s*//\s*namespace codes", text, re.S)
    assert block, "cpp/src/model_errors.hpp has no `namespace codes { ... }  // namespace codes` block"
    outside = text[:block.start()] + text[block.end():]
    assert "inline Error coded(" in outside and "code_table()" in outside, "the scan reads the builders and the table"
    found = set(LITERAL.findall(outside))
    assert found & PROPERTY_STEMS, "the scan finds the property stems of the builders, so it reads their literals"
    stray_here = {lit for lit in found if lit not in names and lit not in PROPERTY_STEMS and not FILE_NAME.search(lit)}
    assert not stray_here, (f"code literals in cpp/src/model_errors.hpp outside namespace codes (use the constants): "
                            f"{sorted(stray_here)}")
