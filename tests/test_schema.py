"""The hand-maintained JSON Schema (axmp schema) must agree with the C++ parser (axmp validate)."""
import copy
import json
import math
from pathlib import Path

import jsonschema
import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core
from axionmultiphysics.convert import to_3d

SCHEMA = json.loads(_core.schema())


def _circuit_design():
    return {"schema": "axionmultiphysics.design/1", "name": "circ",
            "regions": [{"name": "sol", "shape": {"rect": {"r": [1.0, 1.05], "z": [-1, 1]}}, "material": "winding"},
                        {"name": "pc", "shape": {"rect": {"r": [0.5, 0.52], "z": [-0.05, 0.05]}}, "material": "winding"}],
            "windings": [{"region": "sol", "type": "azimuthal", "ampere_turns": 1e5},
                         {"region": "pc", "type": "azimuthal", "turns": 100, "circuit": "loop"}],
            "circuits": [{"name": "loop", "type": "persistent", "flux_linkage": 0}],
            "background": {"type": "uniform", "Bz": 0.01}}


def _hts_design():
    d = axmp.template("core").doc
    d["name"] = "core_hts"
    d["conductors"] = {"tape": {"law": "rebco_kim", "T_op": 20.0, "fill_factor": 0.5, "n_value": 25,
                                "tape_normal": [1, 0]},
                       "tab": {"law": "table", "B_T": [0, 1, 2], "T_K": [4.2, 20.0],
                               "Jc": [4e9, 2e9, 2e9, 1e9, 0, 0]}}
    d["windings"][0]["conductor"] = "nbti"
    d["cost"] = {"load_line_cap": 0.9, "sc_wall_cap": 0.086}
    d["hts"] = {"regions": [{"region": "cavity", "conductor": "tape", "initial": "zero_field_cooled"}],
                "drive": {"t_ramp": 100.0, "t_hold": 10.0, "ramp_down": True, "cycles": 2},
                "steps_per_segment": 8, "field_of_view": [[0.0, 0.0], [0.1, 0.0]],
                "newton_tol": 1e-4, "max_newton": 60, "e_floor": 1e-6, "profile_points": 21}
    return d


def _valid_docs():
    docs = [axmp.template(t["name"]).doc for t in axmp.templates()]
    docs.append(axmp.template("core", iron_t=0.1, fl=0).doc)
    docs.append(to_3d(axmp.template("toroid")).doc)
    docs.append(_circuit_design())
    docs.append(_hts_design())
    for f in sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.json")):
        docs.append(json.loads(f.read_text()))
    return docs


@pytest.mark.parametrize("doc", _valid_docs(), ids=lambda d: d["name"])
def test_valid_designs_pass_schema_and_parser(doc):
    jsonschema.validate(doc, SCHEMA)
    axmp.Design(doc).validate()


INVALID_MUTATIONS = [
    (lambda d: d["windings"].append({"region": "sol", "type": "azimuthal"}), "winding without current or circuit"),
    (lambda d: d.update(unknown_key=1), "unknown top-level key"),
    (lambda d: d["regions"][0].update(colour="red"), "unknown region key"),
    (lambda d: d["background"].update(interpolation="spline"), "unknown interpolation"),
    (lambda d: d.update(cost={"cold_envelope": "sphere"}), "unknown cold envelope"),
    (lambda d: d.update(conductors={"c": {"law": "unobtainium"}}), "unknown conductor law"),
    (lambda d: d.update(conductors={"c": {"law": "rebco_kim", "colour": "red"}}), "unknown conductor key"),
    (lambda d: d.update(hts={"regions": [{"region": "sol"}], "drive": {"t_ramp": 1, "oops": 1}}),
     "unknown hts drive key"),
    (lambda d: d.update(hts={"drive": {"t_ramp": 1}}), "hts without regions"),
    (lambda d: d.update(electrostatics={"boundary": "floating"}), "unknown electrostatics boundary"),
    (lambda d: d.update(electrostatics={"modes": 4}), "unknown electrostatics key"),
    (lambda d: d.setdefault("mesh", {}).update(adapt={"cycles": 2, "watch": ["Q"]}), "unknown adapt.watch quantity"),
    (lambda d: d.setdefault("mesh", {}).update(adapt={"cycles": 2, "watch": []}), "empty adapt.watch"),
]


@pytest.mark.parametrize("mutate,what", INVALID_MUTATIONS)
def test_invalid_designs_fail_schema_and_parser(mutate, what):
    doc = _circuit_design()
    mutate(doc)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, SCHEMA)
    with pytest.raises(Exception):
        axmp.Design(doc).validate()


def test_parameter_expressions_use_standard_precedence():
    """Design parameters: unary minus binds looser than '^', which is right-associative."""
    d = axmp.template("core")
    d.add_param("u", 2.0)
    d.add_param("neg_sq", "-u^2")
    d.add_param("sq_of_neg", "(-u)^2")
    d.add_param("inv_sq", "u^-2")
    d.add_param("tower", "u^3^2")
    d.add_param("gauss", "exp(-u^2/(2*0.5^2))")
    d.add_param("chained", "1 - neg_sq")
    p = d.resolved_params()
    assert p["neg_sq"] == -4.0
    assert p["sq_of_neg"] == 4.0
    assert p["inv_sq"] == 0.25
    assert p["tower"] == 512.0
    assert p["gauss"] == pytest.approx(math.exp(-8.0), rel=1e-14)
    assert p["chained"] == 5.0


# ---------------------------------------------------------------------------------------------------------------------
# The model document (beta) in the schema: `axmp schema` covers both kinds of document (design note 15.1: `if` the key
# `physics` is present, `then` the model rules, `else` today's). References: the committed output of 0.1.0's `axmp schema`
# (tests/data/design_schema_0.1.0.json) for every electromagnetic verdict; the parser for every model verdict.
# ---------------------------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA = ROOT / "cpp" / "tests" / "data" / "model"
OLD_SCHEMA = json.loads((ROOT / "tests" / "data" / "design_schema_0.1.0.json").read_text())
MODEL_NAMES = ["channel", "heated_cavity", "plate_cell", "busbar"]
NEW_SECTIONS = ["domains", "boundaries", "reactions", "physics", "couplings", "study", "outputs"]


def _model_doc(name):
    return json.loads((MODEL_DATA / f"{name}.json").read_text())


def _verdict(schema, doc):
    import jsonschema.validators as v

    return v.validator_for(schema)(schema).is_valid(doc)


def _parser_refuses_model(doc):
    """True when the model parser refuses the document (``_core.explain`` raises the first error of a document with
    errors; an unavailable route is not an error)."""
    try:
        _core.explain(json.dumps(doc), False)
    except axmp.AxionMultiPhysicsError:
        return True
    return False


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_schema_accepts_the_four_model_documents(name):
    doc = _model_doc(name)
    jsonschema.validate(doc, SCHEMA)
    assert not _parser_refuses_model(doc)


def _set(path, value):
    def f(d):
        cur = d
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value
    return f


def _drop(path):
    def f(d):
        cur = d
        for k in path[:-1]:
            cur = cur[k]
        del cur[path[-1]]
    return f


# (id, document, mutation): each is refused by the parser (checked when the list was written) and must be by the schema
MODEL_MUTANTS = [
    ("unknown_top_level_key", "channel", _set(["bogus"], 1)),
    ("em_analysis_section_in_a_model", "channel", _set(["windings"], [])),
    ("unknown_region_key", "channel", _set(["regions", 0, "colour"], "red")),
    ("unknown_instance_key", "channel", _set(["physics", "flow", "bogus"], 1)),
    ("unknown_physics_type", "channel", _set(["physics", "flow", "type"], "fluid.bogus")),
    ("unknown_option", "channel", _set(["physics", "flow", "options", "bogus_option"], 1)),
    ("option_value_not_offered", "channel", _set(["physics", "flow", "options", "viscous_form"], "bogus")),
    ("unknown_condition_kind", "channel", _set(["physics", "flow", "boundary_conditions", "outlet", "kind"], "bogus")),
    ("condition_key_of_another_kind", "channel", _set(["physics", "flow", "boundary_conditions", "walls", "pressure_Pa"], 0)),
    ("condition_missing_a_required_key", "channel", _drop(["physics", "flow", "boundary_conditions", "inlet", "velocity_m_s"])),
    ("unknown_coupling_term", "heated_cavity", _set(["couplings", 0, "term"], "bogus")),
    ("reserved_study_type", "heated_cavity", _set(["study", 0, "type"], "eigen")),
    ("unknown_output_kind", "channel", _set(["outputs", "Q_in", "kind"], "bogus")),
    ("mesh_order_is_electromagnetic", "channel", _set(["mesh", "order"], 2)),
    ("mesh_adapt_not_offered", "channel", _set(["mesh", "adapt"], {"cycles": 2})),
    ("symmetry_xsec_not_offered", "channel", _set(["symmetry"], "xsec")),
    ("unknown_boundary_key", "channel", _set(["boundaries", "inlet", "bogus"], 1)),
    ("unknown_reaction_kind", "plate_cell", _set(["reactions", "electrode", "plate", "kind"], "bogus")),
    ("unknown_material_block_key", "channel", _set(["materials", "water", "fluid", "bogus"], 1)),
]


@pytest.mark.parametrize("mid,base,mutate", MODEL_MUTANTS, ids=[m[0] for m in MODEL_MUTANTS])
def test_schema_rejects_mutants_the_parser_rejects(mid, base, mutate):
    doc = _model_doc(base)
    assert _verdict(SCHEMA, doc) and not _parser_refuses_model(doc)        # the unmutated document passes both
    mutate(doc)
    assert _parser_refuses_model(doc), "the parser must refuse the mutant (the mutant list is wrong)"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, SCHEMA)


def test_a_model_document_without_the_physics_key_is_judged_by_the_electromagnetic_rules():
    """The `if` is the presence of `physics`. Without it, the model sections are unknown fields (today's behaviour)."""
    doc = _model_doc("channel")
    del doc["physics"]
    assert not _verdict(SCHEMA, doc)
    assert not _verdict(OLD_SCHEMA, doc)


def _em_corpus_docs():
    docs = list(_valid_docs())
    for base in ("benchmarks/designs", "tests/g0/designs"):
        for f in sorted((ROOT / base).rglob("*.json")):
            try:
                d = json.loads(f.read_text())
            except ValueError:
                continue
            if isinstance(d, dict):
                docs.append(d)
    corpus = json.loads((ROOT / "tests" / "g0" / "corpus.json").read_text())["documents"]
    for c in corpus:
        try:
            d = c["doc"] if "doc" in c else json.loads(c["text"])
        except ValueError:
            continue
        if isinstance(d, dict):
            docs.append(d)
    return docs


def _em_mutants():
    out = []
    for i, (mutate, what) in enumerate(INVALID_MUTATIONS):
        d = _circuit_design()
        mutate(d)
        out.append(d)
    base = _circuit_design()
    for key, value in [("domains", {}), ("boundaries", {}), ("reactions", {}), ("couplings", []), ("study", []),
                       ("outputs", {})]:
        d = copy.deepcopy(base)
        d[key] = value
        out.append(d)            # a model section without `physics`: refused today, refused after
    d = copy.deepcopy(base)
    d["materials"] = {"m": {"density": 1.0}}        # an electromagnetic material still needs its `kind`
    out.append(d)
    d = copy.deepcopy(base)
    d["mesh"] = {"exterior": "open"}
    out.append(d)
    d = copy.deepcopy(base)
    d.pop("name")
    out.append(d)
    d = copy.deepcopy(base)
    d.pop("regions")
    out.append(d)
    return out


def test_em_accept_reject_is_unchanged():
    """Every electromagnetic document (the valid ones of this file, the examples, the G0 designs and the G0 corpus, plus
    the mutants of the tests above and of design note 15.1's `else` branch) gets the same verdict from the schema of this
    version as from the committed schema of 0.1.0."""
    docs = _em_corpus_docs() + _em_mutants()
    assert len(docs) > 60
    n_ok = n_bad = 0
    for doc in docs:
        if "physics" in doc:
            continue
        new, old = _verdict(SCHEMA, doc), _verdict(OLD_SCHEMA, doc)
        assert new == old, f"verdict changed for {doc.get('name')!r}: 0.1.0 says {old}, now {new}"
        n_ok += new
        n_bad += not new
    assert n_ok > 20 and n_bad > 15       # both verdicts are exercised


def test_every_error_of_an_em_document_is_still_an_error_at_the_same_places():
    """Stronger than the verdict for the hand-made mutants: the set of failing JSON pointers is the same, so a
    different rule is not hiding behind the same verdict."""
    import jsonschema.validators as v

    new_v, old_v = v.validator_for(SCHEMA)(SCHEMA), v.validator_for(OLD_SCHEMA)(OLD_SCHEMA)

    def places(val, doc):
        return sorted({"/" + "/".join(map(str, e.absolute_path)) for e in val.iter_errors(doc)})

    for doc in _em_mutants():
        assert places(new_v, doc) == places(old_v, doc)


def test_the_new_sections_are_marked_beta_and_the_old_ones_are_not():
    """Design note 15.1: `"x-status": "beta"` on each new section (here the seven of the model document)."""
    props = SCHEMA["properties"]
    for key in NEW_SECTIONS:
        assert key in props, key
        assert props[key].get("x-status") == "beta", key
    assert "x-status" not in props["regions"] and "x-status" not in props["windings"]


def test_the_root_keeps_properties_and_forbids_additional_ones():
    """The root keeps `properties` and `additionalProperties: false` (an `if`/`then`/`else` that adds rules,
    not a `oneOf` that replaces the root): the electromagnetic sections are root properties, the shared ones (materials with
    its thermal and mechanical blocks, mesh, symmetry) are the permissive union, and `tests/test_mesh_options.py` finds `surface_size` and `domain_center_z` where it looks."""
    assert SCHEMA.get("additionalProperties") is False
    assert "oneOf" not in SCHEMA and "anyOf" not in SCHEMA
    props = SCHEMA["properties"]
    for key in ["schema", "name", "regions", "materials", "mesh", "symmetry", "windings", "physics"]:
        assert key in props, key
    assert "if" in SCHEMA and "then" in SCHEMA and "else" in SCHEMA
    assert "surface_size" in props["regions"]["items"]["properties"]
    assert "domain_center_z" in props["mesh"]["properties"]
    # the model branch has no `kind` requirement on a material: the channel's material has none
    assert "kind" not in _model_doc("channel")["materials"]["water"]
    jsonschema.validate(_model_doc("channel"), SCHEMA)


def test_the_model_branch_is_strict_about_the_shared_sections():
    """The permissive union of the root is only the root's: inside the model branch the sections are the strict model
    versions, so a model material may not carry an electromagnetic-only key that the model parser refuses."""
    doc = _model_doc("channel")
    doc["mesh"]["domain_center_z"] = 0.0        # an electromagnetic mesh key (root union allows it, the model branch not)
    assert _parser_refuses_model(doc)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, SCHEMA)


REACTOR_ONLY = {"name": "r", "physics": {"rx": {"type": "chem.reactor"}},
                "study": [{"id": "run", "type": "transient", "time": {"t_end_s": 10}}]}


def test_a_geometry_free_model_document_needs_a_name_and_no_regions():
    """A reactor-only model document has no `regions`: the model parser accepts it, so the schema must. The required list
    is in the branches: the model branch asks for `name`, the electromagnetic branch for `name` and `regions`."""
    assert not _parser_refuses_model(REACTOR_ONLY)
    jsonschema.validate(REACTOR_ONLY, SCHEMA)
    nameless = {k: v for k, v in REACTOR_ONLY.items() if k != "name"}
    assert not _verdict(SCHEMA, nameless)
    assert "required" not in SCHEMA


def test_an_electromagnetic_document_without_regions_is_still_rejected():
    """The `regions` requirement stays with the electromagnetic rules (the 0.1.0 behaviour)."""
    doc = axmp.template("toroid").doc
    assert _verdict(SCHEMA, doc)
    del doc["regions"]
    assert not _verdict(SCHEMA, doc)
    assert not _verdict(OLD_SCHEMA, doc)
