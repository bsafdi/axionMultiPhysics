"""The texts of the switch between the two parsers, shared by the tests of model documents at the entry points
(``test_model_guard.py``, ``test_model_cli.py``, ``test_json_duplicate_key.py``; ``cpp/tests/test_model_switch.cpp`` has
the same texts in C++).

Sources. ``model.section_needs_physics`` (message, pointer, hint) and the message and pointer of
``model.em_only_operation`` are copied verbatim from design note section 3.2. ``details.serves_model_documents`` is the
list of the operations of this version that take a model document (``validate``, ``resolve``, ``explain``, ``patch``,
``boundaries`` and ``mesh``; it grows with each milestone as a declared text change). Where the note gives no text,
these are the specification: the hints of ``model.em_only_operation`` (the note's first sentence for the adjoint, the
plan's N17 sentence for the optimizers, and a generic one), the generic message of the backstop in ``parse_design``
(plan N1), the refusal of solving (``route.not_implemented``, plan N2; no milestone is named, as the route texts do not
name one), and ``json.duplicate_key`` (design note 15.2 row 15, plan N14).

Not a test file: pytest collects nothing here.
"""
from __future__ import annotations

SERVES = ["validate", "resolve", "explain", "patch", "boundaries", "mesh"]
TAIL = "In this version a model document is taken by: validate, resolve, explain, patch, boundaries, mesh."

SECTIONS = ["domains", "boundaries", "reactions", "couplings", "study", "outputs"]

ADJOINT_OPERATIONS = {"sensitivities", "gradient"}
OPTIMIZER_OPERATIONS = {"optimize", "global_optimize", "optimize_design", "optimize_global"}


def section_needs_physics(key: str) -> dict:
    """Design note 3.2 item 1, verbatim; no details."""
    return {"error": f"'{key}' belongs to a model document, and this document has no 'physics' section",
            "where": f"/{key}",
            "hint": f"add \"physics\": {{...}} (run `axmp physics` for the available types), or remove '{key}'.",
            "code": "model.section_needs_physics"}


def em_only_hint(operation: str | None) -> str:
    if operation in ADJOINT_OPERATIONS:
        return "adjoint sensitivities are an electromagnetic capability. " + TAIL
    if operation in OPTIMIZER_OPERATIONS:
        return "this version optimizes electromagnetic designs. " + TAIL
    return "this operation needs an electromagnetic design (a document without 'physics'). " + TAIL


def em_only(operation: str | None, name: str | None = "channel") -> dict:
    """``model.em_only_operation``: the message of design note 3.2 for a named operation; the backstop's generic
    message when ``operation`` is None (details.operation is null there)."""
    who = f"'{operation}'" if operation is not None else "this operation"
    doc = f"'{name}' is a model document" if name is not None else "this document is a model document"
    return {"error": f"{who} works on electromagnetic designs, and {doc} (it has a 'physics' section)",
            "where": "/physics",
            "hint": em_only_hint(operation),
            "code": "model.em_only_operation",
            "details": {"operation": operation, "serves_model_documents": SERVES}}


NOT_IMPLEMENTED_HINT = ("solving a model document is not implemented in this version; `axmp validate` reports what "
                        "the document needs. " + TAIL)


def not_implemented(operation: str, name: str | None = "channel") -> dict:
    """``route.not_implemented`` of an operation that will take a model document in a later version."""
    who = f"'{name}' has a 'physics' section" if name is not None else "the document has a 'physics' section"
    return {"error": f"'{operation}' of a model document is not implemented yet ({who})",
            "where": "/physics",
            "hint": NOT_IMPLEMENTED_HINT,
            "code": "route.not_implemented",
            "details": {"operation": operation, "implemented": False, "serves_model_documents": SERVES}}


DUPLICATE_KEY_HINT = ("a model document may not repeat a key in an object: JSON keeps only the last value, silently. "
                      "Keep one of the two.")


def duplicate_key(key: str, pointer: str, origin: str) -> dict:
    """``json.duplicate_key`` at the RFC 6901 pointer of the second occurrence; ``origin`` is the path as given."""
    return {"error": f"duplicate key '{key}' in '{origin}'", "where": pointer, "hint": DUPLICATE_KEY_HINT,
            "code": "json.duplicate_key", "details": {"key": key}}


def composed(expected: dict) -> str:
    """``str(e)``: compose() of core.hpp, the message, then `` [at <where>]`` and `` Hint: <hint>`` when not empty."""
    s = expected["error"]
    if expected.get("where"):
        s += f" [at {expected['where']}]"
    if expected.get("hint"):
        s += f" Hint: {expected['hint']}"
    return s


def cli_json(expected: dict) -> dict:
    """The command line's error JSON (``error_json``): ``ok`` false, then the fields; ``details`` only when given."""
    out = {"ok": False, **{k: v for k, v in expected.items() if k != "details"}}
    if expected.get("details") is not None:
        out["details"] = expected["details"]
    return out


def assert_exception(e: BaseException, expected: dict) -> None:
    """The Python exception of an expected error: the class, ``str``, ``args`` and the four fields."""
    from axionmultiphysics import _core

    assert type(e) is _core.AxionMultiPhysicsError, f"{type(e).__name__}: {e}"
    assert str(e) == composed(expected), f"\n got  {e}\n want {composed(expected)}"
    assert e.args == (composed(expected),)
    assert e.code == expected["code"]
    assert e.where == expected["where"]
    assert e.hint == expected["hint"]
    assert e.details == expected.get("details"), f"details {e.details!r}"


def assert_payload(p: dict, expected: dict) -> None:
    """The MCP server's error payload (``error_payload``) of an expected error."""
    want = {"ok": False, "error": composed(expected), "code": expected["code"], "where": expected["where"],
            "hint": expected["hint"]}
    if expected.get("details") is not None:
        want["details"] = expected["details"]
    assert p == want, f"\n got  {p}\n want {want}"
