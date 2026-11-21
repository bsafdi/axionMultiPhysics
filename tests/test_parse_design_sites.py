"""The call sites of ``parse_design`` against the committed manifest ``tests/data/parse_design_sites.json`` (design note
3.2 item 2): a site that appears or disappears without the manifest (and the table of 3.2) changing fails here, and
every electromagnetic-only site calls ``require_em_document`` before it parses, every served site tests
``is_model_document``, and every site of an operation that takes a model document in a later version refuses it first.

How a site is found: a call ``parse_design(`` outside a comment, in the files under ``cpp/src``, ``cpp/apps`` and
``cpp/python`` (the scope of the table) and under ``cpp/tests`` (listed apart, outside the table); the definition and
the declaration are not calls. How its function is named: in ``cpp/apps/axmp.cpp`` by the command branch
(``if (cmd == "validate")`` -> ``axmp validate``); in ``cpp/python/bindings.cpp`` by the binding (``m.def("validate"``
-> ``_core.validate``; ``.def("__init__"`` of ``nb::class_<...>(m, "Model3D")`` -> ``_core.Model3D.__init__``); in the
library by the enclosing function definition (a line at column 0 that ends with ``{``). The checks of order look at
the text of the function from its start (the anchor above) to the site.

Independent reference: the table of design note 3.2 as read on this tree (development plan section 6 items 4 and 5,
which correct the note's line numbers and three misread sites). Pure Python; reads the source only.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "tests" / "data" / "parse_design_sites.json").read_text("utf-8"))

CALL = re.compile(r"\bparse_design\s*\(")
DEFINITION = re.compile(r"^\s*(?:axmp::)?(?:spec::)?Design\s+$")
CMD = re.compile(r'if\s*\(\s*cmd\s*==\s*"([\w-]+)"\s*\)')
MDEF = re.compile(r'\bm\.def\(\s*"(\w+)"')
CLASS = re.compile(r'nb::class_<[^>]+>\(\s*m\s*,\s*"(\w+)"')
METHOD = re.compile(r'\.def\(\s*"(\w+)"')
FUNC = re.compile(r"^[A-Za-z][^\n;]*?\b([A-Za-z_][\w:]*)\s*\([^\n]*\{\s*$", re.M)


def _mask_comments(text: str) -> str:
    """The text with every // comment and /* */ comment replaced by spaces (positions and lines kept); string
    literals are kept as they are (none of the scanned files has // or /* inside a literal before a call)."""
    out = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), out)


def _anchors(rel: str, text: str) -> list[tuple[int, str]]:
    """(position, function label) of every place where a function, a command branch or a binding starts."""
    out: list[tuple[int, str]] = []
    if rel == "cpp/apps/axmp.cpp":
        out += [(m.start(), f"axmp {m.group(1)}") for m in CMD.finditer(text)]
        out += [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
    elif rel.endswith("bindings.cpp"):
        events = sorted([(m.start(), "def", m.group(1)) for m in MDEF.finditer(text)] +
                        [(m.start(), "class", m.group(1)) for m in CLASS.finditer(text)] +
                        [(m.start(), "method", m.group(1)) for m in METHOD.finditer(text)])
        cls = None
        for pos, kind, name in events:
            if kind == "def":
                cls = None
                out.append((pos, f"_core.{name}"))
            elif kind == "class":
                cls = name
                out.append((pos, f"_core.{name}"))
            elif cls is not None:
                out.append((pos, f"_core.{cls}.{name}"))
        out += [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
    else:
        out += [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
    return sorted(out)


def _sites(rel: str) -> list[tuple[str, int, str]]:
    """(function label, position, text from the function's start to the call) of every call in one file."""
    raw = (ROOT / rel).read_text("utf-8")
    text = _mask_comments(raw)
    anchors = _anchors(rel, text)
    out = []
    for m in CALL.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        if DEFINITION.match(text[line_start:m.start()]):
            continue  # the definition (spec.cpp) or a declaration
        before = [a for a in anchors if a[0] <= m.start()]
        label, start = (before[-1][1], before[-1][0]) if before else ("<file>", 0)
        out.append((label, m.start(), text[start:m.start()]))
    return out


def _scope_files(dirs) -> list[str]:
    files = []
    for d in dirs:
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix in (".cpp", ".hpp", ".h", ".cc") and p.is_file():
                files.append(p.relative_to(ROOT).as_posix())
    return files


def test_the_scanner_finds_what_it_should():
    """The scanner itself, on a text written here: comments are not calls, the definition is not a call, and each
    labelling rule names its function."""
    sample = ('json validate(const json& d) {\n  // parse_design(d) in a comment\n  /* parse_design(x) */\n'
              '  return spec::parse_design(d);\n}\n')
    assert "parse_design(d) in" not in _mask_comments(sample).replace("  ", "")
    assert len(list(CALL.finditer(_mask_comments(sample)))) == 1
    assert [a[1] for a in _anchors("cpp/src/x.cpp", sample)] == ["validate"]
    assert DEFINITION.match("Design ") and not DEFINITION.match("  return ")


def test_call_sites_equal_the_manifest():
    """The sites under cpp/src, cpp/apps and cpp/python equal the manifest's, function by function and count by count;
    the C++ test files that call parse_design are exactly the manifest's list of tests."""
    assert MANIFEST["scope"] == ["cpp/src", "cpp/apps", "cpp/python"]
    found: Counter = Counter()
    for rel in _scope_files(MANIFEST["scope"]):
        for label, _, _ in _sites(rel):
            found[(rel, label)] += 1
    want = Counter({(s["file"], s["function"]): s["count"] for s in MANIFEST["sites"]})
    assert len(want) == len(MANIFEST["sites"]), "a file: function pair is listed twice in the manifest"
    missing = {k: v for k, v in want.items() if found.get(k) != v}
    extra = {k: v for k, v in found.items() if k not in want}
    assert not missing and not extra, (
        f"call sites of parse_design differ from tests/data/parse_design_sites.json\n"
        f"  in the manifest, not (or not that often) in the source: {missing}\n"
        f"    the source has: { {k: found.get(k, 0) for k in missing} }\n"
        f"  in the source, not in the manifest: {extra}\n"
        "A new or removed site changes the manifest and the table of design note 3.2 together.")
    tests = sorted(rel for rel in _scope_files(["cpp/tests"]) if _sites(rel))
    assert tests == sorted(MANIFEST["tests"]), f"C++ test files that call parse_design: {tests}"


def _function_text(rel: str, label: str) -> str:
    """The text of a function or binding without a parse_design site: from its anchor to the next anchor."""
    text = _mask_comments((ROOT / rel).read_text("utf-8"))
    anchors = _anchors(rel, text)
    for i, (pos, name) in enumerate(anchors):
        if name == label:
            end = anchors[i + 1][0] if i + 1 < len(anchors) else len(text)
            return text[pos:end]
    raise AssertionError(f"{rel}: no function {label!r}")


def test_em_only_sites_call_require_em_document():
    """In every function the manifest marks em_only, ``require_em_document(`` comes before each ``parse_design(``;
    in every served one, ``is_model_document(``; in every unserved one, ``refuse_unserved_model_document(``; and the
    same in the entry points that have no site of their own (meshing, served, and the 3D run, unserved), before the
    call of the library function they hand an electromagnetic design to. The other roles (synthetic, behind_entry,
    backstop, em_internal, test) are stated in the manifest and need no call."""
    needs = {"em_only": "require_em_document(", "served": "is_model_document(",
             "unserved": "refuse_unserved_model_document("}
    roles = {(s["file"], s["function"]): s["role"] for s in MANIFEST["sites"]}
    assert set(roles.values()) <= set(needs) | {"synthetic", "behind_entry", "backstop", "em_internal", "test"}
    bad = []
    for (rel, label), role in sorted(roles.items()):
        if role not in needs:
            continue
        for lab, _, before in _sites(rel):
            if lab == label and needs[role] not in before:
                bad.append(f"{rel}: {label} ({role}) parses before calling {needs[role]}")
    for e in MANIFEST["entry_points_without_a_site"]:
        body = _function_text(e["file"], e["function"])
        call = re.search(r"\b" + re.escape(e["calls"]) + r"\s*\(", body)
        assert call, f"{e['file']}: {e['function']} no longer calls {e['calls']}"
        if needs[e["role"]] not in body[:call.start()]:
            bad.append(f"{e['file']}: {e['function']} ({e['role']}) calls {e['calls']} before {needs[e['role']]}")
    assert not bad, "\n".join(bad)


def test_every_em_only_row_of_the_table_has_its_sites():
    """The rows of the table of design note 3.2 that name C++ sites, as read on this tree (plan section 6 item 4):
    the four model constructors and the Model3D restart, sensitivities and the adjoint, the synthetic helpers, the
    served validate, resolve and patch, and solving."""
    by_role: dict[str, set] = {}
    for s in MANIFEST["sites"]:
        by_role.setdefault(s["role"], set()).add(s["function"])
    assert {"_core.Model3D.__init__", "_core.XsecModel.__init__", "_core.Fw2DModel.__init__", "_core.Model.__init__",
            "axmp sensitivities"} <= by_role["em_only"]
    assert by_role["served"] == {"axmp validate", "axmp resolve", "_core.validate", "_core.resolve", "patch_document"}
    assert by_role["synthetic"] == {"_core.conductor_jc", "_core.surface_resistance", "_core.requirement"}
    assert by_role["unserved"] == {"axmp solve"}
    assert {"Model3D::load", "run_design"} <= by_role["backstop"]
    assert by_role["behind_entry"] == {"mesh_any", "run_any"}
    assert by_role["em_internal"] == {"perturbed"}
    assert by_role["test"] == {"_core._parse_design_for_tests"}
