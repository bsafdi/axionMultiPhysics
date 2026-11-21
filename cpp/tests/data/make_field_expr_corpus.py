"""Collect the expression corpus that cpp/tests/test_field_expr_corpus.cpp compares two evaluators on.

The corpus is every string that could be an expression of the design grammar
(cpp/include/axionmultiphysics/expr.hpp) in the three places a user meets one:

  examples/            every .json, .ipynb (code cells) and .py file, recursively
  benchmarks/designs/  every .json file, recursively
  the templates        every name of _core.templates(), instantiated with its defaults

No judgement is applied about which of those strings is "really" an expression: a string is collected when it
could possibly be one, that is when every character of it is in the grammar's alphabet (digits, letters,
'_', '.', parentheses, ',', the operators, whitespace) and it is at most 200 characters long.  A string
outside that alphabet cannot be parsed by the grammar at all, so nothing that any document evaluates is
dropped; everything else is kept and the test requires the two evaluators to agree on it, whether they
evaluate it or refuse it.  That makes the collection reproducible and free of hand-picking: the test, not
this script, decides which strings evaluate, by running the existing evaluator on them.

Each collected string is stored with the parameter map it is to be evaluated against:

  kind "definitions": the "parameters" section of the document it came from, verbatim (numbers as numbers,
      expressions as strings).  The test resolves it exactly as the design parser does, so a string from a
      design document is evaluated against the very map that design gives it.  No number computed by a
      library function is written here, so the file does not depend on the platform's libm.
  kind "values": a probe map, for strings taken from Python sources, where the document that would hold them
      is built at run time and no parameter section exists in the file.  The names are those the file itself
      declares (keys of a "parameters" object, names passed to set_parameter, keyword arguments of a
      template call) together with every template parameter name, since an expression in a notebook refers
      to the parameters of the template it edits.  'pi', 'e' and 'mu0' are left out so that they keep
      meaning the grammar's constants.  The value of a name is a fixed multiple of 1/64 derived from the
      name alone: exactly representable, so the file round-trips through JSON bit for bit, stable when
      another name appears, and positive, so that expressions stay mostly finite.  Values are probes, not
      physics: what the test asserts of them is that two evaluators agree, which does not depend on the
      numbers being physical.

Run it from the repository root, with a built package on PYTHONPATH (the templates come from the library):

  PYTHONPATH=python python cpp/tests/data/make_field_expr_corpus.py

It writes cpp/tests/data/field_expr_corpus.json and prints a summary.  tests/test_field_expr_corpus.py
re-runs it and fails when the committed file is not what it produces.
"""

import argparse
import ast
import hashlib
import json
import os
import sys

# The alphabet of the grammar: a string with any other character cannot be an expression.
ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_.()+-*/^, \t\r\n")
MAX_LEN = 200
# Names the grammar resolves by itself; a probe value for one of them would shadow the constant.
CONSTANTS = ("pi", "e", "mu0")
JSON_ROOTS = ("examples", "benchmarks/designs")


def is_candidate(s):
    return isinstance(s, str) and 0 < len(s) <= MAX_LEN and set(s) <= ALPHABET


def probe_value(name):
    """A fixed value for a probe name: a multiple of 1/64 in [0.125, 8.125), from the name alone."""
    h = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    return (8 + h % 512) / 64.0


def json_strings(node, pointer=""):
    """Every string leaf of a JSON value, with its RFC 6901 pointer."""
    if isinstance(node, dict):
        for key, value in node.items():
            token = key.replace("~", "~0").replace("/", "~1")
            yield from json_strings(value, pointer + "/" + token)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from json_strings(value, pointer + "/" + str(i))
    elif isinstance(node, str):
        yield pointer, node


def collect(pairs):
    """Order (location, text) pairs into corpus entries: one entry per distinct text, first location kept."""
    seen = {}
    for where, text in pairs:
        if not is_candidate(text):
            continue
        if text in seen:
            seen[text]["occurrences"] += 1
        else:
            seen[text] = {"at": where, "text": text, "occurrences": 1}
    return [seen[t] for t in sorted(seen)]


def parameters_section(doc):
    """The document's "parameters" section, if it is the object the parser accepts."""
    if not isinstance(doc, dict):
        return {}
    p = doc.get("parameters")
    if not isinstance(p, dict):
        return {}
    return {k: v for k, v in sorted(p.items()) if isinstance(v, (int, float, str)) and not isinstance(v, bool)}


def python_declared_names(tree):
    """Parameter names a Python source declares: keys of a "parameters" object, set_parameter(...) names,
    keyword arguments of a template call."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "parameters"
                    and isinstance(value, ast.Dict)
                ):
                    for k in value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            names.add(k.value)
        if isinstance(node, ast.Call):
            fn = node.func
            fn_name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fn_name == "set_parameter" and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    names.add(a0.value)
            if fn_name in ("template", "from_template"):
                for kw in node.keywords:
                    if kw.arg:
                        names.add(kw.arg)
            for kw in node.keywords:
                if kw.arg == "parameters" and isinstance(kw.value, ast.Dict):
                    for k in kw.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            names.add(k.value)
    return names


def python_literals(tree, prefix=""):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield "%sline%d col%d" % (prefix, node.lineno, node.col_offset), node.value


def template_documents():
    from axionmultiphysics import _core

    listing = json.loads(_core.templates())
    names = set()
    docs = []
    for entry in sorted(listing, key=lambda e: e["name"]):
        for pname in entry.get("parameters", {}):
            names.add(pname)
        docs.append((entry["name"], json.loads(_core.template(entry["name"], "{}"))))
    return docs, names


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..", "..")),
                    help="repository root to scan")
    ap.add_argument("--out", default=os.path.join(here, "field_expr_corpus.json"),
                    help="output file ('-' for standard output)")
    args = ap.parse_args()
    root = args.root

    template_docs, template_param_names = template_documents()
    documents = []
    skipped = []

    for name, doc in template_docs:
        documents.append({
            "source": "templates:" + name,
            "kind": "template",
            "params": {"kind": "definitions", "defs": parameters_section(doc)},
            "strings": collect(json_strings(doc)),
        })

    for top in JSON_ROOTS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, top)):
            dirnames.sort()
            for filename in sorted(filenames):
                path = os.path.join(dirpath, filename)
                rel = os.path.relpath(path, root)
                ext = os.path.splitext(filename)[1]
                if ext == ".json":
                    with open(path, encoding="utf-8") as fh:
                        doc = json.load(fh)
                    documents.append({
                        "source": rel,
                        "kind": "json",
                        "params": {"kind": "definitions", "defs": parameters_section(doc)},
                        "strings": collect(json_strings(doc)),
                    })
                elif ext in (".py", ".ipynb"):
                    if ext == ".py":
                        with open(path, encoding="utf-8") as fh:
                            sources = [("", fh.read())]
                    else:
                        with open(path, encoding="utf-8") as fh:
                            nb = json.load(fh)
                        sources = [
                            ("cell%d " % i, "".join(c["source"]))
                            for i, c in enumerate(c for c in nb["cells"] if c["cell_type"] == "code")
                        ]
                    pairs = []
                    declared = set()
                    for prefix, text in sources:
                        try:
                            tree = ast.parse(text)
                        except SyntaxError as exc:
                            skipped.append({"path": rel, "reason": "does not parse as Python: %s" % exc.msg})
                            continue
                        declared |= python_declared_names(tree)
                        pairs += list(python_literals(tree, prefix))
                    strings = collect(pairs)
                    used = set()
                    for entry in strings:
                        used |= set(identifiers(entry["text"]))
                    values = {
                        n: probe_value(n)
                        for n in sorted((declared | template_param_names) & used)
                        if n not in CONSTANTS
                    }
                    documents.append({
                        "source": rel,
                        "kind": "notebook" if ext == ".ipynb" else "python",
                        "params": {"kind": "values", "values": values},
                        "strings": strings,
                    })
                else:
                    skipped.append({"path": rel, "reason": "not a JSON, Python or notebook file"})

    documents.sort(key=lambda d: d["source"])
    skipped.sort(key=lambda s: (s["path"], s["reason"]))
    out = {
        "comment": ("Generated by cpp/tests/data/make_field_expr_corpus.py; do not edit by hand. "
                    "Read by cpp/tests/test_field_expr_corpus.cpp."),
        "roots": list(JSON_ROOTS) + ["templates"],
        "counts": {
            "documents": len(documents),
            "strings": sum(len(d["strings"]) for d in documents),
        },
        "documents": documents,
        "skipped": skipped,
    }
    text = json.dumps(out, indent=1, sort_keys=False, ensure_ascii=True) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    sys.stderr.write("%d documents, %d strings, %d files not scanned -> %s\n"
                     % (out["counts"]["documents"], out["counts"]["strings"], len(skipped), args.out))


def identifiers(text):
    """The identifiers of a candidate string, as the grammar's tokenizer would see them: a name that is not
    followed by '(' (a name followed by '(' is a function call, which the grammar resolves by itself)."""
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            k = j
            while k < n and text[k] in " \t\r\n":
                k += 1
            if not (k < n and text[k] == "("):
                yield text[i:j]
            i = j
        else:
            i += 1


if __name__ == "__main__":
    main()
