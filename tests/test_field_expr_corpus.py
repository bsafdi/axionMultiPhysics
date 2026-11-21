"""The committed expression corpus is exactly what its generator produces from this tree.

cpp/tests/test_field_expr_corpus.cpp compares the two expression evaluators on every string that could be an
expression in examples/, in benchmarks/designs/ and in the templates.  That list is generated, not written:
this test regenerates it and refuses a committed file that differs, so that the corpus cannot be trimmed or
edited by hand, and so that an example or a template that gains an expression brings it into the corpus.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = os.path.join(ROOT, "cpp", "tests", "data", "make_field_expr_corpus.py")
CORPUS = os.path.join(ROOT, "cpp", "tests", "data", "field_expr_corpus.json")


def test_corpus_is_what_the_generator_produces(tmp_path):
    produced = tmp_path / "field_expr_corpus.json"
    run = subprocess.run([sys.executable, GENERATOR, "--root", ROOT, "--out", str(produced)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert produced.read_text(encoding="utf-8") == open(CORPUS, encoding="utf-8").read(), (
        "cpp/tests/data/field_expr_corpus.json is not what cpp/tests/data/make_field_expr_corpus.py "
        "produces from this tree. Regenerate it with\n"
        "  PYTHONPATH=python python cpp/tests/data/make_field_expr_corpus.py\n"
        "and review what moved; do not edit it by hand.")


def test_corpus_covers_its_three_roots_and_counts_itself():
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    documents = corpus["documents"]
    assert corpus["counts"]["documents"] == len(documents)
    assert corpus["counts"]["strings"] == sum(len(d["strings"]) for d in documents)
    sources = [d["source"] for d in documents]
    assert any(s.startswith("templates:") for s in sources)
    assert any(s.startswith("examples/") for s in sources)
    assert any(s.startswith("benchmarks/designs/") for s in sources)
    assert len(sources) == len(set(sources)), "a source appears twice"
    for d in documents:
        kind = d["params"]["kind"]
        assert kind in ("definitions", "values")
        if kind == "values":
            # a multiple of 1/64, so that the number in the file is the number the test reads
            for name, value in d["params"]["values"].items():
                assert (value * 64.0).is_integer(), (d["source"], name, value)
                assert name not in ("pi", "e", "mu0")
        else:
            for name, value in d["params"]["defs"].items():
                assert isinstance(value, (int, float, str)) and not isinstance(value, bool)
        for entry in d["strings"]:
            assert entry["text"] and len(entry["text"]) <= 200
            assert entry["occurrences"] >= 1
