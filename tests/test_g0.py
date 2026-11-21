"""G0, the compatibility gate (docs/developer/g0.md): the harness's own rules, and tier A against this platform's baseline.

The rules need no compiled core. Tier A runs when the build is there and a baseline was recorded for its platform;
tiers B and C (a few solves) run with AXMP_SLOW=1, as in the pre-merge subset.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "tests" / "g0"
_spec = importlib.util.spec_from_file_location("g0_harness", G0 / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
sys.modules["g0_harness"] = g0
_spec.loader.exec_module(g0)


def _built():
    """The harness runs the `axmp` executable (AXMP_BIN or AXMP_EXE, the package's own search, build-mpi/, build/) and
    imports the package's core."""
    try:
        sys.path.insert(0, str(ROOT / "python"))
        import axionmultiphysics as axmp

        return (hasattr(axmp, "_core") or bool(axmp.build_info())) and g0.find_axmp() is not None
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------- rules (pure Python)
def test_corpus_has_no_reserved_top_level_key():
    """Before the model document these documents fail with 'unknown field'; after it, with a dedicated refusal. A
    corpus document whose error is such a key would change on purpose, so none is kept (docs/developer/g0.md)."""
    docs = g0.load_corpus()
    assert len(docs) >= 30
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids))
    for d in docs:
        assert d.get("why"), d["id"]
        assert ("doc" in d) != ("text" in d), d["id"]
        if "doc" in d and isinstance(d["doc"], dict):
            assert not set(d["doc"]) & set(g0.RESERVED), d["id"]
        if "text" in d:
            try:
                obj = json.loads(d["text"])
            except ValueError:
                continue
            if isinstance(obj, dict):
                assert not set(obj) & set(g0.RESERVED), d["id"]


def test_manifest_is_complete():
    """Tier A covers every design under examples/, benchmarks/designs/ and tests/g0/designs/ and every template; tier B
    entries exist."""
    if not _built():
        pytest.skip("needs the built package (template list)")
    m = g0.load_manifest()
    found = g0.discover_designs()
    paths = {d["path"] for d in m["tier_a"]["designs"] if "path" in d}
    templates = {d["template"] for d in m["tier_a"]["designs"] if "template" in d}
    assert paths == set(found["files"])
    assert templates == set(found["templates"])
    ids = [d["id"] for d in m["tier_a"]["designs"]]
    assert len(ids) == len(set(ids))
    tb = [e["id"] for e in m["tier_b"]]
    assert tb and len(tb) == len(set(tb))
    for e in m["tier_b"]:
        if "design" in e:
            assert (ROOT / e["design"]).is_file(), e["id"]


def test_mask_pointer_patterns():
    assert g0._pattern("/a/3/b/12") == "/a/*/b/*"
    assert g0._matches("/a/*/b", "/a/7/b") and not g0._matches("/a/*/b", "/a/x/b")
    doc = {"t": 1.0, "xs": [{"t": 2.0, "v": 5}, {"t": 3.0, "v": 6}], "v": 1}
    assert g0.apply_mask(doc, ["/t", "/xs/*/t"]) == {"xs": [{"v": 5}, {"v": 6}], "v": 1}


def _run_of(doc=None, text=None, name="results.json", extra_files=()):
    """An observation as `run_entry` returns it, of a run that wrote one JSON file (given as a document or as text)."""
    text = json.dumps(doc, indent=2) if text is None else text
    return {"status": 0, "stdout": "", "file_set": sorted([name, *extra_files]),
            "files": {name: {"kind": "json", "doc": json.loads(text), "text": text, "sha256": g0._sha(text)}}}


def test_a_changed_number_is_found_by_tier_b_and_tier_c():
    base = {"coupling": {"Psi": 1.0, "L_p": 2.0}, "seconds": 4.2}
    obs_doc = {"coupling": {"Psi": 1.0 + 3e-7, "L_p": 2.0}, "seconds": 9.9}
    pats = {"results.json": ["/seconds"]}
    rec = g0.baseline_entry(_run_of(base), pats)
    obs = _run_of(obs_doc)
    recorded = {"figures": g0.figures(base, pats["results.json"]), "discrete": g0.discrete(base, pats["results.json"])}
    assert any("results.json differs" in d and "/coupling/Psi" in d
               for d in g0.tier_b_compare("x", rec, obs, pats, recorded_leaves=recorded))
    assert any("results.json differs" in d for d in g0.tier_b_compare("x", rec, obs, pats))
    obs_same = _run_of({"coupling": {"Psi": 1.0, "L_p": 2.0}, "seconds": 9.9})
    assert g0.tier_b_compare("x", rec, obs_same, pats) == []
    ref = g0.figures(base, pats["results.json"])
    tol = {"/coupling/Psi": 1e-9, "/coupling/L_p": 1e-9}
    assert g0.tier_c_compare("x", ref, obs, pats["results.json"], tol)[0]  # 3e-7 > 1e-9
    assert not g0.tier_c_compare("x", ref, obs, pats["results.json"], dict(tol, **{"/coupling/Psi": 1e-6}))[0]


def test_tier_c_compares_each_figure_to_its_own_value():
    """A large figure in the same file (the 1e30 that stands for "no margin") must not widen the others' tolerance."""
    ref = {"/coupling/g_factor": 0.0763, "/coupling/L_p_H": 3.356e-7, "/benchmarks/0/violation": 1.25e30}

    def obs(**changed):
        doc = {"coupling": {"g_factor": ref["/coupling/g_factor"], "L_p_H": ref["/coupling/L_p_H"]},
               "benchmarks": [{"violation": 1.25e30}]}
        for k, v in changed.items():
            doc["coupling"][k] = v
        return {"status": 0, "stdout": "", "files": {"results.json": {"kind": "json", "doc": doc}}}

    tol = {"/coupling/g_factor": 1e-9, "/coupling/L_p_H": 1e-9, "/benchmarks/0/violation": 1e-9}
    assert not g0.tier_c_compare("x", ref, obs(), [], tol)[0]
    for name in ("g_factor", "L_p_H"):
        assert g0.tier_c_compare("x", ref, obs(**{name: ref["/coupling/" + name] * 1.01}), [], tol)[0], name
        assert g0.tier_c_compare("x", ref, obs(**{name: ref["/coupling/" + name] * (1 + 1e-6)}), [], tol)[0], name


def _obs(doc):
    return {"status": 0, "stdout": "", "files": {"results.json": {"kind": "json", "doc": doc}}}


def test_tier_c_compares_what_is_not_a_number_exactly():
    """A verdict of a physics check, a string and the shape of the document need no tolerance: tier C compares them
    exactly, outside the leaves measured to differ on the platform."""
    doc = {"checks": [{"name": "flux vs energy", "valid": True, "value": 1.0}], "mesher": {"gmsh_version": "4.15.2-nox"},
           "warnings": ["peak field 3.44 T"], "profile": [0.0, 1.0, 2.0], "margin": None}
    ref_d, ref_l = g0.discrete(doc), g0.layout(doc)
    assert ref_d == {"/checks/0/name": "flux vs energy", "/checks/0/valid": True, "/mesher/gmsh_version": "4.15.2-nox",
                     "/warnings/0": "peak field 3.44 T", "/margin": None}

    def exact(changed, **kw):
        return g0.tier_c_exact("x", ref_d, ref_l, changed, [], **kw)[0]

    import copy

    assert exact(copy.deepcopy(doc)) == []
    flipped = copy.deepcopy(doc)
    flipped["checks"][0]["valid"] = False
    assert any("/checks/0/valid" in d for d in exact(flipped))
    one = copy.deepcopy(doc)
    one["checks"][0]["valid"] = 1  # a number where the verdict was: not equal to True here
    assert exact(one)
    renamed = copy.deepcopy(doc)
    renamed["checks"][0]["name"] = "MUTATED"
    assert any("/checks/0/name" in d for d in exact(renamed))
    grown = copy.deepcopy(doc)
    grown["brand_new_section"] = {"x": 1.0}
    assert any("keys" in d for d in exact(grown))
    longer = copy.deepcopy(doc)
    longer["profile"].append(3.0)
    assert any("/profile" in d for d in exact(longer))
    reordered = {k: doc[k] for k in reversed(list(doc))}
    assert any("order" in d for d in exact(reordered))
    gone = copy.deepcopy(doc)
    gone["margin"] = 0.5  # a null that became a number
    assert any("/margin" in d for d in exact(gone))
    # a leaf measured to differ on the platform is passed over, and only that leaf
    other = copy.deepcopy(doc)
    other["mesher"]["gmsh_version"] = "4.15.2"
    assert exact(other) and exact(other, unasserted_discrete={"/mesher/gmsh_version": ["4.15.2-nox", "4.15.2"]}) == []
    assert exact(flipped, unasserted_discrete={"/mesher/gmsh_version": ["4.15.2-nox", "4.15.2"]})
    assert exact(longer, unasserted_layout=["/profile"]) == []
    assert exact(grown, unasserted_layout=["/profile"])
    # the timing mask of the reference platform applies before the comparison
    timed = dict(copy.deepcopy(doc), seconds=float("nan"))
    assert "/seconds" in g0.discrete(timed) and "/seconds" not in g0.discrete(timed, ["/seconds"])


def test_layout_difference_names_the_pointers():
    a = g0.layout({"a": 1, "xs": [1, 2], "b": {"c": 1}})
    assert g0.layout_difference(a, a) == []
    assert g0.layout_difference(a, g0.layout({"a": 1, "xs": [1, 2, 3], "b": {"c": 1}})) == ["/xs"]
    assert g0.layout_difference(a, g0.layout({"a": 1, "xs": [1, 2], "b": {"c": 1, "d": 2}})) == ["/b/d"]
    assert g0.layout_difference(a, g0.layout({"xs": [1, 2], "a": 1, "b": {"c": 1}})) == ["*"]


def test_mesh_sized_quantities_are_not_asserted():
    assert g0.mesh_sized("/mesh/elements") and g0.mesh_sized("/static/poloidal/solver/nonzeros_L")
    assert g0.mesh_sized("/forces/distributions/lower/cells") and g0.mesh_sized("/mesh/info/mesher/elements_2d")
    assert not g0.mesh_sized("/coupling/g_factor") and not g0.mesh_sized("/mesh/info/settings/order")
    ref = {"/mesh/elements": 100.0, "/coupling/g": 1.0}
    doc = {"mesh": {"elements": 150}, "coupling": {"g": 1.0}}
    obs = {"status": 0, "stdout": "", "files": {"results.json": {"kind": "json", "doc": doc}}}
    diffs, asserted, left_out, unmeasured = g0.tier_c_compare("x", ref, obs, [], {"/coupling/g": 1e-10})
    assert not diffs and (asserted, left_out, unmeasured) == (1, 1, 0)


def test_tier_c_unasserted_list_is_honoured_and_a_zero_reference_is_exact():
    ref = {"/a": 1.0, "/z": 0.0}
    doc = {"a": 2.0, "z": 1e-30}
    obs = {"status": 0, "stdout": "", "files": {"results.json": {"kind": "json", "doc": doc}}}
    tol = {"/a": 1e-10, "/z": 1e-10}
    assert len(g0.tier_c_compare("x", ref, obs, [], tol)[0]) == 2
    assert not g0.tier_c_compare("x", ref, obs, [], tol, unasserted={"/a": [3, "p"], "/z": [3, "p"]})[0]


def test_a_figure_without_a_measured_spread_here_is_not_asserted_at_the_floor():
    """A figure the platform's measurement has no spread for has no tolerance, so it is reported as unmeasured, not
    compared at the floor: that is the strictest assertion the harness can make, and it would be an invented number,
    the same reason a whole entry without a measurement is pending. The case is live the first time a figure is added
    to a solve that a platform already has a record of."""
    ref = {"/old": 1.0, "/added_since": 2.0}
    obs = _obs({"old": 1.0, "added_since": 2.0 * (1 + 1e-6)})
    diffs, asserted, left_out, unmeasured = g0.tier_c_compare("x", ref, obs, [], {"/old": g0.TOL_FLOOR})
    assert not diffs and (asserted, left_out, unmeasured) == (1, 0, 1)
    # once it is measured, the same difference fails
    assert g0.tier_c_compare("x", ref, obs, [], {"/old": g0.TOL_FLOOR, "/added_since": g0.TOL_FLOOR})[0]


def test_every_figure_of_a_measured_entry_has_a_measured_spread():
    """The committed state behind the rule above: on each platform, every figure of the reference record that is
    neither mesh-sized nor listed as unasserted is in that platform's tolerances, so nothing is left uncompared
    today."""
    ref = g0.reference_figures()
    for plat in _spreads():
        tol = g0.tolerances_for(plat)
        for entry in tol["entries"]:
            missing = [p for p in ref["figures"].get(entry, {})
                       if not g0.mesh_sized(p) and p not in tol["unasserted"].get(entry, {})
                       and p not in tol["tolerances"].get(entry, {})]
            assert missing == [], (plat, entry, missing[:5])


def test_tolerance_file_has_a_ceiling_and_a_class_above_it():
    """Up to the ceiling a figure is asserted to ten times its spread, never below the floor. Above it, it is in a
    class of its own, still at ten times its spread, as long as the platform's own value, halved, is outside that
    tolerance; only a figure that spreads more is unasserted."""
    f = G0 / "tolerances.json"
    if not f.is_file():
        pytest.skip("no tolerances recorded yet")
    t = json.loads(f.read_text())
    assert t["spread_ceiling"] <= 1e-7 and t["wide_tolerance_limit"] <= 0.5
    assert t["wide_spread_ceiling"] * (g0.TOL_FACTOR + 0.5) <= t["wide_tolerance_limit"] * (1 + 1e-12)
    for fam in t["families"].values():
        for entry, m in fam["tolerances"].items():
            assert all(t["floor"] <= v <= g0.TOL_FACTOR * t["spread_ceiling"] * (1 + 1e-9) for v in m.values()), entry
            assert not any(g0.mesh_sized(p) for p in m), entry
        for entry, m in fam["over_ceiling"].items():
            for ptr, (tol, spread, plat) in m.items():
                assert t["spread_ceiling"] < spread < t["wide_spread_ceiling"], (entry, ptr)
                assert tol == g0.TOL_FACTOR * spread and tol < t["wide_tolerance_limit"], (entry, ptr)
                assert plat and ptr not in fam["tolerances"][entry] and not g0.mesh_sized(ptr), (entry, ptr)
        for entry, m in fam["unasserted"].items():
            assert all(d >= t["wide_spread_ceiling"] for d, _ in m.values()), entry
            assert not set(m) & (set(fam["tolerances"][entry]) | set(fam["over_ceiling"][entry])), entry


def test_a_figure_over_the_ceiling_fails_for_a_factor_of_two():
    """Every figure of the class above the ceiling, on every measured platform: that platform's record passes, and the
    same record with the figure doubled or halved does not."""
    ref = g0.reference_figures()
    t = json.loads((G0 / "tolerances.json").read_text())
    n = 0
    for plat, s in _spreads().items():
        tol = g0.tolerances_for(plat)
        for entry, m in t["families"][plat]["over_ceiling"].items():
            for ptr in m:
                r, v = ref["figures"][entry][ptr], s["figures"][entry][ptr]
                assert g0.rel_diff(r, v) <= tol["tolerances"][entry][ptr], (plat, entry, ptr)
                for changed in (2 * v, v / 2):
                    assert g0.rel_diff(r, changed) > tol["tolerances"][entry][ptr], (plat, entry, ptr, changed)
                n += 1
    assert n, "no figure is over the ceiling on any platform: the class is empty and this test shows nothing"
    # derived on synthetic records: the three classes, and the boundary of the last
    refd = {"figures": {"e": {"/a": 1.0, "/b": 1.0, "/c": 1.0, "/d": 1.0, "/z": 0.0}}, "discrete": {"e": {}}, "layout": {"e": []}}
    sp = {"p": {"figures": {"e": {"/a": 1.0 + 1e-12, "/b": 1.0 + 1e-4, "/c": 1.047, "/d": 1.048, "/z": 1e-9}},
                "discrete": {"e": {}}, "layout": {"e": []}}}
    fam = g0.derive_tolerances(refd, sp)["families"]["p"]
    assert set(fam["tolerances"]["e"]) == {"/a"} and set(fam["over_ceiling"]["e"]) == {"/b", "/c"}
    assert set(fam["unasserted"]["e"]) == {"/d", "/z"}
    assert fam["over_ceiling"]["e"]["/c"][0] < g0.rel_diff(1.0, 1.047 / 2)  # 0.47 < 0.4765: the halved value fails
    assert g0.TOL_FACTOR * 0.048 > g0.rel_diff(1.0, 1.048 / 2)             # 0.48 > 0.476: it would not, so unasserted


def _spreads():
    """The figure record of every platform but the reference: `spread/<key>.json`, or `baseline/<key>/figures.json` for
    a platform that once was the reference (the same measurement under another file name)."""
    return g0.measured_platforms()


def test_a_platform_cannot_have_its_figures_recorded_in_two_places(tmp_path, monkeypatch):
    """A platform that once was the reference has its figures under `baseline/<key>/figures.json` and every other
    platform under `spread/<key>.json`. Only one of the two is read, so holding both is refused instead of silently
    preferring one, and the reference platform must have no spread file at all."""
    assert g0.ambiguous_figure_records() == []        # the committed records
    monkeypatch.setattr(g0, "G0", tmp_path)
    (tmp_path / "spread").mkdir()
    g0._write(tmp_path / "baseline" / "p" / "figures.json", {"figures": {}})
    assert g0.ambiguous_figure_records() == []
    g0._write(tmp_path / "spread" / "p.json", {"figures": {}})
    assert g0.ambiguous_figure_records() == ["p"]
    assert g0.figure_record("p") == {"figures": {}}    # the spread file is the one that is read
    (tmp_path / "spread" / "p.json").unlink()
    g0._write(tmp_path / "spread" / f"{g0.REFERENCE}.json", {"figures": {}})
    assert g0.ambiguous_figure_records() == [g0.REFERENCE]
    assert g0.REFERENCE not in g0.measured_platforms()  # and it is never a spread against itself


def test_tolerance_file_is_the_derivation_from_the_recorded_spread():
    """tolerances.json is generated, never edited: deriving it again from the committed records gives the same file."""
    ref = g0.reference_figures()
    doc = g0.derive_tolerances(ref, _spreads())
    assert json.dumps(doc, indent=1, sort_keys=True) + "\n" == (G0 / "tolerances.json").read_text("utf-8")


def test_each_measured_platform_has_its_own_tolerances():
    """A platform is compared with the spread measured on it alone: a noisy developer host loosens no CI platform, and
    the host that gave a spread is not checked against the worst case over all platforms."""
    t = json.loads((G0 / "tolerances.json").read_text())
    spreads = _spreads()
    assert set(t["families"]) == set(spreads) | {"any"}
    for plat in spreads:
        assert t["families"][plat]["measured_on"] == [plat]
        assert g0.tolerances_for(plat)["tolerances"] == g0._asserted(t["families"][plat])  # both classes
        assert g0.tolerances_for(plat)["over_ceiling"] == t["families"][plat]["over_ceiling"]
    assert g0.tolerances_for("some-other-host")["tolerances"] == g0._asserted(t["families"]["any"])
    anyfam = t["families"]["any"]
    any_tols = g0._asserted(anyfam)
    for plat, fam in t["families"].items():
        if plat == "any":
            continue
        for entry, tols in g0._asserted(fam).items():
            for ptr, v in tols.items():  # `any` is never tighter than a platform's own
                assert ptr in anyfam["unasserted"][entry] or any_tols[entry][ptr] >= v, (plat, entry, ptr)


def test_each_recorded_platform_passes_its_own_tolerances_and_a_changed_verdict_does_not():
    """The record of every measured platform, compared as `check` compares a run there, has no difference; with one
    verdict flipped, one string changed or one asserted figure moved by ten tolerances, it has."""
    ref = g0.reference_figures()
    for plat, s in _spreads().items():
        tol = g0.tolerances_for(plat)
        for entry in s["figures"]:
            if entry not in ref["figures"]:  # pending: the reference platform has no record of the entry yet
                continue
            un_d, un_l = tol["unasserted_discrete"].get(entry, {}), tol["unasserted_layout"].get(entry, [])

            def differences(figs, disc):
                out = [p for p, v in figs.items() if p in ref["figures"][entry] and not g0.mesh_sized(p)
                       and p not in tol["unasserted"].get(entry, {})
                       and p in tol["tolerances"].get(entry, {})          # unmeasured here: not compared at all
                       and g0.rel_diff(ref["figures"][entry][p], v) > tol["tolerances"][entry][p]]
                out += [p for p in g0.discrete_difference(g0._no_identity(ref["discrete"][entry]), g0._no_identity(disc))
                        if p not in un_d]
                out += [p for p in g0.layout_difference(ref["layout"][entry], s["layout"][entry]) if p not in un_l]
                return out

            assert differences(s["figures"][entry], s["discrete"][entry]) == [], (plat, entry)
            # the same figures as the reference, but for the timings, which each record masks as measured on its platform
            extra = set(s["figures"][entry]) ^ set(ref["figures"][entry])
            assert all(g0.maskable(p) for p in extra), (plat, entry)
            verdict = next(p for p, v in s["discrete"][entry].items() if isinstance(v, bool))
            assert differences(s["figures"][entry], dict(s["discrete"][entry], **{verdict: not s["discrete"][entry][verdict]})) == [verdict]
            name = next(p for p, v in s["discrete"][entry].items()
                        if isinstance(v, str) and p not in un_d and p not in g0.identity_patterns("results.json"))
            assert differences(s["figures"][entry], dict(s["discrete"][entry], **{name: "changed"})) == [name]
            fig, t = next(iter(sorted(tol["tolerances"][entry].items())))
            moved = dict(s["figures"][entry], **{fig: ref["figures"][entry][fig] * (1 + 10 * t) + (1e-200 if ref["figures"][entry][fig] == 0 else 0)})
            assert differences(moved, s["discrete"][entry]) == [fig], (plat, entry, fig)


def test_no_physics_verdict_is_left_unasserted():
    """The non-numeric leaves measured to differ between platforms are build identity and texts that quote a number;
    no boolean (a verdict of a physics check) is among them on any measured platform."""
    t = json.loads((G0 / "tolerances.json").read_text())
    for plat, fam in t["families"].items():
        for entry, m in fam["unasserted_discrete"].items():
            for ptr, (ref, _) in m.items():
                assert not isinstance(ref, bool), (plat, entry, ptr)


def test_tolerances_only_tighten():
    fam = {"measured_on": ["p"], "tolerances": {"e": {"/a": 1e-9, "/b": 1e-9}}, "unasserted": {"e": {}},
           "unasserted_discrete": {"e": {}}, "unasserted_layout": {"e": []}}
    old = {"families": {"p": fam}}

    def new(**kw):
        return {"families": {"p": dict(fam, **kw), "q": dict(fam, tolerances={"e": {}})}}

    assert g0.loosened(old, new()) == []
    assert g0.loosened(old, new(tolerances={"e": {"/a": 1e-10, "/b": 1e-9}})) == []
    assert g0.loosened(old, new(tolerances={"e": {"/a": 1e-8, "/b": 1e-9}})) == ["p: e: /a: tolerance 1e-09 -> 1e-08"]
    assert g0.loosened(old, new(tolerances={"e": {"/a": 1e-9}})) == ["p: e: /b is no longer asserted"]
    assert g0.loosened(old, new(unasserted_discrete={"e": {"/s": ["x", {"p": "y"}]}})) == ["p: e: /s is newly listed in unasserted_discrete"]
    assert g0.loosened(old, new(unasserted_layout={"e": ["/xs"]})) == ["p: e: /xs is newly listed in unasserted_layout"]
    assert g0.loosened(old, {"families": {}}) == ["p: the family is gone"]


def test_tier_a_difference_is_found_and_a_mask_silences_only_its_stream():
    a = {"designs": {"d": {"validate": {"status": 0, "stdout": "ok"}}}, "corpus": {}}
    b = {"designs": {"d": {"validate": {"status": 1, "stdout": "ok"}}}, "corpus": {}}
    assert g0.tier_a_diff(a, b, [])
    assert not g0.tier_a_diff(a, b, [{"entry": "d", "op": "validate", "stream": "status"}])
    c = {"designs": {"d": {"validate": {"status": 0, "stdout": "changed"}}}, "corpus": {}}
    assert g0.tier_a_diff(a, c, [{"entry": "d", "op": "validate", "stream": "status"}])


def test_3d_entries_are_not_run_by_a_build_without_the_3d_engine(monkeypatch):
    entries = g0.load_manifest()["tier_b"]
    assert any(e.get("requires_engine_3d") for e in entries)
    monkeypatch.setattr(g0, "build_kind", lambda: "sequential")
    run = g0._entries(entries, None)
    assert run and not any(e.get("requires_engine_3d") for e in run)
    monkeypatch.setattr(g0, "build_kind", lambda: "mpi")
    assert len(g0._entries(entries, None)) == len(entries)


def test_ranks_are_read_from_the_launcher(monkeypatch):
    monkeypatch.delenv("AXMP_G0_LAUNCH", raising=False)
    assert g0.launched_ranks() == 1
    monkeypatch.setenv("AXMP_G0_LAUNCH", "mpirun -n 4")
    assert g0.launched_ranks() == 4
    monkeypatch.setenv("AXMP_G0_LAUNCH", "srun --overlap -np 2")
    assert g0.launched_ranks() == 2


def test_every_run_gets_one_thread_and_one_blas_kernel(monkeypatch):
    """The BLAS kernel is chosen from the processor unless it is pinned, and the kernels round differently."""
    monkeypatch.setattr(g0.platform, "machine", lambda: "x86_64")
    env = g0.fixed_environment()
    assert env["OMP_NUM_THREADS"] == "1" and env["OPENBLAS_NUM_THREADS"] == "1" and env["OPENBLAS_CORETYPE"] == "HASWELL"
    monkeypatch.setenv("OPENBLAS_CORETYPE", "SKYLAKEX")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    out = g0._run([sys.executable, "-c", "import os; print(os.environ['OPENBLAS_CORETYPE'], os.environ['OMP_NUM_THREADS'])"])
    assert out["stdout"].split() == ["HASWELL", "1"]
    monkeypatch.setattr(g0.platform, "machine", lambda: "arm64")
    assert "OPENBLAS_CORETYPE" not in g0.fixed_environment()


def test_tier_b_is_refused_on_another_fingerprint():
    base = {"compared": {"os": "ubuntu-24.04", "mesher": "4.15.0"}}
    here = {"compared": {"os": "macos-15", "mesher": "4.15.0"}}
    assert g0.fingerprint_mismatch(base, here) == ["os: baseline 'ubuntu-24.04', here 'macos-15'"]
    assert g0.fingerprint_mismatch(base, base) == []


def test_baselines_carry_a_reason_per_mask():
    for p in (G0 / "baseline").glob("*/masks.json"):
        m = json.loads(p.read_text())
        for kind in ("tier_a", "tier_b", "tier_c_by_name"):
            for e in m[kind]:
                assert e.get("reason"), (p, e)


def test_only_timings_and_run_identity_are_maskable():
    """The names are matched whole: a physics leaf whose name merely contains a timing word is not maskable."""
    for ptr in ("/seconds", "/hts/seconds", "/static/poloidal/solver/factorization_seconds", "/timing/solve_s",
                "/timing/electrostatics_s", "/mesh/info/mesher/seconds", "/run/hostname", "/run/pid", "/run/timestamp",
                # the per-solve timings of the 3D solver's log, by exact name and place
                "/solver/solves/0/clean_s", "/solver/solves/3/setup_s", "/solver/solves/*/solve_s"):
        assert g0.maskable(ptr), ptr
    for ptr in ("/hts/steps/*/newton_update", "/hts/steps/*/max_update", "/hts/newton/update", "/thermal/q_wall",
                "/thermal/sc_wall_t", "/thermal/sc_wall_cap_t", "/thermal/wall", "/hts/tau_s", "/hts/steps/*/time_s",
                "/pulse/duration", "/coupling/g_factor", "/hts/timing_s", "/timing/steps/*/dW_J", "/update_date",
                # the 3D solver's log holds physics too, and its timing names mean nothing elsewhere
                "/solver/solves/0/iterations", "/solver/solves/0/relative_residual", "/solver/solves/0/decay_s",
                "/solver/solve_s", "/solver/solves/solve_s", "/hts/solves/0/solve_s", "/solver/solves/0/x/solve_s",
                "/solver/solves/first/solve_s", "/static/setup_s", "/clean_s"):
        assert not g0.maskable(ptr), ptr
    assert g0._classify("/hts/steps/3/newton_update", 1.0) != "timing or run identity"
    assert g0._classify("/timing/solve_s", 1.0) == "timing or run identity"


def test_a_mask_cannot_hide_physics():
    """Only timings and run identity are masked in a JSON file, and no mask blanks a whole stdout."""
    for p in (G0 / "baseline").glob("*/masks.json"):
        m = json.loads(p.read_text())
        for e in m["tier_b"]:
            assert g0.maskable(e["pointer"]), (p, e)
            assert e["reason"] == "timing or run identity", (p, e)
        for e in m["tier_a"]:
            assert e["stream"] != "stdout", (p, e)


def test_record_is_refused_on_a_host_that_only_shares_the_platform_key():
    """The platform key of another cluster node can be the reference platform's. `record` decides by the fingerprint, so
    such a host cannot overwrite the reference baseline; a reference fingerprint without a compiler is never written."""
    ref = json.loads((G0 / "baseline" / g0.REFERENCE / "fingerprint.json").read_text())
    assert g0.refuse_record(ref, g0.REFERENCE, False) == ""
    other = {"compared": dict(ref["compared"], os="ubuntu-24.04", libc="glibc-2.39", strumpack=False)}
    why = g0.refuse_record(other, g0.REFERENCE, False)
    assert "ubuntu-24.04" in why and "AXMP_G0_PLATFORM" in why
    assert g0.refuse_record(other, g0.REFERENCE, True) == ""
    blank = {"compared": dict(ref["compared"], compiler="")}
    assert g0.refuse_record(blank, g0.REFERENCE, False) and g0.refuse_record(blank, g0.REFERENCE, True)
    assert g0.refuse_record(other, "a-host-never-recorded", False) == ""
    assert ref["compared"]["compiler"]
    for p in (G0 / "baseline").glob("*/fingerprint.json"):
        assert "threads" not in json.loads(p.read_text())["compared"], p


def test_the_reference_platform_is_one_pinned_node_type():
    """Tier B means byte-identical, so the reference platform is one node type, not a cluster: the processor is a
    compared fingerprint field, the record carries the pinned model, and `record` refuses another processor even with
    --repin, which covers a compiler or a mesher that moved but never the pin itself."""
    ref = json.loads((G0 / "baseline" / g0.REFERENCE / "fingerprint.json").read_text())
    assert ref["compared"]["cpu"] == g0.REFERENCE_CPU and g0.REFERENCE_CPU
    assert "cpu" not in ref["informational"]  # it moved out of the informational part, it is not in both
    assert g0.fingerprint_mismatch(ref, {"compared": dict(ref["compared"], cpu="Another(R) CPU")}) == [
        f"cpu: baseline {g0.REFERENCE_CPU!r}, here 'Another(R) CPU'"]
    moved = {"compared": dict(ref["compared"], cpu="Another(R) CPU")}
    for repin in (False, True):
        why = g0.refuse_record(moved, g0.REFERENCE, repin)
        assert "pinned to one node type" in why and g0.REFERENCE_CPU in why, repin
    # on a platform that is not the reference the processor is recorded but pins nothing
    assert g0.refuse_record(moved, "a-host-never-recorded", False) == ""
    assert "pinned to one node type" not in g0.refuse_record(moved, "linux-x86_64-sequential", False)


def test_require_reference_forces_the_reference_role_whatever_the_platform_key():
    """`--require-reference` is the host's claim to the reference role, and the role must not depend on the fingerprint
    matching: the reference node's own natural platform key is a hosted one (`linux-x86_64-mpi`, since the key holds
    only the system, the machine and the build kind), and on a hosted key tier B is no part of the gate. Were the flag
    read after the key, a reference host whose fingerprint had moved -- exactly what the flag exists to fail on --
    would be compared as a hosted platform and pass with tier B dropped."""
    hosted = "linux-x86_64-mpi"
    assert hosted in g0.TIER_AC_PLATFORMS and g0.REFERENCE not in g0.TIER_AC_PLATFORMS
    # no claim and no reference fingerprint: the key decides, and that key is compared at tiers A and C
    assert g0.compared_as(hosted, False, False) == (hosted, "")
    # the fingerprint is the reference's: the reference role, under either key
    key, note = g0.compared_as(hosted, True, False)
    assert key == g0.REFERENCE and g0.REFERENCE in note
    # the claim alone, with a fingerprint that is not the reference's: still the reference role, so tier B is compared
    # with the reference's record and the mismatch is a failure
    key, note = g0.compared_as(hosted, False, True)
    assert key == g0.REFERENCE and "--require-reference" in note and "not the recorded one" in note
    # the host that is already named as the reference platform needs no note
    for on_ref in (False, True):
        for claim in (False, True):
            assert g0.compared_as(g0.REFERENCE, on_ref, claim) == (g0.REFERENCE, "")
    # a host of its own keeps its own name: its own record is what it is compared with
    assert g0.compared_as("my-host", False, False) == ("my-host", "")
    assert g0.compared_as("my-host", False, True)[0] == g0.REFERENCE


def test_check_with_require_reference_fails_on_a_drifted_fingerprint_under_either_key(monkeypatch, capfd):
    """`check` itself, not only the role rule: with --require-reference a fingerprint that is not the reference's is a
    failure, whether the host presents under the reference's own name or under the hosted key a cluster node of the
    pinned type has by default. Nothing is solved here: the comparison stops at the fingerprint."""
    if not _built():
        pytest.skip("needs the built package (`check` reads this build's fingerprint)")
    real = g0.fingerprint
    for key in ("linux-x86_64-mpi", g0.REFERENCE):
        monkeypatch.setattr(g0, "fingerprint", lambda key=key: _drifted(real(), key))
        rc = g0.main(["check", "--tiers", "B", "--require-reference"])
        out = capfd.readouterr().out
        assert rc == 1 and "FAIL tier B" in out, (key, out)
        assert "ranks" in out, (key, out)


def test_check_that_compared_nothing_is_not_a_pass(monkeypatch, capfd):
    """A tier that does not run on the host leaves nothing compared, and `check` must not call that a pass: tier B
    alone on a hosted platform, tier C alone on the reference platform. Nothing is solved here."""
    if not _built():
        pytest.skip("needs the built package (`check` reads this build's fingerprint)")
    real = g0.fingerprint
    reference = json.loads((G0 / "baseline" / g0.REFERENCE / "fingerprint.json").read_text())
    hosted = lambda: _drifted(real(), "linux-x86_64-mpi")
    on_reference = lambda: dict(real(), platform=g0.REFERENCE, compared=dict(reference["compared"]))
    for fp, tiers, why in ((hosted, "B", "tier B: not part of the gate"), (on_reference, "C", "tier C: not run on the reference")):
        monkeypatch.setattr(g0, "fingerprint", fp)
        rc = g0.main(["check", "--tiers", tiers])
        out = capfd.readouterr().out
        assert rc == 2 and "G0: nothing compared here" in out and "G0: passed" not in out, (tiers, out)
        assert why in out, (tiers, out)


def _drifted(fp, key):
    """A fingerprint under the platform name `key` with one compared field moved."""
    fp["platform"] = key
    fp["compared"] = dict(fp["compared"], ranks=fp["compared"].get("ranks", 1) + 1)
    return fp


def test_masks_only_shrink():
    committed = {"tier_a": [], "tier_b": [{"entry": "e", "file": "results.json", "pointer": "/t_s"}], "tier_c_by_name": []}
    same = {"tier_a": [], "tier_b": [{"entry": "e", "file": "results.json", "pointer": "/t_s"}], "tier_c_by_name": []}
    smaller = {"tier_a": [], "tier_b": [], "tier_c_by_name": []}
    more = {"tier_a": [{"entry": "d", "op": "validate", "stream": "stdout"}], "tier_b": same["tier_b"], "tier_c_by_name": []}
    assert g0.new_masks(same, committed) == [] and g0.new_masks(smaller, committed) == []
    assert g0.new_masks(more, committed) == [("A", "d", "validate", "stdout")]
    assert g0.new_masks(more, None)


def test_results_layout_is_part_of_tier_b():
    """A reordered or reshaped results.json differs even where its unmasked values are all equal."""
    a = {"x": 1.0, "y": [1, 2], "seconds": 3.0}
    b = {"y": [1, 2], "x": 1.0, "seconds": 9.0}
    pats = {"results.json": ["/seconds"]}

    rec = g0.baseline_entry(_run_of(a), pats)
    assert g0.tier_b_compare("e", rec, _run_of(dict(a, seconds=7.0)), pats) == []
    assert any("order" in d for d in g0.tier_b_compare("e", rec, _run_of(b), pats))


def test_scalar_spans_find_every_leaf_where_it_is_written():
    text = ('{ "a" : -1.50e-3,"s":"x \\"q\\" {[,:]} y", "n": null, "t": true,\n "xs": [1, [2.0, {"k": 3E+2}], []],'
            ' "o": {}, "k/2": {"t": false} }\n')
    doc = json.loads(text)
    spans = g0.scalar_spans(text)
    assert [(p, json.loads(text[a:b])) for p, a, b in spans] == [
        ("/a", -0.0015), ("/s", 'x "q" {[,:]} y'), ("/n", None), ("/t", True), ("/xs/0", 1), ("/xs/1/0", 2.0),
        ("/xs/1/1/k", 300.0), ("/k/2/t", False)]
    assert text[spans[0][1]:spans[0][2]] == "-1.50e-3"  # the value as it is written, not as it parses
    assert sorted(p for p, _, _ in spans) == sorted(p for p, _ in g0.leaves(doc))
    for bad in ('{"a": 1} x', '{"a" 1}', '[1 2]', '{"a": }'):
        with pytest.raises(ValueError):
            g0.scalar_spans(bad)


def test_a_masked_json_file_is_compared_as_text():
    """Outside the masked values a JSON file with a mask is compared as text: another spelling of a number, other
    whitespace or another key order is a difference, and a masked value is not, wherever its name also occurs."""
    text = '{\n  "seconds": 4.25,\n  "coupling": {"Psi": 0.1, "seconds": 7},\n  "xs": [{"seconds": 1.0}, {"seconds": 2.0}]\n}\n'
    pats = {"results.json": ["/seconds", "/xs/*/seconds"]}
    assert g0.masked_text(text, pats["results.json"]) == (
        '{\n  "seconds": <masked>,\n  "coupling": {"Psi": 0.1, "seconds": 7},\n  "xs": [{"seconds": <masked>}, {"seconds": <masked>}]\n}\n')
    assert g0.masked_text(text, []) == text
    rec = g0.baseline_entry(_run_of(text=text), pats)
    assert rec["files"]["results.json"]["masked_leaves"] == 3

    def differs(changed):
        return g0.tier_b_compare("e", rec, _run_of(text=changed), pats)

    assert differs(text.replace("4.25", "9.5").replace("1.0", "3e-9")) == []      # masked values
    assert differs(text.replace('"seconds": 7', '"seconds": 8'))                   # the same name, not masked there
    assert differs(text.replace("0.1", "0.10"))                                     # another spelling of the same number
    assert differs(text.replace("0.1", "1e-1"))
    assert differs(text.replace("\n  ", "\n "))                                     # whitespace
    assert differs(json.dumps(json.loads(text)))                                    # re-serialised: same values
    assert differs('{"coupling": {"Psi": 0.1, "seconds": 7}, "seconds": 4.25, "xs": [{"seconds": 1.0}, {"seconds": 2.0}]}')
    # a file without a mask is compared by its whole text
    plain = g0.baseline_entry(_run_of(text=text, name="scene.json"), {})
    assert g0.tier_b_compare("e", plain, _run_of(text=text, name="scene.json"), {}) == []
    assert g0.tier_b_compare("e", plain, _run_of(text=text.replace("4.25", "4.26"), name="scene.json"), {})


def test_the_files_of_the_run_directory_are_part_of_tier_b():
    """A file that is no longer written, or a new one, is a difference even when its content is not compared; a new
    file that the manifest declares is not."""
    doc = {"x": 1.0}
    rec = g0.baseline_entry(_run_of(doc, extra_files=["axmp_status.json", "mesh_mesher.log"]), {})
    assert rec["file_set"] == ["axmp_status.json", "mesh_mesher.log", "results.json"]
    assert g0.tier_b_compare("e", rec, _run_of(doc, extra_files=["axmp_status.json", "mesh_mesher.log"]), {}) == []
    gone = g0.tier_b_compare("e", rec, _run_of(doc, extra_files=["mesh_mesher.log"]), {})
    assert len(gone) == 1 and "axmp_status.json" in gone[0] and "no longer written" in gone[0]
    new = _run_of(doc, extra_files=["axmp_status.json", "mesh_mesher.log", "run.json"])
    assert any("run.json" in d for d in g0.tier_b_compare("e", rec, new, {}))
    assert g0.tier_b_compare("e", rec, new, {}, allowed_new_files=["run.json"]) == []
    assert g0.tier_b_compare("e", rec, _run_of(doc, extra_files=["mesh_mesher.log", "run.json"]), {}, allowed_new_files=["run.json"])
    # two runs that leave different files are not a record
    a, b = _run_of(doc, extra_files=["core.123"]), _run_of(doc)
    assert "file set" in g0.tier_b_masks(a, b)[1]
    assert g0.load_manifest()["allowed_new_files"] == []


def test_the_version_leaf_is_compared_with_the_build_not_with_the_record(monkeypatch):
    """results.json carries the package version. It is not part of any record, so a version bump changes no baseline,
    and it must be this build's version, so a wrong one is still found."""
    monkeypatch.setattr(g0, "identity_value", lambda what: {"version": "1.2.3"}[what])
    old, new = {"version": "1.2.2", "g": 0.5}, {"version": "1.2.3", "g": 0.5}
    rec = g0.baseline_entry(_run_of(old), {})
    assert g0.tier_b_compare("e", rec, _run_of(new), {}) == []
    assert g0.tier_b_compare("e", rec, _run_of(dict(new, g=0.6)), {})
    assert g0.identity_diffs("e", _run_of(new)) == []
    assert "1.2.2" in g0.identity_diffs("e", _run_of(old))[0] and g0.identity_diffs("e", _run_of({"g": 0.5}))
    figs = g0._figures_of({"e": _run_of(old)}, {"tier_b": []})
    assert figs["discrete"]["e"] == {} and figs["figures"]["e"] == {"/g": 0.5}
    # tier C against a record made when the version was still recorded: the leaf is passed over on both sides
    assert g0.tier_c_exact("e", {"/version": "1.2.2"}, g0.layout(old), new, [])[0] == []
    assert g0.tier_c_exact("e", {"/version": "1.2.2", "/name": "a"}, g0.layout(old), new, [])[0]
    # only that leaf of that file: a `version` elsewhere is an ordinary leaf
    assert g0.tier_b_compare("e", g0.baseline_entry(_run_of({"mesher": {"version": "4"}}), {}), _run_of({"mesher": {"version": "5"}}), {})


def test_an_entry_that_is_not_reproducible_has_no_record_and_a_physics_leaf_is_never_masked():
    a = _run_of({"timing": {"solve_s": 1.0}, "seconds": 2.0, "coupling": {"g": 0.5}})
    b = _run_of({"timing": {"solve_s": 1.5}, "seconds": 2.5, "coupling": {"g": 0.5}})
    masks, unmaskable = g0.tier_b_masks(a, b)
    assert not unmaskable and {m["reason"] for m in masks} == {g0.TIMING}
    c = _run_of({"timing": {"solve_s": 1.5}, "seconds": 2.5, "coupling": {"g": 0.5 + 1e-16}})
    masks, unmaskable = g0.tier_b_masks(a, c)
    assert [m["pointer"] for m in masks if m["reason"] != g0.TIMING] == ["/coupling/g"]
    assert g0.process_not_pinned({"mesh3d_info.json/high_order_canonical": "on", "mesh3d_info.json/address_randomization": "off"}) == []
    assert g0.process_not_pinned({"mesh3d_info.json/high_order_canonical": "n/a", "mesh3d_info.json/address_randomization": "unknown"}) == []
    assert g0.process_not_pinned({"mesh3d_info.json/high_order_canonical": "off"})
    assert g0.process_not_pinned({"mesh3d_info.json/address_randomization": "on"})
    flags = g0.canonical_flags(_run_of({"address_randomization": "off", "high_order_canonical": "on", "x": 1}, name="mesh3d_info.json")["files"])
    assert flags == {"mesh3d_info.json/address_randomization": "off", "mesh3d_info.json/high_order_canonical": "on"}
    assert "AXMP_MESHER_ASLR" in g0.ENV_CLEARED


def test_no_run_inherits_the_switch_that_keeps_address_randomisation_on(monkeypatch):
    monkeypatch.setenv("AXMP_MESHER_ASLR", "on")
    out = g0._run([sys.executable, "-c", "import os; print(os.environ.get('AXMP_MESHER_ASLR', 'unset'))"])
    assert out["stdout"].split() == ["unset"]


def test_only_must_name_entries_of_the_manifest():
    """A check or a record restricted to names that are no entries would compare nothing and report success."""
    import types

    m = g0.load_manifest()
    assert g0._only(types.SimpleNamespace(only=None), m) == (None, "")
    names, bad = g0._only(types.SimpleNamespace(only="template_toroid,example_hts_shield"), m)
    assert names == {"template_toroid", "example_hts_shield"} and not bad
    for value in ("nosuchentry", "template_toroid,nosuchentry", ","):
        assert "names no tier-B entry" in g0._only(types.SimpleNamespace(only=value), m)[1], value


def _platform_record(tmp_path, key, fp, entries, fmt=None, by_name=()):
    base = tmp_path / "baseline" / key
    g0._write(base / "fingerprint.json", fp)
    g0._write(base / "tierB.json", {"format": fmt or g0.FORMAT_B, "platform": key, "entries": entries})
    g0._write(base / "masks.json", {"platform": key, "tier_a": [], "tier_b": [],
                                    "tier_c_by_name": [{"entry": e, "reason": "not reproducible"} for e in by_name]})
    g0._write(base / "tierA.json", {"designs": {"template_toroid": {}}, "corpus": {}})


def test_a_platform_is_compared_with_its_own_record_only_under_its_recorded_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(g0, "G0", tmp_path)
    fp = {"platform": "host-a", "compared": {"os": "x-1", "compiler": "c 1"}, "informational": {"cpu": "A", "build_info": {}}}
    assert g0.own_record("host-a", fp)[0] is None
    _platform_record(tmp_path, "host-a", fp, {"e1": {"status": 0}}, by_name=["e2"])
    own, why = g0.own_record("host-a", fp)
    assert why == [] and set(own["entries"]) == {"e1"} and set(own["by_name"]) == {"e2"}
    moved = dict(fp, compared={"os": "x-2", "compiler": "c 1"})
    own, why = g0.own_record("host-a", moved)
    assert own is None and why == ["os: baseline 'x-1', here 'x-2'"]
    # a record of an older harness compared other things: it is no record
    _platform_record(tmp_path, "host-a", fp, {"e1": {"status": 0}}, fmt="axionmultiphysics.g0/1")
    own, why = g0.own_record("host-a", fp)
    assert own is None and "older harness" in why[0]


def test_a_fingerprint_names_no_directory_of_the_machine():
    info = {"version": "1", "build_dir": "/home/someone/build", "git_dirty_files": "a b",
            "library": {"build_dir": "/home/someone/build", "git_sha": "abc", "git_dirty_files": ""}}
    assert g0._without_paths(info) == {"version": "1", "library": {"git_sha": "abc"}}


def test_fingerprint_report_names_what_moved():
    rec = {"platform": "p", "compared": {"os": "ubuntu-24.04", "mesher": "4.15.2", "cpu": "A"},
           "informational": {"runner_image": "1", "blas": [{"name": "libopenblas.so.0", "package": "libopenblas0-pthread 0.3.26+ds-1"}],
                             "build_info": {"git_sha": "abc", "mfem": "4.10.0"}}}
    here = {"platform": "p", "compared": {"os": "ubuntu-24.04", "mesher": "4.15.3", "cpu": "B"},
            "informational": {"runner_image": "2", "blas": [{"name": "libopenblas.so.0", "package": "libopenblas0-pthread 0.3.26+ds-1ubuntu0.1"}],
                              "build_info": {"git_sha": "def", "mfem": "4.10.0"}}}
    lines = g0.fingerprint_report(rec, here)
    assert any(line.startswith("compared field mesher") for line in lines)
    assert any(line.startswith("compared field cpu: baseline 'A', here 'B'") for line in lines)
    assert any(line.startswith("blas:") and "ubuntu0.1" in line for line in lines)
    assert not any("git_sha" in line or "build_info" in line for line in lines)  # the commit always differs
    assert len(g0.fingerprint_report(rec, rec)) == 1 and "has the fingerprint recorded" in g0.fingerprint_report(rec, rec)[0]
    assert "no fingerprint is recorded" in g0.fingerprint_report(None, here)[0]


def _tolerances_doc(families):
    """A tolerances document that asserts nothing, with a family per platform: what `pending` reads to find out which
    entries a platform compared at tiers A and C has a measurement of."""
    return {"format": g0.FORMAT, "floor": g0.TOL_FLOOR,
            "families": {k: {"measured_on": [k], "tolerances": {e: {} for e in v}, "over_ceiling": {},
                             "unasserted": {}, "unasserted_discrete": {}, "unasserted_layout": {}}
                         for k, v in families.items()}}


def test_pending_lists_what_each_recorded_platform_has_no_comparison_for(tmp_path, monkeypatch):
    """Per recorded platform, the entries the committed records give it no comparison for, with its role and the reason.
    On the reference platform that is a tier-B record (and, for a solve, the figures tier C reads); on a platform
    compared at tiers A and C it is a measurement of its own, which is what gives the entry's figures a tolerance. A
    mesh entry needs none there, because tier C compares its exit status alone. A 3D entry is not expected of a
    sequential build, and a tier-B record of an older harness does not count."""
    manifest = g0.load_manifest()
    corpus = {d["id"]: {} for d in g0.load_corpus()}
    monkeypatch.setattr(g0, "G0", tmp_path)
    monkeypatch.setattr(g0, "load_manifest", lambda: manifest)
    monkeypatch.setattr(g0, "load_corpus", lambda: [{"id": i} for i in corpus])
    all_b = [e["id"] for e in manifest["tier_b"]]
    seq_b = [e["id"] for e in manifest["tier_b"] if not e.get("requires_engine_3d")]
    solves = [e["id"] for e in manifest["tier_b"] if e.get("op", "solve") != "mesh"]
    designs = {d["id"]: {} for d in manifest["tier_a"]["designs"]}

    def tier_a_of(key, ds=designs, cs=corpus):
        g0._write(tmp_path / "baseline" / key / "tierA.json", {"designs": ds, "corpus": cs})

    _platform_record(tmp_path, g0.REFERENCE, {"compared": {}}, {i: {} for i in all_b})
    tier_a_of(g0.REFERENCE)
    g0._write(tmp_path / "baseline" / g0.REFERENCE / "figures.json",
              {"figures": {i: {} for i in solves}, "discrete": {i: {} for i in solves}})
    fams = {}
    for key in g0.TIER_AC_PLATFORMS:
        _platform_record(tmp_path, key, {"compared": {}}, {})
        tier_a_of(key)
        fams[key] = [i for i in solves if key.endswith("-mpi") or i in seq_b]
    fams["any"] = solves
    g0._write(tmp_path / "tolerances.json", _tolerances_doc(fams))
    p = g0.pending()
    assert set(p) == set(g0.RECORDED_PLATFORMS)
    assert p == {k: dict(p[k], tier_a=[], tier_b=[]) for k in p}, p   # nothing pending anywhere
    assert p[g0.REFERENCE]["role"].startswith("reference") and p[g0.REFERENCE]["reason"]
    for key in g0.TIER_AC_PLATFORMS:
        assert p[key]["role"] == "tiers A and C" and p[key]["reason"] != p[g0.REFERENCE]["reason"]
    # a hosted platform without a measurement of one solve: pending there, and only there
    gone = solves[0]
    fams["macos-arm64-sequential"] = [i for i in fams["macos-arm64-sequential"] if i != gone]
    g0._write(tmp_path / "tolerances.json", _tolerances_doc(fams))
    p = g0.pending()
    assert p["macos-arm64-sequential"]["tier_b"] == [gone]
    assert p["linux-x86_64-mpi"]["tier_b"] == [] and p[g0.REFERENCE]["tier_b"] == []
    # one design and one corpus document missing from a platform's text record
    one, doc = next(iter(designs)), next(iter(corpus))
    tier_a_of("linux-x86_64-sequential", {k: v for k, v in designs.items() if k != one},
              {k: v for k, v in corpus.items() if k != doc})
    assert g0.pending()["linux-x86_64-sequential"]["tier_a"] == [one, f"corpus:{doc}"]
    # the reference platform: a sequential platform is not expected to record the 3D entries, the reference is
    _platform_record(tmp_path, g0.REFERENCE, {"compared": {}}, {i: {} for i in seq_b})
    tier_a_of(g0.REFERENCE)
    assert g0.pending()[g0.REFERENCE]["tier_b"] == [i for i in all_b if i not in seq_b]
    # an entry the reference lists as not reproducible there is compared by nothing (tier C does not run on the
    # reference platform), so it is pending there, not passed over
    _platform_record(tmp_path, g0.REFERENCE, {"compared": {}}, {i: {} for i in all_b[1:]}, by_name=[all_b[0]])
    tier_a_of(g0.REFERENCE)
    assert g0.pending()[g0.REFERENCE]["tier_b"] == [all_b[0]]
    # a reference solve without figures cannot serve tier C
    _platform_record(tmp_path, g0.REFERENCE, {"compared": {}}, {i: {} for i in all_b})
    tier_a_of(g0.REFERENCE)
    g0._write(tmp_path / "baseline" / g0.REFERENCE / "figures.json",
              {"figures": {i: {} for i in solves[1:]}, "discrete": {i: {} for i in solves}})
    assert g0.pending()[g0.REFERENCE]["tier_b"] == [solves[0]]
    # a record of an older harness
    _platform_record(tmp_path, g0.REFERENCE, {"compared": {}}, {i: {} for i in all_b}, fmt="axionmultiphysics.g0/1")
    tier_a_of(g0.REFERENCE)
    g0._write(tmp_path / "baseline" / g0.REFERENCE / "figures.json",
              {"figures": {i: {} for i in solves}, "discrete": {i: {} for i in solves}})
    assert g0.pending()[g0.REFERENCE]["tier_b"] == all_b


def test_tier_a_passes_over_an_entry_the_platform_has_no_record_of_and_nothing_else():
    """On a platform compared at tiers A and C a design the committed record does not have is reported as pending, by
    name, not as a difference; a design that IS recorded is compared, and one that was recorded and is gone fails."""
    base = {"designs": {"old": {"validate": {"status": 0, "stdout": "ok"}}}, "corpus": {"c1": {"validate": {"status": 1}}}}
    now = {"designs": {"old": {"validate": {"status": 0, "stdout": "ok"}}, "new": {"validate": {"status": 0}}},
           "corpus": {"c1": {"validate": {"status": 1}}, "c2": {"validate": {"status": 1}}}}
    assert len(g0.tier_a_diff(base, now, [])) == 2                      # both are differences without the list
    assert g0.tier_a_diff(base, now, [], no_record={"new", "corpus:c2"}) == []
    assert len(g0.tier_a_diff(base, now, [], no_record={"new"})) == 1    # only what the list names
    changed = {"designs": dict(now["designs"], old={"validate": {"status": 1, "stdout": "ok"}}), "corpus": now["corpus"]}
    assert any("old" in d for d in g0.tier_a_diff(base, changed, [], no_record={"new", "corpus:c2"}))
    missing = {"designs": {}, "corpus": now["corpus"]}
    assert any("was not run" in d for d in g0.tier_a_diff(base, missing, [], no_record={"old", "corpus:c2"}))


def test_the_reference_platform_has_a_record_of_every_entry_and_every_pending_entry_has_a_reason():
    """The gate's own statement: nothing is pending on the reference platform, because nothing else compares those
    entries byte for byte (`check --require-reference` fails while something is). Elsewhere a pending entry is named
    with the reason, so it does not pass silently."""
    p = g0.pending()
    assert p[g0.REFERENCE] == dict(p[g0.REFERENCE], tier_a=[], tier_b=[]), p[g0.REFERENCE]
    for key, v in p.items():
        assert v["role"] and v["reason"], key
    assert g0.REFERENCE not in g0.TIER_AC_PLATFORMS
    assert set(g0.RECORDED_PLATFORMS) == {g0.REFERENCE, *g0.TIER_AC_PLATFORMS}


def test_the_pending_list_is_the_derivation_from_the_committed_baselines():
    """tests/g0/pending.json marks the manifest entries that have no CI baseline yet. It is generated (`g0.py pending
    --write`), so it cannot go stale: after a baseline is recorded and committed it must be written again."""
    assert json.dumps(g0.pending_document(), indent=1, sort_keys=True) + "\n" == (G0 / "pending.json").read_text("utf-8")


def test_the_gate_holds_solves_of_every_engine_and_section():
    """What the gate must keep covering: a 3D solve at order 1 and on a curved order-2 mesh, an axisymmetric mechanics
    solve and a thermal solve, and valid documents with each design section, shape kind and material block."""
    m = g0.load_manifest()
    solves = {e["id"]: json.loads((ROOT / e["design"]).read_text()) for e in m["tier_b"] if "design" in e and e.get("op") == "solve"}
    d3 = [d for i, d in solves.items() if d.get("symmetry") == "3d"]
    assert {d["mesh"]["order"] for d in d3} >= {1, 2}
    assert all(e.get("requires_engine_3d") for e in m["tier_b"] if "design" in e and json.loads((ROOT / e["design"]).read_text()).get("symmetry") == "3d")
    assert any("mechanics" in d and d.get("symmetry") != "3d" for d in solves.values())
    assert any("thermal" in d and d.get("symmetry") != "3d" for d in solves.values())
    docs = [json.loads((ROOT / d["path"]).read_text()) for d in m["tier_a"]["designs"] if "path" in d]
    keys = set().union(*docs)
    assert keys >= {"circuits", "curves", "mechanics", "scan", "thermal", "windings", "superconductors", "pickup", "benchmarks"}

    def kinds(x, found):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("shape", "profile") and isinstance(v, dict):
                    found |= set(v)
                kinds(v, found)
        elif isinstance(x, list):
            for v in x:
                kinds(v, found)
        return found

    assert kinds(docs, set()) >= {"rect", "polygon", "contour", "frame", "disk", "union", "difference"}
    blocks = set().union(*(set(mat) for d in docs for mat in d.get("materials", {}).values()))
    assert blocks >= {"mechanical", "thermal", "rf"}
    paths = set().union(*(set(c["path"]) for d in docs for c in d.get("curves", [])))
    assert paths >= {"segment", "polyline", "arc"}


def test_the_workflow_checks_the_hosted_platforms_at_tiers_a_and_c_only():
    """`.github/workflows/g0.yml` serves the public repository's CI at a release. The reference platform of tier B is
    not a hosted runner any more, so no job of it claims to be the reference: each one runs a plain `check` (tier A
    against its own record, tier C against the reference's), and none passes --require-reference, which would demand a
    tier-B record of the runner image. It is triggered by the release itself (the tag push of docs/developer/release.md),
    not only by a pull request, because a release is where the three platforms and macOS are covered. The platforms
    that take gmsh from PyPI still install the release the reference platform's mesher is built from, not a range, and a
    record run still passes on the two switches of the dispatch."""
    import re

    text = (ROOT / ".github" / "workflows" / "g0.yml").read_text("utf-8")
    assert re.search(r'^  push:\n    tags: \["v\*"\]$', text, re.M), "no release trigger"
    ref = json.loads((G0 / "baseline" / g0.REFERENCE / "fingerprint.json").read_text())["compared"]["mesher"]
    pins = re.findall(r'"gmsh([^"]*)"', text)
    assert len(pins) == 2 and set(pins) == {"==" + ref.split("-")[0]}, pins
    records = re.findall(r"g0\.py record([^;]*);", text)
    assert len(records) == 3 and all(r.split() == ["$REPIN", "$NEW_MASKS"] for r in records), records
    for name in ("record", "repin", "allow_new_masks"):
        assert re.search(rf"^      {name}:\n", text, re.M), name
    assert "--require-reference" not in text
    checks = re.findall(r"g0\.py check([^;]*);", text)
    assert len(checks) == 3 and all(c.strip() == "" for c in checks), checks  # a plain check, no flag
    # the platforms the workflow runs are the ones the harness compares at tiers A and C, and only those
    for plat in g0.TIER_AC_PLATFORMS:
        assert f"g0-{plat}" in text, plat
    assert g0.REFERENCE not in text


def _synthetic_record(tmp_path, monkeypatch, key, second_g, compared=None, extra=None, text=None):
    """`record` under the platform name `key` on synthetic runs of two entries: `steady` differs between its two runs
    in a timing only, `flaky` also in a physics leaf (its second and later runs have /coupling/g = second_g; with
    second_g = 0.5 it is as steady as the other). `extra` is merged into every results document; `text` is what tier A
    observes (nothing by default). Returns the exit status; `_synthetic_record.calls` then holds the number of runs of each entry."""
    import types

    compared = compared or {"compiler": "c 1", "cpu": g0.REFERENCE_CPU if key == g0.REFERENCE else "some processor"}
    monkeypatch.setattr(g0, "G0", tmp_path)
    monkeypatch.setattr(g0, "fingerprint", lambda: {"platform": key, "compared": dict(compared), "informational": {}})
    monkeypatch.setattr(g0, "load_manifest", lambda: {"tier_b": [{"id": "steady"}, {"id": "flaky"}], "tier_a": {"designs": []}})
    monkeypatch.setattr(g0, "tier_a", lambda work: text or {"designs": {}, "corpus": {}})
    monkeypatch.setattr(g0, "identity_diffs", lambda eid, obs: [])
    monkeypatch.setattr(g0, "build_kind", lambda: "mpi")
    monkeypatch.setattr(g0, "load_corpus", lambda: [])
    calls = _synthetic_record.calls = {}

    def run_entry(e, work):
        n = calls[e["id"]] = calls.get(e["id"], 0) + 1
        doc = {"coupling": {"g": 0.5 if e["id"] == "steady" or n == 1 else second_g, "valid": True},
               "timing": {"solve_s": 1.0 if n == 1 else 3.7}, **(extra or {})}
        text = json.dumps(doc)
        return {"status": 0, "stdout": "", "file_set": ["results.json"],
                "files": {"results.json": {"kind": "json", "doc": doc, "text": text, "sha256": g0._sha(text), "bytes": len(text)}}}

    monkeypatch.setattr(g0, "run_entry", run_entry)
    return g0.cmd_record(types.SimpleNamespace(out=None, spread=None, only=None, repin=True, allow_new_masks=True))


def test_an_entry_that_is_not_reproducible_on_the_reference_is_recorded_nowhere_and_stays_pending(tmp_path, monkeypatch, capsys):
    """One of two disagreeing values is no reference: the entry gets no tier-B record and no figures, so no platform
    can be compared with it; it is pending on the reference platform (where tier C does not run), `record` says so for
    the entry, and `check` fails for it there and says why."""
    assert _synthetic_record(tmp_path, monkeypatch, g0.REFERENCE, 0.500000001) == 0
    base = tmp_path / "baseline" / g0.REFERENCE
    assert set(json.loads((base / "tierB.json").read_text())["entries"]) == {"steady"}
    figs = json.loads((base / "figures.json").read_text())
    assert set(figs["figures"]) == set(figs["discrete"]) == set(figs["layout"]) == {"steady"}
    assert figs["figures"]["steady"] == {"/coupling/g": 0.5}  # the timing is masked
    masks = json.loads((base / "masks.json").read_text())
    assert [m["entry"] for m in masks["tier_c_by_name"]] == ["flaky"] and "/coupling/g" in masks["tier_c_by_name"][0]["reason"]
    assert {m["entry"] for m in masks["tier_b"]} == {"steady"}
    out = capsys.readouterr().out
    assert f"NO RECORD of flaky on {g0.REFERENCE}" in out and "NO RECORD of steady" not in out
    p = g0.pending()[g0.REFERENCE]
    assert (p["tier_a"], p["tier_b"]) == ([], ["flaky"])
    rc = g0.main(["check", "--tiers", "AB", "--require-reference"])
    out = capsys.readouterr().out
    assert rc == 1 and "G0: FAILED" in out
    assert "FAIL tier B: flaky: the reference platform's record lists it as not reproducible" in out and "/coupling/g" in out
    assert "tier B: steady compared with this platform's record" in out and "FAIL tier B: steady" not in out


def test_an_entry_that_is_not_reproducible_elsewhere_is_measured_over_both_of_its_runs(tmp_path, monkeypatch, capsys):
    """On a platform other than the reference such an entry is compared in tier C alone. Both of its runs are kept, the
    spread that its tolerances come from spans both, and its timings are not figures."""
    key = "linux-x86_64-sequential"
    assert _synthetic_record(tmp_path, monkeypatch, key, 0.50000001) == 0
    assert f"NO RECORD of flaky on {key}" in capsys.readouterr().out
    assert set(json.loads((tmp_path / "baseline" / key / "tierB.json").read_text())["entries"]) == {"steady"}
    rec = json.loads((tmp_path / "spread" / f"{key}.json").read_text())
    assert rec["figures"] == {"steady": {"/coupling/g": 0.5}, "flaky": {"/coupling/g": 0.5}}
    assert rec["second_run"]["figures"] == {"flaky": {"/coupling/g": 0.50000001}}
    assert set(rec["second_run"]["discrete"]) == set(rec["second_run"]["layout"]) == {"flaky"}
    ref = {k: {e: v for e in ("steady", "flaky")} for k, v in
           (("figures", {"/coupling/g": 0.5}), ("discrete", {"/coupling/valid": True}), ("layout", rec["layout"]["flaky"]))}
    doc = g0.derive_tolerances(ref, {key: rec})
    fam = doc["families"][key]
    assert fam["tolerances"]["steady"] == {"/coupling/g": g0.TOL_FLOOR}
    assert fam["tolerances"]["flaky"]["/coupling/g"] == pytest.approx(g0.TOL_FACTOR * 2e-8, rel=1e-6)
    # without the second run the figure would have been given the floor, which the platform's own second run fails
    one = g0.derive_tolerances(ref, {key: {k: v for k, v in rec.items() if k != "second_run"}})["families"][key]
    assert one["tolerances"]["flaky"] == {"/coupling/g": g0.TOL_FLOOR}
    # a record without such an entry has no second run at all
    assert _synthetic_record(tmp_path, monkeypatch, key, 0.5) == 0
    assert "second_run" not in json.loads((tmp_path / "spread" / f"{key}.json").read_text())
    assert "NO RECORD" not in capsys.readouterr().out
    # the platform counts as holding the entry: tier C covers it there
    g0._write(tmp_path / "tolerances.json", doc)
    assert g0.pending()[key]["tier_b"] == []


def test_a_check_whose_tier_b_was_refused_says_what_nothing_compared(tmp_path, monkeypatch, capsys):
    """A host with a record of its own whose fingerprint moved is refused tier B and compared in tier C alone. That is
    not a failure (exit status 0), and it must not read like a full pass: `check` says that it ran without tier B and
    how many figures of results.json were then compared with nothing."""
    extra = {"mesh": {"elements": 100}}  # one mesh-sized figure per entry, which tier C leaves out
    assert _synthetic_record(tmp_path, monkeypatch, g0.REFERENCE, 0.5, extra=extra) == 0
    assert _synthetic_record(tmp_path, monkeypatch, "my-host", 0.5, extra=extra) == 0
    g0._write(tmp_path / "tolerances.json", g0.derive_tolerances(g0.reference_figures(), g0.measured_platforms()))
    capsys.readouterr()
    # under its recorded fingerprint: tiers B and C both, and no such line
    assert g0.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "WITHOUT TIER B" not in out and "tier B: steady compared with this platform's record" in out
    assert out.rstrip().endswith("G0: passed")
    # the fingerprint moved
    monkeypatch.setattr(g0, "fingerprint", lambda: {"platform": "my-host", "informational": {},
                                                    "compared": {"compiler": "c 2", "cpu": "some processor"}})
    assert g0.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "tier B REFUSED" in out and "compiler: baseline 'c 1', here 'c 2'" in out
    line = next(x for x in out.splitlines() if x.startswith("G0 ran WITHOUT TIER B on my-host"))
    assert "2 figures" in line and "compiler" in line, line
    assert out.rstrip().splitlines()[-1].startswith("G0: passed WITHOUT TIER B")
    # a difference tier C does see is still a failure
    monkeypatch.setattr(g0, "run_entry", lambda e, work: _run_of({"coupling": {"g": 0.6, "valid": True}, "mesh": {"elements": 100}}))
    assert g0.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "G0 ran WITHOUT TIER B on my-host" in out and out.rstrip().endswith("G0: FAILED")


def test_the_processor_is_no_ground_to_refuse_a_record_where_no_bytes_are_compared(tmp_path, monkeypatch):
    """`cpu` is a compared fingerprint field because it pins the reference platform's bytes. A hosted runner is given
    whichever processor model its fleet has that day, and its record is compared at tiers A and C only, so the model
    pins nothing there: `record` does not refuse for it (a refusal would make the plain form a lottery, and `--repin`
    the habit). Every other compared field still refuses, and so does the processor wherever a record of bytes is
    compared: on the reference platform and on a host with a name of its own."""
    for key in (*g0.TIER_AC_PLATFORMS, *g0.TIER_A_PLATFORMS):
        committed = json.loads((G0 / "baseline" / key / "fingerprint.json").read_text())
        here = {"compared": dict(committed["compared"], cpu="A processor of the fleet (R) @ 2.45GHz")}
        assert g0.refuse_record(here, key, False) == "", key
        moved = {"compared": dict(here["compared"], compiler="another compiler")}
        why = g0.refuse_record(moved, key, False)
        assert "another compiler" in why and "cpu" not in why, key
        assert g0.refuse_record(moved, key, True) == "", key
    monkeypatch.setattr(g0, "G0", tmp_path)
    recorded = {"compared": {"compiler": "c 1", "cpu": "A"}}
    other = {"compared": {"compiler": "c 1", "cpu": "B"}}
    for key in ("linux-x86_64-mpi", g0.TIER_A_PLATFORMS[0], "my-host"):
        g0._write(tmp_path / "baseline" / key / "fingerprint.json", recorded)
    assert g0.refuse_record(other, "linux-x86_64-mpi", False) == ""
    assert g0.refuse_record(other, g0.TIER_A_PLATFORMS[0], False) == ""
    assert "cpu: baseline 'A', here 'B'" in g0.refuse_record(other, "my-host", False)


def test_the_cluster_sequential_build_is_compared_with_a_text_record_of_its_own(tmp_path, monkeypatch, capsys):
    """The development gate's sequential job builds on the cluster, and its text is compared with a record made there,
    under a platform name of its own: under its bare key it would be compared with a hosted platform's record, which
    only the public repository's CI can record again, so an argued text change could not pass the gate. The platform is
    compared at tier A only: `record` runs no solve there and writes the text record and nothing else, `check` says
    that tiers B and C are no part of the gate there, and asked for them alone it compares nothing, which is not a
    pass."""
    key, = g0.TIER_A_PLATFORMS
    assert key == "lawrencium-x86_64-sequential" and key not in g0.RECORDED_PLATFORMS
    # the committed record: the three files of a text record, of every design and corpus document, and no figures
    base = G0 / "baseline" / key
    assert sorted(p.name for p in base.iterdir()) == ["fingerprint.json", "masks.json", "tierA.json"]
    assert not (G0 / "spread" / f"{key}.json").exists() and key not in g0.measured_platforms()
    a, fp, masks = (json.loads((base / n).read_text()) for n in ("tierA.json", "fingerprint.json", "masks.json"))
    assert set(a["designs"]) == {d["id"] for d in g0.load_manifest()["tier_a"]["designs"]}
    assert set(a["corpus"]) == {d["id"] for d in g0.load_corpus()}
    assert fp["platform"] == key and fp["compared"]["build"] == "sequential"
    assert masks["tier_b"] == [] and masks["tier_c_by_name"] == []
    # the jobs of the local gate name their platform (dev/ is not part of a published snapshot)
    jobs = ROOT / "dev" / "ci-local"
    if jobs.is_dir():
        assert f"export AXMP_G0_PLATFORM={key}\n" in (jobs / "gate_seq.slurm").read_text()
        for name in ("gate_mpi.slurm", "gate_g0.slurm"):
            assert f"export AXMP_G0_PLATFORM={g0.REFERENCE}\n" in (jobs / name).read_text(), name
    # `record` there: no solve, the text record alone
    text = {"designs": {"old": {"validate": {"status": 0, "stdout": "ok"}}}, "corpus": {}}
    assert _synthetic_record(tmp_path, monkeypatch, key, 0.5, text=text) == 0
    assert _synthetic_record.calls == {}, "a solve was run on a platform compared at tier A only"
    made = tmp_path / "baseline" / key
    assert json.loads((made / "tierA.json").read_text())["designs"] == text["designs"]
    assert sorted(p.name for p in made.iterdir()) == ["fingerprint.json", "masks.json", "tierA.json"]
    assert not (tmp_path / "spread").exists()
    assert json.loads((made / "masks.json").read_text()) == {"platform": key, "tier_a": [], "tier_b": [], "tier_c_by_name": []}
    capsys.readouterr()
    # `check` there
    assert g0.main(["check"]) == 0
    out = capsys.readouterr().out
    assert f"tiers B and C: not part of the gate on {key}" in out and out.rstrip().endswith("G0: passed")
    assert g0.main(["check", "--tiers", "BC"]) == 2
    assert "G0: nothing compared here" in capsys.readouterr().out
    # a design without a record is a failure there, not pending: the record is made on the cluster, with the change
    new = {"designs": dict(text["designs"], new={"validate": {"status": 0, "stdout": ""}}), "corpus": {}}
    monkeypatch.setattr(g0, "tier_a", lambda work: new)
    assert g0.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "design 'new' is not in the baseline" in out and "PENDING" not in out
    # and a changed text is a difference
    changed = {"designs": {"old": {"validate": {"status": 0, "stdout": "changed"}}}, "corpus": {}}
    monkeypatch.setattr(g0, "tier_a", lambda work: changed)
    assert g0.main(["check", "--tiers", "A"]) == 1
    assert "design 'old' validate stdout differs" in capsys.readouterr().out


# ----------------------------------------------------------------- the gate itself
def _platform_baseline():
    """The platform key whose baseline the gate compares. Where ``AXMP_G0_PLATFORM`` names the platform (a gate job
    sets it), a missing build or baseline fails, so that a gate job cannot pass by skipping; elsewhere it skips."""
    named = bool(os.environ.get("AXMP_G0_PLATFORM"))
    if not _built():
        if named:
            pytest.fail("AXMP_G0_PLATFORM is set but the package is not built or the axmp executable is not found")
        pytest.skip("needs the built package")
    key = g0.platform_key()
    if not (G0 / "baseline" / key / "tierA.json").is_file():
        if named:
            pytest.fail(f"AXMP_G0_PLATFORM={key} is set but no G0 baseline is recorded for it")
        pytest.skip(f"no G0 baseline recorded for {key}")
    return key


def test_a_gate_job_cannot_pass_by_skipping(monkeypatch):
    """With AXMP_G0_PLATFORM set, a missing baseline (or build) fails; without it, the test skips."""
    def outcome():
        """'fail', 'skip' or 'returned', as a value: a skip must not escape as the test's own outcome."""
        try:
            _platform_baseline()
        except pytest.fail.Exception as e:
            return "fail", str(e)
        except pytest.skip.Exception as e:
            return "skip", str(e)
        return "returned", ""

    monkeypatch.setenv("AXMP_G0_PLATFORM", "no-such-platform-x86_64-mpi")
    monkeypatch.setattr(sys.modules[__name__], "_built", lambda: True)
    kind, msg = outcome()
    assert kind == "fail" and "no G0 baseline" in msg, (kind, msg)
    monkeypatch.setattr(sys.modules[__name__], "_built", lambda: False)
    kind, msg = outcome()
    assert kind == "fail" and "not built" in msg, (kind, msg)
    monkeypatch.delenv("AXMP_G0_PLATFORM")
    assert outcome()[0] == "skip"


def test_tier_a_matches_the_baseline(capfd):
    _platform_baseline()
    rc = g0.main(["check", "--tiers", "A"])
    out = capfd.readouterr().out
    assert rc == 0, out


@pytest.mark.skipif(not os.environ.get("AXMP_SLOW"), reason="tiers B and C solve (AXMP_SLOW=1)")
def test_tiers_b_and_c_match_the_baseline(capfd):
    if _platform_baseline() in g0.TIER_A_PLATFORMS:
        pytest.skip("this platform is compared at tier A only")
    rc = g0.main(["check", "--tiers", "BC"])
    out = capfd.readouterr().out
    assert rc == 0, out


def test_dependency_pins_ignore_transport_pins_and_follow_the_others(tmp_path):
    """A pin of a particle-transport program does not change the fingerprint; any other pin does, and a file without
    transport lines hashes as the whole file."""
    import hashlib

    base = (b"11cc8645d622a16510b39a20efc64f34862b41976152d17f9fbf3e91f899766c  STRUMPACK-8.0.0.tar.gz\n"
            b"d5aabe991b8b5569aa26e2b5d4b59ac617ee10ebac3d45fd6ff9c74b2c1a47dd  mfem-4.10.tar.gz\n")
    transport = (b"718752c1829cbfcf174f0e539d60ee3f76843b894444e61d2228d66a6887ef6e  geant4-v11.4.3.tar.gz\n"
                 b"85dce1e623f91499b2973e3f939bf760b0f745ad4f4eb1bd0fbce2074e2e8f5c  pythia8318.tgz\n"
                 b"9555f1d06f82987fbb4658862705515740414fd34b4db6ad2ed76a2dc08d3bde  Xerces-C-3.3.0.tar.gz\n")

    def digest(data: bytes) -> str:
        f = tmp_path / "SHA256SUMS.txt"
        f.write_bytes(data)
        return g0.dependency_pins_digest(f)

    assert digest(base) == hashlib.sha256(base).hexdigest(), "a file without transport lines must hash as before"
    assert digest(base + transport) == digest(base), "a transport pin changed the fingerprint"
    assert digest(base.replace(b"d5aabe99", b"d5aabe98") + transport) != digest(base), "a pin change went unnoticed"
    assert digest(base + b"0" * 64 + b"  zlib-1.3.tar.gz\n") != digest(base), "a new non-transport pin went unnoticed"
