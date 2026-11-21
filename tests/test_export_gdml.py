"""``Design.export_gdml``: an axisymmetric electromagnetic design to GDML through the mesher's partition (beta).

The contract tested (``docs/guides/solids.md`` documents it):

- ``Design.export_gdml(path, composition, *, solids=None, chord_tol_m=None, workdir=None, mesher=None)`` runs
  ``axmp-mesher --export-regions`` on the design's mesher input (the ``mesh_in.json`` that ``axmp mesh`` writes, except
  ``output`` and ``info``), converts the partition with ``solids.from_partition`` and writes GDML at ``path`` (and
  ``solids.json`` at ``solids``); it returns the solids document. With ``workdir`` it keeps ``regions_in.json`` (the
  input) and ``regions.json`` (the partition) there; without, it leaves nothing behind.
- Each region takes its design material, the background region ``air`` the material ``air``. ``composition`` maps
  material names to particle blocks with an optional ``density`` (kg/m3); a material's density is the design's (the
  EM parser's value: built-in defaults, expressions) when it has one, else the composition's; both and different is
  ``particle.composition_invalid``, neither ``particle.density_missing``. A used material without a block is
  ``particle.composition_missing``; unused entries, sheets and disabled regions need none.
- Refusals: a model document (``model.em_only_operation``), a
  mesher without the export mode (``particle.partition_unavailable``), a failed export run
  (``mesh.generation_failed``, with the mesher's message); nothing is written then.

Independent references: the partition of the same input exported by the mesher directly; ``solids.from_partition``
and ``solids.write``/``write_gdml`` on it, with the materials built here from the densities stated by hand
(``_partition_ref``: iron 7874, copper 8960 and nb 8570 are the EM built-ins, the others the design's or the
composition's); ``axmp mesh``'s own input file. Comparisons are exact (bit for bit, byte for byte): the route must not
change a number.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import axionmultiphysics as axmp
import _partition_ref as R
from _partition_ref import mp  # noqa: F401 (a fixture)

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = list(R.DESIGNS)


def _solids():
    from axionmultiphysics import solids
    return solids


def _export(design: dict, path: Path, composition: dict, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return axmp.Design(design).export_gdml(path, composition=composition, **kw)


def _raises(f, *a, **kw):
    from axionmultiphysics import _core
    with pytest.raises(_core.AxionMultiPhysicsError) as ei:
        f(*a, **kw)
    return ei.value


# ============================================================================================ API
def test_export_gdml_is_a_beta_method_with_the_documented_options():
    f = getattr(axmp.Design, "export_gdml", None)
    assert f is not None, "Design has no export_gdml"
    sig = inspect.signature(f)
    assert list(sig.parameters) == ["self", "path", "composition", "solids", "chord_tol_m", "workdir", "mesher"]
    P = inspect.Parameter
    assert sig.parameters["path"].kind is P.POSITIONAL_OR_KEYWORD
    assert sig.parameters["composition"].kind is P.POSITIONAL_OR_KEYWORD
    for k in ("solids", "chord_tol_m", "workdir", "mesher"):
        assert sig.parameters[k].kind is P.KEYWORD_ONLY and sig.parameters[k].default is None, k
    assert "beta" in (f.__doc__ or "").lower()


# ============================================================================================ the route
@pytest.mark.parametrize("name", DESIGNS)
def test_export_gdml_writes_the_solids_of_the_design_s_partition(mp, tmp_path, name):
    """The kept input is ``axmp mesh``'s (``output`` and ``info`` aside), the kept partition is the mesher's export of
    it, the returned document is ``from_partition`` of that partition with the materials stated here, and the files
    are ``write_gdml`` and ``write`` of the document, byte for byte."""
    solids = _solids()
    design = R.design(name)
    _, comp, dens = R.DESIGNS[name]
    gdml, sj, wd = tmp_path / "out" / "d.gdml", tmp_path / "out" / "solids.json", tmp_path / "w"
    (tmp_path / "out").mkdir()
    doc = _export(design, gdml, comp, solids=sj, workdir=wd)
    inp, _, _ = R.em_mesh(design, tmp_path / "mesh")
    kept = json.loads((wd / "regions_in.json").read_text("utf-8"))
    strip = lambda d: {k: v for k, v in d.items() if k not in ("output", "info")}  # noqa: E731
    assert strip(kept) == strip(inp), "regions_in.json is not axmp mesh's input"
    part = json.loads((wd / "regions.json").read_text("utf-8"))
    r, direct, _ = R.run_mesher(mp, inp, tmp_path / "direct", "regions", export=True)
    assert r.returncode == 0, r.stderr[-500:]
    assert R.same_json(R.without_seconds(part), R.without_seconds(direct)) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = solids.from_partition(part, R.solids_materials(name), R.region_materials(design))
    assert doc == want
    solids.write_gdml(want, tmp_path / "ref.gdml")
    solids.write(want, tmp_path / "ref.json")
    assert gdml.read_bytes() == (tmp_path / "ref.gdml").read_bytes()
    assert sj.read_bytes() == (tmp_path / "ref.json").read_bytes()
    used = {doc["regions"][r]["material"] for r in doc["regions"]}
    assert {m["name"]: m["density_kg_m3"] for m in doc["materials"]} == {m: dens[m] for m in used}
    assert set(doc["regions"]) == set(R.region_materials(design))


# ============================================================================================ densities and compositions
def test_densities_come_from_the_design_or_the_composition(mp, tmp_path):
    """The lead density as a parameter expression of the design; the winding's (none in the design) from the
    composition; iron's built-in 7874 also given, equal, in the composition; iron given 7800: refused, naming iron
    and both values; the winding with no density anywhere: refused."""
    d = R.design("yoke")
    d["parameters"] = {"rho_pb": 11340.0}
    d["materials"]["lead"]["density"] = "rho_pb"
    comp = copy.deepcopy(R.YOKE_COMPOSITION)
    comp["iron"]["density"] = 7874.0
    doc = _export(d, tmp_path / "a.gdml", comp)
    dens = {m["name"]: m["density_kg_m3"] for m in doc["materials"]}
    assert dens == {"winding": 8960.0, "iron": 7874.0, "lead": 11340.0, "air": 1.205}

    bad = copy.deepcopy(R.YOKE_COMPOSITION)
    bad["iron"]["density"] = 7800.0
    e = _raises(_export, R.design("yoke"), tmp_path / "b.gdml", bad)
    assert e.code == "particle.composition_invalid" and "iron" in str(e) and "7874" in str(e) and "7800" in str(e), e
    assert not (tmp_path / "b.gdml").exists()

    none = copy.deepcopy(R.YOKE_COMPOSITION)
    del none["winding"]["density"]
    e = _raises(_export, R.design("yoke"), tmp_path / "c.gdml", none)
    assert e.code == "particle.density_missing" and "winding" in str(e), e
    assert not (tmp_path / "c.gdml").exists()


def test_every_used_material_needs_a_composition(mp, tmp_path):
    for missing in ("lead", "air"):
        comp = {k: v for k, v in R.YOKE_COMPOSITION.items() if k != missing}
        e = _raises(_export, R.design("yoke"), tmp_path / f"{missing}.gdml", comp)
        assert e.code == "particle.composition_missing" and f"'{missing}'" in str(e), e
        assert not (tmp_path / f"{missing}.gdml").exists()
    # unused entries are ignored
    comp = dict(R.YOKE_COMPOSITION, nb={"nist": "G4_Nb"}, superconductor={"nist": "G4_Nb", "density": 8570.0})
    doc = _export(R.design("yoke"), tmp_path / "u.gdml", comp)
    assert {m["name"] for m in doc["materials"]} == {"winding", "iron", "lead", "air"}
    # the CORE template without its iron caps needs no iron
    d = R.design("core")
    d["parameters"]["iron_t"] = 0.0
    comp = {k: v for k, v in R.CORE_COMPOSITION.items() if k != "iron"}
    doc = _export(d, tmp_path / "core.gdml", comp)
    assert "iron" not in {m["name"] for m in doc["materials"]}
    assert not any(r.startswith("iron_cap") for r in doc["regions"])


def test_chord_tolerance_is_passed_on(mp, tmp_path):
    solids = _solids()
    doc = _export(R.design("curved"), tmp_path / "x.gdml", R.CURVED_COMPOSITION, chord_tol_m=1e-7,
                  workdir=tmp_path / "w")
    part = json.loads((tmp_path / "w" / "regions.json").read_text("utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = solids.from_partition(part, R.solids_materials("curved"), R.region_materials(R.design("curved")),
                                     chord_tol_m=1e-7)
    assert doc["chord_tol_m"] == 1e-7 and doc == want


# ============================================================================================ refusals
def _stub_mesher(directory: Path) -> Path:
    """A mesher without the export mode: what an older mesher answers to two arguments (the usage, exit 2)."""
    p = directory / "old-mesher"
    p.write_text("#!/bin/sh\necho 'usage: axmp-mesher input.json' >&2\nexit 2\n")
    p.chmod(0o755)
    return p


def _model_document() -> dict:
    import model_mesh_support as S
    return S.doc("split_rect")


# 3D designs are exported in every build: tests/test_export_gdml_3d.py (also the refusal of mesh.symmetry)
@pytest.mark.parametrize("case", ["model_document", "old_mesher", "failed_run"])
def test_export_gdml_refusals(mp, tmp_path, case):
    out = tmp_path / "x.gdml"
    sj = tmp_path / "s.json"
    comp = R.YOKE_COMPOSITION
    if case == "model_document":
        e = _raises(_export, _model_document(), out, comp, solids=sj)
        assert e.code == "model.em_only_operation" and "export_gdml" in str(e), e
    elif case == "old_mesher":
        stub = _stub_mesher(tmp_path)
        e = _raises(_export, R.design("yoke"), out, comp, solids=sj, mesher=str(stub))
        assert e.code == "particle.partition_unavailable" and "--export-regions" in str(e), e
        assert str(stub) in str(e) or "old-mesher" in str(e), e
    else:
        d = R.design("yoke")
        d["regions"].append({"name": "cover", "material": "iron",
                             "shape": {"rect": {"r": [0.0, 0.12], "z": [-0.12, 0.12]}}})
        e = _raises(_export, d, out, comp, solids=sj)
        assert e.code == "mesh.generation_failed" and "covered by later regions" in str(e), e
    assert not out.exists() and not sj.exists()


def test_a_mesher_that_cannot_run_is_the_packages_error(mp, tmp_path):
    """A program that does not exist (or cannot be executed) is ``build.requires_mesher``, the code of a mesher that is
    not installed, not a bare ``FileNotFoundError``; nothing is written."""
    out = tmp_path / "x.gdml"
    for program in (tmp_path / "no-such-mesher", tmp_path):
        e = _raises(_export, R.design("yoke"), out, R.YOKE_COMPOSITION, mesher=str(program))
        assert e.code == "build.requires_mesher" and str(program) in str(e), e
    assert not out.exists()


def test_a_mesher_that_cannot_load_is_build_requires_mesher_with_the_loader_message(mp, tmp_path):
    """Exit status 127 on the probe (a mesher without its libraries): the model path's code, the loader's message kept,
    and a hint that does not say the program was not found."""
    import model_mesh_support as S
    stub = S.stub(tmp_path, "no-libs", status=127, stderr=S.LOADER_GMSH)
    out = tmp_path / "x.gdml"
    e = _raises(_export, R.design("yoke"), out, R.YOKE_COMPOSITION, mesher=str(stub))
    assert e.code == "build.requires_mesher" and "libgmsh.so.4.15" in str(e), e
    assert "not found" not in str(e) and "pip install" in e.hint, e
    assert not out.exists()


def test_a_failed_export_run_carries_the_log_in_its_hint(mp, tmp_path):
    """The mesher's log lives in a directory that is gone when the error is read: its last lines are in the hint; with
    ``workdir`` the log is kept there and named."""
    wrapper = tmp_path / "failing-export"
    wrapper.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--export-regions" ]; then\n'
        '  if [ "$2" = /dev/null ]; then echo "axmp-mesher: cannot read input" >&2; exit 2; fi\n'
        '  echo "first line of the log" >&2; echo "the reason of the failure" >&2; exit 1\n'
        "fi\n"
        f'exec "{mp}" "$@"\n')
    wrapper.chmod(0o755)
    out = tmp_path / "x.gdml"
    e = _raises(_export, R.design("yoke"), out, R.YOKE_COMPOSITION, mesher=str(wrapper))
    assert e.code == "mesh.generation_failed" and "the reason of the failure" in e.hint, e
    assert "first line of the log" in e.hint and "exit status 1" in e.hint, e.hint
    wd = tmp_path / "w"
    e = _raises(_export, R.design("yoke"), out, R.YOKE_COMPOSITION, mesher=str(wrapper), workdir=wd)
    assert str(wd / "regions_mesher.log") in e.hint and (wd / "regions_mesher.log").is_file(), e.hint
    assert "the reason of the failure" in (wd / "regions_mesher.log").read_text()
    # a failure of axmp mesh's own run: the same hint, from its log
    d = R.design("yoke")
    d["regions"].append({"name": "cover", "material": "iron", "shape": {"rect": {"r": [0.0, 0.12], "z": [-0.12, 0.12]}}})
    e = _raises(_export, d, out, R.YOKE_COMPOSITION)
    assert e.code == "mesh.generation_failed" and "covered by later regions" in e.hint, e.hint
    assert not out.exists()


# ============================================================================================ files and reproducibility
def test_without_a_workdir_nothing_is_left_behind(mp, tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (tmp_path / "out").mkdir()
    monkeypatch.chdir(cwd)
    _export(R.design("yoke"), tmp_path / "out" / "y.gdml", R.YOKE_COMPOSITION)
    assert list(cwd.iterdir()) == []
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["y.gdml"]


_CHILD = r"""
import json, sys, warnings
sys.path.insert(0, sys.argv[1])
import _partition_ref as R
import axionmultiphysics as axmp
warnings.simplefilter("ignore")
axmp.Design(R.design("curved")).export_gdml(sys.argv[2], composition=R.CURVED_COMPOSITION, solids=sys.argv[3],
                                            workdir=sys.argv[4])
"""


def test_the_export_is_reproducible_in_fresh_processes(mp, tmp_path):
    """Two fresh processes (other hash seeds, working directories and paths) write the same GDML and solids.json. The
    children import the package this session imports, whatever the working directory (a relative PYTHONPATH too)."""
    outs = []
    for k in (1, 2):
        d = tmp_path / f"run{k}" / ("x" * (10 * k))
        d.mkdir(parents=True)
        home = os.path.dirname(os.path.dirname(os.path.abspath(axmp.__file__)))   # the package this session imports
        env = dict(os.environ, PYTHONHASHSEED=str(k), AXMP_MESHER=mp,
                   PYTHONPATH=os.pathsep.join([home] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        r = subprocess.run([sys.executable, "-c", _CHILD, str(ROOT / "tests"), str(d / "c.gdml"), str(d / "c.json"),
                            str(d / "work")], capture_output=True, text=True, cwd=d, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(((d / "c.gdml").read_bytes(), (d / "c.json").read_bytes()))
    assert outs[0] == outs[1]


# ============================================================================================ documentation
def test_the_guide_documents_the_export():
    guide = (ROOT / "docs" / "guides" / "solids.md").read_text("utf-8")
    for s in ("export_gdml", "--export-regions", "composition", "density", "`air`", "particle.partition_unavailable",
              "mesh.generation_failed", "regions.json"):
        assert s in guide, f"docs/guides/solids.md does not mention {s}"
    assert "| `particle.partition_unavailable` |" in guide, "the codes table has no row for particle.partition_unavailable"
    assert "export_gdml" in (ROOT / "llms.txt").read_text("utf-8")
    # the notes since 0.1.0: "Unreleased" and the release sections above 0.1.0
    notes = (ROOT / "CHANGELOG.md").read_text("utf-8").split("## Unreleased", 1)[1].split("\n## 0.1.0", 1)[0]
    assert "export_gdml" in notes and "--export-regions" in notes
