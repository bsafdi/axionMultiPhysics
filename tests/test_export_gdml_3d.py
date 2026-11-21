"""``Design.export_gdml`` of a 3D electromagnetic design: the design's 3D mesher input (the one ``axmp mesh`` writes),
``axmp-mesher --export-regions`` on it (with the chord tolerance, when given, added to it), and
``solids.from_partition`` on the partition it writes.

The 3D mesher input is written by a library function, the one the 3D engine uses, so the export runs in every build
(the sequential one too) and in every rank of a multi-rank program, each rank on its own with the same results. The
export runs the mesher only with ``--export-regions`` (no electromagnetic mesh). The engine's own refusals (sheets in a
3D design) are reported as themselves; a model reduced by symmetry planes is refused with
``particle.partition_unavailable``. The guide documents the 3D export and its limit for thin layers.

Independent references: ``axmp mesh``'s own mesher input (``_core.mesh``, in a build with the 3D engine), the input the
tests build (``_partition3d_ref.mesher_input_3d``, checked against the engine's there), the mesher run directly, the
densities stated by hand, the files two ranks write.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import axionmultiphysics as axmp
import _partition3d_ref as R3
import _partition_ref as R
from _partition_ref import mp  # noqa: F401 (a fixture)

from axionmultiphysics import solids

ROOT = Path(__file__).resolve().parents[1]
ENGINE_3D = bool(axmp.build_info()["engine_3d"])


def _refused(doc, tmp_path, composition, **kw):
    with pytest.raises(Exception) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            axmp.Design(doc).export_gdml(tmp_path / "x.gdml", composition, solids=tmp_path / "x.json", **kw)
    assert not (tmp_path / "x.gdml").exists() and not (tmp_path / "x.json").exists()
    return ei.value


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("output", "info", "chord_tol_m")}


def test_a_3d_design_reduced_by_symmetry_planes_is_refused(tmp_path):
    doc = R3.design3("dipole")
    doc["mesh"]["symmetry"] = [{"name": "mirror", "axis": "x", "side": 1, "type": "pec"}]
    e = _refused(doc, tmp_path, R3.DIPOLE_COMPOSITION)
    assert getattr(e, "code", None) == "particle.partition_unavailable", e
    assert "mesh.symmetry" in str(e), str(e)


def test_the_engine_s_refusal_of_a_3d_design_is_reported_as_itself(mp, tmp_path):  # noqa: F811
    """Sheets (``curves``) in a 3D design: the 3D engine refuses them ("Model sheets as thin regions"), and so does the
    export, with the engine's own message; the export does not drop them, and does not report a failed mesher run."""
    doc = R3.design3("dipole")
    doc["curves"] = [{"name": "s", "material": "superconductor", "path": {"segment": [[0.5, -0.1], [0.5, 0.1]]}}]
    e = _refused(doc, tmp_path, R3.DIPOLE_COMPOSITION, chord_tol_m=R3.TEST_CHORD)
    assert "zero-thickness curves" in str(e), str(e)
    assert getattr(e, "code", None) not in ("mesh.generation_failed", "particle.partition_unavailable"), e


@pytest.mark.parametrize("name", list(R3.DESIGNS3))
def test_export_gdml_writes_the_solids_of_a_3d_design(mp, tmp_path, name):  # noqa: F811
    """In every build: ``regions_in.json`` is the tests' 3D mesher input (and, in a build with the 3D engine, the
    ``mesh3d_in.json`` that ``axmp mesh`` writes) with the chord tolerance added; ``regions.json`` the mesher's export
    of it; the document ``from_partition``'s; the files the writers'."""
    doc = R3.design3(name)
    comp = R3.DESIGNS3[name][1]
    wd = tmp_path / "work"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = axmp.Design(doc).export_gdml(tmp_path / "x.gdml", comp, solids=tmp_path / "x.json",
                                           chord_tol_m=R3.TEST_CHORD, workdir=wd)
    inp = json.loads((wd / "regions_in.json").read_text("utf-8"))
    assert _strip(inp) == R3.mesher_input_3d(doc)
    assert inp["chord_tol_m"] == R3.TEST_CHORD and Path(inp["info"]) == wd / "regions.json"
    if ENGINE_3D:
        from axionmultiphysics import _core
        out = json.loads(_core.mesh(json.dumps(doc), str(tmp_path / "mesh"), 0, ""))
        assert out["ok"] is True
        ref = json.loads((tmp_path / "mesh" / "mesh3d_in.json").read_text("utf-8"))
        assert _strip(ref) == _strip(inp), "the export's mesher input is not the 3D engine's"
    part = json.loads((wd / "regions.json").read_text("utf-8"))
    direct_in = R3.mesher_input_3d(doc)
    direct_in["chord_tol_m"] = R3.TEST_CHORD
    r, direct, _ = R.run_mesher(mp, direct_in, tmp_path / "direct", "d", export=True)
    assert r.returncode == 0 and direct and direct.get("ok") is True, r.stderr[-1500:]
    assert R.same_json(R.without_seconds(part), R.without_seconds(direct)) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = solids.from_partition(part, R3.solids_materials3(name), R3.region_materials3(doc))
    assert got == want
    solids.write_gdml(want, tmp_path / "w.gdml")
    solids.write(want, tmp_path / "w.json")
    assert (tmp_path / "x.gdml").read_bytes() == (tmp_path / "w.gdml").read_bytes()
    assert (tmp_path / "x.json").read_bytes() == (tmp_path / "w.json").read_bytes()
    assert got["geometry"] == "3d" and got["chord_tol_m"] == R3.TEST_CHORD


def test_a_3d_export_runs_the_mesher_only_with_the_export_flag(mp, tmp_path):  # noqa: F811
    """The program given as ``mesher`` (a wrapper that logs its arguments) is run for the export (and its probe), never
    without ``--export-regions``: no electromagnetic mesh is made."""
    log = tmp_path / "calls.log"
    wrapper = tmp_path / "mesher-wrapper"
    wrapper.write_text(f"#!/bin/sh\necho \"$@\" >> '{log}'\nexec '{mp}' \"$@\"\n", "utf-8")
    wrapper.chmod(0o755)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axmp.Design(R3.design3("dipole")).export_gdml(tmp_path / "x.gdml", R3.DIPOLE_COMPOSITION,
                                                      chord_tol_m=R3.TEST_CHORD, mesher=str(wrapper))
    calls = log.read_text("utf-8").splitlines()
    assert calls and all(c.startswith("--export-regions ") for c in calls), calls
    assert sum(not c.endswith(os.devnull) for c in calls) == 1, calls


def test_the_chord_tolerance_of_a_3d_export_is_the_mesher_s(mp, tmp_path):  # noqa: F811
    """Without ``chord_tol_m`` the mesher input has no such key and the mesher's default applies; ``chord_tol_m`` is
    not passed to the conversion (a 3D partition carries its own)."""
    doc = R3.design3("dipole")
    wd = tmp_path / "work"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = axmp.Design(doc).export_gdml(tmp_path / "x.gdml", R3.DIPOLE_COMPOSITION, workdir=wd)
    inp = json.loads((wd / "regions_in.json").read_text("utf-8"))
    part = json.loads((wd / "regions.json").read_text("utf-8"))
    assert "chord_tol_m" not in inp
    assert part["chord_tol_m"] == min(1e-3, 1e-4 * part["diagonal_m"]) == got["chord_tol_m"]


def test_export_gdml_of_a_3d_design_needs_every_composition(mp, tmp_path):  # noqa: F811
    comp = {k: v for k, v in R3.DIPOLE_COMPOSITION.items() if k != "scint"}
    e = _refused(R3.design3("dipole"), tmp_path, comp, chord_tol_m=R3.TEST_CHORD)
    assert getattr(e, "code", None) == "particle.composition_missing" and "scint" in str(e), e


# ============================================================================================ several ranks
RANK_SCRIPT = r'''
import hashlib, json, os, sys, warnings
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import axionmultiphysics as axmp
import _partition3d_ref as R3
rank = next((int(os.environ[k]) for k in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID")
             if k in os.environ), 0)
size = next((int(os.environ[k]) for k in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS") if k in os.environ), 1)
out = Path(sys.argv[2]) / f"rank{rank}"
out.mkdir(parents=True, exist_ok=True)
rec = {"rank": rank, "size": size}
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axmp.Design(R3.design3("dipole")).export_gdml(out / "x.gdml", R3.DIPOLE_COMPOSITION, solids=out / "x.json",
                                                      chord_tol_m=R3.TEST_CHORD, workdir=out / "work",
                                                      mesher=sys.argv[3])
    part = json.loads((out / "work" / "regions.json").read_text("utf-8"))
    part.pop("seconds", None)
    rec.update(gdml=hashlib.sha256((out / "x.gdml").read_bytes()).hexdigest(),
               solids=hashlib.sha256((out / "x.json").read_bytes()).hexdigest(),
               partition=hashlib.sha256(json.dumps(part, sort_keys=True).encode()).hexdigest())
except Exception as e:
    rec["error"] = f"{getattr(e, 'code', type(e).__name__)}: {e}"[:600]
print("RESULT " + json.dumps(rec), flush=True)
'''


@pytest.mark.skipif(not os.environ.get("AXMP_MPI"), reason="multi-rank test: set AXMP_MPI=1")
def test_every_rank_of_a_multi_rank_program_exports_the_same_files(mp, tmp_path):  # noqa: F811
    """Under the MPI launcher with two ranks, each rank calls ``export_gdml`` on its own (nothing collective): both
    succeed and write the same partition, solids list and GDML."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import _mpi_launch as mpi
    finally:
        sys.path.pop(0)
    if not mpi.available():
        pytest.skip(mpi.skip_reason())
    script = tmp_path / "ranks.py"
    script.write_text(RANK_SCRIPT, "utf-8")
    argv = mpi.command(2) + [sys.executable, str(script), str(Path(__file__).resolve().parent), str(tmp_path), mp]
    pkg = str(Path(axmp.__file__).resolve().parents[1])
    env = mpi.environment(argv, dict(os.environ, PYTHONPATH=os.pathsep.join(
        [pkg] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != pkg])))
    env.setdefault("OMPI_MCA_pml", "ob1")
    env.setdefault("OMPI_MCA_btl", "self,vader")
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=900, cwd=tmp_path)
    recs = [json.loads(ln[len("RESULT "):]) for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("RESULT ")]
    assert sorted(x["rank"] for x in recs) == [0, 1], (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    assert all(x["size"] == 2 for x in recs), recs
    errors = [x for x in recs if "error" in x]
    assert not errors, errors
    a, b = sorted(recs, key=lambda x: x["rank"])
    assert (a["partition"], a["solids"], a["gdml"]) == (b["partition"], b["solids"], b["gdml"]), recs


# ============================================================================================ the guide
def _section(text: str, heading: str) -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def test_the_guide_documents_the_3d_export():
    guide = (ROOT / "docs" / "guides" / "solids.md").read_text("utf-8")
    for s in ("tessellated", "`volumes`", "`triangulation`", "`sagitta_m`", "`chord_tol_m`", "infinity shell",
              "`exterior`", "mesh.symmetry", "ABSOLUTE"):
        assert s in guide, f"docs/guides/solids.md does not mention {s}"
    assert re.search(r"min\(1 mm, 1e-4 × `?diagonal_m`?\)", guide), "the guide does not state the 3D default"
    three_d = _section(guide, "## 3D designs")
    assert "thinner than" in three_d and "partition_invalid" in three_d and "chord_tol_m" in three_d, \
        "the 3D section does not state the limit for thin layers"
    assert re.search(r"\bthin\b|\bthinner\b", _section(guide, "## Limits of this version")), \
        "Limits does not name thin layers in 3D"
    assert "tessellated" in (ROOT / "llms.txt").read_text("utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text("utf-8")
    # the notes since 0.1.0: "Unreleased" and the release sections above 0.1.0
    notes = changelog.split("## Unreleased", 1)[1].split("\n## 0.1.0", 1)[0]
    assert "tessellated" in notes and "3D" in notes
