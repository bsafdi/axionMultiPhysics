"""The solids list in Geant4: each region's volume and mass as Geant4 computes them, and Geant4's overlap check.

Needs the transport program: set ``AXMP_TRANSPORT`` to it (``cmake -DAXMP_BUILD_TRANSPORT=ON``). Without the
variable every test here skips; with it set to something that is not an executable file every test fails. A program
whose ``--check`` document does not report ``"geometry": true`` predates the geometry mode, and the tests skip.

The contract tested (``docs/guides/solids.md`` documents it):

- ``axmp-transport --geometry solids.json`` instantiates the list (one logical volume per solid, placed in its
  mother), runs Geant4's overlap check on every placement (``CheckOverlaps`` with the default resolution and a
  tolerance of zero), and prints one JSON document on stdout:
  ``{"ok": true, "volumes": [{"name", "g4_name", "region", "material", "solid_volume_m3", "volume_m3",
  "density_kg_m3", "mass_kg"}], "materials": [{"name", "g4_name", "density_kg_m3", "mean_excitation_eV", "state",
  "T_K", "pressure_Pa", "elements"}], "overlaps": []}``, where ``name`` and ``material`` are the solids list's
  names and ``g4_name`` the name Geant4 holds (the list's ``gdml_name``; a NIST material keeps its name),
  ``solid_volume_m3`` is ``G4VSolid::GetCubicVolume()`` of the volume's solid, ``volume_m3`` that minus its
  daughters' solids, ``density_kg_m3`` the density of its ``G4Material`` and ``mass_kg = volume_m3 *
  density_kg_m3``; per material, Geant4's mean excitation energy, state ("solid", "liquid", "gas"), temperature,
  pressure and ``elements`` ``{symbol: mass fraction}``; all converted once from Geant4's units on output;
- an overlap refuses the run: a nonzero exit status and ``{"ok": false, "code": "particle.geometry_overlap",
  "error": ..., "overlaps": [{"volume": name, ...}]}``;
- a file that is not a valid solids list, a polycone Geant4 cannot build included, exits with status 2, a message
  on stderr and nothing on stdout.

Tolerances: 1e-9 relative for every solid (Geant4's tube, torus, orb and generic polycone volumes are closed forms;
the residue is the conversion to millimetres and back and the float sums, near 1e-15); the reported relative
difference carries a discretized region from the exact volume to the instantiated one.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess

import pytest

import _solids_ref as ref

pytestmark = pytest.mark.px_transport

FIXTURES = {"native": ref.native, "discretized": ref.discretized, "voids": ref.voids, "degenerate": ref.degenerate}
TIMEOUT = 600


@pytest.fixture(scope="module")
def program():
    path = os.environ.get("AXMP_TRANSPORT")
    if not path:
        pytest.skip("AXMP_TRANSPORT is not set (the transport program is built with -DAXMP_BUILD_TRANSPORT=ON)")
    assert os.path.isfile(path) and os.access(path, os.X_OK), f"AXMP_TRANSPORT={path} is not an executable file"
    r = subprocess.run([path, "--check"], capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, r.stderr[-2000:]
    check = json.loads(r.stdout)
    if check.get("geometry") is not True:
        pytest.skip("this axmp-transport has no geometry mode (its --check does not report \"geometry\": true)")
    return path


def _doc(name, fx=None):
    from axionmultiphysics import solids
    return solids.from_partition(*(fx or FIXTURES[name]()).args())


def _run(program, doc, tmp_path, tag="g"):
    from axionmultiphysics import solids
    path = tmp_path / f"{tag}.json"
    solids.write(doc, path)
    r = subprocess.run([program, "--geometry", str(path)], capture_output=True, text=True, timeout=TIMEOUT)
    return r, json.loads(r.stdout) if r.stdout.strip() else None


@pytest.mark.parametrize("name", list(FIXTURES))
def test_geant4_volumes_and_masses_equal_the_report(program, tmp_path, name):
    doc = _doc(name)
    r, out = _run(program, doc, tmp_path)
    assert r.returncode == 0 and out["ok"] is True, r.stderr[-2000:]
    assert out["overlaps"] == []
    vols = {v["name"]: v for v in out["volumes"]}
    assert set(vols) == {s["name"] for s in doc["solids"]}
    for s in doc["solids"]:
        v = vols[s["name"]]
        assert v["region"] == s["region"] and v["material"] == s["material"]
        assert abs(v["solid_volume_m3"] / ref.solid_volume(s) - 1.0) <= 1e-9, s["name"]
    for region, rep in doc["regions"].items():
        mine = [vols[n] for n in rep["solids"]]
        V = sum(v["volume_m3"] for v in mine)
        M = sum(v["mass_kg"] for v in mine)
        assert all(v["density_kg_m3"] == pytest.approx(rep["density_kg_m3"], rel=1e-12) for v in mine), region
        assert abs(V / rep["volume_solids_m3"] - 1.0) <= 1e-9, (region, V, rep["volume_solids_m3"])
        assert abs(M / rep["mass_solids_kg"] - 1.0) <= 1e-9, (region, M, rep["mass_solids_kg"])
        # against the exact solid of revolution, with the reported deficit (zero for native solids)
        assert abs(V / rep["volume_m3"] - 1.0 - rep["volume_rel_diff"]) <= 1e-9, region


@pytest.mark.parametrize("move", ["daughter_out_of_mother", "siblings_overlap"])
def test_a_planted_overlap_refuses_the_run(program, tmp_path, move):
    doc = copy.deepcopy(_doc("native"))
    b = {s["name"]: s for s in doc["solids"]}
    if move == "daughter_out_of_mother":
        b["ring_0"]["position_m"][2] += 0.25          # the torus pokes out of the top of the air's tube
        culprit = "ring_0"
    else:
        b["insert_0"]["position_m"][2] += 0.22        # the insert runs into the torus beside it
        culprit = "insert_0"
    r, out = _run(program, doc, tmp_path, move)
    assert r.returncode != 0
    assert out is not None and out["ok"] is False and out["code"] == "particle.geometry_overlap"
    assert culprit in json.dumps(out["overlaps"]) or culprit in out["error"]


def test_the_geometry_mode_reads_metres(program, tmp_path):
    """A tube of 1 m radius and 2 m length in the document is a 2 pi m^3 solid in Geant4, not 2 pi mm^3."""
    doc = _doc("native")
    r, out = _run(program, doc, tmp_path)
    w = next(v for v in out["volumes"] if v["name"] == "world")
    p = next(s for s in doc["solids"] if s["name"] == "world")["params"]
    assert w["solid_volume_m3"] == pytest.approx(math.pi * p["rmax_m"] ** 2 * p["z_m"], rel=1e-9)


def test_partial_revolutions_are_read_in_degrees(program, tmp_path):
    """A hand-edited list with partial turns (a torus of 270 degrees from 45, a polycone of 90 from 10, a tube of
    180, a vacuum tube of 120 from 200): Geant4's volumes are those fractions of the full turns. Every list the
    conversion writes is a full turn, where degrees and radians cannot be told apart (360 rad is a full turn too)."""
    edits = {"native": {"ring_0": (45.0, 270.0), "cone_0": (10.0, 90.0), "pin_0": (0.0, 180.0)},
             "voids": {"void_0": (200.0, 120.0)}}
    for name, ed in edits.items():
        doc = copy.deepcopy(_doc(name))
        b = {s["name"]: s for s in doc["solids"]}
        for n, (a, d) in ed.items():
            b[n]["params"]["startphi_deg"], b[n]["params"]["deltaphi_deg"] = a, d
        r, out = _run(program, doc, tmp_path, name)
        assert r.returncode == 0 and out["ok"] is True, r.stderr[-2000:]
        vols = {v["name"]: v for v in out["volumes"]}
        for n in ed:
            assert abs(vols[n]["solid_volume_m3"] / ref.solid_volume(b[n]) - 1.0) <= 1e-9, n


# atomic weights (IUPAC conventional values) for the mass fractions of compositions given by atom counts; Geant4's
# NIST elements average their isotopes, which agrees with these within 1e-4
ATOMIC_WEIGHT = {"H": 1.008, "He": 4.0026, "C": 12.011, "O": 15.999, "F": 18.998}
NIST_I_eV = {"G4_Al": 166.0, "G4_Fe": 286.0, "G4_Cu": 322.0, "G4_Pb": 823.0}   # ICRU 37, Geant4's NIST tables


def test_materials_are_built_as_the_document_states(program, tmp_path):
    """Per material Geant4's density, mean excitation energy, state, temperature, pressure and elements against
    the document: given values exactly (1e-12, two conversions of units); otherwise Geant4's defaults for a
    material the GDML reader builds (state from the density, 293.15 K, 101325 Pa) and, for a NIST material with
    the document's density, the NIST element's excitation energy (1e-9)."""
    native = ref.native()
    native.materials["polyethylene"]["particle"].update(state="liquid", T_K=350.0, pressure_Pa=2.0e5)
    for fx in (native, ref.discretized()):
        doc = _doc(None, fx)
        r, out = _run(program, doc, tmp_path, "m")
        assert r.returncode == 0 and out["ok"] is True, r.stderr[-2000:]
        got = {m["name"]: m for m in out["materials"]}
        assert set(got) == {m["name"] for m in doc["materials"]} | {"G4_Galactic"}
        assert got["G4_Galactic"]["g4_name"] == "G4_Galactic"
        assert got["G4_Galactic"]["density_kg_m3"] == pytest.approx(1e-22, rel=1e-9) and got["G4_Galactic"]["state"] == "gas"
        for m in doc["materials"]:
            g = got[m["name"]]
            assert g["g4_name"] == m["gdml_name"]
            assert g["density_kg_m3"] == pytest.approx(m["density_kg_m3"], rel=1e-12), m["name"]
            assert g["state"] == m.get("state", "solid" if m["density_kg_m3"] > 10.0 else "gas"), m["name"]
            assert g["T_K"] == pytest.approx(m.get("T_K", 293.15), rel=1e-12), m["name"]
            assert g["pressure_Pa"] == pytest.approx(m.get("pressure_Pa", 101325.0), rel=1e-12), m["name"]
            comp = m["composition"]
            if "mean_excitation_eV" in m:
                assert g["mean_excitation_eV"] == pytest.approx(m["mean_excitation_eV"], rel=1e-12), m["name"]
            elif "nist" in comp:
                assert g["mean_excitation_eV"] == pytest.approx(NIST_I_eV[comp["nist"]], rel=1e-9), m["name"]
            else:
                assert g["mean_excitation_eV"] > 0.0
            if "nist" in comp:
                want = {comp["nist"][3:]: 1.0}
            elif comp["by"] == "mass":
                want = comp["elements"]
            else:
                tot = sum(n * ATOMIC_WEIGHT[e] for e, n in comp["elements"].items())
                want = {e: n * ATOMIC_WEIGHT[e] / tot for e, n in comp["elements"].items()}
            assert set(g["elements"]) == set(want), m["name"]
            tol = 1e-12 if "nist" in comp or comp["by"] == "mass" else 1e-3
            for e, w in want.items():
                assert g["elements"][e] == pytest.approx(w, rel=tol), (m["name"], e)


def test_geant4_names_are_the_gdml_names(program, tmp_path):
    """Volumes and materials carry in Geant4 the names the GDML file gives them, so the native path and a GDML
    application name everything alike; the report keeps the solids list's names."""
    for name, (regions, materials) in {
            "native": ({"air": "world", "core": "316L steel", "ring": "a0xb", "insert": "1st", "pin": "void",
                        "cone": "a b"},
                       {"air": "a_b", "iron": "world", "copper": "a b", "steel": "0x1",
                        "polyethylene": "316L steel"}),
            "voids": ({"jacket": "void", "liner": "world", "shell": "void_1"},
                      {"steel": "world_0", "ptfe": "a-b", "lead": "a b"})}.items():
        doc = _doc(None, ref.renamed(FIXTURES[name](), regions, materials))
        r, out = _run(program, doc, tmp_path, name)
        assert r.returncode == 0 and out["ok"] is True, r.stderr[-2000:]
        assert {v["name"]: v["g4_name"] for v in out["volumes"]} == {s["name"]: s["gdml_name"] for s in doc["solids"]}
        # the placements are named as Geant4's GDML reader names them: <volume>_pv, and <world>_PV for the world's
        assert {v["name"]: v["placement"] for v in out["volumes"]} == {
            s["name"]: s["gdml_name"] + ("_PV" if s["mother"] is None else "_pv") for s in doc["solids"]}
        assert {v["name"]: v["material"] for v in out["volumes"]} == {s["name"]: s["material"] for s in doc["solids"]}
        got = {m["name"]: m["g4_name"] for m in out["materials"]}
        assert all(got[m["name"]] == m["gdml_name"] for m in doc["materials"])


def test_an_invalid_polycone_is_an_invalid_file(program, tmp_path):
    """Two vertices of a polycone swapped (a self-crossing outline that Geant4 refuses to build): exit status 2,
    a message on stderr, nothing on stdout, as for any file that is not a valid solids list."""
    doc = copy.deepcopy(_doc("native"))
    rz = next(s for s in doc["solids"] if s["name"] == "cone_0")["params"]["rz_m"]
    rz[1], rz[2] = rz[2], rz[1]
    r, out = _run(program, doc, tmp_path, "bowtie")
    assert r.returncode == 2, (r.returncode, r.stderr[-1000:])
    assert out is None and r.stderr.strip()


@pytest.mark.parametrize("label,rz,code,message", [
    ("crossing", [[0.0, 0.0], [0.2, 0.1], [0.2, -0.1], [0.0, 0.1]], 2, "segments cross"),
    ("collinear", [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], 2, "near zero"),
    ("backtrack", [[0.0, 0.0], [0.2, -0.1], [0.2, 0.1], [0.2, -0.1]], 2, "near zero"),
    ("doubling_back", [[0.0, 0.0], [0.4, 0.0], [0.4, 0.1], [0.2, 0.1], [0.2, 0.0], [0.3, 0.0], [0.3, -0.1],
                       [0.0, -0.1]], 2, "runs back over itself"),
    ("vertices_1e-13_apart", [[0.0, 0.0], [0.2, -0.1], [0.2, -0.1 + 1e-13]], 2, "Too few unique"),
    ("vertices_5e-12_apart", [[0.0, 0.0], [0.2, -0.1], [0.2, -0.1 + 5e-12]], 0, None),
])
def test_polycone_outlines_are_judged_by_geant4(program, tmp_path, label, rz, code, message):
    """Hand-edited outlines of the triangle 'tri_0' of the degenerate fixture. A solid Geant4 refuses to construct
    (a crossing, almost no area, vertices closer than its length tolerance, which it merges, leaving two points) is
    an invalid file: exit status 2, Geant4's message on stderr, nothing on stdout, not a fatal error (exit 4). An
    outline that runs back over itself along a stretch is refused by the program's own check. Vertices 5 pm apart
    are separate points and build."""
    doc = copy.deepcopy(_doc("degenerate"))
    next(s for s in doc["solids"] if s["name"] == "tri_0")["params"]["rz_m"] = rz
    r, out = _run(program, doc, tmp_path, label)
    if code == 2:
        assert r.returncode == 2, (r.returncode, r.stderr[-1000:])
        assert out is None and message in r.stderr and "tri_0" in r.stderr, r.stderr[-1000:]
    else:
        assert r.returncode == 0 and out["ok"] is True, (r.returncode, r.stderr[-1000:])
