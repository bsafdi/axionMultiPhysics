"""An axisymmetric electromagnetic design in Geant4: design -> ``Design.export_gdml`` (``solids.json`` and GDML) ->
``axmp-transport --geometry`` (X2: every region's volume and mass; X3: Geant4's overlap check).

Needs the transport program (``AXMP_TRANSPORT``; every test skips without it, fails when it is set to something that
is not an executable file, and skips when its ``--check`` does not report ``"geometry": true``) and the mesher.

Designs: a solenoid in an iron pot yoke with a lead absorber on the axis; every curve of the 2D shape grammar (a
torus, fillets, a contour's half ellipse, a sphere, a disk cut by a plate, sheets); the CORE template with its iron
caps.

Independent references: Geant4's own solid volumes (closed forms for tubes, tori, orbs and generic polycones); the
solids report of the conversion; the EM mesh of the same design (the integral of 2 pi r over each region's P2
triangles, ``_partition_ref.region_volumes``); the densities stated by hand.

Tolerances: Geant4 against the report 1e-9 (as ``test_solids_transport.py``: closed forms, the conversion to mm and
back, float sums); Geant4 against the mesh: the reported difference between the instantiated and the exact solid
(discretized arcs) plus the P2 boundary bound of ``test_mesher_partition.py`` (10 times the gap estimate; measured at
most 2.4 times on these designs) plus 1e-9 of the volume.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings
import xml.etree.ElementTree as ET

import pytest

import axionmultiphysics as axmp
import _partition_ref as R
from _partition_ref import mp  # noqa: F401 (a fixture)

pytestmark = pytest.mark.px_transport
TIMEOUT = 600
GAP_FACTOR = 10.0


@pytest.fixture(scope="module")
def program():
    path = os.environ.get("AXMP_TRANSPORT")
    if not path:
        pytest.skip("AXMP_TRANSPORT is not set (the transport program is built with -DAXMP_BUILD_TRANSPORT=ON)")
    assert os.path.isfile(path) and os.access(path, os.X_OK), f"AXMP_TRANSPORT={path} is not an executable file"
    r = subprocess.run([path, "--check"], capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, r.stderr[-2000:]
    if json.loads(r.stdout).get("geometry") is not True:
        pytest.skip("this axmp-transport has no geometry mode (its --check does not report \"geometry\": true)")
    return path


@pytest.mark.parametrize("name", list(R.DESIGNS))
def test_a_design_reaches_geant4_with_the_volumes_of_its_mesh(mp, program, tmp_path, name):
    design = R.design(name)
    _, comp, dens = R.DESIGNS[name]
    gdml, sj = tmp_path / f"{name}.gdml", tmp_path / f"{name}.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = axmp.Design(design).export_gdml(gdml, composition=comp, solids=sj)
    r = subprocess.run([program, "--geometry", str(sj)], capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert out["ok"] is True and out["overlaps"] == [], out.get("overlaps")
    vols = {v["name"]: v for v in out["volumes"]}
    assert set(vols) == {s["name"] for s in doc["solids"]}
    _, msh, _ = R.em_mesh(design, tmp_path / "mesh")
    mesh_vols = R.region_volumes(R.Msh(msh))
    assert set(doc["regions"]) == set(mesh_vols) == set(R.region_materials(design))
    region_materials = R.region_materials(design)
    for region, rep in doc["regions"].items():
        assert rep["material"] == region_materials[region], (region, rep["material"])
        mine = [vols[n] for n in rep["solids"]]
        assert all(v["material"] == rep["material"] for v in mine), region
        V = sum(v["volume_m3"] for v in mine)
        M = sum(v["mass_kg"] for v in mine)
        assert all(v["density_kg_m3"] == pytest.approx(dens[rep["material"]], rel=1e-12) for v in mine), region
        assert abs(V / rep["volume_solids_m3"] - 1.0) <= 1e-9, (region, V, rep["volume_solids_m3"])
        assert abs(M / rep["mass_solids_kg"] - 1.0) <= 1e-9, (region, M, rep["mass_solids_kg"])
        V_mesh, gap = mesh_vols[region]
        bound = abs(rep["volume_solids_m3"] - rep["volume_m3"]) + 1e-9 * V + GAP_FACTOR * gap
        assert abs(V - V_mesh) <= bound, f"{region}: Geant4 {V}, mesh {V_mesh} (bound {bound:.3g})"
    # the GDML file: one volume per solid, by its GDML name; valid against Geant4's schema when one is given
    root = ET.parse(gdml).getroot()
    names = {v.get("name") for v in root.iter("volume")}
    assert {s["gdml_name"] for s in doc["solids"]} == names
    xsd = os.environ.get("AXMP_GDML_SCHEMA")
    if xsd and shutil.which("xmllint"):
        r = subprocess.run(["xmllint", "--noout", "--schema", xsd, str(gdml)], capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 0, r.stderr[-3000:]
