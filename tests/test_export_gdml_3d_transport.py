"""A 3D electromagnetic design in Geant4: the mesher's 3D partition (``axmp-mesher --export-regions``) ->
``solids.from_partition`` (tessellated solids) -> ``solids.json`` and GDML -> ``axmp-transport --geometry`` (X2: every
region's volume and mass; X3: Geant4's overlap check).

Needs the transport program (``AXMP_TRANSPORT``; every test skips without it, fails when it is set to something that
is not an executable file, and skips when its ``--check`` does not report ``"geometry": true``) and the mesher. The
route from ``Design.export_gdml`` to the same partition is ``test_export_gdml_3d.py``'s (it needs the 3D engine,
which the transport's build does not have).

Designs: the window-frame iron dipole and the design with every surface kind (``_partition3d_ref``).

Independent references: the triangles' own signed volumes, the closed forms of the region volumes, the densities
stated by hand.

Tolerances: Geant4 against the triangles 1e-9 relative (Geant4 sums facet areas times distances, the reference
triple products: both exact to rounding); Geant4's region volume against the closed form: the report's difference
between the triangulated and the exact volume plus 1e-9 of the volume.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import warnings

import numpy as np
import pytest

import _partition3d_ref as R3
import _partition_ref as R
from _partition_ref import mp  # noqa: F401 (a fixture)

from axionmultiphysics import solids

pytestmark = pytest.mark.px_transport
TIMEOUT = 900
NAMES = list(R3.DESIGNS3)


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


@pytest.fixture(scope="module")
def docs(mp, tmp_path_factory):  # noqa: F811
    out = {}
    for name in NAMES:
        d = tmp_path_factory.mktemp(f"t3_{name}")
        inp = R3.mesher_input_3d(R3.design3(name))
        inp["chord_tol_m"] = R3.TEST_CHORD
        r, part, _ = R.run_mesher(mp, inp, d, "x", export=True)
        assert r.returncode == 0 and part and part.get("ok") is True, (
            f"the 3D export failed: {(part or {}).get('error')!r} {r.stderr[-1500:]}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            doc = solids.from_partition(part, R3.solids_materials3(name), R3.region_materials3(R3.design3(name)))
        out[name] = (part, doc)
    return out


def _geometry(program, doc, path):
    solids.write(doc, path)
    return subprocess.run([program, "--geometry", str(path)], capture_output=True, text=True, timeout=TIMEOUT)


@pytest.mark.parametrize("name", NAMES)
def test_a_3d_design_reaches_geant4_with_the_volumes_of_its_partition(program, docs, name, tmp_path):
    part, doc = docs[name]
    r = _geometry(program, doc, tmp_path / "solids.json")
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout)
    assert out["ok"] is True and out["overlaps"] == [], out.get("overlaps")
    vols = {v["name"]: v for v in out["volumes"]}
    assert set(vols) == {s["name"] for s in doc["solids"]}
    dens = R3.DESIGNS3[name][2]
    exact = R3.exact_volumes(name)
    for s in doc["solids"]:
        g = vols[s["name"]]
        assert g["g4_name"] == s["gdml_name"] and g["material"] == s["material"], (g, s["name"])
        if s["kind"] == "tessellated":
            V = R3.signed_volume(np.array(s["params"]["vertices_m"]), np.array(s["params"]["triangles"]))
            assert abs(g["solid_volume_m3"] / V - 1.0) <= 1e-9, (s["name"], g["solid_volume_m3"], V)
    for region, rep in doc["regions"].items():
        mine = [vols[n] for n in rep["solids"]]
        V = sum(v["volume_m3"] for v in mine)
        M = sum(v["mass_kg"] for v in mine)
        assert all(v["density_kg_m3"] == pytest.approx(dens[rep["material"]], rel=1e-12) for v in mine), region
        assert abs(V / rep["volume_solids_m3"] - 1.0) <= 1e-9, (region, V, rep["volume_solids_m3"])
        assert abs(M / rep["mass_solids_kg"] - 1.0) <= 1e-9, (region, M, rep["mass_solids_kg"])
        bound = abs(rep["volume_solids_m3"] - rep["volume_m3"]) + 1e-9 * exact[region]
        assert abs(V - exact[region]) <= bound + 1e-12 * exact[region], (region, V, exact[region], bound)
    # the GDML file of the same list is valid against Geant4's schema when one is given
    gdml = tmp_path / "x.gdml"
    solids.write_gdml(doc, gdml)
    xsd = os.environ.get("AXMP_GDML_SCHEMA")
    if xsd and shutil.which("xmllint"):
        r = subprocess.run(["xmllint", "--noout", "--schema", xsd, str(gdml)], capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]


def test_an_overlap_of_tessellated_solids_is_refused(program, docs, tmp_path):
    """X3's planted defect: the top coil moved by 3 cm along x, into its pole."""
    _, doc = docs["dipole"]
    bad = copy.deepcopy(doc)
    coil = next(s for s in bad["solids"] if s["region"] == "coil_top")
    coil["params"]["vertices_m"] = [[x - 0.03, y, z] for x, y, z in coil["params"]["vertices_m"]]
    r = _geometry(program, bad, tmp_path / "solids.json")
    assert r.returncode == 3, (r.returncode, r.stderr[-2000:])
    out = json.loads(r.stdout)
    assert out["ok"] is False and out["code"] == "particle.geometry_overlap"
    assert coil["name"] in {o["volume"] for o in out["overlaps"]} | {o["mother"] for o in out["overlaps"]} or \
        any(coil["name"] in o["message"] for o in out["overlaps"]), out["overlaps"]


def _drop(s):
    s["params"]["triangles"].pop()


def _turn(s):
    s["params"]["triangles"] = [t[::-1] for t in s["params"]["triangles"]]


@pytest.mark.parametrize("case", ["open", "inside_out"])
def test_a_tessellated_solid_that_does_not_close_is_refused(program, docs, tmp_path, case):
    """A tessellated solid whose triangles leave an edge used other than once in each direction, or enclose a negative
    volume, is an invalid solids file: exit status 2, nothing on stdout, the solid named on stderr."""
    _, doc = docs["dipole"]
    bad = copy.deepcopy(doc)
    s = next(x for x in bad["solids"] if x["region"] == "tracker_up")
    (_drop if case == "open" else _turn)(s)
    r = _geometry(program, bad, tmp_path / "solids.json")
    assert r.returncode == 2, (r.returncode, r.stderr[-2000:])
    assert r.stdout.strip() == "" and s["name"] in r.stderr, r.stderr[-2000:]
