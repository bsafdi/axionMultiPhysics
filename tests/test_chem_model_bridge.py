"""The bridge from an instance mesh to the chemistry engine, ``im.chem_mesh(order=2)`` and ``m.chem_mesh(domain=None,
order=2)``, and its acceptance tests (design note 5.4, 12.1, and the items "Geometry order" and "Bridge, sequential
build" of the acceptance list of bounded 2D meshes in 18); every build.

``im.chem_mesh(order)`` takes the FIELD order of the current distribution and returns a ``chem.TriMesh`` (a subclass)
with the instance mesh's vertices, triangles, element attributes, boundary segments and dense boundary attributes in
its order; it records ``geometry_order``, and its ``edge_nodes()`` gives the exporter's P2 edge nodes (those off their
chord's midpoint: none on a straight mesh, the circle points on a curved one). A field order 1 on an instance mesh with
nodes is refused (a curved mesh is not straightened silently), and so is ``CurrentDistribution(mesh, order=1)`` on a
mesh that records geometry order 2; both refusals are ValueErrors (a misuse of the API) whose message begins with the
exporter's text. ``import axionmultiphysics`` imports neither the chemistry nor the fluids package; the exporters
import them when called.

The parallel-plate case has a case file with its tolerance record (``cpp/tests/data/model/u2/cases/plate_cell.json``);
``test_the_tolerance_records_follow_the_rule`` checks every case file of the bridge tests, the fluids ones included.

Independent references: the parallel-plate closed form (exact in P2), the analytic circles of the annulus and the
length error of a P2 arc, the exporter's message (``cpp/src/model_errors.hpp``, ``mesh_export_nodes_for_p1``).
Seconds.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import axionmultiphysics as axmp

import model_bridge_cases as C
import model_mesh_support as S
from model_mesh_support import mesher  # noqa: F401 (a fixture)

R1, R2 = 0.002, 0.005
P1_REFUSAL = "chemistry mesh export: the instance mesh of 'ring' has nodes (geometry order 2), and the field order is 1"


@pytest.fixture
def chem(mesher):
    try:
        from axionmultiphysics._core import chem as _c  # noqa: F401
    except ImportError:
        pytest.skip("axionmultiphysics._core has no chem submodule (rebuild)")
    return C.chem()


def test_import_axionmultiphysics_imports_neither_engine(mesher, tmp_path):
    """``import axionmultiphysics``, ``axmp.mesh`` and ``m.instance()`` import neither ``axionmultiphysics.chem`` nor
    ``axionmultiphysics.fluids``; ``im.chem_mesh()`` imports the chemistry package (in a fresh interpreter)."""
    doc = S.write(tmp_path, S.doc("plate_cell"))
    code = ("import json, sys, warnings\nwarnings.simplefilter('ignore')\nimport axionmultiphysics as axmp\n"
            f"m = axmp.mesh(json.load(open({str(doc)!r})))\nim = m.instance()\n"
            "names = ('axionmultiphysics.chem', 'axionmultiphysics.fluids')\n"
            "print(json.dumps([n in sys.modules for n in names]))\n"
            "im.chem_mesh()\nprint(json.dumps('axionmultiphysics.chem' in sys.modules))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert json.loads(lines[-2]) == [False, False], r.stdout
    assert json.loads(lines[-1]) is True, r.stdout


def test_chem_mesh_is_the_instance_mesh(chem):
    """The chemistry mesh of the plate cell (geometry order 2, straight edges): a ``TriMesh`` with the instance mesh's
    arrays in its order, ``geometry_order`` 2, no edge node; ``m.chem_mesh`` is the shorthand; an order other than 1 or
    2 is a ValueError."""
    m = axmp.mesh(S.doc("plate_cell"))
    assert m.geometry_order == 2
    im = m.instance()
    for cm in (im.chem_mesh(), im.chem_mesh(order=2), m.chem_mesh(), m.chem_mesh(None, order=2), m.chem_mesh("bulk")):
        assert isinstance(cm, chem.TriMesh), type(cm)
        assert cm.geometry_order == 2
        assert np.array_equal(cm.vertices(), im.vertices())
        assert np.array_equal(np.asarray(cm.T), im.elements())
        assert np.array_equal(np.asarray(cm.eattr), im.element_attributes())
        assert np.array_equal(np.asarray(cm.B), im.boundary())
        assert np.array_equal(np.asarray(cm.battr), im.boundary_attributes())
        ev, exy = cm.edge_nodes()
        assert np.shape(ev) == (0, 2) and np.shape(exy) == (0, 2), "a straight mesh has no edge node"
    for bad in (0, 3):
        with pytest.raises(ValueError):
            im.chem_mesh(bad)


def test_parallel_plate_cell_through_the_bridge(chem):
    """Bridge, every build: the parallel-plate cell through ``m.chem_mesh()`` with ``chem.CurrentDistribution`` gives
    the closed form, I = kappa dV Hc / gap per metre, and the linear potential, by the tolerance rule (ceiling 1e-9)."""
    case = C.load("plate_cell")
    run = C.measure_plate_cell(case)
    assert run["report"]["converged"], run["report"]
    C.check_tolerances(case, run["quantities"])


def test_annulus_edge_nodes_lie_on_the_circles(chem):
    """The annulus at geometry order 2: the edge nodes ``chem_mesh(2)`` passes to ``CurrentDistribution`` lie on the
    two circles to 1e-12 R, each on an edge whose two vertices are on that circle; the current distribution then
    integrates over the curved boundary: its length of the inner circle is the P2 arc's (relative error at most
    alpha^4/480 for segments of angle up to alpha, twice the leading term alpha^4/960), not the polygon's (alpha^2/24)."""
    m = axmp.mesh(S.doc("annulus"))
    im = m.instance()
    assert im.geometry_order == 2 and im.has_nodes is True
    cm = im.chem_mesh(2)
    assert cm.geometry_order == 2
    ev, exy = (np.asarray(a) for a in cm.edge_nodes())
    assert len(ev) > 0 and ev.shape == (len(exy), 2)
    V = cm.vertices()
    for (a, b), p in zip(ev, exy):
        r = math.hypot(*p)
        R = R1 if abs(r - R1) < abs(r - R2) else R2
        assert abs(r - R) <= 1e-12 * R, (p, r)
        assert abs(math.hypot(*V[a]) - R) <= 1e-12 * R and abs(math.hypot(*V[b]) - R) <= 1e-12 * R
    # the inner circle's segments: their angles and chord length
    inner = np.asarray(cm.B)[np.isin(np.asarray(cm.battr), im.attributes["inner"])]
    A, B = V[inner[:, 0]], V[inner[:, 1]]
    alpha = np.abs(np.arctan2(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0], np.sum(A * B, axis=1)))
    assert abs(alpha.sum() - 2 * math.pi) <= 1e-12, "the segments go once around the inner circle"
    polygon = np.linalg.norm(B - A, axis=1).sum()
    bound = alpha.max() ** 4 / 480
    assert abs(polygon / (2 * math.pi * R1) - 1) > 10 * bound, "the mesh tells a curved length from a straight one"
    cd = chem.CurrentDistribution(cm, order=2)
    cd.set_conductivity({a: 1.0 for a in im.region_attribute.values()})
    cd.add_electrode("inner", im.attributes["inner"], primary=True, phi_s=1.0)
    cd.add_electrode("outer", im.attributes["outer"], primary=True, phi_s=0.0)
    cd.solve()
    w = np.asarray(cd.boundary_quadrature(im.attributes["inner"], 8)["w"])
    assert abs(w.sum() / (2 * math.pi * R1) - 1) <= bound, (w.sum() / (2 * math.pi * R1) - 1, bound)


def test_p1_on_a_mesh_with_nodes_is_refused(chem):
    """On the annulus at geometry order 2, ``im.chem_mesh(1)`` and ``CurrentDistribution(im.chem_mesh(2), order=1)``
    are refused with the exporter's text (a ValueError: a misuse of the API, not a document error); so is
    ``chem_mesh(1)`` of the straight plate cell, whose instance mesh has nodes too."""
    im = axmp.mesh(S.doc("annulus")).instance()
    with pytest.raises(ValueError) as e:
        im.chem_mesh(1)
    assert str(e.value).startswith(P1_REFUSAL), str(e.value)
    cm = im.chem_mesh(2)
    with pytest.raises(ValueError) as e:
        chem.CurrentDistribution(cm, order=1)
    assert str(e.value).startswith(P1_REFUSAL), str(e.value)
    plate = axmp.mesh(S.doc("plate_cell"))
    with pytest.raises(ValueError) as e:
        plate.chem_mesh(order=1)
    assert str(e.value).startswith("chemistry mesh export: the instance mesh of 'bulk' has nodes (geometry order 2), "
                                   "and the field order is 1"), str(e.value)


def test_a_p1_annulus_cell_through_the_bridge(chem):
    """The chemistry half of "Geometry order": the annulus with a P1 current distribution is meshed at geometry order
    1 (the default lowered); its chemistry mesh records order 1, has no edge node, and a P1 ``CurrentDistribution``
    accepts it and solves."""
    m = axmp.mesh(S.doc("annulus_cell"))
    assert m.geometry_order == 1
    im = m.instance()
    assert im.has_nodes is False
    cm = im.chem_mesh(1)
    assert cm.geometry_order == 1
    ev, exy = cm.edge_nodes()
    assert np.shape(ev) == (0, 2) and np.shape(exy) == (0, 2)
    cd = chem.CurrentDistribution(cm, order=1)
    cd.set_conductivity({a: 1.0 for a in im.region_attribute.values()})
    cd.add_electrode("inner", im.attributes["inner"], primary=True, phi_s=0.05)
    cd.add_electrode("outer", im.attributes["outer"], primary=True, phi_s=0.0)
    assert cd.solve()["converged"]


def test_the_tolerance_records_follow_the_rule():
    """Every case file of the bridge tests: its quantities have the note's ceilings (1e-8 for a field of a Stokes
    solve, 1e-9 otherwise), and each record is either empty (asserted null) or follows the rule asserted = max(10 x
    the largest measured value, 1e-12), at most the ceiling (a number edited by hand breaks it)."""
    files = sorted(C.CASES.glob("*.json"))
    assert sorted(p.stem for p in files) == sorted(C.CASE_NAMES), files
    for p in files:
        case = json.loads(p.read_text("utf-8"))
        assert case["name"] == p.stem and case["build"] == C.CASE_NAMES[p.stem]
        assert set(case) >= {"name", "test", "description", "build", "mesh", "instance", "solver", "data", "tolerance",
                             "recorded", "unrecorded_platforms", "document"}, p.name
        assert "direct_tol" in case["solver"] or "rtol" in case["solver"] or "newton" in case["solver"], p.name
        for q, t in case["tolerance"].items():
            assert set(t) == {"what", "ceiling", "measured", "asserted"}, (p.name, q)
            assert t["ceiling"] in (1e-8, 1e-9), (p.name, q, t["ceiling"])
            if t["asserted"] is None:
                assert t["measured"] == {}, (p.name, q)
                continue
            assert t["measured"], (p.name, q)
            assert all(isinstance(v, float) and v >= 0 for v in t["measured"].values()), (p.name, q)
            assert t["asserted"] == C.asserted_value(t["measured"]), (p.name, q, t)
            assert t["asserted"] <= t["ceiling"], (p.name, q, t)


def test_chem_mesh_of_a_two_region_instance(mesher):
    """split_rect, regions a and b: ``m.chem_mesh(domain)`` is the instance of that domain, the element attributes are
    the regions', and a series conductor (kappa by element attribute, primary electrodes on hot and cold) carries
    I = dV H / ((L/2)/k_a + (L/2)/k_b) per metre (phi piecewise linear, the interface a mesh line: exact in P2)."""
    try:
        from axionmultiphysics._core import chem as _c  # noqa: F401
    except ImportError:
        pytest.skip("no chem submodule")
    chem = C.chem()
    L, H, dV, ka, kb = 0.04, 0.01, 0.1, 1.0, 3.0
    m = axmp.mesh(S.doc("split_rect"))
    for dom in ("a", "b", None):
        im, cm = m.instance(dom), m.chem_mesh(dom)
        assert np.array_equal(np.asarray(cm.T), im.elements()) and np.array_equal(np.asarray(cm.battr), im.boundary_attributes())
        assert np.array_equal(np.asarray(cm.eattr), im.element_attributes())
    im = m.instance()
    cm = m.chem_mesh()
    ra = im.region_attribute
    assert set(np.asarray(cm.eattr).tolist()) == {ra["a"], ra["b"]}
    cd = chem.CurrentDistribution(cm, order=2)
    cd.set_conductivity({ra["a"]: ka, ra["b"]: kb})
    cd.add_electrode("hot", im.attributes["hot"], primary=True, phi_s=dV)
    cd.add_electrode("cold", im.attributes["cold"], primary=True, phi_s=0.0)
    assert cd.solve()["converged"]
    I0 = dV * H / (0.5 * L / ka + 0.5 * L / kb)
    q = cd.boundary_quadrature(im.attributes["hot"], 6)
    I = float(np.sum(np.asarray(q["w"]) * ka * np.sum(np.asarray(q["grad"]) * np.asarray(q["normal"]), axis=1)))
    assert abs(I - I0) <= 1e-10 * I0, (I, I0)
