"""The bridge from an instance mesh to the fluids engines, ``im.fluids_mesh()`` and ``m.fluids_mesh(domain)``, and the
engine acceptance tests of bounded 2D meshes that hand an instance mesh to ``solve_heat``, ``solve_boussinesq``
and ``solve_stokes`` (design note 5.4, 12.1 and the acceptance list of bounded 2D meshes in 18).

``im.fluids_mesh()`` returns an ``axionmultiphysics.fluids.Mesh`` that is a copy of the instance mesh: the same
vertices, triangles, boundary segments and dense boundary attributes in the same order, and its nodes exactly when the
instance mesh has them (``Mesh.has_nodes``, a new read-only property). In a build without the fluids module it raises
``AxionMultiPhysicsError`` with the code ``build.requires_fluids`` (the test of that runs in such a build only).
``m.fluids_mesh(domain=None)`` is ``m.instance(domain).fluids_mesh()``.

The acceptance cases are solved through the bridge with conditions on the named attributes, the remaining attributes
and the axis attributes only (no padding, no master number). Each exactness case has a case file,
``cpp/tests/data/model/u2/cases/<name>.json``, with its document, mesh, solver options and tolerance record; the
measurement is ``tests/model_bridge_cases.py``, shared with ``scripts/model_bridge_tolerances.py``, which writes the
records. Every flow case checks first that no triangle has three boundary vertices (the Taylor-Hood check), so that it
tests what it says.

Independent references: the closed forms of the slab, of plane, two-layer and Hagen-Poiseuille flow (each exactly
representable in the discrete space, with the Laplacian form where the flow has an open end), the prescribed mean
inflow for the planar fluxes, the analytic flux of the pipe, the same solve on a copy of the mesh, and the engines'
refusal texts (``scalar_transport.cpp``: SUPG on curved elements, ``add_bc``, ``finalize``; ``stokes.cpp``: an attribute
no boundary element carries). MPI builds; one process; each test runs in a few seconds.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import axionmultiphysics as axmp
from axionmultiphysics import _core

import model_bridge_cases as C
import model_mesh_support as S
from model_mesh_support import mesher  # noqa: F401 (a fixture)


def _has_fluids() -> bool:
    try:
        C.fluids()
    except ImportError:
        return False
    return True


@pytest.fixture
def fluids(mesher):
    """The fluids package (MPI builds); the test skips in a build without it."""
    if not _has_fluids():
        pytest.skip("this build has no fluids module (an MPI build has)")
    return C.fluids()


def _doc(name: str, **mesh) -> dict:
    d = S.doc(name)
    d["mesh"] = dict(d["mesh"], **mesh)
    return d


# ------------------------------------------------------------------------------------------------- the bridge
def test_fluids_mesh_is_the_instance_mesh_copied(fluids):
    """The fluids mesh of an instance has the instance mesh's arrays in its order, its nodes (geometry order 2) or
    none (order 1); ``m.fluids_mesh(domain)`` is the shorthand. Once for this file: the solution's geometry at the
    elements and reference points of ``im.face_points`` is the face points (the engine's mesh is the instance mesh,
    with its element order, in one process)."""
    m = axmp.mesh(S.doc("split_rect"))
    for dom in (None, "a", ["a", "b"]):
        im = m.instance(dom)
        fm = im.fluids_mesh()
        assert isinstance(fm, fluids.Mesh), type(fm)
        assert fm.has_nodes is True and im.has_nodes is True, dom
        assert fm.n_elements == len(im.elements())
        assert np.array_equal(fm.vertices(), im.vertices()), dom
        assert np.array_equal(fm.elements(), im.elements()), dom
        seg, battr = fm.boundary()
        assert np.array_equal(seg, im.boundary()) and np.array_equal(battr, im.boundary_attributes()), dom
        # P2 geometry: the first three nodes of every triangle are its vertices; straight edges: the others are the
        # chord midpoints
        p2, V, E = fm.p2_nodes(), im.vertices(), im.elements()
        assert np.abs(p2[:, :3] - V[E]).max() <= 1e-15 * 0.04
        mids = 0.5 * (V[E] + V[np.roll(E, -1, axis=1)])
        assert np.abs(p2[:, 3:] - mids).max() <= 1e-12 * 0.04
    short = m.fluids_mesh("a")
    assert isinstance(short, fluids.Mesh) and np.array_equal(short.elements(), m.instance("a").elements())
    assert np.array_equal(m.fluids_mesh().elements(), m.instance().elements())
    # geometry order 1: no nodes on either side
    m1 = axmp.mesh(_doc("split_rect", geometry_order=1))
    assert m1.geometry_order == 1
    im1 = m1.instance()
    assert im1.has_nodes is False and im1.fluids_mesh().has_nodes is False
    # the solution's geometry is the face points, on the boundary, on a section and on an unnamed atom
    im = m.instance()
    fm = im.fluids_mesh()
    sol = fluids.solve_heat(fm, 1.0, bcs=[dict(attributes=list(range(1, len(im.master_attribute) + 1)),
                                               kind="dirichlet", value=1.0)])
    assert sol.diagnostics["converged"]
    for name in ("hot", "interface", "mk"):
        assert C.max_face_point_offset(sol, im, name) <= 1e-15 * 0.04, name
    fp = im.face_points(attribute=im.remaining_attributes[0], order=4)
    for i, e in enumerate(fp["elements"]):
        x, _ = sol.geometry(np.array([e], dtype=np.int32), np.ascontiguousarray(fp["ref"][i]))
        assert np.abs(x[0] - fp["points"][i]).max() <= 1e-15 * 0.04


def test_the_fluids_mesh_owns_its_data(fluids):
    """The fluids mesh is a copy: it outlives the instance mesh and the meshed model."""
    import gc

    m = axmp.mesh(S.doc("pipe_axi"))
    im = m.instance("pipe")
    V = im.vertices().copy()
    fm = im.fluids_mesh()
    del m, im
    gc.collect()
    _ = [np.ones(5000) for _ in range(200)]
    assert np.array_equal(fm.vertices(), V) and fm.has_nodes is True


def test_fluids_mesh_in_a_build_without_the_fluids_module(mesher):
    """A build without the fluids module (the sequential build, a wheel): ``im.fluids_mesh()`` and
    ``m.fluids_mesh()`` raise AxionMultiPhysicsError with ``build.requires_fluids``, its message and its hint."""
    if _has_fluids():
        pytest.skip("this build has the fluids module")
    m = axmp.mesh(S.doc("split_rect"))
    want = {"code": "build.requires_fluids", "where": "",
            "error": "the fluids engines are not in this build of axionMultiPhysics (they need an MPI build)",
            "hint": "A wheel from PyPI and the sequential source build have no fluids module; install the MPI source "
                    "build (docs/guides/install.md, \"MPI source build\")."}
    S.assert_exception_is(S.raises(m.instance().fluids_mesh), want)
    S.assert_exception_is(S.raises(m.fluids_mesh, "a"), want)


# ------------------------------------------------------------------------------------------------- heat
def test_heat_slab_through_named_and_remaining(fluids):
    """Heat through the bridge: on the instance mesh of every region of the
    split rectangle, ``solve_heat`` with a fixed temperature on ``hot``, ``robin`` on ``cold`` and ``zero`` on the
    remaining attributes (the unnamed top and bottom: the cavity's shape) gives the slab; the conditions are on the
    named and remaining attributes only. ``solve_boussinesq`` on the same instance mesh, walls everywhere and two fixed
    temperatures, ``t_order = 2``, constructs and converges."""
    case = C.load("heat_slab")
    run = C.measure_heat_slab(case)
    C.check_tolerances(case, run["quantities"])
    im, fm = run["im"], run["fm"]
    D = case["data"]
    every = list(range(1, len(im.master_attribute) + 1))
    sol = fluids.solve_boussinesq(
        fm, mu=1.0e-3, rho0=998.0, cp=4180.0, k=D["k"], beta_g=1.0e-5, T_ref=300.0, gravity="-y",
        velocity_bcs=C.bc(every, "dirichlet", value=(0.0, 0.0)),
        temperature_bcs=[dict(attributes=im.attributes["hot"], kind="dirichlet", value=310.0),
                         dict(attributes=im.attributes["cold"], kind="dirichlet", value=290.0),
                         dict(attributes=im.remaining_attributes, kind="zero")],
        t_order=2, newton_rtol=1e-12, direct_tol=1e-10)
    assert sol.diagnostics["converged"], sol.diagnostics


def test_no_nodes_at_geometry_order_1(fluids):
    """At geometry order 1 neither the instance mesh nor its fluids mesh has nodes, on a rectangle and on an annulus
    (whose boundary is then polygonal), and ``solve_heat(..., supg=True)`` with a callable velocity is accepted; on the
    annulus at geometry order 2 the same call is refused with the engine's text."""
    def supg_solve(im):
        names = {"split_rect": ("hot", "cold"), "annulus": ("inner", "outer")}
        a, b = names["split_rect"] if "hot" in im.attributes else names["annulus"]
        bcs = [dict(attributes=im.attributes[a], kind="dirichlet", value=1.0),
               dict(attributes=im.attributes[b], kind="dirichlet", value=0.0)]
        if im.remaining_attributes:
            bcs.append(dict(attributes=im.remaining_attributes, kind="zero"))
        return fluids.solve_heat(im.fluids_mesh(), 1.0e-4, capacity=1.0,
                                 velocity=lambda X, Y: (np.full_like(X, 1.0e-3), np.zeros_like(X)), supg=True, bcs=bcs)

    for name in ("split_rect", "annulus"):
        m = axmp.mesh(_doc(name, geometry_order=1))
        im = m.instance()
        assert m.geometry_order == 1 and im.geometry_order == 1, name
        assert im.has_nodes is False, name
        fm = im.fluids_mesh()
        assert fm.has_nodes is False, name
        sol = supg_solve(im)
        assert sol.diagnostics["converged"], (name, sol.diagnostics)
    curved = axmp.mesh(S.doc("annulus")).instance()
    assert curved.geometry_order == 2 and curved.has_nodes is True and curved.fluids_mesh().has_nodes is True
    with pytest.raises(_core.AxionMultiPhysicsError, match=r"SUPG on isoparametric \(curved\) elements is not offered"):
        supg_solve(curved)


# ------------------------------------------------------------------------------------------------- two regions
def test_two_regions_one_fluid_enclosed(fluids):
    """(i) Stokes in both regions of the split rectangle, enclosed by a moving lid and walls: the engine constructs,
    reports an enclosed flow with its pressure nullspace, converges, and its velocity and zero-mean pressure are those
    of the same mesh given one element attribute, by the tolerance rule (the interface and the marker are interior:
    no boundary element on them)."""
    case = C.load("two_regions_enclosed")
    run = C.measure_two_regions_enclosed(case)
    im = run["im"]
    assert im.attributes["interface"] == [] and im.attributes["mk"] == []
    d = run["sol"].diagnostics
    assert d["converged"] and d["bc_enclosed"] is True and d["nullspace_detected"] is True, d
    assert run["sol1"].diagnostics["converged"]
    C.check_tolerances(case, run["quantities"])


def test_two_regions_one_fluid_open(fluids):
    """(ii) An inlet on region a, an outlet on region b, Laplacian form, ``do_nothing``: the engine reports an open
    flow with no nullspace and gives plane Poiseuille flow; the flux through the interface section (+e_x, from a to b)
    is the prescribed U H, the flux through the inlet (outward normal) -U H, and the two cancel."""
    case = C.load("two_regions_open")
    run = C.measure_two_regions_open(case)
    d = run["sol"].diagnostics
    assert d["converged"] and d["bc_enclosed"] is False and d["nullspace_detected"] is False, d
    C.check_tolerances(case, run["quantities"])


def test_two_regions_one_fluid_two_layers(fluids):
    """(iii) Two viscosities, ratio 4, in two regions with the interface along the flow (a Python callable of
    position), the two-layer inflow profile as a Python callable, Laplacian form: the two-layer Poiseuille flow."""
    case = C.load("two_layer")
    D = case["data"]
    assert D["nu_upper"] == 4 * D["nu_lower"]
    run = C.measure_two_layer(case)
    assert run["im"].attributes["mid"] == [], "the interface between the layers is interior"
    d = run["sol"].diagnostics
    assert d["converged"] and d["bc_enclosed"] is False and d["nullspace_detected"] is False, d
    C.check_tolerances(case, run["quantities"])


def test_a_number_is_only_a_label(fluids):
    """Case (ii) solved again on a copy of the instance mesh's arrays with the master numbers put back through
    ``im.master_attribute`` (the flow engines accept holes): the same velocity and pressure by the tolerance rule
    (bitwise is expected and recorded, not asserted)."""
    case = C.load("number_is_a_label")
    run = C.measure_number_is_a_label(case)
    assert run["sol2"].diagnostics["converged"]
    assert min(b["attribute"] for b in run["bcs"]) >= 900, "the copy's conditions are keyed by master numbers"
    C.check_tolerances(case, run["quantities"])


def test_master_numbers_are_refused_by_both_engine_families(fluids):
    """Risk 19, the loud half: conditions numbered from ``im.master_attribute`` on the dense instance mesh are refused
    by the transport operator (``add_bc``) and by the Stokes solver, not solved."""
    m = axmp.mesh(S.doc("split_rect"))
    im = m.instance()
    fm = im.fluids_mesh()
    masters = [im.master_attribute[a] for a in sorted(im.master_attribute)]
    first = masters[0]
    with pytest.raises(_core.AxionMultiPhysicsError,
                       match=f"^boundary attribute {first} is not a boundary attribute of the mesh"):
        fluids.solve_heat(fm, 1.0, bcs=[dict(attribute=a, kind="dirichlet", value=1.0) for a in masters])
    with pytest.raises(_core.AxionMultiPhysicsError,
                       match=f"^boundary condition for attribute {first}, which no boundary element of the mesh carries"):
        fluids.solve_stokes(fm, 1.0e-6, bcs=C.bc(masters, "dirichlet", value=(0.0, 0.0)), form="laplacian")


def test_flow_around_a_post(fluids):
    """A flow whose domain is the channel around a solid post (a region that is not in the instance): the instance
    mesh has no element of the post (the triangles cover the channel less the polygon of the cylinder), the cylinder
    carries the atom of the named boundary ``cylinder``, and Stokes with a wall on it converges."""
    L, H, R, xc, yc = 0.04, 0.01, 0.002, 0.01, 0.005
    m = axmp.mesh(S.doc("channel_post"))
    im = m.instance("fluid")
    assert im.regions == ["channel"] and im.region_attribute == {"channel": 1}
    assert set(im.element_attributes().tolist()) == {1}
    cyl = im.attributes["cylinder"]
    assert cyl and not set(cyl) & set(im.attributes["walls"]), im.attributes
    V, E, B, BA = im.vertices(), im.elements(), im.boundary(), im.boundary_attributes()
    seg = B[np.isin(BA, cyl)]
    assert np.abs(np.hypot(V[seg][..., 0] - xc, V[seg][..., 1] - yc) - R).max() <= 1e-9 * R
    # the polygon of the cylinder's vertices: a triangle per segment with the centre (it is convex about the centre)
    a, b = V[seg[:, 0]] - (xc, yc), V[seg[:, 1]] - (xc, yc)
    polygon = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
    t = V[E]
    u, v = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()
    assert abs(area - (L * H - polygon)) <= 1e-12 * L * H, (area, L * H - polygon)
    assert polygon > 0.95 * math.pi * R * R
    fm = C.flow_mesh(im)
    prof = lambda x, y: (6.0 * 0.01 * y * (H - y) / H ** 2, 0.0)  # noqa: E731
    bcs = C.bc(im.attributes["inlet"], "dirichlet", value=prof) + C.bc(im.attributes["outlet"], "do_nothing") + \
        C.bc(im.attributes["walls"] + cyl, "dirichlet", value=(0.0, 0.0))
    sol = fluids.solve_stokes(fm, 1.0e-6, bcs=bcs, form="laplacian", direct_tol=1e-10)
    d = sol.diagnostics
    assert d["converged"] and d["bc_enclosed"] is False, d


# ------------------------------------------------------------------------------------------------- axisymmetric
def test_axisymmetric_pipe_axis(fluids):
    """Axisymmetric, two regions, the axis: the instance mesh of ``pipe`` = [p1, p2] has the axis attribute [1] (the
    master 900 sorts first) on every face on r = 0 of both regions and no boundary element on the interface between
    them; through the bridge, Stokes with the axis condition on ``im.axis_attributes``, the Laplacian form, a
    traction inlet and a do_nothing outlet reproduces Hagen-Poiseuille flow in velocity, pressure and flux."""
    case = C.load("pipe_axi")
    D = case["data"]
    m = axmp.mesh(case["document"])
    im = m.instance(case["instance"])
    # (1) the count, first: the engine would not refuse a free axis
    assert im.axis_attributes == [1] and im.master_attribute[1] == 900, (im.axis_attributes, im.master_attribute)
    count = C.axis_count(im, D["R"], D["Lz"])
    assert abs(count["length"] - D["Lz"]) <= 1e-12 * D["Lz"], count
    assert abs(count["lower"] - D["Lz"] / 2) <= 1e-12 * D["Lz"] and abs(count["upper"] - D["Lz"] / 2) <= 1e-12 * D["Lz"], count
    assert count["max_r"] <= 1e-12 * D["R"] and count["interface_segments"] == 0, count
    # (2) to (4)
    run = C.measure_pipe_axi(case)
    d = run["sol"].diagnostics
    assert d["converged"] and d["bc_enclosed"] is False and d["nullspace_detected"] is False, d
    C.check_tolerances(case, run["quantities"])


# ------------------------------------------------------------------------------------------------- the channel
def test_poiseuille_both_forms(fluids):
    """Bridge: ``axmp.mesh(channel)`` with the parabolic inflow, do_nothing at the outlet and no-slip walls. The
    Laplacian form reproduces plane Poiseuille flow by the tolerance rule. The stress form is not an exactness test:
    it differs from Poiseuille flow by more than 1e-3 of the mean velocity within H of the outlet, and that difference
    decays upstream, its maximum over x < L - 2H at least 100 times below its maximum over x < L - H (the slowest
    symmetric Stokes end mode gives 1.8e3; the case file names both end modes)."""
    case = C.load("poiseuille_channel")
    run = C.measure_poiseuille_channel(case)
    assert run["sol"].diagnostics["converged"] and run["stress"].diagnostics["converged"]
    assert run["stress"].diagnostics["viscous_form"] != run["sol"].diagnostics["viscous_form"]
    C.check_tolerances(case, run["quantities"])
    r = run["recorded"]
    assert r["near_outlet"] > 1e-3, r
    assert r["below_L_minus_2H"] * 100.0 <= r["below_L_minus_H"], r


# ------------------------------------------------------------------------------------------------- copies on curved and axisymmetric meshes
def _fluids_or_skip():
    try:
        return C.fluids()
    except ImportError:
        pytest.skip("this build has no fluids module")


@pytest.mark.parametrize("doc, domain", [("annulus", None), ("pipe_axi", "pipe")])
def test_fluids_mesh_copy_curved_and_axisymmetric(mesher, doc, domain):
    """The copy on a curved mesh (the annulus at geometry order 2) and on the axisymmetric pipe (with its axis): the
    instance's arrays in order, and the P2 nodes of the curved edges are the exporter's (no straightening)."""
    f = _fluids_or_skip()
    m = axmp.mesh(S.doc(doc))
    im = m.instance(domain)
    fm = im.fluids_mesh()
    seg, ba = fm.boundary()
    assert np.array_equal(fm.vertices(), im.vertices()) and np.array_equal(fm.elements(), im.elements())
    assert np.array_equal(seg, im.boundary()) and np.array_equal(ba, im.boundary_attributes())
    assert fm.has_nodes is im.has_nodes is True
    p2, E = fm.p2_nodes(), im.elements()
    node = {}
    for e in range(len(E)):
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            node[(min(E[e, i], E[e, j]), max(E[e, i], E[e, j]))] = p2[e, 3 + k]
    ev, exy = im.chem_mesh(2).edge_nodes()
    assert (len(ev) > 0) == (doc == "annulus")
    scale = float(np.abs(im.vertices()).max())
    for (a, b), x in zip(ev, exy):
        assert np.abs(node[(min(a, b), max(a, b))] - x).max() <= 1e-15 * scale
    if doc == "annulus":
        sol = f.solve_heat(fm, 1.0, bcs=[dict(attributes=im.attributes["inner"], kind="dirichlet", value=1.0),
                                         dict(attributes=im.attributes["outer"], kind="dirichlet", value=0.0)])
        assert C.max_face_point_offset(sol, im, "inner") <= 1e-15 * scale
