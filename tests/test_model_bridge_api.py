"""The Python surface of ``InstanceMesh.chem_mesh`` that the engine acceptance tests of the bridge do not reach: the
chemistry mesh owns its arrays (changing what it hands out changes neither it nor the instance mesh), and the field
order is an integer. Every build."""
from __future__ import annotations

import numpy as np
import pytest

import axionmultiphysics as axmp

import model_mesh_support as S
from model_mesh_support import mesher  # noqa: F401 (a fixture)


@pytest.fixture
def chem(mesher):
    try:
        from axionmultiphysics._core import chem as _c  # noqa: F401
    except ImportError:
        pytest.skip("axionmultiphysics._core has no chem submodule (rebuild)")


def test_the_chem_mesh_owns_its_arrays(chem):
    """The edge nodes of the annulus (a curved mesh), the vertices and the instance mesh are unchanged by writing into
    what ``edge_nodes()`` and ``vertices()`` returned."""
    im = axmp.mesh(S.doc("annulus")).instance()
    V0, T0 = im.vertices().copy(), im.elements().copy()
    cm = im.chem_mesh(order=2)
    ev0, exy0 = (np.array(a) for a in cm.edge_nodes())
    assert len(exy0) > 0
    ev, exy = cm.edge_nodes()
    ev[:] = 0
    exy[:] = 0.0
    cm.vertices()[:] = 0.0
    ev1, exy1 = cm.edge_nodes()
    assert np.array_equal(ev1, ev0) and np.array_equal(exy1, exy0)
    assert np.array_equal(cm.vertices(), V0)
    cm.X[:] = 0.0
    assert np.array_equal(im.vertices(), V0) and np.array_equal(im.elements(), T0)
    assert np.array_equal(im.chem_mesh(2).vertices(), V0)


def test_the_order_1_refusal_names_a_remedy_and_chem_mesh_3_no_internal_name(chem):
    im = axmp.mesh(S.doc("annulus")).instance()
    with pytest.raises(ValueError) as e:
        im.chem_mesh(1)
    assert "geometry order 1" in str(e.value) and "order 2" in str(e.value), str(e.value)
    with pytest.raises(ValueError) as e:
        im.chem_mesh(3)
    assert "chem_mesh_input" not in str(e.value) and "3" in str(e.value), str(e.value)


def test_curved_false_does_not_straighten_a_mesh_with_nodes(chem):
    """``curved=False`` on a chemistry mesh that records geometry order 2 is a ValueError (also the straight plate cell,
    whose instance mesh has nodes); on a mesh of geometry order 1, and for any other TriMesh, it still means straight edges."""
    from axionmultiphysics import chem as c
    for doc in ("annulus", "plate_cell"):
        cm = axmp.mesh(S.doc(doc)).chem_mesh(order=2)
        with pytest.raises(ValueError, match="curved=False"):
            c.CurrentDistribution(cm, order=2, curved=False)
    c.CurrentDistribution(axmp.mesh(S.doc("annulus_cell")).chem_mesh(order=2), order=2, curved=False)
    c.CurrentDistribution(c.meshes.rect(2), order=2, curved=False)


def test_red_refine_of_a_bridge_mesh(chem):
    """A curved bridge mesh is refused (the refined mesh would lose its edge nodes); a straight one is refined."""
    from axionmultiphysics.chem import meshes
    with pytest.raises(ValueError, match="red_refine"):
        meshes.red_refine(axmp.mesh(S.doc("annulus")).chem_mesh(order=2))
    cm = axmp.mesh(S.doc("plate_cell")).chem_mesh(order=2)
    r = meshes.red_refine(cm)
    assert len(r.T) == 4 * len(cm.T) and abs(_area(r) - _area(cm)) <= 1e-12 * _area(cm)


def _area(m):
    X, T = m.vertices(), np.asarray(m.T)
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    return float(np.abs(0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])).sum())


@pytest.mark.parametrize("bad", [True, 1.0, "2", None, 0, 3, -1])
def test_the_field_order_is_the_integer_1_or_2(chem, bad):
    """Anything but the integers 1 and 2 (a bool, a float, a string, None, other integers) is a ValueError, also through
    the shorthand; on a mesh without nodes, where order 1 is otherwise valid, so that a bool cannot pass as 1."""
    m = axmp.mesh(S.doc("annulus_cell"))
    assert m.geometry_order == 1
    with pytest.raises(ValueError):
        m.instance().chem_mesh(bad)
    with pytest.raises(ValueError):
        m.chem_mesh(order=bad)


def test_a_numpy_integer_is_a_field_order(chem):
    cm = axmp.mesh(S.doc("plate_cell")).chem_mesh(order=np.int64(2))
    assert cm.geometry_order == 2


def test_the_geometry_order_is_the_meshs_not_the_field_orders(chem):
    """A mesh of geometry order 1 handed to a field of order 2 records geometry order 1 and has no edge node."""
    m = axmp.mesh(S.doc("annulus_cell"))
    for order in (1, 2):
        cm = m.chem_mesh(order=order)
        assert cm.geometry_order == 1
        ev, exy = cm.edge_nodes()
        assert np.shape(ev) == (0, 2) and np.shape(exy) == (0, 2)
