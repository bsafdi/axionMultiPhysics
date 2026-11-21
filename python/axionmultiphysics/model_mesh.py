"""The meshed model of a model document and the instance meshes of its physics (beta; ``axmp.mesh``).

``axmp.mesh(d)`` validates a model document with the mesher's probe, resolves its named boundaries, runs the full
check, meshes the enabled regions into one master mesh and returns a :class:`MeshedModel`. No attribute number of the
master mesh is handed out: a physics receives an :class:`InstanceMesh`, the mesh of the regions it solves, with its own
dense numbering of the boundary attributes and the map that names them.
"""
from __future__ import annotations

import json as _json
import numbers
from pathlib import Path
from typing import Any

from . import _core

__all__ = ["InstanceMesh", "MeshedModel", "mesh"]


class InstanceMesh:
    """The mesh of the regions of one physics instance (a plain mesh that owns its data).

    ``attributes[name]`` lists the boundary attributes of the boundary ``name`` on this mesh (instance numbering,
    1..n; empty when the boundary is not on its boundary), ``remaining_attributes`` those of the edges no boundary
    names and ``axis_attributes`` the axis of an axisymmetric model. ``master_attribute`` maps each instance attribute
    to the master mesh's number, ``region_attribute`` each region to its element attribute.
    """

    def __init__(self, core):
        self._m = core
        maps = _json.loads(core.maps())
        self.regions: list[str] = maps["regions"]
        self.geometry_order: int = maps["geometry_order"]
        self.has_nodes: bool = maps["has_nodes"]
        self.attributes: dict[str, list[int]] = maps["attributes"]
        self.remaining_attributes: list[int] = maps["remaining_attributes"]
        self.axis_attributes: list[int] = maps["axis_attributes"]
        self.master_attribute: dict[int, int] = {int(k): v for k, v in maps["master_attribute"].items()}
        self.region_attribute: dict[str, int] = maps["region_attribute"]

    def vertices(self):
        """The vertices, an ``(nv, 2)`` float64 array."""
        return self._m.vertices()

    def elements(self):
        """The triangles' vertex indices, an ``(ne, 3)`` array."""
        return self._m.elements()

    def element_attributes(self):
        """The element attribute (the region's) of each triangle, an ``(ne,)`` array."""
        return self._m.element_attributes()

    def boundary(self):
        """The boundary segments' vertex indices, an ``(nbe, 2)`` array."""
        return self._m.boundary()

    def boundary_attributes(self):
        """The instance attribute of each boundary segment, an ``(nbe,)`` array."""
        return self._m.boundary_attributes()

    def fluids_mesh(self):
        """The mesh for the fluids engines (``solve_stokes``, ``solve_heat``, ``solve_boussinesq``): an
        ``axionmultiphysics.fluids.Mesh`` that is a copy of this instance mesh, with its vertices, triangles, boundary
        segments and boundary attributes in the same order and with its nodes when this mesh has them. Pass the
        conditions on ``attributes[name]``, ``remaining_attributes`` and ``axis_attributes``, never on master numbers.
        The copy owns its data (it outlives this mesh and the meshed model).

        Raises AxionMultiPhysicsError with ``build.requires_fluids`` in a build without the fluids module (the
        sequential source build and the wheels have none)."""
        if not hasattr(_core, "_fluids"):
            _core._requires_fluids()
        from . import fluids

        return fluids.Mesh.from_instance(self._m)

    def chem_mesh(self, order: int = 2):
        """The mesh for ``chem.CurrentDistribution``: a ``chem.TriMesh`` with this mesh's vertices, triangles, element
        attributes, boundary segments and attributes in the same order, its ``geometry_order``, and its ``edge_nodes()``
        (the P2 edge nodes that are off their chord, none on a straight mesh). ``order`` is the field order of the
        current distribution, 1 or 2. A field of order 1 on a mesh with nodes (geometry order 2) is a ValueError: a curved
        mesh is not straightened silently; mesh the document with geometry order 1 for a P1 field (the validation
        lowers it for such a model). Any other order is a ValueError."""
        from .chem.meshes import InstanceTriMesh

        if isinstance(order, bool) or not isinstance(order, numbers.Integral) or int(order) not in (1, 2):
            raise ValueError(f"chem_mesh: the field order is 1 or 2, not {order!r}")
        return InstanceTriMesh.from_instance(self._m, int(order), self.geometry_order)

    def section(self, name: str) -> dict:
        """The faces of the boundary ``name`` that are interior to this instance: ``{"name", "faces", "sign",
        "length_m"}``. ``sign`` is +1 where the selection's normal is the face's normal. A boundary with no interior
        face is a ValueError."""
        return _json.loads(self._m.section(name))

    def face_points(self, name: str | None = None, *, order: int = 2, attribute: int | None = None) -> dict:
        """Quadrature on the faces of a boundary or a section (``name``) or of one instance attribute (``attribute``,
        the way to the edges no boundary names, and to the axis): exactly one of the two is given.

        Returns NumPy arrays: ``faces`` (m,), ``elements`` (m,), ``ref`` (m, q, 2) the reference coordinates in the
        adjacent element, ``weights`` (m, q) (the face Jacobian included: they sum to the length), ``normals`` (m, q, 2)
        (outward on the instance boundary; the orientation of the selection on a section) and ``points`` (m, q, 2).
        Misuse (an unknown name, an attribute not on this mesh, both or neither argument) is a ValueError.
        """
        if (name is None) == (attribute is None):
            raise ValueError("give exactly one of name and attribute")
        if name is not None and not name:
            raise ValueError("the name of a boundary is not empty")
        if attribute is not None:
            if isinstance(attribute, bool) or not isinstance(attribute, numbers.Integral):
                raise ValueError(f"the attribute must be an integer, not {attribute!r}")
            if not 0 <= attribute < 2**31:
                raise ValueError(f"the attribute {attribute} is not on this instance mesh")
        return self._m.face_points(name or "", -1 if attribute is None else int(attribute), int(order))


class MeshedModel:
    """A meshed model document: ``symmetry``, ``geometry_order``, ``regions`` (name -> ``{"area_m2"}``, with
    ``"volume_m3"`` when axisymmetric), ``boundaries`` (the table of ``Design.boundaries()``), ``mesh_file`` and
    ``info_file`` (``None`` without a workdir) and :meth:`instance`."""

    def __init__(self, core):
        self._m = core
        self.symmetry: str = core.symmetry
        self.geometry_order: int = core.geometry_order
        self.regions: dict[str, dict] = _json.loads(core.regions_json())
        self.boundaries: dict[str, dict] = _json.loads(core.boundaries_json())
        self.mesh_file: str | None = core.mesh_file or None
        self.info_file: str | None = core.info_file or None

    def instance(self, domain: str | list[str] | None = None) -> InstanceMesh:
        """The mesh of a physics instance: ``None`` for every enabled region, a region or domain name, or a list of
        names. An unknown name, a disabled region or an empty list is a ValueError."""
        if domain is None:
            return InstanceMesh(self._m.instance([], True))
        names = [domain] if isinstance(domain, str) else list(domain)
        return InstanceMesh(self._m.instance(names, False))

    def fluids_mesh(self, domain: str | list[str] | None = None):
        """``self.instance(domain).fluids_mesh()``."""
        return self.instance(domain).fluids_mesh()

    def chem_mesh(self, domain: str | list[str] | None = None, order: int = 2):
        """``self.instance(domain).chem_mesh(order)``."""
        return self.instance(domain).chem_mesh(order)


def mesh(design: Any, workdir: str | Path | None = None, mesher: str | Path | None = None) -> MeshedModel:
    """Mesh a model document (a :class:`Design` or a dict; beta) and return its :class:`MeshedModel`.

    The document is validated (with the mesher's probe), its named boundaries are resolved and checked, and only then is
    it meshed. ``workdir`` keeps the mesher's input, mesh, info file and log (``mesh_file`` and ``info_file`` name the
    last two); without it a temporary directory is used and removed. ``mesher`` names the mesher program to try first.
    Raises AxionMultiPhysicsError with the first error of the document, ``build.requires_mesher`` when the mesher does
    not run, ``mesh.generation_failed`` when it fails; ``route.not_implemented`` for a 3D document, a document on an
    imported mesh (``mesh.file``) and a document with no geometry; an electromagnetic design is refused without a code.
    """
    doc = design.doc if hasattr(design, "doc") else design
    return MeshedModel(_core.mesh_model(_json.dumps(doc), "" if workdir is None else str(workdir),
                                        "" if mesher is None else str(mesher)))
