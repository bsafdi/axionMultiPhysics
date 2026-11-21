"""Independent references for the tests of the 3D partition (``test_mesher_partition_3d.py``, ``test_solids_3d.py``,
``test_export_gdml_3d.py``, ``test_export_gdml_3d_transport.py``). Not a test file: pytest collects nothing here.

- two 3D electromagnetic (EM) designs: a window-frame iron dipole (an iron frame, two cylindrical poles that cut
  notches into it, two ring coils around the poles, two scintillator planes up- and downstream) and a design with
  every surface kind the solid grammar makes (a closed hollow box with a ball in its air pocket, a torus, a cone, a
  block crossed by a rod), with their compositions, the densities the export must use, the exact surfaces their
  faces lie on and the closed forms of their region volumes;
- the 3D mesher input of a design, built as the 3D engine builds it (``Model3D``: the same keys and values);
- a reader of binary MSH 2.2 files (the 3D mesher's format), the volumes of P1 and P2 tetrahedra (a collapsed Gauss
  rule, exact for the cubic Jacobian of a P2 tetrahedron), the boundary faces of a set of tetrahedra and the bound of a
  P2 boundary's deviation from the exact surfaces;
- the checks of a triangulated closed surface (directed edges, signed volume, points inside it by ray casting) and of the
  distance between triangles and the exact surfaces;
- the thin-layer and touching cases of the conversion: a thin spherical shell, a rod in a tube with a thin air gap, a
  thin open cylindrical shell, two boxes that share one edge, and a domain without regions.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORDS3 = ROOT / "tests" / "data" / "mesher_partition" / "reference_records_3d.json"

EXPORT3_KEYS = {"ok", "export_regions", "geometry", "gmsh_version", "domain_radius", "domain_center", "diagonal_m",
                "tolerance_m", "chord_tol_m", "regions", "entities", "volumes", "triangulation", "seconds"}
FACE_KEYS = {"tag", "dim", "side", "curves", "axis", "type", "lo", "hi", "point", "normal", "area_m2"}
SHELL_ATTR = 898          # the 3D engine's attribute of the infinity shell (model3d.cpp)
AIR_ATTR, OUTER_ATTR = 1, 901
TEST_CHORD = 1e-3         # the tests' chord tolerance (m): coarse, so that the triangulations stay small


# ============================================================================================ designs
def _box(lo, hi) -> dict:
    return {"box": {"min": list(lo), "max": list(hi)}}


def dipole_design() -> dict:
    """A window-frame dipole: an iron frame 0.6 x 0.5 x 0.4 m with a window 0.36 x 0.26 m along z, two iron poles
    (cylinders of radius 0.08 along y, from the gap of 0.1 m into the frame's plates, so that they cut a notch of depth
    0.02 into each), two copper ring coils around the poles (r 0.10 to 0.14, 0.06 m high) and two scintillator planes
    of 0.2 x 0.2 x 0.01 m at z = -0.45 and +0.44. Domain radius 1 m (infinity shell to 1.25 m)."""
    return {"schema": "axionmultiphysics.design/1", "name": "window_frame_dipole", "symmetry": "3d",
            "regions": [
                {"name": "yoke", "material": "iron",
                 "solid": {"difference": [_box((-0.30, -0.25, -0.20), (0.30, 0.25, 0.20)),
                                          _box((-0.18, -0.13, -0.25), (0.18, 0.13, 0.25))]}},
                {"name": "pole_top", "material": "iron",
                 "solid": {"cylinder": {"base": [0.0, 0.05, 0.0], "axis": [0.0, 0.10, 0.0], "radius": 0.08}}},
                {"name": "pole_bottom", "material": "iron",
                 "solid": {"cylinder": {"base": [0.0, -0.15, 0.0], "axis": [0.0, 0.10, 0.0], "radius": 0.08}}},
                {"name": "coil_top", "material": "winding",
                 "solid": {"revolve": {"profile": {"rect": {"r": [0.10, 0.14], "z": [0.06, 0.12], "fillet": 0.0}},
                                       "angles": [0.0, 360.0], "center": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0]}}},
                {"name": "coil_bottom", "material": "winding",
                 "solid": {"revolve": {"profile": {"rect": {"r": [0.10, 0.14], "z": [-0.12, -0.06], "fillet": 0.0}},
                                       "angles": [0.0, 360.0], "center": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0]}}},
                {"name": "tracker_up", "material": "scint", "solid": _box((-0.1, -0.1, -0.45), (0.1, 0.1, -0.44))},
                {"name": "tracker_down", "material": "scint", "solid": _box((-0.1, -0.1, 0.44), (0.1, 0.1, 0.45))}],
            "materials": {"scint": {"kind": "vacuum", "density": 1032.0}},
            "windings": [{"name": "top", "region": "coil_top", "type": "azimuthal", "ampere_turns": 4.0e4,
                          "axis": {"direction": [0.0, 1.0, 0.0]}},
                         {"name": "bottom", "region": "coil_bottom", "type": "azimuthal", "ampere_turns": 4.0e4,
                          "axis": {"direction": [0.0, 1.0, 0.0]}}],
            "mesh": {"order": 2, "size": 0.08, "surface_size": 0.04, "domain_radius": 1.0}}


DIPOLE_COMPOSITION = {"iron": {"nist": "G4_Fe"}, "winding": {"nist": "G4_Cu", "density": 8960.0},
                      "scint": {"elements": {"C": 9, "H": 10}, "by": "atoms"},
                      "air": {"nist": "G4_AIR", "density": 1.205}}
DIPOLE_DENSITY = {"iron": 7874.0, "winding": 8960.0, "scint": 1032.0, "air": 1.205}


def shapes_design() -> dict:
    """Every surface kind of the solid grammar: a closed hollow iron box (a second air volume inside, the pocket) with a
    lead ball in the pocket; a copper torus; a ptfe cone; an iron block crossed along x by a copper rod (a later
    region: it cuts a tunnel through the block and is split by the block's faces into three volumes, two faces of
    which lie between volumes of the rod itself). Domain radius 0.9 m."""
    return {"schema": "axionmultiphysics.design/1", "name": "every_surface_kind", "symmetry": "3d",
            "regions": [
                {"name": "can", "material": "iron",
                 "solid": {"difference": [_box((0.10, -0.15, -0.15), (0.40, 0.15, 0.15)),
                                          _box((0.13, -0.12, -0.12), (0.37, 0.12, 0.12))]}},
                {"name": "ball", "material": "lead", "solid": {"sphere": {"center": [0.25, 0.0, 0.0], "radius": 0.06}}},
                {"name": "torus", "material": "copper",
                 "solid": {"torus": {"center": [-0.25, 0.0, 0.10], "axis": [0.0, 0.0, 1.0], "major": 0.08,
                                     "minor": 0.025}}},
                {"name": "cone", "material": "ptfe",
                 "solid": {"cone": {"base": [-0.25, 0.0, -0.25], "axis": [0.0, 0.0, 0.15], "r1": 0.07, "r2": 0.03}}},
                {"name": "block", "material": "iron", "solid": _box((-0.10, -0.35, -0.10), (0.10, -0.20, 0.10))},
                {"name": "rod", "material": "copper",
                 "solid": {"cylinder": {"base": [-0.20, -0.275, 0.0], "axis": [0.40, 0.0, 0.0], "radius": 0.03}}}],
            "materials": {"lead": {"kind": "vacuum", "density": 11350.0}, "ptfe": {"kind": "vacuum", "density": 2200.0}},
            "mesh": {"order": 2, "size": 0.08, "surface_size": 0.03, "domain_radius": 0.9}}


SHAPES_COMPOSITION = {"iron": {"nist": "G4_Fe"}, "copper": {"nist": "G4_Cu"}, "lead": {"nist": "G4_Pb"},
                      "ptfe": {"elements": {"C": 2, "F": 4}, "by": "atoms"},
                      "air": {"nist": "G4_Galactic", "density": 1e-22}}
SHAPES_DENSITY = {"iron": 7874.0, "copper": 8960.0, "lead": 11350.0, "ptfe": 2200.0, "air": 1e-22}

DESIGNS3 = {"dipole": (dipole_design, DIPOLE_COMPOSITION, DIPOLE_DENSITY),
            "shapes": (shapes_design, SHAPES_COMPOSITION, SHAPES_DENSITY)}


# ---------------------------------------------------------------------------------------- thin layers and touching
def _doc(name: str, regions: list, domain_radius: float, materials: dict | None = None) -> dict:
    return {"schema": "axionmultiphysics.design/1", "name": name, "symmetry": "3d", "regions": regions,
            "materials": materials or {},
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02, "domain_radius": domain_radius}}


def thin_sphere_design(t: float, domain_radius: float = 1.0) -> dict:
    """An iron spherical shell r = 0.1 to 0.1 + t m around an air cavity (a daughter of the shell)."""
    return _doc("thin_sphere", [{"name": "shell", "material": "iron", "solid": {"difference": [
        {"sphere": {"center": [0.0, 0.0, 0.0], "radius": 0.1 + t}},
        {"sphere": {"center": [0.0, 0.0, 0.0], "radius": 0.1}}]}}], domain_radius)


def rod_in_tube_design(gap: float, domain_radius: float = 0.5) -> dict:
    """A copper rod (r 0.05, 0.2 m long) in an iron tube (bore 0.05 + gap, r 0.06, open at both ends): siblings in the
    air, separated by an air gap of ``gap``."""
    return _doc("rod_in_tube", [
        {"name": "rod", "material": "copper",
         "solid": {"cylinder": {"base": [0.0, 0.0, -0.1], "axis": [0.0, 0.0, 0.2], "radius": 0.05}}},
        {"name": "tube", "material": "iron", "solid": {"difference": [
            {"cylinder": {"base": [0.0, 0.0, -0.1], "axis": [0.0, 0.0, 0.2], "radius": 0.06}},
            {"cylinder": {"base": [0.0, 0.0, -0.2], "axis": [0.0, 0.0, 0.4], "radius": 0.05 + gap}}]}}],
        domain_radius)


def thin_cylinder_design(t: float, domain_radius: float = 1.0) -> dict:
    """An iron cylindrical shell (bore r 0.05, wall t, 0.2 m long, open: planar annular ends) around a copper rod
    (r 0.02): the shell's own outer and bore surfaces are t apart."""
    return _doc("thin_cylinder", [
        {"name": "shield", "material": "iron", "solid": {"difference": [
            {"cylinder": {"base": [0.0, 0.0, -0.1], "axis": [0.0, 0.0, 0.2], "radius": 0.05 + t}},
            {"cylinder": {"base": [0.0, 0.0, -0.2], "axis": [0.0, 0.0, 0.4], "radius": 0.05}}]}},
        {"name": "rod", "material": "copper",
         "solid": {"cylinder": {"base": [0.0, 0.0, -0.1], "axis": [0.0, 0.0, 0.2], "radius": 0.02}}}],
        domain_radius)


def edge_touch_design() -> dict:
    """Two iron boxes that share one edge (the z axis), and nothing else: the air around them has two cavities that
    meet along that edge."""
    return _doc("edge_touch", [
        {"name": "a", "material": "iron", "solid": _box((-0.1, -0.1, -0.1), (0.0, 0.0, 0.1))},
        {"name": "b", "material": "iron", "solid": _box((0.0, 0.0, -0.1), (0.1, 0.1, 0.1))}], 0.5)


def air_only_input(domain_radius: float = 0.5) -> dict:
    """The 3D mesher input of a domain without regions (only the air)."""
    return mesher_input_3d(_doc("air_only", [], domain_radius))


THIN_COMPOSITION = {"iron": {"nist": "G4_Fe"}, "copper": {"nist": "G4_Cu"}, "air": {"nist": "G4_AIR", "density": 1.205}}
THIN_MATERIALS = {"iron": {"density": 7874.0, "particle": {"nist": "G4_Fe"}},
                  "copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}},
                  "air": {"density": 1.205, "particle": {"nist": "G4_AIR"}}}


def design3(name: str) -> dict:
    return DESIGNS3[name][0]()


def region_materials3(doc: dict) -> dict:
    out = {r["name"]: r.get("material", "vacuum") for r in doc["regions"]}
    out["air"] = "air"
    return out


def solids_materials3(name: str) -> dict:
    _, comp, dens = DESIGNS3[name]
    return {m: {"density": dens[m], "particle": {k: v for k, v in block.items() if k != "density"}}
            for m, block in comp.items()}


def mesher_input_3d(doc: dict) -> dict:
    """The 3D mesher input of an EM design as the 3D engine writes it (``Model3D::run_once``: ``mesh3d_in.json``), for
    the tests' designs (literal numbers, every key the engine's resolved solids carry written out, such as a rectangle's
    ``fillet``, no sheets, no full wave, every key of ``mesh`` given): the domain radius, the
    infinity shell of 1.25 domain radii, the attributes, the sizes, the geometry order, the regions in document order
    with attributes 2, 3, ..., the regions refined at their boundary (every region whose material is not vacuum, or
    that carries a winding), one thread. ``output`` and ``info`` are left to the caller."""
    m = doc.get("mesh", {})
    vacuum = {"vacuum"} | {k for k, v in doc.get("materials", {}).items() if v.get("kind") == "vacuum"}
    wound = {w["region"] for w in doc.get("windings", [])}
    regs, refine = [], []
    for k, r in enumerate(doc["regions"]):
        regs.append({"name": r["name"], "attr": 2 + k, "solid": copy.deepcopy(r["solid"])})
        if r.get("material", "vacuum") not in vacuum or r["name"] in wound:
            refine.append(r["name"])
    out = {"dim": 3, "domain_radius": m.get("domain_radius", 0.0), "infinity_shell": 1.25, "shell_attr": SHELL_ATTR,
           "size": m.get("size", 0.0), "symmetry": [], "surface_size": m.get("surface_size", 0.0),
           "order": m.get("geometry_order", min(m.get("order", 2), 2)), "outer_attr": OUTER_ATTR, "air_attr": AIR_ATTR,
           "regions": regs, "refine_regions": refine, "threads": 1}
    for key in ("far_size", "surface_layer", "threads", "algorithm3d", "curvature_points", "extend_from_boundary",
                "optimize_netgen"):
        if key in m:
            out[key] = m[key]
    return out


# ============================================================================================ exact surfaces
class Plane:
    kind = "plane"

    def __init__(self, point, normal):
        self.p = np.asarray(point, float)
        n = np.asarray(normal, float)
        self.n = n / np.linalg.norm(n)

    def distance(self, x):
        return np.abs((np.asarray(x, float) - self.p) @ self.n)


class Cylinder:
    kind = "cylinder"

    def __init__(self, base, axis, radius):
        self.b = np.asarray(base, float)
        a = np.asarray(axis, float)
        self.a = a / np.linalg.norm(a)
        self.r = float(radius)
        self.rho = self.r                       # smallest curvature radius

    def distance(self, x):
        d = np.asarray(x, float) - self.b
        perp = d - np.outer(d @ self.a, self.a)
        return np.abs(np.linalg.norm(perp, axis=1) - self.r)


class Sphere:
    kind = "sphere"

    def __init__(self, center, radius):
        self.c = np.asarray(center, float)
        self.r = float(radius)
        self.rho = self.r

    def distance(self, x):
        return np.abs(np.linalg.norm(np.asarray(x, float) - self.c, axis=1) - self.r)


class Torus:
    kind = "torus"

    def __init__(self, center, axis, major, minor):
        self.c = np.asarray(center, float)
        a = np.asarray(axis, float)
        self.a = a / np.linalg.norm(a)
        self.R, self.r = float(major), float(minor)
        self.rho = self.r

    def distance(self, x):
        d = np.asarray(x, float) - self.c
        h = d @ self.a
        q = np.linalg.norm(d - np.outer(h, self.a), axis=1)
        return np.abs(np.hypot(q - self.R, h) - self.r)


class Cone:
    """The lateral surface of a cone (base b, axis vector a, radii r1 at the base and r2 at b + a), extended."""
    kind = "cone"

    def __init__(self, base, axis, r1, r2):
        self.b = np.asarray(base, float)
        a = np.asarray(axis, float)
        self.h = float(np.linalg.norm(a))
        self.a = a / self.h
        self.r1, self.r2 = float(r1), float(r2)
        self.rho = min(r1, r2)

    def distance(self, x):
        d = np.asarray(x, float) - self.b
        s = d @ self.a
        q = np.linalg.norm(d - np.outer(s, self.a), axis=1)
        # the generator line through (q, s) = (r1, 0) and (r2, h); distance of (q, s) from it
        t = np.array([self.r2 - self.r1, self.h])
        t /= np.linalg.norm(t)
        nrm = np.array([t[1], -t[0]])
        return np.abs((q - self.r1) * nrm[0] + s * nrm[1])


def curved_surfaces(name: str) -> list:
    """The curved surfaces the faces of a design lie on (the domain sphere included, the shell's outer sphere not)."""
    if name == "dipole":
        y = (0.0, 1.0, 0.0)
        return [Cylinder((0, 0, 0), y, 0.08), Cylinder((0, 0, 0), y, 0.10), Cylinder((0, 0, 0), y, 0.14),
                Sphere((0, 0, 0), 1.0)]
    if name == "shapes":
        return [Sphere((0.25, 0, 0), 0.06), Torus((-0.25, 0, 0.10), (0, 0, 1), 0.08, 0.025),
                Cone((-0.25, 0, -0.25), (0, 0, 0.15), 0.07, 0.03), Cylinder((-0.2, -0.275, 0), (1, 0, 0), 0.03),
                Sphere((0, 0, 0), 0.9)]
    raise KeyError(name)


def domain(name: str) -> tuple[np.ndarray, float]:
    return np.zeros(3), {"dipole": 1.0, "shapes": 0.9}[name]


def exact_volumes(name: str) -> dict:
    """The closed forms of the region volumes (m3), air included."""
    pi = math.pi
    if name == "dipole":
        notch = pi * 0.08 ** 2 * 0.02
        v = {"yoke": 0.6 * 0.5 * 0.4 - 0.36 * 0.26 * 0.4 - 2 * notch,
             "pole_top": pi * 0.08 ** 2 * 0.10, "pole_bottom": pi * 0.08 ** 2 * 0.10,
             "coil_top": pi * (0.14 ** 2 - 0.10 ** 2) * 0.06, "coil_bottom": pi * (0.14 ** 2 - 0.10 ** 2) * 0.06,
             "tracker_up": 0.2 * 0.2 * 0.01, "tracker_down": 0.2 * 0.2 * 0.01}
        v["air"] = 4.0 / 3.0 * pi * 1.0 ** 3 - sum(v.values())
        return v
    if name == "shapes":
        v = {"can": 0.3 ** 3 - 0.24 ** 3, "ball": 4.0 / 3.0 * pi * 0.06 ** 3,
             "torus": 2 * pi ** 2 * 0.08 * 0.025 ** 2,
             "cone": pi * 0.15 / 3.0 * (0.07 ** 2 + 0.07 * 0.03 + 0.03 ** 2),
             "block": 0.2 * 0.15 * 0.2 - pi * 0.03 ** 2 * 0.2, "rod": pi * 0.03 ** 2 * 0.4}
        v["air"] = 4.0 / 3.0 * pi * 0.9 ** 3 - sum(v.values())
        return v
    raise KeyError(name)


def surface_of(points: np.ndarray, surfaces: list, tol: float):
    """The exact surface that every point lies on (within ``tol``): a curved surface of the list, else a plane fitted
    to the points when they are coplanar within ``tol``; None when neither."""
    for s in surfaces:
        if float(np.max(s.distance(points))) <= tol:
            return s
    c = points.mean(axis=0)
    _, sv, vt = np.linalg.svd(points - c, full_matrices=False)
    if len(points) >= 3 and sv[-1] / math.sqrt(len(points)) <= tol:
        pl = Plane(c, vt[-1])
        if float(np.max(pl.distance(points))) <= tol:
            return pl
    return None


# ============================================================================================ binary MSH 2.2
_NODES_OF = {1: 2, 2: 3, 4: 4, 8: 3, 9: 6, 11: 10, 15: 1}


class MshBin:
    """A binary (or ASCII) MSH 2.2 file: ``nodes`` {id: row}, ``xyz`` (n, 3), ``names`` {(dim, physical): name}, and
    per element type ``elements[type]`` = (physical tags, elementary tags, node rows (k, n))."""

    def __init__(self, path: Path):
        data = Path(path).read_bytes()
        pos = 0

        def line():
            nonlocal pos
            j = data.index(b"\n", pos)
            s = data[pos:j].decode()
            pos = j + 1
            return s

        assert line() == "$MeshFormat"
        ver, ftype, dsize = line().split()
        assert ver == "2.2" and dsize == "8", (ver, dsize)
        binary = ftype == "1"
        if binary:
            assert np.frombuffer(data[pos:pos + 4], "<i4")[0] == 1
            pos += 4
            line()
        assert line() == "$EndMeshFormat"
        self.names = {}
        ids = None
        self.elements = {}
        while pos < len(data):
            sec = line()
            if sec == "$PhysicalNames":
                for _ in range(int(line())):
                    d, t, nm = line().split(maxsplit=2)
                    self.names[(int(d), int(t))] = nm.strip('"')
                assert line() == "$EndPhysicalNames"
            elif sec == "$Nodes":
                n = int(line())
                if binary:
                    rec = np.frombuffer(data[pos:pos + n * 28], np.dtype([("id", "<i4"), ("x", "<f8", 3)]))
                    pos += n * 28
                    ids, self.xyz = rec["id"].astype(np.int64), rec["x"].copy()
                    line()
                else:
                    rows = [line().split() for _ in range(n)]
                    ids = np.array([int(r[0]) for r in rows])
                    self.xyz = np.array([[float(v) for v in r[1:4]] for r in rows])
                assert line() == "$EndNodes"
            elif sec == "$Elements":
                n = int(line())
                acc = {}
                if binary:
                    done = 0
                    while done < n:
                        et, cnt, ntag = np.frombuffer(data[pos:pos + 12], "<i4")
                        pos += 12
                        w = 1 + ntag + _NODES_OF[int(et)]
                        blk = np.frombuffer(data[pos:pos + 4 * w * cnt], "<i4").reshape(cnt, w)
                        pos += 4 * w * cnt
                        acc.setdefault(int(et), []).append(blk[:, 1:])
                        done += int(cnt)
                    line()
                else:
                    for _ in range(n):
                        w = [int(v) for v in line().split()]
                        acc.setdefault(w[1], []).append(np.array([w[3:]]))
                assert line() == "$EndElements"
                for et, blks in acc.items():
                    b = np.concatenate(blks)
                    self.elements[et] = b
            else:
                end = "$End" + sec[1:]
                while line() != end:
                    pass
        row = np.full(int(ids.max()) + 1, -1, np.int64)
        row[ids] = np.arange(len(ids))
        self.row = row
        for et, b in list(self.elements.items()):
            # tags: physical, elementary (MSH 2.2 writes at least these two); then the node ids -> rows
            phys, elem = b[:, 0].astype(np.int64), b[:, 1].astype(np.int64)
            nn = _NODES_OF[et]
            nodes = row[b[:, -nn:].astype(np.int64)]
            self.elements[et] = (phys, elem, nodes)

    def region_of(self, phys: np.ndarray, dim: int = 3) -> np.ndarray:
        return np.array([self.names.get((dim, int(p)), f"attribute {p}") for p in phys])


def _gauss01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _tet_rule(n=3):
    """Collapsed (Duffy) Gauss rule on the reference tetrahedron, exact for polynomials of degree <= 2n - 3."""
    u, wu = _gauss01(n)
    pts, wts = [], []
    for a, wa in zip(u, wu):
        for b, wb in zip(u, wu):
            for c, wc in zip(u, wu):
                pts.append((a, (1 - a) * b, (1 - a) * (1 - b) * c))
                wts.append(wa * wb * wc * (1 - a) ** 2 * (1 - b))
    return np.array(pts), np.array(wts)


_TQ, _TW = _tet_rule(3)
_EDGES10 = ((0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3))       # gmsh's mid-edge nodes 4..9


def _p2_tet_grads(q):
    """d N_k / d(xi, eta, zeta) of the 10-node tetrahedron at the points q (m, 3): (m, 10, 3)."""
    xi, eta, ze = q[:, 0], q[:, 1], q[:, 2]
    L = np.stack([1 - xi - eta - ze, xi, eta, ze], 1)
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    g = np.zeros((len(q), 10, 3))
    for i in range(4):
        g[:, i, :] = (4 * L[:, i] - 1)[:, None] * dL[i]
    for k, (i, j) in enumerate(_EDGES10):
        g[:, 4 + k, :] = 4 * (L[:, i][:, None] * dL[j] + L[:, j][:, None] * dL[i])
    return g


_G10 = _p2_tet_grads(_TQ)


def tet_volumes(msh: MshBin) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(physical tag, elementary tag, volume) of every tetrahedron (P1: |det|/6 signed; P2: the integral of det J)."""
    out_p, out_e, out_v = [], [], []
    if 4 in msh.elements:
        p, e, nd = msh.elements[4]
        X = msh.xyz[nd]
        v = np.einsum("ij,ij->i", X[:, 1] - X[:, 0], np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0])) / 6.0
        out_p.append(p), out_e.append(e), out_v.append(v)
    if 11 in msh.elements:
        p, e, nd = msh.elements[11]
        X = msh.xyz[nd]                                            # (t, 10, 3)
        J = np.einsum("qkc,tkd->tqdc", _G10, X)                    # (t, q, 3 coords, 3 ref)
        det = np.linalg.det(J)
        out_p.append(p), out_e.append(e), out_v.append(det @ _TW)
    return np.concatenate(out_p), np.concatenate(out_e), np.concatenate(out_v)


_FACES = (((0, 1, 2), (4, 5, 6)), ((0, 1, 3), (4, 9, 7)), ((0, 2, 3), (6, 8, 7)), ((1, 2, 3), (5, 8, 9)))


def boundary_faces(nd: np.ndarray) -> np.ndarray:
    """The faces (corner rows and, for P2, mid rows: (k, 3) or (k, 6)) used by exactly one tetrahedron of ``nd``."""
    p2 = nd.shape[1] == 10
    faces = []
    for corners, mids in _FACES:
        f = nd[:, list(corners) + (list(mids) if p2 else [])]
        faces.append(f)
    F = np.concatenate(faces)
    key = np.sort(F[:, :3], axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return F[cnt[inv.ravel()] == 1]


_TRI_SAMPLES = np.array([[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2], [0.5, 0.25], [0.25, 0.5], [0.25, 0.25]])


def _p2_tri_points(X6: np.ndarray) -> np.ndarray:
    """The P2 triangle maps (nodes: corners 0, 1, 2; mids 01, 12, 20) at _TRI_SAMPLES: (k, s, 3)."""
    a, b = _TRI_SAMPLES[:, 0], _TRI_SAMPLES[:, 1]
    L = np.stack([1 - a - b, a, b], 1)
    N = np.stack([L[:, 0] * (2 * L[:, 0] - 1), L[:, 1] * (2 * L[:, 1] - 1), L[:, 2] * (2 * L[:, 2] - 1),
                  4 * L[:, 0] * L[:, 1], 4 * L[:, 1] * L[:, 2], 4 * L[:, 2] * L[:, 0]], 1)
    return np.einsum("sk,tkc->tsc", N, X6)


def p2_boundary_bound(msh: MshBin, nd: np.ndarray, surfaces: list, tol: float) -> float:
    """The bound of the volume error of the region of the tetrahedra ``nd`` (P2): the sum over its boundary faces that
    lie on a curved surface of the list (every one of their six nodes within ``tol`` of it) of the face's area times
    the largest distance from that surface of seven points of the curved face. Faces on planes contribute nothing (a
    P2 face whose six nodes lie on a plane is that plane)."""
    F = boundary_faces(nd)
    if F.shape[1] != 6:
        raise ValueError("p2_boundary_bound needs P2 tetrahedra")
    X6 = msh.xyz[F]                                                # (k, 6, 3)
    area = 0.5 * np.linalg.norm(np.cross(X6[:, 1] - X6[:, 0], X6[:, 2] - X6[:, 0]), axis=1)
    pts = _p2_tri_points(X6)
    total = 0.0
    on_some = np.zeros(len(F), bool)
    for s in surfaces:
        on = s.distance(X6.reshape(-1, 3)).reshape(-1, 6).max(axis=1) <= tol
        on &= ~on_some
        if on.any():
            d = s.distance(pts[on].reshape(-1, 3)).reshape(-1, len(_TRI_SAMPLES)).max(axis=1)
            total += float(area[on] @ d)
        on_some |= on
    return total


# ============================================================================================ triangulations
def unbalanced_edges(tris: np.ndarray) -> int:
    """The number of undirected edges of the oriented triangles ``tris`` (k, 3) (node indices) that are not used as often
    in one direction as in the other: zero for a closed, consistently oriented surface (a 2-cycle)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    sign = np.where(e[:, 0] < e[:, 1], 1, -1)
    key = np.sort(e, axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    bal = np.zeros(len(uniq), np.int64)
    np.add.at(bal, inv.ravel(), sign)
    return int(np.count_nonzero(bal))


def signed_volume(X: np.ndarray, tris: np.ndarray) -> float:
    """The volume enclosed by oriented triangles (positive for outward normals): sum of x0 . (x1 x x2) / 6."""
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


def triangle_areas(X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def chord_points(X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per triangle its centroid and edge midpoints: (k, 4, 3)."""
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    return np.stack([(a + b + c) / 3.0, 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)], 1)


def interior_points(X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per triangle seven points inside it (the samples of the chord tolerance): (k, 7, 3)."""
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    s, t = _TRI_SAMPLES[:, 0], _TRI_SAMPLES[:, 1]
    return (a[:, None] * (1 - s - t)[None, :, None] + b[:, None] * s[None, :, None] + c[:, None] * t[None, :, None])


def shells_of(part: dict) -> tuple[np.ndarray, dict, dict]:
    """(nodes (n, 3), {face tag: triangles (k, 3)}, {volume tag: [oriented triangle arrays per shell]})."""
    X = np.array(part["triangulation"]["nodes"], float)
    tri = {f["tag"]: np.array(f["triangles"], np.int64).reshape(-1, 3) for f in part["triangulation"]["faces"]}
    vols = {}
    for v in part["volumes"]:
        shells = []
        for sh in v["shells"]:
            parts = [tri[s] if s > 0 else tri[-s][:, ::-1] for s in sh]
            shells.append(np.concatenate(parts) if parts else np.zeros((0, 3), np.int64))
        vols[v["tag"]] = shells
    return X, tri, vols


_RAYS = np.array([[0.5773502691896258, 0.5773502691896257, 0.5773502691896259],
                  [-0.2672612419124244, 0.5345224838248488, 0.8017837257372732],
                  [0.8017837257372732, -0.2672612419124244, 0.5345224838248488]]) + \
    np.array([[1e-3, -2e-3, 3e-4], [7e-4, 1e-4, -5e-4], [-3e-4, 9e-4, 2e-4]])
_RAYS = _RAYS / np.linalg.norm(_RAYS, axis=1)[:, None]


def _ray_parity(X: np.ndarray, T: np.ndarray, points: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The parity of the number of triangles that the ray p + t d (t > 0) crosses, for each point. The triangles and
    the points are projected on the plane normal to d and binned on a uniform grid there, so that each point is tested
    (Moeller-Trumbore) against the triangles of its own cell only."""
    u = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    B = np.stack([u, w], 1)
    tri2 = (X @ B)[T]                                              # (t, 3, 2)
    lo, hi = tri2.min(axis=1), tri2.max(axis=1)
    p2 = points @ B
    origin = np.minimum(lo.min(axis=0), p2.min(axis=0))
    cell = max(float(np.median((hi - lo).max(axis=1))), 1e-300) * 2.0
    clo = np.floor((lo - origin) / cell).astype(np.int64)
    chi = np.floor((hi - origin) / cell).astype(np.int64)
    ncol = int(max(chi[:, 1].max(), np.floor((p2[:, 1] - origin[1]) / cell).max())) + 1
    span = (chi - clo + 1)
    reps = span[:, 0] * span[:, 1]
    tri_ids = np.repeat(np.arange(len(T)), reps)
    offs = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    cx = clo[tri_ids, 0] + offs // span[tri_ids, 1]
    cy = clo[tri_ids, 1] + offs % span[tri_ids, 1]
    keys = cx * ncol + cy
    order = np.argsort(keys, kind="stable")
    keys, tri_ids = keys[order], tri_ids[order]
    pc = np.floor((p2 - origin) / cell).astype(np.int64)
    pkey = pc[:, 0] * ncol + pc[:, 1]
    start = np.searchsorted(keys, pkey, "left")
    stop = np.searchsorted(keys, pkey, "right")
    counts = stop - start
    pid = np.repeat(np.arange(len(points)), counts)
    tid = tri_ids[np.repeat(start, counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))]
    a = X[T[tid, 0]]
    e1, e2 = X[T[tid, 1]] - a, X[T[tid, 2]] - a
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-300
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = points[pid] - a
    uu = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    vv = inv * (q @ d)
    tt = inv * np.einsum("ij,ij->i", q, e2)
    hit = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 0)
    return np.bincount(pid[hit], minlength=len(points)) % 2


def inside_closed_surface(X: np.ndarray, T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Whether each point lies inside the closed triangulated surface (X, T): ray casting along three fixed irrational
    directions, the majority of the three parities. Points on the surface are undecided by construction; the callers
    test points that should be strictly inside."""
    X, T, points = np.asarray(X, float), np.asarray(T, np.int64), np.asarray(points, float)
    votes = sum(_ray_parity(X, T, points, d) for d in _RAYS)
    return votes >= 2


def float_hex(x) -> str:
    return float(x).hex()


def strip_seconds(info: dict) -> dict:
    """An EM info file without what differs from run to run (``seconds``, and the seconds of each high-order pass)."""
    out = copy.deepcopy(info)
    out.pop("seconds", None)
    for p in out.get("high_order_passes", []):
        p.pop("seconds", None)
    return out


def _json_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def records3() -> dict:
    return json.loads(RECORDS3.read_text("utf-8"))
