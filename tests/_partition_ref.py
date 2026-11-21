"""Independent references for the tests of the mesher's partition (``test_mesher_partition.py``,
``test_export_gdml.py``, ``test_export_gdml_transport.py``). Not a test file: pytest collects nothing here.

- running the mesher on a bounded fixture or an electromagnetic (EM) input, with and without ``--export-regions``;
- the records of the mesher before the partition keys existed (``tests/data/mesher_partition/reference_records.json``)
  and the keys the partition adds;
- an ASCII MSH 2.2 reader, the measures of P1 and P2 triangles (collapsed Gauss rule, exact for these polynomials) and
  the boundary edges of a set of triangles;
- the curves of the partition's table, evaluated from their parameters alone (no gmsh, no ``solids``): the position of
  a point along a curve, its distance from the curve, the tangent, the exact box, the length and Green's integrals by
  Gauss-Legendre quadrature in the angle;
- the gap between a P2 edge and the circle through its three nodes (the error of the mesh's curved boundary);
- five EM designs (a solenoid with an iron yoke, a design with every curve the shape grammar makes, the CORE
  template with its iron caps, clockwise filleted polygons, and sheets written against gmsh's parametrization, an arc
  among them) with their compositions and the densities the export must use.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "cpp" / "tests" / "data" / "model" / "u2" / "mesher"
RECORDS = ROOT / "tests" / "data" / "mesher_partition" / "reference_records.json"

ENTITY_KEYS_ADDED = ("start", "end", "center", "radii", "major_axis", "ccw")
ARC_KEYS = ("center", "radii", "major_axis", "ccw")
TOP_KEYS_ADDED = ("faces",)
EXPORT_KEYS = {"ok", "export_regions", "geometry", "gmsh_version", "domain_radius", "domain_center_z", "diagonal_m",
               "tolerance_m", "regions", "entities", "faces", "seconds"}
ENTITY_FORM = {"tag", "dim", "side", "curves", "axis", "type", "lo", "hi", "point", "tangent", "length_m", "area_m2"}
TIMEOUT = 600


# ============================================================================================ fixtures and runs
def fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text("utf-8"))


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


def bounded_fixtures() -> list[str]:
    """The accepted bounded fixtures (``refuse_*`` are the refusals)."""
    return [n for n in fixture_names() if fixture(n).get("bounded") and not n.startswith("refuse_")]


def refusal_fixtures() -> list[str]:
    return [n for n in fixture_names() if n.startswith("refuse_")]


def run_mesher(mesher: str, inp: dict, directory: Path, name: str = "m", *, export: bool = False,
               args: list | None = None):
    """Runs the mesher on ``inp`` with ``output`` = DIR/<name>.msh and ``info`` = DIR/<name>_info.json. Returns the
    completed process, the info file's JSON (None when it is missing or not JSON) and the mesh path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inp = copy.deepcopy(inp)
    msh, info_path = directory / f"{name}.msh", directory / f"{name}_info.json"
    inp["output"], inp["info"] = str(msh), str(info_path)
    path = directory / f"{name}_in.json"
    path.write_text(json.dumps(inp, indent=1), "utf-8")
    argv = [mesher] + (args if args is not None else (["--export-regions"] if export else [])) + [str(path)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT)
    info = None
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text("utf-8"))
        except json.JSONDecodeError:
            info = None
    return r, info, msh


def strip_added(info: dict) -> dict:
    """The info file without the partition's added keys and without ``seconds``."""
    out = copy.deepcopy(info)
    out.pop("seconds", None)
    for k in TOP_KEYS_ADDED:
        out.pop(k, None)
    for e in out.get("entities", []):
        for k in ENTITY_KEYS_ADDED:
            e.pop(k, None)
    return out


def without_seconds(info: dict) -> dict:
    out = copy.deepcopy(info)
    out.pop("seconds", None)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records() -> dict:
    return json.loads(RECORDS.read_text("utf-8"))


def same_json(a, b, path: str = "") -> list[str]:
    """The differences of two JSON values: types, key order, and every number bit for bit (``float.hex``)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [f"{path}: keys {list(a)} != {list(b)}"]
        out = []
        for k in a:
            out += same_json(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += same_json(x, y, f"{path}/{i}")
        return out
    if type(a) is not type(b):
        return [f"{path}: {a!r} ({type(a).__name__}) != {b!r} ({type(b).__name__})"]
    if isinstance(a, float):
        return [] if a.hex() == b.hex() else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


# ============================================================================================ MSH 2.2
class Msh:
    """An ASCII MSH 2.2 file: ``xyz`` {node id: (x, y)}, ``elements`` [(type, physical, elementary, [node ids])],
    ``names`` {(dim, physical): name}."""

    def __init__(self, path: Path):
        lines = Path(path).read_text().splitlines()
        i = lines.index("$MeshFormat")
        assert lines[i + 1].split()[:2] == ["2.2", "0"], lines[i + 1]
        self.names = {}
        if "$PhysicalNames" in lines:
            i = lines.index("$PhysicalNames")
            for ln in lines[i + 2:i + 2 + int(lines[i + 1])]:
                d, t, name = ln.split(maxsplit=2)
                self.names[(int(d), int(t))] = name.strip('"')
        i = lines.index("$Nodes")
        n = int(lines[i + 1])
        self.xyz = {}
        for ln in lines[i + 2:i + 2 + n]:
            w = ln.split()
            self.xyz[int(w[0])] = (float(w[1]), float(w[2]))
        i = lines.index("$Elements")
        n = int(lines[i + 1])
        self.elements = []
        for ln in lines[i + 2:i + 2 + n]:
            w = [int(x) for x in ln.split()]
            ntags = w[2]
            tags = w[3:3 + ntags]
            self.elements.append((w[1], tags[0], tags[1], w[3 + ntags:]))

    def triangles(self):
        return [e for e in self.elements if e[0] in (2, 9)]

    def lines(self):
        return [e for e in self.elements if e[0] in (1, 8)]

    def p(self, node: int) -> np.ndarray:
        return np.array(self.xyz[node])


def _gauss01(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _triangle_rule(n: int = 5):
    """Collapsed Gauss rule on the reference triangle (exact for polynomials of degree <= 2n - 2)."""
    u, wu = _gauss01(n)
    xi, eta, w = [], [], []
    for a, wa in zip(u, wu):
        for b, wb in zip(u, wu):
            xi.append(a)
            eta.append((1.0 - a) * b)
            w.append(wa * wb * (1.0 - a))
    return np.array(xi), np.array(eta), np.array(w)


_XI, _ETA, _W = _triangle_rule()


def _shape(kind: int):
    """Shape functions and their derivatives at the rule's points: N (q, k), dN/dxi, dN/deta."""
    l1, l2 = _XI, _ETA
    l0 = 1.0 - l1 - l2
    if kind == 2:
        N = np.stack([l0, l1, l2], 1)
        dxi = np.tile([-1.0, 1.0, 0.0], (len(l0), 1))
        deta = np.tile([-1.0, 0.0, 1.0], (len(l0), 1))
        return N, dxi, deta
    N = np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0], 1)
    dxi = np.stack([-(4 * l0 - 1), 4 * l1 - 1, 0 * l0, 4 * (l0 - l1), 4 * l2, -4 * l2], 1)
    deta = np.stack([-(4 * l0 - 1), 0 * l0, 4 * l2 - 1, -4 * l1, 4 * l1, 4 * (l0 - l2)], 1)
    return N, dxi, deta


def triangle_measures(msh: Msh, tris: list) -> tuple[np.ndarray, np.ndarray]:
    """(area, integral of 2 pi r dA) of each triangle of the isoparametric P1 or P2 map. The integrands are
    polynomials of degree <= 4 (P2: r quadratic, det J quadratic), which the 25-point rule integrates exactly. Raises
    on a triangle whose Jacobian is not positive (the mesher orients every face counter-clockwise)."""
    area = np.zeros(len(tris))
    vol = np.zeros(len(tris))
    for kind in (2, 9):
        idx = [k for k, t in enumerate(tris) if t[0] == kind]
        if not idx:
            continue
        N, dxi, deta = _shape(kind)
        X = np.array([[msh.xyz[n] for n in tris[k][3]] for k in idx])       # (t, k, 2)
        x = np.einsum("qk,tkc->tqc", N, X)
        jx = np.einsum("qk,tkc->tqc", dxi, X)
        je = np.einsum("qk,tkc->tqc", deta, X)
        det = jx[..., 0] * je[..., 1] - jx[..., 1] * je[..., 0]
        assert (det > 0).all(), "a triangle of the mesh is not counter-clockwise"
        area[idx] = det @ _W
        vol[idx] = (2.0 * math.pi * x[..., 0] * det) @ _W
    return area, vol


def boundary_edges(msh: Msh, tris: list) -> list:
    """The edges used by exactly one triangle of ``tris``: (a, b, m, triangle index), a -> b in the triangle's own
    (counter-clockwise) order, so that the triangle lies on the left of a -> b; m is the P2 mid node (None for P1)."""
    count = {}
    for k, t in enumerate(tris):
        v = t[3]
        mids = v[3:6] if t[0] == 9 else [None, None, None]
        for (a, b), m in zip(((v[0], v[1]), (v[1], v[2]), (v[2], v[0])), mids):
            key = (min(a, b), max(a, b))
            count.setdefault(key, []).append((a, b, m, k))
    return [uses[0] for uses in count.values() if len(uses) == 1]


def p2_gap(a: np.ndarray, m: np.ndarray, b: np.ndarray) -> float:
    """The area between a P2 edge (the parabola through a, m, b with m at its parameter 1/2) and the circle through the
    same three points: |R^2 (theta - sin theta) / 2 - (2/3) |ab| d|, d the distance of m from the chord (Archimedes:
    the quadratic's segment is 4/3 of the triangle a, m, b, since its tangent at parameter 1/2 is parallel to b - a).
    Zero for three collinear points."""
    ab = b - a
    c = float(np.hypot(*ab))
    d = abs(float(ab[0] * (m - a)[1] - ab[1] * (m - a)[0])) / c
    if d <= 1e-14 * c:
        return 0.0
    am, mb = float(np.hypot(*(m - a))), float(np.hypot(*(b - m)))
    R = am * mb * c / (2.0 * d * c)                          # circumradius: product of the sides / (4 x area)
    theta = 2.0 * math.asin(min(1.0, c / (2.0 * R)))
    if d > R:                                                # the arc is the major one (never on a fine mesh)
        theta = 2.0 * math.pi - theta
    return abs(R * R * (theta - math.sin(theta)) / 2.0 - 2.0 / 3.0 * c * d)


def region_volumes(msh: Msh) -> dict:
    """{region name: (V, B)}: the region's revolved volume in the mesh (the integral of 2 pi r over its triangles) and
    the bound of that volume's boundary error, the sum over the region's boundary edges of 2 pi r_max times the gap
    between the P2 edge and the circle through its nodes (``p2_gap``; zero on straight edges)."""
    names2 = {t: n for (d, t), n in msh.names.items() if d == 2}
    by_region: dict = {}
    for t in msh.triangles():
        by_region.setdefault(names2[t[1]], []).append(t)
    out = {}
    for r, tris in by_region.items():
        _, vol = triangle_measures(msh, tris)
        gap = 0.0
        for a, b, m, _ in boundary_edges(msh, tris):
            if m is not None:
                g = p2_gap(msh.p(a), msh.p(m), msh.p(b))
                gap += 2.0 * math.pi * max(msh.p(a)[0], msh.p(m)[0], msh.p(b)[0]) * g
        out[r] = (float(vol.sum()), gap)
    return out


# ============================================================================================ curves of the table
TWO_PI = 2.0 * math.pi
_GX, _GW = np.polynomial.legendre.leggauss(24)


class Curve:
    """One entity's curve from its table keys alone: a line from ``start`` to ``end``, or the arc
    c + a cos t u + b sin t v (v = u turned by +90 degrees) from ``start`` to ``end``, t increasing iff ``ccw``."""

    def __init__(self, e: dict):
        self.tag = e["tag"]
        self.type = e["type"]
        self.p = np.array(e["start"][:2], dtype=float)
        self.q = np.array(e["end"][:2], dtype=float)
        self.closed = tuple(e["start"]) == tuple(e["end"])
        self.line = e["type"] == "line"
        if self.line:
            return
        self.c = np.array(e["center"][:2], dtype=float)
        self.a, self.b = float(e["radii"][0]), float(e["radii"][1])
        self.u = np.array(e["major_axis"][:2], dtype=float)
        self.v = np.array([-self.u[1], self.u[0]])
        self.sense = 1.0 if e["ccw"] else -1.0
        self.t0 = self.angle(self.p)
        self.sweep = TWO_PI if self.closed else (self.sense * (self.angle(self.q) - self.t0)) % TWO_PI

    # ---------------------------------------------------------------- parametrization
    def angle(self, x) -> float:
        d = np.asarray(x, dtype=float) - self.c
        return math.atan2(float(d @ self.v) / self.b, float(d @ self.u) / self.a)

    def at_angle(self, t: float) -> np.ndarray:
        return self.c + self.a * math.cos(t) * self.u + self.b * math.sin(t) * self.v

    def s(self, x) -> float:
        """The position of x along the curve, 0 at ``start`` and 1 at ``end`` (a point a little before the start of an
        open arc comes out slightly negative, not near 2 pi / sweep)."""
        x = np.asarray(x, dtype=float)
        if self.line:
            d = self.q - self.p
            return float((x - self.p) @ d / (d @ d))
        delta = (self.sense * (self.angle(x) - self.t0)) % TWO_PI
        if not self.closed and delta > self.sweep + 0.5 * (TWO_PI - self.sweep):
            delta -= TWO_PI
        return delta / self.sweep

    def step(self, x, y) -> float:
        """The signed change of s from x to y, for two points of one short edge (no wrap)."""
        if self.line:
            return self.s(y) - self.s(x)
        dt = (self.angle(y) - self.angle(x) + math.pi) % TWO_PI - math.pi
        return self.sense * dt / self.sweep

    def at(self, s: float) -> np.ndarray:
        if self.line:
            return self.p + s * (self.q - self.p)
        return self.at_angle(self.t0 + self.sense * s * self.sweep)

    def tangent(self, s: float) -> np.ndarray:
        """The unit direction from ``start`` to ``end`` at s."""
        if self.line:
            d = self.q - self.p
        else:
            t = self.t0 + self.sense * s * self.sweep
            d = self.sense * (-self.a * math.sin(t) * self.u + self.b * math.cos(t) * self.v)
        return d / np.hypot(*d)

    def distance(self, x) -> float:
        """The distance of x from the full line or ellipse (for an ellipse the first-order distance |F| / |grad F|,
        F = (p/a)^2 + (q/b)^2 - 1, exact to second order in the distance)."""
        x = np.asarray(x, dtype=float)
        if self.line:
            d = self.q - self.p
            return abs(float(d[0] * (x - self.p)[1] - d[1] * (x - self.p)[0])) / float(np.hypot(*d))
        d = x - self.c
        p, q = float(d @ self.u), float(d @ self.v)
        if self.a == self.b:
            return abs(math.hypot(p, q) - self.a)
        F = (p / self.a) ** 2 + (q / self.b) ** 2 - 1.0
        G = math.hypot(2 * p / self.a ** 2, 2 * q / self.b ** 2)
        return abs(F) / G

    # ---------------------------------------------------------------- exact measures
    def box(self) -> tuple[float, float, float, float]:
        pts = [self.p, self.q]
        if not self.line:
            for coord in (0, 1):
                A, B = self.a * self.u[coord], self.b * self.v[coord]
                t1 = math.atan2(B, A)
                for t in (t1, t1 + math.pi):
                    if self.closed or (self.sense * (t - self.t0)) % TWO_PI <= self.sweep:
                        pts.append(self.at_angle(t))
        P = np.array(pts)
        return float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())

    def _quad(self, f) -> float:
        """The integral over s in [0, 1] of f(t, dt/ds) by 16 panels of 24 Gauss points."""
        total = 0.0
        for k in range(16):
            s = (k + 0.5 * (_GX + 1.0)) / 16.0
            t = self.t0 + self.sense * s * self.sweep
            total += float(np.sum(_GW * f(t, self.sense * self.sweep))) / 32.0
        return total

    def length(self) -> float:
        if self.line:
            return float(np.hypot(*(self.q - self.p)))

        def f(t, dtds):
            dx = -self.a * np.sin(t)[:, None] * self.u + self.b * np.cos(t)[:, None] * self.v
            return np.hypot(dx[:, 0], dx[:, 1]) * abs(dtds)
        return self._quad(f)

    def green(self) -> tuple[float, float]:
        """(integral of x0 dx1, integral of x0^2 dx1) from ``start`` to ``end``: the area and volume / pi terms of
        Green's theorem for the face on the curve's left."""
        if self.line:
            (r0, z0), (r1, z1) = self.p, self.q
            return 0.5 * (r0 + r1) * (z1 - z0), (r0 * r0 + r0 * r1 + r1 * r1) * (z1 - z0) / 3.0

        def g(power):
            def f(t, dtds):
                x = self.c[0] + self.a * np.cos(t) * self.u[0] + self.b * np.sin(t) * self.v[0]
                dz = -self.a * np.sin(t) * self.u[1] + self.b * np.cos(t) * self.v[1]
                return x ** power * dz * dtds
            return f
        return self._quad(g(1)), self._quad(g(2))


def curves(table: dict) -> dict:
    return {e["tag"]: Curve(e) for e in table["entities"]}


def loop_measures(loop: list, cs: dict) -> tuple[float, float]:
    """(area, volume) of one loop by Green's theorem: positive for a counter-clockwise loop."""
    area = vol = 0.0
    for s in loop:
        a, v = cs[abs(s)].green()
        sign = 1.0 if s > 0 else -1.0
        area += sign * a
        vol += sign * math.pi * v
    return area, vol


# ============================================================================================ EM designs
def _rect(r, z, fillet=None) -> dict:
    s = {"r": list(r), "z": list(z)}
    if fillet:
        s["fillet"] = fillet
    return {"rect": s}


def yoke_design() -> dict:
    """A solenoid in an iron pot yoke whose end plates have a bore on the axis, and a lead absorber (a sphere) at its
    centre. Straight edges except the sphere and the domain's outer arc."""
    return {"schema": "axionmultiphysics.design/1", "name": "solenoid_in_iron_yoke",
            "regions": [
                {"name": "winding", "material": "winding", "shape": _rect((0.20, 0.24), (-0.30, 0.30))},
                {"name": "yoke", "material": "iron",
                 "shape": {"difference": [_rect((0.0, 0.40), (-0.45, 0.45)), _rect((0.0, 0.30), (-0.35, 0.35)),
                                          _rect((0.0, 0.05), (-0.5, 0.5))]}},
                {"name": "absorber", "material": "lead", "shape": {"disk": {"center": [0.0, 0.0], "radius": 0.1}}}],
            "materials": {"lead": {"kind": "vacuum", "density": 11350.0}},
            "windings": [{"name": "w", "region": "winding", "type": "azimuthal", "ampere_turns": 1.0e5}],
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02, "domain_radius": 1.2}}


YOKE_COMPOSITION = {"winding": {"nist": "G4_Cu", "density": 8960.0}, "iron": {"nist": "G4_Fe"},
                    "lead": {"nist": "G4_Pb"}, "air": {"nist": "G4_AIR", "density": 1.205}}
YOKE_DENSITY = {"winding": 8960.0, "iron": 7874.0, "lead": 11350.0, "air": 1.205}

NOSE = {"center": [0.25, 0.4], "axes": [0.06, 0.1]}       # the contour's elliptical segment (a B-spline in gmsh)


def curved_design() -> dict:
    """Every curve the 2D shape grammar makes: an off-axis disk (a closed circle: a torus), a filleted block (circular
    arcs), a contour with an elliptical segment (a rational B-spline), a disk on the axis (a sphere), a disk cut by a
    later plate (arcs ending where OpenCASCADE intersects a circle and lines), a block on the axis, an open arc sheet
    that ends in the air (embedded) and a closed rectangular sheet (a second face of the air)."""
    return {"schema": "axionmultiphysics.design/1", "name": "every_curve_kind",
            "regions": [
                {"name": "torus", "material": "copper", "shape": {"disk": {"center": [0.5, 0.2], "radius": 0.05}}},
                {"name": "block", "material": "iron", "shape": _rect((0.3, 0.45), (-0.3, -0.1), 0.03)},
                {"name": "nose", "material": "ptfe", "shape": {"contour": {"start": [0.1, 0.3], "segments": [
                    {"line": [0.25, 0.3]},
                    {"ellipse": {"center": NOSE["center"], "axes": NOSE["axes"], "t": [-math.pi / 2, math.pi / 2]}},
                    {"line": [0.1, 0.5]}]}}},
                {"name": "ball", "material": "lead", "shape": {"disk": {"center": [0.0, -0.4], "radius": 0.08}}},
                {"name": "lens", "material": "ptfe", "shape": {"disk": {"center": [0.5, -0.2], "radius": 0.06}}},
                {"name": "plate", "material": "copper", "shape": _rect((0.5, 0.7), (-0.25, -0.15))},
                {"name": "plug", "material": "iron", "shape": _rect((0.0, 0.04), (0.6, 0.7))}],
            "curves": [{"name": "sheet", "material": "superconductor",
                        "path": {"arc": {"center": [0.0, 0.0], "radius": 0.75, "angles": [-math.pi / 3, math.pi / 3]}}},
                       {"name": "can", "material": "superconductor",
                        "path": {"rect": {"r": [0.05, 0.15], "z": [-0.1, 0.1]}}}],
            "materials": {"lead": {"kind": "vacuum", "density": 11350.0}, "ptfe": {"kind": "vacuum", "density": 2200.0}},
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02, "domain_radius": 1.2}}


CURVED_COMPOSITION = {"copper": {"nist": "G4_Cu"}, "iron": {"nist": "G4_Fe"},
                      "ptfe": {"elements": {"C": 2, "F": 4}, "by": "atoms"}, "lead": {"nist": "G4_Pb"},
                      "air": {"nist": "G4_Galactic", "density": 1e-22}}
CURVED_DENSITY = {"copper": 8960.0, "iron": 7874.0, "ptfe": 2200.0, "lead": 11350.0, "air": 1e-22}


def core_design() -> dict:
    """The CORE template with its iron caps (fillets, frames, unions and differences, caps on the axis), at coarser
    sizes than the template's so that the tests stay quick."""
    from axionmultiphysics import _core

    d = json.loads(_core.template("core", json.dumps({"iron_t": 0.05})))
    d["name"] = "core_with_iron_caps"
    d["mesh"] = dict(d.get("mesh", {}), size="max(b, H)/8", surface_size="2*min(w, gn)")
    return d


CORE_COMPOSITION = {"winding": {"nist": "G4_Cu", "density": 8960.0}, "nb": {"nist": "G4_Nb"},
                    "vacuum": {"nist": "G4_Galactic", "density": 1e-22}, "iron": {"nist": "G4_Fe"},
                    "air": {"nist": "G4_AIR", "density": 1.205}}
CORE_DENSITY = {"winding": 8960.0, "nb": 8570.0, "vacuum": 1e-22, "iron": 7874.0, "air": 1.205}

def clockwise_design() -> dict:
    """Clockwise arcs: two polygons written clockwise with a fillet at every corner (the fillets' arcs run clockwise
    along the outline), one with a reflex corner. Every arc of the table but those of the domain is clockwise from
    ``start`` to ``end`` unless gmsh's parametrization runs the other way."""
    return {"schema": "axionmultiphysics.design/1", "name": "clockwise_fillets",
            "regions": [
                {"name": "a", "material": "m",
                 "shape": {"polygon": {"points": [[0.1, 0.2], [0.1, 0.4], [0.3, 0.4], [0.3, 0.2]], "fillet": 0.04}}},
                {"name": "b", "material": "m",
                 "shape": {"polygon": {"points": [[0.4, -0.4], [0.4, 0.0], [0.6, 0.0], [0.6, -0.2], [0.8, -0.2],
                                                  [0.8, -0.4]], "fillet": 0.05}}}],
            "materials": {"m": {"kind": "vacuum", "density": 1000.0}},
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02, "domain_radius": 1.2}}


# the arc sheet of markers_design: written counter-clockwise over the clockwise fillet of region b
MARKER_ARC = {"center": [0.25, 0.35], "radius": 0.05, "angles": [0.0, math.pi / 2]}


def markers_design() -> dict:
    """Sheets written against gmsh's parametrization: an arc sheet drawn counter-clockwise over the fillet (r 0.05 at
    (0.3, 0.4)) of a square written clockwise, so that the arc entity is reversed; a segment drawn upwards along a
    rectangle's left edge; a segment drawn downwards across a disk (two faces of one region); a filleted rectangular
    sheet in the air (trimmed curves). Region b is the square [0.1, 0.3] x [0.2, 0.4] minus the corner the fillet cuts
    off: area 0.04 - 0.05^2 (1 - pi/4)."""
    return {"schema": "axionmultiphysics.design/1", "name": "reversed_arc_marker",
            "regions": [
                {"name": "a", "material": "m", "shape": {"disk": {"center": [0.6, 0.0], "radius": 0.1}}},
                {"name": "b", "material": "m",
                 "shape": {"polygon": {"points": [[0.1, 0.2], [0.1, 0.4], [0.3, 0.4, 0.05], [0.3, 0.2]]}}},
                {"name": "c", "material": "m", "shape": _rect((0.1, 0.3), (-0.4, -0.2))}],
            "curves": [{"name": "s1", "material": "superconductor", "path": {"segment": [[0.6, 0.3], [0.6, -0.3]]}},
                       {"name": "s2", "material": "superconductor", "path": {"arc": MARKER_ARC}},
                       {"name": "s3", "material": "superconductor", "path": {"segment": [[0.1, -0.4], [0.1, -0.2]]}},
                       {"name": "s4", "material": "superconductor",
                        "path": {"rect": {"r": [0.35, 0.5], "z": [0.5, 0.7], "fillet": 0.03}}}],
            "materials": {"m": {"kind": "vacuum", "density": 1000.0}},
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02, "domain_radius": 1.2}}


PLAIN_COMPOSITION = {"m": {"nist": "G4_WATER"}, "air": {"nist": "G4_Galactic", "density": 1e-22}}
PLAIN_DENSITY = {"m": 1000.0, "air": 1e-22}

DESIGNS = {"yoke": (yoke_design, YOKE_COMPOSITION, YOKE_DENSITY),
           "curved": (curved_design, CURVED_COMPOSITION, CURVED_DENSITY),
           "core": (core_design, CORE_COMPOSITION, CORE_DENSITY),
           "clockwise": (clockwise_design, PLAIN_COMPOSITION, PLAIN_DENSITY),
           "markers": (markers_design, PLAIN_COMPOSITION, PLAIN_DENSITY)}
RECORDED = ("yoke", "curved", "core")      # the EM designs of the reference records (keys of DESIGNS)


def design(name: str) -> dict:
    return DESIGNS[name][0]()


def region_materials(doc: dict) -> dict:
    """{region: material} of the enabled regions of an EM design (the tests' designs enable every region, or the
    CORE caps through iron_t > 0), and the background ``air`` -> ``air``."""
    out = {r["name"]: r.get("material", "vacuum") for r in doc["regions"]}
    out["air"] = "air"
    return out


def solids_materials(name: str) -> dict:
    """The ``materials`` argument of ``from_partition`` that the export must build for a design: each material with
    the density the export must use and its block without ``density``."""
    _, comp, dens = DESIGNS[name]
    return {m: {"density": dens[m], "particle": {k: v for k, v in block.items() if k != "density"}}
            for m, block in comp.items()}


def em_mesh(doc: dict, directory: Path) -> tuple[dict, Path, Path]:
    """Meshes an EM design as ``axmp mesh`` does (``_core.mesh``, the mesher of ``$AXMP_MESHER``) into DIR. Returns the
    mesher input (``mesh_in.json``), the mesh and the info file paths."""
    from axionmultiphysics import _core

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = json.loads(_core.mesh(json.dumps(doc), str(directory), 0, ""))
    assert out["ok"] is True, out
    return (json.loads((directory / "mesh_in.json").read_text("utf-8")), directory / "mesh.msh",
            directory / "mesh_info.json")


@pytest.fixture(scope="module")
def mp():
    """The mesher (``$AXMP_MESHER`` or the build's), also put in ``AXMP_MESHER`` for the in-process core and the
    package for the module's tests; skips without one that loads."""
    import model_mesh_support as S

    path = S.mesher_path()
    if not S.mesher_loads(path):
        pytest.skip("no axmp-mesher that loads (set AXMP_MESHER, or build the axmp-mesher target)")
    old = os.environ.get("AXMP_MESHER")
    os.environ["AXMP_MESHER"] = str(Path(path).resolve())
    yield os.environ["AXMP_MESHER"]
    if old is None:
        os.environ.pop("AXMP_MESHER", None)
    else:
        os.environ["AXMP_MESHER"] = old
