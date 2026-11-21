"""Hand-built partitions and independent references for the tests of ``axionmultiphysics.solids``.

A partition here has the shape of the mesher's info file: the keys of the bounded modes' file (``ok``, ``bounded``,
``geometry``, ``exterior``, ``order``, ``gmsh_version``, ``diagonal_m``, ``tolerance_m``, ``regions`` as
``{name: {"attr", "area_m2", "volume_m3"}}``, ``entities``, ``boundaries``, ``boundary_groups``, ``empty``,
``partial``, ``seconds``), with each entity in the JSON form of ``model_selection.hpp`` (``tag``, ``dim``, ``side``,
``curves``, ``axis``, ``type``, ``lo``, ``hi``, ``point``, ``tangent``, ``length_m``, ``area_m2``, and ``atom`` for a
boundary entity), extended additively by the exact curve parameters and the face loops:

- per entity ``start`` and ``end`` ([3]): its end vertices, ``start`` first along ``tangent``; entities that meet
  share the coordinates of their common vertex bit for bit; a closed curve has ``start == end``;
- per arc (``type`` "circle" or "ellipse") ``center`` ([3]), ``radii`` ([a, b], a >= b; a == b for a circle, also
  for the circular arcs that gmsh reports as "ellipse"), ``major_axis`` ([3], the unit vector of the semi-axis a;
  [1, 0, 0] for a circle) and ``ccw`` (true when the curve turns counter-clockwise in (r, z) from ``start`` to
  ``end``). The arc is c + a cos(t) u + b sin(t) v with v the major axis turned by +90 degrees; t increases from
  start to end when ``ccw`` is true;
- ``faces``: ``[{"tag", "region", "loops"}]``, each final face with its region and its loops of signed entity tags,
  the outer loop first. A loop keeps its face on its left: the outer loop is counter-clockwise, the inner loops
  clockwise. A positive tag runs the entity from ``start`` to ``end``, a negative one from ``end`` to ``start``; so
  the face's region is ``side[0]`` of the entities it runs forward and ``side[1]`` of those it runs backward.

The builder below writes such a table from vertices, edges and loops; the sides, boxes, midpoints, tangents, lengths
and revolved areas are derived here, as the mesher derives them from its geometry. The references (closed forms,
quadrature of the exact shapes, the volume of a polygon revolved as a sum of cone frusta, point membership of each
kind of solid) are written from the geometry, not from the implementation.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy import integrate

TWO_PI = 2.0 * math.pi
DEG = math.pi / 180.0          # CLHEP::deg: pi / 180 in double precision
G4_LENGTH = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "km": 1.0e6, "um": 1.0e-3, "nm": 1.0e-6}   # Geant4's internal mm
G4_ANGLE = {"rad": 1.0, "deg": DEG, "mrad": 1.0e-3}


# ============================================================================================ the partition builder
class Partition:
    """Vertices, edges (entities) and faces of an axisymmetric (or planar) partition; ``table()`` writes the JSON."""

    def __init__(self, geometry: str = "axisymmetric"):
        self.geometry = geometry
        self.edges: list[dict] = []
        self.faces: list[tuple[str, list[list[int]]]] = []

    # ---------------------------------------------------------------- edges; each returns its tag (1, 2, ...)
    def line(self, p, q) -> int:
        self.edges.append({"type": "line", "start": tuple(p), "end": tuple(q)})
        return len(self.edges)

    def arc(self, p, q, center, radii, major_axis=(1.0, 0.0), ccw=True, type=None) -> int:
        a, b = float(radii[0]), float(radii[1])
        assert a >= b > 0
        u = np.asarray(major_axis, float)
        u = u / np.hypot(*u)
        self.edges.append({"type": type or ("circle" if a == b else "ellipse"), "start": tuple(p), "end": tuple(q),
                           "center": tuple(center), "radii": (a, b), "major_axis": (float(u[0]), float(u[1])),
                           "ccw": bool(ccw)})
        return len(self.edges)

    def closed(self, center, radii, major_axis=(1.0, 0.0), type=None) -> int:
        """A full circle or ellipse, counter-clockwise, starting at the end of its major semi-axis."""
        u = np.asarray(major_axis, float)
        u = u / np.hypot(*u)
        p = (center[0] + radii[0] * u[0], center[1] + radii[0] * u[1])
        return self.arc(p, p, center, radii, major_axis, True, type)

    def face(self, region: str, *loops) -> None:
        self.faces.append((region, [list(l) for l in loops]))

    # ---------------------------------------------------------------- the table
    def table(self, regions: list[str]) -> dict:
        sides = {t: [None, None] for t in range(1, len(self.edges) + 1)}
        for region, loops in self.faces:
            for loop in loops:
                for s in loop:
                    k = 0 if s > 0 else 1
                    assert sides[abs(s)][k] is None, f"entity {abs(s)} used twice on one side"
                    sides[abs(s)][k] = region
        entities = []
        lo_all, hi_all = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
        for t, e in enumerate(self.edges, start=1):
            g = Curve(e)
            s0, s1 = (x if x is not None else "exterior" for x in sides[t])
            lo, hi = g.box()
            lo_all, hi_all = np.minimum(lo_all, lo), np.maximum(hi_all, hi)
            axis = (self.geometry == "axisymmetric" and e["type"] == "line" and e["start"][0] == 0.0
                    and e["end"][0] == 0.0)
            pm, tm = g.midpoint_and_tangent()
            rec = {"tag": t, "dim": 1, "side": [s0, s1], "curves": [], "axis": axis, "type": e["type"],
                   "lo": [lo[0], lo[1], 0.0], "hi": [hi[0], hi[1], 0.0], "point": [pm[0], pm[1], 0.0],
                   "tangent": [tm[0], tm[1], 0.0], "length_m": g.length(),
                   "area_m2": 0.0 if (axis or self.geometry != "axisymmetric") else g.revolved_area(),
                   "start": [e["start"][0], e["start"][1], 0.0], "end": [e["end"][0], e["end"][1], 0.0]}
            if e["type"] != "line":
                rec.update({"center": [e["center"][0], e["center"][1], 0.0], "radii": [e["radii"][0], e["radii"][1]],
                            "major_axis": [e["major_axis"][0], e["major_axis"][1], 0.0], "ccw": e["ccw"]})
            entities.append(rec)
        # a boundary entity carries its atom (no named boundaries here: one unnamed atom per side pair, the axis 900)
        pairs = sorted({tuple(sorted(x["side"])) for x in entities if x["side"][0] != x["side"][1] and not x["axis"]})
        for x in entities:
            if x["axis"]:
                x["atom"] = 900
            elif x["side"][0] != x["side"][1]:
                x["atom"] = 2001 + pairs.index(tuple(sorted(x["side"])))
        diag = float(np.hypot(*(hi_all - lo_all)))
        reg = {}
        for i, name in enumerate(regions):
            area = vol = 0.0
            for region, loops in self.faces:
                if region == name:
                    for loop in loops:
                        a, v = loop_area_volume(self.edges, loop)
                        area, vol = area + a, vol + v
            reg[name] = {"attr": i + 1, "area_m2": area}
            if self.geometry == "axisymmetric":
                reg[name]["volume_m3"] = vol
        faces = [{"tag": 101 + i, "region": region, "loops": [list(l) for l in loops]}
                 for i, (region, loops) in enumerate(self.faces)]
        return {"ok": True, "bounded": True, "geometry": self.geometry, "exterior": "none", "order": 1,
                "gmsh_version": "4.13.1", "diagonal_m": diag, "tolerance_m": 1e-9 * diag, "regions": reg,
                "entities": entities, "boundaries": {}, "boundary_groups": [], "empty": {}, "partial": {},
                "faces": faces, "seconds": 0.0123}


class Curve:
    """The exact geometry of one edge of the builder (a line or an arc of an ellipse)."""

    def __init__(self, e: dict):
        self.e = e
        self.p, self.q = np.asarray(e["start"], float), np.asarray(e["end"], float)
        if e["type"] != "line":
            self.c = np.asarray(e["center"], float)
            self.a, self.b = e["radii"]
            self.u = np.asarray(e["major_axis"], float)
            self.v = np.array([-self.u[1], self.u[0]])
            self.sense = 1.0 if e["ccw"] else -1.0
            self.t0 = self.param(self.p)
            t1 = self.param(self.q)
            sweep = (self.sense * (t1 - self.t0)) % TWO_PI
            self.sweep = TWO_PI if sweep < 1e-12 else sweep

    def param(self, x):
        d = np.asarray(x, float) - self.c
        return math.atan2(float(d @ self.v) / self.b, float(d @ self.u) / self.a)

    def at(self, f):
        """Points at the fractions f of the curve (0 at start, 1 at end), shape (n, 2)."""
        f = np.atleast_1d(np.asarray(f, float))
        if self.e["type"] == "line":
            return self.p[None, :] + f[:, None] * (self.q - self.p)[None, :]
        t = self.t0 + self.sense * f * self.sweep
        return (self.c[None, :] + self.a * np.cos(t)[:, None] * self.u[None, :]
                + self.b * np.sin(t)[:, None] * self.v[None, :])

    def deriv(self, f):
        f = np.atleast_1d(np.asarray(f, float))
        if self.e["type"] == "line":
            return np.repeat((self.q - self.p)[None, :], len(f), axis=0)
        t = self.t0 + self.sense * f * self.sweep
        k = self.sense * self.sweep
        return k * (-self.a * np.sin(t)[:, None] * self.u[None, :] + self.b * np.cos(t)[:, None] * self.v[None, :])

    def box(self):
        if self.e["type"] == "line":
            return np.minimum(self.p, self.q), np.maximum(self.p, self.q)
        pts = [self.p, self.q]
        for k in range(2):   # extremes of coordinate k: -a sin t u_k + b cos t v_k = 0
            t = math.atan2(self.b * self.v[k], self.a * self.u[k])
            for tt in (t, t + math.pi):
                f = ((self.sense * (tt - self.t0)) % TWO_PI) / self.sweep
                if f <= 1.0:
                    pts.append(self.at(f)[0])
        pts = np.array(pts)
        return pts.min(axis=0), pts.max(axis=0)

    def midpoint_and_tangent(self):
        m, d = self.at(0.5)[0], self.deriv(0.5)[0]
        return m, d / np.hypot(*d)

    def _gl(self, g):
        """Gauss-Legendre over the curve's fraction with 400 nodes in 8 panels: the integrands here are trigonometric
        polynomials of degree <= 3 in t, integrated to rounding."""
        x, w = np.polynomial.legendre.leggauss(50)
        total = 0.0
        for k in range(8):
            f = (k + 0.5 * (x + 1.0)) / 8.0
            total += float(np.sum(w * g(f))) / 16.0
        return total

    def length(self):
        if self.e["type"] == "line":
            return float(np.hypot(*(self.q - self.p)))
        return self._gl(lambda f: np.hypot(*self.deriv(f).T))

    def revolved_area(self):
        return TWO_PI * self._gl(lambda f: self.at(f)[:, 0] * np.hypot(*self.deriv(f).T))

    def green(self):
        """(integral of r dz, integral of r^2 dz) along the curve: the area and volume/pi terms of Green's theorem."""
        if self.e["type"] == "line":
            (r0, z0), (r1, z1) = self.p, self.q
            return 0.5 * (r0 + r1) * (z1 - z0), (r0 * r0 + r0 * r1 + r1 * r1) * (z1 - z0) / 3.0
        return (self._gl(lambda f: self.at(f)[:, 0] * self.deriv(f)[:, 1]),
                self._gl(lambda f: self.at(f)[:, 0] ** 2 * self.deriv(f)[:, 1]))


def loop_area_volume(edges, loop):
    """Signed area and Pappus volume (pi * integral of r^2 dz) of one loop of signed edge tags."""
    a = v = 0.0
    for s in loop:
        g1, g2 = Curve(edges[abs(s) - 1]).green()
        sign = 1.0 if s > 0 else -1.0
        a += sign * g1
        v += sign * g2
    return a, math.pi * v


# ============================================================================================ fixtures
def native():
    """Exact native solids only: a tube with three nested children and an axis notch (orb), a torus, a tube in a
    tube, and a cone frustum beside it (a generic polycone with straight edges)."""
    P = Partition()
    A0, A1, A2, A3, A4 = (0.0, -0.31), (0.53, -0.31), (0.53, 0.43), (0.2, 0.43), (0.0, 0.43)
    cz, R = 0.07, 0.061
    Nb, Nt = (0.0, cz - R), (0.0, cz + R)
    I = [(0.17, -0.25), (0.41, -0.25), (0.41, -0.07), (0.17, -0.07)]
    Q = [(0.23, -0.19), (0.31, -0.19), (0.31, -0.12), (0.23, -0.12)]
    C1, C2 = (0.11, 0.59), (0.0, 0.59)
    bottom, right, top_r, top_l = P.line(A0, A1), P.line(A1, A2), P.line(A2, A3), P.line(A3, A4)
    ax_top, ax_bot = P.line(A4, Nt), P.line(Nb, A0)
    notch = P.arc(Nb, Nt, (0.0, cz), (R, R), ccw=True, type="ellipse")   # gmsh's report of a clipped disk's arc
    core_axis = P.line(Nt, Nb)
    ring = P.closed((0.29, 0.23), (0.083, 0.083))
    ins = [P.line(I[k], I[(k + 1) % 4]) for k in range(4)]
    pin = [P.line(Q[k], Q[(k + 1) % 4]) for k in range(4)]
    c_side, c_top, c_axis = P.line(A3, C1), P.line(C1, C2), P.line(C2, A4)
    P.face("air", [bottom, right, top_r, top_l, ax_top, -notch, ax_bot], [-ring],
           [-ins[3], -ins[2], -ins[1], -ins[0]])
    P.face("core", [notch, core_axis])
    P.face("ring", [ring])
    P.face("insert", ins, [-pin[3], -pin[2], -pin[1], -pin[0]])
    P.face("pin", pin)
    P.face("cone", [-top_l, c_side, c_top, c_axis])
    regions = ["air", "core", "ring", "insert", "pin", "cone"]
    pi = math.pi
    v_core = 4.0 / 3.0 * pi * R ** 3
    v_ring = 2.0 * pi ** 2 * 0.29 * 0.083 ** 2
    v_pin = pi * (0.31 ** 2 - 0.23 ** 2) * 0.07
    v_ins_outer = pi * (0.41 ** 2 - 0.17 ** 2) * 0.18
    exact = {"air": pi * 0.53 ** 2 * 0.74 - v_core - v_ring - v_ins_outer, "core": v_core, "ring": v_ring,
             "insert": v_ins_outer - v_pin, "pin": v_pin,
             "cone": pi * 0.16 / 3.0 * (0.2 ** 2 + 0.2 * 0.11 + 0.11 ** 2)}
    materials = {
        "air": {"density": 1.205, "particle": {"elements": {"N": 0.755, "O": 0.232, "Ar": 0.013}, "by": "mass",
                                               "state": "gas"}},
        "iron": {"density": 7874.0, "particle": {"nist": "G4_Fe"}},
        "copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}},
        "steel": {"density": 7930.0, "particle": {"elements": {"Fe": 0.70, "Cr": 0.19, "Ni": 0.11}, "by": "mass"}},
        "polyethylene": {"density": 940.0, "particle": {"elements": {"C": 1, "H": 2}, "by": "atoms",
                                                         "mean_excitation_eV": 57.4}},
        "unused": {"density": 1000.0, "particle": {"nist": "G4_WATER"}},
    }
    region_materials = {"air": "air", "core": "iron", "ring": "copper", "insert": "steel", "pin": "iron",
                        "cone": "polyethylene"}

    def region_at(r, z):
        r, z = np.asarray(r, float), np.asarray(z, float)
        out = np.full(r.shape, None, dtype=object)
        out[(r < 0.53) & (z > -0.31) & (z < 0.43)] = "air"
        out[r * r + (z - cz) ** 2 < R * R] = "core"
        out[(r - 0.29) ** 2 + (z - 0.23) ** 2 < 0.083 ** 2] = "ring"
        out[(r > 0.17) & (r < 0.41) & (z > -0.25) & (z < -0.07)] = "insert"
        out[(r > 0.23) & (r < 0.31) & (z > -0.19) & (z < -0.12)] = "pin"
        # the cone: below the slanted side from (0.2, 0.43) to (0.11, 0.59)
        out[(z > 0.43) & (z < 0.59) & (r < 0.2 + (0.11 - 0.2) * (z - 0.43) / 0.16)] = "cone"
        return out

    kinds = {"air": "tube", "core": "orb", "ring": "torus", "insert": "tube", "pin": "tube", "cone": "genericPolycone"}
    tree = {"air": None, "cone": None, "core": "air", "ring": "air", "insert": "air", "pin": "insert"}
    return Fixture(P, regions, materials, region_materials, exact, region_at, kinds, tree)


def discretized():
    """Arcs that are not native: a filleted corner shared by two regions (the arc typed "ellipse" with equal radii),
    a half spheroid on the axis, a full ellipse off the axis turned by 30 degrees, and a circular segment."""
    P = Partition()
    Rf, fc = 0.043, (0.207, 0.137)
    Q1, Q2 = (0.25, 0.137), (0.207, 0.18)
    b0, b1, tr, tl = (0.05, 0.0), (0.25, 0.0), (0.25, 0.18), (0.05, 0.18)
    e_bot, e_rlo = P.line(b0, b1), P.line(b1, Q1)
    fil = P.arc(Q1, Q2, fc, (Rf, Rf), ccw=True, type="ellipse")
    e_top, e_left = P.line(Q2, tl), P.line(tl, b0)
    g1, g2 = P.line(Q1, tr), P.line(tr, Q2)
    P.face("pole", [e_bot, e_rlo, fil, e_top, e_left])
    P.face("gap", [g1, g2, -fil])
    dc, da, db = 0.6, 0.12, 0.07          # the spheroid: semi-axis 0.12 along z, 0.07 along r
    Db, Dt = (0.0, dc - da), (0.0, dc + da)
    dome = P.arc(Db, Dt, (0.0, dc), (da, db), major_axis=(0.0, 1.0), ccw=True)
    d_axis = P.line(Dt, Db)
    P.face("dome", [dome, d_axis])
    ang = math.radians(30.0)
    eto = P.closed((0.36, 0.56), (0.05, 0.03), major_axis=(math.cos(ang), math.sin(ang)))
    P.face("eltorus", [eto])
    Lb, Lt, lc, lR = (0.45, 0.25), (0.45, 0.41), (0.39, 0.33), 0.1
    lens_arc, chord = P.arc(Lb, Lt, lc, (lR, lR), ccw=True), P.line(Lt, Lb)
    P.face("lens", [lens_arc, chord])
    regions = ["pole", "gap", "dome", "eltorus", "lens"]
    pi = math.pi

    def segment_integral(lo, hi, centre_r, radius, height):
        """2 pi times the integral over r in [lo, hi] of r * height(r), with r = centre_r + radius * sin(theta) so
        that the square-root ends of a circle are smooth integrands."""
        clip = lambda x: min(1.0, max(-1.0, x))
        th0, th1 = math.asin(clip((lo - centre_r) / radius)), math.asin(clip((hi - centre_r) / radius))
        f = lambda th: (centre_r + radius * math.sin(th)) * height(centre_r + radius * math.sin(th)) * radius * math.cos(th)
        return TWO_PI * integrate.quad(f, th0, th1, epsabs=0, epsrel=1e-13, limit=200)[0]

    circ = lambda r: math.sqrt(max(Rf * Rf - (r - fc[0]) ** 2, 0.0))
    v_gap = segment_integral(fc[0], 0.25, fc[0], Rf, lambda r: 0.18 - fc[1] - circ(r))
    v_pole = (pi * (fc[0] ** 2 - 0.05 ** 2) * 0.18
              + segment_integral(fc[0], 0.25, fc[0], Rf, lambda r: fc[1] + circ(r)))
    lens_h = lambda r: 2.0 * math.sqrt(max(lR * lR - (r - lc[0]) ** 2, 0.0))
    exact = {"pole": v_pole, "gap": v_gap, "dome": 4.0 / 3.0 * pi * db * db * da,
             "eltorus": TWO_PI * 0.36 * pi * 0.05 * 0.03,
             "lens": segment_integral(0.45, lc[0] + lR, lc[0], lR, lens_h)}
    materials = {
        "iron": {"density": 7874.0, "particle": {"nist": "G4_Fe"}},
        "helium": {"density": 0.1786, "particle": {"elements": {"He": 1}, "by": "atoms", "state": "gas",
                                                   "T_K": 273.15, "pressure_Pa": 101325.0}},
        "aluminium": {"density": 2699.0, "particle": {"nist": "G4_Al"}},
        "copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}},
        "glass": {"density": 2200.0, "particle": {"elements": {"Si": 0.4675, "O": 0.5325}, "by": "mass"}},
    }
    region_materials = {"pole": "iron", "gap": "helium", "dome": "aluminium", "eltorus": "copper", "lens": "glass"}

    def region_at(r, z):
        r, z = np.asarray(r, float), np.asarray(z, float)
        out = np.full(r.shape, None, dtype=object)
        box = (r > 0.05) & (r < 0.25) & (z > 0.0) & (z < 0.18)
        corner = (r > fc[0]) & (z > fc[1]) & ((r - fc[0]) ** 2 + (z - fc[1]) ** 2 > Rf * Rf)
        out[box] = "pole"
        out[box & corner] = "gap"
        out[(r / db) ** 2 + ((z - dc) / da) ** 2 < 1.0] = "dome"
        x, y = r - 0.36, z - 0.56
        xu, yv = x * math.cos(ang) + y * math.sin(ang), -x * math.sin(ang) + y * math.cos(ang)
        out[(xu / 0.05) ** 2 + (yv / 0.03) ** 2 < 1.0] = "eltorus"
        out[(r > 0.45) & ((r - lc[0]) ** 2 + (z - lc[1]) ** 2 < lR * lR)] = "lens"
        return out

    kinds = {k: "genericPolycone" for k in regions}
    tree = {k: None for k in regions}
    fx = Fixture(P, regions, materials, region_materials, exact, region_at, kinds, tree)
    fx.arcs = {"fillet": fil, "dome": dome, "eltorus": eto, "lens": lens_arc}
    return fx


def voids():
    """Holes of the domain: a tube with an exterior hole and a hole filled by two faces of one region that enclose
    an exterior void between them; a spherical shell (a half annulus on the axis) around an exterior notch."""
    P = Partition()
    J = [(0.2, -0.2), (0.95, -0.2), (0.95, 0.45), (0.2, 0.45)]
    H = [(0.28, 0.05), (0.4, 0.05), (0.4, 0.25), (0.28, 0.25)]
    jo = [P.line(J[k], J[(k + 1) % 4]) for k in range(4)]
    ho = [P.line(H[k], H[(k + 1) % 4]) for k in range(4)]      # counter-clockwise around the exterior hole
    # the liner: [0.6, 0.9] x [0, 0.3] in two faces, with the exterior void [0.68, 0.82] x [0.1, 0.2] between them
    a_bot = P.line((0.6, 0.0), (0.75, 0.0))
    mid_lo = P.line((0.75, 0.0), (0.75, 0.1))
    a_n = [P.line((0.75, 0.1), (0.68, 0.1)), P.line((0.68, 0.1), (0.68, 0.2)), P.line((0.68, 0.2), (0.75, 0.2))]
    mid_hi = P.line((0.75, 0.2), (0.75, 0.3))
    a_top, a_left = P.line((0.75, 0.3), (0.6, 0.3)), P.line((0.6, 0.3), (0.6, 0.0))
    b_bot, b_right, b_top = P.line((0.75, 0.0), (0.9, 0.0)), P.line((0.9, 0.0), (0.9, 0.3)), P.line((0.9, 0.3), (0.75, 0.3))
    b_n = [P.line((0.75, 0.2), (0.82, 0.2)), P.line((0.82, 0.2), (0.82, 0.1)), P.line((0.82, 0.1), (0.75, 0.1))]
    P.face("jacket", jo, [-ho[3], -ho[2], -ho[1], -ho[0]], [-a_left, -a_top, -b_top, -b_right, -b_bot, -a_bot])
    P.face("liner", [a_bot, mid_lo, *a_n, mid_hi, a_top, a_left])
    P.face("liner", [b_bot, b_right, b_top, -mid_hi, *b_n, -mid_lo])
    sc, sa, sb = 0.8, 0.05, 0.11
    outer = P.arc((0.0, sc - sb), (0.0, sc + sb), (0.0, sc), (sb, sb), ccw=True)
    inner = P.arc((0.0, sc - sa), (0.0, sc + sa), (0.0, sc), (sa, sa), ccw=True)
    ax_t, ax_b = P.line((0.0, sc + sb), (0.0, sc + sa)), P.line((0.0, sc - sa), (0.0, sc - sb))
    P.face("shell", [outer, ax_t, -inner, ax_b])
    regions = ["jacket", "liner", "shell"]
    pi = math.pi
    tube = lambda r0, r1, h: pi * (r1 * r1 - r0 * r0) * h
    exact = {"jacket": tube(0.2, 0.95, 0.65) - tube(0.28, 0.4, 0.2) - tube(0.6, 0.9, 0.3),
             "liner": tube(0.6, 0.9, 0.3) - tube(0.68, 0.82, 0.1),
             "shell": 4.0 / 3.0 * pi * (sb ** 3 - sa ** 3)}
    materials = {
        "steel": {"density": 7930.0, "particle": {"elements": {"Fe": 0.70, "Cr": 0.19, "Ni": 0.11}, "by": "mass"}},
        "ptfe": {"density": 2200.0, "particle": {"elements": {"C": 2, "F": 4}, "by": "atoms"}},
        "lead": {"density": 11350.0, "particle": {"nist": "G4_Pb"}},
    }
    region_materials = {"jacket": "steel", "liner": "ptfe", "shell": "lead"}

    def region_at(r, z):
        r, z = np.asarray(r, float), np.asarray(z, float)
        out = np.full(r.shape, None, dtype=object)
        out[(r > 0.2) & (r < 0.95) & (z > -0.2) & (z < 0.45)] = "jacket"
        out[(r > 0.28) & (r < 0.4) & (z > 0.05) & (z < 0.25)] = None
        out[(r > 0.6) & (r < 0.9) & (z > 0.0) & (z < 0.3)] = "liner"
        out[(r > 0.68) & (r < 0.82) & (z > 0.1) & (z < 0.2)] = None
        rho2 = r * r + (z - sc) ** 2
        out[(rho2 < sb * sb) & (rho2 > sa * sa)] = "shell"
        return out

    kinds = {"jacket": "tube", "liner": "genericPolycone", "shell": "orb"}
    tree = {"jacket": None, "liner": "jacket", "shell": None}
    fx = Fixture(P, regions, materials, region_materials, exact, region_at, kinds, tree)
    # the voids: (kind, mother region or None, cross-section) of each exterior hole
    fx.voids = [("tube", "jacket", (0.28, 0.4, 0.05, 0.25)), ("tube", "jacket", (0.68, 0.82, 0.1, 0.2)),
                ("orb", "shell", (sa, sc))]
    return fx


def flipped(fx):
    """The same partition with every curve run the other way, as another run of the mesher may write it: start and
    end swapped, the tangent reversed, the sides swapped (side[0] stays the region on the left), ``ccw`` negated (every
    arc of the fixtures then turns clockwise) and the signs in the loops flipped, so that every loop is traversed as
    before. Boxes, midpoints, lengths and atoms are unchanged."""
    g = copy.copy(fx)
    t = fx.table()
    for e in t["entities"]:
        e["start"], e["end"] = e["end"], e["start"]
        e["side"] = [e["side"][1], e["side"][0]]
        e["tangent"] = [-x for x in e["tangent"]]
        if "ccw" in e:
            e["ccw"] = not e["ccw"]
    for f in t["faces"]:
        f["loops"] = [[-s for s in loop] for loop in f["loops"]]
    g._table = t
    return g


def irregular(seed: int = 314159):
    """Star-shaped polygons with coordinates drawn from a fixed seed: three faces of one region (one with two holes,
    filled by two faces of a second region). The region volumes are sums of many edge terms of mixed sign and size,
    so their floating-point sum depends on the order in which the terms are added; the exact references are the
    exactly rounded sums (math.fsum) of the frusta under the edges."""
    import random
    rng = random.Random(seed)
    P = Partition()
    terms = {"shard": [], "chip": []}

    def star(cr, cz, R0, n, region_sign):
        th = [TWO_PI * (k + rng.uniform(-0.3, 0.3)) / n for k in range(n)]
        pts = [(cr + R0 * rng.uniform(0.6, 1.0) * math.cos(a), cz + R0 * rng.uniform(0.6, 1.0) * math.sin(a))
               for a in th]
        tags = [P.line(pts[k], pts[(k + 1) % n]) for k in range(n)]
        for k in range(n):
            (r0, z0), (r1, z1) = pts[k], pts[(k + 1) % n]
            for region, sign in region_sign:
                terms[region].append(sign * math.pi / 3.0 * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1))
        return tags

    a = star(0.31, 0.02, 0.1, 13, [("shard", 1)])
    c1 = star(0.33, 0.025, 0.015, 9, [("chip", 1), ("shard", -1)])
    c2 = star(0.29, 0.015, 0.015, 7, [("chip", 1), ("shard", -1)])
    b = star(0.57, 0.03, 0.1, 11, [("shard", 1)])
    c = star(0.33, 0.29, 0.1, 17, [("shard", 1)])
    P.face("shard", a, [-t for t in reversed(c1)], [-t for t in reversed(c2)])
    P.face("chip", c1)
    P.face("chip", c2)
    P.face("shard", b)
    P.face("shard", c)
    regions = ["shard", "chip"]
    exact = {r: math.fsum(v) for r, v in terms.items()}
    materials = {"steel": {"density": 7930.0, "particle": {"elements": {"Fe": 0.70, "Cr": 0.19, "Ni": 0.11},
                                                           "by": "mass"}},
                 "lead": {"density": 11350.0, "particle": {"nist": "G4_Pb"}}}
    region_materials = {"shard": "steel", "chip": "lead"}
    kinds = {"shard": "genericPolycone", "chip": "genericPolycone"}
    tree = {"shard": None, "chip": "shard"}
    return Fixture(P, regions, materials, region_materials, exact, None, kinds, tree)


def thin_film(gap: float, kind: str = "ellipse"):
    """A film thinner than the chord tolerance around a core, off the axis.

    kind "ellipse": between two concentric closed ellipses (a = 0.1 m, b = 0.05 m at r = 0.3 m; the inner one,
    smaller by ``gap`` in both semi-axes, described from the other end of its major axis), so that the two
    discretized chains cross and each face's inner point can fall in the other's polygon.
    kind "torus": between a circle split into two arcs (the film's outer loop, a discretized polycone) and a closed
    circle of radius R - gap inside it (the core, an exact torus that pokes out of the chords of the film)."""
    P = Partition()
    c = (0.3, 0.0)
    if kind == "ellipse":
        eo = P.closed(c, (0.1, 0.05))
        ei = P.closed(c, (0.1 - gap, 0.05 - gap), major_axis=(-1.0, 0.0))
        P.face("film", [eo], [-ei])
        P.face("core", [ei])
        v_in = TWO_PI * c[0] * math.pi * (0.1 - gap) * (0.05 - gap)
        exact = {"film": TWO_PI * c[0] * math.pi * 0.1 * 0.05 - v_in, "core": v_in}
    else:
        R = 0.1
        up = P.arc((0.4, 0.0), (0.2, 0.0), c, (R, R))
        lo = P.arc((0.2, 0.0), (0.4, 0.0), c, (R, R))
        ei = P.closed(c, (R - gap, R - gap))
        P.face("film", [up, lo], [-ei])
        P.face("core", [ei])
        v_in = 2.0 * math.pi ** 2 * c[0] * (R - gap) ** 2
        exact = {"film": 2.0 * math.pi ** 2 * c[0] * R * R - v_in, "core": v_in}
    materials = {"niobium": {"density": 8570.0, "particle": {"nist": "G4_Nb"}},
                 "copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}}}
    return Fixture(P, ["film", "core"], materials, {"film": "niobium", "core": "copper"}, exact, None,
                   {}, {"film": None, "core": "film"})


def bite(gap: float, scale: float = 1.0):
    """A plate ([0.3, 0.6] x [-0.15, 0.15]) with a half-disk bitten out of its right side (radius 0.08 about
    (0.6, 0)), and a separate exact torus of radius 0.08 - gap about the same centre: the plate's chords of the
    concave bite lie inside the circle by their sagitta, so a gap below the sagitta makes the two siblings overlap.
    ``scale`` multiplies every length of the shape (not the gap)."""
    k = scale
    P = Partition()
    c, R = (0.6 * k, 0.0), 0.08 * k
    l1, l2 = P.line((0.3 * k, -0.15 * k), (0.6 * k, -0.15 * k)), P.line((0.6 * k, -0.15 * k), (0.6 * k, -R))
    arc = P.arc((0.6 * k, -R), (0.6 * k, R), c, (R, R), ccw=False)
    l3, l4 = P.line((0.6 * k, R), (0.6 * k, 0.15 * k)), P.line((0.6 * k, 0.15 * k), (0.3 * k, 0.15 * k))
    l5 = P.line((0.3 * k, 0.15 * k), (0.3 * k, -0.15 * k))
    P.face("plate", [l1, l2, arc, l3, l4, l5])
    P.face("ball", [P.closed(c, (R - gap, R - gap))])
    mats = {"copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}}}
    exact = {"plate": math.pi * (0.6 ** 2 - 0.3 ** 2) * 0.3 * k ** 3
             - (TWO_PI * (0.6 * k - 4 * R / (3 * math.pi)) * math.pi * R * R / 2),
             "ball": 2.0 * math.pi ** 2 * 0.6 * k * (R - gap) ** 2}
    return Fixture(P, ["plate", "ball"], mats, {"plate": "copper", "ball": "copper"}, exact, None, {}, {})


def pinch(tilt: float = 0.0):
    """A face (region "c", about [0.2, 0.4] x [-0.1, 0.1]) whose outer loop touches itself at the point (0.3, 0.1): a
    triangular notch, the face of region "t", reaches the top boundary with its tip. The polycone of "c" would pass
    twice through that point. With ``tilt`` the top boundary is sloped on both sides of the point (its corners
    are at z = 0.1 + 0.3 tilt on the right and 0.1 - 0.2 tilt on the left)."""
    P = Partition()
    p, a, b = (0.3, 0.1), (0.25, 0.0), (0.35, 0.0)
    tr, tl = (0.4, 0.1 + 0.3 * tilt), (0.2, 0.1 - 0.2 * tilt)
    pts = [(0.2, -0.1), (0.4, -0.1), tr, p]
    e = [P.line(pts[k], pts[k + 1]) for k in range(3)]
    pb, ba, ap = P.line(p, b), P.line(b, a), P.line(a, p)
    e2, e3 = P.line(p, tl), P.line(tl, (0.2, -0.1))
    P.face("c", e + [pb, ba, ap, e2, e3])
    P.face("t", [-ap, -ba, -pb])
    mats = {"copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}},
            "iron": {"density": 7874.0, "particle": {"nist": "G4_Fe"}}}
    return Fixture(P, ["c", "t"], mats, {"c": "copper", "t": "iron"}, {}, None, {}, {})


def slot(gap: float):
    """One face with a slot ``gap`` wide between two concentric arcs (radii 0.08 and 0.08 + gap about (0.6, 0)):
    the plate [0.3, 0.6] x [-0.15, 0.15] runs along the outer arc clockwise from -90 to 165 degrees, crosses to the
    inner arc at 180 degrees and comes back along it to -90 degrees. The outer arc's chords lie inside its circle by
    their sagitta, so a slot narrower than that makes the face's outline cross itself."""
    P = Partition()
    c, R1 = (0.6, 0.0), 0.08
    R2 = R1 + gap
    end = (c[0] + R2 * math.cos(math.radians(165.0)), R2 * math.sin(math.radians(165.0)))
    l1, l2 = P.line((0.3, -0.15), (0.6, -0.15)), P.line((0.6, -0.15), (0.6, -R2))
    a2 = P.arc((0.6, -R2), end, c, (R2, R2), ccw=False)
    tip = P.line(end, (c[0] - R1, 0.0))
    a1 = P.arc((c[0] - R1, 0.0), (0.6, -R1), c, (R1, R1), ccw=True)
    l3, l4, l5 = P.line((0.6, -R1), (0.6, 0.15)), P.line((0.6, 0.15), (0.3, 0.15)), P.line((0.3, 0.15), (0.3, -0.15))
    P.face("f", [l1, l2, a2, tip, a1, l3, l4, l5])
    mats = {"copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}}}
    a, v = loop_area_volume(P.edges, P.faces[0][1][0])
    return Fixture(P, ["f"], mats, {"f": "copper"}, {"f": v}, None, {}, {})


def tie(reverse: bool):
    """A square cut along its diagonal into two faces of one region: the faces have the same box and area."""
    P = Partition()
    l1, l2 = P.line((0.1, 0.0), (0.2, 0.0)), P.line((0.2, 0.0), (0.2, 0.1))
    d = P.line((0.2, 0.1), (0.1, 0.0))
    l3, l4 = P.line((0.2, 0.1), (0.1, 0.1)), P.line((0.1, 0.1), (0.1, 0.0))
    faces = [("t", [l1, l2, d]), ("t", [-d, l3, l4])]
    for region, loop in (faces[::-1] if reverse else faces):
        P.face(region, loop)
    v = math.pi * (0.2 ** 2 - 0.1 ** 2) * 0.1
    return Fixture(P, ["t"], {"copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}}}, {"t": "copper"},
                   {"t": v}, None, {"t": "genericPolycone"}, {"t": None})


def degenerate():
    """Touching at single points: a triangle with one vertex on the axis, inside a box whose loop passes through that
    axis point twice; two squares that share only a vertex; a closed circle tangent to the axis."""
    P = Partition()
    a1, a2, a3 = P.line((0.0, -0.2), (0.3, -0.2)), P.line((0.3, -0.2), (0.3, 0.2)), P.line((0.3, 0.2), (0.0, 0.2))
    ax1, ax2 = P.line((0.0, 0.2), (0.0, 0.0)), P.line((0.0, 0.0), (0.0, -0.2))
    t1, t2, t3 = P.line((0.0, 0.0), (0.2, -0.1)), P.line((0.2, -0.1), (0.2, 0.1)), P.line((0.2, 0.1), (0.0, 0.0))
    P.face("box", [a1, a2, a3, ax1, -t3, -t2, -t1, ax2])
    P.face("tri", [t1, t2, t3])
    q1 = [(0.4, 0.0), (0.5, 0.0), (0.5, 0.1), (0.4, 0.1)]
    q2 = [(0.5, 0.1), (0.6, 0.1), (0.6, 0.2), (0.5, 0.2)]
    P.face("sq1", [P.line(q1[k], q1[(k + 1) % 4]) for k in range(4)])
    P.face("sq2", [P.line(q2[k], q2[(k + 1) % 4]) for k in range(4)])
    P.face("tang", [P.closed((0.05, -0.5), (0.05, 0.05))])
    regions = ["box", "tri", "sq1", "sq2", "tang"]
    pi = math.pi
    v_tri = 2.0 * pi * 0.2 ** 3 / 3.0             # z in [-r/2, r/2] for r in [0, 0.2]
    exact = {"box": pi * 0.3 ** 2 * 0.4 - v_tri, "tri": v_tri, "sq1": pi * (0.5 ** 2 - 0.4 ** 2) * 0.1,
             "sq2": pi * (0.6 ** 2 - 0.5 ** 2) * 0.1, "tang": 2.0 * pi ** 2 * 0.05 * 0.05 ** 2}
    materials = {"air": {"density": 1.205, "particle": {"elements": {"N": 0.755, "O": 0.232, "Ar": 0.013},
                                                        "by": "mass", "state": "gas"}},
                 "copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}}}
    rm = {k: "copper" for k in regions}
    rm["box"] = "air"

    def region_at(r, z):
        r, z = np.asarray(r, float), np.asarray(z, float)
        out = np.full(r.shape, None, dtype=object)
        out[(r < 0.3) & (np.abs(z) < 0.2)] = "box"
        out[(r < 0.2) & (np.abs(z) < 0.5 * r)] = "tri"
        out[(r > 0.4) & (r < 0.5) & (z > 0.0) & (z < 0.1)] = "sq1"
        out[(r > 0.5) & (r < 0.6) & (z > 0.1) & (z < 0.2)] = "sq2"
        out[(r - 0.05) ** 2 + (z + 0.5) ** 2 < 0.05 ** 2] = "tang"
        return out

    return Fixture(P, regions, materials, rm, exact, region_at, {}, {})


def renamed(fx, regions: dict, materials: dict):
    """The fixture with its regions and materials renamed (``{old: new}``), as a document may name them."""
    g = copy.copy(fx)
    t = fx.table()
    t["regions"] = {regions.get(k, k): v for k, v in t["regions"].items()}
    for e in t["entities"]:
        e["side"] = [regions.get(x, x) for x in e["side"]]
    for f in t["faces"]:
        f["region"] = regions.get(f["region"], f["region"])
    g._table = t
    g.regions = [regions.get(k, k) for k in fx.regions]
    g.materials = {materials.get(k, k): v for k, v in fx.materials.items()}
    g.region_materials = {regions.get(k, k): materials.get(v, v) for k, v in fx.region_materials.items()}
    g.exact = {regions.get(k, k): v for k, v in fx.exact.items()}
    return g


class Fixture:
    def __init__(self, P, regions, materials, region_materials, exact, region_at, kinds, tree):
        self.P, self.regions = P, regions
        self.materials, self.region_materials = materials, region_materials
        self.exact, self.region_at, self.kinds, self.tree = exact, region_at, kinds, tree
        self._table = P.table(regions)
        self.arcs, self.voids = {}, []

    def table(self) -> dict:
        return copy.deepcopy(self._table)

    def args(self):
        return self.table(), copy.deepcopy(self.materials), dict(self.region_materials)


# ============================================================================================ references on a solids document
def polygon_volume_frusta(rz) -> float:
    """Volume of a closed (r, z) polygon revolved by 2 pi, as the signed sum of the cone frusta under its edges:
    pi/3 (z2 - z1)(r1^2 + r1 r2 + r2^2). Positive for a counter-clockwise polygon in (r, z)."""
    p = np.asarray(rz, float)
    r1, z1 = p[:, 0], p[:, 1]
    r2, z2 = np.roll(r1, -1), np.roll(z1, -1)
    return float(math.pi / 3.0 * np.sum((z2 - z1) * (r1 * r1 + r1 * r2 + r2 * r2)))


def polygon_signed_area(rz) -> float:
    p = np.asarray(rz, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def solid_volume(s) -> float:
    """The volume of one solid of the document from its parameters (closed forms of each kind)."""
    p = s["params"]
    turn = p.get("deltaphi_deg", 360.0) / 360.0
    if s["kind"] == "tube":
        return turn * math.pi * (p["rmax_m"] ** 2 - p["rmin_m"] ** 2) * p["z_m"]
    if s["kind"] == "torus":
        return turn * 2.0 * math.pi ** 2 * p["rtor_m"] * (p["rmax_m"] ** 2 - p["rmin_m"] ** 2)
    if s["kind"] == "orb":
        return 4.0 / 3.0 * math.pi * p["r_m"] ** 3
    if s["kind"] == "genericPolycone":
        return turn * abs(polygon_volume_frusta(p["rz_m"]))
    raise AssertionError(f"unknown kind {s['kind']}")


def absolute_z(doc) -> dict:
    """The absolute z of each solid's frame: the sum of the placements along its chain of mothers."""
    by = {s["name"]: s for s in doc["solids"]}
    out = {}

    def z_of(name):
        if name not in out:
            s = by[name]
            out[name] = s["position_m"][2] + (z_of(s["mother"]) if s["mother"] else 0.0)
        return out[name]

    for n in by:
        z_of(n)
    return out


def contains(s, z0, r, z, tol=0.0):
    """Mask of the points (r >= 0, z) inside the cross-section of solid s placed at absolute z0, within tol."""
    p = s["params"]
    r, z = np.asarray(r, float), np.asarray(z, float)
    if s["kind"] == "tube":
        return (r >= p["rmin_m"] - tol) & (r <= p["rmax_m"] + tol) & (np.abs(z - z0) <= 0.5 * p["z_m"] + tol)
    if s["kind"] == "torus":
        d = np.hypot(r - p["rtor_m"], z - z0)
        return (d <= p["rmax_m"] + tol) & (d >= p["rmin_m"] - tol)
    if s["kind"] == "orb":
        return np.hypot(r, z - z0) <= p["r_m"] + tol
    if s["kind"] == "genericPolycone":
        poly = np.asarray(p["rz_m"], float) + np.array([0.0, z0])
        inside = point_in_polygon(poly, r, z)
        if tol > 0:
            inside |= distance_to_polygon(poly, r, z) <= tol
        return inside
    raise AssertionError(s["kind"])


def depth(s, z0, r, z):
    """Signed depth of the points inside solid s: positive inside (the distance to its boundary), negative outside."""
    p = s["params"]
    r, z = np.asarray(r, float), np.asarray(z, float)
    if s["kind"] == "tube":
        return np.minimum.reduce([r - p["rmin_m"], p["rmax_m"] - r, 0.5 * p["z_m"] - np.abs(z - z0)])
    if s["kind"] == "torus":
        return p["rmax_m"] - np.hypot(r - p["rtor_m"], z - z0)
    if s["kind"] == "orb":
        return p["r_m"] - np.hypot(r, z - z0)
    poly = np.asarray(p["rz_m"], float) + np.array([0.0, z0])
    d = distance_to_polygon(poly, r, z)
    return np.where(point_in_polygon(poly, r, z), d, -d)


def boundary_samples(s, z0, n=64):
    """Points on the boundary of the (r >= 0) cross-section of solid s: n per straight side or arc."""
    p = s["params"]
    f = (np.arange(n) + 0.5) / n
    if s["kind"] == "tube":
        r0, r1, h = p["rmin_m"], p["rmax_m"], 0.5 * p["z_m"]
        c = [(r0, z0 - h), (r1, z0 - h), (r1, z0 + h), (r0, z0 + h)]
        poly = np.array(c)
    elif s["kind"] == "torus":
        t = TWO_PI * f
        return np.column_stack([p["rtor_m"] + p["rmax_m"] * np.cos(t), z0 + p["rmax_m"] * np.sin(t)])
    elif s["kind"] == "orb":
        t = math.pi * (f - 0.5)
        arc = np.column_stack([p["r_m"] * np.cos(t), z0 + p["r_m"] * np.sin(t)])
        axis = np.column_stack([np.zeros(n), z0 + p["r_m"] * (2 * f - 1)])
        return np.vstack([arc, axis])
    else:
        poly = np.asarray(p["rz_m"], float) + np.array([0.0, z0])
    pts = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        pts.append(a[None, :] + f[:, None] * (b - a)[None, :])
        pts.append(a[None, :])
    return np.vstack(pts)


def point_in_polygon(poly, x, y):
    """Crossing-number test, vectorized over the points; points exactly on an edge may go either way."""
    poly = np.asarray(poly, float)
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = np.zeros(np.broadcast(x, y).shape, bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for a, b, c, d in zip(x1, y1, x2, y2):
        crosses = (b > y) != (d > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = a + (y - b) * (c - a) / (d - b)
        inside ^= crosses & (x < xi)
    return inside


def distance_to_polygon(poly, x, y):
    poly = np.asarray(poly, float)
    x, y = np.asarray(x, float), np.asarray(y, float)
    best = np.full(np.broadcast(x, y).shape, np.inf)
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        d = b - a
        L2 = float(d @ d)
        t = np.clip(((x - a[0]) * d[0] + (y - a[1]) * d[1]) / L2, 0.0, 1.0) if L2 > 0 else 0.0
        best = np.minimum(best, np.hypot(x - (a[0] + t * d[0]), y - (a[1] + t * d[1])))
    return best


def simple_polygon(poly) -> bool:
    """Whether no two non-adjacent edges of the closed polygon cross, or overlap along a common line."""
    p = np.asarray(poly, float)
    n = len(p)
    a, b = p, np.roll(p, -1, axis=0)

    def orient(o, d, x):
        return (d[..., 0] - o[..., 0]) * (x[..., 1] - o[..., 1]) - (d[..., 1] - o[..., 1]) * (x[..., 0] - o[..., 0])

    for i in range(n):
        j = np.arange(i + 2, n if i > 0 else n - 1)
        if not len(j):
            continue
        d1, d2 = orient(a[i], b[i], a[j]), orient(a[i], b[i], b[j])
        d3, d4 = orient(a[j], b[j], a[i][None, :]), orient(a[j], b[j], b[i][None, :])
        if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
            return False
        col = (d1 == 0) & (d2 == 0)
        if np.any(col):   # collinear: overlapping projections on edge i
            u = b[i] - a[i]
            L2 = float(u @ u)
            ta, tb = (a[j][col] - a[i]) @ u / L2, (b[j][col] - a[i]) @ u / L2
            if np.any(np.maximum(np.minimum(ta, tb), 0.0) < np.minimum(np.maximum(ta, tb), 1.0)):
                return False
    return True


def navigate(doc, r, z):
    """Geant4's view of the document: from the world, step into the one daughter that contains each point, and
    report the region of the deepest volume (None for the world and for voids). Raises when a point lies inside two
    daughters of one volume (an overlap) or outside the world."""
    by = {s["name"]: s for s in doc["solids"]}
    kids = {n: [] for n in by}
    for s in doc["solids"]:
        if s["mother"] is not None:
            kids[s["mother"]].append(s["name"])
    zabs = absolute_z(doc)
    r, z = np.asarray(r, float).ravel(), np.asarray(z, float).ravel()
    world = doc["world"]
    inside = {n: contains(by[n], zabs[n], r, z) for n in by}
    assert inside[world].all(), "a point of the partition lies outside the world"
    out = np.full(r.shape, None, dtype=object)

    def walk(name, idx):
        if len(idx) == 0:
            return
        count = np.zeros(len(idx), int)
        for k in kids[name]:
            count += inside[k][idx]
        if count.max(initial=0) > 1:
            j = idx[np.argmax(count)]
            hit = [k for k in kids[name] if inside[k][j]]
            raise AssertionError(f"daughters {hit} of '{name}' overlap at (r, z) = ({r[j]:.6g}, {z[j]:.6g})")
        taken = np.zeros(len(idx), bool)
        for k in kids[name]:
            m = inside[k][idx]
            taken |= m
            walk(k, idx[m])
        out[idx[~taken]] = by[name]["region"]

    walk(world, np.arange(len(r)))
    return out


def interior_grid(fx, n=181, margin=2e-4):
    """Grid points over the fixture's box, without those within ``margin`` of a region boundary (where the
    discretized solids differ from the exact shapes by up to the chord tolerance)."""
    t = fx.table()
    lo = np.min([e["lo"][:2] for e in t["entities"]], axis=0)
    hi = np.max([e["hi"][:2] for e in t["entities"]], axis=0)
    g0 = np.linspace(lo[0], hi[0], n) + 0.123456789e-3   # offsets that keep the grid off the fixtures' round numbers
    g1 = np.linspace(lo[1], hi[1], n) + 0.0987654321e-3
    R, Z = np.meshgrid(g0, g1)
    R, Z = R.ravel(), Z.ravel()
    box = (R <= hi[0]) & (Z <= hi[1])
    R, Z = R[box], Z[box]
    ref = fx.region_at(R, Z)
    keep = np.ones(len(R), bool)
    for dr, dz in ((margin, 0), (-margin, 0), (0, margin), (0, -margin), (margin, margin), (-margin, -margin),
                   (margin, -margin), (-margin, margin)):
        keep &= fx.region_at(np.maximum(R + dr, 0.0), Z + dz) == ref
    keep &= R > margin
    return R[keep], Z[keep], ref[keep]
