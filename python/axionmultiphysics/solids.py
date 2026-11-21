"""Solids from an axisymmetric or 3D partition, and their GDML export (beta).

The particle transport needs the design as a list of native solids, each placed in its mother. This module builds that
list from the mesher's partition (its entity table with the exact curve parameters and the oriented loops of every
face) and writes it as ``solids.json`` (schema ``axionmultiphysics.solids/1``) or as GDML:

- each face's outer loop, with its notches at the axis filled, is revolved by a full turn: a rectangle is a tube, a
  closed circle off the axis a torus, a half circle on the axis an orb, anything else a generic polycone (whose arcs
  are cut into chords within ``chord_tol``, each curve once, so that neighbours share every vertex);
- the faces nested in a face (in a hole or in a notch) are its daughters; a hole of the domain is a vacuum daughter
  (``void_<k>``), so that it is never filled with the material around it; the world is a vacuum tube around everything;
- per region the report gives the exact volume of the solid of revolution (Pappus, from the curve parameters), the
  volume of the instantiated solids, their masses, the largest sagitta and the relative difference, with a
  ``particle.volume_deficit`` warning above ``VOLUME_WARN``;
- the GDML file holds the same numbers as the list, each written with 17 significant digits, and references NIST
  materials and elements by name without defining them.

A 3D partition (the mesher's ``--export-regions`` on a 3D design, ``"geometry": "3d"``) becomes one **tessellated**
solid per volume: the triangles of its outer shell, turned outwards and copied bit for bit from the partition's node
list, so that solids that touch have the same facets on the face between them; the volumes in a cavity of a volume are
its daughters, a volume in no cavity a daughter of the world (a vacuum orb around every node).

Nothing here needs gmsh or Geant4 (numpy and the standard library). ``docs/guides/solids.md`` describes the format, the
rules and every error code. This module is **beta**: its interface may still change.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import cmath
import json
import math
import re
import warnings
from os import PathLike
from pathlib import Path
from typing import Mapping
from xml.sax.saxutils import quoteattr

import numpy as np

from . import _core
from ._beta import warn_beta

__all__ = ["SCHEMA", "VOLUME_WARN", "SolidsWarning", "default_chord_tol", "from_partition", "write", "write_gdml"]

SCHEMA = "axionmultiphysics.solids/1"
VOLUME_WARN = 1e-4            # relative volume or mass difference above which a region is warned about
CHORD_TOL_MAX_M = 1e-6        # the default chord tolerance: min(1 um, 1e-5 x the region's size)
CHORD_TOL_REL = 1e-5
WORLD_MARGIN = 1e-3           # the world's margin around the partition, relative to its diagonal
VACUUM = "G4_Galactic"
WORLD = "world"
GDML_SCHEMA_LOCATION = "http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd"

TWO_PI = 2.0 * math.pi
# a circle centred within this fraction of its radius from the axis is centred on it (an orb)
_ON_AXIS_REL = 1e-12
# a vertex of an arc may lie off the exact curve by this fraction of the major radius (the mesher's vertices are
# exact to its geometry tolerance); beyond it the entity is refused
_ON_CURVE_REL = 1e-6
_MASS_FRACTION_TOL = 1e-9
_PARTICLE_KEYS = ("nist", "elements", "by", "mean_excitation_eV", "state", "T_K", "pressure_Pa")
_OPTIONAL = ("mean_excitation_eV", "state", "T_K", "pressure_Pa")
_STATES = ("solid", "liquid", "gas")
# the element symbols in the order of Z (1 to 118)
_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb "
    "Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au "
    "Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts "
    "Og").split()
_Z = {sym: k + 1 for k, sym in enumerate(_SYMBOLS)}
# Geant4's NIST element builder has Z = 1 to 107 only: a heavier element is refused here, so that no exported file
# names an element Geant4 cannot build
_MAX_Z = 107


class SolidsWarning(UserWarning):
    """A region whose instantiated solids differ from the exact solid of revolution by more than ``VOLUME_WARN``."""


def default_chord_tol(size_m: float) -> float:
    """The default chord tolerance of a region of the given size (m): min(1 um, 1e-5 x size)."""
    return min(CHORD_TOL_MAX_M, CHORD_TOL_REL * float(size_m))


def _error(code: str, message: str, hint: str) -> Exception:
    exc = _core.AxionMultiPhysicsError(f"{message}. Hint: {hint}")
    exc.code = f"particle.{code}"
    exc.hint = hint
    return exc


def _invalid(message: str, hint: str = "the partition is the mesher's info file of a bounded axisymmetric mode; "
                                       "see docs/guides/solids.md for the keys the conversion reads") -> Exception:
    return _error("partition_invalid", message, hint)


_HINT_3D = ("the partition is the mesher's 3D export (axmp-mesher --export-regions on a 3D input, or Design.export_gdml of "
            "a 3D design): lower chord_tol_m of the export well below the thickness of the thinnest layer (a tenth of "
            "it, say) and export again; see docs/guides/solids.md, \"3D designs\"")


def _invalid3(message: str, hint: str = _HINT_3D) -> Exception:
    return _error("partition_invalid", message, hint)


def _real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ============================================================================================ curves
class _Curve:
    """The exact curve of one entity (a line, or an arc of a circle or an ellipse), or a synthetic segment of the
    axis (``tag`` None) that closes a filled loop or a void."""

    def __init__(self, tag, kind: str, p: tuple, q: tuple, sides=("exterior", "exterior")):
        self.tag, self.kind, self.p, self.q, self.sides = tag, kind, p, q, tuple(sides)
        self.closed = p == q
        self.circle = False
        self.axis = kind == "line" and p[0] == 0.0 and q[0] == 0.0
        self.tol = None            # chord tolerance (arcs)
        self.n = None              # number of chords (arcs)
        self.sagitta = 0.0
        self._points = None
        self.used = False          # discretized into a generic polycone's polygon

    @classmethod
    def line(cls, p, q):
        return cls(None, "line", p, q)

    def set_arc(self, center, radii, major_axis, ccw):
        self.c = center
        self.a, self.b = radii
        ur, uz = major_axis
        self.u = (ur, uz)
        self.v = (-uz, ur)
        self.sense = 1.0 if ccw else -1.0
        self.circle = self.a == self.b
        self.t0 = self._param(self.p)
        if self.closed:
            self.sweep = TWO_PI
        else:
            self.sweep = (self.sense * (self._param(self.q) - self.t0)) % TWO_PI
        A, B = self.a * self.u[0], self.b * self.v[0]
        C, D = self.a * self.u[1], self.b * self.v[1]
        self._abcd = (A, B, C, D)

    def _param(self, x) -> float:
        dr, dz = x[0] - self.c[0], x[1] - self.c[1]
        return math.atan2((dr * self.v[0] + dz * self.v[1]) / self.b, (dr * self.u[0] + dz * self.u[1]) / self.a)

    def off_curve(self, x) -> float:
        """Relative distance of x from the exact ellipse (0 on it)."""
        dr, dz = x[0] - self.c[0], x[1] - self.c[1]
        s = (dr * self.u[0] + dz * self.u[1]) / self.a
        t = (dr * self.v[0] + dz * self.v[1]) / self.b
        return abs(math.hypot(s, t) - 1.0)

    def at(self, t: float) -> tuple:
        A, B, C, D = self._abcd
        ct, st = math.cos(t), math.sin(t)
        return (max(self.c[0] + A * ct + B * st, 0.0), self.c[1] + C * ct + D * st)

    # ---------------------------------------------------------------- exact integrals (forward, start to end)
    def integrals(self) -> tuple[float, float]:
        """(integral of r dz, integral of r^2 dz) from start to end: the area and volume / pi terms of Green's theorem
        for the region on the curve's left."""
        if self.kind == "line":
            (r0, z0), (r1, z1) = self.p, self.q
            return 0.5 * (r0 + r1) * (z1 - z0), (r0 * r0 + r0 * r1 + r1 * r1) * (z1 - z0) / 3.0
        # r(t) = c_r + A cos t + B sin t and dz/dt = -C sin t + D cos t as trigonometric polynomials sum_k c_k e^{ikt}
        A, B, C, D = self._abcd
        r = np.array([0.5 * A + 0.5j * B, self.c[0], 0.5 * A - 0.5j * B])
        dz = np.array([0.5 * D - 0.5j * C, 0.0, 0.5 * D + 0.5j * C])
        t0, dt = self.t0, self.sense * self.sweep
        t1 = t0 + dt

        def integral(coef):
            K = (len(coef) - 1) // 2
            total = coef[K] * dt
            for k in range(1, K + 1):
                for kk in (k, -k):
                    total += coef[K + kk] * (cmath.exp(1j * kk * t1) - cmath.exp(1j * kk * t0)) / (1j * kk)
            return float(total.real)

        rdz = np.convolve(r, dz)
        return integral(rdz), integral(np.convolve(r, rdz))

    # ---------------------------------------------------------------- box
    def box(self) -> tuple[float, float, float, float]:
        """(rmin, zmin, rmax, zmax) of the exact curve."""
        pts = [self.p, self.q]
        if self.kind != "line":
            A, B, C, D = self._abcd
            for t in (math.atan2(B, A), math.atan2(D, C)):
                for tt in (t, t + math.pi):
                    if (self.sense * (tt - self.t0)) % TWO_PI <= self.sweep:
                        dr = A * math.cos(tt) + B * math.sin(tt)
                        dz = C * math.cos(tt) + D * math.sin(tt)
                        pts.append((self.c[0] + dr, self.c[1] + dz))
        rs, zs = [x[0] for x in pts], [x[1] for x in pts]
        return min(rs), min(zs), max(rs), max(zs)

    # ---------------------------------------------------------------- discretization
    def discretize(self, tol: float) -> None:
        """Cuts an arc into the fewest equal parameter steps whose sagitta is within tol (for a circle R (1 - cos(dt/2))
        exactly; for an ellipse its upper bound a (1 - cos(dt/2)))."""
        self.tol = tol
        a = self.a
        x = 1.0 - tol / a
        n = 1 if x <= -1.0 else math.ceil(self.sweep / (2.0 * math.acos(x)))
        n = max(n, math.ceil(self.sweep / (TWO_PI / 3.0)), 1)   # no step above 120 degrees: a polygon, never a chord
        while 2.0 * a * math.sin(self.sweep / (4.0 * n)) ** 2 > tol:
            n += 1
        self.n = n
        self.sagitta = 2.0 * a * math.sin(self.sweep / (4.0 * n)) ** 2

    def points(self) -> list:
        """The vertices from start to end: the end points exactly as the partition states them, between them the
        points at t0 + sense k sweep / n."""
        if self._points is None:
            if self.kind == "line":
                self._points = [self.p, self.q]
            else:
                step = self.sense * self.sweep / self.n
                self._points = [self.p] + [self.at(self.t0 + k * step) for k in range(1, self.n)] + [self.q]
        return self._points


class _Traversal:
    __slots__ = ("curve", "fwd")

    def __init__(self, curve: _Curve, fwd: bool):
        self.curve, self.fwd = curve, fwd

    @property
    def start(self):
        return self.curve.p if self.fwd else self.curve.q

    @property
    def end(self):
        return self.curve.q if self.fwd else self.curve.p

    def points(self):
        pts = self.curve.points()
        return pts if self.fwd else pts[::-1]

    def integrals(self):
        a, v = self.curve.integrals()
        return (a, v) if self.fwd else (-a, -v)

    def reversed(self):
        return _Traversal(self.curve, not self.fwd)

    def tangent_at_start(self):
        """Unit direction of travel at the start."""
        c = self.curve
        if c.kind == "line":
            d = (c.q[0] - c.p[0], c.q[1] - c.p[1])
            d = d if self.fwd else (-d[0], -d[1])
        else:
            t = c.t0 if self.fwd else c.t0 + c.sense * c.sweep
            A, B, C, D = c._abcd
            s = c.sense if self.fwd else -c.sense
            d = (s * (-A * math.sin(t) + B * math.cos(t)), s * (-C * math.sin(t) + D * math.cos(t)))
        n = math.hypot(*d)
        return (d[0] / n, d[1] / n)

    def tangent_at_end(self):
        r = self.reversed().tangent_at_start()
        return (-r[0], -r[1])


def _loop_integrals(trav) -> tuple[float, float]:
    """(signed area, volume / pi) of a closed loop of traversals, summed exactly rounded (independent of order)."""
    terms = [t.integrals() for t in trav]
    return math.fsum(a for a, _ in terms), math.fsum(v for _, v in terms)


def _polygon(trav) -> list:
    """The vertices of a closed loop of traversals, without repeating the first."""
    out = []
    for t in trav:
        for x in t.points()[:-1]:
            if not out or out[-1] != x:
                out.append(x)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


# ============================================================================================ the filled loop
def _filled(trav: list) -> list:
    """The loop with its notches at the axis filled: of the off-axis chains between consecutive axis vertices, those
    that climb (or, between one axis vertex and itself, that turn counter-clockwise) are kept, joined by straight
    segments of the axis; the others (notches) are replaced by the axis."""
    n = len(trav)
    on_axis = [k for k in range(n) if trav[k].start[0] == 0.0]
    if not on_axis:
        return list(trav)
    k0 = on_axis[0]
    trav = trav[k0:] + trav[:k0]
    chains, cur = [], []
    for t in trav:
        if t.curve.axis:
            if cur:
                chains.append(cur)
                cur = []
            continue
        cur.append(t)
        if t.end[0] == 0.0:
            chains.append(cur)
            cur = []
    if cur:
        chains.append(cur)
    kept = []
    for ch in chains:
        s, e = ch[0].start, ch[-1].end
        if e[1] > s[1]:
            kept.append(ch)
        elif e[1] == s[1] and _loop_integrals(ch)[0] > 0.0:
            kept.append(ch)
    if not kept:
        raise _invalid("a face's outer loop has no part off the axis that runs upwards (it is not counter-clockwise)")
    out = []
    for i, ch in enumerate(kept):
        out.extend(ch)
        nxt = kept[(i + 1) % len(kept)][0].start
        if ch[-1].end != nxt:
            out.append(_Traversal(_Curve.line(ch[-1].end, nxt), True))
    return out


def _start_lowest(poly: list) -> list:
    k = min(range(len(poly)), key=lambda i: (poly[i][1], poly[i][0]))
    return poly[k:] + poly[:k]


def _merge_axis_aligned(poly: list) -> list:
    """Drops every vertex between two edges along one line of constant r or constant z."""
    pts = list(poly)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                del pts[i]
                changed = True
                break
    return pts


def _classify(filled: list) -> tuple[str, dict, float]:
    """(kind, params, frame z) of the solid of revolution of a filled loop."""
    full = {"startphi_deg": 0.0, "deltaphi_deg": 360.0}
    curves = [t.curve for t in filled]
    if len(filled) == 1 and curves[0].kind != "line" and curves[0].closed and curves[0].circle \
            and curves[0].c[0] > curves[0].a:
        c = curves[0]
        return "torus", {"rmin_m": 0.0, "rmax_m": c.a, "rtor_m": c.c[0], **full}, c.c[1]
    arcs = [t for t in filled if t.curve.kind != "line"]
    if len(arcs) == 1 and all(c.axis for c in curves if c.kind == "line"):
        c = arcs[0].curve
        if c.circle and abs(c.c[0]) <= _ON_AXIS_REL * c.a and c.p[0] == 0.0 and c.q[0] == 0.0 and not c.closed \
                and abs(c.sweep - math.pi) <= 1e-9 and arcs[0].end[1] > arcs[0].start[1]:
            return "orb", {"r_m": c.a}, c.c[1]
    if not arcs:
        rect = _merge_axis_aligned(_polygon(filled))
        if len(rect) == 4:
            rs, zs = sorted({x[0] for x in rect}), sorted({x[1] for x in rect})
            if len(rs) == 2 and len(zs) == 2 and all(
                    (rect[i][0] == rect[i - 1][0]) != (rect[i][1] == rect[i - 1][1]) for i in range(4)):
                (r0, r1), (z0, z1) = rs, zs
                return "tube", {"rmin_m": r0, "rmax_m": r1, "z_m": z1 - z0, **full}, 0.5 * (z0 + z1)
    poly = _start_lowest(_polygon(filled))
    return "genericPolycone", {**full, "rz_m": [[r, z] for r, z in poly]}, 0.0


def _solid_volume(kind: str, p: dict) -> float:
    """The volume of a solid from its parameters (the closed forms Geant4 uses)."""
    turn = p.get("deltaphi_deg", 360.0) / 360.0
    if kind == "tube":
        return turn * math.pi * (p["rmax_m"] ** 2 - p["rmin_m"] ** 2) * p["z_m"]
    if kind == "torus":
        return turn * 2.0 * math.pi ** 2 * p["rtor_m"] * (p["rmax_m"] ** 2 - p["rmin_m"] ** 2)
    if kind == "orb":
        return 4.0 / 3.0 * math.pi * p["r_m"] ** 3
    rz = p["rz_m"]
    terms = []
    for i in range(len(rz)):
        (r1, z1), (r2, z2) = rz[i - 1], rz[i]
        terms.append((z2 - z1) * (r1 * r1 + r1 * r2 + r2 * r2))
    return turn * abs(math.fsum(terms)) * math.pi / 3.0


# ============================================================================================ geometry helpers
def _point_in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.count_nonzero(crosses & (x < xi)) % 2)


def _distance_to_segments(a: np.ndarray, b: np.ndarray, x: float, y: float) -> np.ndarray:
    d = b - a
    L2 = np.einsum("ij,ij->i", d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(L2 > 0, ((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / L2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(x - (a[:, 0] + t * d[:, 0]), y - (a[:, 1] + t * d[:, 1]))


def _segments(poly: np.ndarray) -> np.ndarray:
    """The edges of a closed polygon as rows (x1, y1, x2, y2)."""
    return np.hstack([poly, np.roll(poly, -1, axis=0)])


_CHUNK = 256


def _inside_polygon(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Crossing-number test of many points (rows of pts) against one polygon."""
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    out = np.zeros(len(pts), bool)
    for s in range(0, len(pts), _CHUNK):
        x, y = pts[s:s + _CHUNK, 0:1], pts[s:s + _CHUNK, 1:2]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        out[s:s + _CHUNK] = np.count_nonzero(crosses & (x < xi), axis=1) % 2 == 1
    return out


def _distance(segs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance of each point (rows of pts) to the nearest of the segments."""
    a, d = segs[:, :2], segs[:, 2:] - segs[:, :2]
    L2 = np.einsum("ij,ij->i", d, d)
    out = np.empty(len(pts))
    for s in range(0, len(pts), _CHUNK):
        x, y = pts[s:s + _CHUNK, 0:1], pts[s:s + _CHUNK, 1:2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(L2 > 0, ((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / L2, 0.0)
        t = np.clip(t, 0.0, 1.0)
        out[s:s + _CHUNK] = np.hypot(x - (a[:, 0] + t * d[:, 0]), y - (a[:, 1] + t * d[:, 1])).min(axis=1)
    return out


def _strictly_inside(poly: np.ndarray, pts: np.ndarray, eps: float) -> bool:
    """Whether a point lies inside the polygon by more than eps (points on its boundary do not count)."""
    lo, hi = poly.min(axis=0) - eps, poly.max(axis=0) + eps
    pts = pts[np.all((pts > lo) & (pts < hi), axis=1)]
    if not len(pts):
        return False
    ins = _inside_polygon(poly, pts)
    return bool(ins.any() and (_distance(_segments(poly), pts[ins]) > eps).any())


def _strictly_outside(poly: np.ndarray, pts: np.ndarray, eps: float) -> bool:
    """Whether a point lies outside the polygon by more than eps."""
    out = ~_inside_polygon(poly, pts)
    return bool(out.any() and (_distance(_segments(poly), pts[out]) > eps).any())


def _crossing(sa: np.ndarray, sb: np.ndarray, eps: float) -> bool:
    """Whether a segment of sa and one of sb cross properly: each one's ends lie on opposite sides of the other's line,
    farther than eps from it (segments that touch or share a piece of line do not count)."""
    bxl, bxh = np.minimum(sb[:, 0], sb[:, 2]), np.maximum(sb[:, 0], sb[:, 2])
    byl, byh = np.minimum(sb[:, 1], sb[:, 3]), np.maximum(sb[:, 1], sb[:, 3])
    for s in range(0, len(sa), _CHUNK):
        A = sa[s:s + _CHUNK]
        axl, axh = np.minimum(A[:, 0], A[:, 2])[:, None], np.maximum(A[:, 0], A[:, 2])[:, None]
        ayl, ayh = np.minimum(A[:, 1], A[:, 3])[:, None], np.maximum(A[:, 1], A[:, 3])[:, None]
        i, j = np.nonzero((axl <= bxh) & (axh >= bxl) & (ayl <= byh) & (ayh >= byl))
        if not len(i):
            continue
        p1, p2, q1, q2 = A[i, :2], A[i, 2:], sb[j, :2], sb[j, 2:]
        da, db = p2 - p1, q2 - q1
        la, lb = np.hypot(da[:, 0], da[:, 1]), np.hypot(db[:, 0], db[:, 1])
        ok = (la > 0) & (lb > 0)
        if not ok.any():
            continue
        p1, p2, q1, q2, da, db, la, lb = p1[ok], p2[ok], q1[ok], q2[ok], da[ok], db[ok], la[ok], lb[ok]

        def side(d, l, o, x):
            return (d[:, 0] * (x[:, 1] - o[:, 1]) - d[:, 1] * (x[:, 0] - o[:, 0])) / l

        d1, d2 = side(da, la, p1, q1), side(da, la, p1, q2)
        d3, d4 = side(db, lb, q1, p1), side(db, lb, q1, p2)
        hit = ((((d1 > eps) & (d2 < -eps)) | ((d1 < -eps) & (d2 > eps)))
               & (((d3 > eps) & (d4 < -eps)) | ((d3 < -eps) & (d4 > eps))))
        if hit.any():
            return True
    return False


def _shape(kind: str, params: dict, frame_z: float) -> tuple:
    """The cross-section of a solid in (r, z): ("poly", vertices), ("disk", centre, R) or ("half", centre, R)."""
    if kind == "tube":
        r0, r1, h = params["rmin_m"], params["rmax_m"], 0.5 * params["z_m"]
        return ("poly", np.array([[r0, frame_z - h], [r1, frame_z - h], [r1, frame_z + h], [r0, frame_z + h]]))
    if kind == "genericPolycone":
        return ("poly", np.asarray(params["rz_m"], float))
    if kind == "torus":
        return ("disk", (params["rtor_m"], frame_z), params["rmax_m"])
    return ("half", (0.0, frame_z), params["r_m"])


def _box(shape: tuple) -> tuple:
    if shape[0] == "poly":
        p = shape[1]
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())
    (cr, cz), R = shape[1], shape[2]
    return max(cr - R, 0.0), cz - R, cr + R, cz + R


def _probe(shape: tuple) -> np.ndarray:
    """A point inside a disk or half-disk."""
    (cr, cz), R = shape[1], shape[2]
    return np.array([[0.5 * R if shape[0] == "half" else cr, cz]])


def _lies_in(d: tuple, m: tuple, eps: float) -> bool:
    """Whether the cross-section d lies in m (within eps)."""
    if m[0] == "poly":
        mp = m[1]
        ms = _segments(mp)
        if d[0] == "poly":
            dp = d[1]
            return not (_strictly_outside(mp, dp, eps) or _strictly_inside(dp, mp, eps)
                        or _crossing(_segments(dp), ms, eps))
        (cr, cz), R = d[1], d[2]
        if d[0] == "half":     # its flat side lies on the axis, where the mother has edges too
            ms = ms[~((np.abs(ms[:, 0]) <= eps) & (np.abs(ms[:, 2]) <= eps))]
        if not _inside_polygon(mp, _probe(d))[0]:
            return False
        return not len(ms) or float(_distance(ms, np.array([[cr, cz]]))[0]) >= R - eps
    (mr, mz), RM = m[1], m[2]
    if d[0] == "poly":
        dp = d[1]
        return bool((np.hypot(dp[:, 0] - mr, dp[:, 1] - mz) <= RM + eps).all())
    (cr, cz), R = d[1], d[2]
    return math.hypot(cr - mr, cz - mz) + R <= RM + eps


def _overlap(a: tuple, b: tuple, eps: float) -> bool:
    """Whether the interiors of two cross-sections overlap by more than eps."""
    if a[0] != "poly" and b[0] == "poly":
        a, b = b, a
    if a[0] == "poly" and b[0] == "poly":
        pa, pb = a[1], b[1]
        return (_strictly_inside(pb, pa, eps) or _strictly_inside(pa, pb, eps)
                or _crossing(_segments(pa), _segments(pb), eps))
    if a[0] == "poly":
        (cr, cz), R = b[1], b[2]
        if float(_distance(_segments(a[1]), np.array([[cr, cz]]))[0]) < R - eps:
            return True
        return bool(_inside_polygon(a[1], _probe(b))[0])
    (r1, z1), R1 = a[1], a[2]
    (r2, z2), R2 = b[1], b[2]
    return math.hypot(r1 - r2, z1 - z2) < R1 + R2 - eps


def _inner_point(loops: list) -> tuple:
    """A point inside the region bounded by the polygons ``loops`` (outer counter-clockwise first, then the holes):
    on the inward normal of the outer polygon's longest edge, a small step from its middle (well inside the region,
    clear of every other edge)."""
    outer = np.asarray(loops[0], float)
    a, b = outer, np.roll(outer, -1, axis=0)
    L = np.hypot(*(b - a).T)
    i = int(np.argmax(L))
    m = 0.5 * (a[i] + b[i])
    d = (b[i] - a[i]) / L[i]
    nrm = np.array([-d[1], d[0]])               # the left of a counter-clockwise edge: inside
    seg_a, seg_b = [], []
    for k, poly in enumerate(loops):
        p = np.asarray(poly, float)
        q = np.roll(p, -1, axis=0)
        keep = np.ones(len(p), bool)
        if k == 0:
            keep[i] = False
        seg_a.append(p[keep])
        seg_b.append(q[keep])
    clear = float(_distance_to_segments(np.vstack(seg_a), np.vstack(seg_b), m[0], m[1]).min())
    step = min(1e-3 * L[i], 0.25 * clear)
    return (float(m[0] + step * nrm[0]), float(m[1] + step * nrm[1]))


# ============================================================================================ reading the partition
def _vec2(e: dict, key: str, tag) -> tuple:
    v = e.get(key)
    if not isinstance(v, (list, tuple)) or len(v) < 2 or not all(_real(x) for x in v[:2]):
        raise _invalid(f"entity {tag} has no valid '{key}' (a list of coordinates in m)")
    return (float(v[0]), float(v[1]))


def _read_entities(table: dict, used_tags: set) -> dict:
    ents = table.get("entities")
    if not isinstance(ents, list):
        raise _invalid("the partition has no list 'entities'")
    by_tag = {}
    for e in ents:
        if not isinstance(e, dict) or "tag" not in e:
            raise _invalid("an entity of the partition is not an object with a 'tag'")
        if e["tag"] in by_tag:
            raise _invalid(f"entity tag {e['tag']} is used twice")
        by_tag[e["tag"]] = e
    curves = {}
    for tag in sorted(used_tags, key=repr):
        if tag not in by_tag:
            raise _invalid(f"a face names entity {tag}, which the partition does not have")
        e = by_tag[tag]
        kind = e.get("type")
        if kind not in ("line", "circle", "ellipse"):
            raise _invalid(f"entity {tag} is of type {kind!r}; the conversion reads lines, circles and ellipses "
                           "(an imported mesh has no exact curves)")
        side = e.get("side")
        if not isinstance(side, (list, tuple)) or len(side) != 2 or not all(isinstance(s, str) for s in side):
            raise _invalid(f"entity {tag} has no valid 'side' (the regions on its left and its right)")
        if "start" not in e or "end" not in e:
            raise _invalid(f"entity {tag} has no 'start' or 'end' (the coordinates of its end vertices)")
        p, q = _vec2(e, "start", tag), _vec2(e, "end", tag)
        if p[0] < 0.0 or q[0] < 0.0:
            raise _invalid(f"entity {tag} has a vertex at negative r")
        c = _Curve(tag, "line" if kind == "line" else "arc", p, q, side)
        if kind == "line":
            if p == q:
                raise _invalid(f"line entity {tag} has the same start and end")
        else:
            center = _vec2(e, "center", tag)
            radii = e.get("radii")
            if not isinstance(radii, (list, tuple)) or len(radii) != 2 or not all(_real(x) for x in radii) \
                    or not radii[0] >= radii[1] > 0:
                raise _invalid(f"arc entity {tag} has no valid 'radii' [a, b] with a >= b > 0")
            a, b = float(radii[0]), float(radii[1])
            if "major_axis" in e:
                u = _vec2(e, "major_axis", tag)
            elif a == b:
                u = (1.0, 0.0)
            else:
                raise _invalid(f"ellipse entity {tag} has no 'major_axis'")
            if abs(math.hypot(*u) - 1.0) > 1e-9:
                raise _invalid(f"arc entity {tag}: 'major_axis' is not a unit vector")
            if not isinstance(e.get("ccw"), bool):
                raise _invalid(f"arc entity {tag} has no boolean 'ccw'")
            c.set_arc(center, (a, b), u, e["ccw"])
            for x in (p, q):
                if c.off_curve(x) > _ON_CURVE_REL:
                    raise _invalid(f"arc entity {tag}: its vertex {list(x)} is not on the curve its centre, radii and "
                                   "major axis describe")
            if not c.closed and c.sweep <= 0.0:
                raise _invalid(f"arc entity {tag} has no extent")
            if c.box()[0] < -_ON_AXIS_REL * a:
                raise _invalid(f"arc entity {tag} crosses the axis (r < 0)")
        curves[tag] = c
    return curves


def _read_faces(table: dict, regions: dict) -> list:
    faces = table.get("faces")
    if not isinstance(faces, list) or not faces:
        raise _invalid("the partition has no 'faces' (each final face with its region and its oriented loops)",
                       "the extended entity table of the bounded modes (or of --export-regions) lists the faces; an "
                       "info file without them predates the particle export")
    out = []
    for f in faces:
        if not isinstance(f, dict) or not isinstance(f.get("region"), str) or not isinstance(f.get("loops"), list):
            raise _invalid("a face is not an object with a 'region' and a list of 'loops'")
        if f["region"] not in regions:
            raise _invalid(f"a face names the region '{f['region']}', which the partition's 'regions' does not have")
        loops = f["loops"]
        if not loops or not all(isinstance(l, list) and l for l in loops):
            raise _invalid(f"a face of region '{f['region']}' has an empty loop")
        for l in loops:
            for s in l:
                if not isinstance(s, int) or isinstance(s, bool) or s == 0:
                    raise _invalid(f"a face of region '{f['region']}' has a loop entry {s!r} that is not a signed "
                                   "entity tag")
        out.append((f["region"], loops))
    return out


def _check_loop(region: str, trav: list) -> None:
    for t in trav:
        want = t.curve.sides[0 if t.fwd else 1]
        if want != region:
            raise _invalid(f"a face of region '{region}' runs entity {t.curve.tag} "
                           f"{'forward' if t.fwd else 'backward'}, but the region on its left that way is '{want}'")
    for k, t in enumerate(trav):
        nxt = trav[(k + 1) % len(trav)]
        if t.end != nxt.start:
            raise _invalid(f"a loop of a face of region '{region}' does not close: entity {t.curve.tag} ends at "
                           f"{list(t.end)} and entity {nxt.curve.tag} starts at {list(nxt.start)}")


# ============================================================================================ voids
def _void_loops(curves: dict) -> list:
    """The loops (as lists of traversals, counter-clockwise) of the exterior pockets of the domain: chains of the
    off-axis boundary entities, the domain on their left, that are clockwise (closed, or open from the axis to the
    axis and closed along it)."""
    edges = []
    for c in curves.values():
        if c.axis or (c.sides[0] == "exterior") == (c.sides[1] == "exterior"):
            continue
        edges.append(_Traversal(c, c.sides[0] != "exterior"))
    edges.sort(key=lambda t: (t.start[1], t.start[0], t.end[1], t.end[0], repr(t.curve.tag)))
    outgoing: dict = {}
    incoming: dict = {}
    for i, t in enumerate(edges):
        outgoing.setdefault(t.start, []).append(i)
        incoming.setdefault(t.end, []).append(i)
    used = [False] * len(edges)

    def next_edge(t):
        cand = [j for j in outgoing.get(t.end, []) if not used[j]]
        if len(cand) <= 1:
            return cand[0] if cand else None
        # several ways on at one vertex: the first counter-clockwise from the way back keeps to one exterior pocket
        back = t.tangent_at_end()
        a0 = math.atan2(-back[1], -back[0])
        return min(cand, key=lambda j: (math.atan2(edges[j].tangent_at_start()[1], edges[j].tangent_at_start()[0])
                                        - a0) % TWO_PI)

    starts = [i for i, t in enumerate(edges) if len(outgoing.get(t.start, [])) > len(incoming.get(t.start, []))]
    chains = []
    for i in starts + list(range(len(edges))):
        if used[i]:
            continue
        chain = [edges[i]]
        used[i] = True
        while chain[-1].end != chain[0].start:
            j = next_edge(chain[-1])
            if j is None:
                break
            used[j] = True
            chain.append(edges[j])
        chains.append(chain)
    out = []
    for ch in chains:
        s, e = ch[0].start, ch[-1].end
        if s != e and (s[0] != 0.0 or e[0] != 0.0):
            raise _invalid(f"the boundary of the domain ends at {list(e) if e[0] != 0.0 else list(s)}, off the axis "
                           "(an entity with the exterior on one side does not continue)")
        if _loop_integrals(ch)[0] < 0.0:
            loop = [t.reversed() for t in reversed(ch)]
            if s != e:
                loop.append(_Traversal(_Curve.line(s, e), True))
            out.append(loop)
    return out


# ============================================================================================ materials
def _materials(regions: list, materials: Mapping, region_materials: Mapping) -> dict:
    """The document records of the materials the regions use, checked in the order of the codes."""
    hint_c = ("give every transported region a material with a 'particle' block: {\"nist\": \"G4_...\"} or "
              "{\"elements\": {...}, \"by\": \"mass\" | \"atoms\"}; vacuum is written explicitly as G4_Galactic")
    used = {}
    for r in regions:
        if not isinstance(region_materials, Mapping) or r not in region_materials:
            raise _error("composition_missing", f"region '{r}' has no material", hint_c)
        name = region_materials[r]
        if not isinstance(name, str) or not isinstance(materials, Mapping) or name not in materials:
            raise _error("composition_missing", f"region '{r}' names the material '{name}', which is not defined",
                         hint_c)
        m = materials[name]
        if not isinstance(m, Mapping) or not isinstance(m.get("particle"), Mapping):
            raise _error("composition_missing", f"material '{name}' (region '{r}') has no 'particle' block", hint_c)
        used[name] = m
    hint_d = "set the material's 'density' in kg/m3 (a positive number); the transport needs it"
    for name, m in used.items():
        rho = m.get("density")
        if not _real(rho) or rho <= 0:
            raise _error("density_missing", f"material '{name}' has no valid density ({rho!r})", hint_d)
    out = {}
    for name in sorted(used):
        out[name] = _composition(name, used[name])
    return out


def _composition(name: str, m: Mapping) -> dict:
    hint = ("a 'particle' block is {\"nist\": \"G4_...\"} or {\"elements\": {symbol: value}, \"by\": \"mass\" | "
            "\"atoms\"}, with optional mean_excitation_eV, state (solid, liquid or gas), T_K and pressure_Pa")

    def bad(what):
        return _error("composition_invalid", f"material '{name}': {what}", hint)

    if name.startswith("G4_"):
        raise _error("composition_invalid", f"material '{name}': a document material may not have a name that starts "
                     "with G4_ (that prefix names NIST materials)", "rename it, and give its composition as "
                     "{\"nist\": \"G4_...\"} if it is a NIST material")
    p = m["particle"]
    for k in p:
        if k not in _PARTICLE_KEYS:
            raise bad(f"unknown key '{k}' in the particle block (known: {', '.join(_PARTICLE_KEYS)})")
    if ("nist" in p) == ("elements" in p):
        raise bad("the particle block needs exactly one of 'nist' and 'elements'")
    if "nist" in p:
        nist = p["nist"]
        if not isinstance(nist, str) or not nist.startswith("G4_") or len(nist) <= 3:
            raise bad(f"'nist' is {nist!r}, not the name of a NIST material (G4_...)")
        if "by" in p:
            raise bad("'by' applies to 'elements', not to a NIST material")
        comp = {"nist": nist}
    else:
        els, by = p["elements"], p.get("by")
        if by not in ("mass", "atoms"):
            raise bad(f"'by' is {by!r}; it is 'mass' (mass fractions) or 'atoms' (atom counts)")
        if not isinstance(els, Mapping) or not els:
            raise bad("'elements' is not a non-empty object {symbol: value}")
        for sym in els:
            if sym not in _Z:
                raise bad(f"'{sym}' is not the symbol of an element")
            if _Z[sym] > _MAX_Z:
                raise bad(f"'{sym}' (Z = {_Z[sym]}) is heavier than the elements Geant4's NIST tables build "
                          f"(Z = 1 to {_MAX_Z})")
        if by == "mass":
            for sym, w in els.items():
                if not _real(w) or w <= 0:
                    raise bad(f"the mass fraction of {sym} is {w!r}, not a positive number")
            total = math.fsum(float(w) for w in els.values())
            if abs(total - 1.0) > _MASS_FRACTION_TOL:
                raise bad(f"the mass fractions sum to {total!r}, not to 1 (within {_MASS_FRACTION_TOL:g}; they are "
                          "not renormalized)")
            elements = {sym: float(els[sym]) for sym in sorted(els)}
        else:
            for sym, n in els.items():
                if not _real(n) or n <= 0 or float(n) != int(n):
                    raise bad(f"the atom count of {sym} is {n!r}, not a positive integer")
            elements = {sym: int(els[sym]) for sym in sorted(els)}
        comp = {"elements": elements, "by": by}
    rec = {"name": name, "density_kg_m3": float(m["density"]), "composition": comp}
    for k in _OPTIONAL:
        if k not in p:
            continue
        v = p[k]
        if k == "state":
            if v not in _STATES:
                raise bad(f"'state' is {v!r}, not one of {', '.join(_STATES)}")
            rec[k] = v
        else:
            if not _real(v) or v <= 0:
                raise bad(f"'{k}' is {v!r}, not a positive number")
            rec[k] = float(v)
    return rec


# ============================================================================================ the conversion
class _Item:
    """A face or a void: its loops, its filled outer loop and its solid."""

    def __init__(self, region, loops):
        self.region, self.loops = region, loops          # loops: lists of traversals, the outer one first
        self.filled = _filled(loops[0])
        self.kind, self.params, self.frame_z = _classify(self.filled)
        self.discretized = self.kind == "genericPolycone" and any(t.curve.kind != "line" for t in self.filled)
        if self.discretized:
            for t in self.filled:
                if t.curve.kind != "line":
                    t.curve.used = True
        self.sagitta = max((t.curve.sagitta for t in self.filled if t.curve.kind != "line"), default=0.0) \
            if self.discretized else 0.0
        self.filled_poly = np.asarray(_polygon(self.filled), float)
        self.filled_area = _loop_integrals(self.filled)[0]
        box = [t.curve.box() for t in loops[0]]
        self.box = (min(b[1] for b in box), min(b[0] for b in box), max(b[3] for b in box), max(b[2] for b in box))
        self.area = math.fsum(_loop_integrals(l)[0] for l in loops)
        outer = _polygon(loops[0])
        self.point = _inner_point([outer] + [_polygon(l) for l in loops[1:]])
        self.vertices = tuple(sorted(outer))      # breaks a tie of box and area by the geometry itself
        self.name = None
        self.mother = None

    def key(self):
        return self.box + (self.area, self.vertices)


# ============================================================================================ GDML names
_GDML_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9._-]*\Z")
_DERIVED = ("_solid", "_pv", "_pos", "_rot")     # the names the GDML file derives from a volume's


def _gdml_valid(name: str) -> bool:
    """A name GDML can hold as it is: an ASCII xs:ID without "0x" (where Geant4's reader cuts names), not G4_..."""
    return bool(_GDML_NAME.match(name)) and "0x" not in name and not name.startswith("G4_")


def _gdml_base(name: str) -> str:
    out = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "._-") else "_" for ch in name)
    if not out or not (out[0].isalpha() or out[0] == "_"):
        out = "_" + out
    while "0x" in out:
        out = out.replace("0x", "0_x")
    return "_" + out if out.startswith("G4_") else out


def _gdml_names(solid_names, material_names) -> tuple[dict, dict]:
    """Maps the solids' and the materials' names to GDML names: valid, free of "0x" and unique across the file
    (with the names derived from a volume's). A valid name that collides with nothing is kept; the others are made
    valid and numbered. The result depends only on the two sets of names (solids before materials, each in name
    order), never on the order of the input."""
    taken: set = {"Default"}    # the name of the setup element, an xs:ID like the others

    def free(n, solid):
        return n not in taken and not (solid and any(n + d in taken for d in _DERIVED))

    def take(n, solid):
        taken.add(n)
        if solid:
            taken.update(n + d for d in _DERIVED)

    entries = [(n, True) for n in sorted(set(solid_names))] + [(n, False) for n in sorted(set(material_names))]
    out = ({}, {})
    for n, solid in entries:
        if _gdml_valid(n) and free(n, solid):
            take(n, solid)
            out[0 if solid else 1][n] = n
    for n, solid in entries:
        if n in out[0 if solid else 1]:
            continue
        base = cand = _gdml_base(n)
        k = 0
        while not free(cand, solid):
            k += 1
            cand = f"{base}_{k}"
        take(cand, solid)
        out[0 if solid else 1][n] = cand
    return out


def from_partition(partition: dict, materials: dict, region_materials: dict, *,
                   chord_tol_m: float | None = None) -> dict:
    """The solids document (``axionmultiphysics.solids/1``) of an axisymmetric partition.

    ``partition`` is the mesher's info file of a bounded mode (or of ``--export-regions``) with the exact curve
    parameters and the face loops; ``materials`` the document's materials ({name: {"density": kg/m3, "particle":
    {...}}}); ``region_materials`` {region: material name}. ``chord_tol_m`` overrides the chord tolerance of every
    region (default: min(1 um, 1e-5 x the diagonal of the region's box)). Raises ``AxionMultiPhysicsError`` with a
    ``particle.*`` code, also for a layer thinner than about the chord tolerance, whose solids would overlap or lose
    their place in the tree (``particle.partition_invalid``: lower ``chord_tol_m``); warns (``SolidsWarning``, code
    ``particle.volume_deficit``) for a region whose instantiated volume or mass differs from the exact one by more than
    ``VOLUME_WARN``. Every solid and material carries a ``gdml_name``, the name GDML and Geant4 use.
    """
    warn_beta("axionmultiphysics.solids")
    if chord_tol_m is not None and (not _real(chord_tol_m) or chord_tol_m <= 0):
        raise ValueError(f"chord_tol_m must be a positive number of metres, not {chord_tol_m!r}")
    if not isinstance(partition, Mapping):
        raise _invalid("the partition is not a JSON object")
    geometry = partition.get("geometry")
    if geometry == "planar":
        raise _error("planar_not_offered", "a planar partition has no solids: particle transport is offered for "
                     "axisymmetric designs (revolved about the axis) and 3D ones",
                     "describe the design as axisymmetric, or transport it in a 3D design")
    if geometry == "3d":
        if chord_tol_m is not None:
            raise ValueError("chord_tol_m is not taken for a 3D partition: its triangulation is the mesher's "
                             "(pass chord_tol_m to the export, which gives it to the mesher)")
        return _from_partition_3d(partition, materials, region_materials)
    if geometry != "axisymmetric":
        raise _invalid(f"the partition's geometry is {geometry!r}; this version converts axisymmetric and 3D "
                       "partitions")
    regions = partition.get("regions")
    if not isinstance(regions, Mapping):
        raise _invalid("the partition has no object 'regions'")
    faces = _read_faces(partition, regions)
    curves = _read_entities(partition, {abs(s) for _, loops in faces for l in loops for s in l})

    # loops as traversals, checked
    face_loops = []
    for region, loops in faces:
        tl = []
        for l in loops:
            trav = [_Traversal(curves[abs(s)], s > 0) for s in l]
            _check_loop(region, trav)
            tl.append(trav)
        a_out = _loop_integrals(tl[0])[0]
        if a_out <= 0.0 or any(_loop_integrals(t)[0] >= 0.0 for t in tl[1:]):
            raise _invalid(f"a face of region '{region}' has its outer loop clockwise or an inner loop "
                           "counter-clockwise (a loop keeps its face on its left, the outer loop first)")
        face_loops.append((region, tl))

    # regions with faces, in document order
    def attr_key(name):
        a = regions[name].get("attr") if isinstance(regions[name], Mapping) else None
        return (0, a, name) if _real(a) else (1, 0, name)

    used_regions = sorted({r for r, _ in face_loops}, key=attr_key)
    mats = _materials(used_regions, materials, region_materials)

    # chord tolerances: per region, then per entity the smallest of its sides
    region_tol = {}
    for r in used_regions:
        boxes = [t.curve.box() for rr, tl in face_loops if rr == r for l in tl for t in l]
        diag = math.hypot(max(b[2] for b in boxes) - min(b[0] for b in boxes),
                          max(b[3] for b in boxes) - min(b[1] for b in boxes))
        region_tol[r] = float(chord_tol_m) if chord_tol_m is not None else default_chord_tol(diag)
    for c in curves.values():
        if c.kind != "line":
            tols = [region_tol[s] for s in c.sides if s in region_tol]
            c.discretize(min(tols))

    # faces and voids, named in a geometric order
    items = [_Item(r, tl) for r, tl in face_loops]
    voids = [_Item(None, [loop]) for loop in _void_loops(curves)]
    for r in used_regions:
        mine = sorted((it for it in items if it.region == r), key=_Item.key)
        for k, it in enumerate(mine):
            it.name = f"{r}_{k}"
    # the vacuum solids are void_<k>, skipping a name that a region's solids already have (a region named "void");
    # a region's solids <region>_<k> are unique, and never "world"
    taken = {it.name for it in items}
    k = 0
    for it in sorted(voids, key=_Item.key):
        while f"void_{k}" in taken:
            k += 1
        it.name = f"void_{k}"
        k += 1
    everything = items + voids

    # the mother tree: the smallest filled loop that contains a point of the item
    for it in everything:
        best = None
        for other in everything:
            if other is it or other.filled_area <= 0.0:
                continue
            if _point_in_polygon(other.filled_poly, *it.point):
                if best is None or (other.filled_area, other.name) < (best.filled_area, best.name):
                    best = other
        it.mother = best

    # the world: a vacuum tube at the origin around everything, with a margin
    allb = [c.box() for c in curves.values()]
    rlo, zlo = min(b[0] for b in allb), min(b[1] for b in allb)
    rhi, zhi = max(b[2] for b in allb), max(b[3] for b in allb)
    margin = WORLD_MARGIN * math.hypot(rhi - rlo, zhi - zlo)
    half = max(abs(zlo), abs(zhi)) + margin
    world = {"name": WORLD, "kind": "tube",
             "params": {"rmin_m": 0.0, "rmax_m": rhi + margin, "z_m": 2.0 * half, "startphi_deg": 0.0,
                        "deltaphi_deg": 360.0},
             "mother": None, "position_m": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0], "region": None,
             "material": VACUUM, "discretized": False, "sagitta_m": 0.0}

    # pre-order from the world, children in name order
    kids: dict = {WORLD: [], **{it.name: [] for it in everything}}
    for it in everything:
        kids[it.mother.name if it.mother else WORLD].append(it)
    solids = [world]

    def visit(name):
        for it in sorted(kids[name], key=lambda x: x.name):
            mz = it.mother.frame_z if it.mother else 0.0
            solids.append({"name": it.name, "kind": it.kind, "params": it.params,
                           "mother": it.mother.name if it.mother else WORLD,
                           "position_m": [0.0, 0.0, it.frame_z - mz], "rotation_deg": [0.0, 0.0, 0.0],
                           "region": it.region,
                           "material": region_materials[it.region] if it.region is not None else VACUUM,
                           "discretized": it.discretized, "sagitta_m": it.sagitta})
            visit(it.name)

    visit(WORLD)

    # A layer thinner than about the chord tolerance makes the chords of its two curves cross: a face can then hold
    # the inner point of its neighbour (the mother relation is no longer a tree) or a daughter can leave its mother.
    # Such a list is refused, never written without the faces or with overlapping solids.
    hint = (f"a layer thinner than about the chord tolerance (here {max(region_tol.values()):.3g} m) makes the chords "
            "of its two curves cross; lower chord_tol_m to well below the layer's thickness (a tenth of it, say). Where "
            "a discretized curve is concave towards the other solid and touches it (a shape touching the inside of "
            "an ellipse, say), lowering chord_tol_m does not help: its chords still pass through the other solid by "
            "about 0.8 chord tolerances; separate them there by a few chord tolerances")
    placed = {s["name"] for s in solids}
    lost = sorted(it.name for it in everything if it.name not in placed)
    if lost:
        regions_lost = sorted({it.region if it.region is not None else "(vacuum)" for it in everything
                               if it.name in lost})
        raise _invalid(f"the solids {lost} (regions {regions_lost}) have no place in the tree of solids: each "
                       "contains the other", hint)
    allbox = (rlo, zlo, rhi, zhi)
    # relative to the size, but never above 5e-13 m: half of Geant4's overlap tolerance (kCarTolerance, 1e-12 m), so
    # that a partition admitted here is not refused by Geant4's own overlap check
    eps = min(1e-12 * math.hypot(allbox[2] - allbox[0], allbox[3] - allbox[1]), 5e-13)
    shapes = {WORLD: _shape("tube", world["params"], 0.0)}
    for it in everything:
        shapes[it.name] = _shape(it.kind, it.params, it.frame_z)
    bad = []
    for s in solids[1:]:
        if not _lies_in(shapes[s["name"]], shapes[s["mother"]], eps):
            bad.append(f"{s['name']} leaves its mother {s['mother']}")
    for parent, ks in kids.items():
        ks = sorted(ks, key=lambda x: x.name)
        boxes = [_box(shapes[x.name]) for x in ks]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                bi, bj = boxes[i], boxes[j]
                if bi[0] > bj[2] or bj[0] > bi[2] or bi[1] > bj[3] or bj[1] > bi[3]:
                    continue
                if _overlap(shapes[ks[i].name], shapes[ks[j].name], eps):
                    bad.append(f"{ks[i].name} overlaps {ks[j].name}")
    # A polycone whose outline passes twice through one point (a face pinched there by a boundary that touches itself
    # or another one) is refused: whether Geant4 builds it depends on rounding in its crossing test.
    for it in everything:
        if it.kind == "genericPolycone":
            pts = [tuple(x) for x in it.params["rz_m"]]
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts.pop()
            again = sorted({x for x in pts if pts.count(x) > 1})
            if again:
                where = f"region '{it.region}'" if it.region is not None else "a vacuum hole"
                raise _invalid(f"the outline of {it.name} ({where}) passes twice through the point (r, z) = "
                               f"({again[0][0]:.9g}, {again[0][1]:.9g}) m: its boundary touches itself or another "
                               "boundary there, and Geant4 may refuse such a solid by rounding",
                               "separate the boundaries that touch (a face must not be pinched at a point); where "
                               "a discretized curve is involved, by a few chord tolerances")
    for it in everything:
        if it.kind == "genericPolycone":
            seg = _segments(shapes[it.name][1])
            if _crossing(seg, seg, eps):
                bad.append(f"{it.name} crosses itself")
    if bad:
        raise _invalid("the solids would overlap: " + "; ".join(bad[:6]) + ("; ..." if len(bad) > 6 else ""), hint)

    # the GDML (and Geant4) names
    gs, gm = _gdml_names([s["name"] for s in solids], list(mats))
    solids = [{"name": s["name"], "gdml_name": gs[s["name"]], **{k: v for k, v in s.items() if k != "name"}}
              for s in solids]
    mats = {n: {"name": n, "gdml_name": gm[n], **{k: v for k, v in m.items() if k != "name"}}
            for n, m in mats.items()}

    # the report
    vol = {s["name"]: _solid_volume(s["kind"], s["params"]) for s in solids}
    report = {}
    out_warnings = []
    for r in used_regions:
        mine = [s["name"] for s in solids if s["region"] == r]
        rho = mats[region_materials[r]]["density_kg_m3"]
        exact = math.pi * math.fsum(t.integrals()[1] for rr, tl in face_loops if rr == r for l in tl for t in l)
        inst = math.fsum([vol[n] for n in mine] + [-vol[k.name] for n in mine for k in kids[n]])
        m_exact, m_inst = exact * rho, inst * rho
        ents = {t.curve.tag: t.curve for rr, tl in face_loops if rr == r for l in tl for t in l}
        rep = {"material": region_materials[r], "density_kg_m3": rho, "solids": mine, "volume_m3": exact,
               "volume_solids_m3": inst, "volume_rel_diff": inst / exact - 1.0, "mass_kg": m_exact,
               "mass_solids_kg": m_inst, "mass_rel_diff": m_inst / m_exact - 1.0,
               "sagitta_max_m": max((c.sagitta for c in ents.values() if c.used), default=0.0),
               "chord_tol_m": region_tol[r]}
        report[r] = rep
        if abs(rep["volume_rel_diff"]) > VOLUME_WARN or abs(rep["mass_rel_diff"]) > VOLUME_WARN:
            out_warnings.append({"code": "particle.volume_deficit", "region": r,
                                 "volume_rel_diff": rep["volume_rel_diff"], "mass_rel_diff": rep["mass_rel_diff"],
                                 "sagitta_max_m": rep["sagitta_max_m"], "chord_tol_m": rep["chord_tol_m"]})
            warnings.warn(f"particle.volume_deficit: region '{r}': its solids differ from the exact solid of "
                          f"revolution by {rep['volume_rel_diff']:.3g} in volume and {rep['mass_rel_diff']:.3g} in "
                          f"mass (more than {VOLUME_WARN:g}; largest sagitta {rep['sagitta_max_m']:.3g} m); lower "
                          "chord_tol_m", SolidsWarning, stacklevel=2)
    from . import __version__
    return {"schema": SCHEMA, "generator": f"axionmultiphysics {__version__}", "geometry": "axisymmetric",
            "units": {"length": "m", "angle": "deg", "density": "kg/m3", "volume": "m3", "mass": "kg"},
            "chord_tol_m": float(chord_tol_m) if chord_tol_m is not None else None, "world": WORLD,
            "materials": [mats[n] for n in sorted(mats)], "solids": solids, "regions": report,
            "warnings": out_warnings}



# ---- the checks of the triangulated solids: nesting, overlap, self-crossing
_RAYS = np.array([[0.5773502691896258, 0.5773502691896257, 0.5773502691896259],
                  [-0.3015113445777636, 0.9045340337332909, 0.3015113445777636],
                  [0.8017837257372732, -0.2672612419124244, 0.5345224838248488]]) + \
    np.array([[1e-3, -2e-3, 3e-4], [-4e-4, 1e-3, 2e-3], [2e-3, 3e-4, -1e-3]])
_RAYS = _RAYS / np.linalg.norm(_RAYS, axis=1)[:, None]


def _boxes(nodes: np.ndarray, tri: np.ndarray):
    v = nodes[tri]
    return v.min(axis=1), v.max(axis=1)


def _box_pairs(lo_a, hi_a, lo_b, hi_b, limit: int = 3_000_000):
    """Index pairs (i, j) whose boxes overlap, found with a uniform 3D grid: the cell is about twice a typical box, a
    box is listed in the (few) cells it covers, and a pair is reported in the one cell that holds the larger of its two
    lower corners, so that none is found twice. The few boxes that cover many cells are tested against everything."""
    na, nb = len(lo_a), len(lo_b)
    empty = np.zeros(0, np.int64)
    if na == 0 or nb == 0:
        return empty, empty
    origin = np.minimum(lo_a.min(axis=0), lo_b.min(axis=0))
    top = np.maximum(hi_a.max(axis=0), hi_b.max(axis=0))
    size = np.concatenate([(hi_a - lo_a).max(axis=1), (hi_b - lo_b).max(axis=1)])
    h = 2.0 * float(np.quantile(size, 0.95))
    h = max(h, float(np.max(top - origin)) / 2e6, 1e-300)
    dims = np.floor((top - origin) / h).astype(np.int64) + 1

    def cells(lo, hi):
        return (np.floor((lo - origin) / h).astype(np.int64).clip(0, dims - 1),
                np.floor((hi - origin) / h).astype(np.int64).clip(0, dims - 1))

    ca0, ca1 = cells(lo_a, hi_a)
    cb0, cb1 = cells(lo_b, hi_b)
    span_a, span_b = ca1 - ca0 + 1, cb1 - cb0 + 1
    big_a = np.nonzero(span_a.max(axis=1) > 3)[0]
    big_b = np.nonzero(span_b.max(axis=1) > 3)[0]
    small_a = np.nonzero(span_a.max(axis=1) <= 3)[0]
    small_b = np.nonzero(span_b.max(axis=1) <= 3)[0]

    def entries(c0, c1, idx):
        """(cell key, box index, cell coordinates) for every cell each box of idx covers."""
        keys, ids, coords = [], [], []
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    cc = c0[idx] + np.array([dx, dy, dz])
                    ok = np.all(cc <= c1[idx], axis=1)
                    cc = cc[ok]
                    keys.append((cc[:, 0] * dims[1] + cc[:, 1]) * dims[2] + cc[:, 2])
                    ids.append(idx[ok])
                    coords.append(cc)
        return np.concatenate(keys), np.concatenate(ids), np.concatenate(coords)

    out_i, out_j = [], []

    def keep(i, j):
        ok = np.all((lo_a[i] <= hi_b[j]) & (hi_a[i] >= lo_b[j]), axis=1)
        i, j = i[ok], j[ok]
        out_i.append(i)
        out_j.append(j)

    ka, ia_e, xa = entries(ca0, ca1, small_a)
    kb, ib_e, _ = entries(cb0, cb1, small_b)
    order = np.argsort(kb, kind="stable")
    kb, ib_e = kb[order], ib_e[order]
    start = np.searchsorted(kb, ka, side="left")
    end = np.searchsorted(kb, ka, side="right")
    cnt = end - start
    csum = np.concatenate([[0], np.cumsum(cnt)])
    e0 = 0
    ne = len(ka)
    while e0 < ne:
        e1 = int(np.searchsorted(csum, csum[e0] + limit, side="right")) - 1
        e1 = min(max(e1, e0 + 1), ne)
        c = cnt[e0:e1]
        total = int(c.sum())
        if total:
            ee = np.repeat(np.arange(e0, e1), c)
            offs = np.arange(total) - np.repeat(csum[e0:e1] - csum[e0], c)
            i = ia_e[ee]
            j = ib_e[np.repeat(start[e0:e1], c) + offs]
            # the pair belongs to the cell of the larger lower corner
            corner = np.floor((np.maximum(lo_a[i], lo_b[j]) - origin) / h).astype(np.int64).clip(0, dims - 1)
            mine = np.all(corner == xa[ee], axis=1)
            keep(i[mine], j[mine])
        e0 = e1
    for i in big_a:  # long boxes of a against all of b
        j = np.nonzero(np.all((lo_b <= hi_a[i]) & (hi_b >= lo_a[i]), axis=1))[0]
        out_i.append(np.full(len(j), i))
        out_j.append(j)
    for j in big_b:  # long boxes of b against the others of a
        i = small_a[np.all((lo_a[small_a] <= hi_b[j]) & (hi_a[small_a] >= lo_b[j]), axis=1)]
        out_i.append(i)
        out_j.append(np.full(len(i), j))
    if not out_i:
        return empty, empty
    return np.concatenate(out_i), np.concatenate(out_j)


def _pierces(p, q, a, b, c, tol):
    """Whether the segment p-q passes through the inside of the triangle a, b, c: its ends more than tol on either side
    of the plane, the crossing point more than tol inside each edge (touching is not piercing)."""
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    u = n / np.where(nn > 0, nn, 1.0)[:, None]
    dp = np.einsum("ij,ij->i", p - a, u)
    dq = np.einsum("ij,ij->i", q - a, u)
    ok = (nn > 0) & (((dp > tol) & (dq < -tol)) | ((dp < -tol) & (dq > tol)))
    t = np.where(ok, dp / np.where(ok, dp - dq, 1.0), 0.0)
    x = p + t[:, None] * (q - p)
    for s, e in ((a, b), (b, c), (c, a)):
        w = e - s
        side = np.einsum("ij,ij->i", np.cross(w, x - s), u) / np.maximum(np.linalg.norm(w, axis=1), 1e-300)
        ok &= side > tol
    return ok


def _cross_pairs(nodes, tri_a, tri_b, ia, ib, tol, chunk: int = 400_000) -> bool:
    """Whether any triangle ia of tri_a pierces or is pierced by its partner ib of tri_b (pairs sharing a node are
    contact, as are the triangles that two touching solids have in common)."""
    for k in range(0, len(ia), chunk):
        ta, tb = tri_a[ia[k:k + chunk]], tri_b[ib[k:k + chunk]]
        share = (ta[:, :, None] == tb[:, None, :]).any(axis=(1, 2))
        ta, tb = ta[~share], tb[~share]
        if len(ta) == 0:
            continue
        A, B = nodes[ta], nodes[tb]
        for e in range(3):
            if _pierces(A[:, e], A[:, (e + 1) % 3], B[:, 0], B[:, 1], B[:, 2], tol).any():
                return True
            if _pierces(B[:, e], B[:, (e + 1) % 3], A[:, 0], A[:, 1], A[:, 2], tol).any():
                return True
    return False


def _surfaces_cross(nodes, tri_a, tri_b, tol) -> bool:
    lo_a, hi_a = _boxes(nodes, tri_a)
    lo_b, hi_b = _boxes(nodes, tri_b)
    ia, ib = _box_pairs(lo_a, hi_a, lo_b, hi_b)
    return _cross_pairs(nodes, tri_a, tri_b, ia, ib, tol)


def _star_shaped(nodes, tri) -> bool:
    """Whether the closed surface is star-shaped about the centre of its box, which proves that it does not cross itself:
    every triangle is positively oriented about the centre (so that the radial map onto a sphere is locally one to one)
    and their solid angles add up to 4 pi (so that it covers the sphere once, which a map of that kind can do only
    without folds). Convex solids, balls, boxes and cylinders among them, pass; a solid with a tunnel does not."""
    v = nodes[tri]
    c = (v.reshape(-1, 3).min(axis=0) + v.reshape(-1, 3).max(axis=0)) / 2.0
    a, b, d = v[:, 0] - c, v[:, 1] - c, v[:, 2] - c
    det = np.einsum("ij,ij->i", a, np.cross(b, d))
    la, lb, ld = (np.linalg.norm(x, axis=1) for x in (a, b, d))
    if not (det > 1e-12 * la * lb * ld).all():
        return False
    den = la * lb * ld + np.einsum("ij,ij->i", a, b) * ld + np.einsum("ij,ij->i", a, d) * lb + \
        np.einsum("ij,ij->i", b, d) * la
    omega = 2.0 * np.arctan2(det, den)
    return abs(float(omega.sum()) - 4.0 * math.pi) < 1e-6


def _self_crosses(nodes, tri, tol) -> bool:
    if _star_shaped(nodes, tri):
        return False
    lo, hi = _boxes(nodes, tri)
    ia, ib = _box_pairs(lo, hi, lo, hi)
    keep = ia < ib
    return _cross_pairs(nodes, tri, tri, ia[keep], ib[keep], tol)


def _inside(nodes, tri, p) -> bool:
    """Whether p lies inside the closed outward surface: the majority of three rays (the parity of their hits)."""
    a, b, c = nodes[tri[:, 0]], nodes[tri[:, 1]], nodes[tri[:, 2]]
    e1, e2 = b - a, c - a
    votes = 0
    for ray in _RAYS:
        h = np.cross(ray, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-300
        f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = p - a
        u = f * np.einsum("ij,ij->i", sv, h)
        q = np.cross(sv, e1)
        v = f * (q @ ray)
        t = f * np.einsum("ij,ij->i", e2, q)
        votes += int(np.count_nonzero(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)) % 2)
    return votes >= 2


def _free_vertices(nodes, tri, other, count: int = 3):
    """Up to `count` vertices of tri that no triangle of `other` has (spread along the list)."""
    mine = np.unique(tri)
    free = mine[~np.isin(mine, np.unique(other))]
    if len(free) == 0:
        return nodes[free]
    return nodes[free[np.linspace(0, len(free) - 1, min(count, len(free))).astype(int)]]

# ============================================================================================ 3D partitions
def _triples(x, what: str, n_nodes: int) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=np.int64).reshape(-1, 3)
    except (TypeError, ValueError, OverflowError):
        raise _invalid3(f"{what} is not a list of triples of node indices") from None
    if a.size and (a.min() < 0 or a.max() >= n_nodes):
        raise _invalid3(f"{what} names a node the node list does not have")
    return a


def _shell_triangles(shell, faces: dict, vol: int) -> np.ndarray:
    """The triangles of a shell: each face's, turned when its tag is negative, the faces in tag order."""
    parts = []
    for s in sorted(shell, key=abs):
        t = faces[abs(s)]
        parts.append(t if s > 0 else t[:, ::-1])
    return np.concatenate(parts) if parts else np.zeros((0, 3), np.int64)


def _closed(tri: np.ndarray, n_nodes: int) -> bool:
    """Every directed edge once, with its reverse once: a closed, consistently oriented surface."""
    if len(tri) == 0:
        return False
    a = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]]).astype(np.int64)
    b = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]]).astype(np.int64)
    fwd = a * n_nodes + b
    rev = b * n_nodes + a
    u, c = np.unique(fwd, return_counts=True)
    return bool((c == 1).all() and np.array_equal(u, np.sort(rev)))


def _signed_volume(nodes: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = nodes[tri[:, 0]], nodes[tri[:, 1]], nodes[tri[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    return math.fsum(det.tolist()) / 6.0


def _from_partition_3d(partition: dict, materials: dict, region_materials: dict) -> dict:
    """The solids of a 3D partition: one tessellated solid per volume (see the module's docstring)."""
    regions = partition.get("regions")
    if not isinstance(regions, Mapping):
        raise _invalid3("the partition has no object 'regions'")
    tri_doc = partition.get("triangulation")
    volumes = partition.get("volumes")
    entities = partition.get("entities")
    if not isinstance(tri_doc, Mapping) or not isinstance(tri_doc.get("nodes"), list) or \
            not isinstance(tri_doc.get("faces"), list):
        raise _invalid3("the 3D partition has no 'triangulation' with 'nodes' and 'faces'")
    if not isinstance(volumes, list) or not volumes or not isinstance(entities, list):
        raise _invalid3("the 3D partition has no 'volumes' (and 'entities')")
    try:
        nodes = np.asarray(tri_doc["nodes"], dtype=float).reshape(-1, 3)
    except (TypeError, ValueError):
        raise _invalid3("the triangulation's nodes are not a list of [x, y, z]") from None
    if not np.isfinite(nodes).all():
        raise _invalid3("the triangulation has a node that is not finite")
    n_nodes = len(nodes)
    faces, sagitta = {}, {}
    for f in tri_doc["faces"]:
        if not isinstance(f, Mapping) or not isinstance(f.get("tag"), int) or isinstance(f.get("tag"), bool):
            raise _invalid3("a face of the triangulation has no integer 'tag'")
        faces[f["tag"]] = _triples(f.get("triangles", []), f"the triangles of face {f['tag']}", n_nodes)
        sagitta[f["tag"]] = float(f.get("sagitta_m", 0.0)) if _real(f.get("sagitta_m", 0.0)) else 0.0
    sides = {}
    for e in entities:
        if not isinstance(e, Mapping) or not isinstance(e.get("tag"), int) or not isinstance(e.get("side"), list) \
                or len(e["side"]) != 2:
            raise _invalid3("an entity of the 3D partition has no 'tag' and two 'side' regions")
        sides[e["tag"]] = e["side"]

    # the volumes, checked
    vol = {}
    for v in volumes:
        if not isinstance(v, Mapping) or not isinstance(v.get("tag"), int) or not isinstance(v.get("shells"), list) \
                or not v["shells"]:
            raise _invalid3("a volume of the partition has no 'tag' and 'shells'")
        tag, region = v["tag"], v.get("region")
        if region not in regions:
            raise _invalid3(f"volume {tag} names the region {region!r}, which the partition does not have")
        shells = []
        for k, sh in enumerate(v["shells"]):
            if not isinstance(sh, list) or not sh or not all(isinstance(s, int) and s != 0 for s in sh) \
                    or len({abs(s) for s in sh}) != len(sh):
                raise _invalid3(f"shell {k} of volume {tag} is not a list of distinct signed face tags")
            for s in sh:
                if abs(s) not in faces or abs(s) not in sides:
                    raise _invalid3(f"volume {tag} names the face {abs(s)}, which the partition does not have")
                if sides[abs(s)][0 if s > 0 else 1] != region:
                    raise _invalid3(f"volume {tag} (region '{region}') uses face {abs(s)} with the sign of the "
                                   f"region {sides[abs(s)][0 if s > 0 else 1]!r}: a face is used + by its side[0] "
                                   "and - by its side[1]")
            tri = _shell_triangles(sh, faces, tag)
            if not _closed(tri, n_nodes):
                raise _invalid3(f"shell {k} of volume {tag} (region '{region}') is not closed: every edge of its "
                               "triangles must be used once in each direction")
            sv = _signed_volume(nodes, tri)
            if (k == 0 and not sv > 0) or (k > 0 and not sv < 0):
                raise _invalid3(f"shell {k} of volume {tag} encloses a {'negative' if k == 0 else 'positive'} volume "
                               "(the outer shell, the first, a positive one, a cavity a negative one)")
            shells.append((sh, tri))
        vol[tag] = {"tag": tag, "region": region, "shells": shells}
    if len(vol) != len(volumes):
        raise _invalid3("two volumes of the partition have the same tag")
    used_regions = sorted({v["region"] for v in vol.values()},
                          key=lambda r: ((0, regions[r].get("attr"), r) if isinstance(regions[r], Mapping)
                                         and _real(regions[r].get("attr")) else (1, 0, r)))
    mats = _materials(used_regions, materials, region_materials)

    # the mothers: a volume whose outer shell uses a face of a cavity (an inner shell) of N is in N's cavity; volumes
    # that touch across faces of their outer shells form a cluster that shares its mother
    use = {}
    for v in vol.values():
        for k, (sh, _) in enumerate(v["shells"]):
            for s in sh:
                use.setdefault(abs(s), []).append((v["tag"], k))
    parent = {t: t for t in vol}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cavity_of = {t: set() for t in vol}
    for f, u in use.items():
        if len(u) > 2:
            raise _invalid3(f"face {f} is used by more than two volumes")
        if len(u) == 2:
            for (a, ka), (b, kb) in (u, u[::-1]):
                if ka == 0 and kb == 0:
                    parent[root(a)] = root(b)
                elif ka == 0:
                    cavity_of[a].add(b)
    holder = {}
    for t in sorted(vol):
        holder.setdefault(root(t), set()).update(cavity_of[t])
    mother = {}
    for t in vol:
        h = holder[root(t)]
        if len(h) > 1:
            raise _invalid3(f"the volumes around volume {t} lie in the cavities of more than one volume: {sorted(h)}")
        mother[t] = next(iter(h)) if h else None
        if mother[t] is not None and root(mother[t]) == root(t):
            raise _invalid3(f"volume {t} is in a cavity of a volume that touches it from outside")

    # solids: the outer shell's triangles, the vertices in node order, named in a geometric order
    solid = {}
    for t, v in vol.items():
        tri = v["shells"][0][1]
        used = np.unique(tri)
        remap = np.full(n_nodes, -1, np.int64)
        remap[used] = np.arange(len(used))
        xyz = nodes[used]
        solid[t] = {"vertices": xyz, "triangles": remap[tri],
                    "volume": _signed_volume(nodes, tri),
                    "box": tuple(xyz.min(axis=0).tolist()) + tuple(xyz.max(axis=0).tolist()),
                    "sagitta": max(sagitta[abs(s)] for s in v["shells"][0][0])}
    name = {}
    for r in used_regions:
        mine = sorted((t for t in vol if vol[t]["region"] == r), key=lambda t: (solid[t]["box"], solid[t]["volume"], t))
        for k, t in enumerate(mine):
            name[t] = f"{r}_{k}"
    taken = set(name.values())
    if WORLD in taken:
        raise _invalid3(f"a region's solid would be named '{WORLD}', the name of the world")

    radius = float(np.max(np.linalg.norm(nodes, axis=1)))
    world = {"name": WORLD, "kind": "orb", "params": {"r_m": 1.001 * radius}, "mother": None,
             "position_m": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0], "region": None, "material": VACUUM,
             "discretized": False, "sagitta_m": 0.0}
    kids = {None: []}
    for t in vol:
        kids.setdefault(t, [])
    for t in vol:
        kids[mother[t]].append(t)
    solids = [world]

    def visit(m):
        for t in sorted(kids[m], key=lambda x: name[x]):
            s = solid[t]
            solids.append({"name": name[t], "kind": "tessellated",
                           "params": {"vertices_m": s["vertices"].tolist(), "triangles": s["triangles"].tolist()},
                           "mother": name[m] if m is not None else WORLD, "position_m": [0.0, 0.0, 0.0],
                           "rotation_deg": [0.0, 0.0, 0.0], "region": vol[t]["region"],
                           "material": region_materials[vol[t]["region"]], "discretized": True,
                           "sagitta_m": s["sagitta"]})
            visit(t)

    visit(None)
    if len(solids) != len(vol) + 1:
        raise _invalid3("the volumes' cavities contain each other: they have no place in the tree of solids")

    # The triangulated solids must nest and not overlap, as the exact ones do: a layer thinner than the chord tolerance
    # makes the chords of its surfaces cross.
    tol = partition.get("tolerance_m")
    if not _real(tol) or tol <= 0:
        tol = 1e-9 * float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0)))
    tri_of = {t: v["shells"][0][1] for t, v in vol.items()}

    def label(t):
        return f"{name[t]} (region '{vol[t]['region']}')"

    for t in sorted(vol, key=lambda x: name[x]):
        if _self_crosses(nodes, tri_of[t], tol):
            raise _invalid3(f"the surface of {label(t)} crosses itself: the chords of two of its surfaces cut into "
                            "each other, as in a layer thinner than the chord tolerance")
    for t in sorted(vol, key=lambda x: name[x]):
        m = mother[t]
        if m is None:
            continue
        if _surfaces_cross(nodes, tri_of[t], tri_of[m], tol) or not all(
                _inside(nodes, tri_of[m], p) for p in _free_vertices(nodes, tri_of[t], tri_of[m])):
            raise _invalid3(f"{label(t)} is not inside its mother {label(m)}: their triangulated surfaces cross or "
                            "a vertex of the daughter lies outside the mother, as in a layer thinner than the chord "
                            "tolerance")
    for m, ks in kids.items():
        ks = sorted(ks, key=lambda x: name[x])
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = ks[i], ks[j]
                lo_a, hi_a = solid[a]["box"][:3], solid[a]["box"][3:]
                lo_b, hi_b = solid[b]["box"][:3], solid[b]["box"][3:]
                if any(lo_a[k] > hi_b[k] + tol or lo_b[k] > hi_a[k] + tol for k in range(3)):
                    continue
                bad = _surfaces_cross(nodes, tri_of[a], tri_of[b], tol) or any(
                    _inside(nodes, tri_of[b], p) for p in _free_vertices(nodes, tri_of[a], tri_of[b])) or any(
                    _inside(nodes, tri_of[a], p) for p in _free_vertices(nodes, tri_of[b], tri_of[a]))
                if bad:
                    raise _invalid3(f"{label(a)} overlaps {label(b)}: their triangulated surfaces cross or one lies "
                                    "inside the other, as in a layer thinner than the chord tolerance")

    gs, gm = _gdml_names([s["name"] for s in solids], list(mats))
    solids = [{"name": s["name"], "gdml_name": gs[s["name"]], **{k: v for k, v in s.items() if k != "name"}}
              for s in solids]
    mats = {n: {"name": n, "gdml_name": gm[n], **{k: v for k, v in m.items() if k != "name"}}
            for n, m in mats.items()}

    chord = partition.get("chord_tol_m")
    report, out_warnings = {}, []
    for r in used_regions:
        mine = sorted((t for t in vol if vol[t]["region"] == r), key=lambda t: name[t])
        rho = mats[region_materials[r]]["density_kg_m3"]
        exact = float(regions[r]["volume_m3"])
        inst = math.fsum([solid[t]["volume"] for t in mine] + [-solid[d]["volume"] for t in mine for d in kids[t]])
        rep = {"material": region_materials[r], "density_kg_m3": rho, "solids": [name[t] for t in mine],
               "volume_m3": exact, "volume_solids_m3": inst, "volume_rel_diff": inst / exact - 1.0,
               "mass_kg": exact * rho, "mass_solids_kg": inst * rho, "mass_rel_diff": inst / exact - 1.0,
               "sagitta_max_m": max(sagitta[abs(s)] for t in mine for sh, _ in vol[t]["shells"] for s in sh),
               "chord_tol_m": chord}
        report[r] = rep
        if abs(rep["volume_rel_diff"]) > VOLUME_WARN:
            out_warnings.append({"code": "particle.volume_deficit", "region": r,
                                 "volume_rel_diff": rep["volume_rel_diff"], "mass_rel_diff": rep["mass_rel_diff"],
                                 "sagitta_max_m": rep["sagitta_max_m"], "chord_tol_m": rep["chord_tol_m"]})
            warnings.warn(f"particle.volume_deficit: region '{r}': its solids differ from the exact region by "
                          f"{rep['volume_rel_diff']:.3g} in volume and {rep['mass_rel_diff']:.3g} in mass (more than "
                          f"{VOLUME_WARN:g}; largest sagitta {rep['sagitta_max_m']:.3g} m); lower chord_tol_m of the "
                          "export", SolidsWarning, stacklevel=3)
    from . import __version__
    return {"schema": SCHEMA, "generator": f"axionmultiphysics {__version__}", "geometry": "3d",
            "units": {"length": "m", "angle": "deg", "density": "kg/m3", "volume": "m3", "mass": "kg"},
            "chord_tol_m": chord, "world": WORLD, "materials": [mats[n] for n in sorted(mats)], "solids": solids,
            "regions": report, "warnings": out_warnings}


# ============================================================================================ writers
def write(doc: dict, path: str | PathLike) -> None:
    """Writes a solids document as strict JSON (UTF-8, a final newline)."""
    text = json.dumps(doc, indent=1, allow_nan=False, ensure_ascii=False) + "\n"
    with open(Path(path), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _num(x: float) -> str:
    """17 significant digits: the text reads back as the same double."""
    return "%.17g" % float(x)


_SUFFIX = {"_m": "lunit", "_deg": "aunit"}


def write_gdml(doc: dict, path: str | PathLike) -> None:
    """Writes the solids document as GDML: every length in m and every angle in deg with 17 significant digits; each
    document material a material of the file (its density, its composition); NIST materials and elements referenced
    by name and never defined; volumes defined after their daughters. Volumes and materials carry their
    ``gdml_name``; a solid, a placement and a position the volume's name with ``_solid``, ``_pv``, ``_pos``. A
    ``tessellated`` solid is a ``tessellated`` element of ``triangular`` facets (``type="ABSOLUTE"``) whose vertices are
    positions in m defined in ``<define>`` (``<volume>_v<k>``), in the list's order and orientation."""
    q = quoteattr
    gname = {s["name"]: s["gdml_name"] for s in doc["solids"]}
    mname = {m["name"]: m["gdml_name"] for m in doc["materials"]}
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
           f'xsi:noNamespaceSchemaLocation={q(GDML_SCHEMA_LOCATION)}>',
           ]
    # the vertices of the tessellated solids: positions in m, named <volume>_v<k> (a longer prefix if a name of the
    # file would collide)
    vertex_names: dict = {}
    taken = {"Default"} | set(mname.values()) | {g + d for g in gname.values() for d in ("",) + _DERIVED}
    for s in doc["solids"]:
        if s["kind"] != "tessellated":
            continue
        n = len(s["params"]["vertices_m"])
        prefix = gname[s["name"]] + "_v"
        while any(f"{prefix}{k}" in taken for k in range(n)):
            prefix += "_"
        vertex_names[s["name"]] = [f"{prefix}{k}" for k in range(n)]
    if vertex_names:
        out.append("  <define>")
        for s in doc["solids"]:
            if s["name"] in vertex_names:
                for nm, (x, y, z) in zip(vertex_names[s["name"]], s["params"]["vertices_m"]):
                    out.append(f'    <position name={q(nm)} x="{_num(x)}" y="{_num(y)}" z="{_num(z)}" unit="m"/>')
        out.append("  </define>")
    else:
        out.append("  <define/>")
    out.append("  <materials>")
    for m in doc["materials"]:
        state = f" state={q(m['state'])}" if "state" in m else ""
        out.append(f"    <material name={q(m['gdml_name'])}{state}>")
        if "T_K" in m:
            out.append(f'      <T value="{_num(m["T_K"])}" unit="K"/>')
        if "pressure_Pa" in m:
            out.append(f'      <P value="{_num(m["pressure_Pa"])}" unit="pascal"/>')
        if "mean_excitation_eV" in m:
            out.append(f'      <MEE value="{_num(m["mean_excitation_eV"])}" unit="eV"/>')
        out.append(f'      <D value="{_num(m["density_kg_m3"])}" unit="kg/m3"/>')
        comp = m["composition"]
        if "nist" in comp:
            out.append(f'      <fraction n="{_num(1.0)}" ref={q(comp["nist"])}/>')
        elif comp["by"] == "mass":
            for sym, w in comp["elements"].items():
                out.append(f'      <fraction n="{_num(w)}" ref={q(sym)}/>')
        else:
            for sym, n in comp["elements"].items():
                out.append(f'      <composite n="{int(n)}" ref={q(sym)}/>')
        out.append("    </material>")
    out.append("  </materials>")
    out.append("  <solids>")
    for s in doc["solids"]:
        if s["kind"] == "tessellated":
            vn = vertex_names[s["name"]]
            out.append(f"    <tessellated name={q(gname[s['name']] + '_solid')} lunit=\"m\">")
            for i, j, k in s["params"]["triangles"]:
                out.append(f'      <triangular vertex1={q(vn[i])} vertex2={q(vn[j])} vertex3={q(vn[k])} '
                           'type="ABSOLUTE"/>')
            out.append("    </tessellated>")
            continue
        attrs, units, rz = [], set(), None
        for k, v in s["params"].items():
            if k == "rz_m":
                rz = v
                units.add("lunit")
                continue
            for suf, unit in _SUFFIX.items():
                if k.endswith(suf):
                    attrs.append(f'{k[:-len(suf)]}="{_num(v)}"')
                    units.add(unit)
                    break
            else:
                raise ValueError(f"solid {s['name']}: parameter {k} has no unit suffix")
        unit_attrs = (['aunit="deg"'] if "aunit" in units else []) + (['lunit="m"'] if "lunit" in units else [])
        head = f"    <{s['kind']} name={q(s['gdml_name'] + '_solid')} " + " ".join(attrs + unit_attrs)
        if rz is None:
            out.append(head + "/>")
        else:
            out.append(head + ">")
            for r, z in rz:
                out.append(f'      <rzpoint r="{_num(r)}" z="{_num(z)}"/>')
            out.append(f"    </{s['kind']}>")
    out.append("  </solids>")
    out.append("  <structure>")
    kids: dict = {s["name"]: [] for s in doc["solids"]}
    for s in doc["solids"]:
        if s["mother"] is not None:
            kids[s["mother"]].append(s)
    by = {s["name"]: s for s in doc["solids"]}

    def volume(name):
        for k in kids[name]:
            volume(k["name"])
        s = by[name]
        g = gname[name]
        out.append(f"    <volume name={q(g)}>")
        out.append(f"      <materialref ref={q(mname.get(s['material'], s['material']))}/>")
        out.append(f"      <solidref ref={q(g + '_solid')}/>")
        for k in kids[name]:
            kg = gname[k["name"]]
            out.append(f"      <physvol name={q(kg + '_pv')}>")
            out.append(f"        <volumeref ref={q(kg)}/>")
            x, y, z = k["position_m"]
            out.append(f'        <position name={q(kg + "_pos")} x="{_num(x)}" y="{_num(y)}" z="{_num(z)}" '
                       'unit="m"/>')
            if any(a != 0.0 for a in k["rotation_deg"]):
                rx, ry, rz_ = k["rotation_deg"]
                out.append(f'        <rotation name={q(kg + "_rot")} x="{_num(rx)}" y="{_num(ry)}" '
                           f'z="{_num(rz_)}" unit="deg"/>')
            out.append("      </physvol>")
        out.append("    </volume>")

    volume(doc["world"])
    out.append("  </structure>")
    out.append('  <setup name="Default" version="1.0">')
    out.append(f"    <world ref={q(gname[doc['world']])}/>")
    out.append("  </setup>")
    out.append("</gdml>")
    with open(Path(path), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
