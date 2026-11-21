"""Solids from the mesher's partition, and their GDML export (``axionmultiphysics.solids``, beta).

The conversion turns an axisymmetric partition (the mesher's entity table with exact curve parameters and face
loops) into the list of native solids that the particle transport instantiates (``solids.json``, schema
``axionmultiphysics.solids/1``): each face's outer loop, with its notches at the axis filled, revolved by 2 pi; the
faces and exterior voids nested in it as its daughters; per region a volume and mass report against the exact
solid of revolution. The GDML writer writes the same list. Nothing here needs gmsh or Geant4: the partitions are
built by hand in ``_solids_ref.py`` in the shape of the mesher's info file.

Independent references: closed-form volumes of the exact solids (cylinder, torus, sphere, cone frustum, spheroid,
elliptic torus) and quadrature of the exact cross-sections (fillet, circular segment); the volume of a revolved
polygon as a sum of cone frusta (not Pappus' centroid form); a point-membership model of each kind of solid that
walks the mother tree as Geant4's navigator does; Geant4's unit values for the GDML attributes.

Tolerances:
- geometric parameters copied from the partition: bit for bit; those computed by one subtraction or halving
  (a tube's length and centre): 2 units in the last place of the operands' scale (2.3e-16 m at 0.5 m);
- volumes of native solids against closed forms: 1e-12 relative. Each side has a few dozen rounded operations of
  relative size 1.1e-16 (the fixtures avoid cancellation: no wall thinner than a quarter of its radius), so 1e-14
  covers them; 1e-12 keeps a margin and is still nine orders below the smallest defect tested (a missing factor 2,
  a factor 1000, a centroid at the middle of the box: all O(1) or above 1e-3 on these shapes);
- exact volumes of curved faces against quadrature: 1e-11 (quad at epsrel 1e-13 on smooth integrands, plus the
  summation in the closed forms);
- discretized solids: the sagitta bounds the volume change, |dV| <= 2 pi r_max s L for curves of length L.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import axionmultiphysics as axmp
import _solids_ref as ref

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {"native": ref.native, "discretized": ref.discretized, "voids": ref.voids, "degenerate": ref.degenerate}
XSI = "{http://www.w3.org/2001/XMLSchema-instance}"
NCNAME = re.compile(r"[A-Za-z_][A-Za-z0-9._-]*")     # an xs:ID of GDML's schema, in ASCII
CODES = ("particle.planar_not_offered", "particle.partition_invalid", "particle.composition_missing",
         "particle.density_missing", "particle.composition_invalid", "particle.volume_deficit")


def solids_module():
    """The module under test (an ImportError here means the API does not exist yet)."""
    return importlib.import_module("axionmultiphysics.solids")


@pytest.fixture(scope="module")
def S():
    return solids_module()


_FX: dict = {}
_DOCS: dict = {}


def fx_of(name):
    if name not in _FX:
        _FX[name] = FIXTURES[name]()
    return _FX[name]


def doc_of(name):
    """The solids document of a fixture with the default chord tolerance (cached; tests must not modify it)."""
    if name not in _DOCS:
        fx = fx_of(name)
        with warnings.catch_warnings():
            warnings.simplefilter("error", solids_module().SolidsWarning)
            _DOCS[name] = solids_module().from_partition(*fx.args())
    return _DOCS[name]


def by_name(doc):
    return {s["name"]: s for s in doc["solids"]}


def by_gdml(doc):
    """The solids by their GDML (and Geant4) name."""
    return {s["gdml_name"]: s for s in doc["solids"]}


def material_gdml(doc, name):
    """The GDML name of a material of the document; a NIST material keeps its name."""
    for m in doc["materials"]:
        if m["name"] == name:
            return m["gdml_name"]
    assert name.startswith("G4_"), name
    return name


def children(doc):
    kids = {s["name"]: [] for s in doc["solids"]}
    for s in doc["solids"]:
        if s["mother"] is not None:
            kids[s["mother"]].append(s["name"])
    return kids


def region_solids(doc, region):
    return [s for s in doc["solids"] if s["region"] == region]


def only(doc, region):
    ss = region_solids(doc, region)
    assert len(ss) == 1, f"region {region}: {len(ss)} solids"
    return ss[0]


def own_volume(doc, s):
    """Volume of a solid minus its daughters' solids, from the parameters (closed forms and frusta)."""
    kids = children(doc)[s["name"]]
    b = by_name(doc)
    return ref.solid_volume(s) - sum(ref.solid_volume(b[k]) for k in kids)


def code_of(excinfo):
    return getattr(excinfo.value, "code", None)


def gdml_of(S, doc, tmp_path, name="d.gdml"):
    path = tmp_path / name
    S.write_gdml(doc, path)
    return ET.parse(path).getroot(), path


def rel(a, b):
    return abs(a / b - 1.0)


# ============================================================================================ the API and the format
def test_module_declares_its_schema_constants_and_beta_status(S):
    assert S.SCHEMA == "axionmultiphysics.solids/1"
    assert S.VOLUME_WARN == 1e-4
    assert issubclass(S.SolidsWarning, UserWarning)
    assert re.search(r"\bbeta\b", S.__doc__ or "", re.I)
    # chord_tol defaults to min(1 um, 1e-5 x region size)
    assert S.default_chord_tol(0.5) == 1e-6
    assert S.default_chord_tol(0.013) == pytest.approx(1.3e-7, rel=1e-15)
    for f in ("from_partition", "write", "write_gdml"):
        assert callable(getattr(S, f))


def test_document_header_units_and_strict_json(S, tmp_path):
    doc = doc_of("native")
    assert doc["schema"] == "axionmultiphysics.solids/1"
    assert doc["geometry"] == "axisymmetric"
    assert doc["units"] == {"length": "m", "angle": "deg", "density": "kg/m3", "volume": "m3", "mass": "kg"}
    assert doc["world"] == "world"
    assert doc["generator"] == f"axionmultiphysics {axmp.__version__}"
    assert doc["warnings"] == []
    path = tmp_path / "solids.json"
    S.write(doc, path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")

    def no_constant(c):
        raise AssertionError(f"non-finite number {c} in solids.json")

    assert json.loads(text, parse_constant=no_constant) == doc
    # no host, path or time in the document
    assert str(tmp_path) not in text and "seconds" not in text


# ============================================================================================ native solids
def test_native_kinds_and_parameters():
    """Every face of the native fixture becomes one exact solid of the kind and size of its shape. The parameters
    are in metres and degrees; values the partition states are copied bit for bit."""
    doc = doc_of("native")
    fx = fx_of("native")
    zabs = ref.absolute_z(doc)
    names = {s["name"] for s in doc["solids"]}
    for region in fx.regions:
        s = only(doc, region)
        assert s["name"] == f"{region}_0"
        assert s["kind"] == fx.kinds[region], region
        assert s["discretized"] is False and s["sagitta_m"] == 0.0
        assert s["position_m"][0] == 0.0 and s["position_m"][1] == 0.0
        assert s["rotation_deg"] == [0.0, 0.0, 0.0]
        if s["kind"] != "orb":
            assert s["params"]["startphi_deg"] == 0.0 and s["params"]["deltaphi_deg"] == 360.0, region
    assert len(names) == len(doc["solids"])
    ulp = 2.3e-16

    def tube(region, rmin, rmax, z0, z1):
        p = only(doc, region)["params"]
        assert set(p) == {"rmin_m", "rmax_m", "z_m", "startphi_deg", "deltaphi_deg"}
        assert p["rmin_m"] == rmin and p["rmax_m"] == rmax, region
        assert abs(p["z_m"] - (z1 - z0)) <= 2 * ulp, region            # the full length, as GDML's tube states it
        assert abs(zabs[f"{region}_0"] - 0.5 * (z0 + z1)) <= 2 * ulp, region

    tube("air", 0.0, 0.53, -0.31, 0.43)                  # its notch at the axis (the core) is filled: a full tube
    tube("insert", 0.17, 0.41, -0.25, -0.07)
    tube("pin", 0.23, 0.31, -0.19, -0.12)
    core = only(doc, "core")
    assert core["params"] == {"r_m": 0.061}              # the arc is typed "ellipse" with equal radii: a circle
    assert abs(zabs["core_0"] - 0.07) <= 2 * ulp
    ring = only(doc, "ring")["params"]
    assert set(ring) == {"rmin_m", "rmax_m", "rtor_m", "startphi_deg", "deltaphi_deg"}
    assert ring["rmin_m"] == 0.0 and ring["rmax_m"] == 0.083 and ring["rtor_m"] == 0.29
    assert abs(zabs["ring_0"] - 0.23) <= 2 * ulp
    cone = only(doc, "cone")["params"]
    assert set(cone) == {"rz_m", "startphi_deg", "deltaphi_deg"}
    rz = [tuple(p) for p in cone["rz_m"]]
    want = [(0.0, 0.43), (0.2, 0.43), (0.11, 0.59), (0.0, 0.59)]
    k = rz.index(want[0])
    assert rz[k:] + rz[:k] == want                       # the partition's vertices, counter-clockwise, no others
    assert zabs["cone_0"] + 0.0 == 0.0                   # a polycone's (r, z) points are absolute


def test_world_is_the_one_root_and_encloses_everything():
    for name in FIXTURES:
        doc = doc_of(name)
        roots = [s for s in doc["solids"] if s["mother"] is None]
        assert [s["name"] for s in roots] == ["world"] and doc["solids"][0]["name"] == "world"
        w = roots[0]
        assert w["kind"] == "tube" and w["region"] is None and w["material"] == "G4_Galactic"
        assert w["params"]["rmin_m"] == 0.0 and w["position_m"] == [0.0, 0.0, 0.0]
        diag = fx_of(name).table()["diagonal_m"]
        for s in doc["solids"]:
            if s["mother"] == "world":
                pts = ref.boundary_samples(s, ref.absolute_z(doc)[s["name"]])
                gap = np.minimum(w["params"]["rmax_m"] - pts[:, 0], 0.5 * w["params"]["z_m"] - np.abs(pts[:, 1]))
                assert gap.min() > 1e-6 * diag, s["name"]     # the axis is shared; the outer faces are not


# ============================================================================================ the mother tree
@pytest.mark.parametrize("name", ["native", "voids"])
def test_mother_tree_follows_the_nesting_of_faces(name):
    doc, fx = doc_of(name), fx_of(name)
    b = by_name(doc)
    seen = set()
    for s in doc["solids"]:       # every solid after its mother
        assert s["mother"] is None or s["mother"] in seen, s["name"]
        seen.add(s["name"])
    for region, mother_region in fx.tree.items():
        for s in region_solids(doc, region):
            m = b[s["mother"]]
            assert (m["region"] if m["name"] != "world" else None) == mother_region, (region, s["mother"])


@pytest.mark.parametrize("name", list(FIXTURES))
def test_daughters_lie_in_their_mothers_and_siblings_do_not_overlap(name):
    """No overlap by construction: every boundary point of a daughter is inside or on its mother, and no boundary
    point of a solid is inside a sibling (both within 1e-12 of the size; coincident faces are allowed)."""
    doc = doc_of(name)
    tol = 1e-12 * fx_of(name).table()["diagonal_m"]
    b, kids, zabs = by_name(doc), children(doc), ref.absolute_z(doc)
    for s in doc["solids"]:
        pts = ref.boundary_samples(s, zabs[s["name"]])
        if s["mother"] is not None:
            m = b[s["mother"]]
            d = ref.depth(m, zabs[m["name"]], pts[:, 0], pts[:, 1])
            assert d.min() >= -tol, f"{s['name']} leaves its mother {m['name']} by {-d.min():.3g} m"
        for k in kids[s["mother"]] if s["mother"] else []:
            if k != s["name"]:
                d = ref.depth(b[k], zabs[k], pts[:, 0], pts[:, 1])
                assert d.max() <= tol, f"{s['name']} enters its sibling {k} by {d.max():.3g} m"


@pytest.mark.parametrize("name", list(FIXTURES))
def test_navigation_finds_the_region_of_the_partition_everywhere(name):
    """Walking the tree as Geant4 does (into the one daughter that contains the point) gives, at every point of a
    grid away from the boundaries, the region of the partition there; outside the regions and in voids, none."""
    fx, doc = fx_of(name), doc_of(name)
    R, Z, want = ref.interior_grid(fx)
    assert len(R) > 20000
    got = ref.navigate(doc, R, Z)
    bad = np.flatnonzero(got != want)
    assert len(bad) == 0, [(R[i], Z[i], want[i], got[i]) for i in bad[:5]]


def test_exterior_holes_and_notches_become_vacuum_daughters():
    doc, fx = doc_of("voids"), fx_of("voids")
    b, zabs = by_name(doc), ref.absolute_z(doc)
    voids = [s for s in doc["solids"] if s["region"] is None and s["name"] != "world"]
    assert len(voids) == len(fx.voids)
    found = []
    for s in voids:
        assert re.fullmatch(r"void_\d+", s["name"]) and s["material"] == "G4_Galactic"
        m = b[s["mother"]]
        p = s["params"]
        if s["kind"] == "tube":
            shape = (p["rmin_m"], p["rmax_m"], zabs[s["name"]] - 0.5 * p["z_m"], zabs[s["name"]] + 0.5 * p["z_m"])
        else:
            shape = (p["r_m"], zabs[s["name"]])
        found.append((s["kind"], m["region"], tuple(round(x, 12) for x in shape)))
    want = [(k, m, tuple(round(x, 12) for x in sh)) for k, m, sh in fx.voids]
    assert sorted(found) == sorted(want)


@pytest.mark.parametrize("name", list(FIXTURES))
def test_every_face_is_in_the_tree(name):
    """No face is ever dropped: each region has one solid per face of the partition."""
    doc, t = doc_of(name), fx_of(name).table()
    for region in fx_of(name).regions:
        nfaces = sum(1 for f in t["faces"] if f["region"] == region)
        assert len(region_solids(doc, region)) == nfaces, region
        assert len(doc["regions"][region]["solids"]) == nfaces, region


@pytest.mark.parametrize("kind", ["ellipse", "torus"])
def test_a_layer_thinner_than_the_chord_tolerance_is_refused_never_dropped(S, kind):
    """A film 0.3 um thick around a core, below the default chord tolerance (1 um): the discretized chains of its
    two curves cross (ellipse: each face's polygon holds the other's inner point; torus: the exact core pokes out of
    the film's chords). The conversion refuses it with the hint to lower chord_tol_m, and never writes a list
    without the film and the core or with overlapping solids. With chord_tol_m = 30 nm the same partition converts
    completely, with the core inside the film and the exact volumes (within 1e-13 of the total: the film's volume
    is a difference of two volumes 2e5 times larger)."""
    fx = ref.thin_film(3e-7, kind)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        S.from_partition(*fx.args())
    assert code_of(e) == "particle.partition_invalid"
    assert "chord_tol_m" in str(e.value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", S.SolidsWarning)
        doc = S.from_partition(*fx.args(), chord_tol_m=3e-8)
    film, core = only(doc, "film"), only(doc, "core")
    assert core["mother"] == film["name"] and film["mother"] == "world"
    zabs = ref.absolute_z(doc)
    pts = ref.boundary_samples(core, zabs[core["name"]], 2 if core["kind"] == "genericPolycone" else 4096)
    assert ref.depth(film, zabs[film["name"]], pts[:, 0], pts[:, 1]).min() > 1e-7      # inside, by most of the gap
    total = sum(fx.exact.values())
    for region in ("film", "core"):
        assert abs(doc["regions"][region]["volume_m3"] - fx.exact[region]) <= 1e-13 * total, region


def _converted(S, fx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", S.SolidsWarning)
        return S.from_partition(*fx.args(), **kw)


def test_siblings_closer_than_their_chords_are_refused(S):
    """A plate with a concave bite and an exact torus in the bite, separate faces (siblings in the world): the
    plate's chords lie inside the bite's circle by their sagitta s (exact for a circle, about 1 um here). A gap of
    0.3 um, and a gap of s - 100 nm (an overlap of 100 nm), are refused with the chord_tol_m hint; a gap of
    s + 100 nm converts with the two solids apart, and so does 0.3 um with chord_tol_m = 30 nm. 100 nm is far above
    rounding (1e-16 m) and below any tolerance a check may use: Geant4's overlap check refuses it."""
    s = only(_converted(S, ref.bite(1e-5)), "plate")["sagitta_m"]
    assert 5e-7 < s <= 1e-6
    for gap in (3e-7, s - 1e-7):
        with pytest.raises(axmp.AxionMultiPhysicsError) as e:
            S.from_partition(*ref.bite(gap).args())
        assert code_of(e) == "particle.partition_invalid" and "chord_tol_m" in str(e.value), gap
    for fx, kw in ((ref.bite(s + 1e-7), {}), (ref.bite(3e-7), {"chord_tol_m": 3e-8})):
        doc = _converted(S, fx, **kw)
        plate, ball = only(doc, "plate"), only(doc, "ball")
        assert plate["mother"] == ball["mother"] == "world" and ball["kind"] == "torus"
        zabs = ref.absolute_z(doc)
        pts = ref.boundary_samples(ball, zabs[ball["name"]], 20000)
        assert ref.depth(plate, zabs[plate["name"]], pts[:, 0], pts[:, 1]).max() < 0.0
        for region in ("plate", "ball"):
            assert rel(doc["regions"][region]["volume_m3"], fx.exact[region]) <= 1e-12, region


def test_a_face_whose_outline_would_cross_itself_is_refused(S):
    """One face with a 0.3 um slot between two concentric arcs: the outer arc's chords (sagitta about 1 um) would
    cross the inner arc's, so the face's polycone would cross itself. Refused with the chord_tol_m hint; with
    chord_tol_m = 30 nm the outline is a simple counter-clockwise polygon and the volume exact."""
    fx = ref.slot(3e-7)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        S.from_partition(*fx.args())
    assert code_of(e) == "particle.partition_invalid" and "chord_tol_m" in str(e.value)
    doc = _converted(S, fx, chord_tol_m=3e-8)
    poly = np.asarray(only(doc, "f")["params"]["rz_m"])
    assert ref.polygon_signed_area(poly) > 0 and ref.simple_polygon(poly)
    assert rel(doc["regions"]["f"]["volume_m3"], fx.exact["f"]) <= 1e-12


def test_overlaps_of_picometres_are_refused_at_any_size(S):
    """The plate with a concave bite and a ball in it, scaled by 20 (tolerance 1e-12 of the diagonal would be 8.5e-12
    m): the ball larger than the bite by 4 pm or 8 pm overlaps the plate's chords, and Geant4's overlap check
    (tolerance 1e-12 m, so a half of it for each side) refuses such a pair, so the conversion refuses it too, at the
    absolute tolerance of 5e-13 m. A ball 1 nm clear of the chords converts."""
    k = 20.0
    s = only(_converted(S, ref.bite(1e-3, k)), "plate")["sagitta_m"]
    assert 5e-7 < s <= 1e-6
    for overlap in (4e-12, 8e-12):
        with pytest.raises(axmp.AxionMultiPhysicsError) as e:
            S.from_partition(*ref.bite(s - overlap, k).args())
        assert code_of(e) == "particle.partition_invalid", overlap
    doc = _converted(S, ref.bite(s + 1e-9, k))
    assert only(doc, "plate")["mother"] == only(doc, "ball")["mother"] == "world"


def test_tangent_curves_are_refused_and_the_hint_says_they_must_be_separated(S):
    """A torus tangent to the inside of a discretized ellipse at its vertex pokes out of the chords by about 0.8
    chord tolerances at every tolerance: it is refused at the default and at a hundred times finer, and the hint
    names the contact, not only the tolerance."""
    P = ref.Partition()
    c = (0.3, 0.0)
    eo = P.closed(c, (0.1, 0.05))
    ei = P.closed((0.3 - 0.1 + 0.02, 0.0), (0.02, 0.02))        # touches the ellipse at its vertex (0.2, 0)
    P.face("film", [eo], [-ei])
    P.face("core", [ei])
    fx = ref.Fixture(P, ["film", "core"], {"copper": {"density": 8960.0, "particle": {"nist": "G4_Cu"}}},
                     {"film": "copper", "core": "copper"}, {}, None, {}, {})
    for kw in ({}, {"chord_tol_m": 1e-8}):
        with pytest.raises(axmp.AxionMultiPhysicsError) as e:
            S.from_partition(*fx.args(), **kw)
        assert code_of(e) == "particle.partition_invalid", kw
        assert "touch" in str(e.value) and "separate" in str(e.value), kw


@pytest.mark.parametrize("tilt", [0.0, 0.5, -0.7])
def test_a_face_pinched_at_a_point_is_refused(S, tilt):
    """A face whose outline passes twice through one point (a notch, a face of its own, whose tip touches the top
    boundary), with a horizontal and with sloped top boundaries: whether Geant4 builds the polycone then depends on
    rounding in its crossing test (about 1 in 15 of the sloped ones it refuses), so the conversion refuses it,
    naming the region and the point, with the hint to separate the boundaries."""
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        S.from_partition(*ref.pinch(tilt).args())
    assert code_of(e) == "particle.partition_invalid"
    msg = str(e.value)
    assert "region 'c'" in msg and "(0.3, 0.1)" in msg and "separate" in msg


def test_faces_touching_the_axis_or_each_other_at_one_point():
    """A triangle with one vertex on the axis inside a box whose loop passes through that point twice (a pocket
    that is not a notch between two axis segments), squares that share one vertex, a circle tangent to the axis:
    all convert, with the exact volumes (the tangent circle is discretized: 1e-11 for its exact volume, the sagitta
    bound for its solid)."""
    doc, fx = doc_of("degenerate"), fx_of("degenerate")
    for region in fx.regions:
        rep = doc["regions"][region]
        assert rel(rep["volume_m3"], fx.exact[region]) <= (1e-11 if region == "tang" else 1e-12), region
        if region != "tang":
            assert rel(rep["volume_solids_m3"], fx.exact[region]) <= 1e-12, region


# ============================================================================================ refusals of the partition
def test_a_planar_partition_is_refused(S):
    fx = fx_of("native")
    table, mats, rm = fx.args()
    table["geometry"] = "planar"
    for e in table["entities"]:
        e["axis"] = False
        e["area_m2"] = 0.0
    for r in table["regions"].values():
        r.pop("volume_m3", None)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        S.from_partition(table, mats, rm)
    assert code_of(e) == "particle.planar_not_offered"
    assert "planar" in str(e.value) and "axisymmetric" in str(e.value)


@pytest.mark.parametrize("case", ["mesh_entity", "open_loop", "wrong_side", "no_faces", "unknown_tag"])
def test_a_partition_that_does_not_describe_faces_is_refused(S, case):
    table, mats, rm = fx_of("native").args()
    if case == "mesh_entity":       # an imported mesh has no exact curves
        table["entities"][0]["type"] = "mesh"
    elif case == "open_loop":
        table["faces"][-1]["loops"][0].pop(1)
    elif case == "wrong_side":      # a face that runs an entity forward while its region is on the entity's right
        table["faces"][4]["loops"][0][0] *= -1
    elif case == "no_faces":
        del table["faces"]
    else:
        table["faces"][0]["loops"][0][0] = 9999
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        S.from_partition(table, mats, rm)
    assert code_of(e) == "particle.partition_invalid", case


# ============================================================================================ materials and compositions
def test_materials_of_the_document():
    doc = doc_of("native")
    mats = {m["name"]: m for m in doc["materials"]}
    assert [m["name"] for m in doc["materials"]] == ["air", "copper", "iron", "polyethylene", "steel"]  # used, sorted
    assert mats["iron"] == {"name": "iron", "gdml_name": "iron", "density_kg_m3": 7874.0,
                            "composition": {"nist": "G4_Fe"}}
    assert mats["steel"]["composition"] == {"elements": {"Cr": 0.19, "Fe": 0.70, "Ni": 0.11}, "by": "mass"}
    assert list(mats["steel"]["composition"]["elements"]) == ["Cr", "Fe", "Ni"]
    assert mats["air"]["state"] == "gas" and mats["air"]["density_kg_m3"] == 1.205
    assert mats["polyethylene"]["composition"] == {"elements": {"C": 1, "H": 2}, "by": "atoms"}
    assert mats["polyethylene"]["mean_excitation_eV"] == 57.4
    fx = fx_of("native")
    for region in fx.regions:
        assert doc["regions"][region]["material"] == fx.region_materials[region]
        assert doc["regions"][region]["density_kg_m3"] == fx.materials[fx.region_materials[region]]["density"]
        for s in region_solids(doc, region):
            assert s["material"] == fx.region_materials[region]


REFUSALS = {
    # case: (edit of (materials, region_materials), code, word the message names)
    "region_without_material": (lambda m, r: r.pop("ring"), "particle.composition_missing", "ring"),
    "material_without_particle": (lambda m, r: m["copper"].pop("particle"), "particle.composition_missing", "copper"),
    "material_not_defined": (lambda m, r: r.update(ring="bronze"), "particle.composition_missing", "bronze"),
    "no_density": (lambda m, r: m["copper"].pop("density"), "particle.density_missing", "copper"),
    "zero_density": (lambda m, r: m["copper"].update(density=0.0), "particle.density_missing", "copper"),
    "nan_density": (lambda m, r: m["copper"].update(density=float("nan")), "particle.density_missing", "copper"),
    "unknown_element": (lambda m, r: m["steel"]["particle"]["elements"].update(
        Xx=m["steel"]["particle"]["elements"].pop("Fe")), "particle.composition_invalid", "Xx"),
    "fractions_not_one": (lambda m, r: m["steel"]["particle"]["elements"].update(Ni=0.10),
                          "particle.composition_invalid", "steel"),
    "atoms_not_integer": (lambda m, r: m["polyethylene"]["particle"]["elements"].update(H=1.5),
                          "particle.composition_invalid", "polyethylene"),
    "nist_and_elements": (lambda m, r: m["copper"]["particle"].update(elements={"Cu": 1.0}, by="mass"),
                          "particle.composition_invalid", "copper"),
    "no_by": (lambda m, r: m["steel"]["particle"].pop("by"), "particle.composition_invalid", "steel"),
    "nist_name_without_prefix": (lambda m, r: m["copper"]["particle"].update(nist="Cu"),
                                 "particle.composition_invalid", "copper"),
    "material_named_like_nist": (lambda m, r: (m.update(G4_Cu=m.pop("copper")), r.update(ring="G4_Cu")),
                                 "particle.composition_invalid", "G4_Cu"),
    "unknown_key": (lambda m, r: m["copper"]["particle"].update(colour="red"), "particle.composition_invalid",
                    "colour"),
    "bad_state": (lambda m, r: m["air"]["particle"].update(state="plasma"), "particle.composition_invalid", "air"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_material_without_a_valid_composition_or_density_is_refused(S, case):
    """A transported material needs a composition and a density; nothing is ever treated as vacuum."""
    edit, code, word = REFUSALS[case]
    table, mats, rm = fx_of("native").args()
    edit(mats, rm)
    with pytest.raises(axmp.AxionMultiPhysicsError) as e:
        S.from_partition(table, mats, rm)
    assert code_of(e) == code, case
    assert word in str(e.value), case


@pytest.mark.parametrize("symbol, built", [("Bh", True), ("Hs", False), ("Mt", False), ("Og", False)])
def test_elements_are_those_geant4_builds_z_1_to_107(S, symbol, built):
    """Geant4's NIST element builder has the elements Z = 1 to 107 (bohrium, Bh, is the last); a composition with a
    heavier one (Hs is Z = 108) is refused at conversion, so no exported file names an element Geant4 cannot build."""
    table, mats, rm = fx_of("native").args()
    els = mats["steel"]["particle"]["elements"]
    els[symbol] = els.pop("Ni")
    if built:
        doc = S.from_partition(table, mats, rm)
        steel = next(m for m in doc["materials"] if m["name"] == "steel")
        assert steel["composition"]["elements"][symbol] == 0.11
    else:
        with pytest.raises(axmp.AxionMultiPhysicsError) as e:
            S.from_partition(table, mats, rm)
        assert code_of(e) == "particle.composition_invalid" and symbol in str(e.value)


# ============================================================================================ volume and mass report
@pytest.mark.parametrize("name", ["native", "voids"])
def test_native_region_volumes_equal_the_exact_solids(name):
    """Pappus volumes of the faces against closed forms of the exact solids (sphere, torus, cylinders, cone
    frustum, spherical shell); nothing is discretized, so the instantiated solids hold the same volumes."""
    doc, fx = doc_of(name), fx_of(name)
    for region in fx.regions:
        rep = doc["regions"][region]
        assert rel(rep["volume_m3"], fx.exact[region]) <= 1e-12, (region, rep["volume_m3"], fx.exact[region])
        assert rel(rep["volume_solids_m3"], fx.exact[region]) <= 1e-12, region
        assert abs(rep["volume_rel_diff"]) <= 1e-12 and rep["sagitta_max_m"] == 0.0, region


@pytest.mark.parametrize("name", list(FIXTURES))
def test_report_equals_the_instantiated_solids(name):
    """A region's instantiated volume is the sum over its solids of the solid minus its daughters, each from the
    closed form of its kind or the frusta of its polygon (the computation Geant4 does), and the relative difference
    is the instantiated against the exact volume."""
    doc, fx = doc_of(name), fx_of(name)
    assert list(doc["regions"]) == fx.regions                      # document order
    for region in fx.regions:
        rep = doc["regions"][region]
        v = sum(own_volume(doc, s) for s in region_solids(doc, region))
        assert rel(rep["volume_solids_m3"], v) <= 1e-12, region
        assert rep["solids"] == [s["name"] for s in region_solids(doc, region)]
        assert abs(rep["volume_rel_diff"] - (rep["volume_solids_m3"] / rep["volume_m3"] - 1.0)) <= 1e-15, region


def test_discretized_region_volumes_against_exact_solids_and_the_sagitta_bound():
    doc, fx = doc_of("discretized"), fx_of("discretized")
    t = fx.table()
    for region in fx.regions:
        rep = doc["regions"][region]
        # the exact solid of revolution: closed forms (spheroid, elliptic torus) or quadrature (fillet, segment)
        assert rel(rep["volume_m3"], fx.exact[region]) <= 1e-11, (region, rep["volume_m3"], fx.exact[region])
        assert rep["sagitta_max_m"] > 0.0
        assert rep["sagitta_max_m"] <= rep["chord_tol_m"] * (1 + 1e-12)
        # the discretization moves the volume by at most 2 pi r_max s L over its curved boundary
        curved = [e for e in t["entities"] if e["type"] != "line" and region in e["side"]]
        L = sum(e["length_m"] for e in curved)
        rmax = max(e["hi"][0] for e in curved)
        dv = abs(rep["volume_solids_m3"] - rep["volume_m3"])
        assert 0.0 < dv <= 2 * math.pi * rmax * rep["sagitta_max_m"] * L, region
    # the pole and the gap share the fillet's vertices: together they fill the rectangle with no gap or sliver
    total = doc["regions"]["pole"]["volume_solids_m3"] + doc["regions"]["gap"]["volume_solids_m3"]
    assert rel(total, math.pi * (0.25 ** 2 - 0.05 ** 2) * 0.18) <= 1e-12


def test_masses_are_volumes_times_the_document_density():
    for name in FIXTURES:
        doc, fx = doc_of(name), fx_of(name)
        for region in fx.regions:
            rep = doc["regions"][region]
            rho = fx.materials[fx.region_materials[region]]["density"]
            assert rep["density_kg_m3"] == rho
            assert rel(rep["mass_kg"], rep["volume_m3"] * rho) <= 2e-16, region
            assert rel(rep["mass_solids_kg"], rep["volume_solids_m3"] * rho) <= 2e-16, region
            assert abs(rep["mass_rel_diff"] - rep["volume_rel_diff"]) <= 1e-15, region
            assert rel(rep["mass_kg"], fx.exact[region] * rho) <= 1e-11, region
    # one by hand: the iron core, a sphere of radius 61 mm, is 7.4864 kg
    assert doc_of("native")["regions"]["core"]["mass_kg"] == pytest.approx(4 / 3 * math.pi * 0.061 ** 3 * 7874.0,
                                                                             rel=1e-12)


def test_a_volume_change_above_the_threshold_warns(S):
    """Above 1e-4 (relative, volume or mass) the region is named in a warning; below it, nothing."""
    assert doc_of("discretized")["warnings"] == []                # default chord tolerance: no warning (checked
    table, mats, rm = fx_of("discretized").args()                 # as an error in doc_of)
    with pytest.warns(S.SolidsWarning) as rec:
        doc = S.from_partition(table, mats, rm, chord_tol_m=1e-3)
    over = sorted(r for r, v in doc["regions"].items() if abs(v["volume_rel_diff"]) > S.VOLUME_WARN)
    assert {"dome", "gap"} <= set(over)                           # a sanity check of the case itself
    assert sorted(w["region"] for w in doc["warnings"]) == over
    assert all(w["code"] == "particle.volume_deficit" for w in doc["warnings"])
    msgs = [str(w.message) for w in rec if issubclass(w.category, S.SolidsWarning)]
    assert sorted(m.split("'")[1] for m in msgs) == over
    assert all(m.startswith("particle.volume_deficit") for m in msgs)
    assert doc["chord_tol_m"] == 1e-3 and all(v["chord_tol_m"] == 1e-3 for v in doc["regions"].values())


# ============================================================================================ discretization
def test_default_chord_tolerance_is_per_region():
    """min(1 um, 1e-5 x the diagonal of the region's box); an entity shared by two regions uses the smaller."""
    doc, t = doc_of("discretized"), fx_of("discretized").table()
    assert doc["chord_tol_m"] is None
    for region, rep in doc["regions"].items():
        ents = [e for e in t["entities"] if region in e["side"]]
        lo = np.min([e["lo"][:2] for e in ents], axis=0)
        hi = np.max([e["hi"][:2] for e in ents], axis=0)
        assert rep["chord_tol_m"] == pytest.approx(min(1e-6, 1e-5 * float(np.hypot(*(hi - lo)))), rel=1e-12), region
    # the gap (a 43 mm corner) has a tolerance below 1 um, and the fillet it shares with the pole uses it
    assert doc["regions"]["gap"]["chord_tol_m"] < 1e-6
    assert doc["regions"]["pole"]["sagitta_max_m"] <= doc["regions"]["gap"]["chord_tol_m"] * (1 + 1e-12)


def _curve(fx, tag):
    return ref.Curve(fx.P.edges[tag - 1])


def test_arcs_are_discretized_within_the_tolerance_on_the_exact_curve():
    """Every vertex of a discretized solid is a vertex of the partition (bit for bit) or a point of an arc; the
    exact arc stays within the reported sagitta of the solid's polygon, and that is within the tolerance."""
    doc, fx = doc_of("discretized"), fx_of("discretized")
    t = fx.table()
    ends = {tuple(e[k][:2]) for e in t["entities"] for k in ("start", "end")}
    for region in fx.regions:
        s = only(doc, region)
        assert s["kind"] == "genericPolycone" and s["discretized"] is True
        poly = np.asarray(s["params"]["rz_m"], float)
        assert ref.polygon_signed_area(poly) > 0                   # counter-clockwise in (r, z)
        arcs = [e for e in t["entities"] if e["type"] != "line" and region in e["side"]]
        for v in map(tuple, poly):
            if v in ends:
                continue
            assert any(_on_arc(e, v) for e in arcs), (region, v)
        worst = 0.0
        for e in arcs:
            pts = _curve(fx, e["tag"]).at(np.linspace(0, 1, 4001))
            worst = max(worst, float(ref.distance_to_polygon(poly, pts[:, 0], pts[:, 1]).max()))
        assert worst <= s["sagitta_m"] * (1 + 1e-6), (region, worst, s["sagitta_m"])
        assert s["sagitta_m"] <= doc["regions"][region]["chord_tol_m"] * (1 + 1e-12)
        assert doc["regions"][region]["sagitta_max_m"] >= s["sagitta_m"]


def _on_arc(e, v):
    c, (a, b) = np.asarray(e["center"][:2]), e["radii"]
    u = np.asarray(e["major_axis"][:2])
    w = np.array([-u[1], u[0]])
    d = np.asarray(v) - c
    return abs((d @ u / a) ** 2 + (d @ w / b) ** 2 - 1.0) <= 1e-12


def test_circular_arcs_use_the_fewest_equal_angle_chords():
    """A circular arc (also one typed "ellipse" with equal radii) is cut into N equal angles, N the fewest with a
    sagitta R (1 - cos(dtheta / 2N)) within the tolerance; the reported sagitta is that value."""
    doc, fx = doc_of("discretized"), fx_of("discretized")
    for region, tag in (("pole", fx.arcs["fillet"]), ("lens", fx.arcs["lens"])):
        e = next(x for x in fx.table()["entities"] if x["tag"] == tag)
        R, c = e["radii"][0], np.asarray(e["center"][:2])
        tol = min(doc["regions"][r]["chord_tol_m"] for r in e["side"] if r != "exterior")
        sweep = _curve(fx, tag).sweep
        N = math.ceil(sweep / (2 * math.acos(1 - tol / R)))
        poly = np.asarray(only(doc, region)["params"]["rz_m"])
        on = [p for p in poly if _on_arc(e, p)]
        assert len(on) == N + 1, (region, len(on), N)
        ang = np.sort(np.unwrap(np.sort(np.arctan2(*(np.asarray(on) - c)[:, ::-1].T))))
        steps = np.diff(ang)
        assert np.allclose(steps, sweep / N, rtol=1e-9, atol=0), region
        assert only(doc, region)["sagitta_m"] == pytest.approx(R * (1 - math.cos(sweep / (2 * N))), rel=1e-9)


def test_neighbours_share_the_vertices_of_a_shared_arc():
    """The fillet is discretized once: the pole and the gap use the same points, bit for bit."""
    doc, fx = doc_of("discretized"), fx_of("discretized")
    e = next(x for x in fx.table()["entities"] if x["tag"] == fx.arcs["fillet"])
    pa = {tuple(p) for p in only(doc, "pole")["params"]["rz_m"] if _on_arc(e, p)}
    pb = {tuple(p) for p in only(doc, "gap")["params"]["rz_m"] if _on_arc(e, p)}
    assert len(pa) > 10 and pa == pb


@pytest.mark.parametrize("name", list(FIXTURES))
def test_clockwise_arcs_give_the_same_solids(S, name):
    """The partition with every curve run the other way (``ccw`` false on every arc, sides swapped, the loops' signs
    flipped) is the same geometry, so it gives the same solids, tree, names and report.

    Tolerances: parameters of native solids are copied from centres, radii and vertices, which the flip does not
    change: bit for bit. A discretized vertex is c + a cos t u + b sin t v with t now counted from the other end;
    t differs by a few units in the last place of numbers up to 2 pi (about 4e-15 rad), which moves the point by
    at most about 5e-16 m at a <= 0.12 m, plus the rounding of the sum (about 1e-16 m): 1e-15 x the diagonal (0.87
    to 1.46 m) holds it. Volumes, masses and sagittas: 1e-12 relative, as for the exact solids above (the arc
    integrals are rounded differently; a sign error in them is an O(1) change)."""
    fx = fx_of(name)
    fl = ref.flipped(fx)
    t = fl.table()
    arcs = [e for e in t["entities"] if e["type"] != "line"]
    assert arcs and all(e["ccw"] is False for e in arcs)      # the case itself: every arc turns clockwise
    a = doc_of(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", S.SolidsWarning)
        b = S.from_partition(*fl.args())
    tol = 1e-15 * t["diagonal_m"]
    key = lambda s: (s["name"], s["kind"], s["mother"], s["region"], s["material"], s["discretized"])
    assert [key(s) for s in b["solids"]] == [key(s) for s in a["solids"]]
    ends = {tuple(e[k][:2]) for e in t["entities"] for k in ("start", "end")}
    for sa, sb in zip(a["solids"], b["solids"]):
        assert sb["rotation_deg"] == sa["rotation_deg"]
        assert sb["sagitta_m"] == pytest.approx(sa["sagitta_m"], rel=1e-12, abs=0.0), sa["name"]
        if sa["kind"] != "genericPolycone":
            assert sb["params"] == sa["params"] and sb["position_m"] == sa["position_m"], sa["name"]
            continue
        assert {k: v for k, v in sb["params"].items() if k != "rz_m"} == \
            {k: v for k, v in sa["params"].items() if k != "rz_m"}
        pa, pb = np.asarray(sa["params"]["rz_m"]), np.asarray(sb["params"]["rz_m"])
        assert len(pb) == len(pa) and ref.polygon_signed_area(pb) > 0, sa["name"]
        d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
        assert d.min(axis=1).max() <= tol and d.min(axis=0).max() <= tol, (sa["name"], d.min(axis=1).max())
        assert {tuple(p) for p in pa if tuple(p) in ends} == {tuple(p) for p in pb if tuple(p) in ends}
    assert list(b["regions"]) == list(a["regions"]) and b["warnings"] == a["warnings"]
    for r, ra in a["regions"].items():
        rb = b["regions"][r]
        assert (rb["solids"], rb["material"], rb["chord_tol_m"]) == (ra["solids"], ra["material"], ra["chord_tol_m"])
        for k in ("volume_m3", "volume_solids_m3", "mass_kg", "mass_solids_kg"):
            assert rel(rb[k], ra[k]) <= 1e-12, (r, k, rb[k], ra[k])
        for k in ("volume_rel_diff", "mass_rel_diff"):
            assert abs(rb[k] - ra[k]) <= 1e-12, (r, k)
        assert rb["sagitta_max_m"] == pytest.approx(ra["sagitta_max_m"], rel=1e-12, abs=0.0)
        assert rel(rb["volume_m3"], fx.exact[r]) <= 1e-11, r              # and against the independent reference


# ============================================================================================ units
def test_lengths_are_metres_and_a_factor_1000_shows(S, tmp_path):
    """solids.json is in metres (and degrees, kg/m3); the GDML file's values, taken with their unit attribute as
    Geant4 takes them, are the same lengths in Geant4's millimetres."""
    doc = doc_of("native")
    p = only(doc, "insert")["params"]
    assert (p["rmin_m"], p["rmax_m"]) == (0.17, 0.41)
    assert doc["regions"]["insert"]["volume_m3"] == pytest.approx(math.pi * (0.41 ** 2 - 0.17 ** 2) * 0.18
                                                                  - math.pi * (0.31 ** 2 - 0.23 ** 2) * 0.07,
                                                                  rel=1e-12)
    root, _ = gdml_of(S, doc, tmp_path)
    sol = {el.get("name"): el for el in root.find("solids")}
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    el = sol[vols[only(doc, "insert")["gdml_name"]].find("solidref").get("ref")]
    assert float(el.get("rmax")) * ref.G4_LENGTH[el.get("lunit")] == pytest.approx(410.0, rel=1e-15)   # mm
    assert float(el.get("z")) * ref.G4_LENGTH[el.get("lunit")] == pytest.approx(180.0, rel=1e-14)
    mat = {m.get("name"): m for m in root.find("materials").findall("material")}
    D = mat[material_gdml(doc, "steel")].find("D")
    assert D.get("unit") == "kg/m3" and float(D.get("value")) == 7930.0


# ============================================================================================ the GDML writer
def test_gdml_structure_mirrors_the_document(S, tmp_path):
    for name in FIXTURES:
        doc = doc_of(name)
        root, path = gdml_of(S, doc, tmp_path, f"{name}.gdml")
        assert root.tag == "gdml"
        assert "gdml.xsd" in root.get(f"{XSI}noNamespaceSchemaLocation", "")
        assert [c.tag for c in root] == ["define", "materials", "solids", "structure", "setup"]
        setup = root.find("setup")
        world = by_name(doc)[doc["world"]]
        assert setup.find("world").get("ref") == world["gdml_name"]
        solids = {el.get("name"): el for el in root.find("solids")}
        assert len(solids) == len(root.find("solids"))
        vols = root.find("structure").findall("volume")
        assert sorted(v.get("name") for v in vols) == sorted(s["gdml_name"] for s in doc["solids"])
        defined, kids = set(), children(doc)
        mats = {m.get("name") for m in root.find("materials").findall("material")}
        b, g = by_name(doc), by_gdml(doc)
        for v in vols:
            s = g[v.get("name")]
            assert v.find("solidref").get("ref") in solids
            mref = v.find("materialref").get("ref")
            # the document's material (a material of the file); NIST only for the vacuum of the world and the voids
            assert mref == material_gdml(doc, s["material"]), v.get("name")
            assert mref in mats or (mref == "G4_Galactic" and s["region"] is None)
            dau = [pv.find("volumeref").get("ref") for pv in v.findall("physvol")]
            assert all(d in defined for d in dau), f"a daughter of {v.get('name')} is defined after it"
            assert sorted(dau) == sorted(b[k]["gdml_name"] for k in kids[s["name"]])
            defined.add(v.get("name"))
        assert vols[-1].get("name") == world["gdml_name"]
        names = [el.get("name") for el in root.iter() if el.get("name") is not None and el.tag != "setup"]
        assert len(names) == len(set(names)), "GDML names are unique across the file"
        assert all(NCNAME.fullmatch(n) and "0x" not in n for n in names), names
        # a name that GDML can hold and that collides with nothing is written unchanged
        assert all(s["gdml_name"] == s["name"] for s in doc["solids"])
        assert all(m["gdml_name"] == m["name"] for m in doc["materials"])


def _check_number(text, what):
    """Rule 1: every number is written with 17 significant digits ('%.17g'), which round-trips a double."""
    assert text == "%.17g" % float(text), f"{what}: '{text}' is not written as %.17g"
    return float(text)


def _length(el, attr, unit_attr, metres, what):
    x = _check_number(el.get(attr), what)
    assert x * ref.G4_LENGTH[el.get(unit_attr)] == metres * 1000.0, f"{what}: {x} {el.get(unit_attr)} != {metres} m"


def _angle(el, attr, degrees, what):
    x = _check_number(el.get(attr), what)
    assert x * ref.G4_ANGLE[el.get("aunit")] == degrees * ref.DEG, what


@pytest.mark.parametrize("name", list(FIXTURES))
def test_gdml_rule_one_every_value_is_the_documents_double(S, tmp_path, name):
    """Each length and angle reaches Geant4 as the same double as on the native path (the document's value times
    CLHEP's m or deg): written with 17 significant digits, in a unit Geant4 multiplies exactly."""
    doc = doc_of(name)
    root, _ = gdml_of(S, doc, tmp_path)
    solids = {el.get("name"): el for el in root.find("solids")}
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    for s in doc["solids"]:
        el = solids[vols[s["gdml_name"]].find("solidref").get("ref")]
        assert el.tag == s["kind"], s["name"]
        numeric = {k for k in el.attrib if k not in ("name", "lunit", "aunit")}
        want = set()
        for k, v in s["params"].items():
            if k == "rz_m":
                pts = el.findall("rzpoint")
                assert len(pts) == len(v)
                for i, (pt, (r, z)) in enumerate(zip(pts, v)):
                    for attr, val in (("r", r), ("z", z)):
                        x = _check_number(pt.get(attr), f"{s['name']} rz[{i}]")
                        assert x * ref.G4_LENGTH[el.get("lunit")] == val * 1000.0, (s["name"], i, attr)
            elif k.endswith("_m"):
                want.add(k[:-2])
                _length(el, k[:-2], "lunit", v, f"{s['name']}.{k}")
            elif k.endswith("_deg"):
                want.add(k[:-4])
                _angle(el, k[:-4], v, f"{s['name']}.{k}")
        assert numeric == want, (s["name"], numeric, want)
        if s["mother"] is not None:
            mother = by_name(doc)[s["mother"]]["gdml_name"]
            pv = [p for p in vols[mother].findall("physvol") if p.find("volumeref").get("ref") == s["gdml_name"]]
            assert len(pv) == 1
            pos = pv[0].find("position")
            for axis, val in zip("xyz", s["position_m"]):
                _length(pos, axis, "unit", val, f"{s['name']} position {axis}")
    for m in doc["materials"]:
        el = next(x for x in root.find("materials").findall("material") if x.get("name") == m["gdml_name"])
        D = el.find("D")
        assert D.get("unit") == "kg/m3" and _check_number(D.get("value"), m["name"]) == m["density_kg_m3"]


def test_gdml_rule_two_nist_materials_are_referenced_never_defined(S, tmp_path):
    """A NIST material (and every element) is resolved by Geant4's NIST manager from its name: the file defines no
    element, no isotope and no material with a NIST name. A document material of NIST composition is a material of
    the file, named as in the document, with the document's density and one fraction of the NIST material."""
    for name in FIXTURES:
        doc = doc_of(name)
        root, _ = gdml_of(S, doc, tmp_path, f"{name}.gdml")
        mats = root.find("materials")
        assert mats.findall("element") == [] and mats.findall("isotope") == []
        defined = {m.get("name"): m for m in mats.findall("material")}
        assert not [n for n in defined if n.startswith("G4_")]
        assert set(defined) == {m["gdml_name"] for m in doc["materials"]}
        for m in doc["materials"]:
            el = defined[m["gdml_name"]]
            comp = m["composition"]
            if "nist" in comp:
                fr = el.findall("fraction")
                assert len(fr) == 1 and fr[0].get("ref") == comp["nist"] and float(fr[0].get("n")) == 1.0
                assert el.findall("composite") == []
        wname = by_name(doc)[doc["world"]]["gdml_name"]
        world = next(v for v in root.find("structure").findall("volume") if v.get("name") == wname)
        assert world.find("materialref").get("ref") == "G4_Galactic"


def test_gdml_compositions_by_mass_by_atoms_and_optional_properties(S, tmp_path):
    doc = doc_of("native")
    root, _ = gdml_of(S, doc, tmp_path)
    byg = {x.get("name"): x for x in root.find("materials").findall("material")}
    m = {k: byg[material_gdml(doc, k)] for k in ("steel", "polyethylene", "air")}
    fr = {f.get("ref"): _check_number(f.get("n"), "steel fraction") for f in m["steel"].findall("fraction")}
    assert fr == {"Cr": 0.19, "Fe": 0.70, "Ni": 0.11}
    comp = {c.get("ref"): c.get("n") for c in m["polyethylene"].findall("composite")}
    assert comp == {"C": "1", "H": "2"}
    mee = m["polyethylene"].find("MEE")
    assert mee.get("unit") == "eV" and float(mee.get("value")) == 57.4
    assert m["air"].get("state") == "gas" and m["steel"].get("state") is None
    order = [c.tag for c in m["polyethylene"]]
    assert order.index("MEE") < order.index("D") < order.index("composite")      # T, P, MEE, D, components
    doc = doc_of("discretized")
    root, _ = gdml_of(S, doc, tmp_path, "d2.gdml")
    he = next(x for x in root.find("materials").findall("material") if x.get("name") == material_gdml(doc, "helium"))
    assert he.get("state") == "gas"
    assert he.find("T").get("unit") == "K" and float(he.find("T").get("value")) == 273.15
    assert he.find("P").get("unit") in ("pascal", "Pa") and float(he.find("P").get("value")) == 101325.0
    assert [c.tag for c in he][:2] == ["T", "P"]


def test_gdml_rule_three_rotations_follow_the_document(S, tmp_path):
    """A placement's rotation is written as its document angles, x then y then z (Geant4 applies rotateX, rotateY,
    rotateZ in that order); in an axisymmetric document every rotation is the identity."""
    for name in FIXTURES:
        doc = doc_of(name)
        root, _ = gdml_of(S, doc, tmp_path, f"{name}.gdml")
        b = by_gdml(doc)
        assert all(s["rotation_deg"] == [0.0, 0.0, 0.0] for s in doc["solids"])
        for v in root.find("structure").findall("volume"):
            for pv in v.findall("physvol"):
                s = b[pv.find("volumeref").get("ref")]
                assert pv.find("rotationref") is None
                rot = pv.find("rotation")
                if rot is not None:
                    assert list(k for k in rot.attrib if k in "xyz") == ["x", "y", "z"]
                    for axis, val in zip("xyz", s["rotation_deg"]):
                        x = _check_number(rot.get(axis), f"{s['name']} rotation")
                        assert x * ref.G4_ANGLE[rot.get("unit")] == val * ref.DEG


@pytest.mark.parametrize("name", list(FIXTURES))
def test_gdml_validates_against_the_gdml_schema_when_one_is_given(S, tmp_path, name):
    """The schema is not part of this repository: set AXMP_GDML_SCHEMA to a gdml.xsd (the Geant4 source tree has
    one in source/persistency/gdml/schema/) and have xmllint on PATH; otherwise this test skips."""
    xsd = os.environ.get("AXMP_GDML_SCHEMA")
    if not xsd:
        pytest.skip("AXMP_GDML_SCHEMA is not set")
    import shutil
    if shutil.which("xmllint") is None:
        pytest.skip("xmllint is not on PATH")
    _, path = gdml_of(S, doc_of(name), tmp_path)
    r = subprocess.run(["xmllint", "--noout", "--schema", xsd, str(path)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]


# ============================================================================================ determinism
_RUNNER = r"""
import json, sys
from axionmultiphysics import solids
args = json.load(open(sys.argv[1]))
doc = solids.from_partition(*args)
solids.write(doc, sys.argv[2])
solids.write_gdml(doc, sys.argv[3])
"""


@pytest.mark.parametrize("name", ["native", "voids"])
def test_the_same_input_gives_identical_bytes_in_fresh_processes(tmp_path, name):
    """Two interpreters with different string hashing, working directories and output paths write byte-identical
    solids.json and GDML."""
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(list(fx_of(name).args())))
    out = []
    for seed in ("1", "2"):
        d = tmp_path / f"run{seed}"
        d.mkdir()
        home = os.path.dirname(os.path.dirname(os.path.abspath(axmp.__file__)))   # the package this session imports
        env = dict(os.environ, PYTHONHASHSEED=seed,
                   PYTHONPATH=os.pathsep.join([home] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        r = subprocess.run([sys.executable, "-c", _RUNNER, str(inp), str(d / "s.json"), str(d / "g.gdml")],
                           cwd=d, env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        out.append(((d / "s.json").read_bytes(), (d / "g.gdml").read_bytes()))
    assert out[0][0] == out[1][0]
    assert out[0][1] == out[1][1]


def _relabelled(table, seed):
    """The same partition as another run of the mesher could write it: entities and faces in another order, other
    tags, loops started at another entity, other region measures, and keys the conversion does not read."""
    t = copy.deepcopy(table)
    rng = random.Random(seed)
    tags = [e["tag"] for e in t["entities"]]
    new = rng.sample(range(500, 500 + 3 * len(tags)), len(tags))
    m = dict(zip(tags, new))
    for e in t["entities"]:
        e["tag"] = m[e["tag"]]
        e.pop("atom", None)
    rng.shuffle(t["entities"])
    for f in t["faces"]:
        f["tag"] = rng.randrange(10_000, 20_000)
        loops = []
        for loop in f["loops"]:
            loop = [m[abs(s)] * (1 if s > 0 else -1) for s in loop]
            k = rng.randrange(len(loop))
            loops.append(loop[k:] + loop[:k])
        f["loops"] = loops[:1] + rng.sample(loops[1:], len(loops) - 1)
    rng.shuffle(t["faces"])
    for r in t["regions"].values():
        r["area_m2"] *= 1.5
        r["volume_m3"] = 123.0
    t["seconds"] = 99.0
    t["boundaries"] = {"inner": {"attributes": [2001]}}
    t["nodes"] = [[0.0, 0.0]]
    return t


@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_output_does_not_depend_on_order_tags_or_mesher_measures(S, tmp_path, name):
    fx = fx_of(name)
    ref_json, ref_gdml = tmp_path / "a.json", tmp_path / "a.gdml"
    doc = doc_of(name)
    S.write(doc, ref_json)
    S.write_gdml(doc, ref_gdml)
    table, mats, rm = fx.args()
    other = S.from_partition(_relabelled(table, 11), dict(reversed(list(mats.items()))), dict(reversed(list(rm.items()))))
    S.write(other, tmp_path / "b.json")
    S.write_gdml(other, tmp_path / "b.gdml")
    assert (tmp_path / "b.json").read_bytes() == ref_json.read_bytes()
    assert (tmp_path / "b.gdml").read_bytes() == ref_gdml.read_bytes()


def _plain_sum(table, region):
    """The region's volume as a left-to-right sum of its edges' frusta, in the order the table lists them."""
    ents = {e["tag"]: e for e in table["entities"]}
    total = 0.0
    for f in table["faces"]:
        if f["region"] == region:
            for loop in f["loops"]:
                for s in loop:
                    e = ents[abs(s)]
                    (r0, z0), (r1, z1) = (e["start"][:2], e["end"][:2]) if s > 0 else (e["end"][:2], e["start"][:2])
                    total += math.pi / 3.0 * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1)
    return total


def test_region_sums_do_not_depend_on_the_input_order(S, tmp_path):
    """Sums over faces, loops and edges are exactly rounded or taken in an order fixed by the geometry: on a
    partition whose sums round differently in different orders, eight other listings of it give the same bytes.
    The exact references are exactly rounded sums (1e-12: each term carries a few roundings, and the terms are at
    most about ten times the region's volume)."""
    fx = ref.irregular()
    table, mats, rm = fx.args()
    base = S.from_partition(table, mats, rm)
    for region, v in fx.exact.items():
        assert rel(base["regions"][region]["volume_m3"], v) <= 1e-12, region
        assert rel(base["regions"][region]["volume_solids_m3"], v) <= 1e-12, region
    S.write(base, tmp_path / "a.json")
    want = (tmp_path / "a.json").read_bytes()
    plain = {_plain_sum(table, "shard")}
    for seed in range(1, 9):
        other = _relabelled(fx.table(), seed)
        plain.add(_plain_sum(other, "shard"))
        S.write(S.from_partition(other, mats, rm), tmp_path / "b.json")
        assert (tmp_path / "b.json").read_bytes() == want, seed
    assert len(plain) > 1, "the case itself: a plain sum in these orders is not always rounded the same"


RENAMED = {
    # fixtures with regions and materials named as a document may name them: spaces, a leading digit, "0x" (where
    # Geant4's reader cuts a name), the generated names world and void_<k>, names that a mapping of the characters
    # GDML refuses may merge ("a b", "a_b", "a-b"), and materials named like the names the file derives from a
    # volume (<volume>_solid, _pv, _pos), also from a volume whose own name is mapped ("a b_0" -> "a_b_0", say)
    "native": ("native", {"air": "world", "core": "316L steel", "ring": "a0xb", "insert": "1st", "pin": "void",
                          "cone": "a b"},
               {"air": "a_b", "iron": "world", "copper": "a b", "steel": "0x1", "polyethylene": "316L steel"}),
    "voids": ("voids", {"jacket": "void", "liner": "world", "shell": "void_1"},
              {"steel": "world_0", "ptfe": "a-b", "lead": "a b"}),
    # the setup element's name is an ID of the file too
    "default": ("native", {"cone": "Default"}, {"polyethylene": "Default", "iron": "Default_1"}),
    "suffixes": ("native", {"cone": "a b", "core": "c d"},
                 {"polyethylene": "a_b_0_pv", "iron": "c_d_0_solid", "copper": "world_solid", "steel": "insert_0_pos",
                  "air": "a_b_0_pos"}),
}


def renamed_fixture(name):
    base, regions, materials = RENAMED[name]
    return ref.renamed(fx_of(base), regions, materials), regions, base


@pytest.mark.parametrize("name", list(RENAMED))
def test_document_names_are_kept_and_gdml_names_are_mapped(S, tmp_path, name):
    """Names are mapped, not refused: solids.json keeps every document name (regions, materials, the solids'
    region and material) and gives each solid and material a ``gdml_name``. The GDML names are xs:IDs (ASCII
    letters, digits, '_', '.', '-', not starting with a digit), never contain "0x", are unique across the file, and
    are what the file uses; the geometry is the same as with plain names, and the bytes are reproducible."""
    fx, regions, base = renamed_fixture(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", S.SolidsWarning)
        doc = S.from_partition(*fx.args())
    # the document's names
    assert list(doc["regions"]) == fx.regions
    used = {fx.region_materials[r] for r in fx.regions}
    assert {m["name"] for m in doc["materials"]} == used
    for s in doc["solids"]:
        if s["region"] is not None:
            assert s["material"] == fx.region_materials[s["region"]]
    names = [s["name"] for s in doc["solids"]]
    assert len(set(names)) == len(names), "solids.json names are unique"
    world = by_name(doc)[doc["world"]]
    assert world["region"] is None and world["mother"] is None
    # the same geometry as with plain names
    orig = doc_of(base)
    for old, new in ((k, regions.get(k, k)) for k in orig["regions"]):
        for k in ("volume_m3", "volume_solids_m3", "mass_kg"):
            assert doc["regions"][new][k] == orig["regions"][old][k], (new, k)
        assert len(doc["regions"][new]["solids"]) == len(orig["regions"][old]["solids"])
    # the GDML names
    gnames = [s["gdml_name"] for s in doc["solids"]] + [m["gdml_name"] for m in doc["materials"]]
    assert all(NCNAME.fullmatch(n) and "0x" not in n for n in gnames), gnames
    assert "Default" not in gnames, "the setup's name is taken"
    root, path = gdml_of(S, doc, tmp_path)
    every = [el.get("name") for el in root.iter() if el.get("name") is not None and el.tag != "setup"]
    assert all(NCNAME.fullmatch(n) and "0x" not in n for n in every), every
    assert len(every) == len(set(every)), "GDML names are unique across the file"
    vols = root.find("structure").findall("volume")
    assert sorted(v.get("name") for v in vols) == sorted(s["gdml_name"] for s in doc["solids"])
    assert sorted(m.get("name") for m in root.find("materials").findall("material")) == \
        sorted(m["gdml_name"] for m in doc["materials"])
    g = by_gdml(doc)
    for v in vols:
        assert v.find("materialref").get("ref") == material_gdml(doc, g[v.get("name")]["material"])
    assert root.find("setup").find("world").get("ref") == world["gdml_name"]
    # reproducible, also from another listing of the partition
    S.write(doc, tmp_path / "a.json")
    other = S.from_partition(_relabelled(fx.table(), 5), *fx.args()[1:])
    S.write(other, tmp_path / "b.json")
    S.write_gdml(other, tmp_path / "b.gdml")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert path.read_bytes() == (tmp_path / "b.gdml").read_bytes()
    xsd = os.environ.get("AXMP_GDML_SCHEMA")
    import shutil
    if xsd and shutil.which("xmllint"):
        r = subprocess.run(["xmllint", "--noout", "--schema", xsd, str(path)], capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 0, r.stderr[-3000:]


def test_faces_that_tie_on_box_and_area_are_named_by_their_geometry(S, tmp_path):
    """A square cut along its diagonal into two faces of one region: same box, same area. Listing the two faces
    in the other order gives the same document and file."""
    for k, rev in enumerate((False, True)):
        doc = S.from_partition(*ref.tie(rev).args())
        assert rel(doc["regions"]["t"]["volume_m3"], ref.tie(rev).exact["t"]) <= 1e-12
        S.write(doc, tmp_path / f"{k}.json")
        S.write_gdml(doc, tmp_path / f"{k}.gdml")
    assert (tmp_path / "0.json").read_bytes() == (tmp_path / "1.json").read_bytes()
    assert (tmp_path / "0.gdml").read_bytes() == (tmp_path / "1.gdml").read_bytes()


# ============================================================================================ documentation
def test_the_guide_states_when_solids_cannot_overlap():
    """Solids do not overlap only while no two curves come closer than the chord tolerance: the guide qualifies
    every "by construction" in the same sentence, and says that a thinner layer is refused with
    particle.partition_invalid and resolved by a lower chord_tol_m."""
    text = " ".join((ROOT / "docs" / "guides" / "solids.md").read_text().split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for x in sentences:
        if "by construction" in x:
            assert re.search(r"chord|thinner|unless|provided", x), x
    assert any("particle.partition_invalid" in x and "chord_tol_m" in x for x in sentences)


def test_the_guide_documents_the_format_the_rules_and_every_code():
    guide = ROOT / "docs" / "guides" / "solids.md"
    assert guide.is_file(), "docs/guides/solids.md is the user page of the solids list and its GDML export"
    text = guide.read_text()
    assert "guides/solids.md" in (ROOT / "mkdocs.yml").read_text()
    for word in ("axionmultiphysics.solids/1", "from_partition", "write_gdml", "17 significant digits", "NIST",
                 "chord_tol", "Pappus", "gdml_name", *CODES):
        assert word in text, word
    assert re.search(r"\bbeta\b", text, re.I)
    assert "axionmultiphysics.solids" in (ROOT / "llms.txt").read_text()
    # the notes since 0.1.0: "Unreleased" and the release sections above 0.1.0
    notes = (ROOT / "CHANGELOG.md").read_text().split("## Unreleased", 1)[1].split("\n## 0.1.0", 1)[0]
    assert "solids" in notes and "GDML" in notes
