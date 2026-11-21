"""Field maps for particle transport (beta): per-region patches of the recovered static field with chord-RMS error
control, in the directory format ``axionmultiphysics.fieldmap/1``.

``Results.export_field_map`` (and ``Results3D.export_field_map``) build a map from a solved design: one patch per
region (adjacent regions with the same constitutive law and no curve between them share one; a closed curve splits a
region into parts), each a regular grid holding the region's recovered field B* (B_h averaged at the nodes of a
continuous space of the solution's order over the patch's own elements), refined until the RMS error of the map's
integrals along fixed chords meets ``tol``. ``write_map`` writes a map from nodal values given by the caller (an
analytic field); ``load`` reads either; ``FieldMap.evaluate`` interpolates it (bilinear in (r, z) for kind 1,
trilinear for kind 3). The numerics live in the C++ library (``_core.FieldMapper``); this module checks the options,
runs the refinement loop and writes the directory. docs/guides/field_maps.md describes the format, the statistic and
every error code.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
import warnings as _warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import _core

SCHEMA = "axionmultiphysics.fieldmap/1"

# Constants of the format: the manifest records them, and none of them is an option.
CHORD_SEED = 0x5DEECE66D       # seed of the chord sequences (scrambled Halton); not the run's seed
HALTON_BASES = (2, 3, 5, 7, 11, 13)
N_BULK = 16384                 # chords per family
N_LAYER = 4096
N_AXIS = 4096
N_TRACK = 4096
CHORD_LENGTH_H_EL = 8.0        # nominal chord length in units of the patch's median element size
FLOOR = 0.1                    # denominator floor, as a fraction of L_c * B_patch
TOL_EFF_FACTOR = 0.1           # tol_eff = max(tol, TOL_EFF_FACTOR * H)
GUARD_TWO_HALVINGS = 3.0       # two consecutive halvings must lower E by at least this factor
FIELD_FREE_RATIO = 1e-10       # B_patch <= this * max B_patch over the map: the patch is written as zero
MIN_NODES = 9                  # at least this many nodes along each axis of a patch
POINTWISE_POINTS = 100000

_GOLDEN = 0x9E3779B97F4A7C15
_M64 = (1 << 64) - 1
_MB = float(2 ** 20)
_POINTWISE_FAMILY = 255
_BOUNDARY_WARN = 1e-3
_CHORD_FILES = ("start", "direction", "segments", "segment_chord", "length", "I_star", "I_abs", "I_h", "I_map")
_NAME = re.compile(r"^[A-Za-z0-9_]+$")


class FieldMapWarning(UserWarning):
    """A map was written but one of its patches is not accepted (``fieldmap.not_converging`` or
    ``fieldmap.cap_reached``): its error statistic does not meet the tolerance. The message starts with the code and
    names the patch; the manifest's ``warnings`` and the patch's ``accepted`` and ``stop_reason`` say the same."""


def _error(code: str, message: str, hint: str) -> Exception:
    exc = _core.AxionMultiPhysicsError(f"{message}. Hint: {hint}")
    exc.code = f"fieldmap.{code}"
    exc.hint = hint
    return exc


def _invalid(message: str, hint: str) -> Exception:
    return _error("invalid_option", message, hint)


# ============================================================================================ sequences and the rule
def halton(seed: int, n: int, dims: int = 6):
    """Points 1..n of the scrambled-Halton sequence of the format, shape (n, dims), float64 in [0, 1)."""
    seed, n, dims = int(seed), int(n), int(dims)
    if not 0 <= seed <= _M64 or n < 0 or not 1 <= dims <= 6:
        raise _invalid(f"halton: seed {seed}, n {n}, dims {dims}", "seed in [0, 2^64), n >= 0, dims 1 to 6")
    return np.asarray(_core.fieldmap_halton(seed, n, dims))


def family_seed(seed: int, index: int) -> int:
    """Seed of chord family ``index`` (bulk 0, layer 1, axis 2, track volume k: 3 + k):
    ``(seed ^ (index * 0x9E3779B97F4A7C15)) mod 2**64``."""
    return (int(seed) ^ ((int(index) * _GOLDEN) & _M64)) & _M64


def decide(E: Mapping[str, Sequence[float]], tol_eff: Mapping[str, float]) -> dict:
    """The refinement decision after the last level in ``E`` (``E[family][level]``, levels 0..L): returns
    ``{"status": "accepted" | "not_converging" | "refine", "level": int | None, "family": str | None,
    "gain": float | None}``.

    Accepted when every family is within its ``tol_eff``. Otherwise, from L >= 2, not converging when a family still
    above its ``tol_eff`` gained less than ``GUARD_TWO_HALVINGS`` over the last two halvings, E(L-2) / E(L) (the
    smallest such gain; ties to the alphabetically first family; E(L) = 0 is an infinite gain); otherwise refine. A
    family already within its tol_eff never stops the loop."""
    fams = sorted(E)
    if not fams:
        raise ValueError("decide: no family")
    lengths = {len(E[f]) for f in fams}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"decide: every family needs the same number of levels, got {sorted(lengths)}")
    missing = [f for f in fams if f not in tol_eff]
    if missing:
        raise ValueError(f"decide: no tol_eff for {missing}")
    L = lengths.pop() - 1
    if all(E[f][L] <= tol_eff[f] for f in fams):
        return {"status": "accepted", "level": L, "family": None, "gain": None}
    if L >= 2:
        worst = None
        for f in fams:
            if E[f][L] <= tol_eff[f]:
                continue
            gain = math.inf if E[f][L] == 0 else E[f][L - 2] / E[f][L]
            if gain < GUARD_TWO_HALVINGS and (worst is None or gain < worst[1]):
                worst = (f, gain)
        if worst:
            return {"status": "not_converging", "level": L, "family": worst[0], "gain": worst[1]}
    return {"status": "refine", "level": None, "family": None, "gain": None}


# ============================================================================================ reading and evaluating
def _interp(origin, spacing, B, pts):
    """Multilinear interpolation of nodal values B (grid + (3,)) at points (n, d); the cell is clipped to the grid. A
    coordinate within 1e-9 of a spacing of the first or last node plane is taken on that plane, so that a point on a
    face of the box and a point clamped onto it get the same value."""
    d = pts.shape[1]
    idx, frac = [], []
    for a in range(d):
        t = (pts[:, a] - origin[a]) / spacing[a]
        last = B.shape[a] - 1
        t = np.where(t >= last - 1e-9, float(last), np.where(t <= 1e-9, 0.0, t))
        i = np.clip(np.floor(t).astype(np.int64), 0, B.shape[a] - 2)
        idx.append(i)
        frac.append(t - i)
    out = np.zeros((pts.shape[0], 3))
    for corner in range(2 ** d):
        w = np.ones(pts.shape[0])
        sel = []
        for a in range(d):
            bit = (corner >> a) & 1
            w = w * (frac[a] if bit else 1.0 - frac[a])
            sel.append(idx[a] + bit)
        out += w[:, None] * B[tuple(sel)]
    return out


class Patch:
    """One patch of a map: ``name``, ``kind`` (1 or 3), ``regions``, ``origin``, ``spacing``, ``shape`` (nodes per
    axis), ``B`` (the written array, read only), ``record`` (its manifest entry), ``evaluate(points)`` (Cartesian, as
    ``FieldMap.evaluate``) and ``evaluate_native(coords)`` (kind 1: (r, z) -> (B_r, B_z, B_phi); kind 3: the patch
    frame's (x, y, z) -> (B_x, B_y, B_z))."""

    def __init__(self, owner: "FieldMap", record: dict, B: np.ndarray):
        self._owner = owner
        self.record = record
        self.name = record["name"]
        self.kind = int(record["kind"])
        self.regions = list(record.get("regions", []))
        self.origin = [float(x) for x in record["origin"]]
        self.spacing = [float(x) for x in record["spacing"]]
        self.shape = tuple(int(n) for n in record["shape"])
        self.margin = [float(x) for x in record.get("margin", self.spacing)]
        B.flags.writeable = False
        self.B = B
        T = np.asarray(record["frame"].get("transform", np.eye(4)), dtype=float) if self.kind == 3 else np.eye(4)
        self._R, self._t = T[:3, :3], T[:3, 3]

    def _clamp(self, q):
        """Points outside the box by at most ``margin`` along every axis, clamped onto the box (a stepper's trial
        points just outside a volume whose faces lie on its patch's box), and whether each point is served."""
        ok = np.ones(len(q), bool)
        c = q.copy()
        for a in range(q.shape[1]):
            lo = self.origin[a]
            hi = self.origin[a] + (self.shape[a] - 1) * self.spacing[a]
            out = np.maximum(np.maximum(lo - q[:, a], q[:, a] - hi), 0.0)
            ok &= out <= self.margin[a] * (1 + 1e-12)
            c[:, a] = np.clip(q[:, a], lo, hi)
        return c, ok

    def _native(self, q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        q0 = q
        d = 2 if self.kind == 1 else 3
        if q.shape[1] != d:
            raise _invalid(f"patch '{self.name}': native coordinates must have shape (n, {d})",
                           "kind 1 takes (r, z), kind 3 the patch frame's (x, y, z)")
        q, ok = self._clamp(q)
        out = np.zeros((len(q), 3))
        if not np.all(ok):
            if self._owner.outside == "refuse":
                bad = q0[~ok][0]
                raise _error("outside_map",
                             f"point {bad.tolist()} lies outside the box of patch '{self.name}' by more than its "
                             f"margin {self.margin} (outside: refuse)",
                             "Extend the map (export the region that holds the point, or a larger track volume), or "
                             "declare outside=\"zero\" when the field there is negligible")
            self._owner.outside_calls += int(np.count_nonzero(~ok))
        if np.any(ok):
            out[ok] = _interp(self.origin, self.spacing, self.B, q[ok])
        return out

    def evaluate_native(self, coords):
        return self._native(coords)

    def evaluate(self, points):
        p = np.atleast_2d(np.asarray(points, dtype=float))
        if p.shape[1] != 3:
            raise _invalid("points must have shape (n, 3) (Cartesian, m)", "pass x, y, z per row")
        if self.kind == 1:
            r = np.hypot(p[:, 0], p[:, 1])
            phi = np.arctan2(p[:, 1], p[:, 0])
            b = self._native(np.column_stack([r, p[:, 2]]))
            c, s = np.cos(phi), np.sin(phi)
            return np.column_stack([b[:, 0] * c - b[:, 2] * s, b[:, 0] * s + b[:, 2] * c, b[:, 1]])
        local = (p - self._t) @ self._R
        return self._native(local) @ self._R.T


class FieldMap:
    """A field map read from a directory (``load``): ``path``, ``manifest`` (the parsed ``manifest.json``),
    ``patches`` (list of ``Patch`` in manifest order), ``patch(name)``, ``evaluate(points, patch=None)`` and
    ``outside_calls`` (points refused as zero under ``outside: "zero"``)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        mf = self.path / "manifest.json"
        if not mf.is_file():
            raise _invalid(f"no field map at {self.path} (manifest.json missing)", "give the directory a map was written to")
        self.manifest = json.loads(mf.read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise _invalid(f"{mf}: schema {self.manifest.get('schema')!r}, not {SCHEMA!r}", "read a map of this format")
        self.outside = self.manifest.get("outside", "refuse")
        self.outside_calls = 0
        self.patches = [Patch(self, rec, np.load(self.path / rec["file"])) for rec in self.manifest["patches"]]
        self._by_name = {p.name: p for p in self.patches}

    def patch(self, name: str) -> Patch:
        if name not in self._by_name:
            raise _invalid(f"no patch named '{name}' in the map (patches: {', '.join(self._by_name)})",
                           "name one of the manifest's patches")
        return self._by_name[name]

    def evaluate(self, points, patch: str | None = None):
        """B [T] at Cartesian points (n, 3) [m], shape (n, 3). ``patch`` names the patch whose grid serves the points
        (required when the map has more than one patch). A point outside that patch's box raises
        ``AxionMultiPhysicsError`` with code ``fieldmap.outside_map`` when the map's ``outside`` is "refuse"; with
        "zero" it gets B = 0 and is counted in ``outside_calls``. The evaluator never decides the region: the transport
        does, by the volume the point is in."""
        if patch is None:
            if len(self.patches) != 1:
                raise _invalid(f"the map has {len(self.patches)} patches: name the one whose grid serves the points",
                               "pass patch=<name>; each patch serves only its own region's volume")
            return self.patches[0].evaluate(points)
        return self.patch(patch).evaluate(points)


def load(path: str | Path) -> FieldMap:
    """Read a map directory written by ``export_field_map`` or ``write_map``."""
    return FieldMap(path)


# ============================================================================================ writing
def _npy_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(arr), version=(1, 0), allow_pickle=False)
    return buf.getvalue()


def _write(path: Path, manifest: dict, files: dict[str, bytes]) -> None:
    """Everything into path.tmp, then renamed to path (an existing path is replaced)."""
    tmp = Path(str(path) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        tmp.mkdir(parents=True)
        for rel, data in files.items():
            f = tmp / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        tmp.rename(path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _frame(kind: int, transform=None) -> dict:
    if kind == 1:
        return {"type": "axisymmetric", "axis": "z"}
    T = np.eye(4) if transform is None else np.asarray(transform, dtype=float)
    return {"type": "cartesian", "transform": T.tolist()}


def _array_record(k: int, B: np.ndarray) -> tuple[dict, bytes]:
    data = _npy_bytes(np.asarray(B, dtype="<f8"))
    return {"dir": f"patch_{k}", "file": f"patch_{k}/B.npy", "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": int(B.size * 8)}, data


def write_map(path: str | Path, patches: Sequence[Mapping[str, Any]], *, outside: str = "refuse") -> FieldMap:
    """Write a map from nodal values: each patch is a dict with ``name``, ``kind`` (1 or 3), ``origin``, ``spacing``,
    ``B`` (kind 1: shape (n_r, n_z, 3) holding (B_r, B_z, B_phi); kind 3: (n_x, n_y, n_z, 3) holding (B_x, B_y, B_z);
    T) and, for kind 3, an optional ``transform`` (4 x 4 placement, global = T @ local). The manifest's ``source`` is
    ``{"type": "nodal_values"}`` and the patches carry no error record."""
    path = Path(path)
    if outside not in ("refuse", "zero"):
        raise _invalid(f"outside={outside!r}", 'outside is "refuse" (the default) or "zero"')
    if not patches:
        raise _invalid("write_map needs at least one patch", "pass a list of patch dicts")
    records, files, names = [], {}, set()
    for k, p in enumerate(sorted(patches, key=lambda q: str(q.get("name")))):
        name = p.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise _invalid(f"patch name {name!r} (missing, empty or repeated)", "give every patch a distinct name")
        names.add(name)
        kind = p.get("kind")
        if kind not in (1, 3):
            raise _invalid(f"patch '{name}': kind {kind!r}", "kind 1 is axisymmetric (r, z), kind 3 Cartesian (x, y, z)")
        d = 2 if kind == 1 else 3
        B = np.asarray(p.get("B"), dtype=float)
        origin = [float(x) for x in np.ravel(p.get("origin", []))]
        spacing = [float(x) for x in np.ravel(p.get("spacing", []))]
        if B.ndim != d + 1 or B.shape[-1] != 3:
            raise _invalid(f"patch '{name}': B has shape {B.shape}, not ({'n_r, n_z' if d == 2 else 'n_x, n_y, n_z'}, 3)",
                           "give B as the grid's nodal values with the three components last")
        if len(origin) != d or len(spacing) != d or not all(math.isfinite(x) for x in origin + spacing):
            raise _invalid(f"patch '{name}': origin and spacing need {d} finite values each", "one per axis")
        if not all(h > 0 for h in spacing):
            raise _invalid(f"patch '{name}': spacing {spacing} must be positive", "node i sits at origin + i * spacing")
        if any(n < 2 for n in B.shape[:-1]):
            raise _invalid(f"patch '{name}': {B.shape[:-1]} nodes; at least 2 per axis", "a grid needs two nodes per axis")
        if not np.all(np.isfinite(B)):
            raise _invalid(f"patch '{name}': B holds non-finite values", "give finite nodal values")
        T = p.get("transform")
        if T is not None:
            T = np.asarray(T, dtype=float)
            if kind != 3 or T.shape != (4, 4) or not np.all(np.isfinite(T)):
                raise _invalid(f"patch '{name}': transform must be a finite 4 x 4 matrix of a kind-3 patch",
                               "global = T @ local; kind 1 is always on the z axis")
            R = T[:3, :3]
            if (np.max(np.abs(R.T @ R - np.eye(3))) > 1e-12 or abs(np.linalg.det(R) - 1.0) > 1e-12
                    or not np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0])):
                raise _invalid(f"patch '{name}': transform is not a placement (a rotation and a translation)",
                               "the 3 x 3 part must be a proper rotation (orthonormal, determinant +1) and the last row "
                               "(0, 0, 0, 1): B is rotated with the patch, so a scale, shear or reflection has no meaning")
        rec, data = _array_record(k, B)
        files[rec["file"]] = data
        records.append({"name": name, **rec, "kind": kind, "frame": _frame(kind, T), "origin": origin,
                        "spacing": spacing, "shape": list(B.shape[:-1]), "margin": list(spacing)})
    manifest = {"schema": SCHEMA, "units": {"length": "m", "B": "T"}, "outside": outside,
                "source": {"type": "nodal_values"}, "patches": records}
    _write(path, manifest, files)
    return load(path)


# ============================================================================================ building from a solve
def _mapper(results):
    """The C++ field-map context of a solved model (cached on the results)."""
    mapper = getattr(results, "_fieldmap_mapper", None)
    if mapper is None:
        mapper = _core.FieldMapper(results._model)
        results._fieldmap_mapper = mapper
        results._fieldmap_patches = {}
    return mapper


def _kind(results) -> int:
    model = getattr(results, "_model", None)
    if isinstance(model, _core.Model3D):
        return 3
    if isinstance(model, _core.Model):
        sym = (getattr(results, "design", None) and results.design.doc.get("symmetry", "axisymmetric")) or "axisymmetric"
        if sym == "axisymmetric":
            return 1
        raise _invalid(f"field maps of '{sym}' results are not implemented", "map an axisymmetric or a 3D solve")
    raise _invalid(f"field maps of {type(results).__name__} are not implemented",
                   "map the static field of an axisymmetric or a 3D solve (Design.solve)")


def _patch_name(regions: Sequence[str], component: int, components: int) -> str:
    name = "+".join(regions)
    return f"{name}#{component}" if components > 1 else name


def _check_options(kind, regions, tol, merge, track_volumes, iterate_on_track_volumes, outside, max_patch_MB,
                   max_total_MB):
    def real(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    if not real(tol) or not 0 < tol < 1:
        raise _invalid(f"tol={tol!r}", "tol is the relative RMS error of the chord integrals, 0 < tol < 1 (default 1e-4)")
    if regions is not None:
        if isinstance(regions, str) or not all(isinstance(r, str) for r in regions) or not list(regions):
            raise _invalid(f"regions={regions!r}", "a non-empty list of region names, or None for every region")
    if not isinstance(merge, bool):
        raise _invalid(f"merge={merge!r}", "True (adjacent regions with one law share a patch) or False")
    if not isinstance(iterate_on_track_volumes, bool):
        raise _invalid(f"iterate_on_track_volumes={iterate_on_track_volumes!r}", "True or False")
    if outside not in ("refuse", "zero"):
        raise _invalid(f"outside={outside!r}", 'outside is "refuse" (the default) or "zero"')
    for nm, v in (("max_patch_MB", max_patch_MB), ("max_total_MB", max_total_MB)):
        if not real(v) or v <= 0:
            raise _invalid(f"{nm}={v!r}", "caps are positive sizes in MB (2^20 bytes)")
    boxes = []
    axes = ("r", "z") if kind == 1 else ("x", "y", "z")
    seen = set()
    for tv in track_volumes or []:
        if not isinstance(tv, Mapping):
            raise _invalid(f"track volume {tv!r} is not a dict", f"each is {{'name', {', '.join(axes)}: [lo, hi]}}")
        name = tv.get("name")
        if not isinstance(name, str) or not _NAME.match(name) or name in seen:
            raise _invalid(f"track volume name {name!r} (letters, digits and _ only, distinct)",
                           "it names the chord family track_<name> and its directory")
        seen.add(name)
        if set(tv) != {"name", *axes}:
            raise _invalid(f"track volume '{name}' has keys {sorted(tv)}; a kind-{kind} map takes name and "
                           f"{', '.join(axes)}", "boxes are in the patch's native coordinates: (r, z) for "
                           "axisymmetric maps, (x, y, z) for 3D maps")
        box = []
        for a in axes:
            v = tv[a]
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(real(x) for x in v) and v[0] < v[1]):
                raise _invalid(f"track volume '{name}': {a} = {v!r}", "[lo, hi] with lo < hi, in m")
            box += [float(v[0]), float(v[1])]
        boxes.append((name, box))
    return boxes


def _level_grid(prec0: dict, level: int):
    shape = [(n - 1) * 2 ** level + 1 for n in prec0["shape0"]]
    spacing = [h / 2 ** level for h in prec0["spacing0"]]
    return prec0["origin"], spacing, shape


def _families(kind, info, boxes, seed, patch_obj):
    """The chord families of a patch: (name, family kind, N, seed, box, acceptance-by-default)."""
    fams = [("bulk", "bulk", N_BULK, family_seed(seed, 0), [])]
    if info["has_jump"]:
        fams.append(("layer", "layer", N_LAYER, family_seed(seed, 1), []))
    if kind == 1 and info["touches_axis"]:
        fams.append(("axis", "axis", N_AXIS, family_seed(seed, 2), []))
    for k, (name, box) in enumerate(boxes):
        if patch_obj.box_meets(box):
            fams.append((f"track_{name}", "track", N_TRACK, family_seed(seed, 3 + k), box))
    return fams


def _make_chords(patch_obj, pname, fam, ftype, n, fseed, box):
    try:
        C = patch_obj.chords(ftype, n, fseed, box)
    except _core.AxionMultiPhysicsError as exc:
        if getattr(exc, "code", "") != "fieldmap.chord_incomplete":
            raise
        raise _error("chord_incomplete", f"patch '{pname}', family {fam}: {str(exc).split(' Hint: ')[0]}",
                     exc.hint) from exc
    if C.n < n:
        if ftype == "track":
            raise _error("track_volume_empty",
                         f"track volume '{fam[len('track_'):]}' meets patch '{pname}' only in a sliver: {C.n} of {n} "
                         f"chords found in {C.attempts} attempts", "enlarge the box or move it into the region")
        raise _core.AxionMultiPhysicsError(f"internal: family {fam} of patch '{pname}' has {C.n} of {n} chords")
    return C


def _round(x):
    return None if x is None or not math.isfinite(x) else float(x)


def export(results, path: str | Path, **options) -> FieldMap:
    """The builder behind ``Results.export_field_map`` (same arguments)."""
    t_start = time.perf_counter()
    path = Path(path)
    opts = dict(regions=None, tol=1e-4, merge=True, track_volumes=None, iterate_on_track_volumes=False,
                outside="refuse", max_patch_MB=512.0, max_total_MB=2048.0)
    unknown = set(options) - set(opts)
    if unknown:
        raise _invalid(f"unknown options {sorted(unknown)}", f"the options are {', '.join(opts)}")
    opts.update(options)
    kind = _kind(results)
    boxes = _check_options(kind, **opts)
    tol, outside = float(opts["tol"]), opts["outside"]
    cap_patch, cap_total = float(opts["max_patch_MB"]) * _MB, float(opts["max_total_MB"]) * _MB
    iterate = opts["iterate_on_track_volumes"]
    mapper = _mapper(results)
    dec = json.loads(mapper.decompose(list(opts["regions"] or []), bool(opts["merge"])))
    timing = {"patches_s": {}}
    # ---- patches: decomposition, recovery, level-0 grids
    t0 = time.perf_counter()
    P = []
    for g in dec["patches"]:
        obj = mapper.patch(g["regions"], g["component"])
        info = json.loads(obj.info())
        name = _patch_name(g["regions"], g["component"], g["components"])
        box = [list(b) for b in info["box"]]
        h_el = info["h_el"]
        shape0 = [max(MIN_NODES, math.ceil((b[1] - b[0]) / h_el) + 1) for b in box]
        spacing0 = [(b[1] - b[0]) / (n - 1) for b, n in zip(box, shape0)]
        P.append({"name": name, "obj": obj, "info": info, "regions": g["regions"],
                  "component": g["component"] if g["components"] > 1 else None, "curves": g["curves"],
                  "box": box, "h_el": h_el, "origin": [b[0] for b in box], "shape0": shape0, "spacing0": spacing0})
    P.sort(key=lambda p: p["name"])
    timing["recovery_s"] = time.perf_counter() - t0
    for name, box in boxes:
        if not any(p["obj"].box_meets(box) for p in P):
            raise _error("track_volume_empty", f"track volume '{name}' meets no mapped patch",
                         "give a box (in the patches' native coordinates) that overlaps a mapped region")
    for p in P:
        nbytes = int(np.prod(p["shape0"])) * 3 * 8
        if nbytes > cap_patch:
            raise _error("patch_too_large",
                         f"patch '{p['name']}': its coarsest grid {p['shape0']} needs {nbytes / _MB:.3g} MB, more than "
                         f"max_patch_MB = {opts['max_patch_MB']}",
                         "raise max_patch_MB, or map fewer regions (its box is set by the region's extent and its "
                         "median element size)")
    Bmax = max(p["obj"].B_patch for p in P)
    warnings = []
    seed = CHORD_SEED
    for p in P:
        tp = time.perf_counter()
        obj = p["obj"]
        p["B_patch"] = obj.B_patch
        p["field_free"] = not (obj.B_patch > FIELD_FREE_RATIO * Bmax)
        origin = p["origin"]
        if p["field_free"]:
            shape, spacing = p["shape0"], p["spacing0"]
            p.update(level=0, B=np.zeros(tuple(shape) + (3,)), shape=shape, spacing=spacing, accepted=True,
                     stop_reason="field_free", families={}, chords={},
                     levels=[{"level": 0, "shape": shape, "spacing": spacing, "E": {}}])
            timing["patches_s"][p["name"]] = time.perf_counter() - tp
            continue
        fams = _families(kind, p["info"], boxes, seed, obj)
        chords = {}
        for fam, ftype, n, fseed, box in fams:
            chords[fam] = _make_chords(obj, p["name"], fam, ftype, n, fseed, box)
        accept = {fam: (not fam.startswith("track_")) or iterate for fam, *_ in fams}
        hist, levels, stats_at, Imap_at = {f: [] for f in chords}, [], {}, {}
        level = 0
        while True:
            o, spacing, shape = _level_grid(p, level)
            B = np.asarray(obj.fill(o, spacing, shape))
            stats, Imap = {}, {}
            for fam, C in chords.items():
                Imap[fam] = np.asarray(C.map_integrals(o, spacing, shape, B))
                stats[fam] = C.statistic(Imap[fam], obj.B_patch)
                hist[fam].append(stats[fam]["E"])
            levels.append({"level": level, "shape": shape, "spacing": spacing,
                           "E": {f: stats[f]["E"] for f in sorted(chords)}})
            tol_eff = {f: max(tol, TOL_EFF_FACTOR * stats[f]["H"]) for f in chords}
            acc = {f: hist[f] for f in chords if accept[f]}
            d = decide(acc, {f: tol_eff[f] for f in acc})
            stop = None
            if d["status"] == "accepted":
                stop = "accepted"
            elif d["status"] == "not_converging":
                stop = "not_converging"
                warnings.append({"code": "fieldmap.not_converging", "patch": p["name"], "family": d["family"],
                                 "message": f"patch '{p['name']}', family {d['family']}: E fell by {d['gain']:.3g} over "
                                            f"the last two halvings (at least {GUARD_TWO_HALVINGS} expected); the map "
                                            f"is written at level {level} and not accepted"})
            else:
                nxt = int(np.prod([2 * n - 1 for n in shape])) * 3 * 8
                if nxt > cap_patch:
                    stop = "cap"
                    warnings.append({"code": "fieldmap.cap_reached", "patch": p["name"],
                                     "message": f"patch '{p['name']}': the next grid needs {nxt / _MB:.3g} MB, more "
                                                f"than max_patch_MB = {opts['max_patch_MB']}; written at level {level} "
                                                f"and not accepted"})
            if stop:
                break
            level += 1
        famrec = {}
        for fam in sorted(chords):
            st, E = stats[fam], hist[fam]
            famrec[fam] = {"n": int(st["n"]), "E": st["E"], "E_q95": st["E_q95"], "E_max": st["E_max"], "H": st["H"],
                           "tol_eff": tol_eff[fam], "floor_active": bool(TOL_EFF_FACTOR * st["H"] > tol),
                           **({"note": f"the map tolerance was raised to a tenth of the FE indicator: tol_eff = "
                                       f"{TOL_EFF_FACTOR} H = {tol_eff[fam]:.3g} > tol = {tol:.3g}; a finer or "
                                       f"higher-order solve lowers H"} if TOL_EFF_FACTOR * st["H"] > tol else {}),
                           "gain_last": _round(E[-2] / E[-1]) if level >= 1 and E[-1] > 0 else None,
                           "gain_two": _round(E[-3] / E[-1]) if level >= 2 and E[-1] > 0 else None,
                           "acceptance": accept[fam]}
        p.update(level=level, B=B, shape=shape, spacing=spacing, accepted=stop == "accepted", stop_reason=stop,
                 families=famrec, chords=chords, Imap=Imap, levels=levels)
        timing["patches_s"][p["name"]] = time.perf_counter() - tp
    # ---- reports at the written level
    t0 = time.perf_counter()
    for p in P:
        p["pointwise"] = None
        if not p["field_free"]:
            p["pointwise"] = json.loads(p["obj"].pointwise(p["origin"], p["spacing"], p["shape"],
                                                           np.ascontiguousarray(p["B"]), POINTWISE_POINTS,
                                                           family_seed(seed, _POINTWISE_FAMILY)))
        p["interfaces"] = []
    for a in P:
        for b in P:
            if a is b:
                continue
            r = json.loads(mapper.interface(a["obj"], a["origin"], a["spacing"], a["shape"], np.ascontiguousarray(a["B"]),
                                            b["obj"], b["origin"], b["spacing"], b["shape"], np.ascontiguousarray(b["B"])))
            if r is not None:
                a["interfaces"].append({"with": b["name"], **r})
    boundary_ratio = None
    if outside == "zero":
        boundary_ratio = mapper.boundary_max([p["obj"] for p in P]) / Bmax if Bmax > 0 else 0.0
        if boundary_ratio > _BOUNDARY_WARN:
            warnings.append({"code": "fieldmap.outside_field_not_small",
                             "message": f"|B| on the mapped domain's outer boundary reaches {boundary_ratio:.3g} of the "
                                        f"largest B_patch: outside=\"zero\" drops a field that is not negligible"})
    timing["reports_s"] = time.perf_counter() - t0
    # ---- the run's cap, then the directory
    total = sum(p["B"].size * 8 for p in P)
    if total > cap_total:
        raise _error("total_cap_exceeded",
                     f"the patches need {total / _MB:.3g} MB together, more than max_total_MB = {opts['max_total_MB']}",
                     "raise max_total_MB, map fewer regions, or relax tol")
    files, records = {}, []
    for k, p in enumerate(P):
        rec, data = _array_record(k, p["B"])
        files[rec["file"]] = data
        for fam, C in p["chords"].items():
            arrays = {"start": C.start, "direction": C.direction, "segments": C.segments,
                      "segment_chord": C.segment_chord, "length": C.length, "I_star": C.I_star, "I_abs": C.I_abs,
                      "I_h": C.I_h, "I_map": p["Imap"][fam]}
            for key in _CHORD_FILES:
                arr = np.asarray(arrays[key])
                arr = arr.astype("<i8") if key == "segment_chord" else arr.astype("<f8")
                files[f"{rec['dir']}/chords/{fam}/{key}.npy"] = _npy_bytes(arr)
        info = p["info"]
        records.append({"name": p["name"], **rec, "kind": kind, "frame": _frame(kind), "regions": p["regions"],
                        "component": p["component"], "curves": p["curves"], "box": p["box"],
                        "touches_axis": bool(info["touches_axis"]), "h_el": p["h_el"], "origin": p["origin"],
                        "spacing": p["spacing"], "shape": p["shape"], "margin": p["spacing0"], "level": p["level"],
                        "levels": p["levels"],
                        "B_patch": p["B_patch"], "field_free": p["field_free"], "accepted": p["accepted"],
                        "stop_reason": p["stop_reason"], "families": p["families"], "pointwise": p["pointwise"],
                        "interfaces": p["interfaces"]})
    info = _build_info()
    threads = info.get("threads")
    build = _without_run_and_paths(info)
    data = getattr(results, "data", {}) or {}
    mesh = data.get("mesh", {}) if isinstance(data, dict) else {}
    doc = results.design.doc
    sol = np.ascontiguousarray(np.asarray(mapper.solution(), dtype="<f8"))
    manifest = {
        "schema": SCHEMA, "units": {"length": "m", "B": "T"}, "tol": tol, "outside": outside,
        **({"boundary_ratio": boundary_ratio} if outside == "zero" else {}),
        "chord_seed": seed,
        "chords": {"bulk": N_BULK, "layer": N_LAYER, "axis": N_AXIS, "track": N_TRACK, "length_h_el": CHORD_LENGTH_H_EL,
                   "floor": FLOOR, "sequence": "scrambled-halton", "bases": list(HALTON_BASES)},
        "control": {"tol_eff_factor": TOL_EFF_FACTOR, "guard_two_halvings": GUARD_TWO_HALVINGS,
                    "field_free_ratio": FIELD_FREE_RATIO, "min_nodes": MIN_NODES,
                    "max_patch_MB": float(opts["max_patch_MB"]), "max_total_MB": float(opts["max_total_MB"]),
                    "iterate_on_track_volumes": iterate, "merge": bool(opts["merge"])},
        "source": {"type": "fe_solution", "symmetry": "axisymmetric" if kind == 1 else "3d",
                   "design": doc.get("name"),
                   "design_sha256": hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"))
                                                   .encode("utf-8")).hexdigest(),
                   "solution_sha256": hashlib.sha256(sol.tobytes()).hexdigest(),
                   "order": mesh.get("order", mapper.order), "elements": mesh.get("elements", mapper.elements),
                   "dofs": mesh.get("dofs", mapper.dofs), "version": _version(), "build_info": build},
        "zero_field_regions": dec["zero_field_regions"],
        "track_volumes": json.loads(json.dumps(list(opts["track_volumes"] or []))),
        "warnings": warnings,
        "patches": records,
    }
    timing["total_s"] = time.perf_counter() - t_start
    timing["threads"] = threads
    manifest["timing"] = timing
    _write(path, manifest, files)
    for p in P:
        if not p["accepted"]:
            code = "fieldmap.cap_reached" if p["stop_reason"] == "cap" else "fieldmap.not_converging"
            _warnings.warn(f"{code}: patch '{p['name']}' of the map {path.name} is not accepted (stop_reason "
                           f"{p['stop_reason']}, level {p['level']}); see the manifest's warnings", FieldMapWarning,
                           stacklevel=3)
    return load(path)


def _build_info() -> dict:
    from . import build_info

    return build_info()


def _without_run_and_paths(info: dict) -> dict:
    """The build information a manifest records: without the thread count (a property of the run, kept under
    ``timing``) and, at every depth, without what names a directory of the machine that built the package
    (``build_dir``, ``git_dirty_files``), so that a map names no local path."""
    return {k: (_without_run_and_paths(v) if isinstance(v, dict) else v) for k, v in info.items()
            if k not in ("threads", "build_dir", "git_dirty_files")}


def _version() -> str:
    from . import __version__

    return __version__


# ============================================================================================ checks against a solve
def _open_map(field_map) -> FieldMap:
    return field_map if isinstance(field_map, FieldMap) else load(field_map)


def _patch_object(results, rec: dict):
    if "regions" not in rec or "box" not in rec:
        raise _invalid(f"patch '{rec.get('name')}' was not built from a solve (write_map)",
                       "recovered_field and chord_error read maps written by export_field_map")
    mapper = _mapper(results)
    key = (tuple(rec["regions"]), rec["component"] or 0)
    cache = results._fieldmap_patches
    if key not in cache:
        cache[key] = mapper.patch(list(rec["regions"]), rec["component"] or 0)
    return cache[key]


def recovered_field(results, field_map: FieldMap | str | Path, patch: str, points):
    """The recovered field B* of ``patch`` at points in the patch's native coordinates ((n, 2) (r, z) for kind 1,
    returning (B_r, B_z, B_phi); (n, 3) for kind 3), from the patch's own elements; NaN where a point lies in none.
    On the axis (kind 1, r = 0) B_r = B_phi = 0."""
    fm = _open_map(field_map)
    rec = fm.patch(patch).record
    obj = _patch_object(results, rec)
    d = 2 if _kind(results) == 1 else 3
    pts = np.ascontiguousarray(np.atleast_2d(np.asarray(points, dtype=float)))
    if pts.shape[1] != d:
        raise _invalid(f"points must have shape (n, {d})", "native coordinates of the patch: (r, z) or (x, y, z)")
    return np.asarray(obj.recovered(pts))


def chord_error(results, field_map: FieldMap | str | Path, *, seed: int | None = None,
                patch: str | None = None) -> dict:
    """E and H of every family of every patch (or of ``patch``), recomputed from the written map read back from disk
    and the results' recovered field, with the chords of ``seed`` (default: the manifest's ``chord_seed``):
    ``{patch: {family: {"n", "E", "E_q95", "E_max", "H"}}}``. Field-free patches are omitted."""
    fm = _open_map(field_map)
    m = fm.manifest
    kind = _kind(results)
    seed = int(m["chord_seed"] if seed is None else seed)
    boxes = []
    for tv in m.get("track_volumes", []):
        axes = ("r", "z") if kind == 1 else ("x", "y", "z")
        boxes.append((tv["name"], [float(x) for a in axes for x in tv[a]]))
    out = {}
    for p in fm.patches:
        rec = p.record
        if (patch is not None and p.name != patch) or rec.get("field_free"):
            continue
        obj = _patch_object(results, rec)
        info = json.loads(obj.info())
        B = np.ascontiguousarray(np.load(fm.path / rec["file"]))
        res = {}
        for fam, ftype, n, fseed, box in _families(kind, info, boxes, seed, obj):
            C = _make_chords(obj, p.name, fam, ftype, n, fseed, box)
            st = C.statistic(np.asarray(C.map_integrals(rec["origin"], rec["spacing"], rec["shape"], B)), obj.B_patch)
            res[fam] = {"n": int(st["n"]), "E": st["E"], "E_q95": st["E_q95"], "E_max": st["E_max"], "H": st["H"]}
        out[p.name] = res
    if patch is not None and patch not in out:
        fm.patch(patch)  # unknown names raise; a field-free patch has no families
    return out
