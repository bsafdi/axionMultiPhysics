"""Validation of a transport run's inputs (sources, outputs, fields, the integrator) against a solids document and the
program's tables, and the parts of the job file they become.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from ._common import (INTEGRATOR, PHYSICS_LISTS, STEPPER, error, real, warn)

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
VARIABLE_KINDS = ("deposit", "hit", "primary")
AGGREGATE_KINDS = ("mean", "std", "quantile", "fraction", "count")
KINDS = VARIABLE_KINDS + AGGREGATE_KINDS + ("histogram", "tally", "hits")
HIT_OF = {"x": "m", "y": "m", "z": "m", "r": "m", "phi": "rad", "t": "s", "E_kin": "eV"}
PRIMARY_OF = {"x": ("m", 0), "y": ("m", 1), "z": ("m", 2), "px": ("eV/c", 3), "py": ("eV/c", 4), "pz": ("eV/c", 5),
              "E_kin": ("eV", 6)}
CONDITIONS = ("gt", "ge", "lt", "le", "defined")
ALIASES = {"muon": ["mu-", "mu+"], "electron": ["e-"], "positron": ["e+"], "pion": ["pi+", "pi-", "pi0"],
           "kaon": ["kaon+", "kaon-", "kaon0L", "kaon0S"], "photon": ["gamma"], "antiproton": ["anti_proton"]}
HINT_OUT = ("docs/guides/particles.md lists the output kinds: deposit, hit, primary (per-event variables); mean, std, "
            "quantile, fraction, count, histogram (aggregates); tally; hits")


def is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def positive_int(x) -> bool:
    return is_int(x) and int(x) > 0


def vector3(x) -> list | None:
    if isinstance(x, (str, bytes)) or not hasattr(x, "__len__") or len(x) != 3:
        return None
    if not all(real(v) for v in x):
        return None
    return [float(v) for v in x]


def as_json(x):
    """A JSON-shaped copy (tuples to lists, numpy scalars to Python numbers)."""
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# ============================================================================================ the source
class AnyParticle(dict):
    """The particle table of an output check after planning: the plan validated the names already."""


def particle_of(name, table: Mapping) -> tuple[str, dict]:
    """(name, entry) of a particle given by name or PDG code; ``particle.unknown_particle`` with candidates."""
    if isinstance(table, AnyParticle):
        return str(name), {"pdg": 0, "mass_eV": 0.0, "charge": 0.0}
    if is_int(name):
        for k in sorted(table):
            if table[k]["pdg"] == int(name):
                return k, table[k]
        raise error("particle.unknown_particle", f"no particle of Geant4's table has the PDG code {int(name)}",
                    "give a particle name of Geant4's table (axmp-transport --check lists it) or its PDG code")
    if isinstance(name, str) and name in table:
        return name, table[name]
    cand = list(ALIASES.get(str(name).lower(), []))
    for c in difflib.get_close_matches(str(name), sorted(table), n=6, cutoff=0.6):
        if c not in cand:
            cand.append(c)
    cand = [c for c in cand if c in table]
    raise error("particle.unknown_particle", f"particle '{name}' is not in Geant4's particle table"
                + (f"; did you mean {', '.join(cand)}" if cand else ""),
                "use a name of Geant4's particle table (axmp-transport --check lists them), e.g. mu-, mu+, e-, "
                "proton, gamma, geantino, or a PDG code")


def source_of(sources, table: Mapping) -> tuple[dict, float, float]:
    """The job's source, the primary's kinetic energy [eV] and its mass [eV]."""
    hint = ("one source: {\"type\": \"gun\", \"particle\": NAME or PDG code, \"energy_eV\" (kinetic) or "
            "\"momentum_eV_c\", \"at_m\": [x, y, z], \"direction\": [dx, dy, dz] or \"isotropic\", \"spot_sigma_m\": "
            "s >= 0}")

    def bad(what):
        return error("particle.source_invalid", f"source: {what}", hint)

    if isinstance(sources, Mapping) or not isinstance(sources, (list, tuple)) or len(sources) != 1:
        n = len(sources) if isinstance(sources, (list, tuple)) else "not a list of"
        raise bad(f"{n} sources given; this version takes exactly one")
    s = sources[0]
    if not isinstance(s, Mapping):
        raise bad("the entry is not an object")
    known = ("type", "particle", "energy_eV", "momentum_eV_c", "at_m", "direction", "spot_sigma_m")
    for k in s:
        if k not in known:
            raise bad(f"unknown key '{k}' (known: {', '.join(known)})")
    if s.get("type") != "gun":
        raise bad(f"type {s.get('type')!r}; this version offers \"gun\"")
    if "particle" not in s:
        raise bad("no 'particle'")
    if ("energy_eV" in s) == ("momentum_eV_c" in s):
        raise bad("give exactly one of energy_eV (kinetic) and momentum_eV_c")
    key = "energy_eV" if "energy_eV" in s else "momentum_eV_c"
    if not real(s[key]) or not s[key] > 0:
        raise bad(f"{key} is {s[key]!r}, not a positive number")
    at = vector3(s.get("at_m"))
    if at is None:
        raise bad("at_m must be three finite numbers (m)")
    d = s.get("direction")
    if d == "isotropic":
        direction = "isotropic"
    else:
        direction = vector3(d)
        if direction is None or not any(v != 0.0 for v in direction):
            raise bad(f"direction {d!r} is neither a non-zero vector nor \"isotropic\"")
    spot = s.get("spot_sigma_m", 0.0)
    if not real(spot) or spot < 0:
        raise bad(f"spot_sigma_m is {spot!r}, not a number >= 0")
    if direction == "isotropic" and spot != 0:
        raise bad("an isotropic source is a point: spot_sigma_m must be 0")
    pname, entry = particle_of(s["particle"], table)
    mass = float(entry["mass_eV"])
    if key == "energy_eV":
        ekin = float(s[key])
    else:
        p = float(s[key])
        ekin = p * p / (math.sqrt(p * p + mass * mass) + mass)
    particle = int(s["particle"]) if is_int(s["particle"]) else pname
    job = {"type": "gun", "particle": particle, key: float(s[key]), "at_m": at, "direction": direction,
           "spot_sigma_m": float(spot)}
    return job, ekin, mass


def _on_or_in_polygon(rz, r, z, tol) -> bool:
    """Whether (r, z) lies in the closed polygon rz (even-odd rule) or within tol of its outline."""
    n = len(rz)
    inside = False
    for i in range(n):
        (r1, z1), (r2, z2) = rz[i], rz[(i + 1) % n]
        dr, dz = r2 - r1, z2 - z1
        L2 = dr * dr + dz * dz
        u = 0.0 if L2 == 0 else min(1.0, max(0.0, ((r - r1) * dr + (z - z1) * dz) / L2))
        if math.hypot(r - (r1 + u * dr), z - (z1 + u * dz)) <= tol:
            return True
        if (z1 > z) != (z2 > z) and r < r1 + (z - z1) * dr / dz:
            inside = not inside
    return inside


def _contains(s: dict, offset, x, y, z, tol) -> bool:
    """Whether the point lies in the solid s (placed at offset), its surface included."""
    p, k = s["params"], s["kind"]
    dx, dy, dz = x - offset[0], y - offset[1], z - offset[2]
    r = math.hypot(dx, dy)
    if p.get("deltaphi_deg", 360.0) < 360.0 and r > tol:
        phi = math.degrees(math.atan2(dy, dx)) % 360.0
        if (phi - p["startphi_deg"]) % 360.0 > p["deltaphi_deg"] + 1e-9:
            return False
    if k == "tube":
        return p["rmin_m"] - tol <= r <= p["rmax_m"] + tol and abs(dz) <= 0.5 * p["z_m"] + tol
    if k == "orb":
        return math.hypot(r, dz) <= p["r_m"] + tol
    if k == "torus":
        return p["rmin_m"] - tol <= math.hypot(r - p["rtor_m"], dz) <= p["rmax_m"] + tol
    return _on_or_in_polygon(p["rz_m"], r, dz, tol)


def region_at(doc: dict, point) -> tuple[bool, str | None]:
    """(inside the world, region) of a point of the solids list: the deepest solid that holds it (its surface
    included) decides; the world and the voids have no region."""
    x, y, z = (float(v) for v in point)
    offsets = solid_offsets(doc)
    kids: dict = {}
    for s in doc["solids"]:
        kids.setdefault(s.get("mother"), []).append(s)
    (world,) = kids[None]
    size = max(1.0, abs(x), abs(y), abs(z))
    tol = 1e-12 * size
    if not _contains(world, offsets[world["name"]], x, y, z, tol):
        return False, None
    if world["kind"] == "tube":
        # the program needs a vertex strictly inside the world: its outer faces are outside
        o, p = offsets[world["name"]], world["params"]
        if math.hypot(x - o[0], y - o[1]) >= p["rmax_m"] - tol or abs(z - o[2]) >= 0.5 * p["z_m"] - tol:
            return False, None
    here = world
    while True:
        nxt = next((c for c in kids.get(here["name"], []) if _contains(c, offsets[c["name"]], x, y, z, tol)), None)
        if nxt is None:
            return True, here.get("region")
        here = nxt


def check_source_position(source: dict, doc: dict) -> None:
    """A gun's position must lie in a region of the solids (``particle.source_invalid``): outside the world the program
    has no volume to start the primary in, and in the world or a void the primary leaves the domain at once."""
    inside, region = region_at(doc, source["at_m"])
    if region is not None:
        return
    where = "outside the world" if not inside else "in the world or a void, outside every region"
    raise error("particle.source_invalid", f"source: at_m {list(source['at_m'])} m lies {where} of the solids",
                "place the gun inside a region of the solids (a point of the transport's domain)")


# ============================================================================================ outputs
class Outputs:
    """The validated outputs: per-event variables, aggregates, histograms, tallies, hits scorers, and the internal
    hits scorers that serve ``hit`` variables (one record per event: the first hit)."""

    def __init__(self):
        self.variables: dict[str, dict] = {}
        self.aggregates: dict[str, dict] = {}
        self.histograms: dict[str, dict] = {}
        self.tallies: dict[str, dict] = {}
        self.hits: dict[str, dict] = {}
        self.scorers: list[dict] = []        # the job's hits entries, user ones first

    def units(self, var: str) -> str:
        return self.variables[var]["units"]


def _regions_of(value, regions, name, what, many=True) -> list[str]:
    if isinstance(value, str):
        value = [value]
    elif not many or not isinstance(value, (list, tuple)) or not value:
        raise error("particle.output_invalid", f"output '{name}': {what} must be a region name"
                    + (" or a non-empty list of them" if many else ""), HINT_OUT)
    for r in value:
        if not isinstance(r, str) or r not in regions:
            raise error("particle.output_invalid", f"output '{name}': {what} names '{r}', which is not a region of the "
                        f"solids (regions: {', '.join(regions)})", "name regions of the solids document")
    return list(value)


def _keys(spec, name, kind, required, optional):
    if not isinstance(spec, Mapping):
        raise error("particle.output_invalid", f"output '{name}': the {kind} spec is not an object", HINT_OUT)
    for k in spec:
        if k not in required + optional:
            raise error("particle.output_invalid", f"output '{name}': unknown key '{k}' of {kind} (known: "
                        f"{', '.join(required + optional)})", HINT_OUT)
    for k in required:
        if k not in spec:
            raise error("particle.output_invalid", f"output '{name}': {kind} needs '{k}'", HINT_OUT)


def check_where(W, name, variables, path="where"):
    bad = lambda what: error("particle.output_invalid", f"output '{name}': {path} {what}",  # noqa: E731
                             "where is {VAR: {\"gt\"|\"ge\"|\"lt\"|\"le\": x, \"defined\": true|false}, ...} or "
                             "{\"any\": [where, ...]}, VAR a per-event variable")
    if not isinstance(W, Mapping):
        raise bad("is not an object")
    if "any" in W:
        if len(W) != 1 or not isinstance(W["any"], (list, tuple)) or not W["any"]:
            raise bad("with 'any' has only that key, a non-empty list")
        for i, w in enumerate(W["any"]):
            check_where(w, name, variables, f"{path}.any[{i}]")
        return
    for var, cond in W.items():
        if var not in variables:
            raise bad(f"names '{var}', which is not a per-event variable of the outputs")
        if not isinstance(cond, Mapping) or not cond:
            raise bad(f"condition on '{var}' is not a non-empty object")
        for op, x in cond.items():
            if op not in CONDITIONS:
                raise bad(f"condition '{op}' on '{var}' is not one of {', '.join(CONDITIONS)}")
            if op == "defined":
                if not isinstance(x, bool):
                    raise bad(f"'defined' on '{var}' must be true or false")
            elif not real(x):
                raise bad(f"'{op}' on '{var}' must be a finite number")


def check_grid(g, name) -> dict:
    hint = ("a grid is {\"type\": \"cylindrical\", \"r_m\": [r0, r1], \"z_m\": [z0, z1], \"bins\": [nr, nz]} or "
            "{\"type\": \"cartesian\", \"x_m\": [..], \"y_m\": [..], \"z_m\": [..], \"bins\": [nx, ny, nz]}")
    if not isinstance(g, Mapping) or g.get("type") not in ("cylindrical", "cartesian"):
        raise error("particle.output_invalid", f"output '{name}': grid type {g.get('type') if isinstance(g, Mapping) else g!r}"
                    " is not cylindrical or cartesian", hint)
    axes = ("r_m", "z_m") if g["type"] == "cylindrical" else ("x_m", "y_m", "z_m")
    for k in g:
        if k not in ("type", "bins") + axes:
            raise error("particle.output_invalid", f"output '{name}': unknown grid key '{k}'", hint)
    out = {"type": g["type"]}
    for a in axes:
        v = g.get(a)
        if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(real(x) for x in v) or not v[0] < v[1]:
            raise error("particle.output_invalid", f"output '{name}': grid {a} must be [lo, hi] with lo < hi", hint)
        if a == "r_m" and v[0] < 0:
            raise error("particle.output_invalid", f"output '{name}': grid r_m starts below 0", hint)
        out[a] = [float(v[0]), float(v[1])]
    b = g.get("bins")
    if not isinstance(b, (list, tuple)) or len(b) != len(axes) or not all(positive_int(x) for x in b):
        raise error("particle.output_invalid", f"output '{name}': grid bins must be {len(axes)} positive integers", hint)
    out["bins"] = [int(x) for x in b]
    return out


def outputs_of(outputs, regions: list[str], table: Mapping) -> Outputs:
    if not isinstance(outputs, Mapping) or not outputs:
        raise error("particle.output_invalid", "outputs must be a non-empty object {name: spec}", HINT_OUT)
    O = Outputs()
    specs = {}
    for name, spec in outputs.items():
        if not isinstance(name, str) or not NAME.match(name):
            raise error("particle.output_invalid", f"output name {name!r} is not [A-Za-z_][A-Za-z0-9_]*", HINT_OUT)
        if not isinstance(spec, Mapping) or len(spec) != 1 or next(iter(spec)) not in KINDS:
            keys = list(spec) if isinstance(spec, Mapping) else spec
            raise error("particle.output_invalid", f"output '{name}': the spec must have exactly one kind key of "
                        f"{', '.join(KINDS)}, not {keys!r}", HINT_OUT)
        kind = next(iter(spec))
        specs[name] = (kind, spec[kind])
    # per-event variables first: the aggregates refer to them
    for name, (kind, s) in specs.items():
        if kind == "deposit":
            _keys(s, name, kind, ["in"], [])
            O.variables[name] = {"kind": kind, "regions": _regions_of(s["in"], regions, name, "'in'"), "units": "eV"}
        elif kind == "hit":
            _keys(s, name, kind, ["in", "of"], ["track"])
            (region,) = _regions_of(s["in"], regions, name, "'in'", many=False)
            if s["of"] not in HIT_OF:
                raise error("particle.output_invalid", f"output '{name}': hit 'of' is {s['of']!r}, not one of "
                            f"{', '.join(HIT_OF)}", HINT_OUT)
            track = s.get("track", "primary")
            if track not in ("primary", "any"):
                raise error("particle.output_invalid", f"output '{name}': track {track!r} is not primary or any",
                            HINT_OUT)
            O.variables[name] = {"kind": kind, "region": region, "of": s["of"], "track": track,
                                 "units": HIT_OF[s["of"]]}
        elif kind == "primary":
            _keys(s, name, kind, ["of"], [])
            if s["of"] not in PRIMARY_OF:
                raise error("particle.output_invalid", f"output '{name}': primary 'of' is {s['of']!r}, not one of "
                            f"{', '.join(PRIMARY_OF)}", HINT_OUT)
            O.variables[name] = {"kind": kind, "of": s["of"], "column": PRIMARY_OF[s["of"]][1],
                                 "units": PRIMARY_OF[s["of"]][0]}
    for name, (kind, s) in specs.items():
        if kind in ("mean", "std", "quantile", "histogram"):
            req = {"mean": ["of"], "std": ["of"], "quantile": ["of", "q"], "histogram": ["of", "bins", "range"]}[kind]
            _keys(s, name, kind, req, ["where"])
            if s["of"] not in O.variables:
                raise error("particle.output_invalid", f"output '{name}': '{s['of']}' is not a per-event variable of "
                            "the outputs", "aggregate a deposit, hit or primary output")
            if "where" in s:
                check_where(s["where"], name, O.variables)
            rec = {"kind": kind, "of": s["of"], "where": s.get("where"), "units": O.units(s["of"])}
            if kind == "quantile":
                if not real(s["q"]) or not 0.0 <= s["q"] <= 1.0:
                    raise error("particle.output_invalid", f"output '{name}': q is {s['q']!r}, not in [0, 1]", HINT_OUT)
                rec["q"] = float(s["q"])
            if kind == "histogram":
                if not positive_int(s["bins"]):
                    raise error("particle.output_invalid", f"output '{name}': bins must be a positive integer",
                                HINT_OUT)
                r = s["range"]
                if not isinstance(r, (list, tuple)) or len(r) != 2 or not all(real(x) for x in r) or not r[0] < r[1]:
                    raise error("particle.output_invalid", f"output '{name}': range must be [lo, hi] with lo < hi",
                                HINT_OUT)
                rec["bins"], rec["range"] = int(s["bins"]), [float(r[0]), float(r[1])]
                O.histograms[name] = rec
            else:
                O.aggregates[name] = rec
        elif kind in ("fraction", "count"):
            _keys(s, name, kind, [], ["where"])
            if "where" in s:
                check_where(s["where"], name, O.variables)
            O.aggregates[name] = {"kind": kind, "where": s.get("where"), "units": "1"}
        elif kind == "tally":
            _keys(s, name, kind, ["grid"], ["in"])
            grid = check_grid(s["grid"], name)
            regs = _regions_of(s["in"], regions, name, "'in'") if "in" in s else list(regions)
            O.tallies[name] = {"grid": grid, "regions": regs}
        elif kind == "hits":
            _keys(s, name, kind, ["in"], ["track", "particles", "max_per_event", "keep_events"])
            regs = _regions_of(s["in"], regions, name, "'in'")
            track = s.get("track", "any")
            if track not in ("primary", "any"):
                raise error("particle.output_invalid", f"output '{name}': track {track!r} is not primary or any",
                            HINT_OUT)
            parts = s.get("particles")
            if parts is not None:
                if not isinstance(parts, (list, tuple)) or not parts:
                    raise error("particle.output_invalid", f"output '{name}': particles must be a non-empty list of "
                                "names or null", HINT_OUT)
                parts = [particle_of(p, table)[0] for p in parts]
            cap = s.get("max_per_event", 100)
            if not positive_int(cap):
                raise error("particle.output_invalid", f"output '{name}': max_per_event must be a positive integer",
                            HINT_OUT)
            keep = s.get("keep_events")
            if keep is not None and not positive_int(keep):
                raise error("particle.output_invalid", f"output '{name}': keep_events must be a positive integer",
                            HINT_OUT)
            O.hits[name] = {"regions": regs, "track": track, "particles": parts, "max_per_event": int(cap),
                            "keep_events": None if keep is None else int(keep)}
            O.scorers.append({"name": name, "regions": regs, "track": track, "particles": parts,
                              "max_per_event": int(cap)})
    # one internal scorer per (region, track) of the hit variables: its first record is the event's first hit
    taken = set(outputs)
    internal: dict[tuple, str] = {}
    for name, v in O.variables.items():
        if v["kind"] != "hit":
            continue
        key = (v["region"], v["track"])
        if key not in internal:
            k, s = len(internal), f"_hit{len(internal)}"
            while s in taken:
                k += 1
                s = f"_hit{k}_"
            taken.add(s)
            internal[key] = s
            O.scorers.append({"name": s, "regions": [v["region"]], "track": v["track"], "particles": None,
                              "max_per_event": 1})
        v["scorer"] = internal[key]
    return O


# ============================================================================================ fields
def read_map_manifest(path: Path) -> dict:
    mf = Path(path) / "manifest.json"
    if not mf.is_file():
        raise error("particle.field_invalid", f"no field map at {path} (manifest.json missing)",
                    "give the directory a map was written to (Results.export_field_map or fieldmap.write_map)")
    try:
        man = json.loads(mf.read_text())
    except ValueError as e:
        raise error("particle.field_invalid", f"{mf} is not JSON ({e})", "give a map directory") from None
    if man.get("schema") != "axionmultiphysics.fieldmap/1":
        raise error("particle.field_invalid", f"{mf}: schema {man.get('schema')!r}, not axionmultiphysics.fieldmap/1",
                    "give a map of this format")
    return man


def manifest_sha256(path) -> str:
    """The SHA-256 of a map's manifest (which holds every patch array's checksum): the map's identity."""
    import hashlib
    return hashlib.sha256((Path(path) / "manifest.json").read_bytes()).hexdigest()


def job_map_path(path, workdir) -> str:
    """How a job file names a map: relative to exchange/ when the map lies inside the workdir (a copied or moved workdir
    then reads its own map), else by its real absolute path (the workdir may be reached by another name)."""
    real = os.path.realpath(path)
    home = os.path.realpath(workdir)
    if os.path.commonpath([real, home]) == home:
        return os.path.relpath(real, os.path.join(home, "exchange"))
    return real


def solid_offsets(doc: dict) -> dict[str, list[float]]:
    """The absolute position of every solid (the sum of the placements along its mothers)."""
    pos: dict[str, list[float]] = {}
    for s in doc["solids"]:
        p = [float(x) for x in s.get("position_m", [0.0, 0.0, 0.0])]
        if s.get("mother") is not None:
            m = pos[s["mother"]]
            p = [m[i] + p[i] for i in range(3)]
        pos[s["name"]] = p
    return pos


def solid_extent(s: dict, offset) -> tuple[float, float, float, float]:
    """(r_lo, r_hi, z_lo, z_hi) of a solid of revolution in absolute coordinates."""
    p, z0 = s["params"], offset[2]
    k = s["kind"]
    if k == "tube":
        return p["rmin_m"], p["rmax_m"], z0 - 0.5 * p["z_m"], z0 + 0.5 * p["z_m"]
    if k == "orb":
        return 0.0, p["r_m"], z0 - p["r_m"], z0 + p["r_m"]
    if k == "torus":
        return p["rtor_m"] - p["rmax_m"], p["rtor_m"] + p["rmax_m"], z0 - p["rmax_m"], z0 + p["rmax_m"]
    rz = np.asarray(p["rz_m"], dtype=float)
    return float(rz[:, 0].min()), float(rz[:, 0].max()), z0 + float(rz[:, 1].min()), z0 + float(rz[:, 1].max())


def patch_box(rec: dict) -> tuple[list, list, list]:
    lo = [float(o) for o in rec["origin"]]
    hi = [lo[a] + (int(rec["shape"][a]) - 1) * float(rec["spacing"][a]) for a in range(len(lo))]
    margin = [float(m) for m in rec.get("margin", rec["spacing"])]
    return lo, hi, margin


def _inside_box(pts, lo, hi, margin) -> bool:
    for x in pts:
        for a in range(len(lo)):
            tol = 1e-12 * max(1.0, abs(lo[a]), abs(hi[a]))
            if x[a] < lo[a] - margin[a] - tol or x[a] > hi[a] + margin[a] + tol:
                return False
    return True


def covers(rec: dict, ext) -> bool:
    """Whether a patch's box enlarged by its margin holds a solid's extent (kind 1: the (r, z) box; kind 3: the eight
    corners of [-r_hi, r_hi]^2 x [z_lo, z_hi] in the patch frame)."""
    lo, hi, margin = patch_box(rec)
    r0, r1, z0, z1 = ext
    if int(rec["kind"]) == 1:
        return _inside_box([(r0, z0), (r1, z1)], lo, hi, margin)
    T = np.asarray(rec.get("frame", {}).get("transform", np.eye(4)), dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    corners = np.array([[sx * r1, sy * r1, z] for sx in (-1, 1) for sy in (-1, 1) for z in (z0, z1)])
    local = (corners - t) @ R
    return _inside_box(local.tolist(), lo, hi, margin)


class Fields:
    def __init__(self):
        self.job: list[dict] = []           # explicit per-patch entries with regions
        self.record: list[dict] = []        # as given, plus each map's tol and its patches' error records
        self.spacing_min: float | None = None
        self.maps: dict[str, dict] = {}     # relative path -> {"boundary_ratio", ...}


def fields_of(fields, doc: dict, regions: list[str], outside: str, exchange: Path, workdir: Path) -> Fields:
    """Validates ``fields`` against the solids; returns the job entries (each region named explicitly) and the record.
    Raises particle.field_invalid, particle.field_ambiguous and, with outside="refuse", particle.field_outside_map."""
    hint = ("a field is {\"type\": \"uniform\", \"B_T\": [bx, by, bz]}, {\"type\": \"current_loop\", \"radius_m\", "
            "\"current_A\", \"z_m\"}, or {\"type\": \"map\", \"path\": dir, \"patch\": name}, each with an optional "
            "\"regions\" list; {\"type\": \"map\", \"path\": dir} attaches every patch of a map")
    F = Fields()
    if fields is None:
        fields = []
    if isinstance(fields, Mapping) or not isinstance(fields, (list, tuple)):
        raise error("particle.field_invalid", "fields must be a list", hint)
    served: dict[str, tuple[int, dict | None, str]] = {}   # region -> (entry index, patch record, patch label)
    field_free: set[str] = set()

    def serve(i, regs, rec, label):
        for r in regs:
            if not isinstance(r, str) or r not in regions:
                raise error("particle.field_invalid", f"fields[{i}]: region '{r}' is not a region of the solids "
                            f"(regions: {', '.join(regions)})", hint)
            if r in served or r in field_free:
                raise error("particle.field_ambiguous", f"region '{r}' is served by two fields (fields["
                            f"{served[r][0] if r in served else i}] and fields[{i}])",
                            "attach exactly one field to each region")
            served[r] = (i, rec, label)

    for i, f in enumerate(fields):
        if not isinstance(f, Mapping):
            raise error("particle.field_invalid", f"fields[{i}] is not an object", hint)
        t = f.get("type")
        allowed = {"uniform": ("type", "B_T", "regions"),
                   "current_loop": ("type", "radius_m", "current_A", "z_m", "regions"),
                   "map": ("type", "path", "patch", "regions")}.get(t)
        if allowed is None:
            raise error("particle.field_invalid", f"fields[{i}]: type {t!r} is not uniform, current_loop or map", hint)
        for k in f:
            if k not in allowed:
                raise error("particle.field_invalid", f"fields[{i}]: unknown key '{k}' of a {t} field", hint)
        regs = f.get("regions")
        if regs is not None and (not isinstance(regs, (list, tuple)) or not regs):
            raise error("particle.field_invalid", f"fields[{i}]: regions must be a non-empty list", hint)
        rec_entry = as_json(dict(f))
        if t == "uniform":
            B = vector3(f.get("B_T"))
            if B is None:
                raise error("particle.field_invalid", f"fields[{i}]: B_T must be three finite numbers (T)", hint)
            regs = list(regs) if regs is not None else list(regions)
            serve(i, regs, None, "")
            F.job.append({"type": "uniform", "B_T": B, "regions": regs})
        elif t == "current_loop":
            a, I, z = f.get("radius_m"), f.get("current_A"), f.get("z_m")
            if not real(a) or a <= 0 or not real(I) or not real(z):
                raise error("particle.field_invalid", f"fields[{i}]: radius_m > 0, current_A and z_m finite numbers",
                            hint)
            regs = list(regs) if regs is not None else list(regions)
            serve(i, regs, None, "")
            F.job.append({"type": "current_loop", "radius_m": float(a), "current_A": float(I), "z_m": float(z),
                          "regions": regs})
        else:
            if not isinstance(f.get("path"), (str, os.PathLike)):
                raise error("particle.field_invalid", f"fields[{i}]: a map needs 'path' (its directory)", hint)
            path = Path(os.path.abspath(os.fspath(f["path"])))
            man = read_map_manifest(path)
            rel_job = job_map_path(path, workdir)
            rel_rec = os.path.relpath(path, workdir)
            rec_entry["path"] = rel_rec
            rec_entry["manifest_sha256"] = manifest_sha256(path)
            first_job_entry = len(F.job)
            patches = {p["name"]: p for p in man.get("patches", [])}
            if "patch" in f:
                if f["patch"] not in patches:
                    raise error("particle.field_invalid", f"fields[{i}]: the map {path.name} has no patch "
                                f"'{f['patch']}' (patches: {', '.join(patches)})", hint)
                use = [(patches[f["patch"]], list(regs) if regs is not None else patches[f["patch"]].get("regions"))]
            else:
                if regs is not None:
                    raise error("particle.field_invalid", f"fields[{i}]: a whole map serves each patch's own regions; "
                                "name a patch to choose regions", hint)
                use = [(p, p.get("regions")) for p in man.get("patches", [])]
                for r in man.get("zero_field_regions", []) or []:
                    if r not in regions:
                        raise error("particle.field_invalid", f"fields[{i}]: the map's zero-field region '{r}' is not "
                                    "a region of the solids", hint)
                    if r in served or r in field_free:
                        raise error("particle.field_ambiguous", f"region '{r}' is served by two fields", hint)
                    field_free.add(r)
            for p, pr in use:
                if not pr:
                    raise error("particle.field_invalid", f"fields[{i}]: patch '{p['name']}' of the map {path.name} "
                                "names no regions (a map written by write_map)",
                                "give \"regions\" with \"patch\" to say which regions it serves")
                serve(i, pr, p, f"patch '{p['name']}' of the map {path.name}")
                F.job.append({"type": "map", "path": rel_job, "patch": p["name"], "regions": list(pr)})
                h = min(float(x) for x in p["spacing"])
                F.spacing_min = h if F.spacing_min is None else min(F.spacing_min, h)
            rec_entry["job_fields"] = list(range(first_job_entry, len(F.job)))
            rec_entry["map"] = {"tol": man.get("tol"), "outside": man.get("outside"),
                                "boundary_ratio": man.get("boundary_ratio"),
                                "patches": {p["name"]: {"accepted": p.get("accepted"), "families": p.get("families")}
                                            for p, _ in use}}
            F.maps[rel_rec] = {"boundary_ratio": man.get("boundary_ratio")}
        F.record.append(rec_entry)
    if outside == "refuse" and F.job:
        offsets = solid_offsets(doc)
        for s in doc["solids"]:
            r = s.get("region")
            if r is None:
                continue
            if r not in served:
                if r in field_free:
                    continue
                raise error("particle.field_outside_map", f"region '{r}' is served by no field while fields are "
                            "given: its tracks would see B = 0 without being told",
                            "attach a field (or a map patch) to it, or pass outside=\"zero\" to make unserved regions "
                            "field-free and count calls outside the maps")
            _, rec, label = served[r]
            if rec is not None and not covers(rec, solid_extent(s, offsets[s["name"]])):
                ext = solid_extent(s, offsets[s["name"]])
                raise error("particle.field_outside_map", f"region '{r}' (solid {s['name']}, r in [{ext[0]:.6g}, "
                            f"{ext[1]:.6g}] m, z in [{ext[2]:.6g}, {ext[3]:.6g}] m) reaches beyond the box of {label} "
                            "plus its margin",
                            "export the map over the whole region (or a larger track volume), or pass "
                            "outside=\"zero\" to take B = 0 there and count the calls")
    return F


# ============================================================================================ the integrator
def integrator_of(integrator, spacing_min, doc: dict) -> dict:
    hint = (f"integrator keys are {', '.join(INTEGRATOR)}: positive numbers (m, or relative for the epsilons), "
            "epsilon_min <= epsilon_max <= 1e-2, delta_intersection_m at most 0.1 of the finest map spacing")
    out = dict(INTEGRATOR)
    if integrator is not None:
        if not isinstance(integrator, Mapping):
            raise error("particle.integrator_invalid", "integrator must be an object", hint)
        for k, v in integrator.items():
            if k not in INTEGRATOR:
                raise error("particle.integrator_invalid", f"unknown integrator key '{k}'", hint)
            if not real(v) or v <= 0:
                raise error("particle.integrator_invalid", f"integrator {k} is {v!r}, not a positive number", hint)
            out[k] = float(v)
    if not out["epsilon_min"] <= out["epsilon_max"] <= 1e-2:
        raise error("particle.integrator_invalid", f"epsilon_min {out['epsilon_min']:g} <= epsilon_max "
                    f"{out['epsilon_max']:g} <= 1e-2 does not hold", hint)
    if spacing_min is not None and out["delta_intersection_m"] > 0.1 * spacing_min * (1 + 1e-12):
        raise error("particle.integrator_invalid", f"delta_intersection_m {out['delta_intersection_m']:g} m exceeds "
                    f"0.1 of the finest spacing of the attached maps ({spacing_min:g} m): the boundary crossings "
                    "would read a patch beyond its continuation", hint)
    thin = [min(s["params"]["rmax_m"] - s["params"]["rmin_m"], s["params"]["z_m"])
            for s in doc["solids"] if s["kind"] == "tube"]
    if thin and out["delta_intersection_m"] > 0.1 * min(thin):
        warn("particle.delta_intersection_not_small", f"delta_intersection_m {out['delta_intersection_m']:g} m exceeds "
             f"0.1 of the thinnest tube of the solids ({min(thin):g} m): boundary crossings in it are located no "
             "better than that", stacklevel=4)
    return out


def physics_list_of(name) -> str:
    if name in PHYSICS_LISTS:
        return name
    if isinstance(name, str) and name.endswith("_HP"):
        raise error("particle.physics_list_unavailable", f"physics list '{name}' (high-precision neutrons) is not "
                    "offered in this version", f"use one of {', '.join(PHYSICS_LISTS)}")
    raise error("particle.physics_list_unknown", f"physics list {name!r} is not offered",
                f"use one of {', '.join(PHYSICS_LISTS)}")


def stepper_record(integ: dict) -> dict:
    return dict(integ, stepper=STEPPER)

