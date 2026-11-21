"""The exactness cases of the engine acceptance tests through the bridge (``im.fluids_mesh()``, ``im.chem_mesh()``): the
case files, the tolerance rule, the flux instrument and the measurement of each case. Used by
``tests/test_fluids_model_bridge.py``, ``tests/test_chem_model_bridge.py`` and ``scripts/model_bridge_tolerances.py``.
Not a test file: pytest collects nothing here.

A case file, ``cpp/tests/data/model/u2/cases/<name>.json``, is written before the first run. It holds the model document
(a copy, so that a record stays tied to the document it was measured on), the instance, the mesh it gets, the solver
options with the linear and Newton tolerances written out, the data of the closed form and, per asserted quantity,
``{"what", "ceiling", "measured", "asserted"}``:

- ``ceiling`` is the design note's (section 18 head): 1e-8 for a field of a direct Stokes solve, 1e-9 for a derived
  flux or balance and for "to round-off";
- ``measured`` maps "<processor model> / <build>" to the value measured there; ``asserted`` is
  ``max(10 x the largest measured value, 1e-12)``. Both are written by ``scripts/model_bridge_tolerances.py --record``
  and by nothing else. The floor 1e-12 reads the rule for cases at round-off, where ten times one
  platform's noise does not bound another's; platforms with no record rely on it.

A test fails when a quantity exceeds ``asserted``, when ``asserted`` is null (no record yet) or above the ceiling.
``recorded`` holds values the same script writes that no test asserts (a bitwise observation, the decay of the stress
form's outlet disturbance far from the outlet).

The flux instrument (the fluids API has no flux accessor; the section integrator comes later): face by face, the field
at the face's quadrature points from ``im.face_points`` (``sol.velocity([element], ref)``, the same element numbering,
since the engine's mesh is the instance mesh copied and one process keeps the element order), ``u . n`` times the
weights; in an axisymmetric case times ``2 pi r``, here, since the velocity is the field and carries no engine weight.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "cpp" / "tests" / "data" / "model" / "u2" / "cases"
FLOOR = 1e-12
FACTOR = 10.0
# the seven sample points of a triangle, in reference coordinates: the vertices, the edge midpoints and the centroid
REF7 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1.0 / 3.0, 1.0 / 3.0]])
# every case, the build it needs ("mpi": the fluids module; "any": every build) and the test that asserts it
CASE_NAMES = {
    "heat_slab": "mpi", "two_regions_enclosed": "mpi", "two_regions_open": "mpi", "number_is_a_label": "mpi",
    "two_layer": "mpi", "pipe_axi": "mpi", "poiseuille_channel": "mpi", "plate_cell": "any",
}


# ------------------------------------------------------------------------------------------------- case files
def case_path(name: str, cases: Path | None = None) -> Path:
    return Path(cases or CASES) / f"{name}.json"


def load(name: str, cases: Path | None = None) -> dict:
    return json.loads(case_path(name, cases).read_text("utf-8"))


def dumps(obj, indent: int = 0, width: int = 118, lead: int = 0) -> str:
    """JSON with every list or object that fits on its line inline (the layout of the case files); ``lead`` is the
    width of what precedes the value on its line (a key)."""
    flat = json.dumps(obj, separators=(", ", ": "))
    if not isinstance(obj, (dict, list)) or not obj or indent + lead + len(flat) <= width:
        return flat
    pad = " " * (indent + 2)
    if isinstance(obj, dict):
        items = [f"{pad}{json.dumps(k)}: {dumps(v, indent + 2, width, len(json.dumps(k)) + 2)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + dumps(v, indent + 2, width) for v in obj]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def save(case: dict, path: Path) -> None:
    Path(path).write_text(dumps(case) + "\n", "utf-8")


def processor_model() -> str:
    """The processor model of this machine (Linux: /proc/cpuinfo; macOS: sysctl), else the platform's word for it."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return " ".join(line.split(":", 1)[1].split())
    except OSError:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def build_kind() -> str:
    import axionmultiphysics as axmp

    return "mpi" if axmp.build_info().get("mpi") else "sequential"


def record_key() -> str:
    return f"{processor_model()} / {build_kind()}"


def asserted_value(measured: dict) -> float:
    """The tolerance rule: ten times the largest measured value, at least the floor."""
    return max(FACTOR * max(float(v) for v in measured.values()), FLOOR)


def check_tolerances(case: dict, quantities: dict) -> None:
    """Every asserted quantity of the case within its tolerance; a missing record, or one above its ceiling, fails."""
    errors = []
    for q, t in case["tolerance"].items():
        if q not in quantities:
            errors.append(f"{q}: not measured")
            continue
        a = t["asserted"]
        if a is None:
            errors.append(f"{q} = {quantities[q]:.3e}: no tolerance record (run scripts/model_bridge_tolerances.py "
                          "--record on every processor model the gate's jobs can land on)")
            continue
        if a > t["ceiling"]:
            errors.append(f"{q}: the asserted tolerance {a:.3e} is above the ceiling {t['ceiling']:.0e}; investigate "
                          "the cause instead of raising the tolerance")
        if not quantities[q] <= a:     # a NaN fails too
            errors.append(f"{q} = {quantities[q]:.3e} > asserted {a:.3e} (measured {t['measured']})")
    assert not errors, f"case {case['name']} ({record_key()}):\n  " + "\n  ".join(errors)


# --------------------------------------------------------------------------------------------- meshes, sampling
def meshed(case: dict):
    import axionmultiphysics as axmp

    return axmp.mesh(case["document"])


def instance(case: dict, m):
    return m.instance(case["instance"])


def fluids():
    """The fluids package, its beta warning silenced (the tests are about the bridge, not the label)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from axionmultiphysics import fluids as f
    return f


def chem():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from axionmultiphysics import chem as c
    return c


def elements(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int32)


def sample_flow(sol, n: int):
    """(x (n,7,2), u (n,7,vdim), p (n,7)) at REF7 of every element."""
    e = elements(n)
    u, _ = sol.velocity(e, REF7)
    x, _ = sol.geometry(e, REF7)
    return x, u, sol.pressure(e, REF7)


def _per_face(fp):
    for i, e in enumerate(fp["elements"]):
        yield i, np.array([e], dtype=np.int32), np.ascontiguousarray(fp["ref"][i])


def face_points(im, name: str | None = None, attribute: int | None = None, order: int = 4) -> dict:
    return im.face_points(name, order=order) if attribute is None else im.face_points(attribute=attribute, order=order)


def volume_flux(sol, im, name: str | None = None, *, attribute: int | None = None, axisymmetric: bool = False) -> float:
    """The flux of the velocity through the faces of a boundary or a section (its normals: outward on the instance
    boundary, the selection's orientation on a section); axisymmetric: the 3D flux, 2 pi r applied here."""
    fp = face_points(im, name, attribute)
    total = 0.0
    for i, e, ref in _per_face(fp):
        u, _ = sol.velocity(e, ref)
        un = np.sum(u[0][:, :2] * fp["normals"][i], axis=1)
        if axisymmetric:
            un = un * 2.0 * math.pi * fp["points"][i][:, 0]
        total += float(np.sum(fp["weights"][i] * un))
    return total


def heat_flux(sol, im, name: str, k: float) -> float:
    """The heat flux -k grad T . n through the faces of a boundary or a section (planar)."""
    fp = face_points(im, name)
    total = 0.0
    for i, e, ref in _per_face(fp):
        _, g = sol.scalar(e, ref)
        total += float(np.sum(fp["weights"][i] * np.sum(-k * g[0] * fp["normals"][i], axis=1)))
    return total


def max_face_point_offset(sol, im, name: str) -> float:
    """The largest distance between the face points of ``name`` and the solution's geometry at the same element and
    reference points: zero when the engine's mesh is the instance mesh with its element order."""
    fp = face_points(im, name)
    worst = 0.0
    for i, e, ref in _per_face(fp):
        x, _ = sol.geometry(e, ref)
        worst = max(worst, float(np.abs(x[0] - fp["points"][i]).max()))
    return worst


def flow_mesh(im):
    """The bridge's fluids mesh, after the Taylor-Hood check: a triangle with three boundary vertices would make the
    case test something else (the mesher does not guarantee it at an acute corner)."""
    fm = im.fluids_mesh()
    th = fm.taylor_hood_check()
    assert list(th["all_boundary_vertices"]) == [], f"triangles with three boundary vertices: {th['all_boundary_vertices']}"
    return fm


def copy_mesh(fm, boundary_attributes, element_attributes):
    """``fluids.Mesh.from_arrays`` of the arrays of a fluids mesh, with other attributes (its nodes kept)."""
    f = fluids()
    seg, _ = fm.boundary()
    return f.Mesh.from_arrays(np.ascontiguousarray(fm.vertices(), dtype=float),
                              np.ascontiguousarray(fm.elements(), dtype=np.int32),
                              np.ascontiguousarray(seg, dtype=np.int32),
                              np.ascontiguousarray(boundary_attributes, dtype=np.int32),
                              element_attributes=[int(a) for a in element_attributes],
                              p2_nodes=np.ascontiguousarray(fm.p2_nodes()) if fm.has_nodes else None)


def bc(attributes, kind: str, **kw) -> list[dict]:
    """One flow condition per attribute (the fluids API takes "attribute")."""
    return [dict(attribute=int(a), kind=kind, **kw) for a in attributes]


# --------------------------------------------------------------------------------------------- the cases
def measure_heat_slab(case: dict) -> dict:
    """``solve_heat`` on the instance mesh of the split rectangle: a fixed temperature on ``hot`` (x = 0), ``robin``
    on ``cold`` (x = L), ``zero`` on the remaining attributes (the unnamed top and bottom), conditions on the named
    and remaining attributes only. The slab: T = T1 - q x / k, q = (T1 - T_ext) / (L/k + 1/h), linear, exact in P2."""
    f, D, S = fluids(), case["data"], case["solver"]
    k, h, T1, Te, L, H = D["k"], D["h"], D["T1"], D["T_ext"], D["L"], D["H"]
    m = meshed(case)
    im = instance(case, m)
    hot, cold, rem = im.attributes["hot"], im.attributes["cold"], im.remaining_attributes
    assert rem, "the rectangle has an unnamed boundary, covered by the remaining attributes"
    fm = im.fluids_mesh()
    bcs = [dict(attributes=hot, kind="dirichlet", value=T1), dict(attributes=cold, kind="robin", h=h, T_ext=Te),
           dict(attributes=rem, kind="zero")]
    # the engine's own check first (every attribute 1..max has a condition), then the same fact from the maps
    sol = f.solve_heat(fm, k, bcs=bcs, order=S["order"], rtol=S["rtol"], maxit=S["maxit"])
    assert sol.diagnostics["converged"], sol.diagnostics
    n = len(im.master_attribute)
    assert sorted(hot + cold + rem) == list(range(1, n + 1)), (
        f"hot {hot}, cold {cold} and remaining {rem} cover 1..{n} once: the condition set needs no padding")
    q = (T1 - Te) / (L / k + 1.0 / h)
    e = elements(fm.n_elements)
    T, g = sol.scalar(e, REF7)
    x, _ = sol.geometry(e, REF7)
    quantities = {
        "T": float(np.abs(T - (T1 - q * x[..., 0] / k)).max() / (T1 - Te)),
        "heat_flux_field": float(max(np.abs(-k * g[..., 0] - q).max(), np.abs(k * g[..., 1]).max()) / q),
        "Q_cold": abs(heat_flux(sol, im, "cold", k) - q * H) / (q * H),
        "Q_interface": abs(heat_flux(sol, im, "interface", k) - q * H) / (q * H),
    }
    return dict(quantities=quantities, recorded={}, m=m, im=im, fm=fm, sol=sol)


def measure_two_regions_enclosed(case: dict) -> dict:
    """(i) Enclosed flow in both regions of the split rectangle: no-slip on ``hot`` and ``cold``, a lid velocity
    4 U x (L - x) / L^2 on the top (the remaining attributes, which also hold the bottom: the lid is a function of
    position) and zero on the bottom. Compared with the same mesh given one element attribute
    (``fluids.Mesh.from_arrays``)."""
    f, D, S = fluids(), case["data"], case["solver"]
    nu, U, L, H = D["nu"], D["U"], D["L"], D["H"]
    m = meshed(case)
    im = instance(case, m)
    fm = flow_mesh(im)

    def lid(x, y):
        return (U * 4.0 * x * (L - x) / L ** 2 if y > 0.5 * H else 0.0, 0.0)

    bcs = bc(im.attributes["hot"] + im.attributes["cold"], "dirichlet", value=(0.0, 0.0)) + \
        bc(im.remaining_attributes, "dirichlet", value=lid)
    opts = dict(form=S["form"], direct_tol=S["direct_tol"], lu_threads=1)
    sol = f.solve_stokes(fm, nu, bcs=bcs, **opts)
    one = copy_mesh(fm, fm.boundary()[1], [1] * fm.n_elements)
    sol1 = f.solve_stokes(one, nu, bcs=bcs, **opts)
    x, u, p = sample_flow(sol, fm.n_elements)
    x1, u1, p1 = sample_flow(sol1, one.n_elements)
    assert np.array_equal(x, x1), "the copy with one element attribute has the same elements and points"
    quantities = {"u_one_attribute": float(np.abs(u - u1).max() / U),
                  "p_one_attribute": float(np.abs(p - p1).max() / np.abs(p).max())}
    return dict(quantities=quantities, recorded={}, m=m, im=im, fm=fm, sol=sol, sol1=sol1)


def _plane_poiseuille(case: dict, im, fm, form: str):
    f, D, S = fluids(), case["data"], case["solver"]
    nu, U, H = D["nu"], D["U_mean"], D["H"]
    inlet, outlet, walls = (im.attributes[D[k]] if D[k] != "remaining" else im.remaining_attributes
                            for k in ("inlet", "outlet", "walls"))

    def profile(x, y):
        return (6.0 * U * y * (H - y) / H ** 2, 0.0)

    bcs = bc(inlet, "dirichlet", value=profile) + bc(outlet, "do_nothing") + bc(walls, "dirichlet", value=(0.0, 0.0))
    return f.solve_stokes(fm, nu, bcs=bcs, form=form, direct_tol=S["direct_tol"], lu_threads=1), bcs


def _poiseuille_errors(case: dict, sol, n: int) -> dict:
    D = case["data"]
    nu, U, L, H = D["nu"], D["U_mean"], D["L"], D["H"]
    x, u, p = sample_flow(sol, n)
    uref = 6.0 * U * x[..., 1] * (H - x[..., 1]) / H ** 2
    dp = 12.0 * nu * U * L / H ** 2
    return {"u": float(max(np.abs(u[..., 0] - uref).max(), np.abs(u[..., 1]).max()) / (1.5 * U)),
            "p": float(np.abs(p - dp * (1.0 - x[..., 0] / L)).max() / dp)}


def measure_two_regions_open(case: dict) -> dict:
    """(ii) Inlet on ``hot`` (x = 0, region a), outlet on ``cold`` (x = L, region b), no-slip on the remaining
    attributes, Laplacian form, ``do_nothing``: plane Poiseuille flow, u = 6 U y (H - y) / H^2, p = 12 nu U (L - x)
    / H^2, exact in P2/P1. Fluxes by the instrument: through ``interface`` (the section, +e_x from a to b) +U H, through
    ``hot`` (outward -e_x) -U H."""
    D = case["data"]
    U, H = D["U_mean"], D["H"]
    m = meshed(case)
    im = instance(case, m)
    fm = flow_mesh(im)
    sol, bcs = _plane_poiseuille(case, im, fm, case["solver"]["form"])
    q_if = volume_flux(sol, im, "interface")
    q_in = volume_flux(sol, im, D["inlet"])
    quantities = dict(_poiseuille_errors(case, sol, fm.n_elements))
    quantities.update({"Q_interface": abs(q_if - U * H) / (U * H), "Q_inlet": abs(q_in + U * H) / (U * H),
                       "Q_interface_plus_inlet": abs(q_if + q_in) / (U * H)})
    return dict(quantities=quantities, recorded={}, m=m, im=im, fm=fm, sol=sol, bcs=bcs)


def measure_number_is_a_label(case: dict) -> dict:
    """Case (ii) solved on the instance mesh and again on a copy of its arrays (``fluids.Mesh.from_arrays``) whose
    boundary attributes are the master numbers put back through ``im.master_attribute``, with the conditions keyed
    by those numbers (the flow engines accept holes in the numbering)."""
    f = fluids()
    run = measure_two_regions_open(case)
    im, fm, sol, bcs = run["im"], run["fm"], run["sol"], run["bcs"]
    master = im.master_attribute
    _, battr = fm.boundary()
    copy = copy_mesh(fm, [master[int(a)] for a in battr], im.element_attributes())
    assert sorted(set(copy.boundary()[1].tolist())) == sorted(master.values()), "the copy carries the master numbers"
    mbcs = [dict(b, attribute=master[b["attribute"]]) for b in bcs]
    S = case["solver"]
    sol2 = f.solve_stokes(copy, case["data"]["nu"], bcs=mbcs, form=S["form"], direct_tol=S["direct_tol"], lu_threads=1)
    x, u, p = sample_flow(sol, fm.n_elements)
    x2, u2, p2 = sample_flow(sol2, copy.n_elements)
    assert np.array_equal(x, x2), "the copy has the same elements and points"
    U, D = case["data"]["U_mean"], case["data"]
    dp = 12.0 * D["nu"] * U * D["L"] / D["H"] ** 2
    quantities = {"u": float(np.abs(u - u2).max() / (1.5 * U)), "p": float(np.abs(p - p2).max() / dp)}
    recorded = {"bitwise": bool(np.array_equal(u, u2) and np.array_equal(p, p2))}
    return dict(quantities=quantities, recorded=recorded, im=im, fm=fm, sol=sol, sol2=sol2, copy=copy, bcs=mbcs)


def two_layer_profile(D: dict):
    """The two-layer Poiseuille profile u(y) under the pressure gradient G = -dp/dx: nu_i u_i'' = -G in each layer,
    u(0) = u(H) = 0, u and nu du/dy continuous at y = h."""
    G, H, h, n1, n2 = D["G"], D["H"], D["h"], D["nu_lower"], D["nu_upper"]
    C = G * (h ** 2 / n1 - (h ** 2 - H ** 2) / n2) / (2.0 * (h / n1 + (H - h) / n2))

    def u(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= h, -G / (2 * n1) * y ** 2 + C / n1 * y, -G / (2 * n2) * (y ** 2 - H ** 2) + C / n2 * (y - H))
    return u


def measure_two_layer(case: dict) -> dict:
    """(iii) The channel split along the flow at y = h = H/2, viscosities nu_lower and nu_upper = 4 nu_lower given as
    a Python callable of position, the two-layer inflow profile as a callable on ``inlet``, ``do_nothing`` on
    ``outlet``, no-slip on ``walls``, Laplacian form: the two-layer Poiseuille flow, quadratic in each layer, and
    p = G (L - x), exact in P2/P1 since the interface is a mesh line."""
    f, D, S = fluids(), case["data"], case["solver"]
    L, h, G = D["L"], D["h"], D["G"]
    m = meshed(case)
    im = instance(case, m)
    fm = flow_mesh(im)
    prof = two_layer_profile(D)

    def nu(x, y):
        return D["nu_lower"] if y < h else D["nu_upper"]

    def inflow(x, y):
        return (float(prof(y)), 0.0)

    bcs = bc(im.attributes["inlet"], "dirichlet", value=inflow) + bc(im.attributes["outlet"], "do_nothing") + \
        bc(im.attributes["walls"], "dirichlet", value=(0.0, 0.0))
    sol = f.solve_stokes(fm, nu, bcs=bcs, form=S["form"], direct_tol=S["direct_tol"], lu_threads=1)
    x, u, p = sample_flow(sol, fm.n_elements)
    umax = float(prof(np.linspace(0.0, D["H"], 4001)).max())
    quantities = {"u": float(max(np.abs(u[..., 0] - prof(x[..., 1])).max(), np.abs(u[..., 1]).max()) / umax),
                  "p": float(np.abs(p - G * (L - x[..., 0])).max() / (G * L))}
    return dict(quantities=quantities, recorded={}, m=m, im=im, fm=fm, sol=sol)


def pipe_conditions(im, p_in: float, sign: float = 1.0) -> list[dict]:
    """One condition per attribute of the pipe's instance mesh: the axis, no-slip on the wall, the traction
    (0, sign p_in, 0) on the inlet (sigma_form n with n = -e_z: a pressure p_in), do_nothing on the outlet."""
    return bc(im.axis_attributes, "axis") + bc(im.attributes["wall"], "dirichlet", value=(0.0, 0.0, 0.0)) + \
        bc(im.attributes["inlet"], "traction", value=(0.0, sign * p_in, 0.0)) + bc(im.attributes["outlet"], "do_nothing")


def axis_count(im, R: float, Lz: float) -> dict:
    """The axis faces of the pipe's instance mesh, measured: their total length, the length below and above the
    region interface z = Lz/2, their largest r, and the boundary segments lying on the interface (none)."""
    fp = im.face_points(attribute=im.axis_attributes[0], order=2) if im.axis_attributes else None
    out = {"length": 0.0, "lower": 0.0, "upper": 0.0, "max_r": 0.0}
    if fp is not None:
        w, z = fp["weights"], fp["points"][..., 1]
        out.update(length=float(w.sum()), lower=float(w[z < 0.5 * Lz].sum()), upper=float(w[z > 0.5 * Lz].sum()),
                   max_r=float(np.abs(fp["points"][..., 0]).max()))
    V, B = im.vertices(), im.boundary()
    on_interface = np.all(np.abs(V[B][..., 1] - 0.5 * Lz) <= 1e-12 * Lz, axis=1)
    out["interface_segments"] = int(on_interface.sum())
    return out


def measure_pipe_axi(case: dict, *, form: str | None = "laplacian", sign: float = 1.0) -> dict:
    """Hagen-Poiseuille flow in the axisymmetric pipe of two regions: u_z = p_in (R^2 - r^2) / (4 nu Lz), u_r =
    u_theta = 0, p = p_in (1 - z/Lz), Q = pi R^4 p_in / (8 nu Lz), exact in P2/P1. ``form`` None leaves the API's
    default (a planted defect of the test), ``sign`` -1 reverses the inlet traction (another)."""
    f, D, S = fluids(), case["data"], case["solver"]
    R, Lz, nu, p_in = D["R"], D["Lz"], D["nu"], D["p_in"]
    m = meshed(case)
    im = instance(case, m)
    fm = flow_mesh(im)
    kw = {} if form is None else {"form": form}
    sol = f.solve_stokes(fm, nu, bcs=pipe_conditions(im, p_in, sign), geometry="axisymmetric",
                         direct_tol=S["direct_tol"], lu_threads=1, **kw)
    x, u, p = sample_flow(sol, fm.n_elements)
    r, z = x[..., 0], x[..., 1]
    umax = p_in * R ** 2 / (4.0 * nu * Lz)
    Q = math.pi * R ** 4 * p_in / (8.0 * nu * Lz)
    quantities = {
        "u_z": float(np.abs(u[..., 1] - p_in * (R ** 2 - r ** 2) / (4.0 * nu * Lz)).max() / umax),
        "u_r": float(np.abs(u[..., 0]).max() / umax),
        "u_theta": float(np.abs(u[..., 2]).max() / umax),
        "p": float(np.abs(p - p_in * (1.0 - z / Lz)).max() / p_in),
        "Q_outlet": abs(volume_flux(sol, im, "outlet", axisymmetric=True) - Q) / Q,
        "Q_inlet": abs(volume_flux(sol, im, "inlet", axisymmetric=True) + Q) / Q,
    }
    return dict(quantities=quantities, recorded={}, m=m, im=im, fm=fm, sol=sol)


def stress_form_decay(case: dict, sol_laplacian, sol_stress, n: int) -> dict:
    """The stress form against plane Poiseuille flow, |u - u_P| / U_mean at the sample points, its maximum near the
    outlet (x > L - H) and over x < L - H, x < L - 2H and x < L - 3H."""
    D = case["data"]
    U, L, H = D["U_mean"], D["L"], D["H"]
    x, u, _ = sample_flow(sol_stress, n)
    xl, ul, _ = sample_flow(sol_laplacian, n)
    assert np.array_equal(x, xl)
    uref = 6.0 * U * x[..., 1] * (H - x[..., 1]) / H ** 2
    d = np.hypot(u[..., 0] - uref, u[..., 1]) / U
    X = x[..., 0]
    out = {"near_outlet": float(d[X > L - H].max()), "below_L_minus_H": float(d[X < L - H].max()),
           "below_L_minus_2H": float(d[X < L - 2 * H].max()), "below_L_minus_3H": float(d[X < L - 3 * H].max())}
    out["ratio_H_2H"] = out["below_L_minus_H"] / out["below_L_minus_2H"] if out["below_L_minus_2H"] > 0 else math.inf
    out["ratio_2H_3H"] = out["below_L_minus_2H"] / out["below_L_minus_3H"] if out["below_L_minus_3H"] > 0 else math.inf
    return out


def measure_poiseuille_channel(case: dict) -> dict:
    """``axmp.mesh(channel)`` (the note's 4.1 channel, L = 4H, mesh.size = H/16): the parabolic inflow, do_nothing
    at the outlet, no-slip walls. The Laplacian form is plane Poiseuille flow (exact in P2/P1); the stress form is
    solved too and its outlet disturbance measured (``recorded``; not an exactness test)."""
    m = meshed(case)
    im = instance(case, m)
    fm = flow_mesh(im)
    sol, _ = _plane_poiseuille(case, im, fm, "laplacian")
    stress, _ = _plane_poiseuille(case, im, fm, "stress")
    quantities = _poiseuille_errors(case, sol, fm.n_elements)
    return dict(quantities=quantities, recorded=stress_form_decay(case, sol, stress, fm.n_elements), m=m, im=im,
                fm=fm, sol=sol, stress=stress)


def measure_plate_cell(case: dict) -> dict:
    """The parallel-plate cell through ``m.chem_mesh()``: primary electrodes, phi_s = dV on ``anode`` (x = 0) and 0
    on ``cathode`` (x = gap), insulation elsewhere (no condition: the current distribution's default), conductivity
    kappa by element attribute. phi = dV (1 - x/gap), linear, exact in P2; the electrode current per metre, kappa
    grad phi . n over the electrode with n outward (anodic positive), is +kappa dV Hc / gap on the anode and its
    negative on the cathode."""
    c, D, S = chem(), case["data"], case["solver"]
    kappa, dV, gap, Hc = D["kappa"], D["dV"], D["gap"], D["Hc"]
    m = meshed(case)
    im = instance(case, m)
    cm = m.chem_mesh(case["instance"], order=S["order"])
    cd = c.CurrentDistribution(cm, order=S["order"])
    cd.set_conductivity({a: kappa for a in im.region_attribute.values()})
    cd.add_electrode("anode", im.attributes["anode"], primary=True, E_eq=0.0, phi_s=dV)
    cd.add_electrode("cathode", im.attributes["cathode"], primary=True, E_eq=0.0, phi_s=0.0)
    report = cd.solve(**S["newton"])
    I0 = kappa * dV * Hc / gap

    def current(name):
        q = cd.boundary_quadrature(im.attributes[name], S["quadrature"])
        return float(np.sum(np.asarray(q["w"]) * kappa * np.sum(np.asarray(q["grad"]) * np.asarray(q["normal"]), axis=1)))

    dq = cd.domain_quadrature(S["quadrature"])
    x, phi = np.asarray(dq["x"]), np.asarray(dq["phi"])
    quantities = {"phi": float(np.abs(phi - dV * (1.0 - x[:, 0] / gap)).max() / dV),
                  "I_anode": abs(current("anode") - I0) / I0, "I_cathode": abs(current("cathode") + I0) / I0}
    return dict(quantities=quantities, recorded={}, m=m, im=im, cm=cm, cd=cd, report=report)


MEASURE = {
    "heat_slab": measure_heat_slab, "two_regions_enclosed": measure_two_regions_enclosed,
    "two_regions_open": measure_two_regions_open, "number_is_a_label": measure_number_is_a_label,
    "two_layer": measure_two_layer, "pipe_axi": measure_pipe_axi, "poiseuille_channel": measure_poiseuille_channel,
    "plate_cell": measure_plate_cell,
}


def environment_note() -> str:
    return f"{record_key()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}"
