"""Independent references for the tests of the particle transport and its driver (``axionmultiphysics.particles``,
``axmp-transport`` modes ``transport`` and ``probe``).

Written from the design note and from physics, never from the implementation:

- seeds: SplitMix64 and the derivation of chunk seeds, from their specification;
- statistics: Clopper-Pearson intervals (scipy's beta distribution), the zero-count limits, batch means, the PDG
  rounding of an error;
- exact trajectories: helices in uniform fields, piecewise circles across a field jump at a sphere, the field and
  vector potential of a current loop (complete elliptic integrals), a Runge-Kutta reference through any field;
- multiple scattering (Highland, PDG 2024 eq. 34.16), the tracking resolution (Gluckstern, PDG 2024 eq. 35.61 to
  35.63);
- hand-built axisymmetric partitions in the shape ``axionmultiphysics.solids.from_partition`` reads (through
  ``_solids_ref.Partition``), and the materials the cases use.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _solids_ref import Partition  # noqa: E402

C_LIGHT = 299792458.0            # m/s
GEV_PER_TM = 0.299792458         # p [GeV/c] = 0.299792458 B [T] R [m] for unit charge
MU0 = 4e-7 * math.pi             # the SI value the package uses for mu0 in its closed forms
M_MU_EV = 105.6583755e6          # muon mass, eV (PDG 2024)
M_P_EV = 938.27208816e6          # proton mass, eV (PDG 2024)

# ============================================================================================ seeds
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def mix(z: int) -> int:
    """SplitMix64's output function (Steele, Lea and Flood 2014; Vigna's public-domain splitmix64.c)."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def key_hash(k) -> int:
    if isinstance(k, bool):
        raise TypeError("a key is an int or a str")
    if isinstance(k, int):
        return k & MASK64
    return int.from_bytes(hashlib.sha256(k.encode("utf-8")).digest()[:8], "little")


def derive_seed(seed: int, *keys) -> int:
    z = seed & MASK64
    for k in keys:
        z = mix(((z ^ key_hash(k)) + GOLDEN) & MASK64)
    return z


def chunk_seed(seed: int, chunk: int, stage: str = "transport", instance: str = "transport") -> int:
    return derive_seed(seed, stage, instance, chunk)


def event_words(chunk_seed_: int, event: int) -> list[int]:
    o0 = mix((chunk_seed_ + (2 * event + 1) * GOLDEN) & MASK64)
    o1 = mix((chunk_seed_ + (2 * event + 2) * GOLDEN) & MASK64)
    return [o0 & 0xFFFFFFFF, o0 >> 32, o1 & 0xFFFFFFFF, o1 >> 32]


# ============================================================================================ statistics
ONE_SIGMA = math.erf(1.0 / math.sqrt(2.0))      # 0.682689...


def clopper_pearson(k: int, n: int, cl: float = ONE_SIGMA) -> tuple[float, float]:
    a = 1.0 - cl
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def zero_fraction_limit(n: int) -> float:
    """The exact one-sided 95 % upper limit on a fraction after 0 of n."""
    return 1.0 - 0.05 ** (1.0 / n)


ZERO_COUNT_LIMIT = -math.log(0.05)              # Poisson, 95 %


def batch_sigma(values, weights) -> tuple[float, float]:
    """vbar = sum w v; sigma^2 = B/(B-1) sum w^2 (v - vbar)^2 with w normalised (the note's batch means, each batch
    weighted by its events)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    w = w / w.sum()
    vbar = float(np.sum(w * v))
    B = len(v)
    return vbar, math.sqrt(B / (B - 1) * float(np.sum(w * w * (v - vbar) ** 2)))


def pdg_round(value: float, error: float) -> tuple[str, str, int]:
    """The PDG rule: the three leading digits of the error decide 2 (100-354), 1 (355-949) or, rounded up to 1000, 2
    significant digits; the value is rounded to the same decimal place. Returns (value, error, decimals) as strings
    at that place (decimals may be negative)."""
    e = abs(error)
    p = math.floor(math.log10(e))
    lead = int(round(e / 10 ** (p - 2)))
    if lead >= 1000:                       # 999.6 and above round to 1000
        p += 1
        lead = int(round(e / 10 ** (p - 2)))
    if lead <= 354:
        nd = 2
    elif lead <= 949:
        nd = 1
    else:
        nd, p = 2, p + 1                   # 950-999 -> 1000: two digits of the next decade
    dec = nd - 1 - p
    q = 10.0 ** (-dec)
    ev = round(e / q) * q
    vv = round(value / q) * q
    fmt = f"{{:.{max(dec, 0)}f}}"
    return fmt.format(vv), fmt.format(ev), dec


# ============================================================================================ fields and trajectories
def helix_z(s, p_t, p_z, B, q=1.0):
    """Position at path length s of a particle of charge q (units of e) starting at the origin with momentum
    (p_t, 0, p_z) [GeV/c] in B along +z [T]. Positive charges turn clockwise seen from +z."""
    p = math.hypot(p_t, p_z)
    R = p_t / (GEV_PER_TM * abs(B))
    sgn = math.copysign(1.0, q * B)
    th = (np.asarray(s, float) * p_t / p) / R
    return np.stack([R * np.sin(th), -sgn * R * (1.0 - np.cos(th)), np.asarray(s, float) * p_z / p], axis=-1)


def helix_x(s, p_par, p_perp, B, q=1.0):
    """Start at the origin with momentum (p_par, 0, p_perp) [GeV/c] in B along +x: the circle lies in (y, z) and
    turns from +z towards +y for a positive charge and positive B (v x B = z x x = +y)."""
    p = math.hypot(p_par, p_perp)
    R = p_perp / (GEV_PER_TM * abs(B))
    sgn = math.copysign(1.0, q * B)
    th = (np.asarray(s, float) * p_perp / p) / R
    return np.stack([np.asarray(s, float) * p_par / p, sgn * R * (1.0 - np.cos(th)), R * np.sin(th)], axis=-1)


def first_root(f, s0, s1, n=4000):
    """The first root of f on [s0, s1] (sampled, then Brent to machine precision)."""
    s = np.linspace(s0, s1, n)
    v = np.array([f(x) for x in s])
    k = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) <= 0)[0]
    if len(k) == 0:
        return None
    return optimize.brentq(f, s[k[0]], s[k[0] + 1], xtol=1e-15, rtol=1e-15, maxiter=200)


def circles_across_sphere(b, p_gev, a, B_in, B_out, R_exit, x0):
    """mu+ (charge +1) in the plane z = 0, starting at (x0, b) along +x, B along +z: B_in inside |x| < a, B_out
    outside. Exact piecewise circles until the circle |x| = R_exit. Returns (x, y, dir_x, dir_y) there."""
    x, y, th = x0, b, 0.0
    inside = math.hypot(x, y) < a
    for _ in range(50):
        Bz = B_in if inside else B_out
        R = p_gev / (GEV_PER_TM * Bz)
        c = np.array([x + R * math.sin(th), y - R * math.cos(th)])   # centre: clockwise turn
        phi0 = math.atan2(y - c[1], x - c[0])
        cn = math.hypot(*c)
        al = math.atan2(c[1], c[0])
        cands = []
        for rad, kind in ((a, "ball"), (R_exit, "exit")):
            K = (rad * rad - cn * cn - R * R) / (2 * R)
            if abs(K) <= cn:
                t = math.acos(K / cn)
                cands += [(al + t, kind), (al - t, kind)]
        best = None
        for phi, kind in cands:
            d = (phi0 - phi) % (2 * math.pi)
            if d > 1e-12 and (best is None or d < best[0]):
                best = (d, phi, kind)
        d, phi, kind = best
        x, y = c[0] + R * math.cos(phi), c[1] + R * math.sin(phi)
        th -= d
        if kind == "exit":
            return x, y, math.cos(th), math.sin(th)
        inside = not inside
    raise RuntimeError("no exit")


def loop_field(r, z, a, I, z0=0.0):
    """B_r, B_z [T] of a circular filament of radius a [m] in the plane z = z0 carrying I [A], counter-clockwise
    seen from +z (B_z > 0 at the centre). Smythe's closed form with K(m), E(m), m = k^2."""
    r = np.asarray(r, float)
    zz = np.asarray(z, float) - z0
    ap = (a + r) ** 2 + zz ** 2
    am = (a - r) ** 2 + zz ** 2
    m = 4 * a * r / ap
    K, E = special.ellipk(m), special.ellipe(m)
    c = MU0 * I / (2 * math.pi)
    bz = c / np.sqrt(ap) * (K + (a * a - r * r - zz * zz) / am * E)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = np.where(r > 1e-12, c / r * zz / np.sqrt(ap) * (-K + (a * a + r * r + zz * zz) / am * E), 0.0)
    return br, bz


def loop_Aphi(r, z, a, I, z0=0.0):
    """A_phi [T m] of the same filament: (mu0 I / (pi k)) sqrt(a / r) [(1 - k^2/2) K - E]."""
    r = np.asarray(r, float)
    zz = np.asarray(z, float) - z0
    m = 4 * a * r / ((a + r) ** 2 + zz ** 2)
    k = np.sqrt(m)
    return MU0 * I / (math.pi * k) * np.sqrt(a / r) * ((1 - m / 2) * special.ellipk(m) - special.ellipe(m))


def canonical_p_phi(x, p_eV_c, q, A_phi):
    """P_phi = r p_phi + q r A_phi in eV/c m: (x p_y - y p_x) + q c r A_phi."""
    x = np.atleast_2d(x)
    p = np.atleast_2d(p_eV_c)
    r = np.hypot(x[:, 0], x[:, 1])
    return x[:, 0] * p[:, 1] - x[:, 1] * p[:, 0] + q * C_LIGHT * r * A_phi(r, x[:, 2])


def solenoid_field(r, z, R, H, tw, NI, nodes=16):
    """B_r, B_z of a uniformly wound rectangular winding r in [R, R + tw], z in [-H/2, H/2] with NI ampere-turns:
    loops integrated over the cross-section by Gauss-Legendre (nodes x nodes)."""
    g, w = np.polynomial.legendre.leggauss(nodes)
    rs = R + tw * (g + 1) / 2
    zs = H * g / 2
    J = NI / (tw * H)
    br = np.zeros(np.shape(r))
    bz = np.zeros(np.shape(r))
    for i in range(nodes):
        for j in range(nodes):
            dI = J * (tw / 2 * w[i]) * (H / 2 * w[j])
            b1, b2 = loop_field(r, z, rs[i], dI, zs[j])
            br += b1
            bz += b2
    return br, bz


def rk_trajectory(field, x0, p0_eV_c, q, s_max, events, rtol=1e-11, atol=1e-12, max_step=np.inf):
    """Integrate dx/ds = u, du/ds = q c (u x B) / |p| (u unit; p in eV/c, B in T) from x0 with momentum p0, through
    ``field(x) -> B`` (one point, (3,) arrays), stopping at the last of ``events`` (functions of (s, y), terminal
    on the last). Returns the list of (event index, state at the event) in the order reached, and the solution."""
    p = float(np.linalg.norm(p0_eV_c))
    u0 = np.asarray(p0_eV_c, float) / p
    k = q * C_LIGHT / p

    def rhs(s, y):
        B = field(y[:3])
        u = y[3:]
        return np.concatenate([u, k * np.cross(u, B)])

    sol = integrate.solve_ivp(rhs, (0.0, s_max), np.concatenate([np.asarray(x0, float), u0]), method="DOP853",
                              rtol=rtol, atol=atol, events=events, dense_output=False, max_step=max_step)
    hits = []
    for i, ys in enumerate(sol.y_events):
        for y in ys:
            hits.append((i, y))
    return hits, sol


def highland(p_eV_c, mass_eV, x_over_X0, z=1.0):
    """PDG 2024 eq. 34.16: theta_0 = 13.6 MeV / (beta c p) z sqrt(x/X0) [1 + 0.038 ln(x z^2 / (X0 beta^2))]."""
    E = math.hypot(p_eV_c, mass_eV)
    beta = p_eV_c / E
    return 13.6e6 / (beta * p_eV_c) * z * math.sqrt(x_over_X0) * (1 + 0.038 * math.log(x_over_X0 * z * z / beta ** 2))


def gluckstern_dp_over_p(eps_m, p_gev, B_T, L_m, N):
    """PDG 2024 eq. 35.61 with p = 0.3 B / k: delta p / p = eps p sqrt(720 / (N + 4)) / (0.3 B L^2)."""
    return eps_m * p_gev * math.sqrt(720.0 / (N + 4)) / (GEV_PER_TM * B_T * L_m ** 2)


def ms_dp_over_p(p_gev, B_T, L_m, X0_m, beta=1.0, highland_log=True):
    """PDG 2024 eq. 35.63: delta k_ms = 0.016 GeV/c / (L p beta) sqrt(L / X0), so delta p / p = delta k p / (0.3 B).
    The equation scales theta_0 = 13.6 MeV / (beta p) sqrt(x / X0) and leaves out Highland's logarithm (eq. 34.16);
    with ``highland_log`` the logarithm of the whole length, 1 + 0.038 ln(L / (X0 beta^2)), multiplies it (0.81 for
    2 m of air: what a transport that reproduces Highland's width gives)."""
    v = 0.016 / (L_m * beta) * math.sqrt(L_m / X0_m) / (GEV_PER_TM * B_T)
    if highland_log:
        v *= 1 + 0.038 * math.log(L_m / (X0_m * beta * beta))
    return v


def lsq_curvature_sigma(z, eps):
    """Exact standard deviation of the fitted curvature 2c of y = a + b z + c z^2 by least squares with equal
    errors eps at abscissae z."""
    A = np.column_stack([np.ones_like(z), z, z * z])
    cov = eps ** 2 * np.linalg.inv(A.T @ A)
    return 2.0 * math.sqrt(cov[2, 2])


def fit_parabola(z, y):
    A = np.column_stack([np.ones_like(z), z, z * z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def fit_circle(x, y):
    """Least-squares circle (algebraic Kasa fit, then Gauss-Newton on the geometric distances). Returns (xc, yc, R)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0] / 2, c[1] / 2
    R = math.sqrt(c[2] + xc * xc + yc * yc)

    def res(v):
        return np.hypot(x - v[0], y - v[1]) - v[2]
    sol = optimize.least_squares(res, [xc, yc, R], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return tuple(sol.x)


def gaussian_core_sigma(theta, fraction=0.98, bins=60):
    """The note's P5 estimator: a Gaussian fitted (Poisson-weighted least squares) to the histogram of the central
    ``fraction`` of the projected angles. Returns sigma."""
    lo, hi = np.quantile(theta, [(1 - fraction) / 2, 1 - (1 - fraction) / 2])
    h, e = np.histogram(theta, bins=bins, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    s0 = float(np.std(theta[(theta >= lo) & (theta <= hi)]))

    def model(x, A, mu, s):
        return A * np.exp(-0.5 * ((x - mu) / s) ** 2)
    w = np.sqrt(np.maximum(h, 1.0))
    popt, _ = optimize.curve_fit(model, c, h, p0=[h.max(), 0.0, s0], sigma=w, absolute_sigma=True, maxfev=20000)
    return abs(popt[2])


# ============================================================================================ partitions
def slabs(z_edges, regions, rmax):
    """Faces [0, rmax] x [z_i, z_{i+1}] (tubes on the axis), region regions[i] each."""
    P = Partition()
    n = len(z_edges) - 1
    assert len(regions) == n
    h = [P.line((0.0, z), (rmax, z)) for z in z_edges]
    right = [P.line((rmax, z_edges[i]), (rmax, z_edges[i + 1])) for i in range(n)]
    axis = [P.line((0.0, z_edges[i]), (0.0, z_edges[i + 1])) for i in range(n)]
    for i in range(n):
        P.face(regions[i], [h[i], right[i], -h[i + 1], -axis[i]])
    return P, list(dict.fromkeys(regions))


def slabs_in_a_shell(z_edges, regions, rmax, t, shell="shell"):
    """``slabs`` closed by a shell of thickness t (region ``shell``): caps [0, rmax] x [z_0 - t, z_0] and
    [0, rmax] x [z_n, z_n + t], and the side [rmax, rmax + t] x [z_0 - t, z_n + t]. Every track that leaves the slabs
    for the world crosses the shell first."""
    edges = [z_edges[0] - t] + list(z_edges) + [z_edges[-1] + t]
    names = [shell] + list(regions) + [shell]
    P = Partition()
    n = len(edges) - 1
    h = [P.line((0.0, z), (rmax, z)) for z in edges]
    right = [P.line((rmax, edges[i]), (rmax, edges[i + 1])) for i in range(n)]
    axis = [P.line((0.0, edges[i]), (0.0, edges[i + 1])) for i in range(n)]
    for i in range(n):
        P.face(names[i], [h[i], right[i], -h[i + 1], -axis[i]])
    sb = P.line((rmax, edges[0]), (rmax + t, edges[0]))
    sr = P.line((rmax + t, edges[0]), (rmax + t, edges[-1]))
    st = P.line((rmax, edges[-1]), (rmax + t, edges[-1]))
    P.face(shell, [sb, sr, -st] + [-right[i] for i in reversed(range(n))])
    return P, list(dict.fromkeys(names))


def shells(r_edges, regions, z0, z1):
    """Faces [r_i, r_{i+1}] x [z0, z1], r_edges[0] = 0 (the first face touches the axis)."""
    assert r_edges[0] == 0.0
    P = Partition()
    n = len(r_edges) - 1
    v = [P.line((r, z0), (r, z1)) for r in r_edges]
    b = [P.line((r_edges[i], z0), (r_edges[i + 1], z0)) for i in range(n)]
    t = [P.line((r_edges[i], z1), (r_edges[i + 1], z1)) for i in range(n)]
    for i in range(n):
        P.face(regions[i], [b[i], v[i + 1], -t[i], -v[i]])
    return P, list(dict.fromkeys(regions))


def ball_in_cylinder(a, R_inner, R_outer, Z):
    """A ball (orb, radius a) on the axis at z = 0 inside a region 'outer' [0, R_inner] x [-Z, Z], with a shell
    'exit' [R_inner, R_outer] x [-Z, Z] around it."""
    P = Partition()
    bottom = P.line((0.0, -Z), (R_inner, -Z))
    mid = P.line((R_inner, -Z), (R_inner, Z))
    top = P.line((R_inner, Z), (0.0, Z))
    ax_top = P.line((0.0, Z), (0.0, a))
    arc = P.arc((0.0, -a), (0.0, a), (0.0, 0.0), (a, a), ccw=True)
    core_axis = P.line((0.0, a), (0.0, -a))
    ax_bot = P.line((0.0, -a), (0.0, -Z))
    sb = P.line((R_inner, -Z), (R_outer, -Z))
    sr = P.line((R_outer, -Z), (R_outer, Z))
    st = P.line((R_inner, Z), (R_outer, Z))
    P.face("outer", [bottom, mid, top, ax_top, -arc, ax_bot])
    P.face("ball", [arc, core_axis])
    P.face("exit", [sb, sr, -st, -mid])
    return P, ["outer", "ball", "exit"]


def rect_holes_in_half_disk(Rd, rects):
    """A half disk of radius Rd on the axis (region 'air') with rectangular holes, each its own face:
    rects = [(region, r0, r1, z0, z1), ...] (r0 > 0)."""
    P = Partition()
    arc = P.arc((0.0, -Rd), (0.0, Rd), (0.0, 0.0), (Rd, Rd), ccw=True)
    axis = P.line((0.0, Rd), (0.0, -Rd))
    holes = []
    for region, r0, r1, z0, z1 in rects:
        b = P.line((r0, z0), (r1, z0))
        rr = P.line((r1, z0), (r1, z1))
        t = P.line((r0, z1), (r1, z1))
        ll = P.line((r0, z0), (r0, z1))
        P.face(region, [b, rr, -t, -ll])
        holes.append([ll, t, -rr, -b])
    P.face("air", [arc, axis], *holes)
    return P, ["air"] + list(dict.fromkeys(r[0] for r in rects))


# ============================================================================================ materials
VACUUM = {"density": 1e-22, "particle": {"nist": "G4_Galactic"}}
AIR = {"density": 1.20479, "particle": {"nist": "G4_AIR"}}
IRON = {"density": 7874.0, "particle": {"nist": "G4_Fe"}}
LEAD = {"density": 11350.0, "particle": {"nist": "G4_Pb"}}
X0_FE_M = 1.757e-2      # PDG: 13.84 g/cm2 / 7.874 g/cm3
X0_PB_M = 0.5612e-2     # PDG: 6.37 g/cm2 / 11.35 g/cm3
X0_AIR_M = 303.9        # PDG: 36.62 g/cm2 / 1.205e-3 g/cm3
# Geant4-free stand-ins for the pure driver tests (no NIST composition, so no program is needed to resolve them)
H_GAS = {"density": 1e-3, "particle": {"elements": {"H": 1}, "by": "atoms"}}
FE_ELEM = {"density": 7874.0, "particle": {"elements": {"Fe": 1.0}, "by": "mass"}}
