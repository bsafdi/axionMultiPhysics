"""Independent references for the field-map tests (tests/test_fieldmap_*.py).

Nothing here imports the package: closed-form fields, the scrambled-Halton sequence of the format written from its
specification, bilinear and trilinear interpolation of a written array, exact chord integrals of such an array, Gauss
quadrature along chords, and a Runge-Kutta tracker. Sources:

- complete elliptic integrals K(m), E(m) by the arithmetic-geometric mean (Abramowitz and Stegun 17.6);
- the circular current loop: B from K and E (J. D. Jackson, Classical Electrodynamics, 3rd ed., eq. 5.37 and problem
  5.15; the form of J. Simpson et al., NASA/TM-2013-217919), the vector potential A_phi (Jackson 5.37);
- the ideal finite solenoid (current sheet): N. Derby and S. Olbert, Am. J. Phys. 78, 229 (2010), eqs. 6 and 8, with
  Bulirsch's generalized complete elliptic integral cel (R. Bulirsch, Numer. Math. 7, 78 (1965); the algorithm as
  printed by Derby and Olbert); a thick winding is the sheet integrated over the radius by Gauss-Legendre;
- a sphere of permeability mu in a uniform field, and a thick spherical shell (Jackson 5.11-5.12: the scalar potential
  matched at each surface); mu = 0 is the ideal superconductor;
- a current loop above a permeable half-space by images (W. R. Smythe, Static and Dynamic Electricity, 3rd ed., 7.18;
  K. J. Binns, P. J. Lawrenson and C. W. Trowbridge, The Analytical and Numerical Solution of Electric and Magnetic
  Fields, ch. 3): in the loop's medium the loop plus an image loop carrying I (mu - 1)/(mu + 1) at the mirror point;
  in the permeable medium the loop's own vacuum field times 2 mu / (mu + 1);
- the two-dimensional quadrupole B = g (y, x, 0).
"""
from __future__ import annotations

import math

import numpy as np

MU0 = 1.25663706212e-6  # CODATA 2018, the value of the package's units (results.MU0)
C_GEV = 0.299792458      # GeV/c per (e T m): p [GeV/c] = 0.299792458 q [e] B [T] R [m]


# ------------------------------------------------------------------------------------------- elliptic integrals
def ellipke(m):
    """K(m) and E(m), parameter m = k^2 in [0, 1), by the arithmetic-geometric mean (A&S 17.6.1-17.6.4)."""
    m = np.asarray(m, dtype=float)
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    s = 0.5 * m
    w = 0.5
    for _ in range(60):
        c = 0.5 * (a - b)
        a, b = 0.5 * (a + b), np.sqrt(a * b)
        w *= 2.0
        s = s + w * c * c
        if np.all(np.abs(c) <= 1e-17 * a):
            break
    K = np.pi / (2.0 * a)
    return K, K * (1.0 - s)


# ------------------------------------------------------------------------------------------- current loop
def loop_B(a, current, r, z):
    """(B_r, B_z) [T] of a circular loop of radius a [m] in the plane z = 0, centred on the axis, current [A]."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    r, z = np.broadcast_arrays(r, z)
    a2 = a * a
    rho2 = r * r + z * z
    alpha2 = a2 + rho2 - 2.0 * a * r
    beta2 = a2 + rho2 + 2.0 * a * r
    beta = np.sqrt(beta2)
    K, E = ellipke(1.0 - alpha2 / beta2)
    C = MU0 * current / np.pi
    Bz = C / (2.0 * alpha2 * beta) * ((a2 - rho2) * E + alpha2 * K)
    rs = np.where(r > 0, r, 1.0)
    Br = np.where(r > 0, C * z / (2.0 * alpha2 * beta * rs) * ((a2 + rho2) * E - alpha2 * K), 0.0)
    return Br, Bz


def loop_psi(a, current, r, z):
    """psi = r A_phi [T m^2] of the loop (Jackson 5.37); psi = 0 on the axis."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    r, z = np.broadcast_arrays(r, z)
    m = 4.0 * a * r / ((a + r) ** 2 + z * z)
    K, E = ellipke(m)
    rs = np.where(r > 0, r, 1.0)
    ms = np.where(m > 0, m, 1.0)
    A = MU0 * current / np.pi * np.sqrt(a / rs) * ((1.0 - 0.5 * m) * K - E) / np.sqrt(ms)
    return np.where(r > 0, r * A, 0.0)


# ------------------------------------------------------------------------------------------- finite solenoid
def cel(kc, p, c, s):
    """Bulirsch's generalized complete elliptic integral (vectorized; kc != 0)."""
    kc, p, c, s = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (kc, p, c, s)))
    k = np.abs(kc)
    em = np.ones_like(k)
    pos = p > 0
    pp = np.where(pos, np.sqrt(np.where(pos, p, 1.0)), 0.0)
    cc = c.copy()
    ss = np.where(pos, s / np.where(pos, pp, 1.0), 0.0)
    if not np.all(pos):
        f = kc * kc
        q = 1.0 - f
        g = 1.0 - p
        f = f - p
        q = q * (s - c * p)
        ppn = np.sqrt(f / g)
        ccn = (c - s) / g
        ssn = -q / (g * g * ppn) + ccn * ppn
        pp = np.where(pos, pp, ppn)
        cc = np.where(pos, cc, ccn)
        ss = np.where(pos, ss, ssn)
    f = cc
    cc = cc + ss / pp
    g = k / pp
    ss = 2.0 * (ss + f * g)
    pp = g + pp
    g = em
    em = k + em
    kk = k
    for _ in range(60):
        done = np.abs(g - k) <= g * 1e-15
        if np.all(done):
            break
        k = np.where(done, k, 2.0 * np.sqrt(kk))
        kk = np.where(done, kk, k * em)
        f = cc
        cc = np.where(done, cc, cc + ss / pp)
        g_new = kk / pp
        ss = np.where(done, ss, 2.0 * (ss + f * g_new))
        pp = np.where(done, pp, g_new + pp)
        g = np.where(done, g, em)
        em = np.where(done, em, k + em)
    return 0.5 * np.pi * (ss + cc * em) / (em * (em + pp))


def sheet_B(a, b, K_s, r, z):
    """(B_r, B_z) of an ideal cylindrical current sheet: radius a, z in [-b, b], azimuthal surface current K_s [A/m]
    (Derby and Olbert eqs. 6 and 8)."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    r, z = np.broadcast_arrays(r, z)
    B0 = MU0 * K_s / np.pi
    Br = np.zeros_like(r)
    Bz = np.zeros_like(r)
    gam = (a - r) / (a + r)
    for sg in (1.0, -1.0):
        zz = z + sg * b
        den = np.sqrt(zz * zz + (r + a) ** 2)
        al = a / den
        be = zz / den
        kc = np.sqrt((zz * zz + (a - r) ** 2) / (zz * zz + (a + r) ** 2))
        Br = Br + sg * B0 * al * cel(kc, 1.0, 1.0, -1.0)
        Bz = Bz + sg * B0 * a / (a + r) * be * cel(kc, gam * gam, 1.0, gam)
    return Br, Bz


_GL24 = np.polynomial.legendre.leggauss(24)


def solenoid_B(R, H, tw, NI, r, z, nodes=24):
    """(B_r, B_z) of a thick solenoid: uniform azimuthal current density NI / (tw H) in R < r' < R + tw, |z'| < H/2;
    the sheet field integrated over the radius (Gauss-Legendre, 24 nodes by default, split at r' = r inside the
    winding)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    r, z = np.broadcast_arrays(r, z)
    J = NI / (tw * H)
    x, w = _GL24 if nodes == 24 else np.polynomial.legendre.leggauss(nodes)
    inside = (r > R) & (r < R + tw)
    pieces = [(np.full_like(r, R), np.where(inside, r, R + tw)), (np.where(inside, r, R + tw), np.full_like(r, R + tw))]
    Br = np.zeros_like(r)
    Bz = np.zeros_like(r)
    for lo, hi in pieces:
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        aa = mid[:, None] + half[:, None] * x[None, :]
        br, bz = sheet_B(aa, 0.5 * H, J, r[:, None], z[:, None])
        Br = Br + (half[:, None] * w[None, :] * br).sum(axis=1)
        Bz = Bz + (half[:, None] * w[None, :] * bz).sum(axis=1)
    return Br, Bz


def solenoid_B_by_loops(R, H, tw, NI, r, z, nr=32, nz=96):
    """The same field as a superposition of loops (Gauss-Legendre over the winding's cross-section): a check of
    ``solenoid_B`` away from the winding."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    xr, wr = np.polynomial.legendre.leggauss(nr)
    xz, wz = np.polynomial.legendre.leggauss(nz)
    J = NI / (tw * H)
    Br = np.zeros(np.broadcast(r, z).shape)
    Bz = np.zeros_like(Br)
    for i in range(nr):
        a = R + 0.5 * tw * (xr[i] + 1.0)
        for j in range(nz):
            zz = 0.5 * H * xz[j]
            dI = J * (0.5 * tw * wr[i]) * (0.5 * H * wz[j])
            br, bz = loop_B(a, dI, r, z - zz)
            Br += br
            Bz += bz
    return Br, Bz


# ------------------------------------------------------------------------------------------- spheres and shells
def sphere_K_Bin(mu):
    """Dipole coefficient K (in units of a^3) and interior field factor of a sphere of permeability mu (mu = 0: the
    ideal superconductor)."""
    if mu == 0:
        return -0.5, 0.0
    return (mu - 1.0) / (mu + 2.0), 3.0 * mu / (mu + 2.0)


def shell_K_Bin(a1, a2, mu):
    """A thick shell a1 < rho < a2 of permeability mu: K in units of a2^3 and the interior field factor. The scalar
    potential -H0 (A rho + B / rho^2) cos(theta) in the three zones (A1, 0), (A2, B2), (1, B3), with Phi and mu dPhi/drho
    continuous at a1 and a2."""
    M = np.array([[a1, -a1, -1.0 / a1 ** 2, 0.0],
                  [1.0, -mu, 2.0 * mu / a1 ** 3, 0.0],
                  [0.0, a2, 1.0 / a2 ** 2, -1.0 / a2 ** 2],
                  [0.0, mu, -2.0 * mu / a2 ** 3, 2.0 / a2 ** 3]])
    rhs = np.array([0.0, 0.0, a2, 1.0])
    A1, A2, B2, B3 = np.linalg.solve(M, rhs)
    return -B3 / a2 ** 3, A1


def dipole_exterior(B0, K, a, r, z):
    """B0 z-hat plus the field of a sphere's induced dipole, K in units of a^3: (B_r, B_z)."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    rho2 = r * r + z * z
    c = B0 * K * a ** 3 / (rho2 ** 2 * np.sqrt(rho2))
    return 3.0 * c * z * r, B0 + c * (3.0 * z * z - rho2)


# ------------------------------------------------------------------------------------------- half-space by images
def loop_over_halfspace(a, current, h, mu, r, z, side):
    """(B_r, B_z) of a loop of radius a at height h > 0 above the plane z = 0, below which the permeability is mu.
    ``side`` "air" evaluates the field of the loop's medium (valid for z >= 0, and its analytic continuation below);
    "iron" the permeable medium's (valid for z <= 0, and its continuation above)."""
    if side == "air":
        b1 = loop_B(a, current, r, np.asarray(z) - h)
        b2 = loop_B(a, current * (mu - 1.0) / (mu + 1.0), r, np.asarray(z) + h)
        return b1[0] + b2[0], b1[1] + b2[1]
    f = 2.0 * mu / (mu + 1.0)
    b = loop_B(a, current, r, np.asarray(z) - h)
    return f * b[0], f * b[1]


def quadrupole_B(g, x, y, z):
    x = np.asarray(x, dtype=float)
    return g * np.asarray(y, dtype=float), g * x, np.zeros_like(x)


def loop_B_cartesian(a, current, x, y, z):
    """The loop's field in Cartesian components."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    Br, Bz = loop_B(a, current, r, z)
    phi = np.arctan2(y, x)
    return Br * np.cos(phi), Br * np.sin(phi), Bz


# ------------------------------------------------------------------------------------------- scrambled Halton
_G = 0x9E3779B97F4A7C15
_M64 = (1 << 64) - 1
BASES = (2, 3, 5, 7, 11, 13)


def _digits(b):
    """Smallest n with b**n >= 2**53, plus one."""
    n = 0
    while b ** n < 2 ** 53:
        n += 1
    return n + 1


def halton_tables(seed, dims=6):
    """Digit permutations: one splitmix64 stream from ``seed``; per dimension d (base b = BASES[d]) and digit j, a
    Fisher-Yates shuffle of 0..b-1 (i from b-1 down to 1, swap with k = next % (i + 1))."""
    state = seed & _M64
    tables = []
    for d in range(dims):
        b = BASES[d]
        per_digit = []
        for _ in range(_digits(b)):
            perm = list(range(b))
            for i in range(b - 1, 0, -1):
                state = (state + _G) & _M64
                z = state
                z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
                z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
                z = z ^ (z >> 31)
                k = z % (i + 1)
                perm[i], perm[k] = perm[k], perm[i]
            per_digit.append(perm)
        tables.append(per_digit)
    return tables


_ONE_MINUS = 1.0 - 2.0 ** -53


def halton(seed, n, dims=6):
    """Points 1..n: u_d = sum_j perm[d][j][digit_j(k)] * b^-(j+1), accumulated in double as v += p * f, f /= b
    (f starting at 1.0 / b), over every one of the digits of the table, then min(v, 1 - 2^-53). Vectorized over the
    points with the same operations, in the same order, for each point."""
    tables = halton_tables(seed, dims)
    out = np.empty((n, dims))
    k = np.arange(1, n + 1, dtype=np.int64)
    for d in range(dims):
        b = BASES[d]
        m = k.copy()
        f = 1.0 / b
        v = np.zeros(n)
        for perm in tables[d]:
            dig = m % b
            m //= b
            v += np.asarray(perm, dtype=np.float64)[dig] * f
            f /= b
        out[:, d] = np.minimum(v, _ONE_MINUS)
    return out


def family_seed(seed, index):
    return (seed ^ ((index * _G) & _M64)) & _M64


# ------------------------------------------------------------------------------------------- map interpolation
def interp_native(origin, spacing, B, pts):
    """Multilinear interpolation of nodal values B (shape grid + (3,)) at points (n, d) of the grid's frame; node i
    along axis a sits at origin[a] + i * spacing[a]; points on or inside the box only."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    d = pts.shape[1]
    shape = B.shape[:d]
    idx = []
    frac = []
    for ax in range(d):
        t = (pts[:, ax] - origin[ax]) / spacing[ax]
        i = np.clip(np.floor(t).astype(np.int64), 0, shape[ax] - 2)
        idx.append(i)
        frac.append(t - i)
    out = np.zeros((pts.shape[0], 3))
    for corner in range(2 ** d):
        w = np.ones(pts.shape[0])
        sel = []
        for ax in range(d):
            bit = (corner >> ax) & 1
            w = w * (frac[ax] if bit else 1.0 - frac[ax])
            sel.append(idx[ax] + bit)
        out += w[:, None] * B[tuple(sel)]
    return out


def kind1_cartesian(origin, spacing, B, xyz):
    """A kind-1 map at Cartesian points: (B_r, B_z, B_phi) at (r, z), projected with cos(phi), sin(phi)."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    b = interp_native(origin, spacing, B, np.column_stack([r, xyz[:, 2]]))
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([b[:, 0] * c - b[:, 2] * s, b[:, 0] * s + b[:, 2] * c, b[:, 1]])


def map_chord_integrals(origin, spacing, B, start, direction, seg, seg_chord, nchords):
    """Exact integrals of a multilinear map along chord segments: each segment p + s v, s in [s0, s1], is cut where it
    crosses a grid line, and each piece (a polynomial of degree d along the line) is integrated by d-point
    Gauss-Legendre, exact for it. Returns (nchords, 3)."""
    start = np.asarray(start, dtype=float)
    direction = np.asarray(direction, dtype=float)
    seg = np.asarray(seg, dtype=float)
    seg_chord = np.asarray(seg_chord, dtype=np.int64)
    d = start.shape[1]
    p = start[seg_chord]
    v = direction[seg_chord]
    s0, s1 = seg[:, 0], seg[:, 1]
    sid_all = [np.arange(len(seg)), np.arange(len(seg))]
    s_all = [s0, s1]
    for ax in range(d):
        x0 = p[:, ax] + s0 * v[:, ax]
        x1 = p[:, ax] + s1 * v[:, ax]
        lo = np.ceil((np.minimum(x0, x1) - origin[ax]) / spacing[ax]).astype(np.int64)
        hi = np.floor((np.maximum(x0, x1) - origin[ax]) / spacing[ax]).astype(np.int64)
        moving = np.abs(v[:, ax]) > 0
        cnt = np.where(moving, np.maximum(hi - lo + 1, 0), 0)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        sid = np.repeat(np.arange(len(seg)), cnt)
        off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        line = origin[ax] + (lo[sid] + off) * spacing[ax]
        s = (line - p[sid, ax]) / v[sid, ax]
        keep = (s > s0[sid]) & (s < s1[sid])
        sid_all.append(sid[keep])
        s_all.append(s[keep])
    sid = np.concatenate(sid_all)
    s = np.concatenate(s_all)
    order = np.lexsort((s, sid))
    sid, s = sid[order], s[order]
    same = sid[1:] == sid[:-1]
    a, b, ps = s[:-1][same], s[1:][same], sid[:-1][same]
    xg, wg = np.polynomial.legendre.leggauss(max(d, 2))
    out = np.zeros((nchords, 3))
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    for q in range(len(xg)):
        sq = mid + half * xg[q]
        pts = p[ps] + sq[:, None] * v[ps]
        val = interp_native(origin, spacing, B, pts)
        wq = half * wg[q]
        for comp in range(3):
            out[:, comp] += np.bincount(seg_chord[ps], weights=wq * val[:, comp], minlength=nchords)
    return out


def chord_points(start, direction, seg, seg_chord, sub_len, npts):
    """Gauss-Legendre points along chord segments: each segment cut into equal pieces no longer than ``sub_len``,
    ``npts`` points per piece. Returns (points (N, d), weights (N,), chord index (N,))."""
    start = np.asarray(start, dtype=float)
    direction = np.asarray(direction, dtype=float)
    seg = np.asarray(seg, dtype=float)
    seg_chord = np.asarray(seg_chord, dtype=np.int64)
    length = seg[:, 1] - seg[:, 0]
    m = np.maximum(np.ceil(length / sub_len).astype(np.int64), 1)
    sid = np.repeat(np.arange(len(seg)), m)
    k = np.arange(int(m.sum())) - np.repeat(np.cumsum(m) - m, m)
    h = length[sid] / m[sid]
    a = seg[sid, 0] + k * h
    xg, wg = np.polynomial.legendre.leggauss(npts)
    s = (a[:, None] + 0.5 * h[:, None] * (xg[None, :] + 1.0)).ravel()
    w = (0.5 * h[:, None] * wg[None, :]).ravel()
    sid = np.repeat(sid, npts)
    c = seg_chord[sid]
    pts = start[c] + s[:, None] * direction[c]
    return pts, w, c


def integrate(values, w, c, nchords):
    """Sum of w * values per chord: (nchords, ncomp)."""
    values = np.atleast_2d(np.asarray(values, dtype=float).T).T
    return np.column_stack([np.bincount(c, weights=w * values[:, k], minlength=nchords) for k in range(values.shape[1])])


# ------------------------------------------------------------------------------------------- the statistic
def per_chord(I_a, I_b, I_abs, L, B_patch, floor=0.1):
    """|I_a - I_b| / max(I_abs, floor L B_patch), per chord."""
    den = np.maximum(I_abs, floor * L * B_patch)
    return np.linalg.norm(np.asarray(I_a) - np.asarray(I_b), axis=1) / den


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


# ------------------------------------------------------------------------------------------- the refinement decision
def decide_rule(E, tol_eff):
    """The test's own statement of the rule: at the last level L, accepted if every family is within its tol_eff;
    else, from L >= 2, not converging if a family still above its tol_eff gained less than 3.0 over the last two
    halvings (the smallest gain; ties to the alphabetically first family); else refine."""
    L = len(next(iter(E.values()))) - 1
    if all(E[f][L] <= tol_eff[f] for f in E):
        return {"status": "accepted", "level": L, "family": None, "gain": None}
    if L >= 2:
        worst = None
        for f in sorted(E):
            if E[f][L] <= tol_eff[f]:
                continue
            g = math.inf if E[f][L] == 0 else E[f][L - 2] / E[f][L]
            if g < 3.0 and (worst is None or g < worst[1]):
                worst = (f, g)
        if worst:
            return {"status": "not_converging", "level": L, "family": worst[0], "gain": worst[1]}
    return {"status": "refine", "level": None, "family": None, "gain": None}


# ------------------------------------------------------------------------------------------- tracking
def rk4_track(field, x0, u0, p_GeV, q, ds, n_steps, stop=None):
    """Integrate a charged particle's path in a static field: dx/ds = u, du/ds = (0.299792458 q / p) u x B(x), with p
    [GeV/c] constant, s the path length [m], B(x) [T] from ``field`` ((3,) -> (3,)). Classical fourth-order
    Runge-Kutta; ``stop(x)`` ends the track. Returns (positions, directions) including the start."""
    k = C_GEV * q / p_GeV
    x = np.array(x0, dtype=float)
    u = np.array(u0, dtype=float)
    u = u / np.linalg.norm(u)
    xs = [x.copy()]
    us = [u.copy()]

    def f(x, u):
        return u, k * np.cross(u, field(x))

    for _ in range(n_steps):
        k1x, k1u = f(x, u)
        k2x, k2u = f(x + 0.5 * ds * k1x, u + 0.5 * ds * k1u)
        k3x, k3u = f(x + 0.5 * ds * k2x, u + 0.5 * ds * k2u)
        k4x, k4u = f(x + ds * k3x, u + ds * k3u)
        x = x + ds / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        u = u + ds / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        xs.append(x.copy())
        us.append(u.copy())
        if stop is not None and stop(x):
            break
    return np.array(xs), np.array(us)


def second_derivative_bound(f, pts, h, delta=1e-4):
    """(h_a^2 / 8) max |d^2 f / dx_a^2| summed over the axes a: the bound of the multilinear interpolation error in a
    cell (per component). The maximum over the cell is taken over its corners and centre (the point's cell, found
    from the grid given by ``h`` = (origin, spacing)); the derivatives by central differences of step ``delta``."""
    origin, spacing = h
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    d = pts.shape[1]
    base = np.stack([origin[a] + np.floor((pts[:, a] - origin[a]) / spacing[a]) * spacing[a] for a in range(d)], axis=1)
    samples = [base + np.asarray([((c >> a) & 1) * spacing[a] for a in range(d)]) for c in range(2 ** d)]
    samples.append(base + 0.5 * np.asarray(spacing))
    bound = np.zeros((pts.shape[0], 3))
    for a in range(d):
        e = np.zeros(d)
        e[a] = delta
        worst = np.zeros((pts.shape[0], 3))
        for q in samples:
            d2 = (f(q + e) - 2.0 * f(q) + f(q - e)) / delta ** 2
            worst = np.maximum(worst, np.abs(d2))
        bound += spacing[a] ** 2 / 8.0 * worst
    return bound


# ------------------------------------------------------------------------------------------- exact chord intersections
def _sphere_roots(p, v, radius):
    """Both parameters s where the line p + s v (v unit) meets the sphere (2D: circle) |x| = radius; NaN if none."""
    b = np.sum(p * v, axis=1)
    c = np.sum(p * p, axis=1) - radius * radius
    disc = b * b - c
    sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
    return np.column_stack([-b - sq, -b + sq])


def exact_chord_intervals(start, direction, length, inside, radii=(), half_plane=False, box=None):
    """The exact intersection of each chord {start + s direction : 0 <= s <= length} with a region given by the
    predicate ``inside`` (points (n, d) -> bool), whose boundary is made of spheres (circles) centred at the origin
    with the given radii, the axis r = 0 (``half_plane``, kind 1: the first coordinate is r) and the planes of
    ``box`` ([[lo, hi], ...]). Every boundary crossing is a root computed in closed form; between consecutive
    breakpoints the predicate is evaluated at the midpoint. Returns (total length (n,), number of separate intervals
    (n,), breakpoints (n, k) sorted, inside flags of the pieces (n, k - 1))."""
    start = np.asarray(start, dtype=float)
    direction = np.asarray(direction, dtype=float)
    n, d = start.shape
    L = np.broadcast_to(np.asarray(length, dtype=float), (n,))
    cols = [np.zeros(n), L.copy()]
    for rad in radii:
        cols += list(_sphere_roots(start, direction, rad).T)
    if half_plane:
        with np.errstate(divide="ignore", invalid="ignore"):
            cols.append(np.where(direction[:, 0] != 0, -start[:, 0] / direction[:, 0], np.nan))
    if box is not None:
        for a in range(d):
            for b in box[a]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    cols.append(np.where(direction[:, a] != 0, (b - start[:, a]) / direction[:, a], np.nan))
    s = np.column_stack(cols)
    s = np.where(np.isfinite(s), np.clip(s, 0.0, L[:, None]), 0.0)
    s.sort(axis=1)
    mid = 0.5 * (s[:, 1:] + s[:, :-1])
    pts = start[:, None, :] + mid[:, :, None] * direction[:, None, :]
    flags = inside(pts.reshape(-1, d)).reshape(n, -1) & (s[:, 1:] > s[:, :-1])
    total = np.sum(np.where(flags, s[:, 1:] - s[:, :-1], 0.0), axis=1)
    # separate intervals: runs of inside pieces, ignoring empty pieces in between
    count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        prev = False
        for j in range(flags.shape[1]):
            if s[i, j + 1] == s[i, j]:
                continue
            if flags[i, j] and not prev:
                count[i] += 1
            prev = flags[i, j]
    return total, count, s, flags
