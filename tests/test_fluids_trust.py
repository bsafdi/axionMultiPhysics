"""Accuracy gates of the fluids module that a caller cannot miss (the C++ side is cpp/tests/test_fluids_trust.cpp).

Every fluids convergence verdict is a ratio of discrete dual norms, and every dual norm, every per-step linear solve
and the initial Leray projection are solves whose accuracy must be recorded and gated against the caller's target.
This file checks the recorded and the refused behaviour through the public Python boundary, plus the places where a
shipped text has to agree with the code it describes.

  - the initial Leray projection: ``initialize()``'s verdict is recorded (``projection_converged``) with the
      backward error of its solve (``projection_residual``) and gated against a bound that depends neither on
      newton_rtol, nor on the shape of the domain, nor on the length unit (a projection solve made inaccurate on
      purpose is refused at every newton_rtol, and so is one whose multiplier alone is inaccurate; stretched
      channels and the tightest published newton_rtol are accepted; the same domain in four length units gives the same measure), and the projected state is discretely
      divergence free to an ABSOLUTE bound (an independent numpy assembly of int N_a div u)
  - the per-step linear solves: every Navier-Stokes history entry carries ``lu_residual``, and the diagnostics carry
      the worst value and the number of steps that missed the refinement target
  - the dual-norm mass solves: the worst mass-solve residual is recorded (``mass_lu_residual_max``) and bounded, for
      the steady, transient and scalar-transport paths
  - ``quad_order_convection`` below the exact degree of the convective form is refused, naming the option
  - the shipped texts: the cavity case definitions describe the Picard count the runner uses, the transient runner
      reads ``initialize()``'s verdict, the STRUMPACK sentence in docs/physics/fluids.md states what was measured,
      and the user-visible additions are in docs, llms.txt and CHANGELOG "Unreleased"
  - the C++ rank-invariance comparison is registered as a multi-rank ctest, so it is actually run
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CMAKELISTS = ROOT / "CMakeLists.txt"
FLUIDS_DOC = ROOT / "docs" / "physics" / "fluids.md"
CHANGELOG = ROOT / "CHANGELOG.md"
LLMS = ROOT / "llms.txt"
CASES = ROOT / "benchmarks" / "fluids" / "registry" / "cases"
F2A_RUNNER = ROOT / "benchmarks" / "fluids" / "cases" / "f2a_ns.py"
F3A_RUNNER = ROOT / "benchmarks" / "fluids" / "cases" / "f3a_transient.py"

B, R, T, L = 1, 2, 3, 4
RHO, MU = 1.25, 0.05

# The dual norm ||R||_* = sqrt(R^T M^-1 R) inherits the mass solve's error as |d(||R||_*^2)| <= eps sqrt(kappa(M))
# ||R||_*^2, with eps the solve's relative residual. 1e-10 keeps the dual norm's relative error below 1e-3 for any
# mass matrix with kappa(M) <= 1e14, which no usable consistent mass matrix exceeds; the solves themselves are
# refined to 1e-15 and reach ~3e-15, so the bound sits five decades above what a healthy solve delivers.
MASS_RESIDUAL_MAX = 1e-10
# The per-step linear solves are refined to min(1e-12, 1e-2 newton_rtol) = 1e-14 at the default newton_rtol and
# measure ~1.4e-15 on the Re = 100, 12x12 cavity below (and up to 8.4e-14 on the published 32x32 and 64x64 cavity
# continuations, 4.3e-13 on the final step of the published 128x128 manufactured-solution solves).
# The reported bound is that refinement target; what is enforced, on the final step only, is
# max(newton_rtol, 1e-11) (cpp/tests/test_fluids_trust.cpp holds the test that it is enforced).
NEWTON_RTOL = 1e-12
LU_REFINE_TARGET = min(1e-12, 1e-2 * NEWTON_RTOL)
# The initial projection is gated on the backward error of its saddle solve: for the momentum and for the continuity
# block, the norm of the residual over the norm of the sizes of the terms of that block, and the larger of the two
# (cpp/include/axionmultiphysics/fluids/sparse_lu.hpp, kProjectionResidualMax, defines it and holds the measurements).
# With Eigen's LU an accurate solve gives 3.8e-17 to 6.9e-17 whatever newton_rtol, the mesh, the shape of the domain
# and the length unit are (AMD EPYC 7452 and Intel Xeon Gold 6330), and the bound is 1e-13 for every newton_rtol.
PROJECTION_MAX = 1e-13
# What a projection solved with Eigen's LU (lu="eigen") is held to in this file: a factor 14 above the largest value
# measured on either processor, and a factor 100 below the bound. A projection solved with the default
# backend (lu=None: STRUMPACK when the build has it) is held to PROJECTION_MAX only, because STRUMPACK's accuracy on
# these cases depends on the processor: on the 1e5 x 1 channel below it measures 5.4e-17 on the AMD EPYC 7452 and
# 6.1e-15 on the Intel Xeon Gold 6330 (docs/physics/fluids.md has the cases it is refused on).
PROJECTION_ROUNDOFF = 1e-15
TIGHT_CASE = CASES / "mms_ns_transient_temporal.json"

def _fluids():
    return pytest.importorskip("axionmultiphysics.fluids", exc_type=ImportError)


def rect(nx, ny, lx=1.0, ly=1.0):
    """A structured right-triangle mesh of [0, lx] x [0, ly] (the unit square by default), and its vertices and
    triangles as numpy arrays."""
    fluids = _fluids()
    xs, ys = lx * np.linspace(0, 1, nx + 1), ly * np.linspace(0, 1, ny + 1)
    V = np.array([[x, y] for y in ys for x in xs])
    vid = lambda i, j: j * (nx + 1) + i  # noqa: E731
    tri, seg, att = [], [], []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tri += [[a, b, c], [a, c, d]]
    for i in range(nx):
        seg += [[vid(i, 0), vid(i + 1, 0)], [vid(i + 1, ny), vid(i, ny)]]
        att += [B, T]
    for j in range(ny):
        seg += [[vid(nx, j), vid(nx, j + 1)], [vid(0, j + 1), vid(0, j)]]
        att += [R, L]
    tri = np.array(tri, dtype=np.int32)
    mesh = fluids.Mesh.from_arrays(np.ascontiguousarray(V), tri, np.array(seg, dtype=np.int32),
                                   np.array(att, dtype=np.int32))
    return mesh, V, tri


def discrete_divergence(snap, V, tri):
    """max_a |int N_a div u| over the P1 vertex hats, assembled here in numpy from the released P2 field.

    This is the quantity the Leray projection drives to zero: a Taylor-Hood velocity is discretely, not pointwise,
    divergence free, so the pointwise divergence of a projected field is O(1) while every pressure-space test of it
    vanishes. The 3-point degree-2 midpoint rule is exact for N_a div u (degree 1 + 1) on straight triangles, so the
    only error here is round-off, and the assembly is independent of the solver's own B operator.
    """
    q = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    w = np.full(3, 1.0 / 6.0)
    N = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], axis=1)
    el = np.arange(snap.n_elements, dtype=np.int32)
    _, grad = snap.velocity(el, q)
    grad = np.asarray(grad)
    div = grad[..., 0, 0] + grad[..., 1, 1]
    P = V[tri]
    det = np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                 - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    out = np.zeros(len(V))
    contrib = det[:, None] * (div * w[None, :]) @ N
    for a in range(3):
        np.add.at(out, tri[:, a], contrib[:, a])
    return float(np.max(np.abs(out)))


def transient(mesh, scheme="newton", **kw):
    fluids = _fluids()
    bcs = [{"attribute": a, "kind": "dirichlet"} for a in (B, R, T, L)]
    return fluids.TransientNavierStokes(mesh, MU, RHO, None, bcs, scheme=scheme, lu="eigen", vectorized=True,
                                        newton_rtol=NEWTON_RTOL, **kw)


def u0_not_divergence_free(X, Y):
    s = X * (1 - X) * Y * (1 - Y)
    return np.stack([s * (1 + X), s * Y], axis=-1)


def lid(X, Y):
    """The singular lid: u = (1, 0) on the lid except at its corner vertices."""
    top = (np.abs(Y - 1.0) < 1e-12) & (X > 1e-12) & (X < 1.0 - 1e-12)
    return np.stack([np.where(top, 1.0, 0.0), 0.0 * X], axis=-1)


def cavity(n=12, re=100.0):
    fluids = _fluids()
    mesh, _, _ = rect(n, n)
    bcs = [{"attribute": a, "kind": "dirichlet", "value": lid} for a in (B, R, T, L)]
    return fluids.solve_navier_stokes(mesh, 1.0 / re, 1.0, None, bcs, lu="eigen", vectorized=True,
                                      newton_rtol=NEWTON_RTOL)


# ------------------------------ projection
def test_initial_projection_verdict_is_recorded_and_gated():
    """The projection's own accuracy is a recorded, gated number, not an ignored boolean."""
    mesh, _, _ = rect(8, 8)
    ts = transient(mesh)
    assert ts.initialize(u0_not_divergence_free, 0.0, True) is True
    d = dict(ts.diagnostics)
    assert "projection_converged" in d, f"the transient diagnostics do not record the projection verdict: {sorted(d)}"
    assert "projection_residual" in d, f"the transient diagnostics do not record the projection residual: {sorted(d)}"
    assert d["projection_converged"] is True
    res = float(d["projection_residual"])
    # Recorded (not the "-1 = not recorded" default) and inside the bound (PROJECTION_MAX above says what it bounds).
    # Measured 4.7e-17 here (578 velocity dofs); a fixed acceptance at 1e-6 is seven orders above the bound.
    assert 0.0 <= res <= PROJECTION_MAX, f"projection_residual {res:.3e} > bound {PROJECTION_MAX:.3e}"


def test_initial_projection_of_the_zero_field_is_accepted():
    """The degenerate case of a field that vanishes: no division by zero, no spurious failure.

    Each block of the residual is compared with the sizes of its own terms, and here every term is zero: the residual
    is exactly zero and so is the measure.
    """
    mesh, _, _ = rect(8, 8)
    ts = transient(mesh)
    assert ts.initialize(lambda X, Y: np.zeros(X.shape + (2,)), 0.0, True) is True
    d = dict(ts.diagnostics)
    assert d["projection_converged"] is True
    assert float(d["projection_residual"]) == 0.0


def u0_stream_function(X, Y):
    """The curl of psi = (x (1 - x) y (1 - y))^2: divergence free, zero on the boundary, not in the P2 space."""
    s = X * (1 - X) * Y * (1 - Y)
    return np.stack([2 * s * X * (1 - X) * (1 - 2 * Y), -2 * s * Y * (1 - Y) * (1 - 2 * X)], axis=-1)


def test_initial_projection_of_a_divergence_free_field_is_accepted():
    """A field that is already divergence free has a continuity right-hand side that is small against its terms.

    A residual relative to that right-hand side is then of order one however accurate the solve is; relative to the
    sizes of the terms (|B| |u0|) it is at round-off. Measured 5.3e-17 on this mesh (the continuity block alone
    1.9e-22). The correction is the interpolation error of the field: 1.4e-5 of the field in the maximum norm here.
    """
    mesh, _, _ = rect(32, 32)
    ts = transient(mesh)
    assert ts.initialize(u0_stream_function, 0.0, True) is True
    d = dict(ts.diagnostics)
    assert d["projection_converged"] is True
    assert 0.0 < float(d["projection_residual"]) <= PROJECTION_ROUNDOFF, d["projection_residual"]


def _perturbed(newton_rtol, eps, n=8, lx=1.0, ly=1.0, lu="eigen"):
    """A transient solver on [0, lx] x [0, ly] whose next projection solve is scaled by 1 + eps (the solver's test
    hook). lu=None is the default backend of the build."""
    mesh, _, _ = rect(n, n, lx, ly)
    fluids = _fluids()
    bcs = [{"attribute": a, "kind": "dirichlet"} for a in (B, R, T, L)]
    kw = {} if lu is None else {"lu": lu}
    ts = fluids.TransientNavierStokes(mesh, MU, RHO, None, bcs, scheme="newton", vectorized=True,
                                      newton_rtol=newton_rtol, **kw)
    ts._c._perturb_projection_solve(float(eps))
    return ts


@pytest.mark.parametrize("lx,n", [(1.0, 8), (10.0, 16)])
def test_initial_projection_with_an_inaccurate_multiplier_is_refused(lx, n):
    """The momentum block is measured, not only the continuity block.

    With the multiplier alone scaled by 1 + 1e-8 (the two-argument form of the solver's test hook) the velocity
    correction, and so the continuity block, is exact, and the whole error is the momentum residual 1e-8 B^T dp.
    Measured 1.3e-9 on the 8x8 square and 1.1e-10 on the 10 x 1 channel (16x16): refused. A measure that looks at the
    continuity block alone sees round-off and accepts both. The interval is the order of magnitude, not a calibration:
    1e-12 is a factor 10 above the bound, and 1e-8 is the perturbation itself, which the measure cannot exceed.
    """
    ts = _perturbed(NEWTON_RTOL, 0.0, n=n, lx=lx)
    ts._c._perturb_projection_solve_blocks(0.0, 1e-8)
    ok = ts.initialize(_scaled(u0_not_divergence_free, lx, 1.0), 0.0, True)
    d = dict(ts.diagnostics)
    res = float(d["projection_residual"])
    assert ok is False and d["projection_converged"] is False, f"a multiplier perturbed by 1e-8 was accepted ({res:.3e})"
    assert 1e-12 < res < 1e-8, res
    # the same solver with the perturbation removed is accepted at round-off
    ts._c._perturb_projection_solve_blocks(0.0, 0.0)
    assert ts.initialize(_scaled(u0_not_divergence_free, lx, 1.0), 0.0, True) is True
    assert 0.0 < float(dict(ts.diagnostics)["projection_residual"]) < PROJECTION_ROUNDOFF


def test_initial_projection_that_misses_its_gate_is_refused_and_cannot_be_stepped_from():
    """An inaccurate projection is reported as a failure that cannot be missed.

    The projection solve is made inaccurate by a known amount: the solver's test hook scales its solution by
    1 + 1e-8, which leaves a backward error of 8.1e-10, almost four decades above the bound (1e-13) and
    below 1e-6. ``initialize`` returns False, the verdict is recorded, every way of advancing from the unprojected
    state raises and names the projection, and the failure clears when a later initialize succeeds. This fails if the
    comparison against the gate is removed or replaced by a fixed acceptance at 1e-6.
    """
    ts = _perturbed(NEWTON_RTOL, 1e-8)
    assert ts.initialize(u0_not_divergence_free, 0.0, True) is False
    d = dict(ts.diagnostics)
    assert d["projection_converged"] is False
    res = float(d["projection_residual"])
    assert PROJECTION_MAX < res < 1e-6, f"the perturbed projection's residual {res:.3e} is not between bound and 1e-6"
    for call in (lambda: ts.advance(0.05), lambda: ts.step_trial(0.05), lambda: ts.adapt(0.1, 1e-3, 1.0, 0.05),
                 lambda: ts.snapshot()):
        with pytest.raises(Exception) as ei:
            call()
        assert "projection" in str(ei.value), str(ei.value)
        # the error names the remedy for a solver that uses STRUMPACK
        assert 'lu="eigen"' in str(ei.value), str(ei.value)
    # a later successful initialize clears the failure: without a projection, and with an accurate one
    assert ts.initialize(u0_not_divergence_free, 0.0, False) is True
    ts.snapshot()
    ts._c._perturb_projection_solve(0.0)
    assert ts.initialize(u0_not_divergence_free, 0.0, True) is True
    assert dict(ts.diagnostics)["projection_converged"] is True
    assert ts.advance(0.05)["converged"]


# The measure of a solve perturbed by eps is K_8 eps on the 8x8 mesh of the unit square: the continuity residual
# eps (B u0 - rhs_p) over || |B| (|u0| + |du|) ||, whose terms are larger than their sum because B differentiates
# (0.68 / N on an N x N mesh; measured 8.060e-10 at eps = 1e-8 with Eigen LU, which every solver of this table uses,
# on the AMD EPYC 7452 and on the Intel Xeon Gold 6330). An unperturbed solve is at round-off (4.7e-17).
K_8 = 0.0806
# (perturbation of the projection solve, accepted?), at every newton_rtol below: the bound does not depend on it.
#   0:      round-off, accepted
#   1e-8:   8.1e-10, refused
#   4e-12:  3.2e-13, refused: the bound is below 3.2e-13 (a bound of 1e-12, or one that follows a loose newton_rtol
#           up, accepts it)
#   4e-13:  3.2e-14, accepted: the bound is above 3.2e-14 (a bound of 1e-14, or one that follows a tight newton_rtol
#           down, refuses it)
# 1e-16 and 1e-20 are below the unit round-off: a bound that followed newton_rtol down refuses every projection there.
GATE_RTOLS = [1e-6, 1e-9, 1e-10, 1e-12, 1e-14, 1e-16, 1e-20]
GATE_EPS = [(0.0, True), (1e-8, False), (4e-12, False), (4e-13, True)]


@pytest.mark.parametrize("rtol", GATE_RTOLS)
@pytest.mark.parametrize("eps,accepted", GATE_EPS)
def test_initial_projection_bound_is_1e_13_at_every_newton_rtol(rtol, eps, accepted):
    """The verdict is the comparison of the backward error with 1e-13, whatever newton_rtol is."""
    ts = _perturbed(rtol, eps)
    ok = ts.initialize(u0_not_divergence_free, 0.0, True)
    d = dict(ts.diagnostics)
    res = float(d["projection_residual"])
    assert ok is accepted and d["projection_converged"] is accepted, (rtol, eps, res)
    assert (res <= PROJECTION_MAX) is accepted, (rtol, eps, res)
    if eps:
        # linear in eps with the measured constant: 5% covers the round-off of the smallest eps (5e-17 against 3e-14)
        assert 0.95 * K_8 * eps < res < 1.05 * K_8 * eps, (eps, res, K_8 * eps)
    else:
        assert 0.0 < res < PROJECTION_ROUNDOFF, f"an unperturbed 8x8 projection is at round-off, got {res:.3e}"


# ------------------------------ projection: the shape of the domain and the length unit
# Channels [0, lx] x [0, 1] on n x n meshes. In a channel of aspect ratio a the multiplier of the projection is of
# size a while its transverse gradient is of size one, so the momentum residual relative to the data is a n times the
# round-off: a gate on a residual relative to the data at 1e-13 refuses the last three (1.3e-13, 3.9e-13, 4.3e-12
# measured in the previous measure of this gate), although their solves are as accurate as the square's.
CHANNELS = [(100.0, 32), (300.0, 32), (1000.0, 32), (1e4, 32), (1e5, 32)]


def _scaled(u0, lx, ly):
    return lambda X, Y: u0(X / lx, Y / ly)


@pytest.mark.parametrize("lu", ["eigen", None])
@pytest.mark.parametrize("lx,n", CHANNELS)
def test_initial_projection_on_a_stretched_channel_is_accepted(lx, n, lu):
    """The accuracy of the projection solve does not depend on the aspect ratio, and neither does the verdict.

    With Eigen's LU: measured 5.2e-17 to 5.9e-17 for aspect ratios 1e2 to 1e5 (the unit square: 5.1e-17), at the
    default newton_rtol and at 1e-6 and 1e-14, and held to round-off. With the default backend (lu=None) the solve is
    held to the verdict only: STRUMPACK measures the same on the AMD EPYC 7452 and 6.1e-15 at 1e5 x 1 on the Intel
    Xeon Gold 6330, accepted on both. The same solve perturbed by 1e-8 is refused at each of them (2.8e-10).
    """
    accurate_max = PROJECTION_ROUNDOFF if lu == "eigen" else PROJECTION_MAX
    for rtol in (1e-6, NEWTON_RTOL, 1e-14):
        ts = _perturbed(rtol, 0.0, n=n, lx=lx, lu=lu)
        ok = ts.initialize(_scaled(u0_not_divergence_free, lx, 1.0), 0.0, True)
        res = float(dict(ts.diagnostics)["projection_residual"])
        assert ok is True, f"{lx:g} x 1, newton_rtol {rtol:g}: an accurate projection was refused (measure {res:.3e})"
        assert 0.0 < res <= accurate_max, (lx, rtol, lu, res)
        ts = _perturbed(rtol, 1e-8, n=n, lx=lx, lu=lu)
        bad = ts.initialize(_scaled(u0_not_divergence_free, lx, 1.0), 0.0, True)
        res = float(dict(ts.diagnostics)["projection_residual"])
        assert bad is False, f"{lx:g} x 1, newton_rtol {rtol:g}: a projection perturbed by 1e-8 was accepted ({res:.3e})"
        assert 2.5e-10 < res < 3.1e-10, (lx, rtol, res)  # measured 2.78e-10 to 2.85e-10 over these channels


# The same domain and the same field in four length units: the mesh, the field's argument and nothing else are scaled.
UNITS = [1e-6, 1e-3, 1.0, 100.0]


@pytest.mark.parametrize("shape,n", [((1000.0, 1.0), 32), ((1.0, 1.0), 16)])
def test_initial_projection_verdict_and_measure_do_not_depend_on_the_length_unit(shape, n):
    """The momentum rows of the projection carry L^2 and the continuity rows L: a measure that adds them weighs the
    two blocks by the length unit, and its verdict follows.

    Each block is measured against its own terms, so the unit cancels. The measure of an accurate solve is round-off
    in every unit (it is not the same number: the pivoting of the factorization depends on the relative size of the
    blocks), and the measure of a solve perturbed by 1e-8 is a property of the discrete problem, the same in every
    unit to the round-off of forming it (relative 1e-6 here: round-off 5e-17 against 2.8e-10; measured identical to
    the four digits printed, 2.839e-10 for the channel).
    """
    got = {}
    for s in UNITS:
        lx, ly = shape[0] * s, shape[1] * s
        ts = _perturbed(NEWTON_RTOL, 0.0, n=n, lx=lx, ly=ly)
        ok = ts.initialize(_scaled(u0_not_divergence_free, lx, ly), 0.0, True)
        res = float(dict(ts.diagnostics)["projection_residual"])
        ts = _perturbed(NEWTON_RTOL, 1e-8, n=n, lx=lx, ly=ly)
        bad = ts.initialize(_scaled(u0_not_divergence_free, lx, ly), 0.0, True)
        got[s] = (ok, res, bad, float(dict(ts.diagnostics)["projection_residual"]))
    assert all(g[0] is True for g in got.values()), f"an accurate projection is refused in some length unit: {got}"
    assert all(0.0 < g[1] < PROJECTION_ROUNDOFF for g in got.values()), got
    assert all(g[2] is False for g in got.values()), f"a perturbed projection is accepted in some length unit: {got}"
    ref = got[1.0][3]
    for s, g in got.items():
        assert abs(g[3] - ref) <= 1e-5 * ref, f"unit {s:g}: the measure of the perturbed solve {g[3]:.6e} != {ref:.6e}"


def test_tightest_published_newton_rtol_passes_the_projection_gate_on_a_32x32_mesh():
    """The tight runs of mms_ns_transient_temporal use newton_rtol = 1e-14, and so may a caller on a finer mesh.

    The projection's backward error on this 32x32 mesh (8450 velocity dofs) is 5.1e-17 at that setting as at every
    other (Eigen LU, AMD EPYC 7452), the bound is 1e-13, and the first step converges to the tight tolerance.
    """
    sens = json.loads(TIGHT_CASE.read_text("utf-8"))
    found = re.findall(r'"newton_rtol_tight":\s*([0-9.eE+-]+)', json.dumps(sens))
    assert found, "mms_ns_transient_temporal.json no longer states newton_rtol_tight"
    rtol = float(found[0])
    assert rtol == 1e-14
    ts = _perturbed(rtol, 0.0, n=32)
    assert ts.initialize(u0_not_divergence_free, 0.0, True) is True
    d = dict(ts.diagnostics)
    res = float(d["projection_residual"])
    assert d["projection_converged"] is True
    assert 0.0 < res <= PROJECTION_ROUNDOFF, f"projection_residual {res:.3e} is not at round-off"
    r = ts.advance(0.025)
    assert r["converged"], r


def test_initial_projection_is_discretely_divergence_free_to_an_absolute_bound():
    """An absolute bound on the discrete divergence, independently assembled.

    The projection solves [M B^T; B 0] with a refined relative residual <= 1e-14 (measured 4e-16 to 7e-16 on these
    meshes), and the right-hand side of this problem has magnitude ~1e-2, so the continuity rows of the projected state are expected near 1e-16; the measured
    value is 1.1e-17 on the meshes below. 1e-14 is three decades above that, eleven decades below the interpolant's
    1.5e-3, and rejects a projection accepted at a fixed 1e-6 (which would leave ~1e-8).
    """
    for n in (4, 8, 16):
        mesh, V, tri = rect(n, n)
        got = {}
        for project in (True, False):
            ts = transient(mesh)
            assert ts.initialize(u0_not_divergence_free, 0.0, project) is True
            got[project] = discrete_divergence(ts.snapshot(), V, tri)
        assert got[False] > 1e-3, f"n={n}: the interpolant's discrete divergence {got[False]:.3e} is too small"
        assert got[True] <= 1e-14, f"n={n}: projected discrete divergence {got[True]:.3e} > 1e-14"


# ------------------------------ per-step LU residual
def test_navier_stokes_history_records_every_step_linear_solve():
    """Every history entry carries its linear solve's refined residual, and the worst value is enforced."""
    s = cavity()
    assert dict(s.diagnostics)["converged"]
    hist = list(s.history)
    assert hist, "the cavity solve produced no history"
    for k, h in enumerate(hist):
        assert "lu_residual" in h, f"history[{k}] ({h.get('kind')}) does not record lu_residual: {sorted(h)}"
        r = float(h["lu_residual"])
        assert r >= 0.0, f"history[{k}] ({h.get('kind')}) lu_residual {r} is the 'not recorded' default"
        assert np.isfinite(r)
        # inside the caller's gate here at every step (the enforced bound is on the final step only, and looser)
        assert r <= NEWTON_RTOL, f"history[{k}] ({h.get('kind')}) lu_residual {r:.3e} > newton_rtol {NEWTON_RTOL:.3e}"
    d = dict(s.diagnostics)
    assert "lu_residual_max" in d, f"the diagnostics do not record the worst per-step LU residual: {sorted(d)}"
    assert "lu_target_missed" in d, f"the diagnostics do not count the steps that missed the target: {sorted(d)}"
    worst = max(float(h["lu_residual"]) for h in hist)
    assert float(d["lu_residual_max"]) == pytest.approx(worst, rel=1e-12, abs=0.0)
    assert int(d["lu_target_missed"]) == sum(1 for h in hist if float(h["lu_residual"]) > LU_REFINE_TARGET)


def test_transient_steps_record_their_linear_solve():
    mesh, _, _ = rect(6, 6)
    ts = transient(mesh)
    assert ts.initialize(u0_not_divergence_free, 0.0, True) is True
    r = ts.advance(0.05)
    assert r["converged"]
    assert "lu_residual" in r, f"a transient step does not record its linear solve's residual: {sorted(r)}"
    assert 0.0 <= float(r["lu_residual"]) <= NEWTON_RTOL
    st = dict(ts.stats)
    assert "lu_residual_max" in st, f"the transient stats do not record the worst LU residual: {sorted(st)}"
    assert 0.0 <= float(st["lu_residual_max"]) <= NEWTON_RTOL


# ------------------------------ dual-norm mass solves
def test_steady_navier_stokes_records_its_dual_norm_mass_solves():
    d = dict(cavity().diagnostics)
    assert "mass_lu_residual_max" in d, f"the dual-norm mass solves are not recorded: {sorted(d)}"
    v = float(d["mass_lu_residual_max"])
    assert 0.0 <= v <= MASS_RESIDUAL_MAX, f"mass_lu_residual_max {v:.3e} > {MASS_RESIDUAL_MAX:.3e}"


def test_transient_records_its_dual_norm_mass_solves():
    mesh, _, _ = rect(6, 6)
    ts = transient(mesh)
    assert ts.initialize(u0_not_divergence_free, 0.0, True) is True
    assert ts.advance(0.05)["converged"]
    d = dict(ts.diagnostics)
    assert "mass_lu_residual_max" in d, f"the dual-norm mass solves are not recorded: {sorted(d)}"
    assert 0.0 <= float(d["mass_lu_residual_max"]) <= MASS_RESIDUAL_MAX


def test_scalar_transport_records_its_dual_norm_mass_solve():
    fluids = _fluids()
    if not hasattr(fluids, "solve_heat"):
        pytest.skip("this build has no scalar transport")
    mesh, _, _ = rect(6, 6)
    bcs = [{"attribute": L, "kind": "dirichlet", "value": 0.0},
           {"attribute": R, "kind": "dirichlet", "value": 1.0},
           {"attribute": B, "kind": "zero"}, {"attribute": T, "kind": "zero"}]
    s = fluids.solve_heat(mesh, 1.0, bcs=bcs)
    d = dict(s.diagnostics)
    assert "mass_lu_residual_max" in d, f"the transport dual-norm mass solve is not recorded: {sorted(d)}"
    assert 0.0 <= float(d["mass_lu_residual_max"]) <= MASS_RESIDUAL_MAX


# ------------------------------ convective rule refusal
@pytest.mark.parametrize("order", [1, 2, 4])
def test_convection_rule_below_the_exact_degree_is_refused(order):
    """P2 velocity on an affine planar mesh: the exact degree is 3p - 1 = 5, so 1, 2 and 4 are under-integration."""
    fluids = _fluids()
    mesh, _, _ = rect(4, 4)
    bcs = [{"attribute": a, "kind": "dirichlet", "value": lid} for a in (B, R, T, L)]
    with pytest.raises(Exception) as ei:
        fluids.solve_navier_stokes(mesh, 0.1, 1.0, None, bcs, lu="eigen", vectorized=True,
                                   quad_order_convection=order)
    assert "quad_order_convection" in str(ei.value), str(ei.value)


def test_convection_rule_at_or_above_the_exact_degree_is_accepted_and_reported():
    fluids = _fluids()
    mesh, _, _ = rect(4, 4)
    bcs = [{"attribute": a, "kind": "dirichlet", "value": lid} for a in (B, R, T, L)]
    for order in (5, 7):
        s = fluids.solve_navier_stokes(mesh, 0.1, 1.0, None, bcs, lu="eigen", vectorized=True,
                                       quad_order_convection=order)
        assert list(dict(s.diagnostics)["quad_orders"])[1] == order
    s0 = fluids.solve_navier_stokes(mesh, 0.1, 1.0, None, bcs, lu="eigen", vectorized=True)
    assert list(dict(s0.diagnostics)["quad_orders"])[1] == 5  # 3p - 1 for P2 on affine triangles


# ------------------------------ shipped texts
CAVITY_CASES = ("selfconv_ns_cavity_continuation.json", "xcode_cavity_re1000.json")


def _case_picard_count(name):
    text = json.loads((CASES / name).read_text("utf-8"))["run"]["parameters"]["nonlinear_solver"]["start"]
    m = re.search(r"n_picard\s*=\s*(\d+)", text)
    assert m, f"{name}: run.parameters.nonlinear_solver.start does not state n_picard: {text[:160]}"
    return int(m.group(1)), text


def _runner_picard_cavity():
    """N_PICARD_CAVITY, read from the runner's source: the runner imports the fluids extension, which a build without
    MPI lacks, and this test must run there too."""
    import ast
    tree = ast.parse(F2A_RUNNER.read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "N_PICARD_CAVITY"
                                                for t in node.targets):
            v = node.value
            assert isinstance(v, ast.Constant) and type(v.value) is int, (
                "benchmarks/fluids/cases/f2a_ns.py: N_PICARD_CAVITY must be assigned an int literal")
            return v.value, tree
    raise AssertionError("benchmarks/fluids/cases/f2a_ns.py must name the cavity path's Picard count "
                         "(N_PICARD_CAVITY) instead of the literal in _cavity()")


def test_cavity_runner_uses_its_named_picard_count():
    import ast
    _, tree = _runner_picard_cavity()
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_cavity"), None)
    assert fn is not None, "benchmarks/fluids/cases/f2a_ns.py has no _cavity()"
    uses = [k for c in ast.walk(fn) if isinstance(c, ast.Call)
            for k in c.keywords if k.arg == "n_picard" and isinstance(k.value, ast.Name)
            and k.value.id == "N_PICARD_CAVITY"]
    assert uses, "_cavity() must pass n_picard=N_PICARD_CAVITY, not a literal"


@pytest.mark.parametrize("name", CAVITY_CASES)
def test_cavity_case_text_states_the_picard_count_the_runner_uses(name):
    """The shipped case definition must describe the shipped run (AGENTS.md: an independent reference, justified)."""
    runner, _ = _runner_picard_cavity()
    stated, text = _case_picard_count(name)
    assert stated == runner, (f"{name} says n_picard = {stated}; the runner uses {runner}: {text[:200]}")


def test_transient_runner_reads_the_projection_verdict():
    """A failed initialize() must make the run fail, the way a missed step gate does."""
    src = F3A_RUNNER.read_text("utf-8")
    bare = [ln.strip() for ln in src.splitlines() if re.match(r"^\s*(ts|tsw|tsr|tight)\.initialize\(", ln)]
    assert not bare, ("benchmarks/fluids/cases/f3a_transient.py discards initialize()'s verdict at "
                      f"{len(bare)} call site(s), e.g. {bare[:2]}")


def test_fluids_doc_states_what_was_measured_about_strumpack():
    text = FLUIDS_DOC.read_text("utf-8")
    assert "occasionally returned an inaccurate solution" not in text, (
        "docs/physics/fluids.md still calls the STRUMPACK failure occasional; it was measured to be deterministic "
        "(bitwise reproducible) and matrix-dependent")
    assert "STRUMPACK" in text
    assert re.search(r"deterministic", text), ("docs/physics/fluids.md does not say that the failure is "
                                               "deterministic and matrix-dependent")


@pytest.mark.parametrize("token", ["projection_residual", "lu_residual", "mass_lu_residual_max",
                                   "quad_order_convection"])
def test_user_visible_additions_are_in_the_changelog(token):
    text = CHANGELOG.read_text("utf-8")
    head, _, _ = text.partition("## 0.1.0")
    assert "## Unreleased" in head
    assert token in head, f'CHANGELOG.md (notes since 0.1.0) does not mention {token}'


@pytest.mark.parametrize("token", ["projection_residual", "mass_lu_residual_max"])
def test_user_visible_additions_are_documented(token):
    assert token in FLUIDS_DOC.read_text("utf-8") or token in LLMS.read_text("utf-8"), (
        f"{token} appears in neither docs/physics/fluids.md nor llms.txt")


# ------------------------------ multi-rank ctest
def test_cpp_rank_invariance_comparison_is_registered_as_a_multi_rank_ctest():
    """cpp/tests/test_fluids_navier_stokes.cpp compares a multi-rank solve with a one-rank one, and returns without a
    check on one rank.

    Every C++ test is registered as a single process, so that comparison is asserted only when somebody launches the
    binary by hand. It needs a registered multi-rank ctest to be run at all.
    """
    cm = CMAKELISTS.read_text("utf-8")
    pat = re.compile(r"add_test\s*\(\s*NAME\s+(\S*test_fluids_navier_stokes\S*)(.*?)\)", re.S)
    multi = [(n, body) for n, body in pat.findall(cm)
             if re.search(r"run_mpi_ctest\.sh\s+([2-9]|\d\d+)\b", body)]
    assert multi, ("CMakeLists.txt registers no multi-rank ctest for test_fluids_navier_stokes, so its "
                   "rank-invariance comparison never runs under ctest")
    assert (ROOT / "cmake" / "run_mpi_ctest.sh").is_file()
    assert "SKIP_RETURN_CODE 77" in cm, "a launcher that cannot start 4 ranks must skip the test, not fail it"
