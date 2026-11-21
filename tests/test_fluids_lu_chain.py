"""The sparse LU retry chain at every fluids caller, through the public Python boundary.

Every fluids direct solve (Stokes 2D and 3D, the steady and transient Navier-Stokes steps and the initial Leray
projection, Boussinesq, GLS, the scalar transport Jacobian, and the dual-norm mass solves of Navier-Stokes, MHD,
Boussinesq and transport) builds its sparse LU from one recorded chain of configurations (sparse_lu.hpp,
default_chain) and accepts a stage on its OWN measure: the projection on its gate (projection_residual <= 1e-13, a
collective function of the stage's solution), every other caller on the refined relative residual against the gate
it has (or, for the callers without one, a recorded new threshold). The C++ side is cpp/tests/test_fluids_lu_callers.cpp
(1 and 2 ranks) and cpp/tests/test_fluids_lu_chain.cpp.

  - the keyword: every entry point takes lu_retry="auto" | "off"; "off" is the first stage alone
  - the records: diagnostics lu_stage, lu_backward_error, lu_attempts (one dict per stage tried), the counts of the
      call (lu_solves, lu_escalated, lu_escalated_final, lu_attempts_total, lu_rejected, lu_failed, lu_accepted_by,
      lu_rejected_seconds, mass_lu_escalated), history lu_attempts and lu_stage per step, projection_lu_attempts,
      TransientNavierStokes.stats summed per step (reset by reset_stats, the projection never booked to a step)
  - every caller escalates and records it (the stage-0 hook with the test-only chain [eigen, eigen], so that this
      runs in every build), every strategy of the steady solver included, and a one-stage chain still tests its solve
  - kGarbage is gone: a semi-implicit step whose first stage misses its gate (1e-7 > 1e-11) escalates and converges
  - a rescued stage leaves no failure text behind
  - lu_stage and lu_backend name the stage that produced the solution; solver stays the configured backend
  - the projection: its attempts are accepted on projection_residual itself, bit for bit; a refused projection's
      error names the attempts and the pin; the bordered system at a small length unit passes the kernel probe; and
      the cases where today's default chain fails while one of its stages passes the gate are rescued, against the
      unit-covariant reference
  - the measured default lists (STRUMPACK builds), through lu_retry="auto" and recorded in lu_attempts: 2D
      strumpack:mc64-5, strumpack:mc64-2, eigen (no stage without matching); 3D strumpack:none, strumpack:mc64-5,
      eigen; a projection that the first stage misses is rescued by MC64 job 2, which ends the walk
  - the aliases lu_fallbacks, lu_fallback_backend, lu_matching_retry(_reason) are kept, derived, for one release
  - the transient runner's note and the shipped texts (CHANGELOG "Unreleased", docs/physics/fluids.md, llms.txt,
      sparse_lu.hpp): the measured 2D list, and the projection numbers attributed to the backend and the pressure
      treatment that produced them

The test hooks are process-wide private bindings (_core._fluids._lu_*_for_testing); an autouse fixture resets them
around every test.
"""
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
FLUIDS_DOC = ROOT / "docs" / "physics" / "fluids.md"
LLMS = ROOT / "llms.txt"
SPARSE_LU_HPP = ROOT / "cpp" / "include" / "axionmultiphysics" / "fluids" / "sparse_lu.hpp"
CASES_DIR = ROOT / "benchmarks" / "fluids" / "cases"

B, R, T, L = 1, 2, 3, 4
RHO, MU = 1.25, 0.05
NEWTON_RTOL = 1e-12
PROJECTION_MAX = 1e-13       # kProjectionResidualMax (sparse_lu.hpp)
FINAL_SOLVE_MAX = 1e-11      # kFinalSolveResidualMax (sparse_lu.hpp)
ATTEMPT_KEYS = {"stage", "outcome", "reason", "measure", "bound", "rel_residual_before", "rel_residual",
                "gate_residual", "backward_error", "constraint_ratios", "refine_steps", "krylov_iterations",
                "factor_nnz", "index_bits", "analyze_seconds", "factor_seconds", "solve_seconds"}
COUNT_KEYS = ["lu_solves", "lu_escalated", "lu_escalated_final", "lu_attempts_total", "lu_rejected", "lu_failed"]


def _fluids():
    return pytest.importorskip("axionmultiphysics.fluids", exc_type=ImportError)


def _hooks():
    _fluids()
    from axionmultiphysics import _core
    return _core._fluids


@pytest.fixture(autouse=True)
def hooks():
    """The private test hooks of the sparse LU chain, reset before and after every test (they are process-wide)."""
    f = _hooks()
    f._lu_reset_hooks_for_testing()
    yield f
    f._lu_reset_hooks_for_testing()


def _strumpack_available():
    try:
        return bool(_fluids().lu_backend_available("strumpack"))
    except BaseException:  # noqa: BLE001  (no fluids module: every test is skipped by the fixture anyway)
        return False


needs_strumpack = pytest.mark.skipif(not _strumpack_available(), reason="needs a build with STRUMPACK")


def rect(nx, ny, lx=1.0, ly=1.0):
    """A structured right-triangle mesh of [0, lx] x [0, ly] (the mesh of tests/test_fluids_trust.py)."""
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
    return fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32),
                                   np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32))


def lid(X, Y):
    """The singular lid: u = (1, 0) on the lid except at its corner vertices."""
    top = (np.abs(Y - 1.0) < 1e-12) & (X > 1e-12) & (X < 1.0 - 1e-12)
    return np.stack([np.where(top, 1.0, 0.0), 0.0 * X], axis=-1)


def lid_bcs():
    return [{"attribute": a, "kind": "dirichlet", "value": lid} for a in (B, R, T, L)]


def walls():
    return [{"attribute": a, "kind": "dirichlet"} for a in (B, R, T, L)]


def field(lx=1.0, ly=1.0):
    """The initial field of the projection tests: not divergence free, zero on the boundary."""
    def u0(X, Y):
        x, y = X / lx, Y / ly
        s = x * (1 - x) * y * (1 - y)
        return np.stack([s * (1 + x), s * y], axis=-1)
    return u0


def cavity(strategy="newton", **kw):
    kw.setdefault("lu", "eigen")
    return _fluids().solve_navier_stokes(rect(8, 8), 0.01, 1.0, None, lid_bcs(), vectorized=True,
                                         newton_rtol=NEWTON_RTOL, strategy=strategy, **kw)


def stokes(**kw):
    kw.setdefault("lu", "eigen")
    return _fluids().solve_stokes(rect(8, 8), 1.0, None, lid_bcs(), **kw)


def transient(scheme="newton", n=8, lx=1.0, ly=1.0, **kw):
    kw.setdefault("lu", "eigen")
    return _fluids().TransientNavierStokes(rect(n, n, lx, ly), MU, RHO, None, walls(), scheme=scheme, vectorized=True,
                                           newton_rtol=NEWTON_RTOL, **kw)


def outcomes(attempts):
    return [(a["stage"], a["outcome"]) for a in attempts]


def two_eigen(hooks):
    """The test-only default chain [eigen, eigen]: every caller escalates to a second, identical stage, in any build."""
    hooks._lu_default_chain_for_testing(["eigen", "eigen"])


def check_attempt_record(a):
    assert set(a) >= ATTEMPT_KEYS, f"an attempt record lacks {sorted(ATTEMPT_KEYS - set(a))}"
    assert isinstance(a["stage"], str) and isinstance(a["outcome"], str) and isinstance(a["reason"], str)
    assert a["outcome"] in ("accepted", "threw", "missed_target", "refused", "not_available"), a["outcome"]
    assert isinstance(a["constraint_ratios"], list)


# ------------------------------------------------------------------------------------------------ the keyword
def _entry_points():
    import warnings
    fluids = _fluids()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the beta module's import warning
        import axionmultiphysics.transport as tr
    return {"solve_stokes": fluids.solve_stokes, "solve_navier_stokes": fluids.solve_navier_stokes,
            "solve_heat": fluids.solve_heat, "solve_boussinesq": fluids.solve_boussinesq,
            "solve_stokes_3d": fluids.solve_stokes_3d, "gls_problem": fluids.gls_problem,
            "TransientNavierStokes": fluids.TransientNavierStokes.__init__,
            "ScalarTransport.solve": tr.ScalarTransport.solve}


@pytest.mark.parametrize("name", ["solve_stokes", "solve_navier_stokes", "solve_heat", "solve_boussinesq",
                                  "solve_stokes_3d", "gls_problem", "TransientNavierStokes", "ScalarTransport.solve"])
def test_every_entry_point_takes_lu_retry(name):
    """lu= keeps naming the first backend; lu_retry="auto" (the default chain, the default) or "off" (its first stage
    only: one attempt, for measurements and for reproducing an older run) is one keyword on every entry point."""
    p = inspect.signature(_entry_points()[name]).parameters
    assert "lu_retry" in p, f"{name} has no lu_retry keyword"
    assert p["lu_retry"].default == "auto", f"{name}: lu_retry defaults to {p['lu_retry'].default!r}, expected 'auto'"


def test_lu_retry_refuses_an_unknown_value():
    with pytest.raises(ValueError, match=r"auto.*off|off.*auto"):
        stokes(lu_retry="sometimes")


def test_lu_retry_off_is_the_first_stage_alone(hooks):
    """With the chain [eigen, eigen] and the first stage of the next solve forced to miss: "auto" escalates to the
    second stage, "off" records the one attempt and returns its solution (the Stokes gate then decides on it; the
    forced miss is the hook's, the solution itself is accurate). Reference: the hook's count."""
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("missed_target", 1)
    d = dict(stokes(lu_retry="auto").diagnostics)
    assert outcomes(d["lu_attempts"]) == [("eigen", "missed_target"), ("eigen", "accepted")], d["lu_attempts"]
    hooks._lu_force_outcome_for_testing("missed_target", 1)
    d = dict(stokes(lu_retry="off").diagnostics)
    assert outcomes(d["lu_attempts"]) == [("eigen", "missed_target")], d["lu_attempts"]
    assert d["lu_attempts"][0]["reason"] == "test hook"
    assert d["converged"] is True and d["lu_stage"] == "eigen"


# ------------------------------------------------------------------------------------------------ the records
def test_stokes_records_its_attempt_on_its_own_measure():
    """Stokes accepts a stage on today's retry trigger: the relative residual over all rows of the CSR against
    direct_tol (kept: the masked gate residual would change when the 3D retry fires). The attempt record carries
    every field of LUAttempt; lu_stage and lu_backward_error are the accepted attempt's. An accurate Eigen solve has a
    blockwise backward error at round-off (about 1e-16 on matrices of this size; 1e-14 leaves a factor 100)."""
    s = stokes(direct_tol=1e-10)
    d = dict(s.diagnostics)
    assert d["converged"] is True
    at = d["lu_attempts"]
    assert isinstance(at, list) and len(at) == 1, at
    check_attempt_record(at[0])
    a = at[0]
    assert (a["stage"], a["outcome"]) == ("eigen", "accepted")
    assert a["bound"] == 1e-10 and a["measure"] == d["lu_residual"], (a["measure"], a["bound"], d["lu_residual"])
    assert "relative residual" in a["reason"] and "<=" in a["reason"], a["reason"]
    assert d["lu_stage"] == "eigen" and d["lu_backend"] == "eigen" and d["solver"] == "direct:eigen"
    assert d["lu_backward_error"] == a["backward_error"] and 0.0 <= a["backward_error"] < 1e-14, a["backward_error"]
    # the caller's row blocks: the pinned pressure row of the enclosed cavity is a constraint row (block id -2)
    assert len(a["constraint_ratios"]) == 1, a["constraint_ratios"]
    assert a["index_bits"] == 32 and a["factor_nnz"] > 0 and a["factor_seconds"] >= 0.0


# ------------------------------------------------------------------------------------------------ every caller
def test_stokes_2d_escalates_and_names_the_stage(hooks):
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("refused", 1)
    d = dict(stokes().diagnostics)
    assert d["converged"] is True
    assert outcomes(d["lu_attempts"]) == [("eigen", "refused"), ("eigen", "accepted")], d["lu_attempts"]
    assert d["lu_attempts"][0]["reason"] == "test hook"
    assert d["lu_stage"] == "eigen" and d["lu_backend"] == "eigen" and d["solver"] == "direct:eigen"


@pytest.fixture
def mesh3d():
    fluids = _fluids()
    sys.path.insert(0, str(CASES_DIR))
    try:
        import f1c_stokes as R3
    finally:
        sys.path.remove(str(CASES_DIR))
    fm = R3.build_mesh({"family": "kuhn", "N": 2})
    faces = R3.Faces({"family": "kuhn", "N": 2})
    bf, ba = R3.boundary_faces(fm, faces)
    mesh = fluids.Mesh3D.from_arrays(np.ascontiguousarray(fm.V, float), np.ascontiguousarray(fm.E, np.int32), bf, ba)
    bcs = [{"attribute": faces.attr(nm), "kind": "dirichlet",
            "value": (lambda X, Y, Z: np.stack([Y * (1 - Y), 0 * X, 0 * X], -1))} for nm in faces.names]
    return mesh, bcs


def test_stokes_3d_escalates_and_keeps_its_aliases(hooks, mesh3d):
    """The 3D solve is on the chain (its matching-free first stage, then the former flip as the second stage): the
    escalation is recorded, and lu_matching_retry / lu_matching_retry_reason are kept for one release as derived
    aliases (True when the solution is not the first stage's; the first stage's reason)."""
    fluids = _fluids()
    mesh, bcs = mesh3d
    d0 = dict(fluids.solve_stokes_3d(mesh, 1.0, None, bcs, lu="eigen").diagnostics)
    assert d0["converged"] is True and outcomes(d0["lu_attempts"]) == [("eigen", "accepted")]
    assert d0["lu_matching_retry"] is False and d0["lu_matching_retry_reason"] == ""
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("refused", 1)
    d = dict(fluids.solve_stokes_3d(mesh, 1.0, None, bcs, lu="eigen").diagnostics)
    assert d["converged"] is True
    assert outcomes(d["lu_attempts"]) == [("eigen", "refused"), ("eigen", "accepted")], d["lu_attempts"]
    assert d["lu_stage"] == "eigen" and d["lu_backend"] == "eigen"
    assert d["lu_matching_retry"] is True and d["lu_matching_retry_reason"] == "test hook", (
        d["lu_matching_retry"], d["lu_matching_retry_reason"])
    # MC64 matching is a STRUMPACK option: a solution from an Eigen stage reports the
    # configured lu_matching (False in the 3D API), the value every 3D record carried before the chain.
    assert d0["lu_matching"] is False and d["lu_matching"] is False, (d0["lu_matching"], d["lu_matching"])


def _counts(d):
    return {k: d[k] for k in COUNT_KEYS + ["lu_accepted_by", "mass_lu_escalated"]}


@pytest.mark.parametrize("strategy", ["newton", "picard", "ptc"])
def test_steady_navier_stokes_escalates_every_strategy_per_step(hooks, strategy):
    """Every step solve of every steady strategy is on the chain (today's lin_solve_ gave the steady strategies one
    attempt, accepted whatever it returned). Four refusals of a stage 0: the velocity and the pressure mass LUs are
    factorized when the solver is built, before any step (the lift's dual norm needs them), so the four are those of
    the two mass LUs, the Stokes start and the second step. Reference: the hook's count."""
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("refused", 4)
    s = cavity(strategy)
    d = dict(s.diagnostics)
    h = s.history
    assert d["converged"] is True, d["failure"]
    assert len(h) >= 3 and h[0]["kind"] == "stokes"
    assert [e["lu_attempts"] for e in h] == [2, 2] + [1] * (len(h) - 2), [e["lu_attempts"] for e in h]
    assert all(e["lu_stage"] == "eigen" for e in h)
    n = len(h)
    assert _counts(d) == {"lu_solves": n, "lu_escalated": 2, "lu_escalated_final": 0, "lu_attempts_total": n + 2,
                          "lu_rejected": 2, "lu_failed": 0, "lu_accepted_by": {"eigen": n}, "mass_lu_escalated": 2}, (
        _counts(d))
    assert d["lu_rejected_seconds"] == 0.0  # a refused stage is never factorized
    # the aliases, derived from this call's counts (kept for one release)
    assert d["lu_fallbacks"] == 2 and d["lu_fallback_backend"] == "eigen"


def test_steady_navier_stokes_counts_belong_to_the_call(hooks):
    """The counts are those of this solve() call: a second solve without a forced rejection reports none (today's
    lu_fallbacks was a counter over the solver's lifetime)."""
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("refused", 3)
    d1 = dict(cavity().diagnostics)
    d2 = dict(cavity().diagnostics)
    assert d1["lu_escalated"] == 1 and d1["mass_lu_escalated"] == 2 and d1["lu_fallbacks"] == 1
    assert d2["lu_escalated"] == 0 and d2["mass_lu_escalated"] == 0 and d2["lu_fallbacks"] == 0
    assert d2["lu_fallback_backend"] == ""


def test_mhd_mass_solve_is_on_the_chain(hooks):
    """The potential's mass LU of inductionless MHD is factorized after the velocity and pressure ones, when the
    solver is built: with four refusals the three mass LUs and the first step (the Stokes start) escalate."""
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("refused", 4)
    fluids = _fluids()
    mesh = rect(8, 8)
    s = fluids.solve_navier_stokes(
        mesh, 0.5, 0.0, force=lambda x, y: (0.0, 0.0, 1.0), bcs=walls(), geometry="planar25d", form="laplacian",
        lu="eigen", mhd={"sigma": 2.0, "B": lambda x, y: (0.0, 3.0, 0.0), "E_z": 0.0,
                         "electric": [{"attribute": a, "kind": "insulating"} for a in (B, R, T, L)]})
    d = dict(s.diagnostics)
    assert d["converged"] is True, d["failure"]
    assert d["mass_lu_escalated"] == 3, d["mass_lu_escalated"]
    h = [e["lu_attempts"] for e in s.history]
    assert h == [2] + [1] * (len(h) - 1) and d["lu_escalated"] == 1, (h, d["lu_escalated"])


def test_boussinesq_steps_and_mass_solves_are_on_the_chain(hooks):
    """Boussinesq: the velocity and pressure mass LUs (built with the flow solver), then the temperature's mass LU
    (factorized at the lift's dual norm), then the first Newton step: four refusals escalate all four. Its step
    threshold max(newton_rtol, 1e-11) is new (no gate on its LU before), and recorded."""
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("refused", 4)
    fluids = _fluids()
    p = fluids.natconv_cavity_parameters(1e3, 0.71)
    s = fluids.solve_boussinesq(
        rect(4, 4), p["mu"], p["rho0"], p["cp"], p["k"], p["beta_g"], T_ref=0.0, gravity="-y",
        velocity_bcs=walls(),
        temperature_bcs=[dict(attribute=L, kind="dirichlet", value=0.5),
                         dict(attribute=R, kind="dirichlet", value=-0.5),
                         dict(attribute=B, kind="zero"), dict(attribute=T, kind="zero")], lu="eigen")
    d = dict(s.diagnostics)
    assert d["converged"] is True, d["failure"]
    assert d["mass_lu_escalated"] == 3 and d["lu_escalated"] == 1, (d["mass_lu_escalated"], d["lu_escalated"])
    h = s.history
    assert h[0]["lu_attempts"] == 2 and all(e["lu_attempts"] == 1 for e in h[1:]), [e["lu_attempts"] for e in h]
    last = d["lu_attempts"][-1]
    assert last["outcome"] == "accepted" and last["bound"] == max(NEWTON_RTOL, FINAL_SOLVE_MAX), last


def _gls(mode="navier_stokes", **kw):
    fluids = _fluids()
    sys.path.insert(0, str(CASES_DIR))
    try:
        import f5_gls_check as C5
    finally:
        sys.path.remove(str(CASES_DIR))
    V, Tr = C5.graded_mesh(6, 6, 20.0)
    bnd = np.array(sorted({int(v) for e in C5.C1.boundary_edges(Tr) for v in e}), dtype=np.int32)
    pt = C5.Patch()
    return fluids.gls_problem(V, Tr, mode=mode, form="stress", mu=pt.mu, rho=pt.rho,
                              force=pt.f("sym", convection=(mode != "stokes")),
                              dirichlet=(bnd, pt.u(V[bnd, 0], V[bnd, 1])), lu="eigen", **kw)


def test_gls_steps_are_on_the_chain(hooks):
    """GLS (its mass solves are sparse Cholesky, not the LU): one refusal escalates its first step."""
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("refused", 1)
    g = _gls()
    d = dict(g.diagnostics)
    assert d["converged"] is True, d["failure"]
    h = g.history
    assert [e["lu_attempts"] for e in h] == [2] + [1] * (len(h) - 1), [e["lu_attempts"] for e in h]
    assert all(e["lu_stage"] == "eigen" for e in h)
    assert d["lu_escalated"] == 1 and d["lu_attempts_total"] == d["lu_solves"] + 1 == len(h) + 1, _counts_partial(d)
    # its threshold: as Boussinesq, max(newton_rtol, kFinalSolveResidualMax), new and recorded
    assert d["lu_attempts"][-1]["bound"] == max(NEWTON_RTOL, FINAL_SOLVE_MAX), d["lu_attempts"][-1]


def _counts_partial(d):
    return {k: d.get(k) for k in COUNT_KEYS}


def test_scalar_transport_jacobian_and_mass_solves_are_on_the_chain(hooks):
    """Scalar transport: the mass LU is factorized at the lift's dual norm, before the first Jacobian solve; a linear
    problem needs one Jacobian solve. Two refusals escalate both, and the last (only) step solve escalated."""
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("refused", 2)
    fluids = _fluids()
    s = fluids.solve_heat(rect(6, 6), 1.0, source=1.0, bcs=[dict(attribute=a, kind="dirichlet", value=0.0)
                                                           for a in (B, R, T, L)], lu="eigen")
    d = dict(s.diagnostics)
    assert d["converged"] is True, d["failure"]
    assert d["mass_lu_escalated"] == 1, d["mass_lu_escalated"]
    assert (d["lu_solves"], d["lu_escalated"], d["lu_escalated_final"], d["lu_attempts_total"]) == (1, 1, 1, 2), (
        _counts_partial(d))
    assert outcomes(d["lu_attempts"]) == [("eigen", "refused"), ("eigen", "accepted")]
    assert d["lu_stage"] == "eigen"
    # its threshold: max(1e-2 rtol, kFinalSolveResidualMax) at the default rtol 1e-12 (plan section 2.2; new, recorded)
    assert d["lu_attempts"][1]["bound"] == max(1e-2 * 1e-12, FINAL_SOLVE_MAX), d["lu_attempts"][1]


# ------------------------------------------------------------------------------------------------ run counts
def test_transient_run_counts_add_up_per_step_and_reset(hooks):
    """Two steps with one forced rejection each: the run's counts are summed per step (today's run stats assigned the
    solver-lifetime counter), each step's diagnostics report its own call, and reset_stats() clears them.
    Reference: the hook's count."""
    two_eigen(hooks)
    ts = transient()
    assert ts.initialize(field(), 0.0, True) is True
    per_call = []
    for forced in (1, 1, 0):
        if forced:
            hooks._lu_force_outcome_for_testing("refused", forced)
        r = ts.advance(0.05)
        assert r["converged"] is True, r["failure"]
        d = dict(ts.diagnostics)
        per_call.append({k: d[k] for k in COUNT_KEYS + ["lu_fallbacks"]})
    assert [c["lu_escalated"] for c in per_call] == [1, 1, 0], per_call
    assert [c["lu_fallbacks"] for c in per_call] == [1, 1, 0], "lu_fallbacks is no longer cumulative"
    st = dict(ts.stats)
    solves = sum(c["lu_solves"] for c in per_call)
    assert solves >= 3 and st["lu_solves"] == solves, (st, per_call)
    assert st["lu_escalated"] == 2 and st["lu_attempts_total"] == solves + 2 and st["lu_rejected"] == 2, st
    assert st["lu_failed"] == 0 and st["lu_accepted_by"] == {"eigen": solves}, st
    assert st["lu_fallbacks"] == 2 and st["lu_fallback_backend"] == "eigen", st
    ts.reset_stats()
    st = dict(ts.stats)
    assert all(st[k] == 0 for k in COUNT_KEYS) and st["lu_accepted_by"] == {} and st["lu_rejected_seconds"] == 0.0, st
    assert st["lu_fallbacks"] == 0 and st["lu_fallback_backend"] == "", st


def test_projection_escalation_is_its_own_and_not_the_first_step(hooks):
    """A rejection forced in the initial projection appears in projection_lu_attempts and in no step count."""
    two_eigen(hooks)
    ts = transient()
    hooks._lu_force_outcome_for_testing("refused", 1)
    assert ts.initialize(field(), 0.0, True) is True
    d = dict(ts.diagnostics)
    assert outcomes(d["projection_lu_attempts"]) == [("eigen", "refused"), ("eigen", "accepted")], d
    r = ts.advance(0.05)
    assert r["converged"] is True
    d = dict(ts.diagnostics)
    st = dict(ts.stats)
    assert d["lu_escalated"] == 0 and st["lu_escalated"] == 0 and st["lu_fallbacks"] == 0, (d["lu_escalated"], st)
    assert st["lu_attempts_total"] == st["lu_solves"], st
    assert len(d["projection_lu_attempts"]) == 2, "the projection's attempts are those of the last projection"


# ------------------------------------------------------------------------------------------------ the step's own bound
def test_semi_implicit_step_escalates_a_solve_above_its_gate(hooks):
    """A semi-implicit step's first stage perturbed by a relative 1e-7 (the LU test hook, stage 0 only) has a refined
    relative residual of about 1e-7: below today's floor kGarbage = 1e-6 (accepted, then the step missed its gate
    max(newton_rtol, 1e-11)), above the step's own bound. It escalates, and the step converges at its gate.
    Reference: the size of the perturbation (the residual of (1 + eps) x is -eps b up to round-off)."""
    two_eigen(hooks)
    ts = transient(scheme="semi")
    assert ts.initialize(field(), 0.0, True) is True
    hooks._lu_perturb_solution_for_testing(1e-7)
    r = ts.advance(0.05)
    hooks._lu_perturb_solution_for_testing(0.0)
    assert r["converged"] is True, r["failure"]
    d = dict(ts.diagnostics)
    at = d["lu_attempts"]
    assert outcomes(at) == [("eigen", "missed_target"), ("eigen", "accepted")], at
    gate = max(max(NEWTON_RTOL, 1e-11), FINAL_SOLVE_MAX)
    assert at[0]["bound"] == gate and at[1]["bound"] == gate, (at[0]["bound"], gate)
    assert 0.5e-7 < at[0]["measure"] < 2e-7, at[0]["measure"]
    assert at[0]["measure"] == at[0]["rel_residual"] and "relative residual" in at[0]["reason"]
    assert at[1]["measure"] <= gate
    assert (d["lu_solves"], d["lu_escalated"], d["lu_escalated_final"], d["lu_rejected"]) == (1, 1, 1, 1), (
        _counts_partial(d))
    spent = at[0]["analyze_seconds"] + at[0]["factor_seconds"] + at[0]["solve_seconds"]
    assert d["lu_rejected_seconds"] == pytest.approx(spent, rel=1e-9, abs=1e-12) and spent > 0, (
        d["lu_rejected_seconds"], spent)


# ------------------------------------------------------------------------------------------------ one-stage chains
def test_a_one_stage_chain_still_tests_its_solve(hooks):
    """No "accepted because the chain had one entry": with lu="eigen" the default chain is [eigen], and every step
    solve perturbed by 1e-7 (above the step bound max(newton_rtol, 1e-11)) is recorded as missed, counted in lu_failed,
    and the solve reports a failure (the final direct solve misses its bound). Reference: the size of the
    perturbation."""
    hooks._lu_perturb_solution_for_testing(1e-7)
    s = cavity()
    hooks._lu_perturb_solution_for_testing(0.0)
    d = dict(s.diagnostics)
    assert d["converged"] is False
    assert d["lu_solves"] >= 1 and d["lu_failed"] == d["lu_solves"] and d["lu_escalated"] == 0, _counts_partial(d)
    assert outcomes(d["lu_attempts"]) == [("eigen", "missed_target")], d["lu_attempts"]
    assert all(e["lu_attempts"] == 1 for e in s.history)


# ------------------------------------------------------------------------------------------------ failure texts
def test_a_rescued_stage_leaves_no_failure_text(hooks):
    """Stage 0 of the first Newton solve of a step is forced to throw and stage 1 rescues it; the step then fails for
    an unrelated reason (newton_maxit = 1 where the step needs 3 Newton iterations: measured 8.9e-4, 7.2e-10, 6.3e-16).
    The failure is that reason, not the stale text of the rejected stage."""
    ctl = transient(newton_maxit=30)
    assert ctl.initialize(field(), 0.0, True) is True
    assert ctl.advance(0.05)["iterations"] >= 2, "the control step converges in one Newton iteration"
    two_eigen(hooks)
    ts = transient(newton_maxit=1)
    assert ts.initialize(field(), 0.0, True) is True
    hooks._lu_force_outcome_for_testing("threw", 1)
    r = ts.advance(0.05)
    assert r["converged"] is False
    assert r["failure"].startswith("Newton budget exhausted"), r["failure"]
    d = dict(ts.diagnostics)
    assert d["failure"].startswith("Newton budget exhausted"), d["failure"]
    assert outcomes(d["lu_attempts"]) == [("eigen", "threw"), ("eigen", "accepted")], d["lu_attempts"]
    assert "test hook" in d["lu_attempts"][0]["reason"]


# ------------------------------------------------------------------------------------------------ the stage's names
@needs_strumpack
def test_lu_stage_and_backend_name_the_stage_that_solved(hooks):
    """An escalation from STRUMPACK to Eigen: lu_stage and lu_backend name Eigen, the stage that produced the solution;
    solver stays the configured backend (documented as such)."""
    hooks._lu_default_chain_for_testing(["strumpack:mc64-5", "eigen"])
    hooks._lu_force_outcome_for_testing("refused", 1)
    d = dict(stokes(lu="strumpack").diagnostics)
    assert d["converged"] is True
    assert outcomes(d["lu_attempts"]) == [("strumpack:mc64-5", "refused"), ("eigen", "accepted")], d["lu_attempts"]
    assert d["lu_stage"] == "eigen" and d["lu_backend"] == "eigen", (d["lu_stage"], d["lu_backend"])
    assert d["solver"] == "direct:strumpack", d["solver"]


# ------------------------------------------------------------------------------------------------ the default lists
# Measured on the systems of the published fluids cases and of the projection scan (AMD EPYC 7662 and Intel Xeon Gold
# 6330): in 2D, STRUMPACK with MC64 job 2 rescued every system that the first stage (job 5) missed (34 of 34 and 30 of
# 30), at STRUMPACK's cost, while the stage without matching rescued none that job 2 did not and threw ZERO_PIVOT on
# every Womersley starter; in 3D every miss of the first stage (without matching) was rescued by job 5. To see a whole
# list, the iterates of stages 0 and 1 are scaled by 1 + 1e-6 (the private hook _lu_perturb_stages_for_testing): their
# relative residual, about 1e-6, misses the Stokes gate direct_tol = 1e-10, and the walk reaches Eigen.
@needs_strumpack
def test_default_2d_chain_is_job_5_then_job_2_then_eigen(hooks):
    """The 2D default chain of a STRUMPACK build through lu_retry="auto", every stage named in lu_attempts in the
    order tried; lu_retry="off" is its first stage alone."""
    hooks._lu_perturb_stages_for_testing([1e-6, 1e-6])
    d = dict(stokes(lu="strumpack", lu_retry="auto").diagnostics)
    at = d["lu_attempts"]
    assert [a["stage"] for a in at] == ["strumpack:mc64-5", "strumpack:mc64-2", "eigen"], outcomes(at)
    assert [a["outcome"] for a in at] == ["missed_target", "missed_target", "accepted"], outcomes(at)
    assert all(a["bound"] == 1e-10 for a in at), [a["bound"] for a in at]
    assert d["converged"] is True and d["lu_stage"] == "eigen" and d["lu_backend"] == "eigen", (
        d["converged"], d["lu_stage"], d["lu_backend"])
    assert d["solver"] == "direct:strumpack", d["solver"]
    # without the hook the first stage is accepted: the list is walked only as far as needed
    hooks._lu_perturb_stages_for_testing([])
    d = dict(stokes(lu="strumpack", lu_retry="auto").diagnostics)
    assert outcomes(d["lu_attempts"]) == [("strumpack:mc64-5", "accepted")], d["lu_attempts"]
    hooks._lu_perturb_stages_for_testing([1e-6])
    d = dict(stokes(lu="strumpack", lu_retry="off").diagnostics)
    assert outcomes(d["lu_attempts"]) == [("strumpack:mc64-5", "missed_target")], d["lu_attempts"]


@needs_strumpack
@pytest.mark.parametrize("matching,stages", [
    (False, ["strumpack:none", "strumpack:mc64-5", "eigen"]),     # the 3D default: without matching first
    (True, ["strumpack:mc64-5", "strumpack:none", "eigen"]),      # matching first: its measured 3D retry, not job 2
])
def test_default_3d_chains(hooks, mesh3d, matching, stages):
    """solve_stokes_3d through lu_retry="auto": the list of its first matching, walked to its end. A stage without
    matching may meet a zero pivot on this saddle point instead of missing (recorded "threw"); either passes the solve
    on."""
    fluids = _fluids()
    mesh, bcs = mesh3d
    hooks._lu_perturb_stages_for_testing([1e-6, 1e-6])
    d = dict(fluids.solve_stokes_3d(mesh, 1.0, None, bcs, lu="strumpack", lu_matching=matching,
                                    lu_retry="auto").diagnostics)
    at = d["lu_attempts"]
    assert [a["stage"] for a in at] == stages, outcomes(at)
    assert all(a["outcome"] in ("missed_target", "threw") for a in at[:2]), outcomes(at)
    assert at[2]["outcome"] == "accepted", outcomes(at)
    assert d["converged"] is True and d["lu_stage"] == "eigen", (d["converged"], d["lu_stage"])
    assert d["lu_matching_retry"] is True, d["lu_matching_retry"]


# ------------------------------------------------------------------------------------------------ the projection
def test_projection_is_accepted_on_its_own_gate(hooks):
    """The projection's acceptance is its gate, as a Verdict: the accepted attempt's measure IS projection_residual
    (the chain and the gate evaluate one function of the same solution, so the two numbers agree bit for bit), its
    bound is kProjectionResidualMax and its reason names the projection residual; a relative residual at the step
    bound (the natural miswiring, lin_solve_ being shared) would give another measure and bound. With [eigen, eigen]
    and stage 0 forced to miss, the same holds for stage 1, and stage 0's measure is the gate of its own (identical)
    solution. The unit square, N = 16: 4.1e-17 to 5.1e-17 measured (Intel Xeon Gold 6330, AMD EPYC 7662)."""
    ts = transient(n=16)
    assert ts.initialize(field(), 0.0, True) is True
    d = dict(ts.diagnostics)
    at = d["projection_lu_attempts"]
    assert len(at) == 1 and at[0]["outcome"] == "accepted", at
    check_attempt_record(at[0])
    assert at[0]["measure"] == d["projection_residual"], (at[0]["measure"], d["projection_residual"])
    assert at[0]["bound"] == PROJECTION_MAX and "projection residual" in at[0]["reason"], at[0]
    assert 0.0 < at[0]["measure"] < 1e-15
    # the caller's row blocks (section 2.2 of the plan): the pinned pressure row is a constraint row (block id -2)
    assert len(at[0]["constraint_ratios"]) == 1, at[0]["constraint_ratios"]
    two_eigen(hooks)
    hooks._lu_force_outcome_for_testing("missed_target", 1)
    assert ts.initialize(field(), 0.0, True) is True
    d = dict(ts.diagnostics)
    at = d["projection_lu_attempts"]
    assert outcomes(at) == [("eigen", "missed_target"), ("eigen", "accepted")], at
    assert at[0]["reason"] == "test hook"
    assert at[1]["measure"] == d["projection_residual"] == at[0]["measure"], [a["measure"] for a in at]
    assert at[0]["bound"] == at[1]["bound"] == PROJECTION_MAX


def test_refused_projection_names_its_attempts_and_the_pin(hooks):
    """A projection that every stage misses: its one stage (lu="eigen", the multiplier set explicitly) perturbed by
    1e-8 (the LU hook, stage 0) measures about 8e-10 against 1e-13. initialize() returns False, the attempt is
    recorded with that measure, and the error raised by stepping from the refused state names the attempt and the
    remedy that works, the pinned pressure (nullspace="pin"; the measured failures on scaled domains are the
    mean-value multiplier's: with the pin every backend passes), instead of advising Eigen as the cure for STRUMPACK."""
    ts = transient(nullspace="multiplier")
    hooks._lu_perturb_solution_for_testing(1e-8)
    ok = ts.initialize(field(), 0.0, True)
    hooks._lu_perturb_solution_for_testing(0.0)
    d = dict(ts.diagnostics)
    assert ok is False and d["projection_converged"] is False
    for call in (lambda: ts.advance(0.05), lambda: ts.snapshot()):
        with pytest.raises(Exception) as ei:
            call()
        msg = str(ei.value)
        assert "projection" in msg and 'nullspace="pin"' in msg, msg
        assert "STRUMPACK returns inaccurate projections" not in msg, msg
        assert "eigen" in msg and "missed_target" in msg, f"the error does not name the attempts: {msg}"
    at = d["projection_lu_attempts"]
    assert outcomes(at) == [("eigen", "missed_target")], at
    assert PROJECTION_MAX < at[0]["measure"] < 1e-6 and at[0]["measure"] == d["projection_residual"], at[0]
    assert len(at[0]["constraint_ratios"]) == 1, "the bordered row is not declared a constraint row (block id -2)"


def test_refused_projection_of_an_open_flow_does_not_advise_a_pressure_treatment():
    """An open flow (walls with one do-nothing side) has no pressure nullspace treatment: the constructor refuses
    nullspace="multiplier" for it. On a 1e10 x 1e3 domain (N = 32, lu="eigen") the projection is refused (measured
    3.6e-8, Intel Xeon Gold 6330); the error must not claim a pinned pressure or advise the multiplier, and keeps the
    length-unit advice."""
    lx, ly = 1e10, 1e3
    bcs = [dict(b) for b in walls()]
    bcs[1] = {"attribute": bcs[1]["attribute"], "kind": "do_nothing"}
    ts = _fluids().TransientNavierStokes(rect(32, 32, lx, ly), MU, RHO, None, bcs, scheme="newton", vectorized=True,
                                         newton_rtol=NEWTON_RTOL, lu="eigen")
    assert ts.initialize(field(lx, ly), 0.0, True) is False
    assert dict(ts.diagnostics)["nullspace"] == "none"
    with pytest.raises(Exception) as ei:
        ts.advance(1.0)
    msg = str(ei.value)
    assert "open boundary" in msg and "length unit" in msg, msg
    assert 'nullspace="pin"' not in msg and 'nullspace="multiplier"' not in msg, msg
    assert "It was solved with" not in msg and "mean-value multiplier" not in msg, msg


def _unit_reference(n, a):
    """The unit-covariant reference: the projection of the same field on [0, a] x [0, 1], a >= 1
    the aspect ratio, solved with Eigen's LU and the pinned pressure. With the length unit scaled by s the momentum
    rows scale like s^2 and the continuity rows like s, so the exact discrete velocity is the same on every mesh of
    the family (same vertex and element numbering): this reference needs no solver of the scaled system."""
    ts = transient(n=n, lx=a, ly=1.0, nullspace="pin")
    assert ts.initialize(field(a, 1.0), 0.0, True) is True
    return _velocity(ts)


P2_NODES = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)


def _velocity(ts):
    snap = ts.snapshot()
    el = np.arange(snap.n_elements, dtype=np.int32)
    u, _ = snap.velocity(el, P2_NODES)
    return np.asarray(u)


def _admitted(n, a):
    """The error the gate admits, relative to the largest velocity: 1.5e-13 N on the squares (docs/physics/fluids.md,
    "The bound refuses relative errors above about 1.5e-13 N"), and in a channel of aspect ratio a an error of the
    momentum block of size eps is seen as eps / (a N) (the same section), so up to 1e-13 a N passes; 1.5e-13 a N
    covers both. Measured (Xeon Gold 6330 / EPYC 7662): side 1e6, N = 16: 1.05e-14 / 1.20e-14 (allowance 2.4e-12);
    N = 32: 2.32e-14 / 2.15e-14 (4.8e-12); 1e6 x 1e4, N = 16: 5.82e-13 / 5.37e-13 (2.4e-10); 1e7 x 1e4, N = 32:
    9.96e-12 / 1.10e-11 (4.8e-9); side 1e-6, N = 32: 4.02e-14 / 4.16e-14 (4.8e-12)."""
    return 1.5e-13 * max(a, 1.0) * n


@pytest.mark.parametrize("lu", ["eigen", None])
def test_bordered_projection_at_a_small_length_unit_passes_the_kernel_probe(lu):
    """The square of side 1e-6 (N = 32) with the mean-value multiplier: the unscaled kernel probe refused it
    (||A z|| / (||A||_F ||z||) = 1.34e-15 against 1e-14, not invariant under row scaling); the blockwise probe, with
    the caller's row blocks and each constraint row its own block, measures 1.0 (every case measured, Intel Xeon Gold
    6330 and AMD EPYC 7662) and lets it through. With the probe off the projection measures 4.5e-17 to 6.0e-17 on
    both processors with both backends, so initialize() is required to succeed, at round-off, and to give the
    unit-covariant reference's velocity. lu=None is the build's default (STRUMPACK when present)."""
    if lu is None and not _strumpack_available():
        pytest.skip("the default backend of this build is Eigen: the other case covers it")
    n, s = 32, 1e-6
    ts = transient(n=n, lx=s, ly=s, nullspace="multiplier", lu=lu)
    ok = ts.initialize(field(s, s), 0.0, True)
    d = dict(ts.diagnostics)
    assert ok is True, f"refused: projection_residual {d['projection_residual']}, failure {d.get('failure')}"
    assert d["lu_kernel_probe"] > 0.1, f"the probe ratio {d['lu_kernel_probe']} is not of order one"
    assert 0.0 < d["projection_residual"] < 1e-15, d["projection_residual"]
    u, ref = _velocity(ts), _unit_reference(n, 1.0)
    err = np.max(np.abs(u - ref)) / np.max(np.abs(ref))
    assert err < _admitted(n, 1.0), f"velocity differs from the unit-covariant reference by {err:.3e}"


# Projections where a stage of the default chain passes the gate on BOTH processors measured (AMD EPYC 7662 / Intel
# Xeon Gold 6330, each stage measured alone) while today's chain fails or refuses them, because it accepts on the
# refined relative residual with the floor kGarbage = 1e-6, which the border row's weight inflates:
#   square, side 1e6, N = 16: strumpack:mc64-5 gate 5.1e-16 / 4.5e-17 (rr 9e-5 / 4.1e-6): today -1 on both
#   square, side 1e6, N = 32: 4.7e-17 / 4.3e-17 (rr 2.1e-5 / 1.9e-5): today -1 on both
#   1e6 x 1e4, N = 16:        5.3e-17 / 5.4e-17 (rr 8e-8 / 1.0e-6): today refused on the Xeon (4.38e-12, Eigen's,
#                             reached through the chain with a stale ZERO_PIVOT text); accepted on the EPYC
RESCUED_AT_STAGE_0 = [(16, 1e6, 1e6), (32, 1e6, 1e6), (16, 1e6, 1e4)]


@needs_strumpack
@pytest.mark.parametrize("n,lx,ly", RESCUED_AT_STAGE_0)
def test_bordered_projection_on_a_large_length_unit_is_accepted_on_its_gate(n, lx, ly):
    """The default STRUMPACK chain with the mean-value multiplier set explicitly (under nullspace="auto" an owner's
    change of the default treatment would make these pinned solves): the first stage is accurate on the projection's own
    measure and is accepted, where today's floor discarded it. Reference: the unit-covariant velocity."""
    ts = transient(n=n, lx=lx, ly=ly, nullspace="multiplier", lu=None)
    ok = ts.initialize(field(lx, ly), 0.0, True)
    d = dict(ts.diagnostics)
    at = d["projection_lu_attempts"]
    assert ok is True, f"refused: {d['projection_residual']}, attempts {outcomes(at)}"
    assert outcomes(at) == [("strumpack:mc64-5", "accepted")], [(a["stage"], a["outcome"], a["reason"]) for a in at]
    assert at[0]["measure"] == d["projection_residual"] <= PROJECTION_MAX
    a = lx / ly
    u, ref = _velocity(ts), _unit_reference(n, a)
    err = np.max(np.abs(u - ref)) / np.max(np.abs(ref))
    assert err < _admitted(n, a), f"velocity differs from the unit-covariant reference by {err:.3e}"


# Projections whose first STRUMPACK stage (MC64 job 5) misses the gate on both processors while STRUMPACK with MC64
# job 2 passes it (each stage measured alone, the multiplier, AMD EPYC 7662 / Intel Xeon Gold 6330; velocity against
# the unit-covariant reference, max norm relative to the largest velocity):
#   1e7 x 1e3, N = 16: job 5 1.8e-5 / 6.4e-5; without matching ZERO_PIVOT on both; job 2 6.1e-17 / 5.9e-17
#                      (velocity 7.6e-11 / 6.8e-11, allowance 2.4e-8); Eigen 5.8e-17 / 5.7e-17
#   1e7 x 1e4, N = 32: job 5 0.051 / 0.060; without matching ZERO_PIVOT on both; job 2 5.8e-17 / 5.8e-17
#                      (velocity 1.1e-11 / 9.8e-12, allowance 4.8e-9); Eigen 5.9e-17 / 7.0e-17
# The former list (without matching second) reached Eigen on both, after the zero pivot.
RESCUED_BY_JOB_2 = [(16, 1e7, 1e3), (32, 1e7, 1e4)]


@needs_strumpack
@pytest.mark.parametrize("n,lx,ly", RESCUED_BY_JOB_2)
def test_projection_missed_by_job_5_is_rescued_by_job_2(n, lx, ly):
    """The default STRUMPACK chain (lu_retry="auto") with the multiplier set explicitly: the first stage misses the
    projection's gate by orders of magnitude, the second, STRUMPACK with MC64 job 2, is accepted at round-off, and the
    walk ends there. Reference: the unit-covariant velocity, within the error the gate admits."""
    ts = transient(n=n, lx=lx, ly=ly, nullspace="multiplier", lu=None, lu_retry="auto")
    ok = ts.initialize(field(lx, ly), 0.0, True)
    d = dict(ts.diagnostics)
    at = d["projection_lu_attempts"]
    assert ok is True, f"refused: {d['projection_residual']}, attempts {outcomes(at)}"
    assert [a["stage"] for a in at] == ["strumpack:mc64-5", "strumpack:mc64-2"], [
        (a["stage"], a["outcome"], a["reason"]) for a in at]
    assert at[0]["outcome"] == "missed_target" and at[0]["measure"] > 1e-6, at[0]
    assert at[1]["outcome"] == "accepted" and at[1]["measure"] == d["projection_residual"] < 1e-15, at[1]
    assert all(a["bound"] == PROJECTION_MAX for a in at), [a["bound"] for a in at]
    u, ref = _velocity(ts), _unit_reference(n, lx / ly)
    err = np.max(np.abs(u - ref)) / np.max(np.abs(ref))
    assert err < _admitted(n, lx / ly), f"velocity differs from the unit-covariant reference by {err:.3e}"


# ------------------------------------------------------------------------------------------------ texts
def _f3a():
    _fluids()
    sys.path.insert(0, str(CASES_DIR))
    try:
        import f3a_transient as F3A
    finally:
        sys.path.remove(str(CASES_DIR))
    return F3A


class _Run:
    """What entry() reads of a transient solver: its diagnostics and run stats."""
    def __init__(self, escalated, attempts, failed=0):
        self.diagnostics = {"strategy": "transient_newton", "solver": "direct:eigen", "lu_backend": "eigen",
                            "lu_stage": "eigen", "nullspace": "pin", "convection": "standard"}
        self.stats = {"all_converged": True, "max_relative_residual": 1e-13, "newton_iterations": 6, "solves": 3,
                      "lu_residual_max": 1e-15, "mass_lu_residual_max": 1e-16, "failure": "",
                      "lu_solves": 6, "lu_escalated": escalated, "lu_escalated_final": 0, "lu_attempts_total": attempts,
                      "lu_rejected": escalated, "lu_failed": failed, "lu_accepted_by": {"eigen": 6},
                      "lu_rejected_seconds": 0.0, "mass_lu_escalated": 0,
                      "lu_fallbacks": escalated, "lu_fallback_backend": "eigen" if escalated else ""}


def test_transient_runner_records_the_retries_and_states_them_in_its_note():
    """The runner records lu_escalated, lu_attempts (the run's attempts) and lu_failed per level, and its note reads
    "N solve(s) needed another stage (M attempts)" from them, not today's transition count ("fallback transition(s)",
    which counted a solve that reached the third backend twice)."""
    F3A = _f3a()
    rec = F3A.Rec({}, True)
    rec.entry("L1", _Run(2, 8), "laplacian", False)
    rec.entry("L2", _Run(0, 6), "laplacian", False)
    rec.entry("L3", _Run(1, 7), "laplacian", False)
    solver = rec.runs[0]["solver"]
    assert (solver["lu_escalated"], solver["lu_attempts"], solver["lu_failed"]) == (2, 8, 0), solver
    notes = rec.result()["notes"]
    assert "3 solve(s) needed another stage (15 attempts)" in notes, notes
    assert "fallback transition" not in notes, notes


def _unreleased():
    """The notes since 0.1.0: "Unreleased" and the release sections above 0.1.0."""
    text = CHANGELOG.read_text("utf-8")
    return text.split("## Unreleased", 1)[1].split("\n## 0.1.0", 1)[0]


@pytest.mark.parametrize("token", ["lu_retry", "lu_stage", "lu_backward_error", "lu_attempts", "lu_escalated",
                                   "projection_lu_attempts", "lu_fallbacks"])
def test_changelog_states_the_chain(token):
    assert token in _unreleased(), f'CHANGELOG.md (notes since 0.1.0) does not mention {token}'


def test_changelog_keeps_the_aliases_for_one_release():
    text = _unreleased()
    assert re.search(r"alias", text) and "lu_matching_retry" in text and "lu_fallback_backend" in text, (
        'CHANGELOG.md (notes since 0.1.0) does not say that lu_fallbacks, lu_fallback_backend and lu_matching_retry are kept '
        "as aliases")


@pytest.mark.parametrize("token", ["lu_retry", "lu_stage", "lu_attempts", "lu_escalated", "projection_lu_attempts"])
def test_fluids_doc_describes_the_records(token):
    assert token in FLUIDS_DOC.read_text("utf-8"), f"docs/physics/fluids.md does not mention {token}"


def test_llms_txt_names_the_keyword_and_the_projection_remedy():
    text = LLMS.read_text("utf-8")
    assert "lu_retry" in text and "lu_attempts" in text
    assert 'nullspace="pin"' in text, "llms.txt does not name the pin as the remedy of a refused projection"
    assert "`lu=\"eigen\"` is the remedy" not in text, "llms.txt still gives Eigen as the remedy for STRUMPACK"


def _flat(text):
    """The text with comment markers and line breaks removed, so that a phrase is found however it is wrapped."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*//+ ?", "", text))


def _sentences(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z`(\-])", _flat(text))


def _section(text, heading):
    """A markdown section: from its heading to the next heading of the same or a higher level."""
    start = text.index(heading + "\n")
    level = len(heading.split()[0])
    m = re.compile(r"(?m)^#{1,%d} " % level).search(text, start + len(heading))
    return text[start:m.start() if m else len(text)]


JOB_2 = re.compile(r"mc64-2|MC64 job 2|job 2")


def test_fluids_doc_states_the_measured_2d_chain():
    """The default chains of docs/physics/fluids.md ("Linear solvers") are the measured ones: after STRUMPACK with
    MC64 matching (job 5) comes STRUMPACK with MC64 job 2, then Eigen; the stage without matching is no longer the
    2D retry (it stays the first stage of the 3D list)."""
    sec = _flat(_section(FLUIDS_DOC.read_text("utf-8"), "### Linear solvers"))
    assert JOB_2.search(sec), 'docs/physics/fluids.md "Linear solvers" does not name the MC64 job 2 stage of the 2D chain'
    assert "STRUMPACK without matching, then Eigen" not in sec, (
        'docs/physics/fluids.md "Linear solvers" still gives the 2D retry as STRUMPACK without matching')
    assert "5e4" in sec and "3e5" in sec, 'docs/physics/fluids.md "Linear solvers" lost the Eigen size caps'


def test_changelog_states_the_measured_2d_chain():
    text = _flat(_unreleased())
    assert JOB_2.search(text), 'CHANGELOG.md (notes since 0.1.0) does not name the MC64 job 2 stage of the 2D chain'
    assert "STRUMPACK with MC64 matching, then without, then Eigen" not in text, (
        'CHANGELOG.md (notes since 0.1.0) still gives the 2D default chain as STRUMPACK with matching, then without')


@pytest.mark.parametrize("path", [FLUIDS_DOC, SPARSE_LU_HPP], ids=["fluids.md", "sparse_lu.hpp"])
def test_the_1e6_x_1e4_refusal_is_attributed_to_eigen_with_the_multiplier(path):
    """The projection refused at 4.4e-12 on 1e6 x 1e4 (N = 16, Intel Xeon Gold 6330) was Eigen's LU solving the
    system bordered by the mean-value multiplier (the chain's last stage), not STRUMPACK: STRUMPACK's own first stage
    passes the gate there at 5.4e-17. Every sentence that quotes the number names Eigen and the multiplier."""
    for sentence in _sentences(path.read_text("utf-8")):
        if "4.4e-12" in sentence:
            assert "Eigen" in sentence and "multiplier" in sentence, (
                f"{path.name} quotes 4.4e-12 without naming Eigen's LU and the mean-value multiplier: {sentence[:400]}")


def test_sparse_lu_header_does_not_give_eigen_as_the_projection_remedy():
    """A refused projection's remedy is the pinned pressure (nullspace="pin"; lu="eigen" helped only because it
    implies the pin): the header must not give Eigen's LU as the remedy without the pin."""
    for sentence in _sentences(SPARSE_LU_HPP.read_text("utf-8")):
        if re.search(r"remed(y|ies)", sentence) and ("LUBackend::EigenLU" in sentence or 'lu="eigen"' in sentence):
            assert "pin" in sentence, f"sparse_lu.hpp gives Eigen as the remedy without the pin: {sentence[:400]}"


def test_fluids_doc_states_how_far_an_accepted_strumpack_projection_is_from_the_reference():
    """An accepted STRUMPACK projection on 1e5 x 1 (N = 32) is about 1e-9 of the largest velocity from the
    unit-covariant reference (7.4e-10 AMD EPYC 7662, 1.0e-9 Intel Xeon Gold 6330, MC64 job 2), within the error the
    gate admits there (4.8e-7), and far more than the 1.0e-11 that the docs quoted as the difference; the docs give
    it."""
    hits = [x for x in _sentences(FLUIDS_DOC.read_text("utf-8"))
            if re.search(r"1e5 x 1(?![0-9.e])", x) and "velocity" in x and re.search(r"\b\d(\.\d+)?e-(9|10)\b", x)]
    assert hits, ("docs/physics/fluids.md does not say how far an accepted STRUMPACK projection on 1e5 x 1 (N = 32) is "
                  "from the reference (about 1e-9 of the largest velocity)")


def test_kgarbage_is_gone_from_the_navier_stokes_solver():
    src = (ROOT / "cpp" / "src" / "fluids" / "navier_stokes.cpp").read_text("utf-8")
    assert "kGarbage" not in src, "navier_stokes.cpp still has the 1e-6 accuracy floor kGarbage"
    assert "lu_fallbacks_solve" not in src, "navier_stokes.cpp still keeps a solver-lifetime fallback counter"
