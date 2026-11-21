"""Transient incompressible Navier-Stokes (beta, like the rest of axionmultiphysics.fluids): variable-step BDF1/BDF2, Newton per
step or semi-implicit, Milne error estimate with a step controller, projected (discretely divergence-free) initial
data; planar and axisymmetric with swirl.  Formulation: docs/guides/fluids.md.

    ts = fluids.TransientNavierStokes(mesh, mu, rho, force, bcs, form="stress", scheme="newton", max_order=2)
    ts.initialize(u0)                      # u0(x, y) -> components; projected onto discretely div-free P2
    r = ts.advance(dt)                     # one step; r["converged"], r["relative_residual"], r["iterations"]
    snap = ts.snapshot()                   # Evaluator-ready: snap.velocity(elements, ref_points), .pressure, .geometry
    ts.adapt(T, rtol, S_u, h0)             # error-controlled integration to T

DYNAMIC units (mu, rho, force density, physical pressure).  Time-dependent data: callables f(x, y, t) (vectorized:
f(X, Y, t) with arrays) for the force, Dirichlet values and traction; evaluated at the new time level t_n, for the
nodal P2 interpolation of the boundary data.  Boundary conditions: dirichlet, axis, traction, do_nothing (slip,
symmetry, periodic, skew convection, BDF order above 2, generalized-alpha and the MHD coupling are refused).  A
snapshot keeps its solver alive.  One MPI rank is needed for point evaluation.  Axisymmetric components are
(u_r, u_z, u_theta) as in solve_navier_stokes.  The relative residual of every step is the steady solver's dual-norm
residual relative to the step's lifted residual.

lu_retry="auto" (default): a direct solve whose solution misses its accuracy gate (a step's: its relative residual
at max(newton_rtol, 1e-11); the initial projection's: its own gate, 1e-13) is solved again with the next configuration
of a recorded chain (docs/physics/fluids.md, "Linear solvers"); "off": the first configuration (lu) alone.
diagnostics holds the records of the last step (solve_navier_stokes's keys) and projection_lu_attempts; stats sums
the step solves' counts over the run (lu_solves, lu_escalated, lu_escalated_final, lu_attempts_total, lu_rejected,
lu_failed, lu_accepted_by, lu_rejected_seconds, mass_lu_escalated; reset_stats() clears them; the initial projection
is never counted as a step), with the aliases lu_fallbacks and lu_fallback_backend kept for one release."""
from __future__ import annotations

from . import _f, _lu_retry, _vectorize, default_lu


class _Clock:
    tf = tg = tr = 0.0

    def set(self, tf, tg, tr):
        self.tf, self.tg, self.tr = float(tf), float(tg), float(tr)


class Snapshot:
    """A stored (velocity, pressure) state with the evaluation interface (velocity, pressure, geometry, boundary
    nodes) that the verification cases in benchmarks/fluids use."""

    def __init__(self, core, owner):
        self._c, self._owner = core, owner

    n_elements = property(lambda s: s._c.n_elements)

    def velocity(self, elements, ref_points):
        return self._c.velocity(elements, ref_points)

    def pressure(self, elements, ref_points):
        return self._c.pressure(elements, ref_points)

    def geometry(self, elements, ref_points):
        return self._c.geometry(elements, ref_points)

    def boundary_nodes(self, attributes):
        return self._c.boundary_nodes(list(attributes))


class TransientNavierStokes:
    def __init__(self, mesh, mu, rho, force=None, bcs=(), *, form="stress", geometry="planar", scheme="newton",
                 max_order=2, order=2, lu=None, nullspace="auto", newton_rtol=1e-12, newton_maxit=30,
                 quad_order_convection=0, direct_tol=1e-10, lu_threads=1, vectorized=False, lu_retry="auto"):
        if scheme not in ("newton", "semi"):
            raise ValueError("scheme must be 'newton' or 'semi'")
        if int(max_order) not in (1, 2):
            raise ValueError("BDF order must be 1 or 2 (higher orders and generalized-alpha are not offered)")
        self._vdim = 3 if geometry == "axisymmetric" else 2
        self._clock = _Clock()
        ck = self._clock
        vdim = self._vdim
        bcs = [dict(b) for b in bcs]
        for b in bcs:
            if b.get("normal_dependent"):
                raise ValueError("a normal-dependent traction is not offered with time-dependent data")
            v = b.get("value")
            if callable(v):
                which = "tr" if b.get("kind") == "traction" else "tg"
                b["value"] = (lambda fn, w: (lambda X, Y: fn(X, Y, getattr(ck, w))))(v, which)
        if callable(force):
            force = (lambda fn: (lambda X, Y: fn(X, Y, ck.tf)))(force)
        if vectorized:
            mu = _vectorize(mu, 1)
            force = _vectorize(force, vdim)
            for b in bcs:
                if callable(b.get("value")):
                    b["value"] = _vectorize(b["value"], vdim)
        self._vec = bool(vectorized)
        self._c = _f.make_transient(mesh, mu, float(rho), force, bcs, "laplacian" if form in ("lap", "laplacian")
                                    else "stress", geometry, int(order), lu or default_lu(), nullspace, scheme,
                                    int(max_order), float(newton_rtol), int(newton_maxit),
                                    int(quad_order_convection), float(direct_tol), int(lu_threads), bool(vectorized),
                                    ck.set, lu_retry=_lu_retry(lu_retry))
        self._mesh = mesh

    def _wrap0(self, fn):
        return _vectorize(fn, self._vdim) if self._vec else fn

    # -- state
    def initialize(self, u0, t0=0.0, project=True):
        """Initial state: the nodal P2 interpolant of u0(x, y) and, with project=True, its L2 (Leray) projection onto
        the discretely divergence-free fields with the nodal Dirichlet data at t0 (project=False keeps the interpolant).

        The projection is a saddle-point solve, accepted on the backward error of that solve (formed with the
        assembled operators): for the momentum and for the continuity block, the norm of the residual over the norm
        of the sizes of the terms of that block, and the larger of the two.  An accurate solve gives about 5e-17
        whatever the mesh, the shape of the domain, the length unit and newton_rtol are, and the gate is 1e-13 at
        every newton_rtol (docs/physics/fluids.md gives the measurements behind it).  The sparse LU accepts a
        configuration on this very gate, and when the first misses it the next configuration of the chain solves
        again (lu_retry); diagnostics["projection_lu_attempts"] records each attempt.  When no configuration passes,
        the error names the other pressure treatment: where the mean-value multiplier (the default with STRUMPACK) was
        refused on scaled or stretched domains the pinned pressure (nullspace="pin") passed with both backends, but
        the pin is not accurate everywhere (with Eigen's LU, lu="eigen", it is refused on 1e9 x 1e4 where the
        multiplier passes with STRUMPACK; the default STRUMPACK chain accepts the pin there with MC64 job 2).

        Returns True when the state is set, and False when the projection solve failed or missed the gate.  After a
        False return the solver holds no valid state: advance, step_trial, adapt, snapshot and trial_snapshot raise an
        error that names the projection and its residual, until a later initialize() returns True or set_history()
        supplies a state.  diagnostics["projection_residual"] (-1: no projection has run) and
        diagnostics["projection_converged"] hold the residual and the verdict of the last projection that ran;
        initialize(project=False) runs none and leaves them as they are."""
        return self._c.initialize(self._wrap0(u0), float(t0), bool(project))

    def set_history(self, times, fields):
        """Replace the history (newest first, 1 to 3 levels) by nodal interpolants of fields(x, y)."""
        self._c.set_history([float(t) for t in times], [self._wrap0(f) for f in fields])

    def advance(self, dt):
        return self._c.advance(float(dt))

    def step_trial(self, dt, force_order=0):
        return self._c.step_trial(float(dt), int(force_order))

    def commit(self):
        self._c.commit()

    def adapt(self, T, rtol, S_u, h0):
        return self._c.adapt(float(T), float(rtol), float(S_u), float(h0))

    def time(self):
        return self._c.time()

    def snapshot(self):
        return Snapshot(self._c.snapshot(), self)

    def trial_snapshot(self):
        return Snapshot(self._c.trial_snapshot(), self)

    def estimate_snapshot(self):
        return Snapshot(self._c.estimate_snapshot(), self)

    @property
    def n_elements(self):
        return self._c.n_elements

    @property
    def stats(self):
        return self._c.stats

    def reset_stats(self):
        self._c.reset_stats()

    @property
    def last_history(self):
        """Relative residuals after each Newton iteration of the last step."""
        return self._c.last_history

    @property
    def diagnostics(self):
        return self._c.diagnostics
