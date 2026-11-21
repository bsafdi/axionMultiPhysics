# Fluids (beta): formulation and numerics

This page collects the details of the Python interface, the formulation, the numerical methods, the MPI behaviour and the verification of `axionmultiphysics.fluids` and `axionmultiphysics.transport` (beta). The [fluids guide](../guides/fluids.md) shows how to use them.

## Using the API

- **Meshes**: `fluids.Mesh.from_arrays(vertices, triangles or quadrilaterals, boundary segments, boundary attributes)`,
  optionally with isoparametric P2 geometry nodes; `Mesh.unit_square(n)` and `Mesh.annulus(...)` for tests;
  `fluids.Mesh3D.from_arrays` in 3D. Meshes are 2D in the (x, y) plane; axisymmetric problems use the meridional half
  plane `(r, z) = (x, y)`.
- **Boundary conditions**: a list of dicts, exactly one per boundary attribute (`{"attribute": 2, "kind":
  "do_nothing"}`); an attribute without a condition, or with two, is refused. The kinds each solver accepts are listed
  in its docstring.
- **Data**: viscosity, force and boundary values are numbers or Python callables `f(x, y)` (`f(x, y, t)` for the time
  stepper); with `vectorized=True` the callables receive arrays and are tabulated once.
- **Units**: `solve_stokes` and `solve_stokes_3d` are kinematic (everything divided by the density, `nu` the kinematic
  viscosity); Navier-Stokes, the time stepper, MHD and Boussinesq are dynamic (`mu`, `rho`, physical pressure).
- **Convergence**: no solver raises on non-convergence. Check `sol.diagnostics["converged"]` (and
  `["relative_residual"]`); the fields of a solve that did not converge are refused. An option that is not offered
  is refused (`fluids.FluidsError`, or `ValueError` for arguments checked in Python) with a message that says what to
  use instead.
- **Results**: point evaluation at reference points of given elements (`sol.velocity(elements, ref_points)`, `pressure`,
  `geometry`), Babuska-Miller forces and torques (`sol.reaction(attributes)`), boundary nodes, and for Navier-Stokes the
  free-DOF residual and Jacobian (`state`, `residual`, `jacobian_action`) for Taylor tests.
- **Build**: the module is compiled only in the MPI source build (see the [installation guide](../guides/install.md));
  the serial wheels and the sequential build do not contain it, and `import axionmultiphysics.fluids` then raises
  `ImportError`. `axionmultiphysics.build_info()["fluids"]` says whether a build has it. STRUMPACK is optional: without
  it the direct solves use Eigen's sparse LU.
- **Beta warning**: the first import of `axionmultiphysics.fluids` in a process emits one
  `axionmultiphysics.BetaWarning`; the first import of `axionmultiphysics.transport` emits two (its own and, because it
  imports `axionmultiphysics.fluids`, the fluids one). Silence them with
  `warnings.simplefilter("ignore", axionmultiphysics.BetaWarning)`.

## MPI

- 2D Stokes and Navier-Stokes solves partition the mesh over `MPI_COMM_WORLD` when the script runs on several ranks
  (`mpirun -n 4 python script.py`, or `srun` in a SLURM job), but point evaluation and `boundary_nodes()` (and
  `blocks()` of a Stokes solution) refuse more than one rank, because element numbers are then rank-local. In
  practice 2D problems run on one rank. The time stepper takes its time steps on any number of ranks (the model-matrix
  test runs them on two); point evaluation of a stored state needs one.
- 3D Stokes is the distributed path: `solve_stokes_3d` is collective over `MPI_COMM_WORLD`; every rank builds the same
  serial mesh, the solver partitions it with METIS, and each rank evaluates on its own elements, addressed by input
  element id (`sol.element_ids()`).
- Scalar transport, Boussinesq and MHD run on one rank and refuse a partitioned mesh. GLS works on vertex and
  triangle arrays and solves serially (on several ranks every rank would solve the same problem).
- The verification cases that need several ranks start them through the launcher of `axionmultiphysics`
  (`$AXMP_MPI_LAUNCHER`, else `srun` inside a SLURM job, else `mpirun`/`mpiexec`).

## Verification

The verification cases (manufactured solutions, closed-form solutions, discrete identities and comparisons with
NGSolve and DOLFINx on the same discrete problem), with their acceptance criteria and the results of the last run,
are listed on the [fluids validation page](../validation/fluids.md). They run with `python benchmarks/fluids/run.py`
from a source checkout (`list`, `show <case>`, `run <case>`, `run --cost short`); the case format is described in
`benchmarks/fluids/README.md`, and the optional generators of the reference data (SymPy, NGSolve, DOLFINx) are in
`tools/fluids_mms/`. The unit tests are `tests/test_fluids_*.py` and `cpp/tests/test_fluids_*.cpp`.

## Formulation and numerics

### Equations

Incompressible flow with a viscous form chosen per solve:

- **Stokes** (kinematic units, everything divided by the density):
  `-div sigma_F(u, p) = f`, `div u = 0`.
- **Steady Navier-Stokes** (dynamic units: dynamic viscosity `mu`, constant density `rho`, physical pressure):
  `rho (u . grad) u - div sigma_F(u, p) = f`, `div u = 0`, with the convective term in the standard form or in Temam's
  skew-symmetric form (`+ rho/2 (div u) u`).
- **Transient Navier-Stokes**: `rho (du/dt + (u . grad) u) - div sigma_F(u, p) = f`, `div u = 0`.

Two viscous forms: the **stress form** `sigma_S = 2 mu eps(u) - p I` (`a(u, v) = int 2 mu eps(u) : eps(v)`, the Cauchy
stress; required for a variable viscosity and for physical shear-stress conditions) and the **Laplacian form**
`sigma_L = mu grad u - p I` (`a(u, v) = int mu grad u : grad v`; its natural condition is the pseudo-traction
`mu du/dn - p n`, so on curved slip or traction walls the solved problem is not the physical one; use the stress form
there). An optional grad-div term `gamma (div u, div v)` (gamma in viscosity units) is available for Stokes.

**Geometries.** Planar 2D; axisymmetric meridional half-plane `(r, z) = (x, y)` with swirl (unknowns `(u_r, u_z,
u_theta)`, every integral weighted by `r`, continuity tested with `r q`, hoop terms `u_r / r`, `u_theta / r`,
centrifugal and Coriolis terms in the convection); 3D (Stokes only, on a partitioned `ParMesh`); planar 2.5D
(z-invariant, three velocity components; offered only with the MHD coupling).

### Discretization

- **Taylor-Hood** P2/P1 (Q2/Q1) elements for velocity and pressure, with isoparametric P2/Q2 geometry on curved
  boundaries. Meshes whose triangles (tetrahedra) violate the Taylor-Hood mesh condition (all vertices on the boundary;
  no interior vertex) are reported by `Mesh.taylor_hood_check()` / `Mesh3D.taylor_hood_check()`.
- **Quadrature**: fixed rules that integrate the polynomial parts exactly on affine elements; no quadrature point lies
  on `r = 0`. The trilinear convective form is integrated at its exact degree, which depends on the element shape and
  on the geometric order `q` of the mesh as well as on the velocity order `p`:

    | element | counts | planar | axisymmetric | straight elements, `p = 2` |
    |---|---|---|---|---|
    | triangle (P`p` on a P`q` geometry) | total degree | `3p + q - 2` | `3p + 2q - 2` | 5 and 6 |
    | quadrilateral (Q`p` on a Q`q` geometry) | degree per direction | `3p + q - 1` | `3p + 2q - 1` | 6 and 7 (4 x 4 Gauss points for both) |

    A straight quadrilateral needs one degree more than a straight triangle because its (bilinear) map is not affine
    and because a tensor-product gradient keeps the full degree `p` in the direction it does not differentiate; the
    count holds on a rectangle too. That is the default rule. `quad_order_convection` selects another one, and a rule
    that does not integrate the form exactly is refused when the solver is built (the error names
    `NavierStokesOptions.quad_order_convection`, the given value, the required minimum, the geometry, the velocity
    order and the mesh's geometric order): an under-integrated convective term changes the discrete operator, while
    the Newton gate measures the residual of whatever system was assembled and would still report convergence. What
    counts is the rule, not the number asked for: an `n`-point Gauss rule is exact to degree `2n - 1`, so on
    quadrilaterals an even degree selects the rule of the next odd one (on straight axisymmetric quadrilaterals 6 is
    accepted, being the same 4 x 4 rule as 7, and 5 is refused). `0` selects the exact degree; a larger value is
    accepted and costs only time. The degree in use is `diagnostics["quad_orders"][1]`.
    A default rule below these degrees under-integrates the form, and the solution moves with it: with the degrees
    of straight triangles (5 and 6) on quadrilaterals, a lid-driven cavity at Re = 100 on 6 x 6 planar
    quadrilaterals differs by 6.8e-3 of its largest velocity (1.6e-3 on 12 x 12), and not at all on axisymmetric
    ones, where degree 6 already selects the exact rule.
- **Boundary conditions** (exactly one per boundary attribute; an attribute without a condition is refused):
  Dirichlet (P2 nodal interpolation of the data, eliminated symmetrically), do-nothing, traction (optionally depending
  on the outward normal of the curved boundary), slip and symmetry (`u . n = 0` strongly, by a local rotation of the
  nodal frame with the consistent, mass-conserving normal; corners are detected from the turning angle), periodic
  (with a prescribed pressure drop or flow rate), axis (`u_r = u_theta = 0` at every velocity node on `r = 0`), and a
  pressure point (`p(x0)` prescribed, raw pressure reported). At a node shared by several conditions Dirichlet wins.
- **Pressure nullspace**: detected algebraically after elimination (`||B^T 1|| / (||B||_F ||1||) <= 1e-10`) and
  required to agree with what the conditions imply (no open boundary); removed by a mean-value multiplier, a pinned
  pressure node, or (MINRES) an orthogonal projection. The continuity right-hand side is projected onto the constants'
  complement before every solve and the flux defect of the interpolated Dirichlet data is reported. A velocity
  nullspace (a rigid motion with zero viscous energy that the boundary conditions leave free) is refused.
- **Forces and torques**: Babuska-Miller variational reactions (the residual tested with a rigid motion supported on
  the body), always with the stress-form residual for the physical force; the Laplacian-form pseudo-traction is
  available as a diagnostic.

### Stabilized equal-order elements (GLS)

`gls_problem` solves planar Stokes, Oseen or Navier-Stokes with continuous P1/P1 and residual-based Galerkin/least
squares stabilization: SUPG + PSPG + LSIC with Shakib-type `tau` built from the metric tensor `G` of the equilateral
reference triangle (`C_I = 36`, `C_G = 1`; `tau_supg` includes `(2/dt)^2` for one implicit BDF1/BDF2 step). The strong
residual keeps the viscosity-gradient term; all element integrals use a fixed 6-point degree-4 rule. One templated
element routine gives the residual, the exact Jacobian (forward-mode automatic differentiation, `tau` differentiated
with the state) and the Jacobian action. Enclosed flows only (every boundary vertex Dirichlet); axisymmetric GLS is not
offered.

### Scalar transport and Boussinesq

`axionmultiphysics.transport.ScalarTransport` (and `fluids.solve_heat`) solves steady
`a b . grad T - div(kappa grad T) = s` with P1 or P2 elements, planar or axisymmetric, in the convective form or one of
two conservative forms (total-flux or diffusive-flux natural data), with SUPG (Shakib `tau` with the longest edge,
evaluated at the quadrature points) and an exact Jacobian in `T` and in the advecting velocity. Flux data use the inward
convention `g = n . kappa grad T`; boundary conditions: Dirichlet, flux, Robin, zero flux, total flux (conservative
form), axis. `fluids.solve_boussinesq` couples it (P2 temperature) to Taylor-Hood Navier-Stokes with the buoyancy
`rho0 beta g (T - T_ref)` and solves the monolithic system by Newton with the exact coupled Jacobian and optional
buoyancy continuation. Both are steady and run on one MPI rank.

### Inductionless MHD

With `geometry="planar25d"` and `mhd={...}`, `solve_navier_stokes` adds the Lorentz force of a prescribed field
`B(x, y)` and conductivity `sigma(x, y)`: the current is `J = sigma(-grad phi + E_z e_z + u x B)` with the electric
potential `phi` (P2) as an extra unknown, the force is `J x B` with the same `J`, and the potential equation is the
conservative weak form `(J, grad psi) = <j_n, psi> - (q, psi)`. Electric conditions: insulating, prescribed normal
current, prescribed potential; a potential nullspace is detected and removed like the pressure one. Laplacian viscous
form only, one MPI rank, no induced field.

### Nonlinear solvers

Steady Navier-Stokes, Boussinesq and MHD share one driver: a Stokes start from the Dirichlet lift, optional Picard
(Oseen) steps, then Newton with the exact Jacobian (one kernel gives the convective residual, the Oseen matrix and the
Newton matrix) and an Armijo line search on the discrete dual norm of the residual; Reynolds-number (viscosity)
continuation and pseudo-transient continuation are available. Every solve stops on the steady relative residual in the
dual norm (`newton_rtol`, default 1e-12) and never raises on non-convergence: `diagnostics["converged"]` is the flag,
and the fields are refused when it is false.

**The round-off floor of the residual.** No iteration can push the residual below the round-off of its own
evaluation. In the interior the momentum rows are differences of O(1) terms that cancel to O(h²) (mu Δu = grad p),
so each row of the computed residual carries about the unit round-off times the size of those terms; the dual norm
weights that high-frequency noise by 1/h, while the denominator ||R(x_lift)||_\* of a traction-driven problem grows
only like h^(-1/2). The attainable *relative* residual therefore grows like h^(-3/2) under refinement and can exceed
a fixed `newton_rtol` on a fine mesh, with the iteration stagnating a few per cent above it. The steady driver
measures that floor instead of guessing it: `residual_floor` is the change of the *computed* residual, in the same
relative dual norm, when every entry of the state is moved by one unit in the last place, up and down alternately —
the smallest change double arithmetic can make to a state. It is an estimate of the residual a representable state
can attain, not a bound: of the two sign patterns measured it is the larger (an all-up perturbation gives about half
of it: 1.1e-12 against 2.2e-12 on the finest level below), and no other pattern was tried. An entry that is exactly
zero does not move in effect, so a state with many zero entries is sampled through its nonzero entries alone (at
Ha = 0 the potential block, u_y and u_z vanish identically: 47% of the unknowns on every level below), which
under-estimates the floor — the conservative direction, since a smaller floor stops less. A solve stops there, and
counts as converged, when all three of these hold: its last step did not reduce the residual at all, the residual is
within `residual_floor_factor` (default 4) of the measured floor, and it is still within a decade of `newton_rtol`.
`stopped_at_floor` is then set in the diagnostics next to the floor, so nothing is accepted silently, and
`residual_floor_factor = 0` leaves `newton_rtol` as the only gate. The stop accepts the iterate it fired on, so the
residual it reports can be slightly larger than the smallest one the full budget would have wandered to (the k = 4
row below: 1.14e-12 at the stop, against 9.97e-13 at the twenty-second step with STRUMPACK); both are the same
solution to within the noise the floor measures. The measurement, like every round-off-level quantity, depends on
the number of MPI ranks (the alternating sign follows the rank-local numbering), so a solve that sits at the floor
may stop a step earlier or later on another rank count and report a slightly different residual, and `converged` can
differ between a serial and a parallel run of such a solve. On the Hartmann strip of `analytic_hartmann_2d`
(traction-driven, 2 x 16 base cells, k red refinements) the measured floor and the achieved residual at Ha = 0 are
(the floor is measured the same way on every level, including the levels that reach `newton_rtol`, where the solver
itself never asks for it; the direct solves are STRUMPACK's, and the Eigen fallback gives the same numbers to
within 10%)

| k | unknowns | \|\|R(x_lift)\|\|_\* | relative residual | measured floor | ratio |
|---|---|---|---|---|---|
| 0 | 680 | 2.930 | 2.8e-14 | 3.8e-14 | 0.74 |
| 1 | 2450 | 4.140 | 6.3e-14 | 1.1e-13 | 0.58 |
| 2 | 9254 | 5.853 | 1.7e-13 | 2.8e-13 | 0.61 |
| 3 | 35918 | 8.277 | 4.9e-13 | 7.8e-13 | 0.63 |
| 4 | 141470 | 11.704 | 1.1e-12 | 2.2e-12 | 0.52 |

- the denominator grows by exactly sqrt(2) per refinement (1.4128 to 1.4141 over these levels) and the measured
  floor by about 2^(3/2) (2.56 to 2.88 per level, h^(-1.47) across the five), as the estimate above says; the
  residual the iteration actually reaches grows a little more slowly, by 2.2 to 2.9 per level (h^(-1.34)), because
  the fraction of the floor it attains varies from level to level;
- the measured floor is the larger of the two sign patterns tried, and it comes out above the residual the iteration
  achieves: the ratio stays between 0.52 and 0.75 here and is 0.88 on the Re = 300 cavity of the unit test, so the
  default factor 4 sits a factor of four and a half above every ratio measured;
- the k = 4 level shows what a gate below the floor means: with the stop switched off and the whole budget allowed
  (3 Picard and 30 Newton steps), only the first step reduces the residual by more than 10%: with the Eigen fallback
  the first four steps stand at 1.14e-12 to 1.15e-12, the fifth at 1.05e-12, and the steps after it wander between
  1.00e-12 and 1.03e-12, so whether the fixed gate of 1e-12 is met at all is decided by round-off and by the LU
  backend rather than by the iteration — the Eigen fallback ends at 1.01e-12 after all 33 steps, while STRUMPACK
  dips just under the gate at the twenty-second;
- Ha >> 1 has orders of magnitude more room at the same refinement (Ha = 1000, k = 4: relative residual 1.0e-15),
  because the velocity is smaller by Ha² at the same drive while the Lorentz term is mass-like and does not cancel.
  The gain sets in slowly: at Ha = 1 on the same finest level the residual is 7.7e-13 against a measured floor of
  1.9e-12, which clears the 1e-12 gate by less than a quarter and without the stop.

**What the stop does not do.** It never accepts a residual more than ten times `newton_rtol`, whatever the measured
floor is. A floor far above the tolerance asked for is not a near miss but an unattainable request — a badly scaled
or far too finely refined problem — and the solve then keeps iterating and fails, with the residual and its ratio to
the floor in the failure message. A channel 1e6 long and 1 high in 4 x 8 cells (each cell 2.5e5 long and 0.125
high) at rho = 1 is such a problem: its measured floor is 2.3e-10, two hundred times the `newton_rtol` of 1e-12
asked for. Its iteration does stagnate there — after the first step the residual wanders between 7.6e-11 and
1.1e-10 for the rest of the budget, and about a quarter of the steps do not reduce it — so the first two
conditions hold and it is the ceiling alone that refuses the stop; the budget runs out at about 8e-11. The
requirement that the last step gained nothing guards a different problem: the same channel ten times longer at
rho = 1e-12 reduces its residual in every one of its 33 steps, by 0.2% to 20% a step (1.1e-9 to 4.4e-10), so its
floor is never measured and nothing is cut short, whereas a rule that fired when a step failed to halve the
residual would fire on its first step. The three conditions are independent, and the unit tests set one gate at a
time so that each is the only clause that can refuse a stop: the magnitude test rejects a stagnation far from the
floor (a Picard iteration on a Re = 2000 cavity stalls at 1e-2, 1e12 times its floor), the ceiling rejects a gate
a hundred times below what round-off allows and the stretched channel above, and the requirement that the last
step gained nothing is what makes the accepted stop of the unit test fall on a step that did not reduce the
residual. The decade itself is chosen between the measured cases, as relative residual over `newton_rtol` at the
stagnation: 1.14 on the Hartmann strip, 2.0 on the unit test's cavity (whose gate is set at half its attainable
residual), 76 on the stretched channel.

The rule applies to the steady strategy `newton` (Stokes start, Picard steps, Newton, continuation stages).
Picard-only, pseudo-transient continuation, the time steps, and the GLS and Boussinesq solvers stop on
`newton_rtol` alone and ignore `residual_floor_factor`.

The dual norm and the linear solves behind it are checked, and a verdict is not reported on an unchecked one. Every
dual-norm mass solve (velocity, pressure, MHD potential, and the transport norm of `solve_heat` and
`solve_boussinesq`) reports its relative residual; the worst value is `diagnostics["mass_lu_residual_max"]` (-1: none
taken). It must stay below `1e-10`: a mass solve with relative residual `eps` changes the dual norm by at most
`eps sqrt(kappa(M))` relative (`kappa` the condition number of the mass matrix), so `1e-10` keeps the norm within
`1e-3` for any `kappa <= 1e14`, and the solves are refined to round-off (about `3e-15`) whatever the conditioning.
Above the bound `solve()` reports `converged = false` with the reason in `failure`. The gate catches an inaccurate
solve, not a singular matrix (a numerically singular mass matrix is solved to round-off residual with a different
answer). The GLS solver refuses a triangulation whose consistent P1 mass matrix cannot be Cholesky-factorized (a vertex
that no triangle uses, a degenerate or duplicated triangle) when it is built, and holds every dual-norm solve with
those factors to the same bound, with the worst residual in its `diagnostics["mass_lu_residual_max"]`; its Cholesky
solves are not refined, and measure about `3e-16` on a 20:1 graded mesh. `solve_stokes` and `solve_stokes_3d` are linear
solves accepted on the residual of their own system and take no dual norm.

Every Navier-Stokes history entry carries `lu_residual`, the refined relative residual of that step's direct solve
(-1: not recorded). `diagnostics["lu_residual_max"]` is the worst of them and `diagnostics["lu_target_missed"]` counts
the steps above the refinement target `min(1e-12, 1e-2 newton_rtol)` (on the published cavity continuations at
N = 32 and 64 a few steps do: the worst residual measured there is 8.4e-14). The enforced bound applies to the final
step, the one whose solution is reported, and is `max(newton_rtol, 1e-11)`: a final direct solve above it reports
`converged = false` ("the final direct solve missed its accuracy target"). Intermediate steps are recorded and counted
but not enforced, and the bound is not `newton_rtol` alone, because the verdict is the nonlinear residual recomputed
from the assembled operators after every step, which an inexact linear solve cannot make pass falsely, whereas the
linear residual of a step is limited by the conditioning of its matrix: the published symmetric-gradient solves at
N = 128 (132 098 velocity dofs) reach 1.4e-13 on their final step at `newton_rtol = 1e-13`, with a nonlinear residual of
7e-15, and 1.0e-12 on an intermediate step; the worst final step over the published steady cases is 4.3e-13, a factor
23 below the bound. The C++ option `NonlinearOptions::final_solve_residual_max` overrides the bound.

The stop at the round-off floor of the residual and these gates are separate questions with separate answers. The
floor stop decides the nonlinear verdict (the residual is as small as a representable state allows, and within a
decade of `newton_rtol`); the gates then check the linear algebra that verdict rests on, and they apply to a floor
stop as to any other solve. They do not conflict: the bound on the final direct solve is `max(newton_rtol, 1e-11)`
and the bound on the mass solves `1e-10`, and the last linear solve of a stagnated iteration is as accurate as any
other (on the Re = 300 cavity of the unit tests, stopped at a relative residual of 1.4e-15, the final direct solve
has a residual of 9.8e-16 and the mass solves 2.0e-16). A floor stop therefore passes the gates on a healthy
problem, and it does not shield a solve whose final direct solve misses its bound: such a solve reports
`stopped_at_floor = true` and `converged = false`, with the direct solve named in `failure`. The time steps and the
initial projection never use the floor stop; the projection has its own gate (below).

### Time integration

`fluids.TransientNavierStokes`: variable-step BDF1/BDF2 (BDF2 starts with one BDF1 step), solved per step either by
Newton or semi-implicitly (an extrapolated advecting velocity, one Oseen solve). The initial field is the discrete
L2 (Leray) projection of the nodal interpolant onto the discretely divergence-free fields with the nodal Dirichlet
data. It solves the saddle-point system `[M B^T; B 0] (du, lambda) = -(R_u, R_p)` for the correction `du` of the
interpolant `u0`, and it is gated on the backward error of that solve, formed with the assembled operators: for each
of the two blocks, the Euclidean norm of the residual over the Euclidean norm of the sizes of the terms of that block,

| block | residual | compared with |
|---|---|---|
| momentum | `M du + B^T lambda + R_u` | `abs(M) abs(du) + abs(B^T) abs(lambda) + abs(R_u)` |
| continuity | `B (u0 + du) - rhs_p` | `abs(B) (abs(u0) + abs(du)) + abs(rhs_p)` |

(`abs` entrywise), and `projection_residual` is the larger of the two ratios. The gate is

`projection_residual <= 1e-13`

at every `newton_rtol`. The measure is the residual in units of the round-off of evaluating it, so an accurate solve
gives a number near the unit round-off whatever the mesh, the shape of the domain and the length unit are, and
whatever `newton_rtol` is (the solve is refined until it stagnates). A residual relative to the data does not have
that property, for two reasons that were both measured. The momentum rows carry the square of the length unit and
the continuity rows the length unit, so one norm over both blocks weighs them by the unit: the same 1000:1 channel
gave 1.4e-15 as `1e-3 x 1e-6` and 1.0e-12 as `1e5 x 100`. And in a channel of aspect ratio a the multiplier is of size
a while its gradient across the channel is of size one, so `B^T lambda` cancels to one part in a n (n cells across) and
the momentum residual relative to `M du` is a n times the round-off:

| channel, 32 x 32 right triangles | 100 x 1 | 1000 x 1 | 1e4 x 1 | 1e5 x 1 |
|---|---|---|---|---|
| momentum residual over `norm(M du)` | 9.9e-14 | 9.7e-13 | 1.1e-11 | 1.0e-10 |
| `projection_residual` | 5.3e-17 | 5.2e-17 | 5.7e-17 | 5.3e-17 |

(Eigen's LU, AMD EPYC 7452; an initial field that is not divergence free.) The sums are over each block and not a
largest ratio over the rows, because a row in a region where the field vanishes has terms at round-off and a ratio
of order one: a field confined to one corner of the 1000 x 1 channel gave a largest row ratio of 1.1e-7 for a
projection whose `projection_residual` is 6.0e-17.

Measured with that field on N x N right-triangle meshes, with Eigen's LU (`lu="eigen"`), at `newton_rtol` from
1e-6 to 1e-14:

| case | `projection_residual` |
|---|---|
| unit square, N = 8, 16, 32, 64, 96, 128 (578 to 132 098 velocity dofs) | 4.6e-17 to 5.3e-17 |
| channels 10 x 1 to 1e5 x 1 and 1 x 100 (N = 32), 100 x 1 to 1e4 x 1 (N = 64, 128) | 4.3e-17 to 5.9e-17 |
| the 1000 x 1 channel as `1e-3 x 1e-6`, `1 x 1e-3`, `1e5 x 100`; the 1e4 x 1 channel as `0.01 x 1e-6`, `1e6 x 100` | 5.0e-17 to 6.0e-17 |
| squares of side 1e-6, 1e-4, 0.01, 100, 1000, 1e4, 1e6 (N = 32) | 4.2e-17 to 5.1e-17 |
| meshes with 32:1 and 100:1 cells and graded meshes | 4.4e-17 to 5.7e-17 |
| a field that is already divergence free (N = 8 to 64); a field confined to one corner | 4.2e-17 to 6.9e-17 |
| nonzero boundary data | 4.5e-17 to 5.1e-17 |
| P3 velocities; axisymmetric (an annulus), P2 and P3 | 4.3e-17 to 5.7e-17 |

(AMD EPYC 7452; the C++ test of the measure gives at most 6.0e-17 on one rank and on two.) On an Intel Xeon Gold
6330 the same solver gives 3.8e-17 to 6.9e-17 on squares and channels of these shapes and units (N = 8 to 64). The
bound `1e-13` is a factor 1400 above the largest value measured with Eigen's LU on either processor. That margin
belongs to lengths between 1e-6 and 1e6. With lengths of 1e7 and more Eigen's LU loses accuracy as well, and which
case loses it depends on the processor: `1e7 x 100` (N = 64) measures 2.3e-14 on the Xeon and is accepted, and
`1e9 x 1e4` is refused at N = 16 on the EPYC (1.3e-11) and at N = 32 on the Xeon (6.5e-12). Choose a length unit
in which the domain is not that large.

STRUMPACK (`lu="strumpack"`, the default of a build that has it, whose pressure nullspace is removed by the
mean-value multiplier unless `nullspace="pin"`) returns measurably inaccurate projections on some scaled or stretched
domains, and which ones depends on the processor. The sparse LU accepts a configuration on the projection's gate
itself (see Linear solvers), so a configuration that misses it is followed by the next one of the chain. Measured with
the same field, the `projection_residual` of each configuration tried, in the order of the chain (S: STRUMPACK with
MC64 matching, job 5; m2: STRUMPACK with MC64 job 2; both with the multiplier; the last one listed is accepted).
The rows with an m2 entry were re-measured for the chain as it is now, on an AMD EPYC 7662 and an Intel Xeon Gold 6330;
the other rows (S accepted, so the rest of the chain is not reached) are the earlier measurement on an AMD EPYC 7452
and the Xeon, with the same values for S where the two overlap:

| case | AMD EPYC 7452 | Intel Xeon Gold 6330 |
|---|---|---|
| `1e5 x 1`, N = 32 | S 2.1e-2, m2 8.1e-15 | S 0.37, m2 5.3e-17 |
| square of side 100, N = 64 | S 1.4e-14 | S 4.3e-17 |
| `1e4 x 100`, N = 16 | S 6.0e-17 | S 7.1e-14 |
| `1e6 x 1e4`, N = 16 | S 5.3e-17 | S 5.4e-17 |
| `10 x 1e-3`, N = 16 | S 1.4e-13, m2 5.5e-17 | S 2.1e-9, m2 5.5e-17 |
| squares of side 1e6, N = 16 and 32 | S 5.1e-16 and 4.7e-17 | S 4.5e-17 and 4.3e-17 |
| square of side 1e-6, N = 32 | S 5.4e-17 | S 5.1e-17 |
| `1e7 x 1e4`, N = 16 | S 6.2e-17 | S 7.9e-17 |
| `1e7 x 1e4`, N = 32 and 64 | S 5.1e-2 and 0.11, m2 5.8e-17 and 5.9e-17 | S 6.0e-2 and 0.10, m2 5.8e-17 and 5.9e-17 |
| `1e8 x 1e4`, N = 16 and 32 | S 5.6e-17 and 4.8e-17 | S 5.3e-17 and 5.2e-17 |
| `1e9 x 1e4`, N = 16 and 32 | S 7.7e-16 and 5.7e-17 | S 4.9e-16 and 2.7e-15 |

Every case is accepted on both processors, and STRUMPACK with MC64 job 2 accepted every system that S missed in
the 54 projection cases measured, in 0.04 s to 1.5 s per factorization (N = 32 to 128) where Eigen's LU with the
multiplier takes 0.5 s to 29.4 s on the systems that S missed at N = 32 to 64 (EPYC 7662).
Eigen's LU (`lu="eigen"`, with its pinned pressure) gives 4.2e-17 to 5.8e-17 on the first five on both processors.
On `10 x 1e-3` (N = 16) the accepted solve, m2, is 5.8e-11 of the largest velocity from the unit-covariant reference
(Eigen's LU, pinned pressure, length unit 1) on both processors, and Eigen's LU with the multiplier 6.1e-11 (EPYC
7662). An accepted STRUMPACK projection can still differ from that reference by much more than the last digits:
on `1e5 x 1` (N = 32) the accepted solve, STRUMPACK with MC64 job 2 and the multiplier, is 7.4e-10 (EPYC 7662) and
1.0e-9 (Xeon Gold 6330) of the largest velocity from the reference, within the 4.8e-7 that the gate admits there
(`1.5e-13 a N` with the aspect ratio a = 1e5 and N = 32: the rescue tolerance; the bound below, about `1.5e-13 N`, is
the error of the state that the gate refuses on the field of the table, and the conditioning of a channel adds the factor a). The largest accepted job-2 error measured is on `1e7 x 100` (N = 64), 1.9e-8 (EPYC 7662) and
3.3e-8 (Xeon), within its allowance of 9.6e-7; in the chain the first configuration is accepted there, so job 2 is
not reached. Accepted first-configuration solves (S, multiplier) differ from Eigen's LU with the pin, at the same length
unit, by 3.3e-13 of the largest velocity on `1e6 x 1e4` (N = 16, an EPYC node) and by 3.4e-11 on `1e4 x 1` (N = 16, Xeon
Gold 6330). The accepted projections on the squares of side 1e6 and 1e-6, on `1e6 x 1e4` (N = 16) and on
`1e7 x 1e4` (N = 32, accepted by job 2) give the velocity of the same projection on the unit-length domain of the same shape (solved with
Eigen's LU and the pin; the exact discrete velocity does not depend on the length unit) to 1.0e-14 to 1.1e-11 of the
largest velocity, within the error the gate admits (below). With nonzero boundary data STRUMPACK measures 1.1e-16 to
1.2e-16 (N = 32 to 128). The squares of side 1e-6 and 1e6 and the channels in the length unit 1e4 used to fail outright
(no measure formed), for two reasons that are gone: a fixed acceptance of the refined relative residual at 1e-6, which
the border row of the multiplier inflates while the solution is accurate (on the squares its configuration S passes the gate; on `1e7 x 1e4` at N = 32 and 64 S misses it and the
Eigen configuration was the accurate one that the floor discarded, and job 2 passes), and a
kernel probe over the whole matrix, whose ratio the length unit scales (1.3e-15 on the bordered square of side 1e-6).
Where no configuration passes, `initialize()` returns `False`, and the error lists the attempts and names the other
pressure treatment. With the multiplier, Eigen's LU is refused on the square of side 1e6 (2.7e-12), on `1e6 x 1e4`
(4.4e-12) and on `1e4 x 100` (8.8e-13; N = 16, Xeon): the refusal on `1e6 x 1e4` is Eigen's LU with the multiplier,
the last configuration of the chain before this one, and not STRUMPACK's, whose first configuration with the
multiplier passes there (5.3e-17 on the EPYC, 5.4e-17 on the Xeon). With the pinned pressure (`nullspace="pin"`) both
backends give about 5e-17 on all three (Xeon). The pin is not accurate everywhere: on `1e9 x 1e4` (N = 32, Xeon) it is refused
(6.5e-12 with Eigen's LU and the pin), while the
multiplier passes with STRUMPACK's first configuration (2.7e-15, the table); at N = 16 both pass on the Xeon (on the EPYC the pin with Eigen is refused there, 1.3e-11).
With the default STRUMPACK chain the pin passes on `1e9 x 1e4`: job 5 misses and STRUMPACK with MC64 job 2 is accepted,
at 2.6e-15 (Xeon, N = 32) and 5.5e-17 (Xeon, N = 16), and at 5.2e-17 (N = 16) and 5.6e-17 (N = 32) on an EPYC 7452.

What the bound refuses: a solve whose solution is wrong by a relative eps leaves the continuity residual
`eps (B u0 - rhs_p)`, the divergence of the interpolant, over terms that are larger than it because `B`
differentiates. With the field above that is a `projection_residual` of 0.68 eps / N on the squares and of 0.024 eps
to 0.028 eps on the channels at N = 32, the same in every length unit and at every `newton_rtol`. A projection solve
perturbed by `1e-8` on purpose measures 8.1e-10 at N = 8 and 5.4e-11 at N = 128, and between 2.1e-11 (nonzero
boundary data, N = 32) and 1.1e-9 over all the cases of the table but one: at least a factor 200 above the bound
on the meshes of the table. The factor falls like 1/N: with nonzero boundary data the measure of that perturbation
is 0.065 / N to 0.08 / N times it, and at N = 192 it is 3.4e-12, a factor 34 above the bound. The bound refuses
relative errors above about `1.5e-13 N` for the field of the table.

Two kinds of error are accepted, and both follow from the measure being a backward error (the size of the change
to the system that the returned solution solves exactly), not an error of the state:

- A field that is already divergence free. Its correction is its interpolation error (1.4e-5 of the field in the
  maximum norm on the 32 x 32 mesh), so perturbing that correction by `1e-8` changes the state by 1.4e-13 of the
  field, measures 7.9e-15, and is accepted.
- An error confined to the momentum block of a long channel. The momentum terms cancel to one part in a n (the
  aspect ratio times the cells across), so an error of relative size eps in that block is seen as about
  eps / (a n). On the `1e4 x 1` and `1e5 x 1` channels (N = 32) a divergence-free error that leaves the state
  wrong by 1.9e-8 of its largest velocity measures 3.5e-14 and 3.5e-15 and is accepted; a multiplier scaled by
  `1 + 1e-8` measures 5.3e-14 and 5.3e-15 and is accepted. On the unit square (N = 8) that multiplier measures
  1.3e-9 and is refused.

Over the ten published transient cases (every level, up to 132 098 velocity dofs) the worst direct solve of any step
is 5.9e-14 (the Womersley time studies) and the worst dual-norm mass solve 3.8e-16. The gate on the projection is
not the gate on the steps, which stays `newton_rtol` and has no stop at a round-off floor: with the field of the scan
above at `newton_rtol = 1e-14`, the steps on the 32 x 32 mesh converge (to 5e-15 to 6e-15), while on the 64 x 64 mesh
the first step stagnates at 1.5e-14 and is reported as not converged.
`initialize()` returns `False` when the gate is missed; the measure and the verdict are
`diagnostics["projection_residual"]` (-1: no projection run) and `diagnostics["projection_converged"]`. A failed
initialization leaves no valid state to step from, so `advance`, `step_trial`, `adapt` and `snapshot` raise an error
naming the projection and its residual until `initialize()` succeeds or a state is supplied with `set_history` (a step's
own non-convergence is still reported, not raised); the error lists the projection's attempts (one per configuration
of the sparse LU, below) and names the other pressure treatment (the pinned pressure, `nullspace="pin"`, after a
projection with the mean-value multiplier; the multiplier after one with the pin). Every step result carries
`lu_residual` and `TransientNavierStokes.stats` carries `lu_residual_max`, `lu_target_missed`,
`mass_lu_residual_max` and the step solves' counts of the sparse LU chain summed over the run (`lu_solves`,
`lu_escalated`, `lu_escalated_final`, `lu_attempts_total`, `lu_rejected`, `lu_failed`, `lu_accepted_by`,
`lu_rejected_seconds`, `mass_lu_escalated`; `reset_stats()` clears them). The initial projection is not a step: its
attempts are `diagnostics["projection_lu_attempts"]` and are never counted in the step statistics.
Adaptive steps use a Milne-device error estimate in the (r-weighted) L2 norm and a step controller with a ratio cap of 2
(below the BDF2 zero-stability bound). Time-dependent force and boundary data are callables of `(x, y, t)`.

### Linear solvers

Direct sparse LU (STRUMPACK when the build has it, with MC64 matching, else Eigen's SparseLU on one rank) with
iterative refinement checked against the system's own residual; every factorization of a system with a pressure
nullspace is probed so that a singular matrix is refused instead of "solved". MINRES with a block-diagonal
preconditioner is available for Stokes. In 3D a direct solve whose refinement stagnates is finished by restarted GMRES
preconditioned with the same factorization.

Every direct solve walks one recorded chain of configurations (`sparse_lu.hpp`, `default_chain`). `lu` names the
first; the default chains are

| first configuration | then |
|---|---|
| STRUMPACK with MC64 matching (job 5; the 2D and axisymmetric solvers) | STRUMPACK with MC64 job 2, then Eigen's LU up to 3e5 unknowns (5e4 on a 3D mesh) |
| STRUMPACK without matching (`solve_stokes_3d`) | STRUMPACK with matching, then Eigen's LU up to 5e4 unknowns (3e5 on a 2D mesh) |
| Eigen's LU | none |

`solve_stokes_3d(lu_matching=True)` starts with the matching and then retries without it (its measured 3D retry), then
Eigen's LU up to 5e4 unknowns: job 2 after a job-5 miss was measured on 2D systems only.

The 2D retry is measured. On the 2D systems that job 5 missed on their caller's gate, job 2 accepted every one: 34 of 34
on an AMD EPYC 7662 (28 bordered projections with the multiplier and the 6 planar `k = 5` Womersley starters, whose
first Newton solve has 74 691 unknowns) and 30 of 30 projections on an Intel Xeon Gold 6330. On the starters it
factorizes in 0.31 s to 0.33 s (EPYC 7662) where Eigen's LU takes 4.0 s to 4.3 s. STRUMPACK without matching rescued
nothing that job 2 did not, and threw ZERO_PIVOT on all six starters and on 8 of 10 sampled Newton steps of the
lid-driven cavity, so it is not in the 2D list; it stays the first configuration of the 3D list, where job 2 costs
what job 5 costs and rescued the same retries. An equilibrated configuration is not on any default chain, because
every projection case measured has a configuration that passes without it, and an iterate is not accepted on its
backward error alone, because no solution that missed its caller's measure had a backward error of 1e-13 or less.

A configuration is accepted on the caller's own accuracy gate; a solution that misses it, or a factorization that
fails, passes the solve on to the next configuration, and every attempt is recorded with its outcome and reason:

| caller | accepted when |
|---|---|
| Navier-Stokes steps (steady, transient, every strategy) | relative residual <= `max(newton_rtol, 1e-11)`: the final step's bound, applied to every step because any step may be the final one |
| initial projection (`TransientNavierStokes.initialize`) | `projection_residual` <= 1e-13, the gate itself |
| Stokes (`solve_stokes`, `solve_stokes_3d`) | relative residual over all rows <= `direct_tol` |
| Boussinesq, GLS steps | relative residual <= `max(newton_rtol, 1e-11)` (a threshold of their own: their LU had no gate) |
| scalar transport Jacobian | relative residual <= `max(1e-2 rtol, 1e-11)` (likewise new) |
| dual-norm mass solves | relative residual <= 1e-10 |

When no configuration is accepted, the solution with the smallest measure is returned and the caller's gate decides
as before (a final step above its bound, a mass solve above 1e-10 or a refused projection is reported as not
converged). Above its size cap an Eigen retry is not factorized (recorded as `refused`, reason `size`), so that a
failing solve cannot turn into hours of factorization (the cap is set by the dimension of the system: a 3D
factorization fills in far more); a first Eigen configuration has no cap. `lu_retry="off"` keeps the first
configuration alone: one attempt, judged and recorded like any other, whose solution is returned whether or not it
meets the caller's measure (recorded `missed_target` then), so that the caller's gate decides (a projection that
misses its gate is refused). It is for measurements and for reproducing the first attempt of an older run; in
`solve_stokes_3d` an older run whose matching retry fired is reproduced by the default chain, whose first two
configurations are that first attempt and its retry, and not by `"off"`. MC64 matching is a STRUMPACK option: the
`lu_matching` of a solution from an Eigen configuration is the configured `lu_matching`.

The records: `lu_stage` names the configuration that produced the solution (`"strumpack:mc64-5"`,
`"strumpack:mc64-2"`, `"strumpack:none"`, `"eigen"`, ...) and `lu_backend` its backend, while `solver` stays the configured one;
`lu_attempts` lists every attempt of the last direct solve (configuration, outcome, reason, the acceptance measure
and bound, the relative residuals before and after refinement, the blockwise backward error and the ratios of the
constraint rows, the fill and the times); `lu_backward_error` is the solution's blockwise backward error (recorded,
not used to accept). Navier-Stokes, Boussinesq, GLS and scalar transport add the counts of the call's step solves:
`lu_solves`, `lu_escalated` (solves that needed more than one configuration), `lu_escalated_final` (whether the last
one did), `lu_attempts_total`, `lu_rejected`, `lu_failed` (solves no configuration accepted), `lu_accepted_by`,
`lu_rejected_seconds`, and `mass_lu_escalated`; their history entries carry `lu_attempts` and `lu_stage`; the
transient solver's `projection_lu_attempts` holds its projection's attempts. `lu_fallbacks` (now the call's
`lu_escalated`), `lu_fallback_backend` and the 3D `lu_matching_retry` / `lu_matching_retry_reason` are kept as
aliases for one release.

A pressure nullspace removed by a pinned row or a bordered row is recognised by the kernel probe block by block: each
of the velocity and pressure row blocks and each constraint row is measured against the sizes of its own terms, so the
bordered system of a square of side 1e-6 is accepted (a probe ratio of 1.0, against 1.3e-15 for the probe over the
whole matrix, which the length unit scales).

Eigen's SparseLU runs on rank 0 alone, whatever the number of ranks, and is limited by that rank's memory and time.
It uses 32-bit indices up to 46 340 unknowns (and 2^28 matrix entries), a bound proved from the size of its factor
arrays (`sparse_lu.hpp`), and 64-bit indices above it, so a large system is factorized, or the factorization fails with an error. Nothing is checked beforehand: when Eigen
itself runs out of memory (or an allocation on rank 0 throws), every rank raises an error that names the unknowns, the matrix
entries, the index width, Eigen's own estimate of the factor and the remedies (STRUMPACK, or `solver="minres"` for
Stokes). Where the job's memory limit or the kernel's OOM killer ends the process first (Linux allocates the factor
lazily), no error can be reported. Large 3D systems need STRUMPACK: Eigen's COLAMD ordering fills in far more than
STRUMPACK's nested dissection.

## Limitations

- Beta: the interface may change; there is no design-file (spec/CLI), viewer or MCP integration, and the module needs
  the MPI source build.
- Ranks: 3D Stokes is the only distributed path. 2D Stokes and Navier-Stokes solves run on several ranks but their
  fields can be evaluated only on one; transport, Boussinesq and MHD refuse more than one rank, and GLS is serial.
- Time stepping: MHD is steady only (the time stepper has no MHD option, and the solver refuses MHD with a
  time-stepping strategy); transient Boussinesq is not offered; BDF orders above 2, generalized-alpha, slip, symmetry
  and periodic conditions and the skew convective form are refused in time steps.
- Not offered: open-boundary stabilization (backflow), turbulence models, compressible or variable-density flow,
  temperature-dependent properties, MHD in axisymmetry, induced magnetic fields, slip, symmetry and periodic conditions
  with convection, iterative (Krylov) solvers for Navier-Stokes, 3D Navier-Stokes.
- STRUMPACK has returned an inaccurate solution for large transient systems (about 7.5e4 unknowns). The failure is
  deterministic (bitwise the same residual on repetition and across jobs) and depends on the matrix: it occurs on
  planar fine-mesh (k = 5) BDF1 starter systems and never on a later step, never at k = 4 and never in axisymmetry.
  The chain of configurations then solves again (see Linear solvers; every attempt is recorded). The published
  verification runs use a build without STRUMPACK, where the chain has one configuration.
- The verification covers manufactured, closed-form and cross-code problems; no published benchmark value (cavity,
  cylinder, natural convection) is compared yet, and some cases were measured only on their first two refinement
  levels (see the validation page).

## References

- I. Babuška and A. Miller, The post-processing approach in the finite element method, Part 1: Calculation of
  displacements, stresses and other higher derivatives of the displacements, *Int. J. Numer. Methods Eng.* **20**,
  1085-1109 (1984). (Variational forces and torques.)
- F. Shakib, T. J. R. Hughes and Z. Johan, A new finite element formulation for computational fluid dynamics: X. The
  compressible Euler and Navier-Stokes equations, *Comput. Methods Appl. Mech. Eng.* **89**, 141-219 (1991).
  (Shakib-type stabilization parameter.)
- R. Temam, *Navier-Stokes Equations: Theory and Numerical Analysis*, North-Holland, Amsterdam (1977). (Skew-symmetric
  convective form.)
