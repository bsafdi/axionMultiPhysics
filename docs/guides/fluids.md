# Fluids (beta)

Viscous incompressible flow, heat carried by a flow and the Lorentz force on a conducting fluid in a magnetic field
obey the Navier-Stokes equations and their couplings. In this guide we solve such problems with
`axionmultiphysics.fluids`, an incompressible-flow library built on MFEM, and its builder `axionmultiphysics.transport`.
Both are beta: their interfaces may change, and their verification covers manufactured, closed-form and cross-code
problems but no published benchmark value or experiment. The module is compiled only in the MPI source build
([installation](install.md)); `axionmultiphysics.build_info()["fluids"]` says whether a build has it.

## Solvers

| feature | entry point | geometry | MPI ranks |
|---|---|---|---|
| Stokes, Taylor-Hood P2/P1 | `fluids.solve_stokes` | planar; axisymmetric with swirl | solve on any number; point evaluation on one |
| 3D Stokes | `fluids.solve_stokes_3d` | tetrahedra or hexahedra, optionally curved | distributed |
| steady Navier-Stokes | `fluids.solve_navier_stokes` | planar; axisymmetric with swirl | solve on any number; point evaluation on one |
| inductionless MHD | `fluids.solve_navier_stokes(..., geometry="planar25d", mhd={...})` | planar 2.5D (z-invariant) | one |
| transient Navier-Stokes (BDF1/BDF2) | `fluids.TransientNavierStokes` | planar; axisymmetric with swirl | time steps on any number; point evaluation on one |
| stabilized P1/P1 (GLS) | `fluids.gls_problem`, `fluids.solve_gls` | planar, enclosed flows | one (serial) |
| steady scalar (heat) transport, SUPG | `fluids.solve_heat`, `axionmultiphysics.transport.ScalarTransport` | planar; axisymmetric | one |
| steady Boussinesq natural convection | `fluids.solve_boussinesq` | planar; axisymmetric | one |

Problems are set up in Python from meshes, coefficients and boundary conditions; there is no design-file, viewer or
MCP interface yet. The docstrings document the arguments (`help(fluids.solve_navier_stokes)`).

## Channel and cavity flow

We solve a Poiseuille channel with the steady Navier-Stokes solver, whose exact solution it reproduces to round-off,
and the lid-driven cavity at Re = 100, steady and integrated in time from rest:

```python
import numpy as np
from axionmultiphysics import fluids

# 1. Poiseuille flow in a channel [0, 4] x [0, 1]: steady Navier-Stokes at Re = rho U H / mu = 100.
#    Boundary attributes: 1 inflow (x = 0), 2 outflow (x = 4), 3 walls.
L, H, nx, ny = 4.0, 1.0, 32, 8
V = np.array([[x, y] for y in np.linspace(0, H, ny + 1) for x in np.linspace(0, L, nx + 1)])
vid = lambda i, j: j * (nx + 1) + i
tri = [t for j in range(ny) for i in range(nx)
       for t in ([vid(i, j), vid(i + 1, j), vid(i, j + 1)], [vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])]
seg = [[vid(i, 0), vid(i + 1, 0)] for i in range(nx)] + [[vid(i + 1, ny), vid(i, ny)] for i in range(nx)] \
    + [[vid(0, j + 1), vid(0, j)] for j in range(ny)] + [[vid(nx, j), vid(nx, j + 1)] for j in range(ny)]
att = [3] * (2 * nx) + [1] * ny + [2] * ny
mesh = fluids.Mesh.from_arrays(V, np.array(tri, np.int32), np.array(seg, np.int32), np.array(att, np.int32))

rho, mu = 1.0, 0.01                                   # dynamic units: density, dynamic viscosity
profile = lambda x, y: (4 * y * (H - y) / H**2, 0.0)  # U = 1 on the centreline
sol = fluids.solve_navier_stokes(mesh, mu, rho, None, [
    {"attribute": 1, "kind": "dirichlet", "value": profile},
    {"attribute": 2, "kind": "do_nothing"},           # mu du/dn - p n = 0 (Laplacian form): p = 0 at the outlet
    {"attribute": 3, "kind": "dirichlet"}], form="laplacian")
d = sol.diagnostics
el = np.arange(mesh.n_elements, dtype=np.int32)       # evaluate at two reference points of every element
ref = np.array([[1 / 3, 1 / 3], [0.6, 0.2]])
x, _ = sol.geometry(el, ref)
u, _ = sol.velocity(el, ref)
p = sol.pressure(el, ref)
print("channel: converged", d["converged"], "| relative residual %.1e" % d["relative_residual"],
      "| max velocity error %.1e" % np.max(np.abs(u[..., 0] - 4 * x[..., 1] * (H - x[..., 1]) / H**2)),
      "| max pressure error %.1e" % np.max(np.abs(p - 8 * mu * (L - x[..., 0]) / H**2)))

# 2. Lid-driven cavity at Re = 100 on the unit square (attributes 1-4: bottom, right, top, left):
#    the steady solution, and the flow started from rest and integrated in time with error control.
cavity = fluids.Mesh.unit_square(16)
lid = lambda x, y: (1.0 if y > 1 - 1e-12 else 0.0, 0.0)
walls = [{"attribute": a, "kind": "dirichlet", "value": lid} for a in (1, 2, 3, 4)]
steady = fluids.solve_navier_stokes(cavity, mu, rho, None, walls)
print("cavity, steady: converged", steady.diagnostics["converged"], "after", steady.diagnostics["iterations"],
      "Newton steps")
ramp = lambda x, y, t: (min(t / 0.5, 1.0) * lid(x, y)[0], 0.0)   # time-dependent data take (x, y, t)
ts = fluids.TransientNavierStokes(cavity, mu, rho, None, [dict(w, value=ramp) for w in walls])
ts.initialize(lambda x, y: (0.0, 0.0))
run = ts.adapt(10.0, 1e-3, 1.0, 0.01)                 # to t = 10: rtol, velocity scale, first step
el = np.arange(cavity.n_elements, dtype=np.int32)
du = ts.snapshot().velocity(el, ref)[0] - steady.velocity(el, ref)[0]
print("cavity, transient: t =", ts.time(), "after", len(run["times"]) - 1, "steps (%d rejected)" % run["rejected"],
      "| max |u(t) - u_steady| %.1e" % np.max(np.abs(du)))
```

On one rank of an MPI build this prints (in a few seconds; the last digits can differ between builds), after the
one-time beta warning on standard error:

```text
channel: converged True | relative residual 1.5e-15 | max velocity error 1.1e-14 | max pressure error 3.8e-15
cavity, steady: converged True after 6 Newton steps
cavity, transient: t = 10.0 after 28 steps (1 rejected) | max |u(t) - u_steady| 4.8e-04
```

The parabolic profile lies in the P2 velocity space and the convective term vanishes on it, so the discrete solution
is exact up to round-off; the cavity's transient solution approaches the steady one.

## Conventions

- A mesh can also come from a model document: `axmp.mesh(d).instance(...).fluids_mesh()` returns a `fluids.Mesh` that
  copies the instance mesh, with the boundary attributes of its named boundaries (`im.attributes[name]`,
  `im.remaining_attributes`, `im.axis_attributes`); see [Meshes of model
  documents](model_meshes.md#handing-a-mesh-to-the-engines).
- Meshes are 2D in the (x, y) plane (`fluids.Mesh.from_arrays`, `Mesh.unit_square`, `Mesh.annulus`);
  axisymmetric problems use the meridional half plane `(r, z) = (x, y)`, and `fluids.Mesh3D.from_arrays` builds 3D
  meshes.
- Boundary conditions are a list of dicts, exactly one per boundary attribute; each solver's docstring lists the
  kinds it accepts.
- `solve_stokes` and `solve_stokes_3d` are kinematic (everything divided by the density); Navier-Stokes, the time
  stepper, MHD and Boussinesq take dynamic units (`mu`, `rho`, physical pressure).
- No solver raises on non-convergence: we check `sol.diagnostics["converged"]`. The fields of a solve that did not
  converge are refused.
- A steady Navier-Stokes or MHD solve with `strategy="newton"` whose step stops reducing the residual at the
  round-off floor of its own residual evaluation (which can lie above `newton_rtol` on a fine mesh) stops there and
  counts as converged, provided the residual is still within a decade of `newton_rtol`. It says so:
  `diagnostics["stopped_at_floor"]` is then true and `diagnostics["residual_floor"]` holds the measured floor. So
  `converged` means `diagnostics["relative_residual"]` is at most `newton_rtol`, or at most ten times `newton_rtol`
  with `stopped_at_floor` set; a caller who needs the number itself reads it. The stop accepts the iterate it fired
  on, so the residual it reports can be a little larger than the one the full iteration budget would have wandered
  to (on the finest Ha = 0 level of `analytic_hartmann_2d` with STRUMPACK: 1.14e-12 at the stop, 9.97e-13 after 22
  steps without it). The measured floor depends on the number of MPI ranks at round-off level, so a solve sitting at
  the floor may stop a step earlier or later, and report a slightly different residual, on another rank count;
  `converged` can then differ between a serial and a parallel run. `residual_floor_factor=0` switches the
  rule off, and `strategy="picard"` and `"ptc"`, the time steps, `solve_gls` and `solve_boussinesq` never use it.
  The [formulation notes](../physics/fluids.md) derive the floor and list its measured values.

- A verdict rests on linear algebra, and the solvers report how accurate it was: `diagnostics["lu_residual_max"]`
  (the direct solves; each history entry has its own `lu_residual`) and `diagnostics["mass_lu_residual_max"]` (the
  mass solves behind the dual norm). A final direct solve above `max(newton_rtol, 1e-11)` or a mass solve above
  `1e-10` makes the solve report `converged = false`, with the reason in `diagnostics["failure"]`.
- `TransientNavierStokes.initialize()` returns `False` when the projection of the initial field onto the discretely
  divergence-free fields misses its accuracy gate (`1e-13` on the backward error of the projection solve, at every
  `newton_rtol`, on any domain shape and in any length unit). We check that return value: after a `False`, `advance`, `step_trial`, `adapt` and
  `snapshot` raise an error naming the projection. `ts.diagnostics["projection_residual"]` and
  `ts.diagnostics["projection_converged"]` hold the residual and the verdict. STRUMPACK (the default direct solver
  of a build that has it) misses this gate on some scaled or stretched domains, depending on the processor; pass
  `lu="eigen"` there (the error says so).
- `quad_order_convection=0` (the default) integrates the convective term exactly for the element shape and geometric
  order of the mesh; a rule that under-integrates it is refused.

3D Stokes is the only distributed solver; the 2D solves can run on several ranks, but their fields are evaluated on
one, and transport, Boussinesq, MHD and GLS run on one rank. Turbulence models, compressible flow, induced magnetic
fields and 3D Navier-Stokes are not implemented.

The [walkthrough](../../examples/fluids_demo.ipynb) solves Stokes problems step by step against exact solutions. The
[formulation notes](../physics/fluids.md) give the details of the Python interface, the equations, the discretization
(including the stabilized equal-order elements) and the solvers, and the
[validation page](../validation/fluids.md) lists the verification cases.
