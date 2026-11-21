# Fluids verification (beta)

The fluids module (`axionmultiphysics.fluids` with `axionmultiphysics.transport`) is **beta**. This page lists every
verification case of the module and what its last run measured. The formulation is described in the
[fluids guide](../guides/fluids.md); the case format and the runner in `benchmarks/fluids/README.md`.

## What the cases check

- **Manufactured solutions** (`mms`): observed orders of convergence of the velocity, pressure, temperature and
  potential errors (L2, H1) against the orders of the elements, for Taylor-Hood Stokes (planar, axisymmetric with
  swirl, 3D), steady and transient Navier-Stokes, SUPG scalar transport, Boussinesq coupling, stabilized P1/P1 (GLS)
  and inductionless MHD, with every boundary-condition kind (Dirichlet, traction, do-nothing, slip and symmetry,
  periodic, axis, pressure point).
- **Closed-form solutions** (`analytic`): Couette and Poiseuille flows (reproduced to round-off where the solution is
  in the discrete space), circular Couette, Hagen-Poiseuille, Stokes drag on a sphere, Kovasznay and Taylor-Green
  flows, Burgers vortex, Womersley flow, Graetz problems, the Hartmann, Shercliff and Hunt ducts, rest states.
- **Discrete identities** (`conservation`): energy balances of the convective forms and of the transient scheme, heat
  balance of the Boussinesq solution, mass conservation, solver and quadrature sensitivity.
- **Other codes** (`cross_code`): NGSolve 6.2.2608 and DOLFINx 0.11.0 solving the same discrete problem (same meshes,
  elements, quadrature and boundary data): Stokes cavities (planar, axisymmetric with swirl, 3D), the Navier-Stokes
  cavity at Re = 1000, a transient cavity, natural convection at Ra = 1e5, stabilized P1/P1 cavities and MHD ducts.
  Agreement shows that the same discrete problem is solved; it says nothing about the adequacy of the mesh or model.

## What is not verified

- **No published benchmark value is compared.** Lid-driven cavity, flow past a cylinder, backward-facing step and
  natural-convection reference values were never extracted into cases, so the `literature` verdict is `NOT_TESTED`
  for every case.
- No comparison with experiments.
- Two cases, `rank_invariance_stokes_3d` and `xcode_stokes_cavity_reg_3d`, have only quick-mode results (their first
  two levels); their orders and fine-mesh comparisons are not confirmed.
- The Eigen fallback (a build without STRUMPACK) crashed before producing any result for six 3D Stokes cases in the
  2026-11-17 run: a segmentation fault in `SparseLU::factorize` on rank 0, in every case in the first solve on the
  finest refinement level (859,812 unknowns in `mms_stokes_3d`), after the three coarser levels had completed (see "Last
  run"). Those six cases have results only from the two STRUMPACK builds. The cause was Eigen's 32-bit factor offsets:
  the factor had passed 2^31 entries in `lusup` (after 3 h 13 min, 27.9 GiB resident on rank 0 in the diagnosis run).
  The Eigen backend now uses 64-bit indices above 46,340 unknowns, so such a system is factorized, or fails with an
  error that names the sizes and the remedies when Eigen itself runs out of memory (nothing is checked before the
  factorization, and the job can be killed by the system before Eigen reports anything); it remains limited by rank 0's memory and time (measured on one core of a shared AMD EPYC 7401 node, the 3D Taylor-Hood Stokes system with 49 073 unknowns (4.5e6 matrix entries) took 122 s (127 s with 32-bit indices) and 2.9 GiB, and the one with 112 725 unknowns (1.0e7 entries) had not been factorized after 2 h and held at least 10.9 GB after 22 min), and these
  six cases are not re-run in that build.
- The Eigen fallback is also much more expensive on the 2D cases it completed: `mms_stokes_2d_periodic` took 2.65 h
  against 1.4 to 1.6 min with STRUMPACK and peaked at 27 GiB of resident memory against 1.0 GiB; the Womersley cases
  took 2.5 to 2.9 h against 1.2 to 2.5 h.
- Features the module does not offer (see the limitations in the guide) are not covered: open-boundary
  stabilization, turbulence models, 3D Navier-Stokes, transient Boussinesq or MHD, multi-rank solves other than 3D
  Stokes.

## Last run

Three runs are combined in the tables: the two below and, for one case, the rerun described in the third paragraph.

*2026-11-08* (axionMultiPhysics 0.1.0): an MPI build (GCC 14, Release; MFEM 4.10.0 with METIS; no STRUMPACK, so the
direct solves use the Eigen fallback) on one 128-core CPU node. Every short case was run in full (`run --cost short`),
the long cases except the 3D (multi-rank), Womersley and periodic ones in full, and every case in quick mode
(`run --all --quick`).

*2026-11-17* (axionMultiPhysics 0.1.0, commit `9d397749`): the twelve cases that the first run had left in quick mode
(the seven 3D Stokes cases, the four Womersley cases and `mms_stokes_2d_periodic`) were run in full, on `lr7` nodes with
GCC 13.2, Open MPI 4.1.6, Release and MFEM 4.10.0 with METIS, in three builds: *strumpack-serial* (serial STRUMPACK
8.0.0), *strumpack-mpi* (STRUMPACK 8.0.0 with MPI and ScaLAPACK; its first build failed to link because CMake picked the
system GCC 8 `gfortran`, and it was rebuilt with the GCC 13 `mpifort`) and *nostrumpack* (Eigen fallback). The 3D Stokes
cases ran at 16 MPI ranks on one node (except `control_taylor_hood_3d`, which runs on one rank), the Womersley and
periodic cases on one rank; on these one-rank cases the strumpack-mpi build recorded the serial STRUMPACK path
(`direct:strumpack`) and its results are bit-identical to strumpack-serial, so distributed STRUMPACK was exercised in 7
of the 12 cases. Outcome: all twelve meet all their criteria in strumpack-serial and in strumpack-mpi. In nostrumpack
the five 2D cases and `control_taylor_hood_3d` meet all their criteria, and the other six 3D Stokes cases crashed (Eigen
`SparseLU::factorize`, segmentation fault on rank 0, after 1 to 2 hours; the 32-bit offset overflow described in
the limitations, since fixed) and have no result. The 3D Stokes errors agree
to 1.2e-10 relative or better between the two STRUMPACK builds, and the 2D errors to 2.6e-10 or better between Eigen and
STRUMPACK; two derived `time_study_spatial_contamination_*` quantities, differences of two nearly equal temporal errors,
differ by up to 8.7e-9. The STRUMPACK solves returned correct results but needed retries: 55 of 192 first factorizations
in strumpack-serial and 56 of 192 in strumpack-mpi failed without row matching and were repeated with it, residuals
before iterative refinement reached 2e-2, and in the Womersley 2D cases 6 transient solves (4 in
`analytic_womersley_2d_space`, 2 in `analytic_womersley_2d_time`) were each repeated twice: both STRUMPACK attempts
(with matching, then without) were rejected, either by a residual above 1e-6 or by a failed factorization or solve (the
records do not say which), and Eigen finished the solve (12 recorded fallback transitions in all). The raw results, logs and run scripts of these runs
are not published; the paragraph above and the table below are their summary.

*2026-11-19* (re-run of the two planar Womersley cases, `analytic_womersley_2d_space` and `analytic_womersley_2d_time`, in
full mode, in the strumpack-serial and the strumpack-mpi configurations of the same cluster (AMD EPYC 7662 node, one
core per case), after the default chain of the sparse LU changed to STRUMPACK with MC64 job 5, then MC64 job 2, then
Eigen's LU): all criteria met in all four runs. The six solves that were repeated twice in the runs above are now
accepted by `strumpack:mc64-2` after one rejected stage (`lu_escalated` 1, `lu_fallbacks` 1,
`lu_fallback_backend` `strumpack:mc64-2` instead of 2 and `eigen`); no solve failed. Every checked and fitted quantity agrees with
the runs above: the quantities of `analytic_womersley_2d_space` and the error series to 2.5e-13 relative or better there, the two
`time_study_spatial_contamination_*` quantities (differences of nearly equal temporal errors, tolerance 0.1) to 5.7e-9
relative, the other series of `analytic_womersley_2d_time` to 2.4e-11, and the fitted orders and drifts to 6.1e-11 (space)
and 8.7e-10 (time). The two configurations agree with each other bit for bit in every quantity and series.

*2026-11-18* (the preparation of release 0.2.0, a build made before the version number was raised, which reports 0.1.0): `analytic_hartmann_2d` alone, in full, in an MPI build with serial
STRUMPACK 8.0.0 (GCC 13.2, Open MPI 4.1.6, Release, MFEM 4.10.0) on one core of a Xeon Gold 6330 node of the
same cluster, after the change of its stopping rule (note below): all criteria met, 288 s. Its row is `report` output for this run, and the totals line was corrected by hand for it (note below); no
other case was rerun.

The table shows the full result where there is one; for the twelve cases of the second run it is the strumpack-mpi
result (the other builds that completed give the same orders, deviations and checks). The quick-mode results of
`rank_invariance_stokes_3d` and `xcode_stokes_cavity_reg_3d` were not rerun and are unchanged.

Columns: *evidence* is the kind of check (above); *run* is `full` (every refinement level) or `quick` (the first two
levels only: orders and Richardson extrapolation cannot be judged, so such a row is a smoke test and its numbers are
indicative); *result* is `all criteria met` (in quick mode `all criteria met (on two levels)` or `smoke only`) or the
failed acceptance items; *measured* gives the observed orders minus the expected ones over the case's error series,
the largest deviation of a comparison as a fraction of its tolerance (1 = at the tolerance), and the number of
identity and bound checks.

**Summary.** 79 of the 81 cases were run in full and all 79 meet all their criteria; the observed orders agree
with the expected ones within each case's tolerance (787 observed orders in the full runs), and the cross-code
comparisons agree within their tolerances. The two cases without a full run are `rank_invariance_stokes_3d` and
`xcode_stokes_cavity_reg_3d`. `analytic_hartmann_2d` was run again after a fix of its stopping rule; see the note
below.

!!! note "`analytic_hartmann_2d`: rerun after the fix"
    What on this page is generated and what is not. The tables below are the output of `benchmarks/fluids/run.py
    report` for the runs of 2026-11-08 and 2026-11-17, except one row: the `analytic_hartmann_2d` row is `report`
    output for a run of that case alone on 2026-11-18. The totals line under the tables ("81 cases: ...") is not
    generated: it is the totals line of the earlier report plus the change of that row (one more case meeting all its
    criteria, one more comparison, four more checks), computed by hand, because the raw results of the earlier runs
    are not kept and `report` cannot be run over all cases again. The page has no marked generated block; the
    summary paragraph above the tables, which quotes these counts, was adjusted by hand in the same way.

    `analytic_hartmann_2d` failed in the first runs (its row then read `FAILED: solver`, with 11 comparisons and 11
    checks). It was run again on 2026-11-18, alone, in full, with the fix described below (a build made in the preparation of
    release 0.2.0, before the version number was raised, so it reports 0.1.0; MPI build with serial STRUMPACK 8.0.0,
    one core of a Xeon Gold 6330 node, 288 s) and meets all its criteria
    with 22 orders, 12 comparisons and 15 checks. The other 78 full runs were made before the fix and were not
    repeated, so the tables are not the output of one build of one commit; whether the fix changes any of them was
    not measured. This note is written by hand.

    The failure was a gate below the round-off floor of the residual, not a wrong solution: the attainable relative
    residual of this problem grows like h^(-3/2), and on that level the measured floor is 2.2e-12 (the
    [formulation notes](../physics/fluids.md) derive it and list the measured series). Whether the gate is met at
    all is decided by round-off there: with the whole iteration budget the Eigen fallback ends at 1.01e-12 and
    STRUMPACK dips just under 1e-12 at its twenty-second step. The solver now measures that floor and stops at it
    when a step no longer reduces the residual and the residual is still within a decade of `newton_rtol`,
    reporting both in the diagnostics (`residual_floor`, `stopped_at_floor`). A re-run of the case alone then
    meets all of its criteria, in both LU backends: with STRUMPACK (MPI build, one core) it stops at 1.14e-12
    against a measured floor of 2.19e-12 after four direct solves instead of 34 — exactly one of the case's 61
    solves — and the state accepted there is gated by a closed-form comparison of the Ha = 0 flow rate (deviation
    1.7e-11 of an allowance of 1.3e-10) and by two patch checks on the same level (6.7e-12 and 2.1e-13 against
    1e-10 and 1e-11). The case also gained a bound of 1e-11 on the largest unscaled Euclidean residual of its
    solves (measured 1.5e-12) and a check that every solve reported one, so it now has 12 comparisons and 15
    checks.

### Taylor-Hood Stokes, planar

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_annulus_slip_2d` | mms, analytic | long | full | all criteria met | 8 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.01 to 0.05; 1 comparison (closed_form), largest deviation 3.4e-05 of the allowance; 2 checks |
| `analytic_couette_2d` | analytic | short | full | all criteria met | 14 comparisons (closed_form), largest deviation 2.2e-04 of the allowance; 2 checks |
| `analytic_noflow_stokes_2d` | analytic | short | full | all criteria met | 6 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.03 to 0.03; 7 checks |
| `analytic_poiseuille_channel_2d` | analytic, conservation | short | full | all criteria met | 15 comparisons (closed_form), largest deviation 5.7e-05 of the allowance; 5 checks |
| `control_p1p1_stokes_2d` | mms | short | full | all criteria met | 7 checks |
| `mms_stokes_2d` | mms, conservation | long | full | all criteria met | 22 orders (expected 2.0, 3.0), observed minus expected -3.8e-03 to 0.06; 7 checks |
| `mms_stokes_2d_do_nothing` | mms | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -1.5e-03 to 0.02; 3 checks |
| `mms_stokes_2d_periodic` | mms | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -1.2e-03 to 7.4e-03; 4 checks |
| `mms_stokes_2d_pressure_point` | mms | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -2.2e-04 to 0.06; 6 checks |
| `mms_stokes_2d_slip` | mms | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -2.1e-03 to 0.06; 4 checks |
| `mms_stokes_2d_traction` | mms | long | full | all criteria met | 16 orders (expected 2.0, 3.0), observed minus expected -2.4e-03 to 0.02; 3 checks |
| `selfconv_stokes_cavity_reg_2d` | conservation | long | full | all criteria met | 4 Richardson/GCI (orders 3.94 to 3.95); 5 checks |
| `xcode_stokes_cavity_reg_2d` | cross_code | long | full | all criteria met | 8 comparisons (ngsolve), largest deviation 6.1e-04 of the allowance; 2 checks |

### Taylor-Hood Stokes, axisymmetric with swirl

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_circular_couette_axi` | analytic, conservation | long | full | all criteria met | 11 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.01 to -3.0e-03; 4 comparisons (closed_form), largest deviation 3.9e-05 of the allowance; 3 checks |
| `analytic_hagen_poiseuille_pipe` | analytic, conservation | short | full | all criteria met | 12 comparisons (closed_form), largest deviation 1.3e-05 of the allowance; 5 checks |
| `analytic_stokes_sphere_drag` | analytic, conservation | long | full | all criteria met | 21 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.05 to 3.8e-03; 4 comparisons (closed_form), largest deviation 0.15 of the allowance; 7 checks |
| `mms_stokes_axi` | mms | long | full | all criteria met | 24 orders (expected 2.0, 3.0), observed minus expected -1.7e-03 to 0.02; 8 checks |
| `mms_stokes_axi_axis` | mms, conservation | long | full | all criteria met | 8 orders (expected 2.0, 3.0), observed minus expected -1.7e-03 to 0.02; 14 checks |
| `mms_stokes_axi_do_nothing` | mms | long | full | all criteria met | 8 orders (expected 2.0, 3.0), observed minus expected -3.7e-03 to 0.01; 4 checks |
| `mms_stokes_axi_traction` | mms | long | full | all criteria met | 18 orders (expected 2.0, 3.0), observed minus expected -5.3e-03 to 9.6e-03; 4 checks |
| `xcode_stokes_axi_cavity_swirl` | cross_code | long | full | all criteria met | 8 comparisons (ngsolve), largest deviation 2.7e-04 of the allowance; 2 checks |

### Taylor-Hood Stokes, 3D (MPI)

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_duct_poiseuille_3d` | analytic, conservation | long | full | all criteria met | 1 Richardson/GCI (orders 3.63 to 3.63); 1 comparison (series), largest deviation 0.33 of the allowance; 17 checks |
| `analytic_stokes_sphere_3d` | analytic | long | full | all criteria met | 28 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.04 to 0.32; 7 Richardson/GCI (orders 3.68 to 3.96); 6 comparisons (closed_form), largest deviation 0.35 of the allowance; 9 checks |
| `control_taylor_hood_3d` | conservation | long | full | all criteria met | 7 checks |
| `mms_stokes_3d` | mms, conservation | long | full | all criteria met | 24 orders (expected 2.0, 3.0), observed minus expected -0.04 to 0.03; 13 checks |
| `mms_stokes_3d_do_nothing` | mms | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -1.6e-03 to 0.03; 9 checks |
| `mms_stokes_3d_slip` | mms, analytic | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -8.3e-03 to 0.02; 11 checks |
| `mms_stokes_3d_traction` | mms | long | full | all criteria met | 16 orders (expected 2.0, 3.0), observed minus expected -0.02 to 0.03; 9 checks |
| `rank_invariance_stokes_3d` | conservation | long | quick | all criteria met (on two levels) | 8 checks |
| `xcode_stokes_cavity_reg_3d` | cross_code | long | quick | all criteria met (on two levels) | 8 comparisons (dolfinx), largest deviation 1.1e-05 of the allowance; 4 checks |

### Steady Navier-Stokes

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_burgers_vortex_axi` | analytic | long | full | all criteria met | 15 orders (expected 2.0, 3.0), observed minus expected 5.7e-04 to 0.10; 4 checks |
| `analytic_circular_couette_ns_axi` | analytic | long | full | all criteria met | 6 orders (expected 2.0), observed minus expected 1.5e-03 to 1.9e-03; 3 checks |
| `analytic_kovasznay_re40` | analytic, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -1.1e-03 to 0.03; 5 checks |
| `cons_ns_energy_identity` | conservation | long | full | all criteria met | 29 checks |
| `mms_ns_axi` | mms, conservation | long | full | all criteria met | 37 orders (expected 2.0, 3.0), observed minus expected 1.4e-03 to 0.12; 5 checks |
| `mms_ns_newton_jacobian` | mms | short | full | all criteria met | 24 checks |
| `mms_ns_steady_2d` | mms, conservation | long | full | all criteria met | 24 orders (expected 2.0, 3.0), observed minus expected 1.5e-03 to 0.13; 7 checks |
| `mms_ns_steady_2d_traction` | mms, conservation | long | full | all criteria met | 16 orders (expected 2.0, 3.0), observed minus expected 1.4e-03 to 0.13; 3 checks |
| `selfconv_ns_cavity_continuation` | conservation | long | full | all criteria met | 8 checks |
| `xcode_cavity_re1000` | cross_code | long | full | all criteria met | 24 comparisons (ngsolve), largest deviation 1.4e-04 of the allowance; 1 check |

### Transient Navier-Stokes

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_taylor_green_2d` | analytic, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected 0.01 to 0.11; 2 checks |
| `analytic_womersley_2d_space` | analytic | long | full | all criteria met | 4 orders (expected 2.0, 3.0), observed minus expected -7.9e-03 to -6.5e-03; 1 check |
| `analytic_womersley_2d_time` | analytic | long | full | all criteria met | 12 orders (expected 1.0, 2.0), observed minus expected -0.06 to 0.02; 3 checks |
| `analytic_womersley_axi_space` | analytic | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -8.8e-03 to -6.6e-03; 1 check |
| `analytic_womersley_axi_time` | analytic | long | full | all criteria met | 12 orders (expected 1.0, 2.0), observed minus expected -0.04 to 0.02; 3 checks |
| `cons_ns_transient_energy` | conservation | long | full | all criteria met | 48 checks |
| `mms_ns_transient_error_control` | mms | long | full | all criteria met | 16 checks |
| `mms_ns_transient_initial` | mms | long | full | all criteria met | 4 orders (expected 2.0, 3.0), observed minus expected -1.0e-03 to -9.8e-04; 3 checks |
| `mms_ns_transient_temporal` | mms | long | full | all criteria met | 96 orders (expected 1.0, 2.0), observed minus expected -0.08 to 9.6e-03; 2 checks |
| `xcode_ns_transient_cavity` | cross_code | long | full | all criteria met | 36 comparisons (ngsolve), largest deviation 2.9e-06 of the allowance; 1 check |

### Scalar transport and Boussinesq

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_advdiff_layer_2d` | analytic | long | full | all criteria met | 7 checks |
| `analytic_conduction_annulus_axi` | analytic | short | full | all criteria met | 4 orders (expected 2.0, 3.0, 4.0), observed minus expected -5.5e-03 to -2.9e-03; 1 check |
| `analytic_conduction_limit_cavity` | analytic, conservation | long | full | all criteria met | 7 checks |
| `analytic_graetz_pipe` | analytic | long | full | all criteria met | 15 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.05 to 2.5e-03; 3 checks |
| `analytic_graetz_planar` | analytic | long | full | all criteria met | 12 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.06 to 0.02; 1 check |
| `analytic_hydrostatic_noflow` | analytic, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -1.9e-05 to 0.57; 17 checks |
| `cons_heat_balance` | conservation | long | full | all criteria met | 2 checks |
| `mms_advdiff_2d` | mms, conservation | long | full | all criteria met | 18 orders (expected 1.0, 2.0, 3.0), observed minus expected -0.03 to 0.04; 9 checks |
| `mms_advdiff_axi` | mms, conservation | long | full | all criteria met | 12 orders (expected 1.0, 2.0, 3.0), observed minus expected -3.4e-03 to 0.01; 10 checks |
| `mms_boussinesq_2d` | mms | long | full | all criteria met | 34 orders (expected 2.0, 3.0), observed minus expected -4.9e-03 to 0.12; 4 checks |
| `mms_boussinesq_axi` | mms | long | full | all criteria met | 24 orders (expected 2.0, 3.0), observed minus expected -3.5e-04 to 0.05; 4 checks |
| `mms_boussinesq_newton_jacobian` | mms | short | full | all criteria met | 16 checks |
| `xcode_natconv_ra1e5` | cross_code | long | full | all criteria met | 28 comparisons (ngsolve), largest deviation 5.1e-06 of the allowance |

### Stabilized P1/P1 (GLS)

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_gls_kovasznay_re40` | analytic, conservation | long | full | all criteria met | 8 orders (expected 1.0, 1.8, 1.9), observed minus expected -0.03 to 0.03; 2 checks |
| `control_gls_no_pspg` | mms | long | full | all criteria met | 14 checks |
| `gls_residual_jacobian` | mms | short | full | all criteria met | 12 checks |
| `gls_tau_pins` | mms | short | full | all criteria met | 21 checks |
| `mms_gls_ns_2d` | mms, conservation | long | full | all criteria met | 32 orders (expected 1.0, 1.5, 1.9, 2.0), observed minus expected -0.10 to 0.08; 2 checks |
| `mms_gls_stokes_2d` | mms, conservation | long | full | all criteria met | 20 orders (expected 1.0, 1.5, 2.0), observed minus expected -0.02 to 0.06; 2 checks |
| `mms_gls_transient_small_dt` | mms | long | full | all criteria met | 8 checks |
| `patch_gls_consistency` | mms | short | full | all criteria met | 6 checks |
| `xcode_gls_cavity` | cross_code | long | full | all criteria met | 48 comparisons (ngsolve), largest deviation 8.0e-08 of the allowance; 1 check |
| `xcode_gls_vs_taylor_hood` | cross_code | long | full | all criteria met | 30 comparisons (ngsolve_th), largest deviation 0.68 of the allowance; 10 checks |

### Inductionless MHD

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_hartmann_2d` | analytic, conservation | long | full | all criteria met | 22 orders (expected 2.0, 3.0), observed minus expected -0.05 to -1.7e-03; 12 comparisons (closed_form), largest deviation 0.13 of the allowance; 15 checks |
| `analytic_hartmann_suction_2d` | analytic | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -0.02 to -4.7e-03; 4 comparisons (closed_form), largest deviation 0.02 of the allowance; 8 checks |
| `analytic_hunt_duct_2d` | analytic, conservation | long | full | all criteria met | 15 comparisons (closed_form), largest deviation 0.07 of the allowance; 10 checks |
| `analytic_mhd_annulus_variable_b` | analytic | long | full | all criteria met | 21 orders (expected 2.0, 3.0), observed minus expected -0.03 to 8.8e-03; 10 comparisons (closed_form), largest deviation 0.04 of the allowance; 9 checks |
| `analytic_shercliff_duct_2d` | analytic, conservation | long | full | all criteria met | 11 comparisons (closed_form), largest deviation 0.03 of the allowance; 12 checks |
| `mms_mhd_2d` | mms, conservation | long | full | all criteria met | 21 orders (expected 2.0, 3.0), observed minus expected -4.8e-03 to 0.13; 5 checks |
| `mms_mhd_2d_insulating` | mms, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -8.6e-03 to 5.4e-04; 6 checks |
| `xcode_mhd_duct_2d` | cross_code | long | full | all criteria met | 14 comparisons (ngsolve), largest deviation 0.01 of the allowance |

81 cases: 79 run in full (79 meet all their criteria), 2 run in quick mode only, 0 not run. The full runs contain 787
observed orders of convergence, 12 Richardson/GCI estimates, 305 comparisons with reference values and 572 checks.

## Reference sources

| source | what it is |
|---|---|
| `closed_form` | exact solution evaluated in double precision |
| `series` | separated-variable series of the rectangular duct in high-precision decimal arithmetic (`tools/fluids_mms/duct_series_reference.py`) |
| `ngsolve` | NGSolve 6.2.2608 on the same discrete problem (scripts and outputs in `tools/fluids_mms/xcode/`) |
| `ngsolve_th` | NGSolve 6.2.2608, Taylor-Hood P2/P1 on the same meshes (the GLS cases compare against it with a tolerance, not to round-off; script `tools/fluids_mms/xcode/ngsolve_th_cavity_reference.py`) |
| `dolfinx` | DOLFINx 0.11.0 with PETSc 3.25.6 on the same discrete problem |

Regenerate the tables with `python benchmarks/fluids/run.py report DIR [DIR ...] --track fluids` from the `--out`
directories of `run` (a full result is preferred to a quick one).
