# Particle transport validation (beta)

Particle transport (`axionmultiphysics.particles` and `axmp-transport`, see the [guide](../guides/particles.md)) is
**beta**. This page lists every check of the transport against an independent reference, what its last run measured,
with its seed and number of events, and what is not checked. The checks are the tests
`tests/test_particles_transport.py`, `tests/test_particles_demo.py` and `tests/test_particles_driver.py`; the references
are written from physics and the design note in `tests/_particles_ref.py`, never from the implementation.

Run: Geant4 11.4.3 (multithreaded, tasking run manager), `FTFP_BERT`, production cut 0.7 mm, Geant4's default
integrator (`G4TDormandPrince45` with `G4IntegrationDriver`, deltaChord 0.25 mm, deltaOneStep 0.01 mm,
deltaIntersection 1 um, epsilon 5e-5 to 1e-3), the sequential build of the package, AMD nodes of the cluster's `cm1`
and `cm2` partitions (identical values on both), `AXMP_SLOW=1`. A single track is deterministic; every other number
is Monte Carlo, given with its statistical error (one standard deviation: of the binomial count; of a standard
deviation from N samples, 1/sqrt(2(N - 1)) relative; of a Gaussian core of N events, by 200 bootstrap resamplings
of the events, 0.9 % relative at N = 1e4, about 1.35 / sqrt(2N), which is the value given for P5). Tolerances are the design note's ceilings or the tests' own; the measured value is given beside
them.

## Fields and tracking (deterministic)

| row | case | reference | measured | asserted |
|---|---|---|---|---|
| probe | maps of kind 1 (B_r, B_z, B_phi all varying) and kind 3 (rotated, translated frame), a uniform field and a current loop, 1603 points in four regions, the world | `FieldMap.evaluate` and the loop's closed form (`tests/_particles_ref.py`) | 1.95e-12 T on a field scale of 5.15 T | 1e-12 of the scale |
| F1 | mu+ at p_T = p_z = 1 GeV/c in 1 T, ten turns (path 297 m), a 1 mm plane at every turn; B along z (Geant4's uniform field, a kind-1 map) and along x (Geant4's, a kind-3 map); seed 1, 1 event each | the exact helix | 1.12e-9 of R (z), 1.15e-9 of R (x), identical for the field and the map; \|p\| constant to 2.2e-16 | 1e-4 of R; 1e-9 |
| F1 | the accuracy parameters at 4, 2 and 1 times Geant4's defaults | the error falls at each halving | 3.37e-8, 6.12e-9, 1.12e-9 of R (a scratch Geant4 program on the same planes: 3.5e-8, 6.2e-9, 1.1e-9) | decreasing |
| F2 | 100 MeV/c protons through the field of a 0.5 m, 1 MA loop: two tracks through five planes and one through 13 planes near the loop's plane | canonical angular momentum P_phi = r p_phi + q r A_phi conserved | 1.6e-7, 2.1e-7 and 2.3e-6 of max r\|p\| at the defaults; 5.1e-8, 7.1e-8 and 1.8e-7 with deltaOneStep 1e-6 m, deltaIntersection 1e-7 m, epsilon 5e-6 to 1e-4. The value belongs to the track: it is the integrator's accuracy (below deltaOneStep / r = 5e-5) and falls as the integrator is tightened | 2.5e-5 at the defaults; 1e-6 tightened, and at least 5 times smaller for the track above 1e-6 |
| F2 | the first track through a kind-1 map of the closed form (1 cm grid) | \|dP_phi\| below q c times the integral of r \|B_map - B_exact\| along the track, plus 2.5e-5 of max r\|p\| | 5 to 1198 eV/c m against bounds of 969 to 8014 | within the bound |
| field jump | a ball (a = 0.25 m) of 1 T in 0.2 T, two map patches each correct two cells beyond its region and zero beyond; mu+ at 0.5 and 0.1 GeV/c, 60 tracks each (seed 3, a 8 cm spot) | piecewise circles with exact circle-circle intersections | 1.3e-6 m and 6e-7 rad (0.5 GeV/c), 1.1e-6 m and 3e-6 rad (0.1 GeV/c). This holds for entries deeper than about deltaChord (0.25 mm): a review measured, for tracks grazing the ball by 1e-7 to 5e-5 m, errors of up to 2.5e-3 m (0.5 GeV/c) and 7.6e-3 m (0.1 GeV/c), Geant4's miss distance, equivalent to at most deltaChord in the impact parameter; the 60 random tracks do not sample that band | 1e-4 m, 1e-4 rad |
| interfaces | 1000 protons of 30 MeV/c, isotropic, on helices through 40 vacuum slabs of 1 cm (1 T in every second slab, 0 T or 0.2 T in the others) inside a 1 mm shell (seed 5) | an interface between two regions is not the domain's boundary: every primary reaches the shell; no zero-length re-entry records a second hit | 0 of 1000 lost and 0 repeated hits in both cases; `exit_eV` equals each primary's kinetic energy with Geant4's proton mass (938.272013 MeV) to 1e-16 | none lost |
| interfaces | 200 electrons of 200 MeV through 20 iron plates of 2 mm and 2 mm gaps in 1 T, inside a 1 mm shell (seed 7) | per event, `exit_eV` at most the net kinetic energy carried into the shell (from every track's hits) | 0 of 200 events exceed it | none |
| stuck track | a mu+ of 0.1 GeV/c crossing an interface where B_x reverses (-0.2 T / +0.2 T) at 1 mrad, 3 events (seed 1) | Geant4 abandons the track as stuck (GeomNav0003); it is ended and counted, the chunk completes | each event abandons its primary: `abandoned_killed` 1, `abandoned_eV` its kinetic energy (3.98e7 eV), `exit_eV` 0; all 3 events merged | as stated |
| deltaIntersection | a job file with patches of 0.04 m and 0.02 m spacing, handed to the program directly | refused above 0.1 of the finest spacing | 0.0021 m refused (exit status 2, nothing written), 0.002 m runs | as stated |
| F4 | mu+ at 10 GeV/c in 1.5 T, 20 planes over 2 m, vacuum (seed 11, 1 event) | the exact crossings | 5.85e-6 m at the last plane (the integration error of Geant4's default parameters, growing along the track) | 1e-5 m |
| F4 | the momentum from a circle fit to those 20 hits | 10 GeV/c | p_fit / p - 1 = 4.8e-5, the integrator's scale bias | 3e-4 (a tenth of a spectrometer's resolution, the design's requirement) |
| F4 | the same with deltaOneStep 1e-6 m, deltaIntersection 1e-7 m, epsilon 5e-6 to 1e-4 | the exact crossings; 10 GeV/c | 2.15e-7 m (27 times below the defaults); 7.2e-7 | 2e-6 m and 1e-5, at least 5 times below the defaults |
| F4 | resolution, 1e4 smearings of the vacuum hits by 100 um | exact least squares; Gluckstern (PDG 2024 eq. 35.61) | (0.3050 +- 0.0022) % against 0.3017 % and 0.3045 % | 2.8 % of the least-squares value |
| F5 | a map that ends at z = 0 (plus its margin) inside a region | refused at set-up with `outside="refuse"`; with `"zero"` the calls are counted and the direction after the map's end is the exact one | refused; counted; within 3e-2 rad | as stated |
| F5 | a field-free daughter inside a mother in 0.5 T | a straight segment in the daughter | y = 0.181319 m against 0.181319 m (an inherited field would give 0.303) | 2e-5 m |

## Multiple scattering and resolution (Monte Carlo)

| row | case | reference | measured | asserted |
|---|---|---|---|---|
| P5 | mu- at 1 GeV/c through 1 cm of Fe, 1e4 events (seed 17); the projected angle's Gaussian core (central 98 %, Lynch and Dahl's definition of Highland's width) | Highland (PDG 2024 eq. 34.16): 10.10 mrad | (10.27 +- 0.10) mrad (+1.7 %) | 10 % |
| P5 | 5 mm Fe + 2.5 mm Pb | Highland with the summed x/X0: 11.55 mrad | (12.15 +- 0.12) mrad (+5.2 %) | 10 % |
| F4 | 2 m of air, 1e4 events (seed 11): the Gaussian core (central 98 %) of the momentum fitted to unsmeared hits | PDG 2024 eq. 35.63 with Highland's logarithm at the whole length: 0.1168 % | (0.1077 +- 0.0009) % (-7.8 %); the standard deviation, which includes single-scattering tails, is 0.1441 % | 15 % + 4 sigma of the estimator (20.4 %) |

## Reproducibility and statistics

| row | case | measured |
|---|---|---|
| S1 | 200 MeV electrons showering in 2 cm of Fe in 1 T (1000 events, 5 chunks, seed 7): 1 against 4 threads, 1 against 3 processes, the same seed twice, a run extended from 700 events (a short last chunk) | every chunk array, the merged 128-bit tallies and `results.json` bitwise equal; per-event hit limits reached and the same |
| S1 | a chunk run alone, after other chunks, and in a 25 000-event chunk (beyond Geant4's refill of its seeds at 10 000); 1 and 4 threads | bitwise equal; the first number of each event equals the MixMax references of the hello mode |
| S1 | a stop rule (2 % on the mean deposit, 80 planned chunks) with 1 and 4 workers | the same prefix and `results.json` |
| S1 | 81 one-event chunks of a 1 GeV electron in 20 cm of Pb with a quantum of (E + m) / 2^63 x 1.1 | the 128-bit batch sums exceed 2^63 and equal the exact integer sums of the chunks |
| S2 | 10 seeds x 2000 muons at 1 GeV/c in 1 cm of Fe, mean deposit | chi2 = 13.99 for 9 degrees of freedom (P(chi2 >) = 0.12) |
| S3 | 1e7 isotropic geantinos (seed 2024, 1000 chunks of 1e4, 8 processes), a disk of 10 deg half-angle | the 1 sigma interval of the mean covers (1 - cos 10 deg) / 2 in (69.6 +- 1.5) % of the chunks, Clopper-Pearson's in (71.8 +- 1.4) % (asserted 68.3 +- 5.9 % and >= 62.4 %); acceptance (7.590 +- 0.027)e-3 against 7.596e-3 |
| S4 | 1000 geantinos away from the disk | 0 hits: no error, the 95 % limit 1 - 0.05^(1/1000) |

The driver's own guarantees (seeds, the order of the merge, extension, prefixes and partial runs, retries, the
128-bit sums, the estimators and their printing) are checked bit for bit with a stand-in program in
`tests/test_particles_driver.py`, including S3 at the level of the estimator (1000 experiments drawn by numpy) and S4
(0 of 10 and of 1e4).

## The demonstration: a solved solenoid and a tracker

A solenoid (radius 0.5 m, length 1 m, winding 0.05 m thick, 2e5 ampere-turns, domain radius 1.5 m) with eight layers
of air 2 mm thick at r = 0.10 to 0.38 m, |z| <= 0.4 m, solved by the package (order 2) and exported as a map at tol
1e-4 with a track volume over the tracker; the partition written by hand with the same regions (the demonstration does
not use the mesher's export); mu+ at 0.2 GeV/c from the centre along +x (bore field 0.181 T, radius of curvature about
3.7 m).

| check | reference | measured | asserted |
|---|---|---|---|
| hits in vacuum (seed 1, 1 event) | the map's own trajectory (a test-side Runge-Kutta through `FieldMap.evaluate`) | 2.1e-7 m | 2e-5 m |
| momentum scale | the same circle fit on the trajectory through the exact closed-form field | R(map) / R(exact) - 1 = 7.6e-6, against the map-and-solve error along the path 7.3e-5 | 2 x 7.3e-5 + 2e-5 |
| circle fit with the path-averaged field | | 1.0018 | 2 % (a model) |
| resolution, 1e4 smearings by 50 um | exact least squares; Gluckstern (large-N form; the equal-spacing form for 8 layers gives 1.765 %) | (1.769 +- 0.013) % against 1.771 % and 1.808 % | 2.8 %; 5 % |
| scattering term in air (4000 events, seed 3): the Gaussian core (central 98 %) of the fitted curvature | PDG 2024 eq. 35.63 with Highland's logarithm (a formula for many planes; eight here: approximate) | (2.365 +- 0.044) % against 2.669 % (-11 %) (the standard deviation, with single-scattering tails, is 3.049 %) | 30 % |

The 3D iron dipole of the demonstration skips: its test calls `particles.transport` without `solids=`, which this
version does not serve (`capabilities()["partition_export"]` is false), although the mesher exports 3D partitions
(`Design.export_gdml`).

## Test status

Every assertion of the three files passes on the program and the driver of this version (the 3D dipole and, where
the application is absent, X1 skip).

X1 (the solids list read back from the exported GDML file by a standalone Geant4 application, `axmp-x1`, with a Geant4
built with GDML and Xerces-C 3.3.0) applies the same exit and hit rules as the program (interfaces are not exits; a
step after a zero-length step records no hit). Against the transport as it was during the development of this version,
with the interface rule in place, its three tests passed: volumes and materials bitwise equal, 1e4 geantino crossings
and their first random numbers bitwise equal, per-region mean deposits of 2000 electrons of 200 MeV in 0.3 T within 4
sigma (and per-event deposits bitwise equal). The comparison has not been run again against the program of this
release, which added the handling of stuck tracks, the per-writer temporary directories and the checks of the gun
afterwards.

## What is not validated

- No comparison with measured data, and no estimate of the physics-list dependence (the three lists run; their
  differences are not quantified).
- The convergence evidence of the design note's rule 3 (results with the production cut, the step and the map
  tolerance halved) is missing; it is left to the next milestone.
- X1 runs only where its application and a Geant4 built with GDML exist; its tests skip elsewhere.
- The energy balance of an event (deposits plus escaping energy against the primary's) is not checked: decays and
  hadronic interactions change the rest-mass energy, which the per-event arrays do not account for.
- The 3D dipole of the demonstration, and the transport through 3D (`tessellated`) solids lists in general (they are
  built and checked by `--geometry` only).
- Field maps of 3D solves through the transport (kind-3 maps are checked with analytic fields only).
