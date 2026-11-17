# API reference

Lumped-element detectors, cavity haloscopes and SRF heterodyne cavities search for axion dark matter. Here we list the public objects of the `axionmultiphysics` Python package, grouped by topic.

## Designs and solves

axionMultiPhysics provides functions and classes for creating designs from templates or files, solving them and accessing the results.

::: axionmultiphysics.design.template
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.design.templates
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.design.load
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.design.Design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.results.Results
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.results.Results3D
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.results.requirement
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.global_opt.optimize
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.global_opt.global_optimize
    options:
      heading_level: 3
      show_root_full_path: false

## Lumped-element scan and reach

`scan.multipole_scan` (with `scan.reach_multiband` and `scan.reach_2211`, module `axionmultiphysics.multipole`) is beta.

::: axionmultiphysics.scan.reach
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.scan.g_qcd
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.scan.reach_fullwave
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.scan.finite_band
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.scan.multipole_scan
    options:
      heading_level: 3
      show_root_full_path: false

## Readout circuit and SQUID chains

The dc SQUID chains (`axionmultiphysics.squid`) are beta.

::: axionmultiphysics.circuit.series_readout
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.circuit.Readout
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.squid.chain
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.squid.Chain
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.squid.benchmark
    options:
      heading_level: 3
      show_root_full_path: false

## Cavity haloscopes

Everything in this section is beta, as are the related modules `axionmultiphysics.cavity_xsec`,
`axionmultiphysics.cavity_coupling` and `axionmultiphysics.haloscope_presets`.

::: axionmultiphysics.cavity.pillbox_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity.coax_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity.pillbox_modes
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity.surface_resistance
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity_track.sweep
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity_track.track
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.cavity_track.Tracking
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.Cavity
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.Receiver
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.scan_rate
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.beta_opt
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.haloscope.g_limit_scan
    options:
      heading_level: 3
      show_root_full_path: false

## SRF heterodyne

Beta: `axionmultiphysics.srf` and `axionmultiphysics.heterodyne`.

::: axionmultiphysics.srf.elliptical_cell_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.srf.tune_degeneracy
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.srf.find_mode
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.srf.figures_of_merit
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.heterodyne.SRFGeometry
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.heterodyne.scan_rate
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.heterodyne.required_volume
    options:
      heading_level: 3
      show_root_full_path: false

## Touchstone (S-, Y- and Z-parameters)

::: axionmultiphysics.touchstone.write_touchstone
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.touchstone.read_touchstone
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.touchstone.port_parameters
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.touchstone.z_to_s
    options:
      heading_level: 3
      show_root_full_path: false

## COMSOL

The import (`axionmultiphysics.comsol_import`) is beta; the export is not.

::: axionmultiphysics.comsol.export_comsol
    options:
      heading_level: 3
      show_root_full_path: false

::: axionmultiphysics.comsol_import.import_comsol
    options:
      heading_level: 3
      show_root_full_path: false

## Fluids and chemistry

The fluids (`axionmultiphysics.fluids`, `axionmultiphysics.transport`) and chemistry (`axionmultiphysics.chem`) modules
are beta. The [fluids](guides/fluids.md) and [chemistry](guides/chemistry.md) guides list their entry points, and
`help(...)` on each entry point documents its arguments.
