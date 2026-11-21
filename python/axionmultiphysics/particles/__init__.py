"""Particle transport through a design's solids and field maps (beta): the driver of ``axmp-transport``.

Geant4 runs in the separate program ``axmp-transport`` (it links Geant4 and its LGPL-3 CLHEP; nothing of Geant4 enters
this process). This module plans a run into chunks of events with seeds derived from one master seed, writes the job
files, starts the program on them (``workers`` processes at a time, a failed process retried once), merges the chunks
in a fixed order and estimates the requested outputs with their statistical errors:

- ``plan`` / ``execute`` / ``merge`` (``run`` does all three), ``extend`` (more events, bitwise equal to a direct
  run), ``load`` (a merged run from disk);
- ``transport(design, results, ...)``: the route from a solved design (its regions as solids, its field as a map);
- ``resolve_materials``: NIST compositions resolved against the Geant4 the program is built with;
- the estimators (``estimate_mean``, ``estimate_fraction``, ``estimate_count``, ``batch_means``) and ``Measured``.

Every result is Monte Carlo with statistical errors only; the dependence on the physics list is not estimated.
``python -m axionmultiphysics.particles {execute,merge,extend} WORKDIR`` runs a planned work directory from the
command line (exit status 5 for a partial result). docs/guides/particles.md describes the formats, the rules and every
code. This module is **beta**: its interface may still change.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

from .._beta import warn_beta as _warn_beta
from ._common import (BATCHES, CHUNK_EVENTS, INTEGRATOR, PHYSICS_LISTS, PRODUCTION_CUT_M, QUANTUM_EV, SCHEMA_CHUNK,
                      SCHEMA_JOB, SCHEMA_RESULTS, SCHEMA_RUN, STEPPER, ParticleWarning)
from ._driver import (ParticleResults, capabilities, chunk_seed, derive_seed, event_words, execute, extend, load,
                      merge, plan, resolve_materials, run, transport)
from ._stats import Measured, batch_means, estimate_count, estimate_fraction, estimate_mean

__all__ = ["SCHEMA_JOB", "SCHEMA_CHUNK", "SCHEMA_RUN", "SCHEMA_RESULTS", "BATCHES", "CHUNK_EVENTS", "QUANTUM_EV",
           "PRODUCTION_CUT_M", "PHYSICS_LISTS", "INTEGRATOR", "STEPPER", "ParticleWarning", "Measured",
           "ParticleResults", "derive_seed", "chunk_seed", "event_words", "capabilities", "resolve_materials", "plan",
           "execute", "merge", "run", "extend", "load", "transport", "estimate_mean", "estimate_fraction",
           "estimate_count", "batch_means"]

_warn_beta("axionmultiphysics.particles")
