"""Constants, errors and warnings shared by the parts of the particle-transport driver.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from .. import _core

SCHEMA_JOB = "axionmultiphysics.transport-job/1"
SCHEMA_CHUNK = "axionmultiphysics.transport/1"
SCHEMA_RUN = "axionmultiphysics.particles-run/1"
SCHEMA_RESULTS = "axionmultiphysics.particles-results/1"
BATCHES = 20
CHUNK_EVENTS = 1000
QUANTUM_EV = 1e-3
PRODUCTION_CUT_M = 7e-4
PHYSICS_LISTS = ("FTFP_BERT", "FTFP_BERT_EMZ", "QGSP_BIC")
# Geant4's own defaults (G4FieldManager, G4ChordFinder), in metres
INTEGRATOR = {"delta_chord_m": 2.5e-4, "delta_one_step_m": 1e-5, "delta_intersection_m": 1e-6,
              "epsilon_min": 5e-5, "epsilon_max": 1e-3, "min_step_m": 1e-5}
STEPPER = "G4TDormandPrince45+G4IntegrationDriver"
STAGE = INSTANCE = "transport"
E_CHARGE = 1.602176634e-19        # J per eV (exact, SI 2019)


class ParticleWarning(UserWarning):
    """A warning of the particle transport; its message starts with its code (for example
    ``events.partial: ...``)."""


def error(code: str, message: str, hint: str) -> Exception:
    exc = _core.AxionMultiPhysicsError(f"{message}. Hint: {hint}")
    exc.code = code
    exc.hint = hint
    return exc


def warn(code: str, message: str, stacklevel: int = 3) -> dict:
    warnings.warn(f"{code}: {message}", ParticleWarning, stacklevel=stacklevel)
    return {"code": code, "message": f"{code}: {message}"}


def real(x) -> bool:
    return (isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))
            and math.isfinite(float(x)))
