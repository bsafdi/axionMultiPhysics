// SPDX-License-Identifier: BSD-3-Clause
// The transport and probe modes: particles through a solids list (docs/guides/solids.md) in per-volume fields, with
// per-event deposits, hits and tallies written per chunk; and the probe of the fields at given points.
#pragma once

#include "job.hpp"

namespace axtr {

// Name of the integration driver every field manager uses: Geant4's templated Dormand-Prince 4(5) stepper with
// G4IntegrationDriver (G4ChordFinder's kTemplatedStepperType), checked when it is created.
inline constexpr const char* kStepper = "G4TDormandPrince45+G4IntegrationDriver";

// Runs the chunks of a transport job (exit statuses as run_job).
int run_transport(const Job& job, int threads);

// Evaluates the job's fields at its probe points, each in the volume that holds it.
int run_probe(const Job& job);

}  // namespace axtr
