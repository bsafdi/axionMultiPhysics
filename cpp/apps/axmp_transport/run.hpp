// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "job.hpp"
#include "runtime.hpp"

namespace axtr {

// Runs the chunks of a validated job in this process with `threads` worker threads and writes their outputs and the
// run record. Returns an ExitCode: kRefused when the set-up is refused (before any event), kRunFailed when the run
// fails. Prints its errors to stderr.
int run_job(const Job& job, int threads);

// The version of Geant4 this program is built with, "X.Y.Z".
std::string geant4_version();

}  // namespace axtr
