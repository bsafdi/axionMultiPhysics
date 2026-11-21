// SPDX-License-Identifier: BSD-3-Clause
// Process-wide pieces shared by the modes: the start time, the routing of Geant4's output to stderr and the handling
// of Geant4 exceptions.
#pragma once

#include <chrono>

class G4VExceptionHandler;

namespace axtr {

// Exit statuses of axmp-transport.
enum ExitCode : int { kOk = 0, kUsage = 2, kRefused = 3, kRunFailed = 4 };

using Clock = std::chrono::steady_clock;

// The time at which the process began running our code (a static initialiser, before main).
Clock::time_point process_start();

// Sends G4cout and G4cerr of the calling thread to stderr, so that stdout stays free for the program's own output.
// Call on the main thread before anything of Geant4 can print, and on every worker thread when it starts (a worker
// has a destination of its own, which otherwise writes to stdout).
void route_geant4_output_to_stderr();

// The handler installed on every thread: a warning is printed to stderr and the run goes on; a stuck track that
// Geant4's navigator abandons (GeomNav0003) is flagged for the transport to end (see take_abandoned_track); any other
// exception, including those that Geant4's own handler answers by aborting one event or one run, prints and ends the
// process with status 4 (no core dump). An event is never dropped silently: a run with an aborted event does not
// produce output.
G4VExceptionHandler* exception_handler();
void install_exception_handler();

// Whether Geant4 abandoned a stuck track on the calling thread since the last call (and clears it). The handler turns
// Geant4's request to abort the event for a stuck track (GeomNav0003) into this flag; the transport ends the track.
bool take_abandoned_track();

}  // namespace axtr
