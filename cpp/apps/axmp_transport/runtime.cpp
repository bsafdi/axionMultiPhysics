// SPDX-License-Identifier: BSD-3-Clause
#include "runtime.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#include "G4ExceptionSeverity.hh"
#include "G4StateManager.hh"
#include "G4UImanager.hh"
#include "G4VExceptionHandler.hh"
#include "G4UIsession.hh"

namespace axtr {

namespace {

const Clock::time_point g_start = Clock::now();

class StderrDestination : public G4UIsession {
 public:
  G4int ReceiveG4cout(const G4String& msg) override {
    std::cerr << msg << std::flush;
    return 0;
  }
  G4int ReceiveG4cerr(const G4String& msg) override {
    std::cerr << msg << std::flush;
    return 0;
  }
};

thread_local bool t_track_abandoned = false;

class Handler : public G4VExceptionHandler {
 public:
  G4bool Notify(const char* origin, const char* code, G4ExceptionSeverity severity, const char* description) override {
    // Geant4's navigator abandons a track that has not moved for many steps (GeomNav0003) and asks for the event to be
    // aborted; the program ends that track instead and keeps the event (the stepping action books it)
    if (severity == EventMustBeAborted && code && std::string(code) == "GeomNav0003") {
      std::cerr << "axmp-transport: warning: Geant4 abandons a stuck track (GeomNav0003); the track is ended and "
                   "counted, the event goes on"
                << std::endl;
      t_track_abandoned = true;
      return false;
    }
    const bool fatal = severity != JustWarning;
    std::cerr << "axmp-transport: " << (fatal ? "error" : "warning") << ": Geant4 exception " << (origin ? origin : "")
              << " (" << (code ? code : "") << "): " << (description ? description : "") << std::endl;
    if (fatal) std::_Exit(kRunFailed);
    return false;
  }
};

}  // namespace

Clock::time_point process_start() { return g_start; }

bool take_abandoned_track() {
  const bool was = t_track_abandoned;
  t_track_abandoned = false;
  return was;
}

void route_geant4_output_to_stderr() {
  static StderrDestination destination;
  G4UImanager::GetUIpointer()->SetCoutDestination(&destination);
}

G4VExceptionHandler* exception_handler() {
  static Handler handler;
  return &handler;
}

void install_exception_handler() { G4StateManager::GetStateManager()->SetExceptionHandler(exception_handler()); }

}  // namespace axtr
