// Internal (not installed): the sparse LU chain in the Python boundary of the fluids solvers, shared by
// fluids_bindings.cpp and fluids_bindings3d.cpp: the lu_retry keyword, and the attempt records and counts in the
// diagnostics dictionaries.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fluids/sparse_lu.hpp"

namespace axmp::fluids::pylu {

namespace nb = nanobind;

// lu_retry: "auto" is the default chain of the backend (an empty lu_chain in the options); "off" is that chain's first
// stage alone, for every sparse LU of the solver: one attempt, judged and recorded like any other, whose solution is
// returned whether or not it meets the caller's measure (the caller's gate decides).
inline std::vector<LUStage> chain_for(const std::string& lu_retry, LUBackend lu, MatchingJob first_matching) {
  AXMP_REQUIRE(lu_retry == "auto" || lu_retry == "off", "unknown lu_retry '" + lu_retry + "'", "lu_retry",
               "Use \"auto\" (the default chain of configurations) or \"off\" (its first stage alone).");
  if (lu_retry == "auto") return {};
  return {default_chain(lu, first_matching).front()};
}

// one dict per attempt, with every field of LUAttempt
inline nb::list attempts_list(const std::vector<LUAttempt>& attempts) {
  nb::list l;
  for (const LUAttempt& a : attempts) {
    nb::dict d;
    d["stage"] = a.stage;
    d["outcome"] = a.outcome;
    d["reason"] = a.reason;
    d["measure"] = a.measure;
    d["bound"] = a.bound;
    d["rel_residual_before"] = a.rel_residual_before;
    d["rel_residual"] = a.rel_residual;
    d["gate_residual"] = a.gate_residual;
    d["backward_error"] = a.backward_error;
    d["constraint_ratios"] = a.constraint_ratios;
    d["refine_steps"] = a.refine_steps;
    d["krylov_iterations"] = a.krylov_iterations;
    d["factor_nnz"] = a.factor_nnz;
    d["index_bits"] = a.index_bits;
    d["analyze_seconds"] = a.analyze_seconds;
    d["factor_seconds"] = a.factor_seconds;
    d["solve_seconds"] = a.solve_seconds;
    l.append(d);
  }
  return l;
}

// the last direct solve: the stage that produced its solution, its backward error and its attempts
inline void put_record(nb::dict& r, const LUStats& st, const std::string& stage, const std::vector<LUAttempt>& at) {
  r["lu_stage"] = stage;
  r["lu_backward_error"] = st.backward_error;
  r["lu_attempts"] = attempts_list(at);
}

// the counts of a call (or of a run); lu_attempts_total because lu_attempts is the last solve's list
inline void put_counts(nb::dict& r, const LUCounts& c, long mass_escalated) {
  r["lu_solves"] = c.solves;
  r["lu_escalated"] = c.escalated;
  r["lu_escalated_final"] = c.escalated_final;
  r["lu_attempts_total"] = c.attempts;
  r["lu_rejected"] = c.rejected;
  r["lu_failed"] = c.failed;
  nb::dict by;
  for (const auto& kv : c.accepted_by) by[kv.first.c_str()] = kv.second;
  r["lu_accepted_by"] = by;
  r["lu_rejected_seconds"] = c.rejected_seconds;
  r["mass_lu_escalated"] = mass_escalated;
}

}  // namespace axmp::fluids::pylu
