// Generic steady nonlinear driver: the Newton / Picard step of the steady Navier-Stokes solver with the Armijo line
// search on the discrete dual-norm merit and the continuation-stage loop, shared by NavierStokesSolver::solve() and the
// Boussinesq solver (the SAME code).  The driver owns no physics: a NonlinearSystem supplies the residual, its dual
// norm and the linear solve for a step; the state x and the residual R are single vectors in the system's own layout
// (Navier-Stokes: (u, p); Boussinesq: (u, p, T)).  The step uses elementwise vector updates only, in a fixed order, so
// that the Navier-Stokes results do not depend on which system drives them.
#pragma once

#include <mfem.hpp>

#include "axionmultiphysics/core.hpp"

#include <string>
#include <vector>

namespace axmp::fluids {

struct NonlinearStep {
  std::string kind;                 // stokes | picard | newton | ptc
  double residual = -1;             // steady relative residual AFTER the step
  double step = 1;                  // lambda used
  double dt = 0;                    // PTC pseudo-time step used (0 otherwise)
  int stage = 0;
  double viscosity_scale = 1;       // the stage scale (Navier-Stokes: viscosity; Boussinesq: buoyancy beta g)
  int lu_refine = 0;
  double lu_residual = -1;          // relative residual of the step's linear solve; -1 if not recorded
  int lu_attempts = 0;              // attempts of the step's linear solve (1: its first stage); 0 if not recorded
  std::string lu_stage;             // the stage that produced the step's solution (LUStage::name()); "" if not recorded
};

enum class StepMode { Newton, Picard };

class NonlinearSystem {
 public:
  virtual ~NonlinearSystem() = default;
  // R(x) at the stage scale s (full layout; essential rows zero)
  virtual void residual(const mfem::Vector& x, double s, mfem::Vector& R) = 0;
  virtual double dual_norm(const mfem::Vector& R) = 0;
  // solve J(x; s, mode) d = -R; d has zero essential entries; false + failure on a failed solve
  virtual bool solve_linear(const mfem::Vector& x, const mfem::Vector& R, double s, StepMode mode, mfem::Vector& d,
                            int& refine, double& lu_residual, std::string& failure) = 0;
  // The record of the last solve_linear's direct solve, for its history entry: the attempts it took and the stage
  // that produced its solution (NonlinearStep::lu_attempts, lu_stage). Default: not recorded (0, "").
  virtual void lu_record(int& attempts, std::string& stage) const {
    attempts = 0;
    stage.clear();
  }
};

struct NewtonControl {
  bool line_search = true;
  double armijo_c1 = 1e-4;
  int line_search_halvings = 6;
  double rtol = 1e-12;
  int maxit = 30;
  int n_picard = 3;
  // Stop at the measured round-off floor of the residual (NonlinearDriver::check_floor) when a step no longer
  // reduces the residual and the residual is within this factor of the floor; 0 leaves rtol as the only rule.
  double floor_factor = 0;
};

class NonlinearDriver {
 public:
  NonlinearDriver(NonlinearSystem& sys, const NewtonControl& ctl, std::vector<NonlinearStep>& history,
                  int& iterations, int& min_step_accepts, std::string& failure)
      : sys_(sys), ctl_(ctl), hist_(history), iterations_(iterations), min_accepts_(min_step_accepts),
        failure_(failure) {}

  // one Newton (or Picard) step at scale s with the stage denominator den: updates x, R and r (relative residual)
  bool step(mfem::Vector& x, mfem::Vector& R, double s, double den, bool picard, int stage, double& r);
  // Round-off floor of the relative residual of the CURRENT stage, measured the last time a step of that stage
  // failed to reduce the residual (-1: not measured in this stage), and whether the iteration stops there.
  double residual_floor() const { return floor_; }
  bool stopped_at_floor() const { return at_floor_; }
  // A stop at the floor is refused when the floor misses ctl.rtol by more than this factor, so that an accepted
  // solve still bounds its relative residual by floor_stop_rtol_ceiling * ctl.rtol.  The rule is meant for a
  // tolerance that round-off makes JUST unattainable.  Measured as relative residual / ctl.rtol at the stagnation:
  // 1.14 on the verification case that needs the rule (analytic_hartmann_2d, Ha = 0, finest level), 2.0 on the unit
  // tests' cavity (whose gate is set at half the attainable residual), and 76 on the unit tests' channel (4 x 8
  // cells, each 2.5e5 long and 0.125 high), which stagnates at a floor two hundred times ctl.rtol and is refused by
  // this ceiling alone.  A decade lies between those, well clear of both sides; it is not a fit to one of them.
  static constexpr double floor_stop_rtol_ceiling = 10;
  // Does the iteration stand at the round-off floor of its own residual evaluation?  Nothing can be gained from
  // another step then: the state x is representable only to one unit in the last place per entry, and moving it by
  // that much already changes the computed residual by the floor measured here.  Three conditions, each measured:
  // the last step OF THIS STAGE did not reduce the residual at all (a step that still converges always reduces it),
  // r is within ctl.floor_factor of the measured floor, and r is within floor_stop_rtol_ceiling of ctl.rtol.  The
  // floor is recorded whenever it is measured, whether or not the stop follows.  The measurement moves every entry
  // of the state by one unit in the last place with alternating sign: an estimate of the attainable residual, the
  // larger of the two sign patterns measured (an all-up perturbation gives about half of it).  An entry that is
  // exactly zero does not move in effect (std::nextafter from zero is subnormal), so a state with many zero entries
  // is sampled through its nonzero entries alone and the floor comes out smaller, which is the conservative
  // direction: a smaller floor stops less.  Like every round-off-level quantity the measurement depends on the
  // number of MPI ranks, so a solve that stops at the floor may stop one step earlier or later on another rank
  // count (the alternating sign follows the rank-local index).  The stop accepts the iterate it fires on; no best
  // iterate of the stage is kept.
  bool check_floor(const mfem::Vector& x, const mfem::Vector& R, double s, double den, double r);
  // the continuation stages: for each scale, (n_picard Picard steps in the first), then Newton to ctl.rtol; `lift_norm`
  // gives ||R(x_lift)||_* at a scale; r, den in/out (den and residual_lift are updated per stage)
  template <class LiftNorm>
  bool run_stages(mfem::Vector& x, mfem::Vector& R, const std::vector<double>& scales, LiftNorm&& lift_norm,
                  double& den, double& residual_lift, double& r);
  std::string scale_name = "scale";  // wording of the stage scale in the failure text ("viscosity scale" for Navier-Stokes)
  bool fail(const std::string& why) {
    if (failure_.empty()) failure_ = why;
    return false;
  }

 private:
  NonlinearSystem& sys_;
  NewtonControl ctl_;
  std::vector<NonlinearStep>& hist_;
  int& iterations_;
  int& min_accepts_;
  std::string& failure_;
  double floor_ = -1;
  bool at_floor_ = false;
  size_t cmp_begin_ = 0;  // first history entry of the current stage that check_floor may compare
};

template <class LiftNorm>
bool NonlinearDriver::run_stages(mfem::Vector& x, mfem::Vector& R, const std::vector<double>& scales,
                                 LiftNorm&& lift_norm, double& den, double& residual_lift, double& r) {
  bool ok = true;
  for (size_t j = 0; ok && j < scales.size(); ++j) {
    const double s = scales[j];
    // Each stage solves another operator against another denominator, so it has its own floor and only residuals of
    // the same stage may be compared: the Stokes start belongs to stage 0, while a later stage re-evaluates its
    // first residual without a history entry and therefore needs two steps of its own before a comparison.
    at_floor_ = false;
    floor_ = -1;
    cmp_begin_ = (j == 0 && !hist_.empty()) ? hist_.size() - 1 : hist_.size();
    if (j > 0) {
      den = lift_norm(s);
      residual_lift = den;
      sys_.residual(x, s, R);
      r = sys_.dual_norm(R) / den;
    }
    if (j == 0)
      for (int k = 0; ok && k < ctl_.n_picard && r > ctl_.rtol && !at_floor_; ++k) {
        ok = step(x, R, s, den, true, 0, r);
        if (ok && r > ctl_.rtol) check_floor(x, R, s, den, r);
      }
    int k = 0;
    while (ok && r > ctl_.rtol && !at_floor_ && k < ctl_.maxit) {
      ok = step(x, R, s, den, false, static_cast<int>(j), r);
      ++k;
      if (ok && r > ctl_.rtol) check_floor(x, R, s, den, r);
    }
    if (ok && r > ctl_.rtol && !at_floor_)
      ok = fail(strcat("Newton budget exhausted in continuation stage ", j + 1, " of ", scales.size(), " (", scale_name,
                       " ", s, ")", floor_ > 0 ? strcat(": the relative residual ", r, " stands at ", r / floor_,
                                                        " times the round-off floor of the residual") : std::string()));
  }
  return ok;
}

}  // namespace axmp::fluids
