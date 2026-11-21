// Generic steady nonlinear driver: see cpp/include/axionmultiphysics/fluids/nonlinear_driver.hpp.
#include "axionmultiphysics/fluids/nonlinear_driver.hpp"

#include <cmath>
#include <limits>

namespace axmp::fluids {

bool NonlinearDriver::step(mfem::Vector& x, mfem::Vector& R, double s, double den, bool picard, int stage,
                           double& r) {
  sys_.residual(x, s, R);
  const double m0 = sys_.dual_norm(R);
  StepMode mode = StepMode::Newton;
  if (picard) mode = StepMode::Picard;
  mfem::Vector d, tx, TR;
  int refine = 0;
  double lures = -1;
  std::string why;
  if (!sys_.solve_linear(x, R, s, mode, d, refine, lures, why)) {
    if (!why.empty() && failure_.empty()) failure_ = why;
    return fail(std::string("linear solve failed in a ") + (picard ? "Picard" : "Newton") + " step");
  }
  double lam = 1.0;
  const bool ls = !picard && ctl_.line_search;
  if (ls) {
    bool ok = false;
    for (int h = 0; h <= ctl_.line_search_halvings; ++h) {
      lam = std::ldexp(1.0, -h);
      tx = x;
      tx.Add(lam, d);
      sys_.residual(tx, s, TR);
      const double m = sys_.dual_norm(TR);
      if (std::isfinite(m) && m <= (1.0 - ctl_.armijo_c1 * lam) * m0) {
        ok = true;
        break;
      }
    }
    if (!ok) ++min_accepts_;  // lam = 2^-halvings accepted without sufficient decrease
    x = tx;
    R = TR;
  } else {
    x.Add(1.0, d);
    sys_.residual(x, s, R);
  }
  r = sys_.dual_norm(R) / den;
  NonlinearStep h;
  h.kind = picard ? "picard" : "newton";
  h.residual = r;
  h.step = lam;
  h.dt = 0.0;
  h.stage = stage;
  h.viscosity_scale = s;
  h.lu_refine = refine;
  h.lu_residual = lures;
  sys_.lu_record(h.lu_attempts, h.lu_stage);
  hist_.push_back(h);
  ++iterations_;
  if (!std::isfinite(r)) return fail("the residual is not finite (divergence)");
  return true;
}

bool NonlinearDriver::check_floor(const mfem::Vector& x, const mfem::Vector& R, double s, double den, double r) {
  at_floor_ = false;
  if (!(ctl_.floor_factor > 0) || !std::isfinite(r) || !(den > 0)) return false;
  if (hist_.size() < cmp_begin_ + 2) return false;  // no two comparable residuals of this stage yet
  const double before = hist_[hist_.size() - 2].residual, after = hist_.back().residual;
  // The last step gained nothing: it did not reduce the residual at all.  A weaker test (the step did not halve it)
  // fires on iterations that still improve: a 4 x 8 channel 1e7 long and 1 high at rho = 1e-12 reduces its residual
  // in each of its 33 steps, by 0.2% to 20% a step and never by half.  A step without any reduction is taken as the
  // iteration's own statement that it has reached the noise of its residual; that alone does not make the residual
  // small, and the two magnitude tests below decide whether to stop.
  if (!(std::isfinite(before) && std::isfinite(after) && after >= before)) return false;
  // the floor: how much the computed residual moves when every entry of the state is moved by one unit in the last
  // place, up and down alternately (the larger of the two sign patterns measured: an all-up perturbation gives about
  // half of it; no other pattern was tried, so this is an estimate of the floor and not a bound).  Essential
  // (lifted) entries are moved too: the boundary data is representable only to the same accuracy.  An entry that is
  // exactly zero does not move in effect, so the floor is sampled through the state's nonzero entries alone, which
  // under-estimates it and therefore stops less often.
  mfem::Vector xp(x.Size()), Rp;
  for (int i = 0; i < x.Size(); ++i)
    xp(i) = std::nextafter(x(i), i % 2 == 0 ? std::numeric_limits<double>::infinity()
                                            : -std::numeric_limits<double>::infinity());
  sys_.residual(xp, s, Rp);
  Rp -= R;
  const double f = sys_.dual_norm(Rp) / den;
  if (!(f > 0) || !std::isfinite(f)) return false;
  floor_ = f;
  // The residual stands at the measured floor AND is still within a decade of the tolerance that was asked for, so
  // that a stop keeps the accepted relative residual bounded by floor_stop_rtol_ceiling * rtol however large the
  // floor is.  A floor far above rtol is not a near miss but an unattainable request (a badly scaled or far too
  // refined problem); such a solve keeps iterating and fails, with the residual and the floor in its message.
  // The stop accepts the iterate it fired on, not the best one of the stage: the step before it had a residual no
  // larger (after >= before), and a longer run may wander lower still, all within the noise the floor measures.
  at_floor_ = r <= ctl_.floor_factor * f && r <= floor_stop_rtol_ceiling * ctl_.rtol;
  return at_floor_;
}

}  // namespace axmp::fluids
