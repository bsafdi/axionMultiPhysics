"""Estimators of the particle-transport driver: the measured value with its error, Clopper-Pearson intervals, exact
zero-count limits, batch means, and the PDG rule for printing a value with its error.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

ONE_SIGMA = math.erf(1.0 / math.sqrt(2.0))     # 0.6827: the central interval of one Gaussian sigma
LIMIT_CL = 0.95                                 # confidence level of the one-sided limit after zero events


class Measured(float):
    """A Monte Carlo estimate: a float that carries its statistical error and how it was obtained.

    Attributes: ``stat_error`` (None when there is none), ``error`` (the estimator: "sem", "clopper_pearson",
    "poisson", "batch_means" or "upper_limit"), ``events`` (events of the sample), ``events_defined`` (events where the
    variable is defined and the selection holds), ``units``, ``mc`` (always True), ``upper_limit`` and ``cl`` (after
    zero events), ``interval`` (a fraction's Clopper-Pearson interval) and ``reason`` (why there is no error).
    Arithmetic on a Measured returns a plain float: the error is never carried along silently."""

    def __new__(cls, value, *, stat_error=None, error=None, events=None, events_defined=None, units="", mc=True,
                upper_limit=None, cl=None, interval=None, reason=None):
        m = float.__new__(cls, value)
        m.stat_error = None if stat_error is None else float(stat_error)
        m.error = error
        m.events = None if events is None else int(events)
        m.events_defined = None if events_defined is None else int(events_defined)
        m.units = units
        m.mc = bool(mc)
        m.upper_limit = None if upper_limit is None else float(upper_limit)
        m.cl = None if cl is None else float(cl)
        m.interval = None if interval is None else (float(interval[0]), float(interval[1]))
        m.reason = reason
        return m

    def __reduce__(self):
        return (_rebuild, (float(self), self.as_dict()))

    def as_dict(self) -> dict:
        """The record of results.json (NaN written as None)."""
        v = float(self)
        d = {"value": v if math.isfinite(v) else None, "stat_error": self.stat_error, "error": self.error,
             "events": self.events, "events_defined": self.events_defined, "units": self.units, "mc": True,
             "beta": True}
        if self.interval is not None:
            d["interval"] = list(self.interval)
        if self.upper_limit is not None:
            d["upper_limit"] = self.upper_limit
            d["cl"] = self.cl
        if self.reason is not None:
            d["reason"] = self.reason
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Measured":
        v = d.get("value")
        return cls(float("nan") if v is None else v, stat_error=d.get("stat_error"), error=d.get("error"),
                   events=d.get("events"), events_defined=d.get("events_defined"), units=d.get("units", ""),
                   upper_limit=d.get("upper_limit"), cl=d.get("cl"), interval=d.get("interval"),
                   reason=d.get("reason"))

    def __str__(self) -> str:
        n = "" if self.events is None else f", {self.events} events"
        if self.upper_limit is not None:
            cl = f"{100 * (self.cl if self.cl is not None else LIMIT_CL):g} % CL"
            return f"< {_significant(self.upper_limit, 2)} ({cl}{n})"
        if self.stat_error is None:
            why = self.reason or "none was estimated"
            return f"{float(self):.6g} (no statistical error: {why}{n})"
        return f"{pdg_text(float(self), self.stat_error)}{' (' + n[2:] + ')' if n else ''}"

    def __repr__(self) -> str:
        return f"Measured({str(self)!r}, units={self.units!r})"


def _rebuild(value, d):
    return Measured.from_dict(dict(d, value=value))


def _significant(x: float, digits: int) -> str:
    """x with ``digits`` significant digits, trailing zeros kept, no exponent for ordinary magnitudes."""
    if x == 0 or not math.isfinite(x):
        return f"{x:g}"
    d = Decimal(repr(float(x)))
    exp = d.adjusted() - digits + 1
    q = d.quantize(Decimal(1).scaleb(exp), rounding=ROUND_HALF_UP)
    if q.adjusted() > d.adjusted():          # 9.96 -> 10.0: one digit more in front, one fewer behind
        q = q.quantize(Decimal(1).scaleb(exp + 1), rounding=ROUND_HALF_UP)
    if -5 <= q.adjusted() < 6:
        return format(q, "f")
    return format(q, "e")


def pdg_text(value: float, error: float) -> str:
    """``"<v> +- <e>"`` by the PDG rule: the three leading digits of the error decide its significant digits (100-354:
    two; 355-949: one; 950-999: rounded up to 1000, two); the value is rounded to the same decimal place. A value
    below 1e-3 or from 1e5 in magnitude is written ``"(<v> +- <e>)e<k>"`` with k its exponent."""
    e = abs(float(error))
    v = float(value)
    if e == 0.0 or not math.isfinite(e) or not math.isfinite(v):
        return f"{v:.6g} +- {e:.3g}"
    p = math.floor(math.log10(e))
    lead = int(round(e / 10.0 ** (p - 2)))
    if lead >= 1000:
        p += 1
        lead = int(round(e / 10.0 ** (p - 2)))
    if lead <= 354:
        digits = 2
    elif lead <= 949:
        digits = 1
    else:
        digits, p = 2, p + 1
    dec = digits - 1 - p                      # decimal places of the error's last significant digit
    dv, de = Decimal(repr(v)), Decimal(repr(e))
    k = 0
    if v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        k = dv.adjusted()
        dv, de = dv.scaleb(-k), de.scaleb(-k)
    places = dec + k
    q = Decimal(1).scaleb(-places)
    sv = format(dv.quantize(q, rounding=ROUND_HALF_UP), "f")
    se = format(de.quantize(q, rounding=ROUND_HALF_UP), "f")
    return f"({sv} +- {se})e{k}" if k else f"{sv} +- {se}"


# ============================================================================================ estimators
def _mean(values: np.ndarray, events: int, units: str) -> Measured:
    d = values[~np.isnan(values)]
    n = len(d)
    if n == 0:
        return Measured(float("nan"), error="sem", events=events, events_defined=0, units=units,
                        reason="no event where the variable is defined and the selection holds")
    m = float(np.mean(d))
    if n < 2:
        return Measured(m, error="sem", events=events, events_defined=n, units=units,
                        reason="fewer than two events")
    return Measured(m, stat_error=float(np.std(d, ddof=1)) / math.sqrt(n), error="sem", events=events,
                    events_defined=n, units=units)


def estimate_mean(values, units: str = "") -> Measured:
    """The mean of the defined values (NaN marks an undefined value and is skipped) with its standard error
    s / sqrt(n) (s with ddof 1, n the defined values); no error below two defined values."""
    x = np.asarray(values, dtype=float).ravel()
    return _mean(x, len(x), units)


def clopper_pearson(k: int, n: int, cl: float = ONE_SIGMA) -> tuple[float, float]:
    """The central Clopper-Pearson interval of a fraction, k of n, at confidence ``cl``."""
    from scipy import stats

    a = 1.0 - cl
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def estimate_fraction(k: int, n: int) -> Measured:
    """k of n events: the value k / n with its Clopper-Pearson interval at one sigma (``interval``) and half its width
    as ``stat_error``; after zero events no error but the exact one-sided 95 % limit 1 - 0.05^(1/n)."""
    k, n = int(k), int(n)
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"a fraction needs 0 <= k <= n and n > 0, not k = {k}, n = {n}")
    lo, hi = clopper_pearson(k, n)
    if k == 0:
        return Measured(0.0, error="upper_limit", events=n, events_defined=n, units="1", interval=(lo, hi),
                        upper_limit=1.0 - (1.0 - LIMIT_CL) ** (1.0 / n), cl=LIMIT_CL)
    return Measured(k / n, stat_error=(hi - lo) / 2, error="clopper_pearson", events=n, events_defined=n, units="1",
                    interval=(lo, hi))


def estimate_count(k: int, units: str = "1", events: int | None = None) -> Measured:
    """A count k with its Poisson error sqrt(k); after zero events no error but the exact 95 % limit -ln 0.05."""
    k = int(k)
    if k < 0:
        raise ValueError(f"a count is not negative ({k})")
    if k == 0:
        return Measured(0.0, error="upper_limit", events=events, events_defined=events, units=units,
                        upper_limit=-math.log(1.0 - LIMIT_CL), cl=LIMIT_CL)
    return Measured(float(k), stat_error=math.sqrt(k), error="poisson", events=events, events_defined=events,
                    units=units)


def batch_means(values, weights) -> tuple[float, float]:
    """vbar = sum w_b v_b and sigma^2 = B / (B - 1) sum w_b^2 (v_b - vbar)^2, with w_b the weights normalised to one
    (each batch weighted by its events)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    vbar = float(np.sum(w * v))
    B = len(v)
    return vbar, math.sqrt(B / (B - 1) * float(np.sum(w * w * (v - vbar) ** 2)))
