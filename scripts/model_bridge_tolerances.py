#!/usr/bin/env python3
"""Measure the exactness cases of the bridge tests and, with ``--record``, write the tolerance records of their case
files.

usage: python scripts/model_bridge_tolerances.py [--record] [--case NAME ...] [--cases DIR] [--mesher PATH]

Each case of ``cpp/tests/data/model/u2/cases`` is measured on this machine, in one process, by the function of
``tests/model_bridge_cases.py`` that its test uses. Without ``--record`` the values are printed beside the asserted
tolerances and nothing is written. With ``--record`` each quantity's ``measured`` gets this machine's value under
"<processor model> / <build>", and its ``asserted`` becomes ``max(10 x the largest measured value, 1e-12)``; ``recorded``
gets the values that no test asserts. A case whose asserted tolerance would exceed its ceiling is not written and the
script exits with status 1: the cause is investigated, a ceiling is never raised. Cases that need the fluids module
are skipped in a build without it.

Run it on every processor model the gate's sequential and MPI jobs can land on (one process each; ``OMP_NUM_THREADS=1``),
and commit the records in a commit of their own that says so. Nobody edits the numbers by hand.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import model_bridge_cases as C  # noqa: E402


def _has_fluids() -> bool:
    try:
        C.fluids()
    except ImportError:
        return False
    return True


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", action="store_true", help="write the measured values and the asserted tolerances")
    ap.add_argument("--case", action="append", help="a case name (repeatable); default: every case")
    ap.add_argument("--cases", type=Path, default=C.CASES, help="the directory of the case files")
    ap.add_argument("--mesher", help="the axmp-mesher program (else $AXMP_MESHER or the package's search)")
    a = ap.parse_args(argv)
    if a.mesher:
        os.environ["AXMP_MESHER"] = str(Path(a.mesher).resolve())
    names = a.case or list(C.CASE_NAMES)
    unknown = [n for n in names if n not in C.CASE_NAMES]
    if unknown:
        ap.error(f"unknown case(s) {unknown}; the cases are {list(C.CASE_NAMES)}")
    key = C.record_key()
    fluids = _has_fluids()
    print(f"machine: {C.environment_note()}")
    status = 0
    for name in names:
        path = C.case_path(name, a.cases)
        case = C.load(name, a.cases)
        if case["build"] == "mpi" and not fluids:
            print(f"{name}: skipped (needs the fluids module of an MPI build)")
            continue
        try:
            run = C.MEASURE[name](case)
        except Exception:  # noqa: BLE001 (reported, and the case is not recorded)
            print(f"{name}: FAILED before measuring\n{traceback.format_exc()}")
            status = 1
            continue
        ok = True
        for q, t in case["tolerance"].items():
            v = float(run["quantities"][q])
            line = f"{name}.{q} = {v:.3e} (ceiling {t['ceiling']:.0e}, asserted {t['asserted']})"
            if a.record:
                measured = dict(t["measured"], **{key: v})
                new = C.asserted_value(measured)
                if not new <= t["ceiling"]:
                    line += f"  ABOVE THE CEILING: asserted would be {new:.3e}"
                    ok = False
                else:
                    t["measured"], t["asserted"] = measured, new
                    line += f" -> asserted {new:.3e}"
            elif t["asserted"] is not None and not v <= t["asserted"]:
                line += "  EXCEEDS the asserted tolerance"
            print(line)
        for k, v in run["recorded"].items():
            print(f"{name}.{k} (recorded) = {v}")
            if a.record:
                case.setdefault("recorded", {}).setdefault(k, {})[key] = _clean(v)
        if a.record:
            if ok:
                C.save(case, path)
                print(f"{name}: recorded under '{key}' in {path}")
            else:
                print(f"{name}: NOT recorded (a tolerance above its ceiling)")
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
