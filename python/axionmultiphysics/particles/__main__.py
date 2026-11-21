"""``python -m axionmultiphysics.particles {execute,merge,extend} WORKDIR``: runs a planned transport run.

Prints one JSON document on stdout: ``{"status": "complete" | "partial", "events", "events_requested",
"missing_chunks"}``. Exit status 0 when complete, 5 when partial, 1 when refused (``{"error", "code"}`` on stdout),
2 for a usage error.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import argparse
import json
import sys


def _positive(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m axionmultiphysics.particles",
                                description="Run, merge or extend a planned particle-transport work directory (beta).")
    p.add_argument("command", choices=("execute", "merge", "extend"))
    p.add_argument("workdir")
    p.add_argument("--workers", type=_positive, default=1, help="processes at once (execute, extend)")
    p.add_argument("--allow-failed-chunks", action="store_true", help="merge every complete chunk, not the prefix")
    p.add_argument("--events", type=_positive, help="the new number of events (extend)")
    a = p.parse_args(argv)
    if a.command == "extend" and a.events is None:
        p.error("extend needs --events N")
    if a.command != "extend" and a.events is not None:
        p.error(f"--events belongs to extend, not to {a.command}")
    from .. import _core
    from . import execute, extend, merge
    try:
        if a.command == "execute":
            execute(a.workdir, workers=a.workers)
            r = merge(a.workdir, allow_failed_chunks=a.allow_failed_chunks)
        elif a.command == "merge":
            r = merge(a.workdir, allow_failed_chunks=a.allow_failed_chunks)
        else:
            r = extend(a.workdir, events=a.events, workers=a.workers, allow_failed_chunks=a.allow_failed_chunks)
    except _core.AxionMultiPhysicsError as e:
        print(json.dumps({"error": str(e), "code": getattr(e, "code", None)}))
        return 1
    print(json.dumps({"status": "partial" if r.partial else "complete", "events": r.n_events,
                      "events_requested": r.events_requested, "missing_chunks": r.missing_chunks}))
    return 5 if r.partial else 0


if __name__ == "__main__":
    sys.exit(main())
