"""Console script ``axmp``: runs the C++ command line shipped in the wheel.

``cpp/apps/axmp.cpp`` is installed as ``axionmultiphysics/bin/axmp``.  This wrapper puts it
on ``PATH`` through the entry point in pyproject.toml and replaces the Python
process with it, so exit codes, signals and streams behave exactly as they do
when the binary is called directly.

``axmp export-comsol ...`` is handled in Python (:mod:`axionmultiphysics.comsol`, docs/comsol_export.md)
before the C++ command line is looked for, and so is ``axmp import-comsol ...``
(:mod:`axionmultiphysics.comsol_import`, docs/comsol_import.md), and ``axmp export-touchstone results.json ...``
(:mod:`axionmultiphysics.touchstone`, docs/physics/touchstone.md).

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import os
import sys

from . import _paths


def axmp_path() -> str | None:
    """The ``axmp`` executable behind the console script: $AXMP_BIN, else the search of :func:`_paths.axmp_path`
    ($AXMP_EXE, the installed package, a development build)."""
    return os.environ.get("AXMP_BIN") or _paths.axmp_path()


def main(argv: list[str] | None = None) -> int:
    args_in = list(argv) if argv is not None else sys.argv[1:]
    if args_in and args_in[0] == "export-comsol":
        from .comsol import main as export_main

        return export_main(args_in[1:])
    if args_in and args_in[0] == "import-comsol":
        from .comsol_import import main as import_main

        return import_main(args_in[1:])
    if args_in and args_in[0] == "export-touchstone":
        from .touchstone import main as touchstone_main

        return touchstone_main(args_in[1:])
    exe = axmp_path()
    if exe is None:
        sys.stderr.write(
            "axmp: the compiled command line was not found.\n"
            "  Expected axionmultiphysics/bin/axmp inside the installed package, or build/axmp in a\n"
            "  source checkout. Set $AXMP_BIN to its path, or reinstall axionMultiPhysics\n"
            "  (see docs/install.md).\n"
        )
        return 127
    _paths.prepare_environment()  # loader path for the bundled libraries and gmsh
    args = [exe] + args_in
    os.execv(exe, args)  # does not return
