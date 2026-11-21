"""Support for the tests of the commands and Python objects of bounded 2D meshes (``test_model_boundaries.py``,
``test_model_mesh_cli.py``, ``test_model_requires_mesher.py``): the documents, the command line, the mesher and the
stub meshers that stand for a wheel without gmsh. Not a test file: pytest collects nothing here.

The documents are the model documents of ``cpp/tests/data/model`` (``channel.json``, ``busbar.json``) and
``cpp/tests/data/model/u2`` (the fixtures of bounded meshes; never under ``examples/``, ``benchmarks/designs/`` or ``tests/g0/designs``,
which G0 scans). A test that needs the mesher takes the fixture ``mesher``: it skips when no mesher loads (it does not
pass vacuously), and it puts the mesher's path in ``AXMP_MESHER`` for the command line and for the in-process core, so
that both find the same program.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cpp" / "tests" / "data" / "model"
FIXTURES = DATA / "u2"

_spec = importlib.util.spec_from_file_location("g0_harness_model_mesh", ROOT / "tests" / "g0" / "g0.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)

# The validation report's keys (tests/test_model_cli.py) and the keys `axmp boundaries --json` adds to it.
REPORT_KEYS = {"ok", "name", "parameters", "errors", "unavailable", "deferred", "warnings"}
TABLE_KEYS = {"symmetry", "boundaries", "edges"}
FIRST_KEYS = {"error", "where", "hint", "code"}

# The loader's lines of a mesher that cannot start: its gmsh library is missing (a wheel installed without the mesh
# extra), or a library that gmsh's Linux wheel needs is missing (no X11/OpenGL libraries on a headless machine).
LOADER_GMSH = ("axmp-mesher: error while loading shared libraries: libgmsh.so.4.15: cannot open shared object file: "
               "No such file or directory")
LOADER_GLU = ("axmp-mesher: error while loading shared libraries: libGLU.so.1: cannot open shared object file: "
              "No such file or directory")
# docs/guides/install.md: the X11/OpenGL libraries gmsh's Linux wheel needs (Debian/Ubuntu package names)
GL_LIBRARIES = "libglu1-mesa libgl1 libxcursor1 libxft2 libxinerama1"


def doc(name: str) -> dict:
    """A model document by name: ``cpp/tests/data/model/u2/<name>.json``, else ``cpp/tests/data/model/<name>.json``."""
    p = FIXTURES / f"{name}.json"
    if not p.is_file():
        p = DATA / f"{name}.json"
    return json.loads(p.read_text("utf-8"))


def em_minimal() -> dict:
    """A valid electromagnetic design: one rectangle (tests/test_model_guard.py)."""
    return {"name": "m", "regions": [{"name": "x", "shape": {"rect": {"r": [0, 1], "z": [0, 1]}}}]}


def write(directory: Path, document: dict, name: str = "doc.json") -> Path:
    p = Path(directory) / name
    p.write_text(json.dumps(document, indent=1), "utf-8")
    return p


def axmp_exe() -> str:
    exe = g0.find_axmp()
    if exe is None:
        pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")
    return str(exe)


def run(*args, env: dict | None = None, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([axmp_exe(), *map(str, args)], capture_output=True, text=True, timeout=timeout,
                          env=env, cwd=cwd)


def run_json(*args, **kw) -> tuple[int, dict]:
    """The exit status and the JSON that the command prints on stdout."""
    r = run(*args, **kw)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"stdout is not JSON ({e}); status {r.returncode}\nstdout: {r.stdout}\nstderr: {r.stderr}")
    return r.returncode, out


def environment(**overrides: str) -> dict:
    env = dict(os.environ)
    env.update(overrides)
    return env


def mesher_path() -> str | None:
    """The mesher the core finds first: ``$AXMP_MESHER``, the package's search, the build directories."""
    return g0.mesher_exe()


def mesher_loads(path: str | None) -> bool:
    """The probe of the mesher, done here independently: run with no argument, it prints its usage and exits with 2."""
    if not path or not os.access(path, os.X_OK):
        return False
    try:
        return subprocess.run([path], capture_output=True, timeout=60).returncode == 2
    except OSError:
        return False


@pytest.fixture
def mesher(monkeypatch) -> str:
    """The path of a mesher that loads, also put in ``AXMP_MESHER``; the test skips without one."""
    path = mesher_path()
    if not mesher_loads(path):
        pytest.skip("no axmp-mesher that loads (set AXMP_MESHER, or build the axmp-mesher target)")
    path = str(Path(path).resolve())
    monkeypatch.setenv("AXMP_MESHER", path)
    return path


def stub(directory: Path, name: str, *, status: int, stderr: str = "", log: Path | None = None) -> Path:
    """A stand-in for the mesher: a shell script that appends its arguments to ``log`` (one line per run; the probe's
    line is empty after the name), prints ``stderr`` on standard error and exits with ``status``."""
    p = Path(directory) / name
    lines = ["#!/bin/sh"]
    if log is not None:
        lines.append(f"echo \"run $*\" >> '{log}'")
    if stderr:
        lines.append(f"echo '{stderr}' >&2")
    lines.append(f"exit {status}")
    p.write_text("\n".join(lines) + "\n")
    p.chmod(0o755)
    return p


def stub_runs(log: Path) -> list[str]:
    """The arguments of each run of a stub ("" for a run without arguments, the probe)."""
    if not log.exists():
        return []
    return [ln[len("run"):].strip() for ln in log.read_text().splitlines()]


def line_elements(msh: Path) -> int:
    """The number of line elements (MSH 2.2 types 1 and 8: two- and three-node lines) of an ASCII MSH 2.2 file."""
    lines = Path(msh).read_text().splitlines()
    i = lines.index("$Elements")
    n = int(lines[i + 1])
    return sum(1 for ln in lines[i + 2:i + 2 + n] if ln.split()[1] in ("1", "8"))


def element_types(msh: Path) -> set[int]:
    lines = Path(msh).read_text().splitlines()
    i = lines.index("$Elements")
    n = int(lines[i + 1])
    return {int(ln.split()[1]) for ln in lines[i + 2:i + 2 + n]}


# Keys that would carry a master number (an atom's attribute, 900 and 2001, ...) or a geometry kernel's entity tag.
NUMBER_KEYS = {"attributes", "attribute", "attr", "atom", "atoms", "master", "master_attribute", "tag", "tags",
               "boundary_groups"}


def keys_anywhere(obj, keys: set[str], path: str = "") -> list[str]:
    """The JSON pointers of every key of ``keys`` at any depth of ``obj``."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                found.append(f"{path}/{k}")
            found += keys_anywhere(v, keys, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += keys_anywhere(v, keys, f"{path}/{i}")
    return found


def assert_finding(f: dict, *, code: str, where: str, message: str | None = None, hint: str | None = None,
                   details: dict | None = None) -> None:
    """A finding of a report (the command line's error JSON without ``ok``), field by field."""
    assert f["code"] == code, f"code {f['code']!r} (want {code!r}): {f}"
    assert f["where"] == where, f"where {f['where']!r} (want {where!r}): {f}"
    if message is not None:
        assert f["error"] == message, f"\n got  {f['error']}\n want {message}"
    if hint is not None:
        assert f["hint"] == hint, f"\n got  {f['hint']}\n want {hint}"
    if details is not None:
        assert f.get("details") == details, f"\n got  {f.get('details')}\n want {details}"


def assert_exception_is(e: BaseException, f: dict) -> None:
    """The exception of the Python API carries the finding ``f`` of the command line: class, ``str`` and the fields."""
    from axionmultiphysics import _core

    assert type(e) is _core.AxionMultiPhysicsError, f"{type(e).__name__}: {e}"
    composed = f["error"] + (f" [at {f['where']}]" if f["where"] else "") + (f" Hint: {f['hint']}" if f["hint"] else "")
    assert str(e) == composed, f"\n got  {e}\n want {composed}"
    assert (e.code, e.where, e.hint) == (f.get("code", ""), f["where"], f["hint"])
    assert e.details == f.get("details"), f"details {e.details!r}"


def raises(f, *a, **kw) -> BaseException:
    try:
        f(*a, **kw)
    except BaseException as e:  # noqa: BLE001 (the tests inspect what was raised)
        return e
    raise AssertionError(f"{getattr(f, '__name__', f)} did not raise")

