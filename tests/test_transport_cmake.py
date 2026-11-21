"""How CMakeLists.txt builds axmp-transport: off by default, off in wheels, and only after an explicit Geant4 check.

The check itself (cmake/AxmpGeant4Check.cmake, ``axmp_check_geant4()``) is tested with mocked versions by the ctest
cases ``cmake_geant4_version_*`` and against find_package's version-range pitfall by ``cmake_geant4_range_*``
(cmake/tests/). Those prove the function right; this file proves it is called where it must be. A configure that
reaches the transport branch needs a Geant4 install, which cannot be assumed in every build, so CMakeLists.txt is
read (comment lines removed) and its commands are located with their enclosing ``if`` blocks:

1. ``option(AXMP_BUILD_TRANSPORT "..." OFF)``, once;
2. under ``if(SKBUILD)`` (a wheel build), ``set(AXMP_BUILD_TRANSPORT OFF CACHE BOOL ... FORCE)``, before the transport
   branch;
3. every ``find_package(Geant4 ...)`` inside ``if(AXMP_BUILD_TRANSPORT)``, so that a default configure never looks
   for Geant4; there, ``find_package(Geant4 ... CONFIG REQUIRED)``, then exactly one ``axmp_check_geant4()`` in the
   whole file, then ``add_executable(axmp-transport ...)``; the module is included before the call;
4. ``target_link_libraries(axmp-transport ...)`` names only Geant4 (``${Geant4_LIBRARIES}`` or ``Geant4::`` targets);
5. pyproject.toml does not switch the transport on.

Pure Python; runs in every build.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMAKELISTS = ROOT / "CMakeLists.txt"


def commands(text: str) -> list[dict]:
    """Every command invocation of a CMake file: name, argument text, line, and the stack of enclosing if-conditions
    (with whether the command sits in that if's else/elseif part)."""
    lines = [("" if re.match(r"^\s*#", l) else l) for l in text.splitlines()]
    src = "\n".join(lines)
    out, stack, i = [], [], 0
    pat = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(", re.M)
    while True:
        m = pat.search(src, i)
        if not m:
            break
        depth, j, quoted = 1, m.end(), False
        while depth and j < len(src):
            ch = src[j]
            if ch == '"' and src[j - 1] != "\\":
                quoted = not quoted
            elif not quoted and ch == "(":
                depth += 1
            elif not quoted and ch == ")":
                depth -= 1
            j += 1
        name, args = m.group(1).lower(), src[m.end():j - 1]
        line = src.count("\n", 0, m.start()) + 1
        if name == "if":
            stack.append({"cond": args.strip(), "line": line, "else": False})
        elif name in ("elseif", "else") and stack:
            stack[-1]["else"] = True
        elif name == "endif" and stack:
            stack.pop()
        out.append({"name": name, "args": " ".join(args.split()), "line": line, "stack": [dict(s) for s in stack]})
        i = j
    return out


def in_if(cmd: dict, cond_re: str) -> bool:
    """The command lies in the true branch of an enclosing if whose condition matches."""
    return any(re.fullmatch(cond_re, s["cond"]) and not s["else"] for s in cmd["stack"])


CMDS = commands(CMAKELISTS.read_text("utf-8"))
TRANSPORT_IF = r"AXMP_BUILD_TRANSPORT"


def _find(name: str, args_re: str = ".*") -> list[dict]:
    return [c for c in CMDS if c["name"] == name and re.search(args_re, c["args"])]


def test_the_transport_option_exists_and_is_off_by_default():
    opts = _find("option", r"^AXMP_BUILD_TRANSPORT\b")
    assert len(opts) == 1, f"option(AXMP_BUILD_TRANSPORT ...) appears {len(opts)} times in CMakeLists.txt"
    assert re.fullmatch(r'AXMP_BUILD_TRANSPORT "[^"]*" OFF', opts[0]["args"]), opts[0]["args"]


def test_a_wheel_build_forces_the_transport_off():
    branch = [c for c in _find("if", rf"^{TRANSPORT_IF}$")]
    assert branch, "no if(AXMP_BUILD_TRANSPORT) branch in CMakeLists.txt"
    forced = [c for c in _find("set", r"^AXMP_BUILD_TRANSPORT OFF CACHE BOOL .*\bFORCE\b")
              if in_if(c, r".*\bSKBUILD\b.*") and c["line"] < branch[0]["line"]]
    assert forced, ("CMakeLists.txt does not force AXMP_BUILD_TRANSPORT off under if(SKBUILD) before the transport "
                    "branch: a wheel must never contain Geant4")
    opt = _find("option", r"^AXMP_BUILD_TRANSPORT\b")[0]
    assert forced[0]["line"] > opt["line"], "the forced value comes before the option() it overrides"


def test_geant4_is_looked_for_only_in_the_transport_branch_and_checked_before_the_program():
    finds = _find("find_package", r"^Geant4\b")
    assert finds, "no find_package(Geant4 ...) in CMakeLists.txt"
    outside = [c["line"] for c in finds if not in_if(c, TRANSPORT_IF)]
    assert not outside, f"find_package(Geant4) outside if(AXMP_BUILD_TRANSPORT), lines {outside}"
    required = [c for c in finds if re.search(r"\bCONFIG\b", c["args"]) and re.search(r"\bREQUIRED\b", c["args"])]
    assert required, f"find_package(Geant4 ... CONFIG REQUIRED) missing: {[c['args'] for c in finds]}"
    calls = _find("axmp_check_geant4")
    assert len(calls) == 1 and calls[0]["args"] == "", (
        f"expected exactly one bare axmp_check_geant4() call, found {[(c['line'], c['args']) for c in calls]}")
    call = calls[0]
    assert in_if(call, TRANSPORT_IF), "axmp_check_geant4() is not inside if(AXMP_BUILD_TRANSPORT)"
    assert required[0]["line"] < call["line"], "axmp_check_geant4() is called before find_package(Geant4)"
    exe = _find("add_executable", r"^axmp-transport\b")
    assert len(exe) == 1 and in_if(exe[0], TRANSPORT_IF), "add_executable(axmp-transport ...) not in its branch, once"
    assert call["line"] < exe[0]["line"], "axmp-transport is defined before the Geant4 version is checked"
    inc = [c for c in _find("include", r"AxmpGeant4Check") if c["line"] < call["line"]]
    assert inc, "cmake/AxmpGeant4Check.cmake is not included before the call"


def test_the_transport_links_only_geant4():
    links = _find("target_link_libraries", r"^axmp-transport\b")
    assert links, "no target_link_libraries(axmp-transport ...)"
    for c in links:
        items = c["args"].split()[1:]
        other = [t for t in items if t not in ("PRIVATE", "PUBLIC", "INTERFACE", "${Geant4_LIBRARIES}")
                 and not t.startswith("Geant4::")]
        assert not other, f"axmp-transport links more than Geant4 (line {c['line']}): {other}"


def test_pyproject_does_not_switch_the_transport_on():
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    assert not re.search(r"AXMP_BUILD_TRANSPORT\W+(ON|TRUE|1)\b", text, re.I), "pyproject.toml enables the transport"
