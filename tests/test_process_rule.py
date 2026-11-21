"""The process rule for particle transport and event generation, as built artefacts and as text.

The rule (AGENTS.md "Dependencies"): ``libaxionmultiphysics``, ``_core`` and ``axmp`` link only permissively licensed
code. Geant4, whose builds always contain LGPL-3 CLHEP, is used only by the separate program ``axmp-transport``, which
links Geant4 and the header-only nlohmann/json and nothing of ours. Pythia (GPL) is used only by the generator file,
which runs as its own process and imports nothing of the package. The package never imports ``pythia8mc``,
``pythia8`` or a Geant4 binding in its own process.

What is checked, and where it is meaningful:

- the linkage of ``_core``, ``libaxionmultiphysics`` and ``axmp`` (every build): the direct ``NEEDED`` entries, the
  whole set the loader resolves (``ldd``), and the symbols the files define (a static copy of a forbidden library
  leaves no ``NEEDED`` entry but leaves its symbols). The built files are inspected, not the CMake target
  properties: what the loader maps into the process is the thing the rule is about, whichever way CMake got there
  (a ``target_link_libraries``, an imported target's interface, a link option or a static archive);
- the linkage of ``axmp-transport`` (only where it is built: ``$AXMP_TRANSPORT``): its direct entries are Geant4's
  libraries, the C and C++ runtimes and Geant4's own external dependencies, and nothing it loads is ours;
- the Python process: after every public module of the package is imported, none of the forbidden modules was
  imported (an import hook records every attempt, so an attempt inside ``try``/``except ImportError`` is caught
  even where the module is not installed), and no source file outside the separate programs names one in an
  import statement or an ``import_module`` / ``__import__`` call;
- the generator file and the driver's command line for it (when the file exists);
- the documentation that carries the rule: AGENTS.md, the licensing table of docs/install.md, the transport's guide
  page with Geant4's acknowledgement, and the pinned checksums.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "python" / "axionmultiphysics"
PROGRAMS = PKG / "_programs"
GENERATOR = PROGRAMS / "axmp_generate.py"
LINUX = sys.platform.startswith("linux")
MAC = sys.platform == "darwin"

# Python modules the package's own process must never import
FORBIDDEN_MODULES = ("pythia8mc", "pythia8", "geant4_pybind", "HepMC3")

# libraries that must not reach _core, libaxionmultiphysics or axmp (matched on the file name, any directory)
FORBIDDEN_LIBS = {
    "Geant4": re.compile(r"^libG4|geant4", re.I),
    "CLHEP": re.compile(r"clhep", re.I),
    "Pythia 8": re.compile(r"pythia8", re.I),
    "FastJet (in the Pythia wheel)": re.compile(r"fastjet|fjcore", re.I),
    "HepMC3": re.compile(r"hepmc", re.I),
    "ROOT": re.compile(r"^lib(Core|RIO|Hist|Tree|Cling|MathCore|Imt|Thread|Net|Graf|Gpad|Matrix|Physics|ROOTDataFrame|"
                       r"ROOTVecOps)\.(so|dylib)"),
}
# symbols (demangled) that a static copy of one of those libraries defines
FORBIDDEN_SYMBOLS = re.compile(
    r"(?:^|[\s(,<*&:])(?:CLHEP|Pythia8|HepMC3|fastjet)::"   # their namespaces
    r"|(?:^|[\s(,<*&])G4[A-Z][A-Za-z0-9_]*::"              # Geant4's classes (G4RunManager::..., G4Track::...)
    r"|(?:typeinfo|vtable) for G4[A-Z]"
    r"|(?:^|[\s(,<*&])(?:TObject|TROOT)::")

# axmp-transport: the direct entries allowed (Geant4, the runtimes, Geant4's own external dependencies) ...
TRANSPORT_NEEDED_ALLOWED = [
    re.compile(p) for p in (
        r"^libG4[A-Za-z0-9_]*\.(so|dylib)",                       # Geant4, its CLHEP (G4clhep) and PTL (G4ptl)
        r"^libCLHEP[-A-Za-z0-9_.]*\.(so|dylib)",                  # a Geant4 built against an external CLHEP
        r"^(libstdc\+\+|libc\+\+|libc|libm|libgcc_s|libpthread|libdl|librt|libatomic|libutil)\.(so|dylib)",
        r"^ld-linux", r"^linux-vdso", r"^libSystem\.",
        r"^lib(expat|z|xerces-c|tbb|tbbmalloc)[-.0-9]*\.(so|dylib)",  # Geant4's external dependencies
        r"^lib(X11|Xext|Xmu|Xt|Xi|GL|GLU|OpenGL|Xm|freetype|fontconfig)\.(so|dylib)",  # Geant4's optional drivers
        r"^libQt\d?[A-Za-z]*\.(so|dylib)",
    )]
# ... and what must never be loaded into it: everything of ours and the field solvers' stack
TRANSPORT_FORBIDDEN_LIBS = re.compile(
    r"axionmultiphysics|^libmfem|^libHYPRE|^lib(par)?metis|^libmpi|^libopen-(pal|rte)|^libstrumpack|^libgmsh|"
    r"^libpython|^libpetsc|^libslepc|^libsuperlu|^lib(umfpack|klu|cholmod|suitesparse)|pythia8|fastjet|hepmc", re.I)
TRANSPORT_FORBIDDEN_SYMBOLS = re.compile(r"(?:^|[\s(,<*&])(?:axmp|mfem|nanobind)::")


# ----------------------------------------------------------------------------------------------- inspecting files
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def needed_entries(path: Path) -> list[str]:
    """The direct dependencies recorded in the file (ELF NEEDED, Mach-O load commands), as file names."""
    if LINUX:
        if shutil.which("readelf"):
            r = _run(["readelf", "-d", "--wide", str(path)])
            assert r.returncode == 0, f"readelf -d {path} failed: {r.stderr}"
            return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", r.stdout)
        if shutil.which("objdump"):
            r = _run(["objdump", "-p", str(path)])
            assert r.returncode == 0, f"objdump -p {path} failed: {r.stderr}"
            return re.findall(r"^\s*NEEDED\s+(\S+)", r.stdout, re.M)
        pytest.skip("neither readelf nor objdump is available to read the NEEDED entries")
    if MAC and shutil.which("otool"):
        r = _run(["otool", "-L", str(path)])
        assert r.returncode == 0, f"otool -L {path} failed: {r.stderr}"
        return [Path(m).name for m in re.findall(r"^\s+(\S+) \(compatibility", r.stdout, re.M)]
    pytest.skip(f"no tool to read the dependencies of a binary on {sys.platform}")


def resolved_libraries(path: Path) -> list[str] | None:
    """Every library the loader maps for the file, direct or not (Linux ldd), as file names of the requested name and
    of the resolved path; None where ldd is not available."""
    if not (LINUX and shutil.which("ldd")):
        return None
    r = _run(["ldd", str(path)])
    names = []
    for line in r.stdout.splitlines():
        m = re.match(r"\s*(\S+)(?:\s+=>\s+(\S+))?", line)
        if not m:
            continue
        names.append(Path(m.group(1)).name)
        if m.group(2) and m.group(2) not in ("not",):
            names.append(Path(m.group(2)).name)
    return names


def defined_symbols(path: Path) -> str | None:
    """The demangled names of the symbols the file defines (symbol table and dynamic table); None without nm."""
    if not shutil.which("nm"):
        return None
    variants = (["nm", "-C", "--defined-only"], ["nm", "-C", "-D", "--defined-only"]) if LINUX else \
        (["nm", "-C", "-g", "--defined-only"],)
    out = []
    for v in variants:
        r = _run(v + [str(path)])
        if r.returncode == 0:
            out.append(r.stdout)
    return "\n".join(out) if out else None


def forbidden_linkage(path: Path) -> list[str]:
    """Every way the file reaches a forbidden library: a NEEDED entry, a library the loader maps, a defined symbol."""
    found = []
    for name in needed_entries(path):
        for what, pat in FORBIDDEN_LIBS.items():
            if pat.search(name):
                found.append(f"NEEDED {name} ({what})")
    for name in resolved_libraries(path) or []:
        for what, pat in FORBIDDEN_LIBS.items():
            if pat.search(name):
                found.append(f"loads {name} ({what})")
    syms = defined_symbols(path)
    if syms:
        hits = [s for s in syms.splitlines() if FORBIDDEN_SYMBOLS.search(s)]
        if hits:
            found.append(f"defines {len(hits)} symbols of a forbidden library, e.g. {hits[:3]}")
    return found


# ----------------------------------------------------------------------------------------------- the built files
def _core_file() -> Path:
    try:
        from axionmultiphysics import _core
    except ImportError as e:
        pytest.skip(f"the compiled core does not import ({e})")
    return Path(_core.__file__).resolve()


def _axmp_file() -> Path:
    for var in ("AXMP_BIN", "AXMP_EXE"):
        if os.environ.get(var):
            return Path(os.environ[var]).resolve()
    try:
        from axionmultiphysics import _paths
        exe = _paths.axmp_path()
    except ImportError:
        exe = None
    if exe:
        return Path(exe).resolve()
    for d in ("build-mpi", "build-seq", "build"):
        if (ROOT / d / "axmp").is_file():
            return (ROOT / d / "axmp").resolve()
    pytest.skip("the axmp executable was not found (AXMP_BIN, AXMP_EXE or a build directory)")


def _library_file(core: Path, axmp: Path) -> Path:
    """libaxionmultiphysics as the loader resolves it for _core; else next to _core (wheel) or to axmp (build tree)."""
    if LINUX and shutil.which("ldd"):
        m = re.search(r"(libaxionmultiphysics[^\s]*)\s+=>\s+(/\S+)", _run(["ldd", str(core)]).stdout)
        if m:
            return Path(m.group(2)).resolve()
    for d in (core.parent, axmp.parent):
        hits = sorted(d.glob("libaxionmultiphysics.so*")) + sorted(d.glob("libaxionmultiphysics*.dylib"))
        if hits:
            return hits[0].resolve()
    pytest.fail(f"libaxionmultiphysics was not found: not resolved by the loader for {core}, and not beside it or "
                f"beside {axmp}")


@pytest.fixture(scope="module")
def built_files() -> dict[str, Path]:
    core = _core_file()
    axmp = _axmp_file()
    return {"_core": core, "axmp": axmp, "libaxionmultiphysics": _library_file(core, axmp)}


@pytest.mark.parametrize("which", ["_core", "libaxionmultiphysics", "axmp"])
def test_no_geant4_clhep_pythia_hepmc3_or_root_in_the_library_core_and_cli(built_files, which):
    path = built_files[which]
    assert path.is_file(), path
    found = forbidden_linkage(path)
    assert not found, (f"{which} ({path}) reaches code the process rule keeps out of it: {found}. Geant4 is linked "
                       f"only by axmp-transport and Pythia only by the generator process (AGENTS.md, Dependencies)")


def test_the_library_core_and_cli_do_link_what_the_check_can_see(built_files):
    """The inspection is not vacuous: the library's own dependency is visible in _core's entries or symbols."""
    core = built_files["_core"]
    needed = needed_entries(core)
    syms = defined_symbols(core) or ""
    assert any("axionmultiphysics" in n for n in needed) or "axmp::" in syms, (
        f"neither a NEEDED entry of libaxionmultiphysics nor an axmp:: symbol in {core}: the inspection reads "
        f"nothing ({needed[:10]})")


# ----------------------------------------------------------------------------------------------- axmp-transport
def _transport_file() -> Path:
    exe = os.environ.get("AXMP_TRANSPORT")
    if not exe:
        pytest.skip("axmp-transport is not built here: set AXMP_TRANSPORT to it (cmake -DAXMP_BUILD_TRANSPORT=ON)")
    p = Path(exe)
    if not (p.is_file() and os.access(p, os.X_OK)):
        pytest.fail(f"AXMP_TRANSPORT={exe} is not an executable file")
    return p.resolve()


@pytest.mark.px_transport
def test_transport_links_only_geant4_and_the_runtimes():
    exe = _transport_file()
    needed = needed_entries(exe)
    other = [n for n in needed if not any(p.search(n) for p in TRANSPORT_NEEDED_ALLOWED)]
    assert not other, (f"axmp-transport has direct dependencies other than Geant4, the C/C++ runtimes and Geant4's "
                       f"own external libraries: {other} (all: {needed})")
    loaded = resolved_libraries(exe) or needed
    ours = sorted({n for n in loaded if TRANSPORT_FORBIDDEN_LIBS.search(n)})
    assert not ours, f"axmp-transport loads code of ours or of the field solvers' stack: {ours}"
    syms = defined_symbols(exe) or ""
    mine = [s for s in syms.splitlines() if TRANSPORT_FORBIDDEN_SYMBOLS.search(s)]
    assert not mine, f"axmp-transport defines symbols of the library or MFEM (a static copy): {mine[:5]}"
    has_geant4 = any(n.startswith("libG4") for n in loaded) or re.search(r"G4[A-Z]\w*::", syms)
    assert has_geant4, f"axmp-transport shows no Geant4 at all (NEEDED {needed}); the inspection reads nothing"


# ----------------------------------------------------------------------------------------------- the Python process
_IMPORT_PROBE = r"""
import importlib, importlib.abc, importlib.machinery, json, pkgutil, sys, traceback, warnings
FORBIDDEN = set(json.loads(sys.argv[1]))
attempts = []

class Tripwire(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # answers for the forbidden names before any real finder: the import then succeeds with an empty module (so code
    # that would carry on after it does), and every execution is recorded, installed or not
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in FORBIDDEN:
            return importlib.machinery.ModuleSpec(name, self)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        attempts.append(module.__name__)

sys.meta_path.insert(0, Tripwire())
warnings.simplefilter("ignore")
import axionmultiphysics
imported, failed = ["axionmultiphysics"], {}
def onerror(name):
    failed[name] = traceback.format_exc(limit=1).strip().splitlines()[-1]
for info in pkgutil.walk_packages(axionmultiphysics.__path__, "axionmultiphysics.", onerror=onerror):
    parts = info.name.split(".")[1:]
    if any(p.startswith("_") for p in parts):
        continue  # private modules: the separate programs and the workers are not imported into the user's process
    try:
        importlib.import_module(info.name)
        imported.append(info.name)
    except BaseException as e:  # an optional dependency, or a module of another build (fluids needs MPI)
        failed[info.name] = f"{type(e).__name__}: {e}"
present = sorted(m for m in sys.modules if m.partition(".")[0] in FORBIDDEN)
print(json.dumps({"attempts": attempts, "present": present, "imported": imported, "failed": failed}))
"""


def test_importing_every_public_module_imports_no_generator_or_geant4_binding(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    r = subprocess.run([sys.executable, "-c", _IMPORT_PROBE, json.dumps(FORBIDDEN_MODULES)], capture_output=True,
                       text=True, timeout=600, env=env, cwd=tmp_path)
    if r.returncode != 0 and "No module named 'axionmultiphysics" in r.stderr:
        pytest.skip("the package does not import here")
    assert r.returncode == 0, f"the import probe failed:\n{r.stderr[-3000:]}"
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert not res["attempts"], (f"importing the package's public modules imported {res['attempts']}: the package "
                                 f"never imports {', '.join(FORBIDDEN_MODULES)} in its own process (AGENTS.md)")
    assert not res["present"], f"sys.modules holds {res['present']} after importing the package"
    # the probe really walked the package (and did not stop at the first module)
    for must in ("axionmultiphysics.design", "axionmultiphysics.results", "axionmultiphysics.circuit"):
        assert must in res["imported"], (must, res["failed"].get(must))
    assert len(res["imported"]) >= 20, (len(res["imported"]), res["failed"])


def _imported_names(tree: ast.AST) -> list[tuple[int, str, int]]:
    """(line, module name, relative level) of every import statement and every import_module/__import__ call with a
    literal name."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name, 0) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append((node.lineno, node.module or "", node.level))
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                out.append((node.lineno, node.args[0].value, 0))
    return out


def test_no_source_file_of_the_package_imports_a_forbidden_module():
    """Static: catches an import inside a function that the import probe never calls."""
    bad = []
    for f in sorted(PKG.rglob("*.py")):
        if PROGRAMS in f.parents:
            continue  # the separate programs: checked below
        for line, name, level in _imported_names(ast.parse(f.read_text("utf-8"), filename=str(f))):
            if level == 0 and name.partition(".")[0] in FORBIDDEN_MODULES:
                bad.append(f"{f.relative_to(ROOT)}:{line}: {name}")
    assert not bad, f"the package imports a module only the generator process may import: {bad}"


# ----------------------------------------------------------------------------------------------- the generator
def _generator_present():
    if not GENERATOR.is_file():
        pytest.skip("the generator file is not in this tree yet (python/axionmultiphysics/_programs/axmp_generate.py)")


def test_the_separate_programs_import_nothing_of_the_package():
    _generator_present()
    bad = []
    for f in sorted(PROGRAMS.rglob("*.py")):
        for line, name, level in _imported_names(ast.parse(f.read_text("utf-8"), filename=str(f))):
            if level > 0 or name.partition(".")[0] == "axionmultiphysics":
                bad.append(f"{f.relative_to(ROOT)}:{line}: {'.' * level}{name}")
    assert not bad, f"a separate program imports the package (it would load _core into the GPL process): {bad}"


def test_the_driver_runs_the_generator_isolated_and_by_path(tmp_path):
    _generator_present()
    from axionmultiphysics import particles  # the driver must exist once the generator does

    job = tmp_path / "job.json"
    argv = particles.generator_command(str(job))
    assert isinstance(argv, list) and all(isinstance(a, str) for a in argv), argv
    assert "-m" not in argv, f"the generator is run as a module (-m imports the package): {argv}"
    assert "-I" in argv, f"the generator is not run isolated (-I): {argv}"
    files = [i for i, a in enumerate(argv) if Path(a).name == GENERATOR.name]
    assert files, f"the generator file is not on the command line: {argv}"
    assert Path(argv[files[0]]).resolve().is_file(), argv[files[0]]
    assert argv.index("-I") < files[0] < len(argv) - 1 and argv[-1] == str(job), argv


# ----------------------------------------------------------------------------------------------- the rule as text
RULE = ("`libaxionmultiphysics`, `_core` and `axmp` link only permissively licensed code. Geant4, whose builds always "
        "contain LGPL-3 CLHEP, is used only by `axmp-transport`. Pythia (GPL) is used only by the generator process, "
        "which links and imports nothing of axionMultiPhysics. Geant4 and Pythia never run in one process. All of "
        "them exchange files. The package never imports `pythia8mc`, `pythia8` or a Geant4 binding in its own "
        "process. The wheels contain no Geant4, CLHEP or Pythia code. No GPL or LGPL code is copied into the "
        "repository.")
G4_NOTICE = re.compile(r"This product includes software developed by Members of the Geant4 Collaboration "
                       r"\(\s*http://cern\.ch/geant4\s*\)\.")
TRANSPORT_GUIDE = ROOT / "docs" / "guides" / "particles.md"


def _plain(text: str) -> str:
    """Markdown reduced to words: no block-quote or list markers, no backticks or emphasis, one space between words."""
    text = re.sub(r"^\s*(>\s*)+", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)
    text = text.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def _section(path: Path, heading: str) -> str:
    text = path.read_text("utf-8")
    m = re.search(rf"^(#+) {re.escape(heading)}\s*$", text, re.M)
    assert m, f"{path.relative_to(ROOT)} has no heading '{heading}'"
    level = len(m.group(1))
    end = re.compile(rf"^#{{1,{level}}} ", re.M).search(text, m.end())
    return text[m.end(): end.start() if end else len(text)]


def test_agents_md_states_the_process_rule():
    deps = _plain(_section(ROOT / "AGENTS.md", "Dependencies"))
    missing = [s for s in re.split(r"(?<=\.) (?=[A-Z`])", _plain(RULE)) if s not in deps]
    assert not missing, f"AGENTS.md 'Dependencies' lacks these sentences of the process rule: {missing}"


def test_install_doc_licensing_table_has_rows_for_the_generator_and_the_transport():
    rows = [r for r in _section(ROOT / "docs" / "install.md", "Licensing of each wheel").splitlines()
            if r.lstrip().startswith("|")]
    gen = [r for r in rows if "axmp_generate.py" in r]
    tra = [r for r in rows if "axmp-transport" in r]
    assert gen, "docs/install.md 'Licensing of each wheel' has no row for the generator file axmp_generate.py"
    assert tra, "docs/install.md 'Licensing of each wheel' has no row for axmp-transport"
    assert "Pythia" in gen[0] and "GPL" in gen[0], f"the generator's row does not say it runs GPL Pythia: {gen[0]}"
    assert "Geant4" in tra[0] and ("CLHEP" in tra[0] or "LGPL" in tra[0]), (
        f"the transport's row does not say it links Geant4 with its LGPL-3 CLHEP: {tra[0]}")


def test_the_transport_guide_carries_geant4s_acknowledgement():
    assert TRANSPORT_GUIDE.is_file(), f"{TRANSPORT_GUIDE.relative_to(ROOT)} (the transport's user page) is missing"
    assert G4_NOTICE.search(_plain(TRANSPORT_GUIDE.read_text("utf-8"))), (
        "the Geant4 licence (clause 2) requires the notice 'This product includes software developed by Members of "
        "the Geant4 Collaboration ( http://cern.ch/geant4 ).' in the user documentation")
    nav = (ROOT / "mkdocs.yml").read_text("utf-8")
    assert "guides/particles.md" in nav, "docs/guides/particles.md is not in the mkdocs navigation"


def test_the_pinned_tarballs_have_the_design_notes_checksums():
    pins = {}
    for line in (ROOT / "third_party" / "SHA256SUMS.txt").read_text("utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            pins[parts[1]] = parts[0]
    expected = {
        # computed from the official downloads; the Geant4 one is the GitLab archive of the tag v11.4.3
        "geant4-v11.4.3.tar.gz": "718752c1829cbfcf174f0e539d60ee3f76843b894444e61d2228d66a6887ef6e",
        "pythia8318.tgz": "85dce1e623f91499b2973e3f939bf760b0f745ad4f4eb1bd0fbce2074e2e8f5c",
        "xerces-c-3.3.0.tar.gz": "9555f1d06f82987fbb4658862705515740414fd34b4db6ad2ed76a2dc08d3bde",
    }
    for name, sha in expected.items():
        assert pins.get(name) == sha, f"third_party/SHA256SUMS.txt: {name} is {pins.get(name)}, expected {sha}"
