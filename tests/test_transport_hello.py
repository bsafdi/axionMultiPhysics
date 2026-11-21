"""axmp-transport: ``--check``, the hello-world mode, and the reproducibility of its events (the part of the
particle-transport reproducibility checks that needs only the program).

Needs the program: set ``AXMP_TRANSPORT`` to it (``cmake -DAXMP_BUILD_TRANSPORT=ON``, the target ``axmp-transport``;
the development gate builds it against the cluster's Geant4). Without the variable every test here skips;
with it set to something that is not an executable file every test fails.

The contract tested (docs/guides/particles.md documents it):

- ``axmp-transport --check`` prints one JSON document on stdout and nothing else;
- ``axmp-transport [--threads N] job.json`` runs the chunks the job lists in one process and writes, per chunk,
  ``<out>/chunk_<k>/manifest.json`` and one NPY file per array, and a record ``<out>/record.json``; relative paths in
  the job resolve against the job file's directory;
- every event is seeded at the top of the primary generator action with ``G4Random::setTheSeeds(words, 4)``, the
  four 32-bit words being the outputs 2e and 2e+1 of the SplitMix64 sequence that starts at the chunk seed, split
  low half first. An event then depends on (chunk seed, event index) only, not on the thread that ran it, the
  number of threads, the chunks run before it in the process, or Geant4's own per-event seeds (which a run of more
  than 10 000 events refills from the master engine);
- the program refuses to start when Geant4's ``SeedOncePerCommunication`` is not 0;
- field calls are counted per event.

The reproducibility claims are bitwise (merged results, per-event records): arrays are compared byte for byte. The
seed words are recomputed here from that specification (SplitMix64 below, checked against its published test
vector), and the first number drawn in each event is compared with MixMax references recorded outside this
repository (``FIRST_FLAT``, with its provenance).

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.px_transport

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 1800  # one process: Geant4's start-up and at most 25 000 cheap events

# ----------------------------------------------------------------------------------------------- seeds, independently
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def splitmix64_mix(z: int) -> int:
    """The output function of SplitMix64 (Steele, Lea and Flood 2014; Vigna's public-domain splitmix64.c)."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def splitmix64_sequence(seed: int, n: int) -> list[int]:
    """The first n outputs of SplitMix64 started at the state ``seed``: the state advances by GOLDEN before each output."""
    return [splitmix64_mix((seed + (i + 1) * GOLDEN) & MASK64) for i in range(n)]


def event_words(chunk_seed: int, event: int) -> list[int]:
    """The four 32-bit words of an event: outputs 2e and 2e+1 of the sequence at the chunk seed, low half first."""
    o0 = splitmix64_mix((chunk_seed + (2 * event + 1) * GOLDEN) & MASK64)
    o1 = splitmix64_mix((chunk_seed + (2 * event + 2) * GOLDEN) & MASK64)
    return [o0 & 0xFFFFFFFF, o0 >> 32, o1 & 0xFFFFFFFF, o1 >> 32]


def test_splitmix64_matches_its_published_test_vector():
    """The SplitMix64 of this file against the vector commonly published for splitmix64.c with the seed 1234567."""
    assert splitmix64_sequence(1234567, 5) == [6457827717110365317, 3203168211198807973, 9817491932198370423,
                                               4593380528125082431, 16408922859458223821]
    seq = splitmix64_sequence(0xFEDCBA9876543210, 8)
    assert event_words(0xFEDCBA9876543210, 3) == [seq[6] & 0xFFFFFFFF, seq[6] >> 32, seq[7] & 0xFFFFFFFF,
                                                  seq[7] >> 32]


# The first number drawn in an event, as a float64 bit pattern, for (chunk seed, event index). Recorded (not computed
# here, and not by any code of this repository): CLHEP::MixMaxRng compiled from the source of the pinned Geant4
# 11.4.3 tarball (sha256 718752c1...ef6e: source/externals/clhep/src/MixMaxRng.cc, RandomEngine.cc,
# engineIDulong.cc), a fresh engine, setSeeds(words, 4) with event_words() above, then flat(). Seeding an engine that
# had already drawn 1000 numbers gave the same values: setSeeds fixes the whole state. The recording program and its
# input are kept outside the repository (it links CLHEP, LGPL-3).
FIRST_FLAT = {
    (0x0000000000000001, 0): 0x3fe7a317b8324f51,
    (0x0000000000000001, 1): 0x3fe65e8cb153f726,
    (0x0000000000000001, 2): 0x3fd328275b0dad3c,
    (0x0000000000000001, 49): 0x3fe3d22e62d6ef7a,
    (0x0000000000000001, 199): 0x3fedf7ff666c07fe,
    (0xfedcba9876543210, 0): 0x3feff4f1d99fa9c6,
    (0xfedcba9876543210, 1): 0x3fee5ed4ddb50cf4,
    (0xfedcba9876543210, 199): 0x3fe006a4c2d80415,
    (0x00000000deadbeef, 0): 0x3fef04d0ffb5b720,
    (0x00000000deadbeef, 7): 0x3fa86959b59020a2,
    (0x00000000deadbeef, 199): 0x3fe1d0581f27be0d,
    (0xab54a98ceb1f0ad2, 0): 0x3fed1c59c8a4140d,
    (0xab54a98ceb1f0ad2, 9): 0x3fcf77abe3fe956b,
    (0xab54a98ceb1f0ad2, 9999): 0x3fc0ef783f6c9343,
    (0xab54a98ceb1f0ad2, 10000): 0x3f99f9fee27af495,
    (0xab54a98ceb1f0ad2, 10001): 0x3fd9682c99b61ebe,
    (0xab54a98ceb1f0ad2, 17777): 0x3fdd8bf9744eadec,
    (0xab54a98ceb1f0ad2, 24999): 0x3fdd9d03da15152f,
}

# chunk seeds: small, top bit set, 32-bit, and one above 2^63 for the long chunk
SEED0, SEED1, SEED2, SEED3 = 1, 0xFEDCBA9876543210, 0xDEADBEEF, 12345678901234567890
LONG = 25_000           # beyond the 10 000 events after which Geant4 refills its per-event seeds from the master
E_LO, E_HI = 5.0e8, 1.5e9   # the hello source's kinetic energy range, eV
FIELD_T = [0.0, 1.0, 0.0]

ARRAYS = {  # name: (dtype, trailing shape)
    "event": ("<u8", ()), "seed_words": ("<u4", (4,)), "first_flat": ("<f8", ()), "ekin_eV": ("<f8", ()),
    "edep_eV": ("<f8", ()), "n_steps": ("<u8", ()), "n_tracks": ("<u8", ()), "field_calls": ("<u8", ()),
}


# ----------------------------------------------------------------------------------------------- running the program
@pytest.fixture(scope="module")
def transport() -> str:
    exe = os.environ.get("AXMP_TRANSPORT")
    if not exe:
        pytest.skip("axmp-transport is not built here: set AXMP_TRANSPORT to it (cmake -DAXMP_BUILD_TRANSPORT=ON)")
    if not (Path(exe).is_file() and os.access(exe, os.X_OK)):
        pytest.fail(f"AXMP_TRANSPORT={exe} is not an executable file")
    return exe


def hello_job(chunks, field_t=FIELD_T, **extra) -> dict:
    job = {
        "schema": "axionmultiphysics.transport-job/1",
        "mode": "hello",
        "out": "out",
        "chunks": [{"index": k, "seed": s, "events": n} for k, s, n in chunks],
        "physics_list": "FTFP_BERT",
        "hello": {"particle": "mu-", "energy_eV": [E_LO, E_HI], "field_T": list(field_t), "target_thickness_m": 0.01},
    }
    job.update(extra)
    return job


def run_job(exe: str, workdir: Path, job: dict, threads: int) -> tuple[subprocess.CompletedProcess, Path]:
    """Write the job into its own directory and run it from another one, so that relative paths must resolve against
    the job file's directory."""
    workdir.mkdir(parents=True)
    (workdir / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    elsewhere = workdir / "cwd"
    elsewhere.mkdir()
    r = subprocess.run([exe, "--threads", str(threads), str(workdir / "job.json")], capture_output=True, text=True,
                       timeout=TIMEOUT, cwd=elsewhere)
    return r, workdir / "out"


def _ok(r: subprocess.CompletedProcess, what: str):
    assert r.returncode == 0, f"{what}: exit status {r.returncode}\nstderr (end):\n{r.stderr[-4000:]}"


def load_chunk(out: Path, k: int) -> dict[str, np.ndarray]:
    """The arrays of chunk k, after checking the manifest against them (schema, dtype, shape, sha256)."""
    d = out / f"chunk_{k}"
    assert (d / "manifest.json").is_file(), f"no {d}/manifest.json (written: {sorted(p.name for p in out.glob('*'))})"
    man = json.loads((d / "manifest.json").read_text("utf-8"))
    assert man.get("schema") == "axionmultiphysics.transport/1" and man.get("mode") == "hello", man.get("schema")
    assert man.get("chunk") == k, man.get("chunk")
    n = man["events"]
    arrays = {}
    for name, (dtype, tail) in ARRAYS.items():
        entry = man["arrays"].get(name)
        assert entry, f"chunk {k}: the manifest lists no array '{name}'"
        f = d / entry["file"]
        assert hashlib.sha256(f.read_bytes()).hexdigest() == entry["sha256"], f"chunk {k}: sha256 of {f.name}"
        a = np.load(f, allow_pickle=False)
        assert a.dtype.str == dtype == entry["dtype"], (name, a.dtype.str, entry["dtype"])
        assert a.shape == (n, *tail) and list(a.shape) == entry["shape"], (name, a.shape, entry["shape"])
        arrays[name] = a
    arrays["_manifest"] = man
    return arrays


def assert_bitwise(a: dict, b: dict, what: str, rows: slice = slice(None)):
    for name in ARRAYS:
        x, y = a[name][rows], b[name][rows]
        if x.tobytes() != y.tobytes():
            bad = np.nonzero(np.any((x != y).reshape(len(x), -1), axis=1))[0]
            pytest.fail(f"{what}: '{name}' differs in {len(bad)} of {len(x)} events, first at event {bad[:5]}")


# One process per configuration, shared by the tests (start-up dominates a small run).
# A and B: four chunks in one process, 1 and 4 threads; C: chunk 2 alone; E and F: one chunk of 25 000 events, 4 and 1
# threads (its first 10 events are chunk 3 of A and B, same seed); N: no field.
FOUR = [(0, SEED0, 200), (1, SEED1, 200), (2, SEED2, 200), (3, SEED3, 10)]


@pytest.fixture(scope="module")
def runs(transport, tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("hello")
    plan = {
        "A": (hello_job(FOUR), 1),
        "B": (hello_job(FOUR), 4),
        "C": (hello_job([(2, SEED2, 200)]), 4),
        "E": (hello_job([(3, SEED3, LONG)]), 4),
        "F": (hello_job([(3, SEED3, LONG)]), 1),
        "N": (hello_job([(0, SEED0, 50)], field_t=[0.0, 0.0, 0.0]), 2),
    }
    res = {}
    for name, (job, threads) in plan.items():
        r, out = run_job(transport, base / name, job, threads)
        res[name] = {"proc": r, "out": out, "job": job, "threads": threads}
    return res


def chunk(runs, name, k):
    _ok(runs[name]["proc"], f"run {name} ({runs[name]['threads']} threads)")
    return load_chunk(runs[name]["out"], k)


# ----------------------------------------------------------------------------------------------- --check
def test_check_prints_only_json_describing_the_installation(transport):
    r = subprocess.run([transport, "--check"], capture_output=True, text=True, timeout=600)
    _ok(r, "--check")
    try:
        info = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"--check must print one JSON document on stdout and nothing else (Geant4's banner and messages "
                    f"go to stderr): {e}\nstdout (start): {r.stdout[:600]}")
    assert info["schema"] == "axionmultiphysics.transport-check/1"
    assert info["program"] == "axmp-transport"
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text("utf-8"), re.M).group(1)
    assert info["program_version"] == version, (info["program_version"], version)
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", info["geant4_version"])
    assert m, info["geant4_version"]
    v = tuple(int(x) for x in m.groups())
    assert (11, 2, 0) <= v < (11, 5, 0), f"a Geant4 outside 11.2 to 11.4 passed the build's check: {v}"
    assert info["geant4_tested"] is (v == (11, 4, 3)), info["geant4_tested"]
    assert info["multithreaded"] is True
    assert "FTFP_BERT" in info["physics_lists"] and "hello" in info["modes"], info
    sets = info["datasets"]
    assert sets and all({"name", "env", "path", "status", "required"} <= set(s) for s in sets), sets
    assert all(s["status"] in ("ok", "missing") for s in sets), sets
    assert any(s["name"] == "G4EMLOW" and s["required"] for s in sets), "G4EMLOW (needed by every list) is not listed"
    missing = [s["name"] for s in sets if s["required"] and s["status"] != "ok"]
    assert info["ok"] is (not missing), (info["ok"], missing)
    assert info["ok"], f"the data sets the offered physics lists need are missing: {missing}"


# ----------------------------------------------------------------------------------------------- reproducibility
def test_hello_run_is_bitwise_equal_for_1_and_4_threads(runs):
    """Also covers 'the same seed twice': a run that is not reproducible (seeds from the clock, seeds per thread,
    events in arrival order) differs between these two runs."""
    for k, _, _ in FOUR:
        assert_bitwise(chunk(runs, "A", k), chunk(runs, "B", k), f"chunk {k}, 1 thread against 4 threads")


def test_a_chunk_alone_equals_the_same_chunk_after_others(runs):
    assert_bitwise(chunk(runs, "C", 2), chunk(runs, "A", 2), "chunk 2 alone against chunk 2 after chunks 0 and 1")
    assert_bitwise(chunk(runs, "C", 2), chunk(runs, "B", 2), "chunk 2 alone against chunk 2 after chunks 0 and 1")


def test_a_chunk_beyond_10000_events_is_bitwise_equal_for_1_and_4_threads(runs):
    assert_bitwise(chunk(runs, "E", 3), chunk(runs, "F", 3), f"a chunk of {LONG} events, 4 threads against 1")


def test_a_short_chunk_is_the_beginning_of_the_long_chunk_with_its_seed(runs):
    """An event depends on (chunk seed, event index) only: a chunk of 10 events is the first 10 of a chunk of 25 000
    (what lets an extension rerun a partial last chunk in full)."""
    assert_bitwise(chunk(runs, "E", 3), chunk(runs, "A", 3), "the first 10 events of the long chunk", slice(0, 10))


def test_events_are_written_in_index_order_with_the_specified_seed_words(runs):
    for name, k, seed, n in (("A", 0, SEED0, 200), ("A", 1, SEED1, 200), ("B", 2, SEED2, 200), ("E", 3, SEED3, LONG)):
        c = chunk(runs, name, k)
        assert c["_manifest"]["chunk_seed"] == seed and c["_manifest"]["events"] == n, c["_manifest"]
        assert np.array_equal(c["event"], np.arange(n, dtype=np.uint64)), f"run {name} chunk {k}: events out of order"
        want = np.array([event_words(seed, e) for e in range(n)], dtype="<u4")
        bad = np.nonzero(np.any(c["seed_words"] != want, axis=1))[0]
        assert not len(bad), (f"run {name} chunk {k}: the seed words of {len(bad)} events are not SplitMix64 of "
                              f"(chunk seed, event), first at event {bad[:5]}: {c['seed_words'][bad[0]]} != "
                              f"{want[bad[0]]}")


def test_the_first_draw_in_generate_primaries_equals_the_reference_mixmax(runs):
    """The reseed is the first statement of GeneratePrimaries: a reseed in BeginOfEventAction (after the primaries are
    made), two words instead of four, or Geant4's own seeds (the initialize-seeds callback, refilled from the master
    after 10 000 events) all give another first number."""
    where = {SEED0: ("A", 0), SEED1: ("B", 1), SEED2: ("A", 2), SEED3: ("E", 3)}
    version = None
    wrong = []
    for (seed, event), bits in FIRST_FLAT.items():
        c = chunk(runs, *where[seed])
        version = c["_manifest"].get("geant4_version")
        got = int(c["first_flat"][event:event + 1].view("<u8")[0])
        if got != bits:
            wrong.append(f"(seed {seed:#x}, event {event}): {got:#018x}, reference {bits:#018x}")
    assert not wrong, (f"first number drawn in GeneratePrimaries against MixMax seeded with setSeeds(words, 4) "
                       f"(reference recorded with Geant4 11.4.3's CLHEP; this build: {version}): {wrong}")


def test_the_hello_source_draws_its_energy_from_the_first_number(runs):
    """Ties the recorded first number to the primary the event actually transported."""
    for name, k in (("A", 0), ("A", 1), ("E", 3)):
        c = chunk(runs, name, k)
        u = c["first_flat"]
        assert np.all((u > 0) & (u < 1)), f"run {name} chunk {k}: a first draw outside (0, 1)"
        assert len(np.unique(u)) == len(u), f"run {name} chunk {k}: repeated first draws"
        want = E_LO + (E_HI - E_LO) * u  # IEEE double, the order of operations the specification fixes
        # one rounding apart at most where a compiler contracts the expression into a fused multiply-add
        assert np.allclose(c["ekin_eV"], want, rtol=4e-16, atol=0), f"run {name} chunk {k}: ekin_eV != lo+(hi-lo)*u"


def test_chunks_with_different_seeds_have_different_events(runs):
    a, b = chunk(runs, "A", 0), chunk(runs, "A", 1)
    assert a["seed_words"].tobytes() != b["seed_words"].tobytes()
    assert np.count_nonzero(a["edep_eV"] != b["edep_eV"]) > 190, "two chunk seeds gave (nearly) the same deposits"


def test_the_hello_events_are_transported(runs):
    """Muons of 0.5 to 1.5 GeV (near minimum ionizing, beta*gamma 5 to 15) crossing 1 cm of silicon: the mean energy
    loss is about 3.9 MeV (1.66 MeV cm2/g x 2.33 g/cm3) and the most probable about 2.9 MeV; the median deposit lies
    well inside 1 to 10 MeV unless no physics runs."""
    c = chunk(runs, "E", 3)
    med = float(np.median(c["edep_eV"]))
    assert 1.0e6 < med < 1.0e7, f"median deposit {med:.4g} eV"
    assert np.count_nonzero(c["edep_eV"] > 0) >= 0.99 * LONG
    assert np.all(c["n_steps"] >= 1) and np.all(c["n_tracks"] >= 1)
    assert np.count_nonzero(c["n_tracks"] > 1) > 0.05 * LONG, "no secondaries (delta rays) at all"


def test_field_calls_are_counted_per_event(runs):
    with_field = chunk(runs, "A", 0)
    assert np.all(with_field["field_calls"] > 0), "an event in a 1 T field with no field call counted"
    none = chunk(runs, "N", 0)
    assert np.all(none["field_calls"] == 0), "field calls counted where no field is attached"
    # the field does not touch the source: the first draws of the first 50 events are those of the run with a field
    assert none["first_flat"].tobytes() == with_field["first_flat"][:50].tobytes()
    for name in ("A", "B", "E"):
        rec = json.loads((runs[name]["out"] / "record.json").read_text("utf-8"))
        total = sum(int(chunk(runs, name, k)["field_calls"].sum()) for k in rec["chunks"])
        assert rec["field_calls_total"] == total, (name, rec["field_calls_total"], total)


def test_the_record_reports_start_up_time_threads_and_seeding_mode(runs):
    for name, r in runs.items():
        _ok(r["proc"], f"run {name}")
        rec = json.loads((r["out"] / "record.json").read_text("utf-8"))
        assert rec["schema"] == "axionmultiphysics.transport-record/1" and rec["status"] == "ok", rec
        assert rec["threads"] == r["threads"], rec["threads"]
        assert rec["chunks"] == [c["index"] for c in r["job"]["chunks"]], rec["chunks"]
        assert rec["seed_once_per_communication"] == 0, rec
        t = rec["timing"]
        assert math.isfinite(t["startup_s"]) and t["startup_s"] > 0 and t["total_s"] >= t["startup_s"], t
        assert not list(r["out"].glob("*.tmp")), f"run {name}: a chunk left in its temporary directory"


# ----------------------------------------------------------------------------------------------- refusals
def test_seed_once_per_communication_other_than_0_is_refused_at_start_up(transport, tmp_path):
    job = hello_job([(0, SEED0, 20)], debug={"seed_once_per_communication": 1})
    r, out = run_job(transport, tmp_path / "sopc", job, 2)
    assert r.returncode == 3, f"exit status {r.returncode} (3: refused at set-up)\nstderr:\n{r.stderr[-3000:]}"
    assert "SeedOncePerCommunication" in r.stderr, r.stderr[-3000:]
    assert not list(out.glob("chunk_*")), "events were written although the start-up check refused the run"


def test_an_invalid_job_is_refused_before_any_event(transport, tmp_path):
    job = hello_job([(0, SEED0, 20)], chunk_evnts=5)  # a misspelt key
    r, out = run_job(transport, tmp_path / "typo", job, 1)
    assert r.returncode == 2, f"exit status {r.returncode} (2: invalid job)\nstderr:\n{r.stderr[-3000:]}"
    assert "chunk_evnts" in r.stderr, "the error does not name the unknown key"
    assert not list(out.glob("chunk_*"))
    job = hello_job([(0, 1 << 64, 20)])  # a chunk seed that is not a 64-bit unsigned integer
    r, out = run_job(transport, tmp_path / "seed", job, 1)
    assert r.returncode == 2, f"exit status {r.returncode} for a chunk seed of 2^64\nstderr:\n{r.stderr[-3000:]}"


def test_an_unknown_particle_is_refused_as_an_invalid_job(transport, tmp_path):
    """Added by the implementer (not part of the test author's SPEC): the particle named in the job is only known to
    Geant4 once the physics list is set, and Initialize() starts the worker threads, so an unknown name must be
    caught before it. Exit status 2, the name in the message, nothing written, whatever the thread count."""
    job = hello_job([(0, SEED0, 20)])
    job["hello"]["particle"] = "nonexistent_particle"
    r, out = run_job(transport, tmp_path / "particle", job, 2)
    assert r.returncode == 2, f"exit status {r.returncode} (2: invalid job)\nstderr:\n{r.stderr[-3000:]}"
    assert "nonexistent_particle" in r.stderr, r.stderr[-3000:]
    assert not out.exists() or not list(out.glob("chunk_*")), "events were written for an unknown particle"
