"""The particle-transport driver ``axionmultiphysics.particles`` (beta), without Geant4.

The design note: §4.4 (failure modes), §7 (seeds, stop rules, what is identical to what), §8.4 (plan, pool, merge,
extend), §10 (estimators and the honesty rules), §13.7 (S1's driver part, S3 at the level of the estimator, S4). The
contract is written out in docs/guides/particles.md.

Everything here runs in every build: the driver is pure Python, and the program it drives is replaced by
``tests/_transport_stub.py``, a stand-in that implements ``axmp-transport``'s documented file contract with a
deterministic function of each event's seed words instead of physics. So the driver's own guarantees (the chunk
seeds, the job files, the order of the merge, grouping, extension, prefixes, the exact 128-bit batch sums, the
estimators, partial runs and their exit status) are tested bit for bit and in seconds. The physics of the real program
is tested in tests/test_particles_transport.py.

References are independent (tests/_particles_ref.py): SplitMix64 and the seed derivation from their specification,
scipy's beta distribution for Clopper-Pearson, the batch-means formula, the PDG rounding rule.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import copy
import json
import re
import math
import os
import stat
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from axionmultiphysics import particles  # noqa: E402  (the module under test)
from axionmultiphysics import solids as solids_mod  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _particles_ref as ref  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUB = Path(__file__).resolve().parent / "_transport_stub.py"
E_KIN = 1.0e9          # eV: the stand-in deposits u E_KIN in the first region
CODES = (
    "build.requires_program", "events.count_missing", "events.invalid", "events.workdir_not_empty",
    "events.chunk_events_mismatch", "events.chunk_mismatch", "events.partial", "stat.too_few_chunks",
    "particle.unknown_particle", "particle.source_invalid", "particle.output_invalid", "particle.field_invalid",
    "particle.field_ambiguous", "particle.field_outside_map", "particle.integrator_invalid",
    "particle.delta_intersection_not_small", "particle.physics_list_unknown", "particle.physics_list_unavailable",
    "particle.tally_overflow", "particle.partition_unavailable", "particle.density_differs_from_nist",
    "particle.looping_killed", "events.workdir_busy", "events.field_map_changed", "particle.track_abandoned",
)


# ============================================================================================ fixtures
@pytest.fixture(scope="module")
def stub(tmp_path_factory) -> str:
    d = tmp_path_factory.mktemp("stub")
    exe = d / "axmp-transport"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{STUB}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(exe)


def slab_doc(thin_layer=0.02, rmax=0.3):
    """gas | iron | gas | layer | gas along z, element compositions only (nothing to resolve)."""
    z = [-0.5, 0.0, 0.1, 0.3, 0.3 + thin_layer, 0.5]
    P, regions = ref.slabs(z, ["gas", "iron", "gas", "layer", "gas"], rmax)
    mats = {"hgas": ref.H_GAS, "fe": ref.FE_ELEM}
    return solids_mod.from_partition(P.table(regions), mats, {"gas": "hgas", "iron": "fe", "layer": "hgas"})


GUN = {"type": "gun", "particle": "e-", "energy_eV": E_KIN, "at_m": [0.0, 0.0, -0.4], "direction": [0.0, 0.0, 1.0]}
DEP = {"E": {"deposit": {"in": "gas"}}, "Em": {"mean": {"of": "E"}}}


def kw(**over):
    k = {"solids": slab_doc(), "sources": [dict(GUN)], "outputs": copy.deepcopy(DEP), "events": 100,
         "chunk_events": 10, "seed": 5}
    k.update(over)
    return k


def code_of(excinfo) -> str:
    return getattr(excinfo.value, "code", None)


def run_stub(wd, stub, workers=1, **over):
    k = kw(**over)
    return particles.run(wd, program=stub, workers=workers, **k)


def strip_timing(obj):
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in ("timing", "programs")}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj


def results_json(wd):
    return strip_timing(json.loads((Path(wd) / "results.json").read_text()))


def chunk_arrays(wd, k):
    d = Path(wd) / "out" / f"chunk_{k}"
    man = json.loads((d / "manifest.json").read_text())
    return {n: np.load(d / e["file"]) for n, e in man["arrays"].items()}, man


def u_of(wd, k):
    return chunk_arrays(wd, k)[0]["first_flat"]


# ============================================================================================ the module
def test_module_declares_its_constants_entry_points_and_beta_status():
    assert "beta" in (particles.__doc__ or "").lower()
    assert particles.SCHEMA_JOB == "axionmultiphysics.transport-job/1"
    assert particles.SCHEMA_CHUNK == "axionmultiphysics.transport/1"
    assert particles.SCHEMA_RUN == "axionmultiphysics.particles-run/1"
    assert particles.SCHEMA_RESULTS == "axionmultiphysics.particles-results/1"
    assert particles.BATCHES == 20 and particles.CHUNK_EVENTS == 1000
    assert particles.QUANTUM_EV == 1e-3 and particles.PRODUCTION_CUT_M == 7e-4
    assert tuple(particles.PHYSICS_LISTS) == ("FTFP_BERT", "FTFP_BERT_EMZ", "QGSP_BIC")
    assert particles.INTEGRATOR == {"delta_chord_m": 2.5e-4, "delta_one_step_m": 1e-5, "delta_intersection_m": 1e-6,
                                    "epsilon_min": 5e-5, "epsilon_max": 1e-3, "min_step_m": 1e-5}
    assert issubclass(particles.ParticleWarning, UserWarning)
    assert issubclass(particles.Measured, float)
    for name in ("derive_seed", "chunk_seed", "event_words", "capabilities", "resolve_materials", "plan", "execute",
                 "merge", "run", "extend", "load", "transport", "estimate_mean", "estimate_fraction",
                 "estimate_count", "batch_means"):
        assert callable(getattr(particles, name)), name


def test_importing_the_driver_loads_no_geant4_or_pythia_module():
    """The driver runs the programs as separate processes (§3): importing it brings no Geant4 or Pythia binding into
    the process."""
    code = ("import sys, axionmultiphysics.particles as p; "
            "bad = [m for m in sys.modules if m.split('.')[0].lower() in "
            "('pythia8', 'pythia8mc', 'geant4_pybind', 'geant4', 'hepmc3', 'pyhepmc', 'root', 'cppyy')]; print(bad)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "[]", r.stdout


# ============================================================================================ seeds (§7.2)
def test_seed_derivation_follows_its_specification():
    cases = [(1, ("transport", "transport", 0)), (1, ("transport", "transport", 1)), (0, ("driver", "x", 7)),
             (2 ** 64 - 1, ("transport", "transport", 2 ** 63)), (12345, ()), (7, ("é-stage", 3))]
    for seed, keys in cases:
        assert particles.derive_seed(seed, *keys) == ref.derive_seed(seed, *keys), (seed, keys)
    # a seed beyond 64 bits is reduced first; ints and strings hash differently
    assert particles.derive_seed(2 ** 64 + 5, "a") == ref.derive_seed(5, "a")
    assert particles.derive_seed(1, "1") != particles.derive_seed(1, 1)
    for k in range(5):
        assert particles.chunk_seed(1, k) == ref.chunk_seed(1, k) == ref.derive_seed(1, "transport", "transport", k)
    assert particles.chunk_seed(3, 4, stage="s", instance="i") == ref.derive_seed(3, "s", "i", 4)


def test_chunk_seeds_are_distinct_and_event_words_are_those_of_the_program():
    seeds = [particles.chunk_seed(1, k) for k in range(20000)]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < 2 ** 64 for s in seeds)
    for cs, e in ((1, 0), (1, 199), (0xFEDCBA9876543210, 3), (12345678901234567890, 24999)):
        assert list(particles.event_words(cs, e)) == ref.event_words(cs, e)


# ============================================================================================ estimators (§10.1)
def test_mean_is_the_standard_error_over_defined_events():
    rng = np.random.default_rng(11)
    x = rng.normal(3.0, 2.0, 501)
    x[::7] = np.nan
    m = particles.estimate_mean(x, units="eV")
    d = x[~np.isnan(x)]
    assert isinstance(m, particles.Measured)
    assert float(m) == pytest.approx(d.mean(), rel=1e-14)
    assert m.stat_error == pytest.approx(d.std(ddof=1) / math.sqrt(len(d)), rel=1e-12)
    assert (m.events, m.events_defined, m.units, m.error, m.mc) == (501, len(d), "eV", "sem", True)
    one = particles.estimate_mean(np.array([2.0, np.nan]))
    assert float(one) == 2.0 and one.stat_error is None and one.events_defined == 1


@pytest.mark.parametrize("k,n", [(1, 10), (7, 10), (10, 10), (76, 10000), (5000, 10000)])
def test_fraction_is_clopper_pearson_at_one_sigma(k, n):
    f = particles.estimate_fraction(k, n)
    lo, hi = ref.clopper_pearson(k, n)
    assert float(f) == k / n
    assert f.interval[0] == pytest.approx(lo, rel=1e-9, abs=1e-15)
    assert f.interval[1] == pytest.approx(hi, rel=1e-9, abs=1e-15)
    assert f.stat_error == pytest.approx((hi - lo) / 2, rel=1e-9)
    assert f.error == "clopper_pearson" and f.events == n and f.upper_limit is None and f.units == "1"


@pytest.mark.parametrize("n", [10, 10000])
def test_zero_observed_events_are_an_exact_upper_limit_never_zero_plus_minus_zero(n):
    """S4: 0 of N is the exact one-sided 95 % limit 1 - 0.05^(1/N) (3.0/N is 16 % too large at N = 10)."""
    f = particles.estimate_fraction(0, n)
    assert float(f) == 0.0 and f.stat_error is None
    assert f.upper_limit == pytest.approx(ref.zero_fraction_limit(n), rel=1e-12)
    assert f.cl == 0.95 and f.error == "upper_limit"
    assert str(f).startswith("< ") and "95 % CL" in str(f) and f"{n} events" in str(f)
    assert "+-" not in str(f) and "0 +- 0" not in str(f)
    c = particles.estimate_count(0)
    assert float(c) == 0.0 and c.stat_error is None and c.upper_limit == pytest.approx(ref.ZERO_COUNT_LIMIT, rel=1e-12)
    c7 = particles.estimate_count(49)
    assert c7.stat_error == 7.0 and c7.error == "poisson" and c7.upper_limit is None


def test_batch_means_weights_each_batch_by_its_events():
    rng = np.random.default_rng(3)
    v = rng.normal(1.0, 0.1, 20)
    w = rng.integers(5, 15, 20)
    vbar, s = particles.batch_means(v, w)
    rv, rs = ref.batch_sigma(v, w)
    assert vbar == pytest.approx(rv, rel=1e-14) and s == pytest.approx(rs, rel=1e-12)
    vb, sb = particles.batch_means(v, np.ones(20))       # equal weights: the standard error of the batch means
    assert sb == pytest.approx(np.std(v, ddof=1) / math.sqrt(20), rel=1e-12)


@pytest.mark.parametrize("value,err,text", [
    (0.99812, 0.00213, "0.9981 +- 0.0021 (2000 events)"),
    (0.99812, 0.00456, "0.998 +- 0.005 (2000 events)"),
    (0.99812, 0.00097, "0.9981 +- 0.0010 (2000 events)"),
    (123.456, 12.3, "123 +- 12 (2000 events)"),
    (1.2345e-7, 3.1e-9, "(1.235 +- 0.031)e-7 (2000 events)"),
    (2.5e6, 1.2e4, "(2.500 +- 0.012)e6 (2000 events)"),
])
def test_measured_prints_no_digit_beyond_what_its_error_supports(value, err, text):
    m = particles.Measured(value, stat_error=err, error="sem", events=2000, events_defined=2000, units="eV")
    assert str(m) == text
    assert float(m) == value and m.units == "eV" and m.mc is True
    assert type(m + 1.0) is float                 # arithmetic loses the error explicitly, never silently keeps it
    if not text.startswith("("):
        v, e, _ = ref.pdg_round(value, err)        # the reference rule agrees with the expected text
        assert text.startswith(f"{v} +- {e} ")


def test_the_mean_estimator_covers_the_truth_as_stated():
    """S3 at the level of the estimator: 1000 independent experiments of 1e4 Bernoulli events with the cone
    acceptance (1 - cos 10 deg) / 2; the 1 sigma interval of `mean` covers the truth in 68.3 +- 5.9 % (4 binomial
    sigma) of them, and Clopper-Pearson's in at least 62.4 % (it never under-covers)."""
    p = (1 - math.cos(math.radians(10))) / 2
    assert p == pytest.approx(7.596e-3, rel=1e-3)
    rng = np.random.default_rng(314159)
    cov_mean = cov_cp = 0
    for _ in range(1000):
        x = (rng.random(10000) < p).astype(float)
        m = particles.estimate_mean(x)
        cov_mean += abs(float(m) - p) <= m.stat_error
        f = particles.estimate_fraction(int(x.sum()), 10000)
        cov_cp += f.interval[0] <= p <= f.interval[1]
    assert abs(cov_mean / 1000 - 0.683) <= 0.059, cov_mean
    assert cov_cp / 1000 >= 0.683 - 0.059, cov_cp


# ============================================================================================ plan (§8.4, §4.4)
def test_plan_writes_job_files_chunk_seeds_and_command_lines(tmp_path, stub):
    wd = tmp_path / "w"
    rec = particles.plan(wd, program=stub, jobs=3, **kw(events=95, chunk_events=10))
    run = json.loads((wd / "run.json").read_text())
    assert rec == run or strip_timing(rec) == strip_timing(run)
    assert run["schema"] == particles.SCHEMA_RUN and run["beta"] is True
    assert (run["seed"], run["stage"], run["instance"], run["chunk_events"], run["events_requested"]) == \
        (5, "transport", "transport", 10, 95)
    ch = run["chunks"]
    assert [c["index"] for c in ch] == list(range(10))
    assert [c["events"] for c in ch] == [10] * 9 + [5]
    assert [c["seed"] for c in ch] == [ref.chunk_seed(5, k) for k in range(10)]
    assert all(c["status"] == "planned" for c in ch)
    jobs = sorted((wd / "exchange").glob("job_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    assert len(jobs) == 3
    blocks = []
    for p in jobs:
        job = json.loads(p.read_text())
        assert job["schema"] == particles.SCHEMA_JOB and job["mode"] == "transport"
        assert not Path(job["solids"]).is_absolute() and not Path(job["out"]).is_absolute()
        assert (p.parent / job["solids"]).resolve() == (wd / "exchange" / "solids.json").resolve()
        assert (p.parent / job["out"]).resolve() == (wd / "out").resolve()
        assert job["source"]["particle"] == "e-" and job["source"]["energy_eV"] == E_KIN
        assert job["integrator"] == particles.INTEGRATOR
        assert (job["physics_list"], job["production_cut_m"], job["quantum_eV"], job["outside"]) == \
            ("FTFP_BERT", 7e-4, 1e-3, "refuse")
        blocks.append([c["index"] for c in job["chunks"]])
        for c in job["chunks"]:
            assert (c["seed"], c["events"]) == (ch[c["index"]]["seed"], ch[c["index"]]["events"])
    assert sum(blocks, []) == list(range(10))                       # contiguous blocks in order
    assert max(map(len, blocks)) - min(map(len, blocks)) <= 1
    cmds = (wd / "exchange" / "commands.txt").read_text().splitlines()
    assert len(cmds) == 3 and all(str(jobs[i]) in cmds[i] or jobs[i].name in cmds[i] for i in range(3))
    assert json.loads((wd / "exchange" / "solids.json").read_text())["schema"] == "axionmultiphysics.solids/1"
    text = json.dumps(strip_timing(run))
    assert str(tmp_path) not in text, "run.json outside programs/timing holds a local path"


def test_planning_twice_writes_identical_job_files(tmp_path, stub):
    a = particles.plan(tmp_path / "a", program=stub, jobs=2, **kw())
    b = particles.plan(tmp_path / "b", program=stub, jobs=2, **kw())
    assert strip_timing(a) == strip_timing(b)
    for name in ("job_0.json", "job_1.json", "solids.json"):
        assert (tmp_path / "a" / "exchange" / name).read_bytes() == (tmp_path / "b" / "exchange" / name).read_bytes()


@pytest.mark.parametrize("events,chunk_events,code", [
    (None, 10, "events.count_missing"), (0, 10, "events.invalid"), (-3, 10, "events.invalid"),
    (2.5, 10, "events.invalid"), (True, 10, "events.invalid"), (100, 0, "events.invalid"),
])
def test_the_event_count_is_required_and_checked(tmp_path, stub, events, chunk_events, code):
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "w", program=stub, **kw(events=events, chunk_events=chunk_events))
    assert code_of(ei) == code
    assert not (tmp_path / "w" / "run.json").exists()


def test_a_workdir_holding_other_files_is_refused(tmp_path, stub):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "notes.txt").write_text("mine")
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "w", program=stub, **kw())
    assert code_of(ei) == "events.workdir_not_empty"


def test_a_near_miss_particle_name_gets_candidates(tmp_path, stub):
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "w", program=stub, **kw(sources=[dict(GUN, particle="muon")]))
    assert code_of(ei) == "particle.unknown_particle"
    assert "mu-" in str(ei.value) and "mu+" in str(ei.value) and "muon" in str(ei.value)
    particles.plan(tmp_path / "pdg", program=stub, **kw(sources=[dict(GUN, particle=13)]))


@pytest.mark.parametrize("change", [
    {"momentum_eV_c": 1e9},                         # both energy and momentum
    {"energy_eV": -1.0}, {"energy_eV": 0.0},
    {"direction": [0.0, 0.0, 0.0]},
    {"direction": "isotropic", "spot_sigma_m": 0.01},
    {"spot_sigma_m": -0.1},
    {"type": "isotope"},
    {"colour": "red"},
])
def test_a_gun_must_state_one_positive_energy_and_a_direction(tmp_path, stub, change):
    src = dict(GUN, **change)
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "w", program=stub, **kw(sources=[src]))
    assert code_of(ei) == "particle.source_invalid"


def test_exactly_one_source(tmp_path, stub):
    for srcs in ([], [dict(GUN), dict(GUN)]):
        with pytest.raises(Exception) as ei:
            particles.plan(tmp_path / f"w{len(srcs)}", program=stub, **kw(sources=srcs))
        assert code_of(ei) == "particle.source_invalid"


def test_physics_lists_are_the_three_offered(tmp_path, stub):
    for name in particles.PHYSICS_LISTS:
        particles.plan(tmp_path / name, program=stub, **kw(physics_list=name))
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "x", program=stub, **kw(physics_list="FTFP_BERTT"))
    assert code_of(ei) == "particle.physics_list_unknown"
    assert all(n in str(ei.value) for n in particles.PHYSICS_LISTS)
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "hp", program=stub, **kw(physics_list="QGSP_BIC_HP"))
    assert code_of(ei) == "particle.physics_list_unavailable"


@pytest.mark.parametrize("outputs", [
    {"E": {"deposit": {"in": "nowhere"}}},
    {"E": {"deposit": {"in": "gas"}, "mean": {"of": "E"}}},
    {"E": {"energy": {"in": "gas"}}},
    {"1E": {"deposit": {"in": "gas"}}},
    {"m": {"mean": {"of": "missing"}}},
    {"E": {"deposit": {"in": "gas"}}, "m": {"mean": {"of": "E", "where": {"E": {"gt": 1, "between": 2}}}}},
    {"E": {"deposit": {"in": "gas"}}, "h": {"histogram": {"of": "E", "bins": 0, "range": [0, 1]}}},
    {"E": {"deposit": {"in": "gas"}}, "h": {"histogram": {"of": "E", "bins": 10, "range": [1, 0]}}},
    {"x": {"hit": {"in": "layer", "of": "w"}}},
    {"t": {"tally": {"grid": {"type": "spherical", "bins": [3]}}}},
    {"t": {"tally": {"grid": {"type": "cylindrical", "r_m": [0, 0.3], "z_m": [-0.5, 0.5], "bins": [0, 4]}}}},
    {"h": {"hits": {"in": ["layer"], "max_per_event": 0}}},
    {"q": {"quantile": {"of": "E", "q": 1.5}}, "E": {"deposit": {"in": "gas"}}},
])
def test_outputs_are_validated_before_anything_runs(tmp_path, stub, outputs):
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "w", program=stub, **kw(outputs=outputs))
    assert code_of(ei) == "particle.output_invalid"
    assert not (tmp_path / "w" / "exchange").exists() or not list((tmp_path / "w" / "exchange").glob("job_*"))


def write_hand_map(path, box_r, box_z, spacing, regions=None, kind=1, zero_field_regions=(), bz=0.5):
    """An axionmultiphysics.fieldmap/1 directory written by hand from the documented format (one patch 'p', B_z = bz);
    an existing directory is replaced."""
    import hashlib
    import io
    import shutil
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    (path / "patch_0").mkdir()
    n = [int(round((box_r[1] - box_r[0]) / spacing)) + 1, int(round((box_z[1] - box_z[0]) / spacing)) + 1]
    B = np.zeros((*n, 3))
    B[..., 1] = bz
    b = io.BytesIO()
    np.save(b, B)
    data = b.getvalue()
    (path / "patch_0" / "B.npy").write_bytes(data)
    rec = {"name": "p", "dir": "patch_0", "file": "patch_0/B.npy", "sha256": hashlib.sha256(data).hexdigest(),
           "bytes": B.size * 8, "kind": kind, "frame": {"type": "axisymmetric", "axis": "z"},
           "origin": [box_r[0], box_z[0]], "spacing": [spacing, spacing], "shape": n, "margin": [spacing, spacing]}
    if regions is not None:
        rec["regions"] = regions
    man = {"schema": "axionmultiphysics.fieldmap/1", "units": {"length": "m", "B": "T"}, "outside": "refuse",
           "source": {"type": "nodal_values"}, "patches": [rec], "zero_field_regions": list(zero_field_regions)}
    (path / "manifest.json").write_text(json.dumps(man))
    return path


def test_fields_serve_each_region_once_and_maps_must_cover_their_regions(tmp_path, stub):
    uni = {"type": "uniform", "B_T": [0.0, 0.0, 1.0]}
    with pytest.raises(Exception) as ei:          # two fields for one region
        particles.plan(tmp_path / "a", program=stub, **kw(fields=[dict(uni), dict(uni, regions=["iron"])]))
    assert code_of(ei) == "particle.field_ambiguous" and "iron" in str(ei.value)
    with pytest.raises(Exception) as ei:          # an unknown region
        particles.plan(tmp_path / "b", program=stub, **kw(fields=[dict(uni, regions=["copper"])]))
    assert code_of(ei) == "particle.field_invalid" and "copper" in str(ei.value)
    small = write_hand_map(tmp_path / "small", (0.0, 0.2), (-0.5, 0.5), 0.05)       # r <= 0.25 with the margin
    big = write_hand_map(tmp_path / "big", (0.0, 0.3), (-0.5, 0.5), 0.05)
    with pytest.raises(Exception) as ei:          # a write_map patch names no regions
        particles.plan(tmp_path / "c", program=stub, **kw(fields=[{"type": "map", "path": str(big), "patch": "p"}]))
    assert code_of(ei) == "particle.field_invalid"
    every = ["gas", "iron", "layer"]
    with pytest.raises(Exception) as ei:          # the solids reach r = 0.3 > 0.2 + 0.05
        particles.plan(tmp_path / "d", program=stub,
                       **kw(fields=[{"type": "map", "path": str(small), "patch": "p", "regions": every}]))
    assert code_of(ei) == "particle.field_outside_map" and "p" in str(ei.value)
    particles.plan(tmp_path / "e", program=stub,
                   **kw(fields=[{"type": "map", "path": str(big), "patch": "p", "regions": every}]))
    with pytest.raises(Exception) as ei:          # iron and layer served by nothing
        particles.plan(tmp_path / "f", program=stub, **kw(fields=[dict(uni, regions=["gas"])]))
    assert code_of(ei) == "particle.field_outside_map" and "iron" in str(ei.value)
    # a whole map: each patch serves its record's regions; zero_field_regions are field-free by the map
    whole = write_hand_map(tmp_path / "whole", (0.0, 0.3), (-0.5, 0.5), 0.05, regions=["gas", "layer"],
                           zero_field_regions=["iron"])
    particles.plan(tmp_path / "g", program=stub, **kw(fields=[{"type": "map", "path": str(whole)}]))
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "h", program=stub, **kw(fields=[{"type": "map", "path": str(whole)}], outside="maybe"))
    assert code_of(ei) == "particle.field_invalid"


def test_the_integrator_is_validated_against_maps_and_thin_layers(tmp_path, stub):
    big = write_hand_map(tmp_path / "big", (0.0, 0.3), (-0.5, 0.5), 0.05)
    f = [{"type": "map", "path": str(big), "patch": "p", "regions": ["gas", "iron", "layer"]}]
    for bad in ({"delta_chord": 1e-3}, {"epsilon_min": 1e-3, "epsilon_max": 1e-4}, {"epsilon_max": 0.1},
                {"delta_one_step_m": -1e-5}, {"delta_intersection_m": 0.006}):
        with pytest.raises(Exception) as ei:
            particles.plan(tmp_path / f"w{len(list(tmp_path.iterdir()))}", program=stub,
                           **kw(fields=f, integrator=bad))
        assert code_of(ei) == "particle.integrator_invalid", bad
    # 0.1 x the patch spacing 0.05 is the largest deltaIntersection a map tolerates
    particles.plan(tmp_path / "ok", program=stub, **kw(fields=f, integrator={"delta_intersection_m": 0.004}))
    with pytest.warns(particles.ParticleWarning, match=r"^particle\.delta_intersection_not_small"):
        particles.plan(tmp_path / "thin", program=stub,
                       **kw(solids=slab_doc(thin_layer=1e-3), integrator={"delta_intersection_m": 2e-4}))
    rec = json.loads((tmp_path / "ok" / "run.json").read_text())
    assert rec["integrator"]["delta_intersection_m"] == 0.004
    assert rec["integrator"]["stepper"] == "G4TDormandPrince45+G4IntegrationDriver"


def test_a_tally_that_could_overflow_a_chunk_is_refused(tmp_path, stub):
    grid = {"type": "cylindrical", "r_m": [0.0, 0.3], "z_m": [0.0, 0.1], "bins": [3, 2]}
    out = dict(copy.deepcopy(DEP), q={"tally": {"grid": grid, "in": ["iron"]}})
    # 10 events x (1 GeV + m_e) / quantum must stay below 2^63
    q_ok = 10 * (E_KIN + 0.51099895e6) / 2.0 ** 63 * 1.001
    particles.plan(tmp_path / "ok", program=stub, **kw(outputs=out, quantum_eV=q_ok))
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "no", program=stub, **kw(outputs=out, quantum_eV=q_ok / 1.002))
    assert code_of(ei) == "particle.tally_overflow"


def test_a_missing_program_names_the_variable_to_set(tmp_path, monkeypatch):
    monkeypatch.delenv("AXMP_TRANSPORT", raising=False)
    monkeypatch.setenv("AXMP_EXE", str(tmp_path / "no_axmp"))
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(Exception) as ei:
        particles.plan(tmp_path / "w", **kw())
    assert code_of(ei) == "build.requires_program"
    assert "AXMP_TRANSPORT" in str(ei.value) and "AXMP_BUILD_TRANSPORT" in str(ei.value)


def test_capabilities_never_raise_and_report_the_partition_export(stub, monkeypatch, tmp_path):
    monkeypatch.setenv("AXMP_TRANSPORT", stub)
    c = particles.capabilities()
    assert c["transport"]["found"] is True and c["transport"]["ok"] is True
    assert c["transport"]["check"]["schema"] == "axionmultiphysics.transport-check/1"
    assert c["partition_export"] == {"axisymmetric": False, "3d": False}
    monkeypatch.setenv("AXMP_TRANSPORT", str(tmp_path / "nothing"))
    monkeypatch.setenv("AXMP_EXE", str(tmp_path / "no_axmp"))
    monkeypatch.setenv("PATH", str(tmp_path))
    c = particles.capabilities()
    assert c["transport"]["found"] is False and c["transport"]["ok"] is False


def test_the_api_route_needs_the_partition_export_or_explicit_solids(tmp_path):
    import axionmultiphysics as axmp
    doc = {"schema": "axionmultiphysics.design/1", "name": "s",
           "regions": [{"name": "winding", "material": "winding", "shape": {"rect": {"r": [0.5, 0.55], "z": [-0.5, 0.5]}}}],
           "windings": [{"name": "w", "region": "winding", "type": "azimuthal", "ampere_turns": 1e5}]}
    with pytest.raises(Exception) as ei:
        particles.transport(axmp.Design(doc), None, sources=[dict(GUN)], outputs=copy.deepcopy(DEP), events=10,
                            workdir=tmp_path / "w")
    assert code_of(ei) == "particle.partition_unavailable"
    assert "solids" in str(ei.value)


# ============================================================================================ NIST compositions
def water_doc(density, block=None):
    P, regions = ref.slabs([-0.5, 0.0, 0.5], ["gas", "water"], 0.3)
    mats = {"hgas": ref.H_GAS, "water": {"density": density, "particle": block or {"nist": "G4_WATER"}}}
    return solids_mod.from_partition(P.table(regions), mats, {"gas": "hgas", "water": "water"})


def test_nist_compositions_are_resolved_against_the_programs_tables(stub):
    same = water_doc(1000.0)
    before = copy.deepcopy(same)
    r = particles.resolve_materials(same, program=stub)
    assert same == before, "the input document was changed"
    water_solids = [s for s in r["solids"] if s.get("region") == "water"]
    assert water_solids and all(s["material"] == "G4_WATER" for s in water_solids)
    assert "water" not in [m["name"] for m in r["materials"]]
    assert r["regions"]["water"]["material"] == "water"
    # another density: one component with NIST's I-value and state carried
    r = particles.resolve_materials(water_doc(998.2), program=stub)
    m = next(m for m in r["materials"] if m["name"] == "water")
    assert (m["mean_excitation_eV"], m["state"], m["density_kg_m3"]) == (78.0, "liquid", 998.2)
    assert all(s["material"] == "water" for s in r["solids"] if s.get("region") == "water")
    # values the document gives are kept; a NIST block with options is never replaced by the NIST material
    r = particles.resolve_materials(water_doc(1000.0, {"nist": "G4_WATER", "mean_excitation_eV": 75.0}), program=stub)
    m = next(m for m in r["materials"] if m["name"] == "water")
    assert (m["mean_excitation_eV"], m["state"]) == (75.0, "liquid")
    # more than 1 % from NIST's density warns (the document's density is still used)
    with pytest.warns(particles.ParticleWarning, match=r"^particle\.density_differs_from_nist"):
        r = particles.resolve_materials(water_doc(1015.0), program=stub)
    assert next(m for m in r["materials"] if m["name"] == "water")["density_kg_m3"] == 1015.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", particles.ParticleWarning)
        particles.resolve_materials(water_doc(1005.0), program=stub)


def test_the_gdml_of_a_resolved_list_references_the_nist_material(tmp_path, stub):
    import xml.etree.ElementTree as ET
    r = particles.resolve_materials(water_doc(1000.0), program=stub)
    solids_mod.write_gdml(r, tmp_path / "a.gdml")
    root = ET.parse(tmp_path / "a.gdml").getroot()
    refs = {v.get("name"): v.find("materialref").get("ref") for v in root.iter("volume")}
    water_vols = [s["gdml_name"] for s in r["solids"] if s.get("region") == "water"]
    assert water_vols and all(refs[v] == "G4_WATER" for v in water_vols)
    assert not [m for m in root.iter("material") if m.get("name", "").startswith("G4_")]
    r = particles.resolve_materials(water_doc(998.2), program=stub)
    solids_mod.write_gdml(r, tmp_path / "b.gdml")
    root = ET.parse(tmp_path / "b.gdml").getroot()
    gname = next(m["gdml_name"] for m in r["materials"] if m["name"] == "water")
    mat = next(m for m in root.iter("material") if m.get("name") == gname)
    assert mat.get("state") == "liquid"
    mee = mat.find("MEE")
    assert mee is not None and float(mee.get("value")) == 78.0 and mee.get("unit") == "eV"


def test_the_driver_resolves_before_it_runs_and_records_it(tmp_path, stub):
    particles.plan(tmp_path / "w", program=stub, **kw(solids=water_doc(998.2), outputs={"E": {"deposit": {"in": "water"}}}))
    rec = json.loads((tmp_path / "w" / "run.json").read_text())
    res = rec["materials_resolved"]["water"]
    assert res["as"] == "one_component" and res["mean_excitation_eV"] == 78.0
    used = json.loads((tmp_path / "w" / "exchange" / "solids.json").read_text())
    assert next(m for m in used["materials"] if m["name"] == "water")["state"] == "liquid"


# ============================================================================================ execute and merge (§7.3)
def test_merged_results_do_not_depend_on_jobs_or_workers(tmp_path, stub):
    a = run_stub(tmp_path / "a", stub, workers=1, jobs=1)
    b = run_stub(tmp_path / "b", stub, workers=3, jobs=3)
    c = run_stub(tmp_path / "c", stub, workers=2, jobs=10, threads=4)
    assert results_json(tmp_path / "a") == results_json(tmp_path / "b") == results_json(tmp_path / "c")
    for r in (b, c):
        for k, v in a.events().items():
            assert v.tobytes() == r.events()[k].tobytes(), k
    for k in range(10):
        x, _ = chunk_arrays(tmp_path / "a", k)
        y, _ = chunk_arrays(tmp_path / "c", k)
        assert all(x[n].tobytes() == y[n].tobytes() for n in x)


def test_the_merge_orders_events_by_chunk_and_event(tmp_path, stub):
    r = run_stub(tmp_path / "w", stub, events=35, chunk_events=10)
    ev = r.events()
    assert list(ev["chunk"]) == [0] * 10 + [1] * 10 + [2] * 10 + [3] * 5
    assert list(ev["event"]) == list(range(10)) * 3 + list(range(5))
    u = np.concatenate([u_of(tmp_path / "w", k) for k in range(4)])
    assert np.array_equal(ev["E"], u * E_KIN)
    want = [ref.event_words(ref.chunk_seed(5, k), e) for k in range(4) for e in range(10 if k < 3 else 5)]
    got = np.concatenate([chunk_arrays(tmp_path / "w", k)[0]["seed_words"] for k in range(4)])
    assert got.tolist() == want
    assert r.n_events == 35 and r.events_requested == 35 and r.partial is False and r.missing_chunks == []
    run = json.loads((tmp_path / "w" / "run.json").read_text())
    assert all(c["status"] == "complete" and c["attempts"] == 1 and len(c["sha256"]) == 64 for c in run["chunks"])


def test_an_extension_from_a_partial_chunk_equals_the_direct_run(tmp_path, stub):
    direct = run_stub(tmp_path / "d", stub, events=40, chunk_events=10)
    run_stub(tmp_path / "e", stub, events=25, chunk_events=10)        # chunk 2 holds 5 events
    ext = particles.extend(tmp_path / "e", events=40)
    assert results_json(tmp_path / "e") == results_json(tmp_path / "d")
    assert ext.events()["E"].tobytes() == direct.events()["E"].tobytes()
    x, mx = chunk_arrays(tmp_path / "e", 2)
    assert mx["events"] == 10                                             # rerun in full and replaced
    y, _ = chunk_arrays(tmp_path / "d", 2)
    assert all(x[n].tobytes() == y[n].tobytes() for n in x)
    with pytest.raises(Exception) as ei:
        particles.extend(tmp_path / "e", events=50, chunk_events=20)
    assert code_of(ei) == "events.chunk_events_mismatch"
    with pytest.raises(Exception) as ei:
        particles.extend(tmp_path / "e", events=30)
    assert code_of(ei) == "events.invalid"


def prefix_rel_error(wd, K):
    x = np.concatenate([u_of(wd, k) for k in range(K)]) * E_KIN
    return x.std(ddof=1) / math.sqrt(len(x)) / abs(x.mean())


def test_a_stop_rule_ends_at_the_first_complete_prefix_that_meets_it_whatever_the_workers(tmp_path, stub):
    """§7.1: evaluated on prefixes of the chunk sequence in index order, never before 20 chunks; complete chunks
    beyond the prefix are not merged; the result does not depend on the number of workers."""
    seen = {}
    for workers in (1, 4):
        wd = tmp_path / f"w{workers}"
        r = run_stub(wd, stub, workers=workers, jobs=25, events=1000, chunk_events=10,
                     stop={"output": "Em", "rel_error": 0.035})
        st = r.record["stop"]
        K = st["chunks"]
        assert st["reason"] == "rel_error" and st["output"] == "Em" and st["rel_error"] == 0.035
        assert 20 <= K < 100 and st["events"] == 10 * K == r.n_events
        assert prefix_rel_error(wd, K) <= 0.035
        assert K == 20 or prefix_rel_error(wd, K - 1) > 0.035
        assert all(c["status"] in ("planned", "beyond_stop") for c in r.record["chunks"][K:])
        assert r.partial is False
        seen[workers] = (K, results_json(wd), r.events()["E"].tobytes())
    assert seen[1] == seen[4]


def test_a_stop_rule_never_fires_before_20_chunks(tmp_path, stub):
    r = run_stub(tmp_path / "w", stub, events=400, chunk_events=10, stop={"output": "Em", "rel_error": 0.5})
    assert r.record["stop"]["chunks"] == 20 and r.n_events == 200


def test_a_failed_job_is_retried_once(tmp_path, stub, monkeypatch):
    monkeypatch.setenv("AXMP_STUB_FAIL_ONCE", str(tmp_path / "failed_once"))
    r = run_stub(tmp_path / "w", stub, events=50, chunk_events=10, debug={"fail_chunks": [2]})
    assert (tmp_path / "failed_once").exists()
    assert r.partial is False and r.n_events == 50
    st = {c["index"]: c for c in r.record["chunks"]}
    assert st[2]["attempts"] == 2 and st[0]["attempts"] == 1
    clean = run_stub(tmp_path / "clean", stub, events=50, chunk_events=10)
    assert results_json(tmp_path / "w") == results_json(tmp_path / "clean")


def test_a_chunk_that_keeps_failing_leaves_the_complete_prefix_and_a_partial_result(tmp_path, stub):
    with pytest.warns(particles.ParticleWarning, match=r"^events\.partial"):
        r = run_stub(tmp_path / "a", stub, jobs=5, events=50, chunk_events=10, debug={"fail_chunks": [2]})
    assert r.partial is True and r.n_events == 20 and r.missing_chunks == [2, 3, 4]
    st = {c["index"]: c["status"] for c in r.record["chunks"]}
    assert st[2] == "failed" and st[3] == st[4] == "complete"
    assert json.loads((tmp_path / "a" / "results.json").read_text())["partial"] is True
    with pytest.warns(particles.ParticleWarning, match=r"^events\.partial"):
        r = particles.merge(tmp_path / "a", allow_failed_chunks=True)
    assert r.partial is True and r.n_events == 40 and r.missing_chunks == [2]
    assert list(np.unique(r.events()["chunk"])) == [0, 1, 3, 4]


def driver_cli(*args):
    return subprocess.run([sys.executable, "-m", "axionmultiphysics.particles", *map(str, args)], capture_output=True,
                          text=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))


def test_the_command_line_exits_5_on_a_partial_run(tmp_path, stub):
    particles.plan(tmp_path / "p", program=stub, jobs=5, **kw(events=50, chunk_events=10, debug={"fail_chunks": [3]}))
    r = driver_cli("execute", tmp_path / "p", "--workers", 2)
    assert r.returncode == 5, (r.returncode, r.stderr[-2000:])
    out = json.loads(r.stdout)
    assert out["status"] == "partial" and out["events"] == 30 and out["events_requested"] == 50
    assert out["missing_chunks"] == [3, 4]
    r = driver_cli("merge", tmp_path / "p")
    assert r.returncode == 5 and json.loads(r.stdout)["status"] == "partial"
    r = driver_cli("merge", tmp_path / "p", "--allow-failed-chunks")
    assert r.returncode == 5 and json.loads(r.stdout)["events"] == 40
    particles.plan(tmp_path / "c", program=stub, **kw(events=30, chunk_events=10))
    r = driver_cli("execute", tmp_path / "c")
    assert r.returncode == 0 and json.loads(r.stdout)["status"] == "complete"
    r = driver_cli("extend", tmp_path / "c", "--events", 45)
    assert r.returncode == 0 and json.loads(r.stdout)["events"] == 45
    r = driver_cli("merge", tmp_path / "nothing_here")
    assert r.returncode == 1 and "code" in json.loads(r.stdout)
    assert driver_cli("frobnicate", tmp_path / "c").returncode == 2


def test_a_chunk_that_disagrees_with_the_plan_is_refused_and_a_corrupt_one_is_missing(tmp_path, stub):
    run_stub(tmp_path / "w", stub, events=30, chunk_events=10)
    d = tmp_path / "w" / "out" / "chunk_1"
    man = json.loads((d / "manifest.json").read_text())
    good = dict(man)
    man["chunk_seed"] = man["chunk_seed"] ^ 1
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(Exception) as ei:
        particles.merge(tmp_path / "w")
    assert code_of(ei) == "events.chunk_mismatch" and "1" in str(ei.value)
    (d / "manifest.json").write_text(json.dumps(good))
    a = np.load(d / "first_flat.npy")
    a[0] = 0.5
    np.save(d / "first_flat.npy", a)
    with pytest.warns(particles.ParticleWarning, match=r"^events\.partial"):
        r = particles.merge(tmp_path / "w")
    assert r.missing_chunks == [1, 2] and r.n_events == 10


# ============================================================================================ L1 tallies (§7.3)
def exact_batch_sums(wd, name, nchunks):
    sums = {}
    for k in range(nchunks):
        T = chunk_arrays(wd, k)[0][f"tally_{name}"].astype(object)
        b = k % 20
        sums[b] = sums.get(b, 0) + T
    return sums


def test_tally_batches_are_exact_128_bit_sums_beyond_2_to_the_63(tmp_path, stub):
    """S1's last item: 200 chunks of 2 events, a quantum that puts each event near 2^62 quanta; every batch holds 10
    chunks, so its sums pass 2^63 and an int64 accumulator would wrap."""
    q = 2 * (E_KIN + 0.51099895e6) / 2.0 ** 63 * 1.0001
    grid = {"type": "cartesian", "x_m": [-0.3, 0.3], "y_m": [-0.3, 0.3], "z_m": [-0.5, 0.5], "bins": [2, 1, 1]}
    out = dict(copy.deepcopy(DEP), q={"tally": {"grid": grid}})
    r = run_stub(tmp_path / "w", stub, workers=4, jobs=4, events=400, chunk_events=2, quantum_eV=q, outputs=out)
    m = tmp_path / "w" / "merged"
    lo = np.load(m / "tally_q_lo.npy")
    hi = np.load(m / "tally_q_hi.npy")
    ne = np.load(m / "tally_q_events.npy")
    assert lo.dtype.str == "<u8" and hi.dtype.str == "<i8" and ne.dtype.str == "<i8"
    assert lo.shape == hi.shape == (20, 2, 1, 1) and ne.tolist() == [20] * 20
    want = exact_batch_sums(tmp_path / "w", "q", 200)
    over = 0
    for b in range(20):
        for i in range(2):
            exact = (int(hi[b, i, 0, 0]) << 64) + int(lo[b, i, 0, 0])
            assert exact == int(want[b].reshape(-1)[i]), (b, i)
            over += exact >= 2 ** 63
    assert over >= 20, "the case does not exercise sums beyond 2^63"
    t = r.tally("q")
    total = [sum(int(want[b].reshape(-1)[i]) for b in range(20)) for i in range(2)]
    assert np.allclose(t["edep_eV"].reshape(-1), [x * q / 400 for x in total], rtol=1e-15, atol=0)
    assert t["quantum_eV"] == q and t["events"] == 400


def test_tally_errors_are_batch_means_and_need_twenty_chunks(tmp_path, stub):
    grid = {"type": "cylindrical", "r_m": [0.0, 0.3], "z_m": [-0.5, 0.5], "bins": [1, 3]}
    out = dict(copy.deepcopy(DEP), q={"tally": {"grid": grid}})
    r = run_stub(tmp_path / "a", stub, events=410, chunk_events=10, outputs=out)     # 41 chunks, uneven batches
    t = r.tally("q")
    sums = exact_batch_sums(tmp_path / "a", "q", 41)
    nb = [sum(10 for k in range(41) if k % 20 == b) for b in range(20)]
    for i in range(3):
        v = [int(sums[b].reshape(-1)[i]) * 1e-3 / nb[b] for b in range(20)]
        vbar, s = ref.batch_sigma(v, nb)
        assert t["stat_error"].reshape(-1)[i] == pytest.approx(s, rel=1e-10)
        assert t["edep_eV"].reshape(-1)[i] == pytest.approx(vbar, rel=1e-12)
    vol = t["volume_m3"].reshape(-1)
    assert vol == pytest.approx([math.pi * 0.09 * (1.0 / 3)] * 3, rel=1e-12)
    assert np.allclose(t["q_dep_J_m3"].reshape(-1), t["edep_eV"].reshape(-1) * 1.602176634e-19 / vol, rtol=1e-12)
    with pytest.warns(particles.ParticleWarning, match=r"^stat\.too_few_chunks"):
        r = run_stub(tmp_path / "b", stub, events=190, chunk_events=10, outputs=out)
    assert r.tally("q")["stat_error"] is None


# ============================================================================================ aggregates (§10)
def test_aggregates_carry_their_estimators_and_the_honesty_fields(tmp_path, stub):
    half = E_KIN / 2
    outputs = {
        "E": {"deposit": {"in": "gas"}}, "Em": {"mean": {"of": "E"}}, "Es": {"std": {"of": "E"}},
        "Eq": {"quantile": {"of": "E", "q": 0.5}},
        "f": {"fraction": {"where": {"E": {"gt": half}}}}, "c": {"count": {"where": {"E": {"gt": half}}}},
        "fz": {"fraction": {"where": {"E": {"lt": 0.0}}}},
        "h": {"histogram": {"of": "E", "bins": 8, "range": [0.1 * E_KIN, 0.9 * E_KIN]}},
        "Em_low": {"mean": {"of": "E", "where": {"any": [{"E": {"lt": 0.2 * E_KIN}}, {"E": {"ge": 0.9 * E_KIN}}]}}},
    }
    r = run_stub(tmp_path / "w", stub, events=1000, chunk_events=25, outputs=outputs)
    x = r.events()["E"]
    chunk = r.events()["chunk"]
    assert np.array_equal(x, np.concatenate([u_of(tmp_path / "w", k) for k in range(40)]) * E_KIN)
    m = r["Em"]
    assert isinstance(m, particles.Measured) and m.units == "eV" and m.error == "sem"
    assert float(m) == pytest.approx(x.mean(), rel=1e-14)
    assert m.stat_error == pytest.approx(x.std(ddof=1) / math.sqrt(1000), rel=1e-12)
    batches = [x[chunk % 20 == b] for b in range(20)]
    for name, stat in (("Es", lambda a: np.std(a, ddof=1)), ("Eq", lambda a: np.quantile(a, 0.5))):
        v = r[name]
        assert float(v) == pytest.approx(stat(x), rel=1e-13) and v.error == "batch_means"
        _, s = ref.batch_sigma([stat(a) for a in batches], [len(a) for a in batches])
        assert v.stat_error == pytest.approx(s, rel=1e-10)
    k = int(np.sum(x > half))
    f = r["f"]
    assert float(f) == k / 1000 and f.interval == pytest.approx(ref.clopper_pearson(k, 1000), rel=1e-9)
    assert float(r["c"]) == k and r["c"].stat_error == pytest.approx(math.sqrt(k))
    assert r["fz"].upper_limit == pytest.approx(ref.zero_fraction_limit(1000), rel=1e-12)
    sel = (x < 0.2 * E_KIN) | (x >= 0.9 * E_KIN)
    assert float(r["Em_low"]) == pytest.approx(x[sel].mean(), rel=1e-14) and r["Em_low"].events_defined == sel.sum()
    h = r.histogram("h")
    edges = np.linspace(0.1 * E_KIN, 0.9 * E_KIN, 9)
    assert np.array_equal(h["edges"], edges)
    assert h["counts"].tolist() == [int(np.sum((x >= edges[i]) & (x < edges[i + 1]))) for i in range(8)]
    assert np.allclose(h["errors"], np.sqrt(h["counts"]))
    assert h["underflow"] == int(np.sum(x < 0.1 * E_KIN)) and h["overflow"] == int(np.sum(x >= 0.9 * E_KIN))
    assert h["undefined"] == 0
    res = json.loads((tmp_path / "w" / "results.json").read_text())
    assert res["schema"] == particles.SCHEMA_RESULTS and res["beta"] is True and res["partial"] is False
    assert res["mc"] == ("Monte Carlo (beta): 1000 events, seed 5; statistical errors only; "
                         "physics-list dependence not estimated")
    for name in ("Em", "Es", "Eq", "f", "c"):
        o = res["outputs"][name]
        assert o["mc"] is True and o["beta"] is True
        assert {"value", "stat_error", "error", "events", "events_defined", "units"} <= set(o)
        assert o["events"] == 1000
    assert res["outputs"]["fz"]["upper_limit"] == pytest.approx(ref.zero_fraction_limit(1000), rel=1e-12)
    assert res["outputs"]["fz"]["cl"] == 0.95 and res["outputs"]["fz"]["stat_error"] is None
    again = particles.load(tmp_path / "w")
    assert float(again["Em"]) == float(m) and again["Em"].stat_error == m.stat_error


def test_hits_are_capped_per_event_and_kept_for_the_first_events(tmp_path, stub):
    outputs = {"E": {"deposit": {"in": "gas"}},
               "trk": {"hits": {"in": ["layer"], "max_per_event": 2, "keep_events": 7}}}
    r = run_stub(tmp_path / "w", stub, events=15, chunk_events=5, outputs=outputs)
    h = r.hits("trk")
    u = np.concatenate([u_of(tmp_path / "w", k) for k in range(3)])
    want_n = np.minimum(np.floor(4 * u).astype(int) + 1, 2)
    keys = list(zip(h["chunk"].tolist(), h["event"].tolist()))
    assert keys == sorted(keys)
    glob = [5 * c + e for c, e in keys]
    assert set(glob) <= set(range(7)), "records beyond the first 7 events in (chunk, event) order"
    assert [glob.count(i) for i in range(7)] == want_n[:7].tolist()
    assert np.array_equal(h["x_m"][:, 0], np.repeat(u[:7], want_n[:7]))
    s = json.loads((tmp_path / "w" / "results.json").read_text())["outputs"]["trk"]
    dropped = np.maximum(np.floor(4 * u).astype(int) + 1 - 2, 0)
    assert s["dropped"] == int(dropped.sum()) and s["records"] == int(want_n[:7].sum())
    assert s["events_with_records"] == 7 and s["events_without_records"] == 8


def test_hit_variables_take_the_first_hit_of_the_primary(tmp_path, stub):
    outputs = {"hx": {"hit": {"in": "layer", "of": "x"}}, "hr": {"hit": {"in": "layer", "of": "r"}},
               "hE": {"hit": {"in": "layer", "of": "E_kin"}}, "hm": {"mean": {"of": "hx"}}}
    r = run_stub(tmp_path / "w", stub, events=20, chunk_events=10, outputs=outputs)
    u = np.concatenate([u_of(tmp_path / "w", k) for k in range(2)])
    ev = r.events()
    assert np.array_equal(ev["hx"], u)
    assert np.allclose(ev["hr"], np.hypot(u, ev["event"].astype(float)), rtol=1e-15)
    assert np.all(ev["hE"] == E_KIN)
    assert float(r["hm"]) == pytest.approx(u.mean(), rel=1e-14) and r["hm"].units == "m"
    jobs = list((tmp_path / "w" / "exchange").glob("job_*.json"))
    hits = json.loads(jobs[0].read_text())["hits"]
    assert any(h["regions"] == ["layer"] and h["track"] == "primary" for h in hits)


def test_calls_outside_the_maps_are_counted_and_warned_with_outside_zero(tmp_path, stub):
    f = [{"type": "uniform", "B_T": [0.0, 0.0, 1.0], "regions": ["gas"]}]
    with pytest.warns(particles.ParticleWarning, match=r"^particle\.field_outside_map") as w:
        r = run_stub(tmp_path / "w", stub, events=30, chunk_events=10, fields=f, outside="zero")
    assert any("30" in str(x.message) for x in w)
    assert r.record["outside"] == "zero"
    assert any(x["code"] == "particle.field_outside_map" for x in r.record["warnings"])


def test_looping_tracks_carrying_more_than_a_millionth_of_the_deposit_are_warned(tmp_path, stub):
    """§4.4: looping particles killed in a field are counted with their energy; a warning above 1e-6 of the
    deposited energy (the stand-in deposits u E_KIN, about E_KIN / 2 per event)."""
    with pytest.warns(particles.ParticleWarning, match=r"^particle\.looping_killed"):
        r = run_stub(tmp_path / "a", stub, events=20, debug={"stub_looping_eV": 2e-6 * E_KIN})
    assert any(w["code"] == "particle.looping_killed" for w in r.record["warnings"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", particles.ParticleWarning)
        r = run_stub(tmp_path / "b", stub, events=20, debug={"stub_looping_eV": 1e-7 * E_KIN})
    assert not any(w["code"] == "particle.looping_killed" for w in r.record["warnings"])


# ============================================================================================ after the first review
EVERY = ["gas", "iron", "layer"]


def test_a_workdir_reached_by_another_name_finds_its_field_maps(tmp_path, stub):
    """The job files carry each field map's real absolute path (run.json keeps a path relative to the workdir), so a
    run planned through a symbolic link executes through the physical path, and the other way round. The program
    resolves a job's paths textually against the job file's directory, as the stand-in does."""
    maps = write_hand_map(tmp_path / "maps" / "m", (0.0, 0.3), (-0.5, 0.5), 0.05, regions=EVERY)
    deep = tmp_path / "deep" / "x" / "phys"
    deep.mkdir(parents=True)
    (tmp_path / "link").symlink_to(deep)
    ref_run = run_stub(tmp_path / "direct", stub, events=30, fields=[{"type": "map", "path": str(maps)}])
    for plan_via, exec_via in ((tmp_path / "link" / "a", deep / "a"), (deep / "b", tmp_path / "link" / "b")):
        particles.plan(plan_via, program=stub, **kw(events=30, fields=[{"type": "map", "path": str(maps)}]))
        r = driver_cli("execute", exec_via)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr[-2000:])
        res = particles.merge(exec_via)
        assert res.partial is False and res.n_events == 30
        assert results_json(exec_via) == results_json(tmp_path / "direct")
        for jf in (exec_via / "exchange").glob("job_*.json"):
            paths = [f["path"] for f in json.loads(jf.read_text())["fields"]]
            assert paths == [os.path.realpath(maps)], paths
        rec = json.loads((exec_via / "run.json").read_text())
        assert all(not os.path.isabs(f["path"]) for f in rec["fields"]), rec["fields"]
    assert ref_run.n_events == 30


def test_one_workdir_is_worked_on_by_one_driver_at_a_time(tmp_path, stub):
    """An exclusive lock on the workdir: while one `execute` runs, a second `execute`, a `plan` into it, an `extend`
    and a `merge` are refused with ``events.workdir_busy`` (the command line: exit 1 and the code), and change
    nothing; the first finishes with the result of an undisturbed run."""
    wd = tmp_path / "w"
    particles.plan(wd, program=stub, **kw(events=40, debug={"stub_sleep_s": 4.0}))
    first = subprocess.Popen([sys.executable, "-m", "axionmultiphysics.particles", "execute", str(wd)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    import time
    t0 = time.monotonic()
    while not list((wd / "out").glob(".stub_running_*")):
        assert time.monotonic() - t0 < 120 and first.poll() is None, "the first execute did not start its program"
        time.sleep(0.1)
    for call in (lambda: particles.execute(wd), lambda: particles.merge(wd), lambda: particles.extend(wd, events=80),
                 lambda: particles.plan(wd, program=stub, **kw(events=40))):
        with pytest.raises(Exception) as ei:
            call()
        assert code_of(ei) == "events.workdir_busy", (code_of(ei), str(ei.value))
    r = driver_cli("execute", wd)
    assert r.returncode == 1 and json.loads(r.stdout)["code"] == "events.workdir_busy"
    out, err = first.communicate(timeout=600)
    assert first.returncode == 0, err[-2000:]
    assert json.loads(out)["status"] == "complete"
    clean = run_stub(tmp_path / "clean", stub, events=40)
    assert results_json(wd) == results_json(tmp_path / "clean") and clean.n_events == 40


@pytest.mark.parametrize("args", [("execute", "--workers", "0"), ("execute", "--workers", "-3"),
                                  ("execute", "--workers", "x"), ("merge", "--workers", "0"),
                                  ("extend", "--events", "0"), ("extend", "--events", "abc"),
                                  ("extend", "--events", "45", "--workers", "0")])
def test_bad_numbers_on_the_command_line_are_usage_errors_and_change_nothing(tmp_path, stub, args):
    wd = tmp_path / "w"
    run_stub(wd, stub, events=30)
    snap = {p.relative_to(wd): p.read_bytes() for p in wd.rglob("*") if p.is_file()}
    r = driver_cli(args[0], wd, *args[1:])
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr[-1000:])
    assert "usage" in r.stderr.lower()
    after = {p.relative_to(wd): p.read_bytes() for p in wd.rglob("*") if p.is_file()}
    assert after == snap, "a refused command changed the workdir"


def test_extend_validates_before_it_writes(tmp_path, stub):
    from axionmultiphysics import AxionMultiPhysicsError
    wd = tmp_path / "w"
    run_stub(wd, stub, events=30)
    snap = {p.relative_to(wd): p.read_bytes() for p in wd.rglob("*") if p.is_file()}
    for bad in ({"events": 45, "workers": 0}, {"events": 45, "workers": -2}, {"events": 45, "chunk_events": 20},
                {"events": 20}):
        with pytest.raises((ValueError, AxionMultiPhysicsError)):
            particles.extend(wd, **bad)
        after = {p.relative_to(wd): p.read_bytes() for p in wd.rglob("*") if p.is_file()}
        assert after == snap, f"extend({bad}) changed the workdir before refusing"
    assert particles.extend(wd, events=45).n_events == 45


@pytest.mark.parametrize("seed", [-1, 2 ** 64, 2 ** 64 + 77, 2 ** 70])
def test_seeds_outside_64_bits_are_refused(tmp_path, stub, seed):
    """A seed is one of 2^64 values: -1 and 2^64 - 1, or 77 and 2^64 + 77, would otherwise be one run recorded as
    two (§10.2 rule 8)."""
    from axionmultiphysics import AxionMultiPhysicsError
    with pytest.raises((ValueError, AxionMultiPhysicsError)) as ei:
        particles.plan(tmp_path / "w", program=stub, **kw(seed=seed))
    if isinstance(ei.value, AxionMultiPhysicsError):
        assert code_of(ei) == "events.invalid"
    assert not (tmp_path / "w" / "run.json").exists()


@pytest.mark.parametrize("seed", [0, 2 ** 64 - 1])
def test_the_extreme_seeds_are_accepted_and_recorded_as_given(tmp_path, stub, seed):
    r = run_stub(tmp_path / "w", stub, events=10, seed=seed)
    assert r.record["seed"] == seed and r.record["chunks"][0]["seed"] == ref.chunk_seed(seed, 0)


def test_a_count_over_no_merged_event_reports_no_estimate(tmp_path, stub):
    """Every chunk failed: no event is merged. A count then has no estimate (NaN, no error, no upper limit), as a
    fraction and a mean have none; an upper limit of 3 'from 0 events' would come from no data."""
    out = dict(copy.deepcopy(DEP), c={"count": {"where": {"E": {"gt": 0.0}}}},
               f={"fraction": {"where": {"E": {"gt": 0.0}}}})
    with pytest.warns(particles.ParticleWarning, match=r"^events\.partial"):
        r = run_stub(tmp_path / "w", stub, events=30, outputs=out, debug={"fail_chunks": [0]})
    assert r.n_events == 0
    for name in ("c", "f", "Em"):
        m = r[name]
        assert math.isnan(float(m)) and m.stat_error is None and m.upper_limit is None, (name, str(m))
        assert not str(m).startswith("<"), str(m)


def test_merging_a_long_plan_is_linear_in_its_chunks(tmp_path, stub, monkeypatch):
    """1e5 planned chunks, the first 5e4 complete: merge lists the 5e4 missing ones in well under 20 s (the
    quadratic list took about 220 s at this size). The chunk reader and the estimator of the driver are replaced by
    stand-ins (complete one-event chunks; the estimator returns the missing list), so only the selection is timed."""
    import signal
    import time
    from axionmultiphysics.particles import _driver
    wd = tmp_path / "w"
    particles.plan(wd, program=stub, **kw(events=100000, chunk_events=1))

    class Chunk:
        def __init__(self, wd_, c, load=()):
            self.k = int(c["index"])
            self.state = "complete" if self.k < 50000 else "missing"
            self.message, self.sha256, self.manifest = "", "0" * 64, {"events": 1}

    monkeypatch.setattr(_driver, "_Chunk", Chunk)
    monkeypatch.setattr(_driver, "_estimate", lambda wd_, run, O, chunks, missing: (len(chunks), missing))

    def too_slow(*_):
        raise TimeoutError("merge took more than 20 s to select 1e5 chunks")
    old = signal.signal(signal.SIGALRM, too_slow)
    signal.alarm(20)
    try:
        t0 = time.monotonic()
        n, missing = particles.merge(wd)
        dt = time.monotonic() - t0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    print(f"merge of 1e5 planned chunks: {dt:.2f} s")
    assert n == 50000 and missing == list(range(50000, 100000))


class _FakeResults:
    """The part of a solved design's results that ``transport`` uses: ``export_field_map`` writes a map (here by
    hand, covering the slab document's regions) and is counted."""

    def __init__(self, bz=0.5):
        self.calls, self.bz = 0, bz

    def export_field_map(self, path, **kw_):
        self.calls += 1
        return write_hand_map(Path(path), (0.0, 0.3), (-0.5, 0.5), 0.05, regions=EVERY, bz=self.bz)


def test_the_api_route_validates_the_call_before_it_exports_the_field_map(tmp_path, stub):
    """A call refused by validation leaves nothing behind (no map exported, no workdir content), and the corrected
    call on the same workdir runs."""
    import axionmultiphysics as axmp
    doc = {"schema": "axionmultiphysics.design/1", "name": "s",
           "regions": [{"name": "winding", "material": "winding", "shape": {"rect": {"r": [0.5, 0.55], "z": [-0.5, 0.5]}}}],
           "windings": [{"name": "w", "region": "winding", "type": "azimuthal", "ampere_turns": 1e5}]}
    design, fake, wd = axmp.Design(doc), _FakeResults(), tmp_path / "w"
    with pytest.raises(Exception) as ei:
        particles.transport(design, fake, solids=slab_doc(), sources=[dict(GUN)], events=20, workdir=wd,
                            outputs={"E": {"deposit": {"in": "nowhere"}}}, program=stub)
    assert code_of(ei) == "particle.output_invalid"
    assert fake.calls == 0
    assert not wd.exists() or not any(wd.iterdir())
    r = particles.transport(design, fake, solids=slab_doc(), sources=[dict(GUN)], events=20, workdir=wd,
                            outputs=copy.deepcopy(DEP), program=stub)
    assert fake.calls == 1 and r.n_events == 20 and r.partial is False


# ============================================================================================ after the second review
def em_design():
    import axionmultiphysics as axmp
    return axmp.Design({"schema": "axionmultiphysics.design/1", "name": "s",
                        "regions": [{"name": "winding", "material": "winding",
                                     "shape": {"rect": {"r": [0.5, 0.55], "z": [-0.5, 0.5]}}}],
                        "windings": [{"name": "w", "region": "winding", "type": "azimuthal", "ampere_turns": 1e5}]})


def api_run(wd, bz, events, stub):
    return particles.transport(em_design(), _FakeResults(bz), solids=slab_doc(), sources=[dict(GUN)], events=events,
                               chunk_events=10, outputs=copy.deepcopy(DEP), workdir=wd, program=stub)


def test_a_copied_workdir_uses_its_own_field_map(tmp_path, stub):
    """A map inside the workdir (transport()'s) is named relative to the workdir in the job files, so a copy of the
    workdir is a run of its own: after the original's map changes, extending the copy uses the copy's map, bit for bit
    the field it started with."""
    import shutil
    api_run(tmp_path / "A", 0.5, 20, stub)
    shutil.copytree(tmp_path / "A", tmp_path / "B")
    write_hand_map(tmp_path / "A" / "fieldmap", (0.0, 0.3), (-0.5, 0.5), 0.05, regions=EVERY, bz=3.0)
    particles.extend(tmp_path / "B", events=40)
    api_run(tmp_path / "R", 0.5, 40, stub)
    for k in range(4):
        x, _ = chunk_arrays(tmp_path / "B", k)
        y, _ = chunk_arrays(tmp_path / "R", k)
        assert x["edep_eV"].tobytes() == y["edep_eV"].tobytes(), f"chunk {k} of the copy used another field"
    assert results_json(tmp_path / "B") == results_json(tmp_path / "R")
    for jf in (tmp_path / "B" / "exchange").glob("job_*.json"):
        for f in json.loads(jf.read_text())["fields"]:
            assert str(tmp_path / "A") not in f["path"] and not os.path.isabs(f["path"]), f["path"]


def test_a_field_map_changed_after_planning_is_refused_before_any_program_runs(tmp_path, stub):
    """run.json records the SHA-256 of each map's manifest; execute and extend compare it before starting a program
    and refuse a changed map (``events.field_map_changed``, naming it), with nothing written."""
    import hashlib
    m = write_hand_map(tmp_path / "maps" / "m", (0.0, 0.3), (-0.5, 0.5), 0.05, regions=EVERY)
    f = [{"type": "map", "path": str(m)}]
    run_stub(tmp_path / "w", stub, events=20, fields=f)
    rec = json.loads((tmp_path / "w" / "run.json").read_text())
    assert [x["manifest_sha256"] for x in rec["fields"]] == [hashlib.sha256((m / "manifest.json").read_bytes()).hexdigest()]
    particles.plan(tmp_path / "p", program=stub, **kw(events=20, fields=f))
    write_hand_map(m, (0.0, 0.3), (-0.5, 0.5), 0.05, regions=EVERY, bz=3.0)
    for wd, call in ((tmp_path / "w", lambda: particles.extend(tmp_path / "w", events=40)),
                     (tmp_path / "p", lambda: particles.execute(tmp_path / "p"))):
        snap = {p.relative_to(wd): p.read_bytes() for p in wd.rglob("*") if p.is_file() and p.name != ".lock"}
        with pytest.raises(Exception) as ei:
            call()
        assert code_of(ei) == "events.field_map_changed" and str(m) in str(ei.value)
        after = {p.relative_to(wd): p.read_bytes() for p in wd.rglob("*") if p.is_file() and p.name != ".lock"}
        assert after == snap, "a refused call changed the workdir"


@pytest.mark.parametrize("cmd", ["merge", "execute"])
def test_events_is_a_usage_error_outside_extend(tmp_path, stub, cmd):
    run_stub(tmp_path / "w", stub, events=20)
    r = driver_cli(cmd, tmp_path / "w", "--events", 50)
    assert r.returncode == 2 and "usage" in r.stderr.lower(), (r.returncode, r.stderr[-500:])


def test_fields_and_a_field_map_together_are_ambiguous(tmp_path, stub):
    fake = _FakeResults()
    m = write_hand_map(tmp_path / "m", (0.0, 0.3), (-0.5, 0.5), 0.05, regions=EVERY)
    with pytest.raises(Exception) as ei:
        particles.transport(em_design(), fake, solids=slab_doc(), sources=[dict(GUN)], events=20,
                            outputs=copy.deepcopy(DEP), workdir=tmp_path / "w", program=stub, field_map=m,
                            fields=[{"type": "uniform", "B_T": [0.0, 0.0, 1.0]}])
    assert code_of(ei) == "particle.field_ambiguous"
    assert fake.calls == 0
    assert not (tmp_path / "w").exists() or not any((tmp_path / "w").iterdir())


def test_execute_takes_no_allow_failed_chunks():
    """`execute` runs the jobs and does not merge, so the merge's option is not its own (``merge`` and ``extend``
    take it; the command line's ``execute`` merges and keeps the flag)."""
    import inspect
    assert "allow_failed_chunks" not in inspect.signature(particles.execute).parameters
    assert "allow_failed_chunks" in inspect.signature(particles.merge).parameters


# ============================================================================================ after the third review
def test_a_gun_outside_every_region_is_refused(tmp_path, stub):
    """The gun must start inside a region of the solids: beyond the world the program has no volume to start the
    primary in, and in the world's margin (or a void) the primary would leave the domain at once. Both are refused
    with ``particle.source_invalid`` before anything is written; the region's own boundary is inside."""
    doc = slab_doc()
    world = next(s for s in doc["solids"] if s["mother"] is None)
    r_slab, r_world = 0.3, world["params"]["rmax_m"]
    assert r_world > r_slab
    for name, at in (("beyond", [0.0, 0.0, -0.5 - world["params"]["z_m"]]),
                     ("margin", [0.5 * (r_slab + r_world), 0.0, 0.0])):
        with pytest.raises(Exception) as ei:
            particles.plan(tmp_path / name, program=stub, **kw(sources=[dict(GUN, at_m=at)]))
        assert code_of(ei) == "particle.source_invalid", (name, str(ei.value))
        assert "outside the world" in str(ei.value) if name == "beyond" else "outside every region" in str(ei.value)
        assert not (tmp_path / name).exists() or not any((tmp_path / name).iterdir())
    particles.plan(tmp_path / "edge", program=stub, **kw(sources=[dict(GUN, at_m=[r_slab, 0.0, -0.4])]))


# ============================================================================================ documentation
def test_the_guide_documents_the_driver_its_formats_and_every_code():
    g = (ROOT / "docs" / "guides" / "particles.md").read_text()
    for s in ("axionmultiphysics.particles", "resolve_materials", "particles.plan", "particles.merge",
              "particles.extend", "particles.transport", "Measured", "axionmultiphysics.transport-job/1",
              "axionmultiphysics.particles-run/1", "axionmultiphysics.particles-results/1", "--nist", "probe",
              "G4IntegrationDriver", "Clopper-Pearson", "batch means", "exit status 5", "chunk_events", "beta"):
        assert s in g, s
    for c in CODES:
        assert f"`{c}`" in g, c
    assert "Statistical error is not the total error" in g or "statistical errors only" in g
    llms = (ROOT / "llms.txt").read_text()
    assert "axionmultiphysics.particles" in llms
    # the notes since 0.1.0 ("Unreleased" and the release sections above 0.1.0) without their Highlights, so that the
    # entry itself must name the driver
    notes = (ROOT / "CHANGELOG.md").read_text().split("## Unreleased", 1)[1].split("\n## 0.1.0", 1)[0]
    notes = re.sub(r"### Highlights\n.*?(?=\n### )", "", notes, flags=re.S)
    assert "axionmultiphysics.particles" in notes and "transport" in notes, (
        "CHANGELOG.md (notes since 0.1.0, outside the Highlights) has no entry for the driver")


def test_the_validation_page_lists_each_row_with_its_numbers():
    p = ROOT / "docs" / "validation" / "particles.md"
    assert p.is_file()
    t = p.read_text()
    for row in ("F1", "F2", "F4", "F5", "P5", "S1", "S2", "S3", "S4", "X1"):
        assert row in t, row
    assert "seed" in t and "events" in t and "beta" in t.lower()
    assert "validation/particles.md" in (ROOT / "mkdocs.yml").read_text()
