"""Rank invariance of the 2D Navier-Stokes direct path (MPI builds, 4 ranks).

The multi-rank comparison in cpp/tests/test_fluids_navier_stokes.cpp compares a solve on all ranks of MPI_COMM_WORLD with a one-rank solve of the same problem, and
returns without checking anything when it is run on one rank, which is how every C++ test is registered. This file
asserts the same property through the public Python boundary, launched on 4 ranks and on 1 rank, so that the property
is checked by the pytest suite (``AXMP_MPI=1``) as well.

Compared quantities: the lift residual ``||R(x_lift)||_*``, the per-step relative residuals of the Newton history and
the final relative residual. All three are global scalars, so they are available on every rank, whereas the solution
fields of a 2D solve can only be evaluated on one rank. The LU itself is rank invariant by construction (the serial
backends gather the matrix to rank 0, factor there and scatter), so what this test exercises is the assembly, the
elimination and the order of the MPI reductions inside the dual norm.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("AXMP_MPI"),
                                reason="multi-rank test: set AXMP_MPI=1")

SCRIPT = textwrap.dedent('''
    import json
    import numpy as np
    from axionmultiphysics import fluids

    B, R, T, L = 1, 2, 3, 4
    N = 12

    def rect(nx, ny):
        xs, ys = np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1)
        V = np.array([[x, y] for y in ys for x in xs])
        vid = lambda i, j: j * (nx + 1) + i
        tri, seg, att = [], [], []
        for j in range(ny):
            for i in range(nx):
                a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
                tri += [[a, b, c], [a, c, d]]
        for i in range(nx):
            seg += [[vid(i, 0), vid(i + 1, 0)], [vid(i + 1, ny), vid(i, ny)]]
            att += [B, T]
        for j in range(ny):
            seg += [[vid(nx, j), vid(nx, j + 1)], [vid(0, j + 1), vid(0, j)]]
            att += [R, L]
        return fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32),
                                       np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32))

    def lid(X, Y):
        top = (np.abs(Y - 1.0) < 1e-12) & (X > 1e-12) & (X < 1.0 - 1e-12)
        return np.stack([np.where(top, 1.0, 0.0), 0.0 * X], axis=-1)

    bcs = [{"attribute": a, "kind": "dirichlet", "value": lid} for a in (B, R, T, L)]
    s = fluids.solve_navier_stokes(rect(N, N), 1.0 / 300, 1.0, None, bcs, lu="eigen", vectorized=True,
                                   nullspace="multiplier")
    d = dict(s.diagnostics)
    out = {"ranks": int(d["ranks"]), "converged": bool(d["converged"]),
           "dofs": [int(d["dofs_velocity"]), int(d["dofs_pressure"])],
           "residual_lift": float(d["residual_lift"]), "relative_residual": float(d["relative_residual"]),
           "history": [float(h["residual"]) for h in s.history]}
    print("RESULT " + json.dumps(out), flush=True)
''')


def _run(tmp_path, nranks):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import _mpi_launch as mpi   # tests/_mpi_launch.py: srun, mpirun or mpiexec
    finally:
        sys.path.pop(0)
    if not mpi.available():
        pytest.skip(mpi.skip_reason())
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "ns_ranks.py"
    script.write_text(SCRIPT)
    argv = mpi.command(nranks) + [sys.executable, str(script)]
    # the children run in tmp_path: the package location must not depend on this process's working directory
    import axionmultiphysics
    pkg = str(Path(axionmultiphysics.__file__).resolve().parents[1])
    env = mpi.environment(argv, dict(os.environ, PYTHONPATH=os.pathsep.join(
        [pkg] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != pkg])))
    env.setdefault("OMPI_MCA_pml", "ob1")
    env.setdefault("OMPI_MCA_btl", "self,vader")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=900, cwd=tmp_path)
    lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("RESULT ")]
    assert lines, (nranks, r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    rec = json.loads(lines[0][len("RESULT "):])
    assert rec["ranks"] == nranks, (f"the launcher started {rec['ranks']} rank(s) instead of {nranks}: see "
                                    "tests/_mpi_launch.py for AXMP_MPI_LAUNCHER")
    return rec


def test_navier_stokes_history_and_residuals_are_rank_invariant(tmp_path):
    one = _run(tmp_path / "r1", 1)
    four = _run(tmp_path / "r4", 4)
    assert one["converged"] and four["converged"]
    assert one["dofs"] == four["dofs"], (one["dofs"], four["dofs"])
    # The dual norm is a sum over the local rows reduced with MPI_Allreduce, so a different partition sums the same
    # terms in a different order: the numbers agree to the reduction's round-off, not bitwise. The lift residual is a
    # single O(1) norm, so 1e-12 relative is ~4 decades above double round-off accumulated over 10^3 terms.
    assert one["residual_lift"] == pytest.approx(four["residual_lift"], rel=1e-12, abs=0.0)
    assert len(one["history"]) == len(four["history"]), (one["history"], four["history"])
    # Each step's relative residual is a ratio of two such norms, and the iteration amplifies the difference: the C++
    # C++ comparison uses 1e-8 relative on all but the last entry, which sits at round-off and is only required to be
    # inside the gate on both.
    for k, (a, b) in enumerate(zip(one["history"][:-1], four["history"][:-1])):
        assert abs(a - b) <= 1e-8 * abs(b) + 1e-14, (k, a, b, one["history"], four["history"])
    for rec in (one, four):
        assert rec["relative_residual"] <= 1e-12
