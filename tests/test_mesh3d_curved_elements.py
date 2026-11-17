"""Order-2 meshes of small fillets must have no inverted curved elements.

The 3D mesher's elastic high-order pass leaves curved tetrahedra with det J <= 1e-3 max |det J| on fillets smaller
than the size floor; the solve then straightens them (geometry approximated) or, if that fails, makes the whole mesh
straight-sided. axmp-mesher therefore runs gmsh's Jacobian-targeted HighOrder optimizer whenever the elastic pass leaves
invalid elements (AXMP_MESHER_HO=auto, the default). In benchmarks/records/curved_elements_study.json
(benchmarks/curved_elements_study.py) the elastic pass alone leaves 1 such element on the probe shell used here and
819 on the DMRadio-Core 30-degree design; the default leaves none on either. Here:
benchmarks/designs/curved_probe_shell.json, a niobium toroidal box shell with 0.5 mm inner fillets at 20 mm / 8 mm
mesh size, meshed with `axmp mesh`, which reports the solve's own check as `curved_elements_invalid`. Set AXMP_SLOW=1.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("AXMP_SLOW"), reason="3D meshing (slow); set AXMP_SLOW=1")

from axionmultiphysics import _paths

DESIGN = Path(__file__).resolve().parents[1] / "benchmarks" / "designs" / "curved_probe_shell.json"


def _mesh(tmp_path, mode):
    axmp = _paths.axmp_path()
    if not axmp:
        pytest.skip("axmp executable not found")
    env = dict(os.environ)
    if mode is None:
        env.pop("AXMP_MESHER_HO", None)
    else:
        env["AXMP_MESHER_HO"] = mode
    wd = tmp_path / (mode or "default")
    p = subprocess.run([axmp, "mesh", str(DESIGN), "-o", str(wd), "-v", "0"], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    info = json.loads((wd / "mesh3d_info.json").read_text())
    return json.loads(p.stdout), info


def test_default_mesh_has_no_inverted_curved_elements(tmp_path):
    out, info = _mesh(tmp_path, None)
    assert info["high_order_optimize"] == "auto"
    assert out["curved_elements_invalid"] == 0, (out, info.get("high_order_counts"))


def test_elastic_pass_alone_leaves_inverted_elements(tmp_path):
    """Keeps the probe meaningful: with only the elastic pass this geometry does invert."""
    out, _ = _mesh(tmp_path, "elastic")
    assert out["curved_elements_invalid"] > 0, out
