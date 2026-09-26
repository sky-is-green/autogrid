"""End-to-end CLI tests: fork flags take the extension path, and the upstream
invocation stays on the original code path (no new report keys)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "autogrid.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=ROOT, capture_output=True, text=True,
                          timeout=600)


def test_extended_cli_scan(tiny_model, tmp_path):
    model_dir, _ = tiny_model
    report = tmp_path / "report.json"
    plan = tmp_path / "plan.json"
    res = _run("--scan", str(model_dir), "--banks",
               "--containers", "ternary,q1_0_g128",
               "--json", str(report), "--plan-out", str(plan))
    assert res.returncode == 0, res.stderr
    assert "MoE banks" in res.stdout

    rep = json.loads(report.read_text())
    assert rep["containers"] == ["ternary", "q1_0_g128"]
    assert rep["moe_summary"]["expert_banks"] == 1
    assert any("container_errs" in r for r in rep["tensors"])

    pl = json.loads(plan.read_text())
    assert pl["schema"].startswith("sky-is-green/autogrid-plan/")
    assert pl["ops"]
    assert "plan -> " in res.stdout


def test_upstream_cli_path_unchanged(tiny_model, tmp_path):
    model_dir, _ = tiny_model
    report = tmp_path / "plain.json"
    res = _run("--scan", str(model_dir), "--json", str(report))
    assert res.returncode == 0, res.stderr
    rep = json.loads(report.read_text())
    # upstream report shape: no fork keys, and the same summary block
    assert "banks" not in rep and "containers" not in rep
    assert {"params_total", "tensors", "free_tensors"} <= set(rep["summary"])
