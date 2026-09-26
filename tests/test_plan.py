"""Correction-plan schema, export and backward-compatibility tests."""
import json

from autogrid_ext import moe, plan


def _report():
    rows = [
        {"tensor": "model.layers.0.mlp.experts.0.gate_proj.weight",
         "cls": "FREE", "rec_k": 6, "params": 1000,
         "errs": {"6": 0.002}},
        {"tensor": "model.layers.0.mlp.gate.weight",
         "cls": "FREE", "rec_k": 8, "params": 100,
         "errs": {"8": 0.0003}},
        {"tensor": "model.layers.1.mlp.experts.0.gate_proj.weight",
         "cls": "STEER", "rec_k": None, "params": 500,
         "errs": {"8": 0.01}},
    ]
    banks, moe_summary = moe.build_banks(rows)
    return {
        "model": "synthetic", "safety": 1.0, "max_k": 8, "tensors": rows,
        "summary": {"tensors": 3, "params_total": 1600,
                    "free_convert_params": 1100, "steer_params": 500},
        "banks": banks, "moe_summary": moe_summary,
    }


def test_build_plan_and_roundtrip(tmp_path):
    p = plan.build_plan(_report(), containers=("ternary", "q1_0_g128"))
    assert p["schema"] == plan.SCHEMA
    assert p["tool"]["upstream_commit"] == "0058823"
    assert p["containers"] == ["ternary", "q1_0_g128"]

    assert len(p["ops"]) == 2
    names = {o["tensor"] for o in p["ops"]}
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" not in names

    cs = p["correction_summary"]
    assert cs["layer_spec"] == "1" and cs["placement"] == "moe_out"
    assert p["corrections"][0]["router_rank"] == 64

    f = tmp_path / "plan.json"
    plan.save_plan(p, f)
    assert len(plan.load_plan(f)) == 2
    assert plan.is_extended_plan(f)

    assert "convert ops" in plan.plan_summary(p)


def test_load_plan_accepts_legacy_list(tmp_path):
    f = tmp_path / "legacy.json"
    f.write_text(json.dumps([{"tensor": "x", "mode": "values", "k": 4}]))
    assert plan.load_plan(f) == [{"tensor": "x", "mode": "values", "k": 4}]
    assert not plan.is_extended_plan(f)


def test_build_plan_without_banks():
    rep = _report()
    rep.pop("banks")
    rep.pop("moe_summary")
    p = plan.build_plan(rep)
    assert p["corrections"] == [] and p["correction_summary"] is None
