"""MoE parsing and bank aggregation tests."""
from autogrid_ext import moe


def test_parse_roles():
    p = moe.parse_tensor_name("model.layers.3.mlp.experts.7.gate_proj.weight")
    assert (p.role, p.layer, p.expert, p.projection) == \
        ("expert", 3, 7, "gate_proj")

    p = moe.parse_tensor_name("model.layers.3.mlp.experts.gate_up_proj.weight")
    assert p.role == "expert_fused" and p.projection == "gate_up_proj"

    for name in ("model.layers.3.mlp.gate.weight",
                 "model.layers.3.mlp.router.weight",
                 "model.layers.3.mlp.router.proj.weight"):
        assert moe.parse_tensor_name(name).role == "router", name

    p = moe.parse_tensor_name("model.layers.3.mlp.gate_proj.weight")
    assert p.role == "dense_ffn"

    assert moe.parse_tensor_name(
        "model.layers.3.self_attn.o_proj.weight").role == "attn"
    assert moe.parse_tensor_name(
        "model.layers.3.mlp.shared_expert.gate_proj.weight").role == \
        "shared_expert"
    assert moe.parse_tensor_name("lm_head.weight").role == "lm_head"


def _row(name, cls="FREE", rec_k=6, params=100):
    return {
        "tensor": name, "cls": cls, "rec_k": rec_k, "params": params,
        "errs": {"2": 0.2, "3": 0.06, "4": 0.02, "5": 0.007,
                 "6": 0.0025, "8": 0.0003},
    }


def test_banks_free_and_steer():
    rows = [
        _row(f"model.layers.0.mlp.experts.{e}.{p}.weight")
        for e in (0, 1) for p in ("gate_proj", "up_proj", "down_proj")
    ] + [
        _row("model.layers.0.mlp.gate.weight", rec_k=8),
        _row("model.layers.1.mlp.experts.0.gate_proj.weight",
             cls="STEER", rec_k=None),
        _row("model.layers.1.mlp.gate.weight", rec_k=8),
    ]
    banks, summary = moe.build_banks(rows)
    by = {b["bank"]: b for b in banks}

    b0 = by["model.layers.0.mlp.experts"]
    assert b0["verdict"] == "FREE@6" and b0["action"] == "convert"
    assert b0["n_tensors"] == 6 and b0["free_at_k"]["6"] == 6
    assert b0["correction"] is None

    b1 = by["model.layers.1.mlp.experts"]
    assert b1["verdict"] == "STEER" and b1["action"] == "correct"
    assert b1["correction"]["placement"] == "moe_out"
    assert b1["correction"]["suggested_rank"] == 512
    assert b1["correction"]["router_targets"] == \
        ["model.layers.1.mlp.gate.weight"]

    assert summary["free_banks"] == 1 and summary["steer_banks"] == 1
    assert summary["correction_layers"] == [1]


def test_build_banks_without_moe():
    banks, summary = moe.build_banks(
        [_row("model.layers.0.self_attn.q_proj.weight")])
    assert banks == [] and summary is None


def test_layer_spec():
    assert moe.layer_spec([0, 1, 2, 5, 7, 8, 9]) == "0-2,5,7-9"
    assert moe.layer_spec([]) == ""
