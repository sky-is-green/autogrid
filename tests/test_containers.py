"""Container simulation tests (absmean vs Lloyd, Q1_0_g128 accounting)."""
import torch

from autogrid_ext import containers as C


def test_bpw_math():
    assert C.container_bpw("q1_0_g128") == (2 * 128 + 16) / 128 == 2.125
    assert abs(C.container_bpw("ternary", 8) - 16.125) < 1e-9
    assert abs(C.container_bpw("ternary", 6) - 12.125) < 1e-9


def test_absmean_matches_the_pipeline_rule():
    torch.manual_seed(0)
    w = torch.randn(2, 128)
    g = w.reshape(2, 1, 128)
    a = g.abs().mean(-1, keepdim=True).clamp_min(1e-8)
    expect = torch.clamp(torch.round(g / a), -1, 1) * a
    got = C.ternary_absmean(w).reshape(2, 1, 128)
    assert torch.equal(got, expect)


def test_lloyd_never_much_worse_than_absmean():
    for seed in range(4):
        torch.manual_seed(seed)
        w = torch.randn(4, 256)
        wa, wl = C.ternary_absmean(w), C.ternary_lloyd(w)
        ra = float((wa.double() - w.double()).norm() / w.double().norm())
        rl = float((wl.double() - w.double()).norm() / w.double().norm())
        assert rl <= ra * 1.05 + 1e-12, (seed, ra, rl)


def test_q1_container_metrics():
    torch.manual_seed(2)
    t = torch.randn(4, 256, dtype=torch.bfloat16)
    m = C.evaluate("q1_0_g128", t, floor=2.0 ** -8, safety=1.0)
    assert m["bpw"] == 2.125
    assert 0 < m["rel_err"] < 1
    assert m["rel_err"] <= m["rel_err_absmean"] + 1e-12
    assert isinstance(m["free"], bool)

    assert C.evaluate("nope", t, 1.0)["error"].startswith("unknown")
    ints = torch.zeros(4, 4, dtype=torch.int8)
    assert "skipped" in C.evaluate("q1_0_g128", ints, 1.0)
