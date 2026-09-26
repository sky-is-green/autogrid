"""Engine tests: the upstream code paths must keep behaving exactly as before
the fork (these run against ``autogrid.py`` unmodified)."""
import torch
from safetensors.torch import load_file

import autogrid


def _codes_on_grid(n=8, group=128):
    codes = torch.randint(-1, 2, (n, group)).float()
    scales = torch.rand(n, 1) + 0.1
    return (codes * scales).reshape(n, group)


def test_encode_error_tracks_the_ladder():
    torch.manual_seed(0)
    w = torch.randn(4, 128)
    prev = None
    for k in (4, 6, 8):
        err = autogrid._encode_rel(w, 128, k)
        assert err <= 4 * 3.0 ** -k
        if prev is not None:
            assert err < prev
        prev = err


def test_resolve_group_divides_width():
    g = autogrid.resolve_group(1000, 128)
    assert 1000 % g == 0 and g <= 128


def test_ternary_probe_finds_placed_weights():
    t = _codes_on_grid()
    on_grid, scales = autogrid._ternary_probe(t)
    assert on_grid and scales is not None and scales.numel() == 8
    off = t + 0.01
    assert autogrid._ternary_probe(off)[0] is False


def test_scan_tensor_classes():
    torch.manual_seed(1)
    bf = autogrid.scan_tensor("w", torch.randn(64, 128, dtype=torch.bfloat16))
    assert bf["cls"] == "FREE" and bf["rec_k"]
    f32 = autogrid.scan_tensor("w", torch.randn(64, 128, dtype=torch.float32))
    assert f32["cls"] == "STEER" and f32["rec_k"] is None
    tern = autogrid.scan_tensor("w", _codes_on_grid())
    assert tern["cls"] == "TERNARY" and tern["mode"] == "scales"
    ints = autogrid.scan_tensor("w", torch.zeros(4, 4, dtype=torch.int8))
    assert ints["cls"] == "SKIP"


def test_convert_and_revert_roundtrip(tiny_model):
    path, tensors = tiny_model
    shard = path / "model.safetensors"
    target = "model.layers.0.mlp.gate.weight"
    before = {n: t.clone() for n, t in tensors.items()}
    backup = path / "backup.safetensors"

    plan = [{"tensor": target, "mode": "values", "k": 4}]
    done, worst, skipped = autogrid.convert_model(path, plan, backup)
    assert done == 1 and not skipped and worst > 0

    changed = load_file(str(shard))[target]
    assert changed.shape == before[target].shape
    assert not torch.equal(changed, before[target])

    assert autogrid.revert_model(path, backup) == 1
    restored = load_file(str(shard))[target]
    assert torch.equal(restored, before[target])
