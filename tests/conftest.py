"""Shared fixtures: keep the fork importable from a source checkout and build a
tiny two-expert safetensors checkpoint for engine-level tests."""
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_model(tmp_path):
    """A tiny model dir with a 2-expert MoE layer, router and embedding."""
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight":
            torch.randn(32, 128, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.gate_proj.weight":
            torch.randn(32, 128, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight":
            torch.randn(128, 16, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.down_proj.weight":
            torch.randn(128, 16, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate.weight":
            torch.randn(4, 128, dtype=torch.bfloat16),
        "model.embed_tokens.weight":
            torch.randn(64, 128, dtype=torch.bfloat16),
    }
    d = tmp_path / "tiny-model"
    d.mkdir()
    save_file(tensors, str(d / "model.safetensors"))
    return d, tensors
