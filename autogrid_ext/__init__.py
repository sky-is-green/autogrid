"""sky-is-green AUTOGRID fork: extensions on top of upstream AUTOGRID.

Upstream (MIT, (c) Cody Dixon — see NOTICE) scans a safetensors checkpoint and
classifies every tensor on the balanced-ternary digit ladder.  This package
adds the fork's needs without touching those code paths:

* :mod:`autogrid_ext.moe` — tensor-role parsing and expert-bank reporting with
  correction policy hints (placement ``moe_out``; see MOE-EXTENSION 2.4).
* :mod:`autogrid_ext.containers` — deployed-container simulation
  (``q1_0_g128``: 2-bit codes + fp16 Lloyd-Max group scale = 2.125 bpw).
* :mod:`autogrid_ext.plan` — versioned correction-plan build/load/export that
  stays compatible with the upstream ``--convert --plan`` file shape.
* :mod:`autogrid_ext.scan` / :mod:`autogrid_ext.cli` — the extended scan and
  the ``--plan-out`` / ``--containers`` / ``--banks`` CLI flags.
"""
from . import containers, moe, plan, scan  # noqa: F401
from .containers import container_bpw, ternary_absmean, ternary_lloyd
from .moe import build_banks, layer_spec, parse_tensor_name
from .plan import build_plan, load_plan, save_plan
from .scan import scan_extended

__version__ = "0.1.0+sky.1"

__all__ = [
    "build_banks", "build_plan", "container_bpw", "containers", "layer_spec",
    "load_plan", "moe", "parse_tensor_name", "plan", "save_plan", "scan",
    "scan_extended", "ternary_absmean", "ternary_lloyd", "__version__",
]
