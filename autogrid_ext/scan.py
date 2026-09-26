"""Extended scanning for the sky-is-green AUTOGRID fork.

Runs the upstream per-tensor classifier unchanged (so rows stay directly
comparable with older reports) and adds what the fork needs:

* bank-grouped MoE reporting (``banks`` / ``moe_summary`` keys), and
* optional deployed-container metrics per tensor (``container_errs``) when
  ``--containers`` asks for more than the upstream digit ladder.
"""
from __future__ import annotations

import time

from safetensors import safe_open

from . import containers as _containers
from . import moe as _moe


def scan_extended(model_path, safety: float = 1.0, max_k: int = 8,
                  containers=("ternary",), progress=lambda *a: None) -> dict:
    import autogrid as ag

    extra = tuple(c for c in containers if c != "ternary")
    for c in extra:
        if c not in ("q1_0_g128",):
            raise KeyError(
                f"unknown container {c!r} (known: ternary, q1_0_g128)")

    ks = tuple(k for k in ag.KS if k <= max_k) or (max_k,)
    wm = ag._shards(model_path)
    names = sorted(wm)
    rows, t0 = [], time.time()
    for i, n in enumerate(names):
        with safe_open(str(wm[n]), framework="pt") as h:
            t = h.get_tensor(n)
        row = ag.scan_tensor(n, t, safety, ks)
        if extra:
            row["container_errs"] = _container_row(ag, row, t, extra, safety)
        rows.append(row)
        progress(i + 1, len(names), n)

    report = {
        "model": str(model_path),
        "safety": safety,
        "max_k": max_k,
        "containers": list(containers),
        "elapsed_s": round(time.time() - t0, 1),
        "tensors": rows,
        "summary": ag._summarize(rows),
    }
    banks, moe_summary = _moe.build_banks(rows)
    if banks:
        report["banks"] = banks
        report["moe_summary"] = moe_summary
    return report


def _container_row(ag, row: dict, t, extra: tuple[str, ...],
                   safety: float) -> dict:
    """Per-tensor container metrics.  TERNARY tensors are measured on their
    scales (that is the remaining float data), like the upstream ladder does."""
    probe = t
    if row.get("cls") == "TERNARY":
        on_grid, scales = ag._ternary_probe(t)
        if not on_grid or scales is None:
            return {c: {"skipped": "ternary probe failed"}
                    for c in extra}
        probe = scales
    if not probe.is_floating_point() or probe.numel() == 0:
        return {c: {"skipped": "not suitable for container simulation"}
                for c in extra}
    return {c: _containers.evaluate(c, probe, row.get("floor"), safety)
            for c in extra}
