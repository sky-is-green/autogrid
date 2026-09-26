"""Correction-plan export for the sky-is-green AUTOGRID fork.

A plan is the bridge between a scan and the training/serving pipeline.  It
carries the convert-compatible ``ops`` list (the same shape ``--convert``
already accepted), the bank report, and the correction hints the MoE work
needs: which layers need branches, where the measured best placement is, and
which tensors/routers are in scope.

The file is self-describing and versioned; ``load_plan`` still accepts the
legacy bare-list plan so old files keep working.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import moe as _moe

SCHEMA = "sky-is-green/autogrid-plan/1"
TOOL = {
    "name": "autogrid",
    "fork": "https://github.com/sky-is-green/autogrid",
    "upstream": "https://github.com/CodeMasterCody3D/autogrid",
    "upstream_commit": "0058823",
}

DEFAULT_NOTES = [
    "ops are the convert-compatible safe set (FREE tensors and TERNARY scales "
    "with a recommended k); STEER tensors are deliberately excluded.",
    "corrections follow the measured MoE policy: residual-stream branches at "
    "the MoE block output (moe_out), rank ~512, plus rank-64 router deltas "
    "when the bank is not free (bonsai2-ternary-forensics MOE-EXTENSION 2.4).",
]


def ops_from_report(report: dict) -> list[dict]:
    """Rows with a recommended k -> the list ``convert_model`` consumes."""
    return [
        {"tensor": r["tensor"], "mode": r.get("mode", "values"), "k": r["rec_k"],
         "cls": r.get("cls")}
        for r in report.get("tensors", []) if r.get("rec_k")
    ]


def corrections_from_banks(banks: list[dict]) -> tuple[list[dict], dict | None]:
    """Per-layer correction entries + an aggregate summary."""
    entries = []
    for b in banks:
        if b.get("role") != "expert_bank" or b.get("action") != "correct":
            continue
        corr = b.get("correction") or {}
        entries.append({
            "placement": corr.get("placement", "moe_out"),
            "layer": b.get("layer"),
            "layer_spec": _moe.layer_spec([b["layer"]]) if b.get("layer") is not None else "",
            "targets": corr.get("targets", []),
            "router_targets": corr.get("router_targets", []),
            "suggested_rank": corr.get("suggested_rank", 512),
            "router_rank": corr.get("router_rank", 64),
            "steer_params": corr.get("steer_params", 0),
            "reason": corr.get("reason", ""),
        })
    if not entries:
        return [], None
    layers = sorted({e["layer"] for e in entries if e["layer"] is not None})
    summary = {
        "layers": layers,
        "layer_spec": _moe.layer_spec(layers),
        "placement": entries[0]["placement"],
        "suggested_rank": entries[0]["suggested_rank"],
        "router_rank": entries[0]["router_rank"],
        "params_in_scope": sum(e["steer_params"] for e in entries),
        "note": ("apply the runtime branch patch (moe-corr-runtime.patch in the "
                 "sibling ternary-serve work) before training moe_out targets"),
    }
    return entries, summary


def build_plan(report: dict, containers=("ternary",),
               notes: list[str] | None = None) -> dict:
    """Report (already bank-annotated) -> a versioned correction plan."""
    ops = ops_from_report(report)
    banks = report.get("banks", [])
    corrections, correction_summary = corrections_from_banks(banks)
    summary = report.get("summary", {})
    params_by_name = {r["tensor"]: r.get("params", 0)
                      for r in report.get("tensors", [])}
    value_ops = [o for o in ops if o.get("mode") != "scales"]
    scale_ops = [o for o in ops if o.get("mode") == "scales"]
    plan = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": TOOL,
        "model": report.get("model"),
        "safety": report.get("safety"),
        "max_k": report.get("max_k"),
        "containers": list(containers),
        "stats": {
            "tensors": summary.get("tensors"),
            "params_total": summary.get("params_total"),
            "free_convert_params": summary.get("free_convert_params"),
            "steer_params": summary.get("steer_params"),
            "ops": len(ops),
            "value_ops": len(value_ops),
            "scale_ops": len(scale_ops),
            "convert_params": sum(params_by_name.get(o["tensor"], 0)
                                  for o in value_ops),
            "correction_layers": len(corrections),
        },
        "ops": ops,
        "banks": banks,
        "moe_summary": report.get("moe_summary"),
        "corrections": corrections,
        "correction_summary": correction_summary,
        "notes": list(notes) if notes else list(DEFAULT_NOTES),
    }
    return plan


def save_plan(plan: dict, path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(plan, indent=1))
    return p


def load_plan(path) -> list[dict]:
    """Accept the legacy bare list or a versioned plan dict -> ops list."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("ops"), list):
        return raw["ops"]
    raise ValueError(f"{path}: not an AUTOGRID plan (no 'ops' list)")


def is_extended_plan(path) -> bool:
    """True when ``path`` parses as a versioned plan object (dict with ops)."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and isinstance(raw.get("ops"), list)


def plan_summary(plan: dict) -> str:
    s = plan.get("stats", {})
    n_banks = len(plan.get("banks") or [])
    lines = [
        f"plan {plan.get('schema')} — {s.get('value_ops', 0)} convert ops "
        f"({s.get('convert_params', 0):,} params) + {s.get('scale_ops', 0)} "
        f"scale ops, "
        f"{plan.get('moe_summary', {}).get('expert_banks', 0) if plan.get('moe_summary') else 0} expert banks"
        + (f", {len(plan.get('corrections') or [])} correction layers"
           if plan.get("corrections") else ""),
    ]
    cs = plan.get("correction_summary")
    if cs:
        lines.append(
            f"  corrections: placement {cs['placement']} · layers "
            f"{cs['layer_spec']} · rank ~{cs['suggested_rank']} "
            f"(routers rank-{cs['router_rank']}) · "
            f"{cs['params_in_scope']:,} steer params in scope")
    if n_banks:
        lines.append(f"  {n_banks} MoE banks recorded")
    return "\n".join(lines)
