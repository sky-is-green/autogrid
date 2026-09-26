"""MoE-aware reporting for the sky-is-green AUTOGRID fork.

Upstream walks tensors one by one.  Routed-expert models have structure that a
flat list hides: an expert *bank* is one decision surface across E experts, and
the correction research (bonsai2-ternary-forensics ``docs/MOE-EXTENSION.md``)
showed that placement relative to the router — not raw capacity — decides
whether corrections recover a ternarised bank.

This module parses tensor names into roles, groups expert tensors into banks,
aggregates the scan rows at bank level, and attaches policy hints: what is
safe to convert, what needs the correction pipeline, and where the measured
best placement is (``moe_out``, rank ~512; router deltas rank-64).
"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict
from dataclasses import dataclass

#: ladder rungs used for free_at_k histograms when the report is narrow.
KS_ALL = (2, 3, 4, 5, 6, 8)

EXPERT_RE = re.compile(
    r"^(?P<base>.+?\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
FUSED_RE = re.compile(
    r"^(?P<base>.+?\.experts)\."
    r"(?P<proj>gate_up_proj|down_proj|gate_proj|up_proj)\.weight$")
ROUTER_RE = re.compile(r"^(?P<base>.+?)\.(?:router(?:\.proj)?|gate)\.weight$")
SHARED_RE = re.compile(r"^(?P<base>.+?)\.shared_experts?\.(?P<rest>.+)$")
DENSE_FFN_RE = re.compile(
    r"^(?P<base>.+?)\.mlp\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)(?:\.|$)")

#: measured placement policy from the MoE extension (see NOTICE / MOE-EXTENSION).
PLACEMENT_POLICY = {
    "placement": "moe_out",
    "suggested_rank": 512,
    "router_rank": 64,
    "reason": ("routing is decided from the residual stream entering the MoE "
               "block; a correction there steers the next layer's decisions, "
               "while per-expert branches act after the decision "
               "(MOE-EXTENSION 2.4)"),
}


@dataclass(frozen=True)
class TensorRole:
    name: str
    layer: int | None
    role: str                     # expert|expert_fused|router|shared_expert|
                                  # attn|dense_ffn|norm|embedding|lm_head|other
    expert: int | None = None
    projection: str | None = None
    base: str | None = None


def parse_tensor_name(name: str) -> TensorRole:
    """Classify one checkpoint tensor name.  Order matters: expert patterns
    first, then router (``.gate.weight`` only — never ``gate_proj``)."""
    layer_m = LAYER_RE.search(name)
    layer = int(layer_m.group("layer")) if layer_m else None

    m = EXPERT_RE.match(name)
    if m:
        return TensorRole(name, layer, "expert", int(m.group("expert")),
                          m.group("proj"), m.group("base"))
    m = FUSED_RE.match(name)
    if m:
        return TensorRole(name, layer, "expert_fused", None,
                          m.group("proj"), m.group("base"))
    m = ROUTER_RE.match(name)
    if m:
        return TensorRole(name, layer, "router", None, "router", m.group("base"))
    m = SHARED_RE.match(name)
    if m:
        return TensorRole(name, layer, "shared_expert", None, m.group("rest"),
                          m.group("base"))
    m = DENSE_FFN_RE.match(name)
    if m:
        return TensorRole(name, layer, "dense_ffn", None,
                          m.group("proj"), m.group("base"))
    if ".self_attn." in name:
        return TensorRole(name, layer, "attn")
    if name.endswith("embed_tokens.weight"):
        return TensorRole(name, layer, "embedding")
    if name.endswith("lm_head.weight"):
        return TensorRole(name, layer, "lm_head")
    if name.endswith(".weight") and ("norm" in name.rsplit(".", 2)[-2] or
                                     ".norm." in name):
        return TensorRole(name, layer, "norm")
    return TensorRole(name, layer, "other")


def _max_errs(rows: list[dict]) -> dict:
    ladder: dict[str, float] = {}
    for r in rows:
        for k, e in (r.get("errs") or {}).items():
            ladder[str(k)] = max(ladder.get(str(k), 0.0), float(e))
    return dict(sorted(ladder.items(), key=lambda kv: int(kv[0])))


def _bank(base: str, rows: list[dict],
          routers_by_layer: dict[int, list[str]]) -> dict:
    roles = [parse_tensor_name(r["tensor"]) for r in rows]
    role = "router" if all(p.role == "router" for p in roles) else "expert_bank"
    layer = next((p.layer for p in roles if p.layer is not None), None)

    recs = [r.get("rec_k") for r in rows]
    all_free = all(k for k in recs)
    convert_k = max(k for k in recs if k) if all_free else None

    ladder = _max_errs(rows)
    free_at_k = {str(rung): sum(1 for rec in recs if rec and rec <= rung)
                 for rung in KS_ALL}

    classes = Counter(r.get("cls", "?") for r in rows)
    params = sum(int(r.get("params", 0)) for r in rows)

    projections: dict[str, dict] = {}
    for p, r in zip(roles, rows):
        if p.projection is None:
            continue
        slot = projections.setdefault(
            p.projection, {"count": 0, "params": 0, "steer": 0})
        slot["count"] += 1
        slot["params"] += int(r.get("params", 0))
        slot["steer"] += 1 if r.get("cls") == "STEER" else 0

    if role == "router":
        verdict, action = "EXACT", "keep_exact"
        correction = {
            "action": "keep_exact",
            "note": ("routers stay exact by default; when the layer's expert "
                     "bank needs correction, train rank-64 router deltas "
                     "alongside the branches (MOE-EXTENSION 2.4)"),
        }
    elif all_free:
        verdict, action, correction = f"FREE@{convert_k}", "convert", None
    else:
        verdict, action = "STEER", "correct"
        steer_params = sum(int(r.get("params", 0)) for r in rows
                           if r.get("cls") == "STEER")
        correction = {
            **PLACEMENT_POLICY,
            "targets": [r["tensor"] for r in rows],
            "router_targets": routers_by_layer.get(layer, []),
            "steer_params": steer_params,
        }

    return {
        "bank": base,
        "layer": layer,
        "role": role,
        "tensors": [r["tensor"] for r in rows],
        "n_tensors": len(rows),
        "params": params,
        "classes": dict(classes),
        "convert_k": convert_k,
        "verdict": verdict,
        "action": action,
        "free_at_k": free_at_k,
        "ladder": ladder,
        "projections": projections,
        "correction": correction,
    }


def build_banks(rows: list[dict]) -> tuple[list[dict], dict | None]:
    """Group scan rows into MoE banks -> (banks, moe_summary).

    Works off rows alone (no tensors needed), so it is cheap to run on any
    upstream report and trivial to unit test.
    """
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    routers_by_layer: dict[int, list[str]] = {}
    for r in rows:
        p = parse_tensor_name(r["tensor"])
        if p.role in ("expert", "expert_fused", "router"):
            groups.setdefault(p.base or r["tensor"], []).append(r)
            if p.role == "router" and p.layer is not None:
                routers_by_layer.setdefault(p.layer, []).append(r["tensor"])
    if not groups:
        return [], None

    banks = [_bank(base, rs, routers_by_layer)
             for base, rs in groups.items()]
    banks.sort(key=lambda b: (b["layer"] if b["layer"] is not None else -1,
                              b["role"], b["bank"]))
    expert_banks = [b for b in banks if b["role"] == "expert_bank"]
    correction_layers = sorted({b["layer"] for b in expert_banks
                                if b["action"] == "correct"
                                and b["layer"] is not None})
    summary = {
        "layers": sorted({b["layer"] for b in expert_banks
                          if b["layer"] is not None}),
        "expert_banks": len(expert_banks),
        "expert_tensors": sum(b["n_tensors"] for b in expert_banks),
        "expert_params": sum(b["params"] for b in expert_banks),
        "routers": sum(1 for b in banks if b["role"] == "router"),
        "free_banks": sum(1 for b in expert_banks if b["action"] == "convert"),
        "steer_banks": sum(1 for b in expert_banks if b["action"] == "correct"),
        "correction_layers": correction_layers,
        "placement_policy": PLACEMENT_POLICY["placement"],
    }
    return banks, summary


def layer_spec(layers: list[int]) -> str:
    """[0,1,2,5] -> '0-2,5' (for the trainer's --layers style arguments)."""
    if not layers:
        return ""
    layers = sorted(set(layers))
    parts, start, prev = [], layers[0], layers[0]
    for n in layers[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)
